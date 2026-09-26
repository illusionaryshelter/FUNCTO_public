"""Run FUNCTO steps 8--12 with local SAM3.1/SD-DINO and Gemini selection.

Demonstration annotations are inputs. Every prediction goes into a fresh run
folder, so the example's bundled historical outputs cannot satisfy a new run.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import time

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path):
    """Read task settings without loading models or credentials."""
    with open(config_path) as source:
        return yaml.safe_load(source)


def prepare_run(source: Path, destination: Path) -> None:
    """Copy only the initial RGB-D observation into an empty output directory."""
    required = [Path("rgb/00000.jpg"), Path("depth/00000.png")]
    for relative in required:
        if not (source / relative).is_file():
            raise FileNotFoundError(source / relative)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Use an empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for relative in required + [Path("intrinsic.json"), Path("config.json")]:
        path = source / relative
        if path.is_file():
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / relative)


def validate_outputs(output: Path) -> dict:
    """Reject incomplete or non-finite results rather than just checking files."""
    import numpy as np
    from PIL import Image

    def read(relative):
        return json.loads((output / relative).read_text())

    with Image.open(output / "rgb/00000.jpg") as image:
        width, height = image.size
    for role in ("tool", "target"):
        mask = np.load(output / "detection_output" / f"{role}_mask.npy", allow_pickle=False)
        box = np.asarray(read(f"detection_output/{role}.json"), dtype=float)
        if mask.shape != (height, width) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError(f"Invalid {role} segmentation mask")
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError(f"Invalid {role} bounding box")
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError(f"{role} bounding box lies outside the observation")
    points = {}
    for name, relative in (
        ("function", "func_point_transfer_output/test_init_frame_func_point_out.json"),
        ("grasp", "grasp_transfer_output/test_init_frame_grasp_point_out.json"),
    ):
        point = np.asarray(read(relative), dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"Invalid {name} image point")
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError(f"{name} point lies outside the observation")
        points[name] = point.tolist()
    frame = read("cam_to_target_trans_output/cam_to_target_trans.json")
    rotation = np.asarray(frame["rotation_matrix"], dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Invalid target frame rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Target frame rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("Target frame contains a reflection instead of a rotation")
    for key in ("centroid", "rotation_center", "target_dim"):
        vector = np.asarray(frame[key], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid target frame vector: {key}")
        if key == "target_dim" and (vector <= 0).any():
            raise ValueError("Target dimensions must be positive")
    track = read("tool_pose_transfer_output/test_tool_track.json")
    for key in ("function_init", "center_init", "grasp_init",
                "function_func", "center_func", "grasp_func"):
        point = np.asarray(track[key], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"Invalid 3D pose point: {key}")
    trajectory = np.asarray(read("tool_traj_transfer_output/test_tool_trajectory.json"), dtype=float)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (2, 3) or len(trajectory) < 2:
        raise ValueError(f"Invalid trajectory shape: {trajectory.shape}")
    if not np.isfinite(trajectory).all():
        raise ValueError("Trajectory contains non-finite values")
    cloud = output / "tool_traj_transfer_output/test_tool_traj_pc.ply"
    if not cloud.is_file() or cloud.stat().st_size == 0:
        raise ValueError("Missing final trajectory point cloud")
    return {"points_2d": points, "trajectory_shape": list(trajectory.shape)}


def test_processing(config, *, data_root=None, output_dir=None, seed=0):
    """Execute all released transfer stages; YAML controls optional refinements."""
    # BEGIN local demo execution: lazy imports keep --help usable without API
    # credentials or model initialization. Pipeline figures are saved headlessly.
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import open3d as o3d
    import torch
    from cam_to_target_trans import cam_to_target_trans, demo_test_deteciton
    from func_point_transfer import func_point_transfer
    from grasp_point_transfer import grasp_point_transfer
    from tool_pose_transfer import tool_pose_transfer
    from tool_traj_transfer import tool_trajectory_transfer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        # Matching has the GPU after each worker exits; bound its allocator too.
        memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 2**30 / memory))
    o3d.utility.random.seed(seed)
    task = config["task_label"]
    tool, target = config["test_tool_label"], config["test_target_label"]
    params = config["params"]
    if data_root is not None:
        root = Path(data_root).expanduser().resolve()
    else:
        root = Path(config["paths"]["base_data_path"]).expanduser()
        root = root.resolve() if root.is_absolute() else (PROJECT_ROOT / root).resolve()
    demo = root / "demo_data" / f"{task}_demo"
    source = root / "test_data" / f"{task}_test"
    output = (Path(output_dir).resolve() if output_dir else
              PROJECT_ROOT / "runs" / f"{task}-{datetime.now():%Y%m%d-%H%M%S-%f}")
    prepare_run(source, output)
    metadata = {"status": "running", "task": task, "seed": seed,
                "demo": str(demo), "test_input": str(source),
                "sd_dino": params["sd_dino_flag"], "stages": []}
    report = output / "run.json"
    stages = [
        ("segmentation", lambda: demo_test_deteciton(output, tool, target)),
        ("target_frame", lambda: cam_to_target_trans(output)),
        ("function_point", lambda: func_point_transfer(
            demo, output, tool, target, task, params["num_candidate_keypoints"], params["sd_dino_flag"])),
        ("grasp_point", lambda: grasp_point_transfer(
            demo, output, tool, target, task, params["num_candidate_keypoints"], params["sd_dino_flag"])),
        ("tool_pose", lambda: tool_pose_transfer(demo, output, tool, target, task, params["vp_flag"])),
        ("trajectory", lambda: tool_trajectory_transfer(demo, output)),
    ]
    report.write_text(json.dumps(metadata, indent=2))
    try:
        for name, stage in stages:
            print(f"FUNCTO stage: {name}", flush=True)
            started = time.monotonic()
            stage()
            metadata["stages"].append({"name": name, "seconds": time.monotonic() - started})
            report.write_text(json.dumps(metadata, indent=2))
        metadata["validation"] = validate_outputs(output)
        metadata["status"] = "passed"
    except BaseException as error:
        # Solver failures may use SystemExit; they must also mark this run failed.
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.write_text(json.dumps(metadata, indent=2))
    print(f"FUNCTO completed: {output}", flush=True)
    return output
    # END local demo execution.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="pour")
    parser.add_argument("--config", type=Path, help="Task YAML; defaults to the repository task config")
    parser.add_argument("--data-root", type=Path, help="Root containing demo_data and test_data")
    parser.add_argument("--output-dir", type=Path, help="An empty directory for this run's predictions")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config or PROJECT_ROOT / "utils_IL/config" / f"config_{args.task}.yaml")
    test_processing(config, data_root=args.data_root, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
