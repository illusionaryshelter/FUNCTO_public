"""Classify one recorded RGB-D view with the production three-state self-filter.

The bounded CPU probe compiles only the registered robot collision assembly.
It neither renders depth nor builds maps, plans motion, or advances physics.
"""

# BEGIN explicit three-state known-assembly observation probe.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recorded_observation import camera_config_from_calibration, classify_robot_observation
from manipulation.robot_self_filter import SELF, ENVIRONMENT, AMBIGUOUS_SHADOW
from robot_fixture import validate_fixture_collision_contract


def read_metadata(path):
    """Bound explicit metadata before deserialization, without path discovery."""
    path = Path(path)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("The mask probe accepts metadata of at most 16 MiB")
    return path.read_text()


def run(args):
    """Validate one current view and preserve labels and their evidence source."""
    if not args.observation_id.strip():
        raise ValueError("Provide an explicit observation identity")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "preparing",
        "scope": "three-state registered collision-model self-filter for one explicit RGB-D snapshot",
        "inputs": {name: str(getattr(args, name).resolve())
                   for name in ("fixture", "feedback", "rgb", "depth", "calibration")},
        "observation_id": args.observation_id,
        "padding_offset_m": args.self_filter_padding_offset_m,
        "padding_scale": args.self_filter_padding_scale,
        "self_filter_source": args.self_filter_source,
        "physics_steps": 0, "mapper_called": False, "planning_called": False,
        "recorded_object_state_read": False, "observed_scene_segmentation_read": False,
        "input_depth_modified": False, "unknown_space_validated": False,
        "planning_scene_validated": False,
        "rendering_called": False,
    }
    try:
        fixture = json.loads(read_metadata(args.fixture))
        feedback = json.loads(read_metadata(args.feedback))
        calibration = json.loads(read_metadata(args.calibration))
        robot = yaml.safe_load(read_metadata(fixture["robot_config_path"]))
        # Recompute source/fixture completeness before classifying any observation.
        # This contract requires visual_root == calibrated base_link and an
        # identity root offset, plus every separate fixed collision primitive.
        report["fixture_collision_contract"] = validate_fixture_collision_contract(fixture, robot)
        names = tuple(fixture["joint_names"])
        positions = np.asarray(feedback["start_q"], dtype=float)
        if (tuple(feedback["joint_names"]) != names or len(set(names)) != len(names)
                or positions.shape != (len(names),) or not np.isfinite(positions).all()
                or feedback["status"] != "validated" or feedback["reference_frame"] != "world"
                or Path(feedback["provenance"]["episode_path"]).resolve()
                != Path(fixture["episode_path"]).resolve()):
            raise ValueError("Masking requires complete validated named feedback from the fixture recording")
        # The existing calibration helper enforces image and model budgets.
        # Validate it before decoding RGB or reading the metric depth array.
        camera_config_from_calibration(calibration)
        shape = (calibration["height"], calibration["width"])
        with Image.open(args.rgb) as source_image:
            if source_image.size != (shape[1], shape[0]):
                raise ValueError("RGB dimensions differ from the explicit calibration")
            rgb = np.asarray(source_image.convert("RGB"))
        depth = np.load(args.depth, allow_pickle=False, mmap_mode="r")
        if depth.shape != shape or depth.dtype.kind != "f":
            raise ValueError("Depth must be a floating-point metre array matching the explicit calibration")
        measured = dict(zip(names, positions.tolist()))
        visual_pose = np.asarray(feedback["base_pose"], float) @ np.asarray(fixture["T_base_visual_root"], float)
        report.update(feedback_source=feedback["source"], feedback_entry_frame=feedback["entry_frame"],
                      feedback_sim_time_s=feedback["sim_time_s"], robot_visual_root=fixture["robot_visual_root"],
                      joint_positions=measured, T_world_robot_visual_root=visual_pose.tolist())
        filtered = classify_robot_observation(
            scene_xml=fixture["source_scene_xml"], robot_root=fixture["robot_visual_root"],
            joint_map=[(name, name, 1.) for name in names], joint_positions=measured,
            T_world_robot_root=visual_pose, observed_depth=depth, calibration=calibration,
            padding_scale=args.self_filter_padding_scale,
            padding_offset_m=args.self_filter_padding_offset_m, source=args.self_filter_source)
        labels = filtered["self_filter"]["labels"]
        report.update(
            fixed_installation_links=sorted({row["link"] for row in fixture["fixed_collision_geometries"]}),
            fixed_installation_collision_geometry_count=len(fixture["fixed_collision_geometries"]),
            collision_ownership_requirement="Fixed primitives remain static collision owners in any later scene",
            provenance=filtered["provenance"],
            counts={"SELF": int((labels == SELF).sum()),
                    "ENVIRONMENT": int((labels == ENVIRONMENT).sum()),
                    "AMBIGUOUS_SHADOW": int((labels == AMBIGUOUS_SHADOW).sum())})
        for name in ("labels", "free_ray_depth_m"):
            np.save(output / f"{name}.npy", filtered["self_filter"][name], allow_pickle=False)
        report["artifacts"] = {name: str(output / f"{name}.npy")
                               for name in ("labels", "free_ray_depth_m")}
        report["status"] = "completed"
        return report
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                      error=f"{type(error).__name__}: {error}")
        (output / "exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": report["status"], "report": str(output / "report.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--feedback", required=True, type=Path)
    parser.add_argument("--rgb", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--observation-id", required=True)
    parser.add_argument("--self-filter-padding-offset-m", required=True, type=float)
    parser.add_argument("--self-filter-padding-scale", required=True, type=float)
    parser.add_argument("--self-filter-source", required=True)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
# END explicit three-state known-assembly observation probe.
