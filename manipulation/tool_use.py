"""Run the existing annotated tool-use capability as an explicit subset.

Only this bridge knows the released tool pipeline's supported action labels.
Object-motion geometry never branches on cabinet/drawer/object task names.
"""

import json
from pathlib import Path
import random

import numpy as np

from .observations import load_calibration
from .pipeline import write_json
from .anchors import bind_anchors, propagate_anchors
from .motion import _poses_array


def prepare_tracked_demo(demo_dir, tracking, destination):
    """Bind initial tool annotations once and expose the resulting demo to FUNCTO.

    This bridge is the only place that requires function/center/grasp points.
    The shared anchor layer accepts any number of points/directions and owners.
    The source annotations are in the released target frame; tracking uses the
    initial camera frame. Convert explicitly, remove reference motion once,
    then convert back to the original target frame for the existing optimizer.
    Later rows of the historical tool trajectory are never consumed here.
    """
    demo_dir, destination = Path(demo_dir), Path(destination)
    frame_indices = np.asarray(tracking["frame_indices"])
    owners = tracking["body_poses"]
    carrier = _poses_array(owners["carrier"])
    reference = _poses_array(owners["reference"])
    relative = _poses_array(tracking["poses"])
    if len(carrier) != len(reference) or len(carrier) != len(frame_indices):
        raise ValueError("Tool and reference must share the complete demonstration timeline")
    if not np.allclose(relative, reference[0] @ np.linalg.inv(reference) @ carrier, atol=1e-8):
        raise ValueError("Tool poses disagree with the declared initial-reference gauge")
    # Released event annotations index the original zero-based video and have
    # no mapping for missing frames. Generic tracking/anchors support sparse
    # IDs, but compressing a legacy timeline could silently change its event.
    if not len(frame_indices) or not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError("The annotated tool subset requires consecutive original frames starting at 0")
    calibration = json.loads((demo_dir / "cam_to_target_trans_output/cam_to_target_trans.json").read_text())
    target_from_camera = np.eye(4)
    rotation = np.asarray(calibration["rotation_matrix"], float)
    center, centroid = np.asarray(calibration["rotation_center"]), np.asarray(calibration["centroid"])
    target_from_camera[:3, :3] = rotation
    target_from_camera[:3, 3] = center - rotation @ center - centroid
    _poses_array(target_from_camera[None])
    initial = json.loads((demo_dir / "tool_pose_track_output/tool_track.json").read_text())
    names = ("function", "center", "grasp")
    points = np.asarray([initial[name][0] for name in names], float)
    if points.shape != (3, 3) or not np.isfinite(points).all():
        raise ValueError("Tool annotations need three finite initial semantic points")
    camera_points = (points-target_from_camera[:3, 3]) @ rotation
    bindings = bind_anchors([
        {"name": name, "owner": "carrier", "kind": "point", "frame_index": int(frame_indices[0]),
         "value": value} for name, value in zip(names, camera_points)
    ], owners, frame_indices)
    camera_anchors = propagate_anchors(bindings, owners, frame_indices)
    relative_anchors = propagate_anchors(bindings, {"carrier": relative}, frame_indices)
    generated = {anchor["name"]: anchor["values"] @ rotation.T + target_from_camera[:3, 3]
                 for anchor in relative_anchors["anchors"]}
    contact = json.loads((demo_dir / "key_point_track_output/keyframe_idx.json").read_text())["func"]
    if type(contact) is not int or not 0 <= contact < len(frame_indices):
        raise ValueError("The demonstrated function frame must be a valid nonnegative row index")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    # Read-only inputs are referenced, not copied into a second RGB-D dataset.
    # The new track alone is written locally, keeping the source demo intact.
    for name in ("rgb", "depth", "intrinsic.json", "config.json", "detection_output",
                 "key_point_track_output", "func_point_track_output", "grasp_det_output",
                 "cam_to_target_trans_output"):
        source = demo_dir / name
        if source.exists():
            (destination / name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    (destination / "tool_pose_track_output").mkdir()
    write_json(destination / "tool_pose_track_output/tool_track.json", generated)
    evidence = {"source": tracking.get("tracker", "TAPIP3D"),
                "frame_indices": frame_indices, "bindings": bindings,
                "camera0_anchors": camera_anchors, "reference_anchors": relative_anchors,
                "target_from_initial_camera": target_from_camera,
                "goal_row_index": contact, "goal_original_frame_id": int(frame_indices[contact]),
                "convention": "camera0 owner poses; fixed material anchors; reference motion removed once"}
    write_json(destination / "anchor_binding.json", evidence)
    return destination, evidence


def run_tool_use(demo_dir, scene_dir, task, instruction, output_dir, *, tracking, client=None):
    """Preserve the verified tool pipeline and expose its result with clear frames."""
    from local_models.gemini import Gemini
    demo_dir, scene_dir, output_dir = Path(demo_dir), Path(scene_dir), Path(output_dir)
    # BEGIN known mechanical constraints cannot disappear in the legacy solver.
    if task["motion"] not in {"auto", "free"}:
        raise ValueError("The tool-use optimizer cannot jointly enforce function events and "
                         "a constrained motion family; refusing to discard the motion constraint")
    # END explicit unsupported constraint combination.
    required = ["key_point_track_output/keyframe_idx.json", "tool_pose_track_output/tool_track.json",
                "func_point_track_output/func_point_2d_init.json", "grasp_det_output/grasp_points.json",
                "cam_to_target_trans_output/cam_to_target_trans.json", "detection_output/tool_mask.npy",
                "detection_output/tool.json"]
    missing = [name for name in required if not (demo_dir/name).is_file()]
    if missing:
        raise ValueError("The tool-use subset needs its demonstrated annotations: " + ", ".join(missing))
    # BEGIN independent goal validation: the demonstrated event is a row index,
    # not Python's permissive negative indexing into the resulting trajectory.
    contact = json.loads((demo_dir / "key_point_track_output/keyframe_idx.json").read_text())["func"]
    if type(contact) is not int or contact < 0:
        raise ValueError("The demonstrated function frame must be a nonnegative integer index")
    # END demonstrated event index validation.
    client = Gemini() if client is None else client
    action = client.chat_json(
        "Select the demonstrated action for the existing FUNCTO tool-use subset. "
        "Return exactly {\"action\": one of pour, scoop, pound, sweep, cut}. "
        "If none fits, return {\"action\": null}; never relabel a direct object action as tool use. "
        "Instruction: " + json.dumps(instruction, ensure_ascii=False)
        + "\nObserved stage description: " + json.dumps(task.get("goal_description", ""), ensure_ascii=False))
    if set(action) != {"action"} or action["action"] not in {"pour", "scoop", "pound", "sweep", "cut"}:
        raise ValueError("Instruction is outside the released annotated tool-use subset")
    from utils_IL.params import CAM_MATRIX
    for directory in (demo_dir, scene_dir):
        camera = load_calibration(directory)
        if not np.allclose(camera.K, CAM_MATRIX) or not np.isclose(camera.depth_scale, .001):
            raise ValueError("The preserved tool subset requires its original calibration; new object paths read calibrated RGB-D")
    # BEGIN the tool subset consumes newly observed motion with fixed semantics.
    demo_dir, anchor_evidence = prepare_tracked_demo(demo_dir, tracking, output_dir / "tracked_demo")
    # END tracked demonstration bridge.
    import matplotlib
    matplotlib.use("Agg")
    import torch
    import open3d as o3d
    from scipy.spatial.transform import Rotation
    from main import prepare_run, validate_outputs
    from cam_to_target_trans import cam_to_target_trans, demo_test_deteciton
    from func_point_transfer import func_point_transfer
    from grasp_point_transfer import grasp_point_transfer
    from tool_pose_transfer import tool_pose_transfer
    from tool_traj_transfer import compute_function_plane_frame, pc_viz, tool_trajectory_transfer
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    o3d.utility.random.seed(0)
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., 8*2**30/total))
    destination = output_dir / "tool_use"
    prepare_run(scene_dir, destination)
    tool, target = task["carrier"]["scene_label"], task["reference"]["scene_label"]
    label = action["action"]
    demo_test_deteciton(destination, tool, target)
    cam_to_target_trans(destination)
    func_point_transfer(demo_dir, destination, tool, target, label, 12, True)
    grasp_point_transfer(demo_dir, destination, tool, target, label, 12, True)
    tool_pose_transfer(demo_dir, destination, tool, target, label, False)
    # BEGIN independent goal reconstruction: use exactly the optimizer's input
    # contact frame, whose origin is function_func and whose axes come from the
    # function/center/grasp triple. Capture it before optimization, so neither
    # result poses nor a later artifact can redefine the target being checked.
    target_file = destination / "tool_pose_transfer_output/test_tool_track.json"
    target_points = json.loads(target_file.read_text())
    triple = np.asarray([target_points[key] for key in
                         ("function_func", "center_func", "grasp_func")], dtype=float)
    if triple.shape != (3, 3) or not np.isfinite(triple).all():
        raise ValueError("Legacy goal needs three finite function/center/grasp points")
    first, second = triple[1] - triple[0], triple[2] - triple[0]
    lengths = np.linalg.norm(first) * np.linalg.norm(second)
    if lengths == 0 or np.linalg.norm(np.cross(first, second)) <= np.finfo(float).eps * lengths:
        raise ValueError("Legacy goal function/center/grasp points must be noncollinear")
    goal_position, goal_angles = compute_function_plane_frame(
        [triple[0]], [triple[1]], [triple[2]])[0]
    goal_pose = np.eye(4)
    goal_pose[:3, 3] = goal_position
    goal_pose[:3, :3] = Rotation.from_euler("xyz", goal_angles).as_matrix()
    # END independent goal reconstruction.
    tool_trajectory_transfer(demo_dir, destination)
    validation = validate_outputs(destination)
    old = np.asarray(json.loads((destination / "tool_traj_transfer_output/test_tool_trajectory.json").read_text()))
    poses = np.repeat(np.eye(4)[None], len(old), axis=0)
    poses[:, :3, 3] = old[:, 0]
    poses[:, :3, :3] = Rotation.from_euler("xyz", old[:, 1]).as_matrix()
    if len(poses) != len(tracking["frame_indices"]):
        raise ValueError("The tool optimizer changed the observed frame timeline")
    # BEGIN compare the optimized event against its independent input target.
    if contact >= len(poses):
        raise ValueError("The demonstrated function frame lies outside the optimized trajectory")
    position_error = float(np.linalg.norm(poses[contact, :3, 3] - goal_pose[:3, 3]))
    rotation_error = float(Rotation.from_matrix(
        poses[contact, :3, :3] @ goal_pose[:3, :3].T).magnitude())
    # The old optimizer enforces equality at this event. These tolerances check
    # numerical equality only; they are not estimates of visual/metric accuracy.
    tolerance = 1e-6
    validation = {**validation, "goal_position_error_m": position_error,
                  "goal_rotation_error_rad": rotation_error,
                  "goal_position_tolerance_m": tolerance, "goal_rotation_tolerance_rad": tolerance,
                  "goal_source": "optimizer input: tool_pose_transfer_output/test_tool_track.json"}
    if position_error > tolerance or rotation_error > tolerance:
        raise ValueError("Optimized tool function frame misses its independent goal: "
                         f"position {position_error:.6g} m, rotation {rotation_error:.6g} rad")
    # END independent optimized-goal comparison.
    _, cloud = pc_viz(destination)
    points = np.asarray(cloud.points)
    if len(points) > 3000:
        points = points[np.linspace(0, len(points)-1, 3000).astype(int)]
    body = (points-poses[0, :3, 3]) @ poses[0, :3, :3]
    # BEGIN destination material anchors follow the optimized rigid tool.
    scene_bindings = bind_anchors([
        {"name": name, "owner": "carrier", "kind": "point",
         "frame_index": int(tracking["frame_indices"][0]), "value": target_points[name + "_init"]}
        for name in ("function", "center", "grasp")
    ], {"carrier": poses}, tracking["frame_indices"])
    scene_anchors = propagate_anchors(scene_bindings, {"carrier": poses}, tracking["frame_indices"])
    # END destination material anchors.
    result = {"status": "passed", "output_type": "object_trajectory", "task": task,
              "carrier_id": "tool", "reference_id": "target", "reference_frame": "legacy_target",
              "pose_convention": "reference_from_fixed_function_frame, metres, 4x4 SE(3)",
              "poses": poses, "body_points": body, "frame_indices": tracking["frame_indices"],
              "goal_pose": goal_pose, "goal_frame_index": contact,
              "model": {"kind": "free", "source": "TAPIP3D rigid motion + preserved tool optimizer"},
              "demonstration_anchors": anchor_evidence,
              "interaction_geometry": scene_anchors,
              "goal_checks": validation, "legacy_output_dir": str(destination.resolve())}
    # BEGIN reuse the existing offline tool visualizer's completed-run contract.
    write_json(destination / "run.json", {
        "status": "passed", "task": label, "seed": 0, "demo": str(demo_dir.resolve()),
        "test_input": str(scene_dir.resolve()), "sd_dino": True,
        "motion_source": result["model"]["source"], "validation": validation})
    # END saved metadata; no additional inference is needed for visualization.
    write_json(output_dir / "trajectory.json", result)
    return result
