"""Audit manifest-declared recorded inputs without constructing a robot plan.

The manifest owns every dataset/episode choice. This script only replays the
public CPU grounding boundary, checks the supplied observation contracts, and
exports small references for review. It neither resolves candidate ambiguity
nor substitutes recorded simulator state for missing live planner inputs.
"""

# BEGIN manifest-driven recorded-input audit and reference-only export.
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.program import ground_program


def json_value(value):
    """Keep numeric grounding evidence in JSON without stringifying arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def read_json(path):
    # These are metadata/reference files, not large trajectory recordings.
    # Refuse unexpectedly large inputs before allocating their decoded JSON.
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError(f"Input exceeds the 16 MiB per-file audit bound: {path}")
    return json.loads(path.read_text())


def hidden_state_guard(event, args):
    """Fail closed if grounding tries to open simulator or evaluation truth."""
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0]))
    if ("evaluation" in path.parts or path.suffix in {".h5", ".hdf5"}
            or path.name in {"model_catalog.json", "scene.xml", "seg_geom_id.npy",
                             "initial_scene_truth.npz"}):
        raise PermissionError(f"Input audit cannot read hidden current state: {path}")


def numeric_shape(value, shape):
    """Missing values are invalid; in particular, no zero state is inferred."""
    try:
        array = np.asarray(value, dtype=float)
        return array.shape == shape and bool(np.isfinite(array).all())
    except (TypeError, ValueError):
        return False


def reference_phases(program):
    """Describe source phase boundaries, without claiming planner phases."""
    demo = program["demo"]
    first, last = demo["frame_indices"][0], demo["frame_indices"][-1]
    if program["skill_kind"] == "grasp":
        boundary = demo["protected_start_frame"]
        return [{"id": 0, "name": "free approach reference", "start_frame": first,
                 "end_frame": boundary},
                {"id": 1, "name": "protected grasp reference", "start_frame": boundary,
                 "end_frame": last}]
    release = demo.get("release_frame")
    phases = [{"id": 0, "name": "held object reference", "start_frame": first,
               "end_frame": last if release is None else release}]
    if release is not None and release < last:
        phases.append({"id": 1, "name": "released hand reference", "start_frame": release,
                       "end_frame": last})
    return phases


def check_observations(segment, root):
    """Check each supplied RGB-D/calibration independently and keep mask scope."""
    output = []
    for view in segment["snapshot"]["views"]:
        calibration = read_json(root / view["calibration_path"])
        with Image.open(root / view["rgb_path"]) as rgb:
            width, height = rgb.size
        if width * height > 1920 * 1080:
            raise ValueError("Observation exceeds this CPU audit's two-megapixel bound")
        # Memory mapping avoids retaining a whole collection of depth images.
        depth = np.load(root / view["depth_path"], mmap_mode="r", allow_pickle=False)
        valid = np.isfinite(depth) & (depth > 0)
        checks = {
            "calibration_dimensions_match_rgb": (calibration["width"], calibration["height"]) == (width, height),
            "depth_dimensions_match_rgb": depth.shape == (height, width),
            "intrinsic_matrix_finite": numeric_shape(calibration.get("K"), (3, 3)),
            "world_camera_transform_finite": numeric_shape(calibration.get("T_world_camera_cv"), (4, 4)),
            "some_valid_depth": bool(valid.any()),
        }
        masks = []
        for mask_record in view["owner_masks"]:
            mask = np.load(root / mask_record["path"], mmap_mode="r", allow_pickle=False)
            aligned = mask.shape == (height, width) and mask.dtype == np.dtype(bool)
            masks.append({**mask_record, "aligned_boolean_image": aligned,
                          "visible_pixels": int(np.count_nonzero(mask))})
            checks["owner_mask_" + str(len(masks))] = aligned
        robot_mask = {"status": "missing"}
        if view["robot_mask_path"] is not None:
            try:
                mask = np.load(root / view["robot_mask_path"], mmap_mode="r", allow_pickle=False)
                aligned = mask.shape == (height, width) and mask.dtype == np.dtype(bool)
                robot_mask = {"status": "passed" if aligned else "failed",
                              "aligned_boolean_image": aligned,
                              "visible_pixels": int(np.count_nonzero(mask)),
                              "scope": "Image shape/type check only; semantic robot coverage is not certified."}
            except (OSError, ValueError) as error:
                robot_mask = {"status": "failed", "error": str(error)}
        output.append({"view_id": view["id"], "status": "passed" if all(checks.values()) else "failed",
                       "checks": checks, "image_dimensions": [width, height],
                       "valid_depth_pixels": int(valid.sum()), "owner_masks": masks,
                       "robot_mask_path": view["robot_mask_path"], "robot_mask_check": robot_mask})
    return output


def check_reference(program, scene, feedback, result):
    """Check source preservation and current EEF connection without a GT pose."""
    demo = program["demo"]
    frames = np.asarray(result["frame_indices"])
    poses = np.asarray(result["eef_poses"], float)
    rotation = poses[:, :3, :3]
    checks = {
        "source_frames_preserved": np.array_equal(frames, demo["frame_indices"]),
        "source_timestamps_preserved": np.array_equal(result["timestamps"], demo["timestamps"]),
        "source_gripper_events_preserved": result["gripper_events"] == demo["gripper_events"],
        "all_eef_poses_finite": numeric_shape(poses, (len(frames), 4, 4)),
        "rotation_matrices_orthonormal": bool(np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-7, rtol=0)),
        "rotations_have_positive_unit_determinant": bool(np.allclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0)),
        "eef_entrance_matches_feedback": bool(np.allclose(poses[0], feedback["eef_pose"], atol=1e-8, rtol=0)),
        "scene_eef_matches_feedback": bool(np.allclose(scene["eef_pose"], feedback["eef_pose"], atol=1e-8, rtol=0)),
        "tool_identity_preserved": all(result[k] == feedback[k] == demo[k] for k in ("tool_id", "tool_frame")),
        "reference_frame_preserved": result["reference_frame"] == feedback["reference_frame"],
    }
    if program["skill_kind"] == "grasp":
        start = int(np.flatnonzero(frames == demo["protected_start_frame"])[0])
        target = (np.asarray(scene["carrier_pose"]) @ np.asarray(scene["region_transform"])
                  @ np.asarray(demo["T_carrier_eef"]))
        checks["protected_suffix_preserved"] = bool(np.allclose(poses[start:], target[start:], atol=1e-12, rtol=0))
        checks["actual_holding_still_requires_verification"] = result["holding_status"] == "requires_verification"
    else:
        checks["release_boundary_preserved"] = result["release_frame"] == demo.get("release_frame")
        if demo.get("release_frame") is not None:
            checks["post_release_state_requires_observation"] = result["released_object_state"] == "requires_observation"
    return checks


def recompute_candidate(segment, candidate, program, feedback, root, output):
    """Replay every declared candidate verbatim; never inject an identity map."""
    row = {"candidate_id": candidate["id"], "view_id": candidate["view_id"],
           "scene_path": candidate["scene_path"], "status": "failed",
           "registration_status": candidate["registration_status"],
           "selected": segment["grounding"]["selected_candidate"] == candidate["id"],
           "reference_status": "reference_not_planned", "input_changes": []}
    try:
        scene = read_json(root / candidate["scene_path"])
        result = ground_program(program, scene)
        checks = check_reference(program, scene, feedback, result)
        row["contract_checks"] = checks
        if not all(checks.values()):
            raise ValueError("Grounded reference failed recorded-input contracts")
        frames = np.asarray(result["frame_indices"], np.int64)
        poses = np.asarray(result["eef_poses"], float)
        phases = reference_phases(program)
        phase_ids = np.zeros(len(frames), dtype=np.int64)
        if len(phases) == 2:
            # The shared boundary sample belongs to the new phase in the NPZ;
            # both human-readable inclusive phase intervals retain that join.
            phase_ids[frames >= phases[1]["start_frame"]] = 1
        artifact = output.parent / "references" / segment["id"] / (candidate["id"].replace("/", "__") + ".npz")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(result[key]) for key in
                  ("eef_poses", "frame_indices", "timestamps", "object_poses", "object_frame_indices")
                  if key in result}
        arrays.update(phase_ids=phase_ids, phase_names=np.asarray([x["name"] for x in phases]),
                      reference_frame=np.asarray(result["reference_frame"]),
                      status=np.asarray("reference_not_planned"))
        np.savez_compressed(artifact, **arrays)
        previous = read_json(root / candidate["result_path"])
        previous_eef = np.asarray(previous["eef_poses"], float)
        comparison = {"path": candidate["result_path"], "eef_shape_equal": previous_eef.shape == poses.shape}
        if previous_eef.shape == poses.shape:
            comparison["max_abs_eef_matrix_difference"] = float(np.abs(previous_eef - poses).max())
        row.update(status="passed", reference_artifact=str(artifact.resolve()),
                   reference_frame=result["reference_frame"], functional_phases=phases,
                   function_checks=result.get("function_checks", []),
                   historical_result_comparison=comparison,
                   metrics={"eef_nodes": len(frames), "object_nodes": len(result.get("object_poses", [])),
                            "source_duration_s": float(result["timestamps"][-1] - result["timestamps"][0]),
                            "eef_translation_path_m": float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())})
        if "holding_source" in result:
            # Preserve the original controlled-entrance premise verbatim;
            # passing geometry checks does not turn it into online Holding.
            row["holding_source"] = result["holding_source"]
    except Exception as error:
        row["error"] = {"type": type(error).__name__, "message": str(error)}
    return row


def planner_readiness(segment, feedback, observations, results, root):
    """Audit declared fields/files, without certifying planner-owned contracts."""
    inputs = segment["planner_inputs"]
    names = feedback.get("joint_names", [])
    valid_names = (isinstance(names, list) and bool(names)
                   and all(isinstance(name, str) and name for name in names)
                   and len(set(names)) == len(names))

    def declared_file(path):
        return isinstance(path, str) and bool(path) and (root / path).is_file()

    checks = {
        "named_joint_positions": valid_names and numeric_shape(feedback.get("qpos"), (len(names),)),
        "measured_joint_velocities": valid_names and numeric_shape(feedback.get("qvel"), (len(names),)),
        "robot_model_file_declared": declared_file(inputs["robot_model_path"]),
        "robot_base_pose_finite_4x4": numeric_shape(inputs["base_pose"], (4, 4)),
        "link_to_tcp_transform_finite_4x4": numeric_shape(inputs["tool_transform"], (4, 4)),
        "robot_mask_files_aligned": all(view["robot_mask_check"]["status"] == "passed" for view in observations),
        "collision_owner_contract_file_declared": declared_file(inputs["complete_collision_owner_contract_path"]),
        "rgbd_observation_contracts": all(view["status"] == "passed" for view in observations),
        "grounding_recompute": bool(results) and all(row["status"] == "passed" for row in results),
    }
    explanations = {
        "named_joint_positions": "Frozen feedback has no finite named joint positions; EEF pose alone does not determine a robot joint state.",
        "measured_joint_velocities": "Synchronized qvel/dq is absent; stationary zero velocity is not assumed.",
        "robot_model_file_declared": "No existing robot-model file is declared. Recorded joint mapping, gripper coupling and sphere layout still require planner validation even when a file is present.",
        "robot_base_pose_finite_4x4": "Finite T_reference_robot_base is not supplied; calibration evidence must not be replaced by identity.",
        "link_to_tcp_transform_finite_4x4": "Finite T_link_tool/TCP calibration is not supplied; tool names do not define this transform.",
        "robot_mask_files_aligned": "Current RGB-D views lack aligned boolean robot-mask files; empty masks cannot be invented.",
        "collision_owner_contract_file_declared": "No existing current collision-owner contract file is declared. Source geometry is listed separately; file presence alone would not certify a conservative sphere cover or completeness.",
        "rgbd_observation_contracts": "At least one declared RGB-D observation failed validation.",
        "grounding_recompute": "At least one explicit grounding candidate failed the public consumer or contract checks.",
    }
    blockers = [{"code": key, "detail": explanations[key]} for key, ok in checks.items() if not ok]
    if any(check.get("containment", {}).get("status") == "not_verified"
           for row in results for check in row.get("function_checks", [])):
        blockers.append({"code": "free_interior_geometry", "detail": "The inside relation explicitly reports free interior geometry not supplied; pose grounding alone does not verify containment."})
    return {"status": "blocked" if blockers else "declared_inputs_present_not_validated_by_planner", "checks": checks,
            "checks_scope": "Field presence, finite numeric dimensions and existing image/file paths only. Calibration provenance, robot/model bindings, mask semantics and collision completeness remain unvalidated by the planner.",
            "blockers": blockers, "feedback_fields": sorted(feedback),
            "collision_geometry_evidence": inputs["collision_geometry_evidence"],
            "candidate_selection_status": segment["grounding"]["status"],
            "selected_candidate": segment["grounding"]["selected_candidate"],
            "selection_note": "Numerical grounding does not resolve alternative bindings. All original candidates remain available for review.",
            "robot_plan_attempted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run this CPU audit with CUDA_VISIBLE_DEVICES='' explicitly")
    sys.addaudithook(hidden_state_guard)
    started = time.monotonic()
    manifest = read_json(args.manifest)
    root = Path(manifest["workspace_root"]).resolve()
    output = args.output.resolve()
    if manifest["schema_version"] != 1 or manifest["snapshots_are_independent"] is not True:
        raise ValueError("Expected an explicit independent-snapshot input manifest")
    segments = []
    for segment in manifest["segments"]:
        program = read_json(root / segment["program_path"])
        actual_range = [program["demo"]["frame_indices"][0], program["demo"]["frame_indices"][-1]]
        if actual_range != segment["source_frame_range"]:
            raise ValueError(f"Manifest source range disagrees with program: {segment['id']}")
        feedback = read_json(root / segment["snapshot"]["feedback_path"])
        observations = check_observations(segment, root)
        results = [recompute_candidate(segment, item, program, feedback, root, output)
                   for item in segment["grounding"]["candidates"]]
        runs = []
        for path in segment["grounding"]["run_paths"]:
            record = read_json(root / path)
            runs.append({"path": path, **{key: record[key] for key in
                         ("view", "status", "stage", "selection_status", "reason", "candidate_count",
                          "selected_candidate", "selected_candidate_id", "error") if key in record}})
        segments.append({"id": segment["id"], "source_frame_range": actual_range,
                         "snapshot_entry_frame": segment["snapshot"]["entry_frame"],
                         "source_library_segment": program["demo"].get("segment_id"),
                         "observations": observations, "historical_runs": runs,
                         "functional_phases": reference_phases(program),
                         "events": {key: program["demo"].get(key, []) for key in
                                    ("gripper_events", "functional_events", "physical_events")},
                         "grounding_recompute": results,
                         "planner_readiness": planner_readiness(segment, feedback, observations, results, root)})
    candidates = [candidate for segment in segments for candidate in segment["grounding_recompute"]]
    blocked_count = sum(row["planner_readiness"]["status"] == "blocked" for row in segments)
    report = {"schema_version": 1, "schema": "functo.recorded_task_input_audit.1",
              "manifest_path": str(args.manifest.resolve()),
              "status": "blocked_before_robot_planning" if blocked_count else "declared_inputs_present_not_validated_by_planner",
              "current_hidden_state_read": False,
              "gpu_used": False, "snapshots_are_independent": True,
              "continuous_execution_tested": False, "reference_status": "reference_not_planned",
              "candidate_count": len(candidates),
              "grounding_passed": sum(row["status"] == "passed" for row in candidates),
              "grounding_failed": sum(row["status"] == "failed" for row in candidates),
              "planner_blocked_segments": blocked_count,
              "resource_use": {"elapsed_seconds": time.monotonic() - started,
                               "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                               "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                               "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                               "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS")},
              "segments": segments}
    if "robot_binding_audit_path" in manifest:
        # Keep the independent CPU robot audit address explicit in the fixture.
        # This report does not manufacture a planner binding from that evidence.
        report["robot_binding_audit_path"] = manifest["robot_binding_audit_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_value) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "candidate_count", "grounding_passed", "grounding_failed", "planner_blocked_segments", "resource_use")}))


if __name__ == "__main__":
    main()
# END manifest-driven recorded-input audit and reference-only export.
