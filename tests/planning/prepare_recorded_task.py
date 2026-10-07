"""Prepare named robot entrances and current planning problems from a manifest.

Recorded HDF5 reads are restricted to explicitly named scalar robot joint
columns and their common timestamp. Robot-only FK validates each entrance
against the frozen RGB-D observation's EEF feedback. Every supplied grounding
candidate reaches the current public problem builder, independently of old
candidate-selection labels. This preparation does not certify a robot plan.
"""

# BEGIN manifest-driven named robot feedback and current problem preparation.
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.motion import _pose_array
from manipulation.planning_problem import build_planning_problem


def json_value(value):
    """Preserve numerical evidence as JSON values, never opaque strings."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def read_json(path):
    """Bound metadata reads before decoding them in this small CPU runner."""
    path = Path(path)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError(f"Input exceeds the 16 MiB metadata bound: {path}")
    return json.loads(path.read_text())


# BEGIN bind current goals while retaining ordinary reference provenance.
def prepare_candidate(program, scene, feedback, skill_id):
    """Return the public PlanningProblem and explicit input provenance.

    The caller owns the active-joint layout. Its supplied positions and
    measured velocities pass through unchanged; this helper neither projects
    joints nor infers finger symmetry. The source program and scene remain
    untouched, including all functional events and protected source samples.
    """
    prepared = deepcopy(scene)
    scene_eef = _pose_array(prepared["eef_pose"], "current scene EEF")
    measured_eef = _pose_array(feedback["eef_pose"], "current robot EEF feedback")
    if not np.allclose(scene_eef, measured_eef, atol=1e-8, rtol=0):
        raise ValueError("Candidate entrance EEF differs from synchronized robot feedback")
    for key in ("reference_frame", "tool_id", "tool_frame"):
        if key in feedback and prepared.get(key) != feedback[key]:
            raise ValueError(f"Candidate and robot feedback disagree on {key}")

    # Some frozen callers recorded the same evidenced carrier map per event.
    # Summarize that existing declaration explicitly at this test boundary;
    # an absent, inconsistent, or unsupported map never becomes an identity.
    changes = []
    bindings = prepared.get("functional_bindings", {})
    if "carrier_mapping" not in prepared and bindings and all(
            "carrier_mapping" in binding and bool(binding.get("source"))
            for binding in bindings.values()):
        maps = [_pose_array(binding["carrier_mapping"], "declared event carrier map")
                for binding in bindings.values()]
        if not all(np.array_equal(value, maps[0]) for value in maps[1:]):
            raise ValueError("Functional events disagree on their explicitly declared carrier map")
        source = {"kind": "identical_explicit_functional_binding_maps",
                  "event_sources": {key: deepcopy(value["source"])
                                    for key, value in bindings.items()}}
        prepared["carrier_mapping"] = maps[0].copy()
        prepared["carrier_mapping_source"] = source
        changes.append({"fields": ["carrier_mapping", "carrier_mapping_source"],
                        "carrier_mapping": maps[0].tolist(), "source": source})

    problem = build_planning_problem(
        [{"skill_id": skill_id, "program": program, "scene": prepared}],
        joint_names=feedback["joint_names"], start_q=feedback["start_q"],
        start_dq=feedback["start_dq"], tool_link=feedback["tool_link"],
        reference_frame=feedback["reference_frame"], base_pose=feedback["base_pose"],
        tool_transform=feedback["tool_transform"])
    provenance = {
        "builder": "manipulation.planning_problem.build_planning_problem",
        "skill_id": skill_id, "scene_input_changes": changes,
        "entrance_eef_max_abs_matrix_error": float(np.abs(scene_eef - measured_eef).max()),
        "joint_layout": list(problem.joint_names),
        "joint_positions_preserved": np.array_equal(problem.start_q, feedback["start_q"]),
        "measured_joint_velocities_preserved": np.array_equal(problem.start_dq, feedback["start_dq"]),
        "source_frame_range": [program["demo"]["frame_indices"][0],
                               program["demo"]["frame_indices"][-1]],
        "source_timestamps": deepcopy(program["demo"]["timestamps"]),
        "couplings_inferred": False,
    }
    return problem, provenance
# END no source approach point is a mandatory waypoint.


def _pose_error(actual, expected):
    """Report independent metre/radian errors for two rigid poses."""
    relative = np.linalg.inv(expected) @ actual
    return {"translation_m": float(np.linalg.norm(relative[:3, 3])),
            "rotation_rad": float(Rotation.from_matrix(relative[:3, :3]).magnitude()),
            "maximum_abs_matrix_error": float(np.abs(actual - expected).max())}


def _declared_pose(value, name):
    """Decode the fixture's explicit position/unit-wxyz calibration contract."""
    position = np.asarray(value["position"], dtype=float)
    quaternion = np.asarray(value["quaternion_wxyz"], dtype=float)
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-10, rtol=0)):
        raise ValueError(f"{name} requires finite XYZ and an explicit unit wxyz quaternion")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = position
    return _pose_array(transform, name)


def _mujoco_pose(position, rotation):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation).reshape(3, 3)
    transform[:3, 3] = position
    return transform


def _finger_sphere_links(model, fingers, collision_spheres):
    """Include every actual sphere link in each native finger's full subtree.

    A pad can live on a fixed child body. Reading only the joint's immediate
    body would leave that pad outside the binary gripper's swept geometry.
    All ownership here comes from the native robot tree and planner sphere
    bindings, not a finger/pad naming convention.
    """
    roots = {name: model.joint(name).bodyid[0] for name in fingers}
    result = {name: [] for name in fingers}
    for link, spheres in collision_spheres.items():
        body_id = model.body(link).id
        if not spheres:
            continue
        for joint_name, root_id in roots.items():
            ancestor = body_id
            while ancestor != 0 and ancestor != root_id:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root_id:
                result[joint_name].append(link)
    values = [link for links in result.values() for link in links]
    if any(not links for links in result.values()) or len(set(values)) != len(values):
        raise ValueError("Native finger subtrees need disjoint nonempty sets of actual collision-sphere links")
    return result


def _extract_named_feedback(manifest, fixture, fixture_path):
    """Read scalar robot q/dq columns only, then check both robot-only FK paths.

    The robot fixture owns the source address mapping and the native model.
    Neither recorded object coordinates, recorded body/site poses, contacts,
    nor the simulator's full scene are read. No physics step is performed.
    """
    import h5py
    import mujoco
    from yourdfpy import URDF
    import yaml

    root = Path(manifest["workspace_root"])
    mjcf_path, urdf_path, episode_path = (root / fixture[key] for key in
                                         ("robot_mjcf_path", "urdf_path", "episode_path"))
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    urdf = URDF.load(urdf_path, build_scene_graph=True, load_meshes=False,
                     build_collision_scene_graph=False, load_collision_meshes=False)
    robot_config = yaml.safe_load((root / fixture["robot_config_path"]).read_text())
    binding = fixture["joint_binding"]
    names = [item["recorded_name"] for item in binding]
    arms, fingers = fixture["arm_joint_names"], fixture["finger_joint_names"]
    if (not names or len(set(names)) != len(names)
            or set(names) != {model.joint(index).name for index in range(model.njnt)}
            or model.nq != len(names) or model.nv != len(names)
            or set(arms).intersection(fingers) or len(arms) + len(fingers) != len(names)
            or set(arms + fingers) != set(names)):
        raise ValueError("Fixture must declare every scalar robot joint exactly once as arm or finger")
    if set(urdf.actuated_joint_names) != set(names):
        raise ValueError("URDF and native robot-only MJCF must expose identical independent joint names")
    if not set(fixture["planning_joint_names"]).issubset(arms):
        raise ValueError("This fixture's explicit active layout must use its declared arm joints")
    finger_links = _finger_sphere_links(
        model, fingers, robot_config["robot_cfg"]["kinematics"]["collision_spheres"])
    for item in binding:
        if any(type(item[key]) is not int or item[key] < 0
               for key in ("qpos_column", "qvel_column")):
            raise ValueError("Recorded robot columns must be explicit nonnegative integer addresses")
        joint = model.joint(item["recorded_name"])
        if joint.type[0] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError("Only scalar native arm/finger joints are admitted")
    if any(len({item[key] for item in binding}) != len(binding)
           for key in ("qpos_column", "qvel_column")):
        raise ValueError("Each recorded scalar column must belong to exactly one named robot joint")
    q_addresses = [int(model.joint(name).qposadr[0]) for name in names]
    dq_addresses = [int(model.joint(name).dofadr[0]) for name in names]
    base = _declared_pose(fixture["base_pose"], "fixture world-from-base")
    tool = _declared_pose(fixture["tool_transform"], "fixture link-from-TCP")
    base_id, site_id = model.body(fixture["base_link"]).id, model.site(fixture["tcp_site"]).id
    result = {}
    with h5py.File(episode_path, "r") as stream:
        q_stream, dq_stream, times = (stream[key] for key in
                                      ("states/qpos", "states/qvel", "states/sim_time"))
        if (q_stream.ndim != 2 or dq_stream.ndim != 2 or times.ndim != 1
                or len(q_stream) != len(dq_stream) or len(q_stream) != len(times)):
            raise ValueError("Recorded robot positions, velocities and timestamps must share their frame count")
        if (max(item["qpos_column"] for item in binding) >= q_stream.shape[1]
                or max(item["qvel_column"] for item in binding) >= dq_stream.shape[1]):
            raise ValueError("A declared robot address lies outside its recorded state stream")
        for segment in manifest["segments"]:
            frame = segment["snapshot"]["entry_frame"]
            if type(frame) is not int or not 0 <= frame < len(times):
                raise ValueError("Every manifest entrance must identify one existing recorded frame")
            frozen_path = root / segment["snapshot"]["feedback_path"]
            frozen = read_json(frozen_path)
            q = np.array([q_stream[frame, item["qpos_column"]] for item in binding])
            dq = np.array([dq_stream[frame, item["qvel_column"]] for item in binding])
            timestamp = float(times[frame])
            if not np.isfinite(q).all() or not np.isfinite(dq).all() or not np.isfinite(timestamp):
                raise ValueError("Recorded named robot q/dq/time must be finite without imputation")
            if frozen["reference_frame"] != segment["snapshot"]["reference_frame"]:
                raise ValueError("Frozen robot feedback and snapshot must declare the same reference frame")
            if frozen["tool_frame"] != fixture["tcp_site"]:
                raise ValueError("The native fixture TCP must equal the frozen feedback's explicit tool frame")
            feedback_pose = _pose_array(frozen["eef_pose"], "frozen entrance EEF")

            # Assign only this robot-only model's named scalar coordinates.
            # Kinematics uses q; dq remains the actual same-frame observation,
            # never zeroed to make a start-at-rest planner input.
            data.qpos[q_addresses], data.qvel[dq_addresses], data.time = q, dq, timestamp
            mujoco.mj_kinematics(model, data)
            native_base = _mujoco_pose(data.xpos[base_id], data.xmat[base_id])
            native_tcp = _mujoco_pose(data.site_xpos[site_id], data.site_xmat[site_id])
            urdf.update_cfg(dict(zip(names, q)))
            planner_tcp = base @ urdf.get_transform(
                fixture["tool_link"], frame_from=fixture["base_link"]) @ tool
            errors = {"native_base_vs_fixture": _pose_error(native_base, base),
                      "native_tcp_vs_frozen_eef": _pose_error(native_tcp, feedback_pose),
                      "urdf_tcp_vs_frozen_eef": _pose_error(planner_tcp, feedback_pose)}
            if any(value["maximum_abs_matrix_error"] > 1e-8 for value in errors.values()):
                raise ValueError(f"{segment['id']} robot entrance FK mismatch: {errors}")

            checks = {}
            for names_key, values_key in (("joint_names", "qpos"),
                                          ("gripper_joint_names", "gripper_qpos")):
                if names_key in frozen or values_key in frozen:
                    old_names, old_q = frozen[names_key], np.asarray(frozen[values_key], float)
                    if (len(set(old_names)) != len(old_names) or not set(old_names).issubset(names)
                            or old_q.shape != (len(old_names),)):
                        raise ValueError("Frozen robot coordinates must have unique explicit names and aligned values")
                    expected = q[[names.index(name) for name in old_names]]
                    error = float(np.abs(expected - old_q).max())
                    if not np.isfinite(error) or error > 1e-12:
                        raise ValueError(f"{segment['id']} recorded and frozen {values_key} disagree: {error}")
                    checks[values_key + "_maximum_abs_error"] = error
            frozen_time = frozen.get("sim_time_s")
            timestamp_error = None if frozen_time is None else abs(timestamp - float(frozen_time))
            if timestamp_error is not None and (not np.isfinite(timestamp_error) or timestamp_error > 1e-10):
                raise ValueError(f"{segment['id']} recorded and frozen timestamps disagree: {timestamp_error}")
            limits = np.array([model.jnt_range[model.joint(name).id] for name in names])
            violation = np.maximum(0., np.maximum(limits[:, 0] - q, q - limits[:, 1]))
            result[segment["id"]] = {
                "status": "validated", "segment_id": segment["id"], "entry_frame": frame,
                "sim_time_s": timestamp, "reference_frame": frozen["reference_frame"],
                # BEGIN a recorded adapter owns this imported feedback identity.
                # The UUID names this validated snapshot, not its output path or
                # a new physical observation. Keep the original recording clock
                # so planning and execution can bind the same q/dq evidence.
                "feedback_id": str(uuid4()), "timestamp_s": timestamp,
                # END moving the saved file cannot replace its evidence identity.
                "joint_names": names, "start_q": q.tolist(), "start_dq": dq.tolist(),
                "arm_joint_names": arms, "finger_joint_names": fingers,
                "finger_link_names": [link for links in finger_links.values() for link in links],
                "finger_sphere_links_by_joint": deepcopy(finger_links),
                "planning_joint_names": fixture["planning_joint_names"],
                "tool_id": frozen["tool_id"], "tool_frame": frozen["tool_frame"],
                "tool_link": fixture["tool_link"], "eef_pose": feedback_pose.tolist(),
                "base_pose": base.tolist(), "tool_transform": tool.tolist(),
                "native_joint_limits": limits.tolist(),
                "measured_joint_limit_violations": {name: float(value) for name, value in zip(names, violation)
                                                      if value > 0},
                "fk_checks": errors, "frozen_joint_checks": checks,
                "timestamp_check": {"q_dq_share_exact_recording_frame": True,
                                    "frozen_timestamp_available": frozen_time is not None,
                                    "frozen_timestamp_s": frozen_time,
                                    "absolute_error_s": timestamp_error},
                "source": (f"Named scalar robot q/dq from {episode_path.resolve()} at frame {frame}, "
                           f"sim_time_s={timestamp:.17g}; native MJCF and URDF FK checked against "
                           f"the frozen EEF feedback at {frozen_path.resolve()}"),
                "provenance": {"robot_fixture_path": str(fixture_path.resolve()),
                               "binding_audit_path": str((root / fixture["binding_audit_path"]).resolve()),
                               "episode_path": str(episode_path.resolve()),
                               "frozen_feedback_path": str(frozen_path.resolve()),
                               "joint_binding": deepcopy(binding),
                               "datasets_read": ["states/qpos", "states/qvel", "states/sim_time"],
                               # BEGIN distinguish snapshot import from live sensing.
                               "feedback_identity_scope": "this validated import of the recorded robot entrance",
                               # END UUID creation does not advance the physical clock.
                               "state_read_scope": "named robot scalar columns at the exact entrance frame",
                               "object_state_or_contact_values_read": False,
                               "physics_stepped": False, "q_dq_values_clipped_or_replaced": False},
            }
    return result


def _select_planning_joints(feedback):
    """Select a declared layout by name without changing the saved full state."""
    selected = deepcopy(feedback)
    names = feedback["planning_joint_names"]
    if (not names or len(set(names)) != len(names)
            or any(name not in feedback["joint_names"] for name in names)):
        raise ValueError("Planning joint names must be a unique nonempty subset of named feedback")
    indices = [feedback["joint_names"].index(name) for name in names]
    selected.update(joint_names=names,
                    start_q=[feedback["start_q"][index] for index in indices],
                    start_dq=[feedback["start_dq"][index] for index in indices])
    return selected


def _problem_summary(problem):
    """Expose compiled constraints and barriers without serializing an executor."""
    return {
        "joint_names": list(problem.joint_names), "phase_count": problem.phase_count,
        "anchor_count": len(problem.anchors), "stop_progress": problem.stop_progress.tolist(),
        # BEGIN phase summaries expose references without invented obligations.
        "phases": [{"name": phase.name,
                    "reference_node_count": len(phase.reference.progress),
                    "metadata": phase.metadata}
                   for phase in problem.phases],
        "source_time_map": problem.source_time_map,
        # END typed goals and process relations remain the compiler's contract.
        "events": [{"event_id": event.event_id, "skill_id": event.skill_id,
                    "kind": event.kind, "source_frame": event.source_frame,
                    "progress": event.progress, "requires_evidence": event.requires_evidence,
                    "stop": event.stop} for event in problem.events],
        "conditions": list(problem.conditions),
        # BEGIN expose actual requirement consumers with the compiled inputs.
        "requirement_consumption": problem.requirement_consumption,
        # END the report is derived from the problem, never a second contract.
        "requirement_counts": {"supports": len(problem.supports),
                               "interiors": len(problem.interiors),
                               "pose_events": len(problem.pose_events),
                               "coordinate_events": len(problem.coordinate_events)},
    }


def _write_json(path, data):
    """Write the exact named artifact only after its values are assembled."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_value, allow_nan=False) + "\n")


# BEGIN preserve explicit per-skill source boundaries at the recorded caller.
def _prepare_segments(manifest, feedback_by_segment, output):
    """Run every current candidate, preserving old labels only as comparison."""
    root = Path(manifest["workspace_root"])
    rows = []
    for segment in manifest["segments"]:
        identifier = segment["id"]
        feedback = feedback_by_segment[identifier]
        feedback_path = output.parent / "feedback" / identifier / "named-feedback.json"
        _write_json(feedback_path, feedback)
        planning_feedback = _select_planning_joints(feedback)
        program = read_json(root / segment["program_path"])
        row = {"segment_id": identifier, "entry_frame": segment["snapshot"]["entry_frame"],
               "feedback_path": str(feedback_path.resolve()),
               "planning_joint_names": list(planning_feedback["joint_names"]),
               "candidates": []}
        for candidate in segment["grounding"]["candidates"]:
            item = {"candidate_id": candidate["id"], "view_id": candidate["view_id"],
                    "scene_path": candidate["scene_path"], "status": "failed",
                    "historical_registration_status": candidate["registration_status"],
                    "historical_selection_used_as_filter": False}
            try:
                problem, provenance = prepare_candidate(
                    program, read_json(root / candidate["scene_path"]),
                    planning_feedback, identifier)
                item.update(status="passed", problem=_problem_summary(problem),
                            provenance=provenance)
            except Exception as error:
                # A candidate-specific failure must not prevent the remaining
                # supplied candidates from reaching the current code path.
                item["error"] = {"type": type(error).__name__, "message": str(error)}
            row["candidates"].append(item)
        rows.append(row)
    return rows


def run(args):
    """Validate entrances first, then compile every declared current candidate."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run this bounded CPU preparation with CUDA_VISIBLE_DEVICES='' explicitly")
    started = time.perf_counter()
    manifest, fixture = read_json(args.manifest), read_json(args.robot_fixture)
    identifiers = [segment["id"] for segment in manifest["segments"]]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("The input manifest needs unique nonempty segment identities")
    feedback = _extract_named_feedback(manifest, fixture, args.robot_fixture)
    segments = _prepare_segments(manifest, feedback, args.output)
    candidates = [candidate for segment in segments for candidate in segment["candidates"]]
    passed = sum(candidate["status"] == "passed" for candidate in candidates)
    report = {
        "scope": "Current CPU PlanningProblem construction from validated named robot entrances; no robot plan or execution is certified",
        "status": "passed" if passed == len(candidates) else "failed",
        "manifest_path": str(args.manifest.resolve()),
        "robot_fixture_path": str(args.robot_fixture.resolve()),
        "snapshots_are_independent": manifest["snapshots_are_independent"],
        "continuous_execution_tested": False, "gpu_used": False,
        "object_state_or_contact_values_read": False,
        "counts": {"validated_entrances": len(feedback), "candidates_attempted": len(candidates),
                   "candidates_constructed": passed, "candidate_failures": len(candidates) - passed},
        "segments": segments,
        "elapsed_s": time.perf_counter() - started,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
    }
    _write_json(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--robot-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"],
                      "counts": report["counts"], "elapsed_s": report["elapsed_s"],
                      "peak_process_rss_mib": report["peak_process_rss_mib"]}))
    return 0 if report["status"] == "passed" else 1
# END preserve explicit per-skill source boundaries at the recorded caller.


if __name__ == "__main__":
    raise SystemExit(main())
# END manifest-driven named robot feedback and current problem preparation.
