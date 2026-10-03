"""Stream independent CPU solid queries for one recorded candidate snapshot.

This is diagnostic evidence, including rejected curves. It neither constructs
an executable PlanningResult nor changes a planner, controller or simulator.
"""

# BEGIN exact captured-query replay, separate from executable preflight.
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicHermiteSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.execution_binding import curve_snapshot
from manipulation.gripper_trajectory import FingerTrajectory


MAX_QUERIES = 4608
MAX_INPUT_BYTES = 64 * 1024**2
MAX_RSS_BYTES = int(1.5 * 1024**3)


def read_json(path):
    """Bound text allocation before parsing the complete scene inventory."""
    path = Path(path)
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"Diagnostic input exceeds 64 MiB: {path}")
    return json.loads(path.read_text())


def captured_queries(snapshot, curve):
    """Validate and exactly deduplicate captured states, retaining every index.

    The original float32 values have already been copied into JSON numbers.
    They are promoted exactly to binary64 for Coal, never regenerated from a
    nearby cubic sample. Native dtype remains part of identity because it also
    controls the separately sampled nominal hand polynomial. Coal's scene
    adapter still compares declared rule intervals in binary64; this is an
    independent solid query, not a bit-identical replay of cuRobo's ESDF cost
    or its quantized rule thresholds.
    """
    if (snapshot.get("classification") != "diagnostic_collision_snapshot_not_executable"
            or snapshot.get("curve") != curve_snapshot(curve)
            or snapshot["formal_clock"]["duration"] != curve.duration):
        raise ValueError("Snapshot and archived curve/clock do not identify the same candidate")
    capture = snapshot["scene_query_capture"]
    if capture.get("status") != "completed" or capture.get("errors") or not capture.get("calls"):
        raise ValueError("Complete captured scene query inputs are required for this diagnostic")
    unique, identities, mapping, total = [], {}, [], 0
    for call_index, call in enumerate(capture["calls"]):
        names = tuple(call["joint_names"])
        if len(set(names)) != len(names) or set(names) != set(curve.joint_names):
            raise ValueError("Captured coordinates must name every archived active joint exactly once")
        times, progress, permission = (np.asarray(call[key], float) for key in
            ("time_s", "progress", "robot_owner_permission_progress"))
        q = np.asarray(call["joint_position"], float)
        if call["time_dtype"] not in ("torch.float32", "torch.float64"):
            raise ValueError("Captured hand queries require their supported original floating dtype")
        # The production tensor comparison converts duration to this same
        # dtype. Its rounded endpoint may be above the binary64 curve end;
        # preserving that value is not extrapolating the archived cubic.
        dtype = np.float32 if call["time_dtype"] == "torch.float32" else np.float64
        native_duration = float(dtype(curve.duration))
        if (times.ndim != 2 or times.shape[0] != 1 or progress.shape != times.shape
                or permission.shape != times.shape or q.shape != (*times.shape, len(names))
                or not all(np.isfinite(value).all() for value in (times, progress, permission, q))
                or np.any(times < 0.) or np.any(times > native_duration)
                or np.any(progress < 0.) or np.any(permission < 0.)):
            raise ValueError("Captured query inputs must be finite single-batch original-clock states")
        total += times.size
        if total > MAX_QUERIES:
            raise ValueError(f"Captured diagnostic query count exceeds the explicit {MAX_QUERIES} budget")
        indices = []
        columns = [names.index(name) for name in curve.joint_names]
        for index in range(times.size):
            coordinates = q[0, index, columns].copy()
            key = (float(times[0, index]), float(progress[0, index]), float(permission[0, index]),
                   call["time_dtype"], call["progress_dtype"], *coordinates.tolist())
            if key not in identities:
                identities[key] = len(unique)
                unique.append({"time_s": key[0], "progress": key[1], "permission_progress": key[2],
                    "time_dtype": key[3], "progress_dtype": key[4], "q": coordinates,
                    "first_capture": {"call": call_index, "index": index}})
            indices.append(identities[key])
        mapping.append({"call": call_index, "shape": list(times.shape), "unique_state_indices": indices})
    return unique, mapping, total


def nominal_hand(scene, curve):
    """Reuse the production hand sampler on CPU, retaining model uncertainty.

    A timed command reference supplies only a conditional diagnostic hand when
    no nominal model was bound. It must not become measured finger motion or
    turn its descendant collisions into established arm/background conflicts.
    """
    hand = scene.get("gripper")
    if hand is None:
        return None, {"source": "no_bound_gripper", "conditional_command_reference": False}
    model = hand.get("nominal_model")
    source = "nominal_model"
    if model is None and hand["controller"] == "timed_reference":
        model, source = hand.get("command_reference"), "command_reference"
    if model is None:
        if hand["controller"] == "position_tracking" and set(hand["joint_names"]) <= set(curve.joint_names):
            return None, {"source": "captured_active_finger_coordinates", "conditional_command_reference": False}
        raise ValueError("Future finger geometry has no explicit nominal model or timed command reference")
    times, positions, velocities = (np.asarray(model[key], float) for key in ("times", "position", "velocity"))
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0 or times[-1] != curve.duration
            or np.any(np.diff(times) <= 0) or positions.shape != velocities.shape
            or positions.shape != (len(times), len(hand["joint_names"]))
            or not np.array_equal(positions[0], np.asarray(hand["current_positions"], float))
            or not all(np.isfinite(value).all() for value in (times, positions, velocities))):
        raise ValueError("Nominal hand must retain the complete original named polynomial and clock")
    # Only the geometric polynomial fields are needed by the existing sampler;
    # nominal_model deliberately does not pretend to be a command contract.
    polynomial = SimpleNamespace(times=times, duration=curve.duration,
        _curve=CubicHermiteSpline(times, positions, velocities, axis=0))

    def sample(time_s, dtype="torch.float64"):
        import torch
        query = torch.tensor(float(time_s), dtype={"torch.float32": torch.float32,
                                                  "torch.float64": torch.float64}[dtype])
        q, dq = FingerTrajectory.sample_torch(polynomial, query)
        return q.numpy().astype(float), dq.numpy().astype(float)

    return sample, {"source": source, "joint_names": list(hand["joint_names"]),
        "conditional_command_reference": source == "command_reference",
        "measured_entrance_preserved": True,
        "conditional_interval": "strictly positive trajectory time; measured entrance remains established",
        "scope": "declared future geometry hypothesis; never observed or predicted physical contact response"}


def unordered_witnesses(value):
    """Keep raw Coal points without assigning octree return order to actors."""
    if isinstance(value, list):
        return [unordered_witnesses(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: unordered_witnesses(item) for key, item in value.items()
              if key not in {"witness_first", "witness_second", "normal_first_to_second"}}
    if "witness_first" in value:
        result["coal_raw_witness_points_unordered"] = [value["witness_first"], value["witness_second"]]
        result["coal_raw_normal"] = value.get("normal_first_to_second")
        result["witness_scope"] = "raw Coal return order; no actor/target assignment, especially for octrees"
    return result


def conditional_hand_issue(issue, names, scene):
    """Identify rows whose geometry depends on an unconfirmed command hand."""
    evidence = issue.get("evidence", issue)
    if any(evidence.get(key) in names for key in ("first", "second", "actor_geometry", "target_geometry")):
        return True
    rule_id = issue.get("rule_id")
    if rule_id is not None:
        materials = scene["native_geometry_materials"]
        return any(rule["rule_id"] == rule_id and rule["actor"] == "robot"
                   and any(materials.get(name) == rule["material_id"] for name in names)
                   for rule in scene["contacts"])
    return False


def query_state(validator, scene_binding, state, active_names, sampler, uncertain_names):
    """Perform one complete native solid query using the recorded arm state."""
    # Production retains measured hand geometry at t=0. Only future command
    # references lack an independently bound hand model; neither scope may
    # erase an initial measured collision or claim future contact tracking.
    uncertain_names = uncertain_names if state["time_s"] > 0. else set()
    positions = dict(zip(active_names, state["q"]))
    if sampler is not None:
        fingers, _ = sampler(state["time_s"], state["time_dtype"])
        for name, value in zip(validator.finger_joint_names, fingers):
            if name in positions:
                raise ValueError("Independent nominal fingers must not replace captured active coordinates")
            positions[name] = float(value)
    if set(positions) != set(validator.joint_names):
        raise ValueError("Captured arm and explicit nominal hand must supply the complete named robot")
    full = [positions[name] for name in validator.joint_names]
    robot = validator.checker.query(joint_names=validator.joint_names, joint_positions=full,
        T_reference_base=validator.T_reference_base, pairs=validator.pairs, frames=[validator.tool_frame])
    tool_pose = np.asarray(robot["frame_poses"][validator.tool_frame]) @ validator.T_frame_tcp
    scene = validator.checker.query_scene(robot, tool_pose=tool_pose, progress=state["progress"],
        finger_positions=np.asarray([positions[name] for name in validator.finger_joint_names]),
        state_kind="nominal", active_robot_owner_permissions=None,
        robot_owner_permission_progress=state["permission_progress"])
    issues = [{"kind": "independent_robot_collision", "evidence": pair}
              for pair in robot["pairs"] if pair["policy"] == "required" and pair["relation"] != "separated"]
    issues += [{"kind": "independent_joint_position", "name": name, **row}
               for name, row in robot["joint_limit_violations"].items()]
    issues += scene["violations"]
    established, conditional = [], []
    for issue in issues:
        hand_issue = (issue.get("name") in validator.finger_joint_names
                      if issue["kind"] == "independent_joint_position" else
                      conditional_hand_issue(issue, uncertain_names, scene_binding))
        (conditional if uncertain_names and hand_issue else established).append(issue)
    if uncertain_names:
        scene["observed_scene_checked"] = False
        scene["unresolved_assumptions"].append({"kind": "command_reference_only_nominal_hand",
            "geometry_names": sorted(uncertain_names), "scope": "extra conditional diagnostic, not established future geometry"})
    for pair in (*robot["pairs"], *scene["pairs"]):
        raw = pair.get("native_pair_evidence", pair)
        if not np.isfinite(raw["signed_distance_m"]):
            raise ValueError("Independent query returned a nonfinite raw distance")
    return unordered_witnesses({"joint_names": list(validator.joint_names), "joint_positions": full,
        "robot": robot, "scene": scene, "established_model_violations": established,
        "conditional_nominal_hand_violations": conditional,
        "established_model_passed": not established and not scene["query_gaps"],
        "all_nominal_solids_passed": not issues and not scene["query_gaps"]})


def update_minima(minima, state_index, state, result, uncertain_names):
    """Keep bounded per-geometry/owner minima while streaming full state rows."""
    for row in result["scene"]["pairs"]:
        raw = row.get("native_pair_evidence", row)
        key = (row["actor"], row["actor_geometry"], row["target_owner"])
        distance = raw["signed_distance_m"]
        if key not in minima or distance < minima[key]["signed_distance_m"]:
            minima[key] = {"state_index": state_index, "time_s": state["time_s"],
                "progress": state["progress"], "permission_progress": state["permission_progress"],
                "signed_distance_m": distance,
                "conditional_nominal_hand": (state["time_s"] > 0. and
                    (row["actor_geometry"] in uncertain_names or row["target_geometry"] in uncertain_names)),
                "scope": "raw signed distance; permission and conditional geometry remain separate",
                "evidence": deepcopy(row)}


def audit(snapshot_path, fixture_path, output):
    """Run all captured queries within the separate finite diagnostic budget."""
    from recorded_simulation import load_curve_archive
    from robot_validation_fixture import build_robot_validator

    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"classification": "independent_collision_diagnostic_not_executable", "status": "running",
        "snapshot_path": str(Path(snapshot_path).resolve()), "fixture_path": str(Path(fixture_path).resolve()),
        "maximum_captured_queries": MAX_QUERIES, "maximum_rss_bytes": MAX_RSS_BYTES,
        "numeric_scope": "Captured native-dtype states with independent binary64 solid/rule queries; not cuRobo cost equivalence",
        "executed_unique_states": 0, "execution_authorized": False, "physics_called": False,
        "continuous_collision_certified": False}
    minima, unresolved = {}, {}
    try:
        snapshot = read_json(snapshot_path)
        curve_path = Path(snapshot_path).parent/snapshot["curve_path"]
        if curve_path.parent.resolve() != Path(snapshot_path).parent.resolve():
            raise ValueError("Snapshot curve reference must stay in the same archive directory")
        curve, metadata = load_curve_archive(curve_path)
        states, mapping, total = captured_queries(snapshot, curve)
        report.update(curve_path=str(curve_path.resolve()), source_classification=metadata["classification"],
            validation_status=snapshot["validation_status"], captured_query_count=total,
            unique_state_count=len(states), capture_to_unique_state=mapping,
            formal_validation=snapshot["validation"],
            clock_scope={"formal_validation_u": "original exported binary64 normalized query parameters",
                         "captured_time_s": "actual native-dtype scene inputs; preserved separately from formal u"})
        problem, scene = snapshot["problem"], snapshot["scene"]
        validator = build_robot_validator(read_json(fixture_path), reference_frame=problem["reference_frame"],
            T_reference_base=np.asarray(problem["base_pose"], float), tool_frame=problem["tool_link"],
            T_frame_tcp=np.asarray(problem["tool_transform"], float))
        sampler, hand_scope = nominal_hand(scene, curve)
        validator.bind_scene(scene, lambda value: float(curve.sample(value)[3]),
            nominal_fingers=None if sampler is None else lambda value: sampler(value))
        validator._bind_release_predictions(curve)
        uncertain = (set(validator.checker.geometry_depending_on(validator.finger_joint_names))
                     if hand_scope["conditional_command_reference"] else set())
        report.update(nominal_hand=hand_scope, conditional_hand_geometry=sorted(uncertain),
            owner_ids=[row["owner_id"] for row in scene["owners"]],
            established_violation_states=0, conditional_hand_violation_states=0, query_gap_states=0)
        with (output/"states.jsonl").open("w") as stream:
            for index, state in enumerate(states):
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > MAX_RSS_BYTES:
                    raise MemoryError("Independent diagnostic exceeded its 1.5 GiB resident memory budget")
                result = query_state(validator, scene, state, curve.joint_names, sampler, uncertain)
                record = {"state_index": index, **{key: value for key, value in state.items() if key != "q"},
                          "captured_active_joint_positions": state["q"].tolist(), **result}
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
                report["executed_unique_states"] += 1
                report["established_violation_states"] += int(bool(result["established_model_violations"]))
                report["conditional_hand_violation_states"] += int(bool(result["conditional_nominal_hand_violations"]))
                report["query_gap_states"] += int(bool(result["scene"]["query_gaps"]))
                for assumption in result["scene"]["unresolved_assumptions"]:
                    identity = json.dumps(assumption, sort_keys=True, allow_nan=False)
                    if identity not in unresolved:
                        unresolved[identity] = {"assumption": deepcopy(assumption), "state_count": 0,
                                                "first_state_index": index}
                    unresolved[identity]["state_count"] += 1
                update_minima(minima, index, state, result, uncertain)
        report.update(status="completed", all_captured_states_checked=True,
            established_model_passed=report["established_violation_states"] == report["query_gap_states"] == 0,
            all_nominal_solids_passed=report["established_violation_states"] == report["conditional_hand_violation_states"] == report["query_gap_states"] == 0)
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}", all_captured_states_checked=False)
        (output/"exception.txt").write_text(traceback.format_exc())
    finally:
        report["scene_geometry_owner_minima"] = list(minima.values())
        report["unresolved_assumptions_summary"] = list(unresolved.values())
        report["minimum_scene_distance"] = min(minima.values(), key=lambda row: row["signed_distance_m"], default=None)
        report["minimum_established_scene_distance"] = min(
            (row for row in minima.values() if not row["conditional_nominal_hand"]),
            key=lambda row: row["signed_distance_m"], default=None)
        report["minimum_conditional_hand_scene_distance"] = min(
            (row for row in minima.values() if row["conditional_nominal_hand"]),
            key=lambda row: row["signed_distance_m"], default=None)
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "robot-fixture", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.snapshot, args.robot_fixture, args.output)
    print(json.dumps({key: result[key] for key in ("status", "executed_unique_states", "elapsed_seconds", "peak_rss_bytes")}))
    raise SystemExit(0 if result["status"] == "completed" else 1)
# END a complete diagnostic never authorizes a rejected trajectory or physical execution.
