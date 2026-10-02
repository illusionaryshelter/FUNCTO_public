#!/usr/bin/env python3
"""Recheck archived IK configurations in the current recorded planning scene.

The old success flags select inputs only. Every saved arm coordinate is kept,
and current source targets, limits, timed fingers, native sphere constraints,
observed scene and nominal pose accuracy are checked again. Native CPU solids
provide independent self/installation evidence. Exact robot unknown-space
queries run only after the other static checks pass. No IK, search, trajectory
solve, dynamics claim, execution, or local reachability proof is performed.
"""

# BEGIN bounded archived-state diagnostic; no production behavior is changed.
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from probe_constraint_generation import cpu, read_json, require_equal, write_report


def archived_targets(problem, args, report, arrays):
    """Bind every archived target to an unchanged current source vertex.

    The 1e-8 comparison is the existing recorded target's identity allowance,
    not a relaxed planning tolerance. Numerical joint grouping is descriptive:
    no row is removed, averaged, projected or replaced by a representative.
    """
    targets, all_q, seen = [], [], set()
    for frame_text, filename in args.target_archive:
        frame, path = int(frame_text), Path(filename)
        if frame in seen:
            raise ValueError("Each source frame must have exactly one archive")
        seen.add(frame)
        with np.load(path, allow_pickle=False) as saved:
            names = saved["joint_names"].tolist()
            positions = saved["joint_position"].copy()
            selected = np.flatnonzero(saved["native_success"].reshape(-1))
            target = saved["target_tool_pose"].copy()
            progress = float(saved["progress"])
        if (len(names) != len(set(names)) or set(names) != set(problem.joint_names)
                or positions.ndim != 2 or positions.shape[1] != len(names)
                or not np.isfinite(positions).all() or not len(selected)
                or target.shape != (4, 4) or not np.isfinite(target).all()):
            raise ValueError("Saved inputs require finite named states and a target pose")
        matches = []
        for phase_index, phase in enumerate(problem.phases):
            source_frames = phase.metadata.get("source_frames", [])
            if frame in source_frames:
                local = float(phase.reference.progress[source_frames.index(frame)])
                matches.append((phase_index+local, phase.reference.sample_numpy(np.array([local]))[0]))
        if not matches or not all(np.allclose(pose, target, atol=1e-8, rtol=0) for _, pose in matches):
            raise ValueError(f"Archived target differs from current source frame {frame}")
        if progress != matches[-1][0]:
            raise ValueError("Archived progress differs from current source-vertex ownership")
        columns = [names.index(name) for name in problem.joint_names]
        values = positions[selected][:, columns].copy()
        # Reject casts that would silently alter an archived coordinate when
        # it enters the actual float32 native planning model.
        if not np.array_equal(values, values.astype(np.float32)):
            raise ValueError("Archived q cannot enter the native model without changing values")
        key = f"frame_{frame}"
        arrays[f"{key}.archive_joint_names"] = np.asarray(names)
        arrays[f"{key}.archive_rows"] = selected
        arrays[f"{key}.joint_position"] = values
        arrays[f"{key}.target_tool_pose"] = target
        row = {"source_frame": frame, "archive": str(path.resolve()), "progress": progress,
               "archive_rows": selected.tolist(), "candidate_count": len(selected),
               "old_native_success_used_only_to_select_rows": True,
               "target_identity_atol": 1e-8,
               "target_maximum_absolute_identity_error": max(float(np.max(np.abs(pose-target))) for _, pose in matches),
               "target_tool_pose": target.tolist(), "states": []}
        targets.append((row, values, target))
        all_q.extend(values)
    if not 1 <= len(all_q) <= 9 or len(targets) > 3:
        raise ValueError("This archived-state diagnostic admits at most nine inputs at three targets")
    report["targets"] = [row for row, _, _ in targets]
    report["arm_configuration_statistics"] = configuration_statistics(np.asarray(all_q), problem.numerical.joint)
    report["input_candidate_count"] = len(all_q)
    return targets


def configuration_statistics(values, tolerance):
    """Count exact rows and deterministic joint-band groups without filtering."""
    representatives, groups = [], []
    for row in values:
        group = next((index for index, value in enumerate(representatives)
                      if np.all(np.abs(row.astype(float)-value.astype(float)) <= tolerance)), None)
        if group is None:
            group = len(representatives)
            representatives.append(row)
        groups.append(group)
    return {"rows": len(values), "exact_unique_rows": len(np.unique(values, axis=0)),
            "joint_numerical_band": float(tolerance), "joint_band_groups": len(representatives),
            "group_per_input_row": groups,
            "group_rule": "First input representative within the existing per-joint numerical band",
            "rows_dropped": 0}


def verify_original_scene(problem, scene, budget, args, report):
    """Require the original recorded evidence and measured entrance identity."""
    checks = report["archive_consistency"]
    source_report = read_json(args.source_run/"planning"/"planning-report.json")
    if asdict(budget) != source_report["budget"] or budget.trajectory_seeds != 1:
        raise ValueError("State queries require exactly the original one-seed budget")
    checks["budget_identical"] = True
    with np.load(args.source_run/"observed-free.npz", allow_pickle=False) as saved:
        for name in ("known_free", "origin", "voxel_size", "observation_ids"):
            require_equal(f"observed_free.{name}", getattr(scene.coverage, name), saved[name], checks)
    with np.load(args.source_run/"entrance-scene.npz", allow_pickle=False) as saved:
        with np.load(args.output/"scene-build"/"entrance-scene.npz", allow_pickle=False) as rebuilt:
            for name in ("joint_names", "actual_q", "actual_dq", "measured_robot_spheres"):
                require_equal(f"entrance.{name}", rebuilt[name], saved[name], checks)
        columns = [saved["joint_names"].tolist().index(name) for name in problem.joint_names]
        require_equal("measured_q0", problem.start_q, saved["actual_q"][columns], checks)
        require_equal("measured_dq0", problem.start_dq, saved["actual_dq"][columns], checks)


def check_states(problem, scene, robot, budget, args, report, arrays, save):
    """Evaluate one immutable arm state at each actual existing anchor time."""
    from manipulation.planning import _CuroboPlanner, _timed_native_state
    from robot_validation_fixture import build_robot_validator

    targets = archived_targets(problem, args, report, arrays)
    verify_original_scene(problem, scene, budget, args, report)
    planner = _CuroboPlanner(problem, scene, robot, budget, "cuda:0")
    try:
        torch, names = planner.torch, list(planner.joint_names)
        checked_scene = planner._verification_scene
        checks = report["archive_consistency"]
        with np.load(args.source_run/"planning"/"optimizer-input-00.npz", allow_pickle=False) as saved:
            require_equal("joint_names", names, saved["joint_names"], checks)
            require_equal("native_q0", cpu(planner.start.position), saved["start_q"], checks)
            require_equal("native_dq0", cpu(planner.start.velocity), saved["start_dq"], checks)
            for key in ("u", "s"):
                require_equal(f"progress_{key}", getattr(planner.progress_map, key), saved[f"progress_{key}"], checks)
        config = read_json(args.source_run/"planning"/"effective-config-00.json")
        if planner.formal_clock != config["formal_clock"]:
            raise ValueError("Current formal clock differs from the original archive")
        checks["formal_clock_identical"] = True
        if checked_scene.robot_coverage is None:
            raise ValueError("The current exact native-solid unknown-space oracle is required")
        cspace = planner._verification_native.get_cost("cspace")
        native_self = planner._verification_native.get_cost("self_collision")
        if cspace is None or not cspace.enabled or native_self is None or not native_self.enabled:
            raise ValueError("Current native bounds and self constraints must remain enabled")
        limits = cspace.config.joint_limits
        columns = [limits.joint_names.index(name) for name in names]
        bounds = cpu(limits.position)[:, columns]
        validator = build_robot_validator(read_json(args.robot_fixture), reference_frame=problem.reference_frame,
            T_reference_base=problem.base_pose, tool_frame=problem.tool_link, T_frame_tcp=problem.tool_transform)
        report.update(formal_clock=planner.formal_clock.copy(), budget=asdict(budget),
            numerical_tolerances=asdict(problem.numerical), joint_names=names,
            position_limits={"lower": bounds[0].tolist(), "upper": bounds[1].tolist()},
            cpu_geometry_model=validator.describe(), scene_workspace_bytes=planner.scene_workspace_bytes,
            anchor_fk_workspace_bytes=planner.anchor_fk_workspace_bytes,
            native_source=str(sys.modules["curobo"].__file__))
        native_self.setup_batch_tensors(1, 1)
        save("current_planner_initialized_without_solve")
        with torch.no_grad():
            for target_row, source_values, target in targets:
                # The planner may use a different native joint ordering. This
                # is only a named permutation; require_equal below checks q.
                values = source_values[:, [problem.joint_names.index(name) for name in names]]
                target_row["configuration_statistics"] = configuration_statistics(values, problem.numerical.joint)
                anchors = planner.progress_map.anchor_parameters([target_row["progress"]])
                target_row["anchor_times_s"] = (anchors[:, 0]*planner.formal_clock["duration"]).tolist()
                for candidate, q in enumerate(values):
                    for time_index, (u, progress) in enumerate(anchors):
                        elapsed = float(u*planner.formal_clock["duration"])
                        label = f"frame_{target_row['source_frame']}.row_{target_row['archive_rows'][candidate]}.time_{time_index}"
                        qt = planner.device_cfg.to_device(q[None, None]).contiguous()
                        require_equal(f"unchanged_q.{label}", cpu(qt)[0, 0], q, checks)
                        # No derivatives are measured for an archived IK q.
                        # Query only configuration-dependent native terms.
                        joint_state = planner.JointState(position=qt, joint_names=names)
                        state = planner._verification_fk.compute_kinematics(joint_state)
                        tool_index = state.tool_poses.tool_frames.index(problem.tool_link)
                        p, r = problem.transform_tool_pose(state.tool_poses.position[:, :, tool_index],
                                                           state.tool_poses.quaternion[:, :, tool_index])
                        s, t = qt.new_full((1, 1), progress), qt.new_full((1, 1), elapsed)
                        timed = _timed_native_state(planner._RobotState(joint_state=joint_state,
                            cuda_robot_model_state=state), problem, checked_scene, names, t)
                        sphere_self = float(native_self.forward(timed.robot_spheres).max().item())
                        scene_terms = {name: float(value.max().item()) for name, value in checked_scene.components(
                            problem.transform_robot_spheres(state.robot_spheres), p, r, s, qt, names,
                            trajectory_time=t).items()}
                        if not all(np.isfinite(value) for value in (sphere_self, *scene_terms.values())):
                            raise FloatingPointError("A current static state constraint is nonfinite")
                        pose = np.eye(4)
                        pose[:3, 3] = cpu(p)[0, 0]
                        pose[:3, :3] = Rotation.from_quat(cpu(r)[0, 0, [1, 2, 3, 0]]).as_matrix()
                        position_error = float(np.linalg.norm(pose[:3, 3]-target[:3, 3]))
                        rotation_error = float(Rotation.from_matrix(target[:3, :3].T@pose[:3, :3]).magnitude())
                        excess = np.maximum(bounds[0]-q, 0)+np.maximum(q-bounds[1], 0)
                        pose_passed = position_error <= problem.numerical.position_m and rotation_error <= problem.numerical.rotation_rad
                        # Match final production solid checks: saved native q
                        # cast to float64, original time and timed hand law.
                        cpu_q = torch.as_tensor(q[None, None], dtype=torch.float64)
                        cpu_t = torch.as_tensor([[elapsed]], dtype=torch.float64)
                        full_names, full_q = checked_scene.full_robot_positions(cpu_q, names, cpu_t)
                        physical = validator.checker.query(joint_names=full_names,
                            joint_positions=cpu(full_q)[0, 0], T_reference_base=problem.base_pose,
                            pairs=validator.pairs, frames=[problem.tool_link])
                        physical_bad = [row for row in physical["pairs"]
                                        if row["policy"] == "required" and row["relation"] != "separated"]
                        known_scene = {key: value for key, value in scene_terms.items() if not key.startswith("unknown/")}
                        other_unknown = {key: value for key, value in scene_terms.items() if key.startswith("unknown/")}
                        gates = {"native_position_bounds": bool(not excess.any()), "native_sphere_self": sphere_self <= 0,
                            "known_scene_and_hand": all(value <= 0 for value in known_scene.values()),
                            "other_scene_evidence": all(value <= 0 for value in other_unknown.values()),
                            "nominal_source_pose": bool(pose_passed),
                            "cpu_native_solid_self_and_installation": not physical_bad,
                            "cpu_full_joint_position_bounds": not physical["joint_limit_violations"]}
                        state_row = {"archive_row": target_row["archive_rows"][candidate], "time_index": time_index,
                            "u": float(u), "progress": float(progress), "time_s": elapsed,
                            "native_time_s": float(t.item()), "joint_position": q.tolist(),
                            "full_joint_names": list(full_names), "full_joint_position": cpu(full_q)[0, 0].tolist(),
                            "actual_tool_pose": pose.tolist(), "position_error_m": position_error,
                            "rotation_error_rad": rotation_error, "position_limit_excess": excess.tolist(),
                            "native_sphere_self_value": sphere_self, "scene_terms": scene_terms,
                            "cpu_geometry": physical, "cpu_required_pair_failures": physical_bad,
                            "static_prerequisite_gates": gates, "exact_robot_unknown_checked": False,
                            "passed_all_tested_static_conditions": False}
                        target_row["states"].append(state_row)
                        arrays[f"{label}.timed_robot_spheres"] = cpu(timed.robot_spheres)
                        if all(gates.values()):
                            coverage = checked_scene.check_robot_coverage(cpu_q, names, cpu_t)
                            if len(coverage) != 1:
                                raise ValueError("Each admitted state requires one exact solid coverage verdict")
                            state_row.update(exact_robot_unknown_checked=True, exact_robot_unknown=coverage[0],
                                passed_all_tested_static_conditions=bool(coverage[0]["passed"]))
                        else:
                            state_row["exact_robot_unknown_skip_reason"] = "At least one earlier static prerequisite failed"
                        save(label)
                target_row["passed_candidate_rows_by_time"] = {
                    str(index): [row["archive_row"] for row in target_row["states"]
                                 if row["time_index"] == index and row["passed_all_tested_static_conditions"]]
                    for index in range(len(anchors))}
                target_row["has_archived_candidate_at_every_tested_time"] = all(target_row["passed_candidate_rows_by_time"].values())
        report["all_targets_have_archived_candidates_at_tested_times"] = all(
            row["has_archived_candidate_at_every_tested_time"] for row, _, _ in targets)
        report["continuous_connection_attempted"] = False
        report["conclusion_scope"] = ("Only these archived rows and existing anchor times were tested. "
            "Missing passing candidates stop this necessary-condition screen; they do not prove unreachability. "
            "Passing states do not prove continuous motion, full endpoint-family process feasibility, dynamics or execution.")
    finally:
        planner.close()


def run(args):
    """Build current observed evidence once, then stop at the planning boundary."""
    import torch
    import run_recorded_skill
    from manipulation import planning
    from curobo.inverse_kinematics import InverseKinematics
    from curobo.trajectory_optimizer import TrajectoryOptimizer

    args.output.mkdir(parents=True, exist_ok=False)
    arrays, started = {}, time.monotonic()
    report = {"status": "preparing", "source_run": str(args.source_run.resolve()),
        "scope": "Current static rechecks of at most nine archived IK configurations; no solve or execution",
        "archive_consistency": {}, "completed_stages": []}

    def save(stage):
        report["completed_stages"].append(stage)
        report["elapsed_seconds"] = time.monotonic()-started
        write_report(args.output/"local-state-report.json", report)
        np.savez_compressed(args.output/"local-state-arrays.npz", **arrays)
        print(json.dumps({"stage": stage, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)

    def forbidden(*unused, **kwargs):
        raise RuntimeError("This archived-state diagnostic must not call IK, optimization or path search")

    # BEGIN bind the current preparation callback without enabling a new solve.
    def capture(problem, scene, robot, output, *, budget, provenance, execution_limits, continuation):
        if continuation is not None:
            raise ValueError("Static archived-state checks do not create a continuation binding")
        if report.get("planning_boundary_intercepted"):
            raise ValueError("The runner reached its planning boundary more than once")
        report["planning_boundary_intercepted"] = True
        check_states(problem, scene, robot, budget, args, report, arrays, save)
        return SimpleNamespace(success=False, diagnostics={"initial_state": None})

    runner_args = SimpleNamespace(**vars(args))
    runner_args.output = args.output/"scene-build"
    runner_args.continuation = None
    runner_args.rebind = None  # This diagnostic constructs its initial binding only.
    runner_args.request_ledger = None  # The diagnostic is a distinct explicitly bounded request.
    # END the archive is checked against its declared original entrance.
    try:
        original = read_json(args.source_run/"run-report.json")
        for key, value in (("segment_id", args.segment), ("candidate_id", args.candidate)):
            if original[key] != value:
                raise ValueError(f"Changed original runner identity: {key}")
        for key in ("manifest", "feedback", "robot_fixture", "execution_limits", "finger_targets"):
            if Path(original[f"{key}_path"]).resolve() != getattr(args, key).resolve():
                raise ValueError(f"Changed original runner input: {key}")
        if args.scene_workspace_mib != 2048 or args.torch_memory_mib != 4096 or args.trajectory_seeds != 1:
            raise ValueError("Keep the original 2-GiB scene, 4-GiB Torch and one-seed budget")
        with patch("recorded_task_planning.plan_and_record", new=capture), \
             patch.object(planning._CuroboPlanner, "reference_seeds", new=forbidden), \
             patch.object(planning._CuroboPlanner, "optimize", new=forbidden), \
             patch.object(planning._CuroboPlanner, "ompl_seed", new=forbidden), \
             patch.object(InverseKinematics, "solve_pose", new=forbidden), \
             patch.object(TrajectoryOptimizer, "solve_pose", new=forbidden):
            run_recorded_skill.run(runner_args)
        if not report.get("planning_boundary_intercepted"):
            raise ValueError("Scene preparation did not reach the diagnostic boundary")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path = runner_args.output/"run-report.json"
        if path.exists():
            prepared = read_json(path)
            prepared.update(planning_called=False, execution_tested=False,
                            scope="Observed scene preparation for archived static state checks only")
            prepared.pop("planning_success", None)
            write_report(path.with_name("scene-build-report.json"), prepared)
            path.unlink()
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if torch.cuda.is_initialized():
            report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
        save("finished")


def parse_args():
    """Keep the original runner choices and archived source-frame IDs explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "robot-fixture", "feedback", "execution-limits", "finger-targets", "source-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--target-archive", nargs=2, action="append", required=True, metavar=("SOURCE_FRAME", "NPZ"))
    parser.add_argument("--segment", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--owner-mask", nargs=2, action="append", required=True)
    parser.add_argument("--owner-geometry", type=Path)
    parser.add_argument("--center", type=float, nargs=3, required=True)
    parser.add_argument("--extent", type=float, nargs=3, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--depth-error", type=float, required=True)
    parser.add_argument("--robot-depth-tolerance", type=float, required=True)
    parser.add_argument("--robot-depth-tolerance-source", required=True)
    parser.add_argument("--initial-state-assumption-source", required=True)
    for name, default in (("knots", 16), ("interpolation-buffer-size", 768), ("ik-seeds", 2),
                          ("trajectory-seeds", 1), ("max-validation-points", 4096), ("max-refinement-points", 16)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--scene-workspace-mib", type=int, choices=(2048,), default=2048)
    parser.add_argument("--torch-memory-mib", type=int, choices=(4096,), default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
# END bounded archived-state diagnostic; no production behavior is changed.
