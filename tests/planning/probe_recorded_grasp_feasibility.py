#!/usr/bin/env python3
"""Probe explicit recorded grasp targets without trajectory optimization.

The ordinary recorded runner builds its RGB-D scene and explicit actual-hand intervals.
A scoped test interception stops at its plan_and_record boundary; no planner,
TrajOpt, path search, simulator, or robot command is invoked. Two bounded native
IK modes then share the same measured-q0/native-Halton seed pool: pose + joint
position limits, and those same requirements plus native self collision.

All returned configurations, including rejected ones, are checked afterwards
against the unchanged original scene and enabled native self checker. Pose-only
success is kinematic evidence, never an executable grasp. Failure of a finite
seed pool does not prove that a target is unreachable.
"""

# BEGIN isolated, bounded goal diagnostics using the existing recorded scene.
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "2"

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prepare_recorded_task import prepare_candidate, read_json, json_value


MODES = (("pose_and_limits", False), ("native_self_collision", True))
LBFGS_ITERATIONS = 80
LM_ITERATIONS = 16


def write_report(path, report):
    """Save strict JSON after each completed stage, including partial failures."""
    path.write_text(json.dumps(report, default=json_value, indent=2, allow_nan=False)+"\n")


def target_inputs(args):
    """Resolve original source IDs before grounding the protected approach.

    This CPU preparation reads observed candidate geometry and named robot
    feedback only. The explicit pre-grasp frame goes into ground_program;
    slicing an already warped historical free prefix would change the goal.
    """
    from manipulation.program import ground_program

    manifest = read_json(args.manifest)
    root = Path(manifest["workspace_root"])
    segment = next(row for row in manifest["segments"] if row["id"] == args.segment)
    candidate = next(row for row in segment["grounding"]["candidates"] if row["id"] == args.candidate)
    program_path, scene_path = root/segment["program_path"], root/candidate["scene_path"]
    program, scene, feedback = read_json(program_path), read_json(scene_path), read_json(args.feedback)
    if program["skill_kind"] != "grasp":
        raise ValueError("This target probe requires an explicit grasp program")
    measured = dict(zip(feedback["joint_names"], feedback["start_q"]))
    velocity = dict(zip(feedback["joint_names"], feedback["start_dq"]))
    names = feedback["arm_joint_names"]
    active_feedback = {**feedback, "joint_names": names,
                       "start_q": [measured[name] for name in names],
                       "start_dq": [velocity[name] for name in names]}
    problem, provenance = prepare_candidate(program, scene, active_feedback, args.segment)
    grounded = ground_program(program, scene)
    targets = []
    for frame in args.target_frame:
        rows = np.flatnonzero(np.asarray(grounded["frame_indices"]) == frame)
        if len(rows) != 1:
            raise ValueError(f"Target frame {frame} must identify exactly one original grounded row")
        matches = []
        for index, phase in enumerate(problem.phases):
            source_frames = phase.metadata.get("source_frames", [])
            if frame in source_frames:
                row = source_frames.index(frame)
                local = float(phase.reference.progress[row])
                # Shared endpoints agree geometrically. The later phase owns
                # that exact source vertex in the current public builder.
                matches.append((index+local, phase.reference.sample_numpy(np.array([local]))[0]))
        if not matches or not all(np.allclose(pose, grounded["eef_poses"][rows[0]], atol=1e-8, rtol=0)
                                  for _, pose in matches):
            raise ValueError("Grounded target and current problem must share exact source-vertex geometry")
        targets.append({"source_frame": int(frame), "source_timestamp_s": float(grounded["timestamps"][rows[0]]),
                        "progress": float(matches[-1][0]),
                        "T_reference_tool": np.array(grounded["eef_poses"][rows[0]], float),
                        "events": [{"event_id": event.event_id, "kind": event.kind,
                                    "requires_evidence": event.requires_evidence}
                                   for event in problem.events if event.source_frame == frame]})
    return problem, targets, {"program_path": str(program_path), "scene_path": str(scene_path),
        "manifest_path": str(args.manifest.resolve()), "feedback_path": str(args.feedback.resolve()),
        "robot_fixture_path": str(args.robot_fixture.resolve()),
        "protected_start_frame": int(grounded["protected_start_frame"]),
        "target_source": "current ground_program reference samples with observed candidate binding; diagnostic IK queries, not hard goals",
        "builder": provenance, "ground_truth_object_state_read": False}


def observed_scene(args):
    """Reuse the runner's preparation, intercepting only its planning boundary.

    The runner has no public build-scene function. A scoped replacement avoids
    duplicating its calibrated masking/owner partition and leaves production
    code untouched. Its transient orchestration report is renamed immediately
    and explicitly records that the real planning function was never called.
    """
    import run_recorded_skill

    captured = {}

    # BEGIN retain every explicit current runner argument without a permissive sink.
    def capture(problem, scene, robot, output, *, robot_validator, budget, provenance, execution_limits,
                continuation=None, rebind=None, request_ledger=None, backend=None, joint_reference=None):
        if continuation is not None or rebind is not None or request_ledger is not None or joint_reference is not None:
            raise ValueError("Prepared grasp diagnostics do not create or continue a planning request")
        if captured:
            raise ValueError("The recorded runner must reach its preparation boundary exactly once")
        captured.update(problem=problem, scene=scene, robot=robot, robot_validator=robot_validator,
                        budget=budget, provenance=provenance,
                        execution_limits=execution_limits)
        # BEGIN preparation metadata satisfies the runner without claiming a solve.
        return SimpleNamespace(success=False, diagnostics={"initial_state": None})
        # END the captured scene remains the diagnostic's actual output.
    # END the preparation boundary preserves the required physical source model.

    runner_args = SimpleNamespace(**vars(args))
    runner_args.output = args.output/"scene-build"
    runner_args.continuation = None
    runner_args.rebind = None  # This diagnostic constructs its initial binding only.
    runner_args.request_ledger = None  # The diagnostic is a distinct explicitly bounded request.
    # Required runner orchestration metadata only: capture prevents any use
    # of its trajectory budget, native trajectory allocator, or path search.
    runner_args.knots = 8
    runner_args.interpolation_buffer_size = 512
    runner_args.trajectory_seeds = 1
    runner_args.max_validation_points = 1024
    # BEGIN diagnostic preparation supplies the current bounded runner contract.
    runner_args.max_task_anchors = 256
    runner_args.maximum_trajectory_solves = 2
    runner_args.planning_seconds = 120.
    runner_args.joint_reference = None
    runner_args.host_process_root_pid = None
    # END capture still prevents all native planning work.
    try:
        with patch("recorded_task_planning.plan_and_record", new=capture):
            run_recorded_skill.run(runner_args)
    finally:
        source = runner_args.output/"run-report.json"
        if source.exists():
            report = read_json(source)
            report.update(planning_called=False, trajectory_planning_called=False,
                          planning_boundary_intercepted=bool(captured),
                          scope="Unchanged observed-scene preparation for isolated target diagnostics")
            report.pop("planning_success", None)
            write_report(source.with_name("scene-build-report.json"), report)
            source.unlink()
    if not captured:
        raise ValueError("Scene preparation did not reach the explicit diagnostic boundary")
    return captured


def isolated_configs(self_collision):
    """Keep native pose/limits costs, explicitly isolate both metrics and IK.

    In this installed native API self_collision_check controls optimizer
    rollouts only. The separately constructed metrics rollout needs its own
    explicit diagnostic setting; otherwise a purported pose-only success flag
    still incorporates native self collision. No production config is changed.
    """
    from curobo.content import get_task_configs_path
    from curobo._src.util.config_io import join_path, load_yaml

    optimizer = deepcopy(load_yaml(join_path(get_task_configs_path(), "ik/lbfgs_ik.yml")))
    metrics = deepcopy(load_yaml(join_path(get_task_configs_path(), "metrics_base.yml")))
    for configuration in (optimizer, metrics):
        for key in ("cost_cfg", "constraint_cfg", "hybrid_cost_constraint_cfg", "convergence_cfg"):
            manager = configuration["rollout"].get(key)
            if manager is not None:
                manager.pop("scene_collision_cfg", None)
                if not self_collision:
                    manager.pop("self_collision_cfg", None)
    return optimizer, metrics


def manager_evidence(solver, self_collision):
    """Check actual constructed managers instead of trusting a flag's name."""
    managers = []
    for index, rollout in enumerate(solver.get_all_rollout_instances()):
        for name in ("cost_manager", "constraint_manager", "hybrid_cost_constraint_manager",
                     "metrics_cost_manager", "metrics_constraint_manager",
                     "metrics_hybrid_cost_constraint_manager", "metrics_convergence_manager"):
            manager = getattr(rollout, name)
            if manager is None:
                continue
            components = manager.get_cost_components()
            enabled = {key: bool(value.enabled) for key, value in components.items()}
            if enabled.get("scene_collision", False) or (not self_collision and enabled.get("self_collision", False)):
                raise ValueError("The constructed native managers failed diagnostic collision isolation")
            managers.append({"rollout_index": index, "rollout_name": rollout.rollout_instance_name,
                             "manager": name, "class": type(manager).__name__, "components_enabled": enabled})
    constraints = solver.metrics_rollout.metrics_constraint_manager
    cspace = constraints.get_cost("cspace")
    native_self = constraints.get_cost("self_collision")
    if cspace is None or not cspace.enabled:
        raise ValueError("Both diagnostic modes require native joint-position limits in final metrics")
    if self_collision and (native_self is None or not native_self.enabled):
        raise ValueError("The collision mode must retain the original native self checker")
    if solver.scene_collision_checker is not None:
        raise ValueError("Native isolated IK must not silently acquire a world collision checker")
    return managers, cspace.config.joint_limits


def cpu_array(tensor):
    return tensor.detach().cpu().numpy().copy()


def model_evidence(solver, robot, output):
    """Record actual loaded radii and padding in native sphere ordering."""
    parameters = solver.kinematics.config.kinematics_config
    spheres = cpu_array(parameters.link_spheres).reshape(-1, 4)
    collision = solver.kinematics.get_self_collision_config()
    pairs, padding = cpu_array(collision.collision_pairs), cpu_array(collision.sphere_padding)
    config = robot["robot_cfg"]["kinematics"]
    source = [(link, row) for link in config["collision_link_names"] for row in config["collision_spheres"][link]]
    radii = np.asarray([row["radius"] for _, row in source], dtype=spheres.dtype)
    if spheres.shape != (len(source), 4) or not np.isfinite(spheres).all() or not np.isfinite(padding).all():
        raise ValueError("Native sphere/padding layout must match the actual bounded robot configuration")
    difference = spheres[:, 3]-radii
    np.savez_compressed(output/"native-collision-model.npz", native_link_spheres=spheres,
                        input_radii_native_dtype=radii, radius_difference=difference,
                        sphere_padding=padding, collision_pairs=pairs,
                        source_sphere_links=np.asarray([name for name, _ in source]))
    return {"sphere_count": len(spheres), "enabled_pair_count": len(pairs),
            "sphere_padding_shape": list(padding.shape), "sphere_padding_min_m": float(padding.min()),
            "sphere_padding_max_m": float(padding.max()), "sphere_padding_nonzero_count": int(np.count_nonzero(padding)),
            "radius_difference_min_m": float(difference.min()), "radius_difference_max_m": float(difference.max()),
            "radius_difference_nonzero_count": int(np.count_nonzero(difference)),
            "radius_comparison": "Input radii cast once to native dtype before subtraction; no tolerance hides added padding",
            "artifact": str((output/"native-collision-model.npz").resolve())}


def native_fk(kinematics, problem, values, names, device):
    """Evaluate each returned q as its own one-state batch in calibrated frames."""
    from curobo.types import JointState

    query = device.to_device(values)[:, None].contiguous()
    state = kinematics.compute_kinematics(JointState.from_position(query, joint_names=list(names)))
    tool = state.tool_poses
    index = tool.tool_frames.index(problem.tool_link)
    position, quaternion = problem.transform_tool_pose(tool.position[:, :, index], tool.quaternion[:, :, index])
    poses = np.tile(np.eye(4), (len(values), 1, 1))
    poses[:, :3, 3] = cpu_array(position[:, 0])
    poses[:, :3, :3] = Rotation.from_quat(cpu_array(quaternion[:, 0])[:, [1, 2, 3, 0]]).as_matrix()
    return query, state, position, quaternion, poses


def full_scene_check(solver, kinematics, problem, scene, values, names, progress, device):
    """Inspect identical returned q under every unchanged native/scene term."""
    import torch

    with torch.no_grad():
        query, state, position, quaternion, poses = native_fk(kinematics, problem, values, names, device)
        native_self = solver.metrics_rollout.metrics_constraint_manager.get_cost("self_collision")
        if native_self is None or not native_self.enabled:
            raise ValueError("Postchecks require the enabled original native self checker")
        native_self.setup_batch_tensors(len(values), 1)
        terms = {"native:self_collision": native_self.forward(state.robot_spheres)}
        terms.update({"scene:"+name: value for name, value in scene.components(
            problem.transform_robot_spheres(state.robot_spheres), position, quaternion,
            query.new_full(query.shape[:2], float(progress)), query, names).items()})
        residuals = {}
        for name, value in terms.items():
            if value.shape[0] != len(values):
                raise ValueError("Each scene component must preserve the independent configuration batch")
            residuals[name] = cpu_array(value).reshape(len(values), -1).max(axis=1)
        records = []
        for row in range(len(values)):
            nonfinite = [name for name, value in residuals.items() if not np.isfinite(value[row])]
            failed = [name for name, value in residuals.items() if not np.isfinite(value[row]) or value[row] > 0]
            records.append({"all_original_collision_terms_zero": not failed, "failed_components": failed,
                            "nonfinite_components": nonfinite,
                            "residuals": {name: None if name in nonfinite else float(value[row])
                                          for name, value in residuals.items()}})
    return records, poses


def probe(args, prepared, targets, report):
    """Run at most six small IK calls, then inspect their unchanged outputs."""
    import torch
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.kinematics import Kinematics
    from curobo.types import DeviceCfg, GoalToolPose, JointState
    from manipulation.planning import (
        _bounded_robot_configuration, _require_planning_memory,
        _require_planning_host_memory, _PlanningWorkspace,
    )

    problem, scene, budget = prepared["problem"], prepared["scene"], prepared["budget"]
    robot, metadata = _bounded_robot_configuration(prepared["robot"], problem, scene, budget)
    # BEGIN the standalone single-state query has the same CPU/GPU ownership.
    allocation = scene.estimated_workspace_allocation(args.ik_seeds, 1, robot_coverage_horizon=1)
    workspace = _PlanningWorkspace(allocation["cpu_shared"], allocation["cpu_live"],
        sum(allocation[name] for name in ("gpu_retained", "gpu_live", "gpu_shared")), 0, 0, False)
    if workspace.gpu_bytes > args.scene_gpu_workspace_mib*1024**2:
        raise ValueError("Actual single-state GPU scene queries exceed the declared workspace budget")
    _require_planning_host_memory(workspace, max_host_rss_bytes=budget.max_host_rss_bytes)
    # END source geometry is never passed to the CUDA preflight as device bytes.
    device = DeviceCfg("cuda:0")
    report["resources"].update(estimated_scene_query_bytes=workspace.total_bytes,
        estimated_cpu_scene_query_bytes=workspace.cpu_bytes, estimated_gpu_scene_query_bytes=workspace.gpu_bytes)
    report["robot_metadata"] = metadata
    report["target_frame_count"] = len(targets)
    report["maximum_native_ik_calls"] = len(targets)*len(MODES)
    report["modes"] = {}
    common_seeds = None
    for mode, self_collision in MODES:
        solver = kinematics = None
        try:
            _require_planning_memory(torch, device.device, workspace,
                                     max_device_used_bytes=budget.max_device_used_bytes)
            optimizer, metrics = isolated_configs(self_collision)
            config = InverseKinematicsCfg.create(
                robot=deepcopy(robot), optimizer_configs=[optimizer], metrics_rollout=metrics,
                self_collision_check=self_collision, device_cfg=device,
                num_seeds=args.ik_seeds, seed_solver_num_seeds=args.ik_seeds,
                max_batch_size=1, max_goalset=1, use_cuda_graph=False, random_seed=123,
                position_tolerance=problem.numerical.position_m,
                orientation_tolerance=problem.numerical.rotation_rad,
                success_requires_convergence=True, optimization_dt=None,
                override_optimizer_num_iters={"lbfgs": LBFGS_ITERATIONS})
            config.exit_early = False
            solver = InverseKinematics(config)
            names = tuple(solver.joint_names)
            if len(solver.tool_frames) != 1 or set(names) != set(problem.joint_names):
                raise ValueError("This bounded probe requires one actual TCP and the exact active robot joints")
            seed_config = solver.seed_ik_solver.config
            if seed_config.num_seeds != args.ik_seeds or seed_config.max_iterations != LM_ITERATIONS:
                raise ValueError("Native LM allocation/iterations differ from the explicitly bounded probe")
            managers, limits = manager_evidence(solver, self_collision)
            limit_columns = [limits.joint_names.index(name) for name in names]
            bounds = cpu_array(limits.position)[:, limit_columns]
            start_columns = [problem.joint_names.index(name) for name in names]
            start_values = problem.start_q[start_columns]
            current = JointState.from_numpy(list(names), start_values[None], device_cfg=device)
            if common_seeds is None:
                common_seeds = cpu_array(solver.seed_ik_solver._generate_seed_configs(
                    batch_size=1, seed_config=current.position[:, None].contiguous()))
                if common_seeds.shape != (1, args.ik_seeds, len(names)) or not np.array_equal(
                        common_seeds[0, 0], cpu_array(current.position[0])):
                    raise ValueError("Native seed pool must preserve q0 as its first unchanged warm start")
                np.savez_compressed(args.output/"common-seeds.npz", joint_names=np.asarray(names),
                                    seeds=common_seeds, measured_q0=start_values)
                report["joint_names"] = list(names)
                report["seed_policy"] = {"source": "One q0 warm start plus native bounded Halton seeds, identical for every target and both modes",
                    "native_halton_sampler_seed": seed_config.sampler_seed, "main_random_seed": config.random_seed,
                    "seed_pool": common_seeds, "independent_branches_not_asserted": True}
            elif names != tuple(report["joint_names"]):
                raise ValueError("Paired native modes changed their named joint layout")
            if self_collision:
                report["native_collision_model"] = model_evidence(solver, robot, args.output)
            kinematics = Kinematics(solver.kinematics.config, compute_jacobian=False)
            mode_report = {"self_collision_requested": self_collision, "actual_managers": managers,
                "effective_position_limits": {"joint_names": list(names), "lower": bounds[0], "upper": bounds[1]},
                "position_tolerance_m": problem.numerical.position_m,
                "rotation_tolerance_rad": problem.numerical.rotation_rad,
                "optimization_dt": None, "exit_early": False,
                "lm_iterations": seed_config.max_iterations, "lbfgs_iterations": LBFGS_ITERATIONS,
                "native_self_model": "Measured fingers remain locked; the original binary-hand envelope is a separate postcheck",
                "targets": []}
            report["modes"][mode] = mode_report
            for target in targets:
                target_pose = target["T_reference_tool"]
                link_pose = np.linalg.inv(problem.base_pose) @ target_pose @ np.linalg.inv(problem.tool_transform)
                quaternion = Rotation.from_matrix(link_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
                goal = GoalToolPose(tool_frames=[problem.tool_link],
                    position=device.to_device(link_pose[None, None, None, None, :3, 3]).contiguous(),
                    quaternion=device.to_device(quaternion[None, None, None, None]).contiguous())
                report["stage"] = f"{mode}:source_frame_{target['source_frame']}"
                write_report(args.output/"probe-report.json", report)
                result = solver.solve_pose(goal, current_state=current,
                    seed_config=device.to_device(common_seeds.copy()).contiguous(), return_seeds=args.ik_seeds)
                values = cpu_array(solver.get_active_js(result.js_solution).position[0])
                if values.shape != (args.ik_seeds, len(names)) or not np.isfinite(values).all():
                    raise ValueError("Native IK must return exactly its bounded finite configuration set")
                with torch.no_grad():
                    _, _, _, _, poses = native_fk(kinematics, problem, values, names, device)
                position_error = np.linalg.norm(poses[:, :3, 3]-target_pose[:3, 3], axis=1)
                rotation_error = Rotation.from_matrix(target_pose[:3, :3].T @ poses[:, :3, :3]).magnitude()
                violation = np.maximum(bounds[0]-values, 0)+np.maximum(values-bounds[1], 0)
                maximum_violation = violation.max(axis=1)
                pose_limits = ((position_error <= problem.numerical.position_m)
                               & (rotation_error <= problem.numerical.rotation_rad) & (maximum_violation == 0))
                success, feasible = cpu_array(result.success[0]), cpu_array(result.feasible[0])
                artifact = args.output/f"{mode}-frame-{target['source_frame']}.npz"
                np.savez_compressed(artifact, joint_names=np.asarray(names), joint_position=values,
                    actual_tool_pose=poses, target_tool_pose=target_pose, target_native_link_pose=link_pose,
                    native_success=success, native_feasible=feasible, position_error_m=position_error,
                    rotation_error_rad=rotation_error, joint_position_limit_violation=violation,
                    input_seeds=common_seeds, progress=np.asarray(target["progress"]))
                mode_report["targets"].append({**target, "artifact": str(artifact.resolve()),
                    "native_success_count": int(success.sum()), "native_feasible_count": int(feasible.sum()),
                    "independent_fk_pose_and_limits_count": int(pose_limits.sum()),
                    "solutions": [{"solution_rank": row, "joint_position": values[row],
                        "native_success": bool(success[row]), "native_feasible": bool(feasible[row]),
                        "native_position_error": cpu_array(result.position_error[0])[row],
                        "native_rotation_error": cpu_array(result.rotation_error[0])[row],
                        "fk_position_error_m": float(position_error[row]), "fk_rotation_error_rad": float(rotation_error[row]),
                        "joint_limit_violation": violation[row], "maximum_joint_limit_violation": float(maximum_violation[row]),
                        "fk_pose_and_limits_satisfied": bool(pose_limits[row])} for row in range(len(values))]})
                report["completed_native_ik_calls"] += 1
                write_report(args.output/"probe-report.json", report)
                del result, goal
            if self_collision:
                # Use this retained enabled native checker for both modes'
                # exact saved q, even when the generating mode disabled it.
                report["stage"] = "original_scene_and_native_self_postchecks"
                for checked_mode in report["modes"].values():
                    for target in checked_mode["targets"]:
                        values = np.asarray([solution["joint_position"] for solution in target["solutions"]])
                        checks, _ = full_scene_check(solver, kinematics, problem, scene, values,
                                                     names, target["progress"], device)
                        for solution, check in zip(target["solutions"], checks):
                            solution["original_collision_postcheck"] = check
                        target["all_original_collision_terms_zero_count"] = sum(
                            check["all_original_collision_terms_zero"] for check in checks)
                initial_checks, initial_poses = full_scene_check(solver, kinematics, problem, scene,
                    start_values[None], names, 0., device)
                report["measured_q0_collision_postcheck"] = initial_checks[0]
                report["measured_q0_fk_pose"] = initial_poses[0]
                write_report(args.output/"probe-report.json", report)
        finally:
            if solver is not None:
                solver.destroy()
            solver = kinematics = None
            gc.collect()
            torch.cuda.empty_cache()


def run(args):
    """Own the small diagnostic artifact directory and resource accounting."""
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "preparing", "segment_id": args.segment, "candidate_id": args.candidate,
        "planning_called": False, "trajectory_planning_called": False, "simulation_called": False,
        "completed_native_ik_calls": 0,
        "resources": {"cpu_threads": 2, "torch_allocator_bytes": args.torch_memory_mib*1024**2,
                      "mapping_bytes": 1024**3, "scene_gpu_workspace_ceiling_bytes": args.scene_gpu_workspace_mib*1024**2,
                      "native_free_memory_allowance_bytes": 2*1024**3,
                      "maximum_seeds": args.ik_seeds, "maximum_batch_size": 1, "maximum_goalset": 1},
        "interpretation": ["Pose-only success provides finite kinematic evidence and does not certify execution",
            "Failure with a bounded seed pool does not prove target unreachability",
            "Native-self failure may come from the shared robot model and does not isolate a target-pose cause",
            "Returned solution ranks are not asserted to be unique branches or original seed indices",
            "Original scene components retain all robot, gripper, owner, background and unknown-space policies",
            "Native collision residuals retain native cost units; they are not relabelled as linear penetration metres",
            "Single-state IK does not validate paths, dynamic limits, controller tracking or physical grasp success"]}
    started = time.monotonic()
    try:
        _, targets, provenance = target_inputs(args)
        report.update(provenance=provenance, targets=targets)
        write_report(args.output/"probe-report.json", report)
        prepared = observed_scene(args)
        report.update(planning_boundary_intercepted=True,
                      scene_build_report=str((args.output/"scene-build"/"scene-build-report.json").resolve()))
        probe(args, prepared, targets, report)
        report.update(status="completed", stage="complete")
    except KeyboardInterrupt:
        report.update(status="interrupted", error="Explicit operator interrupt")
        raise
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (args.output/"exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        # CPU preparation/help must not create or query a CUDA context.
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            torch = sys.modules["torch"]
            report["resources"]["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["resources"]["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_report(args.output/"probe-report.json", report)
    print(json.dumps({"status": report["status"], "completed_native_ik_calls": report["completed_native_ik_calls"],
                      "report": str((args.output/"probe-report.json").resolve())}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "robot-fixture", "feedback", "execution-limits", "finger-targets", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--target-frame", type=int, action="append", required=True,
                        help="One original source frame per target; repeat for at most three targets")
    parser.add_argument("--owner-mask", nargs=3, action="append", required=True,
                        metavar=("OWNER_ID", "VIEW_ID", "MASK_PATH"))
    parser.add_argument("--unknown-policy", type=Path, required=True)
    parser.add_argument("--fine-regions", type=Path)
    parser.add_argument("--contact-bindings", type=Path)
    parser.add_argument("--mapping-memory-mib", type=int, default=1024)
    parser.add_argument("--owner-geometry", type=Path)
    parser.add_argument("--center", type=float, nargs=3, required=True)
    parser.add_argument("--extent", type=float, nargs=3, required=True)
    # BEGIN source-backed probes share the production environment default.
    from manipulation.planning_observation import DEFAULT_ENVIRONMENT_VOXEL_SIZE_M
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_ENVIRONMENT_VOXEL_SIZE_M)
    # END task-owner precision belongs to the common library, not this probe.
    parser.add_argument("--depth-error", type=float, required=True)
    # BEGIN same three-state observation contract as the production runner.
    parser.add_argument("--self-filter-padding-offset-m", type=float, required=True)
    parser.add_argument("--self-filter-padding-scale", type=float, required=True)
    parser.add_argument("--self-filter-source", required=True)
    # END the probe declares its geometry uncertainty explicitly.
    parser.add_argument("--ik-seeds", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--scene-gpu-workspace-mib", type=int, choices=(1024, 2048), default=1024)
    # BEGIN request-level physical limits remain explicit in standalone runs.
    # BEGIN expose the same explicit total cap as the production runner.
    parser.add_argument("--max-host-rss-mib", type=int, choices=(8192, 10240, 12288), default=8192)
    # END the unchanged default still requires an explicit larger allowance.
    parser.add_argument("--max-device-used-mib", type=int, choices=(6144, 8192), default=6144)
    # END defaults preserve the original half-host/half-device limits.
    parser.add_argument("--torch-memory-mib", type=int, choices=(2048, 4096), default=2048)
    args = parser.parse_args()
    if (len(args.target_frame) > 3 or len(set(args.target_frame)) != len(args.target_frame)
            or any(frame < 0 for frame in args.target_frame)):
        parser.error("Use one to three distinct nonnegative source target frames and an explicit pre-grasp frame")
    return args


if __name__ == "__main__":
    run(parse_args())
# END isolated, bounded goal diagnostics using the existing recorded scene.
