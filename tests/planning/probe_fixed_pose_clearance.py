"""Compare native IK solutions at fixed archived or functional tool poses.

The recorded runner owns scene preparation. This bounded diagnostic intercepts
its planning boundary, uses the unchanged native model and source acceptance,
and never solves or accepts a trajectory. CLI identities select every target;
there is no object-, robot-, or skill-specific branch in this probe.
"""

# BEGIN fixed-pose redundancy evidence with unchanged production acceptance.
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import runpy
import signal
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_constraint_generation import cpu, read_json, require_equal, write_report
from probe_local_approach_states import configuration_statistics


def tool_state(planner, q):
    """Compute the actual calibrated TCP and sphere geometry at one named q."""
    joints = planner.JointState(position=q, joint_names=list(planner.joint_names))
    state = planner._verification_fk.compute_kinematics(joints)
    tool = state.tool_poses
    index = tool.tool_frames.index(planner.problem.tool_link)
    # Native layouts and the quaternion VJP retain their production semantics.
    p = planner._native_layout.apply(tool.position)[:, :, index]
    r = planner._orientation_vjp.apply(q, tool.quaternion.detach(),
                                     state.tool_jacobians[..., 3:, :].detach())[:, :, index]
    p, r = planner.problem.transform_tool_pose(p, r)
    spheres = planner.problem.transform_robot_spheres(planner._native_layout.apply(state.robot_spheres))
    return p, r, spheres


def pose_matrix(position, quaternion):
    pose = np.eye(4)
    pose[:3, 3] = cpu(position)[0, 0]
    pose[:3, :3] = Rotation.from_quat(cpu(quaternion)[0, 0, [1, 2, 3, 0]]).as_matrix()
    return pose


def scene_distances(world, scene):
    """Retain source distances without calling a surface distance a solid proof.

    Positive surface distances are useful clearance measurements only together
    with completed source signed-piece and containment checks. Penetration
    values describe tested components, not global penetration depth of a union.
    The minimum includes permitted contacts; their identity remains visible.
    Native installation distances use the required/excluded/mechanical policy.
    """
    values, prohibited = [], []
    active_permissions = set(world["active_robot_owner_permissions"])
    whole_pair_permissions = {(name, rule.target_owner) for rule in scene.robot_owner_permissions
        if rule.rule_id in active_permissions for name in rule.native_geometry_names}
    for row in world["pairs"]:
        if row.get("actor") != "robot":
            continue
        native = row.get("native_pair_evidence")
        if native is not None:
            if native["policy"] != "required":
                continue
            distance = native.get("signed_distance_m")
        else:
            distance = row.get("signed_distance_m")
        if distance is not None:
            values.append((float(distance), row))
            if not row.get("permission") and (row["actor_geometry"], row["target_owner"]) not in whole_pair_permissions:
                prohibited.append((float(distance), row))
    def minimum(items):
        if not items:
            return None
        value, pair = min(items, key=lambda item: item[0])
        return {"distance_m": value, "pair": pair, "recorded_distance_count": len(items)}
    return {"minimum_recorded_all_pairs": minimum(values),
        "minimum_recorded_unpermitted_pairs": minimum(prohibited),
        "finite_patch_surface_distance_ambiguity": bool(scene.contacts),
        "scope": "Recorded modeled-entity distances; failed pairs can stop signed-piece enumeration early; permitted contacts excluded from second statistic"}


def check_state(planner, target, values, bounds, baseline):
    """Check static conditions and record the same state's proxy gradient."""
    from manipulation.planning import _scene_boundary_evaluate
    from manipulation.voxel_layers import LayeredVoxelQuery
    torch, scene, problem = planner.torch, planner._verification_scene, planner.problem
    names = list(planner.joint_names)
    q = planner.device_cfg.to_device(values[None, None]).contiguous().requires_grad_(True)
    p, r, spheres = tool_state(planner, q)
    s, t, u = (q.new_full((1, 1), target[key]) for key in ("progress", "time_s", "u"))
    captured = []
    original_query = LayeredVoxelQuery.__call__
    def capture_query(query, query_spheres, *, key, selection):
        value = original_query(query, query_spheres, key=key, selection=selection)
        if query is scene.background and key == ("robot", "background"):
            captured.append((value, selection))
        return value
    with patch.object(LayeredVoxelQuery, "__call__", new=capture_query):
        proxy = _scene_boundary_evaluate(scene, problem, planner.progress_map, spheres, p, r, s, q, names,
            trajectory_time=t, normalized_time=u)
    raw = float(proxy.item())
    self_guidance = scene.robot_self_collision_guidance(spheres, p, r, s, q, names, trajectory_time=t)
    combined = proxy+self_guidance
    gradient = torch.autograd.grad(proxy.square().sum()*planner.budget.constraint_weight, q, retain_graph=True)[0]
    gradient = cpu(gradient)[0, 0]
    combined_gradient = torch.autograd.grad(combined.square().sum()*planner.budget.constraint_weight, q)[0]
    combined_gradient = cpu(combined_gradient)[0, 0]
    combined_raw = float(combined.item())
    background_bodies = []
    for value, selection in captured:
        distance_values, selected_fine = cpu(value), cpu(selection.sphere_fine)
        for body in scene.robot_collision_bodies:
            indices = body["sphere_indices"]
            body_values = distance_values[..., indices]
            background_bodies.append({"body_id": body["body_id"], "geometry_names": body["geometry_names"],
                "sphere_indices": indices, "selected_fine": selected_fine[..., indices].tolist(),
                "per_sphere_raw_penalty_m": body_values.tolist(),
                "maximum_raw_penalty_m": float(body_values.max()),
                "positive_sphere_count_per_mode": (body_values > 0).sum(-1).tolist()})
    # Run a fresh forward after backward; native workspaces cannot be reused
    # before their autograd computation finishes.
    with torch.no_grad():
        q = q.detach()
        p, r, spheres = tool_state(planner, q)
        actual = pose_matrix(p, r)
        diagnostics, terms, self_diagnostics = {}, {}, {}
        _scene_boundary_evaluate(scene, problem, planner.progress_map, spheres, p, r, s, q, names,
            trajectory_time=t, normalized_time=u, source_geometry=True,
            diagnostics=diagnostics, source_components_out=terms)
        self_value = scene.check_robot_self_collision(q, names, t, diagnostics=self_diagnostics)
        task = problem.residuals(p, r, s, q, names, trajectory_time=t, direction_mode="point")
        task_values = {key: float(value.max().item()) for key, value in task.items()}
        term_values = {key: float(value.max().item()) for key, value in terms.items()}
        coverage = scene.check_robot_coverage(q, names, t)
    excess = np.maximum(bounds[0]-values, 0)+np.maximum(values-bounds[1], 0)
    position_error = float(np.linalg.norm(actual[:3, 3]-target["pose"][:3, 3]))
    rotation_error = float(Rotation.from_matrix(target["pose"][:3, :3].T@actual[:3, :3]).magnitude())
    pose_pass = (position_error <= problem.numerical.position_m
                 and rotation_error <= problem.numerical.rotation_rad)
    world = diagnostics["worst_state"]["scene"]
    # Unknown terms remain separate from formal known-scene acceptance, as in
    # production. Exact coverage is recorded instead of interpreting a sphere
    # guidance residual as a source-solid unknown-space verdict.
    known = {key: value for key, value in term_values.items()
             if not key.startswith("unknown/") or key.startswith("unknown/release_anchor/")}
    gates = {"fixed_target_pose": bool(pose_pass), "joint_position_limits": not bool(excess.any()),
        "source_self_collision": bool(self_value.max().item() == 0),
        "source_known_scene": all(value == 0 for value in known.values()),
        "point_task_constraints": all(value == 0 for value in task_values.values()),
        "robot_coverage_attempt_allowed": bool(coverage) and all(row["attempt_allowed"] for row in coverage)}
    return {"joint_position": values.tolist(), "actual_tool_pose": actual.tolist(),
        "fixed_position_error_m": position_error, "fixed_rotation_error_rad": rotation_error,
        "joint_limit_margins_rad": np.minimum(values-bounds[0], bounds[1]-values).tolist(),
        "joint_limit_margin_rad": float(np.minimum(values-bounds[0], bounds[1]-values).min()),
        "joint_limit_margin_fraction": float((np.minimum(values-bounds[0], bounds[1]-values)
                                              /(bounds[1]-bounds[0])).min()),
        "distance_to_measured_q0_rad": float(np.linalg.norm(values-cpu(planner.start.position)[0])),
        "distance_to_archived_q_rad": float(np.linalg.norm(values-baseline)),
        "source_environment_distances": scene_distances(world, scene),
        "source_scene": world, "source_self": self_diagnostics,
        "unmodelled_geometry_names": diagnostics["unmodelled_geometry_names"],
        "source_components": term_values, "point_task_residuals": task_values,
        "proxy_background_body_values": background_bodies,
        "robot_coverage": coverage, "static_gates": gates, "modeled_static_gates_passed": all(gates.values()),
        "full_robot_coverage_certified": bool(coverage) and all(row["passed"] for row in coverage),
        "proxy_scene_raw": raw, "proxy_scene_squared_weighted": raw*raw*planner.budget.constraint_weight,
        "proxy_scene_squared_gradient_q": gradient.tolist(),
        "proxy_scene_squared_gradient_norm": float(np.linalg.norm(gradient)),
        "proxy_scene_plus_self_raw": combined_raw,
        "proxy_scene_plus_self_squared_gradient_q": combined_gradient.tolist(),
        "proxy_scene_plus_self_squared_gradient_norm": float(np.linalg.norm(combined_gradient))}


def diagnose(problem, scene, robot, robot_validator, budget, args, report, save):
    """Keep one planner allocation and run small, fully evaluated IK batches."""
    from manipulation.planning import _CuroboPlanner, _torch_interpolate
    if any(event.stop for event in problem.events) or problem.mechanisms:
        raise ValueError("This static probe cannot certify required velocity or history-dependent mechanisms")
    original = read_json(args.source_run/"planning"/"planning-report.json")
    if asdict(budget) != original["budget"]:
        raise ValueError("The original recorded planner budget must remain unchanged")
    checks = report["input_identity"]
    for filename in ("observed-free.npz", "entrance-scene.npz"):
        with np.load(args.source_run/filename, allow_pickle=False) as old:
            with np.load(args.output/filename, allow_pickle=False) as new:
                if old.files != new.files:
                    raise ValueError("Scene archive fields differ")
                for key in old.files:
                    require_equal(filename+":"+key, new[key], old[key], checks)
    planner = _CuroboPlanner(problem, scene, robot, budget, "cuda:0", robot_validator=robot_validator)
    try:
        with np.load(args.source_run/"planning"/"optimizer-input-00.npz", allow_pickle=False) as saved:
            for key, value in (("joint_names", planner.joint_names), ("start_q", cpu(planner.start.position)),
                               ("start_dq", cpu(planner.start.velocity)),
                               ("progress_u", planner.progress_map.u), ("progress_s", planner.progress_map.s)):
                require_equal(key, value, saved[key], checks)
        config = read_json(args.source_run/"planning"/"effective-config-00.json")
        if planner.formal_clock != config["formal_clock"]:
            raise ValueError("Fixed diagnostic clock differs from archived clock")
        checks["formal_clock_identical"] = True
        meta = read_json(args.source_run/"planning"/"native-result-00.json")
        with np.load(args.source_run/"planning"/"native-result-00.npz", allow_pickle=False) as saved:
            columns = [meta["states"]["interpolated_trajectory"]["joint_names"].index(name)
                       for name in planner.joint_names]
            archived = saved["interpolated_trajectory.position"][0, 0, :, columns].T.copy()
        if not 0 <= args.sample_index < len(archived) or any(
                not 0 <= index < len(problem.pose_events) for index in args.pose_event):
            raise ValueError("Diagnostic sample and event indices must be nonnegative and in range")
        baseline = archived[args.sample_index]
        q = planner.device_cfg.to_device(baseline[None, None])
        with planner.torch.no_grad():
            p, r, _ = tool_state(planner, q)
            target_pose = pose_matrix(p, r)
        # Use the native float32 normalized grid, not a recomputed decimal time.
        u = q.new_tensor(args.sample_index/(len(archived)-1))
        progress = _torch_interpolate(planner.device_cfg.to_device(planner.progress_map.u),
            planner.device_cfg.to_device(planner.progress_map.s), u.reshape(1, 1))
        targets = [{"label": "archived_sample", "pose": target_pose,
            "progress": float(progress.item()), "u": float(u.item()),
            "time_s": float((u*planner.formal_clock["duration"]).item())}]
        for index in args.pose_event:
            goal = problem.pose_events[index]
            u_goal, s_goal = planner.progress_map.anchor_parameters([goal.progress])[0]
            native_u, native_s = q.new_tensor(u_goal), q.new_tensor(s_goal)
            targets.append({"label": f"pose_event_{index}",
                "pose": goal.target@np.linalg.inv(goal.tool_from_function),
                "u": float(native_u.item()), "progress": float(native_s.item()),
                "time_s": float((native_u*planner.formal_clock["duration"]).item())})
        limits = planner._verification_native.get_cost("cspace").config.joint_limits
        bounds = cpu(limits.position)[:, [limits.joint_names.index(name) for name in planner.joint_names]]
        if args.objective_only:
            from probe_recorded_objective import replay_objective
            replay_objective(planner, args.source_run, args.output, report, save)
            report["status"] = "completed"
            return
        # Archived and measured warm starts; all other initializers come from the
        # existing persistent native sampler. Save and reuse the pool unchanged.
        pool = []
        for batch in range(args.batches):
            warm = planner.torch.cat((q, planner.start.position[:, None]), dim=1)
            seeds = planner.ik.prepare_action_seeds(1, budget.ik_seeds,
                seed_config=warm[:, :budget.ik_seeds] if batch == 0 else None)
            pool.append(cpu(seeds).reshape(1, budget.ik_seeds, -1))
        pool = np.concatenate(pool, axis=1)
        np.savez_compressed(args.output/"initial-seeds.npz", joint_names=np.asarray(planner.joint_names),
                            seeds=pool, archived_q=baseline, measured_q0=cpu(planner.start.position))
        planner.ik.config.exit_early = False
        report.update(formal_clock=planner.formal_clock, original_budget=asdict(budget),
            archived_baseline_origin={"sample_index": args.sample_index,
                "time_s": targets[0]["time_s"], "scope": "Same archived q reused in both target conditions; not the original q at other event times"},
            numerical_tolerances=asdict(problem.numerical), joint_names=list(planner.joint_names),
            joint_position_bounds=bounds.tolist(),
            initial_seed_statistics=configuration_statistics(pool[0], problem.numerical.joint),
            diagnostic_exit_early=False, targets=[])
        save("initialized")
        for target in targets:
            group = {**{key: value for key, value in target.items() if key != "pose"},
                "fixed_target_pose": target["pose"].tolist(), "solutions": []}
            report["targets"].append(group)
            planner.seed_parameter.fill_(target["u"])
            lp = np.linalg.inv(problem.base_pose)@target["pose"]@np.linalg.inv(problem.tool_transform)
            rotation = Rotation.from_matrix(lp[:3, :3]).as_quat()[[3, 0, 1, 2]]
            goal = planner.GoalToolPose(tool_frames=[problem.tool_link],
                position=planner.device_cfg.to_device(lp[None, None, None, None, :3, 3]).contiguous(),
                quaternion=planner.device_cfg.to_device(rotation[None, None, None, None]).contiguous())
            group["archived_q_at_this_target"] = check_state(planner, target, baseline, bounds, baseline)
            save(target["label"]+":baseline")
            for batch in range(args.batches):
                seeds = pool[:, batch*budget.ik_seeds:(batch+1)*budget.ik_seeds].copy()
                result = planner.ik.solve_pose(goal, current_state=planner.start,
                    seed_config=planner.device_cfg.to_device(seeds), return_seeds=budget.ik_seeds)
                values = cpu(planner.ik.get_active_js(result.js_solution).position)[0]
                success = cpu(result.success)[0]
                for rank, value in enumerate(values):
                    evidence = check_state(planner, target, value, bounds, baseline)
                    group["solutions"].append({"batch": batch, "return_rank": rank,
                        "native_ik_success": bool(success[rank]), **evidence})
                save(target["label"]+f":batch_{batch}")
            group["solution_statistics"] = configuration_statistics(
                np.asarray([row["joint_position"] for row in group["solutions"]]), problem.numerical.joint)
        report["status"] = "completed"
    finally:
        planner.close()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--pose-event", type=int, action="append", default=[])
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--diagnostic-seconds", type=int, default=1200)
    parser.add_argument("--objective-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args, runner_arguments = parser.parse_known_args()
    if args.output.exists() or not 1 <= args.batches <= 16:
        raise ValueError("Use a fresh output and at most 16 sequential IK batches")
    report = {"schema": "functo.fixed_pose_clearance.1", "status": "preparing",
        "source_run": str(args.source_run.resolve()),
        "scope": "static modeled-geometry diagnostic, not an executable trajectory",
        "input_identity": {}, "trajectory_optimizer_called": False, "ompl_called": False, "physics_steps": 0}
    start = time.monotonic()
    def save(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic()-start)
        write_report(args.output/"clearance-report.json", report)
        print(json.dumps({"stage": stage, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    def timeout(*_):
        raise TimeoutError("Separate fixed-pose diagnostic budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.diagnostic_seconds)
    def intercept(problem, scene, robot, output, *, robot_validator, budget, provenance, execution_limits,
                  continuation=None, rebind=None, request_ledger=None):
        if any(value is not None for value in (continuation, rebind, request_ledger)):
            raise ValueError("A fixed-input diagnostic cannot continue or reset a planning request")
        diagnose(problem, scene, robot, robot_validator, budget, args, report, save)
        return SimpleNamespace(success=False, diagnostics={"initial_state": None})
    runner = Path(__file__).with_name("run_recorded_skill.py")
    sys.argv = [str(runner), *runner_arguments, "--output", str(args.output)]
    try:
        with patch("recorded_task_planning.plan_and_record", new=intercept):
            runpy.run_path(str(runner), run_name="__main__")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        if args.output.exists():
            report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            save(report["status"])
            source = args.output/"run-report.json"
            if source.exists():
                preparation = read_json(source)
                preparation.update(planning_called=False, fixed_pose_diagnostic_called=True, execution_tested=False)
                preparation.pop("planning_success", None)
                write_report(args.output/"scene-preparation-report.json", preparation)
                source.unlink()


if __name__ == "__main__":
    main()
# END bounded IK evidence never acquires trajectory acceptance authority.
