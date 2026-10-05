"""Evaluate the current native objective at archived controls without solving.

Called by probe_fixed_pose_clearance after unchanged scene/clock/input checks.
The current planner phase is recorded explicitly: archived controls do not
restore an obsolete cost configuration. The native goal lifecycle uses one
evaluation particle; line-search replication and optimizer iterations are
unnecessary for a VJP.
"""

# BEGIN bounded whole-curve objective evidence; no controls are accepted.
import numpy as np

from probe_constraint_generation import cpu, read_json, terms, gradient_summary


def replay_objective(planner, source_run, output, report, save):
    """Compare task, collision and smoothness values and control gradients."""
    from curobo._src.solver.solve_mode import SolveMode
    from curobo._src.solver.solve_state import SolveState

    torch, solver = planner.torch, planner.trajopt
    source = source_run/"planning"
    with np.load(source/"optimizer-input-00.npz", allow_pickle=False) as saved:
        seed = planner.device_cfg.to_device(saved["seed_controls"].copy())
    with np.load(source/"native-result-00.npz", allow_pickle=False) as saved:
        native = {name: saved[name].copy() for name in saved.files}
    meta = read_json(source/"native-result-00.json")
    # BEGIN match production: typed process goals own both trajectory phases.
    solver.disable_tool_pose_tracking()
    goal_pose = planner._goal(planner.problem.phase_count)
    # END native goal metadata supplies storage only.
    state = SolveState(solve_type=SolveMode.SINGLE, num_trajopt_seeds=1, batch_size=1,
        num_envs=1, num_goalset=goal_pose.num_goalset, tool_frames=goal_pose.tool_frames)
    seed_goal = planner.JointState.from_position(seed[..., -1, :].clone())
    seed_goal.dt = seed.new_full((1, 1), planner.budget.maximum_trajectory_dt)
    goal, _ = solver._prepare_goal_buffer(state, goal_pose, planner.start,
        use_implicit_goal=False, seed_goal_state=seed_goal, goal_state=None)
    rollout = solver.optimizer_rollouts[0]
    rollout.update_params(goal, num_particles=1)
    actions = planner.device_cfg.to_device(native["solution"]).reshape(1, solver.action_horizon, solver.action_dim)
    arrays = {"joint_names": np.asarray(planner.joint_names), "actions": cpu(actions)}
    # BEGIN archived variables and the current cost policy are separate identities.
    entry = {"scope": "Current whole-curve optimizer at archived controls, one particle, fixed clock, no solve",
        "phase": planner.phase,
        "components": {}, "native_state_identity": {}, "gradient_direction_trials": []}
    # END this diagnostic never claims to reconstruct a removed historical objective.
    manager = rollout.constraint_manager
    entry["rollout"] = {"process_mode": manager._process_mode, "horizon": rollout.horizon,
        "anchor_u": cpu(manager._anchor_u).tolist(), "anchor_s": cpu(manager._anchor_s).tolist(),
        "formal_clock": planner.formal_clock}
    report["objective"] = entry

    def evaluate(value, component=None, capture=False):
        with torch.set_grad_enabled(component is not None):
            x = value.detach().clone().requires_grad_(component is not None)
            result = rollout.evaluate_action(x)
            values = terms(result.costs_and_constraints)
            total = result.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True).sum()
            scalars = {key: float(v.detach().sum().item()) for key, v in values.items()}
            scalars["total"] = float(total.item())
            if not all(np.isfinite(v) for v in scalars.values()):
                raise FloatingPointError("Native objective contains a nonfinite value")
            derivative = None
            if component is not None:
                loss = total if component == "total" else values[component].sum()
                derivative = (cpu(torch.autograd.grad(loss, x)[0]) if loss.requires_grad
                              else np.zeros(tuple(x.shape), np.float32))
            if capture:
                original_names = meta["states"]["js_solution"]["joint_names"]
                columns = [original_names.index(name) for name in planner.joint_names]
                for field in ("position", "velocity", "acceleration", "jerk"):
                    actual = cpu(getattr(result.state.joint_state, field))
                    expected = native["js_solution."+field][0][..., columns]
                    entry["native_state_identity"][field] = {"identical": bool(np.array_equal(actual, expected)),
                        "maximum_absolute_difference": float(np.abs(actual-expected).max())}
                    arrays["state."+field] = actual
                for key, value in values.items():
                    arrays["term."+key] = cpu(value)
            return scalars, derivative

    entry["values"], _ = evaluate(actions, capture=True)
    gradients = {}
    for key in entry["values"]:
        _, gradient = evaluate(actions, component=key)
        gradients[key] = gradient
        arrays["gradient."+key] = gradient
        entry["components"][key] = {**gradient_summary(gradient, list(planner.joint_names)),
            "norm": float(np.linalg.norm(gradient.astype(float)))}
        save("objective:gradient:"+key)
    total = gradients["total"].astype(float)
    summed = sum(value.astype(float) for key, value in gradients.items() if key != "total")
    entry["gradient_sum_maximum_absolute_difference"] = float(np.abs(total-summed).max())
    norm = float(np.linalg.norm(total))
    component_norm_sum = sum(np.linalg.norm(value.astype(float))
                             for key, value in gradients.items() if key != "total")
    entry["gradient_cancellation_ratio"] = norm/component_norm_sum if component_norm_sum else 0.
    if norm:
        direction = -total/norm
        entry["component_derivatives_along_negative_total_unit_direction"] = {
            key: float(np.sum(value*direction)) for key, value in gradients.items()}
        for step in (1e-5, 1e-4):
            trial = actions+step*planner.device_cfg.to_device(direction)
            values, _ = evaluate(trial)
            entry["gradient_direction_trials"].append({"control_step_norm_rad": step,
                "full_cubic_joint_position_domain_feasible": bool(planner.physical_domain.feasible(trial).all().item()),
                "objective_values": values,
                "total_change": values["total"]-entry["values"]["total"],
                "scope": "Only the full cubic joint-position domain is checked; other trajectory constraints and acceptance are not asserted"})
    np.savez_compressed(output/"objective-arrays.npz", **arrays)
    save("objective:completed")
# END every derivative is copied before another native workspace evaluation.
