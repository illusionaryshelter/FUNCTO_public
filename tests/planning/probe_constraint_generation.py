#!/usr/bin/env python3
"""Replay archived planning variables through the unchanged native objective.

The recorded runner rebuilds its original observed scene. This test intercepts
only its planning boundary, initializes the production planner, and evaluates
saved controls without IK, optimizer iterations, search, or execution. Every
gradient gets a fresh forward and is copied to CPU before another evaluation.
"""

# BEGIN bounded diagnostic replay; production planning and parameters are unchanged.
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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


def read_json(path):
    return json.loads(path.read_text())


def cpu(value):
    """Own the array: native state and gradient storage is reused in place."""
    return value.detach().cpu().numpy().copy()


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


def require_equal(label, actual, expected, checks):
    """Refuse a different experiment instead of silently repairing its input."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    matched = actual.shape == expected.shape and np.array_equal(actual, expected)
    checks[label] = {"identical": bool(matched), "shape": list(actual.shape)}
    if not matched:
        raise ValueError(f"Archive consistency failed: {label}")


def terms(collection):
    """Namespace component names by their actual native manager category."""
    result = {}
    for label, group in (("cost", collection.costs), ("constraint", collection.constraints),
                         ("hybrid", collection.hybrid_costs_constraints)):
        for name, value in zip(group.names, group.values):
            key = f"{label}:{name}"
            if key in result:
                raise ValueError(f"Duplicate native objective component: {key}")
            result[key] = value
    return result


def gradient_summary(value, names):
    """Summarize controls by named joint, retaining the full array separately."""
    if not np.isfinite(value).all():
        raise FloatingPointError("A replay gradient is nonfinite")
    index = np.unravel_index(np.abs(value).argmax(), value.shape)
    return {"maximum_absolute": float(np.abs(value[index])),
            "maximum_coordinate": {"batch": int(index[0]), "control": int(index[1]),
                                   "joint": names[index[2]], "gradient": float(value[index])},
            "joint_maximum_absolute": dict(zip(names, np.abs(value).max(axis=(0, 1)).tolist())),
            "joint_sum": dict(zip(names, value.sum(axis=(0, 1)).tolist()))}


def objective_probe(planner, rollout, actions, label, witness_joint, archive, arrays, report, save):
    """One original objective, sequential component VJPs, then directional FD.

    Native kernels own mutable forward/backward workspaces. No graph is kept
    across calls, including finite-difference calls. A one-particle goal uses
    the original per-particle objective and avoids allocating line-search
    copies, which are only necessary when actually running the optimizer.
    """
    torch = planner.torch
    names = list(planner.joint_names)
    state_fields = ("position", "velocity", "acceleration", "jerk")
    entry = {"components": {}, "finite_difference": []}
    report["objective_replays"][label] = entry

    def evaluate(value, selected=None, capture=False):
        # Only the requested call owns an autograd graph. Copy every result
        # before returning so the next native forward cannot overwrite it.
        with torch.set_grad_enabled(selected is not None):
            x = value.detach().clone().requires_grad_(selected is not None)
            evaluated = rollout.evaluate_action(x)
            components = terms(evaluated.costs_and_constraints)
            scalars = {name: float(component.detach().sum().cpu())
                       for name, component in components.items()}
            scalars["total"] = float(evaluated.costs_and_constraints.get_sum_cost_and_constraint(
                sum_horizon=True).detach().sum().cpu())
            if not all(np.isfinite(number) for number in scalars.values()):
                raise FloatingPointError("A replay objective is nonfinite")
            states = ({name: cpu(getattr(evaluated.state.joint_state, name))
                       for name in state_fields} if capture else None)
            derivative = None
            if selected is not None:
                loss = (evaluated.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True).sum()
                        if selected == "total" else components[selected].sum())
                # Constant disabled preferences are recorded explicitly; they
                # have no path to controls and mathematically contribute zero.
                derivative = (cpu(torch.autograd.grad(loss, x)[0]) if loss.requires_grad
                              else np.zeros(tuple(x.shape), dtype=np.float32))
            return scalars, derivative, states

    baseline, _, state = evaluate(actions, capture=True)
    arrays[f"{label}.controls"] = cpu(actions)
    entry["values"] = baseline
    entry["shape"] = list(actions.shape)
    for field, value in state.items():
        arrays[f"{label}.state.{field}"] = value
        if label == "native_solution":
            native_names = archive["native_names"]
            columns = [native_names.index(name) for name in names]
            # NumPy advanced indexing changes axis order when combined with
            # integer slicing. Select named columns only after slicing B/S.
            expected = archive["native"][f"js_solution.{field}"][0][..., columns]
            difference = np.abs(value.astype(float)-expected.astype(float))
            entry.setdefault("archived_raw_state", {})[field] = {
                "identical": bool(np.array_equal(value, expected)),
                "maximum_absolute_difference": float(difference.max())}
            # A changed native rollout is itself diagnostic evidence. Input
            # identities are strict above; do not hide a numerical mismatch
            # here by replacing the replay state with the archived state.
    del state
    save(f"{label}:values")

    gradients = {}
    for name in (*[key for key in baseline if key != "total"], "total"):
        _, derivative, _ = evaluate(actions, selected=name)
        gradients[name] = derivative
        arrays[f"{label}.gradient.{name}"] = derivative
        entry["components"][name] = gradient_summary(derivative, names)
        save(f"{label}:gradient:{name}")
    component_sum = np.sum([value.astype(np.float64) for key, value in gradients.items()
                            if key != "total"], axis=0)
    discrepancy = component_sum-gradients["total"]
    entry["gradient_sum_check"] = {
        "maximum_absolute_difference": float(np.abs(discrepancy).max()),
        "total_maximum_absolute": float(np.abs(gradients["total"]).max()),
        "relative_to_total_maximum": float(np.abs(discrepancy).max()
            / max(float(np.abs(gradients["total"]).max()), np.finfo(np.float32).tiny))}

    # Check a control coordinate of the physically violating named joint.
    # This includes the complete transition/FK/custom-cost derivative chain.
    joint_index = names.index(witness_joint)
    cspace = gradients["cost:cspace"][0, :, joint_index]
    control_index = int(np.abs(cspace).argmax())
    if not np.any(cspace):
        raise ValueError("The archived violating joint has no native cspace control gradient")
    direction = torch.zeros_like(actions)
    direction[0, control_index, joint_index] = 1
    entry["direction"] = {"joint": witness_joint, "control": control_index,
        "positive_coordinate_direction": True,
        "analytic_by_component": {name: float(value[0, control_index, joint_index])
                                  for name, value in gradients.items()},
        "negative_total_gradient_coordinate": float(-gradients["total"][0, control_index, joint_index]),
        "scope": "Control-space descent coordinate; this is not a projected feasible step"}
    for epsilon in (1e-3, 1e-4):
        upper, _, _ = evaluate(actions+epsilon*direction)
        lower, _, _ = evaluate(actions-epsilon*direction)
        # Report actual float32 input displacement, not an ideal real epsilon.
        actual_delta = float((cpu(actions+epsilon*direction)[0, control_index, joint_index]
                              - cpu(actions-epsilon*direction)[0, control_index, joint_index]))
        finite = {name: (upper[name]-lower[name])/actual_delta for name in baseline}
        entry["finite_difference"].append({"epsilon": epsilon, "actual_coordinate_span": actual_delta,
            "upper_values": upper, "lower_values": lower, "central_derivative": finite,
            "absolute_error": {name: abs(finite[name]-float(value[0, control_index, joint_index]))
                               for name, value in gradients.items()},
            "scope": "Float32 costs; component differences are retained because total loss can lose small changes"})
        save(f"{label}:finite_difference:{epsilon}")


def single_state_probe(planner, rollout, goal, witness, arrays, report, save):
    """Evaluate both live STATE costs at one archived dense row.

    Run after every full rollout: changing a component's horizon directly
    must not leave its owning rollout's cached horizon out of sync. Restore
    each component shape even when the probe fails. Derivatives other than q
    are the archived values from this same row, held constant in dC/dq.
    """
    torch = planner.torch
    names = list(planner.joint_names)
    index = names.index(witness["joint"])
    sources = (("optimizer_cost", rollout.cost_manager.get_cost("cspace"), goal),
               ("verification_constraint", planner._verification_native.get_cost("cspace"), None))
    result = {"source": witness["description"], "costs": {}}
    report["single_state_replay"] = result
    for label, cost, cost_goal in sources:
        if cost is None or not cost.enabled:
            raise ValueError(f"Required live {label} cspace cost is disabled")
        limits = cost.config.joint_limits
        if tuple(limits.joint_names) != tuple(names):
            raise ValueError("Live cspace limit order differs from its actual state order")
        saved_shape = (cost._batch_size, cost._horizon)
        cost.setup_batch_tensors(1, 1)
        row = {"enabled": bool(cost.enabled), "type": str(cost.config.cost_type),
               "weight": cpu(cost._weight).tolist(),
               "activation_distance": cpu(cost.config.activation_distance).tolist(),
               "regularization_weight": cpu(cost.config.squared_l2_regularization_weight).tolist(),
               "use_grad_input": bool(cost.config.use_grad_input), "physical_limits": {}, "raw_excess": {}}
        result["costs"][label] = row
        for field in ("position", "velocity", "acceleration", "jerk"):
            bounds = cpu(getattr(limits, field))
            value = witness[field]
            excess = np.maximum(bounds[0]-value, 0)+np.maximum(value-bounds[1], 0)
            row["physical_limits"][field] = bounds.tolist()
            row["raw_excess"][field] = dict(zip(names, excess.tolist()))

        def evaluate(qvalue, gradient=False):
            with torch.set_grad_enabled(gradient):
                q = planner.device_cfg.to_device(qvalue[None, None]).clone().requires_grad_(gradient)
                js = planner.JointState(position=q, joint_names=names,
                    velocity=planner.device_cfg.to_device(witness["velocity"][None, None]),
                    acceleration=planner.device_cfg.to_device(witness["acceleration"][None, None]),
                    jerk=planner.device_cfg.to_device(witness["jerk"][None, None]),
                    dt=planner.device_cfg.to_device([witness["dt"]]))
                # Match the actual manager call: optimizer uses augmented
                # torque; independent export passes None to this native cost.
                state = rollout.transition_model.compute_augmented_state(js) if cost_goal is not None else None
                value = cost.forward(js, joint_torque=None if state is None else state.joint_torque,
                    target_joint_state=None if cost_goal is None else cost_goal.goal_js,
                    idxs_target_joint_state=None if cost_goal is None else cost_goal.idxs_goal_js,
                    current_joint_state=None if cost_goal is None else cost_goal.current_js,
                    idxs_current_joint_state=None if cost_goal is None else cost_goal.idxs_current_js,
                    current_state_dt=None if cost_goal is None else cost_goal.current_state_dt)
                loss = value.sum()
                derivative = cpu(torch.autograd.grad(loss, q)[0]) if gradient else None
                return float(loss.detach().cpu()), cpu(value), derivative

        try:
            value, per_joint, derivative = evaluate(witness["position"], True)
            row.update(value=value, per_joint_cost=dict(zip(names, per_joint.reshape(-1).tolist())),
                       position_gradient=dict(zip(names, derivative.reshape(-1).tolist())),
                       inverse_dynamics_enabled=bool(rollout.transition_model.compute_inverse_dynamics),
                       torque_input=("native augmented state" if cost_goal is not None else "None as in final export"),
                       finite_difference=[])
            arrays[f"single_state.{label}.position_gradient"] = derivative
            for epsilon in (1e-3, 1e-4):
                upper_q, lower_q = witness["position"].copy(), witness["position"].copy()
                upper_q[index] += epsilon
                lower_q[index] -= epsilon
                upper, _, _ = evaluate(upper_q)
                lower, _, _ = evaluate(lower_q)
                span = float(upper_q[index]-lower_q[index])
                row["finite_difference"].append({"epsilon": epsilon, "actual_coordinate_span": span,
                    "joint": names[index], "central_derivative": (upper-lower)/span,
                    "analytic_derivative": float(derivative[0, 0, index]),
                    "upper_value": upper, "lower_value": lower})
        finally:
            # A manager that has never evaluated a state uses native -1/-1
            # sentinels. Negative sizes are not a restorable allocation.
            if all(value is not None and value > 0 for value in saved_shape):
                cost.setup_batch_tensors(*saved_shape)
        save(f"single_state:{label}")
    first, second = (result["costs"][name]["physical_limits"] for name, _, _ in sources)
    result["physical_limits_identical"] = first == second


# BEGIN isolate native self forward repeatability and its actual timed inputs.
def self_forward_probe(planner, rollout, actions, witness_joint, arrays, report, save):
    """Audit one saved candidate without modifying native costs or geometry.

    Capture the tensor passed to the enabled native self-collision component,
    including timed finger replacement. CPU float64 recomputation implements
    that kernel's maximum squared overlap, not a different collision metric.
    Each CPU iteration owns at most one state's pair array; all saved sphere
    arrays together are below 32 MiB for the admitted 81-by-416 fixture.
    """
    torch = planner.torch
    cost = rollout.constraint_manager.get_cost("self_collision")
    if cost is None or not cost.enabled or cost.config.convert_to_binary:
        raise ValueError("Self audit requires the original enabled, non-binary optimizer component")
    config = cost.config.self_collision_kin_config
    pairs = cpu(config.collision_pairs).astype(np.int64)
    padding = cpu(config.sphere_padding).astype(np.float64)
    weight = float(cpu(cost._weight).reshape(-1)[0])
    names = list(planner.joint_names)
    joint = names.index(witness_joint)
    control = actions.shape[1]-1
    direction = torch.zeros_like(actions)
    direction[0, control, joint] = 1
    arrays["self_audit.joint_names"] = np.asarray(names)
    arrays["self_audit.controls"] = cpu(actions)
    arrays["self_audit.collision_pairs"] = pairs
    arrays["self_audit.sphere_padding"] = padding
    audit = {"scope": "Original one-particle rollout, native self component only; no optimizer step",
        "joint": witness_joint, "control": int(control), "weight": weight,
        "convert_to_binary": bool(cost.config.convert_to_binary),
        "use_grad_input": bool(cost.config.use_grad_input),
        "sphere_count": int(config.num_spheres), "pair_count": int(len(pairs)),
        "num_blocks_per_batch": int(config.num_blocks_per_batch),
        "max_threads_per_block": int(config.max_threads_per_block),
        "cpu_formula": "0.5*w*max(0, max_valid_pair((ri+padding_i+rj+padding_j)^2 - sum((ci-cj)^2)))",
        "gradient_scope": "Center coordinates only; radii are constant in the audited control direction",
        "repeatability": [], "epsilon_ladder": []}
    report["self_forward_audit"] = audit

    def cpu_exact(spheres, sphere_gradient=None):
        # A single [P,3] float64 array is at most 3 MiB under the production
        # 512-sphere cap. Do not create an [H,P,3] tensor or autograd graph.
        horizon = spheres.shape[1]
        values = np.zeros(horizon)
        argmax = np.zeros((horizon, 2), dtype=np.int64)
        gaps = np.zeros(horizon)
        ties = np.zeros(horizon, dtype=np.int64)
        center_gradient = np.zeros_like(spheres, dtype=np.float64)
        selected_gap = np.zeros(horizon)
        native_center_error = np.zeros(horizon)
        selected_pairs = np.full((horizon, 2), -1, dtype=np.int64)
        for time_index, value in enumerate(spheres[0]):
            geometry = value.astype(np.float64)
            radius = geometry[:, 3]+padding
            first, second = pairs[:, 0], pairs[:, 1]
            delta = geometry[first, :3]-geometry[second, :3]
            score = (radius[first]+radius[second])**2-np.sum(delta*delta, axis=-1)
            score *= (radius[first] >= 0) & (radius[second] >= 0)
            selected = int(score.argmax())
            maximum = max(0., float(score[selected]))
            values[time_index] = .5*weight*maximum
            argmax[time_index] = pairs[selected]
            ordered = np.partition(score, len(score)-2)[-2:]
            gaps[time_index] = float(ordered.max()-ordered.min())
            ties[time_index] = int(np.count_nonzero(score == score[selected]))
            if maximum > 0:
                i, j = pairs[selected]
                center_gradient[0, time_index, i, :3] = weight*(geometry[j, :3]-geometry[i, :3])
                center_gradient[0, time_index, j, :3] = -center_gradient[0, time_index, i, :3]
            if sphere_gradient is not None:
                actual = sphere_gradient[0, time_index]
                active = np.flatnonzero(np.any(actual != 0, axis=-1))
                # Native max selects one pair, with arbitrary pair identity
                # only when scores tie. Compare that chosen pair's score and
                # derivative, rather than rejecting a valid tie subgradient.
                expected = np.zeros_like(actual, dtype=np.float64)
                if len(active) == 2:
                    i, j = active
                    selected_pairs[time_index] = active
                    pair_score = (radius[i]+radius[j])**2-np.sum((geometry[i, :3]-geometry[j, :3])**2)
                    selected_gap[time_index] = maximum-max(0., float(pair_score))
                    expected[i, :3] = weight*(geometry[j, :3]-geometry[i, :3])
                    expected[j, :3] = -expected[i, :3]
                elif len(active) != 0:
                    raise ValueError("Native self gradient unexpectedly writes more than one sphere pair")
                elif maximum > 0:
                    selected_gap[time_index] = maximum
                native_center_error[time_index] = float(np.max(np.abs(actual[:, :3]-expected[:, :3])))
        return {"values": values, "argmax": argmax, "top_two_score_gap": gaps,
                "exact_tie_count": ties, "center_gradient": center_gradient,
                "native_selected_pair": selected_pairs, "native_selected_score_gap": selected_gap,
                "native_center_gradient_error": native_center_error}

    def evaluate(value, gradient, label):
        captured = {}
        original = cost.forward

        def record_input(spheres):
            if captured:
                raise ValueError("Expected one native self component per original rollout")
            # Keep the graph tensor only until this call's immediate backward.
            # Detached snapshots own storage, independent of native buffers.
            captured["input"] = spheres
            captured["sphere_snapshot"] = spheres.detach().clone()
            result = original(spheres)
            captured["output_snapshot"] = result.detach().clone()
            return result

        with torch.set_grad_enabled(gradient):
            x = value.detach().clone().requires_grad_(gradient)
            with patch.object(cost, "forward", new=record_input):
                evaluated = rollout.evaluate_action(x)
            self_value = terms(evaluated.costs_and_constraints)["constraint:self_collision"]
            loss = self_value.sum()
            action_gradient = sphere_gradient = None
            if gradient:
                action_grad, sphere_grad = torch.autograd.grad(loss, (x, captured["input"]))
                action_gradient, sphere_gradient = cpu(action_grad), cpu(sphere_grad)
            spheres = cpu(captured["sphere_snapshot"])
            values = cpu(captured["output_snapshot"]).reshape(-1)
            state = cpu(evaluated.state.joint_state.position)
        exact = cpu_exact(spheres, sphere_gradient)
        arrays[f"self_audit.{label}.spheres"] = spheres
        arrays[f"self_audit.{label}.joint_position"] = state
        arrays[f"self_audit.{label}.gpu_per_time"] = values
        for key, data in exact.items():
            arrays[f"self_audit.{label}.cpu_{key}"] = data
        if gradient:
            arrays[f"self_audit.{label}.action_gradient"] = action_gradient
            arrays[f"self_audit.{label}.sphere_gradient"] = sphere_gradient
        errors = values-exact["values"]
        worst = np.argsort(-np.abs(errors), kind="stable")[:5]
        result = {"label": label, "gradient_enabled": gradient,
            "gpu_sum_float32": float(values.sum(dtype=np.float32)),
            "gpu_per_time_sum_float64": float(values.sum(dtype=np.float64)),
            "cpu_sum_float64": float(exact["values"].sum()),
            "maximum_forward_difference": float(np.abs(errors).max()),
            "worst_forward_samples": [{"sample": int(i), "gpu": float(values[i]),
                "cpu_float64": float(exact["values"][i]), "cpu_argmax": exact["argmax"][i].tolist(),
                "cpu_top_two_score_gap": float(exact["top_two_score_gap"][i])} for i in worst]}
        if gradient:
            result.update(action_directional_derivative=float(action_gradient[0, control, joint]),
                maximum_native_center_gradient_error=float(exact["native_center_gradient_error"].max()),
                maximum_native_selected_score_gap=float(exact["native_selected_score_gap"].max()))
        return result, spheres, values, exact, action_gradient, sphere_gradient, state

    baseline = None
    for repetition in range(3):
        for gradient in (False, True):
            label = f"repeat_{repetition}_{'grad' if gradient else 'no_grad'}"
            result = evaluate(actions, gradient, label)
            if baseline is None:
                baseline = result
            row, spheres, values, exact, action_grad, sphere_grad, state = result
            row["versus_first_input"] = {
                "spheres_identical": bool(np.array_equal(spheres, baseline[1])),
                "sphere_maximum_absolute_difference": float(np.abs(spheres-baseline[1]).max()),
                "joint_positions_identical": bool(np.array_equal(state, baseline[6])),
                "gpu_per_time_identical": bool(np.array_equal(values, baseline[2])),
                "gpu_maximum_absolute_difference": float(np.abs(values-baseline[2]).max())}
            audit["repeatability"].append(row)
            if gradient and repetition == 0:
                analytic = result
            save(f"self_audit:{label}")

    for epsilon in (.01, .003, .001, .0003, .0001, .00003):
        lower = evaluate(actions-epsilon*direction, False, f"epsilon_{epsilon}_lower")
        upper = evaluate(actions+epsilon*direction, False, f"epsilon_{epsilon}_upper")
        lo_q = cpu(actions-epsilon*direction)[0, control, joint]
        hi_q = cpu(actions+epsilon*direction)[0, control, joint]
        span = float(hi_q-lo_q)
        gpu_fd = (upper[2].astype(np.float64)-lower[2].astype(np.float64))/span
        cpu_fd = (upper[3]["values"]-lower[3]["values"])/span
        sphere_direction = (upper[1].astype(np.float64)-lower[1].astype(np.float64))/span
        # Contract the baseline sphere derivative with the measured sphere
        # directional change, separating FK/VJP behavior from native max FD.
        cpu_chain = np.sum(analytic[3]["center_gradient"]*sphere_direction, axis=(0, 2, 3))
        native_chain = np.sum(analytic[5].astype(np.float64)[..., :3]
                              *sphere_direction[..., :3], axis=(0, 2, 3))
        mismatch = gpu_fd-cpu_fd
        worst = np.argsort(-np.abs(mismatch), kind="stable")[:5]
        row = {"epsilon": epsilon, "actual_coordinate_span": span,
            "lower": lower[0], "upper": upper[0],
            "gpu_central_derivative_sum_float64": float(gpu_fd.sum()),
            "cpu_central_derivative_sum_float64": float(cpu_fd.sum()),
            "baseline_action_analytic_derivative": analytic[0]["action_directional_derivative"],
            "baseline_cpu_sphere_gradient_times_sphere_fd": float(cpu_chain.sum()),
            "baseline_native_sphere_gradient_times_sphere_fd": float(native_chain.sum()),
            "maximum_radius_directional_change": float(np.abs(sphere_direction[..., 3]).max()),
            "cpu_argmax_changed_samples": int(np.any(upper[3]["argmax"] != lower[3]["argmax"], axis=1).sum()),
            "worst_gpu_vs_cpu_fd_samples": [{"sample": int(i), "gpu_fd": float(gpu_fd[i]),
                "cpu_fd": float(cpu_fd[i]), "lower_gpu": float(lower[2][i]), "upper_gpu": float(upper[2][i]),
                "lower_cpu_argmax": lower[3]["argmax"][i].tolist(),
                "upper_cpu_argmax": upper[3]["argmax"][i].tolist(),
                "baseline_cpu_argmax": analytic[3]["argmax"][i].tolist(),
                "lower_cpu_score_gap": float(lower[3]["top_two_score_gap"][i]),
                "upper_cpu_score_gap": float(upper[3]["top_two_score_gap"][i])} for i in worst]}
        arrays[f"self_audit.epsilon_{epsilon}.gpu_fd_per_time"] = gpu_fd
        arrays[f"self_audit.epsilon_{epsilon}.cpu_fd_per_time"] = cpu_fd
        audit["epsilon_ladder"].append(row)
        save(f"self_audit:epsilon:{epsilon}")
# END isolate native self forward repeatability and its actual timed inputs.


def replay(problem, scene, robot, budget, args, report, save, arrays):
    """Initialize once, bind the installed native goal lifecycle, then replay."""
    import torch
    from manipulation.planning import _CuroboPlanner
    from curobo._src.solver.solve_mode import SolveMode
    from curobo._src.solver.solve_state import SolveState

    source = args.source_run/"planning"
    source_report = read_json(source/"planning-report.json")
    config = read_json(source/"effective-config-00.json")
    native_info = read_json(source/"native-result-00.json")
    checks = report["archive_consistency"]
    if asdict(budget) != source_report["budget"] or budget.trajectory_seeds != 1:
        raise ValueError("Replay requires exactly the original one-seed planning budget")
    checks["budget_identical"] = True
    with np.load(source/"optimizer-input-00.npz", allow_pickle=False) as saved:
        seed_data = {name: saved[name].copy() for name in saved.files}
    with np.load(source/"native-result-00.npz", allow_pickle=False) as saved:
        native = {name: saved[name].copy() for name in saved.files}
    with np.load(source/"effective-config-00.npz", allow_pickle=False) as saved:
        configuration = {name: saved[name].copy() for name in saved.files}
    with np.load(args.source_run/"observed-free.npz", allow_pickle=False) as saved:
        for name, actual in (("known_free", scene.coverage.known_free), ("origin", scene.coverage.origin),
                             ("voxel_size", scene.coverage.voxel_size),
                             ("observation_ids", scene.coverage.observation_ids)):
            require_equal(f"observed_free.{name}", actual, saved[name], checks)
    with np.load(args.source_run/"entrance-scene.npz", allow_pickle=False) as saved:
        # Include the independently controlled fingers and the actual sphere
        # model, not only the active arm slice consumed by the optimizer.
        with np.load(args.output/"scene-build"/"entrance-scene.npz", allow_pickle=False) as rebuilt:
            for field in ("joint_names", "actual_q", "actual_dq", "measured_robot_spheres"):
                require_equal(f"entrance.{field}", rebuilt[field], saved[field], checks)
        columns = [saved["joint_names"].tolist().index(name) for name in problem.joint_names]
        require_equal("measured_q0", problem.start_q, saved["actual_q"][columns], checks)
        require_equal("measured_dq0", problem.start_dq, saved["actual_dq"][columns], checks)
    save("archive_scene_checked")

    planner = _CuroboPlanner(problem, scene, robot, budget, "cuda:0")
    try:
        names = list(planner.joint_names)
        arrays["joint_names"] = np.asarray(names)
        require_equal("joint_names", names, seed_data["joint_names"], checks)
        require_equal("native_active_joint_names", names, native["active_joint_names"], checks)
        require_equal("native_q0", cpu(planner.start.position), seed_data["start_q"], checks)
        require_equal("native_dq0", cpu(planner.start.velocity), seed_data["start_dq"], checks)
        for key in ("u", "s"):
            require_equal(f"progress_{key}", getattr(planner.progress_map, key), seed_data[f"progress_{key}"], checks)
            require_equal(f"configuration.progress_{key}", getattr(planner.progress_map, key),
                          configuration[config[f"progress_{key}"]["npz_key"]], checks)
        if planner.formal_clock != config["formal_clock"]:
            raise ValueError("Rebuilt formal clock differs from archive")
        checks["formal_clock_identical"] = True
        report["formal_clock"] = planner.formal_clock.copy()
        report["budget"] = asdict(budget)
        report["native_source"] = str(sys.modules["curobo"].__file__)
        report["scene_workspace_bytes"] = planner.scene_workspace_bytes
        report["anchor_fk_workspace_bytes"] = planner.anchor_fk_workspace_bytes

        native_names = native_info["states"]["interpolated_trajectory"]["joint_names"]
        columns = [native_names.index(name) for name in names]
        count = int(native["interpolated_last_tstep"].item())
        dense = {field: native[f"interpolated_trajectory.{field}"][0, 0, :count][:, columns]
                 for field in ("position", "velocity", "acceleration", "jerk")}
        bounds = cpu(planner._verification_native.get_cost("cspace").config.joint_limits.position)
        excess = np.maximum(bounds[0]-dense["position"], 0)+np.maximum(dense["position"]-bounds[1], 0)
        sample, joint = np.unravel_index(excess.argmax(), excess.shape)
        if excess[sample, joint] <= 0:
            raise ValueError("Source candidate has no actual dense position-limit violation")
        witness = {field: value[sample].copy() for field, value in dense.items()}
        witness.update(joint=names[joint], dt=float(native["interpolated_trajectory.dt"].item()))
        witness["description"] = {"file": str(source/"native-result-00.npz"),
            "representation": "interpolated_trajectory", "sample": int(sample),
            "time_s": int(sample)*witness["dt"], "joint": names[joint],
            "position": float(witness["position"][joint]), "raw_position_excess": float(excess[sample, joint]),
            "joint_names": names, "derivatives": "q/dq/ddq/jerk all copied from the same archived row"}
        report["witness"] = witness["description"]
        for field in dense:
            arrays[f"single_state.archived.{field}"] = witness[field]

        # Mirror installed TrajOptSolver._solve_impl goal preparation only.
        # Do not invoke _update_rollout_params: that expands line-search
        # particles. RobotRollout.update_params explicitly supports one.
        solver = planner.trajopt
        solver.disable_tool_pose_tracking()
        seed = planner.device_cfg.to_device(seed_data["seed_controls"])
        expected_shape = (1, 1, solver.action_horizon, solver.action_dim)
        if tuple(seed.shape) != expected_shape or tuple(native["solution"].shape) != expected_shape:
            raise ValueError("Archive controls must match the unchanged single-seed action layout")
        goal_pose = planner._goal(problem.phase_count)
        solve_state = SolveState(solve_type=SolveMode.SINGLE, num_trajopt_seeds=1,
            batch_size=1, num_envs=1, num_goalset=goal_pose.num_goalset, tool_frames=goal_pose.tool_frames)
        seed_goal = planner.JointState.from_position(seed[..., -1, :].clone())
        seed_goal.dt = torch.full((1, 1), budget.maximum_trajectory_dt, device=seed.device, dtype=seed.dtype)
        goal, _ = solver._prepare_goal_buffer(solve_state, goal_pose, planner.start,
                                              use_implicit_goal=False, seed_goal_state=seed_goal, goal_state=None)
        rollout = solver.optimizer_rollouts[0]
        rollout.update_params(goal, num_particles=1)
        report["rollout_binding"] = {"batch": 1, "particles": 1,
            "configured_line_search_particles": solver.optimizer.optimizers[0].config.num_particles,
            "horizon": rollout.horizon, "action_horizon": solver.action_horizon,
            "control_space": str(rollout.transition_model.control_space),
            "goal_mapping": "native _prepare_goal_buffer and original optimizer rollout.update_params(num_particles=1)",
            "dt": float(seed_goal.dt.item()), "tool_pose_tracking_disabled_as_in_production": True}
        archive = {"native": native, "native_names": native_info["states"]["js_solution"]["joint_names"]}
        save("planner_initialized_without_solve")
        # BEGIN optional focused audit preserves all original input/goal checks.
        if args.self_forward_audit:
            actions = planner.device_cfg.to_device(native["solution"]).reshape(
                expected_shape[0], *expected_shape[2:])
            self_forward_probe(planner, rollout, actions, witness["joint"], arrays, report, save)
            return
        # END focused audit performs no broad objective replay or solve.
        for label, value in (("seed", seed.reshape(expected_shape[0], *expected_shape[2:])),
                             ("native_solution", planner.device_cfg.to_device(native["solution"]).reshape(
                                 expected_shape[0], *expected_shape[2:]))):
            objective_probe(planner, rollout, value, label, witness["joint"], archive, arrays, report, save)
        single_state_probe(planner, rollout, rollout._num_particles_goal, witness, arrays, report, save)
    finally:
        planner.close()


def run(args):
    """Keep interception scoped and record all incomplete stages on failure."""
    import torch
    import run_recorded_skill
    from manipulation import planning
    from curobo.inverse_kinematics import InverseKinematics
    from curobo.trajectory_optimizer import TrajectoryOptimizer

    args.output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    started = time.monotonic()
    report = {"status": "preparing", "source_run": str(args.source_run.resolve()),
              "scope": "Diagnostic replay of archived controls; no solve, search, execution, or changed constraints",
              "archive_consistency": {}, "objective_replays": {}, "completed_stages": []}

    def save(stage):
        report["completed_stages"].append(stage)
        report["elapsed_seconds"] = time.monotonic()-started
        write_report(args.output/"constraint-generation-report.json", report)
        np.savez_compressed(args.output/"constraint-generation-arrays.npz", **arrays)
        print(json.dumps({"stage": stage, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)

    def forbidden(*unused, **kwargs):
        raise RuntimeError("This diagnostic must not call IK, optimizer solve, or path search")

    def capture(problem, scene, robot, output, *, budget, provenance, execution_limits):
        if report.get("planning_boundary_intercepted"):
            raise ValueError("The recorded runner reached its planning boundary more than once")
        report["planning_boundary_intercepted"] = True
        replay(problem, scene, robot, budget, args, report, save, arrays)
        return SimpleNamespace(success=False, diagnostics={"initial_state": None})

    runner_args = SimpleNamespace(**vars(args))
    runner_args.output = args.output/"scene-build"
    try:
        original = read_json(args.source_run/"run-report.json")
        for key, value in (("segment_id", args.segment), ("candidate_id", args.candidate)):
            if original[key] != value:
                raise ValueError(f"Runner input does not match the source archive: {key}")
        for key in ("manifest", "feedback", "robot_fixture", "execution_limits", "finger_targets"):
            if Path(original[f"{key}_path"]).resolve() != getattr(args, key).resolve():
                raise ValueError(f"Runner path does not match the source archive: {key}")
        if (args.scene_workspace_mib != 2048 or args.torch_memory_mib != 4096
                or args.trajectory_seeds != 1):
            raise ValueError("This replay keeps the original 2-GiB scene, 4-GiB Torch, one-seed budget")
        with patch("recorded_task_planning.plan_and_record", new=capture), \
             patch.object(planning._CuroboPlanner, "reference_seeds", new=forbidden), \
             patch.object(planning._CuroboPlanner, "optimize", new=forbidden), \
             patch.object(planning._CuroboPlanner, "ompl_seed", new=forbidden), \
             patch.object(InverseKinematics, "solve_pose", new=forbidden), \
             patch.object(TrajectoryOptimizer, "solve_pose", new=forbidden):
            run_recorded_skill.run(runner_args)
        if not report.get("planning_boundary_intercepted"):
            raise ValueError("Scene building did not reach the replay boundary")
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # The runner's placeholder result must never look like an executed
        # planning attempt. Preserve its preparation report with explicit scope.
        source = runner_args.output/"run-report.json"
        if source.exists():
            prepared = read_json(source)
            prepared.update(planning_called=False, execution_tested=False,
                            planning_boundary_intercepted=bool(report.get("planning_boundary_intercepted")),
                            scope="Original scene preparation for archived objective replay only")
            prepared.pop("planning_success", None)
            write_report(source.with_name("scene-build-report.json"), prepared)
            source.unlink()
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if torch.cuda.is_initialized():
            report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
        save("finished")


def parse_args():
    """Keep all scene choices explicit, using the original runner's inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "robot-fixture", "feedback", "execution-limits", "finger-targets",
                 "source-run", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--pregrasp-frame", type=int)
    parser.add_argument("--owner-mask", nargs=2, action="append", required=True)
    parser.add_argument("--owner-geometry", type=Path)
    parser.add_argument("--center", type=float, nargs=3, required=True)
    parser.add_argument("--extent", type=float, nargs=3, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--depth-error", type=float, required=True)
    parser.add_argument("--robot-depth-tolerance", type=float, required=True)
    parser.add_argument("--robot-depth-tolerance-source", required=True)
    parser.add_argument("--initial-state-assumption-source", required=True)
    parser.add_argument("--knots", type=int, default=16)
    parser.add_argument("--interpolation-buffer-size", type=int, default=768)
    parser.add_argument("--ik-seeds", type=int, default=2)
    parser.add_argument("--trajectory-seeds", type=int, default=1)
    parser.add_argument("--max-validation-points", type=int, default=4096)
    parser.add_argument("--max-refinement-points", type=int, default=16)
    parser.add_argument("--scene-workspace-mib", type=int, choices=(2048,), default=2048)
    parser.add_argument("--torch-memory-mib", type=int, choices=(4096,), default=4096)
    # BEGIN opt-in isolation of the observed native self finite-difference gap.
    parser.add_argument("--self-forward-audit", action="store_true",
                        help="Only replay saved native_solution: repeated actual self inputs and epsilon ladder")
    # END the default archived objective probe remains unchanged.
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
# END bounded diagnostic replay; production planning and parameters are unchanged.
