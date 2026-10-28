"""Local SLSQP adapter with one batched task/geometry derivative operation.

The lifecycle, fixed-row selection, physical units, objective scaling, request
checks and exception evidence follow FUNCTO unified_solver.solve_trajectory at
9602ebb. This experiment owns the explicit adapter because the original solver
has no derivative-injection argument. SciPy still owns all numerical updates.
Only task Jacobian evaluation and its simultaneous memory accounting change.
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
import torch

from manipulation.resource_admission import require_host_memory
from manipulation.unified_solver import ConstrainedSolution, solver_storage_bytes
from .derivatives import batch_derivative_storage_bytes, task_jacobian

# BEGIN explicit batched derivative integration in the original solver lifecycle.
def solve_trajectory(computation, initial, *, iterations, check_request, pending_geometry_bytes, report_out):
    """Submit the full problem with batched task and geometry derivatives.

    SLSQP may evaluate infeasible trials and starts; the original constraints
    decide feasibility. Its stopping tolerance is a numerical search setting,
    not a task or geometric allowance. Candidate validation retains every
    original physical bound and task tolerance regardless of library status.

    ``pending_geometry_bytes`` is the scene's current unallocated material
    storage plus one live geometry/FK evaluation. Existing field/model arrays
    and previously constructed material samples already belong to measured
    RSS. The caller recomputes this quantity before each serial seed solve.

    ``report_out`` is caller-owned and remains available when a request check
    raises. Its bounded snapshots describe actual evaluated controls and the
    last SLSQP callback iterate; none is a candidate-acceptance certificate.
    """
    initial = np.asarray(initial, dtype=float)
    shape = initial.shape
    cache = {}
    cache_x = None
    counts = {"objective_evaluations": 0, "task_evaluations": 0, "physical_evaluations": 0}
    evaluation_seconds = {"objective": 0., "task": 0., "physical": 0.}
    started = perf_counter()
    # BEGIN retain bounded, same-control evidence across a native interruption.
    report = report_out
    report.update(backend="scipy.optimize.SLSQP", state="running", solver_called=False,
                  task_query_count=len(computation.parameters), completed_evaluations={
                      "objective": 0, "task": 0, "physical": 0}, callback_iterations=0,
                  evaluation_seconds=evaluation_seconds,
                  latest_trial=None, latest_complete_trial=None, last_callback_iterate=None)
    trial_index = 0

    def snapshot_for(x):
        for name in ("latest_trial", "latest_complete_trial"):
            snapshot = report[name]
            if snapshot is not None and np.array_equal(np.asarray(snapshot["controls"]).ravel(), x):
                return snapshot
        return None

    def callback(x):
        report["callback_iterations"] += 1
        matched = snapshot_for(x)
        report["last_callback_iterate"] = {
            "callback_index": report["callback_iterations"], "controls": x.reshape(shape).tolist(),
            "trial_index": None if matched is None else matched["trial_index"],
            "completed_components": [] if matched is None else list(matched["completed_components"])}
        check_request()
    # END callbacks record the library's iterate without another computation.

    try:
        # Establish fixed row counts with ordinary FK before allocating any dense
        # Jacobian or SLSQP workspace. Continuous cubic rows comprise two position,
        # two velocity, four acceleration-endpoint and two jerk rows per segment
        # and joint. Every task declaration retains its own original signed rows.
        check_request()
        initial_controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        counts["task_evaluations"] += 1
        # BEGIN bind row identity using the same complete task/FK evaluation.
        declared_rows = ({}, {})
        component_started = perf_counter()
        try:
            initial_task = computation.task(initial_controls, solver_rows_out=declared_rows)
        finally:
            evaluation_seconds["task"] += perf_counter()-component_started
        report["completed_evaluations"]["task"] += 1
        full_eq_count, full_inequality_count = tuple(value.numel() for value in initial_task)
        solver_rows = tuple(torch.cat([value.reshape(-1) for value in group.values()])
            if group else torch.zeros(0, dtype=torch.bool) for group in declared_rows)
        solver_rows[0][computation.fixed_equality_rows] = False
        eq_count, inequality_count = tuple(int(value.sum()) for value in solver_rows)
        fixed_values = initial_task[0][computation.fixed_equality_rows].detach().numpy()
        fixed_report = {computation.equality_keys[index]: float(value)
            for index, value in zip(computation.fixed_equality_rows, fixed_values)}
        report.update(complete_task_equalities=full_eq_count, fixed_entry_equalities=fixed_report,
                      complete_task_inequalities=full_inequality_count,
                      solver_task_equalities=eq_count, solver_task_inequalities=inequality_count)
        fixed_constraints, fixed_feasible = [], True
        for kind, values, selected, group in zip(("equality", "inequality"),
                                                (value.detach() for value in initial_task),
                                                solver_rows, declared_rows):
            offset = 0
            for key, layout in group.items():
                size = layout.numel()
                indices = torch.nonzero(~selected[offset:offset+size]).flatten()
                constant = values[offset:offset+size][indices].detach().numpy()
                if len(indices):
                    finite = np.isfinite(constant)
                    feasible = finite & (constant == 0. if kind == "equality" else constant <= 0.)
                    fixed_constraints.append({"kind": kind, "key": key,
                        "row_indices": indices.tolist(), "values": constant.tolist(),
                        "finite": finite.tolist(), "feasible": bool(feasible.all())})
                    fixed_feasible = fixed_feasible and bool(feasible.all())
                offset += size
        report["fixed_task_constraints"] = fixed_constraints
        # A fixed violation cannot be changed by controls. Keep its original
        # key, flattened component-row index and value as the failure evidence.
        # Complete task() and final candidate acceptance retain all conditions.
        if not fixed_feasible:
            report.update(state="completed", success=False, status=None, message="Fixed task constraints are infeasible",
                solver_called=False, iterations=0, objective=None, elapsed_seconds=perf_counter()-started,
                maximum_control_update=0., unchanged_from_seed=True, **counts)
            return ConstrainedSolution(initial.copy(), report)
        task_count = eq_count+inequality_count
        jacobian_method = "none" if not task_count else "torch_batched_vector_jacobian_product"
        # END declarations, fixed entrance and the submitted matrix share one layout.
        physical_count = 10*computation.domain.intervals*shape[1]
        memory = solver_storage_bytes(variables=initial.size, task_equalities=eq_count,
            task_inequalities=inequality_count, physical_inequalities=physical_count)
        # BEGIN batched derivative arrays use their actual row/FK dimensions.
        geometry_workspace = computation.geometry.workspace_bytes()
        derivative_memory = batch_derivative_storage_bytes(
            rows=task_count, variables=initial.size,
            samples=len(computation.parameters), frames=len(computation.kinematics.frames),
            joints=len(computation.kinematics.joint_names),
            geometry_workspace=geometry_workspace)
        memory.pop("task_derivative_vectors")
        memory.update(derivative_memory)
        memory["physical_sampling_rows"] = 8*8*computation.domain.intervals*shape[1]*shape[0]
        # END the original forward geometry and solver matrices remain counted separately.
        # BEGIN geometry and constrained-solver buffers coexist in each trial.
        # Material samples may still be lazy at the first objective evaluation.
        # Admit that storage and the live pair/FK graph with the dense solver;
        # separate admissible increments do not bound their simultaneous peak.
        # BEGIN the complete reverse dictionary already owns this scalar increment.
        scalar_geometry_reverse = geometry_workspace.get("additional_largest_pinocchio_reverse", 0)
        memory["pending_geometry_bytes"] = pending_geometry_bytes - scalar_geometry_reverse
        # END keep all original forward scene work, counting each reverse array once.
        report.update(solver_storage_bytes=memory,
            task_jacobian_method=jacobian_method)
        require_host_memory(sum(memory.values()), pending_existing_bytes=0,
                            max_rss_bytes=computation.max_host_rss_bytes)

        def admit_task_derivatives():
            # BEGIN current forward graphs belong to RSS; batched reverse arrays do not.
            require_host_memory(sum(derivative_memory.values()), pending_existing_bytes=0,
                                max_rss_bytes=computation.max_host_rss_bytes)
            # END the same dimensions are admitted before each reverse query.
        del initial_task, initial_controls, declared_rows

        def evaluate(kind, x):
            nonlocal cache_x, trial_index
            check_request()
            if cache_x is None or not np.array_equal(x, cache_x):
                cache.clear()
                cache_x = x.copy()
                trial_index += 1
                report["latest_trial"] = {"trial_index": trial_index,
                    "controls": x.reshape(shape).tolist(), "completed_components": []}
            if kind in cache:
                return cache[kind]
            counts[kind + "_evaluations"] += 1
            # BEGIN actual evaluation time includes interrupted work, not cache hits.
            component_started = perf_counter()
            try:
                if kind == "physical":
                    output = computation.domain.signed_constraints(x.reshape(shape))
                else:
                    controls = torch.tensor(x.reshape(shape), dtype=torch.float64, requires_grad=True)
                    if kind == "objective":
                        components = computation.objective_components(controls)
                        value = sum(components.values())
                        gradient, = torch.autograd.grad(value, controls)
                        output = (float(value.detach()), gradient.detach().numpy().ravel())
                    else:
                        complete = computation.task(controls)
                        values = torch.cat(tuple(value[selected]
                            for value, selected in zip(complete, solver_rows)))
                        jacobian = task_jacobian(values, controls, check_request=check_request,
                                                 admit_work=admit_task_derivatives)
                        output = (values.detach().numpy(), jacobian, (eq_count, inequality_count))
            finally:
                evaluation_seconds[kind] += perf_counter()-component_started
            # END each started component contributes its consumed wall time once.
            cache[kind] = output
            # BEGIN completion and geometry evidence belong to this exact x only.
            report["completed_evaluations"][kind] += 1
            snapshot = report["latest_trial"]
            snapshot["completed_components"].append(kind)
            if kind == "objective":
                snapshot.update(objective=output[0], geometry_objective=deepcopy(computation.geometry_report))
            required = {"objective", "physical"} | ({"task"} if eq_count + inequality_count else set())
            if required.issubset(snapshot["completed_components"]):
                report["latest_complete_trial"] = deepcopy(snapshot)
            # END a later partial evaluation cannot overwrite completed evidence.
            check_request()
            return output

        x0 = initial.ravel().copy()
        # BEGIN one positive constant changes numerical units, not objective order.
        # Production task weights can make the initial gradient O(1e10), which
        # destabilizes SLSQP's first quadratic subproblem. Fix this divisor once;
        # every trial value and derivative use the same factor, while reporting
        # and candidate ranking retain the original physical objective. A fixed
        # positive divisor preserves exact minimizers, but SLSQP's fixed ftol
        # jointly checks objective, gradient and original constraint values;
        # finite-budget iterates and stopping are not invariant to this scale.
        initial_value, initial_gradient = evaluate("objective", x0)
        objective_scale = max(1., abs(initial_value), float(np.abs(initial_gradient).max()))
        report.update(objective_scale=objective_scale, initial_objective=initial_value,
                      initial_gradient_linf=float(np.abs(initial_gradient).max()))

        def scaled_objective(x):
            value, gradient = evaluate("objective", x)
            return value/objective_scale, gradient/objective_scale
        # END argmin f and argmin f/positive_constant have exactly the same set.
        constraints = [{"type": "ineq", "fun": lambda x: -evaluate("physical", x)[0],
                        "jac": lambda x: -evaluate("physical", x)[1]}]
        if eq_count:
            constraints.append({"type": "eq", "fun": lambda x: evaluate("task", x)[0][:eq_count],
                                "jac": lambda x: evaluate("task", x)[1][:eq_count]})
        if inequality_count:
            constraints.append({"type": "ineq", "fun": lambda x: -evaluate("task", x)[0][eq_count:],
                                "jac": lambda x: -evaluate("task", x)[1][eq_count:]})
        report["solver_called"] = True
        result = minimize(scaled_objective, x0, jac=True, method="SLSQP",
            constraints=constraints, callback=callback,
            options={"maxiter": int(iterations), "ftol": 1e-9, "disp": False})
        matched = snapshot_for(result.x)
        report.update({"state": "completed", "success": bool(result.success), "solver_called": True,
                  "status": int(result.status), "message": str(result.message),
                  "iterations": int(result.nit), "objective": float(result.fun)*objective_scale,
                  "objective_scale": objective_scale, "normalized_objective": float(result.fun),
                  "initial_objective": initial_value,
                  "initial_gradient_linf": float(np.abs(initial_gradient).max()),
                  "elapsed_seconds": perf_counter() - started, **counts,
                  "maximum_control_update": float(np.max(np.abs(result.x - x0))),
                  "unchanged_from_seed": bool(np.array_equal(result.x, x0)),
                  "solver_storage_bytes": memory,
                  "task_jacobian_method": jacobian_method,
                  "geometry_objective": ({} if matched is None else deepcopy(matched.get("geometry_objective", {}))),
                  "geometry_trial_index": None if matched is None else matched["trial_index"]})
        return ConstrainedSolution(result.x.reshape(shape).copy(), report)
    except BaseException as error:
        # The caller receives this original exception and the completed report
        # fields; recording never changes budget, cancellation or error policy.
        report.update(state="interrupted", interrupted=True,
                      error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        report.update(elapsed_seconds=perf_counter()-started, **counts)

# END explicit batched derivative integration.
