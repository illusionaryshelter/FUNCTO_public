"""One constrained search on the saved cubic and original task declarations.

SciPy SLSQP owns numerical steps and termination. Physical joint inequalities
are continuous polynomial extrema; task inequalities retain their signed
physical units on a fixed, complete time query. Pinocchio differentiates the
registered robot. The observed-field objective guides geometric search; the
original entity queries independently decide whether a candidate is usable.
"""

# BEGIN fixed-clock constrained trajectory search with explicit model semantics.
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
import torch

from .planning_problem import _first_event_query, _quaternion, quaternion_multiply, quaternion_rotate
from .resource_admission import require_host_memory


def time_mean_weights(times):
    """Trapezoid weights normalized by this grid's physical duration."""
    times = np.asarray(times, dtype=float)
    delta = np.diff(times)
    if len(times) < 2 or np.any(delta <= 0):
        raise ValueError("Physical-time integration requires an increasing grid")
    return (np.r_[delta, 0.] + np.r_[0., delta]) / (2 * (times[-1] - times[0]))


def motion_cost(states, bounds, configuration, native_dt, weights):
    """Evaluate the declared STATE cost in physical units on a fixed clock.

    The maintained cuRobo configuration supplies the four joint-bound and
    velocity/acceleration/jerk regularization weights. Its STATE kernel uses
    .5*w*x². The native retiming factors are bound to the original ordinary
    interval exactly once, regardless of this evaluation grid. This planner
    has no inverse-dynamics state, so the native zero-effort terms are zero.
    """
    q = states[0]
    bound_weights = q.new_tensor(configuration["weight"][:4])
    regularization = q.new_tensor(configuration["squared_l2_regularization_weight"][:3])
    if configuration["retime_weights"]:
        bound_weights = bound_weights * q.new_tensor([1., native_dt, native_dt**2, native_dt**3])
    if configuration["retime_regularization_weights"]:
        regularization = regularization * q.new_tensor([native_dt, native_dt**2, native_dt**3])
    activation = q.new_tensor(configuration["activation_distance"][:4])[:, None]
    margin = activation * (bounds[:, 1] - bounds[:, 0])
    lower, upper = bounds[:, 0] + margin, bounds[:, 1] - margin
    values = torch.stack(states)
    distance = torch.relu(lower[:, None, None] - values) + torch.relu(values - upper[:, None, None])
    per_time = .5 * (distance.square() * bound_weights[:, None, None, None]).sum((0, 3))
    per_time = per_time + .5 * sum(weight * value.square().sum(-1)
        for weight, value in zip(regularization, states[1:]))
    return (per_time * weights).sum()


# BEGIN arbitrary saved curves and numerical controls share one robot/TCP computation.
def robot_tool_states(*, problem, scene, kinematics, q, dq, joint_names, progress, times):
    """Complete nominal model joints and form the reference-frame tool state.

    ``q/dq`` describe one shared physical-time sample layout and named joint
    columns. PinKinematics already includes the base calibration; only the
    fixed link-to-tool transform is composed here. Missing finger columns use
    the scene's declared nominal trajectory and retain its uncertainty model.
    """
    names = tuple(joint_names)
    hand = scene.gripper
    if hand is not None:
        missing = [name for name in hand.joint_names if name not in names]
        if missing:
            hand_state = hand.state(progress, q, names, trajectory_time=times)
            coordinates = hand_state.coordinates
            rate = torch.zeros_like(coordinates)
            model = hand.nominal_finger_trajectory
            if model is not None:
                modeled = model.valid_mask_torch(times)
                _, model_rate = model.sample_torch(torch.where(modeled, times, 0.))
                rate = torch.where(modeled[..., None], model_rate, rate)
            columns = [hand.joint_names.index(name) for name in missing]
            q = torch.cat((q, coordinates[..., columns]), -1)
            dq = torch.cat((dq, rate[..., columns]), -1)
            names += tuple(missing)
    poses, velocities = kinematics.states(q, dq, names)
    position, quaternion = poses[problem.tool_link]
    linear, angular = velocities[problem.tool_link]
    tool_position = q.new_tensor(problem.tool_transform[:3, 3])
    tool_quaternion = q.new_tensor(_quaternion(problem.tool_transform))
    offset = quaternion_rotate(quaternion, tool_position.expand_as(position))
    tool = (position + offset, quaternion_multiply(quaternion, tool_quaternion))
    twist = (linear + torch.linalg.cross(angular, offset, dim=-1), angular)
    return q, dq, names, poses, tool, twist
# END saved candidates retain their own times and nominal hand model without rebinding.


class TrajectoryComputation:
    """Bind the current request to one differentiable CPU trajectory function.

    ``parameters`` is normalized physical time. It includes every declared
    task/scene anchor and ordinary integration samples. ``progress`` retains
    their original skill coordinates. ``ordinary_indices`` selects the
    integration grid; adding an event anchor changes no motion-quality weight.
    All arrays use [one candidate, time, named joint]. Finger states are the
    scene's declared nominal model, including its explicit uncertainty scope.
    """

    def __init__(self, *, problem, scene, kinematics, domain, joint_names,
                 parameters, progress, ordinary_indices, motion_configuration,
                 native_dt, constraint_weight, max_host_rss_bytes):
        self.problem, self.scene, self.kinematics, self.domain = problem, scene, kinematics, domain
        self.joint_names = tuple(joint_names)
        self.parameters = np.asarray(parameters, dtype=float)
        self.progress = torch.as_tensor(np.asarray(progress, dtype=float))[None]
        self.duration = domain.intervals * domain._curve_map.knot_dt
        self.times = torch.as_tensor(self.parameters * self.duration)[None]
        issues = problem.verification_issues(self.progress)
        if issues:
            raise ValueError(f"Fixed task query is incomplete: {issues}")
        self.ordinary_indices = np.asarray(ordinary_indices, dtype=int)
        self.time_weights = torch.as_tensor(time_mean_weights(self.times.numpy()[0, self.ordinary_indices]))[None]
        self.motion_configuration, self.native_dt = motion_configuration, float(native_dt)
        self.constraint_weight = float(constraint_weight)
        self.max_host_rss_bytes = max_host_rss_bytes
        self.rows = tuple(tuple(torch.as_tensor(array) for array in domain._curve_map.query(self.parameters, order))
                          for order in range(4))
        # BEGIN eliminate only equations whose declared pose is fixed by q(0)=q0.
        # This mask depends on the affine curve representation and fixed query
        # layout, never on a trial Jacobian. CubicAffineMap has a zero control
        # row and exactly q0 as its affine offset at physical time zero.
        position_rows, position_offsets = self.rows[0]
        fixed_pose = ((self.times[0] == 0.) & (position_rows == 0.).all(-1)
                      & (position_offsets == torch.as_tensor(domain._curve_map.start_q)).all(-1))
        self.equality_keys, fixed_rows = [], []
        for index, requirement in enumerate(problem.pose_events):
            if (problem.start_progress <= requirement.progress <= problem.end_progress
                    and requirement.component_tolerances.get("yaw") == 0.):
                self.equality_keys.append(f"function_pose:{index}:yaw")
                occurrence = _first_event_query(self.progress, requirement.progress, self.times)[0]
                fixed_rows.append(bool((occurrence & fixed_pose).any()))
        self.fixed_equality_rows = np.flatnonzero(fixed_rows)
        self.variable_equality_rows = np.flatnonzero(~np.asarray(fixed_rows, dtype=bool))
        # END complete task() output and saved-curve acceptance retain every row.
        self.bounds = torch.as_tensor(domain._constraint_bounds)
        self.geometry_report = {}

    def states(self, controls):
        """Use the same affine curve for q and all three physical derivatives."""
        return tuple((row @ controls + offset)[None] for row, offset in self.rows)

    def robot_states(self, q, dq):
        """Complete model joints and compose the calibrated reference-frame TCP."""
        return robot_tool_states(problem=self.problem, scene=self.scene, kinematics=self.kinematics,
            q=q, dq=dq, joint_names=self.joint_names, progress=self.progress, times=self.times)

    def task(self, controls):
        """Return the original signed task rows, preserving fixed dictionary order."""
        q, dq, _, _ = self.states(controls)
        q, dq, names, _, tool, twist = self.robot_states(q, dq)
        groups = self.problem.constraint_components(*tool, self.progress, q, names, dq,
            trajectory_time=self.times, tool_linear_velocity=twist[0], tool_angular_velocity=twist[1])
        empty = controls.reshape(-1)[:0]
        return tuple(torch.cat([value.reshape(-1) for value in group.values()]) if group else empty
                     for group in groups)

    def objective_components(self, controls, *, include_geometry=True):
        """Motion quality and declared task preferences share the same candidate."""
        states = self.states(controls)
        q, dq, names, poses, tool, twist = self.robot_states(*states[:2])
        ordinary = tuple(value[:, self.ordinary_indices] for value in states)
        costs = {"motion_quality": motion_cost(ordinary, self.bounds, self.motion_configuration,
                                               self.native_dt, self.time_weights)}
        preferences = self.problem.optimizer_preferences(*tool, self.progress,
            trajectory_time=self.times, tool_linear_velocity=twist[0], tool_angular_velocity=twist[1],
            phase="quality")
        costs["task_quality"] = self.constraint_weight * preferences.sum()
        if include_geometry:
            costs["observed_geometry"] = self.constraint_weight * self.scene.optimization_objective(
                *tool, self.progress, q, names, trajectory_time=self.times,
                robot_link_poses=poses, report_out=self.geometry_report)
        return costs


@dataclass
class ConstrainedSolution:
    """Library termination and proposed controls; acceptance is a separate query."""
    controls: np.ndarray
    report: dict


# BEGIN scalar control-column JVPs reuse one reverse graph for all task rows.
def task_jacobian(values, controls, *, check_request, admit_work):
    """Return the complete first derivative using ordinary Torch gradients.

    For the original task vector c(x) and an independent covector v, the
    first reverse pass forms p(x,v)=J_c(x).T @ v. Differentiating each p_i
    with respect to v gives column i of J_c. Only v is differentiated in
    that second pass: this requires no second derivative of robot FK with
    respect to x. Both passes use maintained scalar ``autograd.grad``.

    The explicit zero dependence gives constant task rows their exact zero
    Jacobian, including requests containing only fixed progress rows. One
    covector and one returned column have task-vector size; no task-row
    identity matrix or batched robot Jacobian is allocated. ``admit_work``
    accounts for the now-resident first reverse graph before allocating the
    final dense matrix and running its column queries.
    """
    covector = torch.zeros_like(values, requires_grad=True)
    connected = values + controls.sum()*0.
    product, = torch.autograd.grad(connected, controls, grad_outputs=covector, create_graph=True)
    admit_work()
    jacobian = np.empty((values.numel(), controls.numel()))
    for index, value in enumerate(product.reshape(-1)):
        check_request()
        column, = torch.autograd.grad(value, covector, retain_graph=index+1 < controls.numel())
        jacobian[:, index] = column.detach().numpy()
        del column
    return jacobian
# END the numerical problem retains every original row and physical unit.


# BEGIN capacity follows the installed dense solver and serial reverse derivative.
def solver_storage_bytes(*, variables, task_equalities, task_inequalities, physical_inequalities):
    """Count simultaneously live float64 arrays for this SLSQP interface.

    The workspace expression is the one in installed scipy.optimize._slsqp_py.
    Its constraint-normal assembly retains the previous matrix while forming
    equality/inequality stacks, their combined stack and the augmented matrix.
    Cached Jacobians are separate. Scalar control-column derivatives avoid
    the m² identity and batched robot tensors. The first reverse graph is
    included in measured RSS at the second allocation boundary.
    This count covers these owned dense buffers; current RSS accounts for the
    already constructed curve, model, observations and forward task tensors.
    """
    n, meq = int(variables), int(task_equalities)
    task = meq + int(task_inequalities)
    m = task + int(physical_inequalities)
    n1, la = n+1, max(1, m)
    mineq = m-meq+2*n1
    workspace = ((3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2)+2*mineq
                 +(n1+mineq)*(n1-meq)+2*meq+n1+((n+1)*n)//2
                 +2*m+3*n+3*n1+1)
    return {
        "slsqp_workspaces": 8*(workspace+mineq)+4*mineq,
        "cached_constraint_values_and_jacobians": 8*m*(n+1),
        "constraint_normal_assembly": 8*(2*m*n+2*la*(n+1)+la),
        "constraint_value_assembly": 8*3*m,
        # x0, shared cache x, solver x, two bound vectors; ScalarFunction x/g;
        # MemoizeJac x/g; wrapped gradient plus terminal zero; controls and VJP;
        # retained initial objective gradient and its scaled callback result.
        "solver_and_derivative_vectors": 8*(14*n+1+task),
        # Independent task covector, connected original task vector, returned
        # task column, and the first reverse product in control coordinates.
        "task_dual_vectors": 8*(3*task+n),
    }
# END no output-space standard basis is allocated for task derivatives.


def solve_trajectory(computation, initial, *, iterations, check_request, pending_geometry_bytes):
    """Submit one full problem to SciPy SLSQP, with exact first derivatives.

    SLSQP may evaluate infeasible trials and starts; the original constraints
    decide feasibility. Its stopping tolerance is a numerical search setting,
    not a task or geometric allowance. Candidate validation retains every
    original physical bound and task tolerance regardless of library status.

    ``pending_geometry_bytes`` is the scene's current unallocated material
    storage plus one live geometry/FK evaluation. Existing field/model arrays
    and previously constructed material samples already belong to measured
    RSS. The caller recomputes this quantity before each serial seed solve.
    """
    initial = np.asarray(initial, dtype=float)
    shape = initial.shape
    cache = {}
    cache_x = None
    counts = {"objective_evaluations": 0, "task_evaluations": 0, "physical_evaluations": 0}
    started = perf_counter()

    # Establish fixed row counts with ordinary FK before allocating any dense
    # Jacobian or SLSQP workspace. Continuous cubic rows comprise two position,
    # two velocity, four acceleration-endpoint and two jerk rows per segment
    # and joint. Every task declaration retains its own original signed rows.
    check_request()
    initial_controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    initial_task = computation.task(initial_controls)
    counts["task_evaluations"] += 1
    full_eq_count, inequality_count = tuple(value.numel() for value in initial_task)
    fixed_values = initial_task[0][computation.fixed_equality_rows].detach().numpy()
    fixed_report = {computation.equality_keys[index]: float(value)
        for index, value in zip(computation.fixed_equality_rows, fixed_values)}
    report = {"backend": "scipy.optimize.SLSQP", "task_query_count": len(computation.parameters),
              "complete_task_equalities": full_eq_count, "fixed_entry_equalities": fixed_report,
              "solver_task_equalities": len(computation.variable_equality_rows)}
    # An immutable violated equality makes this original request infeasible.
    # Verify its original zero-width set before removing its constant solver
    # row. No angular allowance or numerical-Jacobian rank rule is introduced.
    if not np.all(np.isfinite(fixed_values) & (fixed_values == 0.)):
        report.update(success=False, status=None, message="Fixed entry task equalities are infeasible",
            solver_called=False, iterations=0, objective=None, elapsed_seconds=perf_counter()-started,
            maximum_control_update=0., unchanged_from_seed=True, **counts)
        return ConstrainedSolution(initial.copy(), report)
    eq_count = len(computation.variable_equality_rows)
    physical_count = 10*computation.domain.intervals*shape[1]
    memory = solver_storage_bytes(variables=initial.size, task_equalities=eq_count,
        task_inequalities=inequality_count, physical_inequalities=physical_count)
    # The scalar Pin bridge retains one [time,frame,6,joint] product,
    # two [time,frame,6] spatial covectors, one [time,joint] accumulation
    # and two scalar reductions at its largest reverse allocation. Existing
    # native derivative arrays and the first reverse graph belong to RSS.
    samples = len(computation.parameters)
    frames = len(computation.kinematics.frames)
    joints = len(computation.kinematics.joint_names)
    memory["physical_sampling_rows"] = 8*8*computation.domain.intervals*shape[1]*shape[0]
    memory["pinocchio_reverse_buffers"] = 8*(samples*(6*frames*joints+12*frames+joints)+2)
    # BEGIN geometry and constrained-solver buffers coexist in each trial.
    # Material samples may still be lazy at the first objective evaluation.
    # Admit that storage and the live pair/FK graph with the dense solver;
    # separate admissible increments do not bound their simultaneous peak.
    memory["pending_geometry_bytes"] = pending_geometry_bytes
    require_host_memory(sum(memory.values()), pending_existing_bytes=0,
                        max_rss_bytes=computation.max_host_rss_bytes)

    def admit_task_derivatives():
        # The task values, independent covector and first reverse graph now
        # belong to measured RSS. Later callbacks also retain the allocated
        # SLSQP workspace and constructed scene samples. Only the new dense
        # matrix, one returned column and Pin reverse temporaries are pending.
        rows = eq_count + inequality_count
        increment = 8*rows*(initial.size+1) + memory["pinocchio_reverse_buffers"]
        require_host_memory(increment, pending_existing_bytes=0,
                            max_rss_bytes=computation.max_host_rss_bytes)
    # END resident source geometry is counted by RSS, never added again here.
    del initial_task, initial_controls

    def evaluate(kind, x):
        nonlocal cache_x
        check_request()
        if cache_x is None or not np.array_equal(x, cache_x):
            cache.clear()
            cache_x = x.copy()
        if kind in cache:
            return cache[kind]
        counts[kind + "_evaluations"] += 1
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
                eq, inequality = computation.task(controls)
                eq = eq[computation.variable_equality_rows]
                sizes = (eq.numel(), inequality.numel())
                values = torch.cat((eq, inequality))
                jacobian = task_jacobian(values, controls, check_request=check_request,
                                         admit_work=admit_task_derivatives)
                output = (values.detach().numpy(), jacobian, sizes)
        cache[kind] = output
        check_request()
        return output

    x0 = initial.ravel().copy()
    # BEGIN one positive constant changes numerical units, not objective order.
    # Production task weights can make the initial gradient O(1e10), which
    # destabilizes SLSQP's first quadratic subproblem. Fix this divisor once;
    # every trial value and derivative use the same factor, while reporting
    # and candidate ranking retain the original physical objective.
    initial_value, initial_gradient = evaluate("objective", x0)
    objective_scale = max(1., abs(initial_value), float(np.abs(initial_gradient).max()))

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
    result = minimize(scaled_objective, x0, jac=True, method="SLSQP",
        constraints=constraints, callback=lambda _: check_request(),
        options={"maxiter": int(iterations), "ftol": 1e-9, "disp": False})
    report.update({"success": bool(result.success), "solver_called": True,
              "status": int(result.status), "message": str(result.message),
              "iterations": int(result.nit), "objective": float(result.fun)*objective_scale,
              "objective_scale": objective_scale, "normalized_objective": float(result.fun),
              "initial_objective": initial_value,
              "initial_gradient_linf": float(np.abs(initial_gradient).max()),
              "elapsed_seconds": perf_counter() - started, **counts,
              "maximum_control_update": float(np.max(np.abs(result.x - x0))),
              "unchanged_from_seed": bool(np.array_equal(result.x, x0)),
              "solver_storage_bytes": memory,
              "task_jacobian_method": "scalar_control_columns_from_covector_reverse_graph",
              "geometry_objective": computation.geometry_report.copy()})
    return ConstrainedSolution(result.x.reshape(shape).copy(), report)
# END all numerical search belongs to the maintained solver, not a custom loop.
