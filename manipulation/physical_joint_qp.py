"""Small CPU quadratic programs on the existing fixed-clock cubic curves.

The cubic action is an affine curve with an immutable entrance and clock.
Position, velocity, acceleration and jerk rows constrain both seed fits and
descent steps. Analytic extrema and the same tensor domain determine
physical feasibility. These routines run only outside CUDA capture.
"""

# BEGIN fixed-clock affine sampling and continuous-domain constrained programs.
from __future__ import annotations

import numpy as np
import quadprog
from time import perf_counter
from scipy.interpolate import BSpline, PPoly
from scipy.optimize import linprog
import torch

# BEGIN one coefficient definition serves CPU programs and tensor rollouts.
from .cubic_trajectory import CubicBasis
# END no native CUDA trajectory kernel participates in this representation.


# BEGIN actual program calls and named computation phases share one report schema.
PROGRAM_WORK_FIELDS = ("qp_calls", "lp_calls", "extrema_seconds", "assembly_seconds",
                       "qp_solve_seconds", "lp_solve_seconds", "host_transfer_seconds")


def program_work_report():
    """Start counters for actual library calls and measured computation phases."""
    return {name: 0 if name.endswith("calls") else 0. for name in PROGRAM_WORK_FIELDS}


def _numpy(value, report=None):
    if not isinstance(value, torch.Tensor):
        return np.asarray(value, dtype=float)
    started = perf_counter()
    result = value.detach().cpu().numpy().astype(float)
    if report is not None:
        report["host_transfer_seconds"] += perf_counter()-started
    return result


def _tensor(value, *, dtype, device, report):
    """Measure the explicit NumPy-to-native-array handoff, including dtype copy."""
    started = perf_counter()
    result = torch.as_tensor(value, dtype=dtype, device=device)
    report["host_transfer_seconds"] += perf_counter()-started
    return result
# END timings describe these phases, not an inferred decomposition of whole-request time.


# BEGIN all four physical derivatives share one two-sided QP.
def _solve_qp(hessian, linear, rows, lower, upper, *, report, check_request=None):
    """Solve two-sided linear bounds in the free-coefficient space."""
    started = perf_counter()
    nonzero = np.any(rows != 0., axis=1)
    if np.any(lower[~nonzero] > 0.) or np.any(upper[~nonzero] < 0.):
        report["assembly_seconds"] += perf_counter()-started
        return None
    rows, lower, upper = rows[nonzero], lower[nonzero], upper[nonzero]
    matrix = np.concatenate((rows.T, -rows.T), axis=1)
    rhs = np.r_[lower, -upper]
    report["assembly_seconds"] += perf_counter()-started
    if check_request is not None:
        check_request()
    started = perf_counter()
    report["qp_calls"] += 1
    try:
        # quadprog minimizes .5*x'G*x-a'x with C'x >= b. Physical
        # constraints have two sides; unconstrained free coefficients do not
        # receive the robot's joint-position box.
        result = quadprog.solve_qp(np.asarray(hessian, dtype=float), -linear, matrix, rhs)[0]
    except ValueError as error:
        if "constraints are inconsistent" not in str(error):
            raise
        result = None
    finally:
        report["qp_solve_seconds"] += perf_counter()-started
    # Unexpected numerical exceptions propagate from the library. A normal
    # result, including its declared inconsistent constraints, reaches this
    # same request check before another query or trial can start.
    if check_request is not None:
        check_request()
    return result
# END the measured entrance is already fixed by the cubic coefficient map.


class CubicAffineMap(CubicBasis):
    """Bind measured q0/dq0 to the shared instantaneous-boundary cubic."""

    # BEGIN exactly two measured equalities replace four fixed coefficients.
    def __init__(self, start_q, start_dq, knots, knot_dt):
        super().__init__(knots, knot_dt)
        self.start_q, self.start_dq = _numpy(start_q).reshape(-1), _numpy(start_dq).reshape(-1)
        if (self.start_q.shape != self.start_dq.shape
                or not np.isfinite(self.start_q).all() or not np.isfinite(self.start_dq).all()):
            raise ValueError("Cubic entrance needs matching finite measured q0/dq0")
        # BEGIN fixed physical rows belong to this q0/dq0 and knot-clock binding.
        # q, dq and ddq use every original break; jerk uses each segment's
        # midpoint so both sides of a jerk jump remain represented. Moving
        # extrema add their own rows locally to each QP/LP invocation.
        self.fixed_constraint_times = tuple(np.arange(self.intervals+1, dtype=float)
            if order < 3 else np.arange(self.intervals, dtype=float)+.5 for order in range(4))
        self.fixed_constraint_rows = tuple(self.query_knot_times(times, order)
            for order, times in enumerate(self.fixed_constraint_times))
        # END later iterations reuse these arrays without rebuilding their basis.

    def coefficients(self, actions):
        """Complete coefficients, computed relative to q0 before translation."""
        actions = _numpy(actions)
        return (self.start_q + np.einsum("cn,...nj->...cj", self.matrix, actions-self.start_q)
                + self.velocity_coefficients[:, None]*self.knot_dt*self.start_dq)

    def query(self, progress, derivative=0):
        weights, velocity = self.weights(_numpy(progress), derivative)
        return self._affine_query(weights, velocity, derivative)

    def query_knot_times(self, times, derivative=0):
        """Build rows at original segment coordinates, including jerk sides."""
        weights, velocity = self.weights_at_knot_times(_numpy(times), derivative)
        return self._affine_query(weights, velocity, derivative)

    def _affine_query(self, weights, velocity, derivative):
        offset = velocity[:, None]*self.start_dq-weights.sum(-1, keepdims=True)*self.start_q
        if derivative == 0:
            offset = offset+self.start_q
        return weights, offset

    def evaluate(self, actions, progress, derivative=0):
        actions = _numpy(actions)
        if actions.shape[-2:] != (self.knots, len(self.start_q)):
            raise ValueError("Cubic sampling requires the native free-control layout")
        weights, velocity = self.weights(_numpy(progress), derivative)
        values = (np.einsum("tn,...nj->...tj", weights, actions-self.start_q)
                  + velocity[:, None]*self.start_dq)
        return values+self.start_q if derivative == 0 else values
    # END all derivatives act on relative controls; public actions stay positions.

    def extrema_times(self, controls, joint, derivative=0):
        """Return exact bounds witnesses for derivative 0..3, in knot units.

        Query times use knot units. Values use the model's joint coordinates
        and seconds. Jerk is constant on each segment, so its midpoint
        supplies every segment value without losing a side of a knot jump.
        """
        # BEGIN form derivatives before adding the absolute entrance angle.
        coefficients = (self.matrix@(controls-self.start_q[joint])
                        +self.velocity_coefficients*self.knot_dt*self.start_dq[joint])
        polynomial = PPoly.from_spline(BSpline(self.knot_vector, coefficients, 3))
        entrance = np.flatnonzero(polynomial.x[:-1] == 0.)
        polynomial.c[-1, entrance] = 0.
        polynomial.c[-2, entrance] = self.knot_dt*self.start_dq[joint]
        # Four identical final coefficients represent an exactly stopped
        # interval; summing large signed basis terms must not create motion.
        terminal = polynomial.x[:-1] >= self.intervals-1
        polynomial.c[:-1, terminal] = 0.
        polynomial.c[-1, terminal] = controls[-1]-self.start_q[joint]
        if derivative:
            polynomial = polynomial.derivative(derivative)
        if derivative == 3:
            times = np.arange(self.intervals, dtype=float)+.5
        else:
            roots = polynomial.derivative().roots(extrapolate=False)
            roots = roots[np.isfinite(roots) & (roots >= 0.) & (roots <= self.intervals)]
            times = np.unique(np.r_[np.arange(self.intervals+1), roots])
        values = polynomial(times)/(self.knot_dt**derivative)
        if derivative == 0:
            values = values+self.start_q[joint]
            values[times == 0.] = self.start_q[joint]
        elif derivative == 1:
            values[times == 0.] = self.start_dq[joint]
        return times, values
        # END exact boundary states are preserved without a numerical tolerance.

    # BEGIN only new analytic witness times generate new affine rows.
    def _initial_constraint_rows(self, initial, bounds, joint, report):
        """Combine bound-time rows with this starting curve's extra extrema."""
        known, matrices, lower, upper = [], [], [], []
        for order, (lo_bound, hi_bound) in enumerate(bounds):
            started = perf_counter()
            query, _ = self.extrema_times(initial, joint, order)
            report["extrema_seconds"] += perf_counter()-started
            started = perf_counter()
            fixed_times = self.fixed_constraint_times[order]
            weights, fixed = self.fixed_constraint_rows[order]
            extra = np.setdiff1d(query, fixed_times)
            times, offset = fixed_times, fixed[:, joint]
            if len(extra):
                new_weights, new_fixed = self.query_knot_times(extra, order)
                times = np.r_[fixed_times, extra]
                indices = np.argsort(times)
                times = times[indices]
                weights = np.concatenate((weights, new_weights))[indices]
                offset = np.r_[offset, new_fixed[:, joint]][indices]
            known.append(times)
            matrices.append(weights)
            lower.append(lo_bound-offset)
            upper.append(hi_bound-offset)
            report["assembly_seconds"] += perf_counter()-started
        started = perf_counter()
        result = known, np.concatenate(matrices), np.concatenate(lower), np.concatenate(upper)
        report["assembly_seconds"] += perf_counter()-started
        return result

    def _violated_constraint_rows(self, controls, bounds, joint, known, report):
        """Check the actual curve and assemble only previously absent bad extrema."""
        violation, matrices, lower, upper = False, [], [], []
        for order, (lo_bound, hi_bound) in enumerate(bounds):
            started = perf_counter()
            query, values = self.extrema_times(controls, joint, order)
            report["extrema_seconds"] += perf_counter()-started
            started = perf_counter()
            bad = (values < lo_bound) | (values > hi_bound)
            violation |= bool(np.any(bad))
            extra = np.setdiff1d(query[bad], known[order])
            if len(extra):
                weights, fixed = self.query_knot_times(extra, order)
                matrices.append(weights)
                lower.append(lo_bound-fixed[:, joint])
                upper.append(hi_bound-fixed[:, joint])
                known[order] = np.r_[known[order], extra]
            report["assembly_seconds"] += perf_counter()-started
        started = perf_counter()
        added = ((np.concatenate(matrices), np.concatenate(lower), np.concatenate(upper))
                 if matrices else None)
        report["assembly_seconds"] += perf_counter()-started
        return violation, added
    # END no tolerance changes the identity of an already assembled witness time.

    def solve_joint(self, hessian, linear, bounds, joint, *, origin,
                    initial, native_dtype, step_limit=None, check_request=None, work_report=None):
        """Exchange violated analytic extrema for linear QP rows, at most 24 times.

        The optimization variable is a coefficient increment from ``origin``.
        For fitting, the origin is zero. For descent it is the current legal
        action. Checking the rounded candidate, rather than the double QP
        answer, also exposes violations introduced by the actual action dtype.
        """
        # The caller owns this optional diagnostic dictionary so a completed
        # native call remains visible when the following request check raises.
        report = {} if work_report is None else work_report
        report.update(program_work_report(), rounds=0, constraint_count=0)
        known, rows, lo, hi = self._initial_constraint_rows(initial, bounds, joint, report)
        started = perf_counter()
        offset = rows@origin
        lo, hi = lo-offset, hi-offset
        if step_limit is not None:
            rows = np.r_[rows, np.eye(self.knots)]
            lo, hi = np.r_[lo, -step_limit], np.r_[hi, step_limit]
        report["assembly_seconds"] += perf_counter()-started
        for iteration in range(24):
            report.update(rounds=iteration+1, constraint_count=int(2*len(rows)))
            increment = _solve_qp(hessian, linear, rows, lo, hi,
                                 report=report, check_request=check_request)
            if increment is None:
                return None, {**report, "reason": "finite_constraints_inconsistent"}
            # Match base + direction in the actual float dtype. In particular,
            # float32 addition is not the cast of a float64 coefficient sum.
            rounded = (origin.astype(native_dtype)+increment.astype(native_dtype)).astype(float)
            violation, added = self._violated_constraint_rows(rounded, bounds, joint, known, report)
            if not violation:
                return increment, {**report, "reason": "analytic_extrema_passed"}
            if added is None:
                # A repeated rounded boundary cannot yield a new cutting row.
                # The caller must still restore native-dtype feasibility:
                # shorten a descent direction, or contract a fit toward its
                # legal anchor, using the existing exact tensor limiter.
                return increment, {**report, "reason": "rounded_boundary_repeated"}
            started = perf_counter()
            new_rows, new_lo, new_hi = added
            offset = new_rows@origin
            rows = np.concatenate((rows, new_rows))
            lo, hi = np.r_[lo, new_lo-offset], np.r_[hi, new_hi-offset]
            report["assembly_seconds"] += perf_counter()-started
        return increment, {**report, "reason": "extrema_exchange_limit"}


def projected_descent_step(domain, base, gradient, step_limits, scales, curve=None, *,
                           check_request=None, work_report=None):
    """Project the complete negative gradient into the physical displacement set.

    Physical constraints are separable by joint, so the small diagonal QPs
    are solved serially. The original native step limits bound *increments*,
    not cubic coefficients. The optimizer must separately demand actual cost
    decrease before accepting any returned direction.
    """
    zero = torch.zeros_like(base)
    report = {} if work_report is None else work_report
    report.update(program_work_report(), success=False, reason="invalid_descent_input", rounds=0,
                  constraint_count=0, gradient_dot_step=0., limiter_scale=0.,
                  per_joint_limiter_scale=[])
    if base.ndim != 3 or base.shape[0] != 1 or gradient.shape != base.shape:
        raise ValueError("Physical-domain descent requires one matching action/gradient batch")
    if not bool(domain.feasible(base).all()):
        raise ValueError("Physical-domain descent requires a legal current action")
    domain.validate_scales(scales)
    x, g, limits = _numpy(base, report)[0], _numpy(gradient, report)[0], _numpy(step_limits, report)
    limits = np.broadcast_to(limits, x.shape)
    maximum = float(scales.max())
    if (not np.isfinite(g).all() or not np.isfinite(limits).all()
            or np.any(limits <= 0.) or maximum <= 0.):
        return zero, report
    # BEGIN the complete gradient defines the Euclidean projection metric.
    # The farthest native trial is bounded by the existing displacement box.
    # Dividing all coordinates by one large gradient component would change
    # this subproblem and suppress legal motion in unrelated coordinates.
    target = -g
    # END physical bounds constrain the direction, not the gradient magnitude.
    increments = np.empty_like(x)
    native_dtype = torch.empty((), dtype=base.dtype).numpy().dtype
    if curve is None:
        lower, upper = _numpy(domain.lower, report), _numpy(domain.upper, report)
    else:
        bounds = _numpy(domain.bounds, report)
    hessian = np.eye(x.shape[0])
    for joint in range(x.shape[1]):
        detail = {**program_work_report(), "rounds": 0, "constraint_count": 0}
        try:
            if curve is None:
                detail["rounds"] = 1
                started = perf_counter()
                rows = np.r_[hessian, hessian]
                lo = np.r_[(lower[joint]-x[:, joint])/maximum, -limits[:, joint]/maximum]
                hi = np.r_[(upper[joint]-x[:, joint])/maximum, limits[:, joint]/maximum]
                detail.update(constraint_count=int(2*len(rows)))
                detail["assembly_seconds"] += perf_counter()-started
                increment = _solve_qp(hessian, -target[:, joint], rows, lo, hi,
                                     report=detail, check_request=check_request)
                detail["reason"] = ("finite_constraints_inconsistent" if increment is None
                                    else "position_qp_solved")
            else:
                # Solve the furthest native trial as a displacement; convert
                # back only after the continuous-curve checks. The owned
                # detail records calls even when a request check interrupts.
                increment, outcome = curve.solve_joint(hessian, -maximum*target[:, joint],
                    bounds[:, :, joint], joint, origin=x[:, joint],
                    initial=x[:, joint], native_dtype=native_dtype,
                    step_limit=limits[:, joint], check_request=check_request, work_report=detail)
                detail.update(outcome)
                if increment is not None:
                    increment = increment/maximum
        finally:
            report["rounds"] = max(report["rounds"], detail["rounds"])
            report["constraint_count"] += detail["constraint_count"]
            for name in PROGRAM_WORK_FIELDS:
                report[name] += detail[name]
        if increment is None:
            return zero, {**report, "reason": "direction_qp_not_constructed",
                          "joint_column": joint, "joint_qp_reason": detail["reason"]}
        increments[:, joint] = increment
    proposed = _tensor(increments, dtype=base.dtype, device=base.device, report=report)[None]
    # Physical domains and these diagonal QPs are separable by joint. Correct
    # float-dtype rounding for each joint's *complete curve increment* before
    # combining them; a roundoff-sized outward component in one joint must
    # not shrink valid motion in every other joint. This preserves all ratios
    # between a joint's free controls and never imposes a control-point box.
    dof = base.shape[-1]
    separate = proposed*torch.eye(dof, dtype=base.dtype, device=base.device)[:, None, :]
    corrected = domain.limit_step(base.expand(dof, -1, -1), separate, scales)
    proposed_norms = separate.to(torch.float64).flatten(1).norm(dim=1)
    corrected_norms = corrected.to(torch.float64).flatten(1).norm(dim=1)
    joint_scales = torch.where(proposed_norms > 0.,
        corrected_norms/proposed_norms.clamp_min(torch.finfo(torch.float64).tiny),
        torch.ones_like(proposed_norms))
    combined = corrected.sum(dim=0, keepdim=True)
    # The same original limiter remains the final whole-direction check, in
    # the exact arithmetic/order used by all native trial multipliers.
    direction = domain.limit_step(base, combined, scales)
    actual = base+scales*direction[:, None]
    moved = bool((actual != base[:, None]).any())
    derivative = float((gradient.to(torch.float64)*direction.to(torch.float64)).sum())
    norm = float(combined.to(torch.float64).norm())
    report.update(gradient_dot_step=derivative,
        limiter_scale=float(direction.to(torch.float64).norm())/norm if norm else 0.,
        per_joint_limiter_scale=joint_scales.cpu().tolist())
    if not moved or not np.isfinite(derivative) or derivative >= 0.:
        return zero, {**report, "reason": "no_representable_descent_direction"}
    return direction, {**report, "success": True, "reason": "physical_descent_direction"}


# BEGIN an original-domain linear program constructs the common physical anchor.
def construct_cubic_anchor(domain, curve, dtype, device, *, check_request=None, report=None):
    """Maximize coefficient-space room inside the original physical bounds.

    Each original row ``a @ C <= b`` becomes ``a @ C + ||a||_1*r <= b``
    with a shared radius r >= 0 for one joint. Maximizing r selects a centre
    that is less sensitive to coefficient rounding. The radius is an objective
    variable, not a required margin: r=0 retains every original boundary
    solution, including domains with no interior. The measured q0/dq0 and
    terminal stop remain fixed by the cubic map. Violated analytic extrema
    add rows, and actual action-dtype curves use the unchanged domain test.
    """
    if report is None:
        report = program_work_report()
    report.update(reason="anchor_not_constructed", joint_status=[], rounds=0,
                  constraint_count=0, control_radius=[])
    initial = np.broadcast_to(_numpy(domain._start_q, report), (domain.knots, len(domain.lower)))
    controls = np.empty_like(initial)
    native_dtype = torch.empty((), dtype=dtype).numpy().dtype
    bounds = _numpy(domain.bounds, report)
    for joint in range(initial.shape[-1]):
        known, rows, lower, upper = curve._initial_constraint_rows(
            initial[:, joint], bounds[:, :, joint], joint, report)
        started = perf_counter()
        norm = np.abs(rows).sum(axis=1, keepdims=True)
        matrix = np.r_[np.c_[rows, norm], np.c_[-rows, norm]]
        rhs = np.r_[upper, -lower]
        objective = np.r_[np.zeros(domain.knots), -1.]
        variable_bounds = [(None, None)]*domain.knots+[(0., None)]
        report["assembly_seconds"] += perf_counter()-started
        previous_constraints = report["constraint_count"]
        reason = "extrema_exchange_limit"
        for iteration in range(24):
            report["rounds"] = max(report["rounds"], iteration+1)
            report["constraint_count"] = previous_constraints+len(rhs)
            if check_request is not None:
                check_request()
            started = perf_counter()
            report["lp_calls"] += 1
            try:
                result = linprog(objective, A_ub=matrix, b_ub=rhs,
                    bounds=variable_bounds, method="highs")
            finally:
                report["lp_solve_seconds"] += perf_counter()-started
            if check_request is not None:
                check_request()
            if not result.success:
                reason = ("finite_constraints_inconsistent" if result.status == 2
                          else "anchor_linear_program_failed")
                return None, {**report, "reason": reason, "joint_column": joint,
                              "linear_program_status": int(result.status)}
            rounded = result.x[:-1].astype(native_dtype)
            # The native tensor predicate decides this joint's acceptance;
            # other columns carry the declared shape without deciding it.
            trial = initial.copy()
            trial[:, joint] = rounded
            action = _tensor(trial, dtype=dtype, device=device, report=report)[None]
            started = perf_counter()
            low, high = domain.all_extrema(action)
            valid = bool(((low[0, :, joint] >= domain.bounds[:, 0, joint])
                          & (high[0, :, joint] <= domain.bounds[:, 1, joint])).all())
            report["extrema_seconds"] += perf_counter()-started
            if valid:
                reason = "analytic_extrema_passed"
                break
            _, added = curve._violated_constraint_rows(
                rounded.astype(float), bounds[:, :, joint], joint, known, report)
            if added is None:
                reason = "rounded_anchor_outside_domain"
                break
            # Existing LP rows retain their numeric values. Only newly
            # violated extrema acquire coefficient-radius columns here.
            started = perf_counter()
            new_rows, new_lower, new_upper = added
            norm = np.abs(new_rows).sum(axis=1, keepdims=True)
            matrix = np.r_[matrix, np.c_[new_rows, norm], np.c_[-new_rows, norm]]
            rhs = np.r_[rhs, new_upper, -new_lower]
            report["assembly_seconds"] += perf_counter()-started
        report["joint_status"].append(reason)
        report["control_radius"].append(float(result.x[-1]))
        if reason != "analytic_extrema_passed":
            return None, {**report, "reason": reason, "joint_column": joint}
        controls[:, joint] = rounded
    anchor = _tensor(controls, dtype=dtype, device=device, report=report)[None]
    if not bool(domain.feasible(anchor).all()):
        return None, {**report, "reason": "rounded_anchor_outside_domain"}
    return anchor, {**report, "reason": "physical_anchor_constructed"}
# END anchor feasibility is the same predicate used by native optimization.


# BEGIN reference samples remain soft under the single continuous physical domain.
def fit_cubic_reference(domain, curve, progress, targets, prior_controls=None, *, check_request=None):
    """Fit soft references while preserving the measured entrance and clock.

    The complete bound declaration belongs to ``domain``. Native-dtype
    contraction uses its ordinary line limiter, so fitting, initialization
    and descent cannot accept different derivative domains.
    """
    from .physical_joint_domain import JointDomainInitializationError

    report = {**program_work_report(), "success": False, "reason": "fit_initializer_not_constructed",
        "rounds": 0, "constraint_count": 0, "joint_qp_status": [], "physical_derivative_orders": [0, 1, 2, 3]}
    values = _numpy(targets, report)
    started = perf_counter()
    weights, offset = curve.query(progress)
    report["assembly_seconds"] += perf_counter()-started
    count, dof = domain.knots, len(domain.lower)
    if values.shape != offset.shape or not len(values) or not np.isfinite(values).all():
        raise ValueError("Cubic reference targets must be finite [query, joint] values")
    prior = (np.zeros((count, dof)) if prior_controls is None else _numpy(prior_controls, report))
    if prior.shape != (count, dof) or not np.isfinite(prior).all():
        raise ValueError("Cubic reference prior must match the finite free-control layout")
    source = prior_controls if isinstance(prior_controls, torch.Tensor) else targets
    dtype = source.dtype if isinstance(source, torch.Tensor) else torch.float32
    device = source.device if isinstance(source, torch.Tensor) else domain.lower.device
    native_dtype = torch.empty((), dtype=dtype).numpy().dtype
    started = perf_counter()
    gram = weights.T@weights
    # A machine-precision-scaled ridge makes underdetermined observations
    # strictly convex. It is only a numerical tiebreaker, not another task cost.
    ridge = np.finfo(float).eps*max(1., float(np.trace(gram)))*64
    hessian = gram+ridge*np.eye(count)
    linear = weights.T@(offset-values)-ridge*prior
    controls = np.empty((count, dof))
    report["assembly_seconds"] += perf_counter()-started
    if not bool(domain._entrance_valid):
        return None, {**report, "reason": "algebraic_entrance_invalid"}
    bounds = _numpy(domain.bounds, report)
    for joint in range(dof):
        candidate, detail = curve.solve_joint(hessian, linear[:, joint],
            bounds[:, :, joint], joint, origin=np.zeros(count),
            initial=prior[:, joint], native_dtype=native_dtype, check_request=check_request)
        report["rounds"] = max(report["rounds"], detail["rounds"])
        report["constraint_count"] += detail["constraint_count"]
        report["joint_qp_status"].append(detail["reason"])
        for name in PROGRAM_WORK_FIELDS:
            report[name] += detail[name]
        if candidate is None:
            return None, {**report, "reason": detail["reason"], "joint_column": joint}
        controls[:, joint] = candidate
    action = _tensor(controls, dtype=dtype, device=device, report=report)
    report.update(fit_contraction_scale=1., maximum_bound_excess_before_contraction=[0.]*4,
                  fit_anchor_source="unneeded")
    if not bool(domain.feasible(action[None]).all()):
        # A rounded optimum can cross an active bound. The common limiter
        # contracts the whole coefficient increment toward a feasible anchor
        # in the original arithmetic: anchor + scale * (action - anchor).
        # This produces an approximate fit and reports the actual change.
        low, high = domain.all_extrema(action[None])
        excess = torch.maximum(domain.bounds[:, 0]-low, high-domain.bounds[:, 1])
        try:
            if domain._anchor is None:
                try:
                    domain._anchor = domain._construct_anchor(dtype, device, check_request=check_request)
                finally:
                    # Charge a newly constructed anchor once. Reusing its
                    # already valid coefficients performs no new LP solve.
                    for name in PROGRAM_WORK_FIELDS:
                        report[name] += domain.initialization_report[name]
            anchor = domain._anchor.to(dtype=dtype, device=device)
            domain.validate(anchor)
            delta = action[None]-anchor
            limited = domain.limit_step(anchor, delta, action.new_ones((1, 1, 1, 1)))
            initialized = anchor+limited
            domain.validate(initialized)
        except JointDomainInitializationError as error:
            return None, {**report, "reason": "fit_initializer_not_constructed", "detail": str(error)}
        proposed_norm = float(delta.to(torch.float64).norm())
        accepted_norm = float(limited.to(torch.float64).norm())
        report.update(maximum_bound_excess_before_contraction=excess.clamp_min(0).amax(dim=(0, 2)).cpu().tolist(),
            fit_contraction_scale=accepted_norm/proposed_norm if proposed_norm else 0.,
            fit_anchor_source="physical_domain_anchor")
        action = initialized[0]
    numeric_action = _numpy(action, report)
    residual = curve.evaluate(numeric_action, progress)-values
    low, high = domain.all_extrema(action[None])
    return action, {**report, "success": True, "reason": "physical_reference_fit",
        "reference_rms_joint_units": float(np.sqrt(np.mean(residual**2))),
        "reference_max_error_joint_units": float(np.max(np.abs(residual))),
        "continuous_max_abs_derivatives": torch.maximum(low.abs(), high.abs())[0].cpu().tolist(),
        "query_max_abs_velocity_joint_units_per_s": np.max(np.abs(curve.evaluate(numeric_action, progress, 1)), axis=0).tolist(),
        "continuous_max_abs_velocity_joint_units_per_s": torch.maximum(low.abs(), high.abs())[0, 1].cpu().tolist()}
# END reference residuals describe the fitted curve in the model's joint units.
# END the complete curve validator and rollout own final task/geometry acceptance.
