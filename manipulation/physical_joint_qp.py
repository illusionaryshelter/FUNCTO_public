"""Small CPU quadratic programs on the existing fixed-clock cubic curves.

The cubic action is an affine curve with an immutable entrance and clock.
Position, velocity, acceleration and jerk rows constrain both seed fits and
recovery steps. Analytic extrema and the same tensor domain determine
physical feasibility. These routines run only outside CUDA capture.
"""

# BEGIN fixed-clock affine sampling and bounded continuous-domain QP recovery.
from __future__ import annotations

import numpy as np
import quadprog
from scipy.interpolate import BSpline, PPoly
import torch

# BEGIN one coefficient definition serves CPU programs and tensor rollouts.
from .cubic_trajectory import CubicBasis
# END no native CUDA trajectory kernel participates in this representation.


def _numpy(value):
    return (value.detach().cpu().numpy().astype(float) if isinstance(value, torch.Tensor)
            else np.asarray(value, dtype=float))


# BEGIN all four physical derivatives share one two-sided QP.
def _solve_qp(hessian, linear, rows, lower, upper):
    """Solve two-sided linear bounds in the free-coefficient space."""
    nonzero = np.any(rows != 0., axis=1)
    if np.any(lower[~nonzero] > 0.) or np.any(upper[~nonzero] < 0.):
        return None
    rows, lower, upper = rows[nonzero], lower[nonzero], upper[nonzero]
    try:
        # quadprog minimizes .5*x'G*x-a'x with C'x >= b. Physical
        # constraints have two sides; unconstrained free coefficients do not
        # receive the robot's joint-position box.
        return quadprog.solve_qp(np.asarray(hessian, dtype=float), -linear,
            np.concatenate((rows.T, -rows.T), axis=1), np.r_[lower, -upper])[0]
    except ValueError as error:
        if "constraints are inconsistent" not in str(error):
            raise
        return None
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

    def solve_joint(self, hessian, linear, bounds, joint, *, origin,
                    initial, native_dtype, step_limit=None, interior=False):
        """Exchange violated analytic extrema for linear QP rows, at most 24 times.

        The optimization variable is a coefficient increment from ``origin``.
        For fitting, the origin is zero. For recovery it is the current legal
        action. Checking the rounded candidate, rather than the double QP
        answer, also exposes violations introduced by the actual action dtype.
        """
        times = {derivative: self.extrema_times(initial, joint, derivative)[0]
                 for derivative in range(4)}
        # This small inward margin belongs only to construction of a
        # representable fit anchor. Physical acceptance keeps the exact model
        # limits, and optimizer recovery never activates this margin.
        coordinate_scale = max(1., float(np.max(np.abs(bounds[0]))), float(np.max(np.abs(initial))))
        roundoff = 32*np.finfo(native_dtype).eps*coordinate_scale if interior else 0.
        report = {"rounds": 0, "constraint_count": 0}
        for iteration in range(24):
            row_parts, lower_parts, upper_parts = [], [], []
            for derivative, (lo_bound, hi_bound) in enumerate(bounds):
                # BEGIN use the same exact entrance rows as tensor sampling.
                weights, fixed = self.query_knot_times(times[derivative], derivative)
                offset = fixed[:, joint]+weights@origin
                # END numerical cancellation cannot fabricate an entrance QP row.
                # BEGIN inward construction margins act only on free directions.
                margin = roundoff*np.abs(weights).sum(-1)
                # END the original physical acceptance bounds remain unchanged.
                row_parts.append(weights)
                lower_parts.append(lo_bound-offset+margin)
                upper_parts.append(hi_bound-offset-margin)
            rows, lo, hi = np.concatenate(row_parts), np.concatenate(lower_parts), np.concatenate(upper_parts)
            if step_limit is not None:
                rows = np.r_[rows, np.eye(self.knots)]
                lo, hi = np.r_[lo, -step_limit], np.r_[hi, step_limit]
            report.update(rounds=iteration+1, constraint_count=int(2*len(rows)))
            increment = _solve_qp(hessian, linear, rows, lo, hi)
            if increment is None:
                return None, {**report, "reason": "finite_constraints_inconsistent"}
            # Match base + direction in the actual float dtype. In particular,
            # float32 addition is not the cast of a float64 coefficient sum.
            rounded = (origin.astype(native_dtype)+increment.astype(native_dtype)).astype(float)
            violation, added = False, False
            for derivative, (lo_bound, hi_bound) in enumerate(bounds):
                query, values = self.extrema_times(rounded, joint, derivative)
                bad = (values < lo_bound) | (values > hi_bound)
                violation = violation or bool(np.any(bad))
                extended = np.unique(np.r_[times[derivative], query[bad]])
                added = added or len(extended) != len(times[derivative])
                times[derivative] = extended
            if not violation:
                return increment, {**report, "reason": "analytic_extrema_passed"}
            if not added:
                # A repeated rounded boundary cannot yield a new cutting row.
                # The caller must still restore native-dtype feasibility:
                # shorten a recovery direction, or contract a fit toward its
                # legal anchor, using the existing exact tensor limiter.
                return increment, {**report, "reason": "rounded_boundary_repeated"}
        return increment, {**report, "reason": "extrema_exchange_limit"}


def recover_position_step(domain, base, gradient, step_limits, scales, curve=None):
    """Project a scaled negative gradient, retaining independent joint motion.

    Physical constraints are separable by joint, so the small diagonal QPs
    are solved serially. The original native step limits bound *increments*,
    not cubic coefficients. The optimizer must separately demand actual cost
    decrease before accepting any returned direction.
    """
    zero = torch.zeros_like(base)
    report = {"success": False, "reason": "invalid_recovery_input", "rounds": 0,
              "constraint_count": 0, "gradient_dot_step": 0., "limiter_scale": 0.,
              "per_joint_limiter_scale": []}
    if base.ndim != 3 or base.shape[0] != 1 or gradient.shape != base.shape:
        raise ValueError("Physical-domain recovery requires one matching action/gradient batch")
    if not bool(domain.feasible(base).all()):
        raise ValueError("Physical-domain recovery requires a legal current action")
    domain.validate_scales(scales)
    x, g, limits = _numpy(base)[0], _numpy(gradient)[0], _numpy(step_limits)
    limits = np.broadcast_to(limits, x.shape)
    maximum = float(scales.max())
    if (not np.isfinite(g).all() or not np.isfinite(limits).all()
            or np.any(limits <= 0.) or maximum <= 0.):
        return zero, report
    # Match the native direction normalization before applying its largest
    # trial multiplier. Scaling one gradient preserves its descent sign.
    normalization = max(1., float(np.max(np.abs(g)/limits)))
    target = -g/normalization
    increments = np.empty_like(x)
    native_dtype = base.detach().cpu().numpy().dtype
    if curve is None:
        lower, upper = _numpy(domain.lower), _numpy(domain.upper)
    else:
        bounds = _numpy(domain.bounds)
    hessian = np.eye(x.shape[0])
    for joint in range(x.shape[1]):
        if curve is None:
            rows = np.r_[hessian, hessian]
            lo = np.r_[(lower[joint]-x[:, joint])/maximum, -limits[:, joint]/maximum]
            hi = np.r_[(upper[joint]-x[:, joint])/maximum, limits[:, joint]/maximum]
            increment = _solve_qp(hessian, -target[:, joint], rows, lo, hi)
            detail = {"rounds": 1, "constraint_count": int(2*len(rows)),
                      "reason": "finite_constraints_inconsistent" if increment is None else "position_qp_solved"}
        else:
            # Solve the furthest native trial as a displacement; convert back
            # to a direction only after the continuous-curve checks.
            increment, detail = curve.solve_joint(hessian, -maximum*target[:, joint],
                bounds[:, :, joint], joint, origin=x[:, joint],
                initial=x[:, joint], native_dtype=native_dtype,
                step_limit=limits[:, joint])
            if increment is not None:
                increment = increment/maximum
        report["rounds"] = max(report["rounds"], detail["rounds"])
        report["constraint_count"] += detail["constraint_count"]
        if increment is None:
            return zero, {**report, "reason": "recovery_qp_not_constructed",
                          "joint_column": joint, "joint_qp_reason": detail["reason"]}
        increments[:, joint] = increment
    proposed = torch.as_tensor(increments, dtype=base.dtype, device=base.device)[None]
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


# BEGIN one algebraic QP anchor serves every complete-domain entry point.
def construct_cubic_anchor(domain, curve, dtype, device):
    """Construct a native-dtype feasible curve using the shared four-order QP.

    This construction has no fit/initialize dependency. The position-valued
    constant proposal supplies a finite objective centre; the measured q0,
    dq0, stopped terminal interval and fixed clock remain in the affine map.
    A construction margin accounts for coefficient rounding only. Every
    returned action is checked against the original model bounds.
    """
    initial = np.broadcast_to(_numpy(domain._start_q), (domain.knots, len(domain.lower)))
    controls = np.empty_like(initial)
    native_dtype = torch.empty((), dtype=dtype).numpy().dtype
    bounds = _numpy(domain.bounds)
    report = {"reason": "anchor_not_constructed", "joint_qp_status": [], "rounds": 0,
              "constraint_count": 0}
    for joint in range(initial.shape[-1]):
        candidate, detail = curve.solve_joint(np.eye(domain.knots), -initial[:, joint],
            bounds[:, :, joint], joint, origin=np.zeros(domain.knots),
            initial=initial[:, joint], native_dtype=native_dtype, interior=True)
        report["rounds"] = max(report["rounds"], detail["rounds"])
        report["constraint_count"] += detail["constraint_count"]
        report["joint_qp_status"].append(detail["reason"])
        if candidate is None:
            return None, {**report, "reason": detail["reason"], "joint_column": joint}
        controls[:, joint] = candidate
    anchor = torch.as_tensor(controls, dtype=dtype, device=device)[None]
    if not bool(domain.feasible(anchor).all()):
        reason = ("extrema_exchange_limit" if "extrema_exchange_limit" in report["joint_qp_status"]
                  else "rounded_anchor_outside_domain")
        return None, {**report, "reason": reason}
    return anchor, {**report, "reason": "physical_anchor_constructed"}
# END anchor feasibility is the same predicate used by native optimization.


# BEGIN reference samples remain soft under the single continuous physical domain.
def fit_cubic_reference(domain, curve, progress, targets, prior_controls=None):
    """Fit soft references while preserving the measured entrance and clock.

    The complete bound declaration belongs to ``domain``. Native-dtype
    contraction uses its ordinary line limiter, so fitting, initialization
    and recovery cannot accept different derivative domains.
    """
    from .physical_joint_domain import JointDomainInitializationError

    values = _numpy(targets)
    weights, offset = curve.query(progress)
    count, dof = domain.knots, len(domain.lower)
    if values.shape != offset.shape or not len(values) or not np.isfinite(values).all():
        raise ValueError("Cubic reference targets must be finite [query, joint] values")
    prior = (np.zeros((count, dof)) if prior_controls is None else _numpy(prior_controls))
    if prior.shape != (count, dof) or not np.isfinite(prior).all():
        raise ValueError("Cubic reference prior must match the finite free-control layout")
    source = prior_controls if isinstance(prior_controls, torch.Tensor) else targets
    dtype = source.dtype if isinstance(source, torch.Tensor) else torch.float32
    device = source.device if isinstance(source, torch.Tensor) else domain.lower.device
    native_dtype = torch.empty((), dtype=dtype).numpy().dtype
    gram = weights.T@weights
    # A machine-precision-scaled ridge makes underdetermined observations
    # strictly convex. It is only a numerical tiebreaker, not another task cost.
    ridge = np.finfo(float).eps*max(1., float(np.trace(gram)))*64
    hessian = gram+ridge*np.eye(count)
    linear = weights.T@(offset-values)-ridge*prior
    controls = np.empty((count, dof))
    report = {"success": False, "reason": "fit_initializer_not_constructed", "rounds": 0,
              "constraint_count": 0, "joint_qp_status": [], "physical_derivative_orders": [0, 1, 2, 3]}
    if not bool(domain._entrance_valid):
        return None, {**report, "reason": "algebraic_entrance_invalid"}
    bounds = _numpy(domain.bounds)
    for joint in range(dof):
        candidate, detail = curve.solve_joint(hessian, linear[:, joint],
            bounds[:, :, joint], joint, origin=np.zeros(count),
            initial=prior[:, joint], native_dtype=native_dtype)
        report["rounds"] = max(report["rounds"], detail["rounds"])
        report["constraint_count"] += detail["constraint_count"]
        report["joint_qp_status"].append(detail["reason"])
        if candidate is None:
            return None, {**report, "reason": detail["reason"], "joint_column": joint}
        controls[:, joint] = candidate
    action = torch.as_tensor(controls, dtype=dtype, device=device)
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
                domain._anchor = domain._construct_anchor(dtype, device)
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
    residual = curve.evaluate(action, progress)-values
    low, high = domain.all_extrema(action[None])
    return action, {**report, "success": True, "reason": "physical_reference_fit",
        "reference_rms_joint_units": float(np.sqrt(np.mean(residual**2))),
        "reference_max_error_joint_units": float(np.max(np.abs(residual))),
        "continuous_max_abs_derivatives": torch.maximum(low.abs(), high.abs())[0].cpu().tolist(),
        "query_max_abs_velocity_joint_units_per_s": np.max(np.abs(curve.evaluate(action, progress, 1)), axis=0).tolist(),
        "continuous_max_abs_velocity_joint_units_per_s": torch.maximum(low.abs(), high.abs())[0, 1].cpu().tolist()}
# END reference residuals describe the fitted curve in the model's joint units.
# END the complete curve validator and rollout own final task/geometry acceptance.
