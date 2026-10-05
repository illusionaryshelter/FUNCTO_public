"""Small CPU quadratic programs on the existing fixed-clock cubic curves.

The cubic action is an affine curve with an immutable entrance and clock.
Position rows propose recovery steps; seed fits can additionally constrain
velocity. Analytic extrema and the exact tensor position domain remain the
acceptance oracle. These routines run only outside CUDA capture.
"""

# BEGIN fixed-clock affine sampling and bounded continuous-domain QP recovery.
from __future__ import annotations

import numpy as np
import quadprog
from scipy.interpolate import BSpline, PPoly
import torch


def _numpy(value):
    return (value.detach().cpu().numpy().astype(float) if isinstance(value, torch.Tensor)
            else np.asarray(value, dtype=float))


def _solve_qp(hessian, linear, rows, lower, upper):
    """Use the installed strictly-convex solver; never widen physical bounds."""
    nonzero = np.any(rows != 0., axis=1)
    if np.any(lower[~nonzero] > 0.) or np.any(upper[~nonzero] < 0.):
        return None
    rows, lower, upper = rows[nonzero], lower[nonzero], upper[nonzero]
    try:
        # quadprog minimizes .5*x'G*x-a'x with C'x >= b. Position
        # constraints have two sides; unconstrained free coefficients do not
        # receive the robot's joint-position box.
        return quadprog.solve_qp(np.asarray(hessian, dtype=float), -linear,
            np.concatenate((rows.T, -rows.T), axis=1), np.r_[lower, -upper])[0]
    except ValueError as error:
        if "constraints are inconsistent" not in str(error):
            raise
        return None


class CubicAffineMap:
    """cuRobo degree-three coefficients, including its exact boundary layout."""

    def __init__(self, prefix, knots, knot_dt):
        self.knots, self.knot_dt = knots, float(knot_dt)
        self.knot_vector = np.arange(-3., knots+8.)
        self.matrix = np.zeros((knots+7, knots))
        self.matrix[4:4+knots] = np.eye(knots)
        self.matrix[-3:, -1] = 1.
        self.prefix = np.zeros((knots+7, prefix.shape[-1]))
        self.prefix[:4] = _numpy(prefix)
        self.basis = BSpline(self.knot_vector, self.matrix, 3, extrapolate=False)
        self.fixed = BSpline(self.knot_vector, self.prefix, 3, extrapolate=False)

    def query(self, progress, derivative=0):
        progress = _numpy(progress)
        if (progress.ndim != 1 or not np.isfinite(progress).all()
                or np.any(progress < 0.) or np.any(progress > 1.)
                or derivative not in (0, 1, 2, 3)):
            raise ValueError("Cubic queries need finite normalized times and derivative 0..3")
        times = progress*(self.knots+4)
        scale = self.knot_dt**derivative
        return self.basis(times, nu=derivative)/scale, self.fixed(times, nu=derivative)/scale

    def evaluate(self, actions, progress, derivative=0):
        actions = _numpy(actions)
        if actions.shape[-2:] != (self.knots, self.prefix.shape[-1]):
            raise ValueError("Cubic sampling requires the native free-control layout")
        weights, offset = self.query(progress, derivative)
        return np.einsum("tn,...nj->...tj", weights, actions)+offset

    def extrema_times(self, controls, joint, derivative=0):
        """Return boundaries and extrema of q or dq, in fixed-clock units.

        Query times use knot units. Values use the model's joint coordinates
        and seconds; velocity extrema are roots of the actual acceleration.
        """
        coefficients = self.prefix[:, joint]+self.matrix@controls
        polynomial = PPoly.from_spline(BSpline(self.knot_vector, coefficients, 3))
        if derivative:
            polynomial = polynomial.derivative(derivative)
        roots = polynomial.derivative().roots(extrapolate=False)
        roots = roots[np.isfinite(roots) & (roots >= 0.) & (roots <= self.knots+4)]
        times = np.unique(np.r_[np.arange(self.knots+5), roots])
        return times, polynomial(times)/(self.knot_dt**derivative)

    def velocity_extrema(self, controls):
        """Analytic all-time velocity bounds for one [control,joint] action."""
        controls = _numpy(controls)
        limits = []
        for joint in range(controls.shape[-1]):
            _, values = self.extrema_times(controls[:, joint], joint, 1)
            limits.append((float(values.min()), float(values.max())))
        return np.asarray(limits).T

    def solve_joint(self, hessian, linear, lower, upper, joint, *, origin,
                    initial, native_dtype, step_limit=None, velocity_bounds=None,
                    interior=False):
        """Exchange violated analytic extrema for linear QP rows, at most 24 times.

        The optimization variable is a coefficient increment from ``origin``.
        For fitting, the origin is zero. For recovery it is the current legal
        action. Checking the rounded candidate, rather than the double QP
        answer, also exposes violations introduced by the actual action dtype.
        """
        bounds = {0: (lower, upper)}
        if velocity_bounds is not None:
            bounds[1] = tuple(velocity_bounds)
        times = {derivative: self.extrema_times(initial, joint, derivative)[0]
                 for derivative in bounds}
        # This small inward margin belongs only to construction of a
        # representable fit anchor. Physical acceptance keeps the exact model
        # limits, and optimizer recovery never activates this margin.
        coordinate_scale = max(1., abs(lower), abs(upper), float(np.max(np.abs(initial))))
        roundoff = 32*np.finfo(native_dtype).eps*coordinate_scale if interior else 0.
        report = {"rounds": 0, "constraint_count": 0}
        for iteration in range(24):
            row_parts, lower_parts, upper_parts = [], [], []
            for derivative, (lo_bound, hi_bound) in bounds.items():
                scale = self.knot_dt**derivative
                weights = self.basis(times[derivative], nu=derivative)/scale
                offset = self.fixed(times[derivative], nu=derivative)[:, joint]/scale+weights@origin
                margin = roundoff*np.abs(weights).sum(-1)
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
                return None, {**report, "reason": "sampled_curve_qp_infeasible"}
            # Match base + direction in the actual float dtype. In particular,
            # float32 addition is not the cast of a float64 coefficient sum.
            rounded = (origin.astype(native_dtype)+increment.astype(native_dtype)).astype(float)
            violation, added = False, False
            for derivative, (lo_bound, hi_bound) in bounds.items():
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

    Position constraints are separable by joint, so the small diagonal QPs
    are solved serially. The original native step limits bound *increments*,
    not cubic coefficients. The optimizer must separately demand actual cost
    decrease before accepting any returned direction.
    """
    zero = torch.zeros_like(base)
    report = {"success": False, "reason": "invalid_recovery_input", "rounds": 0,
              "constraint_count": 0, "gradient_dot_step": 0., "limiter_scale": 0.,
              "per_joint_limiter_scale": []}
    if base.ndim != 3 or base.shape[0] != 1 or gradient.shape != base.shape:
        raise ValueError("Position recovery requires one matching action/gradient batch")
    if not bool(domain.feasible(base).all()):
        raise ValueError("Position recovery requires a legal current curve")
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
    lower, upper = _numpy(domain.lower), _numpy(domain.upper)
    hessian = np.eye(x.shape[0])
    for joint in range(x.shape[1]):
        if curve is None:
            rows = np.r_[hessian, hessian]
            lo = np.r_[(lower[joint]-x[:, joint])/maximum, -limits[:, joint]/maximum]
            hi = np.r_[(upper[joint]-x[:, joint])/maximum, limits[:, joint]/maximum]
            increment = _solve_qp(hessian, -target[:, joint], rows, lo, hi)
            detail = {"rounds": 1, "constraint_count": int(2*len(rows))}
        else:
            # Solve the furthest native trial as a displacement; convert back
            # to a direction only after the continuous-curve checks.
            increment, detail = curve.solve_joint(hessian, -maximum*target[:, joint],
                lower[joint], upper[joint], joint, origin=x[:, joint],
                initial=x[:, joint], native_dtype=native_dtype,
                step_limit=limits[:, joint])
            if increment is not None:
                increment = increment/maximum
        report["rounds"] = max(report["rounds"], detail["rounds"])
        report["constraint_count"] += detail["constraint_count"]
        if increment is None:
            return zero, {**report, "reason": "position_qp_infeasible"}
        increments[:, joint] = increment
    proposed = torch.as_tensor(increments, dtype=base.dtype, device=base.device)[None]
    # Position domains and these diagonal QPs are separable by joint. Correct
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


def _fit_feasible(domain, curve, action, velocity_bounds):
    """Seed-only feasibility: the optimizer's position domain stays unchanged."""
    if not bool(domain.feasible(action[None]).all()):
        return False
    if velocity_bounds is None:
        return True
    extrema = curve.velocity_extrema(action)
    return (bool(np.isfinite(extrema).all()) and bool(np.all(extrema[0] >= velocity_bounds[0]))
            and bool(np.all(extrema[1] <= velocity_bounds[1])))


def _fit_anchor(domain, curve, dtype, device, velocity_bounds):
    """Construct an anchor satisfying position and velocity on this clock.

    A position initializer can be reused only after the velocity check.
    Otherwise a small strictly convex QP constructs a numerical interior
    anchor for the same two domains; there is no position-only downgrade.
    """
    from .physical_joint_domain import JointDomainInitializationError
    try:
        anchor = (domain._anchor if domain._anchor is not None
                  else domain._construct_anchor(dtype, device))[0].to(dtype=dtype, device=device)
    except JointDomainInitializationError as error:
        return None, {"fit_anchor_source": "not_constructed", "detail": str(error)}
    if _fit_feasible(domain, curve, anchor, velocity_bounds):
        return anchor, {"fit_anchor_source": "validated_position_initializer"}
    initial = _numpy(anchor)
    lower, upper = _numpy(domain.lower), _numpy(domain.upper)
    controls = np.empty_like(initial)
    native_dtype = anchor.detach().cpu().numpy().dtype
    for joint in range(initial.shape[-1]):
        candidate, detail = curve.solve_joint(np.eye(domain.knots), -initial[:, joint],
            lower[joint], upper[joint], joint, origin=np.zeros(domain.knots),
            initial=initial[:, joint], native_dtype=native_dtype,
            velocity_bounds=velocity_bounds[:, joint], interior=True)
        if candidate is None:
            return None, {"fit_anchor_source": "not_constructed", "detail": detail["reason"]}
        controls[:, joint] = candidate
    anchor = torch.as_tensor(controls, dtype=dtype, device=device)
    if not _fit_feasible(domain, curve, anchor, velocity_bounds):
        return None, {"fit_anchor_source": "not_constructed", "detail": "rounded_anchor_outside_domain"}
    return anchor, {"fit_anchor_source": "position_velocity_qp"}


def fit_cubic_reference(domain, curve, progress, targets, prior_controls=None, velocity_bounds=None):
    """Fit soft references, with optional model velocity bounds only for seeds.

    Every production caller passes the model's [lower,upper] velocity bounds.
    Position-only numerical users may omit them; a requested position-and-
    velocity initialization is never silently weakened on construction failure.
    """
    values = _numpy(targets)
    weights, offset = curve.query(progress)
    count, dof = domain.knots, len(domain.lower)
    if velocity_bounds is not None:
        velocity_bounds = _numpy(velocity_bounds)
        if (velocity_bounds.shape != (2, dof) or not np.isfinite(velocity_bounds).all()
                or np.any(velocity_bounds[0] >= velocity_bounds[1])):
            raise ValueError("Cubic fit velocity bounds must be finite ordered [2,joint] values")
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
    # strictly convex. It is only a numerical tie-break, not another task cost.
    ridge = np.finfo(float).eps*max(1., float(np.trace(gram)))*64
    hessian = gram+ridge*np.eye(count)
    linear = weights.T@(offset-values)-ridge*prior
    controls = np.empty((count, dof))
    report = {"success": False, "reason": "fit_initializer_not_constructed", "rounds": 0,
              "constraint_count": 0, "joint_qp_status": [],
              "velocity_bounds_enforced": velocity_bounds is not None}
    lower, upper = _numpy(domain.lower), _numpy(domain.upper)
    for joint in range(dof):
        candidate, detail = curve.solve_joint(hessian, linear[:, joint],
            lower[joint], upper[joint], joint, origin=np.zeros(count),
            initial=prior[:, joint], native_dtype=native_dtype,
            velocity_bounds=None if velocity_bounds is None else velocity_bounds[:, joint])
        report["rounds"] = max(report["rounds"], detail["rounds"])
        report["constraint_count"] += detail["constraint_count"]
        report["joint_qp_status"].append(detail["reason"])
        if candidate is None:
            return None, {**report, "reason": "fit_initializer_not_constructed"}
        controls[:, joint] = candidate
    action = torch.as_tensor(controls, dtype=dtype, device=device)
    report.update(fit_contraction_scale=1., maximum_bound_excess_before_contraction=0.,
        maximum_velocity_bound_excess_before_contraction=0., fit_anchor_source="unneeded")
    if not _fit_feasible(domain, curve, action, velocity_bounds):
        # At an active extremum the rounded float32 optimum can lie a few
        # ulps outside the exact domain. Contract the entire curve toward an
        # anchor validated against every domain requested by this fit.
        # This is a feasible approximate fit, not a claim of QP optimality;
        # record both the pre-contraction violation and the actual contraction.
        low, high = domain.extrema(action[None])
        excess = torch.maximum(domain.lower-low, high-domain.upper).max()
        if velocity_bounds is None:
            from .physical_joint_domain import JointDomainInitializationError
            try:
                initialized = domain.initialize(action[None])[0]
            except JointDomainInitializationError as error:
                return None, {**report, "reason": "fit_initializer_not_constructed", "detail": str(error)}
            anchor = domain._anchor[0]
            report["fit_anchor_source"] = "position_initializer"
        else:
            velocity_extrema = curve.velocity_extrema(action)
            report["maximum_velocity_bound_excess_before_contraction"] = max(0.,
                float(np.max(velocity_bounds[0]-velocity_extrema[0])),
                float(np.max(velocity_extrema[1]-velocity_bounds[1])))
            anchor, anchor_report = _fit_anchor(domain, curve, dtype, device, velocity_bounds)
            report.update(anchor_report)
            if anchor is None:
                return None, {**report, "reason": "fit_initializer_not_constructed"}
            # Both domains are convex in the free coefficients. Recheck the
            # actual native-dtype sum at each trial; a rounded optimum or a
            # position-only anchor need not satisfy the velocity bounds.
            delta, low_scale, high_scale = action-anchor, 0., 1.
            for _ in range(24):
                middle = .5*(low_scale+high_scale)
                trial = anchor+middle*delta
                if _fit_feasible(domain, curve, trial, velocity_bounds):
                    low_scale = middle
                else:
                    high_scale = middle
            initialized = anchor+low_scale*delta
            if not _fit_feasible(domain, curve, initialized, velocity_bounds):
                return None, {**report, "reason": "rounded_fit_outside_position_velocity_domain"}
        proposed_norm = float((action-anchor).to(torch.float64).norm())
        accepted_norm = float((initialized-anchor).to(torch.float64).norm())
        report.update(maximum_bound_excess_before_contraction=max(0., float(excess)),
            fit_contraction_scale=accepted_norm/proposed_norm if proposed_norm else 0.)
        action = initialized
    residual = curve.evaluate(action, progress)-values
    velocity_extrema = curve.velocity_extrema(action)
    return action, {**report, "success": True, "reason": "physical_reference_fit",
        "reference_rms_joint_units": float(np.sqrt(np.mean(residual**2))),
        "reference_max_error_joint_units": float(np.max(np.abs(residual))),
        "query_max_abs_velocity_joint_units_per_s": np.max(np.abs(curve.evaluate(action, progress, 1)), axis=0).tolist(),
        "continuous_max_abs_velocity_joint_units_per_s": np.max(np.abs(velocity_extrema), axis=0).tolist()}
# END the complete curve validator and rollout own final feasibility/acceptance.
