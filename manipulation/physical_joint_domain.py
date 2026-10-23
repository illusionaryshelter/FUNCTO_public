"""Physical joint domains for native IK and continuous projected-gradient updates.

The domains restrict numerical iterates, not the demonstration or task goal.
No position is clipped after optimization. Cubic checks include the measured
instantaneous entrance and the analytic extrema of q, dq, ddq and jerk.
"""

# BEGIN physical-domain computation shared by initialization and ordinary descent.
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicHermiteSpline, PPoly

# BEGIN the eager optimizer checks each physical predicate synchronously.
from .optimizer_device_checks import require_tensor
# END validation keeps the same physical predicate and exception type.

# BEGIN CPU QPs share the exact cubic representation used by the tensor domain.
from .physical_joint_qp import (CubicAffineMap, construct_cubic_anchor,
                               fit_cubic_reference, projected_descent_step, program_work_report)
# END CPU programs and tensor predicates use the same physical curve.


class JointDomainInitializationError(ValueError):
    """A bounded initializer construction failed for the bound physical domain."""


def cubic_position_report(curve: CubicHermiteSpline, lower: np.ndarray, upper: np.ndarray) -> dict:
    """Check an exported polynomial independently of native free controls."""
    maximum, witness = 0., None
    for joint in range(len(lower)):
        polynomial = PPoly(curve.c[..., joint], curve.x, extrapolate=False)
        roots = polynomial.derivative().roots(extrapolate=False)
        times = np.unique(np.r_[curve.x, roots[np.isfinite(roots)
            & (roots >= curve.x[0]) & (roots <= curve.x[-1])]])
        values = polynomial(times)
        if not np.isfinite(values).all():
            raise ValueError("An exported physical-position polynomial is nonfinite")
        excess = np.maximum(lower[joint]-values, values-upper[joint])
        index = int(excess.argmax())
        if excess[index] > maximum:
            maximum = float(excess[index])
            witness = {"joint_column": joint, "time_s": float(times[index]),
                       "position": float(values[index]), "excess": maximum}
    return {"maximum_bound_excess": maximum, "maximum_violation": witness}


class JointPositionDomain:
    """Named finite scalar joint intervals for native single-state IK."""

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        self.lower = lower.detach().reshape(-1).to(torch.float64)
        self.upper = upper.detach().reshape(-1).to(torch.float64)
        if (not bool(torch.isfinite(self.lower).all() & torch.isfinite(self.upper).all())
                or not bool((self.lower < self.upper).all())):
            raise ValueError("Physical joint domains need finite ordered model bounds")

    def feasible(self, action: torch.Tensor) -> torch.Tensor:
        values = action.to(torch.float64)
        return (torch.isfinite(values) & (values >= self.lower) & (values <= self.upper)).all(dim=(-1, -2))

    # BEGIN host-side projected directions for single-state position boundaries.
    def descent_step(self, base, gradient, step_limits, scales, *, check_request=None, work_report=None):
        """Return a legal descent direction; the caller must verify cost decrease."""
        return projected_descent_step(self, base, gradient, step_limits, scales,
                                      check_request=check_request, work_report=work_report)
    # END independent coordinates are not frozen when another hits its bound.

    # BEGIN keep host validation separate from the unchanged tensor step map.
    def validate(self, action: torch.Tensor) -> None:
        """Raise synchronously before an illegal action reaches a rollout."""
        require_tensor(self.feasible(action).all(),
            "An optimizer action left the physical joint domain",
            error_type=JointDomainInitializationError)

    @staticmethod
    def validate_scales(scales: torch.Tensor) -> None:
        """Validate the fixed native line-search multipliers."""
        require_tensor(torch.isfinite(scales).all() & (scales >= 0).all(),
            "A domain-restricted line search needs finite nonnegative scales")

    def limit_step(self, base: torch.Tensor, step: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Shorten one direction before constructing any native trial action.

        Every nonnegative line-search scale retains its ratio on the same
        direction. Convexity of the bound physical domain justifies the
        scalar bisection. Rechecking the actual native-dtype trial arithmetic
        handles rounding at a physical boundary without widening that domain.
        """
        self.validate(base)
        self.validate_scales(scales)
        limited = self.limit_step_tensor(base, step, scales)
        self.validate(base[:, None]+scales*limited[:, None])
        return limited

    def limit_step_tensor(self, base: torch.Tensor, step: torch.Tensor,
                          scales: torch.Tensor) -> torch.Tensor:
        """Apply the exact step map to an already validated base/scales.

        Return an unchanged full step when every actual trial is feasible.
        Otherwise the fixed bisection preserves the direction ratios and
        checks rounding in the exact arithmetic used by the trial rollout.
        """
        finite = torch.isfinite(step).all(dim=(-1, -2), keepdim=True)
        step = torch.where(finite, step, torch.zeros_like(step))
        maximum = scales.max()
        low = torch.zeros_like(base[:, :1, :1])
        high = torch.ones_like(low)
        full = self.feasible(base+maximum*step)
        # BEGIN skip only iterations proved to leave the entire step unchanged.
        # With every endpoint feasible, the old bisection has low=high=1
        # throughout. Checking every native-dtype trial also proves that none
        # of the eight rounding guards would halve this step. A failed proof
        # continues through the unchanged limiter below, including mixed rows.
        trials = base[:, None]+scales*step[:, None]
        if bool(full.all() & self.feasible(trials).all()):
            return step
        # END the finite-row mask, physical bounds and trial arithmetic survive.
        low = torch.where(full[:, None, None], high, low)
        # A fixed bound keeps resource use independent of requested precision.
        for _ in range(24):
            middle = (low+high)*.5
            trial_step = step*middle
            passed = self.feasible(base+maximum*trial_step)[:, None, None]
            low, high = torch.where(passed, middle, low), torch.where(passed, high, middle)
        limited = step*low
        # All particles are checked in their exact eventual evaluation order;
        # lower alpha values can round differently from the furthest endpoint.
        for _ in range(8):
            trials = base[:, None]+scales*limited[:, None]
            passed = self.feasible(trials).all(dim=1)[:, None, None]
            limited = torch.where(passed, limited, limited*.5)
        trials = base[:, None]+scales*limited[:, None]
        passed = self.feasible(trials).all(dim=1)[:, None, None]
        limited = torch.where(passed, limited, torch.zeros_like(limited))
        return limited
    # END no initial/control coordinate or physical bound is projected/clipped.


class CubicJointDomain(JointPositionDomain):
    """Continuous q/dq/ddq/jerk domain of the measured-entrance cubic.

    Only measured q0/dq0 are fixed. Acceleration and jerk remain optimizable
    immediately; the original three terminal copies retain a stopped end.
    Free controls themselves have no invented box: only the represented
    physical curve and its three time derivatives determine this domain.
    """

    def __init__(self, bounds: torch.Tensor, start_q: torch.Tensor,
                 start_dq: torch.Tensor, knot_dt: float, knots: int):
        # BEGIN one model declaration binds every derivative to named joints.
        self.bounds = bounds.detach().to(torch.float64)
        if (self.bounds.ndim != 3 or self.bounds.shape[:2] != (4, 2)
                or not bool(torch.isfinite(self.bounds).all())
                or not bool((self.bounds[:, 0] < self.bounds[:, 1]).all())):
            raise ValueError("Cubic physical bounds require finite ordered [4,2,joint] values")
        super().__init__(self.bounds[0, 0], self.bounds[0, 1])
        # END position attributes also serve the shared single-state interface.
        self.knots = knots
        q = start_q.detach().reshape(-1).to(torch.float64)
        dq = start_dq.detach().reshape(-1).to(torch.float64)
        if q.shape != self.lower.shape or dq.shape != q.shape:
            raise ValueError("Cubic model bounds and measured entrance must share the named joint axis")
        self._anchor = None
        self.initialization_report = {**program_work_report(), "reason": "not_started"}
        # BEGIN eliminate only the two measured instantaneous equalities.
        self._start_q, self._start_dq = q, dq
        self._curve_map = CubicAffineMap(q, dq, knots, knot_dt)
        self.intervals = self._curve_map.intervals
        self._coefficient_matrix = torch.as_tensor(self._curve_map.matrix,
            device=q.device, dtype=torch.float64)
        self._velocity_coefficients = torch.as_tensor(self._curve_map.velocity_coefficients,
            device=q.device, dtype=torch.float64)*knot_dt
        # At an exact physical boundary an outward measured velocity leaves
        # the interval immediately for every finite-acceleration cubic. This
        # necessary tangent condition is algebraic, not a widened q tolerance.
        self._entrance_valid = (torch.isfinite(q).all() & torch.isfinite(dq).all()
            & (q >= self.lower).all() & (q <= self.upper).all()
            & (dq >= self.bounds[1, 0]).all() & (dq <= self.bounds[1, 1]).all()
            & ((q > self.lower) | (dq >= 0)).all()
            & ((q < self.upper) | (dq <= 0)).all())
        # END domain, CPU QP and native tensor states share these coefficients.

    # BEGIN reference values and cubic coefficients have distinct public APIs.
    def evaluate(self, actions, progress, derivative=0):
        """Sample normalized progress, returning NumPy values in physical units.

        Actions end in [free controls, joints]; output ends in [queries,
        joints]. Derivatives 1..3 use seconds from the immutable knot clock.
        """
        return self._curve_map.evaluate(actions, progress, derivative)

    # BEGIN reference fitting uses the same measured entrance and cubic clock.
    def fit_reference(self, progress, joint_targets, prior_controls=None, *, check_request=None):
        """Fit soft joint references in this complete continuous physical domain.

        Returns [controls,joints] or None with a construction report.
        Progress uses the normalized physical clock; targets use this domain's
        joint column order. The report includes actual joint-reference errors.
        """
        return fit_cubic_reference(self, self._curve_map, progress, joint_targets, prior_controls,
                                   check_request=check_request)
    # END task-space constraints are evaluated by the planner on the fitted curve.

    def descent_step(self, base, gradient, step_limits, scales, *, check_request=None, work_report=None):
        """Find a curve-feasible direction without boxing its free coefficients."""
        return projected_descent_step(self, base, gradient, step_limits, scales, self._curve_map,
                                      check_request=check_request, work_report=work_report)
    # END references are soft observations, never new trajectory task equalities.

    def coefficients(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-2:] != (self.knots, len(self.lower)):
            raise ValueError("Physical cubic domain requires the bound native free-control layout")
        # BEGIN relative controls avoid subtracting large absolute joint angles.
        return self._relative_coefficients(action)+self._start_q
        # END no coefficient clipping or fixed first interval is introduced.

    def _relative_coefficients(self, action):
        return (torch.matmul(self._coefficient_matrix, action.to(torch.float64)-self._start_q)
                +self._velocity_coefficients[:, None]*self._start_dq)

    def all_extrema(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower/upper values with trailing [derivative, named joint] axes.

        Position checks its velocity roots, velocity checks acceleration
        roots, acceleration checks both segment endpoints, and jerk checks
        each constant segment. Both sides of every jerk jump are included.
        All derivatives use physical seconds on the unchanged knot clock.
        """
        # BEGIN evaluate polynomial differences before restoring absolute q0.
        if action.shape[-2:] != (self.knots, len(self.lower)):
            raise ValueError("Physical cubic domain requires the bound free-control layout")
        c = self._relative_coefficients(action)
        p0, p1, p2, p3 = (c[..., offset:offset+self.intervals, :] for offset in range(4))
        first_diff, second_diff, third_diff = p1-p0, p2-p1, p3-p2
        a = (third_diff-2*second_diff+first_diff)/6
        b = (second_diff-first_diff)/2
        d = (first_diff+second_diff)/2
        e = p1+(second_diff-first_diff)/6
        # These are exact identities of the eliminated entrance equations.
        # They must match the tensor state's exact q0/dq0 rows even when q0
        # lies on a physical bound and relative coefficients are nonzero.
        e = torch.cat((torch.zeros_like(e[..., :1, :]), e[..., 1:, :]), dim=-2)
        initial_d = (self._start_dq*self._curve_map.knot_dt).expand_as(d[..., :1, :])
        d = torch.cat((initial_d, d[..., 1:, :]), dim=-2)
        quadratic, linear = 3*a, 2*b
        discriminant = linear.square()-4*quadratic*d
        root = torch.sqrt(torch.clamp_min(discriminant, 0))
        safe_quad = torch.where(quadratic != 0, quadratic, torch.ones_like(quadratic))
        first = (-linear+root)/(2*safe_quad)
        second = (-linear-root)/(2*safe_quad)
        safe_linear = torch.where(linear != 0, linear, torch.ones_like(linear))
        linear_root = -d/safe_linear
        first = torch.where(quadratic == 0, linear_root, first)
        first_valid = ((quadratic != 0) & (discriminant >= 0)) | ((quadratic == 0) & (linear != 0))
        second_valid = (quadratic != 0) & (discriminant >= 0)
        # Invalid/outside roots are replaced by a true endpoint, never clipped
        # into the interval and mistaken for a stationary point.
        first = torch.where(first_valid & (first > 0) & (first < 1), first, torch.zeros_like(first))
        second = torch.where(second_valid & (second > 0) & (second < 1), second, torch.zeros_like(second))
        values = torch.stack((e, a+b+d+e, ((a*first+b)*first+d)*first+e,
                              ((a*second+b)*second+d)*second+e), dim=-2)+self._start_q
        safe_a = torch.where(a != 0, a, torch.ones_like(a))
        velocity_root = -b/(3*safe_a)
        velocity_root = torch.where((a != 0) & (velocity_root > 0) & (velocity_root < 1),
                                    velocity_root, torch.zeros_like(velocity_root))
        h = self._curve_map.knot_dt
        velocities = torch.stack((d, 3*a+2*b+d,
            (3*a*velocity_root+2*b)*velocity_root+d), dim=-2)/h
        accelerations = torch.stack((2*b, 6*a+2*b), dim=-2)/(h*h)
        jerks = (6*a/(h*h*h)).unsqueeze(-2)
        # END no roundoff-sized outward entrance is fabricated or tolerated.
        ranges = (values, velocities, accelerations, jerks)
        return (torch.stack([value.amin(dim=(-2, -3)) for value in ranges], dim=-2),
                torch.stack([value.amax(dim=(-2, -3)) for value in ranges], dim=-2))

    def extrema(self, action: torch.Tensor, derivative=0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return continuous extrema of the requested physical derivative."""
        low, high = self.all_extrema(action)
        return low[..., derivative, :], high[..., derivative, :]

    def feasible(self, action: torch.Tensor) -> torch.Tensor:
        low, high = self.all_extrema(action)
        return (self._entrance_valid & torch.isfinite(action).all(dim=(-1, -2))
                & torch.isfinite(low).all(dim=(-1, -2)) & torch.isfinite(high).all(dim=(-1, -2))
                & (low >= self.bounds[:, 0]).all(dim=(-1, -2))
                & (high <= self.bounds[:, 1]).all(dim=(-1, -2)))

    def _construct_anchor(self, dtype: torch.dtype, device: torch.device, *, check_request=None) -> torch.Tensor:
        """Construct an original-domain LP anchor independently of reference fit."""
        # BEGIN one construction records its actual LP calls, including an interruption.
        self.initialization_report = {**program_work_report(), "reason": "constructing_anchor"}
        constant = self._start_q.to(dtype=dtype, device=device).expand(1, self.knots, -1).clone()
        if not bool(self._entrance_valid):
            self.initialization_report["reason"] = "algebraic_entrance_invalid"
            raise JointDomainInitializationError(
                "algebraic_entrance_invalid: measured q0/dq0 violate model bounds or the position tangent")
        if bool(self.feasible(constant).all()):
            self.initialization_report["reason"] = "constant_curve_feasible"
            return constant
        anchor, report = construct_cubic_anchor(self, self._curve_map, dtype, device,
            check_request=check_request, report=self.initialization_report)
        self.initialization_report = report
        if anchor is None:
            raise JointDomainInitializationError(
                f"No physical-domain cubic initializer was constructed: {report}; "
                "measured q0/dq0 and the frozen knot clock were retained")
        return anchor
        # END reusing this anchor never repeats or recharges its native solve.

    def initialize(self, proposals: torch.Tensor, *, check_request=None) -> torch.Tensor:
        """Keep legal reference seeds and contract only illegal initializers."""
        shape = proposals.shape
        values = proposals.reshape(-1, self.knots, len(self.lower))
        self.initialization_report = {**program_work_report(), "reason": "checking_proposals",
                                      "proposal_count": len(values)}
        if bool(self.feasible(values).all()):
            self.initialization_report["reason"] = "proposals_already_feasible"
            return proposals
        anchor_source = "existing_anchor"
        if self._anchor is None:
            valid = self.feasible(values)
            if bool(valid.any()):
                self._anchor = values[valid][:1].clone()
                anchor_source = "feasible_proposal"
            else:
                self._anchor = self._construct_anchor(proposals.dtype, proposals.device,
                                                       check_request=check_request)
                anchor_source = "physical_domain_anchor"
        base = self._anchor.expand_as(values)
        scales = values.new_ones((1, 1, 1, 1))
        step = self.limit_step(base, values-base, scales)
        initialized = base+step
        self.validate(initialized)
        self.initialization_report.update(reason="physical_proposals_initialized",
            proposal_count=len(values), anchor_source=anchor_source)
        return initialized.reshape(shape)
# END every accepted/exploration/best action represents a domain-valid curve.
