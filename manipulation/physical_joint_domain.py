"""Physical joint domains for native IK and cubic L-BFGS updates.

The domains restrict numerical iterates, not the demonstration or task goal.
No position is clipped after optimization. Cubic checks include the fixed
measured entrance controls and every analytic interior position extremum.
"""

# BEGIN physical-domain step restriction shared by initialization and L-BFGS.
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import BSpline, CubicHermiteSpline, PPoly
from scipy.optimize import linprog

# BEGIN a graph scope records errors without changing any domain arithmetic.
from .optimizer_device_checks import has_active_device_checks, require_tensor
# END public calls outside an optimizer graph still raise synchronously.

# BEGIN CPU QPs share the exact cubic representation used by the tensor domain.
from .physical_joint_qp import CubicAffineMap, fit_cubic_reference, recover_position_step
# END no CPU solver is called by a captured tensor limiter.


class JointDomainInitializationError(ValueError):
    """A bounded initializer construction failed for the unchanged actual."""


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

    # BEGIN graph-external recovery for single-state IK position boundaries.
    def recovery_step(self, base, gradient, step_limits, scales):
        """Return a legal descent direction; the caller must verify cost decrease."""
        return recover_position_step(self, base, gradient, step_limits, scales)
    # END independent coordinates are not frozen when another hits its bound.

    # BEGIN keep host validation separate from the unchanged tensor step map.
    def validate(self, action: torch.Tensor) -> torch.Tensor | None:
        """Raise outside a graph; return a recorded predicate inside its scope.

        Graph callers must use the predicate to isolate invalid actions before
        a rollout. Recording an error alone would still let a captured kernel
        consume that action before the enclosing block reports the failure.
        """
        return require_tensor(self.feasible(action).all(),
            "An optimizer action left the physical joint domain",
            error_type=JointDomainInitializationError)

    @staticmethod
    def validate_scales(scales: torch.Tensor) -> None:
        """Validate the fixed native line-search multipliers outside capture."""
        require_tensor(torch.isfinite(scales).all() & (scales >= 0).all(),
            "A domain-restricted line search needs finite nonnegative scales")

    def limit_step(self, base: torch.Tensor, step: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Shorten one direction before constructing any native trial action.

        Every nonnegative line-search scale retains its ratio on the same
        direction. Convexity of the physical position domain justifies the
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

        Eager calls can return an unchanged full step after checking every
        actual trial. Graph/warmup scopes retain device-only control flow.
        The optimizer separately validates the resulting trial predicates
        before calling either native rollout.
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
        # Exclude both the owned warmup scope and external CUDA capture: a host
        # predicate must neither specialize a graph nor synchronize its stream.
        eager = (not has_active_device_checks()
                 and (step.device.type != "cuda"
                      or not torch.cuda.is_current_stream_capturing()))
        if eager:
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
    """Exact position domain of cuRobo's cubic, with free terminal controls.

    The prefix follows the installed degree-three boundary kernel with
    measured q0/dq0 and zero unmeasured acceleration. Three terminal copies
    match ``use_implicit_goal=False``. Free controls themselves have no
    invented box: only the represented physical q(t) determines this domain.
    """

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor, start_q: torch.Tensor,
                 start_dq: torch.Tensor, knot_dt: float, knots: int):
        super().__init__(lower, upper)
        self.knots = knots
        q = start_q.detach().reshape(-1).to(torch.float64)
        dq = start_dq.detach().reshape(-1).to(torch.float64)
        self.prefix = q[None]+q.new_tensor([-1., 0., 1., 2.])[:, None]*dq[None]*knot_dt
        self._anchor = None
        self._start_q = q
        # BEGIN one native affine representation for seed sampling and CPU QPs.
        self._curve_map = CubicAffineMap(self.prefix, knots, knot_dt)
        # END its fixed coefficients and clock never change during a solve.

    # BEGIN reference values and cubic coefficients have distinct public APIs.
    def evaluate(self, actions, progress, derivative=0):
        """Sample normalized progress, returning NumPy values in physical units.

        Actions end in [free controls, joints]; output ends in [queries,
        joints]. Derivatives 1..3 use seconds from the immutable knot clock.
        """
        return self._curve_map.evaluate(actions, progress, derivative)

    def fit_reference(self, progress, joint_targets, prior_controls=None, velocity_bounds=None):
        """Fit soft samples on this clock, optionally enforcing model velocity.

        Returns [controls,joints] or None with a construction report. Velocity
        bounds apply only to this initializer, not to the optimizer domain or
        its recovery directions for trajectories with existing violations.
        """
        return fit_cubic_reference(self, self._curve_map, progress, joint_targets,
                                   prior_controls, velocity_bounds)

    def recovery_step(self, base, gradient, step_limits, scales):
        """Find a curve-feasible direction without boxing its free coefficients."""
        return recover_position_step(self, base, gradient, step_limits, scales, self._curve_map)
    # END references are soft observations, never new trajectory task equalities.

    def coefficients(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-2:] != (self.knots, len(self.lower)):
            raise ValueError("Physical cubic domain requires the bound native free-control layout")
        shape = action.shape[:-2]
        prefix = self.prefix.expand(*shape, *self.prefix.shape)
        return torch.cat((prefix, action.to(torch.float64),
                          action[..., -1:, :].to(torch.float64).expand(*shape, 3, len(self.lower))), dim=-2)

    def extrema(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate endpoints and every real derivative root of each cubic."""
        c = self.coefficients(action)
        p0, p1, p2, p3 = (c[..., offset:offset+self.knots+4, :] for offset in range(4))
        a = (-p0+3*p1-3*p2+p3)/6
        b = (p0-2*p1+p2)/2
        d = (p2-p0)/2
        e = (p0+4*p1+p2)/6
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
                              ((a*second+b)*second+d)*second+e), dim=-2)
        return values.amin(dim=(-2, -3)), values.amax(dim=(-2, -3))

    def feasible(self, action: torch.Tensor) -> torch.Tensor:
        low, high = self.extrema(action)
        return (torch.isfinite(action).all(dim=(-1, -2)) & torch.isfinite(low).all(-1)
                & torch.isfinite(high).all(-1) & (low >= self.lower).all(-1)
                & (high <= self.upper).all(-1))

    def _construct_anchor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Use the existing SciPy LP and analytic extrema in a finite cutting loop.

        Failure means this bounded initializer was not constructed. It does
        not rewrite dq0 or prove that every representation is infeasible.
        Only this initializer uses a small inward numerical margin, to obtain
        representable free controls. The optimizer's domain remains the full
        physical interval above.
        """
        constant = self._start_q.to(dtype=dtype, device=device).expand(1, self.knots, -1).clone()
        if bool(self.feasible(constant).all()):
            return constant
        # BEGIN share exactly one coefficient map with fitting and recovery.
        knot_vector, matrix = self._curve_map.knot_vector, self._curve_map.matrix
        basis, prefix, fixed = self._curve_map.basis, self._curve_map.prefix, self._curve_map.fixed
        # END the old initializer keeps its existing LP and acceptance rules.
        lower, upper = self.lower.cpu().numpy(), self.upper.cpu().numpy()
        controls = np.empty((self.knots, len(lower)))
        for joint in range(len(lower)):
            samples = np.linspace(0., self.knots+4., 4*(self.knots+4)+1)
            solved = False
            for _ in range(24):
                weights, offset = basis(samples), fixed(samples)[:, joint]
                influence = np.minimum(1., np.abs(weights).sum(-1))
                margin = 32*np.finfo(np.float32).eps*(1+abs(lower[joint])+abs(upper[joint]))*influence
                result = linprog(np.zeros(self.knots),
                    A_ub=np.r_[weights, -weights],
                    b_ub=np.r_[upper[joint]-offset-margin, offset-lower[joint]-margin],
                    bounds=[(None, None)]*self.knots, method="highs")
                if not result.success:
                    break
                candidate = result.x.astype(np.float32).astype(float)
                coefficients = prefix[:, joint]+matrix@candidate
                polynomial = PPoly.from_spline(BSpline(knot_vector, coefficients, 3))
                roots = polynomial.derivative().roots(extrapolate=False)
                roots = roots[np.isfinite(roots) & (roots >= 0) & (roots <= self.knots+4)]
                query = np.unique(np.r_[samples, roots])
                positions = polynomial(query)
                bad = (positions < lower[joint]) | (positions > upper[joint])
                if not np.any(bad):
                    controls[:, joint], solved = candidate, True
                    break
                samples = np.unique(np.r_[samples, query[bad]])
            if not solved:
                raise JointDomainInitializationError(
                    f"No physical-domain cubic initializer was constructed for joint column {joint}; "
                    "measured q0/dq0 and the frozen knot clock were retained")
        anchor = torch.as_tensor(controls, dtype=dtype, device=device)[None]
        self.validate(anchor)
        return anchor

    def initialize(self, proposals: torch.Tensor) -> torch.Tensor:
        """Keep legal reference seeds and contract only illegal initializers."""
        shape = proposals.shape
        values = proposals.reshape(-1, self.knots, len(self.lower))
        if bool(self.feasible(values).all()):
            return proposals
        if self._anchor is None:
            valid = self.feasible(values)
            self._anchor = (values[valid][:1].clone() if bool(valid.any()) else
                            self._construct_anchor(proposals.dtype, proposals.device))
        base = self._anchor.expand_as(values)
        scales = values.new_ones((1, 1, 1, 1))
        step = self.limit_step(base, values-base, scales)
        initialized = base+step
        self.validate(initialized)
        return initialized.reshape(shape)
# END every accepted/exploration/best action represents a domain-valid curve.
