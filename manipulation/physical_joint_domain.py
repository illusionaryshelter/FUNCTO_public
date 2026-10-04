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
from .optimizer_device_checks import require_tensor
# END public calls outside an optimizer graph still raise synchronously.


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
        """Apply the original step map to an already validated base/scales.

        Every operation, bracket, dtype and reduction below is retained from
        the strict wrapper's original implementation. This is deliberately
        ordinary Torch, with no new tracing, compilation, fusion or reduction
        rewrite. The optimizer separately records/checks the resulting trial
        predicate before calling either native rollout.
        """
        finite = torch.isfinite(step).all(dim=(-1, -2), keepdim=True)
        step = torch.where(finite, step, torch.zeros_like(step))
        maximum = scales.max()
        low = torch.zeros_like(base[:, :1, :1])
        high = torch.ones_like(low)
        full = self.feasible(base+maximum*step)
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
        knot_vector = np.arange(-3., self.knots+8.)
        matrix = np.zeros((self.knots+7, self.knots))
        matrix[4:4+self.knots] = np.eye(self.knots)
        matrix[-3:, -1] = 1.
        basis = BSpline(knot_vector, matrix, 3)
        prefix = np.zeros((self.knots+7, len(self.lower)))
        prefix[:4] = self.prefix.cpu().numpy()
        fixed = BSpline(knot_vector, prefix, 3)
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
