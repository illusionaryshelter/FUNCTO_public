"""Model joint limits and exact continuous-domain validation of cubic curves.

The complete signed inequalities are consumed by the constrained optimizer
and by final physical validation. Seed fitting uses the same affine curve.
"""

# BEGIN physical validation consumes the optimizer's original signed rows.
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import CubicHermiteSpline, PPoly

from .cubic_constraints import CubicAffineMap


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
    """Ordered model intervals for the named scalar joints used by IK."""

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        self.lower = lower.detach().reshape(-1).to(torch.float64)
        self.upper = upper.detach().reshape(-1).to(torch.float64)
        if (self.lower.shape != self.upper.shape
                or not bool(torch.isfinite(self.lower).all() & torch.isfinite(self.upper).all())
                or not bool((self.lower < self.upper).all())):
            raise ValueError("Physical joint domains need finite ordered model bounds")


class CubicJointDomain(JointPositionDomain):
    """Continuous q/dq/ddq/jerk bounds with measured entrance and fixed clock.

    ``bounds[derivative,side,joint]`` contains four derivative orders and two
    sides in the same named joint order as q0/dq0. Free controls are position
    coefficients; only the represented curve receives physical model bounds.
    """

    def __init__(self, bounds: torch.Tensor, start_q: torch.Tensor,
                 start_dq: torch.Tensor, knot_dt: float, knots: int):
        self.bounds = bounds.detach().to(torch.float64)
        if (self.bounds.ndim != 3 or self.bounds.shape[:2] != (4, 2)
                or not bool(torch.isfinite(self.bounds).all())
                or not bool((self.bounds[:, 0] < self.bounds[:, 1]).all())):
            raise ValueError("Cubic physical bounds require finite ordered [4,2,joint] values")
        super().__init__(self.bounds[0, 0], self.bounds[0, 1])
        self._constraint_bounds = self.bounds.cpu().numpy().copy()
        self._start_q = start_q.detach().reshape(-1).to(torch.float64)
        self._start_dq = start_dq.detach().reshape(-1).to(torch.float64)
        if self._start_q.shape != self.lower.shape or self._start_dq.shape != self.lower.shape:
            raise ValueError("Cubic model bounds and measured entrance must share the named joint axis")
        self._curve_map = CubicAffineMap(self._start_q, self._start_dq, knots, knot_dt)
        self.knots, self.intervals = self._curve_map.knots, self._curve_map.intervals

    def evaluate(self, actions, progress, derivative=0):
        """Return NumPy samples on normalized progress, with derivatives in seconds."""
        return self._curve_map.evaluate(actions, progress, derivative)

    def signed_constraints(self, actions, *, with_witnesses=False):
        """Return g<=0 and its coefficient Jacobian for all physical bounds.

        Optional witnesses identify derivative order, original segment,
        named joint column, bound side, physical time, value and bound.
        CubicAffineMap defines the fixed row order and active-branch rule.
        """
        return self._curve_map.signed_constraints(
            actions, self._constraint_bounds, with_witnesses=with_witnesses)

    def fit_samples(self, progress, joint_targets, prior_controls=None):
        """Return a least-squares coefficient seed and its joint-sample error."""
        return self._curve_map.fit_samples(progress, joint_targets, prior_controls)

    def feasible(self, actions: torch.Tensor) -> torch.Tensor:
        """Check each complete curve using precisely the optimizer's g<=0 rows.

        The two trailing axes are [free controls,named joints]. Leading axes
        identify independent curves and are preserved in the Boolean result.
        Nonfinite inputs have no physical curve and evaluate as infeasible.
        """
        if actions.shape[-2:] != (self.knots, len(self.lower)):
            raise ValueError("Cubic physical validation requires [free controls,joints] trailing axes")
        curves = actions.detach().cpu().numpy().reshape(-1, self.knots, len(self.lower))
        result = [bool(np.isfinite(curve).all() and np.all(self.signed_constraints(curve)[0] <= 0.))
                  for curve in curves]
        return torch.tensor(result, device=actions.device, dtype=torch.bool).reshape(actions.shape[:-2])

    def validate(self, actions: torch.Tensor) -> None:
        """Raise ValueError when an input curve violates a physical model bound."""
        if not bool(self.feasible(actions).all()):
            raise ValueError("A cubic curve violates the continuous physical joint bounds")
# END validation, optimization and fitting share one measured-entrance curve.
