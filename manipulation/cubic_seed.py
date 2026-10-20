"""Small reference fits and final checks for the shared cubic trajectory.

A seed is a numerical initial guess, not a proof of physical or task success.
NumPy fits the existing affine basis; the shared physical curve constrains
subsequent optimization. The standalone report is a raw polynomial diagnostic.
"""

# BEGIN reference-only cubic algebra, without optimizer-domain machinery.
import numpy as np
import torch
from scipy.interpolate import PPoly

from .cubic_trajectory import CubicBasis


def cubic_position_report(curve, lower, upper):
    """Check joint bounds at endpoints and analytic polynomial extrema."""
    maximum, witness = 0., None
    for joint in range(len(lower)):
        polynomial = PPoly(curve.c[..., joint], curve.x, extrapolate=False)
        roots = polynomial.derivative().roots(extrapolate=False)
        times = np.unique(np.r_[curve.x, roots[np.isfinite(roots)
            & (roots >= curve.x[0]) & (roots <= curve.x[-1])]])
        values = polynomial(times)
        if not np.isfinite(values).all():
            raise ValueError("An exported joint polynomial is nonfinite")
        excess = np.maximum(lower[joint]-values, values-upper[joint])
        index = int(excess.argmax())
        if excess[index] > maximum:
            maximum = float(excess[index])
            witness = {"joint_column": joint, "time_s": float(times[index]),
                       "position": float(values[index]), "excess": maximum}
    return {"maximum_bound_excess": maximum, "maximum_violation": witness}


class CubicSeed:
    """Fit nominal samples while preserving measured entrance q and dq."""

    def __init__(self, start_q, start_dq, knot_dt, controls):
        self.start_q = start_q.detach().reshape(-1)
        self.start_dq = start_dq.detach().reshape(-1)
        self.basis = CubicBasis(controls, knot_dt)

    def fit(self, progress, positions):
        """Return one finite least-squares proposal, without feasibility gates."""
        samples = np.asarray(progress, dtype=float)
        targets = (positions.detach().cpu().numpy() if isinstance(positions, torch.Tensor)
                   else np.asarray(positions, dtype=float))
        q0 = self.start_q.cpu().numpy().astype(float)
        dq0 = self.start_dq.cpu().numpy().astype(float)
        if targets.shape != (len(samples), len(q0)) or not np.isfinite(targets).all():
            raise ValueError("Cubic reference samples must have finite named joint coordinates")
        weights, velocity = self.basis.weights(samples)
        relative, _, rank, _ = np.linalg.lstsq(weights, targets-q0-velocity[:, None]*dq0, rcond=None)
        actions = relative+q0
        if not np.isfinite(actions).all():
            raise ValueError("Native cubic seed fit produced nonfinite controls")
        fitted = q0+weights@relative+velocity[:, None]*dq0
        result = torch.as_tensor(actions, device=self.start_q.device, dtype=self.start_q.dtype)
        return result, {"method": "least_squares", "rank": int(rank),
                        "rms_joint_error": float(np.sqrt(np.mean((fitted-targets)**2)))}

# END reference fits do not replace the shared physical constraint rows.
