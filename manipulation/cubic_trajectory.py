"""One instantaneous-boundary cubic basis for float64 physical trajectories.

Only measured q(0) and dq(0) constrain the entrance. SciPy constructs the
linear basis once; PhysicalCurve reconstructs the actual delivered Hermite64
polynomial from it. No optimizer, simulator or task semantics enter here.
"""

# BEGIN eliminate exactly two entrance equalities in a shared cubic basis.
import numpy as np
from scipy.interpolate import BSpline


class CubicBasis:
    """Uniform cubic coefficients with free entrance acceleration and jerk.

    M position-valued actions represent c2..c(M+1). With h the knot interval,
    c0=x0-2*h*dq0 and c1=1.5*q0+.5*h*dq0-.5*x0 eliminate q0/dq0 exactly.
    Three final copies retain the original stopped final interval. Therefore
    there are M+2 intervals and M+5 complete coefficients. Internally all
    actions are relative to q0, avoiding cancellation of absolute angles in
    derivative evaluation; the public action remains a position coefficient.
    """

    def __init__(self, controls: int, knot_dt: float):
        if (type(controls) is not int or controls < 2
                or not np.isfinite(knot_dt) or knot_dt <= 0):
            raise ValueError("Cubic basis needs at least two controls and a finite positive clock")
        self.knots, self.knot_dt = controls, float(knot_dt)
        self.intervals = controls + 2
        self.knot_vector = np.arange(-3., controls + 6.)
        self.matrix = np.zeros((controls + 5, controls))
        self.matrix[0, 0], self.matrix[1, 0] = 1., -.5
        self.matrix[2:2 + controls] = np.eye(controls)
        self.matrix[-3:, -1] = 1.
        self.velocity_coefficients = np.zeros(controls + 5)
        self.velocity_coefficients[:2] = [-2., .5]
        self.basis = BSpline(self.knot_vector, self.matrix, 3, extrapolate=False)
        self.velocity_basis = BSpline(self.knot_vector, self.velocity_coefficients, 3,
                                      extrapolate=False)

    @property
    def reference_parameters(self):
        """Greville positions of free coefficients, solely for soft seeds."""
        return (np.arange(self.knots) + 1.) / self.intervals

    def weights(self, progress, derivative=0):
        """Return action and entrance-velocity weights in physical seconds.

        q=q0+W@(x-q0)+V*dq0. For derivative>0 the independent q0 term is
        absent. Exact endpoint rows simplify mathematical identities, rather
        than admitting a neighbouring measured state or widening a tolerance.
        """
        progress = np.asarray(progress, dtype=float)
        if (progress.ndim != 1 or not np.isfinite(progress).all()
                or np.any(progress < 0.) or np.any(progress > 1.)
                or derivative not in (0, 1, 2, 3)):
            raise ValueError("Cubic queries need finite normalized times and derivative 0..3")
        return self.weights_at_knot_times(progress * self.intervals, derivative)

    def weights_at_knot_times(self, times, derivative=0):
        """Evaluate exact knot coordinates without a normalize/un-normalize roundtrip."""
        times = np.asarray(times, dtype=float)
        if (times.ndim != 1 or not np.isfinite(times).all()
                or np.any(times < 0.) or np.any(times > self.intervals)
                or derivative not in (0, 1, 2, 3)):
            raise ValueError("Cubic knot queries must lie inside the original curve")
        scale = self.knot_dt**derivative
        weights = self.basis(times, nu=derivative) / scale
        velocity = self.velocity_basis(times, nu=derivative) * self.knot_dt / scale
        entrance = times == 0.
        if derivative < 2:
            weights[entrance] = 0.
            velocity[entrance] = float(derivative == 1)
        # The final four identical supporting coefficients make this whole
        # interval exactly constant, including its left endpoint by C2.
        terminal = times >= self.intervals - 1
        weights[terminal] = 0.
        velocity[terminal] = 0.
        if derivative == 0:
            weights[terminal, -1] = 1.
        return weights, velocity


# END the shared basis fixes only the measured entrance and terminal interval.
