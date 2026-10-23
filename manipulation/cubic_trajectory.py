"""Fixed-clock cubic basis with exactly two measured entrance equalities.

SciPy constructs affine rows for position and its three time derivatives.
The same rows define optimization, physical constraints and Hermite export.
"""

# BEGIN eliminate exactly two entrance equalities in a shared cubic basis.
import numpy as np
from scipy.interpolate import BSpline, PPoly


class CubicBasis:
    """Uniform cubic coefficients with free entrance acceleration and jerk.

    M position-valued actions represent c2..c(M+1). With h the knot interval,
    c0=x0-2*h*dq0 and c1=1.5*q0+.5*h*dq0-.5*x0 eliminate q0/dq0 exactly.
    The M+2 complete coefficients span M-1 intervals. The final position,
    velocity and acceleration remain free unless task rows constrain them.
    Internally all actions are relative to q0, avoiding cancellation of absolute angles in
    derivative evaluation; the public action remains a position coefficient.
    """

    def __init__(self, controls: int, knot_dt: float):
        if (type(controls) is not int or controls < 2
                or not np.isfinite(knot_dt) or knot_dt <= 0):
            raise ValueError("Cubic basis needs at least two controls and a finite positive clock")
        self.knots, self.knot_dt = controls, float(knot_dt)
        # BEGIN retain only the two measured entrance equalities in the basis.
        self.intervals = controls - 1
        self.knot_vector = np.arange(-3., controls + 3.)
        self.matrix = np.zeros((controls + 2, controls))
        self.matrix[0, 0], self.matrix[1, 0] = 1., -.5
        self.matrix[2:2 + controls] = np.eye(controls)
        self.velocity_coefficients = np.zeros(controls + 2)
        self.velocity_coefficients[:2] = [-2., .5]
        self.basis = BSpline(self.knot_vector, self.matrix, 3, extrapolate=False)
        self.velocity_basis = BSpline(self.knot_vector, self.velocity_coefficients, 3,
                                      extrapolate=False)
        # BEGIN polynomial rows expose the same curve to continuous constraints.
        # Each column is converted by SciPy once. Coefficients descend in
        # degree and use local knot time u in [0,1], before physical h scaling.
        # The two entrance identities match weights_at_knot_times exactly.
        columns = [PPoly.from_spline(BSpline(self.knot_vector, column, 3)).c[:, 3:3+self.intervals]
                   for column in self.matrix.T]
        self.polynomial_weights = np.stack(columns, axis=-1)
        self.polynomial_velocity = PPoly.from_spline(self.velocity_basis).c[:, 3:3+self.intervals] * self.knot_dt
        self.polynomial_weights[-2:, 0] = 0.
        self.polynomial_velocity[-2:, 0] = [self.knot_dt, 0.]
        # END these arrays have no task, joint-limit or optimizer semantics.
        # END a stop or final pose enters through explicit task constraints.

    @property
    def reference_parameters(self):
        """Observe every noninitial knot and the final interval midpoint.

        The measured q0/dq0 are already eliminated from the coefficients.
        The M-2 interior knot positions leave two free coefficient directions.
        The final interval midpoint and endpoint provide independent
        observations of those directions, retaining free terminal velocity
        and acceleration. All M queries are at positive times. Their
        spacing follows the spline support rather than a shifted uniform
        observation grid whose terminal amplification grows with M.
        """
        # BEGIN match position observations to the free-terminal spline support.
        times = np.r_[np.arange(1., self.intervals), self.intervals-.5, self.intervals]
        return times/self.intervals
        # END the query count, physical clock and endpoint constraints are unchanged.

    def weights(self, progress, derivative=0):
        """Return action and entrance-velocity weights in physical seconds.

        q=q0+W@(x-q0)+V*dq0. For derivative>0 the independent q0 term is
        absent. Exact entrance rows encode the two measured equalities.
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
        return weights, velocity

    # BEGIN preserve the original polynomial through the public Hermite interface.
    def hermite_nodes(self, actions, start_q, start_dq):
        """Return binary64 time, position and velocity at every original break.

        ``actions`` is one [free controls, named joints] array; the two entry
        vectors use the same joint order. A cubic segment is uniquely fixed
        by its endpoint positions and velocities. Using its original breaks
        therefore preserves that segment without rebuilding many short
        polynomials from independently rounded native state samples.
        """
        actions = np.asarray(actions, dtype=np.float64)
        start_q = np.asarray(start_q, dtype=np.float64)
        start_dq = np.asarray(start_dq, dtype=np.float64)
        if (actions.ndim != 2 or actions.shape[0] != self.knots
                or start_q.shape != (actions.shape[1],) or start_dq.shape != start_q.shape
                or not all(np.isfinite(value).all() for value in (actions, start_q, start_dq))):
            raise ValueError("Cubic export requires one finite named control array and its measured entrance")
        knots = np.arange(self.intervals + 1, dtype=np.float64)
        delta = actions - start_q
        position_weights, position_entry = self.weights_at_knot_times(knots)
        velocity_weights, velocity_entry = self.weights_at_knot_times(knots, 1)
        position = start_q + position_weights @ delta + position_entry[:, None] * start_dq
        velocity = velocity_weights @ delta + velocity_entry[:, None] * start_dq
        return knots * self.knot_dt, position, velocity
    # END original cubic segments survive export independently of sampling density.

# END one basis supplies physical derivatives and continuous-curve export.
