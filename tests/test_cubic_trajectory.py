"""Independent CPU checks of the remaining shared cubic coefficient basis."""

# BEGIN the physical Hermite integration is covered in test_physical_curve.
import unittest

import numpy as np
from scipy.interpolate import BSpline

from manipulation.cubic_trajectory import CubicBasis


def independent_curve(q, dq, h, action):
    """Direct complete-coefficient construction, independent of production rows."""
    coefficients = np.r_[[(action[0]-2*h*dq)],
                         [(1.5*q+.5*h*dq-.5*action[0])],
                         action, np.repeat(action[-1:], 3, axis=0)]
    return BSpline(np.arange(-3., len(action)+6.), coefficients, 3)


class CubicTrajectoryTests(unittest.TestCase):
    def test_regular_dense_and_independent_bspline_preserve_measured_entrances(self):
        basis = CubicBasis(10, .5)
        states = [(np.array([1.4, -.7]), np.array([1e-5, -2e-5])),
                  (np.array([.2, 1.1]), np.array([-.13, .2]))]
        action = np.linspace(-.2, .6, 20).reshape(10, 2)
        for q0, dq0 in states:
            curve = independent_curve(q0, dq0, .5, action)
            for derivative in range(4):
                regular_w, regular_v = basis.weights_at_knot_times(np.arange(49)/4, derivative)
                dense_w, dense_v = basis.weights_at_knot_times(np.arange(97)/8, derivative)
                regular = regular_w@(action-q0)+regular_v[:, None]*dq0
                dense = dense_w@(action-q0)+dense_v[:, None]*dq0
                if derivative == 0:
                    regular, dense = regular+q0, dense+q0
                expected = curve(np.arange(49)/4, derivative)/(.5**derivative)
                np.testing.assert_allclose(regular, expected, rtol=0., atol=2e-14)
                np.testing.assert_array_equal(regular, dense[::2])
                if derivative < 2:
                    np.testing.assert_array_equal(regular[0], q0 if derivative == 0 else dq0)

    def test_first_two_controls_leave_entrance_acceleration_and_jerk_free(self):
        basis = CubicBasis(8, .5)
        acceleration, _ = basis.weights(np.array([0.]), derivative=2)
        jerk, _ = basis.weights(np.array([0.]), derivative=3)
        np.testing.assert_array_equal(np.r_[acceleration[:, :2], jerk[:, :2]],
                                      np.array([[12., 0.], [-44., 8.]]))
        first_sample, _ = basis.weights(np.array([1/(4*basis.intervals)]))
        self.assertTrue(bool((first_sample[0, :2] != 0.).all()))
        for derivative in (0, 1):
            weights, velocity = basis.weights(np.array([0.]), derivative)
            np.testing.assert_array_equal(weights, np.zeros_like(weights))
            np.testing.assert_array_equal(velocity, np.array([float(derivative == 1)]))

    def test_original_curve_family_and_existing_terminal_interval_survive(self):
        basis = CubicBasis(10, .68)
        q, dq = .73, -2.1e-5
        original = np.linspace(-.2, .4, 8)
        embedded = np.r_[q+.68*dq, q+2*.68*dq, original]
        old = BSpline(basis.knot_vector,
            np.r_[q+np.arange(-1, 3)*.68*dq, original, np.repeat(original[-1], 3)], 3)
        progress = np.linspace(0, 1, 301)
        for derivative in range(4):
            weights, velocity = basis.weights(progress, derivative)
            actual = weights@(embedded-q)+velocity*dq+(q if derivative == 0 else 0.)
            np.testing.assert_allclose(actual, old(progress*12, derivative)/(.68**derivative),
                                       atol=2e-14, rtol=0)
        weights, velocity = basis.weights(np.linspace(11/12, 1, 11))
        np.testing.assert_array_equal(weights[:, -1], np.ones(11))
        np.testing.assert_array_equal(velocity, np.zeros(11))


if __name__ == "__main__":
    unittest.main()
# END no obsolete cuRobo rollout adapter remains under a compatibility name.
