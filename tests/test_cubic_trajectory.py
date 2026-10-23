"""Independent polynomial checks for the fixed-clock, free-terminal cubic."""

# BEGIN the mathematical basis is verified without an optimizer adapter.
import unittest

import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline

from manipulation.cubic_trajectory import CubicBasis


def independent_curve(q, dq, clock, controls):
    """Construct complete coefficients directly from the two entrance equalities."""
    coefficients = np.concatenate(((controls[0]-2*clock*dq)[None],
        (1.5*q+.5*clock*dq-.5*controls[0])[None], controls))
    return BSpline(np.arange(-3., len(controls)+3.), coefficients, 3, extrapolate=False)


def evaluate(basis, controls, q, dq, progress, derivative):
    weights, velocity = basis.weights(progress, derivative)
    values = weights@(controls-q)+velocity[:, None]*dq
    return values+q if derivative == 0 else values


class CubicBasisTests(unittest.TestCase):
    def test_all_physical_derivatives_match_independent_complete_coefficients(self):
        rng = np.random.default_rng(19)
        for count in (2, 8, 18):
            for clock in (.125, .68):
                basis = CubicBasis(count, clock)
                q, dq = np.array([1.4, -.7]), np.array([-.13, .2])
                controls = rng.normal(size=(count, 2))
                progress = np.r_[0., rng.uniform(size=23), 1.]
                oracle = independent_curve(q, dq, clock, controls)
                for derivative in range(4):
                    actual = evaluate(basis, controls, q, dq, progress, derivative)
                    np.testing.assert_allclose(actual,
                        oracle(progress*(count-1), derivative)/clock**derivative,
                        rtol=3e-13, atol=2e-12)
                np.testing.assert_array_equal(evaluate(basis, controls, q, dq, [0.], 0), q[None])
                np.testing.assert_array_equal(evaluate(basis, controls, q, dq, [0.], 1), dq[None])

    def test_polynomial_rows_match_the_same_curve_on_every_segment(self):
        basis = CubicBasis(8, .3)
        q, dq = np.array([.73, -.2]), np.array([.13, -.4])
        controls = np.random.default_rng(3).normal(size=(8, 2))
        coefficients = np.einsum("ksn,nj->ksj", basis.polynomial_weights, controls-q)
        coefficients += basis.polynomial_velocity[:, :, None]*dq
        for segment in range(basis.intervals):
            local = np.linspace(0., 1., 9)
            progress = (segment+local)/basis.intervals
            for derivative in range(3):
                expected = evaluate(basis, controls, q, dq, progress, derivative)
                for joint in range(2):
                    actual = np.poly1d(coefficients[:, segment, joint]).deriv(derivative)(local)
                    actual = actual/basis.knot_dt**derivative+(q[joint] if derivative == 0 else 0.)
                    np.testing.assert_allclose(actual, expected[:, joint], rtol=2e-13, atol=1e-12)

    def test_original_hermite_breaks_reconstruct_every_cubic_segment(self):
        basis = CubicBasis(18, .68)
        q, dq = np.array([.73, -1.8]), np.array([1e-7, -.2])
        controls = q+np.random.default_rng(6).normal(size=(18, 2))*.02
        times, position, velocity = basis.hermite_nodes(controls, q, dq)
        reconstructed = CubicHermiteSpline(times, position, velocity)
        # Interior physical times distinguish polynomial preservation from
        # matching only the exported position and velocity samples.
        physical = (np.arange(basis.intervals)[:, None]+np.array([.17, .5, .83])).ravel()*.68
        for derivative in range(4):
            expected = evaluate(basis, controls, q, dq, physical/times[-1], derivative)
            np.testing.assert_allclose(reconstructed(physical, derivative), expected, rtol=3e-12, atol=8e-13)
        np.testing.assert_array_equal(position[0], q)
        np.testing.assert_array_equal(velocity[0], dq)
        self.assertEqual(len(times), basis.intervals+1)

    def test_entrance_acceleration_and_jerk_are_independent_free_directions(self):
        basis = CubicBasis(8, .5)
        rows = np.stack([basis.weights([0.], derivative)[0][0, :2] for derivative in (2, 3)])
        np.testing.assert_array_equal(rows, [[12., 0.], [-44., 8.]])
        self.assertEqual(np.linalg.matrix_rank(rows), 2)
        self.assertTrue(np.all(basis.weights([.01])[0][0, :2] != 0.))

    def test_free_terminal_state_can_continue_measured_velocity_without_stopping(self):
        basis = CubicBasis(10, .68)
        q, dq = np.array([.73]), np.array([.2])
        controls = q+(np.arange(10)+1)[:, None]*.68*dq
        progress = np.linspace(0, 1, 301)
        for derivative in range(4):
            actual = evaluate(basis, controls, q, dq, progress, derivative)
            expected = (q+progress[:, None]*9*.68*dq if derivative == 0 else
                        np.full_like(actual, dq[0] if derivative == 1 else 0.))
            np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
        terminal_rows = np.stack([basis.weights([1.], order)[0][0] for order in range(3)])
        self.assertEqual(np.linalg.matrix_rank(terminal_rows), 3)

    # BEGIN the free terminal interval has independent position observations.
    def test_reference_queries_remain_well_conditioned_as_control_count_grows(self):
        for count in range(2, 30):
            basis = CubicBasis(count, .3)
            progress = basis.reference_parameters
            self.assertEqual(len(progress), count)
            self.assertGreater(progress[0], 0.)
            self.assertEqual(progress[-1], 1.)
            self.assertTrue(np.all(np.diff(progress) > 0.))
            weights, _ = basis.weights(progress)
            self.assertEqual(np.linalg.matrix_rank(weights), count)
            # This is a regression bound on the sampling design, not a
            # runtime condition-number gate or a constraint on the seed.
            self.assertLess(np.linalg.cond(weights), 21.)
    # END no initial observation duplicates the exactly eliminated measured q0.

    def test_physical_clock_rescales_derivatives_of_a_zero_velocity_entry(self):
        slow, fast = CubicBasis(8, .4), CubicBasis(8, .2)
        progress = np.array([0., .12, .51, 1.])
        for derivative in range(4):
            np.testing.assert_allclose(fast.weights(progress, derivative)[0],
                                       slow.weights(progress, derivative)[0]*2**derivative,
                                       rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
# END endpoint state, physical derivatives and export retain one polynomial.
