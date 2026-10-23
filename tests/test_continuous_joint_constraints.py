"""Continuous signed rows, witness locations and active-branch derivatives."""

# BEGIN the solver's physical input is tested independently of its search method.
import unittest

import numpy as np
from scipy.interpolate import CubicHermiteSpline
import torch

from manipulation.physical_joint_domain import CubicJointDomain
from manipulation.cubic_constraints import CubicAffineMap


class ContinuousJointConstraintTests(unittest.TestCase):
    def domain(self, joints=2, *, clock=.3, q=None, dq=None):
        limits = torch.tensor([2., 3., 4., 5.], dtype=torch.float64)
        bounds = torch.stack((-limits, limits), dim=1)[:, :, None].repeat(1, 1, joints)
        return CubicJointDomain(bounds, torch.zeros(joints, dtype=torch.float64) if q is None else q,
            torch.zeros(joints, dtype=torch.float64) if dq is None else dq, clock, 8)

    def test_every_segment_matches_independent_hermite_polynomial_extrema(self):
        domain = self.domain(q=torch.tensor([.2, -.1]), dq=torch.tensor([.1, .2]))
        controls = np.random.default_rng(4).normal(size=(8, 2))
        residual, jacobian, witness = domain.signed_constraints(controls, with_witnesses=True)
        times, position, velocity = domain._curve_map.hermite_nodes(
            controls, domain._start_q.numpy(), domain._start_dq.numpy())
        curve = CubicHermiteSpline(times, position, velocity)
        self.assertEqual(residual.shape, (10*domain.intervals*2,))
        self.assertEqual(jacobian.shape, (len(residual), controls.size))
        for derivative in range(4):
            for segment in range(domain.intervals):
                for joint in range(2):
                    # Keep the original segment to retain the left side of
                    # every jerk jump; querying a whole spline at its right
                    # endpoint would select the next segment's jerk.
                    local = np.poly1d(curve.c[:, segment, joint]).deriv(derivative)
                    candidates = [0., times[segment+1]-times[segment]]
                    if derivative < 2:
                        candidates += [float(root.real) for root in local.deriv().r
                                       if root.imag == 0. and 0. < root.real < candidates[1]]
                    values = local(candidates)
                    rows = ((witness["derivative_order"] == derivative)
                            & (witness["segment_index"] == segment)
                            & (witness["joint_column"] == joint))
                    expected = max(float(domain.bounds[derivative, 0, joint])-values.min(),
                                   values.max()-float(domain.bounds[derivative, 1, joint]))
                    self.assertAlmostEqual(residual[rows].max(), expected, places=10)
        np.testing.assert_allclose(residual,
            (2*witness["bound_side"]-1)*(witness["value"]-witness["bound"]), rtol=0, atol=0)

    def test_selected_jacobian_matches_smooth_branch_finite_differences(self):
        domain = self.domain(q=torch.tensor([.2, .3]), dq=torch.tensor([.1, -.1]))
        controls = np.random.default_rng(0).normal(size=(8, 2))
        _, jacobian = domain.signed_constraints(controls)
        epsilon = 1e-6
        differences = []
        for direction in np.eye(controls.size).reshape(-1, *controls.shape):
            high = domain.signed_constraints(controls+epsilon*direction)[0]
            low = domain.signed_constraints(controls-epsilon*direction)[0]
            differences.append((high-low)/(2*epsilon))
        np.testing.assert_allclose(jacobian, np.stack(differences, axis=-1), rtol=2e-7, atol=3e-8)
        # Each constraint belongs to one named joint, including a moving root.
        _, _, witness = domain.signed_constraints(controls, with_witnesses=True)
        for row, joint in enumerate(witness["joint_column"]):
            self.assertTrue(np.all(jacobian[row].reshape(8, 2)[:, 1-joint] == 0.))

    def test_tied_extrema_use_subgradient_and_full_directional_maximum(self):
        domain = self.domain(joints=1)
        controls = np.zeros((8, 1))
        direction = np.random.default_rng(2).normal(size=controls.shape)
        residual, jacobian, witness = domain.signed_constraints(controls, with_witnesses=True)
        epsilon = 1e-6
        directional = (domain.signed_constraints(controls+epsilon*direction)[0]-residual)/epsilon
        # At the constant curve every time is active. The one-sided derivative
        # is the maximum of all active affine directional values, not the
        # derivative of the one earliest branch returned to the optimizer.
        expected = domain.signed_constraints(direction)[0]-residual
        np.testing.assert_allclose(directional, expected, rtol=2e-9, atol=2e-8)
        selected = jacobian@direction.ravel()
        nonlinear = witness["derivative_order"] < 2
        self.assertTrue(np.all(selected[nonlinear] <= directional[nonlinear]+1e-8))
        self.assertGreater(np.max(directional[nonlinear]-selected[nonlinear]), .1)
        np.testing.assert_allclose(selected[~nonlinear], directional[~nonlinear], atol=2e-8)
        first_position = (witness["derivative_order"] == 0) & (witness["segment_index"] == 1)
        np.testing.assert_array_equal(witness["time_s"][first_position], [.3, .3])

    def test_interior_extremum_remains_a_row_with_its_actual_time(self):
        domain = self.domain(joints=1)
        controls = np.random.default_rng(3).normal(size=(8, 1))
        _, _, witness = domain.signed_constraints(controls, with_witnesses=True)
        local = witness["time_s"]/.3-witness["segment_index"]
        interior = (witness["derivative_order"] < 2) & (local > 1e-12) & (local < 1.-1e-12)
        self.assertTrue(interior.any())
        for row in np.flatnonzero(interior):
            derivative = witness["derivative_order"][row]
            value = domain.evaluate(controls, [witness["time_s"][row]/(.3*domain.intervals)], derivative+1)
            self.assertAlmostEqual(float(value[0, 0]), 0., places=10)

    def test_immutable_invalid_entry_is_an_impossible_constant_constraint(self):
        domain = self.domain(joints=1, q=torch.tensor([2.5], dtype=torch.float64))
        controls = np.full((8, 1), 2.5)
        residual, jacobian, witness = domain.signed_constraints(controls, with_witnesses=True)
        entry_upper = ((witness["derivative_order"] == 0) & (witness["segment_index"] == 0)
                       & (witness["bound_side"] == 1))
        self.assertEqual(float(residual[entry_upper][0]), .5)
        np.testing.assert_array_equal(jacobian[entry_upper], np.zeros((1, controls.size)))

    def test_physical_seconds_scale_each_derivative_and_preserve_fixed_entry(self):
        slow, fast = self.domain(), self.domain(clock=.15)
        controls = np.random.default_rng(1).normal(size=(8, 2))
        _, slow_jacobian, slow_witness = slow.signed_constraints(controls, with_witnesses=True)
        _, fast_jacobian, fast_witness = fast.signed_constraints(controls, with_witnesses=True)
        scales = 2.**slow_witness["derivative_order"]
        np.testing.assert_allclose(fast_witness["value"], scales*slow_witness["value"], atol=1e-10)
        np.testing.assert_allclose(fast_jacobian, scales[:, None]*slow_jacobian, atol=1e-10)
        np.testing.assert_allclose(fast_witness["time_s"], .5*slow_witness["time_s"], atol=1e-15)

    def test_sample_seed_can_be_infeasible_and_recovers_exact_representable_samples(self):
        domain = self.domain(joints=1, q=torch.tensor([.2]), dq=torch.tensor([.1]))
        controls = np.linspace(-8., 9., 8)[:, None]
        progress = np.linspace(0., 1., 100)
        targets = domain.evaluate(controls, progress)
        fitted, report = domain.fit_samples(progress, targets)
        np.testing.assert_allclose(fitted, controls, rtol=0, atol=2e-14)
        self.assertEqual(report["reason"], "unconstrained_sample_fit")
        self.assertLess(report["reference_max_error_joint_units"], 2e-14)
        self.assertGreater(domain.signed_constraints(fitted)[0].max(), 1.)
        np.testing.assert_array_equal(domain.evaluate(fitted, [0.]), [domain._start_q.numpy()])
        np.testing.assert_array_equal(domain.evaluate(fitted, [0.], 1), [domain._start_dq.numpy()])

    def test_underdetermined_sample_fit_retains_prior_nullspace(self):
        domain = self.domain(joints=1)
        prior = np.arange(8, dtype=float)[:, None]
        fitted, _ = domain.fit_samples([1.], [[3.]], prior)
        np.testing.assert_array_equal(fitted[:-3], prior[:-3])
        # Endpoint B-spline weights are [1,4,1]/6 on the final three controls.
        # Their least-norm correction removes the three-unit sample error.
        np.testing.assert_allclose((fitted-prior)[-3:, 0], [-1., -4., -1.], atol=1e-14)
        np.testing.assert_allclose(domain.evaluate(fitted, [1.]), [[3.]], atol=1e-14)

    # BEGIN exercise the most amplified observation direction of the seed fit.
    def test_reference_query_perturbation_has_bounded_coefficient_response(self):
        rng = np.random.default_rng(21)
        for count in (2, 6, 13, 21, 29):
            curve = CubicAffineMap([.2, -.4], [.03, -.02], count, .3)
            controls = curve.start_q+rng.normal(scale=.1, size=(count, 2))
            progress = curve.reference_parameters
            weights, _ = curve.weights(progress)
            left, singular, _ = np.linalg.svd(weights)
            perturbation = 1e-3*left[:, -1, None]*np.array([[1., -1.]])
            targets = curve.evaluate(controls, progress)+perturbation
            fitted, report = curve.fit_samples(progress, targets)
            np.testing.assert_allclose(curve.evaluate(fitted, progress), targets, atol=1e-14)
            amplification = np.linalg.norm(fitted-controls)/np.linalg.norm(perturbation)
            self.assertAlmostEqual(amplification, 1./singular[-1], places=10)
            self.assertLess(amplification, 22.)
            np.testing.assert_allclose(report["sample_matrix_singular_values"], singular, atol=1e-15)
            self.assertAlmostEqual(report["sample_matrix_condition_number"], singular[0]/singular[-1])
            np.testing.assert_array_equal(curve.evaluate(fitted, [0.]), [curve.start_q])
            np.testing.assert_array_equal(curve.evaluate(fitted, [0.], 1), [curve.start_dq])
    # END the two measured entrance equalities hold through the perturbed fit.


if __name__ == "__main__":
    unittest.main()
# END no test runs a robot, simulation, CUDA kernel or optimization search.
