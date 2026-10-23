"""Physical-domain validation shares exact solver rows and named model limits."""

# BEGIN validation checks complete curves without altering their coefficients.
import unittest

import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline
import torch

from manipulation.physical_joint_domain import CubicJointDomain, JointPositionDomain, cubic_position_report


class PhysicalJointDomainTests(unittest.TestCase):
    def domain(self, q=0., dq=0., limits=(1., 100., 1000., 10000.)):
        bounds = torch.tensor([[-value, value] for value in limits], dtype=torch.float64)[:, :, None]
        return CubicJointDomain(bounds, torch.tensor([q], dtype=torch.float64),
            torch.tensor([dq], dtype=torch.float64), .2, 8)

    def test_validation_and_batched_feasibility_use_the_exact_signed_constraints(self):
        domain = self.domain()
        controls = torch.tensor(np.random.default_rng(2).normal(size=(2, 3, 8, 1)), dtype=torch.float64)
        expected = np.array([np.all(domain.signed_constraints(curve)[0] <= 0.)
                            for curve in controls.numpy().reshape(-1, 8, 1)]).reshape(2, 3)
        np.testing.assert_array_equal(domain.feasible(controls).numpy(), expected)
        self.assertIsNone(domain.validate(torch.zeros((8, 1))))
        with self.assertRaisesRegex(ValueError, "continuous physical joint bounds"):
            domain.validate(torch.full((8, 1), 3.))
        invalid = torch.zeros((2, 8, 1))
        invalid[1, 0, 0] = float("nan")
        np.testing.assert_array_equal(domain.feasible(invalid).numpy(), [True, False])

    def test_each_derivative_can_exclude_a_position_legal_curve(self):
        controls = torch.zeros((8, 1), dtype=torch.float64)
        controls[3, 0] = .2
        wide = self.domain()
        self.assertTrue(bool(wide.feasible(controls)))
        _, _, witness = wide.signed_constraints(controls, with_witnesses=True)
        for derivative in (1, 2, 3):
            limits = [1., 100., 1000., 10000.]
            limits[derivative] = np.abs(witness["value"][witness["derivative_order"] == derivative]).max()*.8
            constrained = self.domain(limits=limits)
            original = controls.clone()
            self.assertFalse(bool(constrained.feasible(controls)))
            torch.testing.assert_close(controls, original, rtol=0, atol=0)

    def test_invalid_measured_entry_is_preserved_and_fails_the_same_physical_rows(self):
        for q, dq in ((1.2, 0.), (0., .6), (1., .2), (-1., -.2)):
            domain = self.domain(q=q, dq=dq, limits=(1., .5, 20., 200.))
            controls = torch.full((8, 1), q, dtype=torch.float64)
            self.assertFalse(bool(domain.feasible(controls)))
            np.testing.assert_array_equal(domain.evaluate(controls, [0.]), [[q]])
            np.testing.assert_array_equal(domain.evaluate(controls, [0.], 1), [[dq]])

    def test_control_position_box_does_not_hide_an_interior_violation(self):
        domain = self.domain(q=.995, dq=.2)
        controls = torch.full((8, 1), .995, dtype=torch.float64)
        self.assertTrue(bool((controls.abs() <= 1).all()))
        self.assertFalse(bool(domain.feasible(controls)))
        residual, _, witness = domain.signed_constraints(controls, with_witnesses=True)
        violating = (witness["derivative_order"] == 0) & (residual > 0)
        self.assertTrue(violating.any())
        self.assertTrue(np.all(witness["time_s"][violating] > 0.))

    def test_exact_position_boundary_retains_exact_measured_entry(self):
        for q, inward in ((1., -.2), (-1., .2)):
            for dq in (0., inward):
                domain = self.domain(q=q, dq=dq)
                controls = torch.full((8, 1), q, dtype=torch.float64)
                self.assertTrue(bool(domain.feasible(controls)))
                np.testing.assert_array_equal(domain.evaluate(controls, [0.]), [[q]])
                np.testing.assert_array_equal(domain.evaluate(controls, [0.], 1), [[dq]])

    def test_terminal_derivatives_use_free_coefficients(self):
        domain = self.domain(dq=.1)
        controls = torch.linspace(-.2, .3, 8, dtype=torch.float64).reshape(8, 1)
        coefficients = domain._curve_map.coefficients(controls)[:, 0]
        polynomial = BSpline(domain._curve_map.knot_vector, coefficients, 3)
        for derivative in (1, 2, 3):
            np.testing.assert_allclose(domain.evaluate(controls, [.9, .95, 1.], derivative)[:, 0],
                polynomial(np.array([.9, .95, 1.])*7, derivative)/(.2**derivative), atol=1e-12)
        self.assertGreater(abs(domain.evaluate(controls, [1.], 1)[0, 0]), .1)

    def test_export_report_locates_between_sample_position_violation(self):
        curve = CubicHermiteSpline([0., 1.], [[0.], [0.]], [[4.], [-4.]])
        report = cubic_position_report(curve, np.array([-.8]), np.array([.8]))
        self.assertAlmostEqual(report["maximum_bound_excess"], .2)
        self.assertEqual(report["maximum_violation"]["joint_column"], 0)
        self.assertEqual(report["maximum_violation"]["time_s"], .5)
        self.assertEqual(report["maximum_violation"]["position"], 1.)

    def test_named_position_limits_retain_every_model_coordinate(self):
        lower, upper = torch.tensor([-2., .1]), torch.tensor([3., .8])
        domain = JointPositionDomain(lower, upper)
        torch.testing.assert_close(domain.lower, lower.double(), rtol=0, atol=0)
        torch.testing.assert_close(domain.upper, upper.double(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
# END the physical predicate reports feasibility without constructing a new curve.
