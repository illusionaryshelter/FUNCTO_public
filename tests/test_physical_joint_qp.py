"""CPU oracles for fixed-clock fitting and continuous-boundary recovery."""

# BEGIN independent curve and objective checks for the shared CPU QP interface.
import unittest

import numpy as np
from scipy.interpolate import BSpline, PPoly
import torch

from manipulation.physical_joint_domain import CubicJointDomain, JointPositionDomain


class PhysicalJointQPTest(unittest.TestCase):
    @staticmethod
    def domain(q=0., dq=0.):
        return CubicJointDomain(torch.tensor([-1.]), torch.tensor([1.]),
            torch.tensor([q]), torch.tensor([dq]), .2, 8)

    @staticmethod
    def scales(maximum=1.):
        return torch.tensor([0., .2, maximum]).reshape(1, 3, 1, 1)

    def velocity_oracle(self, domain, controls):
        coefficients = domain.coefficients(controls[None])[0, :, 0].numpy()
        velocity = PPoly.from_spline(BSpline(np.arange(-3., 16.), coefficients, 3)).derivative()
        roots = velocity.derivative().roots(extrapolate=False)
        times = np.r_[np.arange(13), roots[np.isfinite(roots) & (roots >= 0.) & (roots <= 12.)]]
        values = velocity(times)/.2
        return values.min(), values.max()

    def test_affine_sampling_matches_independent_curve_and_physical_derivatives(self):
        domain = self.domain(.2, .1)
        controls = torch.linspace(-.3, .5, 8).reshape(1, 8, 1)
        coefficients = domain.coefficients(controls)[0, :, 0].numpy()
        spline = BSpline(np.arange(-3., 16.), coefficients, 3)
        progress = np.linspace(0., 1., 53)
        for derivative in range(4):
            actual = domain.evaluate(controls, progress, derivative)[0, :, 0]
            expected = spline(progress*12, nu=derivative)/(.2**derivative)
            np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(domain.evaluate(controls, [0., 1.])[0, :, 0], [.2, .5], atol=1e-8)
        np.testing.assert_allclose(domain.evaluate(controls, [0., 1.], 1)[0, :, 0], [.1, 0.], atol=1e-8)

    def test_fitting_preserves_a_legal_curve_with_a_coefficient_outside_joint_box(self):
        domain = self.domain()
        original = torch.zeros((8, 1))
        original[2] = 1.2
        self.assertTrue(bool(domain.feasible(original[None]).all()))
        progress = np.linspace(0., 1., 81)
        targets = domain.evaluate(original, progress)
        fitted, report = domain.fit_reference(progress, targets, prior_controls=original)
        self.assertTrue(report["success"], report)
        self.assertGreater(float(fitted.max()), 1.)
        self.assertTrue(bool(domain.feasible(fitted[None]).all()))
        np.testing.assert_allclose(domain.evaluate(fitted, progress), targets, atol=2e-7, rtol=0.)

    def test_reference_is_soft_and_fixed_entrance_and_clock_are_retained(self):
        domain = self.domain(.2, .1)
        original_prefix = domain.prefix.clone()
        progress = np.linspace(0., 1., 41)
        targets = (2*progress)[:, None]
        fitted, report = domain.fit_reference(progress, targets)
        self.assertTrue(report["success"], report)
        self.assertTrue(bool(domain.feasible(fitted[None]).all()))
        self.assertGreaterEqual(report["reference_max_error_joint_units"], 1.)
        self.assertGreater(report["fit_contraction_scale"], .99)
        self.assertLessEqual(report["fit_contraction_scale"], 1.)
        torch.testing.assert_close(domain.prefix, original_prefix)
        np.testing.assert_allclose(domain.evaluate(fitted, [0.]), [[.2]], atol=1e-8)
        np.testing.assert_allclose(domain.evaluate(fitted, [0.], 1), [[.1]], atol=1e-8)

    def test_moving_extrema_on_boundary_platform_are_exchanged_before_acceptance(self):
        domain = self.domain()
        base = torch.ones((1, 8, 1))
        # The first six coordinates point inward, the last two outward. A
        # scalar restriction of this raw direction almost stops all motion.
        gradient = torch.tensor([1., 1., 1., 1., 1., 1., -1., -1.]).reshape_as(base)
        limits = torch.full((8, 1), .1)
        scales = self.scales()
        raw = domain.limit_step(base, -.1*gradient, scales)
        direction, report = domain.recovery_step(base, gradient, limits, scales)
        self.assertTrue(report["success"], report)
        self.assertGreater(report["rounds"], 1)
        self.assertLess(report["gradient_dot_step"], 0.)
        self.assertGreater(float(direction.norm()), 100*float(raw.norm()))
        self.assertLess(float(direction[0, 0, 0]), 0.)
        self.assertTrue(bool(domain.feasible(base[:, None]+scales*direction[:, None]).all()))
        self.assertGreater(float((base+direction-base).abs().sum()), 1e-4)

    def test_ik_boundary_leaves_other_joint_free_and_honors_largest_native_scale(self):
        domain = JointPositionDomain(torch.tensor([-1., -1.]), torch.tensor([1., 1.]))
        base = torch.tensor([[[1., 0.]]])
        gradient = torch.tensor([[[-1., 1.]]])
        scales = self.scales(2.)
        direction, report = domain.recovery_step(base, gradient, torch.ones((1, 2)), scales)
        self.assertTrue(report["success"], report)
        self.assertEqual(float(direction[0, 0, 0]), 0.)
        self.assertLess(float(direction[0, 0, 1]), -.1)
        self.assertTrue(bool(domain.feasible(base[:, None]+scales*direction[:, None]).all()))
        self.assertLessEqual(float((2*direction).abs().max()), 1.)

    def test_cubic_rounding_on_one_joint_does_not_shrink_an_independent_joint(self):
        domain = CubicJointDomain(torch.full((2,), -1.), torch.ones(2),
            torch.zeros(2), torch.zeros(2), .2, 16)
        base = torch.zeros((1, 16, 2))
        base[..., 0] = 1.
        # The first joint is on a long platform; its free entrance controls
        # can still move, but its nearly active interior extrema round tightly.
        # The second joint is in the interior and has an ordinary descent step.
        gradient = torch.full_like(base, -1.)
        gradient[..., 0] = -6.
        scales = torch.tensor([0., .1, .5, 1.]).reshape(1, 4, 1, 1)
        direction, report = domain.recovery_step(base, gradient, torch.full((16, 2), 1.96), scales)
        self.assertTrue(report["success"], report)
        self.assertGreater(float(direction[..., 1].abs().min()), .1)
        self.assertEqual(report["per_joint_limiter_scale"][1], 1.)
        self.assertTrue(bool(domain.feasible(base[:, None]+scales*direction[:, None]).all()))

    def test_constrained_stationary_state_has_no_fabricated_recovery(self):
        domain = JointPositionDomain(torch.tensor([-1.]), torch.tensor([1.]))
        base = torch.ones((1, 1, 1))
        direction, report = domain.recovery_step(base, -base, base[0], self.scales())
        self.assertFalse(report["success"])
        self.assertEqual(report["reason"], "no_representable_descent_direction")
        torch.testing.assert_close(direction, torch.zeros_like(base))

    def test_seed_fit_enforces_continuous_velocity_at_moving_acceleration_roots(self):
        domain = self.domain(.2, .1)
        original_prefix = domain.prefix.clone()
        progress = np.linspace(0., 1., 41)
        targets = np.sin(6*np.pi*progress)[:, None]
        velocity_bounds = torch.tensor([[-.5], [.5]])
        fitted, report = domain.fit_reference(progress, targets, velocity_bounds=velocity_bounds)
        self.assertTrue(report["success"], report)
        self.assertTrue(report["velocity_bounds_enforced"])
        self.assertGreater(report["rounds"], 1)
        low, high = self.velocity_oracle(domain, fitted)
        self.assertGreaterEqual(low, -.5)
        self.assertLessEqual(high, .5)
        self.assertAlmostEqual(report["continuous_max_abs_velocity_joint_units_per_s"][0],
                               max(abs(low), abs(high)), places=13)
        self.assertTrue(bool(domain.feasible(fitted[None]).all()))
        torch.testing.assert_close(domain.prefix, original_prefix)
        np.testing.assert_allclose(domain.evaluate(fitted, [0.], 1), [[.1]], atol=1e-8)

    def test_rounding_contraction_constructs_anchor_for_both_position_and_velocity(self):
        domain = self.domain(0., .1)
        # Constant *free coefficients* do not mean a constant trajectory when
        # the fixed entrance has nonzero velocity. Its undershoot exceeds the
        # requested speed, so blindly reusing that q-only anchor is incorrect.
        velocity_bounds = torch.tensor([[-.11], [.11]])
        position_anchor = domain._construct_anchor(torch.float32, torch.device("cpu"))[0]
        self.assertGreater(max(abs(v) for v in self.velocity_oracle(domain, position_anchor)), .11)
        fitted, report = domain.fit_reference(np.linspace(0., 1., 16), np.full((16, 1), .5),
                                              velocity_bounds=velocity_bounds)
        self.assertTrue(report["success"], report)
        self.assertEqual(report["fit_anchor_source"], "position_velocity_qp")
        low, high = self.velocity_oracle(domain, fitted)
        self.assertGreaterEqual(low, float(velocity_bounds[0, 0]))
        self.assertLessEqual(high, float(velocity_bounds[1, 0]))
        self.assertTrue(bool(domain.feasible(fitted[None]).all()))

    def test_incompatible_fixed_velocity_reports_failure_without_dropping_bound(self):
        domain = self.domain(.2, .1)
        prefix = domain.prefix.clone()
        fitted, report = domain.fit_reference(np.linspace(0., 1., 16), np.zeros((16, 1)),
                                              velocity_bounds=np.array([[-.05], [.05]]))
        self.assertIsNone(fitted)
        self.assertFalse(report["success"])
        self.assertTrue(report["velocity_bounds_enforced"])
        self.assertEqual(report["reason"], "fit_initializer_not_constructed")
        torch.testing.assert_close(domain.prefix, prefix)

    def test_impossible_fixed_entrance_reports_failed_fit_without_rewriting_it(self):
        domain = self.domain(.99, .2)
        original_prefix = domain.prefix.clone()
        fitted, report = domain.fit_reference(np.linspace(0., 1., 16), np.zeros((16, 1)))
        self.assertIsNone(fitted)
        self.assertFalse(report["success"])
        torch.testing.assert_close(domain.prefix, original_prefix)


if __name__ == "__main__":
    unittest.main()
# END tests require neither a robot model nor CUDA allocations.
