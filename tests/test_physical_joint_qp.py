"""CPU oracles for fixed-clock fitting and continuous-boundary recovery."""

# BEGIN independent curve and objective checks for the shared CPU QP interface.
import unittest

import numpy as np
from scipy.interpolate import BSpline, PPoly
import torch

from manipulation.physical_joint_domain import (CubicJointDomain, JointPositionDomain,
                                                JointDomainInitializationError)


class PhysicalJointQPTest(unittest.TestCase):
    @staticmethod
    def domain(q=0., dq=0., limits=(1., 100., 1000., 10000.)):
        bounds = torch.tensor([[-value, value] for value in limits])[:, :, None]
        return CubicJointDomain(bounds,
            torch.tensor([q]), torch.tensor([dq]), .2, 8)

    @staticmethod
    def scales(maximum=1.):
        return torch.tensor([0., .2, maximum]).reshape(1, 3, 1, 1)

    def velocity_oracle(self, domain, controls):
        coefficients = domain.coefficients(controls[None])[0, :, 0].numpy()
        velocity = PPoly.from_spline(BSpline(np.arange(-3., domain.knots+6.), coefficients, 3)).derivative()
        roots = velocity.derivative().roots(extrapolate=False)
        times = np.r_[np.arange(domain.intervals+1), roots[np.isfinite(roots) & (roots >= 0.) & (roots <= domain.intervals)]]
        values = velocity(times)/.2
        return values.min(), values.max()

    def test_affine_sampling_matches_independent_curve_and_physical_derivatives(self):
        domain = self.domain(.2, .1)
        controls = torch.linspace(-.3, .5, 8).reshape(1, 8, 1)
        coefficients = domain.coefficients(controls)[0, :, 0].numpy()
        spline = BSpline(np.arange(-3., domain.knots+6.), coefficients, 3)
        progress = np.linspace(0., 1., 53)
        for derivative in range(4):
            actual = domain.evaluate(controls, progress, derivative)[0, :, 0]
            expected = spline(progress*domain.intervals, nu=derivative)/(.2**derivative)
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
        original_q, original_dq = domain._start_q.clone(), domain._start_dq.clone()
        progress = np.linspace(0., 1., 41)
        targets = (2*progress)[:, None]
        fitted, report = domain.fit_reference(progress, targets)
        self.assertTrue(report["success"], report)
        self.assertTrue(bool(domain.feasible(fitted[None]).all()))
        self.assertGreaterEqual(report["reference_max_error_joint_units"], 1.)
        self.assertGreater(report["fit_contraction_scale"], .99)
        self.assertLessEqual(report["fit_contraction_scale"], 1.)
        torch.testing.assert_close(domain._start_q, original_q, rtol=0, atol=0)
        torch.testing.assert_close(domain._start_dq, original_dq, rtol=0, atol=0)
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
        bounds = torch.tensor([[-1., 1.], [-100., 100.], [-1000., 1000.], [-10000., 10000.]])
        domain = CubicJointDomain(bounds[:, :, None].expand(-1, -1, 2),
            torch.zeros(2), torch.zeros(2), .2, 16)
        base = torch.zeros((1, 16, 2))
        base[..., 0] = 1.
        # The first joint is on a long platform; its free entrance controls
        # can still move, but its nearly active interior extrema round tightly.
        # The second joint is in the interior and has an ordinary descent step.
        gradient = torch.full_like(base, -1.)
        gradient[..., 0] = -6.
        scales = self.scales(1.5)
        direction, report = domain.recovery_step(base, gradient, torch.full((16, 2), 1.96), scales)
        self.assertTrue(report["success"], report)
        self.assertGreater(float(direction[..., 1].abs().min()), .1)
        self.assertEqual(report["per_joint_limiter_scale"][1], 1.)
        self.assertTrue(bool(domain.feasible(base[:, None]+scales*direction[:, None]).all()))
        self.assertLessEqual(float((scales.max()*direction).abs().max()), 1.96)

    def test_constrained_stationary_state_has_no_fabricated_recovery(self):
        domain = JointPositionDomain(torch.tensor([-1.]), torch.tensor([1.]))
        base = torch.ones((1, 1, 1))
        direction, report = domain.recovery_step(base, -base, base[0], self.scales())
        self.assertFalse(report["success"])
        self.assertEqual(report["reason"], "no_representable_descent_direction")
        torch.testing.assert_close(direction, torch.zeros_like(base))

    def test_seed_fit_enforces_continuous_velocity_at_moving_acceleration_roots(self):
        domain = self.domain(.2, .1, limits=(1., .5, 1000., 10000.))
        original_q, original_dq = domain._start_q.clone(), domain._start_dq.clone()
        progress = np.linspace(0., 1., 41)
        targets = np.sin(6*np.pi*progress)[:, None]
        fitted, report = domain.fit_reference(progress, targets)
        self.assertTrue(report["success"], report)
        self.assertEqual(report["physical_derivative_orders"], [0, 1, 2, 3])
        self.assertGreater(report["rounds"], 1)
        low, high = self.velocity_oracle(domain, fitted)
        self.assertGreaterEqual(low, -.5)
        self.assertLessEqual(high, .5)
        self.assertAlmostEqual(report["continuous_max_abs_velocity_joint_units_per_s"][0],
                               max(abs(low), abs(high)), places=13)
        self.assertTrue(bool(domain.feasible(fitted[None]).all()))
        torch.testing.assert_close(domain._start_q, original_q, rtol=0, atol=0)
        torch.testing.assert_close(domain._start_dq, original_dq, rtol=0, atol=0)
        np.testing.assert_allclose(domain.evaluate(fitted, [0.], 1), [[.1]], atol=1e-8)

    def test_rounding_contraction_uses_one_anchor_for_all_four_model_bounds(self):
        domain = self.domain(0., .1, limits=(1., .11, .4, 2.))
        fitted, report = domain.fit_reference(np.linspace(0., 1., 16), np.full((16, 1), .5))
        self.assertTrue(report["success"], report)
        self.assertEqual(report["fit_anchor_source"], "physical_domain_anchor")
        low, high = self.velocity_oracle(domain, fitted)
        self.assertGreaterEqual(low, float(domain.bounds[1, 0, 0]))
        self.assertLessEqual(high, float(domain.bounds[1, 1, 0]))
        self.assertTrue(bool(domain.feasible(fitted[None]).all()))

    def test_incompatible_fixed_velocity_reports_failure_without_dropping_bound(self):
        domain = self.domain(.2, .1, limits=(1., .05, 2., 20.))
        original_q, original_dq = domain._start_q.clone(), domain._start_dq.clone()
        fitted, report = domain.fit_reference(np.linspace(0., 1., 16), np.zeros((16, 1)))
        self.assertIsNone(fitted)
        self.assertFalse(report["success"])
        self.assertEqual(report["physical_derivative_orders"], [0, 1, 2, 3])
        self.assertEqual(report["reason"], "algebraic_entrance_invalid")
        torch.testing.assert_close(domain._start_q, original_q, rtol=0, atol=0)
        torch.testing.assert_close(domain._start_dq, original_dq, rtol=0, atol=0)

    def test_impossible_fixed_entrance_reports_failed_fit_without_rewriting_it(self):
        domain = self.domain(1., .2)
        original_q, original_dq = domain._start_q.clone(), domain._start_dq.clone()
        fitted, report = domain.fit_reference(np.linspace(0., 1., 16), np.zeros((16, 1)))
        self.assertIsNone(fitted)
        self.assertFalse(report["success"])
        torch.testing.assert_close(domain._start_q, original_q, rtol=0, atol=0)
        torch.testing.assert_close(domain._start_dq, original_dq, rtol=0, atol=0)

    # BEGIN nonzero measured velocity is feasible only with sufficient braking room.
    def test_qp_anchor_brakes_with_the_full_model_acceleration_and_jerk_limits(self):
        domain = self.domain(.8, .6, limits=(1., .8, 2., 20.))
        proposal = torch.full((1, 8, 1), .8)
        self.assertFalse(bool(domain.feasible(proposal).all()))
        initialized = domain.initialize(proposal)
        self.assertTrue(bool(domain.feasible(initialized).all()))
        np.testing.assert_array_equal(domain.evaluate(initialized, [0.])[0, 0],
                                      domain._start_q.numpy())
        np.testing.assert_array_equal(domain.evaluate(initialized, [0.], 1)[0, 0],
                                      domain._start_dq.numpy())
        for derivative in range(4):
            _, values = domain._curve_map.extrema_times(initialized[0, :, 0].numpy(), 0, derivative)
            self.assertGreaterEqual(float(values.min()), float(domain.bounds[derivative, 0, 0]))
            self.assertLessEqual(float(values.max()), float(domain.bounds[derivative, 1, 0]))

    def test_inside_q_and_velocity_intervals_can_still_fail_the_braking_qp(self):
        domain = self.domain(.99, .6, limits=(1., .8, .5, 20.))
        self.assertTrue(bool(domain._entrance_valid))
        # Even maximum braking requires v^2/(2*a)=.36 joint units while
        # only .01 remains. The finite constrained construction must fail.
        fitted, report = domain.fit_reference(np.linspace(0., 1., 41), np.zeros((41, 1)))
        self.assertIsNone(fitted)
        self.assertEqual(report["reason"], "finite_constraints_inconsistent")
        with self.assertRaisesRegex(JointDomainInitializationError, "finite_constraints_inconsistent"):
            domain.initialize(torch.full((1, 8, 1), .99))

    def test_fixed_terminal_stop_needs_more_time_than_the_bound_clock_allows(self):
        domain = self.domain(0., 1., limits=(100., 2., .1, 2.))
        self.assertTrue(bool(domain._entrance_valid))
        self.assertTrue(bool(((domain._start_q >= domain.bounds[0, 0])
                              & (domain._start_q <= domain.bounds[0, 1])).all()))
        self.assertTrue(bool(((domain._start_dq >= domain.bounds[1, 0])
                              & (domain._start_dq <= domain.bounds[1, 1])).all()))
        duration = domain.intervals*domain._curve_map.knot_dt
        # The wide position interval leaves room, while |a| <= .1 requires
        # at least ten seconds to stop the measured unit velocity. The
        # immutable two-second cubic includes its original stopped end.
        self.assertEqual(duration, 2.)
        self.assertLess(duration, float(domain._start_dq/domain.bounds[2, 1]))
        fitted, report = domain.fit_reference(np.linspace(0., 1., 41), np.zeros((41, 1)))
        self.assertIsNone(fitted)
        self.assertFalse(report["success"])
        self.assertEqual(report["reason"], "finite_constraints_inconsistent")
        self.assertEqual(float(domain._start_q), 0.)
        self.assertEqual(float(domain._start_dq), 1.)

    def test_tight_acceleration_and_jerk_fit_and_recovery_share_all_trial_bounds(self):
        domain = self.domain(0., .1, limits=(1., .5, .2, 1.))
        progress = np.linspace(0., 1., 41)
        fitted, report = domain.fit_reference(progress, np.linspace(0., .2, 41)[:, None])
        self.assertTrue(report["success"], report)
        self.assertTrue(bool(domain.feasible(fitted[None]).all()))
        self.assertGreater(report["maximum_bound_excess_before_contraction"][2], 0.)
        self.assertGreater(report["maximum_bound_excess_before_contraction"][3], 0.)
        self.assertLess(report["fit_contraction_scale"], 1.)
        # The actual float32 action is independently checked by piecewise
        # polynomial extrema; no model bound is widened by these assertions.
        for derivative in range(4):
            _, values = domain._curve_map.extrema_times(fitted[:, 0].numpy(), 0, derivative)
            self.assertGreaterEqual(float(values.min()), float(domain.bounds[derivative, 0, 0]))
            self.assertLessEqual(float(values.max()), float(domain.bounds[derivative, 1, 0]))
        base = domain.initialize(torch.zeros((1, 8, 1)))
        direction, recovery = domain.recovery_step(base, -torch.ones_like(base),
                                                   torch.full((8, 1), .1), self.scales(1.5))
        self.assertTrue(recovery["success"], recovery)
        trials = base[:, None]+self.scales(1.5)*direction[:, None]
        self.assertTrue(bool(domain.feasible(trials).all()))
        self.assertLessEqual(float((1.5*direction).abs().max()), float(torch.tensor(.1)))
    # END algebraic entrance checks and bounded continuous construction stay distinct.



if __name__ == "__main__":
    unittest.main()
# END tests require neither a robot model nor CUDA allocations.
