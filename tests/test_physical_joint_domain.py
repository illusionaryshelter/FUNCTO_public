"""CPU analytic oracles for physical cubic domains; no CUDA planning runs."""

# BEGIN physical q, nonzero entrance velocity and native step invariants.
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import BSpline, PPoly
import torch

from manipulation.physical_joint_domain import (CubicJointDomain, JointPositionDomain,
                                                JointDomainInitializationError)


class PhysicalJointDomainTest(unittest.TestCase):
    def domain(self, q=0., dq=0., knots=8, limits=(1., 100., 1000., 10000.)):
        bounds = torch.tensor([[-value, value] for value in limits])[:, :, None]
        return CubicJointDomain(bounds,
            torch.tensor([q]), torch.tensor([dq]), .2, knots)

    def oracle(self, domain, controls):
        coefficients = domain.coefficients(controls).numpy()[0, :, 0]
        spline = BSpline(np.arange(-3., domain.knots+6.), coefficients, 3)
        polynomial = PPoly.from_spline(spline)
        roots = polynomial.derivative().roots(extrapolate=False)
        times = np.unique(np.r_[np.arange(domain.intervals+1), roots[np.isfinite(roots)
            & (roots >= 0) & (roots <= domain.intervals)]])
        return polynomial(times).min(), polynomial(times).max()

    def test_analytic_domain_matches_independent_scipy_extrema(self):
        domain = self.domain(q=.2, dq=.1)
        rng = np.random.default_rng(19)
        for _ in range(30):
            controls = torch.tensor(rng.uniform(-2, 2, (1, 8, 1)), dtype=torch.float32)
            low, high = domain.extrema(controls)
            expected = self.oracle(domain, controls)
            np.testing.assert_allclose([low.item(), high.item()], expected, atol=1e-12, rtol=0.)

    # BEGIN one continuous domain checks every physical derivative.
    def test_all_four_extrema_match_independent_segment_polynomials(self):
        domain = self.domain(q=.2, dq=.1)
        rng = np.random.default_rng(29)
        for _ in range(20):
            controls = torch.tensor(rng.uniform(-1., 1., (1, 8, 1)), dtype=torch.float32)
            coefficients = domain.coefficients(controls).numpy()[0, :, 0]
            polynomial = PPoly.from_spline(BSpline(np.arange(-3., 14.), coefficients, 3))
            low, high = domain.all_extrema(controls)
            for derivative in range(4):
                differentiated = polynomial.derivative(derivative)
                if derivative == 3:
                    times = np.arange(domain.intervals)+.5
                else:
                    roots = differentiated.derivative().roots(extrapolate=False)
                    times = np.r_[np.arange(domain.intervals+1), roots[np.isfinite(roots)
                        & (roots >= 0.) & (roots <= domain.intervals)]]
                expected = differentiated(times)/(.2**derivative)
                np.testing.assert_allclose([low[0, derivative, 0], high[0, derivative, 0]],
                    [expected.min(), expected.max()], atol=4e-12, rtol=1e-13)
                query, cpu_values = domain._curve_map.extrema_times(controls[0, :, 0].numpy(), 0, derivative)
                np.testing.assert_allclose([cpu_values.min(), cpu_values.max()],
                    [expected.min(), expected.max()], atol=4e-12, rtol=1e-13)
                self.assertTrue(bool(np.isfinite(query).all()))

    def test_velocity_acceleration_and_jerk_each_can_exclude_a_position_legal_curve(self):
        controls = torch.zeros((1, 8, 1))
        controls[0, 3, 0] = .2
        wide = self.domain()
        low, high = wide.all_extrema(controls)
        self.assertTrue(bool(wide.feasible(controls).all()))
        for derivative in (1, 2, 3):
            limits = [1., 100., 1000., 10000.]
            limits[derivative] = float(torch.maximum(low.abs(), high.abs())[0, derivative, 0])*.8
            constrained = self.domain(limits=limits)
            self.assertFalse(bool(constrained.feasible(controls).all()))
            initialized = constrained.initialize(controls)
            self.assertTrue(bool(constrained.feasible(initialized).all()))
            scales = torch.tensor([0., .2, 1.5]).reshape(1, 3, 1, 1)
            step = constrained.limit_step(initialized, controls, scales)
            self.assertTrue(bool(constrained.feasible(initialized[:, None]+scales*step[:, None]).all()))

    def test_initial_velocity_outside_model_bounds_is_an_algebraic_failure(self):
        domain = self.domain(dq=.6, limits=(1., .5, 2., 20.))
        with self.assertRaisesRegex(JointDomainInitializationError, "algebraic_entrance_invalid"):
            domain.initialize(torch.zeros((1, 8, 1)))

    def test_stopped_terminal_interval_has_zero_derivatives_on_both_interfaces(self):
        domain = self.domain(dq=.1)
        controls = torch.linspace(-.2, .3, 8).reshape(1, 8, 1)
        for derivative in (1, 2, 3):
            actual = domain.evaluate(controls, [.9, .95, 1.], derivative)
            np.testing.assert_array_equal(actual, np.zeros_like(actual))
            weights, offset = domain._curve_map.query_knot_times([9., 9.5, 10.], derivative)
            np.testing.assert_array_equal(weights, np.zeros_like(weights))
            np.testing.assert_array_equal(offset, np.zeros_like(offset))
    # END finite initial q/dq does not substitute for continuous derivative bounds.

    # BEGIN instantaneous entrance regressions replace fixed-prefix assertions.
    def test_control_box_does_not_hide_nonzero_velocity_curve_violation(self):
        domain = self.domain(q=.995, dq=.2)
        controls = torch.full((1, 8, 1), .995)
        self.assertTrue(bool((controls.abs() <= 1).all()))
        self.assertFalse(bool(domain.feasible(controls).all()))
        initialized = domain.initialize(controls)
        self.assertTrue(bool(domain.feasible(initialized).all()))
        coefficients = domain.coefficients(initialized).numpy()[0, :, 0]
        curve = BSpline(np.arange(-3., 14.), coefficients, 3)
        self.assertAlmostEqual(curve(0), float(torch.tensor(.995)), places=12)
        self.assertAlmostEqual(curve(0, 1)/.2, float(torch.tensor(.2)), places=12)

    def test_outward_velocity_at_physical_bound_is_reported_without_rewriting_actual(self):
        domain = self.domain(q=1., dq=.2)
        original_q, original_dq = domain._start_q.clone(), domain._start_dq.clone()
        with self.assertRaises(JointDomainInitializationError):
            domain.initialize(torch.zeros((1, 8, 1)))
        torch.testing.assert_close(domain._start_q, original_q, rtol=0, atol=0)
        torch.testing.assert_close(domain._start_dq, original_dq, rtol=0, atol=0)

    def test_near_boundary_can_brake_before_the_old_fixed_interval(self):
        domain = self.domain(q=.99, dq=.2)
        control = domain.initialize(torch.zeros((1, 8, 1)))
        self.assertTrue(bool(domain.feasible(control).all()))
        q = domain.evaluate(control, [0., .05])[0, :, 0]
        self.assertEqual(q[0], float(torch.tensor(.99)))
        self.assertNotEqual(q[1], q[0]+.1*float(torch.tensor(.2)))

    def test_exact_position_boundary_has_no_fabricated_roundoff_violation(self):
        for q, inward in ((1., -.2), (-1., .2)):
            for dq in (0., inward):
                domain = self.domain(q=q, dq=dq)
                controls = torch.full((1, 8, 1), q)
                self.assertTrue(bool(domain.feasible(controls).all()))
                low, high = domain.extrema(controls)
                self.assertEqual(float(high if q > 0 else low), q)
                self.assertEqual(domain.evaluate(controls, [0.])[0, 0, 0], q)
                self.assertEqual(domain.evaluate(controls, [0.], 1)[0, 0, 0], float(torch.tensor(dq)))
    # END genuine instantaneous infeasibility stays distinct from a fixed-prefix artefact.

    def test_every_native_particle_is_legal_and_update_can_move(self):
        domain = self.domain()
        base = torch.zeros((2, 8, 1))
        direction = torch.full_like(base, 4.)
        scales = torch.tensor([0., .1, .5, 1.]).reshape(1, 4, 1, 1)
        step = domain.limit_step(base, direction, scales)
        trials = base[:, None]+scales*step[:, None]
        self.assertTrue(bool(domain.feasible(trials).all()))
        self.assertGreater(float(step.abs().sum()), 1.)
        self.assertLess(float(step.max()), 4.)

    def test_bad_seed_is_isolated_using_an_already_valid_other_seed(self):
        domain = self.domain()
        seeds = torch.zeros((1, 2, 8, 1))
        seeds[:, 1] = float('nan')
        result = domain.initialize(seeds)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(bool(domain.feasible(result).all()))
        torch.testing.assert_close(result[:, 0], seeds[:, 0])

    def test_ik_domain_uses_full_named_interval(self):
        domain = JointPositionDomain(torch.tensor([-2., .1]), torch.tensor([3., .8]))
        base = torch.tensor([[[0., .5]]])
        scales = torch.tensor([0., .5, 1.]).reshape(1, 3, 1, 1)
        step = domain.limit_step(base, torch.tensor([[[5., -5.]]]), scales)
        self.assertTrue(bool(domain.feasible(base[:, None]+scales*step[:, None]).all()))
        self.assertGreater(float(step[0, 0, 0]), 0.)

    # BEGIN eager validation preserves legal trials and isolates nonfinite steps.
    def test_mixed_cubic_steps_preserve_finite_row_and_stop_nonfinite_row(self):
        domain = self.domain(q=.2, dq=.1)
        base = torch.full((2, 8, 1), .2)
        direction = torch.tensor([4., float("nan")]).reshape(2, 1, 1).expand_as(base)
        scales = torch.tensor([0., .1, .5, 1.]).reshape(1, 4, 1, 1)
        limited = domain.limit_step(base, direction, scales)
        self.assertTrue(bool(domain.feasible(base[:, None]+scales*limited[:, None]).all()))
        self.assertGreater(float(limited[0].abs().sum()), 0.)
        self.assertTrue(torch.equal(limited[1], torch.zeros_like(limited[1])))

    def test_public_validation_returns_none_or_raises_the_domain_exception(self):
        domain = self.domain()
        self.assertIsNone(domain.validate(torch.zeros((1, 8, 1))))
        bad = torch.full((1, 8, 1), 3.)
        with self.assertRaises(JointDomainInitializationError):
            domain.validate(bad)
    # END no GPU context is initialized by this regression suite.

    # BEGIN a feasible full step retains its exact value and gradient.
    def test_full_step_shortcut_preserves_values_gradients_and_checks_all_trials(self):
        generator = torch.Generator().manual_seed(37)
        for domain, shape in ((JointPositionDomain(torch.tensor([-1., -1.]),
                               torch.tensor([1., 1.])), (2, 1, 2)),
                              (self.domain(q=.2, dq=.1), (2, 8, 1))):
            for dtype in (torch.float32, torch.float64):
                base = torch.full(shape, .2, dtype=dtype)
                scales = torch.tensor([0., .1, .5, 1., 2.], dtype=dtype).reshape(1, 5, 1, 1)
                direction = torch.randn(shape, generator=generator, dtype=dtype)*.03
                actual_input = direction.clone().requires_grad_()
                with patch.object(domain, "feasible", wraps=domain.feasible) as checked:
                    actual = domain.limit_step_tensor(base, actual_input, scales)
                self.assertEqual(checked.call_count, 2)
                self.assertEqual(checked.call_args_list[1].args[0].shape, (2, 5, *shape[1:]))
                torch.testing.assert_close(actual, direction, rtol=0, atol=0)
                weights = torch.randn(shape, generator=generator, dtype=dtype)
                actual_grad = torch.autograd.grad((actual*weights).sum(), actual_input)[0]
                torch.testing.assert_close(actual_grad, weights, rtol=0, atol=0)

    def test_boundary_mixed_rows_and_nonfinite_steps_keep_every_trial_legal(self):
        domain = JointPositionDomain(torch.tensor([-1., -1.]), torch.tensor([1., 1.]))
        scales = torch.tensor([0., .125, .5, 1.]).reshape(1, 4, 1, 1)
        for direction in ([[[.1, -.1]], [[4., -4.]]],
                          [[[float("nan"), .1]], [[.1, -.1]]],
                          [[[float("inf"), .1]], [[float("-inf"), 0.]]],
                          [[[0., 0.]], [[0., -0.]]]):
            base = torch.tensor([[[.9, -.9]], [[1., -1.]]])
            step = torch.tensor(direction)
            actual = domain.limit_step(base, step, scales)
            self.assertTrue(bool(domain.feasible(base[:, None]+scales*actual[:, None]).all()))
            self.assertTrue(bool(torch.isfinite(actual).all()))
            invalid = ~torch.isfinite(step).all(dim=(-1, -2))
            torch.testing.assert_close(actual[invalid], torch.zeros_like(actual[invalid]), rtol=0, atol=0)

    def test_eager_shortcut_does_not_bypass_invalid_base_or_scales(self):
        domain = JointPositionDomain(torch.tensor([-1.]), torch.tensor([1.]))
        base, step = torch.zeros((1, 1, 1)), torch.zeros((1, 1, 1))
        for scale in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                domain.limit_step(base, step, torch.tensor([scale]).reshape(1, 1, 1, 1))
        with self.assertRaises(JointDomainInitializationError):
            domain.limit_step(base+2., step, torch.ones((1, 1, 1, 1)))
    # END shortcut success requires the unchanged public validation contract.


if __name__ == '__main__':
    unittest.main()
# END CPU domain tests do not assert native GPU or complete robot feasibility.
