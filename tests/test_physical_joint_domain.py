"""CPU analytic oracles for physical cubic domains; no CUDA planning runs."""

# BEGIN physical q, nonzero entrance velocity and native step invariants.
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import BSpline, PPoly
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from manipulation.physical_joint_domain import (CubicJointDomain, JointPositionDomain,
                                                JointDomainInitializationError)
from manipulation.optimizer_device_checks import DeviceChecks


class PhysicalJointDomainTest(unittest.TestCase):
    def domain(self, q=0., dq=0., knots=8):
        return CubicJointDomain(torch.tensor([-1.]), torch.tensor([1.]),
            torch.tensor([q]), torch.tensor([dq]), .2, knots)

    def oracle(self, domain, controls):
        coefficients = domain.coefficients(controls).numpy()[0, :, 0]
        spline = BSpline(np.arange(-3., domain.knots+8.), coefficients, 3)
        polynomial = PPoly.from_spline(spline)
        roots = polynomial.derivative().roots(extrapolate=False)
        times = np.unique(np.r_[np.arange(domain.knots+5), roots[np.isfinite(roots)
            & (roots >= 0) & (roots <= domain.knots+4)]])
        return polynomial(times).min(), polynomial(times).max()

    def test_analytic_domain_matches_independent_scipy_extrema(self):
        domain = self.domain(q=.2, dq=.1)
        rng = np.random.default_rng(19)
        for _ in range(30):
            controls = torch.tensor(rng.uniform(-2, 2, (1, 8, 1)), dtype=torch.float32)
            low, high = domain.extrema(controls)
            expected = self.oracle(domain, controls)
            np.testing.assert_allclose([low.item(), high.item()], expected, atol=1e-12, rtol=0.)

    def test_control_box_does_not_hide_nonzero_velocity_prefix_violation(self):
        domain = self.domain(q=.94, dq=.2)
        controls = torch.full((1, 8, 1), .94)
        self.assertTrue(bool((controls.abs() <= 1).all()))
        self.assertFalse(bool(domain.feasible(controls).all()))
        initialized = domain.initialize(controls)
        self.assertTrue(bool(domain.feasible(initialized).all()))
        coefficients = domain.coefficients(initialized).numpy()[0, :, 0]
        curve = BSpline(np.arange(-3., 16.), coefficients, 3)
        self.assertAlmostEqual(curve(0), float(torch.tensor(.94)), places=12)
        self.assertAlmostEqual(curve(0, 1)/.2, float(torch.tensor(.2)), places=12)

    def test_impossible_fixed_prefix_is_reported_without_rewriting_actual(self):
        domain = self.domain(q=.99, dq=.2)
        original = domain.prefix.clone()
        with self.assertRaises(JointDomainInitializationError):
            domain.initialize(torch.zeros((1, 8, 1)))
        torch.testing.assert_close(domain.prefix, original)

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

    # BEGIN enforce device-only arithmetic without needing a CUDA allocation.
    def test_tensor_step_map_never_extracts_a_python_scalar(self):
        class RejectScalarRead(TorchDispatchMode):
            def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
                if operation is torch.ops.aten._local_scalar_dense.default:
                    raise AssertionError("A tensor step read a host scalar")
                return operation(*args, **(kwargs or {}))

        domain = self.domain(q=.2, dq=.1)
        base = torch.full((2, 8, 1), .2)
        direction = torch.tensor([4., float("nan")]).reshape(2, 1, 1).expand_as(base)
        scales = torch.tensor([0., .1, .5, 1.]).reshape(1, 4, 1, 1)
        # BEGIN graph and warmup scopes retain the original tensor-only map.
        with DeviceChecks("cpu").scope(), RejectScalarRead():
            limited = domain.limit_step_tensor(base, direction, scales)
        # END eager calls may now read one full-step feasibility predicate.
        self.assertTrue(bool(domain.feasible(base[:, None]+scales*limited[:, None]).all()))
        self.assertTrue(torch.equal(limited[1], torch.zeros_like(limited[1])))

    def test_public_validation_stays_strict_and_scoped_faults_keep_error_type(self):
        domain = self.domain()
        bad = torch.full((1, 8, 1), 3.)
        with self.assertRaises(JointDomainInitializationError):
            domain.validate(bad)
        checks = DeviceChecks("cpu")
        with checks.scope():
            predicate = domain.validate(bad)
        self.assertEqual(predicate.dtype, torch.bool)
        with self.assertRaises(JointDomainInitializationError):
            checks.raise_if_failed()
        checks.reset()
        checks.raise_if_failed()
    # END no GPU context is initialized by this regression suite.

    # BEGIN exact eager shortcut comparisons against the retained graph map.
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
                reference_input = direction.clone().requires_grad_()
                with patch.object(domain, "feasible", wraps=domain.feasible) as checked:
                    actual = domain.limit_step_tensor(base, actual_input, scales)
                self.assertEqual(checked.call_count, 2)
                self.assertEqual(checked.call_args_list[1].args[0].shape, (2, 5, *shape[1:]))
                with DeviceChecks("cpu").scope():
                    expected = domain.limit_step_tensor(base, reference_input, scales)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                weights = torch.randn(shape, generator=generator, dtype=dtype)
                actual_grad = torch.autograd.grad((actual*weights).sum(), actual_input)[0]
                expected_grad = torch.autograd.grad((expected*weights).sum(), reference_input)[0]
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_boundary_mixed_rows_and_nonfinite_steps_keep_exact_original_map(self):
        domain = JointPositionDomain(torch.tensor([-1., -1.]), torch.tensor([1., 1.]))
        scales = torch.tensor([0., .125, .5, 1.]).reshape(1, 4, 1, 1)
        for direction in ([[[.1, -.1]], [[4., -4.]]],
                          [[[float("nan"), .1]], [[.1, -.1]]],
                          [[[float("inf"), .1]], [[float("-inf"), 0.]]],
                          [[[0., 0.]], [[0., -0.]]]):
            base = torch.tensor([[[.9, -.9]], [[1., -1.]]])
            step = torch.tensor(direction)
            actual = domain.limit_step(base, step, scales)
            with DeviceChecks("cpu").scope():
                expected = domain.limit_step_tensor(base, step, scales)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(bool(domain.feasible(base[:, None]+scales*actual[:, None]).all()))
            self.assertTrue(bool(torch.isfinite(actual).all()))

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
