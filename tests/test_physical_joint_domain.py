"""CPU analytic oracles for physical cubic domains; no CUDA planning runs."""

# BEGIN physical q, nonzero entrance velocity and native step invariants.
import unittest

import numpy as np
from scipy.interpolate import BSpline, PPoly
import torch

from manipulation.physical_joint_domain import (CubicJointDomain, JointPositionDomain,
                                                JointDomainInitializationError)


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


if __name__ == '__main__':
    unittest.main()
# END CPU domain tests do not assert native GPU or complete robot feasibility.
