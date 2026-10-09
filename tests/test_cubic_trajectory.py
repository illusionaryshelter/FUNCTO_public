"""Independent CPU curve/gradient checks of the actual native tensor adapter."""

# BEGIN no GPU context or robot assets are needed to test spline integration.
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.interpolate import BSpline
import torch

from manipulation.cubic_trajectory import CubicBasis, CubicTrajectoryStep, cubic_transition_config_type


def output_state(batch, horizon, dof):
    return SimpleNamespace(position=torch.zeros(batch, horizon, dof))


def independent_curve(q, dq, h, action):
    """Direct complete-coefficient construction, independent of production rows."""
    coefficients = np.r_[[(action[0]-2*h*dq)],
                         [(1.5*q+.5*h*dq-.5*action[0])],
                         action, np.repeat(action[-1:], 3, axis=0)]
    return BSpline(np.arange(-3., len(action)+6.), coefficients, 3)


class CubicTrajectoryTests(unittest.TestCase):
    def test_regular_dense_and_scipy_match_with_two_measured_entrances(self):
        step = CubicTrajectoryStep(10, 4, .125, device="cpu")
        q = torch.tensor([[1.4, -.7], [.2, 1.1]])
        dq = torch.tensor([[1e-5, -2e-5], [-.13, .2]])
        start = SimpleNamespace(position=q, velocity=dq)
        indices = torch.tensor([0, 1, 1, 0], dtype=torch.int32)
        action = torch.linspace(-.2, .6, 80).reshape(4, 10, 2)
        goal = SimpleNamespace(dt=torch.full((4,), .125))
        regular = step.forward(start, action, output_state(4, 49, 2), indices,
            goal, torch.arange(4, dtype=torch.int32), torch.zeros(4, dtype=torch.uint8))
        dense = step.sample(action, start, indices, output_state(4, 101, 2),
                            valid_count=97, dense=True, output_dt=.0625)
        before = torch.cuda.is_initialized()
        for index in range(4):
            curve = independent_curve(q[indices[index]].double().numpy(),
                dq[indices[index]].double().numpy(), .5, action[index].double().numpy())
            for derivative, name in enumerate(("position", "velocity", "acceleration", "jerk")):
                expected = curve(np.arange(49)/4, derivative)/(.5**derivative)
                np.testing.assert_allclose(getattr(regular, name)[index].numpy(), expected,
                                           rtol=2e-7, atol=2e-7)
                torch.testing.assert_close(getattr(regular, name)[index],
                    getattr(dense, name)[index, :97:2], rtol=0, atol=0)
            torch.testing.assert_close(regular.position[index, 0], q[indices[index]], rtol=0, atol=0)
            torch.testing.assert_close(regular.velocity[index, 0], dq[indices[index]], rtol=0, atol=0)
        torch.testing.assert_close(dense.position[:, 96:], action[:, -1:, :].expand(-1, 5, -1), rtol=0, atol=0)
        self.assertEqual(torch.cuda.is_initialized(), before)

    def test_first_two_controls_change_independent_entrance_derivatives_and_have_true_gradients(self):
        step = CubicTrajectoryStep(8, 4, .125, device="cpu", dtype=torch.float64)
        start = SimpleNamespace(position=torch.tensor([[.73]], dtype=torch.float64),
                                velocity=torch.tensor([[-2.1e-5]], dtype=torch.float64))
        index = torch.zeros(1, dtype=torch.int32)
        action = torch.full((1, 8, 1), .73, dtype=torch.float64, requires_grad=True)
        result = step.sample(action, start, index, output_state(1, 41, 1), valid_count=41)
        rows = []
        for value in (result.acceleration[0, 0, 0], result.jerk[0, 0, 0]):
            rows.append(torch.autograd.grad(value, action, retain_graph=True)[0][0, :2, 0])
        torch.testing.assert_close(torch.stack(rows), torch.tensor([[12., 0.], [-44., 8.]], dtype=torch.float64))
        first_sample_gradient = torch.autograd.grad(result.position[0, 1, 0], action)[0][0, :2, 0]
        self.assertTrue(bool((first_sample_gradient != 0).all()))

        def sampled(values):
            state = step.sample(values, start, index, output_state(1, 41, 1), valid_count=41)
            return torch.stack((state.position, state.velocity, state.acceleration, state.jerk))

        self.assertTrue(torch.autograd.gradcheck(sampled, (action,), eps=1e-6, atol=1e-8, rtol=1e-7))

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

    def test_native_clock_and_hidden_goal_are_rejected_before_sampling(self):
        step = CubicTrajectoryStep(8, 4, .125, device="cpu")
        start = SimpleNamespace(position=torch.zeros(1, 1), velocity=torch.zeros(1, 1))
        indices = torch.zeros(1, dtype=torch.int32)
        args = (start, torch.zeros(1, 8, 1), output_state(1, 41, 1), indices)
        with self.assertRaisesRegex(ValueError, "fixed request-wide"):
            step.forward(*args, SimpleNamespace(dt=torch.tensor([.25])), indices, torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "hidden terminal"):
            step.forward(*args, SimpleNamespace(dt=torch.tensor([.125])), indices, torch.ones(1))

    def test_rebind_replaces_clock_and_keeps_new_actual_entrance(self):
        step = CubicTrajectoryStep(8, 4, .125, device="cpu")
        indices = torch.zeros(1, dtype=torch.int32)
        action = torch.zeros(1, 8, 1)
        start = SimpleNamespace(position=torch.zeros(1, 1), velocity=torch.zeros(1, 1))
        step.sample(action, start, indices, output_state(1, 41, 1), valid_count=41)
        old = step._regular_weights
        step.bind_clock(.0625)
        start.position.fill_(.4)
        start.velocity.fill_(-.2)
        result = step.sample(action, start, indices, output_state(1, 41, 1), valid_count=41)
        self.assertIsNot(step._regular_weights, old)
        torch.testing.assert_close(result.position[:, 0], start.position, rtol=0, atol=0)
        torch.testing.assert_close(result.velocity[:, 0], start.velocity, rtol=0, atol=0)
        self.assertEqual(step.knot_dt, .25)


if __name__ == "__main__":
    unittest.main()
# END these tests validate the actual adapter, not robot collision feasibility.
