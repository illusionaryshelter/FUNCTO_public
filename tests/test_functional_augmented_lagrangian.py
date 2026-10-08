"""CPU contracts for signed event constraints and independent-seed AL updates."""

# BEGIN independent analytic values, gradients, identity and buffer-lifetime tests.
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from manipulation.functional_augmented_lagrangian import FunctionalAugmentedLagrangian
from manipulation.planning_problem import PlanningProblem, PoseEventRequirement


class FunctionalAugmentedLagrangianTests(unittest.TestCase):
    def test_phr_uses_signed_feasible_side_and_multiplier_can_decrease(self):
        state = FunctionalAugmentedLagrangian(("position",))
        like = torch.zeros(1, dtype=torch.float64)
        state.bind(like, 1)
        self.assertTrue(bool(state.update({"position": like+1.})[0]))
        self.assertEqual(state.snapshot()["seeds"][0]["multipliers"], {"position": 2.})
        residual = torch.tensor([-2., -.5, 0., .5], dtype=torch.float64, requires_grad=True)
        terms = state.terms({"position": residual})["position"]
        full_phr = torch.tensor([-1., -.75, 0., 1.25], dtype=torch.float64)
        torch.testing.assert_close(terms-full_phr, torch.ones(4, dtype=torch.float64))
        terms.sum().backward()
        torch.testing.assert_close(residual.grad, torch.tensor([0., 1., 2., 3.], dtype=torch.float64))
        state.update({"position": like-.5})
        report = state.snapshot()["seeds"][0]
        self.assertEqual(report["multipliers"], {"position": 1.})
        self.assertEqual(report["penalty"], 2.)
        self.assertEqual(report["previous_complementarity_residual"], .5)

    def test_seed_major_particles_and_independent_parameters_preserve_addresses(self):
        state = FunctionalAugmentedLagrangian(("first", "second"))
        like = torch.zeros(2, dtype=torch.float64)
        state.bind(like, 2)
        pointers = (state._multiplier_buffer.data_ptr(), state._penalty_buffer.data_ptr())
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            state.update({"first": like.repeat(3), "second": like.repeat(3)})
        self.assertEqual(before, state.snapshot())
        state.update({"first": like+torch.tensor([1., -1.]), "second": like})
        state.update({"first": like+torch.tensor([1., -1.]), "second": like})
        # Only seed zero stalls; the peer retains lambda=0, rho=2.
        reports = state.snapshot()["seeds"]
        self.assertEqual([row["penalty"] for row in reports], [20., 2.])
        self.assertEqual([row["multipliers"]["first"] for row in reports], [4., 0.])
        values = state.terms({"first": like.new_zeros(6), "second": like.new_zeros(6)})
        torch.testing.assert_close(values["first"], like.new_tensor([.4, .4, .4, 0., 0., 0.]))
        state.bind(like)
        state.reset()
        self.assertEqual(pointers, (state._multiplier_buffer.data_ptr(), state._penalty_buffer.data_ptr()))
        self.assertEqual(before, state.snapshot())
        state.bind(like, 1)
        self.assertEqual(state._multiplier_buffer.shape, (1, 2))
        self.assertEqual(state.snapshot()["seeds"][0]["outer_updates"], 0)

    def test_stalled_complementarity_grows_bounded_penalty_and_rejects_nonfinite(self):
        state = FunctionalAugmentedLagrangian(("event",))
        residual = torch.ones(1, dtype=torch.float64)
        state.bind(residual)
        state.update({"event": residual})
        self.assertEqual(state.snapshot()["seeds"][0]["penalty"], 2.)
        state.update({"event": residual})
        self.assertEqual(state.snapshot()["seeds"][0]["penalty"], 20.)
        for _ in range(8):
            state.update({"event": residual})
        report = state.snapshot()["seeds"][0]
        self.assertEqual(report["penalty"], 1e6)
        self.assertTrue(report["penalty_capped"])
        before = state.snapshot()
        for invalid in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                state.update({"event": residual*invalid})
            self.assertEqual(before, state.snapshot())

    def test_empty_or_changed_event_identity_is_explicit(self):
        empty = FunctionalAugmentedLagrangian(())
        empty.bind(torch.zeros(1))
        self.assertEqual(empty.terms({}), {})
        self.assertFalse(bool(empty.update({}).any()))
        with self.assertRaisesRegex(ValueError, "identities"):
            empty.update({"new": torch.zeros(1)})
        with self.assertRaisesRegex(ValueError, "distinct"):
            FunctionalAugmentedLagrangian(("same", "same"))

    def test_finite_exact_phr_subproblems_can_remain_strictly_infeasible(self):
        # min (x-1)^2 subject to x<=0 has inner minimizer
        # x=(2-lambda)/(2+rho). Four finite blocks do not guarantee feasibility.
        state = FunctionalAugmentedLagrangian(("x",))
        state.bind(torch.zeros(1, dtype=torch.float64))
        points = []
        for block in range(4):
            x = (2-state._multiplier_buffer[:, 0])/(2+state._penalty_buffer)
            points.append(float(x))
            self.assertGreater(float(x), 0.)
            if block < 3:
                state.update({"x": x})
        self.assertLess(points[-1], points[0])
        self.assertFalse(any(point <= 0. for point in points))

    def test_exact_event_masks_preserve_suffix_ids_duplicate_events_and_missing_values(self):
        past = PoseEventRequirement(.1, np.eye(4), np.eye(4), .01, .02, "past goal")
        first = PoseEventRequirement(.5, np.eye(4), np.eye(4), .01, .02, "first goal")
        second = PoseEventRequirement(.5, np.eye(4), np.eye(4), .02, .04, "distinct same-time goal")
        request = SimpleNamespace(start_progress=.2, pose_events=(past, first, second), coordinate_events=())
        expected = ("function_pose:1:position", "function_pose:1:rotation",
                    "function_pose:2:position", "function_pose:2:rotation")
        self.assertEqual(PlanningProblem.functional_constraint_keys.fget(request), expected)
        progress = torch.tensor([[.5, .8, .5, .5], [.2, .3, .8, 1.]], dtype=torch.float64)
        times = torch.tensor([[.7, 1., .5, .5], [.2, .3, .8, 1.]], dtype=torch.float64)
        position = torch.zeros((2, 4, 3), dtype=torch.float64)
        position[:, :, 0] = .005
        position.requires_grad_()
        quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(2, 4, 4)
        signed, masks = PlanningProblem.functional_inequalities(request, position, quaternion, progress,
                                                                trajectory_time=times)
        self.assertEqual(tuple(signed), expected)
        self.assertEqual(float(signed[expected[0]][0].detach()), -.5)
        self.assertEqual(float(signed[expected[2]][0].detach()), -.75)
        self.assertTrue(all(torch.isnan(value[1]) for value in signed.values()))
        for mask in masks.values():
            torch.testing.assert_close(mask, torch.tensor([[False, False, True, False],
                                                          [False, False, False, False]]))
        derivative = torch.autograd.grad(signed[expected[0]][0], position)[0]
        torch.testing.assert_close(derivative[0, :, 0], torch.tensor([0., 0., 100., 0.], dtype=torch.float64))
        self.assertEqual(float(derivative[1].abs().sum()), 0.)
# END no GPU, physical collision oracle or task-specific target is required.


if __name__ == "__main__":
    unittest.main()
