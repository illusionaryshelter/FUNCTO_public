"""CPU contracts for signed event constraints and one-candidate AL updates."""

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
        like = torch.zeros((), dtype=torch.float64)
        state.bind(like)
        first = state.update({"position": like+1.})
        self.assertEqual(first["multipliers"], {"position": 2.})
        self.assertEqual(first["penalty"], 2.)
        residual = torch.tensor([-2., -.5, 0., .5], dtype=torch.float64, requires_grad=True)
        terms = state.terms({"position": residual})["position"]
        # The native nonnegative objective omits the x-independent -lambda²/2rho
        # constant. Compare with independently specified full PHR values; their
        # difference is exactly one for this fixed lambda=2, rho=2 subproblem.
        full_phr = torch.tensor([-1., -.75, 0., 1.25], dtype=torch.float64)
        torch.testing.assert_close(terms-full_phr, torch.ones(4, dtype=torch.float64))
        self.assertTrue(bool((terms >= 0).all()))
        # The feasible but still multiplier-active point has derivative one;
        # an ordinary positive hinge on g would incorrectly give it zero.
        terms.sum().backward()
        torch.testing.assert_close(residual.grad, torch.tensor([0., 1., 2., 3.], dtype=torch.float64))
        second = state.update({"position": like-.5})
        self.assertEqual(second["multipliers"], {"position": 1.})
        self.assertEqual(second["penalty"], 2.)
        self.assertEqual(second["complementarity_residual"], .5)

    def test_one_seed_update_and_fixed_device_addresses_survive_reset(self):
        state = FunctionalAugmentedLagrangian(("first", "second"))
        like = torch.zeros((), dtype=torch.float64)
        state.bind(like)
        pointers = (state._multiplier_buffer.data_ptr(), state._penalty_buffer.data_ptr())
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            state.update({"first": torch.tensor([1., -1.]), "second": torch.tensor([0., 0.])})
        self.assertEqual(before, state.snapshot())
        state.update({"second": like-1., "first": like+1.})
        self.assertEqual(state.report["multipliers"], {"first": 2., "second": 0.})
        # Repeated line-search rows read the same selected candidate state and
        # cannot change it merely by evaluating costs in a different batch.
        values = state.terms({"first": like.expand(6), "second": like.expand(6)})
        self.assertEqual(tuple(values["first"].shape), (6,))
        state.bind(like)
        state.reset()
        self.assertEqual(pointers, (state._multiplier_buffer.data_ptr(), state._penalty_buffer.data_ptr()))
        self.assertEqual(before, state.snapshot())
        torch.testing.assert_close(state._multiplier_buffer, torch.zeros(2, dtype=torch.float64))
        self.assertEqual(float(state._penalty_buffer), 2.)

    def test_stalled_complementarity_grows_bounded_penalty_and_reports_limit(self):
        state = FunctionalAugmentedLagrangian(("event",))
        scalar = torch.ones((), dtype=torch.float64)
        state.bind(scalar)
        first = state.update({"event": scalar})
        self.assertFalse(first["penalty_increased"])
        second = state.update({"event": scalar})
        self.assertTrue(second["penalty_increased"])
        self.assertEqual(second["penalty"], 20.)
        for _ in range(8):
            report = state.update({"event": scalar})
        self.assertEqual(report["penalty"], 1e6)
        self.assertTrue(report["penalty_capped"])
        self.assertEqual(report["maximum_normalized_violation"], 1.)
        before = state.snapshot()
        for invalid in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                state.update({"event": scalar*invalid})
            self.assertEqual(before, state.snapshot())

    def test_empty_or_changed_event_identity_is_explicit(self):
        empty = FunctionalAugmentedLagrangian(())
        self.assertEqual(empty.terms({}), {})
        self.assertFalse(empty.update({})["changed"])
        with self.assertRaisesRegex(ValueError, "identities"):
            empty.update({"new": torch.zeros(())})
        with self.assertRaisesRegex(ValueError, "distinct"):
            FunctionalAugmentedLagrangian(("same", "same"))

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
