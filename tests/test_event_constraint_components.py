"""Original event identities, physical zero sets and first-arrival derivatives."""

# BEGIN declared event rows retain their material geometry and physical units.
import unittest
from unittest.mock import patch

import numpy as np
import torch

from manipulation.goals import SupportSurface
from manipulation.planning_problem import (
    MotionPhase, NumericalTolerance, PlanningProblem, PoseEventRequirement, PosePath, SupportRequirement,
)


def request_problem(*, pose_events=(), supports=()):
    reference = PosePath([0., 1.], np.repeat(np.eye(4)[None], 2, axis=0), "source declaration")
    return PlanningProblem(("q",), [0.], [0.], "tool", "world", np.eye(4), np.eye(4),
        (MotionPhase("source", reference),), start_progress=.2, pose_events=pose_events,
        supports=supports, numerical=NumericalTolerance(position_m=1e-5))


class EventConstraintContractTests(unittest.TestCase):
    def test_exact_event_masks_preserve_suffix_ids_duplicate_events_and_missing_values(self):
        past = PoseEventRequirement(.1, np.eye(4), np.eye(4), .01, .02, "past goal")
        first = PoseEventRequirement(.5, np.eye(4), np.eye(4), .01, .02, "first goal")
        second = PoseEventRequirement(.5, np.eye(4), np.eye(4), .02, .04, "distinct same-time goal")
        request = request_problem(pose_events=(past, first, second))
        expected = ("function_pose:1:position", "function_pose:1:rotation",
                    "function_pose:2:position", "function_pose:2:rotation")
        progress = torch.tensor([[.5, .8, .5, .5], [.2, .3, .8, 1.]], dtype=torch.float64)
        times = torch.tensor([[.7, 1., .5, .5], [.2, .3, .8, 1.]], dtype=torch.float64)
        position = torch.zeros((2, 4, 3), dtype=torch.float64)
        position[:, :, 0] = .005
        position.requires_grad_()
        quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(2, 4, 4)
        _, values = request.constraint_components(position, quaternion, progress, trajectory_time=times)
        signed = {key: value for key, value in values.items() if key.startswith("function_pose:")}
        self.assertEqual(tuple(signed), expected)
        self.assertEqual(float(signed[expected[0]][0].detach()), -.005)
        self.assertEqual(float(signed[expected[2]][0].detach()), -.015)
        self.assertTrue(all(torch.isnan(value[1]) for value in signed.values()))
        derivative = torch.autograd.grad(signed[expected[0]][0].max(), position)[0]
        torch.testing.assert_close(derivative[0, :, 0], torch.tensor([0., 0., 1., 0.], dtype=torch.float64))
        self.assertEqual(float(derivative[1].abs().sum()), 0.)

    # BEGIN finite support uses four allowed-set components, never a fixed pose.
    @staticmethod
    def support_event(progress=.5, *, gap=.001, normal=.0001):
        contact = np.array([[x, y, 0.] for x in (-.02, .02) for y in (-.02, .02)])
        body = np.concatenate((contact, contact+[0., 0., .05]))
        surface = SupportSurface(np.zeros(3), np.eye(3)[:, :2], [0., 0., 1.],
                                 [[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])
        return SupportRequirement(progress, np.eye(4), body, contact, [0., 0., -1.],
                                  surface, gap, normal, .002, "Explicit synthetic finite support",
                                  hard_components=("gap", "body", "footprint", "normal"))

    def test_support_components_preserve_physical_zero_sets_and_gradient_directions(self):
        event = self.support_event()
        numerical = NumericalTolerance(position_m=1e-5)
        position = torch.tensor([[0., 0., .0005], [0., 0., .0258],
                                 [0., 0., -.0002], [.09, 0., .0005]],
                                dtype=torch.float64, requires_grad=True)
        quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(4, 4).clone().requires_grad_()
        raw = event.raw_inequalities(position, quaternion, numerical)
        signed = event.normalized_inequalities(position, quaternion, numerical)
        hard = event.evaluate(position, quaternion, numerical)
        # Independent metric expectations: 24.8 mm excess gap, 0.19 mm
        # penetration and 11.99 mm boundary excess with the original margin.
        self.assertAlmostEqual(float(raw["gap"][1].detach()), .0248)
        self.assertAlmostEqual(float(raw["body"][2].detach()), .00019)
        self.assertAlmostEqual(float(raw["footprint"][3].detach()), .01199)
        self.assertAlmostEqual(float(signed["footprint"][3].detach()), 11.99)
        expected = torch.tensor([True, False, False, False])
        torch.testing.assert_close(hard <= 0, expected)
        torch.testing.assert_close(torch.stack(list(signed.values())).amax(0) <= 0, expected)
        for name in raw:
            torch.testing.assert_close(raw[name] <= 0, signed[name] <= 0)
        gap_gradient = torch.autograd.grad(signed["gap"][1], position, retain_graph=True)[0]
        boundary_gradient = torch.autograd.grad(signed["footprint"][3], position, retain_graph=True)[0]
        self.assertGreater(float(gap_gradient[1, 2]), 0.)
        self.assertGreater(float(boundary_gradient[3, 0]), 0.)
        # A perfectly aligned face has a finite quaternion derivative, including
        # at chord zero; no acos singularity is introduced by decomposition.
        normal_gradient = torch.autograd.grad(signed["normal"].sum(), quaternion)[0]
        self.assertTrue(bool(torch.isfinite(normal_gradient).all()))

    def test_support_free_tangent_and_yaw_and_original_normal_threshold(self):
        event = self.support_event()
        numerical = NumericalTolerance(position_m=1e-5, rotation_rad=.001)
        angles = torch.tensor([0., .4, -1.2], dtype=torch.float64)
        quaternion = torch.stack((torch.cos(angles/2), torch.zeros_like(angles),
                                  torch.zeros_like(angles), torch.sin(angles/2)), -1)
        position = torch.tensor([[0., 0., .0005], [.02, -.015, .0005], [-.01, .03, .0005]], dtype=torch.float64)
        signed = event.normalized_inequalities(position, quaternion, numerical)
        self.assertTrue(all(bool((value <= 0).all()) for value in signed.values()))
        # The numerical angular scale is larger than the physical declaration;
        # it must not permit this 0.5 mrad tilt under a 0.1 mrad requirement.
        tilt = .0005
        tilted = torch.tensor([[np.cos(tilt/2), np.sin(tilt/2), 0., 0.]], dtype=torch.float64)
        self.assertGreater(float(event.normalized_inequalities(position[:1], tilted, numerical)["normal"]), 0.)
        self.assertGreater(float(event.evaluate(position[:1], tilted, numerical)), 0.)

    def test_support_zero_tolerances_and_shared_scale_do_not_relax_other_requirements(self):
        numerical = NumericalTolerance(position_m=1e-5)
        quaternion = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
        position = torch.tensor([[.09, 0., .0005]], dtype=torch.float64)
        narrow, wide = self.support_event(gap=.0001), self.support_event(gap=.001)
        raw_narrow = narrow.raw_inequalities(position, quaternion, numerical)
        raw_wide = wide.raw_inequalities(position, quaternion, numerical)
        for name in ("body", "footprint", "normal"):
            torch.testing.assert_close(raw_narrow[name], raw_wide[name])
        self.assertGreater(float(narrow.evaluate(position, quaternion, numerical)), 0.)
        self.assertGreater(float(wide.evaluate(position, quaternion, numerical)), 0.)
        exact = self.support_event(gap=0., normal=0.)
        signed = exact.normalized_inequalities(position*0, quaternion, numerical)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in signed.values()))
        self.assertEqual(float(signed["gap"]), 0.)
        self.assertEqual(float(signed["normal"]), 0.)
        lifted = exact.normalized_inequalities(position*0+position.new_tensor([0., 0., 1e-6]),
                                               quaternion, numerical)
        self.assertGreater(float(lifted["gap"]), 0.)
        # NumericalTolerance permits angles larger than pi. Such a scale must
        # use angular distance, not a periodic sine that turns zero/negative.
        large_angle = NumericalTolerance(position_m=1e-5, rotation_rad=4*np.pi)
        tilted = torch.tensor([[np.cos(.001), np.sin(.001), 0., 0.]], dtype=torch.float64)
        large_signed = exact.normalized_inequalities(position*0, tilted, large_angle)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in large_signed.values()))
        self.assertGreater(float(large_signed["normal"]), 0.)

    def test_support_rows_gather_first_event_keep_suffix_ids_and_mark_missing(self):
        past, event = self.support_event(.1), self.support_event(.5)
        request = request_problem(supports=(past, event))
        expected = tuple(f"support:1:{name}" for name in ("gap", "body", "footprint", "normal"))
        progress = torch.tensor([[.5, .8, .5, .5], [.2, .3, .8, 1.]], dtype=torch.float64)
        times = torch.tensor([[.7, 1., .5, .5], [.2, .3, .8, 1.]], dtype=torch.float64)
        position = torch.zeros((2, 4, 3), dtype=torch.float64)
        position[:, :, 2] = .0005
        position.requires_grad_()
        quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(2, 4, 4)
        with patch.object(event, "inequality_components", wraps=event.inequality_components) as evaluated:
            _, values = request.constraint_components(position, quaternion, progress, trajectory_time=times)
        signed = {key: value for key, value in values.items() if key.startswith("support:")}
        self.assertEqual(evaluated.call_count, 1)
        self.assertEqual(evaluated.call_args.args[0].shape, (2, 3))
        self.assertEqual(evaluated.call_args.args[1].shape, (2, 4))
        self.assertEqual(tuple(signed), expected)
        self.assertAlmostEqual(float(signed[expected[0]][0].detach().max()), -.0005)
        self.assertTrue(all(bool(torch.isnan(value[1]).all()) for value in signed.values()))
        derivative = torch.autograd.grad(signed[expected[0]][0].max(), position)[0]
        torch.testing.assert_close(derivative[0, :, 2], torch.tensor([0., 0., 1., 0.], dtype=torch.float64))
        self.assertEqual(float(derivative[1].abs().sum()), 0.)
    # END support scale, event identity and allowed freedom remain generic.
# END no GPU, physical collision oracle or task-specific target is required.


if __name__ == "__main__":
    unittest.main()
