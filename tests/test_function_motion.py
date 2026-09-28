"""Function-frame contracts with one bounded CPU IPOPT demo regression.

The released three-point annotations are used only to reconstruct a historical
test fixture. Production input is already grounded function-frame geometry.
Other tests inject solver outputs to exercise contracts without extra solves.
"""

# BEGIN explicit function-frame and internal-event regressions.
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.function_motion import optimize_trajectory, transfer_function_motion


def pairs(poses):
    return np.stack((poses[:, :3, 3],
                     Rotation.from_matrix(poses[:, :3, :3]).as_euler("xyz")), axis=1)


def example():
    """A rotating body with a material function origin 30 cm from its origin."""
    body = np.repeat(np.eye(4)[None], 10, axis=0)
    angle = .2 + np.maximum(0, np.arange(10) - 5) * .01
    body[:, :3, :3] = Rotation.from_euler("z", angle).as_matrix()
    body_to_function = np.eye(4)
    body_to_function[:3, 3] = [.3, 0., 0.]
    return body @ body_to_function, body


class FunctionMotionTest(unittest.TestCase):
    def test_one_actual_solve_preserves_released_demo_function_origin_and_event(self):
        # Construct only the historical fixture with the released frame gauge:
        # x=center-function, z=x cross (grasp-function), y=z cross x.
        repository = Path(__file__).resolve().parents[1]
        root = repository / "demo_data/pour_demo"
        track = json.loads((root / "tool_pose_track_output/tool_track.json").read_text())
        index = json.loads((root / "key_point_track_output/keyframe_idx.json").read_text())["func"]
        function, center, grasp = (np.asarray(track[name], float)
                                    for name in ("function", "center", "grasp"))
        x = center - function
        x /= np.linalg.norm(x, axis=1)[:, None]
        z = np.cross(x, grasp - function)
        z /= np.linalg.norm(z, axis=1)[:, None]
        reference = np.repeat(np.eye(4)[None], len(function), axis=0)
        reference[:, :3, :3] = np.stack((x, np.cross(z, x), z), axis=-1)
        reference[:, :3, 3] = function
        scene = json.loads((repository / "test_data/pour_test/tool_pose_transfer_output/test_tool_track.json").read_text())
        target_frames = []
        for suffix in ("init", "func"):
            point, center_point, grasp_point = (np.asarray(scene[name + "_" + suffix], float)
                                                 for name in ("function", "center", "grasp"))
            axis = center_point - point
            axis /= np.linalg.norm(axis)
            normal = np.cross(axis, grasp_point - point)
            normal /= np.linalg.norm(normal)
            target = np.eye(4)
            target[:3, :3] = np.column_stack((axis, np.cross(normal, axis), normal))
            target[:3, 3] = point
            target_frames.append(target)
        initial, goal = target_frames
        # Replicate the released tool demo's target-frame z alignment using
        # this fixture's already supplied geometry, without any model call.
        angle = (np.arctan2(initial[1, 3], initial[0, 3])
                 - np.arctan2(function[0, 1], function[0, 0]))
        binding = np.eye(4)
        binding[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
        reference = binding @ reference
        captured = {}

        def same_released_kernel(*args, **kwargs):
            captured["args"], captured["kwargs"] = args, kwargs
            captured["return"] = optimize_trajectory(*args, **kwargs)
            return captured["return"]

        with patch("manipulation.function_motion.optimize_trajectory", side_effect=same_released_kernel) as solve:
            result = transfer_function_motion(reference, initial,
                                              [{"index": index, "pose": goal}])
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(captured["args"][2][0]["index"], index)
        self.assertEqual(len(captured["args"][2]), 1)
        np.testing.assert_allclose(captured["args"][0], pairs(reference), atol=1e-12)
        np.testing.assert_allclose(captured["args"][2][0]["pose"], pairs(goal[None])[0],
                                   atol=1e-12)
        # The adapter's returned geometry must be the same single-event kernel
        # result, without a post-hoc body-origin replacement or endpoint warp.
        raw = np.asarray(captured["return"])
        np.testing.assert_array_equal(result["poses"][:, :3, 3], raw[:, 0])
        np.testing.assert_allclose(result["poses"][:, :3, :3],
                                   Rotation.from_euler("xyz", raw[:, 1]).as_matrix())
        np.testing.assert_allclose(result["poses"][index], goal, atol=1e-6)
        np.testing.assert_allclose(result["poses"][0], initial, atol=1e-6)
        self.assertLess(index, len(reference) - 1)
        self.assertEqual(len(result["poses"]), len(reference))
        self.assertEqual(result["diagnostics"]["solve_attempts"], 1)
        self.assertLess(result["goal_checks"][0]["position_error_m"], 1e-6)

    def test_multiple_internal_events_are_hard_targets_and_function_origin_keeps_lever_arm(self):
        reference, body = example()
        goals = [{"index": 5, "pose": reference[5]}, {"index": 8, "pose": reference[8]}]
        with patch("manipulation.function_motion.optimize_trajectory", return_value=pairs(reference)) as solve:
            result = transfer_function_motion(reference, reference[0], goals)
        self.assertEqual([x["index"] for x in result["goal_checks"]], [5, 8])
        np.testing.assert_allclose(solve.call_args.args[0][:, 0], reference[:, :3, 3])
        self.assertEqual([g["index"] for g in solve.call_args.args[2]], [5, 8])
        self.assertTrue(np.allclose(body[:, :3, 3], 0.))
        self.assertGreater(np.max(np.linalg.norm(np.diff(result["poses"][:, :3, 3], axis=0), axis=1)), .002)
        self.assertLess(result["diagnostics"]["max_translation_increment_violation_m"], 1e-12)
        self.assertIn("not physical velocities", result["diagnostics"]["scope"])

    def test_incompatible_events_rejected_without_solver_or_relaxation(self):
        reference, _ = example()
        goal = reference[8].copy()
        goal[0, 3] += 2.
        with patch("manipulation.function_motion.optimize_trajectory") as solve:
            with self.assertRaisesRegex(ValueError, "incompatible"):
                transfer_function_motion(reference, reference[0], [{"index": 8, "pose": goal}])
        solve.assert_not_called()

    def test_duplicate_out_of_range_and_initial_event_indices_rejected(self):
        reference, _ = example()
        cases = ([{"index": 5, "pose": reference[5]}, {"index": 5, "pose": reference[6]}],
                 [{"index": -1, "pose": reference[-1]}],
                 [{"index": len(reference), "pose": reference[-1]}],
                 [{"index": True, "pose": reference[1]}],
                 [{"index": 0, "pose": reference[0]}], [])
        for goals in cases:
            with self.subTest(goals=goals), patch("manipulation.function_motion.optimize_trajectory") as solve:
                with self.assertRaises(ValueError):
                    transfer_function_motion(reference, reference[0], goals)
                solve.assert_not_called()

    def test_failed_solver_propagates_without_any_retry(self):
        reference, _ = example()
        failure = RuntimeError("fixed function constraints are infeasible")
        with patch("manipulation.function_motion.optimize_trajectory", side_effect=failure) as solve:
            with self.assertRaisesRegex(RuntimeError, "constraints are infeasible") as raised:
                transfer_function_motion(reference, reference[0], [{"index": 8, "pose": reference[8]}])
        self.assertEqual(solve.call_count, 1)
        self.assertIs(raised.exception, failure)

    def test_solver_cannot_redefine_or_miss_an_internal_goal(self):
        reference, _ = example()
        original = reference.copy()
        broken = pairs(reference)
        broken[5, 0, 1] += .004
        with patch("manipulation.function_motion.optimize_trajectory", return_value=broken):
            with self.assertRaisesRegex(ValueError, "node 5"):
                transfer_function_motion(reference, reference[0],
                                         [{"index": 5, "pose": reference[5]},
                                          {"index": 8, "pose": reference[8]}])
        np.testing.assert_array_equal(reference, original)

    def test_solver_success_does_not_hide_intermediate_increment_violations(self):
        reference, _ = example()
        for component in (0, 1):
            broken = pairs(reference)
            broken[3, component, 0] += .05
            with self.subTest(component=component), patch(
                    "manipulation.function_motion.optimize_trajectory", return_value=broken):
                with self.assertRaisesRegex(ValueError, "per-node increment"):
                    transfer_function_motion(reference, reference[0],
                                             [{"index": 8, "pose": reference[8]}])

    def test_numeric_kernel_requires_explicit_nonconflicting_events(self):
        reference, _ = example()
        value = pairs(reference)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            optimize_trajectory(value, value[0], [])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            optimize_trajectory(value, value[0], [{"index": 5, "pose": value[5]},
                                                  {"index": 5, "pose": value[8]}])


if __name__ == "__main__":
    unittest.main()
# END explicit function-frame and internal-event regressions.
