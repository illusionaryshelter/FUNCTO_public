"""CPU grasp geometry contracts; not robot or perception success tests.

Synthetic fixtures deliberately keep the carrier static, move functional
regions, and change the actually established grasp independently of the plan.
"""

# BEGIN independent-grasp and release-boundary regressions.
from copy import deepcopy
import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.grasp import (attached_eef_reference, ground_grasp,
                                retreat_from_release, validate_timeline)
from manipulation.pipeline import serializable


def pose(translation=(0., 0., 0.), angles=(0., 0., 0.)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    value[:3, 3] = translation
    return value


def example():
    count = 11
    relative = np.repeat(np.eye(4)[None], count, axis=0)
    relative[:, 2, 3] = np.linspace(.20, .03, count)
    relative[:, :3, :3] = Rotation.from_euler("y", np.linspace(.1, .4, count)).as_matrix()
    frames = np.arange(101, 101 + 2 * count, 2)
    demo = {
        "T_carrier_eef": relative,
        "frame_indices": frames,
        "timestamps": 6.4 + np.arange(count) * .05,
        "protected_start_frame": int(frames[7]),
        "gripper_events": [
            {"event_id": "opening", "frame_index": int(frames[0]), "command": "open"},
            {"event_id": "closing", "frame_index": int(frames[8]), "command": "close"},
        ],
        "tool_id": "parallel_gripper",
        "tool_frame": "tip_center",
    }
    scene = {
        "carrier_pose": pose((.3, -.2, .4), (.1, -.2, .3)),
        "region_transform": pose((.03, .02, 0.), (0., .1, 0.)),
        "tool_id": demo["tool_id"],
        "tool_frame": demo["tool_frame"],
    }
    scene["eef_pose"] = scene["carrier_pose"] @ scene["region_transform"] @ relative[0]
    scene["eef_pose"][:3, 3] += [.10, -.03, .02]
    return demo, scene


class GraspGeometryTest(unittest.TestCase):
    def test_static_carrier_keeps_complete_bound_source_as_an_initializer(self):
        demo, scene = example()
        before_demo, before_scene = deepcopy(demo), deepcopy(scene)
        result = ground_grasp(demo, scene)
        bound = scene["carrier_pose"] @ scene["region_transform"] @ demo["T_carrier_eef"]
        self.assertGreater(np.linalg.norm(result["eef_poses"][0, :3, 3]
                                         - result["eef_poses"][-1, :3, 3]), .1)
        np.testing.assert_allclose(result["eef_poses"], bound, atol=1e-12)
        np.testing.assert_allclose(result["eef_poses"][7:], bound[7:], atol=1e-14, rtol=0)
        other_start = deepcopy(scene)
        other_start["eef_pose"][:3, 3] += [.21, .12, -.07]
        np.testing.assert_array_equal(ground_grasp(demo, other_start)["eef_poses"][7:],
                                      result["eef_poses"][7:])
        self.assertEqual(len(result["eef_poses"]), len(demo["T_carrier_eef"]))
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["holding_status"], "requires_verification")
        self.assertFalse(result["effect_requirement"]["original_support_separation_required"])
        self.assertNotIn("actual_T_carrier_eef", result)
        np.testing.assert_array_equal(demo["T_carrier_eef"], before_demo["T_carrier_eef"])
        np.testing.assert_array_equal(scene["eef_pose"], before_scene["eef_pose"])
        self.assertEqual(json.loads(json.dumps(serializable(result)))["status"], "candidate")

    def test_region_changes_update_reference_and_goal_without_changing_actual_input(self):
        demo, scene = example()
        first = ground_grasp(demo, scene)
        scene["region_transform"][:3, 3] += [.07, -.01, .02]
        second = ground_grasp(demo, scene)
        self.assertFalse(np.allclose(first["eef_poses"][0], second["eef_poses"][0]))
        self.assertGreater(np.linalg.norm(first["eef_poses"][-1, :3, 3]
                                         - second["eef_poses"][-1, :3, 3]), .07)
        self.assertFalse(np.allclose(first["candidate_T_carrier_eef"],
                                    second["candidate_T_carrier_eef"]))

    # BEGIN contact evidence never creates an independent approach obligation.
    def test_contact_evidence_at_first_row_never_requires_a_source_pose_connection(self):
        demo, scene = example()
        demo["protected_start_frame"] = int(demo["frame_indices"][0])
        result = ground_grasp(demo, scene)
        self.assertEqual(result["status"], "candidate")
        expected = scene["carrier_pose"] @ scene["region_transform"] @ demo["T_carrier_eef"]
        np.testing.assert_allclose(result["eef_poses"], expected, atol=1e-14, rtol=0)
        self.assertNotIn("connection_required", result)
        self.assertNotIn("entry_pose", result)
        self.assertNotIn("verification_barrier", result)
        np.testing.assert_allclose(result["goal_pose"], expected[-1])
    # END contact evidence never creates an independent approach obligation.

    def test_original_frame_ids_timestamps_and_events_remain_separate(self):
        demo, scene = example()
        demo["gripper_qpos"] = np.tile([.024, -.024], (11, 1))
        demo["source_actions"] = np.ones((10, 7))
        result = ground_grasp(demo, scene)
        np.testing.assert_array_equal(result["frame_indices"], demo["frame_indices"])
        np.testing.assert_array_equal(result["timestamps"], demo["timestamps"])
        self.assertEqual(result["gripper_events"], demo["gripper_events"])
        self.assertEqual(result["effect_requirement"]["frame_index"], 121)
        self.assertNotIn("gripper_qpos", result)
        result["gripper_events"][0]["command"] = "hold"
        self.assertEqual(demo["gripper_events"][0]["command"], "open")

    def test_end_boundary_does_not_own_a_second_gripper_command(self):
        demo, scene = example()
        demo["gripper_events"][-1]["frame_index"] = int(demo["frame_indices"][-1])
        with self.assertRaisesRegex(ValueError, "\\[first, last\\)"):
            ground_grasp(demo, scene)

    def test_closing_cannot_be_hidden_inside_a_free_prefix(self):
        demo, scene = example()
        demo["protected_start_frame"] = int(demo["frame_indices"][9])
        with self.assertRaisesRegex(ValueError, "closing intention"):
            ground_grasp(demo, scene)

    def test_tool_identity_and_frame_are_not_human_robot_retargeting(self):
        for key in ("tool_id", "tool_frame"):
            demo, scene = example()
            scene[key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "same " + key):
                ground_grasp(demo, scene)

    def test_grasp_is_equivariant_to_world_and_carrier_frame_changes(self):
        demo, scene = example()
        first = ground_grasp(demo, scene)
        world = pose((-.2, .4, .1), (.3, .7, -.5))
        changed = deepcopy(scene)
        changed["carrier_pose"] = world @ scene["carrier_pose"]
        changed["eef_pose"] = world @ scene["eef_pose"]
        np.testing.assert_allclose(ground_grasp(demo, changed)["eef_poses"],
                                   world @ first["eef_poses"], atol=1e-11)
        demo_gauge = pose((.1, -.2, .3), (-.1, .4, -.3))
        current_gauge = pose((-.2, .1, .2), (.1, .3, -.4))
        changed_demo, changed = deepcopy(demo), deepcopy(scene)
        changed_demo["T_carrier_eef"] = demo_gauge @ demo["T_carrier_eef"]
        changed["carrier_pose"] = scene["carrier_pose"] @ np.linalg.inv(current_gauge)
        changed["region_transform"] = (current_gauge @ scene["region_transform"]
                                        @ np.linalg.inv(demo_gauge))
        np.testing.assert_allclose(ground_grasp(changed_demo, changed)["eef_poses"],
                                   first["eef_poses"], atol=1e-11)

    def test_actual_attachment_controls_holding_and_release_is_independent(self):
        carrier = np.repeat(pose((.2, .1, .3))[None], 5, axis=0)
        carrier[:, 0, 3] += np.linspace(0., .2, len(carrier))
        actual = pose((.03, -.01, .04), (0., .4, .1))
        eef = attached_eef_reference(carrier, actual)
        np.testing.assert_allclose(np.linalg.inv(carrier) @ eef,
                                   np.repeat(actual[None], len(carrier), axis=0), atol=1e-12)
        changed_actual = actual.copy()
        changed_actual[0, 3] += .02
        self.assertFalse(np.allclose(eef, attached_eef_reference(carrier, changed_actual)))
        retreat = np.repeat(pose((.5, .3, .4), (.2, -.1, .3))[None], 4, axis=0)
        retreat[:, 2, 3] += np.linspace(0., .12, len(retreat))
        result = retreat_from_release(retreat, eef[-1])
        np.testing.assert_array_equal(result[0], eef[-1])
        np.testing.assert_allclose(np.linalg.inv(result[0]) @ result,
                                   np.linalg.inv(retreat[0]) @ retreat, atol=1e-12)
        self.assertGreater(np.linalg.norm(result[-1, :3, 3] - result[0, :3, 3]), .119)
        world = pose((.1, -.3, .4), (.4, .1, -.2))
        np.testing.assert_allclose(retreat_from_release(world @ retreat, world @ eef[-1]),
                                   world @ result, atol=1e-12)

    def test_timeline_rejects_ambiguous_ids_and_times(self):
        for frames, times in (([1., 2.], [0., 1.]), ([True, False], [0., 1.]),
                              ([1, 1], [0., 1.]), ([1, 2], [0., 0.]),
                              ([1, 2], [0., np.nan]), ([1, 2], [0.]),
                              ([1, 2], [False, True])):
            with self.subTest(frames=frames, times=times), self.assertRaises(ValueError):
                validate_timeline(frames, times)
        frames, times = validate_timeline([803], [7.4])
        self.assertEqual(frames.tolist(), [803])
        self.assertEqual(times.tolist(), [7.4])

    def test_grasp_rejects_wrong_event_ids_phases_and_invalid_geometry(self):
        mutations = (
            lambda d, s: d["gripper_events"].clear(),
            lambda d, s: d["gripper_events"][-1].update(command="open"),
            lambda d, s: d["gripper_events"][-1].update(frame_index=112),
            lambda d, s: d["gripper_events"][-1].update(event_id="opening"),
            lambda d, s: d.update(protected_start_frame=7),
            lambda d, s: d.update(protected_start_frame=True),
            lambda d, s: s["region_transform"].__setitem__((slice(0, 3), 0), [0., 0., 0.]),
        )
        for mutation in mutations:
            demo, scene = example()
            mutation(demo, scene)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                ground_grasp(demo, scene)


if __name__ == "__main__":
    unittest.main()
# END independent-grasp and release-boundary regressions.
