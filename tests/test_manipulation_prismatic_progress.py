"""Verify prismatic goals and tool phases using self-contained CPU fixtures."""

# BEGIN native prismatic progress contracts through the complete consumer.
from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.program import ground_program


def pose(position=(0., 0., 0.), angles=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:3, 3] = position
    return result


def fixture(amplitude=.1, prefix=None):
    """A nonlinear sliding stroke, actual pauses and an independent release.

    The functional event is row 6, opening intention row 7, and last controlled
    observation row 8. Six micrometres of post-event source settling remain
    distinct from the endpoint-retargeted stroke and the released hand motion.
    """
    if prefix is None:
        prefix = [0., .001, .001, .010, .100, .150, .160]
    offsets = np.asarray([*prefix, prefix[-1] + .000004,
                          prefix[-1] + .000006, *([prefix[-1] + .000006] * 3)])
    source_q = .23 + offsets
    frames = np.arange(100, 112)
    source_poses = np.repeat(np.eye(4)[None], len(frames), axis=0)
    source_poses[:, 2, 3] = source_q
    attachment = pose((.025, -.017, .035), (.1, -.08, .12))
    hand = np.repeat(attachment[None], len(frames), axis=0)
    for index in range(9, len(frames)):
        fraction = (index - 8) / 3
        hand[index] = attachment @ pose((.03 * fraction, 0., .07 * fraction),
                                        (.12 * fraction, 0., 0.))
    zero = pose((.25, -.16, .12), (.3, -.2, .4))
    basis = zero[:3, :3].copy()
    initial = zero.copy()
    initial[:3, 3] += .4 * basis[:, 2]
    event = {"event_id": "function", "frame_index": 106,
             "constraint_kind": "joint_coordinate", "coordinate_value": [float(source_q[6])],
             "coordinate_tolerance": 1e-9, "source": "analytic offline function event"}
    program = {
        "skill_kind": "manipulation", "task": {"relation": "articulation", "motion": "prismatic"},
        "demo": {
            "T_reference_carrier": source_poses, "T_carrier_eef": hand,
            "motion_coordinates": source_q, "motion_coordinates_source": "analytic source observations",
            "frame_indices": frames, "timestamps": 2. + np.arange(len(frames)) * .05,
            "actions": np.zeros((len(frames) - 1, 7)),
            "tool_id": "gripper", "tool_frame": "tool_site",
            "gripper_events": [{"event_id": "closed", "frame_index": 100, "command": "close"},
                               {"event_id": "opening", "frame_index": 107, "command": "open"}],
            "release_frame": 108, "functional_events": [event],
        },
    }
    scene = {
        "reference_frame": "world", "carrier_id": "slider", "reference_id": "parent",
        "tool_id": "gripper", "tool_frame": "tool_site",
        "initial_pose": initial, "eef_pose": initial @ attachment,
        "holding": {"verified": True, "T_carrier_eef": attachment,
                    "source": "synthetic synchronized attachment, not real Holding estimation"},
        "carrier_mapping": np.eye(4), "carrier_mapping_source": "same synthetic material frame",
        "motion_model": {"kind": "prismatic", "reference_pose": zero, "basis": basis,
                         "source": "analytic current slider", "coordinate_limits": [[.2, .8]]},
        "functional_bindings": {"function": {"current_coordinates": [.4 + amplitude],
                                             "carrier_id": "slider", "reference_id": "parent",
                                             "source": "independently specified current target"}},
    }
    return program, scene


def coordinates(result):
    return np.asarray(result["motion_checks"]["model"]["coordinates"])[:, 0]


class PrismaticSourceProgressTest(unittest.TestCase):
    def test_nonlinear_100mm_and_50mm_strokes_keep_pause_without_new_reversal(self):
        for amplitude in (.100, .050, .200):
            with self.subTest(amplitude=amplitude):
                program, scene = fixture(amplitude)
                original = np.asarray(program["demo"]["motion_coordinates"]).copy()
                result = ground_program(program, scene)
                q = coordinates(result)
                progress = (original[:7] - original[0]) / (original[6] - original[0])
                expected = .4 + amplitude * progress
                np.testing.assert_allclose(q[:7], expected, atol=1e-13, rtol=0)
                self.assertTrue(np.all(np.diff(q[:7]) >= -1e-15))
                self.assertGreaterEqual(q[:7].min(), .4 - 1e-14)
                self.assertLessEqual(q[:7].max(), .4 + amplitude + 1e-14)
                self.assertEqual(q[1], q[2])  # The source pause is real timing evidence.
                np.testing.assert_array_equal(program["demo"]["motion_coordinates"], original)
                interval = result["motion_checks"]["coordinate_transfer"]["intervals"][0]
                self.assertAlmostEqual(interval["progress_scale"], amplitude / .16)

    def test_source_small_reversal_is_scaled_without_creating_other_reversals(self):
        program, scene = fixture(.05, [0., .001, .0008, .010, .100, .150, .160])
        result = ground_program(program, scene)
        actual = np.diff(coordinates(result)[:7])
        source = np.diff(program["demo"]["motion_coordinates"][:7])
        np.testing.assert_array_equal(actual < 0, source < 0)
        np.testing.assert_allclose(actual, source * (.05 / .16), atol=1e-13, rtol=0)

    def test_final_event_metric_settling_and_release_retreat_do_not_scale(self):
        program, scene = fixture(.05)
        result = ground_program(program, scene)
        q, demo = coordinates(result), program["demo"]
        np.testing.assert_allclose(q[6:] - q[6], [0., .000004, .000006], atol=1e-13, rtol=0)
        self.assertEqual(len(result["eef_poses"]), 12)
        self.assertEqual(len(result["object_poses"]), 9)
        self.assertEqual(result["release_frame"], 108)
        self.assertEqual(result["released_object_state"], "requires_observation")
        np.testing.assert_array_equal(result["frame_indices"], demo["frame_indices"])
        np.testing.assert_array_equal(result["timestamps"], demo["timestamps"])
        self.assertEqual(result["gripper_events"], demo["gripper_events"])
        np.testing.assert_allclose(result["eef_poses"][:9],
                                   result["object_poses"] @ scene["holding"]["T_carrier_eef"], atol=1e-13)
        expected_last = result["eef_poses"][8] @ pose((.03, 0., .07), (.12, 0., 0.))
        np.testing.assert_allclose(result["eef_poses"][-1], expected_last, atol=1e-13)

    def test_multiple_function_events_scale_each_source_interval(self):
        program, scene = fixture(.1)
        middle = deepcopy(program["demo"]["functional_events"][0])
        middle.update(event_id="middle", frame_index=103,
                      coordinate_value=[float(program["demo"]["motion_coordinates"][3])])
        program["demo"]["functional_events"].insert(0, middle)
        scene["functional_bindings"]["middle"] = {
            "current_coordinates": [.42], "carrier_id": "slider", "reference_id": "parent",
            "source": "independent intermediate functional target",
        }
        result = ground_program(program, scene)
        q = coordinates(result)
        source = np.asarray(program["demo"]["motion_coordinates"])
        expected_first = .4 + .02 * (source[:4] - source[0]) / (source[3] - source[0])
        expected_second = .42 + .08 * (source[3:7] - source[3]) / (source[6] - source[3])
        np.testing.assert_allclose(q[:4], expected_first, atol=1e-13, rtol=0)
        np.testing.assert_allclose(q[3:7], expected_second, atol=1e-13, rtol=0)
        self.assertEqual([row["frame_index"] for row in result["function_checks"]], [103, 106])
        self.assertEqual(len(result["motion_checks"]["coordinate_transfer"]["intervals"]), 2)

    def test_zero_net_return_and_pause_keep_internal_source_motion(self):
        for prefix in ([0., .02, .02, -.01, .005, 0., 0.], [0.] * 7):
            with self.subTest(prefix=prefix):
                program, scene = fixture(0., prefix)
                result = ground_program(program, scene)
                expected = .4 + np.asarray(prefix)
                np.testing.assert_allclose(coordinates(result)[:7], expected, atol=1e-13, rtol=0)
                interval = result["motion_checks"]["coordinate_transfer"]["intervals"][0]
                self.assertEqual(interval["method"], "zero_net_source_motion_preserved")

    def test_zero_net_source_cannot_invent_a_different_current_endpoint(self):
        for prefix in ([0., .02, .02, -.01, .005, 0., 0.], [0.] * 7):
            with self.subTest(prefix=prefix):
                program, scene = fixture(.03, prefix)
                with self.assertRaisesRegex(ValueError, "Zero-net prismatic"):
                    ground_program(program, scene)

    def test_axis_sign_and_coordinate_origin_are_gauges_even_after_scaling(self):
        program, scene = fixture(.05)
        expected = ground_program(program, scene)
        reverse_program, reverse_scene = deepcopy(program), deepcopy(scene)
        reverse_program["demo"]["motion_coordinates"] *= -1
        reverse_program["demo"]["functional_events"][0]["coordinate_value"][0] *= -1
        reverse_scene["motion_model"]["basis"] = scene["motion_model"]["basis"] @ np.diag([1., -1., -1.])
        low, high = scene["motion_model"]["coordinate_limits"][0]
        reverse_scene["motion_model"]["coordinate_limits"] = [[-high, -low]]
        reverse_scene["functional_bindings"]["function"]["current_coordinates"][0] *= -1
        actual = ground_program(reverse_program, reverse_scene)
        np.testing.assert_allclose(actual["object_poses"], expected["object_poses"], atol=1e-13)
        np.testing.assert_allclose(actual["eef_poses"], expected["eef_poses"], atol=1e-13)
        shifted = deepcopy(scene)
        offset = .19
        shifted["motion_model"]["reference_pose"][:3, 3] += offset * scene["motion_model"]["basis"][:, 2]
        shifted["motion_model"]["coordinate_limits"] = (np.asarray([[low, high]]) - offset).tolist()
        shifted["functional_bindings"]["function"]["current_coordinates"][0] -= offset
        shifted_result = ground_program(program, shifted)
        np.testing.assert_allclose(shifted_result["object_poses"], expected["object_poses"], atol=1e-13)

    def test_complete_trajectory_limits_still_reject_an_internal_source_excursion(self):
        program, scene = fixture(0., [0., .02, .02, -.01, .005, 0., 0.])
        scene["motion_model"]["coordinate_limits"] = [[.395, .45]]
        # Both endpoint coordinates are .4 and legal; the retained loop is not.
        with self.assertRaisesRegex(ValueError, "exceeds.*limits"):
            ground_program(program, scene)

    def test_equal_amplitude_keeps_the_raw_translated_source_sequence(self):
        # Identity retargeting has an independent analytic contract: translate
        # every source coordinate by the measured starting offset, retaining
        # pauses, genuine reversals and metric settling. An axis-aligned model
        # makes the coordinate origin exact, so array equality tests actual
        # identity preservation instead of tolerating an accidental rescale.
        program, scene = fixture(.16, [0., .001, .001, .0008, .100, .150, .160])
        scene["motion_model"]["reference_pose"] = np.eye(4)
        scene["motion_model"]["basis"] = np.eye(3)
        scene["initial_pose"] = pose((0., 0., .5))
        attachment = scene["holding"]["T_carrier_eef"]
        scene["eef_pose"] = scene["initial_pose"] @ attachment
        demo = program["demo"]
        raw = np.asarray(demo["motion_coordinates"])
        expected_q = raw - raw[0] + .5
        scene["functional_bindings"]["function"]["current_coordinates"] = [expected_q[6]]
        expected_objects = np.repeat(np.eye(4)[None], 9, axis=0)
        expected_objects[:, 2, 3] = expected_q[:9]
        actual = ground_program(program, scene)
        np.testing.assert_array_equal(coordinates(actual), expected_q[:9])
        np.testing.assert_array_equal(actual["object_poses"], expected_objects)
        np.testing.assert_array_equal(actual["eef_poses"][:9], expected_objects @ attachment)
        np.testing.assert_array_equal(actual["frame_indices"], demo["frame_indices"])
        np.testing.assert_array_equal(actual["object_frame_indices"], demo["frame_indices"][:9])
        np.testing.assert_array_equal(actual["timestamps"], demo["timestamps"])
        self.assertEqual(actual["gripper_events"], demo["gripper_events"])
        self.assertEqual(actual["release_frame"], demo["release_frame"])
        self.assertEqual(coordinates(actual)[1], coordinates(actual)[2])
        self.assertLess(coordinates(actual)[3], coordinates(actual)[2])
        interval = actual["motion_checks"]["coordinate_transfer"]["intervals"][0]
        self.assertEqual(interval["method"], "source_metric_increments_preserved")
        self.assertEqual(interval["progress_scale"], 1.)
        expected_last = (expected_objects[-1] @ attachment
                         @ pose((.03, 0., .07), (.12, 0., 0.)))
        np.testing.assert_allclose(actual["eef_poses"][-1], expected_last, atol=1e-13, rtol=0)


if __name__ == "__main__":
    unittest.main()
# END native prismatic source-progress regression contracts.
