"""Independent CPU contracts for explicit one-dimensional function goals.

Analytic mechanisms and supplied Holding are fixtures, not visual recognition,
joint identification, robot execution or a claim of physical grasp stability.
No simulator asset, current GT, model, network or GPU is accessed.
"""

# BEGIN explicit joint-coordinate goals without invented reference frames.
from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.program import ground_object_program, ground_program


def transform(translation=(0., 0., 0.), angles=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:3, 3] = translation
    return result


def fixture():
    """A tilted sliding mechanism with independent source/current q origins.

    The synthetic timeline has the full recording-sized contract: 176 EEF
    observations, 133 controlled object observations, and separate functional,
    opening-command and release boundaries. Every pose is known analytically.
    """
    frames = np.arange(655, 831)
    event_index, release_index = 129, 132
    source_q = .23 + .16 * np.minimum(np.arange(len(frames)) / event_index, 1.)
    source_basis = Rotation.from_euler("xyz", [.2, -.4, .5]).as_matrix()
    source_zero = transform((.06, -.04, .09), (-.2, .3, .1))
    source_poses = np.repeat(source_zero[None], len(frames), axis=0)
    source_poses[:, :3, 3] += source_q[:, None] * source_basis[:, 2]
    source_holding = transform((.025, -.017, .035), (.13, -.08, .11))
    source_hand = np.repeat(source_holding[None], len(frames), axis=0)
    for index in range(release_index + 1, len(frames)):
        phase = (index - release_index) / (len(frames) - 1 - release_index)
        source_hand[index] = source_holding @ transform((0., 0., .08*phase), (.12*phase, 0., 0.))
    event = {
        "event_id": "demonstrated_open_extent", "frame_index": int(frames[event_index]),
        "constraint_kind": "joint_coordinate", "coordinate_value": [float(source_q[event_index])],
        "coordinate_tolerance": 1e-8, "source": "supplied offline semantic function boundary",
    }
    program = {
        "skill_kind": "manipulation", "task": {"relation": "articulation", "motion": "prismatic"},
        "demo": {
            "T_reference_carrier": source_poses, "T_carrier_eef": source_hand,
            "motion_coordinates": source_q,
            "motion_coordinates_source": "offline observed mechanism coordinates; zero is a coordinate convention",
            "frame_indices": frames, "timestamps": 3.25 + np.arange(len(frames))*.05,
            "actions": np.zeros((len(frames)-1, 7)),
            "tool_id": "parallel_gripper", "tool_frame": "grip_site",
            "gripper_events": [
                {"event_id": "already_closed", "frame_index": 655, "command": "close"},
                {"event_id": "release_intention", "frame_index": 785, "command": "open"},
            ],
            "release_frame": int(frames[release_index]), "functional_events": [event],
        },
    }
    mapping = transform((.03, -.02, .01), (.15, -.1, .25))
    zero = transform((.25, -.16, .12), (.3, -.2, .4))
    # The current signed axis is the source carrier-local axis embedded in the
    # new carrier frame and then the output frame; it is not a handle rod axis.
    basis = zero[:3, :3] @ mapping[:3, :3] @ source_zero[:3, :3].T @ source_basis
    initial_q = .47
    initial = zero.copy()
    initial[:3, 3] += initial_q * basis[:, 2]
    holding = mapping @ source_holding
    scene = {
        "reference_frame": "world", "carrier_id": "current_slider", "reference_id": "current_fixture",
        "tool_id": "parallel_gripper", "tool_frame": "grip_site",
        "initial_pose": initial, "eef_pose": initial @ holding,
        "holding": {"verified": True, "T_carrier_eef": holding,
                    "source": "explicitly supplied synchronized Holding fixture"},
        "carrier_mapping": mapping, "carrier_mapping_source": "explicit source-to-current carrier coordinate mapping",
        "motion_model": {
            "kind": "prismatic", "reference_pose": zero, "basis": basis,
            "source": "supplied same-mechanism prior located by current FK and known Holding",
            "coordinate_limits": [[.40, .70]],
        },
        "functional_bindings": {event["event_id"]: {
            "current_coordinates": [initial_q + .16],
            "carrier_id": "current_slider", "reference_id": "current_fixture",
            "source": "requested demonstrated relative opening extent in current signed coordinates",
        }},
    }
    return program, scene


def coordinates_from_output(result, scene):
    """Independent scalar measurement from output positions and a known axis."""
    model = scene["motion_model"]
    poses = np.asarray(result["object_poses"])
    return (poses[:, :3, 3] - model["reference_pose"][:3, 3]) @ model["basis"][:, 2]


class JointCoordinateGoalTest(unittest.TestCase):
    def test_nonzero_source_and_current_origins_preserve_motion_and_explicit_goal(self):
        program, scene = fixture()
        self.assertNotIn("reference_pose", scene)
        event = program["demo"]["functional_events"][0]
        self.assertNotIn("carrier_frame", event)
        self.assertNotIn("reference_frame", event)
        result = ground_program(program, scene)
        expected = .47 + program["demo"]["motion_coordinates"][:133] - .23
        np.testing.assert_allclose(coordinates_from_output(result, scene), expected, atol=1e-10, rtol=0)
        np.testing.assert_allclose(result["object_poses"][0], scene["initial_pose"], atol=1e-10)
        self.assertAlmostEqual(coordinates_from_output(result, scene)[129], .63, places=10)
        for check in result["function_checks"]:
            self.assertEqual(check["constraint_kind"], "joint_coordinate")
            self.assertEqual(check["frame_index"], 784)

    def test_axis_coordinates_goals_and_limits_reverse_together_without_changing_physical_motion(self):
        program, scene = fixture()
        expected = ground_program(program, scene)
        reverse_program, reverse_scene = deepcopy(program), deepcopy(scene)
        reverse_program["demo"]["motion_coordinates"] *= -1
        reverse_program["demo"]["functional_events"][0]["coordinate_value"][0] *= -1
        reverse_scene["motion_model"]["basis"] = scene["motion_model"]["basis"] @ np.diag([1., -1., -1.])
        low, high = scene["motion_model"]["coordinate_limits"][0]
        reverse_scene["motion_model"]["coordinate_limits"] = [[-high, -low]]
        reverse_scene["functional_bindings"]["demonstrated_open_extent"]["current_coordinates"][0] *= -1
        actual = ground_program(reverse_program, reverse_scene)
        np.testing.assert_allclose(actual["object_poses"], expected["object_poses"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual["eef_poses"], expected["eef_poses"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(coordinates_from_output(actual, reverse_scene),
                                   -coordinates_from_output(expected, scene), atol=1e-10, rtol=0)

    def test_current_coordinate_zero_shift_changes_no_physical_pose(self):
        program, scene = fixture()
        expected = ground_program(program, scene)
        shifted = deepcopy(scene)
        offset = .19
        # q' = q-offset and p_zero' = p_zero+axis*offset describe the same joint.
        shifted["motion_model"]["reference_pose"][:3, 3] += scene["motion_model"]["basis"][:, 2]*offset
        shifted["motion_model"]["coordinate_limits"] = (np.asarray(scene["motion_model"]["coordinate_limits"])-offset).tolist()
        shifted["functional_bindings"]["demonstrated_open_extent"]["current_coordinates"][0] -= offset
        actual = ground_program(program, shifted)
        np.testing.assert_allclose(actual["object_poses"], expected["object_poses"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual["eef_poses"], expected["eef_poses"], atol=1e-10, rtol=0)

    def test_source_coordinate_offset_and_column_shape_change_no_output(self):
        program, scene = fixture()
        expected = ground_program(program, scene)
        changed = deepcopy(program)
        changed["demo"]["motion_coordinates"] = (changed["demo"]["motion_coordinates"] + .71)[:, None]
        changed["demo"]["functional_events"][0]["coordinate_value"][0] += .71
        actual = ground_program(changed, scene)
        np.testing.assert_allclose(actual["object_poses"], expected["object_poses"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual["eef_poses"], expected["eef_poses"], atol=1e-10, rtol=0)

    def test_full_timeline_distinguishes_function_opening_and_release(self):
        program, scene = fixture()
        result = ground_program(program, scene)
        self.assertEqual(len(result["eef_poses"]), 176)
        self.assertEqual(len(result["object_poses"]), 133)
        self.assertEqual(result["release_frame"], 787)
        np.testing.assert_array_equal(result["frame_indices"], program["demo"]["frame_indices"])
        np.testing.assert_array_equal(result["timestamps"], program["demo"]["timestamps"])
        np.testing.assert_array_equal(result["object_frame_indices"], np.arange(655, 788))
        self.assertEqual(result["gripper_events"], program["demo"]["gripper_events"])
        self.assertEqual(result["function_checks"][0]["frame_index"], 784)
        self.assertEqual(result["released_object_state"], "requires_observation")

    def test_actual_holding_drives_eef_and_retreat_is_free_after_release(self):
        program, scene = fixture()
        first = ground_program(program, scene)
        np.testing.assert_allclose(first["eef_poses"][:133],
                                   first["object_poses"] @ scene["holding"]["T_carrier_eef"], atol=1e-10)
        np.testing.assert_allclose(first["eef_poses"][0], scene["eef_pose"], atol=1e-10)
        expected_retreat = first["eef_poses"][132] @ transform((0., 0., .08), (.12, 0., 0.))
        np.testing.assert_allclose(first["eef_poses"][-1], expected_retreat, atol=1e-10)
        # A measured different attachment changes the hand but must not move
        # the object's function goal or substitute a new grasp stage.
        changed = deepcopy(scene)
        changed["holding"]["T_carrier_eef"] = scene["holding"]["T_carrier_eef"] @ transform((.01, -.02, .03), (.1, -.05, .07))
        changed["eef_pose"] = scene["initial_pose"] @ changed["holding"]["T_carrier_eef"]
        second = ground_program(program, changed)
        np.testing.assert_allclose(second["object_poses"], first["object_poses"], atol=1e-10)
        np.testing.assert_allclose(second["eef_poses"][:133],
                                   second["object_poses"] @ changed["holding"]["T_carrier_eef"], atol=1e-10)
        self.assertEqual(second["gripper_events"], program["demo"]["gripper_events"])

    def test_object_only_core_does_not_fabricate_holding_or_eef(self):
        program, scene = fixture()
        expected = ground_program(program, scene)
        independent = deepcopy(scene)
        independent.pop("holding")
        independent.pop("eef_pose")
        actual = ground_object_program(program, independent)
        np.testing.assert_allclose(actual["poses"], expected["object_poses"], atol=1e-10)
        self.assertNotIn("eef_poses", actual)
        self.assertEqual(actual["output_type"], "object_trajectory")

    def test_world_frame_change_preserves_held_and_released_motion(self):
        program, scene = fixture()
        expected = ground_program(program, scene)
        H = transform((-.3, .17, .22), (.41, -.32, .28))
        changed = deepcopy(scene)
        for key in ("initial_pose", "eef_pose"):
            changed[key] = H @ scene[key]
        changed["motion_model"]["reference_pose"] = H @ scene["motion_model"]["reference_pose"]
        changed["motion_model"]["basis"] = H[:3, :3] @ scene["motion_model"]["basis"]
        actual = ground_program(program, changed)
        np.testing.assert_allclose(actual["object_poses"], H @ expected["object_poses"], atol=1e-10)
        np.testing.assert_allclose(actual["eef_poses"], H @ expected["eef_poses"], atol=1e-10)

    def test_source_carrier_gauge_requires_the_explicit_mapping_and_preserves_retreat(self):
        program, scene = fixture()
        expected = ground_program(program, scene)
        Q = transform((.04, -.03, .02), (.3, .2, -.1))
        changed_program, changed_scene = deepcopy(program), deepcopy(scene)
        # Q maps old source carrier coordinates to the new source gauge.
        changed_program["demo"]["T_reference_carrier"] = program["demo"]["T_reference_carrier"] @ np.linalg.inv(Q)
        changed_program["demo"]["T_carrier_eef"] = Q @ program["demo"]["T_carrier_eef"]
        changed_scene["carrier_mapping"] = scene["carrier_mapping"] @ np.linalg.inv(Q)
        actual = ground_program(changed_program, changed_scene)
        np.testing.assert_allclose(actual["object_poses"], expected["object_poses"], atol=1e-10)
        np.testing.assert_allclose(actual["eef_poses"], expected["eef_poses"], atol=1e-10)

    def test_missing_coordinate_mapping_or_evidence_never_falls_back_to_full_frames(self):
        changes = [
            lambda p, s: p["demo"].pop("motion_coordinates"),
            lambda p, s: p["demo"].pop("motion_coordinates_source"),
            lambda p, s: p["demo"].update(motion_coordinates_source=""),
            lambda p, s: p["demo"]["functional_events"][0].pop("source"),
            lambda p, s: s.pop("carrier_mapping"),
            lambda p, s: s.pop("carrier_mapping_source"),
            lambda p, s: s.update(carrier_mapping_source=""),
            lambda p, s: s["functional_bindings"]["demonstrated_open_extent"].pop("source"),
            lambda p, s: s["motion_model"].pop("source"),
            lambda p, s: s.pop("motion_model"),
        ]
        for index, change in enumerate(changes):
            program, scene = fixture()
            change(program, scene)
            with self.subTest(case=index), self.assertRaises(ValueError):
                ground_program(program, scene)

    def test_malformed_coordinates_or_source_event_mismatch_are_rejected(self):
        changes = [
            lambda p, s: p["demo"].update(motion_coordinates=np.zeros(175)),
            lambda p, s: p["demo"].update(motion_coordinates=np.zeros((176, 2))),
            lambda p, s: p["demo"]["motion_coordinates"].__setitem__(20, np.nan),
            lambda p, s: p["demo"]["functional_events"][0].update(coordinate_value=[.9]),
            lambda p, s: p["demo"]["functional_events"][0].update(coordinate_tolerance=0.),
            lambda p, s: p["demo"]["functional_events"][0].update(coordinate_tolerance=np.nan),
            lambda p, s: s["functional_bindings"]["demonstrated_open_extent"].update(current_coordinates=[np.inf]),
            lambda p, s: s["functional_bindings"]["demonstrated_open_extent"].update(current_coordinates=[.6, .7]),
        ]
        for index, change in enumerate(changes):
            program, scene = fixture()
            change(program, scene)
            with self.subTest(case=index), self.assertRaises(ValueError):
                ground_program(program, scene)

    def test_out_of_limit_goal_or_initial_pose_and_off_axis_start_are_rejected(self):
        program, scene = fixture()
        scene["functional_bindings"]["demonstrated_open_extent"]["current_coordinates"] = [.8]
        with self.assertRaises(ValueError):
            ground_program(program, scene)
        program, scene = fixture()
        scene["motion_model"]["coordinate_limits"] = [[.50, .70]]
        with self.assertRaises(ValueError):
            ground_program(program, scene)
        program, scene = fixture()
        scene["initial_pose"][:3, 3] += .01 * scene["motion_model"]["basis"][:, 0]
        scene["eef_pose"] = scene["initial_pose"] @ scene["holding"]["T_carrier_eef"]
        with self.assertRaises(ValueError):
            ground_program(program, scene)

    def test_unverified_or_unsynchronized_holding_cannot_start_complete_program(self):
        for change in (
            lambda s: s["holding"].update(verified=False),
            lambda s: s["holding"].pop("source"),
            lambda s: s["eef_pose"].__setitem__((0, 3), s["eef_pose"][0, 3]+.01),
        ):
            program, scene = fixture()
            change(scene)
            with self.subTest(change=change), self.assertRaises(ValueError):
                ground_program(program, scene)

    def test_joint_function_event_after_release_or_mixed_with_frame_event_is_rejected(self):
        program, scene = fixture()
        program["demo"]["functional_events"][0]["frame_index"] = 789
        with self.assertRaises(ValueError):
            ground_program(program, scene)
        program, scene = fixture()
        other = {"event_id": "full_frame", "frame_index": 784, "carrier_frame": np.eye(4),
                 "reference_frame": np.eye(4), "source": "different constraint kind",
                 "position_tolerance_m": 1e-6, "rotation_tolerance_rad": 1e-6}
        program["demo"]["functional_events"].append(other)
        with self.assertRaises(ValueError):
            ground_program(program, scene)

    def test_model_coordinate_limits_are_optional_but_no_limit_is_invented(self):
        program, scene = fixture()
        scene["motion_model"].pop("coordinate_limits")
        result = ground_program(program, scene)
        self.assertEqual(result["motion_checks"]["limit_status"], "not_supplied")
        self.assertAlmostEqual(coordinates_from_output(result, scene)[129], .63, places=10)

    def test_revolute_explicit_winding_is_not_replaced_by_shorter_modulo_goal(self):
        program, scene = fixture()
        program["task"]["motion"] = "revolute"
        demo = program["demo"]
        phase = np.minimum(np.arange(176) / 129, 1.)
        source_q = .13 + (2*np.pi + .25) * phase
        demo["motion_coordinates"] = source_q
        demo["functional_events"][0]["coordinate_value"] = [float(source_q[129])]
        zero = transform((.3, 0., .1))
        scene["motion_model"] = {"kind": "revolute", "reference_pose": zero,
                                 "basis": np.eye(3), "axis_point": np.array([0., 0., .1]),
                                 "source": "supplied signed hinge fixture", "coordinate_limits": [[0., 7.]]}
        start_q = .2
        # BEGIN actual multi-turn branch is explicit fixture evidence.
        scene["motion_model"].update(initial_coordinates=[start_q],
            initial_coordinates_source="measured current hinge branch, independent of its periodic pose")
        # END a principal-looking pose alone does not identify the current turns.
        rotation = Rotation.from_rotvec([0., 0., start_q]).as_matrix()
        initial = zero.copy()
        initial[:3, :3] = rotation
        initial[:3, 3] = np.array([0., 0., .1]) + rotation @ np.array([.3, 0., 0.])
        scene["initial_pose"] = initial
        scene["eef_pose"] = initial @ scene["holding"]["T_carrier_eef"]
        goal = start_q + 2*np.pi + .25
        scene["functional_bindings"]["demonstrated_open_extent"]["current_coordinates"] = [goal]
        # Source owner motion and its scalar chart describe the same physical
        # hinge; the coordinates deliberately retain a complete revolution.
        demo["T_reference_carrier"] = np.repeat(zero[None], 176, axis=0)
        rotations = Rotation.from_rotvec(source_q[:, None]*np.array([0., 0., 1.])).as_matrix()
        demo["T_reference_carrier"][:, :3, :3] = rotations
        demo["T_reference_carrier"][:, :3, 3] = np.array([0., 0., .1]) + rotations @ np.array([.3, 0., 0.])
        result = ground_program(program, scene)
        actual_q = np.asarray(result["motion_checks"]["model"]["coordinates"])[:, 0]
        self.assertAlmostEqual(actual_q[0], start_q, places=10)
        self.assertAlmostEqual(actual_q[129], goal, places=10)
        self.assertGreater(actual_q[129], 2*np.pi)
        np.testing.assert_allclose(actual_q, start_q + source_q[:133] - source_q[0], atol=1e-10)
# END explicit joint-coordinate goals without invented reference frames.
