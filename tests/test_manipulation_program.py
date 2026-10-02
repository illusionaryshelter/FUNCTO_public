"""Geometric counterexamples for the region/holding/function event consumer.

Correspondences and observations are supplied independently in these CPU
fixtures; the tests do not measure SAM masks or physical robot execution.
"""

# BEGIN function event and actual-holding integration tests.
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.program import ground_object_program, ground_program


def pose(x=0., y=0., z=0.):
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


def fixture():
    objects = np.repeat(np.eye(4)[None], 11, axis=0)
    objects[:, 0, 3] = np.r_[np.linspace(0, .05, 6), np.repeat(.05, 5)]
    hand = np.repeat(pose(z=.02)[None], 11, axis=0)
    hand[8:, 2, 3] += [.01, .02, .03]
    event = {"event_id": "effect", "frame_index": 105, "carrier_frame": pose(),
             "reference_frame": pose(), "source": "independent supplied functional event",
             "position_tolerance_m": 1e-7, "rotation_tolerance_rad": 1e-7}
    program = {"skill_kind": "manipulation", "task": {"relation": "transfer", "motion": "free"},
               "demo": {"T_reference_carrier": objects, "T_carrier_eef": hand,
                        "frame_indices": np.arange(100, 111), "timestamps": np.arange(11) * .05 + 2.,
                        "tool_id": "parallel_gripper", "tool_frame": "eef_site",
                        "gripper_events": [{"event_id": "open", "frame_index": 106, "command": "open"}],
                        "release_frame": 107, "functional_events": [event]}}
    scene = {"reference_frame": "current_reference", "reference_pose": pose(),
             "carrier_id": "new_carrier", "reference_id": "new_target",
             "tool_id": "parallel_gripper", "tool_frame": "eef_site", "initial_pose": pose(),
             "eef_pose": pose(z=.03), "holding": {"verified": True, "source": "synchronized observed hold",
                                                     "T_carrier_eef": pose(z=.03)},
             "functional_bindings": {"effect": {"carrier_id": "new_carrier", "reference_id": "new_target",
                                                  "carrier_frame": pose(), "reference_frame": pose(),
                                                  "source": "supplied current part frames"}}}
    return program, scene


# BEGIN unordered support fixtures have no complete target functional frame.
def on_fixture():
    program, scene = fixture()
    program["task"]["relation"] = "on"
    program["demo"]["functional_events"][0].pop("reference_frame")
    scene.pop("reference_pose")
    body = np.asarray([[x, y, z] for x in (-.01, .01)
                       for y in (-.01, .01) for z in (0., .02)])
    scene["body_points"] = body
    scene["support"] = {
        "origin": [0., 0., 0.], "normal": [0., 0., 1.],
        "basis": [[1., 0.], [0., 1.], [0., 0.]],
        "boundary": [[-.3, -.3], [.3, -.3], [.3, .3], [-.3, .3]],
        "source": "supplied finite support; its tangent axes are a coordinate gauge",
    }
    scene["functional_bindings"]["effect"] = {
        "carrier_id": scene["carrier_id"], "reference_id": scene["reference_id"],
        "carrier_mapping": pose(),
        "source": "explicit same carrier-local frame; independently supplied support",
        "on_geometry": {"carrier_region_points": body[body[:, 2] == 0].copy(),
                        "carrier_outward_normal": [0., 0., -1.],
                        "gap_tolerance": 1e-6, "normal_tolerance": 1e-6},
    }
    return program, scene
# END unordered support fixture.


class ProgramTest(unittest.TestCase):
    def test_source_corridor_deviation_is_reported_without_rejecting_the_bound_goal(self):
        from itertools import product
        from manipulation.goals import InteriorRegion
        program, scene = fixture()
        program["task"]["relation"] = "inside"
        program["demo"]["T_reference_carrier"][2, 1, 3] = .06
        scene["body_points"] = np.array(list(product([-.002, .002], repeat=3)))
        region = InteriorRegion.from_vertices(
            list(product([-.1, .1], [-.02, .02], [-.02, .02])), "observed free entry corridor")
        scene.update(entry_region=region.to_dict(), entry_start_index=1)
        # BEGIN reference deviation does not negate the independent corridor requirement.
        result = ground_program(program, scene)
        self.assertFalse(result["motion_checks"]["entry_corridor"]["reference_satisfies_corridor"])
        self.assertEqual(result["function_checks"][0]["status"], "bound_goal")
        # END reference deviation does not negate the independent corridor requirement.
        program["demo"]["T_reference_carrier"][2, 1, 3] = 0
        result = ground_program(program, scene)
        self.assertEqual(result["motion_checks"]["entry_corridor"]["checked_nodes"], 7)

    def test_two_sided_internal_event_and_release(self):
        program, scene = fixture()
        # Change both functional frames; carrier change must not cancel when
        # connecting the unchanged actual initial object pose.
        scene["functional_bindings"]["effect"]["carrier_frame"] = pose(x=.01)
        scene["functional_bindings"]["effect"]["reference_frame"] = pose(x=.04)
        result = ground_program(program, scene)
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(len(result["object_poses"]), 8)
        self.assertEqual(len(result["eef_poses"]), 11)
        self.assertAlmostEqual(result["object_poses"][5, 0, 3], .08)
        np.testing.assert_allclose(result["object_poses"][5:], np.repeat(pose(x=.08)[None], 3, axis=0), atol=1e-9)
        self.assertAlmostEqual(result["eef_poses"][7, 2, 3], .03)
        self.assertAlmostEqual(result["eef_poses"][-1, 2, 3], .06)
        self.assertEqual(result["released_object_state"], "requires_observation")
        self.assertEqual(result["gripper_events"][0]["frame_index"], 106)
        np.testing.assert_array_equal(result["timestamps"], program["demo"]["timestamps"])

    def test_actual_grasp_changes_hand_not_function_effect(self):
        program, scene = fixture()
        first = ground_program(program, scene)
        scene["holding"]["T_carrier_eef"] = pose(z=.06)
        scene["eef_pose"] = pose(z=.06)
        second = ground_program(program, scene)
        np.testing.assert_allclose(first["object_poses"], second["object_poses"])
        np.testing.assert_allclose(second["eef_poses"][:, 2, 3] - first["eef_poses"][:, 2, 3], .03)

    def test_candidate_or_unsynchronized_holding_is_rejected(self):
        program, scene = fixture()
        scene["holding"]["verified"] = False
        with self.assertRaisesRegex(ValueError, "actual holding"):
            ground_program(program, scene)
        scene["holding"]["verified"] = True
        scene["eef_pose"] = pose(x=1)
        with self.assertRaisesRegex(ValueError, "synchronized"):
            ground_program(program, scene)

    def test_function_event_cannot_be_replaced_by_end_or_release(self):
        program, scene = fixture()
        program["demo"]["functional_events"] = []
        with self.assertRaisesRegex(ValueError, "explicit functional_events"):
            ground_program(program, scene)
        program, scene = fixture()
        program["demo"]["functional_events"][0]["frame_index"] = 110
        with self.assertRaisesRegex(ValueError, "controlled object motion"):
            ground_program(program, scene)

    def test_wrong_owner_and_mechanism_are_not_silently_fixed(self):
        program, scene = fixture()
        scene["functional_bindings"]["effect"]["carrier_id"] = "another_similar_cup"
        with self.assertRaisesRegex(ValueError, "owners"):
            ground_program(program, scene)
        program, scene = fixture()
        program["task"] = {"relation": "articulation", "motion": "prismatic"}
        basis = Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix()
        scene["motion_model"] = {"kind": "prismatic", "reference_pose": pose(), "basis": basis,
                                  "source": "observed axis", "coordinate_limits": [[0, .04]]}
        with self.assertRaisesRegex(ValueError, "mechanism limits"):
            ground_program(program, scene)
        scene["motion_model"]["coordinate_limits"] = [[0, .2]]
        valid = ground_program(program, scene)
        np.testing.assert_allclose(valid["object_poses"][5], pose(x=.05), atol=1e-9)
        scene["functional_bindings"]["effect"]["reference_frame"] = pose(y=.1)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            ground_program(program, scene)

    def test_multiple_function_events_remain_internal_constraints(self):
        program, scene = fixture()
        other = deepcopy(program["demo"]["functional_events"][0])
        other.update(event_id="earlier", frame_index=103)
        program["demo"]["functional_events"].insert(0, other)
        scene["functional_bindings"]["earlier"] = deepcopy(scene["functional_bindings"]["effect"])
        scene["functional_bindings"]["earlier"]["reference_frame"] = pose(x=.01)
        result = ground_program(program, scene)
        self.assertAlmostEqual(result["object_poses"][3, 0, 3], .04)
        self.assertAlmostEqual(result["object_poses"][5, 0, 3], .05)
        self.assertEqual(len(result["function_checks"]), 2)

    def test_nonidentity_reference_owner_pose_preserves_world_equivariance(self):
        program, scene = fixture()
        first = ground_program(program, scene)
        world = pose(x=.4, y=-.3, z=.2)
        world[:3, :3] = Rotation.from_euler("xyz", [.3, -.4, .6]).as_matrix()
        changed = deepcopy(scene)
        changed["reference_frame"] = "camera_with_rotated_reference_owner"
        changed["reference_pose"] = world @ scene["reference_pose"]
        changed["initial_pose"] = world @ scene["initial_pose"]
        changed["eef_pose"] = world @ scene["eef_pose"]
        second = ground_program(program, changed)
        np.testing.assert_allclose(second["object_poses"], world @ first["object_poses"], atol=1e-9)
        np.testing.assert_allclose(second["eef_poses"], world @ first["eef_poses"], atol=1e-9)
        for old_check, new_check in zip(first["function_checks"], second["function_checks"]):
            np.testing.assert_allclose(new_check["goal_pose"], world @ old_check["goal_pose"], atol=1e-9)
            self.assertLess(new_check["position_error_m"], 1e-9)

    def test_soft_reference_projection_cannot_project_a_hard_function_goal(self):
        program, scene = fixture()
        program["task"] = {"relation": "articulation", "motion": "prismatic"}
        # An off-axis *reference* sample may be projected. Both actual start
        # and function target remain required poses of the current mechanism.
        program["demo"]["T_reference_carrier"][2, 1, 3] = .02
        scene["motion_model"] = {
            "kind": "prismatic", "reference_pose": pose(),
            "basis": Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix(),
            "source": "independent current mechanism", "coordinate_limits": [[0, .2]],
        }
        first = ground_program(program, scene)
        np.testing.assert_allclose(first["object_poses"][:, 1, 3], 0., atol=1e-9)
        self.assertGreater(first["motion_checks"]["reference_projection"]["max_position_m"], .019)
        world = pose(x=.3, y=.2, z=-.1)
        world[:3, :3] = Rotation.from_euler("xyz", [.1, .5, -.3]).as_matrix()
        changed = deepcopy(scene)
        changed["reference_pose"] = world
        changed["initial_pose"] = world @ scene["initial_pose"]
        changed["eef_pose"] = world @ scene["eef_pose"]
        changed["motion_model"]["reference_pose"] = world @ scene["motion_model"]["reference_pose"]
        changed["motion_model"]["basis"] = world[:3, :3] @ scene["motion_model"]["basis"]
        second = ground_program(program, changed)
        np.testing.assert_allclose(second["object_poses"], world @ first["object_poses"], atol=1e-9)
        # The same deviation at the required event may not be silently
        # repaired by the reference projection that was allowed above.
        program["demo"]["T_reference_carrier"][5, 1, 3] = .02
        with self.assertRaisesRegex(ValueError, "incompatible"):
            ground_program(program, scene)

    def test_last_internal_event_at_release_remains_exact_and_has_a_retreat(self):
        program, scene = fixture()
        final = deepcopy(program["demo"]["functional_events"][0])
        final.update(event_id="at_release", frame_index=107)
        program["demo"]["functional_events"].append(final)
        scene["functional_bindings"]["at_release"] = deepcopy(scene["functional_bindings"]["effect"])
        scene["functional_bindings"]["at_release"]["reference_frame"] = pose(x=.02)
        result = ground_program(program, scene)
        self.assertEqual(len(result["object_poses"]), 8)
        self.assertAlmostEqual(result["object_poses"][5, 0, 3], .05)
        self.assertAlmostEqual(result["object_poses"][7, 0, 3], .07)
        np.testing.assert_allclose(result["eef_poses"][7],
                                   result["object_poses"][7] @ scene["holding"]["T_carrier_eef"], atol=1e-9)
        self.assertGreater(np.linalg.norm(result["eef_poses"][-1, :3, 3]
                                         - result["eef_poses"][7, :3, 3]), .029)

    def test_actual_rotated_grasp_controls_held_phase_and_release_local_increments(self):
        program, scene = fixture()
        actual = pose(x=.01, y=-.02, z=.04)
        actual[:3, :3] = Rotation.from_euler("xyz", [.2, -.1, .7]).as_matrix()
        scene["holding"]["T_carrier_eef"] = actual
        scene["eef_pose"] = scene["initial_pose"] @ actual
        result = ground_program(program, scene)
        np.testing.assert_allclose(result["eef_poses"][:8], result["object_poses"] @ actual, atol=1e-10)
        demo_hand = program["demo"]["T_reference_carrier"] @ program["demo"]["T_carrier_eef"]
        np.testing.assert_allclose(np.linalg.inv(result["eef_poses"][7]) @ result["eef_poses"][7:],
                                   np.linalg.inv(demo_hand[7]) @ demo_hand[7:], atol=1e-10)

    def test_on_consumes_the_declared_contact_face_in_addition_to_whole_body(self):
        program, scene = on_fixture()
        with patch("manipulation.program.ground_relation", side_effect=AssertionError("on has no full target frame")):
            result = ground_program(program, scene)
        self.assertTrue(result["function_checks"][0]["valid"])
        self.assertIn("body_checks", result["function_checks"][0])
        # The carrier can rest on its lower face while the requested (wrong)
        # top region is 2 cm above the support. Whole-body checks alone pass.
        scene["functional_bindings"]["effect"]["on_geometry"]["carrier_region_points"][:, 2] = .02
        with self.assertRaisesRegex(ValueError, "functional contact face"):
            ground_program(program, scene)

    def test_invalid_inside_margin_cannot_bypass_containment_checks(self):
        program, scene = fixture()
        program["task"]["relation"] = "inside"
        scene["body_points"] = np.asarray([[x, y, z] for x in (-.01, .01)
                                            for y in (-.01, .01) for z in (-.01, .01)])
        scene["interior"] = {"vertices": [[x, y, z] for x in (-.2, .2)
                                           for y in (-.2, .2) for z in (-.2, .2)],
                             "source": "supplied free interior"}
        for margin in (np.nan, np.inf, -.01):
            scene["margin"] = margin
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                ground_program(program, scene)

    def test_object_only_program_rejects_unsupported_relation(self):
        program, scene = fixture()
        program["task"]["relation"] = "unknown_relation"
        with self.assertRaisesRegex(ValueError, "relation"):
            ground_object_program(program, scene)

    def test_simultaneous_regions_must_agree_without_averaging(self):
        program, scene = fixture()
        second = deepcopy(program["demo"]["functional_events"][0])
        second["event_id"] = "other_part"
        program["demo"]["functional_events"].append(second)
        scene["functional_bindings"]["other_part"] = deepcopy(scene["functional_bindings"]["effect"])
        binding = scene["functional_bindings"]["other_part"]
        binding["carrier_frame"] = pose(x=.03)
        binding["reference_frame"] = pose(x=.03)
        result = ground_program(program, scene)
        self.assertEqual(len(result["function_checks"]), 2)
        np.testing.assert_allclose(result["object_poses"][5], pose(x=.05), atol=1e-9)
        binding["reference_frame"] = pose(x=.07)
        with self.assertRaisesRegex(ValueError, "consistent pose candidates"):
            ground_program(program, scene)

    def test_cached_region_binding_checks_both_owners_and_preserves_owner_local_coordinates(self):
        # BEGIN current-owner binder contract; visual accuracy is tested outside
        # this isolated coordinate/owner regression. No current part is supplied.
        from PIL import Image
        from manipulation.perception import bind_current_scene

        program, scene = fixture()
        program["demo"].update(carrier_id="demo_carrier", reference_id="demo_target")
        event = program["demo"]["functional_events"][0]
        event["carrier_frame"] = pose(x=.02)
        event["reference_frame"] = pose(y=.01)
        carrier_map, reference_map = pose(y=.03), pose(z=.04)
        templates = {"effect": {
            "carrier": {"owner_id": "demo_carrier", "region_id": "source_contact"},
            "reference": {"owner_id": "demo_target", "region_id": "source_target"},
        }}
        owners = {role: {"owner_id": scene[role + "_id"], "image": Image.new("RGB", (4, 4)),
                         "owner_mask": np.ones((4, 4), bool), "depth": np.ones((4, 4)),
                         "K": np.eye(3), "frame_convention": "binding_camera", "T_output_camera": pose()}
                  for role in ("carrier", "reference")}
        sources = [{"source_frame": event[role + "_frame"], "geometry_kind": "frame"}
                   for role in ("carrier", "reference")]
        matched = [{**source, "target_frame": transform @ source["source_frame"], "transform": transform}
                   for source, transform in zip(sources, (carrier_map, reference_map))]
        with patch("manipulation.perception.source_function_geometry", side_effect=sources), \
                patch("manipulation.perception.match_functional_region", side_effect=matched) as match:
            bound = bind_current_scene(program, scene, templates, owners)
        self.assertEqual(match.call_count, 2)
        numeric = bound["functional_bindings"]["effect"]
        np.testing.assert_allclose(numeric["carrier_frame"], carrier_map @ event["carrier_frame"])
        np.testing.assert_allclose(numeric["reference_frame"], reference_map @ event["reference_frame"])
        self.assertFalse(np.allclose(numeric["carrier_frame"], scene["functional_bindings"]["effect"]["carrier_frame"]))
        for side, expected_message in ((0, "Demonstration region owner"), (1, "Current owner")):
            wrong_templates, wrong_owners = deepcopy(templates), deepcopy(owners)
            wrong = wrong_templates["effect"]["carrier"] if side == 0 else wrong_owners["carrier"]
            wrong["owner_id"] = "wrong_similar_instance"
            with self.subTest(side=side), patch("manipulation.perception.match_functional_region") as match:
                with self.assertRaisesRegex(ValueError, expected_message):
                    bind_current_scene(program, scene, wrong_templates, wrong_owners)
                match.assert_not_called()
        # END current-owner binder contract.


# BEGIN partial on geometry, carrier mapping, and phase-preservation regressions.
class UnorderedSupportProgramTest(unittest.TestCase):
    def test_support_basis_origin_and_point_order_are_only_coordinate_choices(self):
        program, scene = on_fixture()
        original = ground_program(program, scene)
        for angle, reflected in ((.7, False), (-1.1, True), (np.pi, False)):
            changed = deepcopy(scene)
            surface = changed["support"]
            basis = np.asarray(surface["basis"])
            boundary = np.asarray(surface["boundary"])
            transform = Rotation.from_rotvec([0., 0., angle]).as_matrix()[:2, :2]
            if reflected:
                transform[:, 1] *= -1
            shift = np.array([.08, -.04])
            surface["origin"] = np.asarray(surface["origin"]) + basis @ shift
            surface["basis"] = basis @ transform
            surface["boundary"] = ((boundary - shift) @ transform)[[2, 0, 3, 1]]
            changed["body_points"] = changed["body_points"][::-1]
            face = changed["functional_bindings"]["effect"]["on_geometry"]
            face["carrier_region_points"] = face["carrier_region_points"][[2, 0, 3, 1]]
            with self.subTest(angle=angle, reflected=reflected):
                result = ground_program(program, changed)
                np.testing.assert_allclose(result["object_poses"], original["object_poses"], atol=1e-9)
                np.testing.assert_allclose(result["eef_poses"], original["eef_poses"], atol=1e-9)

    def test_output_frame_is_equivariant_but_source_reference_frame_cancels(self):
        program, scene = on_fixture()
        original = ground_program(program, scene)
        world = pose(x=.4, y=-.3, z=.2)
        world[:3, :3] = Rotation.from_euler("xyz", [.4, -.3, .7]).as_matrix()
        changed = deepcopy(scene)
        changed["initial_pose"] = world @ scene["initial_pose"]
        changed["eef_pose"] = world @ scene["eef_pose"]
        surface = changed["support"]
        surface["origin"] = world[:3, :3] @ surface["origin"] + world[:3, 3]
        surface["normal"] = world[:3, :3] @ surface["normal"]
        surface["basis"] = world[:3, :3] @ surface["basis"]
        transformed = ground_program(program, changed)
        np.testing.assert_allclose(transformed["object_poses"], world @ original["object_poses"], atol=1e-9)
        np.testing.assert_allclose(transformed["eef_poses"], world @ original["eef_poses"], atol=1e-9)
        source_change = pose(x=-.2, y=.1, z=.4)
        source_change[:3, :3] = Rotation.from_euler("xyz", [-.6, .1, -.4]).as_matrix()
        source_program = deepcopy(program)
        source_program["demo"]["T_reference_carrier"] = source_change @ program["demo"]["T_reference_carrier"]
        changed_reference = ground_program(source_program, scene)
        np.testing.assert_allclose(changed_reference["object_poses"], original["object_poses"], atol=1e-9)
        np.testing.assert_allclose(changed_reference["eef_poses"], original["eef_poses"], atol=1e-9)

    def test_nonidentity_carrier_mapping_preserves_noncommuting_motion_and_function_origin(self):
        from manipulation.program import _event_transfer
        program, scene = on_fixture()
        source = program["demo"]["T_reference_carrier"]
        source[:, :3, :3] = Rotation.from_euler("z", np.linspace(.2, .9, len(source))).as_matrix()
        source[:, 2, 3] = .08
        mapping = pose(x=.035, y=-.025)
        mapping[:3, :3] = Rotation.from_euler("z", .6).as_matrix()
        scene["functional_bindings"]["effect"]["carrier_mapping"] = mapping
        material = pose(x=.023, y=.014)
        program["demo"]["functional_events"][0]["carrier_frame"] = material
        scene["initial_pose"] = pose(x=.04, y=.02, z=.12)
        scene["eef_pose"] = scene["initial_pose"] @ scene["holding"]["T_carrier_eef"]
        expected = scene["initial_pose"] @ mapping @ np.linalg.inv(source[0]) @ source @ np.linalg.inv(mapping)
        wrong = scene["initial_pose"] @ np.linalg.inv(source[0]) @ source
        self.assertGreater(np.max(np.abs(expected - wrong)), .005)
        with patch("manipulation.program._event_transfer", wraps=_event_transfer) as transfer:
            result = ground_program(program, scene)
        np.testing.assert_allclose(transfer.call_args.args[0], expected[:8], atol=1e-10)
        np.testing.assert_allclose(transfer.call_args.args[3], mapping @ material, atol=1e-10)
        # The finite support changes height, while the unrequired yaw and XY
        # stay at the mapped preference because this broad support admits it.
        np.testing.assert_allclose(result["object_poses"][5, :3, :3], expected[5, :3, :3], atol=1e-9)
        np.testing.assert_allclose(result["object_poses"][5, :2, 3], expected[5, :2, 3], atol=1e-9)
        self.assertAlmostEqual(result["object_poses"][5, 2, 3], 0.)

    def test_explicit_consistent_carrier_mapping_is_required(self):
        program, scene = on_fixture()
        del scene["functional_bindings"]["effect"]["carrier_mapping"]
        with self.assertRaisesRegex(ValueError, "explicit evidenced carrier_mapping"):
            ground_program(program, scene)
        program, scene = on_fixture()
        event = deepcopy(program["demo"]["functional_events"][0])
        event.update(event_id="second", frame_index=106)
        program["demo"]["functional_events"].append(event)
        second = deepcopy(scene["functional_bindings"]["effect"])
        second["carrier_mapping"] = pose(x=.01)
        scene["functional_bindings"]["second"] = second
        with self.assertRaisesRegex(ValueError, "same carrier_mapping"):
            ground_program(program, scene)

    def test_independent_source_and_current_carrier_gauges_preserve_physical_eef_and_retreat(self):
        program, scene = on_fixture()
        phase = np.linspace(0., 1., len(program["demo"]["frame_indices"]))
        source = program["demo"]["T_reference_carrier"]
        source[:, :3, :3] = Rotation.from_euler("xyz", phase[:, None] * [.12, -.2, .6]).as_matrix()
        source[:, 2, 3] = .06 * np.sin(np.pi * phase)
        mapping = pose(x=.015, y=-.02, z=.035)
        mapping[:3, :3] = Rotation.from_euler("xyz", [.25, -.35, .5]).as_matrix()
        scene["functional_bindings"]["effect"]["carrier_mapping"] = mapping
        material = pose(x=.013, y=-.008, z=.003)
        material[:3, :3] = Rotation.from_euler("xyz", [-.3, .1, .2]).as_matrix()
        program["demo"]["functional_events"][0]["carrier_frame"] = material
        actual = pose(x=.01, y=.004, z=.045)
        actual[:3, :3] = Rotation.from_euler("xyz", [.1, .2, -.3]).as_matrix()
        scene["holding"]["T_carrier_eef"] = actual
        scene["eef_pose"] = scene["initial_pose"] @ actual
        original = ground_program(program, scene)

        # These are coordinate re-expressions of identical physical input,
        # not expected trajectories produced with the implementation formula.
        # Q maps coordinates in a newly chosen local frame into the old one.
        source_gauge = pose(x=-.07, y=.04, z=.025)
        source_gauge[:3, :3] = Rotation.from_euler("xyz", [-.5, .3, -.4]).as_matrix()
        source_changed, same_scene = deepcopy(program), deepcopy(scene)
        source_changed["demo"]["T_reference_carrier"] = source @ source_gauge
        source_changed["demo"]["T_carrier_eef"] = np.linalg.inv(source_gauge) @ program["demo"]["T_carrier_eef"]
        source_changed["demo"]["functional_events"][0]["carrier_frame"] = np.linalg.inv(source_gauge) @ material
        same_scene["functional_bindings"]["effect"]["carrier_mapping"] = mapping @ source_gauge
        source_result = ground_program(source_changed, same_scene)
        np.testing.assert_allclose(source_result["object_poses"], original["object_poses"], atol=1e-9)
        np.testing.assert_allclose(source_result["eef_poses"], original["eef_poses"], atol=1e-9)

        current_gauge = pose(x=.09, y=-.06, z=.035)
        current_gauge[:3, :3] = Rotation.from_euler("xyz", [.4, -.3, .7]).as_matrix()
        changed = deepcopy(scene)
        inverse = np.linalg.inv(current_gauge)
        changed["initial_pose"] = scene["initial_pose"] @ current_gauge
        changed["holding"]["T_carrier_eef"] = inverse @ actual
        changed["body_points"] = scene["body_points"] @ inverse[:3, :3].T + inverse[:3, 3]
        binding = changed["functional_bindings"]["effect"]
        binding["carrier_mapping"] = inverse @ mapping
        face = binding["on_geometry"]
        face["carrier_region_points"] = face["carrier_region_points"] @ inverse[:3, :3].T + inverse[:3, 3]
        face["carrier_outward_normal"] = inverse[:3, :3] @ face["carrier_outward_normal"]
        # Physical robot feedback and world support are unchanged. Every EEF
        # node, including the independent release suffix, must remain the same.
        current_result = ground_program(program, changed)
        np.testing.assert_allclose(current_result["object_poses"], original["object_poses"] @ current_gauge, atol=1e-9)
        np.testing.assert_allclose(current_result["eef_poses"], original["eef_poses"], atol=1e-9)
        self.assertGreater(np.linalg.norm(original["eef_poses"][-1, :3, 3] - original["eef_poses"][7, :3, 3]), .02)

    def test_full_273_node_timeline_keeps_internal_effect_release_and_independent_retreat(self):
        program, scene = on_fixture()
        frames = np.arange(197, 470)
        times = frames.astype(float) / 20.
        count, effect, release = len(frames), 402 - 197, 405 - 197
        source = np.repeat(pose()[None], count, axis=0)
        source[:, 0, 3] = np.r_[np.linspace(0., .12, effect + 1), np.repeat(.12, count - effect - 1)]
        source[:effect + 1, 2, 3] = .08 * np.sin(np.linspace(0., np.pi, effect + 1))
        source_hand = np.repeat(pose(z=.025)[None], count, axis=0)
        source_hand[release:, 1, 3] -= np.linspace(0., .045, count - release)
        source_hand[release:, 2, 3] += np.linspace(0., .075, count - release)
        source_hand[release:, :3, :3] = Rotation.from_euler("x", np.linspace(0., .25, count - release)).as_matrix()
        demo = program["demo"]
        demo.update(frame_indices=frames, timestamps=times,
                    T_reference_carrier=source, T_carrier_eef=source_hand,
                    release_frame=405,
                    gripper_events=[{"event_id": "open", "frame_index": 403, "command": "open"}])
        demo["functional_events"][0]["frame_index"] = 402
        actual = pose(x=.005, y=-.008, z=.04)
        actual[:3, :3] = Rotation.from_euler("xyz", [.2, -.1, .4]).as_matrix()
        scene["holding"]["T_carrier_eef"] = actual
        scene["eef_pose"] = scene["initial_pose"] @ actual
        result = ground_program(program, scene)
        np.testing.assert_array_equal(result["frame_indices"], frames)
        np.testing.assert_array_equal(result["timestamps"], times)
        self.assertEqual(result["object_poses"].shape, (209, 4, 4))
        self.assertEqual(result["eef_poses"].shape, (273, 4, 4))
        self.assertEqual(result["function_checks"][0]["frame_index"], 402)
        self.assertEqual(result["release_frame"], 405)
        self.assertEqual(result["gripper_events"][0]["frame_index"], 403)
        np.testing.assert_allclose(result["eef_poses"][:release + 1], result["object_poses"] @ actual, atol=1e-9)
        source_eef = source @ source_hand
        np.testing.assert_allclose(np.linalg.inv(result["eef_poses"][release]) @ result["eef_poses"][release:],
                                   np.linalg.inv(source_eef[release]) @ source_eef[release:], atol=1e-9)
        self.assertGreater(np.linalg.norm(result["eef_poses"][-1, :3, 3] - result["eef_poses"][release, :3, 3]), .08)
        self.assertEqual(result["released_object_state"], "requires_observation")
# END partial on geometry and complete phase regressions.


if __name__ == "__main__":
    unittest.main()
# END function event and actual-holding integration tests.
