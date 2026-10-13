"""CPU owner-observation grounding tests with real geometric consumers.

The client and image descriptors are controlled fixtures. They test API,
coordinate, recipe and numerical contracts, not Gemini/MARCO recognition
accuracy or successful robot execution.
"""

# BEGIN cached current-owner grounding regressions.
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from manipulation.goals import SupportSurface
from manipulation.perception import bind_carrier_contact, bind_current_scene, ground_current_program
from manipulation.regions import source_function_geometry


def pose(x=0., y=0., z=0.):
    value = np.eye(4)
    value[:3, 3] = [x, y, z]
    return value


# BEGIN MARCO-compatible synthetic features retain exact geometry test identities.
def descriptors(first, second):
    from tests.marco_fixtures import labeled_pair
    return labeled_pair(first, second)
# END only feature inference is mocked.


def evidence():
    # BEGIN sparse pixel labels are an explicit synthetic feature-inference fixture.
    rgb = np.zeros((20, 20, 3), np.uint8)
    for label, (x, y) in enumerate(((4, 4), (14, 4), (4, 14)), start=1):
        rgb[y, x, 0] = label
    image = Image.fromarray(rgb)
    # END program geometry and known source queries are unchanged.
    mask = np.zeros((20, 20), bool)
    mask[3:6, 3:6] = True
    origin = pose(.04, .04, 1.)
    source = {
        "image": image, "mask": mask, "owner_mask": np.ones((20, 20), bool),
        "depth": np.ones((20, 20)), "K": np.diag([100., 100., 1.]),
        "T_camera_owner": origin, "region_id": "demonstrated_contact", "owner_id": "old_carrier",
        "description": "The highlighted surface and two evidenced direction witnesses.",
        "query_pixels": [[4., 4.], [14., 4.], [4., 14.]],
        "query_descriptions": ["interaction origin", "positive horizontal witness", "vertical witness"],
        "frame_recipe": {"kind": "frame", "origin_weights": [1., 0., 0.],
                         "axis_weights": [-1., 1., 0.], "plane_weights": [1., 0., -1.],
                         "source": "fixture establishes the signed x and positive-z function normal"},
    }
    current = {key: deepcopy(source[key]) for key in ("image", "owner_mask", "depth", "K")}
    current.update(owner_id="new_carrier", frame_convention="tracked_owner",
                   T_camera_owner=origin.copy(), T_output_camera=np.linalg.inv(origin))
    return source, current


def fixture(kind="manipulation"):
    source, current = evidence()
    frames = np.arange(10, 15)
    hand = np.repeat(pose(z=.03)[None], 5, axis=0)
    demo = {"carrier_id": "old_carrier", "frame_indices": frames,
            "timestamps": 1. + np.arange(5)*.05, "T_carrier_eef": hand,
            "tool_id": "same_gripper", "tool_frame": "eef_site",
            "gripper_events": [{"event_id": "close", "frame_index": 12, "command": "close"}]}
    scene = {"reference_frame": "output", "tool_id": "same_gripper", "tool_frame": "eef_site",
             "eef_pose": pose(z=.03)}
    owners = {"carrier": current}
    if kind == "grasp":
        demo["protected_start_frame"] = 12
        scene["grasp_goal"] = {"T_carrier_eef": pose(z=.03), "position_tolerance_m": .002,
                               "rotation_tolerance_rad": .03, "source": "declared current-program fixture task relation"}
        return {"skill_kind": kind, "demo": demo}, scene, {"grasp": source}, owners
    reference = deepcopy(source)
    reference.update(owner_id="old_reference", region_id="demonstrated_target")
    owners["reference"] = deepcopy(current)
    owners["reference"]["owner_id"] = "new_reference"
    objects = np.repeat(np.eye(4)[None], 5, axis=0)
    objects[:, 0, 3] = np.linspace(0., .02, 5)
    event = {"event_id": "effect", "frame_index": 13,
             "carrier_frame": source_function_geometry(source)["source_frame"],
             "reference_frame": source_function_geometry(reference)["source_frame"],
             "source": "independently compiled functional event",
             "position_tolerance_m": 1e-6, "rotation_tolerance_rad": 1e-6}
    demo.update(reference_id="old_reference", T_reference_carrier=objects,
                functional_events=[event], gripper_events=[])
    scene["holding"] = {"verified": True, "source": "current synchronized measured hold",
                         "T_carrier_eef": pose(z=.03)}
    program = {"skill_kind": kind, "demo": demo, "task": {"relation": "transfer", "motion": "free"}}
    return program, scene, {"effect": {"carrier": source, "reference": reference}}, owners


def client():
    # Coarse cells merely define search windows. Real masked descriptor matching
    # and RGB-D frame reconstruction must still find the three fixture pixels.
    result = Mock()
    result.chat_json.return_value = {"cells": [7, 9, 17]}
    return result


class CurrentProgramTest(unittest.TestCase):
    # BEGIN optional observed-part evidence reaches the real grasp consumer.
    def test_selected_part_survives_normalization_and_preserves_outside_witnesses(self):
        program, scene, templates, owners = fixture("grasp")
        original_queries = deepcopy(templates["grasp"]["query_pixels"])
        baseline = ground_current_program(program, scene, templates, owners,
                                          client=client(), extractor=descriptors)
        part = np.zeros((20, 20), bool)
        part[3:6, 3:6] = True
        owners["carrier"]["interaction_evidence"] = {
            "mask": part, "owner_id": owners["carrier"]["owner_id"],
            "description": "the visible working surface",
            "source": "detector candidate 3, selected interaction_region"}
        before_owner = owners["carrier"]["owner_mask"].copy()
        model, extractor = client(), Mock(side_effect=descriptors)
        result = ground_current_program(program, scene, templates, owners,
                                        client=model, extractor=extractor)
        self.assertIn("the visible working surface", model.chat_json.call_args.args[0])
        self.assertEqual(extractor.call_count, 1)
        diagnostic = result["region_binding_evidence"]["grasp"]["diagnostics"]
        observed = diagnostic["current_interaction_evidence"]
        self.assertEqual(observed["owner_id"], "new_carrier")
        self.assertEqual(observed["query_inside_mask"], [True, False, False])
        restored = np.zeros_like(part)
        xy = observed["mask_pixels_xy"]
        restored[xy[:, 1], xy[:, 0]] = True
        np.testing.assert_array_equal(restored, part)
        np.testing.assert_array_equal(diagnostic["target_pixels"], original_queries)
        np.testing.assert_allclose(result["eef_poses"], baseline["eef_poses"], atol=1e-12)
        self.assertEqual(result["holding_status"], "requires_verification")
        self.assertEqual(result["owner_observation_evidence"]["carrier"]["interaction_evidence"]["mask_pixel_count"], 9)
        np.testing.assert_array_equal(owners["carrier"]["owner_mask"], before_owner)
        np.testing.assert_array_equal(templates["grasp"]["query_pixels"], original_queries)
        # The blue annotation belongs to the VLM panel only, not MARCO input.
        np.testing.assert_array_equal(extractor.call_args.args[0], templates["grasp"]["image"])
        np.testing.assert_array_equal(extractor.call_args.args[1], owners["carrier"]["image"])

    def test_part_bound_to_another_owner_fails_before_any_model(self):
        program, scene, templates, owners = fixture("grasp")
        owners["carrier"]["interaction_evidence"] = {
            "mask": owners["carrier"]["owner_mask"].copy(), "owner_id": "different_object",
            "description": "a similar visible surface", "source": "semantic candidate"}
        model, extractor = client(), Mock()
        with self.assertRaisesRegex(ValueError, "different selected owner"):
            bind_current_scene(program, scene, templates, owners, client=model, extractor=extractor)
        model.chat_json.assert_not_called()
        extractor.assert_not_called()
    # END optional evidence does not modify source motion, queries or ownership.

    def test_owner_only_inputs_reach_real_grasp_and_manipulation_consumers(self):
        for kind, expected in (("grasp", "eef_trajectory"), ("manipulation", "interaction_trajectory")):
            program, scene, templates, owners = fixture(kind)
            before_depth = owners["carrier"]["depth"].copy()
            model = client()
            with self.subTest(kind=kind):
                result = ground_current_program(program, scene, templates, owners,
                                                client=model, extractor=descriptors)
                self.assertEqual(result["output_type"], expected)
                self.assertEqual(result["status"], "candidate")
                self.assertEqual(model.chat_json.call_count, 1 if kind == "grasp" else 2)
                np.testing.assert_array_equal(owners["carrier"]["depth"], before_depth)
                self.assertNotIn("carrier_pose", scene)
                self.assertNotIn("functional_bindings", scene)
                self.assertNotIn("mask", owners["carrier"])
                self.assertNotIn("region_id", owners["carrier"])
                if kind == "grasp":
                    self.assertEqual(result["holding_status"], "requires_verification")
                    self.assertNotIn("reference", result["owner_observation_evidence"])
                else:
                    self.assertTrue(all(check["position_error_m"] < 1e-8
                                        and check["rotation_error_rad"] < 1e-8
                                        for check in result["function_checks"]))
                    np.testing.assert_allclose(result["object_poses"][3], pose(x=.015), atol=1e-8)

    def test_every_source_frame_is_checked_before_any_current_model(self):
        program, scene, templates, owners = fixture()
        # A second event prevents a per-event validate-then-model implementation
        # from accidentally spending calls before discovering later bad source.
        event = deepcopy(program["demo"]["functional_events"][0])
        event.update(event_id="later", frame_index=14)
        event["reference_frame"] = pose(z=.1)
        program["demo"]["functional_events"].append(event)
        templates["later"] = deepcopy(templates["effect"])
        model, extractor = client(), Mock()
        with self.assertRaisesRegex(ValueError, "Source reference recipe frame"):
            bind_current_scene(program, scene, templates, owners, client=model, extractor=extractor)
        model.chat_json.assert_not_called()
        extractor.assert_not_called()

    def test_ids_and_existing_poses_are_not_silently_overwritten(self):
        for field in ("source_owner", "current_owner", "initial_pose", "carrier_pose", "reference_pose"):
            program, scene, templates, owners = fixture()
            if field == "source_owner":
                templates["effect"]["reference"]["owner_id"] = "wrong_old_instance"
            elif field == "current_owner":
                scene["carrier_id"] = "another_similar_instance"
            else:
                scene[field] = pose(x=.2)
            model = client()
            with self.subTest(field=field), self.assertRaises(ValueError):
                bind_current_scene(program, scene, templates, owners, client=model, extractor=descriptors)
            model.chat_json.assert_not_called()

    def test_binding_camera_gauge_preserves_physical_eef_and_observed_body(self):
        program, scene, templates, owners = fixture()
        tracked = ground_current_program(program, scene, templates, owners, client=client(), extractor=descriptors)
        gauge_owners = deepcopy(owners)
        old_carrier = gauge_owners["carrier"]["T_camera_owner"].copy()
        for observation in gauge_owners.values():
            observation["frame_convention"] = "binding_camera"
            del observation["T_camera_owner"]
        gauge_scene = deepcopy(scene)
        gauge_scene["holding"]["T_carrier_eef"] = old_carrier @ scene["holding"]["T_carrier_eef"]
        changed = ground_current_program(program, gauge_scene, templates, gauge_owners,
                                         client=client(), extractor=descriptors)
        np.testing.assert_allclose(changed["eef_poses"], tracked["eef_poses"], atol=1e-8)
        np.testing.assert_allclose(changed["object_poses"] @ old_carrier, tracked["object_poses"], atol=1e-8)
        for evidence in changed["owner_observation_evidence"].values():
            np.testing.assert_array_equal(evidence["T_camera_owner"], np.eye(4))
        # Keeping an attachment measured in the old owner gauge is a real
        # synchronization error, even though all individual matrices are SE(3).
        with self.assertRaisesRegex(ValueError, "not synchronized"):
            ground_current_program(program, scene, templates, gauge_owners, client=client(), extractor=descriptors)

    def test_output_calibration_is_equivariant_and_must_be_explicit(self):
        program, scene, templates, owners = fixture("grasp")
        first = ground_current_program(program, scene, templates, owners, client=client(), extractor=descriptors)
        world = pose(.3, -.1, .2)
        world[:3, :3] = Rotation.from_rotvec([.2, -.15, .1]).as_matrix()
        moved_owners, moved_scene = deepcopy(owners), deepcopy(scene)
        moved_owners["carrier"]["T_output_camera"] = world @ owners["carrier"]["T_output_camera"]
        moved_scene["eef_pose"] = world @ scene["eef_pose"]
        second = ground_current_program(program, moved_scene, templates, moved_owners,
                                        client=client(), extractor=descriptors)
        np.testing.assert_allclose(second["eef_poses"], world @ first["eef_poses"], atol=1e-8)
        del moved_owners["carrier"]["T_output_camera"]
        model = client()
        with self.assertRaisesRegex(ValueError, "T_output_camera"):
            bind_current_scene(program, moved_scene, templates, moved_owners, client=model)
        model.chat_json.assert_not_called()

    def test_explicit_object_output_never_implies_holding_or_eef_fallback(self):
        program, scene, templates, owners = fixture()
        scene["holding"] = {"verified": False, "candidate_T_carrier_eef": pose(z=.03)}
        body = ground_current_program(program, scene, templates, owners, output="object",
                                      client=client(), extractor=descriptors)
        self.assertEqual(body["output_type"], "object_trajectory")
        self.assertNotIn("eef_poses", body)
        with self.assertRaisesRegex(ValueError, "actual holding"):
            ground_current_program(program, scene, templates, owners, output="eef",
                                   client=client(), extractor=descriptors)
        grasp, grasp_scene, grasp_templates, grasp_owners = fixture("grasp")
        with self.assertRaisesRegex(ValueError, "only defined for manipulation"):
            ground_current_program(grasp, grasp_scene, grasp_templates, grasp_owners, output="object")
        with self.assertRaisesRegex(ValueError, "output must"):
            ground_current_program(program, scene, templates, owners, output="automatic")

    def test_plane_cannot_satisfy_grasp_or_full_frame_manipulation(self):
        for kind in ("grasp", "manipulation"):
            program, scene, templates, owners = fixture(kind)
            template = templates["grasp"] if kind == "grasp" else templates["effect"]["carrier"]
            template["frame_recipe"]["kind"] = "plane"
            model = client()
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "insufficient direction"):
                bind_current_scene(program, scene, templates, owners, client=model, extractor=descriptors)
            model.chat_json.assert_not_called()

    def test_caller_body_geometry_is_retained_with_separate_visible_evidence(self):
        program, scene, templates, owners = fixture("grasp")
        supplied = np.array([[0., 0., -.05], [.1, 0., -.05], [0., .1, -.05]])
        scene.update(body_points=supplied.copy(), body_points_source="current known body model")
        bound = bind_current_scene(program, scene, templates, owners, client=client(), extractor=descriptors)
        np.testing.assert_array_equal(bound["body_points"], supplied)
        self.assertGreater(len(bound["observed_body_points"]), len(supplied))
        self.assertEqual(bound["body_points_source"], "current known body model")

    # BEGIN unordered support and explicitly bound current carrier tests.
    @staticmethod
    def on_fixture():
        program, scene, _, owners = fixture()
        program["task"]["relation"] = "on"
        event = program["demo"]["functional_events"][0]
        event["carrier_frame"] = np.eye(4)
        del event["reference_frame"]
        bottom = np.array([[-.01, -.01, 0.], [.01, -.01, 0.], [.01, .01, 0.], [-.01, .01, 0.]])
        scene.update(carrier_id="new_carrier", reference_id="new_reference", initial_pose=np.eye(4),
                     body_points=np.concatenate((bottom, bottom+[0., 0., .02])),
                     body_points_source="independently bound finite current fixture body",
                     support_direction=[0., 0., 1.], support_direction_source="calibrated fixture physical up",
                     functional_bindings={"effect": {
                         "carrier_id": "new_carrier", "reference_id": "new_reference",
                         "carrier_mapping": np.eye(4),
                         "source": "independently established common source/current carrier coordinates",
                         "on_geometry": {"carrier_region_points": bottom, "carrier_outward_normal": [0., 0., -1.],
                                         "gap_tolerance": 1e-6, "normal_tolerance": 1e-6,
                                         "source": "independently observed fixture bottom face"}}})
        templates = {"effect": {"reference": {
            "owner_id": "old_reference", "region_id": "demo_support", "normal_role": "support",
            "description": "Use the demonstrated top support surface.",
            "source": "offline demonstrated support semantics"}}}
        reference = owners["reference"]
        reference["frame_convention"] = "binding_camera"
        del reference["T_camera_owner"]
        support = SupportSurface(np.array([0., 0., 1.]), np.eye(3)[:, :2], np.array([0., 0., 1.]),
                                 np.array([[0., 0.], [.19, 0.], [.19, .19], [0., .19]]),
                                 source="independent observed reference plane fixture")
        return program, scene, templates, {"reference": reference}, support

    def test_on_consumes_unordered_support_without_any_reference_queries_or_pose(self):
        program, scene, templates, owners, support = self.on_fixture()
        with patch("manipulation.perception.infer_support", return_value=(support, {"fixture": "visible plane"})) as infer, \
                patch("manipulation.perception.match_functional_region") as match:
            result = ground_current_program(program, scene, templates, owners, client=client())
        infer.assert_called_once()
        match.assert_not_called()
        np.testing.assert_allclose(infer.call_args.kwargs["support_direction"], [0., 0., 1.])
        self.assertEqual(result["output_type"], "interaction_trajectory")
        self.assertTrue(result["function_checks"][0]["valid"])
        self.assertEqual(result["function_checks"][0]["constraint_kind"], "on_region")
        self.assertNotIn("rotation_error_rad", result["function_checks"][0])
        self.assertNotIn("support", scene)
        self.assertNotIn("reference_pose", scene)
        self.assertNotIn("reference_frame", program["demo"]["functional_events"][0])
        reference = result["region_binding_evidence"]["effect"]["reference"]
        self.assertEqual(reference["binding_kind"], "unordered_observed_support")
        self.assertNotIn("target_frame", reference)
        np.testing.assert_array_equal(result["frame_indices"], program["demo"]["frame_indices"])

    def test_on_preserves_bound_carrier_geometry_without_requiring_its_hidden_bottom_image(self):
        program, scene, templates, owners, support = self.on_fixture()
        before = scene["body_points"].copy()
        with patch("manipulation.perception.infer_support", return_value=(support, {})):
            bound = bind_current_scene(program, scene, templates, owners, client=client())
        np.testing.assert_array_equal(bound["body_points"], before)
        self.assertNotIn("carrier", owners)
        self.assertEqual(bound["body_points_source"], scene["body_points_source"])
        self.assertEqual(bound["holding"]["source"], scene["holding"]["source"])

    def test_on_rejects_missing_binding_semantics_and_unsupported_corner_recipe_before_models(self):
        for error in ("mapping", "direction", "source", "normal_role", "owner", "corners", "body_source"):
            program, scene, templates, owners, support = self.on_fixture()
            binding = scene["functional_bindings"]["effect"]
            reference = templates["effect"]["reference"]
            if error == "mapping":
                binding["carrier_mapping"] = np.zeros((4, 4))
            elif error == "direction":
                scene["support_direction"] = [0., 0., 0.]
            elif error == "source":
                binding["source"] = ""
            elif error == "normal_role":
                del reference["normal_role"]
            elif error == "owner":
                owners["reference"]["owner_id"] = "wrong_instance"
            elif error == "corners":
                reference["query_pixels"] = [[1, 1], [2, 1], [2, 2]]
            else:
                del scene["body_points_source"]
            with self.subTest(error=error), patch("manipulation.perception.infer_support") as infer:
                with self.assertRaises(ValueError):
                    bind_current_scene(program, scene, templates, owners, client=client())
                infer.assert_not_called()

    def test_on_support_coordinates_do_not_change_target_or_complete_eef_trajectory(self):
        program, scene, templates, owners, support = self.on_fixture()
        with patch("manipulation.perception.infer_support", return_value=(support, {})):
            first = ground_current_program(program, scene, templates, owners, client=client())
        for theta, reflection in ((.8, False), (-1.3, True)):
            uv = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            if reflection:
                uv[:, 0] *= -1
            offset = np.array([.03, -.02])
            changed = SupportSurface(support.origin + support.basis @ offset, support.basis @ uv,
                                     support.normal, (support.boundary-offset) @ uv, source=support.source)
            with self.subTest(theta=theta), patch("manipulation.perception.infer_support", return_value=(changed, {})):
                second = ground_current_program(program, scene, templates, owners, client=client())
            np.testing.assert_allclose(first["eef_poses"], second["eef_poses"], atol=1e-8)

    def test_on_output_frame_change_rotates_side_once_and_preserves_physical_result(self):
        program, scene, templates, owners, support = self.on_fixture()
        with patch("manipulation.perception.infer_support", return_value=(support, {})):
            first = ground_current_program(program, scene, templates, owners, client=client())
        transform = pose(.3, -.2, .1)
        transform[:3, :3] = Rotation.from_rotvec([.2, .1, -.4]).as_matrix()
        owners["reference"]["T_output_camera"] = transform @ owners["reference"]["T_output_camera"]
        scene["initial_pose"] = transform @ scene["initial_pose"]
        scene["eef_pose"] = transform @ scene["eef_pose"]
        scene["support_direction"] = transform[:3, :3] @ scene["support_direction"]
        with patch("manipulation.perception.infer_support", return_value=(support, {})) as infer:
            second = ground_current_program(program, scene, templates, owners, client=client())
        np.testing.assert_allclose(infer.call_args.kwargs["support_direction"], [0., 0., 1.], atol=1e-12)
        np.testing.assert_allclose(second["eef_poses"], transform @ first["eef_poses"], atol=1e-8)

    def test_on_cannot_enlarge_observed_patch_to_fit_the_carrier(self):
        program, scene, templates, owners, support = self.on_fixture()
        tiny = SupportSurface(support.origin, support.basis, support.normal,
                              np.array([[.06, .06], [.061, .06], [.061, .061], [.06, .061]]), source=support.source)
        with patch("manipulation.perception.infer_support", return_value=(tiny, {})):
            bound = bind_current_scene(program, scene, templates, owners, client=client())
        np.testing.assert_array_equal(bound["support"]["boundary"], tiny.boundary)
        from manipulation.program import ground_program
        # BEGIN failed bounded placement keeps the exact observed support.
        grounded = ground_program(program, bound)
        check = grounded["function_checks"][0]
        self.assertEqual(check["reference_generation"]["status"], "failed")
        self.assertFalse(check["checked_member_satisfies_support"])
        np.testing.assert_array_equal(bound["support"]["boundary"], tiny.boundary)
        # END this diagnostic seed is not a placement or feasibility certificate.

    def test_upstream_visible_carrier_contact_remains_available_without_ground_truth(self):
        source, current = evidence()
        source["normal_role"] = "outward"
        source["frame_recipe"]["kind"] = "plane"
        source["mask"][:] = True
        result = bind_carrier_contact(source, current, position_tolerance_m=1e-6,
                                      normal_tolerance_rad=1e-6, client=client(), extractor=descriptors)
        np.testing.assert_allclose(result["carrier_mapping"], np.eye(4), atol=1e-8)
        self.assertEqual(len(result["on_geometry"]["carrier_region_points"]), 3)
        self.assertIn("sparse visible", result["on_geometry"]["coverage"])
        self.assertNotIn("holding", result)
        source["mask"][:] = False
        source["mask"][3:6, 3:6] = True
        with self.assertRaisesRegex(ValueError, "at least 3"):
            bind_carrier_contact(source, current, position_tolerance_m=1e-6,
                                 normal_tolerance_rad=1e-6, client=client(), extractor=descriptors)
    # END unordered support and explicitly bound current carrier tests.


if __name__ == "__main__":
    unittest.main()
# END cached current-owner grounding regressions.
