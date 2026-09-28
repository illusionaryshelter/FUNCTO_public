"""CPU owner-observation grounding tests with real geometric consumers.

The client and image descriptors are controlled fixtures. They test API,
coordinate, recipe and numerical contracts, not Gemini/SD-DINO recognition
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
from manipulation.perception import bind_current_scene, ground_current_program
from manipulation.regions import source_function_geometry


def pose(x=0., y=0., z=0.):
    value = np.eye(4)
    value[:3, 3] = [x, y, z]
    return value


def descriptors(first, second):
    """Image-coordinate-coded features keep corresponding fixture pixels exact."""
    values = []
    for image in (first, second):
        rgb = np.asarray(image, dtype=np.float32)
        x, y = rgb[:, :, 0] / 255, rgb[:, :, 1] / 255
        values.append(np.stack((np.ones_like(x), x, y, x*y))[None])
    return values


def evidence():
    y, x = np.indices((20, 20))
    image = Image.fromarray(np.stack((x*12, y*12, np.full_like(x, 25)), axis=-1).astype(np.uint8))
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

    @staticmethod
    def on_fixture():
        program, scene, templates, owners = fixture()
        program["task"]["relation"] = "on"
        for role, normal_role in (("carrier", "outward"), ("reference", "support")):
            template = templates["effect"][role]
            template["normal_role"] = normal_role
            template["mask"][:] = True
            template["frame_recipe"]["kind"] = "plane"
            if role == "carrier":
                template["frame_recipe"]["plane_weights"] = [-1., 0., 1.]
            template["frame_recipe"]["source"] = f"offline source establishes the signed {normal_role} normal"
            program["demo"]["functional_events"][0][role + "_frame"] = source_function_geometry(template)["source_frame"]
        support = SupportSurface(np.array([0., 0., 1.]), np.eye(3)[:, :2], np.array([0., 0., 1.]),
                                 np.array([[0., 0.], [.19, 0.], [.19, .19], [0., .19]]),
                                 source="independent observed reference plane fixture")
        return program, scene, templates, owners, support

    def test_on_infers_sparse_contact_and_support_without_preprovided_geometry(self):
        program, scene, templates, owners, support = self.on_fixture()
        with patch("manipulation.perception.infer_support", return_value=(support, {"source": support.source})) as infer:
            bound = bind_current_scene(program, scene, templates, owners, client=client(), extractor=descriptors)
        infer.assert_called_once()
        # The support side comes from the signed reference template, not the
        # carrier's starting location. Native camera geometry is passed here.
        np.testing.assert_allclose(infer.call_args.args[4], [.04, .04, 1.05], atol=1e-10)
        face = bound["functional_bindings"]["effect"]["on_geometry"]
        self.assertEqual(len(face["carrier_region_points"]), 3)
        self.assertIn("sparse visible", face["coverage"])
        self.assertIn("tangent is a numerical preference",
                      bound["functional_bindings"]["effect"]["orientation_source"]["carrier"])
        self.assertNotIn("support", scene)
        # Use the actual on consumer: its hard requirement is the face/support
        # relation, and no fictitious full-frame rotation error is reported.
        with patch("manipulation.perception.infer_support", return_value=(support, {"source": support.source})):
            result = ground_current_program(program, scene, templates, owners, output="object",
                                            client=client(), extractor=descriptors)
        check = result["function_checks"][0]
        self.assertEqual(check["constraint_kind"], "on_region")
        self.assertTrue(check["valid"])
        self.assertNotIn("rotation_error_rad", check)
        self.assertIn("sparse visible", result["region_binding_evidence"]["effect"]["on_geometry"]["coverage"])

    def test_provided_contact_model_is_preserved_and_checked_against_current_correspondences(self):
        program, scene, templates, owners, support = self.on_fixture()
        with patch("manipulation.perception.infer_support", return_value=(support, {})):
            known = bind_current_scene(program, scene, templates, owners, client=client(), extractor=descriptors)
        known["functional_bindings"]["effect"]["on_geometry"]["source"] = "independent complete current contact model"
        with patch("manipulation.perception.infer_support") as infer:
            rebound = bind_current_scene(program, known, templates, owners, client=client(), extractor=descriptors)
            infer.assert_not_called()
        self.assertEqual(rebound["functional_bindings"]["effect"]["on_geometry"]["source"],
                         "independent complete current contact model")
        known["functional_bindings"]["effect"]["on_geometry"]["carrier_outward_normal"] *= -1
        with self.assertRaisesRegex(ValueError, "physical contact normal disagree"):
            bind_current_scene(program, known, templates, owners, client=client(), extractor=descriptors)

    def test_direction_queries_cannot_replace_unobserved_contact_samples(self):
        program, scene, templates, owners, support = self.on_fixture()
        # Only the first query is actually in the contact region; the other
        # two establish directions elsewhere on this same owner.
        templates["effect"]["carrier"]["mask"][:] = False
        templates["effect"]["carrier"]["mask"][3:6, 3:6] = True
        with patch("manipulation.perception.infer_support") as infer:
            with self.assertRaisesRegex(ValueError, "at least 3"):
                bind_current_scene(program, scene, templates, owners, client=client(), extractor=descriptors)
            infer.assert_not_called()

    # BEGIN separate functional plane evidence from placement feasibility.
    def test_reference_witnesses_outside_safe_patch_preserve_carrier_fit_checks(self):
        from manipulation.program import ground_object_program

        program, scene, templates, owners, support = self.on_fixture()
        # The three visible reference witnesses are (0.04,0.04), (0.14,0.04)
        # and (0.04,0.14) on the same camera-Z=1 plane. This safe interior patch
        # is fully observed but excludes those corner witnesses. Binding is
        # still meaningful; whether the actual carrier fits is a separate test.
        interior = SupportSurface(support.origin, support.basis, support.normal,
                                  np.array([[.06, .06], [.12, .06], [.12, .12], [.06, .12]]),
                                  source="conservative observed interior placement patch")
        with patch("manipulation.perception.infer_support", return_value=(interior, {})):
            bound = bind_current_scene(program, scene, templates, owners,
                                       client=client(), extractor=descriptors)
        # Retain the finite patch rather than enlarging it to fit the witnesses.
        np.testing.assert_array_equal(bound["support"]["boundary"], interior.boundary)
        self.assertEqual(len(bound["region_binding_evidence"]["effect"]["reference"]["target_points"]), 3)
        # The observed carrier spans 0.19 m, so it cannot fit this 0.06 m patch.
        # Use the real numerical consumer to ensure loosening the *reference
        # witness* gate did not remove the carrier's finite-area requirement.
        with self.assertRaisesRegex(ValueError, "Object geometry does not fit"):
            ground_object_program(program, bound)

    def test_inferred_support_cannot_choose_a_different_plane_or_physical_side(self):
        for wrong in ("height", "normal"):
            program, scene, templates, owners, support = self.on_fixture()
            if wrong == "height":
                support.origin[2] += .02
            else:
                support.normal *= -1
            with self.subTest(wrong=wrong), patch("manipulation.perception.infer_support", return_value=(support, {})):
                with self.assertRaisesRegex(ValueError, "support.*disagrees|support normal disagrees"):
                    bind_current_scene(program, scene, templates, owners, client=client(), extractor=descriptors)
    # END separate functional plane evidence from placement feasibility.

    def test_missing_physical_normal_role_is_a_source_error_before_models(self):
        for role in ("carrier", "reference"):
            program, scene, templates, owners, support = self.on_fixture()
            del templates["effect"][role]["normal_role"]
            model = client()
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "evidenced.*plane normal"):
                bind_current_scene(program, scene, templates, owners, client=model, extractor=descriptors)
            model.chat_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
# END cached current-owner grounding regressions.
