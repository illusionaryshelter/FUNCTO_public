"""CPU counterexamples for rigid registration screening and its core handoff."""

# BEGIN same-geometry projective validation and real-consumer regressions.
from copy import deepcopy
import itertools
import unittest

import numpy as np

from manipulation.perception import ground_registered_candidates
from manipulation.program import ground_program
from manipulation.registration_validation import screen_registrations


def pose(x=0., y=0., z=0.):
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


def observations():
    common = {"depth": np.full((9, 9), 2., np.float32),
              "K": np.array([[10., 0., 4.], [0., 10., 4.], [0., 0., 1.]]),
              "owner_mask": np.ones((9, 9), bool)}
    source = {**deepcopy(common), "owner_id": "source_rigid", "T_camera_owner": pose()}
    current = {**deepcopy(common), "owner_id": "current_rigid", "reference_frame": "output",
               "T_output_camera": pose()}
    return source, {"sensor_left": current, "sensor_right": deepcopy(current)}


def screen(transforms, source=None, views=None, **kwargs):
    default_source, default_views = observations()
    return screen_registrations(
        default_source if source is None else source,
        default_views if views is None else views, transforms, reference_frame="output",
        same_rigid_geometry=kwargs.pop("same_rigid_geometry", True),
        geometry_source="controlled unchanged rigid surface geometry", **kwargs)


def compiled(kind):
    """A small compiled skill with actual source IDs and unchanged events."""
    from test_manipulation_program import fixture
    program, scene = fixture()
    if kind == "grasp":
        demo = {"carrier_id": "source_rigid", "tool_id": "parallel_gripper", "tool_frame": "eef_site",
                "frame_indices": [3, 4, 5], "timestamps": [0., .1, .2],
                "T_carrier_eef": [pose(z=.1), pose(z=.05), pose()],
                "protected_start_frame": 4,
                "gripper_events": [{"event_id": "close", "frame_index": 4, "command": "close"}]}
        return {"skill_kind": "grasp", "demo": demo}, {
            "reference_frame": "output", "carrier_id": "current_rigid",
            "carrier_pose": pose(), "region_transform": pose(), "eef_pose": pose(z=.1),
            "tool_id": "parallel_gripper", "tool_frame": "eef_site"}
    program["demo"].update(carrier_id="source_payload", reference_id="source_rigid")
    scene.update(reference_frame="output", reference_id="current_rigid")
    scene["functional_bindings"]["effect"]["reference_id"] = "current_rigid"
    return program, scene


class RegistrationValidationTest(unittest.TestCase):
    def test_free_space_rejects_but_occlusion_remains_unknown(self):
        result = screen({"near": pose(z=-.5), "same": pose(), "behind": pose(z=.5)})
        self.assertEqual(result["rejected_candidate_ids"], ["near"])
        self.assertEqual(result["supported_candidate_ids"], ["same"])
        self.assertEqual(result["unresolved_candidate_ids"], ["behind"])
        self.assertEqual(result["status"], "unresolved")
        self.assertIsNone(result["selected_candidate_id"])
        for row in result["candidates"]["behind"]["views"].values():
            self.assertEqual(row["summary"]["free_space_conflict_rays"], 0)
            self.assertGreater(row["summary"]["occluded_rays"], 0)

    def test_unique_supported_selected_and_ties_kept(self):
        result = screen({"pose_one": pose(), "pose_two": pose(z=-.5)})
        self.assertEqual(result["selected_candidate_id"], "pose_one")
        tied = screen({"pose_one": pose(), "pose_two": pose()})
        self.assertEqual(tied["status"], "ambiguous")
        self.assertIsNone(tied["selected_candidate_id"])
        rejected = screen({"near": pose(z=-.5)})
        self.assertEqual(rejected["status"], "rejected")

    # BEGIN partial visible-owner support must not invent complete mask coverage.
    def test_partial_owner_evidence_supports_only_observed_membership(self):
        source, views = observations()
        for view in views.values():
            view["owner_mask"][:] = False
            view["owner_mask"][2:7, 2:7] = True
        before = {name: view["owner_mask"].copy() for name, view in views.items()}
        result = screen({"matching": pose(), "in_free_space": pose(z=-.5)}, source, views)
        self.assertEqual(result["selected_candidate_id"], "matching")
        for name, view in views.items():
            matching = result["candidates"]["matching"]["views"][name]
            self.assertTrue(matching["summary"]["positive_support_noncollinear"])
            self.assertGreater(matching["summary"]["unknown_rays"], 0)
            self.assertEqual(matching["summary"]["free_space_conflict_rays"], 0)
            x, y = matching["rays"]["pixels"].T
            outside = ~view["owner_mask"][y, x]
            self.assertTrue(outside.any())
            self.assertTrue(np.all(matching["rays"]["labels"][outside] == 0))
            # Partial membership restricts positive support, not usable depth:
            # a surface in measured free space remains contradictory even on
            # rays outside the selected local evidence mask.
            wrong = result["candidates"]["in_free_space"]["views"][name]
            x, y = wrong["rays"]["pixels"].T
            self.assertTrue(np.any((wrong["rays"]["labels"] == 2) & ~view["owner_mask"][y, x]))
            np.testing.assert_array_equal(view["owner_mask"], before[name])
    # END no whole-owner completion or mask-based free-space invention.

    def test_geometry_applicability_is_explicit_and_never_bypassed(self):
        result = screen({"apparently_matching": pose()}, same_rigid_geometry=False)
        self.assertEqual(result["status"], "not_applicable")
        self.assertIsNone(result["selected_candidate_id"])
        self.assertEqual(result["candidates"], {})
        with self.assertRaises(ValueError):
            screen({"pose": pose()}, same_rigid_geometry="true")

    def test_invalid_depth_empty_masks_and_no_views_do_not_select(self):
        for mode in ("source_invalid", "source_empty", "current_invalid", "current_empty"):
            source, views = observations()
            if mode == "source_invalid":
                source["depth"][:] = np.nan
            elif mode == "source_empty":
                source["owner_mask"][:] = False
            elif mode == "current_invalid":
                views["sensor_right"]["depth"][:] = 0
            else:
                views["sensor_right"]["owner_mask"][:] = False
            with self.subTest(mode=mode):
                result = screen({"pose": pose()}, source, views)
                self.assertEqual(result["status"], "unresolved")
                self.assertIsNone(result["selected_candidate_id"])
        with self.assertRaises(ValueError):
            screen({"pose": pose()}, views={})

    def test_nonowner_support_and_discontinuities_never_certify(self):
        source, views = observations()
        source["owner_mask"][:] = False
        source["owner_mask"][3:6, 3:6] = True  # One interior source ray.
        for depth in (2.2, np.nan, 0.):
            for view in views.values():
                view["depth"][4, 5] = depth
            result = screen({"near": pose(z=-.5)}, source, views)
            self.assertEqual(result["status"], "unresolved")
            self.assertTrue(all(row["summary"]["depth_boundary_or_invalid_rays"] == 1
                                for row in result["candidates"]["near"]["views"].values()))

    def test_world_and_source_gauge_changes_preserve_evidence(self):
        source, views = observations()
        expected = screen({"identity": pose(), "near": pose(z=-.5)}, source, views)
        world = np.array([[0., -1., 0., 3.], [1., 0., 0., -2.],
                          [0., 0., 1., .3], [0., 0., 0., 1.]])
        source_change = pose(x=.1, y=-.2)
        source["T_camera_owner"] = source["T_camera_owner"] @ source_change
        for view in views.values():
            view["T_output_camera"] = world @ view["T_output_camera"]
        actual = screen({"identity": world @ source_change,
                         "near": world @ pose(z=-.5) @ source_change}, source, views)
        self.assertEqual(actual["selected_candidate_id"], expected["selected_candidate_id"])
        for candidate in expected["candidates"]:
            for view in views:
                a = actual["candidates"][candidate]["views"][view]
                b = expected["candidates"][candidate]["views"][view]
                np.testing.assert_array_equal(a["rays"]["labels"], b["rays"]["labels"])
                np.testing.assert_allclose(a["rays"]["predicted_depth_m"], b["rays"]["predicted_depth_m"])

    def test_zbuffer_has_one_vote_per_pixel(self):
        source, views = observations()
        for view in views.values():
            view["K"][:2, :2] *= .25
        result = screen({"near": pose(z=-.5)}, source, views)
        row = result["candidates"]["near"]["views"]["sensor_left"]
        self.assertLess(row["summary"]["unique_projected_rays"], row["summary"]["projected_source_samples"])
        self.assertEqual(row["summary"]["free_space_conflict_rays"], len(row["rays"]["pixels"]))

    def test_names_counts_view_order_and_candidate_order_do_not_choose(self):
        source, views = observations()
        rows = [("third_name", pose()), ("first_name", pose(z=-.5)), ("another", pose(z=-.7))]
        for order in itertools.permutations(rows):
            for camera_order in (views, dict(reversed(list(views.items())))):
                self.assertEqual(screen(dict(order), source, camera_order)["selected_candidate_id"], "third_name")
        renamed = screen({"renamed": pose(), "old_good_label": pose(z=-.5)})
        self.assertEqual(renamed["selected_candidate_id"], "renamed")
        single = screen({"only": pose()}, source, {"one": views["sensor_left"]})
        self.assertEqual(single["selected_candidate_id"], "only")

    def test_malformed_frames_camera_masks_poses_and_gates_rejected(self):
        for gate in (0., -1., np.inf, np.nan, True):
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                screen({"pose": pose()}, max_error=gate)
        for field in ("reference_frame", "owner_id"):
            source, views = observations()
            views["sensor_right"][field] = "different"
            with self.assertRaises(ValueError):
                screen({"pose": pose()}, source, views)
        for field, bad in (("K", np.zeros((3, 3))), ("owner_mask", np.ones((9, 9), int)),
                           ("T_output_camera", np.diag([-1., 1., 1., 1.]))):
            source, views = observations()
            views["sensor_left"][field] = bad
            with self.assertRaises(ValueError):
                screen({"pose": pose()}, source, views)
        with self.assertRaises(ValueError):
            screen({"reflected": np.diag([-1., 1., 1., 1.])})

    def test_grounding_derives_tested_transform_from_actual_grasp_binding(self):
        source, views = observations()
        program, scene = compiled("grasp")
        incorrect = deepcopy(scene)
        incorrect["region_transform"] = pose(z=-.5)
        original = deepcopy(scene)
        expected = ground_program(program, scene)
        result = ground_registered_candidates(
            program, {"good": scene, "bad": incorrect}, source, views, role="carrier",
            same_rigid_geometry=True, geometry_source="same rigid carrier")
        self.assertEqual(result["selected_candidate_id"], "good")
        self.assertEqual(set(result["trajectories"]), {"good"})
        trajectory = result["selected_trajectory"]
        np.testing.assert_allclose(trajectory["eef_poses"], expected["eef_poses"])
        np.testing.assert_array_equal(trajectory["frame_indices"], [3, 4, 5])
        self.assertEqual(trajectory["gripper_events"], expected["gripper_events"])
        self.assertEqual(trajectory["holding_status"], "requires_verification")
        np.testing.assert_array_equal(scene["region_transform"], original["region_transform"])
        self.assertNotIn("registration_validation", scene)

    def test_manipulation_gauge_release_and_controlled_suffix_are_preserved(self):
        source, views = observations()
        program, scene = compiled("manipulation")
        # Same output-from-source mapping, represented in a different current
        # reference gauge. Screening must consume the same product as the core.
        gauge = pose(x=.7, y=-.2)
        scene["reference_pose"] = gauge
        scene["functional_bindings"]["effect"]["reference_frame"] = np.linalg.inv(gauge)
        wrong = deepcopy(scene)
        wrong["functional_bindings"]["effect"]["reference_frame"] = np.linalg.inv(gauge) @ pose(z=-.5)
        expected = ground_program(program, scene)
        result = ground_registered_candidates(
            program, {"good": scene, "bad": wrong}, source, views, role="reference",
            same_rigid_geometry=True, geometry_source="same rigid reference")
        self.assertEqual(result["selected_candidate_id"], "good")
        actual = result["selected_trajectory"]
        for key in ("eef_poses", "object_poses", "frame_indices", "timestamps", "object_frame_indices"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)
        self.assertEqual(actual["release_frame"], expected["release_frame"])
        self.assertEqual(actual["gripper_events"], expected["gripper_events"])
        self.assertFalse(result["robot_execution_verified"])

    def test_wrapper_does_not_upgrade_uncertain_or_inapplicable_geometry(self):
        source, views = observations()
        program, scene = compiled("grasp")
        hidden = deepcopy(scene)
        hidden["region_transform"] = pose(z=.5)
        result = ground_registered_candidates(
            program, {"seen": scene, "hidden": hidden}, source, views, role="carrier",
            same_rigid_geometry=True, geometry_source="same geometry")
        self.assertEqual(result["status"], "unresolved")
        self.assertIsNone(result["selected_trajectory"])
        self.assertEqual(result["trajectories"]["hidden"]["registration_validation"]["status"], "unresolved")
        result = ground_registered_candidates(
            program, {"seen": scene}, source, views, role="carrier",
            same_rigid_geometry=False, geometry_source="new shape has not been related to source geometry")
        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["trajectories"], {})

    def test_mismatched_owner_and_multiple_inconsistent_bindings_are_rejected(self):
        source, views = observations()
        program, scene = compiled("manipulation")
        with self.assertRaises(ValueError):
            ground_registered_candidates(
                program, {"one": scene}, {**source, "owner_id": "wrong"}, views,
                role="reference", same_rigid_geometry=True, geometry_source="same geometry")
        event = deepcopy(program["demo"]["functional_events"][0])
        event.update(event_id="second", frame_index=104)
        program["demo"]["functional_events"].append(event)
        binding = deepcopy(scene["functional_bindings"]["effect"])
        binding["reference_frame"] = pose(x=.5)
        scene["functional_bindings"]["second"] = binding
        with self.assertRaisesRegex(ValueError, "one rigid"):
            ground_registered_candidates(
                program, {"one": scene}, source, views, role="reference",
                same_rigid_geometry=True, geometry_source="same geometry")


if __name__ == "__main__":
    unittest.main()
# END same-geometry projective validation and real-consumer regressions.
