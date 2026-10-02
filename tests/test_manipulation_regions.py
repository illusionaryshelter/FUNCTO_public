"""CPU region consumers: geometry invariants and actual masked correspondence.

Synthetic images encode known correspondences; no model, GPU, HTTP request or
dataset-specific selector is used. These tests establish coordinate and input
contracts, not the quality of predicted SAM regions or dense descriptors.
"""

# BEGIN interaction-region and functional-frame regression tests.
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from manipulation.goals import SupportSurface, check_on
from manipulation.regions import (SupportReferenceNotFound, align_axis, bind_observations, bind_region,
                                  check_region_on, function_frame, ground_on,
                                  ground_relation, match_functional_region,
                                  relation_error, sample_region_pixels,
                                  source_function_geometry)


def pose(angles=(0., 0., 0.), translation=(0., 0., 0.)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    value[:3, 3] = translation
    return value


def transform(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


class FunctionFramesTest(unittest.TestCase):
    def test_frame_preserves_signed_function_axis_and_world_equivariance(self):
        origin, axis, normal = np.array([.1, -.3, .8]), np.array([2., 0., 0.]), np.array([0., 0., 7.])
        frame = function_frame(origin, axis, normal)
        np.testing.assert_array_equal(frame, pose(translation=origin))
        world = pose((.3, -.8, .7), (.4, .2, -.1))
        changed = function_frame(transform(world, origin), world[:3, :3] @ axis,
                                 world[:3, :3] @ normal)
        np.testing.assert_allclose(changed, world @ frame, atol=1e-14)
        # The released code's opposite x/y is a fixed local gauge, not a
        # data-dependent correction to either the current or demonstrated pose.
        old_gauge = np.diag([-1., -1., 1., 1.])
        old = function_frame(origin, -axis, normal)
        np.testing.assert_array_equal(old, frame @ old_gauge)

    def test_missing_parallel_and_nonfinite_directions_are_not_completed(self):
        for axis, normal in (([0, 0, 0], [0, 0, 1]), ([1, 0, 0], [1, 0, 0]),
                             ([1, 0, 0], [1, 0, 1]), ([np.nan, 0, 0], [0, 0, 1])):
            with self.subTest(axis=axis, normal=normal), self.assertRaises(ValueError):
                function_frame([0, 0, 0], axis, normal)

    def test_single_axis_alignment_retains_preferred_twist_and_requires_pi_branch(self):
        preferred = pose((.4, -.2, .6))[:3, :3]
        local = np.array([0., 0., 1.])
        source = preferred @ local
        np.testing.assert_allclose(align_axis(preferred, local, source), preferred, atol=1e-14)
        target = np.array([.2, -.4, .7]); target /= np.linalg.norm(target)
        aligned = align_axis(preferred, local, target)
        np.testing.assert_allclose(aligned @ local, target, atol=1e-14)
        # The minimal correction rotates source toward target about their cross
        # product. Geodesic length must equal the angle between the two axes.
        angle = Rotation.from_matrix(aligned @ preferred.T).magnitude()
        self.assertAlmostEqual(angle, np.arccos(source @ target), places=13)
        with self.assertRaisesRegex(ValueError, "antiparallel_axis"):
            align_axis(preferred, local, -source)
        branch = preferred[:, 0]
        flipped = align_axis(preferred, local, -source, antiparallel_axis=branch)
        np.testing.assert_allclose(flipped @ local, -source, atol=1e-14)
        world = pose((-.8, .4, .1))[:3, :3]
        changed = align_axis(world @ preferred, local, -world @ source,
                             antiparallel_axis=world @ branch)
        np.testing.assert_allclose(changed, world @ flipped, atol=1e-14)
        with self.assertRaisesRegex(ValueError, "perpendicular"):
            align_axis(preferred, local, -source, antiparallel_axis=source)

    def test_two_sided_function_goal_tracks_different_carrier_and_reference_geometry(self):
        demo = pose(translation=[0., 0., .2])
        cd = pose(translation=[0., 0., -.2])
        cn = pose(translation=[0., 0., -.35])
        rd = pose()
        rn = pose(translation=[.7, -.3, .8])
        result = ground_relation(demo, cd, cn, rd, rn)
        np.testing.assert_allclose(result["goal_pose"][:3, 3], [.7, -.3, 1.15], atol=1e-14)
        expected = result["reference_binding"] @ demo @ np.linalg.inv(result["carrier_binding"])
        np.testing.assert_allclose(result["goal_pose"], expected, atol=1e-14)
        errors = relation_error(result["goal_pose"], cn, rn, result["functional_relation"])
        self.assertLess(errors["position_error_m"], 1e-14)
        self.assertLess(errors["rotation_error_rad"], 1e-14)
        # A reference-only transfer leaves the wrong carrier function surface.
        bad = result["reference_binding"] @ demo
        self.assertAlmostEqual(relation_error(bad, cn, rn, result["functional_relation"])["position_error_m"], .15)

    def test_owner_coordinate_changes_do_not_change_the_physical_function_goal(self):
        demo = pose((.6, -.2, .1), (.7, -.3, .5))
        cd, cn = pose((.1, .2, -.7), (.2, .1, .3)), pose((-.2, .5, .4), (-.1, .3, .4))
        rd, rn = pose((.4, .1, -.3), (.2, -.1, .6)), pose((-.5, .2, -.1), (-.4, .2, .3))
        original = ground_relation(demo, cd, cn, rd, rn)
        hcd = pose((.5, -.1, .4), (.3, .2, -.4))
        hcn = pose((-.6, .2, .1), (-.1, .3, -.2))
        hrd = pose((.2, .8, -.2), (-.2, -.1, .1))
        hrn = pose((-.2, .3, .7), (.4, .1, -.2))
        changed = ground_relation(np.linalg.inv(hrd) @ demo @ hcd,
                                  np.linalg.inv(hcd) @ cd, np.linalg.inv(hcn) @ cn,
                                  np.linalg.inv(hrd) @ rd, np.linalg.inv(hrn) @ rn)
        np.testing.assert_allclose(changed["functional_relation"], original["functional_relation"], atol=1e-14)
        np.testing.assert_allclose(changed["goal_pose"], np.linalg.inv(hrn) @ original["goal_pose"] @ hcn, atol=1e-14)

    def test_planar_correspondences_bind_but_collinear_correspondences_reject(self):
        source = np.array([[0, 0, 0], [.2, 0, 0], [0, .3, 0], [.1, .1, 0]], float)
        expected = pose((.5, -.6, .3), (.8, -.2, .4))
        result = bind_region(source, transform(expected, source), max_error=1e-6)
        np.testing.assert_allclose(result["transform"], expected, atol=1e-12)
        self.assertEqual(result["diagnostics"]["num_inliers"], 4)
        line = np.column_stack((np.arange(5, dtype=float), np.zeros((5, 2))))
        with self.assertRaisesRegex(ValueError, "noncollinear"):
            bind_region(line, transform(expected, line))


class RegionObservationTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100., 0., 0.], [0., 100., 0.], [0., 0., 1.]])
        self.depth = np.ones((16, 16))

    def test_local_lift_keeps_pixel_identity_and_nonidentity_owner_pose(self):
        mask = np.zeros((16, 16), bool); mask[3:7, 4:9] = True
        depth = self.depth.copy(); depth[3, 4] = np.nan; depth[4, 4] = 0
        owner = pose((.4, .2, -.3), (.3, -.1, .6))
        result = bind_observations(mask, depth, self.K, owner)
        expected_uv = np.array([[x, y] for y in range(3, 7) for x in range(4, 9)
                                if (y, x) not in {(3, 4), (4, 4)}])
        np.testing.assert_array_equal(result["pixels"], expected_uv)
        expected_camera = np.column_stack((expected_uv / 100., np.ones(len(expected_uv))))
        np.testing.assert_allclose(result["camera_points"], expected_camera, atol=1e-14)
        np.testing.assert_allclose(transform(owner, result["owner_local_points"]), expected_camera, atol=1e-14)
        self.assertTrue(np.isnan(depth[3, 4]))
        self.assertEqual(mask.sum(), 20)

    def test_sampling_never_expands_mask_and_has_bounded_queries(self):
        mask = np.zeros((16, 16), bool); mask[3:14, 2:13] = True
        depth = self.depth.copy(); depth[8, 8] = np.nan
        pixels, valid = sample_region_pixels(mask, depth, count=24)
        self.assertEqual(len(pixels), 24)
        x, y = pixels.astype(int).T
        self.assertTrue(mask[y, x].all())
        self.assertTrue(valid[y, x].all())
        self.assertFalse(np.any(valid & ~mask))
        self.assertFalse(valid[8, 8])
        for invalid in (np.zeros_like(mask), np.ones_like(mask, dtype=float)):
            with self.assertRaises(ValueError):
                sample_region_pixels(invalid, self.depth)


class FunctionalQueryGroundingTest(unittest.TestCase):
    """Run the real coarse selector, matcher and frame recipe on CPU evidence.

    Mocked VLM cell choices and synthetic descriptors stand in for inference.
    No target part, target frame, target query or rigid registration is injected
    into the production interface. The four source queries are grouped geometry
    witnesses, deliberately differing from the former three semantic landmarks.
    """

    def observations(self):
        source_pixels = np.array([[4, 5], [7, 5], [4, 8], [7, 8]])
        target_pixels = np.array([[6, 4], [12, 4], [6, 13], [12, 13]])
        images = []
        for pixels in (source_pixels, target_pixels):
            rgb = np.zeros((16, 16, 3), np.uint8)
            for label, (x, y) in enumerate(pixels, 1):
                rgb[y, x, 0] = label
            images.append(Image.fromarray(rgb))
        common = {"owner_mask": np.ones((16, 16), bool), "depth": np.ones((16, 16)),
                  "K": np.diag([100., 100., 1.])}
        interaction = np.zeros((16, 16), bool); interaction[4:7, 3:6] = True
        demo = dict(deepcopy(common), image=images[0], mask=interaction,
                    T_camera_owner=pose((.3, -.1, .2), (.4, .1, -.2)),
                    region_id="source_surface", owner_id="source_instance",
                    description="Local interaction at the first visible edge",
                    query_pixels=source_pixels,
                    query_descriptions=["left near edge", "right near edge",
                                        "left far edge", "right far edge"],
                    frame_recipe={"kind": "frame", "origin_weights": [.25]*4,
                                  "axis_weights": [-.5, .5, -.5, .5],
                                  "plane_weights": [-.5, -.5, .5, .5],
                                  "source": "Grouped left-to-right functional direction and ordered near/far plane evidence"})
        current = dict(deepcopy(common), image=images[1], owner_id="current_instance",
                       T_camera_owner=pose((-.2, .5, -.3), (-.2, .3, .1)))
        return demo, current, target_pixels

    @staticmethod
    def descriptors(first, second):
        # Exact one-hot directions encode query identity; the actual masked
        # matcher still selects pixels, resolves cosine scores and lifts depth.
        features = []
        for image in (first, second):
            labels = np.asarray(image)[:, :, 0]
            features.append(np.eye(5, dtype=np.float32)[labels].transpose(2, 0, 1)[None])
        return features

    @staticmethod
    def client_for(pixels, size=(16, 16)):
        # Only the semantic API result is controlled. Production grid creation,
        # validation and box expansion remain active in every ordinary test.
        edges = [np.linspace(0, length, 6, dtype=int) for length in size]
        cells = [int(5*(np.searchsorted(edges[1], y, side="right")-1)
                     + np.searchsorted(edges[0], x, side="right")) for x, y in pixels]
        return Mock(chat_json=Mock(return_value={"cells": cells}))

    def run_match(self, demo, current, target_pixels, *, extractor=None):
        return match_functional_region(demo, current, client=self.client_for(target_pixels),
                                       extractor=self.descriptors if extractor is None else extractor)

    def test_current_owner_is_sufficient_and_noncongruent_geometry_is_reconstructed(self):
        demo, current, target_pixels = self.observations()
        before_source, before_current = deepcopy(demo), deepcopy(current)
        extractor, client = Mock(side_effect=self.descriptors), self.client_for(target_pixels)
        with patch("manipulation.regions.rigid_registration", side_effect=AssertionError("not a rigid fit")):
            result = match_functional_region(demo, current, client=client, extractor=extractor)
        self.assertNotIn("mask", current)
        self.assertNotIn("region_id", current)
        self.assertEqual(extractor.call_count, 1)
        self.assertEqual(client.chat_json.call_count, 1)
        self.assertEqual(tuple(image.size for image in extractor.call_args.args), ((16, 16), (16, 16)))
        diagnostic = result["diagnostics"]
        np.testing.assert_array_equal(diagnostic["target_pixels"], target_pixels)
        np.testing.assert_array_equal(diagnostic["interaction_query_indices"], [0])
        np.testing.assert_array_equal(diagnostic["queries_outside_interaction_mask"], [1, 2, 3])
        self.assertEqual(diagnostic["descriptor_pair_extractions"], 1)
        for description in demo["query_descriptions"]:
            self.assertIn(description, client.chat_json.call_args.args[0])
        np.testing.assert_allclose(result["transform"] @ result["source_frame"],
                                   result["target_frame"], atol=1e-14)
        # Different x/y stretches cannot be fitted by one rigid mapping. They
        # can still define the same functional axes at a new meaningful origin.
        self.assertGreater(np.linalg.norm(transform(result["transform"], result["source_points"])
                                          - result["target_points"]), .05)
        source_camera = np.column_stack((demo["query_pixels"]/100., np.ones(4)))
        target_camera = np.column_stack((target_pixels/100., np.ones(4)))
        np.testing.assert_allclose(transform(demo["T_camera_owner"], result["source_points"]), source_camera)
        np.testing.assert_allclose(transform(current["T_camera_owner"], result["target_points"]), target_camera)
        for actual, before in ((demo, before_source), (current, before_current)):
            for name in ("image", "owner_mask", "depth", "K", "T_camera_owner"):
                np.testing.assert_array_equal(np.asarray(actual[name]), np.asarray(before[name]))

    def test_owner_coordinate_gauges_preserve_physical_geometry(self):
        demo, current, pixels = self.observations()
        base = self.run_match(demo, current, pixels)
        hd, hn = pose((.5, -.4, .1), (.2, .5, -.2)), pose((-.3, .7, -.2), (-.2, .3, .4))
        demo["T_camera_owner"] = demo["T_camera_owner"] @ np.linalg.inv(hd)
        current["T_camera_owner"] = current["T_camera_owner"] @ np.linalg.inv(hn)
        changed = self.run_match(demo, current, pixels)
        np.testing.assert_allclose(changed["source_frame"], hd @ base["source_frame"], atol=1e-14)
        np.testing.assert_allclose(changed["target_frame"], hn @ base["target_frame"], atol=1e-14)
        np.testing.assert_allclose(changed["transform"], hn @ base["transform"] @ np.linalg.inv(hd), atol=1e-14)

    def test_cropped_nonsquare_owner_context_restores_full_image_pixels(self):
        demo, current, pixels = self.observations()
        # Identical padded canvases have a 16x18 owner-context crop at (4,3).
        # match_points independently square-pads that crop; its descriptor
        # coordinates and regions' camera coordinates must both stay correct.
        offsets = (np.array([4, 3]), np.array([7, 5]))
        for record, offset in zip((demo, current), offsets):
            rgb = np.zeros((28, 30, 3), np.uint8)
            x, y = offset
            rgb[y:y+16, x:x+16] = np.asarray(record["image"])
            record["image"] = Image.fromarray(rgb)
            record["owner_mask"] = np.zeros((28, 30), bool)
            record["owner_mask"][y+2:y+14, x+2:x+14] = True
            record["depth"] = np.ones((28, 30))
        # Extend the target owner vertically to include its low two queries;
        # this also exercises the unequal crop-square sizes used by the matcher.
        current["owner_mask"][7:21, 9:21] = True
        # BEGIN crop optional part evidence in lockstep with its owner context.
        current_part = np.zeros_like(current["owner_mask"])
        current_part[9:11, 13:15] = True
        current_part[0, 0] = True  # Off-crop spill remains in the final evidence.
        current["interaction_evidence"] = {
            "mask": current_part, "owner_id": current["owner_id"],
            "description": "visible first working edge", "source": "observed semantic selection"}
        # END no target queries or functional geometry are supplied.
        shifted_region = np.zeros((28, 30), bool)
        shifted_region[7:10, 7:10] = demo["mask"][4:7, 3:6]
        demo["mask"] = shifted_region
        demo["query_pixels"] = demo["query_pixels"] + offsets[0]
        boxes = np.tile([0, 0, 16, 18], (4, 1))
        with patch("manipulation.regions.propose_query_regions", return_value={"boxes": boxes, "diagnostics": {}}) as proposal:
            result = match_functional_region(demo, current, extractor=self.descriptors)
        np.testing.assert_array_equal(result["diagnostics"]["target_pixels"], pixels+offsets[1])
        np.testing.assert_array_equal(result["diagnostics"]["crop_offsets_xy"], offsets)
        np.testing.assert_array_equal(proposal.call_args.args[5], demo["query_pixels"]-offsets[0])
        self.assertEqual(proposal.call_args.args[0].size, (16, 16))
        self.assertEqual(proposal.call_args.args[1].size, (16, 18))
        # BEGIN native sparse mask and crop coordinates preserve every raw pixel.
        supplied = proposal.call_args.kwargs["current_interaction_evidence"]
        np.testing.assert_array_equal(supplied["mask"], current_part[5:23, 7:23])
        retained = result["diagnostics"]["current_interaction_evidence"]
        self.assertEqual(retained["mask_pixel_count"], 5)
        self.assertEqual(retained["pixels_outside_owner"], 1)
        self.assertEqual(retained["query_inside_mask"], [True, False, False, False])
        restored = np.zeros_like(current_part)
        xy = retained["mask_pixels_xy"]
        restored[xy[:, 1], xy[:, 0]] = True
        np.testing.assert_array_equal(restored, current_part)
        # END omitted spill in the display crop is not deleted from native evidence.

    def test_plane_basis_marks_free_tangent_and_retains_explicit_normal_role(self):
        demo, current, pixels = self.observations()
        demo["frame_recipe"]["kind"] = "plane"
        demo["frame_recipe"]["source"] = "Ordered surface samples; offline evidence establishes outward negative camera z"
        demo["normal_role"] = "outward"
        result = self.run_match(demo, current, pixels)
        self.assertEqual(result["geometry_kind"], "plane")
        self.assertEqual(result["normal_role"], "outward")
        self.assertIn("preference", result["diagnostics"]["x_axis_source"])
        self.assertIn("not an observed", result["diagnostics"]["target_geometry"]["x_axis_source"])
        self.assertIn("source evidence", result["diagnostics"]["normal_source"])
        np.testing.assert_allclose(current["T_camera_owner"][:3, :3] @ result["target_normal"], [0, 0, -1], atol=1e-14)

    def test_source_spill_is_recorded_without_expanding_or_clipping_interaction(self):
        demo, current, pixels = self.observations()
        demo["owner_mask"][4, 3] = False
        before = demo["mask"].copy()
        result = self.run_match(demo, current, pixels)
        self.assertEqual(result["diagnostics"]["interaction_pixels_outside_owner"], 1)
        np.testing.assert_array_equal(demo["mask"], before)
        np.testing.assert_array_equal(result["diagnostics"]["interaction_query_indices"], [0])

    def test_invalid_source_recipe_or_evidence_fails_before_any_inference(self):
        demo, current, _ = self.observations()
        invalid_recipes = [
            ("origin_weights", [0., 0., 0., 0.]),
            ("axis_weights", [1., 0., 0., 0.]),
            ("plane_weights", [0., 0., 0.]),
            ("axis_weights", [np.nan, 0., 0., 0.]),
            ("axis_weights", [0., 0., 0., 0.]),
            ("plane_weights", [-.5, .5, -.5, .5]),
            ("kind", "axis"), ("source", ""),
        ]
        client, extractor = Mock(), Mock()
        for key, value in invalid_recipes:
            invalid = deepcopy(demo); invalid["frame_recipe"][key] = value
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                match_functional_region(invalid, current, client=client, extractor=extractor)
        for edit in ("missing recipe", "query outside owner", "query missing depth", "region disjoint", "normal role on full frame"):
            invalid = deepcopy(demo)
            if edit == "missing recipe":
                del invalid["frame_recipe"]
            elif edit == "query outside owner":
                invalid["owner_mask"][5, 4] = False
            elif edit == "query missing depth":
                invalid["depth"][5, 4] = np.nan
            elif edit == "region disjoint":
                invalid["owner_mask"][invalid["mask"]] = False
            else:
                invalid["normal_role"] = "outward"
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                match_functional_region(invalid, current, client=client, extractor=extractor)
        client.chat_json.assert_not_called()
        extractor.assert_not_called()

    def test_unresolved_semantics_invalid_current_calibration_or_depth_do_not_guess(self):
        demo, current, pixels = self.observations()
        extractor = Mock()
        with self.assertRaisesRegex(ValueError, "Unresolved functional correspondence"):
            match_functional_region(demo, current,
                                    client=Mock(chat_json=Mock(return_value={"cells": [None]*4})),
                                    extractor=extractor)
        for field, value in (("K", np.zeros((3, 3))), ("depth", np.zeros((16, 16))),
                             ("owner_mask", np.zeros((16, 16), bool))):
            invalid = deepcopy(current); invalid[field] = value
            client = self.client_for(pixels)
            with self.subTest(field=field), self.assertRaises(ValueError):
                match_functional_region(demo, invalid, client=client, extractor=extractor)
            client.chat_json.assert_not_called()
        extractor.assert_not_called()

    def test_candidate_boxes_intersect_owner_and_depth(self):
        demo, current, target = self.observations()
        # Better-looking copies outside the owner or at invalid depth would win
        # row-major matching if either required observation constraint vanished.
        rgb = np.array(current["image"], copy=True)
        for label, (x, y) in enumerate(target, 1):
            rgb[y-1, x, 0] = label
            current["owner_mask"][y-1, x] = False
            rgb[y, x-1, 0] = label
            current["depth"][y, x-1] = 0
        current["image"] = Image.fromarray(rgb)
        result = self.run_match(demo, current, target)
        np.testing.assert_array_equal(result["diagnostics"]["target_pixels"], target)
        empty_scope = np.tile([0, 0, 2, 2], (4, 1))
        current["depth"][:2, :2] = np.nan
        extractor = Mock()
        with patch("manipulation.regions.propose_query_regions", return_value={"boxes": empty_scope, "diagnostics": {}}):
            with self.assertRaisesRegex(ValueError, "no valid current owner depth"):
                match_functional_region(demo, current, extractor=extractor)
        extractor.assert_not_called()

    def test_repeated_or_nonpositive_matches_are_not_replaced_by_invented_geometry(self):
        demo, current, _ = self.observations()
        boxes = np.tile([0, 0, 16, 16], (4, 1))
        def constant(first, second):
            return [np.ones((1, 2, image.height, image.width), np.float32) for image in (first, second)]
        with patch("manipulation.regions.propose_query_regions", return_value={"boxes": boxes, "diagnostics": {}}):
            with self.assertRaisesRegex(ValueError, "nonzero|collinear"):
                match_functional_region(demo, current, extractor=constant)
            def opposite(first, second):
                pair = constant(first, second)
                pair[1] *= -1
                return pair
            with self.assertRaisesRegex(ValueError, "positive-score descriptor"):
                match_functional_region(demo, current, extractor=opposite)

    def test_source_geometry_is_independent_of_visual_inference(self):
        demo, _, _ = self.observations()
        with patch("manipulation.regions.propose_query_regions", side_effect=AssertionError("offline lifting only")):
            result = source_function_geometry(demo)
        self.assertEqual(result["geometry_kind"], "frame")
        np.testing.assert_allclose(np.linalg.det(result["source_frame"][:3, :3]), 1., atol=1e-14)


class SupportRegionTest(unittest.TestCase):
    def setUp(self):
        self.body = np.array([[x, y, z] for x in (-.05, .05) for y in (-.05, .05)
                              for z in (0., .2)])
        self.bottom = self.body[self.body[:, 2] == 0.]
        self.rim = self.body[self.body[:, 2] == .2]
        self.surface = SupportSurface(np.array([.7, -.3, .5]), np.eye(3)[:, :2],
                                      np.array([0., 0., 1.]),
                                      np.array([[-.3, -.3], [-.3, .3], [.3, .3], [.3, -.3]]))
        self.tolerances = {"gap_tolerance": 1e-6, "normal_tolerance": 1e-6}

    def test_placement_aligns_functional_normal_and_checks_full_body_and_face(self):
        preferred = pose((.4, -.3, .6), (1.6, .8, 1.4))
        before = preferred.copy()
        result = ground_on(preferred, self.body, self.bottom, [0, 0, -1], self.surface,
                           margin=.01, **self.tolerances)
        goal, checks = result["goal_pose"], result["checks"]
        np.testing.assert_allclose(goal[:3, :3] @ [0., 0., -1.], -self.surface.normal, atol=1e-14)
        self.assertTrue(checks["region"]["valid"])
        self.assertLess(checks["region"]["max_abs_gap_m"], 1e-14)
        self.assertLess(checks["body"]["boundary_violation_m"], 1e-7)
        self.assertEqual(checks["body"]["checked_points"], 8)
        self.assertEqual(checks["region"]["checked_points"], 4)
        self.assertIn("chosen free component", result["freedom"]["twist"])
        np.testing.assert_array_equal(preferred, before)

    def test_whole_body_lowest_point_cannot_disguise_wrong_functional_face(self):
        upright = pose(translation=self.surface.origin)
        self.assertEqual(check_on(upright, self.body, self.surface)["support_gap_m"], 0.)
        wrong = check_region_on(upright, self.rim, [0, 0, -1], self.surface, **self.tolerances)
        self.assertFalse(wrong["valid"])
        self.assertAlmostEqual(wrong["max_abs_gap_m"], .2)
        with self.assertRaisesRegex(SupportReferenceNotFound, "Specified functional contact face"):
            ground_on(upright, self.body, self.rim, [0, 0, -1], self.surface, **self.tolerances)
        # A directed normal is checked, even when all face points lie on plane.
        wrong_normal = check_region_on(upright, self.bottom, [0, 0, 1], self.surface, **self.tolerances)
        self.assertFalse(wrong_normal["valid"])
        self.assertAlmostEqual(wrong_normal["normal_error_rad"], np.pi)

    def test_region_boundary_is_checked_independently_without_inventing_area(self):
        upright = pose(translation=self.surface.origin)
        outside = self.bottom.copy(); outside[0, 0] = .5
        result = check_region_on(upright, outside, [0, 0, -1], self.surface, **self.tolerances)
        self.assertFalse(result["valid"])
        self.assertAlmostEqual(result["boundary_violation_m"], .2)
        with self.assertRaisesRegex(ValueError, "boundary"):
            ground_on(upright, self.body, outside, [0, 0, -1], self.surface, **self.tolerances)

    def test_opposite_axis_placement_needs_geometric_branch_and_bad_tolerances_reject(self):
        upside_down = pose((0., np.pi, 0.), self.surface.origin)
        with self.assertRaisesRegex(SupportReferenceNotFound, "antiparallel_axis"):
            ground_on(upside_down, self.body, self.bottom, [0, 0, -1], self.surface, **self.tolerances)
        result = ground_on(upside_down, self.body, self.bottom, [0, 0, -1], self.surface,
                           antiparallel_axis=[1, 0, 0], **self.tolerances)
        self.assertTrue(result["checks"]["region"]["valid"])
        for field, value in (("gap_tolerance", -1), ("normal_tolerance", np.pi),
                             ("gap_tolerance", np.nan), ("margin", -.1)):
            kwargs = dict(self.tolerances, **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                ground_on(pose(), self.body, self.bottom, [0, 0, -1], self.surface, **kwargs)

    # BEGIN yaw remains a searched task freedom, independent of surface basis.
    def test_preferred_feasible_yaw_is_preserved_without_searching_other_yaws(self):
        preferred = pose((0., 0., .37), self.surface.origin)
        result = ground_on(preferred, self.body, self.bottom, [0, 0, -1], self.surface, **self.tolerances)
        np.testing.assert_allclose(result["goal_pose"], preferred, atol=1e-12)
        self.assertEqual(result["freedom"]["yaw_search"]["checked_candidates"], 1)
        self.assertEqual(result["freedom"]["yaw_search"]["chosen_offset_rad"], 0.)

    def test_preferred_footprint_failure_can_succeed_at_another_free_yaw(self):
        body = np.array([[x, y, z] for x in (-.14, .14) for y in (-.035, .035) for z in (0., .1)])
        bottom = body[body[:, 2] == 0]
        surface = SupportSurface(self.surface.origin, self.surface.basis, self.surface.normal,
                                 np.array([[-.05, -.2], [-.05, .2], [.05, .2], [.05, -.2]]))
        preferred = pose(translation=surface.origin)
        result = ground_on(preferred, body, bottom, [0, 0, -1], surface, **self.tolerances)
        self.assertTrue(result["checks"]["region"]["valid"])
        self.assertLess(result["checks"]["body"]["boundary_violation_m"], 1e-7)
        search = result["freedom"]["yaw_search"]
        self.assertGreater(abs(search["chosen_offset_rad"]), np.deg2rad(75))
        self.assertGreater(search["checked_candidates"], 1)
        self.assertIn("no global", search["scope"])
        # Re-express the same finite world patch by a reflected/rotated basis
        # and shifted origin. Neither operation changes the selected free yaw.
        rotation = Rotation.from_euler("z", .63).as_matrix()[:2, :2]
        rotation[:, 1] *= -1
        shift = np.array([.035, -.025])
        reparameterized = SupportSurface(
            surface.origin + surface.basis @ shift, surface.basis @ rotation, surface.normal,
            ((surface.boundary - shift) @ rotation)[::-1])
        equivalent = ground_on(preferred, body[::-1], bottom[::-1], [0, 0, -1],
                               reparameterized, **self.tolerances)
        np.testing.assert_allclose(equivalent["goal_pose"], result["goal_pose"], atol=1e-8)
        # Common SE(3) transforms act on the normal and preference, not on a
        # fixed world axis. The whole selected placement must transform too.
        world = pose((.4, -.6, .2), (.2, -.5, .3))
        moved = SupportSurface(transform(world, surface.origin[None])[0],
                               world[:3, :3] @ surface.basis,
                               world[:3, :3] @ surface.normal, surface.boundary)
        changed = ground_on(world @ preferred, body, bottom, [0, 0, -1], moved, **self.tolerances)
        np.testing.assert_allclose(changed["goal_pose"], world @ result["goal_pose"], atol=1e-8)

    def test_exhausted_yaw_samples_do_not_claim_global_infeasibility(self):
        body = np.array([[x, y, z] for x in (-.4, .4) for y in (-.4, .4) for z in (0., .1)])
        with self.assertRaisesRegex(ValueError, "bounded.*not a proof of global infeasibility"):
            ground_on(pose(translation=self.surface.origin), body, body[body[:, 2] == 0],
                      [0, 0, -1], self.surface, **self.tolerances)
    # END free-yaw and coordinate-gauge regressions.


if __name__ == "__main__":
    unittest.main()
# END interaction-region and functional-frame regression tests.
