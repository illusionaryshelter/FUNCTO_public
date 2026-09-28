"""CPU region consumers: geometry invariants and actual masked correspondence.

Synthetic images encode known correspondences; no model, GPU, HTTP request or
dataset-specific selector is used. These tests establish coordinate and input
contracts, not the quality of predicted SAM regions or dense descriptors.
"""

# BEGIN interaction-region and functional-frame regression tests.
from copy import deepcopy
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from manipulation.goals import SupportSurface, check_on
from manipulation.regions import (align_axis, bind_observations, bind_region,
                                  check_region_on, function_frame, ground_on,
                                  ground_relation, match_regions,
                                  relation_error, sample_region_pixels)


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

    def observations(self, target_shift=(2, 1)):
        yy, xx = np.mgrid[:16, :16]
        rgb = np.stack((xx+1, yy+1, np.ones_like(xx)*50), axis=-1).astype(np.uint8)
        scene_rgb = rgb.copy()
        source_mask = np.zeros((16, 16), bool); source_mask[5:9, 4:8] = True
        target_mask = np.zeros((16, 16), bool)
        dx, dy = target_shift
        target_mask[5+dy:9+dy, 4+dx:8+dx] = True
        scene_rgb[5+dy:9+dy, 4+dx:8+dx] = rgb[5:9, 4:8]
        common = {"owner_mask": np.ones((16, 16), bool), "depth": self.depth,
                  "K": self.K}
        demo = dict(common, image=Image.fromarray(rgb), mask=source_mask,
                    T_camera_owner=pose((.3, -.1, .2), (.4, .1, -.2)),
                    region_id="source_surface", owner_id="source_instance")
        scene = dict(common, image=Image.fromarray(scene_rgb), mask=target_mask,
                     T_camera_owner=pose((-.2, .5, -.3), (-.2, .3, .1)),
                     region_id="corresponding_surface", owner_id="target_instance")
        return demo, scene

    @staticmethod
    def descriptors(first, second):
        # Images encode a unique feature direction at each corresponding pixel.
        # Full-resolution synthetic features exercise the unchanged masked
        # matcher, including original-pixel coordinate conventions.
        output = []
        for image in (first, second):
            rgb = np.asarray(image, dtype=np.float32)
            x, y = rgb[:, :, 0] / 16, rgb[:, :, 1] / 16
            output.append(np.stack((np.ones_like(x), x, y, x*y))[None])
        return output

    def test_matching_consumes_both_region_masks_and_retains_owner_image_context(self):
        demo, scene = self.observations()
        observed_sizes = []
        def extractor(first, second):
            observed_sizes.append((first.size, second.size))
            return self.descriptors(first, second)
        result = match_regions(demo, scene, extractor=extractor, max_error=1e-6)
        camera_mapping = pose(translation=[.02, .01, 0])
        expected = np.linalg.inv(scene["T_camera_owner"]) @ camera_mapping @ demo["T_camera_owner"]
        np.testing.assert_allclose(result["transform"], expected, atol=1e-12)
        self.assertEqual(observed_sizes, [((16, 16), (16, 16))])
        data = result["diagnostics"]
        self.assertEqual(data["matching"]["masked_pixel_count"], 16)
        self.assertEqual(len(data["source_pixels"]), 16)
        np.testing.assert_allclose(data["target_pixels"]-data["source_pixels"],
                                   np.broadcast_to([2, 1], (16, 2)), atol=1e-14)
        for key, mask in (("source_pixels", demo["mask"]), ("target_pixels", scene["mask"])):
            x, y = data[key].astype(int).T
            self.assertTrue(mask[y, x].all())
        # Changing only the paired local regions genuinely changes the binding.
        first2, second2 = self.observations(target_shift=(3, 2))
        moved = match_regions(first2, second2, extractor=self.descriptors, max_error=1e-6)
        expected2 = np.linalg.inv(scene["T_camera_owner"]) @ pose(translation=[.03, .02, 0]) @ demo["T_camera_owner"]
        np.testing.assert_allclose(moved["transform"], expected2, atol=1e-12)
        self.assertGreater(np.linalg.norm(moved["transform"]-result["transform"]), .01)

    def test_wrong_owner_empty_regions_and_collapsed_matches_cannot_be_repaired(self):
        demo, scene = self.observations()
        wrong = deepcopy(scene); wrong["owner_mask"][:] = False; wrong["owner_mask"][:3] = True
        with self.assertRaisesRegex(ValueError, "outside its declared owner"):
            match_regions(demo, wrong, extractor=self.descriptors)
        empty = deepcopy(scene); empty["mask"][:] = False
        with self.assertRaisesRegex(ValueError, "no selected pixels"):
            match_regions(demo, empty, extractor=self.descriptors)
        no_depth = deepcopy(scene); no_depth["depth"][:] = 0
        with self.assertRaisesRegex(ValueError, "valid depth"):
            match_regions(demo, no_depth, extractor=self.descriptors)
        def constant(first, second):
            return [np.ones((1, 2, image.height, image.width), np.float32) for image in (first, second)]
        with self.assertRaisesRegex(ValueError, "distinct"):
            match_regions(demo, scene, extractor=constant)


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
        with self.assertRaisesRegex(ValueError, "Specified functional contact face"):
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
        with self.assertRaisesRegex(ValueError, "antiparallel_axis"):
            ground_on(upside_down, self.body, self.bottom, [0, 0, -1], self.surface, **self.tolerances)
        result = ground_on(upside_down, self.body, self.bottom, [0, 0, -1], self.surface,
                           antiparallel_axis=[1, 0, 0], **self.tolerances)
        self.assertTrue(result["checks"]["region"]["valid"])
        for field, value in (("gap_tolerance", -1), ("normal_tolerance", np.pi),
                             ("gap_tolerance", np.nan), ("margin", -.1)):
            kwargs = dict(self.tolerances, **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                ground_on(pose(), self.body, self.bottom, [0, 0, -1], self.surface, **kwargs)


if __name__ == "__main__":
    unittest.main()
# END interaction-region and functional-frame regression tests.
