"""CPU geometry tests; semantic selection is injected and no models are loaded."""

import unittest
# BEGIN explicit support direction and coordinate-gauge regression helpers.
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from manipulation.observations import backproject
from manipulation.regions import ground_on
from manipulation.support_grounding import infer_support
# END explicit support direction and coordinate-gauge regression helpers.


def camera(width, height):
    return np.array([[100., 0., width/2], [0., 100., height/2], [0., 0., 1.]])


# BEGIN support selection with a direction, independent of arbitrary point/gauge choices.
class SupportGroundingTest(unittest.TestCase):
    def test_semantic_choice_can_select_smaller_support_instead_of_largest_side(self):
        width, height = 120, 80
        image = Image.new("RGB", (width, height))
        K = camera(width, height)
        yy, xx = np.indices((height, width))
        side = (xx >= 4) & (xx < 50) & (yy >= 4) & (yy < 76)
        support = (xx >= 75) & (xx < 112) & (yy >= 20) & (yy < 60)
        depth = np.zeros((height, width))
        depth[side] = -0.4 / ((xx[side]-K[0, 2])/K[0, 0])
        depth[support] = 1.
        client = Mock(chat_json=Mock(return_value={"candidate_id": 2}))
        surface, diagnostic = infer_support(image, depth, K, side | support,
                                            support_direction=[0., 0., -1.],
                                            instruction="Place the object on the smaller shelf", client=client)
        self.assertEqual(diagnostic["candidate_count"], 2)
        self.assertEqual(diagnostic["selected_candidate_id"], 2)
        self.assertLessEqual(diagnostic["sampled_point_count"], 6000)
        self.assertAlmostEqual(surface.origin[2], 1., places=10)
        np.testing.assert_allclose(surface.normal, [0., 0., -1.], atol=1e-10)
        self.assertEqual(len(client.chat_json.call_args.kwargs["images"]), 2)
        self.assertIn("smaller shelf", client.chat_json.call_args.args[0])
        # Every vertex projects to the chosen support, not the larger side.
        polygon = np.rint(diagnostic["patch_boundary_pixels"]).astype(int)
        self.assertTrue(support[polygon[:, 1], polygon[:, 0]].all())

    def test_single_plane_still_requires_semantic_confirmation_and_null_stops(self):
        image = Image.new("RGB", (50, 40))
        depth, mask = np.ones((40, 50)), np.ones((40, 50), bool)
        client = Mock(chat_json=Mock(return_value={"candidate_id": None}))
        with self.assertRaisesRegex(ValueError, "No unambiguous"):
            infer_support(image, depth, camera(50, 40), mask, support_direction=[0, 0, -1],
                          instruction="Place it on the table", client=client)
        client.chat_json.assert_called_once()
        self.assertIn("only one candidate", client.chat_json.call_args.args[0])

    def test_hole_is_excluded_instead_of_filled_by_a_convex_hull(self):
        width, height = 100, 90
        image = Image.new("RGB", (width, height))
        K = camera(width, height)
        depth = np.ones((height, width))
        mask = np.zeros((height, width), bool)
        mask[5:85, 5:95] = True
        mask[30:60, 35:65] = False
        surface, diagnostic = infer_support(
            image, depth, K, mask, support_direction=[0, 0, -1], instruction="Use the visible support",
            client=Mock(chat_json=Mock(return_value={"candidate_id": 1})),
        )
        self.assertIn("inscribed disk", diagnostic["patch_method"])
        # An independent 3-D halfspace check includes *every* invalid image
        # pixel, not merely the few reported polygon vertices or hole center.
        y, x = np.nonzero(~mask)
        unobserved = backproject(depth, K, np.column_stack((x, y)))
        local = (unobserved-surface.origin) @ surface.basis
        equations = surface._hull.equations
        inside = np.all(local @ equations[:, :2].T + equations[:, 2] <= 1e-10, axis=1)
        self.assertFalse(inside.any())
        self.assertGreater(diagnostic["patch_pixel_count"], 100)

    def test_long_convex_panel_keeps_its_area_instead_of_always_using_a_disk(self):
        width, height = 120, 40
        image = Image.new("RGB", (width, height))
        mask = np.zeros((height, width), bool)
        mask[10:30, 10:110] = True
        surface, diagnostic = infer_support(
            image, np.ones((height, width)), camera(width, height), mask,
            support_direction=[0, 0, -1],
            instruction="Use the long shelf", client=Mock(chat_json=Mock(return_value={"candidate_id": 1})),
        )
        self.assertEqual(diagnostic["patch_method"], "full observed convex patch")
        self.assertGreater(diagnostic["patch_pixel_count"], 1500)
        vertices = surface.origin + surface.boundary @ surface.basis.T
        self.assertGreater(np.ptp(vertices[:, 0]), 0.9)

    def test_no_valid_depth_fails_before_semantic_selection(self):
        image = Image.new("RGB", (30, 30))
        client = Mock()
        for value in (0., np.nan, np.inf):
            with self.subTest(depth=value), self.assertRaisesRegex(ValueError, "valid depth"):
                infer_support(image, np.full((30, 30), value), camera(30, 30),
                              np.ones((30, 30), bool), support_direction=[0, 0, -1],
                              instruction="Place it", client=client)
        client.chat_json.assert_not_called()

    def test_selection_rejects_geometry_and_non_integer_or_unknown_ids(self):
        image = Image.new("RGB", (30, 30))
        for selection in ({"candidate_id": True}, {"candidate_id": 9},
                          {"candidate_id": 1, "normal": [0, 0, 1]}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                infer_support(image, np.ones((30, 30)), camera(30, 30),
                              np.ones((30, 30), bool), support_direction=[0, 0, -1], instruction="Place it",
                              client=Mock(chat_json=Mock(return_value=selection)))

    def test_direction_magnitude_does_not_change_signed_surface(self):
        image = Image.new("RGB", (40, 30))
        depth, mask = np.ones((30, 40)), np.ones((30, 40), bool)
        baseline = None
        for scale in (1., 0.05, 1e-250, 1e250):
            with self.subTest(scale=scale):
                surface, diagnostics = infer_support(
                    image, depth, camera(40, 30), mask, support_direction=[0, 0, scale],
                    instruction="Use this support", client=Mock(chat_json=Mock(return_value={"candidate_id": 1})),
                )
                np.testing.assert_allclose(surface.normal, [0, 0, 1], atol=1e-12)
                np.testing.assert_allclose(diagnostics["support_direction_camera"], [0, 0, 1])
                vertices = surface.origin + surface.boundary @ surface.basis.T
                if baseline is None:
                    baseline = vertices
                self.assert_same_vertices(vertices, baseline)

    def test_side_uses_a_direction_not_position_relative_to_plane(self):
        image = Image.new("RGB", (40, 30))
        # Both depths retain the same +Z side. Treating [0, 0, 1] as a point
        # would put that witness on opposite sides of these two planes.
        for depth_value in (0.5, 2.):
            with self.subTest(depth_m=depth_value):
                surface, _ = infer_support(
                    image, np.full((30, 40), depth_value), camera(40, 30), np.ones((30, 40), bool),
                    support_direction=[0, 0, 1], instruction="Use this support",
                    client=Mock(chat_json=Mock(return_value={"candidate_id": 1})),
                )
                np.testing.assert_allclose(surface.normal, [0, 0, 1], atol=1e-12)
                self.assertAlmostEqual(surface.origin[2], depth_value)

    def test_opposite_directions_select_opposite_sides_of_same_observed_patch(self):
        image = Image.new("RGB", (40, 30))
        vertices = []
        for sign in (-1, 1):
            surface, _ = infer_support(
                image, np.ones((30, 40)), camera(40, 30), np.ones((30, 40), bool),
                support_direction=[0, 0, sign], instruction="Use this support",
                client=Mock(chat_json=Mock(return_value={"candidate_id": 1})),
            )
            np.testing.assert_allclose(surface.normal, [0, 0, sign], atol=1e-12)
            vertices.append(surface.origin + surface.boundary @ surface.basis.T)
        self.assert_same_vertices(*vertices)

    def test_near_tangent_direction_does_not_certify_support_side(self):
        image = Image.new("RGB", (40, 30))
        for normal_component in (0., 0.05, -0.05):
            with self.subTest(normal_component=normal_component), self.assertRaisesRegex(ValueError, "near-tangent"):
                infer_support(
                    image, np.ones((30, 40)), camera(40, 30), np.ones((30, 40), bool),
                    support_direction=[np.sqrt(1-normal_component**2), 0, normal_component],
                    instruction="Use this support", client=Mock(chat_json=Mock(return_value={"candidate_id": 1})),
                )
        # This is only a side test. A direction just beyond the conservative
        # tangent margin does not acquire a second, arbitrary levelness gate.
        surface, diagnostics = infer_support(
            image, np.ones((30, 40)), camera(40, 30), np.ones((30, 40), bool),
            support_direction=[np.sqrt(1-0.101**2), 0, 0.101], instruction="Use this support",
            client=Mock(chat_json=Mock(return_value={"candidate_id": 1})),
        )
        np.testing.assert_allclose(surface.normal, [0, 0, 1], atol=1e-12)
        self.assertAlmostEqual(diagnostics["support_side_alignment"], 0.101)

    def test_invalid_direction_fails_before_semantic_selection(self):
        client = Mock()
        for direction in ([0, 0, 0], [0, np.nan, 1], [0, 0, np.inf], [1, 0], [[0, 0, 1]]):
            with self.subTest(direction=direction), self.assertRaisesRegex(ValueError, "support direction"):
                infer_support(
                    Image.new("RGB", (30, 30)), np.ones((30, 30)), camera(30, 30),
                    np.ones((30, 30), bool), support_direction=direction, instruction="Use this support", client=client,
                )
        client.chat_json.assert_not_called()

    def test_calibrated_camera_rotation_rotates_the_same_metric_patch(self):
        # Rotating square RGB-D arrays by 90 degrees is a concrete calibrated
        # camera-coordinate rotation, including rays, mask, and side direction.
        size = 80
        K = np.array([[100., 0., 39.5], [0., 100., 39.5], [0., 0., 1.]])
        yy, xx = np.indices((size, size))
        normal = np.array([0.2, -0.3, 1.])
        normal /= np.linalg.norm(normal)
        depth = 1. / (normal[0]*(xx-K[0, 2])/K[0, 0]
                      + normal[1]*(yy-K[1, 2])/K[1, 1] + normal[2])
        mask = np.zeros((size, size), bool)
        mask[6:30, 17:70] = True
        rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        surfaces = []
        for turns, direction in ((0, -normal), (1, rotation @ -normal)):
            surface, _ = infer_support(
                Image.new("RGB", (size, size)), np.rot90(depth, turns), K, np.rot90(mask, turns),
                support_direction=direction, instruction="Use this support",
                client=Mock(chat_json=Mock(return_value={"candidate_id": 1})),
            )
            surfaces.append(surface)
        first, rotated = surfaces
        np.testing.assert_allclose(rotated.origin, rotation @ first.origin, atol=1e-12)
        np.testing.assert_allclose(rotated.normal, rotation @ first.normal, atol=1e-12)
        vertices = first.origin + first.boundary @ first.basis.T
        rotated_vertices = rotated.origin + rotated.boundary @ rotated.basis.T
        self.assert_same_vertices(rotated_vertices, vertices @ rotation.T)

    def test_pca_axes_and_unsigned_normal_are_only_coordinate_gauges(self):
        width, height = 80, 60
        mask = np.zeros((height, width), bool)
        mask[8:52, 10:70] = True
        body = np.array([[x, y, z] for x in (-.025, .025) for y in (-.04, .04) for z in (0., .05)])
        bottom = body[body[:, 2] == 0.]
        preferred = np.eye(4)
        preferred[:3, 3] = [.5, .02, 1.3]
        vertices, goals = [], []
        # Inject only the arbitrary SVD choices. Candidate pixels, finite patch
        # construction, and the downstream full-body placement solver are real.
        for angle, sign in ((0., 1), (.73, -1), (np.pi, 1)):
            candidate = {
                "candidate_id": 1, "mask": mask, "bbox": [10, 8, 70, 52],
                "origin": np.array([0., 0., 1.]), "normal": np.array([0., 0., float(sign)]),
                "first_axis": np.array([np.cos(angle), np.sin(angle), 0.]),
                "sample_inlier_count": int(mask.sum()), "observed_pixel_count": int(mask.sum()), "plane_rms_m": 0.,
            }
            with patch("manipulation.support_grounding._plane_candidates", return_value=([candidate], int(mask.sum()))):
                surface, _ = infer_support(
                    Image.new("RGB", (width, height)), np.ones((height, width)), camera(width, height), mask,
                    support_direction=[0, 0, 1], instruction="Use this support",
                    client=Mock(chat_json=Mock(return_value={"candidate_id": 1})),
                )
            vertices.append(surface.origin + surface.boundary @ surface.basis.T)
            result = ground_on(preferred, body, bottom, [0, 0, -1], surface,
                               margin=.005, gap_tolerance=1e-8, normal_tolerance=1e-8)
            self.assertTrue(result["checks"]["region"]["valid"])
            self.assertLessEqual(result["checks"]["body"]["boundary_violation_m"], 1e-8)
            goals.append(result["goal_pose"])
        for current_vertices, current_goal in zip(vertices[1:], goals[1:]):
            self.assert_same_vertices(current_vertices, vertices[0])
            np.testing.assert_allclose(current_goal, goals[0], atol=1e-9)

    def assert_same_vertices(self, first, second):
        """Compare physical point sets without assigning corner order or yaw."""
        distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
        self.assertLess(np.max(np.min(distances, axis=0)), 1e-10)
        self.assertLess(np.max(np.min(distances, axis=1)), 1e-10)
# END support selection with a direction, independent of arbitrary point/gauge choices.


if __name__ == "__main__":
    unittest.main()
