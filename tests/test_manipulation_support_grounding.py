"""CPU geometry tests; semantic selection is injected and no models are loaded."""

import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from manipulation.observations import backproject
from manipulation.support_grounding import infer_support


def camera(width, height):
    return np.array([[100., 0., width/2], [0., 100., height/2], [0., 0., 1.]])


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
        surface, diagnostic = infer_support(image, depth, K, side | support, [0.3, 0., 0.5],
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
            infer_support(image, depth, camera(50, 40), mask, [0, 0, 0.5],
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
            image, depth, K, mask, [0, 0, 0.5], instruction="Use the visible support",
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
            image, np.ones((height, width)), camera(width, height), mask, [0, 0, 0.5],
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
                              np.ones((30, 30), bool), [0, 0, 0.5],
                              instruction="Place it", client=client)
        client.chat_json.assert_not_called()

    def test_selection_rejects_geometry_and_non_integer_or_unknown_ids(self):
        image = Image.new("RGB", (30, 30))
        for selection in ({"candidate_id": True}, {"candidate_id": 9},
                          {"candidate_id": 1, "normal": [0, 0, 1]}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                infer_support(image, np.ones((30, 30)), camera(30, 30),
                              np.ones((30, 30), bool), [0, 0, 0.5], instruction="Place it",
                              client=Mock(chat_json=Mock(return_value=selection)))


if __name__ == "__main__":
    unittest.main()
