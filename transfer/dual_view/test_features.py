"""Small independent checks of two-view raw scores and unknown intervals."""

import unittest
from copy import deepcopy

import numpy as np

from .features import (ResponseField, _footprint_support, _sampling_corners,
                       interval_score, model_bounds, prepare_view, source_descriptors)
from .test_geometry import _observation, _surface


class BoundsTests(unittest.TestCase):
    def test_layernorm_zero_mean_bound_and_convex_max(self):
        rng = np.random.default_rng(4)
        q = rng.normal(size=(7, 16))
        gamma, beta = rng.normal(size=(2, 16))
        raw = rng.normal(size=(50, 16))
        centered = raw-raw.mean(axis=1, keepdims=True)
        y = centered / np.sqrt(np.mean(centered**2, axis=1, keepdims=True)+1e-5)
        feature = y*gamma+beta
        lower, upper = model_bounds(q, gamma, beta)
        score = q @ feature.T
        self.assertTrue(np.all(score >= lower[:, None]))
        self.assertTrue(np.all(score <= upper[:, None]))
        weights = rng.dirichlet(np.ones(4), size=8)
        interpolated = score[:, :4] @ weights.T
        fused = np.maximum(interpolated, score[:, 4:12])
        self.assertTrue(np.all(fused >= lower[:, None]))
        self.assertTrue(np.all(fused <= upper[:, None]))
        # Constant q*gamma is orthogonal to every zero-mean normalized output.
        low, high = model_bounds(np.ones((1, 16)), np.ones(16), np.zeros(16))
        self.assertLessEqual(abs(low[0]), np.nextafter(0., 1.))
        self.assertLessEqual(abs(high[0]), np.nextafter(0., 1.))

    def test_unknown_keeps_fixed_denominator(self):
        result = interval_score(np.array([2., np.nan]), np.array([True, False]),
                                np.array([-3., -4.]), np.array([3., 4.]))
        self.assertEqual(result, dict(lower=-1., upper=3., coverage=.5))
        complete = interval_score(np.array([2., -1.]), np.ones(2, bool),
                                  np.array([-3., -4.]), np.array([3., 4.]))
        self.assertEqual(complete["lower"], complete["upper"])
        self.assertEqual(complete["lower"], .5)
        with self.assertRaisesRegex(ValueError, "violates"):
            interval_score(np.array([4.]), np.array([True]), np.array([-3.]), np.array([3.]))


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.surface = _surface()
        self.q = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]], np.float32)
        self.feature = np.broadcast_to(np.array([-1., 1., -1., 1.], np.float32)[None, :, None, None],
                                       (1, 4, 9, 9)).copy()
        self.observation = _observation(np.ones((13, 13)))
        self.points = np.array([[0., 0., 1.], [1/256, 0., 1.]])
        self.normals = np.tile([0., 0., -1.], (2, 1))

    def field(self, b=None, surface=None, feature=None):
        views = [prepare_view(self.q, self.feature if feature is None else feature,
                              crop_xyxy=[0, 0, 13, 13], observation=self.observation, view_id="A"),
                 prepare_view(self.q, b, crop_xyxy=[0, 0, 13, 13],
                              observation=self.observation, view_id="B")]
        return ResponseField(self.surface if surface is None else surface, self.q, views,
                             np.ones(4), np.zeros(4))

    def test_single_view_negative_is_not_replaced_by_missing_zero(self):
        sampled = self.field().sample(self.points, self.normals)
        np.testing.assert_allclose(sampled["scores"], [[-1., -1.], [1., 1.]])
        self.assertTrue(sampled["known"].all())
        np.testing.assert_array_equal(sampled["view_valid"], [[True, False], [True, False]])
        self.assertEqual(sampled["view_scores"].shape, (2, 2, 2))
        self.assertTrue(np.isnan(sampled["view_scores"][:, :, 1]).all())
        self.assertEqual(sampled["view_state"][0, 1], "missing_feature")

    def test_raw_max_and_paired_sampling_agree(self):
        b = -self.feature
        field = self.field(b)
        all_rows = field.sample(self.points, self.normals)
        paired = field.sample(self.points, self.normals, np.array([0, 1]))
        np.testing.assert_allclose(all_rows["scores"], np.ones((2, 2)))
        np.testing.assert_array_equal(paired["scores"], all_rows["scores"][[0, 1], [0, 1]])
        self.assertEqual(paired["view_scores"].shape, (2, 2))

    def test_target_raw_response_interpolation_preserves_native_target_scale(self):
        feature = self.feature.copy()
        yy, xx = np.mgrid[:9, :9]
        feature[0, 0] = .03*xx+.04*yy
        feature[0, 1] = -.02*xx+.01*yy
        # A linear feature field lets the expected raw dot be evaluated at
        # the continuous native target coordinate without another sampler.
        uv = np.array([[6.1, 6.2], [6.7, 6.4]])
        points = np.column_stack(((uv-6)/256, np.ones(2)))
        sampled = self.field(feature=feature).sample(points, self.normals)
        grid = uv*(840/13)*(8/839)
        expected = np.array([.03*grid[:, 0]+.04*grid[:, 1],
                             -.02*grid[:, 0]+.01*grid[:, 1]])
        np.testing.assert_allclose(sampled["scores"], expected, rtol=2e-7, atol=1e-8)
        # F/840 would produce different values and is reserved for the source.
        wrong_source_grid = uv*9/13
        self.assertGreater(np.max(np.abs(expected[0]-(.03*wrong_source_grid[:, 0]
                            +.04*wrong_source_grid[:, 1]))), 1e-2)

    def test_missing_normals_and_reversed_normals_never_score(self):
        normals = np.array([[np.nan]*3, [0., 0., 1.]])
        result = self.field().sample(self.points, normals)
        self.assertFalse(result["known"].any())
        self.assertTrue(np.isnan(result["scores"]).all())
        self.assertEqual(result["view_state"][0, 0], "missing_query_normal")
        self.assertEqual(result["view_state"][1, 0], "opposite_query_normal")

    def test_invalid_owner_plane_cannot_bypass_geometry(self):
        surface = deepcopy(self.surface)
        surface["observation_valid"][surface["pixel_to_observation"][0][6, 6]] = False
        result = self.field(surface=surface).sample(self.points[:1], self.normals[:1])
        self.assertFalse(result["known"].any())
        self.assertEqual(result["view_state"][0, 0], "invalid_local_plane")

    def test_partial_feature_support_never_renormalizes(self):
        surface = deepcopy(self.surface)
        surface["observation_entity_member"][surface["pixel_to_observation"][0][5, 5]] = False
        result = self.field(surface=surface).sample(self.points[:1], self.normals[:1])
        self.assertFalse(result["known"].any())
        self.assertEqual(result["view_state"][0, 0], "partial_feature_support")

    def test_zero_weight_corners_and_no_outside_clamp(self):
        inside, corners, weights = _sampling_corners(np.array([[2., 2.], [-1e-9, 2.], [8., 8.]]), (9, 9))
        np.testing.assert_array_equal(inside, [True, False, True])
        np.testing.assert_array_equal(weights[0], [1., 0., 0., 0.])
        np.testing.assert_array_equal(weights[2], [1., 0., 0., 0.])
        self.assertEqual(weights[1].sum(), 0)
        # Zero-weight corners may point outside; they cannot enlarge a footprint.
        lookup = self.surface["pixel_to_observation"][0]
        supported = _footprint_support(self.surface, 0, np.array([lookup[6, 6]]),
            np.array([[[6, 6], [999, 999], [-1, -1], [999, -1]]]),
            np.array([[True, False, False, False]]))
        np.testing.assert_array_equal(supported, [[True, False, False, False]])

    def test_fixed_local_surface_can_connect_around_a_pixel_gap(self):
        surface = deepcopy(self.surface)
        lookup = surface["pixel_to_observation"][0]
        # The minimal interpolation rectangle is disconnected. A valid path
        # around it remains in the same fixed plane and physical-radius ball.
        surface["observation_entity_member"][lookup[6, 6]] = False
        supported = _footprint_support(surface, 0, np.array([lookup[6, 5]]),
            np.array([[[7, 6], [7, 6], [7, 6], [7, 6]]]), np.ones((1, 4), bool))
        self.assertTrue(supported.all())
        # Cutting the complete vertical band leaves no same-view path. The
        # other camera's intact plane must not repair that disconnected graph.
        surface["observation_entity_member"][lookup[:, 6]] = False
        supported = _footprint_support(surface, 0, np.array([lookup[6, 5]]),
            np.array([[[7, 6], [7, 6], [7, 6], [7, 6]]]), np.ones((1, 4), bool))
        self.assertFalse(supported.any())

    def test_each_view_score_must_respect_bound_even_if_other_view_wins(self):
        bad = self.feature.copy()
        bad[:, 0] = -10
        with self.assertRaisesRegex(ValueError, "violates"):
            self.field(-self.feature, feature=bad).sample(self.points, self.normals)

    def test_missing_crop_and_wrong_camera_order_are_explicit(self):
        records = [prepare_view(self.q, self.feature, crop_xyxy=None,
                               observation=self.observation, view_id=name) for name in ("A", "B")]
        field = ResponseField(self.surface, self.q, records, np.ones(4), np.zeros(4))
        sampled = field.sample(self.points, self.normals)
        self.assertFalse(sampled["known"].any())
        self.assertEqual(sampled["view_state"][0, 0], "missing_crop")
        with self.assertRaisesRegex(ValueError, "identities"):
            ResponseField(self.surface, self.q, records[::-1], np.ones(4), np.zeros(4))


class SourceTests(unittest.TestCase):
    def test_source_crop_checks_frozen_queries_before_native_sampler(self):
        source = dict(xyz=np.zeros((2, 3)), pixels=np.array([[1, 1], [8, 8]]),
                      object_mask=np.ones((10, 10), bool))
        with self.assertRaisesRegex(ValueError, "outside"):
            source_descriptors(source, np.array([0, 1]), None, [0, 0, 5, 5])
        with self.assertRaisesRegex(ValueError, "integer"):
            source_descriptors(source, np.array([0]), None, [0., 0., 5., 5.])

    def test_source_uses_native_sampler_and_only_requires_selected_pixels(self):
        source = dict(xyz=np.zeros((3, 3)), pixels=np.array([[3, 4], [5, 6], [9, 9]]),
                      object_mask=np.ones((10, 10), bool))
        yy, xx = np.mgrid[:9, :9]
        feature = np.stack([xx, yy]).astype(np.float32)[None]
        descriptors = source_descriptors(source, np.array([0, 1]), feature, [2, 3, 7, 8])
        # Crop-local coordinates (1,1),(3,3), native scale F/840 and crop
        # resize 840/5 give (1.8,1.8),(5.4,5.4). Unselected (9,9) is irrelevant.
        np.testing.assert_allclose(descriptors, [[1.8, 1.8], [5.4, 5.4]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
