"""Independent checks of raw sampling and fixed-domain observed support."""

import unittest
from copy import deepcopy

import numpy as np
from scipy.stats import percentileofscore

from .features import (ResponseField, _footprint_support, _sampling_corners,
                       EmpiricalSupport, capacity_score, prepare_view, source_descriptors)
from .test_geometry import _observation, _surface


# BEGIN observed-support objective replaces retired latent-bound tests.
class SupportTests(unittest.TestCase):
    def test_mid_cdf_matches_scipy_ties_and_fixed_reference(self):
        reference = np.array([[1., 2., 2., 4.], [10., 20., 30., 40.]])
        known = np.ones(reference.shape, bool)
        support = EmpiricalSupport(reference, known)
        measured = np.array([[0., 1., 2., 3., 4., 5.], [5., 10., 15., 25., 40., 50.]])
        expected = np.stack([np.maximum(2*percentileofscore(row, sample, kind="mean")/100-1, 0)
                             for row, sample in zip(reference, measured)])
        np.testing.assert_allclose(support.utility(measured, np.arange(2)[:, None]), expected)
        # A candidate must not recalibrate itself or mutate the frozen domain.
        reference[:] = -999
        np.testing.assert_allclose(support.utility(measured, np.arange(2)[:, None]), expected)
        with self.assertRaises(ValueError):
            support.references[0][0] = 0
        affine = EmpiricalSupport(3*np.array([[1., 2., 2., 4.]])+17, known[:1])
        np.testing.assert_allclose(affine.utility(3*measured[:1]+17, [[0]]), expected[:1])

    def test_constant_missing_and_empty_reference_have_no_positive_support(self):
        support = EmpiricalSupport(np.array([[7., 7.], [1., 2.]]),
                                   np.array([[True, True], [False, False]]))
        actual = support.utility(np.array([[7., np.nan], [3., -np.inf]]), [[0], [1]])
        self.assertEqual(actual[0, 0], 0.)
        self.assertTrue(np.isnan(actual[0, 1]))
        self.assertTrue(np.isnan(actual[1]).all())
        self.assertEqual(support.diagnostics["constant_rows"], [0])
        self.assertEqual(support.diagnostics["empty_rows"], [1])
        empty = EmpiricalSupport(np.empty((2, 0)), np.empty((2, 0), bool))
        self.assertEqual(empty.utility(np.empty((2, 0)), [[0], [1]]).shape, (2, 0))

    def test_arbitrary_row_indices_and_broadcast_matrix_agree(self):
        support = EmpiricalSupport(np.array([[1., 2., 3.], [10., 20., 30.]]),
                                   np.ones((2, 3), bool))
        paired = support.utility(np.array([3., 30., 2., np.nan]), [0, 1, 0, 1])
        np.testing.assert_allclose(paired, [2/3, 2/3, 0, np.nan], equal_nan=True)
        with self.assertRaisesRegex(ValueError, "identity"):
            support.utility(np.array([1.]), [2])

    def test_capacity_is_global_and_preserves_camera_specific_scores(self):
        result = capacity_score(np.array([[.9, .8], [.85, np.nan]]),
                                np.array([[0, 1], [0, -1]]))
        self.assertAlmostEqual(result["score"], (.8+.85)/2)
        np.testing.assert_array_equal(result["selected_view"], [1, 0])
        np.testing.assert_array_equal(result["element_ids"], [1, 0])
        # Borrowing row 0's A score for its B element would incorrectly yield
        # .85 rather than .45; capacity is coupled to each actual view edge.
        lower_other_view = capacity_score(np.array([[.9, .1], [.8, np.nan]]),
                                         np.array([[0, 1], [0, -1]]))
        self.assertAlmostEqual(lower_other_view["score"], .45)

    def test_duplicate_elements_unknown_and_zero_keep_fixed_query_denominator(self):
        result = capacity_score(np.array([[.9, .8], [.7, np.nan], [0., np.nan],
                                          [np.nan, np.nan]]),
                                np.array([[5, 5], [5, -1], [6, -1], [-1, -1]]))
        self.assertAlmostEqual(result["score"], .9/4)
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["positive_row_count"], 2)
        self.assertEqual(result["capacity_lost_count"], 1)
        np.testing.assert_array_equal(result["matched"], [True, False, False, False])
        all_zero = capacity_score(np.zeros((2, 2)), np.array([[1, 2], [3, 4]]))
        self.assertEqual(all_zero["score"], 0)
        self.assertFalse(all_zero["matched"].any())

    def test_hiding_measurements_cannot_raise_observed_support(self):
        rng = np.random.default_rng(4)
        for _ in range(32):
            utility = rng.random((8, 2))
            elements = rng.integers(0, 8, size=(8, 2))
            original = capacity_score(utility, elements)
            hidden = rng.random(utility.shape) < .5
            utility[hidden], elements[hidden] = np.nan, -1
            after = capacity_score(utility, elements)
            self.assertLessEqual(after["score"], original["score"] + 1e-14)
# END tests cover objective invariants rather than unobserved-score imputation.


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
        return ResponseField(self.surface if surface is None else surface, self.q, views)

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

    # BEGIN RGB evidence may exist without a measurable source normal.
    def test_missing_normal_can_read_rgb_without_oriented_geometry(self):
        normals = np.array([[np.nan]*3, [0., 0., 1.]])
        result = self.field().sample(self.points, normals)
        self.assertTrue(result["known"][:, 0].all())
        self.assertFalse(result["known"][:, 1].any())
        self.assertTrue(np.isnan(result["scores"][:, 1]).all())
        self.assertEqual(result["view_state"][0, 0], "appearance_valid_orientation_unknown")
        self.assertEqual(result["view_state"][1, 0], "opposite_query_normal")
        self.assertFalse(result["geometry"][0]["supported"][0])
        self.assertFalse(result["view_orientation_known"][0, 0])

    def test_unknown_source_normal_cannot_bypass_target_plane_or_footprint(self):
        for key, expected in (("observation_valid", "invalid_local_plane"),
                              ("observation_entity_member", "outside_entity_footprint")):
            surface = deepcopy(self.surface)
            surface[key][surface["pixel_to_observation"][0][6, 6]] = False
            result = self.field(surface=surface).sample(self.points[:1], np.full((1, 3), np.nan))
            self.assertFalse(result["known"].any())
            self.assertEqual(result["view_state"][0, 0], expected)
    # END availability remains qualified by the complete target observation.

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

    def test_missing_crop_and_wrong_camera_order_are_explicit(self):
        records = [prepare_view(self.q, self.feature, crop_xyxy=None,
                               observation=self.observation, view_id=name) for name in ("A", "B")]
        field = ResponseField(self.surface, self.q, records)
        sampled = field.sample(self.points, self.normals)
        self.assertFalse(sampled["known"].any())
        self.assertEqual(sampled["view_state"][0, 0], "missing_crop")
        with self.assertRaisesRegex(ValueError, "identities"):
            ResponseField(self.surface, self.q, records[::-1])


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
