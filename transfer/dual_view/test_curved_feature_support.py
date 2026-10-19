"""Native curved-surface appearance and fixed per-camera reference contracts.

These CPU fixtures use actual depth pixels and analytic response fields. They
exercise the sampling/reference boundary without a model, candidate ranking,
target annotation, simulator or experiment-specific data.
"""

import unittest

import numpy as np
from scipy.ndimage import map_coordinates

from .curved_features import SurfaceResponseField
from .curved_geometry import build_target_surface
from .features import prepare_view
from .test_curved_geometry import observation


# BEGIN fixed appearance references use every actual same-camera entity row.
class CurvedFeatureSupportTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((25, 25), .6)
        self.K = np.array([[400., 0., 12.], [0., 400., 12.], [0., 0., 1.]])
        self.raw = [observation(self.depth, self.K) for _ in range(2)]
        self.queries = np.eye(3, dtype=np.float32)
        yy, xx = np.indices((25, 25))
        self.response = np.array([xx+2*yy-7, 3*xx-yy+2, -2*xx+yy-1], np.float32)

    def field(self, *, missing_second=False, constant=None):
        views = [dict(id=name, depth_m=self.depth, K=self.K,
            T_world_camera_cv=np.eye(4), observation=raw)
            for name, raw in zip(('a', 'b'), self.raw)]
        surface = build_target_surface(views)
        prepared = []
        for index, view in enumerate(views):
            feature = np.full_like(self.response, constant) if constant is not None else self.response+100*index
            if missing_second and index == 1:
                feature = None
            else:
                feature = feature[None]
            prepared.append(prepare_view(self.queries, feature, crop_xyxy=[0, 0, 25, 25],
                observation=view['observation'], view_id=view['id']))
        return SurfaceResponseField(surface, self.queries, prepared)

    def test_native_pixel_sampling_preserves_all_query_and_selected_row_outputs(self):
        field = self.field()
        pixels = np.array([[9, 9], [12, 9], [15, 15]])
        ids = field.surface['pixel_to_observation'][0][pixels[:, 1], pixels[:, 0]]
        points = field.surface['observation_xyz'][ids].copy()
        # A small candidate displacement still associates to the same actual
        # measurement. RGB must stay at its original pixel, not move with the
        # continuous candidate projection or use the source F/840 convention.
        points[:, 0] += .0001
        points = np.vstack((points, [10., 10., .6]))
        normals = np.full_like(points, np.nan)
        actual = field.sample(points, normals, registration_error_m=.002)
        gx, gy = (pixels*(840/25)*24/839).T
        expected = np.array([gx+2*gy-7, 3*gx-gy+2, -2*gx+gy-1])
        np.testing.assert_allclose(actual['view_scores'][:, :3, 0], expected, rtol=0, atol=2e-14)
        np.testing.assert_allclose(actual['scores'][:, :3], expected+100, rtol=0, atol=3e-14)
        np.testing.assert_array_equal(actual['view_valid'][:3], np.ones((3, 2), bool))
        self.assertTrue(np.isnan(actual['scores'][:, 3]).all())
        self.assertFalse(actual['view_valid'][3].any())
        self.assertFalse(actual['view_orientation_known'].any())
        self.assertTrue(np.all(actual['view_state'][:3] == 'appearance_valid_orientation_unknown'))
        rows = np.array([2, 0, 1, 2])
        selected = field.sample(points, normals, rows, registration_error_m=.002)
        np.testing.assert_array_equal(selected['view_scores'], actual['view_scores'][rows, np.arange(4)])
        np.testing.assert_array_equal(selected['known'], actual['known'][rows, np.arange(4)])
        np.testing.assert_array_equal(selected['view_state'], actual['view_state'])
        self.assertEqual(selected['reason_counts'], actual['reason_counts'])

    def test_references_cover_native_entity_measurements_in_each_camera_separately(self):
        # Camera B sees only the right part of the entity. Its high responses
        # cannot enter camera A's reference or supply its missing observations.
        self.raw[1]['entity_member'] &= self.raw[1]['pixels'][:, 0] >= 11
        field = self.field()
        supports, diagnostics = field.prepare_support()
        self.assertEqual(len(supports), 2)
        self.assertEqual([item['view_id'] for item in diagnostics], ['a', 'b'])
        self.assertEqual([item['observation_count'] for item in diagnostics], [625, 350])
        for view_index, support in enumerate(supports):
            ids = np.flatnonzero(field.surface['observation_entity_member']
                & (field.surface['observation_view_index'] == view_index))
            points = field.surface['observation_xyz'][ids]
            sampled = field.sample(points, np.full_like(points, np.nan))
            valid = sampled['view_valid'][:, view_index]
            pixels = field.surface['observation_pixels'][ids[valid]]
            gx, gy = (pixels*(840/25)*24/839).T
            expected = np.array([gx+2*gy-7, 3*gx-gy+2, -2*gx+gy-1])+100*view_index
            self.assertEqual(diagnostics[view_index]['reference_counts'], [int(valid.sum())]*3)
            self.assertEqual(sum(diagnostics[view_index]['reason_counts'].values()), len(ids))
            for row in range(3):
                np.testing.assert_array_equal(support.references[row],
                    np.sort(sampled['view_scores'][row, valid, view_index]))
                np.testing.assert_allclose(support.references[row], np.sort(expected[row]),
                    rtol=0, atol=4e-14)
                self.assertFalse(support.references[row].flags.writeable)
        self.assertLess(supports[0].references[0].max(), supports[1].references[0].min())
        self.assertFalse(diagnostics[0]['is_match_probability'])

    def test_incomplete_positive_corner_stays_unknown_in_reference_and_candidate(self):
        self.raw[0]['entity_member'][8*25+8] = False
        field = self.field()
        supports, diagnostics = field.prepare_support()
        center = field.surface['pixel_to_observation'][0][9, 9]
        points = field.surface['observation_xyz'][[center]]
        sampled = field.sample(points, np.full_like(points, np.nan))
        self.assertEqual(sampled['view_state'][0, 0], 'partial_feature_support')
        self.assertTrue(np.isnan(sampled['view_scores'][:, 0, 0]).all())
        self.assertTrue(sampled['view_valid'][0, 1])
        self.assertGreater(diagnostics[0]['reason_counts']['partial_feature_support'], 0)
        self.assertLess(len(supports[0].references[0]), len(supports[1].references[0]))
        # Missing one footprint corner excludes the whole observation; it
        # never renormalizes surviving corner weights into a finite response.
        ids = np.flatnonzero(field.surface['observation_entity_member']
            & (field.surface['observation_view_index'] == 0))
        points = field.surface['observation_xyz'][ids]
        sampled = field.sample(points, np.full_like(points, np.nan))
        for row in range(3):
            np.testing.assert_array_equal(supports[0].references[row],
                np.sort(sampled['view_scores'][row, sampled['view_valid'][:, 0], 0]))

    def test_missing_camera_stays_empty_and_signed_constants_have_zero_support(self):
        for constant in (3.5, -2.75):
            with self.subTest(constant=constant):
                field = self.field(missing_second=True, constant=constant)
                supports, diagnostics = field.prepare_support()
                self.assertEqual(diagnostics[0]['constant_rows'], [0, 1, 2])
                self.assertEqual(diagnostics[1]['empty_rows'], [0, 1, 2])
                self.assertEqual(diagnostics[1]['reference_counts'], [0, 0, 0])
                self.assertEqual(diagnostics[1]['reason_counts'], {'missing_feature': 625})
                values = np.full(3, constant)
                np.testing.assert_array_equal(supports[0].utility(values, np.arange(3)), np.zeros(3))
                self.assertTrue(np.isnan(supports[1].utility(values, np.arange(3))).all())
                center = field.surface['pixel_to_observation'][0][12, 12]
                points = field.surface['observation_xyz'][[center]]
                sampled = field.sample(points, np.full_like(points, np.nan))
                self.assertEqual(sampled['view_state'][0, 1], 'missing_feature')
                self.assertFalse(sampled['view_valid'][0, 1])
                self.assertTrue(np.isnan(sampled['view_scores'][:, 0, 1]).all())
                np.testing.assert_array_equal(sampled['scores'], np.full((3, 1), constant))

    def test_stable_interpolation_matches_bilinear_values_and_keeps_native_boundary_rules(self):
        rng = np.random.default_rng(17)
        self.response = rng.normal(size=(3, 25, 25)).astype(np.float32)
        field = self.field()
        pixels = np.array([[0, 0], [0, 12], [12, 0], [23, 23], [24, 24]])
        ids = field.surface['pixel_to_observation'][0][pixels[:, 1], pixels[:, 0]]
        points = field.surface['observation_xyz'][ids]
        actual = field.sample(points, np.full_like(points, np.nan))
        np.testing.assert_array_equal(actual['view_valid'][:, 0], [True, True, True, True, False])
        self.assertEqual(actual['view_state'][-1, 0], 'partial_feature_support')
        # SciPy supplies an independent order-1 interpolation reference for a
        # nonlinear field. Only floating-point summation changes: the native
        # coordinate mapping and complete positive footprint remain intact.
        grid = (pixels[:4]*(840/25)*24/839).T[::-1]
        expected = np.array([map_coordinates(response.astype(float), grid,
            order=1, prefilter=False) for response in self.response])
        np.testing.assert_allclose(actual['view_scores'][:, :4, 0], expected,
            rtol=0, atol=5e-16)
# END references and candidates retain complete same-camera evidence semantics.


if __name__ == '__main__':
    unittest.main()
