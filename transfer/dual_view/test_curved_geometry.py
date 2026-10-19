"""CPU geometry contracts using the existing curved-region analytic examples.

These tests exercise missing observations, disconnected depth layers and frame
semantics, not a second implementation of the production predicates. No GPU,
learned model, simulator, target annotation or environment change is needed.
"""

import unittest
import tempfile

import numpy as np
from scipy.spatial.distance import pdist

from . import curved_geometry as curved
from ..sim3 import apply_sim3, fit_sim3
from manipulation.region_transfer import propagate_region_frame
from ..multiview_experiment import save_surface
from .io import target_surface
from .visualization import region_picture


def observation(depth, K, pose=None, region=None):
    """Create native RGB-D records with identity-preserving actual pixels."""
    pose = np.eye(4) if pose is None else pose
    yy, xx = np.nonzero(np.isfinite(depth) & (depth > 0))
    pixels = np.column_stack((xx, yy))
    xyz = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    xyz = xyz*depth[yy, xx, None] @ pose[:3, :3].T+pose[:3, 3]
    mask = np.isfinite(depth) & (depth > 0)
    region = mask if region is None else region
    return dict(xyz=xyz, pixels=pixels, K=K, T_world_camera_cv=pose,
        object_mask=mask, entity_member=np.ones(len(xyz), bool),
        region_mask=region[yy, xx], evidence_only=np.zeros(len(xyz), bool))


def views(depth, K):
    bank = observation(depth, K)
    return [dict(id=name, depth_m=depth, K=K, T_world_camera_cv=np.eye(4), observation=bank)
            for name in ('a', 'b')]


class CurvedGeometryTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[400., 0., 12.], [0., 400., 12.], [0., 0., 1.]])
        self.yy, self.xx = np.indices((25, 25))
        self.plane = .6/(1-.4*(self.xx-12)/400)

    def surface(self, depth):
        return curved.build_target_surface(views(depth, self.K))

    def test_curved_nonplanar_correspondences_recover_transform_without_normals(self):
        ray = (self.xx-12)/400
        radius, center = .03, .6
        depth = (center-np.sqrt(center**2-(1+ray**2)*(center**2-radius**2)))/(1+ray**2)
        source = observation(depth, self.K)
        ids = np.arange(len(source['xyz']))
        pack = curved.prepare_source(source, depth, query_indices=ids, query_region=np.ones(len(ids), bool))
        self.assertEqual(pack['reasons'], [])
        self.assertFalse(pack['query_normal_valid'].any())
        angle = .73
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        expected = (1.4, rotation, np.array([.2, -.1, .3]))
        target = apply_sim3(source['xyz'], *expected)
        actual = fit_sim3(source['xyz'], target)
        np.testing.assert_allclose(apply_sim3(source['xyz'], *actual), target, rtol=0, atol=1e-12)
        surface = self.surface(depth)
        seed = surface['observation_to_element'][surface['pixel_to_observation'][0][12, 12]]
        mode = curved.sample_region(surface, seed)
        self.assertGreater(mode['diagnostics']['sample_count'], 6)
        self.assertIsNone(mode['normal'])

    # BEGIN complete source structure remains distinct from sparse query roles.
    def test_structural_source_keeps_actual_entity_rows_and_calibrated_pitch(self):
        depth = np.full((25, 25), .6)
        region = (self.xx >= 10) & (self.xx <= 14) & (self.yy >= 10) & (self.yy <= 14)
        pose = np.eye(4)
        pose[:3, 3] = [1., -2., .3]
        source = observation(depth, self.K, pose=pose, region=region)
        source['entity_member'] = source['pixels'][:, 0] >= 4
        region_ids = np.flatnonzero(source['region_mask'])
        ids = region_ids[[0, 4, 10, 14, 20, 24]]
        pack = curved.prepare_source(source, depth, query_indices=ids, query_region=np.ones(6, bool))
        actual_ids = np.flatnonzero(source['entity_member'])
        self.assertGreater(len(pack['entity_indices']), len(pack['query_indices']))
        np.testing.assert_array_equal(pack['entity_indices'], actual_ids)
        np.testing.assert_array_equal(pack['entity_xyz'], source['xyz'][actual_ids])
        np.testing.assert_array_equal(pack['entity_region'], source['region_mask'][actual_ids])
        np.testing.assert_allclose(pack['entity_pixel_pitch_m'], .0015, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(pack['source_camera_position_world'], pose[:3, 3])
        self.assertTrue(np.all(pack['entity_position_error_m'] >= .0001))
        frozen = pack['entity_xyz'].copy()
        source['xyz'][:] = 0.
        np.testing.assert_array_equal(pack['entity_xyz'], frozen)
    # END original entity identity is not inferred from a query sampling budget.

    def test_step_edges_do_not_join_the_existing_8mm_or_40mm_counterexamples(self):
        for jump in (.008, .04):
            with self.subTest(jump=jump):
                depth = np.where(self.xx < 12, .6, .6+jump)
                surface = self.surface(depth)
                seed = surface['observation_to_element'][surface['pixel_to_observation'][0][12, 11]]
                mode = curved.sample_region(surface, seed)
                self.assertTrue(np.all(surface['xyz'][mode['mask'], 2] == .6))
                self.assertGreater(surface['diagnostics']['rejected_pixel_edges'], 0)

    def test_missing_column_stays_disconnected_despite_short_diagonal_distance(self):
        depth = np.full((25, 25), .6)
        depth[:, 12] = 0.
        surface = self.surface(depth)
        seed = surface['observation_to_element'][surface['pixel_to_observation'][0][12, 11]]
        mode = curved.sample_region(surface, seed)
        pixels = surface['observation_pixels'][surface['representative_observation'][mode['mask']]]
        self.assertTrue(np.all(pixels[:, 0] < 12))
        self.assertTrue(np.all(surface['pixel_to_observation'][0][:, 12] == -1))

    def test_sampling_uses_actual_points_fixed_spacing_and_graph_radius(self):
        surface = self.surface(self.plane)
        seed = surface['observation_to_element'][surface['pixel_to_observation'][0][12, 12]]
        mode = curved.sample_region(surface, seed)
        samples = surface['xyz'][mode['sample_indices']]
        original = surface['observation_xyz'][surface['representative_observation'][mode['sample_indices']]]
        np.testing.assert_array_equal(samples, original)
        self.assertGreaterEqual(float(pdist(samples).min()), .005*(1-1e-12))
        self.assertLessEqual(mode['diagnostics']['maximum_graph_distance_m'], .03*(1+1e-12))
        self.assertFalse(mode['diagnostics']['sampling_scale_applied'])
        self.assertEqual(surface['diagnostics']['paired_element_count'], 625)

    def test_projection_distinguishes_occlusion_conflict_and_no_depth(self):
        depth = np.full((25, 25), .6)
        depth[4, 4] = 0.
        surface = self.surface(depth)
        p = np.array([[0, 0, .6], [0, 0, .61], [0, 0, .59], [-.012, -.012, .6]])
        record = curved.project_measurements(surface, p, 0)
        self.assertEqual([record['state_names'][i] for i in record['states']],
                         ['measurement_supported', 'occluded', 'depth_model_conflict', 'invalid_depth'])
        self.assertEqual(record['supported'].tolist(), [True, False, False, False])
        self.assertEqual(record['conflict'].tolist(), [False, False, True, False])

    def test_exact_half_pixel_association_remains_ambiguous(self):
        surface = self.surface(np.full((25, 25), .6))
        record = curved.project_measurements(surface, np.array([[.5*.6/400, 0, .6]]), 0)
        self.assertFalse(record['supported'][0])
        self.assertEqual(record['state_names'][record['states'][0]], 'pixel_association_ambiguous')

    # BEGIN actual-point support at depth boundaries without bridging surfaces.
    def test_direct_position_support_survives_a_cut_without_borrowing_pixel_allowance(self):
        surface = self.surface(np.where(self.xx < 12, .6, .64))
        point = np.array([[(11.2-12)*.6/400, 0, .6]])
        direct = curved.project_measurements(surface, point, 0, registration_error_m=.00025)
        self.assertTrue(direct['supported'][0])
        self.assertEqual(direct['support_kind'][0], 'direct_measurement')
        self.assertLess(direct['residual_m'][0], direct['direct_association_bound_m'][0])
        self.assertEqual(direct['association_bound_m'][0], direct['direct_association_bound_m'][0])
        selected = surface['pixel_to_observation'][0][12, 11]
        self.assertEqual(direct['observation_indices'][0], selected)

        # With no registration budget the same actual point is too far away.
        # A slope-derived rounding allowance cannot make the cut continuous.
        unresolved = curved.project_measurements(surface, point, 0)
        self.assertGreater(unresolved['residual_m'][0], unresolved['direct_association_bound_m'][0])
        self.assertLess(unresolved['residual_m'][0], unresolved['association_bound_m'][0])
        self.assertTrue(unresolved['unknown'][0])
        self.assertEqual(unresolved['support_kind'][0], 'none')

        across = surface['pixel_to_observation'][0][12, 12]
        self.assertEqual(surface['depth_adjacency'][selected, across], 0)
        members, _ = curved.bounded_members(surface, int(selected), .03, view_index=0)
        self.assertNotIn(across, members)

    def test_direct_position_support_does_not_fabricate_feature_corner_support(self):
        from .curved_features import SurfaceResponseField

        depth = np.where(self.xx < 12, .6, .64)
        declared = views(depth, self.K)
        surface = curved.build_target_surface(declared)
        queries = np.ones((1, 1), dtype=np.float32)
        # The coarse feature cell around pixel (11,12) has positive corners on
        # both sides of the cut. Position compatibility must not join them.
        prepared = [dict(id=view['id'], query_descriptors=queries,
            observation=view['observation'], response=np.ones((1, 3, 3), dtype=np.float32),
            crop_xyxy=[0, 0, 25, 25]) for view in declared]
        field = SurfaceResponseField(surface, queries, prepared)
        point = np.array([[(11.2-12)*.6/400, 0, .6]])
        result = field.sample(point, np.full_like(point, np.nan), registration_error_m=.00025)
        self.assertTrue(all(record['supported'][0] for record in result['geometry']))
        self.assertFalse(result['known'].any())
        self.assertTrue(np.all(result['view_state'] == 'partial_feature_support'))

    def test_continuous_pixel_model_remains_distinct_from_direct_support(self):
        surface = self.surface(np.full((25, 25), .6))
        point = np.array([[.2*.6/400, 0, .6]])
        result = curved.project_measurements(surface, point, 0)
        self.assertTrue(result['supported'][0])
        self.assertEqual(result['support_kind'][0], 'continuous_pixel_model')
        self.assertGreater(result['residual_m'][0], result['direct_association_bound_m'][0])

    def test_direct_budget_does_not_resolve_an_exact_half_pixel_tie(self):
        surface = self.surface(np.full((25, 25), .6))
        point = np.array([[.5*.6/400, 0, .6]])
        result = curved.project_measurements(surface, point, 0, registration_error_m=.002)
        self.assertLess(result['residual_m'][0], result['direct_association_bound_m'][0])
        self.assertEqual(result['state_names'][result['states'][0]], 'pixel_association_ambiguous')
        self.assertEqual(result['support_kind'][0], 'none')
        self.assertTrue(result['unknown'][0])

    def test_direct_budget_cannot_replace_a_missing_original_pixel(self):
        depth = np.full((25, 25), .6)
        depth[:, 12] = 0.
        surface = self.surface(depth)
        point = np.array([[(11.8-12)*.6/400, 0, .6]])
        # Pixel 11 has a nearby measured point inside this uncertainty bound;
        # pixel 12 is the fixed association and has no measurement to support.
        result = curved.project_measurements(surface, point, 0, registration_error_m=.002)
        self.assertEqual(result['state_names'][result['states'][0]], 'invalid_depth')
        self.assertEqual(result['observation_indices'][0], -1)
        self.assertEqual(result['support_kind'][0], 'none')
        self.assertTrue(result['unknown'][0])
    # END direct support changes no graph, feature footprint or tie convention.

    def test_axial_conflict_is_not_overridden_by_wider_world_distance_bound(self):
        surface = self.surface(np.full((25, 25), .6))
        point = np.array([[.49*.5975/400, 0, .5975]])
        result = curved.project_measurements(surface, point, 0, query_position_error_m=.0001)
        self.assertLess(result['residual_m'][0], result['association_bound_m'][0])
        self.assertTrue(result['conflict'][0])
        self.assertFalse(result['supported'][0])

    def test_subpixel_across_cut_depth_cell_has_no_conflict_certificate(self):
        surface = self.surface(np.where(self.xx < 12, .6, .608))
        point = np.array([[(-.4)*.6/400, 0, .6]])
        result = curved.project_measurements(surface, point, 0)
        self.assertTrue(result['unknown'][0])
        self.assertFalse(result['conflict'][0])
        self.assertFalse(result['supported'][0])
        # The integer ray really measures the farther layer, so comparison at
        # that same ray remains a model conflict rather than subpixel unknown.
        self.assertTrue(curved.project_measurements(surface, [[0, 0, .6]], 0)['conflict'][0])

    def test_lateral_position_uncertainty_can_cross_nominal_integer_ray_depth_cut(self):
        surface = self.surface(np.where(self.xx < 12, .6, .64))
        exact = curved.project_measurements(surface, [[0, 0, .6]], 0)
        uncertain = curved.project_measurements(surface, [[0, 0, .6]], 0, registration_error_m=.003)
        self.assertTrue(exact['conflict'][0])
        self.assertFalse(uncertain['conflict'][0])
        self.assertTrue(uncertain['unknown'][0])
        # The same bound wholly inside a continuous background patch still
        # supplies negative evidence; uncertainty does not erase all conflicts.
        interior = curved.project_measurements(surface, [[10*.6/400, 0, .6]], 0,
                                               registration_error_m=.001)
        self.assertTrue(interior['conflict'][0])

    def test_explicit_query_error_is_not_replaced_by_target_camera_error(self):
        surface = self.surface(np.full((25, 25), .6))
        point = [[0, 0, .6004]]
        self.assertFalse(curved.project_measurements(surface, point, 0)['supported'][0])
        result = curved.project_measurements(surface, point, 0, query_position_error_m=.0005)
        self.assertTrue(result['supported'][0])
        self.assertAlmostEqual(result['association_bound_m'][0], .0006)
        with self.assertRaises(ValueError):
            curved.project_measurements(surface, point, 0, query_position_error_m=-1.)

    def test_entity_diagonal_cannot_cross_nonentity_corners(self):
        depth = np.full((2, 2), .6)
        declared = views(depth, self.K)
        for view in declared:
            view['observation']['entity_member'][:] = [True, False, False, True]
        surface = curved.build_target_surface(declared)
        seed = surface['observation_to_element'][0]
        mode = curved.sample_region(surface, seed)
        self.assertEqual(int(mode['mask'].sum()), 1)
        self.assertGreater(surface['depth_adjacency'][0, 3], 0)
        self.assertEqual(surface['observation_adjacency'][0, 3], 0)

    def test_intersection_resampling_retains_connectivity_and_covers_remaining_points(self):
        surface = self.surface(np.full((25, 25), .6))
        pixels = surface['observation_pixels'][surface['representative_observation']]
        members = np.flatnonzero((pixels[:, 0] < 8) | (pixels[:, 0] > 16))
        seed = surface['observation_to_element'][surface['pixel_to_observation'][0][12, 4]]
        mode = curved.sample_members(surface, members, seed)
        self.assertTrue(np.all(pixels[mode['mask'], 0] < 8))
        from scipy.spatial import cKDTree
        distance = cKDTree(surface['xyz'][mode['sample_indices']]).query(surface['xyz'][mode['mask']])[0]
        self.assertLessEqual(distance.max(), .005)

    def test_intersection_radius_keeps_fixed_graph_metric_for_all_tied_seeds(self):
        from scipy.sparse import coo_matrix
        from scipy.spatial import cKDTree
        # A-B-C is 1.2 radii long, but the full graph also has A-U-C=.9r.
        # U lies outside B's ball. Removing U from the intersection must not
        # redefine its already established distances or make A/B order matter.
        points = .03*np.array([[0., 0., 0.], [.4, 0., 0.], [0., np.sqrt(.48), 0.],
                              [np.sqrt(.055), .84/(2*np.sqrt(.48)), 0.]])
        a, b = np.array([0, 1, 0, 3]), np.array([1, 2, 3, 2])
        weights = np.linalg.norm(points[a]-points[b], axis=1)
        graph = coo_matrix((np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])), shape=(4, 4)).tocsr()
        surface = dict(xyz=points, entity_member=np.ones(4, bool), adjacency=graph,
                       _entity_ids=np.arange(4), _entity_tree=cKDTree(points))
        balls = [geometry for geometry, _ in (curved.bounded_members(surface, seed, .03) for seed in (0, 1))]
        common = np.intersect1d(*balls)
        np.testing.assert_array_equal(common, [0, 1, 2])
        for seed in (0, 1):
            mode = curved.sample_members(surface, common, seed)
            np.testing.assert_array_equal(np.flatnonzero(mode['mask']), common)
            self.assertLessEqual(mode['diagnostics']['maximum_graph_distance_m'], .03)

    def test_curved_cache_roundtrip_and_normal_free_visualization(self):
        declared = views(self.plane, self.K)
        original = curved.build_target_surface(declared)
        with tempfile.TemporaryDirectory() as directory:
            save_surface(directory, original)
            restored = target_surface(dict(region_geometry='surface', views=declared), directory)
            with self.assertRaises(ValueError):
                target_surface(dict(region_geometry='planar', views=declared), directory)
        for name in ('adjacency', 'observation_adjacency', 'depth_adjacency'):
            self.assertEqual((restored[name] != original[name]).nnz, 0)
        seed = restored['observation_to_element'][restored['pixel_to_observation'][0][12, 12]]
        mode = curved.sample_region(restored, seed)
        picture, counts = region_picture(np.zeros((25, 25, 3), np.uint8), restored, mode, 0)
        self.assertEqual(picture.size, (25, 25))
        self.assertEqual(counts['visible_samples'], len(mode['sample_indices']))

    def test_medoid_ties_are_not_resolved_by_source_index(self):
        depth = np.full((4, 4), .6)
        K = np.array([[400., 0., 1.5], [0., 400., 1.5], [0., 0., 1.]])
        source = observation(depth, K)
        ids = np.arange(16)
        pack = curved.prepare_source(source, depth, query_indices=ids, query_region=np.ones(16, bool))
        self.assertEqual(len(pack['anchor_indices']), 4)
        self.assertEqual(len(pack['query_indices']), 16)

    def test_functional_frame_propagates_without_a_surface_normal(self):
        angle = .7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        frame = np.eye(4)
        frame[:3, 3] = [.3, -.2, .9]
        source = dict(T_world_body_at_image=np.eye(4), L_body_region=frame,
                      owner_id='body', timestamp_s=1.)
        transform = dict(scale=1.4, rotation=rotation, translation=np.array([.1, .2, .3]), hypothesis_id=0)
        result = propagate_region_frame(source, transform)['T_world_region']
        np.testing.assert_allclose(result[:3, :3], rotation)
        np.testing.assert_allclose(result[:3, 3], 1.4*rotation@frame[:3, 3]+transform['translation'])
        np.testing.assert_array_equal(source['L_body_region'], frame)


if __name__ == '__main__':
    unittest.main()
