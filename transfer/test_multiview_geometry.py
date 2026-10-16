"""CPU contracts for two calibrated depth observations and one surface graph.

The fixtures are small analytic RGB-D arrays, not model outputs or repository
datasets. Geometry exists independently of an entity mask or appearance cache.
"""

import copy
import unittest

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from transfer.core import TransferConfig
from transfer.multiview_geometry import (build_target_surface, check_points,
                                         project_observations, sample_region)


def calibrated_view(view_id, depth=None, *, mask=None, K=None, T=None,
                    observation_available=True):
    """Provide full depth plus a separately indexed optional entity observation."""
    depth = np.ones((9, 9), dtype=float) if depth is None else np.asarray(depth, dtype=float)
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    mask = valid.copy() if mask is None else np.asarray(mask, dtype=bool).copy()
    K = (np.array([[500., 0., (width-1)/2], [0., 500., (height-1)/2], [0., 0., 1.]])
         if K is None else np.asarray(K, dtype=float))
    T = np.eye(4) if T is None else np.asarray(T, dtype=float)
    observation = None
    if observation_available:
        y, x = np.nonzero(mask & valid)
        pixels = np.column_stack((x, y)).astype(np.int64)
        rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
        xyz = (rays * depth[y, x, None]) @ T[:3, :3].T + T[:3, 3]
        observation = {'xyz': xyz, 'pixels': pixels, 'K': K.copy(),
                       'T_world_camera_cv': T.copy(), 'object_mask': mask,
                       'entity_member': np.ones(len(pixels), bool),
                       'region_mask': np.zeros(len(pixels), bool),
                       'evidence_only': np.zeros(len(pixels), bool)}
    return {'id': view_id, 'K': K.copy(), 'T_world_camera_cv': T.copy(),
            'depth_m': depth.copy(), 'observation': observation, 'evidence': None}


def plane_depth(shape, K, T, point, normal):
    """Intersect calibrated world rays with an analytic plane for the fixture."""
    yy, xx = np.indices(shape)
    pixels = np.column_stack((xx.ravel(), yy.ravel(), np.ones(xx.size)))
    rays = (pixels @ np.linalg.inv(K).T) @ T[:3, :3].T
    numerator = np.dot(np.asarray(point)-T[:3, 3], normal)
    return (numerator/(rays @ normal)).reshape(shape)


def element_at(surface, view_index, x, y):
    """Resolve a known raw pixel through the public provenance arrays."""
    observation = int(surface['pixel_to_observation'][view_index][y, x])
    if observation < 0:
        raise AssertionError('The fixture requested an invalid depth pixel')
    return int(surface['observation_to_element'][observation])


class TestTargetSurface(unittest.TestCase):
    def test_full_depth_survives_small_or_missing_entity_observations(self):
        mask = np.zeros((9, 9), bool)
        mask[3:6, 3:6] = True
        a = calibrated_view('a', mask=mask)
        b = calibrated_view('b', observation_available=False)
        original = copy.deepcopy((a, b))
        surface = build_target_surface([b, a])
        self.assertEqual(surface['view_ids'], ['a', 'b'])
        self.assertEqual(len(surface['observation_xyz']), 162)
        self.assertEqual(surface['view_offsets'].tolist(), [0, 81, 162])
        expected_entity = np.zeros(len(surface['xyz']), bool)
        for x, y in a['observation']['pixels']:
            expected_entity[element_at(surface, 0, x, y)] = True
        np.testing.assert_array_equal(surface['entity_member'], expected_entity)
        self.assertTrue(np.any(~surface['entity_member']))
        for before, after in zip(original, (a, b)):
            np.testing.assert_array_equal(before['depth_m'], after['depth_m'])
        self.assertEqual(len(surface['observation_to_element']), 162)

    def test_groups_have_at_most_one_member_from_each_view_and_real_representatives(self):
        a, b = calibrated_view('a'), calibrated_view('b')
        surface = build_target_surface([a, b])
        count = np.bincount(surface['observation_to_element'], minlength=len(surface['xyz']))
        self.assertTrue(np.any(count == 2))
        self.assertTrue(np.all((count >= 1) & (count <= 2)))
        for index in range(2):
            own = surface['observation_view_index'] == index
            per_view = np.bincount(surface['observation_to_element'][own], minlength=len(count))
            self.assertTrue(np.all(per_view <= 1))
            np.testing.assert_array_equal(surface['member_views'][:, index], per_view == 1)
        np.testing.assert_array_equal(surface['xyz'],
            surface['observation_xyz'][surface['representative_observation']])
        reverse = build_target_surface([b, a])
        for key in ('xyz', 'observation_to_element', 'representative_observation', 'member_views', 'entity_member'):
            np.testing.assert_array_equal(surface[key], reverse[key])
        self.assertEqual((surface['adjacency'] != reverse['adjacency']).nnz, 0)

    def test_observed_by_both_is_distinct_from_having_a_paired_member(self):
        a = calibrated_view('a')
        K = a['K'].copy()
        K[:2, :2] *= .5
        b = calibrated_view('b', K=K)
        surface = build_target_surface([a, b])
        both_visible = np.all(surface['observed_views'], axis=1)
        single_member = surface['member_views'].sum(axis=1) == 1
        self.assertTrue(np.any(both_visible & single_member))
        self.assertTrue(np.all(np.bincount(surface['observation_to_element']) <= 2))

    def test_opposite_sides_and_distinct_parallel_layers_do_not_merge(self):
        a = calibrated_view('a')
        opposite = np.diag([1., -1., -1., 1.])
        opposite[2, 3] = 2.
        alternatives = (calibrated_view('b', T=opposite),
                        calibrated_view('b', depth=np.full((9, 9), 1.02)))
        for b in alternatives:
            with self.subTest(camera=b['T_world_camera_cv'].tolist()):
                surface = build_target_surface([a, b])
                self.assertTrue(np.all(np.bincount(surface['observation_to_element']) == 1))
                self.assertEqual(len(surface['xyz']), 162)

    def test_missing_depth_gap_stays_disconnected_until_another_view_observes_it(self):
        depth = np.ones((9, 9))
        depth[4] = np.nan
        a = calibrated_view('a', depth)
        separated = build_target_surface([a, calibrated_view('b', depth)])
        self.assertEqual(connected_components(separated['adjacency'], directed=False)[0], 2)
        completed = build_target_surface([a, calibrated_view('b')])
        self.assertEqual(connected_components(completed['adjacency'], directed=False)[0], 1)
        b_only = completed['member_views'][:, 1] & ~completed['member_views'][:, 0]
        self.assertTrue(np.any(b_only))

    def test_collinear_or_invalid_depth_is_retained_without_valid_surface_claim(self):
        depth = np.full((9, 9), np.nan)
        depth[4] = 1.
        surface = build_target_surface([calibrated_view('a', depth), calibrated_view('b', depth)])
        self.assertEqual(len(surface['observation_xyz']), 18)
        self.assertFalse(np.any(surface['valid']))
        empty = np.array([[np.nan, 0., -1., np.inf]])
        surface = build_target_surface([calibrated_view('a', empty), calibrated_view('b', empty)])
        self.assertEqual(surface['xyz'].shape, (0, 3))
        self.assertEqual(surface['adjacency'].shape, (0, 0))

    def test_arbitrary_rigid_world_coordinates_preserve_observation_membership(self):
        a, b = calibrated_view('a'), calibrated_view('b')
        original = build_target_surface([a, b])
        rigid = np.eye(4)
        rigid[:3, :3] = Rotation.from_rotvec([.35, -.4, .2]).as_matrix()
        rigid[:3, 3] = [.2, -.3, .4]
        moved = build_target_surface([calibrated_view('a', T=rigid), calibrated_view('b', T=rigid)])
        np.testing.assert_array_equal(original['observation_to_element'], moved['observation_to_element'])
        np.testing.assert_allclose(moved['xyz'], original['xyz'] @ rigid[:3, :3].T + rigid[:3, 3],
                                   rtol=0, atol=1e-12)
        np.testing.assert_array_equal(original['entity_member'], moved['entity_member'])


class TestProjectionAndSampling(unittest.TestCase):
    def test_surface_support_free_space_and_occlusion_remain_distinct(self):
        surface = build_target_surface([calibrated_view('a'), calibrated_view('b')])
        points = np.array([[0., 0., 1.], [0., 0., .98], [0., 0., 1.02],
                           [1., 0., 1.], [0., 0., -1.]])
        projected = project_observations(surface, points, 0)
        np.testing.assert_array_equal(projected['supported'], [True, False, False, False, False])
        np.testing.assert_array_equal(projected['conflict'], [False, True, False, False, False])
        np.testing.assert_array_equal(projected['unknown'], [False, False, True, True, True])
        checked = check_points(surface, points, tolerance_m=0)
        np.testing.assert_array_equal(checked['support'], projected['supported'])
        np.testing.assert_array_equal(checked['conflict'], projected['conflict'])
        np.testing.assert_array_equal(checked['unknown'], projected['unknown'])

    def test_four_corner_depth_gap_is_unknown_but_entity_mask_gap_is_not(self):
        pixel = np.array([4.4, 4.4])
        base = calibrated_view('a')
        point = (np.linalg.inv(base['K']) @ np.r_[pixel, 1.])[None]
        mask = np.ones((9, 9), bool)
        mask[5, 5] = False
        surface = build_target_surface([calibrated_view('a', mask=mask), calibrated_view('b')])
        self.assertTrue(project_observations(surface, point, 0)['supported'][0])
        depth = np.ones((9, 9))
        depth[5, 5] = np.nan
        surface = build_target_surface([calibrated_view('a', depth), calibrated_view('b')])
        projected = project_observations(surface, point, 0)
        self.assertFalse(projected['supported'][0])
        self.assertFalse(projected['conflict'][0])
        self.assertTrue(projected['unknown'][0])

    def test_slanted_plane_support_uses_plane_distance(self):
        K = np.array([[1000., 0., 4.], [0., 1000., 4.], [0., 0., 1.]])
        normal = np.array([1., 0., .055])
        normal /= np.linalg.norm(normal)
        depth = plane_depth((9, 9), K, np.eye(4), [0., 0., 1.], normal)
        a = calibrated_view('a', depth, K=K)
        surface = build_target_surface([a, calibrated_view('b', depth, K=K)])
        pixel = np.array([4.4, 4.3])
        ray = np.linalg.inv(K) @ np.r_[pixel, 1.]
        point = ray * (normal[2]/np.dot(ray, normal))
        projected = project_observations(surface, point[None], 0)
        self.assertTrue(projected['supported'][0])
        np.testing.assert_allclose(projected['pixels'][0], pixel, rtol=0, atol=1e-12)

    def test_sampling_keeps_fixed_metric_radius_and_actual_representatives(self):
        K = np.array([[250., 0., 10.], [0., 250., 10.], [0., 0., 1.]])
        depth = np.ones((21, 21))
        surface = build_target_surface([calibrated_view('a', depth, K=K),
                                        calibrated_view('b', depth, K=K)])
        seed = element_at(surface, 0, 10, 10)
        mode = sample_region(surface, surface['valid'] & surface['entity_member'], seed,
                             sampling_radius_m=.012, sampling_spacing_m=.005,
                             config=TransferConfig())
        members = surface['xyz'][mode['mask']]
        self.assertTrue(mode['mask'][seed])
        self.assertTrue(np.all(np.linalg.norm(members-surface['xyz'][seed], axis=1) <= .012*(1+1e-12)))
        samples = np.asarray(mode['sample_indices'])
        self.assertTrue(np.all(mode['mask'][samples]))
        self.assertGreater(len(samples), 1)
        self.assertGreaterEqual(pdist(surface['xyz'][samples]).min(), .005*(1-1e-12))
        np.testing.assert_array_equal(surface['xyz'][samples],
            surface['observation_xyz'][surface['representative_observation'][samples]])


if __name__ == '__main__':
    unittest.main()
