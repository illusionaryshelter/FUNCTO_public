"""Proposal contracts: bounded final poses, actual geometry and missing data.

The synthetic observations have no target labels or stored predicted poses.
Only the global solver is replaced in the slot test so the real native ICP
must recover a non-unit scale using the exact entity measurement arrays.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from . import curved_proposals as proposals
from . import curved_geometry as geometry
from .test_curved_geometry import observation


class CurvedProposalTests(unittest.TestCase):
    def test_camera_line_neighborhood_stays_missing_without_default_normals(self):
        # Curved depth along one image scanline has 3-D rank two, but cannot
        # identify a surface normal. No axis fallback or FPFH call is legal.
        t = np.linspace(-.01, .01, 15)
        points = np.column_stack((t, np.zeros(len(t)), .6+5*t*t))
        pixels = np.column_stack((np.arange(len(t)), np.zeros(len(t))))
        with patch.object(o3d.pipelines.registration, 'compute_fpfh_feature',
                          side_effect=AssertionError('missing normals cannot form a feature')):
            cloud, feature, report = proposals._descriptor(points, pixels, np.zeros(3), .02)
        self.assertEqual(report['normal_known_count'], 0)
        self.assertEqual(len(cloud.points), 0)
        self.assertEqual(np.asarray(feature.data).shape, (33, 0))

    def test_source_proposal_domain_preserves_region_outside_entity(self):
        depth = np.full((9, 9), .6)
        K = np.array([[400., 0., 4.], [0., 400., 4.], [0., 0., 1.]])
        region = np.zeros(depth.shape, bool)
        region[2:7, 2:7] = True
        bank = observation(depth, K, region=region)
        bank['entity_member'] = bank['pixels'][:, 0] >= 4
        indices = np.flatnonzero(bank['region_mask'])
        prepared = geometry.prepare_source(bank, depth, query_indices=indices,
            query_region=np.ones(len(indices), bool))
        selected = bank['entity_member'] | bank['region_mask']
        np.testing.assert_array_equal(prepared['proposal_xyz'], bank['xyz'][selected])
        np.testing.assert_array_equal(prepared['proposal_pixels'], bank['pixels'][selected])
        self.assertGreater(len(prepared['proposal_xyz']), len(prepared['entity_xyz']))
        self.assertFalse(np.shares_memory(prepared['proposal_xyz'], bank['xyz']))

    @staticmethod
    def inputs():
        """Prepared-shape records isolate native fitting from image rendering."""
        points = np.array([[0., 0., .2], [.3, 0., .2], [0., .4, .2], [0., 0., .7]])
        pixels = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        target = 1.4*points+np.array([.2, -.1, .3])
        source = dict(region_geometry='surface', entity_xyz=points,
            proposal_xyz=points.copy(), proposal_pixels=pixels,
            entity_pixel_pitch_m=np.full(4, .01), source_camera_position_world=np.zeros(3),
            size=float(np.sqrt(np.mean(np.sum((points-points.mean(0))**2, axis=1)))))
        surface = dict(region_geometry='surface', view_ids=['a', 'b'],
            observation_xyz=np.concatenate([target, target]), observation_pixels=np.tile(pixels, (2, 1)),
            observation_entity_member=np.ones(8, bool), observation_view_index=np.repeat([0, 1], 4),
            observation_spacing=np.full(8, .01), T_world_camera_cv=np.tile(np.eye(4), (2, 1, 1)),
            xyz=np.concatenate([target, [[100., 100., 100.]]]),
            entity_member=np.array([True, True, True, True, False]))
        return source, surface, target

    @staticmethod
    def descriptor(points, pixels, camera, voxel):
        feature = o3d.pipelines.registration.Feature()
        feature.data = np.ones((33, len(points)))
        return proposals._cloud(points), feature, dict(input_count=len(points),
            normal_known_count=len(points), voxel_point_count=len(points))

    def test_native_icp_updates_same_32_slots_with_free_scale_and_actual_points(self):
        source, surface, target = self.inputs()
        coarse = np.eye(4)
        coarse[:3, :3] *= 1.399
        coarse[:3, 3] = [.2, -.1, .3]
        global_result = SimpleNamespace(transformation=coarse, fitness=1., inlier_rmse=.01,
            correspondence_set=np.column_stack((np.arange(4), np.arange(4))))
        pairs = o3d.utility.Vector2iVector(global_result.correspondence_set)
        with patch.object(proposals, '_descriptor', side_effect=self.descriptor), \
             patch.object(o3d.pipelines.registration, 'correspondences_from_features', return_value=pairs), \
             patch.object(o3d.pipelines.registration, 'registration_ransac_based_on_correspondence',
                          return_value=global_result) as native:
            actual, report = proposals.build_proposals(source, surface)
        self.assertEqual(len(actual), 32)
        self.assertEqual(native.call_count, 32)
        self.assertEqual(report['global_fit_calls'], 32)
        self.assertEqual(report['local_fit_calls'], 32)
        self.assertEqual(report['aggregate_ransac_iteration_upper_bound'], 3200000)
        self.assertEqual(report['appended_refit_candidates'], 0)
        self.assertEqual([row['proposal_id'] for row in actual], list(range(32)))
        for row in actual:
            scale, rotation, translation = row['transform']
            self.assertAlmostEqual(scale, 1.4, places=12)
            np.testing.assert_allclose(scale*source['entity_xyz']@rotation.T+translation,
                target, rtol=0, atol=1e-12)
            self.assertNotEqual(scale, 1.399)

    def test_invalid_refinement_marks_slot_failed_without_retaining_coarse_pose(self):
        source, surface, _ = self.inputs()
        pairs = o3d.utility.Vector2iVector(np.column_stack((np.arange(4), np.arange(4))))
        coarse = SimpleNamespace(transformation=np.eye(4), fitness=1., inlier_rmse=0.,
            correspondence_set=pairs)
        invalid_matrix = np.eye(4)
        invalid_matrix[:3, :3] *= .01
        invalid = SimpleNamespace(transformation=invalid_matrix, fitness=1., inlier_rmse=0.,
            correspondence_set=pairs)
        with patch.object(proposals, '_descriptor', side_effect=self.descriptor), \
             patch.object(o3d.pipelines.registration, 'correspondences_from_features', return_value=pairs), \
             patch.object(o3d.pipelines.registration, 'registration_ransac_based_on_correspondence', return_value=coarse), \
             patch.object(o3d.pipelines.registration, 'registration_icp', return_value=invalid):
            actual, report = proposals.build_proposals(source, surface)
        self.assertEqual(actual, [])
        self.assertEqual(len(report['slots']), 32)
        self.assertTrue(all(row['failure'] == 'refined_result_outside_proper_Sim3_domain'
                            for row in report['slots']))

    def test_absent_source_entity_consumes_no_native_fit_and_stays_missing(self):
        source, surface, _ = self.inputs()
        source.update(entity_xyz=np.empty((0, 3)), entity_pixel_pitch_m=np.empty(0))
        with patch.object(o3d.pipelines.registration, 'registration_ransac_based_on_correspondence',
                          side_effect=AssertionError('missing geometry cannot call RANSAC')):
            actual, report = proposals.build_proposals(source, surface)
        self.assertEqual(actual, [])
        self.assertEqual(report['global_fit_calls'], 0)
        self.assertEqual(len(report['slots']), 32)
        self.assertTrue(all(row['failure'] == 'insufficient_observed_entity_geometry'
                            for row in report['slots']))

    def test_native_empty_consensus_identity_cannot_start_icp(self):
        source, surface, _ = self.inputs()
        pairs = o3d.utility.Vector2iVector(np.column_stack((np.arange(4), np.arange(4))))
        failure = SimpleNamespace(transformation=np.eye(4), fitness=0., inlier_rmse=0.,
            correspondence_set=o3d.utility.Vector2iVector())
        with patch.object(proposals, '_descriptor', side_effect=self.descriptor), \
             patch.object(o3d.pipelines.registration, 'correspondences_from_features', return_value=pairs), \
             patch.object(o3d.pipelines.registration, 'registration_ransac_based_on_correspondence', return_value=failure), \
             patch.object(o3d.pipelines.registration, 'registration_icp',
                          side_effect=AssertionError('native failure identity is not an initializer')):
            actual, report = proposals.build_proposals(source, surface)
        self.assertEqual(actual, [])
        self.assertEqual(report['global_fit_calls'], 32)
        self.assertEqual(report['local_fit_calls'], 0)
        self.assertTrue(all(row['failure'] == 'global_result_has_no_correspondence_consensus'
                            for row in report['slots']))


if __name__ == '__main__':
    unittest.main()
