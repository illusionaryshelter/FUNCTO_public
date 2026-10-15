"""Independent contracts for graph scoring, multi-point fits and observed regions."""

import os
import subprocess
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from transfer.core import TransferConfig
from transfer.joint_features import source_queries
from transfer.joint_region import (JointConfig, graph_scores, infer_joint_region,
                                   multipoint_proposals, partial_hypothesis, select_observation)
from transfer.test_core import _grid_pixels, _surface_fixture


class JointRegionTests(unittest.TestCase):
    def test_complete_graph_matches_scalar_definition_and_similarity(self):
        rng = np.random.default_rng(41)
        queries, centers = rng.normal(size=(9, 3)), rng.normal(size=(7, 3))
        source_center = np.array([.3, -.2, .1])
        target = queries.copy()
        assignment = np.array([0, 1, -1, 3, 4, -1, 6, 7, 8])
        appearance = np.full((9, 9), .17)
        expected = []
        for center in centers:
            value = (7*.17+2*.5)/9
            for i in range(9):
                for k in range(i+1, 9):
                    if assignment[i] < 0 or assignment[k] < 0:
                        continue
                    a = np.array([np.linalg.norm(source_center-queries[i]),
                                  np.linalg.norm(source_center-queries[k]), np.linalg.norm(queries[i]-queries[k])])
                    b = np.array([np.linalg.norm(center-target[i]),
                                  np.linalg.norm(center-target[k]), np.linalg.norm(target[i]-target[k])])
                    value += np.abs(a/a.sum()-b/b.sum()).sum()/2/36
            expected.append(value)
        actual = graph_scores(centers, source_center, queries, target, assignment, appearance,
                              config=JointConfig(center_chunk=2, edge_chunk=3))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        rotation = Rotation.from_rotvec([.4, -.3, .2]).as_matrix()
        moved = lambda x: 1.25*x@rotation.T+[.5, .7, -.2]
        transformed = graph_scores(moved(centers), source_center, queries, moved(target), assignment, appearance)
        np.testing.assert_allclose(transformed, actual, atol=1e-14)
        truth_score = graph_scores(moved(source_center[None]), source_center, queries,
                                  moved(target), np.arange(9), appearance)[0]
        self.assertAlmostEqual(truth_score, .17)

    def test_mask_context_isolation_and_source_query_identity(self):
        rng = np.random.default_rng(12)
        source = {'xyz': rng.normal(size=(40, 3)), 'region_mask': np.arange(40) < 15,
                  'entity_member': (np.arange(40) >= 3) & (np.arange(40) < 30)}
        ids, region = source_queries(source, 10, 10)
        self.assertTrue(np.all(source['entity_member'][ids[~region]]))
        self.assertFalse(np.any(source['region_mask'][ids[~region]]))
        self.assertTrue(np.any(~source['entity_member'][ids[region]]))
        other = {**source, 'arbitrary_task_name': 'renamed'}
        np.testing.assert_array_equal(source_queries(other, 10, 10)[0], ids)

    def test_multipoint_proposals_use_six_or_twelve_and_fit_real_transform(self):
        rng = np.random.default_rng(25)
        source = rng.normal(size=(18, 3))*.1
        rotation = Rotation.from_rotvec([.2, .1, -.15]).as_matrix()
        target = 1.25*source@rotation.T+[.2, -.3, .1]
        appearance = np.ones((18, 18))
        np.fill_diagonal(appearance, 0)
        proposals = multipoint_proposals(source, target, np.arange(18)[:, None], appearance,
            tolerance=.003, size_ratio=1.25, config=JointConfig(proposal_trials=16))
        self.assertTrue(proposals)
        self.assertEqual({item['proposal_support'] for item in proposals}, {6, 12})
        for item in proposals:
            scale, r, shift = item['transform']
            np.testing.assert_allclose(scale*source@r.T+shift, target, atol=1e-13)

    def test_partial_assignment_is_injective_and_fixed_support_refit(self):
        rng = np.random.default_rng(11)
        queries = rng.normal(size=(12, 3))
        target = 1.1*queries+[.3, .2, .1]
        appearance = np.ones((12, 12))
        np.fill_diagonal(appearance, 0)
        appearance[-3:] = 1
        proposal = {'transform': (1.1, np.eye(3), np.array([.31, .2, .1])),
                    'proposal_id': 0, 'origin': 'multipoint_12', 'proposal_support': 12}
        result = partial_hypothesis(queries, target, np.arange(12), appearance, proposal,
            tolerance=.1, size_ratio=1.1, config=JointConfig(), geometry_config=TransferConfig())
        assigned = result['assignment']
        self.assertEqual(np.sum(assigned < 0), 3)
        self.assertEqual(len(np.unique(assigned[assigned >= 0])), 9)
        self.assertEqual(len(result['final_support']), 9)
        np.testing.assert_allclose(result['transform'][2], [.3, .2, .1], atol=1e-14)

    def fixture(self):
        points, region, _, _ = _surface_fixture()
        transform = np.eye(4)
        transform[:3, 3] = [0, 0, 1]
        source = {'xyz': points, 'pixels': _grid_pixels(points), 'region_mask': region,
                  'entity_member': np.ones(len(points), bool), 'T_world_camera_cv': transform}
        ids, query_region = source_queries(source, 12, 12)
        target = {**source, 'region_mask': np.zeros(len(points), bool)}
        costs = np.ones((len(ids), len(points)))
        costs[np.arange(len(ids)), ids] = 0
        evidence = {'query_indices': ids, 'appearance_cost': costs, 'peak_indices': ids[:, None]}
        return source, target, evidence, query_region

    def test_end_to_end_top1_and_fixed_plane_sampling(self):
        source, target, evidence, query_region = self.fixture()
        result = infer_joint_region(source, target, evidence, query_region=query_region,
                                    config=JointConfig(proposal_trials=8, max_hypotheses=3))
        self.assertTrue(result['accepted'], result['reasons'])
        primary = result['modes'][0]
        np.testing.assert_allclose(primary['center'], source['xyz'][source['region_mask']].mean(0), atol=1e-12)
        self.assertTrue(np.allclose(target['xyz'][primary['mask'], 2], 0))
        self.assertEqual(result['diagnostics']['full_graph_scores_computed'],
                         len(target['xyz'])*result['diagnostics']['hypotheses_admitted'])

    def test_missing_local_surface_does_not_snap_to_distant_layer(self):
        source, target, evidence, query_region = self.fixture()
        keep = np.abs(target['xyz'][:, 2]) > .02
        lookup = np.full(len(keep), -1, int)
        lookup[keep] = np.arange(keep.sum())
        target = {key: (value[keep] if key in ('xyz', 'pixels', 'region_mask', 'entity_member') else value)
                  for key, value in target.items()}
        evidence['appearance_cost'] = evidence['appearance_cost'][:, keep]
        evidence['peak_indices'] = lookup[evidence['peak_indices']]
        result = infer_joint_region(source, target, evidence, query_region=query_region,
            initial_transforms=[(1., np.eye(3), np.zeros(3))],
            config=JointConfig(proposal_trials=8, max_hypotheses=3))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['modes'], [])

    def test_cross_view_selection_uses_score_not_identifier_or_truth(self):
        results = [{'accepted': True, 'diagnostics': {'selected_score': .3}, 'id': 'A'},
                   {'accepted': True, 'diagnostics': {'selected_score': .2}, 'id': 'B'}]
        self.assertEqual(select_observation(results), 1)
        results[0]['id'], results[1]['id'] = 'B', 'A'
        results[0]['target_truth'], results[1]['target_truth'] = 0, 10000
        self.assertEqual(select_observation(results), 1)
        results[1]['accepted'] = False
        self.assertEqual(select_observation(results), 0)

    def test_prediction_process_blocks_truth_reads(self):
        # The read guard is irreversible within one process, so verify it in a
        # separate interpreter instead of affecting the rest of the test suite.
        script = '''
from transfer.joint_experiment import block_evaluation_reads
block_evaluation_reads()
for path in ('/tmp/truth.npz', '/tmp/pair-truth/case.json', '/tmp/evaluation-scenes/scene.xml'):
    try:
        open(path, 'rb')
    except PermissionError:
        continue
    except FileNotFoundError:
        raise AssertionError('Truth read reached the filesystem')
    raise AssertionError('Truth read was permitted')
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                text=True, env={**os.environ, 'PYTHONNOUSERSITE': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_zero_camera_direction_is_invalid_even_with_zero_cosine_cutoff(self):
        source, target, evidence, query_region = self.fixture()
        camera = source['T_world_camera_cv'].copy()
        camera[:3, 3] = source['xyz'][source['region_mask']].mean(0)
        result = infer_joint_region({**source, 'T_world_camera_cv': camera}, target, evidence,
            query_region=query_region, geometry_config=TransferConfig(min_view_cosine=0))
        self.assertFalse(result['accepted'])
        self.assertIn('source_normal_orientation_unobservable', result['reasons'])

    def test_zero_target_camera_direction_never_supplies_the_seed(self):
        source, target, evidence, query_region = self.fixture()
        camera = target['T_world_camera_cv'].copy()
        camera[:3, 3] = source['xyz'][source['region_mask']].mean(0)
        result = infer_joint_region(source, {**target, 'T_world_camera_cv': camera}, evidence,
            query_region=query_region, geometry_config=TransferConfig(min_view_cosine=0),
            config=JointConfig(proposal_trials=8, max_hypotheses=3))
        self.assertTrue(result['accepted'], result['reasons'])
        self.assertGreater(np.linalg.norm(result['modes'][0]['center']-camera[:3, 3]), 0)


if __name__ == '__main__':
    unittest.main()
