"""Small CPU checks for observed-surface inference and its decision contract.

Synthetic correspondences test matching and inference, with geometry helpers
mocked only in orchestration tests. These tests do not claim to validate depth
reconstruction or occlusion; the geometry tests and cache integration do that.
"""

import unittest
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
from scipy.sparse import csr_matrix

from transfer.multiview_grounding import (MultiviewConfig, _appearance_coordinates,
    _equivalent_regions, _nondominated, infer_multiview_region,
    multiview_proposals, partial_hypothesis)
from transfer.sim3 import fit_sim3, rank, rms_size
from transfer.experiment import restore_prediction, store_prediction


# BEGIN mathematical regression fixtures for the replacement dual-view path.
def _fixture():
    """An observed plane with separate task/context query provenance."""
    y, x = np.mgrid[:9, :9]
    points = np.column_stack(((x.ravel()-4)*.005, (y.ravel()-4)*.005, np.ones(x.size)))
    region = (np.abs(points[:, 0]) <= .010001) & (np.abs(points[:, 1]) <= .010001)
    source = {'xyz': points, 'region_mask': region, 'entity_member': np.ones(len(points), bool),
              'pixels': np.column_stack((x.ravel(), y.ravel())),
              'K': np.array([[200., 0, 4], [0, 200., 4], [0, 0, 1]]),
              'T_world_camera_cv': np.eye(4)}
    surface = {'xyz': points.copy(), 'normals': np.tile([0., 0., -1.], (len(points), 1)),
               'spacing': np.full(len(points), .005), 'valid': np.ones(len(points), bool),
               'entity_member': np.ones(len(points), bool), 'adjacency': csr_matrix((len(points), len(points))),
               'view_ids': ['A', 'B'], 'observed_views': np.ones((len(points), 2), bool),
               'member_views': np.ones((len(points), 2), bool)}
    query = np.array([20, 24, 40, 56, 60, 0, 8, 72, 80])
    allowed = np.eye(len(query), dtype=bool)
    edge_views = np.repeat(allowed[:, :, None], 2, axis=2)
    appearance = np.full(edge_views.shape, np.nan)
    appearance[edge_views] = .2
    evidence = {'query_indices': query, 'query_region': region[query], 'candidate_indices': query.copy(),
                'allowed': allowed, 'edge_views': edge_views, 'appearance_available': edge_views.copy(),
                'appearance_cost': appearance}
    return source, surface, evidence


def _proposal(transform=None, *, tau=.01, proposal_id=0):
    if transform is None:
        transform = (1., np.eye(3), np.zeros(3))
    return {'transform': transform, 'origin': f'test_{proposal_id}', 'pool': 'native', 'view_index': 0,
            'proposal_id': proposal_id, 'proposal_support': None, 'proposal_query_rows': np.empty(0, int),
            'proposal_candidate_columns': np.empty(0, int), 'quality': (0, 0), 'tolerance_m': tau}


def _sample(surface, eligible, seed, *, sampling_radius_m, sampling_spacing_m, config):
    """Independent planar sampler for orchestration tests only."""
    mask = eligible & (np.linalg.norm(surface['xyz']-surface['xyz'][seed], axis=1) <= sampling_radius_m+1e-12)
    return {'seed_index': seed, 'center': surface['xyz'][seed].copy(),
            'normal': surface['normals'][seed].copy(), 'mask': mask,
            'sample_indices': np.flatnonzero(mask),
            'diagnostics': {'neighborhood_point_count': int(mask.sum()),
                            'neighborhood_rank': rank(surface['xyz'][mask])[0]}}


def _checks(surface, points, *, tolerance_m):
    """Explicit fixture input: each checked point has observed support."""
    return {'support': np.ones(len(points), bool), 'conflict': np.zeros(len(points), bool),
            'unknown': np.zeros(len(points), bool), 'view_states': np.full((len(points), 2), 'supported')}


class TestAllowedPartialMatching(unittest.TestCase):
    def setUp(self):
        source, _, evidence = _fixture()
        self.points = source['xyz'][evidence['query_indices']]

    def test_all_forbidden_edges_produce_nulls_without_forced_real_matches(self):
        q = len(self.points)
        hypothesis, reason = partial_hypothesis(self.points, self.points, np.zeros((q, q), bool), _proposal())
        self.assertIsNone(hypothesis)
        self.assertEqual(reason, 'insufficient_initial_support')

    def test_maximum_cardinality_precedes_residual_and_respects_forbidden_edges(self):
        q = len(self.points)
        allowed = np.eye(q, dtype=bool)
        allowed[0, 1], allowed[1, 0], allowed[1, 1] = True, True, False
        hypothesis, reason = partial_hypothesis(self.points, self.points, allowed, _proposal(tau=.1))
        self.assertIsNone(reason)
        self.assertEqual(hypothesis['initial_assignment'][0], 1)
        self.assertEqual(hypothesis['initial_assignment'][1], 0)
        self.assertEqual(len(hypothesis['initial_support']), q)
        final = hypothesis['final_support']
        self.assertTrue(allowed[final, hypothesis['assignment'][final]].all())

    def test_missing_rows_remain_null_without_relaxing_the_allowed_graph(self):
        q = len(self.points)
        allowed = np.eye(q, dtype=bool)
        allowed[-2:] = False
        hypothesis, reason = partial_hypothesis(self.points, self.points, allowed, _proposal())
        self.assertIsNone(reason)
        np.testing.assert_array_equal(hypothesis['assignment'][-2:], [-1, -1])
        self.assertEqual(hypothesis['matched_count'], q-2)

    def test_single_refit_nullifies_outlier_and_keeps_proposal_tolerance(self):
        target = self.points.copy()
        target[-1, 0] += .009
        transform = (1., np.eye(3), np.array([-.002, 0, 0]))
        with mock.patch('transfer.multiview_grounding.fit_sim3', return_value=transform) as fit:
            hypothesis, reason = partial_hypothesis(self.points, target, np.eye(len(target), dtype=bool),
                                                   _proposal(tau=.01))
        self.assertIsNone(reason)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(hypothesis['initial_assignment'][-1], len(target)-1)
        self.assertEqual(hypothesis['assignment'][-1], -1)
        self.assertNotIn(len(target)-1, hypothesis['final_support'])
        self.assertAlmostEqual(hypothesis['fit_error'], (len(target)-1)*.2**2/len(target))
        self.assertEqual(hypothesis['tolerance_m'], .01)

    def test_illegal_refit_scale_rejects_without_returning_initial_transform(self):
        with mock.patch('transfer.multiview_grounding.fit_sim3', return_value=(6., np.eye(3), np.zeros(3))):
            hypothesis, reason = partial_hypothesis(self.points, self.points,
                                                   np.eye(len(self.points), dtype=bool), _proposal())
        self.assertIsNone(hypothesis)
        self.assertEqual(reason, 'invalid_refitted_transform')

    def test_larger_refit_scale_cannot_expand_its_initial_tolerance(self):
        points = self.points.copy()
        points[:, :2] *= .5
        # Five central points survive at tau=.01; four corner points would
        # incorrectly re-enter if the fitted scale doubled that same tau.
        with mock.patch('transfer.multiview_grounding.fit_sim3',
                        return_value=(2., np.eye(3), np.array([0., 0., -1.]))):
            hypothesis, reason = partial_hypothesis(points, points,
                np.eye(len(points), dtype=bool), _proposal(tau=.01))
        self.assertIsNone(hypothesis)
        self.assertEqual(reason, 'insufficient_final_support')


class TestMixedProposals(unittest.TestCase):
    def test_six_and_twelve_point_fits_each_consume_the_frozen_fit_budget(self):
        y, x = np.mgrid[:4, :4]
        points = np.column_stack((x.ravel()*.005, y.ravel()*.005, np.ones(x.size)))
        allowed = np.eye(len(points), dtype=bool)
        views = np.repeat(allowed[:, :, None], 2, axis=2)
        with mock.patch('transfer.multiview_grounding.fit_sim3', wraps=fit_sim3) as fitting:
            proposals, diagnostic = multiview_proposals(points, points, allowed, views,
                                                        source_size=rms_size(points))
        self.assertEqual(diagnostic['fit_attempts'], {'A': 85, 'B': 85, 'union': 86})
        self.assertEqual(fitting.call_count, 256)
        self.assertEqual({len(call.args[0]) for call in fitting.call_args_list}, {6, 12})
        self.assertEqual(len(proposals), 30)

    def test_union_can_fit_when_neither_view_has_six_edges_and_budgets_are_separate(self):
        source, surface, evidence = _fixture()
        views = evidence['edge_views'].copy()
        views[:4, :, 1], views[4:, :, 0] = False, False
        query = source['xyz'][evidence['query_indices']]
        proposals, diagnostic = multiview_proposals(query, query, evidence['allowed'], views,
                                                    source_size=rms_size(source['xyz']))
        self.assertTrue(proposals)
        self.assertTrue(all(item['pool'] == 'union' for item in proposals))
        self.assertEqual(diagnostic['grow_attempts'], {'A': 85, 'B': 85, 'union': 86})
        self.assertEqual(diagnostic['fit_attempts']['A'], 0)
        self.assertEqual(diagnostic['fit_attempts']['B'], 0)
        self.assertLessEqual(diagnostic['total_new_proposal_fits'], 256)
        self.assertLessEqual(len(proposals), 10)
        for item in proposals:
            self.assertTrue(np.any(item['proposal_query_rows'] < 4))
            self.assertTrue(np.any(item['proposal_query_rows'] >= 4))
            np.testing.assert_allclose(item['transform'][0], 1., atol=1e-12)

    def test_native_seats_and_per_pool_limits_do_not_borrow_empty_b_seats(self):
        source, surface, evidence = _fixture()
        views = evidence['edge_views'].copy()
        views[:, :, 1] = False
        query = source['xyz'][evidence['query_indices']]
        native = [{'transform': (1., np.eye(3), np.zeros(3)), 'origin': f'native_{i}', 'view_index': i % 2}
                  for i in range(6)]
        proposals, diagnostic = multiview_proposals(query, query, evidence['allowed'], views,
            source_size=rms_size(source['xyz']), initial_transforms=native)
        self.assertEqual(diagnostic['native_retained'], 2)
        self.assertEqual(diagnostic['retained_by_pool']['B'], 0)
        self.assertLessEqual(len(proposals), 22)
        for pool, count in diagnostic['fit_attempts'].items():
            self.assertLessEqual(count, diagnostic['fit_budget_by_pool'][pool])


class TestCommonDomainsAndRegionDecision(unittest.TestCase):
    def test_appearance_uses_final_common_rows_and_omits_empty_view(self):
        available = np.ones((3, 3, 2), bool)
        available[0, 0, 1] = False
        costs = np.full((3, 3, 2), .5)
        costs[0, 0, 0], costs[0, 2, 0], costs[0, 0, 1] = .2, .8, np.nan
        evidence = {'query_indices': np.arange(3), 'appearance_available': available, 'appearance_cost': costs}
        hypotheses = [{'assignment': np.array([0, 1, -1])}, {'assignment': np.array([2, -1, 1])}]
        domains, coordinates = _appearance_coordinates(hypotheses, evidence, np.arange(3), np.ones((3, 2), bool))
        np.testing.assert_array_equal(domains[0], [0])
        self.assertEqual(len(domains[1]), 0)
        np.testing.assert_allclose(coordinates, [[.2], [.8]])

    def test_pareto_retains_tradeoffs_without_summing_coordinates(self):
        np.testing.assert_array_equal(_nondominated(np.array([[0., 1.], [1., 0.], [2., 2.]])), [0, 1])

    def test_equivalence_rejects_transitive_chain_and_missing_overlap(self):
        modes = [{'center': np.array([x, 0, 0]), 'normal': np.array([0., 0., 1.]),
                  'mask': np.ones(3, bool), 'seed_index': i} for i, x in enumerate([0, .004, .008])]
        self.assertTrue(_equivalent_regions(modes[:2], sampling_spacing_m=.005, normal_min_agreement=.65))
        self.assertFalse(_equivalent_regions(modes, sampling_spacing_m=.005, normal_min_agreement=.65))
        modes[0]['mask'][1] = False
        self.assertFalse(_equivalent_regions(modes[:2], sampling_spacing_m=.005, normal_min_agreement=.65))

    def test_configuration_only_exposes_cpu_block_sizes(self):
        self.assertEqual(MultiviewConfig().proposal_trials, 256)
        with self.assertRaises(TypeError):
            MultiviewConfig(proposal_trials=12)
        with self.assertRaises(ValueError):
            MultiviewConfig(center_chunk=0)


class TestInferenceOrchestration(unittest.TestCase):
    def _run(self, source, surface, evidence, proposals=None):
        with mock.patch('transfer.multiview_grounding.geometry.sample_region', side_effect=_sample, create=True), \
             mock.patch('transfer.multiview_grounding.geometry.check_points', side_effect=_checks, create=True):
            if proposals is None:
                return infer_multiview_region(source, surface, evidence)
            with mock.patch('transfer.multiview_grounding.multiview_proposals', return_value=(proposals, {})):
                return infer_multiview_region(source, surface, evidence)

    def test_complete_sparse_candidate_pipeline_maps_anchors_to_surface_ids(self):
        source, surface, evidence = _fixture()
        result = self._run(source, surface, evidence)
        self.assertTrue(result['accepted'], result['reasons'])
        self.assertEqual(result['modes'][0]['type'], 'primary')
        np.testing.assert_array_equal(result['src_anchor_idx'], evidence['query_indices'])
        np.testing.assert_array_equal(result['tgt_anchor_idx'], evidence['candidate_indices'])
        self.assertEqual(result['diagnostics']['assignment_index_domain'], 'evidence.candidate_indices')
        self.assertLessEqual(len(result['diagnostics']['hypotheses']), 32)
        self.assertFalse(set(result['diagnostics']['check_source_indices']) & set(evidence['query_indices']))
        for h in result['diagnostics']['hypotheses']:
            np.testing.assert_array_equal(h['assignment_surface_ids'][h['final_support']],
                                          evidence['candidate_indices'][h['assignment'][h['final_support']]])

    def test_missing_b_appearance_is_warning_while_b_geometry_remains(self):
        source, surface, evidence = _fixture()
        # Unpaired A measurements can still have independent B projection
        # support; membership and observed visibility must not be conflated.
        surface['member_views'][:, 1] = False
        evidence['edge_views'][:, :, 1] = False
        evidence['appearance_available'][:, :, 1] = False
        evidence['appearance_cost'][:, :, 1] = np.nan
        result = self._run(source, surface, evidence, [_proposal()])
        self.assertTrue(result['accepted'], result['reasons'])
        self.assertIn('empty_common_appearance_domain:B', result['diagnostics']['quality_warnings'])
        self.assertEqual(result['diagnostics']['appearance_active_views'], ['A'])
        h = result['diagnostics']['hypotheses'][0]
        self.assertEqual(h['final_observed_view_counts']['both'], len(evidence['query_indices']))
        self.assertEqual(h['final_member_view_counts']['A_only'], len(evidence['query_indices']))
        self.assertEqual(h['final_edge_sources']['A_only'], len(evidence['query_indices']))

    def test_demonstration_queries_survive_source_entity_mask_misses(self):
        source, surface, evidence = _fixture()
        demonstration = evidence['query_indices'][evidence['query_region']][0]
        source['entity_member'][demonstration] = False
        result = self._run(source, surface, evidence, [_proposal()])
        self.assertTrue(result['accepted'], result['reasons'])
        self.assertIn(demonstration, result['src_anchor_idx'])
        # An unlabeled context pixel has no independent demonstration witness.
        context = evidence['query_indices'][~evidence['query_region']][0]
        source['entity_member'][context] = False
        with self.assertRaises(ValueError):
            self._run(source, surface, evidence, [_proposal()])

    def test_partial_result_saves_missing_residuals_as_json_null_and_roundtrips(self):
        source, surface, evidence = _fixture()
        evidence['allowed'][-2:] = False
        evidence['edge_views'][-2:] = False
        result = self._run(source, surface, evidence, [_proposal()])
        self.assertTrue(result['accepted'], result['reasons'])
        self.assertEqual(result['diagnostics']['hypotheses'][0]['residuals'][-2:], [None, None])
        self.assertAlmostEqual(result['diagnostics']['hypotheses'][0]['graph_score'], 0., places=12)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'prediction'
            store_prediction(destination, result, {})
            restored = restore_prediction(destination)
        self.assertTrue(restored['accepted'])
        self.assertEqual(restored['diagnostics']['hypotheses'][0]['residuals'][-2:], [None, None])
        np.testing.assert_array_equal(restored['modes'][0]['sample_indices'], result['modes'][0]['sample_indices'])

    def test_two_tradeoff_regions_are_ambiguous_without_executable_transform(self):
        source, surface, evidence = _fixture()
        n, q = len(surface['xyz']), len(evidence['query_indices'])
        shift = np.array([.1, 0, 0])
        surface['xyz'] = np.vstack((surface['xyz'], surface['xyz']+shift))
        for key in ('normals', 'spacing', 'valid', 'entity_member', 'member_views', 'observed_views'):
            surface[key] = np.concatenate((surface[key], surface[key]))
        surface['adjacency'] = csr_matrix((2*n, 2*n))
        evidence['candidate_indices'] = np.r_[evidence['candidate_indices'], evidence['candidate_indices']+n]
        evidence['allowed'] = np.column_stack((np.eye(q, dtype=bool), np.eye(q, dtype=bool)))
        evidence['edge_views'] = np.repeat(evidence['allowed'][:, :, None], 2, axis=2)
        evidence['appearance_available'] = evidence['edge_views'].copy()
        evidence['appearance_cost'] = np.full((q, 2*q, 2), np.nan)
        row = np.arange(q)
        evidence['appearance_cost'][row, row] = [0., 1.]
        evidence['appearance_cost'][row, row+q] = [1., 0.]
        result = self._run(source, surface, evidence,
                           [_proposal(), _proposal((1., np.eye(3), shift), proposal_id=1)])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reasons'], ['ambiguous_nondominated_regions'])
        self.assertIsNone(result['sim3_scale'])
        self.assertEqual(len(result['src_anchor_idx']), 0)
        self.assertEqual(len(result['modes']), 2)
        self.assertTrue(all(mode['type'] == 'nondominated_competitor' for mode in result['modes']))

    def test_invalid_candidate_identity_or_fabricated_missing_cost_raises(self):
        source, surface, evidence = _fixture()
        evidence['candidate_indices'][-1] = evidence['candidate_indices'][0]
        with self.assertRaises(ValueError):
            self._run(source, surface, evidence)
        source, surface, evidence = _fixture()
        evidence['appearance_cost'][0, 1, 1] = 0
        with self.assertRaises(ValueError):
            self._run(source, surface, evidence)


if __name__ == '__main__':
    unittest.main()
# END reconstructed dual-view mathematical checks.
