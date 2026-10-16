"""Synthetic integration checks for local validity and observed-support choice.

These tests use real RGB-D plane projection and region sampling. Mock proposal
lists isolate decisions from stochastic recall; the final test executes actual
finite proposals and partial fitting. No fixture names or evaluation truth enter.
"""
import unittest
from unittest.mock import patch
import numpy as np
from . import geometry, inference
from .proposals import (FIT_BUDGET, GROWTH_BUDGET, build_proposals,
                        prefix_length, rank_domain)
from .test_geometry import _surface

IDENTITY = (1., np.eye(3), np.zeros(3))


def _source(surface):
    """Eight local rows and four context rows with independent source roles."""
    xy = [(4,4),(6,4),(8,4),(4,6),(8,6),(4,8),(6,8),(8,8),
          (2,2),(10,2),(2,10),(10,10)]
    ids = np.array([y*13+x for x,y in xy])
    return dict(query_xyz=surface['xyz'][ids].copy(), query_indices=ids,
        query_normals=np.tile([0.,0.,-1.], (12,1)), query_normal_valid=np.ones(12,bool),
        query_region=np.arange(12)<8, check_xyz=np.empty((0,3)),
        check_normals=np.empty((0,3)), check_indices=np.empty(0,np.int64),
        check_region=np.empty(0,bool), center=np.array([0.,0.,1.]),
        normal=np.array([0.,0.,-1.]), size=.02, reasons=[], diagnostics={})


class AnalyticAppearance:
    """Observed plane responses, with explicit missing rows and controlled ties."""
    def __init__(self, source, surface, *, unknown=(), paired_ties=False, constant=False):
        self.source, self.surface = source, surface
        self.unknown = np.asarray(unknown, dtype=np.int64)
        self.paired_ties, self.constant = paired_ties, constant

    def sample(self, points, normals, query_rows=None):
        projections = [geometry.project_oriented(self.surface, points, normals, i) for i in range(2)]
        valid = np.column_stack([p['position_supported'] & p['plane_valid'] & p['entity_supported']
            & (~p['query_normal_valid'] | p['oriented_supported']) for p in projections])
        if query_rows is None:
            scores = -np.linalg.norm(self.source['query_xyz'][:,None]-points[None], axis=-1)
            known = np.broadcast_to(valid.any(1), scores.shape).copy()
            known[self.unknown] = False
        else:
            scores = -np.linalg.norm(self.source['query_xyz'][query_rows]-points, axis=-1)
            if self.paired_ties:
                scores[:] = 0.
            valid[np.isin(query_rows, self.unknown)] = False
            known = valid.any(1)
        if self.constant:
            scores[:] = 1.
        scores = np.where(known, scores, np.nan)
        view_scores = np.repeat(scores[...,None], 2, axis=-1)
        view_scores = np.where(valid[None] if query_rows is None else valid, view_scores, np.nan)
        states = np.where(valid, 'synthetic_observed', 'synthetic_missing')
        reasons = {}
        for v, name in enumerate(('A','B')):
            names, counts = np.unique(states[:,v], return_counts=True)
            reasons[name] = dict(zip(names.tolist(), counts.tolist()))
        return dict(scores=scores, known=known, view_scores=view_scores,
            view_valid=valid, view_state=states, reason_counts=reasons, geometry=projections)


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.surface = _surface()
        self.source = _source(self.surface)
        self.appearance = AnalyticAppearance(self.source, self.surface)

    def test_prefix_endpoint_and_complete_tie_sorted_domain(self):
        self.assertEqual(prefix_length(10001, GROWTH_BUDGET-1), 10001)
        ids, order, inverse, diagnostic, domain = rank_domain(
            self.source, self.surface, self.appearance, chunk_size=7)
        self.assertTrue(diagnostic['ranking_complete'])
        for row in range(12):
            scores = -np.linalg.norm(self.surface['xyz'][ids]-self.source['query_xyz'][row], axis=1)
            np.testing.assert_array_equal(order[row], np.lexsort((ids, -scores)))
            np.testing.assert_array_equal(inverse[row,order[row]], np.arange(len(ids)))
        self.assertIs(domain['input_binding'][0], self.source)

    def test_native_and_both_fit_sizes_preserved_with_complete_growth(self):
        proposals, diagnostic, _ = build_proposals(self.source,self.surface,self.appearance,
            initial_transforms=[dict(transform=IDENTITY, origin='native_test',view_index=0)])
        origins = [o for p in proposals for o in p['origins']]
        self.assertIn('native_test', {o['origin'] for o in origins})
        self.assertIn('unified_6_point', {o['origin'] for o in origins})
        self.assertIn('unified_12_point', {o['origin'] for o in origins})
        self.assertLessEqual(diagnostic['proposal_fit_attempts'], FIT_BUDGET)
        self.assertEqual(diagnostic['growth_attempts'], GROWTH_BUDGET)
        for o in origins:
            if o['origin'].startswith('unified'):
                self.assertEqual(len(set(o['surface_ids'])), len(o['surface_ids']))
                self.assertIn(len(o['query_rows']), (6,12))

    def test_empty_appearance_domain_has_no_invented_proposals(self):
        proposals, diagnostic, _ = build_proposals(self.source,self.surface,
            AnalyticAppearance(self.source,self.surface,unknown=np.arange(12)))
        self.assertEqual(proposals, [])
        self.assertEqual(diagnostic['proposal_fit_attempts'], 0)
        self.assertEqual(diagnostic['growth_attempts'], GROWTH_BUDGET)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.surface = _surface()
        self.source = _source(self.surface)

    def run_candidates(self, *transforms, **appearance_options):
        appearance = AnalyticAppearance(self.source,self.surface,**appearance_options)
        domain = rank_domain(self.source,self.surface,appearance)[4]
        proposals = [dict(transform=t, origins=[{'origin':'test_native'}], proposal_id=i)
                     for i,t in enumerate(transforms)]
        with patch.object(inference,'build_proposals',return_value=(proposals,{},domain)), \
             patch.object(inference,'partial_refits',return_value=([],{})):
            return inference.infer(self.source,self.surface,appearance,prepared_domain=domain)

    def test_single_finite_winner_is_an_estimate_not_a_pose_certificate(self):
        result = self.run_candidates(IDENTITY)
        self.assertTrue(result['accepted'])
        self.assertTrue(result['finite_candidate_unique'])
        self.assertFalse(result['transform_certified'])
        self.assertNotIn('transform_identified',result)
        self.assertEqual(result['scheme'],'dual_view_observed_partial')
        self.assertEqual(result['diagnostics']['fixed_query_denominator'],12)
        self.assertGreater(result['diagnostics']['best_score'],0)

    def test_context_query_and_check_conflict_are_retained_without_local_veto(self):
        self.source['query_xyz'][8,2] = .9
        self.source.update(check_xyz=np.array([[0.,0.,.9]]),
            check_normals=np.array([[0.,0.,-1.]]), check_indices=np.array([999]),
            check_region=np.array([False]))
        result = self.run_candidates(IDENTITY)
        self.assertTrue(result['accepted'])
        h = result['diagnostic_hypotheses'][0]
        self.assertEqual(h['context_query_conflict_count'],1)
        self.assertEqual(h['context_check_conflict_count'],1)
        self.assertEqual(h['entity_model_status'],'observed_conflict')
        self.assertIn('entity_transform_not_fully_supported',result['diagnostics']['quality_warnings'])

    def test_local_query_conflict_remains_a_hard_veto(self):
        self.source['query_xyz'][0,2] = .9
        result = self.run_candidates(IDENTITY)
        self.assertFalse(result['accepted'])
        self.assertIn('region_query_geometry_model_conflict',
                      result['diagnostic_hypotheses'][0]['rejections'])

    def test_local_independent_check_still_rejects(self):
        self.source.update(check_xyz=np.array([[0.,0.,.9]]),
            check_normals=np.array([[0.,0.,-1.]]),check_indices=np.array([999]),
            check_region=np.array([True]))
        result = self.run_candidates(IDENTITY)
        self.assertFalse(result['accepted'])
        self.assertIn('region_check_geometry_model_conflict',
                      result['diagnostic_hypotheses'][0]['rejections'])

    def test_unknown_local_appearance_cannot_win_on_context_only(self):
        result = self.run_candidates(IDENTITY,unknown=np.arange(8))
        self.assertFalse(result['accepted'])
        self.assertIn('no_positive_region_appearance',
                      result['diagnostic_hypotheses'][0]['rejections'])
        self.assertEqual(result['diagnostic_hypotheses'][0]['known_count'],4)

    def test_constant_response_is_not_discriminating_appearance(self):
        result = self.run_candidates(IDENTITY,constant=True)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['diagnostic_hypotheses'][0]['score'],0)

    def test_unknown_normal_can_read_rgb_but_does_not_create_oriented_support(self):
        self.source['query_normals'][0] = np.nan
        self.source['query_normal_valid'][0] = False
        result = self.run_candidates(IDENTITY)
        self.assertTrue(result['accepted'])
        h = result['diagnostic_hypotheses'][0]
        self.assertEqual(h['known_count'],12)
        self.assertEqual(h['region_geometry_count'],7)

    def test_tied_estimates_with_common_region_do_not_return_unique_transform(self):
        moved = (1.,np.eye(3),np.array([1/256,0.,0.]))
        result = self.run_candidates(IDENTITY,moved,paired_ties=True)
        self.assertTrue(result['accepted'])
        self.assertFalse(result['finite_candidate_unique'])
        self.assertFalse(result['transform_certified'])
        self.assertIsNone(result['sim3_scale'])
        self.assertEqual(result['diagnostics']['potential_winner_count'],2)

    def test_equal_best_ambiguous_center_remains_unresolved(self):
        moved = (1.,np.eye(3),np.array([1/256,0.,0.]))
        original = geometry.region_for_transform
        def ambiguous(surface,source,transform,**kwargs):
            if transform[2][0] != 0:
                return None,'center_projection_ambiguous'
            return original(surface,source,transform,**kwargs)
        with patch.object(geometry,'region_for_transform',side_effect=ambiguous):
            result = self.run_candidates(IDENTITY,moved,paired_ties=True)
        self.assertFalse(result['accepted'])
        self.assertIn('center_projection_ambiguous',result['reasons'])

    def test_every_declared_candidate_is_scored_and_ties_are_retained(self):
        transforms = [(1.,np.eye(3),np.array([i/8192,0.,0.])) for i in range(40)]
        result = self.run_candidates(*transforms,paired_ties=True)
        self.assertEqual(len(result['diagnostic_hypotheses']),40)
        self.assertEqual(result['diagnostics']['potential_winner_count'],40)

    def test_interruption_never_inherits_acceptance(self):
        with patch.object(inference,'build_proposals',side_effect=InterruptedError('test')):
            result = inference.infer(self.source,self.surface,
                AnalyticAppearance(self.source,self.surface))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'],'search_incomplete')

    def test_end_to_end_records_all_original_and_partial_fit_candidates(self):
        result = inference.infer(self.source,self.surface,AnalyticAppearance(self.source,self.surface),
            initial_transforms=[dict(transform=IDENTITY,origin='native_test',view_index=0)])
        self.assertEqual(result['diagnostics']['search_status'],'finite_domain_complete')
        self.assertTrue(result['accepted'])
        self.assertEqual(result['diagnostics']['evaluated_transform_count'],
                         len(result['diagnostic_hypotheses']))
        self.assertTrue(all(len(h['scores'])==12 for h in result['diagnostic_hypotheses']))
        self.assertGreater(result['diagnostics']['refit_retained_count'],0)

    # BEGIN bounded polishing checks use real plane geometry and observed scores.
    def test_polishing_improves_full_pool_without_replacing_original_candidates(self):
        appearance = AnalyticAppearance(self.source,self.surface)
        domain = rank_domain(self.source,self.surface,appearance)[4]
        shifts = [(1.,np.eye(3),np.array([step/256,0.,0.])) for step in (3,2,1,0)]
        proposal = dict(transform=shifts[0], origins=[{'origin':'test_native'}], proposal_id=0)
        calls = []
        def refinements(source,surface,prepared,seed):
            calls.append(seed)
            if len(calls) == 1:  # The original, noniterative refit finds no update.
                return [], {}
            candidate = dict(transform=shifts[len(calls)-1], origins=[{
                'origin':'one_partial_all_refit', 'parent_proposal_id':seed['proposal_id']}])
            return [candidate], {'parent_proposal_id':seed['proposal_id']}
        with patch.object(inference,'build_proposals',return_value=([proposal],{},domain)), \
             patch.object(inference,'partial_refits',side_effect=refinements):
            result = inference.infer(self.source,self.surface,appearance,prepared_domain=domain)
        self.assertTrue(result['accepted'])
        rounds = result['diagnostics']['polishing']['rounds']
        self.assertEqual(len(rounds),3)
        self.assertTrue(all(r['score_after'] > r['score_before'] for r in rounds))
        self.assertEqual(len(result['diagnostic_hypotheses']),4)
        self.assertEqual(result['diagnostic_hypotheses'][0]['origins'][0]['origin'],'test_native')
        np.testing.assert_array_equal(result['sim3_translation'],np.zeros(3))
        for h in result['diagnostic_hypotheses'][1:]:
            origin = h['origins'][0]
            self.assertEqual(origin['origin'],'polish_partial_all_refit')
            self.assertIn('parent_hypothesis_id',origin)
            self.assertNotIn('parent_proposal_id',origin)

    def test_polishing_stops_without_score_improvement_and_does_not_seed_ties(self):
        identity = self.run_candidates(IDENTITY)
        self.assertEqual(len(identity['diagnostics']['polishing']['rounds']),1)
        self.assertEqual(identity['diagnostics']['polishing']['stop_reason'],
                         'no_strict_score_improvement')
        tied = self.run_candidates(IDENTITY,(1.,np.eye(3),np.array([1/256,0.,0.])),paired_ties=True)
        self.assertEqual(tied['diagnostics']['polishing']['rounds'],[])
        self.assertEqual(tied['diagnostics']['polishing']['stop_reason'],'best_seed_not_unique')
    # END bounded polishing experiment checks.


if __name__ == '__main__':
    unittest.main()
