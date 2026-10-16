"""Small deterministic tests of the independent finite dual-view decision.

The synthetic plane tests exercise actual projection/region construction. Mock
proposal lists isolate the decision from finite random recall; separate tests
execute the complete 256-attempt proposal contract and the combined path.
"""

import unittest
from unittest.mock import patch

import numpy as np

from . import inference
from .proposals import (FIT_BUDGET, GROWTH_BUDGET, build_proposals,
                        prefix_length, rank_domain)
from .test_geometry import _surface


IDENTITY = (1., np.eye(3), np.zeros(3))


def _source(surface):
    """Six noncollinear frozen rows on the interior of a binary-spacing plane."""
    ids = np.array([4 * 13 + 4, 4 * 13 + 6, 4 * 13 + 8,
                    8 * 13 + 4, 8 * 13 + 6, 8 * 13 + 8])
    xyz = surface['xyz'][ids].copy()
    return dict(query_xyz=xyz, query_normals=np.tile([0., 0., -1.], (6, 1)),
        query_normal_valid=np.ones(6, bool), query_indices=ids,
        query_region=np.array([True, True, True, False, False, False]),
        check_xyz=np.empty((0, 3)), check_normals=np.empty((0, 3)),
        check_indices=np.empty(0, np.int64), center=np.array([0., 0., 1.]),
        normal=np.array([0., 0., -1.]), size=.02, reasons=[], diagnostics={})


class AnalyticAppearance:
    """A deterministic synthetic raw field with explicit missing-row support."""

    def __init__(self, source, *, unknown=(), conflict_bonus=False):
        self.source = source
        count = len(source['query_xyz'])
        self.lower, self.upper = np.full(count, -20.), np.full(count, 20.)
        self.unknown = np.asarray(unknown, dtype=np.int64)
        self.conflict_bonus = conflict_bonus

    def sample(self, points, normals, query_rows=None):
        n = len(points)
        view_valid = np.broadcast_to(np.isfinite(normals).all(1)[:, None], (n, 2)).copy()
        if query_rows is None:
            scores = -np.linalg.norm(self.source['query_xyz'][:, None] - points[None], axis=-1)
            known = np.broadcast_to(view_valid.any(1), scores.shape).copy()
            known[self.unknown] = False
        else:
            scores = np.zeros(n)
            if self.conflict_bonus:
                scores[points[:, 2] < .99] = 10.
            known = view_valid.any(1) & ~np.isin(query_rows, self.unknown)
        scores = np.where(known, scores, np.nan)
        states = np.where(view_valid, 'synthetic_valid', 'missing_query_normal')
        reason_counts = {}
        for index, name in enumerate(('A', 'B')):
            unique, numbers = np.unique(states[:, index], return_counts=True)
            reason_counts[name] = dict(zip(unique.tolist(), numbers.tolist()))
        return dict(scores=scores, known=known,
                    view_scores=np.repeat(scores[..., None], 2, axis=-1),
                    view_valid=view_valid,
                    view_state=states, reason_counts=reason_counts)


def _proposal_list(*transforms):
    return ([{'transform': transform, 'origins': [{'origin': 'test_native'}],
              'proposal_id': i} for i, transform in enumerate(transforms)],
            {'proposal_search_complete': True, 'ranking_complete': True})


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.surface = _surface()
        self.source = _source(self.surface)
        self.appearance = AnalyticAppearance(self.source)

    def test_prefix_endpoint_and_complete_tie_sorted_domain(self):
        self.assertEqual(prefix_length(10001, 0), 1)
        self.assertEqual(prefix_length(10001, GROWTH_BUDGET - 1), 10001)
        self.assertEqual(prefix_length(0, GROWTH_BUDGET - 1), 0)
        ids, order, inverse, diagnostic = rank_domain(self.source, self.surface, self.appearance, chunk_size=7)
        self.assertTrue(diagnostic['ranking_complete'])
        self.assertEqual(len(ids), int(np.count_nonzero(self.surface['valid'] & self.surface['entity_member'])))
        self.assertEqual(diagnostic['ranking_view_state_count_unit'], 'representative_point_per_view')
        self.assertEqual(diagnostic['ranking_view_state_counts'],
                         {name: {'synthetic_valid': len(ids)} for name in ('A', 'B')})
        for row in range(6):
            expected_scores = -np.linalg.norm(self.surface['xyz'][ids] - self.source['query_xyz'][row], axis=1)
            expected = np.lexsort((ids, -expected_scores))
            np.testing.assert_array_equal(order[row], expected)
            np.testing.assert_array_equal(inverse[row, order[row]], np.arange(len(ids)))

    def test_native_provenance_and_six_pair_minimum_and_finite_budget(self):
        proposals, diagnostic = build_proposals(self.source, self.surface, self.appearance,
            initial_transforms=[dict(transform=IDENTITY, origin='native_teaserpp', view_index=0),
                                dict(transform=IDENTITY, origin='native_ransac', view_index=1)])
        self.assertTrue(diagnostic['proposal_search_complete'])
        self.assertLessEqual(diagnostic['growth_attempts'], GROWTH_BUDGET)
        self.assertLessEqual(diagnostic['proposal_fit_attempts'], FIT_BUDGET)
        self.assertTrue(diagnostic['growth_attempts'] == GROWTH_BUDGET
                        or diagnostic['proposal_fit_attempts'] == FIT_BUDGET)
        native = [origin for proposal in proposals for origin in proposal['origins']
                  if origin['origin'].startswith('native')]
        self.assertEqual(len(native), 2)
        for proposal in proposals:
            for origin in proposal['origins']:
                if origin['origin'].startswith('unified'):
                    self.assertIn(len(origin['query_rows']), (6, 12))
                    self.assertEqual(len(set(origin['surface_ids'])), len(origin['surface_ids']))

    def test_missing_domain_still_finishes_declared_growth_plan(self):
        appearance = AnalyticAppearance(self.source, unknown=np.arange(6))
        proposals, diagnostic = build_proposals(self.source, self.surface, appearance)
        self.assertEqual(proposals, [])
        self.assertEqual(diagnostic['growth_attempts'], GROWTH_BUDGET)
        self.assertEqual(diagnostic['proposal_fit_attempts'], 0)
        self.assertTrue(diagnostic['ranking_complete'])

    def test_twelve_pair_growth_keeps_six_and_twelve_fits(self):
        ids = np.array([y * 13 + x for y in (4, 6, 8) for x in (3, 5, 7, 9)])
        self.source.update(query_xyz=self.surface['xyz'][ids], query_indices=ids,
            query_normals=np.tile([0., 0., -1.], (12, 1)),
            query_normal_valid=np.ones(12, bool), query_region=np.arange(12) < 6)
        proposals, diagnostic = build_proposals(self.source, self.surface,
                                               AnalyticAppearance(self.source))
        origins = [origin for proposal in proposals for origin in proposal['origins']]
        self.assertIn('unified_6_point', {origin['origin'] for origin in origins})
        self.assertIn('unified_12_point', {origin['origin'] for origin in origins})
        self.assertEqual(diagnostic['proposal_fit_attempts'], FIT_BUDGET)
        # Every grown correspondence respects every earlier distance constraint;
        # checking all pairs also verifies no proposal was fitted from a triple.
        for origin in origins:
            source = self.source['query_xyz'][origin['query_rows']]
            target = self.surface['xyz'][origin['surface_ids']]
            source_distance = np.linalg.norm(source[:, None] - source[None], axis=-1)
            target_distance = np.linalg.norm(target[:, None] - target[None], axis=-1)
            tolerance = .075 * self.source['size'] * origin['pair_scale']
            self.assertTrue(np.all(np.abs(target_distance-origin['pair_scale']*source_distance)
                                   <= 2*tolerance))


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.surface = _surface()
        self.source = _source(self.surface)

    def run_candidates(self, appearance, *transforms):
        # The exact identity refit tests duplicate-transform bookkeeping without
        # introducing SVD last-bit differences unrelated to the tested decision.
        with patch.object(inference, 'build_proposals', return_value=_proposal_list(*transforms)), \
             patch.object(inference, 'fit_sim3', return_value=IDENTITY):
            return inference.infer(self.source, self.surface, appearance)

    def test_single_supported_transform_is_accepted_and_fully_identified(self):
        result = self.run_candidates(AnalyticAppearance(self.source), IDENTITY)
        self.assertTrue(result['accepted'])
        self.assertTrue(result['transform_identified'])
        self.assertEqual(result['diagnostics']['search_status'], 'finite_domain_complete')
        self.assertEqual(result['scheme'], 'dual_view_continuous')
        self.assertEqual(result['diagnostics']['fixed_query_denominator'], 6)
        self.assertEqual(result['diagnostics']['hypotheses_admitted'], 1)
        self.assertIn('finite_search_not_global', result['diagnostics']['quality_warnings'])

    def test_all_region_scores_unknown_cannot_vacuously_accept(self):
        result = self.run_candidates(AnalyticAppearance(self.source, unknown=[0, 1, 2]), IDENTITY)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'], 'insufficient_appearance')
        self.assertGreater(len(result['modes']), 0)
        h = result['diagnostic_hypotheses'][0]
        self.assertEqual(h['known_count'], 3)
        self.assertEqual((h['lower'], h['upper']), (-10., 10.))

    def test_high_raw_score_cannot_override_front_depth_conflict(self):
        front = (1., np.eye(3), np.array([0., 0., -.1]))
        result = self.run_candidates(AnalyticAppearance(self.source, conflict_bonus=True), front, IDENTITY)
        self.assertTrue(result['accepted'])
        bad = result['diagnostic_hypotheses'][0]
        self.assertEqual(bad['lower'], 10.)
        self.assertFalse(bad['admitted_to_H'])
        self.assertIn('query_geometry_model_conflict', bad['rejections'])

    def test_independent_depth_conflict_rejects_and_keeps_a_diagnostic_region(self):
        self.source.update(check_xyz=np.array([[0., 0., .9]]),
                           check_normals=np.array([[0., 0., -1.]]),
                           check_indices=np.array([999]))
        result = self.run_candidates(AnalyticAppearance(self.source), IDENTITY)
        self.assertFalse(result['accepted'])
        self.assertGreater(len(result['modes']), 0)
        self.assertIn('independent_check_geometry_model_conflict',
                      result['diagnostic_hypotheses'][0]['rejections'])

    def test_multiple_transforms_common_region_never_returns_unique_pose(self):
        moved = (1., np.eye(3), np.array([1 / 256, 0., 0.]))
        result = self.run_candidates(AnalyticAppearance(self.source), IDENTITY, moved)
        self.assertTrue(result['accepted'])
        self.assertTrue(result['region_identified'])
        self.assertFalse(result['transform_identified'])
        self.assertIsNone(result['sim3_scale'])
        self.assertIsNone(result['sim3_rotation'])
        self.assertIsNone(result['sim3_translation'])
        self.assertIn('common_indices', result['modes'][0])

    def test_center_ambiguity_remains_a_competing_interval(self):
        moved = (1., np.eye(3), np.array([1 / 256, 0., 0.]))
        original = inference.geometry.region_for_transform
        def ambiguous(surface, source, transform, **sampling):
            if transform[2][0] != 0:
                return None, 'center_projection_ambiguous'
            return original(surface, source, transform, **sampling)
        with patch.object(inference.geometry, 'region_for_transform', side_effect=ambiguous):
            result = self.run_candidates(AnalyticAppearance(self.source), IDENTITY, moved)
        self.assertFalse(result['accepted'])
        self.assertIn('center_projection_ambiguous', result['reasons'])
        ambiguous_h = next(h for h in result['diagnostic_hypotheses'] if h['center_projection_ambiguous'])
        self.assertTrue(ambiguous_h['admitted_to_H'])
        self.assertTrue(ambiguous_h['potential_winner'])

    def test_missing_source_normal_keeps_fixed_denominator_and_loses_support(self):
        self.source['query_normals'][0] = np.nan
        self.source['query_normal_valid'][0] = False
        result = self.run_candidates(AnalyticAppearance(self.source), IDENTITY)
        self.assertFalse(result['accepted'])
        h = result['diagnostic_hypotheses'][0]
        self.assertEqual(result['diagnostics']['fixed_query_denominator'], 6)
        self.assertEqual(h['known_count'], 5)
        self.assertIn('fewer_than_six_oriented_query_supports', h['rejections'])

    def test_interrupted_plan_is_never_accepted(self):
        with patch.object(inference, 'build_proposals', side_effect=InterruptedError('synthetic resource stop')):
            result = inference.infer(self.source, self.surface, AnalyticAppearance(self.source))
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'], 'search_incomplete')
        self.assertEqual(result['diagnostics']['search_status'], 'search_incomplete')

    def test_more_than_thirty_two_candidates_are_all_scored_and_retained(self):
        transforms = [(1., np.eye(3), np.array([i / 8192, 0., 0.])) for i in range(40)]
        result = self.run_candidates(AnalyticAppearance(self.source), *transforms)
        self.assertEqual(len(result['diagnostic_hypotheses']), 40)
        self.assertEqual(result['diagnostics']['potential_winner_count'], 40)
        self.assertTrue(result['accepted'])
        self.assertFalse(result['transform_identified'])

    def test_end_to_end_finite_search_scores_every_generated_transform(self):
        result = inference.infer(self.source, self.surface, AnalyticAppearance(self.source),
            initial_transforms=[dict(transform=IDENTITY, origin='native_test', view_index=0)])
        self.assertEqual(result['diagnostics']['search_status'], 'finite_domain_complete')
        self.assertEqual(result['diagnostics']['evaluated_transform_count'],
                         len(result['diagnostic_hypotheses']))
        self.assertTrue(all(len(h['scores']) == 6 for h in result['diagnostic_hypotheses']))
        self.assertLessEqual(result['diagnostics']['refit_attempts'],
                             result['diagnostics']['proposals']['proposal_count'])


if __name__ == '__main__':
    unittest.main()
