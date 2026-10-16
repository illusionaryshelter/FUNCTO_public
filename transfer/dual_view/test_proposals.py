"""Targeted coverage and shared-domain tests independent of final inference."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist

from .proposals import (FIT_BUDGET, GROWTH_BUDGET, build_proposals,
                        prefix_length, rank_domain)


class _Appearance:
    def __init__(self, queries):
        self.queries = queries
        self.calls = 0

    def sample(self, points, normals):
        self.calls += 1
        scores = -cdist(self.queries, points)
        return {'scores': scores, 'known': np.ones(scores.shape, bool),
            'reason_counts': {view: {'observed': len(points)} for view in ('A', 'B')}}


def _fixture():
    xyz = np.array([[x, y, 0.] for y in (0., .01, .02) for x in (0., .01, .02, .03)])
    source = {'query_xyz': xyz, 'query_region': np.arange(len(xyz)) < 6, 'size': .02}
    surface = {'xyz': xyz.copy(), 'valid': np.ones(len(xyz), bool),
        'entity_member': np.ones(len(xyz), bool),
        'spacing': np.full(len(xyz), .001),
        'normals': np.tile([0., 0., 1.], (len(xyz), 1)), 'view_ids': ['A', 'B']}
    return source, surface, _Appearance(xyz)


class CoverageAndReuseTests(unittest.TestCase):
    def test_full_prefix_cannot_be_cut_off_by_two_fits_per_growth(self):
        source, surface, appearance = _fixture()
        self.assertEqual(2 * GROWTH_BUDGET, FIT_BUDGET)
        self.assertEqual(prefix_length(12345, GROWTH_BUDGET - 1), 12345)
        _, diagnostic, _ = build_proposals(source, surface, appearance)
        self.assertEqual(diagnostic['growth_attempts'], GROWTH_BUDGET)
        self.assertTrue(diagnostic['full_domain_prefix_reached'])
        self.assertLessEqual(diagnostic['proposal_fit_attempts'], FIT_BUDGET)

    def test_shared_domain_avoids_resampling_and_is_bound_to_exact_inputs(self):
        source, surface, appearance = _fixture()
        ids, order, inverse, diagnostic, domain = rank_domain(source, surface, appearance)
        calls = appearance.calls
        with patch.object(appearance, 'sample', side_effect=AssertionError('duplicate ranking')):
            _, reused, returned = build_proposals(source, surface, appearance, prepared_domain=domain)
        self.assertIs(returned, domain)
        self.assertTrue(reused['ranking_reused'])
        self.assertEqual(reused['ranking_seconds'], 0.)
        self.assertEqual(appearance.calls, calls)
        self.assertEqual(domain['utility'].dtype, np.float32)
        with self.assertRaisesRegex(ValueError, 'different prepared inputs'):
            build_proposals(dict(source), surface, appearance, prepared_domain=domain)

    def test_fixed_domain_utility_uses_tie_aware_reference_distribution(self):
        source, surface, appearance = _fixture()
        ids, order, inverse, diagnostic, domain = rank_domain(source, surface, appearance)
        raw = -cdist(source['query_xyz'], surface['xyz'][ids])
        for row in range(len(raw)):
            for col, score in enumerate(raw[row]):
                percentile = (np.count_nonzero(raw[row] < score)
                              + .5 * np.count_nonzero(raw[row] == score)) / len(ids)
                self.assertAlmostEqual(float(domain['utility'][row, col]),
                                       max(2 * percentile - 1, 0), places=6)
        self.assertEqual(diagnostic['ranking_work_qn'], len(raw) * len(ids))


if __name__ == '__main__':
    unittest.main()
