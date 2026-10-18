"""Small CPU tests for optional assignment and fixed-support Sim(3) refits."""

import unittest

import numpy as np
from scipy.spatial import cKDTree

from .refinement import partial_refits


def _fixture():
    xyz = np.array([[x, y, 0.] for y in (0., .1, .2) for x in (0., .1, .2, .3)])
    q = len(xyz)
    source = dict(query_xyz=xyz, query_region=np.arange(q) < 6,
        query_normals=np.tile([0., 0., 1.], (q, 1)),
        # BEGIN fixture declares its original planar geometry explicitly.
        query_normal_valid=np.ones(q, bool), size=1., region_geometry='planar')
        # END explicit fixture geometry.
    surface = dict(xyz=xyz.copy(), normals=np.tile([0., 0., 1.], (q, 1)),
                   normal_min_agreement=.65)
    domain = dict(surface_ids=np.arange(q), xyz=surface['xyz'],
        tree=cKDTree(surface['xyz']), utility=np.eye(q, dtype=np.float32),
        inverse=np.zeros((q, q), dtype=np.int32))
    proposal = dict(proposal_id=0, transform=(1., np.eye(3), np.array([.01, 0., 0.])))
    return source, surface, domain, proposal


class PartialRefitTests(unittest.TestCase):
    def test_one_assignment_corrects_translation_in_both_fixed_support_subsets(self):
        source, surface, domain, proposal = _fixture()
        candidates, diagnostic = partial_refits(source, surface, domain, proposal)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(diagnostic['assigned_query_count'], 12)
        self.assertEqual(len(set(diagnostic['assigned_surface_ids'])), 12)
        for candidate in candidates:
            scale, rotation, translation = candidate['transform']
            self.assertAlmostEqual(scale, 1.)
            np.testing.assert_allclose(rotation, np.eye(3), atol=1e-14)
            np.testing.assert_allclose(translation, 0., atol=1e-14)
        np.testing.assert_array_equal(proposal['transform'][2], [.01, 0., 0.])

    def test_unknown_rows_cannot_be_forced_to_six_real_matches(self):
        source, surface, domain, proposal = _fixture()
        domain['inverse'][:] = -1
        candidates, diagnostic = partial_refits(source, surface, domain, proposal)
        self.assertEqual(candidates, [])
        self.assertEqual(diagnostic['assigned_query_count'], 0)

    def test_neutral_edges_remain_unmatched_even_at_zero_distance(self):
        source, surface, domain, proposal = _fixture()
        proposal['transform'] = (1., np.eye(3), np.zeros(3))
        domain['utility'][:] = 0
        candidates, diagnostic = partial_refits(source, surface, domain, proposal)
        self.assertEqual(candidates, [])
        self.assertEqual(diagnostic['assigned_query_count'], 0)

    def test_target_capacity_does_not_allow_one_point_to_supply_six_supports(self):
        source, surface, domain, proposal = _fixture()
        domain.update(surface_ids=np.array([0]), xyz=surface['xyz'][:1],
            tree=cKDTree(surface['xyz'][:1]), utility=np.ones((12, 1), np.float32),
            inverse=np.zeros((12, 1), np.int32))
        source['size'] = 10.
        candidates, diagnostic = partial_refits(source, surface, domain, proposal)
        self.assertEqual(candidates, [])
        self.assertEqual(diagnostic['assigned_query_count'], 1)

    def test_known_opposite_normals_block_edges_unknown_normals_do_not_invent_them(self):
        source, surface, domain, proposal = _fixture()
        source['query_normals'] *= -1
        candidates, diagnostic = partial_refits(source, surface, domain, proposal)
        self.assertEqual(candidates, [])
        self.assertEqual(diagnostic['assigned_query_count'], 0)
        source['query_normal_valid'][:] = False
        source['query_normals'][:] = np.nan
        candidates, diagnostic = partial_refits(source, surface, domain, proposal)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.isnan(source['query_normals']).all())

    def test_far_high_utility_points_do_not_enter_spatial_union(self):
        source, surface, domain, proposal = _fixture()
        far = surface['xyz'] + 10.
        surface['xyz'] = np.vstack((surface['xyz'], far))
        surface['normals'] = np.tile([0., 0., 1.], (24, 1))
        domain.update(surface_ids=np.arange(24), xyz=surface['xyz'],
            tree=cKDTree(surface['xyz']),
            utility=np.column_stack((domain['utility'], np.ones((12, 12), np.float32))),
            inverse=np.zeros((12, 24), np.int32))
        candidates, diagnostic = partial_refits(source, surface, domain, proposal)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(diagnostic['candidate_union_count'], 12)
        self.assertEqual(diagnostic['distance_pair_count'], 12 * 12)

    def test_region_fit_does_not_average_remote_context_displacement(self):
        source, surface, domain, proposal = _fixture()
        surface['xyz'][6:, 0] += .01
        domain['tree'] = cKDTree(domain['xyz'])
        candidates, diagnostic = partial_refits(source, surface, domain, proposal)
        self.assertEqual(len(candidates), 2)
        local, full = candidates
        np.testing.assert_allclose(local['transform'][2], 0., atol=1e-14)
        self.assertGreater(np.linalg.norm(full['transform'][2]), .001)
        self.assertEqual(diagnostic['fits'][0]['fixed_region_count'], 6)

    def test_full_fit_requires_region_evidence_and_local_fit_requires_six_local_pairs(self):
        source, surface, domain, proposal = _fixture()
        source['query_region'][:] = False
        source['query_region'][0] = True
        candidates, diagnostic = partial_refits(source, surface, domain, proposal)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['origins'][0]['origin'], 'one_partial_all_refit')
        source['query_region'][:] = False
        candidates, diagnostic = partial_refits(source, surface, domain, proposal)
        self.assertEqual(candidates, [])


if __name__ == '__main__':
    unittest.main()
