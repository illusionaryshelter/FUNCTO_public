"""Small reconstruction, cached-candidate and actual-inference checks.

This replaces response averaging tests with the public reconstruction and
cached-candidate interfaces. It never loads feature tensors or image models.
"""

import unittest

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from transfer.core import TransferConfig
from transfer.multiview_features import build_correspondence_candidates
from transfer.multiview_geometry import build_target_surface, sample_region
from transfer.multiview_grounding import infer_multiview_region
from transfer.test_multiview_features import cached_evidence, candidate_column, query_fixture
from transfer.test_multiview_geometry import calibrated_view, element_at


def _calibrated_source_queries():
    """Use exactly representable planar samples with five task/four context queries."""
    calibration = np.array([[256., 0., 4.], [0., 256., 4.], [0., 0., 1.]])
    source = calibrated_view('source', K=calibration)['observation']
    source['region_mask'] = np.all(np.abs(source['pixels']-4) <= 2, axis=1)
    queries = np.array([20, 24, 40, 56, 60, 0, 8, 72, 80])
    return source, calibration, queries, source['region_mask'][queries]


def calibrated_pipeline_fixture(*, competing_regions=False):
    """Build real depth geometry and evidence; no inference helper is mocked.

    The single region lacks B's appearance observation but retains B's depth.
    The repeated-region case gives each camera a different entity observation.
    Both cameras still see the complete plane; their task candidates remain
    equally supported at two distinct locations. Binary-exact coordinates keep
    this deliberate geometric tie independent of decimal roundoff.
    """
    source, calibration, queries, roles = _calibrated_source_queries()
    if competing_regions:
        calibration = calibration.copy()
        calibration[0, 2] = 20
        mask_a, mask_b = np.zeros((9, 41), bool), np.zeros((9, 41), bool)
        mask_a[:, :9], mask_b[:, 32:] = True, True
        a = cached_evidence(calibrated_view('A', np.ones((9, 41)), K=calibration, mask=mask_a),
                            queries, roles, queries[:, None])
        b = cached_evidence(calibrated_view('B', np.ones((9, 41)), K=calibration, mask=mask_b),
                            queries, roles, queries[:, None])
    else:
        a = cached_evidence(calibrated_view('A', K=calibration), queries, roles, queries[:, None])
        b = calibrated_view('B', K=calibration, observation_available=False)
    surface = build_target_surface([a, b])
    evidence = build_correspondence_candidates(source, [a, b], surface,
                                               query_indices=queries, query_region=roles)
    return source, surface, evidence


class TestObservedSurfaceCandidates(unittest.TestCase):
    def test_b_exclusive_observation_enters_same_query_candidate_set(self):
        source, ids, roles = query_fixture()
        a = cached_evidence(calibrated_view('a'), ids, roles, [[40], [41]])
        pose = np.eye(4)
        pose[0, 3] = .01
        b = cached_evidence(calibrated_view('b', T=pose), ids, roles, [[43], [44]])
        surface = build_target_surface([a, b])
        evidence = build_correspondence_candidates(source, [a, b], surface,
                                                   query_indices=ids, query_region=roles)
        a_element = element_at(surface, 0, 4, 4)
        b_element = element_at(surface, 1, 7, 4)
        self.assertNotEqual(a_element, b_element)
        a_column = candidate_column(evidence, a_element)
        b_column = candidate_column(evidence, b_element)
        self.assertTrue(evidence['allowed'][0, a_column])
        self.assertTrue(evidence['allowed'][0, b_column])
        np.testing.assert_array_equal(evidence['edge_views'][0, b_column], [False, True])
        np.testing.assert_array_equal(surface['member_views'][b_element], [False, True])
        np.testing.assert_array_equal(surface['observed_views'][b_element], [False, True])
        self.assertFalse(evidence['appearance_available'][0, b_column, 0])
        self.assertTrue(np.isnan(evidence['appearance_cost'][0, b_column, 0]))

    def test_shared_peak_has_one_target_column_and_two_observation_witnesses(self):
        source, ids, roles = query_fixture()
        a = cached_evidence(calibrated_view('a'), ids, roles, [[40, 40], [40, 40]])
        b = cached_evidence(calibrated_view('b'), ids, roles, [[40, 40], [40, 40]])
        surface = build_target_surface([a, b])
        evidence = build_correspondence_candidates(source, [a, b], surface,
                                                   query_indices=ids, query_region=roles)
        self.assertEqual(len(evidence['candidate_indices']), 1)
        np.testing.assert_array_equal(evidence['allowed'], np.ones((2, 1), bool))
        np.testing.assert_array_equal(evidence['edge_views'], np.ones((2, 1, 2), bool))
        # Unique-target assignment occurs later. Candidate construction must
        # retain both source rows rather than awarding this point to one query.
        np.testing.assert_array_equal(evidence['query_indices'], ids)

    def test_geometry_can_fill_a_gap_without_inventing_unknown_entity_membership(self):
        source, ids, roles = query_fixture()
        depth = np.ones((9, 9))
        depth[4] = np.nan
        a = calibrated_view('a', depth)
        # Native target indices address only a's actually observed entity.
        pixels = a['observation']['pixels']
        upper = int(np.flatnonzero(np.all(pixels == [4, 3], axis=1))[0])
        lower = int(np.flatnonzero(np.all(pixels == [4, 5], axis=1))[0])
        cached_evidence(a, ids, roles, [[upper], [lower]])
        b = calibrated_view('b', observation_available=False)
        surface = build_target_surface([a, b])
        evidence = build_correspondence_candidates(source, [a, b], surface,
                                                   query_indices=ids, query_region=roles)
        self.assertEqual(len(surface['observation_xyz']), 72+81)
        self.assertEqual(connected_components(surface['adjacency'], directed=False)[0], 1)
        unseen_entity_row = np.array([element_at(surface, 1, x, 4) for x in range(9)])
        self.assertFalse(surface['entity_member'][unseen_entity_row].any())
        self.assertFalse(np.isin(evidence['candidate_indices'], unseen_entity_row).any())
        self.assertFalse(evidence['appearance_available'][:, :, 1].any())
        seed = element_at(surface, 0, 4, 3)
        mode = sample_region(surface, surface['entity_member'] & surface['valid'], seed,
                             sampling_radius_m=.03, sampling_spacing_m=.005,
                             config=TransferConfig())
        self.assertFalse(mode['mask'][unseen_entity_row].any())
        self.assertFalse(mode['mask'][element_at(surface, 0, 4, 5)])
        self.assertTrue(mode['mask'][seed])


class TestCalibratedInference(unittest.TestCase):
    def test_real_pipeline_accepts_one_region_with_missing_b_appearance_warning(self):
        source, surface, evidence = calibrated_pipeline_fixture()
        result = infer_multiview_region(source, surface, evidence)
        self.assertTrue(result['accepted'], result['reasons'])
        self.assertEqual(result['reasons'], [])
        self.assertEqual(result['diagnostics']['quality_warnings'], ['empty_common_appearance_domain:B'])
        self.assertEqual(len(surface['observation_xyz']), 162)
        self.assertEqual(result['modes'][0]['type'], 'primary')
        np.testing.assert_allclose(result['modes'][0]['center'], [0., 0., 1.], atol=1e-12)
        np.testing.assert_allclose(result['sim3_translation'], 0., atol=1e-12)
        np.testing.assert_array_equal(source['xyz'][result['src_anchor_idx']],
                                      surface['xyz'][result['tgt_anchor_idx']])
        check_ids = result['diagnostics']['check_source_indices']
        self.assertFalse(set(check_ids) & set(evidence['query_indices']))
        chosen = result['diagnostics']['hypotheses'][result['diagnostics']['primary_hypothesis_index']]
        self.assertEqual(chosen['check_support_count'], len(check_ids))
        self.assertEqual(chosen['check_conflict_count'], 0)
        self.assertEqual(chosen['check_unknown_count'], 0)
        self.assertEqual(chosen['final_observed_view_counts']['both'], len(evidence['query_indices']))

    def test_real_union_pipeline_recovers_when_each_view_has_fewer_than_six_edges(self):
        source, calibration, queries, roles = _calibrated_source_queries()
        peaks_a, peaks_b = np.full((9, 1), -1), np.full((9, 1), -1)
        peaks_a[:4, 0], peaks_b[4:, 0] = queries[:4], queries[4:]
        a = cached_evidence(calibrated_view('A', K=calibration), queries, roles, peaks_a)
        b = cached_evidence(calibrated_view('B', K=calibration), queries, roles, peaks_b)
        surface = build_target_surface([a, b])
        evidence = build_correspondence_candidates(source, [a, b], surface,
                                                   query_indices=queries, query_region=roles)
        result = infer_multiview_region(source, surface, evidence)
        self.assertTrue(result['accepted'], result['reasons'])
        self.assertEqual(result['diagnostics']['fit_attempts']['A'], 0)
        self.assertEqual(result['diagnostics']['fit_attempts']['B'], 0)
        self.assertGreater(result['diagnostics']['fit_attempts']['union'], 0)
        self.assertEqual(result['diagnostics']['matched_query_count'], 9)
        self.assertEqual(result['diagnostics']['quality_warnings'], [])
        selected = result['diagnostics']['hypotheses'][result['diagnostics']['primary_hypothesis_index']]
        self.assertEqual(selected['final_edge_sources'], {'A_only': 4, 'B_only': 5, 'both': 0})
        self.assertTrue(selected['uses_exclusive_edges_from_both_views'])

    def test_real_pipeline_uses_tilted_camera_geometry_without_world_axis_assumption(self):
        source, calibration, queries, roles = _calibrated_source_queries()
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec([.4, -.2, .1]).as_matrix()
        pose[:3, 3] = [.3, -.5, .2]
        a = cached_evidence(calibrated_view('A', K=calibration, T=pose),
                            queries, roles, queries[:, None])
        b = calibrated_view('B', K=calibration, T=pose, observation_available=False)
        surface = build_target_surface([a, b])
        evidence = build_correspondence_candidates(source, [a, b], surface,
                                                   query_indices=queries, query_region=roles)
        result = infer_multiview_region(source, surface, evidence)
        self.assertTrue(result['accepted'], result['reasons'])
        np.testing.assert_allclose(result['sim3_scale'], 1., atol=1e-12)
        np.testing.assert_allclose(result['sim3_rotation'], pose[:3, :3], atol=1e-12)
        np.testing.assert_allclose(result['sim3_translation'], pose[:3, 3], atol=1e-12)
        expected_center = pose[:3, :3] @ np.array([0., 0., 1.]) + pose[:3, 3]
        np.testing.assert_allclose(result['modes'][0]['center'], expected_center, atol=1e-12)
        np.testing.assert_allclose(result['modes'][0]['normal'], -pose[:3, 2], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
