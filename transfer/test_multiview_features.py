"""Cached per-view correspondence evidence retains source and surface identity.

These arrays stand in for already frozen joint_evidence records. No feature
tensor, MARCO call, image decoding, or historical fusion fixture is involved.
"""

import copy
import unittest

import numpy as np

from transfer.multiview_features import build_correspondence_candidates
from transfer.multiview_geometry import build_target_surface
from transfer.test_multiview_geometry import calibrated_view, element_at


def query_fixture():
    """Provide two distinct source queries with explicit region/context roles."""
    source = calibrated_view('source')['observation']
    source['region_mask'][0] = True
    return source, np.array([0, 1], dtype=np.int64), np.array([True, False])


def cached_evidence(view, query_indices, query_region, peaks, *, costs=None, supported=None):
    """Attach a finite native-target-indexed cache without synthesizing features."""
    count = len(view['observation']['xyz'])
    queries = len(query_indices)
    if costs is None:
        costs = np.full((queries, count), .6, dtype=np.float32)
    if supported is None:
        supported = np.ones(count, bool)
    view['evidence'] = {
        'query_indices': np.asarray(query_indices, dtype=np.int64).copy(),
        'query_region': np.asarray(query_region, dtype=bool).copy(),
        'peak_indices': np.asarray(peaks, dtype=np.int64),
        'appearance_cost': np.asarray(costs, dtype=np.float32),
        'supported_target': np.asarray(supported, dtype=bool),
    }
    return view


def candidate_column(evidence, element):
    """Require one shared element column rather than one column per observation."""
    columns = np.flatnonzero(evidence['candidate_indices'] == element)
    if len(columns) != 1:
        raise AssertionError('A proposed surface element must have exactly one column')
    return int(columns[0])


class TestCachedCorrespondenceCandidates(unittest.TestCase):
    def test_peak_union_preserves_query_edges_and_separate_view_costs(self):
        source, ids, roles = query_fixture()
        a, b = calibrated_view('a'), calibrated_view('b')
        costs_a = np.full((2, 81), .6, dtype=np.float32)
        costs_b = np.full((2, 81), .8, dtype=np.float32)
        costs_a[0, 40], costs_b[0, 40] = .12, .76
        costs_a[1, 41], costs_b[1, 42] = .22, .33
        # A small cost for another query's peak must not grant a new edge.
        costs_a[0, 41] = 0.
        cached_evidence(a, ids, roles, [[40, 40, -1], [41, -1, -1]], costs=costs_a)
        cached_evidence(b, ids, roles, [[40, -1, -1], [42, -1, -1]], costs=costs_b)
        surface = build_target_surface([a, b])
        result = build_correspondence_candidates(source, [a, b], surface,
                                                 query_indices=ids, query_region=roles)
        center = candidate_column(result, element_at(surface, 0, 4, 4))
        right = candidate_column(result, element_at(surface, 0, 5, 4))
        farther = candidate_column(result, element_at(surface, 1, 6, 4))
        self.assertEqual(result['allowed'].shape, (2, 3))
        expected = np.zeros((2, 3), bool)
        expected[0, center] = True
        expected[1, [right, farther]] = True
        np.testing.assert_array_equal(result['allowed'], expected)
        np.testing.assert_array_equal(result['edge_views'][0, center], [True, True])
        np.testing.assert_array_equal(result['edge_views'][1, right], [True, False])
        np.testing.assert_array_equal(result['edge_views'][1, farther], [False, True])
        self.assertEqual(result['appearance_cost'][0, center, 0], costs_a[0, 40])
        self.assertEqual(result['appearance_cost'][0, center, 1], costs_b[0, 40])
        self.assertTrue(result['appearance_available'][0, center].all())
        np.testing.assert_array_equal(result['query_indices'], ids)
        np.testing.assert_array_equal(result['query_region'], roles)

    def test_view_order_does_not_change_candidate_identity_or_costs(self):
        source, ids, roles = query_fixture()
        a = cached_evidence(calibrated_view('a'), ids, roles, [[40], [41]])
        b = cached_evidence(calibrated_view('b'), ids, roles, [[42], [43]])
        surface = build_target_surface([a, b])
        forward = build_correspondence_candidates(source, [a, b], surface,
                                                  query_indices=ids, query_region=roles)
        reverse = build_correspondence_candidates(source, [b, a], surface,
                                                  query_indices=ids, query_region=roles)
        for key in ('query_indices', 'query_region', 'candidate_indices', 'allowed',
                    'edge_views', 'appearance_cost', 'appearance_available'):
            np.testing.assert_array_equal(forward[key], reverse[key], err_msg=key)

    def test_missing_view_cache_stays_unavailable_without_erasing_depth(self):
        source, ids, roles = query_fixture()
        a = cached_evidence(calibrated_view('a'), ids, roles, [[40], [41]])
        pose = np.eye(4)
        pose[0, 3] = .1
        b = calibrated_view('b', T=pose, observation_available=False)
        surface = build_target_surface([a, b])
        result = build_correspondence_candidates(source, [a, b], surface,
                                                 query_indices=ids, query_region=roles)
        self.assertEqual(len(surface['observation_xyz']), 162)
        self.assertTrue(np.any(surface['member_views'][:, 1]))
        self.assertFalse(result['appearance_available'][:, :, 1].any())
        self.assertTrue(np.isnan(result['appearance_cost'][:, :, 1]).all())
        self.assertFalse(result['edge_views'][:, :, 1].any())
        self.assertTrue(result['allowed'].any())

    def test_no_appearance_cache_returns_empty_edges_but_keeps_scene_geometry(self):
        source, ids, roles = query_fixture()
        views = [calibrated_view('a', observation_available=False),
                 calibrated_view('b', observation_available=False)]
        surface = build_target_surface(views)
        result = build_correspondence_candidates(source, views, surface,
                                                 query_indices=ids, query_region=roles)
        self.assertEqual(len(surface['observation_xyz']), 162)
        self.assertFalse(surface['entity_member'].any())
        self.assertEqual(result['candidate_indices'].shape, (0,))
        self.assertEqual(result['allowed'].shape, (2, 0))
        self.assertEqual(result['appearance_cost'].shape, (2, 0, 2))

    def test_wrong_query_identity_or_region_roles_are_not_silently_rebound(self):
        source, ids, roles = query_fixture()
        original = cached_evidence(calibrated_view('a'), ids, roles, [[40], [41]])
        b = calibrated_view('b')
        for field, value in (('query_indices', ids[::-1]), ('query_region', ~roles)):
            with self.subTest(field=field):
                a = copy.deepcopy(original)
                a['evidence'][field] = value
                surface = build_target_surface([a, b])
                with self.assertRaises(ValueError):
                    build_correspondence_candidates(source, [a, b], surface,
                                                     query_indices=ids, query_region=roles)

    def test_unsupported_peak_stays_missing_even_if_another_view_proposes_it(self):
        source, ids, roles = query_fixture()
        a = cached_evidence(calibrated_view('a'), ids, roles, [[40], [41]])
        a['evidence']['supported_target'][40] = False
        b = cached_evidence(calibrated_view('b'), ids, roles, [[-1], [40]])
        surface = build_target_surface([a, b])
        result = build_correspondence_candidates(source, [a, b], surface,
                                                 query_indices=ids, query_region=roles)
        center = candidate_column(result, element_at(surface, 0, 4, 4))
        self.assertFalse(result['allowed'][0, center])
        self.assertTrue(result['allowed'][1, center])
        self.assertFalse(result['edge_views'][:, center, 0].any())
        self.assertFalse(result['appearance_available'][:, center, 0].any())
        self.assertTrue(np.isnan(result['appearance_cost'][:, center, 0]).all())
        self.assertTrue(result['appearance_available'][:, center, 1].all())
        self.assertFalse(a['evidence']['supported_target'][40])

    def test_unpaired_surface_cost_uses_projected_pixel_not_best_neighbor(self):
        source, ids, roles = query_fixture()
        a = cached_evidence(calibrated_view('a'), ids, roles, [[41], [-1]])
        coarse = a['K'].copy()
        coarse[:2, :2] *= .5
        costs = np.full((2, 81), .6, dtype=np.float32)
        costs[0, 40], costs[0, 41] = .7, .01
        b = cached_evidence(calibrated_view('b', K=coarse), ids, roles,
                            [[-1], [-1]], costs=costs)
        surface = build_target_surface([a, b])
        element = element_at(surface, 0, 5, 4)
        self.assertFalse(surface['member_views'][element, 1])
        self.assertTrue(surface['observed_views'][element, 1])
        result = build_correspondence_candidates(source, [a, b], surface,
                                                 query_indices=ids, query_region=roles)
        column = candidate_column(result, element)
        self.assertEqual(result['appearance_cost'][0, column, 1], costs[0, 40])
        expected_raw = surface['pixel_to_observation'][1][4, 4]
        self.assertEqual(result['cost_observation_indices'][column, 1], expected_raw)
        self.assertFalse(result['edge_views'][0, column, 1])

    def test_invalid_peak_index_is_a_contract_error(self):
        source, ids, roles = query_fixture()
        b = calibrated_view('b')
        for invalid in ('outside', 'negative'):
            with self.subTest(invalid=invalid):
                a = cached_evidence(calibrated_view('a'), ids, roles, [[40], [41]])
                a['evidence']['peak_indices'][0, 0] = 81 if invalid == 'outside' else -2
                surface = build_target_surface([a, b])
                with self.assertRaises(ValueError):
                    build_correspondence_candidates(source, [a, b], surface,
                                                     query_indices=ids, query_region=roles)


if __name__ == '__main__':
    unittest.main()
