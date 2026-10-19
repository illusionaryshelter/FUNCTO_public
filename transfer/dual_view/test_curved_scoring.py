"""Behavior of absolute geometry, appearance identity and fixed source weights."""

import unittest

import numpy as np

from .curved_scoring import (prepare_scoring, geometry_correspondences,
                             assigned_appearance, score_candidate)
from .features import EmpiricalSupport


def source_fixture():
    xyz = np.array([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.],
        [.5, .5, 0.], [.25, .2, .1], [-.25, .1, 0.],
        [1., 0., 0.], [0., 1., 0.], [3., 0., 0.], [0., -3., 1.]])
    region = np.arange(len(xyz)) < 6
    return dict(query_xyz=xyz, query_region=region,
        center=xyz[region].mean(0), entity_xyz=xyz.copy(), entity_region=region.copy())


def correspondence(target, known=None):
    n = len(target)
    known = np.ones(n, bool) if known is None else np.asarray(known, bool)
    return dict(target_xyz=np.asarray(target, float), matched=known)


# BEGIN new-score behavior, independent of any recorded candidate or task.
class CurvedScoringTests(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture()
        self.context = prepare_scoring(self.source)
        self.identity = (1., np.eye(3), np.zeros(3))
        self.target = self.source['query_xyz'].copy()

    def score(self, target=None, *, appearance=None, known=None, context=None,
              transform=None, tau=.1):
        target = self.target if target is None else target
        return score_candidate(self.context if context is None else context,
            self.identity if transform is None else transform,
            correspondence(target, known),
            np.zeros(len(target)) if appearance is None else appearance,
            registration_error_m=tau)

    def test_common_translation_is_visible_even_when_all_relations_are_unchanged(self):
        shifted = self.target + [.08, -.06, 0.]
        np.testing.assert_allclose(shifted[1:]-shifted[:-1],
            self.target[1:]-self.target[:-1], atol=1e-15)
        aligned, moved = self.score(), self.score(shifted)
        self.assertGreater(aligned['geometry_score'], moved['geometry_score'])
        self.assertGreater(aligned['score'], moved['score'])
        self.assertAlmostEqual(moved['geometry_score'], .5)

    def test_absolute_geometry_detects_wrong_entity_context(self):
        wrong = self.target.copy()
        wrong[~self.source['query_region']] += [0., .4, .5]
        aligned, moved = self.score(), self.score(wrong)
        self.assertEqual(moved['source_scopes'][0]['geometry_score'], 1.)
        self.assertLess(moved['source_scopes'][-1]['geometry_score'], 1.)
        self.assertLess(moved['score'], aligned['score'])

    def test_real_appearance_can_distinguish_equal_geometry(self):
        unsupported = self.score(appearance=np.zeros(len(self.target)))
        supported = self.score(appearance=np.linspace(.1, .9, len(self.target)))
        self.assertEqual(supported['geometry_score'], unsupported['geometry_score'])
        self.assertGreater(supported['appearance_score'], unsupported['appearance_score'])
        self.assertGreater(supported['score'], unsupported['score'])
        self.assertFalse(supported['is_probability'])

    def test_source_scopes_prioritize_region_without_candidate_normalization(self):
        weights = self.context['query_weights']
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertTrue(np.all(weights > 0))
        self.assertGreater(weights[0], weights[-1])
        missing = np.ones(len(weights), bool)
        missing[0] = False
        hidden = self.score(known=missing)
        np.testing.assert_array_equal(hidden['query_weights'], weights)
        self.assertAlmostEqual(hidden['geometry_known_weight'], 1.-weights[0])
        self.assertAlmostEqual(hidden['geometry_score'], 1.-weights[0])

    def test_duplicate_source_scopes_count_once(self):
        source = source_fixture()
        source['query_region'][:] = True
        source['entity_region'][:] = True
        source['center'] = source['query_xyz'].mean(0)
        context = prepare_scoring(source)
        self.assertEqual(context['unique_support_domain_count'], 1)
        self.assertEqual(len(context['scope_aliases'][0]), 3)
        np.testing.assert_array_equal(context['query_weights'], np.full(len(self.target), .1))

    def test_region_outside_entity_remains_in_each_scope(self):
        source = source_fixture()
        source['entity_xyz'] = source['query_xyz'][6:8].copy()
        source['entity_region'] = np.zeros(2, bool)
        source['query_xyz'][0] = [8., 0., 0.]
        source['center'] = source['query_xyz'][source['query_region']].mean(0)
        context = prepare_scoring(source)
        self.assertTrue(np.all(context['domains'][:, 0]))
        self.assertAlmostEqual(context['query_weights'].sum(), 1.)

    def test_hiding_fixed_correspondence_or_feature_never_adds_support(self):
        target = self.target.copy()
        target[-1] += [.04, .02, 0.]
        appearance = np.linspace(0., 1., len(target))
        full = self.score(target, appearance=appearance)
        for row in range(len(target)):
            known = np.ones(len(target), bool)
            known[row] = False
            geometry_hidden = self.score(target, appearance=appearance, known=known)
            features = appearance.copy()
            features[row] = np.nan
            appearance_hidden = self.score(target, appearance=features)
            self.assertLessEqual(geometry_hidden['score'], full['score'])
            self.assertLessEqual(appearance_hidden['score'], full['score'])
            self.assertIsNone(geometry_hidden['absolute_position_residuals_m'][row])
            self.assertIsNone(appearance_hidden['appearance_support'][row])

    def test_missing_and_zero_support_appearance_share_formula_but_keep_distinct_states(self):
        missing = self.score(appearance=np.full(len(self.target), np.nan))
        measured = self.score(appearance=np.zeros(len(self.target)))
        self.assertEqual(missing['score'], measured['score'])
        self.assertEqual(missing['appearance_known_weight'], 0.)
        self.assertAlmostEqual(measured['appearance_known_weight'], 1.)
        self.assertEqual(missing['appearance_support'], [None]*len(self.target))
        self.assertEqual(measured['appearance_support'], [0.]*len(self.target))

    def test_proper_similarity_and_change_of_length_units_preserve_score(self):
        angle = .8
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        transform = (2.3, rotation, np.array([4., -2., 1.]))
        target = transform[0]*self.target@rotation.T+transform[2]
        self.assertAlmostEqual(self.score(target, transform=transform)['geometry_score'], 1.)
        target[-1] += [.02, .04, .01]
        first = self.score(target, transform=transform)
        scaled = dict(self.context, xyz=self.context['xyz']*1000.)
        second = self.score(target*1000., context=scaled,
            transform=(transform[0], rotation, transform[2]*1000.), tau=100.)
        self.assertAlmostEqual(first['score'], second['score'])

    def test_other_camera_cannot_lend_a_high_response_to_assigned_measurement(self):
        reference = np.array([[0., .5, 1.], [0., .5, 1.]])
        supports = tuple(EmpiricalSupport(reference, np.ones_like(reference, bool)) for _ in range(2))
        assigned = dict(matched=np.ones(2, bool), selected_view=np.array([0, 1]))
        sampled = dict(view_scores=np.array([[.2, 1000.], [1000., .8]]))
        actual = assigned_appearance(sampled, assigned, supports)
        np.testing.assert_array_equal(actual['raw_responses'], [.2, .8])
        np.testing.assert_allclose(actual['utilities'], [0., 1./3.])
        sampled['view_scores'][0, 1] = -1000.
        sampled['view_scores'][1, 0] = -1000.
        repeated = assigned_appearance(sampled, assigned, supports)
        np.testing.assert_array_equal(actual['utilities'], repeated['utilities'])

    def test_unknown_selected_camera_cannot_use_other_camera(self):
        reference = np.array([[0., .5, 1.]])
        supports = tuple(EmpiricalSupport(reference, np.ones_like(reference, bool)) for _ in range(2))
        assigned = dict(matched=np.ones(1, bool), selected_view=np.array([0]))
        actual = assigned_appearance(dict(view_scores=np.array([[np.nan, 1000.]])), assigned, supports)
        self.assertTrue(np.isnan(actual['raw_responses'][0]))
        self.assertTrue(np.isnan(actual['utilities'][0]))


class GeometryAssignmentTests(unittest.TestCase):
    """Unchanged measurement-capacity behavior remains a separate contract."""

    def test_matching_prioritizes_region_and_preserves_capacity(self):
        surface = dict(observation_xyz=np.array([[0., 0., 0.], [1., 0., 0.]]),
            observation_position_error_m=np.zeros(2), observation_to_element=np.array([0, 1]))
        projections = [dict(supported=np.array([True, True, True]),
            observation_indices=np.array([0, 0, 1]), residual_m=np.array([.8, .1, .3])),
            dict(supported=np.zeros(3, bool), observation_indices=np.full(3, -1),
                residual_m=np.full(3, np.nan))]
        result = geometry_correspondences(surface, projections, np.array([True, False, False]))
        np.testing.assert_array_equal(result['matched'], [True, False, True])
        np.testing.assert_array_equal(result['capacity_lost'], [False, True, False])
        self.assertEqual(len(np.unique(result['element_ids'][result['matched']])), 2)

    def test_matching_total_cardinality_precedes_residual(self):
        surface = dict(observation_xyz=np.array([[0., 0., 0.], [1., 0., 0.]]),
            observation_position_error_m=np.zeros(2), observation_to_element=np.array([0, 1]))
        projections = [dict(supported=np.ones(2, bool), observation_indices=np.array([0, 0]),
            residual_m=np.array([0., 0.])), dict(supported=np.array([True, False]),
                observation_indices=np.array([1, -1]), residual_m=np.array([100., np.nan]))]
        result = geometry_correspondences(surface, projections, np.array([True, False]))
        np.testing.assert_array_equal(result['element_ids'], [1, 0])

    def test_same_element_uses_actual_lower_residual_camera(self):
        surface = dict(observation_xyz=np.array([[0., 0., 0.], [0., .01, 0.]]),
            observation_position_error_m=np.array([.001, .002]),
            observation_to_element=np.array([0, 0]))
        projections = [dict(supported=np.array([True]), observation_indices=np.array([0]),
            residual_m=np.array([.02])), dict(supported=np.array([True]),
                observation_indices=np.array([1]), residual_m=np.array([.01]))]
        result = geometry_correspondences(surface, projections, np.array([True]))
        self.assertEqual(result['selected_view'][0], 1)
        np.testing.assert_array_equal(result['target_xyz'][0], surface['observation_xyz'][1])
        self.assertEqual(result['observation_indices'][0], 1)

    def test_empty_projection_returns_unknown_rows(self):
        surface = dict(observation_xyz=np.empty((0, 3)),
            observation_position_error_m=np.empty(0), observation_to_element=np.empty(0, int))
        projection = dict(supported=np.zeros(3, bool), observation_indices=np.full(3, -1),
            residual_m=np.full(3, np.nan))
        result = geometry_correspondences(surface, [projection, projection], np.ones(3, bool))
        self.assertFalse(result['matched'].any())
        self.assertTrue(result['unknown'].all())
        self.assertFalse(result['capacity_lost'].any())


# END score tests distinguish actual support from missing observations.

if __name__ == '__main__':
    unittest.main()
