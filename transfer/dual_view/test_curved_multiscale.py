"""Behavioral checks of source relations, missing evidence and measurement bounds."""

import unittest

import numpy as np

from .curved_multiscale import prepare_multiscale, geometry_correspondences, score_multiscale


def source_fixture(error=0.):
    xyz = np.array([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.],
        [.5, .5, 0.], [.25, .2, .1], [-.25, .1, 0.],
        [1., 0., 0.], [0., 1., 0.], [3., 0., 0.], [0., -3., 1.]])
    region = np.arange(len(xyz)) < 6
    return dict(query_xyz=xyz, query_region=region,
        query_position_error_m=np.full(len(xyz), error),
        center=xyz[region].mean(0), entity_xyz=xyz.copy(), entity_region=region.copy())


def correspondence(target, *, known=None, error=0., lost=None):
    n = len(target)
    known = np.ones(n, bool) if known is None else np.asarray(known, bool)
    return dict(target_xyz=np.asarray(target, float), matched=known,
        target_position_error_m=np.full(n, error), unknown=~known,
        capacity_lost=np.zeros(n, bool) if lost is None else np.asarray(lost, bool))


class CurvedMultiscaleTests(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture()
        self.context = prepare_multiscale(self.source)
        self.identity = (1., np.eye(3), np.zeros(3))

    def score(self, target, *, known=None, error=0., lost=None, context=None, transform=None):
        return score_multiscale(self.context if context is None else context,
            self.identity if transform is None else transform,
            correspondence(target, known=known, error=error, lost=lost),
            registration_error_m=.1)

    def test_local_copy_is_disambiguated_by_relations_to_entity(self):
        correct = self.score(self.source['query_xyz'])
        wrong = self.source['query_xyz'].copy()
        wrong[~self.source['query_region']] += [0., .4, .5]
        shifted = self.score(wrong)
        self.assertEqual(shifted['scales'][0]['observed_score'], 1.)
        self.assertLess(shifted['scales'][2]['upper_bound'], 1.)
        self.assertLess(shifted['upper_bound'], correct['lower_bound'])

    def test_partial_truth_is_not_eliminated_by_wrong_complete_coverage(self):
        known = np.arange(len(self.source['query_xyz'])) % 2 == 0
        correct = self.score(self.source['query_xyz'], known=known)
        wrong_xyz = self.source['query_xyz'].copy()
        wrong_xyz[~self.source['query_region']] += [0., .1, 0.]
        wrong = self.score(wrong_xyz)
        self.assertGreater(wrong['score'], correct['score'])
        self.assertGreaterEqual(correct['upper_bound'], wrong['lower_bound'])
        self.assertEqual(correct['upper_bound'], 1.)

    def test_hiding_any_measurement_only_widens_interval(self):
        target = self.source['query_xyz'].copy()
        target[-1] += [.04, .02, 0.]
        full = self.score(target, error=.001)
        for row in range(len(target)):
            known = np.ones(len(target), bool)
            known[row] = False
            hidden = self.score(target, known=known, error=.001)
            self.assertLessEqual(hidden['lower_bound'], full['lower_bound'])
            self.assertGreaterEqual(hidden['upper_bound'], full['upper_bound'])

    def test_measurement_interval_contains_bounded_endpoint_perturbations(self):
        source = source_fixture(.01)
        context = prepare_multiscale(source)
        target = source['query_xyz'].copy()
        target[-2:] += [.03, -.02, .04]
        bounded = self.score(target, error=.02, context=context)
        rng = np.random.default_rng(7)
        for _ in range(100):
            source_delta = rng.normal(size=target.shape)
            source_delta *= .01/np.linalg.norm(source_delta, axis=1)[:, None]
            target_delta = rng.normal(size=target.shape)
            target_delta *= .02/np.linalg.norm(target_delta, axis=1)[:, None]
            exact_context = dict(context, xyz=context['xyz']+source_delta,
                source_position_error_m=np.zeros(len(target)))
            exact = self.score(target+target_delta, context=exact_context)
            self.assertLessEqual(bounded['lower_bound'], exact['score'])
            self.assertGreaterEqual(bounded['upper_bound'], exact['score'])

    def test_proper_similarity_and_world_units_preserve_relations(self):
        angle = .8
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
            [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        transform = (2.3, rotation, np.array([4., -2., 1.]))
        target = transform[0]*self.source['query_xyz']@rotation.T+transform[2]
        result = self.score(target, transform=transform)
        self.assertAlmostEqual(result['score'], 1.)
        # A common change of length units scales both measurements and tau.
        context = dict(self.context, xyz=self.context['xyz']*1000,
            source_position_error_m=self.context['source_position_error_m']*1000)
        target[-1] += [.02, .04, .01]
        first = score_multiscale(self.context, transform, correspondence(target),
            registration_error_m=.1)
        scaled_transform = (transform[0], rotation, transform[2]*1000)
        second = score_multiscale(context, scaled_transform, correspondence(target*1000),
            registration_error_m=100.)
        self.assertAlmostEqual(first['score'], second['score'])

    def test_coincident_domains_do_not_require_external_context(self):
        source = source_fixture()
        source['query_region'][:] = True
        source['entity_region'][:] = True
        source['center'] = source['query_xyz'].mean(0)
        context = prepare_multiscale(source)
        result = self.score(source['query_xyz'], context=context)
        self.assertEqual(context['unique_support_domain_count'], 1)
        self.assertEqual(result['score'], 1.)

    def test_region_rows_outside_entity_remain_in_nested_domains(self):
        source = source_fixture()
        source['entity_xyz'] = source['query_xyz'][6:8].copy()
        source['entity_region'] = np.zeros(2, bool)
        source['query_xyz'][0] = [8., 0., 0.]
        source['center'] = source['query_xyz'][source['query_region']].mean(0)
        context = prepare_multiscale(source)
        self.assertGreaterEqual(context['radii_m'][2], context['radii_m'][0])
        self.assertTrue(np.all(~context['domains'][0] | context['domains'][1]))
        self.assertTrue(np.all(~context['domains'][1] | context['domains'][2]))
        self.assertTrue(np.all(context['domains'][:, 0]))

    def test_two_exact_repeated_embeddings_remain_equal(self):
        first = self.score(self.source['query_xyz'])
        offset = np.array([5., 2., -1.])
        second = self.score(self.source['query_xyz']+offset,
            transform=(1., np.eye(3), offset))
        self.assertEqual(first['lower_bound'], second['lower_bound'])
        self.assertEqual(first['upper_bound'], second['upper_bound'])

    def test_scale_multiplies_source_measurement_bound(self):
        source = source_fixture(.01)
        context = prepare_multiscale(source)
        scaled = self.score(source['query_xyz']*3., context=context,
            transform=(3., np.eye(3), np.zeros(3)))
        self.assertAlmostEqual(scaled['lower_bound'], 1./(1.+(.06/.2)**2))
        self.assertEqual(scaled['upper_bound'], 1.)

    def test_capacity_lost_is_unresolved_not_false_geometry(self):
        known = np.ones(len(self.source['query_xyz']), bool)
        known[-1] = False
        lost = ~known
        result = self.score(self.source['query_xyz'], known=known, lost=lost)
        self.assertEqual(result['upper_bound'], 1.)
        self.assertGreater(result['scales'][2]['capacity_unresolved_relation_count'], 0)

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
        self.assertEqual(result['target_position_error_m'][0], .002)

    def test_empty_projection_returns_unknown_rows(self):
        surface = dict(observation_xyz=np.empty((0, 3)),
            observation_position_error_m=np.empty(0), observation_to_element=np.empty(0, int))
        projection = dict(supported=np.zeros(3, bool), observation_indices=np.full(3, -1),
            residual_m=np.full(3, np.nan))
        result = geometry_correspondences(surface, [projection, projection], np.ones(3, bool))
        self.assertFalse(result['matched'].any())
        self.assertTrue(result['unknown'].all())
        self.assertFalse(result['capacity_lost'].any())


if __name__ == '__main__':
    unittest.main()
