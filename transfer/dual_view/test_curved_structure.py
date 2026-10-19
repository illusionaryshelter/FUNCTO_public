"""Surface identity tests using measured RGB-D and the real appearance field.

The cases distinguish unseen geometry from refuting measurements, prevent
duplicate camera views from manufacturing independent support, and exercise
finite-candidate ambiguity independently of exact appearance-score equality.
No learned model, simulator, ground truth, or GPU is required.
"""

import unittest
from unittest.mock import patch

import numpy as np

from . import curved_geometry as geometry
from .curved_features import SurfaceResponseField
from .curved_inference import evaluate_candidates
from .curved_structure import compare_transforms, entity_evidence
from .proposals import rank_domain
from .test_curved_geometry import observation, views
from . import test_curved_inference as inference_fixture


class CurvedStructureTests(unittest.TestCase):
    def setUp(self):
        # Reuse the established six-query fixture, but retain its full 625
        # measured entity points. The other 619 points are real context even
        # though no context descriptors were selected in this small fixture.
        fixture = inference_fixture.CurvedInferenceTests()
        fixture.setUp()
        self.fixture = fixture
        self.identity = (1., np.eye(3), np.zeros(3))

    def _target_with_unqueried_patch(self, replacement_depth):
        """Change real depth away from every sparse query/check observation.

        A 3-by-3 constant patch supplies complete neighboring depth rays, so a
        nearer source point can be reliable conflict evidence at its center.
        Selecting an empty sparse-sampling cell is independent of any score.
        """
        fixture = self.fixture
        protected = np.r_[fixture.source['query_indices'], fixture.source['check_indices']]
        protected_pixels = np.column_stack((protected % 25, protected // 25))
        center = None
        for y in range(2, 8):
            for x in range(2, 8):
                if np.all(np.max(np.abs(protected_pixels - [x, y]), axis=1) > 1):
                    center = (x, y)
                    break
            if center is not None:
                break
        self.assertIsNotNone(center, 'The fixture needs an unqueried, unchecked actual patch')
        x, y = center
        depth = fixture.depth.copy()
        depth[y - 1:y + 2, x - 1:x + 2] = replacement_depth
        surface = geometry.build_target_surface(views(depth, fixture.K))
        target_observation = observation(depth, fixture.K)
        prepared = [dict(record, observation=target_observation) for record in fixture.field.views]
        field = SurfaceResponseField(surface, fixture.field.queries, prepared)
        domain = rank_domain(fixture.source, surface, field)[4]
        return surface, field, domain

    def test_full_entity_support_uses_actual_context_without_context_queries(self):
        fixture = self.fixture
        self.assertFalse((~fixture.source['query_region']).any())
        evidence, source, target = entity_evidence(
            fixture.source, fixture.surface, self.identity, registration_error_m=0.)
        self.assertEqual(evidence['source_entity_count'], 625)
        self.assertEqual(evidence['source_context_count'], 619)
        self.assertEqual(evidence['context_support_count'], 619)
        self.assertEqual(evidence['distinct_context_correspondence_count'], 619)
        self.assertEqual(evidence['conflict_count'], 0)
        self.assertEqual(evidence['unknown_count'], 0)
        np.testing.assert_allclose(source, target, rtol=0, atol=1e-15)
        # Both cameras observe the same real elements; the matching cannot
        # count 1238 independent context correspondences from 619 source rows.
        ids = np.asarray(evidence['context_observation_indices'])
        elements = fixture.surface['observation_to_element'][ids]
        self.assertEqual(len(np.unique(elements)), len(source))

    def test_surface_capacity_does_not_count_a_contracted_patch_many_times(self):
        fixture = self.fixture
        # This legal Sim(3) contracts the observed sheet onto a small sheet
        # patch. Position support alone is plentiful; target element capacity
        # must still limit independent correspondences to distinct pixels.
        transform = (.2, np.eye(3), np.array([0., 0., .48]))
        evidence, source, target = entity_evidence(
            fixture.source, fixture.surface, transform, registration_error_m=0.)
        self.assertGreater(evidence['context_support_count'], len(source))
        self.assertGreater(len(source), 0)
        self.assertLess(len(source), 619)
        ids = np.asarray(evidence['context_observation_indices'])
        self.assertEqual(len(np.unique(fixture.surface['observation_to_element'][ids])), len(source))
        self.assertEqual(len(target), len(source))

    def test_unqueried_context_conflict_refutes_identity_and_labels_entity(self):
        fixture = self.fixture
        surface, field, domain = self._target_with_unqueried_patch(.63)
        result = evaluate_candidates(fixture.source, surface, field, fixture.proposals, domain)
        first = result['diagnostics']['hypotheses'][0]
        self.assertEqual(first['context_query_conflict_count'], 0)
        self.assertEqual(first['context_check_conflict_count'], 0)
        self.assertGreater(first['entity_structure']['conflict_count'], 0)
        self.assertIn('observed_entity_depth_model_conflict', first['rejections'])
        self.assertEqual(first['entity_model_status'], 'observed_conflict')
        self.assertFalse(first['admitted_to_H'])
        self.assertFalse(result['accepted'])

    def test_missing_unqueried_context_is_unknown_not_negative_evidence(self):
        fixture = self.fixture
        surface, field, domain = self._target_with_unqueried_patch(0.)
        result = evaluate_candidates(fixture.source, surface, field, fixture.proposals, domain)
        first = result['diagnostics']['hypotheses'][0]
        self.assertEqual(first['entity_structure']['conflict_count'], 0)
        self.assertGreater(first['entity_structure']['unknown_count'], 0)
        self.assertEqual(first['entity_model_status'], 'partially_observed')
        self.assertTrue(first['admitted_to_H'], first['rejections'])
        self.assertTrue(result['accepted'], result['reasons'])

    def test_no_actual_context_cannot_be_promoted_to_structural_identity(self):
        fixture = self.fixture
        # All measured source points are deliberately marked functional here;
        # there is genuinely no independent entity context. No source points
        # are deleted or synthesized to manufacture this missing-evidence case.
        raw = observation(fixture.depth, fixture.K)
        source = geometry.prepare_source(raw, fixture.depth,
            query_indices=fixture.source['query_indices'], query_region=np.ones(6, bool))
        domain = rank_domain(source, fixture.surface, fixture.field)[4]
        result = evaluate_candidates(source, fixture.surface, fixture.field, fixture.proposals, domain)
        first = result['diagnostics']['hypotheses'][0]
        self.assertEqual(first['positive_region_matched_count'], 6)
        self.assertEqual(first['entity_structure']['context_support_count'], 0)
        self.assertIn('fewer_than_six_distinct_context_measurements', first['rejections'])
        self.assertFalse(result['region_identified'])
        self.assertFalse(result['accepted'])

    def test_all_candidate_pairs_are_checked_when_reference_agreement_is_nontransitive(self):
        source = dict(self.fixture.source)
        source['entity_position_error_m'] = np.zeros(len(source['entity_xyz']))
        records = []
        # Both extremes agree with the center at a 2 mm pair budget, but they
        # disagree with one another by 3 mm. Reference-only clustering fails.
        for index, translation in enumerate((.0015, 0., .003)):
            records.append(dict(transform=(1., np.eye(3), np.array([translation, 0., 0.])),
                summary=dict(hypothesis_id=index, registration_error_budget_m=.001,
                             score=1. - index * 1e-12)))
        comparison = compare_transforms(source, records)
        self.assertFalse(comparison['agree'])
        self.assertEqual(comparison['compared_pair_count'], 3)
        self.assertEqual([item['hypothesis_ids'] for item in comparison['disagreements']], [[1, 2]])

    def test_unequal_appearance_scores_do_not_hide_supported_pose_disagreement(self):
        fixture = self.fixture
        shifted = dict(transform=(1., np.eye(3), np.array([.0045, 0., 0.])),
                       origins=[dict(origin='translated_measured_sheet')])
        result = evaluate_candidates(fixture.source, fixture.surface, fixture.field,
            fixture.proposals + [shifted], fixture.domain)
        hypotheses = result['diagnostics']['hypotheses']
        first = hypotheses[0]
        other = next(item for item in hypotheses if any(
            origin['origin'] == 'translated_measured_sheet' for origin in item['origins']))
        self.assertTrue(first['admitted_to_H'], first['rejections'])
        self.assertTrue(other['admitted_to_H'], other['rejections'])
        self.assertNotEqual(first['score'], other['score'])
        self.assertEqual(result['status'], 'rejected_ambiguous_structure')
        self.assertFalse(result['accepted'])
        self.assertFalse(result['region_identified'])
        self.assertIsNone(result['sim3_rotation'])

    def test_sampling_anchor_failure_preserves_identity_without_actionable_output(self):
        fixture = self.fixture
        # Only the final sampling-output boundary is injected. All measured
        # geometry, actual appearance, assignments, and structural checks run.
        with patch('transfer.dual_view.curved_inference.geometry.region_for_transform',
                   return_value=(None, 'source_anchor_observation_unavailable')):
            result = evaluate_candidates(fixture.source, fixture.surface, fixture.field,
                                         fixture.proposals, fixture.domain)
        self.assertTrue(result['region_identified'])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'], 'identified_region_sampling_unavailable')
        self.assertIn('source_anchor_observation_unavailable', result['reasons'])
        self.assertGreater(result['diagnostics']['hypotheses_admitted'], 0)
        self.assertIsNone(result['sim3_rotation'])
        self.assertFalse(result['transform_certified'])
        self.assertEqual(result['modes'], [])


if __name__ == '__main__':
    unittest.main()
