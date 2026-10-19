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
        # BEGIN raw candidate appearance is independent of a full-domain rank.
        return surface, field
        # END only the actual target measurements and response field are needed.

    # BEGIN entity counts describe measured source rows, not correspondence capacity.
    def test_full_entity_counts_rows_once_without_context_queries(self):
        fixture = self.fixture
        self.assertFalse((~fixture.source['query_region']).any())
        evidence = entity_evidence(
            fixture.source, fixture.surface, self.identity, registration_error_m=0.)
        self.assertEqual(evidence['source_entity_count'], 625)
        self.assertEqual(evidence['source_context_count'], 619)
        self.assertEqual(evidence['context_support_count'], 619)
        self.assertEqual(evidence['support_count'], 625)
        self.assertEqual(evidence['conflict_count'], 0)
        self.assertEqual(evidence['unknown_count'], 0)
        self.assertEqual(evidence['contradicted_source_indices'], [])
        self.assertEqual(evidence['support_count_unit'],
            'source_measurement_rows_not_independent_correspondences')
        # Identical cameras do not duplicate a source measurement count.
        # Independent query capacity is tested in test_curved_multiscale.
    # END no second matching contract remains in the entity refutation module.

    def test_unqueried_context_conflict_refutes_identity_and_labels_entity(self):
        fixture = self.fixture
        surface, field = self._target_with_unqueried_patch(.63)
        result = evaluate_candidates(fixture.source, surface, field, fixture.proposals)
        first = result['diagnostics']['hypotheses'][0]
        self.assertEqual(first['context_query_conflict_count'], 0)
        self.assertEqual(first['context_check_conflict_count'], 0)
        self.assertGreater(first['entity_structure']['conflict_count'], 0)
        contradicted = first['entity_structure']['contradicted_source_indices']
        self.assertEqual(len(contradicted), first['entity_structure']['conflict_count'])
        self.assertTrue(np.isin(contradicted, fixture.source['entity_indices']).all())
        self.assertIn('observed_entity_depth_model_conflict', first['rejections'])
        self.assertEqual(first['entity_model_status'], 'observed_conflict')
        self.assertFalse(first['admitted_to_H'])
        self.assertFalse(result['accepted'])

    def test_missing_unqueried_context_is_unknown_not_negative_evidence(self):
        fixture = self.fixture
        surface, field = self._target_with_unqueried_patch(0.)
        result = evaluate_candidates(fixture.source, surface, field, fixture.proposals)
        first = result['diagnostics']['hypotheses'][0]
        self.assertEqual(first['entity_structure']['conflict_count'], 0)
        self.assertGreater(first['entity_structure']['unknown_count'], 0)
        self.assertEqual(first['entity_model_status'], 'partially_observed')
        self.assertTrue(first['admitted_to_H'], first['rejections'])
        self.assertTrue(result['accepted'], result['reasons'])

    # BEGIN a whole-entity demonstration has overlapping, valid structural scopes.
    def test_whole_entity_region_does_not_require_nonexistent_external_context(self):
        fixture = self.fixture
        # All measured source points are deliberately marked functional here;
        # there is genuinely no outside-region entity context. This is a valid
        # transfer input; the three source support domains coincide rather
        # than inventing an additional six-point context requirement.
        raw = observation(fixture.depth, fixture.K)
        source = geometry.prepare_source(raw, fixture.depth,
            query_indices=fixture.source['query_indices'], query_region=np.ones(6, bool))
        result = evaluate_candidates(source, fixture.surface, fixture.field, fixture.proposals)
        first = result['diagnostics']['hypotheses'][0]
        self.assertEqual(first['geometry_region_matched_count'], 6)
        self.assertEqual(first['entity_structure']['context_support_count'], 0)
        self.assertEqual(first['multiscale_structure']['unique_support_domain_count'], 1)
        self.assertTrue(first['admitted_to_H'], first['rejections'])
        self.assertTrue(result['region_identified'])
        self.assertTrue(result['accepted'], result['reasons'])
    # END absent external context is not a known structural contradiction.

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

    # BEGIN score-interval ambiguity remains independent of appearance ranking.
    def test_unequal_appearance_scores_do_not_hide_supported_pose_disagreement(self):
        fixture = self.fixture
        shifted = dict(transform=(1., np.eye(3), np.array([.0045, 0., 0.])),
                       origins=[dict(origin='translated_measured_sheet')])
        result = evaluate_candidates(fixture.source, fixture.surface, fixture.field,
            fixture.proposals + [shifted])
        hypotheses = result['diagnostics']['hypotheses']
        first = hypotheses[0]
        other = next(item for item in hypotheses if any(
            origin['origin'] == 'translated_measured_sheet' for origin in item['origins']))
        self.assertTrue(first['admitted_to_H'], first['rejections'])
        self.assertTrue(other['admitted_to_H'], other['rejections'])
        self.assertNotEqual(first['appearance_scores'], other['appearance_scores'])
        self.assertAlmostEqual(first['score'], other['score'])
        self.assertEqual(result['status'], 'rejected_ambiguous_structure')
        self.assertFalse(result['accepted'])
        self.assertFalse(result['region_identified'])
        self.assertIsNone(result['sim3_rotation'])
    # END identical measured relations at repeated positions remain ambiguous.

    def test_sampling_anchor_failure_preserves_identity_without_actionable_output(self):
        fixture = self.fixture
        # Only the final sampling-output boundary is injected. All measured
        # geometry, actual appearance, assignments, and structural checks run.
        with patch('transfer.dual_view.curved_inference.geometry.region_for_transform',
                   return_value=(None, 'source_anchor_observation_unavailable')):
            result = evaluate_candidates(fixture.source, fixture.surface, fixture.field,
                                         fixture.proposals)
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
