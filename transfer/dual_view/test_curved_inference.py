"""CPU end-to-end curved evidence tests with actual depth and response maps.

Only interruption and sampling-unavailability injections replace a stage. The ordinary graph, per-camera
feature footprint, geometric assignment and weighted evidence
score execute on small observations; selection is never mocked into success.
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import curved_geometry as geometry
from .curved_features import SurfaceResponseField
from .curved_inference import evaluate_candidates, infer
from .features import prepare_view
from .test_curved_geometry import observation, views
from ..experiment import store_prediction, restore_prediction
from manipulation.region_transfer import select_reference_transform


class CurvedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((25, 25), .6)
        self.K = np.array([[400., 0., 12.], [0., 400., 12.], [0., 0., 1.]])
        pixels = np.array([[9, 9], [12, 9], [15, 9], [9, 15], [12, 15], [15, 15]])
        region = np.zeros((25, 25), bool)
        region[pixels[:, 1], pixels[:, 0]] = True
        raw = observation(self.depth, self.K, region=region)
        ids = np.flatnonzero(raw['region_mask'])
        self.source = geometry.prepare_source(raw, self.depth,
            query_indices=ids, query_region=np.ones(6, bool))
        self.surface = geometry.build_target_surface(views(self.depth, self.K))
        yy, xx = np.indices((25, 25))
        coordinates = pixels*(840/25)*(24/839)
        feature = np.array([np.exp(-((xx-x)**2+(yy-y)**2)/2)
                            for x, y in coordinates], np.float32)[None]
        queries = np.eye(6, dtype=np.float32)
        prepared = [prepare_view(queries, feature, crop_xyxy=[0, 0, 25, 25],
            observation=raw, view_id=name) for name in ('a', 'b')]
        self.field = SurfaceResponseField(self.surface, queries, prepared)
        # BEGIN the evaluator prepares fixed per-view references from this field.
        self.proposals = [dict(transform=(1., np.eye(3), np.zeros(3)),
                               origins=[dict(origin='measured_identity_fixture')])]
        # END proposals remain geometry-only and independent of response ranks.

    # BEGIN surface selection ranks the new weighted evidence score.
    def test_real_depth_geometry_assignment_accepts_without_normals_or_refits(self):
        result = evaluate_candidates(self.source, self.surface, self.field, self.proposals)
        self.assertTrue(result['accepted'], result['reasons'])
        self.assertTrue(result['region_identified'])
        self.assertFalse(result['transform_certified'])
        self.assertEqual(result['diagnostics']['query_count'], 6)
        self.assertAlmostEqual(result['diagnostics']['source_weight_sum'], 1.)
        first = result['diagnostics']['hypotheses'][0]
        self.assertEqual(first['geometry_region_matched_count'], 6)
        self.assertEqual(len(set(first['regional_assignment_observation_indices'])), 6)
        self.assertTrue(first['correspondence_geometry']['rotation_algebraically_identifiable'])
        self.assertFalse(self.source['query_normal_valid'].any())
        self.assertIsNone(result['modes'][0]['normal'])

    def test_missing_features_do_not_remove_actual_geometry_correspondences(self):
        prepared = [dict(view, response=None) for view in self.field.views]
        field = SurfaceResponseField(self.surface, self.field.queries, prepared)
        result = evaluate_candidates(self.source, self.surface, field, self.proposals)
        first = result['diagnostics']['hypotheses'][0]
        self.assertEqual(first['appearance_known_count'], 0)
        self.assertEqual(first['appearance_scores'], [None]*6)
        self.assertEqual(len(first['regional_assignment_query_rows']), 6)
        self.assertTrue(first['admitted_to_H'], first['rejections'])
        self.assertTrue(result['accepted'], result['reasons'])

    def test_higher_score_selects_same_transform_when_candidate_order_changes(self):
        # The second legal similarity keeps this entire measured sheet at
        # z=.6 m. Its six query positions are supported, but the discrete target
        # relations do not match its 15% scale change. Its 6 mm translation
        # also provides a distinct pose without introducing a selection gate.
        other = dict(transform=(1.15, np.eye(3), np.array([.006, 0., -.09])),
            origins=[dict(origin='supported_but_wrong_relative_geometry')])
        result = evaluate_candidates(self.source, self.surface, self.field,
            self.proposals+[other])
        self.assertEqual(len(result['diagnostics']['hypotheses']), 2)
        first, second = result['diagnostics']['hypotheses']
        self.assertTrue(first['admitted_to_H'], first['rejections'])
        self.assertTrue(second['admitted_to_H'], second['rejections'])
        self.assertTrue(first['selected'])
        self.assertFalse(second['selected'])
        self.assertGreater(first['score'], second['score'])
        self.assertGreater(first['score_components']['geometry_score'],
            second['score_components']['geometry_score'])
        self.assertEqual(result['diagnostics']['ranked_hypothesis_ids'], [0, 1])
        self.assertEqual(result['diagnostics']['top_score_hypothesis_ids'], [0])
        self.assertFalse(result['finite_candidate_unique'])
        self.assertTrue(result['accepted'], result['reasons'])
        reordered = evaluate_candidates(self.source, self.surface, self.field,
            [other]+self.proposals)
        self.assertTrue(reordered['accepted'], reordered['reasons'])
        self.assertEqual(reordered['diagnostics']['selected_hypothesis_id'], 1)
        expected, actual = select_reference_transform(result), select_reference_transform(reordered)
        self.assertEqual(expected['scale'], actual['scale'])
        np.testing.assert_array_equal(expected['rotation'], actual['rotation'])
        np.testing.assert_array_equal(expected['translation'], actual['translation'])

    def test_missing_correspondence_keeps_its_source_weight_in_the_denominator(self):
        # Nine frozen rows leave eight correspondences when a real depth pixel
        # is missing. The score retains that missing ninth rather than taking
        # a mean over only the eight visible rows. No feature is fabricated.
        pixels = np.array([[x, y] for y in (9, 12, 15) for x in (9, 12, 15)])
        region = np.zeros((25, 25), bool)
        region[pixels[:, 1], pixels[:, 0]] = True
        raw = observation(self.depth, self.K, region=region)
        ids = np.flatnonzero(raw['region_mask'])
        source = geometry.prepare_source(raw, self.depth,
            query_indices=ids, query_region=np.ones(9, bool))
        target_depth = self.depth.copy()
        target_depth[9, 9] = 0.
        surface = geometry.build_target_surface(views(target_depth, self.K))
        descriptors = np.eye(9, dtype=np.float32)
        target_observation = observation(target_depth, self.K)
        prepared = [prepare_view(descriptors, None, crop_xyxy=[0, 0, 25, 25],
            observation=target_observation, view_id=name) for name in ('a', 'b')]
        field = SurfaceResponseField(surface, descriptors, prepared)
        result = evaluate_candidates(source, surface, field, self.proposals)
        first = result['diagnostics']['hypotheses'][0]
        self.assertTrue(first['admitted_to_H'], first['rejections'])
        self.assertEqual(len(first['regional_assignment_query_rows']), 8)
        self.assertAlmostEqual(first['score_components']['geometry_score'], 8./9.)
        self.assertAlmostEqual(first['score'], (2./3.)*(8./9.))
        self.assertEqual(first['assigned_appearance_unknown_count'], 9)
        self.assertIn('selected_estimate_has_unknown_appearance',
            result['diagnostics']['quality_warnings'])
        self.assertTrue(first['selected'])
        self.assertEqual(result['status'], 'accepted_observed_region')
        self.assertTrue(result['accepted'], result['reasons'])

    def test_unrelated_candidate_does_not_change_existing_scores_or_source_weights(self):
        single = evaluate_candidates(self.source, self.surface, self.field, self.proposals)
        extra = dict(transform=(1.15, np.eye(3), np.array([.006, 0., -.09])),
            origins=[dict(origin='additional_finite_candidate')])
        expanded = evaluate_candidates(self.source, self.surface, self.field, [extra]+self.proposals)
        before = single['diagnostics']['hypotheses'][0]
        after = expanded['diagnostics']['hypotheses'][1]
        self.assertEqual(before['score_components'], after['score_components'])
        self.assertEqual(single['diagnostics']['appearance_reference'],
            expanded['diagnostics']['appearance_reference'])

    def test_reference_preparation_interruption_publishes_no_candidate(self):
        with patch.object(self.field, 'prepare_support', side_effect=MemoryError('interrupted reference')):
            result = evaluate_candidates(self.source, self.surface, self.field, self.proposals)
        self.assertEqual(result['status'], 'search_incomplete')
        self.assertEqual(result['diagnostics']['search_stage'], 'appearance_reference_preparation')
        self.assertEqual(result['diagnostics']['hypotheses'], [])
        self.assertEqual(result['modes'], [])
        self.assertIsNone(select_reference_transform(result))

    def test_highest_score_missing_region_does_not_promote_available_runner_up(self):
        other = dict(transform=(1.15, np.eye(3), np.array([.006, 0., -.09])),
            origins=[dict(origin='lower_score_available_output')])
        actual_region = geometry.region_for_transform

        def selected_anchor_missing(surface, source, transform, **kwargs):
            if np.array_equal(transform[2], np.zeros(3)):
                return None, 'source_anchor_observation_unavailable'
            return actual_region(surface, source, transform, **kwargs)

        # Only sampling availability is injected. Real depth, assignment and
        # scoring determine the winner, which must remain selected but missing.
        with patch('transfer.dual_view.curved_inference.geometry.region_for_transform',
                   side_effect=selected_anchor_missing):
            result = evaluate_candidates(self.source, self.surface, self.field,
                self.proposals+[other])
        first, second = result['diagnostics']['hypotheses']
        self.assertGreater(first['score'], second['score'])
        self.assertTrue(second['admitted_to_H'])
        self.assertTrue(second['output_available'])
        self.assertEqual(result['diagnostics']['selected_hypothesis_id'], 0)
        self.assertTrue(result['region_identified'])
        self.assertEqual(result['status'], 'identified_region_sampling_unavailable')
        self.assertFalse(result['accepted'])
        self.assertIsNone(select_reference_transform(result))

    def test_explicit_candidates_are_evaluated_once_without_appended_refits(self):
        proposals = [dict(transform=(1., np.eye(3), np.array([shift, 0., 0.])),
            origins=[dict(origin='fixed_slot_'+str(index))])
            for index, shift in enumerate((0., .0015, -.0015))]
        result = evaluate_candidates(self.source, self.surface, self.field, proposals)
        hypotheses = result['diagnostics']['hypotheses']
        self.assertEqual(len(hypotheses), len(proposals))
        self.assertEqual(result['diagnostics']['evaluated_transform_count'], len(proposals))
        self.assertNotIn('refinement_records', result['diagnostics'])
        self.assertEqual([item['origins'] for item in hypotheses],
            [item['origins'] for item in proposals])
    # END score selection and evidence availability are tested with real RGB-D.

    def test_footprint_cache_budget_changes_reuse_but_not_scores(self):
        points = self.source['query_xyz']
        expected = self.field.sample(points, np.full_like(points, np.nan))
        self.field._footprint_cache.clear()
        self.field._footprint_cache_bytes = 0
        self.field._footprint_cache_limit_bytes = 0
        actual = self.field.sample(points, np.full_like(points, np.nan))
        np.testing.assert_array_equal(actual['scores'], expected['scores'])
        self.assertEqual(len(self.field._footprint_cache), 0)

    def test_prediction_roundtrip_preserves_null_normals_and_reference_decision(self):
        result = evaluate_candidates(self.source, self.surface, self.field, self.proposals)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prediction'
            store_prediction(path, result, {})
            restored = restore_prediction(path)
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
                self.assertTrue(all(archive[name].dtype.kind != 'O' for name in archive.files))
        self.assertEqual(restored['status'], result['status'])
        self.assertEqual(restored['region_geometry'], 'surface')
        self.assertIsNone(restored['modes'][0]['normal'])
        expected, actual = select_reference_transform(result), select_reference_transform(restored)
        self.assertEqual(actual['hypothesis_id'], expected['hypothesis_id'])
        np.testing.assert_array_equal(actual['rotation'], expected['rotation'])

    def test_missing_positive_feature_corner_does_not_renormalize(self):
        # Remove one original measurement from a bilinear corner's entity
        # support while retaining the center point's valid depth observation.
        raw_views = views(self.depth, self.K)
        for view in raw_views:
            index = 8*25+8
            view['observation']['entity_member'][index] = False
        surface = geometry.build_target_surface(raw_views)
        field = SurfaceResponseField(surface, self.field.queries, self.field.views)
        result = field.sample(self.source['query_xyz'][:1], np.full((1, 3), np.nan))
        self.assertFalse(result['known'].any())
        self.assertTrue(np.all(result['view_state'] == 'partial_feature_support'))

    def test_interrupted_candidate_stream_cannot_publish_a_partial_winner(self):
        def interrupted():
            yield self.proposals[0]
            raise InterruptedError('fixture interruption after one evaluated proposal')
        result = evaluate_candidates(self.source, self.surface, self.field, interrupted())
        self.assertGreater(len(result['diagnostics']['hypotheses']), 0)
        self.assertEqual(result['status'], 'search_incomplete')
        self.assertFalse(result['accepted'])
        self.assertFalse(result['region_identified'])
        self.assertFalse(result['finite_candidate_unique'])
        self.assertIsNone(result['sim3_rotation'])
        self.assertEqual(result['modes'], [])
        self.assertEqual(len(result['src_anchor_idx']), 0)

    def test_proposal_allocation_failure_is_a_structured_incomplete_search(self):
        with patch('transfer.dual_view.curved_inference.build_proposals',
                   side_effect=MemoryError('fixture bounded allocation failure')):
            result = infer(self.source, self.surface, self.field)
        self.assertEqual(result['status'], 'search_incomplete')
        self.assertEqual(result['diagnostics']['search_stage'], 'proposal_generation')
        self.assertEqual(result['diagnostics']['query_count'], 6)
        self.assertIn('declared_finite_search_interrupted', result['diagnostics']['quality_warnings'])
        self.assertFalse(result['accepted'])


if __name__ == '__main__':
    unittest.main()
