"""CPU end-to-end curved evidence tests with actual depth and response maps.

Only interruption injection replaces a stage. The ordinary graph, per-camera
feature footprint, fixed-domain calibration, assignment and refits execute on
small deterministic observations so failures cannot hide behind mocked scores.
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
from .proposals import rank_domain
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
        self.domain = rank_domain(self.source, self.surface, self.field)[4]
        self.proposals = [dict(transform=(1., np.eye(3), np.zeros(3)),
                               origins=[dict(origin='measured_identity_fixture')])]

    def test_real_depth_feature_assignment_and_refits_accept_without_normals(self):
        result = evaluate_candidates(self.source, self.surface, self.field, self.proposals, self.domain)
        self.assertTrue(result['accepted'], result['reasons'])
        self.assertTrue(result['region_identified'])
        self.assertFalse(result['transform_certified'])
        self.assertEqual(result['diagnostics']['fixed_query_denominator'], 6)
        first = result['diagnostics']['hypotheses'][0]
        self.assertEqual(first['positive_region_matched_count'], 6)
        self.assertEqual(len(set(first['regional_assignment_observation_indices'])), 6)
        self.assertTrue(first['correspondence_geometry']['rotation_algebraically_identifiable'])
        self.assertFalse(self.source['query_normal_valid'].any())
        self.assertIsNone(result['modes'][0]['normal'])

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
        result = evaluate_candidates(self.source, self.surface, self.field, self.proposals, self.domain)
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

    def test_reference_ranking_retains_owner_measurement_error_after_fusion(self):
        declared = views(self.depth, self.K)
        declared[1]['depth_m'] = self.depth+.00015
        declared[1]['observation'] = observation(declared[1]['depth_m'], self.K)
        surface = geometry.build_target_surface(declared)
        prepared = [dict(view) for view in self.field.views]
        prepared[0]['response'] = None
        field = SurfaceResponseField(surface, self.field.queries, prepared)
        domain = rank_domain(self.source, surface, field)[4]
        self.assertEqual(surface['diagnostics']['paired_element_count'], 625)
        center = surface['observation_to_element'][surface['pixel_to_observation'][0][12, 12]]
        position = int(np.flatnonzero(domain['surface_ids'] == center)[0])
        self.assertTrue(np.all(domain['inverse'][:, position] >= 0))

    def test_interrupted_candidate_stream_cannot_publish_a_partial_winner(self):
        def interrupted():
            yield self.proposals[0]
            raise InterruptedError('fixture interruption after one evaluated proposal')
        result = evaluate_candidates(self.source, self.surface, self.field, interrupted(), self.domain)
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
            result = infer(self.source, self.surface, self.field, prepared_domain=self.domain)
        self.assertEqual(result['status'], 'search_incomplete')
        self.assertEqual(result['diagnostics']['search_stage'], 'proposal_generation')
        self.assertEqual(result['diagnostics']['fixed_query_denominator'], 6)
        self.assertIn('declared_finite_search_interrupted', result['diagnostics']['quality_warnings'])
        self.assertFalse(result['accepted'])


if __name__ == '__main__':
    unittest.main()
