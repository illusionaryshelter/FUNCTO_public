"""CPU orchestration checks; numerical kernels have separate CUDA validation.

Only stage boundaries are replaced. Real crop chart conversion and native
proposal packaging run so missing views, selected registration backend, public
diagnostics and failure cleanup remain observable without initializing CUDA.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import io, pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        observation = dict(xyz=np.array([[0., 0., 1.]]), pixels=np.array([[0, 0]]),
                           object_mask=np.ones((2, 2), bool))
        source = dict(observation, region_mask=np.array([True]))
        self.loaded = dict(source=source, source_depth=np.ones((2, 2)),
            source_crop=np.array([0, 0, 2, 2]), query_indices=np.array([0]),
            query_region=np.array([True]), views=[
                dict(id='A', observation=observation, crop_xyxy=np.array([0, 0, 2, 2]),
                     missing_reasons=[]),
                dict(id='B', observation=None, crop_xyxy=None,
                     missing_reasons=['missing_entity_observation'])])
        self.features = {'A': (object(), object())}
        self.surface, self.source = {'fixture': 'target'}, {'reasons': []}
        self.appearance, self.domain = object(), {'fixture': 'fixed-domain'}
        self.native_pairs = dict(src_anchor_idx=np.array([0]), tgt_anchor_idx=np.array([0]),
                                 scores=np.array([.8]))
        self.result = dict(accepted=False, status='insufficient_appearance',
            reasons=['no_positive_observed_support'], modes=[],
            diagnostics=dict(quality_warnings=['finite_search_not_global', 'observed_fixture_warning']))
        anchors = SimpleNamespace(reasons=[], diagnostics={'fixture': 'observed-anchors'},
            scale=1., rotation=np.eye(3), translation=np.zeros(3))
        # Each numerical stage is replaced once, at its ordinary public call.
        # The pipeline's data routing and failure paths execute unchanged.
        patches = [
            ('surface_call', patch.object(pipeline.surface, 'build_target_surface', return_value=self.surface)),
            ('source_call', patch.object(pipeline, 'prepare_source', return_value=self.source)),
            ('pairs_call', patch.object(pipeline.pairs, 'native_region_pairs', return_value=self.native_pairs)),
            ('appearance_call', patch.object(pipeline, 'appearance_field', return_value=self.appearance)),
            ('ranking_call', patch.object(pipeline, 'rank_domain', return_value=(None, None, None, None, self.domain))),
            ('registration_call', patch.object(io, 'prepare_anchors', return_value=anchors)),
            ('infer_call', patch.object(pipeline, 'infer', return_value=self.result)),
            ('threads', patch.object(pipeline.torch, 'set_num_threads')),
            ('synchronize', patch.object(pipeline.torch.cuda, 'synchronize')),
        ]
        for name, boundary in patches:
            setattr(self, name, boundary.start())
            self.addCleanup(boundary.stop)

    def test_actual_assets_and_missing_view_diagnostics_survive_orchestration(self):
        returned = pipeline.run(self.loaded, self.features)
        self.assertIs(returned['surface'], self.surface)
        self.assertIs(returned['source'], self.source)
        self.assertIs(returned['appearance'], self.appearance)
        self.assertIs(returned['domain'], self.domain)
        self.assertIs(returned['native_pairs']['A'], self.native_pairs)
        self.assertEqual(list(returned['native_pairs']), ['A'])
        self.pairs_call.assert_called_once()
        self.assertIs(self.pairs_call.call_args.args[3], self.features['A'])
        self.assertIs(self.infer_call.call_args.args[0], self.source)
        self.assertIs(self.infer_call.call_args.args[1], self.surface)
        self.assertIs(self.infer_call.call_args.args[2], self.appearance)
        self.assertIs(self.infer_call.call_args.kwargs['prepared_domain'], self.domain)
        self.assertIs(self.infer_call.call_args.kwargs['initial_transforms'], returned['native_transforms'])

        # Backend selection is checked at the real anchor-preparation boundary,
        # not merely by inspecting a backend label added to the final report.
        self.registration_call.assert_called_once()
        self.assertEqual(self.registration_call.call_args.kwargs['registration_algorithm'], 'teaserpp')
        self.assertIs(self.registration_call.call_args.kwargs['correspondence_scores'], self.native_pairs['scores'])
        native = returned['native_transforms']
        self.assertEqual(len(native), 1)
        self.assertEqual(native[0]['origin'], 'native_teaserpp')
        self.assertEqual(native[0]['view_index'], 0)
        result = returned['result']
        self.assertIs(result, self.result)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'], 'insufficient_appearance')
        self.assertEqual(result['reasons'], ['no_positive_observed_support'])
        self.assertEqual(result['diagnostics']['quality_warnings'],
                         ['finite_search_not_global', 'observed_fixture_warning'])
        self.assertEqual(result['diagnostics']['input_missing_views'], ['B'])
        self.assertFalse(result['diagnostics']['native_proposals'][1]['available'])
        self.assertEqual(result['diagnostics']['native_proposals'][1]['reasons'],
                         ['missing_entity_observation'])
        self.assertFalse(result['diagnostics']['target_truth_used'])
        self.assertEqual(self.threads.call_args.args, (1,))

    def test_pair_failure_restores_single_thread_and_does_not_continue(self):
        self.pairs_call.side_effect = RuntimeError('native pair stage failed')
        with self.assertRaisesRegex(RuntimeError, 'native pair stage failed'):
            pipeline.run(self.loaded, self.features)
        self.assertEqual(self.threads.call_args.args, (1,))
        self.appearance_call.assert_not_called()
        self.registration_call.assert_not_called()
        self.infer_call.assert_not_called()

    def test_insufficient_source_preserves_reasons_before_matching(self):
        self.source['reasons'] = ['source_region_not_approximately_planar']
        with self.assertRaises(pipeline.InsufficientSource) as raised:
            pipeline.run(self.loaded, self.features)
        self.assertEqual(raised.exception.reasons, self.source['reasons'])
        self.pairs_call.assert_not_called()
        self.appearance_call.assert_not_called()
        self.infer_call.assert_not_called()


class RuntimeContractTests(unittest.TestCase):
    def test_unavailable_cuda_raises_without_selecting_another_backend(self):
        with patch.object(pipeline.torch.cuda, 'is_available', return_value=False), \
             patch.object(pipeline.surface, 'prepare') as surface_prepare:
            with self.assertRaisesRegex(RuntimeError, 'requires CUDA'):
                pipeline.prepare()
        surface_prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
