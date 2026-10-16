"""Contracts for cached input identity, frozen evaluation, and local rendering.

Small synthetic observations exercise missing evidence, unequal image sizes,
and serialization. No model, simulation scene, or truth-driven tuning is used.
"""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from . import multiview_experiment as experiment
from .multiview_geometry import build_target_surface
from .multiview_visualization import joint_region_picture


# BEGIN replacement contracts for joint-surface experiment artifacts.
class MultiviewExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix='joint-surface-experiment-')
        self.root = Path(self.temporary.name)
        self.data, self.cache = self.root / 'data', self.root / 'cache'
        self.data.mkdir()
        self.cache.mkdir()
        self.index = self.root / 'cases.json'
        self.calibration = {'K': np.eye(3).tolist(), 'T_world_camera_cv': np.eye(4).tolist()}
        frames = {}
        for name, shape in (('source', (3, 3)), ('left', (3, 3)), ('right', (2, 4))):
            folder = self.data / name
            folder.mkdir()
            np.save(folder / 'depth_m.npy', np.ones(shape))
            (folder / 'calibration.json').write_text(json.dumps(self.calibration))
            Image.new('RGB', shape[::-1], (20, 30, 40)).save(folder / 'rgb.png')
            frames[name] = {'path': name}
        (self.data / 'manifest.json').write_text(json.dumps({'views': frames}))
        y, x = np.indices((3, 3))
        pixels = np.column_stack((x.ravel(), y.ravel())).astype(np.int64)
        region = np.zeros((3, 3), bool)
        region[0, :2] = True
        np.save(self.data / 'demonstration.npy', region)
        self.source = {'xyz': np.c_[pixels, np.ones(len(pixels))], 'pixels': pixels,
            'object_mask': np.ones((3, 3), bool), 'entity_member': np.ones(len(pixels), bool),
            'region_mask': region.ravel(), 'evidence_only': np.zeros(len(pixels), bool),
            'K': np.eye(3), 'T_world_camera_cv': np.eye(4)}
        self.target = {key: (value[:2] if key in ('xyz', 'pixels', 'entity_member', 'region_mask', 'evidence_only') else value.copy())
                       for key, value in self.source.items()}
        self.evidence = {'query_indices': np.array([0, 3]), 'query_region': np.array([True, False]),
            'appearance_cost': np.array([[.1, 1.], [.2, 1.]], dtype=np.float32),
            'peak_indices': np.array([[0, -1], [0, -1]]), 'supported_target': np.array([True, False]),
            'allowed_feature_cells': np.asarray(1)}
        cases = []
        # Reverse manifest order deliberately: all core indices must still use
        # canonical camera identity order after loading.
        for view, frame, prepared in (('B', 'right', False), ('A', 'left', True)):
            identifier = f'one_task_{view}'
            folder = self.root / identifier
            folder.mkdir()
            cached = self.cache / identifier
            cached.mkdir()
            np.savez(folder / 'source_observation.npz', **self.source)
            Image.new('RGB', (3, 3), (20, 30, 40)).save(folder / 'source.png')
            (folder / 'inputs.json').write_text(json.dumps({'source_crop_xyxy': [0, 0, 3, 3]}))
            (folder / 'preparation.json').write_text(json.dumps({'prepared': prepared,
                'source': {'prepared': True, 'reasons': []},
                'target': {'prepared': prepared, 'reasons': [] if prepared else ['no_entity_observation']}}))
            if prepared:
                np.savez(folder / 'target_observation.npz', **self.target)
                np.savez(cached / 'joint_evidence.npz', **self.evidence)
                (cached / 'frozen.json').write_text(json.dumps({'complete': True, 'target_truth_used': False}))
            cases.append({'id': identifier, 'task': 'one_task', 'view': view, 'source': 'source',
                          'target': frame, 'evidence': 'demonstration.npy', 'input_dir': str(folder),
                          'pair_truth': f'pair-truth/{identifier}.json'})
        self.description = {'data': str(self.data), 'cases': cases}
        self.index.write_text(json.dumps(self.description))

    def tearDown(self):
        self.temporary.cleanup()

    def task(self):
        return experiment.read_manifest(self.index)[1]['tasks'][0]

    def loaded(self):
        return experiment.load_inputs(self.index, self.task(), self.cache)

    def test_manifest_only_groups_explicit_records_without_truth(self):
        with mock.patch.object(experiment, 'read_archive', side_effect=AssertionError('array read forbidden')):
            _, description = experiment.read_manifest(self.index)
        self.assertEqual([task['id'] for task in description['tasks']], ['one_task'])

    def test_duplicate_view_identity_is_rejected(self):
        self.description['cases'][1]['view'] = 'B'
        self.index.write_text(json.dumps(self.description))
        with self.assertRaisesRegex(ValueError, 'unique view'):
            experiment.read_manifest(self.index)

    def test_missing_entity_keeps_full_depth_and_canonical_camera_order(self):
        loaded = self.loaded()
        self.assertEqual([view['id'] for view in loaded['views']], ['A', 'B'])
        missing = loaded['views'][1]
        self.assertIsNone(missing['observation'])
        self.assertIsNone(missing['evidence'])
        self.assertEqual(missing['depth_m'].shape, (2, 4))
        self.assertEqual(missing['missing_reasons'], ['no_entity_observation'])
        np.testing.assert_array_equal(loaded['query_indices'], [0, 3])
        self.assertEqual(float(loaded['views'][0]['evidence']['appearance_cost'][0, 1]), 1.)
        self.assertFalse(loaded['views'][0]['evidence']['supported_target'][1])

    def test_stale_depth_is_rejected_without_snapping(self):
        np.save(self.data / 'left' / 'depth_m.npy', np.full((3, 3), 2.))
        with self.assertRaises(AssertionError):
            self.loaded()

    def test_changed_source_crop_is_rejected(self):
        folder = Path(self.description['cases'][0]['input_dir'])
        (folder / 'inputs.json').write_text(json.dumps({'source_crop_xyxy': [0, 0, 2, 3]}))
        with self.assertRaisesRegex(ValueError, 'identical source'):
            self.loaded()

    def test_surface_roundtrip_keeps_provenance_and_unequal_image_shapes(self):
        loaded = self.loaded()
        original = build_target_surface(loaded['views'])
        experiment.save_surface(self.root, original)
        restored = experiment.load_surface(self.root, loaded['views'])
        for key, value in original.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(restored[key], value, err_msg=key)
            elif key not in ('adjacency', 'pixel_to_observation', 'depth_m'):
                self.assertEqual(restored[key], value)
        self.assertEqual((restored['adjacency'] != original['adjacency']).nnz, 0)
        self.assertEqual([depth.shape for depth in restored['depth_m']], [(3, 3), (2, 4)])
        for actual, expected in zip(restored['pixel_to_observation'], original['pixel_to_observation']):
            np.testing.assert_array_equal(actual, expected)

    def test_native_proposal_does_not_require_old_region_acceptance(self):
        loaded = self.loaded()
        np.savez(self.cache / 'one_task_A' / 'old_teaserpp.npz', accepted=np.asarray(False),
                 sim3_scale=np.asarray(1.), sim3_rotation=np.eye(3), sim3_translation=np.zeros(3))
        proposals, states = experiment.initial_proposals(loaded['views'], 'teaserpp')
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]['view_index'], 0)
        self.assertEqual(proposals[0]['origin'], 'native_teaserpp')
        self.assertTrue(states[0]['available'])
        self.assertFalse(states[1]['available'])

    def test_evaluation_requires_frozen_matrix_before_truth_reads(self):
        with mock.patch.object(experiment, 'read_archive', side_effect=AssertionError('truth read before barrier')):
            with self.assertRaisesRegex(FileNotFoundError, 'Finish every declared prediction'):
                experiment.evaluate_suite(self.root / 'unstarted')

    def test_prediction_process_guard_blocks_truth(self):
        script = ('from transfer.multiview_experiment import block_evaluation_reads; '
                  'block_evaluation_reads(); open("pair-truth/held-out.json")')
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Evaluation-only artifact requested', result.stderr)

    def test_occluded_sample_is_not_painted_in_another_camera(self):
        loaded = self.loaded()
        surface = build_target_surface(loaded['views'])
        seed = int(np.flatnonzero(surface['entity_member'])[0])
        mask = np.zeros(len(surface['xyz']), bool)
        mask[seed] = True
        mode = {'mask': mask, 'sample_indices': np.array([seed]), 'center': surface['xyz'][seed]}
        hidden = {'supported': np.array([False]), 'conflict': np.array([False]),
                  'unknown': np.array([True]), 'pixels': np.array([[1., 1.]])}
        with mock.patch('transfer.multiview_visualization.project_observations', return_value=hidden):
            image, counts = joint_region_picture(np.zeros((2, 4, 3), np.uint8), surface, mode, 1)
        self.assertEqual(counts['visible_samples'], 0)
        self.assertEqual(counts['unknown_samples'], 1)
        self.assertFalse(np.any(np.all(np.asarray(image) == [255, 237, 100], axis=2)))


if __name__ == '__main__':
    unittest.main()
# END replacement experiment contracts.
