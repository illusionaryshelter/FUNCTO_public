"""TAPIP3D's CPU boundary is checked without loading a network or using CUDA."""

# BEGIN calibrated material-point worker contract tests.
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from local_models import runtime, tapip3d


class TAPIPBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.shape = (60, 80)
        self.K = np.array([[90., 0, 40.], [0, 95., 30.], [0, 0, 1.]])
        self.masks = np.zeros((2, *self.shape), dtype=bool)
        self.masks[0, 5:53, 5:36] = True
        self.masks[1, 5:53, 44:76] = True
        self.frames, self.depths = [], []
        for frame in range(3):
            rgb, depth = self.root/f'rgb-{frame}.png', self.root/f'depth-{frame}.png'
            Image.new('RGB', (80, 60), (50, 90, 120)).save(rgb)
            Image.fromarray(np.full(self.shape, 1000, dtype=np.uint16)).save(depth)
            self.frames.append(rgb)
            self.depths.append(depth)

    def predicted(self, kind, paths, options):
        self.assertEqual(kind, 'tapip3d')
        self.assertEqual(len(paths), 3)
        self.assertNotIn('models', options)
        self.assertNotIn('masks', options)
        q = np.asarray(options['queries'], dtype=np.float32)
        E = np.asarray(options['extrinsics'])
        coords = np.repeat(q[None, :, 1:], len(paths), axis=0)
        coords[:, :, 0] += np.arange(len(paths))[:, None]*.002
        camera = np.einsum('tij,tnj->tni', E[:, :3, :3], coords) + E[:, None, :3, 3]
        uvz = camera@self.K.T
        return {'coords':coords, 'visibs':np.ones(coords.shape[:2], bool),
                'query_points':q, 'pixels':uvz[..., :2]/uvz[..., 2:]}

    def test_one_worker_keeps_unique_mask_samples_and_world_query_identity(self):
        E0 = np.eye(4)
        E0[:3, :3] = Rotation.from_euler('xyz', [.2, -.1, .3]).as_matrix()
        E0[:3, 3] = [.1, -.2, .05]
        E = np.repeat(E0[None], 3, axis=0)
        with patch.object(runtime, 'run_file_worker', side_effect=self.predicted) as worker:
            result = tapip3d.track(self.frames, self.depths, K=self.K, depth_scale=.001,
                                  masks=self.masks, extrinsics=E)
        worker.assert_called_once()
        self.assertEqual(result['coords'].shape, (3, 128, 3))
        np.testing.assert_array_equal(np.bincount(result['query_owner']), [64, 64])
        for owner in (0, 1):
            uv = result['query_pixels'][result['query_owner']==owner].astype(int)
            self.assertEqual(len(np.unique(uv, axis=0)), 64)
            self.assertTrue(self.masks[owner, uv[:, 1], uv[:, 0]].all())
        camera = result['query_points'][:, 1:]@E0[:3, :3].T+E0[:3, 3]
        np.testing.assert_allclose(camera[:, 2], 1., atol=1e-7)
        projection = camera@self.K.T
        np.testing.assert_allclose(projection[:, :2]/projection[:, 2:], result['query_pixels'], atol=1e-5)
        self.assertEqual(result['visibs'].dtype, np.bool_)
        self.assertTrue(result['projection_valid'].all())
        self.assertNotIn('confidence', result)

    def test_static_camera_must_be_explicit_and_cannot_replace_external_calibration(self):
        with patch.object(runtime, 'run_file_worker') as worker:
            with self.assertRaisesRegex(ValueError, 'declare stationary_camera'):
                tapip3d.track(self.frames, self.depths, K=self.K, depth_scale=.001, masks=self.masks)
            for flag in ('true', 1, None):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'boolean'):
                    tapip3d.camera_extrinsics(None, 3, stationary_camera=flag)
            with self.assertRaisesRegex(ValueError, 'not both'):
                tapip3d.camera_extrinsics(np.tile(np.eye(4), (3, 1, 1)), 3, stationary_camera=True)
            bad = np.tile(np.eye(4), (3, 1, 1))
            bad[1, 0, 0] = 2
            with self.assertRaisesRegex(ValueError, 'rigid SE'):
                tapip3d.camera_extrinsics(bad, 3)
            worker.assert_not_called()

    def test_masks_invalid_depth_and_missing_files_fail_without_a_model(self):
        kwargs = dict(K=self.K, depth_scale=.001, masks=self.masks, stationary_camera=True)
        with patch.object(runtime, 'run_file_worker') as worker:
            for change in (dict(depth_scale=True), dict(depth_scale=0),
                           dict(masks=self.masks.astype(np.uint8)),
                           dict(masks=np.stack([self.masks[0], self.masks[0]]))):
                with self.subTest(change=list(change)), self.assertRaises(ValueError):
                    tapip3d.track(self.frames, self.depths, **dict(kwargs, **change))
            with self.assertRaises(FileNotFoundError):
                tapip3d.track([self.root/'absent.png'], self.depths[:1], **kwargs)
            Image.fromarray(np.zeros(self.shape, dtype=np.uint16)).save(self.depths[0])
            with self.assertRaisesRegex(ValueError, 'valid measured depth'):
                tapip3d.track(self.frames, self.depths, **kwargs)
            worker.assert_not_called()

    def test_output_identity_visibility_and_original_projection_are_checked(self):
        q, _, _ = tapip3d.sample_queries(np.ones(self.shape), self.K, self.masks, np.eye(4))
        E = np.tile(np.eye(4), (3, 1, 1))
        good = self.predicted('tapip3d', self.frames, {'queries':q, 'extrinsics':E})
        failures = [dict(good, query_points=q[::-1]),
                    dict(good, visibs=good['visibs'].astype(float)),
                    dict(good, coords=good['coords'][:-1]),
                    dict(good, coords=np.full_like(good['coords'], np.nan)),
                    dict(good, pixels=good['pixels']+1)]
        for output in failures:
            with self.subTest(fields=list(output)), self.assertRaises(ValueError):
                tapip3d._outputs(output, 3, q, self.K, E, self.shape)

    def test_file_transport_stays_offline_and_returns_only_point_arrays(self):
        def run(command, **kwargs):
            request = json.loads(Path(command[-1]).read_text())
            self.assertEqual(request['kind'], 'tapip3d')
            self.assertEqual(kwargs['env']['PYTHONNOUSERSITE'], '1')
            self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
            np.savez_compressed(request['output'], coords=np.zeros((3, 2, 3)))
        with patch.object(runtime.subprocess, 'run', side_effect=run) as worker:
            result = runtime.run_file_worker('tapip3d', self.frames, {'queries':[[0,0,0,1]]})
        worker.assert_called_once()
        self.assertEqual(result['coords'].shape, (3, 2, 3))

    def test_worker_uses_existing_rgbd_tracker_without_external_depth_models(self):
        q, _, _ = tapip3d.sample_queries(np.ones(self.shape), self.K, self.masks, np.eye(4))
        E = np.tile(np.eye(4), (3, 1, 1))
        captured = {}
        class FakeTracker:
            def __init__(self, checkpoint):
                captured['checkpoint'] = checkpoint
            def track(self, rgb, depths, K, queries, **options):
                captured.update(rgb=rgb.copy(), depths=depths.copy(), options=options)
                xyz = np.repeat(queries[None, :, 1:], 3, axis=0)
                uvz = xyz@K.T
                return {'coords':xyz, 'visibs':np.ones(xyz.shape[:2],bool),
                        'query_points':queries, 'pixels':uvz[..., :2]/uvz[..., 2:]}
        output = self.root/'result.npz'
        # Restoring sys.path prevents this isolated-boundary test from changing
        # module resolution for later FUNCTO tests in the same interpreter.
        with patch.dict(sys.modules, {'rgbd_tracker':SimpleNamespace(RGBDTracker=FakeTracker)}), \
             patch.object(sys, 'path', list(sys.path)), patch.object(Path, 'is_file', return_value=True):
            tapip3d.worker_main(self.frames, output, depth_paths=self.depths, K=self.K,
                depth_scale=.001, queries=q, extrinsics=E, shape=self.shape)
        np.testing.assert_array_equal(captured['depths'], 1.)
        self.assertEqual(captured['rgb'].shape, (3, 60, 80, 3))
        self.assertEqual(captured['options']['image_size'], (384,512))
        self.assertEqual(captured['options']['num_iters'], 6)
        with np.load(output, allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), {'coords','visibs','query_points','pixels'})


if __name__ == '__main__':
    unittest.main()
# END calibrated material-point worker contract tests.
