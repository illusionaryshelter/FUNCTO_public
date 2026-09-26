"""CPU contracts for bounded native CoTracker scheduling and worker transport."""

# BEGIN local CoTracker regression tests. No checkpoint, API or CUDA is used.
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from local_models import cotracker


class FakePredictor:
    """Model the native cumulative output contract while recording actual input."""

    step = 8

    def __init__(self):
        self.windows = []
        self.queries = None
        self.initial_shape = None
        self.passes = []
        self.query_history = []
        self.image_sizes = []
        self.cache_resets = 0
        self.model = SimpleNamespace(init_video_online_processing=self._reset_cache)

    def _reset_cache(self):
        self.cache_resets += 1

    def __call__(self, video, *, is_first_step, queries=None, **options):
        assert options == {"grid_size": 0, "add_support_grid": True, "one_frame": False}
        assert video.device.type == "cpu"
        assert video.dtype == torch.float32
        assert video.shape[0] == 1 and video.shape[2] == 3
        assert min(video.shape[-2:]) >= 2
        # The green/blue channels test that values are neither divided by 255
        # nor resized before the native predictor sees the original pixels.
        assert torch.all(video[:, :, 1] == 128)
        assert torch.all(video[:, :, 2] == 255)
        frame_ids = video[0, :, 0, 0, 0].to(torch.int64).tolist()
        if is_first_step:
            assert frame_ids == [0]
            self.windows = []
            self.passes.append(self.windows)
            self.initial_shape = tuple(video.shape)
            self.queries = queries.clone()
            self.query_history.append(queries.clone())
            self.image_sizes.append(tuple(video.shape[-2:]))
            return None, None
        self.windows.append(frame_ids)
        count = frame_ids[-1] + 1
        query_count = self.queries.shape[1]
        coords = self.queries[..., 1:].repeat(1, count, 1, 1)
        # Revise the overlap on each call. The adapter must keep the latest
        # cumulative output, rather than concatenate overlapping windows.
        coords[..., 0] += torch.arange(count)[None, :, None] * 0.125
        coords[..., 1] += len(self.windows) * 0.01
        visibility = torch.ones((1, count, query_count), dtype=torch.bool)
        visibility[:, 0] = False  # The adapter restores supplied frame-zero data.
        return coords, visibility


class CoTrackerContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = []
        for frame in range(54):
            path = self.root / f"{frame:05d}.png"
            rgb = np.empty((8, 12, 3), dtype=np.uint8)
            rgb[:] = (frame, 128, 255)
            Image.fromarray(rgb).save(path)
            self.paths.append(path)
        self.queries = np.array([[0., 0.], [5.25, 3.5], [11., 7.]], dtype=np.float32)

    def test_native_short_exact_and_tail_windows_preserve_pixels_and_identity(self):
        expected_starts = {2: [0], 16: [0], 17: [0, 8], 24: [0, 8],
                           54: [0, 8, 16, 24, 32, 40]}
        for frames, starts in expected_starts.items():
            with self.subTest(frames=frames):
                predictor = FakePredictor()
                result = cotracker._run_native(self.paths[:frames], self.queries,
                                               predictor, device="cpu")
                self.assertEqual(predictor.windows,
                                 [list(range(start, min(start + 16, frames))) for start in starts])
                np.testing.assert_array_equal(predictor.queries[0, :, 0], 0)
                np.testing.assert_array_equal(predictor.queries[0, :, 1:], self.queries)
                self.assertEqual(result["pixels"].shape, (frames, 3, 2))
                self.assertEqual(result["visible"].shape, (frames, 3))
                self.assertEqual(result["pixels"].dtype, np.float32)
                self.assertEqual(result["visible"].dtype, np.bool_)
                self.assertEqual(predictor.image_sizes, [(8, 12)])
                np.testing.assert_array_equal(result["crop_xyxy"], [0, 0, 12, 8])
                np.testing.assert_array_equal(result["pixels"][0], self.queries)
                self.assertTrue(result["visible"][0].all())
                np.testing.assert_allclose(result["pixels"][1:, :, 1],
                                           np.broadcast_to(self.queries[:, 1] + len(starts) * 0.01,
                                                           (frames - 1, 3)))
                expected_x = self.queries[:, 0] + (frames - 1) * 0.125
                np.testing.assert_array_equal(result["pixels"][-1, :, 0], expected_x)

    def test_public_boundary_returns_cpu_arrays_and_exact_request(self):
        arrays = {"pixels": np.tile(self.queries, (2, 1, 1)),
                  "visible": np.ones((2, 3), dtype=bool),
                  "crop_xyxy": np.array([0, 0, 12, 8])}
        with patch("local_models.runtime.run_file_worker", return_value=arrays) as worker:
            result = cotracker.track(self.paths[:2], self.queries)
        worker.assert_called_once_with(kind="cotracker", paths=self.paths[:2],
                                       options={"queries_uv": self.queries.tolist()})
        self.assertIs(result["pixels"], arrays["pixels"])
        self.assertIs(result["visible"], arrays["visible"])
        np.testing.assert_array_equal(result["crop_xyxy"], arrays["crop_xyxy"])

    def test_invalid_queries_fail_before_launching_a_worker(self):
        cases = [[], [[np.nan, 1]], [[np.inf, 1]], [[-0.1, 1]], [[12, 1]],
                 [[1, 8]], [1, 2], np.zeros((513, 2)), np.zeros((2, 3))]
        with patch("local_models.runtime.run_file_worker") as worker:
            for queries in cases:
                with self.subTest(queries=np.asarray(queries).shape), self.assertRaises(ValueError):
                    cotracker.track(self.paths[:2], queries)
            worker.assert_not_called()

    def test_query_limit_is_inclusive(self):
        queries = np.tile([[2.5, 3.5]], (512, 1))
        _, validated, _ = cotracker._validate_inputs(self.paths[:2], queries)
        self.assertEqual(validated.shape, (512, 2))

    def test_missing_or_too_short_sequence_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            cotracker._validate_inputs(self.paths[:1], self.queries)
        with self.assertRaisesRegex(FileNotFoundError, "Missing local tracking frames"):
            cotracker._validate_inputs([self.paths[0], self.root / "missing.png"], self.queries)

    def test_inconsistent_frame_dimensions_are_not_silently_rescaled(self):
        Image.new("RGB", (13, 8)).save(self.paths[1])
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            cotracker._run_native(self.paths[:2], self.queries, FakePredictor(), device="cpu")

    def test_cumulative_prediction_length_must_match_each_window(self):
        class TruncatedPredictor(FakePredictor):
            def __call__(self, *args, **kwargs):
                pixels, visible = super().__call__(*args, **kwargs)
                if pixels is not None:
                    return pixels[:, :-1], visible[:, :-1]
                return pixels, visible
        with self.assertRaisesRegex(ValueError, "incomplete window"):
            cotracker._run_native(self.paths[:17], self.queries, TruncatedPredictor(), device="cpu")

    def test_invalid_worker_shapes_visibility_and_numbers_are_rejected(self):
        valid = {"pixels": np.tile(self.queries, (2, 1, 1)),
                 "visible": np.ones((2, 3), dtype=bool),
                 "crop_xyxy": np.array([0, 0, 12, 8])}
        bad = [dict(valid, pixels=valid["pixels"][:1]),
               dict(valid, pixels=valid["pixels"].astype(np.float64)),
               dict(valid, pixels=np.full((2, 3, 2), np.nan, dtype=np.float32)),
               dict(valid, visible=np.ones((2, 2), dtype=bool)),
               dict(valid, visible=np.full((2, 3), np.nan)),
               dict(valid, visible=np.ones((2, 3), dtype=np.float32)),
               dict(valid, crop_xyxy=np.array([0, 0, 13, 8])),
               dict(valid, crop_xyxy=np.array([0, 0, 1, 8])),
               dict(valid, crop_xyxy=np.array([1, 1, 12, 8])),
               dict(valid, crop_xyxy=np.array([0., 0., 12., 8.])),
               {key: value for key, value in valid.items() if key != "crop_xyxy"},
               {"pixels": valid["pixels"]}]
        for arrays in bad:
            with self.subTest(fields=list(arrays)), patch(
                    "local_models.runtime.run_file_worker", return_value=arrays):
                with self.assertRaises(ValueError):
                    cotracker.track(self.paths[:2], self.queries)

    def test_worker_cannot_replace_supplied_initial_observation(self):
        pixels = np.tile(self.queries, (2, 1, 1))
        pixels[0, 0, 0] += 0.25
        with patch("local_models.runtime.run_file_worker",
                   return_value={"pixels": pixels, "visible": np.ones((2, 3), dtype=bool),
                                 "crop_xyxy": np.array([0, 0, 12, 8])}):
            with self.assertRaisesRegex(ValueError, "initial query observations"):
                cotracker.track(self.paths[:2], self.queries)

    def test_two_pass_refinement_reinitializes_and_restores_original_coordinates(self):
        queries = np.array([[4.25, 3.], [5., 4.]], dtype=np.float32)
        predictor = FakePredictor()
        result = cotracker._run_refined(self.paths[:2], queries, predictor, device="cpu")
        self.assertEqual(predictor.passes, [[[0, 1]], [[0, 1]]])
        self.assertEqual(predictor.cache_resets, 1)
        self.assertEqual(predictor.image_sizes, [(8, 12), (4, 4)])
        np.testing.assert_array_equal(result["crop_xyxy"], [3, 2, 7, 6])
        np.testing.assert_array_equal(predictor.query_history[0][0, :, 1:], queries)
        np.testing.assert_array_equal(predictor.query_history[1][0, :, 1:], queries - [3, 2])
        np.testing.assert_array_equal(result["pixels"][0], queries)
        np.testing.assert_allclose(result["pixels"][1], queries + [.125, .01], atol=1e-6)

    def test_crop_envelope_includes_every_prediction_and_clips_image_edges(self):
        queries = np.array([[4., 3.]], dtype=np.float32)
        # Visibility is deliberately absent from the crop API: even an occluded
        # prediction contributes its extent rather than shrinking the context.
        pixels = np.array([[[4., 3.]], [[9., 6.]]], dtype=np.float32)
        np.testing.assert_array_equal(cotracker._refinement_crop(pixels, queries, (12, 8)),
                                      [2, 1, 12, 8])
        far = np.array([[[-5., -4.]], [[16., 10.]]], dtype=np.float32)
        np.testing.assert_array_equal(cotracker._refinement_crop(far, queries, (12, 8)),
                                      [0, 0, 12, 8])
        with self.assertRaisesRegex(ValueError, "Non-finite coarse"):
            cotracker._refinement_crop(np.full((2, 1, 2), np.nan), queries, (12, 8))

    def test_stationary_single_query_crop_has_valid_native_dimensions(self):
        queries = np.array([[11., 7.]], dtype=np.float32)
        pixels = np.tile(queries, (2, 1, 1))
        np.testing.assert_array_equal(cotracker._refinement_crop(pixels, queries, (12, 8)),
                                      [10, 6, 12, 8])

    def test_checkpoint_is_resolved_from_installed_package_without_download(self):
        source = self.root / "checkout" / "cotracker" / "__init__.py"
        checkpoint = source.parents[1] / "checkpoints" / "scaled_online.pth"
        with patch("local_models.cotracker.importlib.util.find_spec",
                   return_value=SimpleNamespace(origin=str(source))) as find_spec:
            with self.assertRaisesRegex(FileNotFoundError, "Local CoTracker3 checkpoint is missing"):
                cotracker._local_checkpoint()
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test fixture, never loaded")
            self.assertEqual(cotracker._local_checkpoint(), checkpoint)
            find_spec.assert_called_with("cotracker")
        with patch("local_models.cotracker.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(ImportError, "no download"):
                cotracker._local_checkpoint()


if __name__ == "__main__":
    unittest.main()
# END local CoTracker regression tests.
