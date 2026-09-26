"""Analytic geometry and small RGB-D sequences exercise observation contracts.

No model inference or external data is used. Tests independently construct the
motions, including camera motion, so a plausible-looking fitted path alone is
not the correctness criterion.
"""

# BEGIN calibrated rigid-part observation tests.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from manipulation.observations import (
    backproject, load_calibration, load_rgbd, poses_from_tracks,
    rigid_registration, track_rgbd,
)


def transform(points, matrix):
    return (matrix @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]


def pose(angles=(0, 0, 0), translation=(0, 0, 0)):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def write_camera(root, width=96, height=80, scale=1000):
    matrix = np.array([[100., 0., width / 2], [0., 110., height / 2], [0., 0., 1.]])
    (root / "intrinsic.json").write_text(json.dumps({
        "width": width, "height": height,
        "intrinsic_matrix": matrix.ravel(order="F").tolist(),
    }))
    (root / "config.json").write_text(json.dumps({"depth_scale": scale}))
    return matrix


class CalibrationAndDepthTest(unittest.TestCase):
    def test_open3d_column_major_and_divisor_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = write_camera(root, width=4, height=3, scale=2000)
            camera = load_calibration(root / "intrinsic.json")
            np.testing.assert_array_equal(camera.K, expected)
            self.assertEqual(camera.depth_scale, 0.0005)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            Image.new("RGB", (4, 3)).save(root / "rgb/00000.jpg")
            raw = np.full((3, 4), 4000, dtype=np.uint16)
            raw[0, 0] = 0
            Image.fromarray(raw).save(root / "depth/00000.png")
            _, depth, _ = load_rgbd(root)
            self.assertEqual(depth[1, 1], 2.0)
            cloud = backproject(depth, expected)
            self.assertEqual(cloud.shape, (11, 3))
            np.testing.assert_allclose(cloud[:, 2], 2.0)

    def test_explicit_metric_units_and_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_camera(root)
            (root / "config.json").write_text(json.dumps({"depth_unit_m": 1.0}))
            self.assertEqual(load_calibration(root).depth_scale, 1.0)
            (root / "config.json").write_text(json.dumps({"depth_unit_m": 1.0, "depth_scale": 1000}))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_calibration(root)

    def test_invalid_depth_is_filtered_only_for_unordered_clouds(self):
        depth = np.array([[0., -1., np.nan], [np.inf, 2., 3.]])
        matrix = np.array([[2., 0., 1.], [0., 3., 0.5], [0., 0., 1.]])
        result = backproject(depth, matrix)
        np.testing.assert_allclose(result, [[0., 1 / 3, 2.], [1.5, 0.5, 3.]])
        for uv in ([0., 0.], [1., 0.], [2., 0.], [0., 1.]):
            with self.subTest(uv=uv), self.assertRaisesRegex(ValueError, "invalid depth"):
                backproject(depth, matrix, [uv])
        for uv in ([-0.1, 1.], [3., 1.], [1., 2.]):
            with self.subTest(uv=uv), self.assertRaisesRegex(ValueError, "out of bounds"):
                backproject(depth, matrix, [uv])
        with self.assertRaisesRegex(ValueError, "finite N-by-2"):
            backproject(depth, matrix, [[np.nan, 1.]])
        selected = np.zeros_like(depth, dtype=bool)
        selected[1, 1] = True
        with self.assertRaisesRegex(ValueError, "outside the mask"):
            backproject(depth, matrix, [[2., 1.]], mask=selected)

    def test_subpixel_rays_and_calibration_shape(self):
        depth = np.full((4, 5), 2.)
        matrix = np.array([[2., 0., 1.], [0., 4., 1.], [0., 0., 1.]])
        np.testing.assert_allclose(backproject(depth, matrix, [[1.25, 1.5]]), [[0.25, 0.25, 2.]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_camera(root, width=4, height=3)
            (root / "rgb").mkdir()
            (root / "depth").mkdir()
            Image.new("RGB", (4, 3)).save(root / "rgb/00000.png")
            Image.fromarray(np.ones((2, 4), dtype=np.uint16)).save(root / "depth/00000.png")
            with self.assertRaisesRegex(ValueError, "dimensions must agree"):
                load_rgbd(root)


class CorrespondenceMotionTest(unittest.TestCase):
    def test_planar_correspondences_observe_rotation_with_outliers(self):
        rng = np.random.default_rng(18)
        source = np.column_stack((rng.uniform(-0.3, 0.3, (80, 2)), np.full(80, 1.4)))
        expected = pose((0.4, -0.25, 0.8), (0.12, -0.2, 0.04))
        target = transform(source, expected)
        target += rng.normal(0, 0.0002, target.shape)
        target[:18] = rng.uniform(-2, 2, (18, 3))
        fitted, diagnostics = rigid_registration(source, target, max_error=0.002)
        np.testing.assert_allclose(fitted[:3, :3], expected[:3, :3], atol=6e-4)
        np.testing.assert_allclose(fitted[:3, 3], expected[:3, 3], atol=6e-4)
        self.assertGreaterEqual(diagnostics["num_inliers"], 60)
        self.assertFalse(diagnostics["inlier_mask"][:18].any())

    def test_minimal_triangle_is_valid_but_collinear_and_nonrigid_are_not(self):
        source = np.array([[0., 0., 1.], [0.2, 0., 1.], [0., 0.3, 1.]])
        expected = pose((0.5, 0.3, -0.2), (0.1, -0.2, 0.3))
        fitted, _ = rigid_registration(source, transform(source, expected))
        np.testing.assert_allclose(fitted, expected, atol=1e-12)
        line = np.column_stack((np.arange(8), np.zeros((8, 2))))
        with self.assertRaisesRegex(ValueError, "noncollinear"):
            rigid_registration(line, line)
        rng = np.random.default_rng(29)
        random = rng.uniform(-1, 1, (40, 3))
        with self.assertRaisesRegex(ValueError, "consensus"):
            rigid_registration(random, random * 1.7, max_error=0.001)

    def test_unordered_arrays_do_not_pass_as_known_correspondence(self):
        rng = np.random.default_rng(11)
        source = rng.uniform(-1, 1, (45, 3))
        # Both clouds contain exactly the same geometry. A nearest-neighbour
        # cloud score would be perfect, but these supplied feature IDs are wrong.
        with self.assertRaisesRegex(ValueError, "consensus"):
            rigid_registration(source, source[rng.permutation(len(source))], max_error=0.001)

    def test_camera_and_reference_compensation_with_missing_observations(self):
        rng = np.random.default_rng(17)
        body = rng.uniform(-0.2, 0.2, (35, 3)) + [0.2, 0.1, 1.3]
        reference = rng.uniform(-0.4, 0.4, (40, 3)) + [-0.4, 0., 1.6]
        actual_body = [np.eye(4), pose((0.1, 0.2, 0.3), (0.1, 0.02, -0.04)),
                       pose((-0.2, 0.1, 0.6), (0.12, 0.08, -0.02))]
        cameras = [np.eye(4), pose((0.2, -0.2, 0.), (0.2, -0.1, 0.03)),
                   pose((0.3, 0.2, -0.1), (-0.2, 0.3, 0.1))]
        tracks = np.asarray([transform(body, camera @ motion) for camera, motion in zip(cameras, actual_body)])
        refs = np.asarray([transform(reference, camera) for camera in cameras])
        valid = np.ones(tracks.shape[:2], dtype=bool)
        valid[1, :5] = False
        tracks[~valid] = np.nan
        result = poses_from_tracks(tracks, valid_mask=valid, reference_tracks=refs,
                                   reference_valid_mask=np.ones(refs.shape[:2], dtype=bool),
                                   frame_indices=[0, 2, 4])
        initial = pose(translation=body.mean(axis=0))
        np.testing.assert_allclose(result["poses"], [matrix @ initial for matrix in actual_body], atol=1e-12)
        for frame, expected in enumerate(actual_body):
            np.testing.assert_allclose(transform(result["points"], result["poses"][frame]),
                                       transform(body, expected), atol=1e-12)
        np.testing.assert_array_equal(result["frame_indices"], [0, 2, 4])
        self.assertIn("compensated", result["camera_assumption"])

    def test_track_contract_and_loss_are_explicit(self):
        rng = np.random.default_rng(12)
        tracks = rng.uniform(-1, 1, (2, 10, 3))
        valid = np.ones((2, 10), dtype=bool)
        tracks[1] = tracks[0] + [0.1, 0., 0.]
        result = poses_from_tracks(tracks, valid_mask=valid)
        self.assertEqual(result["camera_assumption"], "stationary camera required")
        for indices in ([-1, 1], [0, 0], [0., np.nan], ["0", "1"]):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "integer frame IDs"):
                poses_from_tracks(tracks, valid_mask=valid, frame_indices=indices)
        # Unsigned subtraction wraps on descending values. Direct ordering must
        # reject those IDs, while preserving very large, correctly ordered IDs.
        with self.assertRaisesRegex(ValueError, "integer frame IDs"):
            poses_from_tracks(tracks, valid_mask=valid, frame_indices=np.array([2, 1], np.uint64))
        large = np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64)
        np.testing.assert_array_equal(poses_from_tracks(tracks, valid_mask=valid,
                                                       frame_indices=large)["frame_indices"], large)
        valid[1, 2:] = False
        with self.assertRaisesRegex(ValueError, "failed at frame 1"):
            poses_from_tracks(tracks, valid_mask=valid)
        with self.assertRaisesRegex(ValueError, "boolean valid_mask"):
            poses_from_tracks(tracks, valid_mask=valid.astype(int))
        with self.assertRaisesRegex(ValueError, "supplied together"):
            poses_from_tracks(tracks, valid_mask=valid, reference_tracks=tracks)


class SmallRGBDSequenceTest(unittest.TestCase):
    def write_sequence(self, root, *, texture=True, invalidate_second=False):
        import cv2

        matrix = write_camera(root, width=128, height=96)
        (root / "rgb").mkdir()
        (root / "depth").mkdir()
        mask = np.zeros((96, 128), dtype=bool)
        mask[14:82, 16:104] = True
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, (96, 128), dtype=np.uint8) if texture else np.zeros((96, 128), np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        for frame in range(3):
            shifted = cv2.warpAffine(gray, np.array([[1., 0., 2 * frame], [0., 1., frame]]), (128, 96))
            Image.fromarray(shifted).convert("RGB").save(root / "rgb" / f"{frame:05d}.png")
            depth = np.full((96, 128), 1500, dtype=np.uint16)
            if frame == 1 and invalidate_second:
                depth[:] = 0
            Image.fromarray(depth).save(root / "depth" / f"{frame:05d}.png")
        return mask, matrix

    # BEGIN learned-tracker boundary: CPU tests supply independent image tracks,
    # while real calibrated backprojection and rigid registration remain active.
    @staticmethod
    def known_tracks(paths, queries):
        pixels = np.asarray([queries + [2 * frame, frame] for frame in range(len(paths))], np.float32)
        return {"pixels": pixels, "visible": np.ones(pixels.shape[:2], bool),
                "crop_xyxy": np.array([0, 0, 128, 96])}

    def test_tracked_pixels_and_depth_recover_known_metric_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask, matrix = self.write_sequence(root)
            with patch("local_models.cotracker.track", side_effect=self.known_tracks):
                result = track_rgbd(root, mask, max_error=0.004)
            relative = result["poses"] @ np.linalg.inv(result["poses"][0])
            expected = [pose(translation=(2 * frame * 1.5 / matrix[0, 0], frame * 1.5 / matrix[1, 1], 0))
                        for frame in range(3)]
            np.testing.assert_allclose(relative, expected, atol=0.002)
            self.assertLessEqual(len(result["points"]), 3000)
            self.assertEqual(result["poses"].shape, (3, 4, 4))

    def test_visibility_or_depth_loss_does_not_create_a_stationary_success(self):
        for invisible in (True, False):
            with self.subTest(invisible=invisible), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mask, _ = self.write_sequence(root, invalidate_second=not invisible)

                def observations(paths, queries):
                    result = self.known_tracks(paths, queries)
                    if invisible:
                        result["visible"][1] = False
                    return result

                with patch("local_models.cotracker.track", side_effect=observations):
                    with self.assertRaisesRegex(ValueError, "lost body correspondences"):
                        track_rgbd(root, mask)
    # END learned-tracker boundary.


if __name__ == "__main__":
    unittest.main()
# END calibrated rigid-part observation tests.
