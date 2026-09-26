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
    backproject, load_calibration, load_rgbd, poses_from_models,
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

# BEGIN original-mesh pose and fixed-material-point contracts.
class ModelPoseTest(unittest.TestCase):
    def test_arbitrary_camera_motion_is_removed_without_losing_object_rotation(self):
        rng = np.random.default_rng(17)
        material_points = rng.uniform(-0.15, 0.15, (35, 3)) + [0.4, -0.2, 0.3]
        bodies = np.asarray([pose((.1*f, -.06*f, .13*f), (.2+.01*f, .1, 1.2))
                             for f in range(8)])
        cameras = np.asarray([pose(rng.uniform(-.5, .5, 3), rng.uniform(-.2, .2, 3))
                              for _ in bodies])
        reference = pose((.6, -.4, .2), (-.4, .3, 1.8))
        observations = cameras @ bodies
        references = cameras @ reference
        initial_points = transform(material_points, observations[0])
        result = poses_from_models(observations, initial_points,
                                   reference_camera_from_model=references,
                                   frame_indices=np.arange(8)*2)
        np.testing.assert_allclose(result["poses"][0], pose(translation=initial_points.mean(0)), atol=1e-12)
        for frame, world_from_body in enumerate(bodies):
            # Expected material coordinates are constructed in the physical
            # world, then expressed in the first camera independently of the
            # body-origin convention chosen by poses_from_models.
            expected = transform(material_points, cameras[0] @ world_from_body)
            actual = transform(result["points"], result["poses"][frame])
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_array_equal(result["frame_indices"], np.arange(8)*2)
        self.assertIn("compensated", result["camera_assumption"])

    def test_moving_reference_defines_relative_motion_not_a_stationary_world(self):
        material_points = np.array([[.1, .2, .3], [-.1, .3, .2], [.2, -.2, .1]])
        bodies = np.array([pose((0, .1*f, 0), (.2, .1, 1.2)) for f in range(3)])
        refs = np.array([pose((0, 0, .2*f), (-.3, .03*f, 1.4)) for f in range(3)])
        cameras = np.array([pose((.1*f, -.1, .1), (.02*f, .2, -.1)) for f in range(3)])
        initial_points = transform(material_points, cameras[0] @ bodies[0])
        result = poses_from_models(cameras @ bodies, initial_points,
                                   reference_camera_from_model=cameras @ refs)
        for frame in range(3):
            expected = transform(material_points, cameras[0] @ refs[0] @ np.linalg.inv(refs[frame]) @ bodies[frame])
            np.testing.assert_allclose(transform(result["points"], result["poses"][frame]), expected, atol=1e-12)

    def test_nonzero_mesh_center_and_arbitrary_mesh_frame_do_not_change_body_motion(self):
        models = np.array([pose((.2, .3*f, -.1), (.4, .1*f, 1.6)) for f in range(4)])
        refs = np.array([pose((0, -.2, .05*f), (-.4, .2, 1.4)) for f in range(4)])
        points = np.array([[.2, .1, 1.4], [.3, .4, 1.2], [.5, -.2, 1.3]])
        result = poses_from_models(models, points, reference_camera_from_model=refs)
        mesh_change = pose((.6, -.4, .2), (1., -2., 3.))
        reference_change = pose((-.4, .8, -.6), (-.5, .7, -.9))
        reexpressed = poses_from_models(models @ np.linalg.inv(mesh_change), points,
                                      reference_camera_from_model=refs @ np.linalg.inv(reference_change))
        np.testing.assert_allclose(reexpressed["poses"], result["poses"], atol=1e-12)
        np.testing.assert_allclose(reexpressed["model_from_body"], mesh_change @ result["model_from_body"], atol=1e-12)
        np.testing.assert_array_equal(reexpressed["points"], result["points"])

    def test_stationary_camera_and_single_frame_bind_points_once(self):
        initial_model = pose((.4, -.5, .7), (.2, -.1, 1.5))
        material_points = np.array([[.2, .1, 0], [-.1, .2, .1], [.1, -.2, .2]])
        points = transform(material_points, initial_model)
        original_points = points.copy()
        single = poses_from_models(initial_model[None], points)
        np.testing.assert_allclose(transform(single["points"], single["poses"][0]), points, atol=1e-12)
        final_model = pose((-.2, .3, 1.), (.3, .2, 1.6))
        result = poses_from_models([initial_model, final_model], points)
        np.testing.assert_allclose(transform(result["points"], result["poses"][1]),
                                   transform(material_points, final_model), atol=1e-12)
        np.testing.assert_array_equal(points, original_points)
        self.assertEqual(result["camera_assumption"], "stationary camera required")

    def test_invalid_poses_and_point_geometry_are_rejected(self):
        points = np.array([[0., 0., 1.], [.1, 0., 1.], [0., .2, 1.]])
        invalid = [np.eye(4), np.empty((0, 4, 4))]
        for entry in ((0, 0, np.nan), (3, 0, .2), (0, 0, -1.), (1, 1, 2.)):
            matrix = np.eye(4)
            matrix[entry[:2]] = entry[2]
            invalid.append(matrix[None])
        for prediction in invalid:
            with self.subTest(shape=prediction.shape), self.assertRaises(ValueError):
                poses_from_models(prediction, points)
        with self.assertRaisesRegex(ValueError, "same frame sequence"):
            poses_from_models(np.array([np.eye(4)]*2), points, reference_camera_from_model=np.eye(4)[None])
        for geometry in ([], [[0, 1]], [[0, np.nan, 1]]):
            with self.subTest(geometry=geometry), self.assertRaisesRegex(ValueError, "initial_points"):
                poses_from_models(np.eye(4)[None], geometry)

    def test_frame_indices_preserve_unsigned_order_and_reject_invalid_values(self):
        models = np.array([np.eye(4), pose(translation=(.1, 0, 0))])
        points = np.array([[0., 0., 1.], [.1, 0., 1.], [0., .2, 1.]])
        for indices in ([-1, 1], [0, 0], [0., np.nan], ["0", "1"], np.array([2, 1], np.uint64)):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "integer frame IDs"):
                poses_from_models(models, points, frame_indices=indices)
        # Comparing adjacent IDs, rather than subtracting them, avoids unsigned
        # wraparound and preserves legitimately large ordered identifiers.
        large = np.array([2**64-2, 2**64-1], dtype=np.uint64)
        result = poses_from_models(models, points, frame_indices=large)
        np.testing.assert_array_equal(result["frame_indices"], large)
# END original-mesh pose and fixed-material-point contracts.


# BEGIN isolated FoundationPose boundary; no model loading in these CPU tests.
class SmallRGBDSequenceTest(unittest.TestCase):
    def write_sequence(self, root, *, invalidate_second=False):
        matrix = write_camera(root, width=128, height=96)
        (root / "rgb").mkdir()
        (root / "depth").mkdir()
        (root / "meshes").mkdir()
        (root / "meshes/carrier.obj").write_text("v 0 0 0\nv .1 0 0\nv 0 .1 0\nf 1 2 3\n")
        config = json.loads((root / "config.json").read_text())
        config["pose_models"] = {"carrier": {"mesh": "meshes/carrier.obj", "scale": 1.}}
        (root / "config.json").write_text(json.dumps(config))
        mask = np.zeros((96, 128), dtype=bool)
        mask[14:82, 16:104] = True
        for frame in range(3):
            Image.new("RGB", (128, 96), color=(80+frame, 90, 100)).save(root / "rgb" / f"{frame:05d}.png")
            depth = np.full((96, 128), 1500, dtype=np.uint16)
            if frame == 1 and invalidate_second:
                depth[:] = 0
            Image.fromarray(depth).save(root / "depth" / f"{frame:05d}.png")
        return mask, matrix

    @staticmethod
    def known_models(paths, depths, *, K, depth_scale, masks, models):
        # This fake supplies known *model* poses only. It does not pretend to
        # exercise the actual FoundationPose renderer or its quality checks.
        count = len(paths)
        first = pose((.4, -.3, .2), (.5, -.1, 1.3))
        reference = pose((-.3, .4, -.2), (-.4, .2, 1.7))
        output = []
        for frame in range(count):
            body_motion = pose((0, 0, .1*frame), (.03*frame, .01*frame, 0))
            camera = np.eye(4) if len(masks) == 1 else pose((.05*frame, -.03*frame, 0), (.1*frame, 0, 0))
            output.append([camera @ body_motion @ first] + ([] if len(masks) == 1 else [camera @ reference]))
        return {"camera_from_model": np.asarray(output), "validated": np.ones((count, len(masks)), bool),
                "support_fraction": np.full((count, len(masks)), .9)}

    def test_model_pose_boundary_recovers_known_motion_and_preserves_raw_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask, matrix = self.write_sequence(root)
            with patch("local_models.foundationpose.track", side_effect=self.known_models) as worker:
                result = track_rgbd(root, mask)
            relative = result["poses"] @ np.linalg.inv(result["poses"][0])
            expected = [pose((0, 0, .1*f), (.03*f, .01*f, 0)) for f in range(3)]
            np.testing.assert_allclose(relative, expected, atol=1e-12)
            self.assertLessEqual(len(result["points"]), 3000)
            self.assertEqual(result["tracking_camera_from_model"].shape, (3, 1, 4, 4))
            np.testing.assert_allclose(result["tracking_diagnostics"]["support_fraction"], .9)
            np.testing.assert_array_equal(worker.call_args.kwargs["K"], matrix)
            self.assertEqual(worker.call_args.kwargs["depth_scale"], .001)
            self.assertEqual(worker.call_args.kwargs["models"], [{"mesh": str(root / "meshes/carrier.obj"), "scale": 1.}])

    def test_two_entities_use_ordered_masks_and_independent_models_with_explicit_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask, _ = self.write_sequence(root)
            reference_mask = np.zeros_like(mask)
            reference_mask[10:80, 108:124] = True
            config = json.loads((root / "config.json").read_text())
            config["pose_models"]["carrier"]["scale"] = [.001, .002, .003]
            config["pose_models"]["reference"] = {"mesh": str(root / "meshes/carrier.obj"), "scale": 2.}
            (root / "config.json").write_text(json.dumps(config))
            with patch("local_models.foundationpose.track", side_effect=self.known_models) as worker:
                result = track_rgbd(root, mask, reference_mask=reference_mask)
            np.testing.assert_array_equal(worker.call_args.kwargs["masks"], np.stack([mask, reference_mask]))
            self.assertEqual(worker.call_args.kwargs["models"][0]["scale"], [.001, .002, .003])
            self.assertEqual(worker.call_args.kwargs["models"][1]["scale"], 2.)
            relative = result["poses"] @ np.linalg.inv(result["poses"][0])
            np.testing.assert_allclose(relative, [pose((0, 0, .1*f), (.03*f, .01*f, 0)) for f in range(3)], atol=1e-12)

    def test_missing_instance_models_or_files_fail_before_inference(self):
        for specification in (None, {}, {"carrier": {"category": "cup"}},
                              {"carrier": {"mesh": "missing.obj", "scale": 1.}}):
            with self.subTest(specification=specification), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mask, _ = self.write_sequence(root)
                (root / "config.json").write_text(json.dumps({"pose_models": specification}))
                with patch("local_models.foundationpose.track") as worker:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        track_rgbd(root, mask)
                    worker.assert_not_called()

    def test_model_scale_must_be_explicit_finite_positive_and_correctly_shaped(self):
        for scale in (None, 0, -1, True, "1", np.nan, np.inf, [], [1, 2], [1, 0, 1], [1, np.inf, 1], [True, 1, 1]):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mask, _ = self.write_sequence(root)
                config = json.loads((root / "config.json").read_text())
                config["pose_models"]["carrier"]["scale"] = scale
                (root / "config.json").write_text(json.dumps(config))
                with patch("local_models.foundationpose.track") as worker:
                    with self.assertRaisesRegex(ValueError, "scale"):
                        track_rgbd(root, mask)
                    worker.assert_not_called()

    def test_truncated_or_wrong_entity_pose_outputs_are_rejected(self):
        for shape in ((2, 1, 4, 4), (3, 2, 4, 4), (3, 4, 4)):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mask, _ = self.write_sequence(root)
                output = {"camera_from_model": np.zeros(shape), "validated": np.ones((3, 1), bool)}
                with patch("local_models.foundationpose.track", return_value=output):
                    with self.assertRaisesRegex(ValueError, "every frame and selected entity"):
                        track_rgbd(root, mask)

    def test_valid_se3_without_successful_geometric_validation_is_not_accepted(self):
        for validation in (None, np.ones((3, 1), int), np.ones((2, 1), bool), np.array([[True], [False], [True]])):
            with self.subTest(validation=validation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                mask, _ = self.write_sequence(root)
                output = {"camera_from_model": np.tile(np.eye(4), (3, 1, 1, 1))}
                if validation is not None:
                    output["validated"] = validation
                with patch("local_models.foundationpose.track", return_value=output):
                    with self.assertRaisesRegex(ValueError, "geometry validation"):
                        track_rgbd(root, mask)

    def test_worker_failure_and_absent_depth_never_become_stationary_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask, _ = self.write_sequence(root)
            with patch("local_models.foundationpose.track", side_effect=ValueError("rendered depth mismatch")) as worker:
                with self.assertRaisesRegex(ValueError, "rendered depth mismatch"):
                    track_rgbd(root, mask)
                worker.assert_called_once()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask, _ = self.write_sequence(root, invalidate_second=True)
            with patch("local_models.foundationpose.track") as worker:
                with self.assertRaisesRegex(ValueError, "frame 1 has insufficient valid depth"):
                    track_rgbd(root, mask)
                worker.assert_not_called()

    def test_max_frames_limits_both_rgb_and_depth_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask, _ = self.write_sequence(root)
            with patch("local_models.foundationpose.track", side_effect=self.known_models) as worker:
                result = track_rgbd(root, mask, max_frames=2)
            self.assertEqual([len(paths) for paths in worker.call_args.args], [2, 2])
            np.testing.assert_array_equal(result["frame_indices"], [0, 1])
            for value in (1, True, 2.5):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_frames"):
                    track_rgbd(root, mask, max_frames=value)
# END isolated FoundationPose boundary.


if __name__ == "__main__":
    unittest.main()
# END calibrated rigid-part observation tests.
