"""CPU regressions for mathematical guarantees and known failure cases.

These tests never initialize a renderer, a neural model, or a CUDA device.
They do not replace the separate original-scene RGB-D/MARCO experiment.
Every expected target center below follows an independently specified
coordinate transform or an explicit geometric symmetry, never a prediction.
"""

# BEGIN LOCAL_REGION_TRANSFER_MATHEMATICAL_REGRESSIONS
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from transfer.core import (
    TransferConfig,
    fit_sim3,
    fps_indices,
    infer_region,
    prepare_anchors,
    retained_anchor_rank,
    signature_scores,
    spatial_mode_gap,
    transfer_region,
)


def _transform(points, scale=1.7, rotation=None, translation=None):
    """Apply a known fixture transform independently of the fitted model."""
    if rotation is None:
        rotation = Rotation.from_rotvec([0.47, -0.62, 0.31]).as_matrix()
    if translation is None:
        translation = np.array([0.81, -0.26, 0.49])
    return scale * np.asarray(points) @ rotation.T + translation


def _surface_fixture(region_z=0.0, planar_anchors=False):
    """Three visible parallel surfaces and an asymmetric convex source patch.

    The triangular patch makes coordinate-wise median differ from its
    centroid. Its exact centroid is explicitly observed, removing candidate
    discretization as an explanation for a center-equivariance failure.
    Object names and support labels are deliberately absent from inference.
    """
    axis = np.arange(-15, 16, dtype=float) * 0.04
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    layers = [
        np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
        for z in (-0.3, 0.0, 0.3)
    ]
    points = np.concatenate(layers)
    region = (
        np.isclose(points[:, 2], region_z)
        & (points[:, 0] >= -0.2 - 1e-12)
        & (points[:, 1] >= -0.2 - 1e-12)
        & (points[:, 0] + points[:, 1] <= 0.04 + 1e-12)
    )
    centroid = points[region].mean(axis=0)
    points = np.vstack((points, centroid))
    region = np.r_[region, True]
    if region_z:
        mirror = centroid.copy()
        mirror[2] *= -1
        points = np.vstack((points, mirror))
        region = np.r_[region, False]
    # The same observed coordinates identify oracle correspondences after
    # a known transform. They are never selected using a target-region mask.
    anchor_xy = np.isin(np.rint(points[:, 0] / 0.04), [-15, 0, 15])
    anchor_xy &= np.isin(np.rint(points[:, 1] / 0.04), [-15, 0, 15])
    if planar_anchors:
        anchor_xy &= np.isclose(points[:, 2], 0)
    anchors = np.flatnonzero(anchor_xy)
    return points, region, anchors, centroid


class TestDistanceSignature(unittest.TestCase):
    """Check the claimed invariance and counterexamples to broader claims."""

    def test_independent_3d_similarity_frames_preserve_signature_scores(self):
        rng = np.random.default_rng(20261013)
        anchors = rng.normal(size=(17, 3))
        center = np.array([0.17, -0.24, 0.38])
        candidates = np.vstack((center, rng.normal(size=(11, 3))))
        expected, _ = signature_scores(candidates, center, anchors, anchors)
        source_rotation = Rotation.from_rotvec([1.01, -0.23, 0.57]).as_matrix()
        target_rotation = Rotation.from_rotvec([-0.39, 0.64, 0.28]).as_matrix()
        moved, _ = signature_scores(
            _transform(candidates, 0.37, target_rotation, np.array([-2.0, 0.6, 1.1])),
            _transform(center, 2.4, source_rotation, np.array([0.2, -0.3, 0.9])),
            _transform(anchors, 2.4, source_rotation, np.array([0.2, -0.3, 0.9])),
            _transform(anchors, 0.37, target_rotation, np.array([-2.0, 0.6, 1.1])),
        )
        np.testing.assert_allclose(moved, expected, atol=2e-14, rtol=2e-14)
        self.assertLess(moved[0], 2e-14)

    def test_planar_anchor_mirror_is_a_real_distance_ambiguity(self):
        anchors = np.array([
            [-1, -1, 0], [-1, 0, 0], [-1, 1, 0],
            [0, -1, 0], [0, 1, 0],
            [1, -1, 0], [1, 0, 0], [1, 1, 0],
        ], dtype=float)
        center = np.array([0.12, -0.07, 0.3])
        mirror = center * [1, 1, -1]
        candidates = np.vstack((center, mirror))
        scores, _ = signature_scores(candidates, center, anchors, anchors)
        np.testing.assert_allclose(scores, 0, atol=1e-14)
        gap, alternate = spatial_mode_gap(candidates, scores, 0, 0.1)
        self.assertEqual(alternate, 1)
        self.assertLess(abs(gap), 1e-14)

    def test_trimming_can_remove_every_anchor_that_resolves_a_mirror(self):
        # Eight coplanar anchors and two off-plane anchors have full 3-D
        # rank, but a 20% trim can discard precisely the only useful two.
        flat = np.array([
            [-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0],
            [0, 1, 0], [1, -1, 0], [1, 0, 0], [1, 1, 0],
        ], dtype=float)
        anchors = np.vstack((flat, [[0.2, 0.3, 0.7], [-0.4, 0.1, 0.9]]))
        center = np.array([0.12, -0.07, 0.3])
        mirror = center * [1, 1, -1]
        candidates = np.vstack((center, mirror))
        trimmed, _ = signature_scores(
            candidates, center, anchors, anchors, trim_fraction=0.2)
        complete, _ = signature_scores(
            candidates, center, anchors, anchors, trim_fraction=0.0)
        self.assertEqual(np.linalg.matrix_rank(anchors - anchors.mean(0)), 3)
        self.assertLess(abs(trimmed[1]), 1e-14)
        self.assertGreater(complete[1], 0.01)
        retained = retained_anchor_rank(mirror, center, anchors, anchors, trim_fraction=0.2)
        self.assertEqual(retained["source_rank"], 2)
        self.assertEqual(retained["target_rank"], 2)
        self.assertEqual(set(retained["kept_anchor_indices"]), set(range(8)))

    def test_spatial_alternative_survives_more_than_250_dense_low_cost_points(self):
        # An old top-250 shortcut sees only the first spatial cluster.
        phase = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        cluster = np.column_stack((0.01 * np.cos(phase), 0.01 * np.sin(phase),
                                   np.zeros(len(phase))))
        points = np.vstack((cluster, [2, 0, 0]))
        scores = np.r_[np.linspace(0, 1e-6, len(cluster)), 2e-6]
        gap, alternate = spatial_mode_gap(points, scores, 0, 0.2)
        self.assertEqual(alternate, 400)
        self.assertAlmostEqual(gap, 2e-6, places=14)

    def test_anisotropic_transform_does_not_preserve_distance_signature(self):
        anchors = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ], dtype=float)
        center = np.array([0.31, -0.27, 0.16])
        # This independently defined target is the correct affine image.
        # A nonzero score there disproves affine invariance; it does not
        # assert that either method must outperform the other on real data.
        stretch = np.diag([1.9, 0.7, 1.15])
        target_center = center @ stretch.T
        target_anchors = anchors @ stretch.T
        scores, _ = signature_scores(
            target_center[None], center, anchors, target_anchors, trim_fraction=0.0)
        self.assertGreater(scores[0], 0.02)


class TestGeometryContracts(unittest.TestCase):
    """Reject invalid geometry and retain a feature-free oracle boundary."""

    def test_farthest_sampling_never_reuses_a_duplicate_point_identity(self):
        indices = fps_indices(np.zeros((7, 3)), 20)
        # Coincident observations add no geometric information. Early stop
        # is preferable to fabricating seven independent spatial anchors.
        self.assertEqual(len(indices), 1)
        self.assertEqual(len(np.unique(indices)), len(indices))

    def test_similarity_rejects_zero_rank_and_collapsed_target(self):
        source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        for first, second in ((np.zeros((4, 3)), source), (source, np.zeros((4, 3)))):
            with self.subTest(collapsed_source=not np.any(first)):
                with self.assertRaises(ValueError):
                    fit_sim3(first, second)

    def test_collinear_correspondences_are_not_a_valid_similarity_model(self):
        line = np.column_stack((np.arange(8), np.zeros(8), np.zeros(8)))
        with self.assertRaises(ValueError):
            fit_sim3(line, _transform(line))

    def test_duplicate_geometric_anchors_produce_a_rejected_region(self):
        points, region, _, _ = _surface_fixture()
        anchors = np.arange(len(points), len(points) + 8)
        points = np.vstack((points, np.repeat([[0.6, 0.6, 0.3]], len(anchors), axis=0)))
        region = np.r_[region, np.zeros(len(anchors), dtype=bool)]
        result = transfer_region(
            points, points.copy(), region, anchors, anchors, method="oracle_relational")
        self.assertFalse(result["accepted"])
        self.assertIn("degenerate_anchors", result["reasons"])
        self.assertFalse(np.any(result["candidate_mask"]))

    def test_inference_api_does_not_accept_target_ground_truth(self):
        points, region, anchors, _ = _surface_fixture()
        # Ground truth belongs to evaluation, and cannot be supplied to the
        # inference function even accidentally through a permissive kwargs.
        with self.assertRaises(TypeError):
            transfer_region(points, points, region, anchors, anchors,
                            tgt_gt_mask=np.ones(len(points), dtype=bool))

    def test_mask_and_anchor_indices_are_not_silently_coerced(self):
        points, region, anchors, _ = _surface_fixture()
        for invalid in (anchors.astype(float), -np.ones_like(anchors)):
            with self.subTest(index_dtype=invalid.dtype, first=invalid[0]):
                with self.assertRaises(ValueError):
                    prepare_anchors(points, points, invalid, anchors, mode="oracle")
        prepared = prepare_anchors(points, points, anchors, anchors, mode="oracle")
        with self.assertRaises(ValueError):
            infer_region(region.astype(np.int64), prepared, method="oracle_relational")


class TestCompleteInference(unittest.TestCase):
    """Exercise localization and region recovery, not only score helpers."""

    def test_asymmetric_region_center_is_equivariant_in_three_dimensions(self):
        points, region, anchors, centroid = _surface_fixture()
        source_rotation = Rotation.from_rotvec([0.82, -0.44, 0.35]).as_matrix()
        rotated_source = _transform(points, 0.83, source_rotation, np.array([0.1, 0.4, -0.2]))
        source_center = _transform(centroid, 0.83, source_rotation, np.array([0.1, 0.4, -0.2]))
        target = _transform(rotated_source)
        expected = _transform(source_center)
        prepared = prepare_anchors(rotated_source, target, anchors, anchors, mode="oracle")
        result = infer_region(region, prepared, method="oracle_relational")
        self.assertTrue(result["accepted"], result["reasons"])
        np.testing.assert_allclose(result["candidate_center"], expected, atol=2e-12)
        self.assertTrue(np.any(result["candidate_mask"]))

    def test_uniform_similarity_has_no_artificial_relational_advantage(self):
        points, region, anchors, centroid = _surface_fixture()
        target = _transform(points)
        # Reuse the exact PreparedAnchors instance so correspondence choices
        # cannot explain a difference between the two center methods.
        prepared = prepare_anchors(points, target, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        global_result = infer_region(region, prepared, method="global_sim3")
        relation_result = infer_region(region, prepared, method="relational")
        self.assertTrue(global_result["accepted"], global_result["reasons"])
        self.assertTrue(relation_result["accepted"], relation_result["reasons"])
        np.testing.assert_allclose(global_result["candidate_center"], _transform(centroid), atol=2e-12)
        np.testing.assert_allclose(relation_result["candidate_center"], global_result["candidate_center"], atol=2e-12)
        np.testing.assert_array_equal(relation_result["candidate_mask"], global_result["candidate_mask"])

    def test_local_direct_and_global_share_surface_recovery_on_same_similarity(self):
        points, region, anchors, centroid = _surface_fixture()
        target = _transform(points)
        config = TransferConfig(ransac_iterations=100)
        global_prepared = prepare_anchors(
            points, target, anchors, anchors, mode="automatic", config=config)
        local_indices = np.flatnonzero(region)
        direct_prepared = prepare_anchors(
            points, target, local_indices, local_indices, mode="direct", config=config)
        whole = infer_region(region, global_prepared, method="global_sim3")
        local = infer_region(region, direct_prepared, method="direct_local")
        self.assertTrue(whole["accepted"], whole["reasons"])
        self.assertTrue(local["accepted"], local["reasons"])
        np.testing.assert_allclose(local["candidate_center"], _transform(centroid), atol=2e-12)
        np.testing.assert_array_equal(local["candidate_mask"], whole["candidate_mask"])

    def test_missing_local_surface_cannot_be_recentered_onto_distant_support(self):
        points, region, anchors, centroid = _surface_fixture()
        # Retain all registration anchors but remove the observed area around
        # the predicted middle-plane center. Farther parallel planes remain.
        visible = np.linalg.norm(points - centroid, axis=1) > 0.26
        visible[anchors] = True
        target = points[visible]
        remap = np.full(len(points), -1, dtype=int)
        remap[visible] = np.arange(visible.sum())
        prepared = prepare_anchors(points, target, anchors, remap[anchors],
                                   mode="automatic", config=TransferConfig(ransac_iterations=100))
        result = infer_region(region, prepared, method="global_sim3")
        self.assertFalse(result["accepted"])
        self.assertIn("predicted_center_without_nearby_surface", result["reasons"])
        diagnostics = result["diagnostics"]
        self.assertGreater(diagnostics["center_projection_distance_m"],
                           diagnostics["center_projection_limit_m"])

    def test_relation_only_rejects_visible_mirror_supports(self):
        points, region, anchors, _ = _surface_fixture(region_z=0.3, planar_anchors=True)
        prepared = prepare_anchors(points, points, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        result = infer_region(region, prepared, method="relation_only")
        self.assertFalse(result["accepted"])
        self.assertIn("ambiguous_spatial_modes", result["reasons"])
        self.assertLess(abs(result["diagnostics"]["raw_relation_gap"]), 1e-12)
        self.assertIsNotNone(result["diagnostics"]["raw_relation_alternate_index"])

    def test_coplanar_anchors_are_usable_on_a_single_visible_surface(self):
        points, region, _, centroid = _surface_fixture()
        surface = np.isclose(points[:, 2], 0)
        points, region = points[surface], region[surface]
        anchor_x = np.isin(np.rint(points[:, 0] / 0.04), [-15, 0, 15])
        anchor_y = np.isin(np.rint(points[:, 1] / 0.04), [-15, 0, 15])
        anchors = np.flatnonzero(anchor_x & anchor_y)
        target = _transform(points)
        prepared = prepare_anchors(points, target, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        result = infer_region(region, prepared, method="relation_only")
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertEqual(result["diagnostics"]["source_anchor_rank"], 2)
        np.testing.assert_allclose(result["candidate_center"], _transform(centroid), atol=2e-12)


def _pixel_observation(width=64, height=48, pixels=None):
    """Small CPU image chart with one explicit 3-D identity per visible pixel."""
    if pixels is None:
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        pixels = np.column_stack((xx.ravel(), yy.ravel()))
    pixels = np.asarray(pixels, dtype=np.int32)
    mask = np.zeros((height, width), dtype=bool)
    mask[pixels[:, 1], pixels[:, 0]] = True
    xyz = np.column_stack((pixels * 0.01, np.zeros(len(pixels))))
    return {"xyz": xyz, "pixels": pixels, "object_mask": mask}


class TestImageCoordinateBridge(unittest.TestCase):
    """Reuse tiny CPU feature maps to test the maintained MARCO coordinate rule."""

    def test_native_region_queries_agree_with_existing_native_matcher(self):
        from local_models.marco import match_features
        from transfer.features import native_region_pairs

        rng = np.random.default_rng(917)
        feature_pair = tuple(rng.normal(size=(1, 8, 8, 8)).astype(np.float32)
                             for _ in range(2))
        source = _pixel_observation()
        target = _pixel_observation()
        region = ((source["pixels"][:, 0] >= 8) & (source["pixels"][:, 0] <= 32)
                  & (source["pixels"][:, 1] >= 8) & (source["pixels"][:, 1] <= 32))
        pairs = native_region_pairs(source, target, region, feature_pair, count=12)
        expected, _ = match_features(
            feature_pair, source["pixels"][pairs["raw_source_idx"]], (64, 48),
            target["object_mask"])
        self.assertTrue(pairs["raw_valid"].all())
        np.testing.assert_allclose(pairs["raw_target_pixels"], expected, atol=2e-5, rtol=0)
        recovered_pixels = target["pixels"][pairs["raw_target_idx"]]
        np.testing.assert_array_equal(recovered_pixels, np.rint(expected).astype(int))

    def test_invalid_softargmax_in_a_batch_is_recorded_without_replacement(self):
        from transfer.features import native_region_pairs

        source = _pixel_observation(640, 480, pixels=[[0, 0], [639, 0]])
        target = _pixel_observation(640, 480, pixels=[[0, 0], [639, 0]])
        source_map = np.zeros((1, 2, 8, 8), dtype=np.float32)
        target_map = np.zeros_like(source_map)
        source_map[0, :, 0, 0] = [0, 1000]
        source_map[0, :, 0, 7] = [1, 0]
        target_map[0, :, 0, 0] = [1, 1]
        target_map[0, :, 0, 7] = [1, -1]
        # One query is sharply decoded onto a valid endpoint; another lies
        # between disconnected valid endpoints and has no measured depth.
        pairs = native_region_pairs(source, target, np.ones(2, bool),
                                    (source_map, target_map), count=2)
        self.assertEqual(int(pairs["raw_valid"].sum()), 1)
        self.assertEqual(int((~pairs["raw_valid"]).sum()), 1)
        self.assertTrue(np.all(pairs["raw_target_idx"][~pairs["raw_valid"]] == -1))
        self.assertEqual(len(pairs["src_anchor_idx"]), 1)
        self.assertTrue(np.all(pairs["tgt_anchor_idx"] >= 0))
        invalid_pixels = pairs["raw_target_pixels"][~pairs["raw_valid"]]
        for pixel in invalid_pixels:
            self.assertFalse(np.any(np.all(target["pixels"] == np.rint(pixel), axis=1)))

    def test_recorded_crop_changes_pixels_without_changing_point_identity(self):
        from transfer.experiment import feature_observation

        observation = _pixel_observation(pixels=[[19, 13], [27, 31], [47, 39]])
        original_pixels = observation["pixels"].copy()
        chart = feature_observation(observation, [15, 10, 51, 43])
        self.assertIs(chart["xyz"], observation["xyz"])
        np.testing.assert_array_equal(chart["pixels"] + [15, 10], original_pixels)
        np.testing.assert_array_equal(observation["pixels"], original_pixels)
        self.assertTrue(chart["object_mask"][chart["pixels"][:, 1], chart["pixels"][:, 0]].all())
        with self.assertRaises(ValueError):
            feature_observation(observation, [20, 10, 51, 43])


class TestEvaluationIsolation(unittest.TestCase):
    """Verify the runner's data boundary with real CPU inference and fake I/O."""

    def _run_fixture(self, temporary, *, altered_truth=False, malformed_pairs=False):
        from transfer import core, experiment

        points, region, anchors, centroid = _surface_fixture()
        pixels = np.column_stack((np.arange(len(points)) % 64, np.arange(len(points)) // 64))
        observation = _pixel_observation(pixels=pixels)
        observation["xyz"] = points
        source_truth = {
            "region_mask": region, "support_mask": np.isclose(points[:, 2], 0),
            "region_center": centroid, "canonical_xyz": points,
            "region_visible_area_fraction": np.array(1.0),
            "region_center_visible": np.array(True),
        }
        target_truth = dict(source_truth)
        if altered_truth:
            target_truth["region_mask"] = np.roll(region, 50)
            target_truth["region_center"] = centroid + np.array([0.2, 0, 0])
        completed = []
        real_infer = core.infer_region

        def tracked_infer(*args, **kwargs):
            result = real_infer(*args, **kwargs)
            completed.append(kwargs["method"])
            return result

        def read_archive(path):
            path = Path(path)
            if path.name == "observation.npz":
                return observation
            if path.name == "features.npz":
                return {"feature1": np.zeros(1), "feature2": np.zeros(1)}
            if path.name == "evaluation.npz" and path.parent.name == "src":
                return source_truth
            if path.name == "evaluation.npz" and path.parent.name == "tgt":
                self.assertEqual(completed, [
                    "global_sim3", "relation_only", "relational", "direct_local"])
                return target_truth
            raise AssertionError(f"Unexpected archive access: {path}")

        direct_indices = np.flatnonzero(region)
        automatic = {
            "src_anchor_idx": anchors.astype(float) if malformed_pairs else anchors,
            "tgt_anchor_idx": anchors.copy(),
        }
        direct = {
            "src_anchor_idx": direct_indices, "tgt_anchor_idx": direct_indices,
            "raw_source_idx": direct_indices, "raw_target_idx": direct_indices,
            "raw_target_pixels": pixels[direct_indices].astype(float),
            "raw_valid": np.ones(len(direct_indices), dtype=bool),
        }
        root = Path(temporary)
        feature_directory = root / "features" / "pair"
        feature_directory.mkdir(parents=True)
        (feature_directory / "inputs.json").write_text(json.dumps({
            "source_crop_xyxy": [0, 0, 64, 48], "target_crop_xyxy": [0, 0, 64, 48],
        }))
        pairs = [{"id": "pair", "source": "src", "target": "tgt"}]
        with (
            mock.patch.object(experiment, "read_pairs", return_value=pairs),
            mock.patch.object(experiment, "read_archive", side_effect=read_archive),
            mock.patch.object(experiment, "reciprocal_anchors", return_value=automatic),
            mock.patch.object(experiment, "native_region_pairs", return_value=direct),
            mock.patch.object(core, "infer_region", side_effect=tracked_infer),
            mock.patch("transfer.evaluation.comparison_image"),
            redirect_stdout(io.StringIO()),
        ):
            rows = experiment.evaluate_suite(root / "data", root / "features", root / "out")
        return root / "out" / "pair", rows

    def test_target_truth_is_read_after_inference_and_cannot_change_predictions(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_output, first_rows = self._run_fixture(first)
            second_output, second_rows = self._run_fixture(second, altered_truth=True)
            automatic = ("global_sim3", "relation_only", "relational", "direct_local")
            for name in automatic:
                with self.subTest(method=name):
                    with np.load(first_output / f"{name}.npz", allow_pickle=False) as before:
                        with np.load(second_output / f"{name}.npz", allow_pickle=False) as after:
                            np.testing.assert_array_equal(before["candidate_mask"], after["candidate_mask"])
                            np.testing.assert_allclose(before["candidate_center"], after["candidate_center"], atol=0)
                            np.testing.assert_array_equal(before["accepted"], after["accepted"])
            first_error = next(r["center_error_m"] for r in first_rows if r["method"] == "global_sim3")
            second_error = next(r["center_error_m"] for r in second_rows if r["method"] == "global_sim3")
            self.assertLess(first_error, 1e-12)
            self.assertGreater(second_error, 0.19)

    def test_malformed_correspondences_stop_the_run_instead_of_counting_as_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                self._run_fixture(temporary, malformed_pairs=True)


if __name__ == "__main__":
    unittest.main()
# END LOCAL_REGION_TRANSFER_MATHEMATICAL_REGRESSIONS
