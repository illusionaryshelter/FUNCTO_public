"""CPU regressions for fixed-metric surface neighborhoods and their evidence.

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
    sample_neighborhood,
    signature_scores,
    transfer_region,
)


SAMPLING = {"sampling_radius_m": 0.03, "sampling_spacing_m": 0.005, "max_modes": 3}


def _grid_pixels(points):
    """Give each regular fixture plane its own unchanged observation chart.

    Pixels are defined before coordinate transforms or observation removal.
    This fixture helper never infers a replacement chart from a filtered cloud:
    tests that remove observations must index the original pixels as well.
    Distinct planes occupy separate image tiles with an empty border.
    """
    _, u = np.unique(points[:, 0], return_inverse=True)
    _, v = np.unique(points[:, 1], return_inverse=True)
    _, layer = np.unique(points[:, 2], return_inverse=True)
    pixels = np.column_stack((u, v + layer * (v.max() + 8))).astype(np.int64)
    # The degenerate-anchor fixture deliberately repeats 3-D points. Such
    # observations still need distinct pixel identities, although anchor
    # validation rejects them before any neighborhood can be estimated.
    _, first = np.unique(pixels, axis=0, return_index=True)
    repeated = np.setdiff1d(np.arange(len(points)), first)
    if len(repeated):
        pixels[repeated, 0] = pixels[:, 0].max() + 8 + np.arange(len(repeated)) * 8
    return pixels


def _controls(points, source_camera=None, target_camera=None, target_pixels=None):
    """Supply independent observation geometry and one frozen sampling budget."""
    source_camera = np.array([0.0, 0.0, 1.0]) if source_camera is None else np.asarray(source_camera)
    target_camera = source_camera.copy() if target_camera is None else np.asarray(target_camera)
    return {**SAMPLING, "source_camera_origin": source_camera,
            "target_camera_origin": target_camera,
            "target_pixels": _grid_pixels(points) if target_pixels is None else target_pixels}


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
    axis = np.arange(-15, 16, dtype=float) * 0.005
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    layers = [
        np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
        for z in (-0.05, 0.0, 0.05)
    ]
    points = np.concatenate(layers)
    region = (
        np.isclose(points[:, 2], region_z)
        & (points[:, 0] >= -0.02 - 1e-12)
        & (points[:, 1] >= -0.02 - 1e-12)
        & (points[:, 0] + points[:, 1] <= 0.02 + 1e-12)
    )
    centroid = points[region].mean(axis=0)
    if not np.any(np.linalg.norm(points - centroid, axis=1) < 1e-12):
        points = np.vstack((points, centroid))
        region = np.r_[region, True]
    if region_z and not np.any(np.linalg.norm(points - centroid * [1, 1, -1], axis=1) < 1e-12):
        mirror = centroid.copy()
        mirror[2] *= -1
        points = np.vstack((points, mirror))
        region = np.r_[region, False]
    # The same observed coordinates identify oracle correspondences after
    # a known transform. They are never selected using a target-region mask.
    anchor_xy = np.isin(np.rint(points[:, 0] / 0.005), [-15, 0, 15])
    anchor_xy &= np.isin(np.rint(points[:, 1] / 0.005), [-15, 0, 15])
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
            points, points.copy(), region, anchors, anchors,
            method="oracle_relational", **_controls(points))
        self.assertFalse(result["accepted"])
        self.assertIn("degenerate_anchors", result["reasons"])
        self.assertEqual(result["modes"], [])

    def test_inference_api_does_not_accept_target_ground_truth(self):
        points, region, anchors, _ = _surface_fixture()
        # Ground truth belongs to evaluation, and cannot be supplied to the
        # inference function even accidentally through a permissive kwargs.
        with self.assertRaises(TypeError):
            transfer_region(points, points, region, anchors, anchors,
                            tgt_gt_mask=np.ones(len(points), dtype=bool), **_controls(points))

    def test_mask_and_anchor_indices_are_not_silently_coerced(self):
        points, region, anchors, _ = _surface_fixture()
        for invalid in (anchors.astype(float), -np.ones_like(anchors)):
            with self.subTest(index_dtype=invalid.dtype, first=invalid[0]):
                with self.assertRaises(ValueError):
                    prepare_anchors(points, points, invalid, anchors, mode="oracle")
        prepared = prepare_anchors(points, points, anchors, anchors, mode="oracle")
        with self.assertRaises(ValueError):
            infer_region(region.astype(np.int64), prepared, method="oracle_relational", **_controls(points))


def _plane_grid(half_width=0.04, spacing=0.002, z=0.0):
    ticks = np.arange(-round(half_width / spacing), round(half_width / spacing) + 1) * spacing
    xx, yy = np.meshgrid(ticks, ticks, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))


def _mirror_fixture():
    """Two close visible faces with correct but coplanar reference anchors."""
    upper, lower = _plane_grid(z=0.01), _plane_grid(z=-0.01)
    anchors = np.array([
        [-0.06, -0.06, 0], [-0.06, 0, 0], [-0.06, 0.06, 0],
        [0, -0.06, 0], [0, 0.06, 0],
        [0.06, -0.06, 0], [0.06, 0, 0], [0.06, 0.06, 0],
    ])
    points = np.vstack((upper, lower, anchors))
    region = (np.isclose(points[:, 2], 0.01)
              & (np.linalg.norm(points[:, :2], axis=1) <= 0.008 + 1e-12))
    indices = np.arange(len(upper) + len(lower), len(points))
    return points, region, indices


class TestCompleteInference(unittest.TestCase):
    """Test the final neighborhood interface rather than the removed patch API."""

    def test_contract_smoke_returns_observed_neighborhood_modes(self):
        points, region, anchors, _ = _surface_fixture()
        prepared = prepare_anchors(points, points, anchors, anchors, mode="oracle")
        result = infer_region(region, prepared, method="oracle_relational", **_controls(points))
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertTrue(result["modes"])
        self.assertNotIn("candidate_mask", result)
        first = result["modes"][0]
        np.testing.assert_array_equal(first["center"], points[first["seed_index"]])
        self.assertTrue(first["mask"][first["sample_indices"]].all())
        self.assertGreater(len(first["sample_indices"]), 1)

    def test_asymmetric_evidence_center_is_equivariant_in_three_dimensions(self):
        points, region, anchors, centroid = _surface_fixture()
        source_rotation = Rotation.from_rotvec([0.82, -0.44, 0.35]).as_matrix()
        offset = np.array([0.1, 0.4, -0.2])
        rotated_source = _transform(points, 0.83, source_rotation, offset)
        source_center = _transform(centroid, 0.83, source_rotation, offset)
        camera = _transform([0, 0, 1], 0.83, source_rotation, offset)
        target = _transform(rotated_source)
        prepared = prepare_anchors(rotated_source, target, anchors, anchors, mode="oracle")
        result = infer_region(region, prepared, method="oracle_relational",
                              **_controls(points, camera, _transform(camera)))
        self.assertTrue(result["accepted"], result["reasons"])
        np.testing.assert_allclose(result["modes"][0]["center"], _transform(source_center), atol=2e-12)

    def test_uniform_similarity_has_no_artificial_relational_advantage(self):
        points, region, anchors, centroid = _surface_fixture()
        target = _transform(points)
        prepared = prepare_anchors(points, target, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        controls = _controls(points, target_camera=_transform([0, 0, 1]))
        global_result = infer_region(region, prepared, method="global_sim3", **controls)
        relation_result = infer_region(region, prepared, method="relational", **controls)
        self.assertTrue(global_result["accepted"], global_result["reasons"])
        self.assertTrue(relation_result["accepted"], relation_result["reasons"])
        whole, relation = global_result["modes"][0], relation_result["modes"][0]
        np.testing.assert_allclose(whole["center"], _transform(centroid), atol=2e-12)
        np.testing.assert_allclose(relation["center"], whole["center"], atol=2e-12)
        np.testing.assert_array_equal(relation["mask"], whole["mask"])
        np.testing.assert_array_equal(relation["sample_indices"], whole["sample_indices"])

    def test_sampling_radius_is_metric_and_does_not_scale_with_container(self):
        points, region, anchors, _ = _surface_fixture()
        observed_radii = []
        for scale in (1.0, 2.0):
            with self.subTest(container_scale=scale):
                target = points * scale
                prepared = prepare_anchors(points, target, anchors, anchors, mode="automatic",
                                           config=TransferConfig(ransac_iterations=100))
                result = infer_region(region, prepared, method="global_sim3",
                                      **_controls(points, target_camera=[0, 0, scale]))
                self.assertTrue(result["accepted"], result["reasons"])
                first = result["modes"][0]
                delta = target[first["mask"]] - first["center"]
                normal = first["normal"]
                radius = np.linalg.norm(delta - (delta @ normal)[:, None] * normal, axis=1).max()
                self.assertLessEqual(radius, 0.03 + 1e-12)
                self.assertGreater(radius, 0.024)
                observed_radii.append(radius)
        # Doubling the container cannot double the physical sampling window.
        self.assertLess(observed_radii[1] / observed_radii[0], 1.2)

    def test_nonconvex_source_evidence_keeps_its_hole_without_copying_it(self):
        points, _, anchors, _ = _surface_fixture()
        radius = np.linalg.norm(points[:, :2], axis=1)
        region = np.isclose(points[:, 2], 0) & (radius >= 0.009) & (radius <= 0.026)
        original_region = region.copy()
        prepared = prepare_anchors(points, points, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        result = infer_region(region, prepared, method="global_sim3", **_controls(points))
        self.assertTrue(result["accepted"], result["reasons"])
        np.testing.assert_array_equal(region, original_region)
        center_index = int(np.argmin(np.linalg.norm(points, axis=1)))
        self.assertFalse(region[center_index])
        # The target center is observed and usable although source evidence
        # had a hole there. This is sampling, never a filled source footprint.
        self.assertTrue(result["modes"][0]["mask"][center_index])

    def test_local_direct_and_global_share_same_metric_surface_neighborhood(self):
        points, region, anchors, centroid = _surface_fixture()
        target = _transform(points)
        config = TransferConfig(ransac_iterations=100)
        whole_prepared = prepare_anchors(points, target, anchors, anchors,
                                         mode="automatic", config=config)
        local_indices = np.flatnonzero(region)
        local_prepared = prepare_anchors(points, target, local_indices, local_indices,
                                         mode="direct", config=config)
        controls = _controls(points, target_camera=_transform([0, 0, 1]))
        whole = infer_region(region, whole_prepared, method="global_sim3", **controls)
        local = infer_region(region, local_prepared, method="direct_local", **controls)
        self.assertTrue(whole["accepted"], whole["reasons"])
        self.assertTrue(local["accepted"], local["reasons"])
        np.testing.assert_allclose(local["modes"][0]["center"], _transform(centroid), atol=2e-12)
        np.testing.assert_array_equal(local["modes"][0]["mask"], whole["modes"][0]["mask"])

    def test_missing_local_surface_cannot_be_recentered_onto_distant_support(self):
        points, region, anchors, centroid = _surface_fixture()
        visible = np.linalg.norm(points - centroid, axis=1) > 0.04
        visible[anchors] = True
        target = points[visible]
        remap = np.full(len(points), -1, dtype=int)
        remap[visible] = np.arange(visible.sum())
        prepared = prepare_anchors(points, target, anchors, remap[anchors],
                                   mode="automatic", config=TransferConfig(ransac_iterations=100))
        result = infer_region(region, prepared, method="global_sim3",
                              **_controls(points, target_pixels=_grid_pixels(points)[visible]))
        self.assertFalse(result["accepted"])
        self.assertIn("predicted_center_without_nearby_surface", result["reasons"])

    def test_close_parallel_modes_are_ambiguous_even_when_only_one_is_saved(self):
        points, region, anchors = _mirror_fixture()
        prepared = prepare_anchors(points, points, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        for budget in (1, 3):
            with self.subTest(max_modes=budget):
                controls = {**_controls(points), "max_modes": budget}
                result = infer_region(region, prepared, method="relation_only", **controls)
                self.assertFalse(result["accepted"])
                self.assertIn("surface_ambiguity", result["reasons"])
                self.assertLessEqual(len(result["modes"]), budget)
                for mode in result["modes"]:
                    # Face separation is 20 mm, less than the 30 mm radius.
                    self.assertLess(np.ptp(points[mode["mask"], 2]), 1e-12)

    def test_same_neighborhood_has_many_valid_points_without_false_ambiguity(self):
        points = _plane_grid(half_width=0.025, spacing=0.001)
        region = np.linalg.norm(points[:, :2], axis=1) <= 0.008
        anchors = fps_indices(points, 16)
        prepared = prepare_anchors(points, points, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        result = infer_region(region, prepared, method="relation_only", **_controls(points))
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertGreater(len(result["modes"][0]["sample_indices"]), 10)

    def test_disconnected_coplanar_observations_do_not_suppress_each_other(self):
        points = _plane_grid(half_width=0.012, spacing=0.001)
        pixels = _grid_pixels(points)
        observed = ~np.isclose(points[:, 0], 0)
        points, pixels = points[observed], pixels[observed]
        region = points[:, 0] < 0
        anchors = fps_indices(points, 16)
        prepared = prepare_anchors(points, points, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        result = infer_region(region, prepared, method="global_sim3",
                              **_controls(points, target_pixels=pixels))
        self.assertTrue(result["accepted"], result["reasons"])
        # Both small components fit in one 30 mm circle, but no observed
        # neighborhood crosses the missing column. Geometric circle overlap
        # alone therefore cannot establish redundant observed proposals.
        self.assertEqual(len(result["modes"]), 2)
        first, second = result["modes"]
        self.assertLess(first["center"][0], 0)
        self.assertGreater(second["center"][0], 0)
        self.assertFalse(np.any(first["mask"] & second["mask"]))

    def test_distant_same_plane_preference_ambiguity_is_reported_without_rejection(self):
        plane = _plane_grid(half_width=0.09, spacing=0.003)
        # Eight collinear anchors give equal distances at x=+-51 mm. Two
        # off-line anchors make registration rank two, but the 20% relation
        # trim removes their only evidence distinguishing those locations.
        line = np.column_stack((np.zeros(8), np.linspace(-0.13, 0.13, 8), np.zeros(8)))
        references = np.vstack((line, [[-0.14, 0.11, 0.], [0.14, -0.11, 0.]]))
        points = np.vstack((plane, references))
        pixels = np.vstack((_grid_pixels(plane),
                            np.column_stack((np.arange(10) * 8 + 1000, np.zeros(10, dtype=int)))))
        anchors = np.arange(len(plane), len(points))
        region = np.linalg.norm(points - [-0.051, 0., 0.], axis=1) <= 0.01
        prepared = prepare_anchors(points, points, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        result = infer_region(region, prepared, method="relation_only",
                              **_controls(points, target_pixels=pixels))
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertTrue(result["diagnostics"]["position_preference_ambiguous"])
        self.assertFalse(result["diagnostics"]["surface_ambiguity"])

    def test_signed_visible_face_normal_rejects_the_opposite_side(self):
        points = _plane_grid()
        region = np.linalg.norm(points[:, :2], axis=1) <= 0.012
        anchors = fps_indices(points, 16)
        prepared = prepare_anchors(points, points, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        top = infer_region(region, prepared, method="global_sim3", **_controls(points))
        underside = infer_region(region, prepared, method="global_sim3",
                                 **_controls(points, target_camera=[0, 0, -1]))
        self.assertTrue(top["accepted"], top["reasons"])
        self.assertGreater(top["modes"][0]["normal"][2], 0.99)
        self.assertFalse(underside["accepted"])
        self.assertEqual(underside["modes"], [])

    def test_anisotropic_pixel_grid_has_an_observable_planar_neighborhood(self):
        # At 20:1 physical aspect ratio, the 35 Euclidean nearest points are
        # collinear. A bounded pixel neighborhood still observes both surface
        # directions, so the complete inference path must retain this plane.
        uu, vv = np.meshgrid(np.arange(-40, 41), np.arange(-3, 4), indexing="ij")
        points = np.column_stack((uu.ravel() * 0.001,
                                  vv.ravel() * 0.020, np.zeros(uu.size)))
        region = (np.abs(points[:, 0]) <= 0.006 + 1e-12)
        region &= np.abs(points[:, 1]) <= 0.020 + 1e-12
        anchors = fps_indices(points, 16)
        prepared = prepare_anchors(points, points, anchors, anchors, mode="automatic",
                                   config=TransferConfig(ransac_iterations=100))
        result = infer_region(region, prepared, method="global_sim3", **_controls(points))
        self.assertTrue(result["accepted"], result["reasons"])
        primary = result["modes"][0]
        self.assertEqual(primary["diagnostics"]["neighborhood_rank"], 2)
        self.assertGreater(np.ptp(points[primary["mask"], 1]), 0.020)

    def test_camera_origins_are_required_finite_geometric_evidence(self):
        points, region, anchors, _ = _surface_fixture()
        prepared = prepare_anchors(points, points, anchors, anchors, mode="oracle")
        with self.assertRaises(TypeError):
            infer_region(region, prepared, method="oracle_relational", **SAMPLING)
        with self.assertRaises(ValueError):
            infer_region(region, prepared, method="oracle_relational",
                         **_controls(points, target_camera=[0, np.nan, 1]))

    def test_target_pixels_are_required_unique_integer_observations(self):
        points, region, anchors, _ = _surface_fixture()
        prepared = prepare_anchors(points, points, anchors, anchors, mode="oracle")
        controls = _controls(points)
        missing_pixels = dict(controls)
        missing_pixels.pop("target_pixels")
        with self.assertRaises(TypeError):
            infer_region(region, prepared, method="oracle_relational", **missing_pixels)
        duplicate = controls["target_pixels"].copy()
        duplicate[1] = duplicate[0]
        for bad_pixels in (controls["target_pixels"].astype(float),
                           controls["target_pixels"][:-1], duplicate):
            with self.subTest(shape=bad_pixels.shape, dtype=bad_pixels.dtype):
                with self.assertRaises(ValueError):
                    infer_region(region, prepared, method="oracle_relational",
                                 **{**controls, "target_pixels": bad_pixels})


class TestMetricNeighborhoodSampling(unittest.TestCase):
    """Check the public geometric sampler without a descriptor-specific oracle."""

    def _sample(self, points, seed, spacing=0.002, pixels=None):
        return sample_neighborhood(
            points, np.ones(len(points), bool), np.tile([0., 0., 1.], (len(points), 1)),
            np.full(len(points), spacing), seed, sampling_radius_m=0.03,
            sampling_spacing_m=0.005,
            target_pixels=_grid_pixels(points) if pixels is None else pixels)

    def test_fixed_radius_minimum_spacing_and_determinism(self):
        points = _plane_grid()
        seed = int(np.argmin(np.linalg.norm(points, axis=1)))
        first, second = self._sample(points, seed), self._sample(points, seed)
        np.testing.assert_array_equal(first["mask"], second["mask"])
        np.testing.assert_array_equal(first["sample_indices"], second["sample_indices"])
        self.assertGreater(len(first["sample_indices"]), 10)
        self.assertLessEqual(np.linalg.norm(points[first["mask"], :2], axis=1).max(), 0.03 + 1e-12)
        sampled = points[first["sample_indices"]]
        distances = np.linalg.norm(sampled[:, None] - sampled[None], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreaterEqual(distances.min(), 0.005 - 1e-12)

    def test_near_parallel_face_is_not_added_to_seed_neighborhood(self):
        plane = _plane_grid()
        points = np.vstack((plane, plane + [0, 0, 0.02]))
        seed = int(np.argmin(np.linalg.norm(plane, axis=1)))
        mode = self._sample(points, seed)
        self.assertGreater(mode["mask"][:len(plane)].sum(), 250)
        self.assertFalse(mode["mask"][len(plane):].any())

    def test_observation_gap_is_not_filled_by_a_convex_hull(self):
        points = _plane_grid(spacing=0.001)
        pixels = _grid_pixels(points)
        observed = np.abs(points[:, 0]) >= 0.004 - 1e-12
        points, pixels = points[observed], pixels[observed]
        seed = int(np.argmin(np.linalg.norm(points - [-0.005, 0, 0], axis=1)))
        mode = self._sample(points, seed, spacing=0.001, pixels=pixels)
        self.assertTrue(mode["mask"][seed])
        self.assertFalse(mode["mask"][points[:, 0] > 0].any())

    def test_anisotropic_observed_grid_keeps_both_tangent_directions(self):
        # A finite multiple of nearest-point spacing cannot represent this
        # observation topology: four nearest neighbors all lie in one row.
        # Pixel adjacency records the observed neighboring rows explicitly.
        for row_spacing in (0.008, 0.020):
            with self.subTest(row_spacing_m=row_spacing):
                uu, vv = np.meshgrid(np.arange(-40, 41), np.arange(-3, 4), indexing="ij")
                points = np.column_stack((uu.ravel() * 0.001,
                                          vv.ravel() * row_spacing, np.zeros(uu.size)))
                seed = int(np.argmin(np.linalg.norm(points, axis=1)))
                mode = self._sample(points, seed, spacing=0.0015)
                self.assertEqual(mode["diagnostics"]["neighborhood_rank"], 2)
                self.assertGreater(np.ptp(points[mode["mask"], 1]), row_spacing)
                self.assertLessEqual(np.linalg.norm(points[mode["mask"]], axis=1).max(),
                                     0.03 + 1e-12)

    def test_missing_pixel_column_cannot_be_crossed_by_short_metric_edges(self):
        points = _plane_grid(half_width=0.01, spacing=0.001)
        pixels = _grid_pixels(points)
        # The missing column is only 2 mm wide in 3-D; the removed geometric
        # graph would bridge it because its old edge radius was 3 mm.
        observed = ~np.isclose(points[:, 0], 0)
        points, pixels = points[observed], pixels[observed]
        seed = int(np.argmin(np.linalg.norm(points - [-0.002, 0, 0], axis=1)))
        mode = self._sample(points, seed, spacing=0.001, pixels=pixels)
        self.assertFalse(mode["mask"][points[:, 0] > 0].any())

    def test_pixel_translation_preserves_sample_identity(self):
        points = _plane_grid()
        pixels = _grid_pixels(points)
        seed = int(np.argmin(np.linalg.norm(points, axis=1)))
        first = self._sample(points, seed, pixels=pixels)
        moved = self._sample(points, seed, pixels=pixels + [137, 61])
        np.testing.assert_array_equal(first["mask"], moved["mask"])
        np.testing.assert_array_equal(first["sample_indices"], moved["sample_indices"])


def _pixel_observation(width=64, height=48, pixels=None):
    """Small CPU image chart with one explicit 3-D identity per visible pixel."""
    if pixels is None:
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        pixels = np.column_stack((xx.ravel(), yy.ravel()))
    pixels = np.asarray(pixels, dtype=np.int32)
    mask = np.zeros((height, width), dtype=bool)
    mask[pixels[:, 1], pixels[:, 0]] = True
    xyz = np.column_stack((pixels * 0.01, np.zeros(len(pixels))))
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.diag([1., -1., -1.])
    camera_pose[:3, 3] = [0., 0., 1.]
    return {"xyz": xyz, "pixels": pixels, "object_mask": mask,
            "T_world_camera_cv": camera_pose,
            "K": np.array([[100., 0, width / 2], [0, 100., height / 2], [0, 0, 1.]])}


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
        pixels = _grid_pixels(points)
        width, height = (pixels.max(axis=0) + 1).tolist()
        observation = _pixel_observation(width=width, height=height, pixels=pixels)
        observation["xyz"] = points
        source_truth = {
            "region_mask": region, "support_mask": np.isclose(points[:, 2], 0),
            "region_center": centroid, "canonical_xyz": points,
            "world_to_canonical": np.eye(4), "canonical_to_world": np.eye(4),
            "support_id": np.rint((points[:, 2] + 0.05) / 0.05).astype(np.int64),
            "preferred_support_id": np.array(1, dtype=np.int64),
            "region_visible_area_fraction": np.array(1.0),
            "region_center_visible": np.array(True),
        }
        target_truth = dict(source_truth)
        if altered_truth:
            target_truth["region_mask"] = np.roll(region, 50)
            target_truth["region_center"] = centroid + np.array([0.2, 0, 0])
            target_truth["preferred_support_id"] = np.array(0, dtype=np.int64)
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
            "source_crop_xyxy": [0, 0, width, height],
            "target_crop_xyxy": [0, 0, width, height],
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
                            self.assertEqual(set(before.files), set(after.files))
                            # Compare every saved mode and fitted model, not
                            # just the first center or a copied legacy mask.
                            for field in before.files:
                                np.testing.assert_array_equal(before[field], after[field])
            before = next(r for r in first_rows if r["method"] == "global_sim3")
            after = next(r for r in second_rows if r["method"] == "global_sim3")
            self.assertTrue(before["top1_preferred_support_hit"])
            self.assertFalse(after["top1_preferred_support_hit"])
            # The obsolete physical-rectangle center has changed, but the
            # independent evidence-mean reference definition has not.
            self.assertLess(before["center_error_m"], 1e-12)
            self.assertEqual(before["center_error_m"], after["center_error_m"])

    def test_malformed_correspondences_stop_the_run_instead_of_counting_as_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                self._run_fixture(temporary, malformed_pairs=True)


if __name__ == "__main__":
    unittest.main()
# END LOCAL_REGION_TRANSFER_MATHEMATICAL_REGRESSIONS
