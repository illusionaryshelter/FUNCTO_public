"""CPU regressions for interchangeable, task-independent Sim(3) registration.

The scalar reference below freezes the previous arithmetic and selection rule
only as a regression oracle. It is intentionally independent of the optimized
fitter and does not import saved experiments, image models, or target labels.
"""

# BEGIN SCALAR_REGISTRATION_EQUIVALENCE_AND_BACKEND_CONTRACT_TESTS
from fractions import Fraction
import math
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from transfer import registration
from transfer.registration import (
    RegistrationBackendUnavailable,
    _prosac_samples,
    register_sim3,
)
from transfer.sim3 import apply_sim3, fit_sim3


def _reference_fit(source, target, tolerance=1e-6):
    """Freeze the original two rank SVDs and scalar Umeyama arithmetic."""
    for points in (source, target):
        singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
        if singular[0] <= np.finfo(float).tiny:
            raise ValueError("Degenerate fixture")
        if np.count_nonzero(singular > tolerance * singular[0]) < 2:
            raise ValueError("Degenerate fixture")
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    covariance = y.T @ x / len(x)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    correction[-1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = (u * correction) @ vt
    variance = float(np.mean(np.sum(x * x, axis=1)))
    scale = float(singular @ correction / variance)
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        raise ValueError("Degenerate fixture")
    translation = target_mean - scale * (rotation @ source_mean)
    if not np.isfinite(translation).all():
        raise ValueError("Degenerate fixture")
    return scale, rotation, translation


def _reference_consensus(source, target, *, iterations, seed, tolerance_m, size_ratio):
    """Use the original ordered three-point draws and lexicographic winner."""
    rng = np.random.default_rng(seed)
    quality, winner = (-1, -np.inf), None
    for _ in range(iterations):
        sample = rng.choice(len(source), 3, replace=False)
        try:
            scale, rotation, translation = _reference_fit(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not 0.2 <= scale / size_ratio <= 5.0:
            continue
        residuals = np.linalg.norm(scale * (source @ rotation.T) + translation - target, axis=1)
        inliers = residuals <= tolerance_m
        count = int(inliers.sum())
        candidate = (count, -float(np.median(residuals[inliers])) if count else -np.inf)
        if candidate > quality:
            quality = candidate
            winner = (scale, rotation, translation, residuals, inliers)
    return winner


def _reference_registration(source, target, *, min_correspondences=6, **controls):
    """Append the original bounded refinement and mandatory final fit."""
    native = _reference_consensus(source, target, **controls)
    if native is None or native[-1].sum() < min_correspondences:
        return None
    keep = native[-1]
    for _ in range(3):
        transform = _reference_fit(source[keep], target[keep])
        residual = np.linalg.norm(transform[0] * (source @ transform[1].T)
                                  + transform[2] - target, axis=1)
        proposed = residual <= controls["tolerance_m"]
        if np.array_equal(proposed, keep) or proposed.sum() < min_correspondences:
            break
        keep = proposed
    transform = _reference_fit(source[keep], target[keep])
    residual = np.linalg.norm(transform[0] * (source @ transform[1].T)
                              + transform[2] - target, axis=1)
    return *transform, residual, residual <= controls["tolerance_m"]


def _fixture(count=48, *, seed=17, outliers=0):
    """Generate generic correspondences from a known proper similarity."""
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(count, 3))
    scale = 1.7
    rotation = Rotation.from_rotvec([0.21, -0.37, 0.13]).as_matrix()
    translation = np.array([0.4, -0.2, 0.7])
    target = scale * (source @ rotation.T) + translation
    if outliers:
        target[-outliers:] = rng.normal(size=(outliers, 3)) * 3
    return source, target, (scale, rotation, translation)


def _register(source, target, **kwargs):
    controls = dict(tolerance_m=0.02, size_ratio=1.7, iterations=40, seed=0)
    controls.update(kwargs)
    return register_sim3(source, target, **controls)


class TestScalarSimilarity(unittest.TestCase):
    def test_reused_centering_preserves_original_scalar_arithmetic(self):
        rng = np.random.default_rng(29)
        # Include minimal triangles and larger refit sets. Random unrelated
        # targets exercise reflection correction as well as proper matches.
        for count in (3, 6, 19):
            for trial in range(12):
                source, target = rng.normal(size=(2, count, 3))
                with self.subTest(count=count, trial=trial):
                    expected = _reference_fit(source, target)
                    actual = fit_sim3(source, target)
                    for before, after in zip(expected, actual):
                        np.testing.assert_array_equal(after, before)

    def test_known_transform_and_planar_triangle_identify_proper_sim3(self):
        source, target, expected = _fixture()
        plane = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        scale, rotation, translation = expected
        for first, second in ((source, target),
                              (plane, scale * (plane @ rotation.T) + translation)):
            with self.subTest(point_count=len(first)):
                actual = fit_sim3(first, second)
                for value, wanted in zip(actual, expected):
                    np.testing.assert_allclose(value, wanted, rtol=0, atol=2e-14)
                self.assertAlmostEqual(np.linalg.det(actual[1]), 1.0)
                np.testing.assert_allclose(apply_sim3(first, *actual), second, atol=2e-14)

    def test_coincident_collinear_and_collapsed_targets_are_rejected(self):
        source, target, _ = _fixture(count=8)
        line = np.column_stack((np.arange(8), np.zeros((8, 2))))
        for first, second in ((np.zeros_like(source), target), (source, np.zeros_like(target)),
                              (line, target), (source, line)):
            with self.subTest(source_rank=np.linalg.matrix_rank(first - first.mean(0)),
                              target_rank=np.linalg.matrix_rank(second - second.mean(0))):
                with self.assertRaises(ValueError):
                    fit_sim3(first, second)

    def test_repeated_coordinates_do_not_destroy_valid_planar_support(self):
        source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        target = 2 * source + [0.2, 0.3, 0.4]
        result = fit_sim3(source, target)
        np.testing.assert_allclose(apply_sim3(source, *result), target, atol=1e-14)

    def test_public_fitter_keeps_its_input_validation(self):
        source, target, _ = _fixture(count=6)
        invalid = [source[:, :2], source.astype(bool), np.full_like(source, np.nan),
                   np.full_like(source, np.inf), np.array([["x"] * 3] * 6)]
        for points in invalid:
            with self.subTest(dtype=points.dtype, shape=points.shape):
                with self.assertRaises(ValueError):
                    fit_sim3(points, target)
        for first, second in ((source[:2], target[:2]), (source, target[:5])):
            with self.assertRaises(ValueError):
                fit_sim3(first, second)


class TestRansacEquivalence(unittest.TestCase):
    def test_noisy_outlier_consensus_matches_original_scalar_reference(self):
        source, target, _ = _fixture(count=64, outliers=23)
        target[:41] += np.random.default_rng(9).normal(scale=0.002, size=(41, 3))
        controls = dict(iterations=73, seed=6, tolerance_m=0.02, size_ratio=1.7)
        expected = _reference_registration(source, target, **controls)
        actual = _register(source, target, **controls)
        self.assertIsNotNone(expected)
        self.assertIsNone(actual.reason)
        for before, after in zip(expected, (actual.scale, actual.rotation, actual.translation,
                                             actual.residuals, actual.inliers)):
            np.testing.assert_array_equal(after, before)

    def test_one_iteration_preserves_the_exact_random_three_point_draw(self):
        source, target, _ = _fixture(count=8)
        expected = _reference_registration(source, target, iterations=1, seed=4,
                                           tolerance_m=0.02, size_ratio=1.7)
        actual = _register(source, target, iterations=1, seed=4)
        self.assertIsNone(actual.reason)
        for before, after in zip(expected, (actual.scale, actual.rotation, actual.translation,
                                             actual.residuals, actual.inliers)):
            np.testing.assert_array_equal(after, before)

    def test_every_sample_uses_the_original_rng_sequence_and_order(self):
        source, target, _ = _fixture(count=13)
        seen = []
        original = registration._fit_sim3_validated

        def record(first, second, tolerance):
            if len(first) == 3:
                seen.append(first.copy())
            return original(first, second, tolerance)

        with mock.patch.object(registration, "_fit_sim3_validated", side_effect=record):
            _register(source, target, iterations=7, seed=23)
        rng = np.random.default_rng(23)
        self.assertEqual(len(seen), 7)
        for sample in seen:
            np.testing.assert_array_equal(sample, source[rng.choice(len(source), 3, replace=False)])

    def test_equal_quality_keeps_the_first_hypothesis(self):
        # Two disjoint six-point consensuses have equal count and zero median.
        source, _, _ = _fixture(count=12)
        target = source.copy()
        target[6:] += [10., 0., 0.]
        models = [(1., np.eye(3), np.zeros(3)), (1., np.eye(3), np.array([10., 0., 0.]))]
        # The final consensus is fitted twice after the two sampled models.
        with mock.patch.object(registration, "_fit_sim3_validated",
                               side_effect=models + [models[0], models[0]]):
            result = _register(source, target, size_ratio=1., iterations=2, tolerance_m=0.1)
        self.assertIsNone(result.reason)
        np.testing.assert_array_equal(result.translation, np.zeros(3))
        np.testing.assert_array_equal(result.inliers, np.arange(12) < 6)

    def test_residual_threshold_is_inclusive_at_adjacent_floats(self):
        source = np.column_stack((np.zeros(6), np.arange(6), np.arange(6) ** 2)).astype(float)
        target = source.copy()
        target[:, 0] = [0., 0., 0., np.nextafter(1., 0.), 1., np.nextafter(1., np.inf)]
        with mock.patch.object(registration, "_fit_sim3_validated",
                               return_value=(1., np.eye(3), np.zeros(3))):
            result = _register(source, target, size_ratio=1., iterations=1,
                               tolerance_m=1., min_correspondences=3)
        self.assertIsNone(result.reason)
        np.testing.assert_array_equal(result.inliers, [True, True, True, True, True, False])

    def test_scale_gate_preserves_both_boundaries_and_adjacent_floats(self):
        source, _, _ = _fixture(count=6)
        cases = [(np.nextafter(0.2, 0.), False), (0.2, True),
                 (5., True), (np.nextafter(5., np.inf), False)]
        for scale, admitted in cases:
            with self.subTest(scale=scale):
                with mock.patch.object(registration, "_fit_sim3_validated",
                                       return_value=(scale, np.eye(3), np.zeros(3))):
                    result = _register(source, scale * source, size_ratio=1., iterations=1)
                if admitted:
                    self.assertIsNone(result.reason)
                    self.assertTrue(result.inliers.all())
                else:
                    self.assertIsNotNone(result.reason)
                    self.assertFalse(result.inliers.any())

    def test_degenerate_and_insufficient_correspondences_return_rejection(self):
        line = np.column_stack((np.arange(8), np.zeros((8, 2))))
        source, target, _ = _fixture(count=5)
        for first, second in ((line, line.copy()), (np.zeros((8, 3)), np.zeros((8, 3))),
                              (source, target)):
            with self.subTest(point_count=len(first), degenerate=np.ptp(first[:, 1]) == 0):
                result = _register(first, second, iterations=3)
                self.assertIsNotNone(result.reason)
                self.assertFalse(result.inliers.any())


class TestAlternativeBackendContracts(unittest.TestCase):
    def test_prosac_stably_ranks_scores_without_reordering_input_arrays(self):
        scores = np.array([0.1, 0.9, 0.8, 0.9, 0.5, 0.4])
        original = scores.copy()
        samples = list(_prosac_samples(scores, 20, np.random.default_rng(2)))
        self.assertEqual(len(samples), 20)
        self.assertEqual(set(samples[0]), {1, 3, 2})
        self.assertIn(4, samples[1])
        self.assertTrue(set(samples[1]).issubset({1, 3, 2, 4}))
        np.testing.assert_array_equal(scores, original)
        tied = list(_prosac_samples(np.ones(7), 10, np.random.default_rng(2)))
        self.assertEqual(set(tied[0]), {0, 1, 2})

    def test_prosac_growth_matches_combination_schedule_and_forced_newest_rank(self):
        # The second budget has an integer growth increment of six. A float
        # recurrence can represent it just above six and delay growth by one
        # trial, so the independent oracle uses exact rational arithmetic.
        for count, budget in ((9, 64), (7, 21)):
            samples = list(_prosac_samples(np.arange(count, 0, -1.), budget,
                                           np.random.default_rng(11)))
            prefix, boundary = 3, 1
            expected_duration = Fraction(budget * math.comb(prefix, 3), math.comb(count, 3))
            for iteration, sample in enumerate(samples, start=1):
                if iteration > boundary and prefix < count:
                    prefix += 1
                    grown = Fraction(budget * math.comb(prefix, 3), math.comb(count, 3))
                    boundary += math.ceil(grown - expected_duration)
                    expected_duration = grown
                with self.subTest(count=count, budget=budget, iteration=iteration, prefix=prefix):
                    self.assertEqual(len(set(sample)), 3)
                    self.assertTrue(all(0 <= index < prefix for index in sample))
                    if iteration <= boundary:
                        self.assertIn(prefix - 1, sample)

    def test_prosac_local_optimization_recovers_generic_ranked_inliers(self):
        source, target, expected = _fixture(count=40, outliers=15)
        scores = np.r_[np.linspace(1., 0.6, 25), np.linspace(0.4, 0., 15)]
        result = _register(source, target, algorithm="prosac_lo", scores=scores, iterations=16)
        self.assertIsNone(result.reason)
        self.assertGreaterEqual(int(result.inliers.sum()), 25)
        for actual, wanted in zip((result.scale, result.rotation, result.translation), expected):
            np.testing.assert_allclose(actual, wanted, atol=2e-13, rtol=0)

    def test_missing_teaser_is_explicit_and_cannot_call_scalar_fallback(self):
        source, target, _ = _fixture(count=6)
        with mock.patch.dict(sys.modules, {"teaserpp_python": None}):
            with mock.patch.object(registration, "_fit_sim3_validated",
                                   side_effect=AssertionError("Unexpected scalar fallback")):
                with self.assertRaises(RegistrationBackendUnavailable):
                    _register(source, target, algorithm="teaserpp")

    def test_teaser_adapter_rejects_insufficient_clique_before_reading_a_solution(self):
        source, target, _ = _fixture(count=6)

        class InsufficientSolver:
            Params = SimpleNamespace
            ROTATION_ESTIMATION_ALGORITHM = SimpleNamespace(GNC_TLS="native_gnc_tls")

            def __init__(self, params):
                self.params = params

            def solve(self, first, second):
                pass

            def getInlierMaxClique(self):
                return [0, 1]

            def getSolution(self):
                raise AssertionError("An insufficient native clique is not a valid Sim3")

        # This is an adapter contract test, not execution of the TEASER++
        # algorithm and never a source of measured algorithm performance.
        native = SimpleNamespace(RobustRegistrationSolver=InsufficientSolver)
        with mock.patch.dict(sys.modules, {"teaserpp_python": native}):
            with mock.patch.object(registration, "_fit_sim3_validated",
                                   side_effect=AssertionError("Unexpected scalar fallback")):
                result = _register(source, target, algorithm="teaserpp")
        self.assertEqual(result.reason, "degenerate_anchors")
        self.assertFalse(result.inliers.any())

    def test_teaser_adapter_preserves_native_scale_and_original_correspondence_order(self):
        source, target, expected = _fixture(count=8)
        captured = {}

        class KnownSolver:
            Params = SimpleNamespace
            ROTATION_ESTIMATION_ALGORITHM = SimpleNamespace(GNC_TLS="native_gnc_tls")

            def __init__(self, params):
                captured["params"] = params

            def solve(self, first, second):
                captured["source"], captured["target"] = first.copy(), second.copy()

            def getInlierMaxClique(self):
                return list(range(len(source)))

            def getSolution(self):
                return SimpleNamespace(scale=expected[0], rotation=expected[1],
                                       translation=expected[2])

        native = SimpleNamespace(RobustRegistrationSolver=KnownSolver)
        with mock.patch.dict(sys.modules, {"teaserpp_python": native}):
            result = _register(source, target, algorithm="teaserpp")
        self.assertIsNone(result.reason)
        np.testing.assert_array_equal(captured["source"], source.T)
        np.testing.assert_array_equal(captured["target"], target.T)
        self.assertEqual(captured["params"].noise_bound, 0.02)
        self.assertTrue(captured["params"].estimate_scaling)
        self.assertEqual(result.diagnostics["native_solution"]["scale"], expected[0])
        self.assertTrue(result.inliers.all())
        np.testing.assert_allclose(apply_sim3(source, result.scale, result.rotation,
                                             result.translation), target, atol=2e-14)

    def test_invalid_backend_controls_and_prosac_scores_are_rejected(self):
        source, target, _ = _fixture(count=8)
        invalid = [dict(algorithm="unknown"), dict(iterations=0), dict(iterations=True),
                   dict(seed=-1), dict(tolerance_m=0.), dict(tolerance_m=np.nan),
                   dict(size_ratio=0.), dict(size_ratio=np.inf)]
        for controls in invalid:
            with self.subTest(controls=controls):
                with self.assertRaises(ValueError):
                    _register(source, target, **controls)
        for scores in (None, np.ones(7), np.ones((8, 1)), np.full(8, np.nan)):
            with self.subTest(scores=None if scores is None else scores.shape):
                with self.assertRaises(ValueError):
                    _register(source, target, algorithm="prosac_lo", scores=scores)


if __name__ == "__main__":
    unittest.main()
# END SCALAR_REGISTRATION_EQUIVALENCE_AND_BACKEND_CONTRACT_TESTS
