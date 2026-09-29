"""Analytic known-axis consensus cases; no scene or simulator truth is read."""

# BEGIN exact interval-consensus geometry and input contract tests.
import itertools
import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.consensus import axis_translation_hypotheses


def identities(count):
    return [(f"source_{index}", f"target_{index}") for index in range(count)]


def source_points(count):
    # Distinct measured points avoid accidental duplicate votes on either side.
    return np.column_stack((np.arange(count)*.13, np.arange(count)*.017,
                            np.arange(count)*.037))


def run_offsets(offsets, transverse=None, *, max_error=.01):
    source = source_points(len(offsets))
    target = source.copy()
    target[:, 0] += offsets
    if transverse is not None:
        target[:, 1] += transverse
    return axis_translation_hypotheses(
        source, target, direction=[1., 0., 0.],
        pair_ids=identities(len(source)), max_error=max_error)


class AxisTranslationConsensusTest(unittest.TestCase):
    def test_one_pair_observes_scalar_but_remains_weak(self):
        source = np.array([[.3, -.2, .7]])
        result = axis_translation_hypotheses(
            source, source+[[0., .16, 0.]], direction=[0., 70., 0.],
            pair_ids=[("vertex", 7)], max_error=.002)
        self.assertEqual(result["status"], "accepted")
        candidate = result["candidates"][0]
        self.assertAlmostEqual(candidate["offset_m"], .16)
        self.assertTrue(candidate["weak_evidence"])
        self.assertEqual(candidate["support_count"], 1)
        self.assertEqual(candidate["extra_support_count"], 0)
        transform = np.asarray(candidate["transform"])
        np.testing.assert_array_equal(transform[:3, :3], np.eye(3))
        np.testing.assert_allclose(transform[:3, 3], [0., .16, 0.], atol=1e-15)
        np.testing.assert_allclose(candidate["allowed_offset_interval_m"], [.158, .162])
        json.dumps(result, allow_nan=False)

    def test_interval_overlap_is_found_when_no_pair_seed_supports_it(self):
        # A seed at either pair centre sees only itself. The shared interval
        # [-1, +1] mm nevertheless admits both, so centre-only RANSAC misses it.
        result = run_offsets([-.009, .009])
        self.assertEqual(result["status"], "accepted")
        candidate = result["candidates"][0]
        self.assertEqual(candidate["inlier_indices"], [0, 1])
        self.assertAlmostEqual(candidate["offset_m"], 0., places=14)
        np.testing.assert_allclose(candidate["allowed_offset_interval_m"], [-.001, .001], atol=1e-15)
        self.assertFalse(candidate["weak_evidence"])

    def test_closed_intervals_include_a_single_shared_endpoint(self):
        result = run_offsets([0., .02])
        self.assertEqual(result["status"], "accepted")
        candidate = result["candidates"][0]
        self.assertEqual(candidate["support_count"], 2)
        self.assertAlmostEqual(candidate["offset_m"], .01, places=14)
        np.testing.assert_allclose(candidate["allowed_offset_interval_m"], [.01, .01], atol=1e-15)

    def test_full_euclidean_distance_rejects_transverse_error(self):
        result = run_offsets([.2, .201, .2], [0., .001, .03])
        candidate = result["candidates"][0]
        self.assertEqual(candidate["inlier_indices"], [0, 1])
        self.assertAlmostEqual(candidate["offset_m"], .2005)
        self.assertGreater(candidate["residuals_m"][2], .03)
        self.assertEqual(result["diagnostics"]["pair_evidence"][2]["status"],
                         "transverse_inconsistent")
        impossible = run_offsets([.2, .201], [.02, .03])
        self.assertEqual(impossible["status"], "insufficient")
        self.assertEqual(impossible["candidates"], [])

    def test_transverse_gate_tangent_survives_only_arithmetic_roundoff(self):
        source = np.array([[.3, -.2, .7]])
        axis = np.array([.3, -.4, .5]); axis /= np.linalg.norm(axis)
        side = np.cross(axis, [1., 0., 0.]); side /= np.linalg.norm(side)
        target = source+.16*axis+.01*side
        for rotation in (Rotation.identity(), Rotation.from_euler("xyz", [.2, -.5, .7])):
            result = axis_translation_hypotheses(
                rotation.apply(source), rotation.apply(target), direction=rotation.apply(axis),
                pair_ids=[("source", "target")])
            self.assertEqual(result["status"], "accepted")
            candidate = result["candidates"][0]
            self.assertAlmostEqual(candidate["offset_m"], .16, places=14)
            self.assertLessEqual(candidate["residuals_m"][0],
                                 .01+result["diagnostics"]["numerical_tolerance_m"])

    def test_joint_fit_uses_all_inliers_and_ignores_lower_support_motion(self):
        result = run_offsets([.201, .199, .2, -.3], [0., .001, -.001, 0.])
        self.assertEqual(result["status"], "accepted")
        candidate = result["candidates"][0]
        self.assertEqual(candidate["support_count"], 3)
        self.assertEqual(candidate["fit_indices"], [0, 1, 2])
        self.assertEqual(candidate["inlier_indices"], candidate["fit_indices"])
        self.assertAlmostEqual(candidate["offset_m"], .2, places=14)
        self.assertAlmostEqual(candidate["residuals_m"][3], .5, places=14)

    def test_all_equal_maximum_support_sets_remain_separate(self):
        result = run_offsets([.10, .101, .30, .301])
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual([row["inlier_indices"] for row in result["candidates"]],
                         [[0, 1], [2, 3]])
        np.testing.assert_allclose([row["offset_m"] for row in result["candidates"]],
                                   [.1005, .3005])

    def test_overlapping_alternative_sets_are_not_merged_by_near_offsets(self):
        result = run_offsets([0., .019, .038])
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual([row["inlier_indices"] for row in result["candidates"]],
                         [[0, 1], [1, 2]])

    def test_mean_is_clamped_to_joint_euclidean_feasibility_interval(self):
        # The first pair consumes almost all tolerance laterally. Its allowed
        # axial deviation is only 1 mm although the others allow nearly 10 mm.
        transverse = [np.sqrt(.01**2-.001**2), 0., 0.]
        result = run_offsets([0., .009, .0091], transverse)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["support_count"], 3)
        self.assertAlmostEqual(candidate["unconstrained_offset_mean_m"], .0181/3)
        self.assertAlmostEqual(candidate["offset_m"], .001, places=13)
        self.assertLess(candidate["offset_m"], candidate["unconstrained_offset_mean_m"])
        tolerance = result["diagnostics"]["numerical_tolerance_m"]
        self.assertLessEqual(max(candidate["residuals_m"]), .01+tolerance)

    def test_common_rigid_frame_change_preserves_physical_solution(self):
        source = source_points(4)
        axis = np.array([.3, -.4, .5]); axis /= np.linalg.norm(axis)
        target = source+.17*axis
        result = axis_translation_hypotheses(
            source, target, direction=axis, pair_ids=identities(4))
        rotation = Rotation.from_euler("xyz", [.2, -.5, .7])
        translation = np.array([.7, -.3, .4])
        changed = axis_translation_hypotheses(
            rotation.apply(source)+translation, rotation.apply(target)+translation,
            direction=rotation.apply(axis), pair_ids=identities(4))
        first, second = result["candidates"][0], changed["candidates"][0]
        self.assertEqual(first["inlier_pair_ids"], second["inlier_pair_ids"])
        self.assertAlmostEqual(first["offset_m"], second["offset_m"], places=14)
        common = np.eye(4); common[:3, :3] = rotation.as_matrix(); common[:3, 3] = translation
        np.testing.assert_allclose(second["transform"],
                                   common@first["transform"]@np.linalg.inv(common), atol=1e-14)
        np.testing.assert_allclose(first["allowed_offset_interval_m"],
                                   second["allowed_offset_interval_m"], atol=1e-14)

    def test_direction_sign_changes_coordinates_not_translation(self):
        source = source_points(3); target = source+np.array([.21, 0., 0.])
        forward = axis_translation_hypotheses(
            source, target, direction=[1., 0., 0.], pair_ids=identities(3))["candidates"][0]
        reverse = axis_translation_hypotheses(
            source, target, direction=[-1., 0., 0.], pair_ids=identities(3))["candidates"][0]
        self.assertAlmostEqual(forward["offset_m"], -reverse["offset_m"])
        np.testing.assert_array_equal(forward["transform"], reverse["transform"])
        np.testing.assert_allclose(reverse["allowed_offset_interval_m"],
                                   -np.asarray(forward["allowed_offset_interval_m"])[::-1])

    def test_joint_row_permutation_preserves_fit_and_explicit_pair_order(self):
        source = source_points(4); target = source.copy()
        target[:, 0] += [.1, .101, .3, .301]
        ids = [("b", 9), (1, "second"), ("a", 2), (0, "last")]
        expected = axis_translation_hypotheses(
            source, target, direction=[1., 0., 0.], pair_ids=ids)
        for order in itertools.permutations(range(4)):
            indices = list(order)
            result = axis_translation_hypotheses(
                source[indices], target[indices], direction=[1., 0., 0.],
                pair_ids=[ids[i] for i in indices])
            self.assertEqual(result["status"], expected["status"])
            for original, actual in zip(expected["candidates"], result["candidates"]):
                self.assertEqual(original["candidate_id"], actual["candidate_id"])
                self.assertEqual(original["inlier_pair_ids"], actual["inlier_pair_ids"])
                np.testing.assert_array_equal(original["transform"], actual["transform"])
                restored = np.empty(4); restored[indices] = actual["residuals_m"]
                np.testing.assert_array_equal(restored, original["residuals_m"])
                for row, pair in zip(actual["inlier_indices"], actual["inlier_pair_ids"]):
                    self.assertEqual(pair, list(ids[indices[row]]))

    def test_empty_correspondence_set_is_explicit(self):
        result = axis_translation_hypotheses(
            np.empty((0, 3)), np.empty((0, 3)), direction=[0., 0., 1.], pair_ids=[])
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["diagnostics"]["total_seed_count"], 0)

    def test_duplicate_points_or_identities_do_not_gain_votes(self):
        source = source_points(3); target = source+[.2, 0., 0.]
        duplicate_source = source.copy(); duplicate_source[2] = duplicate_source[0]
        duplicate_target = target.copy(); duplicate_target[2] = duplicate_target[0]
        for p, q, ids in (
                (duplicate_source, target, identities(3)),
                (source, duplicate_target, identities(3)),
                (source, target, [(0, 0), (0, 1), (2, 2)]),
                (source, target, [(0, 0), (1, 0), (2, 2)])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                axis_translation_hypotheses(p, q, direction=[1., 0., 0.], pair_ids=ids)

    def test_invalid_geometry_direction_gate_or_identity_raises(self):
        source = source_points(2); target = source+[.2, 0., 0.]
        for direction in ([0., 0., 0.], [1., 0.], [[1., 0., 0.]],
                          [np.nan, 0., 1.], [np.inf, 0., 1.], [1j, 0., 1.],
                          [True, 0, 0], ["1", "0", "0"]):
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                axis_translation_hypotheses(source, target, direction=direction, pair_ids=identities(2))
        for gate in (0., -1., np.inf, np.nan, True, "0.01"):
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                axis_translation_hypotheses(source, target, direction=[1., 0., 0.],
                                          pair_ids=identities(2), max_error=gate)
        for bad_source in (source[:, :2], source.astype(str), source.astype(bool),
                           source.astype(complex), np.full_like(source, np.nan)):
            with self.subTest(source=bad_source), self.assertRaises(ValueError):
                axis_translation_hypotheses(bad_source, target, direction=[1., 0., 0.],
                                          pair_ids=identities(2))
        for ids in ([(True, 0), (1, 1)], [("", 0), (1, 1)], None, identities(1)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                axis_translation_hypotheses(source, target, direction=[1., 0., 0.], pair_ids=ids)


if __name__ == "__main__":
    unittest.main()
# END exact interval-consensus geometry and input contract tests.
