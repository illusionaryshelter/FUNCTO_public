"""Small analytic examples verify explicit-pair consensus without scene truth."""

# BEGIN rigid correspondence consensus property tests.
import itertools
import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.consensus import rigid_correspondence_hypotheses


def moved(points, rotation, translation):
    return rotation.apply(points) + np.asarray(translation)


def pair_ids(count):
    return [(f"source_{i}", f"current_{i}") for i in range(count)]


class RigidConsensusTest(unittest.TestCase):
    def setUp(self):
        self.points = np.array([
            [-.04, -.03, .01], [.045, -.02, .0], [-.02, .05, .02],
            [.03, .035, .04], [.015, -.025, .065],
        ])
        self.rotation = Rotation.from_euler("xyz", [.23, -.31, .49])
        self.translation = np.array([.24, -.13, .67])
        self.targets = moved(self.points, self.rotation, self.translation)

    def assert_pose(self, candidate, rotation, translation, atol=1e-10):
        matrix = np.asarray(candidate["transform"])
        np.testing.assert_allclose(matrix[:3, :3], rotation.as_matrix(), atol=atol)
        np.testing.assert_allclose(matrix[:3, 3], translation, atol=atol)
        np.testing.assert_allclose(matrix[3], [0., 0., 0., 1.], atol=atol)
        self.assertAlmostEqual(np.linalg.det(matrix[:3, :3]), 1., places=10)

    def test_minimal_planar_triangle_observes_pose_but_is_weak(self):
        points = np.array([[.0, .0, .2], [.07, .0, .2], [.0, .05, .2]])
        result = rigid_correspondence_hypotheses(
            points, moved(points, self.rotation, self.translation), pair_ids=pair_ids(3))
        self.assertEqual(result["status"], "accepted")
        candidate = result["candidates"][0]
        self.assert_pose(candidate, self.rotation, self.translation)
        self.assertTrue(candidate["weak_evidence"])
        self.assertEqual(candidate["extra_support_count"], 0)
        self.assertEqual(candidate["support_count"], 3)

    def test_good_extra_points_join_fit_and_all_seed_provenance_is_retained(self):
        result = rigid_correspondence_hypotheses(
            self.points, self.targets, pair_ids=pair_ids(5))
        self.assertEqual(result["status"], "accepted")
        candidate = result["candidates"][0]
        self.assert_pose(candidate, self.rotation, self.translation)
        self.assertEqual(candidate["support_count"], 5)
        self.assertEqual(candidate["extra_support_count"], 2)
        self.assertFalse(candidate["weak_evidence"])
        self.assertEqual(len(candidate["seeds"]), 10)
        self.assertEqual(candidate["fit_pair_ids"], candidate["inlier_pair_ids"])
        self.assertEqual(result["diagnostics"]["total_seed_count"], 10)
        json.dumps(result, allow_nan=False)

    def test_one_bad_pair_is_rejected_and_four_joint_points_recover_pose(self):
        target = self.targets.copy()
        target[4] += [.09, -.06, .11]
        result = rigid_correspondence_hypotheses(
            self.points, target, pair_ids=pair_ids(5), max_error=.001)
        self.assertEqual(result["status"], "accepted")
        candidate = result["candidates"][0]
        self.assert_pose(candidate, self.rotation, self.translation)
        self.assertEqual(candidate["support_count"], 4)
        self.assertEqual(candidate["inlier_indices"], [0, 1, 2, 3])
        self.assertGreater(candidate["residuals_m"][4], .1)

    def test_joint_row_permutation_preserves_pose_support_and_seed_identities(self):
        target = self.targets.copy()
        target[4] += [.09, -.06, .11]
        identities = pair_ids(5)
        expected = rigid_correspondence_hypotheses(
            self.points, target, pair_ids=identities, max_error=.001)
        first = expected["candidates"][0]
        expected_seeds = [s["pair_ids"] for s in first["seeds"]]
        for permutation in itertools.permutations(range(5)):
            indices = list(permutation)
            result = rigid_correspondence_hypotheses(
                self.points[indices], target[indices],
                pair_ids=[identities[i] for i in indices], max_error=.001)
            candidate = result["candidates"][0]
            self.assertEqual(result["status"], expected["status"])
            self.assertEqual(candidate["inlier_pair_ids"], first["inlier_pair_ids"])
            np.testing.assert_array_equal(candidate["transform"], first["transform"])
            self.assertEqual([s["pair_ids"] for s in candidate["seeds"]], expected_seeds)
            restored = np.empty(5)
            restored[indices] = candidate["residuals_m"]
            np.testing.assert_array_equal(restored, first["residuals_m"])
            for index, identity in zip(candidate["inlier_indices"],
                                       candidate["inlier_pair_ids"]):
                self.assertEqual(list(identities[indices[index]]), identity)

    def test_equal_support_incompatible_motions_are_both_returned(self):
        rng = np.random.default_rng(327)
        source = rng.uniform(-1., 1., (6, 3))
        other_rotation = Rotation.from_euler("xyz", [-.71, .19, -.35])
        other_translation = np.array([-.34, .51, -.29])
        target = moved(source, self.rotation, self.translation)
        target[3:] = moved(source[3:], other_rotation, other_translation)
        result = rigid_correspondence_hypotheses(
            source, target, pair_ids=pair_ids(6), max_error=1e-7)
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual([c["inlier_indices"] for c in result["candidates"]],
                         [[0, 1, 2], [3, 4, 5]])
        self.assert_pose(result["candidates"][0], self.rotation, self.translation)
        self.assert_pose(result["candidates"][1], other_rotation, other_translation)
        self.assertTrue(all(c["weak_evidence"] for c in result["candidates"]))

    def test_lower_support_consensus_remains_visible_in_diagnostics(self):
        rng = np.random.default_rng(281)
        source = rng.uniform(-1., 1., (7, 3))
        target = moved(source, self.rotation, self.translation)
        alternative = Rotation.from_euler("xyz", [-.22, .63, -.48])
        target[4:] = moved(source[4:], alternative, [-.7, .8, -.6])
        result = rigid_correspondence_hypotheses(
            source, target, pair_ids=pair_ids(7), max_error=1e-7)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["candidates"][0]["support_count"], 4)
        lower = result["diagnostics"]["lower_support_candidates"]
        self.assertEqual(len(lower), 1)
        self.assertEqual(lower[0]["inlier_indices"], [4, 5, 6])
        self.assert_pose(lower[0], alternative, [-.7, .8, -.6])

    def test_final_fit_and_threshold_membership_describe_same_points(self):
        target = self.targets.copy()
        target[:4] += [[.0003, -.0002, .0001], [-.0001, .0002, .0001],
                       [.0001, .0002, -.0003], [-.0002, .0001, .0002]]
        target[4] += [.09, -.06, .11]
        threshold = .002
        result = rigid_correspondence_hypotheses(
            self.points, target, pair_ids=pair_ids(5), max_error=threshold)
        for candidate in result["candidates"]:
            rows = candidate["fit_indices"]
            self.assertEqual(rows, candidate["inlier_indices"])
            matrix = np.asarray(candidate["transform"])
            residuals = np.linalg.norm(
                self.points @ matrix[:3, :3].T + matrix[:3, 3] - target, axis=1)
            np.testing.assert_allclose(residuals, candidate["residuals_m"], atol=1e-15)
            self.assertEqual(set(np.flatnonzero(residuals <= threshold)), set(rows))
            # Independently solve the whole returned set with SciPy rather than
            # the production Open3D estimator, verifying actual joint optimality.
            p, q = self.points[rows], target[rows]
            rotation, _ = Rotation.align_vectors(q-q.mean(0), p-p.mean(0))
            self.assert_pose(candidate, rotation, q.mean(0)-rotation.apply(p.mean(0)))

    def test_inconsistent_or_degenerate_data_does_not_fall_back(self):
        triangle = np.array([[.0, .0, .0], [.2, .0, .0], [.0, .3, .0]])
        cases = [
            (triangle, 1.7 * triangle, "seed_inconsistent"),
            (triangle, np.array([[.0, .0, .0], [.2, .0, .0], [.4, .0, .0]]),
             "collinear_seed"),
            (np.array([[.0, .0, .0], [.2, .0, .0], [.4, .0, .0]]), triangle,
             "collinear_seed"),
        ]
        for source, target, reason in cases:
            with self.subTest(reason=reason):
                result = rigid_correspondence_hypotheses(
                    source, target, pair_ids=pair_ids(3), max_error=.001)
                self.assertEqual(result["status"], "insufficient")
                self.assertEqual(result["candidates"], [])
                self.assertEqual(result["diagnostics"]["rejected_seeds"][0]["reason"],
                                 reason)

    def test_too_few_correspondences_is_explicit_without_inventing_points(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                result = rigid_correspondence_hypotheses(
                    self.points[:count], self.targets[:count], pair_ids=pair_ids(count))
                self.assertEqual(result["status"], "insufficient")
                self.assertEqual(result["diagnostics"]["total_seed_count"], 0)

    def test_repeated_samples_and_identities_cannot_increase_support(self):
        duplicate_source = self.points.copy()
        duplicate_source[4] = duplicate_source[0]
        duplicate_target = self.targets.copy()
        duplicate_target[4] = duplicate_target[0]
        same_source_id, same_target_id = pair_ids(5), pair_ids(5)
        same_source_id[4] = (same_source_id[0][0], same_source_id[4][1])
        same_target_id[4] = (same_target_id[4][0], same_target_id[0][1])
        for p, q, identities in (
                (duplicate_source, self.targets, pair_ids(5)),
                (self.points, duplicate_target, pair_ids(5)),
                (self.points, self.targets, same_source_id),
                (self.points, self.targets, same_target_id)):
            with self.subTest(identities=identities), self.assertRaises(ValueError):
                rigid_correspondence_hypotheses(p, q, pair_ids=identities)

    def test_invalid_points_thresholds_and_ids_raise(self):
        for threshold in (0., -1., np.inf, np.nan, True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                rigid_correspondence_hypotheses(
                    self.points, self.targets, pair_ids=pair_ids(5), max_error=threshold)
        for value in (np.nan, np.inf):
            p = self.points.copy()
            p[0, 0] = value
            with self.assertRaises(ValueError):
                rigid_correspondence_hypotheses(p, self.targets, pair_ids=pair_ids(5))
        for identities in (pair_ids(4), [(True, i) for i in range(5)],
                           [("", i) for i in range(5)], list(range(5))):
            with self.subTest(identities=identities), self.assertRaises(ValueError):
                rigid_correspondence_hypotheses(
                    self.points, self.targets, pair_ids=identities)


if __name__ == "__main__":
    unittest.main()
# END rigid correspondence consensus property tests.
