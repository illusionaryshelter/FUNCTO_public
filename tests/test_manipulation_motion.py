"""CPU-only geometry contracts for object motion fitting and transfer.

Synthetic trajectories are generated through independent joint placements.
They are numerical fixtures, not evidence of visual articulation recognition,
grasp feasibility, or success of any real manipulation task.
"""

import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

# BEGIN non-constraining motion analysis regressions.
from manipulation.motion import analyze_motion, fit_motion, transfer_motion
# END motion analysis import.


def transform(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def fixture(kind, seed=3, count=41):
    """Compose W_T_joint * joint(q) * joint_T_body at random placements."""
    rng = np.random.default_rng(seed)
    placement = transform(Rotation.random(random_state=rng).as_matrix(), rng.uniform(-0.4, 0.4, 3))
    body = transform(Rotation.random(random_state=rng).as_matrix(), [0.24, -0.12, 0.08])
    phase = np.linspace(0, 1, count)
    joint = np.repeat(np.eye(4)[None], count, axis=0)
    if kind == "prismatic":
        joint[:, 2, 3] = 0.35 * phase
    elif kind == "revolute":
        joint[:, :3, :3] = Rotation.from_euler("z", 2.1 * phase).as_matrix()
    elif kind in ("planar", "translation_plane"):
        joint[:, 0, 3] = 0.3 * phase
        joint[:, 1, 3] = 0.12 * np.sin(4 * phase)
        if kind == "planar":
            joint[:, :3, :3] = Rotation.from_euler("z", 1.3 * phase).as_matrix()
    elif kind == "free":
        joint[:, :3, 3] = np.column_stack(
            (0.2 * phase, 0.12 * np.sin(4 * phase), 0.1 * np.sin(7 * phase))
        )
        angles = np.column_stack((0.6 * np.sin(2 * phase), 0.4 * np.cos(1.3 * phase), 2 * phase))
        joint[:, :3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    else:
        raise AssertionError(kind)
    return placement @ joint @ body


class MotionFittingTest(unittest.TestCase):
    # BEGIN observation analysis must preserve ambiguity and reject invalid data.
    def test_candidates_retain_full_planar_process_without_selecting_a_task(self):
        observed = fixture("planar")
        report = analyze_motion(observed)
        self.assertEqual(report["status"], "observed_compatible")
        self.assertEqual(report["preferred_description"], "planar")
        self.assertEqual(report["compatible_candidates"], ["planar", "free"])
        candidate = report["candidates"]["planar"]
        self.assertEqual(np.asarray(candidate["coordinates"]).shape, (len(observed), 3))
        self.assertEqual(len(candidate["diagnostics"]["position_residuals_m"]), len(observed))
        self.assertLess(candidate["diagnostics"]["max_rotation_residual_rad"], 1e-12)
        # The report is serializable without mutating the input sequence.
        json.dumps(report, allow_nan=False)
        np.testing.assert_array_equal(observed, fixture("planar"))

    def test_static_candidate_report_does_not_assert_physical_free_motion(self):
        observed = np.repeat(transform(translation=[.1, .2, .3])[None], 5, axis=0)
        report = analyze_motion(observed)
        self.assertEqual(report["status"], "insufficient_excitation")
        self.assertIsNone(report["preferred_description"])
        self.assertEqual(report["compatible_candidates"], ["free"])
        self.assertEqual(len(report["candidates"]["free"]["coordinates"]), len(observed))

    def test_candidate_analysis_cannot_fallback_from_invalid_poses(self):
        observed = fixture("free")
        observed[8, :3, 0] *= -1
        with self.assertRaisesRegex(ValueError, "proper"):
            analyze_motion(observed)
    # END observation analysis regressions.

    def test_five_models_reconstruct_arbitrary_placements(self):
        dimensions = {"free": 6, "planar": 3, "translation_plane": 2, "revolute": 1, "prismatic": 1}
        for kind, dimension in dimensions.items():
            for seed in (0, 9, 31):
                with self.subTest(kind=kind, seed=seed):
                    observed = fixture(kind, seed)
                    model = fit_motion(observed, kind)
                    self.assertEqual(model.coordinates.shape, (len(observed), dimension))
                    np.testing.assert_allclose(model.poses(model.coordinates), observed, atol=2e-12)
                    np.testing.assert_allclose(model.coordinates_of(observed), model.coordinates, atol=2e-12)
                    self.assertEqual(json.loads(json.dumps(model.to_dict()))["kind"], kind)

    def test_auto_selects_smallest_identifiable_description(self):
        for kind in ("prismatic", "revolute", "translation_plane", "planar", "free"):
            with self.subTest(kind=kind):
                model = fit_motion(fixture(kind))
                self.assertEqual(model.kind, kind)
                selection = model.diagnostics["selection"]
                self.assertIn("free", selection["compatible_candidates"])
                self.assertEqual(selection["ambiguous"], kind != "free")

    def test_explicit_free_does_not_implicitly_constrain_a_line(self):
        observed = fixture("prismatic")
        model = fit_motion(observed, "free")
        self.assertEqual(model.kind, "free")
        arbitrary_goal = observed[-1].copy()
        arbitrary_goal[:3, 3] += [0.2, -0.1, 0.3]
        arbitrary_goal[:3, :3] = Rotation.from_euler("xyz", [1, 0.4, -0.7]).as_matrix()
        coordinates = model.coordinates_of(arbitrary_goal[None])
        np.testing.assert_allclose(model.poses(coordinates)[0], arbitrary_goal, atol=2e-12)

    def test_static_is_ambiguous_but_explicit_free_is_valid(self):
        static = np.repeat(np.eye(4)[None], 8, axis=0)
        with self.assertRaisesRegex(ValueError, "Static|excited"):
            fit_motion(static)
        for kind in ("prismatic", "revolute", "translation_plane", "planar"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Insufficient"):
                fit_motion(static, kind)
        np.testing.assert_allclose(fit_motion(static, "free").poses(np.zeros((8, 6))), static)

    def test_line_does_not_determine_a_unique_plane(self):
        for kind in ("translation_plane", "planar"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "two-dimensional"):
                fit_motion(fixture("prismatic"), kind)

    def test_position_plane_alone_does_not_imply_planar_pose_motion(self):
        phase = np.linspace(0, 1.2, 31)
        observed = np.repeat(np.eye(4)[None], len(phase), axis=0)
        observed[:, :3, 3] = np.column_stack((0.3 * np.cos(phase), 0.3 * np.sin(phase), np.zeros_like(phase)))
        observed[:, :3, :3] = Rotation.from_euler("x", phase).as_matrix()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            fit_motion(observed, "planar")
        self.assertEqual(fit_motion(observed).kind, "free")

    def test_translation_models_require_constant_orientation(self):
        for kind in ("prismatic", "translation_plane"):
            observed = fixture(kind)
            observed[-1, :3, :3] = Rotation.from_rotvec([0.2, 0, 0]).as_matrix() @ observed[-1, :3, :3]
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "rotation residual"):
                fit_motion(observed, kind)

    def test_revolute_rejects_axial_screw_translation(self):
        phase = np.linspace(0, 1.2, 31)
        observed = np.repeat(np.eye(4)[None], len(phase), axis=0)
        observed[:, :3, :3] = Rotation.from_euler("z", phase).as_matrix()
        observed[:, 2, 3] = 0.1 * phase
        with self.assertRaisesRegex(ValueError, "position residual"):
            fit_motion(observed, "revolute")

    def test_revolute_observable_from_rotation_when_origin_is_on_axis(self):
        observed = np.repeat(np.eye(4)[None], 31, axis=0)
        observed[:, :3, :3] = Rotation.from_euler("y", np.linspace(0, 1.2, 31)).as_matrix()
        model = fit_motion(observed, "revolute")
        np.testing.assert_allclose(model.poses(model.coordinates), observed, atol=2e-12)

    def test_revolute_angle_lift_preserves_more_than_a_full_turn(self):
        rng = np.random.default_rng(17)
        reference = transform(Rotation.random(random_state=rng).as_matrix(), [0.2, -0.1, 0.3])
        joint = np.repeat(np.eye(4)[None], 101, axis=0)
        joint[:, :3, :3] = Rotation.from_euler("z", np.linspace(0, 2.5 * np.pi, 101)).as_matrix()
        observed = reference @ joint @ transform(translation=[0.3, 0.1, 0.02])
        model = fit_motion(observed, "revolute")
        self.assertAlmostEqual(abs(model.coordinates[-1, 0]), 2.5 * np.pi, places=11)
        self.assertTrue(np.all(np.diff(model.coordinates[:, 0]) * model.coordinates[-1, 0] > 0))
        np.testing.assert_allclose(model.poses(model.coordinates), observed, atol=2e-12)

    def test_half_turn_between_observations_has_ambiguous_branch(self):
        observed = np.repeat(np.eye(4)[None], 2, axis=0)
        observed[1, :3, :3] = Rotation.from_euler("z", np.pi).as_matrix()
        with self.assertRaisesRegex(ValueError, "branch is ambiguous"):
            fit_motion(observed, "revolute")

    def test_revolute_coordinate_interpolation_follows_an_arc(self):
        observed = fixture("revolute")
        model = fit_motion(observed, "revolute")
        fractions = np.linspace(0, 1, 100)
        coordinates = fractions[:, None] * model.coordinates[-1]
        path = model.poses(coordinates)[:, :3, 3]
        relative = path - model.axis_point
        axial = relative @ model.axis
        radial = relative - axial[:, None] * model.axis
        np.testing.assert_allclose(axial, axial[0], atol=2e-12)
        np.testing.assert_allclose(np.linalg.norm(radial, axis=1), np.linalg.norm(radial[0]), atol=2e-12)
        midpoint = model.poses(model.coordinates[-1] / 2)[0, :3, 3]
        chord = (observed[0, :3, 3] + observed[-1, :3, 3]) / 2
        self.assertGreater(np.linalg.norm(midpoint - chord), 0.01)

    def test_common_world_transform_preserves_physical_fitted_motion(self):
        change = transform(Rotation.from_euler("xyz", [0.8, -0.4, 1.2]).as_matrix(), [0.4, -0.2, 0.1])
        for kind in ("prismatic", "revolute", "translation_plane", "planar"):
            with self.subTest(kind=kind):
                original = fit_motion(fixture(kind), kind)
                moved = fit_motion(change @ fixture(kind), kind)
                np.testing.assert_allclose(moved.coordinates, original.coordinates, atol=2e-11)
                np.testing.assert_allclose(moved.poses(moved.coordinates), change @ original.poses(original.coordinates), atol=2e-11)

    def test_invalid_se3_and_coordinates_are_rejected(self):
        examples = []
        bad = fixture("free"); bad[0, 0, 0] = np.nan; examples.append(bad)
        bad = fixture("free"); bad[0, :3, 0] *= -1; examples.append(bad)
        bad = fixture("free"); bad[0, 3, 0] = 0.1; examples.append(bad)
        bad = fixture("free"); bad[0, :3, :3] *= 1.1; examples.append(bad)
        for bad in examples:
            with self.subTest(bad=bad[0].tolist()), self.assertRaises(ValueError):
                fit_motion(bad)
        model = fit_motion(fixture("free"), "free")
        with self.assertRaises(ValueError):
            model.poses(np.full((1, 6), np.inf))
        with self.assertRaises(ValueError):
            model.poses(np.zeros((1, 2)))
        for kwargs in ({"kind": "wrong"}, {"position_tolerance": 0}, {"rotation_tolerance": np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_motion(fixture("free"), **kwargs)


class MotionTransferTest(unittest.TestCase):
    def setUp(self):
        self.initial = transform(Rotation.from_euler("xyz", [0.6, -0.3, 0.4]).as_matrix(), [0.4, 0.2, 0.7])
        self.mapping = transform(Rotation.from_euler("xyz", [-0.5, 0.8, 0.2]).as_matrix(), [-0.2, 0.1, -0.3])

    def test_reference_conjugation_and_initial_pose_for_all_modes(self):
        for kind in ("free", "prismatic", "revolute", "translation_plane", "planar"):
            with self.subTest(kind=kind):
                demo = fixture(kind)
                result = transfer_motion(demo, self.initial, self.mapping, kind=kind)
                expected = self.mapping @ demo @ np.linalg.inv(demo[0]) @ np.linalg.inv(self.mapping) @ self.initial
                np.testing.assert_allclose(result["poses"], expected, atol=2e-11)
                np.testing.assert_array_equal(result["poses"][0], self.initial)

    def test_reachable_constrained_goals_preserve_initial_and_terminal_poses(self):
        for kind in ("prismatic", "revolute", "translation_plane", "planar"):
            with self.subTest(kind=kind):
                demo = fixture(kind)
                reference = transfer_motion(demo, self.initial, self.mapping, kind=kind)
                model = reference["model"]
                target_coordinates = model.coordinates[-1] + (0.2 if kind == "revolute" else 0.035)
                goal = model.poses(target_coordinates)[0]
                result = transfer_motion(demo, self.initial, self.mapping, kind=kind, goal_pose=goal)
                np.testing.assert_array_equal(result["poses"][0], self.initial)
                np.testing.assert_array_equal(result["poses"][-1], goal)
                np.testing.assert_allclose(result["model"].poses(result["model"].coordinates), result["poses"], atol=2e-11)

    def test_free_correction_allows_arbitrary_goal_orientation_and_translation(self):
        goal = transform(Rotation.from_euler("xyz", [1.1, 0.7, -2.6]).as_matrix(), [-0.2, 0.3, 0.1])
        result = transfer_motion(fixture("free"), self.initial, self.mapping, goal_pose=goal)
        self.assertEqual(result["model"].kind, "free")
        np.testing.assert_array_equal(result["poses"][0], self.initial)
        np.testing.assert_array_equal(result["poses"][-1], goal)
        np.testing.assert_allclose(result["model"].poses(result["model"].coordinates), result["poses"], atol=2e-11)

    def test_goal_is_not_silently_projected_even_within_fitting_tolerance(self):
        demo = fixture("translation_plane")
        reference = transfer_motion(demo, self.initial, self.mapping, kind="translation_plane")
        goal = reference["poses"][-1].copy()
        goal[:3, 3] += 0.001 * reference["model"].axis
        # The noisy-observation API accepts this residual; exact goals must not.
        reference["model"].coordinates_of(goal[None])
        with self.assertRaisesRegex(ValueError, "not exactly reachable"):
            transfer_motion(demo, self.initial, self.mapping, kind="translation_plane", goal_pose=goal)

    def test_wrong_goal_orientation_is_not_accepted_for_translation(self):
        demo = fixture("prismatic")
        result = transfer_motion(demo, self.initial, self.mapping, kind="prismatic")
        goal = result["poses"][-1].copy()
        goal[:3, :3] = Rotation.from_rotvec([0.3, 0, 0]).as_matrix() @ goal[:3, :3]
        with self.assertRaisesRegex(ValueError, "rotation residual"):
            transfer_motion(demo, self.initial, self.mapping, kind="prismatic", goal_pose=goal)

    def test_goal_branch_does_not_erase_a_demonstrated_full_turn(self):
        demo = np.repeat(np.eye(4)[None], 101, axis=0)
        angle = np.linspace(0, 2 * np.pi, len(demo))
        demo[:, :3, :3] = Rotation.from_euler("z", angle).as_matrix()
        result = transfer_motion(demo, self.initial, kind="revolute", goal_pose=self.initial)
        self.assertAlmostEqual(abs(result["model"].coordinates[-1, 0]), 2 * np.pi, places=10)
        np.testing.assert_array_equal(result["poses"][-1], self.initial)

    def test_one_node_cannot_have_different_start_and_goal(self):
        different = self.initial.copy()
        different[0, 3] += 0.1
        with self.assertRaisesRegex(ValueError, "one-node"):
            transfer_motion(np.eye(4)[None], self.initial, goal_pose=different)
        same = transfer_motion(np.eye(4)[None], self.initial, goal_pose=self.initial)
        np.testing.assert_array_equal(same["poses"][0], self.initial)

    def test_transfer_rejects_invalid_reference_and_initial_transforms(self):
        invalid = np.eye(4); invalid[0, 0] = -1
        with self.assertRaises(ValueError):
            transfer_motion(fixture("free"), invalid)
        with self.assertRaises(ValueError):
            transfer_motion(fixture("free"), self.initial, invalid)


if __name__ == "__main__":
    unittest.main()
