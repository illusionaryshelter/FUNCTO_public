"""CPU-only geometry contracts for object motion fitting and transfer.

Synthetic trajectories are generated through independent joint placements.
They are numerical fixtures, not evidence of visual articulation recognition,
grasp feasibility, or success of any real manipulation task.
"""

import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

# BEGIN non-constraining motion analysis and shared boundary regressions.
from manipulation.motion import analyze_motion, fit_motion, transfer_motion
from manipulation.planning_problem import PosePath, TransferredPosePath, quaternion_rotate, _slerp
# END motion analysis and continuous boundary imports.


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


# BEGIN every legacy geometry test explicitly declares its geometric phase.
class MotionTransferTest(unittest.TestCase):
    def setUp(self):
        self.initial = transform(Rotation.from_euler("xyz", [0.6, -0.3, 0.4]).as_matrix(), [0.4, 0.2, 0.7])
        self.mapping = transform(Rotation.from_euler("xyz", [-0.5, 0.8, 0.2]).as_matrix(), [-0.2, 0.1, -0.3])

    def test_reference_conjugation_and_initial_pose_for_all_modes(self):
        for kind in ("free", "prismatic", "revolute", "translation_plane", "planar"):
            with self.subTest(kind=kind):
                demo = fixture(kind)
                result = transfer_motion(demo, self.initial, self.mapping, kind=kind,
                                         progress=np.linspace(0, 1, len(demo)))
                expected = self.mapping @ demo @ np.linalg.inv(demo[0]) @ np.linalg.inv(self.mapping) @ self.initial
                np.testing.assert_allclose(result["poses"], expected, atol=2e-11)
                np.testing.assert_array_equal(result["poses"][0], self.initial)

    def test_reachable_constrained_goals_preserve_initial_and_terminal_poses(self):
        for kind in ("prismatic", "revolute", "translation_plane", "planar"):
            with self.subTest(kind=kind):
                demo = fixture(kind)
                reference = transfer_motion(demo, self.initial, self.mapping, kind=kind,
                                            progress=np.linspace(0, 1, len(demo)))
                model = reference["model"]
                target_coordinates = model.coordinates[-1] + (0.2 if kind == "revolute" else 0.035)
                goal = model.poses(target_coordinates)[0]
                result = transfer_motion(demo, self.initial, self.mapping, kind=kind, goal_pose=goal,
                                         progress=np.linspace(0, 1, len(demo)))
                np.testing.assert_array_equal(result["poses"][0], self.initial)
                np.testing.assert_array_equal(result["poses"][-1], goal)
                np.testing.assert_allclose(result["model"].poses(result["model"].coordinates), result["poses"], atol=2e-11)

    def test_free_correction_allows_arbitrary_goal_orientation_and_translation(self):
        goal = transform(Rotation.from_euler("xyz", [1.1, 0.7, -2.6]).as_matrix(), [-0.2, 0.3, 0.1])
        demo = fixture("free")
        result = transfer_motion(demo, self.initial, self.mapping, goal_pose=goal,
                                 progress=np.linspace(0, 1, len(demo)))
        self.assertEqual(result["model"].kind, "free")
        np.testing.assert_array_equal(result["poses"][0], self.initial)
        np.testing.assert_array_equal(result["poses"][-1], goal)
        np.testing.assert_allclose(result["model"].poses(result["model"].coordinates), result["poses"], atol=2e-11)

    def test_goal_is_not_silently_projected_even_within_fitting_tolerance(self):
        demo = fixture("translation_plane")
        reference = transfer_motion(demo, self.initial, self.mapping, kind="translation_plane",
                                    progress=np.linspace(0, 1, len(demo)))
        goal = reference["poses"][-1].copy()
        goal[:3, 3] += 0.001 * reference["model"].axis
        # The noisy-observation API accepts this residual; exact goals must not.
        reference["model"].coordinates_of(goal[None])
        with self.assertRaisesRegex(ValueError, "not exactly reachable"):
            transfer_motion(demo, self.initial, self.mapping, kind="translation_plane", goal_pose=goal,
                            progress=np.linspace(0, 1, len(demo)))

    def test_wrong_goal_orientation_is_not_accepted_for_translation(self):
        demo = fixture("prismatic")
        result = transfer_motion(demo, self.initial, self.mapping, kind="prismatic",
                                 progress=np.linspace(0, 1, len(demo)))
        goal = result["poses"][-1].copy()
        goal[:3, :3] = Rotation.from_rotvec([0.3, 0, 0]).as_matrix() @ goal[:3, :3]
        with self.assertRaisesRegex(ValueError, "rotation residual"):
            transfer_motion(demo, self.initial, self.mapping, kind="prismatic", goal_pose=goal,
                            progress=np.linspace(0, 1, len(demo)))

    def test_goal_branch_does_not_erase_a_demonstrated_full_turn(self):
        demo = np.repeat(np.eye(4)[None], 101, axis=0)
        angle = np.linspace(0, 2 * np.pi, len(demo))
        demo[:, :3, :3] = Rotation.from_euler("z", angle).as_matrix()
        result = transfer_motion(demo, self.initial, kind="revolute", goal_pose=self.initial,
                                 progress=np.linspace(0, 1, len(demo)))
        self.assertAlmostEqual(abs(result["model"].coordinates[-1, 0]), 2 * np.pi, places=10)
        np.testing.assert_array_equal(result["poses"][-1], self.initial)

    def test_one_node_cannot_have_different_start_and_goal(self):
        different = self.initial.copy()
        different[0, 3] += 0.1
        with self.assertRaisesRegex(ValueError, "one-node"):
            transfer_motion(np.eye(4)[None], self.initial, goal_pose=different, progress=[0.])
        same = transfer_motion(np.eye(4)[None], self.initial, goal_pose=self.initial, progress=[0.])
        np.testing.assert_array_equal(same["poses"][0], self.initial)

    def test_transfer_rejects_invalid_reference_and_initial_transforms(self):
        invalid = np.eye(4); invalid[0, 0] = -1
        with self.assertRaises(ValueError):
            transfer_motion(fixture("free"), invalid, progress=np.linspace(0, 1, 41))
        with self.assertRaises(ValueError):
            transfer_motion(fixture("free"), self.initial, invalid, progress=np.linspace(0, 1, 41))
# END explicit sampling phase introduces no inferred physical timestamps.


# BEGIN independent continuous-boundary, event-suffix and phase-density checks.
class ContinuousMotionTransferTest(unittest.TestCase):
    def source(self):
        keys = np.array([0., .18, .63, 1.])
        rotations = Rotation.from_euler("z", [.1, .35, .9, 1.2]).as_matrix()
        poses = np.stack([transform(r, [s, .2*s*s, .1]) for s, r in zip(keys, rotations)])
        return PosePath(keys, poses, source="explicit nonuniform source times")

    def test_array_and_continuous_share_boundaries_under_nonuniform_sampling(self):
        source = self.source()
        initial = transform(Rotation.from_euler("xyz", [.2, -.3, .1]).as_matrix(), [.4, -.2, .5])
        goal = transform(Rotation.from_euler("xyz", [-.3, .1, 1.7]).as_matrix(), [1.1, .4, .2])
        path = TransferredPosePath(source, initial, [{"progress": 1., "pose": goal}], "two observed boundaries")
        query = np.unique(np.r_[source.progress, .07, .29, .81])
        dense = transfer_motion(source.sample_numpy(query), initial, goal_pose=goal, progress=query)
        sparse = transfer_motion(source.poses, initial, goal_pose=goal, progress=source.progress)
        np.testing.assert_allclose(path.sample_numpy(query), dense["poses"], atol=2e-12)
        np.testing.assert_allclose(path.sample_numpy(source.progress), sparse["poses"], atol=2e-12)
        np.testing.assert_allclose(path.sample_numpy(0.), initial, atol=2e-12)
        np.testing.assert_allclose(path.sample_numpy(1.), goal, atol=2e-12)
        self.assertGreater(np.linalg.norm(path.sample_numpy(.29)-source.sample_numpy(.29)), .1)

    def test_event_rebind_preserves_suffix_and_continuous_attachment(self):
        raw = self.source()
        # A rotating offset deliberately makes product-knot interpolation wrong.
        raw.attachment = transform(translation=[.22, -.09, .03])
        event = .41
        goal = raw.sample_numpy(event)
        initial = transform(Rotation.from_euler("y", .5).as_matrix(), [-.3, .2, .4])
        path = TransferredPosePath(raw, initial, [{"progress": event, "pose": goal}], "internal functional event")
        self.assertIn(event, path.progress)
        np.testing.assert_allclose(path.sample_numpy(event), goal, atol=2e-12)
        np.testing.assert_allclose(path.sample_numpy([.57, .83, 1.]), raw.sample_numpy([.57, .83, 1.]), atol=2e-12)
        A = transform(Rotation.from_euler("x", .3).as_matrix(), [.13, -.07, .12])
        eef = PosePath(path.progress, np.repeat(np.eye(4)[None], len(path.progress), axis=0),
                       source="same fixed attachment for all progress", parent=path, attachment=np.linalg.inv(A))
        query = np.array([.11, .29, .48, .91])
        np.testing.assert_allclose(eef.sample_numpy(query) @ A, path.sample_numpy(query), atol=2e-12)
        # A source ending at a functional event cannot acquire a constant
        # post-event correction from the earlier displaced entrance.
        np.testing.assert_allclose(eef.sample_numpy(.83) @ A, raw.sample_numpy(.83), atol=2e-12)

    def test_continuous_derivatives_match_pose_finite_differences(self):
        import torch

        raw = self.source()
        goal = transform(Rotation.from_euler("xyz", [.2, .4, -.2]).as_matrix(), [.2, -.3, .6])
        path = TransferredPosePath(raw, transform(translation=[-.2, .1, .3]),
                                   [{"progress": .76, "pose": goal}], "gradient regression")
        query = torch.tensor([.12, .42, .87], dtype=torch.float64, requires_grad=True)
        position, quaternion = path.sample(query)
        axis = torch.tensor([1., 0., 0.], dtype=torch.float64).expand(3, 3)
        value = (position+quaternion_rotate(quaternion, axis)).sum()
        derivative, = torch.autograd.grad(value, query)
        delta = 1e-6
        plus = path.sample_numpy(query.detach().numpy()+delta)
        minus = path.sample_numpy(query.detach().numpy()-delta)
        finite = ((plus[:, :3, 3]+plus[:, :3, 0]).sum(axis=1)
                  -(minus[:, :3, 3]+minus[:, :3, 0]).sum(axis=1))/(2*delta)
        np.testing.assert_allclose(derivative.detach().numpy(), finite, atol=2e-8, rtol=2e-8)

    # BEGIN small rotations obey the same exact law as ordinary rotations.
    def test_small_rotation_density_and_analytic_progress_derivative(self):
        import torch

        query = np.array([.1, .2, .4, .6, .8, .9])
        for angle in (.06, 1e-10, 0.):
            with self.subTest(angle=angle):
                paths = []
                for keys in (np.array([0., 1.]), np.array([0., .3, 1.])):
                    poses = np.repeat(np.eye(4)[None], len(keys), axis=0)
                    poses[:, :3, :3] = Rotation.from_euler("z", angle*keys).as_matrix()
                    paths.append(PosePath(keys, poses, source="same analytic angular trajectory"))
                expected = Rotation.from_euler("z", angle*query).as_matrix()
                for candidate in paths:
                    actual = candidate.sample_numpy(query)[:, :3, :3]
                    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
                    progress = torch.tensor(query, dtype=torch.float64, requires_grad=True)
                    _, quaternion = candidate.sample(progress)
                    axis = torch.tensor([1., 0., 0.], dtype=torch.float64).expand(len(query), 3)
                    direction = quaternion_rotate(quaternion, axis)
                    derivative, = torch.autograd.grad(direction.sum(), progress)
                    np.testing.assert_allclose(derivative.detach().numpy(),
                        angle*(np.cos(angle*query)-np.sin(angle*query)), atol=2e-14, rtol=0)
                # Exercise the same small-angle correction inside the actual
                # transferred path, not only its unadapted input interpolator.
                goal = transform(Rotation.from_euler("z", angle+.02).as_matrix())
                adapted = [TransferredPosePath(candidate, np.eye(4),
                    [{"progress": 1., "pose": goal}], "explicit small-angle target") for candidate in paths]
                exact = Rotation.from_euler("z", (angle+.02)*query).as_matrix()
                for candidate in adapted:
                    np.testing.assert_allclose(candidate.sample_numpy(query)[:, :3, :3], exact, atol=2e-14, rtol=0)

    def test_identity_slerp_has_finite_input_and_progress_gradients(self):
        import torch

        a = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        progress = torch.tensor([.37], dtype=torch.float64, requires_grad=True)
        value = _slerp(a, b, progress)
        weights = torch.tensor([[.2, .3, .4, .5]], dtype=torch.float64)
        gradients = torch.autograd.grad((value*weights).sum(), (a, b, progress))
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
        np.testing.assert_array_equal(value.detach().numpy(), [[1., 0., 0., 0.]])
        np.testing.assert_array_equal(gradients[-1].detach().numpy(), [0.])
    # END sample insertion cannot change a short rotation or manufacture drift.

    def test_missing_conflicting_and_invalid_boundaries_do_not_complete_transfer(self):
        raw = self.source()
        initial = raw.sample_numpy(0.)
        with self.assertRaisesRegex(ValueError, "explicit event goals"):
            TransferredPosePath(raw, initial, [], "missing terminal")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            TransferredPosePath(raw, initial, [{"progress": 0., "pose": transform(translation=[1., 0., 0.])}], "conflict")
        # An explicitly satisfied entrance event is evidence, so preserving
        # its remaining suffix must not require inventing another endpoint.
        entered = TransferredPosePath(raw, initial, [{"progress": 0., "pose": initial}], "event at actual entrance")
        np.testing.assert_allclose(entered.sample_numpy([0., .31, .79, 1.]), raw.sample_numpy([0., .31, .79, 1.]), atol=2e-12)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            TransferredPosePath(raw, initial, [{"progress": 0., "pose": initial},
                {"progress": 0., "pose": transform(translation=[1., 0., 0.])}], "conflicting duplicate entrance")
        for phase in ([0., .2, .2, 1.], [0., .2, .8, .9], [0., .3, np.nan, 1.], [0., 1.]):
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, "progress"):
                transfer_motion(raw.poses, initial, progress=phase)
        with self.assertRaises(TypeError):
            transfer_motion(raw.poses, initial)
# END density changes cannot redefine boundaries, suffixes or rigid holding.


if __name__ == "__main__":
    unittest.main()
