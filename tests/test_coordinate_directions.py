"""CPU contracts for evidenced native directions and original-entry goals."""

# BEGIN actual-time coordinate direction and target-origin regression coverage.
from dataclasses import replace
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from manipulation.motion import MotionModel
from manipulation.planning_problem import (CoordinateDirectionRequirement,
    CoordinateEventRequirement, MechanismRequirement, MotionPhase)
from manipulation.program import ground_program
from tests.test_manipulation_planning_problem import (components, path, pose, problem,
    prismatic_input, revolute_input, compile_inputs)


def direction(kind, *, start=0., end=1., sign=1, minimum=0., tolerance=0.):
    return CoordinateDirectionRequirement(start, end, 0, kind, sign, minimum,
                                          tolerance, "explicit task direction")


def mechanism(directions=(), *, kind="prismatic", start=0., initial=0.):
    basis = np.eye(3) if kind == "revolute" else Rotation.from_euler("y", np.pi/2).as_matrix()
    model = MotionModel(kind, np.zeros((1, 1)), np.eye(4), basis, np.zeros(3),
                        1e-6, 1e-6, {"source": "actual analytic mechanism"})
    return MechanismRequirement(start, 1., model, np.eye(4), None, [initial],
                                "actual current mechanism", tuple(directions))


def evaluate(requirement, coordinates, progress=None, times=None, **kwargs):
    coordinates = np.asarray(coordinates, float)
    progress = np.linspace(requirement.start, 1., len(coordinates)) if progress is None else progress
    times = np.linspace(0., 1., len(coordinates)) if times is None else times
    p, q = components(requirement.model.poses(coordinates[:, None])[None])
    return requirement.direction_residuals(p, q, torch.tensor(np.asarray(progress)[None], dtype=torch.float64),
        trajectory_time=torch.tensor(np.asarray(times)[None], dtype=torch.float64), **kwargs)


class CoordinateDirectionTest(unittest.TestCase):
    def test_reference_direction_never_creates_a_requirement(self):
        self.assertEqual(mechanism().directions, ())
        self.assertEqual(evaluate(mechanism(), [0., -.2, .3]), {})

    def test_net_change_allows_reverse_segment_but_monotone_rejects_it(self):
        net = mechanism([direction("net", minimum=.4)])
        self.assertEqual(float(next(iter(evaluate(net, [0., .4, .2, .5]).values())).max()), 0.)
        monotone = mechanism([direction("monotone")])
        self.assertAlmostEqual(float(next(iter(evaluate(monotone, [0., .4, .2, .5]).values())).max()), .2)
        reverse = mechanism([direction("monotone", sign=-1)])
        self.assertEqual(float(next(iter(evaluate(reverse, [0., -.2, -.4]).values())).max()), 0.)

    def test_net_original_entry_origin_survives_actual_suffix_start(self):
        original = mechanism([direction("net", minimum=.5)], initial=.2)
        retained = original.directions[0]
        self.assertEqual(retained.origin_coordinate, .2)
        fresh = mechanism([retained], start=.4, initial=.5)
        values = evaluate(fresh, [.5, .7], progress=[.4, 1.])
        self.assertLessEqual(float(next(iter(values.values())).max()), 1e-15)
        wrong = evaluate(fresh, [.5, .6], progress=[.4, 1.])
        self.assertAlmostEqual(float(next(iter(wrong.values())).max()), .1)

    def test_middle_net_scope_requires_its_actual_origin_history(self):
        scoped = mechanism([direction("net", start=.2, minimum=.3)])
        self.assertIsNone(scoped.directions[0].origin_coordinate)
        missing = evaluate(scoped, [.3, .7], progress=[.4, 1.])
        self.assertTrue(torch.isinf(next(iter(missing.values()))).any())
        complete = evaluate(scoped, [.3, .7], progress=[.2, 1.])
        self.assertEqual(float(next(iter(complete.values())).max()), 0.)

    def test_actual_time_orders_winding_on_a_progress_plateau(self):
        # Arrival, intermediate plateau queries and departure arrive in mixed
        # array order. Actual time, rather than stable progress order, owns
        # the two-turn lift and the corresponding absolute terminal goal.
        item = mechanism(kind="revolute")
        angles = np.linspace(0., 4*np.pi, 17)
        order = np.array([0, 8, 2, 11, 1, 6, 10, 3, 4, 14, 7, 5, 13, 9, 12, 15, 16])
        p, q = components(item.model.poses(angles[order, None])[None])
        s = np.full(17, .5); s[0], s[-1] = 0., 1.
        progress = torch.tensor(s[order][None], dtype=torch.float64)
        times = torch.tensor((np.arange(17)/16)[order][None], dtype=torch.float64)
        recovered = item.coordinates(p, q, progress, trajectory_time=times)
        np.testing.assert_allclose(recovered.numpy()[0, :, 0], angles[order], atol=1e-12)
        goal = CoordinateEventRequirement(1., item, 4*np.pi, 1e-8, "two turns")
        self.assertLess(float(goal.error(p, q, progress, trajectory_time=times)[0, -1]), 1e-12)

    def test_monotone_checks_motion_while_progress_is_constant(self):
        item = mechanism([direction("monotone")])
        result = evaluate(item, [0., .4, .1, .6], progress=[0., .5, .5, 1.])
        self.assertAlmostEqual(float(next(iter(result.values())).max()), .3)

    def test_rate_uses_seconds_and_actual_twist_and_preserves_gradient(self):
        item = mechanism([direction("local_rate", minimum=.2)])
        velocity = torch.tensor([[[.1, 0., 0.], [.3, 0., 0.]]], dtype=torch.float64, requires_grad=True)
        result = evaluate(item, [0., .5], times=[0., 10.], tool_linear_velocity=velocity,
                          tool_angular_velocity=torch.zeros_like(velocity))
        residual = next(iter(result.values()))
        self.assertAlmostEqual(float(residual.detach().max()), .1)
        residual.sum().backward()
        self.assertAlmostEqual(float(velocity.grad[0, 0, 0]), -1.)
        self.assertEqual(float(velocity.grad[0, 1, 0]), 0.)
        # Geometric-path seed screening uses .5 / 10 seconds, never progress.
        screened = evaluate(item, [0., .5], times=[0., 10.], mode="path")
        self.assertAlmostEqual(float(next(iter(screened.values())).max()), .15)
        with self.assertRaisesRegex(ValueError, "actual tool twist"):
            evaluate(item, [0., .5])
        self.assertEqual(evaluate(item, [.5], mode="point"), {})

    def test_base_tcp_and_carrier_offsets_produce_correct_native_rate(self):
        request = problem([MotionPhase("reference", path([pose(), pose()]))],
            base_pose=pose(angles=(0., 0., np.pi/2)), tool_transform=pose((.3, 0., 0.)))
        linear = torch.zeros((1, 1, 3), dtype=torch.float64)
        angular = torch.tensor([[[0., 0., 2.]]], dtype=torch.float64)
        quaternion = torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64)
        velocity, omega = request.transform_tool_velocity(linear, angular, quaternion)
        np.testing.assert_allclose(velocity.numpy(), [[[-.6, 0., 0.]]], atol=1e-14)
        _, tool_quaternion = request.transform_tool_pose(linear, quaternion)
        item = mechanism()
        item.tool_from_carrier = pose((.2, 0., 0.))
        rate = item.coordinate_rates(tool_quaternion, velocity, omega)
        self.assertAlmostEqual(float(rate.item()), -1.)

    def test_angular_rate_differentiates_the_same_projected_coordinate(self):
        item = mechanism(kind="revolute")
        initial = Rotation.from_euler("xyz", [.1, -.08, .7])
        angular = np.array([.3, -.2, .6])
        delta = 1e-6
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[0, :3, :3] = (Rotation.from_rotvec(-delta*angular)*initial).as_matrix()
        poses[1, :3, :3] = (Rotation.from_rotvec(delta*angular)*initial).as_matrix()
        p, q = components(poses[None])
        values = item.coordinates(p, q, torch.tensor([[0., 1.]], dtype=torch.float64))
        finite_difference = float((values[0, 1, 0]-values[0, 0, 0])/(2*delta))
        center = np.eye(4); center[:3, :3] = initial.as_matrix()
        _, center_q = components(center[None, None])
        rate = item.coordinate_rates(center_q, torch.zeros((1, 1, 3), dtype=torch.float64),
                                     torch.tensor(angular[None, None], dtype=torch.float64))
        self.assertAlmostEqual(float(rate.item()), finite_difference, places=8)

    def test_direction_validation_rejects_implicit_or_contradictory_contract(self):
        with self.assertRaises(ValueError):
            direction("monotone", minimum=.1)
        with self.assertRaises(ValueError):
            direction("net", sign=0)
        with self.assertRaises(ValueError):
            replace(direction("net"), origin_coordinate=.2)

    def test_export_rejects_instantaneous_reverse_rate_between_positive_positions(self):
        from manipulation.planning import _ProgressMap
        from tests.test_manipulation_planning import cartesian_problem, constant_path, export_fixture
        request = cartesian_problem((MotionPhase("free reference", constant_path()),))
        request.mechanisms = (mechanism([direction("local_rate", start=.25, end=.75)]),)
        planner, native = export_fixture(request, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
            [[0., 0., 0.], [.2, 0., 0.], [.4, 0., 0.]],
            [[0., 0., 0.], [-.1, 0., 0.], [0., 0., 0.]], 1.)
        original_fk = planner._verification_fk.compute_kinematics
        def fk(state):
            result = original_fk(state)
            jacobian = state.position.new_zeros((*state.position.shape[:2], 1, 6, 3))
            jacobian[..., 0, :3, :] = torch.eye(3, dtype=state.position.dtype)
            result.tool_jacobians = jacobian
            return result
        planner._verification_fk.compute_kinematics = fk
        curve, _ = planner.export(native)
        self.assertAlmostEqual(curve.residuals["mechanism:0:direction:0:local_rate"], .1)
        report = planner.last_validation["coordinate_direction_resolution"]
        self.assertGreater(report["query_count"], 3)
        self.assertIn("no continuous-time", report["scope"])

    def test_delta_target_keeps_original_origin_and_compiler_evidence(self):
        skill = prismatic_input()
        binding = skill["scene"]["functional_bindings"]["function"]
        binding.update(coordinate_mode="entry_delta", current_coordinates=[.04],
                       entry_coordinates=[.1], entry_source="original synchronized request entry")
        request = compile_inputs(skill)
        goal = request.coordinate_events[0]
        self.assertEqual(goal.coordinate_mode, "entry_delta")
        self.assertAlmostEqual(goal.target_coordinate, .04)
        self.assertAlmostEqual(goal.coordinate_origin, .1)
        self.assertAlmostEqual(goal.absolute_target, .14)
        fresh = replace(goal.mechanism, start=.5, initial_coordinates=np.array([.12]))
        retained = replace(goal, mechanism=fresh)
        self.assertAlmostEqual(retained.absolute_target, .14)
        del binding["entry_source"]
        with self.assertRaisesRegex(ValueError, "coordinate entry"):
            ground_program(skill["program"], skill["scene"])

    def test_compiler_binds_explicit_source_scopes_and_keeps_reference_optional(self):
        skill = prismatic_input()
        skill["coordinate_directions"] = [dict(start_frame=100, end_frame=106,
            coordinate_index=0, kind="net", sign=1, minimum=.04, tolerance=1e-8,
            source="requested positive opening")]
        request = compile_inputs(skill)
        value = request.mechanisms[0].directions[0]
        self.assertEqual((value.start, value.kind), (0., "net"))
        self.assertAlmostEqual(value.end, request.coordinate_events[0].progress)
        self.assertAlmostEqual(value.origin_coordinate, .1)
        self.assertEqual(compile_inputs(prismatic_input()).mechanisms[0].directions, ())

    def test_optimizer_direction_scale_uses_same_relative_accuracy_without_relaxing_gate(self):
        preferences = []
        for minimum, tolerance in ((.1, .01), (.001, .0001)):
            item = mechanism([direction("net", minimum=minimum, tolerance=tolerance)])
            request = problem([MotionPhase("reference", path([pose(), pose()]))], mechanisms=(item,))
            p, q = components(item.model.poses(np.array([[0.], [minimum-2*tolerance]]))[None])
            progress, times = torch.tensor([[0., 1.]]), torch.tensor([[0., 2.]])
            raw = request.residuals(p, q, progress, trajectory_time=times)
            self.assertAlmostEqual(float(raw["mechanism:0:direction:0:net"].max()), tolerance)
            preferences.append(float(request.optimizer_preferences(p, q, progress, trajectory_time=times).sum()))
        np.testing.assert_allclose(preferences, [4., 4.], atol=1e-12)

    def test_observed_actual_lift_is_preserved_separately_from_entry_delta_origin(self):
        skill = revolute_input()
        spec = skill["scene"]["motion_model"]
        spec.update(initial_coordinates=[4*np.pi], coordinate_limits=[[0., 6*np.pi+.1]],
                    initial_coordinates_source="actual observed two completed turns")
        binding = skill["scene"]["functional_bindings"]["function"]
        binding.update(coordinate_mode="entry_delta", current_coordinates=[4*np.pi],
                       entry_coordinates=[2*np.pi], entry_source="original request one-turn entry")
        request = compile_inputs(skill)
        self.assertAlmostEqual(request.mechanisms[0].initial_coordinates[0], 4*np.pi)
        self.assertAlmostEqual(request.coordinate_events[0].coordinate_origin, 2*np.pi)
        self.assertAlmostEqual(request.coordinate_events[0].absolute_target, 6*np.pi)
        grounded = ground_program(skill["program"], skill["scene"])
        self.assertAlmostEqual(grounded["motion_checks"]["current_initial_coordinates"][0], 4*np.pi)

    def test_ambiguous_angular_entry_requires_actual_branch_evidence(self):
        skill = revolute_input()
        spec = skill["scene"]["motion_model"]
        spec["coordinate_limits"] = [[0., 4*np.pi]]
        del spec["initial_coordinates"]
        del spec["initial_coordinates_source"]
        with self.assertRaisesRegex(ValueError, "Ambiguous actual angular branch"):
            compile_inputs(skill)
        spec["initial_coordinates"] = [4*np.pi]
        with self.assertRaisesRegex(ValueError, "Observed initial_coordinates"):
            compile_inputs(skill)

    def test_unique_angular_branch_in_physical_limits_needs_no_extra_turn_sensor(self):
        skill = revolute_input()
        spec = skill["scene"]["motion_model"]
        del spec["initial_coordinates"]
        del spec["initial_coordinates_source"]
        spec["coordinate_limits"] = [[2*np.pi-.1, 2*np.pi+.1]]
        request = compile_inputs(skill)
        self.assertAlmostEqual(request.mechanisms[0].initial_coordinates[0], 2*np.pi)
        self.assertAlmostEqual(request.coordinate_events[0].absolute_target, 2*np.pi)

    def test_actual_entry_branch_validation_does_not_select_a_future_goal_member(self):
        # BEGIN only immutable actual entry has a pose-to-coordinate precondition.
        from manipulation.program import _mechanism_entry_contract
        skill = revolute_input()
        spec, initial = skill["scene"]["motion_model"], skill["scene"]["initial_pose"]
        spec.update(initial_coordinates=[2*np.pi], coordinate_limits=[[2*np.pi-.1, 2*np.pi+.1]],
                    initial_coordinates_source="actual one-turn entry")
        model, limits = _mechanism_entry_contract(spec, initial)
        np.testing.assert_allclose(model.coordinates, [[0.]])
        np.testing.assert_allclose(limits, [[2*np.pi-.1, 2*np.pi+.1]])
        # A physically forbidden actual branch cannot be replaced by another
        # equivalent pose, even though future pose-only allowed sets stay free.
        spec["initial_coordinates"] = [0.]
        with self.assertRaisesRegex(ValueError, "exceeds current mechanism limits"):
            _mechanism_entry_contract(spec, initial)
        spec["initial_coordinates"] = [2*np.pi]
        spec["coordinate_limits"] = [[0., 4*np.pi]]
        _mechanism_entry_contract(spec, initial)
        self.assertEqual(spec["initial_coordinates"], [2*np.pi])
        # END fresh actual branch evidence remains distinct from task targets.


if __name__ == "__main__":
    unittest.main()
# END actual-time coordinate direction and target-origin regression coverage.
