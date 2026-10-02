"""CPU checks of source timing, activation and true native terminal goals."""

# BEGIN frozen binding and target generation contracts independent of CUDA.
from types import SimpleNamespace
from dataclasses import asdict
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from manipulation.planning import (PlanningBudget, _CuroboPlanner, _ProgressMap,
    ContinuationBindingError, _bind_event_clock, _initial_timing,
    _prepare_continuation, _continuation_timing, _fixed_spline_grid,
    _ompl_stage, _scene_boundary_evaluate, _timed_native_state,
    _progress_for_cubic_boundaries, _select_ik_candidates)
from manipulation.planning_problem import (MotionPhase, PlanningEvent, PlanningProblem,
                                          PosePath, PoseEventRequirement)


class PlanningBindingTest(unittest.TestCase):
    def problem(self, events=(), *, source_times=(10., 11., 14.), source_progress=(0., .8, 1.)):
        path = PosePath(np.array([0., 1.]), np.repeat(np.eye(4)[None], 2, 0), "reference only")
        return PlanningProblem(joint_names=("q",), start_q=np.zeros(1), start_dq=np.zeros(1),
            tool_link="tool", reference_frame="world", base_pose=np.eye(4), tool_transform=np.eye(4),
            phases=(MotionPhase("reference", path),), events=tuple(events),
            source_timestamps=np.asarray(source_times), source_progress=np.asarray(source_progress))

    def test_source_seconds_preserve_uneven_rhythm_in_initial_proposal(self):
        problem = self.problem()
        budget, mapping = _initial_timing(problem, PlanningBudget())
        np.testing.assert_array_equal(mapping.u, [0., .25, 1.])
        np.testing.assert_array_equal(mapping.s, [0., .8, 1.])
        self.assertLessEqual(budget.maximum_trajectory_dt, PlanningBudget().maximum_trajectory_dt)
        self.assertNotEqual(budget.maximum_trajectory_dt, PlanningBudget().maximum_trajectory_dt)

    def test_command_goal_query_uses_activation_and_keeps_intent_separate(self):
        event = PlanningEvent("close", "current", .5, 10, "close", "source close")
        problem = self.problem((event,))
        original = _ProgressMap(np.array([0., .37, .9, 1.]), np.array([0., .5, 1., 1.]))
        bound = _bind_event_clock(problem, original, 1., .2)
        record = bound.event_clock[("current", "close")]
        self.assertEqual(record["intent_time_s"], .37)
        self.assertEqual(record["time"], .37)
        self.assertEqual(record["activation_time_s"], .4)
        self.assertAlmostEqual(bound.anchor_parameters([.5])[0, 0], .4)
        np.testing.assert_array_equal(original.u, [0., .37, .9, 1.])

    def test_no_command_coverage_or_merged_ticks_is_a_binding_conflict(self):
        events = (PlanningEvent("open", "current", .31, 1, "open", "original"),
                  PlanningEvent("close", "current", .32, 2, "close", "original"))
        with self.assertRaisesRegex(ValueError, "merges or reverses"):
            _bind_event_clock(self.problem(events), _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 1., .5)
        terminal = PlanningEvent("close", "current", 1., 3, "close", "original")
        with self.assertRaisesRegex(ValueError, "activation/effect coverage"):
            _bind_event_clock(self.problem((terminal,)), _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 1., .1)

    def test_suffix_retains_original_progress_identity(self):
        problem = self.problem(source_times=(3., 4., 5.), source_progress=(0., .7, 1.))
        problem.start_progress = .4
        problem.source_timestamps = np.array([3., 4., 5.])
        problem.source_progress = np.array([.4, .7, 1.])
        _, mapping = _initial_timing(problem, PlanningBudget())
        warped, _ = _progress_for_cubic_boundaries(mapping, 8, [0.], stop_progress=(), anchors=[.4, 1.])
        self.assertEqual(warped.s[0], .4)
        self.assertEqual(warped.s[-1], 1.)
        self.assertFalse(np.any(warped.s < .4))

    def test_only_true_terminal_pose_enables_native_target(self):
        problem = self.problem()
        target = np.eye(4); target[0, 3] = .8
        function = np.eye(4); function[0, 3] = .1
        goal = PoseEventRequirement(.5, function, target, .01, .02, "true goal")
        problem.pose_events = (goal,)
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.problem, planner.torch = problem, torch
        planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float32))
        planner.GoalToolPose = SimpleNamespace
        self.assertIsNone(planner._terminal_goal())
        goal.progress = 1.
        native = planner._terminal_goal()
        np.testing.assert_allclose(native.position.numpy().reshape(3), [.7, 0., 0.], atol=1e-7)
        # Changing the reference path cannot move the actual native goal.
        problem.phases[0].reference.poses[:, 0, 3] = -.8
        torch.testing.assert_close(planner._terminal_goal().position, native.position)
        goal.target[0, 3] = .5
        self.assertAlmostEqual(float(planner._terminal_goal().position.flatten()[0]), .4, places=6)

    def test_nonfinite_single_ik_seed_does_not_hide_a_valid_success(self):
        candidates = np.array([[np.nan, 0.], [1., 2.], [3., 4.]])
        selected, distinct = _select_ik_candidates(candidates, [True, True, False],
            np.array([[0., 0.]]), 1, 1e-6)
        np.testing.assert_array_equal(selected, [1])
        self.assertEqual(distinct, 1)

    def test_native_terminal_envelope_does_not_shrink_offset_function_goal(self):
        from scipy.spatial.transform import Rotation
        problem = self.problem()
        problem.tool_transform[0, 3] = .1
        function = np.eye(4); function[0, 3] = .2
        target = PoseEventRequirement(1., function, np.eye(4), .001, .02, "goal")
        problem.pose_events = (target,)
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.problem = problem
        bound, rotation_bound = planner._terminal_tolerances()
        # The true functional point is exactly on target while the permitted
        # rotation moves both TCP and native link origins through lever arms.
        rotation = Rotation.from_rotvec([0., 0., .019999])
        quaternion = torch.as_tensor(rotation.as_quat()[[3, 0, 1, 2]], dtype=torch.float64)[None]
        tool_position = torch.as_tensor(-rotation.as_matrix()@function[:3, 3], dtype=torch.float64)[None]
        self.assertEqual(float(target.evaluate(tool_position, quaternion)), 0.)
        native_position, _ = problem.reference_to_link_pose(tool_position, quaternion)
        exact_native_target = np.array([-.3, 0., 0.])
        native_error = np.linalg.norm(native_position.numpy()[0]-exact_native_target)
        self.assertGreater(native_error, target.position_tolerance_m)
        self.assertLess(native_error, bound)
        self.assertEqual(rotation_bound, target.rotation_tolerance_rad+problem.numerical.rotation_rad)

    def test_dependency_boundary_inside_reference_interval_uses_device_tick(self):
        event = PlanningEvent("effect", "current", .37, 1, "verify", "dependency",
                              requires_evidence="Holding")
        original = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        bound = _bind_event_clock(self.problem((event,)), original, 1., .1)
        record = bound.event_clock[("current", "effect")]
        self.assertEqual(record["intent_time_s"], .37)
        self.assertEqual(record["consumption_time_s"], .4)
        self.assertEqual(record["time"], .4)
        self.assertIsNone(record["activation_time_s"])
        self.assertEqual(bound.anchor_parameters([.37])[0, 0], .4)

    def continuation(self, problem, *, elapsed=.25):
        from manipulation.execution_binding import plain
        from manipulation.gripper_trajectory import _activation
        rows = [{**plain(event), "time": event.progress,
                 "activation_time_s": event.progress if event.kind in {"open", "close", "hold"} else None}
                for event in problem.events]
        timestep = .001
        local_steps = int(round(elapsed/timestep))
        for row in rows:
            if row["activation_time_s"] is not None:
                step, applied = _activation(row["time"], timestep)
                row.update(activation_time_s=applied, bound_activation_step=step,
                    request_activation_step=step, request_activation_time_s=applied,
                    request_intent_time_s=row["time"], request_time_origin_step=0,
                    device_timestep_s=timestep)
        budget = PlanningBudget(maximum_trajectory_solves=4, search_seconds=2.)
        original = {"candidate_id": "candidate", "planning_budget": asdict(budget),
                    "consumed_planning": {"trajectory_solves": 2, "search_seconds": .7},
                    "physics_budget": 1000, "consumed_physics_steps": 100,
                    "problem": {"events": plain(problem.events)},
                    "curve": {"duration": 1., "progress_u": [0., .5, 1.],
                              "progress_s": [0., .8, 1.], "events": rows}}
        return {"candidate_id": "candidate", "original_binding": original,
                "physics_budget": 1000, "cumulative_physics_steps": 100+local_steps,
                "remaining_physics_steps": 900-local_steps, "consumed_physics_steps": local_steps,
                "device_timestep_s": timestep,
                "actual_feedback": {"joint_names": ["q"], "q": [0.], "dq": [0.]},
                "elapsed_physical_s": elapsed,
                "remaining_events": deepcopy(rows), "applied_events": [], "verified_events": []}

    def test_continuation_keeps_absolute_suffix_clock_and_cumulative_budget(self):
        event = PlanningEvent("close", "current", .9, 1, "close", "task")
        problem = self.problem((event,))
        continuation = self.continuation(problem)
        budget, prior = _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        self.assertEqual(budget.maximum_trajectory_solves, 2)
        self.assertAlmostEqual(budget.search_seconds, 1.3)
        self.assertEqual(prior["consumed_planning"]["trajectory_solves"], 2)
        self.assertEqual(prior["consumed_physics_steps"], 350)
        self.assertEqual(prior["remaining_physics_steps"], 650)
        self.assertEqual(problem.start_progress, .4)
        selected, mapping = _continuation_timing(problem, budget, prior)
        duration = _fixed_spline_grid(intervals=selected.knots+4,
            native_dt=selected.maximum_trajectory_dt, interpolation_steps=selected.interpolation_steps,
            interpolation_dt=selected.interpolation_dt)["duration"]
        times = np.linspace(0., .75, 101)
        np.testing.assert_allclose(np.interp(times/duration, mapping.u, mapping.s),
            np.interp(times+.25, [0., .5, 1.], [0., .8, 1.]), atol=1e-15)
        self.assertEqual(mapping.event_clock[("current", "close")]["activation_time_s"], .65)
        self.assertEqual(mapping.event_clock[("current", "close")]["request_intent_time_s"], .9)

    def test_continuation_never_replays_applied_events_or_guesses_past_effects(self):
        events = (PlanningEvent("earlier", "current", .1, 1, "close", "task"),
                  PlanningEvent("later", "current", .9, 2, "open", "task"))
        problem = self.problem(events)
        continuation = self.continuation(problem)
        with self.assertRaisesRegex(ContinuationBindingError, "overdue"):
            _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        continuation["remaining_events"] = continuation["remaining_events"][1:]
        with self.assertRaisesRegex(ContinuationBindingError, "actual application or verification"):
            _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        continuation["applied_events"] = [{"skill_id": "current", "event_id": "earlier"}]
        _, prior = _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        self.assertEqual([event.event_id for event in problem.events], ["later"])
        self.assertEqual(prior["consumed_events"], [("current", "earlier")])

    def test_unknown_ordinary_effect_evaluation_does_not_block_remaining_motion(self):
        events = (PlanningEvent("observe", "current", .1, 1, "effect", "task"),
                  PlanningEvent("later", "current", .9, 2, "open", "task"))
        problem = self.problem(events)
        continuation = self.continuation(problem)
        continuation["evaluated_events"] = [{"skill_id": "current", "event_id": "observe",
            "kind": "effect", "time": .1, "evaluation_time_s": .1, "evaluation": "U",
            "effects": {}, "source": "current scene has no conclusive effect observation"}]
        continuation["effects"] = {"Holding": {"value": "U", "source": "occluded"}}
        _, prior = _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        self.assertEqual([event.event_id for event in problem.events], ["later"])
        self.assertEqual(prior["effects"]["Holding"]["value"], "U")
        self.assertEqual(prior["consumed_events"], [("current", "observe")])

    def test_continuation_requires_past_goal_proof_and_preserves_original_indices(self):
        problem = self.problem()
        problem.pose_events = tuple(PoseEventRequirement(progress, np.eye(4), np.eye(4), .01, .02, "goal")
                                    for progress in (.1, .9))
        continuation = self.continuation(problem)
        with self.assertRaisesRegex(ContinuationBindingError, "Past goal pose_events:0"):
            _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        continuation["completed_goal_ids"] = [["pose_events", 0]]
        _, prior = _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        self.assertEqual(prior["goal_identities"]["pose_events"], [1])
        self.assertEqual(problem.pose_events[0].progress, .9)

    def test_continuation_cannot_reset_budget_actual_or_event_identity(self):
        for mutation, message in (("budget", "cannot reset"), ("actual", "Fresh planning actual"),
                                  ("consumption", "retain all prior"), ("identity", "candidate identity")):
            with self.subTest(mutation=mutation):
                problem = self.problem()
                continuation = self.continuation(problem)
                budget = None
                if mutation == "budget":
                    budget = PlanningBudget()
                elif mutation == "actual":
                    continuation["actual_feedback"]["q"] = [.1]
                elif mutation == "consumption":
                    continuation["consumed_planning"] = {"trajectory_solves": 1, "search_seconds": .7}
                else:
                    continuation["candidate_id"] = "different"
                with self.assertRaisesRegex(ContinuationBindingError, message):
                    _prepare_continuation(problem, SimpleNamespace(gripper=None), budget, continuation)

    def test_continuation_cannot_reset_physics_budget_or_erase_hold_steps(self):
        for field, value in (("physics_budget", 1200), ("cumulative_physics_steps", 99),
                             ("remaining_physics_steps", 801)):
            with self.subTest(field=field):
                problem = self.problem()
                continuation = self.continuation(problem)
                continuation[field] = value
                with self.assertRaisesRegex(ContinuationBindingError, "cumulative physics-step budget"):
                    _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)

    def test_hold_steps_over_original_cap_are_valid_exhausted_history(self):
        problem = self.problem()
        continuation = self.continuation(problem)
        continuation.update(cumulative_physics_steps=1002, remaining_physics_steps=0)
        with self.assertRaisesRegex(ContinuationBindingError, "physics-step budget is exhausted"):
            _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)

    def test_native_ompl_connector_checks_crossed_event_time_and_progress_plateau(self):
        checked = []
        mapping = _ProgressMap(np.array([0., .3, .7, 1.]), np.array([0., .5, .5, 1.]))
        def valid(state):
            checked.append(state.copy())
            return True
        path, report = _ompl_stage(np.array([0., 0.]), np.array([1., 1.]),
            np.array([-2., 0.]), np.array([2., 1.]), valid,
            lambda state, derivative: np.empty((0, len(state))) if derivative else np.empty(0),
            tolerance=1e-5, step=.2, seconds=1., sample_limit=256,
            query_parameters=[.3, .37, .7])
        self.assertIsNotNone(path)
        self.assertEqual(report["strategy"], "native_direct_connection")
        self.assertTrue(np.all(np.diff(path[:, -1]) > 0))
        self.assertTrue(np.any(np.isclose(path[:, -1], .37, atol=1e-15, rtol=0.)))
        pause = (path[:, -1] >= .3) & (path[:, -1] <= .7)
        self.assertGreater(np.ptp(path[pause, 0]), .3)
        np.testing.assert_array_equal(np.interp(path[pause, -1], mapping.u, mapping.s),
                                      np.full(pause.sum(), .5))

    def test_scene_switch_checks_float32_sides_at_same_q_and_time_without_relocking_plateau(self):
        mapping = _ProgressMap(np.array([0., .25, .75, 1.]), np.array([0., .5, .5, 1.]))
        progress = torch.full((1, 3), .5, dtype=torch.float32)
        clock = torch.tensor([[.25, .5, .75]], dtype=torch.float32)
        q = torch.tensor([[[0.], [.1], [.1]]], dtype=torch.float32)
        observations = []
        def evaluate(spheres, position, quaternion, s, joints, names, *, trajectory_time):
            observations.append((s.clone(), joints.clone(), trajectory_time.clone()))
            return ((s < .5) & (joints[..., 0] > 0)).to(s.dtype)
        scene = SimpleNamespace(required_progress_anchors=(.5,), evaluate=evaluate)
        def check(joints):
            return _scene_boundary_evaluate(scene, self.problem(), mapping,
                torch.zeros((1, 3, 1, 4)), torch.zeros((1, 3, 3)),
                torch.tensor([[[1., 0., 0., 0.]]]).expand(1, 3, 4),
                progress, joints, ("q",), trajectory_time=clock, normalized_time=clock)
        self.assertTrue(bool((check(q) == 0).all()))
        sides, positions, times = observations[-1]
        self.assertLess(float(sides[0, 3]), .5)
        self.assertEqual(float(sides[0, 4]), .5)
        self.assertGreater(float(sides[0, -1]), .5)
        for columns in (slice(0, 3), slice(3, 6), slice(6, 9)):
            torch.testing.assert_close(positions[:, columns], q, rtol=0., atol=0.)
            torch.testing.assert_close(times[:, columns], clock, rtol=0., atol=0.)
        q[:, 0] = .1
        self.assertGreater(float(check(q)[0, 0]), 0.)

    def test_unknown_finger_model_keeps_actual_entrance_palm_and_arm_native_pairs(self):
        problem = self.problem()
        spheres = torch.zeros((1, 2, 3, 4)); spheres[..., 3] = .02
        quaternion = torch.zeros((1, 2, 1, 4)); quaternion[..., 0] = 1.
        native = SimpleNamespace(robot_spheres=spheres,
            tool_jacobians=torch.zeros((1, 2, 1, 6, 1)))
        state = SimpleNamespace(joint_state=SimpleNamespace(position=torch.zeros((1, 2, 1))),
            cuda_robot_model_state=native, robot_spheres=spheres,
            tool_poses=SimpleNamespace(tool_frames=["tool"], position=torch.zeros((1, 2, 1, 3)),
                                       quaternion=quaternion))
        hand = SimpleNamespace(controller="timed_reference", nominal_finger_trajectory=None,
            sphere_indices=(0, 1), sphere_joint_translation=np.array([[[0., 0., 0.]], [[1., 0., 0.]]]),
            state=lambda *args, **kwargs: (None, None, None), spheres=lambda values, *args: values)
        with patch("manipulation.planning._native_gradient_layout_type",
                   return_value=SimpleNamespace(apply=lambda value: value)), \
             patch("manipulation.planning._orientation_vjp_type",
                   return_value=SimpleNamespace(apply=lambda q, r, jac: r)):
            result = _timed_native_state(state, problem, SimpleNamespace(gripper=hand), ("q",),
                                        torch.tensor([[0., .1]]), sphere_padding=.05)
        radii = result.cuda_robot_model_state.robot_spheres[..., 3]
        torch.testing.assert_close(radii[:, 0], spheres[:, 0, :, 3], rtol=0., atol=0.)
        self.assertLess(float(radii[0, 1, 1]+.05), 0.)
        torch.testing.assert_close(radii[:, 1, [0, 2]], spheres[:, 1, [0, 2], 3], rtol=0., atol=0.)
        torch.testing.assert_close(state.cuda_robot_model_state.robot_spheres, spheres, rtol=0., atol=0.)


if __name__ == '__main__':
    unittest.main()
# END these tests claim clock/goal contracts, not physical execution success.
