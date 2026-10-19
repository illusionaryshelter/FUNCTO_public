# BEGIN tests migrated to native motion attempts and retained numerical contracts.
"""CPU checks of source timing, activation and functional IK goal proposals."""

from types import SimpleNamespace
from dataclasses import asdict, replace
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from manipulation.planning import (PlanningBudget, _CuroboPlanner, _ProgressMap,
    ContinuationBindingError, _bind_event_clock, _initial_timing,
    _prepare_continuation, _continuation_timing, _fixed_spline_grid,
    _ompl_stage, _known_scene_cost, _timed_native_state,
    _progress_for_cubic_boundaries)
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

    def test_source_duration_selects_exact_dense_grid_within_existing_capacity(self):
        original = PlanningBudget(knots=16, interpolation_buffer_size=768)
        problem = self.problem(source_times=(0., 4., 9.850000000000293))
        selected, mapping = _initial_timing(problem, original)
        grid = _fixed_spline_grid(intervals=20, native_dt=selected.maximum_trajectory_dt,
            interpolation_steps=selected.interpolation_steps, interpolation_dt=selected.interpolation_dt)
        self.assertEqual((grid["subdivisions"], grid["valid_count"]), (32, 641))
        self.assertGreaterEqual(grid["duration"], problem.source_timestamps[-1])
        self.assertLess(grid["duration"]-problem.source_timestamps[-1], 2e-6)
        self.assertLessEqual(grid["interpolation_dt"], original.interpolation_dt)
        self.assertEqual(grid["interpolation_dt"]*32, grid["knot_dt"])
        self.assertEqual(grid["native_dt"]*4, grid["knot_dt"])
        self.assertEqual(grid["interpolation_dt"]*640, grid["duration"])
        np.testing.assert_array_equal(mapping.u, problem.source_timestamps/problem.source_timestamps[-1])

    def test_source_clock_keeps_general_step_counts_and_original_duration_cap(self):
        # These clocks were already admissible. Source quantization must also
        # preserve the exact native multiplication for odd step counts.
        for steps in (1, 2, 3, 4, 5, 6, 7, 8):
            with self.subTest(steps=steps):
                original = PlanningBudget(knots=6, interpolation_steps=steps,
                    maximum_trajectory_dt=.125, interpolation_dt=.03125, interpolation_buffer_size=512)
                problem = self.problem(source_times=(0., .4, 1.001))
                selected, _ = _initial_timing(problem, original)
                grid = _fixed_spline_grid(intervals=10, native_dt=selected.maximum_trajectory_dt,
                    interpolation_steps=steps, interpolation_dt=selected.interpolation_dt)
                self.assertGreaterEqual(grid["duration"], 1.001)
                self.assertLess(grid["duration"]-1.001, 2e-6)
                self.assertLessEqual(selected.maximum_trajectory_dt, .125)
                longer = self.problem(source_times=(0., 20., 100.))
                capped, _ = _initial_timing(longer, original)
                self.assertEqual(capped.maximum_trajectory_dt, .125)

    def test_selected_source_clock_binds_native_events_and_timed_fingers_together(self):
        from curobo._src.types.control_space import ControlSpace
        from manipulation.planning import JointTrajectory, _formal_native_clock
        from manipulation.planning_scene import PrismaticGripper

        event = PlanningEvent("close", "current", .8, 183, "close", "transferred reference command")
        problem = self.problem((event,), source_times=(0., 9.15, 9.85), source_progress=(0., .8, 1.))
        budget, mapping = _initial_timing(problem, PlanningBudget(knots=16, interpolation_buffer_size=768))
        native = SimpleNamespace(config=SimpleNamespace(minimum_trajectory_dt=budget.maximum_trajectory_dt,
            maximum_trajectory_dt=budget.maximum_trajectory_dt, interpolation_dt=budget.interpolation_dt),
            interpolation_steps=budget.interpolation_steps, action_horizon=budget.knots+2,
            transition_model=SimpleNamespace(control_space=ControlSpace.BSPLINE_3))
        grid = _formal_native_clock(native)
        mapping, _ = _progress_for_cubic_boundaries(mapping, budget.knots,
            stop_progress=problem.stop_progress, anchors=problem.anchors)
        mapping = _bind_event_clock(problem, mapping, grid["duration"], .002)
        seen = []

        def fingers(duration, progress_map):
            seen.append((duration, progress_map))
            return JointTrajectory(("finger",), np.array([0., duration]), np.full((2, 1), .04),
                np.zeros((2, 1)), progress_map.u, progress_map.s, (), (), {})

        def geometry(duration, progress_map, command_reference):
            from manipulation.gripper_trajectory import NominalFingerGeometry
            return NominalFingerGeometry.from_reference(command_reference,
                valid_until_s=duration, source="explicit whole-clock constant fixture")

        hand = PrismaticGripper(controller="timed_reference", control_timestep_s=.002,
            joint_names=("finger",), current_positions=[.04], joint_limits=[[0., .04]], joint_bounds=None,
            finger_trajectory_factory=fingers, nominal_finger_trajectory_factory=geometry,
            coupling_matrix=np.empty((0, 1)), coupling_offset=[], sphere_indices=(0,),
            local_spheres=[[0., 0., 0., .001]], sphere_joint_translation=[[[1.], [0.], [0.]]],
            self_collision_pairs=(), commands=[{"skill_id": "current", "event_id": "close", "occurrence": 0,
                "command": "close", "progress": .8}], source="explicit CPU clock contract")
        hand.bind_clock(grid["duration"], mapping)
        self.assertTrue(all(duration == grid["duration"] and actual_map is mapping for duration, actual_map in seen))
        self.assertEqual(hand.finger_trajectory.duration, hand.nominal_finger_trajectory.duration)
        self.assertEqual(hand.finger_trajectory.duration, grid["duration"])
        record = mapping.event_clock[("current", "close", 0)]
        self.assertEqual(record["activation_time_s"], mapping.anchor_parameters([.8])[0, 0]*grid["duration"])
        self.assertEqual(record["activation_time_s"], record["bound_activation_step"]*.002)
        self.assertEqual(grid["duration"], (grid["valid_count"]-1)*grid["interpolation_dt"])
        self.assertEqual(grid["duration"], (budget.knots+4)*budget.interpolation_steps*grid["native_dt"])

    def test_command_goal_query_uses_activation_and_keeps_intent_separate(self):
        event = PlanningEvent("close", "current", .5, 10, "close", "source close")
        problem = self.problem((event,))
        original = _ProgressMap(np.array([0., .37, .9, 1.]), np.array([0., .5, 1., 1.]))
        bound = _bind_event_clock(problem, original, 1., .2)
        record = bound.event_clock[("current", "close", 0)]
        self.assertEqual(record["intent_time_s"], .37)
        self.assertEqual(record["time"], .37)
        self.assertEqual(record["activation_time_s"], .4)
        self.assertAlmostEqual(bound.anchor_parameters([.5])[0, 0], .4)
        np.testing.assert_array_equal(original.u, [0., .37, .9, 1.])

    def test_uncovered_commands_or_required_occurrence_conflicts_reject_binding(self):
        events = (PlanningEvent("open", "current", .31, 1, "open", "original", stop=True),
                  PlanningEvent("close", "current", .32, 2, "close", "original"))
        with self.assertRaisesRegex(ValueError, "required physical occurrence"):
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
        warped, _ = _progress_for_cubic_boundaries(mapping, 8, stop_progress=(), anchors=[.4, 1.])
        self.assertEqual(warped.s[0], .4)
        self.assertEqual(warped.s[-1], 1.)
        self.assertFalse(np.any(warped.s < .4))


    def test_nominal_goals_do_not_enable_native_convergence_gates(self):
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.trajopt = SimpleNamespace(disable_tool_pose_tracking=Mock(), disable_joint_position_tracking=Mock())
        state, goal = planner._native_request(torch.zeros((1, 2, 4, 1)))
        self.assertIsNone(goal)
        self.assertEqual(state.num_seeds, 2)
        self.assertEqual(state.tool_frames, [])
        planner.trajopt.disable_tool_pose_tracking.assert_called_once()


    def test_dependency_boundary_inside_reference_interval_uses_device_tick(self):
        event = PlanningEvent("effect", "current", .37, 1, "verify", "dependency",
                              requires_evidence="Holding")
        original = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        bound = _bind_event_clock(self.problem((event,)), original, 1., .1)
        record = bound.event_clock[("current", "effect", 0)]
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
        budget = PlanningBudget(maximum_trajectory_solves=2, search_seconds=2.)
        original = {"candidate_id": "candidate", "planning_budget": asdict(budget),
                    "consumed_planning": {"binding_attempts": 1, "initializations": 1, "ik_batches": 1,
                        "ik_queries": 24, "search_queries": 0, "validation_passes": 2, "validation_queries": 20,
                        "trajectory_solves": 2, "search_seconds": .7, "planning_seconds": 1.},
                    "physics_budget": 1000, "consumed_physics_steps": 100,
                    "problem": plain(problem),
                    "curve": {"duration": 1., "progress_u": [0., .5, 1.],
                              "progress_s": [0., .8, 1.], "events": rows, "executable_until": 1.}}
        from manipulation.planning import _request_work_ledger
        ledger = _request_work_ledger(problem, budget)
        ledger["consumed_planning"] = deepcopy(original["consumed_planning"])
        original["request_ledger"] = ledger
        return {"candidate_id": "candidate", "original_binding": original,
                "physics_budget": 1000, "cumulative_physics_steps": 100+local_steps,
                "remaining_physics_steps": 900-local_steps, "consumed_physics_steps": local_steps,
                "confirmed_physics_steps": local_steps, "step_accounting_exact": True,
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
        self.assertAlmostEqual(budget.search_seconds, 2.)
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
        self.assertEqual(mapping.event_clock[("current", "close", 0)]["activation_time_s"], .65)
        self.assertEqual(mapping.event_clock[("current", "close", 0)]["request_intent_time_s"], .9)

    def test_continuation_resource_change_retains_original_work_and_physics(self):
        from manipulation.planning import plan_trajectory

        problem = self.problem((PlanningEvent("close", "current", .9, 1, "close", "task"),))
        continuation = self.continuation(problem)
        original = deepcopy(continuation)
        requested = replace(PlanningBudget(**continuation["original_binding"]["planning_budget"]),
            max_host_rss_bytes=12*1024**3, max_device_used_bytes=8*1024**3,
            max_scene_gpu_workspace_bytes=3*1024**3)
        scene = SimpleNamespace(clone_for_rollout=lambda: SimpleNamespace(
            bind_robot_validator=Mock(), gripper=None))
        with patch("manipulation.planning._CuroboPlanner", side_effect=RuntimeError("native allocation refused")) as native:
            first = plan_trajectory(problem, scene, {}, robot_validator=object(),
                budget=requested, continuation=continuation)
            continued = deepcopy(continuation)
            continued["request_ledger"] = deepcopy(first.request_ledger)
            second = plan_trajectory(problem, scene, {}, robot_validator=object(), continuation=continued)
        self.assertEqual(native.call_count, 1)
        self.assertEqual(continuation, original)
        for call in native.call_args_list:
            admitted = call.args[3]
            self.assertEqual(admitted.max_host_rss_bytes, 12*1024**3)
            self.assertEqual(admitted.max_device_used_bytes, 8*1024**3)
            self.assertEqual(admitted.maximum_trajectory_solves, 2)
            self.assertAlmostEqual(admitted.search_seconds, 1.3)
            self.assertEqual(call.kwargs["continuation"]["consumed_physics_steps"], 350)
            self.assertEqual(call.kwargs["continuation"]["remaining_physics_steps"], 650)
        self.assertEqual(second.request_ledger["request_id"], original["original_binding"]["request_ledger"]["request_id"])
        changes = [row for row in second.request_ledger["binding_records"] if row["mode"] == "resource_limits_changed"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(second.request_ledger["consumed_planning"]["trajectory_solves"], 2)
        self.assertEqual(second.request_ledger["original_events"], original["original_binding"]["request_ledger"]["original_events"])
        self.assertEqual(second.request_ledger["binding_records"][:len(first.request_ledger["binding_records"])],
            first.request_ledger["binding_records"])

    def test_continuation_separates_charges_from_exact_confirmed_steps(self):
        request = self.problem((PlanningEvent("close", "current", .9, 1, "close", "task"),))
        retained = self.continuation(request, elapsed=0.)
        retained.update(consumed_physics_steps=1, cumulative_physics_steps=101, remaining_physics_steps=899)
        _, prior = _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, retained)
        self.assertEqual(prior["elapsed"], 0.)
        self.assertEqual(prior["elapsed_device_steps"], 0)
        self.assertEqual(prior["consumed_physics_steps"], 101)
        self.assertEqual(prior["remaining_physics_steps"], 899)
        for key, value in (("step_accounting_exact", False), ("step_accounting_exact", None),
                           ("confirmed_physics_steps", None), ("confirmed_physics_steps", 2)):
            changed = deepcopy(retained); changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ContinuationBindingError):
                _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, changed)

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
        continuation["applied_events"] = [{"skill_id": "current", "event_id": "earlier", "occurrence": 0}]
        _, prior = _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        self.assertEqual([event.event_id for event in problem.events], ["later"])
        self.assertEqual(prior["consumed_events"], [("current", "earlier", 0)])

    def test_unknown_ordinary_effect_evaluation_does_not_block_remaining_motion(self):
        events = (PlanningEvent("observe", "current", .1, 1, "effect", "task"),
                  PlanningEvent("later", "current", .9, 2, "open", "task"))
        problem = self.problem(events)
        continuation = self.continuation(problem)
        continuation["evaluated_events"] = [{"skill_id": "current", "event_id": "observe", "occurrence": 0,
            "kind": "effect", "time": .1, "evaluation_time_s": .1, "evaluation": "U",
            "effects": {}, "source": "current scene has no conclusive effect observation"}]
        continuation["effects"] = {"Holding": {"value": "U", "source": "occluded"}}
        _, prior = _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        self.assertEqual([event.event_id for event in problem.events], ["later"])
        self.assertEqual(prior["effects"]["Holding"]["value"], "U")
        self.assertEqual(prior["consumed_events"], [("current", "observe", 0)])

    def test_continuation_keeps_future_pose_anchors_without_claiming_past_success(self):
        for claimed in (False, True):
            problem = self.problem()
            problem.pose_events = tuple(PoseEventRequirement(progress, np.eye(4), np.eye(4), .01, .02, "goal")
                                        for progress in (.1, .9))
            continuation = self.continuation(problem)
            if claimed:
                continuation["completed_goal_ids"] = [["pose_events", 0]]
                continuation["goal_evidence"] = [{"identity": ["pose_events", 0], "value": "F"}]
            with self.subTest(claimed=claimed):
                _, prior = _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
            self.assertEqual(prior["goal_identities"]["pose_events"], [1])
            self.assertEqual(problem.pose_events[0].progress, .9)
            self.assertEqual(prior["completed_goal_ids"], [])

    def test_current_owner_observation_is_not_promoted_to_online_pose_success(self):
        from manipulation.planning import _verified_completed_goals
        from test_recorded_simulation import ActualPoseGoalTests
        fixture = ActualPoseGoalTests.fixture(progress=0., actual_relation={
            "target_owner_id": "current_part", "T_owner_target": np.eye(4).tolist()})
        fixture[3].execution_binding["request_ledger"] = {
            "completed_goal_ids": [], "goal_evidence": [], "timing_events": []}
        def observe(feedback, curve_time):
            return {"source": "Synthetic current owner observation", "timestamp_s": feedback["timestamp_s"],
                "scene_owner_poses": {"current_part": np.eye(4).tolist()}}
        report = ActualPoseGoalTests.run_trial(fixture, observe=observe)
        self.assertEqual(report["goal_evidence"], [])
        continuation = report["continuation"]
        self.assertEqual(_verified_completed_goals(continuation, continuation["original_binding"],
            {("pose_events", 7)}, elapsed=continuation["elapsed_physical_s"],
            curve_elapsed=continuation["last_curve_time_s"]), set())

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

    def test_native_ompl_returns_a_bounded_ordinary_joint_connection(self):
        checked = []
        def valid(state):
            checked.append(state.copy())
            return True
        path, report = _ompl_stage(np.array([0., 0.]), np.array([1., 1.]),
            np.array([-2., -2.]), np.array([2., 2.]), valid,
            step=.2, seconds=1., sample_limit=256)
        self.assertIsNotNone(path)
        np.testing.assert_allclose(path[0], [0., 0.])
        np.testing.assert_allclose(path[-1], [1., 1.])
        self.assertLessEqual(len(path), 256)
        self.assertEqual(report["strategy"], "native_rrt_connect")
        self.assertGreater(len(checked), 0)


    def test_ordinary_scene_query_preserves_samples_without_precision_supplements(self):
        progress = torch.tensor([[0., .5, 1.]])
        q = torch.zeros((1, 3, 1))
        components = Mock(return_value={"collision/robot/background": torch.zeros_like(progress)})
        scene = SimpleNamespace(components=components)
        value = _known_scene_cost(scene,
            torch.zeros((1, 3, 1, 4)), torch.zeros((1, 3, 3)), torch.ones((1, 3, 4)),
            progress, q, ("q",), trajectory_time=progress)
        components.assert_called_once()
        torch.testing.assert_close(components.call_args.args[3], progress)
        torch.testing.assert_close(value, torch.zeros_like(progress))


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
            state=lambda *args, **kwargs: SimpleNamespace(coordinates=None, half=None,
                uncertain=kwargs["trajectory_time"] > 0), spheres=lambda values, *args: values)
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
# END native attempt-planner test migration.
