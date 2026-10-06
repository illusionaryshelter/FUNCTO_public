"""CPU checks of source timing, activation and functional IK goal proposals."""

# BEGIN frozen binding and target generation contracts independent of CUDA.
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
    _ompl_stage, _scene_boundary_evaluate, _timed_native_state,
    _progress_for_cubic_boundaries)
from manipulation.planning_ik import rank_ik_candidates
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
        warped, _ = _progress_for_cubic_boundaries(mapping, 8, [0.], stop_progress=(), anchors=[.4, 1.])
        self.assertEqual(warped.s[0], .4)
        self.assertEqual(warped.s[-1], 1.)
        self.assertFalse(np.any(warped.s < .4))

    # BEGIN true goals guide IK while every trajectory goal keeps its original owner.
    def test_functional_ik_goal_uses_actual_target_in_native_link_coordinates(self):
        problem = self.problem()
        target = np.eye(4); target[0, 3] = .8
        function = np.eye(4); function[0, 3] = .1
        goal = PoseEventRequirement(.5, function, target, .01, .02, "true goal")
        problem.pose_events = (goal,)
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.problem, planner.torch = problem, torch
        planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float32))
        planner.GoalToolPose = SimpleNamespace
        native = planner._functional_goal(goal)
        np.testing.assert_allclose(native.position.numpy().reshape(3), [.7, 0., 0.], atol=1e-7)
        # Changing the reference path cannot move the actual native goal.
        problem.phases[0].reference.poses[:, 0, 3] = -.8
        torch.testing.assert_close(planner._functional_goal(goal).position, native.position)
        goal.target[0, 3] = .5
        self.assertAlmostEqual(float(planner._functional_goal(goal).position.flatten()[0]), .4, places=6)
        planner.progress_map = _ProgressMap(np.array([0., .8, 1.]), np.array([0., 1., 1.]))
        torch.testing.assert_close(planner._functional_goal(goal).position,
                                   torch.tensor([[[[[.4, 0., 0.]]]]]))

    def test_nonfinite_single_ik_seed_does_not_hide_a_valid_success(self):
        candidates = np.array([[np.nan, 0.], [1., 2.], [3., 4.]])
        selected = rank_ik_candidates(candidates,
            [{"task_satisfied": flag, "source_valid": flag, "native_pose_ik_success": flag}
             for flag in [True, True, False]],
            np.array([[0., 0.]]), count=1, joint_tolerance=1e-6, query_kind="reference")
        np.testing.assert_array_equal(selected, [1])

    def test_both_trajectory_phases_disable_duplicate_native_pose_objective(self):
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.problem = self.problem()
        planner.problem.pose_events = (PoseEventRequirement(1., np.eye(4), np.eye(4), .001, .02, "goal"),)
        planner.physical_domain = SimpleNamespace(validate=Mock())
        native = SimpleNamespace(solution=torch.zeros((1, 1, 4, 1)))
        planner.trajopt = SimpleNamespace(disable_tool_pose_tracking=Mock(),
            enable_tool_pose_tracking=Mock(), solve_pose=Mock(return_value=native),
            core=SimpleNamespace(optimizers=[SimpleNamespace(_core=SimpleNamespace(
                solve_diagnostics=[{"exhausted": False, "effective_updates": 0}]))]))
        planner.torch, planner.budget, planner.start = torch, PlanningBudget(), object()
        metadata = object()
        planner._goal = Mock(return_value=metadata)
        for phase in ("feasibility", "quality"):
            planner.phase = phase
            self.assertIs(planner.optimize(torch.zeros((1, 1, 4, 1))), native)
            self.assertIs(planner.trajopt.solve_pose.call_args.kwargs["goal_tool_poses"], metadata)
        self.assertEqual(planner.trajopt.disable_tool_pose_tracking.call_count, 2)
        planner.trajopt.enable_tool_pose_tracking.assert_not_called()
    # END full functional tolerances are checked by typed process residuals only.

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
        budget = PlanningBudget(maximum_trajectory_solves=4, search_seconds=2.)
        original = {"candidate_id": "candidate", "planning_budget": asdict(budget),
                    "consumed_planning": {"binding_attempts": 1, "initializations": 1, "ik_batches": 1,
                        "ik_queries": 24, "search_queries": 0, "validation_passes": 2, "validation_queries": 20,
                        "trajectory_solves": 2, "search_seconds": .7, "planning_seconds": 1.},
                    "physics_budget": 1000, "consumed_physics_steps": 100,
                    # BEGIN a real continuation retains all original task geometry.
                    "problem": plain(problem),
                    # END pose completion is checked against this frozen binding.
                    "curve": {"duration": 1., "progress_u": [0., .5, 1.],
                              "progress_s": [0., .8, 1.], "events": rows, "executable_until": 1.}}
        from manipulation.planning import _request_work_ledger
        ledger = _request_work_ledger(problem, budget)
        ledger["consumed_planning"] = deepcopy(original["consumed_planning"])
        original["request_ledger"] = ledger
        return {"candidate_id": "candidate", "original_binding": original,
                "physics_budget": 1000, "cumulative_physics_steps": 100+local_steps,
                "remaining_physics_steps": 900-local_steps, "consumed_physics_steps": local_steps,
                # BEGIN successful synthetic native steps have an explicit exact clock.
                "confirmed_physics_steps": local_steps, "step_accounting_exact": True,
                # END charges alone cannot advance a continuation's original time.
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
        self.assertEqual(mapping.event_clock[("current", "close", 0)]["activation_time_s"], .65)
        self.assertEqual(mapping.event_clock[("current", "close", 0)]["request_intent_time_s"], .9)

    # BEGIN resource changes coexist with an older frozen execution binding.
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
        self.assertEqual(native.call_count, 2)
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
    # END resource changes cannot manufacture new physical time or solver work.

    # BEGIN a paid but stationary attempt never advances the physical clock.
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
    # END an unknown count cannot masquerade as an exact zero-step continuation.

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

    def test_continuation_requires_past_goal_proof_and_preserves_original_indices(self):
        problem = self.problem()
        problem.pose_events = tuple(PoseEventRequirement(progress, np.eye(4), np.eye(4), .01, .02, "goal")
                                    for progress in (.1, .9))
        continuation = self.continuation(problem)
        with self.assertRaisesRegex(ContinuationBindingError, "Past goal pose_events:0"):
            _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        continuation["completed_goal_ids"] = [["pose_events", 0]]
        # BEGIN an ID alone cannot remove an independently required pose goal.
        continuation["actual_feedback"]["timestamp_s"] = .25
        with self.assertRaisesRegex(ContinuationBindingError, "actual allowed-set T evidence"):
            _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        from manipulation.execution_binding import plain
        continuation["goal_evidence"] = [{"identity": ["pose_events", 0], "value": "T",
            "candidate_id": "candidate", "completion_verified": True, "source": "actual fixed-frame measurement",
            "feedback_id": "measured-at-first-native-tick", "timestamp_s": .063, "physical_elapsed_s": .063,
            "evaluated_curve_time_s": .063, "scheduled_curve_time_s": .0625, "evaluation_window_s": None,
            "device_due_time_s": .063,
            "requirement": plain(problem.pose_events[0]), "actual_function_pose": np.eye(4).tolist(),
            "actual_target_pose": np.eye(4).tolist(), "position_error_m": 0., "rotation_error_rad": 0.,
            "measurement_sources": [{"kind": "actual_robot_tcp", "source": "synchronized actual TCP",
                # BEGIN continuation retains the online model-estimate provenance.
                "pose_source": "forward_kinematics_from_measured_arm_q", "independent_observation": False,
                # END this fixture supplies no independently sensed TCP pose.
                "feedback_id": "measured-at-first-native-tick", "timestamp_s": .063,
                "pose": np.eye(4).tolist()}]}]
        # END the nonuniform source map reaches s=.1 at t=.0625, then measures t=.063.
        _, prior = _prepare_continuation(problem, SimpleNamespace(gripper=None), None, continuation)
        self.assertEqual(prior["goal_identities"]["pose_events"], [1])
        self.assertEqual(problem.pose_events[0].progress, .9)

    # BEGIN continuation preserves the same current-state freshness as execution.
    def test_executor_prebinding_owner_or_holding_sample_keeps_its_valid_pose_proof(self):
        from manipulation.execution import ExecutionLimits
        from manipulation.planning import _verified_completed_goals
        from test_recorded_simulation import ActualPoseGoalTests

        relations = ({"target_owner_id": "current_part", "T_owner_target": np.eye(4).tolist()},
            {"function_owner_id": "held_part", "T_owner_function": np.eye(4).tolist(),
             "attachment_condition_id": "current_holding"})
        for relation in relations:
            with self.subTest(relation=relation):
                simulation, curve, targets, result, limits = ActualPoseGoalTests.fixture(
                    progress=0., actual_relation=relation)
                policy = limits.describe()
                policy["observation_policy"]["maximum_age_s"] = .001
                limits = ExecutionLimits.from_mapping(policy)
                # The synthetic executor fixture needs only the original
                # empty fact ledger for this concrete proof handoff. The
                # production executor constructs every new completion row.
                result.execution_binding["request_ledger"] = {
                    "completed_goal_ids": [], "goal_evidence": [], "timing_events": []}
                def observe(feedback, curve_time):
                    return {"source": "Current scene sample acquired half a millisecond before feedback",
                        "timestamp_s": feedback["timestamp_s"]-.0005,
                        "scene_owner_poses": {"current_part": np.eye(4).tolist()},
                        "scene_condition_evidence": {"current_holding": True}}
                report = ActualPoseGoalTests.run_trial(
                    (simulation, curve, targets, result, limits), observe=observe)
                self.assertEqual(report["pose_goal_result"], "T")
                continuation = report["continuation"]
                original = continuation["original_binding"]
                proof = continuation["goal_evidence"][0]
                scene_source = next(row for row in proof["measurement_sources"]
                    if row["kind"] in ("current_owner_pose", "current_verified_attachment"))
                self.assertAlmostEqual(proof["timestamp_s"], 1.3)
                self.assertAlmostEqual(scene_source["timestamp_s"], 1.2995)
                options = dict(elapsed=continuation["elapsed_physical_s"],
                    curve_elapsed=continuation["last_curve_time_s"], timestep=simulation.timestep,
                    progress_map=_ProgressMap(curve.progress_u, curve.progress_s))
                self.assertEqual(_verified_completed_goals(continuation, original, {("pose_events", 7)}, **options),
                                 {("pose_events", 7)})
                # Fresh state may predate this binding, but a future sample
                # or a stale sample still cannot certify the same actual goal.
                for acquisition in (1.3005, 1.298):
                    forged = deepcopy(continuation)
                    source = next(row for row in forged["goal_evidence"][0]["measurement_sources"]
                        if row["kind"] in ("current_owner_pose", "current_verified_attachment"))
                    source["timestamp_s"] = acquisition
                    with self.subTest(acquisition=acquisition), self.assertRaisesRegex(
                            ContinuationBindingError, "actual allowed-set T evidence"):
                        _verified_completed_goals(forged, original, {("pose_events", 7)}, **options)
    # END event/effect occurrence eligibility remains a separate original-clock check.

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
        # BEGIN the unchanged material-mode fixture declares no arrival pair permission.
        def evaluate(spheres, position, quaternion, s, joints, names, *, trajectory_time,
                     robot_owner_permission_progress):
            observations.append((s.clone(), joints.clone(), trajectory_time.clone()))
            return ((s < .5) & (joints[..., 0] > 0)).to(s.dtype)
        scene = SimpleNamespace(required_progress_anchors=(.5,), robot_owner_permissions=(), evaluate=evaluate)
        # END the existing test still proves both original material-mode sides.
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
