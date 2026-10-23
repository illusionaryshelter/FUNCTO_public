"""CPU checks of source timing, activation and functional IK goal proposals."""

# BEGIN frozen binding and target generation contracts independent of CUDA.
from types import SimpleNamespace
from dataclasses import asdict, replace
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from manipulation.planning import (PlanningBudget, _ConstrainedPlanner, _ProgressMap,
    ContinuationBindingError, _bind_event_clock, _initial_timing,
    _prepare_continuation, _continuation_timing, _fixed_spline_grid,
    _ompl_stage, _scene_boundary_evaluate, reference_progress_clock)
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

    # BEGIN source duration owns timing; output spacing cannot stretch it to 16 seconds.
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

    def test_selected_source_clock_binds_events_and_complete_nominal_fingers_together(self):
        from manipulation.planning import JointTrajectory
        from manipulation.planning_scene import PrismaticGripper

        event = PlanningEvent("close", "current", .8, 183, "close", "transferred reference command")
        problem = self.problem((event,), source_times=(0., 9.15, 9.85), source_progress=(0., .8, 1.))
        budget, mapping = _initial_timing(problem, PlanningBudget(knots=16, interpolation_buffer_size=768))
        # BEGIN bind the same independent reference schedule as the planner.
        mapping, _ = reference_progress_clock(mapping, budget.knots,
            stop_progress=problem.stop_progress, anchors=problem.anchors)
        # END arm coefficients retain their own measured entrance constraints.
        # BEGIN the shared fixed grid binds the task and complete hand model.
        grid = _fixed_spline_grid(intervals=budget.knots+4,
            native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
            interpolation_dt=budget.interpolation_dt)
        # END native IK and CPU trajectory consumers receive the same clock.
        mapping = _bind_event_clock(problem, mapping, grid["duration"], .002)
        seen = []

        def fingers(duration, progress_map):
            seen.append((duration, progress_map))
            return JointTrajectory(("left_finger", "right_finger"), np.array([0., duration]),
                np.tile([.04, -.03], (2, 1)), np.zeros((2, 2)),
                progress_map.u, progress_map.s, (), (), {})

        # BEGIN geometry explicitly declares this constant fixture's clock scope.
        def geometry(duration, progress_map, command_reference):
            from manipulation.gripper_trajectory import NominalFingerGeometry
            return NominalFingerGeometry.from_reference(command_reference,
                valid_until_s=duration, source="explicit whole-clock constant fixture")
        # END matching command times alone do not authorize nominal geometry.

        hand = PrismaticGripper(controller="timed_reference", control_timestep_s=.002,
            joint_names=("left_finger", "right_finger"), current_positions=[.04, -.03],
            joint_limits=[[0., .05], [-.05, 0.]], joint_bounds=None,
            finger_trajectory_factory=fingers, nominal_finger_trajectory_factory=geometry,
            # BEGIN the current hand interface binds clocks and joint coordinates.
            coupling_matrix=np.empty((0, 2)), coupling_offset=[],
            commands=[{"skill_id": "current", "event_id": "close", "occurrence": 0,
                "command": "close", "progress": .8}], source="explicit CPU clock contract")
            # END robot collision entities are supplied by their geometry component.
        hand.bind_clock(grid["duration"], mapping)
        self.assertTrue(all(duration == grid["duration"] and actual_map is mapping for duration, actual_map in seen))
        self.assertEqual(hand.finger_trajectory.duration, hand.nominal_finger_trajectory.duration)
        self.assertEqual(hand.finger_trajectory.duration, grid["duration"])
        record = mapping.event_clock[("current", "close", 0)]
        self.assertEqual(record["activation_time_s"], mapping.anchor_parameters([.8])[0, 0]*grid["duration"])
        self.assertEqual(record["activation_time_s"], record["bound_activation_step"]*.002)
        self.assertEqual(grid["duration"], (grid["valid_count"]-1)*grid["interpolation_dt"])
        self.assertEqual(grid["duration"], (budget.knots+4)*budget.interpolation_steps*grid["native_dt"])
        # BEGIN the CPU solver consumes both nominal hand columns on that clock.
        from manipulation.unified_solver import robot_tool_states
        captured = []
        def states(q, dq, names):
            captured.append((q, dq, names))
            zero = q[..., 0] * 0
            position = torch.stack((zero, zero, zero), -1)
            quaternion = torch.stack((zero + 1, zero, zero, zero), -1)
            return {"tool": (position, quaternion)}, {"tool": (position, position)}
        times = torch.tensor([[0., grid["duration"] / 2, grid["duration"]]], dtype=torch.float64)
        progress = torch.as_tensor(np.interp(times.numpy(), mapping.u * grid["duration"], mapping.s))
        arm = torch.zeros((1, 3, 1), dtype=torch.float64)
        robot_tool_states(problem=problem, scene=SimpleNamespace(gripper=hand),
            kinematics=SimpleNamespace(states=states), q=arm, dq=arm,
            joint_names=("q",), progress=progress, times=times)
        self.assertEqual(captured[0][2], ("q", "left_finger", "right_finger"))
        np.testing.assert_allclose(captured[0][0][0, :, 1:].numpy(),
                                    np.tile([.04, -.03], (3, 1)), atol=1e-15)
        self.assertEqual(torch.count_nonzero(captured[0][1]).item(), 0)
        # END command and nominal geometry share actual physical-time samples.
    # END no extra retiming optimizer or relaxed interpolation equality is introduced.

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
        self.assertEqual(mapping.s[0], .4)
        self.assertEqual(mapping.s[-1], 1.)
        self.assertFalse(np.any(mapping.s < .4))

    # BEGIN true goals guide IK while every trajectory goal keeps its original owner.
    def test_functional_ik_goal_uses_actual_target_in_native_link_coordinates(self):
        problem = self.problem()
        target = np.eye(4); target[0, 3] = .8
        function = np.eye(4); function[0, 3] = .1
        goal = PoseEventRequirement(.5, function, target, .01, .02, "true goal")
        problem.pose_events = (goal,)
        planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
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
            [{"task_satisfied": flag, "model_valid": flag, "native_pose_ik_success": flag}
             for flag in [True, True, False]],
            np.array([[0., 0.]]), count=1, joint_tolerance=1e-6, query_kind="reference")
        np.testing.assert_array_equal(selected, [1])

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

    # BEGIN resident rebinding replaces one task clock and its nominal hand together.
    def test_resident_rebind_preserves_ik_and_replaces_entrance_clock_and_scene(self):
        from manipulation.gripper_trajectory import NominalFingerGeometry
        from manipulation.planning import JointTrajectory
        from manipulation.planning_scene import PrismaticGripper

        event = PlanningEvent("close", "instance", .4, 2, "close", "declared command")
        problem = self.problem((event,), source_times=(0., .8, 2.), source_progress=(0., .4, 1.))
        problem.start_q[:], problem.start_dq[:] = .1, .03
        calls = []
        def fingers(duration, mapping):
            calls.append((duration, mapping))
            return JointTrajectory(("finger",), np.array([0., duration]), np.full((2, 1), .04),
                np.zeros((2, 1)), mapping.u, mapping.s, (), (), {})
        def nominal(duration, mapping, command):
            return NominalFingerGeometry.from_reference(command, valid_until_s=duration,
                                                        source="declared complete hand model")
        hand = PrismaticGripper(controller="timed_reference", control_timestep_s=.002,
            joint_names=("finger",), current_positions=[.04], joint_limits=[[0., .05]],
            joint_bounds=None, finger_trajectory_factory=fingers,
            nominal_finger_trajectory_factory=nominal,
            coupling_matrix=np.empty((0, 1)), coupling_offset=[],
            commands=[dict(skill_id="instance", event_id="close", occurrence=0,
                           command="close", progress=.4)], source="declared hand command")
        scene = SimpleNamespace(gripper=hand, required_progress_anchors=(.6,), validate_problem=Mock())
        scene.clone_for_rollout = lambda: SimpleNamespace(gripper=deepcopy(hand))
        planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
        planner.robot_validator = SimpleNamespace(validate_planning_binding=Mock())
        planner.robot_metadata, planner.anchor_capacity = {}, 256
        planner.torch, planner.device_cfg = torch, SimpleNamespace(device="cpu")
        shared_clock = {"obsolete": True}
        planner.formal_clock, planner.joint_names = shared_clock, ("q",)
        planner.JointState = SimpleNamespace(from_numpy=lambda names, positions, *, velocity, device_cfg:
            SimpleNamespace(joint_names=names, position=torch.as_tensor(positions),
                            velocity=torch.as_tensor(velocity)))
        planner._bind_online_robot, planner._bind_physical_domains = Mock(), Mock()
        planner._manager_type = SimpleNamespace(bind_request=Mock())
        planner.seed_parameter = torch.ones((1, 1))
        old_goal = SimpleNamespace(link_goal_poses=object(), goal_js=object())
        rollout = SimpleNamespace(reset=Mock(), _num_particles_goal=old_goal,
                                  _metrics_goal=SimpleNamespace(link_goal_poses=object(), goal_js=object()))
        registry = SimpleNamespace(_goal_buffer=SimpleNamespace(link_goal_poses=object()))
        resident = SimpleNamespace(reset_seed=Mock(), core=SimpleNamespace(
            goal_registry_manager=registry, get_all_rollout_instances=lambda: (rollout,)))
        planner.ik = resident
        identities = {"pose_events": []}
        entrance = np.eye(4); entrance[0, 3] = .1
        with patch("manipulation.planning._planning_workspace_bytes", return_value=SimpleNamespace(pending_cpu_bytes=0)), \
             patch("manipulation.planning._require_planning_host_memory"), \
             patch("manipulation.planning._require_planning_memory"), \
             patch("manipulation.planning._reference_entrance", return_value=(entrance, np.array([.03, 0., 0., 0., 0., 0.]))):
            result = planner.rebind(problem, scene, PlanningBudget(), goal_identities=identities)
        self.assertIs(result, planner)
        self.assertIs(planner.ik, resident)
        self.assertIs(planner.formal_clock, shared_clock)
        self.assertNotIn("obsolete", shared_clock)
        self.assertEqual(calls, [(shared_clock["duration"], planner.progress_map)])
        self.assertEqual(planner._verification_scene.gripper.nominal_finger_trajectory.duration,
                         shared_clock["duration"])
        self.assertIs(planner.finger_trajectory, hand.finger_trajectory)
        self.assertIn(.6, planner.anchors)
        # BEGIN a resident request replaces both reference-clock coordinates.
        self.assertIsNotNone(planner.reference_time_map)
        np.testing.assert_array_equal(planner.source_progress_map.u, [0., .4, 1.])
        self.assertEqual(len(planner.progress_map.anchor_parameters([1.])), 2)
        # END the terminal progress interval is scheduling data only.
        np.testing.assert_array_equal(planner.start.position.numpy(), [[.1]])
        np.testing.assert_array_equal(planner.start.velocity.numpy(), [[.03]])
        scene.validate_problem.assert_called_once_with(problem, goal_identities=identities)
        planner._manager_type.bind_request.assert_called_once_with(problem, planner.budget)
        planner._bind_online_robot.assert_called_once()
        planner._bind_physical_domains.assert_called_once()
        resident.reset_seed.assert_called_once()
        rollout.reset.assert_called_once()
        self.assertIsNone(registry._goal_buffer.link_goal_poses)
        self.assertIsNone(old_goal.link_goal_poses)
        self.assertIsNone(old_goal.goal_js)
        self.assertEqual(torch.count_nonzero(planner.seed_parameter).item(), 0)
    # END request replacement retains the resident solver and one physical clock.

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
        with patch("manipulation.planning._ConstrainedPlanner", side_effect=RuntimeError("native allocation refused")) as native:
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

    # BEGIN past nominal pose anchors never require online target truth.
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
        self.assertEqual(continuation["completed_goal_ids"], [])
        self.assertEqual(continuation["goal_evidence"], [])
    # END independent non-pose effects retain their separate evidence requirements.

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
        # BEGIN the material-mode fixture consumes the explicit geometry scope.
        def evaluate(spheres, position, quaternion, s, joints, names, *, trajectory_time,
                     contact_permission_progress, geometry_evidence_out,
                     geometry_scope):
            self.assertEqual(geometry_scope, "full")
            observations.append((s.clone(), joints.clone(), trajectory_time.clone()))
            return ((s < .5) & (joints[..., 0] > 0)).to(s.dtype)
        scene = SimpleNamespace(required_progress_anchors=(.5,), contacts=(), evaluate=evaluate)
        # END the test proves both original material-mode sides in full geometry.
        def check(joints):
            return _scene_boundary_evaluate(scene, self.problem(), mapping,
                torch.zeros((1, 3, 1, 4)), torch.zeros((1, 3, 3)),
                torch.tensor([[[1., 0., 0., 0.]]]).expand(1, 3, 4),
                progress, joints, ("q",), trajectory_time=clock, normalized_time=clock)
        self.assertTrue(bool((check(q) == 0).all()))
        sides, positions, times = observations[-1]
        # BEGIN only the plateau entrance/exit have distinct side queries.
        self.assertEqual(sides.shape[1], 5)
        self.assertLess(float(sides[0, 3]), .5)
        self.assertGreater(float(sides[0, 4]), .5)
        selected = torch.tensor([0, 1, 2, 0, 2])
        torch.testing.assert_close(positions, q.index_select(1, selected), rtol=0., atol=0.)
        torch.testing.assert_close(times, clock.index_select(1, selected), rtol=0., atol=0.)
        # END all original exact states remain present at unchanged q/time.
        q[:, 0] = .1
        self.assertGreater(float(check(q)[0, 0]), 0.)

    # BEGIN remove the retired timed-sphere adapter regression.
    # Native robot geometry now uses its declared joint-state/shape consumers;
    # _timed_native_state and its radius-masking interface no longer exist.
    # END current hand geometry contracts are covered by their component tests.


if __name__ == '__main__':
    unittest.main()
# END these tests claim clock/goal contracts, not physical execution success.
