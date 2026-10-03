"""Physical-time and retry contracts with CPU-only independent witnesses."""

# BEGIN explicit timing, source-preserving rebinding and non-maintained point goals.
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from manipulation.execution_binding import plain
from manipulation.motion import MotionModel
from manipulation.planning import (PlanningBudget, ContinuationBindingError, _ProgressMap,
    _bind_event_clock, _continuation_timing, _prepare_continuation, _request_work_ledger,
    _validate_bound_event_times, plan_trajectory)
from manipulation.planning_problem import (EventTimeConstraint, TimeBindingError, MotionPhase,
    PlanningEvent, PlanningProblem, PoseEventRequirement, PosePath, PoseProcessRequirement,
    CoordinateDirectionRequirement, CoordinateEventRequirement, MechanismRequirement,
    event_occurrence_key, resolve_event_window)


def problem(events=(), goals=()):
    path = PosePath(np.array([0., 1.]), np.stack([np.eye(4), np.eye(4)]), "original reference")
    return PlanningProblem(("q",), [0.], [0.], "tool", "world", np.eye(4), np.eye(4),
        (MotionPhase("source", path),), tuple(events), pose_events=tuple(goals),
        source_timestamps=np.array([10., 11.]), source_progress=np.array([0., 1.]))


def continuation(request, *, elapsed=.4, curve_time=.4, applied=(), completed=()):
    """One previous native binding plus actual execution facts, independently built."""
    budget = PlanningBudget(maximum_trajectory_solves=4)
    ledger = _request_work_ledger(request, budget)
    ledger["consumed_planning"].update(binding_attempts=1, initializations=1, ik_batches=1,
        ik_queries=8, trajectory_solves=1, validation_passes=1, validation_queries=33, planning_seconds=.1)
    rows = [{**plain(event), "time": event.progress, "intent_time_s": event.progress,
             "activation_time_s": event.progress if event.kind in {"close", "open", "hold"} else None}
            for event in request.events]
    for row in rows:
        if row["activation_time_s"] is not None:
            row.update(bound_activation_step=round(row["time"]/.01),
                request_activation_step=round(row["time"]/.01), request_activation_time_s=row["time"],
                request_intent_time_s=row["time"], request_time_origin_step=0, device_timestep_s=.01)
    applied = set(applied)
    original = {"candidate_id": "previous", "planning_budget": asdict(budget),
        "consumed_planning": deepcopy(ledger["consumed_planning"]), "request_ledger": ledger,
        "physics_budget": 1000, "consumed_physics_steps": 0,
        # BEGIN preserve the original functional allowed sets for actual proof checking.
        "problem": plain(request), "request_events": plain(request.events),
        # END completion IDs alone are not execution evidence.
        "curve": {"duration": 1., "progress_u": [0., 1.], "progress_s": [0., 1.],
                  "events": rows, "executable_until": 1.}}
    return {"candidate_id": "previous", "original_binding": original,
        "physics_budget": 1000, "cumulative_physics_steps": round(elapsed/.01),
        "remaining_physics_steps": 1000-round(elapsed/.01), "consumed_physics_steps": round(elapsed/.01),
        # BEGIN this timed-evidence fixture declares its confirmed native steps.
        "confirmed_physics_steps": round(elapsed/.01), "step_accounting_exact": True,
        # END physical event windows never use attempted-work charges as time.
        "device_timestep_s": .01, "elapsed_physical_s": elapsed, "last_curve_time_s": curve_time,
        "actual_feedback": {"joint_names": ["q"], "q": [0.], "dq": [0.], "timestamp_s": elapsed},
        "remaining_events": [row for row in rows if event_occurrence_key(row) not in applied],
        "applied_events": [row for row in rows if event_occurrence_key(row) in applied],
        "completed_goal_ids": [list(value) for value in completed],
        # BEGIN these fixtures explicitly measured identity poses at their point goals.
        "goal_evidence": [{"identity": list(value), "value": "T", "candidate_id": "previous",
            "completion_verified": True, "source": "actual fixed-frame task measurement",
            "feedback_id": "task-feedback-"+str(value[1]),
            "timestamp_s": request.pose_events[value[1]].progress,
            "physical_elapsed_s": request.pose_events[value[1]].progress,
            "evaluated_curve_time_s": request.pose_events[value[1]].progress,
            "scheduled_curve_time_s": request.pose_events[value[1]].progress, "evaluation_window_s": None,
            "device_due_time_s": request.pose_events[value[1]].progress,
            "requirement": plain(request.pose_events[value[1]]),
            "actual_function_pose": np.eye(4).tolist(), "actual_target_pose": np.eye(4).tolist(),
            "position_error_m": 0., "rotation_error_rad": 0., "measurement_sources": [{
                "kind": "actual_robot_tcp", "source": "actual TCP", "pose": np.eye(4).tolist(),
                # BEGIN preserve the explicit FK-estimate source through continuation.
                "pose_source": "forward_kinematics_from_measured_arm_q", "independent_observation": False,
                # END the scripted identity pose is not independent sensing.
                "feedback_id": "task-feedback-"+str(value[1]),
                "timestamp_s": request.pose_events[value[1]].progress}]}
            for value in completed],
        # END timestamps are independent facts, not inferred source progress at runtime.
        "history": [{"skill_id": row["skill_id"], "event_id": row["event_id"], "occurrence": row["occurrence"],
                     "check_target": "activation", "time_s": row["time"], "value": "T", "source": "actual controller"}
                    for row in rows if event_occurrence_key(row) in applied]}


class PhysicalTimeRequirementsTest(unittest.TestCase):
    # BEGIN a continuation cannot convert labels or altered measurements into task completion.
    def test_pose_completion_rechecks_original_allowance_clock_and_candidate(self):
        goal = PoseEventRequirement(.2, np.eye(4), np.eye(4), .001, .01, "original narrow task goal")
        request = problem(goals=(goal,))
        actual = continuation(request, elapsed=.4, completed=(("pose_events", 0),))
        mutations = {
            "label only": lambda row: row.pop("goal_evidence"),
            "wrong candidate": lambda row: row["goal_evidence"][0].update(candidate_id="different"),
            "not verified": lambda row: row["goal_evidence"][0].update(completion_verified=False),
            "missing measurement source": lambda row: row["goal_evidence"][0].update(measurement_sources=[]),
            # BEGIN continuation cannot silently fill in missing pose provenance.
            "missing FK source": lambda row: row["goal_evidence"][0]["measurement_sources"][0].pop("pose_source"),
            "missing observation flag": lambda row: row["goal_evidence"][0]["measurement_sources"][0].pop("independent_observation"),
            # END absence rejects the saved proof without a compatibility path.
            "late acquisition": lambda row: row["goal_evidence"][0].update(
                timestamp_s=.25, physical_elapsed_s=.25, evaluated_curve_time_s=.25),
            "inconsistent timestamp": lambda row: row["goal_evidence"][0].update(timestamp_s=.3),
            "changed tolerance": lambda row: row["goal_evidence"][0]["requirement"].update(position_tolerance_m=.1),
            "invented window": lambda row: row["goal_evidence"][0].update(evaluation_window_s=[0., 1.]),
            "unlisted goal": lambda row: row.update(completed_goal_ids=[["pose_events", 7]]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = deepcopy(actual)
                mutate(changed)
                with self.assertRaises(ContinuationBindingError):
                    _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, changed)
        for recorded_error in (0., .002):
            with self.subTest(recorded_error=recorded_error):
                changed = deepcopy(actual)
                changed["goal_evidence"][0]["actual_function_pose"][0][3] = .002
                changed["goal_evidence"][0]["position_error_m"] = recorded_error
                with self.assertRaisesRegex(ContinuationBindingError, "actual allowed-set T evidence"):
                    _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, changed)

        # A self-consistent relative difference does not authorize moving the
        # original fixed world target together with the measured tool.
        changed = deepcopy(actual)
        for key in ("actual_function_pose", "actual_target_pose"):
            changed["goal_evidence"][0][key][0][3] = 1.
        changed["goal_evidence"][0]["measurement_sources"][0]["pose"][0][3] = 1.
        with self.assertRaisesRegex(ContinuationBindingError, "actual allowed-set T evidence"):
            _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, changed)
        # The following native sample is not the first sample of this point,
        # even though it falls within one device period of the nominal time.
        changed = deepcopy(actual)
        changed["goal_evidence"][0].update(timestamp_s=.21, physical_elapsed_s=.21, evaluated_curve_time_s=.21)
        changed["goal_evidence"][0]["measurement_sources"][0]["timestamp_s"] = .21
        with self.assertRaisesRegex(ContinuationBindingError, "actual allowed-set T evidence"):
            _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, changed)

    def test_pose_completion_requires_current_owner_measurements_for_relative_goals(self):
        for function_observed in (False, True):
            relation = {"target_owner_id": "target", "T_owner_target": np.eye(4)}
            if function_observed:
                relation.update(function_owner_id="actor", T_owner_function=np.eye(4), attachment_condition_id="actual_hold")
            goal = PoseEventRequirement(.2, np.eye(4), np.eye(4), .001, .01, "relative task", actual_relation=relation)
            request = problem(goals=(goal,))
            actual = continuation(request, elapsed=.4, completed=(("pose_events", 0),))
            actual["original_binding"]["execution_limits"] = {"observation_policy": {
                "source": "current synchronized owner observation", "period_s": .01,
                "maximum_age_s": .01, "required_fields": []}}
            with self.subTest(function_observed=function_observed, missing=True):
                with self.assertRaisesRegex(ContinuationBindingError, "actual allowed-set T evidence"):
                    _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, actual)
            owners = ["target", "actor"] if function_observed else ["target"]
            sources = [] if function_observed else deepcopy(actual["goal_evidence"][0]["measurement_sources"])
            sources += [{"kind": "current_owner_pose", "owner_id": owner, "pose": np.eye(4).tolist(),
                         "source": "current RGB-D observation", "timestamp_s": .2} for owner in owners]
            actual["goal_evidence"][0]["measurement_sources"] = sources
            with self.subTest(function_observed=function_observed, missing=False):
                _, result = _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, actual)
                self.assertEqual(result["goal_identities"]["pose_events"], [])
            stale = deepcopy(actual)
            for item in stale["goal_evidence"][0]["measurement_sources"]:
                if item["kind"] == "current_owner_pose":
                    item["timestamp_s"] = .18
            with self.subTest(function_observed=function_observed, stale=True):
                with self.assertRaisesRegex(ContinuationBindingError, "actual allowed-set T evidence"):
                    _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, stale)

    # BEGIN an actual passed dependency can authorize a later goal on the same clock.
    def test_pose_completion_after_verified_dependency_uses_full_original_curve(self):
        barrier = PlanningEvent("dependency", "s", .1, 1, "verify", "declared prerequisite",
                                requires_evidence="Ready")
        goal = PoseEventRequirement(.2, np.eye(4), np.eye(4), .001, .01, "required post-barrier goal")
        request = problem((barrier,), (goal,))
        actual = continuation(request, elapsed=.4, completed=(("pose_events", 0),))
        # The frozen candidate initially authorizes only the .1 prefix. The
        # executor records this dependency as verified before reaching .2;
        # no stop/hold or new physical origin is inserted into this binding.
        actual["original_binding"]["curve"]["executable_until"] = .1
        actual["remaining_events"] = []
        actual["verified_events"] = deepcopy(actual["original_binding"]["curve"]["events"])
        actual["effects"] = {"Ready": {"value": "T", "source": "synchronized actual prerequisite observation"}}
        _, result = _prepare_continuation(request, SimpleNamespace(gripper=None), None, actual)
        self.assertEqual(result["goal_identities"]["pose_events"], [])
        self.assertIn(("pose_events", 0), result["completed_goal_ids"])
        self.assertEqual(result["original_duration"], 1.)
        self.assertEqual(result["elapsed"], .4)
        self.assertIn(("s", "dependency", 0), result["consumed_events"])
    # END the original initial prefix is not a permanent task-proof deadline.

    def test_pose_completion_uses_original_arrival_window_instead_of_source_time(self):
        timing = EventTimeConstraint("arrival", (.1, .3), "request_physical", "original early arrival window",
                                     origin_time_s=0., goal_id=("pose_events", 0))
        event = PlanningEvent("arrive", "s", .8, 1, "function", "source", timing=(timing,))
        goal = PoseEventRequirement(.8, np.eye(4), np.eye(4), .001, .01, "original task goal")
        original = problem((event,), (goal,))
        for acquisition, accepted in ((.2, True), (.35, False)):
            with self.subTest(acquisition=acquisition):
                request = deepcopy(original)
                actual = continuation(request, elapsed=.4, completed=(("pose_events", 0),))
                actual["goal_evidence"][0].update(timestamp_s=acquisition, physical_elapsed_s=acquisition,
                    evaluated_curve_time_s=acquisition, evaluation_window_s=[.1, .3])
                actual["goal_evidence"][0]["measurement_sources"][0]["timestamp_s"] = acquisition
                if accepted:
                    _, result = _prepare_continuation(request, SimpleNamespace(gripper=None), None, actual)
                    self.assertEqual(result["goal_identities"]["pose_events"], [])
                else:
                    with self.assertRaisesRegex(ContinuationBindingError, "actual allowed-set T evidence"):
                        _prepare_continuation(request, SimpleNamespace(gripper=None), None, actual)

    def test_same_request_cannot_remove_actual_owner_dependency(self):
        goal = PoseEventRequirement(.8, np.eye(4), np.eye(4), .001, .01, "same functional target",
            actual_relation={"target_owner_id": "observed_target", "T_owner_target": np.eye(4)})
        request = problem(goals=(goal,))
        actual = continuation(request)
        changed = problem(goals=(replace(goal, actual_relation=None),))
        with self.assertRaisesRegex(ContinuationBindingError, "goal/process obligations"):
            _prepare_continuation(changed, SimpleNamespace(gripper=None), None, actual)
    # END task tolerance and current-owner dependencies survive retries unchanged.

    def test_compiler_annotates_original_commands_and_goals_without_adding_actions(self):
        from test_manipulation_planning_problem import compile_inputs, grasp_input, manipulation_input
        grasp = grasp_input()
        annotation = {"check_target": "activation", "window_s": [0., 2.], "time_basis": "request_physical",
                      "origin_time_s": 0., "source": "explicit source command deadline"}
        grasp["program"]["demo"]["gripper_events"][1]["timing"] = [annotation]
        request = compile_inputs(grasp)
        close = [event for event in request.events if event.event_id == "close"]
        self.assertEqual(len(close), 1)
        self.assertEqual(close[0].timing[0].check_target, "activation")
        move = manipulation_input()
        move["program"]["demo"]["functional_events"][0]["timing"] = [{
            **annotation, "check_target": "arrival", "goal_id": ["pose_events", 0]}]
        request = compile_inputs(move)
        functional = [event for event in request.events if event.event_id == "function"]
        self.assertEqual(len(functional), 1)
        self.assertEqual(functional[0].timing[0].goal_id, ("pose_events", 0))

    def test_device_tick_is_checked_instead_of_earlier_command_intent(self):
        constraint = EventTimeConstraint("activation", (None, .39), "request_physical", "hard task deadline", origin_time_s=0.)
        event = PlanningEvent("close", "skill", .37, 1, "close", "source", timing=(constraint,))
        request = problem((event,))
        clock = _bind_event_clock(request, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 1., .1)
        record = clock.event_clock[("skill", "close", 0)]
        self.assertEqual(record["intent_time_s"], .37)
        self.assertEqual(record["activation_time_s"], .4)
        with self.assertRaisesRegex(TimeBindingError, "deadline"):
            _validate_bound_event_times(request, clock, 1.)

    def test_command_tick_does_not_retime_an_independent_functional_event(self):
        event = PlanningEvent("close", "skill", .37, 1, "close", "source")
        function = PlanningEvent("contact", "skill", .5, 2, "function", "independent functional instant")
        goal = PoseEventRequirement(.5, np.eye(4), np.eye(4), .01, .02, "independent target")
        request = problem((event, function), (goal,))
        clock = _bind_event_clock(request, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 1., .1)
        self.assertEqual(clock.event_clock[("skill", "close", 0)]["activation_time_s"], .4)
        self.assertEqual(clock.event_clock[("skill", "contact", 0)]["time"], .5)
        self.assertEqual(clock.anchor_parameters([goal.progress])[0, 0], .5)

    def test_ordinary_same_tick_intentions_preserve_superseded_source_record(self):
        events = (PlanningEvent("open", "s", .37, 1, "open", "source"),
                  PlanningEvent("close", "s", .39, 2, "close", "source"))
        request = problem(events)
        clock = _bind_event_clock(request, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 1., .1)
        self.assertEqual(set(clock.event_clock), {("s", "open", 0), ("s", "close", 0)})
        for record in clock.event_clock.values():
            self.assertEqual(record["activation_time_s"], .4)
        self.assertEqual(clock.event_clock[("s", "open", 0)]["scheduled_superseded_by"], ("s", "close", 0))
        self.assertEqual(clock.anchor_parameters([.39])[0, 0], .4)
        # A declared activation obligation cannot be satisfied by losing its
        # command before any physical step; that remains a genuine conflict.
        deadline = EventTimeConstraint("activation", (0., .5), "request_physical", "required activation", origin_time_s=0.)
        request.events = (replace(events[0], timing=(deadline,)), events[1])
        with self.assertRaisesRegex(ValueError, "required physical occurrence"):
            _bind_event_clock(request, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 1., .1)

    def test_origin_requires_actual_matching_occurrence_and_retains_first_truth(self):
        origin = {"skill_id": "s", "event_id": "open", "occurrence": 0, "check_target": "activation"}
        constraint = EventTimeConstraint("effect", (0., .5), "request_physical", "task", origin_event=origin, condition_id="released")
        first = {**origin, "time_s": 4., "value": "T", "source": "actual"}
        self.assertEqual(resolve_event_window(constraint, [first, {**first, "time_s": 4.2}]), (4., 4.5))
        with self.assertRaises(TimeBindingError):
            resolve_event_window(constraint, [{**first, "occurrence": 1}])
        with self.assertRaises(TimeBindingError):
            resolve_event_window(constraint, [{**first, "value": "U"}])

    def test_explicit_remaining_rebind_preserves_overdue_unapplied_event(self):
        event = PlanningEvent("close", "s", .2, 1, "close", "task")
        request = problem((event,))
        history = continuation(request, elapsed=1.2, curve_time=.4)
        scene = SimpleNamespace(gripper=None, required_progress_anchors=())
        with self.assertRaisesRegex(ContinuationBindingError, "clock is exhausted"):
            _prepare_continuation(deepcopy(request), scene, None, history)
        budget, prior = _prepare_continuation(request, scene, None, history,
            rebind={"kind": "remaining", "source": "Executor observed interruption"})
        _, clock = _continuation_timing(request, budget, prior)
        self.assertEqual(event_occurrence_key(request.events[0]), ("s", "close", 0))
        self.assertEqual(request.events[0].source_frame, 1)
        self.assertEqual(prior["request_time_origin_s"], 1.2)
        self.assertEqual(clock.event_clock[("s", "close", 0)]["request_activation_time_s"], 1.2)

    def test_retry_creates_new_occurrence_and_retains_unrelated_completed_goal(self):
        events = (PlanningEvent("close", "s", .2, 1, "close", "task"),
                  PlanningEvent("open", "s", .8, 2, "open", "task"))
        goals = tuple(PoseEventRequirement(progress, np.eye(4), np.eye(4), .01, .02, f"goal-{index}")
                      for index, progress in enumerate((.5, .9)))
        request = problem(events, goals)
        history = continuation(request, elapsed=1.2, curve_time=1., applied={("s", "close", 0), ("s", "open", 0)},
                               completed=(("pose_events", 0), ("pose_events", 1)))
        budget, prior = _prepare_continuation(request, SimpleNamespace(gripper=None, required_progress_anchors=()), None, history,
            rebind={"kind": "retry", "source": "Executor same-goal retry", "retry_event_ids": [["s", "close"]],
                    "retry_goal_ids": [["pose_events", 0]]})
        self.assertEqual([event_occurrence_key(event) for event in request.events], [("s", "close", 1)])
        self.assertIn(("s", "close", 0), prior["consumed_events"])
        self.assertEqual(prior["goal_identities"]["pose_events"], [0])
        self.assertIn(("pose_events", 1), prior["completed_goal_ids"])
        self.assertEqual(len(request.pose_events), 1)
        _, clock = _continuation_timing(request, budget, prior)
        self.assertEqual(set(clock.event_clock), {("s", "close", 1)})

    def test_explicit_rebind_cannot_move_an_original_expired_deadline(self):
        deadline = EventTimeConstraint("activation", (None, .3), "request_physical", "hard deadline", origin_time_s=0.)
        request = problem((PlanningEvent("close", "s", .2, 1, "close", "task", timing=(deadline,)),))
        history = continuation(request, elapsed=.4, curve_time=.4)
        budget, prior = _prepare_continuation(request, SimpleNamespace(gripper=None, required_progress_anchors=()), None, history,
            rebind={"kind": "remaining", "source": "Executor"})
        with self.assertRaisesRegex(TimeBindingError, "original physical deadline"):
            _continuation_timing(request, budget, prior)

    def test_fresh_binding_cannot_delete_goals_or_relax_declared_tolerance(self):
        goal = PoseEventRequirement(.8, np.eye(4), np.eye(4), .01, .02, "original functional target")
        original = problem(goals=(goal,))
        history = continuation(original)
        for goals in ((), (replace(goal, position_tolerance_m=.1),)):
            fresh = problem(goals=goals)
            with self.assertRaisesRegex(ContinuationBindingError, "goal/process obligations"):
                _prepare_continuation(fresh, SimpleNamespace(gripper=None), None, history)

    def test_fresh_binding_cannot_delete_a_real_process_relation(self):
        original = problem()
        original.pose_relations = (PoseProcessRequirement(0., 1., original.phases[0].reference,
            np.zeros((6, 2)), "explicit maintained contact relation"),)
        history = continuation(original)
        with self.assertRaisesRegex(ContinuationBindingError, "goal/process obligations"):
            _prepare_continuation(problem(), SimpleNamespace(gripper=None), None, history)

    def test_net_direction_and_delta_goal_keep_original_entry_during_rebind(self):
        model = MotionModel("prismatic", np.zeros((1, 1)), np.eye(4), np.eye(3), None,
                            1e-6, 1e-6, {"source": "physical slider"})
        direction = CoordinateDirectionRequirement(0., 1., 0, "net", 1, .5, .001, "required positive net change")
        original = problem()
        original.mechanisms = (MechanismRequirement(0., 1., model, np.eye(4), [[0., 2.]], [0.],
                                                   "observed slider coordinate frame", (direction,)),)
        original.coordinate_events = (CoordinateEventRequirement(1., original.mechanisms[0], .5, .001,
            "original half-unit move", coordinate_mode="entry_delta", coordinate_origin=0.),)
        history = continuation(original)
        fresh = deepcopy(original)
        # Fresh actual geometry is at +.2; this must not move the old goal to +.7.
        fresh.mechanisms = (MechanismRequirement(0., 1., model, np.eye(4), [[0., 2.]], [.2],
                                                "observed slider coordinate frame", (direction,)),)
        fresh.coordinate_events = (replace(original.coordinate_events[0], mechanism=fresh.mechanisms[0]),)
        self.assertEqual(fresh.mechanisms[0].directions[0].origin_coordinate, .2)
        _prepare_continuation(fresh, SimpleNamespace(gripper=None), None, history)
        self.assertEqual(fresh.mechanisms[0].directions[0].origin_coordinate, 0.)
        self.assertEqual(fresh.coordinate_events[0].absolute_target, .5)
        self.assertEqual(fresh.mechanisms[0].initial_coordinates[0], .2)
        changed = deepcopy(original)
        changed.coordinate_events = (replace(original.coordinate_events[0], coordinate_origin=.2),)
        with self.assertRaisesRegex(ContinuationBindingError, "goal/process obligations"):
            _prepare_continuation(changed, SimpleNamespace(gripper=None), None, history)

    def test_middle_net_scope_requires_actual_start_coordinate_evidence(self):
        model = MotionModel("prismatic", np.zeros((1, 1)), np.eye(4), np.eye(3), None,
                            1e-6, 1e-6, {"source": "physical slider"})
        direction = CoordinateDirectionRequirement(.2, 1., 0, "net", 1, .5, .001, "middle interval net change")
        request = problem()
        request.mechanisms = (MechanismRequirement(0., 1., model, np.eye(4), [[0., 2.]], [0.],
                                                  "observed physical slider", (direction,)),)
        history = continuation(request)
        with self.assertRaisesRegex(ContinuationBindingError, "actually observed original scope-start"):
            _prepare_continuation(request, SimpleNamespace(gripper=None), None, history)

    # BEGIN past process scopes require observed completion, not curve progress.
    def test_past_direction_requires_scoped_actual_completion_from_its_candidate(self):
        model = MotionModel("prismatic", np.zeros((1, 1)), np.eye(4), np.eye(3), None,
                            1e-6, 1e-6, {"source": "physical slider"})
        direction = CoordinateDirectionRequirement(0., .3, 0, "net", 1, .2, .001, "required net travel")
        request = problem()
        request.mechanisms = (MechanismRequirement(0., 1., model, np.eye(4), [[0., 2.]], [0.],
                                                  "actual entry coordinate", (direction,)),)
        observed = {"identity": [0, 0], "candidate_id": "previous", "value": "T", "scope_complete": True,
                    "source": "actual slider measurements", "acquisition_interval_s": [0., .3],
                    "completion_verified": True, "scope_started_at_s": 0., "scope_ended_at_s": .3}
        for changes in ({"value": "F"}, {"value": "U"}, {"scope_complete": False},
                        {"candidate_id": "unrelated"}, {"acquisition_interval_s": [.3, 0.]},
                        {"completion_verified": False}, {"scope_ended_at_s": .4}):
            with self.subTest(changes=changes):
                history = continuation(request)
                history.update(completed_direction_ids=[[0, 0]], direction_evidence=[{**observed, **changes}])
                with self.assertRaisesRegex(ContinuationBindingError, "actual scoped T evidence"):
                    _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, history)
        history = continuation(request)
        with self.assertRaisesRegex(ContinuationBindingError, "Past direction"):
            _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, history)
        history.update(completed_direction_ids=[[0, 0]], direction_evidence=[observed])
        rebound = deepcopy(request)
        _, prior = _prepare_continuation(rebound, SimpleNamespace(gripper=None), None, history)
        self.assertEqual(prior["completed_direction_ids"], [(0, 0)])
        self.assertEqual(prior["direction_identities"], [[(0, 0)]])
        self.assertEqual(rebound.start_progress, .4)
        monotone = replace(direction, kind="monotone", minimum=0.)
        request.mechanisms = (replace(request.mechanisms[0], directions=(monotone,)),)
        history = continuation(request)
        history.update(completed_direction_ids=[[0, 0]], direction_evidence=[{
            **observed, "acquisition_interval_s": [.2, .3]}])
        with self.assertRaisesRegex(ContinuationBindingError, "actual scoped T evidence"):
            _prepare_continuation(request, SimpleNamespace(gripper=None), None, history)

    def test_explicit_direction_retry_preserves_origin_and_original_identity(self):
        model = MotionModel("prismatic", np.zeros((1, 1)), np.eye(4), np.eye(3), None,
                            1e-6, 1e-6, {"source": "physical slider"})
        direction = CoordinateDirectionRequirement(0., .3, 0, "net", 1, .2, .001, "required net travel")
        request = problem()
        request.mechanisms = (MechanismRequirement(0., 1., model, np.eye(4), [[0., 2.]], [0.],
                                                  "actual entry coordinate", (direction,)),)
        history = continuation(request)
        request.mechanisms = (replace(request.mechanisms[0], initial_coordinates=np.array([.1]),
                                     directions=(direction,)),)
        _, prior = _prepare_continuation(request, SimpleNamespace(gripper=None, required_progress_anchors=()), None, history,
            rebind={"kind": "retry", "source": "actual failed process assessment", "retry_direction_ids": [[0, 0]]})
        self.assertEqual(request.start_progress, 0.)
        self.assertEqual(prior["retry_direction_ids"], [(0, 0)])
        self.assertEqual(prior["direction_identities"], [[(0, 0)]])
        self.assertEqual(request.mechanisms[0].initial_coordinates[0], .1)
        self.assertEqual(request.mechanisms[0].directions[0].origin_coordinate, 0.)
    # END direction retries do not manufacture a fresh displacement target.

    def test_fresh_angular_branch_reanchors_only_the_actual_history_entrance(self):
        model = MotionModel("revolute", np.zeros((1, 1)), np.eye(4), np.eye(3), np.zeros(3),
                            1e-6, 1e-6, {"source": "physical hinge"})
        direction = CoordinateDirectionRequirement(0., 1., 0, "net", 1, 4*np.pi, .001, "original two-turn change")
        original = problem()
        original.mechanisms = (MechanismRequirement(0., 1., model, np.eye(4), [[0., 5*np.pi]], [0.],
                                                   "original physical hinge", (direction,)),)
        original.coordinate_events = (CoordinateEventRequirement(1., original.mechanisms[0], 4*np.pi, .001,
            "original two-turn target", coordinate_mode="entry_delta", coordinate_origin=0.),)
        history = continuation(original)
        for evidence in ({"initial_coordinates_source": "fresh actual multi-turn encoder"},
                         {"initial_coordinate_convention": "unique actual angular branch within current physical limits"}):
            with self.subTest(evidence=evidence):
                fresh = deepcopy(original)
                fresh.mechanisms[0].initial_coordinates[:] = 2*np.pi+.2
                fresh.mechanisms[0].model.diagnostics.update(evidence)
                _, prior = _prepare_continuation(fresh, SimpleNamespace(gripper=None), None, history)
                self.assertEqual(fresh.mechanisms[0].start, .4)
                self.assertIs(fresh.coordinate_events[0].mechanism, fresh.mechanisms[0])
                self.assertEqual(fresh.mechanisms[0].directions[0].start, 0.)
                self.assertEqual(fresh.mechanisms[0].directions[0].origin_coordinate, 0.)
                self.assertEqual(fresh.coordinate_events[0].absolute_target, 4*np.pi)
                self.assertEqual(prior["direction_identities"], [[(0, 0)]])
                position = torch.zeros((1, 1, 3), dtype=torch.float64)
                quaternion = torch.tensor([[[np.cos(.1), 0., 0., np.sin(.1)]]], dtype=torch.float64)
                actual = fresh.mechanisms[0].coordinates(position, quaternion, torch.tensor([[.4]]))
                self.assertAlmostEqual(float(actual[0, 0, 0]), 2*np.pi+.2)
        fresh = deepcopy(original)
        fresh.mechanisms[0].initial_coordinates[:] = 2*np.pi+.2
        with self.assertRaisesRegex(ContinuationBindingError, "current observed lifted coordinates"):
            _prepare_continuation(fresh, SimpleNamespace(gripper=None), None, history)

    def test_consumed_timed_witness_remains_checkable_against_future_origin(self):
        origin = {"skill_id": "s", "event_id": "later", "occurrence": 0, "check_target": "activation"}
        window = EventTimeConstraint("activation", (-.7, -.6), "request_physical", "physical relation", origin_event=origin)
        earlier = PlanningEvent("earlier", "s", .1, 1, "open", "source", timing=(window,))
        request = problem((PlanningEvent("later", "s", .8, 2, "close", "source"),))
        clock = _bind_event_clock(request, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 1., .01)
        facts = [{"skill_id": "s", "event_id": "earlier", "occurrence": 0,
                  "check_target": "activation", "time_s": .1, "value": "T", "source": "actual device"}]
        checks = _validate_bound_event_times(request, clock, 1., history=facts, timing_events=[plain(earlier)])
        self.assertEqual(checks[0]["status"], "actual_time_inside_window")
        self.assertEqual(checks[0]["scheduled_request_time_s"], .1)
        self.assertEqual(set(clock.event_clock), {("s", "later", 0)})

    def test_old_effect_truth_before_window_does_not_revive_expired_obligation(self):
        window = EventTimeConstraint("effect", (.3, .5), "request_physical", "task effect window",
                                     origin_time_s=0., condition_id="done")
        old = PlanningEvent("observe", "s", .1, 1, "effect", "source", timing=(window,))
        history = [{"skill_id": "s", "event_id": "observe", "occurrence": 0, "check_target": "effect",
                    "condition_id": "done", "time_s": .1, "value": "T", "source": "early state observation"}]
        with self.assertRaisesRegex(TimeBindingError, "deadline"):
            _validate_bound_event_times(problem(), _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 1.,
                                        origin_s=.6, history=history, timing_events=[plain(old)])

    def test_failed_public_rebind_does_not_filter_callers_problem_or_scene(self):
        request = problem((PlanningEvent("old", "s", .1, 1, "close", "source"),
                           PlanningEvent("future", "s", .8, 2, "open", "source")))
        history = continuation(request, applied={("s", "old", 0)})
        scene = SimpleNamespace(gripper=None, required_progress_anchors=())
        scene.clone_for_rollout = lambda: deepcopy(scene)
        before = plain(request)
        with patch("manipulation.planning._CuroboPlanner", side_effect=ValueError("native initialization gap")):
            result = plan_trajectory(request, scene, {}, continuation=history)
        self.assertFalse(result.success)
        self.assertIn("native initialization gap", result.diagnostics["reason"])
        self.assertEqual(plain(request), before)
        self.assertEqual(result.request_ledger["consumed_planning"]["initializations"], 2)

    # BEGIN failed continuation work cannot publish unverified completion authority.
    def test_rejected_completion_cannot_poison_a_failed_work_ledger(self):
        goal = PoseEventRequirement(.2, np.eye(4), np.eye(4), .001, .01, "required even without discrete events")
        request = problem(goals=(goal,))
        actual = continuation(request, completed=(("pose_events", 0),))
        actual["goal_evidence"][0]["completion_verified"] = False
        # The executor output already carries appended facts. They still
        # need validation before a planning failure can return a trusted ledger.
        actual["request_ledger"] = deepcopy(actual["original_binding"]["request_ledger"])
        for key in ("completed_goal_ids", "goal_evidence"):
            actual["request_ledger"][key] = deepcopy(actual[key])
        scene = SimpleNamespace(gripper=None, required_progress_anchors=())
        scene.clone_for_rollout = lambda: deepcopy(scene)
        with patch("manipulation.planning._CuroboPlanner") as native:
            failed = plan_trajectory(request, scene, {}, continuation=actual)
            self.assertFalse(failed.success)
            self.assertIn("actual allowed-set T evidence", failed.diagnostics["reason"])
            self.assertEqual(failed.request_ledger["completed_goal_ids"], [])
            self.assertEqual(failed.request_ledger["goal_evidence"], [])
            self.assertTrue(failed.request_ledger["requires_continuation"])
            self.assertEqual(failed.request_ledger["consumed_planning"]["binding_attempts"], 2)
            self.assertIn("continuation_facts_pending_validation", failed.diagnostics)
            retried = plan_trajectory(request, scene, {}, request_ledger=failed.request_ledger)
            self.assertFalse(retried.success)
            self.assertIn("actual continuation facts", retried.diagnostics["reason"])
            native.assert_not_called()

    def test_continuation_cannot_replace_an_original_actual_time_origin(self):
        event = PlanningEvent("origin", "s", 0., 0, "close", "original command")
        request = problem((event,))
        actual = continuation(request, applied={("s", "origin", 0)})
        preserved = deepcopy(actual["history"])
        actual["original_binding"]["request_ledger"]["history"] = preserved
        for replacement in ([], [{**preserved[0], "time_s": .2}]):
            with self.subTest(replacement=replacement):
                changed = deepcopy(actual)
                changed["history"] = replacement
                with self.assertRaisesRegex(ContinuationBindingError, "original physical timing history"):
                    _prepare_continuation(deepcopy(request), SimpleNamespace(gripper=None), None, changed)
    # END source progress and failed retries cannot move physical origins or consume goals.

    def test_failed_first_initialization_does_not_mutate_callers_hand_or_problem(self):
        request = problem((PlanningEvent("future", "s", .8, 2, "open", "source"),))
        scene = SimpleNamespace(gripper=SimpleNamespace(clock=None, commands=["original"]), required_progress_anchors=())
        scene.clone_for_rollout = lambda: deepcopy(scene)
        before = plain(request)
        def partially_initialize(private_problem, private_scene, *args, **kwargs):
            private_problem.events = ()
            private_scene.gripper.clock = {"bound": True}
            private_scene.gripper.commands.clear()
            raise ValueError("native allocation failed after clock binding")
        with patch("manipulation.planning._CuroboPlanner", side_effect=partially_initialize):
            result = plan_trajectory(request, scene, {})
        self.assertFalse(result.success)
        self.assertIn("native allocation failed", result.diagnostics["reason"])
        self.assertEqual(plain(request), before)
        self.assertIsNone(scene.gripper.clock)
        self.assertEqual(scene.gripper.commands, ["original"])
        self.assertEqual(result.request_ledger["consumed_planning"]["initializations"], 1)

    def test_cleanup_overrun_keeps_an_already_qualified_candidate(self):
        from test_manipulation_planning import PlannerRetryContractTests
        fixture = PlannerRetryContractTests()
        planner, _, _ = fixture.scripted_planner([True], [0.])
        clock = {"time": 0.}
        planner.close.side_effect = lambda: clock.update(time=2.)
        with patch("manipulation.planning.time.monotonic", side_effect=lambda: clock["time"]):
            result = fixture.run_planner(planner, PlanningBudget(maximum_trajectory_solves=1, planning_seconds=1.))
        self.assertTrue(result.success)
        self.assertIsNotNone(result.execution_binding)
        self.assertEqual(result.diagnostics["qualified_candidate_elapsed_seconds"], 0.)
        self.assertEqual(result.request_ledger["consumed_planning"]["planning_seconds"], 2.)
        self.assertTrue(result.diagnostics["budget_exhausted_after_qualification"])

    def test_binding_stage_overrun_is_not_an_already_qualified_candidate(self):
        from test_manipulation_planning import PlannerRetryContractTests
        planner, _, _ = PlannerRetryContractTests().scripted_planner([True], [0.])
        clock = {"time": 0.}
        def expensive_binding(*args, **kwargs):
            clock["time"] = 2.
            return {"candidate_id": "too-late"}
        scene = SimpleNamespace(describe=lambda: {})
        scene.clone_for_rollout = lambda: deepcopy(scene)
        with patch("manipulation.planning.time.monotonic", side_effect=lambda: clock["time"]), \
                patch("manipulation.planning._CuroboPlanner", return_value=planner), \
                patch("manipulation.execution_binding.bind_candidate", side_effect=expensive_binding):
            result = plan_trajectory(problem(), scene, {},
                                     budget=PlanningBudget(maximum_trajectory_solves=1, planning_seconds=1.))
        self.assertFalse(result.success)
        self.assertIsNone(result.execution_binding)
        self.assertEqual(result.diagnostics["failure"], "cumulative_planning_budget_exhausted")

    def test_single_goal_on_progress_plateau_does_not_require_later_holding(self):
        goal = PoseEventRequirement(.5, np.eye(4), np.eye(4), .01, .02, "single arrival")
        request = problem(goals=(goal,))
        # The exact first anchor is appended after ordinary rollout queries.
        progress = torch.tensor([[.5, .5, .5]], dtype=torch.float64)
        times = torch.tensor([[.6, .8, .4]], dtype=torch.float64)
        position = torch.tensor([[[.2, 0., 0.], [.3, 0., 0.], [0., 0., 0.]]], dtype=torch.float64, requires_grad=True)
        quaternion = torch.tensor([[[1., 0., 0., 0.]]*3], dtype=torch.float64)
        self.assertEqual(float(request.evaluate(position, quaternion, progress, trajectory_time=times).max().detach()), 0.)
        # Declaring a maintained relation separately still rejects those poses.
        request.pose_relations = (PoseProcessRequirement(.4, .9, request.phases[0].reference, np.zeros((6, 2)), "required hold"),)
        self.assertGreater(float(request.evaluate(position, quaternion, progress, trajectory_time=times).max().detach()), .1)
        request.pose_relations = ()
        bad = position.detach().clone(); bad[0, 2, 0] = .1
        residual = request.evaluate(bad, quaternion, progress, trajectory_time=times)
        self.assertGreater(float(residual[0, 2]), .08)
        self.assertEqual(float(residual[0, :2].max()), 0.)


if __name__ == "__main__":
    unittest.main()
# END no synthetic feedback is promoted to an executed robot result.
