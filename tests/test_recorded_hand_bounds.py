"""Recorded adapter checks for one shared arm/finger clock and actual entrance."""

# BEGIN actual protection intervals no longer stand in for hand trajectories.
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))
from manipulation.execution import FixedFingerTargets
from manipulation.planning import _ProgressMap, _bind_event_clock
from manipulation.planning_problem import PlanningEvent
from run_recorded_skill import finger_trajectory_factory
from manipulation.gripper_trajectory import stationary_finger_command


def inputs():
    feedback = {"joint_names": ["arm", "right", "left"], "finger_joint_names": ["left", "right"],
                "start_q": [1., -.0395, .039], "start_dq": [.1, -.0002, .0001],
                "feedback_id": "actual-snapshot", "sim_time_s": 2.5}
    events = [PlanningEvent("open", "s", 0., 0, "open", "source"),
              PlanningEvent("close", "s", .5, 183, "close", "source"),
              PlanningEvent("verify", "s", 1., 197, "verification", "source", "Holding", True)]
    targets = FixedFingerTargets(("left", "right"), "explicit targets", (
        {"skill_id": "s", "event_id": "open", "kind": "open", "positions": [.04, -.04]},
        {"skill_id": "s", "event_id": "close", "kind": "close", "positions": [0., 0.]}))
    feedback["incoming_finger_command"] = stationary_finger_command(joint_names=["left", "right"],
        position=[.039, -.0395], timestamp_s=2.5, feedback_id="actual-snapshot",
        source="Declared stationary target independent of measured finger velocity")
    return SimpleNamespace(events=events, pose_events=(), coordinate_events=(), supports=()), feedback, targets


def factory(problem, feedback, targets):
    build = finger_trajectory_factory(problem, feedback, targets, incoming_command=feedback["incoming_finger_command"], timestep=.01,
        ctrlrange=[[0., .04], [-.04, 0.]],
        position_limits={"left": [0., .04], "right": [-.04, 0.]},
        velocity_limits={"left": .2, "right": .2},
        acceleration_limits={"left": 15., "right": 15.})
    # BEGIN the fixture uses the same sole clock owner as the planner.
    return lambda duration, progress: build(duration, _bind_event_clock(problem, progress, duration, .01))
    # END the finger factory itself cannot invent the binding.


def test_finger_clock_preserves_measured_entrance_and_original_plateau_event():
    problem, feedback, targets = inputs()
    # Closing occurs at the first end of this progress plateau. The hand keeps
    # moving as physical time passes even though progress is exactly constant.
    progress = _ProgressMap(np.array([0., .4, .6, 1.]), np.array([0., .5, .5, 1.]))
    curve = factory(problem, feedback, targets)(3., progress)
    q, dq = curve.sample(0.)
    np.testing.assert_array_equal(q, [.039, -.0395])
    np.testing.assert_array_equal(dq, [0., 0.])
    np.testing.assert_array_equal(curve.measured_initial_velocity, [.0001, -.0002])
    assert curve.executable_until == 3.
    assert curve.duration == 3.
    assert curve.joint_names == ("left", "right")
    assert not np.array_equal(curve.sample(1.2)[0], curve.sample(1.3)[0])
    np.testing.assert_allclose(curve.sample(3.)[0], [0., 0.], atol=1e-15)
    assert feedback["start_q"] == [1., -.0395, .039]


def test_edited_target_identity_is_rejected_before_curve_generation():
    problem, feedback, targets = inputs()
    edited = FixedFingerTargets(targets.joint_names, targets.source, targets.events[:1])
    with pytest.raises(ValueError, match="original command event"):
        factory(problem, feedback, edited)(3., _ProgressMap(np.array([0., 1.]), np.array([0., 1.])))


def test_missing_named_reference_limit_cannot_be_replaced_by_actual_hardware_interval():
    problem, feedback, targets = inputs()
    with pytest.raises(KeyError, match="right"):
        finger_trajectory_factory(problem, feedback, targets, incoming_command=feedback["incoming_finger_command"], timestep=.01,
            ctrlrange=[[0., .04], [-.04, 0.]],
            position_limits={"left": [0., .04], "right": [-.04, 0.]},
            velocity_limits={"left": .2, "right": .2}, acceleration_limits={"left": 15.})(
                3., _bind_event_clock(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 3., .01))


# BEGIN a new occurrence consumes only its exact candidate clock payload.
def test_retry_factory_uses_new_occurrence_even_when_source_problem_contains_consumed_events():
    from dataclasses import replace
    problem, feedback, targets = inputs()
    subset = FixedFingerTargets(targets.joint_names, targets.source, targets.events[1:])
    build = finger_trajectory_factory(problem, feedback, subset, incoming_command=feedback["incoming_finger_command"], timestep=.01,
        ctrlrange=[[0., .04], [-.04, 0.]], position_limits={"left": [0., .04], "right": [-.04, 0.]},
        velocity_limits={"left": .2, "right": .2}, acceleration_limits={"left": 15., "right": 15.})
    retry = SimpleNamespace(events=[replace(problem.events[1], occurrence=1)],
                            pose_events=(), coordinate_events=(), supports=())
    mapping = _bind_event_clock(retry, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 3., .01)
    curve = build(3., mapping)
    assert [(row["event_id"], row["occurrence"]) for row in curve.events] == [("close", 1)]
    assert curve.events[0]["source_frame"] == 183
    assert problem.events[1].occurrence == 0
# END an earlier occurrence cannot overwrite or be replayed by a hand closure.
# END goals, actual feedback and time-varying reference retain separate meanings.
