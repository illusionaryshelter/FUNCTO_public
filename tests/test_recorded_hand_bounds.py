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
    return SimpleNamespace(events=events), feedback, targets


def factory(problem, feedback, targets):
    build = finger_trajectory_factory(problem, feedback, targets, timestep=.01,
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
    np.testing.assert_array_equal(dq, [.0001, -.0002])
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
        finger_trajectory_factory(problem, feedback, targets, timestep=.01,
            ctrlrange=[[0., .04], [-.04, 0.]],
            position_limits={"left": [0., .04], "right": [-.04, 0.]},
            velocity_limits={"left": .2, "right": .2}, acceleration_limits={"left": 15.})(
                3., _bind_event_clock(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), 3., .01))
# END goals, actual feedback and time-varying reference retain separate meanings.
