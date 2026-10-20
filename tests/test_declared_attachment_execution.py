"""Declared attachment models do not require physical Holding proof to run."""

# BEGIN bounded CPU execution contracts for nominal attachment semantics.
from copy import deepcopy
from unittest.mock import patch

import pytest

from manipulation.execution import (ExecutionGuardError, FixedFingerTargets,
    TrajectoryExecutor, _feedback_json)
from manipulation.execution_binding import rebind_execution_result
from test_holding_owner_permission import holding_fixture, refresh_validation
from test_recorded_simulation import _GeometryChecker, _initial_feedback


def _trial(tmp_path, *, initial_value="absent", observation=None,
           rejected_stage=None, mutate=None):
    """Use the existing synthetic native backend without object truth."""
    simulation, _, targets, result, limits = holding_fixture()
    if mutate is not None:
        mutate(result)
        refresh_validation(result)
    fingers = FixedFingerTargets.from_mapping(targets)
    result = rebind_execution_result(result,
        controller_policy=simulation.describe()["controller_policy"],
        result_only_goal_ids=[], finger_targets=fingers, execution_limits=limits,
        source="Explicit declared-attachment CPU fixture")
    feedback = _initial_feedback(simulation)
    if initial_value != "absent":
        feedback["scene_condition_evidence"] = {"observed_holding": initial_value}
    gate = _GeometryChecker(simulation.joint_names, rejected_stage=rejected_stage,
                            rejected_time=.002)
    executor = TrajectoryExecutor(simulation, geometry_checker=gate)
    original = deepcopy((result.execution_binding, feedback))

    def run():
        with patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            return executor.execute(result, finger_targets=fingers, initial_feedback=feedback,
                execution_limits=limits, observation_callback=observation,
                output=tmp_path / "execution")

    return run, executor, simulation, result, feedback, original


@pytest.mark.parametrize("initial_value", ["absent", False, None])
def test_nominal_playback_needs_no_holding_truth_or_extra_authorization(tmp_path, initial_value):
    run, _, _, result, feedback, original = _trial(tmp_path, initial_value=initial_value)
    report = run()
    assert report["status"] == "curve_finished"
    assert report["executed_steps"] > 0
    assert report["effect_result"] == "U" and report["effects"] == {}
    assert report["completed_goal_ids"] == []
    assert "attachment_assumptions" not in report
    declarations = report["declared_attachment_bindings"]
    assert declarations and all(row["condition_id"] == "observed_holding" for row in declarations)
    assert all("feedback_id" not in row and "timestamp_s" not in row for row in declarations)
    # Physics-budget accounting may be added, but the accepted mathematical
    # inputs and the caller-owned robot feedback must remain unchanged.
    assert {key: result.execution_binding[key] for key in original[0]} == original[0]
    assert _feedback_json(feedback) == _feedback_json(original[1])


@pytest.mark.parametrize("value", [False, None])
def test_predicate_labels_do_not_reject_the_nominal_attachment(tmp_path, value):
    def observe(feedback, curve_time):
        return {"source": "Independent synthetic predicate estimate",
            "timestamp_s": feedback["timestamp_s"],
            "scene_condition_evidence": {"observed_holding": value}}
    run, _, _, *_ = _trial(tmp_path, observation=observe)
    report = run()
    assert report["status"] == "curve_finished"
    assert report["effects"] == {} and report["completed_goal_ids"] == []


def test_declared_attachment_does_not_relax_robot_guard(tmp_path):
    run, executor, _, *_ = _trial(tmp_path, rejected_stage="command_event")
    with pytest.raises(ExecutionGuardError):
        run()
    assert executor.last_execution_report["feedback_failure"]["violations"]


@pytest.mark.parametrize("change", ["owner", "interval", "transform", "kind"])
def test_declared_attachment_keeps_model_identity_and_geometry(tmp_path, change):
    def mutate(result):
        condition = result.execution_binding["problem"]["conditions"][0]
        if change == "owner":
            condition["carrier_id"] = "different_owner"
        elif change == "interval":
            result.execution_binding["scene"]["owners"][0]["attachments"][0]["interval"] = [0., .5]
        elif change == "transform":
            condition["T_carrier_eef"][0][3] = .1
        else:
            condition["kind"] = "candidate_attachment"
    run, _, simulation, *_ = _trial(tmp_path, mutate=mutate)
    with pytest.raises(ValueError):
        run()
    assert simulation.arm_goals == []
# END no successful CPU sequence certifies a physical attachment.
