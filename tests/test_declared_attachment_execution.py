"""Declared attachment models do not require physical Holding proof to run."""

# BEGIN bounded CPU execution contracts for nominal attachment semantics.
from copy import deepcopy
from unittest.mock import patch

import pytest
import numpy as np

from manipulation.execution import (ExecutionGuardError, ExecutionLimits, FixedFingerTargets,
    TrajectoryExecutor, _feedback_json)
from manipulation.execution_binding import rebind_execution_result
from test_recorded_simulation import (_GeometryChecker, _initial_feedback,
    _fixture, _bind_fingers, _result, _limit_values)


def _trial(tmp_path, *, initial_value="absent", observation=None,
           rejected_stage=None):
    """Use the existing synthetic native backend without object truth."""
    # BEGIN use the current bounded execution fixture without retired geometry.
    simulation, curve, targets = _fixture()
    curve.events[2].update(kind="effect", requires_evidence=None, stop=False)
    _bind_fingers(simulation, curve, targets)
    result = _result(curve)
    condition = dict(kind="declared_attachment", carrier_id="material_frame",
        condition_id="observed_holding", progress=0., T_carrier_eef=np.eye(4).tolist(),
        source="Explicit synthetic task relation")
    result.execution_binding["problem"]["conditions"] = [condition]
    result.execution_binding["scene"].update(tool_geometry_names=[], owners=[dict(
        owner_id="material_frame", attachments=[dict(interval=[0., 1.],
            condition_id="observed_holding", release_condition_id=None,
            position_tool_owner=[0., 0., 0.], quaternion_tool_owner=[1., 0., 0., 0.])])])
    result.execution_binding["validation"]["inputs"] = deepcopy({
        key: result.execution_binding[key] for key in ("problem", "scene")})
    limits = ExecutionLimits.from_mapping(_limit_values(simulation, curve))
    # END the fixture's accepted inputs remain separate from later mutations.
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
    # BEGIN check_result rejects modifications to the accepted formal binding.
    run, _, simulation, result, *_ = _trial(tmp_path)
    condition = result.execution_binding["problem"]["conditions"][0]
    if change == "owner":
        condition["carrier_id"] = "different_owner"
    elif change == "interval":
        result.execution_binding["scene"]["owners"][0]["attachments"][0]["interval"] = [0., .5]
    elif change == "transform":
        condition["T_carrier_eef"][0][3] = .1
    else:
        condition["kind"] = "candidate_attachment"
    with pytest.raises(ValueError):
        run()
    assert simulation.arm_goals == []
    # END accepted input identity is checked once by the formal result API.
# END no successful CPU sequence certifies a physical attachment.
