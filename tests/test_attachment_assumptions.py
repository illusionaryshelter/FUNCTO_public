"""Explicit attachment models authorize playback without inventing observations."""

# BEGIN bounded CPU contracts for the separate predicate-estimation boundary.
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from manipulation.execution import (ExecutionGuardError, FixedFingerTargets,
    TrajectoryExecutor, _feedback_json, _holding_dependency_violations)
from manipulation.execution_binding import rebind_execution_result
from test_holding_execution import holding_fixture, refresh_validation
from test_recorded_simulation import _GeometryChecker, _initial_feedback


def _trial(tmp_path, *, assumption=None, initial_value="absent", observation=None,
           rejected_stage=None, mutate=None):
    """Use the existing synthetic native backend, never real simulator truth."""
    simulation, _, targets, result, limits = holding_fixture()
    if mutate is not None:
        mutate(result)
        refresh_validation(result)
    fingers = FixedFingerTargets.from_mapping(targets)
    result = rebind_execution_result(result,
        controller_policy=simulation.describe()["controller_policy"],
        result_only_goal_ids=[], finger_targets=fingers, execution_limits=limits,
        source="Explicit attachment-assumption CPU fixture")
    feedback = _initial_feedback(simulation)
    if initial_value != "absent":
        feedback["scene_condition_evidence"] = {"observed_holding": initial_value}
    gate = _GeometryChecker(simulation.joint_names, rejected_stage=rejected_stage,
                            rejected_time=.002)
    executor = TrajectoryExecutor(simulation, geometry_checker=gate)
    original = deepcopy((result.execution_binding, feedback, assumption))

    def run():
        with patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            return executor.execute(result, finger_targets=fingers, initial_feedback=feedback,
                execution_limits=limits, observation_callback=observation,
                attachment_assumptions=assumption, output=tmp_path / "execution")

    return run, executor, simulation, result, feedback, original


def test_assumption_allows_playback_without_manufacturing_semantics(tmp_path):
    assumption = {"observed_holding": {"kind": "explicit_user_authorization",
        "description": "Assume the accepted attachment remains valid during transport"}}
    run, _, _, result, feedback, original = _trial(tmp_path, assumption=assumption)
    report = run()
    assert report["status"] == "curve_finished"
    assert report["executed_steps"] > 0
    assert report["effect_result"] == "U" and report["effects"] == {}
    assert "scene_condition_evidence" not in report.get("latest_observation", {})
    assert "scene_owner_poses" not in report.get("latest_observation", {})
    assert report["completed_goal_ids"] == []
    assert report["attachment_assumptions"] == [{"kind": "model_assumption",
        "condition_id": "observed_holding", "owner_id": "target", "interval": [0., .75],
        "release_condition_id": "declared_release", "source": assumption["observed_holding"]}]
    # The pre-existing executor adds its three physics-budget accounting fields;
    # every original formal field and all caller-owned evidence stay unchanged.
    assert {key: result.execution_binding[key] for key in original[0]} == original[0]
    assert _feedback_json(feedback) == _feedback_json(original[1])
    assert assumption == original[2]
    assert "scene_condition_evidence" not in report["continuation"]["actual_feedback"]


@pytest.mark.parametrize("initial_value", [False, None])
def test_initial_explicit_false_or_unknown_overrides_assumption(tmp_path, initial_value):
    run, _, simulation, *_ = _trial(tmp_path, initial_value=initial_value,
        assumption={"observed_holding": "Explicit transport assumption"})
    with pytest.raises(ValueError, match="positively verify"):
        run()
    assert simulation.arm_goals == []


@pytest.mark.parametrize("value", [False, None])
def test_current_explicit_false_or_unknown_overrides_assumption(tmp_path, value):
    def observe(feedback, curve_time):
        return {"source": "Independent synthetic predicate estimate",
            "timestamp_s": feedback["timestamp_s"],
            "scene_condition_evidence": {} if curve_time == 0 else {"observed_holding": value}}
    run, executor, _, *_ = _trial(tmp_path, observation=observe,
        assumption={"observed_holding": "Explicit transport assumption"})
    with pytest.raises(ExecutionGuardError, match="actual_holding_unverified"):
        run()
    assert executor.last_execution_report["executed_steps"] == 1


@pytest.mark.parametrize("assumption", [None, {"wrong_condition": "Declared source"},
    {"observed_holding": " "}, {"observed_holding": {}}, ["observed_holding"]])
def test_missing_or_invalid_declaration_cannot_authorize_control(tmp_path, assumption):
    run, _, simulation, *_ = _trial(tmp_path, assumption=assumption, initial_value=True)
    expected = ExecutionGuardError if assumption is None else ValueError
    with pytest.raises(expected):
        run()
    if assumption is not None:
        assert simulation.arm_goals == []


def test_assumption_cannot_relax_robot_guard(tmp_path):
    run, executor, _, *_ = _trial(tmp_path, rejected_stage="command_event",
        assumption={"observed_holding": "Explicit transport assumption"})
    with pytest.raises(ExecutionGuardError):
        run()
    assert executor.last_execution_report["feedback_failure"]["violations"]


@pytest.mark.parametrize("change", ["owner", "interval", "transform", "kind"])
def test_assumption_cannot_rebind_attachment_identity_or_geometry(tmp_path, change):
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
    run, _, simulation, *_ = _trial(tmp_path, mutate=mutate,
        assumption={"observed_holding": "Explicit transport assumption"})
    with pytest.raises(ValueError):
        run()
    assert simulation.arm_goals == []


def test_assumption_uses_bound_lifetime_and_keeps_observed_pose_independent():
    scene = {"owners": [{"owner_id": "target", "attachments": [{
        "condition_id": "holding", "interval": [0., .75], "release_condition_id": "release"}]}]}
    assumed = {"holding": "Explicit transport assumption"}
    assert not _holding_dependency_violations(scene, .2, None, attachment_assumptions=assumed)
    # At the declared release boundary the model is no longer needed, even
    # when an independently supplied predicate correctly reports False.
    contrary = {"scene_condition_evidence": {"holding": False}}
    assert not _holding_dependency_violations(scene, .75, contrary, attachment_assumptions=assumed)
    assert _holding_dependency_violations(scene, .2, contrary, attachment_assumptions=assumed)
    contrary["scene_owner_poses"] = {"target": np.eye(4)}
    assert not _holding_dependency_violations(scene, .2, contrary, attachment_assumptions=assumed)
# END no successful CPU sequencing test certifies a physical attachment.
