"""CPU Holding-permission contracts; mocked clocks never certify physical Holding."""

# BEGIN observed Holding and arrival/close permissions have independent lifetimes.
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
import torch
import test_manipulation_planning_scene as scene_tests
import test_recorded_simulation as recorded_tests

from manipulation.execution import ExecutionGuardError, ExecutionLimits, FixedFingerTargets, TrajectoryExecutor
from manipulation.planning_scene import Attachment, CollisionOwner
from test_independent_scene_validation import checker
from test_manipulation_planning_scene import SphereQuery, inputs, pose_path, scene
from test_recorded_simulation import (_GeometryChecker, _bind_fingers, _execute_bound, _fixture,
    _initial_feedback, _limit_values, _result)
from test_robot_owner_permission import model_binding, native_query, permission


def holding_rule(**changes):
    values = dict(rule_id="current_holding", holding_condition_id="observed_holding",
        arrival_goal_id=(), activation_event={}, interval=(0., 1.))
    values.update(changes)
    return permission(**values)


def holding_fixture(*, release=True):
    simulation, curve, targets = _fixture()
    curve.events[2].update(kind="function", requires_evidence=None, stop=False)
    _bind_fingers(simulation, curve, targets)
    feedback = _initial_feedback(simulation)
    upper = .75 if release else 1.
    condition = {"kind": "observed_attachment", "carrier_id": "target",
        "condition_id": "observed_holding", "contact_rule_ids": ["current_holding"], "progress": 0., "T_carrier_eef": np.eye(4).tolist(),
        "source": {"verified": True, "carrier_id": "target", "robot_feedback_id": feedback["feedback_id"],
            "timestamp_s": feedback["timestamp_s"], "source": "Explicit synthetic entrance Holding observation"}}
    curve.conditions = (deepcopy(condition),)
    result = _result(curve)
    relation = asdict(Attachment((0., upper), [0., 0., 0.], [1., 0., 0., 0.],
        "observed_holding", "declared_release" if release else None))
    relation.update(position_tool_owner=relation["position_tool_owner"].tolist(),
                    quaternion_tool_owner=relation["quaternion_tool_owner"].tolist())
    result.execution_binding["problem"]["conditions"] = [condition]
    result.execution_binding["scene"].update(owners=[{"owner_id": "target", "attachments": [relation]}],
        contacts=[{"rule_id": "current_holding", "actor": "robot", "target_owner": "target",
                   "interval": [0., upper], "sphere_indices": [0], "material_id": "held_pad",
                   "source": "Explicit finite material permission fixture"}])
    refresh_validation(result)
    return simulation, curve, targets, result, ExecutionLimits.from_mapping(_limit_values(simulation, curve))


def refresh_validation(result):
    result.execution_binding["validation"]["inputs"] = deepcopy({
        key: result.execution_binding[key] for key in ("problem", "scene")})


def execute_fixture(fixture, *, checker=None, feedback=None, result_only_goal_ids=(), observation_callback=None):
    simulation, _, targets, result, limits = fixture
    gate = checker or _GeometryChecker(simulation.joint_names)
    executor = TrajectoryExecutor(simulation, geometry_checker=gate)
    if feedback is None:
        feedback = _initial_feedback(simulation)
        feedback["scene_condition_evidence"] = {"observed_holding": True}
    # BEGIN successful sequencing fixtures explicitly supply fresh task facts.
    # The entrance fact must not act as a perpetual geometry permission latch.
    if observation_callback is None:
        def observation_callback(current, curve_time):
            return {"source": "Explicit CPU Holding observation fixture",
                    "timestamp_s": current["timestamp_s"],
                    "scene_condition_evidence": {"observed_holding": True}}
    # END these declared synthetic facts are never simulator contact feedback.
    with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
        report = _execute_bound(executor, result,
            finger_targets=FixedFingerTargets.from_mapping(targets), execution_limits=limits,
            result_only_goal_ids=result_only_goal_ids, initial_feedback=feedback,
            observation_callback=observation_callback, output=Path(temporary)/"trial")
    return report, gate


# BEGIN current evidence and declared release replace runtime geometry latches.
@pytest.mark.parametrize("release", [False, True])
def test_current_holding_dependency_is_independent_of_open_and_ends_at_release(release):
    fixture = holding_fixture(release=release)

    class EntranceChecker(_GeometryChecker):
        def preflight(self, *args, **kwargs):
            self.entrance_arguments = dict(kwargs)
            return super().preflight(*args, **kwargs)

    # After a declared release the attachment estimate is no longer needed.
    # Open alone still does not release a Holding dependency.
    def observation(current, curve_time):
        return {"source": "Fresh declared Holding/release fixture", "timestamp_s": current["timestamp_s"],
                "scene_condition_evidence": {"observed_holding": not release or curve_time < .009}}
    report, gate = execute_fixture(fixture, checker=EntranceChecker(fixture[0].joint_names),
                                   observation_callback=observation)
    assert report["status"] == "curve_finished"
    assert "active_contact_rule_ids" not in gate.entrance_arguments
    # Both the original open and the later close preserve the independently
    # observed attachment; neither command silently means physical release.
    assert [row["kind"] for row in report["events"]][:2] == ["open", "close"]
    assert all("active_contact_rule_ids" not in row for row in gate.states)
    assert report["observed_attachment_bindings"][0]["source"]["verified"] is True
    assert "active_contact_rule_ids" not in report
    assert "holding_contact_permission_changes" not in report
# END entrance provenance is retained separately from current task truth.


@pytest.mark.parametrize("field,value", [
    ("verified", False), ("verified", "U"), ("robot_feedback_id", "old_trial"),
    ("timestamp_s", 1.2), ("timestamp_s", float("nan")), ("carrier_id", "another_owner"), ("source", ""),
])
def test_holding_activation_rejects_stale_false_unknown_or_unsourced_evidence_before_control(field, value):
    fixture = holding_fixture()
    simulation, _, _, result, _ = fixture
    result.execution_binding["problem"]["conditions"][0]["source"][field] = value
    refresh_validation(result)
    with pytest.raises(ValueError):
        execute_fixture(fixture)
    assert simulation.arm_goals == [] and simulation.sim_time == 1.3


@pytest.mark.parametrize("change", ["owner", "interval", "transform", "duplicate", "actor"])
def test_holding_activation_cannot_relabel_a_different_condition_or_attachment(change):
    fixture = holding_fixture()
    binding = fixture[3].execution_binding
    condition = binding["problem"]["conditions"][0]
    if change == "kind":
        condition["kind"] = "candidate_attachment"
    elif change == "owner":
        condition["carrier_id"] = "another_owner"
    elif change == "interval":
        binding["scene"]["owners"][0]["attachments"][0]["interval"] = [0., .5]
    elif change == "transform":
        condition["T_carrier_eef"][0][3] = .1
    elif change == "duplicate":
        binding["problem"]["conditions"].append(deepcopy(condition))
    else:
        binding["scene"]["contacts"][0]["actor"] = "another_owner"
    refresh_validation(fixture[3])
    with pytest.raises(ValueError):
        execute_fixture(fixture)
    assert fixture[0].arm_goals == []


def test_explicit_current_false_holding_cannot_be_hidden_by_verified_entrance_record():
    fixture = holding_fixture()
    feedback = _initial_feedback(fixture[0])
    feedback["scene_condition_evidence"] = {"observed_holding": False}
    with pytest.raises(ValueError, match="positively verify"):
        execute_fixture(fixture, feedback=feedback)
    assert fixture[0].arm_goals == []


def test_current_holding_does_not_bypass_robot_state_rejections():
    fixture = holding_fixture()
    gate = _GeometryChecker(fixture[0].joint_names, rejected_stage="command_event", rejected_time=.002)
    with pytest.raises(ExecutionGuardError):
        execute_fixture(fixture, checker=gate)
    assert gate.states[-1]["stage"] == "command_event"
    assert "active_contact_rule_ids" not in gate.states[-1]


def test_structured_holding_provenance_is_retained_without_rewriting_the_input():
    fixture = holding_fixture()
    result = fixture[3]
    provenance = {"kind": "explicit_test_result", "geometry_source": "current RGB-D and robot FK",
                  "effect_source": "separately declared Holding evidence"}
    result.execution_binding["problem"]["conditions"][0]["source"]["source"] = provenance
    refresh_validation(result)
    report, _ = execute_fixture(fixture)
    assert report["observed_attachment_bindings"][0]["source"]["source"] == provenance


def test_arrival_close_and_later_open_keep_independent_holding_evidence():
    fixture = recorded_tests.GraspContactActivationTests.fixture(reopen=True)
    simulation, _, _, result, _ = fixture
    _, _, _, holding_result, _ = holding_fixture(release=False)
    condition = deepcopy(holding_result.execution_binding["problem"]["conditions"][0])
    condition["source"]["robot_feedback_id"] = _initial_feedback(simulation)["feedback_id"]
    result.execution_binding["problem"]["conditions"].append(condition)
    result.execution_binding["scene"]["owners"] = holding_result.execution_binding["scene"]["owners"]
    result.execution_binding["scene"]["contacts"].extend(
        holding_result.execution_binding["scene"]["contacts"])
    refresh_validation(result)
    report, _ = execute_fixture(fixture, result_only_goal_ids=[["pose_events", 0]])
    assert report["status"] == "curve_finished"
    assert report["grasp_contact_gate_checks"][-1]["control_applied"]
    assert report["observed_attachment_bindings"][0]["condition_id"] == "observed_holding"
    assert "grasp_contact_permission_changes" not in report
    assert "active_contact_rule_ids" not in report


@pytest.mark.parametrize("change", [None, "kind", "owner", "interval", "transform"])
def test_planning_scene_requires_exact_observed_attachment_for_holding_permission(change):
    original, problem = scene_tests.PlanningSceneTest().bound_payload()
    condition = problem.conditions[0]
    condition["T_carrier_eef"] = np.eye(4).tolist()
    condition["T_carrier_eef"][0][3] = -.3
    world = scene(original.owners, robot_owner_permissions=(holding_rule(target_owner="payload"),))
    if change == "kind":
        condition["kind"] = "candidate_attachment"
    elif change == "owner":
        condition["carrier_id"] = "different_owner"
    elif change == "interval":
        world.robot_owner_permissions[0].interval = (0., .5)
    elif change == "transform":
        condition["T_carrier_eef"][0][3] = -.2
    if change is None:
        world.validate_problem(problem)
    else:
        with pytest.raises(ValueError, match="Holding permission"):
            world.validate_problem(problem)


def test_holding_rule_has_no_fake_arrival_or_close_and_old_archive_stays_arrival_based():
    for changes in ({"arrival_goal_id": ("pose_events", 0)}, {"activation_event": {"kind": "close"}},
                    {"holding_condition_id": ""}):
        with pytest.raises(ValueError):
            holding_rule(**changes)
    legacy = asdict(permission())
    legacy.pop("holding_condition_id")
    assert type(permission())(**legacy).holding_condition_id is None


@pytest.mark.parametrize("release", [False, True])
def test_nominal_holding_permission_ends_at_real_release_and_keeps_other_spheres(release):
    relation = Attachment((0., 1.), [0., 0., 0.], [1., 0., 0., 0.], "observed_holding",
                          "declared_release" if release else None)
    owner = CollisionOwner("target", SphereQuery(("target",), center=(0., 0., 0.)), pose_path(),
        "observed target", True, local_spheres=[[0., 0., 0., .1]], attachments=(relation,))
    world = scene((owner,), materials=(None, None), robot_owner_permissions=(holding_rule(),))
    terms = world.components(*inputs(progress=(.5, 1.), sphere_rows=[[[0., 0., 0., .1]]*2]*2))
    torch.testing.assert_close(terms["collision/robot/target"],
        torch.tensor([[.2, .4 if release else .2]], dtype=torch.float64))


def test_native_holding_permission_keeps_false_conditions_unknown_geometry_and_release_boundary(checker):
    model = model_binding(holding_condition_id="observed_holding", arrival_goal_id=(), activation_event={}, interval=(0., 1.))
    model["owners"][0]["attachments"] = [asdict(Attachment((0., 1.), [0., 0., 0.], [1., 0., 0., 0.],
        "observed_holding", "declared_release"))]
    checker.bind_scene(model)
    robot = checker.query(joint_names=checker.joint_names, joint_positions=[0., 0.], T_reference_base=np.eye(4), pairs=())
    result = checker.query_scene(robot, tool_pose=np.eye(4), progress=.5, finger_positions=[0., 0.],
        state_kind="actual", condition_evidence={"observed_holding": False},
        active_robot_owner_permissions={"arrived_close"})
    assert not result["passed"]
    assert any(row["kind"] == "current_scene_binding_invalid" for row in result["violations"])
    assert result["pairs"][0]["permission"] == "arrived_close"
    assert result["pairs"][0]["permission_authorization"] == "current_execution_holding_binding"
    # Nominal and actual geometry reject a stale active set at the real release;
    # a current owner pose supplies geometry without satisfying a release effect.
    for kind in ("actual", "nominal"):
        result = checker.query_scene(robot, tool_pose=np.eye(4), progress=1., finger_positions=[0., 0.],
            state_kind=kind, owner_poses={"target": np.eye(4)}, active_robot_owner_permissions={"arrived_close"})
        assert not result["passed"] and result["pairs"][0]["permission"] is None
    model["owners"][0]["geometry"]["query_gaps"] = [{"reason": "explicit unsupported target primitive"}]
    assert not native_query(checker, model, active={"arrived_close"})["passed"]
# END all physical truth remains outside these CPU contract fixtures.
