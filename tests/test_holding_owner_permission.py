"""CPU Holding-permission contracts; mocked clocks never certify physical Holding."""

# BEGIN declared attachment and arrival/close permissions have independent lifetimes.
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
    upper = .75 if release else 1.
    condition = {"kind": "declared_attachment", "carrier_id": "target",
        "condition_id": "observed_holding", "contact_rule_ids": ["current_holding"], "progress": 0., "T_carrier_eef": np.eye(4).tolist(),
        "source": {"source": "Explicit synthetic nominal attachment model"}}
    curve.conditions = (deepcopy(condition),)
    result = _result(curve)
    relation = asdict(Attachment((0., upper), [0., 0., 0.], [1., 0., 0., 0.],
        "observed_holding", "declared_release" if release else None))
    relation.update(position_tool_owner=relation["position_tool_owner"].tolist(),
                    quaternion_tool_owner=relation["quaternion_tool_owner"].tolist())
    result.execution_binding["problem"]["conditions"] = [condition]
    result.execution_binding["scene"].update(owners=[{"owner_id": "target", "attachments": [relation]}],
        # BEGIN Holding fixtures bind current materials, without a grasp target.
        holding_contacts=[{"rule_id": "current_holding", "condition_id": "observed_holding", "target_owner": "target",
                   "interval": [0., upper], "sphere_indices": [0], "material_id": "held_pad", "native_geometry": "pad",
                   "source": "Explicit current held material fixture"}])
        # END known material identity never fabricates task evidence.
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
    # BEGIN no physical Holding observation is required to use a nominal model.
    # Robot feedback still supplies the ordinary joint/model safety checks.
    # END neither fixture creation nor tick callbacks manufacture Holding=True.
    with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
        report = _execute_bound(executor, result,
            finger_targets=FixedFingerTargets.from_mapping(targets), execution_limits=limits,
            result_only_goal_ids=result_only_goal_ids, initial_feedback=feedback,
            observation_callback=observation_callback, output=Path(temporary)/"trial")
    return report, gate


# BEGIN a declared release bounds the model without any runtime Holding latch.
@pytest.mark.parametrize("release", [False, True])
def test_declared_attachment_runs_without_physical_proof_and_keeps_release_interval(release):
    fixture = holding_fixture(release=release)

    class EntranceChecker(_GeometryChecker):
        def preflight(self, *args, **kwargs):
            self.entrance_arguments = dict(kwargs)
            return super().preflight(*args, **kwargs)

    report, gate = execute_fixture(fixture, checker=EntranceChecker(fixture[0].joint_names))
    assert report["status"] == "curve_finished"
    assert "active_contact_rule_ids" not in gate.entrance_arguments
    # Open/close playback does not assert the physical success of the model.
    assert [row["kind"] for row in report["events"]][:2] == ["open", "close"]
    assert all("active_contact_rule_ids" not in row for row in gate.states)
    attachment = report["declared_attachment_bindings"][0]
    assert attachment["source"] == {"source": "Explicit synthetic nominal attachment model"}
    assert attachment["interval"] == [0., .75 if release else 1.]
    assert "active_contact_rule_ids" not in report
    assert "holding_contact_permission_changes" not in report
# END the report identifies a declaration without promoting it to an observation.


@pytest.mark.parametrize("change", ["kind", "owner", "interval", "transform", "duplicate", "condition"])
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
        binding["scene"]["holding_contacts"][0]["condition_id"] = "another_condition"
    refresh_validation(fixture[3])
    with pytest.raises(ValueError):
        execute_fixture(fixture)
    assert fixture[0].arm_goals == []


@pytest.mark.parametrize("physical_score", [False, True])
def test_physical_holding_score_cannot_veto_or_certify_a_declared_model(physical_score):
    fixture = holding_fixture()
    feedback = _initial_feedback(fixture[0])
    feedback["scene_condition_evidence"] = {"observed_holding": physical_score}
    report, _ = execute_fixture(fixture, feedback=feedback)
    assert report["status"] == "curve_finished"
    assert "verified" not in report["declared_attachment_bindings"][0]["source"]


def test_current_holding_does_not_bypass_robot_state_rejections():
    fixture = holding_fixture()
    gate = _GeometryChecker(fixture[0].joint_names, rejected_stage="command_event", rejected_time=.002)
    with pytest.raises(ExecutionGuardError):
        execute_fixture(fixture, checker=gate)
    assert gate.states[-1]["stage"] == "command_event"
    assert "active_contact_rule_ids" not in gate.states[-1]


def test_structured_model_source_is_retained_without_rewriting_the_input():
    fixture = holding_fixture()
    result = fixture[3]
    provenance = {"kind": "explicit_nominal_model", "geometry_source": "known CAD and robot FK",
                  "attachment_model": "caller supplied identity transform"}
    result.execution_binding["problem"]["conditions"][0]["source"]["source"] = provenance
    refresh_validation(result)
    report, _ = execute_fixture(fixture)
    assert report["declared_attachment_bindings"][0]["source"]["source"] == provenance


def test_arrival_close_and_later_open_keep_the_same_declared_attachment():
    fixture = recorded_tests.GraspContactActivationTests.fixture(reopen=True)
    simulation, _, _, result, _ = fixture
    _, _, _, holding_result, _ = holding_fixture(release=False)
    condition = deepcopy(holding_result.execution_binding["problem"]["conditions"][0])
    result.execution_binding["problem"]["conditions"].append(condition)
    result.execution_binding["scene"]["owners"] = holding_result.execution_binding["scene"]["owners"]
    result.execution_binding["scene"]["holding_contacts"] = deepcopy(
        holding_result.execution_binding["scene"]["holding_contacts"])
    refresh_validation(result)
    report, _ = execute_fixture(fixture, result_only_goal_ids=[["pose_events", 0]])
    assert report["status"] == "curve_finished"
    assert report["grasp_command_checks"][-1]["control_applied"]
    assert report["declared_attachment_bindings"][0]["condition_id"] == "observed_holding"
    assert "grasp_contact_permission_changes" not in report
    assert "active_contact_rule_ids" not in report


@pytest.mark.parametrize("change", [None, "kind", "owner", "interval", "transform"])
def test_planning_scene_requires_exact_declared_attachment_for_holding_permission(change):
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


def test_native_declared_permission_keeps_unknown_geometry_and_release_boundary(checker):
    model = model_binding(holding_condition_id="observed_holding", arrival_goal_id=(), activation_event={}, interval=(0., 1.))
    model["owners"][0]["attachments"] = [asdict(Attachment((0., 1.), [0., 0., 0.], [1., 0., 0., 0.],
        "observed_holding", "declared_release"))]
    checker.bind_scene(model)
    robot = checker.query(joint_names=checker.joint_names, joint_positions=[0., 0.], T_reference_base=np.eye(4), pairs=())
    result = checker.query_scene(robot, tool_pose=np.eye(4), progress=.5, finger_positions=[0., 0.],
        state_kind="actual",
        active_robot_owner_permissions={"arrived_close"})
    assert result["passed"] and not result["violations"]
    assert not result["observed_scene_checked"]
    assert result["pairs"][0]["permission"] == "arrived_close"
    assert result["pairs"][0]["permission_authorization"] == "declared_attachment_model"
    # Nominal and actual geometry reject a stale active set at the real release;
    # a current owner pose supplies geometry without satisfying a release effect.
    for kind in ("actual", "nominal"):
        result = checker.query_scene(robot, tool_pose=np.eye(4), progress=1., finger_positions=[0., 0.],
            state_kind=kind, owner_poses={"target": np.eye(4)}, active_robot_owner_permissions={"arrived_close"})
        assert not result["passed"] and result["pairs"][0]["permission"] is None
    model["owners"][0]["geometry"]["query_gaps"] = [{"reason": "explicit unsupported target primitive"}]
    assert not native_query(checker, model, active={"arrived_close"})["passed"]
# END all physical truth remains outside these CPU contract fixtures.
