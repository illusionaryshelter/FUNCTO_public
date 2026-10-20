"""CPU finite-material activation checks, separate from physical grasp success."""

# BEGIN finite grasp rules retain real arrival gating through the existing audit checker.
from copy import deepcopy

import numpy as np
import pytest
import test_manipulation_planning_scene as scene_tests
import test_recorded_simulation as execution_tests

from manipulation.planning_problem import MotionPhase, PlanningEvent, PoseEventRequirement
from manipulation.execution_binding import curve_snapshot, plain
from manipulation.planning_scene import CollisionOwner, ContactRule, grasp_contact_rule_ids
from test_independent_scene_validation import binding, checker, rule
from test_manipulation_planning_problem import problem
from test_manipulation_planning_scene import SphereQuery, pose_path, scene


def activation(**changes):
    """Name an original close and existing finite material rule, without a pair waiver."""
    row = {"kind": "grasp_contact_activation", "condition_id": "finite_arrival",
        "carrier_id": "target", "arrival_goal_id": ["pose_events", 0],
        "activation_event": {"skill_id": "skill", "event_id": "close", "occurrence": 0, "kind": "close"},
        "interval": [.5, .9], "contact_rule_ids": ["pad_to_target"],
        "source": "Original explicit finite grasp binding"}
    row.update(changes)
    return row


def arrival_source(condition):
    return {"kind": "frozen_perception_grasp_arrival", "carrier_id": condition["carrier_id"],
            "activation_event": deepcopy(condition["activation_event"])}


def model_binding(**rule_changes):
    model = binding([rule(interval=(.5, .9), **rule_changes)])
    model["grasp_contact_activations"] = [activation()]
    return model


def native_query(checker, model, *, progress=.6, kind="actual", active=frozenset()):
    checker.bind_scene(model)
    state = checker.query(joint_names=checker.joint_names, joint_positions=[0., 0.],
        T_reference_base=np.eye(4), pairs=())
    return checker.query_scene(state, tool_pose=np.eye(4), progress=progress,
        finger_positions=[0., 0.], state_kind=kind, active_contact_rule_ids=active)


def test_finite_material_needs_actual_latch_and_unchanged_interval(checker):
    model = model_binding()
    assert not native_query(checker, model)["passed"]
    result = native_query(checker, model, active={"pad_to_target"})
    assert result["passed"] and result["pairs"][0]["permission"] == "pad_to_target"
    assert result["pairs"][0]["contact_region_checks"][0]["contained"]
    assert result["active_robot_owner_permissions"] == []
    assert result["active_contact_rule_ids"] == ["pad_to_target"]
    for progress in (.499, .901):
        assert not native_query(checker, model, progress=progress, active={"pad_to_target"})["passed"]
    assert native_query(checker, model, kind="nominal")["passed"]
    assert not native_query(checker, model)["passed"]  # A prior query is never an execution latch.
    with pytest.raises(ValueError, match="bound scene"):
        native_query(checker, model, active={"other_rule"})
    with pytest.raises(ValueError, match="explicit active set"):
        native_query(checker, model, active=None)


def test_latched_finite_rule_still_checks_entire_material_overlap(checker):
    model = model_binding(patch_half_size=[.01, .01])
    rejected = native_query(checker, model, active={"pad_to_target"})
    assert not rejected["passed"]
    assert rejected["pairs"][0]["permission"] is None
    assert rejected["pairs"][0]["contact_region_checks"][0]["contained"] is False
    # An ordinary independently declared finite rule has no grasp event latch.
    assert native_query(checker, binding([rule()]))["passed"]


# BEGIN execution no longer accepts frozen-scene permission latches.
# Runtime world-query sentries and actual device failures are covered by
# test_execution_responsibilities; finite geometry remains a planning contract.
# END remove the obsolete validator latch forwarding test.


@pytest.mark.parametrize("changes", [
    {"contact_rule_ids": []}, {"contact_rule_ids": ["pad_to_target", "pad_to_target"]},
    {"contact_rule_ids": ["missing"]}, {"carrier_id": "other"}, {"interval": [.4, .9]},
])
def test_activation_cannot_rebind_a_rule_to_another_identity_or_scope(changes):
    with pytest.raises(ValueError):
        grasp_contact_rule_ids(activation(**changes), [rule(interval=(.5, .9))])


def scene_fixture():
    hand = scene_tests.PlanningSceneTest.timed_gripper()
    hand.commands = ({"skill_id": "skill", "event_id": "close", "occurrence": 0,
                      "command": "close", "progress": .5},)
    material = ContactRule(**rule(interval=(.5, .9),
        feature_joint_translation=np.broadcast_to([[1., 0.], [0., 0.], [0., 0.]], (4, 3, 2))))
    owner = CollisionOwner("target", SphereQuery(("target",)), pose_path(), "observed target", True)
    current = scene((owner,), materials=("contact_pad", None), contacts=(material,),
                    gripper=hand, has_gripper=True)
    condition = activation()
    request = problem([MotionPhase("reference", pose_path())], conditions=(condition,),
        events=(PlanningEvent("close", "skill", .5, 10, "close", "original close"),),
        pose_events=(PoseEventRequirement(.5, np.eye(4), np.eye(4), .001, .01, arrival_source(condition)),))
    return current, request, condition


def test_planning_scene_archives_finite_activation_without_whole_hand_permission():
    current, request, condition = scene_fixture()
    current.validate_problem(request)
    archived = current.to_execution_binding()
    assert archived["robot_owner_permissions"] == []
    assert archived["grasp_contact_activations"] == [condition]
    condition["contact_rule_ids"].append("unbound")
    assert archived["grasp_contact_activations"][0]["contact_rule_ids"] == ["pad_to_target"]
    assert current.clone_for_rollout().grasp_contact_activations[0]["contact_rule_ids"] == ["pad_to_target"]


def test_filtered_goal_identity_maps_to_current_arrival_but_consumed_close_needs_rebind():
    current, request, condition = scene_fixture()
    condition["arrival_goal_id"] = ["pose_events", 7]
    with pytest.raises(ValueError, match="fresh arrival/close binding"):
        current.validate_problem(request)
    current.validate_problem(request, goal_identities={"pose_events": [7]})
    assert current.grasp_contact_activations[0]["arrival_goal_id"] == ["pose_events", 7]
    request.events = ()
    with pytest.raises(ValueError, match="prior latch is not reusable"):
        current.validate_problem(request, goal_identities={"pose_events": [7]})


@pytest.mark.parametrize("field,value", [("carrier_id", "other"), ("activation_event", {
    "skill_id": "skill", "event_id": "another_close", "occurrence": 0, "kind": "close"})])
def test_another_arrival_at_the_same_time_cannot_activate_finite_grasp(field, value):
    fixture = finite_execution_fixture()
    bound = fixture[3].execution_binding
    bound["problem"]["pose_events"][1]["source"][field] = value
    bound["validation"]["inputs"] = deepcopy({key: bound[key] for key in ("problem", "scene")})
    with pytest.raises(ValueError, match="exact target owner and original close source"):
        execution_tests.GraspContactActivationTests.run_trial(fixture)


def finite_execution_fixture(*, reopen=False, zero_tick=False):
    fixture = execution_tests.GraspContactActivationTests.fixture(reopen=reopen, zero_tick=zero_tick)
    _, curve, _, result, _ = fixture
    bound = result.execution_binding
    condition = bound["problem"]["conditions"][0]
    condition["contact_rule_ids"] = ["pad_to_target"]
    bound["problem"]["pose_events"][1]["source"] = arrival_source(condition)
    curve.conditions = (deepcopy(condition),)
    bound["curve"] = curve_snapshot(curve)
    bound["scene"].update(robot_owner_permissions=[],
        contacts=plain([rule(target_owner=condition["carrier_id"], interval=condition["interval"])]),
        grasp_contact_activations=[deepcopy(condition)])
    bound["validation"]["inputs"] = deepcopy({key: bound[key] for key in ("problem", "scene")})
    return fixture


@pytest.mark.parametrize("arrived", [False, True])
def test_target_error_does_not_veto_close_or_grant_world_geometry_permissions(arrived):
    fixture = finite_execution_fixture(zero_tick=True)
    if not arrived:
        fixture[0].scripted_tcp_estimate_offset[0] = .002
    report, checker = execution_tests.GraspContactActivationTests.run_trial(fixture)
    # BEGIN target accuracy is posthoc; command timing creates no contact facts.
    assert "active_robot_owner_permissions" not in report
    assert "active_contact_rule_ids" not in report
    assert all("active_contact_rule_ids" not in row for row in checker.states)
    assert report["status"] == "curve_finished"
    assert report["grasp_command_checks"][-1]["control_applied"]
    assert report["completed_goal_ids"] == []
    # END a target estimate miss must still reach the actual close command.


def test_open_after_close_does_not_create_runtime_geometry_permission():
    report, checker = execution_tests.GraspContactActivationTests.run_trial(finite_execution_fixture(reopen=True))
    assert report["status"] == "curve_finished"
    assert "active_contact_rule_ids" not in report
    assert all("active_contact_rule_ids" not in row for row in checker.states)
    assert report["grasp_command_checks"][-1]["control_applied"]
    assert [row["kind"] for row in report["events"]] == ["open", "close", "open"]


def test_planning_permission_interval_does_not_create_an_execution_expiration_gate():
    fixture = finite_execution_fixture()
    bound = fixture[3].execution_binding
    bound["problem"]["conditions"][0]["interval"][1] = .8
    bound["scene"]["contacts"][0]["interval"] = [.25, .8]
    bound["scene"]["grasp_contact_activations"] = deepcopy(bound["problem"]["conditions"])
    fixture[1].conditions = tuple(deepcopy(bound["problem"]["conditions"]))
    bound["curve"] = curve_snapshot(fixture[1])
    bound["validation"]["inputs"] = deepcopy({key: bound[key] for key in ("problem", "scene")})
    report, checker = execution_tests.GraspContactActivationTests.run_trial(fixture)
    assert report["status"] == "curve_finished"
    assert "active_contact_rule_ids" not in report
    assert "finite_contact_permission_expirations" not in report
    assert all("active_contact_rule_ids" not in row for row in checker.states)
# END CPU contracts do not claim an online TSDF migration or simulated grasp success.
