"""CPU contracts for generated grasp bindings and the complete skill builder.

These synthetic transforms test geometry, provenance and event ownership. They
do not assert learned grasp quality, robot reachability or physical Holding.
"""

# BEGIN candidate changes preserve source motion and physical binding identity.
from copy import deepcopy

import numpy as np
import pytest

from manipulation.grasp import bind_grasp_candidate, ground_grasp
from manipulation.planning_problem import build_planning_problem
from test_manipulation_grasp import example, pose
from test_manipulation_planning_problem import grasp_input


def bound_example():
    """Keep a deliberate difference between the prior and last source sample."""
    demo, scene = example()
    scene["reference_frame"] = "world"
    scene["grasp_goal"]["T_carrier_eef"] = (
        scene["grasp_goal"]["T_carrier_eef"] @ pose((.009, -.012, .004), (.1, -.2, .05)))
    scene["finite_contact_evidence"] = {"patch_origin": [.3, -.1, .2],
                                        "source": "independent allowed material domain"}
    return demo, scene


def test_exact_prior_preserves_every_reference_value_and_original_inputs():
    demo, scene = bound_example()
    original_demo, original_scene = deepcopy(demo), deepcopy(scene)
    original = ground_grasp(demo, scene)
    selected = bind_grasp_candidate(scene, original["goal_pose"], source={"branch": "demo"})
    actual = ground_grasp(demo, selected)
    np.testing.assert_array_equal(actual["eef_poses"], original["eef_poses"])
    np.testing.assert_array_equal(actual["goal_pose"], original["goal_pose"])
    assert actual["holding_status"] == "requires_verification"
    assert "actual_T_carrier_eef" not in actual
    np.testing.assert_array_equal(selected["region_transform"], original_scene["region_transform"])
    np.testing.assert_array_equal(selected["eef_pose"], original_scene["eef_pose"])
    np.testing.assert_array_equal(scene["grasp_goal"]["T_carrier_eef"], original_scene["grasp_goal"]["T_carrier_eef"])
    np.testing.assert_array_equal(demo["T_carrier_eef"], original_demo["T_carrier_eef"])
    assert "grasp_reference_prior" not in scene
    assert selected["finite_contact_evidence"] == scene["finite_contact_evidence"]
    selected["finite_contact_evidence"]["patch_origin"][0] += 1.
    assert scene["finite_contact_evidence"] == original_scene["finite_contact_evidence"]
    assert actual["grasp_reference_prior"]["source"] == scene["grasp_goal"]["source"]
    assert actual["grasp_goal"]["source"] == {"branch": "demo"}


def test_rotated_candidate_moves_all_samples_about_the_independent_prior():
    demo, scene = bound_example()
    original = ground_grasp(demo, scene)
    selected_pose = pose((-.2, .4, .25), (-.7, .3, .8))
    selected = bind_grasp_candidate(scene, selected_pose, source="diffusion candidate")
    actual = ground_grasp(demo, selected)
    expected = selected_pose @ np.linalg.inv(original["goal_pose"]) @ original["eef_poses"]
    np.testing.assert_allclose(actual["eef_poses"], expected, atol=2e-15, rtol=0)
    np.testing.assert_allclose(np.linalg.inv(actual["goal_pose"]) @ actual["eef_poses"],
                               np.linalg.inv(original["goal_pose"]) @ original["eef_poses"],
                               atol=2e-15, rtol=0)
    assert not np.allclose(actual["eef_poses"][-1], actual["goal_pose"])
    assert np.all(np.linalg.norm(actual["eef_poses"][:, :3, 3]
                                - original["eef_poses"][:, :3, 3], axis=1) > .1)
    np.testing.assert_array_equal(actual["frame_indices"], original["frame_indices"])
    np.testing.assert_array_equal(actual["timestamps"], original["timestamps"])
    assert actual["gripper_events"] == original["gripper_events"]
    assert actual["effect_requirement"] == original["effect_requirement"]
    for key in ("position_tolerance_m", "rotation_tolerance_rad"):
        assert selected["grasp_goal"][key] == scene["grasp_goal"][key]


def test_rebinding_uses_original_anchor_and_nearby_candidates_are_not_snapped():
    demo, scene = bound_example()
    original = ground_grasp(demo, scene)
    first = bind_grasp_candidate(scene, pose((-.1, .2, .4)), source="first candidate")
    second_pose = pose((.2, -.3, .5), (.4, .2, -.5))
    rebound = bind_grasp_candidate(first, second_pose, source="second candidate")
    direct = bind_grasp_candidate(scene, second_pose, source="second candidate")
    np.testing.assert_array_equal(ground_grasp(demo, rebound)["eef_poses"],
                                  ground_grasp(demo, direct)["eef_poses"])
    np.testing.assert_array_equal(rebound["grasp_reference_prior"]["T_carrier_eef"],
                                  scene["grasp_goal"]["T_carrier_eef"])
    recovered = bind_grasp_candidate(first, original["goal_pose"], source="original prior")
    np.testing.assert_array_equal(ground_grasp(demo, recovered)["eef_poses"], original["eef_poses"])
    nearby = original["goal_pose"].copy()
    nearby[0, 3] += 1e-8
    actual = ground_grasp(demo, bind_grasp_candidate(scene, nearby, source="distinct nearby candidate"))
    assert not np.array_equal(actual["goal_pose"], original["goal_pose"])
    np.testing.assert_allclose(actual["goal_pose"], nearby, atol=1e-15, rtol=0)


def test_candidate_binding_is_equivariant_to_output_and_owner_coordinates():
    demo, scene = bound_example()
    target = pose((-.2, .1, .35), (.6, -.2, .4))
    selected = bind_grasp_candidate(scene, target, source="candidate")
    expected = ground_grasp(demo, selected)
    world = pose((.2, .3, -.1), (.1, -.4, .8))
    owner = pose((.04, -.06, .08), (-.4, .7, -.1))
    changed = deepcopy(selected)
    changed["carrier_pose"] = world @ scene["carrier_pose"] @ np.linalg.inv(owner)
    changed["eef_pose"] = world @ scene["eef_pose"]
    changed["region_transform"] = owner @ scene["region_transform"]
    for key in ("grasp_goal", "grasp_reference_prior"):
        changed[key]["T_carrier_eef"] = owner @ selected[key]["T_carrier_eef"]
    actual = ground_grasp(demo, changed)
    np.testing.assert_allclose(actual["eef_poses"], world @ expected["eef_poses"], atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual["goal_pose"], world @ target, atol=2e-15, rtol=0)


@pytest.mark.parametrize("invalid", [None, {}, {"T_carrier_eef": np.eye(4)},
                                     {"T_carrier_eef": np.eye(4), "source": ""}])
def test_explicit_reference_prior_requires_valid_pose_and_provenance(invalid):
    demo, scene = bound_example()
    scene["grasp_reference_prior"] = invalid
    with pytest.raises(ValueError):
        ground_grasp(demo, scene)
    with pytest.raises(ValueError):
        bind_grasp_candidate(scene, np.eye(4), source="candidate")


def test_complete_builder_updates_all_grasp_goals_without_changing_the_skill_clock():
    skill = grasp_input()
    demo = skill["program"]["demo"]
    demo["protected_start_frame"] = 11
    demo["timestamps"] = np.array([2., 2.02, 2.03, 2.5, 2.7, 3.])
    demo["gripper_events"] = [
        {"event_id": "open", "frame_index": 11, "command": "open"},
        {"event_id": "close-a", "frame_index": 14, "command": "close", "occurrence": 2},
        {"event_id": "hold", "frame_index": 17, "command": "hold"},
        {"event_id": "reopen", "frame_index": 20, "command": "open"},
        {"event_id": "close-b", "frame_index": 23, "command": "close"},
    ]
    start_q, start_dq = np.array([.1, -.2, .3]), np.array([.01, -.02, .03])

    def build(value):
        return build_planning_problem([value], joint_names=("a", "b", "c"),
            start_q=start_q, start_dq=start_dq, tool_link="tool", reference_frame="world",
            base_pose=np.eye(4), tool_transform=np.eye(4))

    original = build(skill)
    target = pose((.3, -.2, .4), (.2, -.5, .7))
    candidate = deepcopy(skill)
    candidate["scene"] = bind_grasp_candidate(skill["scene"], target, source="generated candidate")
    actual = build(candidate)
    expected_local = np.linalg.inv(skill["scene"]["carrier_pose"]) @ target
    assert len(actual.phases) == len(original.phases) == 1
    assert len(actual.pose_events) == len(original.pose_events) == 3
    for event in actual.pose_events:
        np.testing.assert_allclose(event.target, target, atol=1e-15, rtol=0)
        assert event.position_tolerance_m == skill["scene"]["grasp_goal"]["position_tolerance_m"]
        assert event.rotation_tolerance_rad == skill["scene"]["grasp_goal"]["rotation_tolerance_rad"]
    np.testing.assert_allclose(actual.pose_events[0].actual_relation["T_owner_target"], expected_local)
    attachment = next(row for row in actual.conditions if row["kind"] == "candidate_attachment")
    np.testing.assert_allclose(attachment["T_carrier_eef"], expected_local)
    assert attachment["effect_requirement"] == "stable_grasp_from_current_synchronized_evidence"
    np.testing.assert_allclose(actual.phases[0].metadata["candidate_attachment"], expected_local)
    assert [row for row in actual.conditions if row["kind"] == "grasp_contact_activation"] != []
    for name in ("start_q", "start_dq", "source_timestamps", "source_progress", "stop_progress"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(original, name))
    for before, after in zip(original.events, actual.events):
        assert (before.event_id, before.kind, before.source_frame, before.progress, before.occurrence) == (
            after.event_id, after.kind, after.source_frame, after.progress, after.occurrence)
    before_conditions = [row for row in original.conditions if row["kind"] == "grasp_contact_activation"]
    after_conditions = [row for row in actual.conditions if row["kind"] == "grasp_contact_activation"]
    for before, after in zip(before_conditions, after_conditions):
        for key in ("condition_id", "carrier_id", "arrival_goal_id", "activation_event", "interval"):
            assert after[key] == before[key]
    expected_path = target @ np.linalg.inv(original.pose_events[0].target) @ original.phases[0].reference.poses
    np.testing.assert_allclose(actual.phases[0].reference.poses, expected_path, atol=1e-15, rtol=0)
    assert actual.pose_relations == original.pose_relations == ()
# END no model score or nominal attachment is converted into successful Holding.
