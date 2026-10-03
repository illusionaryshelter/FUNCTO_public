"""Small CPU counterexamples for exact captured collision diagnostic queries."""

# BEGIN query identity, hand uncertainty and raw witnesses remain separate.
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))
from audit_recorded_collision import (MAX_QUERIES, captured_queries, nominal_hand,
                                     query_state, unordered_witnesses)
from manipulation.execution_binding import curve_snapshot
from test_recorded_collision_snapshot import candidate


def snapshot(curve):
    lower = float(np.nextafter(np.float32(.5), np.float32(-np.inf)))
    return {"classification": "diagnostic_collision_snapshot_not_executable",
        "curve": curve_snapshot(curve), "formal_clock": {"duration": curve.duration},
        "scene_query_capture": {"status": "completed", "errors": [], "calls": [{
            "time_s": [[.5, .5, .5]], "progress": [[.5, lower, .5]],
            "robot_owner_permission_progress": [[.5, .5, .5]],
            "joint_position": [[[.125], [.125], [.125]]], "joint_names": ["arm"],
            "time_dtype": "torch.float32", "progress_dtype": "torch.float32"}]}}


def test_deduplication_keeps_different_modes_and_original_float_values():
    _, curve, _ = candidate()
    saved = snapshot(curve)
    states, mapping, total = captured_queries(saved, curve)
    assert total == 3 and len(states) == 2
    assert mapping[0]["unique_state_indices"] == [0, 1, 0]
    assert states[1]["progress"] < .5 and states[1]["permission_progress"] == .5
    assert states[0]["q"].tolist() == [.125]  # Never resample a nearby point on the cubic.
    saved["curve"]["position"][0][0] += .1
    with pytest.raises(ValueError, match="same candidate"):
        captured_queries(saved, curve)


def test_complete_query_budget_counts_duplicate_calls_before_deduplication():
    _, curve, _ = candidate()
    saved = snapshot(curve)
    call = saved["scene_query_capture"]["calls"][0]
    count = MAX_QUERIES+1
    call.update(time_s=[[.5]*count], progress=[[.5]*count],
        robot_owner_permission_progress=[[.5]*count], joint_position=[[[.125]]*count])
    with pytest.raises(ValueError, match="explicit 4608 budget"):
        captured_queries(saved, curve)


def test_timed_reference_hand_is_conditional_and_does_not_freeze_actual_fingers():
    _, curve, scene = candidate()
    scene["gripper"].update(controller="timed_reference", joint_names=["left", "right"], nominal_model=None,
                            current_positions=[.02, -.03])
    sampler, scope = nominal_hand(scene, curve)
    assert scope["conditional_command_reference"] and scope["source"] == "command_reference"
    np.testing.assert_allclose(sampler(.5, "torch.float32")[0], [.02, -.03])
    # Keep the production polynomial's float32 arithmetic, including its tiny
    # endpoint residue; replacing this with a command target would hide it.
    np.testing.assert_allclose(sampler(1., "torch.float32")[0], [0., 0.], atol=1e-8, rtol=0)
    assert sampler(1., "torch.float32")[0][1] > 0.
    explicit = deepcopy(scene)
    explicit["gripper"]["nominal_model"] = explicit["gripper"]["command_reference"]
    _, scope = nominal_hand(explicit, curve)
    assert not scope["conditional_command_reference"]
    missing = deepcopy(scene)
    missing["gripper"]["controller"] = "binary"
    with pytest.raises(ValueError, match="no explicit nominal model"):
        nominal_hand(missing, curve)


@pytest.mark.parametrize("time_s,conditional_count", [(0., 0), (.5, 1)])
def test_nominal_hand_collisions_are_not_established_wrist_background_collisions(time_s, conditional_count):
    def pair(name):
        return {"actor": "robot", "actor_geometry": name, "target_owner": "background",
            "target_geometry": "cells", "signed_distance_m": -.001, "permission": None,
            "witness_first": [0., 0., 0.], "witness_second": [0., 0., -.001],
            "normal_first_to_second": [0., 0., 1.]}
    pairs = [pair("wrist"), pair("finger")]

    def query(**kwargs):
        assert kwargs["joint_positions"] == [.125, .02, -.03]
        return {"frame_poses": {"tcp": np.eye(4).tolist()}, "pairs": [], "joint_limit_violations": {}}

    def scene_query(robot, **kwargs):
        assert kwargs["active_robot_owner_permissions"] is None
        assert kwargs["progress"] < .5 and kwargs["robot_owner_permission_progress"] == .5
        return {"pairs": deepcopy(pairs), "query_gaps": [], "unresolved_assumptions": [],
            "observed_scene_checked": True,
            "violations": [{"kind": "independent_scene_collision", "evidence": row} for row in pairs]}

    validator = SimpleNamespace(joint_names=("arm", "left", "right"), finger_joint_names=("left", "right"),
        T_reference_base=np.eye(4), T_frame_tcp=np.eye(4), pairs=(), tool_frame="tcp",
        checker=SimpleNamespace(query=query, query_scene=scene_query))
    state = {"time_s": time_s, "time_dtype": "torch.float32", "progress": .4999999701976776,
             "permission_progress": .5, "q": np.array([.125])}
    result = query_state(validator, {"contacts": []}, state, ("arm",),
                         lambda time, dtype: (np.array([.02, -.03]), np.zeros(2)), {"finger"})
    assert len(result["established_model_violations"]) == 2-conditional_count
    assert result["established_model_violations"][0]["evidence"]["actor_geometry"] == "wrist"
    assert len(result["conditional_nominal_hand_violations"]) == conditional_count
    assert result["scene"]["observed_scene_checked"] is (conditional_count == 0)
    witness = result["scene"]["pairs"][0]
    assert "witness_first" not in witness and "normal_first_to_second" not in witness
    assert witness["coal_raw_witness_points_unordered"] == [[0., 0., 0.], [0., 0., -.001]]


def test_unordered_witness_conversion_preserves_raw_distance_and_nested_evidence():
    raw = {"native_pair_evidence": {"signed_distance_m": .002, "witness_first": [0., 0., .915],
           "witness_second": [0., 0., .917], "normal_first_to_second": [0., 0., 1.]}}
    converted = unordered_witnesses(raw)
    assert converted["native_pair_evidence"]["signed_distance_m"] == .002
    assert converted["native_pair_evidence"]["coal_raw_witness_points_unordered"][0][2] == .915
    assert "witness_first" in raw["native_pair_evidence"]


def test_original_float32_endpoint_may_round_above_the_binary64_curve_duration():
    from manipulation.planning import JointTrajectory
    curve = JointTrajectory(("arm",), np.array([0., .1]), np.array([[0.], [0.]]),
        np.zeros((2, 1)), np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    saved = snapshot(curve)
    call = saved["scene_query_capture"]["calls"][0]
    endpoint = float(np.float32(.1))
    assert endpoint > curve.duration
    call.update(time_s=[[endpoint]], progress=[[1.]], robot_owner_permission_progress=[[1.]],
                joint_position=[[[0.]]])
    states, _, _ = captured_queries(saved, curve)
    assert states[0]["time_s"] == endpoint
    call["time_s"][0][0] = float(np.nextafter(np.float32(.1), np.float32(np.inf)))
    with pytest.raises(ValueError, match="original-clock"):
        captured_queries(saved, curve)
# END diagnostics cannot promote hypothetical hand geometry or change source status.
