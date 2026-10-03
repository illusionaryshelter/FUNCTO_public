"""CPU math/history tests; native truth samples are explicit doubles, never physics."""

# BEGIN offline truth is graded in original frames and cannot become runtime evidence.
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))
from recorded_skill_evaluation import NativeSkillPhysicsEvaluator, _FunctionalGoalEvaluator
from collect.preprocess.segmentation import DEFAULT_SETTINGS


def _pose(position, rotvec):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    pose[:3, 3] = position
    return pose


def _fixture(*, mapped=True, goal_progress=1., duration=.4):
    """Use deliberately different native-body and observed-owner material frames."""
    owner_entry = _pose([.3, -.2, .5], [.1, -.2, .4])
    native_entry = _pose([1., 2., 3.], [-.4, .2, .1])
    owner_target = _pose([.08, -.03, .02], [.2, .1, -.1])
    requirement = {"progress": goal_progress, "tool_from_function": np.eye(4).tolist(),
        "target": (owner_entry @ owner_target).tolist(), "position_tolerance_m": .001,
        "rotation_tolerance_rad": np.deg2rad(1.), "source": "Original synthetic material-frame allowed set",
        "actual_relation": {"target_owner_id": "declared_material", "T_owner_target": owner_target.tolist()}}
    curve = SimpleNamespace(duration=duration, progress_u=np.array([0., 1.]), progress_s=np.array([0., 1.]))
    path = {"progress": [0., 1.], "poses": [owner_entry.tolist(), owner_entry.tolist()],
        "source": "Explicit observed entry owner frame", "motion_model": None,
        "coordinates": None, "attachment": np.eye(4).tolist()}
    binding = {"candidate_id": "synthetic-formal-identity", "problem": {
        "reference_frame": "world", "start_progress": 0., "pose_events": [requirement],
        "conditions": [{"condition_id": "declared_stability", "skill_id": "declared_skill",
            "event_id": "declared_effect", "carrier_id": "declared_material", "progress": 1.,
            "effect_requirement": "stable_grasp_from_current_synchronized_evidence"}]},
        "scene": {"owners": [{"owner_id": "declared_material", "pose_interval": [0., 1.], "pose_path": path}]},
        "goal_identities": {"pose_events": [7]}, "history": [], "timing_events": [],
        "device_policy": {"result_only_goal_ids": [["pose_events", 7]]}}
    result = SimpleNamespace(success=True, trajectory=curve, execution_binding=binding)
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.002), body=lambda name: SimpleNamespace(id=0))
    catalog = {"bodies": [{"name": "native_subject", "id": 0}]}
    mappings = [{"owner_id": "declared_material", "native_body_name": "native_subject"}] if mapped else []
    goals = _FunctionalGoalEvaluator(model, catalog, result, mappings)
    return model, result, goals, owner_entry, native_entry, owner_target, catalog, mappings


def _history(*, mapped=True, goal_progress=1., duration=.4, until=None, error=None, moving=False):
    model, result, goals, owner_entry, native_entry, owner_target, _, _ = _fixture(
        mapped=mapped, goal_progress=goal_progress, duration=duration)
    until = duration if until is None else until
    steps = int(np.ceil(until/model.opt.timestep))
    elapsed = np.arange(steps+1)*model.opt.timestep
    offset = np.linalg.inv(native_entry) @ owner_entry
    evaluator = NativeSkillPhysicsEvaluator.__new__(NativeSkillPhysicsEvaluator)
    evaluator._model, evaluator._planning_result, evaluator._goals = model, result, goals
    evaluator._target, evaluator._reference, evaluator._target_joints = 0, None, []
    evaluator._skill_kind = "grasp"
    evaluator._task_contract = {"skill_kind": "grasp", "scope": "Explicit synthetic history"}
    evaluator.binding = {"scope": "Synthetic evaluator-only identity"}
    evaluator.settings = {key: DEFAULT_SETTINGS[key] for key in (
        "contact_distance_m", "contact_gap_s", "stable_grasp_s", "stable_relative_position_m",
        "stable_relative_rotation_rad", "stable_finger_slide_m", "stable_finger_hinge_rad", "release_absence_s")}
    evaluator._finger_limits = np.array([.001, .001])
    evaluator._samples, evaluator._contacts, evaluator._offsets = [], [], [0]
    for time in elapsed:
        body = (_pose([time*.1, -.03*time, 0.], [0., 0., .2*time]) @ native_entry
                if moving else native_entry.copy())
        tool = body @ offset @ owner_target
        if error is not None:
            tool[:3, 3] += np.asarray(error(float(time)))
        data = SimpleNamespace(xmat=body[:3, :3].reshape(1, 9), xpos=body[:3, 3].reshape(1, 3))
        goals.observe(data)
        evaluator._samples.append((10.+time, time, tool, body, np.array([.01, -.01]),
                                   np.array([True, True]), None, np.empty(0), np.empty(0)))
        evaluator._contacts.append(np.empty((0, 24)))
        evaluator._offsets.append(0)
    return evaluator, result


def _completed(evaluator):
    return {"status": "curve_finished", "executed_steps": len(evaluator._samples)-1,
            "barrier_command_reached": True, "history": [], "completed_goal_ids": []}


def test_nonidentity_observed_frame_moves_with_native_body_and_preserves_original_goal(tmp_path):
    evaluator, result = _history(moving=True)
    before = deepcopy(result.execution_binding)
    execution = _completed(evaluator)
    report = evaluator.finalize(tmp_path, execution_report=execution)
    goal = report["functional_goal_evidence"][0]
    assert goal["identity"] == ["pose_events", 7]
    assert goal["value"] == "T" and report["physics_success"] is True
    assert goal["measurement"]["position_error_m"] < 1e-12
    assert report["effect_evidence"][0]["value"] == "T"
    transform = np.asarray(report["evaluation_owner_bindings"][0]["T_body_owner"])
    assert not np.allclose(transform, np.eye(4))
    owner_target = np.asarray(result.execution_binding["problem"]["pose_events"][0]["actual_relation"]["T_owner_target"])
    expected = evaluator._samples[-1][3] @ transform @ owner_target
    np.testing.assert_allclose(goal["measurement"]["actual_target_pose"], expected, atol=1e-14, rtol=0)
    with np.load(tmp_path/"physics-evidence.npz", allow_pickle=False) as saved:
        assert saved["evaluation_owner_poses_world"].shape == (201, 1, 4, 4)
    assert result.execution_binding == before
    assert execution["completed_goal_ids"] == []  # Offline T never becomes runtime evidence.


def test_stable_grasp_cannot_override_original_functional_accuracy_failure(tmp_path):
    evaluator, _ = _history(error=lambda time: [.002, 0., 0.])
    report = evaluator.finalize(tmp_path, execution_report=_completed(evaluator))
    assert report["stable_grasp_at_stop"] is True
    assert report["effect_evidence"][0]["value"] == "T"
    goal = report["functional_goal_evidence"][0]
    assert goal["position_tolerance_m"] == .001
    assert goal["value"] == "F"
    assert goal["measurement"]["position_error_m"] == pytest.approx(.002, abs=1e-14)
    assert report["physics_outcome"] == "failed" and report["physics_success"] is False


def test_missing_explicit_owner_mapping_keeps_goal_and_whole_result_unknown(tmp_path):
    evaluator, _ = _history(mapped=False)
    report = evaluator.finalize(tmp_path, execution_report=_completed(evaluator))
    assert report["stable_grasp_at_stop"] is True
    assert report["functional_goal_evidence"][0]["value"] == "U"
    assert report["effect_evidence"][0]["value"] == "U"
    assert report["physics_outcome"] == "unverified" and report["physics_success"] is None


def test_early_hold_cannot_supply_unobserved_future_goal_or_stability_window(tmp_path):
    evaluator, _ = _history(until=.2)
    execution = {"status": "feedback_rejected", "executed_steps": 100,
        "barrier_command_reached": False, "history": [], "elapsed_sim_s": .5,
        "abort_hold": {"physical_steps": 150, "physical_stop_confirmed": True},
        "continuation": {"actual_feedback": {"sim_time": 10.5, "tool_pose": evaluator._samples[-1][2].tolist()}}}
    report = evaluator.finalize(tmp_path, execution_report=execution)
    assert report["functional_goal_evidence"][0]["query_count"] == 0
    assert report["functional_goal_evidence"][0]["value"] == "U"
    assert report["effect_evidence"][0]["value"] == "U"
    assert not report["completed_current_curve"]
    assert report["physics_outcome"] == "not_evaluated" and report["physics_success"] is None


def test_first_goal_time_failure_is_not_repaired_by_a_later_perfect_pose(tmp_path):
    evaluator, _ = _history(goal_progress=.5, error=lambda time: [.002 if time == .2 else 0., 0., 0.])
    report = evaluator.finalize(tmp_path, execution_report=_completed(evaluator))
    goal = report["functional_goal_evidence"][0]
    assert goal["sample_elapsed_s"] == .2 and goal["query_count"] == 1
    assert goal["value"] == "F"
    assert goal["measurement"]["position_error_m"] == pytest.approx(.002, abs=1e-14)
    assert report["physics_success"] is False


def test_later_stable_contact_cannot_repair_the_original_effect_checkpoint(tmp_path):
    evaluator, result = _history(duration=.8)
    # Progress first reaches its terminal anchor at .4; the original clock
    # still contains a plateau through .8, not a new effect-arrival window.
    result.trajectory.progress_u = np.array([0., .5, 1.])
    result.trajectory.progress_s = np.array([0., 1., 1.])
    for index, row in enumerate(evaluator._samples):
        replacement = list(row)
        replacement[5] = np.array([row[1] >= .3, row[1] >= .3])
        evaluator._samples[index] = tuple(replacement)
    report = evaluator.finalize(tmp_path, execution_report=_completed(evaluator))
    assert report["functional_goal_evidence"][0]["sample_elapsed_s"] == .4
    assert report["functional_goal_evidence"][0]["value"] == "T"
    assert report["effect_evidence"][0]["sample_elapsed_s"] == .4
    assert report["effect_evidence"][0]["measurement"]["stable_grasp_at_stop"] is False
    assert report["effect_evidence"][0]["value"] == "F"
    assert report["stable_grasp_at_stop"] is True
    assert report["physics_outcome"] == "failed" and report["physics_success"] is False


def test_grasp_lost_after_original_effect_success_fails_the_final_stop_predicate(tmp_path):
    evaluator, result = _history(duration=.8)
    result.trajectory.progress_u = np.array([0., .5, 1.])
    result.trajectory.progress_s = np.array([0., 1., 1.])
    for index, row in enumerate(evaluator._samples):
        replacement = list(row)
        replacement[5] = np.array([row[1] < .7, row[1] < .7])
        evaluator._samples[index] = tuple(replacement)
    report = evaluator.finalize(tmp_path, execution_report=_completed(evaluator))
    assert report["effect_evidence"][0]["sample_elapsed_s"] == .4
    assert report["effect_evidence"][0]["value"] == "T"
    assert report["stable_grasp_at_stop"] is False
    assert report["terminal_evidence"][0]["value"] == "F"
    assert report["physics_success"] is False and report["physics_outcome"] == "failed"


def test_one_ulp_future_goal_uses_next_native_tick(tmp_path):
    evaluator, _ = _history(duration=np.nextafter(.4, np.inf), goal_progress=.5,
        error=lambda time: [.002 if time >= .202 else 0., 0., 0.])
    report = evaluator.finalize(tmp_path, execution_report=_completed(evaluator))
    goal = report["functional_goal_evidence"][0]
    assert goal["scheduled_curve_time_s"] > .2
    assert goal["device_due_time_s"] == goal["sample_elapsed_s"] == .202
    assert goal["value"] == "F"


def test_incomplete_arrival_window_retains_unknown_instead_of_premature_failure(tmp_path):
    evaluator, result = _history(until=.2, error=lambda time: [.002, 0., 0.])
    result.execution_binding["timing_events"] = [{"timing": [{"check_target": "arrival",
        "window_s": [.1, .4], "time_basis": "request_physical", "origin_time_s": 0.,
        "goal_id": ["pose_events", 7], "source": "Original explicit arrival window"}]}]
    report = evaluator.finalize(tmp_path, execution_report={"status": "feedback_rejected",
        "executed_steps": 100, "barrier_command_reached": False, "history": []})
    goal = report["functional_goal_evidence"][0]
    assert goal["measurement"]["value"] == "F" and goal["value"] == "U"
    assert report["physics_outcome"] == "not_evaluated"


@pytest.mark.parametrize("corruption", ["identity", "reflection"])
def test_owner_binding_rejects_native_identity_or_nonrigid_source_frame(corruption):
    model, result, _, _, _, _, catalog, mappings = _fixture()
    if corruption == "identity":
        catalog["bodies"][0]["id"] = 1
    else:
        result.execution_binding["scene"]["owners"][0]["pose_path"]["poses"][0][0][0] *= -1
    with pytest.raises(ValueError):
        _FunctionalGoalEvaluator(model, catalog, result, mappings)


def test_unimplemented_original_process_cannot_inherit_overall_grasp_success(tmp_path):
    evaluator, result = _history()
    result.execution_binding["problem"]["interiors"] = [{"source": "Original independently scoped interior requirement"}]
    report = evaluator.finalize(tmp_path, execution_report=_completed(evaluator))
    assert report["functional_goal_evidence"][0]["value"] == report["effect_evidence"][0]["value"] == "T"
    assert report["process_evidence"][0]["value"] == "U"
    assert report["physics_outcome"] == "unverified" and report["physics_success"] is None


def test_native_contact_probe_only_appends_offline_truth_and_returns_no_runtime_fact():
    import probe_contact_execution_gpu as probe
    from manipulation.planning_problem import PoseEventRequirement

    pose = np.eye(4)
    goal = PoseEventRequirement(1., pose, pose, .001, .01, "Explicit original synthetic goal")
    record = {"elapsed": .2, "curve_time": .2, "feedback": {"tool_pose": pose,
        "source": "Synthetic named proprioceptive TCP", "timestamp_s": 1.2, "feedback_id": "synthetic-1.2"}}
    data = SimpleNamespace(time=1.2, ncon=1, contact=[SimpleNamespace(
        geom1=2, geom2=4, exclude=0, efc_address=0)])
    rows = []
    with patch.object(probe.mujoco, "mj_contactForce", side_effect=lambda model, data, index, force:
                      force.__setitem__(0, 2.5)):
        returned = probe._record_contact_truth(object(), data, record, pad_id=2, slab_id=4,
                                                goal=goal, contact_log=rows)
    assert returned is None
    assert rows[0]["value"] == "T" and rows[0]["normal_force_n"] == 2.5
    assert "effects" not in record and "completed_goal_ids" not in record
    assert set(record["feedback"]) == {"tool_pose", "source", "timestamp_s", "feedback_id"}
# END these tests establish grading semantics, not physical grasp performance.
