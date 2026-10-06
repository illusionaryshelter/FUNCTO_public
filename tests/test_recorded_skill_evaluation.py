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
# BEGIN explicit role validation and pure aggregation share the live evaluator path.
from recorded_skill_evaluation import (
    NativeSkillPhysicsEvaluator, _FunctionalGoalEvaluator,
    summarize_skill_evidence, validate_diagnostic_goal_ids,
)
# END these helpers consume saved evidence without running physics.
from collect.preprocess.segmentation import DEFAULT_SETTINGS


def _pose(position, rotvec):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    pose[:3, 3] = position
    return pose


# BEGIN material-frame fixtures declare their known static transform explicitly.
def _fixture(*, mapped=True, material_frame=True, goal_progress=1., duration=.4):
    """Use a prescribed nonidentity body-from-material geometry transform."""
    owner_entry = _pose([.3, -.2, .5], [.1, -.2, .4])
    native_entry = _pose([1., 2., 3.], [-.4, .2, .1])
    owner_target = _pose([.08, -.03, .02], [.2, .1, -.1])
    requirement = {"progress": goal_progress, "tool_from_function": np.eye(4).tolist(),
        "target": (owner_entry @ owner_target).tolist(), "position_tolerance_m": .001,
        "rotation_tolerance_rad": np.deg2rad(1.), "source": "Original synthetic material-frame allowed set",
        "actual_relation": {"target_owner_id": "declared_material", "T_owner_target": owner_target.tolist()}}
    # BEGIN retain the event contract consumed by the existing result-only validator.
    curve = SimpleNamespace(duration=duration, progress_u=np.array([0., 1.]),
                            progress_s=np.array([0., 1.]), events=[])
    # END a synthetic curve declares its actual empty event list.
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
    if mappings and material_frame:
        mappings[0].update(T_body_owner=(np.linalg.inv(native_entry) @ owner_entry).tolist(),
            source="Prescribed synthetic static body/material coordinates; not inferred from an observation")
    goals = _FunctionalGoalEvaluator(model, catalog, result, mappings)
    return model, result, goals, owner_entry, native_entry, owner_target, catalog, mappings
# END a missing static transform leaves the body identity usable for contact predicates.


# BEGIN identity-only histories exercise effects without claiming material geometry.
def _history(*, mapped=True, material_frame=True, goal_progress=1., duration=.4, until=None, error=None, moving=False,
             diagnostic_goal_ids=()):
    model, result, goals, owner_entry, native_entry, owner_target, _, _ = _fixture(
        mapped=mapped, material_frame=material_frame, goal_progress=goal_progress, duration=duration)
# END synthetic samples and evidence timing remain unchanged below.
    until = duration if until is None else until
    steps = int(np.ceil(until/model.opt.timestep))
    elapsed = np.arange(steps+1)*model.opt.timestep
    offset = np.linalg.inv(native_entry) @ owner_entry
    evaluator = NativeSkillPhysicsEvaluator.__new__(NativeSkillPhysicsEvaluator)
    evaluator._model, evaluator._planning_result, evaluator._goals = model, result, goals
    # BEGIN match constructor role validation in this explicit non-physics double.
    evaluator._diagnostic_goal_ids = validate_diagnostic_goal_ids(
        result.execution_binding, result.trajectory, diagnostic_goal_ids)
    # END tests cannot bypass the same bound eligibility checks.
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


# BEGIN explicit collection completeness is independent of physical completion.
def _complete_collection():
    return {"completed": True, "first_error": None, "disabled": False}


def _completed(evaluator):
    return {"status": "curve_finished", "executed_steps": len(evaluator._samples)-1,
            "barrier_command_reached": True, "history": [], "completed_goal_ids": [],
            "normal_step_accounting": {"step_accounting_exact": True},
            "independent_evidence_sink": {"completed": True}}

# END CPU histories declare their own successful acquisition boundary.


# BEGIN a real nonidentity material transform is retained without entry-pose calibration.
def test_declared_nonidentity_material_frame_moves_with_native_body_and_preserves_original_goal(tmp_path):
    evaluator, result = _history(moving=True)
    before = deepcopy(result.execution_binding)
    execution = _completed(evaluator)
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=execution)
    goal = report["functional_goal_evidence"][0]
    assert goal["identity"] == ["pose_events", 7]
    assert goal["value"] == "T" and report["physics_success"] is True
    assert goal["measurement"]["position_error_m"] < 1e-12
    assert report["effect_evidence"][0]["value"] == "T"
    transform = np.asarray(report["evaluation_owner_bindings"][0]["T_body_owner"])
    assert not np.allclose(transform, np.eye(4))
    assert report["evaluation_owner_bindings"][0]["material_frame_status"] == "declared_static"
    assert report["evaluation_owner_bindings"][0]["source"].startswith("Prescribed synthetic static")
    owner_target = np.asarray(result.execution_binding["problem"]["pose_events"][0]["actual_relation"]["T_owner_target"])
    expected = evaluator._samples[-1][3] @ transform @ owner_target
    np.testing.assert_allclose(goal["measurement"]["actual_target_pose"], expected, atol=1e-14, rtol=0)
    with np.load(tmp_path/"physics-evidence.npz", allow_pickle=False) as saved:
        assert saved["evaluation_owner_poses_world"].shape == (201, 1, 4, 4)
    assert result.execution_binding == before
    assert execution["completed_goal_ids"] == []  # Offline T never becomes runtime evidence.
# END the archive contains only the explicitly declared body's material poses.


def test_stable_grasp_cannot_override_original_functional_accuracy_failure(tmp_path):
    evaluator, _ = _history(error=lambda time: [.002, 0., 0.])
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    assert report["stable_grasp_at_stop"] is True
    assert report["effect_evidence"][0]["value"] == "T"
    goal = report["functional_goal_evidence"][0]
    assert goal["position_tolerance_m"] == .001
    assert goal["value"] == "F"
    assert goal["measurement"]["position_error_m"] == pytest.approx(.002, abs=1e-14)
    assert report["physics_outcome"] == "failed" and report["physics_success"] is False


# BEGIN diagnostic pose metrics cannot veto valid effects or excuse real obligations.
def test_explicit_diagnostic_pose_failure_preserves_metrics_and_allows_original_effect_success(tmp_path):
    evaluator, result = _history(error=lambda time: [.002, 0., 0.],
                                 diagnostic_goal_ids=[["pose_events", 7]])
    binding_before = deepcopy(result.execution_binding)
    execution = _completed(evaluator)
    execution_before = deepcopy(execution)
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=execution)
    goal = report["functional_goal_evidence"][0]
    assert goal["value"] == "F" and goal["evaluation_role"] == "diagnostic"
    assert goal["position_tolerance_m"] == .001
    assert goal["measurement"]["position_error_m"] == pytest.approx(.002, abs=1e-14)
    assert report["effect_evidence"][0]["value"] == report["terminal_evidence"][0]["value"] == "T"
    assert report["effect_evidence"][0]["evaluation_role"] == "required"
    assert report["physics_success"] is True and report["physics_outcome"] == "passed"
    assert report["evaluation_policy"]["diagnostic_goal_ids"] == [["pose_events", 7]]
    assert report["evaluation_policy"]["required_goal_ids"] == []
    # Regrading is explicit each time: a saved role is not an automatic waiver.
    report_before = deepcopy(report)
    strict = summarize_skill_evidence(report, binding=result.execution_binding, trajectory=result.trajectory)
    assert strict["physics_success"] is False and strict["functional_goal_evidence"][0]["value"] == "F"
    assert strict["functional_goal_evidence"][0]["evaluation_role"] == "required"
    assert report == report_before and result.execution_binding == binding_before
    assert execution == execution_before  # Offline T cannot add completed goals or effects.


def _saved_components():
    """An explicit completed report whose pose alone misses its required set."""
    _, result, *_ = _fixture()
    report = {"candidate_id": result.execution_binding["candidate_id"], "evaluation_kind": "grasp",
        "functional_goal_evidence": [{"identity": ["pose_events", 7], "value": "F"}],
        "effect_evidence": [{"condition_id": "declared_stability", "value": "T"}],
        "process_evidence": [], "terminal_evidence": [{"predicate": "stable_grasp_at_final_stop", "value": "T"}],
        "evidence_coverage": {"complete": True}, "completed_current_curve": True}
    return report, result


@pytest.mark.parametrize("component", ["effect_evidence", "process_evidence", "terminal_evidence"])
@pytest.mark.parametrize("value,outcome,success", [("F", "failed", False), ("U", "unverified", None)])
def test_diagnostic_pose_cannot_waive_effect_process_or_terminal_failure_or_unknown(component, value, outcome, success):
    report, result = _saved_components()
    if component == "process_evidence":
        result.execution_binding["problem"]["interiors"] = [{"source": "Required synthetic process"}]
        report[component] = [{"identity": ["interiors", 0], "value": value}]
    else:
        report[component][0]["value"] = value
    grade = summarize_skill_evidence(report, binding=result.execution_binding,
        trajectory=result.trajectory, diagnostic_goal_ids=[["pose_events", 7]])
    assert grade["physics_outcome"] == outcome and grade["physics_success"] is success
    assert grade[component][0]["evaluation_role"] == "required"


@pytest.mark.parametrize("complete,finished,outcome", [
    (False, True, "incomplete_evidence"), (True, False, "not_evaluated"),
])
def test_diagnostic_pose_cannot_waive_missing_evidence_or_incomplete_curve(complete, finished, outcome):
    report, result = _saved_components()
    report["evidence_coverage"]["complete"], report["completed_current_curve"] = complete, finished
    grade = summarize_skill_evidence(report, binding=result.execution_binding,
        trajectory=result.trajectory, diagnostic_goal_ids=[["pose_events", 7]])
    assert grade["physics_success"] is None and grade["physics_outcome"] == outcome


def test_unselected_result_only_goal_remains_a_required_functional_goal():
    report, result = _saved_components()
    binding = result.execution_binding
    binding["goal_identities"]["pose_events"].append(9)
    binding["problem"]["pose_events"].append(deepcopy(binding["problem"]["pose_events"][0]))
    binding["device_policy"]["result_only_goal_ids"].append(["pose_events", 9])
    report["functional_goal_evidence"].append({"identity": ["pose_events", 9], "value": "F"})
    grade = summarize_skill_evidence(report, binding=binding, trajectory=result.trajectory,
                                    diagnostic_goal_ids=[["pose_events", 7]])
    assert grade["physics_success"] is False
    assert grade["evaluation_policy"]["required_goal_ids"] == [["pose_events", 9]]
    assert grade["functional_goal_evidence"][1]["evaluation_role"] == "required"


@pytest.mark.parametrize("violation", ["undeclared", "midcurve", "arrival", "contact_gate",
                                     "successor_origin", "process", "unknown", "duplicate", "completed_arrival"])
def test_diagnostic_role_rejects_online_or_unbound_obligations(violation):
    _, result, *_ = _fixture()
    binding, selected = result.execution_binding, [["pose_events", 7]]
    if violation == "undeclared":
        binding["device_policy"]["result_only_goal_ids"] = []
    elif violation == "midcurve":
        binding["problem"]["pose_events"][0]["progress"] = .5
    elif violation in ("arrival", "completed_arrival"):
        binding["timing_events"] = [{"timing": [{"check_target": "arrival", "goal_id": ["pose_events", 7]}]}]
        if violation == "completed_arrival":
            binding["completed_goal_ids"] = [["pose_events", 7]]
    elif violation == "contact_gate":
        binding["problem"]["conditions"].append({"kind": "grasp_contact_activation",
                                                "arrival_goal_id": ["pose_events", 7]})
    elif violation == "successor_origin":
        binding["timing_events"] = [{"timing": [{"check_target": "effect",
            "origin_event": {"goal_id": ["pose_events", 7]}}]}]
    elif violation == "process":
        selected = [["pose_relations", 0]]
    elif violation == "unknown":
        selected = [["pose_events", 99]]
    elif violation == "duplicate":
        selected *= 2
    with pytest.raises(ValueError):
        validate_diagnostic_goal_ids(binding, result.trajectory, selected)


@pytest.mark.parametrize("corruption", ["candidate", "missing_goal", "duplicate_goal"])
def test_offline_regrading_rejects_mismatched_or_removed_goal_evidence(corruption):
    report, result = _saved_components()
    if corruption == "candidate":
        report["candidate_id"] = "a-different-candidate"
    elif corruption == "missing_goal":
        report["functional_goal_evidence"] = []
    else:
        report["functional_goal_evidence"] *= 2
    with pytest.raises(ValueError):
        summarize_skill_evidence(report, binding=result.execution_binding,
            trajectory=result.trajectory, diagnostic_goal_ids=[["pose_events", 7]])


@pytest.mark.parametrize("component", ["effect_evidence", "process_evidence", "terminal_evidence"])
@pytest.mark.parametrize("corruption", ["missing", "duplicate"])
def test_offline_regrading_cannot_delete_or_duplicate_required_components(component, corruption):
    report, result = _saved_components()
    result.execution_binding["problem"]["interiors"] = [{"source": "Required synthetic process"}]
    report["process_evidence"] = [{"identity": ["interiors", 0], "value": "T"}]
    report[component] = [] if corruption == "missing" else report[component] * 2
    with pytest.raises(ValueError):
        summarize_skill_evidence(report, binding=result.execution_binding,
            trajectory=result.trajectory, diagnostic_goal_ids=[["pose_events", 7]])
# END explicit diagnostic roles never auto-select all result-only goals.


# BEGIN incomplete GT histories cannot inherit a successful retained prefix.
@pytest.mark.parametrize("failure", ["collection", "callback", "inexact_clock", "missing_suffix",
                                    "missing_initial", "missing_interior", "empty"])
def test_incomplete_collection_is_unknown_and_names_missing_sample_ranges(tmp_path, failure):
    evaluator, _ = _history()
    execution = _completed(evaluator)
    collection = _complete_collection()
    expected_missing = []
    if failure == "collection":
        collection.update(completed=False, disabled=True,
            first_error={"type": "ValueError", "message": "capture failed", "phase": "after_step",
                         "curve_time": .4, "sim_time": 10.4})
    elif failure == "callback":
        execution["independent_evidence_sink"]["completed"] = False
    elif failure == "inexact_clock":
        execution["normal_step_accounting"]["step_accounting_exact"] = False
    else:
        missing = ({"missing_suffix": list(range(181, 201)), "missing_initial": [0],
                    "missing_interior": [23, 24], "empty": list(range(201))})[failure]
        expected_missing = [[missing[0], missing[-1]]]
        keep = [index for index in range(201) if index not in missing]
        evaluator._samples = [evaluator._samples[index] for index in keep]
        evaluator._contacts = [evaluator._contacts[index] for index in keep]
        evaluator._offsets = [0]*(len(keep)+1)
        evaluator._goals._owner_pose_samples = [evaluator._goals._owner_pose_samples[index] for index in keep]
    report = evaluator.finalize(tmp_path, execution_report=execution, collection_report=collection)
    assert report["physics_success"] is None and report["physics_outcome"] == "incomplete_evidence"
    assert not report["completed_current_curve"]
    coverage = report["evidence_coverage"]
    assert coverage["complete"] is False and coverage["expected_sample_count"] == 201
    assert coverage["missing_normal_sample_index_ranges"] == expected_missing
    assert report["evidence_collection"]["first_error"] == collection["first_error"]
    if failure == "empty":
        assert not (tmp_path/"physics-evidence.npz").exists()
    else:
        with np.load(tmp_path/"physics-evidence.npz", allow_pickle=False) as archive:
            assert len(archive["sim_time"]) == len(evaluator._samples)
# END retained actual samples are saved without interpolating missing evidence.


def test_missing_explicit_owner_mapping_keeps_goal_and_whole_result_unknown(tmp_path):
    evaluator, _ = _history(mapped=False)
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    assert report["stable_grasp_at_stop"] is True
    assert report["functional_goal_evidence"][0]["value"] == "U"
    assert report["effect_evidence"][0]["value"] == "U"
    assert report["physics_outcome"] == "unverified" and report["physics_success"] is None


def test_early_hold_cannot_supply_unobserved_future_goal_or_stability_window(tmp_path):
    evaluator, _ = _history(until=.2)
    execution = {**_completed(evaluator), "status": "feedback_rejected", "executed_steps": 100,
        "barrier_command_reached": False, "history": [], "elapsed_sim_s": .5,
        "abort_hold": {"physical_steps": 150, "physical_stop_confirmed": True},
        "continuation": {"actual_feedback": {"sim_time": 10.5, "tool_pose": evaluator._samples[-1][2].tolist()}}}
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=execution)
    assert report["functional_goal_evidence"][0]["query_count"] == 0
    assert report["functional_goal_evidence"][0]["value"] == "U"
    assert report["effect_evidence"][0]["value"] == "U"
    assert not report["completed_current_curve"]
    assert report["physics_outcome"] == "not_evaluated" and report["physics_success"] is None


def test_first_goal_time_failure_is_not_repaired_by_a_later_perfect_pose(tmp_path):
    evaluator, _ = _history(goal_progress=.5, error=lambda time: [.002 if time == .2 else 0., 0., 0.])
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
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
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
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
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    assert report["effect_evidence"][0]["sample_elapsed_s"] == .4
    assert report["effect_evidence"][0]["value"] == "T"
    assert report["stable_grasp_at_stop"] is False
    assert report["terminal_evidence"][0]["value"] == "F"
    assert report["physics_success"] is False and report["physics_outcome"] == "failed"


def test_one_ulp_future_goal_uses_next_native_tick(tmp_path):
    evaluator, _ = _history(duration=np.nextafter(.4, np.inf), goal_progress=.5,
        error=lambda time: [.002 if time >= .202 else 0., 0., 0.])
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    goal = report["functional_goal_evidence"][0]
    assert goal["scheduled_curve_time_s"] > .2
    assert goal["device_due_time_s"] == goal["sample_elapsed_s"] == .202
    assert goal["value"] == "F"


def test_incomplete_arrival_window_retains_unknown_instead_of_premature_failure(tmp_path):
    evaluator, result = _history(until=.2, error=lambda time: [.002, 0., 0.])
    result.execution_binding["timing_events"] = [{"timing": [{"check_target": "arrival",
        "window_s": [.1, .4], "time_basis": "request_physical", "origin_time_s": 0.,
        "goal_id": ["pose_events", 7], "source": "Original explicit arrival window"}]}]
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report={**_completed(evaluator), "status": "feedback_rejected",
        "executed_steps": 100, "barrier_command_reached": False, "history": []})
    goal = report["functional_goal_evidence"][0]
    assert goal["measurement"]["value"] == "F" and goal["value"] == "U"
    assert report["physics_outcome"] == "not_evaluated"


@pytest.mark.parametrize("corruption", ["identity", "reflection"])
# BEGIN native identity and static rigid geometry are independently validated.
def test_owner_binding_rejects_native_identity_or_nonrigid_static_frame(corruption):
    model, result, _, _, _, _, catalog, mappings = _fixture()
    if corruption == "identity":
        catalog["bodies"][0]["id"] = 1
    else:
        mappings[0]["T_body_owner"][0][0] *= -1
    with pytest.raises(ValueError):
        _FunctionalGoalEvaluator(model, catalog, result, mappings)
# END the planned estimated entry pose is never used as a static geometry calibration.


def test_unimplemented_original_process_cannot_inherit_overall_grasp_success(tmp_path):
    evaluator, result = _history()
    result.execution_binding["problem"]["interiors"] = [{"source": "Original independently scoped interior requirement"}]
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    assert report["functional_goal_evidence"][0]["value"] == report["effect_evidence"][0]["value"] == "T"
    assert report["process_evidence"][0]["value"] == "U"
    assert report["physics_outcome"] == "unverified" and report["physics_success"] is None


def test_native_contact_probe_only_appends_offline_truth_and_returns_no_runtime_fact():
    import probe_contact_execution_gpu as probe
    from manipulation.planning_problem import PoseEventRequirement

    pose = np.eye(4)
    goal = PoseEventRequirement(1., pose, pose, .001, .01, "Explicit original synthetic goal")
    record = {"elapsed": .2, "curve_time": .2, "feedback": {"tool_pose": pose,
        # BEGIN explicit synthetic robot-pose estimator provenance.
        "tool_pose_source": "Synthetic FK of named measured joints",
        "tool_pose_is_independent_observation": False,
        # END no live native body/site pose is a runtime observation here.
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
    assert set(record["feedback"]) == {"tool_pose", "source", "timestamp_s", "feedback_id",
                                       "tool_pose_source", "tool_pose_is_independent_observation"}
# END these tests establish grading semantics, not physical grasp performance.


# BEGIN native support histories keep the original material frame, clock and role.
# BEGIN support and effect histories can share body identity without a material mapping.
def _support_history(*, delta=None, until=None, mapped=True, material_frame=True, binding_case=None,
                     arrival_window=None, goal_progress=.5):
    """Create explicit CPU body motion, with an independently retreating tool.

    The support basis and native body frame are deliberately nonidentity. The
    original point goal is before release; the final observation can differ
    without silently adding a terminal or maintained support requirement.
    """
    from itertools import product

    evaluator, result = _history(until=until, mapped=mapped, material_frame=material_frame)
    binding, model = result.execution_binding, evaluator._model
    owner = binding["scene"]["owners"][0]
    entry = np.asarray(owner["pose_path"]["poses"][0])
    native_entry = evaluator._samples[0][3]
    offset = np.linalg.inv(native_entry) @ entry
    attachment = _pose([.06, -.02, .08], [.2, -.1, .3])
    surface = {"origin": entry[:3, 3].tolist(), "basis": entry[:3, :2].tolist(),
        "normal": entry[:3, 2].tolist(), "boundary": [[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]],
        "source": "Original observed finite support, frozen in world"}
    support = {"progress": goal_progress, "tool_from_carrier": np.linalg.inv(attachment).tolist(),
        "body_points": list(product((-.04, .04), (-.03, .03), (0., .1))),
        "contact_points": list(product((-.01, .01), (-.01, .01), (0.,))),
        "outward_normal": [0., 0., -1.], "surface": surface,
        "gap_tolerance": .0001, "normal_tolerance": .0001, "margin": .002,
        "source": "Original synthetic bottom-on-finite-support event"}
    held = {"metadata": {"mode": "held", "carrier_id": "declared_material",
        "owner_initial_pose": entry.tolist(), "attachment": attachment.tolist(),
        "attachment_condition_id": "declared_holding", "attachment_interval": [0., 1.],
        "release_condition_id": "declared_release"}}
    holding = {"kind": "observed_attachment", "condition_id": "declared_holding",
        "carrier_id": "declared_material", "progress": 0., "T_carrier_eef": attachment.tolist()}
    release = {"kind": "predicted_release", "condition_id": "declared_release",
        "skill_id": "declared_skill", "event_id": "release_event", "carrier_id": "declared_material",
        "progress": 1., "effect_requirement": "actual_release_from_current_feedback"}
    problem = binding["problem"]
    problem.update(pose_events=[], supports=[support],
        numerical={"position_m": .0001, "rotation_rad": .001, "joint": 1e-5, "velocity": 1e-4},
        phases=[held, {"metadata": {"mode": "released_retreat", "carrier_id": "declared_material"}}],
        conditions=[holding, release])
    binding["goal_identities"] = {"pose_events": [], "supports": [11]}
    binding["device_policy"]["result_only_goal_ids"] = []
    result.trajectory.progress_s = np.array([0., 2.])
    owner["pose_interval"] = [0., 2.]
    if arrival_window is not None:
        binding["timing_events"] = [{"timing": [{"check_target": "arrival",
            "goal_id": ["supports", 11], "window_s": list(arrival_window),
            "time_basis": "request_physical", "origin_time_s": 0.,
            "source": "Explicit original support arrival opportunity"}]}]
    if binding_case == "missing_phase":
        problem["phases"] = []
    elif binding_case == "missing_condition":
        problem["conditions"] = [release]
    elif binding_case == "wrong_transform":
        held["metadata"]["attachment"][0][3] += .01
    elif binding_case in ("duplicate_boundary", "ambiguous_boundary"):
        support["progress"] = 1.
        held["metadata"]["attachment_interval"] = [0., 2.]
        problem["phases"] = [held, deepcopy(held)]
        if binding_case == "ambiguous_boundary":
            problem["phases"][1]["metadata"]["attachment_condition_id"] = "other_holding"
            problem["conditions"].append({**deepcopy(holding), "condition_id": "other_holding"})
    mappings = [{"owner_id": "declared_material", "native_body_name": "native_subject"}] if mapped else []
    if mappings and material_frame:
        mappings[0].update(T_body_owner=offset.tolist(),
            source="Prescribed synthetic static body/material coordinates; not inferred from an observation")
    goals = _FunctionalGoalEvaluator(model, {"bodies": [{"name": "native_subject", "id": 0}]}, result, mappings)
    evaluator._goals, evaluator._skill_kind = goals, "manipulation"
    evaluator._task_contract = {"skill_kind": "manipulation", "task": {"relation": "on", "motion": "free"}}
    evaluator._diagnostic_goal_ids = []
    for index, row in enumerate(evaluator._samples):
        time, elapsed = row[:2]
        change = np.eye(4) if delta is None else delta(float(elapsed))
        carrier = entry @ change
        native = carrier @ np.linalg.inv(offset)
        tool = carrier @ attachment
        if elapsed >= .2:
            tool = _pose([.4, -.3, .5], [.4, .2, -.1]) @ tool
        goals.observe(SimpleNamespace(xmat=native[:3, :3].reshape(1, 9), xpos=native[:3, 3].reshape(1, 3)))
        evaluator._samples[index] = (time, elapsed, tool, native, row[4],
            np.array([elapsed < .1, elapsed < .1]), None, row[7], row[8])
    return evaluator, result
# END all material coordinates are explicit independent fixture geometry.


def test_support_native_material_pose_preserves_original_goal_and_does_not_follow_retreat(tmp_path):
    evaluator, result = _support_history()
    before = deepcopy(result.execution_binding)
    execution = _completed(evaluator)
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=execution)
    goal = report["functional_goal_evidence"][0]
    assert goal["identity"] == ["supports", 11] and goal["value"] == "T"
    assert goal["sample_elapsed_s"] == .1 and goal["query_count"] == 1
    assert goal["gap_tolerance_m"] == goal["normal_tolerance_rad"] == .0001
    assert "position_tolerance_m" not in goal
    assert report["physics_success"] is True and report["effect_evidence"][0]["value"] == "T"
    diagnostic = report["final_support_diagnostics"][0]
    assert diagnostic["measurement"]["value"] == "T" and diagnostic["evaluation_role"] == "diagnostic"
    assert report["terminal_evidence"] == []
    np.testing.assert_allclose(diagnostic["measurement"]["actual_carrier_pose"],
        goal["measurement"]["actual_carrier_pose"], atol=1e-14, rtol=0)
    hypothetical_from_tool = evaluator._samples[-1][2] @ np.asarray(before["problem"]["supports"][0]["tool_from_carrier"])
    assert not np.allclose(diagnostic["measurement"]["actual_carrier_pose"], hypothetical_from_tool)
    assert report["support_runtime_coverage"][0]["value"] == "U"
    assert result.execution_binding == before and execution["completed_goal_ids"] == []


@pytest.mark.parametrize("kind", ["gap", "normal", "whole_body_boundary", "wrong_contact_face",
                                  "body_penetration", "positive_face_gap", "numerical_body_slack"])
def test_support_metrics_match_original_compiled_allowed_set_without_adding_a_full_pose(kind):
    import torch

    evaluator, _ = _support_history()
    identity = ("supports", 11)
    requirement = evaluator._goals._requirements[identity]
    entry = evaluator._goals._owner_pose_samples[0]["declared_material"]
    delta = np.eye(4)
    if kind == "gap":
        delta[2, 3] = .0002
    elif kind == "normal":
        delta[:3, :3] = Rotation.from_rotvec([.0002, 0., 0.]).as_matrix()
    elif kind == "whole_body_boundary":
        delta[0, 3] = .18
    elif kind == "wrong_contact_face":
        requirement.contact_points[:, 2] = .1
    elif kind in ("body_penetration", "numerical_body_slack"):
        requirement.body_points[0, 2] = -.0002 if kind == "body_penetration" else -.00005
    else:
        delta[2, 3] = .00005
        # Tangent translations and yaw remain free inside the finite patch.
        delta[0, 3] = .04
        delta[:3, :3] = Rotation.from_rotvec([0., 0., .7]).as_matrix()
    actual = entry @ delta
    evaluator._goals._owner_pose_samples[0]["declared_material"] = actual
    measured = evaluator._goals._measure_support(identity, 0)
    tool = actual @ np.linalg.inv(requirement.tool_from_carrier)
    position = torch.tensor(tool[:3, 3], dtype=torch.float64)
    quaternion = torch.tensor(Rotation.from_matrix(tool[:3, :3]).as_quat()[[3, 0, 1, 2]], dtype=torch.float64)
    residual = requirement.evaluate(position, quaternion, evaluator._goals._support_numerical).item()
    expected = kind in ("positive_face_gap", "numerical_body_slack")
    assert (measured["value"] == "T") == expected
    assert (residual == 0.) == expected
    assert sum(measured["violations"].values()) == pytest.approx(residual, abs=1e-12)
    if kind == "wrong_contact_face":
        assert measured["body"]["support_gap_m"] < 1e-12
        assert measured["contact_face"]["max_abs_gap_m"] > .09
    if kind == "whole_body_boundary":
        assert measured["contact_face"]["boundary_violation_m"] == 0.
        assert measured["violations"]["body_boundary_m"] > 0.


@pytest.mark.parametrize("case", ["missing_mapping", "missing_phase", "missing_condition",
                                  "wrong_transform", "ambiguous_boundary"])
def test_support_missing_or_ambiguous_material_binding_remains_unknown(tmp_path, case):
    evaluator, _ = _support_history(mapped=case != "missing_mapping", binding_case=case)
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    goal = report["functional_goal_evidence"][0]
    assert goal["value"] == "U" and goal["measurement"]["reason"]
    assert report["physics_success"] is None and report["physics_outcome"] == "unverified"
    assert report["final_support_diagnostics"][0]["measurement"]["value"] == "U"


def test_support_duplicate_same_attachment_at_phase_boundary_is_one_fact(tmp_path):
    evaluator, _ = _support_history(binding_case="duplicate_boundary")
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    goal = report["functional_goal_evidence"][0]
    assert goal["value"] == "T" and goal["sample_elapsed_s"] == .2
    assert goal["measurement"]["carrier_binding"]["attachment_condition_id"] == "declared_holding"


def test_support_original_point_failure_is_not_repaired_by_final_success(tmp_path):
    evaluator, _ = _support_history(delta=lambda time:
        _pose([0., 0., .001 if time == .1 else 0.], [0., 0., 0.]))
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    goal = report["functional_goal_evidence"][0]
    assert goal["value"] == "F" and goal["sample_elapsed_s"] == .1 and goal["query_count"] == 1
    assert report["effect_evidence"][0]["value"] == "T"
    assert report["final_support_diagnostics"][0]["measurement"]["value"] == "T"
    assert report["physics_success"] is False


def test_support_final_diagnostic_failure_does_not_invent_an_original_terminal_obligation(tmp_path):
    evaluator, _ = _support_history(delta=lambda time:
        _pose([0., 0., .002 if time >= .3 else 0.], [0., 0., 0.]))
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    assert report["functional_goal_evidence"][0]["value"] == "T"
    assert report["final_support_diagnostics"][0]["measurement"]["value"] == "F"
    assert report["physics_success"] is True and report["terminal_evidence"] == []


@pytest.mark.parametrize("window,passing_time,expected", [((.1, .16), .14, "T"), ((.1, .12), .14, "F")])
def test_support_declared_arrival_window_preserves_actual_opportunity(tmp_path, window, passing_time, expected):
    evaluator, _ = _support_history(arrival_window=window, delta=lambda time:
        _pose([0., 0., 0. if time == 0. or time == passing_time else .001], [0., 0., 0.]))
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    goal = report["functional_goal_evidence"][0]
    assert goal["value"] == expected and goal["arrival_window_s"] == list(window)
    if expected == "T":
        assert goal["sample_elapsed_s"] == passing_time


def test_support_unobserved_original_time_remains_unknown_and_final_row_names_partial_curve(tmp_path):
    evaluator, _ = _support_history(until=.05)
    execution = {**_completed(evaluator), "status": "feedback_rejected", "barrier_command_reached": False}
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=execution)
    assert report["functional_goal_evidence"][0]["value"] == "U"
    assert report["functional_goal_evidence"][0]["query_count"] == 0
    assert not report["final_support_diagnostics"][0]["completed_current_curve"]
    assert report["physics_success"] is None and report["physics_outcome"] == "not_evaluated"


def test_support_online_coverage_uses_only_execution_evidence_and_never_the_native_verdict(tmp_path):
    evaluator, _ = _support_history()
    execution = _completed(evaluator)
    execution["completed_goal_ids"] = [["supports", 11]]
    assert evaluator._goals.support_runtime_coverage(execution)[0]["value"] == "U"
    execution["goal_evidence"] = [{"identity": ["supports", 11], "value": "F", "completion_verified": False}]
    before = deepcopy(execution)
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=execution)
    assert report["functional_goal_evidence"][0]["value"] == "T"
    assert report["support_runtime_coverage"][0]["value"] == "F" and execution == before
# END independent support evidence is neither a runtime fact nor an implicit command prerequisite.


# BEGIN static material mappings preserve perception error and separate body-only effects.
def test_identity_only_mapping_keeps_grasp_effect_but_material_pose_goal_unknown(tmp_path):
    evaluator, _ = _history(material_frame=False)
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    assert report["stable_grasp_at_stop"] is True
    assert report["effect_evidence"][0]["value"] == report["terminal_evidence"][0]["value"] == "T"
    assert report["functional_goal_evidence"][0]["value"] == "U"
    assert report["physics_success"] is None and report["physics_outcome"] == "unverified"
    binding, = report["evaluation_owner_bindings"]
    assert binding["T_body_owner"] is None and binding["source"] is None
    assert binding["material_frame_status"] == "unmapped"
    with np.load(tmp_path/"physics-evidence.npz", allow_pickle=False) as archive:
        assert archive["evaluation_owner_ids"].size == 0
        assert archive["evaluation_owner_poses_world"].shape == (201, 0, 4, 4)


def test_robot_only_pose_goal_does_not_require_an_object_material_mapping(tmp_path):
    evaluator, result = _history(material_frame=False)
    # This explicit robot/TCP-to-world requirement has no object-owned function
    # or target frame. Body identity still names the independent grasp effect.
    result.execution_binding["problem"]["pose_events"][0]["actual_relation"] = None
    before = deepcopy(result.execution_binding)
    goals = _FunctionalGoalEvaluator(evaluator._model,
        {"bodies": [{"name": "native_subject", "id": 0}]}, result,
        [{"owner_id": "declared_material", "native_body_name": "native_subject"}])
    evaluator._goals = goals
    for row in evaluator._samples:
        native = row[3]
        goals.observe(SimpleNamespace(xmat=native[:3, :3].reshape(1, 9), xpos=native[:3, 3].reshape(1, 3)))
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    assert report["functional_goal_evidence"][0]["value"] == "T"
    assert report["effect_evidence"][0]["value"] == "T" and report["physics_success"] is True
    assert result.execution_binding == before


def test_identity_only_mapping_keeps_release_effect_but_support_unknown(tmp_path):
    evaluator, _ = _support_history(material_frame=False)
    report = evaluator.finalize(tmp_path, collection_report=_complete_collection(), execution_report=_completed(evaluator))
    assert report["effect_evidence"][0]["value"] == "T"
    assert report["functional_goal_evidence"][0]["value"] == "U"
    final, = report["final_support_diagnostics"]
    assert final["measurement"]["value"] == "U" and "static" in final["measurement"]["reason"]
    assert report["physics_success"] is None


@pytest.mark.parametrize("corruption", ["missing_source", "missing_transform", "empty_source",
                                       "nontext_source", "nonfinite_transform", "extra_field"])
def test_static_material_mapping_requires_complete_rigid_geometry_and_provenance(corruption):
    model, result, _, _, _, _, catalog, mappings = _fixture()
    row = mappings[0]
    if corruption == "missing_source":
        row.pop("source")
    elif corruption == "missing_transform":
        row.pop("T_body_owner")
    elif corruption == "empty_source":
        row["source"] = "  "
    elif corruption == "nontext_source":
        row["source"] = {"description": "No implicit source conversion"}
    elif corruption == "nonfinite_transform":
        row["T_body_owner"][0][3] = np.nan
    else:
        row["align_at_entry"] = True
    with pytest.raises(ValueError):
        _FunctionalGoalEvaluator(model, catalog, result, mappings)


@pytest.mark.parametrize("error_kind", ["height", "tilt"])
def test_initial_observed_pose_error_cannot_be_fitted_out_of_static_support_geometry(error_kind):
    evaluator, result = _support_history()
    owner = result.execution_binding["scene"]["owners"][0]
    actual_entry = np.asarray(owner["pose_path"]["poses"][0])
    error = (_pose([0., 0., .01], [0., 0., 0.]) if error_kind == "height"
             else _pose([0., 0., 0.], [.01, 0., 0.]))
    estimated_entry = actual_entry @ error
    owner["pose_path"]["poses"] = [estimated_entry.tolist(), estimated_entry.tolist()]
    before = deepcopy(result.execution_binding)
    declared = [{"owner_id": "declared_material", "native_body_name": "native_subject",
        "T_body_owner": np.eye(4).tolist(),
        "source": "This synthetic object's fixed native body coordinates equal its complete material geometry"}]
    goals = _FunctionalGoalEvaluator(evaluator._model,
        {"bodies": [{"name": "native_subject", "id": 0}]}, result, declared)
    # A rigidly held object moved to compensate the erroneous estimated entry
    # reaches this real pose. Dynamic entry alignment would multiply by error
    # again and report the original perfect support pose, erasing the defect.
    actual_goal = actual_entry @ np.linalg.inv(error)
    for native in (actual_entry, actual_goal):
        goals.observe(SimpleNamespace(xmat=native[:3, :3].reshape(1, 9), xpos=native[:3, 3].reshape(1, 3)))
    measured = goals._measure_support(("supports", 11), 1)
    assert measured["value"] == "F"
    np.testing.assert_allclose(measured["actual_carrier_pose"], actual_goal, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(goals.describe_bindings()[0]["T_body_owner"], np.eye(4))
    if error_kind == "height":
        assert measured["contact_face"]["max_abs_gap_m"] == pytest.approx(.01, abs=1e-14)
        assert measured["body"]["minimum_signed_height_m"] == pytest.approx(-.01, abs=1e-14)
    else:
        assert measured["violations"]["normal_chord"] > 0.
    np.testing.assert_allclose(actual_goal @ error, actual_entry, atol=1e-14, rtol=0)
    assert result.execution_binding == before and declared[0]["T_body_owner"] == np.eye(4).tolist()
# END all transforms and samples are explicit CPU math; no native physics or GPU work.
