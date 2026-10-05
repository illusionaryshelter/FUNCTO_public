"""Keep failure graphics from promoting input records into accepted evidence."""

# BEGIN CPU-only semantic checks for saved planner failure views.
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
from manipulation.collision_records import encode_array
from recorded_planning_failure_visuals import (candidate_document,
    captured_query_summary, failure_witnesses, native_failure_evidence, native_metric_rows, validation_summary)


def test_recorded_mode_inputs_are_counted_once_without_becoming_verdicts():
    time = np.array([[0., .5, 0., .5, 0., .5]], dtype=np.float32)
    position = np.repeat(time[..., None], 2, axis=-1)
    call = {"time_s": encode_array(time), "joint_position": encode_array(position),
            "query_order": ["exact", "left", "right"]}
    snapshot = {"scene_query_capture": {"status": "completed", "calls": [call]}}
    result = captured_query_summary(snapshot)
    assert result["physical_input_rows"] == 2 and result["mode_rows"] == 6
    assert result["time_range_s"] == [0., .5]
    assert result["verdicts_available"] is False
    position[0, 3, 1] += 1
    call["joint_position"] = encode_array(position)
    with pytest.raises(ValueError, match="changed physical"):
        captured_query_summary(snapshot)
    assert captured_query_summary({})["verdicts_available"] is False


def test_native_seed_identity_and_terminal_semantics_are_preserved():
    seed = {"seed_index": 0,
        "metrics": {"success": False, "rollout": {
            "constraints": {"robot_world": 0., "cspace": 1., "invalid": "NaN"},
            "convergence": {"position": 100.}}},
        "positive_constraint_names": {"rollout": ["cspace"]},
        "nonfinite_metric_names": ["rollout.constraints.invalid"]}
    other = deepcopy(seed)
    other["seed_index"] = 1
    document = {"schema": "functo.native_candidate_metrics.1", "seeds": [seed, other]}
    rows = native_metric_rows(document, 0)
    assert len(rows) == 4 and {row["seed"] for row in rows} == {0}
    status = {row["name"]: row["status"] for row in rows}
    assert status["cspace"] == "正约束项"
    assert status["invalid"] == "非有限/未知"
    assert status["position"] == "终点判定不可由此最大值推断"
    assert native_metric_rows(document, None) == []


def test_partial_checked_range_cannot_be_extended_to_whole_curve():
    document = {"schema": "functo.export_validation_progress.1", "checked_points": 2,
        "planned_points": 4, "checked_time_interval_s": [0., .5],
        "validation_complete": False, "status": "partial"}
    summary = validation_summary(document, 1.)
    assert summary["checked_time_interval_s"] == [0., .5]
    assert summary["validation_complete"] is False
    assert validation_summary(None, 1.) is None
    document["checked_time_interval_s"] = [0., 1.1]
    with pytest.raises(ValueError, match="inside this curve"):
        validation_summary(document, 1.)


def test_only_original_rejected_pair_points_become_failure_witnesses():
    evidence = {"actor_geometry": "arm_solid", "target_geometry": "observed_owner",
        "signed_distance_m": -.001, "witness_first": [.1, .2, .3],
        "witness_second": [.101, .2, .3]}
    violation = {"kind": "independent_scene_collision", "evidence": evidence}
    partial = {"source_world_collision": {"worst_state": {"trajectory_time_s": .5,
        "scene": {"violations": [violation], "pairs": [evidence], "query_gaps": []}}},
        "source_self_collision": {"worst_state": {"trajectory_time_s": .5,
            "minimum_pair": evidence, "violations": []}}}
    result = failure_witnesses(partial)
    assert len(result) == 1 and result[0]["time_s"] == .5
    assert result[0]["points_world"] == [evidence["witness_first"], evidence["witness_second"]]
    partial["source_world_collision"]["worst_state"]["scene"]["violations"] = []
    assert failure_witnesses(partial) == []


def test_explicit_document_binding_rejects_another_native_seed(tmp_path):
    (tmp_path / "planning").mkdir()
    record = {"native_result": "native-result-00.npz", "attempt": {"seed_index": 0},
              "export_validation_path": "export-00-validation.json"}
    document = {"schema": "functo.export_validation_progress.1",
                "native_result": "native-result-00.npz", "seed_index": 1}
    (tmp_path / "planning" / record["export_validation_path"]).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="exact native seed"):
        candidate_document(tmp_path, record, "export_validation_path", document["schema"])
    assert candidate_document(tmp_path, {}, "export_validation_path", document["schema"]) == (None, None)
# END none of these checks runs geometry, GPU work or physical simulation.

# BEGIN native rejection has evidence even though independent validation is absent.
def test_native_witnesses_bind_seed_and_do_not_duplicate_material_modes():
    evidence = {"actor_geometry": "arm", "target_geometry": "observed_object",
        "signed_distance_m": .00005, "witness_first": [0., 0., 0.],
        "witness_second": [.00005, 0., 0.]}
    row = {"component": "source_world_collision", "time_s": .5,
        "geometry_witnesses": [{"trajectory_time_s": .5,
            "planning_joint_names": ["j"], "planning_joint_positions": [.25],
            "violations": [{"kind": "margin_not_satisfied", "evidence": evidence}]}]}
    seed = {"seed_index": 0, "metrics": {"rollout": {"source_trace": {
        "status": "recorded", "component_failures": [row, deepcopy(row)]}}}}
    other = deepcopy(seed)
    other["seed_index"] = 1
    other["metrics"]["rollout"]["source_trace"]["component_failures"][0]["time_s"] = 99.
    components, witnesses = native_failure_evidence({"seeds": [seed, other]}, 0)
    assert components[0]["failed_mode_samples"] == 2
    assert components[0]["time_range_s"] == [.5, .5]
    assert len(witnesses) == 1
    assert witnesses[0]["scope"] == "native_rollout"
    assert witnesses[0]["joint_position"] == [.25]
    assert witnesses[0]["signed_distance_m"] == .00005
    assert witnesses[0]["evidence_origin"] == "native_source_trace_not_independent_validation"
    assert failure_witnesses(None) == []
# END positive distance inside the required margin remains a recorded failure.
