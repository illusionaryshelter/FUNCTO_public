"""Failure evidence must retain time/mode identity without changing acceptance."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from manipulation.planning import _scene_boundary_evaluate
from manipulation.planning_diagnostics import bind_trace_to_native_seed, scene_component_failures


def test_component_capture_keeps_modes_nonfinite_and_hard_unknown_only():
    time = torch.tensor([[2., 4., 2., 4., 2., 4.]])
    phase = time / 4
    q = torch.zeros((1, 6, 2))
    p, r = torch.zeros((1, 6, 3)), torch.zeros((1, 6, 4))
    terms = {"collision": torch.tensor([[0., 0., 1., 0., 0., float('nan')]]),
             "unknown/space/robot": torch.ones_like(time),
             "unknown/release_anchor/owner": torch.tensor([[0., 1., 0., 0., 0., 0.]])}
    rows = scene_component_failures(terms, time, phase, phase, q, p, r,
                                    modes=("exact", "left", "right"))
    assert [(row["component"], row["mode"], row["ordered_query_index"]) for row in rows] == [
        ("collision", "left", 0), ("collision", "right", 1),
        ("unknown/release_anchor/owner", "exact", 1)]
    assert rows[0]["time_s"] == 2. and np.isnan(rows[1]["raw_residual"])


def test_ranked_native_seed_is_matched_by_typed_position_and_velocity():
    q = np.zeros((2, 3, 2), np.float32)
    dq = q.copy()
    dq[1] = 1
    trace = {"input_position": q, "input_velocity": dq,
        "component_failures": [{"batch_index": 0, "component": "first"},
                               {"batch_index": 1, "component": "second"}], "world_geometry": {}}
    state = SimpleNamespace(joint_names=["a", "b"],
        position=torch.from_numpy(q[::-1].copy())[None],
        velocity=torch.from_numpy(dq[::-1].copy())[None])
    result = bind_trace_to_native_seed(trace, state, 0, ["a", "b"])
    assert result["source_batch_index"] == 1
    assert result["component_failures"] == [{"batch_index": 1, "component": "second"}]
    state.velocity[0, 0, 0, 0] = 2
    assert bind_trace_to_native_seed(trace, state, 0, ["a", "b"])["status"] == "unmatched"


# BEGIN current boundary diagnostics summarize the same compact online query.
def test_boundary_summary_retains_left_failure_at_original_time_without_requery():
    calls = []
    def evaluate(spheres, p, r, s, q, names, **kwargs):
        calls.append(s.detach().clone())
        value = (s < .5).to(s.dtype)
        if "components_out" in kwargs:
            kwargs["components_out"]["required_rule"] = value
        return value

    scene = SimpleNamespace(required_progress_anchors=[.5], contacts=(), evaluate=evaluate)
    problem = SimpleNamespace(start_progress=0., end_progress=1.)
    mapping = SimpleNamespace(anchor_parameters=lambda _: np.array([[.5, .5]]))
    s = torch.tensor([[.5]])
    def run(diagnostics):
        return _scene_boundary_evaluate(scene, problem, mapping,
            torch.zeros((1, 1, 1, 4)), torch.zeros((1, 1, 3)), torch.zeros((1, 1, 4)),
            s, torch.zeros((1, 1, 2)), ("a", "b"), trajectory_time=s*4,
            normalized_time=s, component_diagnostics=diagnostics)

    diagnostics = {}
    result = run(diagnostics)
    assert result.item() == 1 and len(calls) == 1
    assert set(diagnostics) == {"required_rule"}
    witness = diagnostics["required_rule"]
    assert witness["maximum"] == 1 and witness["time_s"] == 2
    assert witness["progress"] == float(torch.nextafter(s, torch.full_like(s, -float("inf"))))
    assert calls[0].shape == (1, 3)
    # Omitting diagnostics does not rerun/change the scene acceptance query.
    assert torch.equal(run(None), result) and len(calls) == 2
# END source-world observer hooks are not part of the online boundary contract.


def test_world_capture_keeps_each_distinct_failure_without_repeating_queries():
    from test_source_world_deduplication import batch_inputs, recording_scene

    def report(joints, query):
        return {"residual": 1., "unmodelled_geometry_names": [],
            "scene": {"query_gaps": [], "pairs": [{"signed_distance_m": -.001}],
                      "violations": [{"kind": "collision", "entity": float(joints[0])}]}}
    args, kwargs = batch_inputs()
    args[0][1, 0, 0] = .5
    scene, calls = recording_scene(report)
    diagnostics = {"capture_components": True}
    residual = scene.check_world_geometry(*args, **kwargs, diagnostics=diagnostics)
    assert torch.equal(residual, torch.ones_like(residual))
    assert len(calls) == 2 and len(diagnostics["failure_witnesses"]) == 2
    assert diagnostics["failed_states"] == 6
    assert [row["witness_index"] for row in diagnostics["failure_witness_references"]] == [0, 0, 0, 1, 0, 0]
    witnesses = diagnostics["failure_witnesses"]
    assert [row["planning_joint_positions"][0] for row in witnesses] == [.25, .5]
    assert [row["violations"][0]["entity"] for row in witnesses] == [.25, .5]
    assert all(row["trajectory_time_s"] == .5 and row["progress"] == .75 for row in witnesses)


def test_witness_binding_does_not_merge_same_q_with_different_scene_inputs():
    q = np.zeros((2, 3, 2), np.float32)
    q[1, -1, 0] = 1.  # Different complete trajectories meet at the same first q.
    dq = np.zeros_like(q)
    trace = {"input_position": q, "input_velocity": dq,
        "component_failures": [{"batch_index": i, "component": "source_world_collision",
                                "expanded_query_index": 0} for i in range(2)],
        "world_geometry": {"failure_witnesses": [{"owner_pose": "first"}, {"owner_pose": "second"}],
            "failure_witness_references": [{"batch_index": i, "expanded_query_index": 0,
                                             "witness_index": i} for i in range(2)]}}
    state = SimpleNamespace(joint_names=["a", "b"], position=torch.from_numpy(q)[None],
                            velocity=torch.from_numpy(dq)[None])
    report = bind_trace_to_native_seed(trace, state, 1, ["a", "b"])
    assert report["component_failures"][0]["geometry_witnesses"] == [{"owner_pose": "second"}]


def test_visualization_requires_matching_native_constraints_and_complete_scopes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent/"planning"))
    from visualize_native_rejection import matched_failures

    path = tmp_path/"native.npz"
    original = {"seeds": [{"seed_index": 0, "metrics": {
        label: {"constraints": {"process_scene": value}}
        for label, value in (("rollout", 0.), ("interpolated", 100000.))}}]}
    replay = {"status": "completed", "native_result": str(path), "labels": {
        label: {"seeds": [{"seed_index": 0, "constraints": {"process_scene": value},
            "source_trace": {"status": "recorded", "component_failures": [], "world_geometry": {}}}]}
        for label, value in (("rollout", 0.), ("interpolated", 100000.))}}
    assert all(row["equal"] for row in matched_failures(replay, original, path, 0)[2].values())
    replay["labels"]["interpolated"]["seeds"][0]["constraints"]["process_scene"] = 0.
    with pytest.raises(ValueError, match="do not reproduce"):
        matched_failures(replay, original, path, 0)
    replay["labels"].pop("interpolated")
    with pytest.raises(ValueError, match="every archived"):
        matched_failures(replay, original, path, 0)
