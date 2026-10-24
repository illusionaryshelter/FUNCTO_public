"""Current scene reports retain physical time, pair identity and failure points."""

# BEGIN failure localization consumes the current same-query scene interfaces.
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from manipulation.pin_kinematics import PinKinematics
from manipulation.planning import (
    JointTrajectory, PlanningBudget, _ConstrainedPlanner, _ProgressMap,
    _scene_boundary_evaluate,
)
from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath
from test_online_geometry import moving_scene


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
    mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    s = torch.tensor([[.5]])
    def run(diagnostics):
        return _scene_boundary_evaluate(scene, problem, mapping,
            torch.zeros((1, 1, 0, 4)), torch.zeros((1, 1, 3)), torch.zeros((1, 1, 4)),
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
    assert torch.equal(run(None), result) and len(calls) == 2


def test_native_pair_failures_keep_candidate_time_and_mode_on_a_progress_plateau(tmp_path):
    """Two candidate rows collide at different physical ends of one pause.

    The actual registered robot consists of two 2 cm boxes. A named slide
    translates the second box, giving 5/8 mm penetration at q=.015/.012 and 1 cm
    separation at q=.03. Equal progress therefore cannot identify a failure
    without its original candidate, physical time and side of the mode join.
    """
    original = moving_scene(tmp_path)
    scene = SimpleNamespace(required_progress_anchors=(.5,), contacts=(), evaluate=original.evaluate)
    mapping = _ProgressMap(np.array([0., .25, .75, 1.]), np.array([0., .5, .5, 1.]))
    q = torch.tensor([[[.03, -.02], [.012, -.02]],
                      [[.015, -.02], [.03, -.02]]], dtype=torch.float64)
    progress = torch.full((2, 2), .5, dtype=q.dtype)
    parameters = torch.tensor([[.25, .75], [.25, .75]], dtype=q.dtype)
    quaternion = q.new_zeros((2, 2, 4)); quaternion[..., 0] = 1.
    evidence, diagnostics, counts = [], {}, {}
    # BEGIN full evidence is compared with its own original Coal query.
    native_results, original_query = [], original.robot_model.query_pair_batches
    def capture_native(*args, **kwargs):
        # BEGIN retain the actual numeric rows before scene evidence assembly.
        for batch in original_query(*args, **kwargs):
            native_results.extend(batch)
            yield batch
        # END state-major rows retain their original expanded time index.
    with patch.object(original, "evaluate", wraps=original.evaluate) as evaluate, \
         patch.object(original.robot_model, "query_pair_batches", side_effect=capture_native) as native:
        scene.evaluate = evaluate
        residual = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
            mapping, q.new_empty((2, 2, 0, 4)), q.new_zeros((2, 2, 3)), quaternion,
            progress, q, ("left_slide", "right_slide"), trajectory_time=4*parameters,
            normalized_time=parameters, component_diagnostics=diagnostics,
            geometry_evidence_out=evidence, geometry_scope="robot", query_counts=counts)
    np.testing.assert_allclose(residual.numpy(), [[0., .008], [.005, 0.]], atol=1e-10)
    assert evaluate.call_count == 1 and native.call_count == 1
    assert len(native_results) == 8
    assert counts == {"exact_states": 4, "left_states": 2, "right_states": 2, "evaluated_states": 8}
    failures = [row for row in evidence if row["status"] == "intersection"]
    assert {(tuple(row["query_index"]), row["side"], row["time_s"]) for row in failures} == {
        ((0, 1), 0, 3.), ((0, 1), 1, 3.), ((1, 0), 0, 1.), ((1, 0), -1, 1.)}
    assert len(evidence) == 8
    assert {(tuple(row["query_index"]), row["side"], row["time_s"]) for row in evidence} == {
        ((0, 0), 0, 1.), ((0, 1), 0, 3.), ((1, 0), 0, 1.), ((1, 1), 0, 3.),
        ((0, 0), -1, 1.), ((1, 0), -1, 1.), ((0, 1), 1, 3.), ((1, 1), 1, 3.)}
    expanded_shape = tuple(evaluate.call_args.args[3].shape)
    for row in evidence:
        assert (row["first"], row["second"]) == ("first", "second")
        assert row["geometry_entity_key"] == {"first": "robot/first", "second": "robot/second"}
        assert row["component"] == "collision/robot/self"
        # Original Coal witness points reach the report unchanged. Their
        # signed separation along the reported normal explains the rejection.
        native_index = np.ravel_multi_index(tuple(row["state_index"]), expanded_shape)
        saved = native_results[native_index][0]
        assert row["signed_distance_m"] == saved[0]
        for key, columns in (("witness_first", slice(1, 4)),
                             ("witness_second", slice(4, 7)),
                             ("normal_first_to_second", slice(7, 10))):
            np.testing.assert_array_equal(row[key], saved[columns])
        signed = np.dot(np.asarray(row["witness_second"])-row["witness_first"],
                        row["normal_first_to_second"])
        assert signed == pytest.approx(row["signed_distance_m"], abs=1e-10)
    # The bounded display summary names the earliest failure and the largest
    # original pair residual. Both records must match the complete evidence.
    summary = original.last_query_report["nonclear_pairs"]
    assert len(summary) == 2
    selected = {row["selection"][0]: row for row in summary}
    assert selected["first"]["time_s"] == 1.
    assert selected["first"]["state_index"] == [1, 0]
    assert selected["maximum_pair_residual"]["time_s"] == 3.
    assert selected["maximum_pair_residual"]["state_index"] == [0, 1]
    for row in summary:
        complete = next(item for item in failures if item["state_index"] == row["state_index"])
        for key in ("component", "first", "second", "time_s", "progress", "permission_progress",
                    "signed_distance_m", "witness_first", "witness_second", "normal_first_to_second"):
            np.testing.assert_array_equal(row[key], complete[key])
        assert row["pair_residual_value"] == -complete["signed_distance_m"]
    # END reducing display records does not reduce complete pair evidence.
    assert diagnostics["collision/robot/self"]["time_s"] == 3.


def test_candidate_report_owns_its_failure_time_entities_and_native_witnesses(tmp_path):
    """A completed candidate keeps its same-query report after scene reuse."""
    scene = moving_scene(tmp_path)
    names = ("left_slide", "right_slide")
    initial = np.array([.015, -.02])
    path = PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared reference")
    problem = PlanningProblem(names, initial, np.zeros(2), "tool", "world", np.eye(4), np.eye(4),
        (MotionPhase("reference", path),))
    curve = JointTrajectory(names, np.array([0., .05]), np.stack((initial, initial)),
        np.zeros((2, 2)), np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
    planner.torch, planner.problem, planner.joint_names = torch, problem, names
    planner.budget, planner.anchors, planner.check_request = PlanningBudget(), problem.anchors, Mock()
    planner.robot_metadata = {'active_joint_names': list(names), 'links': scene.robot_model.model.nframes}
    planner.pin_kinematics = PinKinematics(scene.robot_model.model,
        frames=("base", "left", "tool"), T_reference_base=np.eye(4))
    record_work = Mock()
    # BEGIN capture the complete same-call records alongside the public summary.
    complete, original_evaluate = [], scene.evaluate
    def capture_complete(*args, **kwargs):
        kwargs["geometry_evidence_out"] = complete
        return original_evaluate(*args, **kwargs)
    with patch.object(scene, "evaluate", side_effect=capture_complete):
        report, parameters, count, context = planner._candidate_metrics(curve, scene, record_work=record_work)
    assert not report["accepted"]
    assert report["collision_residuals"]["collision"] == pytest.approx(.005, abs=1e-10)
    assert report["scene_components"]["collision/robot/self"]["time_s"] == 0.
    record_work.assert_called_once_with(points=count)
    assert count == len(parameters) == 3
    complete_count = len(complete)
    assert complete_count == report["observed_geometry"]["pair_queries"]
    native_complete = [row for row in complete if row["component"] == "collision/robot/self"]
    assert len(native_complete) == 3
    assert [row["state_index"] for row in native_complete] == [[0, 0], [0, 1], [0, 2]]
    np.testing.assert_array_equal([row["time_s"] for row in native_complete], parameters*curve.duration)
    assert all(row["status"] == "intersection" for row in native_complete)
    saved = report["observed_geometry"]["nonclear_pairs"]
    assert len(saved) == 1
    assert saved[0]["selection"] == ["first", "maximum_pair_residual"]
    assert saved[0]["state_index"] == [0, 0] and saved[0]["time_s"] == 0.
    assert {(row["first"], row["second"]) for row in saved} == {("first", "second")}
    for row in native_complete:
        assert (row["first"], row["second"]) == ("first", "second")
        assert row["geometry_entity_key"] == {"first": "robot/first", "second": "robot/second"}
        assert row["signed_distance_m"] == pytest.approx(-.005, abs=1e-10)
        assert np.isfinite(row["witness_first"]).all() and np.isfinite(row["witness_second"]).all()
    for key in ("component", "first", "second", "time_s", "progress", "permission_progress",
                "signed_distance_m", "witness_first", "witness_second", "normal_first_to_second"):
        np.testing.assert_array_equal(saved[0][key], native_complete[0][key])
    scene.last_query_report["nonclear_pairs"].clear()
    assert len(saved) == 1 and len(complete) == complete_count and len(native_complete) == 3
    # END completed candidate summaries own their retained original witnesses.
# END current pair evidence and completed reports retain physical failure identity.
