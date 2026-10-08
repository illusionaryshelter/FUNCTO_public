"""Recovery classification must preserve hard metrics, query modes and seeds."""

# BEGIN same-query collision evidence is a diagnostic, not another constraint.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning import (
    _CuroboPlanner, _ProgressMap, _collision_metric_residuals,
    _dense_collision_witnesses, _scene_boundary_evaluate, _scene_update_metric_residuals, _trajectory_candidate,
)
import test_manipulation_planning as _fixtures


@pytest.mark.parametrize("category", ["scene_update/background/robot", "unsupported/contact/robot/owner",
    "unknown/forbidden/robot", "contact/required/rule", "gripper/controller",
    "conditional/collision/robot/background"])
def test_noncollision_components_do_not_generate_collision_evidence(category):
    calls = []
    def evaluate(spheres, p, r, s, q, names, *, components_out, **kwargs):
        calls.append(1)
        components_out[category] = torch.ones_like(s)
        return torch.zeros_like(s) if category.startswith("conditional/") else torch.ones_like(s)
    scene = SimpleNamespace(required_progress_anchors=(), evaluate=evaluate)
    s = torch.tensor([[0., .5, 1.]])
    evidence = {}
    result = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
        torch.zeros(1, 3, 1, 4), torch.zeros(1, 3, 3), torch.zeros(1, 3, 4), s,
        torch.zeros(1, 3, 1), ("q",), trajectory_time=s, normalized_time=s, scene_evidence_out=evidence)
    assert calls == [1]
    assert bool((result == (0 if category.startswith("conditional/") else 1)).all())
    assert bool((evidence["collision"] == 0).all())
    # BEGIN numerical support is retained without becoming collision evidence.
    assert bool((evidence["scene_update"] == int(category.startswith("scene_update/"))).all())
    # END contact, controller and unknown-space categories remain distinct.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_collision_mode_max_does_not_follow_support_max_or_change_its_gradient(dtype):
    s = torch.tensor([[.5]], dtype=dtype)
    mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    calls = []
    def evaluate(spheres, p, r, progress, q, names, *, components_out=None, **kwargs):
        calls.append(progress.shape[1])
        collision = .25*q[..., 0].square()+2*(progress > .5)
        support = q[..., 0].square()+4*(progress < .5)
        if components_out is not None:
            components_out.update({"collision/robot/owner": collision, "scene_update/robot/owner": support,
                "conditional/collision/robot/owner": torch.full_like(progress, 100.)})
        return collision+support
    scene = SimpleNamespace(required_progress_anchors=(.5,), contacts=(), evaluate=evaluate)
    results = []
    for capture in (False, True):
        q = torch.ones(1, 1, 1, dtype=dtype, requires_grad=True)
        evidence = {} if capture else None
        result = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.), mapping,
            torch.zeros(1, 1, 1, 4, dtype=dtype), torch.zeros(1, 1, 3, dtype=dtype),
            torch.zeros(1, 1, 4, dtype=dtype), s, q, ("q",),
            trajectory_time=s*2, normalized_time=s, scene_evidence_out=evidence)
        results.append((result.detach(), torch.autograd.grad(result.sum(), q)[0]))
        if capture:
            assert evidence["collision"].item() == 2.25  # right-side collision, left-side hard maximum
            # BEGIN each diagnostic channel takes its own boundary-mode maximum.
            assert evidence["scene_update"].item() == 5.
            assert not evidence["scene_update"].requires_grad
            # END the left support witness cannot hide the right collision witness.
            assert not evidence["collision"].requires_grad
    assert calls == [3, 3], "Each evaluation queries exact/left/right geometry in one batch"
    assert results[0][0].item() == 5.25
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def collision_result():
    result = _fixtures.PlannerCandidateSelectionTests.native_candidates(flags=(False, False))
    constraints = result.metrics.costs_and_constraints.constraints
    constraints.names = ["process_scene", "self_collision", "anchor_self_collision"]
    constraints.values = [torch.ones(2, 9, 1)*100000., torch.zeros(2, 9, 1), torch.zeros(2, 9, 1)]
    # BEGIN both channels have explicit seed/time identity.
    result.metrics.debug = torch.zeros(2, 9, 2)
    result.metrics.debug[..., 1] = 100000.
    # END support is deliberately positive beside zero collision.
    result.interpolated_metrics = result.metrics.clone()
    return result


def test_support_only_dense_peaks_do_not_become_collision_witnesses():
    result = _trajectory_candidate(collision_result(), 0)
    report = _dense_collision_witnesses(result, regular_horizon=3, anchor_parameters=(), capacity=4)
    assert report["parameters"] == [] and report["peak_count"] == 0
    assert _collision_metric_residuals(result.interpolated_metrics) == {
        "scene_collision": 0., "self_collision": 0., "anchor_self_collision": 0.}


@pytest.mark.parametrize("component", ["scene_collision", "self_collision", "anchor_self_collision"])
def test_regular_only_collisions_remain_routable_and_do_not_invent_dense_times(component):
    result = collision_result()
    if component == "scene_collision":
        result.metrics.debug[1, 3, 0] = 2.
    else:
        constraints = result.metrics.costs_and_constraints.constraints
        constraints.values[constraints.names.index(component)][1, 3] = 2.
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    native = planner.native_diagnostics(result, seed_index=1)
    assert native["rollout"]["collision_residuals"][component] == 2.
    assert all(value == 0 for value in native["interpolated"]["collision_residuals"].values())
    selected = _trajectory_candidate(result, 1)
    assert _collision_metric_residuals(selected.metrics)[component] == 2.
    assert _dense_collision_witnesses(selected, regular_horizon=3, anchor_parameters=(), capacity=4)["parameters"] == []
    assert all(value == 0 for value in _collision_metric_residuals(_trajectory_candidate(result, 0).metrics).values())


def test_missing_debug_evidence_is_an_error_instead_of_aggregate_fallback():
    result = collision_result()
    result.metrics.debug = None
    with pytest.raises(RuntimeError, match="collision/scene-update evidence"):
        _collision_metric_residuals(result.metrics)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_scene_diagnostic_retains_nonfinite_witness_beside_finite_zero(nonfinite):
    def evaluate(spheres, p, r, s, q, names, *, components_out, **kwargs):
        components_out["scene_update/background/robot"] = s.new_tensor([[nonfinite, 0.]])
        return components_out["scene_update/background/robot"]
    scene = SimpleNamespace(required_progress_anchors=(), evaluate=evaluate)
    s = torch.tensor([[0., 1.]])
    diagnostics = {}
    _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), torch.zeros(1, 2, 1, 4),
        torch.zeros(1, 2, 3), torch.zeros(1, 2, 4), s, torch.zeros(1, 2, 1), ("q",),
        trajectory_time=s, normalized_time=s, component_diagnostics=diagnostics)
    assert not np.isfinite(diagnostics["scene_update/background/robot"]["maximum"])
    assert diagnostics["scene_update/background/robot"]["time_s"] == 0.
# END no test relaxes a hard requirement or asks the solver to repair an impossible goal.


# BEGIN selected support evidence is diagnostic and never creates collision times.
def test_scene_update_evidence_preserves_both_grid_and_seed_identity():
    result = collision_result()
    result.metrics.debug[1, 2, 1] = 200000.
    result.interpolated_metrics.debug[1, 4, 1] = 300000.
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    report = planner.native_diagnostics(result, seed_index=1)
    assert report["rollout"]["scene_update_residuals"] == {"scene_update": 200000.}
    assert report["interpolated"]["scene_update_residuals"] == {"scene_update": 300000.}
    selected = _trajectory_candidate(result, 1)
    assert _scene_update_metric_residuals(selected.metrics) == {"scene_update": 200000.}
    assert _scene_update_metric_residuals(selected.interpolated_metrics) == {"scene_update": 300000.}
    assert _scene_update_metric_residuals(result.metrics, 0) == {"scene_update": 100000.}
    assert _dense_collision_witnesses(selected, regular_horizon=3,
        anchor_parameters=(), capacity=4)["parameters"] == []


@pytest.mark.parametrize("shape", [(2, 9, 1), (2, 9, 3), (1, 9, 2)])
def test_obsolete_or_misaligned_scene_evidence_is_rejected(shape):
    result = collision_result()
    result.metrics.debug = torch.zeros(shape)
    for reader in (_collision_metric_residuals, _scene_update_metric_residuals):
        with pytest.raises(RuntimeError, match="collision/scene-update evidence"):
            reader(result.metrics)
# END no aggregate fallback or legacy single-channel compatibility is permitted.


# BEGIN diagnostic reduction must not hide an invalid sample beside finite zero.
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_scene_update_metrics_retain_nonfinite_witness(nonfinite):
    result = collision_result()
    result.metrics.debug[..., 1].zero_()
    result.metrics.debug[1, 2, 1] = nonfinite
    actual = _scene_update_metric_residuals(result.metrics, 1)["scene_update"]
    assert not np.isfinite(actual)
    if not np.isnan(nonfinite):
        assert actual == nonfinite
    assert _scene_update_metric_residuals(result.metrics, 0) == {"scene_update": 0.}
# END this diagnostic never changes the separate hard acceptance constraint.
