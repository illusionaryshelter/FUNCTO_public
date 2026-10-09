"""Recovery classification must preserve hard metrics, query modes and seeds."""

# BEGIN same-query collision evidence is a diagnostic, not another constraint.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning import (
    _CuroboPlanner, _ProgressMap, _collision_metric_residuals, _coverage_metric_residuals,
    _conditional_unknown_metric_residuals,
    _dense_collision_witnesses, _scene_boundary_evaluate, _scene_update_metric_residuals, _trajectory_candidate,
)
import test_manipulation_planning as _fixtures


# BEGIN optimizer proposals require explicit same-query component evidence.
def candidate_scene_evidence(terms, *, declared=frozenset({"collision/robot/background"})):
    calls = []
    def evaluate(spheres, p, r, s, q, names, *, components_out, **kwargs):
        calls.append(1)
        components_out.update({name: torch.as_tensor(value, dtype=s.dtype).expand_as(s)
                               for name, value in terms.items()})
        return sum((value for name, value in components_out.items()
                    if not name.startswith("conditional/")), torch.zeros_like(s))
    scene = SimpleNamespace(required_progress_anchors=(), evaluate=evaluate,
                            native_collision_components=declared)
    s = torch.tensor([[0., 1.]])
    evidence = {}
    _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
        torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 3), torch.zeros(1, 2, 4), s,
        torch.zeros(1, 2, 1), ("q",), trajectory_time=s, normalized_time=s,
        scene_evidence_out=evidence, optimizer_candidate_evidence=True)
    assert calls == [1], "Candidate evidence must use the original single scene evaluation"
    return evidence


def test_pending_robot_world_candidate_requires_exact_declared_component():
    terms = {"collision/robot/background": 1., "conditional/contact/uncertain": 100.}
    assert not candidate_scene_evidence(terms)["optimizer_hard_invalid"].any()
    assert candidate_scene_evidence(terms, declared=frozenset())["optimizer_hard_invalid"].all()


@pytest.mark.parametrize("hard", ["collision/robot/self", "collision/payload/background",
    "collision/robot/undeclared", "scene_update/background/robot", "unsupported/contact/robot/owner",
    "unknown/forbidden/robot", "contact/required/rule", "contact/material_band/rule", "gripper/controller"])
def test_huge_confirmable_cost_cannot_hide_tiny_nonconfirmable_violation(hard):
    # Float32 aggregate-minus-proxy would lose the 1e-30 term entirely.
    evidence = candidate_scene_evidence({"collision/robot/background": 1e20, hard: 1e-30})
    assert evidence["optimizer_hard_invalid"].all()


@pytest.mark.parametrize("component", ["collision/robot/background", "contact/required/rule"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_proxy_or_hard_component_is_never_pending_eligible(component, bad):
    assert candidate_scene_evidence({component: [[bad, 0.]]})["optimizer_hard_invalid"].tolist() == [[1., 0.]]


def test_candidate_evidence_cannot_cancel_opposite_hard_terms():
    evidence = candidate_scene_evidence({"unknown/forbidden/robot": 1., "contact/required/rule": -1.})
    assert evidence["optimizer_hard_invalid"].all()


@pytest.mark.parametrize("hard_on_left", [False, True])
def test_candidate_evidence_retains_a_hard_violation_only_on_one_mode_side(hard_on_left):
    calls = []
    def evaluate(spheres, p, r, progress, q, names, *, components_out, **kwargs):
        calls.append(progress.shape[1])
        side = progress < .5 if hard_on_left else progress > .5
        other = progress > .5 if hard_on_left else progress < .5
        components_out["scene_update/background/robot"] = side.to(progress.dtype)*1e-30
        components_out["collision/robot/background"] = other.to(progress.dtype)*1e20
        return sum(components_out.values())
    scene = SimpleNamespace(required_progress_anchors=(.5,), contacts=(), evaluate=evaluate,
        native_collision_components=frozenset({"collision/robot/background"}))
    s = torch.tensor([[.5]])
    evidence = {}
    result = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
        torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 3), torch.zeros(1, 1, 4), s,
        torch.zeros(1, 1, 1), ("q",), trajectory_time=s, normalized_time=s,
        scene_evidence_out=evidence, optimizer_candidate_evidence=True)
    assert calls == [3], "Exact and both mode sides remain a single batched evaluation"
    assert result.item() == pytest.approx(1e20)
    assert evidence["optimizer_hard_invalid"].item() == 1.
# END positive broad-phase evidence may be deferred; no original hard requirement is removed.


@pytest.mark.parametrize("category", ["scene_update/background/robot", "unsupported/contact/robot/owner",
    "unknown/forbidden/robot", "contact/required/rule", "gripper/controller",
    "conditional/collision/robot/background", "conditional/unknown/robot", "unknown/other/robot"])
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
    # BEGIN only the explicit forbidden-space component authorizes coverage search.
    assert bool((evidence["coverage"] == int(category.startswith("unknown/forbidden/"))).all())
    # BEGIN only conditional unknown enters the fourth evidence channel.
    assert bool((evidence["conditional_unknown"] == int(category.startswith("conditional/unknown/"))).all())
    # END contact uncertainty and collision uncertainty are separate evidence.
    # END aggregate scene errors and conditional observations stay distinct.
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
            trajectory_time=s*2, normalized_time=s, scene_evidence_out=evidence,
            optimizer_candidate_evidence=capture)
        results.append((result.detach(), torch.autograd.grad(result.sum(), q)[0]))
        if capture:
            assert evidence["collision"].item() == 2.25  # right-side collision, left-side hard maximum
            # BEGIN each diagnostic channel takes its own boundary-mode maximum.
            assert evidence["scene_update"].item() == 5.
            assert evidence["optimizer_hard_invalid"].item() == 1.
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
    # BEGIN all four channels have explicit seed/time identity.
    result.metrics.debug = torch.zeros(2, 9, 4)
    result.metrics.debug[..., 1] = 100000.
    # END support is deliberately positive beside zero collision.
    result.interpolated_metrics = result.metrics.clone()
    return result


def test_support_only_dense_peaks_do_not_become_collision_witnesses():
    result = _trajectory_candidate(collision_result(), 0)
    # BEGIN recovery reads this fixture's actual nine ordinary metric samples.
    report = _dense_collision_witnesses(result, regular_horizon=9, anchor_parameters=(), capacity=4)
    # END the dense and ordinary layouts are validated independently.
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
    # BEGIN retain the ordinary positive's actual nine-point time layout.
    assert _dense_collision_witnesses(selected, regular_horizon=9, anchor_parameters=(), capacity=4)["parameters"] == []
    # END ordinary-only evidence cannot invent a missing dense interval.
    assert all(value == 0 for value in _collision_metric_residuals(_trajectory_candidate(result, 0).metrics).values())


def test_missing_debug_evidence_is_an_error_instead_of_aggregate_fallback():
    result = collision_result()
    result.metrics.debug = None
    with pytest.raises(RuntimeError, match="scene.*evidence"):
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
    # BEGIN use the selected native fixture's complete ordinary metric horizon.
    assert _dense_collision_witnesses(selected, regular_horizon=9,
        anchor_parameters=(), capacity=4)["parameters"] == []
    # END independent support evidence never becomes collision evidence.


@pytest.mark.parametrize("shape", [(2, 9, 1), (2, 9, 2), (2, 9, 3), (1, 9, 4)])
def test_obsolete_or_misaligned_scene_evidence_is_rejected(shape):
    result = collision_result()
    result.metrics.debug = torch.zeros(shape)
    for reader in (_collision_metric_residuals, _scene_update_metric_residuals, _coverage_metric_residuals,
                   _conditional_unknown_metric_residuals):
        with pytest.raises(RuntimeError, match="scene.*evidence"):
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


# BEGIN coverage evidence has its own modes, grids and seed identity.
def test_coverage_mode_max_preserves_all_actor_terms_without_changing_hard_gradient():
    s = torch.tensor([[.5]], dtype=torch.float64)
    calls = []
    def evaluate(spheres, p, r, progress, q, names, *, components_out, **kwargs):
        calls.append(progress.shape[1])
        components_out.update({
            "collision/robot/owner": 2*(progress > .5).to(q.dtype),
            "scene_update/robot/owner": 5*(progress < .5).to(q.dtype),
            "unknown/forbidden/robot": q[..., 0].square()+3*(progress == .5),
            "unknown/forbidden/payload": .5*q[..., 0].square(),
            "conditional/unknown/robot": torch.full_like(progress, 100.)})
        return sum((value for name, value in components_out.items()
                    if not name.startswith("conditional/")), torch.zeros_like(progress))
    scene = SimpleNamespace(required_progress_anchors=(.5,), contacts=(), evaluate=evaluate)
    mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    q = torch.ones(1, 1, 1, dtype=torch.float64, requires_grad=True)
    evidence = {}
    result = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.), mapping,
        torch.zeros(1, 1, 1, 4, dtype=torch.float64), torch.zeros(1, 1, 3, dtype=torch.float64),
        torch.zeros(1, 1, 4, dtype=torch.float64), s, q, ("q",), trajectory_time=s,
        normalized_time=s, scene_evidence_out=evidence)
    assert calls == [3]
    assert evidence["coverage"].item() == 4.5  # exact mode; both robot and payload count
    assert evidence["scene_update"].item() == 5.  # left mode
    assert evidence["collision"].item() == 2.  # right mode
    # BEGIN permitted uncertainty is retained without hard cost or gradients.
    assert evidence["conditional_unknown"].item() == 100.
    assert not evidence["conditional_unknown"].requires_grad
    # END the hard result below is unchanged despite positive unknown evidence.
    assert result.item() == 6.5  # the original left-mode hard maximum
    assert torch.autograd.grad(result.sum(), q)[0].item() == 3.
    assert not evidence["coverage"].requires_grad


def test_forbidden_coverage_retains_both_grids_and_seeds_without_collision_witnesses():
    result = collision_result()
    result.metrics.debug[1, 2, 2] = 2000.
    result.interpolated_metrics.debug[1, 4, 2] = 3000.
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    report = planner.native_diagnostics(result, seed_index=1)
    assert report["rollout"]["coverage_residuals"] == {"unknown_forbidden": 2000.}
    assert report["interpolated"]["coverage_residuals"] == {"unknown_forbidden": 3000.}
    selected = _trajectory_candidate(result, 1)
    assert _coverage_metric_residuals(selected.metrics) == {"unknown_forbidden": 2000.}
    assert _coverage_metric_residuals(selected.interpolated_metrics) == {"unknown_forbidden": 3000.}
    assert _coverage_metric_residuals(result.metrics, 0) == {"unknown_forbidden": 0.}
    assert _dense_collision_witnesses(selected, regular_horizon=9,
        anchor_parameters=(), capacity=4)["parameters"] == []


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_coverage_metrics_do_not_hide_nonfinite_samples(nonfinite):
    result = collision_result()
    result.metrics.debug[1, 2, 2] = nonfinite
    actual = _coverage_metric_residuals(result.metrics, 1)["unknown_forbidden"]
    assert not np.isfinite(actual)
    if not np.isnan(nonfinite):
        assert actual == nonfinite
    assert _coverage_metric_residuals(result.metrics, 0) == {"unknown_forbidden": 0.}
# END coverage authorizes only bounded search, never direct acceptance or collision repair.


# BEGIN allowed unknown retains its own sampled evidence without collision witnesses.
def test_conditional_unknown_retains_both_grids_and_selected_seed_copy():
    result = collision_result()
    result.metrics.debug[1, 2, 3] = .02
    result.interpolated_metrics.debug[1, 4, 3] = .04
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    report = planner.native_diagnostics(result, seed_index=1)
    assert report["rollout"]["conditional_unknown_residuals"]["conditional_unknown"] == pytest.approx(.02)
    assert report["interpolated"]["conditional_unknown_residuals"]["conditional_unknown"] == pytest.approx(.04)
    selected = _trajectory_candidate(result, 1)
    result.metrics.debug.zero_()
    result.interpolated_metrics.debug.zero_()
    assert _conditional_unknown_metric_residuals(selected.metrics)["conditional_unknown"] == pytest.approx(.02)
    assert _conditional_unknown_metric_residuals(selected.interpolated_metrics)["conditional_unknown"] == pytest.approx(.04)
    assert _dense_collision_witnesses(selected, regular_horizon=9,
        anchor_parameters=(), capacity=4)["parameters"] == []


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_conditional_unknown_metrics_preserve_invalid_query_evidence(nonfinite):
    result = collision_result()
    result.metrics.debug[1, 2, 3] = nonfinite
    actual = _conditional_unknown_metric_residuals(result.metrics, 1)["conditional_unknown"]
    assert not np.isfinite(actual)
    if not np.isnan(nonfinite):
        assert actual == nonfinite
    assert _conditional_unknown_metric_residuals(result.metrics, 0) == {"conditional_unknown": 0.}
# END recording conditional uncertainty does not certify it as observed free space.
