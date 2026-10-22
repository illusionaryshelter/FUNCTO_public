"""Saved-curve qualification preserves its observed-field evidence per candidate."""
# BEGIN native reports and formal curve queries retain distinct evidence owners.
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints

from manipulation.planning import _CuroboPlanner
from test_manipulation_planning import PlannerCandidateSelectionTests


def candidate_fixture(*, scene, task=(0., 0.), flags=(False, False)):
    result = PlannerCandidateSelectionTests.native_candidates(flags=flags)
    values = [torch.tensor(v)[:, None, None].expand(2, 3, 1).clone() for v in (scene, task)]
    result.metrics.costs_and_constraints = CostsAndConstraints(constraints=CostCollection(
        names=["process_scene", "skill_process"], values=values))
    result.metrics.debug = torch.cat((values[0], *[torch.zeros_like(values[0]) for _ in range(3)]), -1)
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.seed_strategies = ["reference", "alternate"]
    planner.last_optimizer_diagnostics = []
    planner.scene, planner.budget = object(), SimpleNamespace(max_validation_points=100)
    planner._native_curve = Mock(side_effect=lambda candidate: SimpleNamespace(
        seed_index=int(candidate.solution[0, 0, 0, 0])-1, residuals={}))
    planner._candidate_objective = lambda curve: {"fixed_objective": float(curve.seed_index)}
    planner._candidate_domain_validation = lambda curve: {}
    def metrics(curve, current_scene, *, record_work):
        record_work(points=3)
        index = curve.seed_index
        residuals = {"interpolated:process_scene": scene[index], "interpolated:skill_process": task[index]}
        return {"accepted": all(np.isfinite(v) and v <= 0 for v in residuals.values()),
            "residuals": residuals, "collision_residuals": {"interpolated:scene_collision": scene[index]}}, np.array([0., 1.]), 3
    planner._candidate_metrics = Mock(side_effect=metrics)
    planner._candidate_geometry = Mock(return_value={"query_points": 0, "hard_scene_residual": 0.,
        "scene_categories": {}, "scene_components": {}})
    return planner, result


def test_observed_collision_rejection_keeps_selected_curve_evidence():
    planner, result = candidate_fixture(scene=(.003, .001))
    _, curve, report = planner.select_candidate(result, record_work=Mock())
    assert not report["accepted"] and curve.seed_index == 1
    assert report["residuals"]["interpolated:process_scene"] == pytest.approx(.001)
    assert planner._candidate_metrics.call_count == 2
    planner._candidate_geometry.assert_not_called()
    assert planner.delivery_candidate is curve


def test_another_formally_feasible_seed_can_be_selected():
    planner, result = candidate_fixture(scene=(.001, 0.), flags=(True, False))
    _, curve, report = planner.select_candidate(result, record_work=Mock())
    assert report["accepted"] and curve.seed_index == 1
    assert not report["optimization_diagnostics"]["success"]
    assert planner._candidate_metrics.call_count == 2


@pytest.mark.parametrize("regular,dense", [(.001, 0.), (0., .001)])
def test_original_sampling_grids_remain_provenance_for_qualified_saved_curve(regular, dense):
    planner, result = candidate_fixture(scene=(0., .01), flags=(False, False))
    result.interpolated_metrics = result.metrics.clone()
    for metrics, value in ((result.metrics, regular), (result.interpolated_metrics, dense)):
        metrics.costs_and_constraints.constraints.values[0][0] = value
        metrics.debug[0, :, 0] = value
    _, curve, report = planner.select_candidate(result, record_work=Mock())
    assert report["accepted"] and curve.seed_index == 0
    for label, value in (("rollout", regular), ("interpolated", dense)):
        assert report["optimization_diagnostics"][label]["constraints"]["process_scene"] == pytest.approx(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), .00001])
def test_native_success_cannot_override_invalid_saved_curve_residual(value):
    planner, result = candidate_fixture(scene=(value, .01), flags=(True, False))
    _, _, report = planner.select_candidate(result, record_work=Mock())
    assert not report["accepted"]
    assert planner._candidate_metrics.call_count == 2
# END observed residuals are evaluated on the same curve that could be delivered.
