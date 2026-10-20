"""Selection preserves completed observed-field evidence without another query."""

from types import SimpleNamespace
from unittest.mock import Mock
from copy import deepcopy

import pytest
import torch

from manipulation.planning import _CuroboPlanner


def candidate_fixture(monkeypatch, *, scene, task=(0., 0.), flags=(False, False)):
    """Two completed CPU metric records keep independent candidate identities."""
    values = [torch.tensor(v)[:, None, None].expand(2, 3, 1).clone()
              for v in (scene, task)]
    metrics = SimpleNamespace(
        costs_and_constraints=SimpleNamespace(
            constraints=SimpleNamespace(names=["process_scene", "skill_process"], values=values)),
        convergence=SimpleNamespace(names=[], values=[]),
        debug=torch.cat((values[0], *[torch.zeros_like(values[0]) for _ in range(3)]), -1),
    )
    result = SimpleNamespace(num_seeds=2, success=torch.tensor([flags]),
                             metrics=metrics, interpolated_metrics=None)
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.seed_strategies = ["reference", "alternate"]
    planner.last_optimizer_diagnostics = []
    planner._native_curve = Mock(side_effect=lambda candidate: candidate)
    planner._candidate_metrics = Mock(side_effect=AssertionError("Completed geometry must not be rerun"))
    monkeypatch.setattr("manipulation.planning._trajectory_candidate",
                        lambda result, index: SimpleNamespace(seed_index=index))
    return planner, result


def test_observed_collision_rejection_keeps_its_original_evidence(monkeypatch):
    planner, result = candidate_fixture(monkeypatch, scene=(.003, .001))
    _, curve, report = planner.select_candidate(result)
    assert not report["accepted"]
    assert curve.seed_index == 1
    assert report["residuals"]["rollout:process_scene"] == pytest.approx(.001)
    planner._candidate_metrics.assert_not_called()
    planner._native_curve.assert_called_once()


def test_another_completed_feasible_seed_can_be_selected(monkeypatch):
    planner, result = candidate_fixture(monkeypatch, scene=(.001, 0.), flags=(False, True))
    _, curve, report = planner.select_candidate(result)
    assert report["accepted"] and curve.seed_index == 1
    planner._candidate_metrics.assert_not_called()

@pytest.mark.parametrize("regular,dense", [(.001, 0.), (0., .001)])
def test_both_original_sampling_grids_must_pass(monkeypatch, regular, dense):
    planner, result = candidate_fixture(monkeypatch, scene=(regular, 0.),
                                        task=(0., .1), flags=(True, False))
    result.interpolated_metrics = deepcopy(result.metrics)
    result.interpolated_metrics.costs_and_constraints.constraints.values[0][0] = dense
    result.interpolated_metrics.debug[0, :, 0] = dense
    _, _, report = planner.select_candidate(result)
    assert not report["accepted"]
    assert not report["candidates"][0]["accepted"]
    planner._candidate_metrics.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), .00001])
def test_success_flag_cannot_override_invalid_observed_residual(monkeypatch, value):
    planner, result = candidate_fixture(monkeypatch, scene=(value, .01), flags=(True, False))
    _, _, report = planner.select_candidate(result)
    assert not report["accepted"]
    planner._candidate_metrics.assert_not_called()
