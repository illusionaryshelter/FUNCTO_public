"""Candidate scoring orders completed native proposals before interval checks."""
# BEGIN exercise the actual production selection order with explicit fixture evidence.
from types import SimpleNamespace

import pytest

from manipulation.planning import _CuroboPlanner


@pytest.mark.parametrize("rejected_first", [False, True])
def test_fixed_objective_selects_best_before_interval_queries(monkeypatch, rejected_first):
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.seed_strategies = ["first", "second"]
    planner.last_optimizer_diagnostics = []
    planner.native_diagnostics = lambda result, index: {
        "success": not (rejected_first and index == 0),
        "rollout": {"constraints": {"task": float(rejected_first and index == 0)}}}
    constructed, evaluated = [], []
    monkeypatch.setattr("manipulation.planning._trajectory_candidate", lambda result, index: index)

    def curve(index):
        # A rejected native seed may have no finite exportable curve. It must
        # not hide another sampled-feasible proposal during quality sorting.
        assert not (rejected_first and index == 0)
        constructed.append(index)
        return index

    planner._native_curve = curve
    planner._candidate_objective = lambda index: {"fixed_objective": [20., 1.][index]}

    def qualify(index, *, native_report):
        evaluated.append(index)
        return index, native_report, 3

    planner.evaluate_candidate = qualify
    candidate, trajectory, report = planner.select_candidate(SimpleNamespace(num_seeds=2))
    assert candidate == trajectory == report["selected_seed_index"] == 1
    assert evaluated == [1] and report["interval_query_points"] == 3
    assert constructed == ([1] if rejected_first else [0, 1])


def test_lower_scored_interval_failure_cannot_replace_a_qualified_candidate(monkeypatch):
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.seed_strategies = ["first", "second"]
    planner.last_optimizer_diagnostics = []
    planner.native_diagnostics = lambda result, index: {"success": True, "rollout": {"constraints": {"task": 0.}}}
    monkeypatch.setattr("manipulation.planning._trajectory_candidate", lambda result, index: index)
    planner._native_curve = lambda index: index
    planner._candidate_objective = lambda index: {"fixed_objective": [2., 1.][index]}
    evaluated = []

    def qualify(index, *, native_report):
        evaluated.append(index)
        if index == 1:
            native_report.update(accepted=False, residuals={"interval:process_scene": 1.})
        return index, native_report, 3

    planner.evaluate_candidate = qualify
    _, trajectory, report = planner.select_candidate(SimpleNamespace(num_seeds=2))
    assert trajectory == 0 and report["accepted"]
    assert evaluated == [1, 0] and report["interval_query_points"] == 6
    assert [row["seed_index"] for row in report["candidates"]] == [1, 0]
# END objective rank never substitutes for original hard evidence.
