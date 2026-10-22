"""Saved-curve scoring and qualification retain each native proposal's evidence."""
# BEGIN exercise selection with real native result ownership and explicit verdicts.
from unittest.mock import Mock

import pytest
import torch

from manipulation.planning import _CuroboPlanner
from test_manipulation_planning import PlannerCandidateSelectionTests


def planner(scores, verdicts, *, on_score=None, on_qualify=None):
    instance = _CuroboPlanner.__new__(_CuroboPlanner)
    instance.seed_strategies = ["first", "second"]
    instance.last_optimizer_diagnostics = []
    instance._native_curve = Mock(side_effect=lambda result: int(result.solution[0, 0, 0, 0])-1)
    def score(index):
        if on_score is not None:
            on_score(index)
        return {"fixed_objective": scores[index]}
    instance._candidate_objective = Mock(side_effect=score)
    def qualify(index, *, record_work, objective):
        record_work(passes=1)
        record_work(points=3)
        if on_qualify is not None:
            on_qualify(index)
        residual = verdicts[index]
        return index, {"accepted": residual == 0., "residuals": {"saved_curve": residual},
            **objective}, 3
    instance.evaluate_candidate = Mock(side_effect=qualify)
    return instance


@pytest.mark.parametrize("native_success", [False, True])
def test_fixed_score_orders_all_finite_curves_independently_of_native_success(native_success):
    instance = planner([20., 1.], [0., 0.])
    result = PlannerCandidateSelectionTests.native_candidates(flags=(True, native_success))
    owned, trajectory, report = instance.select_candidate(result, record_work=Mock())
    assert trajectory == report["selected_seed_index"] == 1
    assert report["accepted"] and report["optimization_diagnostics"]["success"] is native_success
    assert instance._native_curve.call_count == 2
    assert instance.evaluate_candidate.call_args.args == (1,)
    assert report["validation_query_points"] == 3 and report["evaluated_candidate_count"] == 1
    torch.testing.assert_close(owned.solution, torch.full((1, 1, 4, 3), 2.))


def test_failed_lower_score_does_not_replace_qualified_curve():
    instance = planner([2., 1.], [0., 1.])
    _, trajectory, report = instance.select_candidate(
        PlannerCandidateSelectionTests.native_candidates(), record_work=Mock())
    assert trajectory == 0 and report["accepted"]
    assert [call.args[0] for call in instance.evaluate_candidate.call_args_list] == [1, 0]
    assert report["validation_query_points"] == 6
    assert [row["seed_index"] for row in report["candidates"]] == [1, 0]


def test_all_failed_curves_use_formal_residuals_for_continuation():
    instance = planner([2., 1.], [.1, 4.])
    _, trajectory, report = instance.select_candidate(
        PlannerCandidateSelectionTests.native_candidates(native_violations=(10., 0.)), record_work=Mock())
    assert trajectory == 0 and not report["accepted"]
    assert report["residuals"] == {"saved_curve": .1}
    assert report["optimization_diagnostics"]["rollout"]["constraints"]["cspace"] == 10.


def test_all_seed_evidence_is_owned_before_scoring_reuses_native_buffers():
    result = PlannerCandidateSelectionTests.native_candidates()
    original = result.solution.clone()
    def overwrite(index):
        result.solution.fill_(91.)
        result.metrics.costs_and_constraints.constraints.values[0].fill_(37.)
    instance = planner([2., 1.], [0., 0.], on_score=overwrite)
    owned, trajectory, report = instance.select_candidate(result, record_work=Mock())
    assert trajectory == 1 and report["optimization_diagnostics"]["rollout"]["constraints"]["cspace"] == 0.
    torch.testing.assert_close(owned.solution, original[:, 1:2], rtol=0, atol=0)


def test_completed_work_remains_recorded_when_later_candidate_is_interrupted():
    started, completed = [], []
    def work(*, passes=0, points=0):
        started.append(passes)
        completed.append(points)
    def interrupted(index):
        if index == 0:
            raise TimeoutError("second saved curve query completed at deadline")
    instance = planner([2., 1.], [0., 1.], on_qualify=interrupted)
    with pytest.raises(TimeoutError, match="deadline"):
        instance.select_candidate(PlannerCandidateSelectionTests.native_candidates(), record_work=work)
    assert sum(started) == 2 and sum(completed) == 6
# END completed query accounting does not wait for the batch's final selection.
