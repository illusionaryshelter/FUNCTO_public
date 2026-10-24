"""First qualified curves retain quality evidence and independent solver status."""
# BEGIN each candidate is one owned coefficient solution and one exact saved curve.
from unittest.mock import Mock

import numpy as np
import pytest

from manipulation.planning import _ConstrainedPlanner
from test_manipulation_planning import PlannerCandidateSelectionTests


def planner(scores, verdicts, *, on_qualify=None):
    instance = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
    instance.seed_strategies, instance.scene = ['first','second'], object()
    instance.last_validation = None
    instance._controls_curve = Mock(side_effect=lambda controls: int(controls[0,0])-1)
    instance._candidate_objective = Mock(side_effect=lambda index, scene: {'fixed_objective':scores[index]})
    def qualify(index, *, record_work, objective):
        record_work(passes=1)
        record_work(points=3)
        if on_qualify is not None:
            on_qualify(index)
        residual = verdicts[index]
        instance.last_validation = {'seed_index': index} if residual == 0. else None
        return index, {'accepted':residual == 0., 'residuals':{'saved_curve':residual}, **objective}, 3
    instance.evaluate_candidate = Mock(side_effect=qualify)
    return instance


# BEGIN quality remains evidence while the original source order determines delivery.
@pytest.mark.parametrize('solver_success', [False,True])
def test_first_accepted_curve_stops_before_a_lower_quality_score(solver_success):
    instance = planner([20.,1.],[0.,0.])
    solutions = PlannerCandidateSelectionTests.solutions(flags=(solver_success,True))
    owned, trajectory, report = instance.select_candidate(solutions, record_work=Mock(), on_candidate=Mock())
    assert trajectory == report['selected_seed_index'] == 0
    assert report['accepted'] and report['optimizer']['success'] is solver_success
    assert report['fixed_objective'] == 20.
    assert instance._controls_curve.call_count == 1
    assert instance.evaluate_candidate.call_args.args == (0,)
    assert report['validation_query_points'] == 3 and report['evaluated_candidate_count'] == 1
    assert owned is solutions[0]
    np.testing.assert_array_equal(owned.controls, np.ones((4,3)))


def test_failed_lower_score_advances_to_the_first_qualified_curve():
    instance = planner([1.,2.],[1.,0.])
    _, trajectory, report = instance.select_candidate(PlannerCandidateSelectionTests.solutions(), record_work=Mock(), on_candidate=Mock())
    assert trajectory == 1 and report['accepted']
    assert [call.args[0] for call in instance.evaluate_candidate.call_args_list] == [0,1]
    assert report['validation_query_points'] == 6
    assert [row['seed_index'] for row in report['candidates']] == [0,1]
    assert instance.last_validation == {'seed_index': 1}


def test_all_failed_curves_use_actual_saved_curve_residuals():
    instance = planner([2.,1.],[.1,4.])
    solutions = PlannerCandidateSelectionTests.solutions()
    solutions[0].report['objective'] = 1e16
    solutions[1].report['objective'] = 0.
    _, trajectory, report = instance.select_candidate(solutions, record_work=Mock(), on_candidate=Mock())
    assert trajectory == 0 and not report['accepted']
    assert report['residuals'] == {'saved_curve':.1}
    assert report['optimizer']['objective'] == 1e16


def test_selected_termination_report_is_owned_after_source_diagnostic_changes():
    solutions = PlannerCandidateSelectionTests.solutions(flags=(True, True))
    instance = planner([2.,1.],[0.,0.])
    _, trajectory, report = instance.select_candidate(solutions, record_work=Mock(), on_candidate=Mock())
    solutions[0].report['success'] = False
    solutions[0].report['message'] = 'later mutation'
    assert trajectory == 0 and report['optimizer']['success']
    assert report['optimizer']['message'] == 'declared numerical termination'


def test_completed_work_remains_recorded_when_later_candidate_is_interrupted():
    started, completed = [], []
    def work(*,passes=0,points=0):
        started.append(passes)
        completed.append(points)
    def interrupted(index):
        if index == 1:
            raise TimeoutError('second saved curve query completed at deadline')
    instance = planner([2.,1.],[1.,1.],on_qualify=interrupted)
    with pytest.raises(TimeoutError, match='deadline'):
        instance.select_candidate(PlannerCandidateSelectionTests.solutions(), record_work=work, on_candidate=Mock())
    assert sum(started) == 2 and sum(completed) == 6


def test_accepted_callback_finishes_before_return_without_resuming_the_generator():
    instance = planner([2., 1.], [1., 0.])
    calls = []
    def results():
        for index, solution in enumerate(PlannerCandidateSelectionTests.solutions()):
            calls.append(('solve', index))
            yield solution
        pytest.fail('Selection requested another solve after complete binding')
    def bind(result, report):
        calls.append(('bind', report['selected_seed_index'], report['accepted']))
        if report['accepted']:
            calls.append(('binding_check_complete', report['selected_seed_index']))
    _, _, report = instance.select_candidate(results(), record_work=Mock(), on_candidate=bind)
    assert report['accepted'] and report['selected_seed_index'] == 1
    assert calls == [('solve', 0), ('bind', 0, False), ('solve', 1),
                     ('bind', 1, True), ('binding_check_complete', 1)]


def test_incomplete_binding_callback_propagates_before_delivery_or_another_seed():
    instance = planner([2., 1.], [0., 0.])
    def interrupted(result, report):
        raise TimeoutError('execution binding exceeded its deadline')
    with pytest.raises(TimeoutError, match='execution binding'):
        instance.select_candidate(PlannerCandidateSelectionTests.solutions(),
            record_work=Mock(), on_candidate=interrupted)
    assert instance.evaluate_candidate.call_count == 1
# END query accounting and request failures precede the first completed delivery.
