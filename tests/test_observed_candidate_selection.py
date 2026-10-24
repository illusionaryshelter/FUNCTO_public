"""Saved-curve qualification preserves observed-field evidence per candidate."""
# BEGIN every report refers to the curve evaluated at the final qualification boundary.
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from manipulation.planning import _ConstrainedPlanner
from test_manipulation_planning import PlannerCandidateSelectionTests


def candidate_fixture(*, scene, task=(0.,0.), flags=(False,False)):
    solutions = PlannerCandidateSelectionTests.solutions(flags=flags)
    planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
    planner.seed_strategies = ['reference','alternate']
    planner.scene, planner.budget = object(), SimpleNamespace(max_validation_points=100)
    planner._controls_curve = Mock(side_effect=lambda controls: SimpleNamespace(seed_index=int(controls[0,0])-1,residuals={}))
    planner._candidate_objective = lambda curve, scene: {'fixed_objective':float(curve.seed_index)}
    planner._candidate_domain_validation = lambda curve: {}
    def metrics(curve,current_scene,*,record_work):
        record_work(points=3)
        index = curve.seed_index
        residuals = {'process_scene':scene[index],'task':task[index]}
        return {'accepted':all(np.isfinite(v) and v <= 0 for v in residuals.values()),
            'residuals':residuals, 'collision_residuals':{'collision':scene[index]}}, np.array([0.,1.]),3,(current_scene, [])
    planner._candidate_metrics = Mock(side_effect=metrics)
    planner._candidate_geometry = Mock(return_value={'query_points':0,'hard_scene_residual':0.,
        'scene_categories':{},'scene_components':{}})
    return planner,solutions


def test_observed_collision_rejection_keeps_selected_curve_evidence():
    planner,solutions = candidate_fixture(scene=(.003,.001))
    _,curve,report = planner.select_candidate(solutions,record_work=Mock(), on_candidate=Mock())
    assert not report['accepted'] and curve.seed_index == 1
    assert report['residuals']['process_scene'] == pytest.approx(.001)
    assert report['collision_residuals'] == {'collision':.001}
    assert planner._candidate_metrics.call_count == 2
    planner._candidate_geometry.assert_not_called()
    assert planner.delivery_candidate is curve


def test_formally_feasible_solution_can_be_selected_after_solver_failure():
    planner,solutions = candidate_fixture(scene=(.001,0.),flags=(True,False))
    _,curve,report = planner.select_candidate(solutions,record_work=Mock(), on_candidate=Mock())
    assert report['accepted'] and curve.seed_index == 1
    assert not report['optimizer']['success']
    assert planner._candidate_metrics.call_count == 2


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),.00001])
def test_solver_success_cannot_override_invalid_saved_curve_residual(value):
    planner,solutions = candidate_fixture(scene=(value,.01),flags=(True,False))
    _,_,report = planner.select_candidate(solutions,record_work=Mock(), on_candidate=Mock())
    assert not report['accepted'] and planner._candidate_metrics.call_count == 2


def test_selected_interval_categories_preserve_unknown_and_collision_meanings():
    planner,solutions = candidate_fixture(scene=(0.,0.))
    planner._candidate_geometry.side_effect = [
        {'query_points':2,'hard_scene_residual':.001,'scene_categories':{'coverage':.001,'conditional_unknown':.2},
         'scene_components':{'unknown/forbidden/robot':{'maximum':.001,'time_s':.5,'progress':.5}}},
        {'query_points':2,'hard_scene_residual':0.,'scene_categories':{'conditional_unknown':.4},'scene_components':{}}]
    _,curve,report = planner.select_candidate(solutions,record_work=Mock(), on_candidate=Mock())
    assert report['accepted'] and curve.seed_index == 1
    first = report['candidates'][0]
    assert first['coverage_residuals']['interval:unknown_forbidden'] == .001
    assert first['collision_residuals']['interval:scene_collision'] == 0.
    assert report['conditional_unknown_residuals']['interval:conditional_unknown'] == .4
    assert report['validation_query_points'] == 10
# END independent accepted uncertainty remains visible without changing hard evidence.
