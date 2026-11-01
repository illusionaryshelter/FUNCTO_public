"""Exercise the final pose policy with real task geometry and solver rows.

The retained candidate state has a known event error. Only scene and joint
producers are fixtures, allowing their independent failures to be varied while
the original PlanningProblem evaluates both nominal constraints and evidence.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.direct_planner import candidate_result
from manipulation.planning_problem import MotionPhase, PlanningProblem, PoseEventRequirement, PosePath


@pytest.mark.parametrize('position_error,rotation_error,collision,joint_excess,accepted,status', [
    (0., 0., 0., 0., True, 'within_tolerance'),
    (.01, 0., 0., 0., True, 'warning'),
    (0., .1, 0., 0., True, 'warning'),
    (.01, .1, .2, 0., False, 'warning'),
    (.01, .1, 0., .3, False, 'warning'),
])
def test_pose_warning_preserves_nominal_constraints_and_physical_verdict(
        monkeypatch, position_error, rotation_error, collision, joint_excess, accepted, status):
    """Pose excess is reported, while real solver rows and physical gates persist."""
    path = PosePath([0., 1.], np.stack((np.eye(4), np.eye(4))), source='declared reference')
    event = PoseEventRequirement(1., np.eye(4), np.eye(4), .002, np.pi/180,
        source='declared functional precision')
    problem = PlanningProblem(('joint',), [0.], [0.], 'tool', 'world',
        np.eye(4), np.eye(4), (MotionPhase('declared interval', path),), pose_events=(event,))
    progress = torch.tensor([[0., 1.]], dtype=torch.float64)
    position = torch.zeros((1, 2, 3), dtype=torch.float64)
    position[0, 1, 0] = position_error
    quaternion = torch.tensor([[[1., 0., 0., 0.],
        [np.cos(rotation_error/2), 0., 0., np.sin(rotation_error/2)]]], dtype=torch.float64)
    q = torch.zeros((1, 2, 1), dtype=torch.float64)
    full = np.zeros((2, 1))
    geometry = SimpleNamespace(bound=SimpleNamespace(owner=SimpleNamespace(),
        records=[{'component': 'registered_pair'}]), spatial_values=np.array([collision]),
        values=np.array([collision]), report={'source': 'same retained candidate'},
        unknown_counts=np.zeros(1))
    point = SimpleNamespace(full_controls=full, geometry=geometry,
        candidate_state=(q, q, ('joint',), (position, quaternion), (position*0, position*0)))
    computation = SimpleNamespace(final_point=point, progress=progress, times=progress,
        parameters=np.array([0., 1.]))
    planner = SimpleNamespace(computation=computation, problem=problem,
        _current_full_controls=full.copy(), terminal=SimpleNamespace(report={}),
        pose_objectives=[], candidate_history=[], candidate_evaluations=[])
    trajectory = SimpleNamespace(residuals={})

    def domain_validation(curve):
        curve.residuals = {'continuous_position': joint_excess}
        return {}

    planner._candidate_domain_validation = domain_validation
    monkeypatch.setattr(candidate_result, 'scene_conditions', lambda point: ({}, {}, {}))
    _, report, _ = candidate_result.evaluate_candidate(planner, trajectory,
        record_work=lambda **kwargs: None, objective=dict(fixed_objective=1., objective_components={}))
    assert report['accepted'] is accepted
    assert report['pose_accuracy']['status'] == status
    assert report['pose_accuracy']['maximum_absolute_measurements']['function:0:position_error_m'] == pytest.approx(position_error)
    assert report['pose_accuracy']['maximum_absolute_measurements']['function:0:rotation_error_rad'] == pytest.approx(rotation_error)
    assert 'function_pose:0' not in report['residuals']
    assert planner.candidate_history[-1]['pose_accuracy'] == report['pose_accuracy']
    if accepted:
        np.testing.assert_array_equal(planner.last_validation['violations'], [0., 0.])
    else:
        assert planner.last_validation is None
    # The very same state still violates the original high-precision solve.
    _, inequalities = problem.constraint_components(position, quaternion, progress,
        q, ('joint',), q, trajectory_time=progress)
    assert inequalities['function_pose:0:position'].item() == pytest.approx(position_error-.002)
    assert inequalities['function_pose:0:rotation'].item() == pytest.approx(rotation_error-np.pi/180)
