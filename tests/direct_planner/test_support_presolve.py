"""Generic geometry, event-query and scoring checks for support presolve."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation.direct_planner.direct_computation import ComputationPoint
from manipulation.direct_planner.sparse_task import SparseTaskRows
from manipulation.direct_planner.support_linearization import linearized_robot_state, task_linearization
from manipulation.direct_planner.support_presolve import implied_support_rows, support_halfspace_bounds
from manipulation.goals import SupportSurface
from manipulation.planning_problem import PoseEventRequirement, SupportRequirement
from manipulation.preference_planner.pose_roles import bind_pose_objectives
from .test_support_linearization import support_case


def declarations():
    """Different carrier/functional rotations and a slightly nonunit normal."""
    surface = SupportSurface([.2, -.1, .03], np.eye(3)[:, :2], [0., 0., 1.+2e-6],
        [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]], 'declared finite surface')
    carrier, functional, target = np.eye(4), np.eye(4), np.eye(4)
    carrier[:3, :3] = Rotation.from_rotvec([.3, -.1, .2]).as_matrix()
    carrier[:3, 3] = [.12, -.03, .07]
    functional[:3, :3] = Rotation.from_rotvec([-.2, .4, .1]).as_matrix()
    functional[:3, 3] = [.08, .01, .03]
    target[:3, 3] = [.2, -.1, .6]
    points = np.array([[-.04, -.02, -.01], [.03, .04, .02], [.02, -.03, .04]])
    support = SupportRequirement(.5, carrier, points, points[:1], [0., 0., -1.],
        surface, .01, .2, .002, 'declared rigid material')
    event = PoseEventRequirement(.5, functional, target, .01, .03,
        'declared functional event', yaw_axis=[0., 0., 1.], yaw_tolerance_rad=np.pi)
    return support, event, SimpleNamespace(position_m=1e-6)


def test_bound_encloses_rotated_material_under_free_yaw_and_frame_offsets():
    support, event, numerical = declarations()
    indices = np.arange(len(support.body_points))
    upper = support_halfspace_bounds(support, event, indices, numerical)
    # The actual producer is independent of the bounding calculation. Pose
    # samples span arbitrary rotations, not only the declared yaw freedom.
    rng = np.random.default_rng(21)
    rotation = Rotation.random(12, random_state=rng)
    displacement = rng.normal(size=(12, 3))
    displacement *= event.position_tolerance_m / np.linalg.norm(displacement, axis=1)[:, None]
    position = event.target[:3, 3]+displacement-rotation.apply(event.tool_from_function[:3, 3])
    values = support.inequality_components(torch.tensor(position),
        torch.tensor(rotation.as_quat()[:, [3, 0, 1, 2]]), numerical)
    for name in ('body', 'footprint'):
        assert np.all(values[name].numpy().reshape(12, -1) <= upper[name][None])
    assert np.all(upper['body'] < 0.) and np.all(upper['footprint'] < 0.)
    # Functional orientation cancels algebraically from the position-ball proof.
    other = event.tool_from_function.copy()
    other[:3, :3] = Rotation.from_rotvec([.7, .3, -.9]).as_matrix()
    changed = support_halfspace_bounds(support, replace(event, tool_from_function=other), indices, numerical)
    for name in upper:
        np.testing.assert_array_equal(changed[name], upper[name])


def test_tangent_halfspace_is_retained_by_outward_bound():
    support, event, _ = declarations()
    support.tool_from_carrier = np.eye(4)
    support.body_points = np.zeros((1, 3))
    support.surface.normal = np.array([0., 0., 1.])
    support.surface.origin = np.zeros(3)
    target = np.eye(4)
    target[2, 3] = .125
    event = replace(event, target=target, tool_from_function=np.eye(4), position_tolerance_m=.125)
    bound = support_halfspace_bounds(support, event, np.array([0]), SimpleNamespace(position_m=0.))
    assert bound['body'][0] >= 0.


@pytest.mark.parametrize('role,expected_removed', [('required', True), ('prior', False)])
def test_real_sparse_binding_keeps_full_initial_scores_and_row_identities(role, expected_removed):
    """A ball-outside initial point never becomes a false fixed-row failure."""
    bound, initial = support_case()
    support = bound.problem.supports[0]
    target = np.eye(4)
    target[:3, 3] = support.surface.origin + .35*support.surface.normal
    event = replace(bound.problem.pose_events[0], progress=support.progress,
        target=target, tool_from_function=np.eye(4), yaw_axis=None, yaw_tolerance_rad=None)
    problem = replace(bound.problem, pose_events=(event,))
    bound.problem = bind_pose_objectives(problem, [dict(goal_id=['pose_events', 0], role=role,
        scales=dict(position=.005, rotation=.02), scale_source='declared test metric')])
    controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    robot, jacobians = linearized_robot_state(bound, *bound.states(controls)[:2])
    masks = ({}, {})
    values, blocks = task_linearization(bound, controls, solver_rows_out=masks,
        robot_state=robot, jacobians=jacobians, support_material_indices=bound.support_material_indices)
    rows = SparseTaskRows(bound, values, blocks, masks, {'robot_state': robot})
    assert bool(rows.report['support_presolve']['removed_task_rows']) == expected_removed
    original = SparseTaskRows(bound, values, blocks, masks, {'robot_state': robot}, support_presolve=False)
    np.testing.assert_array_equal(rows.scoring_values(values).detach(), original.select_values(values).detach())
    assert rows.fixed_constraints == original.fixed_constraints
    assert rows.fixed_feasible == original.fixed_feasible
    assert rows.row_keys == tuple(rows.scoring_row_keys[row] for row in rows.solver_selection)
    for proof in rows.report['support_presolve']['certificates']:
        assert proof['maximum_upper_bound_m'] < 0.
        assert proof['position_condition'] == 'function_pose:0:position'


def test_same_physical_query_uses_original_first_arrival_and_not_progress_proximity():
    support, event, numerical = declarations()
    # A plateau has two physical times; duplicate queries select its first
    # physical occurrence. An event at another progress is a different query.
    computation = SimpleNamespace(problem=SimpleNamespace(supports=(support,),
        pose_events=(event,), numerical=numerical),
        progress=torch.tensor([[0., .5, .5, .75, .5]], dtype=torch.float64),
        times=torch.tensor([[0., 1., 2., 3., 1.]], dtype=torch.float64),
        support_material_indices=(np.arange(len(support.body_points)),))
    keys = (('task', 'inequality', 'function_pose:0:position', 0),
            ('task', 'inequality', 'support:0:body', 0))
    factor = SimpleNamespace(source_rows=np.arange(len(support.body_points)))
    blocks = ({}, {'support:0:body': (slice(0, 3), factor)})
    removed, report = implied_support_rows(computation, keys, {1: (1, 'support:0:body', 0)}, blocks)
    assert removed.tolist() == [False, True]
    assert report['certificates'][0]['query_column'] == 1
    computation.problem.pose_events = (replace(event, progress=.75),)
    removed, _ = implied_support_rows(computation, keys, {1: (1, 'support:0:body', 0)}, blocks)
    assert not removed.any()


def test_initializer_uses_complete_violations_outside_retained_position_ball():
    """Actual score API retains the exact counterexample's original winner."""
    scores = []
    for x, objective in ((2., 0.), (-2., 1.)):
        point = ComputationPoint.__new__(ComputationPoint)
        point.bound = SimpleNamespace(task=SimpleNamespace(equalities=0),
            scoring_row_keys=('ball', 'implied_halfspace'))
        point.scoring_constraint_values = np.array([x*x-1., x-1.5])
        point.constraint_values = point.scoring_constraint_values[:1]
        point.objective_value, point.objective_components = objective, {'declared': objective}
        scores.append(point.initialization_score())
    assert [score['total_positive_violation'] for score in scores] == [3.5, 3.]
    assert min(range(2), key=lambda i: (scores[i]['total_positive_violation'], scores[i]['objective'])) == 1
