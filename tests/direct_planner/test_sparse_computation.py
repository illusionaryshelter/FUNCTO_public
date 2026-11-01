"""Original task/physical math versus direct sparse numerical producers.

These small CPU checks use maintained native Pinocchio/task fixtures. Dense
matrices exist only as independent test references; production producers fill
declared scalar slots and are tested without their old dense entry points.
"""

from dataclasses import replace
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import torch

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                 str(Path(__file__).resolve().parents[2] / 'tests'),
                 str(Path(__file__).resolve().parents[2] / 'tests/planning')]

from manipulation.direct_planner.differentiable_kinematics import DifferentiablePinKinematics
from manipulation.direct_planner.material_hull import material_hull_indices
from manipulation.preference_planner.joint_computation import JointComputation
from manipulation.unified_solver import TrajectoryComputation, task_jacobian
from manipulation.direct_planner.sparse_physical import SparsePhysicalRows
from manipulation.direct_planner.sparse_task import SparseTaskRows
from manipulation.direct_planner.support_linearization import linearized_robot_state, task_linearization
from manipulation.direct_planner.terminal_binding import bind_terminal_hold
from .test_support_linearization import curve_fixture, load_fixture

task_fixture = load_fixture('sparse_original_task_fixture', 'test_task_constraint_components.py')


def dense_reference(pattern, values):
    return csr_matrix((values, pattern.columns, pattern.indptr), shape=pattern.shape).toarray()


@pytest.mark.parametrize('hold_start', [.4, .8, 1.])
def test_continuous_physical_slots_keep_native_values_extrema_and_jacobian(hold_start, monkeypatch):
    original, controls = curve_fixture.computation()
    terminal = bind_terminal_hold(original.domain._curve_map, [0., hold_start, 1.] if hold_start < 1. else [0., 1.],
        [0., 1., 1.] if hold_start < 1. else [0., 1.], source='declared physical sparse fixture')
    reduced = terminal.reduce(controls+np.random.default_rng(41).normal(0., .15, controls.shape))
    full = terminal.expand(reduced)
    wanted_values, wanted_full_jacobian = original.domain.signed_constraints(full)
    wanted_jacobian = terminal.pullback(wanted_full_jacobian)
    rows = SparsePhysicalRows(original.domain, terminal)

    def dense_not_used(*args, **kwargs):
        raise AssertionError('Production sparse physical producer requested the dense Jacobian')

    with monkeypatch.context() as patch:
        patch.setattr(original.domain, 'signed_constraints', dense_not_used)
        patch.setattr(original.domain._curve_map, 'signed_constraints', dense_not_used)
        point = rows.evaluate(full, check_request=lambda: None)
    np.testing.assert_allclose(point.values, wanted_values, rtol=0., atol=2e-13)
    actual = dense_reference(rows.pattern, point.jacobian_values())
    np.testing.assert_allclose(actual, wanted_jacobian, rtol=2e-13, atol=2e-13)
    assert rows.pattern.nnz < np.prod(rows.pattern.shape)
    direction = np.random.default_rng(3).normal(size=reduced.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6
    plus = original.domain.signed_constraints(terminal.expand(reduced+epsilon*direction))[0]
    minus = original.domain.signed_constraints(terminal.expand(reduced-epsilon*direction))[0]
    np.testing.assert_allclose(actual @ direction.ravel(), (plus-minus)/(2*epsilon), rtol=2e-6, atol=2e-8)


def all_task_case(yaw_tolerance):
    original, controls = curve_fixture.computation()
    declarations = task_fixture.full_task()
    names = dict(zip(declarations.joint_names, original.joint_names))
    couplings = tuple(replace(value, joint_names=tuple(names[name] for name in value.joint_names))
                      for value in declarations.couplings)
    problem = replace(original.problem, mechanisms=declarations.mechanisms,
        supports=declarations.supports, interiors=declarations.interiors,
        pose_events=tuple(replace(value, yaw_tolerance_rad=yaw_tolerance) for value in declarations.pose_events),
        coordinate_events=declarations.coordinate_events, pose_relations=declarations.pose_relations,
        couplings=couplings, events=tuple(replace(event, progress=.5) for event in declarations.events))
    kinematics = DifferentiablePinKinematics(original.kinematics.model,
        frames=original.kinematics.frames, T_reference_base=problem.base_pose)
    base = TrajectoryComputation(problem=problem, scene=original.scene, kinematics=kinematics,
        domain=original.domain, joint_names=original.joint_names, parameters=original.parameters,
        progress=original.progress.numpy()[0], ordinary_indices=original.ordinary_indices,
        motion_configuration=original.motion_configuration, native_dt=original.native_dt,
        constraint_weight=original.constraint_weight, max_host_rss_bytes=original.max_host_rss_bytes)
    terminal = bind_terminal_hold(base.domain._curve_map, [0., .8, 1.], [0., 1., 1.],
                                  source='declared full-task sparse fixture')
    bound = JointComputation(base=base, terminal=terminal,
                             geometry=SimpleNamespace(rows=lambda value: value.reshape(-1)[:0]))
    bound.support_material_indices = tuple(material_hull_indices(value.body_points) for value in problem.supports)
    reduced = terminal.reduce(controls+np.random.default_rng(17).normal(0., .07, controls.shape))
    return bound, reduced


@pytest.mark.parametrize('yaw_tolerance', [0., .2])
def test_every_native_task_family_is_covered_by_declared_sparse_columns(yaw_tolerance):
    bound, initial = all_task_case(yaw_tolerance)
    controls = torch.tensor(initial, requires_grad=True)
    context, jacobians = linearized_robot_state(bound, *bound.states(controls)[:2])
    declarations = ({}, {})
    values, blocks = task_linearization(bound, controls, solver_rows_out=declarations,
        robot_state=context, jacobians=jacobians, support_material_indices=bound.support_material_indices)
    rows = SparseTaskRows(bound, values, blocks, declarations, {'robot_state': context})
    actual_values = rows.select_values(values).detach().numpy()
    actual = dense_reference(rows.pattern, rows.jacobian_values(values, controls, blocks,
                                                               check_request=lambda: None))
    reference_controls = torch.tensor(initial, requires_grad=True)
    reference_values = torch.cat(bound.task(reference_controls))[rows.source_indices]
    # BEGIN compare normalized rows to an independent transform of original errors.
    transformed, event_rows = [], []
    for ordinal, (value, key) in enumerate(zip(reference_values, rows.row_keys)):
        parts = key[2].split(':')
        if key[1] == 'inequality' and parts[0] == 'function_pose':
            requirement = bound.problem.pose_events[int(parts[1])]
            tolerance = requirement.component_tolerances[parts[2]]
            error = value+tolerance
            value = ((error/tolerance).square()-1. if parts[2] == 'position'
                     else torch.sin(error/2).square()/np.sin(tolerance/2)**2-1.)
            event_rows.append(ordinal)
        transformed.append(value)
    original_values = reference_values
    reference_values = torch.stack(transformed)
    unchanged = np.setdiff1d(np.arange(len(reference_values)), event_rows)
    np.testing.assert_array_equal(actual_values[unchanged], original_values.detach().numpy()[unchanged])
    # END all other generic task families retain their original signed values.
    expected = task_jacobian(reference_values, reference_controls,
                            check_request=lambda: None, admit_work=lambda: None)
    np.testing.assert_allclose(actual_values, reference_values.detach().numpy(), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert rows.pattern.nnz < np.prod(rows.pattern.shape)
    represented = {key[2].split(':')[0] for key in rows.row_keys}
    assert represented == {'pose_relation', 'mechanism', 'mechanism_limits', 'support',
                           'interior', 'function_pose', 'function_coordinate', 'coupling', 'stop'}
    assert rows.equalities == (1 if yaw_tolerance == 0. else 0)
    direction = np.random.default_rng(21).normal(size=initial.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6
    with torch.no_grad():
        plus = torch.cat(bound.task(torch.tensor(initial+epsilon*direction)))[rows.source_indices].numpy()
        minus = torch.cat(bound.task(torch.tensor(initial-epsilon*direction)))[rows.source_indices].numpy()
    # BEGIN finite differences use the same independent metric transformation.
    for ordinal in event_rows:
        parts = rows.row_keys[ordinal][2].split(':')
        tolerance = bound.problem.pose_events[int(parts[1])].component_tolerances[parts[2]]
        for values_ in (plus, minus):
            error = values_[ordinal]+tolerance
            values_[ordinal] = ((error/tolerance)**2-1. if parts[2] == 'position'
                else np.sin(error/2)**2/np.sin(tolerance/2)**2-1.)
    # END other rows continue to differentiate the original producer verbatim.
    np.testing.assert_allclose(actual @ direction.ravel(), (plus-minus)/(2*epsilon), rtol=2e-6, atol=3e-8)


def test_support_hull_keeps_source_points_and_matches_selected_original_rows():
    bound, initial = all_task_case(.2)
    original = bound.problem.supports[0]
    points = np.vstack([original.body_points, original.body_points.mean(axis=0), original.body_points[0]])
    support = replace(original, body_points=points)
    bound.problem.supports = (support,)
    bound.support_material_indices = (material_hull_indices(points),)
    source_copy = support.body_points.copy()
    controls = torch.tensor(initial, requires_grad=True)
    robot, jacobians = linearized_robot_state(bound, *bound.states(controls)[:2])
    q, dq, names, _, tool, twist = robot
    original_groups = bound.problem.constraint_components(*tool, bound.progress, q, names, dq,
        trajectory_time=bound.times, tool_linear_velocity=twist[0], tool_angular_velocity=twist[1])
    values, blocks = task_linearization(bound, controls, robot_state=robot, jacobians=jacobians,
                                       support_material_indices=bound.support_material_indices)
    expected_blocks, actual_blocks = [], []
    for name, (span, factor) in blocks[1].items():
        selected = original_groups[1][name].reshape(-1)[factor.source_rows]
        torch.testing.assert_close(values[1][span], selected.detach(), rtol=0., atol=0.)
        expected_blocks.append(selected)
        actual_blocks.append(np.stack([factor.row_values(row, np.arange(initial.size))
                                       for row in range(len(factor.source_rows))]))
    expected = task_jacobian(torch.cat(expected_blocks), controls,
                            check_request=lambda: None, admit_work=lambda: None)
    np.testing.assert_allclose(np.vstack(actual_blocks), expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(support.body_points, source_copy)
    assert len(bound.support_material_indices[0]) < len(points)


def test_direct_point_uses_one_native_batch_and_releases_graph_after_derivatives(monkeypatch):
    from .test_direct_integration import robot_case

    planner, initial = robot_case()
    computation = planner.computation
    native, calls = computation.kinematics._evaluate, []

    def evaluate(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(computation.kinematics, '_evaluate', evaluate)
    bound = computation.bind_solver(initial, check_request=lambda: None)
    point = bound.evaluate(initial)
    reference = weakref.ref(point.controls)
    pose_reference = weakref.ref(next(iter(point.context['robot_state'][3].values()))[0])
    assert len(calls) == 1
    gradient = point.objective_gradient().copy()
    sparse_values = point.jacobian_values().copy()
    assert len(calls) == 1
    np.testing.assert_array_equal(point.jacobian_values(), sparse_values)
    gc.collect()
    assert reference() is None and pose_reference() is None
    original_controls = torch.tensor(initial, requires_grad=True)
    original = sum(computation.objective_components(original_controls).values())
    original_gradient, = torch.autograd.grad(original, original_controls)
    assert point.objective_value == pytest.approx(float(original.detach()), rel=0., abs=0.)
    np.testing.assert_allclose(gradient, original_gradient.numpy().ravel(), rtol=2e-13, atol=2e-13)
    assert not computation.trial_history


def test_real_direct_computation_advances_original_infeasible_initial_conditions():
    from manipulation.direct_planner.shared_solver import solve_trajectory
    from .test_direct_integration import robot_case

    planner, initial = robot_case()
    starts, report = [], {}
    result = solve_trajectory(planner.computation.bind_solver(initial, check_request=lambda: None), initial, iterations=2,
        check_request=lambda: None, remaining_seconds=lambda: 5.,
        on_solver_start=lambda: starts.append(True), pending_geometry_bytes=0, report_out=report)
    assert starts == [True] and report['solver_called']
    assert 0 < report['iterations'] <= 2
    assert not report['unchanged_from_seed']
    assert np.isfinite(result.controls).all()
    assert report['jacobian_callbacks'] > 0
    assert report['producer_evaluations']['shared_state'] == report['point_evaluations']
    assert set(report['latest_complete_trial']['objective_components']) == {
        'motion_quality', 'task_preferences', 'pose_prior', 'scene_relations', 'reference_prior',
        'tool_angular_motion'}
    # BEGIN each component uses the same counted objective production boundary.
    for name in ('objective_base', 'scene_relations', 'reference_prior', 'tool_angular_motion'):
        assert report['producer_evaluations'][name] == report['point_evaluations']
    # END child timings remain within the complete objective total.


# BEGIN the registered scene, complete objective and sparse rows share one solve.
def observed_direct_case(tmp_path):
    """Bind the maintained analytic observed scene to the original task APIs.

    The scene fixture retains its registered robot boxes and TSDF samples.
    Explicit task data supplies a two-joint coupling, terminal stop and pose
    prior/reference. This is a small numerical seam, independent of any live
    scene or simulation rollout; original entity qualification is separate.
    """
    from manipulation.direct_planner.direct_computation import DirectComputation
    from manipulation.direct_planner.geometry_constraints import GeometryConstraints
    from manipulation.physical_joint_domain import CubicJointDomain
    from manipulation.planning import _ProgressMap
    from manipulation.planning_problem import (
        JointCoupling, MotionPhase, PlanningEvent, PlanningProblem,
        PoseEventRequirement, PosePath,
    )
    from manipulation.preference_planner.pose_roles import bind_pose_objectives
    from manipulation.timed_reference import BoundToolReference
    from manipulation.direct_planner.reference_objective import ReferenceObjective
    from .test_geometry_constraints import computation as observed_computation

    original, controls = observed_computation(tmp_path)
    curve = original.domain._curve_map
    entrance = np.eye(4)
    entrance[:3, 3] = [curve.start_q[0], .2, 0.]
    target = entrance.copy()
    target[0, 3] += .02
    path = PosePath([0., 1.], np.stack((entrance, target)), 'declared analytic reference')
    problem = PlanningProblem(original.joint_names, curve.start_q, curve.start_dq,
        'left', 'world', np.eye(4), np.eye(4), (MotionPhase('reference', path),),
        pose_events=(PoseEventRequirement(1., np.eye(4), target, .005, .02,
                                         'declared analytic pose prior'),),
        couplings=(JointCoupling(original.joint_names, [[.5, 1.]], [0.],
                                 'declared analytic joint coupling'),),
        events=(PlanningEvent('stop', 'instance', 1., 1, 'hold',
                              'declared analytic terminal stop', stop=True),))
    problem = bind_pose_objectives(problem, [dict(goal_id=['pose_events', 0], role='prior',
        scales=dict(position=.02, rotation=.1), scale_source='declared analytic preference')])
    bounds = np.array([[[0., -.1], [.1, 0.]], [[-1., -1.], [1., 1.]],
                       [[-4., -4.], [4., 4.]], [[-16., -16.], [16., 16.]]])
    domain = CubicJointDomain(torch.tensor(bounds), torch.tensor(curve.start_q),
                             torch.tensor(curve.start_dq), curve.knot_dt, curve.knots)
    base = TrajectoryComputation(problem=problem, scene=original.scene,
        kinematics=original.kinematics, domain=domain, joint_names=original.joint_names,
        parameters=original.parameters, progress=original.parameters,
        ordinary_indices=np.arange(len(original.parameters)),
        motion_configuration=curve_fixture.motion_configuration(), native_dt=.1,
        constraint_weight=10., max_host_rss_bytes=original.max_host_rss_bytes)
    mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    terminal = bind_terminal_hold(domain._curve_map, mapping.u, mapping.s,
                                  source='declared analytic physical clock')
    geometry = GeometryConstraints(base, progress_map=mapping)
    reference = BoundToolReference(problem, mapping.u, mapping.s, base.duration,
                                   entrance, np.zeros(6))
    computation = DirectComputation(base=base, terminal=terminal, geometry=geometry,
        reference_objective=ReferenceObjective(.03, .2, .4, 0., 1.).bind(reference, base.times[0]))
    return computation, terminal.reduce(controls)


def test_observed_scene_direct_point_and_ipopt_share_complete_sparse_problem(tmp_path, monkeypatch):
    """Real scene arrays and original objective enter one counted IPOPT call."""
    from manipulation.online_geometry import OnlinePlanningScene
    from manipulation.direct_planner.shared_solver import solve_trajectory

    from manipulation.direct_planner import geometry_constraints as geometry_module
    from manipulation.direct_planner.batch_distance import BatchDistance
    monkeypatch.setattr(geometry_module, 'BatchDistance',
        lambda geometries, *, query_sources: BatchDistance(geometries,
            query_sources=query_sources, device='cpu'))
    computation, initial = observed_direct_case(tmp_path)
    computation.scene.check_request = lambda: None
    assert isinstance(computation.scene, OnlinePlanningScene)
    native, calls = computation.kinematics._evaluate, []

    def evaluate(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(computation.kinematics, '_evaluate', evaluate)
    bound = computation.bind_solver(initial, check_request=lambda: None)
    point = bound.evaluate(initial)
    controls_ref = weakref.ref(point.controls)
    pose_ref = weakref.ref(point.context['robot_state'][3]['left'][0])
    assert len(calls) == 1
    assert computation.geometry.last_evaluation['query_state_count'] == len(computation.parameters)
    assert computation.geometry.last_evaluation['active_count'] > 0
    assert len(bound.task.row_keys) and len(bound.geometry.row_keys) and len(bound.physical.row_keys)
    assert isinstance(point.geometry.values, np.ndarray)
    assert isinstance(point.geometry.factors, np.ndarray)
    assert point.geometry.factors.shape[1:] == (2, len(computation.joint_names))
    geometry_slice = slice(len(bound.task.row_keys), len(bound.task.row_keys)+len(bound.geometry_envelope.row_keys))
    expected_maxima = np.array([point.geometry.values[rows].max() for rows in bound.geometry_envelope.rows])
    np.testing.assert_array_equal(point.constraint_values[geometry_slice], expected_maxima)
    assert np.all(expected_maxima <= 0.) == np.all(point.geometry.values <= 0.)
    assert np.isfinite(point.objective_gradient()).all()
    assert np.isfinite(point.jacobian_values()).all()
    assert len(calls) == 1
    gc.collect()
    assert controls_ref() is None and pose_ref() is None
    expected_components = {'motion_quality', 'task_preferences', 'pose_prior',
                           'scene_relations', 'reference_prior', 'tool_angular_motion'}
    assert set(point.objective_components) == expected_components
    # BEGIN absent declared relations have zero cost; collision rows remain.
    assert point.objective_components['scene_relations'] == 0.
    assert np.any(point.geometry.values > 0.)
    # END the original material inequality provides the collision evidence.

    starts, report = [], {}
    result = solve_trajectory(computation.bind_solver(initial, check_request=lambda: None), initial, iterations=1,
        check_request=lambda: None, remaining_seconds=lambda: 5.,
        on_solver_start=lambda: starts.append(True), pending_geometry_bytes=0, report_out=report)
    assert starts == [True] and report['solver_called']
    assert report['iterations'] == 1
    assert not report['unchanged_from_seed']
    assert report['solver_jacobian_shape'] == list(bound.pattern.shape)
    assert report['solver_geometry_inequalities'] == len(bound.geometry_envelope.row_keys)
    assert report['original_geometry_inequalities'] == len(bound.geometry.row_keys)
    assert report['solver_physical_inequalities'] == len(bound.physical.row_keys)
    assert report['value_callbacks'] > 0 and report['jacobian_callbacks'] > 0
    assert report['producer_evaluations']['shared_state'] == report['point_evaluations']
    assert report['producer_evaluations']['geometry'] == report['point_evaluations']
    assert len(calls) == 1+report['point_evaluations']
    assert set(report['latest_complete_trial']['objective_components']) == expected_components
    assert np.isfinite(result.controls).all()
# END original scene and task data retain their own units and numerical outcome.
