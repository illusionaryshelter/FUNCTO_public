"""Counterexamples for one delivered curve and simultaneous physical rows."""
# BEGIN actual Hermite64, exact entrance factors, and a registered-solid solve.
import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize
import torch

from manipulation.native_collision import NativeCollisionModel
from manipulation.physical_curve import PhysicalCurve, PHYSICAL_FTOL, aggregate_violation
from manipulation.planning_problem import JointCoupling


def physical(*, q0=(0.,), dq0=(0.,), controls=3, knot_dt=1., times=None,
             position=(-1., 1.), couplings=(), stop_times=(), bounds=None):
    dof = len(q0)
    if bounds is None:
        bounds = {name: np.repeat(np.array(limits)[:, None], dof, axis=1)
                  for name, limits in [('position', position), ('velocity', (-2., 2.)),
                                       ('acceleration', (-10., 10.)), ('jerk', (-40., 40.))]}
    if times is None:
        times = np.linspace(0., (controls+2)*knot_dt, (controls+2)*4+1)
    return PhysicalCurve(np.asarray(q0), np.asarray(dq0), controls, knot_dt, times,
        tuple(f'joint{index}' for index in range(dof)), bounds, couplings=couplings,
        stop_times=stop_times, joint_tolerance=1e-5, velocity_tolerance=1e-4)


def test_actual_export_reconstruction_and_affine_map_have_the_same_boundary():
    current = physical(q0=(.35, -.2), dq0=(.13, -.07), knot_dt=.4)
    actions = np.array([[.7, -.4], [-.1, .3], [.4, .6]])
    curve = current.curve(actions)
    position, velocity = current.export_nodes(actions)
    exported = CubicHermiteSpline(current.times, position, velocity)
    np.testing.assert_array_equal(exported.c, curve.c)
    np.testing.assert_array_equal(position[0], current.q0)
    np.testing.assert_array_equal(velocity[0], current.dq0)
    for derivative in range(4):
        queries = np.linspace(0., current.duration, 53)
        weights, bias = current.sample_map(queries, derivative)
        np.testing.assert_allclose(weights@actions+bias, curve(queries, nu=derivative), atol=2e-12)
    for derivative, expected in [(0, current.q0), (1, current.dq0)]:
        weights, bias = current.sample_map(np.array([0.]), derivative)
        np.testing.assert_array_equal(weights, np.zeros_like(weights))
        np.testing.assert_array_equal(bias[0], expected)


def test_fixed_zero_entrance_is_factored_without_masking_positive_slack():
    current = physical(position=(0., 2.))
    actions = np.array([[.2], [.3], [.4]])
    result = current.constraints(actions)
    lower = result.names.index('position/joint0/lower')
    assert current.curve(actions)(0.)[0] == 0.
    assert result.slack[lower] > 0.
    assert np.linalg.norm(result.jacobian[lower]) > 0.
    assert current._groups[0]['order'][0, 0] == 2
    inward = physical(position=(0., 2.), dq0=(.1,))
    assert inward._groups[0]['order'][0, 0] == 1
    nearby = physical(position=(0., 2.), q0=(1e-12,))
    assert nearby._groups[0]['order'][0, 0] == 0
    assert nearby.constraints(actions).slack[lower] == 1e-12


def test_constructor_exposes_fixed_failures_without_dummy_actions():
    outside = physical(q0=(-.02,), position=(0., 2.))
    assert outside.fixed_violation_lower_bound >= .02
    outward = physical(position=(0., 2.), dq0=(-.2,))
    assert outward.fixed_violation_lower_bound > PHYSICAL_FTOL
    stopped = physical(dq0=(.1,), stop_times=(0.,))
    assert stopped.fixed_names == ('stop:0/joint0/upper',)
    assert stopped.fixed_slack[0] == pytest.approx(-.0999)
    assert stopped.fixed_violation_lower_bound >= .0999
    bounds = {name: np.array([[-value], [value]]) for name, value in
              [('position', 2.), ('velocity', 2.), ('acceleration', 10.), ('jerk', 40.)]}
    bounds['velocity'][0] = .1
    terminal = physical(bounds=bounds)
    assert 'velocity/joint0/lower' in terminal.fixed_names
    assert terminal.fixed_slack[terminal.fixed_names.index('velocity/joint0/lower')] == -.1


def test_continuous_bounds_include_both_acceleration_sides_and_segment_jerk():
    # This fixed grid crosses source knots, so the delivered Hermite curve is
    # only C1. Its left acceleration at a saved time must remain visible.
    current = physical(times=np.array([0., .9, 2.1, 4.9, 5.]), position=(-.2, .2))
    actions = np.array([[.7], [-.6], [.8]])
    curve = current.curve(actions)
    result = current.constraints(actions)
    acceleration = curve.derivative(2)
    limits = current.bounds['acceleration'][:, 0]
    endpoint_values = np.concatenate((acceleration.c[-1, :, 0],
        acceleration.c[0, :, 0]*np.diff(curve.x)+acceleration.c[-1, :, 0]))
    expected = max(0., float(np.max(limits[0]-endpoint_values)), float(np.max(endpoint_values-limits[1])))
    assert result.raw_residuals['continuous_acceleration'] == pytest.approx(expected)
    jerk = 6*curve.c[0, :, 0]
    assert result.raw_residuals['continuous_jerk'] == pytest.approx(max(0., float(np.max(np.abs(jerk)-40.))))
    assert result.raw_residuals['continuous_position'] > 0.


def test_continuous_coupling_and_actual_stop_use_the_original_bands():
    coupling = JointCoupling(('joint0', 'joint1'), np.array([[1., -1.]]), np.array([0.]), 'registered actuator')
    current = physical(q0=(0., 0.), dq0=(0., 0.), couplings=(coupling,), stop_times=(1.2, 1.2))
    actions = np.array([[.2, .2], [.4, .1], [.5, .5]])
    result = current.constraints(actions)
    assert result.raw_residuals['coupling:0'] > .05
    assert result.raw_residuals['stop:0'] > 0.
    assert len([name for name in result.names if name.startswith('coupling:')]) == 2
    assert len([name for name in result.names if name.startswith('stop:')]) == 4
    matching = np.repeat(np.array([[.2], [.3], [.4]]), 2, axis=1)
    assert current.constraints(matching).raw_residuals['coupling:0'] == 0.
    assert current.stop_times.tolist() == [1.2]


def test_canonical_jacobian_matches_smooth_direction_and_row_count_is_small():
    current = physical(q0=(.1, -.2), dq0=(.03, .05), position=(-.9, 1.1), stop_times=(1.37,))
    actions = np.array([[.41, -.07], [-.33, .56], [.81, .47]])
    direction = np.array([[.13, -.21], [.17, .11], [-.19, .23]])
    result = current.constraints(actions)
    assert result.slack.shape == (20,)
    assert result.jacobian.shape == (20, 6)
    epsilon = 1e-5
    plus, minus = current.constraints(actions+epsilon*direction), current.constraints(actions-epsilon*direction)
    np.testing.assert_allclose((plus.slack-minus.slack)/(2*epsilon), result.jacobian@direction.ravel(), atol=1e-7)
    assert result.names == plus.names == minus.names


def test_slsqp_joint_and_registered_box_constraints_share_the_same_curve():
    bounds = {name: np.array([[-value], [value]]) for name, value in
              [('position', .58), ('velocity', 3.), ('acceleration', 20.), ('jerk', 200.)]}
    bounds['position'][0] = 0.
    current = physical(q0=(.1,), controls=3, knot_dt=.4, bounds=bounds)
    robot = NativeCollisionModel(robot_model={
        'urdf_xml': '<robot name="slider"><link name="base"/><link name="moving"/>'
                    '<joint name="joint0" type="prismatic"><parent link="base"/><child link="moving"/>'
                    '<axis xyz="1 0 0"/><limit lower="0" upper="0.58" effort="10" velocity="3"/>'
                    '</joint></robot>',
        'geometries': [{'name': 'moving_box', 'frame': 'moving', 'kind': 'box', 'dimensions': [.1, .1, .1],
                        'T_frame_geometry': np.eye(4).tolist(), 'source': 'registered test solid'}]},
        joint_names=('joint0',), T_reference_base=np.eye(4), self_pairs=(), moving_owners=(),
        owners={'wall': [{'kind': 'cuboid', 'dimensions': [.1, .1, .1], 'pose': [0., 0., 0., 1., 0., 0., 0.]}]})
    weights, bias = current.sample_map(current.times)
    poses = torch.eye(4, dtype=torch.float64).repeat(1, len(current.times), 1, 1, 1)
    poses[..., 0, 3] = .65
    enabled = torch.ones((1, len(current.times), len(robot.pairs)), dtype=torch.bool)
    variable = torch.as_tensor(np.any(weights != 0., axis=1))[None, :, None].expand_as(enabled)
    group = robot.groups.index('collision/robot/wall')

    def collision(actions):
        tensor = torch.tensor(actions.reshape(3, 1), dtype=torch.float64, requires_grad=True)
        q = (torch.from_numpy(weights)@tensor+torch.from_numpy(bias))[None]
        values, valid = robot.signed(q, poses, enabled, variable,
            owner_sources=torch.full(poses.shape[:3], -1, dtype=torch.long, device=q.device))
        assert bool((values[..., 0][valid[..., 0]] >= 0.).all())
        slack = values[0, :, group, 1][valid[0, :, group, 1]].min()
        derivative = torch.autograd.grad(slack, tensor)[0]
        return float(slack.detach()), derivative.detach().numpy().ravel()

    def constraints(actions):
        result = current.constraints(actions)
        value, _ = collision(actions)
        return np.r_[result.slack, value]

    def jacobian(actions):
        result = current.constraints(actions)
        _, derivative = collision(actions)
        return np.vstack((result.jacobian, derivative))

    # The original guess violates both the registered obstacle and position
    # bounds. The objective lies past both; neither condition can be exchanged
    # for objective reduction because they are independent SLSQP inequalities.
    initial = np.full(3, .6)
    assert current.constraints(initial).raw_residuals['continuous_position'] > 0.
    assert collision(initial)[0] < 0.
    result = minimize(lambda x: .5*(x[-1]-1.2)**2, initial,
        jac=lambda x: np.array([0., 0., x[-1]-1.2]), method='SLSQP',
        constraints={'type': 'ineq', 'fun': constraints, 'jac': jacobian},
        options={'ftol': PHYSICAL_FTOL, 'maxiter': 100})
    assert result.success, result.message
    assert aggregate_violation(constraints(result.x)) <= PHYSICAL_FTOL
    assert current.constraints(result.x).violation <= PHYSICAL_FTOL
    assert current.samples(result.x, current.times)[-1, 0] == pytest.approx(.55, abs=2e-6)

# END these tests do not modify robot geometry or adopt task-specific limits.
