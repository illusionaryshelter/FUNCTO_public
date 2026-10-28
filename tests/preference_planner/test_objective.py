"""Analytical checks of the shared trajectory/event-IK pose objective.

Real PlanningProblem, pose geometry and role contracts own the calculation.
The synthetic scene cost below is explicit algebra, not collision evidence.
These CPU tests establish the event measure, units, scales and derivatives;
they do not establish scene feasibility or physical success.
"""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]


from manipulation.planning_problem import (
    MotionPhase, PlanningProblem, PoseEventRequirement, PosePath, PoseProcessRequirement,
)
from manipulation.preference_planner.objective import pose_prior_cost, trajectory_objective_components
from manipulation.preference_planner.pose_roles import bind_pose_objectives


def analytical_problem(*, roles=('prior', 'required'), scales=None, axis=None,
                       yaw_tolerance=None, process=False):
    """Declare a fixed target and independent process relation in metres."""
    path = PosePath(np.array([0., 1.]), np.stack([np.eye(4), np.eye(4)]),
                    source='analytical constant reference')
    events = tuple(PoseEventRequirement(progress=progress, tool_from_function=np.eye(4),
        target=np.eye(4), position_tolerance_m=.002, rotation_tolerance_rad=.03,
        source='analytical target', yaw_axis=axis, yaw_tolerance_rad=yaw_tolerance)
        for progress in (.5, .8)[:len(roles)])
    bounds = np.tile([-np.inf, np.inf], (6, 1))
    bounds[0] = [.01, .01]
    problem = PlanningProblem(joint_names=('joint',), start_q=np.zeros(1), start_dq=np.ones(1),
        tool_link='tool', reference_frame='reference', base_pose=np.eye(4), tool_transform=np.eye(4),
        phases=(MotionPhase('analytical interval', path),), pose_events=events,
        pose_relations=((PoseProcessRequirement(0., 1., path, bounds,
                            source='explicit x process equation'),) if process else ()))
    scales = {'position': .01, 'rotation': .2} if scales is None else scales
    return bind_pose_objectives(problem, [dict(goal_id=['pose_events', index], role=role,
        scales=dict(scales), scale_source='explicit analytical scale')
        for index, role in enumerate(roles)])


def identity_quaternions(position):
    """Constant unit wxyz rotations with the same batch/time dimensions."""
    quaternions = torch.zeros((*position.shape[:-1], 4), dtype=position.dtype)
    quaternions[..., 0] = 1.
    return quaternions


def test_prior_occurs_once_on_plateau_and_required_event_has_no_center_cost():
    problem = analytical_problem()
    position = torch.tensor([[[0., 0., 0.], [.02, 0., 0.], [.3, 0., 0.],
                              [.4, 0., 0.], [.5, 0., 0.]]], dtype=torch.float64)
    progress = torch.tensor([[0., .5, .5, .8, 1.]], dtype=torch.float64)
    times = torch.tensor([[0., .2, .4, .8, 1.]], dtype=torch.float64)
    value = pose_prior_cost(problem, position, identity_quaternions(position), progress, times)
    torch.testing.assert_close(value, torch.tensor([[0., 4., 0., 0., 0.]], dtype=torch.float64))
    excluded = pose_prior_cost(problem, position, identity_quaternions(position), progress, times,
                               goal_ids=(('pose_events', 1),))
    assert not torch.count_nonzero(excluded)


def test_ik_batches_use_same_event_cost_and_position_derivative():
    problem = analytical_problem(roles=('prior',), axis=[0., 0., 1.], yaw_tolerance=np.pi)
    position = torch.tensor([[[.02, 0., 0.]], [[.04, 0., 0.]]], dtype=torch.float64,
                            requires_grad=True)
    angle = torch.tensor([.4, -.7], dtype=torch.float64, requires_grad=True)
    quaternion = torch.stack((torch.cos(angle/2), angle*0., angle*0., torch.sin(angle/2)), -1)[:, None]
    progress = torch.full((2, 1), .5, dtype=torch.float64)
    value = pose_prior_cost(problem, position, quaternion, progress, torch.zeros_like(progress),
                            goal_ids=(('pose_events', 0),))
    torch.testing.assert_close(value, torch.tensor([[4.], [16.]], dtype=torch.float64))
    dp, da = torch.autograd.grad(value.sum(), (position, angle))
    torch.testing.assert_close(dp[..., 0], torch.tensor([[400.], [800.]], dtype=torch.float64))
    torch.testing.assert_close(da, torch.zeros_like(da), atol=1e-25, rtol=0.)


def test_explicit_prior_scale_is_independent_of_original_hard_threshold():
    problem = analytical_problem(roles=('prior',))
    changed = replace(problem, pose_events=(replace(problem.pose_events[0],
        position_tolerance_m=.08, rotation_tolerance_rad=.5),))
    rescaled = analytical_problem(roles=('prior',), scales={'position': .02, 'rotation': .2})
    position = torch.tensor([[[.03, 0., 0.]]], dtype=torch.float64)
    progress = torch.tensor([[.5]], dtype=torch.float64)
    costs = [pose_prior_cost(p, position, identity_quaternions(position), progress,
                             torch.zeros_like(progress)).item() for p in (problem, changed, rescaled)]
    assert costs == pytest.approx([9., 9., 2.25])


@pytest.mark.parametrize('yaw_tolerance', [None, .3, np.pi])
def test_product_prior_position_swing_and_finite_twist_gradients(yaw_tolerance):
    scales = {'position': .03, 'rotation': .4}
    if yaw_tolerance == .3:
        scales['yaw'] = .5
    problem = analytical_problem(roles=('prior',), scales=scales,
        axis=None if yaw_tolerance is None else [0., 0., 1.], yaw_tolerance=yaw_tolerance)
    value = torch.tensor([.02, -.03, .01, .9, .13, -.19, .24],
                         dtype=torch.float64, requires_grad=True)
    progress = torch.tensor([[.5]], dtype=torch.float64)

    def cost(state):
        quaternion = state[3:] / torch.linalg.vector_norm(state[3:])
        return pose_prior_cost(problem, state[:3][None, None], quaternion[None, None],
                               progress, torch.zeros_like(progress)).sum()

    assert torch.autograd.gradcheck(cost, (value,), eps=1e-6, atol=2e-8, rtol=2e-6)


class AlgebraicScene:
    """Explicit y² test cost that records the supplied complete robot layout."""

    def optimization_objective(self, position, quaternion, progress, q, names, *,
                               trajectory_time, robot_link_poses, report_out=None):
        self.received = (progress, q, names, trajectory_time, robot_link_poses)
        if report_out is not None:
            report_out['source'] = 'analytical y squared; no geometry query'
        return position[..., 1].square().sum()


def test_joint_objective_preserves_process_motion_and_geometry_units():
    problem = analytical_problem(process=True)
    position = torch.tensor([[[.01, .03, 0.], [.02, .03, 0.], [.3, .03, 0.],
                              [.04, .03, 0.], [.05, .03, 0.]]], dtype=torch.float64,
                            requires_grad=True)
    progress = torch.tensor([[0., .5, .5, .8, 1.]], dtype=torch.float64)
    times = torch.tensor([[0., .2, .4, .8, 1.]], dtype=torch.float64)
    q = position[..., :1]*0.
    dq = torch.tensor([[[0.], [40.], [1.], [80.], [2.]]], dtype=torch.float64)
    states = (q, dq, q*0., q*0.)
    poses, scene, report = {'tool': 'analytical frame'}, AlgebraicScene(), {}
    names = ('joint',)
    arguments = dict(problem=problem, scene=scene, states=states,
        robot_state=(q, dq, names, poses, (position, identity_quaternions(position)),
                     (position*0., position*0.)), progress=progress, times=times,
        ordinary_indices=np.array([0, 2, 4]),
        bounds=position.new_tensor([[[-100.], [100.]]] * 4),
        motion_configuration=dict(weight=[0.] * 4, squared_l2_regularization_weight=[2., 0., 0.],
            retime_weights=False, retime_regularization_weights=False, activation_distance=[0.] * 4),
        native_dt=.1, constraint_weight=3., geometry_report=report)
    costs = trajectory_objective_components(**arguments)
    # Nonuniform ordinary times 0,.4,1 give weights .2,.5,.3. Extra event
    # velocities 40 and 80 do not change this declared motion integral.
    assert {key: value.item() for key, value in costs.items()} == pytest.approx(dict(
        motion_quality=1.7, task_preferences=3*(.01**2 + .29**2 + .03**2 + .04**2),
        pose_prior=3*(.02**2 + .03**2)/.01**2, observed_geometry=3*5*.03**2))
    gradient, = torch.autograd.grad(sum(costs.values()), position)
    assert torch.isfinite(gradient).all()
    assert gradient[0, 1, 0].item() == pytest.approx(2*3*(.02-.01) + 2*3*.02/.01**2)
    assert gradient[0, 2, 0].item() == pytest.approx(2*3*(.3-.01))
    assert scene.received[0] is progress and scene.received[1] is q
    assert scene.received[2] is names and scene.received[3] is times and scene.received[4] is poses
    assert report['source'] == 'analytical y squared; no geometry query'
    without_geometry = trajectory_objective_components(**arguments, include_geometry=False)
    assert set(without_geometry) == {'motion_quality', 'task_preferences', 'pose_prior'}
