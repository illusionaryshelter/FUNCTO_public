"""Analytic point-IK tests of the actual task manager and event graph."""

# BEGIN use real task declarations with a two-coordinate analytic FK fixture.
import numpy as np
import pytest
import torch

from manipulation.planning_problem import MotionPhase, PlanningProblem, PoseEventRequirement, PosePath
from manipulation.planning_ik import select_ik_chains
from test_scene_query_performance import manager_fixture


def event_problem():
    target = np.eye(4)
    target[0, 3] = .4
    reference = PosePath(np.array([0., 1.]), np.stack((np.eye(4), target)), "analytic reference")
    goal = PoseEventRequirement(.5, np.eye(4), target, .05, .02, "declared allowed set",
                                yaw_axis=[0., 0., 1.], yaw_tolerance_rad=np.pi)
    return PlanningProblem(("first_joint", "second_joint"), np.zeros(2), np.zeros(2),
        "tool", "world", np.eye(4), np.eye(4), (MotionPhase("source", reference),),
        pose_events=(goal,))


def state_at(fixture, q):
    from curobo.types import JointState
    from curobo._src.state.state_robot import RobotState
    joints = JointState.from_position(q, joint_names=list(fixture.problem.joint_names))
    joints.velocity = torch.zeros_like(q)
    joints.dt = torch.ones(1, dtype=q.dtype)
    return RobotState(joint_state=joints, cuda_robot_model_state=fixture.kinematics(joints))


def test_event_manager_drives_toward_allowed_set_and_has_zero_cost_inside(manager_fixture):
    task = event_problem()
    with manager_fixture("seed_optimizer", required=(), capacity=4, task=task) as fixture:
        fixture.configuration.bind_ik_goals((("pose_events", 0),))
        q = torch.tensor([[[.1, 0.]]], dtype=torch.float64, requires_grad=True)
        costs = fixture.manager.compute_costs(state_at(fixture, q))
        value = costs.values[costs.names.index("event_allowed_set")]
        gradient = torch.autograd.grad(value.sum(), q)[0]
        # 0.30 m error minus a 0.05 m allowed radius gives normalized 5.
        assert value.item() == pytest.approx(777.*25.)
        assert gradient[0, 0, 0] == pytest.approx(-777.*2.*5./.05)
        inside = torch.tensor([[[.43, 0.]]], dtype=torch.float64, requires_grad=True)
        costs = fixture.manager.compute_costs(state_at(fixture, inside))
        value = costs.values[costs.names.index("event_allowed_set")]
        assert value.item() == 0
        assert torch.autograd.grad(value.sum(), inside)[0].count_nonzero() == 0
        fixture.configuration.bind_ik_goals(())
        assert "event_allowed_set" not in fixture.manager.compute_costs(state_at(fixture, q)).names


def test_event_metrics_and_projection_share_the_allowed_set(manager_fixture):
    task = event_problem()
    with manager_fixture("seed_metrics", required=(), capacity=4, task=task) as fixture:
        fixture.configuration.bind_ik_goals((("pose_events", 0),))
        q = torch.tensor([[[.43, 0.]], [[.46, 0.]]], dtype=torch.float64)
        costs = fixture.manager.compute_costs(state_at(fixture, q))
        value = costs.values[costs.names.index("event_allowed_set")].flatten()/777.
        np.testing.assert_allclose(value.numpy(), [0., .01], atol=1e-15)


def candidate(**changes):
    return dict(task_satisfied=True, model_valid=True, native_pose_ik_success=False,
                candidate_source="event_allowed_set_ik", **changes)


def test_event_graph_applies_signed_speed_bounds_and_requires_every_event():
    reports = [[candidate(), candidate()], [candidate()]]
    # The close first state -.1 is unreachable with a nonnegative joint rate;
    # .2 is reachable, and its connection to .5 remains within +.4 per second.
    result = select_ik_chains([[[-.1], [.2]], [[.5]]], reports, [[0.]], [1., 2.],
        [[0.], [.4]], count=2, joint_tolerance=1e-8, query_kind="functional")
    assert len(result) == 1 and result[0]["candidate_indices"] == [1, 0]
    reports[1][0]["task_satisfied"] = False
    assert select_ik_chains([[[-.1], [.2]], [[.5]]], reports, [[0.]], [1., 2.],
        [[0.], [.4]], count=2, joint_tolerance=1e-8, query_kind="functional") == []
# END these tests establish point computation and connection necessity, not path acceptance.
