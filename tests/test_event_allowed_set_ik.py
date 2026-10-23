"""Analytic point-IK tests of the original event allowed sets."""

# BEGIN use real task declarations with a two-coordinate analytic FK fixture.
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning import PlanningBudget, _ik_manager_type
from manipulation.planning_problem import MotionPhase, PlanningProblem, PoseEventRequirement, PosePath


# BEGIN exercise the current native IK manager with an analytic CPU robot state.
@pytest.fixture
def manager_fixture(monkeypatch):
    """Bind the production event manager without trajectory or GPU adapters.

    The two named coordinates directly translate the tool in its x/y axes.
    Actual native configuration, state and cost collection objects preserve
    the manager's initialization and query contracts; only FK is analytic.
    """
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo._src.types.tool_pose import ToolPose
    # The manager's final stream barrier is unconditional in the installed
    # GPU library. This CPU-only fixture registers no stream-owning costs.
    monkeypatch.setattr('curobo._src.rollout.cost_manager.cost_manager_robot.synchronize_cuda_streams',
                        lambda events, device: None)

    @contextmanager
    def fixture(mode, *, task):
        device = DeviceCfg('cpu', dtype=torch.float64)
        configuration = _ik_manager_type(task, PlanningBudget(constraint_weight=777.),
            clock={'duration': 1.}, seed_parameter=torch.tensor([[.5]], dtype=torch.float64))
        config = configuration.create({'process_mode': mode}, device_cfg=device)
        manager = config.class_type(device)
        robot = SimpleNamespace(tool_frames=['tool'], total_spheres=0,
            get_self_collision_config=lambda: None)
        transition = SimpleNamespace(joint_names=task.joint_names,
            robot_model=robot, interpolation_steps=1)
        manager.initialize_from_config(config, transition)

        def kinematics(joints):
            q = joints.position
            position = torch.cat((q, torch.zeros_like(q[..., :1])), -1).unsqueeze(-2)
            quaternion = q.new_zeros((*q.shape[:2], 1, 4))
            quaternion[..., 0] = 1.
            jacobian = q.new_zeros((*q.shape[:2], 1, 6, 2))
            jacobian[..., 0, 0] = jacobian[..., 1, 1] = 1.
            return SimpleNamespace(tool_poses=ToolPose(['tool'], position, quaternion),
                tool_jacobians=jacobian)

        yield SimpleNamespace(problem=task, configuration=configuration,
            manager=manager, kinematics=kinematics)

    return fixture
# END event queries use the maintained point-IK manager directly.


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
    with manager_fixture("seed_optimizer", task=task) as fixture:
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
    with manager_fixture("seed_metrics", task=task) as fixture:
        fixture.configuration.bind_ik_goals((("pose_events", 0),))
        q = torch.tensor([[[.43, 0.]], [[.46, 0.]]], dtype=torch.float64)
        costs = fixture.manager.compute_costs(state_at(fixture, q))
        value = costs.values[costs.names.index("event_allowed_set")].flatten()/777.
        np.testing.assert_allclose(value.numpy(), [0., .01], atol=1e-15)

# END event allowed sets remain point proposals for constrained trajectory search.
