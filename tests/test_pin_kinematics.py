"""CPU kinematic derivatives use the same complete model as robot validation."""

# BEGIN nonzero calibration, rotated frames and independent fingers expose VJP signs.
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation.pin_kinematics import PinKinematics
from manipulation.planning_problem import quaternion_multiply, quaternion_rotate
from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker


URDF = """<robot name="kinematic_derivative_fixture">
<link name="base"/><link name="arm"/><link name="wrist"/>
<link name="left"/><link name="right"/><link name="tool"/>
<joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/>
<origin xyz="0.1 0.2 0.3" rpy="0.1 0.2 0.1"/><axis xyz="0 0 1"/>
<limit lower="-2" upper="2" effort="10" velocity="2"/></joint>
<joint name="elbow" type="revolute"><parent link="arm"/><child link="wrist"/>
<origin xyz="0.4 0 0.1" rpy="0.2 -0.1 0.3"/><axis xyz="0 1 0"/>
<limit lower="-2" upper="2" effort="10" velocity="2"/></joint>
<joint name="left_slide" type="prismatic"><parent link="wrist"/><child link="left"/>
<origin xyz="0.1 0.05 0"/><axis xyz="0 1 0"/>
<limit lower="0" upper="0.1" effort="10" velocity="1"/></joint>
<joint name="right_slide" type="prismatic"><parent link="wrist"/><child link="right"/>
<origin xyz="0.1 -0.05 0"/><axis xyz="0 1 0"/>
<limit lower="-0.1" upper="0" effort="10" velocity="1"/></joint>
<joint name="tool_fixed" type="fixed"><parent link="wrist"/><child link="tool"/>
<origin xyz="0.05 0.1 0.2" rpy="0.1 0.2 0.3"/></joint>
</robot>"""


@pytest.fixture
def robot(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF)
    checker = RobotGeometryChecker(path, base_frame="base", geometries=[NativePrimitive(
        "base_fixture", "base", "box", (.1, .1, .1), np.eye(4), "synthetic robot test fixture")])
    calibration = np.eye(4)
    calibration[:3, :3] = Rotation.from_rotvec([.4, -.2, .3]).as_matrix()
    calibration[:3, 3] = [1.2, -.5, .3]
    bridge = PinKinematics(checker.model, frames=("base", "wrist", "tool", "left", "right"),
                           T_reference_base=calibration)
    return checker, bridge, calibration


def sample():
    names = ("right_slide", "elbow", "shoulder", "left_slide")
    position = torch.tensor([[[-.02, .3, -.4, .03], [-.025, -.2, .5, .04]]], dtype=torch.float64)
    velocity = torch.tensor([[[.01, -.5, .3, -.02], [.02, .4, -.6, .01]]], dtype=torch.float64)
    return names, position, velocity


def packed(poses, velocities=None):
    parts = [torch.cat(value, dim=-1) for value in poses.values()]
    if velocities is not None:
        parts += [torch.cat(value, dim=-1) for value in velocities.values()]
    return torch.cat(parts, dim=-1)


def test_poses_match_existing_checker_without_loading_another_model(robot):
    checker, bridge, calibration = robot
    names, q, _ = sample()
    result = bridge.poses(q, names)
    assert bridge.model is checker.model
    for index in range(q.shape[1]):
        original = checker.query(joint_names=names, joint_positions=q[0, index].numpy(),
            T_reference_base=calibration, pairs=(), frames=bridge.frames)
        for frame, (position, quaternion) in result.items():
            matrix = np.asarray(original["frame_poses"][frame])
            np.testing.assert_allclose(position[0, index], matrix[:3, 3], rtol=0, atol=1e-14)
            rotation = Rotation.from_quat(quaternion[0, index].numpy()[[1, 2, 3, 0]]).as_matrix()
            np.testing.assert_allclose(rotation, matrix[:3, :3], rtol=0, atol=1e-14)


def test_poses_and_full_states_have_correct_analytic_gradients(robot):
    _, bridge, _ = robot
    names, q, dq = sample()
    q.requires_grad_()
    dq.requires_grad_()
    assert torch.autograd.gradcheck(lambda value: packed(bridge.poses(value, names)),
                                    (q,), eps=1e-6, atol=2e-8, rtol=2e-6, check_batched_grad=True)
    assert torch.autograd.gradcheck(lambda value, rate: packed(*bridge.states(value, rate, names)),
                                    (q, dq), eps=1e-6, atol=2e-8, rtol=2e-6, check_batched_grad=True)


# BEGIN forward-only evaluation must preserve exact values without derivative work.
@pytest.mark.parametrize("enabled_requires_grad", [False, True])
def test_forward_queries_preserve_values_and_skip_unused_derivatives(robot, monkeypatch, enabled_requires_grad):
    import manipulation.pin_kinematics as module
    _, bridge, _ = robot
    names, q, dq = sample()
    q.requires_grad_()
    dq.requires_grad_()
    expected_poses = packed(bridge.poses(q, names)).detach()
    expected_states = packed(*bridge.states(q, dq, names)).detach()

    def unused(*args, **kwargs):
        raise AssertionError("Forward-only query constructed an unused derivative")

    for name in ("computeJointJacobians", "computeForwardKinematicsDerivatives",
                 "getFrameJacobian", "getFrameVelocityDerivatives"):
        monkeypatch.setattr(module.pin, name, unused)
    if enabled_requires_grad:
        with torch.no_grad():
            actual_poses, actual_states = packed(bridge.poses(q, names)), packed(*bridge.states(q, dq, names))
    else:
        actual_poses = packed(bridge.poses(q.detach(), names))
        actual_states = packed(*bridge.states(q.detach(), dq.detach(), names))
    torch.testing.assert_close(actual_poses, expected_poses, rtol=0, atol=0)
    torch.testing.assert_close(actual_states, expected_states, rtol=0, atol=0)
# END calibrated poses and nonzero LOCAL velocities use the original numerical values.


def test_world_twists_match_time_derivatives_of_poses(robot):
    _, bridge, _ = robot
    names, q, dq = sample()
    poses, twists = bridge.states(q, dq, names)
    epsilon = 1e-6
    plus, minus = bridge.poses(q+epsilon*dq, names), bridge.poses(q-epsilon*dq, names)
    for frame, (linear, angular) in twists.items():
        torch.testing.assert_close(linear, (plus[frame][0]-minus[frame][0])/(2*epsilon),
                                   atol=3e-10, rtol=2e-8)
        quaternion = poses[frame][1]
        omega = torch.cat((torch.zeros_like(angular[..., :1]), angular), dim=-1)
        quaternion_rate = .5*quaternion_multiply(omega, quaternion)
        torch.testing.assert_close(quaternion_rate, (plus[frame][1]-minus[frame][1])/(2*epsilon),
                                   atol=3e-10, rtol=2e-8)


def test_tcp_offset_composition_preserves_pose_and_twist_derivatives(robot):
    _, bridge, _ = robot
    names, q, dq = sample()
    q.requires_grad_()
    dq.requires_grad_()
    local_position = torch.tensor([.07, -.04, .08], dtype=torch.float64)
    local_rotation = torch.from_numpy(Rotation.from_rotvec([.2, -.1, .15]).as_quat()[[3, 0, 1, 2]])

    def tcp(value, rate):
        poses, velocities = bridge.states(value, rate, names)
        position, orientation = poses["tool"]
        linear, angular = velocities["tool"]
        offset = quaternion_rotate(orientation, local_position)
        return torch.cat((position+offset, quaternion_multiply(orientation, local_rotation),
                          linear+torch.linalg.cross(angular, offset, dim=-1), angular), dim=-1)

    assert torch.autograd.gradcheck(tcp, (q, dq), eps=1e-6, atol=3e-8, rtol=2e-6)
    values = tcp(q, dq)
    epsilon = 1e-6
    positions_plus = tcp(q+epsilon*dq, dq)[..., :3]
    positions_minus = tcp(q-epsilon*dq, dq)[..., :3]
    torch.testing.assert_close(values[..., 7:10], (positions_plus-positions_minus)/(2*epsilon),
                               atol=3e-10, rtol=2e-8)


def test_named_fingers_remain_independent_and_fixed_base_has_zero_gradient(robot):
    _, bridge, _ = robot
    names, q, _ = sample()
    q.requires_grad_()
    poses = bridge.poses(q, names)
    left_gradient = torch.autograd.grad(poses["left"][0].sum(), q, retain_graph=True)[0]
    assert torch.all(left_gradient[..., names.index("right_slide")] == 0.)
    assert torch.all(left_gradient[..., names.index("left_slide")] != 0.)
    base_gradient = torch.autograd.grad(torch.cat(poses["base"], dim=-1).sum(), q)[0]
    assert torch.all(base_gradient == 0.)
    with pytest.raises(ValueError, match="every model joint"):
        bridge.poses(q[..., :-1], names[:-1])


# BEGIN a covector derivative recovers the first Jacobian through custom Pin VJPs.
@pytest.mark.parametrize("include_velocity", [False, True])
def test_covector_column_batches_match_all_pose_and_state_input_derivatives(robot, include_velocity):
    _, bridge, _ = robot
    names, q, dq = sample()
    inputs = (torch.stack((q, dq)) if include_velocity else q).requires_grad_()

    def evaluate(value):
        raw = (packed(*bridge.states(value[0], value[1], names)) if include_velocity
               else packed(bridge.poses(value, names)))
        # A nonlinear Euclidean consumer also exercises its own chain rule
        # after LOCAL twists have been rotated to the calibrated axes.
        return (raw.square()+torch.sin(raw)).reshape(-1)

    values = evaluate(inputs)
    covector = torch.zeros_like(values, requires_grad=True)
    product, = torch.autograd.grad(values, inputs, grad_outputs=covector, create_graph=True)
    matrix = torch.empty((values.numel(), inputs.numel()), dtype=inputs.dtype)
    for begin in range(0, inputs.numel(), 5):
        end = min(begin+5, inputs.numel())
        columns, = torch.autograd.grad(product.reshape(-1)[begin:end], covector,
            grad_outputs=torch.eye(end-begin, dtype=inputs.dtype), is_grads_batched=True,
            retain_graph=True)
        matrix[:, begin:end] = columns.T
    original = torch.autograd.functional.jacobian(evaluate, inputs).reshape_as(matrix)
    torch.testing.assert_close(matrix, original, atol=2e-14, rtol=2e-13)
    direction = torch.linspace(-.3, .4, inputs.numel(), dtype=inputs.dtype).reshape_as(inputs)
    epsilon = 1e-6
    difference = (evaluate(inputs+epsilon*direction)-evaluate(inputs-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close(matrix@direction.reshape(-1), difference, atol=2e-9, rtol=2e-7)
# END d(J^T v)/dv uses first robot derivatives and does not request a robot Hessian.
# END numerical checks exercise first derivatives in ordinary float64 CPU arithmetic.
