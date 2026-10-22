"""Installed native pose costs on physical time grids and fixed TCP frames."""

# BEGIN independent value and directional-derivative reference-cost checks.
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.warp import init_warp
from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath
from manipulation.reference_tracking import (
    FixedTimeStateCSpaceCost,
    NativeReferenceCost,
    freeze_cspace_time_weights,
    physical_time_mean_weights,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_warp(tmp_path_factory):
    # CPU kernels are the installed implementation; only their compiled
    # cache lives in the test output directory. No CUDA FK is claimed here.
    previous = wp.config.kernel_cache_dir
    wp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("reference_warp"))
    init_warp(device_cfg=DeviceCfg("cpu"))
    yield
    wp.config.kernel_cache_dir = previous


def pose(position=(0., 0., 0.), angles=(0., 0., 0.)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    value[:3, 3] = position
    return value


def problem(*, base=None, tcp=None, reference=None):
    if reference is None:
        reference = np.stack((pose(), pose((1., 0., 0.))))
    return PlanningProblem(
        joint_names=("declared_joint",), start_q=np.zeros(1), start_dq=np.zeros(1),
        tool_link="declared_link", reference_frame="reference",
        base_pose=np.eye(4) if base is None else base,
        tool_transform=np.eye(4) if tcp is None else tcp,
        phases=(MotionPhase("interval", PosePath(np.linspace(0., 1., len(reference)), reference,
                                                source="analytic skill representation")),),
    )


def mapping(u=(0., 1.), s=(0., 1.)):
    return SimpleNamespace(u=np.array(u), s=np.array(s))


def identity_quaternions(shape):
    result = torch.zeros(*shape, 4)
    result[..., 0] = 1.
    return result


def test_time_average_uses_actual_intervals_and_valid_prefix():
    times = torch.tensor([[0., .2, 1., 1.], [0., .5, 1., 2.]])
    weights = physical_time_mean_weights(times, torch.tensor([3, 4]))
    torch.testing.assert_close(weights, torch.tensor([[.1, .5, .4, 0.], [.125, .25, .375, .25]]))
    # A linear physical-time scalar has an exact trapezoid integral on both
    # nonuniform grids; padding retains no integration mass.
    torch.testing.assert_close((weights * times).sum(-1), torch.tensor([.5, 1.]))
    with pytest.raises(ValueError, match="increase strictly"):
        physical_time_mean_weights(torch.tensor([[0., 0., 1.]]), torch.tensor([3]))


def test_reference_binding_preserves_clock_hold_and_cropped_progress():
    request = problem()
    cost = NativeReferenceCost(request, DeviceCfg("cpu"), pose_weight=(2., 3.))
    times = torch.tensor([[0., 1., 2., 4., 4.]])
    cost.bind(times, [4], mapping((0., .25, .5, 1.), (.2, .7, .7, 1.)), 4.)
    torch.testing.assert_close(cost.goal.position[:, 0, 0, 0, 0], torch.tensor([.2, .7, .7, 1., 1.]))
    torch.testing.assert_close(cost.native_cost.config.weight, torch.tensor([2., 12.]))
    assert cost.native_cost.config.use_grad_input
    assert cost.native_cost.config.use_lie_group
    criterion = cost.native_cost.config.tool_pose_criteria[request.tool_link]
    torch.testing.assert_close(criterion.terminal_pose_axes_weight_factor, torch.ones(6))
    torch.testing.assert_close(criterion.terminal_pose_convergence_tolerance, torch.zeros(2))


def test_task_and_time_rows_survive_seed_expansion_and_padding():
    request = problem()
    cost = NativeReferenceCost(request, DeviceCfg("cpu"), pose_weight=(2., 0.))
    cost.bind(torch.tensor([[0., .2, 1., 1.], [0., .5, .8, 1.]]), [3, 4], mapping(), 1.)
    rows = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    targets = cost.goal.position.reshape(2, 4, 1, 3)[rows.long()]
    offsets = torch.tensor([.1, .2, -.3, .4])[:, None, None, None]
    position = (targets + torch.cat((offsets, offsets * 0, offsets * 0), -1)).requires_grad_()
    tools = ToolPose([request.tool_link], position, identity_quaternions((4, 4, 1)))
    values = cost.forward(tools, rows)
    values.sum().backward()
    weight = cost.time_weights(4, rows)
    torch.testing.assert_close(values, offsets[:, 0, 0, 0].square()[:, None] * weight)
    torch.testing.assert_close(position.grad[..., 0], 2 * offsets[:, :, :, 0] * weight[..., None])
    assert position.grad[1, 3].abs().max().item() == 0.
    # Changing the bound grid recomputes references and weights explicitly.
    cost.bind(torch.tensor([[0., 1.]]), [2], mapping(s=(.4, 1.)), 1.)
    torch.testing.assert_close(cost.goal.position[:, 0, 0, 0, 0], torch.tensor([.4, 1.]))


def test_reference_average_is_independent_of_constant_error_sample_density():
    request = problem(reference=np.stack((pose(), pose())))
    scores = []
    for times in ([0., 1.], [0., .1, .6, 1.], [0., .25, .5, .75, 1.]):
        cost = NativeReferenceCost(request, DeviceCfg("cpu"), pose_weight=(2., 3.))
        cost.bind(torch.tensor([times]), [len(times)], mapping(), 1.)
        position = torch.zeros(3, len(times), 1, 3)
        position[..., 0] = .05
        quaternion = identity_quaternions((3, len(times), 1))
        quaternion[..., 0] = np.cos(.2 / 2)
        quaternion[..., 3] = np.sin(.2 / 2)
        scores.append(cost.forward(ToolPose([request.tool_link], position, quaternion)).sum(-1))
    expected = torch.full((3,), .05 ** 2 + 3 * .2 ** 2)
    for score in scores:
        torch.testing.assert_close(score, expected, rtol=3e-6, atol=1e-7)


class NativeAxisFK(torch.autograd.Function):
    """One analytic revolute joint with the installed C++ FK VJP protocol.

    SciPy independently constructs forward matrices. The backward uses the
    literal native quaternion decoder and a world/base geometric Jacobian.
    Thus the test isolates the production TCP adapter and real Warp cost.
    A separate actual-CUDA probe validates the installed complete FK chain.
    """

    @staticmethod
    def forward(ctx, angle, axis, fixed_rotation, link_radius):
        theta = angle.detach().numpy()
        axis_np = axis.numpy()
        rotations = Rotation.from_rotvec((theta[..., None] * axis_np).reshape(-1, 3))
        fixed = Rotation.from_quat(fixed_rotation.numpy()[[1, 2, 3, 0]])
        q = (rotations * fixed).as_quat()[:, [3, 0, 1, 2]].reshape(*theta.shape, 4)
        radius = rotations.apply(link_radius.numpy()).reshape(*theta.shape, 3)
        quaternion = torch.tensor(q, dtype=angle.dtype)
        offset = torch.tensor(radius, dtype=angle.dtype)
        ctx.save_for_backward(quaternion, offset, axis)
        return offset[..., None, :], quaternion[..., None, :]

    @staticmethod
    def backward(ctx, position_gradient, quaternion_gradient):
        quaternion, offset, axis = ctx.saved_tensors
        force, gradient = position_gradient[..., 0, :], quaternion_gradient[..., 0, :]
        w, vector = quaternion[..., :1], quaternion[..., 1:]
        gw, gv = gradient[..., :1], gradient[..., 1:]
        angular = .5 * (-vector * gw + w * gv - torch.linalg.cross(vector, gv))
        derivative = ((torch.linalg.cross(axis.expand_as(offset), offset) * force).sum(-1)
                      + (angular * axis).sum(-1))
        return derivative, None, None, None


@pytest.mark.parametrize("pose_weight", [(2., 0.), (0., 3.), (2., 3.)])
def test_nonzero_tcp_and_base_calibration_preserve_native_directional_derivative(pose_weight):
    base = pose((.3, -.2, .1), (.4, -.7, .3))
    tcp = pose((.11, -.08, .06), (.7, -.2, 1.1))
    reference = np.stack((pose((.15, .25, .3), (.1, .2, -.3)),
                          pose((.3, .15, .4), (-.2, .4, .6))))
    request = problem(base=base, tcp=tcp, reference=reference)
    cost = NativeReferenceCost(request, DeviceCfg("cpu"), pose_weight=pose_weight)
    cost.bind(torch.tensor([[0., .4, 1., 1.]]), [3], mapping(), 1.)
    axis = torch.tensor([.3, -.4, .5]); axis /= axis.norm()
    fixed = torch.tensor(Rotation.from_euler("xyz", [.2, -.4, .3]).as_quat()[[3, 0, 1, 2]], dtype=torch.float32)
    radius = torch.tensor([.15, -.1, .05])
    angles = torch.tensor([[.2, -.1, .5, .5], [-.3, .4, -.2, -.2]], requires_grad=True)

    def objective(value):
        position, quaternion = NativeAxisFK.apply(value, axis, fixed, radius)
        return cost.forward(ToolPose([request.tool_link], position, quaternion)).sum()

    value = objective(angles)
    value.backward()
    gradient = angles.grad.clone()
    assert gradient[:, -1].abs().max().item() == 0.
    direction = torch.tensor([[.2, -.4, .1, .7], [-.1, .3, -.5, -.2]])
    direction /= direction.norm()
    # The full float32 rotation cost is about 39 here. A .01-rad difference
    # resolves its scalar rounding while remaining in the smooth rotation
    # branch; smaller .002 differences lose about 1e-3 after subtraction.
    epsilon = .01
    plus = objective(angles.detach() + epsilon * direction).item()
    minus = objective(angles.detach() - epsilon * direction).item()
    numerical = (plus - minus) / (2 * epsilon)
    analytic = (gradient * direction).sum().item()
    assert abs(analytic - numerical) < 8e-4

    # Forward physical meaning is checked independently using SciPy matrices
    # in the declared reference frame, including both fixed calibrations.
    initial = angles.detach().numpy()[0]
    expected = []
    weights = cost.weights[0].numpy()
    for index, theta in enumerate(initial):
        moving = Rotation.from_rotvec(axis.numpy() * theta)
        link = np.eye(4)
        link[:3, :3] = (moving * Rotation.from_quat(fixed.numpy()[[1, 2, 3, 0]])).as_matrix()
        link[:3, 3] = moving.apply(radius.numpy())
        current_world = base @ link @ tcp
        goal_base = np.eye(4)
        goal_base[:3, 3] = cost.goal.position[index, 0, 0, 0].numpy()
        goal_base[:3, :3] = Rotation.from_quat(cost.goal.quaternion[index, 0, 0, 0].numpy()[[1, 2, 3, 0]]).as_matrix()
        goal_world = base @ goal_base
        delta = current_world[:3, 3] - goal_world[:3, 3]
        angle = Rotation.from_matrix(current_world[:3, :3] @ goal_world[:3, :3].T).magnitude()
        expected.append(weights[index] * (.5 * pose_weight[0] * np.dot(delta, delta) + pose_weight[1] * angle ** 2))
    position, quaternion = NativeAxisFK.apply(angles.detach()[:1], axis, fixed, radius)
    actual = cost.forward(ToolPose([request.tool_link], position, quaternion))[0].detach().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_fixed_native_state_cost_retains_point_scale_and_weighted_gradients():
    from copy import deepcopy
    from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
    from curobo._src.cost.cost_cspace_state import StateCSpaceCost
    from curobo._src.robot.types.joint_limits import JointLimits
    from curobo._src.state.state_joint import JointState

    device = DeviceCfg("cpu")
    bounds = torch.tensor([[-2.], [2.]])
    config = CSpaceCostCfg(
        weight=[3., 5., 7., 11., 13.], cost_type="STATE", dof=1,
        activation_distance=[0.] * 5, squared_l2_regularization_weight=[2., 3., 5., 7., 11.],
        retime_weights=True, retime_regularization_weights=True, device_cfg=device,
        joint_limits=JointLimits(["declared_joint"], bounds, bounds, bounds, bounds, bounds, device),
    )
    native_dt = .2
    original = deepcopy(config)
    frozen = freeze_cspace_time_weights(config, native_dt)
    assert frozen is config
    torch.testing.assert_close(frozen.weight, torch.tensor([3., 1., .28, .088, 13.]))
    torch.testing.assert_close(frozen.squared_l2_regularization_weight,
                               torch.tensor([.4, .12, .04, 7., 2.2]))
    assert not frozen.retime_weights and not frozen.retime_regularization_weights
    assert frozen.use_grad_input
    cost = FixedTimeStateCSpaceCost(frozen, native_dt)
    cost.setup_batch_tensors(2, 4)
    native = StateCSpaceCost(original)
    native.setup_batch_tensors(2, 4)
    # Include a position bound violation and nonzero torque: the latter
    # detects the energy kernel's direct dt dependence after retiming stops.
    position = torch.full((2, 4, 1), 2.3, requires_grad=True)
    velocity = torch.full((2, 4, 1), .7, requires_grad=True)
    acceleration = torch.full((2, 4, 1), .6, requires_grad=True)
    jerk = torch.full((2, 4, 1), .3, requires_grad=True)
    torque = torch.full((2, 4, 1), .8, requires_grad=True)
    state = JointState(position, velocity, acceleration, ["declared_joint"], jerk,
                       dt=torch.full((2,), native_dt))
    native_values = native.forward(state, joint_torque=torque).detach().clone()
    values = cost.forward(state, joint_torque=torque)
    torch.testing.assert_close(values, native_values)
    times = torch.tensor([[0., .2, 1., 1.], [0., .5, .8, 1.]])
    quadrature = physical_time_mean_weights(times, torch.tensor([3, 4]))
    (values * quadrature[..., None]).sum().backward()
    # Native squared regularizers use half the squared value; energy uses
    # the full square of torque*velocity*fixed_dt.
    energy_weight = frozen.squared_l2_regularization_weight[4]
    expected_velocity = (.4 * .7 + 2 * energy_weight * .8 ** 2 * .7 * native_dt ** 2)
    expected_torque = (7. * .8 + 2 * energy_weight * .7 ** 2 * .8 * native_dt ** 2)
    torch.testing.assert_close(velocity.grad[..., 0], expected_velocity * quadrature)
    torch.testing.assert_close(torque.grad[..., 0], expected_torque * quadrature)
    assert position.grad[0, 3].item() == 0.
    before = state.dt.clone()
    for evaluation_dt in (.05, .4):
        state.dt = torch.full((2,), evaluation_dt)
        actual = cost.forward(state, joint_torque=torque).detach().clone()
        torch.testing.assert_close(actual, native_values)
        torch.testing.assert_close(state.dt, torch.full((2,), evaluation_dt))
    torch.testing.assert_close(before, torch.full((2,), native_dt))


def test_native_manager_rebind_and_cached_forward_keep_cost_lifecycle(monkeypatch):
    from copy import deepcopy
    from unittest.mock import patch
    import curobo.runtime
    from curobo._src.robot.types.joint_limits import JointLimits
    from curobo._src.state.state_joint import JointState
    from manipulation.planning import _process_manager_type

    # Native CPU Warp costs execute normally. The manager's CUDA streams
    # are disabled only for this CPU lifecycle fixture; no FK is requested.
    monkeypatch.setattr(curobo.runtime, "cuda_streams", False)
    request = problem()
    device = DeviceCfg("cpu")
    bounds = torch.tensor([[-2.], [2.]])
    limits = JointLimits(["declared_joint"], bounds, bounds, bounds, bounds, bounds, device)
    transition = SimpleNamespace(
        robot_model=SimpleNamespace(get_self_collision_config=lambda: None, total_spheres=0,
                                    tool_frames=[request.tool_link]),
        action_dim=1, teleport_mode=False, interpolation_steps=1,
        get_state_bounds=lambda: limits,
    )
    clock = {"native_dt": .5, "duration": 1.}
    config_type = _process_manager_type(request, None, mapping(), np.empty(0), None, clock=clock)
    raw = {"process_mode": "reference_objective", "cspace_cfg": {
        "weight": [3., 5., 7., 11., 13.], "cost_type": "STATE", "activation_distance": [0.] * 5,
        "squared_l2_regularization_weight": [2., 3., 5., 7., 11.],
        "retime_weights": True, "retime_regularization_weights": True,
    }, "tool_pose_cfg": {"weight": [2., 3.]}}
    config = config_type.create(deepcopy(raw), device_cfg=device)
    manager = config.class_type(device)
    manager.initialize_from_config(config, transition)
    joint = JointState(position=torch.zeros(2, 4, 1), velocity=torch.full((2, 4, 1), .2),
                       acceleration=torch.zeros(2, 4, 1), jerk=torch.zeros(2, 4, 1),
                       joint_names=["declared_joint"], dt=torch.full((2,), .5))
    position = torch.zeros(2, 4, 1, 3)
    position[..., 0] = torch.tensor([.1, .6, 1.1, 1.1])[None, :, None]
    tools = ToolPose([request.tool_link], position, identity_quaternions((2, 4, 1)))
    state = SimpleNamespace(joint_state=joint, joint_torque=torch.full((2, 4, 1), .3), tool_poses=tools)
    manager.set_dense_counts(torch.tensor([3, 3]))
    first = manager.compute_costs(state)
    assert first.names == ["cspace", "continuous_reference"]
    baseline = [value.detach().clone() for value in first.values]
    assert first.values[1][:, -1].abs().max().item() == 0.
    goal_storage = manager._reference_cost.goal.position.data_ptr()
    # A warmed-up call must not inspect GPU values to decide whether the
    # fixed reference grid changed. The same guard works on CPU tensors.
    with patch.object(torch.Tensor, "__int__", side_effect=AssertionError("tensor int in cached rollout")), \
            patch.object(torch.Tensor, "__float__", side_effect=AssertionError("tensor float in cached rollout")):
        repeated = manager.compute_costs(state)
    assert manager._reference_cost.goal.position.data_ptr() == goal_storage
    for actual, expected in zip(repeated.values, baseline):
        torch.testing.assert_close(actual, expected)

    # Request rebinding reconstructs native costs from the unscaled template
    # while preserving the manager's batch lifecycle. Same-shape calls must
    # therefore initialize the new cost buffers before their first forward.
    previous = manager.get_cost("cspace")
    manager.bind_request_data()
    assert manager.get_cost("cspace") is not previous
    manager.set_dense_counts(torch.tensor([3, 3]))
    rebound = manager.compute_costs(state)
    for actual, expected in zip(rebound.values, baseline):
        torch.testing.assert_close(actual, expected)
    assert manager.get_cost("cspace")._batch_size == 2
    assert manager.get_cost("cspace")._horizon == 4
    torch.testing.assert_close(manager.get_cost("cspace").config.squared_l2_regularization_weight,
                               torch.tensor([1., .75, .625, 7., 5.5]))

    # The explicit fixed-grid scorer uses candidate seconds rather than the
    # JointState dt retained for the native regularization coefficient scale.
    manager._candidate_query = {"times": torch.tensor([[0., .2, .8, 1.]]),
                                "mapping": mapping(), "duration": 1.}
    manager._reference_grid_key = None
    candidate = manager.compute_costs(state)
    expected = NativeReferenceCost(request, device, pose_weight=(2., 3.)).bind(
        manager._candidate_query["times"], [4], mapping(), 1.).forward(tools)[..., None]
    torch.testing.assert_close(candidate.values[candidate.names.index("continuous_reference")], expected)
# END time, pose-origin and native-gradient contracts share production calls.
