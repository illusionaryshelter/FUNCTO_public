"""Scene FK reuses the registered Pinocchio model across tensor dtypes."""

from unittest.mock import patch

import pytest
import torch

from manipulation.observed_robot import DifferentiableRobotKinematics
from test_pin_kinematics import robot, sample, packed
from test_online_geometry import moving_scene, arguments


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scene_adapter_uses_one_existing_model_and_preserves_copy_gradients(robot, dtype):
    checker, reference, calibration = robot
    # Loading another tree is observable independently of numerical agreement.
    with patch("pinocchio.buildModelFromUrdf", side_effect=AssertionError("Unexpected model reload")):
        actual = DifferentiableRobotKinematics(checker.model, frames=reference.frames,
                                               T_reference_base=calibration)
    assert actual.model is checker.model and actual._bridge.model is checker.model
    names, positions, _ = sample()
    source = positions.to(dtype=dtype).requires_grad_(True)
    comparison = source.detach().clone().requires_grad_(True)
    poses = actual.poses(source, names)
    expected = {frame: tuple(value.to(dtype=dtype) for value in pose)
                for frame, pose in reference.poses(comparison.to(dtype=torch.float64), names).items()}
    for frame in reference.frames:
        for value, target in zip(poses[frame], expected[frame]):
            assert value.device == source.device and value.dtype == dtype
            torch.testing.assert_close(value, target, rtol=0, atol=0)
    weights = torch.arange(packed(poses).numel(), dtype=dtype).reshape_as(packed(poses)) / 100
    gradient = torch.autograd.grad((packed(poses) * weights).sum(), source)[0]
    expected_gradient = torch.autograd.grad((packed(expected) * weights).sum(), comparison)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    assert torch.count_nonzero(gradient[..., names.index("left_slide")]) > 0
    assert torch.count_nonzero(gradient[..., names.index("right_slide")]) > 0


def test_scene_adapter_has_the_same_calibrated_pose_derivative(robot):
    checker, reference, calibration = robot
    bridge = DifferentiableRobotKinematics(checker.model, frames=reference.frames,
                                           T_reference_base=calibration)
    names, q, _ = sample()
    q.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda value: packed(bridge.poses(value, names)),
                                    q, eps=1e-6, atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scene_binding_uses_validator_model_and_current_named_finger_state(tmp_path, dtype):
    scene = moving_scene(tmp_path)
    # This fixture replaces the initial static checker with a moving-link
    # checker; give its explicit registered frames to the production binder.
    scene.required_robot_links = tuple(dict.fromkeys(
        row[0] for row in scene.robot_model._geometries.values()))
    validator = scene.robot_validator
    validator.pairs = scene.robot_pairs
    scene.bind_online_robot(validator=validator)
    assert scene.kinematics.model is validator.checker.model
    q = torch.tensor([[[.015, -.02]]], dtype=dtype, requires_grad=True)
    args = arguments(q)
    components = scene.components(*args, trajectory_time=args[3], geometry_scope="robot")
    value = components["collision/robot/self"].sum()
    assert value.item() == pytest.approx(.005, abs=2e-9)
    derivative = torch.autograd.grad(value, q)[0]
    torch.testing.assert_close(derivative, q.new_tensor([[[-1., 0.]]]), rtol=0, atol=2e-7)
    record = scene.to_execution_binding()
    assert "robot_configuration" not in record
    assert record["robot_model"]["urdf_xml"] == validator.checker.urdf_xml
