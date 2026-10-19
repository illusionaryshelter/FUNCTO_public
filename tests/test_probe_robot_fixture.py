"""CPU contracts for explicit source loading and fixed-hand probe hypotheses."""

# BEGIN caller adapters preserve source identities and declared hand geometry.
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
from manipulation.planning_problem import PlanningEvent
from probe_robot_fixture import calibrate_native_hand, fixed_probe_hand, load_probe_fixture


def fixture_inputs(tmp_path):
    names = ["arm", "left", "right"]
    fixture = {"joint_names": names, "arm_joint_names": names[:1], "finger_joint_names": names[1:],
        "finger_link_names": ["left_link", "right_link"], "tool_link": "tool",
        "tool_transform": {"position": [.1, .2, .3], "quaternion_wxyz": [1., 0., 0., 0.]},
        "position_limits": {"arm": [-1., 1.], "left": [0., .04], "right": [-.04, 0.]},
        "velocity_limits": dict.fromkeys(names, .2),
        "collision_geometries": [{"link": "left_link"}, {"link": "right_link"}],
        "robot_config_path": str(tmp_path / "robot.yml")}
    robot = {"robot_cfg": {"kinematics": {"lock_joints": {"left": .03, "right": -.02},
        "collision_link_names": ["left_link", "right_link", "extra_tool"],
        "cspace": {"joint_names": names, "default_joint_position": [.1, .03, -.02],
                   "max_acceleration": [1., 2., 3.]}}}}
    (tmp_path / "robot.yml").write_text(yaml.safe_dump(robot))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(fixture))
    return manifest, fixture, robot


def test_loader_retains_supplied_named_posture_and_source_transform(tmp_path):
    manifest, fixture, original = fixture_inputs(tmp_path)
    positions = np.array([.27, .031, -.022])
    validator = SimpleNamespace(checker=SimpleNamespace(joint_limits=fixture["position_limits"]))
    base = np.eye(4)
    base[1, 3] = .8
    with patch("robot_validation_fixture.build_robot_validator", return_value=validator) as build:
        loaded, robot, actual = load_probe_fixture(manifest, reference_frame="measured_frame",
            T_reference_base=base, full_current_positions=positions)
    assert actual is validator and loaded == fixture
    assert robot["robot_cfg"]["kinematics"]["lock_joints"] == {"left": .031, "right": -.022}
    np.testing.assert_array_equal(robot["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"], positions)
    np.testing.assert_array_equal(build.call_args.kwargs["T_reference_base"], base)
    np.testing.assert_array_equal(build.call_args.kwargs["T_frame_tcp"][:3, 3], [.1, .2, .3])
    assert yaml.safe_load(Path(fixture["robot_config_path"]).read_text()) == original


def test_shared_calibration_keeps_named_signed_coordinates_and_sphere_order():
    """Independent affine CPU FK detects accidental finger-sign/order changes."""
    native_names = ("right", "arm", "left")
    parameters = SimpleNamespace(get_sphere_index_from_link_name=lambda link:
        torch.tensor({"left_link": [1], "right_link": [0]}[link]))

    def fk(state):
        values = state.position
        right, arm, left = values.unbind(-1)
        zero = torch.zeros_like(arm)
        radius = torch.full_like(arm, .01)
        balls = torch.stack((torch.stack((arm, -.1+right, zero, radius), -1),
                             torch.stack((arm, .1+left, zero, radius), -1)), -2)
        position = torch.stack((arm, zero, zero), -1).unsqueeze(-2)
        quaternion = torch.zeros((*position.shape[:-1], 4), dtype=values.dtype)
        quaternion[..., 0] = 1.
        return SimpleNamespace(robot_spheres=balls, tool_poses=SimpleNamespace(
            position=position, quaternion=quaternion))

    native = SimpleNamespace(joint_names=native_names, tool_frames=("tool",),
        config=SimpleNamespace(kinematics_config=parameters), compute_kinematics=fk,
        get_self_collision_config=lambda: SimpleNamespace(collision_pairs=torch.tensor([[0, 1]])))
    robot = {"robot_cfg": {"kinematics": {"lock_joints": {"left": .031, "right": -.022},
                                                   "collision_link_names": ["left_link", "right_link"]}}}
    original = deepcopy(robot)
    feedback = {"joint_names": ("arm", "left", "right"), "start_q": [.2, .031, -.022],
                "finger_joint_names": ("left", "right"), "finger_link_names": ("left_link", "right_link")}
    problem = SimpleNamespace(tool_link="tool", transform_robot_spheres=lambda value: value,
                              transform_tool_pose=lambda position, quaternion: (position, quaternion))
    device = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
    with patch("curobo.kinematics.Kinematics", return_value=native), \
            patch("curobo.kinematics.KinematicsCfg.from_robot_yaml_file", side_effect=lambda value, **kwargs: value), \
            patch("robot_fixture.bind_fixture_gripper_geometry", return_value={'sphere_indices': [1, 0]}), \
            patch("robot_fixture.bind_fixture_prismatic_proxies", side_effect=lambda fixture, robot, geometry, feedback:
                {'geometry': [], 'local_spheres': geometry['local_spheres'],
                 'sphere_joint_translation': geometry['sphere_joint_translation'], 'audit': {}}):
        result = calibrate_native_hand(robot, feedback, problem, device=device,
            position_limits={"left": [0., .04], "right": [-.04, 0.]}, fixture={})
    assert result["sphere_indices"] == (1, 0)
    np.testing.assert_allclose(result["current_positions"], [.031, -.022], atol=0., rtol=0.)
    np.testing.assert_allclose(result["local_spheres"][:, 1], [.1, -.1], atol=1e-15)
    np.testing.assert_allclose(result["sphere_joint_translation"][:, 1, :], np.eye(2), atol=1e-15)
    assert result["affine_fk_max_error_m"] < 1e-15 and robot == original


@pytest.mark.parametrize("with_command", [False, True])
def test_fixed_nominal_hand_preserves_actual_posture_and_original_installation_inventory(tmp_path, with_command):
    _, fixture, robot = fixture_inputs(tmp_path)
    values = np.array([.1, .03, -.02])
    event = PlanningEvent("hold", "explicit_probe", 0., 0, "hold", "fixed geometry hypothesis", stop=False)
    problem = SimpleNamespace(events=(event,) if with_command else (), base_pose=np.eye(4))
    geometry = {"joint_limits": np.array([[0., .04], [-.04, 0.]]), "sphere_indices": (0, 1),
        "local_spheres": np.array([[0., .1, 0., .01], [0., -.1, 0., .01]]),
        "sphere_joint_translation": np.zeros((2, 3, 2)),
        "native_geometry_binding": {},
        "affine_fk_max_error_m": 0.,
        "sphere_indices_by_link": {"left_link": [0], "right_link": [1], "extra_tool": [2, 3]}}
    with patch("probe_robot_fixture.calibrate_native_hand", return_value=geometry) as calibrate, \
            patch("robot_fixture.create_fixed_installation_owners", return_value=(("installation",), [])) as owners:
        hand, retained_owners, report = fixed_probe_hand(fixture, robot, problem, device=object(),
            full_current_positions=values, control_timestep_s=.004)
    np.testing.assert_array_equal(calibrate.call_args.args[1]["start_q"], values)
    assert owners.call_args.kwargs["sphere_indices_by_link"] == {"left_link": [0], "right_link": [1]}
    assert retained_owners == ("installation",) and report["control_timestep_s"] == .004
    hand.bind_clock(1.2, SimpleNamespace())
    positions, velocity = hand.nominal_finger_trajectory.sample(np.array([0., .61, 1.2]))
    np.testing.assert_array_equal(positions, np.broadcast_to(values[1:], (3, 2)))
    np.testing.assert_array_equal(velocity, np.zeros((3, 2)))
    assert tuple(row["command"] for row in hand.commands) == (("hold",) if with_command else ())
    # An edited lock must fail before the native calibration can invent a match.
    robot["robot_cfg"]["kinematics"]["lock_joints"]["right"] = -.01
    with pytest.raises(ValueError, match="agree exactly"):
        fixed_probe_hand(fixture, robot, problem, device=object(), full_current_positions=values)
# END these tests do not claim GPU or physical execution evidence.
