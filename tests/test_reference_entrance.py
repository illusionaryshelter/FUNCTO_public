"""Measured TCP entrance velocity in the calibrated reference frame."""

# BEGIN compare the actual Pinocchio velocity with independent pose differences.
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation

from manipulation.planning import _reference_entrance
from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker
from manipulation.robot_validation import RobotStateValidator


@pytest.mark.parametrize("reverse_order", [False, True])
def test_world_twist_matches_pose_derivative_with_rotated_base_and_tcp(tmp_path, reverse_order):
    """Named feedback, base rotation and the TCP offset all affect the result.

    This analytic robot fixture tests kinematics only. The production caller
    supplies its fixed official robot model through the same validator API.
    A lateral TCP offset makes omitting omega cross offset observably wrong.
    """
    path = tmp_path / "kinematics.urdf"
    path.write_text('''<robot name="analytic_chain">
      <link name="base"/><link name="arm"/><link name="tip"/>
      <joint name="turn" type="revolute"><parent link="base"/><child link="arm"/>
        <origin xyz="0.2 -0.1 0.3" rpy="0.1 0.2 -0.3"/><axis xyz="0 0 1"/>
        <limit lower="-3" upper="3" effort="10" velocity="2"/></joint>
      <joint name="slide" type="prismatic"><parent link="arm"/><child link="tip"/>
        <origin xyz="0.1 0.2 0.3" rpy="-0.2 0.3 0.4"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
    </robot>''')
    checker = RobotGeometryChecker(path, base_frame="base", geometries=[
        NativePrimitive("robot_body", "base", "box", (.1, .1, .1), np.eye(4),
                        "analytic robot fixture")])
    base, tcp = np.eye(4), np.eye(4)
    base[:3, :3] = Rotation.from_euler("xyz", [.4, -.6, .8]).as_matrix()
    base[:3, 3] = [1., -2., .7]
    tcp[:3, :3] = Rotation.from_euler("xyz", [-.1, .5, -.3]).as_matrix()
    tcp[:3, 3] = [.2, -.15, .08]
    names = checker.joint_names
    validator = RobotStateValidator(checker, joint_names=names, arm_joint_names=names,
        finger_joint_names=(), pairs=(), named_velocity_limits=checker.joint_velocity_limits,
        reference_frame="world", T_reference_base=base, tool_frame="tip", T_frame_tcp=tcp,
        fk_translation_tolerance_m=1e-6, fk_rotation_tolerance_rad=1e-6,
        source="analytic kinematic regression")
    feedback = {"turn": (.7, -.43), "slide": (.12, .19)}
    order = names[::-1] if reverse_order else names
    request = SimpleNamespace(joint_names=order,
        start_q=np.array([feedback[name][0] for name in order]),
        start_dq=np.array([feedback[name][1] for name in order]))
    pose, twist = _reference_entrance(request, SimpleNamespace(gripper=None), validator)

    def independent_pose(dt):
        q = pin.neutral(checker.model)
        for name, (position, rate) in feedback.items():
            joint = checker.model.joints[checker.model.getJointId(name)]
            q[joint.idx_q] = position + dt*rate
        data = checker.model.createData()
        pin.framesForwardKinematics(checker.model, data, q)
        return base @ data.oMf[checker.model.getFrameId("tip")].homogeneous @ tcp

    step = 1e-6
    left, right = independent_pose(-step), independent_pose(step)
    velocity = (right[:3, 3]-left[:3, 3])/(2*step)
    angular = Rotation.from_matrix(right[:3, :3] @ left[:3, :3].T).as_rotvec()/(2*step)
    np.testing.assert_allclose(pose, independent_pose(0.), atol=1e-14, rtol=0.)
    np.testing.assert_allclose(twist, np.r_[velocity, angular], atol=3e-10, rtol=0.)
    assert np.linalg.norm(twist[3:]) > .4
# END no nominal target or simulator state enters the measured entrance.
