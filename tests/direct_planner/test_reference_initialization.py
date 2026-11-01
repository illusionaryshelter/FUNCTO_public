"""Registered-joint kinematics and continuous orientation lift checks."""
# BEGIN independent Pinocchio calculations exercise mixed joint axes and branches.
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from manipulation.direct_planner.reference_initialization import SymbolicKinematics


def test_registered_scalar_tree_matches_pinocchio_pose_and_derivative():
    model = pin.Model()
    first = model.addJoint(0, pin.JointModelRY(),
        pin.SE3(Rotation.from_rotvec([.1, -.2, .3]).as_matrix(), np.array([.2, .1, -.3])), 'hinge')
    second = model.addJoint(first, pin.JointModelPX(),
        pin.SE3(Rotation.from_rotvec([-.2, .1, .4]).as_matrix(), np.array([.1, .3, .2])), 'slider')
    model.addJoint(first, pin.JointModelRZ(), pin.SE3.Identity(), 'unrelated')
    frame_placement = pin.SE3(Rotation.from_rotvec([.3, -.1, .2]).as_matrix(), np.array([.2, -.1, .4]))
    frame_id = model.addFrame(pin.Frame('tool_frame', second, frame_placement, pin.FrameType.OP_FRAME))
    base = pin.SE3(Rotation.from_rotvec([-.4, .2, .1]).as_matrix(), np.array([.3, -.2, .5])).homogeneous
    tool = pin.SE3(Rotation.from_rotvec([.2, .4, -.3]).as_matrix(), np.array([.03, -.02, .04])).homogeneous
    model_function = SymbolicKinematics(model, joint_names=('hinge', 'slider'), fixed_positions={},
        frame='tool_frame', reference_from_base=base, frame_from_tool=tool)
    data = model.createData()
    for q in np.random.default_rng(4).uniform([-.8, -.1], [.8, .1], (12, 2)):
        pin.forwardKinematics(model, data, np.r_[q, .7])
        pin.updateFramePlacements(model, data)
        expected = base @ data.oMf[frame_id].homogeneous @ tool
        np.testing.assert_allclose(model_function.pose(q), expected, rtol=0., atol=7e-16)
        quaternion = np.asarray(model_function.orientation_lift(q)).ravel()
        np.testing.assert_allclose(Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix(),
                                   expected[:3, :3], atol=8e-16, rtol=0.)
        numeric = np.column_stack([(np.asarray(model_function.flat(q+np.eye(2)[j]*1e-6)).ravel()
                                  - np.asarray(model_function.flat(q-np.eye(2)[j]*1e-6)).ravel())/2e-6
                                  for j in range(2)])
        np.testing.assert_allclose(model_function.jacobian(q), numeric, rtol=2e-7, atol=2e-10)


def test_continuous_lift_distinguishes_a_full_joint_turn_at_same_rotation():
    model = pin.Model()
    joint = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), 'hinge')
    model.addFrame(pin.Frame('tool_frame', joint, pin.SE3.Identity(), pin.FrameType.OP_FRAME))
    kinematics = SymbolicKinematics(model, joint_names=('hinge',), fixed_positions={},
        frame='tool_frame', reference_from_base=np.eye(4), frame_from_tool=np.eye(4))
    np.testing.assert_allclose(kinematics.pose([0.]), kinematics.pose([2*np.pi]), atol=3e-16)
    np.testing.assert_allclose(kinematics.orientation_lift([0.]),
                               -np.asarray(kinematics.orientation_lift([2*np.pi])), atol=2e-16)
# END the lift is a model-derived path diagnostic, not a task-specific rotation rule.
