"""Differentiable CPU poses and twists from the registered robot model.

The fixed robot's official model, including its fixed end-effector assembly,
is reused across tasks because its geometry and component relationships are
unchanged. This module receives that existing Pinocchio model; joint feedback,
declared nominal finger coordinates and base calibration determine its poses.
Task objects and runtime simulator privileged state do not enter this API.
"""

# BEGIN Pinocchio analytic derivatives bridge the complete named robot to Torch.
import numpy as np
import pinocchio as pin
import torch

from .planning_problem import quaternion_rotate


def _pose_vjp(pose_gradient, quaternion, jacobian):
    """Convert Euclidean wxyz covectors to reference-frame angular covectors.

    For angular displacement omega in reference axes, the unit quaternion
    obeys dQ = .5*[0,omega] tensor_product Q. The cross-product sign below
    follows that left product. Pinocchio supplies the actual robot Jacobian;
    this conversion only changes the coordinates of its output covector.
    """
    scalar, vector = quaternion[..., :1], quaternion[..., 1:]
    gradient = pose_gradient[..., 3:]
    angular = .5*(-gradient[..., :1]*vector+scalar*gradient[..., 1:]
                   +torch.linalg.cross(vector, gradient[..., 1:], dim=-1))
    spatial = torch.cat((pose_gradient[..., :3], angular), dim=-1)
    # BEGIN elementwise contraction supports Torch's batched VJP transform.
    # Sum the frame and spatial-coordinate axes; leading sample axes and
    # any Jacobian-row batching imposed by autograd remain independent.
    return (jacobian*spatial.unsqueeze(-1)).sum(dim=(-3, -2))
    # END the contraction uses the same analytic geometric Jacobian.


class _PinPoses(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, kinematics):
        poses, jacobian = kinematics._evaluate(position.detach().numpy())
        output = torch.from_numpy(poses)
        ctx.save_for_backward(output[..., 3:], torch.from_numpy(jacobian))
        return output

    @staticmethod
    def backward(ctx, gradient):
        quaternion, jacobian = ctx.saved_tensors
        return _pose_vjp(gradient, quaternion, jacobian), None


class _PinStates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, velocity, kinematics):
        poses, jacobian, twists, velocity_position, velocity_velocity = kinematics._evaluate(
            position.detach().numpy(), velocity.detach().numpy())
        output = torch.from_numpy(poses)
        ctx.save_for_backward(output[..., 3:], torch.from_numpy(jacobian),
                              torch.from_numpy(velocity_position), torch.from_numpy(velocity_velocity))
        return output, torch.from_numpy(twists)

    @staticmethod
    def backward(ctx, pose_gradient, twist_gradient):
        quaternion, jacobian, velocity_position, velocity_velocity = ctx.saved_tensors
        position_gradient = _pose_vjp(pose_gradient, quaternion, jacobian)
        # BEGIN batched and scalar reverse passes share ordinary tensor reductions.
        position_gradient = position_gradient+(velocity_position*twist_gradient.unsqueeze(-1)).sum(dim=(-3, -2))
        velocity_gradient = (velocity_velocity*twist_gradient.unsqueeze(-1)).sum(dim=(-3, -2))
        # END frame and spatial-coordinate reductions preserve input sample axes.
        return position_gradient, velocity_gradient, None


class PinKinematics:
    """Reference-frame kinematics for a complete fixed-base scalar joint model.

    Inputs are float64 CPU tensors with a final named-joint axis and arbitrary
    leading sample axes, including [batch,time,joint]. Every movable joint,
    including nominal finger joints, appears exactly once. Returned poses use
    position in metres and unit quaternions in wxyz order. Returned linear and
    angular velocities are evaluated at each frame origin and expressed in
    the calibrated reference axes, in metres/second and radians/second.

    ``model`` is the existing RobotGeometryChecker.model. No model is loaded,
    rebuilt, locked or reduced. ``frames`` names robot frames in that model.
    ``T_reference_base`` maps model-base coordinates into reference coordinates.
    """

    def __init__(self, model, *, frames, T_reference_base):
        self.model = model
        self.joint_names = tuple(str(name) for name in model.names[1:])
        joints = [model.joints[model.getJointId(name)] for name in self.joint_names]
        if (model.nq != len(joints) or model.nv != len(joints)
                or any(joint.nq != 1 or joint.nv != 1 for joint in joints)):
            raise ValueError("PinKinematics requires the complete fixed-base scalar joint model")
        # Scalar model coordinates have q and velocity columns in joint order.
        # Keep their explicit indices so the library remains their authority.
        self._q_columns = np.asarray([int(joint.idx_q) for joint in joints])
        self._v_columns = np.asarray([int(joint.idx_v) for joint in joints])
        self.frames = tuple(dict.fromkeys(frames))
        if any(not model.existFrame(frame) for frame in self.frames):
            raise ValueError("Requested kinematic frames must belong to the registered robot")
        self._frame_ids = tuple(model.getFrameId(frame) for frame in self.frames)
        calibration = np.asarray(T_reference_base, dtype=float)
        self._rotation, self._translation = calibration[:3, :3].copy(), calibration[:3, 3].copy()

    def _ordered(self, value, joint_names):
        names = tuple(joint_names)
        if len(names) != len(self.joint_names) or set(names) != set(self.joint_names):
            raise ValueError("Robot kinematics requires every model joint exactly once")
        if value.device.type != "cpu" or value.dtype != torch.float64 or value.shape[-1] != len(names):
            raise ValueError("Robot kinematics requires float64 CPU named joint tensors")
        return value[..., [names.index(name) for name in self.joint_names]]

    def poses(self, position, joint_names):
        """Return {frame: (position[...,3], quaternion_wxyz[...,4])}."""
        # BEGIN forward-only consumers do not construct unused analytic derivatives.
        ordered = self._ordered(position, joint_names)
        if torch.is_grad_enabled() and ordered.requires_grad:
            packed = _PinPoses.apply(ordered, self)
        else:
            poses, _ = self._evaluate(ordered.detach().numpy(), with_derivatives=False)
            packed = torch.from_numpy(poses)
        # END the differentiable path retains its original analytic VJP.
        return {frame: (packed[..., index, :3], packed[..., index, 3:])
                for index, frame in enumerate(self.frames)}

    def states(self, position, velocity, joint_names):
        """Return (poses, {frame: (linear_velocity, angular_velocity)}).

        Pinocchio supplies LOCAL frame-origin velocity derivatives. Ordinary
        Torch rotation by the returned orientation expresses those vectors
        in reference axes and differentiates the moving coordinate basis.
        This gives Euclidean output derivatives, including the rotation term
        absent from a direct LOCAL_WORLD_ALIGNED spatial derivative.
        """
        if position.shape != velocity.shape:
            raise ValueError("Robot positions and velocities must share their named sample layout")
        # BEGIN choose the derivative calculation from the actual Torch demand.
        q, dq = self._ordered(position, joint_names), self._ordered(velocity, joint_names)
        if torch.is_grad_enabled() and (q.requires_grad or dq.requires_grad):
            packed, twists = _PinStates.apply(q, dq, self)
        else:
            poses, _, rates, _, _ = self._evaluate(q.detach().numpy(), dq.detach().numpy(),
                                                  with_derivatives=False)
            packed, twists = torch.from_numpy(poses), torch.from_numpy(rates)
        # END LOCAL velocities and the following reference-frame rotation are unchanged.
        poses = {frame: (packed[..., index, :3], packed[..., index, 3:])
                 for index, frame in enumerate(self.frames)}
        velocities = {frame: (quaternion_rotate(packed[..., index, 3:], twists[..., index, :3]),
                              quaternion_rotate(packed[..., index, 3:], twists[..., index, 3:]))
                      for index, frame in enumerate(self.frames)}
        return poses, velocities

    def _evaluate(self, position, velocity=None, *, with_derivatives=True):
        """Evaluate library FK, computing first derivatives only on demand."""
        leading = position.shape[:-1]
        count, joints, frames = int(np.prod(leading)), len(self.joint_names), len(self.frames)
        values = position.reshape(count, joints)
        rates = None if velocity is None else velocity.reshape(count, joints)
        poses = np.empty((count, frames, 7))
        # BEGIN derivative buffers exist only for a differentiable caller.
        jacobian = np.empty((count, frames, 6, joints)) if with_derivatives else None
        if rates is not None:
            twists = np.empty((count, frames, 6))
            velocity_position = np.empty_like(jacobian) if with_derivatives else None
            velocity_velocity = np.empty_like(jacobian) if with_derivatives else None
        # END poses and requested twists always retain the same sample layout.
        data = self.model.createData()
        q, dq = np.empty(self.model.nq), np.empty(self.model.nv)
        for sample in range(count):
            q[self._q_columns] = values[sample]
            if rates is None:
                # BEGIN FK alone supplies every value consumed by pure pose queries.
                if with_derivatives:
                    pin.computeJointJacobians(self.model, data, q)
                else:
                    pin.forwardKinematics(self.model, data, q)
                # END Jacobian work remains in the differentiable branch.
            else:
                dq[self._v_columns] = rates[sample]
                # BEGIN velocity FK and velocity derivatives have distinct consumers.
                if with_derivatives:
                    pin.computeForwardKinematicsDerivatives(self.model, data, q, dq, np.zeros(self.model.nv))
                else:
                    pin.forwardKinematics(self.model, data, q, dq)
                # END no_grad does not implicitly skip a manually requested Pin derivative.
            pin.updateFramePlacements(self.model, data)
            for index, frame in enumerate(self._frame_ids):
                placement = data.oMf[frame]
                rotation = self._rotation@placement.rotation
                poses[sample, index, :3] = self._rotation@placement.translation+self._translation
                poses[sample, index, 3:] = pin.Quaternion(rotation).coeffs()[[3, 0, 1, 2]]
                # BEGIN always read the same LOCAL frame-origin velocity values.
                if rates is not None:
                    motion = pin.getFrameVelocity(self.model, data, frame, pin.LOCAL)
                    twists[sample, index, :3] = motion.linear
                    twists[sample, index, 3:] = motion.angular
                if not with_derivatives:
                    continue
                if rates is not None:
                    derivative_q, derivative_v = pin.getFrameVelocityDerivatives(
                        self.model, data, frame, pin.LOCAL)
                    velocity_position[sample, index] = derivative_q[:, self._v_columns]
                    velocity_velocity[sample, index] = derivative_v[:, self._v_columns]
                geometric = pin.getFrameJacobian(self.model, data, frame, pin.LOCAL_WORLD_ALIGNED)
                jacobian[sample, index, :3] = self._rotation@geometric[:3, self._v_columns]
                jacobian[sample, index, 3:] = self._rotation@geometric[3:, self._v_columns]
                # END only backward consumers require these analytic arrays.
        # BEGIN preserve one result layout with absent, rather than computed, derivatives.
        result = poses.reshape(*leading, frames, 7), (None if jacobian is None else
                                                     jacobian.reshape(*leading, frames, 6, joints))
        if rates is not None:
            return (*result, twists.reshape(*leading, frames, 6),
                    None if velocity_position is None else velocity_position.reshape(*leading, frames, 6, joints),
                    None if velocity_velocity is None else velocity_velocity.reshape(*leading, frames, 6, joints))
        return result
        # END poses, LOCAL twists and calibration keep their original numerical definitions.
# END ordinary Torch composition handles fixed TCP offsets after robot FK.
