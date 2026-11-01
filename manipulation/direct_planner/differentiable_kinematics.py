"""Request-local Pinocchio bridges with correct saved-output ownership.

The fixed robot and its installed end-effector keep their official geometry
and fixed component relationships across tasks. The inherited PinKinematics
uses the already registered robot model, calibrated base transform and named
joint states; this module changes only how its autograd nodes retain outputs.

Each node saves the exact pose tensor returned by forward. Saving a view of
that output instead retained discarded forward and covector-derivative graphs
in the audited implementation. Backward obtains the quaternion view from the
saved full output, preserving the same analytic first derivative and storage.
"""

import torch

from manipulation.pin_kinematics import PinKinematics, _pose_vjp
from manipulation.planning_problem import quaternion_rotate


# BEGIN local autograd ownership; native FK and derivative producers are inherited.
class _PinPoses(torch.autograd.Function):
    """Retain the returned pose tensor for the existing pose VJP."""

    @staticmethod
    def forward(ctx, position, kinematics):
        poses, jacobian = kinematics._evaluate(position.detach().numpy())
        output = torch.from_numpy(poses)
        ctx.save_for_backward(output, torch.from_numpy(jacobian))
        return output

    @staticmethod
    def backward(ctx, gradient):
        packed, jacobian = ctx.saved_tensors
        return _pose_vjp(gradient, packed[..., 3:], jacobian), None


class _PinStates(torch.autograd.Function):
    """Preserve the original pose and LOCAL-twist derivative contractions."""

    @staticmethod
    def forward(ctx, position, velocity, kinematics):
        poses, jacobian, twists, velocity_position, velocity_velocity = kinematics._evaluate(
            position.detach().numpy(), velocity.detach().numpy())
        output = torch.from_numpy(poses)
        # BEGIN expose this same native frame Jacobian as a first-order result.
        geometric = torch.from_numpy(jacobian)
        ctx.mark_non_differentiable(geometric)
        # An unused pose/twist output has a zero covector. Keeping it absent
        # also avoids materializing a zero adjoint for the explicit Jacobian.
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(output, geometric,
                              torch.from_numpy(velocity_position), torch.from_numpy(velocity_velocity))
        return output, torch.from_numpy(twists), geometric
        # END the saved pose remains the exact full tensor returned by forward.

    @staticmethod
    def backward(ctx, pose_gradient, twist_gradient, geometric_gradient):
        packed, jacobian, velocity_position, velocity_velocity = ctx.saved_tensors
        # BEGIN missing output covectors are the corresponding exact zero term.
        position_gradient = (None if pose_gradient is None else
                             _pose_vjp(pose_gradient, packed[..., 3:], jacobian))
        velocity_gradient = None
        if twist_gradient is not None:
            position_term = (velocity_position * twist_gradient.unsqueeze(-1)).sum(dim=(-3, -2))
            position_gradient = (position_term if position_gradient is None else
                                 position_gradient + position_term)
            velocity_gradient = (
                velocity_velocity * twist_gradient.unsqueeze(-1)).sum(dim=(-3, -2))
        # END the frame Jacobian is a value result, not a robot Hessian request.
        return position_gradient, velocity_gradient, None
# END saved full outputs release through their own autograd nodes.


class DifferentiablePinKinematics(PinKinematics):
    """Use local bridges while sharing the existing model and numerical APIs.

    Constructor arguments, named-joint ordering, calibration, analytic native
    derivatives and FKSession reuse are inherited from PinKinematics. Poses
    have reference-frame positions in metres and unit wxyz quaternions;
    velocities are expressed in the same reference axes. Inputs remain
    float64 CPU tensors with the complete named-joint axis.

    The planner supplies this instance to both its trajectory computation and
    its necessary-geometry computation before either creates an autograd graph.
    Forward-only candidate queries retain the inherited no-derivative path.
    """

    def poses(self, position, joint_names, *, fk_session=None):
        """Return the inherited frame mapping using this module's pose node."""
        ordered = self._ordered(position, joint_names)
        if torch.is_grad_enabled() and ordered.requires_grad:
            packed = _PinPoses.apply(ordered, self)
        else:
            poses, _ = self._evaluate(ordered.detach().numpy(), with_derivatives=False,
                                      fk_session=fk_session)
            packed = torch.from_numpy(poses)
        return {frame: (packed[..., index, :3], packed[..., index, 3:])
                for index, frame in enumerate(self.frames)}

    def states(self, position, velocity, joint_names, *, fk_session=None):
        """Return poses and reference-frame twists with the original layout.

        Pinocchio provides LOCAL frame-origin twists. Rotating them with the
        returned quaternion keeps the derivative of their moving axes, just
        as in the inherited API. The native evaluator still owns all joint
        and frame Jacobians.
        """
        # BEGIN value consumers and local row linearization share one state path.
        poses, velocities, _ = self._state_outputs(position, velocity, joint_names,
            with_jacobians=False, fk_session=fk_session)
        return poses, velocities

    def states_with_jacobians(self, position, velocity, joint_names):
        """Return poses, twists and the same native geometric frame Jacobians.

        Each Jacobian has shape ``[...,6,len(joint_names)]``. Its first three
        rows differentiate the frame-origin position in reference coordinates;
        its last three give reference-axis angular velocity. Columns follow
        the caller's named input order, including independently named fingers.
        The arrays are first-derivative values at this exact q, not additional
        optimization variables. Poses and twists retain their original VJPs.
        """
        return self._state_outputs(position, velocity, joint_names,
            with_jacobians=True, fk_session=None)

    def _state_outputs(self, position, velocity, joint_names, *, with_jacobians, fk_session):
        """Evaluate one named state batch and expose only requested products."""
        if position.shape != velocity.shape:
            raise ValueError("Robot positions and velocities must share their named sample layout")
        q, dq = self._ordered(position, joint_names), self._ordered(velocity, joint_names)
        if torch.is_grad_enabled() and (q.requires_grad or dq.requires_grad):
            packed, twists, geometric = _PinStates.apply(q, dq, self)
        else:
            poses, geometric, rates, _, _ = self._evaluate(q.detach().numpy(), dq.detach().numpy(),
                with_derivatives=with_jacobians, fk_session=fk_session)
            packed, twists = torch.from_numpy(poses), torch.from_numpy(rates)
            if with_jacobians:
                geometric = torch.from_numpy(geometric)
        poses = {frame: (packed[..., index, :3], packed[..., index, 3:])
                 for index, frame in enumerate(self.frames)}
        velocities = {frame: (quaternion_rotate(packed[..., index, 3:], twists[..., index, :3]),
                              quaternion_rotate(packed[..., index, 3:], twists[..., index, 3:]))
                      for index, frame in enumerate(self.frames)}
        jacobians = None
        if with_jacobians:
            columns = [self.joint_names.index(name) for name in joint_names]
            # One named-column gather serves every requested frame. The native
            # tensor saved by the autograd node remains the original allocation.
            geometric = geometric[..., columns]
            jacobians = {frame: geometric[..., index, :, :]
                         for index, frame in enumerate(self.frames)}
        return poses, velocities, jacobians
        # END forward-only states still omit all unused native derivatives.
