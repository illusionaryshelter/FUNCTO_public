"""Time-averaged native tool-pose tracking for one bound planning request.

Reference poses belong to the skill representation in the declared reference
frame. Native FK supplies robot-link poses in the robot base frame. A fixed
calibration maps that link to the tracked tool; the same calibration is used
for every task. This module owns reference sampling and soft pose costs only.
"""

# BEGIN continuous reference tracking through native pose-cost derivatives.
from __future__ import annotations

from dataclasses import replace

import torch

from curobo._src.cost.cost_cspace_state import StateCSpaceCost

from .motion import _inverse
from .planning_problem import (
    _interpolate,
    _quaternion,
    quaternion_multiply,
    quaternion_rotate,
)


def physical_time_mean_weights(sample_times, valid_count):
    """Return normalized trapezoid weights for valid ordinary samples.

    ``sample_times`` is [A,H] in seconds, and ``valid_count`` is [A]. Each
    row has at least two strictly increasing valid times. Entries after its
    valid prefix are storage padding. The caller supplies the ordinary
    trajectory grid; event and collision-witness samples have another role.
    This function is called when binding a grid, outside optimizer capture.
    """
    if sample_times.ndim != 2 or valid_count.shape != sample_times.shape[:1]:
        raise ValueError("Reference times need [A,H] and valid counts need [A]")
    horizon = sample_times.shape[1]
    if bool(((valid_count < 2) | (valid_count > horizon)).any()):
        raise ValueError("A reference time average requires at least two valid samples")
    intervals = torch.arange(horizon - 1, device=sample_times.device)[None] < valid_count[:, None] - 1
    delta = sample_times[:, 1:] - sample_times[:, :-1]
    if bool((intervals & (~torch.isfinite(delta) | (delta <= 0))).any()):
        raise ValueError("Valid reference sample times must increase strictly")
    delta = torch.where(intervals, delta, 0.)
    half = .5 * delta
    zero = sample_times.new_zeros((sample_times.shape[0], 1))
    weight = torch.cat((half, zero), 1) + torch.cat((zero, half), 1)
    return weight / weight.sum(-1, keepdim=True)


def freeze_cspace_time_weights(config, native_dt):
    """Bind a STATE joint-cost configuration to one physical rollout clock.

    ``config`` is a fresh native CSpaceCostCfg copied from the request's
    ordinary-cost template, before constructing StateCSpaceCost. The fixed
    ``native_dt`` is the time interval generating the admitted ordinary
    optimization states, not the interval of a later scoring/refinement grid.

    The installed kernel retimes bound weights [position,velocity,acceleration,
    jerk,effort] by [1,dt,dt^2,dt^3,1], and regularization weights [velocity,
    acceleration,jerk,effort,energy] by [dt,dt^2,dt^3,1,dt]. Apply each enabled
    rule once to the configuration, then disable runtime retiming. Every
    evaluation grid consequently measures the same physical state cost.
    External quadrature also needs use_grad_input=True in its native kernel.
    The mutated configuration is returned for immediate cost construction;
    rebuilding from the original template owns subsequent request rebinding.
    """
    from curobo._src.cost.cost_cspace_type import CSpaceCostType

    if config.cost_type != CSpaceCostType.STATE:
        raise ValueError("Fixed trajectory-time weights require the native STATE joint cost")
    if config.retime_weights:
        config.weight = config.weight * config.weight.new_tensor(
            [1., native_dt, native_dt ** 2, native_dt ** 3, 1.])
    if config.retime_regularization_weights:
        values = config.squared_l2_regularization_weight
        config.squared_l2_regularization_weight = values * values.new_tensor(
            [native_dt, native_dt ** 2, native_dt ** 3, 1., native_dt])
    config.retime_weights = config.retime_regularization_weights = False
    config.use_grad_input = True
    return config


class FixedTimeStateCSpaceCost(StateCSpaceCost):
    """Evaluate the native STATE cost at its declared coefficient time scale.

    Even with weight retiming disabled, the native energy term is
    weight*(torque*velocity*dt)^2. Its dt must consequently remain the same
    fixed ordinary-rollout interval used by ``freeze_cspace_time_weights``.
    Only this cost's JointState view receives that dt: the source tensors,
    physical sample times, transition model and other managers retain their
    original clock. All value and derivative calculations remain native.
    """

    def __init__(self, config, native_dt):
        super().__init__(config)
        self.native_dt = native_dt

    def setup_batch_tensors(self, batch, horizon):
        if self._batch_size != batch:
            self._fixed_dt = torch.full((batch,), self.native_dt,
                                        **self.device_cfg.as_torch_dict())
        super().setup_batch_tensors(batch, horizon)

    def forward(self, state_batch, *args, **kwargs):
        return super().forward(replace(state_batch, dt=self._fixed_dt), *args, **kwargs)


class NativeRigidToolTransform(torch.autograd.Function):
    """Compose a fixed TCP calibration while preserving native FK's VJP.

    Both input and output quaternion derivatives use cuRobo's pose-cost
    protocol, not Euclidean derivatives with respect to four independent
    coordinates. For unit wxyz q, the native decoder is
    D(q,g)=(-v*g_w+w*g_v-v cross g_v)/2. Its inverse on angular covectors
    tau is E(q,tau)=2*q*[0,tau]. Native FK contracts D with its base-frame
    geometric angular Jacobian. The installed ToolPoseCost emits E.

    With p_T=p_L+R_L*t and q_T=q_L*q_fixed, a position covector f at the
    TCP contributes the link angular covector (R_L*t) cross f. Orientation
    contributes D(q_T,g_T). Their sum is encoded by E(q_L,...) for the
    existing native FK. All vectors in that calculation use the robot base
    frame. The constant transform has no optimization variables.
    """

    @staticmethod
    def forward(ctx, position, quaternion, translation, rotation):
        offset = quaternion_rotate(quaternion, translation)
        tool_quaternion = quaternion_multiply(quaternion, rotation)
        ctx.save_for_backward(quaternion, tool_quaternion, offset)
        return position + offset, tool_quaternion

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, position_gradient, quaternion_gradient):
        link_quaternion, tool_quaternion, offset = ctx.saved_tensors
        w, vector = tool_quaternion[..., :1], tool_quaternion[..., 1:]
        gw, gv = quaternion_gradient[..., :1], quaternion_gradient[..., 1:]
        angular = .5 * (-vector * gw + w * gv - torch.linalg.cross(vector, gv))
        angular = angular + torch.linalg.cross(offset, position_gradient)
        pure_angular = torch.cat((torch.zeros_like(angular[..., :1]), angular), -1)
        native_quaternion_gradient = 2. * quaternion_multiply(link_quaternion, pure_angular)
        return position_gradient.contiguous(), native_quaternion_gradient.contiguous(), None, None


class NativeReferenceCost:
    """Bind a fixed time grid, then evaluate its native reference objective.

    ``pose_weight`` is the pair already declared by the native trajectory
    optimizer: its point energy is wp*||delta_p||^2/2 + wr*theta^2. The
    installed Lie kernel evaluates theta^2/4, so its rotation weight is 4*wr.
    Rotation axes all remain one: changing them would invoke the installed
    kernel's incorrect anisotropic rotation derivative. Zero convergence
    tolerances retain a continuous soft objective everywhere.

    ``bind`` is the request/grid lifecycle boundary. Its [A,H] reference
    rows may be shared by any number of seed or line-search rows. ``forward``
    accepts their explicit row mapping when A is greater than one. Returned
    [B,H] values already contain normalized physical-time weights; summing
    them yields the same objective used for final candidate scoring.
    """

    def __init__(self, problem, device_cfg, *, pose_weight):
        from curobo._src.cost.cost_tool_pose import ToolPoseCost
        from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg

        self.problem = problem
        self.device_cfg = device_cfg
        self.tool_frames = [problem.tool_link]
        self._translation = device_cfg.to_device(problem.tool_transform[:3, 3])
        self._rotation = device_cfg.to_device(_quaternion(problem.tool_transform))
        inverse_base = _inverse(problem.base_pose)
        self._base_translation = device_cfg.to_device(inverse_base[:3, 3])
        self._base_rotation = device_cfg.to_device(_quaternion(inverse_base))
        position_weight, rotation_weight = pose_weight
        config = ToolPoseCostCfg(
            weight=[position_weight, 4. * rotation_weight], device_cfg=device_cfg,
            tool_frames=self.tool_frames, use_lie_group=True, use_grad_input=True,
            _terminal_pose_axes_weight_factor=[1.] * 6,
            _non_terminal_pose_axes_weight_factor=[1.] * 6,
            _terminal_pose_convergence_tolerance=[0., 0.],
            _non_terminal_pose_convergence_tolerance=[0., 0.],
            _project_distance_to_goal=False,
        )
        self.native_cost = ToolPoseCost(config)

    def bind(self, sample_times, valid_count, progress_map, duration):
        """Sample the supplied skill reference at this grid's physical times.

        Seconds are relative to the bound trajectory start. The reference
        coordinate is progress_map(sample_times/duration), including holds
        and a continuation request's cropped progress interval. Sampling
        uses PlanningProblem.sample_reference, preserving composed PosePath
        and mechanism-coordinate behavior. No joint-reference time map is
        applied to this already bound pose-reference clock.

        Call again when the request, mapping, duration or ordinary grid
        changes. The resulting device goals and quadrature weights are
        reused unchanged during all optimization calls on that binding.
        """
        from curobo._src.types.tool_pose import GoalToolPose

        times = torch.as_tensor(sample_times, **self.device_cfg.as_torch_dict()).detach()
        counts = torch.as_tensor(valid_count, device=times.device, dtype=torch.long).reshape(-1)
        self.weights = physical_time_mean_weights(times, counts)
        self.reference_rows, self.horizon = times.shape
        with torch.no_grad():
            progress = _interpolate(progress_map.u, progress_map.s, times / duration)
            position, quaternion = self.problem.sample_reference(progress)
            # The goal is a TCP pose in the robot base frame. Composing the
            # live link-to-TCP transform below preserves that same origin.
            position = self._base_translation + quaternion_rotate(self._base_rotation, position)
            quaternion = quaternion_multiply(self._base_rotation, quaternion)
            self.goal = GoalToolPose(
                self.tool_frames,
                position.reshape(-1, 1, 1, 1, 3).contiguous(),
                quaternion.reshape(-1, 1, 1, 1, 4).contiguous(),
            )
        self._time_indices = torch.arange(self.horizon, device=times.device, dtype=torch.int32)
        return self

    def row_indices(self, batch_size, reference_rows=None):
        """Map each current state row to a bound reference/time-grid row."""
        if reference_rows is None:
            if self.reference_rows != 1:
                raise ValueError("Multiple reference grids require explicit current-to-reference row indices")
            return torch.zeros(batch_size, device=self.weights.device, dtype=torch.int32)
        return reference_rows.reshape(batch_size).to(device=self.weights.device, dtype=torch.int32)

    def time_weights(self, batch_size, reference_rows=None):
        """Return the same [B,H] quadrature for retained ordinary costs."""
        rows = self.row_indices(batch_size, reference_rows)
        return self.weights.index_select(0, rows.long())

    def forward(self, tool_poses, reference_rows=None):
        """Evaluate the existing FK output without another kinematic query.

        The current pose and its cost gradients both use the native FK
        protocol. Ordinary Torch quaternion losses have a different VJP and
        must keep their existing independent derivative connection.
        """
        from curobo._src.types.tool_pose import ToolPose

        batch, horizon = tool_poses.position.shape[:2]
        if horizon != self.horizon:
            raise ValueError("Native reference cost must be rebound when the ordinary grid changes")
        tool_index = tool_poses.tool_frames.index(self.problem.tool_link)
        position, quaternion = NativeRigidToolTransform.apply(
            tool_poses.position[:, :, tool_index:tool_index + 1],
            tool_poses.quaternion[:, :, tool_index:tool_index + 1],
            self._translation, self._rotation,
        )
        rows = self.row_indices(batch, reference_rows)
        target_indices = (rows[:, None] * horizon + self._time_indices[None]).reshape(-1, 1)
        current = ToolPose(self.tool_frames, position.reshape(-1, 1, 1, 3).contiguous(),
                           quaternion.reshape(-1, 1, 1, 4).contiguous())
        self.native_cost.setup_batch_tensors(batch * horizon, 1)
        value = self.native_cost.forward(current, self.goal, target_indices.contiguous())[0]
        return value.reshape(batch, horizon, 2).sum(-1) * self.weights.index_select(0, rows.long())
# END the original reference remains a soft objective on the admitted clock.
