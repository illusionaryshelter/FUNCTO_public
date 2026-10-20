"""One instantaneous-boundary cubic for CPU domains and native tensor states.

Only measured q(0) and dq(0) constrain the entrance. SciPy constructs the
linear basis once; Torch evaluates its actual forward arithmetic and its
gradient. No simulator, skill semantics or custom CUDA kernel enters here.
"""

# BEGIN eliminate exactly two entrance equalities in a shared cubic basis.
from dataclasses import dataclass, fields
from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline
import torch

from .optimizer_device_checks import require_tensor


class CubicBasis:
    """Uniform cubic coefficients with free entrance acceleration and jerk.

    M position-valued actions represent c2..c(M+1). With h the knot interval,
    c0=x0-2*h*dq0 and c1=1.5*q0+.5*h*dq0-.5*x0 eliminate q0/dq0 exactly.
    Three final copies retain the original stopped final interval. Therefore
    there are M+2 intervals and M+5 complete coefficients. Internally all
    actions are relative to q0, avoiding cancellation of absolute angles in
    derivative evaluation; the public action remains a position coefficient.
    """

    def __init__(self, controls: int, knot_dt: float):
        if (type(controls) is not int or controls < 2
                or not np.isfinite(knot_dt) or knot_dt <= 0):
            raise ValueError("Cubic basis needs at least two controls and a finite positive clock")
        self.knots, self.knot_dt = controls, float(knot_dt)
        self.intervals = controls + 2
        self.knot_vector = np.arange(-3., controls + 6.)
        self.matrix = np.zeros((controls + 5, controls))
        self.matrix[0, 0], self.matrix[1, 0] = 1., -.5
        self.matrix[2:2 + controls] = np.eye(controls)
        self.matrix[-3:, -1] = 1.
        self.velocity_coefficients = np.zeros(controls + 5)
        self.velocity_coefficients[:2] = [-2., .5]
        self.basis = BSpline(self.knot_vector, self.matrix, 3, extrapolate=False)
        self.velocity_basis = BSpline(self.knot_vector, self.velocity_coefficients, 3,
                                      extrapolate=False)

    @property
    def reference_parameters(self):
        """Greville positions of free coefficients, solely for soft seeds."""
        return (np.arange(self.knots) + 1.) / self.intervals

    def weights(self, progress, derivative=0):
        """Return action and entrance-velocity weights in physical seconds.

        q=q0+W@(x-q0)+V*dq0. For derivative>0 the independent q0 term is
        absent. Exact endpoint rows simplify mathematical identities, rather
        than admitting a neighbouring measured state or widening a tolerance.
        """
        progress = np.asarray(progress, dtype=float)
        if (progress.ndim != 1 or not np.isfinite(progress).all()
                or np.any(progress < 0.) or np.any(progress > 1.)
                or derivative not in (0, 1, 2, 3)):
            raise ValueError("Cubic queries need finite normalized times and derivative 0..3")
        return self.weights_at_knot_times(progress * self.intervals, derivative)

    def weights_at_knot_times(self, times, derivative=0):
        """Evaluate exact knot coordinates without a normalize/un-normalize roundtrip."""
        times = np.asarray(times, dtype=float)
        if (times.ndim != 1 or not np.isfinite(times).all()
                or np.any(times < 0.) or np.any(times > self.intervals)
                or derivative not in (0, 1, 2, 3)):
            raise ValueError("Cubic knot queries must lie inside the original curve")
        scale = self.knot_dt**derivative
        weights = self.basis(times, nu=derivative) / scale
        velocity = self.velocity_basis(times, nu=derivative) * self.knot_dt / scale
        entrance = times == 0.
        if derivative < 2:
            weights[entrance] = 0.
            velocity[entrance] = float(derivative == 1)
        # The final four identical supporting coefficients make this whole
        # interval exactly constant, including its left endpoint by C2.
        terminal = times >= self.intervals - 1
        weights[terminal] = 0.
        velocity[terminal] = 0.
        if derivative == 0:
            weights[terminal, -1] = 1.
        return weights, velocity


class CubicTrajectoryStep:
    """Small differentiable action-to-JointState component for cuRobo.

    Both regular rollout and dense export use ``sample``. Basis tensors are
    float64 so measured sub-float32 increments are combined relative to q0
    before the final native float32 state cast. Torch owns the complete VJP;
    there is no detached entrance acceleration or handwritten backward.
    """

    def __init__(self, controls, interpolation_steps, native_dt, *, device, dtype=torch.float32):
        self.action_horizon = controls
        self.interpolation_steps = interpolation_steps
        self.horizon = (controls + 2) * interpolation_steps + 1
        self.device, self.dtype = torch.device(device), dtype
        self.batch_size = 0
        self._regular_weights = None
        self._dense_weights = None
        self.bind_clock(native_dt)

    def bind_clock(self, native_dt):
        """Explicit request/rebind operation; no optimizer action changes time."""
        value = float(np.float32(native_dt))
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Cubic transition requires a positive fixed native dt")
        self.native_dt = value
        self.knot_dt = float(np.float32(np.float32(value) * np.float32(self.interpolation_steps)))
        if value * self.interpolation_steps != self.knot_dt:
            raise ValueError("Cubic transition clock must preserve exact native subdivisions")
        self.curve = CubicBasis(self.action_horizon, self.knot_dt)
        # A resident request can change h and the dense valid count. Drop the
        # two bounded grids; never retain prior-request matrices indefinitely.
        self._regular_weights = self._dense_weights = None

    def update_dt(self, dt):
        """Accept the native uniform-clock callback only outside evaluation."""
        values = torch.as_tensor(dt).detach().reshape(-1)
        if not len(values) or not bool(torch.isfinite(values).all() & (values == values[0]).all()):
            raise ValueError("Cubic transition requires one uniform request clock")
        self.bind_clock(float(values[0]))

    def update_batch_size(self, batch_size=None, horizon=None, force_update=False):
        if horizon is not None and horizon != self.horizon:
            raise ValueError("Cubic transition cannot change its bound sample horizon")
        self.batch_size = batch_size

    def _weights(self, *, capacity, valid_count, dense):
        if not 2 <= valid_count <= capacity:
            raise ValueError("Cubic samples need a valid prefix within their output capacity")
        key = (capacity, valid_count)
        cached = self._dense_weights if dense else self._regular_weights
        if cached is None or cached[0] != key:
            # Exact normalized endpoint and all source breakpoints remain on
            # the admitted regular/dense grid. Padding repeats the final state.
            subdivisions, remainder = divmod(valid_count-1, self.curve.intervals)
            if remainder or subdivisions < 1:
                raise ValueError("Cubic output grid must include every polynomial breakpoint")
            times = np.minimum(np.arange(capacity, dtype=float)/subdivisions, self.curve.intervals)
            rows = [self.curve.weights_at_knot_times(times, derivative) for derivative in range(4)]
            action = torch.as_tensor(np.stack([row[0] for row in rows]),
                                     device=self.device, dtype=torch.float64)
            velocity = torch.as_tensor(np.stack([row[1] for row in rows]),
                                       device=self.device, dtype=torch.float64)
            cached = (key, action, velocity)
            if dense:
                self._dense_weights = cached
            else:
                self._regular_weights = cached
        return cached[1:]

    def sample(self, actions, start_state, start_state_idx, output, *, valid_count, dense=False,
               output_dt=None):
        """Populate q/dq/ddq/jerk from one curve without changing its actions."""
        if (actions.ndim != 3 or actions.shape[1] != self.action_horizon
                or actions.device != self.device or actions.dtype != self.dtype
                or output.position.shape[0] != actions.shape[0]
                or output.position.shape[-1] != actions.shape[-1]):
            raise ValueError("Cubic action/output must match the bound native batch and joint layout")
        if start_state_idx is None or start_state_idx.numel() != actions.shape[0]:
            raise ValueError("Cubic transition requires one measured entrance index per action")
        indices = start_state_idx.reshape(-1).long()
        dof = actions.shape[-1]
        q0 = start_state.position.reshape(-1, dof).index_select(0, indices).to(torch.float64)
        dq0 = start_state.velocity.reshape(-1, dof).index_select(0, indices).to(torch.float64)
        weights, velocity = self._weights(capacity=output.position.shape[1],
                                          valid_count=valid_count, dense=dense)
        delta = actions.to(torch.float64) - q0[:, None]
        values = torch.einsum("khn,bnj->kbhj", weights, delta)
        values = values + velocity[:, None, :, None] * dq0[None, :, None]
        # Build each derivative without in-place edits of a tensor needed by
        # autograd. The final cast is the native FK/constraint input contract.
        output.position = (values[0] + q0[:, None]).to(self.dtype).contiguous()
        output.velocity = values[1].to(self.dtype).contiguous()
        output.acceleration = values[2].to(self.dtype).contiguous()
        output.jerk = values[3].to(self.dtype).contiguous()
        dt = self.native_dt if output_dt is None else output_dt
        output.dt = torch.full((actions.shape[0],), dt, device=self.device, dtype=self.dtype)
        return output

    def forward(self, start_state, u_act, out_state_seq, start_state_idx=None,
                goal_state=None, goal_state_idx=None, use_implicit_goal_state=None, **kwargs):
        """Match the native transition signature, with explicit fixed-clock checks."""
        if goal_state is None or goal_state.dt is None or goal_state_idx is None:
            raise ValueError("Cubic transition requires the request's explicit goal clock")
        selected_dt = goal_state.dt.reshape(-1).index_select(0, goal_state_idx.reshape(-1).long())
        require_tensor(torch.isfinite(selected_dt).all() & (selected_dt == self.native_dt).all(),
                       "Cubic transition changed the fixed request-wide spline clock")
        if use_implicit_goal_state is None:
            raise ValueError("Cubic transition requires an explicit free-terminal declaration")
        require_tensor((use_implicit_goal_state == 0).all(),
                       "Cubic transition does not fix a hidden terminal joint goal")
        return self.sample(u_act, start_state, start_state_idx, out_state_seq,
                           valid_count=self.horizon)


@lru_cache(maxsize=1)
def cubic_transition_config_type():
    """Use cuRobo's instance extension API without importing it for CPU algebra."""
    from curobo._src.transition.robot_state_transition import RobotStateTransition
    from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
    from curobo._src.types.control_space import ControlSpace

    class CubicTransition(RobotStateTransition):
        def __init__(self, config):
            super().__init__(config)
            # Keep the native robot model, state storage, FK and cost paths.
            # Its old spline component is never evaluated and is released.
            for name in ("_rollout_step_fn", "_cmd_step_fn"):
                setattr(self, name, CubicTrajectoryStep(config.n_knots, config.interpolation_steps,
                    config.dt_traj_params.base_dt, device=config.device_cfg.device,
                    dtype=config.device_cfg.dtype))

    @dataclass
    class CubicTransitionCfg(RobotStateTransitionCfg):
        class_type: type = CubicTransition

        def __post_init__(self):
            super().__post_init__()
            if self.control_space != ControlSpace.BSPLINE_3:
                raise ValueError("Instantaneous-boundary transition requires cubic position controls")
            self.horizon = (self.n_knots + 2) * self.interpolation_steps + 1

        @staticmethod
        def create(data_dict_in, robot_cfg, device_cfg):
            # The native static factory returns its base class explicitly;
            # preserve its validated parsing, then supply the actual type
            # required by RobotRolloutCfg's strict configuration contract.
            parsed = RobotStateTransitionCfg.create(data_dict_in, robot_cfg, device_cfg)
            values = {field.name: getattr(parsed, field.name) for field in fields(parsed)}
            values["class_type"] = CubicTransition
            return CubicTransitionCfg(**values)

    return CubicTransitionCfg
# END CPU domains, differentiable rollouts and dense export share one cubic.
