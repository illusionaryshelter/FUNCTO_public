# `manipulation/cubic_trajectory.py` 实现契约

代码基线：`__IMPLEMENTATION_COMMIT__`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `CubicBasis`

[定义位置](../../../manipulation/cubic_trajectory.py#L19)。

```python
class CubicBasis():
```

类字段：

无显式条目。

实现说明：

> Uniform cubic coefficients with free entrance acceleration and jerk.
> 
> M position-valued actions represent c2..c(M+1). With h the knot interval,
> c0=x0-2*h*dq0 and c1=1.5*q0+.5*h*dq0-.5*x0 eliminate q0/dq0 exactly.
> Three final copies retain the original stopped final interval. Therefore
> there are M+2 intervals and M+5 complete coefficients. Internally all
> actions are relative to q0, avoiding cancellation of absolute angles in
> derivative evaluation; the public action remains a position coefficient.

## `CubicBasis.__init__`

[定义位置](../../../manipulation/cubic_trajectory.py#L30)。

输入签名：

```python
def __init__(self, controls: int, knot_dt: float):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.isfinite
np.arange
np.zeros
self.matrix
np.eye
self.velocity_coefficients
self.knot_vector
```

写入目标：

```python
self.knots
self.knot_dt
self.intervals
self.knot_vector
self.matrix
self.matrix[0, 0]
self.matrix[1, 0]
self.matrix[2:2 + controls]
self.matrix[-3:, -1]
self.velocity_coefficients
self.velocity_coefficients[:2]
self.basis
self.velocity_basis
```

直接调用符号：

```python
type
np.isfinite
ValueError
float
np.arange
np.zeros
np.eye
BSpline
```

## `CubicBasis.reference_parameters`

[定义位置](../../../manipulation/cubic_trajectory.py#L48)。

输入签名：

```python
def reference_parameters(self):
```

实现说明：

> Greville positions of free coefficients, solely for soft seeds.

显式输出分支：

```python
(np.arange(self.knots) + 1.0) / self.intervals
```

读取的对象属性：

```python
np.arange
self.knots
self.intervals
```

写入目标：

无显式条目。

直接调用符号：

```python
np.arange
```

## `CubicBasis.weights`

[定义位置](../../../manipulation/cubic_trajectory.py#L52)。

输入签名：

```python
def weights(self, progress, derivative=0):
```

实现说明：

> Return action and entrance-velocity weights in physical seconds.
> 
> q=q0+W@(x-q0)+V*dq0. For derivative>0 the independent q0 term is
> absent. Exact endpoint rows simplify mathematical identities, rather
> than admitting a neighbouring measured state or widening a tolerance.

显式输出分支：

```python
self.weights_at_knot_times(progress * self.intervals, derivative)
```

读取的对象属性：

```python
np.asarray
progress.ndim
np.isfinite(progress).all
np.isfinite
np.any
self.weights_at_knot_times
self.intervals
```

写入目标：

```python
progress
```

直接调用符号：

```python
np.asarray
np.isfinite(progress).all
np.isfinite
np.any
ValueError
self.weights_at_knot_times
```

## `CubicBasis.weights_at_knot_times`

[定义位置](../../../manipulation/cubic_trajectory.py#L66)。

输入签名：

```python
def weights_at_knot_times(self, times, derivative=0):
```

实现说明：

> Evaluate exact knot coordinates without a normalize/un-normalize roundtrip.

显式输出分支：

```python
(weights, velocity)
```

读取的对象属性：

```python
np.asarray
times.ndim
np.isfinite(times).all
np.isfinite
np.any
self.intervals
self.knot_dt
self.basis
self.velocity_basis
```

写入目标：

```python
times
scale
weights
velocity
entrance
weights[entrance]
velocity[entrance]
terminal
weights[terminal]
velocity[terminal]
weights[terminal, -1]
```

直接调用符号：

```python
np.asarray
np.isfinite(times).all
np.isfinite
np.any
ValueError
self.basis
self.velocity_basis
float
```

## `CubicTrajectoryStep`

[定义位置](../../../manipulation/cubic_trajectory.py#L90)。

```python
class CubicTrajectoryStep():
```

类字段：

无显式条目。

实现说明：

> Small differentiable action-to-JointState component for cuRobo.
> 
> Both regular rollout and dense export use ``sample``. Basis tensors are
> float64 so measured sub-float32 increments are combined relative to q0
> before the final native float32 state cast. Torch owns the complete VJP;
> there is no detached entrance acceleration or handwritten backward.

## `CubicTrajectoryStep.__init__`

[定义位置](../../../manipulation/cubic_trajectory.py#L99)。

输入签名：

```python
def __init__(self, controls, interpolation_steps, native_dt, *, device, dtype=torch.float32):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
torch.float32
torch.device
self.bind_clock
```

写入目标：

```python
self.action_horizon
self.interpolation_steps
self.horizon
self.device
self.dtype
self.batch_size
self._regular_weights
self._dense_weights
```

直接调用符号：

```python
torch.device
self.bind_clock
```

## `CubicTrajectoryStep.bind_clock`

[定义位置](../../../manipulation/cubic_trajectory.py#L109)。

输入签名：

```python
def bind_clock(self, native_dt):
```

实现说明：

> Explicit request/rebind operation; no optimizer action changes time.

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.float32
np.isfinite
self.interpolation_steps
self.knot_dt
self.action_horizon
```

写入目标：

```python
value
self.native_dt
self.knot_dt
self.curve
self._regular_weights
self._dense_weights
```

直接调用符号：

```python
float
np.float32
np.isfinite
ValueError
CubicBasis
```

## `CubicTrajectoryStep.update_dt`

[定义位置](../../../manipulation/cubic_trajectory.py#L123)。

输入签名：

```python
def update_dt(self, dt):
```

实现说明：

> Accept the native uniform-clock callback only outside evaluation.

显式输出分支：

无显式条目。

读取的对象属性：

```python
torch.as_tensor(dt).detach().reshape
torch.as_tensor(dt).detach
torch.as_tensor
torch.isfinite(values).all
torch.isfinite
(values == values[0]).all
self.bind_clock
```

写入目标：

```python
values
```

直接调用符号：

```python
torch.as_tensor(dt).detach().reshape
torch.as_tensor(dt).detach
torch.as_tensor
len
bool
torch.isfinite(values).all
torch.isfinite
(values == values[0]).all
ValueError
self.bind_clock
float
```

## `CubicTrajectoryStep.update_batch_size`

[定义位置](../../../manipulation/cubic_trajectory.py#L130)。

输入签名：

```python
def update_batch_size(self, batch_size=None, horizon=None, force_update=False):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.horizon
```

写入目标：

```python
self.batch_size
```

直接调用符号：

```python
ValueError
```

## `CubicTrajectoryStep._weights`

[定义位置](../../../manipulation/cubic_trajectory.py#L135)。

输入签名：

```python
def _weights(self, *, capacity, valid_count, dense):
```

显式输出分支：

```python
cached[1:]
```

读取的对象属性：

```python
self._dense_weights
self._regular_weights
self.curve.intervals
self.curve
np.minimum
np.arange
self.curve.weights_at_knot_times
torch.as_tensor
np.stack
self.device
torch.float64
```

写入目标：

```python
key
cached
subdivisions
remainder
times
rows
derivative
action
row
velocity
self._dense_weights
self._regular_weights
```

直接调用符号：

```python
ValueError
divmod
np.minimum
np.arange
self.curve.weights_at_knot_times
range
torch.as_tensor
np.stack
```

## `CubicTrajectoryStep.sample`

[定义位置](../../../manipulation/cubic_trajectory.py#L159)。

输入签名：

```python
def sample(self, actions, start_state, start_state_idx, output, *, valid_count, dense=False, output_dt=None):
```

实现说明：

> Populate q/dq/ddq/jerk from one curve without changing its actions.

显式输出分支：

```python
output
```

读取的对象属性：

```python
actions.ndim
actions.shape
self.action_horizon
actions.device
self.device
actions.dtype
self.dtype
output.position.shape
output.position
start_state_idx.numel
start_state_idx.reshape(-1).long
start_state_idx.reshape
start_state.position.reshape(-1, dof).index_select(0, indices).to
start_state.position.reshape(-1, dof).index_select
start_state.position.reshape
start_state.position
torch.float64
start_state.velocity.reshape(-1, dof).index_select(0, indices).to
start_state.velocity.reshape(-1, dof).index_select
start_state.velocity.reshape
start_state.velocity
self._weights
actions.to
torch.einsum
(values[0] + q0[:, None]).to(self.dtype).contiguous
(values[0] + q0[:, None]).to
values[1].to(self.dtype).contiguous
values[1].to
values[2].to(self.dtype).contiguous
values[2].to
values[3].to(self.dtype).contiguous
values[3].to
self.native_dt
torch.full
```

写入目标：

```python
indices
dof
q0
dq0
weights
velocity
delta
values
output.position
output.velocity
output.acceleration
output.jerk
dt
output.dt
```

直接调用符号：

```python
ValueError
start_state_idx.numel
start_state_idx.reshape(-1).long
start_state_idx.reshape
start_state.position.reshape(-1, dof).index_select(0, indices).to
start_state.position.reshape(-1, dof).index_select
start_state.position.reshape
start_state.velocity.reshape(-1, dof).index_select(0, indices).to
start_state.velocity.reshape(-1, dof).index_select
start_state.velocity.reshape
self._weights
actions.to
torch.einsum
(values[0] + q0[:, None]).to(self.dtype).contiguous
(values[0] + q0[:, None]).to
values[1].to(self.dtype).contiguous
values[1].to
values[2].to(self.dtype).contiguous
values[2].to
values[3].to(self.dtype).contiguous
values[3].to
torch.full
```

## `CubicTrajectoryStep.forward`

[定义位置](../../../manipulation/cubic_trajectory.py#L188)。

输入签名：

```python
def forward(self, start_state, u_act, out_state_seq, start_state_idx=None, goal_state=None, goal_state_idx=None, use_implicit_goal_state=None, **kwargs):
```

实现说明：

> Match the native transition signature, with explicit fixed-clock checks.

显式输出分支：

```python
self.sample(u_act, start_state, start_state_idx, out_state_seq, valid_count=self.horizon)
```

读取的对象属性：

```python
goal_state.dt
goal_state.dt.reshape(-1).index_select
goal_state.dt.reshape
goal_state_idx.reshape(-1).long
goal_state_idx.reshape
torch.isfinite(selected_dt).all
torch.isfinite
(selected_dt == self.native_dt).all
self.native_dt
(use_implicit_goal_state == 0).all
self.sample
self.horizon
```

写入目标：

```python
selected_dt
```

直接调用符号：

```python
ValueError
goal_state.dt.reshape(-1).index_select
goal_state.dt.reshape
goal_state_idx.reshape(-1).long
goal_state_idx.reshape
require_tensor
torch.isfinite(selected_dt).all
torch.isfinite
(selected_dt == self.native_dt).all
(use_implicit_goal_state == 0).all
self.sample
```

## `cubic_transition_config_type`

[定义位置](../../../manipulation/cubic_trajectory.py#L205)。

输入签名：

```python
def cubic_transition_config_type():
```

实现说明：

> Use cuRobo's instance extension API without importing it for CPU algebra.

显式输出分支：

```python
CubicTransitionCfg
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
lru_cache
```

## `cubic_transition_config_type.CubicTransition`

[定义位置](../../../manipulation/cubic_trajectory.py#L211)。

```python
class CubicTransition(RobotStateTransition):
```

类字段：

无显式条目。

## `cubic_transition_config_type.CubicTransition.__init__`

[定义位置](../../../manipulation/cubic_trajectory.py#L212)。

输入签名：

```python
def __init__(self, config):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
config.n_knots
config.interpolation_steps
config.dt_traj_params.base_dt
config.dt_traj_params
config.device_cfg.device
config.device_cfg
config.device_cfg.dtype
```

写入目标：

```python
name
```

直接调用符号：

```python
super().__init__
super
setattr
CubicTrajectoryStep
```

## `cubic_transition_config_type.CubicTransitionCfg`

[定义位置](../../../manipulation/cubic_trajectory.py#L222)。

```python
class CubicTransitionCfg(RobotStateTransitionCfg):
```

类字段：

```python
class_type: type = CubicTransition
```

## `cubic_transition_config_type.CubicTransitionCfg.__post_init__`

[定义位置](../../../manipulation/cubic_trajectory.py#L225)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__post_init__
self.control_space
ControlSpace.BSPLINE_3
self.n_knots
self.interpolation_steps
```

写入目标：

```python
self.horizon
```

直接调用符号：

```python
super().__post_init__
super
ValueError
```

## `cubic_transition_config_type.CubicTransitionCfg.create`

[定义位置](../../../manipulation/cubic_trajectory.py#L232)。

输入签名：

```python
def create(data_dict_in, robot_cfg, device_cfg):
```

显式输出分支：

```python
CubicTransitionCfg(**values)
```

读取的对象属性：

```python
RobotStateTransitionCfg.create
field.name
```

写入目标：

```python
parsed
values
field
values['class_type']
```

直接调用符号：

```python
RobotStateTransitionCfg.create
getattr
fields
CubicTransitionCfg
```
