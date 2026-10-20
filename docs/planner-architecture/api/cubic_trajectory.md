# `manipulation/cubic_trajectory.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `CubicBasis`

[定义位置](../../../manipulation/cubic_trajectory.py#L19)。

```python
class CubicBasis:
    ...
    ...
```

该类型的直接方法：`__init__`、`reference_parameters`、`weights`、`weights_at_knot_times`。

## `CubicBasis.__init__`

[定义位置](../../../manipulation/cubic_trajectory.py#L30)。

输入签名：

```python
def CubicBasis.__init__(self, controls: int, knot_dt: float):
```

直接读取的对象属性：`self.matrix`、`self.velocity_coefficients`、`self.knot_vector`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.knots`、`self.knot_dt`、`self.intervals`、`self.knot_vector`、`self.matrix`、`self.matrix[0, 0]`、`self.matrix[1, 0]`、`self.matrix[2:2 + controls]`、`self.matrix[-3:, -1]`、`self.velocity_coefficients`、`self.velocity_coefficients[:2]`、`self.basis`、`self.velocity_basis`。

直接调用：

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
@property
def CubicBasis.reference_parameters(self):
```

直接读取的对象属性：`self.knots`、`self.intervals`。

输出分支的源码表达式：

```python
(np.arange(self.knots) + 1.0) / self.intervals
```

直接调用：

```python
np.arange
```

## `CubicBasis.weights`

[定义位置](../../../manipulation/cubic_trajectory.py#L52)。

输入签名：

```python
def CubicBasis.weights(self, progress, derivative=0):
```

直接读取的对象属性：`self.weights_at_knot_times`、`self.intervals`。

输出分支的源码表达式：

```python
self.weights_at_knot_times(progress * self.intervals, derivative)
```

直接调用：

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
def CubicBasis.weights_at_knot_times(self, times, derivative=0):
```

直接读取的对象属性：`self.intervals`、`self.knot_dt`、`self.basis`、`self.velocity_basis`。

输出分支的源码表达式：

```python
(weights, velocity)
```

直接写入的状态或数组位置：`weights[entrance]`、`velocity[entrance]`、`weights[terminal]`、`velocity[terminal]`、`weights[terminal, -1]`。

直接调用：

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
class CubicTrajectoryStep:
    ...
    ...
```

该类型的直接方法：`__init__`、`bind_clock`、`update_dt`、`update_batch_size`、`_weights`、`sample`、`forward`。

## `CubicTrajectoryStep.__init__`

[定义位置](../../../manipulation/cubic_trajectory.py#L99)。

输入签名：

```python
def CubicTrajectoryStep.__init__(self, controls, interpolation_steps, native_dt, *, device, dtype=torch.float32):
```

直接读取的对象属性：`self.bind_clock`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.action_horizon`、`self.interpolation_steps`、`self.horizon`、`self.device`、`self.dtype`、`self.batch_size`、`self._regular_weights`、`self._dense_weights`。

直接调用：

```python
torch.device
self.bind_clock
```

## `CubicTrajectoryStep.bind_clock`

[定义位置](../../../manipulation/cubic_trajectory.py#L109)。

输入签名：

```python
def CubicTrajectoryStep.bind_clock(self, native_dt):
```

直接读取的对象属性：`self.interpolation_steps`、`self.knot_dt`、`self.action_horizon`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.native_dt`、`self.knot_dt`、`self.curve`、`self._regular_weights`、`self._dense_weights`。

直接调用：

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
def CubicTrajectoryStep.update_dt(self, dt):
```

直接读取的对象属性：`self.bind_clock`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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
def CubicTrajectoryStep.update_batch_size(self, batch_size=None, horizon=None, force_update=False):
```

直接读取的对象属性：`self.horizon`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.batch_size`。

直接调用：

```python
ValueError
```

## `CubicTrajectoryStep._weights`

[定义位置](../../../manipulation/cubic_trajectory.py#L135)。

输入签名：

```python
def CubicTrajectoryStep._weights(self, *, capacity, valid_count, dense):
```

直接读取的对象属性：`self._dense_weights`、`self._regular_weights`、`self.curve`、`self.device`。

输出分支的源码表达式：

```python
cached[1:]
```

直接写入的状态或数组位置：`self._dense_weights`、`self._regular_weights`。

直接调用：

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
def CubicTrajectoryStep.sample(self, actions, start_state, start_state_idx, output, *, valid_count, dense=False, output_dt=None):
```

直接读取的对象属性：`self.action_horizon`、`self.device`、`self.dtype`、`self._weights`、`self.native_dt`。

输出分支的源码表达式：

```python
output
```

直接写入的状态或数组位置：`output.position`、`output.velocity`、`output.acceleration`、`output.jerk`、`output.dt`。

直接调用：

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
def CubicTrajectoryStep.forward(self, start_state, u_act, out_state_seq, start_state_idx=None, goal_state=None, goal_state_idx=None, use_implicit_goal_state=None, **kwargs):
```

直接读取的对象属性：`self.native_dt`、`self.sample`、`self.horizon`。

输出分支的源码表达式：

```python
self.sample(u_act, start_state, start_state_idx, out_state_seq, valid_count=self.horizon)
```

直接调用：

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
@lru_cache(maxsize=1)
def cubic_transition_config_type():
```

输出分支的源码表达式：

```python
CubicTransitionCfg
```

直接调用：

```python
lru_cache
```

## `cubic_transition_config_type.CubicTransition`

[定义位置](../../../manipulation/cubic_trajectory.py#L211)。

```python
class cubic_transition_config_type.CubicTransition(RobotStateTransition):
    ...
    ...
```

该类型的直接方法：`__init__`。

## `cubic_transition_config_type.CubicTransition.__init__`

[定义位置](../../../manipulation/cubic_trajectory.py#L212)。

输入签名：

```python
def cubic_transition_config_type.CubicTransition.__init__(self, config):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
super().__init__
super
setattr
CubicTrajectoryStep
```

## `cubic_transition_config_type.CubicTransitionCfg`

[定义位置](../../../manipulation/cubic_trajectory.py#L222)。

```python
class cubic_transition_config_type.CubicTransitionCfg(RobotStateTransitionCfg):
    class_type: type = CubicTransition
```

该类型的直接方法：`__post_init__`、`create`。

## `cubic_transition_config_type.CubicTransitionCfg.__post_init__`

[定义位置](../../../manipulation/cubic_trajectory.py#L225)。

输入签名：

```python
def cubic_transition_config_type.CubicTransitionCfg.__post_init__(self):
```

直接读取的对象属性：`self.control_space`、`self.n_knots`、`self.interpolation_steps`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.horizon`。

直接调用：

```python
super().__post_init__
super
ValueError
```

## `cubic_transition_config_type.CubicTransitionCfg.create`

[定义位置](../../../manipulation/cubic_trajectory.py#L232)。

输入签名：

```python
@staticmethod
def cubic_transition_config_type.CubicTransitionCfg.create(data_dict_in, robot_cfg, device_cfg):
```

输出分支的源码表达式：

```python
CubicTransitionCfg(**values)
```

直接写入的状态或数组位置：`values['class_type']`。

直接调用：

```python
RobotStateTransitionCfg.create
getattr
fields
CubicTransitionCfg
```
