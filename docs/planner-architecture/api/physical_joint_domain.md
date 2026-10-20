# `manipulation/physical_joint_domain.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `JointDomainInitializationError`

[定义位置](../../../manipulation/physical_joint_domain.py#L25)。

```python
class JointDomainInitializationError(ValueError):
    ...
    ...
```

## `cubic_position_report`

[定义位置](../../../manipulation/physical_joint_domain.py#L29)。

输入签名：

```python
def cubic_position_report(curve: CubicHermiteSpline, lower: np.ndarray, upper: np.ndarray) -> dict:
```

输出分支的源码表达式：

```python
{'maximum_bound_excess': maximum, 'maximum_violation': witness}
```

直接调用：

```python
range
len
PPoly
polynomial.derivative().roots
polynomial.derivative
np.unique
np.isfinite
polynomial
np.isfinite(values).all
ValueError
np.maximum
int
excess.argmax
float
```

## `JointPositionDomain`

[定义位置](../../../manipulation/physical_joint_domain.py#L49)。

```python
class JointPositionDomain:
    ...
    ...
```

该类型的直接方法：`__init__`、`feasible`、`recovery_step`、`validate`、`validate_scales`、`limit_step`、`limit_step_tensor`。

## `JointPositionDomain.__init__`

[定义位置](../../../manipulation/physical_joint_domain.py#L52)。

输入签名：

```python
def JointPositionDomain.__init__(self, lower: torch.Tensor, upper: torch.Tensor):
```

直接读取的对象属性：`self.lower`、`self.upper`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.lower`、`self.upper`。

直接调用：

```python
lower.detach().reshape(-1).to
lower.detach().reshape
lower.detach
upper.detach().reshape(-1).to
upper.detach().reshape
upper.detach
bool
torch.isfinite(self.lower).all
torch.isfinite
torch.isfinite(self.upper).all
(self.lower < self.upper).all
ValueError
```

## `JointPositionDomain.feasible`

[定义位置](../../../manipulation/physical_joint_domain.py#L59)。

输入签名：

```python
def JointPositionDomain.feasible(self, action: torch.Tensor) -> torch.Tensor:
```

直接读取的对象属性：`self.lower`、`self.upper`。

输出分支的源码表达式：

```python
(torch.isfinite(values) & (values >= self.lower) & (values <= self.upper)).all(dim=(-1, -2))
```

直接调用：

```python
action.to
(torch.isfinite(values) & (values >= self.lower) & (values <= self.upper)).all
torch.isfinite
```

## `JointPositionDomain.recovery_step`

[定义位置](../../../manipulation/physical_joint_domain.py#L64)。

输入签名：

```python
def JointPositionDomain.recovery_step(self, base, gradient, step_limits, scales):
```

输出分支的源码表达式：

```python
recover_position_step(self, base, gradient, step_limits, scales)
```

直接调用：

```python
recover_position_step
```

## `JointPositionDomain.validate`

[定义位置](../../../manipulation/physical_joint_domain.py#L70)。

输入签名：

```python
def JointPositionDomain.validate(self, action: torch.Tensor) -> torch.Tensor | None:
```

直接读取的对象属性：`self.feasible`。

输出分支的源码表达式：

```python
require_tensor(self.feasible(action).all(), 'An optimizer action left the physical joint domain', error_type=JointDomainInitializationError)
```

直接调用：

```python
require_tensor
self.feasible(action).all
self.feasible
```

## `JointPositionDomain.validate_scales`

[定义位置](../../../manipulation/physical_joint_domain.py#L82)。

输入签名：

```python
@staticmethod
def JointPositionDomain.validate_scales(scales: torch.Tensor) -> None:
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
require_tensor
torch.isfinite(scales).all
torch.isfinite
(scales >= 0).all
```

## `JointPositionDomain.limit_step`

[定义位置](../../../manipulation/physical_joint_domain.py#L87)。

输入签名：

```python
def JointPositionDomain.limit_step(self, base: torch.Tensor, step: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
```

直接读取的对象属性：`self.validate`、`self.validate_scales`、`self.limit_step_tensor`。

输出分支的源码表达式：

```python
limited
```

直接调用：

```python
self.validate
self.validate_scales
self.limit_step_tensor
```

## `JointPositionDomain.limit_step_tensor`

[定义位置](../../../manipulation/physical_joint_domain.py#L101)。

输入签名：

```python
def JointPositionDomain.limit_step_tensor(self, base: torch.Tensor, step: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
```

直接读取的对象属性：`self.feasible`。

输出分支的源码表达式：

```python
step
limited
```

直接调用：

```python
torch.isfinite(step).all
torch.isfinite
torch.where
torch.zeros_like
scales.max
torch.ones_like
self.feasible
has_active_device_checks
torch.cuda.is_current_stream_capturing
bool
full.all
self.feasible(trials).all
range
```

## `CubicJointDomain`

[定义位置](../../../manipulation/physical_joint_domain.py#L152)。

```python
class CubicJointDomain(JointPositionDomain):
    ...
    ...
```

该类型的直接方法：`__init__`、`evaluate`、`fit_reference`、`recovery_step`、`coefficients`、`_relative_coefficients`、`extrema`、`feasible`、`_construct_anchor`、`initialize`。

## `CubicJointDomain.__init__`

[定义位置](../../../manipulation/physical_joint_domain.py#L161)。

输入签名：

```python
def CubicJointDomain.__init__(self, lower: torch.Tensor, upper: torch.Tensor, start_q: torch.Tensor, start_dq: torch.Tensor, knot_dt: float, knots: int):
```

直接读取的对象属性：`self._curve_map`、`self.lower`、`self.upper`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.knots`、`self._anchor`、`self._start_q`、`self._start_dq`、`self._curve_map`、`self.intervals`、`self._coefficient_matrix`、`self._velocity_coefficients`、`self._entrance_valid`。

直接调用：

```python
super().__init__
super
start_q.detach().reshape(-1).to
start_q.detach().reshape
start_q.detach
start_dq.detach().reshape(-1).to
start_dq.detach().reshape
start_dq.detach
CubicAffineMap
torch.as_tensor
torch.isfinite(q).all
torch.isfinite
torch.isfinite(dq).all
(q >= self.lower).all
(q <= self.upper).all
((q > self.lower) | (dq >= 0)).all
((q < self.upper) | (dq <= 0)).all
```

## `CubicJointDomain.evaluate`

[定义位置](../../../manipulation/physical_joint_domain.py#L186)。

输入签名：

```python
def CubicJointDomain.evaluate(self, actions, progress, derivative=0):
```

直接读取的对象属性：`self._curve_map`。

输出分支的源码表达式：

```python
self._curve_map.evaluate(actions, progress, derivative)
```

直接调用：

```python
self._curve_map.evaluate
```

## `CubicJointDomain.fit_reference`

[定义位置](../../../manipulation/physical_joint_domain.py#L194)。

输入签名：

```python
def CubicJointDomain.fit_reference(self, progress, joint_targets, prior_controls=None, velocity_bounds=None):
```

直接读取的对象属性：`self._curve_map`。

输出分支的源码表达式：

```python
fit_cubic_reference(self, self._curve_map, progress, joint_targets, prior_controls, velocity_bounds)
```

直接调用：

```python
fit_cubic_reference
```

## `CubicJointDomain.recovery_step`

[定义位置](../../../manipulation/physical_joint_domain.py#L204)。

输入签名：

```python
def CubicJointDomain.recovery_step(self, base, gradient, step_limits, scales):
```

直接读取的对象属性：`self._curve_map`。

输出分支的源码表达式：

```python
recover_position_step(self, base, gradient, step_limits, scales, self._curve_map)
```

直接调用：

```python
recover_position_step
```

## `CubicJointDomain.coefficients`

[定义位置](../../../manipulation/physical_joint_domain.py#L209)。

输入签名：

```python
def CubicJointDomain.coefficients(self, action: torch.Tensor) -> torch.Tensor:
```

直接读取的对象属性：`self.knots`、`self.lower`、`self._relative_coefficients`、`self._start_q`。

输出分支的源码表达式：

```python
self._relative_coefficients(action) + self._start_q
```

直接调用：

```python
len
ValueError
self._relative_coefficients
```

## `CubicJointDomain._relative_coefficients`

[定义位置](../../../manipulation/physical_joint_domain.py#L216)。

输入签名：

```python
def CubicJointDomain._relative_coefficients(self, action):
```

直接读取的对象属性：`self._coefficient_matrix`、`self._start_q`、`self._velocity_coefficients`、`self._start_dq`。

输出分支的源码表达式：

```python
torch.matmul(self._coefficient_matrix, action.to(torch.float64) - self._start_q) + self._velocity_coefficients[:, None] * self._start_dq
```

直接调用：

```python
torch.matmul
action.to
```

## `CubicJointDomain.extrema`

[定义位置](../../../manipulation/physical_joint_domain.py#L220)。

输入签名：

```python
def CubicJointDomain.extrema(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
```

直接读取的对象属性：`self.knots`、`self.lower`、`self._relative_coefficients`、`self.intervals`、`self._start_dq`、`self._curve_map`、`self._start_q`。

输出分支的源码表达式：

```python
(values.amin(dim=(-2, -3)), values.amax(dim=(-2, -3)))
```

直接调用：

```python
len
ValueError
self._relative_coefficients
range
torch.cat
torch.zeros_like
(self._start_dq * self._curve_map.knot_dt).expand_as
linear.square
torch.sqrt
torch.clamp_min
torch.where
torch.ones_like
torch.stack
values.amin
values.amax
```

## `CubicJointDomain.feasible`

[定义位置](../../../manipulation/physical_joint_domain.py#L258)。

输入签名：

```python
def CubicJointDomain.feasible(self, action: torch.Tensor) -> torch.Tensor:
```

直接读取的对象属性：`self.extrema`、`self._entrance_valid`、`self.lower`、`self.upper`。

输出分支的源码表达式：

```python
self._entrance_valid & torch.isfinite(action).all(dim=(-1, -2)) & torch.isfinite(low).all(-1) & torch.isfinite(high).all(-1) & (low >= self.lower).all(-1) & (high <= self.upper).all(-1)
```

直接调用：

```python
self.extrema
torch.isfinite(action).all
torch.isfinite
torch.isfinite(low).all
torch.isfinite(high).all
(low >= self.lower).all
(high <= self.upper).all
```

## `CubicJointDomain._construct_anchor`

[定义位置](../../../manipulation/physical_joint_domain.py#L264)。

输入签名：

```python
def CubicJointDomain._construct_anchor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
```

直接读取的对象属性：`self._start_q`、`self.knots`、`self._entrance_valid`、`self.feasible`、`self.lower`、`self.upper`、`self.intervals`、`self._curve_map`、`result.success`、`result.x`、`self.validate`。

输出分支的源码表达式：

```python
constant
anchor
```

直接写入的状态或数组位置：`controls[:, joint]`。

直接调用：

```python
self._start_q.to(dtype=dtype, device=device).expand(1, self.knots, -1).clone
self._start_q.to(dtype=dtype, device=device).expand
self._start_q.to
bool
JointDomainInitializationError
self.feasible(constant).all
self.feasible
self.lower.cpu().numpy
self.lower.cpu
self.upper.cpu().numpy
self.upper.cpu
np.empty
len
range
np.linspace
self._curve_map.query
np.minimum
np.abs(weights).sum
np.abs
np.finfo
abs
linprog
np.zeros
result.x.astype(np.float32).astype
result.x.astype
self._curve_map.extrema_times
np.any
np.unique
torch.as_tensor
self.validate
```

## `CubicJointDomain.initialize`

[定义位置](../../../manipulation/physical_joint_domain.py#L312)。

输入签名：

```python
def CubicJointDomain.initialize(self, proposals: torch.Tensor) -> torch.Tensor:
```

直接读取的对象属性：`self.knots`、`self.lower`、`self.feasible`、`self._anchor`、`self._construct_anchor`、`self.limit_step`、`self.validate`。

输出分支的源码表达式：

```python
proposals
initialized.reshape(shape)
```

直接写入的状态或数组位置：`self._anchor`。

直接调用：

```python
proposals.reshape
len
bool
self.feasible(values).all
self.feasible
valid.any
values[valid][:1].clone
self._construct_anchor
self._anchor.expand_as
values.new_ones
self.limit_step
self.validate
initialized.reshape
```
