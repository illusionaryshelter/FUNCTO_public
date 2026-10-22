# `manipulation/physical_joint_domain.py` 实现契约

代码基线：`c84b34a`；本文同时描述本次工作区内已实现、待统一提交的机器人查询结构复用与物理时间参考改动。实现提交后统一更新代码基线引用。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `JointDomainInitializationError`

[定义位置](../../../manipulation/physical_joint_domain.py#L25)。

```python
class JointDomainInitializationError(ValueError):
```

类字段：

无显式条目。

实现说明：

> A bounded initializer construction failed for the bound physical domain.

## `cubic_position_report`

[定义位置](../../../manipulation/physical_joint_domain.py#L29)。

输入签名：

```python
def cubic_position_report(curve: CubicHermiteSpline, lower: np.ndarray, upper: np.ndarray) -> dict:
```

实现说明：

> Check an exported polynomial independently of native free controls.

显式输出分支：

```python
{'maximum_bound_excess': maximum, 'maximum_violation': witness}
```

读取的对象属性：

```python
np.ndarray
curve.c
curve.x
polynomial.derivative().roots
polynomial.derivative
np.unique
np.r_
np.isfinite
np.isfinite(values).all
np.maximum
excess.argmax
```

写入目标：

```python
maximum
witness
joint
polynomial
roots
times
values
excess
index
```

直接调用符号：

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
class JointPositionDomain():
```

类字段：

无显式条目。

实现说明：

> Named finite scalar joint intervals for native single-state IK.

## `JointPositionDomain.__init__`

[定义位置](../../../manipulation/physical_joint_domain.py#L52)。

输入签名：

```python
def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
torch.Tensor
lower.detach().reshape(-1).to
lower.detach().reshape
lower.detach
torch.float64
upper.detach().reshape(-1).to
upper.detach().reshape
upper.detach
torch.isfinite(self.lower).all
torch.isfinite
self.lower
torch.isfinite(self.upper).all
self.upper
(self.lower < self.upper).all
```

写入目标：

```python
self.lower
self.upper
```

直接调用符号：

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
def feasible(self, action: torch.Tensor) -> torch.Tensor:
```

显式输出分支：

```python
(torch.isfinite(values) & (values >= self.lower) & (values <= self.upper)).all(dim=(-1, -2))
```

读取的对象属性：

```python
torch.Tensor
action.to
torch.float64
(torch.isfinite(values) & (values >= self.lower) & (values <= self.upper)).all
torch.isfinite
self.lower
self.upper
```

写入目标：

```python
values
```

直接调用符号：

```python
action.to
(torch.isfinite(values) & (values >= self.lower) & (values <= self.upper)).all
torch.isfinite
```

## `JointPositionDomain.recovery_step`

[定义位置](../../../manipulation/physical_joint_domain.py#L64)。

输入签名：

```python
def recovery_step(self, base, gradient, step_limits, scales):
```

实现说明：

> Return a legal descent direction; the caller must verify cost decrease.

显式输出分支：

```python
recover_position_step(self, base, gradient, step_limits, scales)
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
recover_position_step
```

## `JointPositionDomain.validate`

[定义位置](../../../manipulation/physical_joint_domain.py#L70)。

输入签名：

```python
def validate(self, action: torch.Tensor) -> torch.Tensor | None:
```

实现说明：

> Raise outside a graph; return a recorded predicate inside its scope.
>
> Graph callers must use the predicate to isolate invalid actions before
> a rollout. Recording an error alone would still let a captured kernel
> consume that action before the enclosing block reports the failure.

显式输出分支：

```python
require_tensor(self.feasible(action).all(), 'An optimizer action left the physical joint domain', error_type=JointDomainInitializationError)
```

读取的对象属性：

```python
torch.Tensor
self.feasible(action).all
self.feasible
```

写入目标：

无显式条目。

直接调用符号：

```python
require_tensor
self.feasible(action).all
self.feasible
```

## `JointPositionDomain.validate_scales`

[定义位置](../../../manipulation/physical_joint_domain.py#L82)。

输入签名：

```python
def validate_scales(scales: torch.Tensor) -> None:
```

实现说明：

> Validate the fixed native line-search multipliers outside capture.

显式输出分支：

无显式条目。

读取的对象属性：

```python
torch.Tensor
torch.isfinite(scales).all
torch.isfinite
(scales >= 0).all
```

写入目标：

无显式条目。

直接调用符号：

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
def limit_step(self, base: torch.Tensor, step: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
```

实现说明：

> Shorten one direction before constructing any native trial action.
>
> Every nonnegative line-search scale retains its ratio on the same
> direction. Convexity of the bound physical domain justifies the
> scalar bisection. Rechecking the actual native-dtype trial arithmetic
> handles rounding at a physical boundary without widening that domain.

显式输出分支：

```python
limited
```

读取的对象属性：

```python
torch.Tensor
self.validate
self.validate_scales
self.limit_step_tensor
```

写入目标：

```python
limited
```

直接调用符号：

```python
self.validate
self.validate_scales
self.limit_step_tensor
```

## `JointPositionDomain.limit_step_tensor`

[定义位置](../../../manipulation/physical_joint_domain.py#L101)。

输入签名：

```python
def limit_step_tensor(self, base: torch.Tensor, step: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
```

实现说明：

> Apply the exact step map to an already validated base/scales.
>
> Eager calls can return an unchanged full step after checking every
> actual trial. Graph/warmup scopes retain device-only control flow.
> The optimizer separately validates the resulting trial predicates
> before calling either native rollout.

显式输出分支：

```python
step
limited
```

读取的对象属性：

```python
torch.Tensor
torch.isfinite(step).all
torch.isfinite
torch.where
torch.zeros_like
scales.max
torch.ones_like
self.feasible
step.device.type
step.device
torch.cuda.is_current_stream_capturing
torch.cuda
full.all
self.feasible(trials).all
```

写入目标：

```python
finite
step
maximum
low
high
full
eager
trials
_
middle
trial_step
passed
limited
```

直接调用符号：

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
```

类字段：

无显式条目。

实现说明：

> Continuous q/dq/ddq/jerk domain of the measured-entrance cubic.
>
> Only measured q0/dq0 are fixed. Acceleration and jerk remain optimizable
> immediately; the original three terminal copies retain a stopped end.
> Free controls themselves have no invented box: only the represented
> physical curve and its three time derivatives determine this domain.

## `CubicJointDomain.__init__`

[定义位置](../../../manipulation/physical_joint_domain.py#L161)。

输入签名：

```python
def __init__(self, bounds: torch.Tensor, start_q: torch.Tensor, start_dq: torch.Tensor, knot_dt: float, knots: int):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
torch.Tensor
bounds.detach().to
bounds.detach
torch.float64
self.bounds.ndim
self.bounds
self.bounds.shape
torch.isfinite(self.bounds).all
torch.isfinite
(self.bounds[:, 0] < self.bounds[:, 1]).all
super().__init__
start_q.detach().reshape(-1).to
start_q.detach().reshape
start_q.detach
start_dq.detach().reshape(-1).to
start_dq.detach().reshape
start_dq.detach
q.shape
self.lower.shape
self.lower
dq.shape
self._curve_map.intervals
self._curve_map
torch.as_tensor
self._curve_map.matrix
q.device
self._curve_map.velocity_coefficients
torch.isfinite(q).all
torch.isfinite(dq).all
(q >= self.lower).all
(q <= self.upper).all
self.upper
(dq >= self.bounds[1, 0]).all
(dq <= self.bounds[1, 1]).all
((q > self.lower) | (dq >= 0)).all
((q < self.upper) | (dq <= 0)).all
```

写入目标：

```python
self.bounds
self.knots
q
dq
self._anchor
self._start_q
self._start_dq
self._curve_map
self.intervals
self._coefficient_matrix
self._velocity_coefficients
self._entrance_valid
```

直接调用符号：

```python
bounds.detach().to
bounds.detach
bool
torch.isfinite(self.bounds).all
torch.isfinite
(self.bounds[:, 0] < self.bounds[:, 1]).all
ValueError
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
torch.isfinite(dq).all
(q >= self.lower).all
(q <= self.upper).all
(dq >= self.bounds[1, 0]).all
(dq <= self.bounds[1, 1]).all
((q > self.lower) | (dq >= 0)).all
((q < self.upper) | (dq <= 0)).all
```

## `CubicJointDomain.evaluate`

[定义位置](../../../manipulation/physical_joint_domain.py#L196)。

输入签名：

```python
def evaluate(self, actions, progress, derivative=0):
```

实现说明：

> Sample normalized progress, returning NumPy values in physical units.
>
> Actions end in [free controls, joints]; output ends in [queries,
> joints]. Derivatives 1..3 use seconds from the immutable knot clock.

显式输出分支：

```python
self._curve_map.evaluate(actions, progress, derivative)
```

读取的对象属性：

```python
self._curve_map.evaluate
self._curve_map
```

写入目标：

无显式条目。

直接调用符号：

```python
self._curve_map.evaluate
```

## `CubicJointDomain.fit_reference`

[定义位置](../../../manipulation/physical_joint_domain.py#L205)。

输入签名：

```python
def fit_reference(self, progress, joint_targets, prior_controls=None):
```

实现说明：

> Fit soft joint references in this complete continuous physical domain.
>
> Returns [controls,joints] or None with a construction report.
> Progress uses the normalized physical clock; targets use this domain's
> joint column order. The report includes actual joint-reference errors.

显式输出分支：

```python
fit_cubic_reference(self, self._curve_map, progress, joint_targets, prior_controls)
```

读取的对象属性：

```python
self._curve_map
```

写入目标：

无显式条目。

直接调用符号：

```python
fit_cubic_reference
```

## `CubicJointDomain.recovery_step`

[定义位置](../../../manipulation/physical_joint_domain.py#L215)。

输入签名：

```python
def recovery_step(self, base, gradient, step_limits, scales):
```

实现说明：

> Find a curve-feasible direction without boxing its free coefficients.

显式输出分支：

```python
recover_position_step(self, base, gradient, step_limits, scales, self._curve_map)
```

读取的对象属性：

```python
self._curve_map
```

写入目标：

无显式条目。

直接调用符号：

```python
recover_position_step
```

## `CubicJointDomain.coefficients`

[定义位置](../../../manipulation/physical_joint_domain.py#L220)。

输入签名：

```python
def coefficients(self, action: torch.Tensor) -> torch.Tensor:
```

显式输出分支：

```python
self._relative_coefficients(action) + self._start_q
```

读取的对象属性：

```python
torch.Tensor
action.shape
self.knots
self.lower
self._relative_coefficients
self._start_q
```

写入目标：

无显式条目。

直接调用符号：

```python
len
ValueError
self._relative_coefficients
```

## `CubicJointDomain._relative_coefficients`

[定义位置](../../../manipulation/physical_joint_domain.py#L227)。

输入签名：

```python
def _relative_coefficients(self, action):
```

显式输出分支：

```python
torch.matmul(self._coefficient_matrix, action.to(torch.float64) - self._start_q) + self._velocity_coefficients[:, None] * self._start_dq
```

读取的对象属性：

```python
torch.matmul
self._coefficient_matrix
action.to
torch.float64
self._start_q
self._velocity_coefficients
self._start_dq
```

写入目标：

无显式条目。

直接调用符号：

```python
torch.matmul
action.to
```

## `CubicJointDomain.all_extrema`

[定义位置](../../../manipulation/physical_joint_domain.py#L231)。

输入签名：

```python
def all_extrema(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
```

实现说明：

> Return lower/upper values with trailing [derivative, named joint] axes.
>
> Position checks its velocity roots, velocity checks acceleration
> roots, acceleration checks both segment endpoints, and jerk checks
> each constant segment. Both sides of every jerk jump are included.
> All derivatives use physical seconds on the unchanged knot clock.

显式输出分支：

```python
(torch.stack([value.amin(dim=(-2, -3)) for value in ranges], dim=-2), torch.stack([value.amax(dim=(-2, -3)) for value in ranges], dim=-2))
```

读取的对象属性：

```python
torch.Tensor
action.shape
self.knots
self.lower
self._relative_coefficients
self.intervals
torch.cat
torch.zeros_like
(self._start_dq * self._curve_map.knot_dt).expand_as
self._start_dq
self._curve_map.knot_dt
self._curve_map
linear.square
torch.sqrt
torch.clamp_min
torch.where
torch.ones_like
torch.stack
self._start_q
(6 * a / (h * h * h)).unsqueeze
value.amin
value.amax
```

写入目标：

```python
c
p0
p1
p2
p3
offset
first_diff
second_diff
third_diff
a
b
d
e
initial_d
quadratic
linear
discriminant
root
safe_quad
first
second
safe_linear
linear_root
first_valid
second_valid
values
safe_a
velocity_root
h
velocities
accelerations
jerks
ranges
value
```

直接调用符号：

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
(6 * a / (h * h * h)).unsqueeze
value.amin
value.amax
```

## `CubicJointDomain.extrema`

[定义位置](../../../manipulation/physical_joint_domain.py#L286)。

输入签名：

```python
def extrema(self, action: torch.Tensor, derivative=0) -> tuple[torch.Tensor, torch.Tensor]:
```

实现说明：

> Return continuous extrema of the requested physical derivative.

显式输出分支：

```python
(low[..., derivative, :], high[..., derivative, :])
```

读取的对象属性：

```python
torch.Tensor
self.all_extrema
```

写入目标：

```python
low
high
```

直接调用符号：

```python
self.all_extrema
```

## `CubicJointDomain.feasible`

[定义位置](../../../manipulation/physical_joint_domain.py#L291)。

输入签名：

```python
def feasible(self, action: torch.Tensor) -> torch.Tensor:
```

显式输出分支：

```python
self._entrance_valid & torch.isfinite(action).all(dim=(-1, -2)) & torch.isfinite(low).all(dim=(-1, -2)) & torch.isfinite(high).all(dim=(-1, -2)) & (low >= self.bounds[:, 0]).all(dim=(-1, -2)) & (high <= self.bounds[:, 1]).all(dim=(-1, -2))
```

读取的对象属性：

```python
torch.Tensor
self.all_extrema
self._entrance_valid
torch.isfinite(action).all
torch.isfinite
torch.isfinite(low).all
torch.isfinite(high).all
(low >= self.bounds[:, 0]).all
self.bounds
(high <= self.bounds[:, 1]).all
```

写入目标：

```python
low
high
```

直接调用符号：

```python
self.all_extrema
torch.isfinite(action).all
torch.isfinite
torch.isfinite(low).all
torch.isfinite(high).all
(low >= self.bounds[:, 0]).all
(high <= self.bounds[:, 1]).all
```

## `CubicJointDomain._construct_anchor`

[定义位置](../../../manipulation/physical_joint_domain.py#L298)。

输入签名：

```python
def _construct_anchor(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
```

实现说明：

> Construct a same-domain QP anchor independently of fit/initialization.

显式输出分支：

```python
constant
anchor
```

读取的对象属性：

```python
torch.dtype
torch.device
self._start_q.to(dtype=dtype, device=device).expand(1, self.knots, -1).clone
self._start_q.to(dtype=dtype, device=device).expand
self._start_q.to
self._start_q
self.knots
self._entrance_valid
self.feasible(constant).all
self.feasible
self._curve_map
torch.Tensor
```

写入目标：

```python
constant
anchor
report
```

直接调用符号：

```python
self._start_q.to(dtype=dtype, device=device).expand(1, self.knots, -1).clone
self._start_q.to(dtype=dtype, device=device).expand
self._start_q.to
bool
JointDomainInitializationError
self.feasible(constant).all
self.feasible
construct_cubic_anchor
```

## `CubicJointDomain.initialize`

[定义位置](../../../manipulation/physical_joint_domain.py#L313)。

输入签名：

```python
def initialize(self, proposals: torch.Tensor) -> torch.Tensor:
```

实现说明：

> Keep legal reference seeds and contract only illegal initializers.

显式输出分支：

```python
proposals
initialized.reshape(shape)
```

读取的对象属性：

```python
torch.Tensor
proposals.shape
proposals.reshape
self.knots
self.lower
self.feasible(values).all
self.feasible
self._anchor
valid.any
values[valid][:1].clone
self._construct_anchor
proposals.dtype
proposals.device
self._anchor.expand_as
values.new_ones
self.limit_step
self.validate
initialized.reshape
```

写入目标：

```python
shape
values
valid
self._anchor
base
scales
step
initialized
```

直接调用符号：

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
