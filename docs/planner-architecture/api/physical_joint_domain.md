# `manipulation/physical_joint_domain.py` 实现契约

代码基线：本次实现提交同步更新正文和源码；前置代码检查点为 `759d89b`。实际实现提交号由紧随的文档引用提交回填。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `cubic_position_report`

[定义位置](../../../manipulation/physical_joint_domain.py#L17)。

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

[定义位置](../../../manipulation/physical_joint_domain.py#L37)。

```python
class JointPositionDomain():
```

类字段：

无显式条目。

实现说明：

> Ordered model intervals for the named scalar joints used by IK.

## `JointPositionDomain.__init__`

[定义位置](../../../manipulation/physical_joint_domain.py#L40)。

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
self.lower.shape
self.lower
self.upper.shape
self.upper
torch.isfinite(self.lower).all
torch.isfinite
torch.isfinite(self.upper).all
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

## `CubicJointDomain`

[定义位置](../../../manipulation/physical_joint_domain.py#L49)。

```python
class CubicJointDomain(JointPositionDomain):
```

类字段：

无显式条目。

实现说明：

> Continuous q/dq/ddq/jerk bounds with measured entrance and fixed clock.
>
> ``bounds[derivative,side,joint]`` contains four derivative orders and two
> sides in the same named joint order as q0/dq0. Free controls are position
> coefficients; only the represented curve receives physical model bounds.

## `CubicJointDomain.__init__`

[定义位置](../../../manipulation/physical_joint_domain.py#L57)。

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
self.bounds.cpu().numpy().copy
self.bounds.cpu().numpy
self.bounds.cpu
start_q.detach().reshape(-1).to
start_q.detach().reshape
start_q.detach
start_dq.detach().reshape(-1).to
start_dq.detach().reshape
start_dq.detach
self._start_q.shape
self._start_q
self.lower.shape
self.lower
self._start_dq.shape
self._start_dq
self._curve_map.knots
self._curve_map
self._curve_map.intervals
```

写入目标：

```python
self.bounds
self._constraint_bounds
self._start_q
self._start_dq
self._curve_map
self.knots
self.intervals
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
self.bounds.cpu().numpy().copy
self.bounds.cpu().numpy
self.bounds.cpu
start_q.detach().reshape(-1).to
start_q.detach().reshape
start_q.detach
start_dq.detach().reshape(-1).to
start_dq.detach().reshape
start_dq.detach
CubicAffineMap
```

## `CubicJointDomain.evaluate`

[定义位置](../../../manipulation/physical_joint_domain.py#L73)。

输入签名：

```python
def evaluate(self, actions, progress, derivative=0):
```

实现说明：

> Return NumPy samples on normalized progress, with derivatives in seconds.

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

## `CubicJointDomain.signed_constraints`

[定义位置](../../../manipulation/physical_joint_domain.py#L77)。

输入签名：

```python
def signed_constraints(self, actions, *, with_witnesses=False):
```

实现说明：

> Return g<=0 and its coefficient Jacobian for all physical bounds.
>
> Optional witnesses identify derivative order, original segment,
> named joint column, bound side, physical time, value and bound.
> CubicAffineMap defines the fixed row order and active-branch rule.

显式输出分支：

```python
self._curve_map.signed_constraints(actions, self._constraint_bounds, with_witnesses=with_witnesses)
```

读取的对象属性：

```python
self._curve_map.signed_constraints
self._curve_map
self._constraint_bounds
```

写入目标：

无显式条目。

直接调用符号：

```python
self._curve_map.signed_constraints
```

## `CubicJointDomain.fit_samples`

[定义位置](../../../manipulation/physical_joint_domain.py#L87)。

输入签名：

```python
def fit_samples(self, progress, joint_targets, prior_controls=None):
```

实现说明：

> Return a least-squares coefficient seed and its joint-sample error.

显式输出分支：

```python
self._curve_map.fit_samples(progress, joint_targets, prior_controls)
```

读取的对象属性：

```python
self._curve_map.fit_samples
self._curve_map
```

写入目标：

无显式条目。

直接调用符号：

```python
self._curve_map.fit_samples
```

## `CubicJointDomain.feasible`

[定义位置](../../../manipulation/physical_joint_domain.py#L91)。

输入签名：

```python
def feasible(self, actions: torch.Tensor) -> torch.Tensor:
```

实现说明：

> Check each complete curve using precisely the optimizer's g<=0 rows.
>
> The two trailing axes are [free controls,named joints]. Leading axes
> identify independent curves and are preserved in the Boolean result.
> Nonfinite inputs have no physical curve and evaluate as infeasible.

显式输出分支：

```python
torch.tensor(result, device=actions.device, dtype=torch.bool).reshape(actions.shape[:-2])
```

读取的对象属性：

```python
torch.Tensor
actions.shape
self.knots
self.lower
actions.detach().cpu().numpy().reshape
actions.detach().cpu().numpy
actions.detach().cpu
actions.detach
np.isfinite(curve).all
np.isfinite
np.all
self.signed_constraints
torch.tensor(result, device=actions.device, dtype=torch.bool).reshape
torch.tensor
actions.device
torch.bool
```

写入目标：

```python
curves
result
curve
```

直接调用符号：

```python
len
ValueError
actions.detach().cpu().numpy().reshape
actions.detach().cpu().numpy
actions.detach().cpu
actions.detach
bool
np.isfinite(curve).all
np.isfinite
np.all
self.signed_constraints
torch.tensor(result, device=actions.device, dtype=torch.bool).reshape
torch.tensor
```

## `CubicJointDomain.validate`

[定义位置](../../../manipulation/physical_joint_domain.py#L105)。

输入签名：

```python
def validate(self, actions: torch.Tensor) -> None:
```

实现说明：

> Raise ValueError when an input curve violates a physical model bound.

显式输出分支：

无显式条目。

读取的对象属性：

```python
torch.Tensor
self.feasible(actions).all
self.feasible
```

写入目标：

无显式条目。

直接调用符号：

```python
bool
self.feasible(actions).all
self.feasible
ValueError
```
