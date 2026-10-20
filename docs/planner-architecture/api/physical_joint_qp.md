# `manipulation/physical_joint_qp.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_numpy`

[定义位置](../../../manipulation/physical_joint_qp.py#L22)。

输入签名：

```python
def _numpy(value):
```

输出分支的源码表达式：

```python
value.detach().cpu().numpy().astype(float) if isinstance(value, torch.Tensor) else np.asarray(value, dtype=float)
```

直接调用：

```python
isinstance
value.detach().cpu().numpy().astype
value.detach().cpu().numpy
value.detach().cpu
value.detach
np.asarray
```

## `_solve_qp`

[定义位置](../../../manipulation/physical_joint_qp.py#L27)。

输入签名：

```python
def _solve_qp(hessian, linear, rows, lower, upper):
```

输出分支的源码表达式：

```python
None
quadprog.solve_qp(np.asarray(hessian, dtype=float), -linear, np.concatenate((rows.T, -rows.T), axis=1), np.r_[lower, -upper])[0]
```

直接调用：

```python
np.any
quadprog.solve_qp
np.asarray
np.concatenate
str
```

## `CubicAffineMap`

[定义位置](../../../manipulation/physical_joint_qp.py#L45)。

```python
class CubicAffineMap(CubicBasis):
    ...
    ...
```

该类型的直接方法：`__init__`、`coefficients`、`query`、`evaluate`、`extrema_times`、`velocity_extrema`、`solve_joint`。

## `CubicAffineMap.__init__`

[定义位置](../../../manipulation/physical_joint_qp.py#L49)。

输入签名：

```python
def CubicAffineMap.__init__(self, start_q, start_dq, knots, knot_dt):
```

直接读取的对象属性：`self.start_q`、`self.start_dq`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.start_q`、`self.start_dq`。

直接调用：

```python
super().__init__
super
_numpy(start_q).reshape
_numpy
_numpy(start_dq).reshape
np.isfinite(self.start_q).all
np.isfinite
np.isfinite(self.start_dq).all
ValueError
```

## `CubicAffineMap.coefficients`

[定义位置](../../../manipulation/physical_joint_qp.py#L56)。

输入签名：

```python
def CubicAffineMap.coefficients(self, actions):
```

直接读取的对象属性：`self.start_q`、`self.matrix`、`self.velocity_coefficients`、`self.knot_dt`、`self.start_dq`。

输出分支的源码表达式：

```python
self.start_q + np.einsum('cn,...nj->...cj', self.matrix, actions - self.start_q) + self.velocity_coefficients[:, None] * self.knot_dt * self.start_dq
```

直接调用：

```python
_numpy
np.einsum
```

## `CubicAffineMap.query`

[定义位置](../../../manipulation/physical_joint_qp.py#L62)。

输入签名：

```python
def CubicAffineMap.query(self, progress, derivative=0):
```

直接读取的对象属性：`self.weights`、`self.start_dq`、`self.start_q`。

输出分支的源码表达式：

```python
(weights, offset)
```

直接调用：

```python
self.weights
_numpy
weights.sum
```

## `CubicAffineMap.evaluate`

[定义位置](../../../manipulation/physical_joint_qp.py#L69)。

输入签名：

```python
def CubicAffineMap.evaluate(self, actions, progress, derivative=0):
```

直接读取的对象属性：`self.knots`、`self.start_q`、`self.weights`、`self.start_dq`。

输出分支的源码表达式：

```python
values + self.start_q if derivative == 0 else values
```

直接调用：

```python
_numpy
len
ValueError
self.weights
np.einsum
```

## `CubicAffineMap.extrema_times`

[定义位置](../../../manipulation/physical_joint_qp.py#L79)。

输入签名：

```python
def CubicAffineMap.extrema_times(self, controls, joint, derivative=0):
```

直接读取的对象属性：`self.matrix`、`self.start_q`、`self.velocity_coefficients`、`self.knot_dt`、`self.start_dq`、`self.knot_vector`、`self.intervals`。

输出分支的源码表达式：

```python
(times, values)
```

直接写入的状态或数组位置：`polynomial.c[-1, entrance]`、`polynomial.c[-2, entrance]`、`polynomial.c[:-1, terminal]`、`polynomial.c[-1, terminal]`、`values[times == 0.0]`。

直接调用：

```python
PPoly.from_spline
BSpline
np.flatnonzero
polynomial.derivative
polynomial.derivative().roots
np.isfinite
np.unique
np.arange
polynomial
```

## `CubicAffineMap.velocity_extrema`

[定义位置](../../../manipulation/physical_joint_qp.py#L111)。

输入签名：

```python
def CubicAffineMap.velocity_extrema(self, controls):
```

直接读取的对象属性：`self.extrema_times`。

输出分支的源码表达式：

```python
np.asarray(limits).T
```

直接调用：

```python
_numpy
range
self.extrema_times
limits.append
float
values.min
values.max
np.asarray
```

## `CubicAffineMap.solve_joint`

[定义位置](../../../manipulation/physical_joint_qp.py#L120)。

输入签名：

```python
def CubicAffineMap.solve_joint(self, hessian, linear, lower, upper, joint, *, origin, initial, native_dtype, step_limit=None, velocity_bounds=None, interior=False):
```

直接读取的对象属性：`self.extrema_times`、`self.query`、`self.intervals`、`self.knots`。

输出分支的源码表达式：

```python
(None, {**report, 'reason': 'sampled_curve_qp_infeasible'})
(increment, {**report, 'reason': 'analytic_extrema_passed'})
(increment, {**report, 'reason': 'rounded_boundary_repeated'})
(increment, {**report, 'reason': 'extrema_exchange_limit'})
```

直接写入的状态或数组位置：`bounds[1]`、`times[derivative]`。

直接调用：

```python
tuple
self.extrema_times
max
abs
float
np.max
np.abs
np.finfo
range
bounds.items
self.query
np.abs(weights).sum
row_parts.append
lower_parts.append
upper_parts.append
np.concatenate
np.eye
report.update
int
len
_solve_qp
(origin.astype(native_dtype) + increment.astype(native_dtype)).astype
origin.astype
increment.astype
bool
np.any
np.unique
```

## `recover_position_step`

[定义位置](../../../manipulation/physical_joint_qp.py#L182)。

输入签名：

```python
def recover_position_step(domain, base, gradient, step_limits, scales, curve=None):
```

输出分支的源码表达式：

```python
(zero, report)
(zero, {**report, 'reason': 'position_qp_infeasible'})
(zero, {**report, 'reason': 'no_representable_descent_direction'})
(direction, {**report, 'success': True, 'reason': 'physical_descent_direction'})
```

直接写入的状态或数组位置：`report['rounds']`、`report['constraint_count']`、`increments[:, joint]`。

直接调用：

```python
torch.zeros_like
ValueError
bool
domain.feasible(base).all
domain.feasible
domain.validate_scales
_numpy
np.broadcast_to
float
scales.max
np.isfinite(g).all
np.isfinite
np.isfinite(limits).all
np.any
max
np.max
np.abs
np.empty_like
base.detach().cpu().numpy
base.detach().cpu
base.detach
np.eye
range
_solve_qp
int
len
curve.solve_joint
torch.as_tensor
torch.eye
domain.limit_step
base.expand
separate.to(torch.float64).flatten(1).norm
separate.to(torch.float64).flatten
separate.to
corrected.to(torch.float64).flatten(1).norm
corrected.to(torch.float64).flatten
corrected.to
torch.where
proposed_norms.clamp_min
torch.finfo
torch.ones_like
corrected.sum
(actual != base[:, None]).any
(gradient.to(torch.float64) * direction.to(torch.float64)).sum
gradient.to
direction.to
combined.to(torch.float64).norm
combined.to
report.update
direction.to(torch.float64).norm
joint_scales.cpu().tolist
joint_scales.cpu
```

## `_fit_feasible`

[定义位置](../../../manipulation/physical_joint_qp.py#L264)。

输入签名：

```python
def _fit_feasible(domain, curve, action, velocity_bounds):
```

输出分支的源码表达式：

```python
False
True
bool(np.isfinite(extrema).all()) and bool(np.all(extrema[0] >= velocity_bounds[0])) and bool(np.all(extrema[1] <= velocity_bounds[1]))
```

直接调用：

```python
bool
domain.feasible(action[None]).all
domain.feasible
curve.velocity_extrema
np.isfinite(extrema).all
np.isfinite
np.all
```

## `_fit_anchor`

[定义位置](../../../manipulation/physical_joint_qp.py#L275)。

输入签名：

```python
def _fit_anchor(domain, curve, dtype, device, velocity_bounds):
```

输出分支的源码表达式：

```python
(None, {'fit_anchor_source': 'not_constructed', 'detail': str(error)})
(anchor, {'fit_anchor_source': 'validated_position_initializer'})
(None, {'fit_anchor_source': 'not_constructed', 'detail': detail['reason']})
(None, {'fit_anchor_source': 'not_constructed', 'detail': 'rounded_anchor_outside_domain'})
(anchor, {'fit_anchor_source': 'position_velocity_qp'})
```

直接写入的状态或数组位置：`controls[:, joint]`。

直接调用：

```python
(domain._anchor if domain._anchor is not None else domain._construct_anchor(dtype, device))[0].to
domain._construct_anchor
str
_fit_feasible
_numpy
np.empty_like
anchor.detach().cpu().numpy
anchor.detach().cpu
anchor.detach
range
curve.solve_joint
np.eye
np.zeros
torch.as_tensor
```

## `fit_cubic_reference`

[定义位置](../../../manipulation/physical_joint_qp.py#L308)。

输入签名：

```python
def fit_cubic_reference(domain, curve, progress, targets, prior_controls=None, velocity_bounds=None):
```

输出分支的源码表达式：

```python
(None, {**report, 'reason': 'fit_initializer_not_constructed'})
(None, {**report, 'reason': 'fit_initializer_not_constructed', 'detail': str(error)})
(None, {**report, 'reason': 'rounded_fit_outside_position_velocity_domain'})
(action, {**report, 'success': True, 'reason': 'physical_reference_fit', 'reference_rms_joint_units': float(np.sqrt(np.mean(residual ** 2))), 'reference_max_error_joint_units': float(np.max(np.abs(residual))), 'query_max_abs_velocity_joint_units_per_s': np.max(np.abs(curve.evaluate(action, progress, 1)), axis=0).tolist(), 'continuous_max_abs_velocity_joint_units_per_s': np.max(np.abs(velocity_extrema), axis=0).tolist()})
```

直接写入的状态或数组位置：`report['rounds']`、`report['constraint_count']`、`controls[:, joint]`、`report['fit_anchor_source']`、`report['maximum_velocity_bound_excess_before_contraction']`。

直接调用：

```python
_numpy
curve.query
len
np.isfinite(velocity_bounds).all
np.isfinite
np.any
ValueError
np.isfinite(values).all
np.zeros
np.isfinite(prior).all
isinstance
torch.empty((), dtype=dtype).numpy
torch.empty
np.finfo
max
float
np.trace
np.eye
np.empty
range
curve.solve_joint
report['joint_qp_status'].append
torch.as_tensor
report.update
_fit_feasible
domain.extrema
torch.maximum(domain.lower - low, high - domain.upper).max
torch.maximum
domain.initialize
str
curve.velocity_extrema
np.max
_fit_anchor
(action - anchor).to(torch.float64).norm
(action - anchor).to
(initialized - anchor).to(torch.float64).norm
(initialized - anchor).to
curve.evaluate
np.sqrt
np.mean
np.abs
np.max(np.abs(curve.evaluate(action, progress, 1)), axis=0).tolist
np.max(np.abs(velocity_extrema), axis=0).tolist
```
