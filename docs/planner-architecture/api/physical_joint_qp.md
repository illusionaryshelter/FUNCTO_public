# `manipulation/physical_joint_qp.py` 实现契约

代码基线：`本文件所在的 yaw 实现提交`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_numpy`

[定义位置](../../../manipulation/physical_joint_qp.py#L22)。

输入签名：

```python
def _numpy(value):
```

显式输出分支：

```python
value.detach().cpu().numpy().astype(float) if isinstance(value, torch.Tensor) else np.asarray(value, dtype=float)
```

读取的对象属性：

```python
torch.Tensor
value.detach().cpu().numpy().astype
value.detach().cpu().numpy
value.detach().cpu
value.detach
np.asarray
```

写入目标：

无显式条目。

直接调用符号：

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

实现说明：

> Use the installed strictly-convex solver; never widen physical bounds.

显式输出分支：

```python
None
quadprog.solve_qp(np.asarray(hessian, dtype=float), -linear, np.concatenate((rows.T, -rows.T), axis=1), np.r_[lower, -upper])[0]
None
```

读取的对象属性：

```python
np.any
quadprog.solve_qp
np.asarray
np.concatenate
rows.T
np.r_
```

写入目标：

```python
nonzero
rows
lower
upper
```

直接调用符号：

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
```

类字段：

无显式条目。

实现说明：

> Bind measured q0/dq0 to the shared instantaneous-boundary cubic.

## `CubicAffineMap.__init__`

[定义位置](../../../manipulation/physical_joint_qp.py#L49)。

输入签名：

```python
def __init__(self, start_q, start_dq, knots, knot_dt):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
_numpy(start_q).reshape
_numpy(start_dq).reshape
self.start_q.shape
self.start_q
self.start_dq.shape
self.start_dq
np.isfinite(self.start_q).all
np.isfinite
np.isfinite(self.start_dq).all
```

写入目标：

```python
self.start_q
self.start_dq
```

直接调用符号：

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
def coefficients(self, actions):
```

实现说明：

> Complete coefficients, computed relative to q0 before translation.

显式输出分支：

```python
self.start_q + np.einsum('cn,...nj->...cj', self.matrix, actions - self.start_q) + self.velocity_coefficients[:, None] * self.knot_dt * self.start_dq
```

读取的对象属性：

```python
self.start_q
np.einsum
self.matrix
self.velocity_coefficients
self.knot_dt
self.start_dq
```

写入目标：

```python
actions
```

直接调用符号：

```python
_numpy
np.einsum
```

## `CubicAffineMap.query`

[定义位置](../../../manipulation/physical_joint_qp.py#L62)。

输入签名：

```python
def query(self, progress, derivative=0):
```

显式输出分支：

```python
(weights, offset)
```

读取的对象属性：

```python
self.weights
self.start_dq
weights.sum
self.start_q
```

写入目标：

```python
weights
velocity
offset
```

直接调用符号：

```python
self.weights
_numpy
weights.sum
```

## `CubicAffineMap.evaluate`

[定义位置](../../../manipulation/physical_joint_qp.py#L69)。

输入签名：

```python
def evaluate(self, actions, progress, derivative=0):
```

显式输出分支：

```python
values + self.start_q if derivative == 0 else values
```

读取的对象属性：

```python
actions.shape
self.knots
self.start_q
self.weights
np.einsum
self.start_dq
```

写入目标：

```python
actions
weights
velocity
values
```

直接调用符号：

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
def extrema_times(self, controls, joint, derivative=0):
```

实现说明：

> Return boundaries and extrema of q or dq, in fixed-clock units.
>
> Query times use knot units. Values use the model's joint coordinates
> and seconds; velocity extrema are roots of the actual acceleration.

显式输出分支：

```python
(times, values)
```

读取的对象属性：

```python
self.matrix
self.start_q
self.velocity_coefficients
self.knot_dt
self.start_dq
PPoly.from_spline
self.knot_vector
np.flatnonzero
polynomial.x
polynomial.c
self.intervals
polynomial.derivative
polynomial.derivative().roots
np.isfinite
np.unique
np.r_
np.arange
```

写入目标：

```python
coefficients
polynomial
entrance
polynomial.c[-1, entrance]
polynomial.c[-2, entrance]
terminal
polynomial.c[:-1, terminal]
polynomial.c[-1, terminal]
roots
times
values
values[times == 0.0]
```

直接调用符号：

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
def velocity_extrema(self, controls):
```

实现说明：

> Analytic all-time velocity bounds for one [control,joint] action.

显式输出分支：

```python
np.asarray(limits).T
```

读取的对象属性：

```python
controls.shape
self.extrema_times
limits.append
values.min
values.max
np.asarray(limits).T
np.asarray
```

写入目标：

```python
controls
limits
joint
_
values
```

直接调用符号：

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
def solve_joint(self, hessian, linear, lower, upper, joint, *, origin, initial, native_dtype, step_limit=None, velocity_bounds=None, interior=False):
```

实现说明：

> Exchange violated analytic extrema for linear QP rows, at most 24 times.
>
> The optimization variable is a coefficient increment from ``origin``.
> For fitting, the origin is zero. For recovery it is the current legal
> action. Checking the rounded candidate, rather than the double QP
> answer, also exposes violations introduced by the actual action dtype.

显式输出分支：

```python
(None, {**report, 'reason': 'sampled_curve_qp_infeasible'})
(increment, {**report, 'reason': 'analytic_extrema_passed'})
(increment, {**report, 'reason': 'rounded_boundary_repeated'})
(increment, {**report, 'reason': 'extrema_exchange_limit'})
```

读取的对象属性：

```python
self.extrema_times
np.max
np.abs
np.finfo(native_dtype).eps
np.finfo
bounds.items
self.query
self.intervals
np.abs(weights).sum
row_parts.append
lower_parts.append
upper_parts.append
np.concatenate
np.r_
np.eye
self.knots
report.update
(origin.astype(native_dtype) + increment.astype(native_dtype)).astype
origin.astype
increment.astype
np.any
np.unique
```

写入目标：

```python
bounds
bounds[1]
times
derivative
coordinate_scale
roundoff
report
iteration
row_parts
lower_parts
upper_parts
lo_bound
hi_bound
weights
fixed
offset
margin
rows
lo
hi
increment
rounded
violation
added
query
values
bad
extended
times[derivative]
```

直接调用符号：

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

实现说明：

> Project a scaled negative gradient, retaining independent joint motion.
>
> Position constraints are separable by joint, so the small diagonal QPs
> are solved serially. The original native step limits bound *increments*,
> not cubic coefficients. The optimizer must separately demand actual cost
> decrease before accepting any returned direction.

显式输出分支：

```python
(zero, report)
(zero, {**report, 'reason': 'position_qp_infeasible'})
(zero, {**report, 'reason': 'no_representable_descent_direction'})
(direction, {**report, 'success': True, 'reason': 'physical_descent_direction'})
```

读取的对象属性：

```python
torch.zeros_like
base.ndim
base.shape
gradient.shape
domain.feasible(base).all
domain.feasible
domain.validate_scales
np.broadcast_to
x.shape
scales.max
np.isfinite(g).all
np.isfinite
np.isfinite(limits).all
np.any
np.max
np.abs
np.empty_like
base.detach().cpu().numpy().dtype
base.detach().cpu().numpy
base.detach().cpu
base.detach
domain.lower
domain.upper
np.eye
np.r_
curve.solve_joint
torch.as_tensor
base.dtype
base.device
torch.eye
domain.limit_step
base.expand
separate.to(torch.float64).flatten(1).norm
separate.to(torch.float64).flatten
separate.to
torch.float64
corrected.to(torch.float64).flatten(1).norm
corrected.to(torch.float64).flatten
corrected.to
torch.where
proposed_norms.clamp_min
torch.finfo(torch.float64).tiny
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

写入目标：

```python
zero
report
x
g
limits
maximum
normalization
target
increments
native_dtype
lower
upper
hessian
joint
rows
lo
hi
increment
detail
report['rounds']
report['constraint_count']
increments[:, joint]
proposed
dof
separate
corrected
proposed_norms
corrected_norms
joint_scales
combined
direction
actual
moved
derivative
norm
```

直接调用符号：

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

实现说明：

> Seed-only feasibility: the optimizer's position domain stays unchanged.

显式输出分支：

```python
False
True
bool(np.isfinite(extrema).all()) and bool(np.all(extrema[0] >= velocity_bounds[0])) and bool(np.all(extrema[1] <= velocity_bounds[1]))
```

读取的对象属性：

```python
domain.feasible(action[None]).all
domain.feasible
curve.velocity_extrema
np.isfinite(extrema).all
np.isfinite
np.all
```

写入目标：

```python
extrema
```

直接调用符号：

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

实现说明：

> Construct an anchor satisfying position and velocity on this clock.
>
> A position initializer can be reused only after the velocity check.
> Otherwise a small strictly convex QP constructs a numerical interior
> anchor for the same two domains; there is no position-only downgrade.

显式输出分支：

```python
(None, {'fit_anchor_source': 'not_constructed', 'detail': str(error)})
(anchor, {'fit_anchor_source': 'validated_position_initializer'})
(None, {'fit_anchor_source': 'not_constructed', 'detail': detail['reason']})
(None, {'fit_anchor_source': 'not_constructed', 'detail': 'rounded_anchor_outside_domain'})
(anchor, {'fit_anchor_source': 'position_velocity_qp'})
```

读取的对象属性：

```python
(domain._anchor if domain._anchor is not None else domain._construct_anchor(dtype, device))[0].to
domain._anchor
domain._construct_anchor
domain.lower
domain.upper
np.empty_like
anchor.detach().cpu().numpy().dtype
anchor.detach().cpu().numpy
anchor.detach().cpu
anchor.detach
initial.shape
curve.solve_joint
np.eye
domain.knots
np.zeros
torch.as_tensor
```

写入目标：

```python
anchor
initial
lower
upper
controls
native_dtype
joint
candidate
detail
controls[:, joint]
```

直接调用符号：

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

实现说明：

> Fit soft references, with optional model velocity bounds only for seeds.
>
> Every production caller passes the model's [lower,upper] velocity bounds.
> Position-only numerical users may omit them; a requested position-and-
> velocity initialization is never silently weakened on construction failure.

显式输出分支：

```python
(None, {**report, 'reason': 'fit_initializer_not_constructed'})
(None, {**report, 'reason': 'fit_initializer_not_constructed', 'detail': str(error)})
(None, {**report, 'reason': 'fit_initializer_not_constructed'})
(None, {**report, 'reason': 'rounded_fit_outside_position_velocity_domain'})
(action, {**report, 'success': True, 'reason': 'physical_reference_fit', 'reference_rms_joint_units': float(np.sqrt(np.mean(residual ** 2))), 'reference_max_error_joint_units': float(np.max(np.abs(residual))), 'query_max_abs_velocity_joint_units_per_s': np.max(np.abs(curve.evaluate(action, progress, 1)), axis=0).tolist(), 'continuous_max_abs_velocity_joint_units_per_s': np.max(np.abs(velocity_extrema), axis=0).tolist()})
```

读取的对象属性：

```python
curve.query
domain.knots
domain.lower
velocity_bounds.shape
np.isfinite(velocity_bounds).all
np.isfinite
np.any
values.shape
offset.shape
np.isfinite(values).all
np.zeros
prior.shape
np.isfinite(prior).all
torch.Tensor
source.dtype
torch.float32
source.device
domain.lower.device
torch.empty((), dtype=dtype).numpy().dtype
torch.empty((), dtype=dtype).numpy
torch.empty
weights.T
np.finfo(float).eps
np.finfo
np.trace
np.eye
np.empty
domain.upper
curve.solve_joint
report['joint_qp_status'].append
torch.as_tensor
report.update
domain.extrema
torch.maximum(domain.lower - low, high - domain.upper).max
torch.maximum
domain.initialize
domain._anchor
curve.velocity_extrema
np.max
(action - anchor).to(torch.float64).norm
(action - anchor).to
torch.float64
(initialized - anchor).to(torch.float64).norm
(initialized - anchor).to
curve.evaluate
np.sqrt
np.mean
np.abs
np.max(np.abs(curve.evaluate(action, progress, 1)), axis=0).tolist
np.max(np.abs(velocity_extrema), axis=0).tolist
```

写入目标：

```python
values
weights
offset
count
dof
velocity_bounds
prior
source
dtype
device
native_dtype
gram
ridge
hessian
linear
controls
report
lower
upper
joint
candidate
detail
report['rounds']
report['constraint_count']
controls[:, joint]
action
low
high
excess
initialized
anchor
report['fit_anchor_source']
velocity_extrema
report['maximum_velocity_bound_excess_before_contraction']
anchor_report
delta
low_scale
high_scale
_
middle
trial
proposed_norm
accepted_norm
residual
```

直接调用符号：

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
