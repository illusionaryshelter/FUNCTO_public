# `manipulation/physical_joint_qp.py` 实现契约

代码基线：`a56c3e4`。
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

[定义位置](../../../manipulation/physical_joint_qp.py#L28)。

输入签名：

```python
def _solve_qp(hessian, linear, rows, lower, upper):
```

实现说明：

> Solve two-sided linear bounds in the free-coefficient space.

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

[定义位置](../../../manipulation/physical_joint_qp.py#L47)。

```python
class CubicAffineMap(CubicBasis):
```

类字段：

无显式条目。

实现说明：

> Bind measured q0/dq0 to the shared instantaneous-boundary cubic.

## `CubicAffineMap.__init__`

[定义位置](../../../manipulation/physical_joint_qp.py#L51)。

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

[定义位置](../../../manipulation/physical_joint_qp.py#L58)。

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

[定义位置](../../../manipulation/physical_joint_qp.py#L64)。

输入签名：

```python
def query(self, progress, derivative=0):
```

显式输出分支：

```python
self._affine_query(weights, velocity, derivative)
```

读取的对象属性：

```python
self.weights
self._affine_query
```

写入目标：

```python
weights
velocity
```

直接调用符号：

```python
self.weights
_numpy
self._affine_query
```

## `CubicAffineMap.query_knot_times`

[定义位置](../../../manipulation/physical_joint_qp.py#L68)。

输入签名：

```python
def query_knot_times(self, times, derivative=0):
```

实现说明：

> Build rows at original segment coordinates, including jerk sides.

显式输出分支：

```python
self._affine_query(weights, velocity, derivative)
```

读取的对象属性：

```python
self.weights_at_knot_times
self._affine_query
```

写入目标：

```python
weights
velocity
```

直接调用符号：

```python
self.weights_at_knot_times
_numpy
self._affine_query
```

## `CubicAffineMap._affine_query`

[定义位置](../../../manipulation/physical_joint_qp.py#L73)。

输入签名：

```python
def _affine_query(self, weights, velocity, derivative):
```

显式输出分支：

```python
(weights, offset)
```

读取的对象属性：

```python
self.start_dq
weights.sum
self.start_q
```

写入目标：

```python
offset
```

直接调用符号：

```python
weights.sum
```

## `CubicAffineMap.evaluate`

[定义位置](../../../manipulation/physical_joint_qp.py#L79)。

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

[定义位置](../../../manipulation/physical_joint_qp.py#L89)。

输入签名：

```python
def extrema_times(self, controls, joint, derivative=0):
```

实现说明：

> Return exact bounds witnesses for derivative 0..3, in knot units.
>
> Query times use knot units. Values use the model's joint coordinates
> and seconds. Jerk is constant on each segment, so its midpoint
> supplies every segment value without losing a side of a knot jump.

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
np.arange
polynomial.derivative().roots
np.isfinite
np.unique
np.r_
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
times
roots
values
values[times == 0.0]
```

直接调用符号：

```python
PPoly.from_spline
BSpline
np.flatnonzero
polynomial.derivative
np.arange
polynomial.derivative().roots
np.isfinite
np.unique
polynomial
```

## `CubicAffineMap.solve_joint`

[定义位置](../../../manipulation/physical_joint_qp.py#L125)。

输入签名：

```python
def solve_joint(self, hessian, linear, bounds, joint, *, origin, initial, native_dtype, step_limit=None, interior=False):
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
(None, {**report, 'reason': 'finite_constraints_inconsistent'})
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
self.query_knot_times
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
self.extrema_times
range
max
float
np.max
np.abs
np.finfo
enumerate
self.query_knot_times
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

[定义位置](../../../manipulation/physical_joint_qp.py#L185)。

输入签名：

```python
def recover_position_step(domain, base, gradient, step_limits, scales, curve=None):
```

实现说明：

> Project a scaled negative gradient, retaining independent joint motion.
>
> Physical constraints are separable by joint, so the small diagonal QPs
> are solved serially. The original native step limits bound *increments*,
> not cubic coefficients. The optimizer must separately demand actual cost
> decrease before accepting any returned direction.

显式输出分支：

```python
(zero, report)
(zero, {**report, 'reason': 'recovery_qp_not_constructed', 'joint_column': joint, 'joint_qp_reason': detail['reason']})
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
domain.bounds
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
bounds
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

## `construct_cubic_anchor`

[定义位置](../../../manipulation/physical_joint_qp.py#L273)。

输入签名：

```python
def construct_cubic_anchor(domain, curve, dtype, device):
```

实现说明：

> Construct a native-dtype feasible curve using the shared four-order QP.
>
> This construction has no fit/initialize dependency. The position-valued
> constant proposal supplies a finite objective centre; the measured q0,
> dq0, stopped terminal interval and fixed clock remain in the affine map.
> A construction margin accounts for coefficient rounding only. Every
> returned action is checked against the original model bounds.

显式输出分支：

```python
(None, {**report, 'reason': detail['reason'], 'joint_column': joint})
(None, {**report, 'reason': reason})
(anchor, {**report, 'reason': 'physical_anchor_constructed'})
```

读取的对象属性：

```python
np.broadcast_to
domain._start_q
domain.knots
domain.lower
np.empty_like
torch.empty((), dtype=dtype).numpy().dtype
torch.empty((), dtype=dtype).numpy
torch.empty
domain.bounds
initial.shape
curve.solve_joint
np.eye
np.zeros
report['joint_qp_status'].append
torch.as_tensor
domain.feasible(anchor).all
domain.feasible
```

写入目标：

```python
initial
controls
native_dtype
bounds
report
joint
candidate
detail
report['rounds']
report['constraint_count']
controls[:, joint]
anchor
reason
```

直接调用符号：

```python
np.broadcast_to
_numpy
len
np.empty_like
torch.empty((), dtype=dtype).numpy
torch.empty
range
curve.solve_joint
np.eye
np.zeros
max
report['joint_qp_status'].append
torch.as_tensor
bool
domain.feasible(anchor).all
domain.feasible
```

## `fit_cubic_reference`

[定义位置](../../../manipulation/physical_joint_qp.py#L308)。

输入签名：

```python
def fit_cubic_reference(domain, curve, progress, targets, prior_controls=None):
```

实现说明：

> Fit soft references while preserving the measured entrance and clock.
>
> The complete bound declaration belongs to ``domain``. Native-dtype
> contraction uses its ordinary line limiter, so fitting, initialization
> and recovery cannot accept different derivative domains.

显式输出分支：

```python
(None, {**report, 'reason': 'algebraic_entrance_invalid'})
(None, {**report, 'reason': detail['reason'], 'joint_column': joint})
(None, {**report, 'reason': 'fit_initializer_not_constructed', 'detail': str(error)})
(action, {**report, 'success': True, 'reason': 'physical_reference_fit', 'reference_rms_joint_units': float(np.sqrt(np.mean(residual ** 2))), 'reference_max_error_joint_units': float(np.max(np.abs(residual))), 'continuous_max_abs_derivatives': torch.maximum(low.abs(), high.abs())[0].cpu().tolist(), 'query_max_abs_velocity_joint_units_per_s': np.max(np.abs(curve.evaluate(action, progress, 1)), axis=0).tolist(), 'continuous_max_abs_velocity_joint_units_per_s': torch.maximum(low.abs(), high.abs())[0, 1].cpu().tolist()})
```

读取的对象属性：

```python
curve.query
domain.knots
domain.lower
values.shape
offset.shape
np.isfinite(values).all
np.isfinite
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
domain._entrance_valid
domain.bounds
curve.solve_joint
report['joint_qp_status'].append
torch.as_tensor
report.update
domain.feasible(action[None]).all
domain.feasible
domain.all_extrema
torch.maximum
domain._anchor
domain._construct_anchor
domain._anchor.to
domain.validate
domain.limit_step
action.new_ones
delta.to(torch.float64).norm
delta.to
torch.float64
limited.to(torch.float64).norm
limited.to
excess.clamp_min(0).amax(dim=(0, 2)).cpu().tolist
excess.clamp_min(0).amax(dim=(0, 2)).cpu
excess.clamp_min(0).amax
excess.clamp_min
curve.evaluate
np.sqrt
np.mean
np.max
np.abs
torch.maximum(low.abs(), high.abs())[0].cpu().tolist
torch.maximum(low.abs(), high.abs())[0].cpu
low.abs
high.abs
np.max(np.abs(curve.evaluate(action, progress, 1)), axis=0).tolist
torch.maximum(low.abs(), high.abs())[0, 1].cpu().tolist
torch.maximum(low.abs(), high.abs())[0, 1].cpu
```

写入目标：

```python
values
weights
offset
count
dof
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
bounds
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
domain._anchor
anchor
delta
limited
initialized
proposed_norm
accepted_norm
residual
```

直接调用符号：

```python
_numpy
curve.query
len
np.isfinite(values).all
np.isfinite
ValueError
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
bool
range
curve.solve_joint
report['joint_qp_status'].append
torch.as_tensor
report.update
domain.feasible(action[None]).all
domain.feasible
domain.all_extrema
torch.maximum
domain._construct_anchor
domain._anchor.to
domain.validate
domain.limit_step
action.new_ones
str
delta.to(torch.float64).norm
delta.to
limited.to(torch.float64).norm
limited.to
excess.clamp_min(0).amax(dim=(0, 2)).cpu().tolist
excess.clamp_min(0).amax(dim=(0, 2)).cpu
excess.clamp_min(0).amax
excess.clamp_min
curve.evaluate
np.sqrt
np.mean
np.max
np.abs
torch.maximum(low.abs(), high.abs())[0].cpu().tolist
torch.maximum(low.abs(), high.abs())[0].cpu
low.abs
high.abs
np.max(np.abs(curve.evaluate(action, progress, 1)), axis=0).tolist
torch.maximum(low.abs(), high.abs())[0, 1].cpu().tolist
torch.maximum(low.abs(), high.abs())[0, 1].cpu
```
