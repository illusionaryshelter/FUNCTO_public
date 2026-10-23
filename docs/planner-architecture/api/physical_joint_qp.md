# `manipulation/physical_joint_qp.py` 实现契约

代码基线：`82a3d66；包含本次工作区已实现改动，待统一实现提交后更新`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `program_work_report`

[定义位置](../../../manipulation/physical_joint_qp.py#L29)。

输入签名：

```python
def program_work_report():
```

实现说明：

> Start counters for actual library calls and measured computation phases.

显式输出分支：

```python
{name: 0 if name.endswith('calls') else 0.0 for name in PROGRAM_WORK_FIELDS}
```

读取的对象属性：

```python
name.endswith
```

写入目标：

```python
name
```

直接调用符号：

```python
name.endswith
```

## `_numpy`

[定义位置](../../../manipulation/physical_joint_qp.py#L34)。

输入签名：

```python
def _numpy(value, report=None):
```

显式输出分支：

```python
np.asarray(value, dtype=float)
result
```

读取的对象属性：

```python
torch.Tensor
np.asarray
value.detach().cpu().numpy().astype
value.detach().cpu().numpy
value.detach().cpu
value.detach
```

写入目标：

```python
started
result
report['host_transfer_seconds']
```

直接调用符号：

```python
isinstance
np.asarray
perf_counter
value.detach().cpu().numpy().astype
value.detach().cpu().numpy
value.detach().cpu
value.detach
```

## `_tensor`

[定义位置](../../../manipulation/physical_joint_qp.py#L44)。

输入签名：

```python
def _tensor(value, *, dtype, device, report):
```

实现说明：

> Measure the explicit NumPy-to-native-array handoff, including dtype copy.

显式输出分支：

```python
result
```

读取的对象属性：

```python
torch.as_tensor
```

写入目标：

```python
started
result
report['host_transfer_seconds']
```

直接调用符号：

```python
perf_counter
torch.as_tensor
```

## `_solve_qp`

[定义位置](../../../manipulation/physical_joint_qp.py#L54)。

输入签名：

```python
def _solve_qp(hessian, linear, rows, lower, upper, *, report, check_request=None):
```

实现说明：

> Solve two-sided linear bounds in the free-coefficient space.

显式输出分支：

```python
None
result
```

读取的对象属性：

```python
np.any
np.concatenate
rows.T
np.r_
quadprog.solve_qp
np.asarray
```

写入目标：

```python
started
nonzero
report['assembly_seconds']
rows
lower
upper
matrix
rhs
report['qp_calls']
result
report['qp_solve_seconds']
```

直接调用符号：

```python
perf_counter
np.any
np.concatenate
check_request
quadprog.solve_qp
np.asarray
str
```

## `CubicAffineMap`

[定义位置](../../../manipulation/physical_joint_qp.py#L89)。

```python
class CubicAffineMap(CubicBasis):
```

类字段：

无显式条目。

实现说明：

> Bind measured q0/dq0 to the shared instantaneous-boundary cubic.

## `CubicAffineMap.__init__`

[定义位置](../../../manipulation/physical_joint_qp.py#L93)。

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
np.arange
self.intervals
self.query_knot_times
self.fixed_constraint_times
```

写入目标：

```python
self.start_q
self.start_dq
self.fixed_constraint_times
order
self.fixed_constraint_rows
times
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
tuple
np.arange
range
self.query_knot_times
enumerate
```

## `CubicAffineMap.coefficients`

[定义位置](../../../manipulation/physical_joint_qp.py#L109)。

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

[定义位置](../../../manipulation/physical_joint_qp.py#L115)。

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

[定义位置](../../../manipulation/physical_joint_qp.py#L119)。

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

[定义位置](../../../manipulation/physical_joint_qp.py#L124)。

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

[定义位置](../../../manipulation/physical_joint_qp.py#L130)。

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

[定义位置](../../../manipulation/physical_joint_qp.py#L140)。

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

## `CubicAffineMap._initial_constraint_rows`

[定义位置](../../../manipulation/physical_joint_qp.py#L177)。

输入签名：

```python
def _initial_constraint_rows(self, initial, bounds, joint, report):
```

实现说明：

> Combine bound-time rows with this starting curve's extra extrema.

显式输出分支：

```python
result
```

读取的对象属性：

```python
self.extrema_times
self.fixed_constraint_times
self.fixed_constraint_rows
np.setdiff1d
self.query_knot_times
np.r_
np.argsort
np.concatenate
known.append
matrices.append
lower.append
upper.append
```

写入目标：

```python
known
matrices
lower
upper
order
lo_bound
hi_bound
started
query
_
report['extrema_seconds']
fixed_times
weights
fixed
extra
times
offset
new_weights
new_fixed
indices
report['assembly_seconds']
result
```

直接调用符号：

```python
enumerate
perf_counter
self.extrema_times
np.setdiff1d
len
self.query_knot_times
np.argsort
np.concatenate
known.append
matrices.append
lower.append
upper.append
```

## `CubicAffineMap._violated_constraint_rows`

[定义位置](../../../manipulation/physical_joint_qp.py#L206)。

输入签名：

```python
def _violated_constraint_rows(self, controls, bounds, joint, known, report):
```

实现说明：

> Check the actual curve and assemble only previously absent bad extrema.

显式输出分支：

```python
(violation, added)
```

读取的对象属性：

```python
self.extrema_times
np.any
np.setdiff1d
self.query_knot_times
matrices.append
lower.append
upper.append
np.r_
np.concatenate
```

写入目标：

```python
violation
matrices
lower
upper
order
lo_bound
hi_bound
started
query
values
report['extrema_seconds']
bad
extra
weights
fixed
known[order]
report['assembly_seconds']
added
```

直接调用符号：

```python
enumerate
perf_counter
self.extrema_times
bool
np.any
np.setdiff1d
len
self.query_knot_times
matrices.append
lower.append
upper.append
np.concatenate
```

## `CubicAffineMap.solve_joint`

[定义位置](../../../manipulation/physical_joint_qp.py#L231)。

输入签名：

```python
def solve_joint(self, hessian, linear, bounds, joint, *, origin, initial, native_dtype, step_limit=None, check_request=None, work_report=None):
```

实现说明：

> Exchange violated analytic extrema for linear QP rows, at most 24 times.
>
> The optimization variable is a coefficient increment from ``origin``.
> For fitting, the origin is zero. For descent it is the current legal
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
report.update
self._initial_constraint_rows
np.r_
np.eye
self.knots
(origin.astype(native_dtype) + increment.astype(native_dtype)).astype
origin.astype
increment.astype
self._violated_constraint_rows
np.concatenate
```

写入目标：

```python
report
known
rows
lo
hi
started
offset
report['assembly_seconds']
iteration
increment
rounded
violation
added
new_rows
new_lo
new_hi
```

直接调用符号：

```python
report.update
program_work_report
self._initial_constraint_rows
perf_counter
np.eye
range
int
len
_solve_qp
(origin.astype(native_dtype) + increment.astype(native_dtype)).astype
origin.astype
increment.astype
self._violated_constraint_rows
np.concatenate
```

## `projected_descent_step`

[定义位置](../../../manipulation/physical_joint_qp.py#L279)。

输入签名：

```python
def projected_descent_step(domain, base, gradient, step_limits, scales, curve=None, *, check_request=None, work_report=None):
```

实现说明：

> Project the complete negative gradient into the physical displacement set.
>
> Physical constraints are separable by joint, so the small diagonal QPs
> are solved serially. The original native step limits bound *increments*,
> not cubic coefficients. The optimizer must separately demand actual cost
> decrease before accepting any returned direction.

显式输出分支：

```python
(zero, report)
(zero, {**report, 'reason': 'direction_qp_not_constructed', 'joint_column': joint, 'joint_qp_reason': detail['reason']})
(zero, {**report, 'reason': 'no_representable_descent_direction'})
(direction, {**report, 'success': True, 'reason': 'physical_descent_direction'})
```

读取的对象属性：

```python
torch.zeros_like
report.update
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
np.empty_like
torch.empty((), dtype=base.dtype).numpy().dtype
torch.empty((), dtype=base.dtype).numpy
torch.empty
base.dtype
domain.lower
domain.upper
domain.bounds
np.eye
np.r_
detail.update
curve.solve_joint
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
target
increments
native_dtype
lower
upper
bounds
hessian
joint
detail
detail['rounds']
started
rows
lo
hi
detail['assembly_seconds']
increment
detail['reason']
outcome
report['rounds']
report['constraint_count']
name
report[name]
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
report.update
program_work_report
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
np.empty_like
torch.empty((), dtype=base.dtype).numpy
torch.empty
np.eye
range
perf_counter
detail.update
int
len
_solve_qp
curve.solve_joint
max
_tensor
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
direction.to(torch.float64).norm
joint_scales.cpu().tolist
joint_scales.cpu
```

## `construct_cubic_anchor`

[定义位置](../../../manipulation/physical_joint_qp.py#L383)。

输入签名：

```python
def construct_cubic_anchor(domain, curve, dtype, device, *, check_request=None, report=None):
```

实现说明：

> Maximize coefficient-space room inside the original physical bounds.
>
> Each original row ``a @ C <= b`` becomes ``a @ C + ||a||_1*r <= b``
> with a shared radius r >= 0 for one joint. Maximizing r selects a centre
> that is less sensitive to coefficient rounding. The radius is an objective
> variable, not a required margin: r=0 retains every original boundary
> solution, including domains with no interior. The measured q0/dq0 and
> terminal stop remain fixed by the cubic map. Violated analytic extrema
> add rows, and actual action-dtype curves use the unchanged domain test.

显式输出分支：

```python
(None, {**report, 'reason': reason, 'joint_column': joint, 'linear_program_status': int(result.status)})
(None, {**report, 'reason': reason, 'joint_column': joint})
(None, {**report, 'reason': 'rounded_anchor_outside_domain'})
(anchor, {**report, 'reason': 'physical_anchor_constructed'})
```

读取的对象属性：

```python
report.update
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
curve._initial_constraint_rows
np.abs(rows).sum
np.abs
np.r_
np.c_
np.zeros
result.success
result.status
result.x[:-1].astype
result.x
initial.copy
domain.all_extrema
((low[0, :, joint] >= domain.bounds[:, 0, joint]) & (high[0, :, joint] <= domain.bounds[:, 1, joint])).all
curve._violated_constraint_rows
rounded.astype
np.abs(new_rows).sum
report['joint_status'].append
report['control_radius'].append
domain.feasible(anchor).all
domain.feasible
```

写入目标：

```python
report
initial
controls
native_dtype
bounds
joint
known
rows
lower
upper
started
norm
matrix
rhs
objective
variable_bounds
report['assembly_seconds']
previous_constraints
reason
iteration
report['rounds']
report['constraint_count']
report['lp_calls']
result
report['lp_solve_seconds']
rounded
trial
trial[:, joint]
action
low
high
valid
report['extrema_seconds']
_
added
new_rows
new_lower
new_upper
controls[:, joint]
anchor
```

直接调用符号：

```python
program_work_report
report.update
np.broadcast_to
_numpy
len
np.empty_like
torch.empty((), dtype=dtype).numpy
torch.empty
range
curve._initial_constraint_rows
perf_counter
np.abs(rows).sum
np.abs
np.zeros
max
check_request
linprog
int
result.x[:-1].astype
initial.copy
_tensor
domain.all_extrema
bool
((low[0, :, joint] >= domain.bounds[:, 0, joint]) & (high[0, :, joint] <= domain.bounds[:, 1, joint])).all
curve._violated_constraint_rows
rounded.astype
np.abs(new_rows).sum
report['joint_status'].append
report['control_radius'].append
float
domain.feasible(anchor).all
domain.feasible
```

## `fit_cubic_reference`

[定义位置](../../../manipulation/physical_joint_qp.py#L473)。

输入签名：

```python
def fit_cubic_reference(domain, curve, progress, targets, prior_controls=None, *, check_request=None):
```

实现说明：

> Fit soft references while preserving the measured entrance and clock.
>
> The complete bound declaration belongs to ``domain``. Native-dtype
> contraction uses its ordinary line limiter, so fitting, initialization
> and descent cannot accept different derivative domains.

显式输出分支：

```python
(None, {**report, 'reason': 'algebraic_entrance_invalid'})
(None, {**report, 'reason': detail['reason'], 'joint_column': joint})
(None, {**report, 'reason': 'fit_initializer_not_constructed', 'detail': str(error)})
(action, {**report, 'success': True, 'reason': 'physical_reference_fit', 'reference_rms_joint_units': float(np.sqrt(np.mean(residual ** 2))), 'reference_max_error_joint_units': float(np.max(np.abs(residual))), 'continuous_max_abs_derivatives': torch.maximum(low.abs(), high.abs())[0].cpu().tolist(), 'query_max_abs_velocity_joint_units_per_s': np.max(np.abs(curve.evaluate(numeric_action, progress, 1)), axis=0).tolist(), 'continuous_max_abs_velocity_joint_units_per_s': torch.maximum(low.abs(), high.abs())[0, 1].cpu().tolist()})
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
report.update
domain.feasible(action[None]).all
domain.feasible
domain.all_extrema
torch.maximum
domain._anchor
domain._construct_anchor
domain.initialization_report
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
np.max(np.abs(curve.evaluate(numeric_action, progress, 1)), axis=0).tolist
torch.maximum(low.abs(), high.abs())[0, 1].cpu().tolist
torch.maximum(low.abs(), high.abs())[0, 1].cpu
```

写入目标：

```python
report
values
started
weights
offset
report['assembly_seconds']
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
bounds
joint
candidate
detail
report['rounds']
report['constraint_count']
name
report[name]
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
numeric_action
residual
```

直接调用符号：

```python
program_work_report
_numpy
perf_counter
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
_tensor
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
np.max(np.abs(curve.evaluate(numeric_action, progress, 1)), axis=0).tolist
torch.maximum(low.abs(), high.abs())[0, 1].cpu().tolist
torch.maximum(low.abs(), high.abs())[0, 1].cpu
```
