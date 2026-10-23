# `manipulation/cubic_constraints.py` 实现契约

代码基线：`ae2603f`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_numpy`

[定义位置](../../../manipulation/cubic_constraints.py#L18)。

输入签名：

```python
def _numpy(value):
```

实现说明：

> Copy an external Torch value to the CPU numerical interface if needed.

显式输出分支：

```python
np.asarray(value, dtype=float)
```

读取的对象属性：

```python
torch.Tensor
value.detach().cpu().numpy
value.detach().cpu
value.detach
np.asarray
```

写入目标：

```python
value
```

直接调用符号：

```python
isinstance
value.detach().cpu().numpy
value.detach().cpu
value.detach
np.asarray
```

## `CubicAffineMap`

[定义位置](../../../manipulation/cubic_constraints.py#L25)。

```python
class CubicAffineMap(CubicBasis):
```

类字段：

无显式条目。

实现说明：

> Bind measured q0/dq0 to the shared fixed-clock cubic basis.

## `CubicAffineMap.__init__`

[定义位置](../../../manipulation/cubic_constraints.py#L29)。

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

[定义位置](../../../manipulation/cubic_constraints.py#L36)。

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

[定义位置](../../../manipulation/cubic_constraints.py#L42)。

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

[定义位置](../../../manipulation/cubic_constraints.py#L46)。

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

[定义位置](../../../manipulation/cubic_constraints.py#L51)。

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

[定义位置](../../../manipulation/cubic_constraints.py#L57)。

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

## `CubicAffineMap._segment_extrema`

[定义位置](../../../manipulation/cubic_constraints.py#L68)。

输入签名：

```python
def _segment_extrema(self, actions, derivative):
```

实现说明：

> Return per-segment minima/maxima and earliest active knot times.
>
> SciPy finds every interior root of the next derivative. Endpoints
> complete the extrema of each closed cubic segment. At a tie the
> earliest witness supplies one active subgradient; the function is
> piecewise differentiable and is not assigned a fictitious smooth
> derivative across active-witness changes.

显式输出分支：

```python
(values, times)
```

读取的对象属性：

```python
self.intervals
self.start_q
np.einsum
self.polynomial_weights
self.polynomial_velocity
self.start_dq
np.arange
polynomial.derivative(derivative + 1).roots
polynomial.derivative
self.weights_at_knot_times
np.stack
candidates.argmin
candidates.argmax
np.take_along_axis
np.isfinite
np.floor
```

写入目标：

```python
count
dof
coefficients
polynomial
roots_by_joint
endpoints
weights
velocity
endpoints_value
candidates
selected
values
times
joint
roots
root_values
moment
value
segment
side
improves
earlier_tie
values[segment, joint, side]
times[segment, joint, side]
```

直接调用符号：

```python
len
np.einsum
PPoly
np.arange
polynomial.derivative(derivative + 1).roots
polynomial.derivative
self.weights_at_knot_times
np.stack
candidates.argmin
candidates.argmax
np.take_along_axis
range
np.isfinite
np.floor
zip
int
enumerate
```

## `CubicAffineMap.signed_constraints`

[定义位置](../../../manipulation/cubic_constraints.py#L110)。

输入签名：

```python
def signed_constraints(self, actions, bounds, *, with_witnesses=False):
```

实现说明：

> Return exact continuous ``g <= 0`` and its selected Jacobian.
>
> ``actions`` has shape [free controls,joints], ``bounds`` is [4,2,joints],
> and Jacobian columns use C-order action flattening. Rows are ordered
> by derivative, segment, joint, then lower/upper side. Acceleration
> additionally has its left/right endpoint axis before the bound side.
> Position and velocity retain separate per-segment extrema. Their
> unique active extrema use the envelope derivative: stationarity
> removes the moving-root term. At tied extrema the earliest active
> branch gives a valid subgradient. Acceleration endpoint and constant
> segment jerk rows are affine, including both sides of jerk jumps.

显式输出分支：

```python
(*result, {name: np.concatenate([part[name] for part in witnesses]) for name in witnesses[0]})
result
```

读取的对象属性：

```python
self.start_q
actions.shape
self.knots
bounds.shape
self.intervals
np.empty
np.zeros
np.arange
self._segment_extrema
np.broadcast_to
np.array
np.indices
times.shape
indices[1].ravel
indices[-1].ravel
times.ravel
self.weights_at_knot_times
(weights * (actions - self.start_q)[:, joints].T).sum
(actions - self.start_q)[:, joints].T
self.start_dq
values.ravel
witnesses.append
np.full
indices[0].ravel
self.knot_dt
jacobian.reshape
np.concatenate
```

写入目标：

```python
actions
bounds
dof
row_count
residuals
jacobian
witnesses
offset
segments
derivative
values
times
indices
joints
sides
signs
flat_times
weights
velocity
row_indices
residuals[row_indices]
jacobian[row_indices, :, joints]
result
part
name
```

直接调用符号：

```python
_numpy
len
ValueError
np.empty
np.zeros
np.arange
range
self._segment_extrema
np.broadcast_to
np.array
np.indices
indices[1].ravel
indices[-1].ravel
times.ravel
self.weights_at_knot_times
(weights * (actions - self.start_q)[:, joints].T).sum
values.ravel
witnesses.append
np.full
indices[0].ravel
jacobian.reshape
np.concatenate
```

## `CubicAffineMap.fit_samples`

[定义位置](../../../manipulation/cubic_constraints.py#L173)。

输入签名：

```python
def fit_samples(self, progress, targets, prior_controls=None):
```

实现说明：

> Least-squares sample seed with q0/dq0 and clock already eliminated.
>
> This computes only an initial coefficient estimate. The complete
> physical, task and geometry constraints belong to the subsequent
> constrained solve. In an underdetermined fit, NumPy's minimum-norm
> correction retains the supplied prior in the observation nullspace.
> The report includes the same solve's singular values and spectral
> condition number; a zero smallest singular value is recorded as None.

显式输出分支：

```python
(controls, {'success': True, 'reason': 'unconstrained_sample_fit', 'rank': int(rank), 'sample_matrix_singular_values': singular.tolist(), 'sample_matrix_condition_number': condition, 'reference_rms_joint_units': float(np.sqrt(np.mean(residual ** 2))), 'reference_max_error_joint_units': float(np.max(np.abs(residual)))})
```

读取的对象属性：

```python
self.weights
targets.shape
self.start_q
np.isfinite(targets).all
np.isfinite
np.broadcast_to(self.start_q, (self.knots, len(self.start_q))).copy
np.broadcast_to
self.knots
prior.shape
np.isfinite(prior).all
self.start_dq
np.linalg.lstsq
np.linalg
self.evaluate
singular.tolist
np.sqrt
np.mean
np.max
np.abs
```

写入目标：

```python
targets
weights
velocity
prior
target_delta
correction
_
rank
singular
condition
controls
residual
```

直接调用符号：

```python
_numpy
self.weights
len
np.isfinite(targets).all
np.isfinite
ValueError
np.broadcast_to(self.start_q, (self.knots, len(self.start_q))).copy
np.broadcast_to
np.isfinite(prior).all
np.linalg.lstsq
float
self.evaluate
int
singular.tolist
np.sqrt
np.mean
np.max
np.abs
```
