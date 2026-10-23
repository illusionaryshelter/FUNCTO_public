# `manipulation/cubic_trajectory.py` 实现契约

代码基线：`ae2603f`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `CubicBasis`

[定义位置](../../../manipulation/cubic_trajectory.py#L12)。

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
> The M+2 complete coefficients span M-1 intervals. The final position,
> velocity and acceleration remain free unless task rows constrain them.
> Internally all actions are relative to q0, avoiding cancellation of absolute angles in
> derivative evaluation; the public action remains a position coefficient.

## `CubicBasis.__init__`

[定义位置](../../../manipulation/cubic_trajectory.py#L23)。

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
PPoly.from_spline(BSpline(self.knot_vector, column, 3)).c
PPoly.from_spline
self.intervals
self.matrix.T
np.stack
PPoly.from_spline(self.velocity_basis).c
self.velocity_basis
self.knot_dt
self.polynomial_weights
self.polynomial_velocity
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
self.velocity_coefficients
self.velocity_coefficients[:2]
self.basis
self.velocity_basis
columns
column
self.polynomial_weights
self.polynomial_velocity
self.polynomial_weights[-2:, 0]
self.polynomial_velocity[-2:, 0]
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
PPoly.from_spline
np.stack
```

## `CubicBasis.reference_parameters`

[定义位置](../../../manipulation/cubic_trajectory.py#L53)。

输入签名：

```python
def reference_parameters(self):
```

实现说明：

> Observe every noninitial knot and the final interval midpoint.
>
> The measured q0/dq0 are already eliminated from the coefficients.
> The M-2 interior knot positions leave two free coefficient directions.
> The final interval midpoint and endpoint provide independent
> observations of those directions, retaining free terminal velocity
> and acceleration. All M queries are at positive times. Their
> spacing follows the spline support rather than a shifted uniform
> observation grid whose terminal amplification grows with M.

显式输出分支：

```python
times / self.intervals
```

读取的对象属性：

```python
np.r_
np.arange
self.intervals
```

写入目标：

```python
times
```

直接调用符号：

```python
np.arange
```

## `CubicBasis.weights`

[定义位置](../../../manipulation/cubic_trajectory.py#L69)。

输入签名：

```python
def weights(self, progress, derivative=0):
```

实现说明：

> Return action and entrance-velocity weights in physical seconds.
>
> q=q0+W@(x-q0)+V*dq0. For derivative>0 the independent q0 term is
> absent. Exact entrance rows encode the two measured equalities.

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

[定义位置](../../../manipulation/cubic_trajectory.py#L82)。

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

## `CubicBasis.hermite_nodes`

[定义位置](../../../manipulation/cubic_trajectory.py#L99)。

输入签名：

```python
def hermite_nodes(self, actions, start_q, start_dq):
```

实现说明：

> Return binary64 time, position and velocity at every original break.
>
> ``actions`` is one [free controls, named joints] array; the two entry
> vectors use the same joint order. A cubic segment is uniquely fixed
> by its endpoint positions and velocities. Using its original breaks
> therefore preserves that segment without rebuilding many short
> polynomials from independently rounded native state samples.

显式输出分支：

```python
(knots * self.knot_dt, position, velocity)
```

读取的对象属性：

```python
np.asarray
np.float64
actions.ndim
actions.shape
self.knots
start_q.shape
start_dq.shape
np.isfinite(value).all
np.isfinite
np.arange
self.intervals
self.weights_at_knot_times
self.knot_dt
```

写入目标：

```python
actions
start_q
start_dq
value
knots
delta
position_weights
position_entry
velocity_weights
velocity_entry
position
velocity
```

直接调用符号：

```python
np.asarray
all
np.isfinite(value).all
np.isfinite
ValueError
np.arange
self.weights_at_knot_times
```
