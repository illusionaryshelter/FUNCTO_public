# `manipulation/geometry_motion_bound.py` 实现契约

代码基线：`CURRENT_IMPLEMENTATION`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_finite`

[定义位置](../../../manipulation/geometry_motion_bound.py#L19)。

输入签名：

```python
def _finite(value):
```

显式输出分支：

```python
value
```

读取的对象属性：

```python
math.isfinite
```

写入目标：

无显式条目。

直接调用符号：

```python
math.isfinite
ValueError
```

## `_add_up`

[定义位置](../../../manipulation/geometry_motion_bound.py#L25)。

输入签名：

```python
def _add_up(first, second):
```

实现说明：

> Round a nonnegative sum outwards, preserving exact structural zeros.

显式输出分支：

```python
second
first
_finite(math.nextafter(_finite(first + second), math.inf))
```

读取的对象属性：

```python
math.nextafter
math.inf
```

写入目标：

无显式条目。

直接调用符号：

```python
_finite
math.nextafter
```

## `_mul_up`

[定义位置](../../../manipulation/geometry_motion_bound.py#L34)。

输入签名：

```python
def _mul_up(first, second):
```

显式输出分支：

```python
0.0
_finite(math.nextafter(_finite(first * second), math.inf))
```

读取的对象属性：

```python
math.nextafter
math.inf
```

写入目标：

无显式条目。

直接调用符号：

```python
_finite
math.nextafter
```

## `_norm_up`

[定义位置](../../../manipulation/geometry_motion_bound.py#L40)。

输入签名：

```python
def _norm_up(vector):
```

显式输出分支：

```python
0.0 if squared == 0 else _finite(math.nextafter(math.sqrt(squared), math.inf))
```

读取的对象属性：

```python
math.nextafter
math.sqrt
math.inf
```

写入目标：

```python
squared
value
```

直接调用符号：

```python
_add_up
_mul_up
abs
float
_finite
math.nextafter
math.sqrt
```

## `_product_interval`

[定义位置](../../../manipulation/geometry_motion_bound.py#L47)。

输入签名：

```python
def _product_interval(first, second):
```

实现说明：

> Enclose an exact product of two stored binary64 inputs.

显式输出分支：

```python
(0.0, 0.0)
(math.nextafter(value, -math.inf), math.nextafter(value, math.inf))
```

读取的对象属性：

```python
math.nextafter
math.inf
```

写入目标：

```python
value
```

直接调用符号：

```python
_finite
math.nextafter
```

## `_linear_norm_up`

[定义位置](../../../manipulation/geometry_motion_bound.py#L55)。

输入签名：

```python
def _linear_norm_up(matrix):
```

实现说明：

> Bound ||A||_2 by sqrt(||A.T A||_infinity), with outward arithmetic.
>
> Pinocchio placements and calibration store binary64 rotations whose
> entries need not be exactly orthogonal. Assuming norm one can underestimate the
> bound. Signed intervals retain cancellation in the Gram matrix, so this
> remains close to one for rotations instead of multiplying sqrt(3) per link.

显式输出分支：

```python
0.0 if maximum == 0 else _finite(math.nextafter(math.sqrt(maximum), math.inf))
```

读取的对象属性：

```python
np.zeros
math.nextafter
math.inf
math.sqrt
```

写入目标：

```python
gram_abs
i
j
lower
upper
row
lo
hi
gram_abs[i, j]
maximum
total
value
```

直接调用符号：

```python
np.zeros
range
_product_interval
float
_finite
math.nextafter
max
abs
_add_up
math.sqrt
```

## `_array`

[定义位置](../../../manipulation/geometry_motion_bound.py#L82)。

输入签名：

```python
def _array(value, label):
```

显式输出分支：

```python
result.astype(np.float64, copy=True)
```

读取的对象属性：

```python
np.asarray
result.dtype.kind
result.dtype
np.isfinite(result).all
np.isfinite
result.astype
np.float64
```

写入目标：

```python
device
result
```

直接调用符号：

```python
getattr
str
ValueError
np.asarray
np.isfinite(result).all
np.isfinite
result.astype
```

## `_names`

[定义位置](../../../manipulation/geometry_motion_bound.py#L96)。

输入签名：

```python
def _names(values, label):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
value.strip
```

写入目标：

```python
result
value
```

直接调用符号：

```python
tuple
len
set
any
isinstance
value.strip
ValueError
```

## `SphereBody`

[定义位置](../../../manipulation/geometry_motion_bound.py#L105)。

```python
class SphereBody():
```

类字段：

```python
name: str
link: str
spheres: object
```

实现说明：

> One complete rigid sphere cover in an existing Pinocchio frame.
>
> ``spheres`` has shape [S,4], with local centres and nonnegative radii.
> All robot and finger motion belongs to the same registered joint tree;
> attached observed fields retain their declared rigid tool relation.

## `GeometryMotionBound`

[定义位置](../../../manipulation/geometry_motion_bound.py#L119)。

```python
class GeometryMotionBound():
```

类字段：

```python
body_names: tuple
joint_names: tuple
coefficients: np.ndarray
joint_position_limits: np.ndarray
local_radius_bounds_m: np.ndarray
```

实现说明：

> L[g,j] bounds point speed by sum_j L[g,j] * abs(dq[j]).

## `GeometryMotionBound.speed_bounds`

[定义位置](../../../manipulation/geometry_motion_bound.py#L128)。

输入签名：

```python
def speed_bounds(self, joint_speed_upper):
```

实现说明：

> Return one m/s bound per body for complete named |dq| bounds.
>
> Speeds must cover every registered robot coordinate. A missing timed
> finger or unbound joint is an error, not an inferred zero velocity.
> The declared position ranges used to build L must hold on the interval.

显式输出分支：

```python
result
```

读取的对象属性：

```python
self.joint_names
speeds.shape
np.any
np.zeros
self.body_names
self.coefficients
```

写入目标：

```python
speeds
name
result
i
row
coefficient
speed
result[i]
```

直接调用符号：

```python
isinstance
set
ValueError
_array
len
np.any
np.zeros
enumerate
zip
_add_up
float
_mul_up
```

## `build_geometry_motion_bound`

[定义位置](../../../manipulation/geometry_motion_bound.py#L149)。

输入签名：

```python
def build_geometry_motion_bound(*, model, T_reference_base, bodies):
```

实现说明：

> Bound point motion in the calibrated reference frame using Pinocchio.
>
> ``model`` is the exact fixed-base model admitted by RobotGeometryChecker:
> independent bounded scalar revolute/prismatic joints and fixed frames.
> ``T_reference_base`` is that checker's calibration from robot base to the
> geometry reference frame. ``bodies`` bind full local covers to existing
> model frames; the native collision entities themselves remain unchanged.
>
> Pinocchio ``frame.placement`` maps a frame into its parent joint, while
> ``jointPlacements[j]`` maps joint j's zero pose into its parent joint.
> ``joint.calc`` supplies each admitted joint's constant motion subspace
> S=[v;w], with linear rows first. This reads analytic joint metadata at
> neutral configuration; it does not estimate a velocity maximum by FK
> samples. For a local full-cover radius r, the joint contributes
> ||v|| + ||w|| r to its own coordinate. Prismatic motion also increases r
> by ||v|| max(|q_min|, |q_max|) before the ancestor's lever is bounded.
>
> Starting with the frame placement, walk these exact parent relations.
> Each fixed rotation's outward operator-norm bound scales all accumulated
> coefficients and the radius; its translation enlarges only the radius.
> The final calibration rotation scales the whole coefficient row. Its
> fixed translation has no velocity contribution. These triangle-inequality
> bounds cover every sphere point throughout the declared joint domain.

显式输出分支：

```python
GeometryMotionBound(body_names, names, coefficients, limits, radii)
```

读取的对象属性：

```python
model.names
model.joints
model.njoints
joint.idx_q
np.array
model.lowerPositionLimit
model.upperPositionLimit
pin.neutral
joint.createData
joint.calc
np.asarray(data.S).reshape
np.asarray
data.S
gains.append
placement.rotation
model.jointPlacements
placement.translation
body.name
np.zeros
model.getFrameId
body.link
model.nframes
body.spheres
spheres.ndim
spheres.shape
np.any
model.frames
frame.placement.rotation
frame.placement
frame.placement.translation
frame.parentJoint
model.parents
value.setflags
```

写入目标：

```python
names
name
joints
index
columns
joint
limits
neutral
gains
data
subspace
linear
placement
translation
base_linear
bodies
body_names
body
coefficients
radii
body_index
frame_id
spheres
radius
sphere
radii[body_index]
frame
row
joint_id
column
linear_gain
angular_gain
row[column]
extent
row[index]
value
```

直接调用符号：

```python
tuple
str
range
int
np.array
pin.neutral
joint.createData
joint.calc
np.asarray(data.S).reshape
np.asarray
gains.append
_norm_up
_linear_norm_up
_names
np.zeros
len
enumerate
model.getFrameId
ValueError
_array
np.any
max
_add_up
float
_mul_up
abs
value.setflags
GeometryMotionBound
```

## `relative_speed_bound`

[定义位置](../../../manipulation/geometry_motion_bound.py#L225)。

输入签名：

```python
def relative_speed_bound(actor_speed_m_s, *, target_speed_m_s):
```

实现说明：

> Bound separation change by both point speeds; static means explicit 0.
>
> A moving owner with no motion bound must remain unresolved. Requiring the
> target argument prevents silently treating absent owner motion as static.
> For two robot bodies pass their respective speed_bounds entries. For an
> attached payload bind its complete cover to the tool frame before building
> L; for a separately moving owner supply its independently derived bound.

显式输出分支：

```python
_add_up(float(values[0]), float(values[1]))
```

读取的对象属性：

```python
values.shape
np.any
```

写入目标：

```python
values
```

直接调用符号：

```python
_array
np.any
ValueError
_add_up
float
```
