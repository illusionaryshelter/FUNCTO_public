# `manipulation/geometry_motion_bound.py` 实现契约

代码基线：`2498027；包含已实现的中断状态记录改动，待对应实现提交更新`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_finite`

[定义位置](../../../manipulation/geometry_motion_bound.py#L18)。

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

[定义位置](../../../manipulation/geometry_motion_bound.py#L24)。

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

[定义位置](../../../manipulation/geometry_motion_bound.py#L33)。

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

[定义位置](../../../manipulation/geometry_motion_bound.py#L39)。

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

[定义位置](../../../manipulation/geometry_motion_bound.py#L46)。

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

[定义位置](../../../manipulation/geometry_motion_bound.py#L54)。

输入签名：

```python
def _linear_norm_up(matrix):
```

实现说明：

> Bound ||A||_2 by sqrt(||A.T A||_infinity), with outward arithmetic.
>
> Native fixed rotations are stored in float32 and need not be exactly
> orthogonal. Treating their norm as exactly one could underestimate the
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

## `_affine_abs_up`

[定义位置](../../../manipulation/geometry_motion_bound.py#L81)。

输入签名：

```python
def _affine_abs_up(scale, offset, lower, upper):
```

实现说明：

> Maximum |scale*q+offset| over the declared full joint interval.

显式输出分支：

```python
maximum
```

读取的对象属性：

```python
math.nextafter
math.inf
```

写入目标：

```python
maximum
position
lo
hi
```

直接调用符号：

```python
_product_interval
_finite
math.nextafter
max
abs
```

## `_array`

[定义位置](../../../manipulation/geometry_motion_bound.py#L92)。

输入签名：

```python
def _array(value, label, *, integer=False):
```

显式输出分支：

```python
result.astype(np.int64 if integer else np.float64, copy=True)
```

读取的对象属性：

```python
np.asarray
result.dtype.kind
result.dtype
np.isfinite(result).all
np.isfinite
result.astype
np.int64
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

[定义位置](../../../manipulation/geometry_motion_bound.py#L105)。

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

[定义位置](../../../manipulation/geometry_motion_bound.py#L114)。

```python
class SphereBody():
```

类字段：

```python
name: str
link: str
spheres: object
affine_translation: Mapping
```

实现说明：

> One complete cover in a named native link frame.
>
> ``spheres`` has shape [S,4], with local centres and nonnegative radii.
> ``affine_translation`` explicitly maps each independently moving scalar
> joint name to [S,3] local centre translations per unit q. Pass {} for a
> rigid cover. For the existing timed hand, use its TCP-local spheres and
> ``sphere_joint_translation`` here, rather than its locked native fingers.
> Already transformed/offset spheres must not be transformed a second time.

## `GeometryMotionBound`

[定义位置](../../../manipulation/geometry_motion_bound.py#L132)。

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

[定义位置](../../../manipulation/geometry_motion_bound.py#L141)。

输入签名：

```python
def speed_bounds(self, joint_speed_upper):
```

实现说明：

> Return one m/s bound per body for complete named |dq| bounds.
>
> Speeds must cover every native and affine coordinate. A missing timed
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

[定义位置](../../../manipulation/geometry_motion_bound.py#L161)。

输入签名：

```python
def build_geometry_motion_bound(*, fixed_transforms, link_map, joint_map, joint_map_type, joint_offset_map, joint_names, joint_position_limits, link_name_to_idx_map, bodies, extra_joint_limits):
```

实现说明：

> Build small CPU coefficients from CuRobo KinematicsParams metadata.
>
> Copy the identically named native arrays once after model loading.
> ``joint_position_limits`` is ``params.joint_limits.position`` [2,J]; the
> native ``joint_offset_map`` is accepted in its flat [2*N] layout. Parent
> indices come directly from ``link_map``; no alternate FK/tree is built.
> Native root index 0 is fixed and its stored base transform is included.
>
> ``extra_joint_limits`` is mandatory: {} explicitly declares no extra
> affine coordinates, otherwise name -> (lower, upper) supplies every such
> coordinate. Affine translation may also use an existing native coordinate.
> Unmodelled hand motion cannot be represented by silently omitting this
> declaration. Freely moving scene owners are not part of this fixed-base
> tree; their independently bounded speeds must enter relative_speed_bound.
>
> The native CUDA forward kernel uses joint types -1 and 0..5 (fixed,
> xyz-prismatic, xyz-revolute), with negative axes/mimic in scale*q+offset.
> Legacy negative-axis enum codes 6..11 are rejected rather than assumed to
> have the same kernel semantics. Locked joints must already be folded into
> fixed transforms by the existing loader.
>
> Proof: starting at each body's full local radius, walk toward the base.
> A rotating edge contributes |scale|*current_radius to its coordinate; a
> translating edge contributes |scale| and increases the radius by its full
> interval extension. The fixed linear transform multiplies both bounds by
> its operator-norm bound and its translation increases radius. Triangle
> inequalities and addition for repeated mimic coordinates give L. This
> also covers interior sphere points, not only centres or mesh vertices.

显式输出分支：

```python
GeometryMotionBound(body_names, names, coefficients, limits, radii)
```

读取的对象属性：

```python
transforms.ndim
transforms.shape
offsets.shape
offsets.reshape
parents.shape
joints.shape
types.shape
np.any
limits.shape
np.bool_
np.integer
links.values
seen.add
extra.shape
np.concatenate
extra.T
body.name
np.zeros
body.link
body.affine_translation
body.spheres
spheres.ndim
spheres.shape
np.array
body.affine_translation.items
mapping.shape
used_extra.add
local_radii.max
value.setflags
```

写入目标：

```python
transforms
count
parents
joints
types
offsets
names
limits
links
value
start
seen
link
extra_names
extra
name
columns
i
bodies
body
body_names
linear
transform
translation
coefficients
radii
used_extra
body_index
spheres
local_radii
row
mapping
column
norm
extent
sphere_index
gain
local_radii[sphere_index]
row[column]
radius
radii[body_index]
kind
scale
offset
extension
```

直接调用符号：

```python
_array
len
ValueError
offsets.reshape
np.any
_names
isinstance
dict
any
links.values
set
range
seen.add
int
np.concatenate
enumerate
tuple
_linear_norm_up
_norm_up
np.zeros
np.array
_add_up
float
body.affine_translation.items
max
abs
_mul_up
used_extra.add
local_radii.max
map
_affine_abs_up
value.setflags
GeometryMotionBound
```

## `relative_speed_bound`

[定义位置](../../../manipulation/geometry_motion_bound.py#L303)。

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
