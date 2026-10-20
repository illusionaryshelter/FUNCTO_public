# `manipulation/geometry_motion_bound.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_finite`

[定义位置](../../../manipulation/geometry_motion_bound.py#L18)。

输入签名：

```python
def _finite(value):
```

输出分支的源码表达式：

```python
value
```

直接调用：

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

输出分支的源码表达式：

```python
second
first
_finite(math.nextafter(_finite(first + second), math.inf))
```

直接调用：

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

输出分支的源码表达式：

```python
0.0
_finite(math.nextafter(_finite(first * second), math.inf))
```

直接调用：

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

输出分支的源码表达式：

```python
0.0 if squared == 0 else _finite(math.nextafter(math.sqrt(squared), math.inf))
```

直接调用：

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

输出分支的源码表达式：

```python
(0.0, 0.0)
(math.nextafter(value, -math.inf), math.nextafter(value, math.inf))
```

直接调用：

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

输出分支的源码表达式：

```python
0.0 if maximum == 0 else _finite(math.nextafter(math.sqrt(maximum), math.inf))
```

直接写入的状态或数组位置：`gram_abs[i, j]`。

直接调用：

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

输出分支的源码表达式：

```python
maximum
```

直接调用：

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

直接读取的对象属性：`result.dtype`、`result.astype`。

输出分支的源码表达式：

```python
result.astype(np.int64 if integer else np.float64, copy=True)
```

直接调用：

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

输出分支的源码表达式：

```python
result
```

直接调用：

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
class SphereBody:
    name: str
    link: str
    spheres: object
    affine_translation: Mapping
```

## `GeometryMotionBound`

[定义位置](../../../manipulation/geometry_motion_bound.py#L132)。

```python
class GeometryMotionBound:
    body_names: tuple
    joint_names: tuple
    coefficients: np.ndarray
    joint_position_limits: np.ndarray
    local_radius_bounds_m: np.ndarray
```

该类型的直接方法：`speed_bounds`。

## `GeometryMotionBound.speed_bounds`

[定义位置](../../../manipulation/geometry_motion_bound.py#L141)。

输入签名：

```python
def GeometryMotionBound.speed_bounds(self, joint_speed_upper):
```

直接读取的对象属性：`self.joint_names`、`self.body_names`、`self.coefficients`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[i]`。

直接调用：

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

输出分支的源码表达式：

```python
GeometryMotionBound(body_names, names, coefficients, limits, radii)
```

直接写入的状态或数组位置：`local_radii[sphere_index]`、`row[column]`、`radii[body_index]`。

直接调用：

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

输出分支的源码表达式：

```python
_add_up(float(values[0]), float(values[1]))
```

直接调用：

```python
_array
np.any
ValueError
_add_up
float
```
