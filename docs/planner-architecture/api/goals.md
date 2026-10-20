# `manipulation/goals.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_points`

[定义位置](../../../manipulation/goals.py#L15)。

输入签名：

```python
def _points(value, name, minimum=1):
```

输出分支的源码表达式：

```python
points
```

直接调用：

```python
np.asarray
len
ValueError
np.isfinite(points).all
np.isfinite
```

## `_pose`

[定义位置](../../../manipulation/goals.py#L24)。

输入签名：

```python
def _pose(value):
```

输出分支的源码表达式：

```python
pose.copy()
```

直接调用：

```python
np.asarray
np.isfinite(pose).all
np.isfinite
ValueError
np.allclose
np.eye
np.isclose
np.linalg.det
pose.copy
```

## `SupportSurface`

[定义位置](../../../manipulation/goals.py#L38)。

```python
class SupportSurface:
    origin: np.ndarray
    basis: np.ndarray
    normal: np.ndarray
    boundary: np.ndarray
    source: str = 'observed_support_patch'
```

该类型的直接方法：`__post_init__`、`to_dict`。

## `SupportSurface.__post_init__`

[定义位置](../../../manipulation/goals.py#L51)。

输入签名：

```python
def SupportSurface.__post_init__(self):
```

直接读取的对象属性：`self.origin`、`self.basis`、`self.normal`、`self.boundary`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.origin`、`self.basis`、`self.normal`、`self.boundary`、`self._hull`。

直接调用：

```python
np.asarray
ValueError
all
np.isfinite(v).all
np.isfinite
np.column_stack
np.allclose
np.eye
ConvexHull
```

## `SupportSurface.to_dict`

[定义位置](../../../manipulation/goals.py#L72)。

输入签名：

```python
def SupportSurface.to_dict(self):
```

直接读取的对象属性：`self.origin`、`self.basis`、`self.normal`、`self.boundary`、`self.source`。

输出分支的源码表达式：

```python
{'origin': self.origin.tolist(), 'basis': self.basis.tolist(), 'normal': self.normal.tolist(), 'boundary': self.boundary.tolist(), 'source': self.source}
```

直接调用：

```python
self.origin.tolist
self.basis.tolist
self.normal.tolist
self.boundary.tolist
```

## `InteriorRegion`

[定义位置](../../../manipulation/goals.py#L79)。

```python
class InteriorRegion:
    halfspaces: np.ndarray
    source: str
```

该类型的直接方法：`__post_init__`、`from_vertices`、`to_dict`。

## `InteriorRegion.__post_init__`

[定义位置](../../../manipulation/goals.py#L90)。

输入签名：

```python
def InteriorRegion.__post_init__(self):
```

直接读取的对象属性：`self.halfspaces`、`self.source`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.halfspaces`。

直接调用：

```python
np.asarray
len
np.isfinite(self.halfspaces).all
np.isfinite
ValueError
np.linalg.norm
(norms < 1e-10).any
isinstance
self.source.strip
linprog
np.column_stack
np.ones
np.concatenate
np.eye
```

## `InteriorRegion.from_vertices`

[定义位置](../../../manipulation/goals.py#L113)。

输入签名：

```python
@classmethod
def InteriorRegion.from_vertices(cls, vertices, source):
```

输出分支的源码表达式：

```python
cls(hull.equations, source)
```

直接调用：

```python
_points
ConvexHull
ValueError
cls
```

## `InteriorRegion.to_dict`

[定义位置](../../../manipulation/goals.py#L121)。

输入签名：

```python
def InteriorRegion.to_dict(self):
```

直接读取的对象属性：`self.halfspaces`、`self.source`。

输出分支的源码表达式：

```python
{'halfspaces': self.halfspaces.tolist(), 'source': self.source}
```

直接调用：

```python
self.halfspaces.tolist
```

## `_nearest_feasible`

[定义位置](../../../manipulation/goals.py#L125)。

输入签名：

```python
def _nearest_feasible(preferred, matrix, upper):
```

直接读取的对象属性：`result.success`、`result.x`。

输出分支的源码表达式：

```python
result.x
```

直接调用：

```python
np.asarray
LinearConstraint
minimize
np.isfinite(result.x).all
np.isfinite
np.max
ValueError
```

## `place_on`

[定义位置](../../../manipulation/goals.py#L138)。

输入签名：

```python
def place_on(preferred_pose, body_points, surface, *, margin=0.0):
```

输出分支的源码表达式：

```python
(pose, diagnostics)
```

直接写入的状态或数组位置：`pose[:3, 3]`。

直接调用：

```python
_pose
_points
np.isfinite
ValueError
np.max
_nearest_feasible
np.min
check_on
```

## `check_on`

[定义位置](../../../manipulation/goals.py#L164)。

输入签名：

```python
def check_on(pose, body_points, surface):
```

输出分支的源码表达式：

```python
{'support_gap_m': float(abs(np.min(local @ surface.normal))), 'boundary_violation_m': float(max(0, boundary_values.max())), 'minimum_boundary_clearance_m': float(-boundary_values.max()), 'geometry_source': surface.source, 'checked_points': len(world)}
```

直接调用：

```python
_pose
_points
float
abs
np.min
max
boundary_values.max
len
```

## `place_inside`

[定义位置](../../../manipulation/goals.py#L177)。

输入签名：

```python
def place_inside(preferred_pose, body_points, region, *, margin=0.0):
```

输出分支的源码表达式：

```python
(pose, check_inside(pose, body, region))
```

直接写入的状态或数组位置：`pose[:3, 3]`。

直接调用：

```python
_pose
_points
np.isfinite
ValueError
np.max
_nearest_feasible
check_inside
```

## `check_inside`

[定义位置](../../../manipulation/goals.py#L190)。

输入签名：

```python
def check_inside(pose, body_points, region):
```

输出分支的源码表达式：

```python
{'interior_violation_m': float(max(0, values.max())), 'minimum_clearance_m': float(-values.max()), 'geometry_source': region.source, 'checked_points': len(world)}
```

直接调用：

```python
_pose
_points
float
max
values.max
len
```
