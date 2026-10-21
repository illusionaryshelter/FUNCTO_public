# `manipulation/goals.py` 实现契约

代码基线：`9e8b365`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_points`

[定义位置](../../../manipulation/goals.py#L15)。

输入签名：

```python
def _points(value, name, minimum=1):
```

显式输出分支：

```python
points
```

读取的对象属性：

```python
np.asarray
points.ndim
points.shape
np.isfinite(points).all
np.isfinite
```

写入目标：

```python
points
```

直接调用符号：

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

显式输出分支：

```python
pose.copy()
```

读取的对象属性：

```python
np.asarray
pose.shape
np.isfinite(pose).all
np.isfinite
np.allclose
rotation.T
np.eye
np.isclose
np.linalg.det
np.linalg
pose.copy
```

写入目标：

```python
pose
rotation
```

直接调用符号：

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
class SupportSurface():
```

类字段：

```python
origin: np.ndarray
basis: np.ndarray
normal: np.ndarray
boundary: np.ndarray
source: str = 'observed_support_patch'
```

实现说明：

> A finite convex support patch: origin, tangent basis, outward normal.
>
> ``boundary`` is a convex set of XY coordinates in ``basis``. It describes
> observed/known support area, rather than extending a fitted plane forever.

## `SupportSurface.__post_init__`

[定义位置](../../../manipulation/goals.py#L51)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
self.origin
self.basis
self.normal
self.boundary
self.origin.shape
self.normal.shape
self.basis.shape
np.isfinite(v).all
np.isfinite
np.column_stack
np.allclose
axes.T
np.eye
self.boundary.ndim
self.boundary.shape
```

写入目标：

```python
self.origin
self.basis
self.normal
self.boundary
v
axes
self._hull
```

直接调用符号：

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
def to_dict(self):
```

显式输出分支：

```python
{'origin': self.origin.tolist(), 'basis': self.basis.tolist(), 'normal': self.normal.tolist(), 'boundary': self.boundary.tolist(), 'source': self.source}
```

读取的对象属性：

```python
self.origin.tolist
self.origin
self.basis.tolist
self.basis
self.normal.tolist
self.normal
self.boundary.tolist
self.boundary
self.source
```

写入目标：

无显式条目。

直接调用符号：

```python
self.origin.tolist
self.basis.tolist
self.normal.tolist
self.boundary.tolist
```

## `InteriorRegion`

[定义位置](../../../manipulation/goals.py#L79)。

```python
class InteriorRegion():
```

类字段：

```python
halfspaces: np.ndarray
source: str
```

实现说明：

> Known convex free space, expressed by unit-normal halfspaces A x + b<=0.
>
> Vertices/halfspaces must refer to the interior free region supplied by
> observed cavity geometry or upstream scene geometry. An outer object mask
> cannot establish this region. ``source`` remains in the output evidence.

## `InteriorRegion.__post_init__`

[定义位置](../../../manipulation/goals.py#L90)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
self.halfspaces
self.halfspaces.ndim
self.halfspaces.shape
np.isfinite(self.halfspaces).all
np.isfinite
np.linalg.norm
np.linalg
(norms < 1e-10).any
self.source
self.source.strip
np.column_stack
np.ones
ball.success
ball.x
np.concatenate
np.eye
extreme.success
```

写入目标：

```python
self.halfspaces
norms
matrix
offset
ball
direction
extreme
```

直接调用符号：

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
def from_vertices(cls, vertices, source):
```

显式输出分支：

```python
cls(hull.equations, source)
```

读取的对象属性：

```python
hull.equations
```

写入目标：

```python
points
hull
```

直接调用符号：

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
def to_dict(self):
```

显式输出分支：

```python
{'halfspaces': self.halfspaces.tolist(), 'source': self.source}
```

读取的对象属性：

```python
self.halfspaces.tolist
self.halfspaces
self.source
```

写入目标：

无显式条目。

直接调用符号：

```python
self.halfspaces.tolist
```

## `_nearest_feasible`

[定义位置](../../../manipulation/goals.py#L125)。

输入签名：

```python
def _nearest_feasible(preferred, matrix, upper):
```

实现说明：

> Project a desired translation onto finite linear geometric conditions.

显式输出分支：

```python
result.x
```

读取的对象属性：

```python
np.asarray
np.inf
np.sum
result.success
np.isfinite(result.x).all
np.isfinite
result.x
np.max
```

写入目标：

```python
preferred
constraint
result
```

直接调用符号：

```python
np.asarray
LinearConstraint
minimize
np.sum
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

实现说明：

> Preserve desired orientation; find the closest supported translation.
>
> The projected footprint of all supplied object points must lie inside the
> support patch. This sufficient geometric condition does not prove static
> stability or infer unobserved parts of the object.

显式输出分支：

```python
(pose, diagnostics)
```

读取的对象属性：

```python
np.isfinite
pose[:3, :3].T
surface.basis
surface._hull.equations
surface._hull
np.max
matrix.T
surface.origin
np.min
surface.normal
```

写入目标：

```python
pose
body
rotated
footprint
equations
matrix
offset
upper
preferred_xy
xy
height
pose[:3, 3]
diagnostics
```

直接调用符号：

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

显式输出分支：

```python
{'support_gap_m': float(abs(np.min(local @ surface.normal))), 'boundary_violation_m': float(max(0, boundary_values.max())), 'minimum_boundary_clearance_m': float(-boundary_values.max()), 'geometry_source': surface.source, 'checked_points': len(world)}
```

读取的对象属性：

```python
pose[:3, :3].T
surface.origin
surface.basis
surface._hull.equations
surface._hull
equations[:, :2].T
np.min
surface.normal
boundary_values.max
surface.source
```

写入目标：

```python
pose
world
local
xy
equations
boundary_values
```

直接调用符号：

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

实现说明：

> Fit the whole supplied object geometry within a known free region.

显式输出分支：

```python
(pose, check_inside(pose, body, region))
```

读取的对象属性：

```python
np.isfinite
pose[:3, :3].T
region.halfspaces
np.max
matrix.T
```

写入目标：

```python
pose
body
rotated
matrix
offset
upper
pose[:3, 3]
```

直接调用符号：

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

显式输出分支：

```python
{'interior_violation_m': float(max(0, values.max())), 'minimum_clearance_m': float(-values.max()), 'geometry_source': region.source, 'checked_points': len(world)}
```

读取的对象属性：

```python
pose[:3, :3].T
region.halfspaces[:, :3].T
region.halfspaces
values.max
region.source
```

写入目标：

```python
pose
world
values
```

直接调用符号：

```python
_pose
_points
float
max
values.max
len
```
