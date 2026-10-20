# `manipulation/native_field_distance.py` 实现契约

代码基线：`22e6bda65682c1584d89623320024795fc1efa4e`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_transform`

[定义位置](../../../manipulation/native_field_distance.py#L22)。

输入签名：

```python
def _transform(matrix):
```

显式输出分支：

```python
coal.Transform3s(matrix[:3, :3], matrix[:3, 3])
```

读取的对象属性：

```python
coal.Transform3s
```

写入目标：

无显式条目。

直接调用符号：

```python
coal.Transform3s
```

## `_native_bounds`

[定义位置](../../../manipulation/native_field_distance.py#L26)。

输入签名：

```python
def _native_bounds(geometry, transform):
```

显式输出分支：

```python
np.asarray(intervals, dtype=float).T
```

读取的对象属性：

```python
np.eye
np.asarray(intervals, dtype=float).T
np.asarray
```

写入目标：

```python
intervals
axis
```

直接调用符号：

```python
native_projection_bounds
np.eye
np.asarray
```

## `_projection_gap`

[定义位置](../../../manipulation/native_field_distance.py#L32)。

输入签名：

```python
def _projection_gap(geometry, transform, box):
```

实现说明：

> Certify a lower bound using a direction proposed by Coal.
>
> GJK's tolerance is not a distance error bound. Only the native support
> interval and the box's outward projection determine this lower bound.
> A poor direction merely gives zero and leaves the domain to refinement.

显式输出分支：

```python
0.0
max(0.0, float(np.nextafter((gap - error) / (length * (1 + _GAMMA)), -np.inf)))
```

读取的对象属性：

```python
coal.DistanceRequest
coal.DistanceResult
coal.distance
coal.Box
coal.Transform3s
np.eye
np.asarray
result.normal
np.isfinite(direction).all
np.isfinite
np.linalg.norm
np.linalg
np.where
np.nextafter
minimum.sum
np.abs(minimum).sum
np.abs
np.inf
maximum.sum
np.abs(maximum).sum
```

写入目标：

```python
center
request
result
direction
length
native_low
native_high
minimum
maximum
box_low
box_high
gap
error
```

直接调用符号：

```python
coal.DistanceRequest
coal.DistanceResult
coal.distance
coal.Box
coal.Transform3s
np.eye
np.asarray
np.isfinite(direction).all
np.isfinite
RuntimeError
float
np.linalg.norm
native_projection_bounds
np.where
np.nextafter
minimum.sum
np.abs(minimum).sum
np.abs
maximum.sum
np.abs(maximum).sum
max
abs
```

## `_local_interval`

[定义位置](../../../manipulation/native_field_distance.py#L59)。

输入签名：

```python
def _local_interval(world, pose):
```

显式输出分支：

```python
(np.nextafter((local - error).min(0), -np.inf), np.nextafter((local + error).max(0), np.inf))
```

读取的对象属性：

```python
np.nextafter
(local - error).min
np.inf
(local + error).max
```

写入目标：

```python
corners
local
error
```

直接调用符号：

```python
box_corners
inverse_points
inverse_point_error
np.nextafter
(local - error).min
(local + error).max
```

## `query_native_field`

[定义位置](../../../manipulation/native_field_distance.py#L66)。

输入签名：

```python
def query_native_field(geometry, native_pose, field, field_pose, *, precision_m=0.0001, permission=None, check_request=None):
```

实现说明：

> Check penetration over unpermitted native and observed point pairs.
>
> A finite contact permission constrains both endpoints. When its boundary
> cuts the native shape, a local search box B denotes K intersect B, where K
> is the unchanged native shape. B is never accepted as robot material.
> Feasible native witnesses retain interval enclosures of the constructed
> point. Their midpoint is used only for display and a local derivative.

显式输出分支：

```python
report('clear', None, intersection=False, empty=True)
report('intersection', 0.0, intersection=True)
report('unresolved', unresolved_lower)
report('clear', 0.0, intersection=False)
```

读取的对象属性：

```python
np.isfinite
np.asarray
np.eye
np.inf
field.root
field.nodes
heapq.heappop
node.cell
field.occupied_points
np.maximum
np.minimum
np.any
field.cell_origins
np.all
field.voxel_size
np.nextafter
field.cell_value
permission.boxes
field._cell_values
np.abs
np.linalg.norm
np.linalg
np.abs(world).sum
np.abs(target).sum
node.upper
node.lower
np.min
node.children
np.argmax
native_box[1].copy
native_box[0].copy
field.split
```

写入目标：

```python
precision
native_pose
field_pose
transform
native_local
native_world
pending
serial
upper
witness
visits
unresolved_lower
lower
_
native_box
node
wa
wb
policy
partitioned
domain_upper
local_points
points
errors
proposals
index
wanted
feasible
local
world
key
field_local
support
center
radius
value
error
forbidden
inside
target
delta
distance
candidate_upper
diameter
native_diameter
overlap
split_native
axis
middle
hi
lo
hi[axis]
lo[axis]
children
child
```

直接调用符号：

```python
float
np.isfinite
ValueError
np.asarray
_transform
_native_bounds
np.eye
count
report
push
heapq.heappop
check_request
field.occupied_points
transform_points
transform_point_error
np.maximum
np.minimum
enumerate
native_feasible_point_bounds
np.any
_local_interval
np.all
np.nextafter
field.cell_value
permission.boxes
len
np.abs
np.linalg.norm
np.abs(world).sum
np.abs(target).sum
min
np.min
int
np.argmax
native_box[1].copy
native_box[0].copy
field.split
```

## `query_native_field.report`

[定义位置](../../../manipulation/native_field_distance.py#L88)。

输入签名：

```python
def report(status, lower, *, intersection=None, empty=False):
```

显式输出分支：

```python
output
```

读取的对象属性：

```python
np.isfinite
output.update
((a[0] + a[1]) / 2).tolist
((b[0] + b[1]) / 2).tolist
np.asarray(a).tolist
np.asarray
np.asarray(b).tolist
```

写入目标：

```python
output
a
b
output['empty_set']
```

直接调用符号：

```python
dict
np.isfinite
float
output.update
((a[0] + a[1]) / 2).tolist
((b[0] + b[1]) / 2).tolist
np.asarray(a).tolist
np.asarray
np.asarray(b).tolist
```

## `query_native_field.push`

[定义位置](../../../manipulation/native_field_distance.py#L105)。

输入签名：

```python
def push(native_box, node, partitioned=False):
```

显式输出分支：

```python
None
None
None
None
None
```

读取的对象属性：

```python
permission.excludes_cells
node.cell
node.lower
node.upper
permission.boxes
np.linalg.norm
np.linalg
heapq.heappush
```

写入目标：

```python
wa
wb
policy
lower
diameter
```

直接调用符号：

```python
permission.excludes_cells
world_bounds
permission.boxes
box_distance
max
_projection_gap
float
np.linalg.norm
heapq.heappush
int
next
```
