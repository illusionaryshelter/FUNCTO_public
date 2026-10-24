# `manipulation/observed_collision.py` 实现契约

代码基线：`CURRENT_IMPLEMENTATION`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_world`

[定义位置](../../../manipulation/observed_collision.py#L12)。

输入签名：

```python
def _world(point, pose):
```

显式输出分支：

```python
position + quaternion_rotate(quaternion, local)
```

读取的对象属性：

```python
torch.as_tensor
position.dtype
position.device
```

写入目标：

```python
position
quaternion
local
```

直接调用符号：

```python
torch.as_tensor
quaternion_rotate
```

## `pair_residual`

[定义位置](../../../manipulation/observed_collision.py#L18)。

输入签名：

```python
def pair_residual(first, first_pose, first_matrix, second, second_pose, second_matrix, *, permission=None, check_request=None, precision_m=0.0001, zero=None, native_bounds=None):
```

实现说明：

> Return the original classification residual and actual query evidence.
>
> Confirmed penetration has classification value one; clear and unresolved
> queries have value zero. The value is not a metric penetration depth.
> No derivative is attached to this classification. Original numerical
> witnesses and distance bounds remain part of the returned evidence.
>
> ``zero`` may be one scalar from a batch's differentiable zero baseline.
> A standalone call constructs that baseline from its two position tensors.

显式输出分支：

```python
(zero + 1.0 if result['status'] == 'intersection' else zero, result)
```

读取的对象属性：

```python
first_pose[0].sum
second_pose[0].sum
```

写入目标：

```python
native_first
query
options
result
zero
```

直接调用符号：

```python
isinstance
query
first_pose[0].sum
second_pose[0].sum
```

## `query_region_intersection`

[定义位置](../../../manipulation/observed_collision.py#L47)。

输入签名：

```python
def query_region_intersection(field, pose, normals, bounds, *, precision_m=1e-06, check_request=None):
```

实现说明：

> Find a supported zero surface in the finite convex contact region.
>
> An original zero sample or a sign-changing segment inside one supported
> cell proves surface contact by continuity. Negative interior material is
> insufficient. Interval bounds prune cells without a zero; finite unresolved
> boxes remain distinct from missing contact.

显式输出分支：

```python
{'status': 'contact', 'witness': world.tolist(), 'evidence': 'original_zero_sample'}
{'status': 'contact', 'segment': world[[a, b]].tolist(), 'evidence': 'opposite_sign_supported_segment'}
{'status': 'unresolved' if unresolved else 'missing_contact'}
```

读取的对象属性：

```python
np.asarray
field.root
field.nodes
pending.pop
node.cell
expanded.add
field.cell_origins
field.voxel_size
PairPermission._classify
node.lower
node.upper
node.children
pending.extend
field.cell_value
np.min
np.max
field._cell_values
world.tolist
candidates.extend
np.array
np.flatnonzero
world[[a, b]].tolist
np.argmax
np.linalg.norm
np.linalg
node.upper.copy
node.lower.copy
```

写入目标：

```python
region
pending
unresolved
expanded
node
lower
i
corners
values
_
errors
original
zeros
local
world
error
candidates
a
b
start
end
slope
offset
lo
hi
v
w
t
inside
point
e
negative
positive
axis
mid
hi[axis]
lo[axis]
```

直接调用符号：

```python
np.asarray
set
check_request
pending.pop
expanded.add
_Node
PairPermission._classify
world_bounds
pending.extend
box_corners
field.cell_value
np.min
np.max
transform_points
transform_point_error
world.tolist
zip
min
max
candidates.extend
np.array
np.flatnonzero
len
world[[a, b]].tolist
int
np.argmax
np.linalg.norm
node.upper.copy
node.lower.copy
```

## `query_minimum_height`

[定义位置](../../../manipulation/observed_collision.py#L122)。

输入签名：

```python
def query_minimum_height(field, pose, origin, normal, *, precision_m, check_request=None):
```

实现说明：

> Bound the minimum plane height over the represented occupied set.
>
> Spatial boxes provide lower bounds; actual supported occupied points
> provide upper bounds and a rigid-motion derivative witness. The stopping
> precision bounds numerical uncertainty and never changes a task tolerance.

显式输出分支：

```python
dict(status='missing_observation', lower_m=None, upper_m=None, witness_local=None)
dict(status='unresolved', lower_m=final_lower, upper_m=upper, witness_local=None if witness is None else witness.tolist(), visited_boxes=visits)
dict(status='bounded', lower_m=final_lower, upper_m=upper, witness_local=None if witness is None else witness.tolist(), visited_boxes=visits)
```

读取的对象属性：

```python
np.isfinite
field.root
np.asarray
np.inf
field.nodes
heapq.heappop
field.occupied_points
terms.sum
(errors * np.abs(normal)).sum
np.abs
np.abs(terms).sum
np.abs(origin * normal).sum
np.argmin
points[index].copy
field.split
witness.tolist
```

写入目标：

```python
origin
normal
pending
serial
upper
witness
final_lower
visits
lower
_
node
points
world
errors
terms
values
index
children
child
```

直接调用符号：

```python
np.isfinite
ValueError
dict
np.asarray
count
push
heapq.heappop
check_request
field.occupied_points
len
transform_points
transform_point_error
terms.sum
(errors * np.abs(normal)).sum
np.abs
np.abs(terms).sum
np.abs(origin * normal).sum
int
np.argmin
float
points[index].copy
min
field.split
witness.tolist
```

## `query_minimum_height.push`

[定义位置](../../../manipulation/observed_collision.py#L139)。

输入签名：

```python
def push(node):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
node.lower
node.upper
np.where
terms.sum
np.abs(terms).sum
np.abs
np.abs(origin * normal).sum
heapq.heappush
```

写入目标：

```python
lower_box
upper_box
corner
terms
lower
```

直接调用符号：

```python
world_bounds
np.where
float
terms.sum
np.abs(terms).sum
np.abs
np.abs(origin * normal).sum
heapq.heappush
next
```
