# `manipulation/implicit_distance.py` 实现契约

代码基线：`本次 A–D 实现工作树，实际提交引用待补`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `transform_points`

[定义位置](../../../manipulation/implicit_distance.py#L31)。

输入签名：

```python
def transform_points(points, pose):
```

显式输出分支：

```python
np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]
```

读取的对象属性：

```python
np.asarray
pose[:3, :3].T
```

写入目标：

无显式条目。

直接调用符号：

```python
np.asarray
```

## `transform_point_error`

[定义位置](../../../manipulation/implicit_distance.py#L35)。

输入签名：

```python
def transform_point_error(points, pose):
```

实现说明：

> Bound forward-transform arithmetic before cancellation occurs.

显式输出分支：

```python
_GAMMA * (np.abs(points) @ np.abs(pose[:3, :3]).T + np.abs(pose[:3, 3]))
```

读取的对象属性：

```python
np.abs
np.abs(pose[:3, :3]).T
```

写入目标：

无显式条目。

直接调用符号：

```python
np.abs
```

## `inverse_points`

[定义位置](../../../manipulation/implicit_distance.py#L40)。

输入签名：

```python
def inverse_points(points, pose):
```

显式输出分支：

```python
(np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
```

读取的对象属性：

```python
np.asarray
```

写入目标：

无显式条目。

直接调用符号：

```python
np.asarray
```

## `inverse_point_error`

[定义位置](../../../manipulation/implicit_distance.py#L44)。

输入签名：

```python
def inverse_point_error(points, pose):
```

实现说明：

> Bound arithmetic and finite orthogonality error of the rigid inverse.

显式输出分支：

```python
arithmetic + inverse_error[..., None]
```

读取的对象属性：

```python
np.asarray
np.abs
np.linalg.norm
np.linalg
rotation.T
np.eye
np.inf
np.abs(rotation).T
np.max
```

写入目标：

```python
rotation
delta
arithmetic
defect
inverse_error
```

直接调用符号：

```python
np.asarray
np.abs
np.linalg.norm
np.eye
float
ValueError
np.max
```

## `box_corners`

[定义位置](../../../manipulation/implicit_distance.py#L58)。

输入签名：

```python
def box_corners(lower, upper):
```

显式输出分支：

```python
lower + CORNERS * (upper - lower)
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `world_bounds`

[定义位置](../../../manipulation/implicit_distance.py#L62)。

输入签名：

```python
def world_bounds(lower, upper, pose):
```

实现说明：

> Enclose the transformed box including arithmetic round-off.

显式输出分支：

```python
(np.nextafter(center - radius - error, -np.inf), np.nextafter(center + radius + error, np.inf))
```

读取的对象属性：

```python
np.abs
np.nextafter
np.inf
```

写入目标：

```python
middle
half
center
radius
error
```

直接调用符号：

```python
np.abs
np.nextafter
```

## `box_distance`

[定义位置](../../../manipulation/implicit_distance.py#L73)。

输入签名：

```python
def box_distance(a, b):
```

显式输出分支：

```python
max(0.0, float(np.nextafter(value - error, -np.inf)))
```

读取的对象属性：

```python
np.maximum
np.linalg.norm
np.linalg
np.max
np.abs
np.r_
np.nextafter
np.inf
```

写入目标：

```python
gap
value
error
```

直接调用符号：

```python
np.maximum
float
np.linalg.norm
np.max
np.abs
max
np.nextafter
```

## `trilinear`

[定义位置](../../../manipulation/implicit_distance.py#L80)。

输入签名：

```python
def trilinear(values, coordinates):
```

实现说明：

> Return value, analytic local-coordinate gradient, and rounding bound.

显式输出分支：

```python
(value, gradient, error)
```

读取的对象属性：

```python
np.asarray
np.float64
np.where
weights.prod
np.sum
derivatives.append
weights[..., other].prod
np.stack
np.abs
```

写入目标：

```python
u
weights
product
value
derivatives
axis
other
i
gradient
error
```

直接调用符号：

```python
np.asarray
np.where
weights.prod
np.sum
range
derivatives.append
weights[..., other].prod
np.stack
np.abs
```

## `PairPermission`

[定义位置](../../../manipulation/implicit_distance.py#L96)。

```python
class PairPermission():
```

类字段：

无显式条目。

实现说明：

> A union of finite, world-frame permitted point-pair regions.
>
> Each entry contains ``((normals_a, bounds_a), (normals_b, bounds_b))``.
> A point pair is permitted when both endpoints satisfy the corresponding
> halfspaces of at least one entry. Interval, direction and relationship
> activation are evaluated by the caller before constructing this query.

## `PairPermission.__init__`

[定义位置](../../../manipulation/implicit_distance.py#L105)。

输入签名：

```python
def __init__(self, regions, *, excluded_second_cells=()):
```

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
self.regions
self.excluded_second_cells
```

直接调用符号：

```python
tuple
frozenset
```

## `PairPermission.excludes_cells`

[定义位置](../../../manipulation/implicit_distance.py#L112)。

输入签名：

```python
def excludes_cells(self, first, second):
```

显式输出分支：

```python
second >= 0 and second in self.excluded_second_cells
```

读取的对象属性：

```python
self.excluded_second_cells
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `PairPermission._classify`

[定义位置](../../../manipulation/implicit_distance.py#L116)。

输入签名：

```python
def _classify(box, region):
```

显式输出分支：

```python
'inside'
'outside'
'mixed'
```

读取的对象属性：

```python
np.asarray
np.where
minimum_terms.sum
maximum_terms.sum
np.abs(minimum_terms).sum
np.abs
np.abs(maximum_terms).sum
np.all
np.any
```

写入目标：

```python
normals
bounds
low
high
minimum_terms
maximum_terms
minimum
maximum
low_error
high_error
```

直接调用符号：

```python
np.asarray
np.where
minimum_terms.sum
maximum_terms.sum
np.abs(minimum_terms).sum
np.abs
np.abs(maximum_terms).sum
np.all
np.any
```

## `PairPermission.boxes`

[定义位置](../../../manipulation/implicit_distance.py#L131)。

输入签名：

```python
def boxes(self, first, second):
```

显式输出分支：

```python
'permitted'
'mixed' if mixed else 'forbidden'
```

读取的对象属性：

```python
self.regions
self._classify
```

写入目标：

```python
mixed
a
b
ca
cb
```

直接调用符号：

```python
self._classify
```

## `_Node`

[定义位置](../../../manipulation/implicit_distance.py#L143)。

```python
class _Node():
```

类字段：

```python
lower: np.ndarray
upper: np.ndarray
children: tuple = ()
cell: int = -1
points: object = None
```

## `ObservedField`

[定义位置](../../../manipulation/implicit_distance.py#L153)。

```python
class ObservedField():
```

类字段：

无显式条目。

实现说明：

> An immutable local TSDF and its actual interpolation support.
>
> Occupancy exists only in cells whose eight samples were observed. Unknown
> samples never receive a positive value. The spatial tree indexes cells
> that may contain nonpositive values, not a replacement solid.

## `ObservedField.__init__`

[定义位置](../../../manipulation/implicit_distance.py#L161)。

输入签名：

```python
def __init__(self, values, observed, *, origin, voxel_size):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.array
np.float32
np.asarray(origin, dtype=np.float64).copy
np.asarray
np.float64
self.values.ndim
self.values
self.values.shape
self.observed.shape
self.observed
self.origin.shape
self.origin
np.isfinite(self.origin).all
np.isfinite
self.voxel_size
np.isfinite(self.values[self.observed]).all
np.int64
self.shape
np.ones
self.cell_shape
np.full
np.inf
CORNERS.astype
np.minimum
np.argwhere
self.cells
self.values[tuple((self.cells[:, None, :] + CORNERS.astype(int)[None]).transpose(2, 0, 1))].astype
(self.cells[:, None, :] + CORNERS.astype(int)[None]).transpose
np.empty
self.cell_origins
self._cell_values.min
self._cell_values
self._cell_values.max
np.flatnonzero
self._zero_cell_ids
self._initial_cell_bounds
self._initial_occupied_points
self.tree
self.tree.tree
self.valid_cells
self._boundary_only_cells
array.flags
a.nbytes
self.nodes
points.nbytes
self._zero_tree
self._zero_tree.data.flags
self._zero_tree.data
self._zero_tree.data.nbytes
self._zero_tree.indices.nbytes
self._zero_tree.indices
```

写入目标：

```python
self.values
self.observed
self.origin
self.voxel_size
self.shape
self.cell_shape
valid
minimum
corner
slices
k
n
self.valid_cells
self.cells
self.cell_origins
self._cell_values
centers
self.tree
maximum
self._boundary_only_cells
self._zero_cell_ids
self._zero_tree
self.nodes
cell_lower
cell_upper
leaf_points
self.root
array
array.flags.writeable
self.storage_nbytes
a
points
self._zero_tree.data.flags.writeable
```

直接调用符号：

```python
np.array
np.asarray(origin, dtype=np.float64).copy
np.asarray
float
min
np.isfinite(self.origin).all
np.isfinite
np.isfinite(self.values[self.observed]).all
ValueError
np.ones
tuple
np.full
CORNERS.astype
slice
int
zip
np.minimum
np.argwhere
len
self.values[tuple((self.cells[:, None, :] + CORNERS.astype(int)[None]).transpose(2, 0, 1))].astype
(self.cells[:, None, :] + CORNERS.astype(int)[None]).transpose
np.empty
cKDTree
self._cell_values.min
self._cell_values.max
np.flatnonzero
self._initial_cell_bounds
self._initial_occupied_points
build
sum
```

## `ObservedField.__init__.build`

[定义位置](../../../manipulation/implicit_distance.py#L212)。

输入签名：

```python
def build(node):
```

显式输出分支：

```python
identity
```

读取的对象属性：

```python
self.tree.indices
self.tree
node.start_idx
node.end_idx
self.cell_origins[indices].min
self.cell_origins
self.cell_origins[indices].max
self.voxel_size
self.nodes
self.nodes.append
node.split_dim
children.append
node.lesser
node.greater
```

写入目标：

```python
indices
lower
upper
identity
children
index
self.nodes[identity]
```

直接调用符号：

```python
self.cell_origins[indices].min
self.cell_origins[indices].max
len
self.nodes.append
children.append
_Node
int
build
tuple
```

## `ObservedField._initial_cell_bounds`

[定义位置](../../../manipulation/implicit_distance.py#L248)。

输入签名：

```python
def _initial_cell_bounds(self):
```

实现说明：

> Evaluate independent cells in batches, retaining x/y/z ordering.
>
> The formulas match ``cell_value`` and ``tighten`` including coordinate
> error, endpoint slope bounds and outward rounding. Only independent
> cells share a NumPy operation; each axis consumes the previous axis's
> bounds. Parent tree boxes and query-time refinement are unchanged.

显式输出分支：

```python
(lower, upper)
```

读取的对象属性：

```python
self.cell_origins.copy
self.cell_origins
self.voxel_size
self.cells
self._cell_values
np.max
np.abs
np.arange
self.cell_value
np.all
np.any
np.flatnonzero
np.full
a[ids].shape
np.inf
np.divide
roots.min
roots.max
np.maximum
np.nextafter
np.minimum
```

写入目标：

```python
lower
upper
start
stop
lo
hi
samples
scale_error
indices
axis
points
actual
_
rounding
values
a
b
span
shrink_low
shrink_high
selected
low_side
ids
eligible
roots
fraction
value
error
lo[ids, axis]
hi[ids, axis]
```

直接调用符号：

```python
self.cell_origins.copy
range
len
min
np.max
np.abs
np.arange
self.cell_value
np.all
np.any
np.flatnonzero
np.full
np.divide
roots.min
roots.max
np.maximum
np.nextafter
np.minimum
```

## `ObservedField._initial_occupied_points`

[定义位置](../../../manipulation/implicit_distance.py#L294)。

输入签名：

```python
def _initial_occupied_points(self, lower, upper):
```

实现说明：

> Batch the same corner, original-sample and edge-root arithmetic.
>
> Only the immutable original leaves use these arrays. Query-time
> subdivision still evaluates its own smaller bounds with the scalar
> method below. Batches bound temporary storage independently of field
> size, and each retained array contains only that leaf's actual points.

显式输出分支：

```python
output
```

读取的对象属性：

```python
np.asarray(_EDGES).T
np.asarray
self.cells
np.arange
self.cell_value
self.cell_origins
self.voxel_size
((original >= lo[:, None]) & (original <= hi[:, None])).all
np.empty
np.zeros
points.shape
self._cell_values
np.nonzero
np.where
np.minimum
np.maximum
np.abs
local.flags
output.append
```

写入目标：

```python
output
edge_a
edge_b
start
stop
cells
lo
hi
corners
values
_
errors
original
inside
points
kept
points[:, :8]
points[:, 8:16]
kept[:, :8]
kept[:, 8:16]
rows
edges
a
b
va
vb
fraction
point
negative_index
negative
v
e
points[rows, 16 + edges]
kept[rows, 16 + edges]
row
local
local.flags.writeable
```

直接调用符号：

```python
np.asarray
range
len
min
np.arange
self.cell_value
((original >= lo[:, None]) & (original <= hi[:, None])).all
np.empty
np.zeros
np.nonzero
np.where
np.minimum
np.maximum
np.abs
output.append
```

## `ObservedField.bounds`

[定义位置](../../../manipulation/implicit_distance.py#L340)。

输入签名：

```python
def bounds(self):
```

显式输出分支：

```python
None
(root.lower, root.upper)
```

读取的对象属性：

```python
self.root
self.nodes
root.lower
root.upper
```

写入目标：

```python
root
```

直接调用符号：

无显式条目。

## `ObservedField.cell_value`

[定义位置](../../../manipulation/implicit_distance.py#L346)。

输入签名：

```python
def cell_value(self, index, points, point_error=0.0):
```

显式输出分支：

```python
(value, gradient / self.voxel_size, error)
```

读取的对象属性：

```python
np.asarray
self.cell_origins
self.voxel_size
self._cell_values
np.abs
np.stack
np.max
np.sum
```

写入目标：

```python
u
value
gradient
error
coordinate_error
values
slope
i
j
axis
```

直接调用符号：

```python
np.asarray
trilinear
np.abs
np.stack
np.max
range
np.sum
```

## `ObservedField.sample`

[定义位置](../../../manipulation/implicit_distance.py#L364)。

输入签名：

```python
def sample(self, points):
```

实现说明：

> Evaluate only fully supported cells, retaining an explicit valid mask.

显式输出分支：

```python
(value.reshape(layout), valid.reshape(layout), gradient.reshape(*layout, 3), error.reshape(layout))
```

读取的对象属性：

```python
np.asarray
points.shape
points.reshape
self.origin
self.voxel_size
np.floor(grid).astype
np.floor
np.int64
np.minimum
self.cell_shape
((grid >= 0) & (grid <= self.shape - 1)).all
self.shape
np.clip
self.valid_cells
clipped.T
CORNERS.astype
((corner == 0) | (grid == indices)).all
((candidate >= 0) & (candidate < self.cell_shape)).all
np.flatnonzero
candidate[selected].T
np.full
np.nan
np.zeros_like
self.values[tuple((clipped[selected, None, :] + CORNERS.astype(int)[None]).transpose(2, 0, 1))].astype
self.values
(clipped[selected, None, :] + CORNERS.astype(int)[None]).transpose
np.abs
np.stack
np.max
np.sum
value.reshape
valid.reshape
gradient.reshape
error.reshape
```

写入目标：

```python
points
layout
flat
grid
indices
inside
clipped
valid
corner
candidate
eligible
selected
clipped[selected]
valid[selected]
value
gradient
error
samples
v
g
e
coordinate_error
slope
i
j
axis
value[selected]
gradient[selected]
error[selected]
```

直接调用符号：

```python
np.asarray
points.reshape
np.floor(grid).astype
np.floor
np.minimum
((grid >= 0) & (grid <= self.shape - 1)).all
np.clip
tuple
CORNERS.astype
((corner == 0) | (grid == indices)).all
((candidate >= 0) & (candidate < self.cell_shape)).all
np.flatnonzero
len
np.full
np.zeros_like
self.values[tuple((clipped[selected, None, :] + CORNERS.astype(int)[None]).transpose(2, 0, 1))].astype
(clipped[selected, None, :] + CORNERS.astype(int)[None]).transpose
trilinear
np.abs
np.stack
np.max
range
np.sum
value.reshape
valid.reshape
gradient.reshape
error.reshape
```

## `ObservedField.node_values`

[定义位置](../../../manipulation/implicit_distance.py#L407)。

输入签名：

```python
def node_values(self, node):
```

显式输出分支：

```python
self.cell_value(node.cell, box_corners(node.lower, node.upper))[0]
```

读取的对象属性：

```python
self.cell_value
node.cell
node.lower
node.upper
```

写入目标：

无显式条目。

直接调用符号：

```python
self.cell_value
box_corners
```

## `ObservedField.tighten`

[定义位置](../../../manipulation/implicit_distance.py#L410)。

输入签名：

```python
def tighten(self, node):
```

实现说明：

> Clip only provably positive face slabs from a mixed cell's box.
>
> At a fixed axis coordinate the minimum of a trilinear function is at
> one of the four other-coordinate corners. Its first/last possible
> zero therefore bounds the occupied projection on that axis. These
> analytic bounds avoid subdividing planar free slabs into tiny boxes.

显式输出分支：

```python
_Node(lower, upper, cell=node.cell)
```

读取的对象属性：

```python
node.lower.copy
node.lower
node.upper.copy
node.upper
np.max
np.abs
self._cell_values
node.cell
self.cell_value
np.flatnonzero
np.all
np.any
roots.min
np.nextafter
np.inf
roots.max
```

写入目标：

```python
lower
upper
scale
error
axis
actual
_
rounding
values
first
last
a
b
span
roots
value
lower[axis]
upper[axis]
```

直接调用符号：

```python
node.lower.copy
node.upper.copy
float
np.max
np.abs
range
self.cell_value
box_corners
np.flatnonzero
np.all
np.any
roots.min
max
np.nextafter
abs
roots.max
min
_Node
```

## `ObservedField.occupied_points`

[定义位置](../../../manipulation/implicit_distance.py#L440)。

输入签名：

```python
def occupied_points(self, node):
```

实现说明：

> Return points belonging to the local cell set, including zero roots.

显式输出分支：

```python
np.empty((0, 3))
node.points
np.asarray(kept, dtype=float).reshape(-1, 3)
```

读取的对象属性：

```python
node.cell
np.empty
node.points
node.lower
node.upper
self.cell_value
self.cell_origins
self.voxel_size
((original >= node.lower) & (original <= node.upper)).all
kept.extend
self._cell_values
kept.append
np.asarray(kept, dtype=float).reshape
np.asarray
```

写入目标：

```python
points
values
_
errors
kept
original
original_inside
a
b
u
point
negative_index
negative
v
e
fraction
```

直接调用符号：

```python
np.empty
box_corners
self.cell_value
list
((original >= node.lower) & (original <= node.upper)).all
kept.extend
min
max
abs
float
kept.append
np.asarray(kept, dtype=float).reshape
np.asarray
```

## `ObservedField.split`

[定义位置](../../../manipulation/implicit_distance.py#L477)。

输入签名：

```python
def split(self, node):
```

显式输出分支：

```python
tuple((self.nodes[i] for i in node.children))
()
tuple(result)
```

读取的对象属性：

```python
node.children
self.nodes
np.argmax
node.upper
node.lower
node.upper.copy
node.lower.copy
node.cell
self.cell_value
child.cell
child.lower
child.upper
np.min
result.append
self.tighten
```

写入目标：

```python
i
axis
mid
high
low
high[axis]
low[axis]
result
child
values
_
errors
```

直接调用符号：

```python
tuple
int
np.argmax
node.upper.copy
node.lower.copy
_Node
self.cell_value
box_corners
float
np.min
result.append
self.tighten
```

## `ObservedField.local_zero`

[定义位置](../../../manipulation/implicit_distance.py#L495)。

输入签名：

```python
def local_zero(self, point):
```

实现说明：

> Find a local zero-section point solely for an optimization direction.

显式输出分支：

```python
None
best
```

读取的对象属性：

```python
self._zero_tree
self._zero_tree.query
np.asarray
self._zero_cell_ids
np.atleast_1d
self._cell_values
self.cell_origins
seeds.append
self.voxel_size
np.sum
result.success
result.x
self.cell_value
np.linalg.norm
np.linalg
```

写入目标：

```python
_
indices
best
candidate
identity
values
lower
seeds
a
b
target
initial
result
zero
gradient
norm
distance
```

直接调用符号：

```python
self._zero_tree.query
np.asarray
min
len
np.atleast_1d
int
seeds.append
float
np.sum
minimize
trilinear
self.cell_value
np.linalg.norm
```

## `query_distance`

[定义位置](../../../manipulation/implicit_distance.py#L534)。

输入签名：

```python
def query_distance(first, first_pose, second, second_pose, *, precision_m=0.0001, permission=None, check_request=None):
```

实现说明：

> Check penetration between the two supported occupied sets.
>
> ``permission`` classifies world-space box pairs as ``permitted``,
> ``forbidden`` or ``mixed``. Only wholly permitted domains are skipped.
> A witness uses its arithmetic uncertainty box, so a mixed permission
> boundary cannot establish a forbidden contact. Distance intervals remain
> diagnostic; positive clearance is not a constraint.

显式输出分支：

```python
{'status': 'clear', 'lower_m': None, 'upper_m': None, 'empty_set': True, 'full_entity_lower_m': None, 'intersection': False, 'queries': 0, 'witness_first': None, 'witness_second': None, 'scope': 'represented_observed_occupancy', 'distance_scope': 'unpermitted_point_pairs_at_query_pose'}
{'status': 'intersection', 'lower_m': 0.0, 'upper_m': 0.0, 'full_entity_lower_m': 0.0, 'intersection': True, 'queries': visits, 'witness_first': point.tolist(), 'witness_second': point.tolist(), 'scope': 'represented_observed_occupancy', 'distance_scope': 'unpermitted_point_pairs_at_query_pose'}
{'status': 'unresolved' if unresolved else 'clear', 'lower_m': lower, 'full_entity_lower_m': lower if permission is None else full_entity_lower, 'upper_m': upper if np.isfinite(upper) else None, 'intersection': None if unresolved else False, 'queries': visits, 'witness_first': None if witness is None else witness[0].tolist(), 'witness_second': None if witness is None else witness[1].tolist(), 'scope': 'represented_observed_occupancy', 'distance_scope': 'unpermitted_point_pairs_at_query_pose'}
```

读取的对象属性：

```python
np.isfinite
first.root
second.root
np.asarray
first.nodes
second.nodes
node.lower
node.upper
math.inf
heapq.heappop
a.cell
b.cell
first._boundary_only_cells
second._boundary_only_cells
first.occupied_points
second.occupied_points
np.linalg.norm
np.linalg
np.argsort
np.unravel_index
distances.shape
permission.boxes
np.nextafter
np.abs(pa[i]).sum
np.abs
np.abs(pb[j]).sum
np.inf
pa[i].copy
pb[j].copy
np.maximum
np.minimum
np.all
candidates.append
first.cell_origins
second.cell_origins
np.any
first.voxel_size
second.voxel_size
first.cell_value
second.cell_value
first._cell_values
second._cell_values
point.tolist
np.min
a.upper
a.lower
b.upper
b.lower
fields[axis].split
witness[0].tolist
witness[1].tolist
```

写入目标：

```python
precision
poses
fields
roots
root_world
node
pose
full_entity_lower
pending
serial
upper
witness
unresolved_lower
separated_lower
visits
lower
_
a
b
wa
wb
domain_upper
local_a
local_b
pa
pb
errors_a
errors_b
distances
flat
i
j
value
overlap_low
overlap_high
candidates
point
xa
xb
xa_error
xb_error
support_a
support_b
va
ea
vb
eb
inside_a
inside_b
axis
selected
children
child
unresolved
```

直接调用符号：

```python
float
np.isfinite
ValueError
np.asarray
tuple
world_bounds
zip
box_distance
count
push
heapq.heappop
check_request
min
first.occupied_points
second.occupied_points
transform_points
transform_point_error
len
np.linalg.norm
np.argsort
np.unravel_index
permission.boxes
np.nextafter
np.abs(pa[i]).sum
np.abs
np.abs(pb[j]).sum
pa[i].copy
pb[j].copy
np.maximum
np.minimum
np.all
candidates.append
inverse_points
inverse_point_error
np.any
first.cell_value
second.cell_value
point.tolist
np.min
max
int
fields[axis].split
witness[0].tolist
witness[1].tolist
```

## `query_distance.push`

[定义位置](../../../manipulation/implicit_distance.py#L570)。

输入签名：

```python
def push(a, b):
```

显式输出分支：

```python
None
None
```

读取的对象属性：

```python
permission.excludes_cells
a.cell
b.cell
a.lower
a.upper
b.lower
b.upper
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
separated_lower
diameter
```

直接调用符号：

```python
permission.excludes_cells
world_bounds
permission.boxes
box_distance
min
float
np.linalg.norm
heapq.heappush
int
next
```
