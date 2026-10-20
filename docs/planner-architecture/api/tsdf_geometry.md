# `manipulation/tsdf_geometry.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_require_sample_array_budget`

[定义位置](../../../manipulation/tsdf_geometry.py#L31)。

输入签名：

```python
def _require_sample_array_budget(shape):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
math.prod
np.dtype
ValueError
```

## `_Components`

[定义位置](../../../manipulation/tsdf_geometry.py#L39)。

```python
class _Components:
    ...
    ...
```

该类型的直接方法：`__init__`、`append`、`arrays`。

## `_Components.__init__`

[定义位置](../../../manipulation/tsdf_geometry.py#L43)。

输入签名：

```python
def _Components.__init__(self, *, max_host_rss_bytes):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.vertices`、`self.offsets`、`self.max_host_rss_bytes`。

## `_Components.append`

[定义位置](../../../manipulation/tsdf_geometry.py#L48)。

输入签名：

```python
def _Components.append(self, vertices):
```

直接读取的对象属性：`self.vertices`、`self.offsets`、`self.max_host_rss_bytes`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
len
np.dtype
require_host_memory
int
np.unique
np.asarray(vertices, dtype=np.float64).reshape
np.asarray
ValueError
self.vertices.append
self.offsets.append
```

## `_Components.arrays`

[定义位置](../../../manipulation/tsdf_geometry.py#L63)。

输入签名：

```python
def _Components.arrays(self):
```

直接读取的对象属性：`self.offsets`、`self.max_host_rss_bytes`、`self.vertices`。

输出分支的源码表达式：

```python
(vertices, np.asarray(self.offsets, dtype=np.int64))
```

直接调用：

```python
np.dtype
len
require_host_memory
int
np.concatenate
np.empty
np.asarray
```

## `tsdf_geometry_from_samples`

[定义位置](../../../manipulation/tsdf_geometry.py#L72)。

输入签名：

```python
def tsdf_geometry_from_samples(sdf, observed, *, origin, voxel_size, name, source, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
```

输出分支的源码表达式：

```python
{'solids': [record], 'query_gaps': [], 'field_summary': summary}
```

直接写入的状态或数组位置：`covered[slices]`、`covered[tuple(represented.T)]`。

直接调用：

```python
np.asarray
_require_sample_array_budget
float
any
np.dtype
np.isfinite(start).all
np.isfinite
np.isfinite(values[valid]).all
isinstance
ValueError
require_host_memory
int
_Components
np.zeros
tuple
np.empty
all
np.ones
slice
zip
np.flatnonzero(fully_negative).astype
np.flatnonzero
np.unravel_index
values[tuple(indices.T)].astype
cell_valid[tetra].all
scalar.min
crossing.append
np.asarray(crossing, dtype=np.float64).reshape
negative.append
np.concatenate
scalar.max
surface.append
np.argwhere
(negative if values[tuple(index)] < 0 else surface).append
surface.arrays
negative.arrays
sum
list
start.tolist
encode_array
np.count_nonzero
len
```

## `build_tsdf_geometry`

[定义位置](../../../manipulation/tsdf_geometry.py#L190)。

输入签名：

```python
def build_tsdf_geometry(mapper, name, source, *, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
```

输出分支的源码表达式：

```python
tsdf_geometry_from_samples(sdf, observed, origin=origin, voxel_size=size, name=name, source=source, max_host_rss_bytes=max_host_rss_bytes)
```

直接写入的状态或数组位置：`observed[region]`。

直接调用：

```python
ValueError
tuple
int
_require_sample_array_budget
require_host_memory
math.prod
float
config.grid_center.detach().cpu().numpy().astype
config.grid_center.detach().cpu().numpy
config.grid_center.detach().cpu
config.grid_center.detach
np.asarray
mapper.tsdf.export_blocks
blocks['active_block_coords'].detach().cpu().numpy
blocks['active_block_coords'].detach().cpu
blocks['active_block_coords'].detach
blocks['block_data'].detach().cpu().numpy
blocks['block_data'].detach().cpu
blocks['block_data'].detach
np.ceil(np.asarray(shape) / block_size).astype
np.ceil
np.zeros
zip
np.maximum
np.minimum
np.any
slice
np.asarray(payload, dtype=np.float32).reshape
weighted.transpose
np.divide
tsdf_geometry_from_samples
```

## `_component_rank`

[定义位置](../../../manipulation/tsdf_geometry.py#L243)。

输入签名：

```python
def _component_rank(vertices):
```

输出分支的源码表达式：

```python
(int(np.count_nonzero(singular > tolerance)), basis)
```

直接调用：

```python
np.linalg.svd
np.finfo
max
float
np.max
np.abs
int
np.count_nonzero
```

## `_component_triangles`

[定义位置](../../../manipulation/tsdf_geometry.py#L251)。

输入签名：

```python
def _component_triangles(vertices, rank=None, basis=None):
```

输出分支的源码表达式：

```python
np.empty((0, 3, 3), dtype=np.float64)
vertices[ConvexHull(vertices).simplices]
vertices[faces]
```

直接调用：

```python
_component_rank
np.empty
ConvexHull
np.asarray
range
len
```

## `coal_component`

[定义位置](../../../manipulation/tsdf_geometry.py#L265)。

输入签名：

```python
def coal_component(vertices):
```

输出分支的源码表达式：

```python
(coal.Sphere(0.0), pose)
(coal.Capsule(0.0, length), pose)
(shape, pose)
(coal.ConvexBase.convexHull(native, True, 'Qt'), pose)
```

直接写入的状态或数组位置：`pose[:3, 3]`、`pose[:3, :3]`。

直接调用：

```python
np.asarray
len
np.isfinite(points).all
np.isfinite
ValueError
_component_rank
np.eye
coal.Sphere
np.argmin
np.argmax
float
np.linalg.norm
np.abs
np.cross
np.column_stack
coal.Capsule
_component_triangles
coal.BVHModelOBBRSS
shape.beginModel
shape.addTriangle
shape.endModel
coal.StdVec_Vec3s
native.append
coal.ConvexBase.convexHull
```

## `_decode_component_arrays`

[定义位置](../../../manipulation/tsdf_geometry.py#L307)。

输入签名：

```python
def _decode_component_arrays(record, kind, maximum_vertices):
```

输出分支的源码表达式：

```python
(vertices, offsets)
```

直接调用：

```python
decode_array
np.isfinite(vertices).all
np.isfinite
len
np.any
np.diff
ValueError
```

## `_decode_tsdf_source`

[定义位置](../../../manipulation/tsdf_geometry.py#L318)。

输入签名：

```python
def _decode_tsdf_source(record):
```

输出分支的源码表达式：

```python
(origin, shape, size, surface_vertices, surface_offsets, negative_vertices, negative_offsets, cells)
```

直接调用：

```python
ValueError
np.asarray
float
np.isfinite(origin).all
np.isfinite
np.any
_decode_component_arrays
decode_array
np.prod
len
np.diff
```

## `iter_source_components`

[定义位置](../../../manipulation/tsdf_geometry.py#L338)。

输入签名：

```python
def iter_source_components(record):
```

生成输出的源码表达式：

```python
(kind, index, vertices[offsets[index]:offsets[index + 1]])
('negative_cell', index, lower + _CORNERS * size)
```

直接调用：

```python
_decode_tsdf_source
range
len
enumerate
np.asarray
np.unravel_index
tuple
```

## `ObservedTsdfGeometry`

[定义位置](../../../manipulation/tsdf_geometry.py#L356)。

```python
class ObservedTsdfGeometry:
    ...
    ...
```

该类型的直接方法：`pending_storage_total_nbytes`、`pending_storage_nbytes`、`initial_allocation_bound`、`__init__`、`collision_cache_storage_nbytes`、`component_preparation_storage_nbytes`、`prepared_components`、`iter_components`、`_iter_component_ids`、`component_indices`、`component_identity`、`_component_vertices`、`iter_collision_components`、`collision_component`。

## `ObservedTsdfGeometry.pending_storage_total_nbytes`

[定义位置](../../../manipulation/tsdf_geometry.py#L372)。

输入签名：

```python
@classmethod
def ObservedTsdfGeometry.pending_storage_total_nbytes(cls):
```

输出分支的源码表达式：

```python
sum((value.pending_storage_nbytes for value in cls._live_geometries))
```

直接调用：

```python
sum
```

## `ObservedTsdfGeometry.pending_storage_nbytes`

[定义位置](../../../manipulation/tsdf_geometry.py#L376)。

输入签名：

```python
@property
def ObservedTsdfGeometry.pending_storage_nbytes(self):
```

直接读取的对象属性：`self._prepared_components`、`self._collision_cache_capacity`、`self._collision_cache`、`self._prepared_component_capacity_nbytes`。

输出分支的源码表达式：

```python
(self._collision_cache_capacity - len(self._collision_cache)) * 4096 + self._prepared_component_capacity_nbytes - prepared_bytes
```

直接调用：

```python
int
len
```

## `ObservedTsdfGeometry.initial_allocation_bound`

[定义位置](../../../manipulation/tsdf_geometry.py#L390)。

输入签名：

```python
@staticmethod
def ObservedTsdfGeometry.initial_allocation_bound(record):
```

输出分支的源码表达式：

```python
3 * sum(sizes.values()) + count * (4096 + 256 + 64 + 128 + 2 * 512) + counts[0] * (4 * 512 + 512) + 8 * 1024 ** 2 + 16384
```

直接写入的状态或数组位置：`sizes[name]`。

直接调用：

```python
isinstance
set
ValueError
_array_layout
len
name.endswith
counts.append
int
sum
sizes.values
```

## `ObservedTsdfGeometry.__init__`

[定义位置](../../../manipulation/tsdf_geometry.py#L423)。

输入签名：

```python
def ObservedTsdfGeometry.__init__(self, record, *, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
```

直接读取的对象属性：`self.initial_allocation_bound`、`self.pending_storage_total_nbytes`、`self.negative_cell_flat_indices`、`self.shape_xyz`、`self.surface_offsets`、`self.negative_offsets`、`self.surface_vertices`、`self.negative_vertices`、`self.origin`、`self.voxel_size`、`self._lower`、`self._upper`、`self._counts`、`self._prunable`、`self.surface_shape`、`self.surface_lower_dimensional_indices`、`self._spatial_index`、`self._collision_cache_capacity`、`self.collision_cache_storage_nbytes`、`self.component_preparation_storage_nbytes`、`self._live_geometries`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.origin`、`self.shape_xyz`、`self.voxel_size`、`self.surface_vertices`、`self.surface_offsets`、`self.negative_vertices`、`self.negative_offsets`、`self.negative_cell_flat_indices`、`self._counts`、`self._lower`、`self._upper`、`self.bounds`、`self._prunable`、`self._prunable[index]`、`self.surface_lower_dimensional_indices`、`self.surface_shape`、`self._prunable[self._counts[0] + selected]`、`self._spatial_index`、`self.storage_nbytes`、`self._collision_cache_capacity`、`self._collision_cache`、`self._component_table`、`self._prepared_components`、`self._prepared_components_key`、`self._prepared_component_capacity_nbytes`。

直接调用：

```python
ValueError
require_host_memory
self.initial_allocation_bound
self.pending_storage_total_nbytes
RuntimeError
_decode_tsdf_source
len
lower.append
np.minimum.reduceat
upper.append
np.maximum.reduceat
np.column_stack
np.unravel_index
tuple
np.concatenate
np.empty
self._lower.min
self._upper.max
np.ones
sum
range
_component_rank
lower_dimensional.append
triangles.append
_component_triangles
np.asarray
coal.BVHModelOBBRSS
self.surface_shape.beginModel
self.surface_shape.addTriangle
self.surface_shape.endModel
np.diff
np.flatnonzero
np.arange
np.linalg.svd
np.finfo
np.maximum
np.abs(points).max
np.abs
(singular > tolerance[:, None]).sum
ComponentAABBTree
ComponentTable
self._live_geometries.add
```

## `ObservedTsdfGeometry.collision_cache_storage_nbytes`

[定义位置](../../../manipulation/tsdf_geometry.py#L525)。

输入签名：

```python
@property
def ObservedTsdfGeometry.collision_cache_storage_nbytes(self):
```

直接读取的对象属性：`self._collision_cache_capacity`。

输出分支的源码表达式：

```python
self._collision_cache_capacity * 4096
```

## `ObservedTsdfGeometry.component_preparation_storage_nbytes`

[定义位置](../../../manipulation/tsdf_geometry.py#L532)。

输入签名：

```python
@property
def ObservedTsdfGeometry.component_preparation_storage_nbytes(self):
```

直接读取的对象属性：`self._component_table`、`self._prepared_component_capacity_nbytes`。

输出分支的源码表达式：

```python
int(self._component_table.storage_nbytes) + self._prepared_component_capacity_nbytes
```

直接调用：

```python
int
```

## `ObservedTsdfGeometry.prepared_components`

[定义位置](../../../manipulation/tsdf_geometry.py#L536)。

输入签名：

```python
def ObservedTsdfGeometry.prepared_components(self, owner_pose):
```

直接读取的对象属性：`self._prepared_components_key`、`self._component_table`、`self._prepared_component_capacity_nbytes`、`self._prepared_components`。

输出分支的源码表达式：

```python
self._prepared_components
```

直接写入的状态或数组位置：`self._prepared_components`、`self._prepared_components_key`。

直接调用：

```python
np.asarray
np.dtype
ValueError
bool
pose.tobytes
floating_point_mode_key
tuple
np.geterr().items
np.geterr
PreparedComponents
int
RuntimeError
```

## `ObservedTsdfGeometry.iter_components`

[定义位置](../../../manipulation/tsdf_geometry.py#L567)。

输入签名：

```python
def ObservedTsdfGeometry.iter_components(self, bounds=None, *, surface=True, negative=True):
```

直接读取的对象属性：`self._iter_component_ids`、`self._component_vertices`。

生成输出的源码表达式：

```python
(kind, index, self._component_vertices(kind, index))
```

直接调用：

```python
self._iter_component_ids
self._component_vertices
```

## `ObservedTsdfGeometry._iter_component_ids`

[定义位置](../../../manipulation/tsdf_geometry.py#L580)。

输入签名：

```python
def ObservedTsdfGeometry._iter_component_ids(self, bounds=None, *, surface=True, negative=True):
```

直接读取的对象属性：`self.component_indices`、`self.component_identity`。

生成输出的源码表达式：

```python
self.component_identity(int(index))
```

直接调用：

```python
self.component_indices
self.component_identity
int
```

## `ObservedTsdfGeometry.component_indices`

[定义位置](../../../manipulation/tsdf_geometry.py#L586)。

输入签名：

```python
def ObservedTsdfGeometry.component_indices(self, bounds=None, *, surface=True, negative=True):
```

直接读取的对象属性：`self._counts`、`self._spatial_index`。

输出分支的源码表达式：

```python
candidates
```

直接调用：

```python
sum
np.arange
np.asarray
np.isfinite([lower, upper]).all
np.isfinite
np.any
ValueError
self._spatial_index.overlap_ids
```

## `ObservedTsdfGeometry.component_identity`

[定义位置](../../../manipulation/tsdf_geometry.py#L601)。

输入签名：

```python
def ObservedTsdfGeometry.component_identity(self, flat_index):
```

直接读取的对象属性：`self._counts`。

输出分支的源码表达式：

```python
('surface', flat_index)
('negative', flat_index - surfaces)
('negative_cell', flat_index - surfaces - negatives)
```

直接调用：

```python
type
sum
ValueError
```

## `ObservedTsdfGeometry._component_vertices`

[定义位置](../../../manipulation/tsdf_geometry.py#L612)。

输入签名：

```python
def ObservedTsdfGeometry._component_vertices(self, kind, index):
```

直接读取的对象属性：`self.surface_vertices`、`self.surface_offsets`、`self.negative_vertices`、`self.negative_offsets`、`self._counts`、`self._lower`、`self.voxel_size`。

输出分支的源码表达式：

```python
self.surface_vertices[self.surface_offsets[index]:self.surface_offsets[index + 1]]
self.negative_vertices[self.negative_offsets[index]:self.negative_offsets[index + 1]]
self._lower[offset] + _CORNERS * self.voxel_size
```

## `ObservedTsdfGeometry.iter_collision_components`

[定义位置](../../../manipulation/tsdf_geometry.py#L626)。

输入签名：

```python
def ObservedTsdfGeometry.iter_collision_components(self, bounds=None):
```

直接读取的对象属性：`self.component_indices`、`self.collision_component`。

生成输出的源码表达式：

```python
self.collision_component(int(flat_index))
```

直接调用：

```python
self.component_indices
self.collision_component
int
```

## `ObservedTsdfGeometry.collision_component`

[定义位置](../../../manipulation/tsdf_geometry.py#L639)。

输入签名：

```python
def ObservedTsdfGeometry.collision_component(self, flat_index):
```

直接读取的对象属性：`self.component_identity`、`self._collision_cache`、`self._component_vertices`。

输出分支的源码表达式：

```python
(kind, index, *component)
```

直接写入的状态或数组位置：`self._collision_cache[key]`。

直接调用：

```python
self.component_identity
self._collision_cache.get
self._component_vertices
coal_component
local.setflags
points.setflags
```
