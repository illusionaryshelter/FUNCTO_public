# `manipulation/online_observed_geometry.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_readonly`

[定义位置](../../../manipulation/online_observed_geometry.py#L29)。

输入签名：

```python
def _readonly(array):
```

输出分支的源码表达式：

```python
array
```

直接写入的状态或数组位置：`array.flags.writeable`。

## `FrozenObservedGeometry`

[定义位置](../../../manipulation/online_observed_geometry.py#L34)。

```python
class FrozenObservedGeometry:
    ...
    ...
```

该类型的直接方法：`__init__`、`storage_nbytes`、`pending_cache_bytes`、`record`、`_vertices`、`prepare_rules`、`_active_bits`、`component_indices`、`collision_component`、`rule_region`、`surface_points`、`distance_field`。

## `FrozenObservedGeometry.__init__`

[定义位置](../../../manipulation/online_observed_geometry.py#L43)。

输入签名：

```python
def FrozenObservedGeometry.__init__(self, record, *, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
```

直接读取的对象属性：`self.maximum_cache_bytes`、`self.max_host_rss_bytes`、`self._record`、`self.surface_vertices`、`self.surface_offsets`、`self.negative_vertices`、`self.negative_offsets`、`self.negative_cell_flat_indices`、`self.shape_xyz`、`self.origin`、`self.voxel_size`、`self._lower`、`self._upper`、`self._index`、`self._box_local`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.max_host_rss_bytes`、`self.maximum_cache_bytes`、`self._record`、`self.origin`、`self.shape_xyz`、`self.voxel_size`、`self.surface_vertices`、`self.surface_offsets`、`self.negative_vertices`、`self.negative_offsets`、`self.negative_cell_flat_indices`、`self._counts`、`self.component_count`、`self._lower`、`self._upper`、`self.bounds`、`self._index`、`self._base_storage_bytes`、`self._cache`、`self._negative_cell_shape`、`self._box_local`、`self._cache_bytes`、`self._rule_records`、`self._rule_bits`、`self._removed_by_rule`、`self._regions`。

直接调用：

```python
_array_layout
tuple
int
name.endswith
min
ValueError
sum
require_host_memory
sizes.values
np.array_equal
np.asarray
deepcopy
_decode_tsdf_source
len
lower.append
np.minimum.reduceat
upper.append
np.maximum.reduceat
np.column_stack
np.unravel_index
np.ascontiguousarray
np.concatenate
np.empty
_readonly
self._lower.min
self._upper.max
ComponentAABBTree
np.ones
np.zeros
```

## `FrozenObservedGeometry.storage_nbytes`

[定义位置](../../../manipulation/online_observed_geometry.py#L105)。

输入签名：

```python
@property
def FrozenObservedGeometry.storage_nbytes(self):
```

直接读取的对象属性：`self._base_storage_bytes`、`self._removed_by_rule`、`self._cache_bytes`。

输出分支的源码表达式：

```python
self._base_storage_bytes + self._removed_by_rule.nbytes + self._cache_bytes
```

## `FrozenObservedGeometry.pending_cache_bytes`

[定义位置](../../../manipulation/online_observed_geometry.py#L109)。

输入签名：

```python
@property
def FrozenObservedGeometry.pending_cache_bytes(self):
```

直接读取的对象属性：`self.maximum_cache_bytes`、`self._cache_bytes`。

输出分支的源码表达式：

```python
self.maximum_cache_bytes - self._cache_bytes
```

## `FrozenObservedGeometry.record`

[定义位置](../../../manipulation/online_observed_geometry.py#L112)。

输入签名：

```python
def FrozenObservedGeometry.record(self):
```

直接读取的对象属性：`self._record`。

输出分支的源码表达式：

```python
deepcopy(self._record)
```

直接调用：

```python
deepcopy
```

## `FrozenObservedGeometry._vertices`

[定义位置](../../../manipulation/online_observed_geometry.py#L116)。

输入签名：

```python
def FrozenObservedGeometry._vertices(self, component_id):
```

直接读取的对象属性：`self.component_count`、`self._counts`、`self.surface_vertices`、`self.surface_offsets`、`self.negative_vertices`、`self.negative_offsets`、`self._lower`、`self.voxel_size`。

输出分支的源码表达式：

```python
('surface', component_id, self.surface_vertices[self.surface_offsets[component_id]:self.surface_offsets[component_id + 1]])
('negative', local, self.negative_vertices[self.negative_offsets[local]:self.negative_offsets[local + 1]])
('negative_cell', local, self._lower[component_id] + _CORNERS * self.voxel_size)
```

直接调用：

```python
isinstance
ValueError
int
```

## `FrozenObservedGeometry.prepare_rules`

[定义位置](../../../manipulation/online_observed_geometry.py#L130)。

输入签名：

```python
def FrozenObservedGeometry.prepare_rules(self, rules):
```

直接读取的对象属性：`self._rule_records`、`self.component_count`、`self.pending_cache_bytes`、`self.max_host_rss_bytes`、`self.surface_vertices`、`self.surface_offsets`、`self.negative_vertices`、`self.negative_offsets`、`self.voxel_size`、`self._lower`、`self._upper`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`records[rule.rule_id]`、`bits[rule.rule_id]`、`removed[base + start:base + stop]`、`removed[start:stop]`、`regions[rule.rule_id]`、`self._rule_records`、`self._rule_bits`、`self._removed_by_rule`、`self._regions`。

直接调用：

```python
tuple
len
ValueError
np.asarray(rule.patch_origin).tolist
np.asarray
np.asarray(rule.patch_axes).reshape(-1).tolist
np.asarray(rule.patch_axes).reshape
np.asarray(rule.patch_half_size).tolist
float
require_host_memory
int
np.zeros
enumerate
np.isfinite(axes).all
np.isfinite
np.isfinite(origin).all
np.isfinite(widths).all
np.any
np.full
range
min
np.minimum.reduceat
np.maximum.reduceat
(np.maximum(np.abs(lo), np.abs(hi)) <= widths + rule.numerical_position_tolerance_m).all
np.maximum
np.abs
inside.astype
inside.any
np.minimum
lo[inside].min
hi[inside].max
inside.sum
np.abs(axes).sum
(np.abs(center) + support <= widths + rule.numerical_position_tolerance_m).all
(center[inside] - support).min
(center[inside] + support).max
_readonly
```

## `FrozenObservedGeometry._active_bits`

[定义位置](../../../manipulation/online_observed_geometry.py#L205)。

输入签名：

```python
def FrozenObservedGeometry._active_bits(self, active_rule_ids):
```

直接读取的对象属性：`self._rule_bits`。

输出分支的源码表达式：

```python
sum((self._rule_bits[name] for name in identities))
```

直接调用：

```python
frozenset
self._rule_bits.keys
ValueError
sum
```

## `FrozenObservedGeometry.component_indices`

[定义位置](../../../manipulation/online_observed_geometry.py#L211)。

输入签名：

```python
def FrozenObservedGeometry.component_indices(self, bounds, active_rule_ids=()):
```

直接读取的对象属性：`self._active_bits`、`self.component_count`、`self._index`、`self._removed_by_rule`。

输出分支的源码表达式：

```python
ids[self._removed_by_rule[ids] & bits == 0] if bits else ids
```

直接调用：

```python
self._active_bits
np.arange
np.asarray
np.isfinite([lower, upper]).all
np.isfinite
np.any
ValueError
self._index.overlap_ids
```

## `FrozenObservedGeometry.collision_component`

[定义位置](../../../manipulation/online_observed_geometry.py#L224)。

输入签名：

```python
def FrozenObservedGeometry.collision_component(self, component_id):
```

直接读取的对象属性：`self.component_count`、`self._counts`、`self._negative_cell_shape`、`self.pending_cache_bytes`、`self.max_host_rss_bytes`、`self.voxel_size`、`self._lower`、`self._box_local`、`self._cache`、`self._vertices`。

输出分支的源码表达式：

```python
('negative_cell', identity - self._counts[0] - self._counts[1], self._negative_cell_shape, _readonly(pose), self._box_local)
self._cache[identity]
```

直接写入的状态或数组位置：`self._negative_cell_shape`、`self._cache_bytes`、`pose[:3, 3]`、`self._cache[identity]`。

直接调用：

```python
isinstance
ValueError
int
require_host_memory
coal.Box
np.eye
_readonly
self._vertices
coal_component
```

## `FrozenObservedGeometry.rule_region`

[定义位置](../../../manipulation/online_observed_geometry.py#L252)。

输入签名：

```python
def FrozenObservedGeometry.rule_region(self, rule_id):
```

直接读取的对象属性：`self._regions`。

输出分支的源码表达式：

```python
dict(self._regions[rule_id])
```

直接调用：

```python
ValueError
dict
```

## `FrozenObservedGeometry.surface_points`

[定义位置](../../../manipulation/online_observed_geometry.py#L258)。

输入签名：

```python
def FrozenObservedGeometry.surface_points(self, rule_id):
```

直接读取的对象属性：`self._rule_bits`、`self._removed_by_rule`、`self._counts`、`self.surface_vertices`、`self.pending_cache_bytes`、`self.max_host_rss_bytes`、`self.surface_offsets`。

输出分支的源码表达式：

```python
_readonly(np.unique(self.surface_vertices[mask], axis=0))
```

直接调用：

```python
ValueError
len
require_host_memory
int
np.repeat
np.diff
_readonly
np.unique
```

## `FrozenObservedGeometry.distance_field`

[定义位置](../../../manipulation/online_observed_geometry.py#L272)。

输入签名：

```python
def FrozenObservedGeometry.distance_field(self, origin, shape, voxel_size, active_rule_ids=(), *, excluded_component_ids=()):
```

直接读取的对象属性：`self.component_indices`、`self.component_count`、`self._lower`、`self._upper`、`self.pending_cache_bytes`、`self.max_host_rss_bytes`。

输出分支的源码表达式：

```python
None
_readonly(np.nextafter(values.astype(np.float32), np.float32(-np.inf)))
```

直接写入的状态或数组位置：`occupied[tuple((slice(int(a), int(b) + 1) for (a, b) in zip(lo, hi)))]`、`values[occupied]`。

直接调用：

```python
np.asarray
tuple
float
np.isfinite(origin).all
np.isfinite
len
any
isinstance
math.prod
ValueError
int
self.component_indices
np.any
np.unique
np.isin
require_host_memory
np.zeros
np.nextafter
np.ceil(np.nextafter(lower - 0.5, -np.inf)).astype
np.ceil
np.floor(np.nextafter(upper + 0.5, np.inf)).astype
np.floor
np.maximum
np.minimum
zip
slice
math.sqrt
occupied.all
np.full
distance_transform_edt
_readonly
values.astype
np.float32
```
