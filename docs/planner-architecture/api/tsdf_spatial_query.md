# `manipulation/tsdf_spatial_query.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `component_batch_workspace_bytes`

[定义位置](../../../manipulation/tsdf_spatial_query.py#L20)。

输入签名：

```python
def component_batch_workspace_bytes(request):
```

输出分支的源码表达式：

```python
4096 * (4096 + 1024) + COMPONENT_QUERY_WORKERS * (8 * 1024 ** 2 + solver)
```

直接调用：

```python
int
```

## `_transform`

[定义位置](../../../manipulation/tsdf_spatial_query.py#L33)。

输入签名：

```python
def _transform(pose):
```

输出分支的源码表达式：

```python
coal.Transform3s(pose[:3, :3], pose[:3, 3])
```

直接调用：

```python
coal.Transform3s
```

## `iter_component_distances`

[定义位置](../../../manipulation/tsdf_spatial_query.py#L37)。

输入签名：

```python
def iter_component_distances(geometry, actor_shape, actor_pose, owner_pose, bounds, stop_distance, request, *, previous_component, statistics):
```

直接读取的对象属性：`result.min_distance`。

输出分支的源码表达式：

```python
None
```

生成输出的源码表达式：

```python
(identity, component, pose, result, 1)
(identity, component, pose, result, int(count))
(identity, component, pose, near_result, 1)
```

直接写入的状态或数组位置：`statistics['query_count']`、`statistics['candidate_count']`、`statistics['completed']`、`statistics['completed_queries']`、`statistics['workspace_nbytes_upper']`、`execution['physical_leaf_query_count']`、`statistics['seed_query_count']`、`statistics['prior_hint_query_count']`、`errors[seed_id]`、`execution['unconsumed_leaf_query_count']`、`execution['native_batch_count']`、`execution['worker_count_max']`。

直接调用：

```python
geometry.component_indices
len
statistics.setdefault
geometry.prepared_components
_transform
geometry._spatial_index.query
max
int
np.zeros
search.seed_id
prepare
np.asarray
evidence
coal.DistanceResult
record_queries
coal.distance
math.isfinite
float
RuntimeError
certify
retained
np.searchsorted
min
prepared.scan
batch.prefix
```

## `iter_component_distances.record_queries`

[定义位置](../../../manipulation/tsdf_spatial_query.py#L81)。

输入签名：

```python
def iter_component_distances.record_queries(indices):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`queried[positions]`、`statistics['exact_leaf_query_count']`、`execution['consumed_leaf_query_count']`、`execution['unconsumed_leaf_query_count']`。

直接调用：

```python
np.searchsorted
len
```

## `iter_component_distances.prepare`

[定义位置](../../../manipulation/tsdf_spatial_query.py#L93)。

输入签名：

```python
def iter_component_distances.prepare(indices):
```

输出分支的源码表达式：

```python
(indices[:count], new_components)
```

直接写入的状态或数组位置：`errors[identity]`、`new_components[identity]`、`execution['prepared_component_count']`、`execution['reused_component_count']`。

直接调用：

```python
prepared.missing
len
int
geometry.collision_component
np.searchsorted
new_ids.append
shapes.append
poses.append
prepared.register
np.asarray
```

## `iter_component_distances.evidence`

[定义位置](../../../manipulation/tsdf_spatial_query.py#L123)。

输入签名：

```python
def iter_component_distances.evidence(identity, created):
```

输出分支的源码表达式：

```python
(component, prepared.pose(identity))
```

直接调用：

```python
created.get
geometry.collision_component
prepared.pose
```

## `iter_component_distances.certify`

[定义位置](../../../manipulation/tsdf_spatial_query.py#L130)。

输入签名：

```python
def iter_component_distances.certify(component, pose, result):
```

直接读取的对象属性：`result.min_distance`。

输出分支的源码表达式：

```python
False
True
```

直接调用：

```python
float
certified_distance_upper_bound
_transform
math.isfinite
```

## `iter_component_distances.retained`

[定义位置](../../../manipulation/tsdf_spatial_query.py#L166)。

输入签名：

```python
def iter_component_distances.retained():
```

输出分支的源码表达式：

```python
selected
```

直接写入的状态或数组位置：`statistics['retained_pass_count']`、`statistics['node_bound_count']`、`kept[np.searchsorted(candidates, selected)]`、`statistics['bound_pruned_count']`、`pruned[newly_pruned]`。

直接调用：

```python
search.retained_ids
int
np.searchsorted
len
np.insert(selected, position, forced).astype
np.insert
np.zeros
np.count_nonzero
```
