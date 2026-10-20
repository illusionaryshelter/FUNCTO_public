# `manipulation/scene_surface_batch.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `selected_body_pairs`

[定义位置](../../../manipulation/scene_surface_batch.py#L22)。

输入签名：

```python
def selected_body_pairs(actor, targets, decisions, visited):
```

生成输出的源码表达式：

```python
(target, key, selected)
```

直接调用：

```python
decisions.get
tuple
sorted
visited.add
```

## `surface_batch_workspace_bytes`

[定义位置](../../../manipulation/scene_surface_batch.py#L43)。

输入签名：

```python
def surface_batch_workspace_bytes(pair_bound, request):
```

输出分支的源码表达式：

```python
int(pair_bound * 16384 + workers * (8 * 1024 ** 2 + solver))
```

直接调用：

```python
min
int
```

## `SurfaceDistanceBatch`

[定义位置](../../../manipulation/scene_surface_batch.py#L59)。

```python
class SurfaceDistanceBatch:
    ...
    ...
```

该类型的直接方法：`__init__`、`_prepare`、`consume`、`scheduled_query_count`、`cache_hit_count`、`native_worker_count`、`storage_nbytes`。

## `SurfaceDistanceBatch.__init__`

[定义位置](../../../manipulation/scene_surface_batch.py#L67)。

输入签名：

```python
def SurfaceDistanceBatch.__init__(self, actors, targets, decisions, request, surface_cache):
```

直接读取的对象属性：`self._slots`、`self._prepare`、`self._first_shapes`、`self._first_transforms`、`self._second_shapes`、`self._second_transforms`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._request`、`self._cache`、`self._slots`、`self._first_shapes`、`self._first_transforms`、`self._second_shapes`、`self._second_transforms`、`self._batch`、`self._cache_hit_count`、`self._slots[key]`。

直接调用：

```python
set
selected_body_pairs
isinstance
self._prepare
ordered_distances
```

## `SurfaceDistanceBatch._prepare`

[定义位置](../../../manipulation/scene_surface_batch.py#L101)。

输入签名：

```python
def SurfaceDistanceBatch._prepare(self, actor, target, shadow):
```

直接读取的对象属性：`self._request`、`self._cache`、`self._first_shapes`、`self._first_transforms`、`self._second_shapes`、`self._second_transforms`。

输出分支的源码表达式：

```python
(cache_key, signature, token, miss)
```

直接写入的状态或数组位置：`self._cache_hit_count`、`shadow[cache_key]`。

直接调用：

```python
id
self._request.saveToString
actor['pose'].tobytes
target['pose'].tobytes
shadow.get
self._cache.get
coal.Transform3s
len
self._first_shapes.append
self._first_transforms.append
self._second_shapes.append
self._second_transforms.append
```

## `SurfaceDistanceBatch.consume`

[定义位置](../../../manipulation/scene_surface_batch.py#L128)。

输入签名：

```python
def SurfaceDistanceBatch.consume(self, actor, target):
```

直接读取的对象属性：`self._slots`、`self._batch`、`result.min_distance`、`result.getNearestPoint1`、`result.getNearestPoint2`、`self._request`、`self._cache`。

输出分支的源码表达式：

```python
(value, witness_first, witness_second)
```

直接写入的状态或数组位置：`self._cache[cache_key]`。

直接调用：

```python
tuple
sorted
isinstance
self._batch.result
float
np.isfinite
RuntimeError
np.asarray(result.getNearestPoint1()).copy
np.asarray
result.getNearestPoint1
np.asarray(result.getNearestPoint2()).copy
result.getNearestPoint2
self._request.saveToString
self._cache.pop
```

## `SurfaceDistanceBatch.scheduled_query_count`

[定义位置](../../../manipulation/scene_surface_batch.py#L157)。

输入签名：

```python
@property
def SurfaceDistanceBatch.scheduled_query_count(self):
```

直接读取的对象属性：`self._first_shapes`。

输出分支的源码表达式：

```python
len(self._first_shapes)
```

直接调用：

```python
len
```

## `SurfaceDistanceBatch.cache_hit_count`

[定义位置](../../../manipulation/scene_surface_batch.py#L161)。

输入签名：

```python
@property
def SurfaceDistanceBatch.cache_hit_count(self):
```

直接读取的对象属性：`self._cache_hit_count`。

输出分支的源码表达式：

```python
self._cache_hit_count
```

## `SurfaceDistanceBatch.native_worker_count`

[定义位置](../../../manipulation/scene_surface_batch.py#L165)。

输入签名：

```python
@property
def SurfaceDistanceBatch.native_worker_count(self):
```

直接读取的对象属性：`self._batch`。

输出分支的源码表达式：

```python
0 if self._batch is None or isinstance(self._batch, Exception) else self._batch.worker_count
```

直接调用：

```python
isinstance
```

## `SurfaceDistanceBatch.storage_nbytes`

[定义位置](../../../manipulation/scene_surface_batch.py#L169)。

输入签名：

```python
@property
def SurfaceDistanceBatch.storage_nbytes(self):
```

直接读取的对象属性：`self._batch`。

输出分支的源码表达式：

```python
0 if self._batch is None or isinstance(self._batch, Exception) else self._batch.storage_nbytes
```

直接调用：

```python
isinstance
```
