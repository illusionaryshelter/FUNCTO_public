# `manipulation/progress_clock.py` 实现契约

代码基线：`CURRENT_IMPLEMENTATION`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `progress_anchor_parameters`

[定义位置](../../../manipulation/progress_clock.py#L6)。

输入签名：

```python
def progress_anchor_parameters(u, s, anchors):
```

实现说明：

> Return canonical binary64 (time parameter, progress) anchor records.
>
> A plateau contributes its entrance and exit. Native tensor consumers
> additionally verify float32 representability in _ProgressMap; continuous
> CPU geometry retains the original binary64 owner-path clock partitions.

显式输出分支：

```python
pairs
```

读取的对象属性：

```python
np.flatnonzero
pairs.append
np.searchsorted
np.asarray(sorted(set(pairs)), dtype=float).reshape
np.asarray
```

写入目标：

```python
pairs
value
equal
right
fraction
```

直接调用符号：

```python
np.flatnonzero
len
pairs.append
float
int
np.searchsorted
ValueError
np.asarray(sorted(set(pairs)), dtype=float).reshape
np.asarray
sorted
set
```

## `progress_query_parameters`

[定义位置](../../../manipulation/progress_clock.py#L30)。

输入签名：

```python
def progress_query_parameters(u, s, times, duration, anchors):
```

实现说明：

> Sample one physical clock and restore its canonical event coordinates.

显式输出分支：

```python
(parameters, progress)
```

读取的对象属性：

```python
np.asarray
np.interp
```

写入目标：

```python
times
parameters
progress
parameter
value
matching
parameters[matching]
progress[matching]
```

直接调用符号：

```python
np.asarray
np.interp
progress_anchor_parameters
```

## `_ProgressMap`

[定义位置](../../../manipulation/progress_clock.py#L43)。

```python
class _ProgressMap():
```

类字段：

```python
u: np.ndarray
s: np.ndarray
event_clock: dict = field(default_factory=dict, compare=False)
```

实现说明：

> A fixed, monotone phase schedule; repeated s values encode pauses.

## `_ProgressMap.__post_init__`

[定义位置](../../../manipulation/progress_clock.py#L52)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.u.ndim
self.u
self.s.shape
self.s
self.u.shape
np.isfinite(self.u).all
np.isfinite
np.isfinite(self.s).all
np.any
np.diff
self.u.astype
np.float32
```

写入目标：

无显式条目。

直接调用符号：

```python
len
np.isfinite(self.u).all
np.isfinite
np.isfinite(self.s).all
np.any
np.diff
self.u.astype
ValueError
```

## `_ProgressMap.anchor_parameters`

[定义位置](../../../manipulation/progress_clock.py#L60)。

输入签名：

```python
def anchor_parameters(self, anchors):
```

实现说明：

> Preserve both ends of an s plateau, including a paused event.

显式输出分支：

```python
pairs
```

读取的对象属性：

```python
self.u
self.s
np.unique
pairs[:, column].astype
np.float32
```

写入目标：

```python
pairs
column
```

直接调用符号：

```python
progress_anchor_parameters
any
len
np.unique
pairs[:, column].astype
ValueError
```

## `_ProgressMap.query_parameters`

[定义位置](../../../manipulation/progress_clock.py#L72)。

输入签名：

```python
def query_parameters(self, times, duration, anchors):
```

实现说明：

> Restore declared event coordinates at their identical physical keys.
>
> Multiplication by duration creates the authoritative physical time.
> Division need not recover the same binary64 parameter. Event progress
> and normalized time therefore come directly from the same anchor
> record; ordinary times retain piecewise-linear interpolation.

显式输出分支：

```python
progress_query_parameters(self.u, self.s, times, duration, anchors)
```

读取的对象属性：

```python
self.u
self.s
```

写入目标：

无显式条目。

直接调用符号：

```python
progress_query_parameters
```
