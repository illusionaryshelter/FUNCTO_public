# `manipulation/planning_ik.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_validate_reports`

[定义位置](../../../manipulation/planning_ik.py#L13)。

输入签名：

```python
def _validate_reports(reports, count):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
len
ValueError
any
type
report.get
```

## `_measurement`

[定义位置](../../../manipulation/planning_ik.py#L23)。

输入签名：

```python
def _measurement(report, name):
```

输出分支的源码表达式：

```python
(1, 0.0)
(0, number) if np.isfinite(number) else (1, 0.0)
```

直接调用：

```python
report.get
float
np.isfinite
```

## `_quality`

[定义位置](../../../manipulation/planning_ik.py#L32)。

输入签名：

```python
def _quality(report):
```

输出分支的源码表达式：

```python
tuple(values)
```

直接调用：

```python
_measurement
values.extend
tuple
```

## `_tier`

[定义位置](../../../manipulation/planning_ik.py#L41)。

输入签名：

```python
def _tier(report, query_kind):
```

输出分支的源码表达式：

```python
(not report['task_satisfied'], not report['model_valid'], not report['native_pose_ik_success'] if query_kind == 'reference' else False)
```

## `rank_ik_candidates`

[定义位置](../../../manipulation/planning_ik.py#L47)。

输入签名：

```python
def rank_ik_candidates(candidates, reports, previous, *, count, joint_tolerance, query_kind):
```

输出分支的源码表达式：

```python
np.asarray(selected, dtype=int)
```

直接调用：

```python
np.asarray
len
np.isfinite(prior).all
np.isfinite
type
ValueError
_validate_reports
sorted
np.flatnonzero
np.isfinite(values).all
np.any
np.max
np.abs
selected.append
int
```

## `rank_ik_candidates.key`

[定义位置](../../../manipulation/planning_ik.py#L74)。

输入签名：

```python
def rank_ik_candidates.key(index):
```

输出分支的源码表达式：

```python
(*_tier(report, query_kind), float(np.linalg.norm(prior - values[index], axis=-1).min()), *_quality(report), int(index))
```

直接调用：

```python
_tier
float
np.linalg.norm(prior - values[index], axis=-1).min
np.linalg.norm
_quality
int
```

## `select_ik_chains`

[定义位置](../../../manipulation/planning_ik.py#L90)。

输入签名：

```python
def select_ik_chains(candidates_by_query, reports_by_query, previous, query_times_s, velocity_limits, *, count, joint_tolerance):
```

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
[]
result
```

直接写入的状态或数组位置：`next_states[index]`、`indices[query]`。

直接调用：

```python
np.asarray
list
len
np.isfinite(start).all
np.isfinite
np.isfinite(times).all
np.any
np.diff
np.isfinite(bounds).all
type
ValueError
np.max
np.abs
enumerate
zip
_validate_reports
row.get
isinstance
int
np.flatnonzero
np.isfinite(values).all
sorted
retained.append
layers.append
float
_quality
states.items
np.errstate
np.dot
np.isfinite(combined).all
options.append
states.values
np.stack
np.concatenate
np.column_stack
np.interp
range
any
result.append
positions.tolist
```
