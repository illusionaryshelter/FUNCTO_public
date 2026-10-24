# `manipulation/planning_ik.py` 实现契约

代码基线：`CURRENT_IMPLEMENTATION`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_validate_reports`

[定义位置](../../../manipulation/planning_ik.py#L11)。

输入签名：

```python
def _validate_reports(reports, count):
```

实现说明：

> Require separate evidence for task checks, source geometry and pose IK.

显式输出分支：

无显式条目。

读取的对象属性：

```python
report.get
np.bool_
```

写入目标：

```python
report
name
```

直接调用符号：

```python
len
ValueError
any
type
report.get
```

## `_static_qualification`

[定义位置](../../../manipulation/planning_ik.py#L22)。

输入签名：

```python
def _static_qualification(report):
```

实现说明：

> Return known task/model failures and separate unestablished verdicts.
>
> ``planning._ConstrainedPlanner._rank_ik_candidates`` owns these reports. A
> completed task/native check provides finite positive violation values;
> NaN and infinities provide no definite violation measurement. The scene
> component families below are the definite constraints produced by
> ``OnlinePlanningScene.components``: registered-entity intersections,
> required contact, declared initial support and gripper control. Unknown
> coverage, numerical boundaries and missing model relations retain their
> original components but cannot supply a definite failure here.
>
> A failure and an unestablished verdict are mutually exclusive ranking
> states. An explicit violation wins when the same report also contains
> unknown relations; the report itself retains both pieces of evidence.
> Functional consumers use this original allowed-set evidence first.
> The search branch rechecks scene qualification using its own prefix.

显式输出分支：

```python
(bool(task_failed), bool(model_failed), not task_failed and (not evaluated or not report['task_satisfied']), not model_failed and (not evaluated or not report['model_valid']))
```

读取的对象属性：

```python
report.get
report.get('task_violations', {}).values
report.get('native_violations', {}).values
report.get('scene_components', {}).items
name.startswith
```

写入目标：

```python
evaluated
task_failed
value
model_failed
name
component
```

直接调用符号：

```python
report.get
any
positive
report.get('task_violations', {}).values
report.get('native_violations', {}).values
report.get('scene_components', {}).items
name.startswith
bool
```

## `_static_qualification.positive`

[定义位置](../../../manipulation/planning_ik.py#L42)。

输入签名：

```python
def positive(value):
```

显式输出分支：

```python
np.isfinite(value) and value > 0
```

读取的对象属性：

```python
np.isfinite
```

写入目标：

无显式条目。

直接调用符号：

```python
np.isfinite
```

## `rank_ik_candidates`

[定义位置](../../../manipulation/planning_ik.py#L59)。

输入签名：

```python
def rank_ik_candidates(candidates, reports, previous, *, count, joint_tolerance):
```

实现说明：

> Return distinct finite functional proposals ordered by original evidence.
>
> Definite task/model failures and unestablished verdicts precede distance
> to the supplied numerical seed. Current branch validity owns the final
> scene decision. Source index breaks exact ties; no nominal-pose convergence
> flag or unmeasured quality proxy changes this allowed-set ordering.

显式输出分支：

```python
np.asarray(selected, dtype=int)
```

读取的对象属性：

```python
np.asarray
values.ndim
values.shape
prior.ndim
prior.shape
np.isfinite(prior).all
np.isfinite
np.flatnonzero
np.isfinite(values).all
np.any
np.max
np.abs
selected.append
```

写入目标：

```python
values
prior
selected
index
```

直接调用符号：

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

[定义位置](../../../manipulation/planning_ik.py#L76)。

输入签名：

```python
def key(index):
```

显式输出分支：

```python
(*_static_qualification(reports[index]), continuity, int(index))
```

读取的对象属性：

```python
np.linalg.norm(prior - values[index], axis=-1).min
np.linalg.norm
np.linalg
```

写入目标：

```python
continuity
```

直接调用符号：

```python
float
np.linalg.norm(prior - values[index], axis=-1).min
np.linalg.norm
_static_qualification
int
```
