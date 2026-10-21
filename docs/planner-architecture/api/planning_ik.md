# `manipulation/planning_ik.py` 实现契约

代码基线：`a2b3f686b866aea18af47c302bafb62e44212e22`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_validate_reports`

[定义位置](../../../manipulation/planning_ik.py#L13)。

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

## `_measurement`

[定义位置](../../../manipulation/planning_ik.py#L23)。

输入签名：

```python
def _measurement(report, name):
```

实现说明：

> Return missingness and a finite measurement without inventing infinity.

显式输出分支：

```python
(1, 0.0)
(0, number) if np.isfinite(number) else (1, 0.0)
```

读取的对象属性：

```python
report.get
np.isfinite
```

写入目标：

```python
value
number
```

直接调用符号：

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

实现说明：

> Supply additive, independent quality tiebreakers in documented units.

显式输出分支：

```python
tuple(values)
```

读取的对象属性：

```python
values.extend
```

写入目标：

```python
values
name
missing
measurement
```

直接调用符号：

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

实现说明：

> Keep task/source checks independent of optional nominal pose success.

显式输出分支：

```python
(not report['task_satisfied'], not report['model_valid'], not report['native_pose_ik_success'] if query_kind == 'reference' else False)
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `rank_ik_candidates`

[定义位置](../../../manipulation/planning_ik.py#L47)。

输入签名：

```python
def rank_ik_candidates(candidates, reports, previous, *, count, joint_tolerance, query_kind):
```

实现说明：

> Return bounded distinct proposals, without assembling a trajectory.
>
> ``query_kind='reference'`` prefers successful nominal-pose IK within the
> same task/source screening tier. ``query_kind='functional'`` does not use
> that flag: the declared allowed set can pass despite failure to reach its
> nominal centre. Both kinds retain finite recovery proposals, explicitly
> leaving their failed checks in the reports; they never reject a skill.
>
> Within one tier, distance to the nearest supplied previous state precedes
> extra environment clearance, self clearance and joint-limit margin. These
> optional quality measurements cannot buy a large posture jump. Missing or
> nonfinite measurements lose their quality tiebreak against finite ones.
> Numerical duplicates are removed after ordering so better qualified
> evidence owns the representative. This local ordering is only for bounded
> proposals/warm starts; ``select_ik_chains`` chooses complete reference chains.

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

[定义位置](../../../manipulation/planning_ik.py#L74)。

输入签名：

```python
def key(index):
```

显式输出分支：

```python
(*_tier(report, query_kind), float(np.linalg.norm(prior - values[index], axis=-1).min()), *_quality(report), int(index))
```

读取的对象属性：

```python
np.linalg.norm(prior - values[index], axis=-1).min
np.linalg.norm
np.linalg
```

写入目标：

```python
report
```

直接调用符号：

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
def select_ik_chains(candidates_by_query, reports_by_query, previous, query_times_s, velocity_limits, *, count, joint_tolerance, query_kind='reference'):
```

实现说明：

> Connect qualified reference anchors and interpolate missing seed samples.
>
> Inputs retain the caller's candidate indices. ``previous`` is the measured
> start of shape ``[1,D]`` at t=0; strictly increasing ``query_times_s`` are
> physical seconds, and ``velocity_limits`` contains named-order native
> lower/upper bounds of shape ``[2,D]``. No wraparound, retiming or extra IK
> query is introduced. Each report requires an explicit ``candidate_source``;
> only ``native_ik`` may report native pose success.
>
> Finite native IK successes can be anchors even when static task or source
> checks fail. Native success includes the native solver's constraints and
> convergence; it does not establish observation coverage or task success.
> Failed native queries do not become graph vertices or joint targets.
> The graph connects successive SOLVED queries using their actual time gap;
> missing samples are interpolated between the measured start and selected
> anchors, with constant continuation after the last anchor. Those samples
> are explicitly FALLBACK_SEED, not solutions to their requested tool pose.
> With no qualified anchor, return no reference chain: the caller owns its
> independent measured-state initializer. Every emitted chain still needs
> whole-curve optimization and validation.
>
> ``query_kind='functional'`` instead requires a task-qualified state at
> every supplied event and rejects connections whose average velocity is
> outside any joint's signed model bounds. Native pose-centre convergence
> is irrelevant to these continuous allowed sets. Static scene failures
> remain explicit proposal quality; the whole curve still owns acceptance.
> Reference samples are soft observations and do not impose those mandatory
> interpolation nodes or a hard velocity test on a reference polyline.
>
> Dynamic programming retains at most ``count`` paths at each last candidate.
> Failed static task/model check counts rank proposals first, without
> removing the only available transferred-reference chain. Edges then
> integrate squared joint velocity divided by the square of the
> joint's largest absolute native velocity bound. Dividing by the final
> query time and D yields the normalized velocity energy of the selected
> joint-space polyline. This first-order additive cost makes the bounded
> dynamic program exact for these supplied candidates. Clearance and joint
> margin break only equal-energy ties. They never certify the edges.

显式输出分支：

```python
[]
[]
result
```

读取的对象属性：

```python
np.asarray
start.ndim
start.shape
np.isfinite(start).all
np.isfinite
times.shape
np.isfinite(times).all
np.any
np.diff
bounds.shape
np.isfinite(bounds).all
np.max
np.abs
values.ndim
values.shape
row.get
np.flatnonzero
np.isfinite(values).all
retained.append
layers.append
states.items
np.errstate
np.dot
np.isfinite(combined).all
options.append
states.values
np.stack
np.r_
np.concatenate
np.column_stack
np.interp
result.append
positions.tolist
```

写入目标：

```python
pools
values
reports
start
times
bounds
scales
layers
query
rows
row
source
qualified
index
retained
states
previous_time
normalizer
layer
indices
dt
next_states
quality
task_failure
model_failure
options
preceding_index
paths
preceding
displacement
scaled_velocity
edge
cost
path
combined
old
new
next_states[index]
result
item
anchor_queries
_
anchors
anchor_times
anchor_positions
positions
joint
prior
indices[query]
recovery
```

直接调用符号：

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
_tier
_quality
retained.append
layers.append
float
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
