# `manipulation/planning_ik.py` 实现契约

代码基线：`A/B 工作区（实施起点 d97977c）`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `select_ik_tracking_candidate`

[定义位置](../../../manipulation/planning_ik.py#L14)。

输入签名：

```python
def select_ik_tracking_candidate(candidates, candidate_indices, pose_costs, previous, *, reports):
```

实现说明：

> Select one warm start from the original static and native IK evidence.
>
> ``candidates`` contains named-order joint positions with shape ``[N,D]``;
> ``candidate_indices`` identifies the caller's finite, position-domain
> proposals. ``pose_costs[N]`` contains their finite native pose energies
> for the same reference query. ``previous[1,D]`` is the preceding tracking
> state, initially the measured entrance. ``reports[N]`` contains the same
> query's task/model components and original native pose-success flags.
>
> The common qualification prefix precedes all numerical preferences.
> Successful proposals within one prefix minimize Euclidean
> joint displacement before using their pose energy as a tiebreaker.
> Failed proposals minimize pose energy before joint displacement. This
> preserves the failed solver's numerical progress without allowing tiny
> differences between solved poses to select a distant posture. The
> original candidate index breaks exact ties. The returned index names
> at most one warm start. The native SeedManager fills the remaining fixed
> seed budget. An empty input index set has no tracking proposal.

显式输出分支：

```python
None if selected is None else int(selected)
```

读取的对象属性：

```python
np.asarray
np.linalg.norm
np.linalg
```

写入目标：

```python
values
costs
solved
row
prior
selected
```

直接调用符号：

```python
np.asarray
_validate_reports
len
min
_qualification_prefix
float
np.linalg.norm
int
```

## `_validate_reports`

[定义位置](../../../manipulation/planning_ik.py#L49)。

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

[定义位置](../../../manipulation/planning_ik.py#L59)。

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

[定义位置](../../../manipulation/planning_ik.py#L68)。

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

## `_qualification_prefix`

[定义位置](../../../manipulation/planning_ik.py#L78)。

输入签名：

```python
def _qualification_prefix(report, query_kind):
```

实现说明：

> Return known task/model failures, native failure and unknown verdicts.
>
> ``planning._CuroboPlanner._rank_ik_candidates`` owns these reports. A
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
> Functional queries ignore convergence to the nominal pose centre.

显式输出分支：

```python
(bool(task_failed), bool(model_failed), not report['native_pose_ik_success'] if query_kind == 'reference' else False, not task_failed and (not evaluated or not report['task_satisfied']), not model_failed and (not evaluated or not report['model_valid']))
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

## `_qualification_prefix.positive`

[定义位置](../../../manipulation/planning_ik.py#L97)。

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

[定义位置](../../../manipulation/planning_ik.py#L115)。

输入签名：

```python
def rank_ik_candidates(candidates, reports, previous, *, count, joint_tolerance, query_kind):
```

实现说明：

> Return bounded distinct proposals, without assembling a trajectory.
>
> The common prefix compares definite task failure, definite model failure,
> native reference failure, then unestablished task/model verdicts.
> ``query_kind='functional'`` does not use
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

[定义位置](../../../manipulation/planning_ik.py#L143)。

输入签名：

```python
def key(index):
```

显式输出分支：

```python
(*_qualification_prefix(report, query_kind), float(np.linalg.norm(prior - values[index], axis=-1).min()), *_quality(report), int(index))
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
_qualification_prefix
float
np.linalg.norm(prior - values[index], axis=-1).min
np.linalg.norm
_quality
int
```

## `select_ik_chains`

[定义位置](../../../manipulation/planning_ik.py#L159)。

输入签名：

```python
def select_ik_chains(candidates_by_query, reports_by_query, previous, query_times_s, velocity_limits, *, position_limits, pose_costs_by_query, count, joint_tolerance):
```

实现说明：

> Form bounded coarse joint paths from finite position-domain IK states.
>
> Every numerical node retains its native pose-success flag. A legal
> nonconverged state is an approximate proposal, not a solved reference.
> 'pose_costs_by_query' supplies the untruncated native quadratic pose
> energy in the same candidate order. Position bounds admit nodes; velocity
> bounds only normalize the polyline's continuity cost. Neither the nodes
> nor their connecting edges certify the later continuous trajectory.
>
> Dynamic programming sums the common qualification prefix over nodes,
> then compares failed-node pose energy, normalized joint velocity energy
> and successful-node pose energy, in that order. Unqueried or unresolved
> checks contribute only to unestablished-verdict counts. Each last node keeps
> 'count' histories. Samples with no finite legal proposal are filled
> between available nodes and measured q0, without inventing pose success.
> The resulting values initialize the common direct tool-space optimizer;
> they never become mandatory event interpolation conditions.

显式输出分支：

```python
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
positions_bounds.shape
np.isfinite(positions_bounds).all
np.max
np.abs
values.ndim
values.shape
energy.shape
row.get
np.isfinite(values).all
(values >= positions_bounds[0]).all
(values <= positions_bounds[1]).all
np.flatnonzero
retained.append
layers.append
states.items
np.errstate
np.dot
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
costs
start
times
bounds
positions_bounds
scales
layers
query
rows
energy
row
legal
retained
order
index
states
previous_time
normalizer
layer
indices
dt
next_states
report
prefix
options
preceding_index
paths
preceding
scaled_velocity
edge
cost
path
solved
combined
item
value
next_states[index]
result
anchor_queries
_
anchors
anchor_times
anchor_positions
positions
joint
prior
indices[query]
statuses
observation_mask
status
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
np.isfinite(positions_bounds).all
type
ValueError
np.max
np.abs
enumerate
zip
_validate_reports
row.get
np.isfinite(values).all
(values >= positions_bounds[0]).all
(values <= positions_bounds[1]).all
sorted
np.flatnonzero
_qualification_prefix
retained.append
int
layers.append
float
states.items
np.errstate
np.dot
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
all
```
