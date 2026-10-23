# `manipulation/planning_ik.py` 实现契约

代码基线：`774e121-plus-unified-workspace`。
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
> Original native success precedes numerical preferences. Every proposal
> then minimizes Euclidean joint displacement before using pose energy as
> a tiebreaker. Failed native outputs retain their approximation status and
> residual diagnostics under that same continuity-first numerical rule.
> Static task/model evidence breaks only equal numerical choices; an
> unqueried failed pose cannot displace a solved numerical reference.
> The original candidate index breaks exact ties. The returned index names
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
float
np.linalg.norm
_static_qualification
int
```

## `_validate_reports`

[定义位置](../../../manipulation/planning_ik.py#L51)。

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

[定义位置](../../../manipulation/planning_ik.py#L61)。

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

[定义位置](../../../manipulation/planning_ik.py#L70)。

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

## `_static_qualification`

[定义位置](../../../manipulation/planning_ik.py#L80)。

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
> Numerical reference consumers use these values only after their
> numerical criteria. Functional consumers use this original allowed-set
> evidence first and do not require convergence to a nominal pose centre.

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

[定义位置](../../../manipulation/planning_ik.py#L101)。

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

[定义位置](../../../manipulation/planning_ik.py#L118)。

输入签名：

```python
def rank_ik_candidates(candidates, reports, previous, *, count, joint_tolerance, query_kind):
```

实现说明：

> Return bounded distinct proposals, without assembling a trajectory.
>
> Reference proposals compare original native success and numerical
> continuity before static evidence. Functional proposals compare definite
> task/model failures and unestablished verdicts before continuity, without
> using nominal-centre convergence. Both kinds retain finite proposals, explicitly
> leaving their failed checks in the reports; they never reject a skill.
>
> Distance to the nearest supplied previous state precedes
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

[定义位置](../../../manipulation/planning_ik.py#L145)。

输入签名：

```python
def key(index):
```

显式输出分支：

```python
(*prefix, *_quality(report), int(index))
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
continuity
static
prefix
```

直接调用符号：

```python
float
np.linalg.norm(prior - values[index], axis=-1).min
np.linalg.norm
_static_qualification
_quality
int
```

## `select_ik_chains`

[定义位置](../../../manipulation/planning_ik.py#L163)。

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
> Dynamic programming first minimizes native-failure count, then normalized
> joint velocity energy, failed-node pose energy and successful-node pose
> energy. Summed static evidence only breaks equal numerical paths.
> Unqueried or unresolved checks remain unestablished verdicts. Each last node keeps
> 'count' histories. Samples with no finite legal proposal are filled
> between available nodes and measured q0, without inventing pose success.
> The resulting values initialize the common constrained trajectory solve;
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
np.min
np.sum
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
preceding
continuity
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
static
options
preceding_index
paths
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
np.min
np.sum
sorted
np.flatnonzero
_static_qualification
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
