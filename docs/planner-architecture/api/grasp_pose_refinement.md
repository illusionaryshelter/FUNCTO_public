# `manipulation/grasp_pose_refinement.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../16-cross-instance-grasp.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_check_parts`

[定义位置](../../../manipulation/grasp_pose_refinement.py#L23)。

输入签名：

```python
def _check_parts(tool_pose, parts, field):
```

实现说明：

> Query each explicit component/state against the unchanged world field.

显式输出分支：

```python
records
```

读取的对象属性：

```python
np.asarray
np.eye
records.append
```

写入目标：

```python
records
part
pose
started
answer
```

直接调用符号：

```python
np.asarray
perf_counter
query_native_field
np.eye
records.append
dict
```

## `_relation`

[定义位置](../../../manipulation/grasp_pose_refinement.py#L36)。

输入签名：

```python
def _relation(records):
```

实现说明：

> Aggregate the existing entity states without reclassifying uncertainty.

显式输出分支：

```python
'intersection'
'clear_represented_set'
'unresolved'
```

读取的对象属性：

无显式条目。

写入目标：

```python
row
```

直接调用符号：

```python
any
all
```

## `refine_pose`

[定义位置](../../../manipulation/grasp_pose_refinement.py#L45)。

输入签名：

```python
def refine_pose(T_world_tool, parts, field, basis_tool, max_tilt_rad, field_margin_m):
```

实现说明：

> Return one fixed-origin, two-axis orientation proposal and its evidence.
>
> Parts supplies unique id, geometry_id, state_id, registered Coal
> geometry, T_tool_geometry and guide_points_tool for each tested component.
> Guide points are in the same TCP frame as basis_tool's two orthonormal
> columns. Their cross product names the excluded spin axis. Field is the
> existing world-frame FrozenObservedField; field_margin_m is a target
> TSDF scalar value, not a Euclidean clearance or a query precision.
>
> Only a confirmed original intersection invokes SLSQP. A clear baseline is
> already the zero-angle solution; a baseline with only unresolved queries
> retains its original uncertainty. An optimizer failure still publishes
> its returned proposal and the original entity queries at that proposal.

显式输出分支：

```python
dict(status=after_status, proposal_status=proposal_status, T_world_tool_original=original, T_world_tool_proposed=proposed, q=q, angle_rad=angle, angle_deg=float(np.rad2deg(angle)), basis_tool=basis.copy(), axis_tool=axis, rotation_vector_tool=basis @ q, max_tilt_rad=float(max_tilt_rad), field_margin_m=float(field_margin_m), before_status=before_status, after_status=after_status, before_checks=before_checks, after_checks=after_checks, guide_before=guide_before, guide_after=guide_report(final), optimizer=optimizer, history=history, timings=dict(before_checks_seconds=before_seconds, optimization_seconds=optimization_seconds, after_checks_seconds=after_seconds, guide_evaluation_seconds=field_seconds, guide_evaluations=len(history['evaluations']), total_seconds=perf_counter() - started, scope='refine_pose only; geometry and background field are already constructed'), quality_status='not_evaluated_by_geometric_refinement', physical_status='missing_not_executed', scope='fixed TCP origin, two-axis local swing, observed background and official robot entities')
```

读取的对象属性：

```python
np.asarray(T_world_tool).copy
np.asarray
np.cross
np.cumsum
np.r_
np.concatenate
np.zeros
np.asarray(solved.x).copy
solved.x
original.copy
solved.success
solved.status
solved.message
solved.nit
solved.nfev
solved.njev
solved.fun
final['constraints'][:-1].min
np.linalg.norm
np.linalg
np.rad2deg
basis.copy
```

写入目标：

```python
started
original
rotation0
origin
basis
axis
sizes
part
ends
starts
points
before_started
before_checks
before_seconds
before_status
history
cached_q
cached
field_seconds
zero
initial
guide_before
optimization_seconds
after_seconds
optimization_started
solved
q
final
proposed
proposed[:3, :3]
optimizer
after_started
after_checks
proposal_status
after_status
angle
```

直接调用符号：

```python
perf_counter
np.asarray(T_world_tool).copy
np.asarray
np.cross
len
np.cumsum
np.concatenate
_check_parts
_relation
dict
np.zeros
evaluate
guide_report
minimize
float
np.asarray(solved.x).copy
original.copy
bool
int
str
max
final['constraints'][:-1].min
np.linalg.norm
np.rad2deg
basis.copy
```

## `refine_pose.guide_report`

[定义位置](../../../manipulation/grasp_pose_refinement.py#L77)。

输入签名：

```python
def guide_report(sample):
```

实现说明：

> Serialize observed values with explicit indices, never unknown fills.

显式输出分支：

```python
dict(point_count=len(points), known_count=int(known.sum()), unknown_count=int((~known).sum()), parts=rows, supported_indices=supported, unsupported_indices=np.flatnonzero(~known), supported_values_m=sample['values'][known], supported_gradients_world=sample['gradients'][known], supported_arithmetic_errors_m=sample['errors'][known], worst_sample=worst, inactive_constraint_encoding='unsupported row: constant 1, zero Jacobian; no field observation', field_margin_m=float(field_margin_m), field_voxel_size_m=float(field.voxel_size))
```

读取的对象属性：

```python
np.flatnonzero
known[first:last].sum
rows.append
np.argmin
np.searchsorted
known.sum
(~known).sum
field.voxel_size
```

写入目标：

```python
known
supported
rows
part
first
last
count
worst
index
part_index
```

直接调用符号：

```python
np.flatnonzero
zip
int
known[first:last].sum
rows.append
dict
len
np.argmin
np.searchsorted
float
known.sum
(~known).sum
```

## `refine_pose.evaluate`

[定义位置](../../../manipulation/grasp_pose_refinement.py#L111)。

输入签名：

```python
def evaluate(q):
```

实现说明：

> Share this exact rotation's field values and analytic derivatives.

显式输出分支：

```python
cached
cached
```

读取的对象属性：

```python
np.array_equal
np.asarray
Rotation.from_rotvec(vector).as_matrix
Rotation.from_rotvec
rotation.T
field.sample
pin.Jexp3
np.cross
np.ones
np.zeros
field.voxel_size
q.copy
history['evaluations'].append
np.linalg.norm
np.linalg
```

写入目标：

```python
stamp
q
vector
increment
rotation
relative_world
world
values
known
gradients
errors
angular_axes
dphi_dq
constraints
jacobian
constraints[:-1][known]
jacobian[:-1][known]
constraints[-1]
jacobian[-1]
cached_q
cached
seconds
field_seconds
report
```

直接调用符号：

```python
np.array_equal
perf_counter
np.asarray
Rotation.from_rotvec(vector).as_matrix
Rotation.from_rotvec
field.sample
pin.Jexp3
np.cross
np.ones
len
np.zeros
float
q.copy
dict
guide_report
history['evaluations'].append
np.linalg.norm
```

## `refine_pose.iteration`

[定义位置](../../../manipulation/grasp_pose_refinement.py#L151)。

输入签名：

```python
def iteration(q):
```

实现说明：

> Retain numerical iterations independently of entity classification.

显式输出分支：

无显式条目。

读取的对象属性：

```python
history['iterations'].append
np.asarray(q).copy
np.asarray
np.linalg.norm
np.linalg
sample['constraints'].min
```

写入目标：

```python
sample
```

直接调用符号：

```python
evaluate
history['iterations'].append
dict
len
np.asarray(q).copy
np.asarray
float
np.linalg.norm
sample['constraints'].min
```
