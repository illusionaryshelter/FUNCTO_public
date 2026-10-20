# `manipulation/planning.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_fixed_spline_grid`

[定义位置](../../../manipulation/planning.py#L53)。

输入签名：

```python
def _fixed_spline_grid(*, intervals, native_dt, interpolation_steps, interpolation_dt):
```

输出分支的源码表达式：

```python
{'native_dt': float(raw_dt), 'knot_dt': float(knot_dt), 'interpolation_dt': float(output_dt), 'subdivisions': subdivisions, 'valid_count': count, 'duration': duration}
```

直接调用：

```python
type
ValueError
np.errstate
np.float32
np.isfinite([raw_dt, output_dt, knot_dt]).all
np.isfinite
min
int
round
float
```

## `PlanningBudget`

[定义位置](../../../manipulation/planning.py#L88)。

```python
class PlanningBudget:
    ik_seeds: int = 8
    trajectory_seeds: int = 2
    knots: int = 8
    interpolation_steps: int = 4
    ik_iterations: int = 40
    trajectory_iterations: int = 100
    interpolation_dt: float = 0.025
    maximum_trajectory_dt: float = 0.2
    interpolation_buffer_size: int = 512
    constraint_weight: float = 100000.0
    max_task_anchors: int = 256
    max_robot_spheres: int = 256
    max_robot_dof: int = 16
    max_robot_links: int = 64
    max_scene_gpu_workspace_bytes: int = 1024 ** 3
    max_host_rss_bytes: int = DEFAULT_MAX_HOST_RSS_BYTES
    max_device_used_bytes: int = DEFAULT_MAX_DEVICE_USED_BYTES
    max_validation_points: int = 4096
    validation_joint_step: float = 0.025
    validation_spatial_step_m: float = 0.001
    search_seconds: float = 3.0
    maximum_trajectory_solves: int = 2
    planning_seconds: float = 120.0
    random_seed: int = 123
```

该类型的直接方法：`__post_init__`。

## `PlanningBudget.__post_init__`

[定义位置](../../../manipulation/planning.py#L137)。

输入签名：

```python
def PlanningBudget.__post_init__(self):
```

直接读取的对象属性：`self.trajectory_seeds`、`self.ik_seeds`、`self.knots`、`self.interpolation_steps`、`self.ik_iterations`、`self.trajectory_iterations`、`self.max_robot_spheres`、`self.max_robot_dof`、`self.max_robot_links`、`self.max_validation_points`、`self.maximum_trajectory_solves`、`self.max_task_anchors`、`self.max_scene_gpu_workspace_bytes`、`self.max_host_rss_bytes`、`self.max_device_used_bytes`、`self.search_seconds`、`self.maximum_trajectory_dt`、`self.interpolation_dt`、`self.interpolation_buffer_size`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
ValueError
type
np.isfinite
getattr
_fixed_spline_grid
```

## `JointTrajectory`

[定义位置](../../../manipulation/planning.py#L190)。

```python
class JointTrajectory:
    joint_names: tuple[str, ...]
    times: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    progress_u: np.ndarray
    progress_s: np.ndarray
    events: tuple[dict, ...]
    conditions: tuple[dict, ...]
    residuals: dict[str, float]
    finger_trajectory: object | None = None
    satisfied_dependencies: tuple[str, ...] = ()
    _curve: CubicHermiteSpline = field(init=False, repr=False)
```

该类型的直接方法：`__post_init__`、`duration`、`next_verification`、`executable_until`、`sample`。

## `JointTrajectory.__post_init__`

[定义位置](../../../manipulation/planning.py#L214)。

输入签名：

```python
def JointTrajectory.__post_init__(self):
```

直接读取的对象属性：`self.times`、`self.position`、`self.velocity`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._curve`。

直接调用：

```python
CubicHermiteSpline
```

## `JointTrajectory.duration`

[定义位置](../../../manipulation/planning.py#L218)。

输入签名：

```python
@property
def JointTrajectory.duration(self):
```

直接读取的对象属性：`self.times`。

输出分支的源码表达式：

```python
float(self.times[-1])
```

直接调用：

```python
float
```

## `JointTrajectory.next_verification`

[定义位置](../../../manipulation/planning.py#L222)。

输入签名：

```python
@property
def JointTrajectory.next_verification(self):
```

直接读取的对象属性：`self.events`、`self.satisfied_dependencies`、`self.progress_s`。

输出分支的源码表达式：

```python
deepcopy(min(barriers, key=lambda event: event['time'])) if barriers else None
```

直接调用：

```python
deepcopy
min
```

## `JointTrajectory.executable_until`

[定义位置](../../../manipulation/planning.py#L230)。

输入签名：

```python
@property
def JointTrajectory.executable_until(self):
```

直接读取的对象属性：`self.next_verification`、`self.duration`。

输出分支的源码表达式：

```python
self.duration if barrier is None else float(barrier['time'])
```

直接调用：

```python
float
```

## `JointTrajectory.sample`

[定义位置](../../../manipulation/planning.py#L235)。

输入签名：

```python
def JointTrajectory.sample(self, times):
```

直接读取的对象属性：`self.duration`、`self._curve`、`self.progress_u`、`self.progress_s`。

输出分支的源码表达式：

```python
(self._curve(query), self._curve(query, 1), self._curve(query, 2), np.interp(query / self.duration, self.progress_u, self.progress_s))
```

直接调用：

```python
np.asarray
np.isfinite(query).all
np.isfinite
np.any
ValueError
self._curve
np.interp
```

## `cubic_derivative_report`

[定义位置](../../../manipulation/planning.py#L254)。

输入签名：

```python
def cubic_derivative_report(trajectory, acceleration_bounds, *, joint_names, until=None):
```

直接读取的对象属性：`trajectory.joint_names`、`trajectory.times`、`trajectory.position`、`trajectory.velocity`、`trajectory._curve`。

输出分支的源码表达式：

```python
{'scope': 'binary64 polynomial acceleration ranges and knot continuity; no actual-motion or collision certification', 'checked_interval_s': [0.0, stop], 'full_curve_duration_s': float(times[-1]), 'joint_names': list(curve_names), 'checked_piece_count': len(indices), 'acceleration_bounds': bounds.tolist(), 'acceleration': {'minimum': minimum.tolist(), 'maximum': maximum.tolist(), 'minimum_time_s': endpoint_times[low_indices].tolist(), 'maximum_time_s': endpoint_times[high_indices].tolist(), 'maximum_absolute': np.maximum(np.abs(minimum), np.abs(maximum)).tolist(), 'lower_excess': lower_excess.tolist(), 'upper_excess': upper_excess.tolist(), 'maximum_bound_excess': maximum_excess, 'bounds_satisfied': maximum_excess == 0.0, 'maximum_violation': witness}, 'continuity': continuity}
```

直接写入的状态或数组位置：`continuity[f'{label}_maximum_absolute_jump']`。

直接调用：

```python
tuple
np.asarray
len
set
np.isfinite(bounds).all
np.isfinite
np.any
ValueError
names.index
np.isfinite(times).all
np.diff
np.isfinite(position).all
np.isfinite(velocity).all
CubicHermiteSpline
np.array_equal
float
np.flatnonzero
np.array
np.minimum
np.stack
acceleration.reshape
np.stack((times[indices], ends), axis=1).reshape
np.isfinite(values).all
np.arange
values.argmin
values.argmax
np.maximum
excess.max
np.unravel_index
excess.argmax
int
jumps.items
np.isfinite(selected).all
(np.abs(selected).max(axis=0) if len(selected) else np.zeros(len(names))).tolist
np.abs(selected).max
np.abs
np.zeros
list
bounds.tolist
minimum.tolist
maximum.tolist
endpoint_times[low_indices].tolist
endpoint_times[high_indices].tolist
np.maximum(np.abs(minimum), np.abs(maximum)).tolist
lower_excess.tolist
upper_excess.tolist
```

## `PlanningResult`

[定义位置](../../../manipulation/planning.py#L357)。

```python
class PlanningResult:
    success: bool
    trajectory: JointTrajectory | None
    diagnostics: dict
    execution_binding: dict | None = None
    request_ledger: dict | None = None
```

## `_anchor_capacity`

[定义位置](../../../manipulation/planning.py#L370)。

输入签名：

```python
def _anchor_capacity(progress_map, anchors, budget):
```

直接读取的对象属性：`budget.max_task_anchors`。

输出分支的源码表达式：

```python
budget.max_task_anchors
```

直接调用：

```python
len
progress_map.anchor_parameters
MotionPlanningError
```

## `_PlanningWorkspace`

[定义位置](../../../manipulation/planning.py#L383)。

```python
class _PlanningWorkspace:
    cpu_shared_bytes: int
    cpu_live_bytes: int
    gpu_eager_bytes: int
    gpu_graph_bytes: int
    anchor_fk_bytes: int
    trajectory_graph: bool
```

该类型的直接方法：`cpu_bytes`、`gpu_bytes`、`total_bytes`。

## `_PlanningWorkspace.cpu_bytes`

[定义位置](../../../manipulation/planning.py#L394)。

输入签名：

```python
@property
def _PlanningWorkspace.cpu_bytes(self):
```

直接读取的对象属性：`self.cpu_shared_bytes`、`self.cpu_live_bytes`。

输出分支的源码表达式：

```python
self.cpu_shared_bytes + self.cpu_live_bytes
```

## `_PlanningWorkspace.gpu_bytes`

[定义位置](../../../manipulation/planning.py#L398)。

输入签名：

```python
@property
def _PlanningWorkspace.gpu_bytes(self):
```

直接读取的对象属性：`self.gpu_eager_bytes`、`self.trajectory_graph`、`self.gpu_graph_bytes`。

输出分支的源码表达式：

```python
self.gpu_eager_bytes + (self.gpu_graph_bytes if self.trajectory_graph else 0)
```

## `_PlanningWorkspace.total_bytes`

[定义位置](../../../manipulation/planning.py#L402)。

输入签名：

```python
@property
def _PlanningWorkspace.total_bytes(self):
```

直接读取的对象属性：`self.cpu_bytes`、`self.gpu_bytes`。

输出分支的源码表达式：

```python
self.cpu_bytes + self.gpu_bytes
```

## `_planning_workspace_bytes`

[定义位置](../../../manipulation/planning.py#L408)。

输入签名：

```python
def _planning_workspace_bytes(scene, robot_metadata, budget, anchor_capacity):
```

直接读取的对象属性：`budget.knots`、`budget.interpolation_steps`、`budget.trajectory_seeds`、`budget.interpolation_buffer_size`、`budget.max_validation_points`、`scene.required_progress_anchors`、`scene.estimated_workspace_allocation`、`budget.max_scene_gpu_workspace_bytes`。

输出分支的源码表达式：

```python
_PlanningWorkspace(int(cpu_shared_bytes), int(cpu_live_bytes), int(gpu_eager_bytes), int(2 * graph_live), int(anchor_fk_bytes), bool(trajectory_graph))
```

直接调用：

```python
min
scene.estimated_workspace_allocation
sum
max
ResourceLimitError
zip
len
_PlanningWorkspace
int
bool
```

## `_require_planning_host_memory`

[定义位置](../../../manipulation/planning.py#L501)。

输入签名：

```python
def _require_planning_host_memory(cpu_live_bytes, *, encoded_record_bytes_upper, max_host_rss_bytes):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
require_host_memory
ObservedTsdfGeometry.pending_storage_total_nbytes
```

## `_require_planning_memory`

[定义位置](../../../manipulation/planning.py#L519)。

输入签名：

```python
def _require_planning_memory(torch, device, workspace, *, max_device_used_bytes):
```

输出分支的源码表达式：

```python
eager
workspace
```

直接调用：

```python
refusal
replace
```

## `_require_planning_memory.refusal`

[定义位置](../../../manipulation/planning.py#L534)。

输入签名：

```python
def _require_planning_memory.refusal(candidate):
```

输出分支的源码表达式：

```python
error
None
```

直接调用：

```python
require_device_memory
```

## `_bounded_robot_configuration`

[定义位置](../../../manipulation/planning.py#L557)。

输入签名：

```python
def _bounded_robot_configuration(robot, problem, scene, budget):
```

直接读取的对象属性：`budget.max_robot_links`、`budget.max_robot_dof`、`budget.max_robot_spheres`、`scene.robot_sphere_materials`、`problem.tool_link`、`scene.required_robot_links`、`problem.joint_names`。

输出分支的源码表达式：

```python
(configuration, metadata)
```

直接写入的状态或数组位置：`driven[joint.child]`、`parents[name]`、`parents[link['child_link_name']]`、`driven[name]`、`configuration['robot_cfg']['kinematics']['tool_frames']`、`configuration['robot_cfg']['kinematics']['collision_spheres']`。

直接调用：

```python
isinstance
load_yaml
bounded_file
join_path
get_robot_configs_path
kinematics.get
ValueError
len
set
any
extra_links.values
cspace.values
np.asarray
sum
spheres.values
type
extra_spheres.values
spheres.items
counts.update
counts.values
get_assets_path
yourdfpy.URDF.load
parsed.joint_map.values
extra_links.items
link.get
chain.add
selected.add
deepcopy
sorted
```

## `_bounded_robot_configuration.bounded_file`

[定义位置](../../../manipulation/planning.py#L571)。

输入签名：

```python
def _bounded_robot_configuration.bounded_file(path):
```

输出分支的源码表达式：

```python
str(path)
```

直接调用：

```python
Path(path).stat
Path
ValueError
str
```

## `_ProgressMap`

[定义位置](../../../manipulation/planning.py#L687)。

```python
class _ProgressMap:
    u: np.ndarray
    s: np.ndarray
    event_clock: dict = field(default_factory=dict, compare=False)
```

该类型的直接方法：`__post_init__`、`anchor_parameters`。

## `_ProgressMap.__post_init__`

[定义位置](../../../manipulation/planning.py#L696)。

输入签名：

```python
def _ProgressMap.__post_init__(self):
```

直接读取的对象属性：`self.u`、`self.s`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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

[定义位置](../../../manipulation/planning.py#L704)。

输入签名：

```python
def _ProgressMap.anchor_parameters(self, anchors):
```

直接读取的对象属性：`self.s`、`self.u`。

输出分支的源码表达式：

```python
pairs
```

直接调用：

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
any
np.unique
pairs[:, column].astype
```

## `_phase_aligned_parameters`

[定义位置](../../../manipulation/planning.py#L730)。

输入签名：

```python
def _phase_aligned_parameters(source, target, parameters):
```

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
np.asarray(result)
```

直接调用：

```python
np.asarray
np.interp
zip
source.anchor_parameters
target.anchor_parameters
len
MotionPlanningError
result.append
```

## `_candidate_sample_times`

[定义位置](../../../manipulation/planning.py#L754)。

输入签名：

```python
def _candidate_sample_times(trajectory, budget):
```

直接读取的对象属性：`trajectory.times`、`budget.interpolation_dt`、`budget.max_validation_points`、`budget.interpolation_buffer_size`、`trajectory.duration`。

输出分支的源码表达式：

```python
np.r_[np.concatenate([np.linspace(left, right, count + 1)[:-1] for (left, right, count) in zip(trajectory.times[:-1], trajectory.times[1:], counts)]), trajectory.duration]
```

直接调用：

```python
np.maximum
np.ceil(np.diff(trajectory.times) / float(np.float32(budget.interpolation_dt))).astype
np.ceil
np.diff
float
np.float32
int
counts.sum
min
MotionPlanningError
np.concatenate
np.linspace
zip
```

## `_candidate_confirmation_times`

[定义位置](../../../manipulation/planning.py#L765)。

输入签名：

```python
def _candidate_confirmation_times(trajectory, budget, *, regular_count, regular_dt, dense_count, dense_dt):
```

直接读取的对象属性：`trajectory.duration`、`budget.max_validation_points`、`budget.interpolation_buffer_size`。

输出分支的源码表达式：

```python
times
```

直接调用：

```python
_candidate_sample_times
type
ValueError
np.isfinite
MotionPlanningError
queries.append
np.arange
np.unique
np.concatenate
np.any
np.diff
len
min
```

## `_reference_clock_issues`

[定义位置](../../../manipulation/planning.py#L791)。

输入签名：

```python
def _reference_clock_issues(problem, trajectory, timestep):
```

直接读取的对象属性：`trajectory.progress_u`、`trajectory.progress_s`、`problem.events`、`trajectory.duration`。

输出分支的源码表达式：

```python
issues
```

直接调用：

```python
_ProgressMap
MotionPlanningError
float
mapping.anchor_parameters
_activation
issues.append
```

## `_initial_timing`

[定义位置](../../../manipulation/planning.py#L821)。

输入签名：

```python
def _initial_timing(problem, budget):
```

直接读取的对象属性：`problem.source_time_map`、`budget.interpolation_dt`、`budget.knots`、`budget.maximum_trajectory_dt`、`budget.interpolation_steps`。

输出分支的源码表达式：

```python
(selected, _ProgressMap((seconds - seconds[0]) / source_duration, progress.copy()))
```

直接调用：

```python
np.asarray
len
np.isfinite(seconds).all
np.isfinite
np.any
np.diff
ValueError
float
np.float32
_fixed_spline_grid
min
np.nextafter
range
(budget.interpolation_steps - 1).bit_length
max
int
np.ceil
(needed - 1).bit_length
replace
_ProgressMap
progress.copy
```

## `_bind_event_clock`

[定义位置](../../../manipulation/planning.py#L869)。

输入签名：

```python
def _bind_event_clock(problem, progress_map, duration, timestep=None):
```

直接读取的对象属性：`problem.events`、`record.update`。

输出分支的源码表达式：

```python
_ProgressMap(mapping.u, mapping.s, source)
```

直接写入的状态或数组位置：`source[identity]`、`source[key]['scheduled_superseded_by']`。

直接调用：

```python
set
event_occurrence_key
progress_map.anchor_parameters
float
ValueError
_activation
tick_entries.append
asdict
source[identity].update
by_step.setdefault(entry[1], []).append
by_step.setdefault
getattr
by_step.values
sorted
any
superseded.add
command_keys.append
tick_progress.add
dict
np.asarray
np.any
np.diff
np.union1d
_ProgressMap
np.interp
mapping.anchor_parameters
record.update
```

## `_validate_bound_event_times`

[定义位置](../../../manipulation/planning.py#L947)。

输入签名：

```python
def _validate_bound_event_times(problem, progress_map, duration, origin_s=0.0, history=(), goal_identities=None, timing_events=()):
```

直接读取的对象属性：`problem.events`、`record.get`。

输出分支的源码表达式：

```python
checks
```

直接写入的状态或数组位置：`latest[key[:2]]`、`planned[key, constraint.check_target, constraint.goal_id, constraint.condition_id]`。

直接调用：

```python
list
range
len
getattr
row.get
event_occurrence_key
PlanningEvent
all_events.update
max
latest.get
current.items
record.get
candidate_fact
enumerate
abs
float
progress_map.anchor_parameters
condition.get
past_pose_timing
TimeBindingError
goal_identities[collection].index
all_events.items
checks.append
tuple
min
planned.get
resolve_event_window
```

## `_validate_bound_event_times.past_pose_timing`

[定义位置](../../../manipulation/planning.py#L967)。

输入签名：

```python
def _validate_bound_event_times.past_pose_timing(constraint):
```

输出分支的源码表达式：

```python
any((identity and identity[0] in ('pose_events', 'supports', 'coordinate_events') and (identity[1] not in goal_identities[identity[0]]) for identity in identities))
```

直接调用：

```python
(constraint.origin_event or {}).get
any
```

## `_validate_bound_event_times.candidate_fact`

[定义位置](../../../manipulation/planning.py#L981)。

输入签名：

```python
def _validate_bound_event_times.candidate_fact(key, target, value, **identity):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
predicted.append
```

## `ContinuationBindingError`

[定义位置](../../../manipulation/planning.py#L1065)。

```python
class ContinuationBindingError(ValueError):
    ...
    ...
```

## `_obligation_contract`

[定义位置](../../../manipulation/planning.py#L1075)。

输入签名：

```python
def _obligation_contract(problem):
```

输出分支的源码表达式：

```python
contract
```

直接写入的状态或数组位置：`contract[collection]`、`contract['conditions']`、`contract['source_time_map']`、`contract['source_phases']`。

直接调用：

```python
columns.items
type
plain
getattr
item.actual_relation.items
key.startswith
phase.metadata.get
```

## `_restore_original_direction_origins`

[定义位置](../../../manipulation/planning.py#L1113)。

输入签名：

```python
def _restore_original_direction_origins(problem, contract):
```

直接读取的对象属性：`problem.mechanisms`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`mechanism.directions`。

直接调用：

```python
len
getattr
zip
plain
float
semantics
replace
deepcopy
directions.append
tuple
```

## `PlanningWorkExhausted`

[定义位置](../../../manipulation/planning.py#L1146)。

```python
class PlanningWorkExhausted(ValueError):
    ...
    ...
```

## `_planning_work_budget`

[定义位置](../../../manipulation/planning.py#L1155)。

输入签名：

```python
def _planning_work_budget(values):
```

输出分支的源码表达式：

```python
{name: value for (name, value) in values.items() if name not in _ALLOCATION_BUDGET_FIELDS}
```

直接调用：

```python
values.items
```

## `_request_work_ledger`

[定义位置](../../../manipulation/planning.py#L1161)。

输入签名：

```python
def _request_work_ledger(problem, budget, previous=None, continuation=None):
```

直接读取的对象属性：`problem.events`、`continuation.get`。

输出分支的源码表达式：

```python
ledger
```

直接写入的状态或数组位置：`ledger['planning_budget']`、`ledger['requires_continuation']`。

直接调用：

```python
uuid4
asdict
dict.fromkeys
plain
_obligation_contract
deepcopy
ledger.get
isinstance
type
_planning_work_budget
ContinuationBindingError
PlanningBudget
consumed.get
np.isfinite
ledger['binding_records'].append
sorted
tuple
any
len
continuation.get
ledger[name].append
ledger['event_records'].append
ledger['effects'].update
```

## `_check_work_budget`

[定义位置](../../../manipulation/planning.py#L1239)。

输入签名：

```python
def _check_work_budget(budget, consumed, *, started, initial_seconds, stage=None):
```

直接读取的对象属性：`budget.maximum_trajectory_solves`、`budget.planning_seconds`、`budget.max_validation_points`、`budget.ik_seeds`、`budget.trajectory_seeds`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`consumed['planning_seconds']`。

直接调用：

```python
time.monotonic
PlanningWorkExhausted
stage_caps.items
```

## `_prepare_continuation`

[定义位置](../../../manipulation/planning.py#L1257)。

输入签名：

```python
def _prepare_continuation(problem, scene, budget, continuation, *, rebind=None):
```

直接读取的对象属性：`continuation.get`、`feedback.get`、`problem.joint_names`、`problem.start_q`、`problem.start_dq`、`scene.gripper`、`problem.end_progress`、`problem.events`、`problem.mechanisms`、`problem.source_progress`、`problem.progress_to_source_time`、`problem.source_timestamps`、`scene.required_progress_anchors`。

输出分支的源码表达式：

```python
(remaining, {'original_duration': float(duration), 'elapsed': float(elapsed), 'actual_elapsed': float(actual_elapsed), 'elapsed_device_steps': confirmed_steps, 'device_timestep_s': float(timestep), 'u': original_map.u.copy(), 's': original_map.s.copy(), 'events': {identity: deepcopy(old_occurrences[identity]) for identity in pending_ids}, 'request_budget': asdict(request_budget), 'consumed_planning': consumed, 'physics_budget': physics_cap, 'consumed_physics_steps': physics_consumed, 'remaining_physics_steps': physics_remaining, 'request_events': request_events, 'consumed_events': sorted(consumed_events), 'goal_identities': goal_identities, 'retry_goal_ids': sorted(retry_goals), 'direction_identities': direction_identities, 'retry_direction_ids': sorted(retry_directions), 'completed_direction_ids': sorted(completed_directions), 'rebind': deepcopy(rebind), 'scene_anchors': np.asarray(scene.required_progress_anchors, float).copy() if rebind is not None else None, 'control_timestep_s': timestep if rebind is not None else None, 'history': deepcopy(continuation.get('history', ())), 'timing_events': deepcopy(continuation.get('request_ledger', original['request_ledger'])['timing_events']), 'device_policy': deepcopy(continuation.get('device_policy')), 'active_finger_interval': deepcopy(continuation.get('active_finger_interval')), 'completed_goal_ids': [], 'effects': deepcopy(continuation.get('effects', {})), 'previous_request_time_origin_s': float(original.get('request_time_origin_s', 0.0)), 'request_time_origin_s': float(original.get('request_time_origin_s', 0.0)) + float(elapsed), 'request_time_origin_step': int(original.get('request_time_origin_step', 0)) + local_steps})
```

直接写入的状态或数组位置：`selected_events[identity[:2]]`、`selected_events[event.skill_id, event.event_id]`、`selected_events[identity]`、`rebound.start_progress`、`rebound.events`、`mechanism.start`、`rebound.source_timestamps`、`rebound.source_progress`、`hand.commands`。

直接调用：

```python
isinstance
rebind.get
ContinuationBindingError
continuation.get
original.get
list
len
_restore_original_direction_origins
_obligation_contract
PlanningBudget
_planning_work_budget
asdict
deepcopy
consumed.get
np.isfinite
original['consumed_planning'].get
type
max
feedback.get
set
names.index
np.asarray
np.array_equal
curve.get
float
abs
np.isfinite([duration, elapsed]).all
_ProgressMap
np.interp
current_events.items
plain
current.pop
source.pop
event_occurrence_key
tuple
applied.add
current_events.get
pending_ids.discard
row.get
range
getattr
enumerate
row['source'].strip
bool
np.isfinite(interval).all
completed_directions.update
replace
any
occurrences.extend
selected_events.values
min
requirement.model.diagnostics.get
mechanism.model.diagnostics.get
setattr
problem.progress_to_source_time
rebound.__post_init__
policies.get
commands.append
fields
original_map.u.copy
original_map.s.copy
sorted
np.asarray(scene.required_progress_anchors, float).copy
int
```

## `_continuation_timing`

[定义位置](../../../manipulation/planning.py#L1611)。

输入签名：

```python
def _continuation_timing(problem, budget, continuation):
```

直接读取的对象属性：`continuation.get`、`problem.anchors`、`problem.start_progress`、`problem.stop_progress`、`record.get`、`problem.end_progress`、`record.update`、`problem.events`。

输出分支的源码表达式：

```python
(selected, mapping)
(selected, _ProgressMap(seconds / grid['duration'], progress, clock))
```

直接写入的状态或数组位置：`record['request_intent_time_s']`、`record['request_time_origin_step']`、`record['request_activation_step']`、`record['request_activation_time_s']`、`record[name]`、`record['time']`、`record['intent_time_s']`、`clock[identity]`、`record['event']`。

直接调用：

```python
continuation.get
_initial_timing
np.unique
_progress_for_cubic_boundaries
_fixed_spline_grid
_bind_event_clock
mapping.event_clock.values
record.get
_validate_bound_event_times
SimpleNamespace
np.array
float
ContinuationBindingError
continuation['events'].items
deepcopy
event.get
record.update
max
asdict
next
event_occurrence_key
_ProgressMap
```

## `_progress_for_cubic_boundaries`

[定义位置](../../../manipulation/planning.py#L1686)。

输入签名：

```python
def _progress_for_cubic_boundaries(progress_map, knots, *, stop_progress, anchors):
```

输出分支的源码表达式：

```python
(_ProgressMap(warp.u.copy(), progress), warp)
```

直接写入的状态或数组位置：`increments[order[:remaining - int(np.floor(shares).sum())]]`、`segment[-1]`、`parameter[interval * subdivisions:(interval + 1) * subdivisions + 1]`、`parameter[0]`、`parameter[-1]`、`parameter[np.searchsorted(new_u, inverse_pairs[:, 0])]`。

直接调用：

```python
type
ValueError
np.unique
np.asarray
progress_map.anchor_parameters
len
np.diff
np.floor(shares).astype
np.floor
np.argsort
int
np.floor(shares).sum
np.array
enumerate
zip
np.arange
np.linspace
free.extend
np.ones
BSpline
float
np.abs(np.diff(coefficients, n=2)).max
np.abs
max
np.ceil
np.sqrt
spline
range
np.all
_ProgressMap
warp.anchor_parameters
np.isin
warp.u.astype
inverse_pairs[:, 0].astype
np.interp
np.searchsorted
np.abs(np.interp(warp.u, new_u, parameter) - warp.s).max
np.abs(np.interp(new_u, warp.u, warp.s) - parameter).max
dict
supplied.update
supplied.get
warp.u.copy
```

## `_torch_interpolate`

[定义位置](../../../manipulation/planning.py#L1791)。

输入签名：

```python
def _torch_interpolate(keys, values, query):
```

输出分支的源码表达式：

```python
values[right - 1] + alpha * (values[right] - values[right - 1])
```

直接调用：

```python
torch.searchsorted(keys, query.contiguous(), right=True).clamp
torch.searchsorted
query.contiguous
len
((query - keys[right - 1]) / (keys[right] - keys[right - 1])).clamp
```

## `_hermite_at_parameters`

[定义位置](../../../manipulation/planning.py#L1799)。

输入签名：

```python
def _hermite_at_parameters(position, velocity, dt, counts, parameters):
```

输出分支的源码表达式：

```python
(q.to(position.dtype), dq.to(velocity.dtype))
```

直接调用：

```python
parameters.to
torch.minimum(coordinates.floor().long(), counts[:, None] - 2).clamp_min
torch.minimum
coordinates.floor().long
coordinates.floor
(coordinates - left).unsqueeze
left.unsqueeze(-1).expand
left.unsqueeze
position.gather(1, gather).to
position.gather
position.gather(1, gather + 1).to
velocity.gather(1, gather).to
velocity.gather
velocity.gather(1, gather + 1).to
dt.reshape(batch, 1, 1).to
dt.reshape
q.to
dq.to
```

## `_projection_tolerance`

[定义位置](../../../manipulation/planning.py#L1826)。

输入签名：

```python
def _projection_tolerance(problem):
```

直接读取的对象属性：`problem.numerical`、`problem.mechanisms`。

输出分支的源码表达式：

```python
min(bounds) * 0.1
```

直接调用：

```python
bounds.extend
min
```

## `_ompl_terms_accepted`

[定义位置](../../../manipulation/planning.py#L1837)。

输入签名：

```python
def _ompl_terms_accepted(terms):
```

输出分支的源码表达式：

```python
bool(torch.stack(flags).all().item()) if flags else True
```

直接调用：

```python
torch.isfinite(term).all
torch.isfinite
(term <= 0).all
terms.values
bool
torch.stack(flags).all().item
torch.stack(flags).all
torch.stack
```

## `_ompl_stage`

[定义位置](../../../manipulation/planning.py#L1853)。

输入签名：

```python
def _ompl_stage(start, goal, lower, upper, valid, equality, *, tolerance, step, seconds, sample_limit, query_parameters=()):
```

输出分支的源码表达式：

```python
(None, {**report, 'failure': 'equality_codimension_not_below_state_dimension'})
(None, {**report, 'failure': 'singular_equality_jacobian'})
(None, {**report, 'failure': failure, 'boundary': label})
(direct, {**report, 'samples': len(direct), 'strategy': 'native_direct_connection'})
(None, {**report, 'failure': 'whole_search_time_budget_exhausted'})
(None, {**report, 'failure': 'no_exact_path_within_search_budget'})
(None, {**report, 'failure': 'solution_edge_export_rejected'})
(None, {**report, 'failure': 'solution_samples_exceed_budget_or_change_endpoints'})
(samples, {**report, 'samples': len(samples), 'strategy': 'rrt_connect'})
```

直接写入的状态或数组位置：`a[index]`、`b[index]`、`report['initialization_seconds']`、`report['motion_checks']`、`report['direct_connection_accepted']`。

直接调用：

```python
len
np.unique
np.asarray
np.isfinite(query_parameters).all
np.isfinite
np.any
ValueError
time.monotonic
equality
ProcessConstraint
np.isfinite(jacobian).all
np.linalg.matrix_rank
ob.RealVectorStateSpace
ob.RealVectorBounds
enumerate
zip
bounds.setLow
float
bounds.setHigh
ambient.setBounds
ob.ProjectedStateSpace
space.setDelta
ob.ConstrainedSpaceInformation
ob.SpaceInformation
information.setStateValidityCheckingResolution
min
np.linalg.norm
information.setStateValidityChecker
information.setup
information.getMotionValidator
information.setMotionValidator
ForwardValidator
space.allocState
a.copy
start.tolist
b.copy
goal.tolist
range
max
valid_array
checked_edge
ob.ProblemDefinition
definition.setStartAndGoalStates
og.RRTConnect
search.setIntermediateStates
search.setProblemDefinition
search.setup
search.solve
definition.hasExactSolution
definition.getSolutionPath
path.getState
path.getStateCount
any
np.concatenate
np.diff
np.array_equal
```

## `_ompl_stage.ProcessConstraint`

[定义位置](../../../manipulation/planning.py#L1878)。

```python
class _ompl_stage.ProcessConstraint(ob.Constraint):
    ...
    ...
```

该类型的直接方法：`__init__`、`function`、`jacobian`。

## `_ompl_stage.ProcessConstraint.__init__`

[定义位置](../../../manipulation/planning.py#L1879)。

输入签名：

```python
def _ompl_stage.ProcessConstraint.__init__(self):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
super().__init__
super
```

## `_ompl_stage.ProcessConstraint.function`

[定义位置](../../../manipulation/planning.py#L1882)。

输入签名：

```python
def _ompl_stage.ProcessConstraint.function(self, values, output):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`output[:]`。

直接调用：

```python
equality
np.asarray
```

## `_ompl_stage.ProcessConstraint.jacobian`

[定义位置](../../../manipulation/planning.py#L1885)。

输入签名：

```python
def _ompl_stage.ProcessConstraint.jacobian(self, values, output):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`output[:]`。

直接调用：

```python
equality
np.asarray
```

## `_ompl_stage.vector`

[定义位置](../../../manipulation/planning.py#L1910)。

输入签名：

```python
def _ompl_stage.vector(state):
```

输出分支的源码表达式：

```python
np.asarray([state[index] for index in range(dimension)], dtype=float)
```

直接调用：

```python
np.asarray
range
```

## `_ompl_stage.valid_array`

[定义位置](../../../manipulation/planning.py#L1913)。

输入签名：

```python
def _ompl_stage.valid_array(value):
```

输出分支的源码表达式：

```python
False
True
```

直接写入的状态或数组位置：`report['validity_checks']`、`report['first_equality_rejection']`、`report['last_rejection']`、`report['rejections'][reason]`。

直接调用：

```python
time.monotonic
np.isfinite(value).all
np.isfinite
np.any
constraint.isSatisfied
np.asarray
equality
float
np.linalg.norm
np.max
np.abs
valid
report['rejections'].get
```

## `_ompl_stage.edge_key`

[定义位置](../../../manipulation/planning.py#L1944)。

输入签名：

```python
def _ompl_stage.edge_key(first, last):
```

输出分支的源码表达式：

```python
(first.tobytes(), last.tobytes())
```

直接调用：

```python
first.tobytes
last.tobytes
```

## `_ompl_stage.checked_edge`

[定义位置](../../../manipulation/planning.py#L1947)。

输入签名：

```python
def _ompl_stage.checked_edge(a, b):
```

输出分支的源码表达式：

```python
accepted_edges[key]
None
samples
```

直接写入的状态或数组位置：`accepted_edges[key]`。

直接调用：

```python
vector
edge_key
np.array_equal
valid_array
native_validator.checkMotion
og.PathGeometric
path.append
path.interpolate
np.asarray
path.getState
range
path.getStateCount
len
zip
max
int
np.ceil
np.linalg.norm
np.arange
np.unique
right.copy
_project_boundary
checked.append
accepted_edges.pop
next
iter
```

## `_ompl_stage.ForwardValidator`

[定义位置](../../../manipulation/planning.py#L2003)。

```python
class _ompl_stage.ForwardValidator(ob.MotionValidator):
    ...
    ...
```

该类型的直接方法：`__init__`、`checkMotion`。

## `_ompl_stage.ForwardValidator.__init__`

[定义位置](../../../manipulation/planning.py#L2004)。

输入签名：

```python
def _ompl_stage.ForwardValidator.__init__(self):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
super().__init__
super
```

## `_ompl_stage.ForwardValidator.checkMotion`

[定义位置](../../../manipulation/planning.py#L2007)。

输入签名：

```python
def _ompl_stage.ForwardValidator.checkMotion(self, a, b, *unused):
```

输出分支的源码表达式：

```python
checked_edge(a, b) is not None
```

直接写入的状态或数组位置：`report['motion_checks']`。

直接调用：

```python
checked_edge
```

## `_independent_projection`

[定义位置](../../../manipulation/planning.py#L2068)。

输入签名：

```python
def _independent_projection(equality, witness):
```

输出分支的源码表达式：

```python
(equality, 0)
(None, 0)
(lambda value, derivative: equality(value, derivative)[rows], rank)
```

直接调用：

```python
np.asarray
equality
len
np.isfinite(jacobian).all
np.isfinite
qr
int
np.linalg.matrix_rank
np.sort
```

## `_project_allowed_boundary`

[定义位置](../../../manipulation/planning.py#L2091)。

输入签名：

```python
def _project_allowed_boundary(value, residual, process_count, tolerance, lower, upper, *, deadline):
```

输出分支的源码表达式：

```python
None
point if np.all(point >= lower) and np.all(point <= upper) else None
point
reflected
```

直接写入的状态或数组位置：`point[-1]`、`reflected[-1]`。

直接调用：

```python
np.asarray(value, float).copy
np.asarray
len
np.isfinite(point).all
np.isfinite
np.any
range
time.monotonic
residual
np.isfinite(error).all
np.all
np.abs
np.isfinite(jacobian).all
qr
int
np.linalg.matrix_rank
np.sort
AllowedBoundary
point.copy
constraint.project
allowed
```

## `_project_allowed_boundary.allowed`

[定义位置](../../../manipulation/planning.py#L2114)。

输入签名：

```python
def _project_allowed_boundary.allowed(candidate):
```

输出分支的源码表达式：

```python
False
time.monotonic() < deadline and np.isfinite(error).all() and np.all(np.abs(error[:process_count]) <= tolerance) and np.all(error[process_count:] <= 0)
```

直接调用：

```python
time.monotonic
np.isfinite(candidate).all
np.isfinite
np.any
np.asarray
residual
np.isfinite(error).all
np.all
np.abs
```

## `_project_allowed_boundary.AllowedBoundary`

[定义位置](../../../manipulation/planning.py#L2147)。

```python
class _project_allowed_boundary.AllowedBoundary(ob.Constraint):
    ...
    ...
```

该类型的直接方法：`__init__`、`function`、`jacobian`。

## `_project_allowed_boundary.AllowedBoundary.__init__`

[定义位置](../../../manipulation/planning.py#L2148)。

输入签名：

```python
def _project_allowed_boundary.AllowedBoundary.__init__(self):
```

直接读取的对象属性：`self.setMaxIterations`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
super().__init__
super
np.finfo
self.setMaxIterations
```

## `_project_allowed_boundary.AllowedBoundary.function`

[定义位置](../../../manipulation/planning.py#L2158)。

输入签名：

```python
def _project_allowed_boundary.AllowedBoundary.function(self, values, output):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`output[:]`。

直接调用：

```python
residual
np.asarray
```

## `_project_allowed_boundary.AllowedBoundary.jacobian`

[定义位置](../../../manipulation/planning.py#L2161)。

输入签名：

```python
def _project_allowed_boundary.AllowedBoundary.jacobian(self, values, output):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`result[:, -1]`、`output[:]`。

直接调用：

```python
residual(np.asarray(values), True)[rows].copy
residual
np.asarray
```

## `_project_boundary`

[定义位置](../../../manipulation/planning.py#L2187)。

输入签名：

```python
def _project_boundary(value, equality, tolerance, *, deadline=None):
```

输出分支的源码表达式：

```python
value.copy()
None
point
point if constraint.isSatisfied(point) else None
```

直接写入的状态或数组位置：`point[-1]`。

直接调用：

```python
len
equality
value.copy
BoundaryConstraint
range
time.monotonic
constraint.isSatisfied
constraint.project
np.isfinite(point).all
np.isfinite
```

## `_project_boundary.BoundaryConstraint`

[定义位置](../../../manipulation/planning.py#L2201)。

```python
class _project_boundary.BoundaryConstraint(ob.Constraint):
    ...
    ...
```

该类型的直接方法：`__init__`、`function`、`jacobian`。

## `_project_boundary.BoundaryConstraint.__init__`

[定义位置](../../../manipulation/planning.py#L2202)。

输入签名：

```python
def _project_boundary.BoundaryConstraint.__init__(self):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
super().__init__
super
```

## `_project_boundary.BoundaryConstraint.function`

[定义位置](../../../manipulation/planning.py#L2205)。

输入签名：

```python
def _project_boundary.BoundaryConstraint.function(self, values, output):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`output[:]`。

直接调用：

```python
equality
np.asarray
```

## `_project_boundary.BoundaryConstraint.jacobian`

[定义位置](../../../manipulation/planning.py#L2208)。

输入签名：

```python
def _project_boundary.BoundaryConstraint.jacobian(self, values, output):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`output[:]`。

直接调用：

```python
equality
np.asarray
```

## `_orientation_vjp_type`

[定义位置](../../../manipulation/planning.py#L2224)。

输入签名：

```python
def _orientation_vjp_type():
```

输出分支的源码表达式：

```python
WorldQuaternionVJP
```

## `_orientation_vjp_type.WorldQuaternionVJP`

[定义位置](../../../manipulation/planning.py#L2239)。

```python
class _orientation_vjp_type.WorldQuaternionVJP(torch.autograd.Function):
    ...
    ...
```

该类型的直接方法：`forward`、`backward`。

## `_orientation_vjp_type.WorldQuaternionVJP.forward`

[定义位置](../../../manipulation/planning.py#L2241)。

输入签名：

```python
@staticmethod
def _orientation_vjp_type.WorldQuaternionVJP.forward(ctx, joint_position, quaternion, angular_jacobian):
```

输出分支的源码表达式：

```python
quaternion.clone()
```

直接调用：

```python
ctx.save_for_backward
quaternion.detach
angular_jacobian.detach
quaternion.clone
```

## `_orientation_vjp_type.WorldQuaternionVJP.backward`

[定义位置](../../../manipulation/planning.py#L2249)。

输入签名：

```python
@staticmethod
@torch.autograd.function.once_differentiable
def _orientation_vjp_type.WorldQuaternionVJP.backward(ctx, gradient):
```

输出分支的源码表达式：

```python
(joint_gradient, None, None)
```

直接调用：

```python
torch.linalg.cross
torch.einsum
```

## `_native_gradient_layout_type`

[定义位置](../../../manipulation/planning.py#L2265)。

输入签名：

```python
def _native_gradient_layout_type():
```

输出分支的源码表达式：

```python
NativeGradientLayout
```

## `_native_gradient_layout_type.NativeGradientLayout`

[定义位置](../../../manipulation/planning.py#L2275)。

```python
class _native_gradient_layout_type.NativeGradientLayout(torch.autograd.Function):
    ...
    ...
```

该类型的直接方法：`forward`、`backward`。

## `_native_gradient_layout_type.NativeGradientLayout.forward`

[定义位置](../../../manipulation/planning.py#L2277)。

输入签名：

```python
@staticmethod
def _native_gradient_layout_type.NativeGradientLayout.forward(ctx, value):
```

输出分支的源码表达式：

```python
value
```

## `_native_gradient_layout_type.NativeGradientLayout.backward`

[定义位置](../../../manipulation/planning.py#L2281)。

输入签名：

```python
@staticmethod
def _native_gradient_layout_type.NativeGradientLayout.backward(ctx, gradient):
```

输出分支的源码表达式：

```python
gradient.contiguous()
```

直接调用：

```python
gradient.contiguous
```

## `_verification_stop_preference`

[定义位置](../../../manipulation/planning.py#L2288)。

输入签名：

```python
def _verification_stop_preference(progress, joint_velocity, stop_progress, *, trajectory_time=None):
```

输出分支的源码表达式：

```python
torch.where(stopped, joint_velocity.square().sum(-1), torch.zeros_like(progress))
```

直接调用：

```python
torch.zeros_like
np.unique
_first_event_query
torch.where
joint_velocity.square().sum
joint_velocity.square
```

## `_formal_native_clock`

[定义位置](../../../manipulation/planning.py#L2307)。

输入签名：

```python
def _formal_native_clock(trajopt):
```

输出分支的源码表达式：

```python
{**grid, 'source': 'instantaneous-boundary cubic on unchanged exact float32 clock'}
```

直接调用：

```python
ValueError
_fixed_spline_grid
```

## `_joint_state_sample_times`

[定义位置](../../../manipulation/planning.py#L2328)。

输入签名：

```python
def _joint_state_sample_times(joint_state, counts):
```

输出分支的源码表达式：

```python
(indices.to(torch.float64) * joint_state.dt.reshape(-1, 1).to(torch.float64)).to(position.dtype)
```

直接调用：

```python
torch.minimum
torch.arange
(indices.to(torch.float64) * joint_state.dt.reshape(-1, 1).to(torch.float64)).to
indices.to
joint_state.dt.reshape(-1, 1).to
joint_state.dt.reshape
```

## `_timed_native_state`

[定义位置](../../../manipulation/planning.py#L2341)。

输入签名：

```python
def _timed_native_state(state, problem, scene, joint_names, trajectory_time, *, sphere_padding=0.0):
```

直接读取的对象属性：`scene.gripper`、`problem.tool_link`、`problem.transform_tool_pose`、`problem.transform_robot_spheres`、`problem.base_pose`、`result.cuda_robot_model_state`。

输出分支的源码表达式：

```python
state
result
```

直接写入的状态或数组位置：`moving[sphere_index]`、`result.cuda_robot_model_state`、`result.cuda_robot_model_state.robot_spheres`。

直接调用：

```python
tool.tool_frames.index
_native_gradient_layout_type
_orientation_vjp_type().apply
_orientation_vjp_type
tool.quaternion.detach
state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach
problem.transform_tool_pose
layout.apply
problem.transform_robot_spheres
torch.zeros_like
hand.state
hand.spheres
np.zeros
zip
bool
np.any
constant_tensor
torch.where
padding.abs
torch.cat
copy
spheres.contiguous
```

## `_native_collision_padding`

[定义位置](../../../manipulation/planning.py#L2389)。

输入签名：

```python
def _native_collision_padding(manager):
```

输出分支的源码表达式：

```python
0.0 if cost is None or not cost.enabled else cost.config.self_collision_kin_config.sphere_padding
```

直接调用：

```python
manager.get_cost
```

## `_confirm_native_self_collection`

[定义位置](../../../manipulation/planning.py#L2396)。

输入签名：

```python
def _confirm_native_self_collection(collection, confirmation, joint_position, joint_names, trajectory_time):
```

输出分支的源码表达式：

```python
collection
```

直接写入的状态或数组位置：`collection.values[index]`。

直接调用：

```python
enumerate
ValueError
confirmation.confirm_self
```

## `_scene_boundary_evaluate`

[定义位置](../../../manipulation/planning.py#L2421)。

输入签名：

```python
def _scene_boundary_evaluate(scene, problem, progress_map, spheres, position, quaternion, progress, joint_position, joint_names, *, trajectory_time, normalized_time, query_counts=None, component_diagnostics=None, native_confirmation=None, scene_evidence_out=None, optimizer_candidate_evidence=False, optimizer_scene_out=None, sample_valid=None):
```

直接读取的对象属性：`scene.evaluate`、`scene.required_progress_anchors`、`problem.start_progress`、`problem.end_progress`、`scene.contacts`。

输出分支的源码表达式：

```python
result
torch.stack((exact_terms, left_terms, right_terms)).amax(0)
```

直接写入的状态或数组位置：`scene_evidence_out[category]`、`optimizer_scene_out['nonworld_values']`。

直接调用：

```python
ValueError
float
torch.cuda.is_current_stream_capturing
MotionPlanningError
query_counts.update
int
progress.numel
evaluate
finish_optimizer
torch.zeros_like
progress_map.anchor_parameters
constant_tensor
torch.where
torch.nextafter
torch.full_like
left.any(0).nonzero().flatten
left.any(0).nonzero
left.any
right.any(0).nonzero().flatten
right.any(0).nonzero
right.any
torch.cat
lower.index_select
upper.index_select
permission_lower.index_select
left_columns.numel
right_columns.numel
expanded
torch.ones_like
exact_terms.scatter
left_columns[None].expand
right_columns[None].expand
scene_evidence_out.items
exact_evidence.scatter
torch.stack((exact_evidence, left_evidence, right_evidence)).amax
torch.stack
exact.scatter
torch.stack((exact, left_value, right_value)).amax
torch.stack((exact_terms, left_terms, right_terms)).amax
```

## `_scene_boundary_evaluate.evaluate`

[定义位置](../../../manipulation/planning.py#L2440)。

输入签名：

```python
def _scene_boundary_evaluate.evaluate(*args, **kwargs):
```

直接读取的对象属性：`scene.evaluate`。

输出分支的源码表达式：

```python
scene.evaluate(*args, **kwargs, native_confirmation=native_confirmation)
```

直接调用：

```python
scene.evaluate
```

## `_scene_boundary_evaluate.finish_optimizer`

[定义位置](../../../manipulation/planning.py#L2454)。

输入签名：

```python
def _scene_boundary_evaluate.finish_optimizer(times, validity, side):
```

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`optimizer_scene_out['world_objective']`。

直接调用：

```python
torch.argsort
order.gather
times.gather
(~validity).gather(1, order).to
(~validity).gather
collision_path_objective
```

## `_scene_boundary_evaluate.evaluate`

[定义位置](../../../manipulation/planning.py#L2474)。

输入签名：

```python
def _scene_boundary_evaluate.evaluate(*args, **kwargs):
```

输出分支的源码表达式：

```python
value
```

直接写入的状态或数组位置：`kwargs['collision_paths_out']`、`optimizer_scene_out['nonworld_values']`、`scene_evidence_out[category]`、`scene_evidence_out['optimizer_hard_invalid']`、`component_diagnostics[name]`。

直接调用：

```python
base_evaluate
sum
components.items
name.startswith
torch.zeros_like
term.detach
getattr
frozenset
torch.isfinite
value.detach
invalid.to
residual.detach().reshape
residual.detach
int
torch.where(torch.isfinite(flat), flat, torch.full_like(flat, float('inf'))).argmax().item
torch.where(torch.isfinite(flat), flat, torch.full_like(flat, float('inf'))).argmax
torch.where
torch.full_like
float
flat[index].item
np.isfinite
kwargs['trajectory_time'].detach().reshape(-1)[index].item
kwargs['trajectory_time'].detach().reshape
kwargs['trajectory_time'].detach
args[3].detach().reshape(-1)[index].item
args[3].detach().reshape
args[3].detach
```

## `_scene_boundary_evaluate.expanded`

[定义位置](../../../manipulation/planning.py#L2582)。

输入签名：

```python
def _scene_boundary_evaluate.expanded(value):
```

输出分支的源码表达式：

```python
torch.cat((value, value.index_select(1, left_columns), value.index_select(1, right_columns)), dim=1)
```

直接调用：

```python
torch.cat
value.index_select
```

## `_process_manager_type`

[定义位置](../../../manipulation/planning.py#L2649)。

输入签名：

```python
def _process_manager_type(problem, scene, progress_map, anchors, budget, *, clock=None, seed_parameter=None, robot_dict=None, phase='feasibility', functional_constraints=None):
```

直接读取的对象属性：`problem.functional_constraint_keys`。

输出分支的源码表达式：

```python
ProcessCostManagerCfg
```

直接写入的状态或数组位置：`ProcessCostManagerCfg.bind_request`、`ProcessCostManagerCfg.add_collision_witnesses`、`ProcessCostManagerCfg.query_anchor_parameters`、`ProcessCostManagerCfg.query_diagnostics`。

直接调用：

```python
_orientation_vjp_type
_native_gradient_layout_type
FunctionalAugmentedLagrangian
np.empty
staticmethod
```

## `_process_manager_type.query_anchor_parameters`

[定义位置](../../../manipulation/planning.py#L2680)。

输入签名：

```python
def _process_manager_type.query_anchor_parameters():
```

输出分支的源码表达式：

```python
pairs
```

直接调用：

```python
progress_map.anchor_parameters
len
np.column_stack
np.interp
np.concatenate
np.argsort
```

## `_process_manager_type.ProcessCostManager`

[定义位置](../../../manipulation/planning.py#L2690)。

```python
class _process_manager_type.ProcessCostManager(RobotCostManager):
    ...
    ...
```

该类型的直接方法：`initialize_from_config`、`bind_request_data`、`set_dense_counts`、`optimization_constraints`、`optimizer_evidence`、`bind_scene_evidence`、`compute_costs`。

## `_process_manager_type.ProcessCostManager.initialize_from_config`

[定义位置](../../../manipulation/planning.py#L2691)。

输入签名：

```python
def _process_manager_type.ProcessCostManager.initialize_from_config(self, config, transition_model, scene_collision_checker=None, **kwargs):
```

直接读取的对象属性：`self._process_mode`、`scene.clone_for_rollout`、`problem.tool_link`、`self.bind_request_data`、`self.get_cost`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self._process_mode`、`self._dense_counts`、`self._scene_evidence`、`self._optimizer_evidence`、`self._query_counts`、`transition_model.robot_model.compute_jacobian`、`self._scene`、`self._joint_names`、`self._tool_index`、`self._anchor_fk`、`self._anchor_self_collision`。

直接调用：

```python
deepcopy
super().initialize_from_config
super
managers.append
scene.clone_for_rollout
tuple
self._process_mode.startswith
transition_model.robot_model.tool_frames.index
self.bind_request_data
Kinematics
self.get_cost
SelfCollisionCost
```

## `_process_manager_type.ProcessCostManager.bind_request_data`

[定义位置](../../../manipulation/planning.py#L2739)。

输入签名：

```python
def _process_manager_type.ProcessCostManager.bind_request_data(self):
```

直接读取的对象属性：`scene.clone_for_rollout`、`self._query_counts`、`self._process_mode`、`budget.max_task_anchors`、`self.device_cfg`、`self._anchor_query_u`、`self._anchor_u`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self._scene`、`self._dense_counts`、`self._scene_evidence`、`self._optimizer_evidence`、`self._anchor_count`、`self._anchor_valid`、`self._anchor_query_u`。

直接调用：

```python
scene.clone_for_rollout
self._query_counts.clear
self._query_counts.update
self._process_mode.startswith
len
MotionPlanningError
np.arange
np.full
query_anchor_parameters
_anchor_capacity
np.concatenate
np.repeat
self.device_cfg.to_device
getattr
setattr
previous.copy_
torch.as_tensor
hasattr
self._anchor_query_u.copy_
functional_constraints.bind
```

## `_process_manager_type.ProcessCostManager.set_dense_counts`

[定义位置](../../../manipulation/planning.py#L2788)。

输入签名：

```python
def _process_manager_type.ProcessCostManager.set_dense_counts(self, counts):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._dense_counts`。

直接调用：

```python
counts.reshape(-1).long
counts.reshape
```

## `_process_manager_type.ProcessCostManager.optimization_constraints`

[定义位置](../../../manipulation/planning.py#L2793)。

输入签名：

```python
@property
def _process_manager_type.ProcessCostManager.optimization_constraints(self):
```

直接读取的对象属性：`self._process_mode`。

输出分支的源码表达式：

```python
functional_constraints if self._process_mode == 'trajectory_optimizer' else None
```

## `_process_manager_type.ProcessCostManager.optimizer_evidence`

[定义位置](../../../manipulation/planning.py#L2796)。

输入签名：

```python
def _process_manager_type.ProcessCostManager.optimizer_evidence(self, trajectories):
```

直接读取的对象属性：`self._optimizer_evidence`。

输出分支的源码表达式：

```python
(eligible, feasible, signed, fixed_objective)
```

直接调用：

```python
RuntimeError
collection.names.index
source.new_zeros
zip
value.detach().flatten
value.detach
torch.isfinite(detached).all
torch.isfinite
(detached <= 0).all
detached.sum
```

## `_process_manager_type.ProcessCostManager.bind_scene_evidence`

[定义位置](../../../manipulation/planning.py#L2845)。

输入签名：

```python
def _process_manager_type.ProcessCostManager.bind_scene_evidence(self, metrics):
```

直接读取的对象属性：`self._scene_evidence`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`metrics.debug`。

直接调用：

```python
RuntimeError
constraints.names.index
evidence.detach().clone
evidence.detach
```

## `_process_manager_type.ProcessCostManager.compute_costs`

[定义位置](../../../manipulation/planning.py#L2864)。

输入签名：

```python
def _process_manager_type.ProcessCostManager.compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
```

直接读取的对象属性：`self._process_mode`、`self._scene`、`self._dense_counts`、`self._joint_names`、`self._tool_index`、`problem.mechanisms`、`self._map_u`、`self._map_s`、`self._anchor_u`、`self._anchor_count`、`self._anchor_s`、`self._anchor_valid`、`self._anchor_query_u`、`self._anchor_fk`、`self.get_cost`、`self._anchor_self_collision`、`problem.transform_tool_velocity`、`problem.transform_tool_pose`、`problem.transform_robot_spheres`、`problem.residuals`、`self._query_counts`、`problem.functional_inequalities`、`budget.constraint_weight`、`problem.optimizer_preferences`、`problem.stop_progress`、`self._optimizer_evidence`。

输出分支的源码表达式：

```python
super().compute_costs(state, cost_collection, goal, **kwargs)
collection
```

直接写入的状态或数组位置：`self._optimizer_evidence`、`self._scene_evidence`。

直接调用：

```python
super().compute_costs
super
getattr
self._process_mode.startswith
torch.full
_joint_state_sample_times
(seed_parameter * active_clock['duration']).expand
_timed_native_state
_native_collision_padding
_confirm_native_self_collection
orientation_vjp.apply
tool.quaternion.detach
state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach
native_layout.apply
any
torch.einsum
(torch.arange(horizon, device=q.device, dtype=q.dtype)[None, :] / (counts[:, None] - 1)).clamp
torch.arange
_torch_interpolate
self._anchor_u[:, :self._anchor_count].expand
self._anchor_s[:, :self._anchor_count].expand
torch.ones_like
self._anchor_query_u[:, :self._anchor_count].expand
(query_anchor_u * active_clock['duration']).to
_hermite_at_parameters
JointState.from_position
anchor_q.contiguous
list
self._anchor_fk.compute_kinematics
self.get_cost
RobotState
self._anchor_self_collision.setup_batch_tensors
self._anchor_self_collision.forward
native_layout.apply(anchor_native.robot_spheres).contiguous
context['native_confirmation'].confirm_self
torch.cat
q.new_zeros
(anchor_collision * anchor_valid[..., None]).sum
collection.add
anchor_state.tool_poses.quaternion.detach
anchor_state.tool_jacobians[..., 3:, :].detach
torch.argsort
ordered
problem.transform_tool_velocity
problem.transform_tool_pose
problem.transform_robot_spheres
problem.residuals
sum
process_terms.items
name.startswith
torch.zeros_like
anchor_valid.expand(batch, -1).bool
anchor_valid.expand
_scene_boundary_evaluate
context.get
process_value.square
problem.functional_inequalities
functional_constraints.terms(signed).values
functional_constraints.terms
process_terms.values
torch.where(active, original_process, 0.0).detach
torch.where
torch.isfinite(original_process).all
torch.isfinite
(original_process <= 0).all
signed.values
residual.detach
(hard_invalid <= 0).all
scene_value.detach
torch.isfinite(raw_scene).all
(raw_scene <= 0).all
value.detach
signed.items
values.append
problem.optimizer_preferences
active_scene.optimizer_preferences
_verification_stop_preference
value.gather
(value[:, horizon:] * anchor_valid).sum
(regular + folded).unsqueeze
scene_evidence_out[category].gather
(evidence[:, horizon:] * anchor_valid).sum
channels.append
torch.stack
```

## `_process_manager_type.ProcessCostManager.compute_costs.ordered`

[定义位置](../../../manipulation/planning.py#L2984)。

输入签名：

```python
def _process_manager_type.ProcessCostManager.compute_costs.ordered(value):
```

输出分支的源码表达式：

```python
value.gather(1, index)
```

直接调用：

```python
order.reshape(*order.shape, *(1,) * (value.ndim - 2)).expand_as
order.reshape
value.gather
```

## `_process_manager_type.ProcessCostManagerCfg`

[定义位置](../../../manipulation/planning.py#L3123)。

```python
class _process_manager_type.ProcessCostManagerCfg(RobotCostManagerCfg):
    process_mode: str = ''
```

该类型的直接方法：`__post_init__`、`create`。

## `_process_manager_type.ProcessCostManagerCfg.__post_init__`

[定义位置](../../../manipulation/planning.py#L3126)。

输入签名：

```python
def _process_manager_type.ProcessCostManagerCfg.__post_init__(self):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.class_type`。

## `_process_manager_type.ProcessCostManagerCfg.create`

[定义位置](../../../manipulation/planning.py#L3130)。

输入签名：

```python
@staticmethod
def _process_manager_type.ProcessCostManagerCfg.create(data_dict, scene_collision_checker=None, device_cfg=None):
```

输出分支的源码表达式：

```python
ProcessCostManagerCfg(**values, process_mode=data_dict.get('process_mode', ''))
```

直接调用：

```python
RobotCostManagerCfg.create
getattr
fields
ProcessCostManagerCfg
data_dict.get
```

## `_process_manager_type.bind_request`

[定义位置](../../../manipulation/planning.py#L3135)。

输入签名：

```python
def _process_manager_type.bind_request(new_problem, new_scene, new_progress, new_anchors, new_budget, new_constraints):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
np.empty
manager.bind_request_data
```

## `_process_manager_type.add_collision_witnesses`

[定义位置](../../../manipulation/planning.py#L3146)。

输入签名：

```python
def _process_manager_type.add_collision_witnesses(parameters):
```

直接读取的对象属性：`budget.max_task_anchors`。

输出分支的源码表达式：

```python
extra
```

直接调用：

```python
np.asarray(parameters, dtype=np.float32).astype(float).reshape
np.asarray(parameters, dtype=np.float32).astype
np.asarray
np.isfinite(parameters).all
np.isfinite
np.any
ValueError
set
query_anchor_parameters()[:, 0].astype(np.float32).astype
query_anchor_parameters()[:, 0].astype
query_anchor_parameters
sorted
len
MotionPlanningError
np.sort
manager.bind_request_data
```

## `_trajectory_candidate`

[定义位置](../../../manipulation/planning.py#L3174)。

输入签名：

```python
def _trajectory_candidate(result, index):
```

直接读取的对象属性：`result.batch_size`、`result.num_seeds`、`result.seed_rank`。

输出分支的源码表达式：

```python
result
candidate
```

直接写入的状态或数组位置：`ordered.seed_rank`、`ordered.js_solution`、`ordered.interpolated_trajectory`、`ordered.metrics`、`ordered.interpolated_metrics`、`single.dt`。

直接调用：

```python
ValueError
copy
result.seed_rank.new_tensor
range
ordered.get_topk_seeds
getattr
state[:, index:index + 1].clone
state.dt[:, index:index + 1].clone
setattr
RolloutMetrics
CostsAndConstraints
CostCollection
list
value[index:index + 1].detach().clone
value[index:index + 1].detach
metrics.debug[index:index + 1].detach().clone
metrics.debug[index:index + 1].detach
```

## `_scene_metric_channel`

[定义位置](../../../manipulation/planning.py#L3232)。

输入签名：

```python
def _scene_metric_channel(metrics, channel):
```

直接读取的对象属性：`scene.shape`。

输出分支的源码表达式：

```python
metrics.debug[..., channel:channel + 1]
```

直接调用：

```python
constraints.names.index
RuntimeError
```

## `_scene_update_metric_residuals`

[定义位置](../../../manipulation/planning.py#L3241)。

输入签名：

```python
def _scene_update_metric_residuals(metrics, seed_index=None):
```

输出分支的源码表达式：

```python
{}
{'scene_update': float(flat[index].item())}
```

直接调用：

```python
_scene_metric_channel
evidence.reshape
flat.masked_fill(~flat.isfinite(), float('inf')).argmax
flat.masked_fill
flat.isfinite
float
flat[index].item
```

## `_coverage_metric_residuals`

[定义位置](../../../manipulation/planning.py#L3257)。

输入签名：

```python
def _coverage_metric_residuals(metrics, seed_index=None):
```

输出分支的源码表达式：

```python
{}
{'unknown_forbidden': float(flat[index].item())}
```

直接调用：

```python
_scene_metric_channel
evidence.reshape
flat.masked_fill(~flat.isfinite(), float('inf')).argmax
flat.masked_fill
flat.isfinite
float
flat[index].item
```

## `_conditional_unknown_metric_residuals`

[定义位置](../../../manipulation/planning.py#L3275)。

输入签名：

```python
def _conditional_unknown_metric_residuals(metrics, seed_index=None):
```

输出分支的源码表达式：

```python
{}
{'conditional_unknown': float(flat[index].item())}
```

直接调用：

```python
_scene_metric_channel
evidence.reshape
flat.masked_fill(~flat.isfinite(), float('inf')).argmax
flat.masked_fill
flat.isfinite
float
flat[index].item
```

## `_unknown_policy_failure`

[定义位置](../../../manipulation/planning.py#L3289)。

输入签名：

```python
def _unknown_policy_failure(report, policy):
```

输出分支的源码表达式：

```python
None
'needs_observation' if action == 'require_observation' else 'unknown_space_forbidden'
```

直接调用：

```python
report.get
components.items
np.isfinite
name.startswith
residuals.items
name.endswith
```

## `_collision_metric_residuals`

[定义位置](../../../manipulation/planning.py#L3328)。

输入签名：

```python
def _collision_metric_residuals(metrics, seed_index=None):
```

输出分支的源码表达式：

```python
residuals
```

直接写入的状态或数组位置：`residuals['scene_collision']`。

直接调用：

```python
maximum
zip
_scene_metric_channel
```

## `_collision_metric_residuals.maximum`

[定义位置](../../../manipulation/planning.py#L3337)。

输入签名：

```python
def _collision_metric_residuals.maximum(value):
```

输出分支的源码表达式：

```python
float((value if seed_index is None else value[seed_index]).max().item())
```

直接调用：

```python
float
(value if seed_index is None else value[seed_index]).max().item
(value if seed_index is None else value[seed_index]).max
```

## `_dense_collision_witnesses`

[定义位置](../../../manipulation/planning.py#L3348)。

输入签名：

```python
def _dense_collision_witnesses(result, *, regular_horizon, anchor_parameters, capacity):
```

直接读取的对象属性：`result.batch_size`、`result.num_seeds`、`result.interpolated_metrics`、`result.interpolated_last_tstep`、`result.interpolated_trajectory`。

输出分支的源码表达式：

```python
report
```

直接写入的状态或数组位置：`ordinary_measurements['scene_collision']`、`positive[-1]`、`positive_parameters[name]`。

直接调用：

```python
ValueError
int
result.interpolated_last_tstep.reshape(-1)[0].item
result.interpolated_last_tstep.reshape
(torch.arange(count, device=like.device, dtype=like.dtype) / (count - 1)).cpu().numpy
(torch.arange(count, device=like.device, dtype=like.dtype) / (count - 1)).cpu
torch.arange
(torch.arange(regular_horizon, device=like.device, dtype=like.dtype) / (regular_horizon - 1)).cpu().numpy
(torch.arange(regular_horizon, device=like.device, dtype=like.dtype) / (regular_horizon - 1)).cpu
set
regular_u.astype
np.asarray(anchor_parameters, dtype=dense_u.dtype).astype
np.asarray
np.array
float
enumerate
_collision_metric_residuals
zip
measurements.insert
_scene_metric_channel
getattr
dict
ordinary_measurements.get
tensor[0].detach().reshape(regular_horizon, -1).amax(-1).cpu().numpy
tensor[0].detach().reshape(regular_horizon, -1).amax(-1).cpu
tensor[0].detach().reshape(regular_horizon, -1).amax
tensor[0].detach().reshape
tensor[0].detach
np.isfinite
tensor[0, :count].detach().reshape(count, -1).amax(-1).cpu().numpy
tensor[0, :count].detach().reshape(count, -1).amax(-1).cpu
tensor[0, :count].detach().reshape(count, -1).amax
tensor[0, :count].detach().reshape
tensor[0, :count].detach
np.flatnonzero
np.diff
edges.reshape
np.any
len
np.argmax
peaks.append
component_peaks.append
sorted
range
max
candidates.append
report.update
```

## `_CuroboPlanner`

[定义位置](../../../manipulation/planning.py#L3449)。

```python
class _CuroboPlanner:
    ...
    ...
```

该类型的直接方法：`__init__`、`_initialize`、`_bind_online_robot`、`_bind_native_self_collision_pairs`、`rebind`、`_bind_formal_clock`、`_seed_u`、`_initialize_verification_constraints`、`_bind_physical_domains`、`initial_state_diagnostics`、`close`、`_goal`、`_seed_velocity_bounds`、`reference_seeds`、`ompl_seed`、`_functional_goal`、`_rank_ik_candidates`、`_build_goal_ik_candidates`、`_observed_native_result`、`_optimizer_reports`、`promote_collision_witnesses`、`_native_request`、`evaluate_reference`、`bind_reference_candidate`、`_candidate_metrics`、`evaluate_candidate`、`optimize`、`native_diagnostics`、`select_candidate`、`_native_curve`、`_candidate_domain_validation`、`export`。

## `_CuroboPlanner.__init__`

[定义位置](../../../manipulation/planning.py#L3453)。

输入签名：

```python
def _CuroboPlanner.__init__(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None, goal_identities=None):
```

直接读取的对象属性：`self._initialize`、`self.close`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.ik`、`self.trajopt`、`self.check_request`、`self._precision_torch`、`self._caller_precision`、`torch.backends.cuda.matmul.allow_tf32`。

直接调用：

```python
self._initialize
self.close
```

## `_CuroboPlanner._initialize`

[定义位置](../../../manipulation/planning.py#L3476)。

输入签名：

```python
def _CuroboPlanner._initialize(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None, goal_identities=None):
```

直接读取的对象属性：`scene.gripper`、`scene.validate_problem`、`problem.anchors`、`scene.required_progress_anchors`、`problem.start_progress`、`problem.end_progress`、`budget.max_validation_points`、`problem.reference_frame`、`scene.reference_frame`、`budget.knots`、`problem.stop_progress`、`budget.maximum_trajectory_dt`、`budget.interpolation_steps`、`budget.interpolation_dt`、`self.progress_map`、`self.reference_time_map`、`self.robot_metadata`、`self.anchor_capacity`、`self.validation_storage`、`budget.max_host_rss_bytes`、`self._caller_precision`、`self.device_cfg`、`budget.max_device_used_bytes`、`problem.functional_constraint_keys`、`self.anchors`、`self.formal_clock`、`self.seed_parameter`、`self.robot_dict`、`self.functional_constraints`、`budget.ik_iterations`、`budget.trajectory_iterations`、`budget.ik_seeds`、`problem.numerical`、`self._manager_type`、`budget.random_seed`、`self.ik`、`self.joint_names`、`problem.joint_names`、`problem.start_q`、`problem.start_dq`、`self._bind_online_robot`、`scene.clone_for_rollout`、`budget.trajectory_seeds`、`budget.interpolation_buffer_size`、`self._trajectory_configuration`、`self._bind_formal_clock`、`self._initialize_verification_constraints`、`self._bind_native_self_collision_pairs`、`self._bind_physical_domains`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.robot_validator`、`self.progress_map`、`self.source_progress_map`、`self.reference_time_map`、`self.finger_trajectory`、`self.anchor_capacity`、`self.robot_metadata`、`self.validation_storage`、`torch.backends.cuda.matmul.allow_tf32`、`torch.backends.cudnn.allow_tf32`、`self.problem`、`self.scene`、`self.budget`、`self.torch`、`self.JointState`、`self.GoalToolPose`、`self.device_cfg`、`self.scene_workspace_bytes`、`self.scene_cpu_workspace_bytes`、`self.scene_gpu_workspace_bytes`、`self.scene_cpu_live_bytes`、`self.anchor_fk_workspace_bytes`、`self._trajectory_optimizer_graph_admitted`、`self.ik`、`self.trajopt`、`self.anchors`、`self.robot_dict`、`self.seed_parameter`、`self.formal_clock`、`self.functional_constraints`、`self._manager_type`、`self.phase`、`self.goal_ik_candidates`、`self.last_optimizer_diagnostics`、`self.delivery_candidate`、`ik_cfg.use_lm_seed`、`ik_cfg.exit_early`、`self.joint_names`、`self.start`、`self._verification_fk`、`self._orientation_vjp`、`self._native_layout`、`self._verification_scene`、`self._trajectory_configuration`、`self._optimizer_class`、`self._optimizer_config_class`。

直接调用：

```python
robot_validator.validate_planning_binding
scene.validate_problem
_initial_timing
_continuation_timing
np.unique
np.concatenate
np.asarray
len
np.isfinite(anchors).all
np.isfinite
np.any
ValueError
_progress_for_cubic_boundaries
_fixed_spline_grid
_bind_event_clock
_ProgressMap
_phase_aligned_parameters
hand.bind_clock
_anchor_capacity
_bounded_robot_configuration
Path
kinematics.get
get_assets_path
ET.tostring
ET.parse(native_urdf).getroot
ET.parse
checker.model_record
dict
zip
set
any
float
np.float32
locks.items
int
checker._frame_id
str
_planning_workspace_bytes
_require_planning_host_memory
DeviceCfg
_require_planning_memory
self.device_cfg.to_device
FunctionalAugmentedLagrangian
_process_manager_type
configuration
InverseKinematicsCfg.create
deepcopy
InverseKinematics
tuple
RuntimeError
problem.joint_names.index
JointState.from_numpy
list
self._bind_online_robot
Kinematics
_orientation_vjp_type
_native_gradient_layout_type
scene.clone_for_rollout
load_yaml
join_path
get_task_configs_path
transition['transition_model_cfg'].update
cubic_transition_config_type
ProcessTrajectoryOptimizer
TrajectoryOptimizerCfg.create
self._bind_formal_clock
self._initialize_verification_constraints
self._bind_native_self_collision_pairs
self._bind_physical_domains
self._manager_type.bind_request
```

## `_CuroboPlanner._initialize.configuration`

[定义位置](../../../manipulation/planning.py#L3612)。

输入签名：

```python
def _CuroboPlanner._initialize.configuration(name, mode):
```

输出分支的源码表达式：

```python
data
```

直接写入的状态或数组位置：`constraint['process_mode']`。

直接调用：

```python
deepcopy
load_yaml
join_path
get_task_configs_path
data['rollout'].setdefault
constraint.pop
```

## `_CuroboPlanner._initialize.ProcessTrajectoryOptimizer`

[定义位置](../../../manipulation/planning.py#L3697)。

```python
class _CuroboPlanner._initialize.ProcessTrajectoryOptimizer(TrajectoryOptimizer):
    ...
    ...
```

该类型的直接方法：`_prepare_goal_buffer`、`_get_result`、`get_interpolated_trajectory`。

## `_CuroboPlanner._initialize.ProcessTrajectoryOptimizer._prepare_goal_buffer`

[定义位置](../../../manipulation/planning.py#L3699)。

输入签名：

```python
def _CuroboPlanner._initialize.ProcessTrajectoryOptimizer._prepare_goal_buffer(self, solve_state, goal_tool_poses, current_state=None, use_implicit_goal=False, seed_goal_state=None, goal_state=None):
```

输出分支的源码表达式：

```python
(buffer, update_reference)
```

直接写入的状态或数组位置：`buffer.batch_size`。

直接调用：

```python
super()._prepare_goal_buffer
super
```

## `_CuroboPlanner._initialize.ProcessTrajectoryOptimizer._get_result`

[定义位置](../../../manipulation/planning.py#L3717)。

输入签名：

```python
def _CuroboPlanner._initialize.ProcessTrajectoryOptimizer._get_result(self, opt_result, metrics_result, interpolated_metrics, interpolated_trajectory, last_tstep, num_seeds, return_seeds):
```

直接读取的对象属性：`self.metrics_rollout`、`self.additional_metrics_rollouts`。

输出分支的源码表达式：

```python
super()._get_result(opt_result, metrics_result, interpolated_metrics, interpolated_trajectory, last_tstep, num_seeds, return_seeds)
```

直接调用：

```python
self.metrics_rollout.metrics_constraint_manager.bind_scene_evidence
self.additional_metrics_rollouts['interpolated_rollout'].metrics_constraint_manager.bind_scene_evidence
super()._get_result
super
```

## `_CuroboPlanner._initialize.ProcessTrajectoryOptimizer.get_interpolated_trajectory`

[定义位置](../../../manipulation/planning.py#L3726)。

输入签名：

```python
def _CuroboPlanner._initialize.ProcessTrajectoryOptimizer.get_interpolated_trajectory(self, js_optimized):
```

直接读取的对象属性：`self.config`、`self.horizon`、`self.action_dim`、`self.action_horizon`、`self._interpolated_traj_buffer`、`self.device_cfg`、`self.joint_names`、`self._interpolation_dt_buffer`、`self.core`、`self.transition_model`、`self.additional_metrics_rollouts`。

输出分支的源码表达式：

```python
(state, counts, False)
```

直接写入的状态或数组位置：`self._interpolated_traj_buffer`、`self._interpolation_dt_buffer`、`js_optimized.control_space`。

直接调用：

```python
_formal_native_clock
ValueError
getattr
value.numel
bool
(value == expected).all
JointState.zeros
torch.full
(self._interpolation_dt_buffer == grid['interpolation_dt']).all
step.sample
rollout.constraint_manager.set_dense_counts
rollout.metrics_constraint_manager.set_dense_counts
```

## `_CuroboPlanner._bind_online_robot`

[定义位置](../../../manipulation/planning.py#L3790)。

输入签名：

```python
def _CuroboPlanner._bind_online_robot(self):
```

直接读取的对象属性：`self.ik`、`self.robot_validator`、`self.scene`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
params.link_spheres.detach().cpu().numpy().reshape
params.link_spheres.detach().cpu().numpy
params.link_spheres.detach().cpu
params.link_spheres.detach
params.link_sphere_idx_map.detach().cpu().numpy().reshape
params.link_sphere_idx_map.detach().cpu().numpy
params.link_sphere_idx_map.detach().cpu
params.link_sphere_idx_map.detach
int
params.link_name_to_idx_map.items
np.flatnonzero(links == link).tolist
np.flatnonzero
spheres[links == link, :3].tolist
spheres[links == link, 3].tolist
np.unique
self.ik.kinematics.get_self_collision_config
self.scene.bind_online_robot
collision.collision_pairs.detach().cpu().numpy().tolist
collision.collision_pairs.detach().cpu().numpy
collision.collision_pairs.detach().cpu
collision.collision_pairs.detach
collision.sphere_padding.detach().cpu().numpy().tolist
collision.sphere_padding.detach().cpu().numpy
collision.sphere_padding.detach().cpu
collision.sphere_padding.detach
str
validator.T_reference_base.tolist
validator.T_frame_tcp.tolist
```

## `_CuroboPlanner._bind_native_self_collision_pairs`

[定义位置](../../../manipulation/planning.py#L3809)。

输入签名：

```python
def _CuroboPlanner._bind_native_self_collision_pairs(self):
```

直接读取的对象属性：`self._verification_native`、`self.ik`、`self.trajopt`、`self.torch`、`self.scene`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`cost._enabled_before_pair_replacement`、`cost.config.self_collision_kin_config`、`cost._batch_size`、`manager._batch_size`。

直接调用：

```python
solver.core.get_all_rollout_instances
solver.additional_metrics_rollouts.values
managers.extend
set
manager.get_cost
getattr
id
visited.add
self.torch.as_tensor(self.scene.native_self_collision_pairs, dtype=original.collision_pairs.dtype, device=original.collision_pairs.device).reshape
self.torch.as_tensor
hasattr
replace
len
cost.enable_cost
cost.disable_cost
```

## `_CuroboPlanner.rebind`

[定义位置](../../../manipulation/planning.py#L3846)。

输入签名：

```python
def _CuroboPlanner.rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None):
```

直接读取的对象属性：`self.robot_validator`、`scene.gripper`、`scene.validate_problem`、`problem.anchors`、`scene.required_progress_anchors`、`problem.start_progress`、`problem.end_progress`、`budget.knots`、`problem.stop_progress`、`budget.maximum_trajectory_dt`、`budget.interpolation_steps`、`budget.interpolation_dt`、`self.reference_time_map`、`self.robot_metadata`、`self.anchor_capacity`、`budget.max_host_rss_bytes`、`self.torch`、`self.device_cfg`、`budget.max_device_used_bytes`、`self.formal_clock`、`problem.joint_names`、`self.joint_names`、`self.JointState`、`problem.start_q`、`problem.start_dq`、`self._bind_online_robot`、`problem.functional_constraint_keys`、`self._manager_type`、`self.functional_constraints`、`self._bind_native_self_collision_pairs`、`scene.clone_for_rollout`、`self.trajopt`、`self._bind_formal_clock`、`self.seed_parameter`、`self.ik`、`self._bind_physical_domains`。

输出分支的源码表达式：

```python
self
```

直接写入的状态或数组位置：`self.source_progress_map`、`self.reference_time_map`、`self.problem`、`self.scene`、`self.budget`、`self.progress_map`、`self.anchors`、`self.finger_trajectory`、`self.start`、`self.functional_constraints`、`self._verification_scene`、`self.trajopt.config.minimum_trajectory_dt`、`self.trajopt.config.maximum_trajectory_dt`、`transitions[id(rollout.metrics_transition_model)]`、`self.trajopt.config.interpolation_dt`、`self.last_validation`、`self.last_optimizer_diagnostics`、`self.goal_ik_candidates`、`self.delivery_candidate`、`manager._goal_buffer.link_goal_poses`、`goal.link_goal_poses`、`goal.goal_js`。

直接调用：

```python
self.robot_validator.validate_planning_binding
scene.validate_problem
_initial_timing
_continuation_timing
np.unique
_progress_for_cubic_boundaries
_fixed_spline_grid
_bind_event_clock
_ProgressMap
_phase_aligned_parameters
_anchor_capacity
_planning_workspace_bytes
_require_planning_host_memory
_require_planning_memory
scene.gripper.bind_clock
self.formal_clock.clear
self.formal_clock.update
problem.joint_names.index
self.JointState.from_numpy
list
self._bind_online_robot
FunctionalAugmentedLagrangian
self._manager_type.bind_request
self._bind_native_self_collision_pairs
scene.clone_for_rollout
id
self.trajopt.core.get_all_rollout_instances
self.trajopt.additional_metrics_rollouts.values
transitions.values
transition.update_traj_dt
self.trajopt._interpolation_dt_buffer.fill_
self._bind_formal_clock
self.seed_parameter.zero_
solver.reset_seed
solver.core.get_all_rollout_instances
rollout.reset
getattr
self._bind_physical_domains
```

## `_CuroboPlanner._bind_formal_clock`

[定义位置](../../../manipulation/planning.py#L3933)。

输入签名：

```python
def _CuroboPlanner._bind_formal_clock(self):
```

直接读取的对象属性：`self.trajopt`、`self.formal_clock`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
_formal_native_clock
any
native.get
self.formal_clock.items
ValueError
self.formal_clock.update
```

## `_CuroboPlanner._seed_u`

[定义位置](../../../manipulation/planning.py#L3941)。

输入签名：

```python
def _CuroboPlanner._seed_u(self, progress):
```

直接读取的对象属性：`self.progress_map`。

输出分支的源码表达式：

```python
float(np.interp(progress, values, self.progress_map.u[indices]))
```

直接调用：

```python
np.unique
float
np.interp
```

## `_CuroboPlanner._initialize_verification_constraints`

[定义位置](../../../manipulation/planning.py#L3951)。

输入签名：

```python
def _CuroboPlanner._initialize_verification_constraints(self):
```

直接读取的对象属性：`self.device_cfg`、`self._verification_native`、`self.trajopt`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._RobotState`、`self._verification_native`。

直接调用：

```python
RobotCostManager
self._verification_native.initialize_from_config
deepcopy
```

## `_CuroboPlanner._bind_physical_domains`

[定义位置](../../../manipulation/planning.py#L3964)。

输入签名：

```python
def _CuroboPlanner._bind_physical_domains(self):
```

直接读取的对象属性：`self._verification_native`、`self.joint_names`、`self.start`、`self.formal_clock`、`self.trajopt`、`self.ik`、`self.position_domain`、`self.physical_domain`、`self._trajectory_optimizer_graph_admitted`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.position_domain`、`self.physical_domain`。

直接调用：

```python
self._verification_native.get_cost
limits.joint_names.index
JointPositionDomain
CubicJointDomain
bind_solver_components
```

## `_CuroboPlanner.initial_state_diagnostics`

[定义位置](../../../manipulation/planning.py#L3985)。

输入签名：

```python
def _CuroboPlanner.initial_state_diagnostics(self):
```

直接读取的对象属性：`self._verification_native`、`self.joint_names`、`self.start`、`self.torch`、`self.JointState`、`self._verification_fk`、`self._RobotState`、`self.problem`、`self.scene`、`self.robot_validator`、`self.check_request`、`self.progress_map`。

输出分支的源码表达式：

```python
{'passed': not failed, 'residuals': residuals, 'failed_components': failed, 'scene_components': scene_components, 'native_confirmation': deepcopy(native_confirmation.last_report), 'scope': 'measured joint limits and selected online collision model'}
```

直接写入的状态或数组位置：`components[name]`、`components['self_collision']`、`components['scene']`。

直接调用：

```python
self._verification_native.get_cost
limits.joint_names.index
self.torch.no_grad
getattr
self.torch.relu
self.JointState
q.contiguous
dq.contiguous
list
self._verification_fk.compute_kinematics
self._RobotState
_timed_native_state
q.new_zeros
_native_collision_padding
collision.setup_batch_tensors
collision.forward
native.robot_spheres.contiguous
state.tool_poses.tool_frames.index
self.problem.transform_tool_pose
q.new_full
OnlineCollisionConfirmation
native_confirmation.confirm_self
_scene_boundary_evaluate
self.problem.transform_robot_spheres
float
value.max().item
value.max
components.items
residuals.items
np.isfinite
deepcopy
```

## `_CuroboPlanner.close`

[定义位置](../../../manipulation/planning.py#L4042)。

输入签名：

```python
def _CuroboPlanner.close(self):
```

直接读取的对象属性：`self._precision_torch`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._precision_torch.backends.cuda.matmul.allow_tf32`、`self._precision_torch.backends.cudnn.allow_tf32`、`self._caller_precision`。

直接调用：

```python
getattr
solver.destroy
setattr
gc.collect
```

## `_CuroboPlanner._goal`

[定义位置](../../../manipulation/planning.py#L4065)。

输入签名：

```python
def _CuroboPlanner._goal(self, progress):
```

直接读取的对象属性：`self.device_cfg`、`self.problem`、`self.GoalToolPose`。

输出分支的源码表达式：

```python
self.GoalToolPose(tool_frames=[self.problem.tool_link], position=position[:, :, None, None].contiguous(), quaternion=quaternion[:, :, None, None].contiguous())
```

直接调用：

```python
self.device_cfg.to_device
float
self.problem.reference_to_link_pose
self.problem.sample_reference
self.GoalToolPose
position[:, :, None, None].contiguous
quaternion[:, :, None, None].contiguous
```

## `_CuroboPlanner._seed_velocity_bounds`

[定义位置](../../../manipulation/planning.py#L4073)。

输入签名：

```python
def _CuroboPlanner._seed_velocity_bounds(self):
```

直接读取的对象属性：`self._verification_native`、`self.joint_names`。

输出分支的源码表达式：

```python
limits.velocity.detach().cpu().numpy()[:, columns]
```

直接调用：

```python
self._verification_native.get_cost
limits.joint_names.index
limits.velocity.detach().cpu().numpy
limits.velocity.detach().cpu
limits.velocity.detach
```

## `_CuroboPlanner.reference_seeds`

[定义位置](../../../manipulation/planning.py#L4079)。

输入签名：

```python
def _CuroboPlanner.reference_seeds(self, joint_reference=None):
```

直接读取的对象属性：`self.joint_names`、`self.source_progress_map`、`self.reference_time_map`、`self.progress_map`、`self.physical_domain`、`self.device_cfg`、`self._seed_velocity_bounds`、`self.formal_clock`、`self.budget`、`self.torch`、`self.start`、`self.check_request`、`self.seed_parameter`、`self.ik`、`self._goal`、`result.js_solution`、`result.success`、`self._rank_ik_candidates`、`self.reference_ik_distinct_counts`、`self.reference_ik_queries`、`self.problem`、`self.initial_seed_strategies`、`self.trajopt`、`self.reference_seed_fits`、`self._build_goal_ik_candidates`、`self.goal_ik_fit_hints`。

输出分支的源码表达式：

```python
(seeds, [])
(seeds.contiguous(), success_counts)
```

直接写入的状态或数组位置：`self.joint_reference_report`、`self.reference_ik_queries`、`self.reference_ik_distinct_counts`、`self.reference_ik_chains`、`self.reference_seed_fits`、`self.goal_ik_fit_hints`、`self.initial_seed_strategies`、`self.seed_strategies`、`report['reference_status']`、`report[name]`、`self.initial_seed_strategies[1]`、`self.initial_seed_strategies[branch]`、`targets[branch]`、`self.reference_seed_fits[branch]['functional_hint_fit']`、`hint['applied']`、`seeds[0, branch]`。

直接调用：

```python
isinstance
set
MotionPlanningError
joint_reference.joint_names.index
np.isfinite(q).all
np.isfinite
np.isfinite(dq).all
_ProgressMap
_phase_aligned_parameters
np.asarray
self.reference_time_map.anchor_parameters
self.physical_domain.fit_reference
self.device_cfg.to_device
self._seed_velocity_bounds
controls[None, None].expand(1, count, -1, -1).contiguous
controls[None, None].expand
self.physical_domain.validate
self.start.position[:, None].clone
enumerate
self.check_request
float
np.interp
self.seed_parameter.fill_
self.ik.solve_pose
self._goal
self.ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy
self.ik.get_active_js(result.js_solution).position[0].detach().cpu
self.ik.get_active_js(result.js_solution).position[0].detach
self.ik.get_active_js
result.success[0].detach().cpu().numpy
result.success[0].detach().cpu
result.success[0].detach
success_counts.append
int
(flags & np.isfinite(values).all(-1)).sum
np.isfinite(values).all
self._rank_ik_candidates
previous[0].detach().cpu().numpy
previous[0].detach().cpu
previous[0].detach
self.reference_ik_distinct_counts.append
len
np.isfinite(values[index]).all
getattr
measurement[0].detach().cpu().numpy().reshape
measurement[0].detach().cpu().numpy
measurement[0].detach().cpu
measurement[0].detach
ValueError
zip
pools.append
evidence.append
self.reference_ik_queries.append
self.device_cfg.to_device(values[qualified][None]).contiguous
select_ik_chains
self.start.position.detach().cpu().numpy
self.start.position.detach().cpu
self.start.position.detach
targets.append
np.broadcast_to(self.start.position[0].detach().cpu().numpy(), (self.trajopt.action_horizon, len(self.joint_names))).copy
np.broadcast_to
self.start.position[0].detach().cpu().numpy
self.start.position[0].detach().cpu
self.start.position[0].detach
self.initial_seed_strategies.append
self.trajopt._trajectory_seed_generator.generate_constant_seeds
self.physical_domain.initialize(value).reshape
self.physical_domain.initialize
self.physical_domain.evaluate
controls.append
self.reference_seed_fits.append
torch.stack
self._build_goal_ik_candidates
range
list
sorted
pool.items
np.isfinite(row['joint_position']).all
min
samples.append
postures.append
self.goal_ik_fit_hints.append
deepcopy
seeds.contiguous
```

## `_CuroboPlanner.ompl_seed`

[定义位置](../../../manipulation/planning.py#L4277)。

输入签名：

```python
def _CuroboPlanner.ompl_seed(self, initial_seed, candidate_trajectory):
```

直接读取的对象属性：`self.torch`、`self.budget`、`self.trajopt`、`self.problem`、`self.device_cfg`、`self.progress_map`、`self.anchors`、`self.finger_trajectory`、`self.formal_clock`、`self.start`、`self.scene`、`scene.with_release_history`、`self.joint_names`、`self.physical_domain`、`self._seed_velocity_bounds`。

输出分支的源码表达式：

```python
(controls[None, None].contiguous(), report)
(None, report)
```

直接写入的状态或数组位置：`report['chart_times']`、`frame['attempts']`、`evidence['boundary_state']`、`evidence['failure']`、`frame['path']`、`report['budget_exhausted']`、`report['state_queries']`、`report['elapsed_seconds']`、`report['failure']`。

直接调用：

```python
time.monotonic
self.trajopt.transition_model.action_bound_lows.detach().cpu().numpy().reshape
self.trajopt.transition_model.action_bound_lows.detach().cpu().numpy
self.trajopt.transition_model.action_bound_lows.detach().cpu
self.trajopt.transition_model.action_bound_lows.detach
self.trajopt.transition_model.action_bound_highs.detach().cpu().numpy().reshape
self.trajopt.transition_model.action_bound_highs.detach().cpu().numpy
self.trajopt.transition_model.action_bound_highs.detach().cpu
self.trajopt.transition_model.action_bound_highs.detach
_projection_tolerance
self.device_cfg.to_device
self.progress_map.anchor_parameters
dict
np.unique
self.start.position[0].detach().cpu().numpy
self.start.position[0].detach().cpu
self.start.position[0].detach
getattr
boundary_times.append
enumerate
goals_by_time.setdefault(float(parameter), []).append
goals_by_time.setdefault
float
boundary_times.extend
boundaries.tolist
np.random.default_rng
initial_seed[0].detach().cpu().numpy
initial_seed[0].detach().cpu
initial_seed[0].detach
self.scene.clone_for_rollout
search_frame
len
frames.pop
next
anchor_progress.get
np.interp
self.problem.projection_requirements
goals_at_time
equations
boundary_equations
_project_allowed_boundary
list
report['branches'].append
goal.tolist
_independent_projection
invalid_states.clear
max
_ompl_stage
min
evidence.update
deepcopy
np.any
np.diff
sum
torch.no_grad
state_parts
scene.with_release_history
frames.append
np.concatenate
np.array_split
int
np.ceil
positions.append
p.clone
rotations.append
r.clone
progress_values.append
progress.clone
torch.cat
self.problem.evaluate
torch.zeros_like
bool
torch.isfinite(process).all
torch.isfinite
(process <= 0).all
self.problem.verification_issues
self.physical_domain.fit_reference
self._seed_velocity_bounds
report.update
controls[None, None].contiguous
```

## `_CuroboPlanner.ompl_seed.goals_at_time`

[定义位置](../../../manipulation/planning.py#L4336)。

输入签名：

```python
def _CuroboPlanner.ompl_seed.goals_at_time(parameter):
```

输出分支的源码表达式：

```python
tuple(goals_by_time.get(float(parameter), ()))
```

直接调用：

```python
tuple
goals_by_time.get
float
```

## `_CuroboPlanner.ompl_seed.check_search`

[定义位置](../../../manipulation/planning.py#L4340)。

输入签名：

```python
def _CuroboPlanner.ompl_seed.check_search():
```

直接读取的对象属性：`self.check_request`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
self.check_request
time.monotonic
TimeoutError
```

## `_CuroboPlanner.ompl_seed.state_parts`

[定义位置](../../../manipulation/planning.py#L4346)。

输入签名：

```python
def _CuroboPlanner.ompl_seed.state_parts(values):
```

直接读取的对象属性：`self.budget`、`self.JointState`、`self.device_cfg`、`self.joint_names`、`self._verification_fk`、`self.problem`、`self._orientation_vjp`、`self._native_layout`。

输出分支的源码表达式：

```python
(js, state, p, r, _torch_interpolate(map_u, map_s, values[..., -1]))
```

直接写入的状态或数组位置：`work['state_queries']`。

直接调用：

```python
check_search
int
np.prod
min
TimeoutError
values[..., :-1].contiguous
torch.zeros_like
self.JointState
self.device_cfg.to_device
list
self._verification_fk.compute_kinematics
tools.tool_frames.index
self._orientation_vjp.apply
tools.quaternion.detach
state.tool_jacobians[..., 3:, :].detach
self._native_layout.apply
self.problem.transform_tool_pose
_torch_interpolate
```

## `_CuroboPlanner.ompl_seed.equations`

[定义位置](../../../manipulation/planning.py#L4365)。

输入签名：

```python
def _CuroboPlanner.ompl_seed.equations(scene, active, goals=()):
```

输出分支的源码表达式：

```python
evaluate
```

## `_CuroboPlanner.ompl_seed.equations.residual_function`

[定义位置](../../../manipulation/planning.py#L4367)。

输入签名：

```python
def _CuroboPlanner.ompl_seed.equations.residual_function(values):
```

直接读取的对象属性：`self.problem`、`self.joint_names`、`scene.project_residual`。

输出分支的源码表达式：

```python
torch.cat((process, drive, goal), -1).reshape(-1)
```

直接调用：

```python
state_parts
values.reshape
self.problem.project_residual
scene.project_residual
self.problem.goal_residual
torch.cat((process, drive, goal), -1).reshape
torch.cat
```

## `_CuroboPlanner.ompl_seed.equations.evaluate`

[定义位置](../../../manipulation/planning.py#L4375)。

输入签名：

```python
def _CuroboPlanner.ompl_seed.equations.evaluate(values, derivative):
```

直接读取的对象属性：`self.device_cfg`、`self.budget`。

输出分支的源码表达式：

```python
cache[key]
np.empty((0, len(values)))
result
```

直接写入的状态或数组位置：`cache[key]`。

直接调用：

```python
np.asarray(values, dtype=float).tobytes
np.asarray
self.device_cfg.to_device
torch.enable_grad
residual_function(values_t).numel
residual_function
np.empty
len
torch.autograd.functional.jacobian
torch.no_grad
answer.detach().cpu().numpy().astype
answer.detach().cpu().numpy
answer.detach().cpu
answer.detach
min
```

## `_CuroboPlanner.ompl_seed.valid`

[定义位置](../../../manipulation/planning.py#L4399)。

输入签名：

```python
def _CuroboPlanner.ompl_seed.valid(values, scene, confirmation):
```

直接读取的对象属性：`self.device_cfg`、`self.formal_clock`、`self.problem`、`self.joint_names`、`self.progress_map`、`self._RobotState`、`self._verification_native`。

输出分支的源码表达式：

```python
accepted
```

直接调用：

```python
torch.no_grad
self.device_cfg.to_device
state_parts
self.problem.evaluate
goals_at_time
float
_scene_boundary_evaluate
self.problem.transform_robot_spheres
_timed_native_state
self._RobotState
_native_collision_padding
self._verification_native.compute_costs
_confirm_native_self_collection
terms.update
zip
check_search
_ompl_terms_accepted
len
invalid_states.append
progress.item
term.max().item
term.max
terms.items
```

## `_CuroboPlanner.ompl_seed.proposals`

[定义位置](../../../manipulation/planning.py#L4438)。

输入签名：

```python
def _CuroboPlanner.ompl_seed.proposals(begin, parameter):
```

直接读取的对象属性：`self.goal_ik_candidates`、`self.physical_domain`。

生成输出的源码表达式：

```python
('ranked_functional_ik', np.r_[row['joint_position'], parameter], {'functional_ik_candidate_index': index, 'functional_ik_report': deepcopy(row['report'])})
('candidate_curve', np.r_[candidate_trajectory._curve(parameter * candidate_trajectory.duration), parameter], {})
('joint_box_random', np.r_[rng.uniform(lower, upper), parameter], {})
('reference_curve', np.r_[self.physical_domain.evaluate(controls, np.asarray([parameter]))[0], parameter], {})
('previous_boundary', np.r_[begin[:-1], parameter], {})
```

直接调用：

```python
self.goal_ik_candidates.get
float
enumerate
np.isfinite(row['joint_position']).all
np.isfinite
min
len
deepcopy
candidate_trajectory._curve
rng.uniform
self.physical_domain.evaluate
np.asarray
```

## `_CuroboPlanner.ompl_seed.search_frame`

[定义位置](../../../manipulation/planning.py#L4475)。

输入签名：

```python
def _CuroboPlanner.ompl_seed.search_frame(begin, branch_scene, parameter):
```

直接读取的对象属性：`self.robot_validator`。

输出分支的源码表达式：

```python
{'start': begin, 'scene': branch_scene, 'confirmation': confirmation, 'proposals': proposals(begin, parameter), 'attempts': 0, 'path': None}
```

直接调用：

```python
OnlineCollisionConfirmation
proposals
```

## `_CuroboPlanner.ompl_seed.stage_valid`

[定义位置](../../../manipulation/planning.py#L4529)。

输入签名：

```python
def _CuroboPlanner.ompl_seed.stage_valid(value):
```

输出分支的源码表达式：

```python
validity_cache[key]
```

直接写入的状态或数组位置：`validity_cache[key]`。

直接调用：

```python
np.asarray(value, dtype=float).tobytes
np.asarray
valid
```

## `_CuroboPlanner._functional_goal`

[定义位置](../../../manipulation/planning.py#L4608)。

输入签名：

```python
def _CuroboPlanner._functional_goal(self, target):
```

直接读取的对象属性：`self.device_cfg`、`self.problem`、`self.GoalToolPose`。

输出分支的源码表达式：

```python
self.GoalToolPose(tool_frames=[self.problem.tool_link], position=position[:, :, None, None].contiguous(), quaternion=quaternion[:, :, None, None].contiguous())
```

直接调用：

```python
np.linalg.inv
self.device_cfg.to_device
Rotation.from_matrix(pose[:3, :3]).as_quat
Rotation.from_matrix
self.problem.reference_to_link_pose
self.GoalToolPose
position[:, :, None, None].contiguous
quaternion[:, :, None, None].contiguous
```

## `_CuroboPlanner._rank_ik_candidates`

[定义位置](../../../manipulation/planning.py#L4625)。

输入签名：

```python
def _CuroboPlanner._rank_ik_candidates(self, candidates, success, previous, u, goal_ids=()):
```

直接读取的对象属性：`self.progress_map`、`self.scene`、`self.robot_validator`、`self.check_request`、`self.torch`、`self.device_cfg`、`self.JointState`、`self.budget`、`self.joint_names`、`self._verification_fk`、`self.problem`、`self.formal_clock`、`self._RobotState`、`self._verification_native`。

输出分支的源码表达式：

```python
(ranked, reports)
```

直接写入的状态或数组位置：`report['invalid_joint_indices']`。

直接调用：

```python
np.asarray
float
np.interp
OnlineCollisionConfirmation
enumerate
bool
reports.append
np.isfinite(row).all
np.isfinite
np.flatnonzero(~np.isfinite(row)).tolist
np.flatnonzero
self.torch.no_grad
self.device_cfg.to_device(row[None, None]).contiguous
self.device_cfg.to_device
self.torch.zeros_like
self.JointState
list
self._verification_fk.compute_kinematics
state.tool_poses.tool_frames.index
self.problem.transform_tool_pose
q.new_full
self.problem.residuals
task.items
name.startswith
_scene_boundary_evaluate
self.problem.transform_robot_spheres
_timed_native_state
self._RobotState
_native_collision_padding
self._verification_native.compute_costs
_confirm_native_self_collection
dict
zip
report.update
_ompl_terms_accepted
failures
rank_ik_candidates
```

## `_CuroboPlanner._rank_ik_candidates.failures`

[定义位置](../../../manipulation/planning.py#L4675)。

输入签名：

```python
def _CuroboPlanner._rank_ik_candidates.failures(terms):
```

直接读取的对象属性：`self.torch`。

输出分支的源码表达式：

```python
{name: value for (name, value) in maxima.items() if value > 0 or not np.isfinite(value)}
```

直接写入的状态或数组位置：`maxima[name]`。

直接调用：

```python
terms.items
value.detach().reshape
value.detach
self.torch.where(self.torch.isfinite(flat), flat, self.torch.full_like(flat, float('inf'))).argmax
self.torch.where
self.torch.isfinite
self.torch.full_like
float
flat[index].item
maxima.items
np.isfinite
```

## `_CuroboPlanner._build_goal_ik_candidates`

[定义位置](../../../manipulation/planning.py#L4691)。

输入签名：

```python
def _CuroboPlanner._build_goal_ik_candidates(self, reference_seeds):
```

直接读取的对象属性：`self.problem`、`self.progress_map`、`self.goal_ik_candidates`、`self.physical_domain`、`self.seed_parameter`、`self.ik`、`self._functional_goal`、`self.start`、`self.device_cfg`、`self.budget`、`result.js_solution`、`result.success`、`self._rank_ik_candidates`、`self.formal_clock`。

输出分支的源码表达式：

```python
self.goal_ik_candidates
```

直接写入的状态或数组位置：`self.goal_ik_candidates`、`self.goal_ik_candidates[u]`。

直接调用：

```python
reference_seeds[0].detach().cpu().numpy
reference_seeds[0].detach().cpu
reference_seeds[0].detach
sorted
enumerate
float
self.progress_map.anchor_parameters
self.physical_domain.evaluate
np.asarray
self.seed_parameter.fill_
self.ik.solve_pose
self._functional_goal
self.device_cfg.to_device(previous[None]).contiguous
self.device_cfg.to_device
self.ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy
self.ik.get_active_js(result.js_solution).position[0].detach().cpu
self.ik.get_active_js(result.js_solution).position[0].detach
self.ik.get_active_js
result.success[0].detach().cpu().numpy
result.success[0].detach().cpu
result.success[0].detach
tuple
self.problem.goal_requirements_at
getattr
self._rank_ik_candidates
list
candidates[index].tolist
```

## `_CuroboPlanner._observed_native_result`

[定义位置](../../../manipulation/planning.py#L4737)。

输入签名：

```python
def _CuroboPlanner._observed_native_result(self, actions):
```

直接读取的对象属性：`self.torch`、`self.trajopt`。

输出分支的源码表达式：

```python
self.trajopt._get_best_result(result, count)
```

直接写入的状态或数组位置：`joint.joint_names`、`joint.knot`、`joint.knot_dt`。

直接调用：

```python
self.torch.no_grad
self.trajopt.metrics_rollout.compute_metrics_from_action
self.trajopt._interpolate_and_compute_metrics
self.trajopt._get_result
self.trajopt._get_best_result
```

## `_CuroboPlanner._optimizer_reports`

[定义位置](../../../manipulation/planning.py#L4752)。

输入签名：

```python
def _CuroboPlanner._optimizer_reports(self, solution, initial):
```

直接读取的对象属性：`self.trajopt`。

输出分支的源码表达式：

```python
reports
```

直接调用：

```python
range
(solution[0, index] - initial[0, index]).abs
reports.append
bool
(displacement == 0).all().item
(displacement == 0).all
float
displacement.max().item
displacement.max
```

## `_CuroboPlanner.promote_collision_witnesses`

[定义位置](../../../manipulation/planning.py#L4765)。

输入签名：

```python
def _CuroboPlanner.promote_collision_witnesses(self, result):
```

直接读取的对象属性：`self._trajectory_configuration`、`self._manager_type`、`result.js_solution`、`self.budget`。

输出分支的源码表达式：

```python
report
```

直接调用：

```python
MotionPlanningError
self._manager_type.query_anchor_parameters
_dense_collision_witnesses
len
self._manager_type.add_collision_witnesses
report.update
bool
```

## `_CuroboPlanner._native_request`

[定义位置](../../../manipulation/planning.py#L4780)。

输入签名：

```python
def _CuroboPlanner._native_request(self, seed):
```

直接读取的对象属性：`self.problem`、`self.scene`、`self.route`、`self.trajopt`、`self._functional_goal`。

输出分支的源码表达式：

```python
(state, goal)
```

直接写入的状态或数组位置：`self.route`。

直接调用：

```python
next
any
len
np.array_equal
np.eye
self.trajopt.enable_tool_pose_tracking
self.trajopt.disable_tool_pose_tracking
self.trajopt.disable_joint_position_tracking
self._functional_goal
SolveState
```

## `_CuroboPlanner.evaluate_reference`

[定义位置](../../../manipulation/planning.py#L4804)。

输入签名：

```python
def _CuroboPlanner.evaluate_reference(self, seed):
```

直接读取的对象属性：`self.physical_domain`、`self._native_request`、`self.JointState`、`self.torch`、`self.budget`、`self.trajopt`、`self.start`、`self._observed_native_result`。

输出分支的源码表达式：

```python
self._observed_native_result(seed.reshape(-1, self.trajopt.action_horizon, self.trajopt.action_dim))
```

直接写入的状态或数组位置：`endpoint.dt`、`self.last_optimizer_diagnostics`。

直接调用：

```python
self.physical_domain.validate
self._native_request
self.JointState.from_position
seed[:, :, -1].contiguous
self.torch.full
self.trajopt._prepare_goal_buffer
self.trajopt._update_rollout_params
range
self._observed_native_result
seed.reshape
```

## `_CuroboPlanner.bind_reference_candidate`

[定义位置](../../../manipulation/planning.py#L4818)。

输入签名：

```python
def _CuroboPlanner.bind_reference_candidate(self, reference):
```

直接读取的对象属性：`self.joint_names`、`self.problem`、`self.scene`、`record.items`。

输出分支的源码表达式：

```python
(trajectory, candidate_scene, mapping)
```

直接调用：

```python
isinstance
set
len
MotionPlanningError
_ProgressMap
reference.progress_u.copy
reference.progress_s.copy
self.scene.clone_for_rollout
_reference_clock_issues
str
_bind_event_clock
hand.bind_clock
tuple
deepcopy
record.items
mapping.event_clock.values
JointTrajectory
reference.times.copy
reference.position.copy
reference.velocity.copy
mapping.u.copy
mapping.s.copy
getattr
```

## `_CuroboPlanner._candidate_metrics`

[定义位置](../../../manipulation/planning.py#L4846)。

输入签名：

```python
def _CuroboPlanner._candidate_metrics(self, trajectory, candidate_scene, *, native_query=False):
```

直接读取的对象属性：`self.scene`、`self.robot_validator`、`self.check_request`、`trajectory.progress_u`、`trajectory.progress_s`、`self.anchors`、`self._manager_type`、`self.budget`、`self.trajopt`、`self.formal_clock`、`trajectory.joint_names`、`self.joint_names`、`trajectory.sample`、`trajectory.duration`、`self.device_cfg`、`self.JointState`、`self.start`、`trajectory._curve`、`self.torch`。

输出分支的源码表达式：

```python
(report, times / trajectory.duration, len(times) + len(pairs))
```

直接写入的状态或数组位置：`endpoint.dt`、`manager._candidate_query`。

直接调用：

```python
OnlineCollisionConfirmation
_ProgressMap
mapping.anchor_parameters
self._manager_type.query_anchor_parameters
np.setdiff1d
len
np.concatenate
np.column_stack
np.interp
np.argsort
_candidate_confirmation_times
_candidate_sample_times
min
MotionPlanningError
trajectory.joint_names.index
trajectory.sample
tensor
tensor(anchor_q[:, columns])[None].contiguous
tensor(anchor_dq[:, columns])[None].contiguous
self.trajopt.disable_tool_pose_tracking
self.trajopt.disable_joint_position_tracking
SolveState
self.JointState.from_position
tensor(q[-1:, columns])[None].contiguous
self.trajopt._prepare_goal_buffer
self.trajopt._update_rollout_params
getattr
any
RuntimeError
self.JointState
tensor(q[:, columns])[None].contiguous
tensor(dq[:, columns])[None].contiguous
tensor(ddq[:, columns])[None].contiguous
tensor(trajectory._curve(times, 3)[:, columns])[None].contiguous
trajectory._curve
list
self.check_request
self.torch.no_grad
rollout.transition_model.compute_augmented_state
rollout.compute_metrics_from_state
rollout.metrics_constraint_manager.bind_scene_evidence
_collision_metric_residuals
_scene_update_metric_residuals
_coverage_metric_residuals
_conditional_unknown_metric_residuals
maximum
zip
isinstance
bool
metrics.feasible.all().item
metrics.feasible.all
all
np.isfinite
residuals.values
collision_residuals.items
scene_update_residuals.items
coverage_residuals.items
conditional_unknown_residuals.items
deepcopy
```

## `_CuroboPlanner._candidate_metrics.maximum`

[定义位置](../../../manipulation/planning.py#L4946)。

输入签名：

```python
def _CuroboPlanner._candidate_metrics.maximum(value):
```

直接读取的对象属性：`self.torch`。

输出分支的源码表达式：

```python
float(flat[index].item())
```

直接调用：

```python
value.detach().reshape
value.detach
self.torch.where(self.torch.isfinite(flat), flat, self.torch.full_like(flat, float('inf'))).argmax
self.torch.where
self.torch.isfinite
self.torch.full_like
float
flat[index].item
```

## `_CuroboPlanner.evaluate_candidate`

[定义位置](../../../manipulation/planning.py#L4982)。

输入签名：

```python
def _CuroboPlanner.evaluate_candidate(self, trajectory, *, native_report=None, candidate_scene=None):
```

直接读取的对象属性：`self._candidate_metrics`、`self._candidate_domain_validation`、`trajectory.residuals`、`validation.update`。

输出分支的源码表达式：

```python
(trajectory, attempt, points)
```

直接写入的状态或数组位置：`self.delivery_candidate`、`self.last_validation`。

直接调用：

```python
self._candidate_metrics
deepcopy
np.array
self._candidate_domain_validation
attempt.update
bool
all
np.isfinite
residuals.values
validation.update
np.full
max
```

## `_CuroboPlanner.optimize`

[定义位置](../../../manipulation/planning.py#L4999)。

输入签名：

```python
def _CuroboPlanner.optimize(self, seed):
```

直接读取的对象属性：`self.physical_domain`、`self._native_request`、`self.trajopt`、`self.start`、`self.torch`、`self.budget`、`result.solution`、`self._optimizer_reports`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`self.last_optimizer_diagnostics`。

直接调用：

```python
self.physical_domain.validate
self._native_request
seed.detach().clone
seed.detach
self.trajopt._solve_impl
self.torch.full
ValueError
self._optimizer_reports
```

## `_CuroboPlanner.native_diagnostics`

[定义位置](../../../manipulation/planning.py#L5016)。

输入签名：

```python
def _CuroboPlanner.native_diagnostics(self, result, seed_index=None):
```

直接读取的对象属性：`result.success`、`result.metrics`、`result.interpolated_metrics`。

输出分支的源码表达式：

```python
report
```

直接写入的状态或数组位置：`report[label]`。

直接调用：

```python
bool
success.all().item
success.all
maximum
zip
_collision_metric_residuals
_scene_update_metric_residuals
_coverage_metric_residuals
_conditional_unknown_metric_residuals
```

## `_CuroboPlanner.native_diagnostics.maximum`

[定义位置](../../../manipulation/planning.py#L5023)。

输入签名：

```python
def _CuroboPlanner.native_diagnostics.maximum(value):
```

输出分支的源码表达式：

```python
float(flat.max().item())
float(flat[index].item())
```

直接调用：

```python
selected.detach().reshape
selected.detach
flat.is_floating_point
float
flat.max().item
flat.max
torch.where(torch.isfinite(flat), flat, torch.full_like(flat, float('inf'))).argmax
torch.where
torch.isfinite
torch.full_like
flat[index].item
```

## `_CuroboPlanner.select_candidate`

[定义位置](../../../manipulation/planning.py#L5058)。

输入签名：

```python
def _CuroboPlanner.select_candidate(self, result):
```

直接读取的对象属性：`result.num_seeds`、`self.native_diagnostics`、`self.seed_strategies`、`self.last_optimizer_diagnostics`、`self._native_curve`、`self._candidate_metrics`、`self.scene`。

输出分支的源码表达式：

```python
(candidate, trajectory, {**report, 'selected_seed_index': selected, 'candidates': reports, 'confirmation_validation_points': confirmation_points})
```

直接写入的状态或数组位置：`confirmed['accepted']`、`reports[index]`。

直接调用：

```python
range
self.native_diagnostics
metrics.get(label, {}).get('constraints', {}).items
metrics.get(label, {}).get
metrics.get
all
np.isfinite
residuals.values
list
sum
max
float
metrics.get(label, {}).get('collision_residuals', {}).items
metrics.get(label, {}).get('scene_update_residuals', {}).items
metrics.get(label, {}).get('coverage_residuals', {}).items
metrics.get(label, {}).get('conditional_unknown_residuals', {}).items
reports.append
_trajectory_candidate
self._native_curve
sorted
len
original['residuals'].items
name.rsplit
original['residuals'].values
self._candidate_metrics
bool
confirmed['residuals'].values
```

## `_CuroboPlanner._native_curve`

[定义位置](../../../manipulation/planning.py#L5134)。

输入签名：

```python
def _CuroboPlanner._native_curve(self, result):
```

直接读取的对象属性：`result.batch_size`、`result.num_seeds`、`self.trajopt`、`result.get_interpolated_plan`、`self.joint_names`、`self.formal_clock`、`self.problem`、`self.progress_map`、`self.finger_trajectory`。

输出分支的源码表达式：

```python
trajectory
```

直接写入的状态或数组位置：`trajectory.finger_trajectory`、`trajectory.satisfied_dependencies`。

直接调用：

```python
ValueError
self.trajopt.get_active_js
result.get_interpolated_plan
native.position.detach().cpu().numpy().reshape
native.position.detach().cpu().numpy
native.position.detach().cpu
native.position.detach
len
native.velocity.detach().cpu().numpy().reshape
native.velocity.detach().cpu().numpy
native.velocity.detach().cpu
native.velocity.detach
float
native.dt.detach().cpu().reshape
native.dt.detach().cpu
native.dt.detach
np.isfinite(position).all
np.isfinite
np.isfinite(velocity).all
np.arange
event_occurrence_key
asdict
event_records.append
deepcopy
clock.items
JointTrajectory
self.progress_map.u.copy
self.progress_map.s.copy
tuple
getattr
```

## `_CuroboPlanner._candidate_domain_validation`

[定义位置](../../../manipulation/planning.py#L5166)。

输入签名：

```python
def _CuroboPlanner._candidate_domain_validation(self, trajectory):
```

直接读取的对象属性：`self._verification_native`、`trajectory.joint_names`、`trajectory._curve`、`self.joint_names`、`trajectory.position`、`self.start`、`self.problem`、`trajectory.velocity`。

输出分支的源码表达式：

```python
{'continuous_position': position, 'continuous_velocity': velocity, 'cubic_derivatives': derivatives, 'numerical_tolerances': asdict(self.problem.numerical)}
```

直接写入的状态或数组位置：`velocity['maximum_violation']['velocity']`、`trajectory.residuals`。

直接调用：

```python
self._verification_native.get_cost
limits.joint_names.index
limits.position[:, columns].detach().cpu().numpy
limits.position[:, columns].detach().cpu
limits.position[:, columns].detach
cubic_position_report
limits.velocity[:, columns].detach().cpu().numpy
limits.velocity[:, columns].detach().cpu
limits.velocity[:, columns].detach
trajectory._curve.derivative
velocity['maximum_violation'].pop
cubic_derivative_report
limits.acceleration.detach().cpu().numpy
limits.acceleration.detach().cpu
limits.acceleration.detach
self.joint_names.index
max
float
np.max
np.abs
self.start.position[0].detach().cpu().numpy
self.start.position[0].detach().cpu
self.start.position[0].detach
self.start.velocity[0].detach().cpu().numpy
self.start.velocity[0].detach().cpu
self.start.velocity[0].detach
asdict
```

## `_CuroboPlanner.export`

[定义位置](../../../manipulation/planning.py#L5189)。

输入签名：

```python
def _CuroboPlanner.export(self, result, native_report):
```

直接读取的对象属性：`self.evaluate_candidate`、`self._native_curve`。

输出分支的源码表达式：

```python
(trajectory, points)
```

直接调用：

```python
self.evaluate_candidate
self._native_curve
MotionPlanningError
```

## `_verified_dependency_outcomes`

[定义位置](../../../manipulation/planning.py#L5198)。

输入签名：

```python
def _verified_dependency_outcomes(continuation):
```

直接读取的对象属性：`continuation.get`。

输出分支的源码表达式：

```python
set()
satisfied
```

直接调用：

```python
set
event_occurrence_key
continuation.get
row.get
evidence.get
isinstance
np.isfinite
event.get
satisfied.add
```

## `_bind_motion_work_ledger`

[定义位置](../../../manipulation/planning.py#L5220)。

输入签名：

```python
def _bind_motion_work_ledger(ledger, block, continuation, entrance_consumed):
```

输出分支的源码表达式：

```python
ledger['motion_request']['planning_baseline']
```

直接写入的状态或数组位置：`ledger['motion_request']`。

直接调用：

```python
ledger.get
continuation['original_binding'].get
old.get
_verified_dependency_outcomes
bool
event.get
dict.fromkeys
deepcopy
```

## `MotionPlanningError`

[定义位置](../../../manipulation/planning.py#L5239)。

```python
class MotionPlanningError(ValueError):
    ...
    ...
```

该类型的直接方法：`__init__`。

## `MotionPlanningError.__init__`

[定义位置](../../../manipulation/planning.py#L5241)。

输入签名：

```python
def MotionPlanningError.__init__(self, reason, message):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.reason`。

直接调用：

```python
super().__init__
super
```

## `MotionBackend`

[定义位置](../../../manipulation/planning.py#L5246)。

```python
class MotionBackend:
    ...
    ...
```

该类型的直接方法：`__init__`、`_claim`、`_release`、`bind_request`、`plan`、`close`。

## `MotionBackend.__init__`

[定义位置](../../../manipulation/planning.py#L5253)。

输入签名：

```python
def MotionBackend.__init__(self, robot, *, robot_validator, budget=None, device='cuda:0'):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.robot`、`self.robot_validator`、`self.device`、`self.budget`、`self._planner`、`self._closed`、`self._busy`、`self._request_owner`。

直接调用：

```python
deepcopy
PlanningBudget
Lock
```

## `MotionBackend._claim`

[定义位置](../../../manipulation/planning.py#L5261)。

输入签名：

```python
def MotionBackend._claim(self):
```

直接读取的对象属性：`self._busy`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._request_owner`。

直接调用：

```python
self._busy.acquire
MotionPlanningError
get_ident
```

## `MotionBackend._release`

[定义位置](../../../manipulation/planning.py#L5266)。

输入签名：

```python
def MotionBackend._release(self):
```

直接读取的对象属性：`self._request_owner`、`self._busy`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._request_owner`。

直接调用：

```python
get_ident
RuntimeError
self._busy.release
```

## `MotionBackend.bind_request`

[定义位置](../../../manipulation/planning.py#L5272)。

输入签名：

```python
def MotionBackend.bind_request(self, problem, scene, *, budget, continuation=None, goal_identities=None):
```

直接读取的对象属性：`self._request_owner`、`self._closed`、`self.budget`、`self._planner`、`self.robot`、`self.device`、`self.robot_validator`。

输出分支的源码表达式：

```python
self._planner
```

直接写入的状态或数组位置：`self._planner`。

直接调用：

```python
get_ident
MotionPlanningError
getattr
_CuroboPlanner
self._planner.rebind
```

## `MotionBackend.plan`

[定义位置](../../../manipulation/planning.py#L5288)。

输入签名：

```python
def MotionBackend.plan(self, problem, scene, *, continuation=None, request_ledger=None, rebind=None, cancelled=None, deadline=None, joint_reference=None):
```

直接读取的对象属性：`self.robot`、`self.robot_validator`、`self.budget`、`self.device`。

输出分支的源码表达式：

```python
plan_trajectory(problem, scene, self.robot, robot_validator=self.robot_validator, budget=self.budget, device=self.device, continuation=continuation, request_ledger=request_ledger, rebind=rebind, backend=self, cancelled=cancelled, deadline=deadline, joint_reference=joint_reference)
```

直接调用：

```python
plan_trajectory
```

## `MotionBackend.close`

[定义位置](../../../manipulation/planning.py#L5295)。

输入签名：

```python
def MotionBackend.close(self):
```

直接读取的对象属性：`self._claim`、`self._planner`、`self._release`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._planner`、`self._closed`。

直接调用：

```python
self._claim
self._planner.close
self._release
```

## `plan_trajectory`

[定义位置](../../../manipulation/planning.py#L5308)。

输入签名：

```python
def plan_trajectory(problem, scene, robot, *, robot_validator, budget=None, device='cuda:0', continuation=None, request_ledger=None, rebind=None, backend=None, cancelled=None, deadline=None, joint_reference=None):
```

直接读取的对象属性：`scene.clone_for_rollout`、`continuation.get`、`problem.events`、`problem.bound_motion_block`、`scene.gripper`、`budget.ik_seeds`、`budget.trajectory_seeds`、`scene.describe`、`scene.unknown_policy`、`problem.source_time_map`、`trajectory.joint_names`、`trajectory.times`、`trajectory.position`、`trajectory.velocity`、`budget.search_seconds`。

输出分支的源码表达式：

```python
outcome
```

直接写入的状态或数组位置：`diagnostics['wallclock']`、`ledger_input[name]`、`diagnostics['continuation_facts_pending_validation']`、`ledger['requires_continuation']`、`consumed['binding_attempts']`、`current['consumed_planning']`、`diagnostics['consumed_planning']`、`prior['history']`、`prior['timing_events']`、`diagnostics['motion_block']`、`consumed['initializations']`、`scene.gripper.commands`、`planner.check_request`、`planner.satisfied_dependencies`、`optimizer._core.request_guard`、`diagnostics['verification_joint_limits']`、`consumed['validation_queries']`、`diagnostics['initial_state']`、`diagnostics['failure']`、`diagnostics['event_time_checks']`、`consumed['validation_passes']`、`diagnostics['original_reference_candidate']`、`diagnostics['joint_reference']`、`outcome.execution_binding['verification_joint_limits']`、`diagnostics['reference_clock']`、`consumed['ik_batches']`、`diagnostics['functional_ik_candidates']`、`diagnostics['functional_ik_fit_hints']`、`diagnostics['joint_reference']['source_curve_preserved']`、`diagnostics['joint_reference']['candidate_kind']`、`diagnostics['reference_candidate']`、`consumed['trajectory_solves']`、`diagnostics['collision_witness_update']`、`planner.seed_strategies`、`consumed['search_seconds']`、`consumed['search_queries']`、`diagnostics['ompl']`、`diagnostics['returned_candidate']`、`diagnostics['diagnostic_trajectory']`、`diagnostics['diagnostic_candidate_scope']`、`outcome.success`、`outcome.trajectory`、`outcome.execution_binding`、`diagnostics['rollout_query_counts']`、`diagnostics['elapsed_seconds']`、`ledger['consumed_planning']['planning_seconds']`、`diagnostics['wall_time_overrun_s']`、`diagnostics['planning_budget_exhausted']`、`outcome.request_ledger`、`outcome.execution_binding['consumed_planning']`、`outcome.execution_binding['request_ledger']`。

直接调用：

```python
time.monotonic
mark_wallclock
PlanningResult
scene.clone_for_rollout
isinstance
continuation.get
continuation['original_binding'].get
PlanningBudget
deepcopy
_request_work_ledger
asdict
diagnostics.update
check
_restore_original_direction_origins
_obligation_contract
ContinuationBindingError
_prepare_continuation
diagnostics.pop
any
plain
rebind.get
_verified_dependency_outcomes
problem.bound_motion_block
str
MotionPlanningError
_bind_motion_work_ledger
replace
max
motion_baseline.get
_reference_clock_issues
MotionBackend
backend._claim
event_occurrence_key
tuple
backend.bind_request
sorted
scene.describe
getattr
planner._verification_native.get_cost
list
getattr(limits, name).detach().cpu().numpy().tolist
getattr(limits, name).detach().cpu().numpy
getattr(limits, name).detach().cpu
getattr(limits, name).detach
planner.initial_state_diagnostics
_unknown_policy_failure
planner.bind_reference_candidate
_validate_bound_event_times
planner.evaluate_candidate
qualify_candidate
candidate_scene.describe
reference_attempt.get('scene_components', {}).items
reference_attempt.get
name.startswith
np.isfinite
_fixed_spline_grid
float
planner.reference_seeds
bool
planner.evaluate_reference
select_candidate_with_accounting
np.array_equal
planner.optimize
diagnostics['attempts'].append
attempt['collision_residuals'].values
attempt['coverage_residuals'].values
planner.promote_collision_witnesses
native_result.solution.detach().clone().expand(1, budget.trajectory_seeds, -1, -1).contiguous
native_result.solution.detach().clone().expand
native_result.solution.detach().clone
native_result.solution.detach
planner.ompl_seed
report.get
planner.export
type
traceback.format_exc
manager_type.query_diagnostics
backend._release
backend.close
ledger['binding_records'].append
outcome.execution_binding.get
diagnostics.get
```

## `plan_trajectory.mark_wallclock`

[定义位置](../../../manipulation/planning.py#L5331)。

输入签名：

```python
def plan_trajectory.mark_wallclock(name):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`diagnostics['wallclock']['events'][name]`。

直接调用：

```python
time.time_ns
time.monotonic
```

## `plan_trajectory.check`

[定义位置](../../../manipulation/planning.py#L5375)。

输入签名：

```python
def plan_trajectory.check(next_stage=None):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`consumed['planning_seconds']`。

直接调用：

```python
cancelled
MotionPlanningError
time.monotonic
motion_baseline.get
consumed.items
_check_work_budget
```

## `plan_trajectory.qualify_candidate`

[定义位置](../../../manipulation/planning.py#L5514)。

输入签名：

```python
def plan_trajectory.qualify_candidate(trajectory, validation, attempt, *, binding_scene=None):
```

直接读取的对象属性：`trajectory.executable_until`、`trajectory.next_verification`、`problem.events`。

输出分支的源码表达式：

```python
True
```

直接写入的状态或数组位置：`diagnostics['qualified_candidate_elapsed_seconds']`、`outcome.success`、`outcome.trajectory`、`outcome.execution_binding`。

直接调用：

```python
ValueError
diagnostics.update
bind_candidate
deepcopy
any
event_occurrence_key
ledger['timing_events'].append
plain
set
binding.update
range
len
enumerate
getattr
tuple
sorted
check
time.monotonic
mark_wallclock
```

## `plan_trajectory.select_candidate_with_accounting`

[定义位置](../../../manipulation/planning.py#L5631)。

输入签名：

```python
def plan_trajectory.select_candidate_with_accounting(result):
```

输出分支的源码表达式：

```python
(native, curve, report)
```

直接写入的状态或数组位置：`consumed['validation_passes']`、`consumed['validation_queries']`。

直接调用：

```python
planner.select_candidate
report.get
check
```
