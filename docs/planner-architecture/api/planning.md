# `manipulation/planning.py` 实现契约

代码基线：`ae2603f`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_fixed_spline_grid`

[定义位置](../../../manipulation/planning.py#L52)。

输入签名：

```python
def _fixed_spline_grid(*, intervals, native_dt, interpolation_steps, interpolation_dt):
```

实现说明：

> Admit an output grid only when it preserves the native physical clock.
>
> Regular and dense samples must include every polynomial breakpoint on
> identical float32 and saved binary64 time axes. Rounding only proposes
> an integer subdivision count; it never authorizes retiming an input.

显式输出分支：

```python
{'native_dt': float(raw_dt), 'knot_dt': float(knot_dt), 'interpolation_dt': float(output_dt), 'subdivisions': subdivisions, 'valid_count': count, 'duration': duration}
```

读取的对象属性：

```python
np.errstate
np.float32
np.isfinite([raw_dt, output_dt, knot_dt]).all
np.isfinite
```

写入目标：

```python
raw_dt
output_dt
knot_dt
subdivisions
reconstructed
count
duration
```

直接调用符号：

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

## `_derived_spline_grid`

[定义位置](../../../manipulation/planning.py#L87)。

输入签名：

```python
def _derived_spline_grid(*, intervals, native_dt, interpolation_steps, interpolation_dt):
```

实现说明：

> Choose exact subdivisions within the requested float32 spacing bound.
>
> ``interpolation_dt`` is a maximum spacing at request admission. A power
> of two subdivision preserves the represented knot duration exactly; the
> resulting grid is then checked by the same strict actual-clock validator
> used by native, event, finger and exported-curve consumers.

显式输出分支：

```python
_fixed_spline_grid(intervals=intervals, native_dt=native_dt, interpolation_steps=interpolation_steps, interpolation_dt=float(knot_dt) / subdivisions)
```

读取的对象属性：

```python
np.errstate
np.float32
np.isfinite([limit, knot_dt]).all
np.isfinite
np.ceil
(needed - 1).bit_length
```

写入目标：

```python
limit
knot_dt
needed
subdivisions
```

直接调用符号：

```python
np.errstate
np.float32
np.isfinite([limit, knot_dt]).all
np.isfinite
min
ValueError
max
int
np.ceil
float
(needed - 1).bit_length
_fixed_spline_grid
```

## `PlanningBudget`

[定义位置](../../../manipulation/planning.py#L108)。

```python
class PlanningBudget():
```

类字段：

```python
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

实现说明：

> Explicit local resource and numerical-work limits, for a single request.
>
> These defaults bound the current 16 GB RAM / 12 GB GPU integration. More
> seeds or a larger batch must not be silently allocated by native defaults.
> The interpolation buffer covers the largest allowed native knot interval.
> Native solvers use eager execution for the dynamic process/scene tensors.
> ``max_task_anchors`` is the fixed semantic-anchor capacity. Native queries
> and bounded search share the checked scene/native allocation budgets.
> ``maximum_trajectory_dt`` fixes both native time bounds for this request:
> process feasibility and evidence stops take precedence over compression.
> ``interpolation_dt`` is the requested maximum sample spacing until time
> binding replaces it with the exact selected spacing for all consumers.

## `PlanningBudget.__post_init__`

[定义位置](../../../manipulation/planning.py#L159)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.trajectory_seeds
self.ik_seeds
self.knots
self.interpolation_steps
self.ik_iterations
self.trajectory_iterations
self.max_robot_spheres
self.max_robot_dof
self.max_robot_links
self.max_validation_points
self.maximum_trajectory_solves
self.max_task_anchors
self.max_scene_gpu_workspace_bytes
self.max_host_rss_bytes
self.max_device_used_bytes
np.isfinite
self.search_seconds
self.maximum_trajectory_dt
self.interpolation_dt
self.interpolation_buffer_size
```

写入目标：

```python
name
grid
maximum_count
native_count
```

直接调用符号：

```python
ValueError
type
np.isfinite
getattr
_derived_spline_grid
```

## `JointTrajectory`

[定义位置](../../../manipulation/planning.py#L212)。

```python
class JointTrajectory():
```

类字段：

```python
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

实现说明：

> A candidate cubic joint curve and its exact skill-progress mapping.
>
> All arrays are CPU NumPy arrays in named active-joint order.  The cubic
> reconstruction preserves the supplied q/dq polynomial, whether provided
> by a caller or exported from native knot subdivisions. Sampling never
> performs another fit or smoothing pass.

## `JointTrajectory.__post_init__`

[定义位置](../../../manipulation/planning.py#L236)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.times
self.position
self.velocity
```

写入目标：

```python
self._curve
```

直接调用符号：

```python
CubicHermiteSpline
```

## `JointTrajectory.duration`

[定义位置](../../../manipulation/planning.py#L240)。

输入签名：

```python
def duration(self):
```

显式输出分支：

```python
float(self.times[-1])
```

读取的对象属性：

```python
self.times
```

写入目标：

无显式条目。

直接调用符号：

```python
float
```

## `JointTrajectory.next_verification`

[定义位置](../../../manipulation/planning.py#L244)。

输入签名：

```python
def next_verification(self):
```

实现说明：

> The first original evidence barrier, without claiming it was met.

显式输出分支：

```python
deepcopy(min(barriers, key=lambda event: event['time'])) if barriers else None
```

读取的对象属性：

```python
self.events
self.satisfied_dependencies
self.progress_s
```

写入目标：

```python
barriers
event
```

直接调用符号：

```python
deepcopy
min
```

## `JointTrajectory.executable_until`

[定义位置](../../../manipulation/planning.py#L252)。

输入签名：

```python
def executable_until(self):
```

实现说明：

> Limit of this conditional plan before new actual feedback is needed.

显式输出分支：

```python
self.duration if barrier is None else float(barrier['time'])
```

读取的对象属性：

```python
self.next_verification
self.duration
```

写入目标：

```python
barrier
```

直接调用符号：

```python
float
```

## `JointTrajectory.sample`

[定义位置](../../../manipulation/planning.py#L257)。

输入签名：

```python
def sample(self, times):
```

实现说明：

> Return q, dq, ddq and progress at the same finite curve time.
>
> SciPy evaluates an internal breakpoint on its right-hand piece; the
> final endpoint uses the last piece's left limit.  Thus acceleration
> is the analytic, one-sided second derivative of this saved C1 curve,
> not an estimated derivative, a new smoothing pass, or actual qacc.
> Independent acceleration validation still checks both knot sides.

显式输出分支：

```python
(self._curve(query), self._curve(query, 1), self._curve(query, 2), np.interp(query / self.duration, self.progress_u, self.progress_s))
```

读取的对象属性：

```python
np.asarray
np.isfinite(query).all
np.isfinite
np.any
self.duration
self._curve
np.interp
self.progress_u
self.progress_s
```

写入目标：

```python
query
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L276)。

输入签名：

```python
def cubic_derivative_report(trajectory, acceleration_bounds, *, joint_names, until=None):
```

实现说明：

> Check each scalar cubic's acceleration range, including both knot sides.
>
> Acceleration is affine within a cubic piece, so its extrema occur at that
> piece's endpoints. An internal knot has two acceleration limits: evaluating
> only ``curve(knot, 2)`` would lose the previous piece's left limit. A supplied
> ``until`` clips the last piece at the original execution barrier; future
> pieces do not enter that prefix report. Limits are bound by joint identity
> and compared in binary64 without adding a numerical or physical allowance.
>
> The input remains the original CubicHermiteSpline. Its C1 construction and
> raw floating-point knot jumps are reported; acceleration jumps do not add a
> C2 requirement. This checks a reference polynomial, not actual robot motion,
> continuous collision, jerk at jumps, or actuator forces.

显式输出分支：

```python
{'scope': 'binary64 polynomial acceleration ranges and knot continuity; no actual-motion or collision certification', 'checked_interval_s': [0.0, stop], 'full_curve_duration_s': float(times[-1]), 'joint_names': list(curve_names), 'checked_piece_count': len(indices), 'acceleration_bounds': bounds.tolist(), 'acceleration': {'minimum': minimum.tolist(), 'maximum': maximum.tolist(), 'minimum_time_s': endpoint_times[low_indices].tolist(), 'maximum_time_s': endpoint_times[high_indices].tolist(), 'maximum_absolute': np.maximum(np.abs(minimum), np.abs(maximum)).tolist(), 'lower_excess': lower_excess.tolist(), 'upper_excess': upper_excess.tolist(), 'maximum_bound_excess': maximum_excess, 'bounds_satisfied': maximum_excess == 0.0, 'maximum_violation': witness}, 'continuity': continuity}
```

读取的对象属性：

```python
trajectory.joint_names
np.asarray
bounds.shape
np.isfinite(bounds).all
np.isfinite
np.any
names.index
trajectory.times
trajectory.position
trajectory.velocity
times.ndim
np.isfinite(times).all
np.diff
position.shape
velocity.shape
np.isfinite(position).all
np.isfinite(velocity).all
np.array_equal
curve.x
trajectory._curve.x
trajectory._curve
curve.c
trajectory._curve.c
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
excess.shape
jumps.items
np.isfinite(selected).all
(np.abs(selected).max(axis=0) if len(selected) else np.zeros(len(names))).tolist
np.abs(selected).max
np.abs
np.zeros
bounds.tolist
minimum.tolist
maximum.tolist
endpoint_times[low_indices].tolist
endpoint_times[high_indices].tolist
np.maximum(np.abs(minimum), np.abs(maximum)).tolist
lower_excess.tolist
upper_excess.tolist
```

写入目标：

```python
names
curve_names
bounds
name
times
position
velocity
curve
stop
indices
ends
widths
coefficients
acceleration
values
endpoint_times
columns
low_indices
high_indices
minimum
maximum
lower_excess
upper_excess
excess
maximum_excess
witness
bound_index
joint_index
endpoint
internal
c
h
q_end
dq_end
ddq_end
jumps
continuity
label
jump
selected
continuity[f'{label}_maximum_absolute_jump']
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L379)。

```python
class PlanningResult():
```

类字段：

```python
success: bool
trajectory: JointTrajectory | None
diagnostics: dict
execution_binding: dict | None = None
request_ledger: dict | None = None
```

## `_anchor_capacity`

[定义位置](../../../manipulation/planning.py#L392)。

输入签名：

```python
def _anchor_capacity(progress_map, anchors, budget):
```

实现说明：

> Admit semantic anchors into a fixed native workspace; never add witnesses.

显式输出分支：

```python
budget.max_task_anchors
```

读取的对象属性：

```python
progress_map.anchor_parameters
budget.max_task_anchors
```

写入目标：

```python
required
```

直接调用符号：

```python
len
progress_map.anchor_parameters
MotionPlanningError
```

## `_PlanningWorkspace`

[定义位置](../../../manipulation/planning.py#L402)。

```python
class _PlanningWorkspace():
```

类字段：

```python
cpu_resident_bytes: int
cpu_pending_shared_bytes: int
cpu_live_bytes: int
gpu_bytes: int
anchor_fk_bytes: int
```

实现说明：

> Resident geometry, pending shared arrays and one simultaneous query.

## `_PlanningWorkspace.cpu_bytes`

[定义位置](../../../manipulation/planning.py#L412)。

输入签名：

```python
def cpu_bytes(self):
```

显式输出分支：

```python
self.cpu_resident_bytes + self.pending_cpu_bytes
```

读取的对象属性：

```python
self.cpu_resident_bytes
self.pending_cpu_bytes
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `_PlanningWorkspace.pending_cpu_bytes`

[定义位置](../../../manipulation/planning.py#L416)。

输入签名：

```python
def pending_cpu_bytes(self):
```

实现说明：

> Additional scene peak; resident arrays already appear in RSS.

显式输出分支：

```python
self.cpu_pending_shared_bytes + self.cpu_live_bytes
```

读取的对象属性：

```python
self.cpu_pending_shared_bytes
self.cpu_live_bytes
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `_PlanningWorkspace.total_bytes`

[定义位置](../../../manipulation/planning.py#L421)。

输入签名：

```python
def total_bytes(self):
```

实现说明：

> Diagnostic scene total; it is never a CUDA allocation request.

显式输出分支：

```python
self.cpu_bytes + self.gpu_bytes
```

读取的对象属性：

```python
self.cpu_bytes
self.gpu_bytes
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `_planning_workspace_bytes`

[定义位置](../../../manipulation/planning.py#L428)。

输入签名：

```python
def _planning_workspace_bytes(scene, robot_metadata, budget, anchor_capacity):
```

实现说明：

> Reserve one CPU scene query and the native IK point-query transfer.
>
> Trajectory seeds run serially. Task/geometry arrays coexist for one
> candidate; accepted-curve geometry uses its own bounded query. Immutable
> fields are shared. SLSQP's dense constraint/work arrays are admitted once
> their actual row count is known by unified_solver.solve_trajectory.

显式输出分支：

```python
_PlanningWorkspace(int(trajectory['cpu_resident']), int(trajectory['cpu_pending_shared']), int(trajectory['cpu_live'] + pin_bytes), gpu_bytes, 0)
```

读取的对象属性：

```python
budget.knots
budget.interpolation_steps
budget.max_validation_points
scene.required_progress_anchors
scene.estimated_workspace_allocation
budget.ik_seeds
budget.max_scene_gpu_workspace_bytes
scene.robot_model.joint_names
scene.robot_model
```

写入目标：

```python
ordinary
states
mode_count
trajectory
point
gpu_bytes
dof
links
full_dof
pin_bytes
```

直接调用符号：

```python
max
scene.estimated_workspace_allocation
int
ResourceLimitError
len
_PlanningWorkspace
```

## `_require_planning_host_memory`

[定义位置](../../../manipulation/planning.py#L456)。

输入签名：

```python
def _require_planning_host_memory(pending_cpu_bytes, *, encoded_record_bytes_upper, max_host_rss_bytes):
```

实现说明：

> Reserve new CPU work and candidate records.
>
> Three encoded payloads conservatively cover the new validation record,
> its candidate snapshot and serialization copy. Earlier candidates and
> resident source geometry already appear in measured RSS. Rechecking at
> each export therefore accounts for accumulated history without charging
> the same existing map or previous record as a fresh allocation.

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
require_host_memory
```

## `_require_planning_memory`

[定义位置](../../../manipulation/planning.py#L472)。

输入签名：

```python
def _require_planning_memory(torch, device, workspace, *, max_device_used_bytes):
```

实现说明：

> Admit Torch buffers without promising cached memory to native allocators.
>
> CollisionBuffer and native FK buffers use torch.zeros. Unused capacity in
> this process's Torch allocator can therefore serve those tensors after a
> schedule replacement. Existing Mapper fields are already allocated. The
> separate two-GiB allowance covers main FK, general pose/sphere transform
> graphs, solver/model and trajectory storage; it must still fit driver-
> reported free memory independently of the geometry-query tensor cache.
>
> CPU source geometry is already accounted for by host admission.

显式输出分支：

```python
workspace
```

读取的对象属性：

```python
workspace.gpu_bytes
workspace.anchor_fk_bytes
```

写入目标：

无显式条目。

直接调用符号：

```python
require_device_memory
```

## `_bounded_robot_configuration`

[定义位置](../../../manipulation/planning.py#L493)。

输入签名：

```python
def _bounded_robot_configuration(robot, problem, scene, budget):
```

实现说明：

> Admit a pure named-joint tree before native allocation.
>
> The native solvers own joint-domain constraints and tool FK. Registered
> observation fields own all entity collision geometry in the scene.

显式输出分支：

```python
(data, {'active_joint_names': sorted(joints - set(locks)), 'unlocked_dof': len(joints), 'links': len(links), 'selected_link_names': sorted(links), 'spheres': 0})
```

读取的对象属性：

```python
path.stat().st_size
path.stat
row.attrib
tree.findall
kinematics.get
budget.max_robot_links
budget.max_robot_dof
scene.required_robot_links
problem.tool_link
problem.joint_names
kinematics.update
```

写入目标：

```python
data
kinematics
path
tree
links
row
joints
locks
```

直接调用符号：

```python
deepcopy
isinstance
load_yaml
str
Path
path.stat
ValueError
pure_kinematics
tree.findall
kinematics.get
len
set
kinematics.update
sorted
```

## `_ProgressMap`

[定义位置](../../../manipulation/planning.py#L529)。

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

[定义位置](../../../manipulation/planning.py#L538)。

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

[定义位置](../../../manipulation/planning.py#L546)。

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
np.flatnonzero
self.s
pairs.append
self.u
np.searchsorted
np.asarray(sorted(set(pairs)), dtype=float).reshape
np.asarray
np.unique
pairs[:, column].astype
np.float32
```

写入目标：

```python
pairs
value
equal
right
fraction
column
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
any
np.unique
pairs[:, column].astype
```

## `_phase_aligned_parameters`

[定义位置](../../../manipulation/planning.py#L572)。

输入签名：

```python
def _phase_aligned_parameters(source, target, parameters):
```

实现说明：

> Map time parameters through equal progress, retaining a pause's phase.
>
> A progress value with a finite source plateau is not a single time: its
> interior motion must keep its relative position between both endpoints.
> An isolated source value uses the target's first arrival, so an added
> native verification hold cannot postpone its original arrival sample.

显式输出分支：

```python
np.asarray(result)
```

读取的对象属性：

```python
np.asarray
np.interp
source.u
source.s
source.anchor_parameters
target.anchor_parameters
result.append
```

写入目标：

```python
parameters
values
result
parameter
value
old
new
phase
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L596)。

输入签名：

```python
def _candidate_sample_times(trajectory, budget):
```

实现说明：

> Keep every original cubic piece and bound its query spacing/storage.

显式输出分支：

```python
np.r_[np.concatenate([np.linspace(left, right, count + 1)[:-1] for (left, right, count) in zip(trajectory.times[:-1], trajectory.times[1:], counts)]), trajectory.duration]
```

读取的对象属性：

```python
np.maximum
np.ceil(np.diff(trajectory.times) / float(np.float32(budget.interpolation_dt))).astype
np.ceil
np.diff
trajectory.times
np.float32
budget.interpolation_dt
counts.sum
budget.max_validation_points
budget.interpolation_buffer_size
np.r_
np.concatenate
np.linspace
trajectory.duration
```

写入目标：

```python
counts
count
left
right
```

直接调用符号：

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

## `_reference_clock_issues`

[定义位置](../../../manipulation/planning.py#L606)。

输入签名：

```python
def _reference_clock_issues(problem, trajectory, timestep):
```

实现说明：

> Report required device occurrences outside the unchanged curve domain.
>
> The ordinary strict quantizer remains authoritative, including a source
> endpoint just beyond an exact tick in binary64. An unrepresented future
> occurrence is neither clipped, inferred from the endpoint, nor repaired
> by silently extending or retiming the supplied polynomial.

显式输出分支：

```python
issues
```

读取的对象属性：

```python
trajectory.progress_u
trajectory.progress_s
problem.events
event.kind
event.requires_evidence
event.stop
mapping.anchor_parameters
event.progress
trajectory.duration
issues.append
event.event_id
event.skill_id
event.occurrence
```

写入目标：

```python
mapping
issues
event
command
intent
step
applied
```

直接调用符号：

```python
_ProgressMap
MotionPlanningError
float
mapping.anchor_parameters
_activation
issues.append
```

## `_initial_timing`

[定义位置](../../../manipulation/planning.py#L636)。

输入签名：

```python
def _initial_timing(problem, budget):
```

实现说明：

> Quantize source duration, then sample its exact native clock densely.
>
> Source duration/rhythm are soft initialization information. The native
> interval cap bounds one request; an overlong demonstration may therefore
> propose a compressed clock. ``interpolation_dt`` bounds sample spacing at
> admission; the returned budget records the exact selected spacing. Hard
> event deadlines remain separate. Later refinement retains this clock.

显式输出分支：

```python
(selected, _ProgressMap((seconds - seconds[0]) / source_duration, progress.copy()))
```

读取的对象属性：

```python
problem.source_time_map
np.asarray
seconds.ndim
seconds.shape
progress.shape
np.isfinite(seconds).all
np.isfinite
np.any
np.diff
budget.knots
budget.maximum_trajectory_dt
budget.interpolation_steps
budget.interpolation_dt
np.float32
np.nextafter
np.inf
(budget.interpolation_steps - 1).bit_length
progress.copy
```

写入目标：

```python
source
seconds
progress
source_duration
maximum
proposed_dt
native_dt
_
knot_dt
actual
selected
```

直接调用符号：

```python
np.asarray
len
np.isfinite(seconds).all
np.isfinite
np.any
np.diff
ValueError
float
_derived_spline_grid
min
np.float32
np.nextafter
range
(budget.interpolation_steps - 1).bit_length
replace
_ProgressMap
progress.copy
```

## `_bind_event_clock`

[定义位置](../../../manipulation/planning.py#L685)。

输入签名：

```python
def _bind_event_clock(problem, progress_map, duration, timestep=None):
```

显式输出分支：

```python
_ProgressMap(mapping.u, mapping.s, source)
```

读取的对象属性：

```python
problem.events
progress_map.anchor_parameters
event.progress
event.kind
event.requires_evidence
event.stop
tick_entries.append
source[identity].update
entry[0].kind
by_step.setdefault(entry[1], []).append
by_step.setdefault
goal.progress
by_step.values
item[0].source_frame
item.check_target
event.timing
superseded.add
command_keys.append
tick_progress.add
np.asarray
np.any
np.diff
np.union1d
progress_map.u
np.interp
progress_map.s
mapping.anchor_parameters
record.update
mapping.u
mapping.s
```

写入目标：

```python
source
command_keys
tick_progress
tick_entries
event
identity
parameters
intent
command
needs_tick
step
applied
source[identity]
by_step
entry
superseded
true_goals
name
goal
entries
ordered
_
item
key
source[key]['scheduled_superseded_by']
before_u
after_u
independent
fixed
value
moves
before
after
keys
mapping
record
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L763)。

输入签名：

```python
def _validate_bound_event_times(problem, progress_map, duration, origin_s=0.0, history=(), goal_identities=None, timing_events=()):
```

实现说明：

> Check explicit windows against actual history or this tick-bound proposal.
>
> Consumed nodes may still own an unresolved time relation. Their recorded
> witness survives while a future origin is supplied by this candidate.
> Only the latest selected occurrence of an original event owns the current
> completion obligation; older attempts remain unchanged in the ledger.

显式输出分支：

```python
checks
```

读取的对象属性：

```python
row.get
problem.events
all_events.update
latest.get
current.items
progress_map.event_clock
record.get
goal.progress
event.progress
progress_map.anchor_parameters
condition.get
event.timing
constraint.check_target
constraint.goal_id
goal_identities[collection].index
constraint.condition_id
all_events.items
checks.append
planned.get
```

写入目标：

```python
goal_identities
name
history
row
current
event
all_events
latest
key
latest[key[:2]]
predicted
planned
checks
record
collection
index
goal
condition
constraint
value
original_index
identity
planned[key, constraint.check_target, constraint.goal_id, constraint.condition_id]
facts
earliest
latest_time
applicable
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L783)。

输入签名：

```python
def past_pose_timing(constraint):
```

显式输出分支：

```python
any((identity and identity[0] in ('pose_events', 'supports', 'coordinate_events') and (identity[1] not in goal_identities[identity[0]]) for identity in identities))
```

读取的对象属性：

```python
constraint.check_target
constraint.goal_id
(constraint.origin_event or {}).get
constraint.origin_event
```

写入目标：

```python
identities
identity
```

直接调用符号：

```python
(constraint.origin_event or {}).get
any
```

## `_validate_bound_event_times.candidate_fact`

[定义位置](../../../manipulation/planning.py#L797)。

输入签名：

```python
def candidate_fact(key, target, value, **identity):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
predicted.append
```

写入目标：

无显式条目。

直接调用符号：

```python
predicted.append
```

## `ContinuationBindingError`

[定义位置](../../../manipulation/planning.py#L880)。

```python
class ContinuationBindingError(ValueError):
```

类字段：

无显式条目。

实现说明：

> Remaining work lacks facts, budget or a compatible original clock.

## `_obligation_contract`

[定义位置](../../../manipulation/planning.py#L890)。

输入签名：

```python
def _obligation_contract(problem):
```

实现说明：

> Keep original semantic identities and thresholds, allowing fresh geometry.
>
> Current poses, attachment transforms and observed surfaces are rebindable
> inputs. A new observation cannot erase a goal/process, reorder its original
> index, or relax a declared functional threshold under the same request ID.

显式输出分支：

```python
contract
```

读取的对象属性：

```python
columns.items
type(item).__name__
item.actual_relation
item.actual_relation.items
key.startswith
item.model.kind
item.model
phase.reference.source
phase.reference
phase.metadata.get
phase.metadata
```

写入目标：

```python
columns
contract
collection
names
contract[collection]
name
key
value
item
contract['conditions']
row
contract['source_time_map']
contract['source_phases']
phase
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L931)。

输入签名：

```python
def _restore_original_direction_origins(problem, contract):
```

实现说明：

> Retain actual net-change baselines without freezing fresh mechanism poses.
>
> A compiler can derive an entrance baseline from its freshly observed
> initial coordinate. That is correct only for a new request. For the same
> request, recover the original evidenced baseline when every direction
> instruction is unchanged. An explicitly different baseline is rejected
> by the semantic contract instead of silently accepted as a new goal.

显式输出分支：

```python
None
```

读取的对象属性：

```python
problem.mechanisms
mechanism.directions
row.items
direction.start
mechanism.start
direction.origin_coordinate
mechanism.initial_coordinates
direction.coordinate_index
direction.origin_source
mechanism.source
direction.kind
directions.append
```

写入目标：

```python
saved_mechanisms
mechanism
saved
directions
direction
original
current
semantics
name
value
automatic
mechanism.directions
```

直接调用符号：

```python
len
getattr
zip
plain
row.items
float
semantics
replace
deepcopy
directions.append
tuple
```

## `PlanningWorkExhausted`

[定义位置](../../../manipulation/planning.py#L964)。

```python
class PlanningWorkExhausted(ValueError):
```

类字段：

无显式条目。

实现说明：

> The original request has no remaining admitted computation budget.

## `_planning_work_budget`

[定义位置](../../../manipulation/planning.py#L973)。

输入签名：

```python
def _planning_work_budget(values):
```

实现说明：

> Retain every work, geometry, timing and numerical field for comparison.

显式输出分支：

```python
{name: value for (name, value) in values.items() if name not in _ALLOCATION_BUDGET_FIELDS}
```

读取的对象属性：

```python
values.items
```

写入目标：

```python
name
value
```

直接调用符号：

```python
values.items
```

## `_request_work_ledger`

[定义位置](../../../manipulation/planning.py#L979)。

输入签名：

```python
def _request_work_ledger(problem, budget, previous=None, continuation=None):
```

实现说明：

> Copy an append-only request record, never mutate a caller's failed result.

显式输出分支：

```python
ledger
```

读取的对象属性：

```python
uuid4().hex
dict.fromkeys
problem.events
ledger.get
consumed.get
np.isfinite
ledger['binding_records'].append
continuation.get
ledger[name].append
ledger['event_records'].append
ledger['effects'].update
```

写入目标：

```python
ledger
requested_budget
recorded_budget
consumed
name
value
ledger['planning_budget']
original
old
row
record
ledger['requires_continuation']
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1057)。

输入签名：

```python
def _check_work_budget(budget, consumed, *, started, initial_seconds, stage=None):
```

实现说明：

> Check at stage boundaries; report real elapsed overrun, never pretend preemption.

显式输出分支：

无显式条目。

读取的对象属性：

```python
time.monotonic
budget.maximum_trajectory_solves
budget.planning_seconds
budget.max_validation_points
budget.ik_seeds
budget.trajectory_seeds
stage_caps.items
```

写入目标：

```python
consumed['planning_seconds']
cap
stage_caps
name
limit
```

直接调用符号：

```python
time.monotonic
PlanningWorkExhausted
stage_caps.items
```

## `_prepare_continuation`

[定义位置](../../../manipulation/planning.py#L1075)。

输入签名：

```python
def _prepare_continuation(problem, scene, budget, continuation, *, rebind=None):
```

实现说明：

> Validate a fresh actual/scene binding without replaying consumed events.
>
> A continuation is the executor's existing result record. It is not a
> request to restore the old actual state or guess past task success. The
> original absolute progress, event identities and cumulative solve budget
> survive. Nominal pose/support/coordinate anchors leave the suffix by
> progress; physical task effects remain separate independent observations.

显式输出分支：

```python
(remaining, {'original_duration': float(duration), 'elapsed': float(elapsed), 'actual_elapsed': float(actual_elapsed), 'elapsed_device_steps': confirmed_steps, 'device_timestep_s': float(timestep), 'u': original_map.u.copy(), 's': original_map.s.copy(), 'events': {identity: deepcopy(old_occurrences[identity]) for identity in pending_ids}, 'request_budget': asdict(request_budget), 'consumed_planning': consumed, 'physics_budget': physics_cap, 'consumed_physics_steps': physics_consumed, 'remaining_physics_steps': physics_remaining, 'request_events': request_events, 'consumed_events': sorted(consumed_events), 'goal_identities': goal_identities, 'retry_goal_ids': sorted(retry_goals), 'direction_identities': direction_identities, 'retry_direction_ids': sorted(retry_directions), 'completed_direction_ids': sorted(completed_directions), 'rebind': deepcopy(rebind), 'scene_anchors': np.asarray(scene.required_progress_anchors, float).copy() if rebind is not None else None, 'control_timestep_s': timestep if rebind is not None else None, 'history': deepcopy(continuation.get('history', ())), 'timing_events': deepcopy(continuation.get('request_ledger', original['request_ledger'])['timing_events']), 'device_policy': deepcopy(continuation.get('device_policy')), 'active_finger_interval': deepcopy(continuation.get('active_finger_interval')), 'completed_goal_ids': [], 'effects': deepcopy(continuation.get('effects', {})), 'previous_request_time_origin_s': float(original.get('request_time_origin_s', 0.0)), 'request_time_origin_s': float(original.get('request_time_origin_s', 0.0)) + float(elapsed), 'request_time_origin_step': int(original.get('request_time_origin_step', 0)) + local_steps})
```

读取的对象属性：

```python
rebind.get
continuation.get
original.get
consumed.get
np.isfinite
original['consumed_planning'].get
feedback.get
problem.joint_names
names.index
np.asarray
np.array_equal
problem.start_q
problem.start_dq
scene.gripper
hand.control_timestep_s
hand.joint_names
hand.current_positions
curve.get
np.isfinite([duration, elapsed]).all
np.interp
original_map.u
original_map.s
problem.end_progress
event.skill_id
event.event_id
problem.events
current_events.items
current.pop
source.pop
applied.add
current_events.get
event.kind
event.requires_evidence
event.stop
pending_ids.discard
row.get
problem.mechanisms
mechanism.directions
problem.mechanisms[identity[0]].directions
row['source'].strip
direction.kind
direction.origin_coordinate
direction.origin_source
interval.shape
np.isfinite(interval).all
completed_directions.update
event.progress
occurrences.extend
goal.progress
selected_events.values
getattr(problem, name)[index].progress
problem.mechanisms[index].directions[column].start
problem.mechanisms[index].directions
problem.source_progress
requirement.directions
direction.end
direction.start
requirement.end
requirement.angular_column
requirement.start
requirement.model.diagnostics.get
requirement.model.diagnostics
requirement.model
item.progress
item.directions
item.end
rebound.mechanisms
mechanism.angular_column
mechanism.start
mechanism.end
mechanism.model.diagnostics.get
mechanism.model.diagnostics
mechanism.model
problem.progress_to_source_time
np.r_
problem.source_timestamps
rebound.__post_init__
hand.commands
rebound.events
policies.get
commands.append
event.occurrence
item.init
item.name
original_map.u.copy
original_map.s.copy
np.asarray(scene.required_progress_anchors, float).copy
scene.required_progress_anchors
```

写入目标：

```python
original
previous_history
history
request_budget
consumed
key
value
physics_cap
physics_consumed
physics_remaining
local_steps
confirmed_steps
accounting_exact
timestep
remaining
feedback
names
columns
name
actual_q
actual_dq
hand
curve
duration
actual_elapsed
elapsed
old_u
old_s
original_map
curve_elapsed
start_progress
request_events
source_events
row
current_events
event
identity
current
source
remaining_rows
pending_ids
old_occurrences
old_rows
applied
evaluated
stopped
reached
witnessed
consumed_events
due
all_goals
index
all_directions
mechanism
column
completed_directions
claimed_directions
direction
rows
declared_net_origin
verified
interval
scope_start
scope_end
scope_covered
retry_ids
retry_goals
retry_directions
selected_events
selected_events[identity[:2]]
selected_events[event.skill_id, event.event_id]
occurrences
selected_events[identity]
remaining_goal_ids
goal
required_progress
requirement
goal_identities
item
collection
direction_identities
rebound
rebound.start_progress
rebound.events
mechanism.start
source_time
selected
rebound.source_timestamps
rebound.source_progress
policies
commands
policy
hand.commands
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1429)。

输入签名：

```python
def _continuation_timing(problem, budget, continuation):
```

实现说明：

> Use the original suffix, or rebuild every clock consumer for an explicit rebind.

显式输出分支：

```python
(selected, mapping)
(selected, _ProgressMap(seconds / grid['duration'], progress, clock))
```

读取的对象属性：

```python
continuation.get
np.unique
np.r_
problem.anchors
problem.start_progress
selected.knots
problem.stop_progress
selected.maximum_trajectory_dt
selected.interpolation_steps
selected.interpolation_dt
mapping.event_clock.values
mapping.event_clock
record.get
np.array
problem.end_progress
continuation['events'].items
event.get
record.update
problem.events
```

写入目标：

```python
selected
mapping
anchors
_
grid
record
record['request_intent_time_s']
record['request_time_origin_step']
record['request_activation_step']
record['request_activation_time_s']
remaining
proposal
original_times
keep
seconds
progress
clock
identity
event
name
record[name]
timestep
step
record['time']
record['intent_time_s']
clock[identity]
record['event']
```

直接调用符号：

```python
continuation.get
_initial_timing
np.unique
reference_progress_clock
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

## `reference_progress_clock`

[定义位置](../../../manipulation/planning.py#L1506)。

输入签名：

```python
def reference_progress_clock(progress_map, knots, *, stop_progress, anchors):
```

实现说明：

> Bind the fixed reference rhythm and its original-path parameter.
>
> For normalized physical time u=t/T, z(u) selects a time parameter in
> the supplied reference; task progress is old_s(z(u)). This scalar
> B-spline is an explicit scheduling prior, independent of robot q(t).
> Its retained endpoint interval and declared event stops preserve the
> existing event clock. Robot terminal position, velocity and acceleration
> remain free unless explicit task rows constrain them.
>
> Four equal scalar coefficients give a constant reference interval.
> Every source time vertex and mandatory progress anchor is retained,
> including motion inside original progress plateaus. Joint initializers
> consume the same z; reference poses only propose their initial samples.
>
> The saved schedule is piecewise linear. For the scalar uniform cubic,
> |d2z/d(native_t)2| <= max|diff(coefficients, 2)|; L subdivisions bound
> its linearization error by D2/(8*L**2). The 1e-6 bound is dimensionless
> reference-parameter error. It does not change any physical task bound.

显式输出分支：

```python
(_ProgressMap(warp.u.copy(), progress), warp)
```

读取的对象属性：

```python
np.unique
np.asarray
progress_map.anchor_parameters
np.r_
np.diff
np.floor(shares).astype
np.floor
np.argsort
np.floor(shares).sum
np.array
np.arange
np.linspace
free.extend
np.ones
np.abs(np.diff(coefficients, n=2)).max
np.abs
np.ceil
np.sqrt
np.all
progress_map.u
warp.anchor_parameters
np.isin
warp.u.astype
warp.u
np.float32
inverse_pairs[:, 0].astype
np.interp
warp.s
np.searchsorted
np.abs(np.interp(warp.u, new_u, parameter) - warp.s).max
np.abs(np.interp(new_u, warp.u, warp.s) - parameter).max
progress_map.s
supplied.update
supplied.get
warp.u.copy
```

写入目标：

```python
stop_progress
stop_pairs
stops
value
boundaries
count
remaining
shares
increments
order
increments[order[:remaining - int(np.floor(shares).sum())]]
initial_rate
prefix
free
index
left
right
steps
segment
segment[-1]
coefficients
intervals
spline
second_derivative_bound
subdivisions
grid_count
native_t
parameter
interval
local
parameter[interval * subdivisions:(interval + 1) * subdivisions + 1]
parameter[0]
parameter[-1]
warp
source_pairs
retained_parameters
inverse_pairs
auxiliary
new_u
parameter[np.searchsorted(new_u, inverse_pairs[:, 0])]
perturbation
progress
supplied
current
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1611)。

输入签名：

```python
def _torch_interpolate(keys, values, query):
```

实现说明：

> Small tensor interpolation for the fixed progress schedule only.

显式输出分支：

```python
values[right - 1] + alpha * (values[right] - values[right - 1])
```

读取的对象属性：

```python
torch.searchsorted(keys, query.contiguous(), right=True).clamp
torch.searchsorted
query.contiguous
((query - keys[right - 1]) / (keys[right] - keys[right - 1])).clamp
```

写入目标：

```python
right
alpha
```

直接调用符号：

```python
torch.searchsorted(keys, query.contiguous(), right=True).clamp
torch.searchsorted
query.contiguous
len
((query - keys[right - 1]) / (keys[right] - keys[right - 1])).clamp
```

## `_projection_tolerance`

[定义位置](../../../manipulation/planning.py#L1623)。

输入签名：

```python
def _projection_tolerance(problem):
```

实现说明：

> Set native projection accuracy without changing equalities or acceptance.

显式输出分支：

```python
min(bounds) * 0.1
```

读取的对象属性：

```python
problem.numerical.position_m
problem.numerical
problem.numerical.rotation_rad
problem.numerical.joint
problem.mechanisms
relation.model
bounds.extend
model.position_tolerance
model.rotation_tolerance
```

写入目标：

```python
bounds
relation
model
```

直接调用符号：

```python
bounds.extend
min
```

## `_ompl_terms_accepted`

[定义位置](../../../manipulation/planning.py#L1634)。

输入签名：

```python
def _ompl_terms_accepted(terms):
```

实现说明：

> Keep every existing FP comparison and synchronize one boolean only.
>
> OMPL constructs all term tensors before acceptance. They share the state
> device; collecting their finite/nonpositive flags changes no floating
> operation, sum, tolerance, or report value. Boolean AND is exact, unlike
> reassociating a floating-point reduction. The empty collection keeps
> Python ``all``'s identity, although production also supplies scene/task
> terms. Only the final scalar crosses the device/host boundary.

显式输出分支：

```python
bool(torch.stack(flags).all().item()) if flags else True
```

读取的对象属性：

```python
torch.isfinite(term).all
torch.isfinite
(term <= 0).all
terms.values
torch.stack(flags).all().item
torch.stack(flags).all
torch.stack
```

写入目标：

```python
flags
term
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1650)。

输入签名：

```python
def _ompl_stage(start, goal, lower, upper, valid, equality, *, tolerance, step, seconds, sample_limit, query_parameters=()):
```

实现说明：

> Run native directed RRTConnect in (q,u=t/T), retaining checked edge samples.
>
> ``equality(x, derivative)`` returns true equality residuals or their
> Jacobian.  An empty residual selects ordinary OMPL space; inequality
> bands are checked by ``valid`` and are never collapsed to center lines.
> The same exporter accepts edges during tree growth and reconstructs the
> solution, including the native geodesic's final delta-length tail.

显式输出分支：

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
(None, {**report, 'failure': 'whole_search_time_budget_exhausted'})
```

读取的对象属性：

```python
np.unique
np.asarray
np.isfinite(query_parameters).all
np.isfinite
np.any
time.monotonic
np.isfinite(jacobian).all
np.linalg.matrix_rank
np.linalg
ob.RealVectorStateSpace
ob.RealVectorBounds
bounds.setLow
bounds.setHigh
ambient.setBounds
ob.ProjectedStateSpace
space.setDelta
ob.ConstrainedSpaceInformation
ob.SpaceInformation
information.setStateValidityCheckingResolution
np.linalg.norm
information.setStateValidityChecker
information.setup
information.getMotionValidator
information.setMotionValidator
space.allocState
a.copy
start.tolist
b.copy
goal.tolist
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
np.concatenate
np.diff
np.array_equal
```

写入目标：

```python
dimension
query_parameters
deadline
residual
count
report
constraint
endpoint
jacobian
ambient
bounds
index
lo
hi
space
information
native_validator
accepted_edges
a
b
a[index]
b[index]
report['initialization_seconds']
label
value
failure
report['motion_checks']
direct
report['direct_connection_accepted']
definition
search
path
segments
i
segment
samples
```

直接调用符号：

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
valid_array
vector
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

[定义位置](../../../manipulation/planning.py#L1675)。

```python
class ProcessConstraint(ob.Constraint):
```

类字段：

无显式条目。

## `_ompl_stage.ProcessConstraint.__init__`

[定义位置](../../../manipulation/planning.py#L1676)。

输入签名：

```python
def __init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
```

写入目标：

无显式条目。

直接调用符号：

```python
super().__init__
super
```

## `_ompl_stage.ProcessConstraint.function`

[定义位置](../../../manipulation/planning.py#L1679)。

输入签名：

```python
def function(self, values, output):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
```

写入目标：

```python
output[:]
```

直接调用符号：

```python
equality
np.asarray
```

## `_ompl_stage.ProcessConstraint.jacobian`

[定义位置](../../../manipulation/planning.py#L1682)。

输入签名：

```python
def jacobian(self, values, output):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
```

写入目标：

```python
output[:]
```

直接调用符号：

```python
equality
np.asarray
```

## `_ompl_stage.vector`

[定义位置](../../../manipulation/planning.py#L1707)。

输入签名：

```python
def vector(state):
```

显式输出分支：

```python
np.asarray([state[index] for index in range(dimension)], dtype=float)
```

读取的对象属性：

```python
np.asarray
```

写入目标：

```python
index
```

直接调用符号：

```python
np.asarray
range
```

## `_ompl_stage.valid_array`

[定义位置](../../../manipulation/planning.py#L1710)。

输入签名：

```python
def valid_array(value):
```

显式输出分支：

```python
False
True
```

读取的对象属性：

```python
time.monotonic
np.isfinite(value).all
np.isfinite
np.any
constraint.isSatisfied
np.asarray
np.linalg.norm
np.linalg
np.max
np.abs
report['rejections'].get
```

写入目标：

```python
report['validity_checks']
reason
remaining
report['first_equality_rejection']
report['last_rejection']
report['rejections'][reason]
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1741)。

输入签名：

```python
def edge_key(first, last):
```

显式输出分支：

```python
(first.tobytes(), last.tobytes())
```

读取的对象属性：

```python
first.tobytes
last.tobytes
```

写入目标：

无显式条目。

直接调用符号：

```python
first.tobytes
last.tobytes
```

## `_ompl_stage.checked_edge`

[定义位置](../../../manipulation/planning.py#L1744)。

输入签名：

```python
def checked_edge(a, b):
```

显式输出分支：

```python
accepted_edges[key]
None
None
None
None
None
None
None
None
None
samples
```

读取的对象属性：

```python
np.array_equal
native_validator.checkMotion
og.PathGeometric
path.append
path.interpolate
np.asarray
path.getState
path.getStateCount
np.ceil
np.linalg.norm
np.linalg
np.arange
np.unique
np.r_
right.copy
checked.append
accepted_edges.pop
```

写入目标：

```python
first
last
key
path
vertices
i
checked
left
right
subdivisions
fractions
required
fraction
point
samples
accepted_edges[key]
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1800)。

```python
class ForwardValidator(ob.MotionValidator):
```

类字段：

无显式条目。

## `_ompl_stage.ForwardValidator.__init__`

[定义位置](../../../manipulation/planning.py#L1801)。

输入签名：

```python
def __init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
```

写入目标：

无显式条目。

直接调用符号：

```python
super().__init__
super
```

## `_ompl_stage.ForwardValidator.checkMotion`

[定义位置](../../../manipulation/planning.py#L1804)。

输入签名：

```python
def checkMotion(self, a, b, *unused):
```

显式输出分支：

```python
checked_edge(a, b) is not None
```

读取的对象属性：

无显式条目。

写入目标：

```python
report['motion_checks']
```

直接调用符号：

```python
checked_edge
```

## `_independent_projection`

[定义位置](../../../manipulation/planning.py#L1865)。

输入签名：

```python
def _independent_projection(equality, witness):
```

实现说明：

> Select independent local equations without dropping their validity test.
>
> OMPL's fixed-codimension adapter cannot store inactive zero rows or
> duplicate equations. Pivoting selects coordinates for its Newton step;
> the complete original relations still decide state and edge acceptance.

显式输出分支：

```python
(equality, 0)
(None, 0)
(None, 0)
(lambda value, derivative: equality(value, derivative)[rows], rank)
```

读取的对象属性：

```python
np.asarray
np.isfinite(jacobian).all
np.isfinite
jacobian.T
np.linalg.matrix_rank
np.linalg
np.sort
```

写入目标：

```python
jacobian
_
triangular
order
rank
rows
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1888)。

输入签名：

```python
def _project_allowed_boundary(value, residual, process_count, tolerance, lower, upper, *, deadline):
```

实现说明：

> Use native OMPL Newton projection into true boundary allowed sets.
>
> The callback concatenates signed process equations and nonnegative goal
> violations. It adds no nominal target or equation for a free direction.
> Independent rows are refreshed between bounded native Newton iterations,
> so a satisfied band contributes no fictitious equality. Time is fixed.
> Physical bounds and the original goal thresholds are checked unchanged.
>
> A reflected native step is tested for strict membership when the native
> step remains outside. For an active smooth inequality, J*d=-g makes the
> linearized value at x+2*d equal -g rather than zero. This can cross a
> floating-point boundary plateau without changing any physical tolerance.
> Reflection is only a proposal: every original relation must accept it.

显式输出分支：

```python
None
None
None
point if np.all(point >= lower) and np.all(point <= upper) else None
None
None
None
point
reflected
None
```

读取的对象属性：

```python
np.asarray(value, float).copy
np.asarray
np.isfinite(point).all
np.isfinite
np.any
time.monotonic
np.isfinite(error).all
np.all
np.abs
np.isfinite(jacobian).all
jacobian[:, :-1].T
np.linalg.matrix_rank
np.linalg
np.sort
point.copy
constraint.project
```

写入目标：

```python
point
dimension
_
error
jacobian
triangular
order
rank
rows
constraint
previous
point[-1]
reflected
reflected[-1]
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1911)。

输入签名：

```python
def allowed(candidate):
```

显式输出分支：

```python
False
time.monotonic() < deadline and np.isfinite(error).all() and np.all(np.abs(error[:process_count]) <= tolerance) and np.all(error[process_count:] <= 0)
```

读取的对象属性：

```python
time.monotonic
np.isfinite(candidate).all
np.isfinite
np.any
np.asarray
np.isfinite(error).all
np.all
np.abs
```

写入目标：

```python
error
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1944)。

```python
class AllowedBoundary(ob.Constraint):
```

类字段：

无显式条目。

## `_project_allowed_boundary.AllowedBoundary.__init__`

[定义位置](../../../manipulation/planning.py#L1945)。

输入签名：

```python
def __init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
np.finfo(float).tiny
np.finfo
self.setMaxIterations
```

写入目标：

无显式条目。

直接调用符号：

```python
super().__init__
super
np.finfo
self.setMaxIterations
```

## `_project_allowed_boundary.AllowedBoundary.function`

[定义位置](../../../manipulation/planning.py#L1955)。

输入签名：

```python
def function(self, values, output):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
```

写入目标：

```python
output[:]
```

直接调用符号：

```python
residual
np.asarray
```

## `_project_allowed_boundary.AllowedBoundary.jacobian`

[定义位置](../../../manipulation/planning.py#L1958)。

输入签名：

```python
def jacobian(self, values, output):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
residual(np.asarray(values), True)[rows].copy
np.asarray
```

写入目标：

```python
result
result[:, -1]
output[:]
```

直接调用符号：

```python
residual(np.asarray(values), True)[rows].copy
residual
np.asarray
```

## `_project_boundary`

[定义位置](../../../manipulation/planning.py#L1984)。

输入签名：

```python
def _project_boundary(value, equality, tolerance, *, deadline=None):
```

实现说明：

> Refine an IK boundary using native projection while preserving its s.
>
> Alternating native manifold projection and the fixed progress plane is
> bounded.  Failure remains a rejected seed; measured q0 never enters this
> function.  No custom inverse-kinematics or projection solver is used.

显式输出分支：

```python
value.copy()
None
None
point
None
point if constraint.isSatisfied(point) else None
```

读取的对象属性：

```python
value.copy
time.monotonic
constraint.isSatisfied
constraint.project
np.isfinite(point).all
np.isfinite
```

写入目标：

```python
dimension
count
constraint
point
_
point[-1]
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1998)。

```python
class BoundaryConstraint(ob.Constraint):
```

类字段：

无显式条目。

## `_project_boundary.BoundaryConstraint.__init__`

[定义位置](../../../manipulation/planning.py#L1999)。

输入签名：

```python
def __init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
```

写入目标：

无显式条目。

直接调用符号：

```python
super().__init__
super
```

## `_project_boundary.BoundaryConstraint.function`

[定义位置](../../../manipulation/planning.py#L2002)。

输入签名：

```python
def function(self, values, output):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
```

写入目标：

```python
output[:]
```

直接调用符号：

```python
equality
np.asarray
```

## `_project_boundary.BoundaryConstraint.jacobian`

[定义位置](../../../manipulation/planning.py#L2005)。

输入签名：

```python
def jacobian(self, values, output):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
```

写入目标：

```python
output[:]
```

直接调用符号：

```python
equality
np.asarray
```

## `_orientation_vjp_type`

[定义位置](../../../manipulation/planning.py#L2021)。

输入签名：

```python
def _orientation_vjp_type():
```

实现说明：

> Connect native quaternion values to native world geometric Jacobians.
>
> V2's native ToolPoseCost encodes angular gradients as ``q * omega`` and
> its FK backward consumes that matching body encoding.  An ordinary Torch
> scalar loss instead supplies a Euclidean quaternion gradient; feeding it
> to that native pair rotates its angular gradient into the wrong frame.
> The native world-frame geometric Jacobian supplies the correct derivative
> for custom process/scene losses without changing FK or native pose costs.
>
> The independent FK central-difference probe checks this chain, including
> orientations near 180 degrees where the original mismatch changes signs.

显式输出分支：

```python
WorldQuaternionVJP
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `_orientation_vjp_type.WorldQuaternionVJP`

[定义位置](../../../manipulation/planning.py#L2036)。

```python
class WorldQuaternionVJP(torch.autograd.Function):
```

类字段：

无显式条目。

## `_orientation_vjp_type.WorldQuaternionVJP.forward`

[定义位置](../../../manipulation/planning.py#L2038)。

输入签名：

```python
def forward(ctx, joint_position, quaternion, angular_jacobian):
```

显式输出分支：

```python
quaternion.clone()
```

读取的对象属性：

```python
ctx.save_for_backward
quaternion.detach
angular_jacobian.detach
quaternion.clone
```

写入目标：

无显式条目。

直接调用符号：

```python
ctx.save_for_backward
quaternion.detach
angular_jacobian.detach
quaternion.clone
```

## `_orientation_vjp_type.WorldQuaternionVJP.backward`

[定义位置](../../../manipulation/planning.py#L2046)。

输入签名：

```python
def backward(ctx, gradient):
```

显式输出分支：

```python
(joint_gradient, None, None)
```

读取的对象属性：

```python
ctx.saved_tensors
torch.linalg.cross
torch.linalg
torch.einsum
torch.autograd.function.once_differentiable
torch.autograd.function
torch.autograd
```

写入目标：

```python
quaternion
jacobian
w
vector
gw
gv
angular
joint_gradient
```

直接调用符号：

```python
torch.linalg.cross
torch.einsum
```

## `_native_gradient_layout_type`

[定义位置](../../../manipulation/planning.py#L2062)。

输入签名：

```python
def _native_gradient_layout_type():
```

实现说明：

> Meet native FK's contiguous-gradient contract after Torch view ops.
>
> Concatenating main and exact-anchor samples yields strided gradient slices
> for batch sizes above one.  Cloning forward values does not change that
> backward layout, so normalize it at the native output boundary itself.
> This identity changes neither FK values nor their mathematical gradient.

显式输出分支：

```python
NativeGradientLayout
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `_native_gradient_layout_type.NativeGradientLayout`

[定义位置](../../../manipulation/planning.py#L2072)。

```python
class NativeGradientLayout(torch.autograd.Function):
```

类字段：

无显式条目。

## `_native_gradient_layout_type.NativeGradientLayout.forward`

[定义位置](../../../manipulation/planning.py#L2074)。

输入签名：

```python
def forward(ctx, value):
```

显式输出分支：

```python
value
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `_native_gradient_layout_type.NativeGradientLayout.backward`

[定义位置](../../../manipulation/planning.py#L2078)。

输入签名：

```python
def backward(ctx, gradient):
```

显式输出分支：

```python
gradient.contiguous()
```

读取的对象属性：

```python
gradient.contiguous
```

写入目标：

无显式条目。

直接调用符号：

```python
gradient.contiguous
```

## `_scene_boundary_evaluate`

[定义位置](../../../manipulation/planning.py#L2097)。

输入签名：

```python
def _scene_boundary_evaluate(scene, problem, progress_map, spheres, position, quaternion, progress, joint_position, joint_names, *, trajectory_time, normalized_time, query_counts=None, component_diagnostics=None, scene_evidence_out=None, geometry_evidence_out=None, geometry_scope='full'):
```

实现说明：

> Check material/attachment mode limits without changing physical time.
>
> Float64 neighbors vanish when native inputs are float32. Construct the
> side in the actual progress dtype, retaining original samples first so
> release FK keeps its original exact witness. On a progress plateau only
> its entrance consumes the old side and only its exit consumes the new
> side; an action during that plateau never regains an expired old mode.

显式输出分支：

```python
result
torch.stack((exact_terms, left_terms, right_terms)).amax(0)
```

读取的对象属性：

```python
scene.required_progress_anchors
problem.start_progress
problem.end_progress
progress.is_cuda
torch.cuda.is_current_stream_capturing
torch.cuda
query_counts.update
progress.numel
progress.shape
torch.zeros_like
torch.bool
progress_map.anchor_parameters
torch.where
torch.nextafter
torch.full_like
scene.contacts
rule.actor
rule.interval
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
expanded_progress.shape
exact_terms.scatter
left_columns[None].expand
right_columns[None].expand
scene_evidence_out.items
exact_evidence.scatter
torch.stack((exact_evidence, left_evidence, right_evidence)).amax
torch.stack
torch.stack((exact_terms, left_terms, right_terms)).amax
```

写入目标：

```python
pair_evidence
base_evaluate
anchors
value
result
left
right
anchor
parameters
matching
lower
upper
permission_lower
rule
arrival
left_columns
right_columns
expanded_progress
permission_progress
batch
exact
terms
exact_count
left_count
exact_terms
left_terms
right_terms
category
evidence
exact_evidence
left_evidence
right_evidence
scene_evidence_out[category]
```

直接调用符号：

```python
float
torch.cuda.is_current_stream_capturing
MotionPlanningError
query_counts.update
int
progress.numel
evaluate
retain_pair_evidence
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
exact_terms.scatter
left_columns[None].expand
right_columns[None].expand
scene_evidence_out.items
exact_evidence.scatter
torch.stack((exact_evidence, left_evidence, right_evidence)).amax
torch.stack
torch.stack((exact_terms, left_terms, right_terms)).amax
```

## `_scene_boundary_evaluate.evaluate`

[定义位置](../../../manipulation/planning.py#L2113)。

输入签名：

```python
def evaluate(*args, **kwargs):
```

显式输出分支：

```python
scene.evaluate(*args, **kwargs, geometry_evidence_out=pair_evidence, geometry_scope=geometry_scope)
```

读取的对象属性：

```python
scene.evaluate
```

写入目标：

无显式条目。

直接调用符号：

```python
scene.evaluate
```

## `_scene_boundary_evaluate.retain_pair_evidence`

[定义位置](../../../manipulation/planning.py#L2117)。

输入签名：

```python
def retain_pair_evidence(exact_count, left_columns=(), right_columns=()):
```

显式输出分支：

```python
None
```

读取的对象属性：

```python
geometry_evidence_out.append
```

写入目标：

```python
row
batch
column
source
side
```

直接调用符号：

```python
len
int
bool
geometry_evidence_out.append
```

## `_scene_boundary_evaluate.evaluate`

[定义位置](../../../manipulation/planning.py#L2141)。

输入签名：

```python
def evaluate(*args, **kwargs):
```

显式输出分支：

```python
value
```

读取的对象属性：

```python
term.detach
components.items
name.startswith
torch.zeros_like
residual.detach().reshape
residual.detach
torch.where(torch.isfinite(flat), flat, torch.full_like(flat, float('inf'))).argmax().item
torch.where(torch.isfinite(flat), flat, torch.full_like(flat, float('inf'))).argmax
torch.where
torch.isfinite
torch.full_like
flat[index].item
np.isfinite
kwargs['trajectory_time'].detach().reshape(-1)[index].item
kwargs['trajectory_time'].detach().reshape
kwargs['trajectory_time'].detach
args[3].detach().reshape(-1)[index].item
args[3].detach().reshape
args[3].detach
```

写入目标：

```python
components
value
category
prefix
scene_evidence_out[category]
name
term
residual
flat
index
maximum
component_diagnostics[name]
```

直接调用符号：

```python
base_evaluate
sum
term.detach
components.items
name.startswith
torch.zeros_like
residual.detach().reshape
residual.detach
int
torch.where(torch.isfinite(flat), flat, torch.full_like(flat, float('inf'))).argmax().item
torch.where(torch.isfinite(flat), flat, torch.full_like(flat, float('inf'))).argmax
torch.where
torch.isfinite
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

[定义位置](../../../manipulation/planning.py#L2217)。

输入签名：

```python
def expanded(value):
```

显式输出分支：

```python
torch.cat((value, value.index_select(1, left_columns), value.index_select(1, right_columns)), dim=1)
```

读取的对象属性：

```python
torch.cat
value.index_select
```

写入目标：

无显式条目。

直接调用符号：

```python
torch.cat
value.index_select
```

## `_ik_manager_type`

[定义位置](../../../manipulation/planning.py#L2261)。

输入签名：

```python
def _ik_manager_type(problem, budget, *, clock, seed_parameter):
```

实现说明：

> Extend native IK metrics for declared event allowed sets.
>
> Ordinary reference IK uses the library pose objective. An event query
> disables that objective and binds the event's original allowed set.
> Complete trajectory tasks and geometry belong to the constrained solver
> and candidate verifier, so this manager has no trajectory search state.

显式输出分支：

```python
IKCostManagerCfg
```

读取的对象属性：

无显式条目。

写入目标：

```python
orientation_vjp
native_layout
ik_goal_ids
IKCostManagerCfg.bind_request
IKCostManagerCfg.bind_ik_goals
```

直接调用符号：

```python
_orientation_vjp_type
_native_gradient_layout_type
staticmethod
```

## `_ik_manager_type.IKCostManager`

[定义位置](../../../manipulation/planning.py#L2275)。

```python
class IKCostManager(RobotCostManager):
```

类字段：

无显式条目。

## `_ik_manager_type.IKCostManager.initialize_from_config`

[定义位置](../../../manipulation/planning.py#L2276)。

输入签名：

```python
def initialize_from_config(self, config, transition_model, scene_collision_checker=None, **kwargs):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().initialize_from_config
config.process_mode
transition_model.joint_names
transition_model.robot_model.tool_frames.index
transition_model.robot_model.tool_frames
transition_model.robot_model
problem.tool_link
```

写入目标：

```python
self._process_mode
self._joint_names
self._tool_index
transition_model.robot_model.compute_jacobian
```

直接调用符号：

```python
super().initialize_from_config
super
deepcopy
tuple
transition_model.robot_model.tool_frames.index
```

## `_ik_manager_type.IKCostManager.compute_costs`

[定义位置](../../../manipulation/planning.py#L2283)。

输入签名：

```python
def compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
```

显式输出分支：

```python
collection
collection
```

读取的对象属性：

```python
super().compute_costs
self._process_mode
state.joint_state.position
state.joint_state
state.tool_poses
orientation_vjp.apply
tool.quaternion.detach
tool.quaternion
state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach
state.cuda_robot_model_state.tool_jacobians
state.cuda_robot_model_state
problem.transform_tool_pose
native_layout.apply
tool.position
self._tool_index
q.new_full
q.shape
event.progress
seed_parameter.to(q).expand
seed_parameter.to
problem.residuals
self._joint_names
torch.zeros_like
terms.items
name.startswith
problem.goal_residual
process.square
goals.square().sum
goals.square
goals.sum
collection.add
budget.constraint_weight
```

写入目标：

```python
collection
q
tool
orientation
position
quaternion
event
progress
times
terms
process
name
value
goals
```

直接调用符号：

```python
super().compute_costs
super
orientation_vjp.apply
tool.quaternion.detach
state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach
problem.transform_tool_pose
native_layout.apply
getattr
q.new_full
seed_parameter.to(q).expand
seed_parameter.to
problem.residuals
torch.zeros_like
sum
terms.items
name.startswith
problem.goal_residual
process.square
goals.square().sum
goals.square
goals.sum
collection.add
```

## `_ik_manager_type.IKCostManagerCfg`

[定义位置](../../../manipulation/planning.py#L2309)。

```python
class IKCostManagerCfg(RobotCostManagerCfg):
```

类字段：

```python
process_mode: str = ''
```

## `_ik_manager_type.IKCostManagerCfg.__post_init__`

[定义位置](../../../manipulation/planning.py#L2312)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__post_init__
```

写入目标：

```python
self.class_type
```

直接调用符号：

```python
super().__post_init__
super
```

## `_ik_manager_type.IKCostManagerCfg.create`

[定义位置](../../../manipulation/planning.py#L2317)。

输入签名：

```python
def create(data_dict, scene_collision_checker=None, device_cfg=None):
```

显式输出分支：

```python
IKCostManagerCfg(**values, process_mode=data_dict.get('process_mode', ''))
```

读取的对象属性：

```python
RobotCostManagerCfg.create
item.name
data_dict.get
```

写入目标：

```python
base
values
item
```

直接调用符号：

```python
RobotCostManagerCfg.create
getattr
fields
IKCostManagerCfg
data_dict.get
```

## `_ik_manager_type.bind_request`

[定义位置](../../../manipulation/planning.py#L2322)。

输入签名：

```python
def bind_request(new_problem, new_budget):
```

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
problem
budget
ik_goal_ids
```

直接调用符号：

无显式条目。

## `_ik_manager_type.bind_ik_goals`

[定义位置](../../../manipulation/planning.py#L2325)。

输入签名：

```python
def bind_ik_goals(goal_ids):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
getattr(problem, name)[index].progress
```

写入目标：

```python
selected
identity
name
index
ik_goal_ids
```

直接调用符号：

```python
tuple
len
getattr
ValueError
```

## `_unknown_policy_failure`

[定义位置](../../../manipulation/planning.py#L2358)。

输入签名：

```python
def _unknown_policy_failure(report, policy):
```

实现说明：

> Choose an explicit policy outcome without declaring all routes infeasible.
>
> Named scene components establish what caused the aggregate scene residual.
> Missing evidence, nonfinite results, or any other failed hard requirement
> preserve the ordinary failure/recovery path.

显式输出分支：

```python
None
None
None
None
None
None
None
'needs_observation' if action == 'require_observation' else 'unknown_space_forbidden'
```

读取的对象属性：

```python
report.get
components.items
np.isfinite
name.startswith
residuals.items
name.endswith
```

写入目标：

```python
action
components
residuals
coverage_failed
name
entry
value
scene_failed
```

直接调用符号：

```python
report.get
components.items
np.isfinite
name.startswith
residuals.items
name.endswith
```

## `_reference_entrance`

[定义位置](../../../manipulation/planning.py#L2409)。

输入签名：

```python
def _reference_entrance(problem, scene, robot_validator):
```

实现说明：

> Return calibrated TCP pose and world twist from named measured q/dq.
>
> Pinocchio's forward velocity equals its geometric Jacobian times dq.
> LOCAL_WORLD_ALIGNED expresses velocity at the link origin in robot-base
> axes. Base calibration rotates both vectors into the reference frame;
> the TCP lever arm adds omega cross offset to the linear velocity.
> Independently controlled fingers are already verified not to be TCP
> ancestors. Their absent velocity columns therefore have no TCP effect.
> The official fixed robot model is reused across tasks; this computation
> reads neither task-object geometry nor simulator environment state.

显式输出分支：

```python
(world_tool, np.r_[linear, angular])
```

读取的对象属性：

```python
robot_validator.checker
robot_validator._planning_positions
problem.start_q
problem.joint_names
scene.gripper
pin.neutral
checker.model
np.zeros
checker.model.nv
problem.start_dq
robot_validator.joint_names
checker.model.joints
checker.model.getJointId
joint.idx_q
joint.idx_v
checker.model.createData
pin.forwardKinematics
pin.updateFramePlacements
checker._frame_id
robot_validator.tool_frame
pin.getFrameVelocity
pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
pin.ReferenceFrame
robot_validator.T_reference_base
data.oMf[frame].homogeneous
data.oMf
robot_validator.T_frame_tcp
velocity.angular
velocity.linear
np.cross
np.r_
```

写入目标：

```python
checker
positions
_
q
dq
measured_rates
name
position
joint
q[joint.idx_q]
dq[joint.idx_v]
data
frame
velocity
world_link
world_tool
rotation
angular
offset
linear
```

直接调用符号：

```python
robot_validator._planning_positions
pin.neutral
np.zeros
dict
zip
checker.model.getJointId
checker.model.createData
pin.forwardKinematics
pin.updateFramePlacements
checker._frame_id
pin.getFrameVelocity
np.cross
```

## `_ConstrainedPlanner`

[定义位置](../../../manipulation/planning.py#L2447)。

```python
class _ConstrainedPlanner():
```

类字段：

无显式条目。

实现说明：

> Resident IK proposals and one request-local constrained trajectory problem.

## `_ConstrainedPlanner.__init__`

[定义位置](../../../manipulation/planning.py#L2451)。

输入签名：

```python
def __init__(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None, goal_identities=None):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
torch.backends.cuda.matmul.allow_tf32
torch.backends.cuda.matmul
torch.backends.cuda
torch.backends
torch.backends.cudnn.allow_tf32
torch.backends.cudnn
self._initialize
self.close
```

写入目标：

```python
self.ik
self.check_request
self._precision_torch
self._caller_precision
torch.backends.cuda.matmul.allow_tf32
```

直接调用符号：

```python
self._initialize
self.close
```

## `_ConstrainedPlanner._initialize`

[定义位置](../../../manipulation/planning.py#L2474)。

输入签名：

```python
def _initialize(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None, goal_identities=None):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
robot_validator.validate_planning_binding
scene.gripper
scene.validate_problem
np.unique
np.concatenate
problem.anchors
np.asarray
scene.required_progress_anchors
problem.start_progress
problem.end_progress
budget.max_validation_points
np.isfinite(anchors).all
np.isfinite
np.any
problem.reference_frame
scene.reference_frame
budget.knots
problem.stop_progress
budget.maximum_trajectory_dt
budget.interpolation_steps
budget.interpolation_dt
self.progress_map
scene.gripper.control_timestep_s
self.reference_time_map.u
self.reference_time_map
self.reference_time_map.s
hand.bind_clock
hand.finger_trajectory
kinematics.get
robot_validator.checker
robot_validator.checker.base_frame
ET.tostring
ET.parse(native_urdf).getroot
ET.parse
checker.urdf_xml
hand.controller
hand.joint_names
hand.current_positions
np.float32
locks.items
checker.model.frames[checker._frame_id(robot_validator.tool_frame)].parentJoint
checker.model.frames
checker.model
checker._frame_id
robot_validator.tool_frame
checker.model.names
checker.model.parents
self.progress_map.u
self.progress_map.s
self.robot_metadata
self.anchor_capacity
workspace.pending_cpu_bytes
self.validation_storage
budget.max_host_rss_bytes
self.device_cfg.device
self.device_cfg
budget.max_device_used_bytes
workspace.total_bytes
workspace.cpu_bytes
workspace.gpu_bytes
workspace.cpu_live_bytes
workspace.anchor_fk_bytes
self.device_cfg.to_device
self.formal_clock
self.seed_parameter
pose_cost.update
InverseKinematicsCfg.create
budget.ik_seeds
problem.numerical.position_m
problem.numerical
problem.numerical.rotation_rad
self._manager_type
budget.ik_iterations
budget.random_seed
torch.backends.cuda.matmul
torch.backends.cuda
torch.backends
torch.backends.cudnn
self._caller_precision
self.ik.joint_names
self.ik
self.joint_names
self.ik.kinematics.total_spheres
self.ik.kinematics
problem.joint_names.index
problem.joint_names
JointState.from_numpy
problem.start_q
problem.start_dq
self._bind_online_robot
self.ik.kinematics.config
scene.clone_for_rollout
self._initialize_verification_constraints
self._bind_physical_domains
```

写入目标：

```python
self.robot_validator
budget
initial_progress
anchors
self.progress_map
self.source_progress_map
self.reference_time_map
initial_clock
before_tick
hand
self.finger_trajectory
self.anchor_capacity
robot_dict
self.robot_metadata
kinematics
native_urdf
checker
locks
independent_hand
allowed_locks
name
value
joint
entrance_pose
entrance_twist
self.bound_reference
self.validation_storage
workspace
self.problem
self.scene
self.budget
self.torch
self.JointState
self.GoalToolPose
self.device_cfg
self.scene_workspace_bytes
self.scene_cpu_workspace_bytes
self.scene_gpu_workspace_bytes
self.scene_cpu_live_bytes
self.anchor_fk_workspace_bytes
self.ik
self.anchors
self.robot_dict
self.seed_parameter
self.formal_clock
self._manager_type
self.goal_ik_candidates
self.last_optimizer_diagnostics
self.last_optimizer_runtime
self.delivery_candidate
self.initialization_candidate
self.reference_initialization
self.reference_ik_queries
self.reference_ik_chains
self.initial_seed_strategies
self.route
ik_optimizer
ik_optimizer['optimizer']['inner_iters']
pose_cost
self.reference_ik_pose_weights
pose_cost['weight'][1]
self.motion_configuration
ik_cfg
ik_cfg.use_lm_seed
ik_cfg.exit_early
torch.backends.cuda.matmul.allow_tf32
torch.backends.cudnn.allow_tf32
self.joint_names
columns
self.start
self._verification_fk
self._orientation_vjp
self._native_layout
self._verification_scene
```

直接调用符号：

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
reference_progress_clock
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
_reference_entrance
BoundToolReference
_planning_workspace_bytes
_require_planning_host_memory
DeviceCfg
_require_planning_memory
self.device_cfg.to_device
_ik_manager_type
configuration
tuple
pose_cost.update
deepcopy
load_yaml
join_path
get_task_configs_path
InverseKinematicsCfg.create
InverseKinematics
RuntimeError
problem.joint_names.index
JointState.from_numpy
list
self._bind_online_robot
Kinematics
_orientation_vjp_type
_native_gradient_layout_type
scene.clone_for_rollout
self._initialize_verification_constraints
self._bind_physical_domains
```

## `_ConstrainedPlanner._initialize.configuration`

[定义位置](../../../manipulation/planning.py#L2594)。

输入签名：

```python
def configuration(name, mode):
```

显式输出分支：

```python
data
```

读取的对象属性：

```python
data['rollout'].setdefault
constraint.pop
```

写入目标：

```python
data
constraint
constraint['process_mode']
```

直接调用符号：

```python
deepcopy
load_yaml
join_path
get_task_configs_path
data['rollout'].setdefault
constraint.pop
```

## `_ConstrainedPlanner._bind_online_robot`

[定义位置](../../../manipulation/planning.py#L2647)。

输入签名：

```python
def _bind_online_robot(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.scene.bind_online_robot
self.scene
self.robot_validator
self.check_request
```

写入目标：

无显式条目。

直接调用符号：

```python
self.scene.bind_online_robot
self.check_request
```

## `_ConstrainedPlanner.rebind`

[定义位置](../../../manipulation/planning.py#L2651)。

输入签名：

```python
def rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None):
```

实现说明：

> Rebind measured entrance, observations and task clock on resident IK.

显式输出分支：

```python
self
```

读取的对象属性：

```python
self.robot_validator.validate_planning_binding
self.robot_validator
scene.gripper
scene.validate_problem
np.unique
np.r_
problem.anchors
scene.required_progress_anchors
problem.start_progress
problem.end_progress
budget.knots
problem.stop_progress
budget.maximum_trajectory_dt
budget.interpolation_steps
budget.interpolation_dt
scene.gripper.control_timestep_s
self.reference_time_map.u
self.reference_time_map
self.reference_time_map.s
self.robot_metadata
self.anchor_capacity
workspace.pending_cpu_bytes
budget.max_host_rss_bytes
self.torch
self.device_cfg.device
self.device_cfg
budget.max_device_used_bytes
scene.gripper.bind_clock
self.formal_clock.clear
self.formal_clock
self.formal_clock.update
scene.gripper.finger_trajectory
problem.joint_names.index
problem.joint_names
self.joint_names
self.JointState.from_numpy
self.JointState
problem.start_q
problem.start_dq
self._bind_online_robot
mapping.u
mapping.s
self._manager_type.bind_request
self._manager_type
scene.clone_for_rollout
self.seed_parameter.zero_
self.seed_parameter
self.ik.reset_seed
self.ik
self.ik.core.goal_registry_manager
self.ik.core
manager._goal_buffer
self.ik.core.get_all_rollout_instances
rollout.reset
self._bind_physical_domains
```

写入目标：

```python
budget
mapping
anchors
self.source_progress_map
self.reference_time_map
grid
before_tick
workspace
self.problem
self.scene
self.budget
self.progress_map
self.anchors
self.finger_trajectory
columns
name
self.start
entrance_pose
entrance_twist
self.bound_reference
self._verification_scene
self.last_validation
self.last_optimizer_diagnostics
self.goal_ik_candidates
self.last_optimizer_runtime
self.delivery_candidate
self.initialization_candidate
self.reference_initialization
self.reference_ik_queries
self.reference_ik_chains
self.initial_seed_strategies
manager
manager._goal_buffer.link_goal_poses
rollout
goal
goal.link_goal_poses
goal.goal_js
```

直接调用符号：

```python
self.robot_validator.validate_planning_binding
scene.validate_problem
_initial_timing
_continuation_timing
np.unique
reference_progress_clock
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
_reference_entrance
BoundToolReference
self._manager_type.bind_request
scene.clone_for_rollout
self.seed_parameter.zero_
self.ik.reset_seed
self.ik.core.get_all_rollout_instances
rollout.reset
getattr
self._bind_physical_domains
```

## `_ConstrainedPlanner._seed_u`

[定义位置](../../../manipulation/planning.py#L2715)。

输入签名：

```python
def _seed_u(self, progress):
```

实现说明：

> Choose the earliest plateau point for initializer-only state queries.

显式输出分支：

```python
float(np.interp(progress, values, self.progress_map.u[indices]))
```

读取的对象属性：

```python
np.unique
self.progress_map.s
self.progress_map
np.interp
self.progress_map.u
```

写入目标：

```python
values
indices
```

直接调用符号：

```python
np.unique
float
np.interp
```

## `_ConstrainedPlanner._initialize_verification_constraints`

[定义位置](../../../manipulation/planning.py#L2725)。

输入签名：

```python
def _initialize_verification_constraints(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.device_cfg
self._verification_native.initialize_from_config
self._verification_native
self.ik.config.metrics_rollout_config.constraint_cfg
self.ik.config.metrics_rollout_config
self.ik.config
self.ik
self.ik.transition_model
```

写入目标：

```python
self._RobotState
self._verification_native
```

直接调用符号：

```python
RobotCostManager
self._verification_native.initialize_from_config
deepcopy
```

## `_ConstrainedPlanner._bind_physical_domains`

[定义位置](../../../manipulation/planning.py#L2738)。

输入签名：

```python
def _bind_physical_domains(self):
```

实现说明：

> Bind exact feedback and official limits to the single cubic model.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._verification_native.get_cost('cspace').config.joint_limits
self._verification_native.get_cost('cspace').config
self._verification_native.get_cost
self._verification_native
limits.joint_names.index
limits.joint_names
self.joint_names
self.torch.stack([getattr(limits, name)[:, columns] for name in ('position', 'velocity', 'acceleration', 'jerk')]).cpu().double
self.torch.stack([getattr(limits, name)[:, columns] for name in ('position', 'velocity', 'acceleration', 'jerk')]).cpu
self.torch.stack
self.torch
self.problem.joint_names.index
self.problem.joint_names
self.problem
self.torch.as_tensor
self.problem.start_q
self.problem.start_dq
self.formal_clock
self.budget.knots
self.budget
self.problem.tool_link
self.scene.robot_model._geometries.values
self.scene.robot_model._geometries
self.scene.robot_model
self.scene
self.robot_validator.checker.model
self.robot_validator.checker
self.robot_validator
self.problem.base_pose
self.physical_domain.knots
self.physical_domain
self.budget.trajectory_iterations
self._bind_computation
```

写入目标：

```python
limits
columns
name
bounds
self.position_domain
self.physical_domain
frames
row
self.pin_kinematics
self.action_horizon
self.main_trajectory_iterations
```

直接调用符号：

```python
self._verification_native.get_cost
limits.joint_names.index
self.torch.stack([getattr(limits, name)[:, columns] for name in ('position', 'velocity', 'acceleration', 'jerk')]).cpu().double
self.torch.stack([getattr(limits, name)[:, columns] for name in ('position', 'velocity', 'acceleration', 'jerk')]).cpu
self.torch.stack
getattr
JointPositionDomain
self.problem.joint_names.index
CubicJointDomain
self.torch.as_tensor
self.scene.robot_model._geometries.values
PinKinematics
self._bind_computation
```

## `_ConstrainedPlanner._bind_computation`

[定义位置](../../../manipulation/planning.py#L2757)。

输入签名：

```python
def _bind_computation(self, extra_parameters=()):
```

实现说明：

> Freeze sorted physical queries and preserve exact event identities.

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.linspace
self.budget.knots
self.budget
self.budget.interpolation_steps
self.progress_map.anchor_parameters
self.progress_map
self.anchors
np.unique
np.r_
self.budget.max_validation_points
np.interp
self.progress_map.u
self.progress_map.s
np.searchsorted
self.problem
self.scene
self.pin_kinematics
self.physical_domain
self.joint_names
self.motion_configuration
self.formal_clock
self.budget.constraint_weight
self.budget.max_host_rss_bytes
```

写入目标：

```python
ordinary
pairs
parameters
progress
progress[np.searchsorted(parameters, pairs[:, 0])]
self.computation
```

直接调用符号：

```python
np.linspace
self.progress_map.anchor_parameters
np.unique
len
MotionPlanningError
np.interp
np.searchsorted
TrajectoryComputation
```

## `_ConstrainedPlanner.initial_state_diagnostics`

[定义位置](../../../manipulation/planning.py#L2775)。

输入签名：

```python
def initial_state_diagnostics(self):
```

实现说明：

> Test original binary64 feedback and the registered observed scene.

显式输出分支：

```python
{'passed': not failed, 'residuals': residuals, 'failed_components': failed, 'scene_components': scene_components, 'observed_geometry': deepcopy(self.scene.last_query_report), 'scope': 'measured joint limits and selected online collision model'}
```

读取的对象属性：

```python
self.torch.as_tensor
self.torch
self.physical_domain._curve_map.start_q
self.physical_domain._curve_map
self.physical_domain
self.physical_domain._curve_map.start_dq
q.new_full
q.shape
self.problem.start_progress
self.problem
q.new_zeros
self.torch.no_grad
self.physical_domain.bounds
self.torch.relu
self.scene
self.pin_kinematics
self.joint_names
self.progress_map
q.new_empty
value.max().item
value.max
components.items
residuals.items
np.isfinite
self.scene.last_query_report
```

写入目标：

```python
q
dq
progress
times
components
order
name
value
bounds
components[name]
_
position
quaternion
scene_components
components['scene']
residuals
failed
```

直接调用符号：

```python
self.torch.as_tensor
q.new_full
q.new_zeros
self.torch.no_grad
enumerate
self.torch.relu
robot_tool_states
_scene_boundary_evaluate
q.new_empty
float
value.max().item
value.max
components.items
residuals.items
np.isfinite
deepcopy
```

## `_ConstrainedPlanner.close`

[定义位置](../../../manipulation/planning.py#L2812)。

输入签名：

```python
def close(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
solver.destroy
gc.collect
self._precision_torch.backends.cuda.matmul
self._precision_torch.backends.cuda
self._precision_torch.backends
self._precision_torch
self._precision_torch.backends.cudnn
```

写入目标：

```python
name
solver
prior
self._precision_torch.backends.cuda.matmul.allow_tf32
self._precision_torch.backends.cudnn.allow_tf32
self._caller_precision
```

直接调用符号：

```python
getattr
solver.destroy
setattr
gc.collect
```

## `_ConstrainedPlanner._goal`

[定义位置](../../../manipulation/planning.py#L2835)。

输入签名：

```python
def _goal(self, time_s):
```

实现说明：

> Create native link goals from the same physical-time TCP reference.

显式输出分支：

```python
self.GoalToolPose(tool_frames=[self.problem.tool_link], position=position[:, :, None, None].contiguous(), quaternion=quaternion[:, :, None, None].contiguous())
```

读取的对象属性：

```python
self.device_cfg.to_device
self.device_cfg
self.problem.reference_to_link_pose
self.problem
self.bound_reference.sample
self.bound_reference
self.GoalToolPose
self.problem.tool_link
position[:, :, None, None].contiguous
quaternion[:, :, None, None].contiguous
```

写入目标：

```python
times
position
quaternion
```

直接调用符号：

```python
self.device_cfg.to_device
float
self.problem.reference_to_link_pose
self.bound_reference.sample
self.GoalToolPose
position[:, :, None, None].contiguous
quaternion[:, :, None, None].contiguous
```

## `_ConstrainedPlanner._seed_velocity_bounds`

[定义位置](../../../manipulation/planning.py#L2844)。

输入签名：

```python
def _seed_velocity_bounds(self):
```

实现说明：

> Read existing physical speed bounds in the action's named joint order.

显式输出分支：

```python
limits.velocity.detach().cpu().numpy()[:, columns]
```

读取的对象属性：

```python
self._verification_native.get_cost('cspace').config.joint_limits
self._verification_native.get_cost('cspace').config
self._verification_native.get_cost
self._verification_native
limits.joint_names.index
limits.joint_names
self.joint_names
limits.velocity.detach().cpu().numpy
limits.velocity.detach().cpu
limits.velocity.detach
limits.velocity
```

写入目标：

```python
limits
columns
name
```

直接调用符号：

```python
self._verification_native.get_cost
limits.joint_names.index
limits.velocity.detach().cpu().numpy
limits.velocity.detach().cpu
limits.velocity.detach
```

## `_ConstrainedPlanner._solve_ik`

[定义位置](../../../manipulation/planning.py#L2851)。

输入签名：

```python
def _solve_ik(self, goal, seeds):
```

实现说明：

> Submit a bounded IK batch and retain work even if native code fails.
>
> ``seeds`` is [batch, supplied_seed, named_joint]. Native SeedManager
> fills each batch to the request's fixed ``ik_seeds``. The ledger unit
> is one submitted candidate solve, including unsuccessful candidates.
> ``request_work`` is the public request's cumulative ledger, bound
> before either reference initialization or event search is entered.

显式输出分支：

```python
self.ik.solve_pose(goal, current_state=self.start, seed_config=seeds, return_seeds=self.budget.ik_seeds)
```

读取的对象属性：

```python
self.check_request
seeds.shape
self.budget.ik_seeds
self.budget
self.budget.max_validation_points
self.budget.maximum_trajectory_solves
self.request_work
self.ik.solve_pose
self.ik
self.start
```

写入目标：

```python
amount
limit
self.request_work['ik_queries']
```

直接调用符号：

```python
self.check_request
int
PlanningWorkExhausted
self.ik.solve_pose
```

## `_ConstrainedPlanner.reference_seeds`

[定义位置](../../../manipulation/planning.py#L2872)。

输入签名：

```python
def reference_seeds(self, joint_reference=None):
```

实现说明：

> Construct numerical starts without treating references as constraints.

显式输出分支：

```python
(seeds, [row['native_success_count'] for row in self.reference_ik_queries])
```

读取的对象属性：

```python
self.budget.trajectory_seeds
self.budget
self.physical_domain._curve_map.reference_parameters
self.physical_domain._curve_map
self.physical_domain
joint_reference.joint_names
self.joint_names
joint_reference.progress_u
joint_reference.progress_s
self.reference_time_map
self.progress_map
np.interp
self.reference_time_map.u
self.reference_time_map.s
self.source_progress_map
joint_reference.joint_names.index
joint_reference.duration
joint_reference.sample
self.formal_clock
source_times.tolist
self.start.position[:, None].clone
self.start.position
self.start
self._reference_ik_query
self.device_cfg.to_device(layer['values'][tracking:tracking + 1][None]).contiguous
self.device_cfg.to_device
self.device_cfg
self._select_reference_chains
self.reference_ik_chains
targets.append
np.asarray
strategies.append
np.repeat
self.physical_domain._curve_map.start_q
self.physical_domain.fit_samples
controls.append
reports.append
self.torch.as_tensor
self.torch
np.stack
self.torch.float64
self._controls_curve
self.reference_ik_queries
```

写入目标：

```python
self.joint_reference_report
self.reference_ik_queries
self.reference_ik_distinct_counts
self.reference_ik_chains
count
parameters
targets
strategies
source_map
source_parameters
original_parameters
columns
name
source_times
values
previous
source
layers
parameter
layer
layers[float(parameter)]
tracking
chain
controls
reports
target
fitted
report
self.initial_seed_strategies
self.seed_strategies
self.reference_initialization
seeds
self.initialization_candidate
row
```

直接调用符号：

```python
isinstance
set
MotionPlanningError
_ProgressMap
_phase_aligned_parameters
np.interp
joint_reference.joint_names.index
joint_reference.sample
source_times.tolist
self.start.position[:, None].clone
self._reference_ik_query
float
self.device_cfg.to_device(layer['values'][tracking:tracking + 1][None]).contiguous
self.device_cfg.to_device
self._select_reference_chains
targets.append
np.asarray
strategies.append
len
np.repeat
self.physical_domain.fit_samples
controls.append
reports.append
self.torch.as_tensor
np.stack
self._controls_curve
```

## `_ConstrainedPlanner._reference_ik_query`

[定义位置](../../../manipulation/planning.py#L2932)。

输入签名：

```python
def _reference_ik_query(self, u, previous, *, warm_start_source):
```

实现说明：

> Keep one native query's original candidate and static evidence identity.
>
> ``u`` is normalized physical time; ``previous[1,1,D]`` is the declared
> warm start in active-joint order. Each query uses the full tool
> reference, static checks and fixed native seed cap.
> The execution log is append-only; time-ordered pool views are separate.

显式输出分支：

```python
{'values': values, 'reports': reports, 'energies': energy, 'tracking': tracking, 'success_count': success_count, 'query_index': query_index}
```

读取的对象属性：

```python
np.stack
self.position_domain.lower.detach().cpu().numpy
self.position_domain.lower.detach().cpu
self.position_domain.lower.detach
self.position_domain.lower
self.position_domain
self.position_domain.upper.detach().cpu().numpy
self.position_domain.upper.detach().cpu
self.position_domain.upper.detach
self.position_domain.upper
self.reference_ik_queries
self.seed_parameter.fill_
self.seed_parameter
self._solve_ik
self._goal
self.formal_clock
self.ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy
self.ik.get_active_js(result.js_solution).position[0].detach().cpu
self.ik.get_active_js(result.js_solution).position[0].detach
self.ik.get_active_js(result.js_solution).position
self.ik.get_active_js
self.ik
result.js_solution
result.success[0].detach().cpu().numpy
result.success[0].detach().cpu
result.success[0].detach
result.success
(flags & np.isfinite(values).all(-1)).sum
np.isfinite(values).all
np.isfinite
self._rank_ik_candidates
previous[0].detach().cpu().numpy
previous[0].detach().cpu
previous[0].detach
(values >= position_limits[0]).all
(values <= position_limits[1]).all
self.reference_ik_distinct_counts.append
self.reference_ik_distinct_counts
np.isfinite(values[index]).all
getattr(result, attribute)[0].detach().cpu().numpy().reshape
getattr(result, attribute)[0].detach().cpu().numpy
getattr(result, attribute)[0].detach().cpu
getattr(result, attribute)[0].detach
errors.shape
self.reference_ik_pose_weights
np.errstate
report.update
self.reference_ik_queries.append
previous.shape
self.budget.ik_seeds
self.budget
```

写入目标：

```python
position_limits
query_index
result
values
flags
success_count
ranked
reports
legal
qualified
index
report
report['reference_status']
residuals
attribute
name
errors
residuals[attribute]
error
report[name]
wp
wr
energy
proposals
tracking
report['position_domain_valid']
report['quadratic_pose_energy']
value
row
```

直接调用符号：

```python
np.stack
self.position_domain.lower.detach().cpu().numpy
self.position_domain.lower.detach().cpu
self.position_domain.lower.detach
self.position_domain.upper.detach().cpu().numpy
self.position_domain.upper.detach().cpu
self.position_domain.upper.detach
len
self.seed_parameter.fill_
float
self._solve_ik
self._goal
self.ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy
self.ik.get_active_js(result.js_solution).position[0].detach().cpu
self.ik.get_active_js(result.js_solution).position[0].detach
self.ik.get_active_js
result.success[0].detach().cpu().numpy
result.success[0].detach().cpu
result.success[0].detach
int
(flags & np.isfinite(values).all(-1)).sum
np.isfinite(values).all
np.isfinite
self._rank_ik_candidates
previous[0].detach().cpu().numpy
previous[0].detach().cpu
previous[0].detach
(values >= position_limits[0]).all
(values <= position_limits[1]).all
self.reference_ik_distinct_counts.append
enumerate
np.isfinite(values[index]).all
getattr(result, attribute)[0].detach().cpu().numpy().reshape
getattr(result, attribute)[0].detach().cpu().numpy
getattr(result, attribute)[0].detach().cpu
getattr(result, attribute)[0].detach
getattr
ValueError
zip
np.errstate
select_ik_tracking_candidate
bool
report.update
self.reference_ik_queries.append
```

## `_ConstrainedPlanner._select_reference_chains`

[定义位置](../../../manipulation/planning.py#L3010)。

输入签名：

```python
def _select_reference_chains(self, layers):
```

实现说明：

> Select complete numerical chains from the time-ordered pool view.

显式输出分支：

```python
chains
```

读取的对象属性：

```python
self.start.position.detach().cpu().numpy
self.start.position.detach().cpu
self.start.position.detach
self.start.position
self.start
np.asarray
self.formal_clock
self._seed_velocity_bounds
self.budget.trajectory_seeds
self.budget
np.stack
self.position_domain.lower.detach().cpu().numpy
self.position_domain.lower.detach().cpu
self.position_domain.lower.detach
self.position_domain.lower
self.position_domain
self.position_domain.upper.detach().cpu().numpy
self.position_domain.upper.detach().cpu
self.position_domain.upper.detach
self.position_domain.upper
self.problem.numerical.joint
self.problem.numerical
self.problem
```

写入目标：

```python
parameters
rows
u
chains
row
chain
chain['query_parameters']
chain['candidate_sources']
index
```

直接调用符号：

```python
sorted
select_ik_chains
self.start.position.detach().cpu().numpy
self.start.position.detach().cpu
self.start.position.detach
np.asarray
self._seed_velocity_bounds
np.stack
self.position_domain.lower.detach().cpu().numpy
self.position_domain.lower.detach().cpu
self.position_domain.lower.detach
self.position_domain.upper.detach().cpu().numpy
self.position_domain.upper.detach().cpu
self.position_domain.upper.detach
zip
```

## `_ConstrainedPlanner.ompl_seed`

[定义位置](../../../manipulation/planning.py#L3031)。

输入签名：

```python
def ompl_seed(self, initial_seed, candidate_trajectory):
```

实现说明：

> Return a spatial initializer and evidence; never produce a new clock.
>
> Only real relation switches, goal times and scene switches divide
> local OMPL charts. Their boundary configurations are sampled and
> projected into the original allowed sets, then revised on failure.
> Reference controls initialize these proposals without fixing them.
> A selected prefix supplies its actual release FK to its child scene.

显式输出分支：

```python
(self.torch.as_tensor(controls)[None, None], report)
(None, report)
```

读取的对象属性：

```python
self.torch
time.monotonic
self.budget.search_seconds
self.budget
self.position_domain.lower.cpu().numpy().reshape
self.position_domain.lower.cpu().numpy
self.position_domain.lower.cpu
self.position_domain.lower
self.position_domain
self.position_domain.upper.cpu().numpy().reshape
self.position_domain.upper.cpu().numpy
self.position_domain.upper.cpu
self.position_domain.upper
self.problem
self.device_cfg.to_device
self.device_cfg
self.progress_map.u
self.progress_map
self.progress_map.s
self.progress_map.anchor_parameters
self.anchors
self.finger_trajectory
np.unique
np.r_
self.formal_clock
self.finger_trajectory.events
self.start.position[0].detach().cpu().numpy
self.start.position[0].detach().cpu
self.start.position[0].detach
self.start.position
self.start
requirement.start
self.problem.start_progress
boundary_times.append
requirement.end
requirement.progress
goals_by_time.setdefault(float(parameter), []).append
goals_by_time.setdefault
self.scene.required_progress_anchors
self.scene
boundary_times.extend
boundaries.tolist
np.random.default_rng
np.random
self.budget.random_seed
initial_seed[0].detach().cpu().numpy
initial_seed[0].detach().cpu
initial_seed[0].detach
self.scene.clone_for_rollout
self._build_goal_ik_candidates
self.budget.ik_seeds
frames.pop
anchor_progress.get
np.interp
self.problem.projection_requirements
report['branches'].append
goal.tolist
invalid_states.clear
self.budget.validation_joint_step
self.budget.max_validation_points
evidence.update
np.any
np.diff
torch.no_grad
scene.with_release_history
frames.append
np.concatenate
np.array_split
np.ceil
positions.append
p.clone
rotations.append
r.clone
progress_values.append
progress.clone
torch.cat
self.problem.evaluate
self.joint_names
torch.zeros_like
torch.isfinite(process).all
torch.isfinite
(process <= 0).all
self.problem.verification_issues
self.physical_domain.fit_samples
self.physical_domain
initial_seed[0, 0].detach().cpu().numpy
initial_seed[0, 0].detach().cpu
initial_seed[0, 0].detach
report.update
self.torch.as_tensor
```

写入目标：

```python
torch
search_started
deadline
report
lower
upper
tolerance
map_u
map_s
anchor_pairs
query_u
anchor_progress
event
start
boundary_times
name
requirement
goals_by_time
index
parameter
scene_anchors
value
boundaries
report['chart_times']
work
rng
source_controls
invalid_states
scene
frames
stage
frame
lo
hi
frame['attempts']
begin
proposal_source
goal
proposal_evidence
goal_s
middle_s
active
boundary_active
goal_ids
boundary_equations
process_count
evidence
evidence['boundary_state']
constraint
rank
stage_started
seconds
validity_cache
path
stage_report
evidence['failure']
frame['path']
item
_
p
r
progress
child_scene
whole_path
positions
rotations
progress_values
chunk
full_progress
q
process
controls
fit
report['budget_exhausted']
report['state_queries']
report['elapsed_seconds']
report['failure']
```

直接调用符号：

```python
time.monotonic
self.position_domain.lower.cpu().numpy().reshape
self.position_domain.lower.cpu().numpy
self.position_domain.lower.cpu
self.position_domain.upper.cpu().numpy().reshape
self.position_domain.upper.cpu().numpy
self.position_domain.upper.cpu
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
self._build_goal_ik_candidates
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
self.physical_domain.fit_samples
initial_seed[0, 0].detach().cpu().numpy
initial_seed[0, 0].detach().cpu
initial_seed[0, 0].detach
report.update
self.torch.as_tensor
```

## `_ConstrainedPlanner.ompl_seed.goals_at_time`

[定义位置](../../../manipulation/planning.py#L3089)。

输入签名：

```python
def goals_at_time(parameter):
```

显式输出分支：

```python
tuple(goals_by_time.get(float(parameter), ()))
```

读取的对象属性：

```python
goals_by_time.get
```

写入目标：

无显式条目。

直接调用符号：

```python
tuple
goals_by_time.get
float
```

## `_ConstrainedPlanner.ompl_seed.check_search`

[定义位置](../../../manipulation/planning.py#L3093)。

输入签名：

```python
def check_search():
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.check_request
time.monotonic
```

写入目标：

无显式条目。

直接调用符号：

```python
self.check_request
time.monotonic
TimeoutError
```

## `_ConstrainedPlanner.ompl_seed.state_parts`

[定义位置](../../../manipulation/planning.py#L3099)。

输入签名：

```python
def state_parts(values):
```

显式输出分支：

```python
(js, state, p, r, _torch_interpolate(map_u, map_s, values[..., -1]))
```

读取的对象属性：

```python
np.prod
values.shape
self.budget.max_validation_points
self.budget
values[..., :-1].contiguous
torch.zeros_like
self.JointState
self.device_cfg.to_device
self.device_cfg
self.budget.interpolation_dt
self.joint_names
self._verification_fk.compute_kinematics
self._verification_fk
state.tool_poses
tools.tool_frames.index
tools.tool_frames
self.problem.tool_link
self.problem
self._orientation_vjp.apply
self._orientation_vjp
tools.quaternion.detach
tools.quaternion
state.tool_jacobians[..., 3:, :].detach
state.tool_jacobians
self._native_layout.apply
self._native_layout
tools.position
self.problem.transform_tool_pose
```

写入目标：

```python
count
work['state_queries']
joints
zero
js
state
tools
index
quaternion
position
p
r
```

直接调用符号：

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

## `_ConstrainedPlanner.ompl_seed.equations`

[定义位置](../../../manipulation/planning.py#L3118)。

输入签名：

```python
def equations(scene, active, goals=()):
```

显式输出分支：

```python
evaluate
```

读取的对象属性：

无显式条目。

写入目标：

```python
cache
```

直接调用符号：

无显式条目。

## `_ConstrainedPlanner.ompl_seed.equations.residual_function`

[定义位置](../../../manipulation/planning.py#L3120)。

输入签名：

```python
def residual_function(values):
```

显式输出分支：

```python
torch.cat((process, drive, goal), -1).reshape(-1)
```

读取的对象属性：

```python
values.reshape
self.problem.project_residual
self.problem
js.position
self.joint_names
scene.project_residual
self.problem.goal_residual
torch.cat((process, drive, goal), -1).reshape
torch.cat
```

写入目标：

```python
js
_
p
r
progress
process
drive
goal
```

直接调用符号：

```python
state_parts
values.reshape
self.problem.project_residual
scene.project_residual
self.problem.goal_residual
torch.cat((process, drive, goal), -1).reshape
torch.cat
```

## `_ConstrainedPlanner.ompl_seed.equations.evaluate`

[定义位置](../../../manipulation/planning.py#L3128)。

输入签名：

```python
def evaluate(values, derivative):
```

显式输出分支：

```python
cache[key]
np.empty((0, len(values)))
result
```

读取的对象属性：

```python
np.asarray(values, dtype=float).tobytes
np.asarray
self.device_cfg.to_device
self.device_cfg
torch.enable_grad
residual_function(values_t).numel
np.empty
torch.autograd.functional.jacobian
torch.autograd.functional
torch.autograd
torch.no_grad
answer.detach().cpu().numpy().astype
answer.detach().cpu().numpy
answer.detach().cpu
answer.detach
self.budget.max_validation_points
self.budget
```

写入目标：

```python
key
values_t
answer
result
cache[key]
```

直接调用符号：

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

## `_ConstrainedPlanner.ompl_seed.valid`

[定义位置](../../../manipulation/planning.py#L3152)。

输入签名：

```python
def valid(values, scene):
```

显式输出分支：

```python
accepted
```

读取的对象属性：

```python
torch.no_grad
self.device_cfg.to_device
self.device_cfg
self.formal_clock
self.problem.evaluate
self.problem
js.position
self.joint_names
js.velocity
self.progress_map
self.problem.transform_robot_spheres
state.robot_spheres
self._RobotState
self._verification_native.compute_costs
self._verification_native
terms.update
native.names
native.values
invalid_states.append
progress.item
term.max().item
term.max
terms.items
```

写入目标：

```python
values_t
js
state
p
r
progress
clock_time
terms
native_state
native
name
value
accepted
term
```

直接调用符号：

```python
torch.no_grad
self.device_cfg.to_device
state_parts
self.problem.evaluate
goals_at_time
float
_scene_boundary_evaluate
self.problem.transform_robot_spheres
self._RobotState
self._verification_native.compute_costs
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

## `_ConstrainedPlanner.ompl_seed.proposals`

[定义位置](../../../manipulation/planning.py#L3184)。

输入签名：

```python
def proposals(begin, parameter):
```

显式输出分支：

```python
('ranked_functional_ik', np.r_[row['joint_position'], parameter], {'functional_ik_candidate_index': index, 'functional_ik_report': deepcopy(row['report'])})
('candidate_curve', np.r_[candidate_trajectory._curve(parameter * candidate_trajectory.duration), parameter], {})
('joint_box_random', np.r_[rng.uniform(lower, upper), parameter], {})
('reference_curve', np.r_[self.physical_domain.evaluate(controls, np.asarray([parameter]))[0], parameter], {})
('previous_boundary', np.r_[begin[:-1], parameter], {})
('joint_box_random', np.r_[rng.uniform(lower, upper), parameter], {})
```

读取的对象属性：

```python
self.goal_ik_candidates.get
self.goal_ik_candidates
np.isfinite(row['joint_position']).all
np.isfinite
np.r_
candidate_trajectory._curve
candidate_trajectory.duration
rng.uniform
self.physical_domain.evaluate
self.physical_domain
np.asarray
```

写入目标：

```python
entry
qualified
index
row
controls
```

直接调用符号：

```python
self.goal_ik_candidates.get
float
enumerate
np.isfinite(row['joint_position']).all
np.isfinite
deepcopy
candidate_trajectory._curve
rng.uniform
self.physical_domain.evaluate
np.asarray
```

## `_ConstrainedPlanner.ompl_seed.search_frame`

[定义位置](../../../manipulation/planning.py#L3220)。

输入签名：

```python
def search_frame(begin, branch_scene, parameter):
```

显式输出分支：

```python
{'start': begin, 'scene': branch_scene, 'proposals': proposals(begin, parameter), 'attempts': 0, 'path': None}
```

读取的对象属性：

无显式条目。

写入目标：

```python
branch_scene.check_request
```

直接调用符号：

```python
proposals
```

## `_ConstrainedPlanner.ompl_seed.stage_valid`

[定义位置](../../../manipulation/planning.py#L3276)。

输入签名：

```python
def stage_valid(value):
```

显式输出分支：

```python
validity_cache[key]
```

读取的对象属性：

```python
np.asarray(value, dtype=float).tobytes
np.asarray
```

写入目标：

```python
key
validity_cache[key]
```

直接调用符号：

```python
np.asarray(value, dtype=float).tobytes
np.asarray
valid
```

## `_ConstrainedPlanner._functional_goal`

[定义位置](../../../manipulation/planning.py#L3348)。

输入签名：

```python
def _functional_goal(self, target):
```

实现说明：

> Express the declared nominal material pose in native link coordinates.
>
> Native requests require a pose buffer. The constrained route evaluates
> the event's continuous allowed set in its cost manager; this nominal
> pose only supplies the native request's buffer and frame layout.

显式输出分支：

```python
self.GoalToolPose(tool_frames=[self.problem.tool_link], position=position[:, None, None].contiguous(), quaternion=quaternion[:, None, None].contiguous())
```

读取的对象属性：

```python
target.target
np.linalg.inv
np.linalg
target.tool_from_function
self.device_cfg.to_device
self.device_cfg
Rotation.from_matrix(poses[:, :3, :3]).as_quat
Rotation.from_matrix
self.problem.reference_to_link_pose
self.problem
self.GoalToolPose
self.problem.tool_link
position[:, None, None].contiguous
quaternion[:, None, None].contiguous
```

写入目标：

```python
poses
position
quaternion
```

直接调用符号：

```python
np.linalg.inv
self.device_cfg.to_device
Rotation.from_matrix(poses[:, :3, :3]).as_quat
Rotation.from_matrix
self.problem.reference_to_link_pose
self.GoalToolPose
position[:, None, None].contiguous
quaternion[:, None, None].contiguous
```

## `_ConstrainedPlanner._rank_ik_candidates`

[定义位置](../../../manipulation/planning.py#L3368)。

输入签名：

```python
def _rank_ik_candidates(self, candidates, success, previous, u, goal_ids=()):
```

实现说明：

> Rank bounded IK proposals with the same online geometry as trajectory metrics.

显式输出分支：

```python
(ranked, reports)
```

读取的对象属性：

```python
np.asarray
np.interp
self.progress_map.u
self.progress_map
self.progress_map.s
reports.append
np.isfinite(row).all
np.isfinite
np.flatnonzero(~np.isfinite(row)).tolist
np.flatnonzero
self.torch.no_grad
self.torch
self.device_cfg.to_device(row[None, None]).contiguous
self.device_cfg.to_device
self.device_cfg
self.torch.zeros_like
self.JointState
self.budget.interpolation_dt
self.budget
self.joint_names
self._verification_fk.compute_kinematics
self._verification_fk
state.tool_poses.tool_frames.index
state.tool_poses.tool_frames
state.tool_poses
self.problem.tool_link
self.problem
self.problem.transform_tool_pose
state.tool_poses.position
state.tool_poses.quaternion
q.new_full
q.shape
self.formal_clock
self.problem.residuals
task.items
name.startswith
self.scene
self.problem.transform_robot_spheres
state.robot_spheres
self._RobotState
self._verification_native.compute_costs
self._verification_native
constraints.names
constraints.values
report.update
self.budget.ik_seeds
self.problem.numerical.joint
self.problem.numerical
```

写入目标：

```python
values
flags
phase
reports
index
row
report
report['invalid_joint_indices']
report['static_model_query']
q
zero
js
state
p
r
progress
times
task
name
value
scene_components
online
native
constraints
model
ranked
```

直接调用符号：

```python
np.asarray
float
np.interp
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
self._RobotState
self._verification_native.compute_costs
dict
zip
report.update
_ompl_terms_accepted
failures
rank_ik_candidates
```

## `_ConstrainedPlanner._rank_ik_candidates.failures`

[定义位置](../../../manipulation/planning.py#L3422)。

输入签名：

```python
def failures(terms):
```

显式输出分支：

```python
{name: value for (name, value) in maxima.items() if value > 0 or not np.isfinite(value)}
```

读取的对象属性：

```python
terms.items
value.detach().reshape
value.detach
self.torch.where(self.torch.isfinite(flat), flat, self.torch.full_like(flat, float('inf'))).argmax
self.torch.where
self.torch
self.torch.isfinite
self.torch.full_like
flat[index].item
maxima.items
np.isfinite
```

写入目标：

```python
maxima
name
value
flat
index
maxima[name]
```

直接调用符号：

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

## `_ConstrainedPlanner._build_goal_ik_candidates`

[定义位置](../../../manipulation/planning.py#L3438)。

输入签名：

```python
def _build_goal_ik_candidates(self, reference_seeds, *, check_search):
```

实现说明：

> Solve each actual event's allowed set with the existing native IK.
>
> All exact same-time pose, coordinate and support declarations enter
> the shared point objective. A nominal pose only fills the native goal
> buffer while its tracking cost is disabled. The returned states retain
> independent task and scene qualification before curve construction.

显式输出分支：

```python
self.goal_ik_candidates
```

读取的对象属性：

```python
reference_seeds[0].detach().cpu().numpy
reference_seeds[0].detach().cpu
reference_seeds[0].detach
self.problem
self.problem.start_progress
target.progress
self.problem.end_progress
events.setdefault(target.progress, []).append
events.setdefault
events.items
self.progress_map.anchor_parameters
self.progress_map
self.physical_domain.evaluate
self.physical_domain
np.asarray
self.seed_parameter.fill_
self.seed_parameter
self.problem.pose_events
self._goal
self.formal_clock
self._functional_goal
self._manager_type.bind_ik_goals
self._manager_type
self.ik.disable_tool_pose_tracking
self.ik
self._solve_ik
self.device_cfg.to_device(previous[None]).contiguous
self.device_cfg.to_device
self.device_cfg
self.ik.enable_tool_pose_tracking
self.ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy
self.ik.get_active_js(result.js_solution).position[0].detach().cpu
self.ik.get_active_js(result.js_solution).position[0].detach
self.ik.get_active_js(result.js_solution).position
self.ik.get_active_js
result.js_solution
result.success[0].detach().cpu().numpy
result.success[0].detach().cpu
result.success[0].detach
result.success
self._rank_ik_candidates
np.zeros_like
report.update
self.goal_ik_candidates
candidates[index].tolist
```

写入目标：

```python
self.goal_ik_candidates
controls
events
name
index
target
progress
identities
goal_ids
u
previous
pose
native_goal
result
candidates
success
selected
reports
report
self.goal_ik_candidates[u]
value
```

直接调用符号：

```python
reference_seeds[0].detach().cpu().numpy
reference_seeds[0].detach().cpu
reference_seeds[0].detach
enumerate
getattr
events.setdefault(target.progress, []).append
events.setdefault
sorted
events.items
check_search
tuple
float
self.progress_map.anchor_parameters
self.physical_domain.evaluate
np.asarray
self.seed_parameter.fill_
next
self._goal
self._functional_goal
self._manager_type.bind_ik_goals
self.ik.disable_tool_pose_tracking
self._solve_ik
self.device_cfg.to_device(previous[None]).contiguous
self.device_cfg.to_device
self.ik.enable_tool_pose_tracking
self.ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy
self.ik.get_active_js(result.js_solution).position[0].detach().cpu
self.ik.get_active_js(result.js_solution).position[0].detach
self.ik.get_active_js
result.success[0].detach().cpu().numpy
result.success[0].detach().cpu
result.success[0].detach
self._rank_ik_candidates
np.zeros_like
report.update
bool
list
candidates[index].tolist
```

## `_ConstrainedPlanner.evaluate_reference`

[定义位置](../../../manipulation/planning.py#L3492)。

输入签名：

```python
def evaluate_reference(self, seed):
```

显式输出分支：

```python
[ConstrainedSolution(row.detach().cpu().numpy().copy(), {'source': 'reference_initialization', 'unchanged_from_seed': True, 'maximum_control_update': 0.0}) for row in seed[0]]
```

读取的对象属性：

```python
row.detach().cpu().numpy().copy
row.detach().cpu().numpy
row.detach().cpu
row.detach
```

写入目标：

```python
row
```

直接调用符号：

```python
ConstrainedSolution
row.detach().cpu().numpy().copy
row.detach().cpu().numpy
row.detach().cpu
row.detach
```

## `_ConstrainedPlanner.bind_reference_candidate`

[定义位置](../../../manipulation/planning.py#L3499)。

输入签名：

```python
def bind_reference_candidate(self, reference):
```

实现说明：

> Bind current obligations and hand commands without changing q(t).

显式输出分支：

```python
(trajectory, candidate_scene, mapping)
```

读取的对象属性：

```python
reference.joint_names
self.joint_names
reference.times
reference.duration
reference.progress_u.copy
reference.progress_u
reference.progress_s.copy
reference.progress_s
mapping.s
self.problem.start_progress
self.problem
self.problem.end_progress
self.scene.clone_for_rollout
self.scene
candidate_scene.gripper
hand.control_timestep_s
hand.bind_clock
record.items
mapping.event_clock.values
mapping.event_clock
reference.times.copy
reference.position.copy
reference.position
reference.velocity.copy
reference.velocity
mapping.u.copy
mapping.u
mapping.s.copy
self.problem.conditions
hand.finger_trajectory
```

写入目标：

```python
mapping
candidate_scene
hand
issues
events
key
value
record
trajectory
```

直接调用符号：

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

## `_ConstrainedPlanner._candidate_queries`

[定义位置](../../../manipulation/planning.py#L3528)。

输入签名：

```python
def _candidate_queries(self, trajectory, ordinary):
```

显式输出分支：

```python
(mapping, times, progress)
```

读取的对象属性：

```python
trajectory.progress_u
trajectory.progress_s
mapping.anchor_parameters
self.anchors
np.unique
np.r_
trajectory.duration
np.interp
mapping.u
mapping.s
np.searchsorted
```

写入目标：

```python
mapping
pairs
times
progress
progress[np.searchsorted(times, pairs[:, 0] * trajectory.duration)]
```

直接调用符号：

```python
_ProgressMap
mapping.anchor_parameters
np.unique
np.interp
np.searchsorted
```

## `_ConstrainedPlanner._candidate_robot_state`

[定义位置](../../../manipulation/planning.py#L3536)。

输入签名：

```python
def _candidate_robot_state(self, trajectory, candidate_scene, times, progress):
```

显式输出分支：

```python
(states, values)
```

读取的对象属性：

```python
trajectory.joint_names.index
trajectory.joint_names
self.joint_names
self.torch.as_tensor
self.torch
trajectory._curve
self.problem
self.pin_kinematics
```

写入目标：

```python
columns
name
states
order
values
```

直接调用符号：

```python
trajectory.joint_names.index
self.torch.as_tensor
trajectory._curve
range
robot_tool_states
```

## `_ConstrainedPlanner._candidate_objective`

[定义位置](../../../manipulation/planning.py#L3546)。

输入签名：

```python
def _candidate_objective(self, trajectory, candidate_scene):
```

实现说明：

> Rank by motion quality and declared task preferences on fixed queries.

显式输出分支：

```python
{'fixed_objective': sum(components.values()), 'objective_components': components}
```

读取的对象属性：

```python
np.linspace
trajectory.duration
self.budget.knots
self.budget
self.budget.interpolation_steps
self._candidate_queries
np.searchsorted
self.torch.no_grad
self.torch
self._candidate_robot_state
self.computation.bounds
self.computation
self.motion_configuration
self.formal_clock
self.torch.as_tensor
self.problem.optimizer_preferences
self.problem
self.budget.constraint_weight
preferences.sum
components.values
```

写入目标：

```python
ordinary
_
times
progress
indices
states
tool
twist
motion
value
preferences
components
```

直接调用符号：

```python
np.linspace
self._candidate_queries
np.searchsorted
self.torch.no_grad
self._candidate_robot_state
motion_cost
tuple
self.torch.as_tensor
time_mean_weights
self.problem.optimizer_preferences
float
preferences.sum
sum
components.values
```

## `_ConstrainedPlanner._candidate_metrics`

[定义位置](../../../manipulation/planning.py#L3564)。

输入签名：

```python
def _candidate_metrics(self, trajectory, candidate_scene, *, record_work):
```

实现说明：

> Evaluate the exact saved polynomial with original task and scene rules.

显式输出分支：

```python
(report, times / trajectory.duration, len(times))
```

读取的对象属性：

```python
self._candidate_queries
self.budget
self.budget.max_validation_points
self.torch.as_tensor
self.torch
self.torch.float64
self.torch.no_grad
self.check_request
self._candidate_robot_state
self.problem.residuals
self.problem
self.problem.verification_issues
q.new_tensor
q.new_empty
self.joint_names
trajectory.duration
terms.items
np.isfinite
residuals.values
candidate_scene.last_query_report
```

写入目标：

```python
mapping
times
progress
tensor
details
channels
states
q
dq
names
_
tool
twist
terms
issues
terms['query_completeness']
terms['process_scene']
residuals
name
value
accepted
report
target
category
report[target]
```

直接调用符号：

```python
self._candidate_queries
_candidate_sample_times
len
MotionPlanningError
self.torch.as_tensor
self.torch.no_grad
self.check_request
self._candidate_robot_state
self.problem.residuals
tensor
self.problem.verification_issues
q.new_tensor
float
bool
_scene_boundary_evaluate
q.new_empty
record_work
maximum
terms.items
all
np.isfinite
residuals.values
deepcopy
```

## `_ConstrainedPlanner._candidate_metrics.maximum`

[定义位置](../../../manipulation/planning.py#L3585)。

输入签名：

```python
def maximum(value):
```

显式输出分支：

```python
float(value.max()) if bool(self.torch.isfinite(value).all()) else float('inf')
```

读取的对象属性：

```python
self.torch.isfinite(value).all
self.torch.isfinite
self.torch
value.max
```

写入目标：

无显式条目。

直接调用符号：

```python
bool
self.torch.isfinite(value).all
self.torch.isfinite
float
value.max
```

## `_ConstrainedPlanner._candidate_geometry`

[定义位置](../../../manipulation/planning.py#L3600)。

输入签名：

```python
def _candidate_geometry(self, trajectory, candidate_scene, *, remaining_points, record_work):
```

实现说明：

> Check continuous entity separation on the candidate's physical clock.
>
> Scene queries evaluate the original polynomial and its release anchors.
> The motion bound only determines where further geometry is needed;
> collision, contact and observation residuals retain scene semantics.

显式输出分支：

```python
report
```

读取的对象属性：

```python
trajectory.progress_u
trajectory.progress_s
SceneMotionBounds.from_scene
self.problem.tool_link
self.problem
self.problem.tool_transform
trajectory.joint_names.index
trajectory.joint_names
self.joint_names
relation.interval
candidate_scene.owners
owner.attachments
relation.release_condition_id
self.problem.start_progress
self.problem.end_progress
mapping.anchor_parameters
trajectory.duration
np.empty
self.budget.validation_spatial_step_m
self.budget
np.isfinite(hard).all
np.isfinite
```

写入目标：

```python
mapping
motion
columns
name
release_progress
owner
relation
release_times
hard
categories
details
released_scene
report
report['hard_scene_residual']
report['scene_categories']
report['scene_components']
```

直接调用符号：

```python
_ProgressMap
SceneMotionBounds.from_scene
trajectory.joint_names.index
mapping.anchor_parameters
np.empty
len
check_interval_geometry
max
np.isfinite(hard).all
np.isfinite
float
```

## `_ConstrainedPlanner._candidate_geometry.query`

[定义位置](../../../manipulation/planning.py#L3621)。

输入签名：

```python
def query(times_s, *, point_budget):
```

显式输出分支：

```python
{'states': [], 'query_points': 0, 'capacity_exhausted': True}
{'states': list(states.values()), 'query_points': len(times), 'capacity_exhausted': False}
```

读取的对象属性：

```python
np.unique
np.r_
np.asarray
np.empty
self.check_request
trajectory.sample
self.torch.as_tensor
self.torch
self.torch.float64
self.torch.no_grad
self._candidate_robot_state
self.problem
position.new_empty
self.joint_names
trajectory.duration
candidate_scene.with_release_history
query_details.items
details.get
np.isfinite
values.detach().cpu().numpy
values.detach().cpu
values.detach
hard.extend
values.tolist
scene_channels.items
value.detach().cpu().numpy().ravel
value.detach().cpu().numpy
value.detach().cpu
value.detach
categories.get
np.isfinite(numbers).all
numbers.max
row.items
states.setdefault
state['pairs'].append
states.values
```

写入目标：

```python
times
q
dq
_
progress
tensor
states
position
quaternion
evidence
scene_channels
query_details
active_scene
values
released_scene
name
entry
previous
details[name]
value
numbers
categories[name]
row
index
key
pair
state
```

直接调用符号：

```python
np.unique
np.asarray
np.empty
len
self.check_request
trajectory.sample
self.torch.as_tensor
self.torch.no_grad
self._candidate_robot_state
_scene_boundary_evaluate
position.new_empty
tensor
record_work
candidate_scene.with_release_history
query_details.items
details.get
np.isfinite
values.detach().cpu().numpy
values.detach().cpu
values.detach
hard.extend
values.tolist
scene_channels.items
value.detach().cpu().numpy().ravel
value.detach().cpu().numpy
value.detach().cpu
value.detach
categories.get
np.isfinite(numbers).all
float
max
numbers.max
row.items
states.setdefault
state['pairs'].append
list
states.values
```

## `_ConstrainedPlanner.evaluate_candidate`

[定义位置](../../../manipulation/planning.py#L3685)。

输入签名：

```python
def evaluate_candidate(self, trajectory, *, record_work, candidate_scene=None, objective=None):
```

实现说明：

> Query the saved curve, its exact joint domain and interval geometry.
>
> ``record_work`` commits started validation attempts and completed query
> points to the request ledger. ``objective`` may contain the fixed score
> already computed for this same curve during native candidate ranking.

显式输出分支：

```python
(trajectory, attempt, points)
```

读取的对象属性：

```python
self.scene
self._candidate_metrics
self._candidate_domain_validation
trajectory.residuals
np.isfinite
residuals.values
self._candidate_geometry
self.budget.max_validation_points
self.budget
attempt.setdefault
categories.get
attempt.setdefault('scene_components', {}).update
attempt.update
self._candidate_objective
validation.update
np.full
parameters.shape
```

写入目标：

```python
self.delivery_candidate
self.last_validation
scene
attempt
parameters
points
validation
residuals
accepted
value
geometry
attempt['continuous_geometry']
validation['continuous_geometry']
residuals['interval:process_scene']
categories
target
category
component
attempt.setdefault(target, {})[f'interval:{component}']
```

直接调用符号：

```python
record_work
self._candidate_metrics
self._candidate_domain_validation
bool
all
np.isfinite
residuals.values
self._candidate_geometry
attempt.setdefault
categories.get
attempt.setdefault('scene_components', {}).update
attempt.update
self._candidate_objective
validation.update
np.full
max
```

## `_ConstrainedPlanner.optimize`

[定义位置](../../../manipulation/planning.py#L3726)。

输入签名：

```python
def optimize(self, seed, *, iterations=None):
```

显式输出分支：

```python
results
```

读取的对象属性：

```python
self.computation.scene
self.computation
self.robot_metadata
self.budget
self.anchor_capacity
row.detach().cpu().numpy
row.detach().cpu
row.detach
self.budget.trajectory_iterations
self.check_request
workspace.pending_cpu_bytes
results.append
self.last_optimizer_diagnostics.append
self.last_optimizer_diagnostics
result.report
self.last_optimizer_runtime.append
self.last_optimizer_runtime
```

写入目标：

```python
results
self.last_optimizer_diagnostics
self.last_optimizer_runtime
row
workspace
result
```

直接调用符号：

```python
_planning_workspace_bytes
solve_trajectory
row.detach().cpu().numpy
row.detach().cpu
row.detach
results.append
self.last_optimizer_diagnostics.append
self.last_optimizer_runtime.append
deepcopy
```

## `_ConstrainedPlanner.select_candidate`

[定义位置](../../../manipulation/planning.py#L3748)。

输入签名：

```python
def select_candidate(self, results, *, record_work):
```

实现说明：

> Score proposed curves, then verify each on the common exact-curve boundary.

显式输出分支：

```python
(result, trajectory, {**report, 'selected_seed_index': index, 'candidates': reports, 'validation_query_points': query_points, 'evaluated_candidate_count': len(reports)})
```

读取的对象属性：

```python
self.seed_strategies
result.report
np.isfinite(result.controls).all
np.isfinite
result.controls
reports.append
self._controls_curve
self._candidate_objective
self.scene
proposals.append
proposals.sort
self.evaluate_candidate
report.update
report['residuals'].values
np.isfinite(values).all
```

写入目标：

```python
proposals
reports
index
result
provenance
trajectory
objective
selected
selected_key
query_points
report
points
values
severity
value
key
self.delivery_candidate
```

直接调用符号：

```python
enumerate
deepcopy
np.isfinite(result.controls).all
np.isfinite
reports.append
float
self._controls_curve
self._candidate_objective
proposals.append
MotionPlanningError
proposals.sort
self.evaluate_candidate
report.update
list
report['residuals'].values
np.isfinite(values).all
sum
max
len
```

## `_ConstrainedPlanner._controls_curve`

[定义位置](../../../manipulation/planning.py#L3782)。

输入签名：

```python
def _controls_curve(self, actions):
```

实现说明：

> Export one active-joint coefficient array with its original event clock.

显式输出分支：

```python
trajectory
```

读取的对象属性：

```python
self.action_horizon
self.formal_clock
self.torch.Tensor
self.torch
actions.detach().cpu().numpy
actions.detach().cpu
actions.detach
np.asarray
basis.hermite_nodes
self.physical_domain._curve_map.start_q
self.physical_domain._curve_map
self.physical_domain
self.physical_domain._curve_map.start_dq
position.shape
self.joint_names
self.problem.events
self.problem
self.progress_map.event_clock
self.progress_map
event_records.append
clock.items
self.progress_map.u.copy
self.progress_map.u
self.progress_map.s.copy
self.progress_map.s
self.problem.conditions
self.finger_trajectory
```

写入目标：

```python
basis
values
times
position
velocity
event_records
event
clock
key
value
trajectory
trajectory.finger_trajectory
trajectory.satisfied_dependencies
```

直接调用符号：

```python
CubicBasis
isinstance
actions.detach().cpu().numpy
actions.detach().cpu
actions.detach
np.asarray
basis.hermite_nodes
len
ValueError
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

## `_ConstrainedPlanner._candidate_domain_validation`

[定义位置](../../../manipulation/planning.py#L3807)。

输入签名：

```python
def _candidate_domain_validation(self, trajectory):
```

实现说明：

> Check the actual saved polynomial, independent of its proposal basis.

显式输出分支：

```python
{'continuous_position': position, 'continuous_velocity': velocity, 'continuous_jerk': jerk, 'cubic_derivatives': derivatives, 'numerical_tolerances': asdict(self.problem.numerical)}
```

读取的对象属性：

```python
self._verification_native.get_cost('cspace').config.joint_limits
self._verification_native.get_cost('cspace').config
self._verification_native.get_cost
self._verification_native
limits.joint_names.index
limits.joint_names
trajectory.joint_names
limits.position[:, columns].detach().cpu().numpy
limits.position[:, columns].detach().cpu
limits.position[:, columns].detach
limits.position
trajectory._curve
limits.velocity[:, columns].detach().cpu().numpy
limits.velocity[:, columns].detach().cpu
limits.velocity[:, columns].detach
limits.velocity
trajectory._curve.derivative
velocity['maximum_violation'].pop
limits.acceleration.detach().cpu().numpy
limits.acceleration.detach().cpu
limits.acceleration.detach
limits.acceleration
limits.jerk[:, columns].detach().cpu().numpy
limits.jerk[:, columns].detach().cpu
limits.jerk[:, columns].detach
limits.jerk
jerk['maximum_violation'].pop
self.joint_names.index
self.joint_names
np.max
np.abs
trajectory.position
self.physical_domain._curve_map.start_q
self.physical_domain._curve_map
self.physical_domain
self.problem.numerical.joint
self.problem.numerical
self.problem
trajectory.velocity
self.physical_domain._curve_map.start_dq
self.problem.numerical.velocity
```

写入目标：

```python
limits
columns
name
bounds
position
velocity_bounds
velocity
velocity['maximum_violation']['velocity']
derivatives
jerk
jerk['maximum_violation']['jerk']
jerk['scope']
entrance_columns
trajectory.residuals
```

直接调用符号：

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
limits.jerk[:, columns].detach().cpu().numpy
limits.jerk[:, columns].detach().cpu
limits.jerk[:, columns].detach
jerk['maximum_violation'].pop
self.joint_names.index
max
float
np.max
np.abs
asdict
```

## `_ConstrainedPlanner.export`

[定义位置](../../../manipulation/planning.py#L3838)。

输入签名：

```python
def export(self, result, native_report):
```

实现说明：

> Deliver the exact curve already qualified during candidate selection.

显式输出分支：

```python
(self.delivery_candidate, 0)
```

读取的对象属性：

```python
self.last_validation
self.delivery_candidate
```

写入目标：

无显式条目。

直接调用符号：

```python
MotionPlanningError
```

## `_verified_dependency_outcomes`

[定义位置](../../../manipulation/planning.py#L3848)。

输入签名：

```python
def _verified_dependency_outcomes(continuation):
```

实现说明：

> Name only dependencies with the executed occurrence's fresh T record.

显式输出分支：

```python
set()
satisfied
```

读取的对象属性：

```python
continuation.get
row.get
evidence.get
np.isfinite
event.get
satisfied.add
```

写入目标：

```python
verified
row
original
events
satisfied
event
evidence
identity
elapsed
dependency
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L3870)。

输入签名：

```python
def _bind_motion_work_ledger(ledger, block, continuation, entrance_consumed):
```

实现说明：

> Start another two-solve allowance only after a verified block boundary.

显式输出分支：

```python
ledger['motion_request']['planning_baseline']
```

读取的对象属性：

```python
ledger.get
continuation['original_binding'].get
old.get
event.get
dict.fromkeys
```

写入目标：

```python
previous
new_request
old
event
verified
ledger['motion_request']
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L3889)。

```python
class MotionPlanningError(ValueError):
```

类字段：

无显式条目。

实现说明：

> A failed bounded request, never a proof that the mathematical problem is infeasible.

## `MotionPlanningError.__init__`

[定义位置](../../../manipulation/planning.py#L3891)。

输入签名：

```python
def __init__(self, reason, message):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
```

写入目标：

```python
self.reason
```

直接调用符号：

```python
super().__init__
super
```

## `MotionBackend`

[定义位置](../../../manipulation/planning.py#L3896)。

```python
class MotionBackend():
```

类字段：

无显式条目。

实现说明：

> Own one robot's reusable IK and constrained trajectory computation.
>
> The robot, device and tensor capacities are constructor inputs. Requests
> replace goals, measured q/dq, scene values and clocks; they cannot resize
> the robot or silently buy another solve. close() is the owner's lifecycle.

## `MotionBackend.__init__`

[定义位置](../../../manipulation/planning.py#L3903)。

输入签名：

```python
def __init__(self, robot, *, robot_validator, budget=None, device='cuda:0'):
```

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
self.robot
self.robot_validator
self.device
self.budget
self._planner
self._closed
self._busy
self._request_owner
```

直接调用符号：

```python
deepcopy
PlanningBudget
Lock
```

## `MotionBackend._claim`

[定义位置](../../../manipulation/planning.py#L3911)。

输入签名：

```python
def _claim(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._busy.acquire
self._busy
```

写入目标：

```python
self._request_owner
```

直接调用符号：

```python
self._busy.acquire
MotionPlanningError
get_ident
```

## `MotionBackend._release`

[定义位置](../../../manipulation/planning.py#L3916)。

输入签名：

```python
def _release(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._request_owner
self._busy.release
self._busy
```

写入目标：

```python
self._request_owner
```

直接调用符号：

```python
get_ident
RuntimeError
self._busy.release
```

## `MotionBackend.bind_request`

[定义位置](../../../manipulation/planning.py#L3922)。

输入签名：

```python
def bind_request(self, problem, scene, *, budget, continuation=None, goal_identities=None):
```

显式输出分支：

```python
self._planner
```

读取的对象属性：

```python
self._request_owner
self._closed
self.budget
self._planner
self.robot
self.device
self.robot_validator
self._planner.rebind
```

写入目标：

```python
name
self._planner
```

直接调用符号：

```python
get_ident
MotionPlanningError
getattr
_ConstrainedPlanner
self._planner.rebind
```

## `MotionBackend.plan`

[定义位置](../../../manipulation/planning.py#L3938)。

输入签名：

```python
def plan(self, problem, scene, *, continuation=None, request_ledger=None, rebind=None, cancelled=None, deadline=None, joint_reference=None):
```

显式输出分支：

```python
plan_trajectory(problem, scene, self.robot, robot_validator=self.robot_validator, budget=self.budget, device=self.device, continuation=continuation, request_ledger=request_ledger, rebind=rebind, backend=self, cancelled=cancelled, deadline=deadline, joint_reference=joint_reference)
```

读取的对象属性：

```python
self.robot
self.robot_validator
self.budget
self.device
```

写入目标：

无显式条目。

直接调用符号：

```python
plan_trajectory
```

## `MotionBackend.close`

[定义位置](../../../manipulation/planning.py#L3945)。

输入签名：

```python
def close(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._claim
self._planner
self._planner.close
self._release
```

写入目标：

```python
self._planner
self._closed
```

直接调用符号：

```python
self._claim
self._planner.close
self._release
```

## `plan_trajectory`

[定义位置](../../../manipulation/planning.py#L3958)。

输入签名：

```python
def plan_trajectory(problem, scene, robot, *, robot_validator, budget=None, device='cuda:0', continuation=None, request_ledger=None, rebind=None, backend=None, cancelled=None, deadline=None, joint_reference=None):
```

实现说明：

> Generate one fully checked candidate within the original request budget.
>
> ``request_ledger`` continues failed planning attempts. ``continuation``
> carries actual execution facts. An explicit ``rebind`` rebuilds the soft
> schedule of remaining work or creates selected retry occurrences; ordinary
> continuation preserves the original frozen clock. Every request admitted
> including initialization failure carries the cumulative ledger. No GPU stage is
> claimed to be preemptible. A candidate qualifies only after all required
> work completes within budget; later improvement exhaustion preserves that
> independently stored delivery record.

显式输出分支：

```python
outcome
outcome
outcome
outcome
deliver_incumbent()
outcome
deliver_incumbent()
outcome
```

读取的对象属性：

```python
time.monotonic
scene.clone_for_rollout
continuation.get
continuation['original_binding'].get
diagnostics.update
diagnostics.pop
problem.events
rebind.get
problem.bound_motion_block
original_budget.search_seconds
motion_baseline.get
scene.gripper
scene.gripper.control_timestep_s
joint_reference.duration
backend._claim
scene.gripper.commands
backend.bind_request
budget.ik_seeds
budget.trajectory_seeds
planner.budget.maximum_trajectory_dt
planner.budget
scene.describe
planner.scene_workspace_bytes
planner.scene_cpu_workspace_bytes
planner.scene_gpu_workspace_bytes
planner.anchor_fk_workspace_bytes
planner.anchor_capacity
planner._verification_native.get_cost('cspace').config.joint_limits
planner._verification_native.get_cost('cspace').config
planner._verification_native.get_cost
planner._verification_native
limits.joint_names
getattr(limits, name).detach().cpu().numpy().tolist
getattr(limits, name).detach().cpu().numpy
getattr(limits, name).detach().cpu
getattr(limits, name).detach
planner.initial_state_diagnostics
scene.unknown_policy
planner.bind_reference_candidate
candidate.duration
planner.evaluate_candidate
candidate.times.tolist
candidate.times
planner.last_validation
candidate_scene.unknown_policy
reference_attempt.get('scene_components', {}).items
reference_attempt.get
name.startswith
np.isfinite
planner.budget.knots
planner.budget.interpolation_steps
planner.budget.interpolation_dt
problem.source_time_map
planner.progress_map
planner.reference_seeds
planner.ik.config.exit_early
planner.ik.config
planner.ik
planner.reference_ik_distinct_counts
planner.initial_seed_strategies
planner.reference_ik_queries
planner.reference_ik_chains
planner.joint_reference_report
planner.evaluate_reference
planner.optimize
planner.main_trajectory_iterations
diagnostics['attempts'].append
original_budget.maximum_trajectory_solves
attempt['collision_residuals'].values
attempt['coverage_residuals'].values
budget.search_seconds
planner.ompl_seed
report.get
seeds.shape
planner.route
error.reason
type(error).__name__
traceback.format_exc
planner.goal_ik_candidates
backend._release
backend.close
ledger['binding_records'].append
outcome.execution_binding
outcome.execution_binding.get
outcome.success
diagnostics.get
```

写入目标：

```python
started
prior
planner
ledger
trajectory
incumbent
initial_seconds
motion_ready
motion_baseline
owned_backend
backend_claimed
diagnostics
diagnostics['wallclock']
outcome
stage
scene
previous
original_budget
execution_fields
ledger_input
original_ledger
name
ledger_input[name]
diagnostics['continuation_facts_pending_validation']
ledger['requires_continuation']
consumed
entrance_consumed
request_budget
consumed['binding_attempts']
problem
current
current['consumed_planning']
budget
diagnostics['consumed_planning']
prior['history']
prior['timing_events']
satisfied_dependencies
block_record
diagnostics['motion_block']
issues
backend
consumed['initializations']
occurrences
event
scene.gripper.commands
row
planner.check_request
planner.request_work
planner.remaining_request_seconds
planner.satisfied_dependencies
diagnostics['joint_state_computation']
limits
diagnostics['verification_joint_limits']
consumed['validation_queries']
diagnostics['initial_state']
diagnostics['failure']
candidate
candidate_scene
candidate_map
diagnostics['event_time_checks']
reference_attempt
_
diagnostics['original_reference_candidate']
diagnostics['joint_reference']
diagnostics['reference_clock']
policy_failure
missing
value
grid
source_seconds
source_duration
consumed['ik_batches']
seeds
ik_success
diagnostics['functional_ik_candidates']
reference_result
native_result
attempt
diagnostics['reference_candidate']
observation_required
recovery
consumed['trajectory_solves']
collision_failed
coverage_failed
search_started
recovered
report
consumed['search_seconds']
consumed['search_queries']
diagnostics['ompl']
planner.seed_strategies
diagnostics['improvement_end']
diagnostics['diagnostic_trajectory']
diagnostics['diagnostic_candidate_scope']
outcome.success
outcome.trajectory
outcome.execution_binding
diagnostics[name]
diagnostics['elapsed_seconds']
ledger['consumed_planning']['planning_seconds']
request_seconds
diagnostics['wall_time_overrun_s']
diagnostics['planning_budget_exhausted']
outcome.request_ledger
outcome.execution_binding['consumed_planning']
outcome.execution_binding['request_ledger']
```

直接调用符号：

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
candidate.times.tolist
qualify_candidate
reference_attempt.get('scene_components', {}).items
reference_attempt.get
name.startswith
np.isfinite
_fixed_spline_grid
float
planner.reference_seeds
bool
planner.evaluate_reference
select_candidate
qualify_native
planner.optimize
diagnostics['attempts'].append
attempt['collision_residuals'].values
attempt['coverage_residuals'].values
planner.ompl_seed
report.get
deliver_incumbent
type
traceback.format_exc
hasattr
backend._release
backend.close
ledger['binding_records'].append
outcome.execution_binding.get
diagnostics.get
```

## `plan_trajectory.mark_wallclock`

[定义位置](../../../manipulation/planning.py#L3983)。

输入签名：

```python
def mark_wallclock(name):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
time.time_ns
time.monotonic
```

写入目标：

```python
diagnostics['wallclock']['events'][name]
```

直接调用符号：

```python
time.time_ns
time.monotonic
```

## `plan_trajectory.check`

[定义位置](../../../manipulation/planning.py#L4027)。

输入签名：

```python
def check(next_stage=None):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
time.monotonic
motion_baseline.get
consumed.items
```

写入目标：

```python
consumed['planning_seconds']
relative
name
value
```

直接调用符号：

```python
cancelled
MotionPlanningError
time.monotonic
motion_baseline.get
consumed.items
_check_work_budget
```

## `plan_trajectory.record_validation_work`

[定义位置](../../../manipulation/planning.py#L4038)。

输入签名：

```python
def record_validation_work(*, passes=0, points=0):
```

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
consumed['validation_passes']
consumed['validation_queries']
```

直接调用符号：

```python
check
```

## `plan_trajectory.remaining_request_seconds`

[定义位置](../../../manipulation/planning.py#L4146)。

输入签名：

```python
def remaining_request_seconds():
```

显式输出分支：

```python
max(0.0, remaining if deadline is None else min(remaining, deadline - time.monotonic()))
```

读取的对象属性：

```python
original_budget.planning_seconds
motion_baseline.get
time.monotonic
```

写入目标：

```python
remaining
```

直接调用符号：

```python
check
motion_baseline.get
max
min
time.monotonic
```

## `plan_trajectory.qualify_candidate`

[定义位置](../../../manipulation/planning.py#L4184)。

输入签名：

```python
def qualify_candidate(trajectory, validation, attempt, *, binding_scene=None):
```

显式输出分支：

```python
False
True
```

读取的对象属性：

```python
problem.events
event.timing
ledger['timing_events'].append
binding.update
mechanism.directions
bound_scene.describe
diagnostics.get
time.monotonic
```

写入目标：

```python
bound_scene
binding
event
row
retry_goal_ids
retry_direction_ids
column
index
mechanism
value
name
binding['verification_joint_limits']
record
record['qualified_elapsed_seconds']
incumbent
```

直接调用符号：

```python
ValueError
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
bound_scene.describe
diagnostics.get
check
time.monotonic
mark_wallclock
```

## `plan_trajectory.deliver_incumbent`

[定义位置](../../../manipulation/planning.py#L4243)。

输入签名：

```python
def deliver_incumbent():
```

实现说明：

> Restore the selected curve's clock, evidence and binding as one record.

显式输出分支：

```python
outcome
```

读取的对象属性：

```python
diagnostics.update
incumbent['trajectory'].executable_until
incumbent['trajectory'].next_verification
```

写入目标：

```python
outcome.success
outcome.trajectory
outcome.execution_binding
```

直接调用符号：

```python
diagnostics.update
deepcopy
```

## `plan_trajectory.qualify_native`

[定义位置](../../../manipulation/planning.py#L4257)。

输入签名：

```python
def qualify_native(result, attempt):
```

实现说明：

> Export immediately while the selected native validation still belongs to this curve.

显式输出分支：

```python
None
```

读取的对象属性：

```python
planner.export
planner.last_validation
```

写入目标：

```python
curve
_
```

直接调用符号：

```python
check
planner.export
qualify_candidate
```

## `plan_trajectory.select_candidate`

[定义位置](../../../manipulation/planning.py#L4334)。

输入签名：

```python
def select_candidate(result):
```

显式输出分支：

```python
planner.select_candidate(result, record_work=record_validation_work)
```

读取的对象属性：

```python
planner.select_candidate
```

写入目标：

无显式条目。

直接调用符号：

```python
planner.select_candidate
```
