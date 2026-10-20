# `manipulation/planning.py` 实现契约

代码基线：`cf7182cbf3888d9db2e7553df47dce9a0b1c339f`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_fixed_spline_grid`

[定义位置](../../../manipulation/planning.py#L53)。

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

## `PlanningBudget`

[定义位置](../../../manipulation/planning.py#L88)。

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

## `PlanningBudget.__post_init__`

[定义位置](../../../manipulation/planning.py#L137)。

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
_fixed_spline_grid
```

## `JointTrajectory`

[定义位置](../../../manipulation/planning.py#L190)。

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

[定义位置](../../../manipulation/planning.py#L214)。

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

[定义位置](../../../manipulation/planning.py#L218)。

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

[定义位置](../../../manipulation/planning.py#L222)。

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

[定义位置](../../../manipulation/planning.py#L230)。

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

[定义位置](../../../manipulation/planning.py#L235)。

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

[定义位置](../../../manipulation/planning.py#L254)。

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

[定义位置](../../../manipulation/planning.py#L357)。

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

[定义位置](../../../manipulation/planning.py#L370)。

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

[定义位置](../../../manipulation/planning.py#L383)。

```python
class _PlanningWorkspace():
```

类字段：

```python
cpu_shared_bytes: int
cpu_live_bytes: int
gpu_eager_bytes: int
gpu_graph_bytes: int
anchor_fk_bytes: int
trajectory_graph: bool
```

实现说明：

> Scene capacity plus one live query, with optional native graph pools.

## `_PlanningWorkspace.cpu_bytes`

[定义位置](../../../manipulation/planning.py#L394)。

输入签名：

```python
def cpu_bytes(self):
```

显式输出分支：

```python
self.cpu_shared_bytes + self.cpu_live_bytes
```

读取的对象属性：

```python
self.cpu_shared_bytes
self.cpu_live_bytes
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `_PlanningWorkspace.gpu_bytes`

[定义位置](../../../manipulation/planning.py#L398)。

输入签名：

```python
def gpu_bytes(self):
```

显式输出分支：

```python
self.gpu_eager_bytes + (self.gpu_graph_bytes if self.trajectory_graph else 0)
```

读取的对象属性：

```python
self.gpu_eager_bytes
self.trajectory_graph
self.gpu_graph_bytes
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `_PlanningWorkspace.total_bytes`

[定义位置](../../../manipulation/planning.py#L402)。

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

[定义位置](../../../manipulation/planning.py#L408)。

输入签名：

```python
def _planning_workspace_bytes(scene, robot_metadata, budget, anchor_capacity):
```

实现说明：

> Return CPU/GPU scene capacity, anchor FK and graph eligibility.
> 
> The main native rollout retains its original state-sampling horizon. Each trajectory
> manager evaluates the required anchors with a separate native FK model,
> then folds their penalties into that rollout's existing final cost column.

显式输出分支：

```python
_PlanningWorkspace(int(cpu_shared_bytes), int(cpu_live_bytes), int(gpu_eager_bytes), int(2 * graph_live), int(anchor_fk_bytes), bool(trajectory_graph))
```

读取的对象属性：

```python
budget.knots
budget.interpolation_steps
budget.trajectory_seeds
budget.interpolation_buffer_size
budget.max_validation_points
scene.required_progress_anchors
scene.estimated_workspace_allocation
budget.max_scene_gpu_workspace_bytes
```

写入目标：

```python
horizon
allocation_shapes
scene_factor
allocations
copies
batch
length
_
gpu_eager_bytes
value
cpu_shared_bytes
cpu_live_bytes
graph_live
guidance
graph_scene_bytes
trajectory_graph
dof
links
spheres
self_blocks
self_bytes_per_state
bytes_per_state
anchor_fk_bytes
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L500)。

输入签名：

```python
def _require_planning_host_memory(cpu_live_bytes, *, encoded_record_bytes_upper, max_host_rss_bytes):
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

[定义位置](../../../manipulation/planning.py#L516)。

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
> CPU source geometry is already host-admitted and never enters this check.
> If the extra graph pools do not fit but the eager solver does, retain the
> existing eager path with the same geometry, samples and optimizer work.

显式输出分支：

```python
eager
workspace
```

读取的对象属性：

```python
workspace.trajectory_graph
```

写入目标：

```python
error
eager
eager_error
```

直接调用符号：

```python
refusal
replace
```

## `_require_planning_memory.refusal`

[定义位置](../../../manipulation/planning.py#L531)。

输入签名：

```python
def refusal(candidate):
```

显式输出分支：

```python
error
None
```

读取的对象属性：

```python
candidate.gpu_bytes
candidate.anchor_fk_bytes
```

写入目标：

无显式条目。

直接调用符号：

```python
require_device_memory
```

## `_bounded_robot_configuration`

[定义位置](../../../manipulation/planning.py#L554)。

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

[定义位置](../../../manipulation/planning.py#L590)。

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

[定义位置](../../../manipulation/planning.py#L599)。

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

[定义位置](../../../manipulation/planning.py#L607)。

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

[定义位置](../../../manipulation/planning.py#L633)。

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

[定义位置](../../../manipulation/planning.py#L657)。

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

[定义位置](../../../manipulation/planning.py#L671)。

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

[定义位置](../../../manipulation/planning.py#L701)。

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
np.float32
budget.interpolation_dt
budget.knots
budget.maximum_trajectory_dt
budget.interpolation_steps
np.nextafter
np.inf
(budget.interpolation_steps - 1).bit_length
np.ceil
(needed - 1).bit_length
progress.copy
```

写入目标：

```python
source
seconds
progress
source_duration
output_limit
maximum
proposed_dt
native_dt
_
knot_dt
needed
subdivisions
output_dt
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

[定义位置](../../../manipulation/planning.py#L749)。

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

[定义位置](../../../manipulation/planning.py#L827)。

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

[定义位置](../../../manipulation/planning.py#L847)。

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

[定义位置](../../../manipulation/planning.py#L861)。

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

[定义位置](../../../manipulation/planning.py#L945)。

```python
class ContinuationBindingError(ValueError):
```

类字段：

无显式条目。

实现说明：

> Remaining work lacks facts, budget or a compatible original clock.

## `_obligation_contract`

[定义位置](../../../manipulation/planning.py#L955)。

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

[定义位置](../../../manipulation/planning.py#L993)。

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

[定义位置](../../../manipulation/planning.py#L1026)。

```python
class PlanningWorkExhausted(ValueError):
```

类字段：

无显式条目。

实现说明：

> The original request has no remaining admitted computation budget.

## `_planning_work_budget`

[定义位置](../../../manipulation/planning.py#L1035)。

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

[定义位置](../../../manipulation/planning.py#L1041)。

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

[定义位置](../../../manipulation/planning.py#L1119)。

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

[定义位置](../../../manipulation/planning.py#L1137)。

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

[定义位置](../../../manipulation/planning.py#L1491)。

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

[定义位置](../../../manipulation/planning.py#L1566)。

输入签名：

```python
def _progress_for_cubic_boundaries(progress_map, knots, *, stop_progress, anchors):
```

实现说明：

> Return the saved progress map and its shared original-path parameter.
> 
> The cubic retains a stopped terminal interval and explicit verification
> stops. Build a monotone scalar reference clock for those boundaries:
> ``z(new_u)`` parameterizes the ORIGINAL
> path, and the returned skill progress is ``old_s(z(new_u))``. OMPL joint
> initializers must use that same z, including inside old progress pauses.
> 
> Four equal coefficients create a complete constant cubic interval with
> zero first/second derivatives on either side. Only explicit verification
> stops add these intervals; ordinary path vertices remain ordinary points.
> Every original path vertex and mandatory progress anchor is retained.
> 
> The saved representation stays piecewise linear. For a uniform cubic,
> ``|d2z/d(native_t)2| <= max|diff(coefficients, 2)|``. Splitting each native
> interval into L pieces bounds linearization error by D2/(8*L**2). The
> normalized-parameter bound 1e-6 is NOT a physical skill tolerance. Native
> and exported process checks both consume this same saved finite map.

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

[定义位置](../../../manipulation/planning.py#L1671)。

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

## `_hermite_at_parameters`

[定义位置](../../../manipulation/planning.py#L1679)。

输入签名：

```python
def _hermite_at_parameters(position, velocity, dt, counts, parameters):
```

实现说明：

> Evaluate q and dq at exact u values without interpolating FK poses.
> 
> The native degree-three samples include every polynomial breakpoint.
> Adjacent q/dq Hermite evaluation is therefore exact up to floating-point
> arithmetic.  Counts are per seed and exclude padded final-state entries.

显式输出分支：

```python
(q.to(position.dtype), dq.to(velocity.dtype))
```

读取的对象属性：

```python
position.shape
parameters.to
torch.float64
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
position.dtype
dq.to
velocity.dtype
```

写入目标：

```python
batch
_
dof
coordinates
left
fraction
gather
q0
q1
v0
v1
delta
a
q
dq
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L1706)。

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

[定义位置](../../../manipulation/planning.py#L1717)。

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

[定义位置](../../../manipulation/planning.py#L1733)。

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

[定义位置](../../../manipulation/planning.py#L1758)。

```python
class ProcessConstraint(ob.Constraint):
```

类字段：

无显式条目。

## `_ompl_stage.ProcessConstraint.__init__`

[定义位置](../../../manipulation/planning.py#L1759)。

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

[定义位置](../../../manipulation/planning.py#L1762)。

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

[定义位置](../../../manipulation/planning.py#L1765)。

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

[定义位置](../../../manipulation/planning.py#L1790)。

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

[定义位置](../../../manipulation/planning.py#L1793)。

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

[定义位置](../../../manipulation/planning.py#L1824)。

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

[定义位置](../../../manipulation/planning.py#L1827)。

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

[定义位置](../../../manipulation/planning.py#L1883)。

```python
class ForwardValidator(ob.MotionValidator):
```

类字段：

无显式条目。

## `_ompl_stage.ForwardValidator.__init__`

[定义位置](../../../manipulation/planning.py#L1884)。

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

[定义位置](../../../manipulation/planning.py#L1887)。

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

[定义位置](../../../manipulation/planning.py#L1948)。

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

[定义位置](../../../manipulation/planning.py#L1971)。

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

[定义位置](../../../manipulation/planning.py#L1994)。

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

## `_project_boundary`

[定义位置](../../../manipulation/planning.py#L2067)。

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

[定义位置](../../../manipulation/planning.py#L2081)。

```python
class BoundaryConstraint(ob.Constraint):
```

类字段：

无显式条目。

## `_project_boundary.BoundaryConstraint.__init__`

[定义位置](../../../manipulation/planning.py#L2082)。

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

[定义位置](../../../manipulation/planning.py#L2085)。

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

[定义位置](../../../manipulation/planning.py#L2088)。

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

[定义位置](../../../manipulation/planning.py#L2104)。

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

[定义位置](../../../manipulation/planning.py#L2119)。

```python
class WorldQuaternionVJP(torch.autograd.Function):
```

类字段：

无显式条目。

## `_orientation_vjp_type.WorldQuaternionVJP.forward`

[定义位置](../../../manipulation/planning.py#L2121)。

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

[定义位置](../../../manipulation/planning.py#L2129)。

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

[定义位置](../../../manipulation/planning.py#L2145)。

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

[定义位置](../../../manipulation/planning.py#L2155)。

```python
class NativeGradientLayout(torch.autograd.Function):
```

类字段：

无显式条目。

## `_native_gradient_layout_type.NativeGradientLayout.forward`

[定义位置](../../../manipulation/planning.py#L2157)。

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

[定义位置](../../../manipulation/planning.py#L2161)。

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

## `_verification_stop_preference`

[定义位置](../../../manipulation/planning.py#L2168)。

输入签名：

```python
def _verification_stop_preference(progress, joint_velocity, stop_progress, *, trajectory_time=None):
```

实现说明：

> Squared speed at each declared stop's first physical occurrence.
> 
> A verification stop commands zero velocity. Its numerical acceptance
> band still belongs exclusively to PlanningProblem.residuals(). Keeping
> this optimizer objective active inside that band avoids terminating on
> its floating-point edge; ordinary motion receives no added objective.

显式输出分支：

```python
torch.where(stopped, joint_velocity.square().sum(-1), torch.zeros_like(progress))
```

读取的对象属性：

```python
torch.zeros_like
torch.bool
np.unique
torch.where
joint_velocity.square().sum
joint_velocity.square
```

写入目标：

```python
stopped
value
```

直接调用符号：

```python
torch.zeros_like
np.unique
_first_event_query
torch.where
joint_velocity.square().sum
joint_velocity.square
```

## `_formal_native_clock`

[定义位置](../../../manipulation/planning.py#L2187)。

输入签名：

```python
def _formal_native_clock(trajopt):
```

实现说明：

> Bind hand, rollout and dense export to one unchanged physical clock.

显式输出分支：

```python
{**grid, 'source': 'instantaneous-boundary cubic on unchanged exact float32 clock'}
```

读取的对象属性：

```python
trajopt.config
config.minimum_trajectory_dt
config.maximum_trajectory_dt
trajopt.transition_model.control_space
trajopt.transition_model
ControlSpace.BSPLINE_3
trajopt.action_horizon
trajopt.interpolation_steps
config.interpolation_dt
```

写入目标：

```python
config
control
grid
```

直接调用符号：

```python
ValueError
_fixed_spline_grid
```

## `_joint_state_sample_times`

[定义位置](../../../manipulation/planning.py#L2208)。

输入签名：

```python
def _joint_state_sample_times(joint_state, counts):
```

实现说明：

> Round physical time once, without a normalized-float32 roundtrip.

显式输出分支：

```python
(indices.to(torch.float64) * joint_state.dt.reshape(-1, 1).to(torch.float64)).to(position.dtype)
```

读取的对象属性：

```python
joint_state.position
torch.minimum
torch.arange
position.shape
position.device
(indices.to(torch.float64) * joint_state.dt.reshape(-1, 1).to(torch.float64)).to
indices.to
torch.float64
joint_state.dt.reshape(-1, 1).to
joint_state.dt.reshape
joint_state.dt
position.dtype
```

写入目标：

```python
position
indices
```

直接调用符号：

```python
torch.minimum
torch.arange
(indices.to(torch.float64) * joint_state.dt.reshape(-1, 1).to(torch.float64)).to
indices.to
joint_state.dt.reshape(-1, 1).to
joint_state.dt.reshape
```

## `_scene_boundary_evaluate`

[定义位置](../../../manipulation/planning.py#L2222)。

输入签名：

```python
def _scene_boundary_evaluate(scene, problem, progress_map, spheres, position, quaternion, progress, joint_position, joint_names, *, trajectory_time, normalized_time, query_counts=None, component_diagnostics=None, query_role='acceptance', scene_evidence_out=None, optimizer_candidate_evidence=False):
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
progress.shape
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

[定义位置](../../../manipulation/planning.py#L2237)。

输入签名：

```python
def evaluate(*args, **kwargs):
```

显式输出分支：

```python
scene.evaluate(*args, **kwargs, query_role=query_role)
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

## `_scene_boundary_evaluate.expanded`

[定义位置](../../../manipulation/planning.py#L2341)。

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

## `_process_manager_type`

[定义位置](../../../manipulation/planning.py#L2385)。

输入签名：

```python
def _process_manager_type(problem, scene, progress_map, anchors, budget, *, clock=None, seed_parameter=None, robot_dict=None, phase='feasibility', functional_constraints=None):
```

实现说明：

> Build V2's documented manager extension after GPU use is requested.
> 
> Configuration construction must return the derived type and class_type;
> registering a custom name alone is not dispatched by V2's base manager.
> Only constraint_cfg enables this extension. Built-in joint constraints
> remain intact; optimizer and metrics share the same frozen online model.

显式输出分支：

```python
ProcessCostManagerCfg
```

读取的对象属性：

```python
problem.functional_constraint_keys
np.empty
manager._query_counts
manager._process_mode
```

写入目标：

```python
orientation_vjp
native_layout
functional_constraints
managers
collision_witnesses
ProcessCostManagerCfg.bind_request
ProcessCostManagerCfg.add_collision_witnesses
ProcessCostManagerCfg.query_anchor_parameters
ProcessCostManagerCfg.query_diagnostics
manager
```

直接调用符号：

```python
_orientation_vjp_type
_native_gradient_layout_type
FunctionalAugmentedLagrangian
np.empty
staticmethod
deepcopy
```

## `_process_manager_type.query_anchor_parameters`

[定义位置](../../../manipulation/planning.py#L2415)。

输入签名：

```python
def query_anchor_parameters():
```

显式输出分支：

```python
pairs
```

读取的对象属性：

```python
progress_map.anchor_parameters
np.column_stack
np.interp
progress_map.u
progress_map.s
np.concatenate
np.argsort
```

写入目标：

```python
pairs
extra
```

直接调用符号：

```python
progress_map.anchor_parameters
len
np.column_stack
np.interp
np.concatenate
np.argsort
```

## `_process_manager_type.ProcessCostManager`

[定义位置](../../../manipulation/planning.py#L2425)。

```python
class ProcessCostManager(RobotCostManager):
```

类字段：

无显式条目。

## `_process_manager_type.ProcessCostManager.initialize_from_config`

[定义位置](../../../manipulation/planning.py#L2426)。

输入签名：

```python
def initialize_from_config(self, config, transition_model, scene_collision_checker=None, **kwargs):
```

显式输出分支：

```python
None
None
```

读取的对象属性：

```python
super().initialize_from_config
config.process_mode
managers.append
self._process_mode
transition_model.robot_model
scene.clone_for_rollout
transition_model.joint_names
self._process_mode.startswith
transition_model.robot_model.tool_frames.index
transition_model.robot_model.tool_frames
problem.tool_link
self.bind_request_data
transition_model.robot_model.config
```

写入目标：

```python
config
self._process_mode
self._dense_counts
self._scene_evidence
self._optimizer_evidence
self._query_counts
transition_model.robot_model.compute_jacobian
self._scene
self._joint_names
self._tool_index
self._anchor_fk
```

直接调用符号：

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
```

## `_process_manager_type.ProcessCostManager.bind_request_data`

[定义位置](../../../manipulation/planning.py#L2466)。

输入签名：

```python
def bind_request_data(self):
```

实现说明：

> Replace request values in fixed-capacity native manager buffers.

显式输出分支：

```python
None
```

读取的对象属性：

```python
scene.clone_for_rollout
self._query_counts.clear
self._query_counts
self._query_counts.update
self._process_mode
self._process_mode.startswith
progress_map.u
np.r_
np.arange
progress_map.s
np.full
budget.max_task_anchors
np.concatenate
np.repeat
self.device_cfg.to_device
self.device_cfg
previous.shape
tensor.shape
previous.copy_
torch.as_tensor
torch.float64
self.device_cfg.device
self._anchor_query_u.shape
self._anchor_query_u
query_u.shape
self._anchor_query_u.copy_
functional_constraints.bind
self._anchor_u
```

写入目标：

```python
self._scene
self._dense_counts
self._scene_evidence
self._optimizer_evidence
pad
keys
values
pairs
anchor_count
self._anchor_count
self._anchor_valid
name
value
tensor
previous
query_u
self._anchor_query_u
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L2515)。

输入签名：

```python
def set_dense_counts(self, counts):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
counts.reshape(-1).long
counts.reshape
```

写入目标：

```python
self._dense_counts
```

直接调用符号：

```python
counts.reshape(-1).long
counts.reshape
```

## `_process_manager_type.ProcessCostManager.optimization_constraints`

[定义位置](../../../manipulation/planning.py#L2520)。

输入签名：

```python
def optimization_constraints(self):
```

显式输出分支：

```python
functional_constraints if self._process_mode == 'trajectory_optimizer' else None
```

读取的对象属性：

```python
self._process_mode
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `_process_manager_type.ProcessCostManager.optimizer_evidence`

[定义位置](../../../manipulation/planning.py#L2523)。

输入签名：

```python
def optimizer_evidence(self, trajectories):
```

实现说明：

> Read hard evidence and objective from this same rollout.
> 
> Functional multipliers and centering/command preferences are
> objectives. Original task and observed-scene residuals determine
> sampled eligibility; final metrics use the acceptance spacing.

显式输出分支：

```python
(eligible, feasible, signed, fixed_objective)
```

读取的对象属性：

```python
trajectories.costs_and_constraints
constraints.constraints
self._optimizer_evidence
collection.names
collection.values
collection.names.index
source.new_zeros
source.shape
constraints.costs
constraints.hybrid_costs_constraints
costs.names
costs.values
value.detach().flatten
value.detach
torch.isfinite(detached).all
torch.isfinite
(detached <= 0).all
detached.sum
```

写入目标：

```python
constraints
collection
source
feasible
signed
eligible
scene_source
objective_names
fixed_objective
costs
hard
name
value
detached
finite
valid
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L2565)。

输入签名：

```python
def bind_scene_evidence(self, metrics):
```

实现说明：

> Attach the last query's tensor only when its constraint is identical.
> 
> Native CostCollection.merge retains the constraint tensor object.
> Shape equality would not distinguish another candidate/seed batch;
> object identity does, without storing or comparing complete states.
> RolloutMetrics.clone already clones a tensor-valued debug field.

显式输出分支：

无显式条目。

读取的对象属性：

```python
metrics.costs_and_constraints.constraints
metrics.costs_and_constraints
self._scene_evidence
constraints.names
constraints.values
constraints.names.index
evidence.detach().clone
evidence.detach
```

写入目标：

```python
constraints
source
evidence
metrics.debug
```

直接调用符号：

```python
RuntimeError
constraints.names.index
evidence.detach().clone
evidence.detach
```

## `_process_manager_type.ProcessCostManager.compute_costs`

[定义位置](../../../manipulation/planning.py#L2584)。

输入签名：

```python
def compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
```

显式输出分支：

```python
super().compute_costs(state, cost_collection, goal, **kwargs)
collection
collection
```

读取的对象属性：

```python
self._process_mode
super().compute_costs
self._scene
self._process_mode.startswith
state.joint_state.position.shape
state.joint_state.position
state.joint_state
self._dense_counts
torch.full
torch.long
state.joint_state.position.device
state.joint_state.velocity
q.shape
state.tool_poses
orientation_vjp.apply
tool.quaternion.detach
tool.quaternion
state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach
state.cuda_robot_model_state.tool_jacobians
state.cuda_robot_model_state
native_layout.apply
tool.position
self._tool_index
direction.kind
problem.mechanisms
mechanism.directions
torch.einsum
state.robot_spheres
q.device
(torch.arange(horizon, device=q.device, dtype=q.dtype)[None, :] / (counts[:, None] - 1)).clamp
torch.arange
q.dtype
self._map_u
self._map_s
self._anchor_u[:, :self._anchor_count].expand
self._anchor_u
self._anchor_count
self._anchor_s[:, :self._anchor_count].expand
self._anchor_s
self._anchor_valid
torch.ones_like
torch.bool
self._anchor_query_u[:, :self._anchor_count].expand
self._anchor_query_u
(query_anchor_u * active_clock['duration']).to
state.joint_state.dt
JointState.from_position
anchor_q.contiguous
self._joint_names
self._anchor_fk.compute_kinematics
self._anchor_fk
anchor_state.tool_poses.quaternion.detach
anchor_state.tool_poses.quaternion
anchor_state.tool_poses
anchor_state.tool_jacobians[..., 3:, :].detach
anchor_state.tool_jacobians
anchor_state.tool_poses.position
torch.cat
anchor_state.robot_spheres
torch.argsort
problem.transform_tool_velocity
problem.transform_tool_pose
problem.transform_robot_spheres
problem.residuals
process_terms.items
name.startswith
torch.zeros_like
self._query_counts
process_value.square
problem.functional_inequalities
functional_constraints.terms(signed).values
functional_constraints.terms
q.new_zeros
budget.constraint_weight
collection.add
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
problem.stop_progress
value.gather
(value[:, horizon:] * anchor_valid).sum
(regular + folded).unsqueeze
self._optimizer_evidence
scene_evidence_out[category].gather
(evidence[:, horizon:] * anchor_valid).sum
channels.append
torch.stack
```

写入目标：

```python
context
active_scene
active_map
active_clock
batch
horizon
counts
sample_times
collection
q
dq
_
tool
quaternion
position
need_rates
mechanism
direction
twist
spheres
u
progress
anchor_u
anchor_s
anchor_valid
query_anchor_u
anchor_times
anchor_q
anchor_dq
anchor_joints
anchor_state
anchor_quaternion
anchor_position
anchor_twist
all_u
order
inverse
link_quaternion
tool_linear
tool_angular
trajectory_time
process_terms
process_value
name
value
valid
scene_evidence_out
scene_value
values
signed
augmented
folded_augmented
augmented_cost
original_process
feasible
residual
hard_invalid
eligible
raw_scene
self._optimizer_evidence
regular
folded
weighted
channels
category
evidence
regular_evidence
folded_evidence
self._scene_evidence
```

直接调用符号：

```python
super().compute_costs
super
getattr
self._process_mode.startswith
torch.full
_joint_state_sample_times
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
anchor_state.tool_poses.quaternion.detach
anchor_state.tool_jacobians[..., 3:, :].detach
torch.cat
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
_scene_boundary_evaluate
process_value.square
problem.functional_inequalities
functional_constraints.terms(signed).values
functional_constraints.terms
q.new_zeros
collection.add
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

[定义位置](../../../manipulation/planning.py#L2668)。

输入签名：

```python
def ordered(value):
```

显式输出分支：

```python
value.gather(1, index)
```

读取的对象属性：

```python
order.reshape(*order.shape, *(1,) * (value.ndim - 2)).expand_as
order.reshape
order.shape
value.ndim
value.gather
```

写入目标：

```python
index
```

直接调用符号：

```python
order.reshape(*order.shape, *(1,) * (value.ndim - 2)).expand_as
order.reshape
value.gather
```

## `_process_manager_type.ProcessCostManagerCfg`

[定义位置](../../../manipulation/planning.py#L2795)。

```python
class ProcessCostManagerCfg(RobotCostManagerCfg):
```

类字段：

```python
process_mode: str = ''
```

## `_process_manager_type.ProcessCostManagerCfg.__post_init__`

[定义位置](../../../manipulation/planning.py#L2798)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
self.class_type
```

直接调用符号：

无显式条目。

## `_process_manager_type.ProcessCostManagerCfg.create`

[定义位置](../../../manipulation/planning.py#L2802)。

输入签名：

```python
def create(data_dict, scene_collision_checker=None, device_cfg=None):
```

显式输出分支：

```python
ProcessCostManagerCfg(**values, process_mode=data_dict.get('process_mode', ''))
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
ProcessCostManagerCfg
data_dict.get
```

## `_process_manager_type.bind_request`

[定义位置](../../../manipulation/planning.py#L2807)。

输入签名：

```python
def bind_request(new_problem, new_scene, new_progress, new_anchors, new_budget, new_constraints):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.empty
manager.bind_request_data
```

写入目标：

```python
problem
scene
progress_map
anchors
budget
functional_constraints
collision_witnesses
manager
```

直接调用符号：

```python
np.empty
manager.bind_request_data
```

## `_process_manager_type.add_collision_witnesses`

[定义位置](../../../manipulation/planning.py#L2818)。

输入签名：

```python
def add_collision_witnesses(parameters):
```

显式输出分支：

```python
extra
```

读取的对象属性：

```python
np.asarray(parameters, dtype=np.float32).astype(float).reshape
np.asarray(parameters, dtype=np.float32).astype
np.asarray
np.float32
np.isfinite(parameters).all
np.isfinite
np.any
query_anchor_parameters()[:, 0].astype(np.float32).astype
query_anchor_parameters()[:, 0].astype
budget.max_task_anchors
np.sort
np.r_
manager.bind_request_data
```

写入目标：

```python
parameters
known
extra
collision_witnesses
manager
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L2846)。

输入签名：

```python
def _trajectory_candidate(result, index):
```

实现说明：

> Select one original seed through V2's own complete result gather.
> 
> V2 has no arbitrary-index result accessor. Its top-k gather selects from
> ``seed_rank`` for actions, flags and valid lengths. JointState's native
> indexer separately handles the full joint states and their active-joint
> controls; these need not have the same DOF when some joints are locked.
> Never flatten seeds into one apparent temporal trajectory.

显式输出分支：

```python
result
candidate
```

读取的对象属性：

```python
result.batch_size
result.num_seeds
result.seed_rank.new_tensor
result.seed_rank
ordered.get_topk_seeds
state[:, index:index + 1].clone
state.dt
state.dt[:, index:index + 1].clone
metrics.costs_and_constraints.constraints
metrics.costs_and_constraints
constraints.names
value[index:index + 1].detach().clone
value[index:index + 1].detach
constraints.values
metrics.debug
metrics.debug[index:index + 1].detach().clone
metrics.debug[index:index + 1].detach
```

写入目标：

```python
ordered
ordered.seed_rank
other
ordered.js_solution
ordered.interpolated_trajectory
ordered.metrics
ordered.interpolated_metrics
candidate
name
state
single
single.dt
metrics
constraints
selected_metrics
value
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L2904)。

输入签名：

```python
def _scene_metric_channel(metrics, channel):
```

实现说明：

> Read collision (0), support (1), forbidden (2), or conditional unknown (3).

显式输出分支：

```python
metrics.debug[..., channel:channel + 1]
```

读取的对象属性：

```python
metrics.costs_and_constraints.constraints
metrics.costs_and_constraints
constraints.values
constraints.names.index
constraints.names
metrics.debug
metrics.debug.shape
scene.shape
```

写入目标：

```python
constraints
scene
```

直接调用符号：

```python
constraints.names.index
RuntimeError
```

## `_scene_update_metric_residuals`

[定义位置](../../../manipulation/planning.py#L2913)。

输入签名：

```python
def _scene_update_metric_residuals(metrics, seed_index=None):
```

实现说明：

> Report support gaps for the evaluated candidate, never request infeasibility.

显式输出分支：

```python
{}
{'scene_update': float(flat[index].item())}
```

读取的对象属性：

```python
metrics.costs_and_constraints.constraints.names
metrics.costs_and_constraints.constraints
metrics.costs_and_constraints
evidence.reshape
flat.masked_fill(~flat.isfinite(), float('inf')).argmax
flat.masked_fill
flat.isfinite
flat[index].item
```

写入目标：

```python
evidence
flat
index
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L2929)。

输入签名：

```python
def _coverage_metric_residuals(metrics, seed_index=None):
```

实现说明：

> Report forbidden-space violations, not collision or missing map data.
> 
> A different path can stay within the same observed/explicitly permitted
> space. Reporting this category grants no permission to enter unknown space.

显式输出分支：

```python
{}
{'unknown_forbidden': float(flat[index].item())}
```

读取的对象属性：

```python
metrics.costs_and_constraints.constraints.names
metrics.costs_and_constraints.constraints
metrics.costs_and_constraints
evidence.reshape
flat.masked_fill(~flat.isfinite(), float('inf')).argmax
flat.masked_fill
flat.isfinite
flat[index].item
```

写入目标：

```python
evidence
flat
index
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L2947)。

输入签名：

```python
def _conditional_unknown_metric_residuals(metrics, seed_index=None):
```

实现说明：

> Retain the same query's unknown evidence even for native-success curves.

显式输出分支：

```python
{}
{'conditional_unknown': float(flat[index].item())}
```

读取的对象属性：

```python
metrics.costs_and_constraints.constraints.names
metrics.costs_and_constraints.constraints
metrics.costs_and_constraints
evidence.reshape
flat.masked_fill(~flat.isfinite(), float('inf')).argmax
flat.masked_fill
flat.isfinite
flat[index].item
```

写入目标：

```python
evidence
flat
index
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L2961)。

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

## `_collision_metric_residuals`

[定义位置](../../../manipulation/planning.py#L3000)。

输入签名：

```python
def _collision_metric_residuals(metrics, seed_index=None):
```

实现说明：

> Summarize collision evidence from the same observed-scene query.
> 
> The other diagnostic channels retain missing support and coverage as
> distinct results. The aggregate process_scene value is not reclassified.

显式输出分支：

```python
residuals
```

读取的对象属性：

```python
metrics.costs_and_constraints.constraints
metrics.costs_and_constraints
constraints.names
```

写入目标：

```python
constraints
residuals
residuals['scene_collision']
```

直接调用符号：

```python
maximum
_scene_metric_channel
```

## `_collision_metric_residuals.maximum`

[定义位置](../../../manipulation/planning.py#L3007)。

输入签名：

```python
def maximum(value):
```

显式输出分支：

```python
float((value if seed_index is None else value[seed_index]).max().item())
```

读取的对象属性：

```python
(value if seed_index is None else value[seed_index]).max().item
(value if seed_index is None else value[seed_index]).max
```

写入目标：

无显式条目。

直接调用符号：

```python
float
(value if seed_index is None else value[seed_index]).max().item
(value if seed_index is None else value[seed_index]).max
```

## `_dense_collision_witnesses`

[定义位置](../../../manipulation/planning.py#L3017)。

输入签名：

```python
def _dense_collision_witnesses(result, *, regular_horizon, anchor_parameters, capacity):
```

实现说明：

> Pick peaks only in intervals without an existing collision witness.
> 
> Native dense metrics store real samples in a valid prefix. The final
> allocated column also contains folded anchor penalties, so it is never
> interpreted as a new physical-time observation. Work already performed
> for the selected candidate is the only source of additional queries.

显式输出分支：

```python
report
report
```

读取的对象属性：

```python
result.batch_size
result.num_seeds
result.interpolated_metrics
result.interpolated_last_tstep.reshape(-1)[0].item
result.interpolated_last_tstep.reshape
result.interpolated_last_tstep
result.interpolated_trajectory.position
result.interpolated_trajectory
like.shape
(torch.arange(count, device=like.device, dtype=like.dtype) / (count - 1)).cpu().numpy
(torch.arange(count, device=like.device, dtype=like.dtype) / (count - 1)).cpu
torch.arange
like.device
like.dtype
(torch.arange(regular_horizon, device=like.device, dtype=like.dtype) / (regular_horizon - 1)).cpu().numpy
(torch.arange(regular_horizon, device=like.device, dtype=like.dtype) / (regular_horizon - 1)).cpu
regular_u.astype
np.asarray(anchor_parameters, dtype=dense_u.dtype).astype
np.asarray
dense_u.dtype
np.array
result.interpolated_metrics.costs_and_constraints.constraints
result.interpolated_metrics.costs_and_constraints
constraints.names
constraints.values
measurements.insert
ordinary.costs_and_constraints.constraints
ordinary.costs_and_constraints
ordinary_constraints.names
ordinary_constraints.values
ordinary_measurements.get
tensor.shape
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
np.r_
edges.reshape
np.any
np.argmax
peaks.append
component_peaks.append
candidates.append
report.update
```

写入目标：

```python
report
count
like
horizon
dense_u
regular_u
known
eligible
index
u
constraints
measurements
name
tensor
ordinary
ordinary_constraints
ordinary_measurements
ordinary_measurements['scene_collision']
positive_parameters
_
values
positive
positive[-1]
positive_parameters[name]
component_peaks
edges
peaks
left
right
detected
candidates
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L3118)。

```python
class _CuroboPlanner():
```

类字段：

无显式条目。

实现说明：

> Resident native solvers with request-local online constraint inputs.

## `_CuroboPlanner.__init__`

[定义位置](../../../manipulation/planning.py#L3122)。

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
self.trajopt
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

## `_CuroboPlanner._initialize`

[定义位置](../../../manipulation/planning.py#L3145)。

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
self.robot_metadata
self.anchor_capacity
workspace.cpu_live_bytes
self.validation_storage
budget.max_host_rss_bytes
torch.backends.cuda.matmul
torch.backends.cuda
torch.backends
torch.backends.cudnn
self._caller_precision
self.device_cfg.device
self.device_cfg
budget.max_device_used_bytes
workspace.total_bytes
workspace.cpu_bytes
workspace.gpu_bytes
workspace.anchor_fk_bytes
self.device_cfg.to_device
problem.functional_constraint_keys
self.anchors
self.formal_clock
self.seed_parameter
self.robot_dict
self.functional_constraints
budget.ik_iterations
budget.trajectory_iterations
InverseKinematicsCfg.create
budget.ik_seeds
problem.numerical.position_m
problem.numerical
problem.numerical.rotation_rad
self._manager_type
budget.random_seed
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
transition['transition_model_cfg'].update
budget.trajectory_seeds
budget.interpolation_buffer_size
TrajectoryOptimizerCfg.create
self._trajectory_configuration
self._bind_formal_clock
self._initialize_verification_constraints
self._bind_physical_domains
self._manager_type.bind_request
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
self.validation_storage
workspace
torch.backends.cuda.matmul.allow_tf32
torch.backends.cudnn.allow_tf32
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
self._trajectory_optimizer_graph_admitted
self.ik
self.trajopt
self.anchors
self.robot_dict
self.seed_parameter
self.formal_clock
self.functional_constraints
self._manager_type
self.phase
self.goal_ik_candidates
self.last_optimizer_diagnostics
self.delivery_candidate
ik_optimizer
trajectory_optimizer
config
iterations
inner_iterations
ik_cfg
ik_cfg.use_lm_seed
ik_cfg.exit_early
self.joint_names
columns
self.start
self._verification_fk
self._orientation_vjp
self._native_layout
self._verification_scene
transition
self._trajectory_configuration
self._optimizer_class
self._optimizer_config_class
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
self._bind_physical_domains
self._manager_type.bind_request
```

## `_CuroboPlanner._initialize.configuration`

[定义位置](../../../manipulation/planning.py#L3281)。

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

## `_CuroboPlanner._initialize.ProcessTrajectoryOptimizer`

[定义位置](../../../manipulation/planning.py#L3367)。

```python
class ProcessTrajectoryOptimizer(TrajectoryOptimizer):
```

类字段：

无显式条目。

## `_CuroboPlanner._initialize.ProcessTrajectoryOptimizer._prepare_goal_buffer`

[定义位置](../../../manipulation/planning.py#L3369)。

输入签名：

```python
def _prepare_goal_buffer(self, solve_state, goal_tool_poses, current_state=None, use_implicit_goal=False, seed_goal_state=None, goal_state=None):
```

实现说明：

> Preserve native buffers while completing goal-free metadata.
> 
> The installed registry infers its batch only from pose or joint
> tracking goals. Our process constraints own those requirements,
> so seed-only registries retain -1 unless the request supplies
> this metadata. Bind it here for direct solves, candidate metrics
> and resident reuse before any rollout allocates state storage.

显式输出分支：

```python
(buffer, update_reference)
```

读取的对象属性：

```python
super()._prepare_goal_buffer
solve_state.batch_size
```

写入目标：

```python
buffer
update_reference
buffer.batch_size
```

直接调用符号：

```python
super()._prepare_goal_buffer
super
```

## `_CuroboPlanner._initialize.ProcessTrajectoryOptimizer._get_result`

[定义位置](../../../manipulation/planning.py#L3387)。

输入签名：

```python
def _get_result(self, opt_result, metrics_result, interpolated_metrics, interpolated_trajectory, last_tstep, num_seeds, return_seeds):
```

显式输出分支：

```python
super()._get_result(opt_result, metrics_result, interpolated_metrics, interpolated_trajectory, last_tstep, num_seeds, return_seeds)
```

读取的对象属性：

```python
self.metrics_rollout.metrics_constraint_manager.bind_scene_evidence
self.metrics_rollout.metrics_constraint_manager
self.metrics_rollout
self.additional_metrics_rollouts['interpolated_rollout'].metrics_constraint_manager.bind_scene_evidence
self.additional_metrics_rollouts['interpolated_rollout'].metrics_constraint_manager
self.additional_metrics_rollouts
super()._get_result
```

写入目标：

无显式条目。

直接调用符号：

```python
self.metrics_rollout.metrics_constraint_manager.bind_scene_evidence
self.additional_metrics_rollouts['interpolated_rollout'].metrics_constraint_manager.bind_scene_evidence
super()._get_result
super
```

## `_CuroboPlanner._initialize.ProcessTrajectoryOptimizer.get_interpolated_trajectory`

[定义位置](../../../manipulation/planning.py#L3396)。

输入签名：

```python
def get_interpolated_trajectory(self, js_optimized):
```

显式输出分支：

```python
(state, counts, False)
```

读取的对象属性：

```python
self.config.interpolation_type
self.config
TrajInterpolationType.BSPLINE_KNOTS_CUDA
js_optimized.position.shape
js_optimized.position
js_optimized.position.ndim
self.config.num_seeds
self.horizon
self.action_dim
js_optimized.knot
js_optimized.knot.shape
self.action_horizon
value.numel
value.dtype
torch.float32
(value == expected).all
self.config.interpolation_buffer_size
self._interpolated_traj_buffer
self._interpolated_traj_buffer.shape
JointState.zeros
self.device_cfg
self.joint_names
self._interpolation_dt_buffer
self._interpolation_dt_buffer.shape
torch.full
self.device_cfg.device
(self._interpolation_dt_buffer == grid['interpolation_dt']).all
torch.int32
self.core._goal_buffer
self.core
self.transition_model.control_space
self.transition_model
self.transition_model._rollout_step_fn
step.action_horizon
step.knot_dt
step.sample
goal.current_js
goal.idxs_current_js
self.additional_metrics_rollouts
rollout.constraint_manager.set_dense_counts
rollout.constraint_manager
rollout.metrics_constraint_manager.set_dense_counts
rollout.metrics_constraint_manager
```

写入目标：

```python
grid
batch
name
expected
value
capacity
shape
self._interpolated_traj_buffer
self._interpolation_dt_buffer
counts
goal
js_optimized.control_space
step
state
rollout
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L3459)。

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
self.robot_dict
self.robot_validator
self.device_cfg.device
self.device_cfg
self.check_request
```

写入目标：

无显式条目。

直接调用符号：

```python
self.scene.bind_online_robot
self.check_request
```

## `_CuroboPlanner.rebind`

[定义位置](../../../manipulation/planning.py#L3464)。

输入签名：

```python
def rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None):
```

实现说明：

> Reuse IK/TrajOpt while clearing all prior candidate and request state.

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
workspace.cpu_live_bytes
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
problem.functional_constraint_keys
self._manager_type.bind_request
self._manager_type
self.functional_constraints
scene.clone_for_rollout
self.trajopt.config
self.trajopt
rollout.transition_model
self.trajopt.core.get_all_rollout_instances
self.trajopt.core
self.trajopt.additional_metrics_rollouts.values
self.trajopt.additional_metrics_rollouts
rollout.metrics_transition_model
transitions.values
transition.update_traj_dt
self.trajopt._interpolation_dt_buffer
self.trajopt._interpolation_dt_buffer.fill_
self._bind_formal_clock
self.seed_parameter.zero_
self.seed_parameter
self.ik
solver.reset_seed
solver.core.goal_registry_manager
solver.core
manager._goal_buffer
solver.core.get_all_rollout_instances
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
self.functional_constraints
self._verification_scene
self.trajopt.config.minimum_trajectory_dt
self.trajopt.config.maximum_trajectory_dt
transitions
rollout
transitions[id(rollout.metrics_transition_model)]
transition
self.trajopt.config.interpolation_dt
self.last_validation
self.last_optimizer_diagnostics
self.goal_ik_candidates
self.delivery_candidate
solver
manager
manager._goal_buffer.link_goal_poses
rollouts
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

[定义位置](../../../manipulation/planning.py#L3550)。

输入签名：

```python
def _bind_formal_clock(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.trajopt
native.get
self.formal_clock.items
self.formal_clock
self.formal_clock.update
```

写入目标：

```python
native
name
value
```

直接调用符号：

```python
_formal_native_clock
any
native.get
self.formal_clock.items
ValueError
self.formal_clock.update
```

## `_CuroboPlanner._seed_u`

[定义位置](../../../manipulation/planning.py#L3558)。

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

## `_CuroboPlanner._initialize_verification_constraints`

[定义位置](../../../manipulation/planning.py#L3568)。

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
self.trajopt.config.metrics_rollout_config.constraint_cfg
self.trajopt.config.metrics_rollout_config
self.trajopt.config
self.trajopt
self.trajopt.transition_model
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

## `_CuroboPlanner._bind_physical_domains`

[定义位置](../../../manipulation/planning.py#L3581)。

输入签名：

```python
def _bind_physical_domains(self):
```

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
limits.position
self.start.position
self.start
self.start.velocity
self.formal_clock
self.trajopt.action_horizon
self.trajopt
self.ik
self.position_domain
self.physical_domain
self._trajectory_optimizer_graph_admitted
```

写入目标：

```python
limits
columns
name
bounds
self.position_domain
self.physical_domain
cuda
```

直接调用符号：

```python
self._verification_native.get_cost
limits.joint_names.index
JointPositionDomain
CubicJointDomain
bind_solver_components
```

## `_CuroboPlanner.initial_state_diagnostics`

[定义位置](../../../manipulation/planning.py#L3602)。

输入签名：

```python
def initial_state_diagnostics(self):
```

实现说明：

> Test measured q/dq and online collision geometry before proposing motion.

显式输出分支：

```python
{'passed': not failed, 'residuals': residuals, 'failed_components': failed, 'scene_components': scene_components, 'observed_geometry': deepcopy(self.scene.last_query_report), 'scope': 'measured joint limits and selected online collision model'}
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
self.start.position
self.start
self.start.velocity
self.torch.no_grad
self.torch
self.torch.relu
self.JointState
q.contiguous
dq.contiguous
self._verification_fk.compute_kinematics
self._verification_fk
self._RobotState
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
self.problem.start_progress
self.scene
self.progress_map
self.problem.transform_robot_spheres
native.robot_spheres
q.new_zeros
value.max().item
value.max
components.items
residuals.items
np.isfinite
self.scene.last_query_report
```

写入目标：

```python
limits
columns
name
q
dq
components
value
bounds
components[name]
js
state
native
index
position
quaternion
progress
scene_components
components['scene']
residuals
failed
```

直接调用符号：

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
state.tool_poses.tool_frames.index
self.problem.transform_tool_pose
q.new_full
_scene_boundary_evaluate
self.problem.transform_robot_spheres
q.new_zeros
float
value.max().item
value.max
components.items
residuals.items
np.isfinite
deepcopy
```

## `_CuroboPlanner.close`

[定义位置](../../../manipulation/planning.py#L3642)。

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

## `_CuroboPlanner._goal`

[定义位置](../../../manipulation/planning.py#L3665)。

输入签名：

```python
def _goal(self, progress):
```

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
self.problem.sample_reference
self.GoalToolPose
self.problem.tool_link
position[:, :, None, None].contiguous
quaternion[:, :, None, None].contiguous
```

写入目标：

```python
s
position
quaternion
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L3673)。

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

## `_CuroboPlanner.reference_seeds`

[定义位置](../../../manipulation/planning.py#L3679)。

输入签名：

```python
def reference_seeds(self, joint_reference=None):
```

实现说明：

> Build bounded initializers from explicit reference-query evidence.
> 
> Finite native IK successes enter the next warm start and anchor graph.
> Native success includes its pose convergence and native constraints;
> it is not whole-curve task or observation evidence. Static task/scene
> reports rank proposals without deleting transferred reference data.
> Failed native queries are filled from successful anchors and measured
> q0. Every fitted curve still needs the complete candidate constraints.

显式输出分支：

```python
(seeds, [])
(seeds.contiguous(), success_counts)
```

读取的对象属性：

```python
joint_reference.joint_names
self.joint_names
joint_reference.joint_names.index
joint_reference.position
joint_reference.velocity
np.isfinite(q).all
np.isfinite
np.isfinite(dq).all
joint_reference.duration
joint_reference.times
joint_reference.progress_u
joint_reference.progress_s
self.source_progress_map
self.reference_time_map
self.progress_map
np.asarray
self.reference_time_map.anchor_parameters
self.physical_domain.fit_reference
self.physical_domain
self.device_cfg.to_device
self.device_cfg
self._seed_velocity_bounds
self.formal_clock
self.budget.trajectory_seeds
self.budget
controls[None, None].expand(1, count, -1, -1).contiguous
controls[None, None].expand
self.physical_domain.validate
self.torch
self.start.position[:, None].clone
self.start.position
self.start
self.physical_domain._curve_map.reference_parameters
self.physical_domain._curve_map
self.check_request
np.interp
self.progress_map.u
self.progress_map.s
self.seed_parameter.fill_
self.seed_parameter
self.ik.solve_pose
self.ik
self._goal
self.budget.ik_seeds
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
success_counts.append
(flags & np.isfinite(values).all(-1)).sum
np.isfinite(values).all
self._rank_ik_candidates
previous[0].detach().cpu().numpy
previous[0].detach().cpu
previous[0].detach
self.reference_ik_distinct_counts.append
self.reference_ik_distinct_counts
np.isfinite(values[index]).all
measurement[0].detach().cpu().numpy().reshape
measurement[0].detach().cpu().numpy
measurement[0].detach().cpu
measurement[0].detach
errors.shape
pools.append
evidence.append
self.reference_ik_queries.append
self.reference_ik_queries
self.device_cfg.to_device(values[qualified][None]).contiguous
self.start.position.detach().cpu().numpy
self.start.position.detach().cpu
self.start.position.detach
self.problem.numerical.joint
self.problem.numerical
self.problem
self.initial_seed_strategies
targets.append
np.broadcast_to(self.start.position[0].detach().cpu().numpy(), (self.trajopt.action_horizon, len(self.joint_names))).copy
np.broadcast_to
self.start.position[0].detach().cpu().numpy
self.start.position[0].detach().cpu
self.start.position[0].detach
self.trajopt.action_horizon
self.trajopt
self.initial_seed_strategies.append
self.trajopt._trajectory_seed_generator.generate_constant_seeds
self.trajopt._trajectory_seed_generator
self.physical_domain.initialize(value).reshape
self.physical_domain.initialize
self.physical_domain.evaluate
controls.append
self.reference_seed_fits.append
self.reference_seed_fits
torch.stack
self._build_goal_ik_candidates
pool.items
np.isfinite(row['joint_position']).all
samples.append
postures.append
self.goal_ik_fit_hints.append
self.goal_ik_fit_hints
seeds.contiguous
```

写入目标：

```python
self.joint_reference_report
columns
name
q
dq
source_map
original_u
parameters
value
controls
fit
count
seeds
self.reference_ik_queries
self.reference_ik_distinct_counts
self.reference_ik_chains
self.reference_seed_fits
self.goal_ik_fit_hints
self.initial_seed_strategies
self.seed_strategies
torch
velocity_bounds
previous
warm_start_query
control_u
pools
evidence
success_counts
query_index
u
progress
result
values
flags
ranked
reports
qualified
index
report
report['reference_status']
attribute
measurement
errors
error
report[name]
row
chains
retained
targets
chain
self.initial_seed_strategies[1]
branch
target
control
self.initial_seed_strategies[branch]
targets[branch]
pool
samples
postures
parameter
entry
candidates
rank
self.reference_seed_fits[branch]['functional_hint_fit']
hint
hint['applied']
seeds[0, branch]
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L3877)。

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
(controls[None, None].contiguous(), report)
(None, report)
```

读取的对象属性：

```python
self.torch
time.monotonic
self.budget.search_seconds
self.budget
self.trajopt.transition_model.action_bound_lows.detach().cpu().numpy().reshape
self.trajopt.transition_model.action_bound_lows.detach().cpu().numpy
self.trajopt.transition_model.action_bound_lows.detach().cpu
self.trajopt.transition_model.action_bound_lows.detach
self.trajopt.transition_model.action_bound_lows
self.trajopt.transition_model
self.trajopt
self.trajopt.transition_model.action_bound_highs.detach().cpu().numpy().reshape
self.trajopt.transition_model.action_bound_highs.detach().cpu().numpy
self.trajopt.transition_model.action_bound_highs.detach().cpu
self.trajopt.transition_model.action_bound_highs.detach
self.trajopt.transition_model.action_bound_highs
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
self.physical_domain.fit_reference
self.physical_domain
self._seed_velocity_bounds
report.update
controls[None, None].contiguous
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

[定义位置](../../../manipulation/planning.py#L3935)。

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

## `_CuroboPlanner.ompl_seed.check_search`

[定义位置](../../../manipulation/planning.py#L3939)。

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

## `_CuroboPlanner.ompl_seed.state_parts`

[定义位置](../../../manipulation/planning.py#L3945)。

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

## `_CuroboPlanner.ompl_seed.equations`

[定义位置](../../../manipulation/planning.py#L3964)。

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

## `_CuroboPlanner.ompl_seed.equations.residual_function`

[定义位置](../../../manipulation/planning.py#L3966)。

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

## `_CuroboPlanner.ompl_seed.equations.evaluate`

[定义位置](../../../manipulation/planning.py#L3974)。

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

## `_CuroboPlanner.ompl_seed.valid`

[定义位置](../../../manipulation/planning.py#L3998)。

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

## `_CuroboPlanner.ompl_seed.proposals`

[定义位置](../../../manipulation/planning.py#L4030)。

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
min
len
deepcopy
candidate_trajectory._curve
rng.uniform
self.physical_domain.evaluate
np.asarray
```

## `_CuroboPlanner.ompl_seed.search_frame`

[定义位置](../../../manipulation/planning.py#L4067)。

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

## `_CuroboPlanner._functional_goal`

[定义位置](../../../manipulation/planning.py#L4199)。

输入签名：

```python
def _functional_goal(self, target):
```

实现说明：

> Convert one explicitly declared functional pose into native link coordinates.
> 
> The centre is an IK proposal only. All other goals and the original
> allowed tolerances are checked separately; no reference sample becomes
> a hard goal and no full pose is invented for a coordinate/support goal.

显式输出分支：

```python
self.GoalToolPose(tool_frames=[self.problem.tool_link], position=position[:, :, None, None].contiguous(), quaternion=quaternion[:, :, None, None].contiguous())
```

读取的对象属性：

```python
target.target
np.linalg.inv
np.linalg
target.tool_from_function
self.device_cfg.to_device
self.device_cfg
Rotation.from_matrix(pose[:3, :3]).as_quat
Rotation.from_matrix
self.problem.reference_to_link_pose
self.problem
self.GoalToolPose
self.problem.tool_link
position[:, :, None, None].contiguous
quaternion[:, :, None, None].contiguous
```

写入目标：

```python
pose
position
quaternion
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L4216)。

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

## `_CuroboPlanner._build_goal_ik_candidates`

[定义位置](../../../manipulation/planning.py#L4282)。

输入签名：

```python
def _build_goal_ik_candidates(self, reference_seeds):
```

实现说明：

> Retain bounded CPU proposals at actual full-pose functional events.
> 
> One existing-size IK batch is queried per distinct event time. When
> several full-pose events share that time, the first supplies a nominal
> IK centre and all same-time goals remain part of static screening. This
> is a finite proposal set, not a complete search of their intersection.
> Coordinate/support goals never acquire an invented full TCP target.
> Native calls remain on the same tracked solver and frozen clock.

显式输出分支：

```python
self.goal_ik_candidates
```

读取的对象属性：

```python
reference_seeds[0].detach().cpu().numpy
reference_seeds[0].detach().cpu
reference_seeds[0].detach
self.problem.pose_events
self.problem
item[1].progress
target.progress
self.problem.start_progress
self.progress_map.anchor_parameters
self.progress_map
self.goal_ik_candidates
self.physical_domain.evaluate
self.physical_domain
np.asarray
self.seed_parameter.fill_
self.seed_parameter
self.ik.solve_pose
self.ik
self._functional_goal
self.start
self.device_cfg.to_device(previous[None]).contiguous
self.device_cfg.to_device
self.device_cfg
self.budget.ik_seeds
self.budget
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
self.problem.goal_requirements_at
getattr(self.problem, name)[index].progress
self._rank_ik_candidates
self.formal_clock
candidates[index].tolist
```

写入目标：

```python
self.goal_ik_candidates
controls
targets
_
target
u
previous
result
candidates
success
goal_ids
name
index
selected
reports
self.goal_ik_candidates[u]
value
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L4328)。

输入签名：

```python
def _observed_native_result(self, actions):
```

实现说明：

> Evaluate a complete reference seed through native success metrics once.

显式输出分支：

```python
self.trajopt._get_best_result(result, count)
```

读取的对象属性：

```python
actions.shape
self.torch.no_grad
self.torch
self.trajopt.metrics_rollout.compute_metrics_from_action
self.trajopt.metrics_rollout
self.trajopt
metrics.state.joint_state
metrics.state
self.trajopt.joint_names
metrics.actions
joint.dt
self.trajopt.interpolation_steps
self.trajopt._interpolate_and_compute_metrics
self.trajopt._get_result
self.trajopt._get_best_result
```

写入目标：

```python
count
metrics
joint
joint.joint_names
joint.knot
joint.knot_dt
interpolated
trajectory
last
result
```

直接调用符号：

```python
self.torch.no_grad
self.trajopt.metrics_rollout.compute_metrics_from_action
self.trajopt._interpolate_and_compute_metrics
self.trajopt._get_result
self.trajopt._get_best_result
```

## `_CuroboPlanner._optimizer_reports`

[定义位置](../../../manipulation/planning.py#L4343)。

输入签名：

```python
def _optimizer_reports(self, solution, initial):
```

显式输出分支：

```python
reports
```

读取的对象属性：

```python
optimizer._core.solve_diagnostics
optimizer._core
self.trajopt.core.optimizers
self.trajopt.core
self.trajopt
solution.shape
(solution[0, index] - initial[0, index]).abs
reports.append
(displacement == 0).all().item
(displacement == 0).all
displacement.max().item
displacement.max
```

写入目标：

```python
core_reports
optimizer
reports
index
displacement
rows
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L4356)。

输入签名：

```python
def promote_collision_witnesses(self, result):
```

实现说明：

> Rebind every manager's exact queries before the one recovery solve.

显式输出分支：

```python
report
```

读取的对象属性：

```python
self._trajectory_configuration
self._manager_type.query_anchor_parameters
self._manager_type
result.js_solution.position.shape
result.js_solution.position
result.js_solution
self.budget.max_task_anchors
self.budget
self._manager_type.add_collision_witnesses
report.update
```

写入目标：

```python
pairs
report
added
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L4371)。

输入签名：

```python
def _native_request(self, seed):
```

实现说明：

> Bind real terminal poses or the task-only native constraint route.

显式输出分支：

```python
(state, goal)
```

读取的对象属性：

```python
self.problem.pose_events
self.problem
goal.progress
self.problem.end_progress
self.problem.pose_relations
self.problem.mechanisms
self.problem.interiors
self.problem.coordinate_events
self.problem.supports
self.scene.contacts
self.scene
self.route
np.array_equal
target.tool_from_function
np.eye
self.trajopt.enable_tool_pose_tracking
self.trajopt
self.trajopt.disable_tool_pose_tracking
self.trajopt.disable_joint_position_tracking
self._functional_goal
SolveMode.BATCH
seed.shape
goal.tool_frames
```

写入目标：

```python
target
goal
self.route
direct
state
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L4395)。

输入签名：

```python
def evaluate_reference(self, seed):
```

实现说明：

> Accept a feasible complete reference candidate before changing its controls.

显式输出分支：

```python
self._observed_native_result(seed.reshape(-1, self.trajopt.action_horizon, self.trajopt.action_dim))
```

读取的对象属性：

```python
self.physical_domain.validate
self.physical_domain
self._native_request
self.JointState.from_position
self.JointState
seed[:, :, -1].contiguous
self.torch.full
self.torch
seed.shape
self.budget.maximum_trajectory_dt
self.budget
seed.dtype
seed.device
self.trajopt._prepare_goal_buffer
self.trajopt
self.start
self.trajopt._update_rollout_params
self._observed_native_result
seed.reshape
self.trajopt.action_horizon
self.trajopt.action_dim
```

写入目标：

```python
state
goal
endpoint
endpoint.dt
buffer
_
self.last_optimizer_diagnostics
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L4409)。

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

## `_CuroboPlanner._candidate_metrics`

[定义位置](../../../manipulation/planning.py#L4437)。

输入签名：

```python
def _candidate_metrics(self, trajectory, candidate_scene):
```

实现说明：

> Run existing native metrics on exact, possibly nonuniform queries.
> 
> The installed rollout does not forward query kwargs to its managers.
> A short-lived manager context therefore carries actual times and exact
> source-curve anchors; finally clears it even when native evaluation
> raises. All states query the same observed fields at the acceptance
> spacing; task and observation evidence remain tied to that query.

显式输出分支：

```python
(report, times / trajectory.duration, len(times) + len(pairs))
```

读取的对象属性：

```python
self.scene
trajectory.progress_u
trajectory.progress_s
mapping.anchor_parameters
self.anchors
self.budget
self.budget.max_task_anchors
self.budget.max_validation_points
self.budget.interpolation_buffer_size
trajectory.joint_names.index
trajectory.joint_names
self.joint_names
trajectory.sample
trajectory.duration
self.device_cfg.to_device
self.device_cfg
tensor(anchor_q[:, columns])[None].contiguous
tensor(anchor_dq[:, columns])[None].contiguous
self.trajopt.disable_tool_pose_tracking
self.trajopt
self.trajopt.disable_joint_position_tracking
SolveMode.BATCH
self.JointState.from_position
self.JointState
tensor(q[-1:, columns])[None].contiguous
self.budget.maximum_trajectory_dt
self.trajopt._prepare_goal_buffer
self.start
self.trajopt._update_rollout_params
self.trajopt.additional_metrics_rollouts
rollout._cost_manager_list
rollout.metrics_constraint_manager
tensor(q[:, columns])[None].contiguous
tensor(dq[:, columns])[None].contiguous
tensor(ddq[:, columns])[None].contiguous
tensor(trajectory._curve(times, 3)[:, columns])[None].contiguous
trajectory._curve
self.budget.interpolation_dt
self.check_request
self.torch.no_grad
self.torch
rollout.transition_model.compute_augmented_state
rollout.transition_model
rollout.compute_metrics_from_state
metrics.costs_and_constraints.constraints
metrics.costs_and_constraints
rollout.metrics_constraint_manager.bind_scene_evidence
constraints.names
constraints.values
metrics.feasible
metrics.feasible.all().item
metrics.feasible.all
np.isfinite
residuals.values
collision_residuals.items
scene_update_residuals.items
coverage_residuals.items
conditional_unknown_residuals.items
candidate_scene.last_query_report
```

写入目标：

```python
candidate_scene
mapping
pairs
times
columns
name
q
dq
ddq
progress
anchor_q
anchor_dq
_
tensor
context
state
endpoint
endpoint.dt
buffer
rollout
managers
manager
js
manager._candidate_query
native
metrics
constraints
collision_residuals
scene_update_residuals
coverage_residuals
conditional_unknown_residuals
residuals
value
feasible
report
```

直接调用符号：

```python
_ProgressMap
mapping.anchor_parameters
_candidate_sample_times
len
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

## `_CuroboPlanner.evaluate_candidate`

[定义位置](../../../manipulation/planning.py#L4552)。

输入签名：

```python
def evaluate_candidate(self, trajectory, *, native_report=None, candidate_scene=None):
```

实现说明：

> One acceptance boundary; completed native metrics are reused verbatim.

显式输出分支：

```python
(trajectory, attempt, points)
```

读取的对象属性：

```python
self._candidate_metrics
np.array
self._candidate_domain_validation
trajectory.residuals
attempt.update
np.isfinite
residuals.values
validation.update
np.full
parameters.shape
```

写入目标：

```python
attempt
parameters
points
validation
residuals
value
self.delivery_candidate
self.last_validation
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L4569)。

输入签名：

```python
def optimize(self, seed):
```

实现说明：

> One native solve with the current physical domain and unchanged clock.

显式输出分支：

```python
result
```

读取的对象属性：

```python
self.physical_domain.validate
self.physical_domain
self._native_request
seed.detach().clone
seed.detach
self.trajopt._solve_impl
self.trajopt
self.start
seed.shape
self.torch.full
self.torch
self.budget.maximum_trajectory_dt
self.budget
seed.dtype
seed.device
result.solution.shape
result.solution
initial.shape
self._optimizer_reports
```

写入目标：

```python
state
goal
initial
result
self.last_optimizer_diagnostics
```

直接调用符号：

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

[定义位置](../../../manipulation/planning.py#L4586)。

输入签名：

```python
def native_diagnostics(self, result, seed_index=None):
```

实现说明：

> Report the native success flag and original online-model residuals.

显式输出分支：

```python
report
```

读取的对象属性：

```python
result.success
success.all().item
success.all
result.metrics
result.interpolated_metrics
metrics.costs_and_constraints.constraints
metrics.costs_and_constraints
constraints.names
constraints.values
metrics.convergence.names
metrics.convergence
metrics.convergence.values
```

写入目标：

```python
success
report
label
metrics
constraints
report[label]
name
value
```

直接调用符号：

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

## `_CuroboPlanner.select_candidate`

[定义位置](../../../manipulation/planning.py#L4627)。

输入签名：

```python
def select_candidate(self, result):
```

实现说明：

> Select the first native-feasible seed; export is a single later delivery step.

显式输出分支：

```python
(candidate, trajectory, {**report, 'selected_seed_index': selected, 'candidates': reports})
```

读取的对象属性：

```python
result.num_seeds
self.native_diagnostics
metrics.get(label, {}).get('constraints', {}).items
metrics.get(label, {}).get
metrics.get
np.isfinite
residuals.values
metrics.get(label, {}).get('collision_residuals', {}).items
metrics.get(label, {}).get('scene_update_residuals', {}).items
metrics.get(label, {}).get('coverage_residuals', {}).items
metrics.get(label, {}).get('conditional_unknown_residuals', {}).items
self.seed_strategies
self.last_optimizer_diagnostics
reports.append
self._native_curve
```

写入目标：

```python
reports
selected
selected_key
index
metrics
residuals
label
name
value
accepted
v
severity
report
key
candidate
trajectory
```

直接调用符号：

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
```

## `_CuroboPlanner._native_curve`

[定义位置](../../../manipulation/planning.py#L4672)。

输入签名：

```python
def _native_curve(self, result):
```

实现说明：

> Copy the bound native cubic without independent geometry queries.

显式输出分支：

```python
trajectory
```

读取的对象属性：

```python
result.batch_size
result.num_seeds
self.trajopt.get_active_js
self.trajopt
result.get_interpolated_plan
native.position.detach().cpu().numpy().reshape
native.position.detach().cpu().numpy
native.position.detach().cpu
native.position.detach
native.position
self.joint_names
native.velocity.detach().cpu().numpy().reshape
native.velocity.detach().cpu().numpy
native.velocity.detach().cpu
native.velocity.detach
native.velocity
position.shape
native.dt.detach().cpu().reshape
native.dt.detach().cpu
native.dt.detach
native.dt
np.isfinite(position).all
np.isfinite
np.isfinite(velocity).all
np.arange
self.formal_clock
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
native
position
velocity
dt
times
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

[定义位置](../../../manipulation/planning.py#L4704)。

输入签名：

```python
def _candidate_domain_validation(self, trajectory):
```

实现说明：

> Check the actual saved polynomial, independent of its proposal basis.

显式输出分支：

```python
{'continuous_position': position, 'continuous_velocity': velocity, 'cubic_derivatives': derivatives, 'numerical_tolerances': asdict(self.problem.numerical)}
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
self.joint_names.index
self.joint_names
np.max
np.abs
trajectory.position
self.start.position[0].detach().cpu().numpy
self.start.position[0].detach().cpu
self.start.position[0].detach
self.start.position
self.start
self.problem.numerical.joint
self.problem.numerical
self.problem
trajectory.velocity
self.start.velocity[0].detach().cpu().numpy
self.start.velocity[0].detach().cpu
self.start.velocity[0].detach
self.start.velocity
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

[定义位置](../../../manipulation/planning.py#L4727)。

输入签名：

```python
def export(self, result, native_report):
```

实现说明：

> Reuse completed native metrics at the common candidate boundary.

显式输出分支：

```python
(trajectory, points)
```

读取的对象属性：

```python
self.evaluate_candidate
self._native_curve
```

写入目标：

```python
trajectory
attempt
points
```

直接调用符号：

```python
self.evaluate_candidate
self._native_curve
MotionPlanningError
```

## `_verified_dependency_outcomes`

[定义位置](../../../manipulation/planning.py#L4736)。

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

[定义位置](../../../manipulation/planning.py#L4758)。

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

[定义位置](../../../manipulation/planning.py#L4777)。

```python
class MotionPlanningError(ValueError):
```

类字段：

无显式条目。

实现说明：

> A failed bounded request, never a proof that the mathematical problem is infeasible.

## `MotionPlanningError.__init__`

[定义位置](../../../manipulation/planning.py#L4779)。

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

[定义位置](../../../manipulation/planning.py#L4784)。

```python
class MotionBackend():
```

类字段：

无显式条目。

实现说明：

> Own one robot's reusable native IK and trajectory optimizer instances.
> 
> The robot, device and tensor capacities are constructor inputs. Requests
> replace goals, measured q/dq, scene values and clocks; they cannot resize
> the robot or silently buy another solve. close() is the owner's lifecycle.

## `MotionBackend.__init__`

[定义位置](../../../manipulation/planning.py#L4791)。

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

[定义位置](../../../manipulation/planning.py#L4799)。

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

[定义位置](../../../manipulation/planning.py#L4804)。

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

[定义位置](../../../manipulation/planning.py#L4810)。

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
_CuroboPlanner
self._planner.rebind
```

## `MotionBackend.plan`

[定义位置](../../../manipulation/planning.py#L4826)。

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

[定义位置](../../../manipulation/planning.py#L4833)。

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

[定义位置](../../../manipulation/planning.py#L4846)。

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
> claimed to be preemptible: elapsed overrun rejects its candidate.

显式输出分支：

```python
outcome
outcome
outcome
outcome
outcome
outcome
outcome
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
planner.ik
planner.trajopt
solver.core.optimizers
solver.core
optimizer._core
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
planner.last_validation
candidate_scene.describe
outcome.execution_binding
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
planner.reference_ik_distinct_counts
planner.initial_seed_strategies
planner.reference_ik_queries
planner.reference_ik_chains
planner.reference_seed_fits
planner.goal_ik_fit_hints
planner.joint_reference_report
planner.evaluate_reference
planner.select_candidate
trajectory.joint_names
joint_reference.joint_names
np.array_equal
trajectory.times
joint_reference.times
trajectory.position
joint_reference.position
trajectory.velocity
joint_reference.velocity
planner.optimize
diagnostics['attempts'].append
original_budget.maximum_trajectory_solves
attempt['collision_residuals'].values
attempt['coverage_residuals'].values
planner.promote_collision_witnesses
native_result.solution.detach().clone().expand(1, budget.trajectory_seeds, -1, -1).contiguous
native_result.solution.detach().clone().expand
native_result.solution.detach().clone
native_result.solution.detach
native_result.solution
budget.search_seconds
planner.ompl_seed
report.get
seeds.shape
planner.route
planner.export
error.reason
type(error).__name__
traceback.format_exc
manager_type.query_diagnostics
backend._release
backend.close
ledger['binding_records'].append
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
initial_seconds
motion_ready
motion_baseline
original_export
original_ik_solve
original_ik_rank
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
planner.satisfied_dependencies
solver
optimizer
optimizer._core.request_guard
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
points
consumed['validation_passes']
diagnostics['original_reference_candidate']
diagnostics['joint_reference']
outcome.execution_binding['verification_joint_limits']
policy_failure
missing
value
grid
source_seconds
source_duration
diagnostics['reference_clock']
consumed['ik_batches']
seeds
ik_success
diagnostics['functional_ik_candidates']
diagnostics['functional_ik_fit_hints']
reference_result
native_result
attempt
same
diagnostics['joint_reference']['source_curve_preserved']
diagnostics['joint_reference']['candidate_kind']
diagnostics['reference_candidate']
recovery
observation_required
consumed['trajectory_solves']
collision_failed
coverage_failed
promoted
diagnostics['collision_witness_update']
planner.seed_strategies
search_started
recovered
report
consumed['search_seconds']
consumed['search_queries']
diagnostics['ompl']
diagnostics['returned_candidate']
diagnostics['diagnostic_trajectory']
diagnostics['diagnostic_candidate_scope']
outcome.success
outcome.trajectory
outcome.execution_binding
manager_type
diagnostics['rollout_query_counts']
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
planner.select_candidate
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

[定义位置](../../../manipulation/planning.py#L4869)。

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
