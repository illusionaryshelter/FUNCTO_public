# `manipulation/execution.py` 实现契约

代码基线：`2498027；包含已实现的中断状态记录改动，待对应实现提交更新`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `pose_goal_schedule`

[定义位置](../../../manipulation/execution.py#L32)。

输入签名：

```python
def pose_goal_schedule(trajectory, progress, timestep):
```

实现说明：

> Return the first original progress time and its strict native due tick.
>
> A terminal progress plateau starts at its first matching knot. The
> physical observation uses the first device tick covering that time;
> one-ULP future times cannot borrow the preceding tick. This function
> supplies timing only and never constructs an owner pose or goal witness.

显式输出分支：

```python
(scheduled, _activation(scheduled, timestep)[1] if np.isfinite(scheduled) else float('inf'))
```

读取的对象属性：

```python
np.flatnonzero
np.abs
np.asarray
trajectory.progress_s
matches.size
trajectory.progress_u
np.interp
trajectory.duration
np.isfinite
```

写入目标：

```python
matches
first_u
scheduled
```

直接调用符号：

```python
np.flatnonzero
np.abs
np.asarray
float
np.interp
np.isfinite
_activation
```

## `_timing_checks`

[定义位置](../../../manipulation/execution.py#L49)。

输入签名：

```python
def _timing_checks(event, check_target, time_s, history, *, goal_id=None, condition_id=None):
```

实现说明：

> Compare one actual witness with its own original physical-time windows.
>
> The caller supplies a measured activation/arrival/acquisition timestamp.
> No reference event time or planning prediction can stand in for it here.
> Unresolved origins remain U and retain their precise missing requirement.

显式输出分支：

```python
checks
```

读取的对象属性：

```python
event.get
constraint.check_target
constraint.goal_id
constraint.condition_id
constraint.source
check.update
constraint.origin_event
checks.append
```

写入目标：

```python
checks
index
value
constraint
check
lower
upper
```

直接调用符号：

```python
enumerate
event.get
_posthoc_geometry_timing
isinstance
EventTimeConstraint
tuple
float
deepcopy
resolve_event_window
check.update
str
checks.append
```

## `_posthoc_geometry_timing`

[定义位置](../../../manipulation/execution.py#L86)。

输入签名：

```python
def _posthoc_geometry_timing(value):
```

实现说明：

> Identify timing whose witness/origin requires a geometry-goal result.
>
> Pose, support and mechanism-coordinate goals constrain the nominal plan.
> Their physical accuracy belongs to the independent result evaluator.

显式输出分支：

```python
bool(row.get('check_target') == 'arrival' and goal and (goal[0] in {'pose_events', 'coordinate_events', 'supports'}) or (origin_goal and origin_goal[0] in {'pose_events', 'coordinate_events', 'supports'}))
```

读取的对象属性：

```python
row.get
(row.get('origin_event') or {}).get
```

写入目标：

```python
row
goal
origin_goal
```

直接调用符号：

```python
isinstance
asdict
row.get
(row.get('origin_event') or {}).get
bool
```

## `FixedFingerTargets`

[定义位置](../../../manipulation/execution.py#L102)。

```python
class FixedFingerTargets():
```

类字段：

```python
joint_names: tuple[str, ...]
source: str
events: tuple[dict, ...]
```

实现说明：

> Explicit native position targets; neither robot feedback nor a curve.
>
> Each row identifies one original ``(skill_id, event_id, kind)``. Event
> times belong exclusively to the exported, retimed arm trajectory. The
> independent coordinates need not have equal magnitudes or opposite signs.
> ``source`` records where the commanded targets were explicitly chosen.

## `FixedFingerTargets.__post_init__`

[定义位置](../../../manipulation/execution.py#L115)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.joint_names
name.strip
self.source
self.source.strip
self.events
row[key].strip
np.asarray
positions.shape
np.isfinite(positions).all
np.isfinite
rows.append
positions.tolist
identities.add
object.__setattr__
```

写入目标：

```python
names
name
rows
identities
row
key
identity
positions
```

直接调用符号：

```python
isinstance
ValueError
tuple
any
name.strip
len
set
self.source.strip
row[key].strip
np.asarray
np.isfinite(positions).all
np.isfinite
rows.append
deepcopy
positions.tolist
identities.add
object.__setattr__
```

## `FixedFingerTargets.from_mapping`

[定义位置](../../../manipulation/execution.py#L143)。

输入签名：

```python
def from_mapping(cls, value):
```

实现说明：

> Read the required JSON input without deriving or defaulting goals.

显式输出分支：

```python
cls(value['joint_names'], value['source'], value['events'])
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
isinstance
set
ValueError
cls
```

## `FixedFingerTargets.bind`

[定义位置](../../../manipulation/execution.py#L149)。

输入签名：

```python
def bind(self, trajectory, *, finger_joint_names, ctrlrange):
```

实现说明：

> Validate every original command and bind its existing retimed time.

显式输出分支：

```python
sorted(bound, key=lambda event: event['time'])
```

读取的对象属性：

```python
self.joint_names
np.asarray
bounds.shape
np.isfinite(bounds).all
np.isfinite
np.any
self.events
trajectory.events
trajectory.duration
bound.append
positions.tolist
self.source
```

写入目标：

```python
bounds
targets
row
commands
event
identities
bound
identity
target
positions
```

直接调用符号：

```python
tuple
ValueError
np.asarray
len
np.isfinite(bounds).all
np.isfinite
np.any
event_occurrence_key
set
zip
bound.append
deepcopy
positions.tolist
sorted
```

## `_finite_array`

[定义位置](../../../manipulation/execution.py#L180)。

输入签名：

```python
def _finite_array(value, shape, label, *, nonnegative=False):
```

显式输出分支：

```python
result.copy()
```

读取的对象属性：

```python
np.asarray
result.shape
np.isfinite(result).all
np.isfinite
np.any
result.copy
```

写入目标：

```python
result
```

直接调用符号：

```python
np.asarray
np.isfinite(result).all
np.isfinite
np.any
ValueError
result.copy
```

## `_feedback_json`

[定义位置](../../../manipulation/execution.py#L188)。

输入签名：

```python
def _feedback_json(value):
```

实现说明：

> Keep a failed report writable; raw nonfinite feedback remains in NPZ.

显式输出分支：

```python
_feedback_json(value.tolist())
{key: _feedback_json(item) for (key, item) in value.items()}
[_feedback_json(item) for item in value]
float(value) if np.isfinite(value) else None
int(value)
value
```

读取的对象属性：

```python
np.generic
value.item
np.ndarray
value.tolist
value.items
np.floating
np.isfinite
np.integer
```

写入目标：

```python
value
key
item
```

直接调用符号：

```python
isinstance
value.item
_feedback_json
value.tolist
value.items
np.isfinite
float
int
```

## `ExecutionLimits`

[定义位置](../../../manipulation/execution.py#L210)。

```python
class ExecutionLimits():
```

类字段：

```python
source: str
joint_names: tuple[str, ...]
arm_joint_names: tuple[str, ...]
finger_joint_names: tuple[str, ...]
arm_position_error: object
arm_velocity_error: object
tcp_position_error_m: float
tcp_rotation_error_rad: float
stop_arm_velocity: object
stop_finger_velocity: object
initial_fingers: dict
events: tuple[dict, ...]
observation_policy: dict | None
```

实现说明：

> Measured-state limits supplied by the same physical planning contract.
>
> No physical tolerance is inferred from controller gains, numerical solver
> tolerances, actuator targets or a previously failed execution. Finger
> bounds describe permitted *actual* positions after each original event;
> targets may lie outside them when contact prevents complete closure. These
> intervals are runtime protection limits, never a planning geometry sweep.
> One arm/TCP tracking band applies to the path and to command-event gates.
> Event times are owned exclusively by ``JointTrajectory``.

## `ExecutionLimits.__post_init__`

[定义位置](../../../manipulation/execution.py#L240)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.source
self.source.strip
name.strip
object.__setattr__
self.joint_names
self.arm_joint_names
self.finger_joint_names
value.tolist
self.stop_finger_velocity
finger_stop.tolist
self.tcp_position_error_m
self.tcp_rotation_error_rad
np.pi
self.initial_fingers
self.events
self.observation_policy
policy['source'].strip
np.isfinite([policy['period_s'], policy['maximum_age_s']]).all
np.isfinite
```

写入目标：

```python
field
value
name
finger_stop
tcp
rows
row
identities
policy
```

直接调用符号：

```python
isinstance
self.source.strip
ValueError
getattr
any
name.strip
len
set
object.__setattr__
tuple
_finite_array
value.tolist
finger_stop.tolist
float
finger_bounds
policy['source'].strip
np.isfinite([policy['period_s'], policy['maximum_age_s']]).all
np.isfinite
deepcopy
```

## `ExecutionLimits.__post_init__.finger_bounds`

[定义位置](../../../manipulation/execution.py#L267)。

输入签名：

```python
def finger_bounds(row, *, event):
```

显式输出分支：

```python
{**deepcopy(row), 'position_bounds': bounds.tolist(), 'maximum_speed': speed.tolist()}
```

读取的对象属性：

```python
keys.update
row[key].strip
self.finger_joint_names
np.any
bounds.tolist
speed.tolist
```

写入目标：

```python
keys
key
bounds
speed
```

直接调用符号：

```python
keys.update
isinstance
set
ValueError
any
row[key].strip
_finite_array
len
np.any
deepcopy
bounds.tolist
speed.tolist
```

## `ExecutionLimits.from_mapping`

[定义位置](../../../manipulation/execution.py#L309)。

输入签名：

```python
def from_mapping(cls, value):
```

显式输出分支：

```python
cls(**deepcopy(value))
```

读取的对象属性：

```python
cls.__dataclass_fields__
```

写入目标：

无显式条目。

直接调用符号：

```python
isinstance
set
ValueError
cls
deepcopy
```

## `ExecutionLimits.describe`

[定义位置](../../../manipulation/execution.py#L314)。

输入签名：

```python
def describe(self):
```

显式输出分支：

```python
_feedback_json({key: deepcopy(getattr(self, key)) for key in self.__dataclass_fields__})
```

读取的对象属性：

```python
self.__dataclass_fields__
```

写入目标：

```python
key
```

直接调用符号：

```python
_feedback_json
deepcopy
getattr
```

## `ExecutionLimits.bind`

[定义位置](../../../manipulation/execution.py#L317)。

输入签名：

```python
def bind(self, trajectory, *, joint_names, arm_joint_names, finger_joint_names, native_joint_ranges, named_velocity_limits):
```

实现说明：

> Bind native hardware ranges and original events, with no new clock.

显式输出分支：

```python
{'joint_ranges': ranges, 'maximum_speed': speed, 'initial_fingers': deepcopy(self.initial_fingers), 'event_fingers': rows}
```

读取的对象属性：

```python
self.joint_names
self.arm_joint_names
self.finger_joint_names
trajectory.joint_names
np.any
trajectory.events
event.get
np.floating
np.isfinite
self.events
self.initial_fingers
np.asarray
np.r_
self.stop_arm_velocity
self.stop_finger_velocity
```

写入目标：

```python
ranges
speed
name
commands
event
identities
rows
row
narm
bounds
```

直接调用符号：

```python
tuple
ValueError
_finite_array
len
np.any
isinstance
set
event_occurrence_key
any
event.get
np.isfinite
deepcopy
np.asarray
```

## `ExecutionGuardError`

[定义位置](../../../manipulation/execution.py#L361)。

```python
class ExecutionGuardError(RuntimeError):
```

类字段：

无显式条目。

实现说明：

> Actual feedback invalidated the explicit execution contract.

## `continuation_execution_inputs`

[定义位置](../../../manipulation/execution.py#L367)。

输入签名：

```python
def continuation_execution_inputs(continuation, finger_targets, execution_limits, *, rebind=None):
```

实现说明：

> Select remaining or explicitly retried original device policies.
>
> The request retains one immutable full device policy. A retry chooses a
> new occurrence of an existing original event, never new actuator targets
> or looser physical limits. The actual last applied interval is preserved
> independently of the candidate's event clock and occurrence numbering.

显式输出分支：

```python
(FixedFingerTargets(finger_targets.joint_names, finger_targets.source, targets), ExecutionLimits.from_mapping(limits))
```

读取的对象属性：

```python
finger_targets.joint_names
finger_targets.source
finger_targets.events
execution_limits.describe
original.get
rebind.get
pending.update
ExecutionLimits.from_mapping
```

写入目标：

```python
original
supplied
policy
key
commands
row
pending
retries
retry_ids
source_events
limits
limits['initial_fingers']
limits['events']
targets
```

直接调用符号：

```python
isinstance
ValueError
list
deepcopy
execution_limits.describe
set
original.get
rebind.get
any
len
tuple
pending.update
FixedFingerTargets
ExecutionLimits.from_mapping
```

## `TrajectoryExecutor`

[定义位置](../../../manipulation/execution.py#L427)。

```python
class TrajectoryExecutor():
```

类字段：

无显式条目。

实现说明：

> Follow the exported cubic using one concrete backend and geometry gate.
>
> No alternate event clock, interpolation or missing-checker fallback is
> provided. The backend exposes actual named robot state independently from
> actuator control. Geometry validation is mandatory for every execution
> mode, including explicitly labelled rejected-candidate diagnostics.

## `TrajectoryExecutor.__init__`

[定义位置](../../../manipulation/execution.py#L436)。

输入签名：

```python
def __init__(self, backend, *, geometry_checker):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
geometry_checker.name.strip
geometry_checker.name
backend.joint_names
geometry_checker.describe
self._geometry_description
self._geometry_description.get
```

写入目标：

```python
self.backend
self.geometry_checker
self.last_execution_report
name
self._geometry_description
self._geometry_pair_count
```

直接调用符号：

```python
isinstance
getattr
geometry_checker.name.strip
tuple
any
callable
ValueError
deepcopy
geometry_checker.describe
self._geometry_description.get
len
```

## `TrajectoryExecutor._geometry_result`

[定义位置](../../../manipulation/execution.py#L456)。

输入签名：

```python
def _geometry_result(self, method, *args, **kwargs):
```

实现说明：

> Require an explicit verdict; checker errors reject physical control.
>
> A malformed result is a checker failure, never an omitted check. The
> caller retains this failure alongside the same actual feedback used
> for tracking checks, then stops before issuing another command.

显式输出分支：

```python
result
{'passed': False, 'distance_query_completed': False, 'violations': [{'kind': 'geometry_checker_error', 'name': self.geometry_checker.name, 'measured': f'{type(error).__name__}: {error}'}]}
```

读取的对象属性：

```python
self.geometry_checker
result.get
self._geometry_pair_count
distances.tolist
self.geometry_checker.name
type(error).__name__
```

写入目标：

```python
result
row
distances
result['distances_m']
result['distance_query_completed']
```

直接调用符号：

```python
getattr(self.geometry_checker, method)
getattr
isinstance
type
result.get
any
set
len
ValueError
deepcopy
_finite_array
distances.tolist
```

## `TrajectoryExecutor._execution_violations`

[定义位置](../../../manipulation/execution.py#L483)。

输入签名：

```python
def _execution_violations(self, feedback, reference_q, reference_dq, reference_tcp, *, limits, binding, finger_state, expected_sim_time, require_stop=False):
```

实现说明：

> Return explicit named violations; neither clip nor repair feedback.

显式输出分支：

```python
failures
failures
```

读取的对象属性：

```python
self.backend
backend.joint_names
backend.arm_joint_names
backend.finger_joint_names
backend.actuator_names
feedback.get
backend.trial_id
backend.feedback_source
np.floating
np.isfinite
backend.sim_time
backend.feedback_id
np.asarray
value.shape
np.flatnonzero
pose.shape
np.isfinite(pose).all
np.allclose
pose[:3, :3].T
np.eye
np.linalg.det
np.linalg
pose.tolist
np.abs
limits.arm_position_error
limits.arm_velocity_error
limits.stop_arm_velocity
np.linalg.norm
Rotation.from_matrix(reference_tcp[:3, :3].T @ pose[:3, :3]).magnitude
Rotation.from_matrix
reference_tcp[:3, :3].T
limits.tcp_position_error_m
limits.tcp_rotation_error_rad
```

写入目标：

```python
backend
failures
field
expected
value
current_time
expected_id
actual
names
actual[field]
index
pose
q
dq
ranges
speed
narm
measured
threshold
finger_bounds
finger_speed
position_error
rotation_error
error
bound
```

直接调用符号：

```python
tuple
feedback.get
add
list
isinstance
np.isfinite
abs
float
np.asarray
len
np.flatnonzero
np.isfinite(pose).all
np.allclose
np.eye
np.linalg.det
pose.tolist
np.abs
np.linalg.norm
Rotation.from_matrix(reference_tcp[:3, :3].T @ pose[:3, :3]).magnitude
Rotation.from_matrix
```

## `TrajectoryExecutor._execution_violations.add`

[定义位置](../../../manipulation/execution.py#L490)。

输入签名：

```python
def add(kind, name, measured, **allowed):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
failures.append
```

写入目标：

无显式条目。

直接调用符号：

```python
failures.append
```

## `TrajectoryExecutor.execute`

[定义位置](../../../manipulation/execution.py#L581)。

输入签名：

```python
def execute(self, result, *, finger_targets, initial_feedback, execution_limits, output, step_callback=None, observation_callback=None, max_physics_steps=30000, start_position_tolerance=2e-06, start_velocity_tolerance=2e-06, max_hold_steps=250):
```

实现说明：

> Follow the exact bound curve within its authorized execution scope.
>
> # BEGIN independent evidence collection has no control authority.
> ``step_callback(model, data, record)`` is an independent evaluator/camera
> sink for the initial state and each completed normal execution step.
> Its first exception is recorded and disables that sink; it cannot reject
> execution, trigger a hold, or mask an online guard failure. Its return
> value is never consumed. Holds have their own physical accounting.
> # END only the separately declared online observer supplies semantics.
> The
> ``observation_callback(feedback, curve_time)`` supplies current allowed
> scene/effect evidence. A declared dependency ends this finite motion
> block; its actual T/F/U result is retained for a fresh scene binding.
> # BEGIN nominal attachment use and predicate estimation are separate.
> Declared attachments belong to the accepted planning model. No measured
> Holding predicate, object pose or additional authorization is required
> to execute that model. Independent task effects still enter through
> ``observation_callback`` with their acquisition time and provenance.
> # END a nominal attachment is never promoted to physical Holding.
> The last native step can finish less than one timestep after the
> barrier's clock time, but its commanded q/dq/progress is clamped to the
> barrier itself. Finger events use the archived reference's original
> bound times and native activation ticks, without changing the arm
> cubic. The same saved C1 finger reference is sampled at each arm query.
> The complete measured entrance is checked independently of all goals.
> No goal from the later curve is sent to an actuator. Physical tracking
> and actual finger intervals require an explicit ``ExecutionLimits``;
> rejection sends bounded device hold and records its measured result.
> # BEGIN one physical allowance includes every attempted native step.
> ``max_physics_steps`` bounds normal playback and both hold paths
> together. Failed or uncertain attempts remain charged; confirmed
> native advancement is recorded separately and owns physical time.
> # END a hold is not an additional allowance outside the request.

显式输出分支：

```python
report
```

读取的对象属性：

```python
time.monotonic
result.trajectory
self.geometry_checker.describe
self.geometry_checker
self._geometry_description.get
self._geometry_description
execution_limits.observation_policy
self.backend
trajectory.joint_names
backend.arm_joint_names
planned.get
initial_feedback.get
np.asarray
expected.shape
actual_frame.shape
np.isfinite(expected).all
np.isfinite
np.isfinite(actual_frame).all
np.allclose
execution_limits.describe
binding.get
backend.describe().get
backend.describe
backend.timestep
FixedFingerTargets.from_mapping
ExecutionLimits.from_mapping
original_targets.events
original_limits.events
finger_targets.joint_names
original_targets.joint_names
finger_targets.source
original_targets.source
original_rows.get
finger_targets.events
original_intervals.get
execution_limits.events
current_limits.items
original_limits.initial_fingers
binding.update
trajectory.executable_until
trajectory.times
trajectory.position
trajectory.velocity
trajectory.progress_u
trajectory.progress_s
trajectory._curve.c
trajectory._curve
np.isfinite(value).all
trajectory.duration
np.ceil
np.isfinite(tolerances).all
np.any
trajectory.sample
start_q.shape
backend.proprioception
backend.joint_names
backend.finger_joint_names
initial_feedback[key].strip
tool_transform.shape
np.isfinite(tool_transform).all
measured.shape
np.isfinite(measured).all
np.max
np.abs
entrance_errors.values
np.array
tolerances.tolist
backend.finger_ctrlrange
finger_targets.bind
trajectory.finger_trajectory
backend.velocity_limits
finger_reference.validate_execution
backend.joint_ranges[narm:].T
backend.joint_ranges
np.stack
finger_reference.events
execution_limits.bind
backend.finger_control
np.isfinite(target_positions).all
output.mkdir
backend.sim_time
backend.trial_id
trajectory.next_verification
finger_reference.describe
trajectory.events
report.update
limit_binding['joint_ranges'].tolist
binding['problem'].get
binding.get('goal_identities', {}).get
row.get
binding['scene'].get
condition.get
owner.get
np.linalg.inv
np.linalg
np.eye
Rotation.from_quat(np.asarray(relation['quaternion_tool_owner'])[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
pose_requirements.get
finite_permission_ids.intersection
finite_permission_ids.update
activation.get
requirement.actual_relation
requirement.progress
holding_bindings.values
mechanism.get
direction_requirements.items
np.flatnonzero
matches.size
np.interp
binding.get('request_ledger', {}).get
active_timing_events.get
trajectory.satisfied_dependencies
event.get
samples.append
backend.reference_tool_pose
self._execution_violations
entry_close.get
q.tolist
dq.tolist
reference_tcp.tolist
target_positions.tolist
self._geometry_result
backend.reference_tool_pose(q).tolist
due.append
target_positions.copy
finger_reference.sample
backend.apply_control
report['superseded_gripper_events'].extend
report['events'].append
controls.append
control['actuator_ctrl'].copy
requested_torques.append
control['requested_arm_torque'].copy
saturated_torques.append
control['arm_torque_saturated'].copy
control_terms[name].append
type(error).__name__
report.setdefault('cleanup_errors', []).append
report.setdefault
type(hold_error).__name__
type(finalization_error).__name__
consumed_ids.update
resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
resource.getrusage
resource.RUSAGE_SELF
report.get
observed.get
trajectory.conditions
required_effects.add
effects.setdefault
active_timing_events.values
timing_values.append
report['timing_checks'].append
request_ledger.setdefault
rows.extend
(output / 'continuation.json').resolve
continuation_path.name
continuation_path.write_text
json.dumps
np.savez_compressed
np.asarray(values).reshape
control_terms.items
np.full
self._geometry_pair_count
np.nan
backend.actuator_names
np.asarray(controls).reshape
np.asarray(requested_torques).reshape
np.asarray(saturated_torques, dtype=bool).reshape
np.arange
(output / name).resolve
feedback.update
feedback_path.write_text
(output / 'execution-report.json').write_text
```

写入目标：

```python
self.last_execution_report
invocation_started
wallclock
binding
trajectory
self._geometry_description
self._geometry_pair_count
backend
planned
key
expected
actual_frame
binding['execution_limits']
policy
declared_controller
actual_controller
original_targets
original_limits
original_rows
row
original_intervals
current_limits
value
physics_budget
prior_physics_steps
max_physics_steps
result.execution_binding
stop
arrays
prefix_count
count
tolerances
start_q
start_dq
start_ddq
_
actual
names
tool_transform
entrance_errors
shape
measured
entrance_errors[key]
narm
start_errors
finger_bounds
command_events
finger_reference
finger_reference_report
speeds
name
limit_binding
finger_state
target_positions
output
started
sim_start
report
event
report['wallclock']
normal_accounting
report['normal_step_accounting']
report['last_confirmed_curve_time_s']
pose_goals
pose_goal_ids
pose_requirements
index
identity
requirement
pose_requirements[identity]
contact_bindings
finite_conditions
contact_rows
finite_permission_ids
holding_bindings
start_progress
condition
condition_id
owners
relations
owner
relation
lower
upper
actual[:3, 3]
actual[:3, :3]
holding_bindings[condition_id]
activation
arrival
commands
rule_ids
contact_bindings[identity]
report['grasp_command_checks']
report['declared_attachment_bindings']
report['pose_goal_evaluation']
report['completed_direction_ids']
report['direction_evidence']
report['direction_checks']
mechanisms
direction_ids
direction_requirements
mechanism
identities
direction_requirements[tuple(identity)]
direction_starts
direction_end_times
end
matches
end_u
direction_end_times[identity]
direction_boundary_polls
retried_directions
direction_starts[key]
report['direction_evidence_scope']
report['history']
report['effect_history']
timing_events
report['timing_checks']
active_timing_events
current
active_timing_events[identity[:2]]
request_origin
report['feedback_policy']
report['feedback_policy']['direction_boundary_observations']
report['observation_checks']
report['object_changes']
controls
requested_torques
saturated_torques
samples
event_index
control_terms
command
satisfied_dependencies
barriers
barrier_index
ordinary_events
item
ordinary_index
report['geometry_check_counts']['event_gate']
arm_control_saturation_count
observation_cache
next_observation_poll
q
dq
ddq
progress
execution_error
preflight_feedback
entry_commands
entry_close
reference_tcp
failures
report['status']
report['feedback_failure']
preflight
report['geometry_preflight']
active
elapsed
before
due
pending_close
contact_check
next_command
next_finger_positions
next_finger_state
checks
next_time
control
contact_check['control_applied']
upcoming
upcoming_close
report['error']
failure_feedback
expected_steps
accounting
report['physics_step_accounting']
sink
normal
sink['completed']
finalization_error
final_feedback
actual_q
command_q
final_elapsed
accounting['step_accounting_exact']
applied_ids
consumed_ids
barrier_command_reached
observed
effects
required_effects
report['effects']
required_values
report['required_effect_ids']
report['effect_result']
timing_values
constraint
report['timing_result']
direction_values
report['direction_result']
request_ledger
request_ledger['requires_continuation']
request_ledger['history']
request_ledger['effect_history']
request_ledger['timing_events']
request_ledger['timing_checks']
rows
report['request_ledger']
report['continuation']
persistence_errors
continuation_path
report['continuation_path']
values
feedback
completed
feedback_path
feedback['execution_status']
```

直接调用符号：

```python
time.monotonic
mark_wallclock
check_result
deepcopy
self.geometry_checker.describe
isinstance
self._geometry_description.get
ValueError
len
type
callable
tuple
planned.get
initial_feedback.get
np.asarray
np.isfinite(expected).all
np.isfinite
np.isfinite(actual_frame).all
np.allclose
execution_limits.describe
binding.get
set
validate_controller_policy
backend.describe().get
backend.describe
float
validate_result_only_goal_ids
FixedFingerTargets.from_mapping
ExecutionLimits.from_mapping
any
original_rows.get
original_intervals.get
current_limits.items
max
min
binding.update
all
np.isfinite(value).all
int
np.ceil
np.isfinite(tolerances).all
np.any
trajectory.sample
_finite_array
backend.proprioception
initial_feedback[key].strip
np.isfinite(tool_transform).all
abs
np.isfinite(measured).all
np.max
np.abs
entrance_errors.values
np.array
tolerances.tolist
finger_targets.bind
finger_reference.validate_execution
np.stack
list
execution_limits.bind
np.isfinite(target_positions).all
Path
output.mkdir
finger_reference.describe
report.update
limit_binding['joint_ranges'].tolist
binding['problem'].get
binding.get('goal_identities', {}).get
zip
PoseEventRequirement
row.get
binding['scene'].get
condition.get
owner.get
map
np.linalg.inv
_pose_array
np.eye
Rotation.from_quat(np.asarray(relation['quaternion_tool_owner'])[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
event_occurrence_key
pose_requirements.get
grasp_contact_rule_ids
validate_grasp_contact_arrival
finite_permission_ids.intersection
finite_permission_ids.update
activation.get
holding_bindings.values
mechanism.get
direction_requirements.items
np.flatnonzero
np.interp
binding.get('request_ledger', {}).get
active_timing_events.get
frozenset
sorted
dependency_already_observed
event.get
samples.append
record
emit_independent_evidence
backend.reference_tool_pose
self._execution_violations
bool
entry_close.get
check_close_binding
q.tolist
dq.tolist
reference_tcp.tolist
target_positions.tolist
ExecutionGuardError
self._geometry_result
backend.reference_tool_pose(q).tolist
guard
consume_ordinary_events
release_dependencies
range
step_accounting
due.append
target_positions.copy
_timing_checks
reject_timing
finger_reference.sample
backend.apply_control
report['superseded_gripper_events'].extend
report['events'].append
timed_witness
controls.append
control['actuator_ctrl'].copy
requested_torques.append
control['requested_arm_torque'].copy
saturated_torques.append
control['arm_torque_saturated'].copy
control_terms[name].append
normal_step
run_hold
report.setdefault('cleanup_errors', []).append
report.setdefault
consumed_ids.update
resource.getrusage
report.get
observed.get
required_effects.add
effects.setdefault
str
active_timing_events.values
enumerate
_posthoc_geometry_timing
timing_values.append
report['timing_checks'].append
request_ledger.setdefault
rows.extend
(output / 'continuation.json').resolve
persist
continuation_path.write_text
json.dumps
_feedback_json
np.savez_compressed
np.asarray(values).reshape
control_terms.items
np.full
np.asarray(controls).reshape
np.asarray(requested_torques).reshape
np.asarray(saturated_torques, dtype=bool).reshape
np.arange
(output / name).resolve
feedback.update
feedback_path.write_text
(output / 'execution-report.json').write_text
```

## `TrajectoryExecutor.execute.mark_wallclock`

[定义位置](../../../manipulation/execution.py#L625)。

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
wallclock['events'][name]
```

直接调用符号：

```python
time.time_ns
time.monotonic
```

## `TrajectoryExecutor.execute.step_accounting`

[定义位置](../../../manipulation/execution.py#L868)。

输入签名：

```python
def step_accounting():
```

实现说明：

> Separate conservative budget charges from confirmed movement.

显式输出分支：

```python
{'charged_steps': charged, 'physical_steps': sum((row['physical_steps'] for row in phases)), 'step_accounting_exact': all((row['step_accounting_exact'] for row in phases)), 'execution_step_limit': max_physics_steps, 'remaining_charged_steps': max(0, max_physics_steps - charged), 'scope': 'This execute invocation; normal playback and both hold paths'}
```

读取的对象属性：

无显式条目。

写入目标：

```python
phases
key
charged
row
```

直接调用符号：

```python
sum
all
max
```

## `TrajectoryExecutor.execute.run_hold`

[定义位置](../../../manipulation/execution.py#L880)。

输入签名：

```python
def run_hold(key, feedback, *, expected_sim_time):
```

实现说明：

> Allocate only the same invocation's unspent native attempts.
>
> Reserve the complete supplied hold allowance until its owned
> result is available. If the hold itself unexpectedly raises or
> loses its report, no unknown work becomes a free retry.

显式输出分支：

无显式条目。

读取的对象属性：

```python
backend.abort_hold
np.r_
execution_limits.stop_arm_velocity
execution_limits.stop_finger_velocity
outcome.get
```

写入目标：

```python
allowance
report[key]
outcome
```

直接调用符号：

```python
min
step_accounting
backend.abort_hold
isinstance
type
outcome.get
ValueError
```

## `TrajectoryExecutor.execute.normal_step`

[定义位置](../../../manipulation/execution.py#L905)。

输入签名：

```python
def normal_step(curve_time):
```

实现说明：

> Charge before integration and retain its result before capture.
>
> The backend publishes accounting at the native integration
> boundary, before forward refresh. A later refresh/record error
> cannot erase that completed step or replace the primary error.

显式输出分支：

无显式条目。

读取的对象属性：

```python
backend.last_step_accounting
backend.step
outcome.get
type(error).__name__
report.setdefault('cleanup_errors', []).append
report.setdefault
```

写入目标：

```python
previous
previous_attempt
normal_accounting['charged_steps']
primary
outcome
normal_accounting['physical_steps']
normal_accounting['step_accounting_exact']
report['executed_steps']
report['last_normal_step_accounting']
report['last_confirmed_curve_time_s']
```

直接调用符号：

```python
step_accounting
ExecutionGuardError
mark_wallclock
backend.step
isinstance
type
outcome.get
ValueError
report.setdefault('cleanup_errors', []).append
report.setdefault
deepcopy
float
```

## `TrajectoryExecutor.execute.dependency_already_observed`

[定义位置](../../../manipulation/execution.py#L1124)。

输入签名：

```python
def dependency_already_observed(event):
```

显式输出分支：

```python
event['requires_evidence'] in satisfied_dependencies and float(event['progress']) <= float(trajectory.progress_s[0])
```

读取的对象属性：

```python
trajectory.progress_s
```

写入目标：

无显式条目。

直接调用符号：

```python
float
```

## `TrajectoryExecutor.execute.check_directions`

[定义位置](../../../manipulation/execution.py#L1150)。

输入签名：

```python
def check_directions(observation, feedback, curve_time, elapsed):
```

实现说明：

> Require current scoped proof, while leaving future/net-prefix unknowns open.
>
> An observer owns coordinate/rate measurement and its T/F/U decision.
> This boundary owns identity, acquisition freshness, interval coverage,
> and physical stopping. In particular, an old completed attempt or a
> planned coordinate derivative cannot provide a new actual witness.

显式输出分支：

无显式条目。

读取的对象属性：

```python
observation.get
fact.get
np.asarray
fact['source'].strip
interval.shape
np.isfinite(interval).all
np.isfinite
trajectory.sample
execution_limits.observation_policy
direction_requirements.items
trajectory.progress_s
direction_starts.get
by_id.get
requirement.get
report['completed_direction_ids'].append
report['direction_evidence'].append
report['direction_checks'].append
failures.append
'/'.join
target_positions.tolist
```

写入目标：

```python
now
facts
by_id
fact
identity
index
interval
by_id[tuple(identity)]
failures
progress
policy
requirement
active
direction_starts[identity]
scope_start
at_end
fresh
row
retained
interval_due
required_now
value
covered
declared_net_origin
row['completion_verified']
violated
unresolved
check
report['status']
report['feedback_failure']
```

直接调用符号：

```python
float
observation.get
isinstance
ValueError
fact.get
len
any
type
tuple
np.asarray
fact['source'].strip
np.isfinite(interval).all
np.isfinite
trajectory.sample
direction_requirements.items
list
direction_starts.get
by_id.get
deepcopy
requirement.get
bool
report['completed_direction_ids'].append
report['direction_evidence'].append
report['direction_checks'].append
failures.append
'/'.join
map
target_positions.tolist
ExecutionGuardError
```

## `TrajectoryExecutor.execute.effect_timestamp`

[定义位置](../../../manipulation/execution.py#L1255)。

输入签名：

```python
def effect_timestamp(identity, fact, observation):
```

实现说明：

> Distinguish current predicate sampling from a witnessed past occurrence.
>
> A newly read assessment of an explicitly declared occurred-event
> fact can remain T. Its actual witness time and occurrence identity
> stay unchanged; the observation's fresh delivery time cannot be
> relabelled as a new physical event or used by a later retry.

显式输出分支：

```python
acquired
float(witnessed)
acquired
```

读取的对象属性：

```python
fact.get
trajectory.conditions
row.get
conditions[0].get
np.floating
np.isfinite
binding.get
trajectory.events
```

写入目标：

```python
acquired
kind
conditions
row
witnessed
identity_key
expected_pair
available
```

直接调用符号：

```python
float
fact.get
row.get
any
ValueError
len
conditions[0].get
isinstance
np.isfinite
event_occurrence_key
binding.get
max
```

## `TrajectoryExecutor.execute.witness`

[定义位置](../../../manipulation/execution.py#L1290)。

输入签名：

```python
def witness(event, check_target, time_s, value, source, **details):
```

实现说明：

> Append actual evidence without changing an existing first T origin.

显式输出分支：

```python
row
```

读取的对象属性：

```python
item.get
row.get
report['history'].append
```

写入目标：

```python
row
previous
item
```

直接调用符号：

```python
float
deepcopy
event_occurrence_key
item.get
row.get
report['history'].append
```

## `TrajectoryExecutor.execute.timed_witness`

[定义位置](../../../manipulation/execution.py#L1305)。

输入签名：

```python
def timed_witness(event, check_target, time_s, value, source, **details):
```

显式输出分支：

```python
checks
```

读取的对象属性：

```python
details.get
report['timing_checks'].append
```

写入目标：

```python
checks
check
check['evidence_value']
```

直接调用符号：

```python
_timing_checks
details.get
report['timing_checks'].append
witness
```

## `TrajectoryExecutor.execute.reject_timing`

[定义位置](../../../manipulation/execution.py#L1317)。

输入签名：

```python
def reject_timing(feedback, elapsed, checks, *, event=None, point='event_time'):
```

显式输出分支：

```python
None
```

读取的对象属性：

```python
trajectory.events
row.get
target_positions.tolist
```

写入目标：

```python
candidate_ids
row
checks
failures
report['status']
report['feedback_failure']
```

直接调用符号：

```python
event_occurrence_key
row.get
deepcopy
min
float
target_positions.tolist
ExecutionGuardError
```

## `TrajectoryExecutor.execute.resolve_pending_timing`

[定义位置](../../../manipulation/execution.py#L1337)。

输入签名：

```python
def resolve_pending_timing(feedback, elapsed):
```

实现说明：

> Recheck retained actual witnesses once their future origin exists.

显式输出分支：

无显式条目。

读取的对象属性：

```python
pending.get
row.get
fact.get
report['timing_checks'].append
```

写入目标：

```python
events
row
pending
event
constraint
facts
fact
checks
check
check['evidence_value']
active
```

直接调用符号：

```python
event_occurrence_key
list
pending.get
_posthoc_geometry_timing
tuple
row.get
_timing_checks
fact.get
report['timing_checks'].append
reject_timing
```

## `TrajectoryExecutor.execute.check_deadlines`

[定义位置](../../../manipulation/execution.py#L1366)。

输入签名：

```python
def check_deadlines(feedback, elapsed):
```

实现说明：

> An expired absent witness is incomplete U, never a fabricated F.

显式输出分支：

无显式条目。

读取的对象属性：

```python
active_timing_events.values
event.get
row.get
constraint.check_target
constraint.source
report['timing_checks'].append
```

写入目标：

```python
now
event
index
item
constraint
window
row
check
```

直接调用符号：

```python
float
active_timing_events.values
enumerate
event.get
_posthoc_geometry_timing
EventTimeConstraint
resolve_event_window
any
row.get
event_occurrence_key
list
report['timing_checks'].append
reject_timing
```

## `TrajectoryExecutor.execute.observe_timed_effects`

[定义位置](../../../manipulation/execution.py#L1393)。

输入签名：

```python
def observe_timed_effects(observation, feedback, elapsed):
```

实现说明：

> Use acquisition time for effects; cached old binding truth is ineligible.

显式输出分支：

无显式条目。

读取的对象属性：

```python
active_timing_events.values
event.get
observation.get('effects', {}).get
observation.get
evidence.get
row.get
```

写入目标：

```python
event
identities
item
identity
evidence
acquisition
indexes
index
row
checks
```

直接调用符号：

```python
active_timing_events.values
event.get
observation.get('effects', {}).get
observation.get
effect_timestamp
evidence.get
enumerate
all
any
row.get
event_occurrence_key
_timing_checks
timed_witness
reject_timing
```

## `TrajectoryExecutor.execute.observation_at`

[定义位置](../../../manipulation/execution.py#L1428)。

输入签名：

```python
def observation_at(feedback, curve_time, elapsed):
```

显式输出分支：

```python
None
deepcopy(observation_cache)
current
{'source': 'Declared observation freshness policy', 'timestamp_s': now, 'effects': {identity: {'value': 'U', 'source': 'No unexpired effect observation'} for identity in identities}}
```

读取的对象属性：

```python
execution_limits.observation_policy
trajectory.sample
direction_requirements.items
boundaries.append
np.floor
direction_boundary_polls.update
observation.get
observation['source'].strip
np.isfinite
np.nan
observation.get('effects', {}).items
fact.get
report['effect_history'].append
report['object_changes'].extend
report['observation_checks'].append
target_positions.tolist
observation_cache.get
current.get
trajectory.events
identities.update
trajectory.conditions
condition.get
```

写入目标：

```python
policy
now
progress
boundaries
identity
requirement
side
reached
key
periodic
due
missed
next_observation_poll
observation
check
value
field
fact
row
observation_cache
check['sample']
check['acquired_at_s']
check['age_s']
fresh
missing
report['status']
report['feedback_failure']
current
current['effects']
identities
event
condition
```

直接调用符号：

```python
float
trajectory.sample
direction_requirements.items
list
boundaries.append
max
int
np.floor
direction_boundary_polls.update
observation_callback
deepcopy
isinstance
observation.get
observation['source'].strip
np.isfinite
ValueError
type
observation.get('effects', {}).items
fact.get
effect_timestamp
report['effect_history'].append
report['object_changes'].extend
report['observation_checks'].append
target_positions.tolist
ExecutionGuardError
observation_cache.get
current.get
str
identities.update
condition.get
```

## `TrajectoryExecutor.execute.record`

[定义位置](../../../manipulation/execution.py#L1530)。

输入签名：

```python
def record(elapsed, desired_arm_q, arm_curve_dq, arm_curve_ddq, progress, feedback=None):
```

显式输出分支：

```python
{'elapsed': float(elapsed), 'sim_time': feedback['sim_time'], 'curve_time': min(float(elapsed), stop), 'command_q': np.concatenate((np.asarray(desired_arm_q), target_positions)), 'arm_curve_dq': np.asarray(arm_curve_dq).copy(), 'arm_curve_ddq': np.asarray(arm_curve_ddq).copy(), 'reference_finger_q': reference_fingers[0].copy(), 'reference_finger_dq': reference_fingers[1].copy(), 'q': feedback['q'], 'dq': feedback['dq'], 'tool_pose': feedback['tool_pose'], 'tool_pose_source': feedback['tool_pose_source'], 'tool_pose_is_independent_observation': feedback['tool_pose_is_independent_observation'], 'reference_tool_pose': backend.reference_tool_pose(desired_arm_q), 'feedback': deepcopy(feedback), 'finger_state': deepcopy(finger_state), 'geometry_check': None, 'progress': float(progress), 'gripper_command': command}
```

读取的对象属性：

```python
backend.proprioception
finger_reference.sample
np.full
backend.finger_joint_names
np.nan
np.concatenate
np.asarray
np.asarray(arm_curve_dq).copy
np.asarray(arm_curve_ddq).copy
reference_fingers[0].copy
reference_fingers[1].copy
backend.reference_tool_pose
```

写入目标：

```python
feedback
reference_fingers
```

直接调用符号：

```python
backend.proprioception
finger_reference.sample
min
float
np.full
len
np.concatenate
np.asarray
np.asarray(arm_curve_dq).copy
np.asarray(arm_curve_ddq).copy
reference_fingers[0].copy
reference_fingers[1].copy
backend.reference_tool_pose
deepcopy
```

## `TrajectoryExecutor.execute.check_close_binding`

[定义位置](../../../manipulation/execution.py#L1558)。

输入签名：

```python
def check_close_binding(feedback, *, curve_time, physical_elapsed, point, contact_event, failures):
```

实现说明：

> Keep command timing without claiming actual target arrival.
>
> Never compare TCP and predicted target here: there is no current
> target truth. A close command neither proves contact nor Holding.

显式输出分支：

```python
contact_check
```

读取的对象属性：

```python
trajectory.sample
np.asarray(feedback['q']).tolist
np.asarray
report['grasp_command_checks'].append
failures.append
```

写入目标：

```python
contact_check
gate
condition
rule_ids
current_progress
in_interval
due_now
```

直接调用符号：

```python
event_occurrence_key
float
trajectory.sample
list
deepcopy
np.asarray(feedback['q']).tolist
np.asarray
report['grasp_command_checks'].append
failures.append
```

## `TrajectoryExecutor.execute.guard`

[定义位置](../../../manipulation/execution.py#L1593)。

输入签名：

```python
def guard(feedback, *, curve_time, physical_elapsed, point, event=None, sample=None, contact_event=None):
```

显式输出分支：

```python
contact_check
```

读取的对象属性：

```python
observation.get('effects', {}).values
observation.get
effect.get
target_positions.tolist
trajectory.sample
backend.reference_tool_pose
event.get
barrier.get
self._execution_violations
self._geometry_result
failures.extend
check.update
report['event_gate_checks'].append
reference_q.tolist
reference_dq.tolist
reference_tcp.tolist
```

写入目标：

```python
observation
effect
report['latest_observation']
report['status']
report['feedback_failure']
reference_q
reference_dq
_
reference_tcp
barrier
requires_stop
failures
contact_check
geometry
report['geometry_check_counts'][point]
report['last_geometry_check']
sample['geometry_check']
check
check['geometry_check']
```

直接调用符号：

```python
observation_at
observation.get('effects', {}).values
observation.get
isinstance
effect.get
ValueError
deepcopy
float
target_positions.tolist
ExecutionGuardError
observe_timed_effects
check_directions
resolve_pending_timing
trajectory.sample
backend.reference_tool_pose
len
bool
event.get
barrier.get
abs
self._execution_violations
check_close_binding
self._geometry_result
failures.extend
check.update
report['event_gate_checks'].append
reference_q.tolist
reference_dq.tolist
reference_tcp.tolist
check_deadlines
```

## `TrajectoryExecutor.execute.consume_ordinary_events`

[定义位置](../../../manipulation/execution.py#L1667)。

输入签名：

```python
def consume_ordinary_events(feedback, elapsed):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
event.get
report['stopped_events'].append
np.asarray(feedback['dq'])[:narm].tolist
np.asarray
execution_limits.stop_arm_velocity
report.get('latest_observation', {}).get
report.get
trajectory.conditions
condition.get
available.get
evidence.items
fact.get
fact.update
evidence.values
report['evaluated_events'].append
report['reached_events'].append
```

写入目标：

```python
event
available
identities
condition
evidence
identity
observation
fact
acquired
acquisition
checks
row
values
value
ordinary_index
```

直接调用符号：

```python
len
min
float
event.get
guard
report['stopped_events'].append
deepcopy
np.asarray(feedback['dq'])[:narm].tolist
np.asarray
list
report.get('latest_observation', {}).get
report.get
condition.get
available.get
evidence.items
effect_timestamp
fact.get
fact.update
_timing_checks
any
timed_witness
evidence.values
report['evaluated_events'].append
all
report['reached_events'].append
reject_timing
```

## `TrajectoryExecutor.execute.release_dependencies`

[定义位置](../../../manipulation/execution.py#L1726)。

输入签名：

```python
def release_dependencies(feedback, elapsed):
```

显式输出分支：

```python
False
False
True
```

读取的对象属性：

```python
report.get('latest_observation', {}).get
report.get
available_effects.get
active_timing_events.values
timed_event.get
row.get
report['dependency_outcomes'].append
report['verified_events'].append
```

写入目标：

```python
event
checks
available_effects
evidence
timed_event
constraint_index
constraint
witnessed
row
report['status']
barrier_index
report['evidence_passed']
report['next_verification']
report['planned_stop_s']
```

直接调用符号：

```python
len
_timing_checks
float
reject_timing
deepcopy
report.get('latest_observation', {}).get
report.get
available_effects.get
str
active_timing_events.values
enumerate
timed_event.get
_posthoc_geometry_timing
any
event_occurrence_key
row.get
report['dependency_outcomes'].append
run_hold
report['verified_events'].append
timed_witness
```

## `TrajectoryExecutor.execute.emit_independent_evidence`

[定义位置](../../../manipulation/execution.py#L1781)。

输入签名：

```python
def emit_independent_evidence(sample, phase):
```

显式输出分支：

```python
None
```

读取的对象属性：

```python
backend.native_model
backend.native_data
type(error).__name__
```

写入目标：

```python
sink
sink['attempted_callbacks']
sink['disabled']
sink['first_error']
sink['successful_callbacks']
```

直接调用符号：

```python
step_callback
deepcopy
type
str
```

## `TrajectoryExecutor.execute.persist`

[定义位置](../../../manipulation/execution.py#L2158)。

输入签名：

```python
def persist(name, write):
```

实现说明：

> Attempt every artifact and retain exact write failures in memory.

显式输出分支：

```python
False
True
```

读取的对象属性：

```python
persistence_errors.append
report.setdefault('persistence_errors', []).append
report.setdefault
type(error).__name__
```

写入目标：

无显式条目。

直接调用符号：

```python
write
persistence_errors.append
report.setdefault('persistence_errors', []).append
report.setdefault
type
```
