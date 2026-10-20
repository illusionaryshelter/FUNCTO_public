# `manipulation/execution.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `pose_goal_schedule`

[定义位置](../../../manipulation/execution.py#L32)。

输入签名：

```python
def pose_goal_schedule(trajectory, progress, timestep):
```

直接读取的对象属性：`trajectory.progress_s`、`trajectory.progress_u`、`trajectory.duration`。

输出分支的源码表达式：

```python
(scheduled, _activation(scheduled, timestep)[1] if np.isfinite(scheduled) else float('inf'))
```

直接调用：

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

输出分支的源码表达式：

```python
checks
```

直接调用：

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

输出分支的源码表达式：

```python
bool(row.get('check_target') == 'arrival' and goal and (goal[0] in {'pose_events', 'coordinate_events', 'supports'}) or (origin_goal and origin_goal[0] in {'pose_events', 'coordinate_events', 'supports'}))
```

直接调用：

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
class FixedFingerTargets:
    joint_names: tuple[str, ...]
    source: str
    events: tuple[dict, ...]
```

该类型的直接方法：`__post_init__`、`from_mapping`、`bind`。

## `FixedFingerTargets.__post_init__`

[定义位置](../../../manipulation/execution.py#L115)。

输入签名：

```python
def FixedFingerTargets.__post_init__(self):
```

直接读取的对象属性：`self.joint_names`、`self.source`、`self.events`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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
@classmethod
def FixedFingerTargets.from_mapping(cls, value):
```

输出分支的源码表达式：

```python
cls(value['joint_names'], value['source'], value['events'])
```

直接调用：

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
def FixedFingerTargets.bind(self, trajectory, *, finger_joint_names, ctrlrange):
```

直接读取的对象属性：`self.joint_names`、`self.events`、`trajectory.events`、`trajectory.duration`、`self.source`。

输出分支的源码表达式：

```python
sorted(bound, key=lambda event: event['time'])
```

直接调用：

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

直接读取的对象属性：`result.shape`、`result.copy`。

输出分支的源码表达式：

```python
result.copy()
```

直接调用：

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

输出分支的源码表达式：

```python
_feedback_json(value.tolist())
{key: _feedback_json(item) for (key, item) in value.items()}
[_feedback_json(item) for item in value]
float(value) if np.isfinite(value) else None
int(value)
value
```

直接调用：

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
class ExecutionLimits:
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

该类型的直接方法：`__post_init__`、`from_mapping`、`describe`、`bind`。

## `ExecutionLimits.__post_init__`

[定义位置](../../../manipulation/execution.py#L240)。

输入签名：

```python
def ExecutionLimits.__post_init__(self):
```

直接读取的对象属性：`self.source`、`self.joint_names`、`self.arm_joint_names`、`self.finger_joint_names`、`self.stop_finger_velocity`、`self.tcp_position_error_m`、`self.tcp_rotation_error_rad`、`self.initial_fingers`、`self.events`、`self.observation_policy`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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
def ExecutionLimits.__post_init__.finger_bounds(row, *, event):
```

直接读取的对象属性：`self.finger_joint_names`。

输出分支的源码表达式：

```python
{**deepcopy(row), 'position_bounds': bounds.tolist(), 'maximum_speed': speed.tolist()}
```

直接调用：

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
@classmethod
def ExecutionLimits.from_mapping(cls, value):
```

输出分支的源码表达式：

```python
cls(**deepcopy(value))
```

直接调用：

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
def ExecutionLimits.describe(self):
```

直接读取的对象属性：`self.__dataclass_fields__`。

输出分支的源码表达式：

```python
_feedback_json({key: deepcopy(getattr(self, key)) for key in self.__dataclass_fields__})
```

直接调用：

```python
_feedback_json
deepcopy
getattr
```

## `ExecutionLimits.bind`

[定义位置](../../../manipulation/execution.py#L317)。

输入签名：

```python
def ExecutionLimits.bind(self, trajectory, *, joint_names, arm_joint_names, finger_joint_names, native_joint_ranges, named_velocity_limits):
```

直接读取的对象属性：`self.joint_names`、`self.arm_joint_names`、`self.finger_joint_names`、`trajectory.joint_names`、`trajectory.events`、`self.events`、`self.initial_fingers`、`self.stop_arm_velocity`、`self.stop_finger_velocity`。

输出分支的源码表达式：

```python
{'joint_ranges': ranges, 'maximum_speed': speed, 'initial_fingers': deepcopy(self.initial_fingers), 'event_fingers': rows}
```

直接调用：

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
    ...
    ...
```

## `continuation_execution_inputs`

[定义位置](../../../manipulation/execution.py#L367)。

输入签名：

```python
def continuation_execution_inputs(continuation, finger_targets, execution_limits, *, rebind=None):
```

输出分支的源码表达式：

```python
(FixedFingerTargets(finger_targets.joint_names, finger_targets.source, targets), ExecutionLimits.from_mapping(limits))
```

直接写入的状态或数组位置：`limits['initial_fingers']`、`limits['events']`。

直接调用：

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
class TrajectoryExecutor:
    ...
    ...
```

该类型的直接方法：`__init__`、`_geometry_result`、`_execution_violations`、`execute`。

## `TrajectoryExecutor.__init__`

[定义位置](../../../manipulation/execution.py#L436)。

输入签名：

```python
def TrajectoryExecutor.__init__(self, backend, *, geometry_checker):
```

直接读取的对象属性：`self._geometry_description`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.backend`、`self.geometry_checker`、`self.last_execution_report`、`self._geometry_description`、`self._geometry_pair_count`。

直接调用：

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
def TrajectoryExecutor._geometry_result(self, method, *args, **kwargs):
```

直接读取的对象属性：`self.geometry_checker`、`result.get`、`self._geometry_pair_count`。

输出分支的源码表达式：

```python
result
{'passed': False, 'distance_query_completed': False, 'violations': [{'kind': 'geometry_checker_error', 'name': self.geometry_checker.name, 'measured': f'{type(error).__name__}: {error}'}]}
```

直接写入的状态或数组位置：`result['distances_m']`、`result['distance_query_completed']`。

直接调用：

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
def TrajectoryExecutor._execution_violations(self, feedback, reference_q, reference_dq, reference_tcp, *, limits, binding, finger_state, expected_sim_time, require_stop=False):
```

直接读取的对象属性：`self.backend`、`feedback.get`。

输出分支的源码表达式：

```python
failures
```

直接写入的状态或数组位置：`actual[field]`。

直接调用：

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
def TrajectoryExecutor._execution_violations.add(kind, name, measured, **allowed):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
failures.append
```

## `TrajectoryExecutor.execute`

[定义位置](../../../manipulation/execution.py#L581)。

输入签名：

```python
def TrajectoryExecutor.execute(self, result, *, finger_targets, initial_feedback, execution_limits, output, step_callback=None, observation_callback=None, max_physics_steps=30000, start_position_tolerance=2e-06, start_velocity_tolerance=2e-06, max_hold_steps=250):
```

直接读取的对象属性：`result.trajectory`、`self.geometry_checker`、`self._geometry_description`、`self.backend`、`trajectory.joint_names`、`trajectory.executable_until`、`trajectory.times`、`trajectory.position`、`trajectory.velocity`、`trajectory.progress_u`、`trajectory.progress_s`、`trajectory._curve`、`trajectory.duration`、`trajectory.sample`、`trajectory.finger_trajectory`、`trajectory.next_verification`、`trajectory.events`、`trajectory.satisfied_dependencies`、`self._execution_violations`、`self._geometry_result`、`trajectory.conditions`、`feedback.update`。

输出分支的源码表达式：

```python
report
```

直接写入的状态或数组位置：`self.last_execution_report`、`self._geometry_description`、`self._geometry_pair_count`、`binding['execution_limits']`、`result.execution_binding`、`entrance_errors[key]`、`report['wallclock']`、`report['normal_step_accounting']`、`report['last_confirmed_curve_time_s']`、`pose_requirements[identity]`、`actual[:3, 3]`、`actual[:3, :3]`、`holding_bindings[identity]`、`contact_bindings[identity]`、`report['grasp_command_checks']`、`report['declared_attachment_bindings']`、`report['pose_goal_evaluation']`、`report['completed_direction_ids']`、`report['direction_evidence']`、`report['direction_checks']`、`direction_requirements[tuple(identity)]`、`direction_end_times[identity]`、`direction_starts[key]`、`report['direction_evidence_scope']`、`report['history']`、`report['effect_history']`、`report['timing_checks']`、`active_timing_events[identity[:2]]`、`report['feedback_policy']`、`report['feedback_policy']['direction_boundary_observations']`、`report['observation_checks']`、`report['object_changes']`、`report['geometry_check_counts']['event_gate']`、`report['status']`、`report['feedback_failure']`、`report['geometry_preflight']`、`contact_check['control_applied']`、`report['error']`、`report['physics_step_accounting']`、`sink['completed']`、`accounting['step_accounting_exact']`、`report['effects']`、`report['required_effect_ids']`、`report['effect_result']`、`report['timing_result']`、`report['direction_result']`、`request_ledger['requires_continuation']`、`request_ledger['history']`、`request_ledger['effect_history']`、`request_ledger['timing_events']`、`request_ledger['timing_checks']`、`report['request_ledger']`、`report['continuation']`、`report['continuation_path']`、`feedback['execution_status']`。

直接调用：

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
binding['scene'].get
row.get
HoldingContact
condition.get
owner.get
map
np.linalg.inv
_pose_array
np.eye
Rotation.from_quat(np.asarray(relation['quaternion_tool_owner'])[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
rules_by_id.get
finite_permission_ids.add
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
(output / name).resolve
feedback.update
```

## `TrajectoryExecutor.execute.mark_wallclock`

[定义位置](../../../manipulation/execution.py#L625)。

输入签名：

```python
def TrajectoryExecutor.execute.mark_wallclock(name):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`wallclock['events'][name]`。

直接调用：

```python
time.time_ns
time.monotonic
```

## `TrajectoryExecutor.execute.step_accounting`

[定义位置](../../../manipulation/execution.py#L868)。

输入签名：

```python
def TrajectoryExecutor.execute.step_accounting():
```

输出分支的源码表达式：

```python
{'charged_steps': charged, 'physical_steps': sum((row['physical_steps'] for row in phases)), 'step_accounting_exact': all((row['step_accounting_exact'] for row in phases)), 'execution_step_limit': max_physics_steps, 'remaining_charged_steps': max(0, max_physics_steps - charged), 'scope': 'This execute invocation; normal playback and both hold paths'}
```

直接调用：

```python
sum
all
max
```

## `TrajectoryExecutor.execute.run_hold`

[定义位置](../../../manipulation/execution.py#L880)。

输入签名：

```python
def TrajectoryExecutor.execute.run_hold(key, feedback, *, expected_sim_time):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`report[key]`。

直接调用：

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
def TrajectoryExecutor.execute.normal_step(curve_time):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`normal_accounting['charged_steps']`、`normal_accounting['physical_steps']`、`normal_accounting['step_accounting_exact']`、`report['executed_steps']`、`report['last_normal_step_accounting']`、`report['last_confirmed_curve_time_s']`。

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1149)。

输入签名：

```python
def TrajectoryExecutor.execute.dependency_already_observed(event):
```

直接读取的对象属性：`trajectory.progress_s`。

输出分支的源码表达式：

```python
event['requires_evidence'] in satisfied_dependencies and float(event['progress']) <= float(trajectory.progress_s[0])
```

直接调用：

```python
float
```

## `TrajectoryExecutor.execute.check_directions`

[定义位置](../../../manipulation/execution.py#L1175)。

输入签名：

```python
def TrajectoryExecutor.execute.check_directions(observation, feedback, curve_time, elapsed):
```

直接读取的对象属性：`trajectory.sample`、`trajectory.progress_s`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`by_id[tuple(identity)]`、`direction_starts[identity]`、`row['completion_verified']`、`report['status']`、`report['feedback_failure']`。

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1280)。

输入签名：

```python
def TrajectoryExecutor.execute.effect_timestamp(identity, fact, observation):
```

直接读取的对象属性：`trajectory.conditions`、`trajectory.events`。

输出分支的源码表达式：

```python
acquired
float(witnessed)
```

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1315)。

输入签名：

```python
def TrajectoryExecutor.execute.witness(event, check_target, time_s, value, source, **details):
```

输出分支的源码表达式：

```python
row
```

直接调用：

```python
float
deepcopy
event_occurrence_key
item.get
row.get
report['history'].append
```

## `TrajectoryExecutor.execute.timed_witness`

[定义位置](../../../manipulation/execution.py#L1330)。

输入签名：

```python
def TrajectoryExecutor.execute.timed_witness(event, check_target, time_s, value, source, **details):
```

输出分支的源码表达式：

```python
checks
```

直接写入的状态或数组位置：`check['evidence_value']`。

直接调用：

```python
_timing_checks
details.get
report['timing_checks'].append
witness
```

## `TrajectoryExecutor.execute.reject_timing`

[定义位置](../../../manipulation/execution.py#L1342)。

输入签名：

```python
def TrajectoryExecutor.execute.reject_timing(feedback, elapsed, checks, *, event=None, point='event_time'):
```

直接读取的对象属性：`trajectory.events`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`report['status']`、`report['feedback_failure']`。

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1362)。

输入签名：

```python
def TrajectoryExecutor.execute.resolve_pending_timing(feedback, elapsed):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`check['evidence_value']`。

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1391)。

输入签名：

```python
def TrajectoryExecutor.execute.check_deadlines(feedback, elapsed):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1418)。

输入签名：

```python
def TrajectoryExecutor.execute.observe_timed_effects(observation, feedback, elapsed):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1453)。

输入签名：

```python
def TrajectoryExecutor.execute.observation_at(feedback, curve_time, elapsed):
```

直接读取的对象属性：`trajectory.sample`、`trajectory.events`、`trajectory.conditions`。

输出分支的源码表达式：

```python
None
deepcopy(observation_cache)
current
{'source': 'Declared observation freshness policy', 'timestamp_s': now, 'effects': {identity: {'value': 'U', 'source': 'No unexpired effect observation'} for identity in identities}}
```

直接写入的状态或数组位置：`check['sample']`、`check['acquired_at_s']`、`check['age_s']`、`report['status']`、`report['feedback_failure']`、`current['effects']`。

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1555)。

输入签名：

```python
def TrajectoryExecutor.execute.record(elapsed, desired_arm_q, arm_curve_dq, arm_curve_ddq, progress, feedback=None):
```

输出分支的源码表达式：

```python
{'elapsed': float(elapsed), 'sim_time': feedback['sim_time'], 'curve_time': min(float(elapsed), stop), 'command_q': np.concatenate((np.asarray(desired_arm_q), target_positions)), 'arm_curve_dq': np.asarray(arm_curve_dq).copy(), 'arm_curve_ddq': np.asarray(arm_curve_ddq).copy(), 'reference_finger_q': reference_fingers[0].copy(), 'reference_finger_dq': reference_fingers[1].copy(), 'q': feedback['q'], 'dq': feedback['dq'], 'tool_pose': feedback['tool_pose'], 'tool_pose_source': feedback['tool_pose_source'], 'tool_pose_is_independent_observation': feedback['tool_pose_is_independent_observation'], 'reference_tool_pose': backend.reference_tool_pose(desired_arm_q), 'feedback': deepcopy(feedback), 'finger_state': deepcopy(finger_state), 'geometry_check': None, 'progress': float(progress), 'gripper_command': command}
```

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1583)。

输入签名：

```python
def TrajectoryExecutor.execute.check_close_binding(feedback, *, curve_time, physical_elapsed, point, contact_event, failures):
```

直接读取的对象属性：`trajectory.sample`。

输出分支的源码表达式：

```python
contact_check
```

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1618)。

输入签名：

```python
def TrajectoryExecutor.execute.guard(feedback, *, curve_time, physical_elapsed, point, event=None, sample=None, contact_event=None):
```

直接读取的对象属性：`trajectory.sample`、`self._execution_violations`、`self._geometry_result`。

输出分支的源码表达式：

```python
contact_check
```

直接写入的状态或数组位置：`report['latest_observation']`、`report['status']`、`report['feedback_failure']`、`report['geometry_check_counts'][point]`、`report['last_geometry_check']`、`sample['geometry_check']`、`check['geometry_check']`。

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1692)。

输入签名：

```python
def TrajectoryExecutor.execute.consume_ordinary_events(feedback, elapsed):
```

直接读取的对象属性：`trajectory.conditions`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1751)。

输入签名：

```python
def TrajectoryExecutor.execute.release_dependencies(feedback, elapsed):
```

输出分支的源码表达式：

```python
False
True
```

直接写入的状态或数组位置：`report['status']`、`report['evidence_passed']`、`report['next_verification']`、`report['planned_stop_s']`。

直接调用：

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

[定义位置](../../../manipulation/execution.py#L1806)。

输入签名：

```python
def TrajectoryExecutor.execute.emit_independent_evidence(sample, phase):
```

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`sink['attempted_callbacks']`、`sink['disabled']`、`sink['first_error']`、`sink['successful_callbacks']`。

直接调用：

```python
step_callback
deepcopy
type
str
```

## `TrajectoryExecutor.execute.persist`

[定义位置](../../../manipulation/execution.py#L2183)。

输入签名：

```python
def TrajectoryExecutor.execute.persist(name, write):
```

输出分支的源码表达式：

```python
False
True
```

直接调用：

```python
write
persistence_errors.append
report.setdefault('persistence_errors', []).append
report.setdefault
type
```
