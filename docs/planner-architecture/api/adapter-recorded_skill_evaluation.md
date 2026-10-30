# `tests/planning/recorded_skill_evaluation.py` 实现契约

代码基线：`9cdb433`。
职责和数据语义见[组件说明](../11-device-control-and-evaluation.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_actual_pose_goal`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L47)。

输入签名：

```python
def _actual_pose_goal(requirement, feedback, observation):
```

实现说明：

> Evaluate an original pose allowed set in the independent simulation sink.
>
> NEVER call this from the production executor. A robot FK estimate compared
> with a frozen predicted target is not a measured physical tool-to-target
> relation. Target-relative error (including missing target truth) can only
> grade the recorded simulation; it must not veto close, abort, or replan.
> The evaluator supplies native poses with provenance and keeps the original
> tolerances. Its result is never returned as an online control observation.

显式输出分支：

```python
measured
measured
```

读取的对象属性：

```python
requirement.actual_relation
observation.get
np.asarray
relation.get
conditions.get
requirement.tool_from_function
measured['measurement_sources'].append
missing.append
requirement.target
function.tolist
target.tolist
'; '.join
np.eye
actual_requirement.errors_numpy
measured.update
errors.items
requirement.component_tolerances.items
requirement.component_tolerances
```

写入目标：

```python
measured
relation
poses
conditions
missing
function
owner
target
measured['actual_function_pose']
measured['actual_target_pose']
measured['reason']
actual_requirement
errors
name
value
measured['value']
bound
```

直接调用符号：

```python
observation.get
owner_pose
np.asarray
relation.get
conditions.get
actual_tool
measured['measurement_sources'].append
float
missing.append
function.tolist
target.tolist
'; '.join
replace
np.eye
actual_requirement.errors_numpy
measured.update
errors.items
all
requirement.component_tolerances.items
```

## `_actual_pose_goal.owner_pose`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L65)。

输入签名：

```python
def owner_pose(identity):
```

显式输出分支：

```python
None
pose
```

读取的对象属性：

```python
missing.append
measured['measurement_sources'].append
pose.tolist
```

写入目标：

```python
pose
```

直接调用符号：

```python
missing.append
_pose_array
measured['measurement_sources'].append
float
pose.tolist
```

## `_actual_pose_goal.actual_tool`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L74)。

输入签名：

```python
def actual_tool():
```

显式输出分支：

```python
pose
```

读取的对象属性：

```python
measured['measurement_sources'].append
pose.tolist
```

写入目标：

```python
pose
```

直接调用符号：

```python
_pose_array
measured['measurement_sources'].append
float
pose.tolist
```

## `validate_diagnostic_goal_ids`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L123)。

输入签名：

```python
def validate_diagnostic_goal_ids(binding, values):
```

实现说明：

> Validate explicit grading roles for the current geometric goal identities.
>
> Pose, support and mechanism-coordinate goals are evaluated after execution.
> Each remains required for success unless separately selected as diagnostic.
> Execution-policy lists and event references do not choose that grading role;
> process and effect predicates cannot be selected here.

显式输出分支：

```python
validate_result_only_goal_ids(binding, values)
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
validate_result_only_goal_ids
```

## `summarize_skill_evidence`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L136)。

输入签名：

```python
def summarize_skill_evidence(report, *, binding, trajectory, diagnostic_goal_ids=(), diagnostic_evidence_ids=()):
```

实现说明：

> Regrade saved components without measuring, simulating or mutating them.
>
> The caller supplies the same formal binding/curve plus the newly declared
> diagnostic identities. Every goal retains its original T/F/U and metrics.
> Each additional selector names an effect condition, process identity or
> terminal predicate in its evidence collection. All unselected components
> remain required. Complete acquisition and full-curve execution are also
> required; this result never becomes an online observation or dependency fact.

显式输出分支：

```python
{'physics_success': passed, 'physics_outcome': outcome, 'functional_goal_evidence': goals, **other, 'evaluation_policy': {'diagnostic_goal_ids': deepcopy(selected), 'diagnostic_evidence_ids': selectors, 'required_goal_ids': [row['identity'] for row in goals if row['evaluation_role'] == 'required'], 'criterion_source': 'Explicit goal and evidence identities declared by the evaluator caller. Unselected original components retain required roles.', 'scope': 'Explicit evaluator grading roles only; no change to planning, execution or runtime facts'}, 'success_definition': 'The complete current curve is physically observed and every required evidence component is T. Explicit diagnostic components retain their original metrics and T/F/U. With no required component the per-stage grade is unverified; subsequent task effectiveness is assessed from completed sequence evidence.'}
```

读取的对象属性：

```python
binding.get('goal_identities', {}).get
binding.get
row.get
binding['problem'].get
other.items
remaining.remove
other.values
```

写入目标：

```python
selected
goals
expected
collection
index
identities
row
row['evaluation_role']
other
name
effect_ids
expected_effects
process_ids
expected_processes
kind
terminal_ids
expected_terminal
identity_fields
selectors
remaining
rows
identity
selected_role
components
required
values
passed
outcome
```

直接调用符号：

```python
validate_diagnostic_goal_ids
ValueError
deepcopy
binding.get('goal_identities', {}).get
binding.get
tuple
len
set
row.get
binding['problem'].get
range
list
other.items
remaining.remove
other.values
any
```

## `_body_pose`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L227)。

输入签名：

```python
def _body_pose(data, index):
```

显式输出分支：

```python
pose
```

读取的对象属性：

```python
np.eye
data.xmat[index].reshape
data.xmat
data.xpos
```

写入目标：

```python
pose
pose[:3, :3]
pose[:3, 3]
```

直接调用符号：

```python
np.eye
data.xmat[index].reshape
```

## `_site_pose`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L233)。

输入签名：

```python
def _site_pose(data, index):
```

显式输出分支：

```python
pose
```

读取的对象属性：

```python
np.eye
data.site_xmat[index].reshape
data.site_xmat
data.site_xpos
```

写入目标：

```python
pose
pose[:3, :3]
pose[:3, 3]
```

直接调用符号：

```python
np.eye
data.site_xmat[index].reshape
```

## `_FunctionalGoalEvaluator`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L240)。

```python
class _FunctionalGoalEvaluator():
```

类字段：

无显式条目。

实现说明：

> Bind observed owner frames to native bodies without changing the candidate.
>
> A material frame needs a declared static body-from-owner transform and
> its source. A body identity alone supports contact/effect attribution but
> supplies no material geometry. Initial observed pose errors are never
> fitted away with native truth. No returned row is runtime input.

## `_FunctionalGoalEvaluator.__init__`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L249)。

输入签名：

```python
def __init__(self, model, catalog, planning_result, owner_bindings):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
planning_result.trajectory
planning_result.execution_binding
planning_result.success
self._trajectory
self._binding
problem.get
self._binding.get('goal_identities', {}).get
self._binding.get
self._requirements
self._support_bindings
self._support_binding
self._unsupported_goals.append
self._unsupported_goals
self._binding['scene'].get
self._requirements.values
requirement.actual_relation
owner_ids.update
self._support_bindings.values
row[key].strip
self._owners
model.body(name).id
model.body
row['source'].strip
```

写入目标：

```python
self._model
self._trajectory
self._binding
problem
self._requirements
self._unsupported_goals
self._support_bindings
self._support_numerical
collection
rows
identities
index
row
identity
self._requirements[identity]
supplied
supplied['surface']
requirement
self._support_bindings[identity]
owner_ids
relation
key
self._owners
name
bodies
body
transform
source
self._owners[identity]
self._owner_pose_samples
```

直接调用符号：

```python
isinstance
ValueError
problem.get
NumericalTolerance
deepcopy
self._binding.get('goal_identities', {}).get
self._binding.get
len
set
any
type
zip
PoseEventRequirement
SupportSurface
SupportRequirement
self._support_binding
self._unsupported_goals.append
list
self._binding['scene'].get
self._requirements.values
owner_ids.update
self._support_bindings.values
row[key].strip
model.body
row['source'].strip
_pose_array
int
```

## `_FunctionalGoalEvaluator._support_binding`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L334)。

输入签名：

```python
def _support_binding(requirement, problem):
```

实现说明：

> Resolve one material carrier from the held phases touching this goal.
>
> Both neighbours can include a shared boundary. Repeated references to
> the same declared attachment are one fact; two different facts, an
> incomplete binding or a conflicting transform remain unknown. This
> identifies a material frame for offline measurement, not actual Holding.

显式输出分支：

```python
{'reason': f'Original support carrier binding is incomplete or inconsistent: {error}'}
{'reason': 'Original support has no unique held carrier/attachment binding'}
next(iter(matched.values()))
```

读取的对象属性：

```python
problem.get
phase.get
metadata.get
requirement.progress
np.asarray
span.shape
np.isfinite(span).all
np.isfinite
row.get
conditions[0].get
np.allclose
requirement.tool_from_carrier
matched.values
```

写入目标：

```python
matched
index
phase
metadata
carrier
condition_id
value
span
attachment
conditions
row
declared
matched[carrier, condition_id]
```

直接调用符号：

```python
enumerate
problem.get
phase.get
metadata.get
any
isinstance
ValueError
np.asarray
np.isfinite(span).all
np.isfinite
_pose_array
row.get
len
conditions[0].get
np.allclose
_inverse
next
iter
matched.values
```

## `_FunctionalGoalEvaluator.observe`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L378)。

输入签名：

```python
def observe(self, data):
```

实现说明：

> Read and validate native transforms; do not publish observations.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._owners.items
self._owners
self._owner_pose_samples.append
self._owner_pose_samples
```

写入目标：

```python
poses
identity
binding
body
poses[identity]
```

直接调用符号：

```python
self._owners.items
_pose_array
_body_pose
self._owner_pose_samples.append
```

## `_FunctionalGoalEvaluator.describe_bindings`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L390)。

输入签名：

```python
def describe_bindings(self):
```

显式输出分支：

```python
[{key: value.tolist() if isinstance(value, np.ndarray) else value for (key, value) in row.items()} for row in self._owners.values()]
```

读取的对象属性：

```python
np.ndarray
value.tolist
row.items
self._owners.values
self._owners
```

写入目标：

```python
key
value
row
```

直接调用符号：

```python
isinstance
value.tolist
row.items
self._owners.values
```

## `_FunctionalGoalEvaluator._measure`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L394)。

输入签名：

```python
def _measure(self, identity, index, times, tools):
```

显式输出分支：

```python
self._measure_support(identity, index)
measured
```

读取的对象属性：

```python
self._measure_support
self._binding
self._owner_pose_samples
self._requirements
```

写入目标：

```python
feedback
observation
measured
measured['source']
```

直接调用符号：

```python
self._measure_support
float
_actual_pose_goal
```

## `_FunctionalGoalEvaluator._measure_support`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L415)。

输入签名：

```python
def _measure_support(self, identity, index):
```

实现说明：

> Evaluate the observed material body against the frozen world patch.
>
> ``check_region_on`` supplies gap/normal diagnostics and ``check_on``
> supplies the whole-body footprint. Their convenience ``valid``/absolute
> minimum-height criteria are not a replacement for SupportRequirement:
> only its specified face supplies the contact gap, and its original
> numerical position budget applies to body height and footprint alone.
> All four measurements remain available; the serialized hard_components
> alone determine this requirement's T/F result, including an empty set.

显式输出分支：

```python
measured
measured
measured
```

读取的对象属性：

```python
self._requirements
self._support_bindings
self._owner_pose_samples
requirement.contact_points
requirement.outward_normal
requirement.surface
requirement.gap_tolerance
requirement.normal_tolerance
requirement.margin
requirement.body_points
pose[:3, :3].T
np.min
requirement.surface.origin
requirement.surface.normal
np.linalg.norm
np.linalg
np.sin
self._support_numerical.position_m
self._support_numerical
requirement.hard_components
measured.update
pose.tolist
```

写入目标：

```python
requirement
binding
measured
measured['reason']
carrier
poses
pose
region
body
points
height
chord
angular_bound
numerical
violations
component_measurements
passed
name
```

直接调用符号：

```python
deepcopy
check_region_on
check_on
float
np.min
np.linalg.norm
np.sin
max
all
measured.update
list
pose.tolist
```

## `_FunctionalGoalEvaluator.evaluate`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L473)。

输入签名：

```python
def evaluate(self, times, elapsed, tools, execution_report, *, full_curve_completed):
```

实现说明：

> Grade original goal times using trajectory samples, excluding holds.

显式输出分支：

```python
evidence
```

读取的对象属性：

```python
self._unsupported_goals
self._binding.get
self._binding
execution_report.get
self._requirements.items
self._requirements
self._trajectory
requirement.progress
self._model.opt.timestep
self._model.opt
self._model
np.isfinite
requirement.source
row.update
requirement.gap_tolerance
requirement.normal_tolerance
requirement.margin
self._support_numerical.position_m
self._support_numerical
requirement.position_tolerance_m
requirement.rotation_tolerance_rad
requirement.yaw_axis
requirement.yaw_axis.tolist
requirement.yaw_tolerance_rad
event.get
evidence.append
np.ones
np.flatnonzero
indexes.size
self._measure
```

写入目标：

```python
evidence
origin
request_times
history
timing_events
identity
requirement
scheduled
due
row
arrivals
event
item
windows
row['reason']
lower
value
upper
row['arrival_window_s']
mask
indexes
closed
unknown
index
measured
row['query_count']
row['value']
```

直接调用符号：

```python
deepcopy
float
self._binding.get
execution_report.get
self._requirements.items
pose_goal_schedule
list
np.isfinite
row.update
requirement.yaw_axis.tolist
event.get
tuple
resolve_event_window
EventTimeConstraint
evidence.append
max
min
np.ones
len
np.flatnonzero
bool
self._measure
int
```

## `_FunctionalGoalEvaluator.final_support_diagnostics`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L549)。

输入签名：

```python
def final_support_diagnostics(self, times, elapsed, *, full_curve_completed):
```

实现说明：

> Measure the last retained body sample without inventing a hold window.

显式输出分支：

```python
[{'identity': list(identity), 'evaluation_role': 'diagnostic', 'completed_current_curve': bool(full_curve_completed), 'sample_index': len(times) - 1, 'sample_sim_time_s': float(times[-1]), 'sample_elapsed_s': float(elapsed[-1]), 'measurement': self._measure_support(identity, len(times) - 1), 'scope': 'Last normal-curve sample only; no declared terminal support or stability interval, and no effect on required original goals'} for identity in self._support_bindings]
```

读取的对象属性：

```python
self._measure_support
self._support_bindings
```

写入目标：

```python
identity
```

直接调用符号：

```python
list
bool
len
float
self._measure_support
```

## `_FunctionalGoalEvaluator.support_runtime_coverage`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L561)。

输入签名：

```python
def support_runtime_coverage(self, execution_report):
```

实现说明：

> Expose missing typed support observations without adding a dependency.

显式输出分支：

```python
coverage
```

读取的对象属性：

```python
self._support_bindings
execution.get
row.get
coverage.append
```

写入目标：

```python
execution
coverage
identity
rows
row
completed
value
```

直接调用符号：

```python
execution.get
tuple
row.get
list
any
coverage.append
len
```

## `_FunctionalGoalEvaluator.arrays`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L581)。

输入签名：

```python
def arrays(self):
```

显式输出分支：

```python
{'evaluation_owner_ids': np.asarray(identities, dtype=str), 'evaluation_owner_poses_world': poses.reshape(len(self._owner_pose_samples), len(identities), 4, 4)}
```

读取的对象属性：

```python
self._owners.items
self._owners
np.asarray
self._owner_pose_samples
poses.reshape
```

写入目标：

```python
identities
key
binding
poses
row
```

直接调用符号：

```python
self._owners.items
np.asarray
poses.reshape
len
```

## `_analyze_history`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L591)。

输入签名：

```python
def _analyze_history(times, touches, tool_poses, target_poses, fingers, finger_limits, settings):
```

实现说明：

> Apply the source segmentation definition to this trial's measurements.
>
> Two-sided contact must refer to this same rigid target body. Internal
> contact gaps may be bridged, but neither endpoint is extended. Success at
> the stop requires the final completed stability window, not an earlier
> grasp that was subsequently lost.

显式输出分支：

```python
({'stable_grasp_observed': first_stable is not None, 'first_stable_sim_time': None if first_stable is None else float(times[first_stable]), 'first_stable_evidence': first_evidence, 'stable_grasp_at_stop': final_stable is not None, 'stable_evidence_at_stop': final_evidence, 'failure_reasons': reasons, 'current_bilateral_target_contact': bool(simultaneous[-1]), 'bilateral_contact_present_at_entry': bool(simultaneous[0]), 'longest_gap_bridged_bilateral_contact_s': longest, 'final_window_metrics': final_metrics, 'maximum_target_translation_from_entry_m': float(np.linalg.norm(target_poses[:, :3, 3] - target_poses[0, :3, 3], axis=1).max()), 'maximum_target_rotation_from_entry_rad': float((target_rotation[0].inv() * target_rotation).magnitude().max()), 'final_target_pose_world': target_poses[-1].tolist(), 'final_tool_pose_world': tool_poses[-1].tolist()}, relative, simultaneous, bridged)
```

读取的对象属性：

```python
np.linalg.inv
np.linalg
Rotation.from_matrix
np.all
np.searchsorted
np.linalg.norm
np.ptp
np.max
(rotations[lower].inv() * rotations[lower:]).magnitude
rotations[lower].inv
np.ptp(fingers[lower:], axis=0).tolist
np.mean
reasons.append
bridged[lower:].all
np.any
np.asarray
np.linalg.norm(target_poses[:, :3, 3] - target_poses[0, :3, 3], axis=1).max
(target_rotation[0].inv() * target_rotation).magnitude().max
(target_rotation[0].inv() * target_rotation).magnitude
target_rotation[0].inv
target_poses[-1].tolist
tool_poses[-1].tolist
```

写入目标：

```python
relative
positions
rotations
simultaneous
bridged
runs
first_stable
first_evidence
final_stable
final_evidence
longest
begin
end
duration
found
evidence
final_begin
lower
final_metrics
reasons
target_rotation
```

直接调用符号：

```python
np.linalg.inv
Rotation.from_matrix
np.all
_bridge_gaps
_runs
float
max
_first_stable_grasp
len
int
np.searchsorted
np.linalg.norm
np.ptp
np.max
(rotations[lower].inv() * rotations[lower:]).magnitude
rotations[lower].inv
np.ptp(fingers[lower:], axis=0).tolist
np.mean
reasons.append
bridged[lower:].all
np.any
np.asarray
bool
np.linalg.norm(target_poses[:, :3, 3] - target_poses[0, :3, 3], axis=1).max
(target_rotation[0].inv() * target_rotation).magnitude().max
(target_rotation[0].inv() * target_rotation).magnitude
target_rotation[0].inv
target_poses[-1].tolist
tool_poses[-1].tolist
```

## `NativeSkillPhysicsEvaluator`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L665)。

```python
class NativeSkillPhysicsEvaluator():
```

类字段：

无显式条目。

实现说明：

> Collect native skill evidence and grade only supported predicates.

## `NativeSkillPhysicsEvaluator.__init__`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L668)。

输入签名：

```python
def __init__(self, model, *, catalog, actor, target_body_name, finger_joint_names, tcp_site, binding_source, program, reference_body_name, planning_result, evaluation_owner_bindings, maximum_contact_rows=1000000, diagnostic_goal_ids=(), diagnostic_evidence_ids=()):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._skill_kind
program.get
program['demo'].get
model.site(tcp_site).id
model.site
self._site
model.body(target_body_name).id
model.body
model.body(reference_body_name).id
self._target
self._target_joints
model.joint
native.qposadr
native.dofadr
np.asarray
self._target_geoms
model.geom(row['name']).id
model.geom
model.geom_bodyid
model.joint(name).qposadr
model.jnt_type
model.joint(name).id
mujoco.mjtJoint.mjJNT_SLIDE
mujoco.mjtJoint
self.settings
model.geom(index).name
self._groups
self._reference
planning_result.execution_binding
```

写入目标：

```python
self._model
self._skill_kind
self._task_contract
resolved
_
joint
self._maximum_contact_rows
self._site
target
row
self._target
self._reference
reference
self._target_joints
native
self._target_qpos
self._target_qvel
self._target_geoms
relevant
self._groups
group
self._finger_indices
name
self.settings
key
self._finger_limits
self.binding
index
self._planning_result
self._goals
self._diagnostic_goal_ids
self._diagnostic_evidence_ids
self._samples
self._contacts
self._offsets
```

直接调用符号：

```python
ValueError
deepcopy
program.get
program['demo'].get
_resolve_actor
set
type
model.site
len
model.body
int
model.joint
np.asarray
model.geom
sorted
list
_FunctionalGoalEvaluator
validate_diagnostic_goal_ids
```

## `NativeSkillPhysicsEvaluator.observe`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L748)。

输入签名：

```python
def observe(self, model, data, record):
```

实现说明：

> Read one native state after forward; never write state or controls.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._model
data.time
self._samples
np.isclose
model.opt.timestep
model.opt
self._offsets
data.ncon
self._maximum_contact_rows
np.zeros
self._groups
np.empty
data.contact
contact.geom1
contact.geom2
contact.exclude
contact.efc_address
contact.dist
self.settings
self._target_geoms
mujoco.mj_contactForce
np.r_
contact.pos
contact.frame
self._site
self._target
data.qpos[self._finger_indices].copy
data.qpos
self._finger_indices
np.isfinite(tool).all
np.isfinite
np.isfinite(target).all
np.isfinite(fingers).all
self._reference
data.qpos[self._target_qpos].copy
self._target_qpos
data.qvel[self._target_qvel].copy
data.qvel
self._target_qvel
np.isfinite(sample[7]).all
np.isfinite(sample[8]).all
np.isfinite(reference).all
self._goals.observe
self._goals
self._samples.append
self._contacts.append
self._contacts
self._offsets.append
```

写入目标：

```python
now
touch
contacts
index
contact
a
b
valid
group_index
group
touch[group_index]
force
force_valid
contacts[index]
tool
target
fingers
reference
sample
```

直接调用符号：

```python
ValueError
float
np.isclose
np.zeros
len
np.empty
range
int
enumerate
mujoco.mj_contactForce
_site_pose
_body_pose
data.qpos[self._finger_indices].copy
np.isfinite(tool).all
np.isfinite
np.isfinite(target).all
np.isfinite(fingers).all
data.qpos[self._target_qpos].copy
data.qvel[self._target_qvel].copy
np.isfinite(sample[7]).all
np.isfinite(sample[8]).all
np.isfinite(reference).all
self._goals.observe
self._samples.append
self._contacts.append
self._offsets.append
```

## `NativeSkillPhysicsEvaluator.finalize`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L792)。

输入签名：

```python
def finalize(self, output, *, execution_report, collection_report):
```

实现说明：

> Save the evidence and grade only a completed physical prefix.

显式输出分支：

```python
report
report
```

读取的对象属性：

```python
output.mkdir
np.asarray
self._samples
self._model.opt.timestep
self._model.opt
self._model
reasons.append
np.isfinite(times).all
np.isfinite
np.isfinite(elapsed).all
np.isclose
np.allclose
np.diff
np.arange
np.rint
rounded[np.isclose(observed_ticks, rounded, atol=1e-05, rtol=0)].astype
np.int64
np.empty
np.setdiff1d
np.split
np.flatnonzero
self._skill_kind
self._planning_result.execution_binding
self._planning_result
self._diagnostic_goal_ids
self._diagnostic_evidence_ids
(output / 'evaluation-report.json').write_text
json.dumps
self._finger_limits
self.settings
np.any
self._planning_result.trajectory
trajectory.progress_s
trajectory.duration
self._goals.evaluate
self._goals
self._goals.final_support_diagnostics
self._goals.support_runtime_coverage
binding['problem'].get
condition.get
self._goals._owners.get
self._goals._owners
self._target
effect_evidence.append
indexes.size
row.update
binding.get
event.get
process_evidence.append
requirement.get
terminal_evidence.append
self._goals.describe_bindings
self._task_contract
self.binding
self._finger_limits.tolist
self._offsets
touches[-1].tolist
target_q[0].tolist
target_q[-1].tolist
(target_q[-1] - target_q[0]).tolist
report.update
np.concatenate
self._contacts
self._reference
np.savez_compressed
self._target_joints
contacts[:, 0].astype
np.int32
contacts[:, 1].astype
contacts[:, 2].astype
contacts[:, 3].astype
contacts[:, 8:17].reshape
contacts[:, 23].astype
self._goals.arrays
```

写入目标：

```python
output
capture_state
expected
times
row
elapsed
timestep
reasons
observed_ticks
rounded
absent
runs
missing_ranges
coverage
a
b
report
tools
targets
fingers
touches
analysis
relative
simultaneous
bridged
completed
any_touch
no_contact_runs
final_absence
absence_duration
release_confirmed
trajectory
full_curve_completed
goal_evidence
final_support
support_runtime
effect_evidence
binding
condition
predicate
owner
row['reason']
_
due
row['device_due_time_s']
indexes
index
timed
event
item
prefix
prior_touch
duration
process_evidence
collection
requirement
terminal_evidence
grade
target_q
target_dq
report['grasp_stability_diagnostic']
contacts
extra
joint
```

直接调用符号：

```python
Path
output.mkdir
deepcopy
np.asarray
float
reasons.append
len
np.isfinite(times).all
np.isfinite
np.isfinite(elapsed).all
np.isclose
np.allclose
np.diff
np.arange
np.rint
rounded[np.isclose(observed_ticks, rounded, atol=1e-05, rtol=0)].astype
np.empty
np.setdiff1d
np.split
np.flatnonzero
int
(output / 'evaluation-report.json').write_text
json.dumps
_analyze_history
np.any
_runs
next
bool
pose_goal_schedule
self._goals.evaluate
self._goals.final_support_diagnostics
self._goals.support_runtime_coverage
binding['problem'].get
condition.get
self._goals._owners.get
effect_evidence.append
row.update
binding.get
event.get
enumerate
process_evidence.append
requirement.get
terminal_evidence.append
summarize_skill_evidence
self._goals.describe_bindings
self._finger_limits.tolist
touches[-1].tolist
target_q[0].tolist
target_q[-1].tolist
(target_q[-1] - target_q[0]).tolist
report.update
np.concatenate
np.savez_compressed
contacts[:, 0].astype
contacts[:, 1].astype
contacts[:, 2].astype
contacts[:, 3].astype
contacts[:, 8:17].reshape
contacts[:, 23].astype
self._goals.arrays
```
