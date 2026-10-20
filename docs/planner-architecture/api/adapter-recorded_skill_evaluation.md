# `tests/planning/recorded_skill_evaluation.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_actual_pose_goal`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L46)。

输入签名：

```python
def _actual_pose_goal(requirement, feedback, observation):
```

输出分支的源码表达式：

```python
measured
```

直接写入的状态或数组位置：`measured['actual_function_pose']`、`measured['actual_target_pose']`、`measured['reason']`。

直接调用：

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
np.linalg.norm
Rotation.from_matrix(target[:3, :3].T @ function[:3, :3]).magnitude
Rotation.from_matrix
measured.update
```

## `_actual_pose_goal.owner_pose`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L64)。

输入签名：

```python
def _actual_pose_goal.owner_pose(identity):
```

输出分支的源码表达式：

```python
None
pose
```

直接调用：

```python
missing.append
_pose_array
measured['measurement_sources'].append
float
pose.tolist
```

## `_actual_pose_goal.actual_tool`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L73)。

输入签名：

```python
def _actual_pose_goal.actual_tool():
```

输出分支的源码表达式：

```python
pose
```

直接调用：

```python
_pose_array
measured['measurement_sources'].append
float
pose.tolist
```

## `validate_diagnostic_goal_ids`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L115)。

输入签名：

```python
def validate_diagnostic_goal_ids(binding, values):
```

输出分支的源码表达式：

```python
validate_result_only_goal_ids(binding, values)
```

直接调用：

```python
validate_result_only_goal_ids
```

## `summarize_skill_evidence`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L128)。

输入签名：

```python
def summarize_skill_evidence(report, *, binding, trajectory, diagnostic_goal_ids=()):
```

输出分支的源码表达式：

```python
{'physics_success': passed, 'physics_outcome': outcome, 'functional_goal_evidence': goals, **other, 'evaluation_policy': {'diagnostic_goal_ids': deepcopy(selected), 'required_goal_ids': [row['identity'] for row in goals if row['evaluation_role'] == 'required'], 'criterion_source': 'Explicit diagnostic_goal_ids declared by the evaluator caller. Unselected original goals, bound skill effects, scoped processes and declared terminal predicates retain their required roles.', 'scope': 'Explicit evaluator grading roles only; no change to planning, execution or runtime facts'}, 'success_definition': 'The complete current curve is physically observed and every required goal, original effect, scoped process and final-stop predicate is T. Explicit diagnostic goals retain their original metrics and T/F/U but do not veto success; unsupported or unobserved required predicates remain U.'}
```

直接写入的状态或数组位置：`row['evaluation_role']`。

直接调用：

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
other.values
any
```

## `_body_pose`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L204)。

输入签名：

```python
def _body_pose(data, index):
```

输出分支的源码表达式：

```python
pose
```

直接写入的状态或数组位置：`pose[:3, :3]`、`pose[:3, 3]`。

直接调用：

```python
np.eye
data.xmat[index].reshape
```

## `_site_pose`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L210)。

输入签名：

```python
def _site_pose(data, index):
```

输出分支的源码表达式：

```python
pose
```

直接写入的状态或数组位置：`pose[:3, :3]`、`pose[:3, 3]`。

直接调用：

```python
np.eye
data.site_xmat[index].reshape
```

## `_FunctionalGoalEvaluator`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L217)。

```python
class _FunctionalGoalEvaluator:
    ...
    ...
```

该类型的直接方法：`__init__`、`_support_binding`、`observe`、`describe_bindings`、`_measure`、`_measure_support`、`evaluate`、`final_support_diagnostics`、`support_runtime_coverage`、`arrays`。

## `_FunctionalGoalEvaluator.__init__`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L226)。

输入签名：

```python
def _FunctionalGoalEvaluator.__init__(self, model, catalog, planning_result, owner_bindings):
```

直接读取的对象属性：`self._trajectory`、`self._binding`、`problem.get`、`self._requirements`、`self._support_bindings`、`self._support_binding`、`self._unsupported_goals`、`self._owners`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._model`、`self._trajectory`、`self._binding`、`self._requirements`、`self._unsupported_goals`、`self._support_bindings`、`self._support_numerical`、`self._requirements[identity]`、`supplied['surface']`、`self._support_bindings[identity]`、`self._owners`、`self._owners[identity]`、`self._owner_pose_samples`。

直接调用：

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
row[key].strip
model.body
row['source'].strip
_pose_array
int
```

## `_FunctionalGoalEvaluator._support_binding`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L300)。

输入签名：

```python
@staticmethod
def _FunctionalGoalEvaluator._support_binding(requirement, problem):
```

直接读取的对象属性：`problem.get`。

输出分支的源码表达式：

```python
{'reason': f'Original support carrier binding is incomplete or inconsistent: {error}'}
{'reason': 'Original support has no unique held carrier/attachment binding'}
next(iter(matched.values()))
```

直接写入的状态或数组位置：`matched[carrier, condition_id]`。

直接调用：

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

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L344)。

输入签名：

```python
def _FunctionalGoalEvaluator.observe(self, data):
```

直接读取的对象属性：`self._owners`、`self._owner_pose_samples`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`poses[identity]`。

直接调用：

```python
self._owners.items
_pose_array
_body_pose
self._owner_pose_samples.append
```

## `_FunctionalGoalEvaluator.describe_bindings`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L356)。

输入签名：

```python
def _FunctionalGoalEvaluator.describe_bindings(self):
```

直接读取的对象属性：`self._owners`。

输出分支的源码表达式：

```python
[{key: value.tolist() if isinstance(value, np.ndarray) else value for (key, value) in row.items()} for row in self._owners.values()]
```

直接调用：

```python
isinstance
value.tolist
row.items
self._owners.values
```

## `_FunctionalGoalEvaluator._measure`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L360)。

输入签名：

```python
def _FunctionalGoalEvaluator._measure(self, identity, index, times, tools):
```

直接读取的对象属性：`self._measure_support`、`self._binding`、`self._owner_pose_samples`、`self._requirements`。

输出分支的源码表达式：

```python
self._measure_support(identity, index)
measured
```

直接写入的状态或数组位置：`measured['source']`。

直接调用：

```python
self._measure_support
float
_actual_pose_goal
```

## `_FunctionalGoalEvaluator._measure_support`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L381)。

输入签名：

```python
def _FunctionalGoalEvaluator._measure_support(self, identity, index):
```

直接读取的对象属性：`self._requirements`、`self._support_bindings`、`self._owner_pose_samples`、`self._support_numerical`。

输出分支的源码表达式：

```python
measured
```

直接写入的状态或数组位置：`measured['reason']`。

直接调用：

```python
deepcopy
check_region_on
check_on
float
np.min
np.linalg.norm
np.sin
max
measured.update
any
violations.values
pose.tolist
```

## `_FunctionalGoalEvaluator.evaluate`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L428)。

输入签名：

```python
def _FunctionalGoalEvaluator.evaluate(self, times, elapsed, tools, execution_report, *, full_curve_completed):
```

直接读取的对象属性：`self._unsupported_goals`、`self._binding`、`self._requirements`、`self._trajectory`、`self._model`、`self._support_numerical`、`self._measure`。

输出分支的源码表达式：

```python
evidence
```

直接写入的状态或数组位置：`row['reason']`、`row['arrival_window_s']`、`row['query_count']`、`row['value']`。

直接调用：

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

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L500)。

输入签名：

```python
def _FunctionalGoalEvaluator.final_support_diagnostics(self, times, elapsed, *, full_curve_completed):
```

直接读取的对象属性：`self._measure_support`、`self._support_bindings`。

输出分支的源码表达式：

```python
[{'identity': list(identity), 'evaluation_role': 'diagnostic', 'completed_current_curve': bool(full_curve_completed), 'sample_index': len(times) - 1, 'sample_sim_time_s': float(times[-1]), 'sample_elapsed_s': float(elapsed[-1]), 'measurement': self._measure_support(identity, len(times) - 1), 'scope': 'Last normal-curve sample only; no declared terminal support or stability interval, and no effect on required original goals'} for identity in self._support_bindings]
```

直接调用：

```python
list
bool
len
float
self._measure_support
```

## `_FunctionalGoalEvaluator.support_runtime_coverage`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L512)。

输入签名：

```python
def _FunctionalGoalEvaluator.support_runtime_coverage(self, execution_report):
```

直接读取的对象属性：`self._support_bindings`。

输出分支的源码表达式：

```python
coverage
```

直接调用：

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

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L532)。

输入签名：

```python
def _FunctionalGoalEvaluator.arrays(self):
```

直接读取的对象属性：`self._owners`、`self._owner_pose_samples`。

输出分支的源码表达式：

```python
{'evaluation_owner_ids': np.asarray(identities, dtype=str), 'evaluation_owner_poses_world': poses.reshape(len(self._owner_pose_samples), len(identities), 4, 4)}
```

直接调用：

```python
self._owners.items
np.asarray
poses.reshape
len
```

## `_analyze_history`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L542)。

输入签名：

```python
def _analyze_history(times, touches, tool_poses, target_poses, fingers, finger_limits, settings):
```

输出分支的源码表达式：

```python
({'stable_grasp_observed': first_stable is not None, 'first_stable_sim_time': None if first_stable is None else float(times[first_stable]), 'first_stable_evidence': first_evidence, 'stable_grasp_at_stop': final_stable is not None, 'stable_evidence_at_stop': final_evidence, 'failure_reasons': reasons, 'current_bilateral_target_contact': bool(simultaneous[-1]), 'bilateral_contact_present_at_entry': bool(simultaneous[0]), 'longest_gap_bridged_bilateral_contact_s': longest, 'final_window_metrics': final_metrics, 'maximum_target_translation_from_entry_m': float(np.linalg.norm(target_poses[:, :3, 3] - target_poses[0, :3, 3], axis=1).max()), 'maximum_target_rotation_from_entry_rad': float((target_rotation[0].inv() * target_rotation).magnitude().max()), 'final_target_pose_world': target_poses[-1].tolist(), 'final_tool_pose_world': tool_poses[-1].tolist()}, relative, simultaneous, bridged)
```

直接调用：

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

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L616)。

```python
class NativeSkillPhysicsEvaluator:
    ...
    ...
```

该类型的直接方法：`__init__`、`observe`、`finalize`。

## `NativeSkillPhysicsEvaluator.__init__`

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L619)。

输入签名：

```python
def NativeSkillPhysicsEvaluator.__init__(self, model, *, catalog, actor, target_body_name, finger_joint_names, tcp_site, binding_source, program, reference_body_name, planning_result, evaluation_owner_bindings, maximum_contact_rows=1000000, diagnostic_goal_ids=()):
```

直接读取的对象属性：`self._skill_kind`、`self._site`、`self._target`、`self._target_joints`、`self._target_geoms`、`self.settings`、`self._groups`、`self._reference`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._model`、`self._skill_kind`、`self._task_contract`、`self._maximum_contact_rows`、`self._site`、`self._target`、`self._reference`、`self._target_joints`、`self._target_qpos`、`self._target_qvel`、`self._target_geoms`、`self._groups`、`self._finger_indices`、`self.settings`、`self._finger_limits`、`self.binding`、`self._planning_result`、`self._goals`、`self._diagnostic_goal_ids`、`self._samples`、`self._contacts`、`self._offsets`。

直接调用：

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

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L698)。

输入签名：

```python
def NativeSkillPhysicsEvaluator.observe(self, model, data, record):
```

直接读取的对象属性：`self._model`、`self._samples`、`self._offsets`、`self._maximum_contact_rows`、`self._groups`、`self.settings`、`self._target_geoms`、`self._site`、`self._target`、`self._finger_indices`、`self._reference`、`self._target_qpos`、`self._target_qvel`、`self._goals`、`self._contacts`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`touch[group_index]`、`contacts[index]`。

直接调用：

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

[定义位置](../../../tests/planning/recorded_skill_evaluation.py#L742)。

输入签名：

```python
def NativeSkillPhysicsEvaluator.finalize(self, output, *, execution_report, collection_report):
```

直接读取的对象属性：`self._samples`、`self._model`、`self._skill_kind`、`self._planning_result`、`self._diagnostic_goal_ids`、`self._finger_limits`、`self.settings`、`trajectory.progress_s`、`trajectory.duration`、`self._goals`、`self._target`、`self._task_contract`、`self.binding`、`self._offsets`、`self._contacts`、`self._reference`、`self._target_joints`。

输出分支的源码表达式：

```python
report
```

直接写入的状态或数组位置：`row['reason']`、`row['device_due_time_s']`、`report['grasp_stability_diagnostic']`。

直接调用：

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
