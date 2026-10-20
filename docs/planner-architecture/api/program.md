# `manipulation/program.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `ReferenceGenerationError`

[定义位置](../../../manipulation/program.py#L22)。

```python
class ReferenceGenerationError(ValueError):
    ...
    ...
```

## `_mechanism_entry_contract`

[定义位置](../../../manipulation/program.py#L26)。

输入签名：

```python
def _mechanism_entry_contract(model_spec, initial, *, require_lift=False):
```

输出分支的源码表达式：

```python
(None, None)
(model, limits)
```

直接调用：

```python
_current_model
_current_initial_coordinates
np.any
ValueError
```

## `_unfitted_reference`

[定义位置](../../../manipulation/program.py#L44)。

输入签名：

```python
def _unfitted_reference(reference, model, error, coordinates=None):
```

输出分支的源码表达式：

```python
(output, {'model': serialized, 'reference_generation': {'status': 'failed', 'reason': str(error), 'seed': 'bound source reference; not a task feasibility result'}})
```

直接写入的状态或数组位置：`model.coordinates`。

直接调用：

```python
reference.copy
model._coordinates_unchecked
np.asarray(coordinates, float).copy
np.asarray
model.poses
model.to_dict
str
```

## `_index`

[定义位置](../../../manipulation/program.py#L60)。

输入签名：

```python
def _index(frame, frames, name):
```

输出分支的源码表达式：

```python
int(found[0])
```

直接调用：

```python
isinstance
ValueError
np.flatnonzero
len
int
```

## `_positive`

[定义位置](../../../manipulation/program.py#L70)。

输入签名：

```python
def _positive(value, name):
```

输出分支的源码表达式：

```python
value
```

直接调用：

```python
isinstance
ValueError
float
np.isfinite
```

## `_intentions`

[定义位置](../../../manipulation/program.py#L79)。

输入签名：

```python
def _intentions(events, frames):
```

输出分支的源码表达式：

```python
deepcopy(events)
```

直接调用：

```python
isinstance
ValueError
set
event.get
_index
seen.add
deepcopy
```

## `_current_model`

[定义位置](../../../manipulation/program.py#L101)。

输入签名：

```python
def _current_model(value):
```

输出分支的源码表达式：

```python
(model, limits)
```

直接写入的状态或数组位置：`basis_pose[:3, :3]`。

直接调用：

```python
value.get
ValueError
_pose_array
np.eye
np.asarray
np.isfinite(point).all
np.isfinite
_tolerances
MotionModel
np.zeros
np.isfinite(limits).all
np.any
```

## `_exact_coordinates`

[定义位置](../../../manipulation/program.py#L140)。

输入签名：

```python
def _exact_coordinates(model, pose):
```

输出分支的源码表达式：

```python
coordinates
```

直接调用：

```python
model.coordinates_of
_errors
model.poses
ValueError
```

## `_current_initial_coordinates`

[定义位置](../../../manipulation/program.py#L149)。

输入签名：

```python
def _current_initial_coordinates(model_spec, model, pose, *, require_lift=False):
```

输出分支的源码表达式：

```python
coordinates
coordinates.astype(float, copy=True)
```

直接写入的状态或数组位置：`coordinates[angular]`、`model.diagnostics['initial_coordinate_convention']`、`model.diagnostics['initial_coordinates_source']`。

直接调用：

```python
ValueError
_exact_coordinates
model_spec.get
np.asarray
int
np.ceil
np.floor
np.isfinite(coordinates).all
np.isfinite
_joint_source
_errors
model.poses
coordinates.astype
```

## `_event_transfer`

[定义位置](../../../manipulation/program.py#L191)。

输入签名：

```python
def _event_transfer(reference, start, goals, function_frame, model_spec=None, *, timestamps, coordinate_reference=None, coordinate_goals=None):
```

输出分支的源码表达式：

```python
(output, diagnostics)
```

直接写入的状态或数组位置：`output[begin:end + 1]`、`diagnostics['model']`、`diagnostics['reference_projection']`、`diagnostics['coordinate_transfer']`、`initial_q[angular]`、`goal_q[angular]`、`part[0]`、`part[-1]`、`values[begin:end + 1]`、`model.coordinates`、`diagnostics['limit_status']`。

直接调用：

```python
validate_timeline
np.arange
len
np.allclose
ReferenceGenerationError
np.empty_like
ValueError
_pose_array
enumerate
transfer_motion
interval_progress
_inverse
_current_model
np.asarray(coordinate_reference, float).copy
np.asarray
model._coordinates_unchecked
np.isfinite(coordinates).all
np.isfinite
_errors
model.poses
p.max
r.max
float
coordinates.copy
coordinates[0].copy
coordinate_goals[begin].copy
_exact_coordinates
np.round
coordinate_goals[end].copy
np.isfinite([source_delta, target_delta]).all
np.finfo
max
abs
np.array_equal
diagnostics['coordinate_transfer']['intervals'].append
np.any
model.to_dict
```

## `_event_transfer.interval_progress`

[定义位置](../../../manipulation/program.py#L220)。

输入签名：

```python
def _event_transfer.interval_progress(begin, end):
```

输出分支的源码表达式：

```python
np.zeros(1) if begin == end else (part - part[0]) / (part[-1] - part[0])
```

直接调用：

```python
np.zeros
```

## `_coordinate_vector`

[定义位置](../../../manipulation/program.py#L350)。

输入签名：

```python
def _coordinate_vector(value, name):
```

输出分支的源码表达式：

```python
vector.astype(float, copy=True)
```

直接调用：

```python
np.asarray
np.isfinite(vector).all
np.isfinite
ValueError
vector.astype
```

## `_joint_source`

[定义位置](../../../manipulation/program.py#L359)。

输入签名：

```python
def _joint_source(value, name):
```

输出分支的源码表达式：

```python
deepcopy(value)
```

直接调用：

```python
isinstance
value.strip
ValueError
deepcopy
```

## `_joint_reference`

[定义位置](../../../manipulation/program.py#L367)。

输入签名：

```python
def _joint_reference(demo, scene, frames, last_held, task, initial):
```

直接读取的对象属性：`scene.get`。

输出分支的源码表达式：

```python
(objects, checks, diagnostics)
```

直接写入的状态或数组位置：`unique_goals[index]`。

直接调用：

```python
task.get
ValueError
scene.get
isinstance
spec.get
_mechanism_entry_contract
_joint_source
demo.get
np.asarray
len
np.isfinite(raw).all
np.isfinite
raw.astype
_current_initial_coordinates
model.poses
set
event.get
_index
_positive
_coordinate_vector
np.max
np.abs
bindings.get
binding.get
unique_goals.setdefault
target.copy
recorded.append
int
recorded[-1].update
names.add
unique_goals.items
max
float
min
initial_coordinates.copy
_event_transfer
_unfitted_reference
_errors
checks.append
event.items
checks[-1].update
actual[index].copy
diagnostics.update
source_coordinates[0].copy
model.diagnostics.get
deepcopy
```

## `_check_entry_corridor`

[定义位置](../../../manipulation/program.py#L539)。

输入签名：

```python
def _check_entry_corridor(objects, scene, margin, motion_checks):
```

直接读取的对象属性：`scene.get`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`motion_checks['entry_corridor']`。

直接调用：

```python
scene.get
type
len
ValueError
_interior
min
check_inside
float
bool
```

## `_functional_events`

[定义位置](../../../manipulation/program.py#L559)。

输入签名：

```python
def _functional_events(demo, scene, frames, poses, last_held, task, initial):
```

直接读取的对象属性：`scene.get`。

输出分支的源码表达式：

```python
output
```

直接写入的状态或数组位置：`relation['reference_generation']`、`relation['goal_pose']`、`relation['reference_binding']`、`relation['function_frame']`、`relation['constraint_kind']`。

直接调用：

```python
demo.get
isinstance
ValueError
scene.get
_pose_array
set
event.get
_index
current.get
binding.get
np.allclose
_inverse
_surface
np.asarray
ground_on
face.get
str
ground_relation
output.append
int
_positive
names.add
```

## `_object_reference`

[定义位置](../../../manipulation/program.py#L670)。

输入签名：

```python
def _object_reference(demo, scene, frames, last_held, task, initial):
```

直接读取的对象属性：`scene.get`。

输出分支的源码表达式：

```python
result
(objects, checks, motion_checks)
```

直接写入的状态或数组位置：`error['body_checks']`、`error['freedom']`、`error['body_checks']['minimum_signed_height_m']`、`error['checked_member_satisfies_support']`、`error['containment']`、`checks[-1]['reference_generation']`。

直接调用：

```python
task.get
ValueError
scene.get
isinstance
np.isfinite
any
demo.get
event.get
_joint_reference
_check_entry_corridor
_poses_array
len
_functional_events
_inverse
enumerate
_errors
sum
np.linalg.norm
np.sin
min
unique_goals.setdefault
initial.copy
list
unique_goals.items
model_spec.get
_mechanism_entry_contract
reference[:last_held + 1].copy
transfer_function_motion
_unfitted_reference
_event_transfer
_surface
check_region_on
check_on
deepcopy
np.asarray
float
np.min
bool
relation_error
_pose_array
check_inside
_interior
checks.append
```

## `ground_object_program`

[定义位置](../../../manipulation/program.py#L832)。

输入签名：

```python
def ground_object_program(program, scene):
```

直接读取的对象属性：`scene.get`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result['body_points']`。

直接调用：

```python
program.get
ValueError
isinstance
scene.get
validate_timeline
demo.get
len
_index
_pose_array
_object_reference
```

## `ground_program`

[定义位置](../../../manipulation/program.py#L870)。

输入签名：

```python
def ground_program(program, scene):
```

直接读取的对象属性：`scene.get`、`result.update`。

输出分支的源码表达式：

```python
result
{'status': 'candidate', 'skill_kind': kind, 'output_type': 'interaction_trajectory', 'reference_frame': frame_name, 'carrier_id': scene['carrier_id'], 'reference_id': scene['reference_id'], 'task': task, 'object_poses': objects, 'object_frame_indices': frames[:last_held + 1], 'eef_poses': eef, 'frame_indices': frames[:last_held + 1], 'timestamps': times[:last_held + 1], 'source_frame_indices': frames, 'source_timestamps': times, 'source_gripper_events': commands, 'gripper_events': controlled_commands, 'release_frame': release, 'function_checks': checks, 'motion_checks': motion_checks, 'tool_id': scene['tool_id'], 'tool_frame': scene['tool_frame'], 'holding_source': holding['source'], 'released_object_state': 'requires_observation' if release is not None else 'still_attached_reference', 'pose_convention': 'column-vector reference_from_body/eef; metres; 4x4 SE(3)', 'path_scope': 'geometry reference; robot IK, collision, forces and timing remain external'}
```

直接调用：

```python
scene.get
isinstance
ValueError
ground_grasp
result.update
validate_timeline
_intentions
demo.get
len
_index
any
event.get
holding.get
_pose_array
np.allclose
task.get
_object_reference
attached_eef_reference
```
