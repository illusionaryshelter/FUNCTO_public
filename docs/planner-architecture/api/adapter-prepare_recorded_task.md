# `tests/planning/prepare_recorded_task.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `json_value`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L29)。

输入签名：

```python
def json_value(value):
```

输出分支的源码表达式：

```python
value.tolist()
value.item()
```

直接调用：

```python
isinstance
value.tolist
value.item
TypeError
type
```

## `read_json`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L38)。

输入签名：

```python
def read_json(path):
```

输出分支的源码表达式：

```python
json.loads(path.read_text())
```

直接调用：

```python
Path
path.stat
ValueError
json.loads
path.read_text
```

## `prepare_candidate`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L47)。

输入签名：

```python
def prepare_candidate(program, scene, feedback, skill_id):
```

直接读取的对象属性：`problem.joint_names`、`problem.start_q`、`problem.start_dq`。

输出分支的源码表达式：

```python
(problem, provenance)
```

直接写入的状态或数组位置：`prepared['carrier_mapping']`、`prepared['carrier_mapping_source']`。

直接调用：

```python
deepcopy
_pose_array
np.allclose
ValueError
prepared.get
all
bool
binding.get
bindings.values
np.array_equal
bindings.items
maps[0].copy
changes.append
maps[0].tolist
build_planning_problem
float
np.abs(scene_eef - measured_eef).max
np.abs
list
```

## `_pose_error`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L106)。

输入签名：

```python
def _pose_error(actual, expected):
```

输出分支的源码表达式：

```python
{'translation_m': float(np.linalg.norm(relative[:3, 3])), 'rotation_rad': float(Rotation.from_matrix(relative[:3, :3]).magnitude()), 'maximum_abs_matrix_error': float(np.abs(actual - expected).max())}
```

直接调用：

```python
np.linalg.inv
float
np.linalg.norm
Rotation.from_matrix(relative[:3, :3]).magnitude
Rotation.from_matrix
np.abs(actual - expected).max
np.abs
```

## `_declared_pose`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L114)。

输入签名：

```python
def _declared_pose(value, name):
```

输出分支的源码表达式：

```python
_pose_array(transform, name)
```

直接写入的状态或数组位置：`transform[:3, :3]`、`transform[:3, 3]`。

直接调用：

```python
np.asarray
np.isfinite(position).all
np.isfinite
np.isfinite(quaternion).all
np.isclose
np.linalg.norm
ValueError
np.eye
Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
_pose_array
```

## `_mujoco_pose`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L128)。

输入签名：

```python
def _mujoco_pose(position, rotation):
```

输出分支的源码表达式：

```python
transform
```

直接写入的状态或数组位置：`transform[:3, :3]`、`transform[:3, 3]`。

直接调用：

```python
np.eye
np.asarray(rotation).reshape
np.asarray
```

## `_finger_sphere_links`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L135)。

输入签名：

```python
def _finger_sphere_links(model, fingers, collision_spheres):
```

直接读取的对象属性：`result.values`。

输出分支的源码表达式：

```python
result
```

直接调用：

```python
model.joint
collision_spheres.items
model.body
roots.items
int
result[joint_name].append
result.values
any
len
set
ValueError
```

## `_extract_named_feedback`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L161)。

输入签名：

```python
def _extract_named_feedback(manifest, fixture, fixture_path):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`data.qpos[q_addresses]`、`data.qvel[dq_addresses]`、`data.time`、`checks[values_key + '_maximum_abs_error']`、`result[segment['id']]`。

直接调用：

```python
Path
mujoco.MjModel.from_xml_path
str
mujoco.MjData
URDF.load
yaml.safe_load
(root / fixture['robot_config_path']).read_text
len
set
model.joint
range
set(arms).intersection
ValueError
set(fixture['planning_joint_names']).issubset
_finger_sphere_links
any
type
int
_declared_pose
model.body
model.site
h5py.File
max
read_json
np.array
float
np.isfinite(q).all
np.isfinite
np.isfinite(dq).all
_pose_array
mujoco.mj_kinematics
_mujoco_pose
urdf.update_cfg
dict
zip
urdf.get_transform
_pose_error
errors.values
np.asarray
set(old_names).issubset
names.index
np.abs(expected - old_q).max
np.abs
frozen.get
abs
np.maximum
uuid4
q.tolist
dq.tolist
finger_links.values
deepcopy
feedback_pose.tolist
base.tolist
tool.tolist
limits.tolist
episode_path.resolve
frozen_path.resolve
fixture_path.resolve
(root / fixture['binding_audit_path']).resolve
```

## `_select_planning_joints`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L317)。

输入签名：

```python
def _select_planning_joints(feedback):
```

输出分支的源码表达式：

```python
selected
```

直接调用：

```python
deepcopy
len
set
any
ValueError
feedback['joint_names'].index
selected.update
```

## `_problem_summary`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L331)。

输入签名：

```python
def _problem_summary(problem):
```

直接读取的对象属性：`problem.joint_names`、`problem.phase_count`、`problem.anchors`、`problem.stop_progress`、`problem.phases`、`problem.source_time_map`、`problem.events`、`problem.conditions`、`problem.requirement_consumption`、`problem.supports`、`problem.interiors`、`problem.pose_events`、`problem.coordinate_events`。

输出分支的源码表达式：

```python
{'joint_names': list(problem.joint_names), 'phase_count': problem.phase_count, 'anchor_count': len(problem.anchors), 'stop_progress': problem.stop_progress.tolist(), 'phases': [{'name': phase.name, 'reference_node_count': len(phase.reference.progress), 'metadata': phase.metadata} for phase in problem.phases], 'source_time_map': problem.source_time_map, 'events': [{'event_id': event.event_id, 'skill_id': event.skill_id, 'kind': event.kind, 'source_frame': event.source_frame, 'progress': event.progress, 'requires_evidence': event.requires_evidence, 'stop': event.stop} for event in problem.events], 'conditions': list(problem.conditions), 'requirement_consumption': problem.requirement_consumption, 'requirement_counts': {'supports': len(problem.supports), 'interiors': len(problem.interiors), 'pose_events': len(problem.pose_events), 'coordinate_events': len(problem.coordinate_events)}}
```

直接调用：

```python
list
len
problem.stop_progress.tolist
```

## `_write_json`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L358)。

输入签名：

```python
def _write_json(path, data):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
path.parent.mkdir
path.write_text
json.dumps
```

## `_prepare_segments`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L365)。

输入签名：

```python
def _prepare_segments(manifest, feedback_by_segment, output):
```

输出分支的源码表达式：

```python
rows
```

直接写入的状态或数组位置：`item['error']`。

直接调用：

```python
Path
_write_json
_select_planning_joints
read_json
str
feedback_path.resolve
list
prepare_candidate
item.update
_problem_summary
type
row['candidates'].append
rows.append
```

## `run`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L400)。

输入签名：

```python
def run(args):
```

输出分支的源码表达式：

```python
report
```

直接调用：

```python
os.environ.get
ValueError
time.perf_counter
read_json
len
set
_extract_named_feedback
_prepare_segments
sum
str
args.manifest.resolve
args.robot_fixture.resolve
resource.getrusage
_write_json
```

## `main`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L431)。

输入签名：

```python
def main():
```

输出分支的源码表达式：

```python
0 if report['status'] == 'passed' else 1
```

直接调用：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
run
print
json.dumps
str
args.output.resolve
```
