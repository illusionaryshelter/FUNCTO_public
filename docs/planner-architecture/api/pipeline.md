# `manipulation/pipeline.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `motion_kind`

[定义位置](../../../manipulation/pipeline.py#L26)。

输入签名：

```python
def motion_kind(task):
```

输出分支的源码表达式：

```python
'free'
task['motion']
```

## `serializable`

[定义位置](../../../manipulation/pipeline.py#L44)。

输入签名：

```python
def serializable(value):
```

输出分支的源码表达式：

```python
value.tolist()
value.item()
{key: serializable(item) for (key, item) in value.items()}
[serializable(item) for item in value]
value
```

直接调用：

```python
isinstance
value.tolist
value.item
serializable
value.items
```

## `write_json`

[定义位置](../../../manipulation/pipeline.py#L57)。

输入签名：

```python
def write_json(path, value):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
Path(path).write_text
Path
json.dumps
serializable
```

## `_matrix`

[定义位置](../../../manipulation/pipeline.py#L62)。

输入签名：

```python
def _matrix(value, name):
```

输出分支的源码表达式：

```python
matrix
```

直接调用：

```python
np.asarray
np.isfinite(matrix).all
np.isfinite
ValueError
np.allclose
np.eye
np.isclose
np.linalg.det
```

## `_surface`

[定义位置](../../../manipulation/pipeline.py#L73)。

输入签名：

```python
def _surface(value):
```

输出分支的源码表达式：

```python
SupportSurface(np.asarray(value['origin']), np.asarray(value['basis']), np.asarray(value['normal']), np.asarray(value['boundary']), source=value.get('source', 'provided_support_geometry'))
```

直接调用：

```python
SupportSurface
np.asarray
value.get
```

## `_interior`

[定义位置](../../../manipulation/pipeline.py#L79)。

输入签名：

```python
def _interior(value):
```

输出分支的源码表达式：

```python
InteriorRegion(np.asarray(value['halfspaces']), value['source'])
InteriorRegion.from_vertices(value['vertices'], value['source'])
```

直接调用：

```python
ValueError
InteriorRegion
np.asarray
InteriorRegion.from_vertices
```

## `run_stage`

[定义位置](../../../manipulation/pipeline.py#L89)。

输入签名：

```python
def run_stage(observation):
```

直接读取的对象属性：`scene.get`。

输出分支的源码表达式：

```python
{'status': 'passed', 'output_type': 'object_trajectory', 'carrier_id': observation['carrier_id'], 'reference_id': observation['reference_id'], 'reference_frame': observation['reference_frame'], 'pose_convention': 'column-vector reference_from_fixed_body, metres, 4x4 SE(3)', 'task': task, 'poses': poses, 'frame_indices': frame_indices, 'body_points': body, 'model': model.to_dict(), 'goal_pose': goal, 'goal_geometry': None if geometry is None else geometry.to_dict(), 'goal_checks': goal_report, 'motion_checks': transferred['diagnostics'], 'interaction_geometry': observation.get('interaction_geometry'), 'interaction_geometry_status': 'provided' if observation.get('interaction_geometry') is not None else 'not_estimated_for_object_trajectory', 'observation_evidence': observation.get('evidence', {}), 'geometry_coverage': scene.get('geometry_coverage', 'supplied_points_only'), 'path_scope': 'demonstration transfer; no complete-scene collision or force guarantee'}
```

直接写入的状态或数组位置：`observed_structure['pose_frame']`、`preferred[:3, 3]`、`goal_report['containment_status']`、`transferred['diagnostics']['observed_structure']`、`transferred['diagnostics']['applied_constraint']`。

直接调用：

```python
ValueError
isinstance
observation.get
observation[key].strip
scene.get
np.isscalar
np.isfinite
_matrix
np.asarray
len
np.isfinite(body).all
analyze_motion
motion_kind
np.linspace
transfer_motion
np.asarray(preliminary['poses'][-1]).copy
_surface
check_on
place_on
_interior
check_inside
place_inside
np.allclose
type
min
goal_report.update
max
float
list
range
(frame_indices < 0).any
(frame_indices[1:] <= frame_indices[:-1]).any
model.to_dict
geometry.to_dict
```

## `_move_scene_geometry`

[定义位置](../../../manipulation/pipeline.py#L255)。

输入签名：

```python
def _move_scene_geometry(scene, delta):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`scene['reference_transform']`、`scene[key]`、`surface.origin`、`surface.basis`、`surface.normal`、`scene['support']`。

直接调用：

```python
_matrix
_surface
surface.to_dict
_interior
InteriorRegion(np.column_stack((normals, offsets)), region.source).to_dict
InteriorRegion
np.column_stack
```

## `run_sequence`

[定义位置](../../../manipulation/pipeline.py#L275)。

输入签名：

```python
def run_sequence(observations):
```

输出分支的源码表达式：

```python
{'status': 'passed', 'reference_frame': frame, 'stages': output, 'final_entity_poses': states, 'unobserved_entity_states': invalidated}
```

直接写入的状态或数组位置：`body_frames[carrier]`、`scene['initial_pose']`、`invalidated[dependent]`、`states[carrier]`、`placements[carrier]`。

直接调用：

```python
isinstance
ValueError
deepcopy
observation.get
states[carrier].copy
_matrix
np.linalg.inv
_move_scene_geometry
run_stage
placements.pop
np.allclose
changed.pop
list
placements.items
states.pop
changed.append
np.asarray(result['poses'][-1]).copy
np.asarray
output.append
```

## `run_prepared`

[定义位置](../../../manipulation/pipeline.py#L333)。

输入签名：

```python
def run_prepared(input_path, output_dir):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`demo['trajectory_file']`、`demo[name]`、`report['status']`。

直接调用：

```python
Path
json.loads
path.read_text
destination.exists
destination.is_dir
any
destination.iterdir
FileExistsError
destination.mkdir
str
path.resolve
write_json
(path.parent / demo['trajectory_file']).resolve
np.load
ValueError
arrays[name].copy
ground_program
run_sequence
run_stage
report.update
type
```
