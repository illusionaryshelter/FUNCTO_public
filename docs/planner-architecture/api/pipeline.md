# `manipulation/pipeline.py` 实现契约

代码基线：`82a3d66`。本次仅文档提交引用该实现提交，组件代码与该基线一致。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `motion_kind`

[定义位置](../../../manipulation/pipeline.py#L26)。

输入签名：

```python
def motion_kind(task):
```

实现说明：

> Select the applied family from semantic evidence, not trajectory rank.
>
> The existing semantic ``motion`` proposal can specify planar motion for a
> whole-stage support/contact requirement; complete poses must still fit it.
> ``auto`` for a general object task does not authorize a new mechanical
> restriction. ``run_stage`` separately reports every compatible observed
> family, so keeping a carry free does not hide its planar/linear geometry.
> An articulation task supplies mechanism evidence and permits automatic
> selection only of a geometrically supported joint family downstream.

显式输出分支：

```python
'free'
task['motion']
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `serializable`

[定义位置](../../../manipulation/pipeline.py#L44)。

输入签名：

```python
def serializable(value):
```

实现说明：

> Convert numeric outputs without unsafe pickle or silent NaN encoding.

显式输出分支：

```python
value.tolist()
value.item()
{key: serializable(item) for (key, item) in value.items()}
[serializable(item) for item in value]
value
```

读取的对象属性：

```python
np.ndarray
value.tolist
np.generic
value.item
value.items
```

写入目标：

```python
key
item
```

直接调用符号：

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

显式输出分支：

无显式条目。

读取的对象属性：

```python
Path(path).write_text
json.dumps
```

写入目标：

无显式条目。

直接调用符号：

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

显式输出分支：

```python
matrix
```

读取的对象属性：

```python
np.asarray
matrix.shape
np.isfinite(matrix).all
np.isfinite
np.allclose
matrix[:3, :3].T
np.eye
np.isclose
np.linalg.det
np.linalg
```

写入目标：

```python
matrix
```

直接调用符号：

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

显式输出分支：

```python
SupportSurface(np.asarray(value['origin']), np.asarray(value['basis']), np.asarray(value['normal']), np.asarray(value['boundary']), source=value.get('source', 'provided_support_geometry'))
```

读取的对象属性：

```python
np.asarray
value.get
```

写入目标：

无显式条目。

直接调用符号：

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

显式输出分支：

```python
InteriorRegion(np.asarray(value['halfspaces']), value['source'])
InteriorRegion.from_vertices(value['vertices'], value['source'])
```

读取的对象属性：

```python
np.asarray
InteriorRegion.from_vertices
```

写入目标：

无显式条目。

直接调用符号：

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

实现说明：

> Create and validate an object trajectory from a grounded demonstration.
>
> Required fields: ``task`` (relation/interaction/motion), ``demo_poses``,
> ``scene`` (initial_pose/body_points/reference_transform), ``carrier_id``,
> ``reference_id`` and ``reference_frame``. All scene geometry shares that
> reference frame; object points use a fixed body frame. Optional grasp or
> contact geometry is evidence only, never a fabricated rigid EE transform.

显式输出分支：

```python
{'status': 'passed', 'output_type': 'object_trajectory', 'carrier_id': observation['carrier_id'], 'reference_id': observation['reference_id'], 'reference_frame': observation['reference_frame'], 'pose_convention': 'column-vector reference_from_fixed_body, metres, 4x4 SE(3)', 'task': task, 'poses': poses, 'frame_indices': frame_indices, 'body_points': body, 'model': model.to_dict(), 'goal_pose': goal, 'goal_geometry': None if geometry is None else geometry.to_dict(), 'goal_checks': goal_report, 'motion_checks': transferred['diagnostics'], 'interaction_geometry': observation.get('interaction_geometry'), 'interaction_geometry_status': 'provided' if observation.get('interaction_geometry') is not None else 'not_estimated_for_object_trajectory', 'observation_evidence': observation.get('evidence', {}), 'geometry_coverage': scene.get('geometry_coverage', 'supplied_points_only'), 'path_scope': 'demonstration transfer; no complete-scene collision or force guarantee'}
```

读取的对象属性：

```python
observation.get
observation[key].strip
scene.get
np.bool_
np.isscalar
np.number
np.isfinite
np.asarray
body.ndim
body.shape
np.isfinite(body).all
np.linspace
preliminary['model'].kind
np.asarray(preliminary['poses'][-1]).copy
np.allclose
goal_report.update
frame_indices.shape
frame_indices.dtype.kind
frame_indices.dtype
(frame_indices < 0).any
(frame_indices[1:] <= frame_indices[:-1]).any
model.kind
model.to_dict
geometry.to_dict
```

写入目标：

```python
task
relation
interaction
key
scene
margin
start
transform
body
demo
observed_structure
observed_structure['pose_frame']
options
preliminary
exact_goal
preferred
preferred[:3, 3]
geometry
goal
goal_report
transferred
poses
goal_report['containment_status']
region
index
checks
pose
clearance
item
frame_indices
model
transferred['diagnostics']['observed_structure']
transferred['diagnostics']['applied_constraint']
```

直接调用符号：

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

实现说明：

> Move reference-attached goals when a previous stage moved that entity.

显式输出分支：

无显式条目。

读取的对象属性：

```python
surface.origin
surface.basis
surface.normal
surface.to_dict
region.halfspaces
delta[:3, :3].T
InteriorRegion(np.column_stack((normals, offsets)), region.source).to_dict
np.column_stack
region.source
```

写入目标：

```python
scene['reference_transform']
key
scene[key]
surface
surface.origin
surface.basis
surface.normal
scene['support']
region
normals
offsets
```

直接调用符号：

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

实现说明：

> Run an already ordered stage list, updating carrier/reference states.
>
> Each stage's ``reference_pose`` describes the reference at capture time.
> These transforms and every initial carrier pose must share one world or
> otherwise explicitly registered reference frame. This is state handoff,
> not an autonomous task-order planner.

显式输出分支：

```python
{'status': 'passed', 'reference_frame': frame, 'stages': output, 'final_entity_poses': states, 'unobserved_entity_states': invalidated}
```

读取的对象属性：

```python
observation.get
states[carrier].copy
np.linalg.inv
np.linalg
placements.pop
np.allclose
changed.pop
placements.items
states.pop
changed.append
np.asarray(result['poses'][-1]).copy
np.asarray
output.append
```

写入目标：

```python
states
body_frames
placements
invalidated
output
frame
original
observation
scene
carrier
reference
body_frame
body_frames[carrier]
scene['initial_pose']
captured
delta
result
changed
moved
dependent
owner
invalidated[dependent]
states[carrier]
placements[carrier]
```

直接调用符号：

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

实现说明：

> Replay grounded observations, preserving failures and fresh-run semantics.

显式输出分支：

```python
result
```

读取的对象属性：

```python
json.loads
path.read_text
destination.exists
destination.is_dir
destination.iterdir
destination.mkdir
path.resolve
(path.parent / demo['trajectory_file']).resolve
path.parent
np.load
arrays.files
arrays[name].copy
report.update
type(error).__name__
```

写入目标：

```python
path
destination
inputs
report
demo
array_path
demo['trajectory_file']
arrays
name
demo[name]
result
report['status']
```

直接调用符号：

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
