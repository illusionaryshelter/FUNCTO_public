# `tests/planning/prepare_recorded_task.py` 实现契约

代码基线：`3266476`。
职责和数据语义见[组件说明](../11-device-control-and-evaluation.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `json_value`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L29)。

输入签名：

```python
def json_value(value):
```

实现说明：

> Preserve numerical evidence as JSON values, never opaque strings.

显式输出分支：

```python
value.tolist()
value.item()
```

读取的对象属性：

```python
np.ndarray
value.tolist
np.generic
value.item
type(value).__name__
```

写入目标：

无显式条目。

直接调用符号：

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

实现说明：

> Bound metadata reads before decoding them in this small CPU runner.

显式输出分支：

```python
json.loads(path.read_text())
```

读取的对象属性：

```python
path.stat().st_size
path.stat
json.loads
path.read_text
```

写入目标：

```python
path
```

直接调用符号：

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

实现说明：

> Return the public PlanningProblem and explicit input provenance.
>
> The caller owns the active-joint layout. Its supplied positions and
> measured velocities pass through unchanged; this helper neither projects
> joints nor infers finger symmetry. The source program and scene remain
> untouched, including all functional events and protected source samples.

显式输出分支：

```python
(problem, provenance)
```

读取的对象属性：

```python
np.allclose
prepared.get
binding.get
bindings.values
np.array_equal
bindings.items
maps[0].copy
changes.append
maps[0].tolist
np.abs(scene_eef - measured_eef).max
np.abs
problem.joint_names
problem.start_q
problem.start_dq
```

写入目标：

```python
prepared
scene_eef
measured_eef
key
changes
bindings
binding
maps
value
source
prepared['carrier_mapping']
prepared['carrier_mapping_source']
problem
provenance
```

直接调用符号：

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

实现说明：

> Report independent metre/radian errors for two rigid poses.

显式输出分支：

```python
{'translation_m': float(np.linalg.norm(relative[:3, 3])), 'rotation_rad': float(Rotation.from_matrix(relative[:3, :3]).magnitude()), 'maximum_abs_matrix_error': float(np.abs(actual - expected).max())}
```

读取的对象属性：

```python
np.linalg.inv
np.linalg
np.linalg.norm
Rotation.from_matrix(relative[:3, :3]).magnitude
Rotation.from_matrix
np.abs(actual - expected).max
np.abs
```

写入目标：

```python
relative
```

直接调用符号：

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

实现说明：

> Decode the fixture's explicit position/unit-wxyz calibration contract.

显式输出分支：

```python
_pose_array(transform, name)
```

读取的对象属性：

```python
np.asarray
position.shape
quaternion.shape
np.isfinite(position).all
np.isfinite
np.isfinite(quaternion).all
np.isclose
np.linalg.norm
np.linalg
np.eye
Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
```

写入目标：

```python
position
quaternion
transform
transform[:3, :3]
transform[:3, 3]
```

直接调用符号：

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

显式输出分支：

```python
transform
```

读取的对象属性：

```python
np.eye
np.asarray(rotation).reshape
np.asarray
```

写入目标：

```python
transform
transform[:3, :3]
transform[:3, 3]
```

直接调用符号：

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

实现说明：

> Include every actual sphere link in each native finger's full subtree.
>
> A pad can live on a fixed child body. Reading only the joint's immediate
> body would leave that pad outside the binary gripper's swept geometry.
> All ownership here comes from the native robot tree and planner sphere
> bindings, not a finger/pad naming convention.

显式输出分支：

```python
result
```

读取的对象属性：

```python
model.joint(name).bodyid
model.joint
collision_spheres.items
model.body(link).id
model.body
roots.items
model.body_parentid
result[joint_name].append
result.values
```

写入目标：

```python
roots
name
result
link
spheres
body_id
joint_name
root_id
ancestor
values
links
```

直接调用符号：

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

实现说明：

> Read scalar robot q/dq columns only, then check both robot-only FK paths.
>
> The robot fixture owns the source address mapping and the native model.
> Neither recorded object coordinates, recorded body/site poses, contacts,
> nor the simulator's full scene are read. No physics step is performed.

显式输出分支：

```python
result
```

读取的对象属性：

```python
mujoco.MjModel.from_xml_path
mujoco.MjModel
mujoco.MjData
URDF.load
yaml.safe_load
(root / fixture['robot_config_path']).read_text
model.joint(index).name
model.joint
model.njnt
model.nq
model.nv
set(arms).intersection
urdf.actuated_joint_names
set(fixture['planning_joint_names']).issubset
joint.type
mujoco.mjtJoint.mjJNT_HINGE
mujoco.mjtJoint
mujoco.mjtJoint.mjJNT_SLIDE
model.joint(name).qposadr
model.joint(name).dofadr
model.body(fixture['base_link']).id
model.body
model.site(fixture['tcp_site']).id
model.site
h5py.File
q_stream.ndim
dq_stream.ndim
times.ndim
q_stream.shape
dq_stream.shape
np.array
np.isfinite(q).all
np.isfinite
np.isfinite(dq).all
data.qpos
data.qvel
mujoco.mj_kinematics
data.xpos
data.xmat
data.site_xpos
data.site_xmat
urdf.update_cfg
urdf.get_transform
errors.values
np.asarray
set(old_names).issubset
old_q.shape
names.index
np.abs(expected - old_q).max
np.abs
frozen.get
model.jnt_range
model.joint(name).id
np.maximum
q.tolist
dq.tolist
finger_links.values
feedback_pose.tolist
base.tolist
tool.tolist
limits.tolist
episode_path.resolve
frozen_path.resolve
fixture_path.resolve
(root / fixture['binding_audit_path']).resolve
```

写入目标：

```python
root
mjcf_path
urdf_path
episode_path
key
model
data
urdf
robot_config
binding
names
item
arms
fingers
index
finger_links
joint
q_addresses
name
dq_addresses
base
tool
base_id
site_id
result
stream
q_stream
dq_stream
times
segment
frame
frozen_path
frozen
q
dq
timestamp
feedback_pose
data.qpos[q_addresses]
data.qvel[dq_addresses]
data.time
native_base
native_tcp
planner_tcp
errors
value
checks
names_key
values_key
old_names
old_q
expected
error
checks[values_key + '_maximum_abs_error']
frozen_time
timestamp_error
limits
violation
result[segment['id']]
links
link
```

直接调用符号：

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
deepcopy
finger_links.values
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

[定义位置](../../../tests/planning/prepare_recorded_task.py#L320)。

输入签名：

```python
def _select_planning_joints(feedback):
```

实现说明：

> Select a declared layout by name without changing the saved full state.

显式输出分支：

```python
selected
```

读取的对象属性：

```python
feedback['joint_names'].index
selected.update
```

写入目标：

```python
selected
names
name
indices
index
```

直接调用符号：

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

[定义位置](../../../tests/planning/prepare_recorded_task.py#L334)。

输入签名：

```python
def _problem_summary(problem):
```

实现说明：

> Expose compiled constraints and barriers without serializing an executor.

显式输出分支：

```python
{'joint_names': list(problem.joint_names), 'phase_count': problem.phase_count, 'anchor_count': len(problem.anchors), 'stop_progress': problem.stop_progress.tolist(), 'phases': [{'name': phase.name, 'reference_node_count': len(phase.reference.progress), 'metadata': phase.metadata} for phase in problem.phases], 'source_time_map': problem.source_time_map, 'events': [{'event_id': event.event_id, 'skill_id': event.skill_id, 'kind': event.kind, 'source_frame': event.source_frame, 'progress': event.progress, 'requires_evidence': event.requires_evidence, 'stop': event.stop} for event in problem.events], 'conditions': list(problem.conditions), 'requirement_consumption': problem.requirement_consumption, 'requirement_counts': {'supports': len(problem.supports), 'interiors': len(problem.interiors), 'pose_events': len(problem.pose_events), 'coordinate_events': len(problem.coordinate_events)}}
```

读取的对象属性：

```python
problem.joint_names
problem.phase_count
problem.anchors
problem.stop_progress.tolist
problem.stop_progress
phase.name
phase.reference.progress
phase.reference
phase.metadata
problem.phases
problem.source_time_map
event.event_id
event.skill_id
event.kind
event.source_frame
event.progress
event.requires_evidence
event.stop
problem.events
problem.conditions
problem.requirement_consumption
problem.supports
problem.interiors
problem.pose_events
problem.coordinate_events
```

写入目标：

```python
phase
event
```

直接调用符号：

```python
list
len
problem.stop_progress.tolist
```

## `_write_json`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L361)。

输入签名：

```python
def _write_json(path, data):
```

实现说明：

> Write the exact named artifact only after its values are assembled.

显式输出分支：

无显式条目。

读取的对象属性：

```python
path.parent.mkdir
path.parent
path.write_text
json.dumps
```

写入目标：

无显式条目。

直接调用符号：

```python
path.parent.mkdir
path.write_text
json.dumps
```

## `_prepare_segments`

[定义位置](../../../tests/planning/prepare_recorded_task.py#L368)。

输入签名：

```python
def _prepare_segments(manifest, feedback_by_segment, output):
```

实现说明：

> Run every current candidate, preserving old labels only as comparison.

显式输出分支：

```python
rows
```

读取的对象属性：

```python
output.parent
feedback_path.resolve
item.update
type(error).__name__
row['candidates'].append
rows.append
```

写入目标：

```python
root
rows
segment
identifier
feedback
feedback_path
planning_feedback
program
row
candidate
item
problem
provenance
item['error']
```

直接调用符号：

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

[定义位置](../../../tests/planning/prepare_recorded_task.py#L403)。

输入签名：

```python
def run(args):
```

实现说明：

> Validate entrances first, then compile every declared current candidate.

显式输出分支：

```python
report
```

读取的对象属性：

```python
os.environ.get
os.environ
time.perf_counter
args.manifest
args.robot_fixture
args.output
args.manifest.resolve
args.robot_fixture.resolve
resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
resource.getrusage
resource.RUSAGE_SELF
```

写入目标：

```python
started
manifest
fixture
identifiers
segment
feedback
segments
candidates
candidate
passed
report
```

直接调用符号：

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

[定义位置](../../../tests/planning/prepare_recorded_task.py#L434)。

输入签名：

```python
def main():
```

显式输出分支：

```python
0 if report['status'] == 'passed' else 1
```

读取的对象属性：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
json.dumps
args.output.resolve
args.output
```

写入目标：

```python
parser
args
report
```

直接调用符号：

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
