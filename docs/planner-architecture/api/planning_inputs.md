# `manipulation/planning_inputs.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../14-grounding-planning-portfolio.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `save_observation_partitions`

[定义位置](../../../manipulation/planning_inputs.py#L14)。

输入签名：

```python
def save_observation_partitions(scene, output):
```

实现说明：

> Save original entity/background masks and the admitted fusion mask.
>
> Future --owner-mask inputs use the saved semantic entity files and
> --background-mask uses the saved background candidate. The fusion
> background records this run's filtering and one-pixel boundary; feeding
> it back as a candidate would incorrectly grow the excluded boundary.
> A reused background field produces no new fusion-background mask.

显式输出分支：

```python
result
```

读取的对象属性：

```python
scene.observation_partition_masks.items
scene.observation_partition_masks
folder.mkdir
scene.observation_report
masks['owner_masks'].items
np.save
path.resolve
(folder / 'partition-report.json').write_text
json.dumps
```

写入目标：

```python
result
name
masks
folder
row
index
identity
mask
path
row['owner_masks'][identity]
key
filename
report_key
row[report_key]
result[name]
```

直接调用符号：

```python
scene.observation_partition_masks.items
Path
folder.mkdir
deepcopy
enumerate
masks['owner_masks'].items
np.save
str
path.resolve
(folder / 'partition-report.json').write_text
json.dumps
```

## `finger_trajectory_factory`

[定义位置](../../../manipulation/planning_inputs.py#L51)。

输入签名：

```python
def finger_trajectory_factory(problem, feedback, finger_targets, *, incoming_command, timestep, ctrlrange, position_limits, velocity_limits, acceleration_limits):
```

实现说明：

> Bind original events once the native formal duration/map are known.
>
> Actual-position protection bounds are deliberately absent from this
> constructor. They continue guarding measured feedback in the executor;
> they cannot predict a hand shape or replace the original position goals.
> The returned reference continues the explicitly supplied active command.
> Measured q/dq remain separate entrance observations, including when contact
> prevents the physical fingers from reaching the issued target.

显式输出分支：

```python
build
```

读取的对象属性：

无显式条目。

写入目标：

```python
names
measured
speeds
build.timestep
```

直接调用符号：

```python
tuple
dict
zip
float
```

## `finger_trajectory_factory.build`

[定义位置](../../../manipulation/planning_inputs.py#L68)。

输入签名：

```python
def build(duration_s, progress_map):
```

显式输出分支：

```python
FingerTrajectory.build(joint_names=names, incoming_command=incoming_command, measured_initial_position=[measured[name] for name in names], measured_initial_velocity=[speeds[name] for name in names], commands=commands, duration=duration_s, executable_until=stop, timestep=timestep, position_limits=np.asarray([position_limits[name] for name in names]).T, velocity_limits=np.stack((-velocity, velocity)), acceleration_limits=np.stack((-acceleration, acceleration)), source=f'Original explicit actuator targets: {finger_targets.source}; C1 command reference under native planner acceleration bounds, not calibrated actual acceleration/tracking limits', feedback_id=feedback['feedback_id'], timestamp_s=feedback['timestamp_s'])
```

读取的对象属性：

```python
progress_map.event_clock.items
progress_map.event_clock
events.append
clock.items
finger_targets.bind
np.asarray
FingerTrajectory.build
np.asarray([position_limits[name] for name in names]).T
np.stack
finger_targets.source
```

写入目标：

```python
events
identity
clock
event
key
value
barriers
row
stop
commands
velocity
name
acceleration
```

直接调用符号：

```python
progress_map.event_clock.items
deepcopy
event_occurrence_key
ValueError
events.append
clock.items
min
finger_targets.bind
SimpleNamespace
np.asarray
FingerTrajectory.build
np.stack
```

## `position_controlled_hand`

[定义位置](../../../manipulation/planning_inputs.py#L105)。

输入签名：

```python
def position_controlled_hand(robot, feedback, problem, *, device, execution_limits, position_limits, velocity_limits, trajectory_factory, fixture, execution_events=None):
```

实现说明：

> Bind independent command reference after checking actual execution limits.
>
> The shared hand contract uses the registered complete robot FK for its
> prismatic coordinates. This boundary owns the measured q/dq checks,
> execution policy, command identities and synchronous reference factory.

显式输出分支：

```python
(hand, report)
```

读取的对象属性：

```python
problem.events
execution_limits.bind
problem.joint_names
np.asarray
np.any
np.abs
trajectory_factory.timestep
np.empty
event.event_id
event.skill_id
event.occurrence
event.kind
event.progress
execution_limits.source
actual_fingers.tolist
execution_limits.describe
```

写入目标：

```python
execution_events
measured
finger_names
binding
event
name
actual_fingers
actual_dq
initial_bounds
hand
report
```

直接调用符号：

```python
dict
zip
tuple
execution_limits.bind
SimpleNamespace
asdict
np.asarray
np.any
np.abs
ValueError
PrismaticGripper
np.empty
len
actual_fingers.tolist
execution_limits.describe
```

## `position_controlled_hand.nominal_geometry`

[定义位置](../../../manipulation/planning_inputs.py#L132)。

输入签名：

```python
def nominal_geometry(duration_s, progress_map, command_reference):
```

显式输出分支：

```python
bind_nominal_finger_geometry(command_reference, problem.conditions, progress_map.event_clock, progress_interval=(progress_map.s[0], progress_map.s[-1]))
```

读取的对象属性：

```python
command_reference.duration
problem.conditions
progress_map.event_clock
progress_map.s
```

写入目标：

无显式条目。

直接调用符号：

```python
ValueError
bind_nominal_finger_geometry
```

## `require_feedback_identity`

[定义位置](../../../manipulation/planning_inputs.py#L157)。

输入签名：

```python
def require_feedback_identity(feedback):
```

实现说明：

> Keep the capture adapter's identity across planning and execution.
>
> A file path only names storage and can change when a capture is copied.
> It cannot replace the ID carried by the current named robot measurement
> and its explicitly sourced event/relation records.

显式输出分支：

```python
identity
```

读取的对象属性：

```python
feedback.get
identity.strip
```

写入目标：

```python
identity
```

直接调用符号：

```python
feedback.get
isinstance
identity.strip
ValueError
```

## `owner_observation_inputs`

[定义位置](../../../manipulation/planning_inputs.py#L170)。

输入签名：

```python
def owner_observation_inputs(problem, scene_input, views, mask_rows, *, source, observed_owners=()):
```

实现说明：

> Resolve masks only for entities selected for RGB-D reconstruction.
>
> Existing models retain their registered identity and motion. They do not
> participate in the new observation-mask contract, including under complete
> occlusion. New entity coordinates come from the explicit task binding.

显式输出分支：

```python
(frames, masks)
```

读取的对象属性：

```python
scene_input.get
reference.strip
identity.strip
evidence.strip
owner.owner_id
frames.items
```

写入目标：

```python
carrier
initial
_
frames
reference
frames[reference]
observations
record
identity
evidence
frames[identity]
existing
owner
frame
view_ids
view
masks
row
path
masks[owner, view]
```

直接调用符号：

```python
problem_owner_binding
scene_input.get
isinstance
reference.strip
ValueError
_pose_array
set
identity.strip
evidence.strip
frames.items
len
```

## `load_observed_owner`

[定义位置](../../../manipulation/planning_inputs.py#L227)。

输入签名：

```python
def load_observed_owner(path):
```

实现说明：

> Restore the same pre-relationship field and the declared rigid motion.

显式输出分支：

```python
CollisionOwner(row['owner_id'], field, PosePath(np.array([0.0, 1.0]), np.stack((pose, pose)), row['source']), row['source'], moving=bool(row['attachments']), attachments=tuple((Attachment(**relation) for relation in row['attachments'])))
```

读取的对象属性：

```python
json.loads
Path(path).read_text
FrozenObservedField.load
np.asarray
np.array
np.stack
```

写入目标：

```python
row
field
pose
relation
```

直接调用符号：

```python
json.loads
Path(path).read_text
Path
FrozenObservedField.load
np.asarray
CollisionOwner
PosePath
np.array
np.stack
bool
tuple
Attachment
```

## `problem_owner_binding`

[定义位置](../../../manipulation/planning_inputs.py#L241)。

输入签名：

```python
def problem_owner_binding(problem, owner_id):
```

实现说明：

> Copy the exact current owner pose and deduplicated held spans.
>
> The inverse attachment maps owner-local points into the tool frame.
> Release IDs and progress come from the current compiler's held metadata,
> never a source frame guessed to be the executable release time.

显式输出分支：

```python
(initial, attachments)
```

读取的对象属性：

```python
problem.phases
phase.metadata.get
phase.metadata
phases[0].metadata
np.allclose
metadata.get
np.linalg.inv
np.linalg
inverse[:3, 3].tolist
Rotation.from_matrix(inverse[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist
Rotation.from_matrix(inverse[:3, :3]).as_quat
Rotation.from_matrix
attachments.append
```

写入目标：

```python
phases
phase
initial
attachments
metadata
inverse
relation
same_span
item
```

直接调用符号：

```python
phase.metadata.get
ValueError
_pose_array
np.allclose
metadata.get
np.linalg.inv
list
inverse[:3, 3].tolist
Rotation.from_matrix(inverse[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist
Rotation.from_matrix(inverse[:3, :3]).as_quat
Rotation.from_matrix
attachments.append
```

## `load_joint_reference`

[定义位置](../../../manipulation/planning_inputs.py#L277)。

输入签名：

```python
def load_joint_reference(path, problem):
```

实现说明：

> Read a bounded named q/dq recording without changing current task goals.
>
> All coordinates and timestamps come from the supplied record. Shape,
> ordering and finite-value checks prevent malformed guidance from entering
> a native seed fit; goal/domain acceptance still belongs to the backend.

显式输出分支：

```python
None
JointTrajectory(names, t, q, dq, t / t[-1], progress, (), (), {})
```

读取的对象属性：

```python
path.stat().st_size
path.stat
np.load
record.files
record['joint_names'].tolist
np.asarray(record[key], float).copy
np.asarray
problem.joint_names
t.ndim
np.any
np.diff
q.shape
dq.shape
progress.shape
t.shape
problem.start_progress
problem.phase_count
np.isfinite(value).all
np.isfinite
```

写入目标：

```python
path
record
names
t
q
dq
progress
key
value
```

直接调用符号：

```python
Path
path.stat
ValueError
np.load
set
tuple
record['joint_names'].tolist
np.asarray(record[key], float).copy
np.asarray
len
np.any
np.diff
all
np.isfinite(value).all
np.isfinite
JointTrajectory
```
