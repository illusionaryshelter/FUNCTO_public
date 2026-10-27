# `tests/planning/run_recorded_skill.py` 实现契约

原适配器算法代码基线：`be44b54`。当前默认预算接入对应实际代码提交 `9602ebb`，本次文档回填保持该实现内容。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `save_observation_partitions`

[定义位置](../../../tests/planning/run_recorded_skill.py#L29)。

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

## `load_contact_bindings`

[定义位置](../../../tests/planning/run_recorded_skill.py#L68)。

输入签名：

```python
def load_contact_bindings(path):
```

实现说明：

> Validate recorded contact input and construct independent support facts.
>
> Ordinary contact features bind through explicit source task declarations.
> Initial support carries observed plane evidence and an existing interval;
> its entrance geometry and height baseline belong to scene construction.

显式输出分支：

```python
(inputs, tuple((InitialSupportRelation(**row) for row in declarations)))
```

读取的对象属性：

```python
json.loads
Path(path).read_text
inputs['source'].strip
inputs.get
```

写入目标：

```python
inputs
required
optional
declarations
row
```

直接调用符号：

```python
json.loads
Path(path).read_text
Path
isinstance
set
ValueError
inputs['source'].strip
inputs.get
any
tuple
InitialSupportRelation
```

## `load_joint_reference`

[定义位置](../../../tests/planning/run_recorded_skill.py#L94)。

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

## `finger_trajectory_factory`

[定义位置](../../../tests/planning/run_recorded_skill.py#L125)。

输入签名：

```python
def finger_trajectory_factory(problem, feedback, finger_targets, *, timestep, ctrlrange, position_limits, velocity_limits, acceleration_limits):
```

实现说明：

> Bind original events once the native formal duration/map are known.
>
> Actual-position protection bounds are deliberately absent from this
> constructor. They continue guarding measured feedback in the executor;
> they cannot predict a hand shape or replace the original position goals.
> The returned reference starts at measured q/dq and remains a command,
> including when physical contact prevents a target from being reached.

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

[定义位置](../../../tests/planning/run_recorded_skill.py#L141)。

输入签名：

```python
def build(duration_s, progress_map):
```

显式输出分支：

```python
FingerTrajectory.build(joint_names=names, initial_position=[measured[name] for name in names], initial_velocity=[speeds[name] for name in names], commands=commands, duration=duration_s, executable_until=stop, timestep=timestep, position_limits=np.asarray([position_limits[name] for name in names]).T, velocity_limits=np.stack((-velocity, velocity)), acceleration_limits=np.stack((-acceleration, acceleration)), source=f'Original explicit actuator targets: {finger_targets.source}; C1 command reference under native planner acceleration bounds, not calibrated actual acceleration/tracking limits', feedback_id=feedback['feedback_id'], timestamp_s=feedback['sim_time_s'])
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

[定义位置](../../../tests/planning/run_recorded_skill.py#L176)。

输入签名：

```python
def position_controlled_hand(robot, feedback, problem, *, device, execution_limits, position_limits, velocity_limits, trajectory_factory, fixture, execution_events=None):
```

实现说明：

> Bind independent command reference after checking actual execution limits.
>
> The shared test adapter calibrates exactly the same native spheres and
> prismatic translations. This recorded boundary still owns the measured
> q/dq checks, execution policy, command identities and reference factory.

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

[定义位置](../../../tests/planning/run_recorded_skill.py#L203)。

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

[定义位置](../../../tests/planning/run_recorded_skill.py#L230)。

输入签名：

```python
def require_feedback_identity(feedback):
```

实现说明：

> Keep the capture adapter's identity across planning and execution.
>
> A file path only names storage and can change when a capture is copied.
> It cannot replace the ID already carried by current Holding evidence.

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

## `recorded_owner_observation_inputs`

[定义位置](../../../tests/planning/run_recorded_skill.py#L242)。

输入签名：

```python
def recorded_owner_observation_inputs(problem, scene_input, views, mask_rows, *, source, observed_owners=()):
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
Path
```

## `run_prepared`

[定义位置](../../../tests/planning/run_recorded_skill.py#L305)。

输入签名：

```python
def run_prepared(args, *, problem, provenance):
```

实现说明：

> Run the existing scene/solver pipeline on the supplied continuous path.
>
> Problem construction belongs to the caller. The actual feedback is checked
> again here, so an offline reference cannot replace current robot state.
> This function never reconstructs or resamples the supplied reference.

显式输出分支：

无显式条目。

读取的对象属性：

```python
args.output.mkdir
args.output
args.segment
args.candidate
args.manifest.resolve
args.manifest
args.feedback.resolve
args.feedback
args.robot_fixture.resolve
args.robot_fixture
time.monotonic
args.host_process_root_pid
os.getpid
args.mapping_memory_mib
args.max_host_rss_mib
args.max_device_used_mib
json.loads
args.manifest.read_text
(root / candidate['scene_path']).read_text
args.robot_fixture.read_text
args.feedback.read_text
FixedFingerTargets.from_mapping
args.finger_targets.read_text
args.finger_targets
args.finger_targets.resolve
ExecutionLimits.from_mapping
args.execution_limits.read_text
args.execution_limits
args.execution_limits.resolve
args.continuation
args.continuation.read_text
args.rebind
args.rebind.read_text
args.request_ledger
args.request_ledger.read_text
args.request_ledger.resolve
problem.joint_names
np.array_equal
problem.start_q
problem.start_dq
np.allclose
phase.metadata.get
phase.metadata
problem.phases
problem.reference_frame
args.observed_owner
args.owner_mask
owner.owner_id
args.voxel_size
owner_resolutions.update
owner.query.voxel_size
owner.query
args.background_mask
yaml.safe_load
Path(fixture['robot_config_path']).read_text
problem.base_pose
problem.tool_link
problem.tool_transform
robot_validator.describe
robot_validator.checker.query
robot_validator.checker
torch.cuda.is_available
torch.cuda
args.torch_memory_mib
args.scene_gpu_workspace_mib
torch.set_num_threads
torch.cuda.set_per_process_memory_fraction
torch.cuda.get_device_properties(0).total_memory
torch.cuda.get_device_properties
torch.cuda.reset_peak_memory_stats
torch.get_num_threads
args.contact_bindings
row.get
problem.conditions
row.rule_id
row.relation_id
problem.requirement_consumption
frame['T_output_owner'].tolist
supplied_masks[identity, view['id']].resolve
owner_frames.items
owner.query.summary
args.observed_owner.resolve
background_masks.items
(root / capture_path).read_text
(root / view['calibration_path']).read_text
np.array
Image.open(root / view['rgb_path']).convert
Image.open
np.load
np.asarray
args.self_padding_scale
args.self_padding_offset_m
args.self_filter_source
view_output.mkdir
np.save
(labels == SELF).sum
(labels == ENVIRONMENT).sum
(labels == AMBIGUOUS_SHADOW).sum
(view_output / 'self-filter-report.json').write_text
json.dumps
supplied.dtype
np.bool_
supplied.shape
depth.shape
resolved_path.resolve
supplied.sum
(supplied & (labels == SELF)).sum
(supplied & (labels == AMBIGUOUS_SHADOW)).sum
visible.sum
(view_output / 'owner-mask-report.json').write_text
(view_output / 'self-filter-report.json').resolve
next(iter(observations.values()))['depth'].shape
observations.values
args.extent
torch.tensor
args.center
args.fine_regions
args.fine_regions.read_text
row.items
args.depth_error
args.unknown_policy.read_text
args.unknown_policy
scene.observation_report
scene.fixed_geometry_names
np.savez_compressed
scene.coverage.known_free
scene.coverage
scene.coverage.origin
scene.coverage.voxel_size
scene.coverage.observation_ids
scene.unknown_policy
scene.validate_problem
entrance_fk['geometry_poses'].values
robot_validator.checker.geometry_names
args.knots
args.interpolation_buffer_size
args.ik_seeds
args.trajectory_seeds
args.trajectory_iterations
args.max_validation_points
args.max_task_anchors
args.maximum_trajectory_solves
args.planning_seconds
args.search_seconds
execution_limits.describe
args.joint_reference
result.diagnostics.get
result.diagnostics
backend.close
report.update
result.success
(args.output / 'exception.txt').write_text
traceback.format_exc
type(error).__name__
```

写入目标：

```python
report
started
report['wallclock']
report['host_process_scope']
manifest
root
segment
s
candidate
c
scene_input
fixture
feedback
finger_targets
report['finger_targets_path']
execution_limits
report['execution_limits_path']
continuation
rebind
request_ledger
report['request_ledger_input']
measured
velocities
active
planning_feedback
name
field
phase
observed_owners
owner_frames
supplied_masks
task_owners
owner_resolutions
owner
background_masks
path
robot
robot_validator
report['independent_robot_model']
robot['robot_cfg']['kinematics']['lock_joints']
reference_factory
entrance_fk
report['resource_limits']
device
hand
hand_report
report['gripper']
tool_members
report['tool_geometry_names']
contacts
initial_support_relations
report['contact_binding']
contact_inputs
row
report['requirement_consumption']
report['owner_observations']
view
identity
frame
capture_path
capture
capture_fk
filter_geometries
observations
calibration
rgb
depth
filtered
labels
view_output
filter_report
owner_masks
mask_report
index
supplied
visible
resolved_path
owner_masks[identity]
mask_report[identity]
observations[view['id']]
observations[view['id']]['background_mask']
height
width
config
fine_regions
requests
fine_regions[identity]
key
value
scene
report['observation_build']
report['observation_partitions']
report['fixed_robot_installation']
report['unknown_policy']
report['entrance_geometry_diagnostic']
budget
report['planning_called']
backend
result
report['planner_wallclock']
report['elapsed_seconds']
```

直接调用符号：

```python
args.output.mkdir
str
args.manifest.resolve
args.feedback.resolve
args.robot_fixture.resolve
time.monotonic
configure_host_process_root
os.getpid
min
ValueError
mark_wallclock
json.loads
args.manifest.read_text
Path
next
(root / candidate['scene_path']).read_text
args.robot_fixture.read_text
args.feedback.read_text
require_feedback_identity
FixedFingerTargets.from_mapping
args.finger_targets.read_text
args.finger_targets.resolve
ExecutionLimits.from_mapping
args.execution_limits.read_text
args.execution_limits.resolve
args.continuation.read_text
args.rebind.read_text
args.request_ledger.read_text
isinstance
args.request_ledger.resolve
continuation_execution_inputs
dict
zip
tuple
list
np.array_equal
getattr
np.allclose
phase.metadata.get
load_observed_owner
recorded_owner_observation_inputs
scene_task_owner_ids
observation_resolutions
owner_resolutions.update
len
yaml.safe_load
Path(fixture['robot_config_path']).read_text
build_robot_validator
robot_validator.describe
finger_trajectory_factory
float
robot_validator.checker.query
torch.cuda.is_available
RuntimeError
torch.set_num_threads
torch.cuda.set_per_process_memory_fraction
torch.cuda.get_device_properties
torch.cuda.reset_peak_memory_stats
torch.get_num_threads
DeviceCfg
position_controlled_hand
tool_geometry_names
load_contact_bindings
ContactRule
any
row.get
bind_grasp_contact_rules
frame['T_output_owner'].tolist
supplied_masks[identity, view['id']].resolve
owner_frames.items
owner.query.summary
args.observed_owner.resolve
background_masks.items
(root / capture_path).read_text
robot_self_filter_shapes
(root / view['calibration_path']).read_text
np.array
Image.open(root / view['rgb_path']).convert
Image.open
np.load
classify_depth
np.asarray
view_output.mkdir
np.save
int
(labels == SELF).sum
(labels == ENVIRONMENT).sum
(labels == AMBIGUOUS_SHADOW).sum
(view_output / 'self-filter-report.json').write_text
json.dumps
enumerate
resolved_path.resolve
supplied.sum
(supplied & (labels == SELF)).sum
(supplied & (labels == AMBIGUOUS_SHADOW)).sum
visible.sum
(view_output / 'owner-mask-report.json').write_text
(view_output / 'self-filter-report.json').resolve
iter
observations.values
MapperCfg
torch.tensor
args.fine_regions.read_text
row.items
build_scene_from_rgbd
args.unknown_policy.read_text
save_observation_partitions
np.savez_compressed
scene.validate_problem
entrance_fk['geometry_poses'].values
PlanningBudget
MotionBackend
plan_and_record
execution_limits.describe
load_joint_reference
result.diagnostics.get
backend.close
report.update
(args.output / 'exception.txt').write_text
traceback.format_exc
type
_write_diagnostic_json
```

## `run_prepared.mark_wallclock`

[定义位置](../../../tests/planning/run_recorded_skill.py#L339)。

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
report['wallclock']['events'][name]
```

直接调用符号：

```python
time.time_ns
time.monotonic
print
```

## `run`

[定义位置](../../../tests/planning/run_recorded_skill.py#L690)。

输入签名：

```python
def run(args):
```

实现说明：

> Compile recorded program/scene inputs once, then run the shared request.

显式输出分支：

```python
run_prepared(args, problem=problem, provenance=provenance)
```

读取的对象属性：

```python
json.loads
args.manifest.read_text
args.manifest
args.segment
args.candidate
args.feedback.read_text
args.feedback
(root / segment['program_path']).read_text
(root / candidate['scene_path']).read_text
args.output.mkdir
args.output
type(error).__name__
(args.output / 'exception.txt').write_text
traceback.format_exc
```

写入目标：

```python
manifest
root
segment
row
candidate
feedback
q
dq
active
selected
name
problem
provenance
```

直接调用符号：

```python
json.loads
args.manifest.read_text
Path
next
args.feedback.read_text
dict
zip
prepare_candidate
(root / segment['program_path']).read_text
(root / candidate['scene_path']).read_text
args.output.mkdir
_write_diagnostic_json
type
(args.output / 'exception.txt').write_text
traceback.format_exc
run_prepared
```

## `parse_args`

[定义位置](../../../tests/planning/run_recorded_skill.py#L719)。

输入签名：

```python
def parse_args(argv=None):
```

实现说明：

> Share the existing resource/scene CLI without a second option schema.

`--planning-seconds` 省略时直接取 `PlanningBudget.planning_seconds`，当前为 60 s。`run_prepared` 将解析值传给原 `PlanningBudget` 构造器，随后由 `plan_and_record → plan_trajectory` 维护原完整请求的累计计时。显式命令值继续通过原参数接口解析。

显式输出分支：

```python
parser.parse_args(argv)
```

读取的对象属性：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
PlanningBudget.planning_seconds
```

写入目标：

```python
parser
name
```

直接调用符号：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
```
