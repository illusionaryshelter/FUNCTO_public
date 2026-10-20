# `tests/planning/run_recorded_skill.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `save_observation_partitions`

[定义位置](../../../tests/planning/run_recorded_skill.py#L29)。

输入签名：

```python
def save_observation_partitions(scene, output):
```

直接读取的对象属性：`scene.observation_partition_masks`、`scene.observation_report`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[name]`。

直接调用：

```python
scene.observation_partition_masks.items
Path
folder.mkdir
np.save
str
path.resolve
deepcopy
(folder / 'partition-report.json').write_text
json.dumps
```

## `load_contact_bindings`

[定义位置](../../../tests/planning/run_recorded_skill.py#L51)。

输入签名：

```python
def load_contact_bindings(path):
```

输出分支的源码表达式：

```python
(inputs, tuple((InitialSupportRelation(**row) for row in declarations)))
```

直接调用：

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

[定义位置](../../../tests/planning/run_recorded_skill.py#L77)。

输入签名：

```python
def load_joint_reference(path, problem):
```

直接读取的对象属性：`record.files`、`problem.joint_names`、`problem.start_progress`、`problem.phase_count`。

输出分支的源码表达式：

```python
None
JointTrajectory(names, t, q, dq, t / t[-1], progress, (), (), {})
```

直接调用：

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

[定义位置](../../../tests/planning/run_recorded_skill.py#L108)。

输入签名：

```python
def finger_trajectory_factory(problem, feedback, finger_targets, *, timestep, ctrlrange, position_limits, velocity_limits, acceleration_limits):
```

输出分支的源码表达式：

```python
build
```

直接写入的状态或数组位置：`build.timestep`。

直接调用：

```python
tuple
dict
zip
float
```

## `finger_trajectory_factory.build`

[定义位置](../../../tests/planning/run_recorded_skill.py#L124)。

输入签名：

```python
def finger_trajectory_factory.build(duration_s, progress_map):
```

输出分支的源码表达式：

```python
FingerTrajectory.build(joint_names=names, initial_position=[measured[name] for name in names], initial_velocity=[speeds[name] for name in names], commands=commands, duration=duration_s, executable_until=stop, timestep=timestep, position_limits=np.asarray([position_limits[name] for name in names]).T, velocity_limits=np.stack((-velocity, velocity)), acceleration_limits=np.stack((-acceleration, acceleration)), source=f'Original explicit actuator targets: {finger_targets.source}; C1 command reference under native planner acceleration bounds, not calibrated actual acceleration/tracking limits', feedback_id=feedback['feedback_id'], timestamp_s=feedback['sim_time_s'])
```

直接调用：

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

[定义位置](../../../tests/planning/run_recorded_skill.py#L159)。

输入签名：

```python
def position_controlled_hand(robot, feedback, problem, *, device, execution_limits, position_limits, velocity_limits, trajectory_factory, fixture, execution_events=None):
```

直接读取的对象属性：`problem.events`、`problem.joint_names`。

输出分支的源码表达式：

```python
(hand, report, {key: geometry[key] for key in ('robot_spheres', 'tool_position', 'tool_quaternion', 'sphere_indices_by_link')})
```

直接调用：

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
calibrate_native_hand
PrismaticGripper
np.empty
len
list
geometry['current_positions'].tolist
geometry['joint_limits'].tolist
execution_limits.describe
```

## `position_controlled_hand.nominal_geometry`

[定义位置](../../../tests/planning/run_recorded_skill.py#L189)。

输入签名：

```python
def position_controlled_hand.nominal_geometry(duration_s, progress_map, command_reference):
```

直接读取的对象属性：`problem.conditions`。

输出分支的源码表达式：

```python
bind_nominal_finger_geometry(command_reference, problem.conditions, progress_map.event_clock, progress_interval=(progress_map.s[0], progress_map.s[-1]))
```

直接调用：

```python
ValueError
bind_nominal_finger_geometry
```

## `require_feedback_identity`

[定义位置](../../../tests/planning/run_recorded_skill.py#L231)。

输入签名：

```python
def require_feedback_identity(feedback):
```

直接读取的对象属性：`feedback.get`。

输出分支的源码表达式：

```python
identity
```

直接调用：

```python
feedback.get
isinstance
identity.strip
ValueError
```

## `recorded_owner_observation_inputs`

[定义位置](../../../tests/planning/run_recorded_skill.py#L243)。

输入签名：

```python
def recorded_owner_observation_inputs(problem, scene_input, views, mask_rows, *, source):
```

输出分支的源码表达式：

```python
(frames, masks)
```

直接写入的状态或数组位置：`frames[reference]`、`frames[identity]`、`masks[owner, view]`。

直接调用：

```python
_problem_binding
scene_input.get
isinstance
reference.strip
ValueError
_pose_array
set
identity.strip
evidence.strip
len
Path
```

## `run_prepared`

[定义位置](../../../tests/planning/run_recorded_skill.py#L301)。

输入签名：

```python
def run_prepared(args, *, problem, provenance):
```

直接读取的对象属性：`problem.joint_names`、`problem.start_q`、`problem.start_dq`、`problem.phases`、`problem.reference_frame`、`problem.base_pose`、`problem.tool_link`、`problem.tool_transform`、`problem.conditions`、`problem.requirement_consumption`、`scene.observation_report`、`scene.background`、`scene.owners`、`scene.coverage`、`scene.reference_frame`、`scene.unknown_policy`、`scene.robot_sphere_materials`、`scene.robot_collision_bodies`、`scene.robot_geometry_source`、`scene.contacts`、`scene.holding_contacts`、`scene.initial_support_relations`、`scene.native_geometry_materials`、`scene.robot_owner_permissions`、`scene.numerical_position_tolerance_m`、`scene.numerical_rotation_tolerance_rad`、`scene.validate_problem`、`result.diagnostics`、`result.success`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`report['wallclock']`、`report['host_process_scope']`、`report['finger_targets_path']`、`report['execution_limits_path']`、`report['request_ledger_input']`、`report['robot_collision_contract']`、`report['robot_collision_contract_validated']`、`report['independent_robot_model']`、`robot['robot_cfg']['kinematics']['lock_joints']`、`report['resource_limits']`、`report['gripper']`、`prepared['sphere_materials']`、`material_binding['owner_materials'][prepared['owner_id']]`、`report['owner_material_geometry']`、`report['contact_binding']`、`material_binding['robot_sphere_materials']`、`material_binding['materials']`、`report['requirement_consumption']`、`report['owner_observations']`、`owner_masks[identity]`、`mask_report[identity]`、`observations[view['id']]`、`fine_regions[identity]`、`report['complete_carrier_geometry']`、`report['observation_build']`、`report['observation_partitions']`、`scene.observation_report`、`report['fixed_robot_installation']`、`report['unknown_policy']`、`report['entrance_geometry_diagnostic']`、`report['planning_called']`、`report['planner_wallclock']`、`report['elapsed_seconds']`。

直接调用：

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
recorded_owner_observation_inputs
scene_task_owner_ids
observation_resolutions
yaml.safe_load
Path(fixture['robot_config_path']).read_text
validate_fixture_collision_contract
build_robot_validator
robot_validator.describe
sum
len
robot['robot_cfg']['kinematics']['collision_spheres'].values
mujoco.MjModel.from_xml_path
native_model.joint
np.flatnonzero
finger_actuators.append
int
finger_trajectory_factory
float
native_model.actuator_ctrlrange[finger_actuators].copy
torch.cuda.is_available
RuntimeError
torch.set_num_threads
torch.cuda.set_per_process_memory_fraction
torch.cuda.get_device_properties
torch.cuda.reset_peak_memory_stats
torch.get_num_threads
DeviceCfg
position_controlled_hand
row.get
args.owner_geometry.read_text
bind_box_owner_materials
material_binding['native_geometry_materials'].update
load_contact_bindings
bind_fixture_contact_materials
build_material_contact_rules
any
bind_grasp_contact_rules
contact_inputs.get
bind_holding_contacts
report['contact_binding']['policies'].append
report['contact_binding'].update
args.contact_bindings.resolve
asdict
frame['T_output_owner'].tolist
supplied_masks[identity, view['id']].resolve
owner_frames.items
args.owner_geometry.resolve
(root / view['calibration_path']).read_text
np.array
Image.open(root / view['rgb_path']).convert
Image.open
np.load
classify_robot_observation
np.asarray
view_output.mkdir
np.save
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
body_geometries.setdefault(row['link'], []).append
body_geometries.setdefault
entrance_fk['sphere_indices_by_link'].items
args.fine_regions.read_text
row.items
create_box_owner
build_scene_from_rgbd
args.unknown_policy.read_text
save_observation_partitions
create_fixed_installation_owners
type(scene)
type
np.savez_compressed
scene.validate_problem
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
_write_diagnostic_json
```

## `run_prepared.mark_wallclock`

[定义位置](../../../tests/planning/run_recorded_skill.py#L335)。

输入签名：

```python
def run_prepared.mark_wallclock(name):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`report['wallclock']['events'][name]`。

直接调用：

```python
time.time_ns
time.monotonic
```

## `run`

[定义位置](../../../tests/planning/run_recorded_skill.py#L821)。

输入签名：

```python
def run(args):
```

输出分支的源码表达式：

```python
run_prepared(args, problem=problem, provenance=provenance)
```

直接调用：

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

[定义位置](../../../tests/planning/run_recorded_skill.py#L850)。

输入签名：

```python
def parse_args(argv=None):
```

输出分支的源码表达式：

```python
parser.parse_args(argv)
```

直接调用：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
```
