# `tests/planning/run_recorded_skill.py` 实现契约

代码基线：`96ee2dc-working-tree`。
职责和数据语义见[组件说明](../11-device-control-and-evaluation.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `load_contact_bindings`

[定义位置](../../../tests/planning/run_recorded_skill.py#L39)。

输入签名：

```python
def load_contact_bindings(path):
```

实现说明：

> Read finite contact features and their source task declarations.

显式输出分支：

```python
inputs
```

读取的对象属性：

```python
json.loads
Path(path).read_text
inputs['source'].strip
```

写入目标：

```python
inputs
required
optional
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
```

## `run_prepared`

[定义位置](../../../tests/planning/run_recorded_skill.py#L55)。

输入签名：

```python
def run_prepared(args, *, problem, provenance, backend_factory=MotionBackend):
```

实现说明：

> Run the existing scene/solver pipeline on the supplied continuous path.
>
> Problem construction belongs to the caller. The actual feedback is checked
> again here, so an offline reference cannot replace current robot state.
> This function never reconstructs or resamples the supplied reference.
> backend_factory consumes robot, robot_validator and budget, and returns
> the resident backend whose plan method owns the chosen solver lifecycle.

显式输出分支：

```python
result
```

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
args.inherited_initial_state
args.inherited_initial_state.read_text
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
owner
view
path
task_owners
owner_resolutions
background_masks
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
report['contact_binding']
contact_inputs
row
report['requirement_consumption']
report['owner_observations']
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
read_evidence_json
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
args.inherited_initial_state.read_text
args.unknown_policy.read_text
save_observation_partitions
np.savez_compressed
scene.validate_problem
entrance_fk['geometry_poses'].values
PlanningBudget
backend_factory
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

[定义位置](../../../tests/planning/run_recorded_skill.py#L91)。

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

[定义位置](../../../tests/planning/run_recorded_skill.py#L457)。

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

[定义位置](../../../tests/planning/run_recorded_skill.py#L486)。

输入签名：

```python
def parse_args(argv=None):
```

实现说明：

> Share the existing resource/scene CLI without a second option schema.

显式输出分支：

```python
parser.parse_args(argv)
```

读取的对象属性：

```python
argparse.ArgumentParser
parser.add_argument
PlanningBudget.trajectory_iterations
PlanningBudget.planning_seconds
parser.parse_args
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
