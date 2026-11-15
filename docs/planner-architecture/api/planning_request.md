# `manipulation/planning_request.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../14-grounding-planning-portfolio.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `reexpress_contact_bindings`

[定义位置](../../../manipulation/planning_request.py#L23)。

输入签名：

```python
def reexpress_contact_bindings(contact_bindings, material_contacts, owner_bindings, *, source):
```

实现说明：

> Bind source task patches to current charts and official robot features.
>
> ``owner_bindings`` maps each source contact owner/actor to its current id
> and ``T_current_source`` chart transform. Robot material rows contain the
> official current TCP features. Other material rows retain the task actor's
> source chart and use its own transform, independently of the target patch.
> ``build_material_contact_rules`` assembles the existing contact contract. Patch
> dimensions, gap/normal allowances, intervals and event activation remain
> the original source declarations. Raw source records remain caller-owned.

显式输出分支：

```python
result
```

读取的对象属性：

```python
result.pop
np.asarray
transform[:3, :3].T
np.einsum
materials['owner_materials'].setdefault
row.pop
```

写入目标：

```python
result
feature_keys
materials
row
feature
key
feature['robot_body']
materials['materials'][row['material_id']]
actor
transform
feature['feature_points']
feature['feature_normal']
feature['feature_joint_translation']
materials['owner_materials'].setdefault(actor['owner_id'], {})[row['material_id']]
row['actor']
binding
relocation
row['target_owner']
row['patch_origin']
row['patch_axes']
row['source']
result['contacts']
rule
result['source']
```

直接调用符号：

```python
deepcopy
result.pop
dict
np.asarray
np.einsum
materials['owner_materials'].setdefault
row.pop
asdict
build_material_contact_rules
```

## `bind_task_inputs`

[定义位置](../../../manipulation/planning_request.py#L76)。

输入签名：

```python
def bind_task_inputs(hypothesis, request):
```

实现说明：

> Express original typed task geometry in this hypothesis's current chart.
>
> F places the source functional region. The current owner's calibrated
> chart and source owner-from-region relation yield the source-to-current
> local mapping. The independent E target supplies the existing grasp goal
> contract; complete H_g placement belongs only to ``reference_phase``.
> Manipulation consumes the current typed scene's attachment, mechanism,
> finite support and functional bindings through the original compiler.

显式输出分支：

```python
dict(program=deepcopy(program), scene=scene, contact_bindings=contacts)
```

读取的对象属性：

```python
np.asarray
np.linalg.inv
np.linalg
binding.get
scene.update
goal.update
scene.pop
```

写入目标：

```python
original
binding
program
source_scene
feedback
chart
function
mapping
relocation
owners
scene
role
scene[role + '_id']
owners[source_scene[role + '_id']]
goal
scene['reference_pose']
event
current
owner_role
current[owner_role + '_id']
frame_key
current[frame_key]
scene['region_binding_evidence']
contacts
```

直接调用符号：

```python
np.asarray
np.linalg.inv
deepcopy
binding.get
scene.update
dict
goal.update
scene.pop
reexpress_contact_bindings
```

## `bind`

[定义位置](../../../manipulation/planning_request.py#L135)。

输入签名：

```python
def bind(hypothesis, request):
```

实现说明：

> Bind original task relations and one complete nominal reference.
>
> ``task`` contains the task compiler's typed source ``program`` and
> ``scene``; ``task_binding`` supplies current entity charts and resolved
> robot materials. The compiler retains task goals, events, release,
> attachments and process relations. ``reference_phase`` replaces only its
> nominal guidance path and preserves the compiler's task metadata.

显式输出分支：

```python
(problem, provenance)
```

读取的对象属性：

```python
feedback['joint_names'].index
np.asarray
problem.phases
original_phase.metadata
phase.metadata
phase.reference.progress
phase.reference
```

写入目标：

```python
feedback
task
names
columns
name
problem
key
original_phase
source
phase
phase.metadata
provenance
```

直接调用符号：

```python
require_feedback_identity
bind_task_inputs
tuple
feedback['joint_names'].index
build_planning_problem
dict
deepcopy
np.asarray
reference_phase
replace
bind_pose_objectives
```

## `plan`

[定义位置](../../../manipulation/planning_request.py#L181)。

输入签名：

```python
def plan(hypothesis, request):
```

实现说明：

> Prepare one observed scene, call the direct backend once, and close it.
>
> Capture feedback places robot shapes for RGB-D self filtering. Current
> entrance feedback independently defines q/dq and the physical reference
> boundary. Both records retain their original identities and clocks. The
> supplied total host/device limits cover the explicit portfolio process
> root, including sibling workers; native scene admission uses that scope.

显式输出分支：

```python
result
```

读取的对象属性：

```python
output.mkdir
time.monotonic
request.get
os.getpid
FixedFingerTargets.from_mapping
ExecutionLimits.from_mapping
observations_in.items
view['owner_masks'].items
owner.owner_id
yaml.safe_load
Path(fixture['robot_config_path']).read_text
problem.reference_frame
problem.base_pose
problem.tool_link
problem.tool_transform
validator.describe
torch.set_num_threads
torch.cuda.set_per_process_memory_fraction
torch.cuda
torch.cuda.get_device_properties(0).total_memory
torch.cuda.get_device_properties
torch.cuda.reset_peak_memory_stats
row.get
problem.conditions
row.rule_id
problem.requirement_consumption
validator.checker.query
validator.checker
np.asarray
folder.mkdir
np.save
next(iter(observations.values()))['depth'].shape
observations.values
torch.tensor
scene.observation_report
np.savez_compressed
scene.coverage.known_free
scene.coverage
scene.coverage.origin
scene.coverage.voxel_size
scene.coverage.observation_ids
scene.validate_problem
execution_limits.describe
backend.close
report.update
result.success
result.diagnostics.get
result.diagnostics
type(error).__name__
(output / 'exception.txt').write_text
traceback.format_exc
```

写入目标：

```python
output
report
started
report['host_process_scope']
problem
provenance
feedback
fixture
task
resources
mapper
budget
finger_targets
execution_limits
continuation
rebind
measured
observed_owners
path
observations_in
views
identity
mask_rows
view
owner
mask
owner_frames
masks
task_owners
report['owner_voxel_sizes_m']
robot
validator
report['independent_robot_model']
robot['robot_cfg']['kinematics']['lock_joints']
name
factory
hand
report['gripper']
tool_members
contact_input
contacts
row
report['contact_binding']
report['requirement_consumption']
capture
capture_fk
shapes
observations
filtered
folder
observations[identity]
observations[identity]['background_mask']
height
width
config
scene
report['observation_build']
report['observation_partitions']
report['planning_called']
backend
result
report['elapsed_seconds']
```

直接调用符号：

```python
Path
output.mkdir
dict
time.monotonic
configure_host_process_root
request.get
os.getpid
bind
bind_task_inputs
PlanningBudget
FixedFingerTargets.from_mapping
ExecutionLimits.from_mapping
continuation_execution_inputs
zip
mark
tuple
load_observed_owner
observations_in.items
view['owner_masks'].items
owner_observation_inputs
str
scene_task_owner_ids
observation_resolutions
yaml.safe_load
Path(fixture['robot_config_path']).read_text
build_robot_validator
validator.describe
finger_trajectory_factory
float
torch.set_num_threads
torch.cuda.set_per_process_memory_fraction
torch.cuda.get_device_properties
torch.cuda.reset_peak_memory_stats
position_controlled_hand
DeviceCfg
tool_geometry_names
ContactRule
any
row.get
bind_grasp_contact_rules
validator.checker.query
robot_self_filter_shapes
classify_depth
np.asarray
folder.mkdir
np.save
_write_diagnostic_json
next
iter
observations.values
MapperCfg
torch.tensor
build_scene_from_rgbd
save_observation_partitions
np.savez_compressed
scene.validate_problem
plain
DirectPreferenceBackend
ReferenceObjective
plan_and_record
execution_limits.describe
load_joint_reference
backend.close
report.update
result.diagnostics.get
type
(output / 'exception.txt').write_text
traceback.format_exc
```

## `plan.mark`

[定义位置](../../../manipulation/planning_request.py#L213)。

输入签名：

```python
def mark(name):
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
dict
time.time_ns
time.monotonic
```
