# `manipulation/program.py` 实现契约

代码基线：本次实现提交同步更新正文和源码；前置代码检查点为 `759d89b`。实际实现提交号由紧随的文档引用提交回填。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `ReferenceGenerationError`

[定义位置](../../../manipulation/program.py#L22)。

```python
class ReferenceGenerationError(ValueError):
```

类字段：

无显式条目。

实现说明：

> A source-profile fit failed after validating the independent goals.

## `_mechanism_entry_contract`

[定义位置](../../../manipulation/program.py#L26)。

输入签名：

```python
def _mechanism_entry_contract(model_spec, initial, *, require_lift=False):
```

实现说明：

> Validate immutable actual entry without selecting a goal-set member.
>
> A full-frame target's centre need not lie on the mechanism or within its
> limits: another pose in the declared task allowed set may be feasible.
> The typed goal and physical mechanism remain separate solver obligations.
> Native scalar goals additionally admit exact interval-intersection checks
> where their signed coordinate meaning is explicitly supplied.

显式输出分支：

```python
(None, None)
(model, limits)
```

读取的对象属性：

```python
np.any
```

写入目标：

```python
model
limits
start
```

直接调用符号：

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

实现说明：

> Retain the bound source as a seed, with no feasibility claim about it.

显式输出分支：

```python
(output, {'model': serialized, 'reference_generation': {'status': 'failed', 'reason': str(error), 'seed': 'bound source reference; not a task feasibility result'}})
```

读取的对象属性：

```python
reference.copy
model._coordinates_unchecked
np.asarray(coordinates, float).copy
np.asarray
model.poses
model.coordinates
model.to_dict
```

写入目标：

```python
output
serialized
model.coordinates
```

直接调用符号：

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

实现说明：

> Resolve an original frame ID without silently treating it as an index.

显式输出分支：

```python
int(found[0])
```

读取的对象属性：

```python
np.bool_
np.integer
np.flatnonzero
```

写入目标：

```python
found
```

直接调用符号：

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

显式输出分支：

```python
value
```

读取的对象属性：

```python
np.bool_
np.isfinite
```

写入目标：

```python
value
```

直接调用符号：

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

实现说明：

> Validate manipulation commands, which need not contain a new close.

显式输出分支：

```python
deepcopy(events)
```

读取的对象属性：

```python
event.get
seen.add
```

写入目标：

```python
seen
previous
event
name
index
```

直接调用符号：

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

实现说明：

> Bind the existing FK model to evidenced current mechanism parameters.
>
> The caller supplies its reference pose, signed coordinate basis, and (for
> a hinge) axis point, all in the result reference frame. A source fit or an
> action name alone does not establish a new instance's mechanism. Optional
> coordinate limits have shape (dimension,2) and are checked on every node.
> ``initial_coordinates`` with ``initial_coordinates_source`` binds a fresh
> actual lifted angular branch. A turn-sensitive request without this input
> requires exactly one angular member within its physical coordinate limits.

显式输出分支：

```python
(model, limits)
```

读取的对象属性：

```python
value.get
np.eye
np.asarray
point.shape
np.isfinite(point).all
np.isfinite
np.zeros
limits.shape
model.dimension
np.isfinite(limits).all
np.any
```

写入目标：

```python
kind
dimensions
pose
basis_pose
basis_pose[:3, :3]
basis
point
ptol
rtol
model
limits
```

直接调用符号：

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

显式输出分支：

```python
coordinates
```

读取的对象属性：

```python
model.coordinates_of
model.poses
```

写入目标：

```python
coordinates
p
r
```

直接调用符号：

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

实现说明：

> Preserve an evidenced lifted actual coordinate after checking its FK.
>
> Without an explicit observation, pose alone fixes only a principal angle.
> An original entry-delta target has its own separate immutable origin and
> must not be inferred from this fresh actual measurement during a rebind.

显式输出分支：

```python
coordinates
coordinates.astype(float, copy=True)
```

读取的对象属性：

```python
model.kind
model_spec.get
np.asarray
np.ceil
np.pi
np.floor
model.diagnostics
coordinates.shape
model.dimension
coordinates.dtype.kind
coordinates.dtype
np.isfinite(coordinates).all
np.isfinite
model.poses
coordinates.astype
```

写入目标：

```python
coordinates
angular
limits
low
high
first
last
coordinates[angular]
model.diagnostics['initial_coordinate_convention']
source
p
r
model.diagnostics['initial_coordinates_source']
```

直接调用符号：

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

实现说明：

> Transfer in a physical function frame, preserving every internal event.
>
> Free intervals reuse the existing SE(3) transfer kernel on a material
> functional frame, not on an arbitrary body origin. Constrained intervals
> stay in one caller-bound current FK model; we never fit a new axis after
> each event. A model projection is allowed only for the soft reference.
> This fitting attempt uses nominal event centres on the model. Failure to
> fit one such centre is a reference-generation failure; the separately
> retained typed goal allowed sets do not acquire exact pose equalities.
>
> Native joint events additionally supply their complete coordinate reference
> and indexed coordinate goals. These are not recovered from their FK poses:
> doing so would discard explicitly recorded revolute winding numbers.
>
> Native prismatic intervals between hard events retain the source's scalar
> progress profile while adapting its displacement to the new endpoints.
> The unanchored suffix after the final event retains source metric increments;
> this does not rescale the final controlled object motion before release.
>
> Endpoint corrections use the recorded timestamp fraction within each
> interval. Source sample density cannot redefine that fraction. Native
> prismatic progress scaling remains a source-coordinate operation, not a
> time correction; its pauses and reversals must retain their own meaning.

显式输出分支：

```python
(output, diagnostics)
```

读取的对象属性：

```python
np.arange
np.allclose
np.empty_like
np.asarray(coordinate_reference, float).copy
np.asarray
model._coordinates_unchecked
coordinates.shape
model.dimension
np.isfinite(coordinates).all
np.isfinite
model.poses
p.max
r.max
model.kind
coordinates.copy
coordinates[0].copy
coordinate_goals[begin].copy
np.pi
np.round
coordinate_goals[end].copy
np.isfinite([source_delta, target_delta]).all
np.finfo(float).eps
np.finfo
np.array_equal
diagnostics['coordinate_transfer']['intervals'].append
np.any
model.to_dict
```

写入目标：

```python
_
times
knots
index
goal
output
diagnostics
native
frame
function_reference
position
begin
initial
end
result
output[begin:end + 1]
diagnostics['model']
model
limits
coordinates
p
r
diagnostics['reference_projection']
angular
values
diagnostics['coordinate_transfer']
initial_q
initial_q[angular]
part
goal_q
goal_q[angular]
source_delta
target_delta
roundoff
scale
mode
part[0]
part[-1]
values[begin:end + 1]
model.coordinates
diagnostics['limit_status']
```

直接调用符号：

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
def interval_progress(begin, end):
```

显式输出分支：

```python
np.zeros(1) if begin == end else (part - part[0]) / (part[-1] - part[0])
```

读取的对象属性：

```python
np.zeros
```

写入目标：

```python
part
```

直接调用符号：

```python
np.zeros
```

## `_coordinate_vector`

[定义位置](../../../manipulation/program.py#L350)。

输入签名：

```python
def _coordinate_vector(value, name):
```

实现说明：

> One prismatic distance in metres or one unwrapped revolute angle.

显式输出分支：

```python
vector.astype(float, copy=True)
```

读取的对象属性：

```python
np.asarray
vector.shape
vector.dtype.kind
vector.dtype
np.isfinite(vector).all
np.isfinite
vector.astype
```

写入目标：

```python
vector
```

直接调用符号：

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

实现说明：

> Evidence may be a description or an existing structured source record.

显式输出分支：

```python
deepcopy(value)
```

读取的对象属性：

```python
value.strip
```

写入目标：

无显式条目。

直接调用符号：

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

实现说明：

> Bind measured source coordinates and explicit current coordinate goals.
>
> Source ``motion_coordinates`` cover every original observation; only their
> controlled prefix through release generates a reference. Their evidence is in
> ``motion_coordinates_source``. Every event declares ``joint_coordinate``,
> ``coordinate_value`` at its original source frame and a positive
> ``coordinate_tolerance``. The current binding declares
> ``current_coordinates`` in the caller's current model, matching owner IDs
> and its own evidence source. Its ``coordinate_mode`` is ``absolute`` by
> default; ``entry_delta`` instead adds these values to explicitly supplied
> original-request ``entry_coordinates`` with ``entry_source`` evidence.
> The increment origin never follows a continuation's new actual state.
> Distance units are metres; angle units are
> unwrapped radians. The caller must bind the same signed coordinate sense
> on source and current mechanisms; fitting the source does not establish a
> current joint axis or a new instance's limits.
>
> The observed initial pose and any evidenced lifted coordinate define q0. The
> soft reference is q_ref=q_demo-q_demo[0]+q0. Native prismatic intervals
> between hard events then scale their source progress to the explicit
> current endpoint displacement; same-endpoint zero-net loops remain intact.
> After the final event, metric source increments remain unchanged.
> For revolute motion, a pose cannot reveal accumulated revolutions. Supply
> ``motion_model.initial_coordinates`` and its ``initial_coordinates_source``
> or physical limits selecting exactly one equivalent angle. Merely setting
> a reference pose does not observe the actual turn count. Original target
> origins and fresh current lifted coordinates remain independent inputs.
> Functional goals are evaluated directly by current FK, without a reference-owner
> functional frame or source/current body correspondence. Source hand motion
> and body poses are not inputs to this explicitly scalar route.

显式输出分支：

```python
(objects, checks, diagnostics)
```

读取的对象属性：

```python
task.get
scene.get
spec.get
demo.get
np.asarray
raw.ndim
raw.shape
raw.dtype.kind
raw.dtype
np.isfinite(raw).all
np.isfinite
raw.astype
model.poses
event.get
np.max
np.abs
bindings.get
binding.get
unique_goals.setdefault
target.copy
recorded.append
recorded[-1].update
names.add
unique_goals.items
initial_coordinates.copy
checks.append
event.items
checks[-1].update
actual[index].copy
model.kind
diagnostics.update
source_coordinates[0].copy
model.diagnostics.get
model.diagnostics
```

写入目标：

```python
spec
model
limits
source_evidence
raw
source_coordinates
initial_coordinates
coordinates
reference
events
bindings
names
previous
unique_goals
recorded
event
name
index
evidence
tolerance
source_value
binding
binding_source
coordinate_mode
raw_target
entry
entry_source
target
preferred
same_scope
lower
upper
unique_goals[index]
goals
value
objects
diagnostics
actual
checks
goal
error
p
r
key
```

直接调用符号：

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

实现说明：

> Keep the existing optional observed-corridor check on both goal paths.

显式输出分支：

```python
None
```

读取的对象属性：

```python
scene.get
```

写入目标：

```python
begin
corridor
clearance
pose
motion_checks['entry_corridor']
```

直接调用符号：

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

显式输出分支：

```python
output
```

读取的对象属性：

```python
demo.get
scene.get
event.get
current.get
binding.get
np.allclose
np.asarray
function_body[:3, :3].T
face.get
output.append
names.add
```

写入目标：

```python
events
current
on_relation
reference_pose
carrier_mapping
output
previous
names
event
name
index
binding
mapping
source_binding
preferred
function_frame
face
surface
function_body
body_points
contact_points
reference_failure
grounded
relation
relation['reference_generation']
relation['goal_pose']
relation['reference_binding']
relation['function_frame']
relation['constraint_kind']
key
```

直接调用符号：

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
deepcopy
names.add
```

## `_material_pose_requirement`

[定义位置](../../../manipulation/program.py#L674)。

输入签名：

```python
def _material_pose_requirement(event):
```

实现说明：

> Bind one grounded functional event without promoting its reference path.

显式输出分支：

```python
PoseEventRequirement(0.0, material, event['goal_pose'] @ material, event['position_tolerance_m'], event['rotation_tolerance_rad'], event['source'], yaw_axis=event.get('yaw_axis'), yaw_tolerance_rad=event.get('yaw_tolerance_rad'))
```

读取的对象属性：

```python
event.get
```

写入目标：

```python
material
```

直接调用符号：

```python
PoseEventRequirement
event.get
```

## `_object_reference`

[定义位置](../../../manipulation/program.py#L684)。

输入签名：

```python
def _object_reference(demo, scene, frames, last_held, task, initial):
```

实现说明：

> Shared function/region consumer for object and complete EEF outputs.

显式输出分支：

```python
result
(objects, checks, motion_checks)
```

读取的对象属性：

```python
task.get
scene.get
np.bool_
np.number
np.isfinite
demo.get
event.get
np.linalg.norm
np.linalg
np.sin
requirement.errors_numpy
requirement.component_tolerances.items
requirement.component_tolerances
unique_goals.setdefault
initial.copy
unique_goals.items
model_spec.get
reference[:last_held + 1].copy
np.asarray
pose[:3, :3].T
np.min
surface.origin
surface.normal
components.items
checks.append
```

写入目标：

```python
margin
name
declared_events
event
result
source
events
first
reference
unique_goals
event_index
index
goal
requirement
previous_event
position_error
rotation_error
translation_bound
item
rotation_bound
errors
bound
goal_pairs
motion
model_spec
function_frame
bound_model
_
fitting_goals
objects
motion_checks
solved
checks
binding
pose
face
surface
error
error['body_checks']
error['freedom']
body_world
minimum_height
error['body_checks']['minimum_signed_height_m']
satisfies
error['checked_member_satisfies_support']
components
error['allowed_set_errors']
value
inside
error['containment']
checks[-1]['reference_generation']
```

直接调用符号：

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
_material_pose_requirement
_errors
sum
np.linalg.norm
np.sin
radius
requirement.errors_numpy
requirement.component_tolerances.items
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
components.items
check_inside
_interior
checks.append
```

## `_object_reference.radius`

[定义位置](../../../manipulation/program.py#L732)。

输入签名：

```python
def radius(item):
```

显式输出分支：

```python
min(np.pi, item['rotation_tolerance_rad'] + yaw)
```

读取的对象属性：

```python
item.get
np.pi
```

写入目标：

```python
yaw
```

直接调用符号：

```python
item.get
min
```

## `ground_object_program`

[定义位置](../../../manipulation/program.py#L859)。

输入签名：

```python
def ground_object_program(program, scene):
```

实现说明：

> Return only function-constrained body motion when no attachment is known.
>
> Human tool demonstrations may establish object motion before robot grasp
> binding exists. This geometry-only entry preserves their function events
> without fabricating an EEF trajectory. The complete skill consumer requires
> a declared attachment model separately. It uses the same object numerical core.

显式输出分支：

```python
result
```

读取的对象属性：

```python
program.get
scene.get
demo.get
```

写入目标：

```python
key
demo
frames
times
last
initial
objects
checks
motion_checks
result
result['body_points']
```

直接调用符号：

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

[定义位置](../../../manipulation/program.py#L897)。

输入签名：

```python
def ground_program(program, scene):
```

实现说明：

> Generate one grasp/manipulation geometry reference from a compiled demo.
>
> ``program`` contains ``skill_kind``, ``demo`` and (for manipulation) task
> ``relation``/``motion``. Every functional event names its source frame ID,
> carrier-local material function frame and evidence source. Full-frame
> relations additionally supply source/current reference functional frames,
> a current carrier frame and metric event tolerances. Their explicit
> ``scene.reference_pose`` embeds the current reference owner's local frame
> in ``scene.reference_frame`` (the common output frame).
>
> # BEGIN explicit current grasp goal, separate from source-reference motion.
> Grasp scenes require ``grasp_goal`` with current carrier-local
> ``T_carrier_eef``, positive task ``position_tolerance_m`` and
> ``rotation_tolerance_rad`` (strictly below pi), plus a nonempty ``source``.
> Region correspondence still transforms the ordinary demonstration;
> changing that reference cannot silently rewrite the independent grasp goal.
> # END no device/numerical tolerance or last reference sample supplies this input.
>
> Articulation events may instead declare ``constraint_kind=joint_coordinate``.
> Source ``motion_coordinates`` / ``motion_coordinates_source`` preserve the
> full observed signed scalar sequence; event ``coordinate_value`` and
> ``coordinate_tolerance`` describe its source function state. Current
> bindings provide ``current_coordinates`` with source/owner evidence in the
> already bound ``scene.motion_model``. These are absolute unless explicitly
> marked ``coordinate_mode=entry_delta`` with preserved ``entry_coordinates``
> and ``entry_source`` from the original request entrance.
> No reference pose, source body-pose sequence, surrogate region frame or
> source/current body correspondence is needed for this scalar route.
>
> An ``on`` relation instead supplies finite ``scene.support`` in the output
> frame and, per event, ``carrier_mapping`` (current-carrier-from-source-
> carrier), owner IDs, evidence source and ``on_geometry`` contact samples,
> signed normal and tolerances. It requires no current reference pose or
> complete reference functional frame. Source motion is aligned with the
> current initial carrier pose before the support constrains the event goal;
> its tangential position and yaw remain preferences, not observations.
> Every event for one carrier must use the same explicit carrier mapping.
>
> # BEGIN nominal attachment is a model, never simulator success evidence.
> Manipulation additionally requires ``scene.holding`` with the declared
> ``T_carrier_eef`` and its model source. ``initial_pose`` and robot
> ``eef_pose`` must use that same model; this equality checks coordinates,
> not whether the physical object is actually held. No verified flag or
> simulator object pose is required. ``release_frame`` is an observed source
> phase boundary, independent of the opening command and function event. The
> object and EEF references stop at this boundary. Complete source times and
> commands remain separate evidence; released-hand motion is not generated.
> Only the current tool identity and calibrated frame are required for held
> manipulation, because the declared attachment supplies its tool relation.
> # END physical holding success belongs to independent result assessment.
>
> Grasp source frames only initialize the current single-skill plan. Its
> bound terminal grasp relation is a goal; no approach frame is a waypoint.

显式输出分支：

```python
result
{'status': 'candidate', 'skill_kind': kind, 'output_type': 'interaction_trajectory', 'reference_frame': frame_name, 'carrier_id': scene['carrier_id'], 'reference_id': scene['reference_id'], 'task': task, 'object_poses': objects, 'object_frame_indices': frames[:last_held + 1], 'eef_poses': eef, 'frame_indices': frames[:last_held + 1], 'timestamps': times[:last_held + 1], 'source_frame_indices': frames, 'source_timestamps': times, 'source_gripper_events': commands, 'gripper_events': controlled_commands, 'release_frame': release, 'function_checks': checks, 'motion_checks': motion_checks, 'tool_id': scene['tool_id'], 'tool_frame': scene['tool_frame'], 'holding_source': holding['source'], 'released_object_state': 'requires_observation' if release is not None else 'still_attached_reference', 'pose_convention': 'column-vector reference_from_body/eef; metres; 4x4 SE(3)', 'path_scope': 'geometry reference; robot IK, collision, forces and timing remain external'}
```

读取的对象属性：

```python
scene.get
result.update
demo.get
event.get
holding.get
np.allclose
task.get
```

写入目标：

```python
kind
demo
frame_name
result
key
frames
times
commands
release
last_held
event
field
holding
attachment
initial
eef_initial
task
objects
checks
motion_checks
eef
controlled_commands
```

直接调用符号：

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
