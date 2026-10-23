# `manipulation/planning_scene.py` 实现契约

代码基线：`774e121-plus-unified-workspace`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_array`

[定义位置](../../../manipulation/planning_scene.py#L22)。

输入签名：

```python
def _array(value, shape, label):
```

显式输出分支：

```python
result.copy()
```

读取的对象属性：

```python
np.asarray
result.shape
np.isfinite(result).all
np.isfinite
result.copy
```

写入目标：

```python
result
```

直接调用符号：

```python
np.asarray
np.isfinite(result).all
np.isfinite
ValueError
result.copy
```

## `_text`

[定义位置](../../../manipulation/planning_scene.py#L29)。

输入签名：

```python
def _text(value, label):
```

显式输出分支：

```python
value
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
```

## `_tensor`

[定义位置](../../../manipulation/planning_scene.py#L35)。

输入签名：

```python
def _tensor(value, reference):
```

显式输出分支：

```python
constant_tensor(value, reference)
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
constant_tensor
```

## `_interval`

[定义位置](../../../manipulation/planning_scene.py#L40)。

输入签名：

```python
def _interval(progress, interval):
```

显式输出分支：

```python
(progress >= interval[0]) & (progress <= interval[1])
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `_valid_interval`

[定义位置](../../../manipulation/planning_scene.py#L44)。

输入签名：

```python
def _valid_interval(value):
```

显式输出分支：

```python
tuple(result)
```

读取的对象属性：

无显式条目。

写入目标：

```python
result
```

直接调用符号：

```python
_array
ValueError
tuple
```

## `_world_points`

[定义位置](../../../manipulation/planning_scene.py#L51)。

输入签名：

```python
def _world_points(points, position, quaternion):
```

显式输出分支：

```python
quaternion_rotate(quaternion.unsqueeze(-2), points) + position.unsqueeze(-2)
```

读取的对象属性：

```python
quaternion.unsqueeze
position.unsqueeze
```

写入目标：

无显式条目。

直接调用符号：

```python
quaternion_rotate
quaternion.unsqueeze
position.unsqueeze
```

## `ObservedSpace`

[定义位置](../../../manipulation/planning_scene.py#L56)。

```python
class ObservedSpace():
```

类字段：

无显式条目。

实现说明：

> Immutable union of cells certified free by calibrated depth rays.
>
> ``known[z,y,x]`` identifies the closed voxel centred at ``origin + h*[x,y,z]``.
> This evidence is queried for moving scene entities. The robot view policy
> is recorded separately by OnlinePlanningScene.

## `ObservedSpace.__init__`

[定义位置](../../../manipulation/planning_scene.py#L63)。

输入签名：

```python
def __init__(self, known, *, origin, voxel_size, observation_ids, reference_frame):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
mask.dtype
np.bool_
mask.ndim
mask.shape
mask.size
np.isfinite
self.observation_ids
mask.copy
self._known.flags
self._known
```

写入目标：

```python
mask
self.origin
self.voxel_size
self.observation_ids
row
self.reference_frame
self._known
self._known.flags.writeable
```

直接调用符号：

```python
np.asarray
min
ValueError
_array
np.isfinite
float
tuple
_text
mask.copy
```

## `ObservedSpace.known_free`

[定义位置](../../../manipulation/planning_scene.py#L79)。

输入签名：

```python
def known_free(self):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
self._known.copy
self._known
result.flags
```

写入目标：

```python
result
result.flags.writeable
```

直接调用符号：

```python
self._known.copy
```

## `ObservedSpace.clone_for_rollout`

[定义位置](../../../manipulation/planning_scene.py#L84)。

输入签名：

```python
def clone_for_rollout(self):
```

实现说明：

> Return the fixed evidence shared by all candidate trajectories.

显式输出分支：

```python
self
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `Attachment`

[定义位置](../../../manipulation/planning_scene.py#L90)。

```python
class Attachment():
```

类字段：

```python
interval: tuple
position_tool_owner: object
quaternion_tool_owner: object
condition_id: str
release_condition_id: str | None
```

实现说明：

> Conditional rigid attachment over [start,end), never a Holding claim.
>
> The owner's external PosePath supplies its observed pose before attachment.
> After release the owner remains at the same rollout's actual release FK,
> preserving the optimization variables shared with subsequent constraints.
> An open command alone creates no release.

## `Attachment.__post_init__`

[定义位置](../../../manipulation/planning_scene.py#L105)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.interval
self.position_tool_owner
self.quaternion_tool_owner
np.isclose
np.linalg.norm
np.linalg
self.condition_id
self.release_condition_id
```

写入目标：

```python
self.interval
self.position_tool_owner
self.quaternion_tool_owner
```

直接调用符号：

```python
_valid_interval
_array
np.isclose
np.linalg.norm
ValueError
_text
```

## `Attachment.active`

[定义位置](../../../manipulation/planning_scene.py#L118)。

输入签名：

```python
def active(self, progress):
```

实现说明：

> Return membership in the tool assembly at the supplied progress.

显式输出分支：

```python
progress >= self.interval[0]
(progress >= self.interval[0]) & (progress < self.interval[1])
```

读取的对象属性：

```python
self.release_condition_id
self.interval
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `CollisionOwner`

[定义位置](../../../manipulation/planning_scene.py#L127)。

```python
class CollisionOwner():
```

类字段：

```python
owner_id: str
query: object
pose_path: PosePath
source: str
moving: bool = False
attachments: tuple[Attachment, ...] = ()
pose_interval: tuple = (0.0, 1.0)
```

实现说明：

> Observed local entity and its declared motion in the planning frame.

## `CollisionOwner.__post_init__`

[定义位置](../../../manipulation/planning_scene.py#L137)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.owner_id
self.source
self.query.owner_ids
self.query
self.pose_path
self.pose_path.parent
self.pose_interval
self.moving
self.attachments
relation.interval
self.attachments[index - 1].interval
self.attachments[index - 1].release_condition_id
```

写入目标：

```python
self.pose_interval
self.attachments
index
relation
self.moving
```

直接调用符号：

```python
_text
frozenset
ValueError
type
_valid_interval
TypeError
tuple
enumerate
isinstance
```

## `CollisionOwner.pose`

[定义位置](../../../manipulation/planning_scene.py#L158)。

输入签名：

```python
def pose(self, position, quaternion, progress, released_poses=()):
```

显式输出分支：

```python
(predicted_p, predicted_q)
```

读取的对象属性：

```python
self.predicted_pose
self.attachments
relation.position_tool_owner
relation.quaternion_tool_owner
relation.release_condition_id
relation.active
torch.where
relation.interval
```

写入目标：

```python
predicted_p
predicted_q
relation
release
held_p
held_q
held
released
```

直接调用符号：

```python
self.predicted_pose
len
ValueError
zip
quaternion_rotate
_tensor
quaternion_multiply
relation.active
torch.where
```

## `CollisionOwner.predicted_pose`

[定义位置](../../../manipulation/planning_scene.py#L181)。

输入签名：

```python
def predicted_pose(self, progress):
```

显式输出分支：

```python
self.pose_path.sample(((progress - start) / (end - start)).clamp(0, 1))
```

读取的对象属性：

```python
self.pose_interval
self.pose_path.sample
self.pose_path
((progress - start) / (end - start)).clamp
```

写入目标：

```python
start
end
```

直接调用符号：

```python
self.pose_path.sample
((progress - start) / (end - start)).clamp
```

## `ContactRule`

[定义位置](../../../manipulation/planning_scene.py#L189)。

```python
class ContactRule():
```

类字段：

```python
rule_id: str
actor: str
target_owner: str
interval: tuple
material_id: str
feature_points: object
feature_normal: object
patch_origin: object
patch_axes: object
patch_half_size: object
gap_bounds: tuple
normal_tolerance_rad: float
modeling_allowance_m: float
geometry_error_m: float
source: str
required: bool = False
feature_joint_translation: object = None
numerical_position_tolerance_m: float = 1e-06
numerical_angle_tolerance_rad: float = 1e-06
feature_kind: str = 'polygon'
robot_body: str | None = None
```

实现说明：

> Declared source feature and finite target patch in their entity frames.
>
> Task points express allowed contact geometry, not an occupied body. A
> robot feature identifies its registered body explicitly. Owner feature
> points are entity-local. Robot feature points use TCP coordinates at zero
> finger displacement; feature_joint_translation maps the named prismatic
> coordinates into that same TCP chart. Gap and normal limits retain their
> task meaning independently of separation margins.

## `ContactRule.__post_init__`

[定义位置](../../../manipulation/planning_scene.py#L221)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.interval
np.asarray
self.feature_points
points.ndim
points.shape
np.isfinite(points).all
np.isfinite
self.feature_kind
points.copy
self.feature_normal
np.isclose
np.linalg.norm
np.linalg
np.cross
np.stack
np.array
np.linalg.svd(points - points[0], full_matrices=True)[2][:2].T
np.linalg.svd
ConvexHull((points - points[0]) @ basis).equations
basis.T
self.patch_origin
self.patch_axes
self.patch_half_size
self.gap_bounds
np.allclose
self.patch_axes.T
np.eye
np.linalg.det
np.any
self.normal_tolerance_rad
math.pi
self.modeling_allowance_m
self.geometry_error_m
self.feature_joint_translation
mapping.ndim
mapping.shape
np.isfinite(mapping).all
mapping.copy
self.actor
self.robot_body
np.isfinite([self.numerical_position_tolerance_m, self.numerical_angle_tolerance_rad]).all
self.numerical_position_tolerance_m
self.numerical_angle_tolerance_rad
```

写入目标：

```python
name
self.interval
points
self.feature_points
self.feature_normal
delta
length
tangent
lateral
self._feature_tangent
self._feature_edge_normals
self._feature_edge_offsets
basis
edges
self.patch_origin
self.patch_axes
self.patch_half_size
self.gap_bounds
mapping
self.feature_joint_translation
```

直接调用符号：

```python
_text
getattr
_valid_interval
np.asarray
np.isfinite(points).all
np.isfinite
len
ValueError
points.copy
_array
np.isclose
np.linalg.norm
np.cross
np.stack
np.array
np.linalg.svd
ConvexHull
tuple
np.allclose
np.eye
np.linalg.det
np.any
np.isfinite(mapping).all
mapping.copy
np.isfinite([self.numerical_position_tolerance_m, self.numerical_angle_tolerance_rad]).all
min
```

## `grasp_contact_rule_ids`

[定义位置](../../../manipulation/planning_scene.py#L278)。

输入签名：

```python
def grasp_contact_rule_ids(condition, contacts):
```

实现说明：

> Validate the finite rules selected by one explicit grasp activation.
>
> ``contacts`` contains serialized ContactRule records. The condition names
> their existing identities; it cannot change material geometry or enlarge
> owner/interval scope. Ordinary material rules without a grasp activation
> keep their independently declared semantics.

显式输出分支：

```python
tuple(identities)
```

读取的对象属性：

```python
condition.get
value.strip
event.get
rows.get
rule.get
```

写入目标：

```python
identities
value
event
arrival
interval
rows
row
identity
rule
```

直接调用符号：

```python
condition.get
isinstance
any
value.strip
len
set
ValueError
_text
event_occurrence_key
event.get
type
_valid_interval
rows.get
rule.get
tuple
```

## `validate_grasp_contact_arrival`

[定义位置](../../../manipulation/planning_scene.py#L314)。

输入签名：

```python
def validate_grasp_contact_arrival(condition, requirement):
```

实现说明：

> Require the exact owner/event provenance of the frozen arrival target.
>
> Equal progress and a valid collection index are insufficient: another
> owner's pose at that instant must never authorize these material rules.

显式输出分支：

无显式条目。

读取的对象属性：

```python
requirement.source
source.get
```

写入目标：

```python
source
```

直接调用符号：

```python
isinstance
source.get
ValueError
```

## `GripperState`

[定义位置](../../../manipulation/planning_scene.py#L327)。

```python
class GripperState(NamedTuple):
```

类字段：

```python
coordinates: torch.Tensor
half: torch.Tensor
controller_residual: torch.Tensor
uncertain: torch.Tensor
```

实现说明：

> Coordinates/envelope and certainty shared by every collision consumer.
>
> ``uncertain`` identifies unmodelled future finger positions. A zero-width
> command reference is not evidence that this flag is false. ``half`` is the
> mechanical interval used only for conservative conditional diagnostics.

## `PrismaticGripper`

[定义位置](../../../manipulation/planning_scene.py#L340)。

```python
class PrismaticGripper():
```

类字段：

无显式条目。

实现说明：

> Bind native finger coordinates to a supported controller and geometry.
>
> Position tracking uses explicitly modeled solver coordinates. Timed
> actuator commands and optional nominal future geometry share the bound
> elapsed clock but remain separate objects. Binary control without a future
> model retains its mechanical envelope only as unresolved diagnostic
> evidence. A sampled width or command is never measured future contact.
>
> ``joint_bounds(progress)`` is the solver tracking contract. Independent
> ``finger_trajectory_factory`` supplies commands;
> ``nominal_finger_trajectory_factory(duration, progress_map, command_reference)``
> supplies declared future geometry from this binding's exact command.
> ``control_timestep_s`` is the explicit native device period shared by
> every command clock; it is not inferred from either trajectory factory.
> Without that geometry, only the actual initial fingers and invariant
> robot components contribute hard collision claims. Actual execution always
> checks the measured complete state. ``coupling_matrix`` and
> ``coupling_offset`` describe the native A q = b drive constraint.

## `PrismaticGripper.__init__`

[定义位置](../../../manipulation/planning_scene.py#L361)。

输入签名：

```python
def __init__(self, *, controller, joint_names, current_positions, joint_limits, joint_bounds, coupling_matrix, coupling_offset, commands, source, control_timestep_s, numerical_joint_tolerance=1e-05, finger_trajectory_factory=None, nominal_finger_trajectory_factory=None):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.isscalar
np.isfinite
self.joint_names
np.any
self.joint_limits
self.current_positions
np.asarray
coupling.ndim
coupling.shape
np.isfinite(coupling).all
coupling.copy
np.linalg.matrix_rank
np.linalg
np.linalg.svd
np.eye
np.allclose
self.coupling_offset
self.commands
event.get
np.nan
```

写入目标：

```python
self.controller
self.control_timestep_s
self.joint_names
name
count
self.current_positions
self.joint_limits
self.finger_trajectory_factory
self.finger_trajectory
self.nominal_finger_trajectory_factory
self.nominal_finger_trajectory
self.joint_bounds
self.numerical_joint_tolerance
coupling
self.coupling_matrix
self.coupling_offset
rank
self._drive_basis
self.commands
event
identities
self.source
```

直接调用符号：

```python
ValueError
isinstance
np.isscalar
np.isfinite
float
tuple
_text
len
set
_array
np.any
callable
TypeError
np.asarray
np.isfinite(coupling).all
coupling.copy
np.linalg.matrix_rank
np.linalg.svd
np.eye
np.allclose
deepcopy
event.get
event_occurrence_key
```

## `PrismaticGripper.bind_clock`

[定义位置](../../../manipulation/planning_scene.py#L438)。

输入签名：

```python
def bind_clock(self, duration, progress_map):
```

显式输出分支：

```python
None
```

读取的对象属性：

```python
self.controller
self.finger_trajectory_factory
trajectory.joint_names
self.joint_names
trajectory.duration
np.array_equal
trajectory.sample
np.array
self.current_positions
self.nominal_finger_trajectory_factory
self.finger_trajectory
nominal.joint_names
nominal.duration
nominal.sample
```

写入目标：

```python
trajectory
self.finger_trajectory
nominal
self.nominal_finger_trajectory
```

直接调用符号：

```python
self.finger_trajectory_factory
tuple
np.array_equal
trajectory.sample
np.array
ValueError
self.nominal_finger_trajectory_factory
isinstance
nominal.sample
```

## `PrismaticGripper.state`

[定义位置](../../../manipulation/planning_scene.py#L459)。

输入签名：

```python
def state(self, progress, joint_position, joint_names, *, trajectory_time=None, measured=False):
```

显式输出分支：

```python
GripperState(coordinates, torch.zeros_like(coordinates), violations, certain)
GripperState(coordinates, half, violations, uncertain)
```

读取的对象属性：

```python
progress.shape
self.joint_names
self.joint_limits
self.current_positions
torch.zeros_like
torch.bool
tuple(joint_names).index
torch.long
coordinates.shape
self.numerical_joint_tolerance
torch.relu(limits[:, 0] - coordinates - tolerance).sum
torch.relu
torch.relu(coordinates - limits[:, 1] - tolerance).sum
self.coupling_matrix
torch.relu((coordinates @ coupling.T - _tensor(self.coupling_offset, progress)).abs() - tolerance * coupling.abs().sum(-1)).sum
(coordinates @ coupling.T - _tensor(self.coupling_offset, progress)).abs
coupling.T
self.coupling_offset
coupling.abs().sum
coupling.abs
self.controller
limits[:, 0].expand
limits[:, 1].expand
trajectory_time.shape
torch.where
torch.ones_like
self.nominal_finger_trajectory
model.valid_mask_torch
model.sample_torch
self.joint_bounds
lower.shape
upper.shape
torch.relu(lower - upper - tolerance).sum
torch.relu(limits[:, 0] - lower - tolerance).sum
torch.relu(upper - limits[:, 1] - tolerance).sum
torch.relu((upper - lower).abs() - tolerance).sum
(upper - lower).abs
torch.relu((coordinates - middle).abs() - tolerance).sum
(coordinates - middle).abs
(progress == 0 if self.controller == 'position_tracking' else certain if trajectory_time is None else trajectory_time == 0).to
progress.dtype
torch.relu(lower - initial_positions - tolerance).sum
torch.relu(initial_positions - upper - tolerance).sum
torch.relu((coordinates @ coupling.T - _tensor(self.coupling_offset, progress)).abs() - coupling_tolerance).sum
modeled.to
```

写入目标：

```python
expected
limits
initial_positions
certain
columns
name
coordinates
tolerance
violations
coupling
modeled
lower
upper
initial
uncertain
model
_
middle
half
indices
coupling_tolerance
```

直接调用符号：

```python
len
_tensor
torch.zeros_like
set
ValueError
tuple(joint_names).index
tuple
constant_tensor
torch.relu(limits[:, 0] - coordinates - tolerance).sum
torch.relu
torch.relu(coordinates - limits[:, 1] - tolerance).sum
torch.relu((coordinates @ coupling.T - _tensor(self.coupling_offset, progress)).abs() - tolerance * coupling.abs().sum(-1)).sum
(coordinates @ coupling.T - _tensor(self.coupling_offset, progress)).abs
coupling.abs().sum
coupling.abs
GripperState
limits[:, 0].expand
limits[:, 1].expand
torch.where
torch.ones_like
model.valid_mask_torch
model.sample_torch
self.joint_bounds
torch.relu(lower - upper - tolerance).sum
torch.relu(limits[:, 0] - lower - tolerance).sum
torch.relu(upper - limits[:, 1] - tolerance).sum
torch.relu((upper - lower).abs() - tolerance).sum
(upper - lower).abs
torch.relu((coordinates - middle).abs() - tolerance).sum
(coordinates - middle).abs
(progress == 0 if self.controller == 'position_tracking' else certain if trajectory_time is None else trajectory_time == 0).to
torch.relu(lower - initial_positions - tolerance).sum
torch.relu(initial_positions - upper - tolerance).sum
torch.relu((coordinates @ coupling.T - _tensor(self.coupling_offset, progress)).abs() - coupling_tolerance).sum
modeled.to
```
