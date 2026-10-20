# `manipulation/planning_scene.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_array`

[定义位置](../../../manipulation/planning_scene.py#L49)。

输入签名：

```python
def _array(value, shape, label):
```

直接读取的对象属性：`result.shape`、`result.copy`。

输出分支的源码表达式：

```python
result.copy()
```

直接调用：

```python
np.asarray
np.isfinite(result).all
np.isfinite
ValueError
result.copy
```

## `_text`

[定义位置](../../../manipulation/planning_scene.py#L56)。

输入签名：

```python
def _text(value, label):
```

输出分支的源码表达式：

```python
value
```

直接调用：

```python
isinstance
value.strip
ValueError
```

## `_tensor`

[定义位置](../../../manipulation/planning_scene.py#L62)。

输入签名：

```python
def _tensor(value, reference):
```

输出分支的源码表达式：

```python
constant_tensor(value, reference)
```

直接调用：

```python
constant_tensor
```

## `_interval`

[定义位置](../../../manipulation/planning_scene.py#L68)。

输入签名：

```python
def _interval(progress, interval):
```

输出分支的源码表达式：

```python
(progress >= interval[0]) & (progress <= interval[1])
```

## `_valid_interval`

[定义位置](../../../manipulation/planning_scene.py#L72)。

输入签名：

```python
def _valid_interval(value):
```

输出分支的源码表达式：

```python
tuple(result)
```

直接调用：

```python
_array
ValueError
tuple
```

## `_world_points`

[定义位置](../../../manipulation/planning_scene.py#L79)。

输入签名：

```python
def _world_points(points, position, quaternion):
```

输出分支的源码表达式：

```python
quaternion_rotate(quaternion.unsqueeze(-2), points) + position.unsqueeze(-2)
```

直接调用：

```python
quaternion_rotate
quaternion.unsqueeze
position.unsqueeze
```

## `_local_spheres`

[定义位置](../../../manipulation/planning_scene.py#L84)。

输入签名：

```python
def _local_spheres(spheres, position, quaternion):
```

输出分支的源码表达式：

```python
torch.cat((points, spheres[..., 3:4]), dim=-1)
```

直接调用：

```python
quaternion_rotate
quaternion_conjugate(quaternion).unsqueeze
quaternion_conjugate
position.unsqueeze
torch.cat
```

## `_world_spheres`

[定义位置](../../../manipulation/planning_scene.py#L90)。

输入签名：

```python
def _world_spheres(spheres, position, quaternion):
```

输出分支的源码表达式：

```python
torch.cat((points, radii), dim=-1)
```

直接调用：

```python
_tensor
_world_points
local[..., 3:4].expand
torch.cat
```

## `_self_pair_chunk_size`

[定义位置](../../../manipulation/planning_scene.py#L102)。

输入签名：

```python
def _self_pair_chunk_size(states, pairs):
```

输出分支的源码表达式：

```python
min(pairs, max(1, _SELF_PAIR_STATE_ELEMENTS // states)) if pairs else 0
```

直接调用：

```python
min
max
```

## `_coverage_plane_chunk_size`

[定义位置](../../../manipulation/planning_scene.py#L117)。

输入签名：

```python
def _coverage_plane_chunk_size(states, points, planes):
```

输出分支的源码表达式：

```python
min(planes, max(1, _COVERAGE_POINT_PLANE_ELEMENTS // (states * points)))
```

直接调用：

```python
any
type
ValueError
min
max
```

## `_coverage_halfspace_max`

[定义位置](../../../manipulation/planning_scene.py#L128)。

输入签名：

```python
def _coverage_halfspace_max(points, halfspaces):
```

输出分支的源码表达式：

```python
signed.max(dim=-1).values
```

直接调用：

```python
torch.linalg.vector_norm
signed.max
```

## `_pair_penetration`

[定义位置](../../../manipulation/planning_scene.py#L142)。

输入签名：

```python
def _pair_penetration(spheres, pairs):
```

输出分支的源码表达式：

```python
torch.relu(first[..., 3] + second[..., 3] - distance).sum(-1)
```

直接调用：

```python
spheres.index_select
torch.linalg.vector_norm
torch.relu(first[..., 3] + second[..., 3] - distance).sum
torch.relu
```

## `_self_pair_sum`

[定义位置](../../../manipulation/planning_scene.py#L155)。

输入签名：

```python
def _self_pair_sum(spheres, pair_indices):
```

输出分支的源码表达式：

```python
total
```

直接调用：

```python
spheres.new_zeros
len
constant_tensor
_self_pair_chunk_size
math.prod
torch.is_grad_enabled
range
checkpoint
_pair_penetration
```

## `CuroboSphereQuery`

[定义位置](../../../manipulation/planning_scene.py#L181)。

```python
class CuroboSphereQuery:
    ...
    ...
```

该类型的直接方法：`__init__`、`clone_for_rollout`、`__call__`。

## `CuroboSphereQuery.__init__`

[定义位置](../../../manipulation/planning_scene.py#L204)。

输入签名：

```python
def CuroboSphereQuery.__init__(self, scene_collision, *, owner_ids, source, max_query_spheres=3000000, max_buffers=64, max_cache_bytes=64 * 1024 * 1024):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.scene_collision`、`self.owner_ids`、`self.source`、`self.max_query_spheres`、`self.max_buffers`、`self.max_cache_bytes`、`self._cache_bytes`、`self._buffers`。

直接调用：

```python
callable
getattr
TypeError
frozenset
_text
type
ValueError
```

## `CuroboSphereQuery.clone_for_rollout`

[定义位置](../../../manipulation/planning_scene.py#L222)。

输入签名：

```python
def CuroboSphereQuery.clone_for_rollout(self):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result._buffers`、`result._cache_bytes`。

直接调用：

```python
copy
```

## `CuroboSphereQuery.__call__`

[定义位置](../../../manipulation/planning_scene.py#L228)。

输入签名：

```python
def CuroboSphereQuery.__call__(self, spheres, *, key):
```

直接读取的对象属性：`self.max_query_spheres`、`self._buffers`、`self._cache_bytes`、`self.max_cache_bytes`、`self.max_buffers`、`self.scene_collision`。

输出分支的源码表达式：

```python
value.clone()
```

直接写入的状态或数组位置：`self._cache_bytes`、`self._buffers[slot]`。

直接调用：

```python
ValueError
math.prod
tuple
str
self._buffers.get
len
config.is_same_torch_device
CollisionBuffer.from_shape
spheres.new_ones
spheres.new_zeros
torch.autograd.graph.saved_tensors_hooks
self.scene_collision.get_sphere_distance_raw
spheres.contiguous
value.clone
```

## `CoverageField`

[定义位置](../../../manipulation/planning_scene.py#L273)。

```python
class CoverageField:
    ...
    ...
```

该类型的直接方法：`__init__`、`clearance`、`violation`。

## `CoverageField.__init__`

[定义位置](../../../manipulation/planning_scene.py#L281)。

输入签名：

```python
def CoverageField.__init__(self, allowed, *, origin, voxel_size):
```

直接读取的对象属性：`self.origin`、`self.voxel_size`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.origin`、`self.voxel_size`、`self._distance_squared`、`self._device_distance_squared`、`self._roundoff_scale_m`、`self._shape_xyz`。

直接调用：

```python
np.asarray
min
ValueError
_array
np.isfinite
float
distance_transform_edt
np.pad
np.square(distance).astype
np.square
np.maximum
np.nextafter
torch.from_numpy
np.array
np.abs(self.origin).sum
np.abs
np.abs(upper).sum
```

## `CoverageField.clearance`

[定义位置](../../../manipulation/planning_scene.py#L303)。

输入签名：

```python
def CoverageField.clearance(self, centre):
```

直接读取的对象属性：`self.origin`、`self.voxel_size`、`self._shape_xyz`、`self._device_distance_squared`、`self._distance_squared`、`self._roundoff_scale_m`。

输出分支的源码表达式：

```python
clearance
```

直接写入的状态或数组位置：`self._device_distance_squared[key]`。

直接调用：

```python
_tensor
self._distance_squared.to
F.grid_sample
grid.reshape
coordinates.floor
(fraction * (1 - fraction)).sum
math.sqrt
values.reshape
squared_bound.clamp_min(half_diagonal ** 2).sqrt
squared_bound.clamp_min
torch.minimum(coordinates, extent - coordinates).amin
torch.minimum
torch.finfo
(clearance - roundoff).clamp_min
```

## `CoverageField.violation`

[定义位置](../../../manipulation/planning_scene.py#L340)。

输入签名：

```python
def CoverageField.violation(self, spheres):
```

直接读取的对象属性：`self.clearance`。

输出分支的源码表达式：

```python
torch.where(radius > 0, torch.relu(radius - clearance), torch.zeros_like(radius))
```

直接调用：

```python
self.clearance
torch.where
torch.relu
torch.zeros_like
```

## `ObservedSpace`

[定义位置](../../../manipulation/planning_scene.py#L347)。

```python
class ObservedSpace(CoverageField):
    ...
    ...
```

该类型的直接方法：`__init__`、`known_free`、`clone_for_rollout`。

## `ObservedSpace.__init__`

[定义位置](../../../manipulation/planning_scene.py#L364)。

输入签名：

```python
def ObservedSpace.__init__(self, known, *, origin, voxel_size, observation_ids, reference_frame, occupancy_evidence=None, known_free=None):
```

直接读取的对象属性：`self.observation_ids`、`record.get`、`self._known_free`、`self._known`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.observation_ids`、`self.reference_frame`、`self.occupancy_evidence`、`self._known_free`、`self._known_free.flags.writeable`、`self._known`、`self._known.flags.writeable`。

直接调用：

```python
np.asarray
min
ValueError
tuple
_text
isinstance
set
len
record.get
owner_ids.add
deepcopy
np.any
free.copy
super().__init__
super
mask.copy
```

## `ObservedSpace.known_free`

[定义位置](../../../manipulation/planning_scene.py#L420)。

输入签名：

```python
@property
def ObservedSpace.known_free(self):
```

直接读取的对象属性：`self._known_free`、`result.flags`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result.flags.writeable`。

直接调用：

```python
ValueError
self._known_free.copy
```

## `ObservedSpace.clone_for_rollout`

[定义位置](../../../manipulation/planning_scene.py#L429)。

输入签名：

```python
def ObservedSpace.clone_for_rollout(self):
```

直接读取的对象属性：`self.origin`、`self._shape_xyz`、`self.occupancy_evidence`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result.origin`、`result._shape_xyz`、`result.occupancy_evidence`。

直接调用：

```python
copy
self.origin.copy
self._shape_xyz.copy
deepcopy
```

## `Attachment`

[定义位置](../../../manipulation/planning_scene.py#L447)。

```python
class Attachment:
    interval: tuple
    position_tool_owner: object
    quaternion_tool_owner: object
    condition_id: str
    release_condition_id: str | None
```

该类型的直接方法：`__post_init__`。

## `Attachment.__post_init__`

[定义位置](../../../manipulation/planning_scene.py#L462)。

输入签名：

```python
def Attachment.__post_init__(self):
```

直接读取的对象属性：`self.interval`、`self.position_tool_owner`、`self.quaternion_tool_owner`、`self.condition_id`、`self.release_condition_id`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.interval`、`self.position_tool_owner`、`self.quaternion_tool_owner`。

直接调用：

```python
_valid_interval
_array
np.isclose
np.linalg.norm
ValueError
_text
```

## `CollisionOwner`

[定义位置](../../../manipulation/planning_scene.py#L476)。

```python
class CollisionOwner:
    owner_id: str
    query: object
    pose_path: object
    source: str
    geometry_complete: bool
    local_spheres: object = None
    sphere_materials: tuple = ()
    moving: bool = False
    attachments: tuple[Attachment, ...] = ()
    sphere_cover_error_m: float = 0.0
    pose_interval: tuple = (0.0, 1.0)
    robot_excluded_spheres: tuple = ()
    robot_exclusion_source: str | None = None
    robot_exclusion_geometry_source: str | None = None
    native_geometry_names: tuple = ()
```

该类型的直接方法：`__post_init__`、`pose`、`predicted_pose`。

## `CollisionOwner.__post_init__`

[定义位置](../../../manipulation/planning_scene.py#L505)。

输入签名：

```python
def CollisionOwner.__post_init__(self):
```

直接读取的对象属性：`self.owner_id`、`self.source`、`self.query`、`self.pose_path`、`self.pose_interval`、`self.geometry_complete`、`self.moving`、`self.attachments`、`self.native_geometry_names`、`self.robot_excluded_spheres`、`self.robot_exclusion_source`、`self.robot_exclusion_geometry_source`、`self.local_spheres`、`self.sphere_materials`、`self.sphere_cover_error_m`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.pose_interval`、`self.attachments`、`self.native_geometry_names`、`self.moving`、`self.robot_excluded_spheres`、`self.local_spheres`。

直接调用：

```python
_text
frozenset
ValueError
callable
getattr
TypeError
type
_valid_interval
tuple
len
set
enumerate
isinstance
any
np.allclose
np.asarray
np.isfinite(value).all
np.isfinite
np.any
value.copy
```

## `CollisionOwner.pose`

[定义位置](../../../manipulation/planning_scene.py#L571)。

输入签名：

```python
def CollisionOwner.pose(self, position, quaternion, progress, released_poses=()):
```

直接读取的对象属性：`self.predicted_pose`、`self.attachments`。

输出分支的源码表达式：

```python
(predicted_p, predicted_q)
```

直接调用：

```python
self.predicted_pose
len
ValueError
zip
quaternion_rotate
_tensor
quaternion_multiply
torch.where
```

## `CollisionOwner.predicted_pose`

[定义位置](../../../manipulation/planning_scene.py#L593)。

输入签名：

```python
def CollisionOwner.predicted_pose(self, progress):
```

直接读取的对象属性：`self.pose_interval`、`self.pose_path`。

输出分支的源码表达式：

```python
self.pose_path.sample(((progress - start) / (end - start)).clamp(0, 1))
```

直接调用：

```python
self.pose_path.sample
((progress - start) / (end - start)).clamp
```

## `_native_box_edge_geometry`

[定义位置](../../../manipulation/planning_scene.py#L599)。

输入签名：

```python
def _native_box_edge_geometry(rule, material):
```

输出分支的源码表达式：

```python
{'body_coordinates': coordinates, 'lateral_depth_ratio': slope, 'vertices_actor': vertices, 'adjacent_normals_actor': adjacent, 'normal_cone_coefficients': coefficients}
```

直接写入的状态或数组位置：`endpoints[:, axis]`、`endpoints[:, fixed]`。

直接调用：

```python
_pose_array
np.finfo
np.allclose
np.eye
abs
np.linalg.det
ValueError
_array
np.any
type
isinstance
len
any
range
np.zeros
np.asarray
max
float
np.abs(endpoints).max
np.abs
half.max
np.linalg.norm
np.min
np.isclose
np.cross
np.nextafter
ratios.max
list
product
np.column_stack
```

## `ContactRule`

[定义位置](../../../manipulation/planning_scene.py#L659)。

```python
class ContactRule:
    rule_id: str
    actor: str
    target_owner: str
    interval: tuple
    sphere_indices: tuple
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
    minimum_thickness_m: float
    source: str
    required: bool = False
    feature_joint_translation: object = None
    numerical_position_tolerance_m: float = 1e-06
    numerical_angle_tolerance_rad: float = 1e-06
    feature_kind: str = 'polygon'
    online_material: object = None
```

该类型的直接方法：`__post_init__`。

## `ContactRule.__post_init__`

[定义位置](../../../manipulation/planning_scene.py#L701)。

输入签名：

```python
def ContactRule.__post_init__(self):
```

直接读取的对象属性：`self.interval`、`self.sphere_indices`、`self.feature_points`、`self.feature_kind`、`self.feature_normal`、`self.patch_origin`、`self.patch_axes`、`self.patch_half_size`、`self.gap_bounds`、`self.normal_tolerance_rad`、`self.modeling_allowance_m`、`self.geometry_error_m`、`self.minimum_thickness_m`、`self.feature_joint_translation`、`self.numerical_position_tolerance_m`、`self.numerical_angle_tolerance_rad`、`self.online_material`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.interval`、`self.sphere_indices`、`self.feature_points`、`self.feature_normal`、`self._feature_tangent`、`self._feature_edge_normals`、`self._feature_edge_offsets`、`self.patch_origin`、`self.patch_axes`、`self.patch_half_size`、`self.gap_bounds`、`self.feature_joint_translation`、`self._native_edge_geometry`、`self.online_material`。

直接调用：

```python
_text
getattr
_valid_interval
any
type
ValueError
tuple
np.asarray
len
np.isfinite(points).all
np.isfinite
points.copy
_array
np.isclose
np.linalg.norm
np.max
np.abs
np.finfo
max
float
np.abs(points).max
abs
np.cross
np.stack
np.array
np.linalg.svd
ConvexHull
np.allclose
np.eye
np.linalg.det
np.any
np.isfinite(value).all
value.copy
min
isinstance
value.get
set
np.isscalar
_native_box_edge_geometry
deepcopy
```

## `RobotOwnerContactPermission`

[定义位置](../../../manipulation/planning_scene.py#L805)。

```python
class RobotOwnerContactPermission:
    rule_id: str
    target_owner: str
    native_geometry_names: tuple
    sphere_indices: tuple
    interval: tuple
    arrival_goal_id: tuple
    activation_event: dict
    source: str
    holding_condition_id: str | None = None
```

该类型的直接方法：`__post_init__`。

## `RobotOwnerContactPermission.__post_init__`

[定义位置](../../../manipulation/planning_scene.py#L828)。

输入签名：

```python
def RobotOwnerContactPermission.__post_init__(self):
```

直接读取的对象属性：`self.native_geometry_names`、`self.sphere_indices`、`self.interval`、`self.arrival_goal_id`、`self.holding_condition_id`、`self.activation_event`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self.native_geometry_names`、`self.sphere_indices`、`self.interval`、`self.arrival_goal_id`、`self.activation_event`。

直接调用：

```python
_text
getattr
isinstance
ValueError
tuple
len
set
any
type
_valid_interval
event_occurrence_key
deepcopy
```

## `grasp_contact_rule_ids`

[定义位置](../../../manipulation/planning_scene.py#L865)。

输入签名：

```python
def grasp_contact_rule_ids(condition, contacts):
```

输出分支的源码表达式：

```python
tuple(identities)
```

直接调用：

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

[定义位置](../../../manipulation/planning_scene.py#L901)。

输入签名：

```python
def validate_grasp_contact_arrival(condition, requirement):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
isinstance
source.get
ValueError
```

## `GripperState`

[定义位置](../../../manipulation/planning_scene.py#L916)。

```python
class GripperState(NamedTuple):
    coordinates: torch.Tensor
    half: torch.Tensor
    controller_residual: torch.Tensor
    uncertain: torch.Tensor
```

## `PrismaticGripper`

[定义位置](../../../manipulation/planning_scene.py#L930)。

```python
class PrismaticGripper:
    ...
    ...
```

该类型的直接方法：`__init__`、`bind_clock`、`state`、`spheres`。

## `PrismaticGripper.__init__`

[定义位置](../../../manipulation/planning_scene.py#L953)。

输入签名：

```python
def PrismaticGripper.__init__(self, *, controller, joint_names, current_positions, joint_limits, joint_bounds, coupling_matrix, coupling_offset, sphere_indices, local_spheres, sphere_joint_translation, self_collision_pairs, commands, source, control_timestep_s, numerical_joint_tolerance=1e-05, finger_trajectory_factory=None, nominal_finger_trajectory_factory=None, self_collision_geometry=()):
```

直接读取的对象属性：`self.joint_names`、`self.joint_limits`、`self.current_positions`、`self.coupling_offset`、`self.sphere_indices`、`self.local_spheres`、`self.self_collision_pairs`、`self.sphere_joint_translation`、`self._self_pair_indices`、`self.commands`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.controller`、`self.control_timestep_s`、`self.joint_names`、`self.current_positions`、`self.joint_limits`、`self.finger_trajectory_factory`、`self.finger_trajectory`、`self.nominal_finger_trajectory_factory`、`self.nominal_finger_trajectory`、`self.joint_bounds`、`self.numerical_joint_tolerance`、`self.coupling_matrix`、`self.coupling_offset`、`self._drive_basis`、`self.sphere_indices`、`self.local_spheres`、`self.sphere_joint_translation`、`self.self_collision_pairs`、`self._self_pair_indices`、`self.self_collision_model`、`self.commands`、`self.source`。

直接调用：

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
any
type
np.asarray(self.self_collision_pairs, dtype=np.int64).reshape
PreparedPrismaticSelfCollision
deepcopy
event.get
event_occurrence_key
```

## `PrismaticGripper.bind_clock`

[定义位置](../../../manipulation/planning_scene.py#L1056)。

输入签名：

```python
def PrismaticGripper.bind_clock(self, duration, progress_map):
```

直接读取的对象属性：`self.controller`、`self.finger_trajectory_factory`、`trajectory.joint_names`、`self.joint_names`、`trajectory.duration`、`trajectory.sample`、`self.current_positions`、`self.nominal_finger_trajectory_factory`、`self.finger_trajectory`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self.finger_trajectory`、`self.nominal_finger_trajectory`。

直接调用：

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

[定义位置](../../../manipulation/planning_scene.py#L1077)。

输入签名：

```python
def PrismaticGripper.state(self, progress, joint_position, joint_names, *, trajectory_time=None, measured=False):
```

直接读取的对象属性：`self.joint_names`、`self.joint_limits`、`self.current_positions`、`self.numerical_joint_tolerance`、`self.coupling_matrix`、`self.coupling_offset`、`self.controller`、`self.nominal_finger_trajectory`、`self.joint_bounds`。

输出分支的源码表达式：

```python
GripperState(coordinates, torch.zeros_like(coordinates), violations, certain)
GripperState(coordinates, half, violations, uncertain)
```

直接调用：

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

## `PrismaticGripper.spheres`

[定义位置](../../../manipulation/planning_scene.py#L1170)。

输入签名：

```python
def PrismaticGripper.spheres(self, robot_spheres, position, quaternion, coordinates, half):
```

直接读取的对象属性：`self.local_spheres`、`self.sphere_joint_translation`、`self.sphere_indices`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[..., constant_tensor(list(self.sphere_indices), result, dtype=torch.long), :]`。

直接调用：

```python
_tensor
torch.einsum
mapping.abs
torch.linalg.vector_norm
torch.cat
_world_points
robot_spheres.clone
constant_tensor
list
```

## `_validate_contact_and_gripper_inputs`

[定义位置](../../../manipulation/planning_scene.py#L1186)。

输入签名：

```python
def _validate_contact_and_gripper_inputs(owner_materials, robot_materials, contacts, gripper, has_gripper):
```

输出分支的源码表达式：

```python
contacts
```

直接调用：

```python
type
isinstance
ValueError
list
max
len
tuple
any
gripper.sphere_indices.index
np.any
set
min
```

## `_source_world_binding_key`

[定义位置](../../../manipulation/planning_scene.py#L1239)。

输入签名：

```python
def _source_world_binding_key(scene):
```

直接读取的对象属性：`scene.gripper`、`scene.robot_validator`、`scene._source_scene_checker`、`scene._self_collision_owner_ids`。

输出分支的源码表达式：

```python
(id(scene.robot_validator), id(scene._source_scene_checker), tuple(sorted(scene._self_collision_owner_ids)), hand_key)
```

直接调用：

```python
tuple
typed
type
id
sorted
```

## `_source_world_binding_key.typed`

[定义位置](../../../manipulation/planning_scene.py#L1247)。

输入签名：

```python
def _source_world_binding_key.typed(value):
```

输出分支的源码表达式：

```python
(value.dtype.str, value.shape, value.tobytes())
```

直接调用：

```python
np.asarray
value.tobytes
```

## `_SourceWorldStateReuse`

[定义位置](../../../manipulation/planning_scene.py#L1264)。

```python
class _SourceWorldStateReuse:
    ...
    ...
```

该类型的直接方法：`__init__`、`binding_for`、`begin_candidate`、`finish_candidate`、`lookup`、`record`、`lookup_batch`、`record_batch`、`close`。

## `_SourceWorldStateReuse.__init__`

[定义位置](../../../manipulation/planning_scene.py#L1274)。

输入签名：

```python
def _SourceWorldStateReuse.__init__(self, ordinary, dense):
```

直接读取的对象属性：`self._binding`、`self.storage_nbytes`、`self.capacity_nbytes`、`self.close`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._scenes`、`self._binding`、`self._candidate`、`self._reports`、`self._batches`、`self._state_nbytes`、`self.storage_nbytes`、`self.capacity_nbytes`、`self.stored_states`、`self.reused_states`、`self.capacity_misses`、`self.stored_batches`、`self.reused_batches`、`self.batch_capacity_misses`。

直接调用：

```python
_source_world_binding_key
ValueError
object
_owned_record_bytes
self.close
```

## `_SourceWorldStateReuse.binding_for`

[定义位置](../../../manipulation/planning_scene.py#L1297)。

输入签名：

```python
def _SourceWorldStateReuse.binding_for(self, scene):
```

直接读取的对象属性：`self._scenes`、`self._binding`、`self.close`。

输出分支的源码表达式：

```python
self._binding
None
```

直接调用：

```python
any
all
_source_world_binding_key
self.close
```

## `_SourceWorldStateReuse.begin_candidate`

[定义位置](../../../manipulation/planning_scene.py#L1306)。

输入签名：

```python
def _SourceWorldStateReuse.begin_candidate(self):
```

直接读取的对象属性：`self.finish_candidate`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._candidate`。

直接调用：

```python
self.finish_candidate
object
```

## `_SourceWorldStateReuse.finish_candidate`

[定义位置](../../../manipulation/planning_scene.py#L1310)。

输入签名：

```python
def _SourceWorldStateReuse.finish_candidate(self):
```

直接读取的对象属性：`self._reports`、`self._state_nbytes`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.storage_nbytes`、`self._state_nbytes`。

直接调用：

```python
self._reports.clear
```

## `_SourceWorldStateReuse.lookup`

[定义位置](../../../manipulation/planning_scene.py#L1316)。

输入签名：

```python
def _SourceWorldStateReuse.lookup(self, scene, key):
```

直接读取的对象属性：`self._scenes`、`self._reports`。

输出分支的源码表达式：

```python
deepcopy(self._reports[key])
None
```

直接写入的状态或数组位置：`self.reused_states`。

直接调用：

```python
deepcopy
```

## `_SourceWorldStateReuse.record`

[定义位置](../../../manipulation/planning_scene.py#L1322)。

输入签名：

```python
def _SourceWorldStateReuse.record(self, scene, key, report):
```

直接读取的对象属性：`self._scenes`、`self._reports`、`self.storage_nbytes`、`self.capacity_nbytes`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self.capacity_misses`、`self._reports[key]`、`self.storage_nbytes`、`self._state_nbytes`、`self.stored_states`。

直接调用：

```python
_owned_record_bytes
deepcopy
```

## `_SourceWorldStateReuse.lookup_batch`

[定义位置](../../../manipulation/planning_scene.py#L1337)。

输入签名：

```python
def _SourceWorldStateReuse.lookup_batch(self, scene, key):
```

直接读取的对象属性：`self._batches`、`self._scenes`。

输出分支的源码表达式：

```python
(stored[1].copy(), deepcopy(stored[2]))
None
```

直接写入的状态或数组位置：`self.reused_batches`。

直接调用：

```python
self._batches.get
id
stored[1].copy
deepcopy
```

## `_SourceWorldStateReuse.record_batch`

[定义位置](../../../manipulation/planning_scene.py#L1344)。

输入签名：

```python
def _SourceWorldStateReuse.record_batch(self, scene, key, residuals, diagnostics):
```

直接读取的对象属性：`self._scenes`、`self._batches`、`self.storage_nbytes`、`self.capacity_nbytes`、`self._state_nbytes`、`self.finish_candidate`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self.storage_nbytes`、`self.batch_capacity_misses`、`self._batches[slot]`、`self.stored_batches`。

直接调用：

```python
any
id
self._batches.pop
_owned_record_bytes
self.finish_candidate
residuals.copy
deepcopy
```

## `_SourceWorldStateReuse.close`

[定义位置](../../../manipulation/planning_scene.py#L1372)。

输入签名：

```python
def _SourceWorldStateReuse.close(self):
```

直接读取的对象属性：`self._scenes`、`self._reports`、`self._batches`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`scene._source_world_reuse`、`self._state_nbytes`、`self._scenes`、`self._binding`、`self._candidate`、`self.storage_nbytes`。

直接调用：

```python
getattr
self._reports.clear
self._batches.clear
```

## `PlanningScene`

[定义位置](../../../manipulation/planning_scene.py#L1385)。

```python
class PlanningScene:
    ...
    ...
```

该类型的直接方法：`native_world_collision_components`、`__init__`、`validate_robot_geometry`、`validate_problem`、`bind_robot_validator`、`bind_source_scene`、`begin_source_world_reuse`、`end_source_world_reuse`、`check_robot_self_collision`、`check_world_geometry`、`clone_for_rollout`、`background_refinement_witnesses`、`refine_background`、`robot_coverage_guidance`、`full_robot_positions`、`check_robot_coverage`、`estimated_workspace_allocation`、`estimated_workspace_bytes`、`project_residual`、`_release_pose`、`with_release_history`、`_permission`、`robot_self_collision_guidance`、`components`、`source_components`、`evaluate_source`、`evaluate`、`optimizer_preferences`、`to_execution_binding`、`describe`。

## `PlanningScene.native_world_collision_components`

[定义位置](../../../manipulation/planning_scene.py#L1402)。

输入签名：

```python
@property
def PlanningScene.native_world_collision_components(self):
```

输出分支的源码表达式：

```python
frozenset()
```

直接调用：

```python
frozenset
```

## `PlanningScene.__init__`

[定义位置](../../../manipulation/planning_scene.py#L1412)。

输入签名：

```python
def PlanningScene.__init__(self, *, background, owners, coverage, reference_frame, robot_sphere_materials, robot_geometry_source, contacts=(), gripper=None, has_gripper=True, robot_coverage=None, numerical_position_tolerance_m=0.0001, numerical_rotation_tolerance_rad=0.001, native_geometry_materials=None, robot_owner_permissions=(), robot_collision_bodies=()):
```

直接读取的对象属性：`self.owners`、`self._coverage_geometry`、`self.robot_sphere_materials`、`self.robot_geometry_source`、`self._occupied_owner_ids`、`self.required_robot_links`、`self.robot_owner_permissions`、`self.contacts`、`self._contact_cells`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.background`、`self.robot_validator`、`self._self_collision_owner_ids`、`self._source_scene_checker`、`self._max_host_rss_bytes`、`self._source_world_reuse`、`self._last_source_world_execution`、`self.native_geometry_materials`、`self.owners`、`self.reference_frame`、`self.coverage`、`self.robot_coverage`、`self._coverage_geometry`、`self._coverage_samples`、`self._device_coverage_grids`、`self._has_free_cells`、`self._free_distance`、`self._free_mask`、`self.observation_report`、`self.robot_sphere_materials`、`self.robot_geometry_source`、`self.robot_collision_bodies`、`self._occupied_owner_ids`、`self.required_robot_links`、`self.numerical_position_tolerance_m`、`self.numerical_rotation_tolerance_rad`、`self.contacts`、`self.gripper`、`self.grasp_contact_activations`、`self.robot_owner_permissions`、`self._contact_cells`、`self._contact_cells[owner.owner_id]`、`self._fixed_release_poses`、`self.required_progress_anchors`。

直接调用：

```python
frozenset
dict
tuple
len
set
ValueError
isinstance
_text
robot_coverage.validate_observation_grid
robot_coverage.guidance_geometry
np.asarray
samples.append
np.concatenate
np.linspace
min
vertices[faces[np.linspace(0, len(faces) - 1, min(16, len(faces)), dtype=int)]].mean
bool
free.any
distance_transform_edt
np.pad
torch.from_numpy
distance.astype
free.astype
any
validate_body_groups
max
robot_evidence.get
sorted
np.isfinite
float
_validate_contact_and_gripper_inputs
TypeError
LayeredContactCellGeometry
query_geometry
make_contact_geometry
anchors.extend
```

## `PlanningScene.validate_robot_geometry`

[定义位置](../../../manipulation/planning_scene.py#L1559)。

输入签名：

```python
def PlanningScene.validate_robot_geometry(self, current_spheres_output, link_poses_output):
```

直接读取的对象属性：`self.coverage`、`self.robot_sphere_materials`、`self.required_robot_links`。

输出分支的源码表达式：

```python
None
```

直接调用：

```python
_array
len
np.allclose
ValueError
isinstance
set
evidence['link_poses_output'].items
_pose_array
```

## `PlanningScene.validate_problem`

[定义位置](../../../manipulation/planning_scene.py#L1586)。

输入签名：

```python
def PlanningScene.validate_problem(self, problem, *, goal_identities=None):
```

直接读取的对象属性：`self.reference_frame`、`problem.reference_frame`、`self.robot_coverage`、`self._coverage_geometry`、`problem.base_pose`、`problem.joint_names`、`problem.start_q`、`self.gripper`、`self.owners`、`self.background`、`problem.conditions`、`self.contacts`、`problem.pose_events`、`problem.events`、`self.robot_owner_permissions`、`self.coverage`、`problem.phases`、`problem.phase_count`、`problem.couplings`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`actual[:3, 3]`、`actual[:3, :3]`、`initial_poses[carrier]`、`actual_rows[tuple((row[columns[name]] if name in columns else 0.0 for name in hand.joint_names)) + (target,)]`、`self.grasp_contact_activations`。

直接调用：

```python
ValueError
np.array_equal
dict
zip
any
set
len
row.get
asdict
list
range
goal_identities.get
type
enumerate
grasp_contact_rule_ids
finite_ids.intersection
event_occurrence_key
pose_ids.index
validate_grasp_contact_arrival
finite_ids.update
finite_activations.append
deepcopy
matched.add
conditions.get
condition.get
_inverse
_pose_array
np.eye
Rotation.from_quat(np.asarray(relation.quaternion_tool_owner)[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
np.asarray
np.allclose
tuple
self.coverage.occupancy_evidence['owners'].items
by_id[identity].predicted_pose
torch.tensor
same_pose
matrix
p.numpy
q.numpy
names.index
meta.get
owner.predicted_pose
isinstance
np.unique
((keys - begin) / (end - begin)).clip
path.sample_numpy
np.stack
np.interp
meta.keys
_valid_interval
used_attachments.add
declared.get
Counter
native_current.update
problem.joint_names.index
```

## `PlanningScene.validate_problem.matrix`

[定义位置](../../../manipulation/planning_scene.py#L1682)。

输入签名：

```python
def PlanningScene.validate_problem.matrix(position, quaternion):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[:3, 3]`、`result[:3, :3]`。

直接调用：

```python
np.eye
np.asarray
Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
```

## `PlanningScene.validate_problem.same_pose`

[定义位置](../../../manipulation/planning_scene.py#L1688)。

输入签名：

```python
def PlanningScene.validate_problem.same_pose(first, second, label):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
np.allclose
ValueError
```

## `PlanningScene.bind_robot_validator`

[定义位置](../../../manipulation/planning_scene.py#L1846)。

输入签名：

```python
def PlanningScene.bind_robot_validator(self, validator, problem):
```

直接读取的对象属性：`self.robot_validator`、`self.gripper`、`self.reference_frame`、`self.owners`、`problem.start_q`、`problem.joint_names`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`local[:3, :3]`、`local[:3, 3]`、`self.robot_validator`、`self._self_collision_owner_ids`。

直接调用：

```python
isinstance
TypeError
ValueError
validator.validate_planning_binding
set
checker.geometry_depending_on
validator._planning_positions
checker.query
query_geometry
len
geometry.get
np.array_equal
np.broadcast_to
np.asarray
np.eye
Rotation.from_quat(pose7[[4, 5, 6, 3]]).as_matrix
Rotation.from_quat
np.allclose
checked.add
used.add
frozenset
```

## `PlanningScene.bind_source_scene`

[定义位置](../../../manipulation/planning_scene.py#L1900)。

输入签名：

```python
def PlanningScene.bind_source_scene(self, *, max_host_rss_bytes=None, previous_checker=None):
```

直接读取的对象属性：`self.end_source_world_reuse`、`self.robot_validator`、`self._source_scene_checker`、`self._max_host_rss_bytes`、`self.to_execution_binding`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._last_source_world_execution`、`checker._data`、`self._source_scene_checker`、`self._max_host_rss_bytes`。

直接调用：

```python
self.end_source_world_reuse
ValueError
copy
checker.model.createData
checker.bind_scene
self.to_execution_binding
```

## `PlanningScene.begin_source_world_reuse`

[定义位置](../../../manipulation/planning_scene.py#L1930)。

输入签名：

```python
def PlanningScene.begin_source_world_reuse(self, dense_scene):
```

直接读取的对象属性：`self.robot_validator`、`self._source_scene_checker`。

输出分支的源码表达式：

```python
session
```

直接写入的状态或数组位置：`self._source_world_reuse`、`dense_scene._source_world_reuse`。

直接调用：

```python
ValueError
PlanningScene.end_source_world_reuse
_SourceWorldStateReuse
```

## `PlanningScene.end_source_world_reuse`

[定义位置](../../../manipulation/planning_scene.py#L1949)。

输入签名：

```python
def PlanningScene.end_source_world_reuse(self):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
getattr
session.close
```

## `PlanningScene.check_robot_self_collision`

[定义位置](../../../manipulation/planning_scene.py#L1956)。

输入签名：

```python
def PlanningScene.check_robot_self_collision(self, joint_position, joint_names, trajectory_time, *, valid=None, diagnostics=None):
```

直接读取的对象属性：`self.robot_validator`、`self.gripper`、`result.new_tensor`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[valid]`。

直接调用：

```python
ValueError
len
torch.ones_like
joint_position.detach()[valid].cpu().numpy
joint_position.detach()[valid].cpu
joint_position.detach
trajectory_time.detach()[valid].cpu().numpy
trajectory_time.detach()[valid].cpu
trajectory_time.detach
set
zip
self.robot_validator.check_planning_self
residuals.append
unmodelled.update
float
list
row.tolist
torch.zeros_like
result.new_tensor
diagnostics.update
sorted
```

## `PlanningScene.check_world_geometry`

[定义位置](../../../manipulation/planning_scene.py#L1998)。

输入签名：

```python
def PlanningScene.check_world_geometry(self, joint_position, joint_names, trajectory_time, progress, *, geometry_query, robot_owner_permission_progress, valid=None, diagnostics=None):
```

直接读取的对象属性：`self.robot_validator`、`self._source_scene_checker`、`self._self_collision_owner_ids`、`result.new_tensor`、`self.gripper`、`scene.get`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`selected_values[key]`、`diagnostics['component_capture_error']`、`self._last_source_world_execution`、`result[valid]`、`execution['call_local_duplicates']`、`clearance['measured_count']`、`pose[:3, :3]`、`pose[:3, 3]`、`owner_poses[identity]`、`execution['physical_state_queries']`、`execution['reused_state_reports']`、`state_residuals[state_key]`、`clearance['complete_query']`、`clearance['distance_complete']`、`clearance['limitations']`、`clearance[name]`、`failure_references[state_key]`、`execution['completed']`。

直接调用：

```python
ValueError
len
torch.ones_like
geometry_query['nominal_owner_poses'].items
geometry_query['layer_selection'].items
fields.extend
iter
masked_cpu_snapshot
next
range
list
state_fields.extend
owner_values.values
selected_values.values
selection.items
getattr
reuse.binding_for
tuple
selected_values.items
diagnostics.get
valid.nonzero().detach().cpu().numpy
valid.nonzero().detach().cpu
valid.nonzero().detach
valid.nonzero
type
set
all
np.isfinite(value).all
np.isfinite
query_indices.tobytes
value.tobytes
reuse.lookup_batch
torch.zeros_like
result.new_tensor
diagnostics.update
execution.update
deepcopy
unmodelled.update
float
enumerate
zip
value[index].tobytes
residuals.append
reference_failure
int
reuse.lookup
owner_values.items
np.eye
Rotation.from_quat(q[index, [1, 2, 3, 0]]).as_matrix
Rotation.from_quat
encode_array
value.items
self.robot_validator.check_planning_world
reuse.record
(report.get('scene', {}).get('unpermitted_clearance') or {}).get
report.get('scene', {}).get
report.get
scene.get
sorted
min
diagnostics.setdefault
witnesses.append
row.tolist
sum
np.isfinite(residuals).all
reuse.record_batch
np.asarray
```

## `PlanningScene.check_world_geometry.reference_failure`

[定义位置](../../../manipulation/planning_scene.py#L2122)。

输入签名：

```python
def PlanningScene.check_world_geometry.reference_failure(index, witness_index):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`diagnostics['component_capture_error']`。

直接调用：

```python
diagnostics.setdefault('failure_witness_references', []).append
diagnostics.setdefault
int
type
```

## `PlanningScene.clone_for_rollout`

[定义位置](../../../manipulation/planning_scene.py#L2273)。

输入签名：

```python
def PlanningScene.clone_for_rollout(self):
```

直接读取的对象属性：`self.background`、`self.owners`、`result.owners`、`self.contacts`、`self.robot_owner_permissions`、`self.grasp_contact_activations`、`self._contact_cells`、`result._contact_cells`、`self.gripper`、`self.coverage`、`self.observation_report`、`self._coverage_geometry`、`self._coverage_samples`、`self._fixed_release_poses`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result._source_world_reuse`、`result._last_source_world_execution`、`result.background`、`result.owners`、`owner.query`、`owner.pose_path`、`owner.attachments`、`owner.local_spheres`、`result.contacts`、`result.robot_owner_permissions`、`result.grasp_contact_activations`、`result._contact_cells`、`value.last_report`、`value.last_required_report`、`result.gripper`、`result.coverage`、`result.observation_report`、`result._coverage_geometry`、`result._coverage_samples`、`result._fixed_release_poses`。

直接调用：

```python
copy
self.background.clone_for_rollout
tuple
owner.query.clone_for_rollout
deepcopy
owner.local_spheres.copy
self._contact_cells.items
result._contact_cells.values
self.coverage.clone_for_rollout
points.copy
value.clone
self._fixed_release_poses.items
```

## `PlanningScene.background_refinement_witnesses`

[定义位置](../../../manipulation/planning_scene.py#L2318)。

输入签名：

```python
@torch.no_grad()
def PlanningScene.background_refinement_witnesses(self, robot_spheres, position, quaternion, progress, joint_position=None, joint_names=(), *, trajectory_time, include_satisfied=False):
```

直接读取的对象属性：`self.background`、`self.gripper`、`self.robot_collision_bodies`。

输出分支的源码表达式：

```python
[]
witnesses
```

直接写入的状态或数组位置：`values[..., uncertain]`、`fine_values[..., uncertain]`。

直接调用：

```python
isinstance
ValueError
self.gripper.state
self.gripper.spheres
query.region.select
query.inflated_spheres
query.coarse
query.fine
zip
np.any
values.clone
fine_values.clone
torch.where
query.region.guidance_weights
torch.as_tensor
query.region.pose[:3].copy
trajectory_time.detach().cpu().numpy
trajectory_time.detach().cpu
trajectory_time.detach
envelopes.cpu().numpy
envelopes.cpu
selected.body_fine.cpu().numpy
selected.body_fine.cpu
weights.cpu().numpy
weights.cpu
values.cpu().numpy
values.cpu
fine_values.cpu().numpy
fine_values.cpu
enumerate
part.sum
(part > 0).sum
fine_costs[..., body['sphere_indices']].sum
np.argwhere
witnesses.append
int
float
bounds[batch, sample, column].tolist
bool
torch.no_grad
```

## `PlanningScene.refine_background`

[定义位置](../../../manipulation/planning_scene.py#L2384)。

输入签名：

```python
def PlanningScene.refine_background(self, witnesses):
```

直接读取的对象属性：`self.background`、`self.clone_for_rollout`、`result.observation_report`、`result.robot_validator`、`result._source_scene_checker`、`result.bind_source_scene`、`self._source_scene_checker`。

输出分支的源码表达式：

```python
(None, {'status': 'background_refinement_unavailable', 'active_witnesses': len(witnesses)})
(None, report)
(result, report)
```

直接写入的状态或数组位置：`result.background`、`result._source_scene_checker`、`result.observation_report['partitions']['observed_background']['fine']`。

直接调用：

```python
isinstance
len
query.refinement_source.refine
self.clone_for_rollout
report.get
deepcopy
result.observation_report.setdefault('background_refinements', []).append
result.observation_report.setdefault
result.bind_source_scene
```

## `PlanningScene.robot_coverage_guidance`

[定义位置](../../../manipulation/planning_scene.py#L2419)。

输入签名：

```python
def PlanningScene.robot_coverage_guidance(self, robot_link_poses, progress):
```

直接读取的对象属性：`self.robot_coverage`、`self._coverage_geometry`、`self._device_coverage_grids`、`self._free_distance`、`self._free_mask`、`self.coverage`、`self._coverage_samples`。

输出分支的源码表达式：

```python
torch.zeros_like(progress)
result
```

直接写入的状态或数组位置：`self._device_coverage_grids[key]`。

直接调用：

```python
torch.zeros_like
set
ValueError
_tensor
self._free_distance.to
self._free_mask.to
zip
_world_points
torch.is_grad_enabled
checkpoint
point_violation
torch.maximum
```

## `PlanningScene.robot_coverage_guidance.point_violation`

[定义位置](../../../manipulation/planning_scene.py#L2443)。

输入签名：

```python
def PlanningScene.robot_coverage_guidance.point_violation(points):
```

直接读取的对象属性：`self.coverage`、`self._has_free_cells`。

输出分支的源码表达式：

```python
torch.minimum(free_cost, initial_cost).amax(-1)
```

直接调用：

```python
torch.maximum
torch.zeros_like
torch.minimum
grid.reshape
F.grid_sample(distance_grid, sample_grid, align_corners=True, mode='bilinear', padding_mode='border').reshape
F.grid_sample
F.grid_sample(mask_grid, sample_grid, align_corners=True, mode='nearest', padding_mode='zeros').reshape
torch.linalg.vector_norm
torch.where
torch.full_like
float
_coverage_plane_chunk_size
math.prod
len
range
_coverage_halfspace_max
torch.relu
torch.minimum(free_cost, initial_cost).amax
```

## `PlanningScene.full_robot_positions`

[定义位置](../../../manipulation/planning_scene.py#L2479)。

输入签名：

```python
def PlanningScene.full_robot_positions(self, joint_position, joint_names, trajectory_time):
```

直接读取的对象属性：`self.gripper`。

输出分支的源码表达式：

```python
(names, values)
```

直接调用：

```python
tuple
set
ValueError
hand.state
torch.zeros_like
torch.cat
```

## `PlanningScene.check_robot_coverage`

[定义位置](../../../manipulation/planning_scene.py#L2496)。

输入签名：

```python
def PlanningScene.check_robot_coverage(self, joint_position, joint_names, trajectory_time, *, entrance=False):
```

直接读取的对象属性：`self.robot_coverage`、`self.gripper`、`self.full_robot_positions`。

输出分支的源码表达式：

```python
[]
reports
```

直接调用：

```python
set
hand.nominal_finger_trajectory.valid_mask_torch
unknown.detach().cpu().numpy().reshape
unknown.detach().cpu().numpy
unknown.detach().cpu
unknown.detach
self.full_robot_positions
positions.detach().cpu().numpy().reshape
positions.detach().cpu().numpy
positions.detach().cpu
positions.detach
len
trajectory_time.detach().cpu().numpy().reshape
trajectory_time.detach().cpu().numpy
trajectory_time.detach().cpu
trajectory_time.detach
zip
float
reports.append
self.robot_coverage.baseline_joint_names.index
baseline.astype(np.float32).astype
baseline.astype
np.all
np.array_equal
self.robot_coverage.check_state
```

## `PlanningScene.estimated_workspace_allocation`

[定义位置](../../../manipulation/planning_scene.py#L2549)。

输入签名：

```python
def PlanningScene.estimated_workspace_allocation(self, batch, horizon, *, robot_coverage_horizon, collision_paths=False):
```

直接读取的对象属性：`self.robot_sphere_materials`、`self.owners`、`self.background`、`self.robot_collision_bodies`、`self.gripper`、`self.robot_coverage`、`self.coverage`、`self._free_distance`、`self._free_mask`、`self._coverage_geometry`、`self._contact_cells`、`self.robot_validator`、`self._source_scene_checker`、`self.contacts`、`self.robot_owner_permissions`。

输出分支的源码表达式：

```python
{'gpu_retained': int(retained_bytes), 'gpu_live': int(gpu_live_bytes), 'cpu_shared': int(shared_cpu_bytes), 'gpu_shared': int(shared_gpu_bytes), 'cpu_live': int(cpu_live_bytes), 'gpu_graph_live': int(graph_live_bytes)}
```

直接调用：

```python
type
min
ValueError
len
actors.update
sum
isinstance
actors.items
targets.items
targets.values
id
finger_models.values
self.coverage._distance_squared.numel
self._free_distance.numel
self._free_mask.numel
_self_pair_chunk_size
max
range
math.ceil
_coverage_plane_chunk_size
self._contact_cells.values
hasattr
value.layers.values
value.workspace_bytes
actors.values
collision_path_workspace_allocation
int
```

## `PlanningScene.estimated_workspace_bytes`

[定义位置](../../../manipulation/planning_scene.py#L2776)。

输入签名：

```python
def PlanningScene.estimated_workspace_bytes(self, batch, horizon, *, robot_coverage_guidance=True):
```

直接读取的对象属性：`self.estimated_workspace_allocation`。

输出分支的源码表达式：

```python
sum((allocation[name] for name in ('gpu_retained', 'gpu_live', 'cpu_shared', 'gpu_shared', 'cpu_live')))
```

直接调用：

```python
self.estimated_workspace_allocation
sum
```

## `PlanningScene.project_residual`

[定义位置](../../../manipulation/planning_scene.py#L2786)。

输入签名：

```python
def PlanningScene.project_residual(self, progress, joint_position, joint_names):
```

直接读取的对象属性：`self.gripper`。

输出分支的源码表达式：

```python
joint_position[..., :0]
(joint_position[..., indices] - command) @ _tensor(hand._drive_basis.T, joint_position)
```

直接调用：

```python
tuple(joint_names).index
tuple
hand.joint_bounds
constant_tensor
_tensor
```

## `PlanningScene._release_pose`

[定义位置](../../../manipulation/planning_scene.py#L2808)。

输入签名：

```python
@staticmethod
def PlanningScene._release_pose(relation, position, quaternion, progress):
```

输出分支的源码表达式：

```python
((p + quaternion_rotate(q, _tensor(relation.position_tool_owner, p)), quaternion_multiply(q, _tensor(relation.quaternion_tool_owner, q))), matches.any(-1))
```

直接调用：

```python
ValueError
(progress - relation.interval[1]).abs
matches.to(torch.int64).argmax
matches.to
torch.gather
index[:, None, None].expand
quaternion_rotate
_tensor
quaternion_multiply
matches.any
```

## `PlanningScene.with_release_history`

[定义位置](../../../manipulation/planning_scene.py#L2830)。

输入签名：

```python
def PlanningScene.with_release_history(self, position, quaternion, progress):
```

直接读取的对象属性：`self.clone_for_rollout`、`self.owners`、`result._fixed_release_poses`、`self._release_pose`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result._fixed_release_poses[key]`。

直接调用：

```python
self.clone_for_rollout
enumerate
bool
(progress >= relation.interval[1] - 1e-07).any
self._release_pose
found.all
ValueError
```

## `PlanningScene._permission`

[定义位置](../../../manipulation/planning_scene.py#L2853)。

输入签名：

```python
def PlanningScene._permission(self, rule, actor_spheres, actor_pose, target_pose, progress, coordinates, half, *, uncertain_hand=None):
```

输出分支的源码表达式：

```python
(policy, residual * active.to(progress.dtype), (active & ~certain).to(progress.dtype))
```

直接调用：

```python
_tensor
torch.einsum
ValueError
torch.ones_like
_world_points
quaternion_rotate
quaternion_conjugate(target_pose[1]).unsqueeze
quaternion_conjugate
target_pose[0].unsqueeze
torch.atan2
torch.linalg.vector_norm
torch.linalg.cross
(actor_normal * target_normal).sum
torch.relu
_interval
actor_pose[1].unsqueeze
inverse_target.unsqueeze
(edge_normals * origin.unsqueeze(-2)).sum
origin.unsqueeze
(material_normal * origin).sum
rows.append
material_normal.unsqueeze
bounds.append
(sign * material_height + rule.modeling_allowance_m + tolerance).unsqueeze
enumerate
axis.expand
(axis @ patch_origin + width + tolerance).expand
(tangent * patch_normal).sum(-1).unsqueeze
(tangent * patch_normal).sum
(origin * (patch_normal - gap_axis)).sum
(material_normal * patch_normal).sum
cosine.unsqueeze
(sign * gap_axis).unsqueeze
(limit - sign * gap_offset + tolerance).unsqueeze
torch.cat
active.to
(active & ~certain).to
```

## `PlanningScene.robot_self_collision_guidance`

[定义位置](../../../manipulation/planning_scene.py#L2924)。

输入签名：

```python
def PlanningScene.robot_self_collision_guidance(self, robot_spheres, position, quaternion, progress, joint_position, joint_names, *, trajectory_time):
```

直接读取的对象属性：`self.gripper`、`self.owners`、`self._self_collision_owner_ids`。

输出分支的源码表达式：

```python
total
```

直接调用：

```python
torch.zeros_like
set
hand.state
hand.spheres
zip
np.any
_self_pair_sum
np.asarray(known_pairs, dtype=np.int64).reshape
np.asarray
torch.where
range
_local_spheres
owner.predicted_pose
owner.query
values.new_ones
enabled.index_fill_
constant_tensor
list
values[..., known].sum
values.sum
```

## `PlanningScene.components`

[定义位置](../../../manipulation/planning_scene.py#L2971)。

输入签名：

```python
def PlanningScene.components(self, robot_spheres, position, quaternion, progress, joint_position=None, joint_names=None, *, trajectory_time=None, robot_owner_permission_progress=None, layer_selection_out=None, geometry_query_out=None):
```

直接读取的对象属性：`self.robot_sphere_materials`、`self.gripper`、`self.owners`、`self._fixed_release_poses`、`self._release_pose`、`self._occupied_owner_ids`、`self.numerical_position_tolerance_m`、`self.numerical_rotation_tolerance_rad`、`self.robot_coverage`、`self.coverage`、`self.background`、`self.robot_collision_bodies`、`self.contacts`、`self._permission`、`self._contact_cells`、`self._self_collision_owner_ids`、`self.robot_owner_permissions`。

输出分支的源码表达式：

```python
costs
```

直接写入的状态或数组位置：`geometry_query_out['layer_selection']`、`costs['gripper/controller']`、`costs[f'unknown/release_anchor/{owner.owner_id}/{index}']`、`release_states[owner.owner_id]`、`poses[owner.owner_id]`、`geometry_query_out['nominal_owner_poses']`、`cover[:, 3]`、`actors[owner.owner_id]`、`costs[f'unknown/owner/{owner.owner_id}']`、`costs[f'attachment/start/{owner.owner_id}/{index}']`、`costs[f'unknown/space/{actor}']`、`policy['active']`、`costs[f'contact/{rule.rule_id}']`、`costs[f'unknown/contact/{rule.rule_id}']`、`costs[f'unknown/contact_model/{actor}/{owner.owner_id}']`。

直接调用：

```python
len
ValueError
isinstance
TypeError
self.gripper.state
self.gripper.spheres
torch.zeros_like
zip
np.any
set
range
enumerate
self._release_pose
((progress >= relation.interval[1]) & ~found[:, None]).to
releases.append
owner.pose
owner.local_spheres.copy
_world_spheres
torch.ones_like
owner.predicted_pose
quaternion_rotate
_tensor
quaternion_multiply
(progress - relation.interval[0]).abs
torch.linalg.vector_norm
quaternion_rotvec
quaternion_conjugate
torch.relu
boundary.to
actors.items
self.coverage.violation(actor_spheres).amax
self.coverage.violation
layer_selection
self.background.background_guidance
field_query
collision_terms
_local_spheres
self._permission
cells.required_contact
policy['active'].to
torch.maximum
unresolved.to
policies.append
owner.query.inflated_spheres
cells.query
torch.where
torch.stack([policy['active'].to(progress.dtype) for policy in policies]).amax
torch.stack
values.new_ones
enabled.index_fill_
constant_tensor
list
torch.zeros
selected.index_fill_
_interval
any
active.unsqueeze
```

## `PlanningScene.components.layer_selection`

[定义位置](../../../manipulation/planning_scene.py#L2999)。

输入签名：

```python
def PlanningScene.components.layer_selection(query, actor, target, queried_spheres):
```

直接读取的对象属性：`self.robot_collision_bodies`。

输出分支的源码表达式：

```python
None
selected
```

直接写入的状态或数组位置：`layer_selection_out[f'{actor}/{target}']`、`geometry_query_out['layer_selection'][f'{actor}/{target}']`。

直接调用：

```python
isinstance
list
range
query.region.select
selected.record
```

## `PlanningScene.components.field_query`

[定义位置](../../../manipulation/planning_scene.py#L3012)。

输入签名：

```python
def PlanningScene.components.field_query(query, queried_spheres, key, selected):
```

输出分支的源码表达式：

```python
query(queried_spheres, key=key, selection=selected) if selected is not None else query(queried_spheres, key=key)
```

直接调用：

```python
query
```

## `PlanningScene.components.collision_terms`

[定义位置](../../../manipulation/planning_scene.py#L3045)。

输入签名：

```python
def PlanningScene.components.collision_terms(values, key, actor):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`costs['unknown/' + key]`、`costs[key]`。

直接调用：

```python
constant_tensor
values[..., uncertain_columns].sum
values[..., known_columns].sum
torch.where
values.sum
```

## `PlanningScene.source_components`

[定义位置](../../../manipulation/planning_scene.py#L3227)。

输入签名：

```python
def PlanningScene.source_components(self, robot_spheres, position, quaternion, progress, joint_position, joint_names, *, trajectory_time, robot_owner_permission_progress=None, layer_selection_out=None, valid=None, diagnostics=None):
```

直接读取的对象属性：`self.components`、`self.check_world_geometry`、`result.items`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result['source_world_collision']`。

直接调用：

```python
self.components
self.check_world_geometry
torch.where
torch.zeros_like
result.items
```

## `PlanningScene.evaluate_source`

[定义位置](../../../manipulation/planning_scene.py#L3253)。

输入签名：

```python
def PlanningScene.evaluate_source(self, robot_spheres, position, quaternion, progress, joint_position, joint_names, *, trajectory_time, robot_owner_permission_progress=None, layer_selection_out=None, valid=None, diagnostics=None, components_out=None):
```

直接读取的对象属性：`self.source_components`。

输出分支的源码表达式：

```python
torch.stack(known).sum(0) if known else torch.zeros_like(progress)
```

直接调用：

```python
self.source_components
components_out.update
terms.items
key.startswith
torch.stack(known).sum
torch.stack
torch.zeros_like
```

## `PlanningScene.evaluate`

[定义位置](../../../manipulation/planning_scene.py#L3270)。

输入签名：

```python
def PlanningScene.evaluate(self, robot_spheres, position, quaternion, progress, joint_position=None, joint_names=None, *, trajectory_time=None, robot_owner_permission_progress=None, layer_selection_out=None):
```

直接读取的对象属性：`self.components`。

输出分支的源码表达式：

```python
torch.stack(known).sum(0) if known else torch.zeros_like(progress)
```

直接调用：

```python
self.components
terms.items
key.startswith
torch.stack(known).sum
torch.stack
torch.zeros_like
```

## `PlanningScene.optimizer_preferences`

[定义位置](../../../manipulation/planning_scene.py#L3287)。

输入签名：

```python
def PlanningScene.optimizer_preferences(self, progress, joint_position, joint_names):
```

直接读取的对象属性：`self.gripper`。

输出分支的源码表达式：

```python
torch.zeros_like(progress)
((joint_position[..., indices] - command) ** 2).sum(-1)
```

直接调用：

```python
torch.zeros_like
ValueError
hand.joint_bounds
len
tuple(joint_names).index
tuple
constant_tensor
((joint_position[..., indices] - command) ** 2).sum
```

## `PlanningScene.to_execution_binding`

[定义位置](../../../manipulation/planning_scene.py#L3314)。

输入签名：

```python
def PlanningScene.to_execution_binding(self):
```

直接读取的对象属性：`self.owners`、`self.gripper`、`self.reference_frame`、`self.robot_geometry_source`、`self.robot_validator`、`self.robot_collision_bodies`、`self.native_geometry_materials`、`self.robot_sphere_materials`、`self.background`、`self.contacts`、`self.robot_owner_permissions`、`self.grasp_contact_activations`、`self._contact_cells`、`self.coverage`、`self.observation_report`。

输出分支的源码表达式：

```python
{'reference_frame': self.reference_frame, 'robot_geometry_source': self.robot_geometry_source, 'robot_self_collision': None if self.robot_validator is None else self.robot_validator.planning_binding_record(), 'robot_collision_bodies': deepcopy(self.robot_collision_bodies), 'native_geometry_materials': deepcopy(self.native_geometry_materials), 'robot_sphere_materials': list(self.robot_sphere_materials), 'gripper': gripper, 'background': {'owner_ids': sorted(self.background.owner_ids), 'source': self.background.source, 'geometry': geometry(self.background)}, 'owners': owners, 'contacts': plain([asdict(rule) for rule in self.contacts]), 'robot_owner_permissions': plain([asdict(rule) for rule in self.robot_owner_permissions]), 'grasp_contact_activations': plain(self.grasp_contact_activations), 'contact_query_models': {identity: {'complete_components': value.layer_cell_counts if hasattr(value, 'layers') else len(value.centres), 'query_gaps': list(value.gaps), 'cell_half_diagonal_m': value.cell_half_diagonal_m, 'maximum_candidate_pairs': value.max_candidate_pairs, 'scope': 'observed TSDF zero surface and negative known components, or explicitly supplied native solids'} for (identity, value) in self._contact_cells.items()}, 'coverage': {'origin': self.coverage.origin.tolist(), 'voxel_size': self.coverage.voxel_size, 'shape_zyx': list(self.coverage._known.shape), 'known_packbits': np.packbits(self.coverage._known).tolist(), 'free_packbits': None if self.coverage._known_free is None else np.packbits(self.coverage._known_free).tolist(), 'observation_ids': list(self.coverage.observation_ids), 'occupancy_evidence': plain(self.coverage.occupancy_evidence)}, 'observation_report': plain(self.observation_report), 'scope': 'Supplied observed/native geometry only; hidden space remains unresolved'}
```

直接调用：

```python
owners.append
geometry
plain
asdict
list
hand.nominal_finger_trajectory.describe
self.robot_validator.planning_binding_record
deepcopy
sorted
hasattr
len
self._contact_cells.items
self.coverage.origin.tolist
np.packbits(self.coverage._known).tolist
np.packbits
np.packbits(self.coverage._known_free).tolist
```

## `PlanningScene.to_execution_binding.plain`

[定义位置](../../../manipulation/planning_scene.py#L3322)。

输入签名：

```python
def PlanningScene.to_execution_binding.plain(value):
```

输出分支的源码表达式：

```python
value.tolist()
value.item()
value.detach().cpu().tolist()
{str(k): plain(v) for (k, v) in value.items()}
[plain(v) for v in value]
value
```

直接调用：

```python
isinstance
value.tolist
value.item
value.detach().cpu().tolist
value.detach().cpu
value.detach
str
plain
value.items
```

## `PlanningScene.describe`

[定义位置](../../../manipulation/planning_scene.py#L3407)。

输入签名：

```python
def PlanningScene.describe(self):
```

直接读取的对象属性：`self.reference_frame`、`self.robot_validator`、`self._self_collision_owner_ids`、`self._source_scene_checker`、`self.coverage`、`self.observation_report`、`self._occupied_owner_ids`、`self.background`、`self.owners`、`self._contact_cells`、`self.robot_owner_permissions`、`self.numerical_position_tolerance_m`、`self.numerical_rotation_tolerance_rad`、`self.gripper`、`self.robot_coverage`、`self.required_progress_anchors`。

输出分支的源码表达式：

```python
{'reference_frame': self.reference_frame, 'robot_self_collision': None if self.robot_validator is None else {'name': self.robot_validator.name, 'source': self.robot_validator.source, 'installation_owner_ids': sorted(self._self_collision_owner_ids), 'acceptance': 'all known required source-solid pairs at each validation state', 'sphere_proxy': 'optimizer guidance only'}, 'world_collision_acceptance': {'source_scene_bound': self._source_scene_checker is not None, 'geometry': 'source robot solids against observed TSDF zero surfaces and negative known material', 'scope': 'all valid native states, required finite contact, selected layers and conditional owner poses', 'sphere_esdf_proxy': 'optimizer guidance only'}, 'optimizer_collision_activation': {'distance_m': 0.0, 'scope': 'CuroboSphereQuery raw sphere/ESDF soft activation', 'hard_clearance_owner': 'source geometry pair policy', 'calibration': 'positive global width requires backend regression evidence'}, 'observation_ids': self.coverage.observation_ids, 'observation_build': deepcopy(self.observation_report), 'current_occupied_evidence': deepcopy(self.coverage.occupancy_evidence), 'occupied_owner_ids': tuple(sorted(self._occupied_owner_ids)), 'background_owner_ids': tuple(sorted(self.background.owner_ids)), 'owner_geometry_complete': {owner.owner_id: owner.geometry_complete for owner in self.owners}, 'contact_queries': {identity: {'complete_components': value.layer_cell_counts if hasattr(value, 'layers') else len(value.centres), 'query_gaps': list(value.gaps), 'last_report': deepcopy(value.last_report), 'required_report': deepcopy(value.last_required_report)} for (identity, value) in self._contact_cells.items()}, 'native_installation_exclusions': {owner.owner_id: {'robot_sphere_indices': owner.robot_excluded_spheres, 'policy_source': owner.robot_exclusion_source, 'robot_geometry_source': owner.robot_exclusion_geometry_source} for owner in self.owners if owner.robot_excluded_spheres}, 'conditional_attachments': {owner.owner_id: [relation.condition_id for relation in owner.attachments] for owner in self.owners if owner.attachments}, 'robot_owner_permissions': [asdict(rule) for rule in self.robot_owner_permissions], 'release_requires_evidence': {owner.owner_id: [relation.release_condition_id for relation in owner.attachments if relation.release_condition_id is not None] for owner in self.owners if any((relation.release_condition_id is not None for relation in owner.attachments))}, 'attachment_numerical_tolerances': {'position_m': self.numerical_position_tolerance_m, 'rotation_rad': self.numerical_rotation_tolerance_rad}, 'gripper_controller': None if self.gripper is None else self.gripper.controller, 'gripper_control_timestep_s': None if self.gripper is None else self.gripper.control_timestep_s, 'gripper_timing_contract': None if self.gripper is None else 'time-independent diagnostic envelope; future width remains unresolved' if self.gripper.controller == 'binary' else 'same-time named command and separately declared nominal geometry; actual q/dq supervised' if self.gripper.controller == 'timed_reference' else 'native position tracking; final solver joint-rate limits apply', 'robot_new_volume_coverage': None if self.robot_coverage is None else {'initial_state_assumption': 'native solid V0 at measured full named q0 is a valid initial occupied volume', 'diagnostic': 'CPU full-volume V(q) minus (V0 union original observed free cells); deficits remain eligible uncertainty', 'optimizer': 'bounded native body points only; not a coverage certificate'}, 'numerical_joint_tolerance': None if self.gripper is None else self.gripper.numerical_joint_tolerance, 'gripper_projection_dimension': 0 if self.gripper is None or self.gripper.controller != 'position_tracking' else len(self.gripper._drive_basis), 'gripper_coupling': None if self.gripper is None else {'joint_names': self.gripper.joint_names, 'matrix': self.gripper.coupling_matrix.tolist(), 'target': self.gripper.coupling_offset.tolist(), 'source': self.gripper.source}, 'required_progress_anchors': self.required_progress_anchors, 'scope': 'observed-space, sphere-proxy discrete checks; not continuous or physical contact certification'}
```

直接调用：

```python
sorted
deepcopy
tuple
hasattr
len
list
self._contact_cells.items
asdict
any
self.gripper.coupling_matrix.tolist
self.gripper.coupling_offset.tolist
```
