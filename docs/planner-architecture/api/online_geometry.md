# `manipulation/online_geometry.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_plain`

[定义位置](../../../manipulation/online_geometry.py#L35)。

输入签名：

```python
def _plain(value):
```

输出分支的源码表达式：

```python
value.tolist()
value.item()
{str(k): _plain(v) for (k, v) in value.items()}
[_plain(v) for v in value]
value
```

直接调用：

```python
isinstance
value.tolist
value.item
str
_plain
value.items
```

## `FrozenESDF`

[定义位置](../../../manipulation/online_geometry.py#L47)。

```python
class FrozenESDF:
    ...
    ...
```

该类型的直接方法：`__init__`、`from_source`、`clone_for_rollout`、`_sample`、`signed_distance`、`supported`、`supports_centres`、`intersects_geometry`、`__call__`、`prepare_contacts`、`required_contact`、`contact_query`、`record`、`from_record`。

## `FrozenESDF.__init__`

[定义位置](../../../manipulation/online_geometry.py#L56)。

输入签名：

```python
def FrozenESDF.__init__(self, distance, *, origin, voxel_size, source_geometry, owner_ids, source):
```

直接读取的对象属性：`self.origin`、`self.voxel_size`、`self.distance`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.origin`、`self.voxel_size`、`self.distance`、`self.distance.flags.writeable`、`self.source_geometry`、`self.owner_ids`、`self.source`、`self._device`、`self._contact_model`、`self.upper`、`self.geometry_lower`、`self.geometry_upper`。

直接调用：

```python
np.asarray
min
np.isfinite(distance).all
np.isfinite
isinstance
ValueError
np.asarray(origin, dtype=float).copy
float
np.isfinite(self.origin).all
distance.copy
frozenset
str
```

## `FrozenESDF.from_source`

[定义位置](../../../manipulation/online_geometry.py#L81)。

输入签名：

```python
@classmethod
def FrozenESDF.from_source(cls, source_geometry, *, origin, shape, voxel_size, owner_ids, source):
```

输出分支的源码表达式：

```python
cls(distance, origin=origin, voxel_size=voxel_size, source_geometry=source_geometry, owner_ids=owner_ids, source=source)
```

直接调用：

```python
source_geometry.distance_field
np.zeros
cls
```

## `FrozenESDF.clone_for_rollout`

[定义位置](../../../manipulation/online_geometry.py#L90)。

输入签名：

```python
def FrozenESDF.clone_for_rollout(self):
```

输出分支的源码表达式：

```python
copy(self)
```

直接调用：

```python
copy
```

## `FrozenESDF._sample`

[定义位置](../../../manipulation/online_geometry.py#L96)。

输入签名：

```python
def FrozenESDF._sample(self, value, points, name):
```

直接读取的对象属性：`self._device`、`self.origin`、`self.voxel_size`、`self.distance`。

输出分支的源码表达式：

```python
F.grid_sample(self._device[key], grid.reshape(1, 1, 1, -1, 3), align_corners=True, padding_mode='zeros', mode='bilinear').reshape(points.shape[:-1])
```

直接写入的状态或数组位置：`self._device[key]`。

直接调用：

```python
torch.as_tensor
np.array
value.transpose
_tensor
np.asarray
F.grid_sample(self._device[key], grid.reshape(1, 1, 1, -1, 3), align_corners=True, padding_mode='zeros', mode='bilinear').reshape
F.grid_sample
grid.reshape
```

## `FrozenESDF.signed_distance`

[定义位置](../../../manipulation/online_geometry.py#L106)。

输入签名：

```python
def FrozenESDF.signed_distance(self, points):
```

直接读取的对象属性：`self._sample`、`self.distance`。

输出分支的源码表达式：

```python
self._sample(self.distance, points, 'distance')
```

直接调用：

```python
self._sample
```

## `FrozenESDF.supported`

[定义位置](../../../manipulation/online_geometry.py#L109)。

输入签名：

```python
def FrozenESDF.supported(self, spheres):
```

直接读取的对象属性：`self.origin`、`self.upper`。

输出分支的源码表达式：

```python
interior
```

直接调用：

```python
spheres[..., 3:4].clamp_min
((spheres[..., :3] - radius >= _tensor(self.origin, spheres)) & (spheres[..., :3] + radius <= _tensor(self.upper, spheres))).all
_tensor
```

## `FrozenESDF.supports_centres`

[定义位置](../../../manipulation/online_geometry.py#L118)。

输入签名：

```python
def FrozenESDF.supports_centres(self, spheres):
```

直接读取的对象属性：`self.origin`、`self.upper`。

输出分支的源码表达式：

```python
((spheres[..., :3] >= _tensor(self.origin, spheres)) & (spheres[..., :3] <= _tensor(self.upper, spheres))).all(-1)
```

直接调用：

```python
((spheres[..., :3] >= _tensor(self.origin, spheres)) & (spheres[..., :3] <= _tensor(self.upper, spheres))).all
_tensor
```

## `FrozenESDF.intersects_geometry`

[定义位置](../../../manipulation/online_geometry.py#L122)。

输入签名：

```python
def FrozenESDF.intersects_geometry(self, spheres):
```

直接读取的对象属性：`self.geometry_lower`、`self.geometry_upper`。

输出分支的源码表达式：

```python
torch.zeros(spheres.shape[:-1], dtype=torch.bool, device=spheres.device)
((spheres[..., :3] + spheres[..., 3:4] >= _tensor(self.geometry_lower, spheres)) & (spheres[..., :3] - spheres[..., 3:4] <= _tensor(self.geometry_upper, spheres))).all(-1)
```

直接调用：

```python
torch.zeros
((spheres[..., :3] + spheres[..., 3:4] >= _tensor(self.geometry_lower, spheres)) & (spheres[..., :3] - spheres[..., 3:4] <= _tensor(self.geometry_upper, spheres))).all
_tensor
```

## `FrozenESDF.__call__`

[定义位置](../../../manipulation/online_geometry.py#L128)。

输入签名：

```python
def FrozenESDF.__call__(self, spheres, *, key=None):
```

直接读取的对象属性：`self.signed_distance`、`self.intersects_geometry`、`self.supports_centres`。

输出分支的源码表达式：

```python
torch.where((spheres[..., 3] > 0) & intersects & supported, values, torch.zeros_like(values))
```

直接调用：

```python
torch.relu
self.signed_distance
self.intersects_geometry
self.supports_centres
torch.where
torch.zeros_like
```

## `FrozenESDF.prepare_contacts`

[定义位置](../../../manipulation/online_geometry.py#L143)。

输入签名：

```python
def FrozenESDF.prepare_contacts(self, rules, maximum_radii):
```

直接读取的对象属性：`self.source_geometry`、`self._contact_model`、`self.distance`、`self.origin`、`self.voxel_size`、`self.source`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self._contact_model`。

直接调用：

```python
self.source_geometry.prepare_rules
self._contact_model.validate_binding
PreparedContactFields.prepare
```

## `FrozenESDF.required_contact`

[定义位置](../../../manipulation/online_geometry.py#L155)。

输入签名：

```python
def FrozenESDF.required_contact(self, rule, policy):
```

直接读取的对象属性：`self._contact_model`。

输出分支的源码表达式：

```python
self._contact_model.required_contact(rule, policy)
```

直接调用：

```python
ValueError
self._contact_model.required_contact
```

## `FrozenESDF.contact_query`

[定义位置](../../../manipulation/online_geometry.py#L160)。

输入签名：

```python
def FrozenESDF.contact_query(self, spheres, policies):
```

直接读取的对象属性：`self._contact_model`、`self.intersects_geometry`。

输出分支的源码表达式：

```python
(ordinary, torch.zeros_like(ordinary))
self._contact_model.query(spheres, policies, ordinary, self.intersects_geometry(spheres))
```

直接调用：

```python
self
torch.zeros_like
ValueError
self._contact_model.query
self.intersects_geometry
```

## `FrozenESDF.record`

[定义位置](../../../manipulation/online_geometry.py#L169)。

输入签名：

```python
def FrozenESDF.record(self):
```

直接读取的对象属性：`self.distance`、`self.source_geometry`、`self.origin`、`self.voxel_size`、`self.owner_ids`、`self.source`、`self._contact_model`。

输出分支的源码表达式：

```python
{'kind': 'frozen_observed_field_v3', 'distance': encode_array(self.distance), 'source_geometry': self.source_geometry.record(), 'origin': self.origin.tolist(), 'voxel_size': self.voxel_size, 'owner_ids': sorted(self.owner_ids), 'source': self.source, 'contact_model': None if self._contact_model is None else self._contact_model.record()}
```

直接调用：

```python
encode_array
self.source_geometry.record
self.origin.tolist
sorted
self._contact_model.record
```

## `FrozenESDF.from_record`

[定义位置](../../../manipulation/online_geometry.py#L177)。

输入签名：

```python
@classmethod
def FrozenESDF.from_record(cls, row):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result._contact_model`。

直接调用：

```python
row.get
FrozenPrimitives
cls
decode_array
FrozenObservedGeometry
ValueError
PreparedContactFields.from_record
```

## `FrozenPrimitives`

[定义位置](../../../manipulation/online_geometry.py#L193)。

```python
class FrozenPrimitives(FrozenESDF):
    ...
    ...
```

该类型的直接方法：`__init__`、`signed_distance`、`supported`、`_sample`、`prepare_contacts`、`record`。

## `FrozenPrimitives.__init__`

[定义位置](../../../manipulation/online_geometry.py#L200)。

输入签名：

```python
def FrozenPrimitives.__init__(self, *, primitives, owner_ids, source, contact_voxel_size_m=None):
```

直接读取的对象属性：`self.primitives`、`self.origin`、`self.upper`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.primitives`、`self.owner_ids`、`self.source`、`self.contact_voxel_size_m`、`self._device`、`self._contact_model`、`self.distance`、`row['pose']`、`self.origin`、`self.upper`、`self.geometry_lower`、`self.geometry_upper`、`self.voxel_size`。

直接调用：

```python
deepcopy
len
ValueError
frozenset
str
np.empty
np.asarray
list
product
np.isfinite(dims).all
np.isfinite
np.any
row.get
np.isfinite(pose).all
np.isclose
np.linalg.norm
pose.tolist
np.repeat
np.array
Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix
Rotation.from_quat
lower.append
points.min
upper.append
points.max
np.min
np.max
float
```

## `FrozenPrimitives.signed_distance`

[定义位置](../../../manipulation/online_geometry.py#L229)。

输入签名：

```python
def FrozenPrimitives.signed_distance(self, points):
```

直接读取的对象属性：`self.primitives`。

输出分支的源码表达式：

```python
torch.stack(values).amin(0)
```

直接调用：

```python
_tensor
quaternion_rotate
quaternion_conjugate
local.abs
torch.linalg.vector_norm
torch.relu
delta.amax(-1).clamp_max
delta.amax
torch.stack
local[..., 2].abs
values.append
torch.stack(values).amin
```

## `FrozenPrimitives.supported`

[定义位置](../../../manipulation/online_geometry.py#L250)。

输入签名：

```python
def FrozenPrimitives.supported(self, spheres):
```

输出分支的源码表达式：

```python
torch.ones(spheres.shape[:-1], dtype=torch.bool, device=spheres.device)
```

直接调用：

```python
torch.ones
```

## `FrozenPrimitives._sample`

[定义位置](../../../manipulation/online_geometry.py#L255)。

输入签名：

```python
def FrozenPrimitives._sample(self, value, points, name):
```

直接读取的对象属性：`self.signed_distance`。

输出分支的源码表达式：

```python
self.signed_distance(points)
torch.ones(points.shape[:-1], dtype=points.dtype, device=points.device)
```

直接调用：

```python
self.signed_distance
torch.ones
```

## `FrozenPrimitives.prepare_contacts`

[定义位置](../../../manipulation/online_geometry.py#L261)。

输入签名：

```python
def FrozenPrimitives.prepare_contacts(self, rules, maximum_radii):
```

直接读取的对象属性：`self._contact_model`、`self.contact_voxel_size_m`、`self.voxel_size`、`self.geometry_lower`、`self.geometry_upper`、`self.signed_distance`、`self.source`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`distance[start:start + len(flat)]`、`self._contact_model`。

直接调用：

```python
self._contact_model.validate_binding
ValueError
max
maximum_radii.values
math.sqrt
np.asarray
list
product
np.minimum
points.min
np.maximum
points.max
np.ceil((upper - lower) / self.voxel_size).astype
np.ceil
math.prod
np.empty
torch.no_grad
range
np.arange
min
np.column_stack
np.unravel_index
tuple
len
self.signed_distance(torch.as_tensor(points, dtype=torch.float64)).numpy
self.signed_distance
torch.as_tensor
distance.reshape
PreparedContactFields.prepare
```

## `FrozenPrimitives.record`

[定义位置](../../../manipulation/online_geometry.py#L298)。

输入签名：

```python
def FrozenPrimitives.record(self):
```

直接读取的对象属性：`self.primitives`、`self.owner_ids`、`self.source`、`self.contact_voxel_size_m`、`self._contact_model`。

输出分支的源码表达式：

```python
{'kind': 'frozen_primitives_v2', 'primitives': deepcopy(self.primitives), 'owner_ids': sorted(self.owner_ids), 'source': self.source, 'contact_voxel_size_m': self.contact_voxel_size_m, 'contact_model': None if self._contact_model is None else self._contact_model.record()}
```

直接调用：

```python
deepcopy
sorted
self._contact_model.record
```

## `OnlinePlanningScene`

[定义位置](../../../manipulation/online_geometry.py#L304)。

```python
class OnlinePlanningScene:
    ...
    ...
```

该类型的直接方法：`__init__`、`_bind_initial_supports`、`native_world_collision_components`、`native_collision_components`、`native_world_collision_pairs`、`bind_online_robot`、`_prepare_self_collision_pairs`、`_self_collision_pair_tensors`、`_prepare_contacts`、`clone_for_rollout`、`validate_problem`、`_prepare_unknown_coverage`、`_unknown`、`components`、`evaluate`、`estimated_workspace_allocation`、`query_state`、`describe`、`to_execution_binding`、`from_binding`。

## `OnlinePlanningScene.__init__`

[定义位置](../../../manipulation/online_geometry.py#L306)。

输入签名：

```python
def OnlinePlanningScene.__init__(self, *, background, owners, coverage, reference_frame, robot_sphere_materials, robot_geometry_source, contacts=(), gripper=None, has_gripper=True, unknown_policy, robot_collision_bodies=(), native_geometry_materials=None, robot_owner_permissions=(), numerical_position_tolerance_m=1e-06, numerical_rotation_tolerance_rad=1e-06, initial_support_relations=(), holding_contacts=()):
```

直接读取的对象属性：`self.robot_sphere_materials`、`self.owners`、`self.contacts`、`self.holding_contacts`、`self.native_geometry_materials`、`self.robot_collision_bodies`、`self.unknown_policy`、`self._prepare_unknown_coverage`、`self._prepare_self_collision_pairs`、`self._owner_collision_path_radii`、`self._bind_initial_supports`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.background`、`self.owners`、`self.coverage`、`self.reference_frame`、`self.robot_geometry_source`、`self.robot_sphere_materials`、`self.contacts`、`self.gripper`、`self.has_gripper`、`self.robot_collision_bodies`、`self.native_geometry_materials`、`self.holding_contacts`、`self.unknown_policy`、`self.robot_sphere_geometry`、`self.robot_self_collision_pairs`、`self.native_self_collision_pairs`、`self.robot_self_collision_padding`、`self._collision_path_indices`、`self._collision_path_radii`、`self._collision_path_device`、`self._owner_collision_path_radii`、`radii.flags.writeable`、`self._owner_collision_path_radii[owner.owner_id]`、`self.robot_model`、`self.required_robot_links`、`self.robot_owner_permissions`、`self.grasp_contact_activations`、`self.observation_report`、`self._fixed_release_poses`、`self.numerical_position_tolerance_m`、`self.numerical_rotation_tolerance_rad`、`self.required_progress_anchors`、`native_owners[owner.owner_id]`。

直接调用：

```python
ValueError
isinstance
any
tuple
len
deepcopy
validate_holding_materials
set
np.asarray
np.isfinite([lower, upper]).all
np.isfinite
np.any
row.get
self._prepare_unknown_coverage
self._prepare_self_collision_pairs
np.isfinite(radii).all
float
sorted
bind_native_owner_geometry
next
require_native_material
self.native_geometry_materials.get
check_native_owner_material
self._bind_initial_supports
```

## `OnlinePlanningScene._bind_initial_supports`

[定义位置](../../../manipulation/online_geometry.py#L417)。

输入签名：

```python
def OnlinePlanningScene._bind_initial_supports(self, relations, records=None):
```

直接读取的对象属性：`self.contacts`、`self.owners`、`self.numerical_position_tolerance_m`、`self.required_progress_anchors`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.initial_support_relations`、`self._initial_support_models`、`self.required_progress_anchors`。

直接调用：

```python
tuple
len
any
isinstance
ValueError
set
PreparedInitialSupport
sorted
```

## `OnlinePlanningScene.native_world_collision_components`

[定义位置](../../../manipulation/online_geometry.py#L444)。

输入签名：

```python
@property
def OnlinePlanningScene.native_world_collision_components(self):
```

直接读取的对象属性：`self.native_world_collision_pairs`。

输出分支的源码表达式：

```python
frozenset(self.native_world_collision_pairs)
```

直接调用：

```python
frozenset
```

## `OnlinePlanningScene.native_collision_components`

[定义位置](../../../manipulation/online_geometry.py#L457)。

输入签名：

```python
@property
def OnlinePlanningScene.native_collision_components(self):
```

直接读取的对象属性：`self.native_world_collision_components`。

输出分支的源码表达式：

```python
self.native_world_collision_components | frozenset({'collision/robot/self'})
```

直接调用：

```python
frozenset
```

## `OnlinePlanningScene.native_world_collision_pairs`

[定义位置](../../../manipulation/online_geometry.py#L470)。

输入签名：

```python
@property
def OnlinePlanningScene.native_world_collision_pairs(self):
```

直接读取的对象属性：`self.owners`。

输出分支的源码表达式：

```python
pairs
```

直接写入的状态或数组位置：`pairs[key]`。

直接调用：

```python
isinstance
all
ValueError
```

## `OnlinePlanningScene.bind_online_robot`

[定义位置](../../../manipulation/online_geometry.py#L487)。

输入签名：

```python
def OnlinePlanningScene.bind_online_robot(self, *, robot_sphere_geometry, robot_self_collision_pairs, robot_self_collision_padding=None, robot_geometry_source=None, robot_model=None):
```

直接读取的对象属性：`self.robot_geometry_source`、`self.robot_sphere_materials`、`self.gripper`、`self._prepare_self_collision_pairs`、`self._collision_path_indices`、`self._collision_path_radii`、`self._prepare_contacts`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`radii[ids]`、`self.robot_sphere_geometry`、`self.robot_self_collision_pairs`、`self.native_self_collision_pairs`、`self.robot_self_collision_padding`、`self.robot_model`、`self.required_robot_links`、`path_radii[list(self.gripper.sphere_indices)]`、`self._collision_path_indices`、`self._collision_path_radii`、`self._collision_path_indices.flags.writeable`、`self._collision_path_radii.flags.writeable`、`self._collision_path_device`。

直接调用：

```python
ValueError
np.zeros
len
np.asarray
np.isfinite(centres).all
np.isfinite
np.isfinite(sizes).all
indices.extend
sorted
list
range
np.asarray(robot_self_collision_pairs, dtype=np.int64).reshape
pairs.min
pairs.max
np.any
np.asarray(robot_self_collision_padding).reshape
np.isfinite(padding).all
deepcopy
pairs.tolist
self.gripper.self_collision_model.filter_pairs
self._prepare_self_collision_pairs
padding.tolist
tuple
radii.copy
np.isfinite(path_radii).all
np.flatnonzero
self._prepare_contacts
```

## `OnlinePlanningScene._prepare_self_collision_pairs`

[定义位置](../../../manipulation/online_geometry.py#L538)。

输入签名：

```python
def OnlinePlanningScene._prepare_self_collision_pairs(self):
```

直接读取的对象属性：`self.gripper`、`self.native_self_collision_pairs`、`self.robot_sphere_materials`、`self._uncertain_sphere_indices`、`self._self_collision_pair_arrays`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._uncertain_sphere_indices`、`moving[list(self._uncertain_sphere_indices)]`、`self._self_collision_pair_arrays`、`value.flags.writeable`、`self._self_collision_pair_device`。

直接调用：

```python
tuple
zip
np.any
np.asarray(self.native_self_collision_pairs, dtype=np.int64).reshape
np.asarray
np.zeros
len
list
moving[pairs].any
```

## `OnlinePlanningScene._self_collision_pair_tensors`

[定义位置](../../../manipulation/online_geometry.py#L558)。

输入签名：

```python
def OnlinePlanningScene._self_collision_pair_tensors(self, spheres):
```

直接读取的对象属性：`self._self_collision_pair_device`、`self._self_collision_pair_arrays`。

输出分支的源码表达式：

```python
self._self_collision_pair_device[key]
```

直接写入的状态或数组位置：`self._self_collision_pair_device[key]`。

直接调用：

```python
tuple
torch.as_tensor
np.array
```

## `OnlinePlanningScene._prepare_contacts`

[定义位置](../../../manipulation/online_geometry.py#L572)。

输入签名：

```python
def OnlinePlanningScene._prepare_contacts(self, robot_radii):
```

直接读取的对象属性：`self.owners`、`self.contacts`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`maximum[rule.rule_id]`。

直接调用：

```python
tuple
float
max
list
owner.query.prepare_contacts
```

## `OnlinePlanningScene.clone_for_rollout`

[定义位置](../../../manipulation/online_geometry.py#L585)。

输入签名：

```python
def OnlinePlanningScene.clone_for_rollout(self):
```

直接读取的对象属性：`self._fixed_release_poses`、`self.background`、`self.owners`、`result.owners`、`self.contacts`、`self.unknown_policy`、`self.gripper`、`result.gripper`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result._fixed_release_poses`、`result.background`、`result.owners`、`owner.query`、`owner.attachments`、`result.contacts`、`result.unknown_policy`、`result.gripper`、`result.gripper.commands`。

直接调用：

```python
copy
dict
self.background.clone_for_rollout
tuple
owner.query.clone_for_rollout
deepcopy
```

## `OnlinePlanningScene.validate_problem`

[定义位置](../../../manipulation/online_geometry.py#L606)。

输入签名：

```python
def OnlinePlanningScene.validate_problem(self, problem, *, goal_identities=None):
```

直接读取的对象属性：`problem.reference_frame`、`self.reference_frame`、`self.contacts`、`problem.pose_events`、`problem.conditions`、`self.holding_contacts`、`self.owners`、`self.gripper`、`problem.events`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.grasp_contact_activations`。

直接调用：

```python
ValueError
asdict
list
range
len
row.get
grasp_contact_rule_ids
validate_grasp_contact_arrival
pose_ids.index
activations.append
deepcopy
tuple
sum
set
validate_holding_conditions
declaration.get
release.get
```

## `OnlinePlanningScene._prepare_unknown_coverage`

[定义位置](../../../manipulation/online_geometry.py#L647)。

输入签名：

```python
def OnlinePlanningScene._prepare_unknown_coverage(self):
```

直接读取的对象属性：`self.unknown_policy`、`self.coverage`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self._permitted_coverage`、`allowed[source_slices]`。

直接调用：

```python
min
float
range
max
math.isfinite
ValueError
index_lower.append
math.floor
index_upper.append
math.ceil
tuple
zip
math.prod
require_host_memory
np.array
np.asarray
np.isfinite([field_origin, field_upper]).all
np.isfinite
np.zeros
slice
np.arange
enumerate
CoverageField
```

## `OnlinePlanningScene._unknown`

[定义位置](../../../manipulation/online_geometry.py#L704)。

输入签名：

```python
def OnlinePlanningScene._unknown(self, spheres, uncertain=None):
```

直接读取的对象属性：`self._permitted_coverage`、`self.coverage`、`self.unknown_policy`。

输出分支的源码表达式：

```python
(hard.amax(-1), torch.where(retained, unresolved, 0.0).amax(-1))
```

直接调用：

```python
self.coverage.violation
torch.no_grad
self._permitted_coverage.clearance
torch.minimum(centre - _tensor(row['lower_m'], centre), _tensor(row['upper_m'], centre) - centre).amin
torch.minimum
_tensor
sum
abs
float
torch.finfo
torch.maximum
(margin - roundoff).clamp_min
torch.where
torch.relu
torch.zeros_like
torch.isfinite
hard.amax
torch.where(retained, unresolved, 0.0).amax
```

## `OnlinePlanningScene.components`

[定义位置](../../../manipulation/online_geometry.py#L754)。

输入签名：

```python
def OnlinePlanningScene.components(self, robot_spheres, position, quaternion, progress, joint_position=None, joint_names=None, *, trajectory_time=None, active_contact_rule_ids=None, contact_permission_progress=None, owner_poses=None, native_confirmation=None, collision_paths_out=None):
```

直接读取的对象属性：`self.robot_sphere_materials`、`self.contacts`、`self._collision_path_indices`、`self.native_world_collision_components`、`self.gripper`、`self._uncertain_sphere_indices`、`self.robot_self_collision_pairs`、`self.robot_self_collision_padding`、`self._self_collision_pair_tensors`、`self.owners`、`self._fixed_release_poses`、`self._release_pose`、`self._unknown`、`self.background`、`self._permission`、`self.holding_contacts`、`self._initial_support_models`。

输出分支的源码表达式：

```python
costs
```

直接写入的状态或数组位置：`costs['gripper/controller']`、`confirmation['finger_positions']`、`costs['conditional/hand_geometry']`、`moving[uncertain_indices]`、`padded[..., 3]`、`costs['collision/robot/self']`、`costs['conditional/collision/robot/self']`、`costs['collision/robot/self_prismatic']`、`costs['conditional/collision/robot/self_prismatic']`、`owner_mode_keys[owner.owner_id]`、`poses[owner.owner_id]`、`costs[f'unsupported/release_anchor/{owner.owner_id}/{index}']`、`confirmation['owner_poses']`、`costs[f'unsupported/payload/{owner.owner_id}']`、`cover[:, 3]`、`actors[owner.owner_id]`、`costs[f'unknown/forbidden/{actor}']`、`costs[f'conditional/unknown/{actor}']`、`policy['active']`、`confirmation['effective_rules'][rule.rule_id]`、`selected[list(rule.sphere_indices)]`、`costs[f'conditional/finger/{rule.rule_id}']`、`costs[f'unsupported/required_contact_geometry/{rule.rule_id}']`、`costs[f'contact/normal/{rule.rule_id}']`、`costs[f'contact/{term}/{rule.rule_id}']`、`costs[f'contact/required/{rule.rule_id}']`、`selected[list(contact.sphere_indices)]`、`costs[f'initial_support/depth/{relation.relation_id}']`、`confirmation['effective_initial_supports'][relation.relation_id]`、`values[..., list(owner.robot_excluded_spheres)]`。

直接调用：

```python
len
ValueError
set
isinstance
TypeError
frozenset
self.gripper.state
self.gripper.spheres
torch.zeros_like
list
uncertain_hand.to
torch.zeros
spheres.clone
_tensor
self._self_collision_pair_tensors
_self_pair_sum
torch.where
(all_self - known_self).clamp_min
native_confirmation.confirm_self
self.gripper.self_collision_model.components
np.asarray
_tensor(matrix[:3, 3], progress).expand
_tensor(Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]], progress).expand
Rotation.from_matrix(matrix[:3, :3]).as_quat
Rotation.from_matrix
started.to
modes.append
tuple
enumerate
releases.append
self._release_pose
((progress >= relation.interval[1]) & ~found[:, None]).to
owner.pose
torch.ones_like
owner.local_spheres.copy
_world_spheres
actors.items
self._unknown
max_constraint
((body[..., 3] > 0) & self.background.intersects_geometry(body) & ~self.background.supports_centres(body)).to
self.background.intersects_geometry
self.background.supports_centres
sum_collision
self.background
_local_spheres
pair_modes.extend
self._permission
policies.append
pair_modes.append
quaternion_rotate
quaternion_conjugate
native_material_conditions
material_terms.items
owner.query.required_contact
owner.query.contact_query
owner.query
self._initial_support_models.values
model.components
model.collision_values
values.clone
owner.query.intersects_geometry
(intersects & ~owner.query.supports_centres(local) & (local[..., 3] > 0)).to
owner.query.supports_centres
native_confirmation
```

## `OnlinePlanningScene.components.sum_collision`

[定义位置](../../../manipulation/online_geometry.py#L821)。

输入签名：

```python
def OnlinePlanningScene.components.sum_collision(values, name, actor, centres, runtime_radii, mode_keys):
```

直接读取的对象属性：`self._collision_path_device`、`self._collision_path_indices`、`self._collision_path_radii`、`self._owner_collision_path_radii`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`selected[uncertain_indices]`、`costs['conditional/' + name]`、`costs[name]`、`self._collision_path_device[cache_key]`、`collision_paths_out[name]`、`confirmation['targets'][name]`。

直接调用：

```python
torch.zeros
uncertain_hand.unsqueeze
torch.where(uncertainty, values, 0.0).sum
torch.where
torch.where(uncertainty, 0.0, values).sum
values.sum
tuple
torch.tensor
masked.index_select
centres.index_select
runtime_radii.index_select
key.index_select
radii.numel
SphereCollisionPath
```

## `OnlinePlanningScene.components.max_constraint`

[定义位置](../../../manipulation/online_geometry.py#L866)。

输入签名：

```python
def OnlinePlanningScene.components.max_constraint(values, name, actor):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`costs[name]`、`costs['conditional/' + name]`。

直接调用：

```python
torch.zeros_like
torch.where(uncertainty, 0.0, values).amax
torch.where
torch.where(uncertainty, values, 0.0).amax
```

## `OnlinePlanningScene.evaluate`

[定义位置](../../../manipulation/online_geometry.py#L1075)。

输入签名：

```python
def OnlinePlanningScene.evaluate(self, *args, **kwargs):
```

直接读取的对象属性：`self.components`。

输出分支的源码表达式：

```python
torch.stack(hard).sum(0)
```

直接调用：

```python
kwargs.pop
self.components
components_out.update
terms.items
name.startswith
torch.stack(hard).sum
torch.stack
```

## `OnlinePlanningScene.estimated_workspace_allocation`

[定义位置](../../../manipulation/online_geometry.py#L1085)。

输入签名：

```python
def OnlinePlanningScene.estimated_workspace_allocation(self, batch, horizon, robot_coverage_horizon=0, *, collision_paths=False):
```

直接读取的对象属性：`self.background`、`self.owners`、`self._initial_support_models`、`self.coverage`、`self._permitted_coverage`、`self.gripper`、`self._self_collision_pair_arrays`、`self.robot_sphere_geometry`、`self.robot_sphere_materials`、`self._uncertain_sphere_indices`、`self.contacts`、`self.initial_support_relations`、`self.holding_contacts`、`self.unknown_policy`。

输出分支的源码表达式：

```python
dict(cpu_shared=int(cpu_shared), cpu_live=int(cpu_live + upload_scratch), gpu_shared=int(gpu_shared), gpu_live=gpu_live, gpu_retained=0, gpu_graph_live=gpu_live)
```

直接写入的状态或数组位置：`sources[id(query.distance)]`、`observed_sources[id(geometry)]`、`ordinary_caches[id(query._device)]`、`contact_models[id(model)]`、`sources[id(row['distance'])]`、`sources[id(array)]`。

直接调用：

```python
self._initial_support_models.values
id
isinstance
getattr
sum
cover._distance_squared.numel
coverages.values
sources.values
observed_sources.values
ordinary_caches.values
contact_models.values
len
int
max
actor_sizes.items
actor_sizes.values
collision_path_workspace_allocation
_self_pair_chunk_size
dict
```

## `OnlinePlanningScene.query_state`

[定义位置](../../../manipulation/online_geometry.py#L1244)。

输入签名：

```python
def OnlinePlanningScene.query_state(self, spheres, tool_pose, progress, *, joint_position=None, joint_names=None, trajectory_time=0.0, active_contact_rule_ids=frozenset(), owner_poses=None, native_confirmation=None):
```

直接读取的对象属性：`self.components`、`self.unknown_policy`。

输出分支的源码表达式：

```python
report
```

直接写入的状态或数组位置：`report['native_confirmation']`。

直接调用：

```python
frozenset
np.asarray
torch.as_tensor
Rotation.from_matrix(tool[:3, :3]).as_quat
Rotation.from_matrix
torch.tensor
float
self.components
value.item
terms.items
values.items
name.startswith
np.isfinite
any
k.startswith
all
violations.items
name.split
deepcopy
```

## `OnlinePlanningScene.describe`

[定义位置](../../../manipulation/online_geometry.py#L1289)。

输入签名：

```python
def OnlinePlanningScene.describe(self):
```

直接读取的对象属性：`self.reference_frame`、`self.robot_geometry_source`、`self.owners`、`self.unknown_policy`、`self.required_progress_anchors`、`self.observation_report`。

输出分支的源码表达式：

```python
{'model': 'frozen_online_esdf_planar_contact_v2', 'reference_frame': self.reference_frame, 'robot_geometry_source': self.robot_geometry_source, 'owners': [o.owner_id for o in self.owners], 'unknown_policy': deepcopy(self.unknown_policy), 'required_progress_anchors': self.required_progress_anchors, 'observation_report': deepcopy(self.observation_report), 'scope': 'fixed observed fields and registered native face/edge contact; conservative projection bounds; conditional unknown is not free evidence'}
```

直接调用：

```python
deepcopy
```

## `OnlinePlanningScene.to_execution_binding`

[定义位置](../../../manipulation/online_geometry.py#L1296)。

输入签名：

```python
def OnlinePlanningScene.to_execution_binding(self):
```

直接读取的对象属性：`self.owners`、`self.gripper`、`self.reference_frame`、`self.robot_geometry_source`、`self.robot_model`、`self.numerical_position_tolerance_m`、`self.numerical_rotation_tolerance_rad`、`self.robot_sphere_materials`、`self.robot_sphere_geometry`、`self.robot_self_collision_pairs`、`self.robot_self_collision_padding`、`self.background`、`self.contacts`、`self.holding_contacts`、`self.initial_support_relations`、`self._initial_support_models`、`self.has_gripper`、`self.unknown_policy`、`self.robot_collision_bodies`、`self.native_geometry_materials`、`self.grasp_contact_activations`、`self.coverage`、`self.observation_report`。

输出分支的源码表达式：

```python
{'schema': 'online_scene_v3', 'reference_frame': self.reference_frame, 'robot_geometry_source': self.robot_geometry_source, 'robot_model': deepcopy(self.robot_model), 'numerical_position_tolerance_m': self.numerical_position_tolerance_m, 'numerical_rotation_tolerance_rad': self.numerical_rotation_tolerance_rad, 'robot_sphere_materials': list(self.robot_sphere_materials), 'robot_sphere_geometry': deepcopy(self.robot_sphere_geometry), 'robot_self_collision_pairs': deepcopy(self.robot_self_collision_pairs), 'robot_self_collision_padding': list(self.robot_self_collision_padding), 'background': self.background.record(), 'owners': owners, 'contacts': _plain([asdict(r) for r in self.contacts]), 'holding_contacts': _plain([asdict(row) for row in self.holding_contacts]), 'initial_support_relations': [row.record() for row in self.initial_support_relations], 'initial_support_preparation': {name: model.record() for (name, model) in self._initial_support_models.items()}, 'gripper': hand_record, 'has_gripper': self.has_gripper, 'unknown_policy': deepcopy(self.unknown_policy), 'robot_collision_bodies': _plain(self.robot_collision_bodies), 'native_geometry_materials': deepcopy(self.native_geometry_materials), 'grasp_contact_activations': deepcopy(self.grasp_contact_activations), 'coverage': {'known_free': encode_array(self.coverage.known_free), 'origin': self.coverage.origin.tolist(), 'voxel_size': self.coverage.voxel_size, 'observation_ids': list(self.coverage.observation_ids)}, 'observation_report': deepcopy(self.observation_report)}
```

直接写入的状态或数组位置：`row['pose_path']`、`row['attachments']`、`row['query']`、`hand_record['self_collision_geometry']`、`hand_record['prepared_self_collision']`、`hand_record['nominal_model']`。

直接调用：

```python
_plain
getattr
ValueError
asdict
owner.query.record
owners.append
deepcopy
hand.self_collision_model.record
hand.nominal_finger_trajectory.describe
list
self.background.record
row.record
model.record
self._initial_support_models.items
encode_array
self.coverage.origin.tolist
```

## `OnlinePlanningScene.from_binding`

[定义位置](../../../manipulation/online_geometry.py#L1345)。

输入签名：

```python
@classmethod
def OnlinePlanningScene.from_binding(cls, row):
```

直接读取的对象属性：`scene._bind_initial_supports`、`scene.bind_online_robot`。

输出分支的源码表达式：

```python
scene
```

直接写入的状态或数组位置：`record['query']`、`record['pose_path']`、`record['attachments']`、`spec['coupling_matrix']`、`spec['controller']`、`spec['joint_bounds']`、`hand.nominal_finger_trajectory`、`scene.grasp_contact_activations`、`scene.observation_report`。

直接调用：

```python
row.get
ValueError
deepcopy
FrozenESDF.from_record
PosePath
tuple
Attachment
owners.append
CollisionOwner
spec.pop
np.asarray(spec['coupling_matrix'], dtype=float).reshape
np.asarray
len
PrismaticGripper
NominalFingerGeometry.from_mapping
np.array_equal
hand.self_collision_model.validate_record
decode_array
ObservedSpace
cls
ContactRule
HoldingContact
InitialSupportRelation
scene._bind_initial_supports
scene.bind_online_robot
```
