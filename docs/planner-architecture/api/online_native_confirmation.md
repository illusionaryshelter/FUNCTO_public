# `manipulation/online_native_confirmation.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_pose7`

[定义位置](../../../manipulation/online_native_confirmation.py#L39)。

输入签名：

```python
def _pose7(row):
```

输出分支的源码表达式：

```python
pose
```

直接写入的状态或数组位置：`pose[:3, :3]`、`pose[:3, 3]`。

直接调用：

```python
np.asarray
np.eye
Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix
Rotation.from_quat
```

## `_primitive`

[定义位置](../../../manipulation/online_native_confirmation.py#L48)。

输入签名：

```python
def _primitive(row):
```

输出分支的源码表达式：

```python
(shape, vertices, _pose7(row['pose']))
```

直接调用：

```python
np.asarray
coal.Box
list
product
coal.Sphere
coal.Cylinder
coal.Capsule
ValueError
_pose7
```

## `_bounds`

[定义位置](../../../manipulation/online_native_confirmation.py#L67)。

输入签名：

```python
def _bounds(vertices, shape, pose):
```

输出分支的源码表达式：

```python
(pose[:3, 3] - extent, pose[:3, 3] + extent)
_solid_axis_bounds(vertices, shape, pose)
```

直接调用：

```python
isinstance
np.abs
_solid_axis_bounds
```

## `_PrimitiveContactCells`

[定义位置](../../../manipulation/online_native_confirmation.py#L78)。

```python
class _PrimitiveContactCells:
    ...
    ...
```

该类型的直接方法：`__init__`、`nearby`。

## `_PrimitiveContactCells.__init__`

[定义位置](../../../manipulation/online_native_confirmation.py#L86)。

输入签名：

```python
def _PrimitiveContactCells.__init__(self, query, rules):
```

直接读取的对象属性：`self.centers`、`self.rule_bits`、`self.removed_by_rule`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`occupied[start:start + len(flat)]`、`self.centers`、`self.centers.flags.writeable`、`self.h`、`self.shape`、`self.tree`、`self.rule_bits`、`self.removed_by_rule`、`self.removed_by_rule[start:start + len(centers)]`、`self.removed_by_rule.flags.writeable`。

直接调用：

```python
isinstance
TypeError
ValueError
math.prod
require_host_memory
np.empty
torch.no_grad
range
np.arange
min
np.column_stack
np.unravel_index
query.signed_distance(torch.as_tensor(points, dtype=torch.float64)).numpy
query.signed_distance
torch.as_tensor
len
math.sqrt
occupied.reshape
int
np.count_nonzero
np.ascontiguousarray
np.argwhere
float
coal.Box
cKDTree
enumerate
np.zeros
np.asarray
np.abs(axes).sum
np.abs
(np.abs(local) + half_support <= widths + rule.numerical_position_tolerance_m).all
contained.astype
```

## `_PrimitiveContactCells.nearby`

[定义位置](../../../manipulation/online_native_confirmation.py#L139)。

输入签名：

```python
def _PrimitiveContactCells.nearby(self, bounds, active_rules, boundary):
```

直接读取的对象属性：`self.tree`、`self.h`、`self.centers`、`self.rule_bits`、`self.removed_by_rule`。

输出分支的源码表达式：

```python
(np.empty(0, dtype=int), 0)
(np.sort(ids), found - len(ids))
```

直接调用：

```python
np.empty
np.asarray
self.tree.query_ball_point
float
half.max
len
(np.abs(self.centers[ids] - middle) <= half).all
np.abs
sum
np.sort
```

## `NativeCollisionConfirmation`

[定义位置](../../../manipulation/online_native_confirmation.py#L158)。

```python
class NativeCollisionConfirmation:
    ...
    ...
```

该类型的直接方法：`__init__`、`confirm`。

## `NativeCollisionConfirmation.__init__`

[定义位置](../../../manipulation/online_native_confirmation.py#L173)。

输入签名：

```python
def NativeCollisionConfirmation.__init__(self, checker, scene):
```

直接读取的对象属性：`scene.robot_collision_bodies`、`self.sphere_geometry`、`scene.robot_sphere_materials`、`scene.native_world_collision_pairs`、`scene.owners`、`self.owner_geometries`、`self.actor_spheres`、`scene.contacts`、`scene._initial_support_models`、`self.rules`、`scene.native_geometry_materials`、`self.geometry_rules`、`scene.background`、`self.targets`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.checker`、`self.scene`、`self.boundary`、`self.sphere_geometry`、`self.sphere_geometry[index]`、`self.owner_geometries`、`self.actor_spheres`、`self.owner_geometries[owner.owner_id]`、`self.actor_spheres[owner.owner_id]`、`actor_groups[owner.owner_id]`、`actor_materials[owner.owner_id]`、`self.rules`、`self.initial_supports`、`self.geometry_rules`、`self.targets`、`self.targets[name]`、`self.preparation`。

直接调用：

```python
isinstance
TypeError
perf_counter
max
tuple
any
ValueError
geometry_spheres.setdefault(name, set()).update
geometry_spheres.setdefault
set
range
len
scene.native_world_collision_pairs.values
bind_native_owner_geometry
_primitive
geometry.items
self.rules.values
scene.native_geometry_materials.get
check_native_owner_material
self.geometry_rules.setdefault((rule.actor, name, rule.target_owner), set()).add
self.geometry_rules.setdefault
_PrimitiveContactCells
observed.prepare_rules
self.targets.items
```

## `NativeCollisionConfirmation.confirm`

[定义位置](../../../manipulation/online_native_confirmation.py#L250)。

输入签名：

```python
def NativeCollisionConfirmation.confirm(self, *, robot_state, candidates, owner_poses, effective_contact_rule_ids, check_request=None, actor='robot', effective_initial_support_ids=()):
```

直接读取的对象属性：`self.rules`、`self.initial_supports`、`self.actor_spheres`、`self.targets`、`self.scene`、`self.checker`、`self.owner_geometries`、`self.geometry_rules`、`record.update`、`self.boundary`。

输出分支的源码表达式：

```python
{'pairs': pairs, 'cleared_targets': [name for (name, values) in statuses.items() if values and all((value == 'clear' for value in values))], 'collision_targets': [name for (name, values) in statuses.items() if 'collision' in values], 'unresolved_targets': [name for (name, values) in statuses.items() if 'unresolved' in values], 'statistics': stats, 'scope': 'Registered native actor solids against frozen observed source components or calibrated primitives at one state; finite source-component permissions; other scene constraints remain owned by the caller'}
```

直接写入的状态或数组位置：`stats['native_pairs']`、`record['target_representation']`、`stats['nearby_cells']`、`stats['permitted_cells']`、`stats['nearby_source_components']`、`stats['retained_source_components']`、`request.enable_signed_distance`、`stats['coal_queries']`、`record['coal_queries']`、`record['minimum_queried_distance_m']`、`stats['elapsed_seconds']`。

直接调用：

```python
perf_counter
callable
TypeError
frozenset
self.rules.keys
ValueError
self.initial_supports.keys
candidates.items
tuple
sorted
set
any
type
np.eye
_rigid_pose
np.linalg.inv
check_request
_bounds
self.geometry_rules.get
int
pairs.append
record.update
coal.Transform3s
enumerate
np.any
obstacles.append
cells.nearby
len
zip
observed.component_indices
np.setdiff1d
source_obstacles
coal.DistanceRequest
coal.DistanceResult
coal.distance
float
np.isfinite
np.asarray(distance.getNearestPoint1()).tolist
np.asarray
distance.getNearestPoint1
np.asarray(distance.getNearestPoint2()).tolist
distance.getNearestPoint2
statuses.items
all
```

## `NativeCollisionConfirmation.confirm.source_obstacles`

[定义位置](../../../manipulation/online_native_confirmation.py#L366)。

输入签名：

```python
def NativeCollisionConfirmation.confirm.source_obstacles():
```

生成输出的源码表达式：

```python
(other, local_pose, {'source_component_id': int(index), 'source_component_kind': kind, 'source_component_index': int(component)})
```

直接调用：

```python
observed.collision_component
int
```

## `OnlineCollisionConfirmation`

[定义位置](../../../manipulation/online_native_confirmation.py#L413)。

```python
class OnlineCollisionConfirmation:
    ...
    ...
```

该类型的直接方法：`__init__`、`confirm_self`、`__call__`。

## `OnlineCollisionConfirmation.__init__`

[定义位置](../../../manipulation/online_native_confirmation.py#L424)。

输入签名：

```python
def OnlineCollisionConfirmation.__init__(self, validator, scene, check_request=None):
```

直接读取的对象属性：`self.self_report`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.validator`、`self.scene`、`self.engine`、`self.check_request`、`self._self_states`、`self.self_report`、`self.last_report`。

## `OnlineCollisionConfirmation.confirm_self`

[定义位置](../../../manipulation/online_native_confirmation.py#L441)。

输入签名：

```python
def OnlineCollisionConfirmation.confirm_self(self, residual, joint_position, joint_names, trajectory_time, *, source):
```

直接读取的对象属性：`self.check_request`、`self._self_states`、`self.validator`、`self.scene`、`self.self_report`。

输出分支的源码表达式：

```python
residual
result
```

直接写入的状态或数组位置：`self._self_states[key]`、`self.self_report['candidate_states']`、`record['sphere_residuals'][source]`、`result[batch, index]`。

直接调用：

```python
ValueError
torch.isfinite
bool
positive.any().item
positive.any
tuple
len
joint_position.detach().cpu().numpy
joint_position.detach().cpu
joint_position.detach
trajectory_time.detach().cpu().numpy
trajectory_time.detach().cpu
trajectory_time.detach
residual.detach().cpu().numpy
residual.detach().cpu
residual.detach
positive.nonzero(as_tuple=False).detach().cpu().numpy
positive.nonzero(as_tuple=False).detach().cpu
positive.nonzero(as_tuple=False).detach
positive.nonzero
self.check_request
q.tobytes
time_s.tobytes
self._self_states.get
self.validator.check_planning_self
float
int
self.self_report['states'].append
max
record['sphere_residuals'].get
np.isfinite
residual.clone
```

## `OnlineCollisionConfirmation.__call__`

[定义位置](../../../manipulation/online_native_confirmation.py#L496)。

输入签名：

```python
def OnlineCollisionConfirmation.__call__(self, costs, evidence):
```

直接读取的对象属性：`self.scene`、`self.self_report`、`self.engine`、`self.validator`、`self.check_request`、`self.last_report`。

输出分支的源码表达式：

```python
costs
result
```

直接写入的状态或数组位置：`candidates[name]`、`self.last_report`、`self.engine`、`by_actor.setdefault(actor, {})[target]`、`pose[:3, :3]`、`pose[:3, 3]`、`owner_poses[owner]`、`record['actor']`、`record['query_index']`、`record['sphere_residuals']`、`result[name]`、`result[name][batch, index]`、`self.last_report['candidate_states']`、`self.last_report['actor_queries']`。

直接调用：

```python
frozenset
set
ValueError
evidence['targets'].items
bool
(value > 0).any().item
(value > 0).any
value.detach().cpu().numpy
value.detach().cpu
value.detach
np.isfinite(raw).all
np.isfinite
costs[name].detach().cpu().numpy
costs[name].detach().cpu
costs[name].detach
np.where
np.any
NativeCollisionConfirmation
any
evidence['joint_position'].detach().cpu().numpy
evidence['joint_position'].detach().cpu
evidence['joint_position'].detach
evidence['finger_positions'].detach().cpu().numpy
evidence['finger_positions'].detach().cpu
evidence['finger_positions'].detach
p.detach().cpu().numpy
p.detach().cpu
p.detach
q.detach().cpu().numpy
q.detach().cpu
q.detach
evidence['owner_poses'].items
evidence['effective_rules'].items
evidence['effective_initial_supports'].items
np.argwhere
np.logical_or.reduce
candidates.values
dict
candidates.items
by_actor.setdefault
np.flatnonzero(value[batch, index] > 0).tolist
np.flatnonzero
zip
named.update
self.validator.checker.query
poses.items
np.eye
Rotation.from_quat(q[batch, index][[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
by_actor.items
self.engine.confirm
effective.items
effective_supports.items
int
float
costs[name][batch, index].item
self.last_report['states'].append
costs[name].clone
len
```
