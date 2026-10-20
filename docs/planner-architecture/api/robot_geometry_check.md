# `manipulation/robot_geometry_check.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_name`

[定义位置](../../../manipulation/robot_geometry_check.py#L23)。

输入签名：

```python
def _name(value, label):
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

## `_rigid_pose`

[定义位置](../../../manipulation/robot_geometry_check.py#L29)。

输入签名：

```python
def _rigid_pose(value, label='T_reference_base'):
```

输出分支的源码表达式：

```python
pose.astype(float, copy=True)
```

直接调用：

```python
np.asarray
np.isfinite(pose).all
np.isfinite
np.allclose
np.eye
np.isclose
np.linalg.det
ValueError
pose.astype
```

## `NativeSolid`

[定义位置](../../../manipulation/robot_geometry_check.py#L41)。

```python
class NativeSolid:
    name: str
    frame: str
    vertices_frame: np.ndarray
    source: str
```

## `PairQuery`

[定义位置](../../../manipulation/robot_geometry_check.py#L59)。

```python
class PairQuery:
    first: str
    second: str
    policy: str
    source: str
```

## `NativePrimitive`

[定义位置](../../../manipulation/robot_geometry_check.py#L77)。

```python
class NativePrimitive:
    name: str
    frame: str
    shape_kind: str
    dimensions: tuple
    T_frame_geometry: np.ndarray
    source: str
```

## `_convex_solid`

[定义位置](../../../manipulation/robot_geometry_check.py#L94)。

输入签名：

```python
def _convex_solid(vertices):
```

输出分支的源码表达式：

```python
coal.ConvexBase.convexHull(native, True, 'Qt')
```

直接调用：

```python
np.asarray
len
np.isfinite(points).all
np.isfinite
np.linalg.matrix_rank
points.mean
ValueError
coal.StdVec_Vec3s
native.append
coal.ConvexBase.convexHull
```

## `RobotGeometryChecker`

[定义位置](../../../manipulation/robot_geometry_check.py#L108)。

```python
class RobotGeometryChecker:
    ...
    ...
```

该类型的直接方法：`__init__`、`_frame_id`、`geometry_names`、`geometry_sources`、`model_record`、`geometry_depending_on`、`bind_scene`、`query_scene`、`_select_state_layers`、`_consume_layer_selection`、`query`、`_native_pair_distances`。

## `RobotGeometryChecker.__init__`

[定义位置](../../../manipulation/robot_geometry_check.py#L117)。

输入签名：

```python
def RobotGeometryChecker.__init__(self, urdf_path, *, base_frame, geometries):
```

直接读取的对象属性：`self.urdf_path`、`self.model`、`self.joint_names`、`self._q_columns`、`record.name`、`self._geometries`、`self._frame_id`、`record.frame`、`record.vertices_frame`、`self._solid_vertices`、`record.dimensions`、`record.shape_kind`、`record.T_frame_geometry`、`record.source`、`self._geometry_model`、`self._request`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.urdf_path`、`self.base_frame`、`self._model_urdf`、`self._geometry_model`、`self.model`、`self._data`、`self.joint_names`、`self._q_columns`、`self._q_columns[name]`、`self.joint_limits`、`self.joint_velocity_limits`、`self._geometries`、`self._solid_vertices`、`self._solid_vertices[name]`、`self._geometries[name]`、`self._request`、`self._request.enable_signed_distance`、`self.numerics`、`self._native_distance_batches`、`self.native_query_storage_nbytes`。

直接调用：

```python
any
callable
getattr
ImportError
str
Path(urdf_path).resolve
Path
_name
ET.parse(self.urdf_path).getroot
ET.parse
ET.tostring
tree.findall
len
set
ValueError
node.find
pin.buildModelFromUrdf
self.model.createData
tuple
self.model.getJointId
int
self._q_columns.values
range
np.isfinite(lower).all
np.isfinite
np.isfinite(upper).all
np.any
float
self._q_columns.items
self._frame_id
isinstance
_convex_solid
np.eye
np.asarray(record.vertices_frame, float).copy
np.asarray
{'box': 3, 'cylinder': 2}.get
np.isfinite(dimensions).all
coal.Box
coal.Cylinder
_rigid_pose
_box_vertices
identity.update
self._solid_vertices[name].tolist
list
local_pose.tolist
self._geometry_model.append
coal.DistanceRequest
min
sum
name.encode
```

## `RobotGeometryChecker._frame_id`

[定义位置](../../../manipulation/robot_geometry_check.py#L231)。

输入签名：

```python
def RobotGeometryChecker._frame_id(self, name):
```

直接读取的对象属性：`self.model`。

输出分支的源码表达式：

```python
index
```

直接调用：

```python
_name
int
self.model.getFrameId
ValueError
```

## `RobotGeometryChecker.geometry_names`

[定义位置](../../../manipulation/robot_geometry_check.py#L240)。

输入签名：

```python
@property
def RobotGeometryChecker.geometry_names(self):
```

直接读取的对象属性：`self._geometries`。

输出分支的源码表达式：

```python
tuple(self._geometries)
```

直接调用：

```python
tuple
```

## `RobotGeometryChecker.geometry_sources`

[定义位置](../../../manipulation/robot_geometry_check.py#L245)。

输入签名：

```python
@property
def RobotGeometryChecker.geometry_sources(self):
```

直接读取的对象属性：`self._geometries`。

输出分支的源码表达式：

```python
{name: value[2] for (name, value) in self._geometries.items()}
```

直接调用：

```python
self._geometries.items
```

## `RobotGeometryChecker.model_record`

[定义位置](../../../manipulation/robot_geometry_check.py#L250)。

输入签名：

```python
def RobotGeometryChecker.model_record(self):
```

直接读取的对象属性：`self._model_urdf`、`self._geometry_model`。

输出分支的源码表达式：

```python
{'urdf_xml': self._model_urdf, 'geometries': deepcopy(self._geometry_model)}
```

直接调用：

```python
deepcopy
```

## `RobotGeometryChecker.geometry_depending_on`

[定义位置](../../../manipulation/robot_geometry_check.py#L257)。

输入签名：

```python
def RobotGeometryChecker.geometry_depending_on(self, joint_names):
```

直接读取的对象属性：`self.joint_names`、`self._geometries`、`self.model`、`self._frame_id`、`result.append`。

输出分支的源码表达式：

```python
tuple(result)
```

直接调用：

```python
set
ValueError
self._geometries.items
int
self._frame_id
str
result.append
tuple
```

## `RobotGeometryChecker.bind_scene`

[定义位置](../../../manipulation/robot_geometry_check.py#L273)。

输入签名：

```python
def RobotGeometryChecker.bind_scene(self, scene_binding, *, previous_checker=None, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
```

直接读取的对象属性：`self._scene_body_groups`、`self._scene_layers`、`self._scene_gaps`、`self._scene_owners`、`self._geometries`、`self._scene_native_aliases`、`self._scene_shapes`、`self._robot_owner_permissions`、`self._gated_contact_rule_ids`、`self.native_query_storage_nbytes`、`self._scene_binding`、`self.scene_storage_nbytes`、`self._request`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`previous_shapes[row['owner'], row['layer'], row['name']]`、`previous_records[owner, layer, record['name']]`、`self._scene_binding`、`self._scene_owners`、`self._scene_shapes`、`self._has_observed_tsdf`、`self._scene_gaps`、`self._scene_native_aliases`、`self._scene_release_predictions`、`self._scene_surface_distance_cache`、`self._scene_component_hints`、`self._scene_layers`、`self._scene_body_groups`、`self._scene_layers[owner_id]`、`path['motion_model']`、`self._scene_owners[owner_id]`、`self._scene_native_aliases[owner_id]`、`self._robot_owner_permissions`、`self._gated_contact_rule_ids`、`self.scene_storage_nbytes`、`self.scene_query_workspace_nbytes`。

直接调用：

```python
isinstance
partition.get
layers.items
require_host_memory
_owned_record_bytes
ObservedTsdfGeometry.pending_storage_total_nbytes
deepcopy
tuple
scene_binding.get
any
validate_body_groups
len
set
ValueError
float
np.isfinite
_pose7
FineRegion
geometry['layers'].items
self._scene_gaps.append
MotionModel
PosePath
previous_shapes.get
_scene_shape
self._scene_shapes.append
RobotOwnerContactPermission
self._geometries.keys
grasp_contact_rule_ids
self._gated_contact_rule_ids.intersection
condition_ids.add
self._gated_contact_rule_ids.update
sum
np.asarray
component_batch_workspace_bytes
surface_batch_workspace_bytes
```

## `RobotGeometryChecker.query_scene`

[定义位置](../../../manipulation/robot_geometry_check.py#L453)。

输入签名：

```python
def RobotGeometryChecker.query_scene(self, robot_state, *, tool_pose, progress, finger_positions, state_kind, owner_poses=None, nominal_owner_poses=None, active_robot_owner_permissions=frozenset(), robot_owner_permission_progress=None, active_contact_rule_ids=frozenset(), layer_selection=None, layer_selection_index=(0, 0), select_layers=False):
```

直接读取的对象属性：`self._gated_contact_rule_ids`、`self._scene_binding`、`self._robot_owner_permissions`、`self._scene_owners`、`self._scene_gaps`、`self._scene_release_predictions`、`self._scene_native_aliases`、`self._geometries`、`self._solid_vertices`、`self._scene_shapes`、`self._scene_layers`、`self._select_state_layers`、`self._consume_layer_selection`、`self._request`、`self._scene_surface_distance_cache`、`self.numerics`、`self._scene_component_hints`、`result.getContacts`、`result.numContacts`、`self._scene_body_groups`、`self._has_observed_tsdf`。

输出分支的源码表达式：

```python
{'passed': not violations and (not gaps), 'violations': violations, 'query_gaps': gaps, 'unresolved_assumptions': unresolved, 'pairs': rows, 'state_kind': state_kind, 'required_contact_checks': required_checks, 'unpermitted_clearance': clearance, 'component_searches': list(component_searches.values()), 'layer_selection_checks': selection_checks, 'layer_selection': selection_snapshot, 'active_robot_owner_permissions': sorted((rule.rule_id for rule in permitted_rules)), 'active_contact_rule_ids': sorted((rule['rule_id'] for rule in contact_rules if rule['rule_id'] in self._gated_contact_rule_ids and rule['interval'][0] <= progress <= rule['interval'][1])), 'robot_owner_permission_progress': float(permission_progress), 'progress': float(progress), 'observed_scene_checked': not gaps and (not uncertain_owners) and (not unmodelled_geometries), 'supplied_scene_models_checked': not gaps, 'complete_scene_geometry': not self._has_observed_tsdf and all((owner['geometry_complete'] for (owner, _) in self._scene_owners.values())), 'continuous_collision_certified': False, 'unknown_space_certified': False, 'scope': 'source solids and observed TSDF zero/negative components at this state; complete adapted contact intersections'}
```

直接写入的状态或数组位置：`poses[identity]`、`offset[:3, :3]`、`offset[:3, 3]`、`pose[:3, 3]`、`pose[:3, :3]`、`rows[summary['index']]['component_summary']['tested']`、`row[label + '_tsdf_component']`、`request.enable_contact`、`request.num_max_contacts`、`row['contacts']`、`row['contact_report_truncated']`、`row['permission']`、`row['permission_scope']`、`row['permission_authorization']`、`row['boundary_intersection_check']`、`row['relation']`、`evidence['raw_solid_overlap']`、`evidence['measurement_allowance_m']`、`evidence['margin_geometry']`、`clearance['omitted_finite_patch_subset_count']`、`clearance['omitted_ambiguous_surface_count']`。

直接调用：

```python
frozenset
np.isfinite
ValueError
hasattr
isinstance
any
set
self._scene_owners.keys
deepcopy
np.eye
self._scene_owners.items
unresolved.append
enumerate
_rigid_pose
uncertain_owners.add
Rotation.from_quat(np.asarray(relation['quaternion_tool_owner'])[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
np.asarray
self._scene_release_predictions.get
gaps.append
prediction['pose'].copy
path.sample
torch.tensor
np.clip
p.numpy
Rotation.from_quat(q.numpy()[[1, 2, 3, 0]]).as_matrix
q.numpy
robot_state.get
self._scene_native_aliases.values
self._geometries.items
actors.append
materials.get
self._scene_native_aliases.get
np.allclose
current.update
targets.append
self._select_state_layers
self._consume_layer_selection
SurfaceDistanceBatch
max
_scene_component_pairs
len
second.get
tuple
decisions.get
rows.append
_record_unpermitted_clearance
coal.Transform3s
coal.DistanceResult
coal.distance
float
RuntimeError
component_summaries.get
np.asarray(distance.getNearestPoint1()).tolist
distance.getNearestPoint1
np.asarray(distance.getNearestPoint2()).tolist
distance.getNearestPoint2
np.asarray(distance.normal).tolist
list
row.update
coal.CollisionRequest
coal.CollisionResult
coal.collide
np.asarray(contact.pos).tolist
np.asarray(contact.normal).tolist
result.getContacts
result.numContacts
abs
_overlap_within_region
_world_vertices
np.zeros
overlap.get
_feature_region
evidence.update
evidence.get
evidence.items
row['contact_region_checks'].append
violations.append
current.get
_summarize_component_result
materials.items
name.startswith
sorted
next
iter
_solid_intersects_region
check.get
required_checks.append
row.get
limitations.append
clearance.update
component_searches.values
all
self._scene_owners.values
```

## `RobotGeometryChecker._select_state_layers`

[定义位置](../../../manipulation/robot_geometry_check.py#L1046)。

输入签名：

```python
def RobotGeometryChecker._select_state_layers(self, robot_state, actors, poses):
```

直接读取的对象属性：`self._scene_layers`、`self._scene_binding`、`self._scene_body_groups`、`self._scene_owners`。

输出分支的源码表达式：

```python
{}
result
```

直接写入的状态或数组位置：`world[group['sphere_indices'], :3]`、`world[group['sphere_indices'], 3]`、`actor_spheres[identity]`、`result[f'{actor}/{target}']`。

直接调用：

```python
len
np.empty
np.asarray
np.isfinite(local).all
np.isfinite
np.any
ValueError
_rigid_pose
sorted
np.column_stack
float
list
range
np.linalg.inv
actor_spheres.items
region.select(torch.from_numpy(local)[None, None], groups).record
region.select
torch.from_numpy
```

## `RobotGeometryChecker._consume_layer_selection`

[定义位置](../../../manipulation/robot_geometry_check.py#L1095)。

输入签名：

```python
def RobotGeometryChecker._consume_layer_selection(self, selections, index, actors, poses, gaps, unmodelled):
```

直接读取的对象属性：`self._scene_layers`、`self._scene_body_groups`。

输出分支的源码表达式：

```python
({}, [], {})
(decisions, checks, snapshot)
```

直接写入的状态或数组位置：`snapshot[key]`、`decisions[actor, row['name'], target]`。

直接调用：

```python
isinstance
len
any
type
ValueError
self._scene_layers.items
np.linalg.inv
sorted
selections.get
gaps.append
set
decode_array
np.dtype
np.isfinite(envelope).all
np.isfinite
np.any
layer['region'].validate_selection_envelope
torch.tensor
slice_layer_selection
tuple
enumerate
bool
envelope[index[0], index[1], column].tolist
lower[index[0], index[1], column].tolist
upper[index[0], index[1], column].tolist
_solid_axis_bounds
np.stack
np.min
np.max
max
float
item.update
actual.tolist
checks.append
```

## `RobotGeometryChecker.query`

[定义位置](../../../manipulation/robot_geometry_check.py#L1210)。

输入签名：

```python
def RobotGeometryChecker.query(self, *, joint_names, joint_positions, T_reference_base, pairs, frames=()):
```

直接读取的对象属性：`self.joint_names`、`self.model`、`self._q_columns`、`self.joint_limits`、`self.base_frame`、`self._geometries`、`self._frame_id`、`self._data`、`self._native_pair_distances`、`self.numerics`、`result.min_distance`、`result.getNearestPoint1`、`result.getNearestPoint2`、`result.normal`。

输出分支的源码表达式：

```python
{'joint_names': list(names), 'joint_positions': values.astype(float).tolist(), 'joint_limit_violations': violations, 'frame_poses': {name: pose.tolist() for (name, pose) in poses.items()}, 'geometry_poses': {name: pose.tolist() for (name, pose) in geometry_poses.items()}, 'geometry_sources': {name: value[2] for (name, value) in self._geometries.items()}, 'numerics': dict(self.numerics), 'pairs': rows}
```

直接写入的状态或数组位置：`q[self._q_columns[name]]`、`violations[name]`。

直接调用：

```python
tuple
any
isinstance
len
set
ValueError
np.asarray
np.isfinite(values).all
np.isfinite
_rigid_pose
pin.neutral
zip
float
dict.fromkeys
self._geometries.values
self._frame_id
frozenset
_name
identities.add
pair_list.append
pin.forwardKinematics
pin.updateFramePlacements
frame_ids.items
self._geometries.items
self._native_pair_distances
max
result.getNearestPoint1
result.getNearestPoint2
np.isfinite(np.concatenate(([distance], first, second, normal))).all
np.concatenate
RuntimeError
rows.append
first.tolist
second.tolist
normal.tolist
np.linalg.norm
list
values.astype(float).tolist
values.astype
pose.tolist
poses.items
geometry_poses.items
dict
```

## `RobotGeometryChecker._native_pair_distances`

[定义位置](../../../manipulation/robot_geometry_check.py#L1291)。

输入签名：

```python
def RobotGeometryChecker._native_pair_distances(self, pairs, poses):
```

直接读取的对象属性：`self._native_distance_batches`、`self._geometries`、`self._request`。

生成输出的源码表达式：

```python
data.distanceResults[index]
```

直接写入的状态或数组位置：`self._native_distance_batches`、`data.distanceRequests[index]`、`data.distance_functors`、`data.oMg[index]`。

直接调用：

```python
tuple
range
len
dict.fromkeys
enumerate
pin.GeometryModel
model.addGeometryObject
pin.GeometryObject
pin.SE3.Identity
model.addCollisionPair
pin.CollisionPair
pin.GeometryData
data.distance_functors.copy
blocks.append
pin.SE3
pin.computeDistances
```

## `_pose7`

[定义位置](../../../manipulation/robot_geometry_check.py#L1352)。

输入签名：

```python
def _pose7(value):
```

输出分支的源码表达式：

```python
pose
```

直接写入的状态或数组位置：`pose[:3, :3]`、`pose[:3, 3]`。

直接调用：

```python
np.asarray
np.isfinite(row).all
np.isfinite
np.isclose
np.linalg.norm
ValueError
np.eye
Rotation.from_quat(row[[4, 5, 6, 3]]).as_matrix
Rotation.from_quat
```

## `_box_vertices`

[定义位置](../../../manipulation/robot_geometry_check.py#L1364)。

输入签名：

```python
def _box_vertices(dimensions):
```

输出分支的源码表达式：

```python
np.asarray(list(product((-1.0, 1.0), repeat=3))) * np.asarray(dimensions) / 2
```

直接调用：

```python
np.asarray
list
product
```

## `_world_vertices`

[定义位置](../../../manipulation/robot_geometry_check.py#L1369)。

输入签名：

```python
def _world_vertices(vertices, pose):
```

输出分支的源码表达式：

```python
{'tsdf': vertices, 'pose': pose}
{'tsdf_component': np.asarray(vertices['tsdf_component']) @ pose[:3, :3].T + pose[:3, 3]}
None if vertices is None else np.asarray(vertices) @ pose[:3, :3].T + pose[:3, 3]
```

直接调用：

```python
isinstance
np.asarray
```

## `_owned_record_bytes`

[定义位置](../../../manipulation/robot_geometry_check.py#L1381)。

输入签名：

```python
def _owned_record_bytes(value):
```

输出分支的源码表达式：

```python
size
```

直接调用：

```python
getsizeof
isinstance
sum
_owned_record_bytes
value.items
```

## `_record_unpermitted_clearance`

[定义位置](../../../manipulation/robot_geometry_check.py#L1395)。

输入签名：

```python
def _record_unpermitted_clearance(summary, distance, margin, actor_geometry, target_owner, target_geometry, distance_scope, *, measured_count=1):
```

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`summary['measured_count']`、`summary['minimum_recorded_distance_m']`、`summary['minimum_pair']`、`summary['minimum_recorded_adjusted_distance_m']`、`summary['minimum_adjusted_pair']`。

## `_summarize_component_result`

[定义位置](../../../manipulation/robot_geometry_check.py#L1428)。

输入签名：

```python
def _summarize_component_result(key, rows, previous_count, issues, issue_counts, summaries, stopped, *, measured_count=1):
```

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`summaries[key]`、`rows[index]`、`rows[index]['component_summary']`、`issue['pair_index']`、`summary['conditional_count']`、`summary['conditional_index']`、`unresolved[summary['conditional_index']]['component_issue_count']`。

直接调用：

```python
len
any
zip
rows.pop
int
current.get
stopped.add
```

## `_transform_bounds`

[定义位置](../../../manipulation/robot_geometry_check.py#L1476)。

输入签名：

```python
def _transform_bounds(bounds, pose):
```

输出分支的源码表达式：

```python
(points.min(axis=0), points.max(axis=0))
```

直接调用：

```python
_box_vertices
np.asarray
_world_vertices
points.min
points.max
```

## `_component_rows`

[定义位置](../../../manipulation/robot_geometry_check.py#L1484)。

输入签名：

```python
def _component_rows(row, world_bounds, margin):
```

输出分支的源码表达式：

```python
None
```

生成输出的源码表达式：

```python
row
{**row, 'shape': shape, 'pose': row['pose'] @ local, 'vertices': {'tsdf_component': points}, 'tsdf_component_id': (kind, index)}
```

直接调用：

```python
isinstance
_transform_bounds
np.linalg.inv
geometry.iter_collision_components
```

## `_scene_component_pairs`

[定义位置](../../../manipulation/robot_geometry_check.py#L1508)。

输入签名：

```python
def _scene_component_pairs(actor, targets, decisions, boundary, evidence_rows, request, visited, stopped, surface_queries, *, hints, statistics):
```

生成输出的源码表达式：

```python
(first, second, key)
```

直接调用：

```python
selected_body_pairs
max
isinstance
surface_queries.consume
evidence_rows.append
witness_first.tolist
witness_second.tolist
_solid_axis_bounds
_component_rows
_target_component_runs
```

## `_target_component_runs`

[定义位置](../../../manipulation/robot_geometry_check.py#L1557)。

输入签名：

```python
def _target_component_runs(first, target, world_bounds, margin, request, *, hints, statistics):
```

输出分支的源码表达式：

```python
None
```

生成输出的源码表达式：

```python
target
{**target, 'shape': shape, 'pose': pose, 'vertices': {'tsdf_component': points}, 'tsdf_component_id': (kind, component_id), '_prequeried_distance': distance, '_component_count': measured}
```

直接写入的状态或数组位置：`hints[identity]`。

直接调用：

```python
isinstance
_transform_bounds
np.linalg.inv
hints.get
statistics.setdefault
float
iter_component_distances
```

## `_solid_axis_bounds`

[定义位置](../../../manipulation/robot_geometry_check.py#L1600)。

输入签名：

```python
def _solid_axis_bounds(vertices, shape, pose):
```

输出分支的源码表达式：

```python
_transform_bounds(vertices.bounds, pose)
(points.min(axis=0), points.max(axis=0))
None
(pose[:3, 3] - extent, pose[:3, 3] + extent)
```

直接调用：

```python
isinstance
_transform_bounds
_world_vertices
points.min
points.max
np.linalg.norm
np.abs
np.full
```

## `_scene_shape`

[定义位置](../../../manipulation/robot_geometry_check.py#L1622)。

输入签名：

```python
def _scene_shape(record, *, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
```

输出分支的源码表达式：

```python
(None, None, _pose7(record['pose']))
(shape, shape, _pose7(record['pose']))
(shape, vertices, _pose7(record['pose']))
```

直接调用：

```python
ObservedTsdfGeometry
_pose7
np.asarray
_convex_solid
{'cuboid': 3, 'cylinder': 2, 'sphere': 1}.get
np.isfinite(sizes).all
np.isfinite
np.any
ValueError
{'cuboid': coal.Box, 'cylinder': coal.Cylinder, 'sphere': coal.Sphere}[kind]
_box_vertices
```

## `_feature_region_halfspaces`

[定义位置](../../../manipulation/robot_geometry_check.py#L1649)。

输入签名：

```python
def _feature_region_halfspaces(rule, actor_pose, target_pose, finger_positions):
```

输出分支的源码表达式：

```python
(None, None, 'missing_named_finger_geometry')
(None, None, 'material_normal_outside_permission')
(region, np.asarray(derivatives), None)
```

直接调用：

```python
np.asarray
np.einsum
np.zeros
_world_vertices
float
np.arctan2
np.linalg.norm
np.cross
rule.get
ValueError
np.finfo
max
np.abs(world).max
np.abs
abs
rows.append
bounds.append
derivatives.append
np.linalg.svd
ConvexHull
enumerate
```

## `_feature_region`

[定义位置](../../../manipulation/robot_geometry_check.py#L1752)。

输入签名：

```python
def _feature_region(rule, actor_pose, target_pose, finger_positions):
```

输出分支的源码表达式：

```python
(None, reason)
(None, 'finite_material_patch_gap_region_empty')
(None, 'finite_material_patch_region_query_incomplete')
(region, None)
```

直接调用：

```python
_feature_region_halfspaces
linprog
np.zeros
```

## `_solid_intersects_region`

[定义位置](../../../manipulation/robot_geometry_check.py#L1775)。

输入签名：

```python
def _solid_intersects_region(vertices, region):
```

直接读取的对象属性：`result.status`、`result.success`、`result.x`。

输出分支的源码表达式：

```python
{'completed': False, 'reason': 'required_curved_target_needs_complete_region_adapter'}
{'completed': bound.status == 2, 'intersects': False, 'reason': 'required_region_empty' if bound.status == 2 else 'required_region_bounds_incomplete'}
{**result, 'component_queries': count, 'geometry_scope': 'observed_tsdf_zero_set'}
{'completed': True, 'intersects': False, 'component_queries': count, 'geometry_scope': 'observed_tsdf_zero_set'}
convex_component_intersects_region(vertices['tsdf_component'], region)
{'completed': True, 'intersects': False, 'method': 'complete_target_region_feasibility'}
{'completed': False, 'reason': 'required_target_region_query_incomplete', 'solver_status': int(result.status)}
{'completed': True, 'intersects': True, 'witness': result.x.tolist(), 'method': 'complete_target_region_feasibility', 'numerical_certificate': False}
```

直接写入的状态或数组位置：`(lower if sign == 1.0 else upper)[column]`。

直接调用：

```python
isinstance
np.empty
enumerate
np.eye
linprog
_transform_bounds
np.linalg.inv
geometry.iter_components
convex_component_intersects_region
_world_vertices
ConvexHull
np.zeros
np.concatenate
int
result.x.tolist
```

## `_overlap_within_region`

[定义位置](../../../manipulation/robot_geometry_check.py#L1821)。

输入签名：

```python
def _overlap_within_region(first_vertices, second_vertices, region, *, first_margin_m=0.0, second_margin_m=0.0):
```

直接读取的对象属性：`result.success`、`result.status`、`result.x`。

输出分支的源码表达式：

```python
{'completed': False, 'reason': 'curved_contact_requires_complete_local_region_adapter'}
component_component_overlap_within_region(first_vertices['tsdf_component'], second_vertices['tsdf_component'], region, first_margin_m=first_margin_m, second_margin_m=second_margin_m)
component_component_overlap_within_region(solid, component, region, first_margin_m=solid_margin, second_margin_m=component_margin)
convex_component_overlap_within_region((planes[:, :3], -planes[:, 3]), component, region, actor_margin=solid_margin)
{'completed': True, 'contained': True, 'intersection_empty': True, 'solver_status': int(result.status), 'method': 'empty_complete_convex_intersection', 'numerical_certificate': False}
{'completed': False, 'reason': 'overlap_support_query_incomplete', 'solver_status': int(result.status)}
{'completed': True, 'contained': False, 'maximum_constraint_excess_m': value, 'outside_witness': result.x.tolist(), 'method': 'whole_convex_intersection_support'}
{'completed': True, 'contained': True, 'maximum_constraint_excess_m': max(maxima), 'method': 'whole_convex_intersection_support', 'numerical_certificate': False}
```

直接调用：

```python
isinstance
component_component_overlap_within_region
ConvexHull
convex_component_overlap_within_region
np.concatenate
np.linalg.norm
zip
linprog
int
float
maxima.append
result.x.tolist
max
```
