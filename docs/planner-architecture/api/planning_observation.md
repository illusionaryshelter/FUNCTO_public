# `manipulation/planning_observation.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_views`

[定义位置](../../../manipulation/planning_observation.py#L47)。

输入签名：

```python
def _views(views, reference_frame, depth_range_m):
```

直接读取的对象属性：`record.get`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[identity]`。

直接调用：

```python
_text
np.asarray
np.isfinite(limits).all
np.isfinite
ValueError
isinstance
len
views.items
record.get
min
_intrinsics(record['K']).copy
_intrinsics
abs
np.where
_pose_array(record['T_output_camera'], 'T_output_camera').copy
_pose_array
```

## `_mapper_intrinsics`

[定义位置](../../../manipulation/planning_observation.py#L79)。

输入签名：

```python
def _mapper_intrinsics(matrix):
```

输出分支的源码表达式：

```python
native
```

直接写入的状态或数组位置：`native[:2, 2]`。

直接调用：

```python
_intrinsics(matrix).copy
_intrinsics
```

## `_rectangle_min_depth`

[定义位置](../../../manipulation/planning_observation.py#L97)。

输入签名：

```python
def _rectangle_min_depth(depth, lower_xy, upper_xy):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[selected]`。

直接调用：

```python
np.empty
len
np.floor(np.log2(sizes)).astype
np.floor
np.log2
np.unique
(blocks == block).all
map
minimum_filter
np.minimum
```

## `_known_free_voxels`

[定义位置](../../../manipulation/planning_observation.py#L128)。

输入签名：

```python
def _known_free_voxels(views, *, ray_depths, origin, shape, voxel_size, depth_error_m):
```

输出分支的源码表达式：

```python
known.reshape(shape)
```

直接写入的状态或数组位置：`known[indices[inside][furthest[inside] < low]]`。

直接调用：

```python
tuple
np.asarray
len
any
type
math.prod
np.isfinite(origin).all
np.isfinite
ValueError
isinstance
set
views.items
np.isfinite(rays).all
np.any
max
views.values
require_host_memory
int
min
np.zeros
np.indices((2, 2, 2)).reshape
np.indices
np.finfo
range
np.arange
np.unravel_index
np.column_stack
np.abs
(camera[..., 2] - camera_error[:, None, 2] > 1e-09).all
np.isfinite(camera).all
np.isfinite(camera_error).all
np.nextafter
(uv - uv_error).min
(uv + uv_error).max
(camera[..., 2] + camera_error[:, None, 2]).max
np.isfinite(lower).all
np.isfinite(upper).all
(lower >= 0).all
(upper <= [width - 1, height - 1]).all
np.rint(lower[inside]).astype
np.rint
np.rint(upper[inside]).astype
_rectangle_min_depth
known.reshape
```

## `_pose_evidence`

[定义位置](../../../manipulation/planning_observation.py#L218)。

输入签名：

```python
def _pose_evidence(record, views, reference_frame, label):
```

直接读取的对象属性：`record.get`。

输出分支的源码表达式：

```python
{'source': source, 'observation_ids': list(identities), 'max_point_error_m': float(error)}
```

直接调用：

```python
isinstance
record.get
ValueError
_text
len
set
np.isfinite
list
float
```

## `_robot_solid_poses`

[定义位置](../../../manipulation/planning_observation.py#L234)。

输入签名：

```python
def _robot_solid_poses(current_robot, links, views, reference_frame):
```

输出分支的源码表达式：

```python
(poses, evidence)
```

直接写入的状态或数组位置：`pose[:3, 3]`、`pose[:3, :3]`、`poses[link]`、`spheres[:, :3]`。

直接调用：

```python
_pose_evidence
current_robot.get
isinstance
TypeError
np.asarray
len
ValueError
tuple
np.isfinite(position).all
np.isfinite
set
model.config.kinematics_config.joint_limits.position.detach().cpu().numpy
model.config.kinematics_config.joint_limits.position.detach().cpu
model.config.kinematics_config.joint_limits.position.detach
np.any
_pose_array
_text
locked.position.detach().cpu().numpy().reshape
locked.position.detach().cpu().numpy
locked.position.detach().cpu
locked.position.detach
np.isfinite(values).all
float
zip
Kinematics
torch.no_grad
model.device_cfg.to_device
fk.compute_kinematics
JointState.from_position
list
state.tool_poses.position[0, 0].detach().cpu().numpy().copy
state.tool_poses.position[0, 0].detach().cpu().numpy
state.tool_poses.position[0, 0].detach().cpu
state.tool_poses.position[0, 0].detach
state.tool_poses.quaternion[0, 0].detach().cpu().numpy().copy
state.tool_poses.quaternion[0, 0].detach().cpu().numpy
state.tool_poses.quaternion[0, 0].detach().cpu
state.tool_poses.quaternion[0, 0].detach
state.robot_spheres[0, 0].detach().cpu().numpy().astype
state.robot_spheres[0, 0].detach().cpu().numpy
state.robot_spheres[0, 0].detach().cpu
state.robot_spheres[0, 0].detach
model.tool_frames.index
np.eye
Rotation.from_quat(wxyz[index, [1, 2, 3, 0]]).as_matrix
Rotation.from_quat
evidence.update
position.tolist
base.tolist
int
spheres.tolist
pose.tolist
poses.items
```

## `_known_solid_evidence`

[定义位置](../../../manipulation/planning_observation.py#L301)。

输入签名：

```python
def _known_solid_evidence(known_solids, owner_frames, current_robot, views, reference_frame):
```

输出分支的源码表达式：

```python
((), {'solids': [], 'owners': {}, 'robot': None})
(records, {'solids': summaries, 'owners': owners, 'robot': robot})
```

直接写入的状态或数组位置：`evidence['T_output_owner']`、`owners[body]`、`poses['owner', body]`、`errors['owner', body]`、`poses['robot_link', link]`、`errors['robot_link', link]`、`row['T_output_solid']`、`row['max_pose_error_m']`、`row['numerical_inward_margin_m']`。

直接调用：

```python
isinstance
len
ValueError
known_solids.items
_text
supplied.get
TypeError
np.isfinite(mesh.vertices).all
np.isfinite
_pose_array
trimesh.Trimesh
np.asarray(mesh.vertices).copy
np.asarray
np.asarray(mesh.faces).copy
records.append
float
_pose_evidence
frame.get
view.get
bool
np.any
pose.tolist
_robot_solid_poses
sorted
link_poses.items
np.abs
np.abs(row['mesh'].vertices).max
max
np.linalg.norm
np.finfo
np.abs(row['T_output_solid']).max
summaries.append
row.items
summaries[-1].update
row['T_body_solid'].tolist
row['T_output_solid'].tolist
```

## `_add_known_solid_voxels`

[定义位置](../../../manipulation/planning_observation.py#L401)。

输入签名：

```python
def _add_known_solid_voxels(known, records, *, origin, voxel_size):
```

输出分支的源码表达式：

```python
0
added
```

直接写入的状态或数组位置：`known[z[contained], y[contained], x[contained]]`。

直接调用：

```python
known.copy
np.asarray
math.sqrt
np.array
np.isfinite(world_box).all
np.isfinite
ValueError
np.ceil
world_box.min
np.floor
world_box.max
np.any
np.maximum(low_float, 0).astype
np.maximum
np.minimum(high_float, shape_xyz - 1).astype
np.minimum
tuple
max
min
len
range
math.prod
np.arange
np.unravel_index
np.column_stack
trimesh.proximity.signed_distance
int
np.count_nonzero
```

## `observed_audit_space_from_rgbd`

[定义位置](../../../manipulation/planning_observation.py#L466)。

输入签名：

```python
def observed_audit_space_from_rgbd(views, *, origin, shape, voxel_size, depth_error_m, reference_frame, depth_range_m, known_solids=None, owner_frames=None, current_robot=None):
```

输出分支的源码表达式：

```python
ObservedSpace(known, origin=origin, voxel_size=voxel_size, observation_ids=tuple(normalized), reference_frame=reference_frame, occupancy_evidence=evidence, known_free=known_free)
```

直接调用：

```python
_views
_known_free_voxels
normalized.items
known.copy
_known_solid_evidence
_add_known_solid_voxels
ObservedSpace
tuple
```

## `_audit_partition_masks`

[定义位置](../../../manipulation/planning_observation.py#L509)。

输入签名：

```python
def _audit_partition_masks(views, owner_ids):
```

输出分支的源码表达式：

```python
masks
```

直接写入的状态或数组位置：`masks[name]`、`masks[name][identity]`、`observed[identity]`、`masks[name][None]`。

直接调用：

```python
views.items
np.asarray
ValueError
isinstance
set
robot.copy
np.any
mask.copy
int
np.count_nonzero
any
observed.values
```

## `_self_filtered_views`

[定义位置](../../../manipulation/planning_observation.py#L541)。

输入签名：

```python
def _self_filtered_views(views, reference_frame, depth_range_m):
```

输出分支的源码表达式：

```python
normalized
```

直接写入的状态或数组位置：`labels[~valid]`、`normalized[name]['self_filter']`。

直接调用：

```python
_views
views.items
ValueError
validate_self_filter
evidence['labels'].copy
np.where
```

## `_partition_masks`

[定义位置](../../../manipulation/planning_observation.py#L565)。

输入签名：

```python
def _partition_masks(views, owner_ids):
```

输出分支的源码表达式：

```python
masks
```

直接写入的状态或数组位置：`masks[name]`、`masks[name][identity]`、`observed[identity]`、`masks[name][None]`。

直接调用：

```python
views.items
np.asarray
ValueError
isinstance
set
validate_self_filter
np.zeros
np.any
int
np.count_nonzero
masks[name][None].setflags
any
observed.values
```

## `_self_filter_report`

[定义位置](../../../manipulation/planning_observation.py#L611)。

输入签名：

```python
def _self_filter_report(views):
```

输出分支的源码表达式：

```python
reports
```

直接写入的状态或数组位置：`reports[name]`。

直接调用：

```python
views.items
int
np.count_nonzero
view['owner_masks'].items
```

## `_mapper_memory_bound`

[定义位置](../../../manipulation/planning_observation.py#L636)。

输入签名：

```python
def _mapper_memory_bound(config):
```

输出分支的源码表达式：

```python
int(2 * (tsdf + hashes + camera + esdf) + 16 * 1024 ** 2)
```

直接调用：

```python
math.prod
math.ceil
int
```

## `_require_audit_mapping_memory`

[定义位置](../../../manipulation/planning_observation.py#L657)。

输入签名：

```python
def _require_audit_mapping_memory(estimated_bytes, device, *, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES, max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES):
```

输出分支的源码表达式：

```python
required_bytes
```

直接调用：

```python
torch.device
ValueError
type
require_host_memory
ObservedTsdfGeometry.pending_storage_total_nbytes
require_device_memory
```

## `_require_native_voxel_layout`

[定义位置](../../../manipulation/planning_observation.py#L694)。

输入签名：

```python
def _require_native_voxel_layout(voxel, native):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
storage.get_idx
storage.params[0, layer, :3].detach().cpu
storage.params[0, layer, :3].detach
tuple
int
len
bool
torch.isfinite(values).all
torch.isfinite
(values <= 0).any
ValueError
math.floor
float
values.tolist
round
list
```

## `scene_task_owner_ids`

[定义位置](../../../manipulation/planning_observation.py#L732)。

输入签名：

```python
def scene_task_owner_ids(scene):
```

直接读取的对象属性：`scene.get`。

输出分支的源码表达式：

```python
(carrier,)
(carrier, reference)
```

直接调用：

```python
isinstance
ValueError
_text
scene.get
```

## `observation_resolutions`

[定义位置](../../../manipulation/planning_observation.py#L752)。

输入签名：

```python
def observation_resolutions(owner_ids, *, task_owner_ids, environment_voxel_size):
```

输出分支的源码表达式：

```python
{identity: TASK_VOXEL_SIZE_M if identity in task else float(environment_voxel_size) for identity in owners}
```

直接调用：

```python
isinstance
ValueError
tuple
_text
len
set
np.isfinite
float
```

## `_owner_mapper_config`

[定义位置](../../../manipulation/planning_observation.py#L778)。

输入签名：

```python
def _owner_mapper_config(config, views, masks, owner_id, pose, voxel_size, depth_error_m, query_radius_m=0.0):
```

输出分支的源码表达式：

```python
replace(config, grid_center=torch.as_tensor((lower + upper) / 2, dtype=torch.float32), extent_meters_xyz=extent, extent_esdf_meters_xyz=extent, voxel_size=voxel_size, esdf_voxel_size=voxel_size, truncation_distance=2 * voxel_size, max_visible_blocks_per_integration=None)
```

直接调用：

```python
isinstance
np.isfinite
ValueError
float
_inverse
views.items
np.nonzero
np.column_stack
np.ones
len
np.linalg.inv
clouds.append
np.concatenate
points.min
points.max
tuple
replace
torch.as_tensor
```

## `_fine_mapper_config`

[定义位置](../../../manipulation/planning_observation.py#L817)。

输入签名：

```python
def _fine_mapper_config(config, request):
```

输出分支的源码表达式：

```python
replace(config, grid_center=torch.as_tensor(center, dtype=torch.float32), extent_meters_xyz=tuple(extent), extent_esdf_meters_xyz=tuple(extent), voxel_size=size, esdf_voxel_size=size, truncation_distance=2 * size, max_visible_blocks_per_integration=None)
```

直接调用：

```python
np.asarray
float
np.isfinite(center).all
np.isfinite
np.isfinite(extent).all
np.any
np.isfinite([size, margin]).all
ValueError
_text
tuple
math.ceil
min
math.prod
replace
torch.as_tensor
```

## `_map_partition`

[定义位置](../../../manipulation/planning_observation.py#L848)。

输入签名：

```python
def _map_partition(normalized, masks, identity, config, inverse, depth_error_m, request=None):
```

输出分支的源码表达式：

```python
(query, statistics, region)
```

直接写入的状态或数组位置：`voxel.name`、`query.independent_geometry`。

直接调用：

```python
DeviceCfg
Mapper
normalized.items
np.where(masks[name][identity], view['depth'], 0.0).astype
np.where
CameraObservation
config_device.to_device
torch.as_tensor
np.asarray(view['image']).copy
np.asarray
_mapper_intrinsics
Pose.from_matrix
list
mapper.integrate
mapper.get_stats
RuntimeError
mapper.compute_esdf
SceneCollision.from_config
SceneCollisionCfg
SceneCfg
_require_native_voxel_layout
CuroboSphereQuery
','.join
build_tsdf_geometry
tuple
export_source_bits
_known_free_voxels(local_views, ray_depths={name: row['depth'] for (name, row) in local_views.items()}, origin=fine_origin, shape=shape[::-1], voxel_size=voxel.voxel_size, depth_error_m=depth_error_m).transpose
_known_free_voxels
local_views.items
math.ceil
math.sqrt
float
bits.tobytes().hex
bits.tobytes
np.packbits(free).tobytes().hex
np.packbits(free).tobytes
np.packbits
config.grid_center.cpu().tolist
config.grid_center.cpu
_mapper_memory_bound
```

## `_FrozenBackgroundRefinement`

[定义位置](../../../manipulation/planning_observation.py#L941)。

```python
class _FrozenBackgroundRefinement:
    ...
    ...
```

该类型的直接方法：`__init__`、`storage_bytes`、`refine`。

## `_FrozenBackgroundRefinement.__init__`

[定义位置](../../../manipulation/planning_observation.py#L945)。

输入签名：

```python
def _FrozenBackgroundRefinement.__init__(self, normalized, masks, config, request, depth_error_m, *, max_mapping_bytes, max_host_rss_bytes, max_device_used_bytes):
```

直接读取的对象属性：`self.views`、`self.masks`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.views`、`self.masks`、`self.views[name]`、`self.views[name][key]`、`self.masks[name]`、`self.config`、`self.request`、`self.depth_error_m`、`self.max_mapping_bytes`、`self.max_host_rss_bytes`、`self.max_device_used_bytes`。

直接调用：

```python
normalized.items
np.asarray(view[key]).copy
np.asarray
value.setflags
masks[name][None].copy
mask.setflags
deepcopy
float
```

## `_FrozenBackgroundRefinement.storage_bytes`

[定义位置](../../../manipulation/planning_observation.py#L966)。

输入签名：

```python
@property
def _FrozenBackgroundRefinement.storage_bytes(self):
```

直接读取的对象属性：`self.views`、`self.masks`。

输出分支的源码表达式：

```python
sum((value.nbytes for view in self.views.values() for value in view.values())) + sum((value[None].nbytes for value in self.masks.values()))
```

直接调用：

```python
sum
self.views.values
view.values
self.masks.values
```

## `_FrozenBackgroundRefinement.refine`

[定义位置](../../../manipulation/planning_observation.py#L971)。

输入签名：

```python
def _FrozenBackgroundRefinement.refine(self, current, witnesses):
```

直接读取的对象属性：`self.max_mapping_bytes`、`self.max_host_rss_bytes`、`self.max_device_used_bytes`、`self.request`、`self.config`、`self.views`、`self.masks`、`self.depth_error_m`。

输出分支的源码表达式：

```python
(None, report)
(replacement, report)
```

直接写入的状态或数组位置：`report['status']`、`center[expanded]`、`shape[axis]`、`report['requested_cells']`、`report['estimated_mapping_bytes']`、`report['mapping_reused']`、`support_report['previous_cuboid_envelopes']`、`replacement.refinement_source`、`replacement.mapping_report`。

直接调用：

```python
len
ValueError
np.asarray
np.isfinite(envelopes).all
np.isfinite
np.any
report.update
sum
int
sorted
float
tuple
np.minimum
envelopes[:, 0].min
np.maximum
envelopes[:, 1].max
expanded.any
current.region.pose[:3].copy
list
((lower[expanded] + upper[expanded]) / 2).astype
np.flatnonzero
max
math.ceil
math.prod
np.nextafter
RuntimeError
center.tolist
extent.tolist
actual_extent.tolist
current.fine.clone_for_rollout
deepcopy
_fine_mapper_config
_mapper_memory_bound
_require_audit_mapping_memory
str
_map_partition
np.eye
LayeredVoxelQuery
current.coarse.clone_for_rollout
np.argsort
replacement.region.prepare_guidance_boxes
np.concatenate
np.array_equal
```

## `build_audit_scene_from_rgbd`

[定义位置](../../../manipulation/planning_observation.py#L1101)。

输入签名：

```python
def build_audit_scene_from_rgbd(views, owner_frames, *, mapper_cfg, depth_error_m, owner_voxel_sizes_m, reference_frame, robot_sphere_materials, robot_geometry_source, gripper=None, has_gripper=True, contacts=(), known_solids=None, current_robot=None, max_mapping_bytes=256 * 1024 ** 2, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES, max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES, native_geometry_materials=None, robot_owner_permissions=(), fine_regions=None, robot_collision_bodies=()):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`frames[identity]`、`sizes[identity]`、`configs[identity]`、`queries[identity, layer]`、`statistics.setdefault('observed_background' if identity is None else identity, {})[layer]`、`regions[identity]`、`queries[None].refinement_source`、`queries[None].mapping_report`、`result.observation_report`。

直接调用：

```python
isinstance
TypeError
ValueError
type
min
_views
any
normalized.values
len
owner_frames.items
_text
_pose_array
set
owner_voxel_sizes_m.items
np.isfinite
float
tuple
dict
validate_body_groups
_validate_contact_and_gripper_inputs
dict.fromkeys
torch.device
_audit_partition_masks
mapper_cfg.grid_center.detach().cpu().numpy
mapper_cfg.grid_center.detach().cpu
mapper_cfg.grid_center.detach
frames.items
_owner_mapper_config
configs.items
jobs.update
_fine_mapper_config
fine_regions.items
_mapper_memory_bound
jobs.items
sum
estimates.values
jobs.values
_require_sample_array_budget
math.ceil
np.asarray
_known_free_voxels
normalized.items
known.copy
int
known.sum
_known_solid_evidence
_add_known_solid_voxels
ObservedSpace
_require_audit_mapping_memory
np.eye
_inverse
_map_partition
statistics.setdefault
LayeredVoxelQuery
_FrozenBackgroundRefinement
deepcopy
CollisionOwner
PosePath
np.array
np.repeat
PlanningScene
```

## `_require_mapping_memory`

[定义位置](../../../manipulation/planning_observation.py#L1306)。

输入签名：

```python
def _require_mapping_memory(estimated_bytes, device, *, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES, max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES):
```

输出分支的源码表达式：

```python
required
```

直接调用：

```python
torch.device
type
ValueError
require_host_memory
require_device_memory
```

## `_background_query_config`

[定义位置](../../../manipulation/planning_observation.py#L1325)。

输入签名：

```python
def _background_query_config(config, query_radius_m):
```

输出分支的源码表达式：

```python
(derived, query_center, report)
```

直接调用：

```python
isinstance
np.isfinite
ValueError
float
config.grid_center.detach().cpu().numpy().astype
config.grid_center.detach().cpu().numpy
config.grid_center.detach().cpu
config.grid_center.detach
tuple
math.ceil
np.asarray
max
zip
min
np.nextafter
replace
list
source_lower.tolist
source_upper.tolist
center.tolist
query_center.tolist
(query_center - query_half).tolist
(query_center + query_half).tolist
```

## `_validate_background_query_support`

[定义位置](../../../manipulation/planning_observation.py#L1380)。

输入签名：

```python
def _validate_background_query_support(query, query_radius_m):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
np.any
ValueError
```

## `_map_online_partition`

[定义位置](../../../manipulation/planning_observation.py#L1394)。

输入签名：

```python
def _map_online_partition(normalized, masks, identity, config, inverse, *, query_center=None, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
```

输出分支的源码表达式：

```python
(query, {'allocated_blocks': stats['num_allocated'], 'shape_xyz': shape.tolist(), 'voxel_size_m': float(config.esdf_voxel_size), 'origin': origin.tolist(), 'observed_geometry': exported['field_summary'], 'source_preparation_seconds': source_seconds, 'proposal_preparation_seconds': proposal_seconds, 'proposal_model': 'Conservative component-AABB distance bound; values are not source occupancy or penetration'})
```

直接调用：

```python
DeviceCfg
Mapper
normalized.items
np.where(masks[name][identity], view['depth'], 0.0).astype
np.where
CameraObservation
device.to_device
torch.as_tensor
np.asarray(view['image']).copy
np.asarray
_mapper_intrinsics
Pose.from_matrix
list
mapper.integrate
mapper.get_stats
ValueError
','.join
perf_counter
build_tsdf_geometry
FrozenObservedGeometry
math.ceil
float
config.grid_center.detach().cpu().numpy
config.grid_center.detach().cpu
config.grid_center.detach
FrozenESDF.from_source
tuple
shape.tolist
origin.tolist
```

## `build_scene_from_rgbd`

[定义位置](../../../manipulation/planning_observation.py#L1447)。

输入签名：

```python
def build_scene_from_rgbd(views, owner_frames, *, mapper_cfg, depth_error_m, task_owner_ids, reference_frame, robot_sphere_materials, robot_geometry_source, unknown_policy, gripper=None, has_gripper=True, contacts=(), max_mapping_bytes=256 * 1024 ** 2, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES, max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES, native_geometry_materials=None, robot_owner_permissions=(), fine_regions=None, robot_collision_bodies=(), complete_owners=(), initial_support_relations=(), holding_contacts=()):
```

输出分支的源码表达式：

```python
scene
```

直接写入的状态或数组位置：`direct[identity]`、`configs[identity]`、`configs[None]`、`queries[identity]`、`reports['observed_background' if identity is None else identity]`、`reports['observed_background']['query_support']`、`scene.observation_partition_masks`、`endpoint_partition_report[name]`、`scene.observation_report`。

直接调用：

```python
isinstance
TypeError
ValueError
type
min
_self_filtered_views
any
normalized.values
len
tuple
observation_resolutions
_pose_array
_text
owner_frames.items
_partition_masks
getattr
np.isfinite
np.isclose
owner.pose_path.sample
torch.tensor
position.detach().cpu().numpy
position.detach().cpu
position.detach
quaternion.detach().cpu().numpy
quaternion.detach().cpu
quaternion.detach
np.isfinite(position).all
np.isfinite(quaternion).all
np.linalg.norm
np.allclose
Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
_validate_contact_and_gripper_inputs
np.asarray
np.isfinite(values).all
max
float
values[:, 3].max
frames.items
_owner_mapper_config
dict
set
requests.items
_fine_mapper_config
np.einsum
np.abs
hand_radii.max
_background_query_config
_mapper_memory_bound
configs.items
configs.values
math.ceil
math.prod
sum
estimates.values
estimates.items
list
background_cfg.grid_center.detach().cpu().numpy
background_cfg.grid_center.detach().cpu
background_cfg.grid_center.detach
normalized.items
_known_free_voxels
ObservedSpace
_require_mapping_memory
np.eye
_inverse
_map_online_partition
_validate_background_query_support
background_support.update
queries[None].origin.tolist
queries[None].upper.tolist
queries[None].geometry_lower.tolist
queries[None].geometry_upper.tolist
CollisionOwner
PosePath
np.array
np.stack
OnlinePlanningScene
int
np.count_nonzero
owner_pixels.values
direct.items
known.sum
origin.tolist
deepcopy
_self_filter_report
```
