# `manipulation/grounding_geometry_evidence.py` 实现契约

代码基线：`96ee2dc-working-tree`。
职责和数据语义见[组件说明](../16-cross-instance-grasp.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `query_tool_candidates`

[定义位置](../../../manipulation/grounding_geometry_evidence.py#L25)。

输入签名：

```python
def query_tool_candidates(views, context, geometry, tool_poses, candidate_ids, interaction_time, observation_model, mask_scopes, *, finger_joint_positions_m, inspection_state_ids):
```

实现说明：

> Return one field per entity and candidate/state/entity query records.
>
> mask_scopes maps an observation-entity ID to one Boolean image mask per
> view. The fixed official robot model is reusable because its geometry and
> installed component relations persist across tasks. The caller supplies one
> named-order finger configuration and inspection-state ID per query row,
> evaluated by FK. The same candidate may therefore appear at several
> explicit openings without acquiring a new candidate identity. Task objects and
> the environment are represented solely by their observed RGB-D TSDFs.
> All supplied poses use one field per mask scope. Their official-shape
> vertex union bounds storage; changing the displayed candidate never
> changes the observed representation being compared. A one-pose input is
> the same operation used by the complete grounding prediction.

显式输出分支：

```python
(arrays, reports)
```

读取的对象属性：

```python
np.asarray
geometry.at
np.concatenate
(tool_pose @ poses[part['geom_id']])[:3, :3].T
geometry.collision_geometries
world_vertices.min
world_vertices.max
views[0]['depth_m'].shape
view['image'].convert
mask_scopes.items
torch.tensor
torch.float32
arrays.update
field.values
field.observed
field.weights
field.origin
field.voxel_size
np.eye
results.append
```

写入目标：

```python
tool_poses
states
positions
component_poses
row
state
world_vertices
tool_pose
poses
part
size
padding
lower
upper
height
width
images
view
arrays
reports
candidate_id
state_id
reports[candidate_id][state_id]
identifier
masks
config
observations
mask
image
field
mapping
prefix
fingers
results
pose
answer
statuses
status
value
reports[candidate_id][state_id][identifier]
```

直接调用符号：

```python
np.asarray
geometry.at
np.concatenate
zip
world_vertices.min
world_vertices.max
view['image'].convert
mask_scopes.items
MapperCfg
tuple
torch.tensor
dict
integrate_entity
arrays.update
query_native_field
_convex_solid
np.eye
results.append
all
float
len
set
```

## `storage_domain`

[定义位置](../../../manipulation/grounding_geometry_evidence.py#L112)。

输入签名：

```python
def storage_domain(parts, tool_poses_world, max_tilt_rad, settings):
```

实现说明：

> Allocate samples around the complete rotated official assembly.
>
> A point at distance rho from the TCP moves by at most
> ``2*rho*sin(max_tilt_rad/2)``.  The union of all initial hypotheses is
> enlarged by that bound and the TSDF interpolation/truncation sample band.
> These bounds allocate array storage; they are never collision geometry.

显式输出分支：

```python
dict(initial_union_lower_m=initial.min(0).tolist(), initial_union_upper_m=initial.max(0).tolist(), maximum_component_radius_m=rho, max_tilt_rad=float(max_tilt_rad), rotation_displacement_bound_m=rotation_pad, interpolation_truncation_storage_padding_m=storage_pad, requested_sample_lower_m=low.tolist(), requested_sample_upper_m=high.tolist(), center_m=((low + high) / 2).tolist(), extent_m=(sample_count * spacing).tolist(), role='finite TSDF sample allocation; no represented collision body')
```

读取的对象属性：

```python
coal.Cylinder
allocation.append
np.concatenate
np.linalg.norm(vertices, axis=1).max
np.linalg.norm
np.linalg
np.asarray
initial.append
pose[:3, :3].T
np.sin
np.floor
initial.min
np.ceil
initial.max
np.rint((high - low) / spacing).astype
np.rint
np.int64
initial.min(0).tolist
initial.max(0).tolist
low.tolist
high.tolist
((low + high) / 2).tolist
(sample_count * spacing).tolist
```

写入目标：

```python
allocation
part
bounds
vertices
rho
initial
value
pose
rotation_pad
spacing
storage_pad
low
high
sample_count
```

直接调用符号：

```python
isinstance
_native_bounds
_transform
allocation.append
box_corners
np.concatenate
float
np.linalg.norm(vertices, axis=1).max
np.linalg.norm
np.asarray
initial.append
np.sin
np.floor
initial.min
np.ceil
initial.max
np.rint((high - low) / spacing).astype
np.rint
dict
initial.min(0).tolist
initial.max(0).tolist
low.tolist
high.tolist
((low + high) / 2).tolist
(sample_count * spacing).tolist
```

## `background_model_masks`

[定义位置](../../../manipulation/grounding_geometry_evidence.py#L156)。

输入签名：

```python
def background_model_masks(views, local_material_observation_pixels, local_material_observation_view_indices):
```

实现说明：

> Exclude declared owner material by its exact original measurement pixels.
>
> The region evidence producer has already established owner membership and
> supplies the corresponding observation pixels/view indices. This consumer
> performs pixel indexing only. Original reference, owner and background
> masks remain separate inputs; derived masks describe model admission.

显式输出分支：

```python
(backgrounds, materials, reports)
```

读取的对象属性：

```python
np.asarray
pixels.ndim
pixels.shape
pixels.dtype.kind
pixels.dtype
view_indices.shape
view_indices.dtype.kind
view_indices.dtype
np.any
supplied.dtype
np.bool_
supplied.shape
np.asarray(view['depth_m']).shape
np.zeros
np.count_nonzero
```

写入目标：

```python
views
pixels
view_indices
backgrounds
materials
reports
index
view
identity
supplied
selected
material
material[selected[:, 1], selected[:, 0]]
backgrounds[identity]
materials[identity]
reports[identity]
```

直接调用符号：

```python
list
np.asarray
len
np.any
ValueError
enumerate
np.zeros
dict
int
np.count_nonzero
```

## `prepare_grasp_observations`

[定义位置](../../../manipulation/grounding_geometry_evidence.py#L200)。

输入签名：

```python
def prepare_grasp_observations(views, *, fixture, capture_frames, reference_frame, mapping_settings, owner_id, local_material_observation_pixels, local_material_observation_view_indices):
```

实现说明：

> Return normalized views, explicit partitions, reports and source arrays.
>
> The reference observation keeps the selected functional component. The
> independent owner observation and its original mask provide only the
> declared visible owner material. Static background endpoints come from
> background_mask; an empty mask preserves unassigned pixels as unknown.
> The same official robot FK/classifier and existing semantic boundary
> operation determine which measured endpoints enter each field.

显式输出分支：

```python
(normalized, masks, observation_reports, observation_arrays)
```

读取的对象属性：

```python
np.asarray
view['image'].convert
owner_mask.sum
np.count_nonzero
np.isfinite
depth.shape
observation_arrays.update
normalized.items
observation_reports[name].update
```

写入目标：

```python
views
model_backgrounds
local_materials
admission_reports
filter_geometries
observations
observation_reports
observation_arrays
index
view
name
owner_mask
background_mask
depth
image
K
transform
phase
filtered
observations[name]
observation_reports[name]
normalized
masks
row
background
```

直接调用符号：

```python
list
background_model_masks
robot_self_filter_shapes
enumerate
str
np.asarray
view['image'].convert
perf_counter
classify_depth
dict
int
owner_mask.sum
np.count_nonzero
len
np.isfinite
observation_arrays.update
_self_filtered_views
_partition_masks
normalized.items
observation_reports[name].update
```

## `prepare_grasp_scene`

[定义位置](../../../manipulation/grounding_geometry_evidence.py#L284)。

输入签名：

```python
def prepare_grasp_scene(views, *, fixture, capture_feedback, tool_geometry, entrance_finger_positions, native_finger_positions, T_tool_grasp, tool_poses_world, max_tilt_rad, mapping_settings, owner_id, local_material_observation_pixels, local_material_observation_view_indices):
```

实现说明：

> Build the final tilt assembly and observed background once.
>
> Views use the public grounding fields image, depth_m, K,
> T_world_camera_cv, observation, owner_observation, owner_mask, owner_evidence,
> background_mask and id. Capture feedback supplies named
> coordinates, robot-base calibration and TCP calibration at image time.
> Both explicit finger states use the fixture's finger-joint order. The
> complete official robot filters capture rays; only the exact TCP-fixed
> assembly and finger descendants participate in local candidate rotation.

显式输出分支：

```python
dict(parts=parts, basis_tool=basis_tool, field=field, states=states, metadata=metadata, arrays=arrays)
```

读取的对象属性：

```python
np.asarray(T_tool_grasp)[:3, :2].copy
np.asarray
next(iter(normalized.values()))['depth'].shape
normalized.values
torch.tensor
normalized.items
','.join
field.values
field.observed
field.weights
field.origin
field.voxel_size
```

写入目标：

```python
started
parts
states
capture_frames
components
part_arrays
robot_report
basis_tool
domain
robot_seconds
filter_started
normalized
masks
observation_reports
observation_arrays
filter_seconds
spacing
height
width
phase
config
configuration_seconds
field
mapping
name
row
integration_seconds
arrays
metadata
```

直接调用符号：

```python
perf_counter
build_tool_parts
np.asarray(T_tool_grasp)[:3, :2].copy
np.asarray
storage_domain
prepare_grasp_observations
float
next
iter
normalized.values
MapperCfg
tuple
torch.tensor
integrate_entity
dict
normalized.items
','.join
list
```

## `group_state_checks`

[定义位置](../../../manipulation/grounding_geometry_evidence.py#L357)。

输入签名：

```python
def group_state_checks(checks, states, guide):
```

实现说明：

> Share each rigid-component query under every declared finger state.

显式输出分支：

```python
dict(states=output, guide_support=guide)
```

读取的对象属性：

无显式条目。

写入目标：

```python
output
identity
rows
row
statuses
status
value
output[identity]
```

直接调用符号：

```python
all
dict
```
