# `manipulation/planning_observation.py` 实现契约

代码基线：`A/B 工作区（实施起点 d97977c）`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_views`

[定义位置](../../../manipulation/planning_observation.py#L25)。

输入签名：

```python
def _views(views, reference_frame, depth_range_m):
```

实现说明：

> Normalize an explicitly calibrated snapshot without filling bad depth.

显式输出分支：

```python
result
```

读取的对象属性：

```python
np.asarray
limits.shape
np.isfinite(limits).all
np.isfinite
views.items
record.get
raw_depth.ndim
raw_depth.shape
raw_depth.size
_intrinsics(record['K']).copy
np.where
_pose_array(record['T_output_camera'], 'T_output_camera').copy
```

写入目标：

```python
limits
result
pixels
identity
record
raw_depth
depth
matrix
valid
result[identity]
```

直接调用符号：

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

[定义位置](../../../manipulation/planning_observation.py#L56)。

输入签名：

```python
def _mapper_intrinsics(matrix):
```

实现说明：

> Preserve input rays under the Mapper's half-integer pixel convention.
>
> Input calibration places pixel (u, v) at the integer centre used by
> backprojection and free-ray evidence. The native depth integrator emits
> allocation rays at (u+.5, v+.5) and casts positive projected coordinates
> to integers when reading depth. Adding .5 to both principal coordinates
> preserves the physical rays and nearest-centre depth selection. The
> caller's calibration is copied, so geometry outside this Mapper boundary
> keeps its original coordinates; RGB/feature kernels are not redefined.

显式输出分支：

```python
native
```

读取的对象属性：

```python
_intrinsics(matrix).copy
```

写入目标：

```python
native
native[:2, 2]
```

直接调用符号：

```python
_intrinsics(matrix).copy
_intrinsics
```

## `_rectangle_min_depth`

[定义位置](../../../manipulation/planning_observation.py#L72)。

输入签名：

```python
def _rectangle_min_depth(depth, lower_xy, upper_xy):
```

实现说明：

> Query inclusive, in-image integer rectangles with bounded image storage.
>
> For width w take a=2**floor(log2(w)), and similarly b for height h.
> The four a-by-b blocks anchored inside the rectangle at its corners cover
> it exactly: a<=w<2a and b<=h<2b. Their minimum is the rectangle minimum,
> without enlarging its footprint. SciPy owns the sliding-minimum operation;
> only one block-size image is retained at a time, with no persistent cache.
> The caller has already checked the full continuous projection's bounds.

显式输出分支：

```python
result
```

读取的对象属性：

```python
np.empty
depth.dtype
np.floor(np.log2(sizes)).astype
np.floor
np.log2
np.int64
np.unique
(blocks == block).all
lower_xy[selected].T
(upper_xy[selected] - block + 1).T
np.minimum
```

写入目标：

```python
result
sizes
blocks
block
selected
width
height
filtered
left
top
right
bottom
result[selected]
```

直接调用符号：

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

[定义位置](../../../manipulation/planning_observation.py#L101)。

输入签名：

```python
def _known_free_voxels(views, *, ray_depths, origin, shape, voxel_size, depth_error_m):
```

实现说明：

> Prove each complete voxel free under the supplied pixel-ray model.
>
> With all camera-z coordinates positive, the perspective projection of a
> convex combination of corners is a positive depth-weighted combination
> of their projections. Thus the eight projected corners bound the entire
> cube, including after camera rotation. The rectangle's minimum ray depth
> must exceed the furthest corner's camera-z by depth_error_m.
> ``ray_depths`` is mandatory: online callers pass self-filtered free-ray
> evidence; historical audit callers explicitly select original depth.
> Pixel quantization matches observations.backproject: continuous coordinates
> must lie in [0,W-1]x[0,H-1], then use nearest-pixel np.rint. Missing rays
> remain zero. Subpixel material and unprovided calibration error are not
> certified by this discrete sensor model.

显式输出分支：

```python
known.reshape(shape)
```

读取的对象属性：

```python
np.asarray
math.prod
origin.shape
np.isfinite(origin).all
np.isfinite
views.items
rays.shape
view['depth'].shape
np.isfinite(rays).all
np.any
np.asarray(row['depth']).size
views.values
np.zeros
np.indices((2, 2, 2)).reshape(3, -1).T
np.indices((2, 2, 2)).reshape
np.indices
np.finfo(np.float64).eps
np.finfo
np.float64
depth.shape
np.arange
np.unravel_index
np.column_stack
np.abs
(camera[..., 2] - camera_error[:, None, 2] > 1e-09).all
np.isfinite(camera).all
np.isfinite(camera_error).all
matrix[:2].T
np.nextafter
(uv - uv_error).min
np.inf
(uv + uv_error).max
(camera[..., 2] + camera_error[:, None, 2]).max
np.isfinite(lower).all
np.isfinite(upper).all
(lower >= 0).all
(upper <= [width - 1, height - 1]).all
np.rint(lower[inside]).astype
np.rint
np.int64
np.rint(upper[inside]).astype
known.reshape
```

写入目标：

```python
shape
origin
n
name
view
rays
cells
chunk
largest_image
row
known
offsets
eps
depth
matrix
transform
height
width
rotation
translation
start
indices
z
y
x
lattice
corners
camera
scale
camera_error
front
numerator
uv
numerator_error
z_error
uv_error
lower
upper
furthest
inside
lower_xy
upper_xy
low
known[indices[inside][furthest[inside] < low]]
```

直接调用符号：

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

## `_self_filtered_views`

[定义位置](../../../manipulation/planning_observation.py#L189)。

输入签名：

```python
def _self_filtered_views(views, reference_frame, depth_range_m):
```

实现说明：

> Validate measured classification before applying mapper depth limits.
>
> A sensor-valid measurement can lie outside this mapper's configured range.
> Such a pixel becomes ambiguous for this map and contributes neither an
> endpoint nor a free ray. No caller arrays or original depths are changed.

显式输出分支：

```python
normalized
```

读取的对象属性：

```python
views.items
evidence['labels'].copy
np.where
```

写入目标：

```python
normalized
name
original
evidence
valid
labels
labels[~valid]
normalized[name]['self_filter']
```

直接调用符号：

```python
_views
views.items
ValueError
validate_self_filter
evidence['labels'].copy
np.where
```

## `_partition_masks`

[定义位置](../../../manipulation/planning_observation.py#L213)。

输入签名：

```python
def _partition_masks(views, owner_ids, *, build_background=True):
```

实现说明：

> Partition measured endpoints with one eight-neighbour boundary ring.
>
> Check owner/owner overlap on the supplied masks before self filtering;
> uncertainty is not permission to silently choose a semantic winner.
> Only entities being integrated appear in owner_ids. A supplied background
> mask selects the static endpoints of the original observation independently
> of retained moving entities. Without that mask, the new-owner complement
> defines the background candidates. Environment pixels on the nonbackground
> side seed one 3-by-3 dilation; adjacent background candidates are withheld
> from integration. They acquire neither an entity identity nor free-space
> evidence. SELF, shadow and invalid-depth pixels do not seed this ring.
> Entity masks and independently filtered free rays remain unchanged.
> Retained background bypasses this integration.

显式输出分支：

```python
masks
```

读取的对象属性：

```python
views.items
np.asarray
image.dtype
np.uint8
image.shape
depth.shape
np.zeros
mask.dtype
np.bool_
mask.shape
np.any
np.count_nonzero
view.get
background.dtype
background.shape
np.ones
masks[name][None].setflags
observed.values
```

写入目标：

```python
masks
observed
identity
name
view
image
depth
supplied
evidence
environment
occupied
masks[name]
mask
masks[name][identity]
observed[identity]
background
boundary
masks[name][None]
count
```

直接调用符号：

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
view.get
binary_dilation
np.ones
masks[name][None].setflags
any
observed.values
```

## `_self_filter_report`

[定义位置](../../../manipulation/planning_observation.py#L278)。

输入签名：

```python
def _self_filter_report(views):
```

实现说明：

> Record excluded semantic pixels without converting unknown to free.

显式输出分支：

```python
reports
```

读取的对象属性：

```python
views.items
np.count_nonzero
view['owner_masks'].items
```

写入目标：

```python
reports
name
view
evidence
labels
reports[name]
identity
mask
```

直接调用符号：

```python
views.items
int
np.count_nonzero
view['owner_masks'].items
```

## `_mapper_memory_bound`

[定义位置](../../../manipulation/planning_observation.py#L302)。

输入签名：

```python
def _mapper_memory_bound(config):
```

实现说明：

> Conservative native buffer budget from the installed MapperCfg fields.
>
> Includes the real 10,000-block floor at block_size=8, TSDF/RGB storage,
> hashes and per-block metadata, camera candidate/support buffers, dense
> ESDF+PBA storage, retained collision-field copies, and a 2x scratch margin.
> CUDA context, loaded models and allocator caches are additional process
> costs; this bound is checked before creating any mapper.

显式输出分支：

```python
int(2 * (tsdf + hashes + camera + esdf) + 16 * 1024 ** 2)
```

读取的对象属性：

```python
math.prod
math.ceil
config.esdf_voxel_size
config.extent_esdf_meters_xyz
config.image_height
config.image_width
config.truncation_distance
config.block_size
config.voxel_size
config.max_blocks
config.color_grid_size
config.hash_capacity
config.max_support_pixels_per_block_camera
```

写入目标：

```python
cells
x
pixels
samples
blocks
tsdf
hashes
camera
esdf
```

直接调用符号：

```python
math.prod
math.ceil
int
```

## `scene_task_owner_ids`

[定义位置](../../../manipulation/planning_observation.py#L322)。

输入签名：

```python
def scene_task_owner_ids(scene):
```

实现说明：

> Read the manipulated body and functional reference from a bound scene.
>
> These are semantic identities already established by task grounding. An
> additional collision owner, including an initial support surface, is an
> environment owner unless it is also the current functional reference.
> No object name, dataset, geometry size or observed success selects a role.

显式输出分支：

```python
(carrier,)
(carrier, reference)
```

读取的对象属性：

```python
scene.get
```

写入目标：

```python
carrier
reference
```

直接调用符号：

```python
isinstance
ValueError
_text
scene.get
```

## `observation_resolutions`

[定义位置](../../../manipulation/planning_observation.py#L342)。

输入签名：

```python
def observation_resolutions(owner_ids, *, task_owner_ids, environment_voxel_size):
```

实现说明：

> Resolve the common 2 mm task / coarser environment mapping contract.
>
> Task identities select the shared task resolution. Reused entity fields
> preserve their original samples. Resource admission checks each new map
> at the requested spacing.

显式输出分支：

```python
{identity: TASK_VOXEL_SIZE_M if identity in task else float(environment_voxel_size) for identity in owners}
```

读取的对象属性：

```python
np.bool_
np.isfinite
```

写入目标：

```python
owners
task
identity
```

直接调用符号：

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

[定义位置](../../../manipulation/planning_observation.py#L366)。

输入签名：

```python
def _owner_mapper_config(config, views, masks, owner_id, pose, voxel_size, depth_error_m):
```

实现说明：

> Enclose every labelled depth sample in a finite owner-local grid.
>
> The background and raw free-space grid keep their own resolution. This
> tighter grid spends precision on the observed owner's surfaces without
> asserting unseen geometry or dropping a labelled surface to save memory.
> Padding covers truncation and complete sample cells; memory admission is
> still applied to all partitions together before any native allocation.

显式输出分支：

```python
replace(config, grid_center=torch.as_tensor((lower + upper) / 2, dtype=torch.float32), extent_meters_xyz=extent, extent_esdf_meters_xyz=extent, voxel_size=voxel_size, esdf_voxel_size=voxel_size, truncation_distance=2 * voxel_size, max_visible_blocks_per_integration=None)
```

读取的对象属性：

```python
np.bool_
np.isfinite
views.items
np.nonzero
np.column_stack
np.ones
np.linalg.inv(view['K']).T
np.linalg.inv
np.linalg
clouds.append
transform[:3, :3].T
np.concatenate
points.min
points.max
torch.as_tensor
torch.float32
```

写入目标：

```python
voxel_size
clouds
inverse
name
view
rows
columns
pixels
camera
transform
points
lower
upper
padding
extent
```

直接调用符号：

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

[定义位置](../../../manipulation/planning_observation.py#L400)。

输入签名：

```python
def _fine_mapper_config(config, request):
```

实现说明：

> Freeze a caller-evidenced small region in this partition's local frame.
>
> The request is generic metric geometry, not an object/task heuristic.
> Sensor allowance is supplied with its measurement evidence. Numerical
> support is separately computed from this Mapper's truncation and samples.

显式输出分支：

```python
replace(config, grid_center=torch.as_tensor(center, dtype=torch.float32), extent_meters_xyz=tuple(extent), extent_esdf_meters_xyz=tuple(extent), voxel_size=size, esdf_voxel_size=size, truncation_distance=2 * size, max_visible_blocks_per_integration=None)
```

读取的对象属性：

```python
np.asarray
center.shape
extent.shape
np.isfinite(center).all
np.isfinite
np.isfinite(extent).all
np.any
np.isfinite([size, margin]).all
config.esdf_voxel_size
math.ceil
math.prod
torch.as_tensor
torch.float32
```

写入目标：

```python
center
extent
size
margin
shape
value
```

直接调用符号：

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

## `_require_mapping_memory`

[定义位置](../../../manipulation/planning_observation.py#L429)。

输入签名：

```python
def _require_mapping_memory(estimated_bytes, device, *, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES, max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES):
```

实现说明：

> Admit only the fixed online Mapper and numerical-array working memory.

显式输出分支：

```python
required
```

读取的对象属性：

```python
torch.device
device.type
```

写入目标：

```python
device
required
```

直接调用符号：

```python
torch.device
type
ValueError
require_host_memory
require_device_memory
```

## `build_scene_from_rgbd`

[定义位置](../../../manipulation/planning_observation.py#L447)。

输入签名：

```python
def build_scene_from_rgbd(views, owner_frames, *, mapper_cfg, depth_error_m, task_owner_ids, reference_frame, robot_model, fixed_geometry_names, tool_geometry_names, contacts=(), gripper=None, has_gripper=True, unknown_policy, max_mapping_bytes, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES, max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES, fine_regions=None, observed_owners=(), background=None, coverage=None, initial_support_relations=()):
```

实现说明：

> Combine newly observed entities with existing fixed entity fields.
>
> owner_frames names only entities to integrate from the supplied masks and
> depth. observed_owners supplies existing fields and their declared motion;
> these entities require no image, mask or visible depth. Their samples and
> local support are passed unchanged to the planning scene.
>
> An existing background and coverage are reused in world coordinates. A new
> background uses each view's explicit background_mask when supplied, or the
> complement of newly mapped owners, after withholding the one-pixel
> eight-neighbour semantic boundary. The caller supplies a static partition
> when the observation also contains separately retained moving entities.
> The existing unmapped_pixels count includes the withheld boundary;
> independently measured free-ray coverage is unchanged.
>
> observation_partition_masks keeps original owner masks and the original
> background candidate separately from the admitted fusion background.
> Reusing these original inputs applies the boundary operation once; a
> previously filtered fusion mask is not a new semantic background input.

显式输出分支：

```python
scene
```

读取的对象属性：

```python
owner.query
owner.owner_id
owner_frames.items
set(frames).intersection
coverage.reference_frame
mapper_cfg.num_cameras
mapper_cfg.enable_static
mapper_cfg.decay_factor
mapper_cfg.frustum_decay_factor
mapper_cfg.depth_minimum_distance
mapper_cfg.depth_maximum_distance
normalized.values
mapper_cfg.voxel_size
configs.update
frames.items
({} if fine_regions is None else fine_regions).items
configs.items
estimates.values
np.asarray
mapper_cfg.grid_shape
mapper_cfg.grid_center.detach().cpu().numpy
mapper_cfg.grid_center.detach().cpu
mapper_cfg.grid_center.detach
mapper_cfg.grid_center
normalized.items
np.eye
','.join
np.array
np.stack
owners.extend
direct.values
view['owner_masks'].items
view.get
np.ones
view['depth'].shape
semantic.values
candidate.setflags
mask.setflags
scene.observation_partition_masks
masks[name].get
np.count_nonzero
owners_count.values
coverage._known
owner.query.summary
direct.items
resolutions.items
owner.query.voxel_size
known.sum
known.size
```

写入目标：

```python
direct
owner
direct[owner.owner_id]
frames
identity
row
all_ids
needs_mapping
normalized
masks
configs
resolutions
configs[None]
pose
_
request
configs[identity]
estimates
config
shape_xyz
origin
ray_depths
name
known
coverage
queries
reports
inverse
observations
key
view
query
report
queries[identity]
reports[name]
owners
source
scene
scene.observation_partition_masks
semantic
mask
candidate
supplied_background
scene.observation_partition_masks[name]
endpoint_report
environment_count
owners_count
background_count
endpoint_report[name]
scene.observation_report
value
```

直接调用符号：

```python
isinstance
ValueError
_pose_array
_text
owner_frames.items
set(frames).intersection
set
tuple
TypeError
bool
min
_self_filtered_views
any
normalized.values
_partition_masks
observation_resolutions
configs.update
_owner_mapper_config
frames.items
({} if fine_regions is None else fine_regions).items
_fine_mapper_config
_mapper_memory_bound
configs.items
max
estimates.values
np.asarray
mapper_cfg.grid_center.detach().cpu().numpy
mapper_cfg.grid_center.detach().cpu
mapper_cfg.grid_center.detach
normalized.items
_known_free_voxels
map
ObservedSpace
np.eye
_inverse
integrate_entity
','.join
CollisionOwner
PosePath
np.array
np.stack
owners.extend
direct.values
OnlinePlanningScene
view['owner_masks'].items
view.get
np.ones
semantic.values
candidate.setflags
mask.setflags
masks[name].get
int
np.count_nonzero
sum
owners_count.values
owner.query.summary
direct.items
resolutions.items
known.sum
_self_filter_report
```
