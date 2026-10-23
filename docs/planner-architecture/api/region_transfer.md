# `manipulation/region_transfer.py` 实现契约

代码基线：`1526e60`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `RegionInputMissing`

[定义位置](../../../manipulation/region_transfer.py#L21)。

```python
class RegionInputMissing(ValueError):
```

类字段：

无显式条目。

实现说明：

> A declared input is absent; this is distinct from malformed evidence.

## `RegionInputMissing.__init__`

[定义位置](../../../manipulation/region_transfer.py#L24)。

输入签名：

```python
def __init__(self, reasons):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
'; '.join
self.reasons
```

写入目标：

```python
self.reasons
```

直接调用符号：

```python
list
super().__init__
super
'; '.join
```

## `_text`

[定义位置](../../../manipulation/region_transfer.py#L29)。

输入签名：

```python
def _text(value, name):
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

## `_time`

[定义位置](../../../manipulation/region_transfer.py#L35)。

输入签名：

```python
def _time(value, name):
```

显式输出分支：

```python
float(value)
```

读取的对象属性：

```python
np.bool_
np.isscalar
np.isfinite
```

写入目标：

无显式条目。

直接调用符号：

```python
isinstance
np.isscalar
np.isfinite
ValueError
float
```

## `_feature`

[定义位置](../../../manipulation/region_transfer.py#L42)。

输入签名：

```python
def _feature(value, name):
```

显式输出分支：

```python
feature
```

读取的对象属性：

```python
np.asarray
feature.dtype
np.float32
feature.ndim
feature.shape
np.isfinite(feature).all
np.isfinite
```

写入目标：

```python
feature
```

直接调用符号：

```python
np.asarray
min
np.isfinite(feature).all
np.isfinite
ValueError
```

## `_observation`

[定义位置](../../../manipulation/region_transfer.py#L50)。

输入签名：

```python
def _observation(observation, depth, K, camera, crop, name):
```

实现说明：

> Check pixel/point/camera identity without changing any point ordering.

显式输出分支：

```python
(depth, K, camera)
```

读取的对象属性：

```python
np.asarray
depth.ndim
depth.dtype.kind
depth.dtype
depth.shape
mask.dtype
np.bool_
mask.shape
values.dtype
values.shape
np.array_equal
bounds.shape
bounds.dtype.kind
bounds.dtype
```

写入目标：

```python
depth
K
camera
count
mask
field
values
pixels
bounds
```

直接调用符号：

```python
isinstance
ValueError
np.asarray
min
_calibration
_check_observation
len
np.array_equal
feature_observation
```

## `prepare_inputs`

[定义位置](../../../manipulation/region_transfer.py#L80)。

输入签名：

```python
def prepare_inputs(source, views, *, same_state_source):
```

实现说明：

> Build resident inputs from one schema-2 source and two selected views.
>
> Each view declares ``id``, ``entity_id``, original ``timestamp_s``, complete
> ``depth_m/K/T_world_camera_cv``, and its ``observation/feature/crop_xyxy``.
> Missing entity/feature evidence is represented by observation=None and
> explicit ``missing_reasons``; complete depth still enters the computation.
> ``same_state_source`` records why both views describe the same geometry,
> such as simultaneous acquisition or a frozen scene without a physics step.
> It is supplied evidence, not inferred from camera names or time proximity.
>
> ``source.owner_id`` names the source region's rigid body. View entity IDs
> name the selected semantic entity and do not identify a current rigid part.
> All large arrays remain shared and keep their original point identities.

显式输出分支：

```python
(loaded, feature_pairs)
```

读取的对象属性：

```python
source.get
np.any
view.get
ids.add
identities.add
np.asarray
target_depth.ndim
target_depth.dtype.kind
target_depth.dtype
target_depth.shape
record.update
np.all
current_feature.shape
feature.shape
prepared.append
```

写入目标：

```python
state_source
geometry
missing
name
observation
depth
_
feature
prepared
feature_pairs
identities
ids
view
target_depth
K
camera
record
target
reasons
reason
current_feature
feature_pairs[name]
record['missing_reasons']
query_indices
query_region
loaded
```

直接调用符号：

```python
_text
isinstance
ValueError
source.get
RegionInputMissing
_time
_observation
np.any
_feature
len
set
view.get
ids.add
identities.add
np.asarray
min
_calibration
dict
any
record.update
list
np.all
prepared.append
source_queries
next
iter
```

## `select_reference_transform`

[定义位置](../../../manipulation/region_transfer.py#L173)。

输入签名：

```python
def select_reference_transform(result):
```

实现说明：

> Choose the best existing hypothesis supporting the accepted primary.
>
> A tied common region does not certify a unique rotation. Its supported
> highest-score hypothesis with the smallest original ID initializes the
> reference without changing the accepted region or any numerical decision.

显式输出分支：

```python
None
{'hypothesis_id': chosen['hypothesis_id'], 'score': float(chosen['score']), 'scale': float(scale), 'rotation': rigid[:3, :3], 'translation': rigid[:3, 3], 'region_hypothesis_ids': list(supported), 'finite_candidate_unique': bool(result['finite_candidate_unique']), 'paired_query_rows': np.asarray(chosen['paired_query_rows'], dtype=np.int64), 'paired_surface_indices': np.asarray(chosen['paired_surface_indices'], dtype=np.int64), 'selection': 'highest_existing_score_then_smallest_hypothesis_id'}
```

读取的对象属性：

```python
result.get
np.isfinite
np.eye
np.bool_
np.asarray
np.int64
```

写入目标：

```python
primary
mode
supported
hypotheses
item
index
candidates
chosen
scale
rotation
translation
rigid
rigid[:3, :3]
rigid[:3, 3]
```

直接调用符号：

```python
result.get
len
ValueError
any
np.isfinite
min
np.eye
_pose_array
isinstance
float
list
bool
np.asarray
```

## `propagate_region_frame`

[定义位置](../../../manipulation/region_transfer.py#L208)。

输入签名：

```python
def propagate_region_frame(source, transform, *, current_region_owner=None, same_state_source=None):
```

实现说明：

> Map the source image-time frame; scale its origin, rotate its unit axes.
>
> ``current_region_owner`` optionally supplies owner_id, T_world_body,
> timestamp_s and source for the physical body carrying this region. It may
> be an active drawer inside the selected cabinet entity. Without that body
> pose, the current world frame is still available; no body-local frame or
> persistent moving-region attachment is manufactured. Binding the local
> frame additionally requires ``same_state_source`` to state why the owner
> pose and region observations describe this same geometry. Their original
> timestamps remain distinct when the declared evidence is a static scene.

显式输出分支：

```python
_region_frame_record(source, frame, current, transform['hypothesis_id'], 'selected-candidate reference frame; not a unique-pose certificate', current_region_owner, same_state_source)
```

读取的对象属性：

```python
np.asarray
np.eye
```

写入目标：

```python
source_body
local
frame
rotation
current
current[:3, :3]
current[:3, 3]
```

直接调用符号：

```python
_pose_array
np.asarray
np.eye
apply_sim3
_region_frame_record
```

## `_region_frame_record`

[定义位置](../../../manipulation/region_transfer.py#L235)。

输入签名：

```python
def _region_frame_record(source, frame, current, hypothesis_id, scope, current_region_owner, same_state_source):
```

实现说明：

> Store world/source frames and bind an explicitly supplied current owner.
>
> Both frame constructors use this same SE(3) bookkeeping. The owner pose
> and its acquisition provenance have the same contract as before.

显式输出分支：

```python
result
```

读取的对象属性：

```python
current_region_owner.get
result.update
```

写入目标：

```python
result
state_source
owner_id
pose
```

直接调用符号：

```python
_text
current_region_owner.get
_time
_pose_array
result.update
_inverse
dict
```

## `construct_interaction_region_frame`

[定义位置](../../../manipulation/region_transfer.py#L261)。

输入签名：

```python
def construct_interaction_region_frame(source, transform, primary, *, current_region_owner=None, same_state_source=None):
```

实现说明：

> Construct a manipulation interaction frame from an accepted region.
>
> W denotes the current calibrated world. The origin is primary.center in W,
> the observed seed used by the accepted region (not its mask centroid).
> For declared planar geometry, z is the signed observed normal and x is
> the transferred source functional x projected into that tangent plane;
> y=z cross x. For declared surface geometry there is no single region
> normal: the source functional rotation is transferred unchanged.
> Source functional frames and represented motion are never rewritten.
>
> The accepted primary and its supporting transform are supplied together
> by run_region_transfer. This API changes reference construction only;
> region acceptance and candidate evidence retain their original values.

显式输出分支：

```python
result
```

读取的对象属性：

```python
np.asarray
np.linalg.norm
np.linalg
np.eye
```

写入目标：

```python
source_frame
rotated
geometry
normal
tangent
current
axes_source
current[:3, :3]
current[:3, 3]
result
result['construction']
```

直接调用符号：

```python
_pose_array
np.asarray
np.linalg.norm
function_frame
np.eye
ValueError
_region_frame_record
dict
```

## `run_region_transfer`

[定义位置](../../../manipulation/region_transfer.py#L304)。

输入签名：

```python
def run_region_transfer(source, views, *, skill_kind, same_state_source, current_region_owner=None):
```

实现说明：

> Call the existing resident dual-view pipeline once and bind its frame.
>
> ``skill_kind`` is the existing representation.kind declaration. Grasp
> retains source-frame Sim(3) propagation. Manipulation constructs its
> interaction frame from the accepted primary center and declared geometry.
> The numerical region pipeline is identical for both declarations.
> The caller first runs dual_view.pipeline.prepare in its serial supervised
> GPU process. Missing inputs and an unsupported source plane retain explicit
> states; other malformed-input/runtime errors propagate. Returned loaded
> point banks and pipeline assets retain their separate index domains.
> When a current region-owner pose is supplied, ``same_state_source`` must
> also cover that pose, not merely the two images: either synchronized state
> records or explicit absence of owner motion between their acquisition times.

显式输出分支：

```python
{'status': 'missing', 'reasons': error.reasons, 'reference_frame': None, 'outputs': None}
{'status': 'rejected', 'reasons': error.reasons, 'reference_frame': None, 'outputs': None, 'loaded': loaded}
{'status': result['status'], 'reasons': list(result['reasons']), 'reference_frame': frame, 'selected_transform': selected, 'outputs': outputs, 'loaded': loaded}
```

读取的对象属性：

```python
error.reasons
pipeline.run
pipeline.InsufficientSource
rows.ndim
target_ids.shape
rows.shape
np.any
```

写入目标：

```python
loaded
features
outputs
result
selected
frame
rows
target_ids
selected['source_point_indices']
primary
mode
frame['target_entity_id']
```

直接调用符号：

```python
ValueError
prepare_inputs
pipeline.run
select_reference_transform
np.any
len
next
construct_interaction_region_frame
propagate_region_frame
list
```

## `frame_from_deformation`

[定义位置](../../../manipulation/region_transfer.py#L359)。

输入签名：

```python
def frame_from_deformation(coarse_source_points, deformed_source_points, source_region_indices, coarse_frame):
```

实现说明：

> Fit one rigid frame from OAReg's same-source-index deformation output.
>
> Inputs and output points already use the target-world coordinates. OAReg
> preserved the source order; q[i] is an estimated deformation, not target
> measurement i. Open3D fits p[i] -> q[i] with scaling disabled, then updates
> the already coarse-aligned frame once. No trajectory is deformed here.

显式输出分支：

```python
{'T_world_region': frame, 'T_local_correction': correction, 'source_region_indices': indices.copy(), 'rigid_fit_rms_m': float(np.sqrt(np.mean(np.sum((predicted - deformed[indices]) ** 2, axis=1)))), 'scope': 'local rigid approximation of deformed source points; not measured target correspondences'}
```

读取的对象属性：

```python
np.asarray
source.shape
deformed.shape
indices.ndim
indices.dtype.kind
indices.dtype
np.unique
np.any
o3d.geometry.PointCloud
o3d.geometry
o3d.utility.Vector3dVector
o3d.utility
o3d.utility.Vector2iVector
np.column_stack
np.arange
o3d.pipelines.registration.TransformationEstimationPointToPoint(False).compute_transformation
o3d.pipelines.registration.TransformationEstimationPointToPoint
o3d.pipelines.registration
o3d.pipelines
correction[:3, :3].T
indices.copy
np.sqrt
np.mean
np.sum
```

写入目标：

```python
source
deformed
indices
first
second
first.points
second.points
correspondence
correction
frame
predicted
```

直接调用符号：

```python
_points
np.asarray
len
np.unique
np.any
ValueError
min
rank
o3d.geometry.PointCloud
o3d.utility.Vector3dVector
o3d.utility.Vector2iVector
np.column_stack
np.arange
o3d.pipelines.registration.TransformationEstimationPointToPoint(False).compute_transformation
o3d.pipelines.registration.TransformationEstimationPointToPoint
_pose_array
indices.copy
float
np.sqrt
np.mean
np.sum
```
