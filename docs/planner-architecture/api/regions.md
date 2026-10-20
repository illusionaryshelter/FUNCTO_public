# `manipulation/regions.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `SupportReferenceNotFound`

[定义位置](../../../manipulation/regions.py#L29)。

```python
class SupportReferenceNotFound(ValueError):
    ...
    ...
```

## `_vector`

[定义位置](../../../manipulation/regions.py#L40)。

输入签名：

```python
def _vector(value, name):
```

输出分支的源码表达式：

```python
vector.copy()
```

直接调用：

```python
np.asarray
np.isfinite(vector).all
np.isfinite
ValueError
vector.copy
```

## `_unit`

[定义位置](../../../manipulation/regions.py#L47)。

输入签名：

```python
def _unit(value, name):
```

输出分支的源码表达式：

```python
vector / length
```

直接调用：

```python
_vector
np.linalg.norm
np.isfinite
ValueError
```

## `function_frame`

[定义位置](../../../manipulation/regions.py#L55)。

输入签名：

```python
def function_frame(origin, axis, plane_normal):
```

输出分支的源码表达式：

```python
pose
```

直接写入的状态或数组位置：`pose[:3, :3]`、`pose[:3, 3]`。

直接调用：

```python
_vector
_unit
float
abs
ValueError
np.eye
np.column_stack
np.cross
```

## `source_region_frame`

[定义位置](../../../manipulation/regions.py#L80)。

输入签名：

```python
def source_region_frame(*, T_world_body, T_world_camera, points_world=None, T_world_function=None, interaction_point_world=None):
```

输出分支的源码表达式：

```python
{'T_world_function': frame, 'L_body_region': _inverse(body) @ frame, 'T_world_body_at_image': body, 'details': details}
```

直接调用：

```python
_pose_array
_points
points.mean
np.linalg.svd
np.finfo
max
ValueError
float
abs
np.linalg.norm
_vector
function_frame
np.sqrt
np.mean
_inverse
```

## `align_axis`

[定义位置](../../../manipulation/regions.py#L121)。

输入签名：

```python
def align_axis(preferred_rotation, local_axis, target_axis, *, antiparallel_axis=None):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`frame[:3, :3]`。

直接调用：

```python
np.asarray
ValueError
np.eye
_pose_array
_unit
np.cross
float
np.linalg.norm
np.clip
Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix
Rotation.from_rotvec
np.arctan2
abs
Rotation.from_rotvec(np.pi * branch).as_matrix
np.allclose
```

## `ground_relation`

[定义位置](../../../manipulation/regions.py#L159)。

输入签名：

```python
def ground_relation(demo_pose, demo_carrier_frame, scene_carrier_frame, demo_reference_frame, scene_reference_frame):
```

输出分支的源码表达式：

```python
{'goal_pose': rn @ functional @ _inverse(cn), 'functional_relation': functional, 'carrier_binding': cn @ _inverse(cd), 'reference_binding': rn @ _inverse(rd)}
```

直接调用：

```python
_pose_array
_inverse
```

## `relation_error`

[定义位置](../../../manipulation/regions.py#L187)。

输入签名：

```python
def relation_error(pose, carrier_frame, reference_frame, functional_relation):
```

输出分支的源码表达式：

```python
{'position_error_m': float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])), 'rotation_error_rad': float(Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude())}
```

直接调用：

```python
_pose_array
_inverse
float
np.linalg.norm
Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude
Rotation.from_matrix
```

## `_support_arguments`

[定义位置](../../../manipulation/regions.py#L204)。

输入签名：

```python
def _support_arguments(support, margin, gap_tolerance, normal_tolerance):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
isinstance
ValueError
any
np.isscalar
np.isfinite
```

## `check_region_on`

[定义位置](../../../manipulation/regions.py#L214)。

输入签名：

```python
def check_region_on(pose, carrier_region_points, carrier_outward_normal, support, *, gap_tolerance, normal_tolerance, margin=0.0):
```

输出分支的源码表达式：

```python
{'valid': bool(gap_error <= gap_tolerance + 1e-10 and normal_error <= normal_tolerance + 1e-10 and (violation <= 1e-07)), 'max_abs_gap_m': gap_error, 'minimum_signed_gap_m': float(gaps.min()), 'maximum_signed_gap_m': float(gaps.max()), 'boundary_violation_m': violation, 'normal_error_rad': normal_error, 'gap_tolerance_m': float(gap_tolerance), 'normal_tolerance_rad': float(normal_tolerance), 'margin_m': float(margin), 'checked_points': len(region), 'geometry_source': support.source}
```

直接调用：

```python
_support_arguments
_pose_array
_points
_unit
float
max
signed_boundary.max
np.arccos
np.clip
np.max
np.abs
bool
gaps.min
gaps.max
len
```

## `ground_on`

[定义位置](../../../manipulation/regions.py#L255)。

输入签名：

```python
def ground_on(preferred_pose, body_points, carrier_region_points, carrier_outward_normal, support, *, margin=0.0, gap_tolerance, normal_tolerance, antiparallel_axis=None):
```

输出分支的源码表达式：

```python
{'goal_pose': goal, 'checks': {'body': body_check, 'region': region_check}, 'freedom': {'required': 'specified face on finite support with opposing signed normals', 'twist': 'closest accepted sampled yaw to preferred orientation; chosen free component, not observed', 'tangential_position': 'nearest preferred position satisfying supplied full-body bounds at chosen yaw', 'yaw_search': {'chosen_offset_rad': float(yaw), 'step_degrees': 5.0, 'checked_candidates': checked, 'scope': 'bounded yaw samples; no global feasibility or optimality guarantee'}, 'antiparallel_branch': 'explicit geometric axis' if antiparallel_axis is not None else None}}
```

直接写入的状态或数组位置：`preferred[:3, :3]`、`height_pose[:3, 3]`、`candidate[:3, :3]`。

直接调用：

```python
_support_arguments
_points
_unit
_pose_array
align_axis
str
SupportReferenceNotFound
preferred.copy
float
np.min
check_region_on
range
np.deg2rad
Rotation.from_rotvec(yaw * support.normal).as_matrix
Rotation.from_rotvec
place_on
np.linalg.norm
valid.append
```

## `bind_region`

[定义位置](../../../manipulation/regions.py#L366)。

输入签名：

```python
def bind_region(source_points, target_points, *, max_error=0.01):
```

输出分支的源码表达式：

```python
{'transform': transform, 'diagnostics': diagnostics}
```

直接调用：

```python
rigid_registration
```

## `bind_observations`

[定义位置](../../../manipulation/regions.py#L379)。

输入签名：

```python
def bind_observations(mask, depth, K, T_camera_owner):
```

输出分支的源码表达式：

```python
{'owner_local_points': local, 'pixels': pixels, 'camera_points': camera}
```

直接调用：

```python
np.asarray
ValueError
_mask
_pose_array
np.isfinite
np.nonzero
len
np.column_stack((cols, rows)).astype
np.column_stack
backproject
```

## `sample_region_pixels`

[定义位置](../../../manipulation/regions.py#L403)。

输入签名：

```python
def sample_region_pixels(mask, depth, count=24):
```

输出分支的源码表达式：

```python
(candidates[selected], valid.copy())
```

直接调用：

```python
np.asarray
ValueError
isinstance
_mask
np.isfinite
cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype
cv2.erode
valid.astype
np.ones
interior.sum
np.nonzero
len
np.column_stack((x, y)).astype
np.column_stack
np.linspace(0, len(candidates) - 1, 6000).astype
np.linspace
int
np.argmin
np.sum
candidates.mean
np.full
min
np.minimum
np.argmax
selected.append
valid.copy
```

## `_context_crop`

[定义位置](../../../manipulation/regions.py#L442)。

输入签名：

```python
def _context_crop(image, owner_mask, region_mask):
```

输出分支的源码表达式：

```python
(image.crop((x0, y0, x1, y1)), region_mask[y0:y1, x0:x1], np.array([x0, y0]))
```

直接调用：

```python
np.nonzero
max
int
x.max
x.min
y.max
y.min
min
image.crop
np.array
```

## `_owner_observation`

[定义位置](../../../manipulation/regions.py#L454)。

输入签名：

```python
def _owner_observation(record, name):
```

输出分支的源码表达式：

```python
(image, owner, depth, pose)
```

直接调用：

```python
isinstance
ValueError
required.issubset
sorted
set
record['owner_id'].strip
_image
_mask
np.asarray
_intrinsics
np.any
np.isfinite
_pose_array
```

## `_source_observation`

[定义位置](../../../manipulation/regions.py#L476)。

输入签名：

```python
def _source_observation(demo):
```

输出分支的源码表达式：

```python
(image, owner, region, depth, pose, pixels, local)
```

直接调用：

```python
_owner_observation
required.issubset
ValueError
sorted
set
isinstance
demo[name].strip
_mask
np.any
np.asarray
len
np.isfinite(pixels).all
np.isfinite
any
value.strip
pixels.astype
backproject
```

## `_recipe_frame`

[定义位置](../../../manipulation/regions.py#L511)。

输入签名：

```python
def _recipe_frame(points, recipe):
```

输出分支的源码表达式：

```python
(frame, {'recipe': normalized, 'axis_length_m': float(np.linalg.norm(a)), 'plane_hint_length_m': float(np.linalg.norm(b)), 'direction_independence_sine': independence, 'x_axis_source': provenance, 'normal_source': 'ordered recipe directions; physical outward sign requires declared source evidence', 'shape_assumption': 'functional query correspondence; no local or whole-owner rigid congruence'})
```

直接写入的状态或数组位置：`weights[key]`。

直接调用：

```python
isinstance
ValueError
required.issubset
sorted
set
recipe['source'].strip
_points
np.asarray
len
np.isfinite(values).all
np.isfinite
values.astype
np.isclose
values.sum
_unit
np.cross
float
np.linalg.norm
function_frame
```

## `source_function_geometry`

[定义位置](../../../manipulation/regions.py#L564)。

输入签名：

```python
def source_function_geometry(demo):
```

输出分支的源码表达式：

```python
{'source_frame': frame, 'geometry_kind': demo['frame_recipe']['kind'], 'normal_role': normal_role, 'source_points': points, 'source_origin': frame[:3, 3].copy(), 'source_normal': frame[:3, 2].copy(), 'diagnostics': diagnostics}
```

直接调用：

```python
_source_observation
_recipe_frame
np.rint(pixels).astype
np.rint
demo.get
ValueError
diagnostics.update
list
np.flatnonzero
int
np.count_nonzero
frame[:3, 3].copy
frame[:3, 2].copy
```

## `match_functional_region`

[定义位置](../../../manipulation/regions.py#L596)。

输入签名：

```python
def match_functional_region(demo, current, *, client=None, extractor=None):
```

输出分支的源码表达式：

```python
{**source, 'target_frame': target_frame, 'transform': target_frame @ _inverse(source['source_frame']), 'target_points': target_points, 'target_origin': target_frame[:3, 3].copy(), 'target_normal': target_frame[:3, 2].copy(), 'diagnostics': diagnostics}
```

直接写入的状态或数组位置：`candidates[top:bottom, left:right]`、`diagnostics['current_interaction_evidence']`。

直接调用：

```python
source_function_geometry
_source_observation
_owner_observation
_selected_interaction_evidence
current.get
_context_crop
interaction['mask'][y:y + second_crop.height, x:x + second_crop.width].copy
propose_query_regions
np.asarray
len
np.any
ValueError
np.isfinite
enumerate
second_valid[top:bottom, left:right].any
np.zeros_like
match_points
targets.append
matching.append
backproject
_recipe_frame
dict
np.flatnonzero(np.all(target_pixels == pixel, axis=1)).tolist
np.flatnonzero
np.all
np.unique
np.count_nonzero
diagnostics.update
np.nonzero
np.rint(target_pixels).astype
np.rint
list
np.column_stack
int
mask.sum
mask[nearest[:, 1], nearest[:, 0]].tolist
_inverse
target_frame[:3, 3].copy
target_frame[:3, 2].copy
```

## `match_functional_region.cached_pair`

[定义位置](../../../manipulation/regions.py#L656)。

输入签名：

```python
def match_functional_region.cached_pair(first_image, second_image):
```

输出分支的源码表达式：

```python
pair_cache
```

直接调用：

```python
actual
```
