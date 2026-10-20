# `manipulation/regions.py` 实现契约

代码基线：`pending-implementation-commit`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `SupportReferenceNotFound`

[定义位置](../../../manipulation/regions.py#L29)。

```python
class SupportReferenceNotFound(ValueError):
```

类字段：

无显式条目。

实现说明：

> The existing support initializer could not provide a nominal member.
>
> This may be a missing preferred alignment branch, a failed exact-normal
> height member or an exhausted bounded yaw search. It carries no claim that
> the continuous allowed set is empty. Invalid numerical/structural geometry
> keeps its ordinary input error; callers retain the original support relation.

## `_vector`

[定义位置](../../../manipulation/regions.py#L40)。

输入签名：

```python
def _vector(value, name):
```

显式输出分支：

```python
vector.copy()
```

读取的对象属性：

```python
np.asarray
vector.shape
np.isfinite(vector).all
np.isfinite
vector.copy
```

写入目标：

```python
vector
```

直接调用符号：

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

显式输出分支：

```python
vector / length
```

读取的对象属性：

```python
np.linalg.norm
np.linalg
np.isfinite
```

写入目标：

```python
vector
length
```

直接调用符号：

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

实现说明：

> Construct an owner-local SE(3) frame from two observed signed directions.
>
> x follows the operational axis, z the function-plane normal, and y=z cross x.
> Axis and normal must be perpendicular; only floating-point residuals below
> 1e-6 are removed. A missing/parallel/unsigned direction is not completed from
> a world axis. Region centroids or PCA axes are not introduced by this API.
>
> This follows MimicFunc A.2.2's center-to-function direction. The released
> ``compute_function_plane_frame`` uses the opposite x and y, an equivalent
> fixed local z-pi gauge only when all associated poses use it consistently.

显式输出分支：

```python
pose
```

读取的对象属性：

```python
np.eye
np.column_stack
np.cross
```

写入目标：

```python
point
x
z
cosine
pose
pose[:3, :3]
pose[:3, 3]
```

直接调用符号：

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

实现说明：

> Bind one supplied source frame, or construct it from the source region.
>
> All inputs belong to the region-image timestamp.  A supplied interaction
> point from another time must first be transported with its owning body.
> The fitted normal faces the source camera; the longer projected camera x/y
> axis fixes the tangent (x wins ties).  These axes are a coordinate convention,
> not an inferred approach direction or a task orientation constraint.

显式输出分支：

```python
{'T_world_function': frame, 'L_body_region': _inverse(body) @ frame, 'T_world_body_at_image': body, 'details': details}
```

读取的对象属性：

```python
points.mean
np.linalg.svd
np.linalg
np.finfo(float).eps
np.finfo
np.linalg.norm
np.sqrt
np.mean
```

写入目标：

```python
body
camera
frame
details
points
center
_
singular
vectors
normal
facing
axes
projected
lengths
selected
origin
```

直接调用符号：

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

实现说明：

> Choose the rotation nearest a preference that maps one signed axis.
>
> The remaining twist is selected from ``preferred_rotation``, not measured
> from the single axis. Return one numerical rotation candidate; callers must
> retain that distinction when stating task requirements. For opposite axes,
> all perpendicular half-turn axes tie. The caller must supply an explicit
> geometric ``antiparallel_axis`` in the target/reference frame to break that
> ambiguity; no fixed world axis is substituted.

显式输出分支：

```python
result
```

读取的对象属性：

```python
np.asarray
preferred.shape
np.eye
np.cross
np.linalg.norm
np.linalg
np.clip
Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix
Rotation.from_rotvec
np.arctan2
Rotation.from_rotvec(np.pi * branch).as_matrix
np.pi
np.allclose
```

写入目标：

```python
preferred
frame
frame[:3, :3]
source
target
cross
sine
cosine
correction
branch
result
```

直接调用符号：

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

实现说明：

> Preserve an explicit functional relation across different owner shapes.
>
> ``demo_pose`` is T_Rd_Cd. Each functional frame F is T_owner_function in
> that owner's *fixed* local coordinates, not an observation-camera pose.
> Let S=inv(F_Rd) D_d F_Cd. The event goal is D_n=F_Rn S inv(F_Cn).
> This is equivalent to A_R D_d inv(A_C), with A_X=F_Xn inv(F_Xd).
> No congruence of the complete objects or their region point sets is needed.
>
> The returned pose is a functional-event target, not a freely normalized
> trajectory reference: normalizing it by its initial pose would cancel A_C.
> A complete frame target must not silently replace a partially observed
> plane/axis relation; such callers first choose and label their free pose
> components and must validate the actual partial relation independently.

显式输出分支：

```python
{'goal_pose': rn @ functional @ _inverse(cn), 'functional_relation': functional, 'carrier_binding': cn @ _inverse(cd), 'reference_binding': rn @ _inverse(rd)}
```

读取的对象属性：

无显式条目。

写入目标：

```python
demo
cd
cn
rd
rn
functional
```

直接调用符号：

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

实现说明：

> Independently check function-frame position and SO(3) angle residuals.
>
> Errors are metric diagnostics, with no built-in task acceptance threshold.
> They do not establish physical contact, grasp stability or task success.

显式输出分支：

```python
{'position_error_m': float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])), 'rotation_error_rad': float(Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude())}
```

读取的对象属性：

```python
np.linalg.norm
np.linalg
Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude
Rotation.from_matrix
expected[:3, :3].T
```

写入目标：

```python
body
carrier
reference
expected
actual
```

直接调用符号：

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

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.isscalar
np.isfinite
np.pi
```

写入目标：

```python
values
value
```

直接调用符号：

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

实现说明：

> Check the specifically declared contact face against a finite support.
>
> Every supplied owner-local contact-face sample must be within the support
> boundary and plane gap tolerance. Its signed outward normal must oppose the
> support normal. Unlike ``check_on``, a different, lower part of the carrier
> cannot satisfy this contact-face check. No convex hull is fabricated for the
> carrier region and no missing support geometry is completed.
>
> This validates supplied part geometry, not the semantic part name: if a
> caller labels the rim as the bottom, geometry cannot recover that label.
> Return diagnostics plus ``valid``; invalid numerical inputs raise, whereas
> a valid but unsatisfied geometric relation returns ``valid=False``.

显式输出分支：

```python
{'valid': bool(gap_error <= gap_tolerance + 1e-10 and normal_error <= normal_tolerance + 1e-10 and (violation <= 1e-07)), 'max_abs_gap_m': gap_error, 'minimum_signed_gap_m': float(gaps.min()), 'maximum_signed_gap_m': float(gaps.max()), 'boundary_violation_m': violation, 'normal_error_rad': normal_error, 'gap_tolerance_m': float(gap_tolerance), 'normal_tolerance_rad': float(normal_tolerance), 'margin_m': float(margin), 'checked_points': len(region), 'geometry_source': support.source}
```

读取的对象属性：

```python
body[:3, :3].T
support.origin
support.normal
support._hull.equations
support._hull
support.basis
boundary[:, :2].T
signed_boundary.max
np.arccos
np.clip
np.max
np.abs
gaps.min
gaps.max
support.source
```

写入目标：

```python
body
region
outward
points
local
gaps
boundary
signed_boundary
violation
normal_error
gap_error
```

直接调用符号：

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

实现说明：

> Instantiate a partial face-on-support relation using existing geometry.
>
> First rotate the specified carrier outward normal opposite the support
> normal, selecting the closest twist from the explicit pose preference.
> Then ``place_on`` moves the *whole supplied body geometry* into the finite
> support. If that preferred twist cannot produce an accepted candidate, try
> 5-degree yaw increments about the physical support normal, in increasing
> absolute angle. Both signs at the same angle are compared by translation
> change. Finally verify the declared functional contact face independently;
> the body's lowest point alone cannot make this API report success.
>
> The chosen twist and tangential location are numerical preferences, not
> measured task requirements. Their feasible range is bounded by the supplied
> support and body geometry. The output certifies this sampled geometric
> relation only, not hidden-shape coverage, force stability or collision-free
> robot execution. The finite yaw search is not a global infeasibility proof.
> A missing antiparallel branch or failed exact-normal height member is a
> reference-initialization failure, not a proof against the allowed relation.

显式输出分支：

```python
{'goal_pose': goal, 'checks': {'body': body_check, 'region': region_check}, 'freedom': {'required': 'specified face on finite support with opposing signed normals', 'twist': 'closest accepted sampled yaw to preferred orientation; chosen free component, not observed', 'tangential_position': 'nearest preferred position satisfying supplied full-body bounds at chosen yaw', 'yaw_search': {'chosen_offset_rad': float(yaw), 'step_degrees': 5.0, 'checked_candidates': checked, 'scope': 'bounded yaw samples; no global feasibility or optimality guarantee'}, 'antiparallel_branch': 'explicit geometric axis' if antiparallel_axis is not None else None}}
```

读取的对象属性：

```python
support.normal
preferred.copy
preferred[:3, :3].T
support.origin
np.min
np.deg2rad
Rotation.from_rotvec(yaw * support.normal).as_matrix
Rotation.from_rotvec
np.linalg.norm
np.linalg
valid.append
```

写入目标：

```python
body
region
outward
preferred
preferred[:3, :3]
height_pose
rotated_body
plane_height
height_pose[:3, 3]
vertical
checked
last_region
step
angle
angles
valid
yaw
candidate
candidate[:3, :3]
goal
body_check
region_check
distance
best
item
_
detail
```

直接调用符号：

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

实现说明：

> Fit corresponding owner-local surface samples, preserving fit evidence.
>
> Source/target rows must already represent paired semantic-region locations.
> This optional estimator is useful for locally congruent geometry; it is not
> a requirement that functionally equivalent objects have congruent shapes.
> The existing robust rigid estimator rejects collinear and nonrigid consensus.
> A planar, noncollinear correspondence remains a valid source of full SE(3).

显式输出分支：

```python
{'transform': transform, 'diagnostics': diagnostics}
```

读取的对象属性：

无显式条目。

写入目标：

```python
transform
diagnostics
```

直接调用符号：

```python
rigid_registration
```

## `bind_observations`

[定义位置](../../../manipulation/regions.py#L379)。

输入签名：

```python
def bind_observations(mask, depth, K, T_camera_owner):
```

实现说明：

> Lift visible region pixels into their fixed owner frame, without mutation.
>
> Depth is registered metric camera-z depth. ``T_camera_owner`` is that
> observation's calibrated camera-from-owner pose; it need not be identity.
> Nonpositive/nonfinite depth contributes no surface point. An empty valid
> region fails rather than becoming a whole-owner observation. Hidden geometry
> is not reconstructed or filled. Returned pixels use original-image (u,v).

显式输出分支：

```python
{'owner_local_points': local, 'pixels': pixels, 'camera_points': camera}
```

读取的对象属性：

```python
np.asarray
depth.ndim
depth.size
depth.shape
np.isfinite
np.nonzero
np.column_stack((cols, rows)).astype
np.column_stack
```

写入目标：

```python
depth
region
pose
valid
rows
cols
pixels
camera
local
```

直接调用符号：

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

实现说明：

> Choose bounded, dispersed source queries strictly inside a visible ROI.
>
> Erode a valid-depth ROI by one pixel when enough points remain, reducing
> boundary-depth contamination. Farthest-point sampling considers at most
> 6,000 candidate pixels and stores no dense distance matrix. This controls
> descriptor workload; it neither invents a center semantic point nor expands
> a failed local mask to an owner mask. Returns ``(pixels_uv, valid_mask)``.

显式输出分支：

```python
(candidates[selected], valid.copy())
```

读取的对象属性：

```python
np.asarray
depth.ndim
depth.size
np.bool_
np.integer
depth.shape
np.isfinite
cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype
cv2.erode
valid.astype
np.uint8
np.ones
cv2.BORDER_CONSTANT
interior.sum
np.nonzero
np.column_stack((x, y)).astype
np.column_stack
np.linspace(0, len(candidates) - 1, 6000).astype
np.linspace
np.argmin
np.sum
candidates.mean
np.full
np.inf
np.minimum
np.argmax
selected.append
valid.copy
```

写入目标：

```python
depth
region
valid
interior
y
x
candidates
selected
distances
index
```

直接调用符号：

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

实现说明：

> Keep owner context for descriptors while restricting their candidate ROI.

显式输出分支：

```python
(image.crop((x0, y0, x1, y1)), region_mask[y0:y1, x0:x1], np.array([x0, y0]))
```

读取的对象属性：

```python
np.nonzero
x.max
x.min
y.max
y.min
image.width
image.height
image.crop
np.array
```

写入目标：

```python
y
x
padding
x0
y0
x1
y1
```

直接调用符号：

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

实现说明：

> Validate observable owner data without requiring its functional answer.

显式输出分支：

```python
(image, owner, depth, pose)
```

读取的对象属性：

```python
required.issubset
record['owner_id'].strip
image.height
image.width
np.asarray
depth.shape
np.any
np.isfinite
```

写入目标：

```python
required
image
shape
owner
depth
pose
```

直接调用符号：

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

实现说明：

> Require source function evidence, including its geometric query roles.
>
> Function-direction witnesses may lie outside the true interaction patch,
> but must remain on the same selected source owner. The interaction mask is
> never expanded to make it encompass those witnesses. Small segmentation
> disagreements are reported instead of treating pixel containment as proof
> of owner identity; query ownership itself is checked by backprojection.

显式输出分支：

```python
(image, owner, region, depth, pose, pixels, local)
```

读取的对象属性：

```python
required.issubset
demo[name].strip
image.height
image.width
np.any
np.asarray
pixels.ndim
pixels.shape
pixels.dtype.kind
pixels.dtype
np.isfinite(pixels).all
np.isfinite
value.strip
pixels.astype
```

写入目标：

```python
image
owner
depth
pose
required
name
region
pixels
descriptions
value
camera
local
```

直接调用符号：

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

实现说明：

> Rebuild a function frame, independent of cross-instance point distances.
>
> For Q (N,3), o=Q.T@w, a=Q.T@axis_weights and b=Q.T@plane_weights.
> sum(w)=1 and both direction sums=0 make the construction SE(3)-equivariant.
> x=unit(a), z=unit(b cross x), y=z cross x. Different point distances and
> shapes are allowed. Zero or parallel directions cannot produce a frame.
>
> In ``plane`` mode the same arithmetic supplies a reproducible plane basis,
> but its x direction is merely a chosen tangent, never an observed functional
> angle. Its ordered normal is not automatically a physical outward normal;
> the source recipe must describe its sign evidence for the relevant caller.

显式输出分支：

```python
(frame, {'recipe': normalized, 'axis_length_m': float(np.linalg.norm(a)), 'plane_hint_length_m': float(np.linalg.norm(b)), 'direction_independence_sine': independence, 'x_axis_source': provenance, 'normal_source': 'ordered recipe directions; physical outward sign requires declared source evidence', 'shape_assumption': 'functional query correspondence; no local or whole-owner rigid congruence'})
```

读取的对象属性：

```python
required.issubset
recipe['source'].strip
np.asarray
values.shape
values.dtype.kind
values.dtype
np.isfinite(values).all
np.isfinite
values.astype
np.isclose
values.sum
geometry.T
np.cross
np.linalg.norm
np.linalg
```

写入目标：

```python
required
geometry
weights
key
expected_sum
values
weights[key]
origin
a
b
x
hint
cross
independence
frame
provenance
normalized
```

直接调用符号：

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

实现说明：

> Validate and lift compiled source functional queries without inference.
>
> Source includes the true interaction mask, owner context, explicit queries,
> per-query descriptions and a geometry recipe. It need not contain fixed
> function/center/grasp landmarks. This computes the recipe's source frame;
> a program binder separately checks agreement with its compiled event frame.
> No visual model, current scene, API request or point matching is involved.

显式输出分支：

```python
{'source_frame': frame, 'geometry_kind': demo['frame_recipe']['kind'], 'normal_role': normal_role, 'source_points': points, 'source_origin': frame[:3, 3].copy(), 'source_normal': frame[:3, 2].copy(), 'diagnostics': diagnostics}
```

读取的对象属性：

```python
np.rint(pixels).astype
np.rint
demo.get
diagnostics.update
np.flatnonzero
np.count_nonzero
frame[:3, 3].copy
frame[:3, 2].copy
```

写入目标：

```python
_
owner
region
pixels
points
frame
diagnostics
rows
in_interaction
normal_role
```

直接调用符号：

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

实现说明：

> Infer current functional geometry from source evidence and owner RGB-D.
>
> Current input requires image, owner_mask, depth, K, T_camera_owner and
> owner_id. Optional interaction_evidence contains a selected visible part,
> semantic description and provenance, never a functional frame or queries.
> That part guides the coarse VLM request while preserving owner context;
> attachment and direction witnesses can remain outside its observed mask.
> The VLM chooses coarse owner-local candidate cells from the source example;
> the existing dense matcher searches each proposed box intersected with the
> current owner and valid depth. Those query correspondences reconstruct the
> current functional frame under the same source recipe. Unlike rigid fitting,
> no cross-instance pairwise-distance equality is assumed.
>
> Owner-context crops are fixed for all queries. One descriptor pair is
> extracted and reused; each matcher invocation retains its bounded sampling.
> Repeated target pixels are not arbitrarily reassigned: recipe degeneracy,
> together with reported correspondence ambiguity, exposes missing geometry.
> A geometric frame candidate is not proof of semantic or physical success.

显式输出分支：

```python
{**source, 'target_frame': target_frame, 'transform': target_frame @ _inverse(source['source_frame']), 'target_points': target_points, 'target_origin': target_frame[:3, 3].copy(), 'target_normal': target_frame[:3, 2].copy(), 'diagnostics': diagnostics}
```

读取的对象属性：

```python
current.get
first_crop.height
first_crop.width
interaction['mask'][y:y + second_crop.height, x:x + second_crop.width].copy
second_crop.height
second_crop.width
np.asarray
boxes.shape
boxes.dtype.kind
boxes.dtype
np.any
np.isfinite
second_valid[top:bottom, left:right].any
np.zeros_like
targets.append
matching.append
np.flatnonzero(np.all(target_pixels == pixel, axis=1)).tolist
np.flatnonzero
np.all
np.unique
np.count_nonzero
diagnostics.update
np.nonzero
np.rint(target_pixels).astype
np.rint
second.size
np.column_stack
mask.sum
mask[nearest[:, 1], nearest[:, 0]].tolist
target_frame[:3, 3].copy
target_frame[:3, 2].copy
```

写入目标：

```python
source
first
owner1
region
depth1
_
pixels
second
owner2
depth2
pose2
interaction
first_crop
first_owner
first_offset
second_crop
second_owner
second_offset
x
y
first_region
cropped_interaction
proposals
boxes
x1
y1
x2
y2
first_valid
second_depth
second_valid
index
left
top
right
bottom
pair_cache
targets
matching
candidates
candidates[top:bottom, left:right]
found
diagnostic
target_pixels
camera_points
target_points
target_frame
target_diagnostics
diagnostics
duplicate_groups
pixel
mask
yy
xx
nearest
diagnostics['current_interaction_evidence']
key
```

直接调用符号：

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
def cached_pair(first_image, second_image):
```

显式输出分支：

```python
pair_cache
```

读取的对象属性：

无显式条目。

写入目标：

```python
actual
pair_cache
```

直接调用符号：

```python
actual
```
