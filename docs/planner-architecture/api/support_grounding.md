# `manipulation/support_grounding.py` 实现契约

代码基线：`CURRENT_IMPLEMENTATION`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_plane_candidates`

[定义位置](../../../manipulation/support_grounding.py#L31)。

输入签名：

```python
def _plane_candidates(depth, K, valid):
```

实现说明：

> Consensus on at most 6000 points, followed by bounded full-mask checks.
>
> Candidate masks contain actual valid-depth pixels within the fitted plane
> tolerance. The masks are never replaced by a hull, box, or filled contour.
> Closely intersecting planes may share observed boundary pixels; this does
> not authorize extending either plane through unobserved regions.

显式输出分支：

```python
(candidates, sample_count)
```

读取的对象属性：

```python
np.flatnonzero
np.linspace(0, len(flat) - 1, min(_MAX_POINTS, len(flat))).astype
np.linspace
depth.shape
np.column_stack
np.ceil
o3d.geometry.PointCloud
o3d.geometry
o3d.utility.Vector3dVector
o3d.utility
cloud.segment_plane
np.ones
points.mean
np.linalg.svd
np.linalg
np.zeros_like
np.abs
mask.flat
np.nonzero
candidates.append
xx.min
yy.min
xx.max
yy.max
np.sqrt
np.mean
```

写入目标：

```python
flat
selected
width
pixels
remaining
sample_count
minimum
candidates
_
cloud
indices
points
keep
keep[indices]
origin
singular
axes
normal
mask
start
batch
xy
xyz
in_plane
mask.flat[batch[in_plane]]
yy
xx
```

直接调用符号：

```python
np.flatnonzero
np.linspace(0, len(flat) - 1, min(_MAX_POINTS, len(flat))).astype
np.linspace
len
min
np.column_stack
backproject
max
int
np.ceil
range
o3d.geometry.PointCloud
o3d.utility.Vector3dVector
cloud.segment_plane
np.ones
points.mean
np.linalg.svd
np.zeros_like
np.abs
np.nonzero
candidates.append
xx.min
yy.min
xx.max
yy.max
float
np.sqrt
np.mean
```

## `_covered_polygon`

[定义位置](../../../manipulation/support_grounding.py#L86)。

输入签名：

```python
def _covered_polygon(polygon, mask):
```

实现说明：

> Check the polygon's whole rasterized area and densely sampled boundary.
>
> Boundary checks alone miss holes. Conversely, filled-polygon checks alone
> hide their subpixel rounding convention. We check both, at quarter-pixel
> boundary spacing, against an already one-pixel-eroded observed mask. This
> supplies a conservative pixel-resolution certificate; it cannot certify
> unmeasured subpixel holes or the physical quality of depth observations.

显式输出分支：

```python
(False, 0)
(False, int(raster.sum()))
(False, int(raster.sum()))
(True, int(raster.sum()))
```

读取的对象属性：

```python
cv2.contourArea
polygon.astype
np.float32
np.zeros
mask.shape
np.uint8
cv2.fillConvexPoly
np.rint(polygon * 256).astype
np.rint
np.int32
np.any
raster.astype
raster.sum
np.roll
np.ceil
np.linalg.norm
np.linalg
np.linspace
np.rint(samples).astype
mask[pixels[:, 1], pixels[:, 0]].all
```

写入目标：

```python
raster
first
second
count
samples
pixels
```

直接调用符号：

```python
len
cv2.contourArea
polygon.astype
np.zeros
cv2.fillConvexPoly
np.rint(polygon * 256).astype
np.rint
np.any
raster.astype
int
raster.sum
zip
np.roll
max
np.ceil
np.linalg.norm
np.linspace
np.rint(samples).astype
mask[pixels[:, 1], pixels[:, 0]].all
```

## `_observed_patch`

[定义位置](../../../manipulation/support_grounding.py#L114)。

输入签名：

```python
def _observed_patch(mask):
```

实现说明：

> Retain a full convex observation, or a conservative large local patch.
>
> For ordinary convex panels the full safe hull is retained, so a long shelf
> is not always shrunk to a small disk. If that hull crosses a hole, concavity,
> or gap, OpenCV's exact Euclidean distance transform supplies the largest
> inscribed disk center. Its radius is reduced by one pixel and a 64-vertex
> inscribed polygon is checked again. We never fill or close the input mask.

显式输出分支：

```python
(polygon, {'patch_method': method, 'patch_pixel_count': count, 'patch_area_pixels_squared': float(cv2.contourArea(polygon.astype(np.float32))), 'patch_boundary_pixels': polygon.tolist(), 'mask_erosion_pixels': 1, 'coverage_check': 'entire rasterized area and quarter-pixel boundary samples inside eroded observed mask'})
```

读取的对象属性：

```python
cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype
cv2.erode
mask.astype
np.uint8
np.ones
cv2.BORDER_CONSTANT
np.nonzero
cv2.convexHull(np.column_stack((xx, yy)).astype(np.int32)).reshape(-1, 2).astype
cv2.convexHull(np.column_stack((xx, yy)).astype(np.int32)).reshape
cv2.convexHull
np.column_stack((xx, yy)).astype
np.column_stack
np.int32
cv2.distanceTransform
core.astype
cv2.DIST_L2
cv2.DIST_MASK_PRECISE
np.unravel_index
np.argmax
distances.shape
np.arange
np.pi
np.array
np.cos
np.sin
cv2.contourArea
polygon.astype
np.float32
polygon.tolist
```

写入目标：

```python
core
yy
xx
polygon
covered
count
method
distances
row
col
radius
angles
```

直接调用符号：

```python
cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype
cv2.erode
mask.astype
np.ones
np.nonzero
len
ValueError
cv2.convexHull(np.column_stack((xx, yy)).astype(np.int32)).reshape(-1, 2).astype
cv2.convexHull(np.column_stack((xx, yy)).astype(np.int32)).reshape
cv2.convexHull
np.column_stack((xx, yy)).astype
np.column_stack
_covered_polygon
cv2.distanceTransform
core.astype
np.unravel_index
int
np.argmax
float
np.arange
np.array
np.cos
np.sin
cv2.contourArea
polygon.astype
polygon.tolist
```

## `infer_support`

[定义位置](../../../manipulation/support_grounding.py#L154)。

输入签名：

```python
def infer_support(image, depth_m, K, reference_mask, *, support_direction, instruction, client=None):
```

实现说明：

> Select a task support from visible reference-instance plane candidates.
>
> RGB and depth must be aligned, depth must be metric, and ``K`` must describe
> that image. All returned geometry uses camera coordinates. Even a single
> candidate requires semantic confirmation: its largest plane may be a side
> panel. Selection accepts exactly ``{"candidate_id": ID_or_null}``.
> ``support_direction`` is a nonzero direction in these same camera
> coordinates, supplied from known calibration and the intended physical
> support side. For a top support it can be calibrated world-up. It is not a
> point, the current carrier position, a matched query, or an inferred object
> axis. Only its direction matters. Near-tangent directions are rejected;
> accepting its sign does not establish a stable or sufficiently level face.
>
> Unordered observed points establish the plane. Its sample mean and PCA
> tangent axis define computational coordinates only: neither is a physical
> object centre, placement target, or task-required yaw. The returned finite
> patch remains certified against actual visible support pixels.

显式输出分支：

```python
(surface, diagnostics)
```

读取的对象属性：

```python
np.asarray
depth.shape
mask.shape
direction.shape
np.isfinite(direction).all
np.isfinite
np.max
np.abs
np.linalg.norm
np.linalg
instruction.strip
valid.sum
json.dumps
_client(client).chat_json
chosen['normal'].copy
np.column_stack
np.ones
np.linalg.inv(np.asarray(K, float)).T
np.linalg.inv
np.any
np.isfinite(distance).all
np.cross
surface.source
direction.tolist
```

写入目标：

```python
image
mask
depth
direction
scale
valid
candidates
sample_count
prompt
candidate
selected
number
chosen
origin
normal
side
polygon
patch_diagnostics
rays
denominators
distance
vertices
first
basis
surface
diagnostics
key
```

直接调用符号：

```python
_image
_mask
np.asarray
ValueError
np.isfinite(direction).all
np.isfinite
float
np.max
np.abs
np.linalg.norm
isinstance
instruction.strip
valid.sum
_plane_candidates
json.dumps
_client(client).chat_json
_client
_candidate_sheet
set
type
len
chosen['normal'].copy
abs
_observed_patch
np.column_stack
np.ones
np.linalg.inv
np.any
np.isfinite(distance).all
np.cross
SupportSurface
direction.tolist
```
