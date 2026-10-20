# `manipulation/support_grounding.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_plane_candidates`

[定义位置](../../../manipulation/support_grounding.py#L31)。

输入签名：

```python
def _plane_candidates(depth, K, valid):
```

输出分支的源码表达式：

```python
(candidates, sample_count)
```

直接写入的状态或数组位置：`keep[indices]`、`mask.flat[batch[in_plane]]`。

直接调用：

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

输出分支的源码表达式：

```python
(False, 0)
(False, int(raster.sum()))
(True, int(raster.sum()))
```

直接调用：

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

输出分支的源码表达式：

```python
(polygon, {'patch_method': method, 'patch_pixel_count': count, 'patch_area_pixels_squared': float(cv2.contourArea(polygon.astype(np.float32))), 'patch_boundary_pixels': polygon.tolist(), 'mask_erosion_pixels': 1, 'coverage_check': 'entire rasterized area and quarter-pixel boundary samples inside eroded observed mask'})
```

直接调用：

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

输出分支的源码表达式：

```python
(surface, diagnostics)
```

直接调用：

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
