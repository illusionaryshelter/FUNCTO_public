# `manipulation/observations.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `Camera`

[定义位置](../../../manipulation/observations.py#L28)。

```python
class Camera:
    K: np.ndarray
    width: int
    height: int
    depth_scale: float = 0.001
```

## `_intrinsics`

[定义位置](../../../manipulation/observations.py#L37)。

输入签名：

```python
def _intrinsics(K):
```

输出分支的源码表达式：

```python
matrix
```

直接调用：

```python
np.asarray
np.isfinite(matrix).all
np.isfinite
np.allclose
abs
np.linalg.det
ValueError
```

## `load_calibration`

[定义位置](../../../manipulation/observations.py#L47)。

输入签名：

```python
def load_calibration(path) -> Camera:
```

输出分支的源码表达式：

```python
Camera(matrix, *dimensions, scales[0] if scales else 0.001)
```

直接调用：

```python
Path
path.is_dir
json.loads
path.read_text
np.asarray
raw.reshape
_intrinsics(raw).copy
_intrinsics
any
isinstance
ValueError
config_path.is_file
metadata.append
config_path.read_text
float
np.isfinite
scales.append
np.allclose
matrix.setflags
Camera
```

## `_frame_file`

[定义位置](../../../manipulation/observations.py#L87)。

输入签名：

```python
def _frame_file(directory, index, suffixes):
```

输出分支的源码表达式：

```python
found[0]
```

直接调用：

```python
Path
path.is_file
len
ValueError
```

## `load_rgbd`

[定义位置](../../../manipulation/observations.py#L95)。

输入签名：

```python
def load_rgbd(directory, index=0):
```

输出分支的源码表达式：

```python
(rgb, depth, camera)
```

直接调用：

```python
Path
load_calibration
Image.open
_frame_file
image.convert
np.load
np.asarray
ValueError
```

## `_mask`

[定义位置](../../../manipulation/observations.py#L117)。

输入签名：

```python
def _mask(mask, shape, name='mask'):
```

输出分支的源码表达式：

```python
array
```

直接调用：

```python
np.asarray
ValueError
array.any
```

## `backproject`

[定义位置](../../../manipulation/observations.py#L126)。

输入签名：

```python
def backproject(depth_m, K, pixels_uv=None, mask=None):
```

输出分支的源码表达式：

```python
rays * values[:, None]
```

直接调用：

```python
np.asarray
_intrinsics
ValueError
_mask
np.isfinite
np.nonzero
np.column_stack((cols, rows)).astype
np.column_stack
np.isfinite(pixels).all
np.any
np.rint(pixels).astype
np.rint
np.all
selected_mask[rows, cols].all
np.ones
len
np.linalg.inv
```

## `_noncollinear`

[定义位置](../../../manipulation/observations.py#L165)。

输入签名：

```python
def _noncollinear(points):
```

输出分支的源码表达式：

```python
False
bool(spans[1] > max(1e-09, spans[0] * 0.0001))
```

直接调用：

```python
len
np.linalg.svd
points.mean
bool
max
```

## `_RigidCorrespondences`

[定义位置](../../../manipulation/observations.py#L173)。

```python
class _RigidCorrespondences:
    ...
    ...
```

该类型的直接方法：`estimate`、`residuals`。

## `_RigidCorrespondences.estimate`

[定义位置](../../../manipulation/observations.py#L181)。

输入签名：

```python
def _RigidCorrespondences.estimate(self, source, target):
```

直接读取的对象属性：`self.params`。

输出分支的源码表达式：

```python
False
bool(np.isfinite(self.params).all())
```

直接写入的状态或数组位置：`self.params`。

直接调用：

```python
_noncollinear
o3d.geometry.PointCloud
o3d.utility.Vector3dVector
np.arange
len
o3d.utility.Vector2iVector
np.column_stack
o3d.pipelines.registration.TransformationEstimationPointToPoint
estimator.compute_transformation
bool
np.isfinite(self.params).all
np.isfinite
```

## `_RigidCorrespondences.residuals`

[定义位置](../../../manipulation/observations.py#L194)。

输入签名：

```python
def _RigidCorrespondences.residuals(self, source, target):
```

直接读取的对象属性：`self.params`。

输出分支的源码表达式：

```python
np.linalg.norm(predicted - target, axis=1)
```

直接调用：

```python
np.linalg.norm
```

## `rigid_registration`

[定义位置](../../../manipulation/observations.py#L199)。

输入签名：

```python
def rigid_registration(source, target, max_error=0.01):
```

输出分支的源码表达式：

```python
(transform, {'num_correspondences': len(source), 'num_inliers': int(inliers.sum()), 'inlier_mask': inliers, 'residuals_m': errors, 'inlier_rmse_m': float(np.sqrt(np.mean(errors[inliers] ** 2))), 'max_inlier_error_m': float(errors[inliers].max()), 'source_spans_m': np.linalg.svd(source[inliers] - source[inliers].mean(axis=0), compute_uv=False)})
```

直接调用：

```python
np.asarray
np.isfinite(source).all
np.isfinite
np.isfinite(target).all
ValueError
_noncollinear
ransac
float
np.random.default_rng
model.residuals
max
int
np.ceil
len
inliers.sum
np.asarray(model.params).copy
np.allclose
np.eye
np.isclose
np.linalg.det
np.sqrt
np.mean
errors[inliers].max
np.linalg.svd
source[inliers].mean
```

## `_model_poses`

[定义位置](../../../manipulation/observations.py#L249)。

输入签名：

```python
def _model_poses(value, name):
```

输出分支的源码表达式：

```python
poses
```

直接调用：

```python
np.asarray
len
np.isfinite(poses).all
np.isfinite
ValueError
np.allclose
np.swapaxes
np.eye
np.linalg.det
```

## `poses_from_models`

[定义位置](../../../manipulation/observations.py#L269)。

输入签名：

```python
def poses_from_models(camera_from_model, initial_points, *, reference_camera_from_model=None, frame_indices=None):
```

输出分支的源码表达式：

```python
{'poses': poses, 'points': points - origin, 'frame_indices': indices.copy(), 'model_from_body': model_from_body, 'body_origin_camera0_m': origin, 'body_frame': 'initial observed geometry centroid; axes aligned with initial camera', 'reference_frame': 'initial camera coordinates', 'camera_assumption': 'reference motion compensated; reference defines relative motion' if reference is not None else 'stationary camera required', 'units': 'metres', 'pose_convention': 'reference_from_body; column vectors'}
```

直接写入的状态或数组位置：`initial[:3, 3]`、`poses[0]`。

直接调用：

```python
_model_poses
np.asarray
len
np.isfinite(points).all
np.isfinite
ValueError
np.arange
np.issubdtype
(indices < 0).any
(indices[1:] <= indices[:-1]).any
points.mean
np.eye
np.linalg.solve
indices.copy
```

## `poses_from_tracks`

[定义位置](../../../manipulation/observations.py#L329)。

输入签名：

```python
def poses_from_tracks(tracked, initial_points, *, frame_indices=None, max_error=0.01):
```

输出分支的源码表达式：

```python
{'poses': poses, 'points': local_points['carrier'], 'frame_indices': indices.copy(), 'body_poses': body_poses, 'body_points': local_points, 'body_origin_camera0_m': origins['carrier'], 'camera_from_initial_camera': E @ np.linalg.inv(E[0]), 'body_frame': 'initial observed geometry centroid; axes aligned with initial camera', 'reference_frame': 'initial camera coordinates', 'camera_assumption': 'reference motion compensated; reference defines relative motion' if 'reference' in body_poses else 'supplied camera extrinsics remove camera motion', 'units': 'metres', 'pose_convention': 'reference_from_body; column vectors', 'tracking_method': 'TAPIP3D calibrated RGB-D material tracks + rigid correspondence fit', 'tracking_coords': coords.copy(), 'tracking_query_points': queries.copy(), 'tracking_visibility': visibility.copy(), 'tracking_query_owner': owners.copy(), 'tracking_owner_names': roles, 'tracking_projection_valid': projected.copy(), 'tracking_diagnostics': {'fit_threshold_m': float(max_error), 'per_owner': diagnostics, 'interpretation': 'visibility and fit residuals do not certify pose accuracy'}}
```

直接写入的状态或数组位置：`initial[:3, 3]`、`bodies[0]`、`body_poses[role]`、`local_points[role]`、`origins[role]`、`diagnostics[role]`、`poses[0]`。

直接调用：

```python
list
ValueError
np.asarray
tracked.get
np.isfinite(coords).all
np.isfinite
np.isfinite(queries).all
np.all
set
owners.tolist
range
len
camera_extrinsics
np.arange
(indices < 0).any
(indices[1:] <= indices[:-1]).any
enumerate
np.isfinite(geometry).all
np.flatnonzero
_noncollinear
rigid_registration
estimates.append
records.append
int
accepted_ids.copy
geometry.mean
np.eye
np.linalg.solve
body_poses['carrier'].copy
indices.copy
np.linalg.inv
coords.copy
queries.copy
visibility.copy
owners.copy
projected.copy
float
```

## `track_rgbd`

[定义位置](../../../manipulation/observations.py#L424)。

输入签名：

```python
def track_rgbd(directory, mask, *, reference_mask=None, max_frames=None, stationary_camera=None, extrinsics=None):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`geometries[role]`、`result['tracking_pixels']`、`result['camera_assumption']`。

直接调用：

```python
Path
sorted
int
(root / 'rgb').iterdir
path.suffix.lower
path.stem.isdigit
isinstance
ValueError
len
load_rgbd
_mask
np.any
config_path.is_file
json.loads
config_path.read_text
config.get
camera_extrinsics
zip
np.count_nonzero
np.isfinite
backproject
np.linspace
np.array_equal
frame_paths.append
_frame_file
depth_paths.append
track
np.stack
np.asarray
tracked.get
poses_from_tracks
np.asarray(tracked['pixels']).copy
```
