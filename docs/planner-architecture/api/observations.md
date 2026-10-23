# `manipulation/observations.py` 实现契约

代码基线：`d7bcafc`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `Camera`

[定义位置](../../../manipulation/observations.py#L28)。

```python
class Camera():
```

类字段：

```python
K: np.ndarray
width: int
height: int
depth_scale: float = 0.001
```

实现说明：

> Pinhole calibration; ``depth_scale`` is metres per raw depth unit.

## `_intrinsics`

[定义位置](../../../manipulation/observations.py#L37)。

输入签名：

```python
def _intrinsics(K):
```

显式输出分支：

```python
matrix
```

读取的对象属性：

```python
np.asarray
matrix.shape
np.isfinite(matrix).all
np.isfinite
np.allclose
np.linalg.det
np.linalg
```

写入目标：

```python
matrix
```

直接调用符号：

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

实现说明：

> Read Open3D calibration, including its column-major flattened matrix.
>
> ``path`` may be intrinsic.json or its directory. ``depth_unit_m`` explicitly
> gives metres per raw value. The existing Open3D ``depth_scale`` metadata is
> a *divisor*, e.g. 1000 means millimetres. Either field may occur in intrinsic
> or its sibling config.json; conflicting declarations are rejected. When no
> scale is declared, the repository's millimetre convention (0.001) applies.
> This intentionally does not guess conventions from the magnitude of data.

显式输出分支：

```python
Camera(matrix, *dimensions, scales[0] if scales else 0.001)
```

读取的对象属性：

```python
path.is_dir
json.loads
path.read_text
np.asarray
raw.shape
raw.reshape
_intrinsics(raw).copy
path.parent
config_path.is_file
metadata.append
config_path.read_text
np.isfinite
scales.append
np.allclose
matrix.setflags
```

写入目标：

```python
path
data
raw
matrix
dimensions
value
scales
metadata
config_path
record
field
```

直接调用符号：

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

显式输出分支：

```python
found[0]
```

读取的对象属性：

```python
path.is_file
```

写入目标：

```python
files
suffix
found
path
```

直接调用符号：

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

实现说明：

> Read one aligned RGB-D pair; invalid depths remain invalid for selection.
>
> Depth must already be registered to RGB. Intrinsic image dimensions are
> checked, rather than rescaling calibration or guessing a depth alignment.

显式输出分支：

```python
(rgb, depth, camera)
```

读取的对象属性：

```python
Image.open
image.convert
depth_path.suffix
np.load
np.asarray
rgb.size
camera.width
camera.height
raw_depth.shape
camera.depth_scale
```

写入目标：

```python
root
camera
image
rgb
depth_path
raw_depth
depth
```

直接调用符号：

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

显式输出分支：

```python
array
```

读取的对象属性：

```python
np.asarray
array.shape
array.dtype
np.bool_
array.any
```

写入目标：

```python
array
```

直接调用符号：

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

实现说明：

> Backproject metric depth, preserving explicit correspondence identities.
>
> Without ``pixels_uv``, select every finite positive depth (within mask, if
> given); no maximum range is imposed. With explicit N-by-2 (u, v) positions,
> sample nearest-pixel depth and retain subpixel rays. Any invalid selected
> depth, out-of-bounds pixel or mask exclusion raises, rather than dropping a
> row and accidentally changing correspondence IDs. No image mutation occurs.

显式输出分支：

```python
rays * values[:, None]
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
pixels.ndim
pixels.shape
np.isfinite(pixels).all
np.any
np.rint(pixels).astype
np.rint
np.int64
rounded.T
np.all
selected_mask[rows, cols].all
np.ones
np.linalg.inv(matrix).T
np.linalg.inv
np.linalg
```

写入目标：

```python
depth
matrix
selected_mask
valid
rows
cols
pixels
values
rounded
rays
```

直接调用符号：

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

实现说明：

> Three noncollinear correspondences suffice, including planar surfaces.

显式输出分支：

```python
False
bool(spans[1] > max(1e-09, spans[0] * 0.0001))
```

读取的对象属性：

```python
np.linalg.svd
np.linalg
points.mean
```

写入目标：

```python
spans
```

直接调用符号：

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
class _RigidCorrespondences():
```

类字段：

无显式条目。

实现说明：

> Use Open3D's rigid estimator inside correspondence-scored RANSAC.
>
> Scoring must use the supplied feature pairs, not nearest-neighbour cloud
> distance: a wrong feature permutation can still align two cloud surfaces.
> skimage supplies the maintained RANSAC sampling/termination implementation.

## `_RigidCorrespondences.estimate`

[定义位置](../../../manipulation/observations.py#L181)。

输入签名：

```python
def estimate(self, source, target):
```

显式输出分支：

```python
False
bool(np.isfinite(self.params).all())
```

读取的对象属性：

```python
o3d.geometry.PointCloud
o3d.geometry
o3d.utility.Vector3dVector
o3d.utility
np.arange
np.int32
o3d.utility.Vector2iVector
np.column_stack
o3d.pipelines.registration.TransformationEstimationPointToPoint
o3d.pipelines.registration
o3d.pipelines
estimator.compute_transformation
np.isfinite(self.params).all
np.isfinite
self.params
```

写入目标：

```python
first
second
indices
pairs
estimator
self.params
```

直接调用符号：

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
def residuals(self, source, target):
```

显式输出分支：

```python
np.linalg.norm(predicted - target, axis=1)
```

读取的对象属性：

```python
self.params[:3, :3].T
self.params
np.linalg.norm
np.linalg
```

写入目标：

```python
predicted
```

直接调用符号：

```python
np.linalg.norm
```

## `rigid_registration`

[定义位置](../../../manipulation/observations.py#L199)。

输入签名：

```python
def rigid_registration(source, target, max_error=0.01):
```

实现说明：

> Fit a rigid source-to-target transform from known corresponding N-by-3 rows.
>
> Returns ``(T, diagnostics)``. The accepted consensus must contain at least
> half of the supplied correspondences and three noncollinear points; there
> is no rank-three requirement. ``max_error`` is a metric per-pair threshold,
> not an accuracy claim. Failure is explicit, including a nonrigid majority.

显式输出分支：

```python
(transform, {'num_correspondences': len(source), 'num_inliers': int(inliers.sum()), 'inlier_mask': inliers, 'residuals_m': errors, 'inlier_rmse_m': float(np.sqrt(np.mean(errors[inliers] ** 2))), 'max_inlier_error_m': float(errors[inliers].max()), 'source_spans_m': np.linalg.svd(source[inliers] - source[inliers].mean(axis=0), compute_uv=False)})
```

读取的对象属性：

```python
np.asarray
source.ndim
source.shape
target.shape
np.isfinite(source).all
np.isfinite
np.isfinite(target).all
np.random.default_rng
np.random
model.residuals
np.ceil
inliers.sum
np.asarray(model.params).copy
model.params
np.allclose
rotation.T
np.eye
np.isclose
np.linalg.det
np.linalg
np.sqrt
np.mean
errors[inliers].max
np.linalg.svd
source[inliers].mean
```

写入目标：

```python
source
target
model
inliers
errors
required
transform
rotation
```

直接调用符号：

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

实现说明：

> Check the transform convention without repairing invalid predictions.
>
> These checks establish only numerical SE(3) validity. They do not measure
> pose accuracy, object identity, symmetry ambiguity, or tracking success.

显式输出分支：

```python
poses
```

读取的对象属性：

```python
np.asarray
poses.ndim
poses.shape
np.isfinite(poses).all
np.isfinite
np.allclose
np.swapaxes
np.eye
np.linalg.det
np.linalg
```

写入目标：

```python
poses
rotations
```

直接调用符号：

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

实现说明：

> Convert validated model poses to a fixed observed body frame.
>
> ``camera_from_model[t]`` must be FoundationPose's returned transform of the
> *original* mesh, not its internally centered ``pose_last``. ``initial_points``
> are finite metric observations in the first camera frame. With their
> centroid c, P0=[I,c] maps body to camera zero. The single binding
> X=inv(B0)@P0 maps this body into the carrier mesh. For a tracked reference R,
> returned poses are R0@inv(Rt)@Bt@X; without a reference they are Bt@X and
> require a stationary camera. This removes reference motion, including any
> actual reference-object motion, rather than asserting a fixed world frame.
>
> Local points inv(P0)@p0 remain material-point hypotheses throughout the
> sequence. No further depth lookup changes their identity. This function is
> pure coordinate conversion, accepts a single frame, and makes no tracking
> quality claim; the FoundationPose worker performs observation validation.

显式输出分支：

```python
{'poses': poses, 'points': points - origin, 'frame_indices': indices.copy(), 'model_from_body': model_from_body, 'body_origin_camera0_m': origin, 'body_frame': 'initial observed geometry centroid; axes aligned with initial camera', 'reference_frame': 'initial camera coordinates', 'camera_assumption': 'reference motion compensated; reference defines relative motion' if reference is not None else 'stationary camera required', 'units': 'metres', 'pose_convention': 'reference_from_body; column vectors'}
```

读取的对象属性：

```python
np.asarray
points.ndim
points.shape
np.isfinite(points).all
np.isfinite
np.arange
indices.shape
np.issubdtype
indices.dtype
np.integer
(indices < 0).any
(indices[1:] <= indices[:-1]).any
points.mean
np.eye
np.linalg.solve
np.linalg
indices.copy
```

写入目标：

```python
models
points
count
indices
reference
origin
initial
initial[:3, 3]
model_from_body
poses
poses[0]
```

直接调用符号：

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

实现说明：

> Fit independent bodies and preserve their common first-camera gauge.
>
> ``body_poses[owner]`` maps that owner's fixed initial geometry-centroid
> body into camera-zero coordinates with actual object motion retained.
> ``poses`` removes the reference body's motion, if present, using
> Br[0] @ inv(Br[t]) @ Bc[t]. This is relative motion, not a claim that the
> reference is stationary. Raw TAPIP3D predictions remain in their supplied
> world coordinates; they are never overwritten with the rigid projection.

显式输出分支：

```python
{'poses': poses, 'points': local_points['carrier'], 'frame_indices': indices.copy(), 'body_poses': body_poses, 'body_points': local_points, 'body_origin_camera0_m': origins['carrier'], 'camera_from_initial_camera': E @ np.linalg.inv(E[0]), 'body_frame': 'initial observed geometry centroid; axes aligned with initial camera', 'reference_frame': 'initial camera coordinates', 'camera_assumption': 'reference motion compensated; reference defines relative motion' if 'reference' in body_poses else 'supplied camera extrinsics remove camera motion', 'units': 'metres', 'pose_convention': 'reference_from_body; column vectors', 'tracking_method': 'TAPIP3D calibrated RGB-D material tracks + rigid correspondence fit', 'tracking_coords': coords.copy(), 'tracking_query_points': queries.copy(), 'tracking_visibility': visibility.copy(), 'tracking_query_owner': owners.copy(), 'tracking_owner_names': roles, 'tracking_projection_valid': projected.copy(), 'tracking_diagnostics': {'fit_threshold_m': float(max_error), 'per_owner': diagnostics, 'interpretation': 'visibility and fit residuals do not certify pose accuracy'}}
```

读取的对象属性：

```python
np.asarray
tracked.get
coords.ndim
coords.shape
coords.dtype.kind
coords.dtype
np.isfinite(coords).all
np.isfinite
queries.shape
np.isfinite(queries).all
np.all
owners.shape
owners.dtype.kind
owners.dtype
owners.tolist
visibility.shape
visibility.dtype
np.bool_
projected.shape
projected.dtype
np.arange
indices.shape
indices.dtype.kind
indices.dtype
(indices < 0).any
(indices[1:] <= indices[:-1]).any
E[0, :3, :3].T
geometry.ndim
geometry.shape
np.isfinite(geometry).all
np.flatnonzero
estimates.append
records.append
accepted_ids.copy
geometry.mean
np.eye
np.linalg.solve
np.linalg
body_poses['carrier'].copy
indices.copy
np.linalg.inv
coords.copy
queries.copy
visibility.copy
owners.copy
projected.copy
```

写入目标：

```python
roles
coords
queries
owners
visibility
projected
count
query_count
E
indices
camera0_queries
camera0_coords
body_poses
local_points
origins
diagnostics
owner
role
geometry
ids
estimates
records
frame
valid
accepted_ids
estimate
fit
origin
initial
initial[:3, 3]
bodies
bodies[0]
body_poses[role]
local_points[role]
origins[role]
diagnostics[role]
poses
reference
poses[0]
```

直接调用符号：

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

实现说明：

> Recover body motion from local RGB-D without an instance mesh.
>
> Camera stationarity must be declared explicitly, or by the existing
> config.json ``stationary_camera`` boolean. Supplied ``extrinsics`` are
> camera-from-world matrices for this selected frame sequence. The reference
> part has its own tracks and motion; it never supplies camera calibration.

显式输出分支：

```python
result
```

读取的对象属性：

```python
path.stem
(root / 'rgb').iterdir
path.suffix.lower
path.suffix
path.stem.isdigit
depth.shape
np.any
config_path.is_file
json.loads
config_path.read_text
config.get
np.count_nonzero
np.isfinite
camera.K
np.linspace
np.array_equal
current_camera.K
camera.depth_scale
current_camera.depth_scale
frame_paths.append
depth_paths.append
np.stack
np.asarray(tracked.get('coords')).shape
np.asarray
tracked.get
np.asarray(tracked['pixels']).copy
```

写入目标：

```python
root
indices
path
_
depth
camera
body_mask
ref_mask
masks
roles
config_path
config
stationary_camera
E
geometries
role
selected
geometry
geometries[role]
frame_paths
depth_paths
index
current_camera
tracked
result
result['tracking_pixels']
result['camera_assumption']
```

直接调用符号：

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
