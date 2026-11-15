# `transfer/cross_instance/response.py` 实现契约

代码基线：`96ee2dc-working-tree`。
职责和数据语义见[组件说明](../../../transfer/cross_instance/ARCHITECTURE.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `sample_native_pixels`

[定义位置](../../../transfer/cross_instance/response.py#L26)。

输入签名：

```python
def sample_native_pixels(response, pixels, crop_xyxy):
```

实现说明：

> Return [Q,N] native values, with a Torch graph for continuous pixels.
>
> Precondition: callers qualify image/crop/grid and observed geometry before
> treating these values as evidence. ``grid_sample`` padding is numerical
> behavior only. The public observed query below samples qualified points
> and leaves every unsupported response NaN.
>
> ``align_corners=True`` turns the target feature coordinate
> ``local * resize_scale * (F-1) / 839`` into the normalized coordinate below.
> The native dot-product grid remains unchanged; interpolation uses float64.

显式输出分支：

```python
sampled[0, :, :, 0]
```

读取的对象属性：

```python
pixels.to
torch.float64
np.asarray
torch.as_tensor
pixels.dtype
pixels.device
functional.grid_sample
```

写入目标：

```python
pixels
crop
offset
scale
grid
values
sampled
```

直接调用符号：

```python
pixels.to
np.asarray
torch.as_tensor
resize_scales
functional.grid_sample
```

## `sample_native_world`

[定义位置](../../../transfer/cross_instance/response.py#L50)。

输入签名：

```python
def sample_native_world(response, world, *, crop_xyxy, K, T_world_camera_cv):
```

实现说明：

> Sample prequalified world points without detaching their Torch graph.
>
> Position derivatives describe the current valid interpolation piece.
> Owner labels, missing depth and depth discontinuities are discrete support
> conditions evaluated by ``query_view``; their boundaries have no derivative
> supplied by this function.

显式输出分支：

```python
sample_native_pixels(response, pixels, crop_xyxy)
```

读取的对象属性：

```python
world.to
torch.float64
torch.as_tensor
world.dtype
world.device
intrinsic.T
```

写入目标：

```python
world
pose
intrinsic
camera
homogeneous
pixels
```

直接调用符号：

```python
world.to
torch.as_tensor
sample_native_pixels
```

## `project_world`

[定义位置](../../../transfer/cross_instance/response.py#L68)。

输入签名：

```python
def project_world(world, K, T_world_camera_cv):
```

实现说明：

> Project world metres to original CV pixels and optical depth metres.

显式输出分支：

```python
(pixels, camera[:, 2])
```

读取的对象属性：

```python
np.asarray
np.asarray(K).T
np.errstate
np.rint
np.finfo(float).eps
np.finfo
np.maximum
np.abs
np.where
```

写入目标：

```python
pose
camera
homogeneous
pixels
nearest
roundoff
```

直接调用符号：

```python
np.asarray
np.errstate
np.rint
np.finfo
np.maximum
np.abs
np.where
```

## `query_view`

[定义位置](../../../transfer/cross_instance/response.py#L84)。

输入签名：

```python
def query_view(world, view, *, depth_error_m, max_depth_slope):
```

实现说明：

> Qualify RGB-D support and sample this camera's native response grid.
>
> Positive image-pixel bilinear corners must have observed owner depth.
> Their optical-depth range, expanded by the recorded measurement error,
> must contain the query depth. Pairwise depth changes use the original
> transverse-ray slope bound. This is local observation compatibility, not
> a reconstructed surface, normal estimate or unique material association.
> Every requested point retains a state and original measurement identities.

显式输出分支：

```python
dict(view_id=view['id'], response=values, valid=pending, unknown=~pending, reason=reason, projected_pixels=pixels, optical_depth_m=optical_depth, depth_interval_m=np.column_stack((depth_low, depth_high)), depth_pixel_xy=corners, depth_weights=weights, observation_ids=observation_ids, depth_footprint_supported=depth_supported, owner_supported=owner_supported, slope_supported=slope_supported, feature_xy=feature_xy, feature_corner_xy=feature_corners, feature_weights=feature_weights)
```

读取的对象属性：

```python
np.asarray
depth.shape
np.clip
np.isfinite
np.all
np.min
np.where
np.inf
np.max
np.linalg.inv
np.linalg
np.ones
np.column_stack
np.zeros
inverse_intrinsic.T
np.minimum
np.linalg.norm
np.abs
np.full
np.isfinite(pixels).all
response.shape
np.array
np.nan
np.flatnonzero
torch.as_tensor
torch.float64
sample_native_pixels(response, selected_pixels, crop).detach().numpy
sample_native_pixels(response, selected_pixels, crop).detach
```

写入目标：

```python
world
pixels
optical_depth
depth
owner
lookup
image_inside
corners
weights
positive
height
width
px
py
corner_depth
corner_observation
measured
depth_supported
owner_supported
depth_low
depth_high
inverse_intrinsic
slope_supported
left
right
delta_pixel
delta_ray
transverse
bound
active_pair
reason
pending
finite
response
crop
local
crop_inside
feature_shape
feature_xy
feature_inside
feature_corners
feature_weights
values
active
selected_pixels
values[:, active]
observation_ids
```

直接调用符号：

```python
np.asarray
project_world
_sampling_corners
np.clip
np.isfinite
np.all
np.min
np.where
np.max
np.linalg.inv
np.ones
len
range
np.column_stack
np.zeros
np.minimum
np.linalg.norm
np.abs
np.full
np.isfinite(pixels).all
reject
resize_scales
np.array
np.flatnonzero
torch.as_tensor
sample_native_pixels(response, selected_pixels, crop).detach().numpy
sample_native_pixels(response, selected_pixels, crop).detach
sample_native_pixels
dict
```

## `query_view.reject`

[定义位置](../../../transfer/cross_instance/response.py#L128)。

输入签名：

```python
def reject(condition, label):
```

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
selected
reason[selected]
pending[selected]
```

直接调用符号：

无显式条目。

## `query_views`

[定义位置](../../../transfer/cross_instance/response.py#L168)。

输入签名：

```python
def query_views(world, views, *, source_row_ids, depth_error_m, max_depth_slope):
```

实现说明：

> Return mean raw evidence for one point domain, retaining every view.
>
> View support counts normalize appearance evidence only. They do not create
> geometric samples, target capacity or a new owner statistic. No-view points
> remain unknown/NaN; adding the absent camera never creates zero evidence.

显式输出分支：

```python
dict(response=fused, valid=count > 0, unknown=count == 0, source_row_ids=np.asarray(source_row_ids), world_points=np.asarray(world), view_ids=np.asarray([entry['view_id'] for entry in per_view]), view_response=values, view_valid=valid, view_count=count, views=per_view)
```

读取的对象属性：

```python
np.stack
valid.sum
np.where(valid[None], values, 0.0).sum
np.where
np.full
total.shape
np.nan
np.divide
np.asarray
```

写入目标：

```python
per_view
view
valid
entry
values
count
total
fused
```

直接调用符号：

```python
query_view
np.stack
valid.sum
np.where(valid[None], values, 0.0).sum
np.where
np.full
np.divide
dict
np.asarray
```
