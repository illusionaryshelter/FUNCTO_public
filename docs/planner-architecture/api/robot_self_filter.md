# `manipulation/robot_self_filter.py` 实现契约

代码基线：`__IMPLEMENTATION_COMMIT__`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `validate_self_filter`

[定义位置](../../../manipulation/robot_self_filter.py#L23)。

输入签名：

```python
def validate_self_filter(value, depth_m):
```

实现说明：

> Validate the only production observation contract, without mutation.
> 
> Unknown/SELF pixels contribute neither measured surfaces nor free rays.
> ENVIRONMENT rays retain exactly their original finite positive depth; an
> endpoint cannot be extended, guessed, or silently restored by a consumer.

显式输出分支：

```python
{'labels': labels, 'free_ray_depth_m': free, 'source': source}
```

读取的对象属性：

```python
np.asarray
depth.ndim
depth.size
labels.dtype
np.uint8
labels.shape
depth.shape
np.isin(labels, (SELF, ENVIRONMENT, AMBIGUOUS_SHADOW)).all
np.isin
free.shape
np.isfinite(free).all
np.isfinite
np.any
np.array_equal
```

写入目标：

```python
depth
labels
free
source
environment
```

直接调用符号：

```python
np.asarray
isinstance
set
ValueError
np.isin(labels, (SELF, ENVIRONMENT, AMBIGUOUS_SHADOW)).all
np.isin
np.isfinite(free).all
np.isfinite
np.any
_name
np.array_equal
```

## `classify_depth`

[定义位置](../../../manipulation/robot_self_filter.py#L50)。

输入签名：

```python
def classify_depth(*, observed_depth, K, T_world_camera, bodies, body_poses, block_checkpoints, depth_error_m, source, max_host_rss_bytes, max_device_used_bytes, device='cuda:0'):
```

实现说明：

> Classify measured endpoints using complete named robot FK at capture.
> 
> A strictly negative supported endpoint is SELF. A supported predicted hit
> before the endpoint, including its declared depth error, is ambiguous.
> Remaining valid pixels keep their original environment depth. Missing
> robot observations are reported without filling their hidden geometry.

显式输出分支：

```python
{'self_filter': filtered, 'predicted_depth_m': np.where(np.isfinite(nearest), nearest, 0.0), 'provenance': {'model': 'same_observed_robot_fields', 'bodies': counts, 'depth_error_m': float(depth_error_m)}}
```

读取的对象属性：

```python
np.asarray
depth.ndim
depth.size
np.isfinite
depth.shape
np.indices
np.stack
np.ones_like
np.linalg.inv(np.asarray(K, float)).T
np.linalg.inv
np.linalg
np.where
world_camera[:3, :3].T
np.where(valid, ENVIRONMENT, AMBIGUOUS_SHADOW).astype
np.uint8
np.full
np.inf
body.field
body.name
field.sample
field.shape
field.voxel_size
np.nextafter
np.zeros
field.origin
torch.tensor
torch.float32
Mapper.load_blocks
mapper.render
tensor.to_device
Pose.from_matrix
rendered.detach().cpu().numpy().reshape
rendered.detach().cpu().numpy
rendered.detach().cpu
rendered.detach
render_valid.detach().cpu().numpy().reshape
render_valid.detach().cpu().numpy
render_valid.detach().cpu
render_valid.detach
camera_local[:3, :3].T
np.minimum
own.sum
accepted.sum
```

写入目标：

```python
depth
height
width
y
x
rays
valid
endpoints_camera
world_camera
endpoints
labels
nearest
tensor
counts
body
counts[body.name]
field
matrix
values
supported
_
error
own
labels[own]
extent
center
config
bound
mapper
camera_local
rendered
render_valid
candidates
hits
supported_hits
accepted
shadow
labels[shadow]
filtered
```

直接调用符号：

```python
np.asarray
np.isfinite
ValueError
np.indices
np.stack
np.ones_like
np.linalg.inv
np.where
np.where(valid, ENVIRONMENT, AMBIGUOUS_SHADOW).astype
np.full
DeviceCfg
field.sample
inverse_points
np.nextafter
np.zeros
MapperCfg
tuple
torch.tensor
_mapper_memory_bound
require_host_memory
require_device_memory
Mapper.load_blocks
mapper.render
tensor.to_device
_mapper_intrinsics
Pose.from_matrix
rendered.detach().cpu().numpy().reshape
rendered.detach().cpu().numpy
rendered.detach().cpu
rendered.detach
render_valid.detach().cpu().numpy().reshape
render_valid.detach().cpu().numpy
render_valid.detach().cpu
render_valid.detach
np.minimum
int
own.sum
accepted.sum
validate_self_filter
float
```
