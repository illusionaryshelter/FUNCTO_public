# `manipulation/robot_self_filter.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `RobotCollisionShape`

[定义位置](../../../manipulation/robot_self_filter.py#L28)。

```python
class RobotCollisionShape:
    name: str
    kind: str
    T_world_geometry: np.ndarray
    source: str
    dimensions: tuple = ()
    vertices: np.ndarray | None = None
```

## `validate_self_filter`

[定义位置](../../../manipulation/robot_self_filter.py#L46)。

输入签名：

```python
def validate_self_filter(value, depth_m):
```

输出分支的源码表达式：

```python
{'labels': labels, 'free_ray_depth_m': free, 'source': source}
```

直接调用：

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

## `_shape`

[定义位置](../../../manipulation/robot_self_filter.py#L72)。

输入签名：

```python
def _shape(record, scale):
```

直接读取的对象属性：`record.kind`、`record.dimensions`、`record.vertices`。

输出分支的源码表达式：

```python
_convex_solid(vertices * scale)
coal.Box(*dimensions) if record.kind == 'box' else coal.Cylinder(*dimensions)
```

直接调用：

```python
ValueError
np.asarray
len
_convex_solid
{'box': 3, 'cylinder': 2}.get
np.isfinite(dimensions).all
np.isfinite
np.any
coal.Box
coal.Cylinder
```

## `_world_bounds`

[定义位置](../../../manipulation/robot_self_filter.py#L94)。

输入签名：

```python
def _world_bounds(shape, pose, numerical_band):
```

输出分支的源码表达式：

```python
(np.nextafter(centre - extent, -np.inf), np.nextafter(centre + extent, np.inf))
```

直接调用：

```python
shape.computeLocalAABB
np.asarray
np.abs
np.isfinite(np.r_[centre, extent]).all
np.isfinite
ValueError
np.nextafter
```

## `_segment_rotations`

[定义位置](../../../manipulation/robot_self_filter.py#L105)。

输入签名：

```python
def _segment_rotations(directions):
```

输出分支的源码表达式：

```python
np.stack((first, np.cross(directions, first), directions), axis=2)
```

直接写入的状态或数组位置：`helper[:, 0]`、`helper[close]`。

直接调用：

```python
np.zeros_like
np.abs
np.cross
np.linalg.norm
np.stack
```

## `classify_depth`

[定义位置](../../../manipulation/robot_self_filter.py#L116)。

输入签名：

```python
def classify_depth(*, observed_depth, K, T_world_camera, geometries, padding_scale, padding_offset_m, source):
```

直接读取的对象属性：`record.name`、`record.source`、`record.T_world_geometry`、`result.clear`。

输出分支的源码表达式：

```python
{'self_filter': filtered, 'provenance': {'source': source, 'padding_scale': float(padding_scale), 'padding_offset_m': float(padding_offset_m), 'scale_centre': "each component's explicit geometry-local origin; pose translation unchanged", 'padding_definition': 'uniform local scaling followed by Coal Euclidean swept-sphere dilation', 'numerical_boundary_m': band, 'numerical_boundary_is_proved_error_bound': False, 'counts': {**counts, 'self_pixels': int(np.count_nonzero(labels == SELF)), 'environment_pixels': int(np.count_nonzero(labels == ENVIRONMENT)), 'ambiguous_shadow_pixels': int(np.count_nonzero(labels == AMBIGUOUS_SHADOW))}, 'geometries': [{'name': row[0].name, 'kind': row[0].kind, 'source': row[0].source, 'T_world_geometry': np.asarray(row[0].T_world_geometry).tolist()} for row in bodies], 'input_depth_modified': False, 'renderer_used': False, 'free_ray_policy': 'only ENVIRONMENT retains original depth; SELF and AMBIGUOUS_SHADOW are zero', 'self_scope': 'strict interior of declared nominal collision model; not certified physical robot identity', 'shadow_scope': 'closed finite camera-to-observation segment; not an infinite ray'}}
```

直接写入的状态或数组位置：`request.enable_signed_distance`、`counts['point_queries']`、`state[index]`、`counts['nonfinite_queries']`、`segment.halfLength`、`counts['segment_queries']`、`labels[pixels]`。

直接调用：

```python
np.asarray
ValueError
_intrinsics
_rigid_pose
_name
isinstance
np.isscalar
np.isfinite
len
coal.DistanceRequest
max
float
set
names.add
_shape
nominal.clone
padded.setSweptSphereRadius
bodies.append
coal.Transform3s
_world_bounds
np.full
np.flatnonzero
depth.ravel
coal.Sphere
coal.Capsule
coal.DistanceResult
np.linalg.inv
range
np.divmod
np.column_stack
np.ones
np.errstate
np.hypot
np.isfinite(endpoints).all
np.where(finite, ENVIRONMENT, AMBIGUOUS_SHADOW).astype
np.where
np.all
query_pose.setIdentity
query_pose.setTranslation
result.clear
coal.distance
np.minimum
np.maximum
_segment_rotations
enumerate
query_pose.setRotation
labels.reshape
validate_self_filter
int
np.count_nonzero
np.asarray(row[0].T_world_geometry).tolist
```
