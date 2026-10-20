# `manipulation/robot_self_filter.py` 实现契约

代码基线：`pending-implementation-commit`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `RobotCollisionShape`

[定义位置](../../../manipulation/robot_self_filter.py#L28)。

```python
class RobotCollisionShape():
```

类字段：

```python
name: str
kind: str
T_world_geometry: np.ndarray
source: str
dimensions: tuple = ()
vertices: np.ndarray | None = None
```

实现说明：

> One native component in geometry-local coordinates, with a rigid pose.
>
> ``box`` dimensions are full x/y/z lengths; ``cylinder`` dimensions are
> radius/full z-axis length. ``convex`` vertices describe one filled native
> convex component. Scaling is about this explicitly supplied geometry's
> local origin, never its link or world origin; the adapter must preserve
> the native geometry centre when constructing this record.

## `validate_self_filter`

[定义位置](../../../manipulation/robot_self_filter.py#L46)。

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

## `_shape`

[定义位置](../../../manipulation/robot_self_filter.py#L72)。

输入签名：

```python
def _shape(record, scale):
```

实现说明：

> Build a maintained-library solid; no triangle-surface containment test.

显式输出分支：

```python
_convex_solid(vertices * scale)
coal.Box(*dimensions) if record.kind == 'box' else coal.Cylinder(*dimensions)
```

读取的对象属性：

```python
record.kind
record.dimensions
np.asarray
record.vertices
vertices.ndim
vertices.shape
{'box': 3, 'cylinder': 2}.get
dimensions.shape
np.isfinite(dimensions).all
np.isfinite
np.any
coal.Box
coal.Cylinder
```

写入目标：

```python
vertices
dimensions
count
```

直接调用符号：

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

实现说明：

> Enclose the library's local AABB, including its swept-sphere padding.

显式输出分支：

```python
(np.nextafter(centre - extent, -np.inf), np.nextafter(centre + extent, np.inf))
```

读取的对象属性：

```python
shape.computeLocalAABB
np.asarray
shape.aabb_local.min_
shape.aabb_local
shape.aabb_local.max_
np.abs
np.isfinite(np.r_[centre, extent]).all
np.isfinite
np.r_
np.nextafter
np.inf
```

写入目标：

```python
lower
upper
centre
extent
```

直接调用符号：

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

实现说明：

> Place each Coal capsule's z axis along its finite observation segment.

显式输出分支：

```python
np.stack((first, np.cross(directions, first), directions), axis=2)
```

读取的对象属性：

```python
np.zeros_like
np.abs
np.cross
np.linalg.norm
np.linalg
np.stack
```

写入目标：

```python
helper
helper[:, 0]
close
helper[close]
first
```

直接调用符号：

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

实现说明：

> Return three-state labels and conservative per-pixel free-ray depths.
>
> SELF wins when an endpoint is strictly inside any nominal component, even
> if another component shadows it: this is explicitly model-relative point
> ownership. Otherwise a padded finite-segment hit is AMBIGUOUS_SHADOW. Only
> ENVIRONMENT retains evidence. The numerical band uses unchanged Coal
> GJK/EPA defaults as a conservative diagnostic convention, not a proved error
> bound or a sensor accuracy claim. Query exceptions abort; nonfinite results
> remain ambiguous. No GPU, renderer, scene object state or policy is used.

显式输出分支：

```python
{'self_filter': filtered, 'provenance': {'source': source, 'padding_scale': float(padding_scale), 'padding_offset_m': float(padding_offset_m), 'scale_centre': "each component's explicit geometry-local origin; pose translation unchanged", 'padding_definition': 'uniform local scaling followed by Coal Euclidean swept-sphere dilation', 'numerical_boundary_m': band, 'numerical_boundary_is_proved_error_bound': False, 'counts': {**counts, 'self_pixels': int(np.count_nonzero(labels == SELF)), 'environment_pixels': int(np.count_nonzero(labels == ENVIRONMENT)), 'ambiguous_shadow_pixels': int(np.count_nonzero(labels == AMBIGUOUS_SHADOW))}, 'geometries': [{'name': row[0].name, 'kind': row[0].kind, 'source': row[0].source, 'T_world_geometry': np.asarray(row[0].T_world_geometry).tolist()} for row in bodies], 'input_depth_modified': False, 'renderer_used': False, 'free_ray_policy': 'only ENVIRONMENT retains original depth; SELF and AMBIGUOUS_SHADOW are zero', 'self_scope': 'strict interior of declared nominal collision model; not certified physical robot identity', 'shadow_scope': 'closed finite camera-to-observation segment; not an infinite ray'}}
```

读取的对象属性：

```python
np.asarray
depth.ndim
depth.size
np.bool_
np.isscalar
np.isfinite
coal.DistanceRequest
request.gjk_tolerance
request.epa_tolerance
record.name
names.add
record.source
record.T_world_geometry
nominal.clone
padded.setSweptSphereRadius
bodies.append
coal.Transform3s
np.full
np.uint8
np.flatnonzero
depth.ravel
coal.Sphere
coal.Capsule
coal.DistanceResult
np.linalg.inv
np.linalg
np.divmod
depth.shape
np.column_stack
np.ones
inverse_K.T
np.errstate
camera[:3, :3].T
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
query_pose.setRotation
labels.reshape
np.count_nonzero
row[0].name
row[0].kind
row[0].source
np.asarray(row[0].T_world_geometry).tolist
row[0].T_world_geometry
```

写入目标：

```python
depth
matrix
camera
source
request
request.enable_signed_distance
band
bodies
names
record
pose
nominal
padded
labels
valid
counts
point
segment
query_pose
result
origin
inverse_K
start
pixels
rows
cols
rays
endpoints
delta
lengths
finite
state
_
lower
upper
candidates
index
distance
counts['point_queries']
state[index]
counts['nonfinite_queries']
segment_lower
segment_upper
rotations
row
segment.halfLength
counts['segment_queries']
labels[pixels]
filtered
```

直接调用符号：

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
