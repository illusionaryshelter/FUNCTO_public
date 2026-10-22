# `manipulation/robot_geometry_check.py` 实现契约

代码基线：`a56c3e4`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_name`

[定义位置](../../../manipulation/robot_geometry_check.py#L29)。

输入签名：

```python
def _name(value, label):
```

显式输出分支：

```python
value
```

读取的对象属性：

```python
value.strip
```

写入目标：

无显式条目。

直接调用符号：

```python
isinstance
value.strip
ValueError
```

## `_rigid_pose`

[定义位置](../../../manipulation/robot_geometry_check.py#L35)。

输入签名：

```python
def _rigid_pose(value, label='T_reference_base'):
```

显式输出分支：

```python
pose.astype(float, copy=True)
```

读取的对象属性：

```python
np.asarray
pose.shape
pose.dtype.kind
pose.dtype
np.isfinite(pose).all
np.isfinite
np.allclose
pose[:3, :3].T
np.eye
np.isclose
np.linalg.det
np.linalg
pose.astype
```

写入目标：

```python
pose
```

直接调用符号：

```python
np.asarray
np.isfinite(pose).all
np.isfinite
np.allclose
np.eye
np.isclose
np.linalg.det
ValueError
pose.astype
```

## `NativeSolid`

[定义位置](../../../manipulation/robot_geometry_check.py#L47)。

```python
class NativeSolid():
```

类字段：

```python
name: str
frame: str
vertices_frame: np.ndarray
source: str
```

实现说明：

> One native convex solid, with geometry transforms already baked in.
>
> Compiled mesh scale/recentering must be applied exactly once by the source
> adapter. Baking the compiled geom transform into these vertices also keeps
> any source affine scaling out of Coal's rigid Transform3s. A native box is
> represented exactly by its eight corners. Curved primitives require their
> own exact representation and must not be silently tessellated here.

## `PairQuery`

[定义位置](../../../manipulation/robot_geometry_check.py#L64)。

```python
class PairQuery():
```

类字段：

```python
first: str
second: str
policy: str
source: str
```

实现说明：

> Caller-supplied policy identity; even excluded pairs retain raw results.
>
> ``required`` must remain separated; ``mechanical_contact`` explicitly
> permits contact between exactly these two robot solids within their
> independently checked mechanical joint domain. ``excluded`` records native
> filtering; ``diagnostic`` makes no acceptance claim. All retain distances.
> No pair is inferred from adjacency, a robot name, or a task name.

## `NativePrimitive`

[定义位置](../../../manipulation/robot_geometry_check.py#L81)。

```python
class NativePrimitive():
```

类字段：

```python
name: str
frame: str
shape_kind: str
dimensions: tuple
T_frame_geometry: np.ndarray
source: str
```

实现说明：

> An analytic native box or cylinder, without mesh approximation.
>
> Box dimensions are full x/y/z lengths. Cylinder dimensions are radius and
> full axial length; the local cylinder axis is z. All lengths are metres and
> T_frame_geometry is a proper rigid pose. Explicitly retain a separate geom
> for every component instead of taking a hull over the assembly.

## `_convex_solid`

[定义位置](../../../manipulation/robot_geometry_check.py#L98)。

输入签名：

```python
def _convex_solid(vertices):
```

显式输出分支：

```python
coal.ConvexBase.convexHull(native, True, 'Qt')
```

读取的对象属性：

```python
np.asarray
points.ndim
points.shape
points.dtype.kind
points.dtype
np.isfinite(points).all
np.isfinite
np.linalg.matrix_rank
np.linalg
points.mean
coal.StdVec_Vec3s
native.append
coal.ConvexBase.convexHull
coal.ConvexBase
```

写入目标：

```python
points
native
point
```

直接调用符号：

```python
np.asarray
len
np.isfinite(points).all
np.isfinite
np.linalg.matrix_rank
points.mean
ValueError
coal.StdVec_Vec3s
native.append
coal.ConvexBase.convexHull
```

## `RobotGeometryChecker`

[定义位置](../../../manipulation/robot_geometry_check.py#L112)。

```python
class RobotGeometryChecker():
```

类字段：

无显式条目。

实现说明：

> Complete named robot FK and declared native-solid numerical evidence.

## `RobotGeometryChecker.__init__`

[定义位置](../../../manipulation/robot_geometry_check.py#L115)。

输入签名：

```python
def __init__(self, urdf_path, *, base_frame, geometries):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
Path(urdf_path).resolve
ET.parse(self.urdf_path).getroot
ET.parse
self.urdf_path
ET.tostring
node.attrib
tree.findall
node.find('child').attrib
node.find
pin.buildModelFromUrdf
self.model.createData
self.model
self.model.names
self.joint_names
self.model.joints
self.model.getJointId
joint.nq
joint.nv
self._q_columns
joint.idx_q
self._q_columns.values
self.model.nq
self.model.lowerPositionLimit
self.model.upperPositionLimit
np.isfinite(lower).all
np.isfinite
np.isfinite(upper).all
np.any
self._q_columns.items
self.model.velocityLimit
record.name
self._geometries
self._frame_id
record.frame
record.vertices_frame
np.eye
np.asarray
record.dimensions
{'box': 3, 'cylinder': 2}.get
record.shape_kind
dimensions.shape
dimensions.dtype.kind
dimensions.dtype
np.isfinite(dimensions).all
coal.Box
coal.Cylinder
record.T_frame_geometry
record.source
identity.update
np.asarray(record.vertices_frame, float).tolist
local_pose.tolist
self._geometry_model.append
self._geometry_model
coal.DistanceRequest
self._request
self._request.gjk_tolerance
self._request.epa_tolerance
name.encode
self._request.epa_max_iterations
```

写入目标：

```python
self.urdf_path
self.base_frame
tree
self.urdf_xml
self._geometry_model
links
node
joints
joint_names
roots
self.model
self._data
self.joint_names
name
self._q_columns
joint
self._q_columns[name]
lower
upper
self.joint_limits
column
self.joint_velocity_limits
self._geometries
record
shape
local_pose
dimensions
count
value
self._geometries[name]
identity
self._request
self._request.enable_signed_distance
self.numerics
self._native_distance_batches
pair_count
block_geometries
name_bytes
self.native_query_storage_nbytes
```

直接调用符号：

```python
str
Path(urdf_path).resolve
Path
_name
ET.parse(self.urdf_path).getroot
ET.parse
ET.tostring
tree.findall
len
set
ValueError
node.find
pin.buildModelFromUrdf
self.model.createData
tuple
self.model.getJointId
int
self._q_columns.values
range
np.isfinite(lower).all
np.isfinite
np.isfinite(upper).all
np.any
float
self._q_columns.items
self._frame_id
isinstance
_convex_solid
np.eye
np.asarray
{'box': 3, 'cylinder': 2}.get
np.isfinite(dimensions).all
coal.Box
coal.Cylinder
_rigid_pose
identity.update
np.asarray(record.vertices_frame, float).tolist
list
local_pose.tolist
self._geometry_model.append
coal.DistanceRequest
min
sum
name.encode
```

## `RobotGeometryChecker._frame_id`

[定义位置](../../../manipulation/robot_geometry_check.py#L215)。

输入签名：

```python
def _frame_id(self, name):
```

显式输出分支：

```python
index
```

读取的对象属性：

```python
self.model.getFrameId
self.model
self.model.nframes
```

写入目标：

```python
index
```

直接调用符号：

```python
_name
int
self.model.getFrameId
ValueError
```

## `RobotGeometryChecker.geometry_names`

[定义位置](../../../manipulation/robot_geometry_check.py#L223)。

输入签名：

```python
def geometry_names(self):
```

实现说明：

> Stable native-solid order; an acceptance layer must cover all pairs.

显式输出分支：

```python
tuple(self._geometries)
```

读取的对象属性：

```python
self._geometries
```

写入目标：

无显式条目。

直接调用符号：

```python
tuple
```

## `RobotGeometryChecker.geometry_sources`

[定义位置](../../../manipulation/robot_geometry_check.py#L228)。

输入签名：

```python
def geometry_sources(self):
```

显式输出分支：

```python
{name: value[2] for (name, value) in self._geometries.items()}
```

读取的对象属性：

```python
self._geometries.items
self._geometries
```

写入目标：

```python
name
value
```

直接调用符号：

```python
self._geometries.items
```

## `RobotGeometryChecker.model_record`

[定义位置](../../../manipulation/robot_geometry_check.py#L231)。

输入签名：

```python
def model_record(self):
```

实现说明：

> Serialize the consumed URDF and every explicit native solid input.

显式输出分支：

```python
{'urdf_xml': self.urdf_xml, 'geometries': deepcopy(self._geometry_model)}
```

读取的对象属性：

```python
self.urdf_xml
self._geometry_model
```

写入目标：

无显式条目。

直接调用符号：

```python
deepcopy
```

## `RobotGeometryChecker.geometry_depending_on`

[定义位置](../../../manipulation/robot_geometry_check.py#L235)。

输入签名：

```python
def geometry_depending_on(self, joint_names):
```

显式输出分支：

```python
tuple(result)
```

读取的对象属性：

```python
self.joint_names
self._geometries.items
self._geometries
self.model.frames[self._frame_id(frame)].parentJoint
self.model.frames
self.model
self._frame_id
self.model.names
result.append
self.model.parents
```

写入目标：

```python
unknown
result
name
frame
_
joint
```

直接调用符号：

```python
set
ValueError
self._geometries.items
int
self._frame_id
str
result.append
tuple
```

## `RobotGeometryChecker.query`

[定义位置](../../../manipulation/robot_geometry_check.py#L249)。

输入签名：

```python
def query(self, *, joint_names, joint_positions, T_reference_base, pairs, frames=(), check_request=None):
```

实现说明：

> Return JSON-ready FK, raw signed distances, witnesses and limits.
>
> Only the explicitly requested pairs are queried. This function reports
> numerical geometry relations without an overall collision-valid flag:
> policy exclusions, unknown space and continuous motion belong to the
> original caller, and no inferred ignore can enter this diagnostic.

显式输出分支：

```python
{'joint_names': list(names), 'joint_positions': values.astype(float).tolist(), 'joint_limit_violations': violations, 'frame_poses': {name: pose.tolist() for (name, pose) in poses.items()}, 'geometry_poses': {name: pose.tolist() for (name, pose) in geometry_poses.items()}, 'geometry_sources': {name: value[2] for (name, value) in self._geometries.items()}, 'numerics': dict(self.numerics), 'pairs': rows}
```

读取的对象属性：

```python
self.joint_names
np.asarray
values.shape
values.dtype.kind
values.dtype
np.isfinite(values).all
np.isfinite
pin.neutral
self.model
self._q_columns
self.joint_limits
dict.fromkeys
self.base_frame
self._geometries.values
self._geometries
self._frame_id
pair.first
pair.second
pair.policy
pair.source
identities.add
pair_list.append
pin.forwardKinematics
self._data
pin.updateFramePlacements
self._data.oMf[index].homogeneous
self._data.oMf
frame_ids.items
self._geometries.items
self._native_pair_distances
self.numerics
result.min_distance
result.getNearestPoint1
result.getNearestPoint2
result.normal
np.isfinite(np.concatenate(([distance], first, second, normal))).all
np.concatenate
rows.append
first.tolist
second.tolist
normal.tolist
np.linalg.norm
np.linalg
values.astype(float).tolist
values.astype
pose.tolist
poses.items
geometry_poses.items
```

写入目标：

```python
names
name
values
world_base
q
violations
value
q[self._q_columns[name]]
lower
upper
violations[name]
selected_frames
frame_ids
pair_list
identities
pair
identity
poses
index
geometry_poses
frame
_
local
distances
rows
boundary
result
distance
first
second
normal
relation
pose
```

直接调用符号：

```python
check_request
tuple
any
isinstance
len
set
ValueError
np.asarray
np.isfinite(values).all
np.isfinite
_rigid_pose
pin.neutral
zip
float
dict.fromkeys
self._geometries.values
self._frame_id
frozenset
_name
identities.add
pair_list.append
pin.forwardKinematics
pin.updateFramePlacements
frame_ids.items
self._geometries.items
self._native_pair_distances
max
result.getNearestPoint1
result.getNearestPoint2
np.isfinite(np.concatenate(([distance], first, second, normal))).all
np.concatenate
RuntimeError
rows.append
first.tolist
second.tolist
normal.tolist
np.linalg.norm
list
values.astype(float).tolist
values.astype
pose.tolist
poses.items
geometry_poses.items
dict
```

## `RobotGeometryChecker._native_pair_distances`

[定义位置](../../../manipulation/robot_geometry_check.py#L334)。

输入签名：

```python
def _native_pair_distances(self, pairs, poses):
```

实现说明：

> Yield original-order results without retaining warm solver state.
>
> Pinocchio's geometry model has a dense geometry-to-pair map. Limiting
> each workspace to 128 pairs (at most 256 geometries) bounds that map to
> 256 by 256 integer entries even for a large, sparse pair list. Every block
> of the most recent complete pair schema is retained, so larger fixed
> schemas do not rebuild blocks on every state. Changing pair order
> releases the old schema before allocating its replacement.
> All shape allocations finish before functors retain GeometryObject
> addresses. The geometry model outlives both the working and pristine
> functor vectors.
>
> A batch may perform later independent distance work before Python checks
> an earlier result. Every nonfinite result still rejects the whole call;
> native exceptions propagate with Pinocchio's pair context and no partial
> report is returned. This interface never owns scene contact permissions.

显式输出分支：

```python
data.distanceResults[index]
```

读取的对象属性：

```python
pair.first
pair.second
self._native_distance_batches
dict.fromkeys
pin.GeometryModel
model.addGeometryObject
pin.GeometryObject
pin.SE3.Identity
pin.SE3
self._geometries
model.addCollisionPair
pin.CollisionPair
pin.GeometryData
data.distanceRequests
self._request
data.distance_functors.copy
data.distance_functors
blocks.append
data.oMg
pin.computeDistances
model.collisionPairs
data.distanceResults
```

写入目标：

```python
identities
pair
schema
self._native_distance_batches
blocks
start
block_pairs
names
name
columns
index
model
first
second
data
data.distanceRequests[index]
pristine
data.distance_functors
pose
data.oMg[index]
```

直接调用符号：

```python
tuple
range
len
dict.fromkeys
enumerate
pin.GeometryModel
model.addGeometryObject
pin.GeometryObject
pin.SE3.Identity
model.addCollisionPair
pin.CollisionPair
pin.GeometryData
data.distance_functors.copy
blocks.append
pin.SE3
pin.computeDistances
```
