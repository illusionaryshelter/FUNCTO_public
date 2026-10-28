# `manipulation/robot_geometry_check.py` 实现契约

代码基线：`8b0e326`。
共享验收的数据所有权及调用次序见[共享验收组件](../10-shared-candidate-validation.md)。

职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_name`

[定义位置](../../../manipulation/robot_geometry_check.py#L34)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L40)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L52)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L69)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L86)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L103)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L117)。

```python
class RobotGeometryChecker():
```

类字段：

无显式条目。

实现说明：

> Complete named robot FK and declared native-solid numerical evidence.

## `RobotGeometryChecker.__init__`

[定义位置](../../../manipulation/robot_geometry_check.py#L120)。

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
self._native_pair_table
self._query_schema
pair_count
self.native_query_storage_nbytes
self.native_query_temporary_nbytes
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
max
```

## `RobotGeometryChecker._frame_id`

[定义位置](../../../manipulation/robot_geometry_check.py#L222)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L230)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L235)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L238)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L242)。

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

## `RobotGeometryChecker._prepare_query`

[定义位置](../../../manipulation/robot_geometry_check.py#L257)。

输入签名：

```python
def _prepare_query(self, *, joint_names, joint_positions, T_reference_base, pairs, frames=(), check_request=None, fk_session=None):
```

实现说明：

> Validate one complete named state and compute its float64 FK poses.

显式输出分支：

```python
(names, values, violations, poses, geometry_poses, pair_list)
```

读取的对象属性：

```python
self._query_schema
self.joint_names
self._q_columns
self.joint_limits
np.asarray
values.shape
values.dtype.kind
values.dtype
np.isfinite(values).all
np.isfinite
pin.neutral
self.model
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
pin.forwardKinematics
self._data
pin.updateFramePlacements
self._data.oMf
fk_session.placements
placements[index].homogeneous
frame_ids.items
self._geometries.items
```

写入目标：

```python
names
schema
same_names
columns
limits
name
values
world_base
q
violations
value
column
lower
upper
q[column]
violations[name]
requested_frames
pair_list
selected_frames
frame_ids
identities
pair
identity
self._query_schema
placements
poses
index
geometry_poses
frame
_
local
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
pin.forwardKinematics
pin.updateFramePlacements
fk_session.placements
frame_ids.items
self._geometries.items
```

## `RobotGeometryChecker._pair_table`

[定义位置](../../../manipulation/robot_geometry_check.py#L324)。

输入签名：

```python
def _pair_table(self, pairs):
```

实现说明：

> Return the fixed table after _prepare_query validates the pair schema.

显式输出分支：

```python
(schema[1], schema[2])
```

读取的对象属性：

```python
pair.first
pair.second
self._native_pair_table
dict.fromkeys
np.asarray([(columns[a], columns[b]) for (a, b) in identities], dtype=np.intp).reshape
np.asarray
np.intp
self._geometries
```

写入目标：

```python
identities
pair
schema
self._native_pair_table
names
name
columns
index
indices
a
b
table
```

直接调用符号：

```python
tuple
dict.fromkeys
enumerate
np.asarray([(columns[a], columns[b]) for (a, b) in identities], dtype=np.intp).reshape
np.asarray
NativePairTable
```

## `RobotGeometryChecker.native_query_resident_nbytes`

[定义位置](../../../manipulation/robot_geometry_check.py#L342)。

输入签名：

```python
def native_query_resident_nbytes(self):
```

实现说明：

> Current table storage; model-owned shapes and transient arrays are separate.

显式输出分支：

```python
0
int(schema[2].storage_nbytes) + 256 + len(schema[0]) * 80 + len(schema[1]) * 16
```

读取的对象属性：

```python
self._native_pair_table
schema[2].storage_nbytes
```

写入目标：

```python
schema
```

直接调用符号：

```python
int
len
```

## `RobotGeometryChecker.pair_evidence`

[定义位置](../../../manipulation/robot_geometry_check.py#L349)。

输入签名：

```python
def pair_evidence(self, pair, row):
```

实现说明：

> Attach declared identity and numerical relation to an owned numeric row.
>
> Columns are distance, first/second witness xyz, normal xyz, witness
> identity error, and an optional full-entity separation lower bound.
> Distances and witnesses are metres in the calibrated reference frame;
> Coal's unit direction is dimensionless. Policy remains caller-owned.

显式输出分支：

```python
{'first': pair.first, 'second': pair.second, 'policy': pair.policy, 'policy_source': pair.source, 'signed_distance_m': distance, 'relation': relation, 'witness_first': row[1:4], 'witness_second': row[4:7], 'normal_first_to_second': row[7:10], 'witness_identity_error_m': row[10]}
```

读取的对象属性：

```python
self.numerics
pair.first
pair.second
pair.policy
pair.source
```

写入目标：

```python
boundary
distance
relation
```

直接调用符号：

```python
max
```

## `RobotGeometryChecker.query`

[定义位置](../../../manipulation/robot_geometry_check.py#L367)。

输入签名：

```python
def query(self, *, joint_names, joint_positions, T_reference_base, pairs, frames=(), check_request=None):
```

实现说明：

> Return JSON-ready FK, limits and raw distances for explicit model pairs.
>
> This complete report serves robot validation and execution preflight.
> Pair policy and unknown motion are interpreted by those consumers.

显式输出分支：

```python
{'joint_names': list(names), 'joint_positions': values.astype(float).tolist(), 'joint_limit_violations': violations, 'frame_poses': {name: pose.tolist() for (name, pose) in poses.items()}, 'geometry_poses': {name: pose.tolist() for (name, pose) in geometry_poses.items()}, 'geometry_sources': {name: value[2] for (name, value) in self._geometries.items()}, 'numerics': dict(self.numerics), 'pairs': rows}
```

读取的对象属性：

```python
self._prepare_query
self._pair_table
np.asarray([[geometry_poses[name] for name in geometry_names]], dtype=np.float64).reshape
np.asarray
np.float64
table.query(matrices, np.zeros((1, len(pairs)), bool), self._request)[0].tolist
table.query
np.zeros
self._request
rows.append
self.pair_evidence
values.astype(float).tolist
values.astype
pose.tolist
poses.items
geometry_poses.items
self._geometries.items
self._geometries
self.numerics
```

写入目标：

```python
names
values
violations
poses
geometry_poses
pairs
geometry_names
table
matrices
name
numeric
rows
pair
row
pose
value
```

直接调用符号：

```python
self._prepare_query
self._pair_table
np.asarray([[geometry_poses[name] for name in geometry_names]], dtype=np.float64).reshape
np.asarray
len
check_request
table.query(matrices, np.zeros((1, len(pairs)), bool), self._request)[0].tolist
table.query
np.zeros
zip
rows.append
self.pair_evidence
list
values.astype(float).tolist
values.astype
pose.tolist
poses.items
geometry_poses.items
self._geometries.items
dict
```

## `RobotGeometryChecker.query_pair_batches`

[定义位置](../../../manipulation/robot_geometry_check.py#L398)。

输入签名：

```python
def query_pair_batches(self, *, joint_names, joint_positions, T_reference_base, pairs, projection_mask, check_request=None, fk_session=None, parallel=False):
```

实现说明：

> Yield owned numeric rows [state, pair, 12] in input order.
>
> Every state uses _prepare_query's original float64 FK and matrix
> products. Projection bounds are computed only at true mask entries.
> The scene retains the owned numeric block and materializes only final
> selected evidence. Each chunk has at most MAX_BATCH_SIZE pair evaluations,
> except that a larger single state remains one complete query.
> Cancellation is checked during state preparation, before/after each
> native call, and by the scene while consuming each pair. Errors within
> a native chunk reject it as a whole, with state and pair indices.

显式输出分支：

```python
numeric
```

读取的对象属性：

```python
np.asarray
positions.ndim
positions.shape
positions.dtype.kind
positions.dtype
np.isfinite(positions).all
np.isfinite
mask.dtype
np.bool_
mask.shape
dict.fromkeys
pair.first
pair.second
np.empty
np.float64
self._prepare_query
self._pair_table
table.query
self._request
np.linspace
executor.submit
chunks.append
future.result
np.concatenate
```

写入目标：

```python
joint_names
pairs
positions
mask
state_count
geometry_names
pair
name
executor
begin
end
matrices
offset
position
_
geometry_poses
column
matrices[offset, column]
table
numeric
cuts
futures
left
right
chunks
future
```

直接调用符号：

```python
tuple
np.asarray
len
np.isfinite(positions).all
np.isfinite
ValueError
max
dict.fromkeys
ThreadPoolExecutor
nullcontext
range
min
np.empty
enumerate
self._prepare_query
self._pair_table
check_request
table.query
np.linspace
executor.submit
zip
chunks.append
future.result
RuntimeError
np.concatenate
```
