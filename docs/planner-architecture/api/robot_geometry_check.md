# `manipulation/robot_geometry_check.py` 实现契约

代码基线：`__IMPLEMENTATION_COMMIT__`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_name`

[定义位置](../../../manipulation/robot_geometry_check.py#L20)。

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

[定义位置](../../../manipulation/robot_geometry_check.py#L26)。

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

## `PairQuery`

[定义位置](../../../manipulation/robot_geometry_check.py#L38)。

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

> An explicit pair policy, independent of distance or kinematic adjacency.
> 
> Required pairs consume the requested separation. Excluded and mechanical
> contact pairs keep their declaration without manufacturing a distance.

## `RobotGeometryChecker`

[定义位置](../../../manipulation/robot_geometry_check.py#L50)。

```python
class RobotGeometryChecker():
```

类字段：

无显式条目。

实现说明：

> Rigid link poses, named limits and the common implicit-set query.

## `RobotGeometryChecker.__init__`

[定义位置](../../../manipulation/robot_geometry_check.py#L53)。

输入签名：

```python
def __init__(self, urdf_path, *, base_frame, geometries):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
Path(urdf_path).resolve
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
self.model.lowerPositionLimit
self.model.upperPositionLimit
self._q_columns.values
self.model.nq
np.isfinite([lower, upper]).all
np.isfinite
np.any
self._q_columns.items
self.model.velocityLimit
body.name
self.bodies
self._frame_id
body.frame
```

写入目标：

```python
self.urdf_path
self.base_frame
tree
self.urdf_xml
links
node
joints
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
i
self.joint_velocity_limits
self.bodies
body
self.bodies[body.name]
self.numerics
self.native_query_storage_nbytes
```

直接调用符号：

```python
str
Path(urdf_path).resolve
Path
_name
pure_kinematics
ET.tostring
tree.findall
node.find
ValueError
any
pin.buildModelFromUrdf
self.model.createData
tuple
self.model.getJointId
int
set
self._q_columns.values
range
np.isfinite([lower, upper]).all
np.isfinite
np.any
float
self._q_columns.items
isinstance
self._frame_id
```

## `RobotGeometryChecker._frame_id`

[定义位置](../../../manipulation/robot_geometry_check.py#L94)。

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
int
self.model.getFrameId
_name
ValueError
```

## `RobotGeometryChecker.geometry_names`

[定义位置](../../../manipulation/robot_geometry_check.py#L101)。

输入签名：

```python
def geometry_names(self):
```

显式输出分支：

```python
tuple(self.bodies)
```

读取的对象属性：

```python
self.bodies
```

写入目标：

无显式条目。

直接调用符号：

```python
tuple
```

## `RobotGeometryChecker.geometry_sources`

[定义位置](../../../manipulation/robot_geometry_check.py#L105)。

输入签名：

```python
def geometry_sources(self):
```

显式输出分支：

```python
{name: body.source for (name, body) in self.bodies.items()}
```

读取的对象属性：

```python
body.source
self.bodies.items
self.bodies
```

写入目标：

```python
name
body
```

直接调用符号：

```python
self.bodies.items
```

## `RobotGeometryChecker.model_record`

[定义位置](../../../manipulation/robot_geometry_check.py#L108)。

输入签名：

```python
def model_record(self):
```

显式输出分支：

```python
{'urdf_xml': self.urdf_xml, 'geometries': [body.record() for body in self.bodies.values()]}
```

读取的对象属性：

```python
self.urdf_xml
body.record
self.bodies.values
self.bodies
```

写入目标：

```python
body
```

直接调用符号：

```python
body.record
self.bodies.values
```

## `RobotGeometryChecker.geometry_depending_on`

[定义位置](../../../manipulation/robot_geometry_check.py#L112)。

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
self.bodies.items
self.bodies
self.model.frames[self._frame_id(body.frame)].parentJoint
self.model.frames
self.model
self._frame_id
body.frame
self.model.names
result.append
self.model.parents
```

写入目标：

```python
unknown
result
name
body
joint
```

直接调用符号：

```python
set
ValueError
self.bodies.items
int
self._frame_id
str
result.append
tuple
```

## `RobotGeometryChecker.query`

[定义位置](../../../manipulation/robot_geometry_check.py#L126)。

输入签名：

```python
def query(self, *, joint_names, joint_positions, T_reference_base, pairs, frames=(), separation_m=0.002, check_request=None):
```

实现说明：

> Report represented-set evidence; unknown observations stay separate.
> 
> ``clear`` certifies the requested separation only on represented
> observed sets. A missing body does not report a clear distance and
> does not impose a robot-camera-field-of-view restriction.

显式输出分支：

```python
{'joint_names': list(names), 'joint_positions': values.astype(float).tolist(), 'joint_limit_violations': violations, 'frame_poses': {name: pose.tolist() for (name, pose) in poses.items()}, 'geometry_poses': {name: pose.tolist() for (name, pose) in geometry_poses.items()}, 'geometry_sources': self.geometry_sources, 'numerics': dict(self.numerics), 'pairs': rows, 'separation_m': float(separation_m)}
```

读取的对象属性：

```python
np.asarray
self.joint_names
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
body.frame
self.bodies.values
self.bodies
pin.forwardKinematics
self._data
pin.updateFramePlacements
self._data.oMf[self._frame_id(name)].homogeneous
self._data.oMf
self._frame_id
self.bodies.items
pair.first
pair.second
self.bodies.keys
pair.policy
identities.add
pair.source
row.update
first.rigid_component
second.rigid_component
first.field
second.field
first.name
second.name
self.numerics
rows.append
values.astype(float).tolist
values.astype
pose.tolist
poses.items
geometry_poses.items
self.geometry_sources
```

写入目标：

```python
names
values
base
q
violations
name
value
q[self._q_columns[name]]
lower
upper
violations[name]
selected
body
poses
geometry_poses
rows
identities
pair
identity
first
second
row
pose
```

直接调用符号：

```python
tuple
np.asarray
len
set
np.isfinite(values).all
np.isfinite
ValueError
_rigid_pose
pin.neutral
zip
float
dict.fromkeys
self.bodies.values
pin.forwardKinematics
pin.updateFramePlacements
self._frame_id
self.bodies.items
check_request
frozenset
self.bodies.keys
identities.add
_name
row.update
query_distance
rows.append
list
values.astype(float).tolist
values.astype
pose.tolist
poses.items
geometry_poses.items
dict
```
