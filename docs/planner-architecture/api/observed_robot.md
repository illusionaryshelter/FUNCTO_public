# `manipulation/observed_robot.py` 实现契约

代码基线：`__IMPLEMENTATION_COMMIT__`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `ObservedBody`

[定义位置](../../../manipulation/observed_robot.py#L16)。

```python
class ObservedBody():
```

类字段：

```python
name: str
frame: str
field: FrozenObservedField | None
rigid_component: str
source: str
```

实现说明：

> An observed component rigidly fixed in the named link coordinate frame.

## `ObservedBody.record`

[定义位置](../../../manipulation/observed_robot.py#L25)。

输入签名：

```python
def record(self):
```

显式输出分支：

```python
{'name': self.name, 'frame': self.frame, 'field': None if self.field is None else self.field.record(), 'rigid_component': self.rigid_component, 'source': self.source}
```

读取的对象属性：

```python
self.name
self.frame
self.field
self.field.record
self.rigid_component
self.source
```

写入目标：

无显式条目。

直接调用符号：

```python
self.field.record
```

## `ObservedBody.from_record`

[定义位置](../../../manipulation/observed_robot.py#L31)。

输入签名：

```python
def from_record(cls, row):
```

显式输出分支：

```python
cls(row['name'], row['frame'], None if row['field'] is None else FrozenObservedField.from_record(row['field']), row['rigid_component'], row['source'])
```

读取的对象属性：

```python
FrozenObservedField.from_record
```

写入目标：

无显式条目。

直接调用符号：

```python
cls
FrozenObservedField.from_record
```

## `pure_kinematics`

[定义位置](../../../manipulation/observed_robot.py#L37)。

输入签名：

```python
def pure_kinematics(path):
```

实现说明：

> Read the declared kinematic tree and reject geometry-bearing inputs.

显式输出分支：

```python
root
```

读取的对象属性：

```python
ET.parse(path).getroot
ET.parse
root.findall
```

写入目标：

```python
root
```

直接调用符号：

```python
ET.parse(path).getroot
ET.parse
root.findall
ValueError
```

## `DifferentiableRobotKinematics`

[定义位置](../../../manipulation/observed_robot.py#L45)。

```python
class DifferentiableRobotKinematics():
```

类字段：

无显式条目。

实现说明：

> Use all named joints, including nominal fingers, for observed link poses.
> 
> The planning solver may optimize only the arm. The scene appends the
> already selected measured/nominal finger coordinates before invoking this
> complete tree. No future finger pose is obtained from the solver's locked
> entrance configuration.

## `DifferentiableRobotKinematics.__init__`

[定义位置](../../../manipulation/observed_robot.py#L54)。

输入签名：

```python
def __init__(self, configuration, *, frames, T_reference_base, device):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
KinematicsCfg.from_basic_urdf
self.model.joint_names
self.model
np.asarray
base[:3, 3].copy
Rotation.from_matrix(base[:3, :3]).as_quat
Rotation.from_matrix
```

写入目标：

```python
kinematics
native
self.model
self.joint_names
base
self.position
self.quaternion
```

直接调用符号：

```python
pure_kinematics
KinematicsCfg.from_basic_urdf
list
DeviceCfg
str
Kinematics
tuple
np.asarray
base[:3, 3].copy
Rotation.from_matrix(base[:3, :3]).as_quat
Rotation.from_matrix
```

## `DifferentiableRobotKinematics.poses`

[定义位置](../../../manipulation/observed_robot.py#L71)。

输入签名：

```python
def poses(self, position, joint_names):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
self.joint_names
position[..., [names.index(name) for name in self.joint_names]].contiguous
names.index
self.model.compute_kinematics
self.model
JointState.from_position
torch.as_tensor
self.position
position.dtype
position.device
self.quaternion
state.tool_poses
tools.tool_frames
tools.quaternion
tools.position
```

写入目标：

```python
names
ordered
name
state
base_p
base_q
tools
result
index
frame
q
p
result[frame]
```

直接调用符号：

```python
tuple
set
ValueError
position[..., [names.index(name) for name in self.joint_names]].contiguous
names.index
self.model.compute_kinematics
JointState.from_position
list
torch.as_tensor
enumerate
quaternion_rotate
quaternion_multiply
```
