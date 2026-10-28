# `manipulation/observed_robot.py` 实现契约

代码基线：`8b0e326`。
共享验收的数据所有权及调用次序见[共享验收组件](../10-shared-candidate-validation.md)。

职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `pure_kinematics`

[定义位置](../../../manipulation/observed_robot.py#L11)。

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

[定义位置](../../../manipulation/observed_robot.py#L26)。

```python
class DifferentiableRobotKinematics():
```

类字段：

无显式条目。

实现说明：

> Place native entities with the already registered Pinocchio model.
>
> The planning solver may optimize only the arm. The scene appends the
> already selected measured/nominal finger coordinates before invoking this
> complete model. Every call uses the same CPU float64 Pinocchio algorithm.
> Tensor copies connect CUDA/float32 proposal inputs to that algorithm and
> restore its output to the input device/dtype while retaining autograd.
>
> The official robot and fixed end-effector configuration are reusable
> across tasks because their shape and component relationships remain
> fixed. This adapter receives the validator's existing kinematic model;
> its poses come from named joints and the same base calibration.

## `DifferentiableRobotKinematics.__init__`

[定义位置](../../../manipulation/observed_robot.py#L42)。

输入签名：

```python
def __init__(self, model, *, frames, T_reference_base):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._bridge.joint_names
self._bridge
self._bridge.frames
```

写入目标：

```python
self.model
self._bridge
self.joint_names
self.frames
```

直接调用符号：

```python
PinKinematics
```

## `DifferentiableRobotKinematics.poses`

[定义位置](../../../manipulation/observed_robot.py#L48)。

输入签名：

```python
def poses(self, position, joint_names, *, fk_session=None):
```

实现说明：

> Return original-device link poses with the same named-joint VJP.

显式输出分支：

```python
{frame: tuple((value.to(device=position.device, dtype=position.dtype) for value in pose)) for (frame, pose) in poses.items()}
```

读取的对象属性：

```python
position.to
torch.float64
self._bridge.poses
self._bridge
value.to
position.device
position.dtype
poses.items
```

写入目标：

```python
cpu_position
poses
value
frame
pose
```

直接调用符号：

```python
position.to
self._bridge.poses
tuple
value.to
poses.items
```
