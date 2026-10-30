# `manipulation/robot_gravity.py` 实现契约

代码基线：`3266476`。
职责和数据语义见[组件说明](../11-device-control-and-evaluation.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `RobotGravity`

[定义位置](../../../manipulation/robot_gravity.py#L14)。

```python
class RobotGravity():
```

类字段：

无显式条目。

实现说明：

> Own scratch dynamics data for an explicitly supplied robot-only model.
>
> ``joint_names`` defines encoder input/output order and ``source`` identifies
> the calibrated fixed model. Revolute coordinates use rad and output N m;
> slide coordinates use m and N.

## `RobotGravity.__init__`

[定义位置](../../../manipulation/robot_gravity.py#L21)。

输入签名：

```python
def __init__(self, model, joint_names, *, source):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
mujoco.MjData
np.asarray
model.joint(name).id
model.joint
self.joint_names
model.jnt_qposadr
model.jnt_dofadr
model.nq
model.nv
self.qpos
np.empty
```

写入目标：

```python
self.model
self.source
self.joint_names
self.data
joints
name
self.qpos
self.dofs
self._force
```

直接调用符号：

```python
str
tuple
mujoco.MjData
np.asarray
model.joint
len
set
ValueError
np.empty
```

## `RobotGravity.compute`

[定义位置](../../../manipulation/robot_gravity.py#L33)。

输入签名：

```python
def compute(self, q):
```

实现说明：

> Evaluate g(q) with kinematics and recursive Newton–Euler dynamics.
>
> Zero velocity and acceleration belong only to this scratch calculation.
> No physical step, scene constraint solve or feedback mutation occurs.
> The caller supplies measured robot q at its current control time.

显式输出分支：

```python
self._force[self.dofs].copy()
```

读取的对象属性：

```python
self.data.qpos
self.data
self.qpos
np.asarray
self.data.qvel
mujoco.mj_kinematics
self.model
mujoco.mj_comPos
mujoco.mj_comVel
mujoco.mj_rne
self._force
self._force[self.dofs].copy
self.dofs
```

写入目标：

```python
self.data.qpos[self.qpos]
self.data.qvel[:]
```

直接调用符号：

```python
np.asarray
mujoco.mj_kinematics
mujoco.mj_comPos
mujoco.mj_comVel
mujoco.mj_rne
self._force[self.dofs].copy
```
