# `manipulation/pin_kinematics.py` 实现契约

代码基线：`be44b54`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_pose_vjp`

[定义位置](../../../manipulation/pin_kinematics.py#L18)。

输入签名：

```python
def _pose_vjp(pose_gradient, quaternion, jacobian):
```

实现说明：

> Convert Euclidean wxyz covectors to reference-frame angular covectors.
>
> For angular displacement omega in reference axes, the unit quaternion
> obeys dQ = .5*[0,omega] tensor_product Q. The cross-product sign below
> follows that left product. Pinocchio supplies the actual robot Jacobian;
> this conversion only changes the coordinates of its output covector.

显式输出分支：

```python
(jacobian * spatial.unsqueeze(-1)).sum(dim=(-3, -2))
```

读取的对象属性：

```python
torch.linalg.cross
torch.linalg
torch.cat
(jacobian * spatial.unsqueeze(-1)).sum
spatial.unsqueeze
```

写入目标：

```python
scalar
vector
gradient
angular
spatial
```

直接调用符号：

```python
torch.linalg.cross
torch.cat
(jacobian * spatial.unsqueeze(-1)).sum
spatial.unsqueeze
```

## `_PinPoses`

[定义位置](../../../manipulation/pin_kinematics.py#L38)。

```python
class _PinPoses(torch.autograd.Function):
```

类字段：

无显式条目。

## `_PinPoses.forward`

[定义位置](../../../manipulation/pin_kinematics.py#L40)。

输入签名：

```python
def forward(ctx, position, kinematics):
```

显式输出分支：

```python
output
```

读取的对象属性：

```python
kinematics._evaluate
position.detach().numpy
position.detach
torch.from_numpy
ctx.save_for_backward
```

写入目标：

```python
poses
jacobian
output
```

直接调用符号：

```python
kinematics._evaluate
position.detach().numpy
position.detach
torch.from_numpy
ctx.save_for_backward
```

## `_PinPoses.backward`

[定义位置](../../../manipulation/pin_kinematics.py#L47)。

输入签名：

```python
def backward(ctx, gradient):
```

显式输出分支：

```python
(_pose_vjp(gradient, quaternion, jacobian), None)
```

读取的对象属性：

```python
ctx.saved_tensors
```

写入目标：

```python
quaternion
jacobian
```

直接调用符号：

```python
_pose_vjp
```

## `_PinStates`

[定义位置](../../../manipulation/pin_kinematics.py#L52)。

```python
class _PinStates(torch.autograd.Function):
```

类字段：

无显式条目。

## `_PinStates.forward`

[定义位置](../../../manipulation/pin_kinematics.py#L54)。

输入签名：

```python
def forward(ctx, position, velocity, kinematics):
```

显式输出分支：

```python
(output, torch.from_numpy(twists))
```

读取的对象属性：

```python
kinematics._evaluate
position.detach().numpy
position.detach
velocity.detach().numpy
velocity.detach
torch.from_numpy
ctx.save_for_backward
```

写入目标：

```python
poses
jacobian
twists
velocity_position
velocity_velocity
output
```

直接调用符号：

```python
kinematics._evaluate
position.detach().numpy
position.detach
velocity.detach().numpy
velocity.detach
torch.from_numpy
ctx.save_for_backward
```

## `_PinStates.backward`

[定义位置](../../../manipulation/pin_kinematics.py#L63)。

输入签名：

```python
def backward(ctx, pose_gradient, twist_gradient):
```

显式输出分支：

```python
(position_gradient, velocity_gradient, None)
```

读取的对象属性：

```python
ctx.saved_tensors
(velocity_position * twist_gradient.unsqueeze(-1)).sum
twist_gradient.unsqueeze
(velocity_velocity * twist_gradient.unsqueeze(-1)).sum
```

写入目标：

```python
quaternion
jacobian
velocity_position
velocity_velocity
position_gradient
velocity_gradient
```

直接调用符号：

```python
_pose_vjp
(velocity_position * twist_gradient.unsqueeze(-1)).sum
twist_gradient.unsqueeze
(velocity_velocity * twist_gradient.unsqueeze(-1)).sum
```

## `PinKinematics`

[定义位置](../../../manipulation/pin_kinematics.py#L73)。

```python
class PinKinematics():
```

类字段：

无显式条目。

实现说明：

> Reference-frame kinematics for a complete fixed-base scalar joint model.
>
> Inputs are float64 CPU tensors with a final named-joint axis and arbitrary
> leading sample axes, including [batch,time,joint]. Every movable joint,
> including nominal finger joints, appears exactly once. Returned poses use
> position in metres and unit quaternions in wxyz order. Returned linear and
> angular velocities are evaluated at each frame origin and expressed in
> the calibrated reference axes, in metres/second and radians/second.
>
> ``model`` is the existing RobotGeometryChecker.model. No model is loaded,
> rebuilt, locked or reduced. ``frames`` names robot frames in that model.
> ``T_reference_base`` maps model-base coordinates into reference coordinates.

## `PinKinematics.__init__`

[定义位置](../../../manipulation/pin_kinematics.py#L88)。

输入签名：

```python
def __init__(self, model, *, frames, T_reference_base):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
model.names
model.joints
model.getJointId
self.joint_names
model.nq
model.nv
joint.nq
joint.nv
np.asarray
joint.idx_q
joint.idx_v
dict.fromkeys
model.existFrame
self.frames
model.getFrameId
calibration[:3, :3].copy
calibration[:3, 3].copy
```

写入目标：

```python
self.model
self.joint_names
name
joints
joint
self._q_columns
self._v_columns
self.frames
frame
self._frame_ids
calibration
self._rotation
self._translation
```

直接调用符号：

```python
tuple
str
model.getJointId
len
any
ValueError
np.asarray
int
dict.fromkeys
model.existFrame
model.getFrameId
calibration[:3, :3].copy
calibration[:3, 3].copy
```

## `PinKinematics._ordered`

[定义位置](../../../manipulation/pin_kinematics.py#L106)。

输入签名：

```python
def _ordered(self, value, joint_names):
```

显式输出分支：

```python
value[..., [names.index(name) for name in self.joint_names]]
```

读取的对象属性：

```python
self.joint_names
value.device.type
value.device
value.dtype
torch.float64
value.shape
names.index
```

写入目标：

```python
names
name
```

直接调用符号：

```python
tuple
len
set
ValueError
names.index
```

## `PinKinematics.poses`

[定义位置](../../../manipulation/pin_kinematics.py#L114)。

输入签名：

```python
def poses(self, position, joint_names):
```

实现说明：

> Return {frame: (position[...,3], quaternion_wxyz[...,4])}.

显式输出分支：

```python
{frame: (packed[..., index, :3], packed[..., index, 3:]) for (index, frame) in enumerate(self.frames)}
```

读取的对象属性：

```python
_PinPoses.apply
self._ordered
self.frames
```

写入目标：

```python
packed
index
frame
```

直接调用符号：

```python
_PinPoses.apply
self._ordered
enumerate
```

## `PinKinematics.states`

[定义位置](../../../manipulation/pin_kinematics.py#L120)。

输入签名：

```python
def states(self, position, velocity, joint_names):
```

实现说明：

> Return (poses, {frame: (linear_velocity, angular_velocity)}).
>
> Pinocchio supplies LOCAL frame-origin velocity derivatives. Ordinary
> Torch rotation by the returned orientation expresses those vectors
> in reference axes and differentiates the moving coordinate basis.
> This gives Euclidean output derivatives, including the rotation term
> absent from a direct LOCAL_WORLD_ALIGNED spatial derivative.

显式输出分支：

```python
(poses, velocities)
```

读取的对象属性：

```python
position.shape
velocity.shape
_PinStates.apply
self._ordered
self.frames
```

写入目标：

```python
packed
twists
poses
index
frame
velocities
```

直接调用符号：

```python
ValueError
_PinStates.apply
self._ordered
enumerate
quaternion_rotate
```

## `PinKinematics._evaluate`

[定义位置](../../../manipulation/pin_kinematics.py#L140)。

输入签名：

```python
def _evaluate(self, position, velocity=None):
```

实现说明：

> Evaluate library FK and first derivatives at each admitted sample.

显式输出分支：

```python
(*result, twists.reshape(*leading, frames, 6), velocity_position.reshape(*leading, frames, 6, joints), velocity_velocity.reshape(*leading, frames, 6, joints))
result
```

读取的对象属性：

```python
position.shape
np.prod
self.joint_names
self.frames
position.reshape
velocity.reshape
np.empty
np.empty_like
self.model.createData
self.model
self.model.nq
self.model.nv
self._q_columns
pin.computeJointJacobians
self._v_columns
pin.computeForwardKinematicsDerivatives
np.zeros
pin.updateFramePlacements
self._frame_ids
data.oMf
self._rotation
placement.rotation
placement.translation
self._translation
pin.Quaternion(rotation).coeffs
pin.Quaternion
pin.getFrameJacobian
pin.LOCAL_WORLD_ALIGNED
pin.getFrameVelocityDerivatives
pin.LOCAL
pin.getFrameVelocity
motion.linear
motion.angular
poses.reshape
jacobian.reshape
twists.reshape
velocity_position.reshape
velocity_velocity.reshape
```

写入目标：

```python
leading
count
joints
frames
values
rates
poses
jacobian
twists
velocity_position
velocity_velocity
data
q
dq
sample
q[self._q_columns]
dq[self._v_columns]
index
frame
placement
rotation
poses[sample, index, :3]
poses[sample, index, 3:]
geometric
derivative_q
derivative_v
motion
twists[sample, index, :3]
twists[sample, index, 3:]
velocity_position[sample, index]
velocity_velocity[sample, index]
jacobian[sample, index, :3]
jacobian[sample, index, 3:]
result
```

直接调用符号：

```python
int
np.prod
len
position.reshape
velocity.reshape
np.empty
np.empty_like
self.model.createData
range
pin.computeJointJacobians
pin.computeForwardKinematicsDerivatives
np.zeros
pin.updateFramePlacements
enumerate
pin.Quaternion(rotation).coeffs
pin.Quaternion
pin.getFrameJacobian
pin.getFrameVelocityDerivatives
pin.getFrameVelocity
poses.reshape
jacobian.reshape
twists.reshape
velocity_position.reshape
velocity_velocity.reshape
```
