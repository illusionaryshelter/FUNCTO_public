# `manipulation/validation_kinematics.py` 实现契约

代码基线：`8b0e326`。
共享验收的数据所有权及调用次序见[共享验收组件](../10-shared-candidate-validation.md)。

职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `FKSession`

[定义位置](../../../manipulation/validation_kinematics.py#L13)。

```python
class FKSession():
```

类字段：

无显式条目。

实现说明：

> Own FK values while explicit acceptance calls share them.
>
> Public methods do not install callbacks or modify classes. A caller passes
> this owner to PinKinematics and RobotGeometryChecker; differentiable
> calculations continue through the existing analytic derivative API.

## `FKSession.__init__`

[定义位置](../../../manipulation/validation_kinematics.py#L20)。

输入签名：

```python
def __init__(self, model, *, frames):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
dict.fromkeys
model.existFrame
self.frames
model.getFrameId
model.createData
```

写入目标：

```python
self.model
self.frames
frame
self._frame_ids
self._data
self._states
self._counts
```

直接调用符号：

```python
tuple
dict.fromkeys
any
model.existFrame
ValueError
model.getFrameId
model.createData
dict
```

## `FKSession.__enter__`

[定义位置](../../../manipulation/validation_kinematics.py#L32)。

输入签名：

```python
def __enter__(self):
```

显式输出分支：

```python
self
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `FKSession.__exit__`

[定义位置](../../../manipulation/validation_kinematics.py#L35)。

输入签名：

```python
def __exit__(self, *exception):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._states.clear
self._states
self._counts
```

写入目标：

```python
self._data
self._counts['numeric_bytes']
```

直接调用符号：

```python
self._states.clear
```

## `FKSession.placements`

[定义位置](../../../manipulation/validation_kinematics.py#L40)。

输入签名：

```python
def placements(self, model, q):
```

实现说明：

> Return owned model-base SE3 values for the declared frame IDs.

显式输出分支：

```python
state[0]
```

读取的对象属性：

```python
self._state
```

写入目标：

```python
state
_
```

直接调用符号：

```python
self._state
```

## `FKSession.describe`

[定义位置](../../../manipulation/validation_kinematics.py#L45)。

输入签名：

```python
def describe(self):
```

实现说明：

> Return computation counts and owned numeric storage, not timing.

显式输出分支：

```python
{**self._counts, 'scope': 'one candidate check; complete model-order q; original Pinocchio FK', 'storage_scope': 'owned SE3 numeric values, complete input keys, projected poses and LOCAL twist arrays; Python containers and one Data are separate'}
```

读取的对象属性：

```python
self._counts
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `FKSession._charge`

[定义位置](../../../manipulation/validation_kinematics.py#L51)。

输入签名：

```python
def _charge(self, count):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._counts
```

写入目标：

```python
self._counts['numeric_bytes']
self._counts['peak_numeric_bytes']
```

直接调用符号：

```python
max
```

## `FKSession._state`

[定义位置](../../../manipulation/validation_kinematics.py#L56)。

输入签名：

```python
def _state(self, model, q, dq=None, frame_ids=()):
```

实现说明：

> Return owned original placements and requested LOCAL velocities.
>
> The q-only and q/dq Pinocchio overloads compute identical placements
> for the same complete q.  A missing velocity request is computed by
> the maintained q/dq overload, with its own full input and frame key.
> No velocity, Jacobian or model relationship is reconstructed here.

显式输出分支：

```python
(state, None if dq is None else state[1][velocity_key])
(state, None)
(state, twists)
```

读取的对象属性：

```python
self.model
self._data
self._states
q.tobytes
states.get
dq.tobytes
self._counts
pin.forwardKinematics
pin.updateFramePlacements
pin.SE3
data.oMf
self._frame_ids
self._charge
np.empty
pin.getFrameVelocity
pin.LOCAL
motion.linear
motion.angular
twists.flags
twists.nbytes
```

写入目标：

```python
data
states
q_key
state
velocity_key
self._counts['hits']
self._counts['computed']
placements
frame
states[q_key]
self._counts['snapshots']
self._counts['frames']
twists
index
motion
twists[index, :3]
twists[index, 3:]
twists.flags.writeable
state[1][velocity_key]
self._counts['velocity_results']
```

直接调用符号：

```python
ValueError
q.tobytes
states.get
dq.tobytes
tuple
pin.forwardKinematics
pin.updateFramePlacements
pin.SE3
len
self._charge
np.empty
enumerate
pin.getFrameVelocity
```

## `FKSession.evaluate`

[定义位置](../../../manipulation/validation_kinematics.py#L102)。

输入签名：

```python
def evaluate(self, bridge, position, velocity=None):
```

显式输出分支：

```python
(*result, twists.reshape(*leading, frames, 6), None, None)
result
```

读取的对象属性：

```python
position.shape
np.prod
bridge.joint_names
bridge.frames
position.reshape
velocity.reshape
np.empty
bridge.model.nq
bridge.model
bridge.model.nv
bridge._rotation.tobytes
bridge._rotation
bridge._translation.tobytes
bridge._translation
bridge._q_columns
bridge._v_columns
self._state
bridge._frame_ids
self._charge
placement.rotation
placement.translation
pin.Quaternion(rotation).coeffs
pin.Quaternion
pose.flags
self._counts
pose.nbytes
poses.reshape
twists.reshape
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
twists
q
dq
calibration
sample
q[bridge._q_columns]
dq[bridge._v_columns]
state
local
placements
_
projections
projections[calibration]
projected
index
frame
placement
rotation
pose
pose[:3]
pose[3:]
pose.flags.writeable
projected[frame]
self._counts['projected_frames']
poses[sample, index]
twists[sample]
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
bridge._rotation.tobytes
bridge._translation.tobytes
range
self._state
self._charge
sum
map
enumerate
pin.Quaternion(rotation).coeffs
pin.Quaternion
poses.reshape
twists.reshape
```
