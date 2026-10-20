# `manipulation/gripper_trajectory.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../03-attachment-contact-and-hand.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_text`

[定义位置](../../../manipulation/gripper_trajectory.py#L31)。

输入签名：

```python
def _text(value, label):
```

输出分支的源码表达式：

```python
value
```

直接调用：

```python
isinstance
value.strip
ValueError
```

## `_array`

[定义位置](../../../manipulation/gripper_trajectory.py#L37)。

输入签名：

```python
def _array(value, shape, label):
```

直接读取的对象属性：`result.shape`、`result.copy`。

输出分支的源码表达式：

```python
result.copy()
```

直接调用：

```python
np.asarray
np.isfinite(result).all
np.isfinite
ValueError
result.copy
```

## `_activation`

[定义位置](../../../manipulation/gripper_trajectory.py#L44)。

输入签名：

```python
def _activation(time_s, timestep):
```

输出分支的源码表达式：

```python
(step, step * timestep)
```

直接调用：

```python
int
math.ceil
```

## `_extrema`

[定义位置](../../../manipulation/gripper_trajectory.py#L62)。

输入签名：

```python
def _extrema(curve, names, limits):
```

输出分支的源码表达式：

```python
{'scope': _LIMITS_SCOPE, 'joint_extrema': rows, 'maximum_bound_excess': maximum, 'bounds_satisfied': maximum == 0.0}
```

直接写入的状态或数组位置：`joint[label]`、`rows[name]`。

直接调用：

```python
enumerate
PPoly
scalar.derivative(order + 1).roots
scalar.derivative
np.isfinite
np.unique
scalar
np.diff
np.stack((2 * scalar.c[1], 6 * scalar.c[0] * width + 2 * scalar.c[1]), axis=1).reshape
np.stack
np.stack((curve.x[:-1], curve.x[1:]), axis=1).reshape
np.isfinite(values).all
ValueError
int
values.argmin
values.argmax
max
float
maxima.append
```

## `_segment`

[定义位置](../../../manipulation/gripper_trajectory.py#L100)。

输入签名：

```python
def _segment(q0, dq0, goal, start, end, timestep, names, limits):
```

输出分支的源码表达式：

```python
start
finish
```

直接调用：

```python
ValueError
np.array_equal
np.all
np.maximum
np.abs
max
float
np.max
_activation
int
math.floor
range
candidates.append
CubicHermiteSpline
np.stack
np.zeros_like
_extrema
```

## `_commands`

[定义位置](../../../manipulation/gripper_trajectory.py#L150)。

输入签名：

```python
def _commands(values, names, duration, stop, timestep, position_limits):
```

直接读取的对象属性：`result.append`、`result.sort`。

输出分支的源码表达式：

```python
tuple(result)
```

直接写入的状态或数组位置：`row['superseded']`。

直接调用：

```python
isinstance
ValueError
set
value.keys
deepcopy
value.items
_text
event_occurrence_key
identities.add
np.isfinite
_array
len
np.any
row.get
type
_activation
float
row.update
goal.tolist
result.append
result.sort
enumerate
tuple
```

## `FingerTrajectory`

[定义位置](../../../manipulation/gripper_trajectory.py#L219)。

```python
class FingerTrajectory:
    joint_names: tuple[str, ...]
    times: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    duration: float
    executable_until: float
    timestep: float
    position_limits: np.ndarray
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray
    events: tuple[dict, ...]
    segments: tuple[dict, ...]
    source: str
    feedback_id: str
    timestamp_s: float
    _curve: CubicHermiteSpline = field(init=False, repr=False)
    _bounds_report: dict = field(init=False, repr=False)
    _torch_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
```

该类型的直接方法：`__post_init__`、`_limits`、`build`、`_validate_segments`、`sample`、`sample_torch`、`torch_storage_bytes`、`describe`、`from_mapping`、`validate_execution`。

## `FingerTrajectory.__post_init__`

[定义位置](../../../manipulation/gripper_trajectory.py#L243)。

输入签名：

```python
def FingerTrajectory.__post_init__(self):
```

直接读取的对象属性：`self.joint_names`、`self.source`、`self.feedback_id`、`self.duration`、`self.executable_until`、`self.timestep`、`self.timestamp_s`、`self.times`、`self.position`、`self.velocity`、`self.events`、`self.position_limits`、`self.segments`、`self._limits`、`self._validate_segments`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
isinstance
ValueError
tuple
len
set
any
name.strip
object.__setattr__
_text
np.isfinite([self.duration, self.executable_until, self.timestep, self.timestamp_s]).all
np.isfinite
math.ceil
_array
getattr
np.any
np.asarray
np.isfinite(times).all
np.diff
times.copy
_commands
deepcopy
CubicHermiteSpline
_extrema
self._limits
self._validate_segments
getattr(self, label).setflags
```

## `FingerTrajectory._limits`

[定义位置](../../../manipulation/gripper_trajectory.py#L284)。

输入签名：

```python
def FingerTrajectory._limits(self):
```

输出分支的源码表达式：

```python
{label: getattr(self, label + '_limits') for label in ('position', 'velocity', 'acceleration')}
```

直接调用：

```python
getattr
```

## `FingerTrajectory.build`

[定义位置](../../../manipulation/gripper_trajectory.py#L288)。

输入签名：

```python
@classmethod
def FingerTrajectory.build(cls, *, joint_names, initial_position, initial_velocity, commands, duration, executable_until, timestep, position_limits, velocity_limits, acceleration_limits, source, feedback_id, timestamp_s):
```

输出分支的源码表达式：

```python
cls(names, np.asarray(knots), np.asarray(positions), np.asarray(velocities), float(duration), float(executable_until), float(timestep), bounds['position'], bounds['velocity'], bounds['acceleration'], events, tuple(segments), source, feedback_id, float(timestamp_s))
```

直接调用：

```python
tuple
len
_array
np.isfinite([duration, executable_until, timestep]).all
np.isfinite
ValueError
np.any
_commands
windows.append
q.copy
enumerate
np.asarray
dq.copy
_segment
segments.append
float
goal.tolist
append
np.zeros
goal.copy
cls
```

## `FingerTrajectory.build.append`

[定义位置](../../../manipulation/gripper_trajectory.py#L325)。

输入签名：

```python
def FingerTrajectory.build.append(time_s, value, rate):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
knots.append
float
positions.append
value.copy
velocities.append
rate.copy
np.array_equal
ValueError
```

## `FingerTrajectory._validate_segments`

[定义位置](../../../manipulation/gripper_trajectory.py#L356)。

输入签名：

```python
def FingerTrajectory._validate_segments(self):
```

直接读取的对象属性：`self.position`、`self.velocity`、`self.events`、`self.segments`、`self.duration`、`self.joint_names`、`self.times`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
event_occurrence_key
set
isinstance
ValueError
float
np.isfinite([begin, finish, stop]).all
np.isfinite
_array
len
np.array_equal
used.add
expected_times.append
expected_q.append
goal.copy
expected_dq.append
np.zeros
np.any
```

## `FingerTrajectory.sample`

[定义位置](../../../manipulation/gripper_trajectory.py#L406)。

输入签名：

```python
def FingerTrajectory.sample(self, times):
```

直接读取的对象属性：`self.duration`、`self._curve`。

输出分支的源码表达式：

```python
(self._curve(query), self._curve(query, 1))
```

直接调用：

```python
np.asarray
np.isfinite(query).all
np.isfinite
np.any
ValueError
self._curve
```

## `FingerTrajectory.sample_torch`

[定义位置](../../../manipulation/gripper_trajectory.py#L413)。

输入签名：

```python
def FingerTrajectory.sample_torch(self, times):
```

直接读取的对象属性：`self.duration`、`self._torch_cache`、`self.times`、`self._curve`。

输出分支的源码表达式：

```python
(q, dq)
```

直接写入的状态或数组位置：`self._torch_cache[key]`。

直接调用：

```python
isinstance
times.is_floating_point
ValueError
require_tensor
torch.isfinite(times).all
torch.isfinite
(times < 0.0).any
(times > self.duration).any
torch.where
torch.zeros_like
torch.as_tensor
self.times.copy
self._curve.c.copy
torch.searchsorted(knots, times.contiguous(), right=True).clamp
torch.searchsorted
times.contiguous
len
```

## `FingerTrajectory.torch_storage_bytes`

[定义位置](../../../manipulation/gripper_trajectory.py#L447)。

输入签名：

```python
@property
def FingerTrajectory.torch_storage_bytes(self):
```

直接读取的对象属性：`self.times`、`self._curve`。

输出分支的源码表达式：

```python
(self.times.size + self._curve.c.size) * (4 + 8)
```

## `FingerTrajectory.describe`

[定义位置](../../../manipulation/gripper_trajectory.py#L452)。

输入签名：

```python
def FingerTrajectory.describe(self):
```

直接读取的对象属性：`self.joint_names`、`self.source`、`self.feedback_id`、`self.timestamp_s`、`self.duration`、`self.executable_until`、`self.timestep`、`self.times`、`self.position`、`self.velocity`、`self.position_limits`、`self.velocity_limits`、`self.acceleration_limits`、`self.events`、`self.segments`、`self._bounds_report`。

输出分支的源码表达式：

```python
{'joint_names': list(self.joint_names), 'source': self.source, 'feedback_id': self.feedback_id, 'timestamp_s': self.timestamp_s, 'duration': self.duration, 'executable_until': self.executable_until, 'timestep': self.timestep, 'times': self.times.tolist(), 'position': self.position.tolist(), 'velocity': self.velocity.tolist(), 'position_limits': self.position_limits.tolist(), 'velocity_limits': self.velocity_limits.tolist(), 'acceleration_limits': self.acceleration_limits.tolist(), 'events': deepcopy(list(self.events)), 'segments': deepcopy(list(self.segments)), 'scope': _SCOPE, 'limits_scope': _LIMITS_SCOPE, 'bounds_report': deepcopy(self._bounds_report)}
```

直接调用：

```python
list
self.times.tolist
self.position.tolist
self.velocity.tolist
self.position_limits.tolist
self.velocity_limits.tolist
self.acceleration_limits.tolist
deepcopy
```

## `FingerTrajectory.from_mapping`

[定义位置](../../../manipulation/gripper_trajectory.py#L463)。

输入签名：

```python
@classmethod
def FingerTrajectory.from_mapping(cls, value):
```

直接读取的对象属性：`result.describe`。

输出分支的源码表达式：

```python
result
```

直接调用：

```python
cls.__dataclass_fields__.items
isinstance
set
ValueError
cls
deepcopy
result.describe
```

## `FingerTrajectory.validate_execution`

[定义位置](../../../manipulation/gripper_trajectory.py#L473)。

输入签名：

```python
def FingerTrajectory.validate_execution(self, trajectory, commands, *, initial_feedback, timestep, position_limits, velocity_limits):
```

直接读取的对象属性：`trajectory.duration`、`self.duration`、`trajectory.executable_until`、`self.executable_until`、`self.timestep`、`self.joint_names`、`self.feedback_id`、`self.timestamp_s`、`self.position`、`self.velocity`、`self.position_limits`、`self.velocity_limits`、`self.events`、`self._bounds_report`。

输出分支的源码表达式：

```python
deepcopy(self._bounds_report)
```

直接调用：

```python
tuple
ValueError
tuple(initial_feedback['joint_names']).index
np.array_equal
np.asarray
_array
np.any
_commands
deepcopy
```

## `NominalFingerGeometry`

[定义位置](../../../manipulation/gripper_trajectory.py#L505)。

```python
class NominalFingerGeometry:
    joint_names: tuple[str, ...]
    times: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    duration: float
    valid_until_s: float
    source: str
    _curve: CubicHermiteSpline = field(init=False, repr=False)
    _torch_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)
```

该类型的直接方法：`__post_init__`、`valid_mask`、`valid_mask_torch`、`sample`、`sample_torch`、`torch_storage_bytes`、`describe`、`from_reference`、`from_mapping`。

## `NominalFingerGeometry.__post_init__`

[定义位置](../../../manipulation/gripper_trajectory.py#L525)。

输入签名：

```python
def NominalFingerGeometry.__post_init__(self):
```

直接读取的对象属性：`self.joint_names`、`self.source`、`self.duration`、`self.valid_until_s`、`self.times`、`self.position`、`self.velocity`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
tuple
len
set
any
isinstance
name.strip
ValueError
_text
np.isfinite([self.duration, self.valid_until_s]).all
np.isfinite
np.asarray
np.isfinite(times).all
np.any
np.diff
object.__setattr__
times.copy
_array
value.setflags
CubicHermiteSpline
```

## `NominalFingerGeometry.valid_mask`

[定义位置](../../../manipulation/gripper_trajectory.py#L547)。

输入签名：

```python
def NominalFingerGeometry.valid_mask(self, times):
```

直接读取的对象属性：`self.duration`、`self.valid_until_s`。

输出分支的源码表达式：

```python
np.isfinite(query) & (query >= 0.0) & (query <= endpoint)
```

直接调用：

```python
np.asarray
np.isfinite(query).all
np.isfinite
np.any
ValueError
```

## `NominalFingerGeometry.valid_mask_torch`

[定义位置](../../../manipulation/gripper_trajectory.py#L561)。

输入签名：

```python
def NominalFingerGeometry.valid_mask_torch(self, times):
```

直接读取的对象属性：`self.duration`、`self.valid_until_s`。

输出分支的源码表达式：

```python
admitted & (times <= self.valid_until_s)
```

直接调用：

```python
isinstance
times.is_floating_point
ValueError
torch.isfinite
require_tensor
admitted.all
```

## `NominalFingerGeometry.sample`

[定义位置](../../../manipulation/gripper_trajectory.py#L571)。

输入签名：

```python
def NominalFingerGeometry.sample(self, times):
```

输出分支的源码表达式：

```python
FingerTrajectory.sample(self, times)
```

直接调用：

```python
FingerTrajectory.sample
```

## `NominalFingerGeometry.sample_torch`

[定义位置](../../../manipulation/gripper_trajectory.py#L574)。

输入签名：

```python
def NominalFingerGeometry.sample_torch(self, times):
```

输出分支的源码表达式：

```python
FingerTrajectory.sample_torch(self, times)
```

直接调用：

```python
FingerTrajectory.sample_torch
```

## `NominalFingerGeometry.torch_storage_bytes`

[定义位置](../../../manipulation/gripper_trajectory.py#L579)。

输入签名：

```python
@property
def NominalFingerGeometry.torch_storage_bytes(self):
```

输出分支的源码表达式：

```python
FingerTrajectory.torch_storage_bytes.fget(self)
```

直接调用：

```python
FingerTrajectory.torch_storage_bytes.fget
```

## `NominalFingerGeometry.describe`

[定义位置](../../../manipulation/gripper_trajectory.py#L582)。

输入签名：

```python
def NominalFingerGeometry.describe(self):
```

直接读取的对象属性：`self.joint_names`、`self.times`、`self.position`、`self.velocity`、`self.duration`、`self.valid_until_s`、`self.source`。

输出分支的源码表达式：

```python
{'joint_names': list(self.joint_names), 'times': self.times.tolist(), 'position': self.position.tolist(), 'velocity': self.velocity.tolist(), 'duration': self.duration, 'valid_until_s': self.valid_until_s, 'source': self.source, 'scope': _GEOMETRY_SCOPE}
```

直接调用：

```python
list
self.times.tolist
self.position.tolist
self.velocity.tolist
```

## `NominalFingerGeometry.from_reference`

[定义位置](../../../manipulation/gripper_trajectory.py#L589)。

输入签名：

```python
@classmethod
def NominalFingerGeometry.from_reference(cls, reference, *, valid_until_s, source):
```

输出分支的源码表达式：

```python
cls(reference.joint_names, reference.times, reference.position, reference.velocity, reference.duration, valid_until_s, source)
```

直接调用：

```python
cls
```

## `NominalFingerGeometry.from_mapping`

[定义位置](../../../manipulation/gripper_trajectory.py#L595)。

输入签名：

```python
@classmethod
def NominalFingerGeometry.from_mapping(cls, value):
```

直接读取的对象属性：`result.describe`。

输出分支的源码表达式：

```python
result
```

直接调用：

```python
cls.__dataclass_fields__.items
isinstance
set
ValueError
cls
deepcopy
result.describe
```

## `bind_nominal_finger_geometry`

[定义位置](../../../manipulation/gripper_trajectory.py#L605)。

输入签名：

```python
def bind_nominal_finger_geometry(reference, conditions, event_clock, *, progress_interval):
```

输出分支的源码表达式：

```python
NominalFingerGeometry(reference.joint_names, times, positions, velocities, reference.duration, cutoff, source)
```

直接调用：

```python
np.asarray
np.isfinite([begin, end]).all
np.isfinite
ValueError
row.get
boundaries.extend
event_occurrence_key
event_clock.items
boundaries.append
np.array
np.tile
np.zeros_like
repr
min
NominalFingerGeometry
```
