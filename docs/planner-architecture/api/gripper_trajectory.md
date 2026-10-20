# `manipulation/gripper_trajectory.py` 实现契约

代码基线：`cf7182cbf3888d9db2e7553df47dce9a0b1c339f`。
职责和数据语义见[组件说明](../03-attachment-contact-and-hand.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_text`

[定义位置](../../../manipulation/gripper_trajectory.py#L31)。

输入签名：

```python
def _text(value, label):
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

## `_array`

[定义位置](../../../manipulation/gripper_trajectory.py#L37)。

输入签名：

```python
def _array(value, shape, label):
```

显式输出分支：

```python
result.copy()
```

读取的对象属性：

```python
np.asarray
result.shape
np.isfinite(result).all
np.isfinite
result.copy
```

写入目标：

```python
result
```

直接调用符号：

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

实现说明：

> Select the first native control tick at or after the original event.
> 
> The product is checked as well as the quotient: binary64 division must
> not round a slightly future event down to an earlier physical command.
> No physical or numerical early-application allowance is introduced.

显式输出分支：

```python
(step, step * timestep)
```

读取的对象属性：

```python
math.ceil
```

写入目标：

```python
step
```

直接调用符号：

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

实现说明：

> Inspect all scalar polynomial extrema and both acceleration knot sides.
> 
> SciPy supplies polynomial roots. Values at stored q/dq endpoints use the
> spline's endpoint constraints; ddq is affine on each cubic and needs both
> one-sided endpoints. Reported excess is never clipped by an allowance.

显式输出分支：

```python
{'scope': _LIMITS_SCOPE, 'joint_extrema': rows, 'maximum_bound_excess': maximum, 'bounds_satisfied': maximum == 0.0}
```

读取的对象属性：

```python
curve.c
curve.x
scalar.derivative(order + 1).roots
scalar.derivative
np.isfinite
np.unique
np.r_
np.diff
np.stack((2 * scalar.c[1], 6 * scalar.c[0] * width + 2 * scalar.c[1]), axis=1).reshape
np.stack
scalar.c
np.stack((curve.x[:-1], curve.x[1:]), axis=1).reshape
np.isfinite(values).all
values.argmin
values.argmax
maxima.append
```

写入目标：

```python
maxima
rows
column
name
scalar
joint
order
label
roots
times
values
width
low
high
lower
upper
excess
joint[label]
rows[name]
maximum
```

直接调用符号：

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

实现说明：

> Find the first bounded Hermite ramp ending on the native control grid.
> 
> This is a finite search over realizable control ticks, not a new dynamic
> model or a time-optimality claim. Every candidate uses SciPy's same cubic
> and exact scalar limits. The original deadline is also considered when
> it falls between ticks; failure never extends the source event/barrier.

显式输出分支：

```python
start
finish
```

读取的对象属性：

```python
np.array_equal
np.all
np.maximum
np.abs
np.max
math.floor
candidates.append
np.stack
np.zeros_like
```

写入目标：

```python
speeds
accelerations
required
first
_
available
last
last_time
candidates
step
finish
ramp
```

直接调用符号：

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

显式输出分支：

```python
tuple(result)
```

读取的对象属性：

```python
value.keys
value.items
identities.add
np.isfinite
np.any
row.get
row.update
goal.tolist
result.append
result.sort
```

写入目标：

```python
result
identities
required
value
row
key
item
label
identity
goal
step
request_step
origin_step
intended_step
requested_activation
applied
in_prefix
index
next_row
same_tick
row['superseded']
```

直接调用符号：

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
class FingerTrajectory():
```

类字段：

```python
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

实现说明：

> An owned, serializable reference; actual q/dq always come from feedback.

## `FingerTrajectory.__post_init__`

[定义位置](../../../manipulation/gripper_trajectory.py#L243)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.joint_names
name.strip
object.__setattr__
self.source
self.feedback_id
np.isfinite([self.duration, self.executable_until, self.timestep, self.timestamp_s]).all
np.isfinite
self.duration
self.executable_until
self.timestep
self.timestamp_s
math.ceil
np.any
np.asarray
self.times
times.ndim
np.isfinite(times).all
np.diff
self.position
self.velocity
position.shape
times.copy
self.events
self.position_limits
self.segments
self._limits
self._validate_segments
getattr(self, label).setflags
```

写入目标：

```python
names
name
label
bound
times
position
velocity
events
curve
report
```

直接调用符号：

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
def _limits(self):
```

显式输出分支：

```python
{label: getattr(self, label + '_limits') for label in ('position', 'velocity', 'acceleration')}
```

读取的对象属性：

无显式条目。

写入目标：

```python
label
```

直接调用符号：

```python
getattr
```

## `FingerTrajectory.build`

[定义位置](../../../manipulation/gripper_trajectory.py#L288)。

输入签名：

```python
def build(cls, *, joint_names, initial_position, initial_velocity, commands, duration, executable_until, timestep, position_limits, velocity_limits, acceleration_limits, source, feedback_id, timestamp_s):
```

实现说明：

> Compile explicit event goals without changing their original clock.
> 
> Before a later first event, a bounded C1 takeover returns to the
> measured initial coordinate and zero reference velocity. This is an
> explicit position-hold reference, not fictitious stationary feedback.
> The initial measured velocity is always preserved as the first slope.

显式输出分支：

```python
cls(names, np.asarray(knots), np.asarray(positions), np.asarray(velocities), float(duration), float(executable_until), float(timestep), bounds['position'], bounds['velocity'], bounds['acceleration'], events, tuple(segments), source, feedback_id, float(timestamp_s))
```

读取的对象属性：

```python
np.isfinite([duration, executable_until, timestep]).all
np.isfinite
np.any
windows.append
q.copy
np.asarray
dq.copy
segments.append
goal.tolist
np.zeros
goal.copy
```

写入目标：

```python
names
count
q
dq
bounds
label
value
events
active
row
windows
index
event
end
knots
positions
velocities
segments
begin
goal
finish
```

直接调用符号：

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
def append(time_s, value, rate):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
knots.append
positions.append
value.copy
velocities.append
rate.copy
np.array_equal
```

写入目标：

无显式条目。

直接调用符号：

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
def _validate_segments(self):
```

实现说明：

> Validate saved phase boundaries without regenerating or retiming.
> 
> Reconstruct only the explicitly saved knots from event targets and
> ramp endpoints. A changed archive cannot hide motion before an event,
> replace the initial slope, or omit conditional commands in the suffix.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.position
self.velocity
self.events
self.segments
np.isfinite([begin, finish, stop]).all
np.isfinite
self.duration
self.joint_names
np.array_equal
used.add
expected_times.append
expected_q.append
goal.copy
expected_dq.append
np.zeros
np.any
self.times
```

写入目标：

```python
expected_times
expected_q
expected_dq
active
row
used
end
segment
begin
finish
stop
key
goal
identity
event
moment
```

直接调用符号：

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
def sample(self, times):
```

实现说明：

> Return commanded q/dq at the supplied shared physical times.

显式输出分支：

```python
(self._curve(query), self._curve(query, 1))
```

读取的对象属性：

```python
np.asarray
np.isfinite(query).all
np.isfinite
np.any
self.duration
self._curve
```

写入目标：

```python
query
```

直接调用符号：

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
def sample_torch(self, times):
```

实现说明：

> Evaluate the same saved polynomial without native FK or new timing.

显式输出分支：

```python
(q, dq)
```

读取的对象属性：

```python
torch.Tensor
times.is_floating_point
torch.isfinite(times).all
torch.isfinite
(times < 0.0).any
(times > self.duration).any
self.duration
torch.where
torch.zeros_like
times.device
times.dtype
self._torch_cache
torch.as_tensor
self.times.copy
self.times
self._curve.c.copy
self._curve.c
self._curve
torch.searchsorted(knots, times.contiguous(), right=True).clamp
torch.searchsorted
times.contiguous
```

写入目标：

```python
message
valid
times
key
self._torch_cache[key]
knots
coefficients
indices
offset
selected
q
dq
```

直接调用符号：

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
def torch_storage_bytes(self):
```

实现说明：

> Reserve both planning dtypes before lazy polynomial tensor creation.

显式输出分支：

```python
(self.times.size + self._curve.c.size) * (4 + 8)
```

读取的对象属性：

```python
self.times.size
self.times
self._curve.c.size
self._curve.c
self._curve
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `FingerTrajectory.describe`

[定义位置](../../../manipulation/gripper_trajectory.py#L452)。

输入签名：

```python
def describe(self):
```

实现说明：

> Serialize owned values; a loader must not generate another curve.

显式输出分支：

```python
{'joint_names': list(self.joint_names), 'source': self.source, 'feedback_id': self.feedback_id, 'timestamp_s': self.timestamp_s, 'duration': self.duration, 'executable_until': self.executable_until, 'timestep': self.timestep, 'times': self.times.tolist(), 'position': self.position.tolist(), 'velocity': self.velocity.tolist(), 'position_limits': self.position_limits.tolist(), 'velocity_limits': self.velocity_limits.tolist(), 'acceleration_limits': self.acceleration_limits.tolist(), 'events': deepcopy(list(self.events)), 'segments': deepcopy(list(self.segments)), 'scope': _SCOPE, 'limits_scope': _LIMITS_SCOPE, 'bounds_report': deepcopy(self._bounds_report)}
```

读取的对象属性：

```python
self.joint_names
self.source
self.feedback_id
self.timestamp_s
self.duration
self.executable_until
self.timestep
self.times.tolist
self.times
self.position.tolist
self.position
self.velocity.tolist
self.velocity
self.position_limits.tolist
self.position_limits
self.velocity_limits.tolist
self.velocity_limits
self.acceleration_limits.tolist
self.acceleration_limits
self.events
self.segments
self._bounds_report
```

写入目标：

无显式条目。

直接调用符号：

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
def from_mapping(cls, value):
```

实现说明：

> Load the archived reference exactly, validating all saved claims.

显式输出分支：

```python
result
```

读取的对象属性：

```python
cls.__dataclass_fields__.items
cls.__dataclass_fields__
definition.init
result.describe
```

写入目标：

```python
fields
name
definition
result
key
```

直接调用符号：

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
def validate_execution(self, trajectory, commands, *, initial_feedback, timestep, position_limits, velocity_limits):
```

实现说明：

> Bind a saved reference to current actual feedback and the arm clock.

显式输出分支：

```python
deepcopy(self._bounds_report)
```

读取的对象属性：

```python
trajectory.duration
self.duration
trajectory.executable_until
self.executable_until
self.timestep
self.joint_names
self.feedback_id
self.timestamp_s
tuple(initial_feedback['joint_names']).index
np.array_equal
self.position
np.asarray
self.velocity
self.position_limits.shape
self.position_limits
self.velocity_limits.shape
self.velocity_limits
np.any
self.events
self._bounds_report
```

写入目标：

```python
columns
name
native_position
native_velocity
expected
```

直接调用符号：

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
class NominalFingerGeometry():
```

类字段：

```python
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

实现说明：

> An owned finger polynomial with an explicit physical validity prefix.
> 
> A free-motion model reuses the original command polynomial. A held model
> freezes measured entrance q; its zero derivative describes that geometric
> assumption, not the measured entrance dq or the executable command curve.
> Sampling evaluates the polynomial on its full stored clock. Collision
> consumers must use ``valid_mask`` before claiming geometry at a query.

## `NominalFingerGeometry.__post_init__`

[定义位置](../../../manipulation/gripper_trajectory.py#L525)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.joint_names
name.strip
self.source
np.isfinite([self.duration, self.valid_until_s]).all
np.isfinite
self.duration
self.valid_until_s
np.asarray
self.times
times.ndim
np.isfinite(times).all
np.any
np.diff
object.__setattr__
times.copy
self.position
self.velocity
value.setflags
```

写入目标：

```python
names
name
times
value
```

直接调用符号：

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
def valid_mask(self, times):
```

实现说明：

> Use the same exact typed boundary as the planner's Torch query.
> 
> A float32 endpoint is its declared representation of the binary64
> source time, even when rounding lies above it. Its next representable
> value is outside. No epsilon or progress-based entrance is admitted.

显式输出分支：

```python
np.isfinite(query) & (query >= 0.0) & (query <= endpoint)
```

读取的对象属性：

```python
np.asarray
self.duration
query.dtype
np.isfinite(query).all
np.isfinite
np.any
self.valid_until_s
```

写入目标：

```python
query
terminal
endpoint
```

直接调用符号：

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
def valid_mask_torch(self, times):
```

显式输出分支：

```python
admitted & (times <= self.valid_until_s)
```

读取的对象属性：

```python
torch.Tensor
times.is_floating_point
torch.isfinite
self.duration
admitted.all
self.valid_until_s
```

写入目标：

```python
message
admitted
```

直接调用符号：

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
def sample(self, times):
```

显式输出分支：

```python
FingerTrajectory.sample(self, times)
```

读取的对象属性：

```python
FingerTrajectory.sample
```

写入目标：

无显式条目。

直接调用符号：

```python
FingerTrajectory.sample
```

## `NominalFingerGeometry.sample_torch`

[定义位置](../../../manipulation/gripper_trajectory.py#L574)。

输入签名：

```python
def sample_torch(self, times):
```

显式输出分支：

```python
FingerTrajectory.sample_torch(self, times)
```

读取的对象属性：

```python
FingerTrajectory.sample_torch
```

写入目标：

无显式条目。

直接调用符号：

```python
FingerTrajectory.sample_torch
```

## `NominalFingerGeometry.torch_storage_bytes`

[定义位置](../../../manipulation/gripper_trajectory.py#L579)。

输入签名：

```python
def torch_storage_bytes(self):
```

显式输出分支：

```python
FingerTrajectory.torch_storage_bytes.fget(self)
```

读取的对象属性：

```python
FingerTrajectory.torch_storage_bytes.fget
FingerTrajectory.torch_storage_bytes
```

写入目标：

无显式条目。

直接调用符号：

```python
FingerTrajectory.torch_storage_bytes.fget
```

## `NominalFingerGeometry.describe`

[定义位置](../../../manipulation/gripper_trajectory.py#L582)。

输入签名：

```python
def describe(self):
```

显式输出分支：

```python
{'joint_names': list(self.joint_names), 'times': self.times.tolist(), 'position': self.position.tolist(), 'velocity': self.velocity.tolist(), 'duration': self.duration, 'valid_until_s': self.valid_until_s, 'source': self.source, 'scope': _GEOMETRY_SCOPE}
```

读取的对象属性：

```python
self.joint_names
self.times.tolist
self.times
self.position.tolist
self.position
self.velocity.tolist
self.velocity
self.duration
self.valid_until_s
self.source
```

写入目标：

无显式条目。

直接调用符号：

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
def from_reference(cls, reference, *, valid_until_s, source):
```

实现说明：

> Own the same polynomial under a separately declared geometry scope.

显式输出分支：

```python
cls(reference.joint_names, reference.times, reference.position, reference.velocity, reference.duration, valid_until_s, source)
```

读取的对象属性：

```python
reference.joint_names
reference.times
reference.position
reference.velocity
reference.duration
```

写入目标：

无显式条目。

直接调用符号：

```python
cls
```

## `NominalFingerGeometry.from_mapping`

[定义位置](../../../manipulation/gripper_trajectory.py#L595)。

输入签名：

```python
def from_mapping(cls, value):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
cls.__dataclass_fields__.items
cls.__dataclass_fields__
definition.init
result.describe
```

写入目标：

```python
fields
name
definition
result
key
```

直接调用符号：

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

实现说明：

> Declare geometry from contact identities and the nominal held model.
> 
> ``reference`` remains the original C1 command, including measured dq.
> ``event_clock`` is the already-bound original occurrence clock; no inverse
> progress map, replacement event, stop or execution barrier is introduced.
> Missing consumed contact identities cannot establish a free-motion suffix.

显式输出分支：

```python
NominalFingerGeometry(reference.joint_names, times, positions, velocities, reference.duration, cutoff, source)
```

读取的对象属性：

```python
np.asarray
np.isfinite([begin, end]).all
np.isfinite
row.get
boundaries.extend
reference.events
event_clock.items
boundaries.append
np.array
reference.duration
np.tile
reference.position
np.zeros_like
reference.feedback_id
reference.times
reference.velocity
reference.joint_names
```

写入目标：

```python
begin
end
holding
row
boundaries
matched
identity
clock
times
positions
velocities
origin
cutoff
time
_
source
```

直接调用符号：

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
