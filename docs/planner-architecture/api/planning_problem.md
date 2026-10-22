# `manipulation/planning_problem.py` 实现契约

代码基线：`c84b34a`；本文同时描述本次工作区内已实现、待统一提交的机器人查询结构复用与物理时间参考改动。实现提交后统一更新代码基线引用。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_tensor`

[定义位置](../../../manipulation/planning_problem.py#L24)。

输入签名：

```python
def _tensor(value, like):
```

显式输出分支：

```python
constant_tensor(value, like)
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
constant_tensor
```

## `quaternion_conjugate`

[定义位置](../../../manipulation/planning_problem.py#L31)。

输入签名：

```python
def quaternion_conjugate(q):
```

实现说明：

> Conjugate unit wxyz quaternions without detaching their gradients.

显式输出分支：

```python
q * _tensor([1.0, -1.0, -1.0, -1.0], q)
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
_tensor
```

## `quaternion_multiply`

[定义位置](../../../manipulation/planning_problem.py#L36)。

输入签名：

```python
def quaternion_multiply(a, b):
```

实现说明：

> Hamilton product with ordinary Torch broadcasting.

显式输出分支：

```python
torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True), aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)
```

读取的对象属性：

```python
torch.broadcast_tensors
torch.cat
(av * bv).sum
torch.linalg.cross
torch.linalg
```

写入目标：

```python
aw
av
bw
bv
```

直接调用符号：

```python
torch.broadcast_tensors
torch.cat
(av * bv).sum
torch.linalg.cross
```

## `quaternion_rotate`

[定义位置](../../../manipulation/planning_problem.py#L45)。

输入签名：

```python
def quaternion_rotate(q, points):
```

实现说明：

> Rotate vectors; a further point-set axis may be present before XYZ.

显式输出分支：

```python
points + 2 * (q[..., :1] * cross + torch.linalg.cross(v, cross))
```

读取的对象属性：

```python
points.ndim
q.ndim
q.unsqueeze
torch.broadcast_tensors
torch.linalg.cross
torch.linalg
```

写入目标：

```python
q
v
points
cross
```

直接调用符号：

```python
q.unsqueeze
torch.broadcast_tensors
torch.linalg.cross
```

## `quaternion_rotvec`

[定义位置](../../../manipulation/planning_problem.py#L55)。

输入签名：

```python
def quaternion_rotvec(q):
```

实现说明：

> Principal rotation vector with a finite derivative at the identity.

显式输出分支：

```python
q[..., 1:] * (2 * torch.atan2(length, q[..., :1]) / length)
```

读取的对象属性：

```python
torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min
torch.linalg.vector_norm
torch.linalg
torch.where
torch.sqrt
(q[..., 1:] ** 2).sum
torch.atan2
```

写入目标：

```python
q
length
```

直接调用符号：

```python
torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min
torch.linalg.vector_norm
torch.where
torch.sqrt
(q[..., 1:] ** 2).sum
torch.atan2
```

## `quaternion_axis_angle`

[定义位置](../../../manipulation/planning_problem.py#L65)。

输入签名：

```python
def quaternion_axis_angle(relative, axis):
```

实现说明：

> Project a relative quaternion onto rotations about an explicit unit axis.
>
> Both arguments use the same coordinate frame. The result lies in [-pi,pi]
> and is unchanged by the quaternion sign. A perpendicular half-turn has no
> unique twist; choosing zero there leaves its remaining rotation equal to
> pi. Callers therefore retain the actual tilt instead of accepting a zero
> quaternion or differentiating atan2 at (0,0).

显式输出分支：

```python
torch.atan2(torch.where(singular, torch.zeros_like(sine), sine), torch.where(singular, torch.ones_like(cosine), cosine))
```

读取的对象属性：

```python
(relative[..., 1:] * axis).sum
w.square
along.square
torch.finfo(relative.dtype).eps
torch.finfo
relative.dtype
torch.atan2
torch.where
torch.zeros_like
torch.ones_like
```

写入目标：

```python
w
along
sine
cosine
singular
```

直接调用符号：

```python
(relative[..., 1:] * axis).sum
w.square
along.square
torch.finfo
torch.atan2
torch.where
torch.zeros_like
torch.ones_like
```

## `_quaternion`

[定义位置](../../../manipulation/planning_problem.py#L83)。

输入签名：

```python
def _quaternion(pose):
```

显式输出分支：

```python
Rotation.from_matrix(np.asarray(pose)[..., :3, :3]).as_quat()[..., [3, 0, 1, 2]]
```

读取的对象属性：

```python
Rotation.from_matrix(np.asarray(pose)[..., :3, :3]).as_quat
Rotation.from_matrix
np.asarray
```

写入目标：

无显式条目。

直接调用符号：

```python
Rotation.from_matrix(np.asarray(pose)[..., :3, :3]).as_quat
Rotation.from_matrix
np.asarray
```

## `_slerp`

[定义位置](../../../manipulation/planning_problem.py#L88)。

输入签名：

```python
def _slerp(a, b, fraction):
```

实现说明：

> Shortest-arc interpolation with finite identity first derivatives.
>
> A normalized linear blend is only an approximation to Slerp and changes
> angular progress when another sample splits a short segment. The chord
> half-angle identity avoids acos near one, and sinc weights evaluate the
> same spherical formula at zero without a small-angle alternate curve.

显式输出分支：

```python
result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1e-15)
```

读取的对象属性：

```python
torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp_min
torch.linalg.vector_norm
torch.linalg
torch.linalg.vector_norm(b, dim=-1, keepdim=True).clamp_min
(a * b).sum
torch.where
torch.atan2
torch.pi
torch.sinc
torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min
```

写入目标：

```python
a
b
dot
angle
fraction
normalized_angle
denominator
first
second
result
```

直接调用符号：

```python
torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp_min
torch.linalg.vector_norm
torch.linalg.vector_norm(b, dim=-1, keepdim=True).clamp_min
(a * b).sum
torch.where
torch.atan2
torch.sinc
torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min
```

## `_interpolate`

[定义位置](../../../manipulation/planning_problem.py#L113)。

输入签名：

```python
def _interpolate(knots, values, progress, *, side='right'):
```

实现说明：

> Piecewise linear scalar/vector interpolation, preserving progress grad.

显式输出分支：

```python
data[0].expand(*progress.shape, *data.shape[1:])
data[index - 1] + fraction[(...,) + extra] * (data[index] - data[index - 1])
```

读取的对象属性：

```python
data[0].expand
progress.shape
data.shape
torch.searchsorted(keys, progress.contiguous(), right=side == 'right').clamp
torch.searchsorted
progress.contiguous
((progress - keys[index - 1]) / (keys[index] - keys[index - 1])).clamp
data.ndim
```

写入目标：

```python
keys
data
index
fraction
extra
```

直接调用符号：

```python
_tensor
len
data[0].expand
torch.searchsorted(keys, progress.contiguous(), right=side == 'right').clamp
torch.searchsorted
progress.contiguous
((progress - keys[index - 1]) / (keys[index] - keys[index - 1])).clamp
```

## `PosePath`

[定义位置](../../../manipulation/planning_problem.py#L126)。

```python
class PosePath():
```

类字段：

```python
progress: np.ndarray
poses: np.ndarray
source: object
motion_model: MotionModel | None = None
coordinates: np.ndarray | None = None
attachment: np.ndarray = field(default_factory=lambda : np.eye(4))
parent: PosePath | None = None
```

实现说明：

> A current-scene reference, optionally parameterized by mechanism FK.
>
> A mechanism path interpolates its signed/unwrapped coordinates and then
> evaluates FK. It never interpolates a hinge as Cartesian chord segments.
> ``attachment`` maps the path body to the requested tool, if applicable.
> # BEGIN composed reference paths share one normalized physical-time span.
> An optional ``parent`` maps this path's reference frame into the final
> reference frame. Both paths sample the same progress before composition;
> a changing parent is supplied geometry, not a prediction of external motion.
> # END fixed left/right transforms remain ordinary path special cases.

## `PosePath.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L148)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray(self.progress, float).copy
np.asarray
self.progress
self.poses
self.attachment
self.progress.shape
np.isfinite(self.progress).all
np.isfinite
np.any
np.diff
self.source
self.parent
np.isin(self.parent.progress, self.progress).all
np.isin
self.parent.progress
self.motion_model
self.motion_model.kind
np.asarray(self.coordinates, float).copy
self.coordinates
self.coordinates.shape
self.motion_model.dimension
np.isfinite(self.coordinates).all
np.allclose
self.motion_model.poses
```

写入目标：

```python
self.progress
self.poses
self.attachment
self.coordinates
self._quaternions
```

直接调用符号：

```python
np.asarray(self.progress, float).copy
np.asarray
_poses_array
_pose_array
len
np.isfinite(self.progress).all
np.isfinite
np.any
np.diff
ValueError
isinstance
TypeError
np.isin(self.parent.progress, self.progress).all
np.isin
np.asarray(self.coordinates, float).copy
np.isfinite(self.coordinates).all
np.allclose
self.motion_model.poses
_quaternion
```

## `PosePath.sample`

[定义位置](../../../manipulation/planning_problem.py#L176)。

输入签名：

```python
def sample(self, progress, *, side='right'):
```

实现说明：

> Return position/quaternion; ``side`` selects a knot's actual branch.

显式输出分支：

```python
(position, quaternion)
```

读取的对象属性：

```python
self.motion_model
self.progress
self.poses
self._quaternions
values[0].expand
progress.shape
torch.searchsorted(keys, progress.contiguous(), right=side == 'right').clamp
torch.searchsorted
progress.contiguous
((progress - keys[index - 1]) / (keys[index] - keys[index - 1])).clamp
self.coordinates
model.reference_pose
model.axis
model.kind
torch.cat
torch.cos
torch.sin
base_q.expand
model.axis_point
(base_p - center).expand
model.basis[:, :2].T
model.basis
_tensor(self.attachment[:3, 3], progress).expand
self.attachment
self.parent
self.parent.sample
```

写入目标：

```python
position
keys
values
quaternion
index
fraction
model
coordinate
base_p
base_q
axis
angle
half
turn
center
offset
parent_position
parent_quaternion
```

直接调用符号：

```python
_interpolate
_tensor
len
values[0].expand
torch.searchsorted(keys, progress.contiguous(), right=side == 'right').clamp
torch.searchsorted
progress.contiguous
((progress - keys[index - 1]) / (keys[index] - keys[index - 1])).clamp
_slerp
_quaternion
torch.cat
torch.cos
torch.sin
quaternion_multiply
base_q.expand
quaternion_rotate
(base_p - center).expand
_tensor(self.attachment[:3, 3], progress).expand
self.parent.sample
```

## `PosePath.sample_numpy`

[定义位置](../../../manipulation/planning_problem.py#L220)。

输入签名：

```python
def sample_numpy(self, progress):
```

显式输出分支：

```python
output
```

读取的对象属性：

```python
torch.as_tensor
np.asarray
torch.float64
self.sample
np.broadcast_to(np.eye(4), (*values.shape, 4, 4)).copy
np.broadcast_to
np.eye
values.shape
position.detach().numpy
position.detach
Rotation.from_quat(quaternion.detach().numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape
Rotation.from_quat(quaternion.detach().numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix
Rotation.from_quat
quaternion.detach().numpy().reshape
quaternion.detach().numpy
quaternion.detach
```

写入目标：

```python
values
position
quaternion
output
output[..., :3, 3]
output[..., :3, :3]
```

直接调用符号：

```python
torch.as_tensor
np.asarray
self.sample
np.broadcast_to(np.eye(4), (*values.shape, 4, 4)).copy
np.broadcast_to
np.eye
position.detach().numpy
position.detach
Rotation.from_quat(quaternion.detach().numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape
Rotation.from_quat(quaternion.detach().numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix
Rotation.from_quat
quaternion.detach().numpy().reshape
quaternion.detach().numpy
quaternion.detach
```

## `NumericalTolerance`

[定义位置](../../../manipulation/planning_problem.py#L232)。

```python
class NumericalTolerance():
```

类字段：

```python
position_m: float = 0.0001
rotation_rad: float = 0.001
joint: float = 1e-05
velocity: float = 0.0001
```

实现说明：

> Solver acceptance errors; these are never inferred skill freedom.
>
> Pose intervals use target-local XYZ and rotation-vector components, so
> ``position_m`` and ``rotation_rad`` apply to each component of those
> intervals. Reports expose both component maxima and vector norms; a norm
> must not be compared as if it were this componentwise allowance.

## `NumericalTolerance.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L245)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.isfinite
vars(self).values
```

写入目标：

```python
v
```

直接调用符号：

```python
any
np.isfinite
vars(self).values
vars
ValueError
```

## `MotionPhase`

[定义位置](../../../manipulation/planning_problem.py#L252)。

```python
class MotionPhase():
```

类字段：

```python
name: str
reference: PosePath
metadata: dict = field(default_factory=dict)
```

实现说明：

> One reference/interaction interval; neither its nodes nor end are goals.

## `MotionPhase.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L258)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.name
```

写入目标：

无显式条目。

直接调用符号：

```python
isinstance
ValueError
```

## `reference_phase`

[定义位置](../../../manipulation/planning_problem.py#L264)。

输入签名：

```python
def reference_phase(reference, *, skill_id, source):
```

实现说明：

> Carry a generated nominal path and its physical boundary declaration.
>
> ``reference`` is the result of representation.generate_reference. Reuse
> its explicit eef_path, including the separately sampled parent, body
> correspondence and attachment. Rebuilding a path from composed EEF or
> object knots would cut rotating offsets and change intermediate poses.
>
> The returned MotionPhase is reference data only. Its raw composed path
> and metadata['reference_boundaries'] remain distinct until the physical
> clock and measured tool twist are bound by BoundToolReference. It declares neither a
> task goal nor a physical attachment/collision permission. The caller must
> independently supply real robot state, typed requirements and scene
> evidence to PlanningProblem and the existing planning-scene binder. Pass
> metadata['reference_timestamps'] and phase.reference.progress to that
> problem's source_timestamps/source_progress for a multi-sample phase. A
> single observation has no duration to supply as a timed motion interval.
> Original observed frame IDs/times and events remain separate from inserted
> event-time knots.

显式输出分支：

```python
MotionPhase(skill_id, path, metadata={'skill_id': skill_id, 'reference_source': deepcopy(source), 'reference_timestamps': times.copy(), 'source_frame_indices': frames, 'source_timestamps': original_times, 'source_events': deepcopy(reference.get('events', [])), 'reference_boundaries': deepcopy(reference['eef_boundaries']), 'scope': 'Geometric reference only; task requirements and scene permissions are independent'})
```

读取的对象属性：

```python
reference.get
np.asarray
times.shape
np.isfinite(times).all
np.isfinite
np.any
np.diff
np.zeros
np.array_equal
path.progress
np.allclose
path.sample_numpy
times.copy
```

写入目标：

```python
frames
original_times
path
poses
times
progress
```

直接调用符号：

```python
isinstance
ValueError
validate_timeline
reference.get
_poses_array
np.asarray
len
np.isfinite(times).all
np.isfinite
np.any
np.diff
np.zeros
np.array_equal
np.allclose
path.sample_numpy
MotionPhase
deepcopy
times.copy
```

## `PoseProcessRequirement`

[定义位置](../../../manipulation/planning_problem.py#L317)。

```python
class PoseProcessRequirement():
```

类字段：

```python
start: float
end: float
reference: PosePath
pose_bounds: np.ndarray
source: object
```

实现说明：

> An explicitly required pose relation on an independent progress scope.
>
> Bounds use target-local XYZ/rotation-vector components. ``source`` must
> identify the task/model evidence requiring this relation, rather than the
> mere existence of a demonstration path. Ordinary reference phases never
> create this requirement. Its path is a relation target, not a joint path.

## `PoseProcessRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L331)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray(self.pose_bounds, float).copy
np.asarray
self.pose_bounds
self.source
self.start
self.end
np.inf
self.pose_bounds.shape
np.isnan(self.pose_bounds).any
np.isnan
np.any
np.isposinf(self.pose_bounds[:, 0]).any
np.isposinf
np.isneginf(self.pose_bounds[:, 1]).any
np.isneginf
```

写入目标：

```python
self.pose_bounds
```

直接调用符号：

```python
np.asarray(self.pose_bounds, float).copy
np.asarray
np.isnan(self.pose_bounds).any
np.isnan
np.any
np.isposinf(self.pose_bounds[:, 0]).any
np.isposinf
np.isneginf(self.pose_bounds[:, 1]).any
np.isneginf
ValueError
```

## `PoseProcessRequirement.error`

[定义位置](../../../manipulation/planning_problem.py#L340)。

输入签名：

```python
def error(self, position, quaternion, progress):
```

显式输出分支：

```python
_pose_error(position, quaternion, *self.reference.sample(local))
```

读取的对象属性：

```python
((progress - self.start) / (self.end - self.start)).clamp
self.start
self.end
self.reference.sample
self.reference
```

写入目标：

```python
local
```

直接调用符号：

```python
((progress - self.start) / (self.end - self.start)).clamp
_pose_error
self.reference.sample
```

## `TimeBindingError`

[定义位置](../../../manipulation/planning_problem.py#L347)。

```python
class TimeBindingError(ValueError):
```

类字段：

无显式条目。

实现说明：

> An explicit event-time relation cannot be resolved from its own evidence.

## `event_occurrence_key`

[定义位置](../../../manipulation/planning_problem.py#L351)。

输入签名：

```python
def event_occurrence_key(event):
```

实现说明：

> Identify one occurrence; original event names alone never identify a retry.

显式输出分支：

```python
identity
```

读取的对象属性：

```python
event.get
```

写入目标：

```python
get
identity
value
```

直接调用符号：

```python
isinstance
getattr
get
any
type
TimeBindingError
```

## `EventTimeConstraint`

[定义位置](../../../manipulation/planning_problem.py#L362)。

```python
class EventTimeConstraint():
```

类字段：

```python
check_target: str
window_s: tuple[float | None, float | None]
time_basis: str
source: object
origin_time_s: float | None = None
origin_event: dict | None = None
goal_id: tuple[str, int] | None = None
condition_id: str | None = None
```

实现说明：

> An evidenced window in the original request's physical seconds.
>
> Offsets are relative to exactly one explicit origin. A goal arrival and
> an observed effect name their separate evidence targets. Planning wall
> time and demonstration timestamps are never substituted for physics time.

## `EventTimeConstraint.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L378)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.window_s
self.check_target
self.time_basis
self.source
np.isfinite
self.origin_time_s
self.origin_event
self.origin_event.get
self.goal_id
object.__setattr__
self.condition_id
```

写入目标：

```python
window
value
```

直接调用符号：

```python
tuple
len
all
any
np.isfinite
TimeBindingError
event_occurrence_key
self.origin_event.get
type
object.__setattr__
isinstance
deepcopy
```

## `resolve_event_window`

[定义位置](../../../manipulation/planning_problem.py#L406)。

输入签名：

```python
def resolve_event_window(constraint, history, *, predicted_origins=()):
```

实现说明：

> Resolve an absolute window; only actual T witnesses can be past origins.
>
> Predictions are accepted only when the caller explicitly supplies them for
> a future origin in this candidate. They are never appended to real history.
> Repeated observations of the same occurrence use its first actual T
> witness. Retries cannot inherit it because their occurrence is different.

显式输出分支：

```python
tuple((None if value is None else float(origin) + value for value in constraint.window_s))
```

读取的对象属性：

```python
constraint.origin_time_s
constraint.origin_event
row.get
np.isfinite(values).all
np.isfinite
constraint.window_s
```

写入目标：

```python
constraint
origin
key
target
witnesses
row
identities
values
value
```

直接调用符号：

```python
isinstance
EventTimeConstraint
event_occurrence_key
selected
row.get
str
len
TimeBindingError
np.isfinite(values).all
np.isfinite
float
min
tuple
```

## `resolve_event_window.selected`

[定义位置](../../../manipulation/planning_problem.py#L420)。

输入签名：

```python
def selected(row):
```

显式输出分支：

```python
event_occurrence_key(row) == key and row.get('check_target') == target and ('goal_id' not in constraint.origin_event or tuple(row.get('goal_id') or ()) == tuple(constraint.origin_event['goal_id'])) and ('condition_id' not in constraint.origin_event or row.get('condition_id') == constraint.origin_event['condition_id'])
```

读取的对象属性：

```python
row.get
constraint.origin_event
```

写入目标：

无显式条目。

直接调用符号：

```python
event_occurrence_key
row.get
tuple
```

## `PlanningEvent`

[定义位置](../../../manipulation/planning_problem.py#L442)。

```python
class PlanningEvent():
```

类字段：

```python
event_id: str
skill_id: str
progress: float
source_frame: int
kind: str
source: object
requires_evidence: str | None = None
stop: bool = False
occurrence: int = 0
timing: tuple[EventTimeConstraint, ...] = ()
```

实现说明：

> Original source identity plus a unique skill-instance location.
>
> A verification barrier grants no future evidence. If it needs a stop,
> zero joint velocity is part of the generated trajectory at this anchor.

## `PlanningEvent.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L460)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
object.__setattr__
self.timing
```

写入目标：

```python
value
```

直接调用符号：

```python
event_occurrence_key
object.__setattr__
tuple
isinstance
EventTimeConstraint
```

## `JointCoupling`

[定义位置](../../../manipulation/planning_problem.py#L469)。

```python
class JointCoupling():
```

类字段：

```python
joint_names: tuple[str, ...]
matrix: np.ndarray
target: np.ndarray
source: object
```

实现说明：

> Actual actuator coupling A q = b; not an inferred finger symmetry.

## `JointCoupling.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L476)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
self.matrix
self.target
self.matrix.ndim
self.matrix.shape
self.joint_names
self.source
np.isfinite(self.matrix).all
np.isfinite
np.isfinite(self.target).all
```

写入目标：

```python
self.matrix
self.target
```

直接调用符号：

```python
np.asarray
len
np.isfinite(self.matrix).all
np.isfinite
np.isfinite(self.target).all
ValueError
```

## `_pose_error`

[定义位置](../../../manipulation/planning_problem.py#L483)。

输入签名：

```python
def _pose_error(position, quaternion, target_position, target_quaternion):
```

显式输出分支：

```python
torch.cat((quaternion_rotate(inverse, position - target_position), quaternion_rotvec(quaternion_multiply(inverse, quaternion))), -1)
```

读取的对象属性：

```python
torch.cat
```

写入目标：

```python
inverse
```

直接调用符号：

```python
quaternion_conjugate
torch.cat
quaternion_rotate
quaternion_rotvec
quaternion_multiply
```

## `_first_event_query`

[定义位置](../../../manipulation/planning_problem.py#L491)。

输入签名：

```python
def _first_event_query(progress, anchor, trajectory_time=None):
```

实现说明：

> Select exactly one earliest event query, never an entire progress plateau.
>
> Complete rollout callers append exact anchors after the regular samples,
> so they provide actual query times. Ordered CPU/export queries may omit
> times. Duplicate representations of the same physical instant count once.

显式输出分支：

```python
matches & (indices == first)
```

读取的对象属性：

```python
(progress - anchor).abs
trajectory_time.shape
progress.shape
torch.where(matches, trajectory_time, float('inf')).amin
torch.where
matches.to(torch.int64).argmax
matches.to
torch.int64
torch.arange
progress.device
```

写入目标：

```python
matches
earliest
first
indices
```

直接调用符号：

```python
(progress - anchor).abs
ValueError
torch.where(matches, trajectory_time, float('inf')).amin
torch.where
float
matches.to(torch.int64).argmax
matches.to
torch.arange
```

## `anchor_pose`

[定义位置](../../../manipulation/planning_problem.py#L511)。

输入签名：

```python
def anchor_pose(position, quaternion, progress, anchor):
```

实现说明：

> Read an exact anchor from the full horizon, allowing duplicate queries.
>
> The backend appends actual FK at each event before calling constraints.
> Missing anchors deliberately produce nonfinite values, never a nearest
> sample surrogate. Outer validation must reject nonfinite solver results.

显式输出分支：

```python
(p, q)
```

读取的对象属性：

```python
(progress - anchor).abs
matches.to(torch.int64).argmax
matches.to
torch.int64
torch.gather
index[..., None, None].expand
index.shape
matches.any
torch.where
torch.full_like
```

写入目标：

```python
matches
index
p
q
found
```

直接调用符号：

```python
(progress - anchor).abs
matches.to(torch.int64).argmax
matches.to
torch.gather
index[..., None, None].expand
matches.any
torch.where
torch.full_like
float
```

## `_compose`

[定义位置](../../../manipulation/planning_problem.py#L534)。

输入签名：

```python
def _compose(p, q, transform):
```

显式输出分支：

```python
(p + quaternion_rotate(q, t), quaternion_multiply(q, _tensor(_quaternion(transform), q)))
```

读取的对象属性：

```python
_tensor(transform[:3, 3], p).expand_as
```

写入目标：

```python
t
```

直接调用符号：

```python
_tensor(transform[:3, 3], p).expand_as
_tensor
quaternion_rotate
quaternion_multiply
_quaternion
```

## `_mechanism_errors`

[定义位置](../../../manipulation/planning_problem.py#L540)。

输入签名：

```python
def _mechanism_errors(model, tool_from_carrier, position, quaternion):
```

实现说明：

> Differentiable counterpart of MotionModel.coordinates_of's pose check.
>
> Project the actual path body onto the same physical model, then measure
> position norm and SO(3) angle. The tool offset is removed before projecting:
> a small tool error does not establish compatibility of an attached body.
> This pointwise membership check does not recover winding. The independent
> mechanism requirement lifts the ordered actual candidate coordinates.

显式输出分支：

```python
(torch.linalg.vector_norm(error[..., :3], dim=-1), torch.linalg.vector_norm(error[..., 3:], dim=-1))
```

读取的对象属性：

```python
model.reference_pose
model.axis
base_q.expand_as
model.kind
torch.cat
torch.cos
torch.sin
model.axis_point
(base_p - center).expand_as
((p - base_p) * axis).sum
model.basis
basis.T
torch.linalg.vector_norm
torch.linalg
```

写入目标：

```python
p
q
base_p
base_q
axis
projected_q
relative
angle
half
turn
center
projected_p
basis
error
```

直接调用符号：

```python
_compose
_tensor
_quaternion
base_q.expand_as
quaternion_multiply
quaternion_conjugate
quaternion_axis_angle
torch.cat
torch.cos
torch.sin
quaternion_rotate
(base_p - center).expand_as
((p - base_p) * axis).sum
_pose_error
torch.linalg.vector_norm
```

## `CoordinateDirectionRequirement`

[定义位置](../../../manipulation/planning_problem.py#L579)。

```python
class CoordinateDirectionRequirement():
```

类字段：

```python
start: float
end: float
coordinate_index: int
kind: str
sign: int
minimum: float
tolerance: float
source: object
origin_coordinate: float | None = None
origin_source: object = None
```

实现说明：

> An evidenced signed change, monotonicity, or physical coordinate rate.
>
> ``minimum`` is a displacement for ``net`` and a rate for ``local_rate``;
> ``monotone`` uses zero and permits only the stated displacement tolerance.
> No direction is inferred from the demonstration's coordinate sequence.

## `CoordinateDirectionRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L597)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.source
self.start
self.end
np.inf
self.coordinate_index
self.kind
self.sign
self.minimum
self.tolerance
self.origin_coordinate
self.origin_source
np.isfinite
```

写入目标：

无显式条目。

直接调用符号：

```python
type
ValueError
np.isfinite
```

## `MechanismRequirement`

[定义位置](../../../manipulation/planning_problem.py#L612)。

```python
class MechanismRequirement():
```

类字段：

```python
start: float
end: float
model: MotionModel
tool_from_carrier: np.ndarray
coordinate_limits: np.ndarray | None
initial_coordinates: np.ndarray
source: object
directions: tuple[CoordinateDirectionRequirement, ...] = ()
```

实现说明：

> Current physical manifold/limits, independent of the seed reference.
>
> Angular coordinates are lifted from the ordered candidate trajectory, with
> the actual initial coordinate fixing its branch. A single state can check
> manifold membership and principal-angle feasibility only; it cannot certify
> winding. Final verification must include the complete scoped trajectory.

## `MechanismRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L629)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.directions
self.tool_from_carrier
np.asarray(self.initial_coordinates, float).copy
np.asarray
self.initial_coordinates
self.source
self.start
self.end
np.inf
self.model.kind
self.model
self.initial_coordinates.shape
self.model.dimension
np.isfinite(self.initial_coordinates).all
np.isfinite
self.coordinate_limits
np.asarray(self.coordinate_limits, float).copy
self.coordinate_limits.shape
np.isfinite(self.coordinate_limits).all
np.any
value.coordinate_index
value.end
value.kind
value.start
value.origin_coordinate
```

写入目标：

```python
self.directions
self.tool_from_carrier
self.initial_coordinates
self.coordinate_limits
value
```

直接调用符号：

```python
tuple
_pose_array
np.asarray(self.initial_coordinates, float).copy
np.asarray
np.isfinite(self.initial_coordinates).all
np.isfinite
ValueError
np.asarray(self.coordinate_limits, float).copy
np.isfinite(self.coordinate_limits).all
np.any
any
isinstance
replace
float
```

## `MechanismRequirement.angular_column`

[定义位置](../../../manipulation/planning_problem.py#L652)。

输入签名：

```python
def angular_column(self):
```

显式输出分支：

```python
0 if self.model.kind == 'revolute' else 2 if self.model.kind == 'planar' else None
```

读取的对象属性：

```python
self.model.kind
self.model
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `MechanismRequirement.coordinates`

[定义位置](../../../manipulation/planning_problem.py#L655)。

输入签名：

```python
def coordinates(self, position, quaternion, progress, *, trajectory_time=None):
```

实现说明：

> Recover actual candidate coordinates; no reference values enter the lift.

显式输出分支：

```python
((p - base_p) * axis).sum(-1, keepdim=True)
planar
angle[..., None] if model.kind == 'revolute' else torch.cat((planar, angle[..., None]), -1)
```

读取的对象属性：

```python
self.tool_from_carrier
self.model
model.reference_pose
model.axis
model.kind
((p - base_p) * axis).sum
model.basis
(relative[..., 1:] * axis).sum
w.square
along.square
torch.finfo(q.dtype).eps
torch.finfo
q.dtype
torch.atan2
torch.where
trajectory_time.shape
progress.shape
torch.isfinite(trajectory_time).all
torch.isfinite
torch.zeros_like
torch.argsort
torch.gather
self.start
self.end
within.to(torch.int64).argmax
within.to
torch.int64
self.initial_coordinates
self.angular_column
np.pi
torch.round
torch.sin
torch.cos
torch.cat
torch.cumsum
((sorted_progress - self.start).abs() <= 1e-07).any
(sorted_progress - self.start).abs
angle.gather
```

写入目标：

```python
p
q
model
base_p
axis
planar
relative
along
w
singular
angle
message
valid
trajectory_time
order
sorted_angle
sorted_progress
within
first
initial_angle
initial
branch
delta
increments
cumulative
lifted
complete_start
```

直接调用符号：

```python
_compose
_tensor
((p - base_p) * axis).sum
quaternion_multiply
quaternion_conjugate
_quaternion
(relative[..., 1:] * axis).sum
w.square
along.square
torch.finfo
torch.atan2
torch.where
ValueError
require_tensor
torch.isfinite(trajectory_time).all
torch.isfinite
torch.zeros_like
torch.argsort
torch.gather
within.to(torch.int64).argmax
within.to
float
torch.round
torch.sin
torch.cos
torch.cat
torch.cumsum
((sorted_progress - self.start).abs() <= 1e-07).any
(sorted_progress - self.start).abs
angle.gather
```

## `MechanismRequirement.errors`

[定义位置](../../../manipulation/planning_problem.py#L703)。

输入签名：

```python
def errors(self, position, quaternion):
```

显式输出分支：

```python
_mechanism_errors(self.model, self.tool_from_carrier, position, quaternion)
```

读取的对象属性：

```python
self.model
self.tool_from_carrier
```

写入目标：

无显式条目。

直接调用符号：

```python
_mechanism_errors
```

## `MechanismRequirement.project_residual`

[定义位置](../../../manipulation/planning_problem.py#L706)。

输入签名：

```python
def project_residual(self, position, quaternion, progress):
```

实现说明：

> Independent manifold equations; the allowed native coordinate stays free.

显式输出分支：

```python
torch.cat((translation, rotation), -1)
torch.cat((translation, off_axis), -1)
```

读取的对象属性：

```python
self.tool_from_carrier
self.model
model.reference_pose
model.basis
model.kind
torch.cat
self.coordinates
self.angular_column
torch.cos
torch.sin
model.axis_point
(base_p - center).expand_as
```

写入目标：

```python
p
q
model
base_p
base_q
basis
relative
rotation
translation
angle
turn
center
projected
off_axis
```

直接调用符号：

```python
_compose
_tensor
_quaternion
quaternion_multiply
quaternion_conjugate
quaternion_rotvec
torch.cat
self.coordinates
torch.cos
torch.sin
quaternion_rotate
(base_p - center).expand_as
```

## `MechanismRequirement.history_complete`

[定义位置](../../../manipulation/planning_problem.py#L730)。

输入签名：

```python
def history_complete(self, progress):
```

显式输出分支：

```python
((progress - self.start).abs() <= 1e-07).any(-1)
```

读取的对象属性：

```python
((progress - self.start).abs() <= 1e-07).any
(progress - self.start).abs
self.start
```

写入目标：

无显式条目。

直接调用符号：

```python
((progress - self.start).abs() <= 1e-07).any
(progress - self.start).abs
```

## `MechanismRequirement.coordinate_rates`

[定义位置](../../../manipulation/planning_problem.py#L734)。

输入签名：

```python
def coordinate_rates(self, quaternion, tool_linear_velocity, tool_angular_velocity):
```

实现说明：

> Map actual reference-frame tool twist to native carrier coordinates.

显式输出分支：

```python
(linear * axis).sum(-1, keepdim=True)
planar
angular if self.model.kind == 'revolute' else torch.cat((planar, angular), -1)
```

读取的对象属性：

```python
_tensor(self.tool_from_carrier[:3, 3], quaternion).expand_as
self.tool_from_carrier
torch.linalg.cross
torch.linalg
self.model.axis
self.model
self.model.kind
(linear * axis).sum
self.model.basis
self.model.reference_pose
(vector * axis).sum
(tool_angular_velocity * axis).sum
(torch.linalg.cross(tool_angular_velocity, vector) * axis).sum
(tool_angular_velocity * vector).sum
(w.square() + along.square()).clamp_min
w.square
along.square
torch.finfo(w.dtype).eps
torch.finfo
w.dtype
torch.cat
```

写入目标：

```python
offset
linear
axis
planar
carrier_q
relative
w
vector
along
numerator
angular
```

直接调用符号：

```python
quaternion_rotate
_tensor(self.tool_from_carrier[:3, 3], quaternion).expand_as
_tensor
torch.linalg.cross
(linear * axis).sum
quaternion_multiply
_quaternion
quaternion_conjugate
(vector * axis).sum
(tool_angular_velocity * axis).sum
(torch.linalg.cross(tool_angular_velocity, vector) * axis).sum
(tool_angular_velocity * vector).sum
(w.square() + along.square()).clamp_min
w.square
along.square
torch.finfo
torch.cat
```

## `MechanismRequirement.direction_residuals`

[定义位置](../../../manipulation/planning_problem.py#L759)。

输入签名：

```python
def direction_residuals(self, position, quaternion, progress, *, trajectory_time, tool_linear_velocity=None, tool_angular_velocity=None, mode='trajectory', use_tolerance=True):
```

实现说明：

> Check finite ordered queries; only final curves use instantaneous rates.
>
> A geometric OMPL path uses physical-time secants solely to screen a
> seed. Its downstream cubic must still pass true J(q) dq rate checks.
> Single-state seed membership deliberately proves no path direction.

显式输出分支：

```python
{}
result
```

读取的对象属性：

```python
self.directions
trajectory_time.shape
progress.shape
torch.argsort
torch.gather
self.coordinates
order[..., None].expand_as
value.kind
self.coordinate_rates
direction.tolerance
direction.start
self.start
direction.end
direction.coordinate_index
torch.zeros_like
direction.kind
(scoped_progress - direction.start).abs
(scoped_progress - direction.end).abs
at_start.to(torch.int64).argmax
at_start.to
torch.int64
torch.flip(at_end, [-1]).to(torch.int64).argmax
torch.flip(at_end, [-1]).to
torch.flip
direction.origin_coordinate
values.gather
direction.sign
torch.relu
direction.minimum
at_end.any
at_start.any
torch.ones_like
torch.where
scoped_progress.amin
violation.scatter
elapsed.clamp_min
torch.finfo(elapsed.dtype).tiny
torch.finfo
elapsed.dtype
torch.cat
```

写入目标：

```python
order
restore
times
scoped_progress
coordinates
rates
value
result
index
direction
tolerance
active
values
violation
at_start
at_end
first
last
origin
change
error
complete
elapsed
pair
signed
result[f'direction:{index}:{direction.kind}']
```

直接调用符号：

```python
ValueError
torch.argsort
torch.gather
self.coordinates
order[..., None].expand_as
any
self.coordinate_rates
enumerate
max
torch.zeros_like
(scoped_progress - direction.start).abs
(scoped_progress - direction.end).abs
at_start.to(torch.int64).argmax
at_start.to
torch.flip(at_end, [-1]).to(torch.int64).argmax
torch.flip(at_end, [-1]).to
torch.flip
values.gather
torch.relu
at_end.any
at_start.any
torch.ones_like
torch.where
float
scoped_progress.amin
violation.scatter
elapsed.clamp_min
torch.finfo
torch.cat
```

## `SupportRequirement`

[定义位置](../../../manipulation/planning_problem.py#L826)。

```python
class SupportRequirement():
```

类字段：

```python
progress: float
tool_from_carrier: np.ndarray
body_points: np.ndarray
contact_points: np.ndarray
outward_normal: np.ndarray
surface: object
gap_tolerance: float
normal_tolerance: float
margin: float
source: object
hard_components: tuple[str, ...] = ('body', 'footprint')
```

实现说明：

> A finite observed support and a specific material face at one event.

## `SupportRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L844)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.tool_from_carrier
self.body_points
self.contact_points
self.outward_normal
self.surface
self.margin
self.gap_tolerance
self.normal_tolerance
self.source
self.hard_components
```

写入目标：

```python
self.tool_from_carrier
self.body_points
self.contact_points
self.outward_normal
self.hard_components
```

直接调用符号：

```python
_pose_array
_points
_unit
_support_arguments
isinstance
ValueError
tuple
len
set
```

## `SupportRequirement.raw_inequalities`

[定义位置](../../../manipulation/planning_problem.py#L862)。

输入签名：

```python
def raw_inequalities(self, position, quaternion, numerical):
```

实现说明：

> Return four signed per-pose requirements, each satisfied at <= 0.
>
> Lengths remain in metres and the normal error remains a chord length.
> Keeping feasible-side values lets AL multipliers decrease without
> inventing a tangent location or yaw target inside this finite support.

显式输出分支：

```python
{'gap': gaps.abs().amax(-1) - self.gap_tolerance, 'body': -height.amin(-1) - numerical.position_m, 'footprint': footprint + self.margin - numerical.position_m, 'normal': chord - angular_bound}
```

读取的对象属性：

```python
self.tool_from_carrier
_tensor(self.body_points, p).expand
self.body_points
p.shape
self.body_points.shape
_tensor(self.contact_points, p).expand
self.contact_points
self.contact_points.shape
self.surface.origin
self.surface
self.surface.normal
self.surface.basis
self.surface._hull.equations
self.surface._hull
(xy @ equations[:, :2].T + equations[:, 2]).amax
equations[:, :2].T
_tensor(self.outward_normal, q).expand
self.outward_normal
q.shape
torch.linalg.vector_norm
torch.linalg
np.sin
self.normal_tolerance
gaps.abs().amax
gaps.abs
self.gap_tolerance
height.amin
numerical.position_m
self.margin
```

写入目标：

```python
p
q
body
contact
origin
normal
gaps
height
xy
equations
footprint
outward
chord
angular_bound
```

直接调用符号：

```python
_compose
quaternion_rotate
_tensor(self.body_points, p).expand
_tensor
_tensor(self.contact_points, p).expand
(xy @ equations[:, :2].T + equations[:, 2]).amax
_tensor(self.outward_normal, q).expand
torch.linalg.vector_norm
np.sin
gaps.abs().amax
gaps.abs
height.amin
```

## `SupportRequirement.normalized_inequalities`

[定义位置](../../../manipulation/planning_problem.py#L889)。

输入签名：

```python
def normalized_inequalities(self, position, quaternion, numerical):
```

实现说明：

> Scale units for AL without changing any acceptance right-hand side.
>
> The three length requirements share the declared gap scale, bounded
> below by numerical precision. In particular, a tight boundary rounding
> allowance does not become its own large optimization weight. The same
> rule uses chord lengths for the normal and supports zero declarations.

显式输出分支：

```python
{name: value / (normal_scale if name == 'normal' else length_scale) for (name, value) in self.raw_inequalities(position, quaternion, numerical).items()}
```

读取的对象属性：

```python
self.gap_tolerance
numerical.position_m
np.sin
self.normal_tolerance
numerical.rotation_rad
np.pi
self.raw_inequalities(position, quaternion, numerical).items
self.raw_inequalities
```

写入目标：

```python
length_scale
normal_scale
name
value
```

直接调用符号：

```python
max
np.sin
min
self.raw_inequalities(position, quaternion, numerical).items
self.raw_inequalities
```

## `SupportRequirement.evaluate`

[定义位置](../../../manipulation/planning_problem.py#L906)。

输入签名：

```python
def evaluate(self, position, quaternion, numerical):
```

实现说明：

> Sum positive violations of the explicitly declared hard components.

显式输出分支：

```python
sum((torch.relu(components[name]) for name in self.hard_components), position[..., 0] * 0.0)
```

读取的对象属性：

```python
self.raw_inequalities
torch.relu
self.hard_components
```

写入目标：

```python
components
name
```

直接调用符号：

```python
self.raw_inequalities
sum
torch.relu
```

## `PoseEventRequirement`

[定义位置](../../../manipulation/planning_problem.py#L917)。

```python
class PoseEventRequirement():
```

类字段：

```python
progress: float
tool_from_function: np.ndarray
target: np.ndarray
position_tolerance_m: float
rotation_tolerance_rad: float
source: object
actual_relation: dict | None = None
yaw_axis: np.ndarray | None = None
yaw_tolerance_rad: float | None = None
```

实现说明：

> A material-frame event with explicit position and orientation tolerances.
>
> These acceptance thresholds are not inferred freedom for the neighbouring
> motion. In particular, a larger generic solver tolerance never replaces a
> tighter threshold already specified by the source functional event.
>
> ``actual_relation=None`` explicitly describes a fixed tool calibration
> against a fixed world target. Owner-relative goals instead name the
> currently observed target owner and its material target transform. A held
> function additionally names its owner/material frame and the already bound
> attachment condition. These owner inputs describe the observations consumed
> by the independent posthoc evaluator. Online control consumes its own robot
> and task-effect feedback; this geometric metadata belongs to evaluation.

## `PoseEventRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L949)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.tool_from_function
self.target
self.source
np.isfinite
self.progress
self.position_tolerance_m
np.inf
self.rotation_tolerance_rad
np.pi
self.yaw_axis
self.yaw_tolerance_rad
np.bool_
self.actual_relation
relation[owner_key].strip
relation.get
condition.strip
```

写入目标：

```python
self.tool_from_function
self.target
self.yaw_axis
self.yaw_tolerance_rad
relation
allowed
owner_key
pose_key
relation[pose_key]
condition
self.actual_relation
```

直接调用符号：

```python
_pose_array
np.isfinite
ValueError
_unit
isinstance
float
set
deepcopy
relation[owner_key].strip
relation.get
condition.strip
```

## `PoseEventRequirement.component_tolerances`

[定义位置](../../../manipulation/planning_problem.py#L993)。

输入签名：

```python
def component_tolerances(self):
```

实现说明：

> Metric bounds for actual constrained components; free yaw is absent.

显式输出分支：

```python
result
```

读取的对象属性：

```python
self.position_tolerance_m
self.rotation_tolerance_rad
self.yaw_axis
self.yaw_tolerance_rad
np.pi
```

写入目标：

```python
result
result['yaw']
```

直接调用符号：

无显式条目。

## `PoseEventRequirement.error_components`

[定义位置](../../../manipulation/planning_problem.py#L1000)。

输入签名：

```python
def error_components(self, position, quaternion):
```

实现说明：

> Return physical errors plus signed yaw/full-rotation diagnostics.
>
> C is the constrained material frame and C* its target. The relative
> quaternion represents R_C*^T R_C, so yaw_axis is used in C* coordinates.
> Removing its nearest axis rotation leaves the swing/tilt quaternion.
> This is a finite-rotation decomposition, not Euler-angle subtraction.

显式输出分支：

```python
result
```

读取的对象属性：

```python
self.tool_from_function
self.target
torch.linalg.vector_norm
torch.linalg
self.yaw_axis
torch.cat
torch.cos
torch.sin
result.update
angle.abs
```

写入目标：

```python
p
q
relative
rotation
result
axis
angle
half
turn
swing
```

直接调用符号：

```python
_compose
quaternion_multiply
quaternion_conjugate
_tensor
_quaternion
torch.linalg.vector_norm
quaternion_rotvec
quaternion_axis_angle
torch.cat
torch.cos
torch.sin
result.update
angle.abs
```

## `PoseEventRequirement.errors`

[定义位置](../../../manipulation/planning_problem.py#L1024)。

输入签名：

```python
def errors(self, position, quaternion):
```

实现说明：

> Position and constrained orientation; use error_components for yaw.

显式输出分支：

```python
(values['position'], values['rotation'])
```

读取的对象属性：

```python
self.error_components
```

写入目标：

```python
values
```

直接调用符号：

```python
self.error_components
```

## `PoseEventRequirement.errors_numpy`

[定义位置](../../../manipulation/planning_problem.py#L1029)。

输入签名：

```python
def errors_numpy(self, tool_poses):
```

实现说明：

> Evaluate the same tensor geometry for CPU preparation and diagnostics.

显式输出分支：

```python
{name: value.detach().numpy() for (name, value) in self.error_components(torch.as_tensor(poses[..., :3, 3].copy(), dtype=torch.float64), torch.as_tensor(_quaternion(poses), dtype=torch.float64)).items()}
```

读取的对象属性：

```python
np.asarray
value.detach().numpy
value.detach
self.error_components(torch.as_tensor(poses[..., :3, 3].copy(), dtype=torch.float64), torch.as_tensor(_quaternion(poses), dtype=torch.float64)).items
self.error_components
torch.as_tensor
poses[..., :3, 3].copy
torch.float64
```

写入目标：

```python
poses
name
value
```

直接调用符号：

```python
np.asarray
value.detach().numpy
value.detach
self.error_components(torch.as_tensor(poses[..., :3, 3].copy(), dtype=torch.float64), torch.as_tensor(_quaternion(poses), dtype=torch.float64)).items
self.error_components
torch.as_tensor
poses[..., :3, 3].copy
_quaternion
```

## `PoseEventRequirement.evaluate`

[定义位置](../../../manipulation/planning_problem.py#L1037)。

输入签名：

```python
def evaluate(self, position, quaternion):
```

显式输出分支：

```python
sum((torch.relu(values[name] - bound) for (name, bound) in self.component_tolerances.items()))
```

读取的对象属性：

```python
self.error_components
torch.relu
self.component_tolerances.items
self.component_tolerances
```

写入目标：

```python
values
name
bound
```

直接调用符号：

```python
self.error_components
sum
torch.relu
self.component_tolerances.items
```

## `CoordinateEventRequirement`

[定义位置](../../../manipulation/planning_problem.py#L1046)。

```python
class CoordinateEventRequirement():
```

类字段：

```python
progress: float
mechanism: MechanismRequirement
target_coordinate: float
tolerance: float
source: object
coordinate_mode: str = 'absolute'
coordinate_origin: float | None = None
```

实现说明：

> A signed metric/unwrapped native event in one current mechanism.

## `CoordinateEventRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L1056)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.mechanism.model.kind
self.mechanism.model
self.mechanism
self.source
self.mechanism.start
self.progress
self.mechanism.end
np.isfinite
self.target_coordinate
self.tolerance
np.inf
self.coordinate_mode
self.coordinate_origin
```

写入目标：

无显式条目。

直接调用符号：

```python
np.isfinite
ValueError
```

## `CoordinateEventRequirement.absolute_target`

[定义位置](../../../manipulation/planning_problem.py#L1068)。

输入签名：

```python
def absolute_target(self):
```

显式输出分支：

```python
self.target_coordinate if self.coordinate_mode == 'absolute' else self.coordinate_origin + self.target_coordinate
```

读取的对象属性：

```python
self.coordinate_mode
self.target_coordinate
self.coordinate_origin
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `CoordinateEventRequirement.kind`

[定义位置](../../../manipulation/planning_problem.py#L1072)。

输入签名：

```python
def kind(self):
```

显式输出分支：

```python
self.mechanism.model.kind
```

读取的对象属性：

```python
self.mechanism.model.kind
self.mechanism.model
self.mechanism
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `CoordinateEventRequirement.error`

[定义位置](../../../manipulation/planning_problem.py#L1075)。

输入签名：

```python
def error(self, position, quaternion, progress, *, trajectory_time=None):
```

显式输出分支：

```python
difference.abs()
```

读取的对象属性：

```python
self.mechanism.coordinates
self.mechanism
self.absolute_target
self.kind
torch.atan2
torch.sin
torch.cos
torch.where
self.mechanism.history_complete
difference.abs
```

写入目标：

```python
coordinate
difference
local
```

直接调用符号：

```python
self.mechanism.coordinates
torch.atan2
torch.sin
torch.cos
torch.where
self.mechanism.history_complete
difference.abs
```

## `CoordinateEventRequirement.evaluate`

[定义位置](../../../manipulation/planning_problem.py#L1084)。

输入签名：

```python
def evaluate(self, position, quaternion, progress, *, trajectory_time=None):
```

显式输出分支：

```python
torch.relu(self.error(position, quaternion, progress, trajectory_time=trajectory_time) - self.tolerance)
```

读取的对象属性：

```python
torch.relu
self.error
self.tolerance
```

写入目标：

无显式条目。

直接调用符号：

```python
torch.relu
self.error
```

## `InteriorRequirement`

[定义位置](../../../manipulation/planning_problem.py#L1091)。

```python
class InteriorRequirement():
```

类字段：

```python
start: float
end: float
tool_from_carrier: np.ndarray
body_points: np.ndarray
halfspaces: np.ndarray
margin: float
source: object
```

实现说明：

> All material body samples remain inside evidenced convex free geometry.

## `InteriorRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L1101)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.tool_from_carrier
self.body_points
InteriorRegion(self.halfspaces, str(self.source)).halfspaces
self.halfspaces
self.source
self.start
self.end
np.isfinite
self.margin
```

写入目标：

```python
self.tool_from_carrier
self.body_points
self.halfspaces
```

直接调用符号：

```python
_pose_array
_points
InteriorRegion
str
np.isfinite
ValueError
```

## `InteriorRequirement.evaluate`

[定义位置](../../../manipulation/planning_problem.py#L1109)。

输入签名：

```python
def evaluate(self, position, quaternion, numerical):
```

显式输出分支：

```python
torch.relu(error - numerical.position_m)
```

读取的对象属性：

```python
self.tool_from_carrier
_tensor(self.body_points, p).expand
self.body_points
p.shape
self.body_points.shape
self.halfspaces
(points @ h[:, :3].T + h[:, 3] + self.margin).amax
h[:, :3].T
self.margin
torch.relu
numerical.position_m
```

写入目标：

```python
p
q
points
h
error
```

直接调用符号：

```python
_compose
quaternion_rotate
_tensor(self.body_points, p).expand
_tensor
(points @ h[:, :3].T + h[:, 3] + self.margin).amax
torch.relu
```

## `PlanningProblem`

[定义位置](../../../manipulation/planning_problem.py#L1120)。

```python
class PlanningProblem():
```

类字段：

```python
joint_names: tuple[str, ...]
start_q: np.ndarray
start_dq: np.ndarray
tool_link: str
reference_frame: str
base_pose: np.ndarray
tool_transform: np.ndarray
phases: tuple[MotionPhase, ...]
events: tuple[PlanningEvent, ...] = ()
couplings: tuple[JointCoupling, ...] = ()
numerical: NumericalTolerance = field(default_factory=NumericalTolerance)
conditions: tuple[dict, ...] = ()
supports: tuple[SupportRequirement, ...] = ()
interiors: tuple[InteriorRequirement, ...] = ()
pose_events: tuple[PoseEventRequirement, ...] = ()
coordinate_events: tuple[CoordinateEventRequirement, ...] = ()
pose_relations: tuple[PoseProcessRequirement, ...] = ()
mechanisms: tuple[MechanismRequirement, ...] = ()
source_timestamps: np.ndarray | None = None
source_progress: np.ndarray | None = None
start_progress: float = 0.0
end_progress: float | None = None
```

实现说明：

> One current skill; source timing and real requirements have separate owners.
>
> Pose paths initialize the search and support deviation reports. Only the
> explicit goal/process collections below constrain generated robot motion.
> ``source_timestamps`` and ``source_progress`` retain the bound source map;
> the planner may choose execution timing once and freeze that binding.

## `PlanningProblem.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L1153)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.joint_names
np.asarray(self.start_q, float).copy
np.asarray
self.start_q
np.asarray(self.start_dq, float).copy
self.start_dq
self.start_q.shape
self.start_dq.shape
np.isfinite(self.start_q).all
np.isfinite
np.isfinite(self.start_dq).all
self.phases
self.tool_link
self.reference_frame
self.base_pose
self.tool_transform
self.end_progress
self.phase_count
np.isfinite([self.start_progress, self.end_progress]).all
self.start_progress
phase.metadata
skill_ids.extend
event.skill_id
self.events
event.event_id
event.source
event.progress
event.source_frame
identities.add
self.pose_relations
self.mechanisms
self.interiors
requirement.end
self.pose_events
self.coordinate_events
self.supports
requirement.progress
event.mechanism
self.source_timestamps
self.source_progress
np.asarray(self.source_timestamps, float).copy
np.asarray(self.source_progress, float).copy
self.source_timestamps.ndim
self.source_progress.shape
self.source_timestamps.shape
np.isfinite(self.source_timestamps).all
np.isfinite(self.source_progress).all
np.any
np.diff
```

写入目标：

```python
self.joint_names
name
self.start_q
self.start_dq
n
self.base_pose
self.tool_transform
self.end_progress
skill_ids
phase
event
value
identities
identity
requirement
mechanism
self.source_timestamps
self.source_progress
```

直接调用符号：

```python
tuple
setattr
getattr
np.asarray(self.start_q, float).copy
np.asarray
np.asarray(self.start_dq, float).copy
len
set
any
isinstance
np.isfinite(self.start_q).all
np.isfinite
np.isfinite(self.start_dq).all
ValueError
_pose_array
float
np.isfinite([self.start_progress, self.end_progress]).all
skill_ids.extend
type
identities.add
np.asarray(self.source_timestamps, float).copy
np.asarray(self.source_progress, float).copy
np.isfinite(self.source_timestamps).all
np.isfinite(self.source_progress).all
np.any
np.diff
```

## `PlanningProblem.phase_count`

[定义位置](../../../manipulation/planning_problem.py#L1208)。

输入签名：

```python
def phase_count(self):
```

显式输出分支：

```python
len(self.phases)
```

读取的对象属性：

```python
self.phases
```

写入目标：

无显式条目。

直接调用符号：

```python
len
```

## `PlanningProblem.bound_motion_block`

[定义位置](../../../manipulation/planning_problem.py#L1212)。

输入签名：

```python
def bound_motion_block(self, goal_identities=None, satisfied_dependencies=()):
```

实现说明：

> Copy the authorized interval without inventing stops or new clocks.
>
> Only an explicit unmet evidence dependency ends a block. Ordinary
> events and reference phase boundaries do not split moving process
> relations. Original phase coordinates and process origins survive;
> point-goal identities refer to the request ledger made before this
> copy. The boundary event remains available for actual T/F/U reporting.

显式输出分支：

```python
(block, record)
```

读取的对象属性：

```python
self.events
event.requires_evidence
event.progress
self.start_progress
self.end_progress
boundary.progress
block.events
goal_identities.get
row.progress
block.source_timestamps
block.source_progress
block.start_progress
np.r_
np.interp
block.conditions
condition.get
block.__post_init__
self.phase_count
block.source_time_map
```

写入目标：

```python
satisfied
dependencies
event
boundary
end
block
block.end_progress
block.events
identities
name
rows
original
selected
index
row
identities[name]
keep
progress
block.source_timestamps
block.source_progress
block.conditions
condition
record
```

直接调用符号：

```python
frozenset
min
float
ValueError
deepcopy
tuple
getattr
list
range
len
goal_identities.get
set
enumerate
setattr
np.interp
condition.get
block.__post_init__
sorted
asdict
```

## `PlanningProblem.requirement_consumption`

[定义位置](../../../manipulation/planning_problem.py#L1271)。

输入签名：

```python
def requirement_consumption(self):
```

实现说明：

> Describe existing consumers without introducing another request format.
>
> Collection/index identities address the requirements used by residuals,
> scene binding and execution. The returned records are independent copies;
> changing them cannot change a goal or grant execution evidence. Ordinary
> reference samples are deliberately absent. This inventory describes the
> current complete-skill compiler, not a completed finite-block backend.

显式输出分支：

```python
tuple(rows)
```

读取的对象属性：

```python
requirement.progress
requirement.start
requirement.end
self.start_progress
self.phase_count
requirement.source
requirement.coordinate_limits
requirement.directions
direction.start
direction.end
direction.source
self.conditions
condition.get
phase.metadata
self.phases
phase.metadata.get
self.events
event.kind
event.requires_evidence
event.stop
event.timing
consumers.append
consumers.extend
event.progress
event.source
```

写入目标：

```python
rows
name
category
index
requirement
scope
consumers
categories
direction_index
direction
condition
kind
point
spans
phase
event
```

直接调用符号：

```python
enumerate
getattr
hasattr
float
add
condition.get
ValueError
tuple
phase.metadata.get
len
consumers.append
consumers.extend
event_occurrence_key
```

## `PlanningProblem.requirement_consumption.add`

[定义位置](../../../manipulation/planning_problem.py#L1282)。

输入签名：

```python
def add(identity, categories, consumers, scope, source, **details):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
rows.append
```

写入目标：

无显式条目。

直接调用符号：

```python
rows.append
deepcopy
```

## `PlanningProblem.source_time_map`

[定义位置](../../../manipulation/planning_problem.py#L1362)。

输入签名：

```python
def source_time_map(self):
```

实现说明：

> Return source seconds and progress; no phase-count clock is invented.

显式输出分支：

```python
{'source_times': self.source_timestamps.copy(), 'progress': self.source_progress.copy()}
```

读取的对象属性：

```python
self.source_timestamps
self.source_timestamps.copy
self.source_progress.copy
self.source_progress
```

写入目标：

无显式条目。

直接调用符号：

```python
ValueError
self.source_timestamps.copy
self.source_progress.copy
```

## `PlanningProblem.source_time_to_progress`

[定义位置](../../../manipulation/planning_problem.py#L1368)。

输入签名：

```python
def source_time_to_progress(self, source_time):
```

显式输出分支：

```python
np.interp(source_time, mapping['source_times'], mapping['progress'])
```

读取的对象属性：

```python
self.source_time_map
np.interp
```

写入目标：

```python
mapping
```

直接调用符号：

```python
np.interp
```

## `PlanningProblem.progress_to_source_time`

[定义位置](../../../manipulation/planning_problem.py#L1372)。

输入签名：

```python
def progress_to_source_time(self, progress):
```

显式输出分支：

```python
np.interp(progress, mapping['progress'], mapping['source_times'])
```

读取的对象属性：

```python
self.source_time_map
np.interp
```

写入目标：

```python
mapping
```

直接调用符号：

```python
np.interp
```

## `PlanningProblem.terminal_pose_target`

[定义位置](../../../manipulation/planning_problem.py#L1377)。

输入签名：

```python
def terminal_pose_target(self):
```

实现说明：

> A true complete terminal pose goal for native H=1 goal consumers.

显式输出分支：

```python
next((goal for goal in self.pose_events if goal.progress == self.end_progress), None)
```

读取的对象属性：

```python
self.pose_events
goal.progress
self.end_progress
```

写入目标：

```python
goal
```

直接调用符号：

```python
next
```

## `PlanningProblem.anchors`

[定义位置](../../../manipulation/planning_problem.py#L1382)。

输入签名：

```python
def anchors(self):
```

实现说明：

> Exact evaluation sites, never fixed IK states or mandatory robot poses.

显式输出分支：

```python
anchors[(anchors >= self.start_progress) & (anchors <= self.end_progress)]
```

读取的对象属性：

```python
np.unique
self.start_progress
self.end_progress
e.progress
self.events
r.progress
self.supports
self.interiors
r.start
r.end
self.pose_events
self.coordinate_events
self.mechanisms
r.directions
d.start
d.end
self.pose_relations
r.reference.progress
r.reference
```

写入目标：

```python
anchors
e
r
s
d
```

直接调用符号：

```python
np.unique
float
```

## `PlanningProblem.stop_progress`

[定义位置](../../../manipulation/planning_problem.py#L1396)。

输入签名：

```python
def stop_progress(self):
```

显式输出分支：

```python
np.array([event.progress for event in self.events if event.stop and event.progress >= self.start_progress], float)
```

读取的对象属性：

```python
np.array
event.progress
self.events
event.stop
self.start_progress
```

写入目标：

```python
event
```

直接调用符号：

```python
np.array
```

## `PlanningProblem.sample_reference`

[定义位置](../../../manipulation/planning_problem.py#L1399)。

输入签名：

```python
def sample_reference(self, progress):
```

显式输出分支：

```python
(position, quaternion)
```

读取的对象属性：

```python
self.phases[0].reference.sample
self.phases[0].reference
self.phases
progress.clamp
phase.reference.sample
phase.reference
(progress - index).clamp
torch.where
```

写入目标：

```python
position
quaternion
index
phase
p
q
selected
```

直接调用符号：

```python
self.phases[0].reference.sample
progress.clamp
enumerate
phase.reference.sample
(progress - index).clamp
torch.where
```

## `PlanningProblem.transform_tool_pose`

[定义位置](../../../manipulation/planning_problem.py#L1408)。

输入签名：

```python
def transform_tool_pose(self, position, quaternion):
```

显式输出分支：

```python
(tool_p + quaternion_rotate(tool_q, _tensor(self.tool_transform[:3, 3], position).expand_as(position)), quaternion_multiply(tool_q, _tensor(_quaternion(self.tool_transform), quaternion)))
```

读取的对象属性：

```python
self.base_pose
_tensor(self.tool_transform[:3, 3], position).expand_as
self.tool_transform
```

写入目标：

```python
bp
bq
tool_q
tool_p
```

直接调用符号：

```python
_tensor
_quaternion
quaternion_multiply
quaternion_rotate
_tensor(self.tool_transform[:3, 3], position).expand_as
```

## `PlanningProblem.reference_to_link_pose`

[定义位置](../../../manipulation/planning_problem.py#L1415)。

输入签名：

```python
def reference_to_link_pose(self, position, quaternion):
```

显式输出分支：

```python
(_tensor(inverse_base[:3, 3], position) + quaternion_rotate(bq, link_position), quaternion_multiply(bq, link_quaternion))
```

读取的对象属性：

```python
self.tool_transform
self.base_pose
_tensor(inverse_tool[:3, 3], position).expand_as
```

写入目标：

```python
inverse_tool
inverse_base
link_position
link_quaternion
bq
```

直接调用符号：

```python
_inverse
quaternion_rotate
_tensor(inverse_tool[:3, 3], position).expand_as
_tensor
quaternion_multiply
_quaternion
```

## `PlanningProblem.transform_robot_spheres`

[定义位置](../../../manipulation/planning_problem.py#L1423)。

输入签名：

```python
def transform_robot_spheres(self, spheres):
```

显式输出分支：

```python
torch.cat((centers + _tensor(self.base_pose[:3, 3], spheres), spheres[..., 3:]), -1)
```

读取的对象属性：

```python
self.base_pose
torch.cat
```

写入目标：

```python
centers
```

直接调用符号：

```python
quaternion_rotate
_tensor
_quaternion
torch.cat
```

## `PlanningProblem.transform_tool_velocity`

[定义位置](../../../manipulation/planning_problem.py#L1429)。

输入签名：

```python
def transform_tool_velocity(self, linear, angular, link_quaternion):
```

实现说明：

> Transform native base-frame link twist to reference-frame tool twist.

显式输出分支：

```python
(world_linear + torch.linalg.cross(world_angular, offset), world_angular)
```

读取的对象属性：

```python
self.base_pose
_tensor(self.tool_transform[:3, 3], linear).expand_as
self.tool_transform
torch.linalg.cross
torch.linalg
```

写入目标：

```python
base
world_linear
world_angular
tool_orientation
offset
```

直接调用符号：

```python
_tensor
_quaternion
quaternion_rotate
quaternion_multiply
_tensor(self.tool_transform[:3, 3], linear).expand_as
torch.linalg.cross
```

## `PlanningProblem.residuals`

[定义位置](../../../manipulation/planning_problem.py#L1439)。

输入签名：

```python
def residuals(self, position, quaternion, progress, joint_position=None, joint_names=None, joint_velocity=None, *, trajectory_time=None, point_goal_ids=None, tool_linear_velocity=None, tool_angular_velocity=None, direction_mode='trajectory'):
```

实现说明：

> Nonnegative hard violations for the declared point or trajectory scope.

显式输出分支：

```python
result
```

读取的对象属性：

```python
position.shape
progress.shape
quaternion.shape
torch.relu
self.start_progress
self.end_progress
self.numerical.position_m
self.numerical
self.numerical.rotation_rad
self.pose_relations
requirement.error
requirement.pose_bounds
(torch.relu(bounds[:, 0] - error - tolerance) + torch.relu(error - bounds[:, 1] - tolerance)).sum
requirement.start
requirement.end
torch.where
self.mechanisms
requirement.model
requirement.errors
model.position_tolerance
model.rotation_tolerance
requirement.coordinate_limits
requirement.coordinates
requirement.angular_column
torch.ceil
np.pi
torch.minimum
torch.remainder
excess.clone
requirement.history_complete
excess.sum
requirement.direction_residuals(position, quaternion, progress, trajectory_time=trajectory_time, tool_linear_velocity=tool_linear_velocity, tool_angular_velocity=tool_angular_velocity, mode=direction_mode).items
requirement.direction_residuals
self.supports
mask.to(torch.int64).argmax
mask.to
torch.int64
position.gather
selected[..., None].expand
selected.shape
quaternion.gather
requirement.evaluate
mask.any
torch.zeros_like(progress, dtype=value.dtype).scatter
torch.zeros_like
value.dtype
self.interiors
self.pose_events
self.coordinate_events
self.couplings
tuple(joint_names).index
coupling.joint_names
coupling.matrix.T
coupling.matrix
coupling.target
torch.relu(error.abs() - self.numerical.joint).sum
error.abs
self.numerical.joint
self.events
event.stop
event.skill_id
event.event_id
event.progress
torch.relu(joint_velocity.abs() - self.numerical.velocity).sum
joint_velocity.abs
self.numerical.velocity
```

写入目标：

```python
result
selected_goals
tolerance
index
requirement
error
bounds
violation
mask
result[f'pose_relation:{index}']
model
p
r
result[f'mechanism:{index}']
coordinates
excess
angular
angle
low
high
branch
allowed
local_excess
excess[..., angular]
result[f'mechanism_limits:{index}']
name
value
result[f'mechanism:{index}:{name}']
selected
event_position
event_quaternion
result[f'support:{index}']
result[f'interior:{index}']
result[f'function_pose:{index}']
result[f'function_coordinate:{index}']
coupling
columns
column_index
result[f'coupling:{index}']
event
result[f'stop:{event.skill_id}:{event.event_id}']
```

直接调用符号：

```python
ValueError
torch.relu
set
_tensor
enumerate
requirement.error
(torch.relu(bounds[:, 0] - error - tolerance) + torch.relu(error - bounds[:, 1] - tolerance)).sum
torch.where
requirement.errors
requirement.coordinates
torch.ceil
torch.minimum
torch.remainder
excess.clone
requirement.history_complete
excess.sum
requirement.direction_residuals(position, quaternion, progress, trajectory_time=trajectory_time, tool_linear_velocity=tool_linear_velocity, tool_angular_velocity=tool_angular_velocity, mode=direction_mode).items
requirement.direction_residuals
event_mask
mask.to(torch.int64).argmax
mask.to
position.gather
selected[..., None].expand
quaternion.gather
requirement.evaluate
mask.any
torch.zeros_like(progress, dtype=value.dtype).scatter
torch.zeros_like
tuple(joint_names).index
tuple
constant_tensor
torch.relu(error.abs() - self.numerical.joint).sum
error.abs
_first_event_query
torch.relu(joint_velocity.abs() - self.numerical.velocity).sum
joint_velocity.abs
```

## `PlanningProblem.residuals.event_mask`

[定义位置](../../../manipulation/planning_problem.py#L1448)。

输入签名：

```python
def event_mask(collection, index, requirement):
```

显式输出分支：

```python
_first_event_query(progress, requirement.progress, trajectory_time) if selected_goals is None or (collection, index) in selected_goals else torch.zeros_like(progress, dtype=torch.bool)
```

读取的对象属性：

```python
requirement.progress
torch.zeros_like
torch.bool
```

写入目标：

无显式条目。

直接调用符号：

```python
_first_event_query
torch.zeros_like
```

## `PlanningProblem.evaluate`

[定义位置](../../../manipulation/planning_problem.py#L1534)。

输入签名：

```python
def evaluate(self, position, quaternion, progress, joint_position=None, joint_names=None, joint_velocity=None, *, trajectory_time=None, point_goal_ids=None, tool_linear_velocity=None, tool_angular_velocity=None, direction_mode='trajectory'):
```

显式输出分支：

```python
sum(self.residuals(position, quaternion, progress, joint_position, joint_names, joint_velocity, trajectory_time=trajectory_time, point_goal_ids=point_goal_ids, tool_linear_velocity=tool_linear_velocity, tool_angular_velocity=tool_angular_velocity, direction_mode=direction_mode).values())
```

读取的对象属性：

```python
self.residuals(position, quaternion, progress, joint_position, joint_names, joint_velocity, trajectory_time=trajectory_time, point_goal_ids=point_goal_ids, tool_linear_velocity=tool_linear_velocity, tool_angular_velocity=tool_angular_velocity, direction_mode=direction_mode).values
self.residuals
```

写入目标：

无显式条目。

直接调用符号：

```python
sum
self.residuals(position, quaternion, progress, joint_position, joint_names, joint_velocity, trajectory_time=trajectory_time, point_goal_ids=point_goal_ids, tool_linear_velocity=tool_linear_velocity, tool_angular_velocity=tool_angular_velocity, direction_mode=direction_mode).values
self.residuals
```

## `PlanningProblem.functional_constraint_keys`

[定义位置](../../../manipulation/planning_problem.py#L1544)。

输入签名：

```python
def functional_constraint_keys(self):
```

实现说明：

> Stable event/component identities for the current request suffix.

显式输出分支：

```python
tuple([f'function_pose:{index}:{component}' for (index, requirement) in enumerate(self.pose_events) if requirement.progress >= self.start_progress for component in requirement.component_tolerances] + [f'function_coordinate:{index}' for (index, requirement) in enumerate(self.coordinate_events) if requirement.progress >= self.start_progress] + [f'support:{index}:{component}' for (index, requirement) in enumerate(self.supports) if requirement.progress >= self.start_progress for component in requirement.hard_components])
```

读取的对象属性：

```python
self.pose_events
requirement.progress
self.start_progress
requirement.component_tolerances
self.coordinate_events
self.supports
requirement.hard_components
```

写入目标：

```python
index
requirement
component
```

直接调用符号：

```python
tuple
enumerate
```

## `PlanningProblem.functional_inequalities`

[定义位置](../../../manipulation/planning_problem.py#L1561)。

输入签名：

```python
def functional_inequalities(self, position, quaternion, progress, *, trajectory_time=None):
```

实现说明：

> Return signed event residuals and their single physical query masks.
>
> For input progress ``[batch, horizon]``, residual values have shape
> ``[batch]`` and Boolean masks have shape ``[batch, horizon]``. A pose
> event supplies position and orientation inequalities, plus a yaw row
> when a declared axis has yaw_tolerance_rad < pi. Orientation means
> full SO(3) angle without an axis and remaining swing with an axis;
> a coordinate event uses its existing signed/unwrapped history before
> taking the absolute error. Those values are
> ``error/original_tolerance-1`` for positive bounds. Exact zero yaw uses
> ``error/numerical.rotation_rad``; this scale preserves the zero-width
> allowed set. A support event supplies its declared hard physical
> inequalities divided by positive scales; their zero sets remain the
> original finite-support requirements.
>
> The first actual event instant is used even when progress pauses or an
> exact anchor duplicates a regular sample. A missing required event is
> NaN, not a falsely feasible zero. Events before a continued request's
> start are absent from both dictionaries and retain their original IDs.

显式输出分支：

```python
(signed, masks)
```

读取的对象属性：

```python
position.shape
progress.shape
quaternion.shape
self.pose_events
requirement.progress
self.start_progress
requirement.error_components
requirement.component_tolerances.items
requirement.component_tolerances
self.coordinate_events
requirement.error
requirement.tolerance
self.supports
mask.to(torch.int64).argmax
mask.to
torch.int64
position.gather
selected[..., None].expand
selected.shape
quaternion.gather
requirement.normalized_inequalities
self.numerical
requirement.hard_components
torch.where
mask.any
value.squeeze
```

写入目标：

```python
signed
masks
index
requirement
mask
errors
name
bound
error
selected
event_position
event_quaternion
components
component
value
key
signed[key]
masks[key]
```

直接调用符号：

```python
ValueError
enumerate
_first_event_query
requirement.error_components
requirement.component_tolerances.items
add
requirement.error
mask.to(torch.int64).argmax
mask.to
position.gather
selected[..., None].expand
quaternion.gather
requirement.normalized_inequalities
torch.where
mask.any
value.squeeze
float
```

## `PlanningProblem.functional_inequalities.add`

[定义位置](../../../manipulation/planning_problem.py#L1587)。

输入签名：

```python
def add(key, error, tolerance, mask):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.numerical.rotation_rad
self.numerical
torch.where(mask, (error - tolerance) / scale, 0.0).sum
torch.where
mask.any
```

写入目标：

```python
scale
value
signed[key]
masks[key]
```

直接调用符号：

```python
torch.where(mask, (error - tolerance) / scale, 0.0).sum
torch.where
mask.any
float
```

## `PlanningProblem.optimizer_preferences`

[定义位置](../../../manipulation/planning_problem.py#L1629)。

输入签名：

```python
def optimizer_preferences(self, position, quaternion, progress, *, trajectory_time=None, tool_linear_velocity=None, tool_angular_velocity=None, phase='quality'):
```

实现说明：

> Guide process relations and optional quality-stage goal centering.
>
> Functional feasibility is supplied only by the signed inequality AL
> terms. ``quality`` additionally evaluates soft support components and
> prefers the original goal center inside the same enforced tolerance.
> Process equations and directional commands
> retain their existing semantics in both phases; this method never
> changes the independently evaluated acceptance residuals.

显式输出分支：

```python
result
```

读取的对象属性：

```python
torch.zeros_like
self.supports
requirement.progress
mask.to(torch.int64).argmax
mask.to
torch.int64
position.gather
selected[..., None].expand
selected.shape
quaternion.gather
requirement.normalized_inequalities
self.numerical
torch.relu(value).square
torch.relu
components.items
requirement.hard_components
torch.where
mask.any
torch.zeros_like(progress).scatter
self.pose_relations
np.isfinite(requirement.pose_bounds).all
np.isfinite
requirement.pose_bounds
exact.any
np.flatnonzero
torch.long
requirement.error
(error - _tensor(requirement.pose_bounds[exact, 0], position)).square().sum
(error - _tensor(requirement.pose_bounds[exact, 0], position)).square
requirement.start
requirement.end
self.pose_events
requirement.error_components
(errors[name] / (bound if bound > 0 else self.numerical.rotation_rad)).square
self.numerical.rotation_rad
requirement.component_tolerances.items
requirement.component_tolerances
self.coordinate_events
(error / requirement.tolerance).square
requirement.tolerance
self.mechanisms
requirement.direction_residuals
requirement.directions
values.values
direction.tolerance
direction.minimum
direction.coordinate_index
requirement.angular_column
requirement.model.rotation_tolerance
requirement.model
requirement.model.position_tolerance
direction.kind
trajectory_time.amax
trajectory_time.amin
duration.clamp_min
torch.finfo(duration.dtype).eps
torch.finfo
duration.dtype
(residual / scale).square
```

写入目标：

```python
result
requirement
mask
selected
event_position
event_quaternion
components
preference
name
value
exact
exact_columns
error
errors
bound
values
direction
residual
scale
duration
```

直接调用符号：

```python
ValueError
torch.zeros_like
_first_event_query
mask.to(torch.int64).argmax
mask.to
position.gather
selected[..., None].expand
quaternion.gather
requirement.normalized_inequalities
sum
torch.relu(value).square
torch.relu
components.items
torch.where
mask.any
torch.zeros_like(progress).scatter
np.isfinite(requirement.pose_bounds).all
np.isfinite
exact.any
constant_tensor
np.flatnonzero
requirement.error
(error - _tensor(requirement.pose_bounds[exact, 0], position)).square().sum
(error - _tensor(requirement.pose_bounds[exact, 0], position)).square
_tensor
requirement.error_components
(errors[name] / (bound if bound > 0 else self.numerical.rotation_rad)).square
requirement.component_tolerances.items
(error / requirement.tolerance).square
requirement.direction_residuals
zip
values.values
trajectory_time.amax
trajectory_time.amin
duration.clamp_min
torch.finfo
(residual / scale).square
```

## `PlanningProblem.measurements`

[定义位置](../../../manipulation/planning_problem.py#L1713)。

输入签名：

```python
def measurements(self, position, quaternion, progress, *, trajectory_time=None):
```

实现说明：

> Report source deviations separately from independent physical/goal errors.

显式输出分支：

```python
result
```

读取的对象属性：

```python
self.phases
phase.reference.sample
phase.reference
(progress - index).clamp
torch.where
torch.linalg.vector_norm
torch.linalg
error[..., :3].abs().amax
error[..., :3].abs
error[..., 3:].abs().amax
error[..., 3:].abs
self.mechanisms
requirement.errors
requirement.start
requirement.end
self.pose_events
requirement.progress
requirement.error_components(position, quaternion).items
requirement.error_components
self.coordinate_events
requirement.kind
requirement.error
```

写入目标：

```python
result
index
phase
p
q
error
mask
prefix
result[f'{prefix}_position_error_m']
result[f'{prefix}_rotation_error_rad']
result[f'{prefix}_position_component_error_m']
result[f'{prefix}_rotation_component_error_rad']
requirement
r
result[f'mechanism:{index}:position_error_m']
result[f'mechanism:{index}:rotation_error_rad']
name
value
unit
result[f'function:{index}:{name}_error_{unit}']
result[f'coordinate:{index}:error_{unit}']
```

直接调用符号：

```python
enumerate
phase.reference.sample
(progress - index).clamp
_pose_error
torch.where
torch.linalg.vector_norm
error[..., :3].abs().amax
error[..., :3].abs
error[..., 3:].abs().amax
error[..., 3:].abs
requirement.errors
_first_event_query
requirement.error_components(position, quaternion).items
requirement.error_components
requirement.error
```

## `PlanningProblem.verification_issues`

[定义位置](../../../manipulation/planning_problem.py#L1744)。

输入签名：

```python
def verification_issues(self, progress):
```

实现说明：

> Require actual goal/scope queries and the beginning of angular history.
>
> Sampling resolution/continuous interval coverage remains the curve
> verifier's responsibility. A point query alone proves no winding.

显式输出分支：

```python
issues
```

读取的对象属性：

```python
torch.as_tensor
self.anchors
((query - anchor).abs() <= 1e-07).any(-1).all
((query - anchor).abs() <= 1e-07).any
(query - anchor).abs
issues.append
self.mechanisms
requirement.angular_column
requirement.history_complete(query).all
requirement.history_complete
```

写入目标：

```python
query
issues
anchor
index
requirement
```

直接调用符号：

```python
torch.as_tensor
((query - anchor).abs() <= 1e-07).any(-1).all
((query - anchor).abs() <= 1e-07).any
(query - anchor).abs
issues.append
enumerate
requirement.history_complete(query).all
requirement.history_complete
```

## `PlanningProblem.remaining_obligations`

[定义位置](../../../manipulation/planning_problem.py#L1762)。

输入签名：

```python
def remaining_obligations(self, progress, applied_ids=()):
```

实现说明：

> Read a suffix of this binding; this does not construct a resumed plan.
>
> ``applied_ids`` contains (skill_id, event_id, occurrence) triples. An unapplied command
> before ``progress`` remains visible as overdue; elapsed time is not
> evidence that it was applied. Scope intersections retain their original
> requirement objects so target paths are never silently reparameterized.
> Ordinary continuation retains the frozen clock. Explicit rebinding may
> rebuild the schedule while retaining declared absolute time relations.

显式输出分支：

```python
{'progress': float(progress), 'original_phase_count': self.phase_count, 'phases': tuple(((index, deepcopy(phase)) for (index, phase) in enumerate(self.phases) if index + 1 >= progress)), 'events': tuple(deepcopy(pending)), 'overdue_event_ids': tuple((event_occurrence_key(event) for event in pending if event.progress < progress and event.kind in {'open', 'close', 'hold'})), 'goals': {name: tuple((deepcopy(requirement) for requirement in getattr(self, name) if requirement.progress >= progress)) for name in ('pose_events', 'coordinate_events', 'supports')}, 'past_goal_assessment_required': tuple(((name, index) for name in ('pose_events', 'coordinate_events', 'supports') for (index, requirement) in enumerate(getattr(self, name)) if requirement.progress < progress)), 'process': scoped, 'conditions': deepcopy(self.conditions), 'source_time_map': self.source_time_map, 'required_actual_inputs': tuple(actual_inputs), 'clock_policy': 'ordinary continuation uses elapsed+t; explicit rebinding preserves declared physical-time relations'}
```

读取的对象属性：

```python
np.isfinite
self.phase_count
applied.add
self.events
requirement.start
requirement.end
self.mechanisms
requirement.angular_column
actual_inputs.append
phase.metadata
self.phases
phase.metadata.get
', '.join
event.progress
event.kind
requirement.progress
self.conditions
self.source_time_map
```

写入目标：

```python
applied
identity
value
pending
event
scoped
name
scoped[name]
requirement
actual_inputs
index
owners
phase
```

直接调用符号：

```python
np.isfinite
ValueError
set
isinstance
len
any
type
applied.add
tuple
event_occurrence_key
deepcopy
max
float
getattr
enumerate
actual_inputs.append
sorted
phase.metadata.get
', '.join
```

## `PlanningProblem.projection_requirements`

[定义位置](../../../manipulation/planning_problem.py#L1816)。

输入签名：

```python
def projection_requirements(self, progress, side='at'):
```

实现说明：

> Select actual equations in a scope; a boundary can be queried one-sided.

显式输出分支：

```python
tuple(selected) + tuple((('couplings', index) for index in range(len(self.couplings))))
```

读取的对象属性：

```python
np.isfinite
requirement.start
requirement.end
requirement.pose_bounds
(np.isfinite(bounds[:, 0]) & (bounds[:, 0] == bounds[:, 1])).any
selected.append
self.couplings
```

写入目标：

```python
selected
collection
index
requirement
active
bounds
```

直接调用符号：

```python
np.isfinite
ValueError
enumerate
getattr
bool
(np.isfinite(bounds[:, 0]) & (bounds[:, 0] == bounds[:, 1])).any
selected.append
tuple
range
len
```

## `PlanningProblem.goal_requirements_at`

[定义位置](../../../manipulation/planning_problem.py#L1833)。

输入签名：

```python
def goal_requirements_at(self, progress):
```

实现说明：

> Identify original true goals at this progress, never reference endpoints.

显式输出分支：

```python
tuple(((name, index) for name in ('pose_events', 'coordinate_events', 'supports') for (index, goal) in enumerate(getattr(self, name)) if abs(goal.progress - progress) <= 1e-07))
```

读取的对象属性：

```python
goal.progress
```

写入目标：

```python
name
index
goal
```

直接调用符号：

```python
tuple
enumerate
getattr
abs
```

## `PlanningProblem.goal_residual`

[定义位置](../../../manipulation/planning_problem.py#L1838)。

输入签名：

```python
def goal_residual(self, position, quaternion, progress, *, goal_ids, normalized=False):
```

实现说明：

> Differentiable violations of allowed goal sets for initialization only.
>
> ``normalized`` divides each physical excess by its declaration's
> positive conditioning scale for event IK. Both forms have exactly the
> same allowed set. Winding acceptance still requires the complete
> ordered candidate path; a local query has no winding history.

显式输出分支：

```python
torch.cat(pieces, -1) if pieces else position[..., :0]
```

读取的对象属性：

```python
goal.error_components
pieces.extend
torch.relu
self.numerical.rotation_rad
self.numerical
goal.component_tolerances.items
goal.component_tolerances
pieces.append
goal.evaluate
goal.tolerance
goal.normalized_inequalities
goal.raw_inequalities
goal.hard_components
torch.cat
```

写入目标：

```python
pieces
collection
index
goal
values
name
bound
```

直接调用符号：

```python
ValueError
getattr
goal.error_components
pieces.extend
torch.relu
goal.component_tolerances.items
pieces.append
goal.evaluate
goal.normalized_inequalities
goal.raw_inequalities
torch.cat
```

## `PlanningProblem.project_residual`

[定义位置](../../../manipulation/planning_problem.py#L1870)。

输入签名：

```python
def project_residual(self, position, quaternion, progress, phase_index=None, joint_position=None, joint_names=None, active_requirements=None):
```

实现说明：

> Real equality/manifold equations, never source-stage or band centers.

显式输出分支：

```python
torch.cat(pieces, -1) if pieces else position[..., :0]
```

读取的对象属性：

```python
self.pose_relations
np.isfinite
requirement.pose_bounds
exact.any
np.flatnonzero
torch.long
requirement.error
requirement.start
requirement.end
pieces.append
torch.where
self.mechanisms
requirement.project_residual
self.couplings
tuple(joint_names).index
coupling.joint_names
torch.int64
coupling.matrix.T
coupling.matrix
coupling.target
torch.cat
```

写入目标：

```python
pieces
selected
name
index
requirement
exact
exact_columns
error
mask
coupling
columns
```

直接调用符号：

```python
set
any
len
getattr
ValueError
enumerate
np.isfinite
exact.any
constant_tensor
np.flatnonzero
requirement.error
_tensor
pieces.append
torch.where
requirement.project_residual
tuple(joint_names).index
tuple
torch.cat
```

## `_path_from_rows`

[定义位置](../../../manipulation/planning_problem.py#L1914)。

输入签名：

```python
def _path_from_rows(poses, source, *, timestamps, model=None, coordinates=None, attachment=None):
```

显式输出分支：

```python
PosePath(progress, poses, source, model, coordinates, np.eye(4) if attachment is None else attachment)
```

读取的对象属性：

```python
np.asarray
times.shape
np.isfinite(times).all
np.isfinite
np.any
np.diff
np.array
np.eye
```

写入目标：

```python
times
progress
```

直接调用符号：

```python
np.asarray
len
np.isfinite(times).all
np.isfinite
np.any
np.diff
ValueError
np.array
PosePath
np.eye
```

## `_model_from_diagnostics`

[定义位置](../../../manipulation/planning_problem.py#L1924)。

输入签名：

```python
def _model_from_diagnostics(value):
```

显式输出分支：

```python
None
MotionModel(value['kind'], np.asarray(value['coordinates'], float), np.asarray(value['reference_pose'], float), np.asarray(value['basis'], float), None if value['axis_point'] is None else np.asarray(value['axis_point'], float), value['position_tolerance'], value['rotation_tolerance'], value['diagnostics'])
```

读取的对象属性：

```python
np.asarray
```

写入目标：

无显式条目。

直接调用符号：

```python
MotionModel
np.asarray
```

## `_validate_skill_requirements`

[定义位置](../../../manipulation/planning_problem.py#L1936)。

输入签名：

```python
def _validate_skill_requirements(spec):
```

实现说明：

> Check requirement-bearing records while retaining scene evidence metadata.
>
> A spelling error in a declared requirement must not become an inert dict
> entry. Observed scene records and provenance are intentionally not closed
> schemas: geometry consumers validate their own necessary input fields.

显式输出分支：

无显式条目。

读取的对象属性：

```python
demo.get
spec.get
event.get
evidence.strip
item.check_target
item.get
event['kind'].strip
```

写入目标：

```python
program
demo
unsupported
label
record
rejected
common
groups
events
allowed
event
evidence
actual_kind
coordinate_fields
pose_fields
incompatible
activation_timing
item
directions
direction
```

直接调用符号：

```python
fields
set
ValueError
sorted
demo.get
spec.get
isinstance
type
event.get
evidence.strip
any
item.get
event['kind'].strip
```

## `_validate_skill_requirements.fields`

[定义位置](../../../manipulation/planning_problem.py#L1943)。

输入签名：

```python
def fields(record, allowed, label):
```

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
unknown
```

直接调用符号：

```python
isinstance
ValueError
set
sorted
```

## `build_planning_problem`

[定义位置](../../../manipulation/planning_problem.py#L2032)。

输入签名：

```python
def build_planning_problem(skills, *, joint_names, start_q, start_dq, tool_link, reference_frame, base_pose, tool_transform, numerical=NumericalTolerance(), couplings=()):
```

实现说明：

> Bind one current skill's actual entry, independent goals and source map.
>
> Every ordinary source row remains available for initialization/deviation
> reporting. Only explicit functional events, physical models, scoped
> geometry and requested execution events become planning requirements.
> Effects remain to be evaluated; they do not automatically require a stop
> or grant evidence to a future skill.

显式输出分支：

```python
problem
```

读取的对象属性：

```python
spec.get
scene.get
demo.get
frames.tolist
times.tolist
candidate.tolist
pose_events.append
np.eye
conditions.append
events.append
command.get
attachment.tolist
event.get
mechanisms.append
reference_model.coordinates
supports.append
coordinate_events.append
np.asarray
interiors.append
interior.halfspaces
interior.source
objects[-1].tolist
mechanisms[0].start
mechanisms[0].end
directions.append
row.get
mechanisms[0].__post_init__
problem.conditions
problem.events
event.timing
constraint.check_target
constraint.goal_id
constraint.condition_id
```

写入目标：

```python
spec
skill_id
program
scene
demo
frames
times
identity
value
carrier
initial_key
initial
scene[initial_key]
source
grounded
phases
events
conditions
supports
interiors
pose_events
coordinate_events
mechanisms
frame_progress
path
candidate
grasp_goal
commands
index
command
begin
end
later
activation
arrival_source
arrival_id
objects
attachment
attachment_id
last_held
release
held_end
model
limits
mechanism
event
reference_model
coordinates
release_id
progress
binding
check
row
face
material
interior
frame
direction_specs
directions
mechanisms[0].directions
problem
condition_ids
constraint
collection
```

直接调用符号：

```python
NumericalTolerance
isinstance
len
ValueError
_validate_skill_requirements
spec.get
deepcopy
validate_timeline
scene.get
any
demo.get
_pose_array
frames.tolist
times.tolist
ground_program
_path_from_rows
append_interval
candidate.tolist
pose_events.append
PoseEventRequirement
np.eye
conditions.append
events.append
PlanningEvent
int
enumerate
next
command.get
attachment.tolist
_current_model
MechanismRequirement
_inverse
_current_initial_coordinates
bool
event.get
mechanisms.append
_model_from_diagnostics
supports.append
SupportRequirement
_surface
coordinate_events.append
CoordinateEventRequirement
float
np.asarray
_interior
interiors.append
InteriorRequirement
tuple
objects[-1].tolist
directions.append
CoordinateDirectionRequirement
row.get
mechanisms[0].__post_init__
PlanningProblem
getattr
```

## `build_planning_problem.append_interval`

[定义位置](../../../manipulation/planning_problem.py#L2090)。

输入签名：

```python
def append_interval(begin, end, path, metadata):
```

实现说明：

> A mode boundary partitions reference coordinates, never robot q.

显式输出分支：

```python
None
```

读取的对象属性：

```python
phases.append
initial.tolist
frames[begin:end + 1].tolist
path.progress
```

写入目标：

```python
frame_progress[int(frames[begin])]
index
row
frame_progress[int(frames[row])]
```

直接调用符号：

```python
int
float
len
phases.append
MotionPhase
initial.tolist
frames[begin:end + 1].tolist
range
```
