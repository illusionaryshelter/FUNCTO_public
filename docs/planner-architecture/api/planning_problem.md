# `manipulation/planning_problem.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_tensor`

[定义位置](../../../manipulation/planning_problem.py#L24)。

输入签名：

```python
def _tensor(value, like):
```

输出分支的源码表达式：

```python
constant_tensor(value, like)
```

直接调用：

```python
constant_tensor
```

## `quaternion_conjugate`

[定义位置](../../../manipulation/planning_problem.py#L31)。

输入签名：

```python
def quaternion_conjugate(q):
```

输出分支的源码表达式：

```python
q * _tensor([1.0, -1.0, -1.0, -1.0], q)
```

直接调用：

```python
_tensor
```

## `quaternion_multiply`

[定义位置](../../../manipulation/planning_problem.py#L36)。

输入签名：

```python
def quaternion_multiply(a, b):
```

输出分支的源码表达式：

```python
torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True), aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)
```

直接调用：

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

输出分支的源码表达式：

```python
points + 2 * (q[..., :1] * cross + torch.linalg.cross(v, cross))
```

直接调用：

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

输出分支的源码表达式：

```python
q[..., 1:] * (2 * torch.atan2(length, q[..., :1]) / length)
```

直接调用：

```python
torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min
torch.linalg.vector_norm
torch.where
torch.sqrt
(q[..., 1:] ** 2).sum
torch.atan2
```

## `_quaternion`

[定义位置](../../../manipulation/planning_problem.py#L64)。

输入签名：

```python
def _quaternion(pose):
```

输出分支的源码表达式：

```python
Rotation.from_matrix(np.asarray(pose)[..., :3, :3]).as_quat()[..., [3, 0, 1, 2]]
```

直接调用：

```python
Rotation.from_matrix(np.asarray(pose)[..., :3, :3]).as_quat
Rotation.from_matrix
np.asarray
```

## `_slerp`

[定义位置](../../../manipulation/planning_problem.py#L69)。

输入签名：

```python
def _slerp(a, b, fraction):
```

输出分支的源码表达式：

```python
result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1e-15)
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L94)。

输入签名：

```python
def _interpolate(knots, values, progress):
```

输出分支的源码表达式：

```python
data[0].expand(*progress.shape, *data.shape[1:])
data[index - 1] + fraction[(...,) + extra] * (data[index] - data[index - 1])
```

直接调用：

```python
_tensor
len
data[0].expand
torch.searchsorted(keys, progress.contiguous(), right=True).clamp
torch.searchsorted
progress.contiguous
((progress - keys[index - 1]) / (keys[index] - keys[index - 1])).clamp
```

## `PosePath`

[定义位置](../../../manipulation/planning_problem.py#L107)。

```python
class PosePath:
    progress: np.ndarray
    poses: np.ndarray
    source: object
    motion_model: MotionModel | None = None
    coordinates: np.ndarray | None = None
    attachment: np.ndarray = field(default_factory=lambda : np.eye(4))
    parent: PosePath | None = None
```

该类型的直接方法：`__post_init__`、`sample`、`sample_numpy`。

## `PosePath.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L129)。

输入签名：

```python
def PosePath.__post_init__(self):
```

直接读取的对象属性：`self.progress`、`self.poses`、`self.attachment`、`self.source`、`self.parent`、`self.motion_model`、`self.coordinates`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.progress`、`self.poses`、`self.attachment`、`self.coordinates`、`self._quaternions`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L156)。

输入签名：

```python
def PosePath.sample(self, progress):
```

直接读取的对象属性：`self.motion_model`、`self.progress`、`self.poses`、`self._quaternions`、`self.coordinates`、`self.attachment`、`self.parent`。

输出分支的源码表达式：

```python
(position, quaternion)
```

直接调用：

```python
_interpolate
_tensor
len
values[0].expand
torch.searchsorted(keys, progress.contiguous(), right=True).clamp
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

[定义位置](../../../manipulation/planning_problem.py#L199)。

输入签名：

```python
def PosePath.sample_numpy(self, progress):
```

直接读取的对象属性：`self.sample`。

输出分支的源码表达式：

```python
output
```

直接写入的状态或数组位置：`output[..., :3, 3]`、`output[..., :3, :3]`。

直接调用：

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

## `TransferredPosePath`

[定义位置](../../../manipulation/planning_problem.py#L211)。

```python
class TransferredPosePath(PosePath):
    ...
    ...
```

该类型的直接方法：`__init__`、`sample`。

## `TransferredPosePath.__init__`

[定义位置](../../../manipulation/planning_problem.py#L226)。

输入签名：

```python
def TransferredPosePath.__init__(self, reference, initial_pose, goals, source):
```

直接读取的对象属性：`self._interval_starts`、`self.sample_numpy`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._reference`、`self._interval_starts`、`self._interval_spans`、`self._right_positions`、`self._right_quaternions`、`self._translations`、`corrections[:, :3, :3]`、`self._correction_quaternions`。

直接调用：

```python
isinstance
len
ValueError
_pose_array
goal.get
np.isfinite
float
np.allclose
normalized.append
normalized.sort
any
zip
starts.append
ends.append
parameters.append
free_transfer_parameters
reference.sample_numpy
np.asarray
np.stack
_quaternion
np.repeat
np.eye
np.unique
self.sample_numpy
super().__init__
super
```

## `TransferredPosePath.sample`

[定义位置](../../../manipulation/planning_problem.py#L277)。

输入签名：

```python
def TransferredPosePath.sample(self, progress):
```

直接读取的对象属性：`self._interval_starts`、`self._interval_spans`、`self._reference`、`self._right_positions`、`self._right_quaternions`、`self._translations`、`self._correction_quaternions`。

输出分支的源码表达式：

```python
(position, quaternion_multiply(correction, quaternion))
```

直接调用：

```python
_tensor
(torch.searchsorted(keys, progress.contiguous(), right=True) - 1).clamp
torch.searchsorted
progress.contiguous
len
((progress - keys[index]) / _tensor(self._interval_spans, progress)[index]).clamp
self._reference.sample
quaternion_rotate
quaternion_multiply
_tensor([1.0, 0.0, 0.0, 0.0], progress).expand
_slerp
```

## `NumericalTolerance`

[定义位置](../../../manipulation/planning_problem.py#L295)。

```python
class NumericalTolerance:
    position_m: float = 0.0001
    rotation_rad: float = 0.001
    joint: float = 1e-05
    velocity: float = 0.0001
```

该类型的直接方法：`__post_init__`。

## `NumericalTolerance.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L308)。

输入签名：

```python
def NumericalTolerance.__post_init__(self):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
any
np.isfinite
vars(self).values
vars
ValueError
```

## `MotionPhase`

[定义位置](../../../manipulation/planning_problem.py#L315)。

```python
class MotionPhase:
    name: str
    reference: PosePath
    metadata: dict = field(default_factory=dict)
```

该类型的直接方法：`__post_init__`。

## `MotionPhase.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L321)。

输入签名：

```python
def MotionPhase.__post_init__(self):
```

直接读取的对象属性：`self.name`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
isinstance
ValueError
```

## `reference_phase`

[定义位置](../../../manipulation/planning_problem.py#L327)。

输入签名：

```python
def reference_phase(reference, *, skill_id, source):
```

输出分支的源码表达式：

```python
MotionPhase(skill_id, path, metadata={'skill_id': skill_id, 'reference_source': deepcopy(source), 'reference_timestamps': times.copy(), 'source_frame_indices': frames, 'source_timestamps': original_times, 'source_events': deepcopy(reference.get('events', [])), 'scope': 'Geometric reference only; task requirements and scene permissions are independent'})
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L375)。

```python
class PoseProcessRequirement:
    start: float
    end: float
    reference: PosePath
    pose_bounds: np.ndarray
    source: object
```

该类型的直接方法：`__post_init__`、`error`。

## `PoseProcessRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L389)。

输入签名：

```python
def PoseProcessRequirement.__post_init__(self):
```

直接读取的对象属性：`self.pose_bounds`、`self.source`、`self.start`、`self.end`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.pose_bounds`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L398)。

输入签名：

```python
def PoseProcessRequirement.error(self, position, quaternion, progress):
```

直接读取的对象属性：`self.start`、`self.end`、`self.reference`。

输出分支的源码表达式：

```python
_pose_error(position, quaternion, *self.reference.sample(local))
```

直接调用：

```python
((progress - self.start) / (self.end - self.start)).clamp
_pose_error
self.reference.sample
```

## `TimeBindingError`

[定义位置](../../../manipulation/planning_problem.py#L405)。

```python
class TimeBindingError(ValueError):
    ...
    ...
```

## `event_occurrence_key`

[定义位置](../../../manipulation/planning_problem.py#L409)。

输入签名：

```python
def event_occurrence_key(event):
```

输出分支的源码表达式：

```python
identity
```

直接调用：

```python
isinstance
get
any
type
TimeBindingError
```

## `EventTimeConstraint`

[定义位置](../../../manipulation/planning_problem.py#L420)。

```python
class EventTimeConstraint:
    check_target: str
    window_s: tuple[float | None, float | None]
    time_basis: str
    source: object
    origin_time_s: float | None = None
    origin_event: dict | None = None
    goal_id: tuple[str, int] | None = None
    condition_id: str | None = None
```

该类型的直接方法：`__post_init__`。

## `EventTimeConstraint.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L436)。

输入签名：

```python
def EventTimeConstraint.__post_init__(self):
```

直接读取的对象属性：`self.window_s`、`self.check_target`、`self.time_basis`、`self.source`、`self.origin_time_s`、`self.origin_event`、`self.goal_id`、`self.condition_id`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L464)。

输入签名：

```python
def resolve_event_window(constraint, history, *, predicted_origins=()):
```

输出分支的源码表达式：

```python
tuple((None if value is None else float(origin) + value for value in constraint.window_s))
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L478)。

输入签名：

```python
def resolve_event_window.selected(row):
```

输出分支的源码表达式：

```python
event_occurrence_key(row) == key and row.get('check_target') == target and ('goal_id' not in constraint.origin_event or tuple(row.get('goal_id') or ()) == tuple(constraint.origin_event['goal_id'])) and ('condition_id' not in constraint.origin_event or row.get('condition_id') == constraint.origin_event['condition_id'])
```

直接调用：

```python
event_occurrence_key
row.get
tuple
```

## `PlanningEvent`

[定义位置](../../../manipulation/planning_problem.py#L500)。

```python
class PlanningEvent:
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

该类型的直接方法：`__post_init__`。

## `PlanningEvent.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L518)。

输入签名：

```python
def PlanningEvent.__post_init__(self):
```

直接读取的对象属性：`self.timing`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
event_occurrence_key
object.__setattr__
tuple
isinstance
EventTimeConstraint
```

## `JointCoupling`

[定义位置](../../../manipulation/planning_problem.py#L527)。

```python
class JointCoupling:
    joint_names: tuple[str, ...]
    matrix: np.ndarray
    target: np.ndarray
    source: object
```

该类型的直接方法：`__post_init__`。

## `JointCoupling.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L534)。

输入签名：

```python
def JointCoupling.__post_init__(self):
```

直接读取的对象属性：`self.matrix`、`self.target`、`self.joint_names`、`self.source`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.matrix`、`self.target`。

直接调用：

```python
np.asarray
len
np.isfinite(self.matrix).all
np.isfinite
np.isfinite(self.target).all
ValueError
```

## `_pose_error`

[定义位置](../../../manipulation/planning_problem.py#L541)。

输入签名：

```python
def _pose_error(position, quaternion, target_position, target_quaternion):
```

输出分支的源码表达式：

```python
torch.cat((quaternion_rotate(inverse, position - target_position), quaternion_rotvec(quaternion_multiply(inverse, quaternion))), -1)
```

直接调用：

```python
quaternion_conjugate
torch.cat
quaternion_rotate
quaternion_rotvec
quaternion_multiply
```

## `_first_event_query`

[定义位置](../../../manipulation/planning_problem.py#L549)。

输入签名：

```python
def _first_event_query(progress, anchor, trajectory_time=None):
```

输出分支的源码表达式：

```python
matches & (indices == first)
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L569)。

输入签名：

```python
def anchor_pose(position, quaternion, progress, anchor):
```

输出分支的源码表达式：

```python
(p, q)
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L592)。

输入签名：

```python
def _compose(p, q, transform):
```

输出分支的源码表达式：

```python
(p + quaternion_rotate(q, t), quaternion_multiply(q, _tensor(_quaternion(transform), q)))
```

直接调用：

```python
_tensor(transform[:3, 3], p).expand_as
_tensor
quaternion_rotate
quaternion_multiply
_quaternion
```

## `_mechanism_errors`

[定义位置](../../../manipulation/planning_problem.py#L598)。

输入签名：

```python
def _mechanism_errors(model, tool_from_carrier, position, quaternion):
```

输出分支的源码表达式：

```python
(torch.linalg.vector_norm(error[..., :3], dim=-1), torch.linalg.vector_norm(error[..., 3:], dim=-1))
```

直接调用：

```python
_compose
_tensor
_quaternion
base_q.expand_as
quaternion_multiply
quaternion_conjugate
(relative[..., 1:] * axis).sum
w.square
along.square
torch.finfo
torch.atan2
torch.where
torch.zeros_like
torch.ones_like
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

[定义位置](../../../manipulation/planning_problem.py#L644)。

```python
class CoordinateDirectionRequirement:
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

该类型的直接方法：`__post_init__`。

## `CoordinateDirectionRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L662)。

输入签名：

```python
def CoordinateDirectionRequirement.__post_init__(self):
```

直接读取的对象属性：`self.source`、`self.start`、`self.end`、`self.coordinate_index`、`self.kind`、`self.sign`、`self.minimum`、`self.tolerance`、`self.origin_coordinate`、`self.origin_source`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
type
ValueError
np.isfinite
```

## `MechanismRequirement`

[定义位置](../../../manipulation/planning_problem.py#L677)。

```python
class MechanismRequirement:
    start: float
    end: float
    model: MotionModel
    tool_from_carrier: np.ndarray
    coordinate_limits: np.ndarray | None
    initial_coordinates: np.ndarray
    source: object
    directions: tuple[CoordinateDirectionRequirement, ...] = ()
```

该类型的直接方法：`__post_init__`、`angular_column`、`coordinates`、`errors`、`project_residual`、`history_complete`、`coordinate_rates`、`direction_residuals`。

## `MechanismRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L694)。

输入签名：

```python
def MechanismRequirement.__post_init__(self):
```

直接读取的对象属性：`self.directions`、`self.tool_from_carrier`、`self.initial_coordinates`、`self.source`、`self.start`、`self.end`、`self.model`、`self.coordinate_limits`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.directions`、`self.tool_from_carrier`、`self.initial_coordinates`、`self.coordinate_limits`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L717)。

输入签名：

```python
@property
def MechanismRequirement.angular_column(self):
```

直接读取的对象属性：`self.model`。

输出分支的源码表达式：

```python
0 if self.model.kind == 'revolute' else 2 if self.model.kind == 'planar' else None
```

## `MechanismRequirement.coordinates`

[定义位置](../../../manipulation/planning_problem.py#L720)。

输入签名：

```python
def MechanismRequirement.coordinates(self, position, quaternion, progress, *, trajectory_time=None):
```

直接读取的对象属性：`self.tool_from_carrier`、`self.model`、`self.start`、`self.end`、`self.initial_coordinates`、`self.angular_column`。

输出分支的源码表达式：

```python
((p - base_p) * axis).sum(-1, keepdim=True)
planar
angle[..., None] if model.kind == 'revolute' else torch.cat((planar, angle[..., None]), -1)
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L768)。

输入签名：

```python
def MechanismRequirement.errors(self, position, quaternion):
```

直接读取的对象属性：`self.model`、`self.tool_from_carrier`。

输出分支的源码表达式：

```python
_mechanism_errors(self.model, self.tool_from_carrier, position, quaternion)
```

直接调用：

```python
_mechanism_errors
```

## `MechanismRequirement.project_residual`

[定义位置](../../../manipulation/planning_problem.py#L771)。

输入签名：

```python
def MechanismRequirement.project_residual(self, position, quaternion, progress):
```

直接读取的对象属性：`self.tool_from_carrier`、`self.model`、`self.coordinates`、`self.angular_column`。

输出分支的源码表达式：

```python
torch.cat((translation, rotation), -1)
torch.cat((translation, off_axis), -1)
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L795)。

输入签名：

```python
def MechanismRequirement.history_complete(self, progress):
```

直接读取的对象属性：`self.start`。

输出分支的源码表达式：

```python
((progress - self.start).abs() <= 1e-07).any(-1)
```

直接调用：

```python
((progress - self.start).abs() <= 1e-07).any
(progress - self.start).abs
```

## `MechanismRequirement.coordinate_rates`

[定义位置](../../../manipulation/planning_problem.py#L799)。

输入签名：

```python
def MechanismRequirement.coordinate_rates(self, quaternion, tool_linear_velocity, tool_angular_velocity):
```

直接读取的对象属性：`self.tool_from_carrier`、`self.model`。

输出分支的源码表达式：

```python
(linear * axis).sum(-1, keepdim=True)
planar
angular if self.model.kind == 'revolute' else torch.cat((planar, angular), -1)
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L824)。

输入签名：

```python
def MechanismRequirement.direction_residuals(self, position, quaternion, progress, *, trajectory_time, tool_linear_velocity=None, tool_angular_velocity=None, mode='trajectory', use_tolerance=True):
```

直接读取的对象属性：`self.directions`、`self.coordinates`、`self.coordinate_rates`、`self.start`。

输出分支的源码表达式：

```python
{}
result
```

直接写入的状态或数组位置：`result[f'direction:{index}:{direction.kind}']`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L891)。

```python
class SupportRequirement:
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
```

该类型的直接方法：`__post_init__`、`raw_inequalities`、`normalized_inequalities`、`evaluate`。

## `SupportRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L904)。

输入签名：

```python
def SupportRequirement.__post_init__(self):
```

直接读取的对象属性：`self.tool_from_carrier`、`self.body_points`、`self.contact_points`、`self.outward_normal`、`self.surface`、`self.margin`、`self.gap_tolerance`、`self.normal_tolerance`、`self.source`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.tool_from_carrier`、`self.body_points`、`self.contact_points`、`self.outward_normal`。

直接调用：

```python
_pose_array
_points
_unit
_support_arguments
isinstance
ValueError
```

## `SupportRequirement.raw_inequalities`

[定义位置](../../../manipulation/planning_problem.py#L916)。

输入签名：

```python
def SupportRequirement.raw_inequalities(self, position, quaternion, numerical):
```

直接读取的对象属性：`self.tool_from_carrier`、`self.body_points`、`self.contact_points`、`self.surface`、`self.outward_normal`、`self.normal_tolerance`、`self.gap_tolerance`、`self.margin`。

输出分支的源码表达式：

```python
{'gap': gaps.abs().amax(-1) - self.gap_tolerance, 'body': -height.amin(-1) - numerical.position_m, 'footprint': footprint + self.margin - numerical.position_m, 'normal': chord - angular_bound}
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L943)。

输入签名：

```python
def SupportRequirement.normalized_inequalities(self, position, quaternion, numerical):
```

直接读取的对象属性：`self.gap_tolerance`、`self.normal_tolerance`、`self.raw_inequalities`。

输出分支的源码表达式：

```python
{name: value / (normal_scale if name == 'normal' else length_scale) for (name, value) in self.raw_inequalities(position, quaternion, numerical).items()}
```

直接调用：

```python
max
np.sin
min
self.raw_inequalities(position, quaternion, numerical).items
self.raw_inequalities
```

## `SupportRequirement.evaluate`

[定义位置](../../../manipulation/planning_problem.py#L959)。

输入签名：

```python
def SupportRequirement.evaluate(self, position, quaternion, numerical):
```

直接读取的对象属性：`self.raw_inequalities`。

输出分支的源码表达式：

```python
sum((torch.relu(value) for value in self.raw_inequalities(position, quaternion, numerical).values()))
```

直接调用：

```python
sum
torch.relu
self.raw_inequalities(position, quaternion, numerical).values
self.raw_inequalities
```

## `PoseEventRequirement`

[定义位置](../../../manipulation/planning_problem.py#L968)。

```python
class PoseEventRequirement:
    progress: float
    tool_from_function: np.ndarray
    target: np.ndarray
    position_tolerance_m: float
    rotation_tolerance_rad: float
    source: object
    actual_relation: dict | None = None
```

该类型的直接方法：`__post_init__`、`errors`、`evaluate`。

## `PoseEventRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L993)。

输入签名：

```python
def PoseEventRequirement.__post_init__(self):
```

直接读取的对象属性：`self.tool_from_function`、`self.target`、`self.source`、`self.progress`、`self.position_tolerance_m`、`self.rotation_tolerance_rad`、`self.actual_relation`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.tool_from_function`、`self.target`、`relation[pose_key]`、`self.actual_relation`。

直接调用：

```python
_pose_array
np.isfinite
ValueError
isinstance
set
deepcopy
relation[owner_key].strip
relation.get
condition.strip
```

## `PoseEventRequirement.errors`

[定义位置](../../../manipulation/planning_problem.py#L1023)。

输入签名：

```python
def PoseEventRequirement.errors(self, position, quaternion):
```

直接读取的对象属性：`self.tool_from_function`、`self.target`。

输出分支的源码表达式：

```python
(torch.linalg.vector_norm(error[..., :3], dim=-1), torch.linalg.vector_norm(error[..., 3:], dim=-1))
```

直接调用：

```python
_compose
_pose_error
_tensor
_quaternion
torch.linalg.vector_norm
```

## `PoseEventRequirement.evaluate`

[定义位置](../../../manipulation/planning_problem.py#L1029)。

输入签名：

```python
def PoseEventRequirement.evaluate(self, position, quaternion):
```

直接读取的对象属性：`self.errors`、`self.position_tolerance_m`、`self.rotation_tolerance_rad`。

输出分支的源码表达式：

```python
torch.relu(p - self.position_tolerance_m) + torch.relu(r - self.rotation_tolerance_rad)
```

直接调用：

```python
self.errors
torch.relu
```

## `CoordinateEventRequirement`

[定义位置](../../../manipulation/planning_problem.py#L1037)。

```python
class CoordinateEventRequirement:
    progress: float
    mechanism: MechanismRequirement
    target_coordinate: float
    tolerance: float
    source: object
    coordinate_mode: str = 'absolute'
    coordinate_origin: float | None = None
```

该类型的直接方法：`__post_init__`、`absolute_target`、`kind`、`error`、`evaluate`。

## `CoordinateEventRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L1047)。

输入签名：

```python
def CoordinateEventRequirement.__post_init__(self):
```

直接读取的对象属性：`self.mechanism`、`self.source`、`self.progress`、`self.target_coordinate`、`self.tolerance`、`self.coordinate_mode`、`self.coordinate_origin`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
np.isfinite
ValueError
```

## `CoordinateEventRequirement.absolute_target`

[定义位置](../../../manipulation/planning_problem.py#L1059)。

输入签名：

```python
@property
def CoordinateEventRequirement.absolute_target(self):
```

直接读取的对象属性：`self.coordinate_mode`、`self.target_coordinate`、`self.coordinate_origin`。

输出分支的源码表达式：

```python
self.target_coordinate if self.coordinate_mode == 'absolute' else self.coordinate_origin + self.target_coordinate
```

## `CoordinateEventRequirement.kind`

[定义位置](../../../manipulation/planning_problem.py#L1063)。

输入签名：

```python
@property
def CoordinateEventRequirement.kind(self):
```

直接读取的对象属性：`self.mechanism`。

输出分支的源码表达式：

```python
self.mechanism.model.kind
```

## `CoordinateEventRequirement.error`

[定义位置](../../../manipulation/planning_problem.py#L1066)。

输入签名：

```python
def CoordinateEventRequirement.error(self, position, quaternion, progress, *, trajectory_time=None):
```

直接读取的对象属性：`self.mechanism`、`self.absolute_target`、`self.kind`。

输出分支的源码表达式：

```python
difference.abs()
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1075)。

输入签名：

```python
def CoordinateEventRequirement.evaluate(self, position, quaternion, progress, *, trajectory_time=None):
```

直接读取的对象属性：`self.error`、`self.tolerance`。

输出分支的源码表达式：

```python
torch.relu(self.error(position, quaternion, progress, trajectory_time=trajectory_time) - self.tolerance)
```

直接调用：

```python
torch.relu
self.error
```

## `InteriorRequirement`

[定义位置](../../../manipulation/planning_problem.py#L1082)。

```python
class InteriorRequirement:
    start: float
    end: float
    tool_from_carrier: np.ndarray
    body_points: np.ndarray
    halfspaces: np.ndarray
    margin: float
    source: object
```

该类型的直接方法：`__post_init__`、`evaluate`。

## `InteriorRequirement.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L1092)。

输入签名：

```python
def InteriorRequirement.__post_init__(self):
```

直接读取的对象属性：`self.tool_from_carrier`、`self.body_points`、`self.halfspaces`、`self.source`、`self.start`、`self.end`、`self.margin`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.tool_from_carrier`、`self.body_points`、`self.halfspaces`。

直接调用：

```python
_pose_array
_points
InteriorRegion
str
np.isfinite
ValueError
```

## `InteriorRequirement.evaluate`

[定义位置](../../../manipulation/planning_problem.py#L1100)。

输入签名：

```python
def InteriorRequirement.evaluate(self, position, quaternion, numerical):
```

直接读取的对象属性：`self.tool_from_carrier`、`self.body_points`、`self.halfspaces`、`self.margin`。

输出分支的源码表达式：

```python
torch.relu(error - numerical.position_m)
```

直接调用：

```python
_compose
quaternion_rotate
_tensor(self.body_points, p).expand
_tensor
(points @ h[:, :3].T + h[:, 3] + self.margin).amax
torch.relu
```

## `PlanningProblem`

[定义位置](../../../manipulation/planning_problem.py#L1111)。

```python
class PlanningProblem:
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

该类型的直接方法：`__post_init__`、`phase_count`、`bound_motion_block`、`requirement_consumption`、`source_time_map`、`source_time_to_progress`、`progress_to_source_time`、`terminal_pose_target`、`anchors`、`stop_progress`、`sample_reference`、`transform_tool_pose`、`reference_to_link_pose`、`transform_robot_spheres`、`transform_tool_velocity`、`residuals`、`evaluate`、`functional_constraint_keys`、`functional_inequalities`、`optimizer_preferences`、`measurements`、`verification_issues`、`remaining_obligations`、`projection_requirements`、`goal_requirements_at`、`goal_residual`、`project_residual`。

## `PlanningProblem.__post_init__`

[定义位置](../../../manipulation/planning_problem.py#L1144)。

输入签名：

```python
def PlanningProblem.__post_init__(self):
```

直接读取的对象属性：`self.joint_names`、`self.start_q`、`self.start_dq`、`self.phases`、`self.tool_link`、`self.reference_frame`、`self.base_pose`、`self.tool_transform`、`self.end_progress`、`self.phase_count`、`self.start_progress`、`self.events`、`self.pose_relations`、`self.mechanisms`、`self.interiors`、`self.pose_events`、`self.coordinate_events`、`self.supports`、`self.source_timestamps`、`self.source_progress`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.joint_names`、`self.start_q`、`self.start_dq`、`self.base_pose`、`self.tool_transform`、`self.end_progress`、`self.source_timestamps`、`self.source_progress`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1199)。

输入签名：

```python
@property
def PlanningProblem.phase_count(self):
```

直接读取的对象属性：`self.phases`。

输出分支的源码表达式：

```python
len(self.phases)
```

直接调用：

```python
len
```

## `PlanningProblem.bound_motion_block`

[定义位置](../../../manipulation/planning_problem.py#L1203)。

输入签名：

```python
def PlanningProblem.bound_motion_block(self, goal_identities=None, satisfied_dependencies=()):
```

直接读取的对象属性：`self.events`、`self.start_progress`、`self.end_progress`、`self.phase_count`。

输出分支的源码表达式：

```python
(block, record)
```

直接写入的状态或数组位置：`block.end_progress`、`block.events`、`identities[name]`、`block.source_timestamps`、`block.source_progress`、`block.conditions`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1262)。

输入签名：

```python
@property
def PlanningProblem.requirement_consumption(self):
```

直接读取的对象属性：`self.start_progress`、`self.phase_count`、`self.conditions`、`self.phases`、`self.events`。

输出分支的源码表达式：

```python
tuple(rows)
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1273)。

输入签名：

```python
def PlanningProblem.requirement_consumption.add(identity, categories, consumers, scope, source, **details):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
rows.append
deepcopy
```

## `PlanningProblem.source_time_map`

[定义位置](../../../manipulation/planning_problem.py#L1353)。

输入签名：

```python
@property
def PlanningProblem.source_time_map(self):
```

直接读取的对象属性：`self.source_timestamps`、`self.source_progress`。

输出分支的源码表达式：

```python
{'source_times': self.source_timestamps.copy(), 'progress': self.source_progress.copy()}
```

直接调用：

```python
ValueError
self.source_timestamps.copy
self.source_progress.copy
```

## `PlanningProblem.source_time_to_progress`

[定义位置](../../../manipulation/planning_problem.py#L1359)。

输入签名：

```python
def PlanningProblem.source_time_to_progress(self, source_time):
```

直接读取的对象属性：`self.source_time_map`。

输出分支的源码表达式：

```python
np.interp(source_time, mapping['source_times'], mapping['progress'])
```

直接调用：

```python
np.interp
```

## `PlanningProblem.progress_to_source_time`

[定义位置](../../../manipulation/planning_problem.py#L1363)。

输入签名：

```python
def PlanningProblem.progress_to_source_time(self, progress):
```

直接读取的对象属性：`self.source_time_map`。

输出分支的源码表达式：

```python
np.interp(progress, mapping['progress'], mapping['source_times'])
```

直接调用：

```python
np.interp
```

## `PlanningProblem.terminal_pose_target`

[定义位置](../../../manipulation/planning_problem.py#L1368)。

输入签名：

```python
@property
def PlanningProblem.terminal_pose_target(self):
```

直接读取的对象属性：`self.pose_events`、`self.end_progress`。

输出分支的源码表达式：

```python
next((goal for goal in self.pose_events if goal.progress == self.end_progress), None)
```

直接调用：

```python
next
```

## `PlanningProblem.anchors`

[定义位置](../../../manipulation/planning_problem.py#L1373)。

输入签名：

```python
@property
def PlanningProblem.anchors(self):
```

直接读取的对象属性：`self.start_progress`、`self.end_progress`、`self.events`、`self.supports`、`self.interiors`、`self.pose_events`、`self.coordinate_events`、`self.mechanisms`、`self.pose_relations`。

输出分支的源码表达式：

```python
anchors[(anchors >= self.start_progress) & (anchors <= self.end_progress)]
```

直接调用：

```python
np.unique
float
```

## `PlanningProblem.stop_progress`

[定义位置](../../../manipulation/planning_problem.py#L1387)。

输入签名：

```python
@property
def PlanningProblem.stop_progress(self):
```

直接读取的对象属性：`self.events`、`self.start_progress`。

输出分支的源码表达式：

```python
np.array([event.progress for event in self.events if event.stop and event.progress >= self.start_progress], float)
```

直接调用：

```python
np.array
```

## `PlanningProblem.sample_reference`

[定义位置](../../../manipulation/planning_problem.py#L1390)。

输入签名：

```python
def PlanningProblem.sample_reference(self, progress):
```

直接读取的对象属性：`self.phases`。

输出分支的源码表达式：

```python
(position, quaternion)
```

直接调用：

```python
self.phases[0].reference.sample
progress.clamp
enumerate
phase.reference.sample
(progress - index).clamp
torch.where
```

## `PlanningProblem.transform_tool_pose`

[定义位置](../../../manipulation/planning_problem.py#L1399)。

输入签名：

```python
def PlanningProblem.transform_tool_pose(self, position, quaternion):
```

直接读取的对象属性：`self.base_pose`、`self.tool_transform`。

输出分支的源码表达式：

```python
(tool_p + quaternion_rotate(tool_q, _tensor(self.tool_transform[:3, 3], position).expand_as(position)), quaternion_multiply(tool_q, _tensor(_quaternion(self.tool_transform), quaternion)))
```

直接调用：

```python
_tensor
_quaternion
quaternion_multiply
quaternion_rotate
_tensor(self.tool_transform[:3, 3], position).expand_as
```

## `PlanningProblem.reference_to_link_pose`

[定义位置](../../../manipulation/planning_problem.py#L1406)。

输入签名：

```python
def PlanningProblem.reference_to_link_pose(self, position, quaternion):
```

直接读取的对象属性：`self.tool_transform`、`self.base_pose`。

输出分支的源码表达式：

```python
(_tensor(inverse_base[:3, 3], position) + quaternion_rotate(bq, link_position), quaternion_multiply(bq, link_quaternion))
```

直接调用：

```python
_inverse
quaternion_rotate
_tensor(inverse_tool[:3, 3], position).expand_as
_tensor
quaternion_multiply
_quaternion
```

## `PlanningProblem.transform_robot_spheres`

[定义位置](../../../manipulation/planning_problem.py#L1414)。

输入签名：

```python
def PlanningProblem.transform_robot_spheres(self, spheres):
```

直接读取的对象属性：`self.base_pose`。

输出分支的源码表达式：

```python
torch.cat((centers + _tensor(self.base_pose[:3, 3], spheres), spheres[..., 3:]), -1)
```

直接调用：

```python
quaternion_rotate
_tensor
_quaternion
torch.cat
```

## `PlanningProblem.transform_tool_velocity`

[定义位置](../../../manipulation/planning_problem.py#L1420)。

输入签名：

```python
def PlanningProblem.transform_tool_velocity(self, linear, angular, link_quaternion):
```

直接读取的对象属性：`self.base_pose`、`self.tool_transform`。

输出分支的源码表达式：

```python
(world_linear + torch.linalg.cross(world_angular, offset), world_angular)
```

直接调用：

```python
_tensor
_quaternion
quaternion_rotate
quaternion_multiply
_tensor(self.tool_transform[:3, 3], linear).expand_as
torch.linalg.cross
```

## `PlanningProblem.residuals`

[定义位置](../../../manipulation/planning_problem.py#L1430)。

输入签名：

```python
def PlanningProblem.residuals(self, position, quaternion, progress, joint_position=None, joint_names=None, joint_velocity=None, *, trajectory_time=None, point_goal_ids=None, tool_linear_velocity=None, tool_angular_velocity=None, direction_mode='trajectory'):
```

直接读取的对象属性：`self.start_progress`、`self.end_progress`、`self.numerical`、`self.pose_relations`、`self.mechanisms`、`self.supports`、`self.interiors`、`self.pose_events`、`self.coordinate_events`、`self.couplings`、`self.events`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[f'pose_relation:{index}']`、`result[f'mechanism:{index}']`、`excess[..., angular]`、`result[f'mechanism_limits:{index}']`、`result[f'mechanism:{index}:{name}']`、`result[f'support:{index}']`、`result[f'interior:{index}']`、`result[f'function_pose:{index}']`、`result[f'function_coordinate:{index}']`、`result[f'coupling:{index}']`、`result[f'stop:{event.skill_id}:{event.event_id}']`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1439)。

输入签名：

```python
def PlanningProblem.residuals.event_mask(collection, index, requirement):
```

输出分支的源码表达式：

```python
_first_event_query(progress, requirement.progress, trajectory_time) if selected_goals is None or (collection, index) in selected_goals else torch.zeros_like(progress, dtype=torch.bool)
```

直接调用：

```python
_first_event_query
torch.zeros_like
```

## `PlanningProblem.evaluate`

[定义位置](../../../manipulation/planning_problem.py#L1523)。

输入签名：

```python
def PlanningProblem.evaluate(self, position, quaternion, progress, joint_position=None, joint_names=None, joint_velocity=None, *, trajectory_time=None, point_goal_ids=None, tool_linear_velocity=None, tool_angular_velocity=None, direction_mode='trajectory'):
```

直接读取的对象属性：`self.residuals`。

输出分支的源码表达式：

```python
sum(self.residuals(position, quaternion, progress, joint_position, joint_names, joint_velocity, trajectory_time=trajectory_time, point_goal_ids=point_goal_ids, tool_linear_velocity=tool_linear_velocity, tool_angular_velocity=tool_angular_velocity, direction_mode=direction_mode).values())
```

直接调用：

```python
sum
self.residuals(position, quaternion, progress, joint_position, joint_names, joint_velocity, trajectory_time=trajectory_time, point_goal_ids=point_goal_ids, tool_linear_velocity=tool_linear_velocity, tool_angular_velocity=tool_angular_velocity, direction_mode=direction_mode).values
self.residuals
```

## `PlanningProblem.functional_constraint_keys`

[定义位置](../../../manipulation/planning_problem.py#L1533)。

输入签名：

```python
@property
def PlanningProblem.functional_constraint_keys(self):
```

直接读取的对象属性：`self.pose_events`、`self.start_progress`、`self.coordinate_events`、`self.supports`。

输出分支的源码表达式：

```python
tuple([f'function_pose:{index}:{component}' for (index, requirement) in enumerate(self.pose_events) if requirement.progress >= self.start_progress for component in ('position', 'rotation')] + [f'function_coordinate:{index}' for (index, requirement) in enumerate(self.coordinate_events) if requirement.progress >= self.start_progress] + [f'support:{index}:{component}' for (index, requirement) in enumerate(self.supports) if requirement.progress >= self.start_progress for component in ('gap', 'body', 'footprint', 'normal')])
```

直接调用：

```python
tuple
enumerate
```

## `PlanningProblem.functional_inequalities`

[定义位置](../../../manipulation/planning_problem.py#L1548)。

输入签名：

```python
def PlanningProblem.functional_inequalities(self, position, quaternion, progress, *, trajectory_time=None):
```

直接读取的对象属性：`self.pose_events`、`self.start_progress`、`self.coordinate_events`、`self.supports`、`self.numerical`。

输出分支的源码表达式：

```python
(signed, masks)
```

直接写入的状态或数组位置：`signed[key]`、`masks[key]`。

直接调用：

```python
ValueError
enumerate
_first_event_query
requirement.errors
add
requirement.error
mask.to(torch.int64).argmax
mask.to
position.gather
selected[..., None].expand
quaternion.gather
requirement.normalized_inequalities
components.items
torch.where
mask.any
value.squeeze
float
```

## `PlanningProblem.functional_inequalities.add`

[定义位置](../../../manipulation/planning_problem.py#L1570)。

输入签名：

```python
def PlanningProblem.functional_inequalities.add(key, error, tolerance, mask):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`signed[key]`、`masks[key]`。

直接调用：

```python
torch.where(mask, error / tolerance - 1.0, 0.0).sum
torch.where
mask.any
float
```

## `PlanningProblem.optimizer_preferences`

[定义位置](../../../manipulation/planning_problem.py#L1606)。

输入签名：

```python
def PlanningProblem.optimizer_preferences(self, position, quaternion, progress, *, trajectory_time=None, tool_linear_velocity=None, tool_angular_velocity=None, phase='quality'):
```

直接读取的对象属性：`self.pose_relations`、`self.pose_events`、`self.coordinate_events`、`self.mechanisms`。

输出分支的源码表达式：

```python
result
```

直接调用：

```python
ValueError
torch.zeros_like
np.isfinite(requirement.pose_bounds).all
np.isfinite
exact.any
constant_tensor
np.flatnonzero
requirement.error
(error - _tensor(requirement.pose_bounds[exact, 0], position)).square().sum
(error - _tensor(requirement.pose_bounds[exact, 0], position)).square
_tensor
torch.where
requirement.errors
(p / requirement.position_tolerance_m).square
(r / requirement.rotation_tolerance_rad).square
_first_event_query
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

[定义位置](../../../manipulation/planning_problem.py#L1671)。

输入签名：

```python
def PlanningProblem.measurements(self, position, quaternion, progress, *, trajectory_time=None):
```

直接读取的对象属性：`self.phases`、`self.mechanisms`、`self.pose_events`、`self.coordinate_events`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[f'{prefix}_position_error_m']`、`result[f'{prefix}_rotation_error_rad']`、`result[f'{prefix}_position_component_error_m']`、`result[f'{prefix}_rotation_component_error_rad']`、`result[f'mechanism:{index}:position_error_m']`、`result[f'mechanism:{index}:rotation_error_rad']`、`result[f'function:{index}:position_error_m']`、`result[f'function:{index}:rotation_error_rad']`、`result[f'coordinate:{index}:error_{unit}']`。

直接调用：

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
requirement.error
```

## `PlanningProblem.verification_issues`

[定义位置](../../../manipulation/planning_problem.py#L1700)。

输入签名：

```python
def PlanningProblem.verification_issues(self, progress):
```

直接读取的对象属性：`self.anchors`、`self.mechanisms`。

输出分支的源码表达式：

```python
issues
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1718)。

输入签名：

```python
def PlanningProblem.remaining_obligations(self, progress, applied_ids=()):
```

直接读取的对象属性：`self.phase_count`、`self.events`、`self.mechanisms`、`self.phases`、`self.conditions`、`self.source_time_map`。

输出分支的源码表达式：

```python
{'progress': float(progress), 'original_phase_count': self.phase_count, 'phases': tuple(((index, deepcopy(phase)) for (index, phase) in enumerate(self.phases) if index + 1 >= progress)), 'events': tuple(deepcopy(pending)), 'overdue_event_ids': tuple((event_occurrence_key(event) for event in pending if event.progress < progress and event.kind in {'open', 'close', 'hold'})), 'goals': {name: tuple((deepcopy(requirement) for requirement in getattr(self, name) if requirement.progress >= progress)) for name in ('pose_events', 'coordinate_events', 'supports')}, 'past_goal_assessment_required': tuple(((name, index) for name in ('pose_events', 'coordinate_events', 'supports') for (index, requirement) in enumerate(getattr(self, name)) if requirement.progress < progress)), 'process': scoped, 'conditions': deepcopy(self.conditions), 'source_time_map': self.source_time_map, 'required_actual_inputs': tuple(actual_inputs), 'clock_policy': 'ordinary continuation uses elapsed+t; explicit rebinding preserves declared physical-time relations'}
```

直接写入的状态或数组位置：`scoped[name]`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1772)。

输入签名：

```python
def PlanningProblem.projection_requirements(self, progress, side='at'):
```

直接读取的对象属性：`self.couplings`。

输出分支的源码表达式：

```python
tuple(selected) + tuple((('couplings', index) for index in range(len(self.couplings))))
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1789)。

输入签名：

```python
def PlanningProblem.goal_requirements_at(self, progress):
```

输出分支的源码表达式：

```python
tuple(((name, index) for name in ('pose_events', 'coordinate_events', 'supports') for (index, goal) in enumerate(getattr(self, name)) if abs(goal.progress - progress) <= 1e-07))
```

直接调用：

```python
tuple
enumerate
getattr
abs
```

## `PlanningProblem.goal_residual`

[定义位置](../../../manipulation/planning_problem.py#L1794)。

输入签名：

```python
def PlanningProblem.goal_residual(self, position, quaternion, progress, *, goal_ids):
```

直接读取的对象属性：`self.numerical`。

输出分支的源码表达式：

```python
torch.cat(pieces, -1) if pieces else position[..., :0]
```

直接调用：

```python
ValueError
getattr
goal.errors
pieces.extend
torch.relu
pieces.append
goal.evaluate
goal.raw_inequalities(position, quaternion, self.numerical).values
goal.raw_inequalities
torch.cat
```

## `PlanningProblem.project_residual`

[定义位置](../../../manipulation/planning_problem.py#L1822)。

输入签名：

```python
def PlanningProblem.project_residual(self, position, quaternion, progress, phase_index=None, joint_position=None, joint_names=None, active_requirements=None):
```

直接读取的对象属性：`self.pose_relations`、`self.mechanisms`、`self.couplings`。

输出分支的源码表达式：

```python
torch.cat(pieces, -1) if pieces else position[..., :0]
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1866)。

输入签名：

```python
def _path_from_rows(poses, source, *, timestamps, model=None, coordinates=None, attachment=None):
```

输出分支的源码表达式：

```python
PosePath(progress, poses, source, model, coordinates, np.eye(4) if attachment is None else attachment)
```

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1876)。

输入签名：

```python
def _model_from_diagnostics(value):
```

输出分支的源码表达式：

```python
None
MotionModel(value['kind'], np.asarray(value['coordinates'], float), np.asarray(value['reference_pose'], float), np.asarray(value['basis'], float), None if value['axis_point'] is None else np.asarray(value['axis_point'], float), value['position_tolerance'], value['rotation_tolerance'], value['diagnostics'])
```

直接调用：

```python
MotionModel
np.asarray
```

## `_validate_skill_requirements`

[定义位置](../../../manipulation/planning_problem.py#L1888)。

输入签名：

```python
def _validate_skill_requirements(spec):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L1895)。

输入签名：

```python
def _validate_skill_requirements.fields(record, allowed, label):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
isinstance
ValueError
set
sorted
```

## `build_planning_problem`

[定义位置](../../../manipulation/planning_problem.py#L1979)。

输入签名：

```python
def build_planning_problem(skills, *, joint_names, start_q, start_dq, tool_link, reference_frame, base_pose, tool_transform, numerical=NumericalTolerance(), couplings=()):
```

直接读取的对象属性：`scene.get`、`problem.conditions`、`problem.events`。

输出分支的源码表达式：

```python
problem
```

直接写入的状态或数组位置：`scene[initial_key]`、`mechanisms[0].directions`。

直接调用：

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

[定义位置](../../../manipulation/planning_problem.py#L2037)。

输入签名：

```python
def build_planning_problem.append_interval(begin, end, path, metadata):
```

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`frame_progress[int(frames[begin])]`、`frame_progress[int(frames[row])]`。

直接调用：

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
