# `manipulation/timed_reference.py` 实现契约

代码基线：`be44b54`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `BoundToolReference`

[定义位置](../../../manipulation/timed_reference.py#L22)。

```python
class BoundToolReference():
```

类字段：

无显式条目。

实现说明：

> One explicit clock and the TCP pose/twist measured at its entrance.
>
> ``initial_twist`` is [v_W, omega_W], in metres/second and radians/second.
> Phase metadata ``reference_boundaries`` declares an initial pose and
> nominal event goals in that phase's [0,1] coordinates. An event's common
> output velocity is the right-side nominal physical velocity; its left
> correction cancels any nominal derivative jump at that declared join.
> Other nominal corners remain part of the supplied reference.
>
> The source generator only declares geometric boundaries. This constructor
> binds their actual first-arrival times and constructs the corrections once.
> Plain phases without that declaration retain their nominal reference.

## `BoundToolReference.__init__`

[定义位置](../../../manipulation/timed_reference.py#L37)。

输入签名：

```python
def __init__(self, problem, progress_u, progress_s, duration, initial_pose, initial_twist):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray(progress_u, float).copy
np.asarray
np.asarray(progress_s, float).copy
np.asarray(initial_twist, float).copy
self.progress_u.ndim
self.progress_u
self.progress_s.shape
self.progress_s
self.progress_u.shape
np.isfinite(self.progress_u).all
np.isfinite
np.isfinite(self.progress_s).all
np.any
np.diff
self.duration
self.initial_twist.shape
self.initial_twist
np.isfinite(self.initial_twist).all
phase.reference
problem.phases
phase.metadata.get
phase.metadata
self.source_boundaries.append
self.source_boundaries
declaration.get
declared.append
goal.get
np.empty
self._bind_boundaries
```

写入目标：

```python
self.progress_u
self.progress_s
self.duration
self.initial_pose
self.initial_twist
self.nominal_phases
phase
self.boundaries
self.source_boundaries
declared
index
declaration
goal
progress
self._starts
self._ends
self._translation
self._rotation
self._restore_suffix
self.diagnostics
```

直接调用符号：

```python
np.asarray(progress_u, float).copy
np.asarray
np.asarray(progress_s, float).copy
float
_pose_array
np.asarray(initial_twist, float).copy
len
np.isfinite(self.progress_u).all
np.isfinite
np.isfinite(self.progress_s).all
np.any
np.diff
np.isfinite(self.initial_twist).all
ValueError
tuple
enumerate
phase.metadata.get
self.source_boundaries.append
deepcopy
declaration.get
declared.append
goal.get
np.empty
self._bind_boundaries
```

## `BoundToolReference._first_time`

[定义位置](../../../manipulation/timed_reference.py#L85)。

输入签名：

```python
def _first_time(self, progress):
```

实现说明：

> Invert only a declared event's first arrival, preserving hold spans.

显式输出分支：

```python
self.duration * self.progress_u[index]
self.duration * (self.progress_u[index - 1] + fraction * (self.progress_u[index] - self.progress_u[index - 1]))
```

读取的对象属性：

```python
np.searchsorted
self.progress_s
self.duration
self.progress_u
```

写入目标：

```python
index
fraction
```

直接调用符号：

```python
int
np.searchsorted
```

## `BoundToolReference._nominal`

[定义位置](../../../manipulation/timed_reference.py#L94)。

输入签名：

```python
def _nominal(self, times, *, side='right'):
```

实现说明：

> Sample original composed paths using the specified one-sided clock.

显式输出分支：

```python
(position, quaternion)
```

读取的对象属性：

```python
self.progress_u
self.duration
self.progress_s
self.nominal_phases[0].sample
self.nominal_phases
progress.clamp
path.sample
(progress - index).clamp
torch.where
```

写入目标：

```python
progress
position
quaternion
index
path
p
q
selected
```

直接调用符号：

```python
_interpolate
self.nominal_phases[0].sample
progress.clamp
enumerate
path.sample
(progress - index).clamp
torch.where
```

## `BoundToolReference._nominal_boundary`

[定义位置](../../../manipulation/timed_reference.py#L105)。

输入签名：

```python
def _nominal_boundary(self, time, side):
```

实现说明：

> Use analytic Torch derivatives, including rotating TCP lever arms.

显式输出分支：

```python
(pose, np.r_[linear[0].detach().numpy(), angular[0].detach().numpy()])
```

读取的对象属性：

```python
torch.tensor
torch.float64
torch.func.jvp
torch.func
self._nominal
torch.ones_like
np.eye
position[0].detach().numpy
position[0].detach
Rotation.from_quat(quaternion[0].detach().numpy()[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
quaternion[0].detach().numpy
quaternion[0].detach
np.r_
linear[0].detach().numpy
linear[0].detach
angular[0].detach().numpy
angular[0].detach
```

写入目标：

```python
value
position
quaternion
linear
derivative
angular
pose
pose[:3, 3]
pose[:3, :3]
```

直接调用符号：

```python
torch.tensor
torch.func.jvp
self._nominal
torch.ones_like
quaternion_multiply
quaternion_conjugate
np.eye
position[0].detach().numpy
position[0].detach
Rotation.from_quat(quaternion[0].detach().numpy()[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
quaternion[0].detach().numpy
quaternion[0].detach
linear[0].detach().numpy
linear[0].detach
angular[0].detach().numpy
angular[0].detach
```

## `BoundToolReference._rotation_coordinate`

[定义位置](../../../manipulation/timed_reference.py#L121)。

输入签名：

```python
def _rotation_coordinate(matrix, previous):
```

实现说明：

> Choose a continuous correction lift without logging the source path.
>
> Only the finite boundary correction is represented in exponential
> coordinates. Source mechanism turns and parent/attachment motion are
> evaluated directly by the original PosePath at every query.

显式输出分支：

```python
principal
min(candidates, key=lambda candidate: np.linalg.norm(candidate - previous))
```

读取的对象属性：

```python
Rotation.from_matrix(matrix).as_rotvec
Rotation.from_matrix
np.linalg.norm
np.linalg
np.rint
np.dot
np.pi
```

写入目标：

```python
principal
angle
axis
turns
candidates
k
```

直接调用符号：

```python
Rotation.from_matrix(matrix).as_rotvec
Rotation.from_matrix
np.linalg.norm
int
np.rint
np.dot
min
```

## `BoundToolReference._correction`

[定义位置](../../../manipulation/timed_reference.py#L138)。

输入签名：

```python
def _correction(nominal, nominal_twist, target, target_twist, previous=None):
```

显式输出分支：

```python
(b, phi, db, dphi)
```

读取的对象属性：

```python
nominal[:3, :3].T
BoundToolReference._rotation_coordinate
np.linalg.solve
np.linalg
pin.Jexp3
```

写入目标：

```python
rotation
phi
b
db
dphi
```

直接调用符号：

```python
BoundToolReference._rotation_coordinate
np.linalg.solve
pin.Jexp3
```

## `BoundToolReference._bind_boundaries`

[定义位置](../../../manipulation/timed_reference.py#L148)。

输入签名：

```python
def _bind_boundaries(self, declared):
```

显式输出分支：

```python
None
```

读取的对象属性：

```python
self._first_time
np.allclose
rows.append
self.diagnostics
self._nominal_boundary
self.duration
self._restore_suffix
self.progress_s
self.initial_pose
self.initial_twist
self._correction
translations.append
CubicHermiteSpline(times, [a[0], b[0]], [a[2], b[2]]).c
rotations.append
CubicHermiteSpline(times, [a[1], b[1]], [a[3], b[3]]).c
starts.append
ends.append
self.diagnostics['joins'].append
right_twist.tolist
right['twist'].tolist
b[2].tolist
b[3].tolist
np.asarray
row['pose'].tolist
row['twist'].tolist
self.diagnostics.update
self._ends
```

写入目标：

```python
rows
declaration
row
self.diagnostics['adaptation_scope']
final_nominal
_
self._restore_suffix
beginning
row['twist']
boundaries
translations
rotations
starts
ends
prior_phi
left
right
left_nominal
left_twist
right_nominal
right_twist
a
b
times
self._starts
self._ends
self._translation
self._rotation
self.boundaries
```

直接调用符号：

```python
sorted
deepcopy
self._first_time
_pose_array
np.allclose
ValueError
rows.append
self._nominal_boundary
bool
float
zip
self._correction
translations.append
CubicHermiteSpline
rotations.append
starts.append
ends.append
self.diagnostics['joins'].append
right_twist.tolist
right['twist'].tolist
b[2].tolist
b[3].tolist
np.asarray
row['pose'].tolist
row['twist'].tolist
self.diagnostics.update
```

## `BoundToolReference.sample`

[定义位置](../../../manipulation/timed_reference.py#L201)。

输入签名：

```python
def sample(self, times_s):
```

实现说明：

> Return W-frame TCP position (...,3) and unit wxyz quaternion (...,4).

显式输出分支：

```python
(position, quaternion)
(torch.where(active, corrected_position, position), torch.where(active, corrected_quaternion, quaternion))
```

读取的对象属性：

```python
self._nominal
self._starts
(torch.searchsorted(starts, times_s.contiguous(), right=True) - 1).clamp
torch.searchsorted
times_s.contiguous
self._translation
self._rotation
torch.linalg.vector_norm
torch.linalg
torch.cat
torch.cos
torch.sinc
torch.pi
self._restore_suffix
self._ends
torch.where
```

写入目标：

```python
position
quaternion
starts
index
delta
translation
phi
angle
correction
corrected_position
corrected_quaternion
active
```

直接调用符号：

```python
self._nominal
len
_tensor
(torch.searchsorted(starts, times_s.contiguous(), right=True) - 1).clamp
torch.searchsorted
times_s.contiguous
polynomial
torch.linalg.vector_norm
torch.cat
torch.cos
torch.sinc
quaternion_multiply
torch.where
```

## `BoundToolReference.sample.polynomial`

[定义位置](../../../manipulation/timed_reference.py#L210)。

输入签名：

```python
def polynomial(coefficients):
```

显式输出分支：

```python
((values[..., 0, :] * delta + values[..., 1, :]) * delta + values[..., 2, :]) * delta + values[..., 3, :]
```

读取的对象属性：

无显式条目。

写入目标：

```python
values
```

直接调用符号：

```python
_tensor
```

## `BoundToolReference.sample_numpy`

[定义位置](../../../manipulation/timed_reference.py#L225)。

输入签名：

```python
def sample_numpy(self, times_s):
```

实现说明：

> Return homogeneous TCP poses with the input time-array shape.

显式输出分支：

```python
poses
```

读取的对象属性：

```python
torch.as_tensor
np.asarray
torch.float64
self.sample
np.broadcast_to(np.eye(4), (*times.shape, 4, 4)).copy
np.broadcast_to
np.eye
times.shape
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
times
position
quaternion
poses
poses[..., :3, 3]
poses[..., :3, :3]
```

直接调用符号：

```python
torch.as_tensor
np.asarray
self.sample
np.broadcast_to(np.eye(4), (*times.shape, 4, 4)).copy
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
