# `manipulation/motion.py` 实现契约

代码基线：`82a3d66；包含本次工作区已实现改动，待统一实现提交后更新`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_poses_array`

[定义位置](../../../manipulation/motion.py#L33)。

输入签名：

```python
def _poses_array(value: Any, name: str='poses') -> np.ndarray:
```

实现说明：

> Validate SE(3) without projecting reflections or invalid rotations.

显式输出分支：

```python
poses
```

读取的对象属性：

```python
np.array
poses.ndim
poses.shape
np.isfinite(poses).all
np.isfinite
np.allclose
np.swapaxes
np.eye
np.linalg.det
np.linalg
np.ndarray
```

写入目标：

```python
poses
rotations
products
```

直接调用符号：

```python
np.array
len
ValueError
np.isfinite(poses).all
np.isfinite
np.allclose
np.swapaxes
np.eye
np.linalg.det
```

## `_pose_array`

[定义位置](../../../manipulation/motion.py#L51)。

输入签名：

```python
def _pose_array(value: Any, name: str) -> np.ndarray:
```

显式输出分支：

```python
_poses_array(pose[None], name)[0]
```

读取的对象属性：

```python
np.asarray
pose.shape
np.ndarray
```

写入目标：

```python
pose
```

直接调用符号：

```python
np.asarray
ValueError
_poses_array
```

## `_tolerances`

[定义位置](../../../manipulation/motion.py#L58)。

输入签名：

```python
def _tolerances(position: float, rotation: float) -> tuple[float, float]:
```

显式输出分支：

```python
(position, rotation)
```

读取的对象属性：

```python
np.isfinite
np.pi
```

写入目标：

```python
position
rotation
```

直接调用符号：

```python
float
np.isfinite
ValueError
```

## `_inverse`

[定义位置](../../../manipulation/motion.py#L67)。

输入签名：

```python
def _inverse(pose: np.ndarray) -> np.ndarray:
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
np.ndarray
np.eye
pose[:3, :3].T
```

写入目标：

```python
result
result[:3, :3]
result[:3, 3]
```

直接调用符号：

```python
np.eye
```

## `_errors`

[定义位置](../../../manipulation/motion.py#L74)。

输入签名：

```python
def _errors(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
```

显式输出分支：

```python
(positions, angles)
```

读取的对象属性：

```python
np.ndarray
np.linalg.norm
np.linalg
np.swapaxes
Rotation.from_matrix(relative).magnitude
Rotation.from_matrix
```

写入目标：

```python
positions
relative
angles
```

直接调用符号：

```python
np.linalg.norm
np.swapaxes
Rotation.from_matrix(relative).magnitude
Rotation.from_matrix
```

## `_basis`

[定义位置](../../../manipulation/motion.py#L81)。

输入签名：

```python
def _basis(axis: np.ndarray, reference_rotation: np.ndarray) -> np.ndarray:
```

实现说明：

> Choose a reproducible right-handed basis using the body-frame gauge.
>
> Axis sign and in-plane basis are coordinate conventions, not estimates of
> opening direction or a semantic zero.  Using the reference orientation
> makes the convention equivariant under a common world-frame transform.

显式输出分支：

```python
np.column_stack((first, np.cross(axis, first), axis))
```

读取的对象属性：

```python
np.ndarray
np.asarray
np.linalg.norm
np.linalg
np.isfinite
reference_rotation.T
np.flatnonzero
np.abs
np.max
lengths.max
np.column_stack
np.cross
```

写入目标：

```python
axis
norm
dots
strongest
projected
lengths
index
first
```

直接调用符号：

```python
np.asarray
np.linalg.norm
np.isfinite
ValueError
np.flatnonzero
np.abs
np.max
lengths.max
np.column_stack
np.cross
```

## `_position_svd`

[定义位置](../../../manipulation/motion.py#L104)。

输入签名：

```python
def _position_svd(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
```

显式输出分支：

```python
(singular[:3] / np.sqrt(len(positions)), vectors)
```

读取的对象属性：

```python
np.ndarray
positions.mean
np.linalg.svd
np.linalg
np.pad
np.sqrt
```

写入目标：

```python
centered
_
singular
vectors
```

直接调用符号：

```python
positions.mean
np.linalg.svd
np.pad
max
len
np.sqrt
```

## `_angles`

[定义位置](../../../manipulation/motion.py#L113)。

输入签名：

```python
def _angles(relative_rotations: np.ndarray, axis: np.ndarray) -> np.ndarray:
```

实现说明：

> Closest rotation about axis, then a continuous observed angle lift.
>
> atan2 uses the trace-maximizing projection onto the axis subgroup.  Its
> residual is checked separately; projection does not establish compatibility.
> A half-turn between samples has two equally plausible motion directions.

显式输出分支：

```python
np.unwrap(principal)
```

读取的对象属性：

```python
np.ndarray
np.column_stack
np.trace
np.einsum
np.arctan2
np.angle
np.exp
np.diff
np.any
np.abs
np.pi
np.unwrap
```

写入目标：

```python
skew_vectors
sine
cosine
principal
increments
```

直接调用符号：

```python
np.column_stack
np.trace
np.einsum
np.arctan2
np.angle
np.exp
np.diff
np.any
np.abs
ValueError
np.unwrap
```

## `MotionModel`

[定义位置](../../../manipulation/motion.py#L140)。

```python
class MotionModel():
```

类字段：

```python
kind: str
coordinates: np.ndarray
reference_pose: np.ndarray
basis: np.ndarray
axis_point: np.ndarray | None
position_tolerance: float
rotation_tolerance: float
diagnostics: dict[str, Any]
```

实现说明：

> One identifiable motion family fitted to a fixed-body pose sequence.
>
> ``coordinates`` contains the fitted reference sequence, or the returned
> sequence after ``transfer_motion``.  For free motion its columns are world
> position and principal rotation vector, not an angular interpolation rule.
> For planar motion columns are x/y in ``basis[:, :2]`` and lifted yaw.
> Translation-plane columns are x/y; revolute/prismatic use one angle/distance.
> ``axis`` is ``basis[:, 2]``.  A revolute axis passes through ``axis_point``.

## `MotionModel.dimension`

[定义位置](../../../manipulation/motion.py#L161)。

输入签名：

```python
def dimension(self) -> int:
```

显式输出分支：

```python
_DIMENSIONS[self.kind]
```

读取的对象属性：

```python
self.kind
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `MotionModel.axis`

[定义位置](../../../manipulation/motion.py#L165)。

输入签名：

```python
def axis(self) -> np.ndarray:
```

显式输出分支：

```python
self.basis[:, 2].copy()
```

读取的对象属性：

```python
self.basis[:, 2].copy
self.basis
np.ndarray
```

写入目标：

无显式条目。

直接调用符号：

```python
self.basis[:, 2].copy
```

## `MotionModel.poses`

[定义位置](../../../manipulation/motion.py#L168)。

输入签名：

```python
def poses(self, coordinates: Any) -> np.ndarray:
```

实现说明：

> Evaluate FK at coordinates; interpolation, when used, belongs here.
>
> No world-position chord is substituted for a revolute arc.  A single
> coordinate vector is accepted and still returns shape (1, 4, 4).

显式输出分支：

```python
output
```

读取的对象属性：

```python
np.asarray
values.ndim
values.shape
self.dimension
self.kind
np.isfinite(values).all
np.isfinite
np.repeat
np.eye
self.reference_pose
Rotation.from_rotvec(values[:, 3:]).as_matrix
Rotation.from_rotvec
self.basis[:, :2].T
self.basis
Rotation.from_rotvec(values[:, 2, None] * self.axis).as_matrix
self.axis
Rotation.from_rotvec(values * self.axis).as_matrix
self.axis_point
np.einsum
np.ndarray
```

写入目标：

```python
values
output
position
orientation
output[:, :3, 3]
output[:, :3, :3]
rotations
```

直接调用符号：

```python
np.asarray
len
ValueError
np.isfinite(values).all
np.isfinite
np.repeat
np.eye
Rotation.from_rotvec(values[:, 3:]).as_matrix
Rotation.from_rotvec
Rotation.from_rotvec(values[:, 2, None] * self.axis).as_matrix
Rotation.from_rotvec(values * self.axis).as_matrix
np.einsum
```

## `MotionModel._coordinates_unchecked`

[定义位置](../../../manipulation/motion.py#L207)。

输入签名：

```python
def _coordinates_unchecked(self, poses: np.ndarray) -> np.ndarray:
```

显式输出分支：

```python
np.column_stack((poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_rotvec()))
(positions @ self.axis)[:, None]
positions @ self.basis[:, :2]
angle[:, None]
np.column_stack((positions @ self.basis[:, :2], angle))
```

读取的对象属性：

```python
np.ndarray
self.reference_pose
self.kind
np.column_stack
Rotation.from_matrix(poses[:, :3, :3]).as_rotvec
Rotation.from_matrix
self.axis
self.basis
self.reference_pose[:3, :3].T
```

写入目标：

```python
positions
relative
angle
```

直接调用符号：

```python
np.column_stack
Rotation.from_matrix(poses[:, :3, :3]).as_rotvec
Rotation.from_matrix
_angles
```

## `MotionModel.coordinates_of`

[定义位置](../../../manipulation/motion.py#L223)。

输入签名：

```python
def coordinates_of(self, poses: Any) -> np.ndarray:
```

实现说明：

> Return coordinates only when full poses match this family.
>
> Compatibility uses the declared fitting tolerances.  Returned angles
> are unwrapped in sequence order; a single pose has a principal branch.
> ``transfer_motion`` separately checks exact endpoint reachability and
> chooses the goal branch nearest the reference terminal coordinate.

显式输出分支：

```python
coordinates
```

读取的对象属性：

```python
self._coordinates_unchecked
self.poses
position_error.max
self.position_tolerance
self.kind
rotation_error.max
self.rotation_tolerance
np.ndarray
```

写入目标：

```python
observed
coordinates
position_error
rotation_error
```

直接调用符号：

```python
_poses_array
self._coordinates_unchecked
_errors
self.poses
position_error.max
ValueError
rotation_error.max
```

## `MotionModel.to_dict`

[定义位置](../../../manipulation/motion.py#L246)。

输入签名：

```python
def to_dict(self) -> dict[str, Any]:
```

实现说明：

> JSON-compatible model parameters, coordinates, and evidence.

显式输出分支：

```python
{'kind': self.kind, 'dimension': self.dimension, 'reference_pose': self.reference_pose.tolist(), 'basis': self.basis.tolist(), 'axis_point': None if self.axis_point is None else self.axis_point.tolist(), 'coordinates': self.coordinates.tolist(), 'position_tolerance': self.position_tolerance, 'rotation_tolerance': self.rotation_tolerance, 'diagnostics': self.diagnostics, 'pose_semantics': 'fixed object/part frame in reference frame; not end effector'}
```

读取的对象属性：

```python
self.kind
self.dimension
self.reference_pose.tolist
self.reference_pose
self.basis.tolist
self.basis
self.axis_point
self.axis_point.tolist
self.coordinates.tolist
self.coordinates
self.position_tolerance
self.rotation_tolerance
self.diagnostics
```

写入目标：

无显式条目。

直接调用符号：

```python
self.reference_pose.tolist
self.basis.tolist
self.axis_point.tolist
self.coordinates.tolist
```

## `_fit_one`

[定义位置](../../../manipulation/motion.py#L262)。

输入签名：

```python
def _fit_one(poses: np.ndarray, kind: str, ptol: float, rtol: float) -> MotionModel:
```

显式输出分支：

```python
model
```

读取的对象属性：

```python
np.ndarray
poses[0].copy
reference[:3, :3].T
Rotation.from_matrix(relative).magnitude().max
Rotation.from_matrix(relative).magnitude
Rotation.from_matrix
np.linalg.norm(positions - positions[0], axis=1).max
np.linalg.norm
np.linalg
np.eye
spread.tolist
np.concatenate
np.linalg.svd
singular.tolist
np.linalg.lstsq
(positions - positions[0]).reshape
axis_singular.tolist
np.empty
model.coordinates_of
model.poses
model.coordinates
diagnostics.update
position_error.tolist
rotation_error.tolist
position_error.max
rotation_error.max
```

写入目标：

```python
reference
positions
relative
rotation_excitation
translation_excitation
spread
vectors
basis
axis_point
diagnostics
rotation_constraints
_
singular
right
normal
diagnostics['rotation_axis_singular_values']
constraints
system
offsets
rank
axis_singular
diagnostics['axis_location_singular_values']
diagnostics['axis_location_noise_gain']
model
model.coordinates
position_error
rotation_error
diagnostics['angular_branch']
```

直接调用符号：

```python
poses[0].copy
float
Rotation.from_matrix(relative).magnitude().max
Rotation.from_matrix(relative).magnitude
Rotation.from_matrix
np.linalg.norm(positions - positions[0], axis=1).max
np.linalg.norm
_position_svd
np.eye
spread.tolist
ValueError
_basis
np.concatenate
np.linalg.svd
singular.tolist
np.linalg.lstsq
(positions - positions[0]).reshape
axis_singular.tolist
MotionModel
np.empty
model.coordinates_of
_errors
model.poses
diagnostics.update
position_error.tolist
rotation_error.tolist
position_error.max
rotation_error.max
```

## `_candidate_models`

[定义位置](../../../manipulation/motion.py#L330)。

输入签名：

```python
def _candidate_models(observed: np.ndarray, ptol: float, rtol: float):
```

实现说明：

> Fit the existing families without interpreting them as physical joints.

显式输出分支：

```python
(accepted, rejected, excited)
```

读取的对象属性：

```python
np.ndarray
accepted.append
model.kind
free.diagnostics
```

写入目标：

```python
accepted
rejected
candidate
rejected[candidate]
free
model
excited
```

直接调用符号：

```python
accepted.append
_fit_one
str
next
```

## `analyze_motion`

[定义位置](../../../manipulation/motion.py#L346)。

输入签名：

```python
def analyze_motion(poses: Any, position_tolerance: float=0.005, rotation_tolerance: float=0.035) -> dict[str, Any]:
```

实现说明：

> Describe observed motion without selecting an applied task constraint.
>
> Every candidate is checked against the complete SE(3) sequence, including
> orientation. A line may be compatible with a prismatic description while
> remaining an unconstrained carry task. Similarly, a planar observation
> alone cannot establish contact or a physical planar joint. Accepted models
> retain all fitted coordinates and per-frame residuals; no terminal-only
> compression replaces the observed motion. Near-static observations remain
> valid poses but report that their motion structure is not identifiable.
>
> Invalid poses raise before candidate fitting. In particular, ``free`` does
> not repair invalid rotations, missing observations or a lost track.

显式输出分支：

```python
{'status': 'observed_compatible' if excited else 'insufficient_excitation', 'preferred_description': accepted[0].kind if excited else None, 'compatible_candidates': [model.kind for model in accepted], 'candidates': {model.kind: model.to_dict() for model in accepted}, 'rejected_candidates': rejected, 'position_tolerance_m': ptol, 'rotation_tolerance_rad': rtol, 'interpretation': 'observed geometry only; task and reference evidence determine applied constraints'}
```

读取的对象属性：

```python
accepted[0].kind
model.kind
model.to_dict
```

写入目标：

```python
observed
ptol
rtol
accepted
rejected
excited
model
```

直接调用符号：

```python
_poses_array
_tolerances
_candidate_models
model.to_dict
```

## `fit_motion`

[定义位置](../../../manipulation/motion.py#L379)。

输入签名：

```python
def fit_motion(poses: Any, kind: str='auto', position_tolerance: float=0.005, rotation_tolerance: float=0.035) -> MotionModel:
```

实现说明：

> Fit one of five rigid motion descriptions using complete poses.
>
> ``auto`` selects the lowest-dimensional identifiable compatible family and
> reports all tested alternatives.  Free SE(3) containment is not evidence
> against a constrained description, nor is a constrained fit proof of a
> physical joint.  Near-static input cannot identify a family in auto mode.
> Tolerances are explicit acceptance scales, not estimated sensor precision.

显式输出分支：

```python
_fit_one(observed, kind, ptol, rtol)
selected
```

读取的对象属性：

```python
selected.diagnostics
selected.kind
model.kind
```

写入目标：

```python
observed
ptol
rtol
accepted
rejected
excited
selected
selected.diagnostics['selection']
model
```

直接调用符号：

```python
_poses_array
_tolerances
ValueError
_fit_one
_candidate_models
len
```

## `free_transfer_parameters`

[定义位置](../../../manipulation/motion.py#L418)。

输入签名：

```python
def free_transfer_parameters(reference_start, initial_pose, reference_goal=None, goal_pose=None):
```

实现说明：

> Bind one free-motion interval without sampling or inventing a clock.
>
> The reference is evaluated at the caller's fixed material frame. ``right``
> attaches its observed entrance to the actual entrance. The two terminal
> corrections are subsequently weighted by the caller's explicit interval
> progress. A suffix without another goal uses only ``right``; retaining a
> previous interval's terminal correction would change its local motion.
> These are geometric boundary parameters, not robot task constraints.

显式输出分支：

```python
{'right': right, 'translation_correction': translation, 'rotation_correction': rotation}
```

读取的对象属性：

```python
np.array_equal
np.eye
np.zeros
mapped_goal[:3, :3].T
```

写入目标：

```python
reference_start
initial
right
translation
rotation
reference_goal
goal
mapped_goal
```

直接调用符号：

```python
_pose_array
ValueError
np.array_equal
np.eye
_inverse
np.zeros
```

## `transfer_motion`

[定义位置](../../../manipulation/motion.py#L447)。

输入签名：

```python
def transfer_motion(demo_poses: Any, initial_pose: Any, reference_transform: Any=np.eye(4), kind: str='free', goal_pose: Any | None=None, position_tolerance: float=0.005, rotation_tolerance: float=0.035, *, progress: Any) -> dict[str, Any]:
```

实现说明：

> Transfer object motion, preserving the initial pose and reachable goal.
>
> A maps demonstration-reference coordinates to test-reference coordinates:
> ``T[k] = A @ Td[k] @ inv(Td[0]) @ inv(A) @ initial_pose``.
> The supplied mapping is an input assumption; it is not evidence that an
> unseen mechanism shares the demonstration axis or shape.
>
> A terminal correction follows explicitly supplied normalized progress in
> model coordinates.  For free motion, translation uses an additive world
> correction and orientation uses a left-multiplied Slerp correction.  For
> constrained motion no incompatible terminal pose is silently projected.
> ``progress`` must cover [0,1] and align with every pose (a single pose uses
> [0]). Callers with recorded times normalize those times; untimed geometric
> callers explicitly declare their geometric phase. Inserting a sample at
> an existing phase therefore does not redistribute the boundary correction.
> No physical timestamps or end-effector control are inferred.

显式输出分支：

```python
{'poses': output, 'model': model, 'diagnostics': diagnostics}
```

读取的对象属性：

```python
np.eye
np.asarray
phase.shape
np.isfinite(phase).all
np.isfinite
np.any
np.diff
model.coordinates.copy
model.coordinates
transform.tolist
model.kind
mapped.copy
Rotation.from_matrix
np.stack
rotations.as_matrix
model.coordinates_of
np.round
np.pi
model.poses
model.diagnostics
```

写入目标：

```python
demo
initial
transform
goal
phase
reference
boundary
mapped
mapped[0]
model
coordinates
diagnostics
p_error
r_error
output
output[:, :3, 3]
rotations
output[:, :3, :3]
goal_coordinates
angular_column
turns
goal_coordinates[angular_column]
diagnostics['goal_angle_branch']
projected_goal
initial_error
output[0]
endpoint_error
output[-1]
model.coordinates
diagnostics['model_fit']
```

直接调用符号：

```python
np.eye
_poses_array
_pose_array
np.asarray
len
np.isfinite(phase).all
np.isfinite
np.any
np.diff
ValueError
free_transfer_parameters
fit_motion
model.coordinates.copy
transform.tolist
_errors
max
mapped.copy
Slerp([0, 1], Rotation.from_matrix(np.stack((np.eye(3), boundary['rotation_correction']))))
Slerp
Rotation.from_matrix
np.stack
rotations.as_matrix
model.coordinates_of
np.round
model.poses
```
