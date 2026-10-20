# `manipulation/motion.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_poses_array`

[定义位置](../../../manipulation/motion.py#L33)。

输入签名：

```python
def _poses_array(value: Any, name: str='poses') -> np.ndarray:
```

输出分支的源码表达式：

```python
poses
```

直接调用：

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

输出分支的源码表达式：

```python
_poses_array(pose[None], name)[0]
```

直接调用：

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

输出分支的源码表达式：

```python
(position, rotation)
```

直接调用：

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

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[:3, :3]`、`result[:3, 3]`。

直接调用：

```python
np.eye
```

## `_errors`

[定义位置](../../../manipulation/motion.py#L74)。

输入签名：

```python
def _errors(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
```

输出分支的源码表达式：

```python
(positions, angles)
```

直接调用：

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

输出分支的源码表达式：

```python
np.column_stack((first, np.cross(axis, first), axis))
```

直接调用：

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

输出分支的源码表达式：

```python
(singular[:3] / np.sqrt(len(positions)), vectors)
```

直接调用：

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

输出分支的源码表达式：

```python
np.unwrap(principal)
```

直接调用：

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
class MotionModel:
    kind: str
    coordinates: np.ndarray
    reference_pose: np.ndarray
    basis: np.ndarray
    axis_point: np.ndarray | None
    position_tolerance: float
    rotation_tolerance: float
    diagnostics: dict[str, Any]
```

该类型的直接方法：`dimension`、`axis`、`poses`、`_coordinates_unchecked`、`coordinates_of`、`to_dict`。

## `MotionModel.dimension`

[定义位置](../../../manipulation/motion.py#L161)。

输入签名：

```python
@property
def MotionModel.dimension(self) -> int:
```

直接读取的对象属性：`self.kind`。

输出分支的源码表达式：

```python
_DIMENSIONS[self.kind]
```

## `MotionModel.axis`

[定义位置](../../../manipulation/motion.py#L165)。

输入签名：

```python
@property
def MotionModel.axis(self) -> np.ndarray:
```

直接读取的对象属性：`self.basis`。

输出分支的源码表达式：

```python
self.basis[:, 2].copy()
```

直接调用：

```python
self.basis[:, 2].copy
```

## `MotionModel.poses`

[定义位置](../../../manipulation/motion.py#L168)。

输入签名：

```python
def MotionModel.poses(self, coordinates: Any) -> np.ndarray:
```

直接读取的对象属性：`self.dimension`、`self.kind`、`self.reference_pose`、`self.basis`、`self.axis`、`self.axis_point`。

输出分支的源码表达式：

```python
output
```

直接写入的状态或数组位置：`output[:, :3, 3]`、`output[:, :3, :3]`。

直接调用：

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
def MotionModel._coordinates_unchecked(self, poses: np.ndarray) -> np.ndarray:
```

直接读取的对象属性：`self.reference_pose`、`self.kind`、`self.axis`、`self.basis`。

输出分支的源码表达式：

```python
np.column_stack((poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_rotvec()))
(positions @ self.axis)[:, None]
positions @ self.basis[:, :2]
angle[:, None]
np.column_stack((positions @ self.basis[:, :2], angle))
```

直接调用：

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
def MotionModel.coordinates_of(self, poses: Any) -> np.ndarray:
```

直接读取的对象属性：`self._coordinates_unchecked`、`self.poses`、`self.position_tolerance`、`self.kind`、`self.rotation_tolerance`。

输出分支的源码表达式：

```python
coordinates
```

直接调用：

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
def MotionModel.to_dict(self) -> dict[str, Any]:
```

直接读取的对象属性：`self.kind`、`self.dimension`、`self.reference_pose`、`self.basis`、`self.axis_point`、`self.coordinates`、`self.position_tolerance`、`self.rotation_tolerance`、`self.diagnostics`。

输出分支的源码表达式：

```python
{'kind': self.kind, 'dimension': self.dimension, 'reference_pose': self.reference_pose.tolist(), 'basis': self.basis.tolist(), 'axis_point': None if self.axis_point is None else self.axis_point.tolist(), 'coordinates': self.coordinates.tolist(), 'position_tolerance': self.position_tolerance, 'rotation_tolerance': self.rotation_tolerance, 'diagnostics': self.diagnostics, 'pose_semantics': 'fixed object/part frame in reference frame; not end effector'}
```

直接调用：

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

输出分支的源码表达式：

```python
model
```

直接写入的状态或数组位置：`diagnostics['rotation_axis_singular_values']`、`diagnostics['axis_location_singular_values']`、`diagnostics['axis_location_noise_gain']`、`model.coordinates`、`diagnostics['angular_branch']`。

直接调用：

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

输出分支的源码表达式：

```python
(accepted, rejected, excited)
```

直接写入的状态或数组位置：`rejected[candidate]`。

直接调用：

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

输出分支的源码表达式：

```python
{'status': 'observed_compatible' if excited else 'insufficient_excitation', 'preferred_description': accepted[0].kind if excited else None, 'compatible_candidates': [model.kind for model in accepted], 'candidates': {model.kind: model.to_dict() for model in accepted}, 'rejected_candidates': rejected, 'position_tolerance_m': ptol, 'rotation_tolerance_rad': rtol, 'interpretation': 'observed geometry only; task and reference evidence determine applied constraints'}
```

直接调用：

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

输出分支的源码表达式：

```python
_fit_one(observed, kind, ptol, rtol)
selected
```

直接写入的状态或数组位置：`selected.diagnostics['selection']`。

直接调用：

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

输出分支的源码表达式：

```python
{'right': right, 'translation_correction': translation, 'rotation_correction': rotation}
```

直接调用：

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

输出分支的源码表达式：

```python
{'poses': output, 'model': model, 'diagnostics': diagnostics}
```

直接写入的状态或数组位置：`mapped[0]`、`output[:, :3, 3]`、`output[:, :3, :3]`、`goal_coordinates[angular_column]`、`diagnostics['goal_angle_branch']`、`output[0]`、`output[-1]`、`model.coordinates`、`diagnostics['model_fit']`。

直接调用：

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
