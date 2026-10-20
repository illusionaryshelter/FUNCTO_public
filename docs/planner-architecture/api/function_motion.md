# `manipulation/function_motion.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `FunctionReferenceError`

[定义位置](../../../manipulation/function_motion.py#L24)。

```python
class FunctionReferenceError(ValueError):
    ...
    ...
```

## `_pose_pair`

[定义位置](../../../manipulation/function_motion.py#L29)。

输入签名：

```python
def _pose_pair(value, name):
```

输出分支的源码表达式：

```python
value.copy()
```

直接调用：

```python
np.asarray
np.isfinite(value).all
np.isfinite
ValueError
value.copy
```

## `_keyframes`

[定义位置](../../../manipulation/function_motion.py#L36)。

输入签名：

```python
def _keyframes(function_keyframes, n_points):
```

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
sorted(result, key=lambda item: item['index'])
```

直接调用：

```python
isinstance
ValueError
set
value.get
int
seen.add
result.append
_pose_pair
sorted
```

## `optimize_trajectory`

[定义位置](../../../manipulation/function_motion.py#L56)。

输入签名：

```python
def optimize_trajectory(reference_trajectory, static_pose, function_keyframes):
```

输出分支的源码表达式：

```python
np.stack((solution.value(positions), solution.value(euler_angles)), axis=1)
```

直接调用：

```python
np.asarray
len
np.isfinite(reference).all
np.isfinite
ValueError
_pose_pair
_keyframes
ca.Opti
opti.variable
range
ca.sumsqr
np.array(ref_pos).reshape
np.array
ca.fmod
np.array(ref_euler).reshape
opti.subject_to
np.array(static_pos).reshape
np.array(static_ori).reshape
np.array(function_pos).reshape
np.array(function_ori).reshape
opti.minimize
opti.solver
opti.solve
FunctionReferenceError
np.stack
solution.value
```

## `transfer_function_motion`

[定义位置](../../../manipulation/function_motion.py#L123)。

输入签名：

```python
def transfer_function_motion(reference_poses, initial_pose, function_goals):
```

输出分支的源码表达式：

```python
{'status': 'candidate', 'poses': output, 'goal_checks': checks[1:], 'diagnostics': {'initial_check': checks[0], 'kernel': 'released_function_origin_xyz_euler_objective', 'angular_constraint_flag': 2, 'solve_attempts': 1, 'function_event_indices': [event['index'] for event in normalized], 'translation_increment_limit_m': 0.01, 'angular_increment_limits_rad': angular_limits, 'max_translation_increment_m': float(steps.max()), 'max_translation_increment_violation_m': translation_violation, 'max_angular_increment_violation_rad': angular_violation, 'translation_increment_check_tolerance_m': 1e-06, 'angular_increment_check_tolerance_rad': 1e-06, 'frame_origin': 'supplied fixed function reference; never substituted with carrier origin', 'scope': 'saved-node function-frame geometry; increments are not physical velocities; no complete-scene collision, IK, force, stability, or timing guarantee'}}
```

直接写入的状态或数组位置：`output[:, :3, 3]`、`output[:, :3, :3]`。

直接调用：

```python
_poses_array
_pose_array
len
ValueError
isinstance
pose_goals.append
event.get
raw_goals.append
Rotation.from_matrix(goal[:3, :3]).as_euler
Rotation.from_matrix
_keyframes
pose_goals.sort
np.stack
Rotation.from_matrix(reference[:, :3, :3]).as_euler
Rotation.from_matrix(initial[:3, :3]).as_euler
np.where
np.arange
np.linalg.norm
angular_limits[previous_index:index].sum
FunctionReferenceError
optimize_trajectory
np.asarray
np.isfinite(raw).all
np.isfinite
np.repeat
np.eye
Rotation.from_euler('xyz', raw[:, 1]).as_matrix
Rotation.from_euler
_errors
int
float
checks.append
np.diff
np.fmod
max
(steps - 0.01).max
(euler_steps - angular_limits).max
steps.max
```
