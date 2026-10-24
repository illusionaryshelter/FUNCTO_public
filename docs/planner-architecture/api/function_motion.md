# `manipulation/function_motion.py` 实现契约

代码基线：`13fefbc`。本次文档引用提交仅回填实际实现基线，源码与该代码提交相同。组件正文描述当前实现；阶段验证位于项目外 `sim-version-plans`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `FunctionReferenceError`

[定义位置](../../../manipulation/function_motion.py#L24)。

```python
class FunctionReferenceError(ValueError):
```

类字段：

无显式条目。

实现说明：

> The sampled reference kernel failed; this is not task infeasibility.

## `_pose_pair`

[定义位置](../../../manipulation/function_motion.py#L29)。

输入签名：

```python
def _pose_pair(value, name):
```

显式输出分支：

```python
value.copy()
```

读取的对象属性：

```python
np.asarray
value.shape
np.isfinite(value).all
np.isfinite
value.copy
```

写入目标：

```python
value
```

直接调用符号：

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

实现说明：

> Validate the single explicit event-list contract in the numerical kernel.

显式输出分支：

```python
sorted(result, key=lambda item: item['index'])
```

读取的对象属性：

```python
value.get
np.bool_
np.integer
seen.add
result.append
```

写入目标：

```python
result
seen
value
index
```

直接调用符号：

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

实现说明：

> Preserve the original sampled objective and equality constraints.
>
> ``function_keyframes=[{'index': i, 'pose': (position, xyz_angles)}, ...]``
> supplies every hard function event. Physical contact is not required
> (pouring is an example). The trajectory length is taken from the supplied
> reference, and the original strict angular increment limits are fixed.
> This function performs one solve and returns an (N,2,3) pose-pair array.
> Solver failures propagate; there is no weaker constraint mode or retry.

显式输出分支：

```python
np.stack((solution.value(positions), solution.value(euler_angles)), axis=1)
```

读取的对象属性：

```python
np.asarray
reference.ndim
reference.shape
np.isfinite(reference).all
np.isfinite
ca.Opti
opti.variable
ca.sumsqr
np.array(ref_pos).reshape
np.array
ca.fmod
np.array(ref_euler).reshape
np.pi
opti.subject_to
np.array(static_pos).reshape
np.array(static_ori).reshape
np.array(function_pos).reshape
np.array(function_ori).reshape
opti.minimize
opti.solver
opti.solve
np.stack
solution.value
```

写入目标：

```python
reference
n_points
static
events
opti
static_pos
static_ori
positions
euler_angles
cost
t
ref_pos
ref_euler
step_cost
event
function_pos
function_ori
solution
```

直接调用符号：

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

实现说明：

> Optimize explicit function frames with one or more hard function events.
>
> ``reference_poses`` is (N,4,4) and ``initial_pose`` is (4,4), all in one
> grounded reference frame. ``function_goals`` contains ``{'index': i,
> 'pose': T_reference_function}``; indices are array rows, after the initial
> node and before N. The caller maps original event frame IDs to these rows
> without replacing source timestamps or dropping intermediate events.
>
> All events are hard equalities in one solve with the strict original limits. The
> returned independent SE(3) checks compare against the supplied goals, not
> against redefined goals derived from the optimizer's output. Passing these
> checks establishes geometry at saved nodes, not task/robot success.

显式输出分支：

```python
{'status': 'candidate', 'poses': output, 'goal_checks': checks[1:], 'diagnostics': {'initial_check': checks[0], 'kernel': 'released_function_origin_xyz_euler_objective', 'angular_constraint_flag': 2, 'solve_attempts': 1, 'function_event_indices': [event['index'] for event in normalized], 'translation_increment_limit_m': 0.01, 'angular_increment_limits_rad': angular_limits, 'max_translation_increment_m': float(steps.max()), 'max_translation_increment_violation_m': translation_violation, 'max_angular_increment_violation_rad': angular_violation, 'translation_increment_check_tolerance_m': 1e-06, 'angular_increment_check_tolerance_rad': 1e-06, 'frame_origin': 'supplied fixed function reference; never substituted with carrier origin', 'scope': 'saved-node function-frame geometry; increments are not physical velocities; no complete-scene collision, IK, force, stability, or timing guarantee'}}
```

读取的对象属性：

```python
pose_goals.append
event.get
raw_goals.append
Rotation.from_matrix(goal[:3, :3]).as_euler
Rotation.from_matrix
pose_goals.sort
np.stack
Rotation.from_matrix(reference[:, :3, :3]).as_euler
Rotation.from_matrix(initial[:3, :3]).as_euler
np.where
np.arange
np.pi
np.linalg.norm
np.linalg
angular_limits[previous_index:index].sum
np.asarray
raw.shape
np.isfinite(raw).all
np.isfinite
np.repeat
np.eye
Rotation.from_euler('xyz', raw[:, 1]).as_matrix
Rotation.from_euler
checks.append
np.diff
np.fmod
(steps - 0.01).max
(euler_steps - angular_limits).max
steps.max
```

写入目标：

```python
reference
initial
pose_goals
raw_goals
event
goal
normalized
reference_pairs
initial_pair
angular_limits
previous_index
previous_pose
index
pair
distance
angle
raw
output
output[:, :3, 3]
output[:, :3, :3]
checks
item
position
rotation
check
steps
euler_steps
translation_violation
angular_violation
```

直接调用符号：

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
int
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
float
checks.append
np.diff
np.fmod
max
(steps - 0.01).max
(euler_steps - angular_limits).max
steps.max
```
