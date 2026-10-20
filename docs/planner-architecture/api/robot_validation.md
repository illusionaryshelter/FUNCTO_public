# `manipulation/robot_validation.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_positive_scalar`

[定义位置](../../../manipulation/robot_validation.py#L20)。

输入签名：

```python
def _positive_scalar(value, label):
```

输出分支的源码表达式：

```python
float(value)
```

直接调用：

```python
isinstance
np.isscalar
np.isfinite
ValueError
float
```

## `RobotStateValidator`

[定义位置](../../../manipulation/robot_validation.py#L26)。

```python
class RobotStateValidator:
    ...
    ...
```

该类型的直接方法：`__init__`、`_planning_positions`、`validate_planning_binding`、`check_planning_self`、`check_planning_world`、`planning_binding_record`、`describe`、`_pose_difference`、`check_state`、`preflight`。

## `RobotStateValidator.__init__`

[定义位置](../../../manipulation/robot_validation.py#L43)。

输入签名：

```python
def RobotStateValidator.__init__(self, checker, *, joint_names, arm_joint_names, finger_joint_names, pairs, named_velocity_limits, reference_frame, T_reference_base, tool_frame, T_frame_tcp, fk_translation_tolerance_m, fk_rotation_tolerance_rad, source):
```

直接读取的对象属性：`self.joint_names`、`self.arm_joint_names`、`self.finger_joint_names`、`self.tool_frame`、`self.fk_rotation_tolerance_rad`、`self.velocity_limits`、`self.pairs`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.checker`、`self.joint_names`、`self.arm_joint_names`、`self.finger_joint_names`、`self.reference_frame`、`self.tool_frame`、`self.T_reference_base`、`self.T_frame_tcp`、`self.fk_translation_tolerance_m`、`self.fk_rotation_tolerance_rad`、`self.source`、`self.velocity_limits`、`self.pairs`、`self.required_pairs`。

直接调用：

```python
isinstance
TypeError
tuple
len
set
ValueError
_name
checker._frame_id
_rigid_pose
_positive_scalar
frozenset
combinations
any
```

## `RobotStateValidator._planning_positions`

[定义位置](../../../manipulation/robot_validation.py#L90)。

输入签名：

```python
def RobotStateValidator._planning_positions(self, joint_positions, joint_names, *, time_s, gripper):
```

直接读取的对象属性：`self.joint_names`、`self.finger_joint_names`、`self.arm_joint_names`、`self.checker`。

输出分支的源码表达式：

```python
(np.asarray([named[name] for name in self.joint_names]), unmodelled)
```

直接调用：

```python
tuple
np.asarray
len
set
np.isfinite(values).all
np.isfinite
ValueError
np.isscalar
dict
zip
values.astype
gripper.nominal_finger_trajectory.valid_mask
float
model.sample
sum
self.checker.geometry_depending_on
np.isfinite(fingers).all
named.update
fingers.astype
```

## `RobotStateValidator.validate_planning_binding`

[定义位置](../../../manipulation/robot_validation.py#L150)。

输入签名：

```python
def RobotStateValidator.validate_planning_binding(self, problem, gripper):
```

直接读取的对象属性：`problem.reference_frame`、`self.reference_frame`、`problem.tool_link`、`self.tool_frame`、`problem.base_pose`、`self.T_reference_base`、`problem.tool_transform`、`self.T_frame_tcp`、`self._planning_positions`、`problem.start_q`、`problem.joint_names`、`self.checker`、`self.finger_joint_names`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
np.array_equal
ValueError
self._planning_positions
dict
zip
```

## `RobotStateValidator.check_planning_self`

[定义位置](../../../manipulation/robot_validation.py#L171)。

输入签名：

```python
def RobotStateValidator.check_planning_self(self, joint_positions, joint_names, *, time_s, gripper):
```

直接读取的对象属性：`self._planning_positions`、`self.pairs`、`self.checker`、`self.joint_names`、`self.T_reference_base`。

输出分支的源码表达式：

```python
{'passed': passed, 'residual': float(not passed), 'queried_pairs': len(pairs), 'minimum_required_clearance_m': None if minimum is None else minimum['signed_distance_m'], 'unmodelled_geometry_names': list(unmodelled), 'violations': violations, 'minimum_pair': minimum, 'joint_limit_violations': limits, 'rounded_limit_representations': rounded_limits, 'joint_names': result['joint_names'], 'joint_positions': result['joint_positions']}
```

直接调用：

```python
self._planning_positions
set
tuple
self.checker.query
min
dict
np.asarray
zip
limits.items
rounded_limits.append
str
float
len
list
```

## `RobotStateValidator.check_planning_world`

[定义位置](../../../manipulation/robot_validation.py#L217)。

输入签名：

```python
def RobotStateValidator.check_planning_world(self, joint_positions, joint_names, *, time_s, gripper, scene_checker, progress, nominal_owner_poses, layer_selection, robot_owner_permission_progress):
```

直接读取的对象属性：`self._planning_positions`、`self.pairs`、`self.joint_names`、`self.T_reference_base`、`self.tool_frame`、`self.T_frame_tcp`、`self.finger_joint_names`。

输出分支的源码表达式：

```python
{'residual': float(not scene['passed']), 'scene': scene, 'unmodelled_geometry_names': list(unmodelled), 'joint_names': list(self.joint_names), 'joint_positions': positions.tolist()}
```

直接写入的状态或数组位置：`robot['unmodelled_geometry_names']`。

直接调用：

```python
isinstance
TypeError
self._planning_positions
set
scene_checker._scene_native_aliases.values
tuple
scene_checker.query
np.asarray
self.joint_names.index
scene_checker.query_scene
float
list
positions.tolist
```

## `RobotStateValidator.planning_binding_record`

[定义位置](../../../manipulation/robot_validation.py#L257)。

输入签名：

```python
def RobotStateValidator.planning_binding_record(self):
```

直接读取的对象属性：`self.describe`、`record.pop`、`self.checker`。

输出分支的源码表达式：

```python
record
```

直接写入的状态或数组位置：`record['pair_order']`、`record['model']`、`record['scope']`。

直接调用：

```python
self.describe
record.pop
self.checker.model_record
```

## `RobotStateValidator.describe`

[定义位置](../../../manipulation/robot_validation.py#L268)。

输入签名：

```python
def RobotStateValidator.describe(self):
```

直接读取的对象属性：`self.name`、`self.source`、`self.joint_names`、`self.arm_joint_names`、`self.finger_joint_names`、`self.checker`、`self.reference_frame`、`self.T_reference_base`、`self.tool_frame`、`self.T_frame_tcp`、`self.velocity_limits`、`self.required_pairs`、`self.pairs`、`self.fk_translation_tolerance_m`、`self.fk_rotation_tolerance_rad`。

输出分支的源码表达式：

```python
{'name': self.name, 'source': self.source, 'joint_names': list(self.joint_names), 'arm_joint_names': list(self.arm_joint_names), 'finger_joint_names': list(self.finger_joint_names), 'urdf_path': self.checker.urdf_path, 'base_frame': self.checker.base_frame, 'reference_frame': self.reference_frame, 'T_reference_base': self.T_reference_base.tolist(), 'tool_frame': self.tool_frame, 'T_frame_tcp': self.T_frame_tcp.tolist(), 'joint_position_limits': {name: list(self.checker.joint_limits[name]) for name in self.joint_names}, 'joint_velocity_limits': dict(self.velocity_limits), 'geometry_sources': self.checker.geometry_sources, 'pair_order': [asdict(pair) for pair in self.required_pairs], 'pair_policy': [asdict(pair) for pair in self.pairs], 'numerics': dict(self.checker.numerics), 'fk_model_translation_tolerance_m': self.fk_translation_tolerance_m, 'fk_model_rotation_tolerance_rad': self.fk_rotation_tolerance_rad, 'fk_tolerance_scope': 'Consistency of known robot models/calibration at the same measured joints; even a 1e-8 tolerance is not physical Cartesian sensing accuracy', 'robot_pose_scope': 'Fixed calibrated base and TCP FK estimates from measured joint positions; no independent Cartesian observation', 'scope': 'known robot/installation native pair policy, model FK and joint limits', 'observed_scene_checked': False, 'unknown_space_checked': False, 'continuous_collision_certified': False, 'world_collision_checked': False}
```

直接调用：

```python
list
self.T_reference_base.tolist
self.T_frame_tcp.tolist
dict
asdict
```

## `RobotStateValidator._pose_difference`

[定义位置](../../../manipulation/robot_validation.py#L292)。

输入签名：

```python
def RobotStateValidator._pose_difference(self, actual, expected):
```

输出分支的源码表达式：

```python
{'translation_m': float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])), 'rotation_rad': float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())}
```

直接调用：

```python
float
np.linalg.norm
Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude
Rotation.from_matrix
```

## `RobotStateValidator.check_state`

[定义位置](../../../manipulation/robot_validation.py#L296)。

输入签名：

```python
def RobotStateValidator.check_state(self, feedback, *, stage, trajectory_time):
```

直接读取的对象属性：`self.joint_names`、`self.arm_joint_names`、`self.finger_joint_names`、`self.reference_frame`、`self.tool_frame`、`self.T_frame_tcp`、`self.checker`、`self.T_reference_base`、`self.required_pairs`、`self._pose_difference`、`self.fk_translation_tolerance_m`、`self.fk_rotation_tolerance_rad`、`self.velocity_limits`。

输出分支的源码表达式：

```python
report
```

直接调用：

```python
_name
np.isscalar
np.isfinite
ValueError
tuple
_rigid_pose
np.array_equal
np.asarray
len
np.isfinite(velocities).all
self.checker.query
self._pose_difference
violations.append
result['joint_limit_violations'].items
zip
abs
float
enumerate
deepcopy
min
```

## `RobotStateValidator.preflight`

[定义位置](../../../manipulation/robot_validation.py#L368)。

输入签名：

```python
def RobotStateValidator.preflight(self, trajectory, initial_feedback, execution_limits, *, start_time=0.0, until=None, current_time=None):
```

直接读取的对象属性：`trajectory.joint_names`、`self.arm_joint_names`、`self.joint_names`、`self.finger_joint_names`、`trajectory.duration`、`self.check_state`、`trajectory.times`、`trajectory.position`、`trajectory.velocity`、`trajectory.executable_until`、`trajectory._curve`、`self.checker`、`self.velocity_limits`。

输出分支的源码表达式：

```python
{'passed': not violations and initial['passed'], 'violations': violations, 'scope': 'Known robot geometry at measured entrance, model FK and scalar reference q/dq', 'checked_interval_s': [float(start_time), stop], 'reference_cubic_extrema': extrema, 'actual_state': initial, 'future_finger_states_assumed': False, 'reference_collision_checked': False, 'continuous_collision_certified': False, 'observed_scene_checked': False, 'world_collision_checked': False, 'unknown_space_checked': False}
```

直接写入的状态或数组位置：`row[label]`、`extrema[name]`。

直接调用：

```python
tuple
ValueError
isinstance
np.isscalar
np.isfinite
self.check_state
float
np.asarray
len
np.isfinite(times).all
np.any
np.diff
np.isfinite(positions).all
np.isfinite(velocities).all
CubicHermiteSpline
np.array_equal
deepcopy
np.unique
enumerate
scalar.derivative(order + 1).roots
scalar.derivative
scalar
np.isfinite(values).all
int
np.argmin
np.argmax
violations.append
abs
```
