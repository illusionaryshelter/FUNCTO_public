# `manipulation/robot_validation.py` 实现契约

代码基线：`22e6bda65682c1584d89623320024795fc1efa4e`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_positive_scalar`

[定义位置](../../../manipulation/robot_validation.py#L20)。

输入签名：

```python
def _positive_scalar(value, label):
```

显式输出分支：

```python
float(value)
```

读取的对象属性：

```python
np.isscalar
np.isfinite
```

写入目标：

无显式条目。

直接调用符号：

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
class RobotStateValidator():
```

类字段：

```python
name = 'pinocchio_coal_robot_state'
```

实现说明：

> Known robot entities, explicit mechanical policy and measured-state FK.
>
> Required pairs need positive native separation. A numerical boundary is
> reported separately from penetration. Excluded and mechanical-contact
> declarations do not waive joint limits or model-identity checks.

## `RobotStateValidator.__init__`

[定义位置](../../../manipulation/robot_validation.py#L36)。

输入签名：

```python
def __init__(self, checker, *, joint_names, arm_joint_names, finger_joint_names, pairs, named_velocity_limits, reference_frame, T_reference_base, tool_frame, T_frame_tcp, fk_translation_tolerance_m, fk_rotation_tolerance_rad, source):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.joint_names
self.arm_joint_names
self.finger_joint_names
checker.joint_names
checker._frame_id
self.tool_frame
self.fk_rotation_tolerance_rad
np.pi
self.velocity_limits
checker.joint_velocity_limits
self.pairs
pair.policy
checker.geometry_names
pair.first
pair.second
pair.source
```

写入目标：

```python
self.checker
self.joint_names
self.arm_joint_names
self.finger_joint_names
self.reference_frame
self.tool_frame
self.T_reference_base
self.T_frame_tcp
self.fk_translation_tolerance_m
self.fk_rotation_tolerance_rad
self.source
self.velocity_limits
name
self.pairs
self.required_pairs
pair
expected
supplied
```

直接调用符号：

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

[定义位置](../../../manipulation/robot_validation.py#L83)。

输入签名：

```python
def _planning_positions(self, joint_positions, joint_names, *, time_s, gripper):
```

实现说明：

> Complete only declared coordinates; identify unknown descendants.
>
> Pinocchio requires the complete tree coordinates even when an actuator
> has no future geometry model. Interior placeholders affect only that
> actuator's descendants, which are explicitly excluded from acceptance.

显式输出分支：

```python
(np.asarray([named[name] for name in self.joint_names]), unmodelled)
```

读取的对象属性：

```python
np.asarray
self.joint_names
values.shape
values.dtype.kind
values.dtype
np.isfinite(values).all
np.isfinite
np.isscalar
values.astype
gripper.joint_names
self.finger_joint_names
gripper.controller
self.arm_joint_names
gripper.current_positions
gripper.nominal_finger_trajectory
gripper.nominal_finger_trajectory.valid_mask
model.joint_names
np.asarray(time_s).dtype
time_dtype.kind
model.duration
model.valid_until_s
model.sample
self.checker.joint_limits
self.checker
self.checker.geometry_depending_on
fingers.shape
fingers.dtype.kind
fingers.dtype
np.isfinite(fingers).all
named.update
fingers.astype
```

写入目标：

```python
names
values
named
unmodelled
fingers
model
time_dtype
sample_time
_
name
```

直接调用符号：

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

[定义位置](../../../manipulation/robot_validation.py#L143)。

输入签名：

```python
def validate_planning_binding(self, problem, gripper):
```

实现说明：

> Bind model identity and measured entrance before allocating solvers.
>
> Collision feasibility remains the entrance gate's responsibility. This
> checks the source model, calibration and coordinate partition without
> reconstructing geometry or replacing the supplied measured entrance.

显式输出分支：

无显式条目。

读取的对象属性：

```python
problem.reference_frame
self.reference_frame
problem.tool_link
self.tool_frame
np.array_equal
problem.base_pose
self.T_reference_base
problem.tool_transform
self.T_frame_tcp
self._planning_positions
problem.start_q
problem.joint_names
gripper.joint_limits
self.checker.joint_limits
self.checker
self.finger_joint_names
gripper.controller
gripper.current_positions
```

写入目标：

```python
name
named
```

直接调用符号：

```python
np.array_equal
ValueError
self._planning_positions
dict
zip
```

## `RobotStateValidator.check_planning_self`

[定义位置](../../../manipulation/robot_validation.py#L165)。

输入签名：

```python
def check_planning_self(self, joint_positions, joint_names, *, time_s, gripper):
```

实现说明：

> Check all known required native pairs at this one physical instant.
>
> Coal evaluates the declared native collision entities. ``passed``
> covers only known
> geometry; an unmodelled future hand is reported explicitly. Scalar
> joint limits still apply even when mechanical contact is permitted.

显式输出分支：

```python
{'passed': passed, 'residual': float(not passed), 'queried_pairs': len(pairs), 'minimum_required_clearance_m': None if minimum is None else minimum['signed_distance_m'], 'unmodelled_geometry_names': list(unmodelled), 'violations': violations, 'minimum_pair': minimum, 'joint_limit_violations': limits, 'rounded_limit_representations': rounded_limits, 'joint_names': result['joint_names'], 'joint_positions': result['joint_positions']}
```

读取的对象属性：

```python
self._planning_positions
self.pairs
pair.policy
pair.first
pair.second
self.checker.query
self.checker
self.joint_names
self.T_reference_base
np.asarray
active.dtype.kind
active.dtype
limits.items
rounded_limits.append
```

写入目标：

```python
positions
unmodelled
unknown
pairs
pair
result
violations
row
minimum
limits
rounded_limits
active
active_values
name
boundary
passed
```

直接调用符号：

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

## `RobotStateValidator.planning_binding_record`

[定义位置](../../../manipulation/robot_validation.py#L212)。

输入签名：

```python
def planning_binding_record(self):
```

实现说明：

> Freeze the same source geometry, complete pair policy and numerics.

显式输出分支：

```python
record
```

读取的对象属性：

```python
self.describe
record.pop
self.checker.model_record
self.checker
```

写入目标：

```python
record
record['pair_order']
record['model']
record['scope']
```

直接调用符号：

```python
self.describe
record.pop
self.checker.model_record
```

## `RobotStateValidator.describe`

[定义位置](../../../manipulation/robot_validation.py#L225)。

输入签名：

```python
def describe(self):
```

实现说明：

> Static report defines the order of every per-state distance vector.

显式输出分支：

```python
{'name': self.name, 'source': self.source, 'joint_names': list(self.joint_names), 'arm_joint_names': list(self.arm_joint_names), 'finger_joint_names': list(self.finger_joint_names), 'urdf_path': self.checker.urdf_path, 'base_frame': self.checker.base_frame, 'reference_frame': self.reference_frame, 'T_reference_base': self.T_reference_base.tolist(), 'tool_frame': self.tool_frame, 'T_frame_tcp': self.T_frame_tcp.tolist(), 'joint_position_limits': {name: list(self.checker.joint_limits[name]) for name in self.joint_names}, 'joint_velocity_limits': dict(self.velocity_limits), 'geometry_sources': self.checker.geometry_sources, 'pair_order': [asdict(pair) for pair in self.required_pairs], 'pair_policy': [asdict(pair) for pair in self.pairs], 'numerics': dict(self.checker.numerics), 'fk_model_translation_tolerance_m': self.fk_translation_tolerance_m, 'fk_model_rotation_tolerance_rad': self.fk_rotation_tolerance_rad, 'fk_tolerance_scope': 'Consistency of known robot models/calibration at the same measured joints; even a 1e-8 tolerance is not physical Cartesian sensing accuracy', 'robot_pose_scope': 'Fixed calibrated base and TCP FK estimates from measured joint positions; no independent Cartesian observation', 'scope': 'known robot/installation native pair policy, model FK and joint limits', 'observed_scene_checked': False, 'unknown_space_checked': False, 'continuous_collision_certified': False, 'world_collision_checked': False}
```

读取的对象属性：

```python
self.name
self.source
self.joint_names
self.arm_joint_names
self.finger_joint_names
self.checker.urdf_path
self.checker
self.checker.base_frame
self.reference_frame
self.T_reference_base.tolist
self.T_reference_base
self.tool_frame
self.T_frame_tcp.tolist
self.T_frame_tcp
self.checker.joint_limits
self.velocity_limits
self.checker.geometry_sources
self.required_pairs
self.pairs
self.checker.numerics
self.fk_translation_tolerance_m
self.fk_rotation_tolerance_rad
```

写入目标：

```python
name
pair
```

直接调用符号：

```python
list
self.T_reference_base.tolist
self.T_frame_tcp.tolist
dict
asdict
```

## `RobotStateValidator._pose_difference`

[定义位置](../../../manipulation/robot_validation.py#L250)。

输入签名：

```python
def _pose_difference(self, actual, expected):
```

显式输出分支：

```python
{'translation_m': float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])), 'rotation_rad': float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())}
```

读取的对象属性：

```python
np.linalg.norm
np.linalg
Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude
Rotation.from_matrix
expected[:3, :3].T
```

写入目标：

无显式条目。

直接调用符号：

```python
float
np.linalg.norm
Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude
Rotation.from_matrix
```

## `RobotStateValidator.check_state`

[定义位置](../../../manipulation/robot_validation.py#L255)。

输入签名：

```python
def check_state(self, feedback, *, stage, trajectory_time):
```

实现说明：

> Gate full actual q, after the executor validates identity and time.
>
> Clock freshness and physical tracking use the execution layer's own
> # BEGIN independent models do not manufacture independent measurements.
> guard. This independently checks model-frame identity, joint limits,
> robot solid separation and backend FK versus Pinocchio FK at the same
> measured q. Fixed-base/TCP consistency tolerances, including 1e-8 values,
> are numerical/model requirements, not sensor-accuracy certificates.
> # END original physical tracking limits remain owned by ExecutionLimits.
> No actuator command is read by this function.

显式输出分支：

```python
report
```

读取的对象属性：

```python
np.isscalar
np.isfinite
self.joint_names
self.arm_joint_names
self.finger_joint_names
self.reference_frame
self.tool_frame
np.array_equal
self.T_frame_tcp
np.asarray
velocities.shape
velocities.dtype.kind
velocities.dtype
np.isfinite(velocities).all
self.checker.query
self.checker
self.T_reference_base
self.required_pairs
self._pose_difference
self.fk_translation_tolerance_m
self.fk_rotation_tolerance_rad
violations.append
result['joint_limit_violations'].items
self.velocity_limits
```

写入目标：

```python
key
expected
tool_transform
base
estimated_tool
velocities
result
expected_tool
base_error
tool_error
violations
name
error
metric
maximum
row
velocity
index
required
report
```

直接调用符号：

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

[定义位置](../../../manipulation/robot_validation.py#L330)。

输入签名：

```python
def preflight(self, trajectory, initial_feedback, execution_limits, *, start_time=0.0, until=None, current_time=None):
```

实现说明：

> Check actual entrance and reference cubic q/dq over its whole prefix.
>
> Each scalar cubic's exact polynomial derivative supplies its critical
> times through SciPy's maintained root solver. Values are evaluated in
> binary64; this is a polynomial range check, not formal exact arithmetic.
> It certifies neither future finger positions nor continuous collision.
> The original evidence barrier defines the end of the checked prefix.

显式输出分支：

```python
{'passed': not violations and initial['passed'], 'violations': violations, 'scope': 'Known robot geometry at measured entrance, model FK and scalar reference q/dq', 'checked_interval_s': [float(start_time), stop], 'reference_cubic_extrema': extrema, 'actual_state': initial, 'future_finger_states_assumed': False, 'reference_collision_checked': False, 'continuous_collision_certified': False, 'observed_scene_checked': False, 'world_collision_checked': False, 'unknown_space_checked': False}
```

读取的对象属性：

```python
trajectory.joint_names
self.arm_joint_names
execution_limits.joint_names
self.joint_names
execution_limits.arm_joint_names
execution_limits.finger_joint_names
self.finger_joint_names
np.isscalar
np.isfinite
trajectory.duration
self.check_state
np.asarray
trajectory.times
trajectory.position
trajectory.velocity
times.ndim
np.isfinite(times).all
np.any
np.diff
positions.shape
velocities.shape
np.isfinite(positions).all
np.isfinite(velocities).all
trajectory.executable_until
np.array_equal
curve.x
trajectory._curve.x
trajectory._curve
curve.c
trajectory._curve.c
np.unique
np.r_
scalar.derivative(order + 1).roots
scalar.derivative
np.isfinite(values).all
np.argmin
np.argmax
self.checker.joint_limits
self.checker
violations.append
self.velocity_limits
```

写入目标：

```python
current_time
initial
times
positions
velocities
expected_shape
stop
curve
violations
extrema
endpoints
index
name
scalar
row
order
label
roots
candidates
values
lower_index
upper_index
row[label]
lower
upper
bound
expected
value
extrema[name]
```

直接调用符号：

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
