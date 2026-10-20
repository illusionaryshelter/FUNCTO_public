# `manipulation/robosuite_execution.py` 实现契约

代码基线：`pending-implementation-commit`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `RobosuiteExecutionBackend`

[定义位置](../../../manipulation/robosuite_execution.py#L23)。

```python
class RobosuiteExecutionBackend():
```

类字段：

无显式条目。

实现说明：

> One named robot control binding in an existing native simulation.

## `RobosuiteExecutionBackend.__init__`

[定义位置](../../../manipulation/robosuite_execution.py#L26)。

输入签名：

```python
def __init__(self, sim, *, base_link, tcp_site, arm_joint_names, finger_joint_names, arm_kp, arm_kd, velocity_limits, source):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
gains.shape
np.isfinite(gains).all
np.isfinite
np.any
source.strip
name.strip
sim.model._model
sim.model
sim.data._data
sim.data
self.arm_joint_names
self.finger_joint_names
self.joint_names
speed.tolist
self.native_model.body(base_link).id
self.native_model.body
self.native_model
self.native_model.site(tcp_site).id
self.native_model.site
self.native_model.joint(name).id
self.native_model.joint
self.native_model.jnt_limited[self._joint_ids].all
self.native_model.jnt_limited
self._joint_ids
np.isin(self.native_model.jnt_type[self._joint_ids], [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]).all
np.isin
self.native_model.jnt_type
mujoco.mjtJoint.mjJNT_HINGE
mujoco.mjtJoint
mujoco.mjtJoint.mjJNT_SLIDE
self.native_model.jnt_qposadr[self._joint_ids].copy
self.native_model.jnt_qposadr
self.native_model.jnt_dofadr[self._joint_ids].copy
self.native_model.jnt_dofadr
np.flatnonzero
self.native_model.actuator_trntype
mujoco.mjtTrn.mjTRN_JOINT
mujoco.mjtTrn
self.native_model.actuator_trnid
actuator_ids.append
self.native_model.actuator(index).name
self.native_model.actuator
self._actuator_names
self._actuator_ids
model.actuator_ctrllimited[self._actuator_ids].all
model.actuator_ctrllimited
np.isfinite(model.actuator_ctrlrange[self._actuator_ids]).all
model.actuator_ctrlrange
model.actuator_gear
self._arm_actuators
model.actuator_gaintype
mujoco.mjtGain.mjGAIN_FIXED
mujoco.mjtGain
model.actuator_dyntype
mujoco.mjtDyn.mjDYN_NONE
mujoco.mjtDyn
model.actuator_gainprm
model.actuator_biastype
mujoco.mjtBias.mjBIAS_NONE
mujoco.mjtBias
self._finger_actuators
mujoco.mjtBias.mjBIAS_AFFINE
model.actuator_biasprm
model.opt.timestep
model.opt
self._dt
self._sim
self.native_data.ctrl[self._finger_actuators].copy
self.native_data.ctrl
self.native_data
np.isfinite(restored_ctrl).all
self._grip_controller.set_goal
self._grip_controller
restored_ctrl.copy
np.array_equal
self._grip_controller.run_controller
self._prepare_reference_fk
self._trial_id
self.reference_frame
self.tool_link
self.tool_transform.tolist
self.tool_transform
self.actuator_names
self.timestep
model.actuator_ctrlrange[self._actuator_ids].tolist
model.actuator_forcelimited[self._actuator_ids].astype(bool).tolist
model.actuator_forcelimited[self._actuator_ids].astype
model.actuator_forcelimited
model.actuator_forcerange[self._actuator_ids].tolist
model.actuator_forcerange
self._arm_controller.kp.tolist
self._arm_controller.kp
self._arm_controller
self._arm_controller.kd.tolist
self._arm_controller.kd
restored_ctrl.tolist
```

写入目标：

```python
gains
name
names
self._sim
self.native_model
self.native_data
self._trial_id
self.arm_joint_names
self.finger_joint_names
self.joint_names
speed
self._velocity_limits
self._base_id
self._site_id
self._joint_ids
self._qpos_ids
self._qvel_ids
actuator_ids
joint_id
matches
self._actuator_ids
self._actuator_names
index
self._feedback_source
model
narm
self._arm_actuators
self._finger_actuators
finger_ids
finger_gain
self._dt
self._step_attempts
self._last_step_accounting
bounds
self._arm_controller
restored_ctrl
widths
self._grip_controller
self._description
```

直接调用符号：

```python
np.asarray
np.isfinite(gains).all
np.isfinite
np.any
ValueError
isinstance
source.strip
any
name.strip
str
uuid4
tuple
len
set
_finite_array
dict
zip
speed.tolist
self.native_model.body
self.native_model.site
self.native_model.joint
self.native_model.jnt_limited[self._joint_ids].all
np.isin(self.native_model.jnt_type[self._joint_ids], [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]).all
np.isin
self.native_model.jnt_qposadr[self._joint_ids].copy
self.native_model.jnt_dofadr[self._joint_ids].copy
np.flatnonzero
actuator_ids.append
int
self.native_model.actuator
model.actuator_ctrllimited[self._actuator_ids].all
np.isfinite(model.actuator_ctrlrange[self._actuator_ids]).all
float
JointPositionController
indexes
self.native_data.ctrl[self._finger_actuators].copy
np.isfinite(restored_ctrl).all
SimpleGripController
self._grip_controller.set_goal
restored_ctrl.copy
np.array_equal
self._grip_controller.run_controller
self._prepare_reference_fk
list
self.tool_transform.tolist
model.actuator_ctrlrange[self._actuator_ids].tolist
model.actuator_forcelimited[self._actuator_ids].astype(bool).tolist
model.actuator_forcelimited[self._actuator_ids].astype
model.actuator_forcerange[self._actuator_ids].tolist
self._arm_controller.kp.tolist
self._arm_controller.kd.tolist
restored_ctrl.tolist
```

## `RobosuiteExecutionBackend.__init__.indexes`

[定义位置](../../../manipulation/robosuite_execution.py#L118)。

输入签名：

```python
def indexes(start, stop):
```

显式输出分支：

```python
{'joints': self._joint_ids[start:stop].tolist(), 'qpos': self._qpos_ids[start:stop].tolist(), 'qvel': self._qvel_ids[start:stop].tolist(), 'actuators': self._actuator_ids[start:stop].tolist()}
```

读取的对象属性：

```python
self._joint_ids[start:stop].tolist
self._joint_ids
self._qpos_ids[start:stop].tolist
self._qpos_ids
self._qvel_ids[start:stop].tolist
self._qvel_ids
self._actuator_ids[start:stop].tolist
self._actuator_ids
```

写入目标：

无显式条目。

直接调用符号：

```python
self._joint_ids[start:stop].tolist
self._qpos_ids[start:stop].tolist
self._qvel_ids[start:stop].tolist
self._actuator_ids[start:stop].tolist
```

## `RobosuiteExecutionBackend.timestep`

[定义位置](../../../manipulation/robosuite_execution.py#L195)。

输入签名：

```python
def timestep(self):
```

显式输出分支：

```python
self._dt
```

读取的对象属性：

```python
self._dt
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `RobosuiteExecutionBackend.sim_time`

[定义位置](../../../manipulation/robosuite_execution.py#L199)。

输入签名：

```python
def sim_time(self):
```

显式输出分支：

```python
float(self.native_data.time)
```

读取的对象属性：

```python
self.native_data.time
self.native_data
```

写入目标：

无显式条目。

直接调用符号：

```python
float
```

## `RobosuiteExecutionBackend.trial_id`

[定义位置](../../../manipulation/robosuite_execution.py#L203)。

输入签名：

```python
def trial_id(self):
```

显式输出分支：

```python
self._trial_id
```

读取的对象属性：

```python
self._trial_id
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `RobosuiteExecutionBackend.feedback_source`

[定义位置](../../../manipulation/robosuite_execution.py#L207)。

输入签名：

```python
def feedback_source(self):
```

显式输出分支：

```python
self._feedback_source
```

读取的对象属性：

```python
self._feedback_source
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `RobosuiteExecutionBackend.feedback_id`

[定义位置](../../../manipulation/robosuite_execution.py#L211)。

输入签名：

```python
def feedback_id(self):
```

显式输出分支：

```python
f'{self._trial_id}@native_time={self.sim_time:.17g}'
```

读取的对象属性：

```python
self._trial_id
self.sim_time
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `RobosuiteExecutionBackend.reference_frame`

[定义位置](../../../manipulation/robosuite_execution.py#L215)。

输入签名：

```python
def reference_frame(self):
```

显式输出分支：

```python
'world'
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `RobosuiteExecutionBackend.tool_link`

[定义位置](../../../manipulation/robosuite_execution.py#L222)。

输入签名：

```python
def tool_link(self):
```

显式输出分支：

```python
self.native_model.site(self._site_id).name
```

读取的对象属性：

```python
self.native_model.site(self._site_id).name
self.native_model.site
self.native_model
self._site_id
```

写入目标：

无显式条目。

直接调用符号：

```python
self.native_model.site
```

## `RobosuiteExecutionBackend.tool_transform`

[定义位置](../../../manipulation/robosuite_execution.py#L229)。

输入签名：

```python
def tool_transform(self):
```

显式输出分支：

```python
np.eye(4)
```

读取的对象属性：

```python
np.eye
```

写入目标：

无显式条目。

直接调用符号：

```python
np.eye
```

## `RobosuiteExecutionBackend.actuator_names`

[定义位置](../../../manipulation/robosuite_execution.py#L233)。

输入签名：

```python
def actuator_names(self):
```

显式输出分支：

```python
self._actuator_names
```

读取的对象属性：

```python
self._actuator_names
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `RobosuiteExecutionBackend.joint_ranges`

[定义位置](../../../manipulation/robosuite_execution.py#L237)。

输入签名：

```python
def joint_ranges(self):
```

显式输出分支：

```python
self.native_model.jnt_range[self._joint_ids].copy()
```

读取的对象属性：

```python
self.native_model.jnt_range[self._joint_ids].copy
self.native_model.jnt_range
self.native_model
self._joint_ids
```

写入目标：

无显式条目。

直接调用符号：

```python
self.native_model.jnt_range[self._joint_ids].copy
```

## `RobosuiteExecutionBackend.velocity_limits`

[定义位置](../../../manipulation/robosuite_execution.py#L241)。

输入签名：

```python
def velocity_limits(self):
```

显式输出分支：

```python
dict(self._velocity_limits)
```

读取的对象属性：

```python
self._velocity_limits
```

写入目标：

无显式条目。

直接调用符号：

```python
dict
```

## `RobosuiteExecutionBackend.finger_ctrlrange`

[定义位置](../../../manipulation/robosuite_execution.py#L245)。

输入签名：

```python
def finger_ctrlrange(self):
```

显式输出分支：

```python
self.native_model.actuator_ctrlrange[self._finger_actuators].copy()
```

读取的对象属性：

```python
self.native_model.actuator_ctrlrange[self._finger_actuators].copy
self.native_model.actuator_ctrlrange
self.native_model
self._finger_actuators
```

写入目标：

无显式条目。

直接调用符号：

```python
self.native_model.actuator_ctrlrange[self._finger_actuators].copy
```

## `RobosuiteExecutionBackend.finger_control`

[定义位置](../../../manipulation/robosuite_execution.py#L249)。

输入签名：

```python
def finger_control(self):
```

实现说明：

> Read the current native command, not measured finger position.

显式输出分支：

```python
self.native_data.ctrl[self._finger_actuators].copy()
```

读取的对象属性：

```python
self.native_data.ctrl[self._finger_actuators].copy
self.native_data.ctrl
self.native_data
self._finger_actuators
```

写入目标：

无显式条目。

直接调用符号：

```python
self.native_data.ctrl[self._finger_actuators].copy
```

## `RobosuiteExecutionBackend.actuator_control`

[定义位置](../../../manipulation/robosuite_execution.py#L254)。

输入签名：

```python
def actuator_control(self):
```

显式输出分支：

```python
self.native_data.ctrl[self._actuator_ids].copy()
```

读取的对象属性：

```python
self.native_data.ctrl[self._actuator_ids].copy
self.native_data.ctrl
self.native_data
self._actuator_ids
```

写入目标：

无显式条目。

直接调用符号：

```python
self.native_data.ctrl[self._actuator_ids].copy
```

## `RobosuiteExecutionBackend.describe`

[定义位置](../../../manipulation/robosuite_execution.py#L257)。

输入签名：

```python
def describe(self):
```

显式输出分支：

```python
deepcopy(self._description)
```

读取的对象属性：

```python
self._description
```

写入目标：

无显式条目。

直接调用符号：

```python
deepcopy
```

## `RobosuiteExecutionBackend.apply_control`

[定义位置](../../../manipulation/robosuite_execution.py#L261)。

输入签名：

```python
def apply_control(self, arm_q, arm_dq, arm_ddq, finger_positions):
```

实现说明：

> Apply one named q/dq/ddq reference and native finger positions.
>
> Every arm vector follows ``arm_joint_names`` from this binding. The
> executor samples all three at its declared clamped step endpoint;
> none is substituted by measured velocity or finite differences here.
> Installed robosuite supplies the position/damping torque, mass matrix
> and bias at the same current state. Adding M*(Kd*dq_ref + ddq_ref)
> yields M*(ddq_ref + Kp*(q_ref-q) + Kd*(dq_ref-dq)) + bias, without
> changing gains or canceling unknown contact/passive forces.
>
> This method writes only actuator ctrl and does not integrate physics.
> Its raw torque components and references belong to the same call;
> saturation is applied only to their total, never to each component.

显式输出分支：

```python
{'actuator_ctrl': self.actuator_control, 'requested_arm_torque': torque.copy(), 'arm_torque_saturated': bounded != torque, 'arm_control_saturated_coordinates': int(np.count_nonzero(bounded != torque)), 'reference_arm_q': arm_q.copy(), 'reference_arm_dq': arm_dq.copy(), 'reference_arm_ddq': arm_ddq.copy(), 'arm_position_feedback_torque': position_feedback.copy(), 'arm_velocity_feedback_torque': velocity_feedback.copy(), 'arm_velocity_feedforward_torque': velocity_feedforward.copy(), 'arm_acceleration_feedforward_torque': acceleration_feedforward.copy(), 'arm_bias_torque': bias.copy()}
```

读取的对象属性：

```python
self.arm_joint_names
self.finger_joint_names
self.finger_ctrlrange
np.any
self._grip_controller.set_goal
self._grip_controller
finger_positions.copy
self._arm_controller.set_goal
self._arm_controller
arm_q.copy
controller.run_controller
controller.mass_matrix
controller.kp
controller.kd
controller.joint_pos
controller.joint_vel
controller.torque_compensation
self.native_model.actuator_ctrlrange
self.native_model
self._arm_actuators
np.clip
np.asarray
self._grip_controller.run_controller
np.array_equal
self.native_data.ctrl
self.native_data
self._finger_actuators
self.actuator_control
torque.copy
np.count_nonzero
arm_dq.copy
arm_ddq.copy
position_feedback.copy
velocity_feedback.copy
velocity_feedforward.copy
acceleration_feedforward.copy
bias.copy
```

写入目标：

```python
narm
arm_q
arm_dq
arm_ddq
finger_positions
bounds
controller
position_controller_torque
mass
kp
kd
actual_q
actual_dq
bias
position_feedback
velocity_feedback
velocity_feedforward
acceleration_feedforward
torque
bounded
finger_ctrl
self.native_data.ctrl[self._arm_actuators]
self.native_data.ctrl[self._finger_actuators]
```

直接调用符号：

```python
len
_finite_array
np.any
ValueError
self._grip_controller.set_goal
finger_positions.copy
self._arm_controller.set_goal
arm_q.copy
controller.run_controller
np.clip
np.asarray
self._grip_controller.run_controller
np.array_equal
RuntimeError
torque.copy
int
np.count_nonzero
arm_dq.copy
arm_ddq.copy
position_feedback.copy
velocity_feedback.copy
velocity_feedforward.copy
acceleration_feedforward.copy
bias.copy
```

## `RobosuiteExecutionBackend.last_step_accounting`

[定义位置](../../../manipulation/robosuite_execution.py#L326)。

输入签名：

```python
def last_step_accounting(self):
```

实现说明：

> Return the latest attempt's facts independently of later sampling.

显式输出分支：

```python
deepcopy(self._last_step_accounting)
```

读取的对象属性：

```python
self._last_step_accounting
```

写入目标：

无显式条目。

直接调用符号：

```python
deepcopy
```

## `RobosuiteExecutionBackend.step`

[定义位置](../../../manipulation/robosuite_execution.py#L330)。

输入签名：

```python
def step(self):
```

实现说明：

> Integrate one native step and retain its count even if refresh fails.
>
> One entered attempt is conservatively charged. A normal native return
> confirms its one completed step; an exceptional return can still be
> resolved by the native clock. A stationary exceptional clock confirms
> zero completed steps without refunding the attempted work. An unreadable
> or discontinuous clock leaves only a lower bound on completed steps.
> No accounting or derived-feedback error replaces a native exception.

显式输出分支：

```python
self.last_step_accounting
```

读取的对象属性：

```python
self._step_attempts
self.native_data.time
self.native_data
np.isfinite
type(error).__name__
self._sim.step
self._sim
self.timestep
mujoco.mj_forward
self.native_model
self.last_step_accounting
```

写入目标：

```python
self._step_attempts
report
self._last_step_accounting
before
report['before_sim_time']
report['clock_error']
report['native_step_attempted']
report['native_step_returned']
report['native_error']
after
report['after_sim_time']
finite
advanced
stationary
report['clock_advance_confirmed']
report['physical_steps']
report['step_accounting_exact']
report['feedback_refresh_error']
```

直接调用符号：

```python
float
np.isfinite
ValueError
type
self._sim.step
bool
abs
int
mujoco.mj_forward
```

## `RobosuiteExecutionBackend._prepare_reference_fk`

[定义位置](../../../manipulation/robosuite_execution.py#L397)。

输入签名：

```python
def _prepare_reference_fk(self):
```

实现说明：

> Require a fixed, controlled TCP chain before creating scratch data.
>
> This arm-curve executor has no commanded finger *path*. A TCP that
> depends on fingers cannot use measured fingers to hide its tracking
> error, so it requires a separately defined reference and is rejected
> here. Uncontrolled or mocap-driven ancestors are likewise not silently
> taken from an environment snapshot as a moving robot-base reference.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.native_model
model.site_bodyid
self._site_id
ancestors.add
model.body_parentid
self._base_id
model.jnt_bodyid
self._joint_ids[:len(self.arm_joint_names)].tolist
self._joint_ids
self.arm_joint_names
model.body_mocapid
base_ancestors.add
mujoco.MjData
self._reference_data.qpos
self._reference_data
self._qpos_ids
self.native_data.qpos
self.native_data
mujoco.mj_kinematics
np.eye
self._calibrated_base_pose
self._reference_data.xmat[self._base_id].reshape
self._reference_data.xmat
self._reference_data.xpos
```

写入目标：

```python
model
body
ancestors
chain
index
owner
arm_ids
base_ancestors
self._reference_data
self._reference_data.qpos[self._qpos_ids]
self._calibrated_base_pose
self._calibrated_base_pose[:3, :3]
self._calibrated_base_pose[:3, 3]
```

直接调用符号：

```python
int
set
ancestors.add
ValueError
enumerate
self._joint_ids[:len(self.arm_joint_names)].tolist
len
any
base_ancestors.add
mujoco.MjData
mujoco.mj_kinematics
np.eye
self._reference_data.xmat[self._base_id].reshape
```

## `RobosuiteExecutionBackend.reference_tool_pose`

[定义位置](../../../manipulation/robosuite_execution.py#L434)。

输入签名：

```python
def reference_tool_pose(self, arm_q):
```

实现说明：

> Compute model TCP FK of explicit arm q without observing or stepping.

显式输出分支：

```python
pose
```

读取的对象属性：

```python
self.arm_joint_names
self._reference_data.qpos
self._reference_data
self._qpos_ids
mujoco.mj_kinematics
self.native_model
np.eye
self._reference_data.site_xmat[self._site_id].reshape
self._reference_data.site_xmat
self._site_id
self._reference_data.site_xpos
```

写入目标：

```python
values
self._reference_data.qpos[self._qpos_ids[:len(values)]]
pose
pose[:3, :3]
pose[:3, 3]
```

直接调用符号：

```python
_finite_array
len
mujoco.mj_kinematics
np.eye
self._reference_data.site_xmat[self._site_id].reshape
```

## `RobosuiteExecutionBackend.abort_hold`

[定义位置](../../../manipulation/robosuite_execution.py#L447)。

输入签名：

```python
def abort_hold(self, feedback, finger_positions, *, expected_sim_time, stop_velocity, max_steps):
```

实现说明：

> Apply hold through existing controllers, then measure bounded settling.
>
> The arm goal is the failed sample's actual q. The last applied finger
> goal is retained, so an event rejected by supervision is never issued
> by fault handling. Hold telemetry is separate from the failed curve.
> No state reset or claimed zero velocity substitutes for feedback.

显式输出分支：

```python
result
result
```

读取的对象属性：

```python
self.proprioception
feedback.get
self.joint_names
self.arm_joint_names
self.finger_joint_names
self._trial_id
self._feedback_source
self.native_data.time
self.native_data
np.asarray
np.isfinite(clocks).all
np.isfinite
np.max
np.abs
q.shape
np.any
self.joint_ranges
result['samples'].append
limits.tolist
np.all
q[:narm].copy
result.update
goal.tolist
np.zeros_like(goal).tolist
np.zeros_like
self.actuator_names
np.asarray(finger_positions).tolist
self.apply_control
result['controls'].append
self.sim_time
np.ndarray
value.tolist
control.items
self._step_attempts
self.step
self.last_step_accounting
result['step_accounting'].append
type(error).__name__
result.get
self.timestep
```

写入目标：

```python
result
feedback
current_time
clocks
q
measured_entry_dq
limits
ranges
narm
outside
result['stop_velocity']
result['physical_stop_confirmed']
result['reason']
goal
index
control
result['native_ctrl_written']
key
value
previous_attempt
result['charged_steps']
result['joint_position_bounds_satisfied']
step_error
accounting
result['step_accounting_exact']
result['step_accounting_error']
result['physical_steps']
actual
measured_q
measured_dq
actual_time
```

直接调用符号：

```python
type
self.proprioception
tuple
feedback.get
ValueError
float
np.asarray
np.isfinite(clocks).all
np.isfinite
np.max
np.abs
_finite_array
len
np.any
retain_position_faults
result['samples'].append
limits.tolist
bool
np.all
q[:narm].copy
result.update
goal.tolist
np.zeros_like(goal).tolist
np.zeros_like
list
np.asarray(finger_positions).tolist
range
self.apply_control
result['controls'].append
isinstance
value.tolist
control.items
self.step
result['step_accounting'].append
result.get
RuntimeError
abs
```

## `RobosuiteExecutionBackend.proprioception`

[定义位置](../../../manipulation/robosuite_execution.py#L606)。

输入签名：

```python
def proprioception(self):
```

实现说明：

> Read encoders/commands/time and estimate TCP with calibrated FK.
>
> The tool pose is a model estimate from this sample's measured arm q,
> not an independent TCP observation. Acceleration and actuator force
> are not available sensors in this binding and are not exported.

显式输出分支：

```python
{'joint_names': self.joint_names, 'reference_frame': self.reference_frame, 'tool_link': self.tool_link, 'tool_transform': self.tool_transform, 'arm_joint_names': self.arm_joint_names, 'finger_joint_names': self.finger_joint_names, 'trial_id': self._trial_id, 'feedback_id': f'{self._trial_id}@native_time={timestamp:.17g}', 'source': self._feedback_source, 'timestamp_s': timestamp, 'q': q, 'dq': self.native_data.qvel[self._qvel_ids].copy(), 'actuator_names': self.actuator_names, 'base_pose': self._calibrated_base_pose.copy(), 'tool_pose': tool, 'sim_time': timestamp, 'base_pose_source': 'known_fixed_robot_model_calibration', 'tool_pose_source': 'forward_kinematics_from_measured_arm_q', 'tool_pose_is_independent_observation': False, 'actuator_ctrl': {'joint_names': self.joint_names, 'actuator_names': self._actuator_names, 'values': self.native_data.ctrl[self._actuator_ids].tolist(), 'sim_time': timestamp, 'timestamp_s': timestamp, 'semantics': 'native actuator command state; never measured joint position'}}
```

读取的对象属性：

```python
self.native_data.time
self.native_data
self.native_data.qpos[self._qpos_ids].copy
self.native_data.qpos
self._qpos_ids
self.reference_tool_pose
self.arm_joint_names
self.joint_names
self.reference_frame
self.tool_link
self.tool_transform
self.finger_joint_names
self._trial_id
self._feedback_source
self.native_data.qvel[self._qvel_ids].copy
self.native_data.qvel
self._qvel_ids
self.actuator_names
self._calibrated_base_pose.copy
self._calibrated_base_pose
self._actuator_names
self.native_data.ctrl[self._actuator_ids].tolist
self.native_data.ctrl
self._actuator_ids
```

写入目标：

```python
timestamp
q
tool
```

直接调用符号：

```python
float
self.native_data.qpos[self._qpos_ids].copy
self.reference_tool_pose
len
self.native_data.qvel[self._qvel_ids].copy
self._calibrated_base_pose.copy
self.native_data.ctrl[self._actuator_ids].tolist
```
