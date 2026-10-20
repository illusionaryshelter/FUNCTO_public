# `manipulation/robosuite_execution.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `RobosuiteExecutionBackend`

[定义位置](../../../manipulation/robosuite_execution.py#L23)。

```python
class RobosuiteExecutionBackend:
    ...
    ...
```

该类型的直接方法：`__init__`、`timestep`、`sim_time`、`trial_id`、`feedback_source`、`feedback_id`、`reference_frame`、`tool_link`、`tool_transform`、`actuator_names`、`joint_ranges`、`velocity_limits`、`finger_ctrlrange`、`finger_control`、`actuator_control`、`describe`、`apply_control`、`last_step_accounting`、`step`、`_prepare_reference_fk`、`reference_tool_pose`、`abort_hold`、`proprioception`。

## `RobosuiteExecutionBackend.__init__`

[定义位置](../../../manipulation/robosuite_execution.py#L26)。

输入签名：

```python
def RobosuiteExecutionBackend.__init__(self, sim, *, base_link, tcp_site, arm_joint_names, finger_joint_names, arm_kp, arm_kd, velocity_limits, source):
```

直接读取的对象属性：`self.arm_joint_names`、`self.finger_joint_names`、`self.joint_names`、`self.native_model`、`self._joint_ids`、`self._actuator_names`、`self._actuator_ids`、`self._arm_actuators`、`self._finger_actuators`、`self._dt`、`self._sim`、`self.native_data`、`self._grip_controller`、`self._prepare_reference_fk`、`self._trial_id`、`self.reference_frame`、`self.tool_link`、`self.tool_transform`、`self.actuator_names`、`self.timestep`、`self._arm_controller`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._sim`、`self.native_model`、`self.native_data`、`self._trial_id`、`self.arm_joint_names`、`self.finger_joint_names`、`self.joint_names`、`self._velocity_limits`、`self._base_id`、`self._site_id`、`self._joint_ids`、`self._qpos_ids`、`self._qvel_ids`、`self._actuator_ids`、`self._actuator_names`、`self._feedback_source`、`self._arm_actuators`、`self._finger_actuators`、`self._dt`、`self._step_attempts`、`self._last_step_accounting`、`self._arm_controller`、`self._grip_controller`、`self._description`。

直接调用：

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
def RobosuiteExecutionBackend.__init__.indexes(start, stop):
```

直接读取的对象属性：`self._joint_ids`、`self._qpos_ids`、`self._qvel_ids`、`self._actuator_ids`。

输出分支的源码表达式：

```python
{'joints': self._joint_ids[start:stop].tolist(), 'qpos': self._qpos_ids[start:stop].tolist(), 'qvel': self._qvel_ids[start:stop].tolist(), 'actuators': self._actuator_ids[start:stop].tolist()}
```

直接调用：

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
@property
def RobosuiteExecutionBackend.timestep(self):
```

直接读取的对象属性：`self._dt`。

输出分支的源码表达式：

```python
self._dt
```

## `RobosuiteExecutionBackend.sim_time`

[定义位置](../../../manipulation/robosuite_execution.py#L199)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.sim_time(self):
```

直接读取的对象属性：`self.native_data`。

输出分支的源码表达式：

```python
float(self.native_data.time)
```

直接调用：

```python
float
```

## `RobosuiteExecutionBackend.trial_id`

[定义位置](../../../manipulation/robosuite_execution.py#L203)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.trial_id(self):
```

直接读取的对象属性：`self._trial_id`。

输出分支的源码表达式：

```python
self._trial_id
```

## `RobosuiteExecutionBackend.feedback_source`

[定义位置](../../../manipulation/robosuite_execution.py#L207)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.feedback_source(self):
```

直接读取的对象属性：`self._feedback_source`。

输出分支的源码表达式：

```python
self._feedback_source
```

## `RobosuiteExecutionBackend.feedback_id`

[定义位置](../../../manipulation/robosuite_execution.py#L211)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.feedback_id(self):
```

直接读取的对象属性：`self._trial_id`、`self.sim_time`。

输出分支的源码表达式：

```python
f'{self._trial_id}@native_time={self.sim_time:.17g}'
```

## `RobosuiteExecutionBackend.reference_frame`

[定义位置](../../../manipulation/robosuite_execution.py#L215)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.reference_frame(self):
```

输出分支的源码表达式：

```python
'world'
```

## `RobosuiteExecutionBackend.tool_link`

[定义位置](../../../manipulation/robosuite_execution.py#L222)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.tool_link(self):
```

直接读取的对象属性：`self.native_model`、`self._site_id`。

输出分支的源码表达式：

```python
self.native_model.site(self._site_id).name
```

直接调用：

```python
self.native_model.site
```

## `RobosuiteExecutionBackend.tool_transform`

[定义位置](../../../manipulation/robosuite_execution.py#L229)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.tool_transform(self):
```

输出分支的源码表达式：

```python
np.eye(4)
```

直接调用：

```python
np.eye
```

## `RobosuiteExecutionBackend.actuator_names`

[定义位置](../../../manipulation/robosuite_execution.py#L233)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.actuator_names(self):
```

直接读取的对象属性：`self._actuator_names`。

输出分支的源码表达式：

```python
self._actuator_names
```

## `RobosuiteExecutionBackend.joint_ranges`

[定义位置](../../../manipulation/robosuite_execution.py#L237)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.joint_ranges(self):
```

直接读取的对象属性：`self.native_model`、`self._joint_ids`。

输出分支的源码表达式：

```python
self.native_model.jnt_range[self._joint_ids].copy()
```

直接调用：

```python
self.native_model.jnt_range[self._joint_ids].copy
```

## `RobosuiteExecutionBackend.velocity_limits`

[定义位置](../../../manipulation/robosuite_execution.py#L241)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.velocity_limits(self):
```

直接读取的对象属性：`self._velocity_limits`。

输出分支的源码表达式：

```python
dict(self._velocity_limits)
```

直接调用：

```python
dict
```

## `RobosuiteExecutionBackend.finger_ctrlrange`

[定义位置](../../../manipulation/robosuite_execution.py#L245)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.finger_ctrlrange(self):
```

直接读取的对象属性：`self.native_model`、`self._finger_actuators`。

输出分支的源码表达式：

```python
self.native_model.actuator_ctrlrange[self._finger_actuators].copy()
```

直接调用：

```python
self.native_model.actuator_ctrlrange[self._finger_actuators].copy
```

## `RobosuiteExecutionBackend.finger_control`

[定义位置](../../../manipulation/robosuite_execution.py#L249)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.finger_control(self):
```

直接读取的对象属性：`self.native_data`、`self._finger_actuators`。

输出分支的源码表达式：

```python
self.native_data.ctrl[self._finger_actuators].copy()
```

直接调用：

```python
self.native_data.ctrl[self._finger_actuators].copy
```

## `RobosuiteExecutionBackend.actuator_control`

[定义位置](../../../manipulation/robosuite_execution.py#L254)。

输入签名：

```python
@property
def RobosuiteExecutionBackend.actuator_control(self):
```

直接读取的对象属性：`self.native_data`、`self._actuator_ids`。

输出分支的源码表达式：

```python
self.native_data.ctrl[self._actuator_ids].copy()
```

直接调用：

```python
self.native_data.ctrl[self._actuator_ids].copy
```

## `RobosuiteExecutionBackend.describe`

[定义位置](../../../manipulation/robosuite_execution.py#L257)。

输入签名：

```python
def RobosuiteExecutionBackend.describe(self):
```

直接读取的对象属性：`self._description`。

输出分支的源码表达式：

```python
deepcopy(self._description)
```

直接调用：

```python
deepcopy
```

## `RobosuiteExecutionBackend.apply_control`

[定义位置](../../../manipulation/robosuite_execution.py#L261)。

输入签名：

```python
def RobosuiteExecutionBackend.apply_control(self, arm_q, arm_dq, arm_ddq, finger_positions):
```

直接读取的对象属性：`self.arm_joint_names`、`self.finger_joint_names`、`self.finger_ctrlrange`、`self._grip_controller`、`self._arm_controller`、`self.native_model`、`self._arm_actuators`、`self.native_data`、`self._finger_actuators`、`self.actuator_control`。

输出分支的源码表达式：

```python
{'actuator_ctrl': self.actuator_control, 'requested_arm_torque': torque.copy(), 'arm_torque_saturated': bounded != torque, 'arm_control_saturated_coordinates': int(np.count_nonzero(bounded != torque)), 'reference_arm_q': arm_q.copy(), 'reference_arm_dq': arm_dq.copy(), 'reference_arm_ddq': arm_ddq.copy(), 'arm_position_feedback_torque': position_feedback.copy(), 'arm_velocity_feedback_torque': velocity_feedback.copy(), 'arm_velocity_feedforward_torque': velocity_feedforward.copy(), 'arm_acceleration_feedforward_torque': acceleration_feedforward.copy(), 'arm_bias_torque': bias.copy()}
```

直接写入的状态或数组位置：`self.native_data.ctrl[self._arm_actuators]`、`self.native_data.ctrl[self._finger_actuators]`。

直接调用：

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
@property
def RobosuiteExecutionBackend.last_step_accounting(self):
```

直接读取的对象属性：`self._last_step_accounting`。

输出分支的源码表达式：

```python
deepcopy(self._last_step_accounting)
```

直接调用：

```python
deepcopy
```

## `RobosuiteExecutionBackend.step`

[定义位置](../../../manipulation/robosuite_execution.py#L330)。

输入签名：

```python
def RobosuiteExecutionBackend.step(self):
```

直接读取的对象属性：`self._step_attempts`、`self.native_data`、`self._sim`、`self.timestep`、`self.native_model`、`self.last_step_accounting`。

输出分支的源码表达式：

```python
self.last_step_accounting
```

直接写入的状态或数组位置：`self._step_attempts`、`self._last_step_accounting`、`report['before_sim_time']`、`report['clock_error']`、`report['native_step_attempted']`、`report['native_step_returned']`、`report['native_error']`、`report['after_sim_time']`、`report['clock_advance_confirmed']`、`report['physical_steps']`、`report['step_accounting_exact']`、`report['feedback_refresh_error']`。

直接调用：

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
def RobosuiteExecutionBackend._prepare_reference_fk(self):
```

直接读取的对象属性：`self.native_model`、`self._site_id`、`self._base_id`、`self._joint_ids`、`self.arm_joint_names`、`self._reference_data`、`self._qpos_ids`、`self.native_data`、`self._calibrated_base_pose`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._reference_data`、`self._reference_data.qpos[self._qpos_ids]`、`self._calibrated_base_pose`、`self._calibrated_base_pose[:3, :3]`、`self._calibrated_base_pose[:3, 3]`。

直接调用：

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
def RobosuiteExecutionBackend.reference_tool_pose(self, arm_q):
```

直接读取的对象属性：`self.arm_joint_names`、`self._reference_data`、`self._qpos_ids`、`self.native_model`、`self._site_id`。

输出分支的源码表达式：

```python
pose
```

直接写入的状态或数组位置：`self._reference_data.qpos[self._qpos_ids[:len(values)]]`、`pose[:3, :3]`、`pose[:3, 3]`。

直接调用：

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
def RobosuiteExecutionBackend.abort_hold(self, feedback, finger_positions, *, expected_sim_time, stop_velocity, max_steps):
```

直接读取的对象属性：`self.proprioception`、`feedback.get`、`self.joint_names`、`self.arm_joint_names`、`self.finger_joint_names`、`self._trial_id`、`self._feedback_source`、`self.native_data`、`self.joint_ranges`、`result.update`、`self.actuator_names`、`self.apply_control`、`self.sim_time`、`self._step_attempts`、`self.step`、`self.last_step_accounting`、`result.get`、`self.timestep`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result['stop_velocity']`、`result['physical_stop_confirmed']`、`result['reason']`、`result['native_ctrl_written']`、`result['charged_steps']`、`result['joint_position_bounds_satisfied']`、`result['step_accounting_exact']`、`result['step_accounting_error']`、`result['physical_steps']`。

直接调用：

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

## `RobosuiteExecutionBackend.abort_hold.retain_position_faults`

[定义位置](../../../manipulation/robosuite_execution.py#L493)。

输入签名：

```python
def RobosuiteExecutionBackend.abort_hold.retain_position_faults(positions, *, phase, sim_time):
```

直接读取的对象属性：`self.joint_names`。

输出分支的源码表达式：

```python
outside
```

直接写入的状态或数组位置：`result['joint_position_bounds_satisfied']`。

直接调用：

```python
np.flatnonzero
bool
result['joint_position_violation_samples'].append
float
int
```

## `RobosuiteExecutionBackend.proprioception`

[定义位置](../../../manipulation/robosuite_execution.py#L606)。

输入签名：

```python
def RobosuiteExecutionBackend.proprioception(self):
```

直接读取的对象属性：`self.native_data`、`self._qpos_ids`、`self.reference_tool_pose`、`self.arm_joint_names`、`self.joint_names`、`self.reference_frame`、`self.tool_link`、`self.tool_transform`、`self.finger_joint_names`、`self._trial_id`、`self._feedback_source`、`self._qvel_ids`、`self.actuator_names`、`self._calibrated_base_pose`、`self._actuator_names`、`self._actuator_ids`。

输出分支的源码表达式：

```python
{'joint_names': self.joint_names, 'reference_frame': self.reference_frame, 'tool_link': self.tool_link, 'tool_transform': self.tool_transform, 'arm_joint_names': self.arm_joint_names, 'finger_joint_names': self.finger_joint_names, 'trial_id': self._trial_id, 'feedback_id': f'{self._trial_id}@native_time={timestamp:.17g}', 'source': self._feedback_source, 'timestamp_s': timestamp, 'q': q, 'dq': self.native_data.qvel[self._qvel_ids].copy(), 'actuator_names': self.actuator_names, 'base_pose': self._calibrated_base_pose.copy(), 'tool_pose': tool, 'sim_time': timestamp, 'base_pose_source': 'known_fixed_robot_model_calibration', 'tool_pose_source': 'forward_kinematics_from_measured_arm_q', 'tool_pose_is_independent_observation': False, 'actuator_ctrl': {'joint_names': self.joint_names, 'actuator_names': self._actuator_names, 'values': self.native_data.ctrl[self._actuator_ids].tolist(), 'sim_time': timestamp, 'timestamp_s': timestamp, 'semantics': 'native actuator command state; never measured joint position'}}
```

直接调用：

```python
float
self.native_data.qpos[self._qpos_ids].copy
self.reference_tool_pose
len
self.native_data.qvel[self._qvel_ids].copy
self._calibrated_base_pose.copy
self.native_data.ctrl[self._actuator_ids].tolist
```
