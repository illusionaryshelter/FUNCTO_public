# `manipulation/gripper_state.py` 实现契约

代码基线：`9e8b365`。
职责和数据语义见[组件说明](../03-attachment-contact-and-hand.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_names`

[定义位置](../../../manipulation/gripper_state.py#L16)。

输入签名：

```python
def _names(values, label):
```

显式输出分支：

```python
names
```

读取的对象属性：

```python
name.strip
```

写入目标：

```python
names
name
```

直接调用符号：

```python
isinstance
ValueError
tuple
any
name.strip
len
set
```

## `_source`

[定义位置](../../../manipulation/gripper_state.py#L26)。

输入签名：

```python
def _source(value, label):
```

显式输出分支：

```python
deepcopy(value)
```

读取的对象属性：

```python
value.strip
```

写入目标：

无显式条目。

直接调用符号：

```python
isinstance
value.strip
ValueError
deepcopy
```

## `build_post_grasp_robot_config`

[定义位置](../../../manipulation/gripper_state.py#L33)。

输入签名：

```python
def build_post_grasp_robot_config(robot, feedback):
```

实现说明：

> Return a new measured-hand robot config and an arm-only start state.
>
> ``feedback`` is one complete named robot snapshot: q/dq, arm_joint_names,
> finger_joint_names, feedback_id, timestamp_s, source, reference_frame,
> base_pose, tool_pose, tool_link and tool_transform. Commanded finger targets
> and actuator ctrl may be retained in that snapshot but are never geometry
> inputs. Finger coordinates need only their robot feedback: object poses,
> contact evidence and attachment assumptions belong to separate models.
>
> The original cspace layout, arm rest posture, collision geometry and pair
> policy are preserved. Only finger locks and their matching default entries
> are updated from actual q. Planning q/dq contains only the named arm joints;
> full measured state remains available separately. The result describes a
> current nominal held-hand configuration, not a bound on future finger
> motion or a complete environment/payload collision scene. New measurements
> or a release require a new binding, not reuse of a stale model.

显式输出分支：

```python
{'robot_config': result, 'planning_feedback': active_feedback, 'full_feedback': deepcopy(feedback), 'hand_binding': binding}
```

读取的对象属性：

```python
np.asarray
q.shape
dq.shape
np.isfinite(q).all
np.isfinite
np.isfinite(dq).all
value.strip
tool_link.strip
kinematics.get
np.asarray(cspace['default_joint_position'], float).copy
defaults.shape
np.isfinite(defaults).all
locked.items
configured.index
defaults.tolist
names.index
q[indices].tolist
dq[indices].tolist
base.tolist
tool.tolist
tool_transform.tolist
locked.values
```

写入目标：

```python
value
names
arm
fingers
q
dq
identifier
frame
source
timestamp
base
tool
tool_transform
tool_link
result
kinematics
cspace
configured
name
previous_locks
defaults
measured
locked
defaults[configured.index(name)]
kinematics['lock_joints']
cspace['default_joint_position']
indices
active_feedback
binding
```

直接调用符号：

```python
all
isinstance
TypeError
_names
set
ValueError
np.asarray
len
np.isfinite(q).all
np.isfinite
np.isfinite(dq).all
any
value.strip
_source
float
_pose_array
tool_link.strip
deepcopy
tuple
kinematics.get
np.asarray(cspace['default_joint_position'], float).copy
np.isfinite(defaults).all
dict
zip
locked.items
configured.index
defaults.tolist
names.index
list
q[indices].tolist
dq[indices].tolist
base.tolist
tool.tolist
tool_transform.tolist
locked.values
```
