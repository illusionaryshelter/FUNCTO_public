# `manipulation/gripper_state.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../03-attachment-contact-and-hand.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_names`

[定义位置](../../../manipulation/gripper_state.py#L16)。

输入签名：

```python
def _names(values, label):
```

输出分支的源码表达式：

```python
names
```

直接调用：

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

输出分支的源码表达式：

```python
deepcopy(value)
```

直接调用：

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

输出分支的源码表达式：

```python
{'robot_config': result, 'planning_feedback': active_feedback, 'full_feedback': deepcopy(feedback), 'hand_binding': binding}
```

直接写入的状态或数组位置：`defaults[configured.index(name)]`、`kinematics['lock_joints']`、`cspace['default_joint_position']`。

直接调用：

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
