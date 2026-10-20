# `manipulation/grasp.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `validate_timeline`

[定义位置](../../../manipulation/grasp.py#L22)。

输入签名：

```python
def validate_timeline(frame_indices, timestamps):
```

输出分支的源码表达式：

```python
(frames.copy(), times.astype(float, copy=True))
```

直接调用：

```python
any
isinstance
ValueError
np.asarray
len
np.any
np.isfinite(times).all
np.isfinite
frames.copy
times.astype
```

## `_name`

[定义位置](../../../manipulation/grasp.py#L44)。

输入签名：

```python
def _name(record, key):
```

直接读取的对象属性：`record.get`。

输出分支的源码表达式：

```python
value
```

直接调用：

```python
record.get
isinstance
value.strip
ValueError
```

## `_gripper_events`

[定义位置](../../../manipulation/grasp.py#L51)。

输入签名：

```python
def _gripper_events(value, frames):
```

输出分支的源码表达式：

```python
events
```

直接调用：

```python
isinstance
ValueError
set
map
_name
event.get
int
ids.add
events.append
deepcopy
any
```

## `_grasp_goal`

[定义位置](../../../manipulation/grasp.py#L93)。

输入签名：

```python
def _grasp_goal(value):
```

输出分支的源码表达式：

```python
{'T_carrier_eef': _pose_array(value['T_carrier_eef'], 'grasp_goal.T_carrier_eef'), 'position_tolerance_m': position_tolerance, 'rotation_tolerance_rad': rotation_tolerance, 'source': _source(value['source'], 'Grasp task goal and allowed set')}
```

直接调用：

```python
isinstance
set
ValueError
any
_tolerances
_pose_array
_source
```

## `_reference_prior`

[定义位置](../../../manipulation/grasp.py#L109)。

输入签名：

```python
def _reference_prior(value):
```

输出分支的源码表达式：

```python
{'T_carrier_eef': _pose_array(value['T_carrier_eef'], 'grasp_reference_prior.T_carrier_eef'), 'source': _source(value['source'], 'Original grasp reference anchor')}
```

直接调用：

```python
isinstance
set
ValueError
_pose_array
_source
```

## `bind_grasp_candidate`

[定义位置](../../../manipulation/grasp.py#L117)。

输入签名：

```python
def bind_grasp_candidate(scene, T_reference_tool, *, source):
```

直接读取的对象属性：`scene.get`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result['grasp_reference_prior']`、`result['grasp_goal']`。

直接调用：

```python
isinstance
ValueError
_name
_pose_array
_grasp_goal
scene.get
_reference_prior
goal['T_carrier_eef'].copy
deepcopy
np.array_equal
prior['T_carrier_eef'].copy
_inverse
_source
```

## `ground_grasp`

[定义位置](../../../manipulation/grasp.py#L156)。

输入签名：

```python
def ground_grasp(demo, scene):
```

直接读取的对象属性：`scene.get`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result['grasp_reference_prior']`。

直接调用：

```python
_name
ValueError
_poses_array
validate_timeline
len
_gripper_events
isinstance
any
_pose_array
_grasp_goal
scene.get
_reference_prior
np.array_equal
_inverse
int
grasp_goal['T_carrier_eef'].copy
```

## `attached_eef_reference`

[定义位置](../../../manipulation/grasp.py#L228)。

输入签名：

```python
def attached_eef_reference(carrier_poses, actual_T_carrier_eef):
```

输出分支的源码表达式：

```python
carrier @ actual
```

直接调用：

```python
_poses_array
_pose_array
```
