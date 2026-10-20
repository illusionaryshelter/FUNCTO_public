# `manipulation/representation.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_timeline`

[定义位置](../../../manipulation/representation.py#L18)。

输入签名：

```python
def _timeline(timestamps):
```

输出分支的源码表达式：

```python
validate_timeline(np.arange(len(times)), times)[1]
```

直接调用：

```python
np.asarray
validate_timeline
np.arange
len
```

## `_progress`

[定义位置](../../../manipulation/representation.py#L23)。

输入签名：

```python
def _progress(times):
```

输出分支的源码表达式：

```python
np.zeros(1) if len(times) == 1 else (times - times[0]) / (times[-1] - times[0])
```

直接调用：

```python
len
np.zeros
```

## `_aligned_poses`

[定义位置](../../../manipulation/representation.py#L27)。

输入签名：

```python
def _aligned_poses(value, times, name):
```

输出分支的源码表达式：

```python
poses
```

直接调用：

```python
_poses_array
len
ValueError
```

## `sample_poses`

[定义位置](../../../manipulation/representation.py#L34)。

输入签名：

```python
def sample_poses(poses, timestamps, query_times):
```

输出分支的源码表达式：

```python
path.sample_numpy(progress)
```

直接调用：

```python
_timeline
_aligned_poses
np.asarray
np.isfinite(query).all
np.isfinite
np.any
ValueError
PosePath
_progress
len
np.zeros_like
path.sample_numpy
```

## `trajectory_summary`

[定义位置](../../../manipulation/representation.py#L51)。

输入签名：

```python
def trajectory_summary(poses, timestamps, *, approach_time=None, interaction_time=None):
```

输出分支的源码表达式：

```python
_sampled_summary(lambda time: sample_poses(values, times, time), approach_time, interaction_time)
```

直接调用：

```python
_timeline
_aligned_poses
_sampled_summary
```

## `_sampled_summary`

[定义位置](../../../manipulation/representation.py#L64)。

输入签名：

```python
def _sampled_summary(sample, approach_time, interaction_time):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result['missing']['interaction_pose']`、`result['interaction_pose']`、`result['missing']['approach_direction']`、`result['approach_direction']`。

直接调用：

```python
sample
float
ValueError
np.linalg.norm
np.finfo
max
```

## `_relative`

[定义位置](../../../manipulation/representation.py#L88)。

输入签名：

```python
def _relative(parent, child):
```

输出分支的源码表达式：

```python
np.stack([_inverse(p) @ c for (p, c) in zip(parent, child)])
```

直接调用：

```python
np.stack
_inverse
zip
```

## `build_representation`

[定义位置](../../../manipulation/representation.py#L93)。

输入签名：

```python
def build_representation(*, timestamps, frame_indices, kind, T_world_carrier=None, T_world_reference=None, T_world_eef=None, L_G=None, L_R=None, approach_time=None, interaction_time=None, release_time=None, events=(), provenance=None):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[name]`、`result['C_D']`、`result['H_g']`、`result['missing']['H_g']`、`result['H_m']`、`result['missing']['H_m']`、`result['missing']['C_D']`、`result['summaries'][name]`。

直接调用：

```python
ValueError
validate_timeline
_progress
deepcopy
list
_aligned_poses
_pose_array
np.isfinite
_relative
_inverse
all
_sampled_summary
trajectory_summary
```

## `_component_path`

[定义位置](../../../manipulation/representation.py#L162)。

输入签名：

```python
def _component_path(value, times, path_times, *, source, parent=None, attachment=None):
```

输出分支的源码表达式：

```python
PosePath(_progress(path_times), values, source=source, parent=parent, attachment=np.eye(4) if attachment is None else attachment)
```

直接调用：

```python
np.asarray
np.repeat
_pose_array
len
sample_poses
PosePath
_progress
np.eye
```

## `_right_path`

[定义位置](../../../manipulation/representation.py#L179)。

输入签名：

```python
def _right_path(parent, transform, source):
```

输出分支的源码表达式：

```python
PosePath(parent.progress, np.repeat(np.eye(4)[None], len(parent.progress), axis=0), source=source, parent=parent, attachment=transform)
```

直接调用：

```python
PosePath
np.repeat
np.eye
len
```

## `_require_same_pose`

[定义位置](../../../manipulation/representation.py#L185)。

输入签名：

```python
def _require_same_pose(first, second, message):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
_errors
max
ValueError
```

## `_boundary_record`

[定义位置](../../../manipulation/representation.py#L192)。

输入签名：

```python
def _boundary_record(path, current, goal, event_progress):
```

输出分支的源码表达式：

```python
dict(status='passed', actual_start=current, adapted_start=start, nominal_goal=goal, adapted_interaction=event, start_position_error_m=float(p0[0]), start_rotation_error_rad=float(r0[0]), interaction_position_error_m=float(pg[0]), interaction_rotation_error_rad=float(rg[0]), scope='Geometric boundaries; not physical task or robot feasibility verification')
```

直接调用：

```python
path.sample_numpy
_require_same_pose
_errors
dict
float
```

## `_adapt_boundary`

[定义位置](../../../manipulation/representation.py#L206)。

输入签名：

```python
def _adapt_boundary(path, current, event_progress, source):
```

输出分支的源码表达式：

```python
path
TransferredPosePath(path, current, [{'progress': event_progress, 'pose': goal}], source=source)
```

直接调用：

```python
path.sample_numpy
len
_require_same_pose
TransferredPosePath
```

## `generate_reference`

[定义位置](../../../manipulation/representation.py#L216)。

输入签名：

```python
def generate_reference(representation, *, T_world_function=None, J=None, A=None, T_world_eef=None, T_world_body=None, motion_model=None, source_coordinates=None, current_coordinates=None, contact_motion=False, held_until_time=None):
```

直接读取的对象属性：`result.get`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[name]`、`result['T_world_function']`、`result['missing']['interaction_time']`、`result['missing']['eef_poses']`、`result['method']`、`result['missing']['object_poses']`、`result['source_coordinates']`、`result['current_coordinates']`、`result['coordinates']`、`result['L_body_material']`、`result['status']`。

直接调用：

```python
validate_timeline
dict
deepcopy
representation.get
ValueError
float
any
np.isfinite
np.unique
min
_pose_array
_aligned_poses
np.asarray
len
result.get
_inverse
_require_same_pose
_component_path
function_path
diagnostic
_adapt_boundary
publish
raw.sample_numpy
isinstance
np.asarray(current_coordinates, dtype=float).reshape
np.isfinite(coordinates).all
np.isfinite(current).all
coordinates.copy
current.copy
motion_model.poses
mechanism_path
adapted_body.coordinates.copy
sample_poses
material_path
_right_path
raw_body.sample_numpy
raw_eef.sample_numpy
adapted_eef.sample_numpy
```

## `generate_reference.function_path`

[定义位置](../../../manipulation/representation.py#L287)。

输入签名：

```python
def generate_reference.function_path(clock):
```

输出分支的源码表达式：

```python
_component_path(result['T_world_function'], times, clock, source='current functional frame F')
```

直接调用：

```python
_component_path
```

## `generate_reference.diagnostic`

[定义位置](../../../manipulation/representation.py#L290)。

输入签名：

```python
def generate_reference.diagnostic(name, path, clock):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`result['bound_' + name + '_timestamps']`、`result['bound_' + name + '_poses']`。

直接调用：

```python
clock.copy
path.sample_numpy
_progress
```

## `generate_reference.publish`

[定义位置](../../../manipulation/representation.py#L294)。

输入签名：

```python
def generate_reference.publish(name, path, current, nominal_goal):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`result['boundary_checks'][name]`、`result[name + '_path']`、`result[name + '_timestamps']`、`result[name + '_poses']`、`result['summaries'][name]`。

直接调用：

```python
_boundary_record
path_times.copy
path.sample_numpy
_progress
representation.get
_sampled_summary
```

## `generate_reference.publish.sample`

[定义位置](../../../manipulation/representation.py#L302)。

输入签名：

```python
def generate_reference.publish.sample(time):
```

输出分支的源码表达式：

```python
path.sample_numpy(progress)
```

直接调用：

```python
len
path.sample_numpy
```

## `generate_reference.mechanism_path`

[定义位置](../../../manipulation/representation.py#L335)。

输入签名：

```python
def generate_reference.mechanism_path(clock):
```

输出分支的源码表达式：

```python
PosePath(_progress(clock), motion_model.poses(values), source='current mechanism FK from source scalar displacement', motion_model=motion_model, coordinates=values)
```

直接调用：

```python
np.interp
PosePath
_progress
motion_model.poses
```

## `generate_reference.material_path`

[定义位置](../../../manipulation/representation.py#L368)。

输入签名：

```python
def generate_reference.material_path(clock):
```

输出分支的源码表达式：

```python
_component_path(Q, times, clock, source='source chart-invariant relative displacement', parent=function_path(clock), attachment=_inverse(Qc))
```

直接调用：

```python
_component_path
function_path
_inverse
```
