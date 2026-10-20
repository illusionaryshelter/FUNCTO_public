# `manipulation/planning_diagnostics.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `AttemptWitnesses`

[定义位置](../../../manipulation/planning_diagnostics.py#L13)。

```python
class AttemptWitnesses:
    ...
    ...
```

该类型的直接方法：`__init__`、`_validated_parameters`、`parameters`、`current_parameters`、`last_selection`、`_native_parameters`、`_validation_parameters`、`add_native`、`add_validation`、`set_current`、`select_parameters`。

## `AttemptWitnesses.__init__`

[定义位置](../../../manipulation/planning_diagnostics.py#L26)。

输入签名：

```python
def AttemptWitnesses.__init__(self):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._parameters`、`self._current`、`self._selection_counts`、`self._last_selection`。

直接调用：

```python
set
```

## `AttemptWitnesses._validated_parameters`

[定义位置](../../../manipulation/planning_diagnostics.py#L33)。

输入签名：

```python
@staticmethod
def AttemptWitnesses._validated_parameters(values, *, source):
```

输出分支的源码表达式：

```python
parameters
```

直接调用：

```python
np.asarray
np.isfinite(parameters).all
np.isfinite
np.any
ValueError
```

## `AttemptWitnesses.parameters`

[定义位置](../../../manipulation/planning_diagnostics.py#L41)。

输入签名：

```python
@property
def AttemptWitnesses.parameters(self):
```

直接读取的对象属性：`self._parameters`。

输出分支的源码表达式：

```python
np.asarray(sorted(self._parameters), dtype=float)
```

直接调用：

```python
np.asarray
sorted
```

## `AttemptWitnesses.current_parameters`

[定义位置](../../../manipulation/planning_diagnostics.py#L46)。

输入签名：

```python
@property
def AttemptWitnesses.current_parameters(self):
```

直接读取的对象属性：`self._current`。

输出分支的源码表达式：

```python
np.asarray(sorted(self._current), dtype=float)
```

直接调用：

```python
np.asarray
sorted
```

## `AttemptWitnesses.last_selection`

[定义位置](../../../manipulation/planning_diagnostics.py#L51)。

输入签名：

```python
@property
def AttemptWitnesses.last_selection(self):
```

直接读取的对象属性：`self._last_selection`。

输出分支的源码表达式：

```python
deepcopy(self._last_selection)
```

直接调用：

```python
deepcopy
```

## `AttemptWitnesses._native_parameters`

[定义位置](../../../manipulation/planning_diagnostics.py#L56)。

输入签名：

```python
@classmethod
def AttemptWitnesses._native_parameters(cls, native_metrics, *, duration):
```

输出分支的源码表达式：

```python
cls._validated_parameters(parameters, source='Native failure witnesses')
```

直接调用：

```python
float
np.isfinite
ValueError
native_metrics.get
parameters.extend
metrics.get
trace.get
parameters.append
cls._validated_parameters
```

## `AttemptWitnesses._validation_parameters`

[定义位置](../../../manipulation/planning_diagnostics.py#L88)。

输入签名：

```python
@classmethod
def AttemptWitnesses._validation_parameters(cls, validation):
```

输出分支的源码表达式：

```python
parameters[failed]
```

直接调用：

```python
cls._validated_parameters
np.asarray
ValueError
np.isfinite
```

## `AttemptWitnesses.add_native`

[定义位置](../../../manipulation/planning_diagnostics.py#L97)。

输入签名：

```python
def AttemptWitnesses.add_native(self, native_metrics, *, duration):
```

直接读取的对象属性：`self._native_parameters`、`self._parameters`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
self._native_parameters
self._parameters.update
parameters.tolist
```

## `AttemptWitnesses.add_validation`

[定义位置](../../../manipulation/planning_diagnostics.py#L102)。

输入签名：

```python
def AttemptWitnesses.add_validation(self, validation):
```

直接读取的对象属性：`self._parameters`、`self._validation_parameters`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
self._parameters.update
self._validation_parameters(validation).tolist
self._validation_parameters
```

## `AttemptWitnesses.set_current`

[定义位置](../../../manipulation/planning_diagnostics.py#L106)。

输入签名：

```python
def AttemptWitnesses.set_current(self, native_metrics, *, validation=None, duration):
```

直接读取的对象属性：`self._native_parameters`、`self._validation_parameters`、`self._parameters`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._current`、`self._last_selection`。

直接调用：

```python
self._native_parameters
np.empty
self._validation_parameters
set
np.r_[native, independent].tolist
self._parameters.update
```

## `AttemptWitnesses.select_parameters`

[定义位置](../../../manipulation/planning_diagnostics.py#L122)。

输入签名：

```python
def AttemptWitnesses.select_parameters(self, *, fixed_parameters, regular_parameters, max_points):
```

直接读取的对象属性：`self._validated_parameters`、`self._current`、`self._selection_counts`、`self._parameters`。

输出分支的源码表达式：

```python
np.asarray(sorted(selected), dtype=float)
```

直接写入的状态或数组位置：`self._selection_counts[parameter]`、`self._last_selection`。

直接调用：

```python
type
ValueError
self._validated_parameters
set
fixed.tolist
regular.tolist
sorted
self._selection_counts.get
len
np.asarray
```

## `capture_component_failures`

[定义位置](../../../manipulation/planning_diagnostics.py#L162)。

输入签名：

```python
def capture_component_failures(components, times, progress, joints, position, quaternion, *, normalized_time=None, mode='exact'):
```

输出分支的源码表达式：

```python
rows
```

直接写入的状态或数组位置：`row['normalized_time']`。

直接调用：

```python
tuple
len
ValueError
components.items
cpu
values.reshape(*shape, -1).max
values.reshape
np.argwhere
np.isfinite
any
float
int
q.tolist
p.tolist
rotation.tolist
rows.append
```

## `scene_component_failures`

[定义位置](../../../manipulation/planning_diagnostics.py#L205)。

输入签名：

```python
def scene_component_failures(components, times, progress, permission, joints, position, quaternion, *, modes):
```

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
result
```

直接调用：

```python
components.items
name.startswith
len
cpu
np.argwhere
np.isfinite
result.append
float
int
q.tolist
p.tolist
rotation.tolist
```

## `bind_trace_to_native_seed`

[定义位置](../../../manipulation/planning_diagnostics.py#L235)。

输入签名：

```python
def bind_trace_to_native_seed(trace, state, seed_index, joint_names):
```

输出分支的源码表达式：

```python
{'status': 'unavailable', 'reason': 'No source metrics trace was retained'}
{'status': 'error', 'error': trace['error']}
{'status': 'unmatched', 'reason': 'No exact named q/dq input matches this result seed'}
{'status': 'recorded', 'source_batch_index': index, 'component_failures': failures, 'world_geometry': deepcopy(trace['world_geometry']), 'world_geometry_scope': 'Worst witness across this metrics call; joint/time identity is explicit', 'scope': 'Original source components before mode maximum, constraint weighting and anchor folding'}
```

直接写入的状态或数组位置：`row['geometry_witnesses']`。

直接调用：

```python
state.joint_names.index
getattr(state, name)[0, seed_index].detach().cpu().numpy
getattr(state, name)[0, seed_index].detach().cpu
getattr(state, name)[0, seed_index].detach
getattr
range
len
all
actual.tobytes
stored[index].tobytes
zip
deepcopy
trace['world_geometry'].get
```
