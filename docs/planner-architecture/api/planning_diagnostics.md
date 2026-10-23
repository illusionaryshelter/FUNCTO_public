# `manipulation/planning_diagnostics.py` 实现契约

代码基线：`82a3d66；包含本次工作区已实现改动，待统一实现提交后更新`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `AttemptWitnesses`

[定义位置](../../../manipulation/planning_diagnostics.py#L13)。

```python
class AttemptWitnesses():
```

类字段：

无显式条目。

实现说明：

> Keep historical failures separate from the next optimizer's queries.
>
> The ledger records normalized physical time, not source progress: a pause
> can map several distinct trajectory times to one progress value. Exact
> repeated times share one query. Nearby but distinct times remain distinct
> because neither a numerical tolerance nor a previous passing query proves
> the intervening source geometry safe. Only the explicitly selected current
> candidate supplies the bounded optimizer query set. Historical parameters
> remain mandatory inputs to independent export validation after exchange;
> this scheduler neither discards evidence nor certifies motion.

## `AttemptWitnesses.__init__`

[定义位置](../../../manipulation/planning_diagnostics.py#L26)。

输入签名：

```python
def __init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
self._parameters
self._current
self._selection_counts
self._last_selection
```

直接调用符号：

```python
set
```

## `AttemptWitnesses._validated_parameters`

[定义位置](../../../manipulation/planning_diagnostics.py#L33)。

输入签名：

```python
def _validated_parameters(values, *, source):
```

显式输出分支：

```python
parameters
```

读取的对象属性：

```python
np.asarray
parameters.ndim
np.isfinite(parameters).all
np.isfinite
np.any
```

写入目标：

```python
parameters
```

直接调用符号：

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
def parameters(self):
```

实现说明：

> Return an owned sorted snapshot; callers cannot erase the ledger.

显式输出分支：

```python
np.asarray(sorted(self._parameters), dtype=float)
```

读取的对象属性：

```python
np.asarray
self._parameters
```

写入目标：

无显式条目。

直接调用符号：

```python
np.asarray
sorted
```

## `AttemptWitnesses.current_parameters`

[定义位置](../../../manipulation/planning_diagnostics.py#L46)。

输入签名：

```python
def current_parameters(self):
```

实现说明：

> Return failures belonging to the explicitly selected candidate.

显式输出分支：

```python
np.asarray(sorted(self._current), dtype=float)
```

读取的对象属性：

```python
np.asarray
self._current
```

写入目标：

无显式条目。

直接调用符号：

```python
np.asarray
sorted
```

## `AttemptWitnesses.last_selection`

[定义位置](../../../manipulation/planning_diagnostics.py#L51)。

输入签名：

```python
def last_selection(self):
```

实现说明：

> Return owned scheduling evidence, not a constraint-slack verdict.

显式输出分支：

```python
deepcopy(self._last_selection)
```

读取的对象属性：

```python
self._last_selection
```

写入目标：

无显式条目。

直接调用符号：

```python
deepcopy
```

## `AttemptWitnesses._native_parameters`

[定义位置](../../../manipulation/planning_diagnostics.py#L56)。

输入签名：

```python
def _native_parameters(cls, native_metrics, *, duration):
```

实现说明：

> Read exact times from bound native source-constraint traces.
>
> Traces can carry the original normalized query parameter or their
> physical ``time_s`` and frozen duration; the latter ratio is used
> exactly, without clipping or epsilon merging. Unavailable/unmatched
> observer traces contain no attributed witness and are left to their
> existing diagnostic report.

显式输出分支：

```python
cls._validated_parameters(parameters, source='Native failure witnesses')
```

读取的对象属性：

```python
np.isfinite
native_metrics.get
parameters.extend
metrics.get
trace.get
parameters.append
cls._validated_parameters
```

写入目标：

```python
duration
parameters
label
metrics
row
trace
```

直接调用符号：

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
def _validation_parameters(cls, validation):
```

实现说明：

> Read failed times from a completed independent curve check.

显式输出分支：

```python
parameters[failed]
```

读取的对象属性：

```python
cls._validated_parameters
np.asarray
violations.shape
parameters.shape
np.isfinite
```

写入目标：

```python
parameters
violations
failed
```

直接调用符号：

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
def add_native(self, native_metrics, *, duration):
```

实现说明：

> Accumulate evidence from any visited seed without changing priority.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._native_parameters
self._parameters.update
self._parameters
parameters.tolist
```

写入目标：

```python
parameters
```

直接调用符号：

```python
self._native_parameters
self._parameters.update
parameters.tolist
```

## `AttemptWitnesses.add_validation`

[定义位置](../../../manipulation/planning_diagnostics.py#L102)。

输入签名：

```python
def add_validation(self, validation):
```

实现说明：

> Accumulate completed validation failures without changing priority.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._parameters.update
self._parameters
self._validation_parameters(validation).tolist
self._validation_parameters
```

写入目标：

无显式条目。

直接调用符号：

```python
self._parameters.update
self._validation_parameters(validation).tolist
self._validation_parameters
```

## `AttemptWitnesses.set_current`

[定义位置](../../../manipulation/planning_diagnostics.py#L106)。

输入签名：

```python
def set_current(self, native_metrics, *, validation=None, duration):
```

实现说明：

> Select one continuation candidate's native and independent failures.
>
> The caller must invoke this only after candidate selection. Observers
> and unselected seeds use add_native/add_validation instead. Validate
> both inputs before mutation, so a malformed completed validation cannot
> partly replace current priority or add unbound native evidence.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._native_parameters
np.empty
self._validation_parameters
np.r_[native, independent].tolist
np.r_
self._parameters.update
self._parameters
```

写入目标：

```python
native
independent
current
self._current
self._last_selection
```

直接调用符号：

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
def select_parameters(self, *, fixed_parameters, regular_parameters, max_points):
```

实现说明：

> Schedule bounded current failures fairly; retain all history.
>
> Fixed anchors and regular optimizer samples cover only their supplied
> exact representations. The caller supplies native float32 aliases for
> fixed anchors when those aliases represent actual existing queries.
> Among remaining current failures, choose the least frequently selected
> times first and break ties by time. Boolean collision failures and
> process errors in different units therefore never compete by magnitude.
> Pending failures remain explicit even at zero capacity. A changed query
> set is a gradient-work decision, not evidence of positive signed slack
> or a claim that an omitted historical failure cannot recur.

显式输出分支：

```python
np.asarray(sorted(selected), dtype=float)
```

读取的对象属性：

```python
self._validated_parameters
fixed.tolist
regular.tolist
self._current
self._selection_counts.get
self._selection_counts
self._parameters
np.asarray
```

写入目标：

```python
fixed
regular
covered
eligible
ordered
selected
parameter
self._selection_counts[parameter]
self._last_selection
value
```

直接调用符号：

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

实现说明：

> Capture process/self residual times before anchor columns are folded.
>
> Inputs use the already ordered native query layout ``[batch, time, ...]``.
> Each component's trailing dimensions are reduced only within that same
> state. The resulting rows share the existing scene-trace schema, so the
> caller can attach them to the exact named joint-state trace without
> repeating FK, process evaluation or a source-geometry query.

显式输出分支：

```python
rows
```

读取的对象属性：

```python
value.detach().cpu().numpy().copy
value.detach().cpu().numpy
value.detach().cpu
value.detach
times.shape
components.items
values.ndim
values.shape
values.reshape(*shape, -1).max
values.reshape
np.argwhere
np.isfinite
value.shape
parameters.shape
q.tolist
p.tolist
rotation.tolist
rows.append
```

写入目标：

```python
cpu
shape
rows
state
parameters
name
tensor
values
batch
column
value
time_s
phase
q
p
rotation
row
row['normalized_time']
```

直接调用符号：

```python
value.detach().cpu().numpy().copy
value.detach().cpu().numpy
value.detach().cpu
value.detach
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

实现说明：

> Retain every positive/nonfinite hard component in its original mode.
>
> Unknown observation coverage remains informational under the existing
> scene law. Missing release anchors are hard requirements under that same
> law. A mode index refers to the pre-fold query, never to its maximum.

显式输出分支：

```python
result
```

读取的对象属性：

```python
value.detach().cpu().numpy().copy
value.detach().cpu().numpy
value.detach().cpu
value.detach
components.items
name.startswith
times.shape
np.argwhere
np.isfinite
result.append
q.tolist
p.tolist
rotation.tolist
```

写入目标：

```python
cpu
hard
name
value
result
state
width
tensor
values
batch
column
time_s
phase
permitted
q
p
rotation
```

直接调用符号：

```python
value.detach().cpu().numpy().copy
value.detach().cpu().numpy
value.detach().cpu
value.detach
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

实现说明：

> Bind by complete typed q/dq bytes, independently of native seed ranking.
>
> Native result sorting can permute seeds after metrics evaluation. Comparing
> the original named states avoids attributing another seed's scene witness.
> Identical q/dq rows share the same fixed-clock scene evaluation.

显式输出分支：

```python
{'status': 'unavailable', 'reason': 'No source metrics trace was retained'}
{'status': 'error', 'error': trace['error']}
{'status': 'unmatched', 'reason': 'No exact named q/dq input matches this result seed'}
{'status': 'recorded', 'source_batch_index': index, 'component_failures': failures, 'world_geometry': deepcopy(trace['world_geometry']), 'world_geometry_scope': 'Worst witness across this metrics call; joint/time identity is explicit', 'scope': 'Original source components before mode maximum, constraint weighting and anchor folding'}
```

读取的对象属性：

```python
state.joint_names.index
state.joint_names
getattr(state, name)[0, seed_index].detach().cpu().numpy
getattr(state, name)[0, seed_index].detach().cpu
getattr(state, name)[0, seed_index].detach
actual.dtype
stored[index].dtype
actual.shape
stored[index].shape
actual.tobytes
stored[index].tobytes
trace['world_geometry'].get
```

写入目标：

```python
columns
name
arrays
matches
index
actual
stored
failures
row
witnesses
references
reference
row['geometry_witnesses']
```

直接调用符号：

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
