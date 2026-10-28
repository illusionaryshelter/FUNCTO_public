# `tests/planning/recorded_task_planning.py` 实现契约

代码基线：`本轮实现提交待回填`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_json_value`

[定义位置](../../../tests/planning/recorded_task_planning.py#L24)。

输入签名：

```python
def _json_value(value):
```

显式输出分支：

```python
value.tolist()
value.item()
```

读取的对象属性：

```python
np.ndarray
value.tolist
np.generic
value.item
type(value).__name__
```

写入目标：

无显式条目。

直接调用符号：

```python
isinstance
value.tolist
value.item
TypeError
type
```

## `_diagnostic_value`

[定义位置](../../../tests/planning/recorded_task_planning.py#L33)。

输入签名：

```python
def _diagnostic_value(value):
```

实现说明：

> Convert owned CPU evidence without silently dropping NaN/Inf failures.

显式输出分支：

```python
_diagnostic_value(value.tolist())
_diagnostic_value(value.item())
{key: _diagnostic_value(item) for (key, item) in value.items()}
[_diagnostic_value(item) for item in value]
'NaN' if np.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
value
```

读取的对象属性：

```python
np.ndarray
value.tolist
np.generic
value.item
value.items
np.isfinite
np.isnan
```

写入目标：

```python
key
item
```

直接调用符号：

```python
isinstance
_diagnostic_value
value.tolist
value.item
value.items
np.isfinite
np.isnan
```

## `_write_json_value`

[定义位置](../../../tests/planning/recorded_task_planning.py#L48)。

输入签名：

```python
def _write_json_value(path, document):
```

实现说明：

> Replace one complete value document using the existing numeric rules.

显式输出分支：

无显式条目。

读取的对象属性：

```python
json.dumps
path.with_name
path.name
temporary.write_text
temporary.replace
temporary.unlink
```

写入目标：

```python
path
rendered
temporary
```

直接调用符号：

```python
Path
json.dumps
_diagnostic_value
path.with_name
temporary.write_text
temporary.replace
temporary.unlink
```

## `_write_diagnostic_json`

[定义位置](../../../tests/planning/recorded_task_planning.py#L65)。

输入签名：

```python
def _write_diagnostic_json(path, document):
```

实现说明：

> Write a report and relative references to its full shared evidence.

显式输出分支：

无显式条目。

读取的对象属性：

```python
path.parent
```

写入目标：

```python
path
```

直接调用符号：

```python
Path
_write_json_value
externalize_evidence
```

## `save_curve`

[定义位置](../../../tests/planning/recorded_task_planning.py#L72)。

输入签名：

```python
def save_curve(path, curve, *, classification, execution_limits, fk=None, validation_status='completed', joint_limits=None, planning_result=None, source=None):
```

实现说明：

> Preserve the exact exported cubic, event identities and optional FK.
>
> ``fk`` contains independently evaluated native FK of this same exported
> curve, in the problem reference frame. Native planner callers also supply
> their effective bounds; the shared production calculus then reports both
> the full curve and its original executable prefix. This recorder never
> changes classification based on an additional report.

显式输出分支：

无显式条目。

读取的对象属性：

```python
planning_result.trajectory
ExecutionLimits.from_mapping
limits.describe
np.asarray
curve.joint_names
curve.times
curve.position
curve.velocity
curve.progress_u
curve.progress_s
curve.finger_trajectory
arrays.update
fingers.joint_names
fingers.times
fingers.position
fingers.velocity
binding.get
curve.executable_until
curve.duration
fingers.describe
curve.satisfied_dependencies
curve.events
curve.conditions
curve.residuals
curve.next_verification
path.parent
json.dumps
np.savez_compressed
path.with_suffix('.json').write_text
path.with_suffix
```

写入目标：

```python
binding
limits
binding['execution_limits']
planning_result.execution_binding
arrays
fingers
derivative_metadata
full
prefix
metadata
rendered
```

直接调用符号：

```python
ValueError
ExecutionLimits.from_mapping
check_result
limits.describe
dict
np.asarray
arrays.update
binding.get
cubic_derivative_report
deepcopy
fingers.describe
list
externalize_evidence
json.dumps
len
np.savez_compressed
path.with_suffix('.json').write_text
path.with_suffix
```

## `plan_and_record`

[定义位置](../../../tests/planning/recorded_task_planning.py#L160)。

输入签名：

```python
def plan_and_record(problem, scene, robot, output, *, robot_validator, budget, provenance, execution_limits, backend, joint_reference=None, continuation=None, rebind=None, request_ledger=None):
```

实现说明：

> Call the bound backend once; persistence cannot initiate another solve.
>
> The caller owns the backend lifetime so successive requests can reuse its
> native models. Diagnostic curves have no formal binding and cannot enter
> the simulation executor. No monkeypatch, observer-side export, source
> geometry query, or candidate-sized tensor history is installed here.

显式输出分支：

```python
result
```

读取的对象属性：

```python
output.mkdir
time.monotonic
problem.phase_count
planning.plan_trajectory
result.diagnostics.pop
result.diagnostics
report.update
result.success
result.request_ledger
ledger_path.resolve
result.diagnostics.get
result.trajectory
diagnostic.residuals
report['candidates'].append
type(error).__name__
(output / 'exception.txt').write_text
traceback.format_exc
resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
resource.getrusage
resource.RUSAGE_SELF
torch.cuda.is_initialized
torch.cuda
torch.cuda.max_memory_allocated
torch.cuda.max_memory_reserved
```

写入目标：

```python
output
started
report
result
diagnostic
ledger_path
report['request_ledger_path']
bounds
entry
entry['curve_saved']
entry['diagnostic_error']
report['elapsed_seconds']
report['peak_rss_bytes']
```

直接调用符号：

```python
Path
output.mkdir
time.monotonic
asdict
deepcopy
planning.plan_trajectory
result.diagnostics.pop
report.update
_write_diagnostic_json
str
ledger_path.resolve
result.diagnostics.get
save_curve
report['candidates'].append
type
(output / 'exception.txt').write_text
traceback.format_exc
resource.getrusage
torch.cuda.is_initialized
torch.cuda.max_memory_allocated
torch.cuda.max_memory_reserved
```
