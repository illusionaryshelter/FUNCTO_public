# `tests/planning/recorded_task_planning.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_json_value`

[定义位置](../../../tests/planning/recorded_task_planning.py#L23)。

输入签名：

```python
def _json_value(value):
```

输出分支的源码表达式：

```python
value.tolist()
value.item()
```

直接调用：

```python
isinstance
value.tolist
value.item
TypeError
type
```

## `_diagnostic_value`

[定义位置](../../../tests/planning/recorded_task_planning.py#L32)。

输入签名：

```python
def _diagnostic_value(value):
```

输出分支的源码表达式：

```python
_diagnostic_value(value.tolist())
_diagnostic_value(value.item())
{key: _diagnostic_value(item) for (key, item) in value.items()}
[_diagnostic_value(item) for item in value]
'NaN' if np.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
value
```

直接调用：

```python
isinstance
_diagnostic_value
value.tolist
value.item
value.items
np.isfinite
np.isnan
```

## `_write_diagnostic_json`

[定义位置](../../../tests/planning/recorded_task_planning.py#L47)。

输入签名：

```python
def _write_diagnostic_json(path, document):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
Path
json.dumps
_diagnostic_value
path.with_name
temporary.write_text
temporary.replace
temporary.unlink
```

## `save_curve`

[定义位置](../../../tests/planning/recorded_task_planning.py#L63)。

输入签名：

```python
def save_curve(path, curve, *, classification, execution_limits, fk=None, validation_status='completed', joint_limits=None, planning_result=None):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`binding['execution_limits']`、`planning_result.execution_binding`。

直接调用：

```python
ValueError
ExecutionLimits.from_mapping
check_result
limits.describe
deepcopy
dict
np.asarray
arrays.update
cubic_derivative_report
json.dumps
fingers.describe
list
len
np.savez_compressed
path.with_suffix('.json').write_text
path.with_suffix
```

## `plan_and_record`

[定义位置](../../../tests/planning/recorded_task_planning.py#L137)。

输入签名：

```python
def plan_and_record(problem, scene, robot, output, *, robot_validator, budget, provenance, execution_limits, backend, joint_reference=None, continuation=None, rebind=None, request_ledger=None):
```

直接读取的对象属性：`problem.phase_count`、`result.diagnostics`、`result.success`、`result.request_ledger`、`result.trajectory`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`report['request_ledger_path']`、`entry['curve_saved']`、`entry['diagnostic_error']`、`report['elapsed_seconds']`、`report['peak_rss_bytes']`。

直接调用：

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
