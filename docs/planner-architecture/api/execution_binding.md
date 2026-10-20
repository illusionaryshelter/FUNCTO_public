# `manipulation/execution_binding.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `plain`

[定义位置](../../../manipulation/execution_binding.py#L17)。

输入签名：

```python
def plain(value):
```

输出分支的源码表达式：

```python
{item.name: plain(getattr(value, item.name)) for item in fields(value) if not item.name.startswith('_')}
plain(value.tolist())
plain(value.item())
{str(key): plain(item) for (key, item) in value.items()}
[plain(item) for item in value]
value
value if np.isfinite(value) else {'unbounded': '+' if value > 0 else '-'}
```

直接调用：

```python
is_dataclass
plain
getattr
fields
item.name.startswith
isinstance
value.tolist
value.item
str
value.items
np.isnan
ValueError
np.isfinite
TypeError
type
```

## `curve_snapshot`

[定义位置](../../../manipulation/execution_binding.py#L43)。

输入签名：

```python
def curve_snapshot(trajectory):
```

直接读取的对象属性：`trajectory.joint_names`、`trajectory.times`、`trajectory.position`、`trajectory.velocity`、`trajectory._curve`、`trajectory.progress_u`、`trajectory.progress_s`、`trajectory.events`、`trajectory.conditions`、`trajectory.duration`、`trajectory.executable_until`、`trajectory.next_verification`、`trajectory.satisfied_dependencies`、`trajectory.finger_trajectory`。

输出分支的源码表达式：

```python
plain({'joint_names': trajectory.joint_names, 'times': trajectory.times, 'position': trajectory.position, 'velocity': trajectory.velocity, 'coefficients': trajectory._curve.c, 'progress_u': trajectory.progress_u, 'progress_s': trajectory.progress_s, 'events': trajectory.events, 'conditions': trajectory.conditions, 'duration': trajectory.duration, 'executable_until': trajectory.executable_until, 'next_verification': trajectory.next_verification, 'satisfied_dependencies': trajectory.satisfied_dependencies, 'fingers': None if trajectory.finger_trajectory is None else trajectory.finger_trajectory.describe()})
```

直接调用：

```python
plain
trajectory.finger_trajectory.describe
```

## `bind_candidate`

[定义位置](../../../manipulation/execution_binding.py#L59)。

输入签名：

```python
def bind_candidate(trajectory, *, problem, scene, validation, native_report):
```

直接读取的对象属性：`scene.to_execution_binding`。

输出分支的源码表达式：

```python
{'candidate_id': candidate_id, 'curve': curve_snapshot(trajectory), **deepcopy(inputs), 'validation': {'candidate_id': candidate_id, 'completed': True, 'inputs': inputs, 'scope': 'same accepted joint curve, event clock and online scene; actual entrance preflight pending', 'export': plain(validation), 'native': plain(native_report)}, 'execution_limits': None}
```

直接调用：

```python
np.asarray
native_report.get
np.isfinite(queries).all
np.isfinite
np.isfinite(violations).all
np.any
ValueError
str
uuid4
plain
scene.to_execution_binding
curve_snapshot
deepcopy
```

## `_relaxed_pose_contract`

[定义位置](../../../manipulation/execution_binding.py#L90)。

输入签名：

```python
def _relaxed_pose_contract(problem, identities, changes, ledger, request_id):
```

输出分支的源码表达式：

```python
(expected, revised_ledger)
```

直接写入的状态或数组位置：`revised_ledger['request_id']`、`contract[name]`、`goal[name]`。

直接调用：

```python
isinstance
ValueError
deepcopy
expected.get
identities.get
len
set
ledger.get
request_id.strip
type
seen.add
indexes.index
revised_ledger.get('obligation_contract', {}).get
revised_ledger.get
np.isfinite
contract.get
```

## `relax_pose_goal_tolerances`

[定义位置](../../../manipulation/execution_binding.py#L150)。

输入签名：

```python
def relax_pose_goal_tolerances(result, *, changes, source):
```

直接读取的对象属性：`result.request_ledger`、`result.success`、`result.trajectory`、`result.diagnostics`。

输出分支的源码表达式：

```python
revised
```

直接写入的状态或数组位置：`binding['request_ledger']`。

直接调用：

```python
check_result
isinstance
source.strip
ValueError
binding.get
ledger.get
plain
_relaxed_pose_contract
str
uuid4
binding.update
binding['validation'].update
deepcopy
PlanningResult
```

## `check_result`

[定义位置](../../../manipulation/execution_binding.py#L189)。

输入签名：

```python
def check_result(result, *, execution_limits=None):
```

直接读取的对象属性：`result.success`、`result.trajectory`、`result.execution_binding`。

输出分支的源码表达式：

```python
deepcopy(binding)
```

直接调用：

```python
isinstance
ValueError
binding.get
binding['candidate_id'].strip
evidence.get
evidence.get('native', {}).get
np.asarray
evidence.get('export', {}).get
np.isfinite(violations).all
np.isfinite
np.any
np.isfinite(queries).all
curve_snapshot
set
revision['source_candidate_id'].strip
evidence['inputs'].get
_relaxed_pose_contract
ledger.get
plain
execution_limits.describe
deepcopy
```

## `validate_controller_policy`

[定义位置](../../../manipulation/execution_binding.py#L241)。

输入签名：

```python
def validate_controller_policy(value, *, joint_names):
```

输出分支的源码表达式：

```python
result
```

直接调用：

```python
isinstance
set
ValueError
plain
tuple
np.asarray
len
np.isfinite(gains).all
np.isfinite
np.any
```

## `validate_result_only_goal_ids`

[定义位置](../../../manipulation/execution_binding.py#L269)。

输入签名：

```python
def validate_result_only_goal_ids(binding, values):
```

输出分支的源码表达式：

```python
selected
```

直接调用：

```python
isinstance
ValueError
binding.get
len
type
list
identities.get
selected.append
binding['problem'].get
```

## `rebind_execution_result`

[定义位置](../../../manipulation/execution_binding.py#L308)。

输入签名：

```python
def rebind_execution_result(result, *, controller_policy, result_only_goal_ids, finger_targets, execution_limits, source):
```

直接读取的对象属性：`result.request_ledger`、`result.trajectory`、`result.success`、`result.diagnostics`。

输出分支的源码表达式：

```python
PlanningResult(result.success, result.trajectory, deepcopy(result.diagnostics), binding, deepcopy(ledger) if 'request_ledger' in binding else None)
```

直接写入的状态或数组位置：`binding['validation']['source_candidate_id']`、`binding['validation']['candidate_id']`。

直接调用：

```python
check_result
isinstance
source.strip
ValueError
binding.get
ledger.get
validate_controller_policy
validate_result_only_goal_ids
execution_limits.describe
ExecutionLimits.from_mapping(previous_limits).describe
ExecutionLimits.from_mapping
any
list
deepcopy
previous_policy.get
str
uuid4
binding.update
plain
PlanningResult
```
