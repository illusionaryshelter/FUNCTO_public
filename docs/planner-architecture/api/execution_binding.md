# `manipulation/execution_binding.py` 实现契约

代码基线：`CURRENT_IMPLEMENTATION`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `plain`

[定义位置](../../../manipulation/execution_binding.py#L21)。

输入签名：

```python
def plain(value):
```

实现说明：

> Copy declared input fields without retaining solver/cache objects.
>
> Unbounded intervals are explicit JSON data; NaN never represents a
> missing measurement. Private dataclass caches are deliberately excluded.

显式输出分支：

```python
plain(value.to_record())
{item.name: plain(getattr(value, item.name)) for item in fields(value) if not item.name.startswith('_')}
plain(value.tolist())
plain(value.item())
{str(key): plain(item) for (key, item) in value.items()}
[plain(item) for item in value]
value
value if np.isfinite(value) else {'unbounded': '+' if value > 0 else '-'}
```

读取的对象属性：

```python
value.to_record
item.name
item.name.startswith
np.ndarray
value.tolist
np.generic
value.item
value.items
np.isnan
np.isfinite
type(value).__name__
```

写入目标：

```python
item
key
```

直接调用符号：

```python
is_dataclass
isinstance
plain
value.to_record
getattr
fields
item.name.startswith
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

[定义位置](../../../manipulation/execution_binding.py#L52)。

输入签名：

```python
def curve_snapshot(trajectory):
```

实现说明：

> Record the polynomial, original events and conditional time interval.
>
> Initialization diagnostics and accepted candidates use this same numeric
> record. Execution authorization is produced by ``bind_candidate`` after
> complete candidate qualification.

显式输出分支：

```python
plain({'joint_names': trajectory.joint_names, 'times': trajectory.times, 'position': trajectory.position, 'velocity': trajectory.velocity, 'coefficients': trajectory._curve.c, 'progress_u': trajectory.progress_u, 'progress_s': trajectory.progress_s, 'events': trajectory.events, 'conditions': trajectory.conditions, 'duration': trajectory.duration, 'executable_until': trajectory.executable_until, 'next_verification': trajectory.next_verification, 'satisfied_dependencies': trajectory.satisfied_dependencies, 'fingers': None if trajectory.finger_trajectory is None else trajectory.finger_trajectory.describe()})
```

读取的对象属性：

```python
trajectory.joint_names
trajectory.times
trajectory.position
trajectory.velocity
trajectory._curve.c
trajectory._curve
trajectory.progress_u
trajectory.progress_s
trajectory.events
trajectory.conditions
trajectory.duration
trajectory.executable_until
trajectory.next_verification
trajectory.satisfied_dependencies
trajectory.finger_trajectory
trajectory.finger_trajectory.describe
```

写入目标：

无显式条目。

直接调用符号：

```python
plain
trajectory.finger_trajectory.describe
```

## `bind_candidate`

[定义位置](../../../manipulation/execution_binding.py#L74)。

输入签名：

```python
def bind_candidate(trajectory, *, problem, scene, validation, native_report):
```

实现说明：

> Bind the accepted joint curve and the exact online geometry snapshot.
>
> Native-backend metrics and the unchanged Hermite curve name one candidate,
> whether it was supplied directly or produced by trajectory optimization.
> Execution checks current measured state against this same scene binding;
> an independent source-mesh audit cannot authorize or veto the handoff.

显式输出分支：

```python
{'candidate_id': candidate_id, 'curve': curve_snapshot(trajectory), **deepcopy(inputs), 'validation': {'candidate_id': candidate_id, 'completed': True, 'inputs': inputs, 'scope': 'same accepted joint curve, event clock and online scene; actual entrance preflight pending', 'export': plain(validation), 'native': plain(native_report)}, 'execution_limits': None}
```

读取的对象属性：

```python
np.asarray
native_report.get
violations.size
queries.shape
violations.shape
np.isfinite(queries).all
np.isfinite
np.isfinite(violations).all
np.any
scene.to_execution_binding
```

写入目标：

```python
violations
queries
candidate_id
inputs
```

直接调用符号：

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

[定义位置](../../../manipulation/execution_binding.py#L105)。

输入签名：

```python
def _relaxed_pose_contract(problem, identities, changes, ledger, request_id):
```

实现说明：

> Derive only larger pose-goal sets; every other mathematical input stays.
>
> Goal identities belong to the existing typed pose-event collection. Skill
> selection is the caller's task policy, never a dataset/skill branch here.
> This implication needs no new IK solve: a fixed pose inside the old metric
> balls also lies inside larger concentric balls with the same angle metric.

显式输出分支：

```python
(expected, revised_ledger)
```

读取的对象属性：

```python
expected.get
identities.get
ledger.get
request_id.strip
seen.add
indexes.index
revised_ledger.get('obligation_contract', {}).get
revised_ledger.get
np.isfinite
np.pi
contract.get
```

写入目标：

```python
expected
goals
indexes
revised_ledger
revised_ledger['request_id']
seen
enlarged
change
identity
goal
contract
rows
name
value
old
contract[name]
goal[name]
```

直接调用符号：

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

[定义位置](../../../manipulation/execution_binding.py#L165)。

输入签名：

```python
def relax_pose_goal_tolerances(result, *, changes, source):
```

实现说明：

> Create a new task binding while retaining the original stricter proof.
>
> The archived native/export inputs stay original. Only explicit monotone
> goal changes are derived; geometry, curve, timing, physical limits and
> already spent planning work remain unchanged. This cannot revise an
> executed world or silently continue a request with different obligations.

显式输出分支：

```python
revised
```

读取的对象属性：

```python
source.strip
binding.get
ledger.get
result.request_ledger
binding.update
binding['validation'].update
result.success
result.trajectory
result.diagnostics
```

写入目标：

```python
binding
ledger
changes
problem
revised_ledger
source_id
identity
binding['request_ledger']
revised
```

直接调用符号：

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

[定义位置](../../../manipulation/execution_binding.py#L204)。

输入签名：

```python
def check_result(result, *, execution_limits=None):
```

实现说明：

> Reject a bare curve, stale report, changed clock or candidate mutation.

显式输出分支：

```python
deepcopy(binding)
```

读取的对象属性：

```python
result.success
result.trajectory
result.execution_binding
binding.get
binding['candidate_id'].strip
evidence.get
evidence.get('native', {}).get
np.asarray
evidence.get('export', {}).get
violations.size
queries.shape
violations.shape
np.isfinite(violations).all
np.isfinite
np.any
np.isfinite(queries).all
revision['source_candidate_id'].strip
evidence['inputs'].get
ledger.get
execution_limits.describe
```

写入目标：

```python
binding
evidence
violations
queries
revision
ledger
expected_problem
expected_ledger
declared
```

直接调用符号：

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

[定义位置](../../../manipulation/execution_binding.py#L256)。

输入签名：

```python
def validate_controller_policy(value, *, joint_names):
```

实现说明：

> Validate the one supported named trajectory-control interpretation.

显式输出分支：

```python
result
```

读取的对象属性：

```python
np.asarray
gains.shape
np.isfinite(gains).all
np.isfinite
np.any
```

写入目标：

```python
keys
result
names
key
gains
dt
```

直接调用符号：

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

[定义位置](../../../manipulation/execution_binding.py#L284)。

输入签名：

```python
def validate_result_only_goal_ids(binding, values):
```

实现说明：

> Validate explicit result roles without weakening online consumers.
>
> Every selected goal keeps its original identity and mathematical allowed
> set. Pose, support and mechanism-coordinate accuracy are exclusively
> posthoc at every progress. Robot-feedback and independent task-effect
> dependencies are separate and are never disabled by this list.

显式输出分支：

```python
selected
```

读取的对象属性：

```python
binding.get
identities.get
selected.append
binding['problem'].get
```

写入目标：

```python
identities
selected
value
identity
indexes
requirements
```

直接调用符号：

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

[定义位置](../../../manipulation/execution_binding.py#L323)。

输入签名：

```python
def rebind_execution_result(result, *, controller_policy, result_only_goal_ids, finger_targets, execution_limits, source):
```

实现说明：

> Make a new explicit execution binding from unchanged formal evidence.
>
> This is a device-contract change at a separately declared entrance, not
> continuation of an executed world. The original curve, planning inputs,
> checks and cumulative request ledger remain unchanged. A completed native
> solve is not rerun or relabelled as an actual tracking measurement. The
> executor still has to verify current actual feedback and run its complete
> independent preflight before the first control write.

显式输出分支：

```python
PlanningResult(result.success, result.trajectory, deepcopy(result.diagnostics), binding, deepcopy(ledger) if 'request_ledger' in binding else None)
```

读取的对象属性：

```python
source.strip
binding.get
result.request_ledger
ledger.get
result.trajectory.joint_names
result.trajectory
execution_limits.describe
ExecutionLimits.from_mapping(previous_limits).describe
ExecutionLimits.from_mapping
finger_targets.joint_names
finger_targets.source
finger_targets.events
previous_policy.get
binding.update
result.success
result.diagnostics
```

写入目标：

```python
binding
ledger
policy
result_ids
limits
previous_limits
key
old_observation
new_observation
targets
previous_policy
source_id
identity
binding['validation']['source_candidate_id']
binding['validation']['candidate_id']
```

直接调用符号：

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
