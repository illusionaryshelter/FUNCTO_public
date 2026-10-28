# `manipulation/pair_evidence.py` 实现契约

代码基线：`WORKING_IMPLEMENTATION`。
共享验收的数据所有权及调用次序见[共享验收组件](../10-shared-candidate-validation.md)。

职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `PairDescriptor`

[定义位置](../../../manipulation/pair_evidence.py#L25)。

```python
class PairDescriptor():
```

类字段：

```python
component: str
first: str
second: str
entity_first: str
entity_second: str
policy: str = None
policy_source: str = None
```

实现说明：

> Declared pair identity, shared across every state of one scene query.
>
> ``first`` and ``second`` are the original display names.  Namespaced entity
> identities distinguish robot solids, observed owners and the background.
> Native policy/source fields retain the caller's fixed PairQuery contract.

## `PairDescriptor.key`

[定义位置](../../../manipulation/pair_evidence.py#L42)。

输入签名：

```python
def key(self):
```

显式输出分支：

```python
(self.component, self.entity_first, self.entity_second)
```

读取的对象属性：

```python
self.component
self.entity_first
self.entity_second
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `PairDescriptor.display`

[定义位置](../../../manipulation/pair_evidence.py#L45)。

输入签名：

```python
def display(self):
```

实现说明：

> Build the small identity record used by final interval reports.

显式输出分支：

```python
{'component': self.component, 'first': self.first, 'second': self.second, 'geometry_entity_key': {'first': self.entity_first, 'second': self.entity_second}}
```

读取的对象属性：

```python
self.component
self.first
self.second
self.entity_first
self.entity_second
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `PairBatch`

[定义位置](../../../manipulation/pair_evidence.py#L51)。

```python
class PairBatch():
```

类字段：

无显式条目。

实现说明：

> Complete rectangular [expanded query state, declared entity pair] data.
>
> A physical time may have several relation-side queries.  Its expanded
> ``state_index`` and actual progress/permission progress remain independent
> of ``query_index`` and ``side``, which are assigned by the scene-boundary
> caller.  This object owns one call only; subsequent calls allocate another
> object.  Array users read the result after ``seal``.  Observed payloads are
> transferred from their original producer and are subsequently read only.

## `PairBatch.__init__`

[定义位置](../../../manipulation/pair_evidence.py#L62)。

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
self.descriptors
self.pair_columns
self.observed_payloads
self._observed_permissions
self.native_numeric
self.native_count
self._query_layout_bound
```

直接调用符号：

无显式条目。

## `PairBatch.bind_clock`

[定义位置](../../../manipulation/pair_evidence.py#L71)。

输入签名：

```python
def bind_clock(self, times, progress, permission_progress):
```

实现说明：

> Own the original clock dtype and expanded two-axis query layout.

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray(times).shape
np.asarray
np.indices(self.state_shape, dtype=np.intp).reshape(2, -1).T.copy
np.indices(self.state_shape, dtype=np.intp).reshape(2, -1).T
np.indices(self.state_shape, dtype=np.intp).reshape
np.indices
self.state_shape
np.intp
np.asarray(times).reshape(-1).copy
np.asarray(times).reshape
np.asarray(progress).reshape(-1).copy
np.asarray(progress).reshape
np.asarray(permission_progress).reshape(-1).copy
np.asarray(permission_progress).reshape
self.time_s.size
self.time_s
self.state_index.copy
self.state_index
np.zeros
self.state_count
np.int8
```

写入目标：

```python
self.state_shape
self.state_index
self.time_s
self.progress
self.permission_progress
self.state_count
self.query_index
self.side
```

直接调用符号：

```python
tuple
np.asarray
np.indices(self.state_shape, dtype=np.intp).reshape(2, -1).T.copy
np.indices(self.state_shape, dtype=np.intp).reshape
np.indices
np.asarray(times).reshape(-1).copy
np.asarray(times).reshape
np.asarray(progress).reshape(-1).copy
np.asarray(progress).reshape
np.asarray(permission_progress).reshape(-1).copy
np.asarray(permission_progress).reshape
self.state_index.copy
np.zeros
```

## `PairBatch.native_pair_count`

[定义位置](../../../manipulation/pair_evidence.py#L83)。

输入签名：

```python
def native_pair_count(self):
```

实现说明：

> Number of leading descriptor columns owned by the native kernel.

显式输出分支：

```python
self.native_count
```

读取的对象属性：

```python
self.native_count
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `PairBatch._register`

[定义位置](../../../manipulation/pair_evidence.py#L87)。

输入签名：

```python
def _register(self, descriptor):
```

显式输出分支：

```python
column
```

读取的对象属性：

```python
self.descriptors
self.descriptors.append
self.pair_columns
descriptor.key
```

写入目标：

```python
column
self.pair_columns[descriptor.key]
```

直接调用符号：

```python
len
self.descriptors.append
```

## `PairBatch.begin_native`

[定义位置](../../../manipulation/pair_evidence.py#L93)。

输入签名：

```python
def begin_native(self, pairs, *, boundary, include_bounds):
```

实现说明：

> Reserve native output once; each unchanged kernel chunk fills a slice.
>
> The transient native chunk retains the original kernel's allocation
> and error boundary.  Its values are copied into the one result owner;
> no Python float/list objects are created for clear native pairs.

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.empty
self.state_count
np.float64
np.zeros
self._register
pair.first
pair.second
pair.policy
pair.source
```

写入目标：

```python
self.native_count
self.native_boundary
self.native_include_bounds
self.native_numeric
self._native_unmodelled
pair
```

直接调用符号：

```python
len
float
bool
np.empty
np.zeros
self._register
PairDescriptor
```

## `PairBatch.retain_native_chunk`

[定义位置](../../../manipulation/pair_evidence.py#L109)。

输入签名：

```python
def retain_native_chunk(self, start, numeric, unmodelled):
```

实现说明：

> Retain every raw column, including rows whose motion is unmodelled.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.native_numeric
self._native_unmodelled
```

写入目标：

```python
end
self.native_numeric[start:end]
self._native_unmodelled[start:end]
```

直接调用符号：

```python
len
```

## `PairBatch.append_observed`

[定义位置](../../../manipulation/pair_evidence.py#L115)。

输入签名：

```python
def append_observed(self, index, component, first, second, evidence, *, geometry_entity_key, permission_kind='none'):
```

实现说明：

> Keep one original observed/declared-relation payload without wrapping it.
>
> The registered scene enumerates each observed pair over every expanded
> state, including skipped internal or uncertain states.  Model-specific
> payload keys are preserved intact; a twelve-column native layout does
> not define the schema of an observed-field query.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.pair_columns.get
self.pair_columns
descriptor.key
self._register
self.observed_payloads
self.state_count
self._observed_permissions
np.empty
np.int8
np.ravel_multi_index
self.state_shape
```

写入目标：

```python
descriptor
column
self.observed_payloads[column]
self._observed_permissions[column]
state
self.observed_payloads[column][state]
self._observed_permissions[column][state]
```

直接调用符号：

```python
PairDescriptor
self.pair_columns.get
self._register
np.empty
int
np.ravel_multi_index
```

## `PairBatch.seal`

[定义位置](../../../manipulation/pair_evidence.py#L135)。

输入签名：

```python
def seal(self):
```

实现说明：

> Prepare common scalar columns and one original traversal index.
>
> Missing bounds are represented by a separate Boolean validity column.
> A numeric NaN therefore remains a numeric NaN; it is not substituted
> for an absent value.  The complete original payload still determines
> field presence when a final row is materialized.

显式输出分支：

```python
self
```

读取的对象属性：

```python
self.state_count
self.descriptors
np.empty
np.int8
np.zeros
np.float64
self.native_count
self.native_numeric
np.where
self.native_boundary
self.status
self._native_unmodelled
self.witness_present
self.native_include_bounds
self.lower
self.lower_valid
self.observed_payloads.items
self.observed_payloads
self.permission_kind
self._observed_permissions
self.full_lower
self.full_lower_valid
self.upper
self.upper_valid
payload.get
self.empty_set
np.indices((self.state_count, count), dtype=np.intp).reshape(2, -1).T
np.indices((self.state_count, count), dtype=np.intp).reshape
np.indices
np.intp
np.indices((len(self.descriptors) - count, self.state_count), dtype=np.intp).reshape(2, -1)[::-1].T
np.indices((len(self.descriptors) - count, self.state_count), dtype=np.intp).reshape
np.concatenate
self.time_s
self.progress
self.permission_progress
self.state_index
self.order
value.flags
self.native_numeric.flags
```

写入目标：

```python
shape
self.status
self.permission_kind
self.lower
self.full_lower
self.upper
self.lower_valid
self.full_lower_valid
self.upper_valid
self.empty_set
self.witness_present
count
distance
status
self.status[:, :count]
self.witness_present[:, :count]
self.lower[:, :count]
self.lower_valid[:, :count]
column
payloads
self.permission_kind[:, column]
state
payload
self.status[state, column]
field
values
valid
value
values[state, column]
valid[state, column]
self.empty_set[state, column]
self.witness_present[state, column]
native
observed
observed[:, 1]
self.order
self.descriptors
value.flags.writeable
self.native_numeric.flags.writeable
```

直接调用符号：

```python
len
np.empty
np.zeros
np.where
self.observed_payloads.items
enumerate
payload.get
bool
np.indices((self.state_count, count), dtype=np.intp).reshape
np.indices
np.indices((len(self.descriptors) - count, self.state_count), dtype=np.intp).reshape
np.concatenate
tuple
```

## `PairBatch.set_query_layout`

[定义位置](../../../manipulation/pair_evidence.py#L197)。

输入签名：

```python
def set_query_layout(self, query_index, side):
```

实现说明：

> Bind exact physical query identities after relation-side expansion.

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray(query_index, dtype=np.intp).reshape(self.state_count, 2).copy
np.asarray(query_index, dtype=np.intp).reshape
np.asarray
np.intp
self.state_count
np.asarray(side, dtype=np.int8).reshape(self.state_count).copy
np.asarray(side, dtype=np.int8).reshape
np.int8
self.query_index.flags
self.query_index
self.side.flags
self.side
```

写入目标：

```python
self.query_index
self.side
self.query_index.flags.writeable
self.side.flags.writeable
self._query_layout_bound
```

直接调用符号：

```python
np.asarray(query_index, dtype=np.intp).reshape(self.state_count, 2).copy
np.asarray(query_index, dtype=np.intp).reshape
np.asarray
np.asarray(side, dtype=np.int8).reshape(self.state_count).copy
np.asarray(side, dtype=np.int8).reshape
```

## `PairBatch.materialize`

[定义位置](../../../manipulation/pair_evidence.py#L204)。

输入签名：

```python
def materialize(self, state, column, *, query_layout=True):
```

实现说明：

> Return one original complete row at a diagnostic/output boundary.

显式输出分支：

```python
row
```

读取的对象属性：

```python
self.descriptors
self.state_index[state].tolist
self.state_index
self.time_s
self.progress
self.permission_progress
descriptor.display
self.permission_kind
self.native_count
self._native_unmodelled
row.update
self.native_numeric
self.native_boundary
descriptor.policy
descriptor.policy_source
numeric[1:4].tolist
numeric[4:7].tolist
numeric[7:10].tolist
self.native_include_bounds
self.observed_payloads
self._query_layout_bound
self.query_index[state].tolist
self.query_index
self.side
```

写入目标：

```python
state
column
descriptor
row
numeric
distance
relation
```

直接调用符号：

```python
int
self.state_index[state].tolist
float
descriptor.display
row.update
numeric[1:4].tolist
numeric[4:7].tolist
numeric[7:10].tolist
self.query_index[state].tolist
```

## `EvidenceState`

[定义位置](../../../manipulation/pair_evidence.py#L238)。

```python
class EvidenceState():
```

类字段：

```python
time_s: float
side: int
progress: float
residual: float
batch: PairBatch
source_rows: tuple
```

实现说明：

> One physical time/relation side, retaining every expanded source row.
>
> Rectangular scene queries emit every registered pair for every source
> row, including skipped relations. Duplicate rows keep their original
> order; interval lookup has the original dictionary's last-row semantics.
> Progress outside the block is the original physical query's progress.

## `EvidenceState.pair_columns`

[定义位置](../../../manipulation/pair_evidence.py#L254)。

输入签名：

```python
def pair_columns(self):
```

显式输出分支：

```python
self.batch.pair_columns if self.source_rows else {}
```

读取的对象属性：

```python
self.source_rows
self.batch.pair_columns
self.batch
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `EvidenceState.source`

[定义位置](../../../manipulation/pair_evidence.py#L258)。

输入签名：

```python
def source(self):
```

显式输出分支：

```python
self.source_rows[-1]
```

读取的对象属性：

```python
self.source_rows
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `EvidenceState.tokens`

[定义位置](../../../manipulation/pair_evidence.py#L261)。

输入签名：

```python
def tokens(self):
```

实现说明：

> Yield indices in the original within-state evidence order.

显式输出分支：

```python
(source, pair)
(source, pair)
```

读取的对象属性：

```python
self.source_rows
self.batch.native_pair_count
self.batch
self.batch.descriptors
```

写入目标：

```python
source
pair
```

直接调用符号：

```python
range
len
```

## `EvidenceState.pair_record`

[定义位置](../../../manipulation/pair_evidence.py#L270)。

输入签名：

```python
def pair_record(self, pair):
```

显式输出分支：

```python
{key: value for (key, value) in row.items() if key not in {'state_index', 'query_index', 'time_s', 'side'}}
```

读取的对象属性：

```python
self.batch.materialize
self.batch
self.source
row.items
```

写入目标：

```python
row
key
value
```

直接调用符号：

```python
self.batch.materialize
row.items
```

## `_geometry_query_states`

[定义位置](../../../manipulation/pair_evidence.py#L275)。

输入签名：

```python
def _geometry_query_states(evidence, times, progress, residuals):
```

实现说明：

> Group state indices once; entity identities stay in the shared block.

显式输出分支：

```python
[EvidenceState(time, side, float(progress[columns[time, side]]), float(residuals[columns[time, side]]), evidence, tuple(sources)) for ((time, side), sources) in groups.items()]
```

读取的对象属性：

```python
evidence.descriptors
evidence.query_index
evidence.side
groups.setdefault(key, []).append
groups.setdefault
columns.setdefault
groups.items
```

写入目标：

```python
groups
time
columns
column
source
_
key
side
sources
```

直接调用符号：

```python
float
enumerate
int
groups.setdefault(key, []).append
groups.setdefault
columns.setdefault
EvidenceState
tuple
groups.items
```

## `selected_witness_records`

[定义位置](../../../manipulation/pair_evidence.py#L289)。

输入签名：

```python
def selected_witness_records(*, states=None, batch=None):
```

实现说明：

> Select original witnesses from arrays, then build only selected records.
>
> Initial task checks consume grouped state order. Later interval queries
> consume the scene's original production order. These two orders differ
> for observed pairs; both preserve first-on-equal ranking and discoveries.

显式输出分支：

```python
(records, total)
```

读取的对象属性：

```python
block.status
intersections.append
block.witness_present
block.descriptors
descriptor.entity_first
descriptor.entity_second
block.native_pair_count
block.lower_valid
block.lower
block.upper_valid
block.upper
block.observed_payloads
payload.get
np.inf
closest.values
block.materialize
row.items
records.append
row.get
```

写入目标：

```python
intersections
closest
total
block
source
pair
time
progress
status
descriptor
key
lower
upper
payload
rank
closest[key]
selected
value
records
row
entity
```

直接调用符号：

```python
sources
intersections.append
float
payload.get
closest.values
block.materialize
row.items
records.append
row.get
```

## `selected_witness_records.sources`

[定义位置](../../../manipulation/pair_evidence.py#L299)。

输入签名：

```python
def sources():
```

显式输出分支：

```python
(state.batch, source, pair, state.time_s, state.progress)
(batch, source, pair, batch.time_s[source], batch.progress[source])
```

读取的对象属性：

```python
state.tokens
state.batch
state.time_s
state.progress
batch.order
batch.time_s
batch.progress
```

写入目标：

```python
state
source
pair
```

直接调用符号：

```python
state.tokens
```
