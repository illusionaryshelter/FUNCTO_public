# `manipulation/holding_contact.py` 实现契约

代码基线：`__IMPLEMENTATION_COMMIT__`。
职责和数据语义见[组件说明](../03-attachment-contact-and-hand.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `HoldingContact`

[定义位置](../../../manipulation/holding_contact.py#L12)。

```python
class HoldingContact():
```

类字段：

```python
rule_id: str
condition_id: str
target_owner: str
material_id: str
robot_body: str
interval: tuple
source: str
```

实现说明：

> One registered robot body assigned an internal attachment contact.
> 
> The caller declares this internal contact in its attachment model. The
> same nominal attachment places the owner for external collision checks.
> This does not establish physical Holding. Only this exact material
> solid/owner pair is internal; other robot solids and other owners remain
> obstacles. Release ends the relationship. No historical contact normal,
> gap, target patch, arrival pose, or close command appears in this record.

## `HoldingContact.__post_init__`

[定义位置](../../../manipulation/holding_contact.py#L30)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
object.__setattr__
self.interval
```

写入目标：

```python
name
```

直接调用符号：

```python
_text
getattr
object.__setattr__
_valid_interval
```

## `bind_holding_contacts`

[定义位置](../../../manipulation/holding_contact.py#L36)。

输入签名：

```python
def bind_holding_contacts(problem, material_binding, declarations, *, source):
```

实现说明：

> Bind declared attachment/material associations atomically.
> 
> The caller names the robot's contact materials. Their registered body identities come from the observation partition
> and the model declaration. The condition identifies that same nominal model;
> no measured Holding flag is required or inferred.

显式输出分支：

```python
tuple(contacts)
```

读取的对象属性：

```python
problem.conditions
row.get
phase.metadata
problem.phases
phase.metadata.get
contacts.append
rule_ids.append
replacements.get
```

写入目标：

```python
conditions
row
ids
materials
replacements
contacts
condition
identity
owner
selected
name
spans
phase
interval
rule_ids
material
rule_id
replacements[identity]
problem.conditions
```

直接调用符号：

```python
_text
row.get
isinstance
len
set
ValueError
any
tuple
phase.metadata.get
contacts.append
HoldingContact
rule_ids.append
deepcopy
replacements.get
```

## `validate_holding_conditions`

[定义位置](../../../manipulation/holding_contact.py#L74)。

输入签名：

```python
def validate_holding_conditions(contacts, owners, conditions):
```

实现说明：

> Check model identity, without comparing a goal or a physical object pose.

显式输出分支：

无显式条目。

读取的对象属性：

```python
contact.rule_id
condition.get
row.owner_id
row.attachments
relation.condition_id
consumed.intersection
by_rule.get
contact.condition_id
contact.target_owner
contact.interval
relation.interval
np.eye
relation.position_tool_owner
Rotation.from_quat(np.asarray(relation.quaternion_tool_owner)[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
np.asarray
relation.quaternion_tool_owner
declared.shape
np.isfinite(declared).all
np.isfinite
np.allclose
np.linalg.inv
np.linalg
consumed.update
```

写入目标：

```python
by_rule
contact
consumed
condition
ids
owner
relations
row
relation
identity
actual
actual[:3, 3]
actual[:3, :3]
declared
```

直接调用符号：

```python
set
condition.get
isinstance
len
consumed.intersection
ValueError
by_rule.get
np.eye
Rotation.from_quat(np.asarray(relation.quaternion_tool_owner)[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
np.asarray
np.isfinite(declared).all
np.isfinite
np.allclose
np.linalg.inv
consumed.update
```
