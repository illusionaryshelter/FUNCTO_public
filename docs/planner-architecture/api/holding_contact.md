# `manipulation/holding_contact.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../03-attachment-contact-and-hand.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `HoldingContact`

[定义位置](../../../manipulation/holding_contact.py#L12)。

```python
class HoldingContact:
    rule_id: str
    condition_id: str
    target_owner: str
    material_id: str
    native_geometry: str
    sphere_indices: tuple
    interval: tuple
    source: str
```

该类型的直接方法：`__post_init__`。

## `HoldingContact.__post_init__`

[定义位置](../../../manipulation/holding_contact.py#L31)。

输入签名：

```python
def HoldingContact.__post_init__(self):
```

直接读取的对象属性：`self.sphere_indices`、`self.interval`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
_text
getattr
tuple
len
set
any
type
ValueError
object.__setattr__
_valid_interval
```

## `bind_holding_contacts`

[定义位置](../../../manipulation/holding_contact.py#L42)。

输入签名：

```python
def bind_holding_contacts(problem, material_binding, declarations, *, source):
```

直接读取的对象属性：`problem.conditions`、`problem.phases`。

输出分支的源码表达式：

```python
tuple(contacts)
```

直接写入的状态或数组位置：`replacements[identity]`、`problem.conditions`。

直接调用：

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

## `validate_holding_materials`

[定义位置](../../../manipulation/holding_contact.py#L81)。

输入签名：

```python
def validate_holding_materials(contacts, owners, sphere_materials, native_materials, collision_bodies):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`sphere_geometry[index]`。

直接调用：

```python
set
tuple
geometry_indices.setdefault(name, set()).update
geometry_indices.setdefault
isinstance
ValueError
native_materials.get
geometry_indices.get
any
len
sphere_geometry.get
identities.add
pairs.add
```

## `validate_holding_conditions`

[定义位置](../../../manipulation/holding_contact.py#L115)。

输入签名：

```python
def validate_holding_conditions(contacts, owners, conditions):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`actual[:3, 3]`、`actual[:3, :3]`。

直接调用：

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
