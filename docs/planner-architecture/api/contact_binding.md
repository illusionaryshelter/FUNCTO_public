# `manipulation/contact_binding.py` 实现契约

代码基线：`774e121-plus-unified-workspace`。
职责和数据语义见[组件说明](../03-attachment-contact-and-hand.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `bind_grasp_contact_rules`

[定义位置](../../../manipulation/contact_binding.py#L9)。

输入签名：

```python
def bind_grasp_contact_rules(problem, contacts, activations):
```

实现说明：

> Attach declared rule IDs atomically, preserving every goal and event.
>
> ``activations`` maps each compiled grasp condition ID to its finite rule
> IDs. Neither owner coincidence nor matching nominal time establishes this
> association implicitly. Scene validation subsequently checks the actual
> pose event and close occurrence against the bound robot/owner models.

显式输出分支：

无显式条目。

读取的对象属性：

```python
problem.conditions
row.get
used.intersection
used.update
replacements.get
```

写入目标：

```python
conditions
row
identities
records
rule
replacements
used
condition
row['contact_rule_ids']
selected
replacements[row['condition_id']]
problem.conditions
```

直接调用符号：

```python
row.get
isinstance
len
set
ValueError
asdict
deepcopy
grasp_contact_rule_ids
used.intersection
used.update
tuple
replacements.get
```

## `build_material_contact_rules`

[定义位置](../../../manipulation/contact_binding.py#L42)。

输入签名：

```python
def build_material_contact_rules(material_binding, declarations):
```

实现说明：

> Combine source task features with explicit finite target permissions.
>
> Robot features name their observed body. Other features use their actor's
> local coordinates. The material record supplies feature geometry; each
> declaration supplies the target, active interval and task allowances.

显式输出分支：

```python
tuple(result)
```

读取的对象属性：

```python
set(row).intersection
result.append
```

写入目标：

```python
feature_fields
result
declaration
row
actor
materials
material
name
row[name]
row['robot_body']
row['source']
```

直接调用符号：

```python
deepcopy
set(row).intersection
set
ValueError
result.append
ContactRule
tuple
```
