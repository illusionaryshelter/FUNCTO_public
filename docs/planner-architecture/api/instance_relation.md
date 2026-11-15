# `manipulation/instance_relation.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../15-skill-grounding.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `select_bindings`

[定义位置](../../../manipulation/instance_relation.py#L46)。

输入签名：

```python
def select_bindings(bindings, instruction, *, client=None):
```

实现说明：

> Return joint semantic evidence for explicit source/current bindings.
>
> Each mapping value provides ``source``, ``views`` with real candidates,
> ``selection_scope``, ``source_owner_id``, ``current_entity_id`` and
> ``owner_role``. Images from every binding enter this one model call. The
> same parser resolves the actual masks used by local semantic selection.
> Empty candidate sets produce explicit missing evidence without inference.

显式输出分支：

```python
{'bindings': results, 'vlm_used': called, 'raw_response': response}
```

读取的对象属性：

```python
bindings.items
binding_id.strip
context.update
images.extend
rules.append
views.values
prepared.values
'\n'.join
json.dumps
_client(client).chat_json
prepared.items
reason.strip
```

写入目标：

```python
images
contexts
prepared
rules
binding_id
binding
rule
pictures
candidates
context
offset
context['image_order']
item
key
contexts[binding_id]
prepared[binding_id]
called
views
prompt
response
evidence
response['bindings'][binding_id]
name
results
fields
row
relation
reason
selected
results[binding_id]
```

直接调用符号：

```python
ValueError
bindings.items
isinstance
binding_id.strip
prepare_selection_evidence
len
context.update
images.extend
rules.append
any
views.values
prepared.values
'\n'.join
json.dumps
_client(client).chat_json
_client
prepared.items
set
reason.strip
parse_selection_response
```

## `region_strategy`

[定义位置](../../../manipulation/instance_relation.py#L111)。

输入签名：

```python
def region_strategy(relation):
```

实现说明：

> Dispatch an established semantic relation without changing its evidence.

显式输出分支：

```python
{'same_instance': 'same_instance', 'same_category_different_instance': 'cross_instance', 'different_category': None, 'unresolved': None}[relation]
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
ValueError
```
