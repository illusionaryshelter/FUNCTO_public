# `manipulation/grounding_records.py` 实现契约

代码基线：`96ee2dc-working-tree`。
职责和数据语义见[组件说明](../15-skill-grounding.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_save_reference`

[定义位置](../../../manipulation/grounding_records.py#L17)。

输入签名：

```python
def _save_reference(directory, hypothesis, record):
```

实现说明：

> Save the same nominal expression and bound source-clock samples.

显式输出分支：

```python
None
None
```

读取的对象属性：

```python
hypothesis.get
np.savez_compressed
reference.items
np.ndarray
np.asarray
bound.sample_numpy
bound.initial_pose
bound.initial_twist
bound.to_record
np.save
np.linspace
```

写入目标：

```python
reference
key
value
name
bound
source_clock
elapsed
record['reference_samples']
```

直接调用符号：

```python
hypothesis.get
np.savez_compressed
reference.items
isinstance
write_json
plain
dict
np.asarray
bound.sample_numpy
float
bound.to_record
np.save
np.linspace
```

## `save`

[定义位置](../../../manipulation/grounding_records.py#L46)。

输入签名：

```python
def save(result, request, output_directory):
```

实现说明：

> Save common evidence, every hypothesis, and an immediately openable HTML.
>
> The manifest references ``hypotheses/<id>`` for both one and many results.
> No hypothesis is selected by this writer. The planning portfolio later
> archives and chooses its own completed curves. An absent physical trace
> remains explicitly missing in every reference view.

显式输出分支：

```python
record
```

读取的对象属性：

```python
directory.mkdir
result.items
selection.items
view.items
selection['views'].items
np.savez_compressed
(directory / 'error.txt').write_text
result.get
np.savez
np.ndarray
region.get
folder.mkdir
hypothesis.items
hypothesis.get
row.update
diagnostics.get
links.append
folder.relative_to
single.get
```

写入目标：

```python
directory
excluded
record
key
value
selection
record['selection']
identity
view
masks
item
record['selection']['mask_evidence']
binding_id
views
region
numerical
record['numerical_evidence']
report_fields
links
hypothesis
folder
row
diagnostics
reference
name
record['reference_hypotheses']
record['reference_links']
single
```

直接调用符号：

```python
Path
directory.mkdir
plain
result.items
selection.items
view.items
selection['views'].items
np.savez_compressed
(directory / 'error.txt').write_text
result.get
np.savez
isinstance
write_json
region.get
save_surface
store_prediction
folder.mkdir
hypothesis.items
hypothesis.get
row.update
diagnostics.get
_save_reference
write_reference_html
links.append
dict
str
folder.relative_to
len
single.get
```
