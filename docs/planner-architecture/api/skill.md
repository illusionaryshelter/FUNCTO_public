# `manipulation/skill.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_semantics`

[定义位置](../../../manipulation/skill.py#L32)。

输入签名：

```python
def _semantics(value):
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
deepcopy
result[field].strip
reference['text'].strip
any
```

## `_read_json`

[定义位置](../../../manipulation/skill.py#L52)。

输入签名：

```python
def _read_json(path):
```

输出分支的源码表达式：

```python
value
```

直接调用：

```python
json.loads
path.read_text
isinstance
ValueError
```

## `_json`

[定义位置](../../../manipulation/skill.py#L59)。

输入签名：

```python
def _json(path, value):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
path.write_text
json.dumps
```

## `_json.convert`

[定义位置](../../../manipulation/skill.py#L60)。

输入签名：

```python
def _json.convert(item):
```

输出分支的源码表达式：

```python
item.tolist()
item.item()
```

直接调用：

```python
isinstance
item.tolist
item.item
TypeError
type
```

## `_metadata`

[定义位置](../../../manipulation/skill.py#L70)。

输入签名：

```python
def _metadata(value):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result['semantic']`。

直接调用：

```python
isinstance
value.keys
value['skill_id'].strip
ValueError
deepcopy
_semantics
```

## `_source`

[定义位置](../../../manipulation/skill.py#L81)。

输入签名：

```python
def _source(value):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result['image']`、`result['depth_m']`、`result['T_world_body_at_image']`、`result['L_body_region']`、`result['crop_xyxy']`、`bank['object_mask']`、`bank['K']`、`bank['T_world_camera_cv']`、`result['region_image_mask']`、`result['observation']`、`result['missing']`、`result['feature']`。

直接调用：

```python
isinstance
value.keys
ValueError
dict
np.asarray
Image.fromarray
_image
value['owner_id'].strip
np.isscalar
np.isfinite
_pose_array
set
observation.items
len
np.isfinite(xyz).all
np.any
_mask
np.array_equal
_intrinsics
bank['region_mask'].any
deepcopy
value.get
min
np.isfinite(feature).all
result['missing'].pop
result['missing'].setdefault
```

## `_representation`

[定义位置](../../../manipulation/skill.py#L161)。

输入签名：

```python
def _representation(value):
```

直接读取的对象属性：`result.update`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[name]`、`result['progress']`。

直接调用：

```python
isinstance
value.get
ValueError
dict
validate_timeline
result.update
_poses_array
len
_pose_array
np.asarray
np.zeros
np.array_equal
```

## `_split_arrays`

[定义位置](../../../manipulation/skill.py#L185)。

输入签名：

```python
def _split_arrays(record):
```

直接读取的对象属性：`record.items`。

输出分支的源码表达式：

```python
({key: value for (key, value) in record.items() if key not in arrays}, arrays)
```

直接调用：

```python
record.items
isinstance
any
arrays.values
ValueError
```

## `_source_motion_agreement`

[定义位置](../../../manipulation/skill.py#L192)。

输入签名：

```python
def _source_motion_agreement(metadata, source, representation):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
ValueError
np.flatnonzero
len
np.isclose
np.allclose
```

## `save_grounding_skill`

[定义位置](../../../manipulation/skill.py#L212)。

输入签名：

```python
def save_grounding_skill(directory, *, metadata, source, representation):
```

输出分支的源码表达式：

```python
manifest
```

直接调用：

```python
_metadata
_source
_representation
_source_motion_agreement
Path
root.mkdir
src['image'].save
_split_arrays
src.items
source_arrays.update
src['observation'].items
_json
np.savez_compressed
dict
```

## `_read_arrays`

[定义位置](../../../manipulation/skill.py#L244)。

输入签名：

```python
def _read_arrays(path):
```

输出分支的源码表达式：

```python
{key: archive[key] for key in archive.files}
```

直接调用：

```python
np.load
```

## `load_grounding_skill`

[定义位置](../../../manipulation/skill.py#L249)。

输入签名：

```python
def load_grounding_skill(directory):
```

输出分支的源码表达式：

```python
{'manifest': manifest, 'metadata': meta, 'source': source, 'representation': representation}
```

直接写入的状态或数组位置：`paths[name]`。

直接调用：

```python
Path(directory).resolve
Path
_read_json
manifest.get
ValueError
isinstance
set
files.items
Path(relative).is_absolute
(root / relative).resolve
path.is_relative_to
path.is_file
_metadata
_read_arrays
key.removeprefix
source_arrays.items
key.startswith
source_meta.keys
observed.keys
motion_meta.keys
motion_arrays.keys
Image.open
_source
opened.convert
_representation
_source_motion_agreement
```

## `observe_interaction_reference`

[定义位置](../../../manipulation/skill.py#L289)。

输入签名：

```python
def observe_interaction_reference(skill, instruction, views, *, detector=None, client=None):
```

输出分支的源码表达式：

```python
{'interaction_reference': deepcopy(reference), 'owner_role': metadata['owner_role'], 'sam_queries': labels, 'selection_scope': scope, 'detections': detections, 'selection': selection}
```

直接写入的状态或数组位置：`source['mask']`、`images[name]`、`detections[name]`、`selection_views[name]`、`selection_views[name]['selected_owner']`。

直接调用：

```python
isinstance
skill.keys
ValueError
_metadata
observed.get
instruction.strip
views.items
name.strip
_image
images.items
detector
labels.copy
set
enumerate
row.get
type
used.add
rows.append
_detection
select_grounding_candidates
deepcopy
```
