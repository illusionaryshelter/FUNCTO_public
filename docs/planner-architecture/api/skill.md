# `manipulation/skill.py` 实现契约

代码基线：`82a3d66`。本次仅文档提交引用该实现提交，组件代码与该基线一致。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_semantics`

[定义位置](../../../manipulation/skill.py#L32)。

输入签名：

```python
def _semantics(value):
```

实现说明：

> Read the actual semantic record; contact-free parts may be unnamed.

显式输出分支：

```python
result
```

读取的对象属性：

```python
result[field].strip
reference['text'].strip
```

写入目标：

```python
result
field
reference
character
```

直接调用符号：

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

显式输出分支：

```python
value
```

读取的对象属性：

```python
json.loads
path.read_text
path.name
```

写入目标：

```python
value
```

直接调用符号：

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

显式输出分支：

无显式条目。

读取的对象属性：

```python
path.write_text
json.dumps
```

写入目标：

无显式条目。

直接调用符号：

```python
path.write_text
json.dumps
```

## `_json.convert`

[定义位置](../../../manipulation/skill.py#L60)。

输入签名：

```python
def convert(item):
```

显式输出分支：

```python
item.tolist()
item.item()
```

读取的对象属性：

```python
np.ndarray
item.tolist
np.generic
item.item
type(item).__name__
```

写入目标：

无显式条目。

直接调用符号：

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

显式输出分支：

```python
result
```

读取的对象属性：

```python
value.keys
value['skill_id'].strip
```

写入目标：

```python
result
result['semantic']
```

直接调用符号：

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

实现说明：

> Check source identities and calibrated array dimensions, not task success.

显式输出分支：

```python
result
```

读取的对象属性：

```python
value.keys
Image.Image
np.asarray
array.dtype
np.uint8
array.ndim
array.shape
Image.fromarray
image.height
image.width
depth.shape
depth.dtype.kind
depth.dtype
value['owner_id'].strip
np.bool_
np.integer
np.isscalar
np.isfinite
crop.shape
crop.dtype.kind
crop.dtype
observation.items
xyz.ndim
xyz.shape
np.isfinite(xyz).all
pixels.shape
pixels.dtype.kind
pixels.dtype
np.any
bank[key].dtype
bank[key].shape
np.array_equal
bank['region_mask'].any
value.get
feature.ndim
feature.shape
feature.dtype
np.float32
np.isfinite(feature).all
result['missing'].pop
result['missing'].setdefault
```

写入目标：

```python
required
result
image
array
result['image']
shape
depth
result['depth_m']
result['T_world_body_at_image']
result['L_body_region']
crop
result['crop_xyxy']
observation
names
bank
key
xyz
pixels
bank['object_mask']
bank['K']
bank['T_world_camera_cv']
mask
result['region_image_mask']
result['observation']
result['missing']
feature
result['feature']
```

直接调用符号：

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

显式输出分支：

```python
result
```

读取的对象属性：

```python
value.get
result.update
np.asarray
np.zeros
np.array_equal
```

写入目标：

```python
result
frames
times
name
poses
result[name]
progress
expected
result['progress']
```

直接调用符号：

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

显式输出分支：

```python
({key: value for (key, value) in record.items() if key not in arrays}, arrays)
```

读取的对象属性：

```python
record.items
np.ndarray
array.dtype.hasobject
array.dtype
arrays.values
```

写入目标：

```python
arrays
key
value
array
```

直接调用符号：

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

实现说明：

> The image's frame and the frame used to form H must be the same one.

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.flatnonzero
np.isclose
np.allclose
```

写入目标：

```python
expected
row
role
local
track
```

直接调用符号：

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

实现说明：

> Write one new ``functo.grounding_skill.2`` bundle, manifest last.
>
> ``metadata`` includes skill_id, owner_role, semantic and roles, and may carry
> independent task conditions, provenance and missing inputs. ``source`` is
> one actual image plus its native observation point bank, calibrated depth,
> crop, timestamp_s/frame_index, owner_id, T_world_body_at_image and explicit
> region_geometry (planar or surface). Optional
> L_body_region and region_image_mask describe the supplied source region;
> optional feature/feature_model record its spatial MARCO map. No source
> directory is opened. ``representation`` is build_representation's result.
> Arrays live in NPZ, source RGB in PNG; other records are strict JSON.

显式输出分支：

```python
manifest
```

读取的对象属性：

```python
root.mkdir
src['image'].save
src.items
source_arrays.update
src['observation'].items
np.savez_compressed
```

写入目标：

```python
meta
src
motion
root
source_meta
source_arrays
key
value
motion_meta
motion_arrays
manifest
```

直接调用符号：

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

显式输出分支：

```python
{key: archive[key] for key in archive.files}
```

读取的对象属性：

```python
np.load
archive.files
```

写入目标：

```python
archive
key
```

直接调用符号：

```python
np.load
```

## `load_grounding_skill`

[定义位置](../../../manipulation/skill.py#L249)。

输入签名：

```python
def load_grounding_skill(directory):
```

实现说明：

> Read only this version-2 package's files; provenance paths stay opaque.
>
> Return ``manifest, metadata, source, representation``. The source's nested
> ``observation`` is directly usable by the dual-view adapter. Loading never
> invokes models, re-extracts regions or fabricates absent motion components.

显式输出分支：

```python
{'manifest': manifest, 'metadata': meta, 'source': source, 'representation': representation}
```

读取的对象属性：

```python
Path(directory).resolve
manifest.get
files.items
Path(relative).is_absolute
(root / relative).resolve
path.is_relative_to
path.is_file
key.removeprefix
source_arrays.items
key.startswith
source_meta.keys
observed.keys
motion_meta.keys
motion_arrays.keys
Image.open
opened.convert
```

写入目标：

```python
root
manifest
files
paths
name
relative
path
paths[name]
meta
source_meta
source_arrays
motion_meta
motion_arrays
bank
key
value
observed
opened
source
representation
```

直接调用符号：

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

实现说明：

> Detect and semantically select the reference chosen during extraction.
>
> Each view supplies an ``image`` and may supply separately observed
> ``selected_owner`` context for a part. The detector receives precisely the
> stored noun phrase. Object references select semantic entities, potentially
> containing several physical bodies; part references
> select functional components, which are neither whole-owner masks nor exact
> demonstrated contact patches. No alternate query or score-based fallback is
> introduced when a view is occluded or ambiguous. Independent current owner
> evidence, if a downstream consumer needs it, remains a separate observation.

显式输出分支：

```python
{'interaction_reference': deepcopy(reference), 'owner_role': metadata['owner_role'], 'sam_queries': labels, 'selection_scope': scope, 'detections': detections, 'selection': selection}
```

读取的对象属性：

```python
skill.keys
observed.get
instruction.strip
views.items
name.strip
images.items
labels.copy
row.get
used.add
rows.append
```

写入目标：

```python
metadata
semantic
observed
reference
region
scope
source
source['mask']
images
name
view
images[name]
detector
detections
selection_views
labels
image
raw
rows
used
index
row
number
detections[name]
selection_views[name]
selection_views[name]['selected_owner']
selection
```

直接调用符号：

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
