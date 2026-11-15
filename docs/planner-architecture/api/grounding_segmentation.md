# `transfer/segmentation.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../15-skill-grounding.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_prompt`

[定义位置](../../../transfer/segmentation.py#L25)。

输入签名：

```python
def _prompt(value):
```

显式输出分支：

```python
value
```

读取的对象属性：

```python
value.strip
```

写入目标：

无显式条目。

直接调用符号：

```python
isinstance
value.strip
ValueError
```

## `_rgb`

[定义位置](../../../transfer/segmentation.py#L31)。

输入签名：

```python
def _rgb(value):
```

显式输出分支：

```python
image
```

读取的对象属性：

```python
np.asarray
image.dtype
np.uint8
image.ndim
image.shape
```

写入目标：

```python
image
```

直接调用符号：

```python
np.asarray
ValueError
min
```

## `_detections`

[定义位置](../../../transfer/segmentation.py#L40)。

输入签名：

```python
def _detections(value, shape, prompt):
```

实现说明：

> Validate returned instances without thresholding or editing their masks.

显式输出分支：

```python
result
```

读取的对象属性：

```python
np.asarray
mask.dtype
np.bool_
mask.shape
np.integer
np.floating
np.isfinite
box.shape
box.dtype.kind
box.dtype
np.isfinite(box).all
result.append
box.astype(float).tolist
box.astype
mask.copy
```

写入目标：

```python
result
index
detection
mask
score
box
```

直接调用符号：

```python
isinstance
TypeError
enumerate
ValueError
np.asarray
np.isfinite
np.isfinite(box).all
result.append
float
box.astype(float).tolist
box.astype
mask.copy
```

## `store_detections`

[定义位置](../../../transfer/segmentation.py#L68)。

输入签名：

```python
def store_detections(output_dir, rgb, prompt, detections):
```

实现说明：

> Save every returned candidate and the exact RGB before selecting one.
>
> Existing evidence is never overwritten. Empty model output is a recorded
> outcome, whereas malformed output raises an exception. Candidate indices
> refer to the ordered public SAM result, including its existing empty-mask
> boundary handling; they are not inferred simulator instance identifiers.

显式输出分支：

```python
metadata
```

读取的对象属性：

```python
image.shape
output_dir.mkdir
Image.fromarray(image).save
Image.fromarray
records.append
records[-1].update
instance['mask'].sum
np.savez_compressed
```

写入目标：

```python
image
prompt
instances
output_dir
arrays
records
instance
key
arrays[key]
name
metadata
```

直接调用符号：

```python
_rgb
_prompt
_detections
Path
output_dir.mkdir
Image.fromarray(image).save
Image.fromarray
records.append
records[-1].update
int
instance['mask'].sum
np.savez_compressed
list
len
write_json
```

## `load_detections`

[定义位置](../../../transfer/segmentation.py#L96)。

输入签名：

```python
def load_detections(output_dir):
```

实现说明：

> Restore stored candidates in their original order without pickle.

显式输出分支：

```python
[{key: value for (key, value) in item.items() if key != 'index'} for item in checked]
```

读取的对象属性：

```python
json.loads
(output_dir / 'candidates.json').read_text
Image.open
image.height
image.width
np.load
arrays.files
mask.sum
detections.append
item.items
```

写入目标：

```python
output_dir
metadata
prompt
image
shape
records
arrays
i
detections
index
record
mask
checked
key
value
item
```

直接调用符号：

```python
Path
json.loads
(output_dir / 'candidates.json').read_text
_prompt
Image.open
ValueError
len
np.load
set
range
enumerate
int
mask.sum
detections.append
_detections
item.items
```

## `detect_image`

[定义位置](../../../transfer/segmentation.py#L124)。

输入签名：

```python
def detect_image(image_path, prompt, output_dir):
```

实现说明：

> Run one full RGB image and one literal prompt in the existing worker.
>
> The caller serializes complete stages and applies the explicit external
> resource supervisor. No boxes, prior masks, synonym attempts, or alternate
> detections are supplied to SAM. Worker errors propagate to the supervisor.

显式输出分支：

```python
store_detections(output_dir, np.asarray(image), prompt, detections)
```

读取的对象属性：

```python
output_dir.exists
Image.open
opened.convert
np.asarray
```

写入目标：

```python
prompt
output_dir
opened
image
detections
```

直接调用符号：

```python
_prompt
Path
output_dir.exists
FileExistsError
Image.open
opened.convert
detect
store_detections
np.asarray
```

## `_calibration`

[定义位置](../../../transfer/segmentation.py#L142)。

输入签名：

```python
def _calibration(K, T_world_camera_cv):
```

显式输出分支：

```python
(K, T)
```

读取的对象属性：

```python
np.asarray
K.shape
T.shape
K.dtype.kind
K.dtype
T.dtype.kind
T.dtype
np.isfinite(K).all
np.isfinite
np.isfinite(T).all
K.astype
T.astype
np.allclose
np.linalg.det
np.linalg
np.finfo(float).tiny
np.finfo
T[:3, :3].T
np.eye
np.isclose
```

写入目标：

```python
K
T
```

直接调用符号：

```python
np.asarray
np.isfinite(K).all
np.isfinite
np.isfinite(T).all
ValueError
K.astype
T.astype
np.allclose
abs
np.linalg.det
np.finfo
np.eye
np.isclose
```

## `_rms`

[定义位置](../../../transfer/segmentation.py#L159)。

输入签名：

```python
def _rms(points):
```

显式输出分支：

```python
None
float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))
```

读取的对象属性：

```python
np.sqrt
np.mean
np.sum
points.mean
```

写入目标：

无显式条目。

直接调用符号：

```python
len
float
np.sqrt
np.mean
np.sum
points.mean
```

## `_crop`

[定义位置](../../../transfer/segmentation.py#L165)。

输入签名：

```python
def _crop(mask, margin_fraction):
```

显式输出分支：

```python
np.asarray([max(0, int(x.min()) - margin), max(0, int(y.min()) - margin), min(width, int(x.max()) + margin + 1), min(height, int(y.max()) + margin + 1)], dtype=np.int64)
```

读取的对象属性：

```python
np.nonzero
np.ceil
np.ptp
mask.shape
np.asarray
x.min
y.min
x.max
y.max
np.int64
```

写入目标：

```python
y
x
margin
height
width
```

直接调用符号：

```python
np.nonzero
int
np.ceil
max
np.ptp
np.asarray
x.min
y.min
min
x.max
y.max
```

## `_selected_geometry`

[定义位置](../../../transfer/segmentation.py#L175)。

输入签名：

```python
def _selected_geometry(image, depth, K, T, selected, evidence, *, source, crop_margin_fraction, result):
```

实现说明：

> Backproject an already chosen mask, never reconsider its identity.
>
> Source evidence remains a separate observation even where SAM missed it.
> The selected mask itself is preserved, including pixels without depth;
> only the measured point bank needs positive finite optical depth.

显式输出分支：

```python
result
result
```

读取的对象属性：

```python
np.isfinite
entity.any
result['reasons'].append
np.nonzero
np.column_stack((x, y)).astype
np.column_stack
np.int32
np.ones
np.linalg.inv(K).T
np.linalg.inv
np.linalg
T[:3, :3].T
raw_mask.copy
K.copy
T.copy
(region & entity_member).sum
valid_evidence.sum
result['diagnostics'].update
raw_mask.sum
entity.sum
entity_member.sum
(region & ~entity_member).sum
points[region].mean(0).tolist
points[region].mean
result.update
image[top:bottom, left:right].copy
```

写入目标：

```python
valid_depth
valid_evidence
raw_mask
entity
bank_mask
y
x
pixels
rays
points
entity_member
region
observation
crop_mask
bounds
left
top
right
bottom
sam_rms
bank_rms
covered
evidence_count
```

直接调用符号：

```python
np.isfinite
entity.any
result['reasons'].append
np.nonzero
np.column_stack((x, y)).astype
np.column_stack
np.ones
len
np.linalg.inv
raw_mask.copy
K.copy
T.copy
_crop
float
_rms
int
(region & entity_member).sum
valid_evidence.sum
result['diagnostics'].update
raw_mask.sum
entity.sum
entity_member.sum
(region & ~entity_member).sum
points[region].mean(0).tolist
points[region].mean
result.update
image[top:bottom, left:right].copy
```

## `prepare_selected_observation`

[定义位置](../../../transfer/segmentation.py#L222)。

输入签名：

```python
def prepare_selected_observation(rgb, depth_m, K, T_world_camera_cv, selected, *, source_region_mask=None, crop_margin_fraction=0.15):
```

实现说明：

> Prepare the entity already selected by the current instruction.
>
> ``selected`` is one actual SAM detection, with its unchanged label, score,
> box and mask. This entry accepts no list to rank. A depth-empty selected
> entity reports a missing observation; it never tries another instance.
> A given source region may extend outside the selected SAM mask because
> its identity was supplied separately from this incomplete visual mask.

显式输出分支：

```python
result
_selected_geometry(image, depth, K, T, checked, evidence, source=source, crop_margin_fraction=crop_margin_fraction, result=result)
```

读取的对象属性：

```python
np.asarray
depth.shape
image.shape
depth.dtype.kind
depth.dtype
selected.get
np.bool_
np.integer
np.floating
np.isfinite
np.zeros
evidence.dtype
evidence.shape
evidence.copy
evidence.sum
valid_evidence.sum
(evidence & ~valid_evidence).sum
valid_evidence.any
result['reasons'].append
```

写入目标：

```python
image
depth
prompt
checked
K
T
source
evidence
valid_evidence
result
```

直接调用符号：

```python
_rgb
np.asarray
ValueError
isinstance
TypeError
_prompt
selected.get
_detections
_calibration
np.isfinite
np.zeros
evidence.copy
float
int
evidence.sum
valid_evidence.sum
(evidence & ~valid_evidence).sum
valid_evidence.any
result['reasons'].append
_selected_geometry
```

## `prepare_mask_observation`

[定义位置](../../../transfer/segmentation.py#L272)。

输入签名：

```python
def prepare_mask_observation(rgb, depth_m, K, T_world_camera_cv, mask, *, source, crop_margin_fraction=0.15):
```

实现说明：

> Backproject an existing observed mask with its own explicit provenance.
>
> This accepts current owner evidence already supplied by another semantic
> binding. It uses the same pixel order, calibration and crop computation
> as selected SAM observations, while retaining its separate mask and source.
> No detector score, candidate ID or complete hidden shape is manufactured.

显式输出分支：

```python
_selected_geometry(image, depth, K, T, {'mask': mask}, np.zeros_like(mask), source=False, crop_margin_fraction=crop_margin_fraction, result=result)
```

读取的对象属性：

```python
np.asarray
depth.shape
image.shape
depth.dtype.kind
depth.dtype
mask.dtype
np.bool_
mask.shape
source.strip
np.zeros_like
```

写入目标：

```python
image
depth
mask
K
T
result
```

直接调用符号：

```python
_rgb
np.asarray
ValueError
isinstance
source.strip
_calibration
dict
float
_selected_geometry
np.zeros_like
```

## `prepare_observation`

[定义位置](../../../transfer/segmentation.py#L299)。

输入签名：

```python
def prepare_observation(rgb, depth_m, K, T_world_camera_cv, detections, *, prompt, source_region_mask=None, crop_margin_fraction=0.15):
```

实现说明：

> Bind one predicted instance and backproject the permitted RGB-D evidence.
>
> Source selection maximizes valid-depth evidence coverage, then native
> score, then the earliest candidate index. At least one given evidence pixel
> must be covered. Target selection uses score and then index only, before
> examining the selected instance's depth. A selected depth-empty instance
> fails explicitly; a lower-ranked candidate never silently replaces it.
>
> Source points are SAM-valid-depth UNION given-evidence-valid-depth. Its
> original SAM image mask remains unchanged, and ``entity_member``,
> ``region_mask`` and ``evidence_only`` distinguish the point roles. Global
> matching must sample entity_member; local queries must sample region_mask.
> Target points contain only SAM-valid-depth pixels. No existing entity-only
> point cloud is accepted because it could have already removed SAM errors
> using ground truth. The source crop includes the original SAM and supplied
> evidence masks; the target crop uses the selected SAM-valid-depth mask.

显式输出分支：

```python
result
result
result
_selected_geometry(image, depth, K, T, selected, evidence, source=source, crop_margin_fraction=crop_margin_fraction, result=result)
```

读取的对象属性：

```python
image.shape
np.asarray
depth.shape
depth.dtype.kind
depth.dtype
np.bool_
np.integer
np.floating
np.isfinite
np.zeros
evidence.dtype
evidence.shape
evidence.copy
valid_evidence.sum
np.count_nonzero
candidate_records.append
item['mask'].sum
evidence.sum
(evidence & ~valid_depth).sum
result['reasons'].append
```

写入目标：

```python
image
prompt
shape
depth
K
T
instances
valid_depth
source
evidence
valid_evidence
evidence_count
candidate_records
item
covered
selection
diagnostics
result
selected_index
selection['selected_index']
selected
```

直接调用符号：

```python
_rgb
_prompt
np.asarray
ValueError
_calibration
isinstance
np.isfinite
_detections
np.zeros
evidence.copy
int
valid_evidence.sum
np.count_nonzero
candidate_records.append
item['mask'].sum
len
float
evidence.sum
(evidence & ~valid_depth).sum
result['reasons'].append
min
range
_selected_geometry
```

## `main`

[定义位置](../../../transfer/segmentation.py#L377)。

输入签名：

```python
def main():
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
args.image
args.prompt
args.output
json.dumps
```

写入目标：

```python
parser
args
metadata
```

直接调用符号：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
detect_image
print
json.dumps
str
```
