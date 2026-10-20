# `manipulation/grounding.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_image`

[定义位置](../../../manipulation/grounding.py#L25)。

输入签名：

```python
def _image(image):
```

输出分支的源码表达式：

```python
image.convert('RGB')
```

直接调用：

```python
isinstance
min
ValueError
image.convert
```

## `_task`

[定义位置](../../../manipulation/grounding.py#L31)。

输入签名：

```python
def _task(task):
```

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[field]`、`result[role]`、`result['goal_description']`。

直接调用：

```python
set
isinstance
task.keys
ValueError
_CHOICES.items
any
label.strip
labels.values
value.strip
labels.items
goal.strip
```

## `_client`

[定义位置](../../../manipulation/grounding.py#L58)。

输入签名：

```python
def _client(client):
```

输出分支的源码表达式：

```python
Gemini()
client
```

直接调用：

```python
Gemini
```

## `infer_task`

[定义位置](../../../manipulation/grounding.py#L66)。

输入签名：

```python
def infer_task(instruction, demo_image, scene_image, *, demo_final_image=None, client=None):
```

输出分支的源码表达式：

```python
_task(_client(client).chat_json(prompt, images=images))
```

直接调用：

```python
isinstance
instruction.strip
ValueError
_image
images.append
json.dumps
sorted
_CHOICES.items
_task
_client(client).chat_json
_client
```

## `_mask`

[定义位置](../../../manipulation/grounding.py#L139)。

输入签名：

```python
def _mask(mask, image, name):
```

输出分支的源码表达式：

```python
mask.astype(bool, copy=False)
```

直接调用：

```python
np.asarray
ValueError
np.isfinite(mask).all
np.isfinite
np.logical_or(mask == 0, mask == 1).all
np.logical_or
mask.any
mask.astype
```

## `_detection`

[定义位置](../../../manipulation/grounding.py#L150)。

输入签名：

```python
def _detection(value, image, labels, number):
```

输出分支的源码表达式：

```python
{**value, 'score': score, 'bbox': bbox.tolist(), 'mask': mask, 'candidate_id': number}
```

直接调用：

```python
isinstance
value.keys
ValueError
np.isscalar
float
np.asarray
np.isfinite
np.isfinite(bbox).all
_mask
np.nonzero
xx.max
xx.min
yy.max
yy.min
bbox.tolist
```

## `_candidate_sheet`

[定义位置](../../../manipulation/grounding.py#L178)。

输入签名：

```python
def _candidate_sheet(image, detections):
```

输出分支的源码表达式：

```python
sheet
```

直接调用：

```python
min
len
Image.new
enumerate
Image.fromarray
item['mask'].astype
Image.composite
ImageDraw.Draw
draw.rectangle
panel.thumbnail
sheet.paste
ImageDraw.Draw(sheet).text
str
```

## `select_roles`

[定义位置](../../../manipulation/grounding.py#L201)。

输入签名：

```python
def select_roles(image, task, domain='demo', *, detector=None, client=None):
```

输出分支的源码表达式：

```python
{'carrier': carrier, 'reference': reference}
```

直接调用：

```python
_image
_task
ValueError
list
dict.fromkeys
requested.values
detector
isinstance
_detection
enumerate
requested.items
candidates.items
all
len
candidates.values
json.dumps
_client(client).chat_json
_client
_candidate_sheet
set
selected.items
type
np.array_equal
```

## `_selected_interaction_evidence`

[定义位置](../../../manipulation/grounding.py#L261)。

输入签名：

```python
def _selected_interaction_evidence(record, image, owner_mask, *, owner_id=None):
```

输出分支的源码表达式：

```python
None
{'mask': mask, 'owner_id': record['owner_id'], 'description': record['description'].strip(), 'source': record['source'].strip()}
```

直接调用：

```python
isinstance
set
ValueError
record[name].strip
np.asarray
_mask(mask, image, 'interaction_evidence.mask').copy
_mask
np.any
record['description'].strip
record['source'].strip
```

## `propose_query_regions`

[定义位置](../../../manipulation/grounding.py#L290)。

输入签名：

```python
def propose_query_regions(source_image, current_image, source_owner_mask, current_owner_mask, source_region_mask, query_pixels, *, description, query_descriptions, client=None, current_interaction_evidence=None):
```

输出分支的源码表达式：

```python
{'boxes': np.asarray(boxes, dtype=np.int64), 'diagnostics': diagnostics}
```

直接写入的状态或数组位置：`cells[number]`、`details['current_interaction_evidence']`、`diagnostics['current_interaction_evidence']`。

直接调用：

```python
_image
_mask
_selected_interaction_evidence
np.any
ValueError
int
np.count_nonzero
np.asarray
len
pixels.astype
np.isfinite(pixels).all
np.isfinite
np.rint(pixels).astype
np.rint
source_owner[nearest[:, 1], nearest[:, 0]].all
isinstance
description.strip
any
item.strip
owner_panel
Image.composite
Image.new
Image.fromarray
source_region.astype
display_panel
ImageDraw.Draw
enumerate
np.array
source_draw.ellipse
source_draw.text
max
min
str
interaction['mask'].astype
np.linspace
range
map
np.nonzero
current_draw.rectangle
np.argmin
current_draw.text
point.tolist
query_descriptions[index].strip
cells.items
mask.sum
json.dumps
_client(client).chat_json
_client
set
type
boxes.append
selected.copy
cells[number].copy
list
source_scale.tolist
current_scale.tolist
```

## `propose_query_regions.owner_panel`

[定义位置](../../../manipulation/grounding.py#L348)。

输入签名：

```python
def propose_query_regions.owner_panel(image, owner):
```

输出分支的源码表达式：

```python
Image.fromarray(array)
```

直接写入的状态或数组位置：`array[~owner]`、`array[owner & ~interior]`。

直接调用：

```python
np.array
np.rint(array[~owner].astype(float) * 0.45).astype
np.rint
array[~owner].astype
np.pad
Image.fromarray
```

## `propose_query_regions.display_panel`

[定义位置](../../../manipulation/grounding.py#L366)。

输入签名：

```python
def propose_query_regions.display_panel(panel):
```

输出分支的源码表达式：

```python
(panel.resize(size, Image.Resampling.BICUBIC), scales)
```

直接调用：

```python
max
tuple
int
round
np.asarray
panel.resize
```

## `match_points`

[定义位置](../../../manipulation/grounding.py#L526)。

输入签名：

```python
def match_points(image1, image2, mask1, mask2, pixels1, *, extractor=None):
```

输出分支的源码表达式：

```python
(np.empty((0, 2), dtype=float), {'point_count': 0, 'extractor_called': False})
match_features(extractor(image1, image2), pixels, image1.size, mask2)
```

直接调用：

```python
_image
_mask
np.asarray
ValueError
pixels.astype
np.isfinite(pixels).all
np.isfinite
np.any
np.rint(pixels).astype
np.rint
mask1[nearest[:, 1], nearest[:, 0]].all
len
np.empty
match_features
extractor
```
