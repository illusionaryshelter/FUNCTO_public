# `manipulation/grounding.py` 实现契约

代码基线：`9e8b365`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_image`

[定义位置](../../../manipulation/grounding.py#L25)。

输入签名：

```python
def _image(image):
```

显式输出分支：

```python
image.convert('RGB')
```

读取的对象属性：

```python
Image.Image
image.size
image.convert
```

写入目标：

无显式条目。

直接调用符号：

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

实现说明：

> Reject extra fields, particularly ungrounded numeric geometry from a VLM.

显式输出分支：

```python
result
```

读取的对象属性：

```python
task.keys
_CHOICES.items
label.strip
labels.values
value.strip
labels.items
goal.strip
```

写入目标：

```python
required
result
field
values
result[field]
role
labels
label
result[role]
key
value
goal
result['goal_description']
```

直接调用符号：

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

显式输出分支：

```python
Gemini()
client
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
Gemini
```

## `infer_task`

[定义位置](../../../manipulation/grounding.py#L66)。

输入签名：

```python
def infer_task(instruction, demo_image, scene_image, *, demo_final_image=None, client=None):
```

实现说明：

> Infer semantic candidates for one active-carrier stage, never metric geometry.
>
> The optional demonstration final image provides outcome context. This is not
> a multistage planner: stage splitting and subsequent state updates belong to
> the caller. Motion remains a proposal until validated against observations.

显式输出分支：

```python
_task(_client(client).chat_json(prompt, images=images))
```

读取的对象属性：

```python
instruction.strip
images.append
json.dumps
_CHOICES.items
_client(client).chat_json
```

写入目标：

```python
images
image_order
prompt
key
value
```

直接调用符号：

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

显式输出分支：

```python
mask.astype(bool, copy=False)
```

读取的对象属性：

```python
np.asarray
mask.shape
image.height
image.width
mask.dtype.kind
mask.dtype
np.isfinite(mask).all
np.isfinite
np.logical_or(mask == 0, mask == 1).all
np.logical_or
mask.any
mask.astype
```

写入目标：

```python
mask
```

直接调用符号：

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

显式输出分支：

```python
{**value, 'score': score, 'bbox': bbox.tolist(), 'mask': mask, 'candidate_id': number}
```

读取的对象属性：

```python
value.keys
np.bool_
np.isscalar
np.asarray
np.isfinite
bbox.shape
np.isfinite(bbox).all
image.width
image.height
np.nonzero
xx.max
xx.min
yy.max
yy.min
bbox.tolist
```

写入目标：

```python
score
bbox
x1
y1
x2
y2
mask
yy
xx
```

直接调用符号：

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

实现说明：

> Number each mask in its own panel so nested parts cannot hide each other.
>
> Pillow avoids importing the legacy annotation stack (torchvision and its
> visualization dependencies) merely to prepare a semantic selection image.

显式输出分支：

```python
sheet
```

读取的对象属性：

```python
Image.new
Image.fromarray
item['mask'].astype
np.uint8
Image.composite
image.size
ImageDraw.Draw
draw.rectangle
panel.thumbnail
sheet.paste
ImageDraw.Draw(sheet).text
```

写入目标：

```python
panel_width
panel_height
columns
rows
sheet
index
item
alpha
panel
draw
left
top
```

直接调用符号：

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

实现说明：

> Bind carrier/reference to actual SAM instances, selecting only if ambiguous.
>
> Both roles are required for this stage interface. Masks may overlap for a
> moving part and its body, but identical masks or one selected ID are rejected.
> This image-level check is not proof of physical identity in an occluded scene.

显式输出分支：

```python
{'carrier': carrier, 'reference': reference}
```

读取的对象属性：

```python
dict.fromkeys
requested.values
requested.items
candidates.items
candidates.values
json.dumps
_client(client).chat_json
selected.items
np.array_equal
```

写入目标：

```python
image
task
requested
role
labels
detector
values
detections
index
item
candidates
label
numbers
selected
prompt
number
carrier
reference
```

直接调用符号：

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

实现说明：

> Validate an observed part without promoting it to a physical answer.
>
> The semantic selection's provenance belongs to the record. A mask is
> permitted to spill outside its independently selected owner; it is retained
> unchanged, provided the two have some overlap. Queries need not lie in it.

显式输出分支：

```python
None
{'mask': mask, 'owner_id': record['owner_id'], 'description': record['description'].strip(), 'source': record['source'].strip()}
```

读取的对象属性：

```python
record[name].strip
np.asarray
mask.dtype
np.bool_
_mask(mask, image, 'interaction_evidence.mask').copy
np.any
record['description'].strip
record['source'].strip
```

写入目标：

```python
required
name
mask
```

直接调用符号：

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

实现说明：

> Ask once for coarse current-owner cells corresponding to source queries.
>
> Images are owner-context crops; xy queries and returned integer xyxy boxes
> use their respective crop coordinates. Right/bottom box edges are exclusive.
> Source queries may lie outside the interaction region, because functional
> directions can require evidence elsewhere on the same owner. Descriptions
> explain that evidence; they are actually passed to the semantic selector.
> Noisy source region pixels outside the owner remain visible and are counted;
> nonempty overlap is required, while every query must belong to the owner.
>
> The current owner is partitioned by a 5x5 image grid, discarding empty cells.
> A selected cell gets a quarter-cell margin for the downstream dense search.
> These boxes are program-defined search windows, NOT predicted part masks or
> metric geometry. The caller must intersect each window with the current
> owner and valid depth, then assess the resulting correspondences. Null or
> invalid semantic selections fail explicitly rather than guessing a region.
>
> Optional selected current interaction evidence supplies an observed mask,
> owner identity, semantic description and provenance. Its blue overlay and
> cell coverage guide the same VLM request; they neither change the owner nor
> forbid attachment/direction queries outside the visible part. Without this
> evidence, the request and images remain the owner-only form.

显式输出分支：

```python
{'boxes': np.asarray(boxes, dtype=np.int64), 'diagnostics': diagnostics}
```

读取的对象属性：

```python
np.any
np.count_nonzero
np.asarray
pixels.dtype.kind
pixels.dtype
pixels.ndim
pixels.shape
pixels.astype
np.isfinite(pixels).all
np.isfinite
source_image.width
source_image.height
np.rint(pixels).astype
np.rint
source_owner[nearest[:, 1], nearest[:, 0]].all
description.strip
item.strip
Image.composite
Image.new
source_image.size
Image.fromarray
source_region.astype
np.uint8
ImageDraw.Draw
np.array
source_draw.ellipse
source_draw.text
source_panel.width
source_panel.height
current_image.size
interaction['mask'].astype
np.linspace
current_image.width
current_image.height
np.nonzero
current_draw.rectangle
np.argmin
current_draw.text
current_panel.width
current_panel.height
point.tolist
query_descriptions[index].strip
cells.items
mask.sum
json.dumps
_client(client).chat_json
boxes.append
selected.copy
cells[number].copy
source_panel.size
source_scale.tolist
current_panel.size
current_scale.tolist
np.int64
```

写入目标：

```python
source_image
current_image
source_owner
current_owner
interaction
source_region
source_spill
pixels
nearest
item
source_panel
source_scale
source_draw
number
x
y
current_native
current_panel
current_scale
current_draw
x_edges
y_edges
cells
row
column
x0
x1
y0
y1
yy
xx
cells[number]
display_box
index
tx
ty
details
point
box
interaction_instruction
mask
details['current_interaction_evidence']
key
prompt
response
selected
boxes
dx
dy
diagnostics
diagnostics['current_interaction_evidence']
```

直接调用符号：

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
def owner_panel(image, owner):
```

实现说明：

> Retain owner appearance and context while marking the chosen instance.

显式输出分支：

```python
Image.fromarray(array)
```

读取的对象属性：

```python
np.array
np.rint(array[~owner].astype(float) * 0.45).astype
np.rint
array[~owner].astype
np.uint8
np.pad
Image.fromarray
```

写入目标：

```python
array
array[~owner]
padded
interior
array[owner & ~interior]
```

直接调用符号：

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
def display_panel(panel):
```

实现说明：

> Scale presentation only; small native crops otherwise lose their parts
> under grid labels. Pixel and edge mappings below stay explicit, and the
> unannotated images used by correspondence models are never resized here.

显式输出分支：

```python
(panel.resize(size, Image.Resampling.BICUBIC), scales)
```

读取的对象属性：

```python
panel.size
np.asarray
panel.resize
Image.Resampling.BICUBIC
Image.Resampling
```

写入目标：

```python
scale
size
length
scales
```

直接调用符号：

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

实现说明：

> Transfer xy queries with MARCO features and its native masked decoder.
>
> The extractor receives original RGB images; MARCO owns resize and bottom/
> right padding. Returned subpixel coordinates refer to the original target
> image. Source support and output mask membership use the same nearest-pixel
> rule as RGB-D backprojection. Empty query sets do not initialize a model.

显式输出分支：

```python
(np.empty((0, 2), dtype=float), {'point_count': 0, 'extractor_called': False})
match_features(extractor(image1, image2), pixels, image1.size, mask2)
```

读取的对象属性：

```python
np.asarray
pixels.dtype.kind
pixels.dtype
pixels.ndim
pixels.shape
pixels.astype
np.isfinite(pixels).all
np.isfinite
np.any
image1.width
image1.height
np.rint(pixels).astype
np.rint
mask1[nearest[:, 1], nearest[:, 0]].all
np.empty
image1.size
```

写入目标：

```python
image1
image2
mask1
mask2
pixels
nearest
extractor
```

直接调用符号：

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
