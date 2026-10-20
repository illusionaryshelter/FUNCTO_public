# `manipulation/selection.py` 实现契约

代码基线：`__IMPLEMENTATION_COMMIT__`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_selected_owner`

[定义位置](../../../manipulation/selection.py#L145)。

输入签名：

```python
def _selected_owner(record, image, selection_scope):
```

实现说明：

> Validate identity evidence without treating pixel overlap as identity.

显式输出分支：

```python
{'owner_id': record['owner_id'], 'mask': mask, 'description': record['description'].strip(), 'source': record['source'].strip()}
```

读取的对象属性：

```python
record[name].strip
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
ValueError
isinstance
set
record[name].strip
_mask
record['description'].strip
record['source'].strip
```

## `_box`

[定义位置](../../../manipulation/selection.py#L163)。

输入签名：

```python
def _box(mask):
```

实现说明：

> Compute display bounds from an existing mask, without changing its pixels.

显式输出分支：

```python
(int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
```

读取的对象属性：

```python
np.nonzero
xx.min
yy.min
xx.max
yy.max
```

写入目标：

```python
yy
xx
```

直接调用符号：

```python
np.nonzero
int
xx.min
yy.min
xx.max
yy.max
```

## `_highlight`

[定义位置](../../../manipulation/selection.py#L169)。

输入签名：

```python
def _highlight(image, mask, colour, *, outline=True):
```

显式输出分支：

```python
panel
```

读取的对象属性：

```python
Image.composite
Image.new
image.size
Image.fromarray
mask.astype
np.uint8
ImageDraw.Draw(panel).rectangle
ImageDraw.Draw
```

写入目标：

```python
panel
```

直接调用符号：

```python
Image.composite
Image.new
Image.fromarray
mask.astype
ImageDraw.Draw(panel).rectangle
ImageDraw.Draw
_box
```

## `_context_crop`

[定义位置](../../../manipulation/selection.py#L177)。

输入签名：

```python
def _context_crop(image, box):
```

实现说明：

> Enlarge small candidate evidence for presentation, never for detection.

显式输出分支：

```python
crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.NEAREST)
```

读取的对象属性：

```python
math.ceil
image.crop
math.floor
image.width
image.height
crop.width
crop.height
crop.resize
Image.Resampling.NEAREST
Image.Resampling
```

写入目标：

```python
x0
y0
x1
y1
margin
crop
scale
```

直接调用符号：

```python
max
math.ceil
image.crop
math.floor
min
crop.resize
round
```

## `_panels`

[定义位置](../../../manipulation/selection.py#L189)。

输入签名：

```python
def _panels(image, candidates):
```

实现说明：

> Keep candidate masks independent and show a local crop of each one.

显式输出分支：

```python
sheet
```

读取的对象属性：

```python
math.ceil
Image.new
sheet.paste
crop.width
crop.height
ImageDraw.Draw(sheet).text
ImageDraw.Draw
```

写入目标：

```python
columns
rows
sheet
index
candidate
panel
crop
left
top
```

直接调用符号：

```python
min
len
math.ceil
Image.new
enumerate
_highlight
_context_crop
_box
sheet.paste
ImageDraw.Draw(sheet).text
ImageDraw.Draw
str
```

## `_prepare`

[定义位置](../../../manipulation/selection.py#L206)。

输入签名：

```python
def _prepare(source, instruction, views, selection_scope):
```

实现说明：

> Allowlist semantic evidence; source poses and simulator IDs are not sent.

显式输出分支：

```python
(prompt, images, prepared)
```

读取的对象属性：

```python
instruction.strip
source.keys
source.get
description.strip
query.strip
source[key].strip
reference['text'].strip
source_evidence.update
source_evidence.get('interaction_reference', {}).get
source_evidence.get
ImageDraw.Draw(panel).rectangle
ImageDraw.Draw
views.items
name.strip
view.keys
item.get
item['label'].strip
dict.fromkeys
used.add
candidates.append
images.append
image_order.append
images.extend
image_order.extend
json.dumps
```

写入目标：

```python
required_source
description
queries
query
source_evidence
semantic_fields
key
empty_part_allowed
reference
source_image
entity_scope
evidence_scope
source_region
owner
panel
source_extent
images
image_order
image_order[0]
prepared
descriptions
has_selected_owner
name
view
image
raw
item
labels
candidates
used
index
number
prepared[name]
descriptions[name]
selected_owner
descriptions[name]['selected_owner']
owner_panel
context
content
prompt
```

直接调用符号：

```python
ValueError
isinstance
instruction.strip
source.keys
source.get
description.strip
any
query.strip
source[key].strip
set
reference['text'].strip
source_evidence.update
source_evidence.get('interaction_reference', {}).get
source_evidence.get
_image
_mask
_highlight
ImageDraw.Draw(panel).rectangle
ImageDraw.Draw
_box
_context_crop
views.items
name.strip
view.keys
item.get
item['label'].strip
list
dict.fromkeys
enumerate
type
used.add
candidates.append
_detection
images.append
image_order.append
_selected_owner
images.extend
image_order.extend
_panels
json.dumps
```

## `select_grounding_candidates`

[定义位置](../../../manipulation/selection.py#L340)。

输入签名：

```python
def select_grounding_candidates(source, instruction, views, *, selection_scope, client=None):
```

实现说明：

> Select semantic candidates using a fixed, automatically assembled prompt.
> 
> ``entity`` selects a semantic object instance, which may contain multiple
> physical bodies. It requires source owner_mask as visible entity context;
> a source contact mask is optional. This scope does not bind a region to a
> body and does not replace the existing physical owner selection scopes.
> 
> ``source`` consumes existing ``image``, interaction ``mask``, ``description``
> and optional ``query_descriptions``. ``selection_scope='owner'`` additionally
> requires ``source.owner_mask``. These are compiled demonstration evidence;
> only ``instruction`` describes the current request. No source file path,
> skill-name branch, source pose or current oracle geometry is used.
> 
> ``selection_scope='interaction_reference'`` selects the functional component
> named in the offline ``source.interaction_reference`` (scope ``part``), with
> its ``object_description`` and ``interaction_part_description``. The source
> red contact mask locates the demonstrated interaction; it does not require
> the selected component to have exactly the same contact-patch extent. This
> scope needs no complete current owner mask and never manufactures one.
> 
> ``selection_scope='owner_evidence'`` interprets ``source.mask`` as visible
> evidence of the rigid entity described by the source, not as a contact mask.
> Its optional ``owner_mask`` provides wider source context only. Neither
> source nor current evidence must cover a complete owner. A nonempty current
> selection requires an explicit VLM ``same_owner=True`` assertion: all masks
> are complementary/overlapping evidence of one target, never alternatives
> between instances. Empty/unresolved decisions require ``same_owner=None``.
> This semantic membership claim does not establish measured motion rigidity.
> 
> Each ``views[view_id]`` contains an image and detector candidates. Explicit
> positive integer candidate IDs are preserved; otherwise the input index
> starting at one is used. A normal single-view call supplies one mapping
> entry. For ``entity`` and ``interaction_reference``, the same request also
> reports ``same_entity`` as true, false or unknown for the uniquely selected
> instances across views. Component references must identify the same actual
> component, not merely components of one owner. This explicit semantic
> assertion does not establish rigidity or geometric registration. Other
> scopes share context only; their local IDs establish no association.
> 
> For ``interaction_region`` or ``interaction_reference``, a view may supply ``selected_owner`` with
> exactly owner_id, mask, description and source. That actual current mask is
> shown independently in green, and the same request must verify that each
> selected part belongs to that instance. Spatial overlap alone is not used
> to associate, select, clip or merge masks. Unclear membership remains null.
> Omitting this record preserves the original request and behavior.
> 
> Each result has a status: selected, alternatives (all semantically allowed),
> unresolved, no_match, or no_candidates. ``regions`` retains the separate
> selected masks. Owner and local interaction scopes are explicit in the
> returned record; this function never silently constructs an owner_mask.
> Geometry, correspondence, accessibility and grasp feasibility remain the
> responsibilities of the existing downstream consumers.
> 
> Owner-evidence selections use ``selected`` for one or several masks, carry
> the semantic ``same_owner`` assertion, and never merge their pixels or
> upgrade them to a complete ``owner_mask``. Entity/component-reference
> selections retain ``alternatives`` but must report unknown association
> when any view offers interchangeable targets instead of one instance.

显式输出分支：

```python
output
```

读取的对象属性：

```python
candidates.values
_client(client).chat_json
response.update
response['views'].values
candidates.items
', '.join
reason.strip
result.items
result.values
output.update
```

写入目标：

```python
prompt
images
candidates
called
response
name
decision
decision['same_owner']
response_fields
result
items
evidence_scope
required
ids
reason
by_id
item
number
status
result[name]
result[name]['same_owner']
output
association
unique
row
ambiguous
```

直接调用符号：

```python
_prepare
any
candidates.values
_client(client).chat_json
_client
response.update
response['views'].values
isinstance
set
ValueError
candidates.items
', '.join
sorted
reason.strip
type
len
result.items
result.values
output.update
```
