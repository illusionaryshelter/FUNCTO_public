# `manipulation/selection.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../01-inputs-and-task-model.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_selected_owner`

[定义位置](../../../manipulation/selection.py#L145)。

输入签名：

```python
def _selected_owner(record, image, selection_scope):
```

输出分支的源码表达式：

```python
{'owner_id': record['owner_id'], 'mask': mask, 'description': record['description'].strip(), 'source': record['source'].strip()}
```

直接调用：

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

输出分支的源码表达式：

```python
(int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
```

直接调用：

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

输出分支的源码表达式：

```python
panel
```

直接调用：

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

输出分支的源码表达式：

```python
crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.NEAREST)
```

直接调用：

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

输出分支的源码表达式：

```python
sheet
```

直接调用：

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

输出分支的源码表达式：

```python
(prompt, images, prepared)
```

直接写入的状态或数组位置：`image_order[0]`、`prepared[name]`、`descriptions[name]`、`descriptions[name]['selected_owner']`。

直接调用：

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

直接读取的对象属性：`result.items`、`result.values`。

输出分支的源码表达式：

```python
output
```

直接写入的状态或数组位置：`decision['same_owner']`、`result[name]`、`result[name]['same_owner']`。

直接调用：

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
