# `config/ground_skill.py` 实现契约

代码基线：`1526e60`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_plain`

[定义位置](../../../config/ground_skill.py#L26)。

输入签名：

```python
def _plain(value):
```

显式输出分支：

```python
value.tolist()
value.item()
{key: _plain(item) for (key, item) in value.items()}
[_plain(item) for item in value]
value
```

读取的对象属性：

```python
np.ndarray
value.tolist
np.generic
value.item
value.items
```

写入目标：

```python
key
item
```

直接调用符号：

```python
isinstance
value.tolist
value.item
_plain
value.items
```

## `ground_skill`

[定义位置](../../../config/ground_skill.py#L38)。

输入签名：

```python
def ground_skill(skill_directory, scene_directory, output, *, instruction, relations=None):
```

实现说明：

> Save every stage and return the native region/reference status.
>
> ``relations`` may provide explicitly sourced J/A records; it does not
> provide a target region. Missing A preserves an object-only reference.
> The current scene supplies actual robot feedback, calibrated RGB-D and an
> acquisition-state record. The scene's segmentation truth is never opened.

显式输出分支：

```python
record
```

读取的对象属性：

```python
output.mkdir
json.loads
(scene_directory / 'robot_feedback.json').read_text
(scene_directory / 'layout.json').read_text
(folder / 'calibration.json').read_text
Image.open
opened.convert
views.append
np.load
np.asarray
Path(skill_directory).resolve
scene_directory.resolve
relations.get
feedback.get
record['stages'].append
semantic_client.model
semantic_client.base_url
association['views'].items
association.get
selected.items
view.update
np.savez_compressed
Image.fromarray(prepared['crop_rgb']).save
Image.fromarray
view['image'].crop
record.update
transfer.get
outputs['result']['diagnostics'].get
reference.items
np.ndarray
reference.get
phase.reference.sample_numpy
phase.reference
np.linspace
np.save
type(error).__name__
(output / 'error.txt').write_text
traceback.format_exc
```

写入目标：

```python
skill
scene_directory
output
feedback
layout
views
name
folder
calibration
opened
rgb
record
transfer
reference
stage
relations
field
context
semantic_client
record['semantic_model']
detection_views
observed
view
selection_record
association
key
selection_record[key]
unique_views
value
associated
selected
selection_record['views'][name]
view['missing_reasons']
prepared
record['selection']
view['feature']
record['runtime']
same_state
outputs
record['region_geometry']
record['region_warnings']
record['region_timing_seconds']
record['functional_frame']
frame
arrays
record['reference_rotation_transfer']
phase
sample
record['planning_reference']
record['status']
```

直接调用符号：

```python
load_grounding_skill
Path
output.mkdir
json.loads
(scene_directory / 'robot_feedback.json').read_text
(scene_directory / 'layout.json').read_text
(folder / 'calibration.json').read_text
Image.open
opened.convert
views.append
dict
np.load
np.asarray
str
Path(skill_directory).resolve
scene_directory.resolve
relations.get
ValueError
feedback.get
record['stages'].append
Gemini
iter
observe_interaction_reference
association['views'].items
len
association.get
selected.items
prepare_selected_observation
view.update
np.savez_compressed
Image.fromarray(prepared['crop_rgb']).save
Image.fromarray
write_json
extract_image
view['image'].crop
tuple
prepare
run_region_transfer
record.update
transfer.get
list
outputs['result']['diagnostics'].get
save_surface
store_prediction
_plain
any
generate_reference
reference.items
isinstance
reference.get
reference_phase
phase.reference.sample_numpy
np.linspace
np.save
type
(output / 'error.txt').write_text
traceback.format_exc
write_reference_html
```

## `ground_skill.recorded_detection`

[定义位置](../../../config/ground_skill.py#L94)。

输入签名：

```python
def recorded_detection(image, labels):
```

显式输出分支：

```python
detected
```

读取的对象属性：

```python
np.asarray
```

写入目标：

```python
view
detected
```

直接调用符号：

```python
next
detect
store_detections
np.asarray
```

## `main`

[定义位置](../../../config/ground_skill.py#L217)。

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
args.relations
json.loads
args.relations.read_text
args.skill
args.scene
args.output
args.instruction
json.dumps
```

写入目标：

```python
parser
args
relations
record
key
```

直接调用符号：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
json.loads
args.relations.read_text
ground_skill
print
json.dumps
SystemExit
```
