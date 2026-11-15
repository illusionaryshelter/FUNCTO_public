# `manipulation/grounding_observations.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../15-skill-grounding.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `load_scene`

[定义位置](../../../manipulation/grounding_observations.py#L24)。

输入签名：

```python
def load_scene(scene_directory, view_ids):
```

实现说明：

> Read a saved sensor capture without opening native simulation state.
>
> Each explicit view folder supplies RGB, optical depth and calibration.
> Layout is acquisition metadata; only its declared capture time is used.
> Robot feedback retains its own named joint/pose/time fields unchanged.

显式输出分支：

```python
dict(views=views, feedback=feedback, acquisition=layout, scene=str(directory.resolve()), simulation_time_s=layout['sim_time_s'])
```

读取的对象属性：

```python
json.loads
(directory / 'robot_feedback.json').read_text
(directory / 'layout.json').read_text
(folder / 'calibration.json').read_text
Image.open
opened.convert
views.append
np.load
np.asarray
directory.resolve
```

写入目标：

```python
directory
feedback
layout
views
identity
folder
calibration
opened
image
```

直接调用符号：

```python
Path
json.loads
(directory / 'robot_feedback.json').read_text
(directory / 'layout.json').read_text
(folder / 'calibration.json').read_text
Image.open
opened.convert
views.append
dict
np.load
np.asarray
str
directory.resolve
```

## `prepare`

[定义位置](../../../manipulation/grounding_observations.py#L49)。

输入签名：

```python
def prepare(request, *, detector=None, client=None, feature_extractor=None, on_stage=None):
```

实现说明：

> Prepare explicit bindings once and retain every missing-input state.
>
> ``request['bindings']`` maps program-given binding IDs to ``skill``,
> ``views``, ``source_owner_id``, ``current_entity_id`` and ``owner_role``.
> The views are ordered raw sensor records. An optional output directory
> records real detections before the VLM call and stores exact FP32 feature
> values in uncompressed NPZ files. It does not build any region surface;
> the selected numerical strategy owns that representation.

显式输出分支：

```python
dict(bindings=prepared_bindings, semantic_result=semantic, stage_seconds=dict(select_current_entity=selection_seconds, prepare_observations=perf_counter() - started - selection_seconds))
```

读取的对象属性：

```python
request.get
request['bindings'].items
row.items
view.items
row['views'].items
semantic['bindings'].items
np.asarray
view.update
view['missing_reasons'].extend
view['owner_evidence'].update
np.zeros_like
Image.fromarray
np.savez_compressed
np.savez
Image.fromarray(observation['crop_rgb']).save
views.append
prepared_bindings[binding_id].update
```

写入目标：

```python
detector
client
feature_extractor
output
binding_inputs
prepared_bindings
started
binding_id
binding
views
ordered
prepared
key
view
binding_inputs[binding_id]
prepared_bindings[binding_id]
semantic
selection_seconds
saved
value
view_id
identity
row
owner_id
owner_source
associated
strategy
raw
selected
view['missing_reasons']
observation
owner
owner_prepared
scope
view['background_mask']
view['semantic_selection']
view['feature']
folder
```

直接调用符号：

```python
Gemini
request.get
Path
perf_counter
on_stage
request['bindings'].items
iter
prepare_interaction_candidates
dict
select_bindings
row.items
view.items
row['views'].items
semantic['bindings'].items
write_json
region_strategy
len
prepare_selected_observation
np.asarray
view.update
ValueError
prepare_mask_observation
view['missing_reasons'].extend
view['owner_evidence'].update
np.zeros_like
feature_extractor
Image.fromarray
np.savez_compressed
np.savez
Image.fromarray(observation['crop_rgb']).save
views.append
prepared_bindings[binding_id].update
```

## `prepare.recorded_detection`

[定义位置](../../../manipulation/grounding_observations.py#L79)。

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
folder.mkdir
np.asarray
```

写入目标：

```python
view
detected
folder
```

直接调用符号：

```python
next
detector
len
folder.mkdir
store_detections
np.asarray
```
