# `manipulation/skill_grounding.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../15-skill-grounding.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_validate_relations`

[定义位置](../../../manipulation/skill_grounding.py#L22)。

输入签名：

```python
def _validate_relations(request):
```

实现说明：

> Validate declared coordinate identities before model computation.

显式输出分支：

无显式条目。

读取的对象属性：

```python
request.get
relations.get
relations.get('A_context', {}).items
feedback.get
```

写入目标：

```python
feedback
relations
field
value
```

直接调用符号：

```python
request.get
relations.get
ValueError
relations.get('A_context', {}).items
feedback.get
```

## `_reference`

[定义位置](../../../manipulation/skill_grounding.py#L34)。

输入签名：

```python
def _reference(skill, F, feedback, relations, anchor, support):
```

实现说明：

> Use the existing complete relation and its declared task geometry.

显式输出分支：

```python
generate_observed_reference(skill['representation'], functional_frame=F, J=relations['J'], A=relations['A'], T_world_eef=feedback['eef_pose'], **support)
generate_reference(skill['representation'], T_world_function=F, tool_reference_anchor=anchor, T_world_eef=feedback.get('eef_pose'), **parameters)
```

读取的对象属性：

```python
feedback.get
```

写入目标：

```python
parameters
key
```

直接调用符号：

```python
generate_observed_reference
generate_reference
feedback.get
```

## `_publish_reference`

[定义位置](../../../manipulation/skill_grounding.py#L46)。

输入签名：

```python
def _publish_reference(record, hypothesis, skill, feedback, relations, support):
```

实现说明：

> Publish the nominal operands before binding the measured entrance.

显式输出分支：

```python
None
```

读取的对象属性：

```python
record.update
reference.get
reference['eef_boundaries']['goals'][0]['pose'].copy
np.asarray
hypothesis.update
bound.diagnostics
bound.sample_numpy
```

写入目标：

```python
reference
hypothesis['reference']
name
record['reference_rotation_transfer']
hypothesis['reference_status']
hypothesis['T_world_tool']
hypothesis['tool_reference_anchor_world']
record['stage']
phase
source_clock
elapsed
bound
poses
```

直接调用符号：

```python
_reference
record.update
dict
reference.get
reference['eef_boundaries']['goals'][0]['pose'].copy
reference_phase
np.asarray
BoundToolReference
SimpleNamespace
float
hypothesis.update
bound.sample_numpy
trajectory_summary
```

## `_ground`

[定义位置](../../../manipulation/skill_grounding.py#L93)。

输入签名：

```python
def _ground(request, prepared, record):
```

显式输出分支：

```python
record
record
record
record
record
```

读取的对象属性：

```python
record.update
record['stages'].append
request.get
region.get
region['outputs']['result']['diagnostics'].get
error.diagnostics
record['reference_hypotheses'].extend
diagnostics.update
type(error).__name__
traceback.format_exc
hypothesis.update
failures.append
row['reference_diagnostics'].get
record['stages'].extend
```

写入目标：

```python
skill
feedback
binding_id
binding
record['stage']
selection
view
record['runtime']
region
frame
F
view_ids
missing_views
record['functional_frame']
field
grasp
record['grasp_failure']
record['grasp_result']
hypotheses
failures
hypothesis
diagnostics
hypothesis['status']
hypothesis['reference_diagnostics']
row
```

直接调用符号：

```python
record.update
record['stages'].append
any
prepare_runtime
run_region_transfer
request.get
len
set
dict
run_cross_instance_region
region.get
list
region['outputs']['result']['diagnostics'].get
ValueError
run_cross_instance_grasp
record['reference_hypotheses'].extend
_publish_reference
diagnostics.update
type
str
traceback.format_exc
hypothesis.update
failures.append
row['reference_diagnostics'].get
record['stages'].extend
```

## `_run`

[定义位置](../../../manipulation/skill_grounding.py#L210)。

输入签名：

```python
def _run(request, prepared, *, detector=None, client=None, feature_extractor=None):
```

显式输出分支：

```python
_ground(request, prepared, result)
result
```

读取的对象属性：

```python
request.get
result['stages'].append
result.update
type(error).__name__
traceback.format_exc
```

写入目标：

```python
skill
result
started
result['stage']
prepared
result['seconds']
```

直接调用符号：

```python
dict
request.get
perf_counter
_validate_relations
result['stages'].append
prepare
result.update
_ground
type
str
traceback.format_exc
```

## `ground_skill`

[定义位置](../../../manipulation/skill_grounding.py#L236)。

输入签名：

```python
def ground_skill(request, *, detector=None, client=None, feature_extractor=None):
```

实现说明：

> Ground a complete loaded skill using one shared observation stage.
>
> The request declares bindings, primary ``reference_binding_id``, sensor
> inputs, capture/entrance feedback, source/current relations, prerequisite
> and start context, and the selected algorithms' original parameters.
> Grounding returns all complete references and their actual failure/missing
> evidence. It starts neither planning nor physical execution.

显式输出分支：

```python
_run(request, None, detector=detector, client=client, feature_extractor=feature_extractor)
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
_run
```

## `ground_prepared_skill`

[定义位置](../../../manipulation/skill_grounding.py#L249)。

输入签名：

```python
def ground_prepared_skill(request, prepared_observations):
```

实现说明：

> Consume an explicit completed observation stage without model reruns.
>
> This stage API supports saved sensor evidence and deterministic numerical
> comparisons. The supplied object is the same ``prepare`` output, including
> actual selection, instance relation and source provenance for each binding.
> It is not an inferred relation or a substitute target-region label.

显式输出分支：

```python
_run(request, prepared_observations)
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
_run
```
