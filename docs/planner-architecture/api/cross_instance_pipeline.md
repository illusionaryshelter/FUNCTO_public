# `transfer/cross_instance/pipeline.py` 实现契约

代码基线：`96ee2dc-working-tree`。
职责和数据语义见[组件说明](../../../transfer/cross_instance/ARCHITECTURE.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `run_arrays`

[定义位置](../../../transfer/cross_instance/pipeline.py#L29)。

输入签名：

```python
def run_arrays(arrays, views, *, input_context, input_evidence, coarse_steps=40, refinement_steps=200, seed=0):
```

实现说明：

> Run original coarse/CPU-float64 refinement on prepared evidence.
>
> Input arrays use the keys returned by ``prepare_evidence``.  Source rows
> and current element IDs stay in their supplied order.  Coarse registration
> retains 25 original seeds and uses ``coarse_steps`` fits per seed.  Neural
> refinement uses ``min(30, N-1)`` source neighbors, ``refinement_steps``
> L-BFGS blocks and the explicit random seed (original default zero).
>
> Native queries of coarse and refined predictions are diagnostics: unknown
> response support does not alter correspondence, fitting or frame output.
> Numerical refinement failure retains its partial arrays and report in a
> result with ``status='failed'``.  Input appearance missing from the actual
> target domain retains a ``missing`` result before either numerical fit.

显式输出分支：

```python
result
result
result
```

读取的对象属性：

```python
result.update
arrays['target_native_valid'].all
np.isfinite(raw).all
np.isfinite
arrays.update
error.arrays
error.report
cdist(mapped, target).argmin
np.unique
per_view.append
reasons.tolist
counts.tolist
query['valid'].sum
query['unknown'].sum
np.max
np.abs
```

写入目标：

```python
arrays
views
stage_seconds
result
source
target
source_owner
target_owner
source_frame
raw
result['stage']
phase
coarse_arrays
coarse
stage_seconds['coarse_registration']
selected
coarse_points
coarse_rotation
coarse_frame
refined
refinement
stage_seconds['identity_refinement']
result['refinement']
mapped
matching
support_summary
name
points
query
per_view
view
reasons
counts
support_summary[name]
stage_seconds['mapped_observation_queries']
result['mapped_observation_support']
frame
evidence
arrays['region_frame_world']
stage_seconds['region_frame_and_direction']
```

直接调用符号：

```python
dict
list
result.update
len
arrays['target_native_valid'].all
np.isfinite(raw).all
np.isfinite
perf_counter
register_candidate_context
int
arrays.update
read_region_frame
refine_identity_mapping
min
str
cdist(mapped, target).argmin
cdist
query_views
query_arrays
np.unique
per_view.append
zip
reasons.tolist
counts.tolist
query['valid'].sum
query['unknown'].sum
orientation_evidence
float
np.max
np.abs
```

## `run_region_transfer`

[定义位置](../../../transfer/cross_instance/pipeline.py#L126)。

输入签名：

```python
def run_region_transfer(skill, views, *, observation_model, context=None, surface=None, sampling_radius_m=0.03, coarse_steps=40, refinement_steps=200, seed=0):
```

实现说明：

> Prepare two named observations and compute one full region result.
>
> ``surface`` is an explicit replay input with its original element IDs;
> otherwise the selected cross-instance strategy constructs the observed
> surface exactly once.  ``context`` preserves the public entry's source,
> prerequisite and start-state identities without interpreting task names.

显式输出分支：

```python
result
```

读取的对象属性：

无显式条目。

写入目标：

```python
phase
arrays
prepared
provenance
metadata
evidence_seconds
result
result['grounding_stage_seconds']['prepare_observed_evidence']
```

直接调用符号：

```python
perf_counter
prepare_evidence
run_arrays
```
