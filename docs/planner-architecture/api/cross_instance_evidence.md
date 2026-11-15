# `transfer/cross_instance/evidence.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../../../transfer/cross_instance/ARCHITECTURE.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `query_arrays`

[定义位置](../../../transfer/cross_instance/evidence.py#L30)。

输入签名：

```python
def query_arrays(query, prefix):
```

实现说明：

> Flatten a completed native query without changing rows or unknowns.
>
> The point axis is shared across ``response``, the Boolean support arrays,
> and all per-view measurement/pixel interpolation provenance.  Unknown
> raw responses retain NaN in the numerical archive.

显式输出分支：

```python
output
```

读取的对象属性：

```python
np.stack
```

写入目标：

```python
output
key
output[prefix + '_view_' + key]
entry
```

直接调用符号：

```python
np.stack
```

## `prepare_evidence`

[定义位置](../../../transfer/cross_instance/evidence.py#L47)。

输入签名：

```python
def prepare_evidence(skill, views, *, observation_model, context=None, surface=None, sampling_radius_m=0.03):
```

实现说明：

> Return ``arrays, prepared_views, context, metadata`` for region fitting.
>
> ``skill['source']`` supplies ``observation``, ``feature``, ``crop_xyxy``,
> ``T_world_body_at_image`` and ``L_body_region``.  Every view supplies its
> ``id``, ``depth_m``, ``K``, ``T_world_camera_cv``, ``observation``, native
> FP32 ``feature`` and the exact feature producer's ``crop_xyxy``.  These
> fields describe the selected reference.  ``owner_observation``,
> ``owner_mask``, ``owner_evidence`` and
> ``background_mask`` separately identify the declared physical owner and
> observed background.  Object-scope inputs can share one bank; a component
> reference keeps its own mask and crop while geometry uses its owner bank.
>
> ``owner_evidence`` contains ``owner_id``, ``scope``, ``source``,
> ``reference_material_owner_id`` and ``reference_material_source``.  Equal
> declared owner IDs admit the actual reference measurements as local
> material observations of that owner, including measurements outside an
> independent owner mask.  All original masks and surface memberships stay
> intact; exact element IDs form the owner's unique observed input domain.
>
> ``observation_model`` is the existing ``ObservationModel`` or its explicit
> field mapping.  A supplied surface contains the original observation to
> element association.  Its stored ``view_ids`` must match this view order:
> reordering only the camera inputs would change measurement identities.
> ``context`` records provenance and prerequisite/start state supplied by
> the public grounding entry; this numerical component does not infer them.

显式输出分支：

```python
(arrays, prepared_views, dict(context or {}), metadata)
```

读取的对象属性：

```python
np.flatnonzero
np.zeros
np.unique
np.r_
np.asarray
np.full
depth.shape
np.int64
prepared_views.append
feature_sources.append
feature.nbytes
prepared['response'].nbytes
np.bincount
descriptors.shape
arrays.update
np.count_nonzero
query['valid'].all
np.isfinite(query['response']).all
np.isfinite
```

写入目标：

```python
views
view
source
bank
source_ids
source_frame
source_points
source_pixels
source_owner
descriptors
model
model_values
geometry_views
surface
observation_pixels
observation_views
reference_member
local_material_member
index
ids
pixels
reference_member[ids]
evidence
material_owner
local_material_member[ids]
observation_ids
target_ids
observation_elements
target_points
independent_owner_observations
local_material_observations
independent_owner_elements
local_material_elements
owner_observations
owner_element_ids
owner_observation_elements
target_owner
prepared_views
feature_sources
identifier
depth
reference_mask
crop
lookup
view_observations
uv
lookup[uv[:, 1], uv[:, 0]]
feature
prepared
query
arrays
metadata
```

直接调用符号：

```python
list
len
any
isinstance
ValueError
np.flatnonzero
source_descriptors
ObservationModel
asdict
dict
build_target_surface
np.zeros
enumerate
np.unique
np.asarray
np.full
prepare_view
prepared_views.append
feature_sources.append
query_views
np.bincount
arrays.update
query_arrays
int
np.count_nonzero
query['valid'].all
np.isfinite(query['response']).all
np.isfinite
```
