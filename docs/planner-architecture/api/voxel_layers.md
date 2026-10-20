# `manipulation/voxel_layers.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_measured_spheres`

[定义位置](../../../manipulation/voxel_layers.py#L27)。

输入签名：

```python
def _measured_spheres(spheres, margin):
```

输出分支的源码表达式：

```python
spheres
torch.cat((spheres[..., :3], rounded + (radius - radius.detach())), dim=-1)
```

直接调用：

```python
radius.detach().to
radius.detach
torch.nextafter
torch.full
target.to
torch.where
rounded.to
torch.cat
```

## `validate_body_groups`

[定义位置](../../../manipulation/voxel_layers.py#L48)。

输入签名：

```python
def validate_body_groups(groups, sphere_count):
```

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
tuple(result)
```

直接调用：

```python
set
deepcopy
isinstance
any
type
len
geometry_names.intersection
ValueError
names.add
geometry_names.update
indices.extend
result.append
sorted
list
range
tuple
```

## `LayerSelection`

[定义位置](../../../manipulation/voxel_layers.py#L76)。

```python
class LayerSelection:
    sphere_fine: torch.Tensor
    body_fine: torch.Tensor
    influence_aabb: torch.Tensor
    index_lower: torch.Tensor
    index_upper: torch.Tensor
    body_ids: tuple
```

该类型的直接方法：`record`。

## `LayerSelection.record`

[定义位置](../../../manipulation/voxel_layers.py#L86)。

输入签名：

```python
def LayerSelection.record(self):
```

直接读取的对象属性：`self.body_ids`、`self.body_fine`、`self.influence_aabb`、`self.index_lower`、`self.index_upper`。

输出分支的源码表达式：

```python
{'body_ids': list(self.body_ids), 'body_fine': encode_array(self.body_fine.detach().cpu()), 'influence_aabb_grid_m': encode_array(self.influence_aabb.detach().cpu()), 'index_lower': encode_array(self.index_lower.detach().cpu()), 'index_upper': encode_array(self.index_upper.detach().cpu())}
```

直接调用：

```python
list
encode_array
self.body_fine.detach().cpu
self.body_fine.detach
self.influence_aabb.detach().cpu
self.influence_aabb.detach
self.index_lower.detach().cpu
self.index_lower.detach
self.index_upper.detach().cpu
self.index_upper.detach
```

## `FineRegion`

[定义位置](../../../manipulation/voxel_layers.py#L97)。

```python
class FineRegion:
    ...
    ...
```

该类型的直接方法：`__init__`、`storage_bytes`、`_constants`、`_group_indices`、`_box_count`、`_envelope_support`、`validate_selection_envelope`、`select`、`prepare_guidance_boxes`、`guidance_weights`。

## `FineRegion.__init__`

[定义位置](../../../manipulation/voxel_layers.py#L107)。

输入签名：

```python
def FineRegion.__init__(self, record):
```

直接读取的对象属性：`self.pose`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.record`、`self._prefix`、`self._device_prefix`、`self._device_constants`、`self._device_groups`、`self.shape`、`self.pose`、`self.size`、`self.support`、`self.margin`、`self.guidance_boxes`、`self._device_guidance_boxes`。

直接调用：

```python
deepcopy
np.asarray
ValueError
tuple
int
float
min
math.prod
np.isfinite(pose).all
np.isfinite
np.array_equal
np.isfinite([size, support, margin]).all
np.frombuffer
bytes.fromhex
len
math.ceil
np.any
np.unpackbits(packed)[:len(bits)].astype
np.unpackbits
(~valid).reshape(tuple(shape)).astype
(~valid).reshape
np.pad
range
prefix.cumsum
torch.from_numpy
pose.copy
self.pose.setflags
np.empty
```

## `FineRegion.storage_bytes`

[定义位置](../../../manipulation/voxel_layers.py#L155)。

输入签名：

```python
@property
def FineRegion.storage_bytes(self):
```

直接读取的对象属性：`self._prefix`、`self.guidance_boxes`。

输出分支的源码表达式：

```python
self._prefix.numel() * self._prefix.element_size() + (3 + 3 + 7 + 2) * 8 + self.guidance_boxes.nbytes
```

直接调用：

```python
self._prefix.numel
self._prefix.element_size
```

## `FineRegion._constants`

[定义位置](../../../manipulation/voxel_layers.py#L161)。

输入签名：

```python
def FineRegion._constants(self, device):
```

直接读取的对象属性：`self._device_constants`、`self.shape`、`self.pose`。

输出分支的源码表达式：

```python
self._device_constants[device]
```

直接写入的状态或数组位置：`self._device_constants[device]`。

直接调用：

```python
torch.tensor
```

## `FineRegion._group_indices`

[定义位置](../../../manipulation/voxel_layers.py#L173)。

输入签名：

```python
def FineRegion._group_indices(self, device, groups):
```

直接读取的对象属性：`self._device_groups`。

输出分支的源码表达式：

```python
self._device_groups[key]
```

直接写入的状态或数组位置：`self._device_groups[key]`。

直接调用：

```python
tuple
torch.tensor
```

## `FineRegion._box_count`

[定义位置](../../../manipulation/voxel_layers.py#L182)。

输入签名：

```python
def FineRegion._box_count(self, lower, upper):
```

直接读取的对象属性：`self._device_prefix`、`self._prefix`、`self._constants`。

输出分支的源码表达式：

```python
count
```

直接写入的状态或数组位置：`self._device_prefix[key]`。

直接调用：

```python
str
self._prefix.to
self._constants
torch.maximum
torch.zeros_like
torch.minimum
torch.zeros
enumerate
tuple
```

## `FineRegion._envelope_support`

[定义位置](../../../manipulation/voxel_layers.py#L199)。

输入签名：

```python
def FineRegion._envelope_support(self, envelope):
```

直接读取的对象属性：`self._constants`、`self.size`、`self.support`、`self._box_count`。

输出分支的源码表达式：

```python
(bounds & inside & (self._box_count(lo, hi) == 0), lo, hi)
```

直接调用：

```python
ValueError
require_tensor
torch.isfinite(envelope).all
torch.isfinite
(low > high).any
torch.where
torch.zeros_like
self._constants
torch.nextafter
torch.floor(lo_coordinate).to
torch.floor
torch.ceil(hi_coordinate).to
torch.ceil
((low > allowed_low) & (high < allowed_high)).all
((lo >= 0) & (hi < shape)).all
self._box_count
```

## `FineRegion.validate_selection_envelope`

[定义位置](../../../manipulation/voxel_layers.py#L229)。

输入签名：

```python
def FineRegion.validate_selection_envelope(self, influence_aabb_grid_m, index_lower, index_upper):
```

直接读取的对象属性：`self._envelope_support`。

输出分支的源码表达式：

```python
{'indices_match': ((index_lower == lo) & (index_upper == hi)).all(-1), 'fine_supported': supported}
```

直接调用：

```python
self._envelope_support
ValueError
((index_lower == lo) & (index_upper == hi)).all
```

## `FineRegion.select`

[定义位置](../../../manipulation/voxel_layers.py#L238)。

输入签名：

```python
def FineRegion.select(self, spheres, groups):
```

直接读取的对象属性：`self.margin`、`self._constants`、`self._group_indices`、`self._envelope_support`。

输出分支的源码表达式：

```python
LayerSelection(per_sphere, chosen, envelope, lo, hi, tuple((group['body_id'] for group in groups)))
```

直接写入的状态或数组位置：`per_sphere[..., idx]`。

直接调用：

```python
ValueError
validate_body_groups
require_tensor
torch.isfinite(spheres).all
torch.isfinite
(spheres[..., 3] < 0).any
torch.where
torch.zeros_like
_measured_spheres(spheres.detach(), self.margin).to
_measured_spheres
spheres.detach
self._constants
torch.nextafter
self._group_indices
torch.nextafter(local_low[..., idx, :] - radius, toward_low).amin
torch.nextafter(local_high[..., idx, :] + radius, toward_high).amax
lower.append
upper.append
torch.stack
self._envelope_support
torch.empty
enumerate
LayerSelection
tuple
```

## `FineRegion.prepare_guidance_boxes`

[定义位置](../../../manipulation/voxel_layers.py#L285)。

输入签名：

```python
def FineRegion.prepare_guidance_boxes(self, envelopes_world, *, maximum_boxes=16):
```

直接读取的对象属性：`self.pose`、`self._envelope_support`、`self._prefix`、`self.shape`、`self.size`、`self.support`、`self.guidance_boxes`。

输出分支的源码表达式：

```python
{'witnesses': len(envelopes), 'source_supported_witnesses': int(selected.sum()), 'unsupported_witnesses': int((~selected).sum()), 'cuboids': len(boxes), 'duplicate_cuboid_candidates': duplicate_boxes, 'supported_uncovered_envelopes': int((selected.numpy() & ~covered).sum()), 'cuboid_capacity_reached': len(boxes) == maximum_boxes, 'overlap_m': 4 * self.size, 'cuboids_grid_m': self.guidance_boxes.tolist()}
```

直接写入的状态或数组位置：`trial[axis]`、`lo[axis]`、`hi[axis]`、`self.guidance_boxes`、`self._device_guidance_boxes`。

直接调用：

```python
np.asarray
len
np.isfinite(envelopes).all
np.isfinite
np.any
type
ValueError
self._envelope_support
torch.from_numpy
self._prefix.numpy
set
np.finfo
float
np.abs(self.pose[:3]).max
np.abs
(shape * self.size).max
np.stack
np.flatnonzero
selected.numpy
lower[index].numpy().copy
lower[index].numpy
upper[index].numpy().copy
upper[index].numpy
range
int
lo.copy
supported
hi.copy
np.maximum
np.minimum
np.all
tuple
box.ravel
box_keys.add
boxes.append
np.asarray(boxes, dtype=np.float64).reshape
self.guidance_boxes.setflags
np.zeros
selected.sum
(~selected).sum
(selected.numpy() & ~covered).sum
self.guidance_boxes.tolist
```

## `FineRegion.prepare_guidance_boxes.supported`

[定义位置](../../../manipulation/voxel_layers.py#L304)。

输入签名：

```python
def FineRegion.prepare_guidance_boxes.supported(lo, hi):
```

输出分支的源码表达式：

```python
count == 0
```

直接调用：

```python
tuple
int
enumerate
```

## `FineRegion.guidance_weights`

[定义位置](../../../manipulation/voxel_layers.py#L385)。

输入签名：

```python
def FineRegion.guidance_weights(self, spheres, groups):
```

直接读取的对象属性：`self.margin`、`self._device_guidance_boxes`、`self.guidance_boxes`、`self._constants`、`self._group_indices`、`self.size`。

输出分支的源码表达式：

```python
(sphere_weights, body_weights)
```

直接写入的状态或数组位置：`self._device_guidance_boxes[device]`、`sphere_weights[..., idx]`。

直接调用：

```python
validate_body_groups
_measured_spheres(spheres, self.margin).to
_measured_spheres
torch.tensor
self.guidance_boxes.copy
self._constants
self._group_indices
(centers[..., idx, :] - expanded[..., idx, 3:4]).amin
(centers[..., idx, :] + expanded[..., idx, 3:4]).amax
len
torch.minimum(low[..., None, :] - boxes[:, 0], boxes[:, 1] - high[..., None, :]).amin
torch.minimum
(clearance / (4 * self.size)).clamp
(phase.square() * (3.0 - 2.0 * phase)).amax
phase.square
low.sum
weights.append
weight.to
torch.stack
torch.empty
enumerate
```

## `LayeredVoxelQuery`

[定义位置](../../../manipulation/voxel_layers.py#L423)。

```python
class LayeredVoxelQuery:
    ...
    ...
```

该类型的直接方法：`__init__`、`clone_for_rollout`、`inflated_spheres`、`__call__`、`background_guidance`。

## `LayeredVoxelQuery.__init__`

[定义位置](../../../manipulation/voxel_layers.py#L426)。

输入签名：

```python
def LayeredVoxelQuery.__init__(self, coarse, fine, region):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.coarse`、`self.fine`、`self.region`、`self.owner_ids`、`self.source`、`self.independent_geometry`、`self.refinement_source`、`self.mapping_report`。

直接调用：

```python
ValueError
FineRegion
query_geometry
deepcopy
```

## `LayeredVoxelQuery.clone_for_rollout`

[定义位置](../../../manipulation/voxel_layers.py#L438)。

输入签名：

```python
def LayeredVoxelQuery.clone_for_rollout(self):
```

直接读取的对象属性：`self.coarse`、`self.fine`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result.coarse`、`result.fine`。

直接调用：

```python
copy
self.coarse.clone_for_rollout
self.fine.clone_for_rollout
```

## `LayeredVoxelQuery.inflated_spheres`

[定义位置](../../../manipulation/voxel_layers.py#L443)。

输入签名：

```python
def LayeredVoxelQuery.inflated_spheres(self, spheres):
```

直接读取的对象属性：`self.region`。

输出分支的源码表达式：

```python
_measured_spheres(spheres, self.region.margin)
```

直接调用：

```python
_measured_spheres
```

## `LayeredVoxelQuery.__call__`

[定义位置](../../../manipulation/voxel_layers.py#L447)。

输入签名：

```python
def LayeredVoxelQuery.__call__(self, spheres, *, key, selection):
```

直接读取的对象属性：`self.inflated_spheres`、`self.coarse`、`self.fine`。

输出分支的源码表达式：

```python
torch.where(selection.sphere_fine, fine, coarse)
```

直接调用：

```python
ValueError
self.inflated_spheres
self.coarse
self.fine
torch.where
```

## `LayeredVoxelQuery.background_guidance`

[定义位置](../../../manipulation/voxel_layers.py#L458)。

输入签名：

```python
def LayeredVoxelQuery.background_guidance(self, spheres, *, key, groups):
```

直接读取的对象属性：`self.inflated_spheres`、`self.coarse`、`self.fine`、`self.region`。

输出分支的源码表达式：

```python
coarse + weights * (fine - coarse)
```

直接调用：

```python
self.inflated_spheres
self.coarse
self.fine
self.region.guidance_weights
```
