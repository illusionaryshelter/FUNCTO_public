# `manipulation/observed_mapping.py` 实现契约

代码基线：`c84b34a`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `FrozenObservedField`

[定义位置](../../../manipulation/observed_mapping.py#L25)。

```python
class FrozenObservedField(ObservedField):
```

类字段：

无显式条目。

实现说明：

> Raw weighted observations, support and entity identity in one frame.

## `FrozenObservedField.__init__`

[定义位置](../../../manipulation/observed_mapping.py#L28)。

输入签名：

```python
def __init__(self, values, observed, *, weights, origin, voxel_size, name, source):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
np.asarray(weights, dtype=np.float32).copy
np.asarray
np.float32
self.weights.shape
self.weights
self.values.shape
self.values
np.isfinite(self.weights).all
np.isfinite
np.any
self.weights.flags
self.name
self.weights.nbytes
```

写入目标：

```python
self.weights
self.weights.flags.writeable
self.name
self.source
self.owner_ids
self.storage_nbytes
```

直接调用符号：

```python
super().__init__
super
np.asarray(weights, dtype=np.float32).copy
np.asarray
np.isfinite(self.weights).all
np.isfinite
np.any
ValueError
str
frozenset
```

## `FrozenObservedField.clone_for_rollout`

[定义位置](../../../manipulation/observed_mapping.py#L39)。

输入签名：

```python
def clone_for_rollout(self):
```

实现说明：

> A rollout changes rigid poses while sharing immutable observations.

显式输出分支：

```python
self
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `FrozenObservedField.record`

[定义位置](../../../manipulation/observed_mapping.py#L43)。

输入签名：

```python
def record(self):
```

实现说明：

> Store the samples defining the set; indexes are reconstructed.

显式输出分支：

```python
{'kind': 'observed_trilinear_field_v1', 'name': self.name, 'source': self.source, 'values': encode_array(self.values), 'observed': encode_array(self.observed), 'weights': encode_array(self.weights), 'origin': self.origin.tolist(), 'voxel_size': self.voxel_size}
```

读取的对象属性：

```python
self.name
self.source
self.values
self.observed
self.weights
self.origin.tolist
self.origin
self.voxel_size
```

写入目标：

无显式条目。

直接调用符号：

```python
encode_array
self.origin.tolist
```

## `FrozenObservedField.from_record`

[定义位置](../../../manipulation/observed_mapping.py#L53)。

输入签名：

```python
def from_record(cls, record):
```

显式输出分支：

```python
cls(decode_array(record['values']), decode_array(record['observed']), weights=decode_array(record['weights']), origin=record['origin'], voxel_size=record['voxel_size'], name=record['name'], source=record['source'])
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
ValueError
cls
decode_array
```

## `FrozenObservedField.save`

[定义位置](../../../manipulation/observed_mapping.py#L60)。

输入签名：

```python
def save(self, path):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
Path(path).write_text
json.dumps
self.record
```

写入目标：

无显式条目。

直接调用符号：

```python
Path(path).write_text
Path
json.dumps
self.record
```

## `FrozenObservedField.load`

[定义位置](../../../manipulation/observed_mapping.py#L64)。

输入签名：

```python
def load(cls, path):
```

显式输出分支：

```python
cls.from_record(json.loads(Path(path).read_text(encoding='utf-8')))
```

读取的对象属性：

```python
cls.from_record
json.loads
Path(path).read_text
```

写入目标：

无显式条目。

直接调用符号：

```python
cls.from_record
json.loads
Path(path).read_text
Path
```

## `FrozenObservedField.summary`

[定义位置](../../../manipulation/observed_mapping.py#L67)。

输入签名：

```python
def summary(self):
```

显式输出分支：

```python
{'name': self.name, 'source': self.source, 'shape_xyz': self.shape.tolist(), 'origin_m': self.origin.tolist(), 'voxel_size_m': self.voxel_size, 'observed_samples': int(np.count_nonzero(self.observed)), 'supported_cells': int(np.count_nonzero(self.valid_cells)), 'occupied_candidate_cells': len(self.cells), 'representation': 'nonpositive_trilinear_values_on_fully_observed_cells', 'storage_bytes': self.storage_nbytes}
```

读取的对象属性：

```python
self.name
self.source
self.shape.tolist
self.shape
self.origin.tolist
self.origin
self.voxel_size
np.count_nonzero
self.observed
self.valid_cells
self.cells
self.storage_nbytes
```

写入目标：

无显式条目。

直接调用符号：

```python
self.shape.tolist
self.origin.tolist
int
np.count_nonzero
len
```

## `freeze_mapper`

[定义位置](../../../manipulation/observed_mapping.py#L79)。

输入签名：

```python
def freeze_mapper(mapper, *, name, source, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
```

实现说明：

> Decode the native weighted TSDF block export into XYZ sample arrays.
>
> Native block coordinates are centered, and local samples are indexed as
> z*B*B+y*B+x. The origin denotes the first sample center, not a voxel face.

显式输出分支：

```python
FrozenObservedField(values, observed, weights=weights, origin=origin, voxel_size=size, name=name, source=source)
```

读取的对象属性：

```python
mapper.config
config.enable_static
np.asarray
config.grid_shape
np.int64
math.prod
mapper.tsdf.block_size
mapper.tsdf
config.max_blocks
config.voxel_size
config.grid_center.detach().cpu().numpy().astype
config.grid_center.detach().cpu().numpy
config.grid_center.detach().cpu
config.grid_center.detach
config.grid_center
mapper.tsdf.export_blocks
exported['active_block_coords'].detach().cpu().numpy
exported['active_block_coords'].detach().cpu
exported['active_block_coords'].detach
exported['block_data'].detach().cpu().numpy
exported['block_data'].detach().cpu
exported['block_data'].detach
np.ceil(shape / block_size).astype
np.ceil
np.zeros
np.float32
np.maximum
np.minimum
np.any
np.asarray(payload, dtype=np.float32).reshape(block_size, block_size, block_size, 2).transpose
np.asarray(payload, dtype=np.float32).reshape
np.divide
config.minimum_tsdf_weight
```

写入目标：

```python
config
shape
block_size
export_bytes
size
center
origin
exported
keys
data
block_offset
values
weights
key
payload
base
lower
upper
region
a
b
local
weighted
weights[region]
observed
```

直接调用符号：

```python
ValueError
np.asarray
math.prod
int
require_host_memory
float
config.grid_center.detach().cpu().numpy().astype
config.grid_center.detach().cpu().numpy
config.grid_center.detach().cpu
config.grid_center.detach
mapper.tsdf.export_blocks
exported['active_block_coords'].detach().cpu().numpy
exported['active_block_coords'].detach().cpu
exported['active_block_coords'].detach
exported['block_data'].detach().cpu().numpy
exported['block_data'].detach().cpu
exported['block_data'].detach
np.ceil(shape / block_size).astype
np.ceil
np.zeros
tuple
zip
np.maximum
np.minimum
np.any
slice
np.asarray(payload, dtype=np.float32).reshape(block_size, block_size, block_size, 2).transpose
np.asarray(payload, dtype=np.float32).reshape
np.divide
FrozenObservedField
```

## `integrate_entity`

[定义位置](../../../manipulation/observed_mapping.py#L123)。

输入签名：

```python
def integrate_entity(observations, *, config, name, source, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES, max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES, block_checkpoint=None):
```

实现说明：

> Integrate an iterable of calibrated, identity-partitioned RGB-D views.
>
> Each row provides depth (camera-z meters), an entity mask, RGB, K with
> integer pixel centers and T_entity_camera in the OpenCV camera convention.
> Rows may describe different times: their poses already express the camera
> in the same entity-local frame. Unknown pixels contribute no depth.

显式输出分支：

```python
(result, {'observation_count': count, 'mapper': stats, **result.summary()})
```

读取的对象属性：

```python
config.device
np.where(row['mask'], row['depth'], 0.0).astype
np.where
np.float32
depth.shape
config.image_height
config.image_width
device.to_device
torch.as_tensor
np.asarray(row['image']).copy
np.asarray
torch.uint8
device.device
Pose.from_matrix
mapper.integrate
mapper.get_stats
mapper.save_blocks
result.summary
```

写入目标：

```python
bound
device
mapper
count
row
depth
observation
stats
result
```

直接调用符号：

```python
_mapper_memory_bound
require_host_memory
require_device_memory
DeviceCfg
Mapper
enumerate
np.where(row['mask'], row['depth'], 0.0).astype
np.where
ValueError
CameraObservation
device.to_device
torch.as_tensor
np.asarray(row['image']).copy
np.asarray
_mapper_intrinsics
Pose.from_matrix
list
mapper.integrate
mapper.get_stats
mapper.save_blocks
freeze_mapper
result.summary
```
