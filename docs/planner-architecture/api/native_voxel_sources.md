# `manipulation/native_voxel_sources.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_source_kernel`

[定义位置](../../../manipulation/native_voxel_sources.py#L17)。

输入签名：

```python
def _source_kernel(lookup):
```

输出分支的源码表达式：

```python
kernel
```

## `_source_kernel.kernel`

[定义位置](../../../manipulation/native_voxel_sources.py#L19)。

输入签名：

```python
@wp.kernel
def _source_kernel.kernel(tsdf: BlockSparseTSDFWarp, site: wp.array(dtype=wp.int32), center: wp.array(dtype=wp.float32), size: wp.float32, minimum_weight: wp.float32, ny: wp.int32, nz: wp.int32, bits: wp.array(dtype=wp.uint8)):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`bits[i]`。

直接调用：

```python
wp.array
wp.tid
lookup
wp.int32
wp.uint8
```

## `_mask_source_sign_bounds`

[定义位置](../../../manipulation/native_voxel_sources.py#L40)。

输入签名：

```python
def _mask_source_sign_bounds(bits, *, source_center, source_shape_xyz, source_voxel_size, query_center, query_voxel_size):
```

输出分支的源码表达式：

```python
bits
```

直接写入的状态或数组位置：`selection[axis]`、`bits[tuple(selection)]`。

直接调用：

```python
np.asarray
min
np.any
np.isfinite([*source_center, *query_center, source_voxel_size, query_voxel_size]).all
np.isfinite
ValueError
float
enumerate
np.arange
slice
tuple
np.uint8
```

## `export_source_bits`

[定义位置](../../../manipulation/native_voxel_sources.py#L79)。

输入签名：

```python
def export_source_bits(mapper, config, voxel, *, max_cells=MAX_FINE_CELLS):
```

输出分支的源码表达式：

```python
_mask_source_sign_bounds(exported, source_center=config.grid_center.detach().cpu().numpy(), source_shape_xyz=config.grid_shape[::-1], source_voxel_size=float(np.float32(config.voxel_size)), query_center=voxel.pose[:3], query_voxel_size=float(voxel.voxel_size))
```

直接调用：

```python
ValueError
tuple
len
type
math.prod
torch.empty
torch.as_tensor
mapper.integrator._site_index.reshape
site.numel
bits.numel
wp.launch
_source_kernel
mapper.tsdf.get_warp_data
wp.from_torch
float
str
wp.stream_from_torch
torch.cuda.current_stream
np.ascontiguousarray
bits.cpu().numpy().reshape
bits.cpu().numpy
bits.cpu
_mask_source_sign_bounds
config.grid_center.detach().cpu().numpy
config.grid_center.detach().cpu
config.grid_center.detach
np.float32
```
