# `manipulation/collision_records.py` 实现契约

代码基线：`82a3d66`。本次仅文档提交引用该实现提交，组件代码与该基线一致。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_array_layout`

[定义位置](../../../manipulation/collision_records.py#L28)。

输入签名：

```python
def _array_layout(shape, dtype):
```

实现说明：

> Check integer dimensions and byte size before any payload allocation.

显式输出分支：

```python
(_DTYPES[dtype], byte_count)
```

读取的对象属性：

```python
math.prod
_DTYPES[dtype].itemsize
```

写入目标：

```python
value
byte_count
```

直接调用符号：

```python
isinstance
len
any
type
ValueError
math.prod
```

## `encode_array`

[定义位置](../../../manipulation/collision_records.py#L42)。

输入签名：

```python
def encode_array(value):
```

实现说明：

> Encode a NumPy array or an already detached CPU Torch tensor.
>
> Payload length is exactly ``4 * ceil(raw_bytes / 3)`` ASCII bytes, where
> ``raw_bytes = prod(shape) * dtype.itemsize``. Shape and dtype metadata add
> only a small per-array cost. GPU transfer and graph detachment must occur
> explicitly at the recording call site, before using this CPU codec.

显式输出分支：

```python
{'shape': shape, 'dtype': dtype, 'data_base64': base64.b64encode(raw).decode('ascii')}
```

读取的对象属性：

```python
torch.Tensor
value.device.type
value.device
value.requires_grad
value.numpy
np.ndarray
value.shape
value.dtype.newbyteorder('<').str
value.dtype.newbyteorder
value.dtype
value.astype(target_dtype, copy=False).tobytes
value.astype
base64.b64encode(raw).decode
base64.b64encode
```

写入目标：

```python
value
shape
dtype
target_dtype
_
raw
```

直接调用符号：

```python
isinstance
ValueError
value.numpy
list
value.dtype.newbyteorder
_array_layout
value.astype(target_dtype, copy=False).tobytes
value.astype
base64.b64encode(raw).decode
base64.b64encode
```

## `decode_array`

[定义位置](../../../manipulation/collision_records.py#L69)。

输入签名：

```python
def decode_array(record):
```

实现说明：

> Decode a strictly bounded record as a read-only NumPy array.
>
> The array shares its immutable decoded bytes, avoiding an additional full
> payload copy. A caller intentionally editing values must request copy().
> Exact encoded length is checked before base64 decoding, then exact raw
> length before reshape. No metadata-driven large allocation is performed.

显式输出分支：

```python
np.frombuffer(raw, dtype=dtype).reshape(record['shape'])
```

读取的对象属性：

```python
base64.b64decode
binascii.Error
np.frombuffer(raw, dtype=dtype).reshape
np.frombuffer
```

写入目标：

```python
dtype
byte_count
encoded
raw
```

直接调用符号：

```python
isinstance
set
ValueError
_array_layout
len
base64.b64decode
np.frombuffer(raw, dtype=dtype).reshape
np.frombuffer
```

## `slice_layer_selection`

[定义位置](../../../manipulation/collision_records.py#L92)。

输入签名：

```python
def slice_layer_selection(records, index=(0, 0)):
```

实现说明：

> Retain one [batch, time] state for each actor/target layer record.
>
> Input and output map actor/target names to LayerSelection.record() values.
> Singleton batch/time axes are preserved, so independent geometry checking
> uses the same schema for a full bounded chunk and an archived single state.
> Each array is decoded separately; only its small selected slice is encoded
> into the output. No full-horizon decision record is copied into each state.

显式输出分支：

```python
result
```

读取的对象属性：

```python
records.items
fine.dtype
np.dtype
fine.ndim
fine.shape
array.shape
array.dtype.kind
array.dtype
```

写入目标：

```python
value
batch
time
result
target
record
body_ids
fine
state_shape
sliced
name
suffix
kind
array
sliced[name]
result[target]
```

直接调用符号：

```python
isinstance
len
any
type
ValueError
records.items
set
decode_array
np.dtype
list
encode_array
```
