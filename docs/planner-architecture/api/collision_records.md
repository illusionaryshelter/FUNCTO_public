# `manipulation/collision_records.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_array_layout`

[定义位置](../../../manipulation/collision_records.py#L28)。

输入签名：

```python
def _array_layout(shape, dtype):
```

输出分支的源码表达式：

```python
(_DTYPES[dtype], byte_count)
```

直接调用：

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

输出分支的源码表达式：

```python
{'shape': shape, 'dtype': dtype, 'data_base64': base64.b64encode(raw).decode('ascii')}
```

直接调用：

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

输出分支的源码表达式：

```python
np.frombuffer(raw, dtype=dtype).reshape(record['shape'])
```

直接调用：

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

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`sliced[name]`、`result[target]`。

直接调用：

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
