# `manipulation/tensor_cpu_snapshot.py` 实现契约

代码基线：`a2b3f686b866aea18af47c302bafb62e44212e22`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `masked_cpu_snapshot`

[定义位置](../../../manipulation/tensor_cpu_snapshot.py#L30)。

输入签名：

```python
def masked_cpu_snapshot(tensors, valid):
```

实现说明：

> Copy ``tensor.detach()[valid]`` fields together without numeric casts.
>
> All fields share a device and the mask's leading dimensions; their tail
> shapes and dtypes may differ. Byte concatenation performs no arithmetic,
> including for signed zeros and NaN payloads. Each returned NumPy array
> owns only its field's selected bytes, not a view retaining the whole pack.
>
> Let P be the unmasked payload size and K the selected payload size, K<=P.
> Contiguous source copies plus the pack use at most 2P temporary bytes.
> The masked gather uses P+K plus its row indices; discard P before the CPU
> copy. At unpacking, discard the device selection before producing K field
> bytes from K host bytes. Thus peak extra payload is at most 2P, rather than
> retaining all three full-size representations together. The scene caller
> admits one additional P plus mask indices beyond its existing P allowance.

显式输出分支：

```python
()
tuple(result)
```

读取的对象属性：

```python
torch.Tensor
valid.dtype
torch.bool
tensors[0].device
valid.shape
valid.numel
value.layout
torch.strided
value.device
value.shape
valid.ndim
value.dtype
math.prod
metadata.append
value.element_size
torch.cat
value.detach().clone(memory_format=torch.contiguous_format).reshape(rows, width).view
value.detach().clone(memory_format=torch.contiguous_format).reshape
value.detach().clone
value.detach
torch.contiguous_format
torch.uint8
valid.reshape
selected.cpu().numpy
selected.cpu
host[:, offset:end].copy().view(dtype).reshape
host[:, offset:end].copy().view
host[:, offset:end].copy
result.append
```

写入目标：

```python
tensors
device
leading
rows
metadata
value
tail
width
packed
_
selected
host
count
offset
result
element_size
dtype
end
field
```

直接调用符号：

```python
tuple
isinstance
ValueError
valid.numel
TypeError
math.prod
metadata.append
value.element_size
torch.cat
value.detach().clone(memory_format=torch.contiguous_format).reshape(rows, width).view
value.detach().clone(memory_format=torch.contiguous_format).reshape
value.detach().clone
value.detach
zip
valid.reshape
selected.cpu().numpy
selected.cpu
len
host[:, offset:end].copy().view(dtype).reshape
host[:, offset:end].copy().view
host[:, offset:end].copy
result.append
```
