# `manipulation/tensor_cpu_snapshot.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `masked_cpu_snapshot`

[定义位置](../../../manipulation/tensor_cpu_snapshot.py#L30)。

输入签名：

```python
def masked_cpu_snapshot(tensors, valid):
```

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
()
tuple(result)
```

直接调用：

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
