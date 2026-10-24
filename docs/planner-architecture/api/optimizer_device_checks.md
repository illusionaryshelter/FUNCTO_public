# `manipulation/optimizer_device_checks.py` 实现契约

代码基线：`be44b54`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `require_tensor`

[定义位置](../../../manipulation/optimizer_device_checks.py#L12)。

输入签名：

```python
def require_tensor(condition, message, *, error_type=ValueError):
```

实现说明：

> Require one scalar Boolean predicate and preserve its declared error.

显式输出分支：

无显式条目。

读取的对象属性：

```python
torch.Tensor
condition.dtype
torch.bool
condition.ndim
```

写入目标：

无显式条目。

直接调用符号：

```python
isinstance
ValueError
bool
error_type
```

## `constant_tensor`

[定义位置](../../../manipulation/optimizer_device_checks.py#L21)。

输入签名：

```python
def constant_tensor(value, like, *, dtype=None):
```

实现说明：

> Convert declared constants with the original dtype, layout and gradients.

显式输出分支：

```python
torch.as_tensor(value, dtype=like.dtype if dtype is None else dtype, device=like.device)
```

读取的对象属性：

```python
torch.as_tensor
like.dtype
like.device
```

写入目标：

无显式条目。

直接调用符号：

```python
torch.as_tensor
```
