# `manipulation/optimizer_device_checks.py` 实现契约

代码基线：`__IMPLEMENTATION_COMMIT__`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `has_active_device_checks`

[定义位置](../../../manipulation/optimizer_device_checks.py#L23)。

输入签名：

```python
def has_active_device_checks():
```

实现说明：

> Whether this context must retain device-only optimizer control flow.

显式输出分支：

```python
_active_checks.get() is not None
```

读取的对象属性：

```python
_active_checks.get
```

写入目标：

无显式条目。

直接调用符号：

```python
_active_checks.get
```

## `DeviceChecks`

[定义位置](../../../manipulation/optimizer_device_checks.py#L29)。

```python
class DeviceChecks():
```

类字段：

无显式条目。

实现说明：

> One fixed device mask with stable, warmup-registered error categories.

## `DeviceChecks.__init__`

[定义位置](../../../manipulation/optimizer_device_checks.py#L32)。

输入签名：

```python
def __init__(self, device):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
torch.zeros
torch.int64
```

写入目标：

```python
self._mask
self._errors
self._constants
```

直接调用符号：

```python
torch.zeros
```

## `DeviceChecks.storage_bytes`

[定义位置](../../../manipulation/optimizer_device_checks.py#L38)。

输入签名：

```python
def storage_bytes(self):
```

实现说明：

> Device tensor payload; host key metadata uses native host admission.

显式输出分支：

```python
self._mask.numel() * self._mask.element_size() + sum((value.numel() * value.element_size() for value in self._constants.values()))
```

读取的对象属性：

```python
self._mask.numel
self._mask
self._mask.element_size
value.numel
value.element_size
self._constants.values
self._constants
```

写入目标：

```python
value
```

直接调用符号：

```python
self._mask.numel
self._mask.element_size
sum
value.numel
value.element_size
self._constants.values
```

## `DeviceChecks.clear_constants`

[定义位置](../../../manipulation/optimizer_device_checks.py#L43)。

输入签名：

```python
def clear_constants(self):
```

实现说明：

> Release only after all graphs referencing this scope were reset.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._constants.clear
self._constants
```

写入目标：

无显式条目。

直接调用符号：

```python
self._constants.clear
```

## `DeviceChecks.reset`

[定义位置](../../../manipulation/optimizer_device_checks.py#L47)。

输入签名：

```python
def reset(self):
```

实现说明：

> Clear before a block, outside its graph; keep the captured address.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._mask.zero_
self._mask
```

写入目标：

无显式条目。

直接调用符号：

```python
self._mask.zero_
```

## `DeviceChecks.scope`

[定义位置](../../../manipulation/optimizer_device_checks.py#L52)。

输入签名：

```python
def scope(self):
```

实现说明：

> Apply only to this execution context, including its backward pass.

显式输出分支：

```python
self
```

读取的对象属性：

```python
_active_checks.set
_active_checks.reset
```

写入目标：

```python
token
```

直接调用符号：

```python
_active_checks.set
_active_checks.reset
```

## `DeviceChecks.record`

[定义位置](../../../manipulation/optimizer_device_checks.py#L60)。

输入签名：

```python
def record(self, condition, message, error_type):
```

显式输出分支：

```python
condition
```

读取的对象属性：

```python
condition.device
self._mask.device
self._mask
self._errors
self._mask.bitwise_or_
(~condition).to
torch.int64
```

写入目标：

```python
key
self._errors[key]
```

直接调用符号：

```python
ValueError
len
self._mask.bitwise_or_
(~condition).to
```

## `DeviceChecks.raise_if_failed`

[定义位置](../../../manipulation/optimizer_device_checks.py#L73)。

输入签名：

```python
def raise_if_failed(self):
```

实现说明：

> Read once after the complete block, before returning any result.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._mask.item
self._mask
self._errors.items
self._errors
```

写入目标：

```python
faults
message
error_type
bit
```

直接调用符号：

```python
int
self._mask.item
self._errors.items
error_type
```

## `require_tensor`

[定义位置](../../../manipulation/optimizer_device_checks.py#L81)。

输入签名：

```python
def require_tensor(condition, message, *, error_type=ValueError):
```

实现说明：

> Validate eagerly, or return the recorded predicate for safe selection.
> 
> A returned tensor is never a Python branch condition. The guarded caller
> must use it in ``torch.where`` with an explicitly safe value. Outside an
> optimizer scope this function raises immediately or returns ``None``.

显式输出分支：

```python
None
checks.record(condition, message, error_type)
```

读取的对象属性：

```python
torch.Tensor
condition.dtype
torch.bool
condition.ndim
_active_checks.get
checks.record
```

写入目标：

```python
checks
```

直接调用符号：

```python
isinstance
ValueError
_active_checks.get
bool
error_type
checks.record
```

## `_host_constant_key`

[定义位置](../../../manipulation/optimizer_device_checks.py#L99)。

输入签名：

```python
def _host_constant_key(value, dtype):
```

实现说明：

> Describe host bits/types/layout without any numeric promotion.

显式输出分支：

```python
(pickle.dumps(value, protocol=5), strides, dtype)
```

读取的对象属性：

```python
np.ndarray
value.strides
pickle.dumps
```

写入目标：

```python
strides
```

直接调用符号：

```python
isinstance
pickle.dumps
```

## `constant_tensor`

[定义位置](../../../manipulation/optimizer_device_checks.py#L105)。

输入签名：

```python
def constant_tensor(value, like, *, dtype=None):
```

实现说明：

> Keep model constants alive for capture using their original conversion.
> 
> The cache belongs to a single optimizer's graph lifetime. Keys describe
> host values only; tensor inputs always keep the original ``as_tensor``
> path, including their gradients. No device values are read to form keys.
> Host serialization describes a key without promoting mixed Python values;
> the actual conversion receives the original value/type. Array strides are
> also part of the key because layout can select a different numeric kernel.

显式输出分支：

```python
torch.as_tensor(value, dtype=target_dtype, device=like.device)
checks._constants[key]
```

读取的对象属性：

```python
like.dtype
_active_checks.get
torch.Tensor
like.device.type
like.device
torch.as_tensor
checks._mask.device
checks._mask
checks._constants
```

写入目标：

```python
target_dtype
checks
key
checks._constants[key]
```

直接调用符号：

```python
_active_checks.get
isinstance
torch.as_tensor
ValueError
_host_constant_key
```
