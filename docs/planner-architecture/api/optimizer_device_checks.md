# `manipulation/optimizer_device_checks.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `has_active_device_checks`

[定义位置](../../../manipulation/optimizer_device_checks.py#L23)。

输入签名：

```python
def has_active_device_checks():
```

输出分支的源码表达式：

```python
_active_checks.get() is not None
```

直接调用：

```python
_active_checks.get
```

## `DeviceChecks`

[定义位置](../../../manipulation/optimizer_device_checks.py#L29)。

```python
class DeviceChecks:
    ...
    ...
```

该类型的直接方法：`__init__`、`storage_bytes`、`clear_constants`、`reset`、`scope`、`record`、`raise_if_failed`。

## `DeviceChecks.__init__`

[定义位置](../../../manipulation/optimizer_device_checks.py#L32)。

输入签名：

```python
def DeviceChecks.__init__(self, device):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._mask`、`self._errors`、`self._constants`。

直接调用：

```python
torch.zeros
```

## `DeviceChecks.storage_bytes`

[定义位置](../../../manipulation/optimizer_device_checks.py#L38)。

输入签名：

```python
@property
def DeviceChecks.storage_bytes(self):
```

直接读取的对象属性：`self._mask`、`self._constants`。

输出分支的源码表达式：

```python
self._mask.numel() * self._mask.element_size() + sum((value.numel() * value.element_size() for value in self._constants.values()))
```

直接调用：

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
def DeviceChecks.clear_constants(self):
```

直接读取的对象属性：`self._constants`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
self._constants.clear
```

## `DeviceChecks.reset`

[定义位置](../../../manipulation/optimizer_device_checks.py#L47)。

输入签名：

```python
def DeviceChecks.reset(self):
```

直接读取的对象属性：`self._mask`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
self._mask.zero_
```

## `DeviceChecks.scope`

[定义位置](../../../manipulation/optimizer_device_checks.py#L52)。

输入签名：

```python
@contextmanager
def DeviceChecks.scope(self):
```

生成输出的源码表达式：

```python
self
```

直接调用：

```python
_active_checks.set
_active_checks.reset
```

## `DeviceChecks.record`

[定义位置](../../../manipulation/optimizer_device_checks.py#L60)。

输入签名：

```python
def DeviceChecks.record(self, condition, message, error_type):
```

直接读取的对象属性：`self._mask`、`self._errors`。

输出分支的源码表达式：

```python
condition
```

直接写入的状态或数组位置：`self._errors[key]`。

直接调用：

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
def DeviceChecks.raise_if_failed(self):
```

直接读取的对象属性：`self._mask`、`self._errors`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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

输出分支的源码表达式：

```python
None
checks.record(condition, message, error_type)
```

直接调用：

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

输出分支的源码表达式：

```python
(pickle.dumps(value, protocol=5), strides, dtype)
```

直接调用：

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

输出分支的源码表达式：

```python
torch.as_tensor(value, dtype=target_dtype, device=like.device)
checks._constants[key]
```

直接写入的状态或数组位置：`checks._constants[key]`。

直接调用：

```python
_active_checks.get
isinstance
torch.as_tensor
ValueError
_host_constant_key
```
