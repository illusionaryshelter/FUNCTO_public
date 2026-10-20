# `manipulation/resource_admission.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `configure_host_process_root`

[定义位置](../../../manipulation/resource_admission.py#L14)。

输入签名：

```python
def configure_host_process_root(root_pid):
```

输出分支的源码表达式：

```python
{'pid': identity[0], 'create_time': identity[1]}
```

直接调用：

```python
type
ValueError
psutil.Process
current.parents
next
root.create_time
```

## `ResourceLimitError`

[定义位置](../../../manipulation/resource_admission.py#L41)。

```python
class ResourceLimitError(ValueError):
    ...
    ...
```

该类型的直接方法：`__init__`。

## `ResourceLimitError.__init__`

[定义位置](../../../manipulation/resource_admission.py#L44)。

输入签名：

```python
def ResourceLimitError.__init__(self, resource, required_bytes, limit_bytes, *, description='Resource capacity exceeded'):
```

直接读取的对象属性：`self.resource`、`self.required_bytes`、`self.limit_bytes`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.resource`、`self.required_bytes`、`self.limit_bytes`。

直接调用：

```python
str
int
super().__init__
super
```

## `require_host_memory`

[定义位置](../../../manipulation/resource_admission.py#L52)。

输入签名：

```python
def require_host_memory(new_bytes, *, pending_existing_bytes, max_rss_bytes):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
type
ValueError
psutil.Process
process.create_time
RuntimeError
any
parent.create_time
current.parents
process.memory_info
process.children
child.memory_info
psutil.virtual_memory
int
ResourceLimitError
```

## `require_device_memory`

[定义位置](../../../manipulation/resource_admission.py#L105)。

输入签名：

```python
def require_device_memory(torch, device, *, native_bytes, torch_bytes=0, max_device_used_bytes):
```

输出分支的源码表达式：

```python
new_bytes
```

直接调用：

```python
type
ValueError
map
torch.cuda.mem_get_info
max
int
torch.cuda.memory_reserved
torch.cuda.memory_allocated
ResourceLimitError
```
