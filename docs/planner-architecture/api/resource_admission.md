# `manipulation/resource_admission.py` 实现契约

代码基线：`13fefbc`。本次文档引用提交仅回填实际实现基线，源码与该代码提交相同。组件正文描述当前实现；阶段验证位于项目外 `sim-version-plans`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `configure_host_process_root`

[定义位置](../../../manipulation/resource_admission.py#L16)。

输入签名：

```python
def configure_host_process_root(root_pid):
```

实现说明：

> Bind this worker's host budget to itself or an explicit live ancestor.
>
> The standalone default remains this process plus its children. A worker
> launched by a retained simulation instead supplies that simulation's PID,
> so later geometry checks include its parent and sibling workers too. The
> root is never guessed from a shell/session tree. Once explicitly bound,
> this process cannot narrow or replace that scope for a subsequent check.

显式输出分支：

```python
{'pid': identity[0], 'create_time': identity[1]}
```

读取的对象属性：

```python
psutil.Process
current.parents
item.pid
root.pid
root.create_time
```

写入目标：

```python
current
ancestors
root
item
identity
_HOST_PROCESS_ROOT_IDENTITY
```

直接调用符号：

```python
type
ValueError
psutil.Process
current.parents
next
root.create_time
```

## `ResourceLimitError`

[定义位置](../../../manipulation/resource_admission.py#L43)。

```python
class ResourceLimitError(ValueError):
```

类字段：

无显式条目。

实现说明：

> A measured capacity refusal, distinct from a model or runtime error.

## `ResourceLimitError.__init__`

[定义位置](../../../manipulation/resource_admission.py#L46)。

输入签名：

```python
def __init__(self, resource, required_bytes, limit_bytes, *, description='Resource capacity exceeded'):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
self.resource
self.required_bytes
self.limit_bytes
```

写入目标：

```python
self.resource
self.required_bytes
self.limit_bytes
```

直接调用符号：

```python
str
int
super().__init__
super
```

## `require_host_memory`

[定义位置](../../../manipulation/resource_admission.py#L54)。

输入签名：

```python
def require_host_memory(new_bytes, *, pending_existing_bytes, max_rss_bytes):
```

实现说明：

> Admit new host storage while honoring every live lazy reservation.
>
> RSS already contains allocated arrays, native shapes and tree nodes. Add
> only the proposed allocation and still-unmaterialized storage promised to
> existing geometry. In particular, do not subtract a capacity estimate from
> RSS or charge every resident shape twice. The caller counts shared geometry
> once by identity before supplying ``pending_existing_bytes``.

显式输出分支：

无显式条目。

读取的对象属性：

```python
psutil.Process
process.create_time
current.pid
parent.pid
parent.create_time
current.parents
process.memory_info().rss
process.memory_info
process.children
child.memory_info().rss
child.memory_info
psutil.NoSuchProcess
psutil.virtual_memory().available
psutil.virtual_memory
```

写入目标：

```python
name
value
process
root_pid
created
current
parent
rss
child
available
additional
required
```

直接调用符号：

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

[定义位置](../../../manipulation/resource_admission.py#L107)。

输入签名：

```python
def require_device_memory(torch, device, *, native_bytes, torch_bytes=0, max_device_used_bytes):
```

实现说明：

> Admit new buffers against current whole-device usage and free capacity.
>
> ``native_bytes`` belongs to Mapper or other native allocators, which cannot
> reuse pages held by Torch's caching allocator. Only ``torch_bytes`` may use
> this process's reserved but currently unallocated Torch pages. All existing
> fields, CUDA contexts, cached pages and other applications already appear
> in driver-reported device usage; none is added again as a new allocation.
>
> This is a conservative capacity estimate at the allocation boundary, not
> an allocator lock: concurrent allocations and fragmentation can still make
> a later native allocation fail. No cache is cleared or limit increased.

显式输出分支：

```python
new_bytes
```

读取的对象属性：

```python
torch.cuda.mem_get_info
torch.cuda
torch.cuda.memory_reserved
torch.cuda.memory_allocated
```

写入目标：

```python
name
value
free_bytes
total_bytes
reusable_torch_bytes
required_bytes
new_bytes
projected_used
```

直接调用符号：

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
