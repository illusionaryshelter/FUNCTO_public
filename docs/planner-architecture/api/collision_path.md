# `manipulation/collision_path.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_collision_path_chunk_size`

[定义位置](../../../manipulation/collision_path.py#L18)。

输入签名：

```python
def _collision_path_chunk_size(states, spheres):
```

输出分支的源码表达式：

```python
min(spheres, max(1, _COLLISION_PATH_SAMPLE_ELEMENTS // states)) if spheres else 0
```

直接调用：

```python
min
max
```

## `_sphere_chunk_objective`

[定义位置](../../../manipulation/collision_path.py#L23)。

输入签名：

```python
def _sphere_chunk_objective(c, x, radius, valid, edges, order, begin, end, *mode_keys):
```

输出分支的源码表达式：

```python
(peak + arc / radius[begin:end]).sum(dim=-1)
```

直接调用：

```python
torch.where
ordered
values.amax
edges[..., None].expand
torch.linalg.vector_norm
((first_cost + second_cost) * 0.5 * length).sum
(peak + arc / radius[begin:end]).sum
```

## `_sphere_chunk_objective.ordered`

[定义位置](../../../manipulation/collision_path.py#L30)。

输入签名：

```python
def _sphere_chunk_objective.ordered(value):
```

输出分支的源码表达式：

```python
value
value.gather(1, indices)
```

直接调用：

```python
order.reshape(*order.shape, *(1,) * (value.ndim - 2)).expand_as
order.reshape
value.gather
```

## `SphereCollisionPath`

[定义位置](../../../manipulation/collision_path.py#L56)。

```python
class SphereCollisionPath:
    residual: torch.Tensor
    centres: torch.Tensor
    radii: torch.Tensor
    mode_keys: tuple[torch.Tensor, ...] = ()
```

## `collision_path_objective`

[定义位置](../../../manipulation/collision_path.py#L73)。

输入签名：

```python
def collision_path_objective(paths: Mapping[str, SphereCollisionPath], times: torch.Tensor, valid: torch.Tensor, *, order: torch.Tensor | None=None) -> torch.Tensor:
```

输出分支的源码表达式：

```python
total
```

直接调用：

```python
times.is_floating_point
ValueError
times.gather
valid.gather
times.new_zeros
paths.values
any
_collision_path_chunk_size
torch.is_grad_enabled
range
min
checkpoint
_sphere_chunk_objective
```

## `collision_path_workspace_allocation`

[定义位置](../../../manipulation/collision_path.py#L139)。

输入签名：

```python
def collision_path_workspace_allocation(batch, horizon, *, geometry_spheres, path_spheres, max_stream_spheres, mode_bytes_per_state):
```

输出分支的源码表达式：

```python
{'cpu_shared': int(geometry_spheres) * 48, 'gpu_shared': int(geometry_spheres) * 28, 'cpu_live': scalar_elements * 8 + indices_and_modes, 'gpu_live': scalar_elements * 4 + indices_and_modes}
```

直接调用：

```python
int
_collision_path_chunk_size
```
