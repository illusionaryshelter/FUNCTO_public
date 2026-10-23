# `manipulation/functional_augmented_lagrangian.py` 实现契约

代码基线：`82a3d66；包含本次工作区已实现改动，待统一实现提交后更新`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `FunctionalAugmentedLagrangian`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L15)。

```python
class FunctionalAugmentedLagrangian():
```

类字段：

```python
initial_penalty = 2.0
maximum_penalty = 1000000.0
penalty_growth = 10.0
required_reduction = 0.5
```

实现说明：

> One multiplier vector and penalty per independently optimized seed.
>
> ``bind`` declares the seed layout before evaluation/capture. Rollout rows
> are seed-major, then line-search-particle-major, exactly as GradientOptCore
> reshapes actions. ``update`` requires one actual continuation point per
> seed. It cannot average particles or implicitly select a different seed.

## `FunctionalAugmentedLagrangian.__init__`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L29)。

输入签名：

```python
def __init__(self, keys):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.keys
```

写入目标：

```python
self.keys
key
self._multiplier_buffer
self._penalty_buffer
self.num_seeds
```

直接调用符号：

```python
tuple
any
isinstance
len
set
ValueError
```

## `FunctionalAugmentedLagrangian.bind`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L38)。

输入签名：

```python
def bind(self, like, num_seeds=None):
```

实现说明：

> Keep addresses for an unchanged layout; a changed layout starts fresh.
>
> Manager rebinding only supplies dtype/device and therefore retains the
> existing seed count. The core explicitly declares its count at each
> solve entry, before either initial or ordinary rollout evaluation.

显式输出分支：

```python
None
```

读取的对象属性：

```python
like.is_floating_point
self.num_seeds
self._multiplier_buffer
self._multiplier_buffer.device
like.device
self._multiplier_buffer.dtype
like.dtype
like.new_zeros
self.keys
like.new_empty
torch.zeros
torch.int64
torch.bool
self.reset
```

写入目标：

```python
count
self.num_seeds
self._multiplier_buffer
self._penalty_buffer
self._previous_residual
self._updates
self._penalty_capped
```

直接调用符号：

```python
like.is_floating_point
ValueError
isinstance
like.new_zeros
len
like.new_empty
torch.zeros
self.reset
```

## `FunctionalAugmentedLagrangian.reset`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L62)。

输入签名：

```python
def reset(self):
```

实现说明：

> Reset an independent solve in place, preserving captured addresses.

显式输出分支：

```python
None
```

读取的对象属性：

```python
self._multiplier_buffer
self._multiplier_buffer.zero_
self._penalty_buffer.fill_
self._penalty_buffer
self.initial_penalty
self._previous_residual.fill_
self._previous_residual
self._updates.zero_
self._updates
self._penalty_capped.zero_
self._penalty_capped
```

写入目标：

无显式条目。

直接调用符号：

```python
self._multiplier_buffer.zero_
self._penalty_buffer.fill_
self._previous_residual.fill_
float
self._updates.zero_
self._penalty_capped.zero_
```

## `FunctionalAugmentedLagrangian._residuals`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L72)。

输入签名：

```python
def _residuals(self, signed):
```

显式输出分支：

```python
self._multiplier_buffer
torch.stack(values, dim=-1)
```

读取的对象属性：

```python
self.keys
self._multiplier_buffer
like.ndim
like.numel
self.num_seeds
like.device
self._multiplier_buffer.device
like.dtype
self._multiplier_buffer.dtype
value.shape
like.shape
value.device
value.dtype
torch.stack
```

写入目标：

```python
values
key
like
value
```

直接调用符号：

```python
set
ValueError
RuntimeError
like.numel
any
torch.stack
```

## `FunctionalAugmentedLagrangian.terms`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L89)。

输入签名：

```python
def terms(self, signed):
```

实现说明：

> Nonnegative PHR terms, broadcasting only within each seed's particles.
>
> Omitting the x-independent ``-lambda**2/(2*rho)`` preserves gradients
> and minimizers while retaining the native tracker's nonnegative cost
> convention. Such costs cannot be compared across parameter updates.

显式输出分支：

```python
{}
{key: costs[:, index] for (index, key) in enumerate(self.keys)}
```

读取的对象属性：

```python
self.keys
self._residuals
residual.shape
self.num_seeds
residual.view
self._penalty_buffer
torch.relu
self._multiplier_buffer
(shifted.square() / (2 * penalty)).reshape
shifted.square
```

写入目标：

```python
residual
particles
penalty
shifted
costs
index
key
```

直接调用符号：

```python
self._residuals
residual.view
len
torch.relu
(shifted.square() / (2 * penalty)).reshape
shifted.square
enumerate
```

## `FunctionalAugmentedLagrangian.update`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L107)。

输入签名：

```python
def update(self, signed):
```

实现说明：

> Return changed seed rows after one block's inexact PHR update.
>
> ``max(abs(max(g, -lambda/rho)))`` measures feasibility/complementarity
> progress, not stationarity or full KKT convergence. The first update
> establishes the comparison value. Finite steps may approach a boundary
> from its infeasible side indefinitely; acceptance stays outside here.

显式输出分支：

```python
torch.zeros(self.num_seeds, dtype=torch.bool, device=residual.device)
changed
```

读取的对象属性：

```python
self._residuals
residual.shape
self._multiplier_buffer.shape
self._multiplier_buffer
torch.isfinite(residual).all
torch.isfinite
self.keys
torch.zeros
self.num_seeds
torch.bool
residual.device
torch.maximum(residual, -self._multiplier_buffer / self._penalty_buffer[:, None]).abs().amax
torch.maximum(residual, -self._multiplier_buffer / self._penalty_buffer[:, None]).abs
torch.maximum
self._penalty_buffer
torch.relu
torch.isfinite(multipliers).all
self.required_reduction
self._previous_residual
self.penalty_growth
torch.where
proposed_penalty.clamp_max
self.maximum_penalty
(multipliers != self._multiplier_buffer).any
self._penalty_capped.copy_
self._penalty_capped
self._multiplier_buffer.copy_
self._penalty_buffer.copy_
self._previous_residual.copy_
self._updates.add_
self._updates
torch.no_grad
```

写入目标：

```python
residual
measure
multipliers
increase
proposed_penalty
penalty
changed
```

直接调用符号：

```python
self._residuals
ValueError
bool
torch.isfinite(residual).all
torch.isfinite
torch.zeros
torch.maximum(residual, -self._multiplier_buffer / self._penalty_buffer[:, None]).abs().amax
torch.maximum(residual, -self._multiplier_buffer / self._penalty_buffer[:, None]).abs
torch.maximum
torch.relu
torch.isfinite(multipliers).all
torch.where
proposed_penalty.clamp_max
(multipliers != self._multiplier_buffer).any
self._penalty_capped.copy_
self._multiplier_buffer.copy_
self._penalty_buffer.copy_
self._previous_residual.copy_
self._updates.add_
torch.no_grad
```

## `FunctionalAugmentedLagrangian.snapshot`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L137)。

输入签名：

```python
def snapshot(self):
```

实现说明：

> Copy small per-seed diagnostics only at a completed solve boundary.

显式输出分支：

```python
{'constraint_keys': list(self.keys), 'seeds': []}
{'constraint_keys': list(self.keys), 'maximum_penalty': self.maximum_penalty, 'seeds': [{'seed_index': index, 'multipliers': dict(zip(self.keys, multipliers[index])), 'penalty': penalties[index], 'outer_updates': updates[index], 'previous_complementarity_residual': previous[index] if updates[index] else None, 'penalty_capped': capped[index]} for index in range(self.num_seeds)]}
```

读取的对象属性：

```python
self._multiplier_buffer
self.keys
self._multiplier_buffer.detach().cpu().tolist
self._multiplier_buffer.detach().cpu
self._multiplier_buffer.detach
self._penalty_buffer.detach().cpu().tolist
self._penalty_buffer.detach().cpu
self._penalty_buffer.detach
self._penalty_buffer
self._previous_residual.detach().cpu().tolist
self._previous_residual.detach().cpu
self._previous_residual.detach
self._previous_residual
self._updates.cpu().tolist
self._updates.cpu
self._updates
self._penalty_capped.cpu().tolist
self._penalty_capped.cpu
self._penalty_capped
self.maximum_penalty
self.num_seeds
```

写入目标：

```python
multipliers
penalties
previous
updates
capped
index
```

直接调用符号：

```python
list
self._multiplier_buffer.detach().cpu().tolist
self._multiplier_buffer.detach().cpu
self._multiplier_buffer.detach
self._penalty_buffer.detach().cpu().tolist
self._penalty_buffer.detach().cpu
self._penalty_buffer.detach
self._previous_residual.detach().cpu().tolist
self._previous_residual.detach().cpu
self._previous_residual.detach
self._updates.cpu().tolist
self._updates.cpu
self._penalty_capped.cpu().tolist
self._penalty_capped.cpu
dict
zip
range
```

## `FunctionalAugmentedLagrangian.report`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L153)。

输入签名：

```python
def report(self):
```

显式输出分支：

```python
self.snapshot()
```

读取的对象属性：

```python
self.snapshot
```

写入目标：

无显式条目。

直接调用符号：

```python
self.snapshot
```
