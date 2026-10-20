# `manipulation/functional_augmented_lagrangian.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `FunctionalAugmentedLagrangian`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L15)。

```python
class FunctionalAugmentedLagrangian:
    ...
    ...
```

该类型的直接方法：`__init__`、`bind`、`reset`、`_residuals`、`terms`、`update`、`snapshot`、`report`。

## `FunctionalAugmentedLagrangian.__init__`

[定义位置](../../../manipulation/functional_augmented_lagrangian.py#L29)。

输入签名：

```python
def FunctionalAugmentedLagrangian.__init__(self, keys):
```

直接读取的对象属性：`self.keys`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.keys`、`self._multiplier_buffer`、`self._penalty_buffer`、`self.num_seeds`。

直接调用：

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
def FunctionalAugmentedLagrangian.bind(self, like, num_seeds=None):
```

直接读取的对象属性：`self.num_seeds`、`self._multiplier_buffer`、`self.keys`、`self.reset`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self.num_seeds`、`self._multiplier_buffer`、`self._penalty_buffer`、`self._previous_residual`、`self._updates`、`self._penalty_capped`。

直接调用：

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
def FunctionalAugmentedLagrangian.reset(self):
```

直接读取的对象属性：`self._multiplier_buffer`、`self._penalty_buffer`、`self.initial_penalty`、`self._previous_residual`、`self._updates`、`self._penalty_capped`。

输出分支的源码表达式：

```python
None
```

直接调用：

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
def FunctionalAugmentedLagrangian._residuals(self, signed):
```

直接读取的对象属性：`self.keys`、`self._multiplier_buffer`、`self.num_seeds`。

输出分支的源码表达式：

```python
self._multiplier_buffer
torch.stack(values, dim=-1)
```

直接调用：

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
def FunctionalAugmentedLagrangian.terms(self, signed):
```

直接读取的对象属性：`self.keys`、`self._residuals`、`self.num_seeds`、`self._penalty_buffer`、`self._multiplier_buffer`。

输出分支的源码表达式：

```python
{}
{key: costs[:, index] for (index, key) in enumerate(self.keys)}
```

直接调用：

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
@torch.no_grad()
def FunctionalAugmentedLagrangian.update(self, signed):
```

直接读取的对象属性：`self._residuals`、`self._multiplier_buffer`、`self.keys`、`self.num_seeds`、`self._penalty_buffer`、`self.required_reduction`、`self._previous_residual`、`self.penalty_growth`、`self.maximum_penalty`、`self._penalty_capped`、`self._updates`。

输出分支的源码表达式：

```python
torch.zeros(self.num_seeds, dtype=torch.bool, device=residual.device)
changed
```

直接调用：

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
def FunctionalAugmentedLagrangian.snapshot(self):
```

直接读取的对象属性：`self._multiplier_buffer`、`self.keys`、`self._penalty_buffer`、`self._previous_residual`、`self._updates`、`self._penalty_capped`、`self.maximum_penalty`、`self.num_seeds`。

输出分支的源码表达式：

```python
{'constraint_keys': list(self.keys), 'seeds': []}
{'constraint_keys': list(self.keys), 'maximum_penalty': self.maximum_penalty, 'seeds': [{'seed_index': index, 'multipliers': dict(zip(self.keys, multipliers[index])), 'penalty': penalties[index], 'outer_updates': updates[index], 'previous_complementarity_residual': previous[index] if updates[index] else None, 'penalty_capped': capped[index]} for index in range(self.num_seeds)]}
```

直接调用：

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
@property
def FunctionalAugmentedLagrangian.report(self):
```

直接读取的对象属性：`self.snapshot`。

输出分支的源码表达式：

```python
self.snapshot()
```

直接调用：

```python
self.snapshot
```
