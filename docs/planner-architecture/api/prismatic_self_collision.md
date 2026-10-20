# `manipulation/prismatic_self_collision.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../03-attachment-contact-and-hand.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `PreparedPrismaticSelfCollision`

[定义位置](../../../manipulation/prismatic_self_collision.py#L16)。

```python
class PreparedPrismaticSelfCollision:
    ...
    ...
```

该类型的直接方法：`__init__`、`filter_pairs`、`components`、`storage_nbytes`、`record`、`validate_record`。

## `PreparedPrismaticSelfCollision.__init__`

[定义位置](../../../manipulation/prismatic_self_collision.py#L31)。

输入签名：

```python
def PreparedPrismaticSelfCollision.__init__(self, geometry, *, joint_names, sphere_indices, local_spheres, sphere_joint_translation, collision_pairs):
```

直接读取的对象属性：`self.joint_names`、`self.MAX_GEOMETRIES`、`self.MAX_VERTICES`、`self.MAX_AXES`、`self.pair_geometry_ids`、`self.replaced_pairs`、`self.gaps`、`self.coefficients`、`self.axis_pairs`、`self.moving_pairs`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.joint_names`、`self.geometry`、`self.pair_geometry_ids`、`self.replaced_pairs`、`self.gaps`、`self.coefficients`、`self.axis_pairs`、`self.moving_pairs`、`self._device`、`value.flags.writeable`。

直接调用：

```python
tuple
len
isinstance
ValueError
enumerate
sorted
map
np.asarray(collision_pairs).reshape
np.asarray
set
any
type
np.isfinite(vertices).all
np.isfinite
np.isfinite(mapping).all
np.array_equal
np.broadcast_to
ConvexHull
np.unique
np.sort
np.concatenate
np.linalg.norm
rows.append
dict
seen.extend
ids.add
deepcopy
list
combinations
product
np.cross(first['edges'][:, None], second['edges'][None]).reshape
np.cross
gaps.append
np.column_stack
b.min
a.max
a.min
b.max
coefficients.append
axis_pairs.extend
moving.append
bool
np.any
self.pair_geometry_ids.append
self.replaced_pairs.extend
np.empty
```

## `PreparedPrismaticSelfCollision.filter_pairs`

[定义位置](../../../manipulation/prismatic_self_collision.py#L115)。

输入签名：

```python
def PreparedPrismaticSelfCollision.filter_pairs(self, pairs, padding):
```

直接读取的对象属性：`self.replaced_pairs`。

输出分支的源码表达式：

```python
[list(map(int, pair)) for pair in pairs if tuple(sorted(map(int, pair))) not in replaced]
```

直接调用：

```python
np.asarray(pairs, dtype=np.int64).reshape
np.asarray
tuple
sorted
map
set
ValueError
np.any
list
```

## `PreparedPrismaticSelfCollision.components`

[定义位置](../../../manipulation/prismatic_self_collision.py#L127)。

输入签名：

```python
def PreparedPrismaticSelfCollision.components(self, state):
```

直接读取的对象属性：`self.pair_geometry_ids`、`self._device`、`self.coefficients`、`self.gaps`、`self.axis_pairs`、`self.moving_pairs`。

输出分支的源码表达式：

```python
(zero, zero)
(torch.where(conditional, 0.0, residual).sum(-1), torch.where(conditional, residual, 0.0).sum(-1))
```

直接写入的状态或数组位置：`self._device[key]`。

直接调用：

```python
len
coordinates.new_zeros
torch.tensor
self.coefficients.copy
self.gaps.copy
self.axis_pairs.copy
self.moving_pairs.copy
coefficient.abs
torch.maximum
coordinates.new_full
separation.scatter_reduce
index.expand_as
torch.relu
torch.where(conditional, 0.0, residual).sum
torch.where
torch.where(conditional, residual, 0.0).sum
```

## `PreparedPrismaticSelfCollision.storage_nbytes`

[定义位置](../../../manipulation/prismatic_self_collision.py#L154)。

输入签名：

```python
@property
def PreparedPrismaticSelfCollision.storage_nbytes(self):
```

直接读取的对象属性：`self.gaps`、`self.coefficients`、`self.axis_pairs`、`self.moving_pairs`。

输出分支的源码表达式：

```python
sum((value.nbytes for value in (self.gaps, self.coefficients, self.axis_pairs, self.moving_pairs)))
```

直接调用：

```python
sum
```

## `PreparedPrismaticSelfCollision.record`

[定义位置](../../../manipulation/prismatic_self_collision.py#L157)。

输入签名：

```python
def PreparedPrismaticSelfCollision.record(self):
```

直接读取的对象属性：`self.geometry`、`self.pair_geometry_ids`、`self.replaced_pairs`、`self.gaps`、`self.coefficients`、`self.axis_pairs`、`self.moving_pairs`。

输出分支的源码表达式：

```python
{'kind': 'prepared_prismatic_self_v1', 'geometry': deepcopy(self.geometry), 'pair_geometry_ids': [list(pair) for pair in self.pair_geometry_ids], 'replaced_pairs': [list(pair) for pair in self.replaced_pairs], 'gaps': self.gaps.tolist(), 'coefficients': self.coefficients.tolist(), 'axis_pairs': self.axis_pairs.tolist(), 'moving_pairs': self.moving_pairs.tolist()}
```

直接调用：

```python
deepcopy
list
self.gaps.tolist
self.coefficients.tolist
self.axis_pairs.tolist
self.moving_pairs.tolist
```

## `PreparedPrismaticSelfCollision.validate_record`

[定义位置](../../../manipulation/prismatic_self_collision.py#L164)。

输入签名：

```python
def PreparedPrismaticSelfCollision.validate_record(self, record):
```

直接读取的对象属性：`self.record`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
self.record
ValueError
```
