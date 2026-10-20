# `manipulation/native_owner_geometry.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../03-attachment-contact-and-hand.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `bind_native_owner_geometry`

[定义位置](../../../manipulation/native_owner_geometry.py#L15)。

输入签名：

```python
def bind_native_owner_geometry(owner):
```

输出分支的源码表达式：

```python
(geometry, sphere_geometry, groups)
```

直接写入的状态或数组位置：`pose[:3, :3]`、`pose[:3, 3]`、`geometry[name]`、`groups[name]`。

直接调用：

```python
hasattr
ValueError
np.asarray
list
product
row.get
isinstance
set
name.strip
binding['source'].strip
len
any
type
math.prod
sphere_geometry.keys
np.eye
Rotation.from_quat(xyz_wxyz[[4, 5, 6, 3]]).as_matrix
Rotation.from_quat
range
np.any
np.linalg.norm(corners - balls[:, None, :3], axis=-1).max
np.linalg.norm
frozenset
sphere_geometry.update
```

## `check_native_owner_material`

[定义位置](../../../manipulation/native_owner_geometry.py#L64)。

输入签名：

```python
def check_native_owner_material(rule, native):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
np.asarray
list
product
np.concatenate
np.finfo
max
float
np.abs(body).max
np.abs
np.linalg.norm
len
distances.min(0).max
distances.min
distances.min(1).max
ValueError
```
