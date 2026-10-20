# `manipulation/initial_support.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../03-attachment-contact-and-hand.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `InitialSupportRelation`

[定义位置](../../../manipulation/initial_support.py#L24)。

```python
class InitialSupportRelation:
    relation_id: str
    actor: str
    target_owner: str
    interval: tuple
    plane_origin: object
    plane_normal: object
    source: dict
```

该类型的直接方法：`__post_init__`、`record`。

## `InitialSupportRelation.__post_init__`

[定义位置](../../../manipulation/initial_support.py#L41)。

输入签名：

```python
def InitialSupportRelation.__post_init__(self):
```

直接读取的对象属性：`self.actor`、`self.target_owner`、`self.interval`、`self.plane_origin`、`self.plane_normal`、`self.source`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.interval`、`self.plane_origin`、`self.plane_normal`、`self.source`。

直接调用：

```python
_text
getattr
ValueError
_valid_interval
_array
np.isclose
np.linalg.norm
isinstance
evidence.get
any
value.strip
evidence[key].strip
deepcopy
```

## `InitialSupportRelation.record`

[定义位置](../../../manipulation/initial_support.py#L64)。

输入签名：

```python
def InitialSupportRelation.record(self):
```

输出分支的源码表达式：

```python
row
```

直接写入的状态或数组位置：`row[key]`、`row['interval']`。

直接调用：

```python
asdict
row[key].tolist
list
```

## `_initial_pose`

[定义位置](../../../manipulation/initial_support.py#L72)。

输入签名：

```python
def _initial_pose(owner):
```

输出分支的源码表达式：

```python
pose
```

直接写入的状态或数组位置：`pose[:3, 3]`、`pose[:3, :3]`。

直接调用：

```python
owner.pose_path.sample
torch.tensor
np.eye
position.detach().cpu().numpy
position.detach().cpu
position.detach
Rotation.from_quat(quaternion.detach().cpu().numpy()[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
quaternion.detach().cpu().numpy
quaternion.detach().cpu
quaternion.detach
```

## `PreparedInitialSupport`

[定义位置](../../../manipulation/initial_support.py#L80)。

```python
class PreparedInitialSupport:
    ...
    ...
```

该类型的直接方法：`__init__`、`_select_components`、`components`、`collision_values`、`record`。

## `PreparedInitialSupport.__init__`

[定义位置](../../../manipulation/initial_support.py#L90)。

输入签名：

```python
def PreparedInitialSupport.__init__(self, relation, actor, target, numerical_tolerance, *, record=None):
```

直接读取的对象属性：`self.actor_vertices`、`self.initial_height`、`self._select_components`、`record.get`、`self.baseline_height`、`self.numerical_tolerance`、`self.kind`、`self.selected_component_ids`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.relation`、`self.target_query`、`self.numerical_tolerance`、`self.actor_vertices`、`self.actor_vertices.flags.writeable`、`self.initial_height`、`self.baseline_height`、`self._device`、`self.kind`、`self.selected_component_ids`、`self.selected_component_ids.flags.writeable`、`self.query`。

直接调用：

```python
isinstance
ValueError
np.isfinite
any
bind_native_owner_geometry
float
np.concatenate
geometry.values
np.linalg.inv
_initial_pose
((entrance - relation.plane_origin) @ relation.plane_normal).min
min
self._select_components
len
target.query.source_geometry.distance_field
record.get
relation.record
decode_array
np.any
np.diff
np.array_equal
np.array
FrozenESDF
```

## `PreparedInitialSupport._select_components`

[定义位置](../../../manipulation/initial_support.py#L151)。

输入签名：

```python
def PreparedInitialSupport._select_components(self, geometry, mapping, *, candidates=None):
```

直接读取的对象属性：`self.target_query`、`result.min_distance`。

输出分支的源码表达式：

```python
np.asarray(selected, dtype=np.int64)
```

直接写入的状态或数组位置：`request.enable_signed_distance`。

直接调用：

```python
geometry.values
np.asarray
boxes.append
coal.Box
coal.Transform3s
points.min
points.max
coal.DistanceRequest
max
float
np.unique
np.concatenate
source.component_indices
source._vertices
int
vertices.min
vertices.max
np.any
source.collision_component
np.eye
coal.DistanceResult
coal.distance
np.isfinite
ValueError
selected.append
```

## `PreparedInitialSupport.components`

[定义位置](../../../manipulation/initial_support.py#L188)。

输入签名：

```python
def PreparedInitialSupport.components(self, actor_pose, target_pose, progress):
```

直接读取的对象属性：`self._device`、`self.actor_vertices`、`self.relation`、`self.baseline_height`、`self.numerical_tolerance`。

输出分支的源码表达式：

```python
(active, residual)
```

直接写入的状态或数组位置：`self._device[key]`。

直接调用：

```python
tuple
torch.tensor
quaternion_rotate
quaternion.unsqueeze
position.unsqueeze
quaternion_conjugate(target_pose[1]).unsqueeze
quaternion_conjugate
target_pose[0].unsqueeze
((local - origin) * normal).sum(-1).amin
((local - origin) * normal).sum
torch.where
torch.relu
```

## `PreparedInitialSupport.collision_values`

[定义位置](../../../manipulation/initial_support.py#L205)。

输入签名：

```python
def PreparedInitialSupport.collision_values(self, spheres, ordinary, active):
```

直接读取的对象属性：`self.kind`、`self.query`。

输出分支的源码表达式：

```python
torch.where(active.unsqueeze(-1), values, ordinary)
```

直接调用：

```python
torch.zeros_like
self.query
torch.where
active.unsqueeze
```

## `PreparedInitialSupport.record`

[定义位置](../../../manipulation/initial_support.py#L210)。

输入签名：

```python
def PreparedInitialSupport.record(self):
```

直接读取的对象属性：`self.relation`、`self.initial_height`、`self.baseline_height`、`self.numerical_tolerance`、`self.selected_component_ids`、`self.kind`、`self.query`。

输出分支的源码表达式：

```python
{'model': 'initial_observed_support_v1', 'relation': self.relation.record(), 'initial_height_m': self.initial_height, 'baseline_height_m': self.baseline_height, 'numerical_position_tolerance_m': self.numerical_tolerance, 'selected_component_ids': encode_array(self.selected_component_ids), 'kind': self.kind, **({'distance': encode_array(self.query.distance)} if self.query is not None else {})}
```

直接调用：

```python
self.relation.record
encode_array
```
