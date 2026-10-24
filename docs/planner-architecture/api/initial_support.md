# `manipulation/initial_support.py` 实现契约

代码基线：`be44b54`。
职责和数据语义见[组件说明](../03-attachment-contact-and-hand.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `InitialSupportRelation`

[定义位置](../../../manipulation/initial_support.py#L20)。

```python
class InitialSupportRelation():
```

类字段：

```python
relation_id: str
actor: str
target_owner: str
interval: tuple
plane_origin: object
plane_normal: object
source: dict
```

实现说明：

> Declared current actor and observed support-plane layer.
>
> The target owner must be the separately observed support-plane layer, not
> an arbitrary compound obstacle relabelled as support. Its finite allowed
> material is established by entrance observed-field intersection, not a rectangle
> copied from a future placement feature.

## `InitialSupportRelation.__post_init__`

[定义位置](../../../manipulation/initial_support.py#L37)。

输入签名：

```python
def __post_init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.actor
self.target_owner
self.interval
self.plane_origin
self.plane_normal
np.isclose
np.linalg.norm
np.linalg
self.source
evidence.get
value.strip
evidence[key].strip
```

写入目标：

```python
name
self.interval
self.plane_origin
self.plane_normal
evidence
value
key
self.source
```

直接调用符号：

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

[定义位置](../../../manipulation/initial_support.py#L60)。

输入签名：

```python
def record(self):
```

显式输出分支：

```python
row
```

读取的对象属性：

```python
row[key].tolist
```

写入目标：

```python
row
key
row[key]
row['interval']
```

直接调用符号：

```python
asdict
row[key].tolist
list
```

## `_initial_pose`

[定义位置](../../../manipulation/initial_support.py#L68)。

输入签名：

```python
def _initial_pose(owner):
```

显式输出分支：

```python
pose
```

读取的对象属性：

```python
owner.pose_path.sample
owner.pose_path
torch.tensor
torch.float64
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

写入目标：

```python
position
quaternion
pose
pose[:3, 3]
pose[:3, :3]
```

直接调用符号：

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

[定义位置](../../../manipulation/initial_support.py#L76)。

```python
class PreparedInitialSupport():
```

类字段：

无显式条目。

实现说明：

> Freeze entrance-touched cells and bound the whole observed actor height.
>
> A component is the nonpositive trilinear set within one fully supported
> source cell. Only actual entrance intersection selects it. Adjacent cells
> retain the ordinary nonpenetration query, including at permission
> boundaries. Selected material is identified by its original cell.

## `PreparedInitialSupport.__init__`

[定义位置](../../../manipulation/initial_support.py#L85)。

输入签名：

```python
def __init__(self, relation, actor, target, numerical_tolerance):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
actor.owner_id
relation.actor
target.owner_id
relation.target_owner
actor.moving
target.moving
actor.query
target.query
np.isfinite
item.interval
relation.interval
actor.attachments
np.linalg.inv
np.linalg
relation.plane_origin
relation.plane_normal
self.numerical_tolerance
actor.query.bounds
target.query.root
target.query.nodes
pending.pop
node.lower
node.upper
node.children
pending.extend
target.query._cell_values[node.cell].reshape
target.query._cell_values
node.cell
np.ones
target.query.cell_origins
target.query.voxel_size
np.eye
self.selected_cells.append
self.selected_cells
self.unresolved_cells.append
self.unresolved_cells
```

写入目标：

```python
item
self.relation
self.actor_field
self.target_field
self.numerical_tolerance
mapping
self.initial_height
self.selected_cells
self.unresolved_cells
actor_bounds
pending
node
i
values
cell
result
```

直接调用符号：

```python
isinstance
ValueError
np.isfinite
any
float
np.linalg.inv
_initial_pose
query_minimum_height
world_bounds
pending.pop
box_distance
pending.extend
target.query._cell_values[node.cell].reshape
ObservedField
np.ones
query_distance
np.eye
self.selected_cells.append
self.unresolved_cells.append
tuple
sorted
```

## `PreparedInitialSupport.permission`

[定义位置](../../../manipulation/initial_support.py#L124)。

输入签名：

```python
def permission(self, target_matrix, *, additional=None):
```

实现说明：

> Remove selected original cell IDs from this pair's second domain.

显式输出分支：

```python
PairPermission(regions, excluded_second_cells=excluded)
```

读取的对象属性：

```python
additional.regions
self.selected_cells
excluded.update
additional.excluded_second_cells
```

写入目标：

```python
regions
excluded
```

直接调用符号：

```python
list
set
excluded.update
PairPermission
```

## `PreparedInitialSupport.components`

[定义位置](../../../manipulation/initial_support.py#L131)。

输入签名：

```python
def components(self, actor_pose, target_pose, progress, *, check_request=None):
```

实现说明：

> Compare height intervals; expose unresolved evidence independently.

显式输出分支：

```python
{f'initial_support/{self.relation.relation_id}': torch.stack(residuals).reshape(progress.shape), f'unresolved/initial_support/{self.relation.relation_id}': torch.stack(unresolved).reshape(progress.shape)}
```

读取的对象属性：

```python
self.initial_height
np.ndindex
progress.shape
actor_pose[0][index].sum
target_pose[0][index].sum
self.relation.interval
self.relation
progress[index].detach
np.linalg.inv
np.linalg
self.actor_field
self.relation.plane_origin
self.relation.plane_normal
self.numerical_tolerance
progress.new_tensor
((relative - progress.new_tensor(self.relation.plane_origin)) * progress.new_tensor(self.relation.plane_normal)).sum
direction.detach
residuals.append
unresolved.append
self.relation.relation_id
torch.stack(residuals).reshape
torch.stack
torch.stack(unresolved).reshape
```

写入目标：

```python
actor_matrices
target_matrices
residuals
unresolved
initial
index
zero
active
value
uncertain
mapping
height
baseline_lower
baseline_upper
certain
violation
magnitude
local
world
relative
direction
```

直接调用符号：

```python
_matrices
np.ndindex
tuple
actor_pose[0][index].sum
target_pose[0][index].sum
float
progress[index].detach
np.linalg.inv
query_minimum_height
min
progress.new_tensor
quaternion_rotate
quaternion_conjugate
((relative - progress.new_tensor(self.relation.plane_origin)) * progress.new_tensor(self.relation.plane_normal)).sum
direction.detach
residuals.append
unresolved.append
torch.stack(residuals).reshape
torch.stack
torch.stack(unresolved).reshape
```

## `PreparedInitialSupport.record`

[定义位置](../../../manipulation/initial_support.py#L165)。

输入签名：

```python
def record(self):
```

显式输出分支：

```python
dict(model='initial_observed_support_v2', relation=self.relation.record(), initial_height=self.initial_height, numerical_position_tolerance_m=self.numerical_tolerance, selected_cells=list(self.selected_cells), unresolved_cells=list(self.unresolved_cells), component_semantics='nonpositive set in each fully supported source trilinear cell')
```

读取的对象属性：

```python
self.relation.record
self.relation
self.initial_height
self.numerical_tolerance
self.selected_cells
self.unresolved_cells
```

写入目标：

无显式条目。

直接调用符号：

```python
dict
self.relation.record
list
```
