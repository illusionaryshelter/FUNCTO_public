# `manipulation/contact_geometry.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../03-attachment-contact-and-hand.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `convex_component_intersects_region`

[定义位置](../../../manipulation/contact_geometry.py#L23)。

输入签名：

```python
def convex_component_intersects_region(vertices, region):
```

直接读取的对象属性：`result.status`、`result.success`、`result.x`。

输出分支的源码表达式：

```python
{'completed': True, 'intersects': False, 'numerical_certificate': False}
{'completed': False, 'reason': 'component_region_query_incomplete', 'solver_status': int(result.status)}
{'completed': True, 'intersects': True, 'witness': (result.x @ vertices).tolist(), 'numerical_certificate': False}
```

直接调用：

```python
np.asarray
map
linprog
np.zeros
len
np.ones
int
(result.x @ vertices).tolist
```

## `_component_intersection_support`

[定义位置](../../../manipulation/contact_geometry.py#L44)。

输入签名：

```python
def _component_intersection_support(vertices, region, *, A_ub=None, b_ub=None, A_eq=None, b_eq=None, variable_bounds=(0.0, None)):
```

直接读取的对象属性：`result.status`、`result.success`、`result.x`。

输出分支的源码表达式：

```python
{'completed': True, 'contained': True, 'intersection_empty': True, 'numerical_certificate': False, 'method': 'complete_component_intersection_support'}
{'completed': False, 'contained': False, 'reason': 'component_intersection_support_incomplete', 'solver_status': int(result.status)}
{'completed': True, 'contained': False, 'intersection_empty': False, 'outside_witness': point.tolist(), 'maximum_constraint_excess_m': excess, 'numerical_certificate': False, 'method': 'complete_component_intersection_support'}
{'completed': True, 'contained': True, 'intersection_empty': False, 'maximum_constraint_excess_m': maximum, 'numerical_certificate': False, 'method': 'complete_component_intersection_support'}
```

直接调用：

```python
map
list
zip
np.zeros
linprog
int
float
np.all
np.asarray
np.any
min
np.min
max
point.tolist
```

## `convex_component_overlap_within_region`

[定义位置](../../../manipulation/contact_geometry.py#L84)。

输入签名：

```python
def convex_component_overlap_within_region(actor_halfspaces, component_vertices, region, actor_margin=0.0):
```

输出分支的源码表达式：

```python
_component_intersection_support(vertices, region, A_ub=normals @ vertices.T, b_ub=shifted, A_eq=np.ones((1, len(vertices))), b_eq=np.ones(1))
```

直接调用：

```python
np.asarray
map
float
np.linalg.norm
_component_intersection_support
np.ones
len
```

## `_component_support_planes`

[定义位置](../../../manipulation/contact_geometry.py#L99)。

输入签名：

```python
def _component_support_planes(vertices):
```

输出分支的源码表达式：

```python
(planes[:, :3], -planes[:, 3], rank)
(normals, bounds, rank)
```

直接调用：

```python
np.asarray
np.linalg.svd
np.finfo
max
float
np.abs(vertices).max
np.abs
int
np.count_nonzero
ConvexHull
np.empty
np.array
coordinate.max
coordinate.min
np.concatenate
```

## `component_component_overlap_within_region`

[定义位置](../../../manipulation/contact_geometry.py#L135)。

输入签名：

```python
def component_component_overlap_within_region(first_vertices, second_vertices, region, *, first_margin_m=0.0, second_margin_m=0.0):
```

输出分支的源码表达式：

```python
evidence
_component_intersection_support(objective_vertices, region, A_eq=equalities, b_eq=np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
```

直接写入的状态或数组位置：`evidence['margin_geometry']`、`equalities[0, :n]`、`equalities[1, n:]`、`equalities[2:, :n]`、`equalities[2:, n:]`。

直接调用：

```python
np.asarray
_component_support_planes
_component_intersection_support
np.eye
np.concatenate
np.linalg.norm
len
np.zeros
np.zeros_like
np.array
```

## `_component_region_slack`

[定义位置](../../../manipulation/contact_geometry.py#L165)。

输入签名：

```python
def _component_region_slack(vertices, normals, bounds):
```

直接读取的对象属性：`result.success`、`result.message`、`result.x`、`result.ineqlin`、`result.fun`。

输出分支的源码表达式：

```python
bounds.new_tensor(result.fun) + ((normals - normals.detach()) * normals.new_tensor(-dual[:, None] * point)).sum() + ((bounds - bounds.detach()) * bounds.new_tensor(dual)).sum()
```

直接调用：

```python
np.asarray
normals.detach().cpu().numpy
normals.detach().cpu
normals.detach
bounds.detach().cpu().numpy
bounds.detach().cpu
bounds.detach
linprog
np.zeros
len
np.ones
ValueError
bounds.new_tensor
((normals - normals.detach()) * normals.new_tensor(-dual[:, None] * point)).sum
normals.new_tensor
((bounds - bounds.detach()) * bounds.new_tensor(dual)).sum
```

## `region_slack`

[定义位置](../../../manipulation/contact_geometry.py#L184)。

输入签名：

```python
def region_slack(normals, bounds):
```

直接读取的对象属性：`result.success`、`result.message`、`result.ineqlin`、`result.fun`、`result.x`。

输出分支的源码表达式：

```python
value + ((normals - normals.detach()) * normals.new_tensor(np.asarray(gradient_a)).reshape(normals.shape)).sum((-1, -2)) + ((bounds - bounds.detach()) * bounds.new_tensor(np.asarray(gradient_b)).reshape(bounds.shape)).sum(-1)
```

直接调用：

```python
normals.detach().cpu().numpy().reshape
normals.detach().cpu().numpy
normals.detach().cpu
normals.detach
bounds.detach().cpu().numpy().reshape
bounds.detach().cpu().numpy
bounds.detach().cpu
bounds.detach
len
ValueError
zip
linprog
np.ones
np.asarray
values.append
gradient_a.append
gradient_b.append
bounds.new_tensor(values).reshape
bounds.new_tensor
((normals - normals.detach()) * normals.new_tensor(np.asarray(gradient_a)).reshape(normals.shape)).sum
normals.new_tensor(np.asarray(gradient_a)).reshape
normals.new_tensor
((bounds - bounds.detach()) * bounds.new_tensor(np.asarray(gradient_b)).reshape(bounds.shape)).sum
bounds.new_tensor(np.asarray(gradient_b)).reshape
```

## `query_geometry`

[定义位置](../../../manipulation/contact_geometry.py#L213)。

输入签名：

```python
def query_geometry(query):
```

输出分支的源码表达式：

```python
deepcopy(explicit)
{'solids': [], 'query_gaps': ['No single independent native scene model']}
{'solids': solids, 'query_gaps': gaps}
```

直接写入的状态或数组位置：`row['dimensions']`、`row['vertices']`。

直接调用：

```python
getattr
deepcopy
isinstance
len
gaps.append
list
shape.get_trimesh_mesh
np.asarray(mesh.vertices).tolist
np.asarray
solids.append
```

## `_bind_required_surfaces`

[定义位置](../../../manipulation/contact_geometry.py#L250)。

输入签名：

```python
def _bind_required_surfaces(target, surfaces):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`target.surface_components`、`target.surface_centres`、`target.surface_radii`、`target.surface_tree`、`target.maximum_surface_radius`。

直接调用：

```python
tuple
np.asarray([points.mean(0) for points in surfaces], float).reshape
np.asarray
points.mean
np.linalg.norm(points - centre, axis=1).max
np.linalg.norm
zip
cKDTree
float
target.surface_radii.max
```

## `_required_surface_contact`

[定义位置](../../../manipulation/contact_geometry.py#L259)。

输入签名：

```python
def _required_surface_contact(target, policy):
```

输出分支的源码表达式：

```python
output.reshape(shape)
output.index_fill(0, selected, 1.0).reshape(shape)
```

直接写入的状态或数组位置：`target.last_required_report`。

直接调用：

```python
policy['active'].reshape
torch.nonzero(active, as_tuple=False).flatten
torch.nonzero
policy['feature_points'].reshape
len
points.mean
torch.linalg.vector_norm(points - centre[:, None], dim=-1).amax
torch.linalg.vector_norm
output.reshape
output.index_fill(0, selected, 1.0).reshape
output.index_fill
selected.detach().cpu().numpy
selected.detach().cpu
selected.detach
centre[selected].detach().cpu().numpy
centre[selected].detach().cpu
centre[selected].detach
np.nextafter
radius[selected].detach().cpu().numpy
radius[selected].detach().cpu
radius[selected].detach
target.surface_tree.query_ball_point
int
lengths.sum
ValueError
policy['normals'].reshape
policy['bounds'].reshape
enumerate
_component_region_slack
values.append
torch.stack(slack).amin
torch.stack
target.surface_tree.query
centre.new_tensor
torch.relu
output.index_copy
```

## `ContactCellGeometry`

[定义位置](../../../manipulation/contact_geometry.py#L306)。

```python
class ContactCellGeometry:
    ...
    ...
```

该类型的直接方法：`__init__`、`complete`、`workspace_bytes`、`storage_bytes`、`required_contact`、`query`。

## `ContactCellGeometry.__init__`

[定义位置](../../../manipulation/contact_geometry.py#L322)。

输入签名：

```python
def ContactCellGeometry.__init__(self, geometry):
```

直接读取的对象属性：`self.gaps`、`self.centres`、`self.max_cells`、`self.halves`、`self.rotations`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.gaps`、`self.cell_half_diagonal_m`、`self.centres`、`self.halves`、`self.rotations`、`self.tree`、`self.maximum_cell_radius`、`self.last_report`、`self.last_required_report`。

直接调用：

```python
list
geometry.get
np.asarray
np.isfinite(pose).all
np.isfinite
np.isclose
np.linalg.norm
ValueError
Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix
Rotation.from_quat
np.isfinite(dimensions).all
np.any
np.zeros
self.gaps.append
centres.append
halves.append
rotations.append
np.broadcast_to(rotation, (len(local), 3, 3)).copy
np.broadcast_to
len
np.concatenate
np.empty
cKDTree
float
np.linalg.norm(self.halves, axis=1).max
zip
surfaces.extend
range
_bind_required_surfaces
```

## `ContactCellGeometry.complete`

[定义位置](../../../manipulation/contact_geometry.py#L361)。

输入签名：

```python
@property
def ContactCellGeometry.complete(self):
```

直接读取的对象属性：`self.gaps`。

输出分支的源码表达式：

```python
not self.gaps
```

## `ContactCellGeometry.workspace_bytes`

[定义位置](../../../manipulation/contact_geometry.py#L364)。

输入签名：

```python
def ContactCellGeometry.workspace_bytes(self):
```

直接读取的对象属性：`self.pair_chunk`。

输出分支的源码表达式：

```python
int(self.pair_chunk * 1600)
```

直接调用：

```python
int
```

## `ContactCellGeometry.storage_bytes`

[定义位置](../../../manipulation/contact_geometry.py#L368)。

输入签名：

```python
@property
def ContactCellGeometry.storage_bytes(self):
```

直接读取的对象属性：`self.centres`、`self.halves`、`self.rotations`、`self.surface_centres`、`self.surface_radii`、`self.surface_components`。

输出分支的源码表达式：

```python
int(self.centres.nbytes + self.halves.nbytes + self.rotations.nbytes + self.surface_centres.nbytes + self.surface_radii.nbytes + sum((vertices.nbytes for vertices in self.surface_components)))
```

直接调用：

```python
int
sum
```

## `ContactCellGeometry.required_contact`

[定义位置](../../../manipulation/contact_geometry.py#L373)。

输入签名：

```python
def ContactCellGeometry.required_contact(self, policy):
```

输出分支的源码表达式：

```python
_required_surface_contact(self, policy)
```

直接调用：

```python
_required_surface_contact
```

## `ContactCellGeometry.query`

[定义位置](../../../manipulation/contact_geometry.py#L377)。

输入签名：

```python
def ContactCellGeometry.query(self, spheres, policies):
```

直接读取的对象属性：`self.complete`、`self.maximum_cell_radius`、`self.tree`、`self.centres`、`self.max_candidate_pairs`、`self.max_candidate_plane_products`、`self.pair_chunk`、`self.halves`、`self.rotations`、`self.cell_half_diagonal_m`。

输出分支的源码表达式：

```python
(output.reshape(spheres.shape[:-1]), enabled.reshape(spheres.shape[:-1]))
```

直接写入的状态或数组位置：`enabled[:, list(policy['sphere_indices'])]`、`best_values[ib[winner]]`、`best_cells[ib[winner]]`、`best_allowances[ib[winner]]`、`self.last_report`。

直接调用：

```python
ValueError
math.prod
spheres.reshape
torch.zeros
list
policy['active'].reshape
torch.nonzero(enabled.reshape(-1), as_tuple=False).flatten
torch.nonzero
enabled.reshape
base.detach().cpu().numpy().astype
base.detach().cpu().numpy
base.detach().cpu
base.detach
len
np.isfinite(supplied).all
np.isfinite
np.any
np.nextafter
self.tree.query_ball_point
int
np.sum
max
min
np.repeat
np.arange
np.concatenate(neighbors).astype
np.concatenate
np.empty
np.zeros
np.full
chosen.detach().cpu().numpy
chosen.detach().cpu
chosen.detach
torch.no_grad
range
torch.as_tensor
spheres.new_tensor
torch.einsum
local.abs
torch.linalg.vector_norm
torch.relu
delta.amax(-1).clamp_max
delta.amax
torch.zeros_like
np.isin
policy['normal_valid'].reshape
policy['normals'].reshape
policy['bounds'].reshape
(torch.einsum('nki,nij->nkj', normals, rotation).abs() * half[:, None]).sum
torch.einsum('nki,nij->nkj', normals, rotation).abs
(torch.minimum(sphere_support, box_support) <= bounds).all
torch.minimum
torch.maximum
allowed.to
torch.relu(raw - allowance).cpu().numpy
torch.relu(raw - allowance).cpu
allowance.cpu().numpy
allowance.cpu
np.count_nonzero
np.lexsort
np.flatnonzero
output.index_copy
output.reshape
```

## `ObservedContactGeometry`

[定义位置](../../../manipulation/contact_geometry.py#L489)。

```python
class ObservedContactGeometry:
    ...
    ...
```

该类型的直接方法：`__init__`、`complete`、`storage_bytes`、`workspace_bytes`、`required_contact`、`_distance_model`、`_closest_points`、`query`。

## `ObservedContactGeometry.__init__`

[定义位置](../../../manipulation/contact_geometry.py#L504)。

输入签名：

```python
def ObservedContactGeometry.__init__(self, geometry):
```

直接读取的对象属性：`self.max_cells`、`self.gaps`、`self.centres`、`self.radii`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.gaps`、`self.vertices`、`self.negative`、`self.centres`、`self.radii`、`self.tree`、`self.maximum_cell_radius`、`self._distance_models`、`self.last_report`、`self.last_required_report`。

直接调用：

```python
list
geometry.get
np.asarray
np.isfinite(pose).all
np.isfinite
np.isclose
np.linalg.norm
ValueError
Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix
Rotation.from_quat
iter_source_components
components.append
negative.append
surfaces.append
len
np.isfinite(dimensions).all
np.any
surfaces.extend
range
self.gaps.append
tuple
np.asarray([points.mean(0) for points in components], float).reshape
points.mean
np.linalg.norm(points - centre, axis=1).max
zip
cKDTree
float
self.radii.max
_bind_required_surfaces
```

## `ObservedContactGeometry.complete`

[定义位置](../../../manipulation/contact_geometry.py#L552)。

输入签名：

```python
@property
def ObservedContactGeometry.complete(self):
```

直接读取的对象属性：`self.gaps`。

输出分支的源码表达式：

```python
not self.gaps
```

## `ObservedContactGeometry.storage_bytes`

[定义位置](../../../manipulation/contact_geometry.py#L556)。

输入签名：

```python
@property
def ObservedContactGeometry.storage_bytes(self):
```

直接读取的对象属性：`self._distance_models`、`self.centres`、`self.radii`、`self.negative`、`self.surface_centres`、`self.surface_radii`、`self.vertices`、`self.surface_components`。

输出分支的源码表达式：

```python
int(self.centres.nbytes + self.radii.nbytes + self.negative.nbytes + self.surface_centres.nbytes + self.surface_radii.nbytes + sum((points.nbytes for points in self.vertices)) + sum((points.nbytes for points in self.surface_components)) + model_bytes)
```

直接调用：

```python
sum
self._distance_models.values
model.values
isinstance
int
```

## `ObservedContactGeometry.workspace_bytes`

[定义位置](../../../manipulation/contact_geometry.py#L566)。

输入签名：

```python
def ObservedContactGeometry.workspace_bytes(self):
```

直接读取的对象属性：`self.pair_chunk`。

输出分支的源码表达式：

```python
int(self.pair_chunk * 4096)
```

直接调用：

```python
int
```

## `ObservedContactGeometry.required_contact`

[定义位置](../../../manipulation/contact_geometry.py#L571)。

输入签名：

```python
def ObservedContactGeometry.required_contact(self, policy):
```

输出分支的源码表达式：

```python
_required_surface_contact(self, policy)
```

直接调用：

```python
_required_surface_contact
```

## `ObservedContactGeometry._distance_model`

[定义位置](../../../manipulation/contact_geometry.py#L574)。

输入签名：

```python
def ObservedContactGeometry._distance_model(self, index):
```

直接读取的对象属性：`self._distance_models`、`self.vertices`。

输出分支的源码表达式：

```python
model
```

直接写入的状态或数组位置：`model['point']`、`model['endpoints']`、`model['triangles']`、`self._distance_models[index]`。

直接调用：

```python
self._distance_models.get
_component_support_planes
coordinate.argmin
coordinate.argmax
np.linalg.svd
ConvexHull
np.array
range
len
```

## `ObservedContactGeometry._closest_points`

[定义位置](../../../manipulation/contact_geometry.py#L598)。

输入签名：

```python
def ObservedContactGeometry._closest_points(self, indices, points):
```

直接读取的对象属性：`self._distance_model`、`self.negative`。

输出分支的源码表达式：

```python
(nearest, signs)
```

直接写入的状态或数组位置：`nearest[rows]`、`nearest[rows[inside]]`、`signs[rows[inside]]`。

直接调用：

```python
np.empty_like
np.ones
len
np.argsort
np.flatnonzero
np.diff
zip
int
self._distance_model
np.clip
closest_point(np.tile(triangles, (len(rows), 1, 1)), np.repeat(current, len(triangles), axis=0)).reshape
closest_point
np.tile
np.repeat
np.square(witnesses - current[:, None]).sum
np.square
np.arange
squared.argmin
values.max
values[inside].argmax
```

## `ObservedContactGeometry.query`

[定义位置](../../../manipulation/contact_geometry.py#L638)。

输入签名：

```python
def ObservedContactGeometry.query(self, spheres, policies):
```

直接读取的对象属性：`self.complete`、`self.maximum_cell_radius`、`self.tree`、`self.max_candidate_pairs`、`self.max_candidate_plane_products`、`self.pair_chunk`、`self._closest_points`、`self.vertices`。

输出分支的源码表达式：

```python
(output.reshape(spheres.shape[:-1]), enabled.reshape(spheres.shape[:-1]))
```

直接写入的状态或数组位置：`enabled[:, list(policy['sphere_indices'])]`、`best_values[selected]`、`best_components[selected]`、`best_allowances[selected]`、`best_points[selected]`、`best_signs[selected]`、`self.last_report`。

直接调用：

```python
ValueError
tuple
math.prod
spheres.reshape
torch.zeros
list
policy['active'].reshape
torch.nonzero(enabled.reshape(-1), as_tuple=False).flatten
torch.nonzero
enabled.reshape
flat[chosen].detach().cpu().numpy().astype
flat[chosen].detach().cpu().numpy
flat[chosen].detach().cpu
flat[chosen].detach
len
np.isfinite(supplied).all
np.isfinite
np.any
np.nextafter
self.tree.query_ball_point
int
np.sum
max
np.repeat
np.arange
np.concatenate(neighbors).astype
np.concatenate
np.empty
np.zeros
np.full
np.ones
chosen.detach().cpu().numpy
chosen.detach().cpu
chosen.detach
min
range
self._closest_points
np.linalg.norm
np.asarray
np.pad
policy['active'].reshape(-1).detach().cpu().numpy
policy['active'].reshape(-1).detach().cpu
policy['active'].reshape(-1).detach
policy['normal_valid'].reshape(-1).detach().cpu().numpy
policy['normal_valid'].reshape(-1).detach().cpu
policy['normal_valid'].reshape(-1).detach
policy['normal_valid'].reshape
np.isin
policy['normals'].detach().cpu().numpy().reshape
policy['normals'].detach().cpu().numpy
policy['normals'].detach().cpu
policy['normals'].detach
policy['bounds'].detach().cpu().numpy().reshape
policy['bounds'].detach().cpu().numpy
policy['bounds'].detach().cpu
policy['bounds'].detach
np.einsum
np.einsum('nki,nvi->nkv', normals, padded).max
(np.minimum(sphere_support, component_support) <= bounds).all
np.minimum
np.maximum
float
np.count_nonzero
np.lexsort
np.flatnonzero
torch.as_tensor
spheres.new_tensor
torch.linalg.vector_norm
torch.relu
output.index_copy
output.reshape
```

## `make_contact_geometry`

[定义位置](../../../manipulation/contact_geometry.py#L741)。

输入签名：

```python
def make_contact_geometry(geometry):
```

输出分支的源码表达式：

```python
ObservedContactGeometry(geometry)
ContactCellGeometry(geometry)
```

直接调用：

```python
any
ObservedContactGeometry
ContactCellGeometry
```

## `LayeredContactCellGeometry`

[定义位置](../../../manipulation/contact_geometry.py#L750)。

```python
class LayeredContactCellGeometry:
    ...
    ...
```

该类型的直接方法：`__init__`、`__copy__`、`gaps`、`complete`、`pair_chunk`、`max_candidate_pairs`、`cell_half_diagonal_m`、`layer_cell_counts`、`workspace_bytes`、`storage_bytes`、`_validate_mask`、`_policy_layer`、`_masked_policy`、`query`、`required_contact`。

## `LayeredContactCellGeometry.__init__`

[定义位置](../../../manipulation/contact_geometry.py#L760)。

输入签名：

```python
def LayeredContactCellGeometry.__init__(self, geometry):
```

直接读取的对象属性：`self.geometry`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.geometry`、`self.layers`、`self.last_report`、`self.last_required_report`。

直接调用：

```python
set
geometry.get
ValueError
deepcopy
make_contact_geometry
```

## `LayeredContactCellGeometry.__copy__`

[定义位置](../../../manipulation/contact_geometry.py#L771)。

输入签名：

```python
def LayeredContactCellGeometry.__copy__(self):
```

直接读取的对象属性：`self.__dict__`、`self.layers`、`result.layers`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result.__dict__`、`result.layers`、`result.last_report`、`result.last_required_report`、`layer.last_report`、`layer.last_required_report`。

直接调用：

```python
type(self).__new__
type
self.__dict__.copy
copy
self.layers.items
result.layers.values
```

## `LayeredContactCellGeometry.gaps`

[定义位置](../../../manipulation/contact_geometry.py#L786)。

输入签名：

```python
@property
def LayeredContactCellGeometry.gaps(self):
```

直接读取的对象属性：`self.layers`。

输出分支的源码表达式：

```python
[f'{name}: {gap}' for (name, layer) in self.layers.items() for gap in layer.gaps]
```

直接调用：

```python
self.layers.items
```

## `LayeredContactCellGeometry.complete`

[定义位置](../../../manipulation/contact_geometry.py#L790)。

输入签名：

```python
@property
def LayeredContactCellGeometry.complete(self):
```

直接读取的对象属性：`self.layers`。

输出分支的源码表达式：

```python
all((layer.complete for layer in self.layers.values()))
```

直接调用：

```python
all
self.layers.values
```

## `LayeredContactCellGeometry.pair_chunk`

[定义位置](../../../manipulation/contact_geometry.py#L794)。

输入签名：

```python
@property
def LayeredContactCellGeometry.pair_chunk(self):
```

直接读取的对象属性：`self.layers`。

输出分支的源码表达式：

```python
sum((layer.pair_chunk for layer in self.layers.values()))
```

直接调用：

```python
sum
self.layers.values
```

## `LayeredContactCellGeometry.max_candidate_pairs`

[定义位置](../../../manipulation/contact_geometry.py#L800)。

输入签名：

```python
@property
def LayeredContactCellGeometry.max_candidate_pairs(self):
```

直接读取的对象属性：`self.layers`。

输出分支的源码表达式：

```python
sum((layer.max_candidate_pairs for layer in self.layers.values()))
```

直接调用：

```python
sum
self.layers.values
```

## `LayeredContactCellGeometry.cell_half_diagonal_m`

[定义位置](../../../manipulation/contact_geometry.py#L804)。

输入签名：

```python
@property
def LayeredContactCellGeometry.cell_half_diagonal_m(self):
```

直接读取的对象属性：`self.layers`。

输出分支的源码表达式：

```python
max((layer.cell_half_diagonal_m for layer in self.layers.values()))
```

直接调用：

```python
max
self.layers.values
```

## `LayeredContactCellGeometry.layer_cell_counts`

[定义位置](../../../manipulation/contact_geometry.py#L810)。

输入签名：

```python
@property
def LayeredContactCellGeometry.layer_cell_counts(self):
```

直接读取的对象属性：`self.layers`。

输出分支的源码表达式：

```python
{name: len(layer.centres) for (name, layer) in self.layers.items()}
```

直接调用：

```python
len
self.layers.items
```

## `LayeredContactCellGeometry.workspace_bytes`

[定义位置](../../../manipulation/contact_geometry.py#L813)。

输入签名：

```python
def LayeredContactCellGeometry.workspace_bytes(self):
```

直接读取的对象属性：`self.layers`。

输出分支的源码表达式：

```python
sum((layer.workspace_bytes() for layer in self.layers.values()))
```

直接调用：

```python
sum
layer.workspace_bytes
self.layers.values
```

## `LayeredContactCellGeometry.storage_bytes`

[定义位置](../../../manipulation/contact_geometry.py#L817)。

输入签名：

```python
@property
def LayeredContactCellGeometry.storage_bytes(self):
```

直接读取的对象属性：`self.layers`。

输出分支的源码表达式：

```python
sum((layer.storage_bytes for layer in self.layers.values()))
```

直接调用：

```python
sum
self.layers.values
```

## `LayeredContactCellGeometry._validate_mask`

[定义位置](../../../manipulation/contact_geometry.py#L821)。

输入签名：

```python
@staticmethod
def LayeredContactCellGeometry._validate_mask(fine_mask):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
isinstance
ValueError
```

## `LayeredContactCellGeometry._policy_layer`

[定义位置](../../../manipulation/contact_geometry.py#L826)。

输入签名：

```python
@staticmethod
def LayeredContactCellGeometry._policy_layer(policy, fine_mask):
```

输出分支的源码表达式：

```python
choices[..., 0]
```

直接调用：

```python
isinstance
ValueError
tuple
any
list
bool
(choices == choices[..., :1]).all
```

## `LayeredContactCellGeometry._masked_policy`

[定义位置](../../../manipulation/contact_geometry.py#L844)。

输入签名：

```python
@staticmethod
def LayeredContactCellGeometry._masked_policy(policy, selected):
```

输出分支的源码表达式：

```python
dict(policy, active=policy['active'] & selected)
```

直接调用：

```python
dict
```

## `LayeredContactCellGeometry.query`

[定义位置](../../../manipulation/contact_geometry.py#L849)。

输入签名：

```python
def LayeredContactCellGeometry.query(self, spheres, policies, fine_mask):
```

直接读取的对象属性：`self._validate_mask`、`self.complete`、`self._policy_layer`、`self._masked_policy`、`self.layers`、`self.layer_cell_counts`。

输出分支的源码表达式：

```python
(torch.where(fine_mask, results['fine'][0], results['coarse'][0]), torch.where(fine_mask, results['fine'][1], results['coarse'][1]))
```

直接写入的状态或数组位置：`results[name]`、`self.last_report`。

直接调用：

```python
self._validate_mask
ValueError
tuple
self._policy_layer
self._masked_policy
zip
self.layers[name].query
deepcopy
self.layers.items
torch.where
```

## `LayeredContactCellGeometry.required_contact`

[定义位置](../../../manipulation/contact_geometry.py#L871)。

输入签名：

```python
def LayeredContactCellGeometry.required_contact(self, policy, fine_mask):
```

直接读取的对象属性：`self._validate_mask`、`self.complete`、`self._policy_layer`、`self.layers`、`self._masked_policy`、`self.layer_cell_counts`。

输出分支的源码表达式：

```python
torch.where(choice, results['fine'], results['coarse'])
```

直接写入的状态或数组位置：`layer.last_required_report`、`results[name]`、`self.last_required_report`。

直接调用：

```python
self._validate_mask
ValueError
self._policy_layer
layer.required_contact
self._masked_policy
deepcopy
self.layers.items
torch.where
```
