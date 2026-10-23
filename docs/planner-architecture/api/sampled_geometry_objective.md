# `manipulation/sampled_geometry_objective.py` 实现契约

代码基线：`774e121-plus-unified-workspace`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_world`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L23)。

输入签名：

```python
def _world(points, pose):
```

实现说明：

> Transform fixed entity-frame material points into the reference frame.

显式输出分支：

```python
pose[0] + quaternion_rotate(pose[1].expand(len(local), 4), local)
```

读取的对象属性：

```python
torch.as_tensor
pose[0].dtype
pose[0].device
pose[1].expand
```

写入目标：

```python
local
```

直接调用符号：

```python
torch.as_tensor
quaternion_rotate
pose[1].expand
len
```

## `_local`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L29)。

输入签名：

```python
def _local(world, pose):
```

显式输出分支：

```python
quaternion_rotate(quaternion_conjugate(pose[1]).expand(len(world), 4), world - pose[0])
```

读取的对象属性：

```python
quaternion_conjugate(pose[1]).expand
```

写入目标：

无显式条目。

直接调用符号：

```python
quaternion_rotate
quaternion_conjugate(pose[1]).expand
quaternion_conjugate
len
```

## `_numeric_points`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L33)。

输入签名：

```python
def _numeric_points(points, source_pose, target_pose):
```

实现说明：

> Evaluate coordinate changes before retaining only active derivatives.
>
> Dense observed material arrays stay in NumPy. Autograd later stores only
> negative active samples, whose selected coordinate expressions determine
> the exact same objective derivative away from branch changes.

显式输出分支：

```python
(world, (world - target_position) @ target_rotation)
```

读取的对象属性：

```python
item.detach().cpu().numpy
item.detach().cpu
item.detach
Rotation.from_quat(source_quaternion[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
Rotation.from_quat(target_quaternion[[1, 2, 3, 0]]).as_matrix
source_rotation.T
```

写入目标：

```python
source_position
source_quaternion
item
target_position
target_quaternion
source_rotation
target_rotation
world
```

直接调用符号：

```python
item.detach().cpu().numpy
item.detach().cpu
item.detach
Rotation.from_quat(source_quaternion[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
Rotation.from_quat(target_quaternion[[1, 2, 3, 0]]).as_matrix
```

## `_allowed`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L48)。

输入签名：

```python
def _allowed(world, permission):
```

实现说明：

> Apply the existing finite point-pair domain to an overlap point.
>
> A sample contributes only where the same reference-frame point belongs
> to both represented entities. Thus both endpoints of a permitted pair
> coincide; the two original finite regions are still tested separately.
> Branch derivatives are zero in the interior of each permission region.

显式输出分支：

```python
allowed
```

读取的对象属性：

```python
np.zeros
permission.regions
np.all
np.asarray(first[0]).T
np.asarray
np.asarray(second[0]).T
```

写入目标：

```python
allowed
first
second
```

直接调用符号：

```python
np.zeros
len
np.all
np.asarray
```

## `MaterialSamples`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L64)。

```python
class MaterialSamples():
```

类字段：

无显式条目。

实现说明：

> Fixed original occupied material and zero-edge points of one entity.
>
> Observed sample cell identities are retained, including duplicate points
> shared by adjacent cells, so an initial-support declaration removes only
> its selected original cells. Native points are constructed by the existing
> feasible-point API from the local projection box's 3-by-3-by-3 grid. The
> box only proposes coordinates; it never supplies collision material.

## `MaterialSamples.__init__`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L74)。

输入签名：

```python
def __init__(self, geometry):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
geometry.nodes
node.cell
np.concatenate
geometry.occupied_points
np.empty
np.full
np.int64
geometry.voxel_size
geometry.cell_origins
geometry._cell_values
zero_points.extend
zero_points.append
np.unique
np.asarray(zero_points).reshape
np.asarray
np.eye
np.linspace
self.native_bounds.T
self.native_bounds
np.mean
self.points
np.linalg.norm
np.linalg
self.points.flags
self.cells.flags
self.cells
self.zero_points.flags
self.zero_points
np.stack
self.points.min
self.points.max
geometry.root
root.lower
root.upper
np.abs
geometry.origin
np.nextafter
np.inf
self.point_bounds
self.negative_bounds
bounds.flags
```

写入目标：

```python
self.geometry
leaves
node
self.points
self.cells
self.length_scale
zero_points
origin
values
first
second
fraction
self.zero_points
self.native_bounds
identity
proposals
low
high
points
point
self.points.flags.writeable
self.cells.flags.writeable
self.zero_points.flags.writeable
self.point_bounds
self.negative_bounds
root
bounds
error
bounds.flags.writeable
```

直接调用符号：

```python
isinstance
np.concatenate
geometry.occupied_points
np.empty
np.full
len
zip
zero_points.extend
zero_points.append
np.unique
np.asarray(zero_points).reshape
np.asarray
_transform
np.eye
_native_bounds
list
product
np.linspace
np.mean
native_feasible_point_bounds
float
np.linalg.norm
np.stack
self.points.min
self.points.max
np.abs
np.nextafter
```

## `_pair_separated`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L138)。

输入签名：

```python
def _pair_separated(first, first_pose, second, second_pose):
```

实现说明：

> Prove that both finite negative-part sums and their pose gradients vanish.
>
> For each direction, transform the source sample box into the target
> coordinates, enclosing forward arithmetic and the rigid inverse's
> arithmetic/orthogonality error. Strict box separation then excludes
> every source sample from the target's possible negative-value region.
> Both directions must pass. Finite permissions only remove contributions,
> so the same implication holds for any region or original-cell exclusion.
> Overlapping or touching boxes supply no conclusion and run the full sum.

显式输出分支：

```python
False
True
```

读取的对象属性：

```python
np.eye
Rotation.from_quat(quaternion.detach().cpu().numpy()[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
quaternion.detach().cpu().numpy
quaternion.detach().cpu
quaternion.detach
position.detach().cpu().numpy
position.detach().cpu
position.detach
matrices.append
source.point_bounds
target.negative_bounds
np.any
```

写入目标：

```python
matrices
position
quaternion
matrix
matrix[:3, :3]
matrix[:3, 3]
source
source_matrix
target
target_matrix
relative
```

直接调用符号：

```python
np.eye
Rotation.from_quat(quaternion.detach().cpu().numpy()[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
quaternion.detach().cpu().numpy
quaternion.detach().cpu
quaternion.detach
position.detach().cpu().numpy
position.detach().cpu
position.detach
matrices.append
_local_interval
world_bounds
np.any
```

## `_field_direction`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L169)。

输入签名：

```python
def _field_direction(source, source_pose, target, target_pose, permission, *, reverse, report):
```

实现说明：

> Mean squared supported TSDF violation with its exact spatial gradient.

显式输出分支：

```python
zero
zero
torch.relu(-signed / target.length_scale).square().sum() / len(source.points)
```

读取的对象属性：

```python
source_pose[0].sum
target_pose[0].sum
source.points
permission.excluded_second_cells
target.geometry.sample
target.geometry
np.isin
source.cells
np.count_nonzero
np.flatnonzero
coordinates.new_tensor
((coordinates - coordinates.detach()) * gradient).sum
coordinates.detach
np.all
torch.relu(-signed / target.length_scale).square().sum
torch.relu(-signed / target.length_scale).square
torch.relu
target.length_scale
```

写入目标：

```python
zero
world
relative
excluded
values
supported
gradients
_
active
report['supported_sample_count']
report['unsupported_sample_count']
report['active_sample_count']
selected
coordinates
gradient
signed
report['negative_sample_count']
report['zero_gradient_negative_count']
```

直接调用符号：

```python
source_pose[0].sum
target_pose[0].sum
len
_numeric_points
target.geometry.sample
_allowed
np.isin
tuple
int
np.count_nonzero
np.flatnonzero
_local
_world
coordinates.new_tensor
((coordinates - coordinates.detach()) * gradient).sum
coordinates.detach
np.all
torch.relu(-signed / target.length_scale).square().sum
torch.relu(-signed / target.length_scale).square
torch.relu
```

## `_native_direction`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L199)。

输入签名：

```python
def _native_direction(source, source_pose, target, target_pose, permission, *, report):
```

实现说明：

> Query original observed material against the unchanged native entity.
>
> The zero-radius sphere denotes a mathematical query point. It does not
> approximate either entity. Box pruning removes only samples with a
> nonnegative native distance, whose squared negative-part cost is zero.
> Coal owns the signed value and nearest-feature normal in native coordinates.

显式输出分支：

```python
zero
zero
torch.relu(-signed / source.length_scale).square().sum() / len(source.points)
```

读取的对象属性：

```python
source_pose[0].sum
target_pose[0].sum
source.points
np.all
target.native_bounds
permission.excluded_second_cells
np.isin
source.cells
np.flatnonzero
coal.DistanceRequest
coal.Sphere
np.eye
coal.DistanceResult
coal.distance
target.geometry
coal.Transform3s
values.append
gradients.append
np.asarray
result.normal
coordinates.new_tensor
((coordinates - coordinates.detach()) * coordinates.new_tensor(np.asarray(gradients))).sum
coordinates.detach
np.count_nonzero
torch.relu(-signed / source.length_scale).square().sum
torch.relu(-signed / source.length_scale).square
torch.relu
source.length_scale
```

写入目标：

```python
zero
world
numeric
active
selected
report['native_point_query_count']
request
request.enable_signed_distance
point
identity
values
gradients
index
result
distance
coordinates
signed
report['negative_native_sample_count']
```

直接调用符号：

```python
source_pose[0].sum
target_pose[0].sum
len
_numeric_points
np.all
_allowed
np.isin
tuple
np.flatnonzero
coal.DistanceRequest
coal.Sphere
_transform
np.eye
coal.DistanceResult
coal.distance
coal.Transform3s
values.append
gradients.append
np.asarray
_local
_world
coordinates.new_tensor
((coordinates - coordinates.detach()) * coordinates.new_tensor(np.asarray(gradients))).sum
coordinates.detach
int
np.count_nonzero
torch.relu(-signed / source.length_scale).square().sum
torch.relu(-signed / source.length_scale).square
torch.relu
```

## `_pair_value`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L238)。

输入签名：

```python
def _pair_value(first, first_pose, second, second_pose, *, permission, report):
```

实现说明：

> Evaluate both finite directions before reducing their pose derivative.

显式输出分支：

```python
(forward + backward) / 2
```

读取的对象属性：

```python
report.setdefault
first.geometry
```

写入目标：

```python
report
key
forward
backward
```

直接调用符号：

```python
report.setdefault
_field_direction
isinstance
_native_direction
```

## `pair_objective`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L254)。

输入签名：

```python
def pair_objective(first, first_pose, second, second_pose, *, permission=None, report=None):
```

实现说明：

> Return the actual finite value and its first derivative to both poses.
>
> One temporary point graph is reduced to fourteen pose derivatives before
> processing the next pair. The caller's trajectory graph therefore stores
> positions/quaternions per pair rather than material points per trajectory
> state. SLSQP consumes this first derivative; a second derivative through
> the detached spatial/nearest-feature query is outside this API's scope.

显式输出分支：

```python
coordinates.sum() * 0
_pair_value(first, first_pose, second, second_pose, permission=permission, report=report)
value.detach() + ((coordinates - coordinates.detach()) * derivative).sum()
```

读取的对象属性：

```python
torch.cat
report.setdefault
coordinates.sum
torch.is_grad_enabled
coordinates.requires_grad
torch.enable_grad
coordinates.detach().requires_grad_
coordinates.detach
local.sum
torch.autograd.grad
torch.autograd
value.detach
((coordinates - coordinates.detach()) * derivative).sum
```

写入目标：

```python
report
coordinates
report['zero_contribution_separated_pair_count']
key
local
value
derivative
```

直接调用符号：

```python
torch.cat
report.setdefault
_pair_separated
coordinates.sum
torch.is_grad_enabled
_pair_value
torch.enable_grad
coordinates.detach().requires_grad_
coordinates.detach
local.sum
torch.autograd.grad
value.detach
((coordinates - coordinates.detach()) * derivative).sum
```

## `contact_objective`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L286)。

输入签名：

```python
def contact_objective(samples, normals, bounds):
```

实现说明：

> Minimum region violation, retaining only the selected branch's graph.

显式输出分支：

```python
normals.sum() * 0 + bounds.sum() * 0
violation.square().sum() / samples.length_scale ** 2
```

读取的对象属性：

```python
samples.zero_points
normals.sum
bounds.sum
np.maximum
normals.detach().cpu().numpy().T
normals.detach().cpu().numpy
normals.detach().cpu
normals.detach
bounds.detach().cpu().numpy
bounds.detach().cpu
bounds.detach
np.argmin
np.square(numeric).sum
np.square
normals.new_tensor
torch.relu
normals.T
violation.square().sum
violation.square
samples.length_scale
```

写入目标：

```python
numeric
selected
point
violation
```

直接调用符号：

```python
len
normals.sum
bounds.sum
np.maximum
normals.detach().cpu().numpy
normals.detach().cpu
normals.detach
bounds.detach().cpu().numpy
bounds.detach().cpu
bounds.detach
int
np.argmin
np.square(numeric).sum
np.square
normals.new_tensor
torch.relu
violation.square().sum
violation.square
```

## `native_pair_objective`

[定义位置](../../../manipulation/sampled_geometry_objective.py#L298)。

输入签名：

```python
def native_pair_objective(first, first_pose, first_matrix, second, second_pose, second_matrix):
```

实现说明：

> Squared signed native penetration with the same nearest-point derivative.
>
> At a unique smooth closest feature, the signed-distance derivative is
> n.T @ (J_second - J_first). Coal supplies the signed scalar and witnesses
> at this trial; rotations and translations of those material witnesses
> remain connected to the supplied differentiable robot link poses.

显式输出分支：

```python
torch.relu(-signed / length).square()
```

读取的对象属性：

```python
coal.DistanceRequest
coal.DistanceResult
coal.distance
first.geometry
second.geometry
np.asarray
result.getNearestPoint1
result.getNearestPoint2
(relative * relative.new_tensor(np.asarray(result.normal))).sum
relative.new_tensor
result.normal
derivative.new_tensor
derivative.detach
first.length_scale
second.length_scale
torch.relu(-signed / length).square
torch.relu
```

写入目标：

```python
request
result
request.enable_signed_distance
distance
first_point
second_point
relative
derivative
signed
length
```

直接调用符号：

```python
coal.DistanceRequest
coal.DistanceResult
coal.distance
_transform
inverse_points
np.asarray
result.getNearestPoint1
result.getNearestPoint2
_world
(relative * relative.new_tensor(np.asarray(result.normal))).sum
relative.new_tensor
derivative.new_tensor
derivative.detach
torch.relu(-signed / length).square
torch.relu
```
