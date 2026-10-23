# `manipulation/grasp_region.py` 实现契约

代码基线：`d7bcafc`。
职责和数据语义见[组件说明](../03-attachment-contact-and-hand.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_array`

[定义位置](../../../manipulation/grasp_region.py#L21)。

输入签名：

```python
def _array(value, shape, label):
```

显式输出分支：

```python
result.astype(float, copy=True)
```

读取的对象属性：

```python
np.asarray
result.shape
result.dtype.kind
result.dtype
np.isfinite(result).all
np.isfinite
result.astype
```

写入目标：

```python
result
```

直接调用符号：

```python
np.asarray
np.isfinite(result).all
np.isfinite
ValueError
result.astype
```

## `_rule_geometry`

[定义位置](../../../manipulation/grasp_region.py#L29)。

输入签名：

```python
def _rule_geometry(record, joint_count):
```

实现说明：

> Read task feature coordinates, finite patches and their tolerances.
>
> Features remain source task data. The complete ContactRule also binds
> body identity and activity; this existential query uses its geometric
> fields and the caller's continuous finger-coordinate limits.

显式输出分支：

```python
result
```

读取的对象属性：

```python
result[key].strip
np.asarray
points.ndim
points.shape
np.isclose
np.linalg.norm
np.linalg
np.allclose
axes.T
np.eye
np.linalg.det
np.any
np.bool_
np.number
np.isfinite
math.pi
```

写入目标：

```python
names
required
result
key
points
kind
result['feature_points']
result['feature_normal']
normal
mapping
result['feature_joint_translation']
result['patch_origin']
result['patch_axes']
axes
result['patch_half_size']
half
result['gap_bounds']
gap
scalars
value
result[key]
```

直接调用符号：

```python
isinstance
ValueError
set
any
result[key].strip
np.asarray
len
_array
np.isclose
np.linalg.norm
np.allclose
np.eye
np.linalg.det
np.any
np.isfinite
float
min
```

## `_feature_region_halfspaces`

[定义位置](../../../manipulation/grasp_region.py#L92)。

输入签名：

```python
def _feature_region_halfspaces(rule, actor_pose, target_pose, finger_positions):
```

实现说明：

> Return world task inequalities and their affine finger derivative.
>
> A @ x <= b describes the finite source feature and target patch. Rigid
> prismatic feature motion changes b by db_dq @ dq. Source point heights
> are retained; the region does not assert observed target occupancy.

显式输出分支：

```python
(None, None, 'missing_named_finger_geometry')
(None, None, 'material_normal_outside_permission')
(region, np.asarray(derivatives), None)
```

读取的对象属性：

```python
np.asarray
np.einsum
np.zeros
actor_pose[:3, :3].T
np.arctan2
np.linalg.norm
np.linalg
np.cross
world.shape
rows.append
bounds.append
derivatives.append
np.linalg.svd(local_points - local_points[0], full_matrices=True)[2][:2].T
np.linalg.svd
ConvexHull(local).equations
np.max
translation.shape
```

写入目标：

```python
points
local_points
mapping
translation
world
normal
axes
origin
tolerance
angle
rows
bounds
derivatives
kind
delta
length
local_tangent
lateral
tangent
axis
extent
basis
local
edges
edge
allowance
column
half
sign
gap_axis
gap_offset
gap_derivative
height
cosine
limit
region
```

直接调用符号：

```python
np.asarray
np.einsum
np.zeros
float
np.arctan2
np.linalg.norm
np.cross
ValueError
rows.append
bounds.append
derivatives.append
np.linalg.svd
ConvexHull
np.max
enumerate
```

## `check_grasp_region_compatibility`

[定义位置](../../../manipulation/grasp_region.py#L181)。

输入签名：

```python
def check_grasp_region_compatibility(contact_rules, *, T_reference_tool, T_reference_owner, finger_position_limits):
```

实现说明：

> Check existential compatibility with fixed, material-specific patches.
>
> ``contact_rules`` contains only the robot rules bound to this grasp arrival
> and one owner. Their feature coordinates use the named tool frame, patch
> coordinates use the owner frame, and affine finger columns use the same
> coordinate order as ``finger_position_limits`` (J by 2, metres). The caller
> binds those identities and supplies existing calibrated geometry. Rule
> ``required`` flags, clocks and collision permissions are never changed.
>
> Rules for one material are alternatives (OR); all materials must fit at
> one common hypothetical q (AND). A separate intersection point per material
> and shared continuous q variables give a linear feasibility problem for
> each rule combination. No finite set of sampled openings substitutes for
> the full declared coordinate box. The query does not assume synchronous
> fractional closure or assert that physical contact will stop at its q.
>
> ``compatible`` has a hypothetical witness; ``contradicted`` means every
> finite permission combination is geometrically incompatible; solver
> incompleteness gives ``unresolved``. Missing observed target sides cannot
> cause rejection because observed target geometry is not an input here.
> A caller must check the final planned arrival poses too if it lets the
> trajectory optimizer move within a tolerance around this nominal pose.

显式输出分支：

```python
report
report
report
report
```

读取的对象属性：

```python
np.asarray
limits.ndim
limits.shape
np.any
grouped.setdefault(row['material_id'], []).append
grouped.setdefault
grouped.values
np.array_equal
limits.mean
report['normal_rejected_rule_ids'].append
report.update
derivative.shape
np.zeros
valid.append
choices.append
matrices.append
bounds.append
np.concatenate
solved.success
(q_reference + solved.x[:joint_count]).tolist
solved.x
solved.x[joint_count:].reshape(material_count, 3).tolist
solved.x[joint_count:].reshape
np.max
solved.status
report['incomplete_rule_combinations'].append
```

写入目标：

```python
tool
owner
limits
records
row
grouped
fixed
alternatives
first
key
material_ids
report
q_reference
choices
material
valid
rule
region
derivative
reason
joint_count
material_count
variable_count
variable_bounds
selected
matrices
bounds
index
_
block
block[:, :joint_count]
block[:, joint_count + 3 * index:joint_count + 3 * index + 3]
matrix
bound
solved
report['checked_rule_combinations']
identities
```

直接调用符号：

```python
_pose_array
np.asarray
ValueError
_array
np.any
_rule_geometry
len
grouped.setdefault(row['material_id'], []).append
grouped.setdefault
grouped.values
any
np.array_equal
sorted
limits.mean
_feature_region_halfspaces
report['normal_rejected_rule_ids'].append
report.update
np.zeros
valid.append
choices.append
list
map
product
enumerate
matrices.append
bounds.append
np.concatenate
linprog
(q_reference + solved.x[:joint_count]).tolist
solved.x[joint_count:].reshape(material_count, 3).tolist
solved.x[joint_count:].reshape
float
np.max
report['incomplete_rule_combinations'].append
int
```
