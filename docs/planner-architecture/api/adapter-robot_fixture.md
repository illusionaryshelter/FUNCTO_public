# `tests/planning/robot_fixture.py` 实现契约

代码基线：`pending-implementation-commit`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_pose`

[定义位置](../../../tests/planning/robot_fixture.py#L38)。

输入签名：

```python
def _pose(position, quaternion):
```

显式输出分支：

```python
value
```

读取的对象属性：

```python
np.eye
Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
np.asarray
```

写入目标：

```python
value
value[:3, :3]
value[:3, 3]
```

直接调用符号：

```python
np.eye
Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
np.asarray
```

## `_pose_dict`

[定义位置](../../../tests/planning/robot_fixture.py#L45)。

输入签名：

```python
def _pose_dict(transform):
```

显式输出分支：

```python
{'position': transform[:3, 3].tolist(), 'quaternion_wxyz': Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()}
```

读取的对象属性：

```python
transform[:3, 3].tolist
Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist
Rotation.from_matrix(transform[:3, :3]).as_quat
Rotation.from_matrix
```

写入目标：

无显式条目。

直接调用符号：

```python
transform[:3, 3].tolist
Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist
Rotation.from_matrix(transform[:3, :3]).as_quat
Rotation.from_matrix
```

## `_native_geom_pose`

[定义位置](../../../tests/planning/robot_fixture.py#L51)。

输入签名：

```python
def _native_geom_pose(model, geom_id):
```

实现说明：

> Use MuJoCo's actual homogeneous geom matrix, not SciPy normalization.

显式输出分支：

```python
result
```

读取的对象属性：

```python
model.geom_pos
model.geom_quat
np.isfinite(position).all
np.isfinite
np.isfinite(quaternion).all
np.eye
np.empty
mujoco.mju_quat2Mat
rotation.reshape
```

写入目标：

```python
position
quaternion
result
rotation
result[:3, :3]
result[:3, 3]
```

直接调用符号：

```python
np.isfinite(position).all
np.isfinite
np.isfinite(quaternion).all
ValueError
np.eye
np.empty
mujoco.mju_quat2Mat
rotation.reshape
```

## `_cover_cells`

[定义位置](../../../tests/planning/robot_fixture.py#L63)。

输入签名：

```python
def _cover_cells(vertices, *, count=None, box_halfsize=None, subdivisions=None):
```

实现说明：

> Partition the complete native solid, rather than sampled surfaces.
>
> Mesh partitions are exact intersections of convex-hull halfspaces with
> Voronoi halfspaces. Sobol points only choose sites; they are not coverage
> evidence. A ball enclosing every vertex encloses its entire convex cell.
> Box cells use an analytic Cartesian partition with the same certificate.

显式输出分支：

```python
([-box_halfsize + (np.asarray(index) + 0.5) * step + offsets for index in product(*(range(value) for value in bins))], None)
(cells, sites)
```

读取的对象属性：

```python
np.asarray
bins.shape
np.any
hull.equations
qmc.Sobol(3, scramble=False).random_base2
qmc.Sobol
vertices.min
np.ptp
np.all
planes[:, :3].T
np.delete
np.einsum
np.linalg.norm
np.linalg
np.column_stack
cells.append
HalfspaceIntersection(np.concatenate((planes, cuts)), site).intersections
np.concatenate
```

写入目标：

```python
bins
step
offsets
index
value
hull
planes
points
sites
_
cells
site
others
normals
distances
cuts
```

直接调用符号：

```python
np.asarray
np.any
ValueError
list
product
range
ConvexHull
qmc.Sobol(3, scramble=False).random_base2
qmc.Sobol
vertices.min
np.ptp
np.all
min
len
kmeans2
enumerate
np.delete
np.einsum
np.linalg.norm
np.column_stack
cells.append
HalfspaceIntersection
np.concatenate
```

## `_fit_cell_sphere_center`

[定义位置](../../../tests/planning/robot_fixture.py#L101)。

输入签名：

```python
def _fit_cell_sphere_center(vertices, planes):
```

实现说明：

> Minimize the maximum outward native support-plane excess of a ball.
>
> The convex program is min t subject to ||c-v|| <= r for EVERY cell vertex
> and n.c+d+r <= t for EVERY normalized native hull plane. A smaller radius
> alone need not reduce false collisions: moving a center inward can admit
> a larger ball whose exterior follows the native solid more closely.
>
> Trimesh supplies only a feasible initial center, never an optimality or
> coverage certificate. SciPy solves a unit-scaled five-variable problem;
> explicit norm/plane constraints and objective checks accompany its status.
> The caller recomputes coverage after float32 serialization. This directional
> plane-excess objective is NOT a Euclidean Hausdorff error bound.

显式输出分支：

```python
(fitted.x[:3] * scale + origin, {'constraint_violation_m': max(0.0, -residual) * scale, 'initial_support_plane_excess_m': initial_excess * scale, 'fitted_support_plane_excess_m': measured_excess * scale})
```

读取的对象属性：

```python
np.asarray
vertices.ndim
vertices.shape
planes.ndim
planes.shape
np.isfinite(vertices).all
np.isfinite
np.isfinite(planes).all
np.linalg.norm
np.linalg
np.any
vertices.mean
np.ptp
trimesh.nsphere.minimum_nsphere
trimesh.nsphere
np.linalg.norm(points - initial_center, axis=1).max
(normals @ initial_center + constants + initial_radius).max
np.r_
np.array
fitted.success
np.isfinite(fitted.x).all
fitted.x
fitted.message
constraints(fitted.x).min
np.linalg.norm(points - fitted.x[:3], axis=1).max
(normals @ fitted.x[:3] + constants + covering_radius).max
```

写入目标：

```python
vertices
planes
lengths
origin
scale
points
normals
constants
initial_center
_
initial_radius
initial_excess
initial
fitted
residual
covering_radius
measured_excess
```

直接调用符号：

```python
np.asarray
len
np.isfinite(vertices).all
np.isfinite
np.isfinite(planes).all
ValueError
np.linalg.norm
np.any
vertices.mean
float
np.ptp
trimesh.nsphere.minimum_nsphere
np.linalg.norm(points - initial_center, axis=1).max
(normals @ initial_center + constants + initial_radius).max
minimize
np.array
np.isfinite(fitted.x).all
constraints(fitted.x).min
constraints
np.linalg.norm(points - fitted.x[:3], axis=1).max
(normals @ fitted.x[:3] + constants + covering_radius).max
max
```

## `_fit_cell_sphere_center.constraints`

[定义位置](../../../tests/planning/robot_fixture.py#L137)。

输入签名：

```python
def constraints(value):
```

显式输出分支：

```python
np.r_[value[3] - np.linalg.norm(points - value[:3], axis=1), value[4] - normals @ value[:3] - constants - value[3]]
```

读取的对象属性：

```python
np.r_
np.linalg.norm
np.linalg
```

写入目标：

无显式条目。

直接调用符号：

```python
np.linalg.norm
```

## `_fit_cell_sphere_center.jacobian`

[定义位置](../../../tests/planning/robot_fixture.py#L141)。

输入签名：

```python
def jacobian(value):
```

显式输出分支：

```python
np.vstack((np.column_stack((delta / distance[:, None], np.ones(len(points)), np.zeros(len(points)))), np.column_stack((-normals, -np.ones(len(normals)), np.ones(len(normals))))))
```

读取的对象属性：

```python
np.linalg.norm
np.linalg
np.any
np.vstack
np.column_stack
np.ones
np.zeros
```

写入目标：

```python
delta
distance
```

直接调用符号：

```python
np.linalg.norm
np.any
ValueError
np.vstack
np.column_stack
np.ones
len
np.zeros
```

## `_geom_cover`

[定义位置](../../../tests/planning/robot_fixture.py#L171)。

输入签名：

```python
def _geom_cover(model, geom_id, specification, destination):
```

显式输出分支：

```python
(spheres, record, geometry)
```

读取的对象属性：

```python
model.geom_type
mujoco.mjtGeom.mjGEOM_BOX
mujoco.mjtGeom
specification.get
model.geom_size
np.linalg.solve(local[:3, :3].T, planes[:, :3].T).T
np.linalg.solve
np.linalg
local[:3, :3].T
planes[:, :3].T
np.linalg.norm
np.column_stack
fitting.append
np.asarray(center, dtype=np.float32).astype
np.asarray
np.float32
np.nextafter
distances.max
np.inf
spheres.append
center.tolist
proof_vertices.extend
offsets.append
margins.append
normals_body.T
np.savez_compressed
np.empty
model.geom(geom_id).name
model.geom
model.body(model.geom_bodyid[geom_id]).name
model.body
model.geom_bodyid
local.tolist
excess.max
model.geom_contype
model.geom_conaffinity
```

写入目标：

```python
geometry
local
is_box
cells
sites
planes
normals_body
normal_lengths
offsets_body
body_planes
spheres
proof_vertices
offsets
margins
fitting
cell
body_vertices
center
fit
distances
radius
centers
sphere
radii
excess
record
row
```

直接调用符号：

```python
_collision_geometry
_native_geom_pose
set
ValueError
_cover_cells
specification.get
np.linalg.solve
np.linalg.norm
np.column_stack
_fit_cell_sphere_center
fitting.append
np.asarray(center, dtype=np.float32).astype
np.asarray
float
np.nextafter
np.float32
distances.max
spheres.append
center.tolist
proof_vertices.extend
offsets.append
len
margins.append
np.savez_compressed
np.empty
model.geom
model.body
local.tolist
str
min
excess.max
max
int
```

## `_native_ignored_pairs`

[定义位置](../../../tests/planning/robot_fixture.py#L240)。

输入签名：

```python
def _native_ignored_pairs(model, collision_links):
```

实现说明：

> Keep exactly the native welded/parent and collision-mask exclusions.
>
> A link pair is ignored only when all its native geom pairs are excluded.
> Coarse proxy overlap is not a reason to remove an enabled collision pair.

显式输出分支：

```python
result
```

读取的对象属性：

```python
model.opt.disableflags
model.opt
mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
mujoco.mjtDisableBit
model.body(name).id
model.body
np.flatnonzero
model.geom_bodyid
model.body(other_name).id
model.body_weldid
model.body_parentid
model.geom_contype
model.geom_conaffinity
result.setdefault(name, []).append
result.setdefault
```

写入目标：

```python
result
filter_parent
i
name
body
first
other_name
other
second
weld_a
weld_b
parent_a
parent_b
welded
parent
masked
a
b
```

直接调用符号：

```python
int
enumerate
model.body
np.flatnonzero
all
result.setdefault(name, []).append
result.setdefault
```

## `_interval_bounds`

[定义位置](../../../tests/planning/robot_fixture.py#L268)。

输入签名：

```python
def _interval_bounds(value):
```

实现说明：

> Serialize an interval without rounding either endpoint inward.
>
> mpmath owns all interval arithmetic. The single nextafter at this output
> boundary only encloses conversion of each arbitrary-precision endpoint
> to the binary64 number which JSON can round-trip exactly.

显式输出分支：

```python
bounds
```

读取的对象属性：

```python
np.nextafter
value.a
np.inf
value.b
np.isfinite(bounds).all
np.isfinite
```

写入目标：

```python
bounds
```

直接调用符号：

```python
float
np.nextafter
np.isfinite(bounds).all
np.isfinite
ValueError
```

## `_interval_rotation`

[定义位置](../../../tests/planning/robot_fixture.py#L282)。

输入签名：

```python
def _interval_rotation(context, quaternion, *, normalize):
```

实现说明：

> Enclose the native body/geom quaternion conventions using iv.
>
> Compiled floating values are the inputs, not decimal reconstructions.
> Native body FK normalizes its accumulated quaternion. Native geom FK does
> NOT normalize the geom quaternion again: its homogeneous quadratic matrix
> retains the compiled norm residual. Applying normalization to a geom here
> would silently change its support points at large geometry scales.
> C(q_body_unit*q_geom) = R_body*C(q_geom), so the native geom matrix can be
> projected locally without assuming that it is an exact unit rotation.

显式输出分支：

```python
[[w * w + x * x - y * y - z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), w * w - x * x + y * y - z * z, 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), w * w - x * x - y * y + z * z]]
```

读取的对象属性：

```python
np.asarray
values.shape
np.isfinite(values).all
np.isfinite
context.mpf
context.sqrt
norm.a
abs(norm - 1).b
np.finfo(float).eps
np.finfo
```

写入目标：

```python
values
q
value
norm
w
x
y
z
```

直接调用符号：

```python
np.asarray
np.isfinite(values).all
np.isfinite
ValueError
context.mpf
float
context.sqrt
sum
abs
np.finfo
```

## `_hinge_separation_certificate`

[定义位置](../../../tests/planning/robot_fixture.py#L306)。

输入签名：

```python
def _hinge_separation_certificate(model, parent, child):
```

实现说明：

> Prove separation of complete native solids for every hinge angle.
>
> Each mesh is its original native convex hull; a linear projection of that
> hull is bounded by projections of ALL compiled vertices, including a
> nonconvex authored mesh's vertices. Boxes use their existing eight-corner
> geometry. Hinge rotation preserves axial projections, so this argument
> contains no angle samples and is unaffected by ancestor/descendant motion.
> This exporter still only admits an origin-centred, single scalar joint.
> Unknown geometry or an unproved separation leaves the query enabled.

显式输出分支：

```python
None
None
None
None
None
None
{'kind': 'direct_hinge_complete_solid_axial_separation', 'joint': model.joint(joint).name, 'parent_body': model.body(parent).name, 'child_body': model.body(child).name, 'joint_axis_child': model.jnt_axis[joint].tolist(), 'joint_pivot_child': model.jnt_pos[joint].tolist(), 'child_position_parent': model.body_pos[child].tolist(), 'child_quaternion_parent_wxyz': model.body_quat[child].tolist(), 'native_joint_range': model.jnt_range[joint].tolist(), 'domain': 'all real hinge angles; other joints and common rigid motion unrestricted', 'all_enabled_geometries': records, 'native_geom_pair_parameters': pair_parameters, 'body_axis_projection_intervals_m': intervals, 'separation_direction': 'child_above_parent' if direction == 0 else 'parent_above_child', 'separation_lower_bound_m': _interval_bounds(separation)[0], 'required_distance_upper_bound_m': _interval_bounds(required)[1], 'certified_slack_lower_bound_m': _interval_bounds(separation - required)[0], 'margin_override_enabled': override, 'option_o_margin_m': float(model.opt.o_margin), 'native_ccd_tolerance_m': float(model.opt.ccd_tolerance), 'arithmetic': 'local mpmath.iv, 113 bits; compiled binary inputs; outward binary64 endpoints', 'native_pair_remains_enabled': True, 'scope': 'this native self pair only; no environment, object or other-link query removed'}
```

读取的对象属性：

```python
model.body_parentid
model.body_jntnum
model.body_jntadr
model.jnt_type
mujoco.mjtJoint.mjJNT_HINGE
mujoco.mjtJoint
np.any
model.jnt_pos
np.flatnonzero
model.geom_bodyid
model.geom_contype
model.geom_conaffinity
model.geom_type
mujoco.mjtGeom.mjGEOM_BOX
mujoco.mjtGeom
mujoco.mjtGeom.mjGEOM_MESH
model.body_pos
model.jnt_axis
model.jnt_range
model.geom_margin
np.concatenate
model.geom_gap
np.asarray
model.opt.o_margin
model.opt
model.opt.ccd_tolerance
np.isfinite(value).all
np.isfinite
context.mpf
context.sqrt
norm.a
abs(norm - 1).b
np.finfo(float).eps
np.finfo
model.body_quat
model.geom_pos
np.isfinite(vertices).all
np.isfinite(position).all
model.geom_quat
value.a
value.b
body_intervals.append
records.append
model.geom(index).name
model.geom
model.body(model.geom_bodyid[index]).name
model.body
position.tolist
model.geom_quat[index].tolist
intervals.append
separations[index].a
model.opt.enableflags
mujoco.mjtEnableBit.mjENBL_OVERRIDE
mujoco.mjtEnableBit
pair_parameters.append
model.geom(first).name
model.geom(second).name
separation.a
required.b
model.joint(joint).name
model.joint
model.body(parent).name
model.body(child).name
model.jnt_axis[joint].tolist
model.jnt_pos[joint].tolist
model.body_pos[child].tolist
model.body_quat[child].tolist
model.jnt_range[joint].tolist
```

写入目标：

```python
joint
ids
body
group
enabled
a
b
index
context
context.prec
dot
inputs
value
axis
norm
parent_rotation
parent_axis
row
parent_offset
records
intervals
body_axis
offset
body_intervals
geometry
vertices
position
rotation
coefficients
column
translation
projections
vertex
bounds
separations
direction
separation
override
required
pair_parameters
first
second
margin
gap
include
```

直接调用符号：

```python
int
np.any
np.flatnonzero
any
len
MPIntervalContext
sum
zip
np.concatenate
np.asarray
np.isfinite(value).all
np.isfinite
ValueError
context.mpf
float
context.sqrt
dot
abs
np.finfo
_interval_rotation
_collision_geometry
np.isfinite(vertices).all
np.isfinite(position).all
_interval_bounds
min
max
body_intervals.append
records.append
model.geom
model.body
position.tolist
model.geom_quat[index].tolist
intervals.append
range
bool
pair_parameters.append
model.joint
model.jnt_axis[joint].tolist
model.jnt_pos[joint].tolist
model.body_pos[child].tolist
model.body_quat[child].tolist
model.jnt_range[joint].tolist
```

## `_mechanical_contact_contract`

[定义位置](../../../tests/planning/robot_fixture.py#L437)。

输入签名：

```python
def _mechanical_contact_contract(model, links, pairs, native_exclusions):
```

实现说明：

> Map exact geom permissions only when native link ignores are exact.
>
> MuJoCo contact enablement asks physics to resolve a pair; it is not itself
> a requirement that the two solids always remain separated. The caller may
> therefore explicitly permit mechanical contact between named native geoms.
> This declaration never changes their native contact masks or model state.
>
> The maintained cuRobo YAML interface ignores whole link pairs. Use it only
> when every native-enabled geom pair between those links is explicitly
> permitted. A mixed link pair with an unpermitted geom is unsupported here,
> rather than broadening permission to that geom or editing kernel buffers.

显式输出分支：

```python
(declared, exclusions)
```

读取的对象属性：

```python
mujoco.mjtDisableBit.mjDSBL_CONTACT
mujoco.mjtDisableBit
mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
model.opt.disableflags
model.opt
row[key].strip
mujoco.mj_name2id
mujoco.mjtObj.mjOBJ_GEOM
mujoco.mjtObj
model.geom_bodyid
model.body(index).name
model.body
native_exclusions.get
requested.add
link_pairs.add
link_order.__getitem__
declared.append
np.flatnonzero
model.body(first).id
model.body(second).id
model.geom(a).name
model.geom
model.geom(b).name
exclusions.setdefault(first, []).append
exclusions.setdefault
```

写入目标：

```python
disabled
link_order
index
name
declared
requested
link_pairs
row
key
first
second
identity
ids
bodies
names
exclusions
first_geoms
second_geoms
a
b
```

直接调用符号：

```python
isinstance
ValueError
int
enumerate
set
any
row[key].strip
frozenset
len
mujoco.mj_name2id
model.body
native_exclusions.get
enabled
requested.add
link_pairs.add
tuple
sorted
declared.append
np.flatnonzero
model.geom
exclusions.setdefault(first, []).append
exclusions.setdefault
```

## `_mechanical_contact_contract.enabled`

[定义位置](../../../tests/planning/robot_fixture.py#L458)。

输入签名：

```python
def enabled(first, second):
```

显式输出分支：

```python
bool(model.geom_contype[first] & model.geom_conaffinity[second] or model.geom_contype[second] & model.geom_conaffinity[first])
```

读取的对象属性：

```python
model.geom_contype
model.geom_conaffinity
```

写入目标：

无显式条目。

直接调用符号：

```python
bool
```

## `_collision_query_contract`

[定义位置](../../../tests/planning/robot_fixture.py#L501)。

输入签名：

```python
def _collision_query_contract(model, kinematics, *, mechanical_contact_pairs):
```

实现说明：

> Separate native filtering, explicit mechanical contact and separation proofs.

显式输出分支：

```python
{'native_policy_exclusions': native, 'native_enabled_sphere_pair_count': original_count, 'mechanical_contact_pairs': mechanical, 'mechanical_contact_link_exclusions': contact_exclusions, 'mechanical_contact_sphere_pair_count': sum((len(spheres[first]) * len(spheres[second]) for (first, others) in contact_exclusions.items() for second in others)), 'certificate_completed_pairs': completed, 'effective_query_pairs': queries, 'effective_query_sphere_pair_count': sum((len(spheres[a]) * len(spheres[b]) for (a, b) in queries))}
```

读取的对象属性：

```python
model.npair
model.nexclude
model.neq
model.ntendon
model.nflex
model.nplugin
mujoco.get_mjcb_contactfilter
native.get
contact_exclusions.get
model.body(first).id
model.body
model.body(second).id
model.body_parentid
np.isfinite(clearances).all
np.isfinite
queries.append
completed.append
contact_exclusions.items
```

写入目标：

```python
links
spheres
native
mechanical
contact_exclusions
completed
queries
original_count
index
first
second
a
b
parent
child
clearances
certificate
value
others
```

直接调用符号：

```python
mujoco.get_mjcb_contactfilter
ValueError
len
set
_native_ignored_pairs
_mechanical_contact_contract
enumerate
native.get
contact_exclusions.get
model.body
np.isfinite(clearances).all
np.isfinite
all
_hinge_separation_certificate
queries.append
completed.append
sum
contact_exclusions.items
```

## `_effective_ignored_pairs`

[定义位置](../../../tests/planning/robot_fixture.py#L550)。

输入签名：

```python
def _effective_ignored_pairs(contract):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
contract['native_policy_exclusions'].items
contract['mechanical_contact_link_exclusions'].items
result.setdefault(first, []).extend
result.setdefault
result.setdefault(first, []).append
```

写入目标：

```python
result
name
others
first
certificate
second
```

直接调用符号：

```python
list
contract['native_policy_exclusions'].items
contract['mechanical_contact_link_exclusions'].items
result.setdefault(first, []).extend
result.setdefault
result.setdefault(first, []).append
```

## `_source_collision_names`

[定义位置](../../../tests/planning/robot_fixture.py#L561)。

输入签名：

```python
def _source_collision_names(scene, collision_root):
```

实现说明：

> Require the existing expanded-source contract before using any proof.

显式输出分支：

```python
{node.get('name') for node in source_root.iter('geom') if int(node.get('contype', '1')) or int(node.get('conaffinity', '1'))}
```

读取的对象属性：

```python
scene.find
scene.iter
node.get
source_root.iter
```

写入目标：

```python
tag
source_root
node
```

直接调用符号：

```python
scene.find
ValueError
any
len
next
scene.iter
node.get
source_root.iter
int
```

## `_fixed_installation_records`

[定义位置](../../../tests/planning/robot_fixture.py#L576)。

输入签名：

```python
def _fixed_installation_records(model, *, root_body, sphere_geom_names, sphere_links):
```

实现说明：

> Export native fixed robot geometry and only its native pair exclusions.
>
> These bodies belong to the calibrated robot subtree. They are complete
> static collision owners, not environment reconstructions or extra moving
> spheres. In particular a cylinder remains an analytic cylinder here; the
> cuRobo adapter separately records its conservative polygonal approximation.

显式输出分支：

```python
result
```

读取的对象属性：

```python
model.body(root_body).id
model.body
model.nbody
model.body_parentid
bodies.add
model.ngeom
model.geom_bodyid
model.geom_contype
model.geom_conaffinity
model.geom(index).name
model.geom
mujoco.MjData
mujoco.mj_kinematics
data.xpos
data.xquat
model.body_jntnum
model.geom_type
mujoco.mjtGeom.mjGEOM_BOX
mujoco.mjtGeom
(2 * model.geom_size[index]).tolist
model.geom_size
mujoco.mjtGeom.mjGEOM_CYLINDER
np.isfinite(dimensions).all
np.isfinite
np.any
np.asarray
model.body(body).name
policy.get
excluded.append
np.eye
data.geom_xmat[index].reshape
data.geom_xmat
data.geom_xpos
result.append
_native_geom_pose(model, index).tolist
(np.linalg.inv(base) @ native_world).tolist
np.linalg.inv
np.linalg
```

写入目标：

```python
root
bodies
body
enabled
index
extras
data
base
result
cursor
kind
shape_kind
dimensions
link
excluded
other
policy
native_world
native_world[:3, :3]
native_world[:3, 3]
```

直接调用符号：

```python
model.body
range
int
bodies.add
model.geom
len
ValueError
mujoco.MjData
mujoco.mj_kinematics
_pose
(2 * model.geom_size[index]).tolist
float
np.isfinite(dimensions).all
np.isfinite
np.any
np.asarray
_native_ignored_pairs
policy.get
excluded.append
np.eye
data.geom_xmat[index].reshape
result.append
_native_geom_pose(model, index).tolist
_native_geom_pose
(np.linalg.inv(base) @ native_world).tolist
np.linalg.inv
```

## `_validate_fixed_installation`

[定义位置](../../../tests/planning/robot_fixture.py#L642)。

输入签名：

```python
def _validate_fixed_installation(fixture, models, sphere_geom_names, sphere_links):
```

实现说明：

> Rebuild installation identity, rigid placement and pair policy on reload.

显式输出分支：

```python
len(supplied)
```

读取的对象属性：

```python
np.array_equal
np.eye
np.allclose
```

写入目标：

```python
supplied
model
expected
actual
native
key
same
```

直接调用符号：

```python
np.array_equal
np.eye
ValueError
_fixed_installation_records
len
zip
set
np.allclose
```

## `fixed_installation_proxy`

[定义位置](../../../tests/planning/robot_fixture.py#L665)。

输入签名：

```python
def fixed_installation_proxy(record):
```

实现说明：

> Prepare a bounded, outward-rounded cuRobo shape without device access.
>
> Native cylinders are retained in the manifest for independent checking.
> cuRobo's default cylinder tessellation is inscribed, so this adapter uses
> a circumscribed 64-sided prism and checks its serialized float32 support
> planes. Its small, reported enlargement is collision padding only: no
> volume is added to observed-space evidence.

显式输出分支：

```python
{'kind': 'cuboid', 'arguments': {'name': record['geom'], 'pose': identity, 'dims': outward.tolist()}, 'approximation': {'kind': 'outward-rounded native box', 'maximum_face_expansion_m': float(((outward - dimensions) / 2).max())}}
{'kind': 'mesh', 'arguments': {'name': record['geom'], 'pose': identity, 'vertices': vertices.tolist(), 'faces': mesh.faces.tolist()}, 'approximation': {'kind': 'circumscribed native cylinder prism', 'sections': sections, 'maximum_radial_expansion_m': float(np.linalg.norm(vertices[:, :2], axis=1).max() - radius), 'maximum_axial_expansion_m': float(half_length - length / 2), 'minimum_radial_cover_allowance_m': radial_allowance, 'minimum_axial_cover_allowance_m': axial_allowance}}
```

读取的对象属性：

```python
np.asarray
dimensions.shape
np.isfinite(dimensions).all
np.isfinite
np.any
np.nextafter(dimensions.astype(np.float32), np.float32(np.inf)).astype
np.nextafter
dimensions.astype
np.float32
np.inf
np.isfinite(outward).all
outward.tolist
((outward - dimensions) / 2).max
np.cos
np.pi
trimesh.creation.cylinder
trimesh.creation
np.asarray(mesh.vertices, dtype=np.float32).astype
mesh.vertices
ConvexHull(vertices).equations
np.linalg.norm
np.linalg
(-planes[sides, 3] / radial[sides]).min
vertices[:, 2].max
vertices[:, 2].min
np.isfinite(vertices).all
np.abs
vertices.tolist
mesh.faces.tolist
mesh.faces
np.linalg.norm(vertices[:, :2], axis=1).max
```

写入目标：

```python
kind
dimensions
identity
outward
sections
radius
length
outer_radius
half_length
mesh
vertices
planes
radial
sides
radial_allowance
axial_allowance
```

直接调用符号：

```python
np.asarray
np.isfinite(dimensions).all
np.isfinite
np.any
ValueError
np.nextafter(dimensions.astype(np.float32), np.float32(np.inf)).astype
np.nextafter
dimensions.astype
np.float32
np.isfinite(outward).all
outward.tolist
float
((outward - dimensions) / 2).max
np.cos
trimesh.creation.cylinder
np.asarray(mesh.vertices, dtype=np.float32).astype
ConvexHull
np.linalg.norm
(-planes[sides, 3] / radial[sides]).min
min
vertices[:, 2].max
vertices[:, 2].min
np.isfinite(vertices).all
np.abs
vertices.tolist
mesh.faces.tolist
np.linalg.norm(vertices[:, :2], axis=1).max
```

## `create_fixed_installation_owners`

[定义位置](../../../tests/planning/robot_fixture.py#L713)。

输入签名：

```python
def create_fixed_installation_owners(fixture, *, device_cfg, T_output_base, sphere_indices_by_link):
```

实现说明：

> Bind validated robot primitives as existing complete static scene owners.
>
> The runner first calls validate_fixture_collision_contract. Link indices
> here come from the actual loaded cuRobo model, never from an assumed YAML
> or URDF ordering. These owners remain obstacles to payloads and every
> robot link whose native installation pair is enabled.

显式输出分支：

```python
(tuple(owners), reports)
```

读取的对象属性：

```python
np.asarray
dimensions.tolist
owners.append
np.array
np.stack
reports.append
pose.tolist
```

写入目标：

```python
base
records
expected_links
row
flattened
link
index
owners
reports
record
owner_id
native_source
dimensions
kind
primitive
query
pose
excluded
```

直接调用符号：

```python
_pose_array
set
ValueError
int
sorted
list
range
np.asarray
dimensions.tolist
float
FrozenPrimitives
tuple
owners.append
CollisionOwner
PosePath
np.array
np.stack
reports.append
pose.tolist
```

## `_validate_fixture_kinematics`

[定义位置](../../../tests/planning/robot_fixture.py#L779)。

输入签名：

```python
def _validate_fixture_kinematics(fixture, kinematics, models):
```

实现说明：

> Bind the consumed URDF's entire tree, not just the certified hinge.
>
> The current exporter has one explicit configuration shape. Extra loader
> fields can replace fixed transforms, add links or change joint offsets;
> accepting them would let the GPU move spheres using unproved kinematics.
> Feedback-dependent lock/default positions remain free because the proof
> covers every angle. All buffers remain the exporter's explicit zero.

显式输出分支：

无显式条目。

读取的对象属性：

```python
path.is_absolute
path.parent
buffers.values
set(kinematics['lock_joints']).issubset
fixture.get
position_limits.items
np.asarray
values.shape
values.dtype.kind
values.dtype
np.isfinite(values).all
np.isfinite
URDF.load
link.name
actual.robot.links
actual.robot
joint.name
actual.robot.joints
model.body(fixture['base_link']).id
model.body
model.body_parentid
model.body_jntnum
model.nbody
bodies.add
model.body(body).name
model.body_jntadr
model.jnt_type
mujoco.mjtJoint.mjJNT_HINGE
mujoco.mjtJoint
mujoco.mjtJoint.mjJNT_SLIDE
np.any
model.jnt_pos
model.jnt_limited
model.qpos0
model.jnt_qposadr
model.joint(index).name
model.joint
native_joints.add
model.jnt_axis
model.jnt_range
np.array_equal
expected_joints.add
model.body(model.body_parentid[body]).name
model.body_pos
model.body_quat
model.site(fixture['tcp_site']).id
model.site
model.site_bodyid
model.body(model.site_bodyid[site]).name
model.site_pos
model.site_quat
```

写入目标：

```python
fields
cspace_fields
path
buffers
value
joint_names
position_limits
name
bounds
values
actual
actual_links
link
actual_joints
joint
model
root
bodies
body
expected_links
expected_joints
native_joints
count
joint_name
kind
axis
limits
index
site
tcp_joint
```

直接调用符号：

```python
set
ValueError
Path
path.is_absolute
isinstance
any
buffers.values
len
set(kinematics['lock_joints']).issubset
fixture.get
position_limits.items
np.asarray
np.isfinite(values).all
np.isfinite
URDF.load
model.body
range
bodies.add
sorted
int
np.any
model.joint
native_joints.add
np.array_equal
expected_joints.add
require_joint
model.site
```

## `_validate_fixture_kinematics.require_joint`

[定义位置](../../../tests/planning/robot_fixture.py#L835)。

输入签名：

```python
def require_joint(name, parent, child, kind, position, quaternion, axis=None, limits=None):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
actual.joint_map.get
actual.joint_map
trimesh.transformations.compose_matrix
trimesh.transformations
trimesh.transformations.translation_from_matrix
trimesh.transformations.euler_from_matrix
joint.parent
joint.child
joint.type
joint.mimic
np.array_equal
joint.origin
joint.axis
joint.limit
joint.limit.lower
joint.limit.upper
```

写入目标：

```python
joint
pose
serialized_pose
```

直接调用符号：

```python
actual.joint_map.get
_pose
trimesh.transformations.compose_matrix
trimesh.transformations.translation_from_matrix
trimesh.transformations.euler_from_matrix
np.array_equal
ValueError
```

## `validate_fixture_collision_contract`

[定义位置](../../../tests/planning/robot_fixture.py#L900)。

输入签名：

```python
def validate_fixture_collision_contract(fixture, robot_config):
```

实现说明：

> Reload source geometry and re-prove every completed self query on CPU.
>
> A saved list of ignored names is never sufficient evidence. Every native
> enabled geometry is matched to its original cover input, the serialized
> spheres are matched to the enclosing-cell evidence, and the query contract
> is recomputed from BOTH the original source and exported robot model.
> Pose feedback/default joint positions are deliberately not certificate
> inputs: this certificate is valid throughout its stated hinge domain.

显式输出分支：

```python
{'sphere_count': fixture['sphere_count'], 'complete_native_geometry_count': len(records), 'complete_fixed_installation_geometry_count': fixed_count, 'native_enabled_sphere_pair_count': expected['native_enabled_sphere_pair_count'], 'mechanical_contact_pairs': expected['mechanical_contact_pairs'], 'mechanical_contact_link_exclusions': expected['mechanical_contact_link_exclusions'], 'mechanical_contact_sphere_pair_count': expected['mechanical_contact_sphere_pair_count'], 'certificate_completed_pairs': [row['links'] for row in expected['certificate_completed_pairs']], 'effective_query_sphere_pair_count': expected['effective_query_sphere_pair_count'], 'minimum_cell_enclosure_allowance_m': float(minimum_allowance), 'source_and_robot_certificate_equal': True, 'scope': 'CPU static geometry and self-query contract; no feedback-pose, scene or trajectory claim'}
```

读取的对象属性：

```python
ET.parse(fixture['source_scene_xml']).getroot
ET.parse
mujoco.MjModel.from_xml_path
mujoco.MjModel
np.inf
np.load
model.geom(row['geom']).id
model.geom
model.body(model.geom_bodyid[index]).name
model.body
model.geom_bodyid
np.array_equal
model.geom_contype
model.geom_conaffinity
centers.shape
vertex_offsets.shape
np.any
np.diff
vertices.ndim
vertices.shape
np.isfinite(value).all
np.isfinite
source.geom_type
mujoco.mjtGeom.mjGEOM_BOX
mujoco.mjtGeom
specification.get
source.geom_size
expected_pose[:3, :3].T
np.cumsum
np.concatenate
np.empty
np.linalg.norm(cell - centers[index], axis=1).max
np.linalg.norm
np.linalg
offsets.values
expected.items
```

写入目标：

```python
mechanical_contact_pairs
records
names
row
scene
enabled
kinematics
links
robot
source
fixed_count
offsets
name
minimum_allowance
saved
model
index
native
expected_pose
centers
radii
count
vertex_offsets
vertices
value
specification
is_box
cells
sites
rebuilt
cell
rebuilt_offsets
start
balls
ball
allowance
offsets[row['link']]
expected
source_expected
key
```

直接调用符号：

```python
ValueError
ET.parse(fixture['source_scene_xml']).getroot
ET.parse
_source_collision_names
len
set
mujoco.MjModel.from_xml_path
_validate_fixture_kinematics
_validate_fixed_installation
np.load
model.geom
_collision_geometry
_native_geom_pose
model.body
np.array_equal
np.any
np.diff
all
np.isfinite(value).all
np.isfinite
_cover_cells
specification.get
np.cumsum
np.concatenate
np.empty
range
np.linalg.norm(cell - centers[index], axis=1).max
np.linalg.norm
min
any
sum
offsets.values
_collision_query_contract
expected.items
_effective_ignored_pairs
float
```

## `export_fixture`

[定义位置](../../../tests/planning/robot_fixture.py#L1025)。

输入签名：

```python
def export_fixture(*, scene_path, catalog_path, episode_path, root_body, collision_root, tcp_site, arm_joint_names, finger_joint_names, velocity_limits, effort_limits, cover_specifications, initial_frame, output, binding_audit_path, max_acceleration, max_jerk, mechanical_contact_pairs):
```

实现说明：

> Build one exact native model and its explicitly budgeted collision cover.
>
> Velocity/effort limits and the initial recorded frame are explicit inputs;
> MJCF has no velocity-limit field. No coordinate is clipped or re-signed.
> Mechanical contact permissions are explicit native geom pairs with a
> model-level source; [] declares that there is no such permission.
> Unsupported source features fail before artifacts can be used for planning.

显式输出分支：

```python
manifest
```

读取的对象属性：

```python
Path(value).resolve
Path(output).resolve
output.mkdir
ET.parse(scene_path).getroot
ET.parse
model.nq
model.joint(i).name
model.joint
model.njnt
mujoco.mj_saveLastXML
scene.find
xml.getroot().find
xml.getroot
xml.getroot().remove
xml.getroot().append
xml.write
mujoco.MjModel.from_xml_path
mujoco.MjModel
json.loads
catalog_path.read_text
binding.append
h5py.File
np.asarray
np.unique
np.linspace
np.column_stack
mujoco.MjData
dict.fromkeys
model.body(model.geom_bodyid[model.geom(name).id]).name
model.body
model.geom_bodyid
model.geom(name).id
model.geom
mesh_directory.mkdir
model.body(root_body).id
model.nbody
model.body(body).name
model.body_ipos
model.body_iquat
model.body_mass
np.diag
model.body_inertia
np.flatnonzero
model.geom(int(geom_id)).name
collisions.append
covers.setdefault(name, []).extend
covers.setdefault
model.geom_type
mujoco.mjtGeom.mjGEOM_BOX
mujoco.mjtGeom
model.geom_size
trimesh.Trimesh(vertices=geometry['vertices'], faces=geometry['faces'], process=False).convex_hull.export
trimesh.Trimesh(vertices=geometry['vertices'], faces=geometry['faces'], process=False).convex_hull
trimesh.Trimesh
model.geom_pos
model.geom_quat
records.append
links.append
model.body_jntnum
model.body_parentid
model.body(model.body_parentid[body]).name
model.body_pos
model.body_quat
model.body_jntadr
model.joint(joint_id).name
np.any
model.jnt_pos
model.jnt_type
mujoco.mjtJoint.mjJNT_HINGE
mujoco.mjtJoint
mujoco.mjtJoint.mjJNT_SLIDE
model.qpos0
model.jnt_qposadr
model.jnt_limited
model.jnt_axis[joint_id].copy
model.jnt_axis
model.jnt_range
joints.append
model.site(tcp_site).id
model.site
model.body(model.site_bodyid[site_id]).name
model.site_bodyid
model.site_pos
model.site_quat
urdf.write_xml_file
URDF.load
model.joint(name).id
np.isfinite(q0).all
np.isfinite
np.random.default_rng(7341).uniform
np.random.default_rng
np.random
np.concatenate
data.qpos
model.joint(name).qposadr
mujoco.mj_kinematics
data.xpos
data.xquat
reference.update_cfg
reference.get_transform
np.linalg.norm
np.linalg
Rotation.from_matrix(predicted[:3, :3].T @ actual[:3, :3]).magnitude
Rotation.from_matrix
predicted[:3, :3].T
np.eye
data.site_xmat[site_id].reshape
data.site_xmat
data.site_xpos
Rotation.from_matrix(predicted_tcp[:3, :3].T @ actual_tcp[:3, :3]).magnitude
predicted_tcp[:3, :3].T
covers.values
q0[len(arm_joint_names):].tolist
q0.tolist
config_path.write_text
yaml.safe_dump
model.jnt_bodyid
model.body(name).id
finger_links.append
model.body(collision_root).id
Path(binding_audit_path).resolve
np.eye(4).tolist
model.jnt_range[model.joint(name).id].tolist
frames.tolist
manifest.update
json.dumps
yaml.safe_load
config_path.read_text
(output / 'manifest.json').write_text
```

写入目标：

```python
scene_path
catalog_path
episode_path
value
output
scene
enabled_geoms
assembly_geoms
names
model
i
robot_mjcf
xml
old
catalog
catalog_joints
joint
binding
name
row
stream
q0
frames
recorded
data
sphere_links
fixed_geometries
fixed_by_geom
record
links
joints
covers
records
mesh_directory
root_id
body
inertial
collisions
geom_id
geom_name
fixed
shape
spheres
geometry
mesh_path
record['collision_mesh_path']
joint_id
native
joint.name
joint.type
joint.axis
joint.limit
site_id
urdf
urdf_path
reference
limits
probes
probes[0]
probes[1]
max_position
max_rotation
sample
data.qpos[model.joint(name).qposadr[0]]
base
predicted
actual
actual_tcp
actual_tcp[:3, :3]
actual_tcp[:3, 3]
predicted_tcp
sphere_count
config
collision_contract
config['robot_cfg']['kinematics']['self_collision_ignore']
config_path
finger_roots
finger_links
ancestor
collision_root_id
cursor
manifest
manifest['collision_contract_validation']
```

直接调用符号：

```python
Path(value).resolve
Path
Path(output).resolve
output.mkdir
ET.parse(scene_path).getroot
ET.parse
_source_collision_names
set
ValueError
list
len
_robot_model
sorted
model.joint
range
mujoco.mj_saveLastXML
str
scene.find
xml.getroot().find
xml.getroot
xml.getroot().remove
xml.getroot().append
deepcopy
xml.write
mujoco.MjModel.from_xml_path
json.loads
catalog_path.read_text
binding.append
h5py.File
np.asarray
np.unique
np.linspace
np.column_stack
mujoco.MjData
dict.fromkeys
model.body
model.geom
_fixed_installation_records
mesh_directory.mkdir
Inertial
_pose
float
np.diag
np.flatnonzero
int
Geometry
Box
Cylinder
collisions.append
Collision
_geom_cover
covers.setdefault(name, []).extend
covers.setdefault
trimesh.Trimesh(vertices=geometry['vertices'], faces=geometry['faces'], process=False).convex_hull.export
trimesh.Trimesh
Mesh
records.append
links.append
Link
Joint
np.any
min
model.jnt_axis[joint_id].copy
Limit
joints.append
model.site
URDF
Robot
urdf.write_xml_file
URDF.load
np.isfinite(q0).all
np.isfinite
np.random.default_rng(7341).uniform
np.random.default_rng
np.concatenate
zip
mujoco.mj_kinematics
reference.update_cfg
dict
reference.get_transform
max
np.linalg.norm
Rotation.from_matrix(predicted[:3, :3].T @ actual[:3, :3]).magnitude
Rotation.from_matrix
np.eye
data.site_xmat[site_id].reshape
Rotation.from_matrix(predicted_tcp[:3, :3].T @ actual_tcp[:3, :3]).magnitude
sum
map
covers.values
_native_ignored_pairs
q0[len(arm_joint_names):].tolist
q0.tolist
_collision_query_contract
_effective_ignored_pairs
config_path.write_text
yaml.safe_dump
finger_links.append
Path(binding_audit_path).resolve
_pose_dict
np.eye(4).tolist
model.jnt_range[model.joint(name).id].tolist
frames.tolist
manifest.update
validate_fixture_collision_contract
json.dumps
yaml.safe_load
config_path.read_text
(output / 'manifest.json').write_text
```

## `bind_fixture_gripper_geometry`

[定义位置](../../../tests/planning/robot_fixture.py#L1271)。

输入签名：

```python
def bind_fixture_gripper_geometry(fixture, robot_cfg, sphere_indices_by_link, finger_joint_names):
```

实现说明：

> Map the finger assembly's exact native solids to their planner spheres.
>
> The nearest common ancestor of the named finger joints owns the palm and
> both finger branches. Reject any additional movable joint below that root
> so an arm or another mechanism cannot acquire hand contact permission.
> Only the isolated robot model is read; no task/object state is available.
> The caller has already validated the complete fixture collision contract.

显式输出分支：

```python
{'root_body': model.body(root).name, 'native_geometry_names': geometries, 'sphere_indices': selected, 'finger_joint_names': list(names), 'source': 'Isolated calibrated robot: common finger ancestor and complete native collision solids, with exact runtime sphere-cover indices'}
```

读取的对象属性：

```python
mujoco.MjModel.from_xml_path
mujoco.MjModel
model.joint(name).id
model.joint
model.jnt_bodyid
model.nbody
model.njnt
model.geom(geom).name
model.geom
model.ngeom
model.geom_bodyid
model.geom_contype
model.geom_conaffinity
model.geom(record['geom']).id
np.load
model.body(model.geom_bodyid[geom]).name
model.body
np.array_equal
selected.extend
geometries.append
model.body(root).name
```

写入目标：

```python
kinematics
links
indices
link
index
names
model
joints
name
chains
joint
roots
body
chain
root
bodies
expected
geom
offsets
selected
geometries
record
count
start
offsets[link]
saved
balls
ball
```

直接调用符号：

```python
int
set
len
sorted
list
range
any
ValueError
tuple
mujoco.MjModel.from_xml_path
model.joint
ancestors
all
model.geom
np.load
model.body
np.array_equal
selected.extend
geometries.append
```

## `bind_fixture_gripper_geometry.ancestors`

[定义位置](../../../tests/planning/robot_fixture.py#L1294)。

输入签名：

```python
def ancestors(body):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
result.append
model.body_parentid
```

写入目标：

```python
result
body
```

直接调用符号：

```python
result.append
int
```

## `bind_fixture_prismatic_self_geometry`

[定义位置](../../../tests/planning/robot_fixture.py#L1340)。

输入签名：

```python
def bind_fixture_prismatic_self_geometry(fixture, robot_cfg, geometry, feedback):
```

实现说明：

> Freeze native hand convex solids in the same zero-coordinate TCP frame.
>
> MuJoCo is used only during this fixture binding. The complete hand subtree
> was already checked by ``bind_fixture_gripper_geometry``; each source joint
> must be prismatic, and every native solid/sphere uses the same measured
> affine translation. Online code receives vertices and constant matrices,
> never a source model handle or callback.

显式输出分支：

```python
{'geometry': result, 'local_spheres': canonical_spheres, 'sphere_joint_translation': canonical_mapping, 'audit': {'position_tolerance_m': 2e-06, 'comparison': 'maximum absolute coordinate error over complete mechanical joint limits', 'geometry_errors': errors}}
```

读取的对象属性：

```python
mujoco.MjModel.from_xml_path
mujoco.MjModel
mujoco.MjData
model.njnt
model.joint(joint).name
model.joint
data.qpos
model.jnt_qposadr
model.joint(name).id
model.jnt_type
mujoco.mjtJoint.mjJNT_SLIDE
mujoco.mjtJoint
np.asarray
limits[:, 0].copy
coordinates.append
model.site(fixture['tcp_site']).id
model.site
model.geom(row['geom']).id
model.geom
np.load
saved['native_vertices'].copy
saved['sphere_centers_body'].copy
np.array_equal
records.append
mujoco.mj_forward
np.eye
data.site_xmat[site].reshape
data.site_xmat
data.site_xpos
model.geom_bodyid
data.geom_xmat[geom].reshape(3, 3).T
data.geom_xmat[geom].reshape
data.geom_xmat
data.geom_xpos
data.xmat[body].reshape(3, 3).T
data.xmat[body].reshape
data.xmat
data.xpos
pose_rows.append
snapshots.append
np.array
model.site_bodyid
np.zeros
np.allclose
model.jnt_bodyid
tcp[:3, :3].T
data.xaxis
limits.mean
np.abs
np.einsum
np.abs(delta_zero).max
np.abs(delta_mapping).max
bound.max
errors.append
result.append
zero_vertices.tolist
mapping.tolist
```

写入目标：

```python
names
partition
hand
selected
model
data
measured
joint
name
data.qpos[model.jnt_qposadr[joint]]
joints
limits
coordinates
column
value
value[column]
tool
T_frame_tcp
site
records
offsets
link
row
size
start
offsets[link]
ids
native_id
native
saved
vertices
centres
snapshots
fingers
frame
frame[:3, :3]
frame[:3, 3]
tcp
pose_rows
record
geom
body
world_vertices
world_centres
result
errors
canonical_spheres
canonical_mapping
hand_rows
index
tcp_ancestors
mapping
geom_ancestors
snapshot
shifts
joint_body
dependency
mapping[:, column]
zero_vertices
zero_centres
selected_rows
i
delta_zero
delta_mapping
middle
half
bound
diagnostic
canonical_spheres[selected_rows, :3]
canonical_mapping[selected_rows]
```

直接调用符号：

```python
tuple
bind_fixture_gripper_geometry
set
ValueError
mujoco.MjModel.from_xml_path
mujoco.MjData
dict
zip
range
model.joint
any
np.asarray
limits[:, 0].copy
len
coordinates.append
_pose
model.site
model.geom
_collision_geometry
np.load
saved['native_vertices'].copy
saved['sphere_centers_body'].copy
np.array_equal
_native_geom_pose
records.append
mujoco.mj_forward
np.eye
data.site_xmat[site].reshape
data.geom_xmat[geom].reshape
data.xmat[body].reshape
pose_rows.append
snapshots.append
np.array
enumerate
ancestors
np.zeros
np.allclose
int
limits.mean
np.abs
np.einsum
float
np.abs(delta_zero).max
np.abs(delta_mapping).max
bound.max
errors.append
result.append
list
zero_vertices.tolist
mapping.tolist
```

## `bind_fixture_prismatic_self_geometry.ancestors`

[定义位置](../../../tests/planning/robot_fixture.py#L1416)。

输入签名：

```python
def ancestors(body):
```

显式输出分支：

```python
values
```

读取的对象属性：

```python
values.add
model.body_parentid
```

写入目标：

```python
values
body
```

直接调用符号：

```python
set
values.add
int
```

## `bind_fixture_contact_materials`

[定义位置](../../../tests/planning/robot_fixture.py#L1478)。

输入签名：

```python
def bind_fixture_contact_materials(fixture, robot_cfg, sphere_indices_by_link, feedback, material_faces):
```

实现说明：

> Bind explicit pad faces without labeling any adjacent finger or palm.
>
> The caller supplies the sphere indices from its actual loaded kinematics.
> Per-link cover records are checked against both serialized YAML spheres
> and their saved native certificates before they acquire a material ID.
> Only the isolated robot MJCF is loaded; task-object truth is never read.

显式输出分支：

```python
{'robot_sphere_materials': sphere_materials, 'native_geometry_materials': bound_geometries, 'materials': materials, 'source': 'Explicit native robot box faces, exact cover certificates and runtime sphere identities; no object-contact permission inferred'}
```

读取的对象属性：

```python
mujoco.MjModel.from_xml_path
mujoco.MjModel
model.geom(record['geom']).id
model.geom
np.load
model.body(model.geom_bodyid[geom]).name
model.body
model.geom_bodyid
np.array_equal
np.eye
np.linalg.norm
np.linalg
model.geom_size
np.max
```

写入目标：

```python
kinematics
links
indices
link
index
declarations
row
model
materials
offsets
name
sphere_materials
bound_geometries
record
count
start
offsets[link]
declaration
material
geom
native
saved
centers
radii
balls
ball
normal_geom
pose
normal_body
face_body
material['normal_cover_excess_m']
selected
material['sphere_indices']
material['cover_certificate_path']
sphere_materials[index]
bound_geometries[record['geom']]
```

直接调用符号：

```python
set
ValueError
int
len
sorted
list
range
any
mujoco.MjModel.from_xml_path
calibrate_box_material_features
dict
zip
model.geom
_collision_geometry
np.load
model.body
np.array_equal
_native_geom_pose
np.eye
np.linalg.norm
float
np.max
```

## `main`

[定义位置](../../../tests/planning/robot_fixture.py#L1549)。

输入签名：

```python
def main():
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
json.loads
args.spec.read_text
args.spec
json.dumps
Path(json.loads(args.spec.read_text())['output']).resolve
```

写入目标：

```python
parser
args
manifest
```

直接调用符号：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
export_fixture
json.loads
args.spec.read_text
print
json.dumps
str
Path(json.loads(args.spec.read_text())['output']).resolve
Path
```
