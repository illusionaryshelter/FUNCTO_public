# `tests/planning/robot_fixture.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_pose`

[定义位置](../../../tests/planning/robot_fixture.py#L38)。

输入签名：

```python
def _pose(position, quaternion):
```

输出分支的源码表达式：

```python
value
```

直接写入的状态或数组位置：`value[:3, :3]`、`value[:3, 3]`。

直接调用：

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

输出分支的源码表达式：

```python
{'position': transform[:3, 3].tolist(), 'quaternion_wxyz': Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()}
```

直接调用：

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

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result[:3, :3]`、`result[:3, 3]`。

直接调用：

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

输出分支的源码表达式：

```python
([-box_halfsize + (np.asarray(index) + 0.5) * step + offsets for index in product(*(range(value) for value in bins))], None)
(cells, sites)
```

直接调用：

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

输出分支的源码表达式：

```python
(fitted.x[:3] * scale + origin, {'constraint_violation_m': max(0.0, -residual) * scale, 'initial_support_plane_excess_m': initial_excess * scale, 'fitted_support_plane_excess_m': measured_excess * scale})
```

直接调用：

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
def _fit_cell_sphere_center.constraints(value):
```

输出分支的源码表达式：

```python
np.r_[value[3] - np.linalg.norm(points - value[:3], axis=1), value[4] - normals @ value[:3] - constants - value[3]]
```

直接调用：

```python
np.linalg.norm
```

## `_fit_cell_sphere_center.jacobian`

[定义位置](../../../tests/planning/robot_fixture.py#L141)。

输入签名：

```python
def _fit_cell_sphere_center.jacobian(value):
```

输出分支的源码表达式：

```python
np.vstack((np.column_stack((delta / distance[:, None], np.ones(len(points)), np.zeros(len(points)))), np.column_stack((-normals, -np.ones(len(normals)), np.ones(len(normals))))))
```

直接调用：

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

输出分支的源码表达式：

```python
(spheres, record, geometry)
```

直接调用：

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

直接读取的对象属性：`result.setdefault`。

输出分支的源码表达式：

```python
result
```

直接调用：

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

输出分支的源码表达式：

```python
bounds
```

直接调用：

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

输出分支的源码表达式：

```python
[[w * w + x * x - y * y - z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), w * w - x * x + y * y - z * z, 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), w * w - x * x - y * y + z * z]]
```

直接调用：

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

输出分支的源码表达式：

```python
None
{'kind': 'direct_hinge_complete_solid_axial_separation', 'joint': model.joint(joint).name, 'parent_body': model.body(parent).name, 'child_body': model.body(child).name, 'joint_axis_child': model.jnt_axis[joint].tolist(), 'joint_pivot_child': model.jnt_pos[joint].tolist(), 'child_position_parent': model.body_pos[child].tolist(), 'child_quaternion_parent_wxyz': model.body_quat[child].tolist(), 'native_joint_range': model.jnt_range[joint].tolist(), 'domain': 'all real hinge angles; other joints and common rigid motion unrestricted', 'all_enabled_geometries': records, 'native_geom_pair_parameters': pair_parameters, 'body_axis_projection_intervals_m': intervals, 'separation_direction': 'child_above_parent' if direction == 0 else 'parent_above_child', 'separation_lower_bound_m': _interval_bounds(separation)[0], 'required_distance_upper_bound_m': _interval_bounds(required)[1], 'certified_slack_lower_bound_m': _interval_bounds(separation - required)[0], 'margin_override_enabled': override, 'option_o_margin_m': float(model.opt.o_margin), 'native_ccd_tolerance_m': float(model.opt.ccd_tolerance), 'arithmetic': 'local mpmath.iv, 113 bits; compiled binary inputs; outward binary64 endpoints', 'native_pair_remains_enabled': True, 'scope': 'this native self pair only; no environment, object or other-link query removed'}
```

直接写入的状态或数组位置：`context.prec`。

直接调用：

```python
int
np.any
np.flatnonzero
any
len
MPIntervalContext
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
zip
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

输出分支的源码表达式：

```python
(declared, exclusions)
```

直接调用：

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
def _mechanical_contact_contract.enabled(first, second):
```

输出分支的源码表达式：

```python
bool(model.geom_contype[first] & model.geom_conaffinity[second] or model.geom_contype[second] & model.geom_conaffinity[first])
```

直接调用：

```python
bool
```

## `_collision_query_contract`

[定义位置](../../../tests/planning/robot_fixture.py#L501)。

输入签名：

```python
def _collision_query_contract(model, kinematics, *, mechanical_contact_pairs):
```

输出分支的源码表达式：

```python
{'native_policy_exclusions': native, 'native_enabled_sphere_pair_count': original_count, 'mechanical_contact_pairs': mechanical, 'mechanical_contact_link_exclusions': contact_exclusions, 'mechanical_contact_sphere_pair_count': sum((len(spheres[first]) * len(spheres[second]) for (first, others) in contact_exclusions.items() for second in others)), 'certificate_completed_pairs': completed, 'effective_query_pairs': queries, 'effective_query_sphere_pair_count': sum((len(spheres[a]) * len(spheres[b]) for (a, b) in queries))}
```

直接调用：

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

直接读取的对象属性：`result.setdefault`。

输出分支的源码表达式：

```python
result
```

直接调用：

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

直接读取的对象属性：`scene.find`、`scene.iter`。

输出分支的源码表达式：

```python
{node.get('name') for node in source_root.iter('geom') if int(node.get('contype', '1')) or int(node.get('conaffinity', '1'))}
```

直接调用：

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

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`native_world[:3, :3]`、`native_world[:3, 3]`。

直接调用：

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

输出分支的源码表达式：

```python
len(supplied)
```

直接调用：

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

输出分支的源码表达式：

```python
{'kind': 'cuboid', 'arguments': {'name': record['geom'], 'pose': identity, 'dims': outward.tolist()}, 'approximation': {'kind': 'outward-rounded native box', 'maximum_face_expansion_m': float(((outward - dimensions) / 2).max())}}
{'kind': 'mesh', 'arguments': {'name': record['geom'], 'pose': identity, 'vertices': vertices.tolist(), 'faces': mesh.faces.tolist()}, 'approximation': {'kind': 'circumscribed native cylinder prism', 'sections': sections, 'maximum_radial_expansion_m': float(np.linalg.norm(vertices[:, :2], axis=1).max() - radius), 'maximum_axial_expansion_m': float(half_length - length / 2), 'minimum_radial_cover_allowance_m': radial_allowance, 'minimum_axial_cover_allowance_m': axial_allowance}}
```

直接调用：

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

输出分支的源码表达式：

```python
(tuple(owners), reports)
```

直接调用：

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

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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
def _validate_fixture_kinematics.require_joint(name, parent, child, kind, position, quaternion, axis=None, limits=None):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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

输出分支的源码表达式：

```python
{'sphere_count': fixture['sphere_count'], 'complete_native_geometry_count': len(records), 'complete_fixed_installation_geometry_count': fixed_count, 'native_enabled_sphere_pair_count': expected['native_enabled_sphere_pair_count'], 'mechanical_contact_pairs': expected['mechanical_contact_pairs'], 'mechanical_contact_link_exclusions': expected['mechanical_contact_link_exclusions'], 'mechanical_contact_sphere_pair_count': expected['mechanical_contact_sphere_pair_count'], 'certificate_completed_pairs': [row['links'] for row in expected['certificate_completed_pairs']], 'effective_query_sphere_pair_count': expected['effective_query_sphere_pair_count'], 'minimum_cell_enclosure_allowance_m': float(minimum_allowance), 'source_and_robot_certificate_equal': True, 'scope': 'CPU static geometry and self-query contract; no feedback-pose, scene or trajectory claim'}
```

直接写入的状态或数组位置：`offsets[row['link']]`。

直接调用：

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

直接读取的对象属性：`scene.find`。

输出分支的源码表达式：

```python
manifest
```

直接写入的状态或数组位置：`record['collision_mesh_path']`、`joint.name`、`joint.type`、`joint.axis`、`joint.limit`、`probes[0]`、`probes[1]`、`data.qpos[model.joint(name).qposadr[0]]`、`actual_tcp[:3, :3]`、`actual_tcp[:3, 3]`、`config['robot_cfg']['kinematics']['self_collision_ignore']`、`manifest['collision_contract_validation']`。

直接调用：

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

输出分支的源码表达式：

```python
{'root_body': model.body(root).name, 'native_geometry_names': geometries, 'sphere_indices': selected, 'finger_joint_names': list(names), 'source': 'Isolated calibrated robot: common finger ancestor and complete native collision solids, with exact runtime sphere-cover indices'}
```

直接写入的状态或数组位置：`offsets[link]`。

直接调用：

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
def bind_fixture_gripper_geometry.ancestors(body):
```

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
result
```

直接调用：

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

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
{'geometry': result, 'local_spheres': canonical_spheres, 'sphere_joint_translation': canonical_mapping, 'audit': {'position_tolerance_m': 2e-06, 'comparison': 'maximum absolute coordinate error over complete mechanical joint limits', 'geometry_errors': errors}}
```

直接写入的状态或数组位置：`data.qpos[model.jnt_qposadr[joint]]`、`value[column]`、`offsets[link]`、`frame[:3, :3]`、`frame[:3, 3]`、`mapping[:, column]`、`canonical_spheres[selected_rows, :3]`、`canonical_mapping[selected_rows]`。

直接调用：

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
def bind_fixture_prismatic_self_geometry.ancestors(body):
```

输出分支的源码表达式：

```python
values
```

直接调用：

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

输出分支的源码表达式：

```python
{'robot_sphere_materials': sphere_materials, 'native_geometry_materials': bound_geometries, 'materials': materials, 'source': 'Explicit native robot box faces, exact cover certificates and runtime sphere identities; no object-contact permission inferred'}
```

直接写入的状态或数组位置：`offsets[link]`、`material['normal_cover_excess_m']`、`material['sphere_indices']`、`material['cover_certificate_path']`、`sphere_materials[index]`、`bound_geometries[record['geom']]`。

直接调用：

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

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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
