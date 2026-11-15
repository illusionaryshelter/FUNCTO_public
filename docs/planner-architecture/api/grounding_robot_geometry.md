# `manipulation/grounding_robot_geometry.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_pose`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L32)。

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

## `_native_geom_pose`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L38)。

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

## `_native_ignored_pairs`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L48)。

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

## `_known_shapes`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L75)。

输入签名：

```python
def _known_shapes(fixture, model):
```

实现说明：

> Read explicitly registered entities from the compiled robot-only asset.
>
> Mesh vertices and their fixed local transforms are the model's original
> collision representation. Boxes and cylinders stay analytic. The second
> result retains geometry-local coordinates for shadow padding about each
> entity's own origin.

显式输出分支：

```python
(shapes, local_records)
```

读取的对象属性：

```python
model.geom(name).id
model.geom
model.body(model.geom_bodyid[index]).name
model.body
model.geom_bodyid
model.geom_type
mujoco.mjtGeom.mjGEOM_BOX
mujoco.mjtGeom
mujoco.mjtGeom.mjGEOM_CYLINDER
model.geom_size
shapes.append
mujoco.mjtGeom.mjGEOM_MESH
local[:3, :3].T
local_records.append
```

写入目标：

```python
shapes
local_records
row
name
frame
index
local
source
kind
size
label
dimensions
fields
native
vertices
```

直接调用符号：

```python
model.geom
model.body
ValueError
_native_geom_pose
tuple
shapes.append
NativePrimitive
dict
_collision_geometry
NativeSolid
local_records.append
```

## `robot_self_filter_shapes`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L110)。

输入签名：

```python
def robot_self_filter_shapes(fixture, frame_poses):
```

实现说明：

> Place the same asset's original components at image-capture joint FK.

显式输出分支：

```python
tuple((RobotCollisionShape(T_world_geometry=np.asarray(frame_poses[frame]) @ local, **fields) for (frame, local, fields) in records))
```

读取的对象属性：

```python
mujoco.MjModel.from_xml_path
mujoco.MjModel
np.asarray
```

写入目标：

```python
model
_
records
frame
local
fields
```

直接调用符号：

```python
mujoco.MjModel.from_xml_path
_known_shapes
tuple
RobotCollisionShape
np.asarray
```

## `tool_geometry_names`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L121)。

输入签名：

```python
def tool_geometry_names(fixture):
```

实现说明：

> Return registered geometry under the named finger mechanism's root.
>
> This source adapter uses the fixture's declared finger joints to identify
> their common parent assembly. The production scene receives only explicit
> registered geometry IDs; it has no finger-count or robot-name convention.

显式输出分支：

```python
tuple((row['geom'] for row in fixture['collision_geometries'] if root in ancestors(int(model.geom_bodyid[model.geom(row['geom']).id]))))
```

读取的对象属性：

```python
mujoco.MjModel.from_xml_path
mujoco.MjModel
model.body_parentid
model.jnt_bodyid
model.joint(name).id
model.joint
model.geom_bodyid
model.geom(row['geom']).id
model.geom
```

写入目标：

```python
model
chains
name
root
body
chain
row
```

直接调用符号：

```python
mujoco.MjModel.from_xml_path
ancestors
int
model.joint
next
all
tuple
model.geom
```

## `tool_geometry_names.ancestors`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L130)。

输入签名：

```python
def ancestors(body):
```

显式输出分支：

```python
chain
```

读取的对象属性：

```python
chain.append
model.body_parentid
```

写入目标：

```python
chain
body
```

直接调用符号：

```python
chain.append
int
```

## `_native_pair_policies`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L145)。

输入签名：

```python
def _native_pair_policies(model, geometry_names):
```

实现说明：

> Retain native body filtering and each geom's exact bitmask decision.
>
> Link-level policy is already defined by the maintained fixture adapter.
> Here individual geom masks additionally preserve excluded pairs inside a
> partially enabled link pair. Sphere separation certificates are deliberately
> absent: their native solids still require an actual distance query.

显式输出分支：

```python
result
```

读取的对象属性：

```python
mujoco.mjtDisableBit.mjDSBL_CONTACT
mujoco.mjtDisableBit
mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
model.opt.disableflags
model.opt
model.geom_bodyid
model.geom(name).id
model.geom
model.body(body).name
model.body
bodies.items
dict.fromkeys
links.values
model.geom(first).id
model.geom(second).id
model.geom_contype
model.geom_conaffinity
ignored.get
reasons.append
result.append
'; '.join
```

写入目标：

```python
disabled
bodies
name
links
body
ignored
result
first
second
a
b
link_a
link_b
masked
body_excluded
reasons
```

直接调用符号：

```python
int
ValueError
model.geom
model.body
bodies.items
_native_ignored_pairs
list
dict.fromkeys
links.values
combinations
ignored.get
reasons.append
result.append
'; '.join
```

## `build_robot_validator`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L181)。

输入签名：

```python
def build_robot_validator(fixture, *, reference_frame, T_reference_base, tool_frame, T_frame_tcp):
```

实现说明：

> Build the mandatory CPU gate from the same complete source robot model.
>
> # BEGIN the robot pose channel is calibrated FK, not a native pose sensor.
> Calibration identity is explicit. The recording backend estimates the
> world-frame TCP from measured robot joints and the known fixed model, with
> an identity link-from-TCP transform. This independent model checks that FK
> convention; a caller cannot relabel the frame or silently add an offset.
> # END matching models do not establish independent physical TCP accuracy.

显式输出分支：

```python
RobotStateValidator(checker, joint_names=fixture['joint_names'], arm_joint_names=fixture['arm_joint_names'], finger_joint_names=fixture['finger_joint_names'], pairs=pairs, named_velocity_limits=fixture['velocity_limits'], reference_frame=reference_frame, T_reference_base=T_reference_base, tool_frame=tool_frame, T_frame_tcp=T_frame_tcp, fk_translation_tolerance_m=1e-08, fk_rotation_tolerance_rad=1e-08, source=f"Known robot model {fixture['robot_mjcf_path']}; {len(fixture['collision_geometries'])} robot entities and {len(fixture['fixed_collision_geometries'])} fixed installations")
```

读取的对象属性：

```python
mujoco.MjModel.from_xml_path
mujoco.MjModel
model.nq
model.nbody
model.ngeom
np.array_equal
shape.name
permitted.get
pairs.append
checker.joint_limits
model.jnt_range
model.joint(name).id
model.joint
```

写入目标：

```python
model
expected_tool_transform
shapes
_
geometry_names
shape
policy
permitted
row
pairs
first
second
identity
reason
declaration
checker
name
```

直接调用符号：

```python
mujoco.MjModel.from_xml_path
ValueError
_pose
np.array_equal
_known_shapes
tuple
_native_pair_policies
frozenset
permitted.get
pairs.append
PairQuery
RobotGeometryChecker
model.joint
RobotStateValidator
len
```

## `_model_q`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L240)。

输入签名：

```python
def _model_q(model, names, positions):
```

实现说明：

> Place the named encoder coordinates in the fixed model's q ordering.

显式输出分支：

```python
q
```

读取的对象属性：

```python
pin.neutral
model.joints[model.getJointId(name)].idx_q
model.joints
model.getJointId
```

写入目标：

```python
q
name
value
q[model.joints[model.getJointId(name)].idx_q]
```

直接调用符号：

```python
pin.neutral
zip
model.getJointId
```

## `_frame_placements`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L247)。

输入签名：

```python
def _frame_placements(model, data, q):
```

实现说明：

> Return model-base FK for every registered frame at an explicit state.

显式输出分支：

```python
data.oMf
```

读取的对象属性：

```python
pin.forwardKinematics
pin.updateFramePlacements
data.oMf
```

写入目标：

无显式条目。

直接调用符号：

```python
pin.forwardKinematics
pin.updateFramePlacements
```

## `_depends_on_joint`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L253)。

输入签名：

```python
def _depends_on_joint(model, joint, selected):
```

实现说明：

> Test mechanism membership from the original kinematic parent chain.

显式输出分支：

```python
True
False
```

读取的对象属性：

```python
model.parents
```

写入目标：

```python
joint
```

直接调用符号：

```python
int
```

## `_coal_geometry`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L261)。

输入签名：

```python
def _coal_geometry(kind, dimensions, vertices):
```

实现说明：

> Rebuild the original analytic primitive or original native convex solid.

显式输出分支：

```python
coal.Box(*dimensions)
coal.Cylinder(*dimensions)
_convex_solid(vertices)
```

读取的对象属性：

```python
coal.Box
coal.Cylinder
```

写入目标：

无显式条目。

直接调用符号：

```python
coal.Box
coal.Cylinder
_convex_solid
ValueError
```

## `build_tool_parts`

[定义位置](../../../manipulation/grounding_robot_geometry.py#L271)。

输入签名：

```python
def build_tool_parts(fixture, feedback, entrance_finger_positions, native_finger_positions, tool_geometry):
```

实现说明：

> Bind TCP-fixed geometry and declared finger states to the official asset.
>
> A frame with the TCP's exact ``parentJoint`` has a constant joint-local
> placement, including any fixed intervening frames.  Such parts require no
> arm IK.  Finger descendants are placed by their declared two slide states;
> their model-base FK cancels when converted into the same TCP coordinates.
> Other movable arm links do not belong to this local rigid-motion model.
>
> The robot and end effector are the fixed official configuration reused
> across tasks.  Their shapes and fixed transforms therefore remain valid;
> calibrated encoders place them at capture, independently of task objects.

显式输出分支：

```python
(parts, states, capture_frames, component_records, cache_arrays, {'seconds': perf_counter() - started, 'tcp_frame': fixture['tool_link'], 'tcp_parent_joint': tcp_parent, 'rigid_geometry_ids': [part['geometry_id'] for part in parts if part['state_id'] == 'tcp_fixed'], 'finger_geometry_ids': list(dict.fromkeys((part['geometry_id'] for part in parts if part['state_id'] != 'tcp_fixed'))), 'source': fixture['source'], 'robot_mjcf_path': fixture['robot_mjcf_path'], 'kinematics_urdf_path': fixture['urdf_path'], 'T_frame_tcp': T_frame_tcp.tolist(), 'T_frame_tcp_source': 'capture feedback tool_transform; fixed official fixture tool_transform declaration', 'fixture_tool_transform': fixture['tool_transform'], 'guide_definition': 'original MaterialSamples points plus original native polytope vertices or on-analytic-surface display/guide samples', 'capture_frame_pose_source': 'named image-capture joint encoders, calibration and official FK'})
```

读取的对象属性：

```python
mujoco.MjModel.from_xml_path
mujoco.MjModel
pin.buildModelFromUrdf
model.createData
model.frames
model.getFrameId
tcp_frame.parentJoint
model.getJointId
np.asarray
np.linalg.inv
np.linalg
dict.fromkeys
capture_placements[model.getFrameId(name)].homogeneous
np.asarray(native_finger_positions).tolist
states.values
jaw.at
frame.parentJoint
native.geom(declaration['geom']).id
native.geom
native.geom_type
mujoco.mjtGeom.mjGEOM_CYLINDER
mujoco.mjtGeom
native.geom_size
trimesh.creation.cylinder
trimesh.creation
display.vertices
display.faces
mujoco.mjtGeom.mjGEOM_BOX
(2 * native.geom_size[geom_id]).tolist
np.unique
np.concatenate
material.points
bound_states.items
tcp_frame.placement.inverse
tcp_frame.placement
frame.placement
initial_q.copy
model.joints[model.getJointId(name)].idx_q
model.joints
placed[model.getFrameId(fixture['tool_link'])].inverse
tool_frame.homogeneous
transform[:3, :3].T
parts.append
component_records.append
transform.tolist
vertices_tool.tolist
faces.tolist
cache_arrays.update
T_frame_tcp.tolist
```

写入目标：

```python
started
native
model
data
tcp_frame
tcp_parent
finger_joints
name
initial_q
T_frame_tcp
T_tcp_frame
capture_placements
frame_names
row
world_base
capture_frames
states
jaw
state
state['signed_pad_gap_m']
parts
cache_arrays
component_records
declaration
frame
fixed
finger
geom_id
native_transform
native_kind
kind
dimensions
display
vertices
faces
display_source
native_geometry
geometry
material
guide_local
source
bound_states
state_id
tool_frame
q
value
q[model.joints[model.getJointId(name)].idx_q]
placed
transform
vertices_tool
guide_tool
identity
part
index
```

直接调用符号：

```python
perf_counter
mujoco.MjModel.from_xml_path
pin.buildModelFromUrdf
model.createData
model.getFrameId
int
model.getJointId
_model_q
np.asarray
np.linalg.inv
_frame_placements
tuple
dict.fromkeys
list
np.asarray(native_finger_positions).tolist
states.values
float
jaw.at
_depends_on_joint
native.geom
_native_geom_pose
trimesh.creation.cylinder
_collision_geometry
(2 * native.geom_size[geom_id]).tolist
_coal_geometry
MaterialSamples
np.unique
np.concatenate
bound_states.items
tcp_frame.placement.inverse
initial_q.copy
zip
placed[model.getFrameId(fixture['tool_link'])].inverse
dict
len
parts.append
component_records.append
transform.tolist
vertices_tool.tolist
faces.tolist
cache_arrays.update
T_frame_tcp.tolist
```
