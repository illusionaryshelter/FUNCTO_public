# `manipulation/tool_geometry.py` 实现契约

代码基线：`ae2603f`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_pose`

[定义位置](../../../manipulation/tool_geometry.py#L19)。

输入签名：

```python
def _pose(position, rotation):
```

显式输出分支：

```python
value
```

读取的对象属性：

```python
np.eye
np.asarray(rotation, dtype=float).reshape
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
np.asarray(rotation, dtype=float).reshape
np.asarray
```

## `_convex_planes`

[定义位置](../../../manipulation/tool_geometry.py#L26)。

输入签名：

```python
def _convex_planes(vertices):
```

实现说明：

> Use the actual collision vertices, not a fitted axis-aligned box.
>
> MuJoCo uses convex collision geometry for mesh geoms. SciPy's hull therefore
> gives the corresponding solid halfspaces, ``n @ p + d <= 0``. Normalize the
> equations explicitly so their signed plane values have metric units. They
> are halfspace tests, not a claimed Euclidean mesh signed-distance field.

显式输出分支：

```python
(hull, planes)
```

读取的对象属性：

```python
np.asarray(hull.equations, dtype=float).copy
np.asarray
hull.equations
np.linalg.norm
np.linalg
```

写入目标：

```python
hull
planes
```

直接调用符号：

```python
ConvexHull
np.asarray(hull.equations, dtype=float).copy
np.asarray
np.linalg.norm
```

## `_collision_geometry`

[定义位置](../../../manipulation/tool_geometry.py#L40)。

输入签名：

```python
def _collision_geometry(model, geom_id, role):
```

实现说明：

> Extract a compiled geom-local mesh and its native convex collision solid.
>
> MuJoCo's compiled ``mesh_vert`` already includes asset scaling and mesh
> recentering / principal-axis rotation. The matching compiled geom transform
> includes the compensating mesh pose. Applying ``mesh_pos``, ``mesh_quat`` or
> asset scale again would move the hand and fingers twice. Callers must pair
> these vertices with ``data.geom_xpos / geom_xmat`` from the same model.

显式输出分支：

```python
{'geom_id': int(geom_id), 'name': model.geom(geom_id).name, 'role': role, 'vertices': vertices, 'faces': faces, 'halfspaces': planes, 'collision_representation': representation}
```

读取的对象属性：

```python
model.geom_type
mujoco.mjtGeom.mjGEOM_MESH
mujoco.mjtGeom
model.geom_dataid
model.mesh_vertadr
model.mesh_vertnum
model.mesh_faceadr
model.mesh_facenum
np.asarray(model.mesh_vert[vertex_start:vertex_start + vertex_count], dtype=float).copy
np.asarray
model.mesh_vert
np.asarray(model.mesh_face[face_start:face_start + face_count], dtype=np.int64).copy
model.mesh_face
np.int64
mujoco.mjtGeom.mjGEOM_BOX
np.array
model.geom_size
np.asarray(hull.simplices, dtype=np.int64).copy
hull.simplices
np.cross
np.einsum
faces.ndim
faces.shape
np.any
model.geom(geom_id).name
model.geom
```

写入目标：

```python
kind
mesh_id
vertex_start
vertex_count
face_start
face_count
vertices
faces
representation
_
planes
x
y
z
hull
cross
flip
faces[flip]
```

直接调用符号：

```python
int
np.asarray(model.mesh_vert[vertex_start:vertex_start + vertex_count], dtype=float).copy
np.asarray
np.asarray(model.mesh_face[face_start:face_start + face_count], dtype=np.int64).copy
_convex_planes
np.array
np.asarray(hull.simplices, dtype=np.int64).copy
np.cross
np.einsum
ValueError
np.any
len
model.geom
```

## `ParallelJawToolGeometry`

[定义位置](../../../manipulation/tool_geometry.py#L94)。

```python
class ParallelJawToolGeometry():
```

类字段：

无显式条目。

实现说明：

> A standalone two-slide tool with explicitly identified opposing pads.
>
> Construct this through the robot-specific asset factory below. The object
> owns its FK scratch data; it does not read or mutate a simulator's live
> state. Static collision meshes stay geom-local. ``at(q)`` supplies their
> current transforms in the named tool frame without copying the meshes.

## `ParallelJawToolGeometry.__init__`

[定义位置](../../../manipulation/tool_geometry.py#L103)。

输入签名：

```python
def __init__(self, model, *, tool_id, tool_frame, native_tool_site, joint_names, collision_roles, pad_names, open_joint_positions_m, closed_joint_positions_m, asset_path, native_grasp_body):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
value.strip
model.nq
mujoco.MjData
model.site(native_tool_site).id
model.site
model.body(native_grasp_body).id
model.body
np.array
model.joint(name).id
model.joint
np.any
model.jnt_type
self._joint_ids
mujoco.mjtJoint.mjJNT_SLIDE
mujoco.mjtJoint
np.all
model.jnt_limited
model.jnt_qposadr[self._joint_ids].copy
model.jnt_qposadr
model.jnt_range[self._joint_ids].copy
model.jnt_range
self._positions
model.geom(name).id
model.geom
model.geom_type
mujoco.mjtGeom.mjGEOM_BOX
mujoco.mjtGeom
self._pad_ids
self._forward
self.open_joint_positions_m
self._data.geom_xpos
self._data
self._data.geom_xmat
np.linalg.norm
np.linalg
self._data.xaxis
np.allclose
np.abs
self.joint_axes_tool
self.closing_axis_tool
pose[:3, :3].T
np.argmax
np.isclose
self._pad_faces.append
self._pad_faces
np.sign
self.at
self.closed_joint_positions_m
```

写入目标：

```python
label
value
self.tool_id
self.tool_frame
self.asset_path
self.joint_names
self._model
self._data
self._site
self._grasp_body
self._joint_ids
name
self._qpos_addresses
self.joint_ranges_m
self.open_joint_positions_m
self.closed_joint_positions_m
self.collision_geometries
role
self._pad_ids
index
tool_inverse
pad_poses
between
self.closing_axis_tool
self.joint_axes_tool
self._pad_faces
side
pose
inward
projections
axis
opened
closed
self.signed_gap_range_m
```

直接调用符号：

```python
isinstance
value.strip
ValueError
len
str
tuple
mujoco.MjData
model.site
model.body
np.array
model.joint
np.any
np.all
model.jnt_qposadr[self._joint_ids].copy
model.jnt_range[self._joint_ids].copy
self._positions
_collision_geometry
model.geom
any
self._forward
_pose
np.linalg.norm
np.allclose
np.abs
enumerate
int
np.argmax
np.isclose
abs
self._pad_faces.append
float
np.sign
self.at
max
```

## `ParallelJawToolGeometry._positions`

[定义位置](../../../manipulation/tool_geometry.py#L174)。

输入签名：

```python
def _positions(self, value):
```

显式输出分支：

```python
values
```

读取的对象属性：

```python
np.asarray
values.shape
values.dtype.kind
values.dtype
np.isfinite(values).all
np.isfinite
values.astype
np.any
self.joint_ranges_m
```

写入目标：

```python
values
```

直接调用符号：

```python
np.asarray
np.isfinite(values).all
np.isfinite
ValueError
values.astype
np.any
```

## `ParallelJawToolGeometry._forward`

[定义位置](../../../manipulation/tool_geometry.py#L185)。

输入签名：

```python
def _forward(self, positions):
```

显式输出分支：

```python
np.linalg.inv(_pose(self._data.site_xpos[self._site], self._data.site_xmat[self._site]))
```

读取的对象属性：

```python
self._data.qpos
self._data
self._qpos_addresses
mujoco.mj_kinematics
self._model
np.linalg.inv
np.linalg
self._data.site_xpos
self._site
self._data.site_xmat
```

写入目标：

```python
self._data.qpos[self._qpos_addresses]
```

直接调用符号：

```python
mujoco.mj_kinematics
np.linalg.inv
_pose
```

## `ParallelJawToolGeometry.at`

[定义位置](../../../manipulation/tool_geometry.py#L195)。

输入签名：

```python
def at(self, joint_positions_m):
```

实现说明：

> Return exact tool-local geometry for an explicit native joint tuple.
>
> ``T_tool_pad`` has +z pointing into the jaw opening and x/y tangent to
> its actual inner contact face. ``signed_gap_m`` is separation of those
> faces, not q1-q2: native soft-contact pads may overlap at a joint limit.
> A negative native gap is reported honestly, never silently clamped.
> No force-closure, collision-free approach, or stable grasp is certified.

显式输出分支：

```python
{'tool_id': self.tool_id, 'tool_frame': self.tool_frame, 'joint_positions_m': positions, 'geoms': geoms, 'pads': pads, 'signed_gap_m': gap}
```

读取的对象属性：

```python
self._positions
self._forward
self.collision_geometries
self._data.geom_xpos
self._data
self._data.geom_xmat
geoms.append
self._pad_ids
self._pad_faces
np.cross
self._model.geom_size
self._model
pads.append
np.column_stack
self._model.geom_size[index, tangent_axes].copy
self.tool_id
self.tool_frame
```

写入目标：

```python
positions
tool_inverse
geoms
by_id
geometry
index
pose
by_id[index]
pads
side
face
axis
sign
normal
tangent_axes
other
x_axis
y_axis
center
gap
```

直接调用符号：

```python
self._positions
self._forward
_pose
geoms.append
enumerate
zip
range
np.cross
pads.append
np.column_stack
self._model.geom_size[index, tangent_axes].copy
float
```

## `ParallelJawToolGeometry.grasp_generation_geometry`

[定义位置](../../../manipulation/tool_geometry.py#L243)。

输入签名：

```python
def grasp_generation_geometry(self):
```

实现说明：

> Export native geometry in a +X-closing, +Z-approach base frame.
>
> The factory's grasp body already declares those axis conventions. This
> method verifies them against the actual pad faces and computes the two
> inner boxes from their material corners, at fully open and midpoint
> native joint coordinates. It certifies neither target contact nor a
> collision-free grasp. The output contains no scene or object evidence.

显式输出分支：

```python
{'T_tool_grasp': base.copy(), 'sweep_volume_params': {'extents_open': opened, 'offset_open': open_center, 'extents_mid': middle, 'offset_mid': mid_center, 'gripper_type': 0, 'fingertip_depth': float(open_center[2])}, 'source': {'kind': 'standalone_native_tool_asset', 'asset_path': self.asset_path, 'tool_id': self.tool_id, 'tool_frame': self.tool_frame}}
```

读取的对象属性：

```python
self._forward
self.open_joint_positions_m
self._data.xpos
self._data
self._grasp_body
self._data.xmat
np.allclose
self.closing_axis_tool
np.linalg.inv
np.linalg
self.closed_joint_positions_m
base.copy
self.asset_path
self.tool_id
self.tool_frame
```

写入目标：

```python
inverse_tool
base
inverse_base
opened
open_center
middle
mid_center
```

直接调用符号：

```python
self._forward
_pose
np.allclose
ValueError
np.linalg.inv
inner_box
base.copy
float
```

## `ParallelJawToolGeometry.grasp_generation_geometry.inner_box`

[定义位置](../../../manipulation/tool_geometry.py#L259)。

输入签名：

```python
def inner_box(q):
```

显式输出分支：

```python
(hi - lo, (hi + lo) / 2)
```

读取的对象属性：

```python
self.at
self._forward
self._data.xpos
self._data
self._grasp_body
self._data.xmat
np.allclose
np.array
np.sort
np.abs
corners.append
transform[:3, :3].T
corners[0][:, 1:].min
corners[1][:, 1:].min
corners[0][:, 1:].max
corners[1][:, 1:].max
np.concatenate
points.min
points.max
np.any
```

写入目标：

```python
state
current_base
corners
side
pad
transform
expected_normal
half
local
x
y
points
lo
hi
```

直接调用符号：

```python
self.at
self._forward
_pose
np.allclose
ValueError
enumerate
np.array
np.sort
np.abs
corners.append
corners[0][:, 1:].min
corners[1][:, 1:].min
corners[0][:, 1:].max
corners[1][:, 1:].max
np.concatenate
points.min
points.max
np.any
```

## `load_robosuite_panda_tool`

[定义位置](../../../manipulation/tool_geometry.py#L311)。

输入签名：

```python
def load_robosuite_panda_tool(*, tool_id, tool_frame):
```

实现说明：

> Load the installed Panda hand asset without importing a robot scene.
>
> Robot-specific names belong in this adapter, not in grasp filtering. The
> external tool name is bound to the asset's native ``grip_site``; neither is
> interpreted as a finger-pad centre. No command-to-q or demo-to-current-state
> substitution is made. The Panda's second native slide has negative q when
> open, unlike robot descriptions that use two positive finger coordinates.

显式输出分支：

```python
ParallelJawToolGeometry(model, tool_id=tool_id, tool_frame=tool_frame, native_tool_site='grip_site', joint_names=('finger_joint1', 'finger_joint2'), collision_roles=(('hand_collision', 'palm'), ('finger1_collision', 'finger'), ('finger2_collision', 'finger'), ('finger1_pad_collision', 'pad'), ('finger2_pad_collision', 'pad')), pad_names=('finger1_pad_collision', 'finger2_pad_collision'), open_joint_positions_m=(0.04, -0.04), closed_joint_positions_m=(0.0, 0.0), asset_path=asset, native_grasp_body='right_gripper')
```

读取的对象属性：

```python
spec.submodule_search_locations
mujoco.MjModel.from_xml_path
mujoco.MjModel
```

写入目标：

```python
spec
asset
model
```

直接调用符号：

```python
find_spec
ImportError
Path
next
iter
mujoco.MjModel.from_xml_path
str
ParallelJawToolGeometry
```
