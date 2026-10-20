# `manipulation/tool_geometry.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../03-attachment-contact-and-hand.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_pose`

[定义位置](../../../manipulation/tool_geometry.py#L19)。

输入签名：

```python
def _pose(position, rotation):
```

输出分支的源码表达式：

```python
value
```

直接写入的状态或数组位置：`value[:3, :3]`、`value[:3, 3]`。

直接调用：

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

输出分支的源码表达式：

```python
(hull, planes)
```

直接调用：

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

输出分支的源码表达式：

```python
{'geom_id': int(geom_id), 'name': model.geom(geom_id).name, 'role': role, 'vertices': vertices, 'faces': faces, 'halfspaces': planes, 'collision_representation': representation}
```

直接写入的状态或数组位置：`faces[flip]`。

直接调用：

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
class ParallelJawToolGeometry:
    ...
    ...
```

该类型的直接方法：`__init__`、`_positions`、`_forward`、`at`、`grasp_generation_geometry`。

## `ParallelJawToolGeometry.__init__`

[定义位置](../../../manipulation/tool_geometry.py#L103)。

输入签名：

```python
def ParallelJawToolGeometry.__init__(self, model, *, tool_id, tool_frame, native_tool_site, joint_names, collision_roles, pad_names, open_joint_positions_m, closed_joint_positions_m, asset_path, native_grasp_body):
```

直接读取的对象属性：`self._joint_ids`、`self._positions`、`self._pad_ids`、`self._forward`、`self.open_joint_positions_m`、`self._data`、`self.joint_axes_tool`、`self.closing_axis_tool`、`self._pad_faces`、`self.at`、`self.closed_joint_positions_m`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.tool_id`、`self.tool_frame`、`self.asset_path`、`self.joint_names`、`self._model`、`self._data`、`self._site`、`self._grasp_body`、`self._joint_ids`、`self._qpos_addresses`、`self.joint_ranges_m`、`self.open_joint_positions_m`、`self.closed_joint_positions_m`、`self.collision_geometries`、`self._pad_ids`、`self.closing_axis_tool`、`self.joint_axes_tool`、`self._pad_faces`、`self.signed_gap_range_m`。

直接调用：

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
def ParallelJawToolGeometry._positions(self, value):
```

直接读取的对象属性：`self.joint_ranges_m`。

输出分支的源码表达式：

```python
values
```

直接调用：

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
def ParallelJawToolGeometry._forward(self, positions):
```

直接读取的对象属性：`self._data`、`self._qpos_addresses`、`self._model`、`self._site`。

输出分支的源码表达式：

```python
np.linalg.inv(_pose(self._data.site_xpos[self._site], self._data.site_xmat[self._site]))
```

直接写入的状态或数组位置：`self._data.qpos[self._qpos_addresses]`。

直接调用：

```python
mujoco.mj_kinematics
np.linalg.inv
_pose
```

## `ParallelJawToolGeometry.at`

[定义位置](../../../manipulation/tool_geometry.py#L195)。

输入签名：

```python
def ParallelJawToolGeometry.at(self, joint_positions_m):
```

直接读取的对象属性：`self._positions`、`self._forward`、`self.collision_geometries`、`self._data`、`self._pad_ids`、`self._pad_faces`、`self._model`、`self.tool_id`、`self.tool_frame`。

输出分支的源码表达式：

```python
{'tool_id': self.tool_id, 'tool_frame': self.tool_frame, 'joint_positions_m': positions, 'geoms': geoms, 'pads': pads, 'signed_gap_m': gap}
```

直接写入的状态或数组位置：`by_id[index]`。

直接调用：

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
def ParallelJawToolGeometry.grasp_generation_geometry(self):
```

直接读取的对象属性：`self._forward`、`self.open_joint_positions_m`、`self._data`、`self._grasp_body`、`self.closing_axis_tool`、`self.closed_joint_positions_m`、`self.asset_path`、`self.tool_id`、`self.tool_frame`。

输出分支的源码表达式：

```python
{'T_tool_grasp': base.copy(), 'sweep_volume_params': {'extents_open': opened, 'offset_open': open_center, 'extents_mid': middle, 'offset_mid': mid_center, 'gripper_type': 0, 'fingertip_depth': float(open_center[2])}, 'source': {'kind': 'standalone_native_tool_asset', 'asset_path': self.asset_path, 'tool_id': self.tool_id, 'tool_frame': self.tool_frame}}
```

直接调用：

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
def ParallelJawToolGeometry.grasp_generation_geometry.inner_box(q):
```

直接读取的对象属性：`self.at`、`self._forward`、`self._data`、`self._grasp_body`。

输出分支的源码表达式：

```python
(hi - lo, (hi + lo) / 2)
```

直接调用：

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

输出分支的源码表达式：

```python
ParallelJawToolGeometry(model, tool_id=tool_id, tool_frame=tool_frame, native_tool_site='grip_site', joint_names=('finger_joint1', 'finger_joint2'), collision_roles=(('hand_collision', 'palm'), ('finger1_collision', 'finger'), ('finger2_collision', 'finger'), ('finger1_pad_collision', 'pad'), ('finger2_pad_collision', 'pad')), pad_names=('finger1_pad_collision', 'finger2_pad_collision'), open_joint_positions_m=(0.04, -0.04), closed_joint_positions_m=(0.0, 0.0), asset_path=asset, native_grasp_body='right_gripper')
```

直接调用：

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
