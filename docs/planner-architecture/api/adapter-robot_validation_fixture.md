# `tests/planning/robot_validation_fixture.py` 实现契约

代码基线：`c84b34a`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_known_shapes`

[定义位置](../../../tests/planning/robot_validation_fixture.py#L25)。

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

[定义位置](../../../tests/planning/robot_validation_fixture.py#L60)。

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

[定义位置](../../../tests/planning/robot_validation_fixture.py#L71)。

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

[定义位置](../../../tests/planning/robot_validation_fixture.py#L80)。

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

[定义位置](../../../tests/planning/robot_validation_fixture.py#L95)。

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

[定义位置](../../../tests/planning/robot_validation_fixture.py#L131)。

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
