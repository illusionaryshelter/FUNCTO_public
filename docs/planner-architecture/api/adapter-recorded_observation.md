# `tests/planning/recorded_observation.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_text`

[定义位置](../../../tests/planning/recorded_observation.py#L36)。

输入签名：

```python
def _text(value: Any, label: str) -> str:
```

输出分支的源码表达式：

```python
value
```

直接调用：

```python
isinstance
value.strip
ValueError
```

## `camera_config_from_calibration`

[定义位置](../../../tests/planning/recorded_observation.py#L42)。

输入签名：

```python
def camera_config_from_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]:
```

输出分支的源码表达式：

```python
{'position_world_m': pose[:3, 3].tolist(), 'quaternion_world_camera_gl_wxyz': quaternion.tolist(), 'frustum_center_over_near': float(((width - 1) / 2 - matrix[0, 2]) / matrix[0, 0]), 'frustum_bottom_over_near': float(top - height / matrix[1, 1]), 'frustum_top_over_near': float(top), 'frustum_halfwidth_over_near': float(width / (2 * matrix[0, 0]))}
```

直接调用：

```python
type
min
ValueError
_intrinsics
abs
_pose_array
float
np.isfinite([near, far]).all
np.isfinite
np.diag
Rotation.from_matrix(rotation_gl).as_quat
Rotation.from_matrix
pose[:3, 3].tolist
quaternion.tolist
```

## `_robot_assets`

[定义位置](../../../tests/planning/recorded_observation.py#L68)。

输入签名：

```python
def _robot_assets(scene: ET.Element, robot: ET.Element, scene_xml: Path) -> ET.Element:
```

直接读取的对象属性：`scene.find`。

输出分支的源码表达式：

```python
assets
```

直接调用：

```python
scene.find
node.get
geom.get
robot.iter
tuple
ValueError
len
material.get
requested.add
compiler.get
ET.Element
sorted
deepcopy
copied.attrib.items
attribute.startswith
Path
path.is_absolute
path.is_file
copied.set
str
path.resolve
assets.append
```

## `prepare_robot_model`

[定义位置](../../../tests/planning/recorded_observation.py#L111)。

输入签名：

```python
def prepare_robot_model(*, scene_xml: str | Path, robot_root: str, joint_map: Sequence[tuple[str, str, float]], joint_positions: Mapping[str, float], T_world_robot_root: Any, calibration: Mapping[str, Any]):
```

直接读取的对象属性：`scene.find`。

输出分支的源码表达式：

```python
(model, data, provenance)
```

直接写入的状态或数组位置：`data.qpos[joint.qposadr[0]]`。

直接调用：

```python
camera_config_from_calibration
Path(scene_xml).resolve
Path
_text
_pose_array
ET.parse(scene_xml).getroot
ET.parse
scene.find
worldbody.iter
node.get
len
ValueError
any
source.iter
deepcopy
robot.iter
list
parent.remove
robot.attrib.pop
robot.set
' '.join
map
Rotation.from_matrix(base[:3, :3]).as_quat
Rotation.from_matrix
ET.Element
compiler.set
minimal.append
_robot_assets
ET.SubElement(minimal, 'worldbody').append
ET.SubElement
mujoco.MjModel.from_xml_string
ET.tostring
mujoco.MjData
tuple
np.asarray
set
np.isin(signs, (-1.0, 1.0)).all
np.isin
mujoco.mj_id2name
range
np.isfinite(measured).all
np.isfinite
zip
model.joint
mujoco.mj_kinematics
str
float
base.tolist
```

## `classify_robot_observation`

[定义位置](../../../tests/planning/recorded_observation.py#L217)。

输入签名：

```python
def classify_robot_observation(*, scene_xml: str | Path, robot_root: str, joint_map: Sequence[tuple[str, str, float]], joint_positions: Mapping[str, float], T_world_robot_root: Any, calibration: Mapping[str, Any], observed_depth: Any, padding_scale: float, padding_offset_m: float, source: str) -> dict[str, Any]:
```

输出分支的源码表达式：

```python
classified
```

直接写入的状态或数组位置：`pose[:3, :3]`、`pose[:3, 3]`。

直接调用：

```python
np.asarray
ValueError
_text
prepare_robot_model
range
model.geom
np.eye
data.geom_xmat[index].reshape
_collision_geometry
tuple
float
Path(scene_xml).resolve
Path
geometries.append
RobotCollisionShape
identities.append
model.body
int
classify_depth
classified['provenance'].update
len
```
