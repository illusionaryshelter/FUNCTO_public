# `tests/planning/robot_validation_fixture.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_native_pair_policies`

[定义位置](../../../tests/planning/robot_validation_fixture.py#L24)。

输入签名：

```python
def _native_pair_policies(model, geometry_names):
```

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
result
```

直接调用：

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

[定义位置](../../../tests/planning/robot_validation_fixture.py#L60)。

输入签名：

```python
def build_robot_validator(fixture, *, reference_frame, T_reference_base, tool_frame, T_frame_tcp):
```

输出分支的源码表达式：

```python
RobotStateValidator(checker, joint_names=fixture['joint_names'], arm_joint_names=fixture['arm_joint_names'], finger_joint_names=fixture['finger_joint_names'], pairs=pairs, named_velocity_limits=fixture['velocity_limits'], reference_frame=reference_frame, T_reference_base=T_reference_base, tool_frame=tool_frame, T_frame_tcp=T_frame_tcp, fk_translation_tolerance_m=1e-08, fk_rotation_tolerance_rad=1e-08, source=f"Source-verified fixture {fixture['robot_mjcf_path']}; {validated['complete_native_geometry_count']} articulated and {validated['complete_fixed_installation_geometry_count']} fixed native solids")
```

直接调用：

```python
yaml.safe_load
Path(fixture['robot_config_path']).read_text
Path
validate_fixture_collision_contract
mujoco.MjModel.from_xml_path
ValueError
_pose
np.array_equal
_native_shapes
tuple
_native_pair_policies
frozenset
permitted.get
pairs.append
PairQuery
RobotGeometryChecker
model.joint
RobotStateValidator
```
