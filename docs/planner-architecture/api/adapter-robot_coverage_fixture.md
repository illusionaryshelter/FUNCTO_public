# `tests/planning/robot_coverage_fixture.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `build_robot_coverage`

[定义位置](../../../tests/planning/robot_coverage_fixture.py#L24)。

输入签名：

```python
def build_robot_coverage(fixture, feedback, coverage, *, initialization_source, max_queries=2048, max_working_bytes=2 * 1024 ** 3):
```

直接读取的对象属性：`feedback.get`。

输出分支的源码表达式：

```python
RobotCoverage(checker=checker, solids=selected, joint_names=feedback['joint_names'], baseline_joint_positions=feedback['start_q'], T_reference_base=feedback['base_pose'], free_grid=coverage.known_free, origin=coverage.origin, voxel_size=coverage.voxel_size, reference_frame=coverage.reference_frame, initialization_source=f'{initialization_source}; measured snapshot: {source}', max_queries=max_queries, max_working_bytes=max_working_bytes)
```

直接调用：

```python
isinstance
initialization_source.strip
ValueError
yaml.safe_load
Path(fixture['robot_config_path']).read_text
Path
validate_fixture_collision_contract
tuple
_named
feedback.get
source.strip
_pose
np.array_equal
np.asarray
mujoco.MjModel.from_xml_path
_native_shapes
any
set
RobotGeometryChecker
checker.query
np.allclose
RobotCoverage
```
