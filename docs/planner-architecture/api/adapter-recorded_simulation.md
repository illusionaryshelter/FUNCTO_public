# `tests/planning/recorded_simulation.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `load_curve_archive`

[定义位置](../../../tests/planning/recorded_simulation.py#L31)。

输入签名：

```python
def load_curve_archive(path):
```

输出分支的源码表达式：

```python
(curve, metadata)
```

直接写入的状态或数组位置：`curve.finger_trajectory`。

直接调用：

```python
Path
path.with_suffix
metadata_path.stat
ValueError
json.loads
metadata_path.read_text
np.load
str
arrays['classification'].item
JointTrajectory
tuple
arrays['joint_names'].tolist
arrays['times'].copy
arrays['position'].copy
arrays['velocity'].copy
arrays['progress_u'].copy
arrays['progress_s'].copy
set
FingerTrajectory.from_mapping
arrays['finger_joint_names'].tolist
np.array_equal
len
any
original.items
```

## `load_saved_trajectory`

[定义位置](../../../tests/planning/recorded_simulation.py#L90)。

输入签名：

```python
def load_saved_trajectory(path):
```

输出分支的源码表达式：

```python
result
```

直接调用：

```python
load_curve_archive
ValueError
isinstance
metadata.get
ExecutionLimits.from_mapping
execution_limits.describe
metadata['execution_binding'].get
PlanningResult
str
Path(path).resolve
Path
check_result
```

## `restore_recorded_simulation`

[定义位置](../../../tests/planning/recorded_simulation.py#L121)。

输入签名：

```python
def restore_recorded_simulation(*, scene_path, catalog_path, episode_path, frame_index, base_link, tcp_site, arm_joint_names, finger_joint_names, arm_kp, arm_kd, velocity_limits):
```

输出分支的源码表达式：

```python
(backend, report)
```

直接调用：

```python
type
ValueError
json.loads
Path(catalog_path).read_text
Path
int
mujoco.MjModel.from_xml_path
str
Path(scene_path).resolve
MjSim
mujoco.mj_stateSize
h5py.File
np.asarray
float
np.isfinite(state).all
np.isfinite
mujoco.mj_setState
mujoco.mj_forward
np.isclose
Path(episode_path).resolve
RobosuiteExecutionBackend
deepcopy
backend.describe
```
