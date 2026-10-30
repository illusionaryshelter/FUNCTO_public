# `tests/planning/recorded_simulation.py` 实现契约

代码基线：`9cdb433`。
职责和数据语义见[组件说明](../11-device-control-and-evaluation.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `load_curve_archive`

[定义位置](../../../tests/planning/recorded_simulation.py#L32)。

输入签名：

```python
def load_curve_archive(path):
```

实现说明：

> Inspect exact recorded arrays, including rejected diagnostic curves.

显式输出分支：

```python
(curve, metadata)
```

读取的对象属性：

```python
path.with_suffix
metadata_path.stat().st_size
metadata_path.stat
np.load
arrays['classification'].item
arrays['joint_names'].tolist
arrays['times'].copy
arrays['position'].copy
arrays['velocity'].copy
arrays['progress_u'].copy
arrays['progress_s'].copy
curve.events
arrays.files
FingerTrajectory.from_mapping
fingers.duration
curve.duration
fingers.executable_until
curve.executable_until
arrays['finger_joint_names'].tolist
fingers.joint_names
np.array_equal
fingers.times
fingers.position
fingers.velocity
fingers.events
original.items
curve.next_verification
```

写入目标：

```python
path
metadata_path
metadata
arrays
curve
command_events
row
saved_fingers
finger_arrays
curve.finger_trajectory
fingers
originals
original
key
value
```

直接调用符号：

```python
Path
path.with_suffix
metadata_path.stat
ValueError
read_evidence_json
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

[定义位置](../../../tests/planning/recorded_simulation.py#L93)。

输入签名：

```python
def load_saved_trajectory(path):
```

实现说明：

> Validate a formal source archive against its own original physical limits.
>
> Loading does not adopt a new controller or observation contract. A caller
> must explicitly bind that contract before the source can be executed.

显式输出分支：

```python
result
```

读取的对象属性：

```python
metadata.get
ExecutionLimits.from_mapping
execution_limits.describe
metadata['execution_binding'].get
Path(path).resolve
```

写入目标：

```python
curve
metadata
execution_limits
result
```

直接调用符号：

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

[定义位置](../../../tests/planning/recorded_simulation.py#L124)。

输入签名：

```python
def restore_recorded_simulation(*, scene_path, catalog_path, episode_path, frame_index, base_link, tcp_site, arm_joint_names, finger_joint_names, arm_kp, arm_kd, velocity_limits, robot_mjcf_path):
```

实现说明：

> Restore once, then return the production backend and setup provenance.
>
> Full world integration state is used only for this explicit setup. The
> returned backend never reloads a recording and does not expose object
> states through robot proprioception. Python controller history is absent
> from the recording and is not described as restored.

显式输出分支：

```python
(backend, report)
```

读取的对象属性：

```python
json.loads
Path(catalog_path).read_text
mujoco.mjtState.mjSTATE_INTEGRATION
mujoco.mjtState
mujoco.MjModel.from_xml_path
mujoco.MjModel
Path(scene_path).resolve
sim.data._data
sim.data
mujoco.mj_stateSize
h5py.File
np.asarray
state.shape
np.isfinite(state).all
np.isfinite
mujoco.mj_setState
mujoco.mj_forward
np.isclose
data.time
Path(episode_path).resolve
backend.trial_id
backend.describe
```

写入目标：

```python
catalog
state_spec
spec
model
sim
data
expected_size
stream
state
recorded_time
source
gravity
backend
report
```

直接调用符号：

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
RobotGravity
tuple
RobosuiteExecutionBackend
deepcopy
backend.describe
```
