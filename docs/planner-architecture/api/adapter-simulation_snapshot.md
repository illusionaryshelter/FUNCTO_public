# `tests/planning/simulation_snapshot.py` 实现契约

代码基线：`9cdb433`。
职责和数据语义见[组件说明](../11-device-control-and-evaluation.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `save_snapshot`

[定义位置](../../../tests/planning/simulation_snapshot.py#L17)。

输入签名：

```python
def save_snapshot(backend, path):
```

实现说明：

> Write native integration state and committed controller state to NPZ.
>
> The backend's snapshot API requires a completed physical-step boundary:
> its integral has already consumed that interval, and no newly issued
> control or requested output rebase is awaiting a physical step. The
> caller owns this paused backend throughout the synchronous save. Native
> and controller state therefore describe the same instant without another
> clock or a second step ledger. The parent directory must already exist.

显式输出分支：

```python
path
```

读取的对象属性：

```python
backend.snapshot_control_state
json.dumps
mujoco.mjtState.mjSTATE_INTEGRATION
mujoco.mjtState
np.empty
mujoco.mj_stateSize
backend.native_model
mujoco.mj_getState
backend.native_data
path.open
np.savez_compressed
np.asarray
np.int64
```

写入目标：

```python
control_state
control_json
state_spec
native_state
path
stream
```

直接调用符号：

```python
backend.snapshot_control_state
json.dumps
int
np.empty
mujoco.mj_stateSize
mujoco.mj_getState
Path
path.open
np.savez_compressed
np.asarray
```

## `restore_native_snapshot`

[定义位置](../../../tests/planning/simulation_snapshot.py#L42)。

输入签名：

```python
def restore_native_snapshot(model, data, path):
```

实现说明：

> Restore native state and return the corresponding backend state record.
>
> ``model`` must be the same explicit physical model used for the snapshot.
> The caller refreshes derived native quantities and constructs the backend,
> then passes this returned record to ``restore_control_state``. That method
> checks named coordinates, controller policy and native time before loading
> the integral. Restoration never derives an integral from saturated ctrl.

显式输出分支：

```python
control_state
```

读取的对象属性：

```python
np.load
json.loads
archive['control_state_json'].item
archive['state_spec'].item
mujoco.mj_setState
```

写入目标：

```python
archive
control_state
state_spec
```

直接调用符号：

```python
np.load
json.loads
archive['control_state_json'].item
int
archive['state_spec'].item
mujoco.mj_setState
```
