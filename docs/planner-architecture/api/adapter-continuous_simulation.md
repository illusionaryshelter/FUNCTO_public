# `tests/planning/continuous_simulation.py` 实现契约

代码基线：`3266476`。
职责和数据语义见[组件说明](../11-device-control-and-evaluation.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_read`

[定义位置](../../../tests/planning/continuous_simulation.py#L22)。

输入签名：

```python
def _read(path):
```

显式输出分支：

```python
json.loads(Path(path).read_text())
```

读取的对象属性：

```python
json.loads
Path(path).read_text
```

写入目标：

无显式条目。

直接调用符号：

```python
json.loads
Path(path).read_text
Path
```

## `_write`

[定义位置](../../../tests/planning/continuous_simulation.py#L26)。

输入签名：

```python
def _write(path, value):
```

实现说明：

> Create a new observation/report, preserving any prior trial evidence.

显式输出分支：

无显式条目。

读取的对象属性：

```python
Path(path).open
json.dump
stream.write
```

写入目标：

```python
stream
```

直接调用符号：

```python
Path(path).open
Path
json.dump
_feedback_json
stream.write
```

## `ContinuousSimulationSession`

[定义位置](../../../tests/planning/continuous_simulation.py#L33)。

```python
class ContinuousSimulationSession():
```

类字段：

无显式条目。

实现说明：

> One restore, explicit actual feedback, and independent per-stage scores.
>
> The caller may run expensive child processes between calls. The session
> retains the same backend instance and never restores between stages. No
> hidden integration-state comparison participates in stage admission.

## `ContinuousSimulationSession.__init__`

[定义位置](../../../tests/planning/continuous_simulation.py#L41)。

输入签名：

```python
def __init__(self, *, manifest_path, fixture_path, controller_part, output):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
Path(output).resolve
self.output.mkdir
self.output
self.manifest
self.root
Path(self.fixture['episode_path']).resolve
self.fixture
(recording / 'episode.h5').resolve
self.environment
```

写入目标：

```python
self.output
self.manifest_path
self.fixture_path
self.manifest
self.fixture
self.root
recording
self.environment
self.controller_part
controller
self.simulation
self.trial_id
self.stages
self.finalization_errors
```

直接调用符号：

```python
Path(output).resolve
Path
self.output.mkdir
_read
Path(self.fixture['episode_path']).resolve
(recording / 'episode.h5').resolve
ValueError
```

## `ContinuousSimulationSession._adopt`

[定义位置](../../../tests/planning/continuous_simulation.py#L66)。

输入签名：

```python
def _adopt(self, simulation, restored):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.simulation
simulation.proprioception
self.output
self.trial_id
self.simulation.proprioception
```

写入目标：

```python
self.simulation
self.trial_id
```

直接调用符号：

```python
RuntimeError
simulation.proprioception
_write
self.simulation.proprioception
```

## `ContinuousSimulationSession.execute`

[定义位置](../../../tests/planning/continuous_simulation.py#L80)。

输入签名：

```python
def execute(self, specification, *, observation_callback=None, planning_result=None, step_callback=None, reference_feedback=None, diagnostic_evidence_ids=()):
```

实现说明：

> Execute from the same backend; physical scores never gate a stage.
>
> Only the independent observation callback can provide online task
> evidence. Native physics grading remains in the returned report, even
> when false, unavailable or failed. Attachment declarations are already
> part of the compiled model and need no second assumption channel.

显式输出分支：

```python
report
```

读取的对象属性：

```python
Path(spec[key]).resolve
self.fixture_path.resolve
self.fixture_path
self.simulation
self.simulation.proprioception
self._adopt
errors.append
self.stages
args.segment
self.trial_id
report.get
args.output
self.stages.append
self.output
self.finalization_errors.extend
self.finalization_errors
'; '.join
```

写入目标：

```python
spec
key
spec[key]
args
before
report
primary_error
inputs
after
errors
entry
error
```

直接调用符号：

```python
deepcopy
isinstance
_read
Path(spec[key]).resolve
Path
self.fixture_path.resolve
ValueError
SimpleNamespace
self.simulation.proprioception
run
errors.append
len
report.get
str
int
repr
self.stages.append
_write
self.finalization_errors.extend
RuntimeError
'; '.join
```

## `ContinuousSimulationSession.capture`

[定义位置](../../../tests/planning/continuous_simulation.py#L148)。

输入签名：

```python
def capture(self, destination, *, views, tool_id):
```

实现说明：

> Export only synchronized camera measurements and named robot feedback.
>
> Camera-native segmentation is deliberately discarded. The returned
> manifest exposes no simulator body ID, object pose, force, contact or
> integration-state array. The same observed images may serve both task
> perception and collision mapping, with their masks produced separately.

显式输出分支：

```python
destination / 'capture.json'
```

读取的对象属性：

```python
Path(destination).resolve
destination.mkdir
self.simulation.proprioception
self.simulation
self.trial_id
self.stages
self.root
folder.mkdir
self.simulation.native_model
self.simulation.native_data
camera.calibration
np.allclose
camera.capture
Image.fromarray(frame['rgb']).save
Image.fromarray
np.save
camera.close
self.finalization_errors.append
self.finalization_errors
captured.append
```

写入目标：

```python
destination
feedback
capture_id
captured
view
calibration
folder
camera
capture_error
native
key
frame
calibration['timestamp_s']
```

直接调用符号：

```python
Path(destination).resolve
Path
destination.mkdir
self.simulation.proprioception
len
_write
_read
folder.mkdir
FixedCamera
camera_config_from_calibration
np.allclose
ValueError
camera.capture
Image.fromarray(frame['rgb']).save
Image.fromarray
np.save
camera.close
self.finalization_errors.append
repr
captured.append
```
