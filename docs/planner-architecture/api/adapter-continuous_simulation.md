# `tests/planning/continuous_simulation.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_read`

[定义位置](../../../tests/planning/continuous_simulation.py#L22)。

输入签名：

```python
def _read(path):
```

输出分支的源码表达式：

```python
json.loads(Path(path).read_text())
```

直接调用：

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

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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
class ContinuousSimulationSession:
    ...
    ...
```

该类型的直接方法：`__init__`、`_adopt`、`execute`、`capture`。

## `ContinuousSimulationSession.__init__`

[定义位置](../../../tests/planning/continuous_simulation.py#L41)。

输入签名：

```python
def ContinuousSimulationSession.__init__(self, *, manifest_path, fixture_path, controller_part, output):
```

直接读取的对象属性：`self.output`、`self.manifest`、`self.root`、`self.fixture`、`self.environment`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.output`、`self.manifest_path`、`self.fixture_path`、`self.manifest`、`self.fixture`、`self.root`、`self.environment`、`self.controller_part`、`self.simulation`、`self.trial_id`、`self.stages`、`self.finalization_errors`。

直接调用：

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
def ContinuousSimulationSession._adopt(self, simulation, restored):
```

直接读取的对象属性：`self.simulation`、`self.output`、`self.trial_id`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.simulation`、`self.trial_id`。

直接调用：

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
def ContinuousSimulationSession.execute(self, specification, *, observation_callback=None):
```

直接读取的对象属性：`self.fixture_path`、`self.simulation`、`self._adopt`、`self.stages`、`self.trial_id`、`self.output`、`self.finalization_errors`。

输出分支的源码表达式：

```python
report
```

直接写入的状态或数组位置：`spec[key]`。

直接调用：

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

[定义位置](../../../tests/planning/continuous_simulation.py#L139)。

输入签名：

```python
def ContinuousSimulationSession.capture(self, destination, *, views, tool_id):
```

直接读取的对象属性：`self.simulation`、`self.trial_id`、`self.stages`、`self.root`、`self.finalization_errors`。

输出分支的源码表达式：

```python
destination / 'capture.json'
```

直接调用：

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
