# `tests/planning/run_recorded_simulation.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_read`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L40)。

输入签名：

```python
def _read(path):
```

输出分支的源码表达式：

```python
json.loads(path.read_text())
```

直接调用：

```python
Path
path.stat
ValueError
json.loads
path.read_text
```

## `_one`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L47)。

输入签名：

```python
def _one(rows, key, value):
```

输出分支的源码表达式：

```python
found[0]
```

直接调用：

```python
len
ValueError
```

## `_goal_identity`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L55)。

输入签名：

```python
def _goal_identity(value):
```

输出分支的源码表达式：

```python
[collection, int(raw_index)]
```

直接调用：

```python
value.partition
raw_index.isdecimal
argparse.ArgumentTypeError
int
```

## `_TrialVideos`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L65)。

```python
class _TrialVideos:
    ...
    ...
```

该类型的直接方法：`__init__`、`capture`、`close`。

## `_TrialVideos.__init__`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L68)。

输入签名：

```python
def _TrialVideos.__init__(self, simulation, *, views, workspace_root, output, fps):
```

直接读取的对象属性：`self._entries`、`self.records`、`self.close`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._cv2`、`self._simulation`、`self._entries`、`self._counter`、`self._last_elapsed`、`self.records`、`self._stride`、`entry['writer']`、`entry['record']`。

直接调用：

```python
len
ValueError
np.isclose
round
int
(workspace_root / view['calibration_path']).resolve
_read
FixedCamera
camera_config_from_calibration
self._entries.append
np.allclose
any
native['rendering_device']['renderer'].lower
RuntimeError
cv2.VideoWriter
str
cv2.VideoWriter_fourcc
float
writer.isOpened
path.resolve
(output / f"video-times-{view['id']}.npy").resolve
self.records.append
self.close
```

## `_TrialVideos.capture`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L120)。

输入签名：

```python
def _TrialVideos.capture(self, elapsed, *, final=False):
```

直接读取的对象属性：`self._last_elapsed`、`self._counter`、`self._stride`、`self._entries`、`self._simulation`、`self._cv2`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self._counter`、`entry['record']['frame_count']`、`self._last_elapsed`。

直接调用：

```python
abs
entry['camera'].capture
entry['writer'].write
self._cv2.cvtColor
entry['times'].append
float
```

## `_TrialVideos.close`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L134)。

输入签名：

```python
def _TrialVideos.close(self):
```

直接读取的对象属性：`self._entries`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
entry['writer'].release
entry['camera'].close
np.save
np.asarray
```

## `run`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L158)。

输入签名：

```python
def run(args, *, simulation=None, initial_feedback=None, observation_callback=None, step_callback=None, on_simulation_ready=None):
```

输出分支的源码表达式：

```python
report
```

直接写入的状态或数组位置：`report['evidence_collection']`、`report['skill_kind']`、`report['trajectory_classification']`、`differences[key]`、`report['initial_feedback_errors']`、`report['evidence_collection']['evaluator']['initialized']`、`report['videos']`、`report['evidence_collection']['video']['initialized']`、`report['evidence_collection']['capture_callback']['initialized']`、`report['status']`、`report['execution_tested']`、`report['exception_write_error']`、`report['executor_wallclock']`、`report['execution_persistence_errors']`、`report['independent_evidence_sink']`、`report['geometry_preflight_summary']`、`report['feedback_failure_summary']`、`report['final_robot_feedback']`、`collection['completed']`、`report['evaluation_coverage']`、`report['evaluation_error']`、`report['evaluation_report_write_error']`、`report['video_final_elapsed_s']`、`report['video_final_capture_error']`、`report['video_error']`、`video_state['completed']`、`capture_state['completed']`、`report['native_end_time_s']`、`report['native_end_time_error']`、`report['evidence_complete']`。

直接调用：

```python
args.output.resolve
output.mkdir
time.monotonic
mark_wallclock
str
args.trajectory.resolve
args.finger_targets.resolve
args.execution_limits.resolve
deepcopy
ValueError
callable
_read
Path
_one
isinstance
any
tuple
np.asarray
np.isfinite(tool_transform).all
np.isfinite
FixedFingerTargets.from_mapping
ExecutionLimits.from_mapping
(root / manifest['source_recording']).resolve
Path(fixture['episode_path']).resolve
load_saved_trajectory
restore_recorded_simulation
simulation.proprioception
report.update
float
initial_feedback.get
np.isfinite(measured).all
np.isfinite(expected).all
np.max
np.abs
max
differences.values
rebind_execution_result
simulation.describe
rebound_path.parent.mkdir
save_curve
execution_limits.describe
args.manifest.resolve
args.robot_fixture.resolve
args.feedback.resolve
(root / segment['program_path']).resolve
args.evaluator_evidence.resolve
on_simulation_ready
program['demo']['roles'].get
NativeSkillPhysicsEvaluator
evidence_failure
_TrialVideos
build_robot_validator
TrajectoryExecutor
executor.execute
type
(output / 'exception.txt').write_text
traceback.format_exc
execution.get
sum
failure.get
report['feedback_failure_summary'].update
len
first.items
execution.get('continuation', {}).get
bool
evaluator.finalize
(output / 'evaluation-report.json').write_text
json.dumps
_feedback_json
videos.capture
videos.close
all
report['evidence_collection'].values
resource.getrusage
(output / 'simulation-report.json').write_text
```

## `run.mark_wallclock`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L203)。

输入签名：

```python
def run.mark_wallclock(name):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`wallclock['events'][name]`。

直接调用：

```python
time.time_ns
time.monotonic
```

## `run.evidence_failure`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L255)。

输入签名：

```python
def run.evidence_failure(name, error, phase, record=None):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`state['first_error']`。

直接调用：

```python
state.update
float
np.isfinite
type
str
```

## `run.observe`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L454)。

输入签名：

```python
def run.observe(model, data, record):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`state['attempted_callbacks']`、`state['successful_callbacks']`。

直接调用：

```python
evaluator.observe
evidence_failure
videos.capture
step_callback
deepcopy
```
