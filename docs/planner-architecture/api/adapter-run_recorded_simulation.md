# `tests/planning/run_recorded_simulation.py` 实现契约

代码基线：`9cdb433`。
职责和数据语义见[组件说明](../11-device-control-and-evaluation.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_read`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L41)。

输入签名：

```python
def _read(path):
```

显式输出分支：

```python
json.loads(path.read_text())
```

读取的对象属性：

```python
path.stat().st_size
path.stat
json.loads
path.read_text
```

写入目标：

```python
path
```

直接调用符号：

```python
Path
path.stat
ValueError
json.loads
path.read_text
```

## `_one`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L48)。

输入签名：

```python
def _one(rows, key, value):
```

显式输出分支：

```python
found[0]
```

读取的对象属性：

无显式条目。

写入目标：

```python
found
row
```

直接调用符号：

```python
len
ValueError
```

## `_goal_identity`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L56)。

输入签名：

```python
def _goal_identity(value):
```

实现说明：

> Parse a CLI identity without inferring a goal from its task or label.

显式输出分支：

```python
[collection, int(raw_index)]
```

读取的对象属性：

```python
value.partition
raw_index.isdecimal
argparse.ArgumentTypeError
```

写入目标：

```python
collection
separator
raw_index
```

直接调用符号：

```python
value.partition
raw_index.isdecimal
argparse.ArgumentTypeError
int
```

## `_TrialVideos`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L66)。

```python
class _TrialVideos():
```

类字段：

无显式条目。

实现说明：

> Stream native fixed-camera frames directly to the collector's codec.

## `_TrialVideos.__init__`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L69)。

输入签名：

```python
def __init__(self, simulation, *, views, workspace_root, output, fps):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
simulation.native_model.opt.timestep
simulation.native_model.opt
simulation.native_model
np.isclose
(workspace_root / view['calibration_path']).resolve
simulation.native_data
self._entries.append
self._entries
camera.calibration
np.allclose
native['rendering_device']['renderer'].lower
cv2.VideoWriter
cv2.VideoWriter_fourcc
writer.isOpened
path.resolve
(output / f"video-times-{view['id']}.npy").resolve
self.records.append
self.records
self.close
```

写入目标：

```python
view
self._cv2
self._simulation
self._entries
self._counter
self._last_elapsed
self.records
ratio
self._stride
calibration_path
calibration
camera
entry
native
token
path
writer
entry['writer']
record
entry['record']
```

直接调用符号：

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

[定义位置](../../../tests/planning/run_recorded_simulation.py#L121)。

输入签名：

```python
def capture(self, elapsed, *, final=False):
```

显式输出分支：

```python
None
None
```

读取的对象属性：

```python
self._last_elapsed
self._counter
self._stride
self._entries
entry['camera'].capture
self._simulation.native_data
self._simulation
entry['writer'].write
self._cv2.cvtColor
self._cv2
self._cv2.COLOR_RGB2BGR
entry['times'].append
self._simulation.native_data.time
```

写入目标：

```python
due
self._counter
entry
frame
entry['record']['frame_count']
self._last_elapsed
```

直接调用符号：

```python
abs
entry['camera'].capture
entry['writer'].write
self._cv2.cvtColor
entry['times'].append
float
```

## `_TrialVideos.close`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L135)。

输入签名：

```python
def close(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._entries
entry['writer'].release
entry['camera'].close
np.save
np.asarray
```

写入目标：

```python
first_error
entry
```

直接调用符号：

```python
entry['writer'].release
entry['camera'].close
np.save
np.asarray
```

## `run`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L159)。

输入签名：

```python
def run(args, *, simulation=None, initial_feedback=None, observation_callback=None, step_callback=None, on_simulation_ready=None, planning_result=None, reference_feedback=None, diagnostic_evidence_ids=()):
```

实现说明：

> Execute one skill from a recorded entrance or an explicit live handoff.
>
> The native physics evaluator remains independent and never authorizes a
> runtime dependency or supplies owner poses. A caller may provide an
> existing proprioceptive or visual observation callback through the same
> contract; the CLI does not create a semantic sensor from simulator truth.
>
> # BEGIN declared attachment models remain separate from observations.
> Attachment models belong to the accepted planning result and require no
> separate playback authorization or measured physical Holding predicate.
> Future S3E estimates belong in ``observation_callback``: current RGB images
> and bound predicates produce T/F/U with original acquisition provenance.
> Missing/unmatched VLM answers must remain unknown, not default to true.
> No VLM is called here and the planner does not own predicate estimation.
> # END the runner forwards observations without modifying robot feedback.
>
> ``simulation`` and ``initial_feedback`` must be supplied together. A live
> handoff uses the complete production proprioception record, retaining its
> trial, feedback identity, measured velocities and original native time;
> ``args.feedback`` then identifies the caller's saved handoff artifact and
> is not decoded as a recorded entrance. The caller owns the borrowed
> backend and its cameras. This function never restores or closes them.
>
> ``planning_result`` supplies an already returned or loaded formal result.
> With no in-memory result the archive loader supplies it. Both inputs enter
> the same device rebinding and actual entrance checks before execution.
>
> ``on_simulation_ready(simulation, restore_report)`` may retain a newly
> restored backend for later skills. It runs once after the recorded
> entrance and execution binding have been checked, before evaluator setup
> or execution. Its return value is ignored; an exception prevents control.
> The hook is incompatible with a borrowed backend and must not mutate or
> advance the simulator. Executor geometry and control checks remain intact.
>
> ``step_callback(model, data, record)`` is an independent, read-only capture
> sink for the initial sample and normal completed steps, not an online
> effect observer. Its return value is ignored; acquisition failures are
> reported without replacing physical execution errors. Holds remain in
> the executor's separate accounting, so the caller may capture the current
> post-hold state directly after this call returns or raises.

显式输出分支：

```python
report
```

读取的对象属性：

```python
args.output.resolve
args.output
output.mkdir
time.monotonic
args.segment
args.trajectory.resolve
args.trajectory
args.finger_targets.resolve
args.finger_targets
args.execution_limits.resolve
args.execution_limits
args.no_video
args.diagnostic_goal_ids
args.manifest
args.robot_fixture
args.feedback
np.asarray
tool_transform.shape
np.isfinite(tool_transform).all
np.isfinite
FixedFingerTargets.from_mapping
ExecutionLimits.from_mapping
(root / manifest['source_recording']).resolve
args.controller_part
Path(fixture['episode_path']).resolve
planning_result.trajectory
curve.events
simulation.proprioception
report.update
simulation.trial_id
initial_feedback.get
measured.shape
expected.shape
np.isfinite(measured).all
np.isfinite(expected).all
np.max
np.abs
differences.values
planning_result.execution_binding
simulation.describe
args.result_only_goal_ids
rebound_path.parent.mkdir
rebound_path.parent
execution_limits.describe
args.manifest.resolve
args.robot_fixture.resolve
args.feedback.resolve
(root / segment['program_path']).resolve
args.evaluator_evidence.resolve
args.evaluator_evidence
program['demo']['roles'].get
source_manifest_path.parent
simulation.native_model
args.views
videos.records
simulation.request_control_rebase
executor.execute
args.max_physics_steps
args.start_position_tolerance
args.start_velocity_tolerance
type(error).__name__
(output / 'exception.txt').write_text
traceback.format_exc
type(write_error).__name__
executor.last_execution_report
execution.get
failure.get
report['feedback_failure_summary'].update
first.items
np.generic
execution.get('continuation', {}).get
evaluator.finalize
(output / 'evaluation-report.json').write_text
json.dumps
videos.capture
videos.close
simulation.sim_time
report['evidence_collection'].values
resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
resource.getrusage
resource.RUSAGE_SELF
```

写入目标：

```python
output
started
wallclock
borrowed_simulation
report
evaluator
videos
execution
executor
caught
report['evidence_collection']
name
configured
manifest
fixture
root
segment
program
report['skill_kind']
initial_feedback
feedback
key
tool_transform
finger_targets
execution_limits
recording
source_manifest_path
source
environment
arm_config
planning_result
curve
event
restore_report
simulation
current
differences
measured
expected
differences[key]
report['initial_feedback_errors']
source_candidate
report['trajectory_classification']
rebound_path
evidence
source_segment
actor
catalog
reference_role
reference_name
source_catalog_path
source_catalog
report['evidence_collection']['evaluator']['initialized']
views
report['videos']
report['evidence_collection']['video']['initialized']
report['evidence_collection']['capture_callback']['initialized']
validator
report['status']
entrance
report['execution_tested']
report['exception_write_error']
report['executor_wallclock']
hold_steps
accounting
confirmed_steps
exact_steps
normal_exact
report['execution_persistence_errors']
report['independent_evidence_sink']
preflight
report['geometry_preflight_summary']
failure
violations
first
report['feedback_failure_summary']
value
report['final_robot_feedback']
collection
collection['completed']
assessment
report['evaluation_coverage']
report['evaluation_error']
retained
missing
report['evaluation_report_write_error']
report['video_final_elapsed_s']
report['video_final_capture_error']
report['video_error']
video_state
video_state['completed']
capture_state
capture_state['completed']
native_end
report['native_end_time_s']
report['native_end_time_error']
report['evidence_complete']
state
```

直接调用符号：

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
list
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
simulation.request_control_rebase
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
write_evidence_json
```

## `run.mark_wallclock`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L209)。

输入签名：

```python
def mark_wallclock(name):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
time.time_ns
time.monotonic
```

写入目标：

```python
wallclock['events'][name]
```

直接调用符号：

```python
time.time_ns
time.monotonic
```

## `run.evidence_failure`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L262)。

输入签名：

```python
def evidence_failure(name, error, phase, record=None):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
state.update
simulation.sim_time
np.isfinite
type(error).__name__
```

写入目标：

```python
state
sim_time
state['first_error']
```

直接调用符号：

```python
state.update
float
np.isfinite
type
str
```

## `run.observe`

[定义位置](../../../tests/planning/run_recorded_simulation.py#L476)。

输入签名：

```python
def observe(model, data, record):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
evaluator.observe
videos.capture
```

写入目标：

```python
phase
state
state['attempted_callbacks']
state['successful_callbacks']
```

直接调用符号：

```python
evaluator.observe
evidence_failure
videos.capture
step_callback
deepcopy
```
