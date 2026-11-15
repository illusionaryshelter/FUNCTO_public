# `manipulation/planning_portfolio.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../14-grounding-planning-portfolio.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_write`

[定义位置](../../../manipulation/planning_portfolio.py#L25)。

输入签名：

```python
def _write(path, value):
```

实现说明：

> Persist a complete finite lifecycle document before the next stage.

显式输出分支：

无显式条目。

读取的对象属性：

```python
path.with_name
path.name
temporary.write_text
json.dumps
temporary.replace
```

写入目标：

```python
path
temporary
```

直接调用符号：

```python
Path
path.with_name
temporary.write_text
json.dumps
temporary.replace
```

## `save_inputs`

[定义位置](../../../manipulation/planning_portfolio.py#L34)。

输入签名：

```python
def save_inputs(directory, value):
```

实现说明：

> Save CPU arrays and declared path operands without native objects.
>
> NumPy files preserve dtype and binary values. Continuous PosePath objects
> declare their own interpolation/composition operands. The numeric array
> references name immutable inputs; each worker loads its own mutable model
> and cache state after process startup.

显式输出分支：

```python
manifest
```

读取的对象属性：

```python
directory.mkdir
```

写入目标：

```python
directory
count
manifest
```

直接调用符号：

```python
Path
directory.mkdir
_write
encode
```

## `save_inputs.encode`

[定义位置](../../../manipulation/planning_portfolio.py#L47)。

输入签名：

```python
def encode(item):
```

显式输出分支：

```python
{'array_file': str(path.resolve())}
encode(item.to_record())
{field.name: encode(getattr(item, field.name)) for field in fields(item) if not field.name.startswith('_')}
{str(key): encode(row) for (key, row) in item.items()}
[encode(row) for row in item]
item.item()
str(item)
item
```

读取的对象属性：

```python
np.ndarray
np.save
path.resolve
item.to_record
field.name
field.name.startswith
item.items
np.generic
item.item
```

写入目标：

```python
path
count
field
key
row
```

直接调用符号：

```python
isinstance
np.save
str
path.resolve
encode
item.to_record
is_dataclass
getattr
fields
field.name.startswith
item.items
item.item
```

## `load_inputs`

[定义位置](../../../manipulation/planning_portfolio.py#L74)。

输入签名：

```python
def load_inputs(path):
```

实现说明：

> Load this worker's exact arrays and restore complete path operations.

显式输出分支：

```python
value
```

读取的对象属性：

```python
json.loads
Path(path).read_text
key.endswith
PosePath.from_record
```

写入目标：

```python
value
reference
key
reference[key]
```

直接调用符号：

```python
decode
json.loads
Path(path).read_text
Path
tuple
key.endswith
isinstance
PosePath.from_record
```

## `load_inputs.decode`

[定义位置](../../../manipulation/planning_portfolio.py#L78)。

输入签名：

```python
def decode(item):
```

显式输出分支：

```python
np.load(item['array_file'], allow_pickle=False)
{key: decode(value) for (key, value) in item.items()}
[decode(value) for value in item]
item
```

读取的对象属性：

```python
np.load
item.items
```

写入目标：

```python
key
value
```

直接调用符号：

```python
isinstance
set
np.load
decode
item.items
```

## `_worker`

[定义位置](../../../manipulation/planning_portfolio.py#L95)。

输入签名：

```python
def _worker(input_path):
```

实现说明：

> Record import, preparation, planning and metric failures separately.

显式输出分支：

无显式条目。

读取的对象属性：

```python
Path(input_path).parent.parent
Path(input_path).parent
time.time_ns
time.monotonic
lifecycle.update
result.success
archive.exists
np.savez_compressed
type(error).__name__
traceback.format_exc
preparation_report.exists
json.loads
preparation_report.read_text
report_path.exists
report_path.read_text
```

写入目标：

```python
folder
lifecycle
started
payload
hypothesis
request
result
archive
diagnostic
lifecycle['source_curve']
_
metrics
arrays
preparation_report
current
lifecycle['stage']
report_path
report
lifecycle['accepted']
lifecycle['planning_status']
```

直接调用符号：

```python
Path
dict
time.time_ns
str
time.monotonic
_write
load_inputs
lifecycle.update
plan
bool
archive.exists
load_curve_archive
measure
_write_diagnostic_json
np.savez_compressed
type
traceback.format_exc
preparation_report.exists
json.loads
preparation_report.read_text
report_path.exists
report_path.read_text
```

## `_resource_sample`

[定义位置](../../../manipulation/planning_portfolio.py#L153)。

输入签名：

```python
def _resource_sample(root_pid):
```

实现说明：

> Measure the complete explicit process tree and whole CUDA device zero.

显式输出分支：

```python
dict(unix_time_ns=time.time_ns(), host_rss_bytes=rss, device_used_bytes=device)
```

读取的对象属性：

```python
psutil.Process
root.memory_info().rss
root.memory_info
root.children
child.memory_info().rss
child.memory_info
psutil.NoSuchProcess
subprocess.run
query.stdout.splitlines()[0].strip
query.stdout.splitlines
query.stdout
time.time_ns
```

写入目标：

```python
root
rss
child
query
device
```

直接调用符号：

```python
psutil.Process
root.memory_info
root.children
child.memory_info
subprocess.run
int
query.stdout.splitlines()[0].strip
query.stdout.splitlines
dict
time.time_ns
```

## `_stop_workers`

[定义位置](../../../manipulation/planning_portfolio.py#L170)。

输入签名：

```python
def _stop_workers(processes):
```

实现说明：

> Stop live worker groups and reap them before returning to the caller.

显式输出分支：

无显式条目。

读取的对象属性：

```python
process.poll
os.killpg
process.pid
signal.SIGTERM
time.monotonic
process.wait
subprocess.TimeoutExpired
signal.SIGKILL
```

写入目标：

```python
live
process
deadline
```

直接调用符号：

```python
process.poll
os.killpg
time.monotonic
process.wait
max
```

## `plan_grounding_hypotheses`

[定义位置](../../../manipulation/planning_portfolio.py#L191)。

输入签名：

```python
def plan_grounding_hypotheses(grounding_result, planning_request):
```

实现说明：

> Plan the complete hypothesis set and deliver its original selected curve.
>
> ``reference_hypotheses`` carry stable ids, F/E/B, complete references and
> evidence. ``planning_request`` supplies source task declarations and
> current chart/material bindings plus observation, feedback, robot, budget
> and fixed preference data. Processes share files read-only, never native
> objects or a backend. A numerical rejection is a completed planning result;
> startup, preparation and resource failures retain their actual stage.

显式输出分支：

```python
dict(selection=selection, comparison=comparison, workers=manifest['workers'], selected_id=selected_id, selected_curve=selected_curve, selected_result=selected_result, output_directory=str(output))
```

读取的对象属性：

```python
output.mkdir
planning_request.get
os.getpid
time.monotonic
time.time_ns
folder.mkdir
manifest['workers'].append
request.update
type(error).__name__
traceback.format_exc
os.environ
Path(__file__).resolve().parents
Path(__file__).resolve
os.pathsep
env.get
(Path(worker['output_directory']) / 'run.log').open
logs.append
sys.executable
subprocess.Popen
subprocess.STDOUT
processes.append
launched.append
worker.update
process.pid
process.poll
manifest['resource_samples'].append
manifest.update
time.sleep
process.wait
log.close
state_path.exists
json.loads
state_path.read_text
state.update
metrics_path.exists
state.get
cases.append
```

写入目标：

```python
hypotheses
ids
row
output
resources
host_limit
device_limit
root_pid
start
manifest
processes
launched
logs
hypothesis
folder
worker
request
frozen
key
inputs
worker['input_path']
env
project
env['PYTHONPATH']
log
command
process
manifest['status']
next_sample
sample
worker['process_returncode']
cases
state_path
state
worker['result']
metrics_path
case
comparison
selection
selected_id
selected_result
selected_curve
```

直接调用符号：

```python
len
set
ValueError
Path
output.mkdir
planning_request.get
os.getpid
configure_host_process_root
time.monotonic
dict
time.time_ns
folder.mkdir
str
manifest['workers'].append
deepcopy
request.update
save_inputs
_write
type
traceback.format_exc
Path(__file__).resolve
env.get
(Path(worker['output_directory']) / 'run.log').open
logs.append
subprocess.Popen
processes.append
launched.append
worker.update
any
process.poll
_resource_sample
manifest['resource_samples'].append
manifest.update
_stop_workers
time.sleep
process.wait
log.close
state_path.exists
json.loads
state_path.read_text
state.update
metrics_path.exists
read_evidence_json
state.get
cases.append
select
load_planning_result
_write_diagnostic_json
```
