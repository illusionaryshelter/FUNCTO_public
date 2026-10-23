# `manipulation/runtime_profiling.py` 实现契约

代码基线：`774e121-plus-unified-workspace`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_write_profile`

[定义位置](../../../manipulation/runtime_profiling.py#L18)。

输入签名：

```python
def _write_profile(profiler, output, elapsed, succeeded):
```

实现说明：

> Keep the full pstats archive and bounded, sortable JSON summaries.
>
> Self time excludes nested calls and can be summed across functions or
> files. Cumulative time includes callees and must not be added across rows.
> Both include measurement overhead; neither is a separate unprofiled run.

显式输出分支：

无显式条目。

读取的对象属性：

```python
output.mkdir
profiler.dump_stats
pstats.Stats
statistics.stats.items
statistics.stats
functions.append
modules.setdefault
statistics.total_calls
modules.values
(output / 'runtime.json').write_text
json.dumps
```

写入目标：

```python
output
statistics
functions
modules
filename
line
name
primitive
calls
own
cumulative
_
row
module
module['calls']
module['self_seconds']
report
```

直接调用符号：

```python
Path
output.mkdir
profiler.dump_stats
str
pstats.Stats
statistics.stats.items
functions.append
modules.setdefault
sum
sorted
modules.values
(output / 'runtime.json').write_text
json.dumps
```

## `run_with_optional_profile`

[定义位置](../../../manipulation/runtime_profiling.py#L59)。

输入签名：

```python
def run_with_optional_profile(operation, *args, enabled=False, output, **kwargs):
```

实现说明：

> Call an operation unchanged, optionally recording its host call tree.
>
> Profiling is explicitly opt-in. It is disabled before exporting statistics
> so report generation is not misreported as application work. An operation's
> return value or exception is preserved even if profile export fails.
> The operation owns its output directory lifecycle; this helper creates its
> profile subdirectory only after the operation has returned or raised.

显式输出分支：

```python
operation(*args, **kwargs)
result
```

读取的对象属性：

```python
cProfile.Profile
time.perf_counter
profiler.enable
profiler.disable
type(error).__name__
sys.stderr
```

写入目标：

```python
profiler
succeeded
started
result
elapsed
```

直接调用符号：

```python
operation
cProfile.Profile
time.perf_counter
profiler.enable
profiler.disable
_write_profile
print
type
```
