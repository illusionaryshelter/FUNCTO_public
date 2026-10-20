# `manipulation/runtime_profiling.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_write_profile`

[定义位置](../../../manipulation/runtime_profiling.py#L18)。

输入签名：

```python
def _write_profile(profiler, output, elapsed, succeeded):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`module['calls']`、`module['self_seconds']`。

直接调用：

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

输出分支的源码表达式：

```python
operation(*args, **kwargs)
result
```

直接调用：

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
