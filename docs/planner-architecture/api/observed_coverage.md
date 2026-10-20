# `manipulation/observed_coverage.py` 实现契约

代码基线：`__IMPLEMENTATION_COMMIT__`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `query_coverage`

[定义位置](../../../manipulation/observed_coverage.py#L9)。

输入签名：

```python
def query_coverage(field, pose, coverage, regions, *, precision_m=0.0001, check_request=None):
```

实现说明：

> Bound only represented occupancy; boxes can prove coverage, not occupancy.
> 
> Unknown-space witnesses must be actual occupied points. An overlapping
> bounding box only schedules subdivision. Robot callers do not invoke
> this owner-policy query: robot field-of-view absence allows an attempt.

显式输出分支：

```python
{'status': 'unknown_witness', 'witness': point.tolist(), 'visited_boxes': visited}
{'status': 'unresolved' if unresolved else 'covered', 'visited_boxes': visited}
```

读取的对象属性：

```python
coverage._known
coverage.origin
coverage.voxel_size
np.asarray
known.shape
field.root
field.nodes
pending.pop
node.lower
node.upper
field.occupied_points
point.tolist
node.cell
np.linalg.norm
np.linalg
field.split
pending.extend
```

写入目标：

```python
known
origin
h
shape
regions
row
pending
unresolved
visited
node
box
local
points
errors
point
error
children
```

直接调用符号：

```python
np.asarray
check_request
pending.pop
world_bounds
covered
field.occupied_points
len
transform_points
transform_point_error
zip
outside
point.tolist
np.linalg.norm
field.split
pending.extend
```

## `query_coverage.primitive_covered`

[定义位置](../../../manipulation/observed_coverage.py#L21)。

输入签名：

```python
def primitive_covered(lower, upper):
```

显式输出分支：

```python
True
False
False
bool(values.all())
```

读取的对象属性：

```python
np.all
np.floor((lower - origin) / h + 0.5).astype
np.floor
np.ceil((upper - origin) / h + 0.5).astype
np.ceil
incident.append
np.arange
np.ix_
values.any
values.all
```

写入目标：

```python
lo
hi
incident
axis
first
last
indices
values
```

直接调用符号：

```python
any
np.all
np.floor((lower - origin) / h + 0.5).astype
np.floor
np.ceil((upper - origin) / h + 0.5).astype
np.ceil
range
int
incident.append
np.arange
max
min
len
np.ix_
reversed
values.any
bool
values.all
```

## `query_coverage.covered`

[定义位置](../../../manipulation/observed_coverage.py#L44)。

输入签名：

```python
def covered(box):
```

显式输出分支：

```python
True
False
True
```

读取的对象属性：

```python
cuts.append
np.unique
np.asarray
```

写入目标：

```python
lower
upper
cuts
axis
internal
region
value
indices
lo
index
hi
```

直接调用符号：

```python
primitive_covered
range
cuts.append
np.unique
product
max
len
np.asarray
zip
min
```

## `query_coverage.outside`

[定义位置](../../../manipulation/observed_coverage.py#L63)。

输入签名：

```python
def outside(box):
```

显式输出分支：

```python
False
True
not bool(known[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1].any())
```

读取的对象属性：

```python
np.all
np.maximum
np.ceil((lower - origin) / h - 0.5).astype
np.ceil
np.minimum
np.floor((upper - origin) / h + 0.5).astype
np.floor
np.any
known[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1].any
```

写入目标：

```python
lower
upper
lo
hi
```

直接调用符号：

```python
any
np.all
np.maximum
np.ceil((lower - origin) / h - 0.5).astype
np.ceil
np.minimum
np.floor((upper - origin) / h + 0.5).astype
np.floor
np.any
bool
known[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1].any
```
