# `manipulation/robot_coverage.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `CoverageBudgetExceeded`

[定义位置](../../../manipulation/robot_coverage.py#L35)。

```python
class CoverageBudgetExceeded(ValueError):
    ...
    ...
```

## `_readonly`

[定义位置](../../../manipulation/robot_coverage.py#L39)。

输入签名：

```python
def _readonly(value):
```

直接读取的对象属性：`result.setflags`。

输出分支的源码表达式：

```python
result
```

直接调用：

```python
np.asarray(value).copy
np.asarray
result.setflags
```

## `_free_rectangles`

[定义位置](../../../manipulation/robot_coverage.py#L45)。

输入签名：

```python
def _free_rectangles(mask):
```

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
np.asarray(result, dtype=np.int64).reshape(-1, 2, 3)
```

直接调用：

```python
enumerate
np.flatnonzero
np.diff
np.pad
row.astype
len
CoverageBudgetExceeded
int
edges.reshape
tuple
rectangles.append
active.pop
active.setdefault
rectangles.extend
active.items
set
result.append
previous.pop
previous.setdefault
previous.items
np.asarray(result, dtype=np.int64).reshape
np.asarray
```

## `_oriented_hull`

[定义位置](../../../manipulation/robot_coverage.py#L83)。

输入签名：

```python
def _oriented_hull(vertices):
```

输出分支的源码表达式：

```python
(faces, hull.equations.copy())
```

直接写入的状态或数组位置：`faces[reversed_faces]`。

直接调用：

```python
ConvexHull
hull.simplices.copy
np.cross
np.einsum
hull.equations.copy
```

## `RobotCoverage`

[定义位置](../../../manipulation/robot_coverage.py#L100)。

```python
class RobotCoverage:
    ...
    ...
```

该类型的直接方法：`__init__`、`baseline_joint_positions`、`validate_observation_grid`、`_named`、`_check_native`、`_mesh`、`_reserve_work`、`_boolean`、`guidance_geometry`、`describe`、`_witness`、`check_state`。

## `RobotCoverage.__init__`

[定义位置](../../../manipulation/robot_coverage.py#L115)。

输入签名：

```python
def RobotCoverage.__init__(self, *, checker, solids, joint_names, baseline_joint_positions, T_reference_base, free_grid, origin, voxel_size, reference_frame, initialization_source, max_queries=2048, max_working_bytes=_MAX_WORKING_BYTES):
```

直接读取的对象属性：`self._named`、`self._baseline_q`、`self._baseline_float32`、`self.origin`、`self.voxel_size`、`self._mesh`、`self._solids`、`self.baseline_joint_names`、`self._base`、`self.required_link_names`、`self._baseline_poses`、`self._check_native`、`self._reserve_work`、`self._free`、`self._free_boxes`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._checker`、`self.reference_frame`、`self.initialization_source`、`self._base`、`self.baseline_joint_names`、`self._baseline_q`、`self._baseline_float32`、`self.max_queries`、`self.max_working_bytes`、`self._queries`、`self._maximum_work_estimate`、`self._largest_refused_work_estimate`、`self._maximum_native_tolerance`、`self._free`、`self.origin`、`self.voxel_size`、`self._free_boxes`、`self._solids`、`self.required_link_names`、`self._baseline_poses`、`row['baseline_pose']`、`planes[:, :3]`、`planes[:, 3]`、`row['baseline_halfspaces_world']`、`self._baseline_shapes`、`self._baseline_bounds`。

直接调用：

```python
isinstance
TypeError
type
ValueError
_name
_rigid_pose
tuple
self._named
self._baseline_q.astype(np.float32).astype
self._baseline_q.astype
np.isfinite(self._baseline_float32).all
np.isfinite
np.asarray
min
_readonly
np.isfinite(self.origin).all
float
_free_rectangles
indices.astype
len
set
np.isfinite(values).all
CoverageBudgetExceeded
np.asarray(values, dtype=float).copy
_oriented_hull
self._mesh
self._solids.append
names.add
dict.fromkeys
checker.query
pose.copy
row['halfspaces_frame'].copy
self._check_native
row['shape'].transform
world_shape.is_empty
world_shapes.append
np.asarray([shape.bounding_box() for shape in world_shapes], float).reshape
shape.bounding_box
self._reserve_work
int
```

## `RobotCoverage.baseline_joint_positions`

[定义位置](../../../manipulation/robot_coverage.py#L215)。

输入签名：

```python
@property
def RobotCoverage.baseline_joint_positions(self):
```

直接读取的对象属性：`self._baseline_q`。

输出分支的源码表达式：

```python
_readonly(self._baseline_q)
```

直接调用：

```python
_readonly
```

## `RobotCoverage.validate_observation_grid`

[定义位置](../../../manipulation/robot_coverage.py#L219)。

输入签名：

```python
def RobotCoverage.validate_observation_grid(self, *, known_free, origin, voxel_size, reference_frame):
```

直接读取的对象属性：`self.reference_frame`、`self.origin`、`self.voxel_size`、`self._free`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
np.asarray
np.array_equal
ValueError
```

## `RobotCoverage._named`

[定义位置](../../../manipulation/robot_coverage.py#L235)。

输入签名：

```python
def RobotCoverage._named(self, names, positions):
```

直接读取的对象属性：`self.baseline_joint_names`。

输出分支的源码表达式：

```python
np.asarray([values[names.index(name)] for name in self.baseline_joint_names], float)
```

直接调用：

```python
tuple
np.asarray
len
set
np.isfinite(values).all
np.isfinite
ValueError
names.index
```

## `RobotCoverage._check_native`

[定义位置](../../../manipulation/robot_coverage.py#L243)。

输入签名：

```python
def RobotCoverage._check_native(self, shape):
```

直接读取的对象属性：`self._maximum_native_tolerance`。

输出分支的源码表达式：

```python
shape
```

直接写入的状态或数组位置：`self._maximum_native_tolerance`。

直接调用：

```python
shape.status
RuntimeError
shape.num_tri
CoverageBudgetExceeded
shape.is_empty
float
shape.get_tolerance
np.isfinite
max
```

## `RobotCoverage._mesh`

[定义位置](../../../manipulation/robot_coverage.py#L260)。

输入签名：

```python
def RobotCoverage._mesh(self, vertices, faces):
```

直接读取的对象属性：`self._check_native`。

输出分支的源码表达式：

```python
self._check_native(manifold.Manifold(manifold.Mesh64(np.ascontiguousarray(vertices, dtype=np.float64), np.ascontiguousarray(faces, dtype=np.uint64))))
```

直接调用：

```python
self._check_native
manifold.Manifold
manifold.Mesh64
np.ascontiguousarray
```

## `RobotCoverage._reserve_work`

[定义位置](../../../manipulation/robot_coverage.py#L265)。

输入签名：

```python
def RobotCoverage._reserve_work(self, estimated):
```

直接读取的对象属性：`self.max_working_bytes`、`self._largest_refused_work_estimate`、`self._maximum_work_estimate`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._largest_refused_work_estimate`、`self._maximum_work_estimate`。

直接调用：

```python
max
CoverageBudgetExceeded
```

## `RobotCoverage._boolean`

[定义位置](../../../manipulation/robot_coverage.py#L271)。

输入签名：

```python
def RobotCoverage._boolean(self, shapes, operation):
```

直接读取的对象属性：`self._free`、`self._free_boxes`、`self._reserve_work`、`self._check_native`。

输出分支的源码表达式：

```python
self._check_native(manifold.Manifold.batch_boolean(shapes, operation))
```

直接调用：

```python
list
np.asarray
shape.num_tri
np.asarray([shape.bounding_box() for shape in shapes], float).reshape
shape.bounding_box
range
len
np.all
int
counts[:index][overlap].sum
counts.sum
self._reserve_work
self._check_native
manifold.Manifold.batch_boolean
```

## `RobotCoverage.guidance_geometry`

[定义位置](../../../manipulation/robot_coverage.py#L289)。

输入签名：

```python
def RobotCoverage.guidance_geometry(self):
```

直接读取的对象属性：`self.reference_frame`、`self._base`、`self._checker`、`self.baseline_joint_names`、`self._baseline_q`、`self.initialization_source`、`self._solids`。

输出分支的源码表达式：

```python
{'reference_frame': self.reference_frame, 'T_reference_base': self._base.copy(), 'urdf_path': self._checker.urdf_path, 'baseline_joint_names': list(self.baseline_joint_names), 'baseline_joint_positions': self._baseline_q.copy(), 'initialization_source': self.initialization_source, 'solids': [{key: value.copy() if isinstance(value, np.ndarray) else value for (key, value) in row.items() if key != 'shape'} for row in self._solids]}
```

直接调用：

```python
self._base.copy
list
self._baseline_q.copy
isinstance
value.copy
row.items
```

## `RobotCoverage.describe`

[定义位置](../../../manipulation/robot_coverage.py#L301)。

输入签名：

```python
def RobotCoverage.describe(self):
```

直接读取的对象属性：`self.reference_frame`、`self.initialization_source`、`self._solids`、`self.required_link_names`、`self.baseline_joint_names`、`self._baseline_q`、`self._base`、`self._free`、`self._free_boxes`、`self.max_working_bytes`、`self.max_queries`、`self._queries`、`self._maximum_work_estimate`、`self._largest_refused_work_estimate`、`self._maximum_native_tolerance`。

输出分支的源码表达式：

```python
{'predicate': 'V(q) minus (V0 union F) is empty', 'residual_classification': 'unclassified: no occupied-space input; F deficit does not establish unknown or collision', 'reference_frame': self.reference_frame, 'initialization_source': self.initialization_source, 'initial_validity_is_assumption': True, 'hidden_environment_validated': False, 'new_free_cells_added': 0, 'native_solid_names': [row['name'] for row in self._solids], 'required_link_names': list(self.required_link_names), 'baseline_joint_names': list(self.baseline_joint_names), 'baseline_joint_positions': self._baseline_q.tolist(), 'T_reference_base': self._base.tolist(), 'free_cells': int(self._free.sum()), 'free_boxes': len(self._free_boxes), 'max_working_bytes': self.max_working_bytes, 'max_queries': self.max_queries, 'queries': self._queries, 'maximum_native_work_estimate_bytes': self._maximum_work_estimate, 'largest_refused_native_work_estimate_bytes': self._largest_refused_work_estimate, 'maximum_native_tolerance_m': self._maximum_native_tolerance, 'numerical_tolerance_is_error_bound': False, 'acceptance_uses_volume_threshold': False, 'continuous_path_certified': False, 'scope': 'native-model whole-volume floating-point state queries only'}
```

直接调用：

```python
list
self._baseline_q.tolist
self._base.tolist
int
self._free.sum
len
```

## `RobotCoverage._witness`

[定义位置](../../../manipulation/robot_coverage.py#L325)。

输入签名：

```python
def RobotCoverage._witness(self, bad, row, pose):
```

直接读取的对象属性：`self._solids`、`self._free_boxes`。

输出分支的源码表达式：

```python
None
{'solid': row['name'], 'frame': row['frame'], 'point_reference': point.tolist(), 'point_frame': local.tolist()}
```

直接调用：

```python
bad.to_mesh64
np.asarray
float
np.linalg.norm
vertices.max
vertices.min
np.isfinite
np.cross
triangle.mean
bad.ray_cast
zip
np.any
any
np.all
len
point.tolist
local.tolist
```

## `RobotCoverage.check_state`

[定义位置](../../../manipulation/robot_coverage.py#L365)。

输入签名：

```python
def RobotCoverage.check_state(self, *, joint_names, joint_positions, entrance=False):
```

直接读取的对象属性：`self._named`、`self._baseline_q`、`self._baseline_float32`、`self.baseline_joint_names`、`self._checker`、`self._queries`、`self.max_queries`、`self._base`、`self.required_link_names`、`self._solids`、`self._check_native`、`self._baseline_bounds`、`self._boolean`、`self._baseline_shapes`、`self._free_boxes`、`self._mesh`、`self._witness`、`self._maximum_native_tolerance`。

输出分支的源码表达式：

```python
report
```

直接写入的状态或数组位置：`self._queries`、`report['failure']`、`report['numerics']`。

直接调用：

```python
self._named
type
ValueError
np.array_equal
self._baseline_q.copy
list
q.tolist
float
zip
report.update
self._checker.query
enumerate
np.asarray
self._check_native
row['shape'].transform
candidate.is_empty
RuntimeError
np.asarray(candidate.bounding_box(), float).reshape
candidate.bounding_box
np.flatnonzero(np.all(self._baseline_bounds[:, 0] <= bounds[1], axis=1) & np.all(bounds[0] <= self._baseline_bounds[:, 1], axis=1)).tolist
np.flatnonzero
np.all
overlapping.remove
overlapping.insert
self._boolean
bad.is_empty
np.asarray(bad.bounding_box(), float).reshape
bad.bounding_box
range
len
self._mesh
np.where
bad.volume
np.isfinite
int
bad.num_tri
self._witness
str
```
