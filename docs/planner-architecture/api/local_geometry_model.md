# `manipulation/local_geometry_model.py` 实现契约

代码基线：`a56c3e4`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_point_world`

[定义位置](../../../manipulation/local_geometry_model.py#L18)。

输入签名：

```python
def _point_world(point, pose):
```

显式输出分支：

```python
position + quaternion_rotate(quaternion, local)
```

读取的对象属性：

```python
torch.as_tensor
position.dtype
position.device
```

写入目标：

```python
position
quaternion
local
```

直接调用符号：

```python
torch.as_tensor
quaternion_rotate
```

## `LocalPairModel`

[定义位置](../../../manipulation/local_geometry_model.py#L25)。

```python
class LocalPairModel():
```

类字段：

```python
point: tuple
surface_point: tuple
surface_normal: tuple
surface_on_second: bool
length_scale_m: float
```

实现说明：

> A material point and an observed zero-section tangent in local frames.
>
> ``surface_on_second`` chooses which pair entity owns the zero point and
> normal; the other entity owns ``point``. All coordinates and the positive
> smoothing length are metres, except the unit normal. The normal rotates
> with its entity in every trial, including its rotation derivative.

## `LocalPairModel.value`

[定义位置](../../../manipulation/local_geometry_model.py#L40)。

输入签名：

```python
def value(self, first_pose, second_pose):
```

显式输出分支：

```python
self.length_scale_m * torch_functional.softplus(-signed / self.length_scale_m)
```

读取的对象属性：

```python
self.surface_on_second
torch.as_tensor
self.surface_normal
surface_pose[0].dtype
surface_pose[0].device
((_point_world(self.point, point_pose) - _point_world(self.surface_point, surface_pose)) * normal_world).sum
self.point
self.surface_point
self.length_scale_m
torch_functional.softplus
```

写入目标：

```python
point_pose
surface_pose
normal
normal_world
signed
```

直接调用符号：

```python
torch.as_tensor
quaternion_rotate
((_point_world(self.point, point_pose) - _point_world(self.surface_point, surface_pose)) * normal_world).sum
_point_world
torch_functional.softplus
```

## `LocalContactModel`

[定义位置](../../../manipulation/local_geometry_model.py#L52)。

```python
class LocalContactModel():
```

类字段：

```python
point: tuple
```

实现说明：

> One fixed observed zero point in the target entity's coordinates.

## `LocalContactModel.value`

[定义位置](../../../manipulation/local_geometry_model.py#L57)。

输入签名：

```python
def value(self, normals, bounds):
```

实现说明：

> Return the norm of the declared region's half-space excesses.
>
> Both region coefficients and the saved point are in the target frame.
> This is a continuous optimization objective, not contact evidence or
> an exact Euclidean distance to the intersection of half-spaces.

显式输出分支：

```python
torch.linalg.vector_norm(excess, dim=-1)
```

读取的对象属性：

```python
torch.as_tensor
self.point
normals.dtype
normals.device
torch.relu
(normals * point).sum
torch.linalg.vector_norm
torch.linalg
```

写入目标：

```python
point
excess
```

直接调用符号：

```python
torch.as_tensor
torch.relu
(normals * point).sum
torch.linalg.vector_norm
```

## `_StateModels`

[定义位置](../../../manipulation/local_geometry_model.py#L70)。

```python
class _StateModels():
```

类字段：

```python
pairs: dict = field(default_factory=dict)
contacts: dict = field(default_factory=dict)
```

## `LocalGeometryModels`

[定义位置](../../../manipulation/local_geometry_model.py#L75)。

```python
class LocalGeometryModels():
```

类字段：

无显式条目。

实现说明：

> Capture and freeze local objectives on actual optimization base states.
>
> The caller begins capture, evaluates the repeated base actions without
> consuming their cost, then finishes capture before evaluating cost and
> gradient. A state key uses seed identity, physical time, relation progress
> and permission progress. It never depends on candidate joint values.
> Recording empty states distinguishes a valid empty model from a rollout
> whose time or relation layout was never captured.

## `LocalGeometryModels.__init__`

[定义位置](../../../manipulation/local_geometry_model.py#L86)。

输入签名：

```python
def __init__(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
self.num_seeds
self.num_particles
self.capturing
self._states
self._pending
self._previous_layout
self.repeated_base
```

直接调用符号：

无显式条目。

## `LocalGeometryModels.repeated_base_evaluation`

[定义位置](../../../manipulation/local_geometry_model.py#L99)。

输入签名：

```python
def repeated_base_evaluation(self):
```

实现说明：

> Declare identical particle inputs while keeping independent gradients.
>
> The scene reuses numeric query records within one call only. It still
> forms each row's scalar from that row's differentiable entity poses.
> A trial evaluation never enters this scope merely because it has the
> same physical-time layout as a base evaluation.

显式输出分支：

```python
None
```

读取的对象属性：

```python
self.repeated_base
```

写入目标：

```python
previous
self.repeated_base
```

直接调用符号：

无显式条目。

## `LocalGeometryModels.begin_capture`

[定义位置](../../../manipulation/local_geometry_model.py#L115)。

输入签名：

```python
def begin_capture(self, num_seeds, num_particles):
```

实现说明：

> Start replacement from one actual base trajectory per seed.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.capturing
self.num_seeds
self.num_particles
```

写入目标：

```python
self._previous_layout
self.num_seeds
self.num_particles
self._pending
self.capturing
```

直接调用符号：

```python
RuntimeError
ValueError
int
```

## `LocalGeometryModels.state_key`

[定义位置](../../../manipulation/local_geometry_model.py#L126)。

输入签名：

```python
def state_key(self, row, time_s, progress, permission_progress):
```

实现说明：

> Map repeated particle rows onto the same semantic seed state.

显式输出分支：

```python
(seed, float(time_s), float(progress), float(permission_progress))
```

读取的对象属性：

```python
self.num_particles
self.num_seeds
```

写入目标：

```python
seed
```

直接调用符号：

```python
RuntimeError
int
ValueError
float
```

## `LocalGeometryModels.state`

[定义位置](../../../manipulation/local_geometry_model.py#L135)。

输入签名：

```python
def state(self, key):
```

实现说明：

> Read one frozen state, or create its replacement during capture.

显式输出分支：

```python
self._pending.setdefault(key, _StateModels())
self._states[key]
```

读取的对象属性：

```python
self.capturing
self._pending.setdefault
self._pending
self._states
```

写入目标：

无显式条目。

直接调用符号：

```python
self._pending.setdefault
_StateModels
RuntimeError
```

## `LocalGeometryModels.captures_row`

[定义位置](../../../manipulation/local_geometry_model.py#L144)。

输入签名：

```python
def captures_row(self, row):
```

实现说明：

> Repeated copies of a base seed need only one local direction solve.

显式输出分支：

```python
self.capturing and int(row) % self.num_particles == 0
```

读取的对象属性：

```python
self.capturing
self.num_particles
```

写入目标：

无显式条目。

直接调用符号：

```python
int
```

## `LocalGeometryModels.finish_capture`

[定义位置](../../../manipulation/local_geometry_model.py#L148)。

输入签名：

```python
def finish_capture(self):
```

实现说明：

> Freeze the replacement and report exactly which seeds changed.

显式输出分支：

```python
changed
```

读取的对象属性：

```python
self.capturing
np.zeros
self.num_seeds
self._states.keys
self._states
self._pending.keys
self._pending
self._states.get
self._pending.get
```

写入目标：

```python
changed
key
seed
changed[seed]
self._states
self._pending
self._previous_layout
self.capturing
```

直接调用符号：

```python
RuntimeError
np.zeros
self._states.keys
self._pending.keys
self._states.get
self._pending.get
```

## `LocalGeometryModels.abort_capture`

[定义位置](../../../manipulation/local_geometry_model.py#L163)。

输入签名：

```python
def abort_capture(self):
```

实现说明：

> Discard an interrupted replacement and restore its previous layout.
>
> The solver exits on the original interruption; this method only closes
> the capture lifecycle and does not evaluate an earlier objective.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.capturing
self._previous_layout
```

写入目标：

```python
self.num_seeds
self.num_particles
self._previous_layout
self._pending
self.capturing
```

直接调用符号：

无显式条目。
