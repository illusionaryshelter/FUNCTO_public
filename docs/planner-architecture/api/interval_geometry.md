# `manipulation/interval_geometry.py` 实现契约

代码基线：`fe0a4f8`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_cpu`

[定义位置](../../../manipulation/interval_geometry.py#L22)。

输入签名：

```python
def _cpu(value):
```

显式输出分支：

```python
value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
```

读取的对象属性：

```python
value.detach().cpu().numpy
value.detach().cpu
value.detach
np.asarray
```

写入目标：

无显式条目。

直接调用符号：

```python
hasattr
value.detach().cpu().numpy
value.detach().cpu
value.detach
np.asarray
```

## `_polynomial_ranges`

[定义位置](../../../manipulation/interval_geometry.py#L26)。

输入签名：

```python
def _polynomial_ranges(curve, start, end, derivative=0):
```

实现说明：

> Enclose a cubic or its derivative using Bernstein convex hulls.
>
> Every intersected original polynomial piece contributes. Power-to-
> Bernstein conversion uses long-double intermediates and a binary64
> outward allowance proportional to the arithmetic's absolute terms.
> This is an interval bound, rather than a sampled derivative maximum.

显式输出分支：

```python
(lower, upper)
```

读取的对象属性：

```python
curve.derivative
polynomial.c.shape
polynomial.c
np.full
np.inf
np.full_like
polynomial.x
np.longdouble
polynomial.c[:, i].astype
np.zeros_like
shifted.append
np.abs
np.asarray
np.finfo(float).eps
np.finfo
np.nextafter
controls.min
controls.max
np.all
np.where
np.minimum
np.maximum
```

写入目标：

```python
polynomial
degree
lower
upper
i
left
right
a
b
x
h
source
shifted
arithmetic_scale
k
terms
j
term
controls
error
lo
hi
structural_zero
```

直接调用符号：

```python
curve.derivative
np.full
np.full_like
enumerate
zip
max
float
min
np.longdouble
polynomial.c[:, i].astype
np.zeros_like
range
comb
shifted.append
sum
np.abs
np.asarray
np.finfo
np.nextafter
controls.min
controls.max
np.all
np.where
np.minimum
np.maximum
```

## `_progress_times`

[定义位置](../../../manipulation/interval_geometry.py#L65)。

输入签名：

```python
def _progress_times(trajectory, values):
```

实现说明：

> Map progress events to all physical crossings, including plateaus.

显式输出分支：

```python
output
```

读取的对象属性：

```python
np.asarray
trajectory.progress_u
trajectory.duration
trajectory.progress_s
output.extend
times[progress == value].tolist
output.append
```

写入目标：

```python
times
progress
output
value
a
b
sa
sb
```

直接调用符号：

```python
np.asarray
output.extend
times[progress == value].tolist
zip
output.append
float
```

## `SceneMotionBounds`

[定义位置](../../../manipulation/interval_geometry.py#L79)。

```python
class SceneMotionBounds():
```

类字段：

```python
scene: object
robot: object
attached_names: dict
owner_radii: dict
```

实现说明：

> Point-speed coefficients bound to one scene's complete entity models.
>
> Local enclosing spheres are used only to integrate rotational velocity
> into displacement bounds. Collision queries retain the original native
> entities and original observed fields.

## `SceneMotionBounds.from_scene`

[定义位置](../../../manipulation/interval_geometry.py#L92)。

输入签名：

```python
def from_scene(cls, scene, *, tool_link, T_link_tool):
```

显式输出分支：

```python
cls(scene, robot, attached_names, radii)
```

读取的对象属性：

```python
scene.robot_model._geometries.items
scene.robot_model._geometries
scene.robot_model
scene._native_local_bounds
bodies.append
scene.owners
owner.query.root
owner.query
np.zeros
owner.query.nodes
node.lower
node.upper
owner.owner_id
owner.attachments
np.eye
Rotation.from_quat(np.asarray(relation.quaternion_tool_owner)[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
np.asarray
relation.quaternion_tool_owner
relation.position_tool_owner
scene.kinematics.model.config.kinematics_config
scene.kinematics.model.config
scene.kinematics.model
scene.kinematics
params.joint_names
params.joint_limits.position
params.joint_limits
params.link_name_to_idx_map
```

写入目标：

```python
bodies
attached_names
radii
name
frame
_
local
source_radius
point
radius
owner
corners
node
radii[owner.owner_id]
index
relation
local[:3, :3]
local[:3, 3]
local_radius
tool
complete_radius
attached_names[owner.owner_id, index]
params
arguments
key
robot
```

直接调用符号：

```python
scene.robot_model._geometries.items
max
_norm_up
box_corners
_add_up
_mul_up
_linear_norm_up
bodies.append
SphereBody
np.zeros
enumerate
np.eye
Rotation.from_quat(np.asarray(relation.quaternion_tool_owner)[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
np.asarray
_cpu
getattr
build_geometry_motion_bound
tuple
cls
```

## `SceneMotionBounds.declaration_times`

[定义位置](../../../manipulation/interval_geometry.py#L130)。

输入签名：

```python
def declaration_times(self, trajectory):
```

实现说明：

> Map event times to the actual left/right relation limits.
>
> A progress plateau has an incoming side only at its entrance and an
> outgoing side only at its exit. Its interior retains the exact mode.

显式输出分支：

```python
result
```

读取的对象属性：

```python
self.scene.required_progress_anchors
self.scene
trajectory.progress_s
result.setdefault(first, set()).add
result.setdefault
result.setdefault(last, set()).add
```

写入目标：

```python
result
value
crossing
first
last
```

直接调用符号：

```python
_progress_times
min
max
result.setdefault(first, set()).add
result.setdefault
set
result.setdefault(last, set()).add
```

## `SceneMotionBounds.partition_times`

[定义位置](../../../manipulation/interval_geometry.py#L148)。

输入签名：

```python
def partition_times(self, trajectory):
```

显式输出分支：

```python
np.unique([float(value) for value in result if 0.0 <= value <= trajectory.duration])
```

读取的对象属性：

```python
trajectory.duration
np.asarray
trajectory.progress_u
self.declaration_times
self.scene.owners
self.scene
owner.pose_interval
result.extend
owner.pose_path.progress
owner.pose_path
self.scene.gripper
gripper.nominal_finger_trajectory
result.append
gripper.nominal_finger_trajectory.valid_until_s
np.unique
```

写入目标：

```python
result
owner
a
b
gripper
value
```

直接调用符号：

```python
np.asarray
self.declaration_times
result.extend
_progress_times
result.append
float
np.unique
```

## `SceneMotionBounds._owner_speed`

[定义位置](../../../manipulation/interval_geometry.py#L162)。

输入签名：

```python
def _owner_speed(self, owner, start_progress, end_progress, duration):
```

显式输出分支：

```python
self.attached_names[owner.owner_id, index]
0.0
0.0
result
```

读取的对象属性：

```python
self.owner_radii
owner.owner_id
owner.attachments
relation.active
self.attached_names
relation.release_condition_id
relation.interval
owner.pose_interval
np.clip
np.asarray
owner.pose_path.progress
owner.pose_path
np.nextafter
np.inf
path.attachment
path.progress
path.motion_model
path.poses
(Rotation.from_matrix(path.poses[i, :3, :3]).inv() * Rotation.from_matrix(path.poses[i + 1, :3, :3])).magnitude
Rotation.from_matrix(path.poses[i, :3, :3]).inv
Rotation.from_matrix
path.coordinates
model.kind
model.reference_pose
model.axis_point
model.axis
model.basis
```

写入目标：

```python
middle
radius
index
relation
a
b
u0
u1
path
rate
result
reach
i
left
right
scale
linear
angle
speed
model
delta
arm
rotation
```

直接调用符号：

```python
reversed
tuple
enumerate
relation.active
np.clip
np.asarray
len
float
np.nextafter
_add_up
_norm_up
_mul_up
_linear_norm_up
zip
(Rotation.from_matrix(path.poses[i, :3, :3]).inv() * Rotation.from_matrix(path.poses[i + 1, :3, :3])).magnitude
Rotation.from_matrix(path.poses[i, :3, :3]).inv
Rotation.from_matrix
abs
max
```

## `SceneMotionBounds.interval`

[定义位置](../../../manipulation/interval_geometry.py#L204)。

输入签名：

```python
def interval(self, trajectory, start, end):
```

实现说明：

> Return entity speed bounds and explicit unavailable coordinates.
>
> The caller first verifies continuous joint positions against the same
> model limits. The planner performs this strict polynomial-domain
> check before invoking interval geometry. A nominal hand curve must
> likewise retain its declared model's coordinate domain.

显式输出分支：

```python
{'speeds': speeds, 'unavailable': unavailable}
```

读取的对象属性：

```python
trajectory._curve
trajectory.joint_names
np.maximum
self.scene.gripper
self.scene
gripper.nominal_finger_trajectory
nominal.valid_until_s
nominal._curve
nominal.joint_names
self.robot.body_names
self.robot
self.robot.coefficients
self.robot.joint_names
rates.get
np.interp
np.asarray
trajectory.duration
trajectory.progress_u
trajectory.progress_s
self.scene.owners
owner.owner_id
self._owner_speed
self.attached_names.values
self.attached_names
speeds.pop
unavailable.pop
```

写入目标：

```python
lower
upper
rates
gripper
nominal
lo
hi
name
value
rates[name]
speeds
unavailable
coefficients
missing
joint
coefficient
unavailable[name]
speeds[name]
s0
s1
owner
```

直接调用符号：

```python
_polynomial_ranges
dict
zip
np.maximum
abs
_add_up
_mul_up
float
rates.get
np.interp
np.asarray
self._owner_speed
isinstance
self.attached_names.values
speeds.pop
unavailable.pop
```

## `check_interval_geometry`

[定义位置](../../../manipulation/interval_geometry.py#L252)。

输入签名：

```python
def check_interval_geometry(trajectory, motion, query_geometry, *, max_query_points, spatial_step_m, initial_states=()):
```

实现说明：

> Refine unresolved time intervals using bounded whole-entity movement.
>
> ``query_geometry(times_s, point_budget=...)`` returns a mapping with
> ``states``, actual physical ``query_points`` (including required anchors),
> and ``capacity_exhausted``. Each state contains ``time_s``, ``side``
> (-1/0/+1), and complete ``pairs``. Each pair's ``geometry_entity_key``
> names first/second entities as ``robot/<geometry ID>``, ``owner/<owner ID>``
> or ``background``; bare first/second names are display identities.
> Multiple relation sides at one physical
> time are distinct evidence states. Initial states were already charged by
> the caller; ``max_query_points`` is its remaining shared point budget.

显式输出分支：

```python
{'confirmed_intersections': intersections, 'proved_intervals': proved, 'unresolved_intervals': unresolved, 'query_points': consumed, 'queried_times_s': sorted(queried_times), 'spatial_step_m': float(spatial_step_m), 'scope': 'continuous_declared_geometry_model'}
```

读取的对象属性：

```python
np.isfinite
motion.declaration_times
np.asarray
motion.partition_times
pending.popleft
unresolved.extend
pending.clear
motion.interval
speeds.values
unresolved.append
maps[0].get
maps[1].get
failures.append
proofs.append
a.get
b.get
row.get
proved.append
pending.appendleft
```

写入目标：

```python
cache
intersections
seen_intersections
queried_times
consumed
declarations
partition
pending
proved
unresolved
start
end
states
exhausted
a
b
bounds
speeds
unavailable
movement
base
state
maps
row
failures
proofs
key
display
item
name
relative
finite_permission
distance
change
value
reason
middle
```

直接调用符号：

```python
int
ValueError
np.isfinite
set
retain
motion.declaration_times
np.asarray
motion.partition_times
deque
zip
map
pending.popleft
get_states
unresolved.extend
float
pending.clear
motion.interval
_mul_up
max
speeds.values
any
unresolved.append
sorted
maps[0].get
maps[1].get
failures.append
proofs.append
a.get
b.get
_add_up
row.get
all
min
proved.append
pending.appendleft
```

## `check_interval_geometry.retain`

[定义位置](../../../manipulation/interval_geometry.py#L273)。

输入签名：

```python
def retain(states):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
seen_intersections.add
intersections.append
```

写入目标：

```python
state
key
cache[key]
pair
entities
identity
```

直接调用符号：

```python
float
int
seen_intersections.add
intersections.append
```

## `check_interval_geometry.get_states`

[定义位置](../../../manipulation/interval_geometry.py#L290)。

输入签名：

```python
def get_states(start, end):
```

显式输出分支：

```python
([cache.get(key) for key in keys], exhausted)
```

读取的对象属性：

```python
declarations.get
trajectory.duration
queried_times.update
cache.get
```

写入目标：

```python
keys
needed
time
side
exhausted
response
used
consumed
state
key
```

直接调用符号：

```python
float
declarations.get
sorted
query_geometry
int
ValueError
retain
queried_times.update
bool
cache.get
```
