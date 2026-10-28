# `manipulation/interval_geometry.py` 实现契约

代码基线：`8b0e326`。
共享验收的数据所有权及调用次序见[共享验收组件](../10-shared-candidate-validation.md)。

职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `prepared_polynomial`

[定义位置](../../../manipulation/interval_geometry.py#L27)。

输入签名：

```python
def prepared_polynomial(shape, coefficients, breakpoints, derivative):
```

实现说明：

> Prepare one complete binary64 polynomial, including mutation identity.
>
> Exact coefficient and breakpoint bytes define the existing PPoly input.
> SciPy computes the original derivative once. Long-double source arrays
> retain the same values and ordering used by the original interval formula.
> Changed coefficients or breakpoints produce a distinct cache entry.

显式输出分支：

```python
(polynomial, sources)
```

读取的对象属性：

```python
PPoly.construct_fast
np.frombuffer(coefficients, dtype=np.float64).reshape
np.frombuffer
np.float64
curve.derivative
polynomial.c[:, index].astype
polynomial.c
np.longdouble
polynomial.x
```

写入目标：

```python
curve
polynomial
sources
index
```

直接调用符号：

```python
PPoly.construct_fast
np.frombuffer(coefficients, dtype=np.float64).reshape
np.frombuffer
curve.derivative
tuple
polynomial.c[:, index].astype
range
len
lru_cache
```

## `_polynomial_ranges`

[定义位置](../../../manipulation/interval_geometry.py#L44)。

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
curve.c.shape
curve.c
curve.c.tobytes
curve.x.tobytes
curve.x
polynomial.c.shape
polynomial.c
np.full
np.inf
np.full_like
np.searchsorted
polynomial.x
np.longdouble
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
sources
degree
lower
upper
first
stop
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
prepared_polynomial
curve.c.tobytes
curve.x.tobytes
np.full
np.full_like
max
int
np.searchsorted
min
len
range
float
np.longdouble
np.zeros_like
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

[定义位置](../../../manipulation/interval_geometry.py#L89)。

输入签名：

```python
def _progress_times(trajectory, values):
```

实现说明：

> Map progress events to all physical crossings, including plateaus.

显式输出分支：

```python
(progress_anchor_parameters(u, s, anchors)[:, 0] * trajectory.duration).tolist()
```

读取的对象属性：

```python
np.asarray
trajectory.progress_u
trajectory.progress_s
(progress_anchor_parameters(u, s, anchors)[:, 0] * trajectory.duration).tolist
trajectory.duration
```

写入目标：

```python
u
s
anchors
value
```

直接调用符号：

```python
np.asarray
float
(progress_anchor_parameters(u, s, anchors)[:, 0] * trajectory.duration).tolist
progress_anchor_parameters
```

## `SceneMotionBounds`

[定义位置](../../../manipulation/interval_geometry.py#L97)。

```python
class SceneMotionBounds():
```

类字段：

```python
scene: object
robot: object
attached_names: dict
owner_radii: dict
segment_bounds: dict = field(default_factory=dict)
```

实现说明：

> Point-speed coefficients bound to one scene's complete entity models.
>
> Local enclosing spheres are used only to integrate rotational velocity
> into displacement bounds. Collision queries retain the original native
> entities and original observed fields.

## `SceneMotionBounds.from_scene`

[定义位置](../../../manipulation/interval_geometry.py#L111)。

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
scene.kinematics.model
scene.kinematics
scene.robot_validator.T_reference_base
scene.robot_validator
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
build_geometry_motion_bound
cls
```

## `SceneMotionBounds.declaration_times`

[定义位置](../../../manipulation/interval_geometry.py#L150)。

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

[定义位置](../../../manipulation/interval_geometry.py#L168)。

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

[定义位置](../../../manipulation/interval_geometry.py#L182)。

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
self.segment_bounds
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
prepared.append
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
prepared
reach
i
left
right
linear
angle
speed
model
delta
arm
rotation
self.segment_bounds[owner.owner_id]
result
reciprocal
scale
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
prepared.append
max
```

## `SceneMotionBounds.interval`

[定义位置](../../../manipulation/interval_geometry.py#L231)。

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
self.scene.required_progress_anchors
self.scene.owners
owner.pose_interval
anchors.extend
owner.pose_path.progress
owner.pose_path
trajectory.progress_s
trajectory.progress_u
trajectory.duration
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
anchors
owner
a
b
_
s0
s1
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
list
anchors.extend
progress_query_parameters
self._owner_speed
isinstance
self.attached_names.values
speeds.pop
unavailable.pop
```

## `check_interval_geometry`

[定义位置](../../../manipulation/interval_geometry.py#L289)。

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
{'confirmed_intersections': intersections, 'proved_intervals': proved, 'unresolved_intervals': unresolved, 'query_points': consumed, 'initial_query_points': initial_points, 'query_batch_sizes': batch_sizes, 'queried_times_s': sorted(queried_times), 'spatial_step_m': float(spatial_step_m), 'scope': 'continuous_declared_geometry_model'}
```

读取的对象属性：

```python
np.isfinite
motion.declaration_times
np.asarray
motion.partition_times
batch_sizes.append
queried_times.update
state.time_s
unresolved.extend
cache.get
motion.interval
speeds.values
unresolved.append
first.batch
second.batch
first.pair_columns
second.pair_columns
first.source_rows
second.source_rows
maps[0].get
maps[1].get
blocks[0].descriptors
blocks[1].descriptors
entries.append
descriptor.display
failures.append
first.source
second.source
blocks[0].status
blocks[1].status
proofs.append
blocks[0].empty_set
blocks[1].empty_set
blocks[0].permission_kind
blocks[1].permission_kind
block.full_lower
block.full_lower_valid
block.lower
block.lower_valid
distance.append
proved.append
pending.extend
```

写入目标：

```python
cache
intersections
seen_intersections
pair_catalogs
queried_times
consumed
batch_sizes
initial_points
time
side
declarations
partition
pending
proved
unresolved
known_times
needed
start
end
remaining
requested
exhausted
response
used
state
a
b
current
states
key
bounds
speeds
unavailable
movement
base
first
second
blocks
maps
catalog_key
entries
descriptor
pair_catalogs[catalog_key]
failures
proofs
item
ia
ib
statuses
name
relative
permissions
finite_permission
distance
block
index
permitted
values
valid
change
value
reason
row
middle
```

直接调用符号：

```python
int
ValueError
np.isfinite
set
retain
len
motion.declaration_times
np.asarray
motion.partition_times
list
zip
sorted
boundary_keys
query_geometry
batch_sizes.append
queried_times.update
bool
unresolved.extend
float
cache.get
motion.interval
_mul_up
max
speeds.values
any
unresolved.append
id
maps[0].get
maps[1].get
entries.append
descriptor.display
failures.append
proofs.append
_add_up
distance.append
all
min
proved.append
pending.extend
```

## `check_interval_geometry.retain`

[定义位置](../../../manipulation/interval_geometry.py#L313)。

输入签名：

```python
def retain(states):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
state.time_s
state.side
state.batch
np.any
block.status
state.source_rows
state.tokens
block.descriptors
descriptor.key
seen_intersections.add
block.materialize
row.items
intersections.append
```

写入目标：

```python
state
key
cache[key]
block
source
column
descriptor
identity
row
pair
name
value
```

直接调用符号：

```python
any
np.any
state.tokens
seen_intersections.add
block.materialize
row.items
intersections.append
```

## `check_interval_geometry.boundary_keys`

[定义位置](../../../manipulation/interval_geometry.py#L337)。

输入签名：

```python
def boundary_keys(start, end):
```

显式输出分支：

```python
[(float(start), 1 if 1 in declarations.get(start, ()) and start > 0 else 0), (float(end), -1 if -1 in declarations.get(end, ()) and end < trajectory.duration else 0)]
```

读取的对象属性：

```python
declarations.get
trajectory.duration
```

写入目标：

无显式条目。

直接调用符号：

```python
float
declarations.get
```
