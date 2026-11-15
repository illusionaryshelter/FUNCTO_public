# `manipulation/trajectory_metrics.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../14-grounding-planning-portfolio.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `polynomial_extrema`

[定义位置](../../../manipulation/trajectory_metrics.py#L16)。

输入签名：

```python
def polynomial_extrema(curve):
```

实现说明：

> Measure both segment endpoints and all real stationary points.
>
> Each segment is evaluated independently, so both one-sided derivatives
> at a knot are included. This matters for piecewise cubic jerk, whose
> jumps do not represent a globally continuous jerk signal.

显式输出分支：

```python
rows
```

读取的对象属性：

```python
curve.c.shape
curve.c
curve.x
piece.derivative().roots
piece.derivative
np.r_
piece.x
np.isfinite
candidates.extend
rows.append
```

写入目标：

```python
rows
joint
candidates
segment
piece
stationary
times
t
minimum
maximum
```

直接调用符号：

```python
range
len
PPoly
piece.derivative().roots
piece.derivative
np.isfinite
candidates.extend
float
piece
min
max
rows.append
dict
```

## `polynomial_square_integral`

[定义位置](../../../manipulation/trajectory_metrics.py#L40)。

输入签名：

```python
def polynomial_square_integral(curve):
```

实现说明：

> Integrate each squared scalar polynomial exactly in physical seconds.

显式输出分支：

```python
total
```

读取的对象属性：

```python
np.zeros
curve.c.shape
curve.c
np.diff
curve.x
np.polymul
np.polyint
np.polyval
```

写入目标：

```python
total
segment
width
joint
squared
integral
total[joint]
```

直接调用符号：

```python
np.zeros
enumerate
np.diff
range
len
np.polymul
np.polyint
np.polyval
```

## `joint_metrics`

[定义位置](../../../manipulation/trajectory_metrics.py#L52)。

输入签名：

```python
def joint_metrics(curve, names, limits):
```

实现说明：

> Use the planner's saved named model bounds for all four derivatives.

显式输出分支：

```python
results
```

读取的对象属性：

```python
limits['joint_names'].index
np.asarray
curve.x
curve.derivative
row.update
np.mean
```

写入目标：

```python
columns
name
position_bounds
ranges
duration
results
order
key
derivative
bounds
extrema
index
row
lower
upper
low_margin
high_margin
use_lower
closest
closest_fraction
saved_validation
results[key]
```

直接调用符号：

```python
limits['joint_names'].index
np.asarray
float
enumerate
curve.derivative
polynomial_extrema
row.update
min
cubic_position_report
dict
max
abs
np.mean
polynomial_square_integral
```

## `robot_pair_metrics`

[定义位置](../../../manipulation/trajectory_metrics.py#L84)。

输入签名：

```python
def robot_pair_metrics(validator, curve, fingers, times):
```

实现说明：

> Query every required official pair, retaining pair and witness identity.
>
> The registered robot and its installed hand have fixed reusable official
> shapes and component relationships. Named joint curves and calibration
> place those shapes. Task objects remain in the original observed scene.
> A uniform clock plus saved knots and original scene witness time gives
> brackets; SciPy refines sampled distance minima. This numerical review
> describes these saved robot curves and leaves original acceptance intact.

显式输出分支：

```python
(dict(minimum=min(minima, key=lambda r: r['evidence']['signed_distance_m']), pair_minima=minima, sample_count=len(times), pair_count=len(pairs), numerical_boundary_m=boundary, scope='all required official robot pairs; saved-curve samples and bracketed local minima'), distances)
```

读取的对象属性：

```python
validator.checker
validator.required_pairs
validator.arm_joint_names
validator.finger_joint_names
np.column_stack
np.concatenate
checker.query_pair_batches
validator.T_reference_base
np.zeros
checker.numerics
values.argmin
refined.fun
refined.x
minima.append
```

写入目标：

```python
checker
pairs
names
positions
numeric
distances
boundary
minima
column
pair
values
initial
best
peaks
_
index
refined
evidence
```

直接调用符号：

```python
np.column_stack
curve
fingers
np.concatenate
list
checker.query_pair_batches
np.zeros
len
max
enumerate
int
values.argmin
float
find_peaks
minimize_scalar
query
dict
min
minima.append
```

## `robot_pair_metrics.query`

[定义位置](../../../manipulation/trajectory_metrics.py#L108)。

输入签名：

```python
def query(t):
```

显式输出分支：

```python
report['pairs'][0]
```

读取的对象属性：

```python
checker.query
np.r_
validator.T_reference_base
```

写入目标：

```python
report
```

直接调用符号：

```python
checker.query
curve
fingers
```

## `movement`

[定义位置](../../../manipulation/trajectory_metrics.py#L128)。

输入签名：

```python
def movement(poses):
```

实现说明：

> Measure full sampled TCP travel; SO(3) angles use consecutive matrices.

显式输出分支：

```python
dict(length_m=float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()), rotation_travel_deg=float(np.degrees(Rotation.from_matrix(poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude().sum())))
```

读取的对象属性：

```python
np.asarray
np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum
np.linalg.norm
np.linalg
np.diff
np.degrees
Rotation.from_matrix(poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude().sum
Rotation.from_matrix(poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude
Rotation.from_matrix
poses[:-1, :3, :3].transpose
```

写入目标：

```python
poses
```

直接调用符号：

```python
np.asarray
dict
float
np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum
np.linalg.norm
np.diff
np.degrees
Rotation.from_matrix(poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude().sum
Rotation.from_matrix(poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude
Rotation.from_matrix
poses[:-1, :3, :3].transpose
```

## `deviation`

[定义位置](../../../manipulation/trajectory_metrics.py#L136)。

输入签名：

```python
def deviation(poses, reference, times):
```

实现说明：

> Compare the same physical instants in calibrated W, not Euler differences.

显式输出分支：

```python
dict(position_rms_m=float(np.sqrt(np.mean(p * p))), position_max_m=float(p.max()), position_max_time_s=float(times[p.argmax()]), rotation_rms_deg=float(np.sqrt(np.mean(r * r))), rotation_max_deg=float(r.max()), rotation_max_time_s=float(times[r.argmax()]))
```

读取的对象属性：

```python
np.linalg.norm
np.linalg
np.degrees
Rotation.from_matrix(reference[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3]).magnitude
Rotation.from_matrix
reference[:, :3, :3].transpose
np.sqrt
np.mean
p.max
p.argmax
r.max
r.argmax
```

写入目标：

```python
p
r
```

直接调用符号：

```python
np.linalg.norm
np.degrees
Rotation.from_matrix(reference[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3]).magnitude
Rotation.from_matrix
reference[:, :3, :3].transpose
dict
float
np.sqrt
np.mean
p.max
p.argmax
r.max
r.argmax
```

## `robot_tool_poses`

[定义位置](../../../manipulation/trajectory_metrics.py#L145)。

输入签名：

```python
def robot_tool_poses(fixture, feedback, names, positions):
```

实现说明：

> Evaluate official URDF FK at each original named joint sample.
>
> This is the same fixed robot model used by the original recorded-curve
> measurement. It supplies robot kinematics only; task entities retain
> their independent RGB-D fields. Each sample uses a local mutable FK data
> object, so one hypothesis cannot overwrite another hypothesis's poses.

显式输出分支：

```python
np.asarray(poses)
```

读取的对象属性：

```python
pin.buildModelFromUrdf
model.createData
pin.neutral
model.joints[model.getJointId(str(name))].idx_q
model.joints
model.getJointId
model.getFrameId
np.asarray
pin.forwardKinematics
pin.updateFramePlacements
poses.append
data.oMf[tcp].homogeneous
data.oMf
```

写入目标：

```python
model
data
q
slots
name
tcp
base
tool
poses
values
q[slots]
```

直接调用符号：

```python
pin.buildModelFromUrdf
model.createData
pin.neutral
model.getJointId
str
model.getFrameId
np.asarray
pin.forwardKinematics
pin.updateFramePlacements
poses.append
```

## `measure`

[定义位置](../../../manipulation/trajectory_metrics.py#L168)。

输入签名：

```python
def measure(result, *, hypothesis_id, fixture, feedback, source_report, source_curve, curve=None):
```

实现说明：

> Measure a formal or diagnostic original curve without changing status.
>
> A caller may supply the producer's diagnostic curve for a rejected result.
> Missing curves produce a missing measurement record. Robot required-pair
> queries use 1001 uniform times, original arm/finger knots and the original
> scene witness, followed by the unchanged bracketed scalar refinement.
> Arm-joint position margins retain radians and exclude prismatic fingers.

显式输出分支：

```python
({**base, 'metrics_status': 'missing_curve', 'failure': result.diagnostics.get('failure')}, {})
(values, dict(times=times, tool_poses=poses, required_pair_distances_m=distances))
```

读取的对象属性：

```python
result.trajectory
result.success
result.diagnostics.get
result.diagnostics
curve.times
curve.position
curve.velocity
curve.finger_trajectory
finger.times
finger.position
finger.velocity
np.unique
np.r_
np.linspace
np.asarray
curve.joint_names
finger.joint_names
np.column_stack
curve.duration
```

写入目标：

```python
curve
base
d
arm
finger
fingers
witness
times
validator
native
distances
poses
values
key
```

直接调用符号：

```python
dict
bool
str
result.diagnostics.get
CubicHermiteSpline
np.unique
np.linspace
build_robot_validator
np.asarray
robot_pair_metrics
robot_tool_poses
list
np.column_stack
arm
fingers
joint_metrics
movement
float
```
