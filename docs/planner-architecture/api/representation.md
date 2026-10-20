# `manipulation/representation.py` 实现契约

代码基线：`cf7182cbf3888d9db2e7553df47dce9a0b1c339f`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_timeline`

[定义位置](../../../manipulation/representation.py#L18)。

输入签名：

```python
def _timeline(timestamps):
```

显式输出分支：

```python
validate_timeline(np.arange(len(times)), times)[1]
```

读取的对象属性：

```python
np.asarray
np.arange
```

写入目标：

```python
times
```

直接调用符号：

```python
np.asarray
validate_timeline
np.arange
len
```

## `_progress`

[定义位置](../../../manipulation/representation.py#L23)。

输入签名：

```python
def _progress(times):
```

显式输出分支：

```python
np.zeros(1) if len(times) == 1 else (times - times[0]) / (times[-1] - times[0])
```

读取的对象属性：

```python
np.zeros
```

写入目标：

无显式条目。

直接调用符号：

```python
len
np.zeros
```

## `_aligned_poses`

[定义位置](../../../manipulation/representation.py#L27)。

输入签名：

```python
def _aligned_poses(value, times, name):
```

显式输出分支：

```python
poses
```

读取的对象属性：

无显式条目。

写入目标：

```python
poses
```

直接调用符号：

```python
_poses_array
len
ValueError
```

## `sample_poses`

[定义位置](../../../manipulation/representation.py#L34)。

输入签名：

```python
def sample_poses(poses, timestamps, query_times):
```

实现说明：

> Interpolate one actual pose track at original seconds, without clamping.
> 
> Reuse PosePath's translation/Slerp interpolation.  A one-pose observation
> remains one observation: it cannot be extrapolated to a longer recording.

显式输出分支：

```python
path.sample_numpy(progress)
```

读取的对象属性：

```python
np.asarray
np.isfinite(query).all
np.isfinite
np.any
np.zeros_like
path.sample_numpy
```

写入目标：

```python
times
values
query
path
progress
```

直接调用符号：

```python
_timeline
_aligned_poses
np.asarray
np.isfinite(query).all
np.isfinite
np.any
ValueError
PosePath
_progress
len
np.zeros_like
path.sample_numpy
```

## `trajectory_summary`

[定义位置](../../../manipulation/representation.py#L51)。

输入签名：

```python
def trajectory_summary(poses, timestamps, *, approach_time=None, interaction_time=None):
```

实现说明：

> Read interaction pose and displacement direction from this same track.
> 
> The returned direction describes the origin of the supplied pose, not the
> surface normal or another attached origin.  Missing events or zero net
> displacement do not discard the complete motion.

显式输出分支：

```python
_sampled_summary(lambda time: sample_poses(values, times, time), approach_time, interaction_time)
```

读取的对象属性：

无显式条目。

写入目标：

```python
times
values
```

直接调用符号：

```python
_timeline
_aligned_poses
_sampled_summary
sample_poses
```

## `_sampled_summary`

[定义位置](../../../manipulation/representation.py#L64)。

输入签名：

```python
def _sampled_summary(sample, approach_time, interaction_time):
```

实现说明：

> Read both summary values from the actual track's sampling operation.

显式输出分支：

```python
result
result
```

读取的对象属性：

```python
np.linalg.norm
np.linalg
np.finfo(float).eps
np.finfo
```

写入目标：

```python
result
result['missing']['interaction_pose']
result['interaction_pose']
result['missing']['approach_direction']
start
delta
length
result['approach_direction']
```

直接调用符号：

```python
sample
float
ValueError
np.linalg.norm
np.finfo
max
```

## `_relative`

[定义位置](../../../manipulation/representation.py#L88)。

输入签名：

```python
def _relative(parent, child):
```

实现说明：

> Synchronized parent-from-child poses; no initial-pose normalization.

显式输出分支：

```python
np.stack([_inverse(p) @ c for (p, c) in zip(parent, child)])
```

读取的对象属性：

```python
np.stack
```

写入目标：

```python
p
c
```

直接调用符号：

```python
np.stack
_inverse
zip
```

## `build_representation`

[定义位置](../../../manipulation/representation.py#L93)。

输入签名：

```python
def build_representation(*, timestamps, frame_indices, kind, T_world_carrier=None, T_world_reference=None, T_world_eef=None, L_G=None, L_R=None, approach_time=None, interaction_time=None, release_time=None, events=(), provenance=None):
```

实现说明：

> Compile available synchronized observations into the PRD representation.
> 
> ``carrier`` means the grasped body for grasp and the manipulated body for
> manipulation. ``reference`` is the physical body carrying F_R. L_G and L_R
> map functional coordinates into their respective fixed body coordinates.
> Raw tracks, original frame IDs/times and events remain in the result.  A
> missing source E does not block H_m; no event or pose is manufactured.

显式输出分支：

```python
result
```

读取的对象属性：

```python
np.isfinite
```

写入目标：

```python
frames
times
result
name
value
result[name]
result['C_D']
result['H_g']
result['missing']['H_g']
key
relative
result['H_m']
result['missing']['H_m']
result['missing']['C_D']
initial
result['summaries'][name]
```

直接调用符号：

```python
ValueError
validate_timeline
_progress
deepcopy
list
_aligned_poses
_pose_array
np.isfinite
_relative
_inverse
all
_sampled_summary
sample_poses
trajectory_summary
```

## `_component_path`

[定义位置](../../../manipulation/representation.py#L162)。

输入签名：

```python
def _component_path(value, times, path_times, *, source, parent=None, attachment=None):
```

实现说明：

> Interpolate one relation before composing its parent/right attachment.
> 
> All components share the same physical-time interval. At release, each
> component is independently restricted to that interval; interpolating the
> already multiplied object/EEF samples would cut rotating lever-arm arcs.

显式输出分支：

```python
PosePath(_progress(path_times), values, source=source, parent=parent, attachment=np.eye(4) if attachment is None else attachment)
```

读取的对象属性：

```python
np.asarray
poses.shape
np.repeat
np.eye
```

写入目标：

```python
poses
values
```

直接调用符号：

```python
np.asarray
np.repeat
_pose_array
len
sample_poses
PosePath
_progress
np.eye
```

## `_right_path`

[定义位置](../../../manipulation/representation.py#L179)。

输入签名：

```python
def _right_path(parent, transform, source):
```

实现说明：

> Sample the physical parent before applying a fixed attached relation.

显式输出分支：

```python
PosePath(parent.progress, np.repeat(np.eye(4)[None], len(parent.progress), axis=0), source=source, parent=parent, attachment=transform)
```

读取的对象属性：

```python
parent.progress
np.repeat
np.eye
```

写入目标：

无显式条目。

直接调用符号：

```python
PosePath
np.repeat
np.eye
len
```

## `_require_same_pose`

[定义位置](../../../manipulation/representation.py#L185)。

输入签名：

```python
def _require_same_pose(first, second, message):
```

实现说明：

> Check geometric arithmetic, without inventing a physical task tolerance.

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
position
rotation
```

直接调用符号：

```python
_errors
max
ValueError
```

## `_boundary_record`

[定义位置](../../../manipulation/representation.py#L192)。

输入签名：

```python
def _boundary_record(path, current, goal, event_progress):
```

实现说明：

> Compare to supplied entry evidence and a separately bound nominal goal.

显式输出分支：

```python
dict(status='passed', actual_start=current, adapted_start=start, nominal_goal=goal, adapted_interaction=event, start_position_error_m=float(p0[0]), start_rotation_error_rad=float(r0[0]), interaction_position_error_m=float(pg[0]), interaction_rotation_error_rad=float(rg[0]), scope='Geometric boundaries; not physical task or robot feasibility verification')
```

读取的对象属性：

```python
path.sample_numpy
```

写入目标：

```python
start
event
p0
r0
pg
rg
```

直接调用符号：

```python
path.sample_numpy
_require_same_pose
_errors
dict
float
```

## `_adapt_boundary`

[定义位置](../../../manipulation/representation.py#L206)。

输入签名：

```python
def _adapt_boundary(path, current, event_progress, source):
```

实现说明：

> Use the shared transfer kernel; an event at entry must already agree.

显式输出分支：

```python
path
TransferredPosePath(path, current, [{'progress': event_progress, 'pose': goal}], source=source)
```

读取的对象属性：

```python
path.sample_numpy
path.progress
```

写入目标：

```python
goal
```

直接调用符号：

```python
path.sample_numpy
len
_require_same_pose
TransferredPosePath
```

## `generate_reference`

[定义位置](../../../manipulation/representation.py#L216)。

输入签名：

```python
def generate_reference(representation, *, T_world_function=None, J=None, A=None, T_world_eef=None, T_world_body=None, motion_model=None, source_coordinates=None, current_coordinates=None, contact_motion=False, held_until_time=None):
```

实现说明：

> Adapt source motion to the actual entry AND nominal interaction pose.
> 
> J maps current body coordinates B_N into demonstrated coordinates B_D;
> # BEGIN A is the declared rigid attachment model, not physical truth.
> A maps the modeled body B_N into tool E. Robot feedback E plus A determines
> the modeled body pose. If both poses are supplied, they must use this same
> model; the equality does not verify physical holding or target arrival.
> # END no simulator body measurement is required to bind held motion.
> Free object motion is adapted in a fixed material frame, independently of
> the body's coordinate chart and the grasp. The original motion first binds
> the nominal goal at interaction_time; that goal is not inferred from the
> adapted output. All intermediate samples use the shared transfer kernel.
> 
> Grasp motion requires actual E. A scalar mechanism instead uses its current
> model and current coordinate, preserving its physical manifold and winding.
> Production object/tool paths stop at release or an earlier held_until_time.
> Full source evidence and unadapted bound diagnostics remain separately named.
> Missing entry/interaction evidence cannot produce a production path. These
> are geometric references, not collision, support or robot-execution proofs.

显式输出分支：

```python
result
result
result
```

读取的对象属性：

```python
representation.get
np.isfinite
np.unique
np.r_
np.asarray(T_world_function).shape
np.asarray
result.get
raw.sample_numpy
motion_model.kind
coordinates.shape
np.asarray(current_coordinates, dtype=float).reshape
current.shape
np.isfinite(coordinates).all
np.isfinite(current).all
coordinates.copy
current.copy
motion_model.poses
adapted_body.coordinates.copy
adapted_body.coordinates
raw_body.sample_numpy
raw_eef.sample_numpy
adapted_eef.sample_numpy
```

写入目标：

```python
frames
times
result
kind
extra
key
value
full_times
ends
end
path_times
interaction
name
result[name]
result['T_world_function']
result['missing']['interaction_time']
event_progress
current_body
current_eef
result['missing']['eef_poses']
raw
adapted
result['method']
raw_body
adapted_body
result['missing']['object_poses']
coordinates
current
result['source_coordinates']
result['current_coordinates']
model_entry
model_eef
result['coordinates']
H0
Q
Qc
L_body
result['L_body_material']
raw_material
adapted_material
inverse_A
raw_eef
relation
adapted_eef
result['status']
```

直接调用符号：

```python
validate_timeline
dict
deepcopy
representation.get
ValueError
float
any
np.isfinite
np.unique
min
_pose_array
_aligned_poses
np.asarray
len
result.get
_inverse
_require_same_pose
_component_path
function_path
diagnostic
_adapt_boundary
publish
raw.sample_numpy
isinstance
np.asarray(current_coordinates, dtype=float).reshape
np.isfinite(coordinates).all
np.isfinite(current).all
coordinates.copy
current.copy
motion_model.poses
mechanism_path
adapted_body.coordinates.copy
sample_poses
material_path
_right_path
raw_body.sample_numpy
raw_eef.sample_numpy
adapted_eef.sample_numpy
```

## `generate_reference.function_path`

[定义位置](../../../manipulation/representation.py#L287)。

输入签名：

```python
def function_path(clock):
```

显式输出分支：

```python
_component_path(result['T_world_function'], times, clock, source='current functional frame F')
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
_component_path
```

## `generate_reference.diagnostic`

[定义位置](../../../manipulation/representation.py#L290)。

输入签名：

```python
def diagnostic(name, path, clock):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
clock.copy
path.sample_numpy
```

写入目标：

```python
result['bound_' + name + '_timestamps']
result['bound_' + name + '_poses']
```

直接调用符号：

```python
clock.copy
path.sample_numpy
_progress
```

## `generate_reference.publish`

[定义位置](../../../manipulation/representation.py#L294)。

输入签名：

```python
def publish(name, path, current, nominal_goal):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
path_times.copy
path.sample_numpy
representation.get
```

写入目标：

```python
result['boundary_checks'][name]
result[name + '_path']
result[name + '_timestamps']
result[name + '_poses']
approach
result['summaries'][name]
```

直接调用符号：

```python
_boundary_record
path_times.copy
path.sample_numpy
_progress
representation.get
_sampled_summary
```

## `generate_reference.publish.sample`

[定义位置](../../../manipulation/representation.py#L302)。

输入签名：

```python
def sample(time):
```

显式输出分支：

```python
path.sample_numpy(progress)
```

读取的对象属性：

```python
path.sample_numpy
```

写入目标：

```python
progress
```

直接调用符号：

```python
len
path.sample_numpy
```
