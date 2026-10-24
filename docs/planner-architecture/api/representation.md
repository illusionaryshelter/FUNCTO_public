# `manipulation/representation.py` 实现契约

代码基线：本次实现提交同步更新正文和源码；前置代码检查点为 `759d89b`。实际实现提交号由紧随的文档引用提交回填。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_timeline`

[定义位置](../../../manipulation/representation.py#L22)。

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

[定义位置](../../../manipulation/representation.py#L27)。

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

[定义位置](../../../manipulation/representation.py#L31)。

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

[定义位置](../../../manipulation/representation.py#L38)。

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

[定义位置](../../../manipulation/representation.py#L55)。

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

[定义位置](../../../manipulation/representation.py#L68)。

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

[定义位置](../../../manipulation/representation.py#L92)。

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

[定义位置](../../../manipulation/representation.py#L97)。

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

[定义位置](../../../manipulation/representation.py#L166)。

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

[定义位置](../../../manipulation/representation.py#L183)。

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

[定义位置](../../../manipulation/representation.py#L189)。

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

## `_RotationOffsetPath`

[定义位置](../../../manipulation/representation.py#L197)。

```python
class _RotationOffsetPath(PosePath):
```

类字段：

无显式条目。

实现说明：

> Rotate a material frame in F while preserving its world position curve.
>
> ``nominal`` gives T_W_M(s), ``functional`` gives T_W_F(s), and ``rotation``
> is the constant left correction D expressed in F. This path evaluates
> R_W_M(s) = R_W_F(s) D R_W_F(s)^T R_nominal(s), with the original material
> origin p_W_M(s). Sampling both parents before composition preserves their
> continuous source interpolation and progress derivatives. Body and tool
> offsets are applied by the ordinary child PosePath after this operation.

## `_RotationOffsetPath.__init__`

[定义位置](../../../manipulation/representation.py#L208)。

输入签名：

```python
def __init__(self, nominal, functional, rotation, *, source):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
Rotation.from_matrix(rotation).as_quat
Rotation.from_matrix
nominal.sample_numpy
nominal.progress
functional.sample_numpy
np.swapaxes
super().__init__
```

写入目标：

```python
self.nominal
self.functional
self.rotation_quaternion
poses
frame
poses[:, :3, :3]
```

直接调用符号：

```python
Rotation.from_matrix(rotation).as_quat
Rotation.from_matrix
nominal.sample_numpy
functional.sample_numpy
np.swapaxes
super().__init__
super
```

## `_RotationOffsetPath.sample`

[定义位置](../../../manipulation/representation.py#L219)。

输入签名：

```python
def sample(self, progress, *, side='right'):
```

显式输出分支：

```python
(position, quaternion_multiply(functional, corrected))
```

读取的对象属性：

```python
self.nominal.sample
self.nominal
self.functional.sample
self.functional
self.rotation_quaternion
```

写入目标：

```python
position
orientation
_
functional
local
corrected
```

直接调用符号：

```python
self.nominal.sample
self.functional.sample
quaternion_multiply
quaternion_conjugate
_tensor
```

## `_initial_rotation_offset`

[定义位置](../../../manipulation/representation.py#L227)。

输入签名：

```python
def _initial_rotation_offset(nominal, functional, current, flags):
```

实现说明：

> Select the initial RPY anchor, then retain the full relative rotation.
>
> All rotations here map the same moving material frame M into functional
> reference F. ``flags`` is [roll, pitch, yaw] in SciPy's extrinsic ``xyz``
> convention: R = Rz(yaw) Ry(pitch) Rx(roll). True selects the current-run
> initial component; False selects the functional nominal initial component.
> The selection occurs once at entry. D = R_selected R_nominal(0)^T then
> carries the full SO(3) relative motion, including noncommuting rotations.
> For a yaw-only selection, D is a pure F-z rotation throughout the path.

显式输出分支：

```python
(rotation, {'is_initial_relative_rotation_rpy': flags.tolist(), 'rpy_convention': 'extrinsic xyz; Rz(yaw) Ry(pitch) Rx(roll); radians', 'rotation_frame': 'current functional reference F', 'moving_frame': 'fixed material frame M; calibrated tool E for a tool reference', 'nominal_initial_rpy_rad': nominal_rpy, 'current_initial_rpy_rad': current_rpy, 'selected_initial_rpy_rad': selected, 'functional_rotation_offset': rotation, 'meaning': 'select initial RPY anchor, preserve complete initial-relative SO(3) motion'})
```

读取的对象属性：

```python
functional.sample_numpy
frame.T
nominal.sample_numpy
Rotation.from_matrix(nominal_initial).as_euler
Rotation.from_matrix
Rotation.from_matrix(current_initial).as_euler
np.where
Rotation.from_euler('xyz', selected).as_matrix
Rotation.from_euler
nominal_initial.T
flags.tolist
```

写入目标：

```python
frame
nominal_initial
current_initial
nominal_rpy
current_rpy
selected
rotation
```

直接调用符号：

```python
functional.sample_numpy
nominal.sample_numpy
Rotation.from_matrix(nominal_initial).as_euler
Rotation.from_matrix
Rotation.from_matrix(current_initial).as_euler
np.where
Rotation.from_euler('xyz', selected).as_matrix
Rotation.from_euler
flags.tolist
```

## `generate_reference`

[定义位置](../../../manipulation/representation.py#L259)。

输入签名：

```python
def generate_reference(representation, *, T_world_function=None, J=None, A=None, T_world_eef=None, T_world_body=None, motion_model=None, source_coordinates=None, current_coordinates=None, contact_motion=False, held_until_time=None):
```

实现说明：

> Bind complete nominal source motion and declare entry/event boundaries.
>
> J maps current body coordinates B_N into demonstrated coordinates B_D;
> # BEGIN A is the declared rigid attachment model, not physical truth.
> A maps the modeled body B_N into tool E. Robot feedback E plus A determines
> the modeled body pose. If both poses are supplied, they must use this same
> model; the equality does not verify physical holding or target arrival.
> # END no simulator body measurement is required to bind held motion.
> Free object motion is bound in a fixed material frame, independently of
> the body's coordinate chart and the grasp. The original motion and any
> explicit initial-relative rotation declaration supply the nominal goal.
> Returned paths retain this complete nominal composition; their separately
> named boundaries enter BoundToolReference
> only after the planner has bound physical time and measured TCP velocity.
>
> Grasp motion requires actual E. A scalar mechanism instead uses its current
> model and current coordinate, preserving its physical manifold and winding.
> Production object/tool paths stop at release or an earlier held_until_time.
> Full source evidence and generated bound diagnostics remain separately named.
> Missing entry/interaction evidence cannot produce a boundary declaration.
> Reference geometry and its boundary data remain separate from task/scene
> qualification and the eventual planned joint trajectory.

显式输出分支：

```python
result
result
result
```

读取的对象属性：

```python
representation.get
np.asarray
rotation_flags.shape
rotation_flags.dtype
np.dtype
np.isfinite
np.unique
np.r_
np.asarray(T_world_function).shape
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
raw_body.coordinates.copy
raw_body.coordinates
raw_body.sample_numpy
raw_eef.sample_numpy
```

写入目标：

```python
frames
times
result
kind
rotation_flags
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
correction
result['rotation_transfer']
result['method']
raw_body
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
original
raw_material
inverse_A
raw_eef
relation
result['status']
```

直接调用符号：

```python
validate_timeline
dict
deepcopy
representation.get
ValueError
np.asarray
np.dtype
float
any
np.isfinite
np.unique
min
_pose_array
_aligned_poses
len
result.get
_inverse
_require_same_pose
_component_path
function_path
_initial_rotation_offset
_RotationOffsetPath
diagnostic
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
raw_body.coordinates.copy
sample_poses
material_path
_right_path
raw_body.sample_numpy
raw_eef.sample_numpy
```

## `generate_reference.function_path`

[定义位置](../../../manipulation/representation.py#L345)。

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

[定义位置](../../../manipulation/representation.py#L348)。

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

[定义位置](../../../manipulation/representation.py#L352)。

输入签名：

```python
def publish(name, path, current, nominal_goal):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
path_times.tolist
np.asarray(current).copy
np.asarray
np.asarray(nominal_goal).copy
path_times.copy
path.sample_numpy
representation.get
```

写入目标：

```python
source
result[name + '_boundaries']
result['boundary_checks'][name]
result[name + '_path']
result[name + '_timestamps']
result[name + '_poses']
approach
result['summaries'][name]
```

直接调用符号：

```python
float
path_times.tolist
np.asarray(current).copy
np.asarray
np.asarray(nominal_goal).copy
path_times.copy
path.sample_numpy
_progress
representation.get
_sampled_summary
```

## `generate_reference.publish.sample`

[定义位置](../../../manipulation/representation.py#L369)。

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

## `generate_reference.mechanism_path`

[定义位置](../../../manipulation/representation.py#L408)。

输入签名：

```python
def mechanism_path(clock):
```

显式输出分支：

```python
PosePath(_progress(clock), motion_model.poses(values), source='current mechanism FK from source scalar displacement', motion_model=motion_model, coordinates=values)
```

读取的对象属性：

```python
np.interp
motion_model.poses
```

写入目标：

```python
values
```

直接调用符号：

```python
np.interp
PosePath
_progress
motion_model.poses
```

## `generate_reference.material_path`

[定义位置](../../../manipulation/representation.py#L452)。

输入签名：

```python
def material_path(clock):
```

显式输出分支：

```python
_RotationOffsetPath(nominal, functional, correction, source=result['rotation_transfer'])
nominal
```

读取的对象属性：

无显式条目。

写入目标：

```python
functional
nominal
```

直接调用符号：

```python
function_path
_component_path
_inverse
_RotationOffsetPath
```
