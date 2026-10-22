# `manipulation/reference_tracking.py` 实现契约

代码基线：`a56c3e4`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `physical_time_mean_weights`

[定义位置](../../../manipulation/reference_tracking.py#L27)。

输入签名：

```python
def physical_time_mean_weights(sample_times, valid_count):
```

实现说明：

> Return normalized trapezoid weights for valid ordinary samples.
>
> ``sample_times`` is [A,H] in seconds, and ``valid_count`` is [A]. Each
> row has at least two strictly increasing valid times. Entries after its
> valid prefix are storage padding. The caller supplies the ordinary
> trajectory grid; event and collision-witness samples have another role.
> This function is called when binding a grid, outside optimizer capture.

显式输出分支：

```python
weight / weight.sum(-1, keepdim=True)
```

读取的对象属性：

```python
sample_times.ndim
valid_count.shape
sample_times.shape
((valid_count < 2) | (valid_count > horizon)).any
torch.arange
sample_times.device
(intervals & (~torch.isfinite(delta) | (delta <= 0))).any
torch.isfinite
torch.where
sample_times.new_zeros
torch.cat
weight.sum
```

写入目标：

```python
horizon
intervals
delta
half
zero
weight
```

直接调用符号：

```python
ValueError
bool
((valid_count < 2) | (valid_count > horizon)).any
torch.arange
(intervals & (~torch.isfinite(delta) | (delta <= 0))).any
torch.isfinite
torch.where
sample_times.new_zeros
torch.cat
weight.sum
```

## `freeze_cspace_time_weights`

[定义位置](../../../manipulation/reference_tracking.py#L52)。

输入签名：

```python
def freeze_cspace_time_weights(config, native_dt):
```

实现说明：

> Bind a STATE joint-cost configuration to one physical rollout clock.
>
> ``config`` is a fresh native CSpaceCostCfg copied from the request's
> ordinary-cost template, before constructing StateCSpaceCost. The fixed
> ``native_dt`` is the time interval generating the admitted ordinary
> optimization states, not the interval of a later scoring/refinement grid.
>
> The installed kernel retimes bound weights [position,velocity,acceleration,
> jerk,effort] by [1,dt,dt^2,dt^3,1], and regularization weights [velocity,
> acceleration,jerk,effort,energy] by [dt,dt^2,dt^3,1,dt]. Apply each enabled
> rule once to the configuration, then disable runtime retiming. Every
> evaluation grid consequently measures the same physical state cost.
> External quadrature also needs use_grad_input=True in its native kernel.
> The mutated configuration is returned for immediate cost construction;
> rebuilding from the original template owns subsequent request rebinding.

显式输出分支：

```python
config
```

读取的对象属性：

```python
config.cost_type
CSpaceCostType.STATE
config.retime_weights
config.weight
config.weight.new_tensor
config.retime_regularization_weights
config.squared_l2_regularization_weight
values.new_tensor
```

写入目标：

```python
config.weight
values
config.squared_l2_regularization_weight
config.retime_weights
config.retime_regularization_weights
config.use_grad_input
```

直接调用符号：

```python
ValueError
config.weight.new_tensor
values.new_tensor
```

## `FixedTimeStateCSpaceCost`

[定义位置](../../../manipulation/reference_tracking.py#L85)。

```python
class FixedTimeStateCSpaceCost(StateCSpaceCost):
```

类字段：

无显式条目。

实现说明：

> Evaluate the native STATE cost at its declared coefficient time scale.
>
> Even with weight retiming disabled, the native energy term is
> weight*(torque*velocity*dt)^2. Its dt must consequently remain the same
> fixed ordinary-rollout interval used by ``freeze_cspace_time_weights``.
> Only this cost's JointState view receives that dt: the source tensors,
> physical sample times, transition model and other managers retain their
> original clock. All value and derivative calculations remain native.

## `FixedTimeStateCSpaceCost.__init__`

[定义位置](../../../manipulation/reference_tracking.py#L96)。

输入签名：

```python
def __init__(self, config, native_dt):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
```

写入目标：

```python
self.native_dt
```

直接调用符号：

```python
super().__init__
super
```

## `FixedTimeStateCSpaceCost.setup_batch_tensors`

[定义位置](../../../manipulation/reference_tracking.py#L100)。

输入签名：

```python
def setup_batch_tensors(self, batch, horizon):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._batch_size
torch.full
self.native_dt
self.device_cfg.as_torch_dict
self.device_cfg
super().setup_batch_tensors
```

写入目标：

```python
self._fixed_dt
```

直接调用符号：

```python
torch.full
self.device_cfg.as_torch_dict
super().setup_batch_tensors
super
```

## `FixedTimeStateCSpaceCost.forward`

[定义位置](../../../manipulation/reference_tracking.py#L106)。

输入签名：

```python
def forward(self, state_batch, *args, **kwargs):
```

显式输出分支：

```python
super().forward(replace(state_batch, dt=self._fixed_dt), *args, **kwargs)
```

读取的对象属性：

```python
super().forward
self._fixed_dt
```

写入目标：

无显式条目。

直接调用符号：

```python
super().forward
super
replace
```

## `NativeRigidToolTransform`

[定义位置](../../../manipulation/reference_tracking.py#L110)。

```python
class NativeRigidToolTransform(torch.autograd.Function):
```

类字段：

无显式条目。

实现说明：

> Compose a fixed TCP calibration while preserving native FK's VJP.
>
> Both input and output quaternion derivatives use cuRobo's pose-cost
> protocol, not Euclidean derivatives with respect to four independent
> coordinates. For unit wxyz q, the native decoder is
> D(q,g)=(-v*g_w+w*g_v-v cross g_v)/2. Its inverse on angular covectors
> tau is E(q,tau)=2*q*[0,tau]. Native FK contracts D with its base-frame
> geometric angular Jacobian. The installed ToolPoseCost emits E.
>
> With p_T=p_L+R_L*t and q_T=q_L*q_fixed, a position covector f at the
> TCP contributes the link angular covector (R_L*t) cross f. Orientation
> contributes D(q_T,g_T). Their sum is encoded by E(q_L,...) for the
> existing native FK. All vectors in that calculation use the robot base
> frame. The constant transform has no optimization variables.

## `NativeRigidToolTransform.forward`

[定义位置](../../../manipulation/reference_tracking.py#L128)。

输入签名：

```python
def forward(ctx, position, quaternion, translation, rotation):
```

显式输出分支：

```python
(position + offset, tool_quaternion)
```

读取的对象属性：

```python
ctx.save_for_backward
```

写入目标：

```python
offset
tool_quaternion
```

直接调用符号：

```python
quaternion_rotate
quaternion_multiply
ctx.save_for_backward
```

## `NativeRigidToolTransform.backward`

[定义位置](../../../manipulation/reference_tracking.py#L136)。

输入签名：

```python
def backward(ctx, position_gradient, quaternion_gradient):
```

显式输出分支：

```python
(position_gradient.contiguous(), native_quaternion_gradient.contiguous(), None, None)
```

读取的对象属性：

```python
ctx.saved_tensors
torch.linalg.cross
torch.linalg
torch.cat
torch.zeros_like
position_gradient.contiguous
native_quaternion_gradient.contiguous
torch.autograd.function.once_differentiable
torch.autograd.function
torch.autograd
```

写入目标：

```python
link_quaternion
tool_quaternion
offset
w
vector
gw
gv
angular
pure_angular
native_quaternion_gradient
```

直接调用符号：

```python
torch.linalg.cross
torch.cat
torch.zeros_like
quaternion_multiply
position_gradient.contiguous
native_quaternion_gradient.contiguous
```

## `NativeReferenceCost`

[定义位置](../../../manipulation/reference_tracking.py#L147)。

```python
class NativeReferenceCost():
```

类字段：

无显式条目。

实现说明：

> Bind a fixed time grid, then evaluate its native reference objective.
>
> ``pose_weight`` is the pair already declared by the native trajectory
> optimizer: its point energy is wp*||delta_p||^2/2 + wr*theta^2. The
> installed Lie kernel evaluates theta^2/4, so its rotation weight is 4*wr.
> Rotation axes all remain one: changing them would invoke the installed
> kernel's incorrect anisotropic rotation derivative. Zero convergence
> tolerances retain a continuous soft objective everywhere.
>
> ``bind`` is the request/grid lifecycle boundary. Its [A,H] reference
> rows may be shared by any number of seed or line-search rows. ``forward``
> accepts their explicit row mapping when A is greater than one. Returned
> [B,H] values already contain normalized physical-time weights; summing
> them yields the same objective used for final candidate scoring.

## `NativeReferenceCost.__init__`

[定义位置](../../../manipulation/reference_tracking.py#L164)。

输入签名：

```python
def __init__(self, problem, device_cfg, *, pose_weight):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
problem.tool_link
device_cfg.to_device
problem.tool_transform
problem.base_pose
self.tool_frames
```

写入目标：

```python
self.problem
self.device_cfg
self.tool_frames
self._translation
self._rotation
inverse_base
self._base_translation
self._base_rotation
position_weight
rotation_weight
config
self.native_cost
```

直接调用符号：

```python
device_cfg.to_device
_quaternion
_inverse
ToolPoseCostCfg
ToolPoseCost
```

## `NativeReferenceCost.bind`

[定义位置](../../../manipulation/reference_tracking.py#L188)。

输入签名：

```python
def bind(self, sample_times, valid_count, progress_map, duration):
```

实现说明：

> Sample the supplied skill reference at this grid's physical times.
>
> Seconds are relative to the bound trajectory start. The reference
> coordinate is progress_map(sample_times/duration), including holds
> and a continuation request's cropped progress interval. Sampling
> uses PlanningProblem.sample_reference, preserving composed PosePath
> and mechanism-coordinate behavior. No joint-reference time map is
> applied to this already bound pose-reference clock.
>
> Call again when the request, mapping, duration or ordinary grid
> changes. The resulting device goals and quadrature weights are
> reused unchanged during all optimization calls on that binding.

显式输出分支：

```python
self
```

读取的对象属性：

```python
torch.as_tensor(sample_times, **self.device_cfg.as_torch_dict()).detach
torch.as_tensor
self.device_cfg.as_torch_dict
self.device_cfg
torch.as_tensor(valid_count, device=times.device, dtype=torch.long).reshape
times.device
torch.long
times.shape
torch.no_grad
progress_map.u
progress_map.s
self.problem.sample_reference
self.problem
self._base_translation
self._base_rotation
self.tool_frames
position.reshape(-1, 1, 1, 1, 3).contiguous
position.reshape
quaternion.reshape(-1, 1, 1, 1, 4).contiguous
quaternion.reshape
torch.arange
self.horizon
torch.int32
```

写入目标：

```python
times
counts
self.weights
self.reference_rows
self.horizon
progress
position
quaternion
self.goal
self._time_indices
```

直接调用符号：

```python
torch.as_tensor(sample_times, **self.device_cfg.as_torch_dict()).detach
torch.as_tensor
self.device_cfg.as_torch_dict
torch.as_tensor(valid_count, device=times.device, dtype=torch.long).reshape
physical_time_mean_weights
torch.no_grad
_interpolate
self.problem.sample_reference
quaternion_rotate
quaternion_multiply
GoalToolPose
position.reshape(-1, 1, 1, 1, 3).contiguous
position.reshape
quaternion.reshape(-1, 1, 1, 1, 4).contiguous
quaternion.reshape
torch.arange
```

## `NativeReferenceCost.row_indices`

[定义位置](../../../manipulation/reference_tracking.py#L223)。

输入签名：

```python
def row_indices(self, batch_size, reference_rows=None):
```

实现说明：

> Map each current state row to a bound reference/time-grid row.

显式输出分支：

```python
torch.zeros(batch_size, device=self.weights.device, dtype=torch.int32)
reference_rows.reshape(batch_size).to(device=self.weights.device, dtype=torch.int32)
```

读取的对象属性：

```python
self.reference_rows
torch.zeros
self.weights.device
self.weights
torch.int32
reference_rows.reshape(batch_size).to
reference_rows.reshape
```

写入目标：

无显式条目。

直接调用符号：

```python
ValueError
torch.zeros
reference_rows.reshape(batch_size).to
reference_rows.reshape
```

## `NativeReferenceCost.time_weights`

[定义位置](../../../manipulation/reference_tracking.py#L231)。

输入签名：

```python
def time_weights(self, batch_size, reference_rows=None):
```

实现说明：

> Return the same [B,H] quadrature for retained ordinary costs.

显式输出分支：

```python
self.weights.index_select(0, rows.long())
```

读取的对象属性：

```python
self.row_indices
self.weights.index_select
self.weights
rows.long
```

写入目标：

```python
rows
```

直接调用符号：

```python
self.row_indices
self.weights.index_select
rows.long
```

## `NativeReferenceCost.forward`

[定义位置](../../../manipulation/reference_tracking.py#L236)。

输入签名：

```python
def forward(self, tool_poses, reference_rows=None):
```

实现说明：

> Evaluate the existing FK output without another kinematic query.
>
> The current pose and its cost gradients both use the native FK
> protocol. Ordinary Torch quaternion losses have a different VJP and
> must keep their existing independent derivative connection.

显式输出分支：

```python
value.reshape(batch, horizon, 2).sum(-1) * self.weights.index_select(0, rows.long())
```

读取的对象属性：

```python
tool_poses.position.shape
tool_poses.position
self.horizon
tool_poses.tool_frames.index
tool_poses.tool_frames
self.problem.tool_link
self.problem
NativeRigidToolTransform.apply
tool_poses.quaternion
self._translation
self._rotation
self.row_indices
(rows[:, None] * horizon + self._time_indices[None]).reshape
self._time_indices
self.tool_frames
position.reshape(-1, 1, 1, 3).contiguous
position.reshape
quaternion.reshape(-1, 1, 1, 4).contiguous
quaternion.reshape
self.native_cost.setup_batch_tensors
self.native_cost
self.native_cost.forward
self.goal
target_indices.contiguous
value.reshape(batch, horizon, 2).sum
value.reshape
self.weights.index_select
self.weights
rows.long
```

写入目标：

```python
batch
horizon
tool_index
position
quaternion
rows
target_indices
current
value
```

直接调用符号：

```python
ValueError
tool_poses.tool_frames.index
NativeRigidToolTransform.apply
self.row_indices
(rows[:, None] * horizon + self._time_indices[None]).reshape
ToolPose
position.reshape(-1, 1, 1, 3).contiguous
position.reshape
quaternion.reshape(-1, 1, 1, 4).contiguous
quaternion.reshape
self.native_cost.setup_batch_tensors
self.native_cost.forward
target_indices.contiguous
value.reshape(batch, horizon, 2).sum
value.reshape
self.weights.index_select
rows.long
```
