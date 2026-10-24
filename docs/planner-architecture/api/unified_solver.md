# `manipulation/unified_solver.py` 实现契约

代码基线：本次实现提交同步更新正文和源码；前置代码检查点为 `759d89b`。实际实现提交号由紧随的文档引用提交回填。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `time_mean_weights`

[定义位置](../../../manipulation/unified_solver.py#L23)。

输入签名：

```python
def time_mean_weights(times):
```

实现说明：

> Trapezoid weights normalized by this grid's physical duration.

显式输出分支：

```python
(np.r_[delta, 0.0] + np.r_[0.0, delta]) / (2 * (times[-1] - times[0]))
```

读取的对象属性：

```python
np.asarray
np.diff
np.any
np.r_
```

写入目标：

```python
times
delta
```

直接调用符号：

```python
np.asarray
np.diff
len
np.any
ValueError
```

## `motion_cost`

[定义位置](../../../manipulation/unified_solver.py#L32)。

输入签名：

```python
def motion_cost(states, bounds, configuration, native_dt, weights):
```

实现说明：

> Evaluate the declared STATE cost in physical units on a fixed clock.
>
> The maintained cuRobo configuration supplies the four joint-bound and
> velocity/acceleration/jerk regularization weights. Its STATE kernel uses
> .5*w*x². The native retiming factors are bound to the original ordinary
> interval exactly once, regardless of this evaluation grid. This planner
> has no inverse-dynamics state, so the native zero-effort terms are zero.

显式输出分支：

```python
(per_time * weights).sum()
```

读取的对象属性：

```python
q.new_tensor
torch.stack
torch.relu
(distance.square() * bound_weights[:, None, None, None]).sum
distance.square
value.square().sum
value.square
(per_time * weights).sum
```

写入目标：

```python
q
bound_weights
regularization
activation
margin
lower
upper
values
distance
per_time
weight
value
```

直接调用符号：

```python
q.new_tensor
torch.stack
torch.relu
(distance.square() * bound_weights[:, None, None, None]).sum
distance.square
sum
value.square().sum
value.square
zip
(per_time * weights).sum
```

## `robot_tool_states`

[定义位置](../../../manipulation/unified_solver.py#L60)。

输入签名：

```python
def robot_tool_states(*, problem, scene, kinematics, q, dq, joint_names, progress, times):
```

实现说明：

> Complete nominal model joints and form the reference-frame tool state.
>
> ``q/dq`` describe one shared physical-time sample layout and named joint
> columns. PinKinematics already includes the base calibration; only the
> fixed link-to-tool transform is composed here. Missing finger columns use
> the scene's declared nominal trajectory and retain its uncertainty model.

显式输出分支：

```python
(q, dq, names, poses, tool, twist)
```

读取的对象属性：

```python
scene.gripper
hand.joint_names
hand.state
hand_state.coordinates
torch.zeros_like
hand.nominal_finger_trajectory
model.valid_mask_torch
model.sample_torch
torch.where
hand.joint_names.index
torch.cat
kinematics.states
problem.tool_link
q.new_tensor
problem.tool_transform
tool_position.expand_as
torch.linalg.cross
torch.linalg
```

写入目标：

```python
names
hand
missing
name
hand_state
coordinates
rate
model
modeled
_
model_rate
columns
q
dq
poses
velocities
position
quaternion
linear
angular
tool_position
tool_quaternion
offset
tool
twist
```

直接调用符号：

```python
tuple
hand.state
torch.zeros_like
model.valid_mask_torch
model.sample_torch
torch.where
hand.joint_names.index
torch.cat
kinematics.states
q.new_tensor
_quaternion
quaternion_rotate
tool_position.expand_as
quaternion_multiply
torch.linalg.cross
```

## `TrajectoryComputation`

[定义位置](../../../manipulation/unified_solver.py#L97)。

```python
class TrajectoryComputation():
```

类字段：

无显式条目。

实现说明：

> Bind the current request to one differentiable CPU trajectory function.
>
> ``parameters`` is normalized physical time. It includes every declared
> task/scene anchor and ordinary integration samples. ``progress`` retains
> their original skill coordinates. ``ordinary_indices`` selects the
> integration grid; adding an event anchor changes no motion-quality weight.
> All arrays use [one candidate, time, named joint]. Finger states are the
> scene's declared nominal model, including its explicit uncertainty scope.

## `TrajectoryComputation.__init__`

[定义位置](../../../manipulation/unified_solver.py#L108)。

输入签名：

```python
def __init__(self, *, problem, scene, kinematics, domain, joint_names, parameters, progress, ordinary_indices, motion_configuration, native_dt, constraint_weight, max_host_rss_bytes):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
torch.as_tensor
domain.intervals
domain._curve_map.knot_dt
domain._curve_map
self.parameters
self.duration
problem.verification_issues
self.progress
self.times.numpy
self.times
self.ordinary_indices
domain._curve_map.query
self.rows
(position_rows == 0.0).all
(position_offsets == torch.as_tensor(domain._curve_map.start_q)).all
domain._curve_map.start_q
problem.pose_events
problem.start_progress
requirement.progress
problem.end_progress
requirement.component_tolerances.get
requirement.component_tolerances
self.equality_keys.append
self.equality_keys
fixed_rows.append
(occurrence & fixed_pose).any
np.flatnonzero
domain._constraint_bounds
```

写入目标：

```python
self.problem
self.scene
self.kinematics
self.domain
self.joint_names
self.parameters
self.progress
self.duration
self.times
issues
self.ordinary_indices
self.time_weights
self.motion_configuration
self.native_dt
self.constraint_weight
self.max_host_rss_bytes
self.rows
array
order
position_rows
position_offsets
fixed_pose
self.equality_keys
fixed_rows
index
requirement
occurrence
self.fixed_equality_rows
self.variable_equality_rows
self.bounds
self.geometry_report
```

直接调用符号：

```python
tuple
np.asarray
torch.as_tensor
problem.verification_issues
ValueError
time_mean_weights
self.times.numpy
float
domain._curve_map.query
range
(position_rows == 0.0).all
(position_offsets == torch.as_tensor(domain._curve_map.start_q)).all
enumerate
requirement.component_tolerances.get
self.equality_keys.append
_first_event_query
fixed_rows.append
bool
(occurrence & fixed_pose).any
np.flatnonzero
```

## `TrajectoryComputation.states`

[定义位置](../../../manipulation/unified_solver.py#L147)。

输入签名：

```python
def states(self, controls):
```

实现说明：

> Use the same affine curve for q and all three physical derivatives.

显式输出分支：

```python
tuple(((row @ controls + offset)[None] for (row, offset) in self.rows))
```

读取的对象属性：

```python
self.rows
```

写入目标：

```python
row
offset
```

直接调用符号：

```python
tuple
```

## `TrajectoryComputation.robot_states`

[定义位置](../../../manipulation/unified_solver.py#L151)。

输入签名：

```python
def robot_states(self, q, dq):
```

实现说明：

> Complete model joints and compose the calibrated reference-frame TCP.

显式输出分支：

```python
robot_tool_states(problem=self.problem, scene=self.scene, kinematics=self.kinematics, q=q, dq=dq, joint_names=self.joint_names, progress=self.progress, times=self.times)
```

读取的对象属性：

```python
self.problem
self.scene
self.kinematics
self.joint_names
self.progress
self.times
```

写入目标：

无显式条目。

直接调用符号：

```python
robot_tool_states
```

## `TrajectoryComputation.task`

[定义位置](../../../manipulation/unified_solver.py#L156)。

输入签名：

```python
def task(self, controls, *, solver_rows_out=None):
```

实现说明：

> Return complete signed rows and optionally their declaration selection.
>
> ``solver_rows_out`` receives the producing condition's Boolean masks,
> before the curve-specific fixed entrance equalities are removed. Both
> values and masks come from this single robot/task evaluation.

显式输出分支：

```python
tuple((torch.cat([value.reshape(-1) for value in group.values()]) if group else empty for group in groups))
```

读取的对象属性：

```python
self.states
self.robot_states
self.problem.constraint_components
self.problem
self.progress
self.times
controls.reshape
torch.cat
value.reshape
group.values
```

写入目标：

```python
q
dq
_
names
tool
twist
groups
empty
value
group
```

直接调用符号：

```python
self.states
self.robot_states
self.problem.constraint_components
controls.reshape
tuple
torch.cat
value.reshape
group.values
```

## `TrajectoryComputation.objective_components`

[定义位置](../../../manipulation/unified_solver.py#L172)。

输入签名：

```python
def objective_components(self, controls, *, include_geometry=True):
```

实现说明：

> Motion quality and declared task preferences share the same candidate.

显式输出分支：

```python
costs
```

读取的对象属性：

```python
self.states
self.robot_states
self.ordinary_indices
self.bounds
self.motion_configuration
self.native_dt
self.time_weights
self.problem.optimizer_preferences
self.problem
self.progress
self.times
self.constraint_weight
preferences.sum
self.scene.optimization_objective
self.scene
self.geometry_report
```

写入目标：

```python
states
q
dq
names
poses
tool
twist
ordinary
value
costs
preferences
costs['task_quality']
costs['observed_geometry']
```

直接调用符号：

```python
self.states
self.robot_states
tuple
motion_cost
self.problem.optimizer_preferences
preferences.sum
self.scene.optimization_objective
```

## `ConstrainedSolution`

[定义位置](../../../manipulation/unified_solver.py#L191)。

```python
class ConstrainedSolution():
```

类字段：

```python
controls: np.ndarray
report: dict
```

实现说明：

> Library termination and proposed controls; acceptance is a separate query.

## `task_jacobian`

[定义位置](../../../manipulation/unified_solver.py#L198)。

输入签名：

```python
def task_jacobian(values, controls, *, check_request, admit_work):
```

实现说明：

> Return exact first derivatives using fewer scalar reverse queries.
>
> When output rows are fewer than controls, each scalar task row supplies
> one ordinary reverse gradient. Otherwise control-column queries reuse
> the reverse graph described below. Both paths use the existing analytic
> Pinocchio backward; neither differentiates it twice with respect to q.
>
> For the original task vector c(x) and an independent covector v, the
> first reverse pass forms p(x,v)=J_c(x).T @ v. Differentiating each p_i
> with respect to v gives column i of J_c. Only v is differentiated in
> that second pass: this requires no second derivative of robot FK with
> respect to x. Both passes use maintained scalar ``autograd.grad``.
>
> Explicit zero dependence gives constant rows their exact zero Jacobian.
> No output identity matrix or batched robot Jacobian is allocated.
> ``admit_work`` precedes the final dense allocation. For the column path
> it follows the first reverse graph, whose storage then belongs to RSS.

显式输出分支：

```python
jacobian
jacobian
```

读取的对象属性：

```python
controls.sum
values.numel
controls.numel
np.empty
torch.autograd.grad
torch.autograd
row.detach().numpy().reshape
row.detach().numpy
row.detach
torch.zeros_like
product.reshape
column.detach().numpy
column.detach
```

写入目标：

```python
connected
jacobian
index
value
row
jacobian[index]
covector
product
column
jacobian[:, index]
```

直接调用符号：

```python
controls.sum
values.numel
controls.numel
admit_work
np.empty
enumerate
check_request
torch.autograd.grad
row.detach().numpy().reshape
row.detach().numpy
row.detach
torch.zeros_like
product.reshape
column.detach().numpy
column.detach
```

## `solver_storage_bytes`

[定义位置](../../../manipulation/unified_solver.py#L241)。

输入签名：

```python
def solver_storage_bytes(*, variables, task_equalities, task_inequalities, physical_inequalities):
```

实现说明：

> Count simultaneously live float64 arrays for this SLSQP interface.
>
> The workspace expression is the one in installed scipy.optimize._slsqp_py.
> Its constraint-normal assembly retains the previous matrix while forming
> equality/inequality stacks, their combined stack and the augmented matrix.
> Cached Jacobians are separate. Scalar row/column derivatives avoid the
> m² identity and batched robot tensors. The column path's first reverse
> graph is included in measured RSS at the second allocation boundary.
> This count covers these owned dense buffers; current RSS accounts for the
> already constructed curve, model, observations and forward task tensors.

显式输出分支：

```python
{'slsqp_workspaces': 8 * (workspace + mineq) + 4 * mineq, 'cached_constraint_values_and_jacobians': 8 * m * (n + 1), 'constraint_normal_assembly': 8 * (2 * m * n + 2 * la * (n + 1) + la), 'constraint_value_assembly': 8 * 3 * m, 'solver_and_derivative_vectors': 8 * (14 * n + 1 + task), 'task_derivative_vectors': 0 if not task else 8 * (task + n if task < n else 3 * task + n)}
```

读取的对象属性：

无显式条目。

写入目标：

```python
n
meq
task
m
n1
la
mineq
workspace
```

直接调用符号：

```python
int
max
```

## `solve_trajectory`

[定义位置](../../../manipulation/unified_solver.py#L280)。

输入签名：

```python
def solve_trajectory(computation, initial, *, iterations, check_request, pending_geometry_bytes, report_out):
```

实现说明：

> Submit one full problem to SciPy SLSQP, with exact first derivatives.
>
> SLSQP may evaluate infeasible trials and starts; the original constraints
> decide feasibility. Its stopping tolerance is a numerical search setting,
> not a task or geometric allowance. Candidate validation retains every
> original physical bound and task tolerance regardless of library status.
>
> ``pending_geometry_bytes`` is the scene's current unallocated material
> storage plus one live geometry/FK evaluation. Existing field/model arrays
> and previously constructed material samples already belong to measured
> RSS. The caller recomputes this quantity before each serial seed solve.
>
> ``report_out`` is caller-owned and remains available when a request check
> raises. Its bounded snapshots describe actual evaluated controls and the
> last SLSQP callback iterate; none is a candidate-acceptance certificate.

显式输出分支：

```python
ConstrainedSolution(initial.copy(), report)
ConstrainedSolution(result.x.reshape(shape).copy(), report)
```

读取的对象属性：

```python
np.asarray
initial.shape
report.update
computation.parameters
torch.tensor
torch.float64
computation.task
value.numel
torch.cat
value.reshape
group.values
torch.zeros
torch.bool
computation.fixed_equality_rows
value.sum
initial_task[0][computation.fixed_equality_rows].detach().numpy
initial_task[0][computation.fixed_equality_rows].detach
computation.equality_keys
value.detach
group.items
layout.numel
torch.nonzero(~selected[offset:offset + size]).flatten
torch.nonzero
values[offset:offset + size][indices].detach().numpy
values[offset:offset + size][indices].detach
np.isfinite
fixed_constraints.append
indices.tolist
constant.tolist
finite.tolist
feasible.all
initial.copy
initial.size
computation.domain.intervals
computation.domain
computation.kinematics.frames
computation.kinematics
computation.kinematics.joint_names
memory.values
computation.max_host_rss_bytes
initial.ravel().copy
initial.ravel
np.abs(initial_gradient).max
np.abs
constraints.append
result.x
result.success
result.status
result.message
result.nit
result.fun
np.max
np.array_equal
matched.get
result.x.reshape(shape).copy
result.x.reshape
type(error).__name__
```

写入目标：

```python
initial
shape
cache
cache_x
counts
evaluation_seconds
started
report
trial_index
initial_controls
counts['task_evaluations']
declared_rows
component_started
initial_task
evaluation_seconds['task']
report['completed_evaluations']['task']
full_eq_count
full_inequality_count
value
solver_rows
group
solver_rows[0][computation.fixed_equality_rows]
eq_count
inequality_count
fixed_values
fixed_report
index
fixed_constraints
fixed_feasible
kind
values
selected
offset
key
layout
size
indices
constant
finite
feasible
report['fixed_task_constraints']
task_count
jacobian_method
physical_count
memory
samples
frames
joints
memory['physical_sampling_rows']
memory['pinocchio_reverse_buffers']
memory['pending_geometry_bytes']
x0
initial_value
initial_gradient
objective_scale
constraints
report['solver_called']
result
matched
```

直接调用符号：

```python
np.asarray
perf_counter
report.update
len
check_request
torch.tensor
computation.task
tuple
value.numel
torch.cat
value.reshape
group.values
torch.zeros
int
value.sum
initial_task[0][computation.fixed_equality_rows].detach().numpy
initial_task[0][computation.fixed_equality_rows].detach
float
zip
value.detach
group.items
layout.numel
torch.nonzero(~selected[offset:offset + size]).flatten
torch.nonzero
values[offset:offset + size][indices].detach().numpy
values[offset:offset + size][indices].detach
np.isfinite
fixed_constraints.append
indices.tolist
constant.tolist
finite.tolist
bool
feasible.all
ConstrainedSolution
initial.copy
solver_storage_bytes
require_host_memory
sum
memory.values
initial.ravel().copy
initial.ravel
evaluate
max
abs
np.abs(initial_gradient).max
np.abs
constraints.append
minimize
snapshot_for
str
np.max
np.array_equal
deepcopy
matched.get
result.x.reshape(shape).copy
result.x.reshape
type
```

## `solve_trajectory.snapshot_for`

[定义位置](../../../manipulation/unified_solver.py#L313)。

输入签名：

```python
def snapshot_for(x):
```

显式输出分支：

```python
snapshot
None
```

读取的对象属性：

```python
np.array_equal
np.asarray(snapshot['controls']).ravel
np.asarray
```

写入目标：

```python
name
snapshot
```

直接调用符号：

```python
np.array_equal
np.asarray(snapshot['controls']).ravel
np.asarray
```

## `solve_trajectory.callback`

[定义位置](../../../manipulation/unified_solver.py#L320)。

输入签名：

```python
def callback(x):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
x.reshape(shape).tolist
x.reshape
```

写入目标：

```python
report['callback_iterations']
matched
report['last_callback_iterate']
```

直接调用符号：

```python
snapshot_for
x.reshape(shape).tolist
x.reshape
list
check_request
```

## `solve_trajectory.admit_task_derivatives`

[定义位置](../../../manipulation/unified_solver.py#L409)。

输入签名：

```python
def admit_task_derivatives():
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
initial.size
computation.max_host_rss_bytes
```

写入目标：

```python
rows
returned
increment
```

直接调用符号：

```python
require_host_memory
```

## `solve_trajectory.evaluate`

[定义位置](../../../manipulation/unified_solver.py#L422)。

输入签名：

```python
def evaluate(kind, x):
```

显式输出分支：

```python
cache[kind]
output
```

读取的对象属性：

```python
np.array_equal
cache.clear
x.copy
x.reshape(shape).tolist
x.reshape
computation.domain.signed_constraints
computation.domain
torch.tensor
torch.float64
computation.objective_components
components.values
torch.autograd.grad
torch.autograd
value.detach
gradient.detach().numpy().ravel
gradient.detach().numpy
gradient.detach
computation.task
torch.cat
values.detach().numpy
values.detach
snapshot['completed_components'].append
snapshot.update
computation.geometry_report
required.issubset
```

写入目标：

```python
cache_x
trial_index
report['latest_trial']
counts[kind + '_evaluations']
component_started
output
controls
components
value
gradient
complete
values
selected
jacobian
evaluation_seconds[kind]
cache[kind]
report['completed_evaluations'][kind]
snapshot
required
report['latest_complete_trial']
```

直接调用符号：

```python
check_request
np.array_equal
cache.clear
x.copy
x.reshape(shape).tolist
x.reshape
perf_counter
computation.domain.signed_constraints
torch.tensor
computation.objective_components
sum
components.values
torch.autograd.grad
float
value.detach
gradient.detach().numpy().ravel
gradient.detach().numpy
gradient.detach
computation.task
torch.cat
tuple
zip
task_jacobian
values.detach().numpy
values.detach
snapshot['completed_components'].append
snapshot.update
deepcopy
set
required.issubset
```

## `solve_trajectory.scaled_objective`

[定义位置](../../../manipulation/unified_solver.py#L484)。

输入签名：

```python
def scaled_objective(x):
```

显式输出分支：

```python
(value / objective_scale, gradient / objective_scale)
```

读取的对象属性：

无显式条目。

写入目标：

```python
value
gradient
```

直接调用符号：

```python
evaluate
```
