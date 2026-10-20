# `manipulation/curobo_solver_components.py` 实现契约

代码基线：`__IMPLEMENTATION_COMMIT__`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `PhysicalLineSearch`

[定义位置](../../../manipulation/curobo_solver_components.py#L34)。

```python
class PhysicalLineSearch(ApproxWolfeLineSearchStrategy):
```

类字段：

无显式条目。

实现说明：

> Keep native Wolfe selection, restricting its points before any rollout.

## `PhysicalLineSearch.__init__`

[定义位置](../../../manipulation/curobo_solver_components.py#L37)。

输入签名：

```python
def __init__(self, domain: JointPositionDomain, safe_action: torch.Tensor):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
torch.Tensor
super().__init__
```

写入目标：

```python
self.domain
self.safe_action
```

直接调用符号：

```python
super().__init__
super
```

## `PhysicalLineSearch._prepare_search_points`

[定义位置](../../../manipulation/curobo_solver_components.py#L42)。

输入签名：

```python
def _prepare_search_points(self, x, step_direction, context):
```

显式输出分支：

```python
(points.view(x.shape[0], context.n_linesearch, context.opt_dim).detach().requires_grad_(True), direction.view(x.shape[0], 1, context.opt_dim))
```

读取的对象属性：

```python
super()._prepare_search_points
x.detach
self.domain.validate
self.domain
torch.where
self.safe_action
self.domain.limit_step_tensor
direction.reshape_as
context.line_search_scale
torch.zeros_like
self.jit_get_x_set
points.view(x.shape[0], context.n_linesearch, context.opt_dim).detach().requires_grad_
points.view(x.shape[0], context.n_linesearch, context.opt_dim).detach
points.view
x.shape
context.n_linesearch
context.opt_dim
direction.view
```

写入目标：

```python
_
direction
base
valid
points
```

直接调用符号：

```python
super()._prepare_search_points
super
x.detach
self.domain.validate
torch.where
self.domain.limit_step_tensor
direction.reshape_as
torch.zeros_like
self.jit_get_x_set
points.view(x.shape[0], context.n_linesearch, context.opt_dim).detach().requires_grad_
points.view(x.shape[0], context.n_linesearch, context.opt_dim).detach
points.view
direction.view
```

## `PhysicalGradientCore`

[定义位置](../../../manipulation/curobo_solver_components.py#L72)。

```python
class PhysicalGradientCore(GradientOptCore):
```

类字段：

```python
_candidate_buffers = ('_candidate_action', '_candidate_objective', '_candidate_rank')
_progress_buffers = ('_exhausted', '_stalled_steps', '_effective_updates', '_iterations', '_history_skips', '_history_resets', '_last_displacement', '_last_curvature')
```

实现说明：

> Native gradient lifecycle with a legal initial best state and trials.

## `PhysicalGradientCore.__init__`

[定义位置](../../../manipulation/curobo_solver_components.py#L75)。

输入签名：

```python
def __init__(self, *args, domain: JointPositionDomain, history_buffers=None, **kwargs):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._guarded_step_direction
super().__init__
self.device_cfg.device
self.device_cfg
```

写入目标：

```python
self.domain
self._history_buffers
self._safe_action
self._native_step_direction_fn
kwargs['step_direction_fn']
self._all_exhausted
self.request_guard
self._recovery_reports
self._functional_constraints
self._candidate_tracking
self._al_blocks_remaining
self._last_functional_signed
self._device_checks
self.graph_capture_bytes
self.graph_snapshot_bytes
```

直接调用符号：

```python
super().__init__
super
DeviceChecks
```

## `PhysicalGradientCore.finish_init`

[定义位置](../../../manipulation/curobo_solver_components.py#L99)。

输入签名：

```python
def finish_init(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._deferred_cuda_graph
self._executors
self._guarded_opt_step
self.device_cfg.device
self.device_cfg
self._prepare_initial_iteration_state
```

写入目标：

```python
self._executors['_opt_step']
self._executors['_prepare_initial_iteration_state']
```

直接调用符号：

```python
create_graph_executor
```

## `PhysicalGradientCore._check_request`

[定义位置](../../../manipulation/curobo_solver_components.py#L112)。

输入签名：

```python
def _check_request(self, *, wait_for_device=False):
```

实现说明：

> Check cancellation/budget without inventing another deadline.
> 
> Checks surround bounded calls; an already submitted CUDA kernel or
> backward pass remains non-preemptible. Captured steps are checked by
> their host replay boundary, never by a timer frozen into the graph.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.request_guard
self.device_cfg.device.type
self.device_cfg.device
self.device_cfg
torch.cuda.is_current_stream_capturing
torch.cuda
torch.cuda.current_stream(self.device_cfg.device).synchronize
torch.cuda.current_stream
```

写入目标：

无显式条目。

直接调用符号：

```python
torch.cuda.is_current_stream_capturing
torch.cuda.current_stream(self.device_cfg.device).synchronize
torch.cuda.current_stream
self.request_guard
```

## `PhysicalGradientCore._opt_step`

[定义位置](../../../manipulation/curobo_solver_components.py#L130)。

输入签名：

```python
def _opt_step(self, iteration_state):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
self._check_request
self.use_cuda_graph
self._guarded_opt_step
self._replay_graph
```

写入目标：

```python
result
```

直接调用符号：

```python
self._check_request
self._guarded_opt_step
self._replay_graph
```

## `PhysicalGradientCore._run_initial_evaluation`

[定义位置](../../../manipulation/curobo_solver_components.py#L137)。

输入签名：

```python
def _run_initial_evaluation(self, iteration_state):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
self._check_request
super()._run_initial_evaluation
```

写入目标：

```python
result
```

直接调用符号：

```python
self._check_request
super()._run_initial_evaluation
super
```

## `PhysicalGradientCore._cost_and_gradient_with_guard`

[定义位置](../../../manipulation/curobo_solver_components.py#L145)。

输入签名：

```python
def _cost_and_gradient_with_guard(self, x, rollout):
```

实现说明：

> Keep native cost/backward arithmetic, checking their host boundary.
> 
> cuRobo's two cost callbacks have no before-backward hook. This owned
> implementation retains their detach, reshape, reduction, backward
> vector and return tensors. The only added operation is the same
> request check before forward, before backward and after backward.

显式输出分支：

```python
(cost, g_x)
```

读取的对象属性：

```python
self._check_request
x.detach().requires_grad_
x.detach
x_n.view
self.config.num_problems
self.config
self.config.num_particles
self.action_horizon
self.action_dim
rollout.evaluate_action
trajectories.costs_and_constraints.get_sum_cost_and_constraint
trajectories.costs_and_constraints
costs.shape
costs.view
cost.backward
self._l_vec
x_n.grad.detach
x_n.grad
self._candidate_tracking
rollout.constraint_manager.optimizer_evidence
rollout.constraint_manager
self._retain_sampled_candidate
```

写入目标：

```python
x_n
x_in
trajectories
costs
cost
g_x
hard_eligible
sampled_feasible
signed
objective
self._last_functional_signed
```

直接调用符号：

```python
self._check_request
x.detach().requires_grad_
x.detach
x_n.view
rollout.evaluate_action
trajectories.costs_and_constraints.get_sum_cost_and_constraint
log_and_raise
costs.view
cost.backward
x_n.grad.detach
rollout.constraint_manager.optimizer_evidence
self._retain_sampled_candidate
```

## `PhysicalGradientCore._compute_cost_constraint_and_gradient`

[定义位置](../../../manipulation/curobo_solver_components.py#L177)。

输入签名：

```python
def _compute_cost_constraint_and_gradient(self, x):
```

显式输出分支：

```python
self._cost_and_gradient_with_guard(x, self.rollout_fn)
```

读取的对象属性：

```python
self._cost_and_gradient_with_guard
self.rollout_fn
```

写入目标：

无显式条目。

直接调用符号：

```python
self._cost_and_gradient_with_guard
```

## `PhysicalGradientCore._compute_cost_constraint_and_gradient_initial`

[定义位置](../../../manipulation/curobo_solver_components.py#L180)。

输入签名：

```python
def _compute_cost_constraint_and_gradient_initial(self, x):
```

显式输出分支：

```python
self._cost_and_gradient_with_guard(x, rollout)
```

读取的对象属性：

```python
self._rollout_list
self._cost_and_gradient_with_guard
```

写入目标：

```python
rollout
```

直接调用符号：

```python
len
self._cost_and_gradient_with_guard
```

## `PhysicalGradientCore._begin_functional_solve`

[定义位置](../../../manipulation/curobo_solver_components.py#L188)。

输入签名：

```python
def _begin_functional_solve(self, action):
```

实现说明：

> Declare seed layout before the first evaluation and reset in place.
> 
> A collision/search recovery can replace or replicate seeds. Each such
> native optimize call therefore starts its own dual sequence; only its
> existing inner blocks share parameters. Graph-owned addresses survive
> unchanged layouts, and graph warmup snapshots include the incumbent.

显式输出分支：

```python
None
```

读取的对象属性：

```python
self.rollout_fn
self._functional_constraints
self._candidate_tracking
self.config.fixed_iters
self.config
self._functional_constraints.bind
self.config.num_problems
self._functional_constraints.reset
self._candidate_action.copy_
self._candidate_action
action.detach().view_as
action.detach
self._candidate_objective.fill_
self._candidate_objective
self._candidate_rank.zero_
self._candidate_rank
```

写入目标：

```python
manager
self._functional_constraints
self._candidate_tracking
self._last_functional_signed
```

直接调用符号：

```python
getattr
ValueError
self._functional_constraints.bind
self._functional_constraints.reset
self._candidate_action.copy_
action.detach().view_as
action.detach
self._candidate_objective.fill_
float
self._candidate_rank.zero_
```

## `PhysicalGradientCore._retain_sampled_candidate`

[定义位置](../../../manipulation/curobo_solver_components.py#L211)。

输入签名：

```python
def _retain_sampled_candidate(self, action, hard_eligible, sampled_feasible, objective):
```

实现说明：

> Keep one action per seed, preferring stronger sampled evidence.
> 
> Rank 0 has no candidate; positive ranks record the caller's hard
> eligibility and sampled-feasibility flags. The observed-scene caller
> requires every hard component in both flags. Ordinary/dense final
> metrics still decide acceptance. Within one rank,
> the unchanged parameter-free objective orders all line-search trials,
> including points which Wolfe did not select. One slot bounds storage.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.config.num_problems
self.config
self.config.num_particles
action.detach().view
action.detach
self.action_horizon
self.action_dim
hard_eligible.view
torch.isfinite(actions).flatten(2).all
torch.isfinite(actions).flatten
torch.isfinite
objective.detach().view
objective.detach
torch.where
sampled_feasible.view(count, particles).to
sampled_feasible.view
torch.uint8
ranks.amax
ranked.min
actions.gather(1, index[:, None, None, None].expand(-1, 1, self.action_horizon, self.action_dim)).squeeze
actions.gather
index[:, None, None, None].expand
self._candidate_rank
self._candidate_objective
self._candidate_action.copy_
self._candidate_action
self._candidate_objective.copy_
self._candidate_rank.copy_
torch.no_grad
```

写入目标：

```python
count
particles
actions
valid
values
ranks
rank
ranked
value
index
selected
improved
```

直接调用符号：

```python
action.detach().view
action.detach
hard_eligible.view
torch.isfinite(actions).flatten(2).all
torch.isfinite(actions).flatten
torch.isfinite
objective.detach().view
objective.detach
torch.where
sampled_feasible.view(count, particles).to
sampled_feasible.view
ranks.amax
float
ranked.min
actions.gather(1, index[:, None, None, None].expand(-1, 1, self.action_horizon, self.action_dim)).squeeze
actions.gather
index[:, None, None, None].expand
self._candidate_action.copy_
self._candidate_objective.copy_
self._candidate_rank.copy_
torch.no_grad
```

## `PhysicalGradientCore._finish_functional_block`

[定义位置](../../../manipulation/curobo_solver_components.py#L240)。

输入签名：

```python
def _finish_functional_block(self, state):
```

实现说明：

> Change parameters only between existing blocks, never add a solve.
> 
> Native exploration_action is the next line search's actual base. Its
> residual cannot be inferred from the last callback, which evaluated
> several different trials. Re-evaluate that base in the ordinary layout,
> update each seed independently, then recompute changed-objective values.
> No update follows the final block, which has no remaining iterations.

显式输出分支：

```python
state
state
state
state
```

读取的对象属性：

```python
self._al_blocks_remaining
self._functional_constraints
owner.keys
self.config.num_problems
self.config
self.config.num_particles
state.exploration_action.detach().clone
state.exploration_action.detach
state.exploration_action
base[:, None].expand(-1, particles, -1, -1).contiguous().view
base[:, None].expand(-1, particles, -1, -1).contiguous
base[:, None].expand
self.opt_dim
self._check_request
torch.no_grad
repeated.view
self.action_horizon
self.action_dim
self.rollout_fn.evaluate_action
self.rollout_fn
self.rollout_fn.constraint_manager.optimizer_evidence
self.rollout_fn.constraint_manager
self._retain_sampled_candidate
value.view
raw_signed.items
owner.update
changed.any
self._compute_cost_constraint_and_gradient
costs.detach
gradients.detach().view
gradients.detach
(torch.isfinite(cost) & torch.isfinite(gradient).flatten(1).all(-1))[changed].all
torch.isfinite
torch.isfinite(gradient).flatten(1).all
torch.isfinite(gradient).flatten
self._restart_changed_objective
```

写入目标：

```python
self._al_blocks_remaining
owner
count
particles
base
repeated
action
trajectories
eligible
feasible
raw_signed
objective
self._last_functional_signed
signed
name
value
changed
costs
gradients
cost
gradient
```

直接调用符号：

```python
state.exploration_action.detach().clone
state.exploration_action.detach
base[:, None].expand(-1, particles, -1, -1).contiguous().view
base[:, None].expand(-1, particles, -1, -1).contiguous
base[:, None].expand
self._check_request
torch.no_grad
repeated.view
self.rollout_fn.evaluate_action
self.rollout_fn.constraint_manager.optimizer_evidence
self._retain_sampled_candidate
value.view
raw_signed.items
owner.update
bool
changed.any
self._compute_cost_constraint_and_gradient
costs.detach
gradients.detach().view
gradients.detach
(torch.isfinite(cost) & torch.isfinite(gradient).flatten(1).all(-1))[changed].all
torch.isfinite
torch.isfinite(gradient).flatten(1).all
torch.isfinite(gradient).flatten
ValueError
self._restart_changed_objective
```

## `PhysicalGradientCore._restart_changed_objective`

[定义位置](../../../manipulation/curobo_solver_components.py#L287)。

输入签名：

```python
def _restart_changed_objective(self, state, base, cost, gradient, changed):
```

实现说明：

> Invalidate only changed rows; retain all cumulative work counters.

显式输出分支：

无显式条目。

读取的对象属性：

```python
state.step_direction
self.config.initial_step_scale
self.config
state.best_iteration
state.current_iteration
state.converged
self._best.cost
self._best
self._best.action
self._best.iteration
self._best.converged
self._best.previous_step_direction
self._on_reinitialize
self._on_initial_state
self._history_buffers
self._history_buffers.step_q_buffer
self._stalled_steps
self._exhausted
self._last_displacement
self._last_curvature
self._history_resets.add_
self._history_resets
changed.to
torch.int64
self._exhausted.all
torch.no_grad
```

写入目标：

```python
name
state.converged[changed]
self._best.converged[changed]
self._best.previous_step_direction[changed]
self._stalled_steps[changed]
self._exhausted[changed]
self._last_displacement[changed]
self._last_curvature[changed]
self._all_exhausted
```

直接调用符号：

```python
copy_rows
getattr
self._on_reinitialize
self._on_initial_state
self._history_resets.add_
changed.to
bool
self._exhausted.all
torch.no_grad
```

## `PhysicalGradientCore._restart_changed_objective.copy_rows`

[定义位置](../../../manipulation/curobo_solver_components.py#L289)。

输入签名：

```python
def copy_rows(target, value):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
changed.reshape
target.ndim
target.copy_
torch.where
value.reshape_as
```

写入目标：

```python
mask
```

直接调用符号：

```python
changed.reshape
target.copy_
torch.where
value.reshape_as
```

## `PhysicalGradientCore._reset_progress`

[定义位置](../../../manipulation/curobo_solver_components.py#L328)。

输入签名：

```python
def _reset_progress(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self._progress_buffers
getattr(self, name).zero_
self.config.num_problems
self.config
```

写入目标：

```python
name
self._all_exhausted
self._recovery_reports
_
```

直接调用符号：

```python
getattr(self, name).zero_
getattr
range
```

## `PhysicalGradientCore._guarded_step_direction`

[定义位置](../../../manipulation/curobo_solver_components.py#L335)。

输入签名：

```python
def _guarded_step_direction(self, state):
```

实现说明：

> Admit only actual, finite, positive-curvature exploration samples.
> 
> The native CUDA operation fuses history insertion and two-loop
> recursion. Run it on the existing full batch, then restore skipped
> rows in place; neither graph addresses nor active-row arithmetic
> change. A zero displacement never evicts a useful history pair.

显式输出分支：

```python
history.step_q_buffer.view_as(state.exploration_action)
```

读取的对象属性：

```python
self._history_buffers
state.exploration_action.reshape_as
state.exploration_action
history.x_0
state.exploration_gradient.reshape_as
state.exploration_gradient
history.grad_0
history.x_0.clone
history.grad_0.clone
displacement.double().flatten
displacement.double
gradient_change.double().flatten
gradient_change.double
(s * y).sum
(displacement != 0).flatten(1).any
(displacement != 0).flatten
torch.isfinite(s).all
torch.isfinite
torch.isfinite(y).all
torch.linalg.vector_norm
torch.linalg
self._exhausted
torch.finfo(q.dtype).eps
torch.finfo
q.dtype
getattr(history, name).clone
self._native_step_direction_fn
gradient.double().flatten
gradient.double
direction.double().flatten
direction.double
torch.isfinite(d).all
(g * d).sum
saved.items
torch.where
tensor.copy_
torch.zeros_like
history.x_0.copy_
history.grad_0.copy_
self.config.initial_step_scale
self.config
history.step_q_buffer.copy_
history.step_q_buffer
direction.reshape_as
self._history_skips.add_
self._history_skips
(~self._exhausted & ~valid).to
torch.int64
self._history_resets.add_
self._history_resets
reset.to
self._last_curvature.copy_
self._last_curvature
history.step_q_buffer.view_as
torch.no_grad
```

写入目标：

```python
history
q
gradient
reference_q
reference_gradient
displacement
gradient_change
s
y
curvature
moved
finite
scale
valid
saved
name
direction
g
d
descending
reset
retain
before
tensor
restored
fallback
```

直接调用符号：

```python
state.exploration_action.reshape_as
state.exploration_gradient.reshape_as
history.x_0.clone
history.grad_0.clone
displacement.double().flatten
displacement.double
gradient_change.double().flatten
gradient_change.double
(s * y).sum
(displacement != 0).flatten(1).any
(displacement != 0).flatten
torch.isfinite(s).all
torch.isfinite
torch.isfinite(y).all
torch.linalg.vector_norm
torch.finfo
getattr(history, name).clone
getattr
self._native_step_direction_fn
gradient.double().flatten
gradient.double
direction.double().flatten
direction.double
torch.isfinite(d).all
(g * d).sum
saved.items
torch.where
tensor.copy_
torch.zeros_like
history.x_0.copy_
history.grad_0.copy_
history.step_q_buffer.copy_
direction.reshape_as
self._history_skips.add_
(~self._exhausted & ~valid).to
self._history_resets.add_
reset.to
self._last_curvature.copy_
history.step_q_buffer.view_as
torch.no_grad
```

## `PhysicalGradientCore._guarded_opt_step`

[定义位置](../../../manipulation/curobo_solver_components.py#L380)。

输入签名：

```python
def _guarded_opt_step(self, state):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
field.name
value.clone
torch.Tensor
super()._opt_step
before.items
self._exhausted.reshape
self._exhausted
tensor.ndim
tensor.copy_
torch.where
(result.exploration_action - previous).abs().flatten(1).amax
(result.exploration_action - previous).abs().flatten
(result.exploration_action - previous).abs
result.exploration_action
torch.finfo(previous.dtype).eps
torch.finfo
previous.dtype
previous.abs().flatten(1).amax(-1).clamp_min
previous.abs().flatten(1).amax
previous.abs().flatten
previous.abs
result.exploration_cost
self._last_displacement.copy_
self._last_displacement
displacement.double
self._effective_updates.add_
self._effective_updates
effective.to
torch.int64
self._iterations.add_
self._iterations
(~self._exhausted).to
self._stalled_steps.copy_
self._stalled_steps
torch.zeros_like
```

写入目标：

```python
before
field
value
result
name
previous
tensor
mask
displacement
resolution
effective
```

直接调用符号：

```python
value.clone
fields
isinstance
getattr
super()._opt_step
super
before.items
self._exhausted.reshape
tensor.copy_
torch.where
(result.exploration_action - previous).abs().flatten(1).amax
(result.exploration_action - previous).abs().flatten
(result.exploration_action - previous).abs
torch.finfo
previous.abs().flatten(1).amax(-1).clamp_min
previous.abs().flatten(1).amax
previous.abs().flatten
previous.abs
self._last_displacement.copy_
displacement.double
self._effective_updates.add_
effective.to
self._iterations.add_
(~self._exhausted).to
self._stalled_steps.copy_
torch.zeros_like
```

## `PhysicalGradientCore.solve_diagnostics`

[定义位置](../../../manipulation/curobo_solver_components.py#L404)。

输入签名：

```python
def solve_diagnostics(self):
```

实现说明：

> Read per-seed progress once outside graphs after an optimizer solve.

显式输出分支：

```python
reports
```

读取的对象属性：

```python
getattr(self, name).detach().cpu().tolist
getattr(self, name).detach().cpu
getattr(self, name).detach
self._progress_buffers
self._recovery_reports
self.config.num_problems
self.config
self._candidate_tracking
self._functional_constraints.snapshot
self._functional_constraints
self._candidate_rank.detach().cpu().tolist
self._candidate_rank.detach().cpu
self._candidate_rank.detach
self._candidate_rank
self._candidate_objective.detach().cpu().tolist
self._candidate_objective.detach().cpu
self._candidate_objective.detach
self._candidate_objective
report.update
```

写入目标：

```python
values
name
reports
row
index
duals
ranks
objective
report
```

直接调用符号：

```python
getattr(self, name).detach().cpu().tolist
getattr(self, name).detach().cpu
getattr(self, name).detach
getattr
len
sum
list
range
self._functional_constraints.snapshot
self._candidate_rank.detach().cpu().tolist
self._candidate_rank.detach().cpu
self._candidate_rank.detach
self._candidate_objective.detach().cpu().tolist
self._candidate_objective.detach().cpu
self._candidate_objective.detach
enumerate
report.update
```

## `PhysicalGradientCore._recover_stalled`

[定义位置](../../../manipulation/curobo_solver_components.py#L433)。

输入签名：

```python
def _recover_stalled(self, state):
```

实现说明：

> Try one physically legal Armijo recovery per stalled block/seed.
> 
> QP construction is host work. Its trials use the normal main rollout
> layout, including a fresh zero-scale cost/gradient, and never enter
> approximate-Wolfe's non-descending exploration fallback.

显式输出分支：

```python
state
state
```

读取的对象属性：

```python
self._check_request
self._stalled_steps
self.config.inner_iters
self.config
self._exhausted
stalled.nonzero().flatten().tolist
stalled.nonzero().flatten
stalled.nonzero
self._line_search_context
state.exploration_action.detach().clone
state.exploration_action.detach
state.exploration_action
base[:, None].expand(-1, context.n_linesearch, -1, -1).contiguous
base[:, None].expand
context.n_linesearch
self._compute_cost_constraint_and_gradient
repeated.reshape
self.config.num_problems
self.opt_dim
base_costs.detach
base_gradients.reshape_as(repeated)[:, 0].detach
base_gradients.reshape_as
state.best_action.detach()[:, None].expand_as(repeated).contiguous
state.best_action.detach()[:, None].expand_as
state.best_action.detach
state.best_action
repeated_best.reshape
torch.zeros_like
state.best_cost[index].copy_
state.best_cost
best_costs.detach
state.best_action[index].copy_
state.best_iteration[index].copy_
state.best_iteration
state.current_iteration
state.exploration_cost[index].copy_
state.exploration_cost
state.exploration_gradient[index].copy_
state.exploration_gradient
self.domain.recovery_step
self.domain
self.action_horizon_step_max.reshape
self.action_horizon_step_max
self.action_horizon
self.action_dim
context.line_search_scale
direction[index:index + 1].copy_
self.domain.limit_step
trials.reshape
costs.detach().squeeze
costs.detach
gradients.reshape_as
(base_gradient[index].double()[None] * actual_step.double()).flatten(1).sum
(base_gradient[index].double()[None] * actual_step.double()).flatten
base_gradient[index].double
actual_step.double
torch.isfinite
torch.isfinite(gradients[index]).flatten(1).all
torch.isfinite(gradients[index]).flatten
costs[index].double
base_cost[index].double
context.line_search_c_1
valid.nonzero().flatten
valid.nonzero
available.numel
remaining.append
available[costs[index, available].argmin()].item
costs[index, available].argmin
getattr(state, name)[index].copy_
state.step_direction[index].copy_
state.step_direction
self.config.initial_step_scale
self._effective_updates
reports[index].update
context.line_search_scale.reshape
actual_step[chosen].abs().max
actual_step[chosen].abs
direction[index].zero_
direction.mul_
state.step_direction[index].zero_
self._history_buffers
getattr(history, name)[:, index].zero_
history.x_0[index].copy_
history.x_0
state.exploration_action[index].reshape_as
history.grad_0[index].copy_
history.grad_0
state.exploration_gradient[index].reshape_as
history.step_q_buffer[index].copy_
history.step_q_buffer
state.step_direction[index].reshape_as
self._history_resets
self._recovery_reports[index].append
self._recovery_reports
self._exhausted.all
```

写入目标：

```python
stalled
indices
context
base
repeated
base_costs
base_gradients
base_cost
base_gradient
repeated_best
best_costs
_
direction
reports
index
proposed
report
reports[index]
pending
backtrack
trials
costs
gradients
remaining
reports[index]['line_search_evaluations']
actual_step
slope
finite
valid
available
chosen
q
gradient
cost
name
self._stalled_steps[index]
self._effective_updates[index]
self._exhausted[index]
reports[index]['acceptance_reason']
history
self._history_resets[index]
self._all_exhausted
```

直接调用符号：

```python
self._check_request
stalled.nonzero().flatten().tolist
stalled.nonzero().flatten
stalled.nonzero
state.exploration_action.detach().clone
state.exploration_action.detach
base[:, None].expand(-1, context.n_linesearch, -1, -1).contiguous
base[:, None].expand
self._compute_cost_constraint_and_gradient
repeated.reshape
base_costs.detach
base_gradients.reshape_as(repeated)[:, 0].detach
base_gradients.reshape_as
state.best_action.detach()[:, None].expand_as(repeated).contiguous
state.best_action.detach()[:, None].expand_as
state.best_action.detach
repeated_best.reshape
torch.zeros_like
state.best_cost[index].copy_
best_costs.detach
bool
state.best_action[index].copy_
state.best_iteration[index].copy_
state.exploration_cost[index].copy_
state.exploration_gradient[index].copy_
self.domain.recovery_step
self.action_horizon_step_max.reshape
direction[index:index + 1].copy_
range
self.domain.limit_step
trials.reshape
costs.detach().squeeze
costs.detach
gradients.reshape_as
(base_gradient[index].double()[None] * actual_step.double()).flatten(1).sum
(base_gradient[index].double()[None] * actual_step.double()).flatten
base_gradient[index].double
actual_step.double
torch.isfinite
torch.isfinite(gradients[index]).flatten(1).all
torch.isfinite(gradients[index]).flatten
costs[index].double
base_cost[index].double
valid.nonzero().flatten
valid.nonzero
available.numel
remaining.append
int
available[costs[index, available].argmin()].item
costs[index, available].argmin
getattr(state, name)[index].copy_
getattr
state.step_direction[index].copy_
reports[index].update
float
context.line_search_scale.reshape
actual_step[chosen].abs().max
actual_step[chosen].abs
direction[index].zero_
direction.mul_
state.step_direction[index].zero_
getattr(history, name)[:, index].zero_
history.x_0[index].copy_
state.exploration_action[index].reshape_as
history.grad_0[index].copy_
state.exploration_gradient[index].reshape_as
history.step_q_buffer[index].copy_
state.step_direction[index].reshape_as
self._recovery_reports[index].append
self._exhausted.all
```

## `PhysicalGradientCore._create_initial_iteration_state`

[定义位置](../../../manipulation/curobo_solver_components.py#L550)。

输入签名：

```python
def _create_initial_iteration_state(self, action):
```

显式输出分支：

```python
state
```

读取的对象属性：

```python
self.domain.validate
self.domain
torch.where
self._safe_action
super()._create_initial_iteration_state
action.clone
```

写入目标：

```python
valid
action
state
state.best_action
```

直接调用符号：

```python
self.domain.validate
torch.where
super()._create_initial_iteration_state
super
action.clone
```

## `PhysicalGradientCore.update_num_problems`

[定义位置](../../../manipulation/curobo_solver_components.py#L560)。

输入签名：

```python
def update_num_problems(self, num_problems):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.reset_cuda_graph
torch.empty
self.action_horizon
self.action_dim
self.device_cfg.device
self.device_cfg
self.device_cfg.dtype
self._line_search_context
self._create_line_search_context
self._compute_cost_constraint_and_gradient_initial
self.domain
self._safe_action
super().update_num_problems
self._progress_buffers
torch.bool
torch.float64
torch.int64
torch.zeros
self._reset_progress
torch.zeros_like
torch.full
torch.uint8
self._line_search_strategy
self._initial_line_search_strategy
```

写入目标：

```python
self._safe_action
self._line_search_context
self._initial_line_search_context
self._line_search_strategy
self._initial_line_search_strategy
name
dtype
self._candidate_action
self._candidate_objective
self._candidate_rank
self._line_search_strategy.safe_action
self._initial_line_search_strategy.safe_action
```

直接调用符号：

```python
self.reset_cuda_graph
torch.empty
self._create_line_search_context
PhysicalLineSearch
super().update_num_problems
super
setattr
torch.zeros
self._reset_progress
torch.zeros_like
torch.full
float
```

## `PhysicalGradientCore._prepare_solve`

[定义位置](../../../manipulation/curobo_solver_components.py#L595)。

输入签名：

```python
def _prepare_solve(self, action):
```

实现说明：

> Check external inputs before either initial rollout or captured work.

显式输出分支：

无显式条目。

读取的对象属性：

```python
action.view
self.config.num_problems
self.config
self.action_horizon
self.action_dim
self.domain.validate
self.domain
self._line_search_context
self._initial_line_search_context
self.domain.validate_scales
context.line_search_scale
self._safe_action.copy_
self._safe_action
action.detach
```

写入目标：

```python
action
context
```

直接调用符号：

```python
action.view
self.domain.validate
self.domain.validate_scales
self._safe_action.copy_
action.detach
```

## `PhysicalGradientCore.reinitialize`

[定义位置](../../../manipulation/curobo_solver_components.py#L603)。

输入签名：

```python
def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
self._prepare_solve
self._reset_progress
self._begin_functional_solve
super().reinitialize
```

写入目标：

```python
result
```

直接调用符号：

```python
self._prepare_solve
self._reset_progress
self._begin_functional_solve
super().reinitialize
super
```

## `PhysicalGradientCore.optimize`

[定义位置](../../../manipulation/curobo_solver_components.py#L613)。

输入签名：

```python
def optimize(self, seed_action):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
self._iteration_state
self._on_reinitialize
self._prepare_solve
self._reset_progress
self._best.clear
self._best
self._begin_functional_solve
self.outer_iters
super().optimize
self._candidate_tracking
torch.where
self._candidate_rank
self._candidate_action
```

写入目标：

```python
self._al_blocks_remaining
result
```

直接调用符号：

```python
self._on_reinitialize
self._prepare_solve
self._reset_progress
self._best.clear
self._begin_functional_solve
super().optimize
super
torch.where
```

## `PhysicalGradientCore._capture_state`

[定义位置](../../../manipulation/curobo_solver_components.py#L639)。

输入签名：

```python
def _capture_state(self):
```

实现说明：

> Snapshot the native closure state omitted by GraphExecutor.clone().
> 
> Native fused L-BFGS mutates these exact allocations. Its graph executor
> clones iteration-state inputs, but its three warmups also update shared
> quasi-Newton history. Restoring values in place preserves both their
> logical state and the addresses recorded by CUDA. Rollout scratch is
> deliberately not reset: its outputs are overwritten by each evaluation
> and its newly allocated fixed buffers must remain owned by that rollout.

显式输出分支：

```python
records
```

读取的对象属性：

```python
self._history_buffers
self._best
self._device_checks
records.append
tensor.clone
self._progress_buffers
self._candidate_buffers
self._line_search_strategy
self._initial_line_search_strategy
strategy._output_buffers
saved.numel
saved.element_size
```

写入目标：

```python
records
owner
names
name
tensor
strategy
self.graph_snapshot_bytes
_
saved
```

直接调用符号：

```python
getattr
records.append
tensor.clone
sum
saved.numel
saved.element_size
```

## `PhysicalGradientCore._restore_capture_state`

[定义位置](../../../manipulation/curobo_solver_components.py#L673)。

输入签名：

```python
def _restore_capture_state(records):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
tensor.copy_
```

写入目标：

```python
owner
name
tensor
saved
```

直接调用符号：

```python
getattr
RuntimeError
tensor.copy_
```

## `PhysicalGradientCore._dispatch`

[定义位置](../../../manipulation/curobo_solver_components.py#L679)。

输入签名：

```python
def _dispatch(self, method_name, *args):
```

显式输出分支：

```python
self._finish_functional_block(args[0])
result
result
```

读取的对象属性：

```python
self._check_request
self._all_exhausted
self._finish_functional_block
self.use_cuda_graph
super()._dispatch
self._recover_stalled
self._device_checks.reset
self._device_checks
self._device_checks.scope
self._executors
self._replay_graph
self._device_checks.raise_if_failed
getattr(self._best, target).copy_
self._best
```

写入目标：

```python
result
source
target
```

直接调用符号：

```python
self._check_request
self._finish_functional_block
super()._dispatch
super
self._recover_stalled
self._device_checks.reset
self._device_checks.scope
self._replay_graph
self._device_checks.raise_if_failed
getattr(self._best, target).copy_
getattr
```

## `PhysicalGradientCore._replay_graph`

[定义位置](../../../manipulation/curobo_solver_components.py#L715)。

输入签名：

```python
def _replay_graph(self, method_name, *args):
```

实现说明：

> Replay one native step inside its caller's shared fault boundary.

显式输出分支：

```python
result
```

读取的对象属性：

```python
self._executors
executor.is_initialized
torch.cuda.memory_allocated
torch.cuda
self.device_cfg.device
self.device_cfg
self._capture_state
self._device_checks.reset
self._device_checks
self._device_checks.scope
executor.warmup
executor.reset
self._restore_capture_state
executor._graph_input
stored.copy_
```

写入目标：

```python
executor
before
records
self.graph_capture_bytes
stored
supplied
result
```

直接调用符号：

```python
torch.cuda.memory_allocated
self._capture_state
self._device_checks.reset
self._device_checks.scope
executor.warmup
executor.reset
self._restore_capture_state
max
zip
stored.copy_
executor
```

## `PhysicalGradientCore.reset_cuda_graph`

[定义位置](../../../manipulation/curobo_solver_components.py#L747)。

输入签名：

```python
def reset_cuda_graph(self):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().reset_cuda_graph
self._device_checks.clear_constants
self._device_checks
```

写入目标：

```python
self.graph_capture_bytes
```

直接调用符号：

```python
super().reset_cuda_graph
super
hasattr
self._device_checks.clear_constants
```

## `PhysicalGradientCore.reset_shape`

[定义位置](../../../manipulation/curobo_solver_components.py#L753)。

输入签名：

```python
def reset_shape(self):
```

显式输出分支：

```python
super().reset_shape()
```

读取的对象属性：

```python
self.reset_cuda_graph
super().reset_shape
```

写入目标：

无显式条目。

直接调用符号：

```python
self.reset_cuda_graph
super().reset_shape
super
```

## `PhysicalGradientCore.extra_workspace_bytes`

[定义位置](../../../manipulation/curobo_solver_components.py#L760)。

输入签名：

```python
def extra_workspace_bytes(self):
```

实现说明：

> Known safety/snapshot storage, excluding measured CUDA graph pools.

显式输出分支：

```python
safe_bytes + progress_bytes + self._device_checks.storage_bytes + self.graph_snapshot_bytes
```

读取的对象属性：

```python
self._safe_action
self._safe_action.numel
self._safe_action.element_size
getattr(self, name).numel
getattr(self, name).element_size
self._progress_buffers
self._candidate_buffers
self._functional_constraints
value.numel
value.element_size
owner._multiplier_buffer
owner._penalty_buffer
owner._previous_residual
owner._updates
owner._penalty_capped
self._device_checks.storage_bytes
self._device_checks
self.graph_snapshot_bytes
```

写入目标：

```python
safe_bytes
progress_bytes
name
owner
value
```

直接调用符号：

```python
self._safe_action.numel
self._safe_action.element_size
sum
getattr(self, name).numel
getattr
getattr(self, name).element_size
value.numel
value.element_size
```

## `PhysicalGradientCore.bind_domain`

[定义位置](../../../manipulation/curobo_solver_components.py#L775)。

输入签名：

```python
def bind_domain(self, domain: JointPositionDomain):
```

实现说明：

> Rebind the measured entrance and clear history for the next request.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.reset_cuda_graph
self._reset_progress
self._line_search_strategy
self._initial_line_search_strategy
```

写入目标：

```python
self.domain
self._iteration_state
self._line_search_strategy.domain
self._initial_line_search_strategy.domain
```

直接调用符号：

```python
self.reset_cuda_graph
self._reset_progress
```

## `LayoutGoalManager`

[定义位置](../../../manipulation/curobo_solver_components.py#L787)。

```python
class LayoutGoalManager(GoalManager):
```

类字段：

无显式条目。

实现说明：

> Allow goalset padding only when all other solve-layout fields match.

## `LayoutGoalManager._get_padded_goalset_for_links`

[定义位置](../../../manipulation/curobo_solver_components.py#L790)。

输入签名：

```python
def _get_padded_goalset_for_links(self, solve_state, current_solve_state, current_goal_buffer, links_goal_pose):
```

显式输出分支：

```python
None
super()._get_padded_goalset_for_links(solve_state, current_solve_state, current_goal_buffer, links_goal_pose)
```

读取的对象属性：

```python
current_solve_state.num_goalset
super()._get_padded_goalset_for_links
```

写入目标：

无显式条目。

直接调用符号：

```python
replace
super()._get_padded_goalset_for_links
super
```

## `bind_solver_components`

[定义位置](../../../manipulation/curobo_solver_components.py#L802)。

输入签名：

```python
def bind_solver_components(solver, domain: JointPositionDomain, *, use_optimizer_graph=False) -> None:
```

实现说明：

> Compose only a FUNCTO-owned solver, before its first solve.
> 
> Reuse the existing native rollouts and L-BFGS history owner. Assign the new
> core before resizing so native L-BFGS callbacks access that core's dimensions.
> Repeated binding updates the physical domain without discarding goal state.

显式输出分支：

无显式条目。

读取的对象属性：

```python
solver.core.config.use_cuda_graph
solver.core.config
solver.core
solver.core.optimizers
optimizer.config.line_search_type
optimizer.config
LineSearchType.APPROX_WOLFE
optimizer.device_cfg.device.type
optimizer.device_cfg.device
optimizer.device_cfg
optimizer.config.use_cuda_kernel_step_direction
optimizer.config.use_cuda_kernel_line_search
optimizer.config.fixed_iters
optimizer.config.store_debug
curobo_runtime.debug_nan
curobo_runtime.cuda_graphs
optimizer._core
optimizer.use_cuda_graph
optimizer._core.bind_domain
optimizer._core._iteration_state
optimizer.get_all_rollout_instances
optimizer._get_step_direction_impl
optimizer._on_reinitialize
optimizer._on_initial_state
optimizer._on_resize
optimizer._on_shift
optimizer._qn
core.update_num_problems
optimizer.config.num_problems
core.finish_init
solver.core.goal_registry_manager
manager._solve_state
solver.core.device_cfg
```

写入目标：

```python
optimizer
core
optimizer._core
manager
solver.core.goal_registry_manager
```

直接调用符号：

```python
ValueError
type
isinstance
optimizer._core.bind_domain
PhysicalGradientCore
optimizer.get_all_rollout_instances
core.update_num_problems
core.finish_init
LayoutGoalManager
```
