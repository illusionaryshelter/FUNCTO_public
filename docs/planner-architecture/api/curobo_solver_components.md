# `manipulation/curobo_solver_components.py` 实现契约

代码基线：`2498027`。
职责和数据语义见[组件说明](../04-solver-and-search.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `PhysicalLineSearch`

[定义位置](../../../manipulation/curobo_solver_components.py#L22)。

```python
class PhysicalLineSearch(ArmijoLineSearchStrategy):
```

类字段：

无显式条目。

实现说明：

> One native Armijo search, contracting only unresolved seed directions.
>
> A finite set of trial scales can miss every decreasing step even for a
> valid descent direction. Reuse the native selector at successively smaller
> directions until a represented decreasing step is found or all positive
> trials equal the base in the action dtype. The caller's request guard bounds
> every evaluation; AL parameters and local geometry stay fixed throughout.

## `PhysicalLineSearch.__init__`

[定义位置](../../../manipulation/curobo_solver_components.py#L32)。

输入签名：

```python
def __init__(self, domain):
```

显式输出分支：

无显式条目。

读取的对象属性：

无显式条目。

写入目标：

```python
self.domain
self.evaluations
self.backtracks
```

直接调用符号：

无显式条目。

## `PhysicalLineSearch._prepare_search_points`

[定义位置](../../../manipulation/curobo_solver_components.py#L37)。

输入签名：

```python
def _prepare_search_points(self, x, step_direction, context):
```

显式输出分支：

```python
(points.view(base.shape[0], context.n_linesearch, context.opt_dim).detach().requires_grad_(True), direction.view(base.shape[0], 1, context.opt_dim))
```

读取的对象属性：

```python
x.detach
self.domain.limit_step
self.domain
step_direction.detach
context.line_search_scale
self.jit_get_x_set
points.view(base.shape[0], context.n_linesearch, context.opt_dim).detach().requires_grad_
points.view(base.shape[0], context.n_linesearch, context.opt_dim).detach
points.view
base.shape
context.n_linesearch
context.opt_dim
direction.view
```

写入目标：

```python
base
direction
points
self._evaluated_direction
```

直接调用符号：

```python
x.detach
self.domain.limit_step
step_direction.detach
self.jit_get_x_set
points.view(base.shape[0], context.n_linesearch, context.opt_dim).detach().requires_grad_
points.view(base.shape[0], context.n_linesearch, context.opt_dim).detach
points.view
direction.view
```

## `PhysicalLineSearch._update_best_solution`

[定义位置](../../../manipulation/curobo_solver_components.py#L50)。

输入签名：

```python
def _update_best_solution(self, iteration_state, context):
```

显式输出分支：

```python
iteration_state
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `PhysicalLineSearch._compute_costs_and_gradients`

[定义位置](../../../manipulation/curobo_solver_components.py#L56)。

输入签名：

```python
def _compute_costs_and_gradients(self, x_set, context):
```

显式输出分支：

```python
(torch.where(finite, costs, float('inf')), gradients)
```

读取的对象属性：

```python
super()._compute_costs_and_gradients
torch.isfinite
torch.isfinite(gradients).all
finite[:, 0].all
costs[:, 0, 0].detach
torch.where
```

写入目标：

```python
costs
gradients
finite
self._base_cost
```

直接调用符号：

```python
super()._compute_costs_and_gradients
super
torch.isfinite
torch.isfinite(gradients).all
bool
finite[:, 0].all
ValueError
costs[:, 0, 0].detach
torch.where
float
```

## `PhysicalLineSearch.search`

[定义位置](../../../manipulation/curobo_solver_components.py#L66)。

输入签名：

```python
def search(self, iteration_state, context):
```

显式输出分支：

```python
super()._update_best_solution(selected, context)
```

读取的对象属性：

```python
iteration_state.exploration_action.detach
iteration_state.exploration_action
iteration_state.step_direction.detach().clone
iteration_state.step_direction.detach
iteration_state.step_direction
torch.ones
base.device
torch.bool
torch.zeros
torch.int64
pending.any
super().search
pending.reshape
value.ndim
torch.where
(base + context.line_search_scale.max() * self._evaluated_direction != base).flatten(1).any
(base + context.line_search_scale.max() * self._evaluated_direction != base).flatten
context.line_search_scale.max
context.line_search_scale
self._evaluated_direction
(proposal.exploration_action != base).flatten(1).any
(proposal.exploration_action != base).flatten
proposal.exploration_action
proposal.exploration_cost
self._base_cost
self.backtracks.add_
self.backtracks
pending.to
super()._update_best_solution
```

写入目标：

```python
base
direction
count
pending
self.backtracks
self.evaluations
selected
trial_state
proposal
name
value
mask
represented
moved
```

直接调用符号：

```python
iteration_state.exploration_action.detach
iteration_state.step_direction.detach().clone
iteration_state.step_direction.detach
len
torch.ones
torch.zeros
bool
pending.any
replace
super().search
super
getattr
pending.reshape
setattr
torch.where
(base + context.line_search_scale.max() * self._evaluated_direction != base).flatten(1).any
(base + context.line_search_scale.max() * self._evaluated_direction != base).flatten
context.line_search_scale.max
(proposal.exploration_action != base).flatten(1).any
(proposal.exploration_action != base).flatten
self.backtracks.add_
pending.to
super()._update_best_solution
```

## `PhysicalGradientCore`

[定义位置](../../../manipulation/curobo_solver_components.py#L99)。

```python
class PhysicalGradientCore(GradientOptCore):
```

类字段：

```python
_progress_buffers = ('_iterations', '_effective_updates', '_last_displacement', '_objective_updates', '_line_evaluations', '_line_backtracks')
_candidate_buffers = ('_candidate_action', '_candidate_objective', '_candidate_valid')
_direction_work_fields = ('qp_calls', 'lp_calls', 'extrema_seconds', 'assembly_seconds', 'qp_solve_seconds', 'lp_solve_seconds', 'host_transfer_seconds')
```

实现说明：

> Native iteration lifecycle with one ordinary physical descent interface.

## `PhysicalGradientCore.__init__`

[定义位置](../../../manipulation/curobo_solver_components.py#L102)。

输入签名：

```python
def __init__(self, config, rollout_list, *, domain: JointPositionDomain):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
LineSearchType.ARMIJO
super().__init__
self._physical_step_direction
self.update_num_problems
config.num_problems
```

写入目标：

```python
self.domain
self.request_guard
self._functional_constraints
self._local_geometry_models
self._candidate_tracking
self._last_functional_signed
self._al_blocks_remaining
config.line_search_type
config.solver_name
config.use_cuda_kernel_line_search
```

直接调用符号：

```python
super().__init__
super
self.update_num_problems
```

## `PhysicalGradientCore._check_request`

[定义位置](../../../manipulation/curobo_solver_components.py#L119)。

输入签名：

```python
def _check_request(self, *, wait_for_device=False):
```

实现说明：

> Charge completed device work to the same request as host QPs.

显式输出分支：

无显式条目。

读取的对象属性：

```python
self.request_guard
self.device_cfg.device.type
self.device_cfg.device
self.device_cfg
torch.cuda.current_stream(self.device_cfg.device).synchronize
torch.cuda.current_stream
torch.cuda
self._batch_work
```

写入目标：

```python
started
self._batch_work['device_wait_seconds']
```

直接调用符号：

```python
perf_counter
torch.cuda.current_stream(self.device_cfg.device).synchronize
torch.cuda.current_stream
self.request_guard
```

## `PhysicalGradientCore._run_initial_evaluation`

[定义位置](../../../manipulation/curobo_solver_components.py#L128)。

输入签名：

```python
def _run_initial_evaluation(self, iteration_state):
```

显式输出分支：

```python
state
```

读取的对象属性：

```python
self._check_request
super()._run_initial_evaluation
state.best_cost.copy_
state.best_cost
state.exploration_cost
state.best_action.copy_
state.best_action
state.exploration_action
state.best_iteration.copy_
state.best_iteration
state.current_iteration
self._best.cost.copy_
self._best.cost
self._best
self._best.action.copy_
self._best.action
self._best.iteration.copy_
self._best.iteration
state.step_direction.copy_
state.step_direction
self._physical_step_direction
```

写入目标：

```python
state
```

直接调用符号：

```python
self._check_request
super()._run_initial_evaluation
super
state.best_cost.copy_
state.best_action.copy_
state.best_iteration.copy_
self._best.cost.copy_
self._best.action.copy_
self._best.iteration.copy_
state.step_direction.copy_
self._physical_step_direction
```

## `PhysicalGradientCore._cost_and_gradient_with_guard`

[定义位置](../../../manipulation/curobo_solver_components.py#L146)。

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
self._batch_work
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
started
trajectories
costs
self._batch_work['forward_host_seconds']
self._batch_work['cost_gradient_calls']
cost
self._batch_work['backward_host_seconds']
g_x
feasible
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
perf_counter
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

[定义位置](../../../manipulation/curobo_solver_components.py#L184)。

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

[定义位置](../../../manipulation/curobo_solver_components.py#L188)。

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

[定义位置](../../../manipulation/curobo_solver_components.py#L193)。

输入签名：

```python
def _begin_functional_solve(self, action):
```

实现说明：

> Declare seed layout before the first evaluation and reset in place.
>
> A collision/search recovery can replace or replicate seeds. Each such
> native optimize call therefore starts its own dual sequence; only its
> existing inner blocks share parameters. The current seed layout owns the multiplier and incumbent buffers.

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
self._candidate_valid.zero_
self._candidate_valid
self._local_geometry_models
self._refresh_local_geometry
```

写入目标：

```python
manager
self._functional_constraints
self._local_geometry_models
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
self._candidate_valid.zero_
self._refresh_local_geometry
```

## `PhysicalGradientCore._refresh_local_geometry`

[定义位置](../../../manipulation/curobo_solver_components.py#L219)。

输入签名：

```python
def _refresh_local_geometry(self, action):
```

实现说明：

> Capture original geometry evidence at repeated actual seed bases.
>
> CuRobo SolverCore constructs every rollout of one optimizer from the
> same ``optimizer_rollout_configs[i]``. FUNCTO binds the same cubic
> clock and process manager to those instances. Their geometry query
> layouts therefore coincide, and the ordinary rollout captures the
> model shared by initial and ordinary gradient callbacks. The same
> completed rollout also supplies original hard evidence and a fixed
> candidate score at block boundaries. Its changing local-model value
> never enters candidate ranking or a line search.

显式输出分支：

```python
(torch.as_tensor(changed, dtype=torch.bool, device=action.device), trajectories)
```

读取的对象属性：

```python
self._local_geometry_models
self.config.num_problems
self.config
self.config.num_particles
action.detach().view
action.detach
self.action_horizon
self.action_dim
base[:, None].expand(-1, particles, -1, -1).contiguous().flatten
base[:, None].expand(-1, particles, -1, -1).contiguous
base[:, None].expand
models.begin_capture
torch.no_grad
self._check_request
self.rollout_fn.evaluate_action
self.rollout_fn
self._batch_work
models.finish_capture
models.abort_capture
torch.as_tensor
torch.bool
action.device
```

写入目标：

```python
models
count
particles
base
repeated
started
trajectories
self._batch_work['geometry_capture_host_seconds']
self._batch_work['geometry_capture_calls']
changed
```

直接调用符号：

```python
action.detach().view
action.detach
base[:, None].expand(-1, particles, -1, -1).contiguous().flatten
base[:, None].expand(-1, particles, -1, -1).contiguous
base[:, None].expand
models.begin_capture
torch.no_grad
self._check_request
perf_counter
self.rollout_fn.evaluate_action
models.finish_capture
models.abort_capture
torch.as_tensor
```

## `PhysicalGradientCore._retain_sampled_candidate`

[定义位置](../../../manipulation/curobo_solver_components.py#L257)。

输入签名：

```python
def _retain_sampled_candidate(self, action, feasible, objective):
```

实现说明：

> Retain one sampled-feasible candidate per seed by a fixed objective.
>
> All task and geometry hard evidence uses the common declaration.
> The objective omits changing multipliers and local collision models,
> so candidates from different blocks can be compared. Final curve
> validation still owns acceptance.

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
feasible.view
torch.isfinite(actions).flatten(2).all
torch.isfinite(actions).flatten
torch.isfinite
objective.detach().view
objective.detach
torch.where
ranked.min
actions.gather(1, index[:, None, None, None].expand(-1, 1, self.action_horizon, self.action_dim)).squeeze
actions.gather
index[:, None, None, None].expand
valid.any
self._candidate_valid
self._candidate_objective
self._candidate_action.copy_
self._candidate_action
self._candidate_objective.copy_
self._candidate_valid.logical_or_
torch.no_grad
```

写入目标：

```python
count
particles
actions
valid
values
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
feasible.view
torch.isfinite(actions).flatten(2).all
torch.isfinite(actions).flatten
torch.isfinite
objective.detach().view
objective.detach
torch.where
float
ranked.min
actions.gather(1, index[:, None, None, None].expand(-1, 1, self.action_horizon, self.action_dim)).squeeze
actions.gather
index[:, None, None, None].expand
valid.any
self._candidate_action.copy_
self._candidate_objective.copy_
self._candidate_valid.logical_or_
torch.no_grad
```

## `PhysicalGradientCore._finish_functional_block`

[定义位置](../../../manipulation/curobo_solver_components.py#L281)。

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
self._candidate_tracking
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
self._local_geometry_models
torch.zeros
torch.bool
base.device
self.rollout_fn.evaluate_action
self.rollout_fn
self._batch_work
self._refresh_local_geometry
self.rollout_fn.constraint_manager.optimizer_evidence
self.rollout_fn.constraint_manager
self._retain_sampled_candidate
value.view
raw_signed.items
owner.update
changed.any
self._compute_cost_constraint_and_gradient
self._local_geometry_models.repeated_base_evaluation
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
changed
started
trajectories
self._batch_work['base_evidence_host_seconds']
self._batch_work['base_evidence_calls']
feasible
raw_signed
objective
self._last_functional_signed
signed
name
value
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
torch.zeros
perf_counter
self.rollout_fn.evaluate_action
self._refresh_local_geometry
self.rollout_fn.constraint_manager.optimizer_evidence
self._retain_sampled_candidate
value.view
raw_signed.items
owner.update
bool
changed.any
self._compute_cost_constraint_and_gradient
self._local_geometry_models.repeated_base_evaluation
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

## `PhysicalGradientCore._physical_step_direction`

[定义位置](../../../manipulation/curobo_solver_components.py#L344)。

输入签名：

```python
def _physical_step_direction(self, state, selected=None):
```

实现说明：

> Use the full current gradient at first, ordinary and restarted steps.

显式输出分支：

```python
direction
```

读取的对象属性：

```python
state.exploration_action.detach
state.exploration_action
state.exploration_gradient.detach
state.exploration_gradient
self.action_horizon_step_max.reshape
self.action_horizon_step_max
self.action_horizon
self.action_dim
self._line_search_context.line_search_scale
self._line_search_context
torch.zeros_like
selected.nonzero().flatten().tolist
selected.nonzero().flatten
selected.nonzero
self._check_request
dict.fromkeys
self._direction_work_fields
self._direction_calls
self.domain.descent_step
self.domain
self._direction_reports
type(error).__name__
self._direction_work
direction[index:index + 1].copy_
torch.no_grad
```

写入目标：

```python
base
gradient
limits
scales
direction
indices
index
work
self._direction_calls[index]
value
report
self._direction_reports[index]
name
self._direction_work[index][name]
```

直接调用符号：

```python
state.exploration_action.detach
state.exploration_gradient.detach
self.action_horizon_step_max.reshape
torch.zeros_like
range
len
selected.nonzero().flatten().tolist
selected.nonzero().flatten
selected.nonzero
self._check_request
dict.fromkeys
self.domain.descent_step
type
direction[index:index + 1].copy_
torch.no_grad
```

## `PhysicalGradientCore._restart_changed_objective`

[定义位置](../../../manipulation/curobo_solver_components.py#L375)。

输入签名：

```python
def _restart_changed_objective(self, state, base, cost, gradient, changed):
```

实现说明：

> Re-score changed objective rows and rebuild their physical direction.

显式输出分支：

无显式条目。

读取的对象属性：

```python
state.best_iteration
state.current_iteration
state.converged
self._best.cost
self._best
self._best.action
self._best.iteration
self._best.converged
self._best.previous_step_direction
self._physical_step_direction
state.step_direction
self._objective_updates.add_
self._objective_updates
changed.to
torch.int64
torch.no_grad
```

写入目标：

```python
name
state.converged[changed]
self._best.converged[changed]
self._best.previous_step_direction[changed]
direction
```

直接调用符号：

```python
copy_rows
getattr
self._physical_step_direction
self._objective_updates.add_
changed.to
torch.no_grad
```

## `PhysicalGradientCore._restart_changed_objective.copy_rows`

[定义位置](../../../manipulation/curobo_solver_components.py#L377)。

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

## `PhysicalGradientCore._opt_step`

[定义位置](../../../manipulation/curobo_solver_components.py#L397)。

输入签名：

```python
def _opt_step(self, state):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
self._check_request
state.exploration_action.detach().clone
state.exploration_action.detach
state.exploration_action
state.exploration_cost.detach().clone
state.exploration_cost.detach
state.exploration_cost
super()._opt_step
(result.exploration_action - before).abs().flatten(1).amax
(result.exploration_action - before).abs().flatten
(result.exploration_action - before).abs
result.exploration_action
result.exploration_cost
self._iterations.add_
self._iterations
self._effective_updates.add_
self._effective_updates
improved.to
torch.int64
self._last_displacement.copy_
self._last_displacement
displacement.double
self._line_evaluations.add_
self._line_evaluations
self._line_search_strategy.evaluations
self._line_search_strategy
self._line_backtracks.add_
self._line_backtracks
self._line_search_strategy.backtracks
```

写入目标：

```python
before
cost
result
displacement
improved
```

直接调用符号：

```python
self._check_request
state.exploration_action.detach().clone
state.exploration_action.detach
state.exploration_cost.detach().clone
state.exploration_cost.detach
super()._opt_step
super
(result.exploration_action - before).abs().flatten(1).amax
(result.exploration_action - before).abs().flatten
(result.exploration_action - before).abs
self._iterations.add_
self._effective_updates.add_
improved.to
self._last_displacement.copy_
displacement.double
self._line_evaluations.add_
self._line_backtracks.add_
```

## `PhysicalGradientCore._create_initial_iteration_state`

[定义位置](../../../manipulation/curobo_solver_components.py#L411)。

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
super()._create_initial_iteration_state
action.clone
```

写入目标：

```python
state
state.best_action
```

直接调用符号：

```python
self.domain.validate
super()._create_initial_iteration_state
super
action.clone
```

## `PhysicalGradientCore._reset_progress`

[定义位置](../../../manipulation/curobo_solver_components.py#L426)。

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
dict.fromkeys
self._direction_work_fields
```

写入目标：

```python
name
self._direction_reports
_
self._direction_calls
self._direction_work
self._batch_work
```

直接调用符号：

```python
getattr(self, name).zero_
getattr
range
dict.fromkeys
```

## `PhysicalGradientCore.update_num_problems`

[定义位置](../../../manipulation/curobo_solver_components.py#L439)。

输入签名：

```python
def update_num_problems(self, num_problems):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().update_num_problems
self.domain
self._progress_buffers
torch.float64
torch.int64
torch.zeros
self.device_cfg.device
self.device_cfg
self._reset_progress
torch.empty
self.action_horizon
self.action_dim
self.device_cfg.dtype
torch.full
torch.bool
```

写入目标：

```python
self._line_search_strategy
self._initial_line_search_strategy
name
dtype
self._candidate_action
self._candidate_objective
self._candidate_valid
```

直接调用符号：

```python
super().update_num_problems
super
PhysicalLineSearch
setattr
torch.zeros
self._reset_progress
torch.empty
torch.full
float
```

## `PhysicalGradientCore._prepare_solve`

[定义位置](../../../manipulation/curobo_solver_components.py#L453)。

输入签名：

```python
def _prepare_solve(self, action):
```

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
```

## `PhysicalGradientCore.reinitialize`

[定义位置](../../../manipulation/curobo_solver_components.py#L459)。

输入签名：

```python
def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
```

显式输出分支：

```python
super().reinitialize(action, mask=mask, clear_optimizer_state=clear_optimizer_state, reset_num_iters=reset_num_iters)
```

读取的对象属性：

```python
self._prepare_solve
self._reset_progress
self._begin_functional_solve
super().reinitialize
```

写入目标：

无显式条目。

直接调用符号：

```python
self._prepare_solve
self._reset_progress
self._begin_functional_solve
super().reinitialize
super
```

## `PhysicalGradientCore.optimize`

[定义位置](../../../manipulation/curobo_solver_components.py#L466)。

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
self._prepare_solve
self._reset_progress
self._best.clear
self._best
self._begin_functional_solve
self.outer_iters
super().optimize
self._candidate_tracking
torch.where
self._candidate_valid
self._candidate_action
```

写入目标：

```python
self._al_blocks_remaining
result
```

直接调用符号：

```python
self._prepare_solve
self._reset_progress
self._best.clear
self._begin_functional_solve
super().optimize
super
torch.where
```

## `PhysicalGradientCore._dispatch`

[定义位置](../../../manipulation/curobo_solver_components.py#L483)。

输入签名：

```python
def _dispatch(self, method_name, *args):
```

显式输出分支：

```python
self._finish_functional_block(result) if method_name == '_opt_iters' else result
```

读取的对象属性：

```python
self._check_request
super()._dispatch
self._finish_functional_block
```

写入目标：

```python
result
```

直接调用符号：

```python
self._check_request
super()._dispatch
super
self._finish_functional_block
```

## `PhysicalGradientCore.solve_diagnostics`

[定义位置](../../../manipulation/curobo_solver_components.py#L489)。

输入签名：

```python
def solve_diagnostics(self):
```

实现说明：

> Per-seed direction work and shared batch evaluation work.
>
> ``shared_batch_work`` repeats the same optimizer-wide counters in
> each seed record; it is counted once per optimizer, not summed over
> seeds. Host call durations include their internal synchronous work.
> Explicit stream waiting is measured separately after those calls.

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
self._direction_calls
self._direction_work
self._batch_work
self._direction_reports
self.config.num_problems
self.config
self._candidate_tracking
self._functional_constraints.snapshot
self._functional_constraints
self._candidate_valid.detach().cpu().tolist
self._candidate_valid.detach().cpu
self._candidate_valid.detach
self._candidate_valid
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
index
duals
valid
objective
report
```

直接调用符号：

```python
getattr(self, name).detach().cpu().tolist
getattr(self, name).detach().cpu
getattr(self, name).detach
getattr
dict
range
self._functional_constraints.snapshot
self._candidate_valid.detach().cpu().tolist
self._candidate_valid.detach().cpu
self._candidate_valid.detach
self._candidate_objective.detach().cpu().tolist
self._candidate_objective.detach().cpu
self._candidate_objective.detach
enumerate
report.update
```

## `PhysicalGradientCore.extra_workspace_bytes`

[定义位置](../../../manipulation/curobo_solver_components.py#L519)。

输入签名：

```python
def extra_workspace_bytes(self):
```

显式输出分支：

```python
sum((value.numel() * value.element_size() for value in tensors))
```

读取的对象属性：

```python
self._progress_buffers
self._candidate_buffers
self._functional_constraints
tensors.extend
owner._multiplier_buffer
owner._penalty_buffer
owner._previous_residual
owner._updates
owner._penalty_capped
value.numel
value.element_size
```

写入目标：

```python
tensors
name
owner
value
```

直接调用符号：

```python
getattr
tensors.extend
sum
value.numel
value.element_size
```

## `PhysicalGradientCore.bind_domain`

[定义位置](../../../manipulation/curobo_solver_components.py#L527)。

输入签名：

```python
def bind_domain(self, domain):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
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
self._reset_progress
```

## `LayoutGoalManager`

[定义位置](../../../manipulation/curobo_solver_components.py#L535)。

```python
class LayoutGoalManager(GoalManager):
```

类字段：

无显式条目。

实现说明：

> Allow goalset padding only when all other solve-layout fields match.

## `LayoutGoalManager._get_padded_goalset_for_links`

[定义位置](../../../manipulation/curobo_solver_components.py#L538)。

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

[定义位置](../../../manipulation/curobo_solver_components.py#L546)。

输入签名：

```python
def bind_solver_components(solver, domain):
```

实现说明：

> Replace native stages with the shared core before their first solve.
>
> SolverCore and MultiStageOptimizer reference the same stage list. Updating
> its entries preserves native rollout, goal and seed ownership while dropping
> unused quasi-Newton buffers and callbacks. Existing bindings only replace
> the current request's mathematical domain.

显式输出分支：

无显式条目。

读取的对象属性：

```python
solver.core.optimizers
solver.core
optimizer.bind_domain
optimizer.config
optimizer.get_all_rollout_instances
solver.core.goal_registry_manager
solver.core.device_cfg
```

写入目标：

```python
index
optimizer
solver.core.optimizers[index]
manager
solver.core.goal_registry_manager
```

直接调用符号：

```python
enumerate
isinstance
optimizer.bind_domain
PhysicalGradientCore
optimizer.get_all_rollout_instances
LayoutGoalManager
```
