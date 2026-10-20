# `manipulation/curobo_solver_components.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../04-solver-and-search.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `PhysicalLineSearch`

[定义位置](../../../manipulation/curobo_solver_components.py#L34)。

```python
class PhysicalLineSearch(ApproxWolfeLineSearchStrategy):
    ...
    ...
```

该类型的直接方法：`__init__`、`_prepare_search_points`。

## `PhysicalLineSearch.__init__`

[定义位置](../../../manipulation/curobo_solver_components.py#L37)。

输入签名：

```python
def PhysicalLineSearch.__init__(self, domain: JointPositionDomain, safe_action: torch.Tensor):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.domain`、`self.safe_action`。

直接调用：

```python
super().__init__
super
```

## `PhysicalLineSearch._prepare_search_points`

[定义位置](../../../manipulation/curobo_solver_components.py#L42)。

输入签名：

```python
def PhysicalLineSearch._prepare_search_points(self, x, step_direction, context):
```

直接读取的对象属性：`self.domain`、`self.safe_action`、`self.jit_get_x_set`。

输出分支的源码表达式：

```python
(points.view(x.shape[0], context.n_linesearch, context.opt_dim).detach().requires_grad_(True), direction.view(x.shape[0], 1, context.opt_dim))
```

直接调用：

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
    ...
    ...
```

该类型的直接方法：`__init__`、`finish_init`、`_check_request`、`_opt_step`、`_run_initial_evaluation`、`_cost_and_gradient_with_guard`、`_compute_cost_constraint_and_gradient`、`_compute_cost_constraint_and_gradient_initial`、`_begin_functional_solve`、`_retain_sampled_candidate`、`_finish_functional_block`、`_restart_changed_objective`、`_reset_progress`、`_guarded_step_direction`、`_guarded_opt_step`、`solve_diagnostics`、`_recover_stalled`、`_create_initial_iteration_state`、`update_num_problems`、`_prepare_solve`、`reinitialize`、`optimize`、`_capture_state`、`_restore_capture_state`、`_dispatch`、`_replay_graph`、`reset_cuda_graph`、`reset_shape`、`extra_workspace_bytes`、`bind_domain`。

## `PhysicalGradientCore.__init__`

[定义位置](../../../manipulation/curobo_solver_components.py#L75)。

输入签名：

```python
def PhysicalGradientCore.__init__(self, *args, domain: JointPositionDomain, history_buffers=None, **kwargs):
```

直接读取的对象属性：`self._guarded_step_direction`、`self.device_cfg`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.domain`、`self._history_buffers`、`self._safe_action`、`self._native_step_direction_fn`、`kwargs['step_direction_fn']`、`self._all_exhausted`、`self.request_guard`、`self._recovery_reports`、`self._functional_constraints`、`self._candidate_tracking`、`self._al_blocks_remaining`、`self._last_functional_signed`、`self._device_checks`、`self.graph_capture_bytes`、`self.graph_snapshot_bytes`。

直接调用：

```python
super().__init__
super
DeviceChecks
```

## `PhysicalGradientCore.finish_init`

[定义位置](../../../manipulation/curobo_solver_components.py#L99)。

输入签名：

```python
def PhysicalGradientCore.finish_init(self):
```

直接读取的对象属性：`self._deferred_cuda_graph`、`self._executors`、`self._guarded_opt_step`、`self.device_cfg`、`self._prepare_initial_iteration_state`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._executors['_opt_step']`、`self._executors['_prepare_initial_iteration_state']`。

直接调用：

```python
create_graph_executor
```

## `PhysicalGradientCore._check_request`

[定义位置](../../../manipulation/curobo_solver_components.py#L112)。

输入签名：

```python
def PhysicalGradientCore._check_request(self, *, wait_for_device=False):
```

直接读取的对象属性：`self.request_guard`、`self.device_cfg`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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
def PhysicalGradientCore._opt_step(self, iteration_state):
```

直接读取的对象属性：`self._check_request`、`self.use_cuda_graph`、`self._guarded_opt_step`、`self._replay_graph`。

输出分支的源码表达式：

```python
result
```

直接调用：

```python
self._check_request
self._guarded_opt_step
self._replay_graph
```

## `PhysicalGradientCore._run_initial_evaluation`

[定义位置](../../../manipulation/curobo_solver_components.py#L137)。

输入签名：

```python
def PhysicalGradientCore._run_initial_evaluation(self, iteration_state):
```

直接读取的对象属性：`self._check_request`。

输出分支的源码表达式：

```python
result
```

直接调用：

```python
self._check_request
super()._run_initial_evaluation
super
```

## `PhysicalGradientCore._cost_and_gradient_with_guard`

[定义位置](../../../manipulation/curobo_solver_components.py#L145)。

输入签名：

```python
def PhysicalGradientCore._cost_and_gradient_with_guard(self, x, rollout):
```

直接读取的对象属性：`self._check_request`、`self.config`、`self.action_horizon`、`self.action_dim`、`self._l_vec`、`self._candidate_tracking`、`self._retain_sampled_candidate`。

输出分支的源码表达式：

```python
(cost, g_x)
```

直接写入的状态或数组位置：`self._last_functional_signed`。

直接调用：

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
def PhysicalGradientCore._compute_cost_constraint_and_gradient(self, x):
```

直接读取的对象属性：`self._cost_and_gradient_with_guard`、`self.rollout_fn`。

输出分支的源码表达式：

```python
self._cost_and_gradient_with_guard(x, self.rollout_fn)
```

直接调用：

```python
self._cost_and_gradient_with_guard
```

## `PhysicalGradientCore._compute_cost_constraint_and_gradient_initial`

[定义位置](../../../manipulation/curobo_solver_components.py#L180)。

输入签名：

```python
def PhysicalGradientCore._compute_cost_constraint_and_gradient_initial(self, x):
```

直接读取的对象属性：`self._rollout_list`、`self._cost_and_gradient_with_guard`。

输出分支的源码表达式：

```python
self._cost_and_gradient_with_guard(x, rollout)
```

直接调用：

```python
len
self._cost_and_gradient_with_guard
```

## `PhysicalGradientCore._begin_functional_solve`

[定义位置](../../../manipulation/curobo_solver_components.py#L188)。

输入签名：

```python
def PhysicalGradientCore._begin_functional_solve(self, action):
```

直接读取的对象属性：`self.rollout_fn`、`self._functional_constraints`、`self._candidate_tracking`、`self.config`、`self._candidate_action`、`self._candidate_objective`、`self._candidate_rank`。

输出分支的源码表达式：

```python
None
```

直接写入的状态或数组位置：`self._functional_constraints`、`self._candidate_tracking`、`self._last_functional_signed`。

直接调用：

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
@torch.no_grad()
def PhysicalGradientCore._retain_sampled_candidate(self, action, hard_eligible, sampled_feasible, objective):
```

直接读取的对象属性：`self.config`、`self.action_horizon`、`self.action_dim`、`self._candidate_rank`、`self._candidate_objective`、`self._candidate_action`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

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
def PhysicalGradientCore._finish_functional_block(self, state):
```

直接读取的对象属性：`self._al_blocks_remaining`、`self._functional_constraints`、`self.config`、`self.opt_dim`、`self._compute_cost_constraint_and_gradient`、`self._last_functional_signed`、`self._check_request`、`self.action_horizon`、`self.action_dim`、`self._restart_changed_objective`。

输出分支的源码表达式：

```python
state
```

直接写入的状态或数组位置：`self._al_blocks_remaining`。

直接调用：

```python
state.exploration_action.detach().clone
state.exploration_action.detach
base[:, None].expand(-1, particles, -1, -1).contiguous().view
base[:, None].expand(-1, particles, -1, -1).contiguous
base[:, None].expand
self._compute_cost_constraint_and_gradient
value.view
self._last_functional_signed.items
owner.update
self._check_request
bool
changed.any
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

[定义位置](../../../manipulation/curobo_solver_components.py#L274)。

输入签名：

```python
@torch.no_grad()
def PhysicalGradientCore._restart_changed_objective(self, state, base, cost, gradient, changed):
```

直接读取的对象属性：`self.config`、`self._best`、`self._on_reinitialize`、`self._on_initial_state`、`self._history_buffers`、`self._stalled_steps`、`self._exhausted`、`self._last_displacement`、`self._last_curvature`、`self._history_resets`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`state.converged[changed]`、`self._best.converged[changed]`、`self._best.previous_step_direction[changed]`、`self._stalled_steps[changed]`、`self._exhausted[changed]`、`self._last_displacement[changed]`、`self._last_curvature[changed]`、`self._all_exhausted`。

直接调用：

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

[定义位置](../../../manipulation/curobo_solver_components.py#L276)。

输入签名：

```python
def PhysicalGradientCore._restart_changed_objective.copy_rows(target, value):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
changed.reshape
target.copy_
torch.where
value.reshape_as
```

## `PhysicalGradientCore._reset_progress`

[定义位置](../../../manipulation/curobo_solver_components.py#L315)。

输入签名：

```python
def PhysicalGradientCore._reset_progress(self):
```

直接读取的对象属性：`self._progress_buffers`、`self.config`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._all_exhausted`、`self._recovery_reports`。

直接调用：

```python
getattr(self, name).zero_
getattr
range
```

## `PhysicalGradientCore._guarded_step_direction`

[定义位置](../../../manipulation/curobo_solver_components.py#L322)。

输入签名：

```python
@torch.no_grad()
def PhysicalGradientCore._guarded_step_direction(self, state):
```

直接读取的对象属性：`self._history_buffers`、`self._exhausted`、`self._native_step_direction_fn`、`self.config`、`self._history_skips`、`self._history_resets`、`self._last_curvature`。

输出分支的源码表达式：

```python
history.step_q_buffer.view_as(state.exploration_action)
```

直接调用：

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

[定义位置](../../../manipulation/curobo_solver_components.py#L367)。

输入签名：

```python
def PhysicalGradientCore._guarded_opt_step(self, state):
```

直接读取的对象属性：`self._exhausted`、`result.exploration_action`、`result.exploration_cost`、`self._last_displacement`、`self._effective_updates`、`self._iterations`、`self._stalled_steps`。

输出分支的源码表达式：

```python
result
```

直接调用：

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

[定义位置](../../../manipulation/curobo_solver_components.py#L391)。

输入签名：

```python
@property
def PhysicalGradientCore.solve_diagnostics(self):
```

直接读取的对象属性：`self._progress_buffers`、`self._recovery_reports`、`self.config`、`self._candidate_tracking`、`self._functional_constraints`、`self._candidate_rank`、`self._candidate_objective`。

输出分支的源码表达式：

```python
reports
```

直接调用：

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

[定义位置](../../../manipulation/curobo_solver_components.py#L420)。

输入签名：

```python
def PhysicalGradientCore._recover_stalled(self, state):
```

直接读取的对象属性：`self._check_request`、`self._stalled_steps`、`self.config`、`self._exhausted`、`self._line_search_context`、`self._compute_cost_constraint_and_gradient`、`self.opt_dim`、`self.domain`、`self.action_horizon_step_max`、`self.action_horizon`、`self.action_dim`、`self._effective_updates`、`self._history_buffers`、`self._history_resets`、`self._recovery_reports`。

输出分支的源码表达式：

```python
state
```

直接写入的状态或数组位置：`reports[index]`、`reports[index]['line_search_evaluations']`、`self._stalled_steps[index]`、`self._effective_updates[index]`、`self._exhausted[index]`、`reports[index]['acceptance_reason']`、`self._history_resets[index]`、`self._all_exhausted`。

直接调用：

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

[定义位置](../../../manipulation/curobo_solver_components.py#L537)。

输入签名：

```python
def PhysicalGradientCore._create_initial_iteration_state(self, action):
```

直接读取的对象属性：`self.domain`、`self._safe_action`。

输出分支的源码表达式：

```python
state
```

直接写入的状态或数组位置：`state.best_action`。

直接调用：

```python
self.domain.validate
torch.where
super()._create_initial_iteration_state
super
action.clone
```

## `PhysicalGradientCore.update_num_problems`

[定义位置](../../../manipulation/curobo_solver_components.py#L547)。

输入签名：

```python
def PhysicalGradientCore.update_num_problems(self, num_problems):
```

直接读取的对象属性：`self.reset_cuda_graph`、`self.action_horizon`、`self.action_dim`、`self.device_cfg`、`self._line_search_context`、`self._create_line_search_context`、`self._compute_cost_constraint_and_gradient_initial`、`self.domain`、`self._safe_action`、`self._progress_buffers`、`self._reset_progress`、`self._line_search_strategy`、`self._initial_line_search_strategy`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self._safe_action`、`self._line_search_context`、`self._initial_line_search_context`、`self._line_search_strategy`、`self._initial_line_search_strategy`、`self._candidate_action`、`self._candidate_objective`、`self._candidate_rank`、`self._line_search_strategy.safe_action`、`self._initial_line_search_strategy.safe_action`。

直接调用：

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

[定义位置](../../../manipulation/curobo_solver_components.py#L582)。

输入签名：

```python
def PhysicalGradientCore._prepare_solve(self, action):
```

直接读取的对象属性：`self.config`、`self.action_horizon`、`self.action_dim`、`self.domain`、`self._line_search_context`、`self._initial_line_search_context`、`self._safe_action`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
action.view
self.domain.validate
self.domain.validate_scales
self._safe_action.copy_
action.detach
```

## `PhysicalGradientCore.reinitialize`

[定义位置](../../../manipulation/curobo_solver_components.py#L590)。

输入签名：

```python
def PhysicalGradientCore.reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
```

直接读取的对象属性：`self._prepare_solve`、`self._reset_progress`、`self._begin_functional_solve`。

输出分支的源码表达式：

```python
result
```

直接调用：

```python
self._prepare_solve
self._reset_progress
self._begin_functional_solve
super().reinitialize
super
```

## `PhysicalGradientCore.optimize`

[定义位置](../../../manipulation/curobo_solver_components.py#L600)。

输入签名：

```python
def PhysicalGradientCore.optimize(self, seed_action):
```

直接读取的对象属性：`self._on_reinitialize`、`self._prepare_solve`、`self._reset_progress`、`self._best`、`self._begin_functional_solve`、`self.outer_iters`、`self._candidate_tracking`、`self._candidate_rank`、`self._candidate_action`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`self._iteration_state`、`self._al_blocks_remaining`。

直接调用：

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

[定义位置](../../../manipulation/curobo_solver_components.py#L628)。

输入签名：

```python
def PhysicalGradientCore._capture_state(self):
```

直接读取的对象属性：`self._history_buffers`、`self._best`、`self._device_checks`、`self._progress_buffers`、`self._candidate_buffers`、`self._line_search_strategy`、`self._initial_line_search_strategy`。

输出分支的源码表达式：

```python
records
```

直接写入的状态或数组位置：`self.graph_snapshot_bytes`。

直接调用：

```python
getattr
records.append
tensor.clone
sum
saved.numel
saved.element_size
```

## `PhysicalGradientCore._restore_capture_state`

[定义位置](../../../manipulation/curobo_solver_components.py#L662)。

输入签名：

```python
@staticmethod
def PhysicalGradientCore._restore_capture_state(records):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
getattr
RuntimeError
tensor.copy_
```

## `PhysicalGradientCore._dispatch`

[定义位置](../../../manipulation/curobo_solver_components.py#L668)。

输入签名：

```python
def PhysicalGradientCore._dispatch(self, method_name, *args):
```

直接读取的对象属性：`self._check_request`、`self._all_exhausted`、`self._finish_functional_block`、`self.use_cuda_graph`、`self._recover_stalled`、`self._device_checks`、`self._executors`、`self._replay_graph`、`self._best`。

输出分支的源码表达式：

```python
self._finish_functional_block(args[0])
result
```

直接调用：

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

[定义位置](../../../manipulation/curobo_solver_components.py#L704)。

输入签名：

```python
def PhysicalGradientCore._replay_graph(self, method_name, *args):
```

直接读取的对象属性：`self._executors`、`self.device_cfg`、`self._capture_state`、`self._device_checks`、`self._restore_capture_state`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`self.graph_capture_bytes`。

直接调用：

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

[定义位置](../../../manipulation/curobo_solver_components.py#L736)。

输入签名：

```python
def PhysicalGradientCore.reset_cuda_graph(self):
```

直接读取的对象属性：`self._device_checks`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.graph_capture_bytes`。

直接调用：

```python
super().reset_cuda_graph
super
hasattr
self._device_checks.clear_constants
```

## `PhysicalGradientCore.reset_shape`

[定义位置](../../../manipulation/curobo_solver_components.py#L742)。

输入签名：

```python
def PhysicalGradientCore.reset_shape(self):
```

直接读取的对象属性：`self.reset_cuda_graph`。

输出分支的源码表达式：

```python
super().reset_shape()
```

直接调用：

```python
self.reset_cuda_graph
super().reset_shape
super
```

## `PhysicalGradientCore.extra_workspace_bytes`

[定义位置](../../../manipulation/curobo_solver_components.py#L749)。

输入签名：

```python
@property
def PhysicalGradientCore.extra_workspace_bytes(self):
```

直接读取的对象属性：`self._safe_action`、`self._progress_buffers`、`self._candidate_buffers`、`self._functional_constraints`、`self._device_checks`、`self.graph_snapshot_bytes`。

输出分支的源码表达式：

```python
safe_bytes + progress_bytes + self._device_checks.storage_bytes + self.graph_snapshot_bytes
```

直接调用：

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

[定义位置](../../../manipulation/curobo_solver_components.py#L764)。

输入签名：

```python
def PhysicalGradientCore.bind_domain(self, domain: JointPositionDomain):
```

直接读取的对象属性：`self.reset_cuda_graph`、`self._reset_progress`、`self._line_search_strategy`、`self._initial_line_search_strategy`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.domain`、`self._iteration_state`、`self._line_search_strategy.domain`、`self._initial_line_search_strategy.domain`。

直接调用：

```python
self.reset_cuda_graph
self._reset_progress
```

## `LayoutGoalManager`

[定义位置](../../../manipulation/curobo_solver_components.py#L776)。

```python
class LayoutGoalManager(GoalManager):
    ...
    ...
```

该类型的直接方法：`_get_padded_goalset_for_links`。

## `LayoutGoalManager._get_padded_goalset_for_links`

[定义位置](../../../manipulation/curobo_solver_components.py#L779)。

输入签名：

```python
def LayoutGoalManager._get_padded_goalset_for_links(self, solve_state, current_solve_state, current_goal_buffer, links_goal_pose):
```

输出分支的源码表达式：

```python
None
super()._get_padded_goalset_for_links(solve_state, current_solve_state, current_goal_buffer, links_goal_pose)
```

直接调用：

```python
replace
super()._get_padded_goalset_for_links
super
```

## `bind_solver_components`

[定义位置](../../../manipulation/curobo_solver_components.py#L791)。

输入签名：

```python
def bind_solver_components(solver, domain: JointPositionDomain, *, use_optimizer_graph=False) -> None:
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`optimizer._core`、`solver.core.goal_registry_manager`。

直接调用：

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
