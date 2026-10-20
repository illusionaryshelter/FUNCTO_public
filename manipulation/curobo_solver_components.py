"""FUNCTO-owned components for the native planning solvers.

cuRobo still owns L-BFGS, rollout evaluation and goal-buffer allocation. This
module composes its existing internal components at the FUNCTO solver boundary:
physical joint domains constrain every evaluated search point, and a goal buffer
is reused only when its problem/seed layout is unchanged. No cuRobo class,
factory registry or module function is modified process-wide.

These extensions deliberately depend on cuRobo's internal component interfaces.
They support the approximate-Wolfe L-BFGS configuration used by this planner.
An explicitly admitted native CUDA-kernel optimizer may capture each native step;
the solver's separate CPU geometry metrics remain outside CUDA graphs.
"""

# BEGIN instance-local native solver composition owned entirely by FUNCTO.
from dataclasses import fields, replace

import torch
from curobo import runtime as curobo_runtime

from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.gradient.lbfgs import LBFGSOpt
from curobo._src.optim.gradient.line_search_strategy import (
    ApproxWolfeLineSearchStrategy,
    LineSearchType,
)
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.util.cuda_graph_util import create_graph_executor

from .physical_joint_domain import JointPositionDomain
from .optimizer_device_checks import DeviceChecks


class PhysicalLineSearch(ApproxWolfeLineSearchStrategy):
    """Keep native Wolfe selection, restricting its points before any rollout."""

    def __init__(self, domain: JointPositionDomain, safe_action: torch.Tensor):
        super().__init__()
        self.domain = domain
        self.safe_action = safe_action

    def _prepare_search_points(self, x, step_direction, context):
        # Native preparation checks shapes and applies step scaling/terminal
        # fixing. Its temporary points are never evaluated. Retain its adjusted
        # direction, shorten it against the physical domain, then form the
        # actual points and matching direction consumed by Wolfe selection.
        _, direction = super()._prepare_search_points(x, step_direction, context)
        base = x.detach()
        # BEGIN defer only error reporting; a bad action never enters a rollout.
        # Outside a capture scope validate raises exactly as before. Inside the
        # scope it records a device error and the complete batch is replaced by
        # the current solve's independently validated seed. Valid values are
        # copied by where without any arithmetic or reduction rewrite.
        valid = self.domain.validate(base)
        if valid is not None:
            base = torch.where(valid, base, self.safe_action)
        direction = self.domain.limit_step_tensor(
            base, direction.reshape_as(base), context.line_search_scale)
        valid = self.domain.validate(base[:, None]+context.line_search_scale*direction[:, None])
        if valid is not None:
            # A zero step evaluates the already valid base at every multiplier.
            direction = torch.where(valid, direction, torch.zeros_like(direction))
        # END static multipliers are checked at each native solve entry.
        points = self.jit_get_x_set(direction, base, context.line_search_scale)
        return (
            points.view(x.shape[0], context.n_linesearch, context.opt_dim)
                  .detach().requires_grad_(True),
            direction.view(x.shape[0], 1, context.opt_dim),
        )


class PhysicalGradientCore(GradientOptCore):
    """Native gradient lifecycle with a legal initial best state and trials."""

    def __init__(self, *args, domain: JointPositionDomain, history_buffers=None, **kwargs):
        self.domain = domain
        self._history_buffers = history_buffers
        self._safe_action = None
        # BEGIN guard the installed native callback, including its fused CUDA path.
        self._native_step_direction_fn = kwargs["step_direction_fn"]
        kwargs["step_direction_fn"] = self._guarded_step_direction
        self._all_exhausted = False
        self.request_guard = None
        self._recovery_reports = []
        # BEGIN continuation state is local to one native optimize invocation.
        self._functional_constraints = None
        self._candidate_tracking = False
        self._al_blocks_remaining = 0
        self._last_functional_signed = None
        # END independent seeds never inherit another solve's dual variables.
        # END no third-party optimizer or process-wide callback is replaced.
        super().__init__(*args, **kwargs)
        # BEGIN scope graph checks independently of the CPU metrics rollout.
        self._device_checks = DeviceChecks(self.device_cfg.device)
        self.graph_capture_bytes = 0
        self.graph_snapshot_bytes = 0
        # END graph allocation is measured separately from known tensor storage.

    def finish_init(self):
        # BEGIN capture one guarded native step; recovery runs between blocks.
        # The ordinary native kernels retain their arithmetic. The owned step
        # protects invalid history and freezes exhausted rows on the device.
        if self._deferred_cuda_graph:
            self._executors["_opt_step"] = create_graph_executor(
                capture_fn=self._guarded_opt_step, device=self.device_cfg.device, use_cuda_graph=True)
            self._executors["_prepare_initial_iteration_state"] = create_graph_executor(
                capture_fn=self._prepare_initial_iteration_state,
                device=self.device_cfg.device, use_cuda_graph=True)
        # END metrics and the original Python iteration loop stay outside capture.

    # BEGIN the same request deadline bounds each actual native optimization step.
    def _check_request(self, *, wait_for_device=False):
        """Check cancellation/budget without inventing another deadline.

        Checks surround bounded calls; an already submitted CUDA kernel or
        backward pass remains non-preemptible. Captured steps are checked by
        their host replay boundary, never by a timer frozen into the graph.
        """
        if self.request_guard is not None:
            # Synchronize only this core's current CUDA stream at completed
            # forward/backward/step boundaries. Otherwise a fast CPU could
            # enqueue a whole overdue block before the host clock sees it.
            # Graph capture cannot synchronize internally; its host replay
            # boundary completes the captured step before checking instead.
            if (wait_for_device and self.device_cfg.device.type == "cuda"
                    and not torch.cuda.is_current_stream_capturing()):
                torch.cuda.current_stream(self.device_cfg.device).synchronize()
            self.request_guard()

    def _opt_step(self, iteration_state):
        self._check_request()
        result = (self._guarded_opt_step(iteration_state) if not self.use_cuda_graph
                  else self._replay_graph("_opt_step", iteration_state))
        self._check_request(wait_for_device=True)
        return result

    def _run_initial_evaluation(self, iteration_state):
        # Native reinitialize bypasses _dispatch and the preparation wrapper;
        # this hook covers both initialization entrances with their real cost.
        self._check_request()
        result = super()._run_initial_evaluation(iteration_state)
        self._check_request(wait_for_device=True)
        return result

    def _cost_and_gradient_with_guard(self, x, rollout):
        """Keep native cost/backward arithmetic, checking their host boundary.

        cuRobo's two cost callbacks have no before-backward hook. This owned
        implementation retains their detach, reshape, reduction, backward
        vector and return tensors. The only added operation is the same
        request check before forward, before backward and after backward.
        """
        from curobo._src.util.logging import log_and_raise
        self._check_request()
        x_n = x.detach().requires_grad_(True)
        x_in = x_n.view(self.config.num_problems * self.config.num_particles,
                       self.action_horizon, self.action_dim)
        trajectories = rollout.evaluate_action(x_in, use_cuda_graph=False)
        costs = trajectories.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True)
        if costs.shape != (self.config.num_problems * self.config.num_particles,):
            log_and_raise(f"costs.shape != (num_problems*num_particles,): {costs.shape}")
        cost = costs.view(self.config.num_problems, self.config.num_particles, 1)
        # A forward evaluation that consumes the deadline must not initiate
        # another expensive backward pass or publish its partial candidate.
        self._check_request(wait_for_device=True)
        cost.backward(gradient=self._l_vec, retain_graph=False)
        self._check_request(wait_for_device=True)
        g_x = x_n.grad.detach()
        # BEGIN retain graded candidate evidence from this exact completed query.
        if self._candidate_tracking:
            hard_eligible, sampled_feasible, signed, objective = rollout.constraint_manager.optimizer_evidence(trajectories)
            self._last_functional_signed = signed
            self._retain_sampled_candidate(x_in, hard_eligible, sampled_feasible, objective)
        # END this performs no FK, scene query, or dense validation.
        return cost, g_x

    def _compute_cost_constraint_and_gradient(self, x):
        return self._cost_and_gradient_with_guard(x, self.rollout_fn)

    def _compute_cost_constraint_and_gradient_initial(self, x):
        rollout = self._rollout_list[1] if len(self._rollout_list) > 1 else self._rollout_list[0]
        return self._cost_and_gradient_with_guard(x, rollout)
    # END no native iteration count, line-search arithmetic or budget is relaxed.

    # BEGIN fixed-budget inexact PHR continuation and sampled-feasible retention.
    _candidate_buffers = ("_candidate_action", "_candidate_objective", "_candidate_rank")

    def _begin_functional_solve(self, action):
        """Declare seed layout before the first evaluation and reset in place.

        A collision/search recovery can replace or replicate seeds. Each such
        native optimize call therefore starts its own dual sequence; only its
        existing inner blocks share parameters. Graph-owned addresses survive
        unchanged layouts, and graph warmup snapshots include the incumbent.
        """
        manager = getattr(self.rollout_fn, "constraint_manager", None)
        self._functional_constraints = getattr(manager, "optimization_constraints", None)
        self._candidate_tracking = self._functional_constraints is not None
        self._last_functional_signed = None
        if not self._candidate_tracking:
            return
        if not self.config.fixed_iters:
            raise ValueError("Functional block continuation requires the configured fixed iteration budget")
        self._functional_constraints.bind(action, self.config.num_problems)
        self._functional_constraints.reset()
        self._candidate_action.copy_(action.detach().view_as(self._candidate_action))
        self._candidate_objective.fill_(float("inf"))
        self._candidate_rank.zero_()

    @torch.no_grad()
    def _retain_sampled_candidate(self, action, hard_eligible, sampled_feasible, objective):
        """Keep one action per seed, preferring stronger sampled evidence.

        Rank 0 has no candidate; positive ranks record the caller's hard
        eligibility and sampled-feasibility flags. The caller's eligibility
        checks delivery requirements; sampled feasibility additionally tracks
        the unchanged nominal optimization targets. In particular, support
        gap and normal errors cannot discard an otherwise eligible curve.
        Ordinary/dense final metrics still decide acceptance. Within one rank,
        the unchanged parameter-free objective orders all line-search trials,
        including points which Wolfe did not select. One slot bounds storage.
        """
        count, particles = self.config.num_problems, self.config.num_particles
        actions = action.detach().view(count, particles, self.action_horizon, self.action_dim)
        # BEGIN hard evidence gates both ranks, including inconsistent caller flags.
        valid = hard_eligible.view(count, particles) & torch.isfinite(actions).flatten(2).all(-1)
        values = objective.detach().view(count, particles)
        valid = valid & torch.isfinite(values)
        ranks = torch.where(valid, 1+sampled_feasible.view(count, particles).to(torch.uint8), 0)
        rank = ranks.amax(-1)
        ranked = torch.where(valid & (ranks == rank[:, None]), values, float("inf"))
        value, index = ranked.min(-1)
        selected = actions.gather(1, index[:, None, None, None].expand(-1, 1,
                                  self.action_horizon, self.action_dim)).squeeze(1)
        improved = (rank > 0) & ((rank > self._candidate_rank)
            | ((rank == self._candidate_rank) & (value < self._candidate_objective)))
        self._candidate_action.copy_(torch.where(improved[:, None, None], selected, self._candidate_action))
        self._candidate_objective.copy_(torch.where(improved, value, self._candidate_objective))
        self._candidate_rank.copy_(torch.where(improved, rank, self._candidate_rank))
        # END retained candidates preserve the supplied hard evidence.

    def _finish_functional_block(self, state):
        """Change parameters only between existing blocks, never add a solve.

        Native exploration_action is the next line search's actual base. Its
        residual cannot be inferred from the last callback, which evaluated
        several different trials. Re-evaluate that base in the ordinary layout,
        update each seed independently, then recompute changed-objective values.
        No update follows the final block, which has no remaining iterations.
        """
        if not self._al_blocks_remaining:
            return state
        self._al_blocks_remaining -= 1
        owner = self._functional_constraints
        if not self._al_blocks_remaining or owner is None or not owner.keys:
            return state
        count, particles = self.config.num_problems, self.config.num_particles
        base = state.exploration_action.detach().clone()
        repeated = base[:, None].expand(-1, particles, -1, -1).contiguous().view(count, particles, self.opt_dim)
        # BEGIN the old objective supplies residuals and candidate evidence only.
        # Its gradient is never consumed after the multipliers change. Keep the
        # exact exploration base and candidate-retention side effect, but do
        # not construct an autograd graph for this observation.
        self._check_request()
        with torch.no_grad():
            action = repeated.view(count * particles, self.action_horizon, self.action_dim)
            trajectories = self.rollout_fn.evaluate_action(action, use_cuda_graph=False)
            eligible, feasible, raw_signed, objective = (
                self.rollout_fn.constraint_manager.optimizer_evidence(trajectories))
            self._last_functional_signed = raw_signed
            self._retain_sampled_candidate(action, eligible, feasible, objective)
            signed = {name: value.view(count, particles)[:, 0]
                      for name, value in raw_signed.items()}
        self._check_request(wait_for_device=True)
        # END changed-objective costs/gradients below still restart its history.
        changed = owner.update(signed)
        self._check_request(wait_for_device=True)
        if not bool(changed.any()):
            return state
        costs, gradients = self._compute_cost_constraint_and_gradient(repeated)
        cost = costs.detach()[:, 0, 0]
        gradient = gradients.detach().view(count, particles, self.action_horizon, self.action_dim)[:, 0]
        if not bool((torch.isfinite(cost) & torch.isfinite(gradient).flatten(1).all(-1))[changed].all()):
            raise ValueError("Changed functional objective has nonfinite continuation cost or gradient")
        self._restart_changed_objective(state, base, cost, gradient, changed)
        return state

    @torch.no_grad()
    def _restart_changed_objective(self, state, base, cost, gradient, changed):
        """Invalidate only changed rows; retain all cumulative work counters."""
        def copy_rows(target, value):
            mask = changed.reshape((-1,)+(1,)*(target.ndim-1))
            target.copy_(torch.where(mask, value.reshape_as(target), target))

        for name in ("action", "exploration_action", "best_action"):
            copy_rows(getattr(state, name), base)
        for name in ("cost", "exploration_cost", "best_cost"):
            copy_rows(getattr(state, name), cost)
        for name in ("gradient", "exploration_gradient"):
            copy_rows(getattr(state, name), gradient)
        copy_rows(state.step_direction, -self.config.initial_step_scale*gradient)
        copy_rows(state.best_iteration, state.current_iteration)
        state.converged[changed] = 0
        # The native tracker can own separate tensors (in particular in graph
        # replay). Rebind its scalar/action ordering without clearing time or
        # iteration counters through native reinitialize/BestTracker.clear.
        copy_rows(self._best.cost, cost)
        copy_rows(self._best.action, base)
        copy_rows(self._best.iteration, state.current_iteration)
        self._best.converged[changed] = 0
        self._best.previous_step_direction[changed] = 0
        if self._on_reinitialize is not None:
            self._on_reinitialize(changed)
        if self._on_initial_state is not None:
            self._on_initial_state(state, changed)
        if self._history_buffers is not None:
            copy_rows(self._history_buffers.step_q_buffer, state.step_direction)
        self._stalled_steps[changed] = 0
        self._exhausted[changed] = False
        self._last_displacement[changed] = 0
        self._last_curvature[changed] = 0
        self._history_resets.add_(changed.to(torch.int64))
        self._all_exhausted = bool(self._exhausted.all())
    # END finite block continuation cannot itself declare a feasible solution.

    # BEGIN real-displacement history contract and bounded recovery lifecycle.
    _progress_buffers = ("_exhausted", "_stalled_steps", "_effective_updates", "_iterations",
                         "_history_skips", "_history_resets", "_last_displacement", "_last_curvature")

    def _reset_progress(self):
        for name in self._progress_buffers:
            getattr(self, name).zero_()
        self._all_exhausted = False
        self._recovery_reports = [[] for _ in range(self.config.num_problems)]

    @torch.no_grad()
    def _guarded_step_direction(self, state):
        """Admit only actual, finite, positive-curvature exploration samples.

        The native CUDA operation fuses history insertion and two-loop
        recursion. Run it on the existing full batch, then restore skipped
        rows in place; neither graph addresses nor active-row arithmetic
        change. A zero displacement never evicts a useful history pair.
        """
        history = self._history_buffers
        q = state.exploration_action.reshape_as(history.x_0)
        gradient = state.exploration_gradient.reshape_as(history.grad_0)
        reference_q, reference_gradient = history.x_0.clone(), history.grad_0.clone()
        displacement = q-reference_q
        gradient_change = gradient-reference_gradient
        s, y = displacement.double().flatten(1), gradient_change.double().flatten(1)
        curvature = (s*y).sum(-1)
        moved = (displacement != 0).flatten(1).any(-1)
        finite = torch.isfinite(s).all(-1) & torch.isfinite(y).all(-1)
        scale = torch.linalg.vector_norm(s, dim=-1)*torch.linalg.vector_norm(y, dim=-1)
        valid = (~self._exhausted & moved & finite
                 & (curvature > torch.finfo(q.dtype).eps*scale))
        saved = {name: getattr(history, name).clone() for name in ("s", "y", "rho", "alpha")}
        direction = self._native_step_direction_fn(state)
        g, d = gradient.double().flatten(1), direction.double().flatten(1)
        descending = torch.isfinite(d).all(-1) & ((g*d).sum(-1) < 0.)
        reset = ~self._exhausted & ((moved & ~valid) | (valid & ~descending))
        retain = valid & descending
        for name, before in saved.items():
            tensor = getattr(history, name)
            restored = torch.where(retain[None, :, None, None], tensor, before)
            tensor.copy_(torch.where(reset[None, :, None, None], torch.zeros_like(tensor), restored))
        # Reset references to the actual state and its matching gradient,
        # including skipped zero steps; zeroing references fabricates a step.
        history.x_0.copy_(torch.where(self._exhausted[:, None, None], reference_q, q))
        history.grad_0.copy_(torch.where(self._exhausted[:, None, None], reference_gradient, gradient))
        fallback = -self.config.initial_step_scale*state.exploration_gradient
        fallback = torch.where(torch.isfinite(fallback), fallback, torch.zeros_like(fallback))
        direction = torch.where(retain[:, None, None], direction, fallback)
        direction = torch.where(self._exhausted[:, None, None], torch.zeros_like(direction), direction)
        history.step_q_buffer.copy_(direction.reshape_as(history.step_q_buffer))
        self._history_skips.add_((~self._exhausted & ~valid).to(torch.int64))
        self._history_resets.add_(reset.to(torch.int64))
        self._last_curvature.copy_(torch.where(self._exhausted, self._last_curvature, curvature))
        return history.step_q_buffer.view_as(state.exploration_action)

    def _guarded_opt_step(self, state):
        # Native best tracking mutates aliases. Snapshot before it runs so a
        # retired row cannot change its candidate or counters while peers run.
        before = {field.name: value.clone() for field in fields(state)
                  if isinstance(value := getattr(state, field.name), torch.Tensor)}
        result = super()._opt_step(state)
        for name, previous in before.items():
            tensor = getattr(result, name)
            mask = self._exhausted.reshape((-1,)+(1,)*(tensor.ndim-1))
            tensor.copy_(torch.where(mask, previous, tensor))
        previous = before["exploration_action"]
        displacement = (result.exploration_action-previous).abs().flatten(1).amax(-1)
        resolution = torch.finfo(previous.dtype).eps*previous.abs().flatten(1).amax(-1).clamp_min(1.)
        effective = ((displacement > resolution)
                     & (result.exploration_cost < before["exploration_cost"])
                     & ~self._exhausted)
        self._last_displacement.copy_(torch.where(self._exhausted, self._last_displacement, displacement.double()))
        self._effective_updates.add_(effective.to(torch.int64))
        self._iterations.add_((~self._exhausted).to(torch.int64))
        self._stalled_steps.copy_(torch.where(effective, torch.zeros_like(self._stalled_steps),
                                             self._stalled_steps+(~self._exhausted).to(torch.int64)))
        return result

    @property
    def solve_diagnostics(self):
        """Read per-seed progress once outside graphs after an optimizer solve."""
        values = {name: getattr(self, name).detach().cpu().tolist() for name in self._progress_buffers}
        reports = [{"seed_index": index, "iterations": values["_iterations"][index],
                 "effective_updates": values["_effective_updates"][index],
                 "history_skips": values["_history_skips"][index],
                 "history_resets": values["_history_resets"][index],
                 "last_displacement": values["_last_displacement"][index],
                 "last_curvature": values["_last_curvature"][index],
                 "exhausted": values["_exhausted"][index],
                 "stop_reason": "recovery_without_descent" if values["_exhausted"][index] else "iteration_budget",
                 "recovery_attempts": len(self._recovery_reports[index]),
                 "recovery_accepted": sum(row["accepted"] for row in self._recovery_reports[index]),
                 "recoveries": list(self._recovery_reports[index])}
                for index in range(self.config.num_problems)]
        # BEGIN report independent continuation and sampled-only incumbents.
        if self._candidate_tracking:
            duals = self._functional_constraints.snapshot()["seeds"]
            ranks = self._candidate_rank.detach().cpu().tolist()
            objective = self._candidate_objective.detach().cpu().tolist()
            for index, report in enumerate(reports):
                report.update(functional_continuation=duals[index],
                              retained_candidate_rank=ranks[index],
                              sampled_feasible_candidate=ranks[index] == 2,
                              pending_collision_confirmation=ranks[index] == 1,
                              retained_candidate_objective=objective[index] if ranks[index] else None)
        return reports
        # END final ordinary/dense acceptance remains a separate planner report.

    def _recover_stalled(self, state):
        """Try one physically legal Armijo recovery per stalled block/seed.

        QP construction is host work. Its trials use the normal main rollout
        layout, including a fresh zero-scale cost/gradient, and never enter
        approximate-Wolfe's non-descending exploration fallback.
        """
        # BEGIN recovery uses the same bounded request as ordinary steps.
        self._check_request()
        stalled = ((self._stalled_steps >= self.config.inner_iters) & ~self._exhausted)
        indices = stalled.nonzero().flatten().tolist()
        if not indices:
            return state
        context = self._line_search_context
        base = state.exploration_action.detach().clone()
        # Recompute the main objective BEFORE constructing its QP. An initial
        # rollout or objective rebind can carry a different gradient at the
        # same action; the cost, direction and acceptance reference must agree.
        repeated = base[:, None].expand(-1, context.n_linesearch, -1, -1).contiguous()
        base_costs, base_gradients = self._compute_cost_constraint_and_gradient(
            repeated.reshape(self.config.num_problems, context.n_linesearch, self.opt_dim))
        base_cost = base_costs.detach()[:, 0, 0]
        base_gradient = base_gradients.reshape_as(repeated)[:, 0].detach()
        # Best tracking may also carry the initial rollout's cost. Re-score
        # its actual candidate under this same objective before comparing any
        # recovery; preserve the existing candidate on exact cost ties.
        repeated_best = state.best_action.detach()[:, None].expand_as(repeated).contiguous()
        best_costs, _ = self._compute_cost_constraint_and_gradient(
            repeated_best.reshape(self.config.num_problems, context.n_linesearch, self.opt_dim))
        direction = torch.zeros_like(base)
        reports = {}
        for index in indices:
            self._check_request()
            state.best_cost[index].copy_(best_costs.detach()[index, 0, 0])
            if bool(base_cost[index] < state.best_cost[index]):
                state.best_cost[index].copy_(base_cost[index])
                state.best_action[index].copy_(base[index])
                state.best_iteration[index].copy_(state.current_iteration[index])
            state.exploration_cost[index].copy_(base_cost[index])
            state.exploration_gradient[index].copy_(base_gradient[index])
            proposed, report = self.domain.recovery_step(base[index:index+1],
                base_gradient[index:index+1],
                self.action_horizon_step_max.reshape(self.action_horizon, self.action_dim),
                context.line_search_scale)
            self._check_request()
            direction[index:index+1].copy_(proposed)
            reports[index] = {**report, "accepted": False, "line_search_evaluations": 0}
        pending = [index for index in indices if reports[index]["success"]]
        # A descent direction can overshoot every native trial scale. Keep
        # their ratios and halve the complete direction at most eight times;
        # this is a bounded recovery search, independent of normal Wolfe work.
        for backtrack in range(8):
            self._check_request()
            if not pending:
                break
            direction = self.domain.limit_step(base, direction, context.line_search_scale)
            trials = base[:, None]+context.line_search_scale*direction[:, None]
            costs, gradients = self._compute_cost_constraint_and_gradient(
                trials.reshape(self.config.num_problems, context.n_linesearch, self.opt_dim))
            costs = costs.detach().squeeze(-1)
            gradients = gradients.reshape_as(trials)
            remaining = []
            for index in pending:
                reports[index]["line_search_evaluations"] += 1
                actual_step = trials[index]-base[index]
                slope = (base_gradient[index].double()[None]*actual_step.double()).flatten(1).sum(-1)
                finite = torch.isfinite(costs[index]) & torch.isfinite(gradients[index]).flatten(1).all(-1)
                valid = (finite & (slope < 0.) & (costs[index] < base_cost[index])
                         & (costs[index].double() <= base_cost[index].double()+context.line_search_c_1*slope))
                available = valid.nonzero().flatten()
                if not available.numel():
                    remaining.append(index)
                    continue
                chosen = int(available[costs[index, available].argmin()].item())
                q, gradient, cost = trials[index, chosen], gradients[index, chosen], costs[index, chosen]
                for name in ("action", "exploration_action"):
                    getattr(state, name)[index].copy_(q)
                for name in ("gradient", "exploration_gradient"):
                    getattr(state, name)[index].copy_(gradient)
                for name in ("cost", "exploration_cost"):
                    getattr(state, name)[index].copy_(cost)
                if bool(cost < state.best_cost[index]):
                    state.best_action[index].copy_(q)
                    state.best_cost[index].copy_(cost)
                    state.best_iteration[index].copy_(state.current_iteration[index])
                state.step_direction[index].copy_(-self.config.initial_step_scale*gradient)
                self._stalled_steps[index] = 0
                self._effective_updates[index] += 1
                reports[index].update(accepted=True,
                    native_multiplier_after_backtracking=float(context.line_search_scale.reshape(-1)[chosen])*.5**backtrack,
                    actual_maximum_displacement=float(actual_step[chosen].abs().max()),
                    cost_before=float(base_cost[index]), cost_after=float(cost), actual_slope=float(slope[chosen]))
                direction[index].zero_()
            pending = remaining
            direction.mul_(.5)
            self._check_request()
        for index in indices:
            if not reports[index]["accepted"]:
                self._exhausted[index] = True
                state.step_direction[index].zero_()
                reports[index]["acceptance_reason"] = "no_finite_strict_armijo_decrease"
            # Recovery creates a fresh local metric. Clear only this row and
            # bind its exact accepted (or retained) exploration state/gradient.
            history = self._history_buffers
            for name in ("s", "y", "rho", "alpha"):
                getattr(history, name)[:, index].zero_()
            history.x_0[index].copy_(state.exploration_action[index].reshape_as(history.x_0[index]))
            history.grad_0[index].copy_(state.exploration_gradient[index].reshape_as(history.grad_0[index]))
            history.step_q_buffer[index].copy_(state.step_direction[index].reshape_as(history.step_q_buffer[index]))
            self._history_resets[index] += 1
            self._recovery_reports[index].append(reports[index])
        self._all_exhausted = bool(self._exhausted.all())
        self._check_request()
        # END the two re-evaluations also use guarded forward/backward callbacks.
        return state
    # END exhausted rows preserve their best candidate while other seeds continue.

    def _create_initial_iteration_state(self, action):
        valid = self.domain.validate(action)
        if valid is not None:
            action = torch.where(valid, action, self._safe_action)
        state = super()._create_initial_iteration_state(action)
        # A zero-filled best action can violate the mechanical limits or the
        # fixed cubic entrance even if no evaluated trial is selected as best.
        state.best_action = action.clone()
        return state

    def update_num_problems(self, num_problems):
        # BEGIN every native resize replaces referenced buffers, even at B=B.
        self.reset_cuda_graph()
        self._safe_action = torch.empty(
            (num_problems, self.action_horizon, self.action_dim),
            device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        # END a solve entry fills this buffer only after validating its seed.
        if self._line_search_context is None:
            # Install both strategies before native resizing. The initialization
            # rollout and ordinary iterations must obey exactly the same domain.
            self._line_search_context = self._create_line_search_context()
            self._initial_line_search_context = self._create_line_search_context(
                compute_costs_and_gradients=self._compute_cost_constraint_and_gradient_initial)
            self._line_search_strategy = PhysicalLineSearch(self.domain, self._safe_action)
            self._initial_line_search_strategy = PhysicalLineSearch(self.domain, self._safe_action)
        super().update_num_problems(num_problems)
        # BEGIN resize every captured progress allocation with the native batch.
        for name in self._progress_buffers:
            dtype = (torch.bool if name == "_exhausted" else torch.float64
                     if name in ("_last_displacement", "_last_curvature") else torch.int64)
            setattr(self, name, torch.zeros(num_problems, device=self.device_cfg.device, dtype=dtype))
        self._reset_progress()
        # BEGIN bounded incumbent storage has exactly the existing seed layout.
        self._candidate_action = torch.zeros_like(self._safe_action)
        self._candidate_objective = torch.full((num_problems,), float("inf"),
            device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        # BEGIN one byte records evidence strength without another candidate allocation.
        self._candidate_rank = torch.zeros(num_problems, device=self.device_cfg.device, dtype=torch.uint8)
        # END graph snapshots and workspace accounting use this same persistent buffer.
        # END resize invalidated captured references before replacing these buffers.
        # END recovery records never leak across differently shaped solves.
        self._line_search_strategy.safe_action = self._safe_action
        self._initial_line_search_strategy.safe_action = self._safe_action

    # BEGIN preserve native iterations and mutable history across graph warmup.
    def _prepare_solve(self, action):
        """Check external inputs before either initial rollout or captured work."""
        action = action.view(self.config.num_problems, self.action_horizon, self.action_dim)
        self.domain.validate(action)
        for context in (self._line_search_context, self._initial_line_search_context):
            self.domain.validate_scales(context.line_search_scale)
        self._safe_action.copy_(action.detach())

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        self._prepare_solve(action)
        self._reset_progress()
        self._begin_functional_solve(action)
        result = super().reinitialize(action, mask=mask,
            clear_optimizer_state=clear_optimizer_state, reset_num_iters=reset_num_iters)
        # BEGIN native reinitialize bypasses the graph dispatch initializer.
        return result
        # END the seed has its actual goal and physical clock before observation.

    def optimize(self, seed_action):
        # BEGIN native reinitialize owns preparation of its cached initial state.
        # Consuming that state must preserve the AL parameters, candidate
        # evidence and gradient computed together. A direct optimize call has
        # no prepared state and starts the same lifecycle here.
        if self._iteration_state is None:
            if self._on_reinitialize is not None:
                self._on_reinitialize(None)
            self._prepare_solve(seed_action)
            self._reset_progress()
            self._best.clear()
            self._begin_functional_solve(seed_action)
        # END the native loop consumes and clears the single-use initial state.
        # BEGIN run AL inside the original loop and retain one candidate per seed.
        self._al_blocks_remaining = self.outer_iters
        try:
            result = super().optimize(seed_action)
            if self._candidate_tracking:
                # BEGIN both evidence ranks remain subject to final candidate validation.
                result = torch.where(self._candidate_rank[:, None, None] > 0, self._candidate_action, result)
                # END no extra seed or validation call is introduced here.
            return result
        finally:
            self._al_blocks_remaining = 0
        # END no iteration budget, solver call or result layout is added.

    def _capture_state(self):
        """Snapshot the native closure state omitted by GraphExecutor.clone().

        Native fused L-BFGS mutates these exact allocations. Its graph executor
        clones iteration-state inputs, but its three warmups also update shared
        quasi-Newton history. Restoring values in place preserves both their
        logical state and the addresses recorded by CUDA. Rollout scratch is
        deliberately not reset: its outputs are overwritten by each evaluation
        and its newly allocated fixed buffers must remain owned by that rollout.
        """
        records = []
        for owner, names in ((self._history_buffers,
                ("s", "y", "rho", "x_0", "grad_0", "alpha", "step_q_buffer")),
                (self._best, ("cost", "action", "iteration", "current_iteration",
                              "previous_step_direction", "converged")),
                (self._device_checks, ("_mask",))):
            if owner is None:
                continue
            for name in names:
                tensor = getattr(owner, name)
                if tensor is not None:
                    records.append((owner, name, tensor, tensor.clone()))
        for name in self._progress_buffers+self._candidate_buffers:
            tensor = getattr(self, name)
            records.append((self, name, tensor, tensor.clone()))
        for strategy in (self._line_search_strategy, self._initial_line_search_strategy):
            if strategy._output_buffers is not None:
                for tensor in strategy._output_buffers:
                    records.append((None, None, tensor, tensor.clone()))
        self.graph_snapshot_bytes = sum(saved.numel()*saved.element_size()
                                        for _, _, _, saved in records)
        return records

    @staticmethod
    def _restore_capture_state(records):
        for owner, name, tensor, saved in records:
            if owner is not None and getattr(owner, name) is not tensor:
                raise RuntimeError("Native optimizer rebound a captured history allocation")
            tensor.copy_(saved)

    def _dispatch(self, method_name, *args):
        # BEGIN cancellation/deadline authorization is checked outside native kernels.
        self._check_request()
        # END an in-flight native kernel is not claimed to be preemptible.
        # BEGIN stop exhausted solves before any further native rollout block.
        if method_name == "_opt_iters" and self._all_exhausted:
            return self._finish_functional_block(args[0])
        if not self.use_cuda_graph:
            result = super()._dispatch(method_name, *args)
            if method_name == "_opt_iters":
                result = self._recover_stalled(result)
                result = self._finish_functional_block(result)
            return result
        # One fault boundary surrounds the original inner-iteration block.
        # Its native loop calls our _opt_step and records each replayed state;
        # the scalar mask is neither reset nor read between those steps.
        self._device_checks.reset()
        with self._device_checks.scope():
            result = (self._replay_graph(method_name, *args)
                      if method_name in self._executors
                      else super()._dispatch(method_name, *args))
        self._device_checks.raise_if_failed()
        if method_name == "_opt_iters":
            result = self._recover_stalled(result)
            result = self._finish_functional_block(result)
        # END QP and Armijo admission occur outside capture and its fault scope.
        # GraphExecutor clones iteration-state inputs. Native eager states
        # instead alias these persistent tracker tensors; carry updated bytes
        # back once per original block for later solves without reinitialize.
        # Native initializer cost/action fields are clones, not tracker aliases.
        for source, target in (("best_iteration", "iteration"),
                               ("current_iteration", "current_iteration"),
                               ("converged", "converged")):
            getattr(self._best, target).copy_(getattr(result, source))
        return result

    def _replay_graph(self, method_name, *args):
        """Replay one native step inside its caller's shared fault boundary."""
        executor = self._executors[method_name]
        if not executor.is_initialized:
            before = torch.cuda.memory_allocated(self.device_cfg.device)
            records = self._capture_state()
            try:
                # Warmup is not a requested optimizer step. Its errors and
                # history updates are rolled back together; an existing real
                # fault is preserved by the mask snapshot, never discarded.
                self._device_checks.reset()
                with self._device_checks.scope():
                    executor.warmup(*args)
            except BaseException:
                executor.reset()
                raise
            finally:
                self._restore_capture_state(records)
            del records
            self.graph_capture_bytes += max(0, torch.cuda.memory_allocated(
                self.device_cfg.device)-before)
        # Native GraphExecutor checks only action.data_ptr() before copying a
        # whole iteration state. A returned state can share that action while
        # its next step_direction points at the L-BFGS output allocation. Copy
        # every field explicitly; otherwise the next block can replay the old
        # first direction. Passing the stored inputs avoids a duplicate copy.
        for stored, supplied in zip(executor._graph_input, args):
            stored.copy_(supplied)
        with self._device_checks.scope():
            result = executor(*executor._graph_input)
        return result

    def reset_cuda_graph(self):
        super().reset_cuda_graph()
        if hasattr(self, "_device_checks"):
            self._device_checks.clear_constants()
        self.graph_capture_bytes = 0

    def reset_shape(self):
        # Rollout shape resets can replace FK/cost scratch referenced by either
        # executor. Retain the native reset semantics after releasing graphs.
        self.reset_cuda_graph()
        return super().reset_shape()

    @property
    def extra_workspace_bytes(self):
        """Known safety/snapshot storage, excluding measured CUDA graph pools."""
        safe_bytes = 0 if self._safe_action is None else self._safe_action.numel()*self._safe_action.element_size()
        progress_bytes = sum(getattr(self, name).numel()*getattr(self, name).element_size()
                             for name in self._progress_buffers+self._candidate_buffers)
        # BEGIN account for per-seed dual tensors once at their owning core.
        if self._functional_constraints is not None:
            owner = self._functional_constraints
            progress_bytes += sum(value.numel()*value.element_size() for value in
                (owner._multiplier_buffer, owner._penalty_buffer, owner._previous_residual,
                 owner._updates, owner._penalty_capped) if value is not None)
        # END the workspace report includes all newly persistent core tensors.
        return safe_bytes+progress_bytes+self._device_checks.storage_bytes+self.graph_snapshot_bytes
    # END native kernels are reused; guarded history and recovery are owned here.

    def bind_domain(self, domain: JointPositionDomain):
        """Rebind the measured entrance and clear history for the next request."""
        # BEGIN a captured closure must not keep the previous bounds/prefix.
        self.reset_cuda_graph()
        # END the next solve validates and binds its own safe seed again.
        self.domain = domain
        self._iteration_state = None
        self._reset_progress()
        self._line_search_strategy.domain = domain
        self._initial_line_search_strategy.domain = domain


class LayoutGoalManager(GoalManager):
    """Allow goalset padding only when all other solve-layout fields match."""

    def _get_padded_goalset_for_links(
        self, solve_state, current_solve_state, current_goal_buffer, links_goal_pose,
    ):
        # num_goalset counts target alternatives; num_seeds counts optimizer
        # candidates. Padding the former cannot preserve indices for a changed
        # latter. Returning None invokes native allocation and solver resizing.
        if replace(solve_state, num_goalset=current_solve_state.num_goalset) != current_solve_state:
            return None
        return super()._get_padded_goalset_for_links(
            solve_state, current_solve_state, current_goal_buffer, links_goal_pose)


def bind_solver_components(solver, domain: JointPositionDomain, *, use_optimizer_graph=False) -> None:
    """Compose only a FUNCTO-owned solver, before its first solve.

    Reuse the existing native rollouts and L-BFGS history owner. Assign the new
    core before resizing so native L-BFGS callbacks access that core's dimensions.
    Repeated binding updates the physical domain without discarding goal state.
    """
    if solver.core.config.use_cuda_graph:
        raise ValueError("FUNCTO source-geometry metrics must stay outside CUDA graphs")
    for optimizer in solver.core.optimizers:
        if (type(optimizer) is not LBFGSOpt
                or optimizer.config.line_search_type != LineSearchType.APPROX_WOLFE):
            raise ValueError("FUNCTO physical domains require approximate-Wolfe L-BFGS")
        # BEGIN explicit graph admission never substitutes an optimizer backend.
        if use_optimizer_graph and (optimizer.device_cfg.device.type != "cuda"
                or not optimizer.config.use_cuda_kernel_step_direction
                or not optimizer.config.use_cuda_kernel_line_search
                or not optimizer.config.fixed_iters or optimizer.config.store_debug
                or curobo_runtime.debug_nan):
            # Native NaN debugging reads device predicates in Python per step;
            # report this incompatible request rather than disabling debugging.
            raise ValueError("Optimizer graphs require the unchanged fixed-iteration native CUDA L-BFGS kernels without synchronous debugging")
        if use_optimizer_graph and not curobo_runtime.cuda_graphs:
            raise ValueError("The native runtime must enable the explicitly requested optimizer CUDA graph")
        if isinstance(optimizer._core, PhysicalGradientCore):
            if optimizer.use_cuda_graph != use_optimizer_graph:
                raise ValueError("An existing optimizer cannot change its admitted CUDA graph scope")
            optimizer._core.bind_domain(domain)
            continue
        if optimizer.use_cuda_graph:
            raise ValueError("Bind FUNCTO components before enabling optimizer CUDA graphs")
        # END CPU solvers use the same owned guard around the native eager path.
        if optimizer._core._iteration_state is not None:
            raise ValueError("Bind FUNCTO optimizer components before native solving")
        core = PhysicalGradientCore(
            optimizer.config, optimizer.get_all_rollout_instances(),
            step_direction_fn=optimizer._get_step_direction_impl,
            on_reinitialize=optimizer._on_reinitialize,
            on_initial_state=optimizer._on_initial_state,
            on_resize=optimizer._on_resize,
            on_shift=optimizer._on_shift,
            domain=domain, history_buffers=optimizer._qn, use_cuda_graph=use_optimizer_graph,
        )
        optimizer._core = core
        core.update_num_problems(optimizer.config.num_problems)
        core.finish_init()
    manager = solver.core.goal_registry_manager
    if not isinstance(manager, LayoutGoalManager):
        # The public solve_state property raises before the first solve; the
        # uninitialized marker belongs to this internal composition boundary.
        if manager._solve_state is not None:
            raise ValueError("Bind FUNCTO goal components before native solving")
        solver.core.goal_registry_manager = LayoutGoalManager(solver.core.device_cfg)
# END no extension is installed in third-party source or a global registry.
