"""Instance-local cuRobo composition for physically constrained descent.

The installed GradientOptCore owns objective evaluation, seed layouts and
iteration management. FUNCTO supplies a direction in the declared joint domain
and uses the installed Armijo selector with one ordinary backtracking search.
Every QP runs on the host; robot rollouts retain their configured device. The
same implementation serves explicit position and fixed-clock cubic domains.
"""

# BEGIN shared constrained directions and native Armijo selection.
from dataclasses import replace
from time import perf_counter

import torch
from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.gradient.line_search_strategy import ArmijoLineSearchStrategy, LineSearchType
from curobo._src.solver.manager_goal import GoalManager

from .physical_joint_domain import JointPositionDomain


class PhysicalLineSearch(ArmijoLineSearchStrategy):
    """One native Armijo search, contracting only unresolved seed directions.

    A finite set of trial scales can miss every decreasing step even for a
    valid descent direction. Reuse the native selector at successively smaller
    directions until a represented decreasing step is found or all positive
    trials equal the base in the action dtype. The caller's request guard bounds
    every evaluation; AL parameters and local geometry stay fixed throughout.
    """

    def __init__(self, domain):
        self.domain = domain
        self.evaluations = 0
        self.backtracks = None

    def _prepare_search_points(self, x, step_direction, context):
        # The QP already applies the original displacement limits. Reapplying
        # native scale_action would divide every coordinate by a global norm.
        # Only actual floating-point trial feasibility is checked here; the
        # existing domain limiter resolves rounding without changing bounds.
        base = x.detach()
        direction = self.domain.limit_step(base, step_direction.detach(), context.line_search_scale)
        points = self.jit_get_x_set(direction, base, context.line_search_scale)
        self._evaluated_direction = direction
        return (points.view(base.shape[0], context.n_linesearch, context.opt_dim)
                      .detach().requires_grad_(True),
                direction.view(base.shape[0], 1, context.opt_dim))

    def _update_best_solution(self, iteration_state, context):
        # Native Armijo calls this after every scale batch. A backtrack is a
        # function evaluation, not another optimizer iteration. Commit once
        # after the independent seed results have been collected below.
        return iteration_state

    def _compute_costs_and_gradients(self, x_set, context):
        costs, gradients = super()._compute_costs_and_gradients(x_set, context)
        finite = torch.isfinite(costs) & torch.isfinite(gradients).all(-1, keepdim=True)
        if not bool(finite[:, 0].all()):
            raise ValueError("Armijo requires a finite objective and gradient at its actual base")
        self._base_cost = costs[:, 0, 0].detach()
        # A trial with no finite derivative cannot initialize the next ordinary
        # step. Native Armijo then excludes it in the same scalar comparison.
        return torch.where(finite, costs, float("inf")), gradients

    def search(self, iteration_state, context):
        base = iteration_state.exploration_action.detach()
        direction = iteration_state.step_direction.detach().clone()
        count = len(base)
        pending = torch.ones(count, device=base.device, dtype=torch.bool)
        self.backtracks = torch.zeros(count, device=base.device, dtype=torch.int64)
        self.evaluations = 0
        selected = None
        while bool(pending.any()):
            trial_state = replace(iteration_state, step_direction=direction)
            proposal = super().search(trial_state, context)
            self.evaluations += 1
            if selected is None:
                selected = proposal
            else:
                # Other seeds keep their first accepted action and matching
                # cost/gradient while a peer searches smaller trial scales.
                for name in ("action", "cost", "gradient", "exploration_action",
                             "exploration_cost", "exploration_gradient"):
                    value = getattr(proposal, name)
                    mask = pending.reshape((-1,)+(1,)*(value.ndim-1))
                    setattr(selected, name, torch.where(mask, value, getattr(selected, name)))
            represented = (base+context.line_search_scale.max()*self._evaluated_direction != base).flatten(1).any(-1)
            moved = ((proposal.exploration_action != base).flatten(1).any(-1)
                     & (proposal.exploration_cost < self._base_cost))
            # A zero native selection means no supplied positive trial passed.
            # Initial evaluation also enters here with an exactly zero direction.
            pending = pending & represented & ~moved
            self.backtracks.add_(pending.to(torch.int64))
            direction = torch.where(pending[:, None, None], self._evaluated_direction*.5, direction)
        return super()._update_best_solution(selected, context)


class PhysicalGradientCore(GradientOptCore):
    """Native iteration lifecycle with one ordinary physical descent interface."""

    def __init__(self, config, rollout_list, *, domain: JointPositionDomain):
        self.domain = domain
        self.request_guard = None
        self._functional_constraints = None
        self._local_geometry_models = None
        self._candidate_tracking = False
        self._last_functional_signed = None
        self._al_blocks_remaining = 0
        # Reuse the installed core's configuration object and shared fields.
        # Quasi-Newton fields belong to the library schema, not this algorithm.
        config.line_search_type = LineSearchType.ARMIJO
        config.solver_name = "physical_projected_gradient"
        config.use_cuda_kernel_line_search = False
        super().__init__(config, rollout_list,
            step_direction_fn=self._physical_step_direction, use_cuda_graph=False)
        self.update_num_problems(config.num_problems)

    def _check_request(self, *, wait_for_device=False):
        """Charge completed device work to the same request as host QPs."""
        if self.request_guard is not None:
            if wait_for_device and self.device_cfg.device.type == "cuda":
                started = perf_counter()
                torch.cuda.current_stream(self.device_cfg.device).synchronize()
                self._batch_work["device_wait_seconds"] += perf_counter()-started
            self.request_guard()

    def _run_initial_evaluation(self, iteration_state):
        self._check_request()
        state = super()._run_initial_evaluation(iteration_state)
        self._check_request(wait_for_device=True)
        # BEGIN the evaluated seed initializes best tracking at any cost scale.
        # The library's finite cost sentinel can be below a valid objective.
        # This first query evaluates only the seed, so its measured scalar and
        # action define the initial incumbent independently of that sentinel.
        state.best_cost.copy_(state.exploration_cost)
        state.best_action.copy_(state.exploration_action)
        state.best_iteration.copy_(state.current_iteration)
        self._best.cost.copy_(state.exploration_cost)
        self._best.action.copy_(state.exploration_action)
        self._best.iteration.copy_(state.current_iteration)
        # END every later best comparison uses a completed objective query.
        state.step_direction.copy_(self._physical_step_direction(state))
        return state

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
        started = perf_counter()
        trajectories = rollout.evaluate_action(x_in, use_cuda_graph=False)
        costs = trajectories.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True)
        self._batch_work["forward_host_seconds"] += perf_counter()-started
        self._batch_work["cost_gradient_calls"] += 1
        if costs.shape != (self.config.num_problems * self.config.num_particles,):
            log_and_raise(f"costs.shape != (num_problems*num_particles,): {costs.shape}")
        cost = costs.view(self.config.num_problems, self.config.num_particles, 1)
        # A forward evaluation that consumes the deadline must not initiate
        # another expensive backward pass or publish its partial candidate.
        self._check_request(wait_for_device=True)
        started = perf_counter()
        cost.backward(gradient=self._l_vec, retain_graph=False)
        self._batch_work["backward_host_seconds"] += perf_counter()-started
        self._check_request(wait_for_device=True)
        g_x = x_n.grad.detach()
        # BEGIN retain graded candidate evidence from this exact completed query.
        if self._candidate_tracking:
            feasible, signed, objective = rollout.constraint_manager.optimizer_evidence(trajectories)
            self._last_functional_signed = signed
            self._retain_sampled_candidate(x_in, feasible, objective)
        # END this performs no FK, scene query, or dense validation.
        return cost, g_x


    def _compute_cost_constraint_and_gradient(self, x):
        return self._cost_and_gradient_with_guard(x, self.rollout_fn)


    def _compute_cost_constraint_and_gradient_initial(self, x):
        rollout = self._rollout_list[1] if len(self._rollout_list) > 1 else self._rollout_list[0]
        return self._cost_and_gradient_with_guard(x, rollout)


    def _begin_functional_solve(self, action):
        """Declare seed layout before the first evaluation and reset in place.

        A collision/search recovery can replace or replicate seeds. Each such
        native optimize call therefore starts its own dual sequence; only its
        existing inner blocks share parameters. The current seed layout owns the multiplier and incumbent buffers.
        """
        manager = getattr(self.rollout_fn, "constraint_manager", None)
        self._functional_constraints = getattr(manager, "optimization_constraints", None)
        self._local_geometry_models = getattr(manager, "local_geometry_models", None)
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
        self._candidate_valid.zero_()
        # BEGIN initial and ordinary line searches share a captured base model.
        if self._local_geometry_models is not None:
            self._refresh_local_geometry(action)


    def _refresh_local_geometry(self, action):
        """Capture original geometry evidence at repeated actual seed bases.

        CuRobo SolverCore constructs every rollout of one optimizer from the
        same ``optimizer_rollout_configs[i]``. FUNCTO binds the same cubic
        clock and process manager to those instances. Their geometry query
        layouts therefore coincide, and the ordinary rollout captures the
        model shared by initial and ordinary gradient callbacks. The same
        completed rollout also supplies original hard evidence and a fixed
        candidate score at block boundaries. Its changing local-model value
        never enters candidate ranking or a line search.
        """
        models = self._local_geometry_models
        count, particles = self.config.num_problems, self.config.num_particles
        base = action.detach().view(count, self.action_horizon, self.action_dim)
        repeated = base[:, None].expand(-1, particles, -1, -1).contiguous().flatten(0, 1)
        models.begin_capture(count, particles)
        try:
            with torch.no_grad():
                # BEGIN one same-layout native rollout captures the block model.
                # SolverCore's shared optimizer rollout configuration is the
                # construction contract; the initial callback reads this store.
                self._check_request()
                started = perf_counter()
                trajectories = self.rollout_fn.evaluate_action(repeated, use_cuda_graph=False)
                self._batch_work["geometry_capture_host_seconds"] += perf_counter()-started
                self._batch_work["geometry_capture_calls"] += 1
                # END both gradient entry points consume these frozen records.
            changed = models.finish_capture()
        except BaseException:
            models.abort_capture()
            raise
        self._check_request(wait_for_device=True)
        # BEGIN the exact capture query owns reusable hard candidate evidence.
        return torch.as_tensor(changed, dtype=torch.bool, device=action.device), trajectories


    @torch.no_grad()
    def _retain_sampled_candidate(self, action, feasible, objective):
        """Retain one sampled-feasible candidate per seed by a fixed objective.

        All task and geometry hard evidence uses the common declaration.
        The objective omits changing multipliers and local collision models,
        so candidates from different blocks can be compared. Final curve
        validation still owns acceptance.
        """
        count, particles = self.config.num_problems, self.config.num_particles
        actions = action.detach().view(count, particles, self.action_horizon, self.action_dim)
        # BEGIN hard evidence gates every retained candidate independently.
        valid = feasible.view(count, particles) & torch.isfinite(actions).flatten(2).all(-1)
        values = objective.detach().view(count, particles)
        valid = valid & torch.isfinite(values)
        ranked = torch.where(valid, values, float("inf"))
        value, index = ranked.min(-1)
        selected = actions.gather(1, index[:, None, None, None].expand(-1, 1,
                                  self.action_horizon, self.action_dim)).squeeze(1)
        improved = valid.any(-1) & (~self._candidate_valid | (value < self._candidate_objective))
        self._candidate_action.copy_(torch.where(improved[:, None, None], selected, self._candidate_action))
        self._candidate_objective.copy_(torch.where(improved, value, self._candidate_objective))
        self._candidate_valid.logical_or_(improved)


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
        if not self._al_blocks_remaining or not self._candidate_tracking:
            return state
        count, particles = self.config.num_problems, self.config.num_particles
        base = state.exploration_action.detach().clone()
        repeated = base[:, None].expand(-1, particles, -1, -1).contiguous().view(count, particles, self.opt_dim)
        # BEGIN capture and hard evidence consume the same actual base query.
        # Hard residuals and fixed ranking exclude the local geometry cost,
        # so this query can capture the replacement model and supply both.
        # A changed objective still receives a new gradient below.
        self._check_request()
        with torch.no_grad():
            action = repeated.view(count * particles, self.action_horizon, self.action_dim)
            if self._local_geometry_models is None:
                changed = torch.zeros(count, dtype=torch.bool, device=base.device)
                started = perf_counter()
                trajectories = self.rollout_fn.evaluate_action(action, use_cuda_graph=False)
                self._batch_work["base_evidence_host_seconds"] += perf_counter()-started
                self._batch_work["base_evidence_calls"] += 1
            else:
                changed, trajectories = self._refresh_local_geometry(base)
            feasible, raw_signed, objective = (
                self.rollout_fn.constraint_manager.optimizer_evidence(trajectories))
            self._last_functional_signed = raw_signed
            self._retain_sampled_candidate(action, feasible, objective)
            signed = {name: value.view(count, particles)[:, 0]
                      for name, value in raw_signed.items()}
        self._check_request(wait_for_device=True)
        # END the completed query preserves every sampled-candidate decision.
        # BEGIN geometry and functional multipliers independently change cost.
        changed |= owner.update(signed)
        # END empty functional keys cannot suppress a real geometry refresh.
        self._check_request(wait_for_device=True)
        if not bool(changed.any()):
            return state
        # BEGIN this exact base layout also shares numeric geometry evidence.
        if self._local_geometry_models is None:
            costs, gradients = self._compute_cost_constraint_and_gradient(repeated)
        else:
            with self._local_geometry_models.repeated_base_evaluation():
                costs, gradients = self._compute_cost_constraint_and_gradient(repeated)
        # END each particle still backpropagates through its own FK and scalar.
        cost = costs.detach()[:, 0, 0]
        gradient = gradients.detach().view(count, particles, self.action_horizon, self.action_dim)[:, 0]
        if not bool((torch.isfinite(cost) & torch.isfinite(gradient).flatten(1).all(-1))[changed].all()):
            raise ValueError("Changed functional objective has nonfinite continuation cost or gradient")
        self._restart_changed_objective(state, base, cost, gradient, changed)
        return state


    @torch.no_grad()
    def _physical_step_direction(self, state, selected=None):
        """Use the full current gradient at first, ordinary and restarted steps."""
        base = state.exploration_action.detach()
        gradient = state.exploration_gradient.detach()
        limits = self.action_horizon_step_max.reshape(self.action_horizon, self.action_dim)
        scales = self._line_search_context.line_search_scale
        direction = torch.zeros_like(base)
        indices = range(len(base)) if selected is None else selected.nonzero().flatten().tolist()
        for index in indices:
            self._check_request()
            work = dict.fromkeys(self._direction_work_fields, 0)
            self._direction_calls[index] += 1
            try:
                value, report = self.domain.descent_step(base[index:index+1],
                    gradient[index:index+1], limits, scales,
                    check_request=self._check_request, work_report=work)
                self._check_request(wait_for_device=True)
            except BaseException as error:
                self._direction_reports[index] = {**work, "success": False,
                    "reason": "direction_evaluation_interrupted", "error_type": type(error).__name__}
                raise
            finally:
                # A library solve completed immediately before a deadline
                # still consumed work, even though no direction may be used.
                for name in self._direction_work_fields:
                    self._direction_work[index][name] += work[name]
            direction[index:index+1].copy_(value)
            self._direction_reports[index] = report
        return direction

    @torch.no_grad()
    def _restart_changed_objective(self, state, base, cost, gradient, changed):
        """Re-score changed objective rows and rebuild their physical direction."""
        def copy_rows(target, value):
            mask = changed.reshape((-1,)+(1,)*(target.ndim-1))
            target.copy_(torch.where(mask, value.reshape_as(target), target))
        for name in ("action", "exploration_action", "best_action"):
            copy_rows(getattr(state, name), base)
        for name in ("cost", "exploration_cost", "best_cost"):
            copy_rows(getattr(state, name), cost)
        for name in ("gradient", "exploration_gradient"):
            copy_rows(getattr(state, name), gradient)
        copy_rows(state.best_iteration, state.current_iteration)
        state.converged[changed] = 0
        copy_rows(self._best.cost, cost)
        copy_rows(self._best.action, base)
        copy_rows(self._best.iteration, state.current_iteration)
        self._best.converged[changed] = 0
        self._best.previous_step_direction[changed] = 0
        direction = self._physical_step_direction(state, changed)
        copy_rows(state.step_direction, direction)
        self._objective_updates.add_(changed.to(torch.int64))

    def _opt_step(self, state):
        self._check_request()
        before, cost = state.exploration_action.detach().clone(), state.exploration_cost.detach().clone()
        result = super()._opt_step(state)
        self._check_request(wait_for_device=True)
        displacement = (result.exploration_action-before).abs().flatten(1).amax(-1)
        improved = (displacement > 0) & (result.exploration_cost < cost)
        self._iterations.add_(1)
        self._effective_updates.add_(improved.to(torch.int64))
        self._last_displacement.copy_(displacement.double())
        self._line_evaluations.add_(self._line_search_strategy.evaluations)
        self._line_backtracks.add_(self._line_search_strategy.backtracks)
        return result

    def _create_initial_iteration_state(self, action):
        self.domain.validate(action)
        state = super()._create_initial_iteration_state(action)
        # The native zero-filled best action is not necessarily in this domain.
        state.best_action = action.clone()
        return state

    _progress_buffers = ("_iterations", "_effective_updates", "_last_displacement",
                         "_objective_updates", "_line_evaluations", "_line_backtracks")
    _candidate_buffers = ("_candidate_action", "_candidate_objective", "_candidate_valid")
    # BEGIN host solver accounting records actual calls, not maximum joint rounds.
    _direction_work_fields = ("qp_calls", "lp_calls", "extrema_seconds", "assembly_seconds",
                              "qp_solve_seconds", "lp_solve_seconds", "host_transfer_seconds")
    # END completed direction reports preserve their original numerical scope.

    def _reset_progress(self):
        for name in self._progress_buffers:
            getattr(self, name).zero_()
        self._direction_reports = [{} for _ in range(self.config.num_problems)]
        self._direction_calls = [0]*self.config.num_problems
        self._direction_work = [dict.fromkeys(self._direction_work_fields, 0)
                                for _ in range(self.config.num_problems)]
        # BEGIN batch calls share work across seeds and must be counted once.
        self._batch_work = dict.fromkeys(("cost_gradient_calls", "geometry_capture_calls",
            "forward_host_seconds", "backward_host_seconds", "geometry_capture_host_seconds",
            "base_evidence_calls", "base_evidence_host_seconds", "device_wait_seconds"), 0)
        # END host call times and explicit stream waits are disjoint intervals.

    def update_num_problems(self, num_problems):
        super().update_num_problems(num_problems)
        self._line_search_strategy = PhysicalLineSearch(self.domain)
        self._initial_line_search_strategy = PhysicalLineSearch(self.domain)
        for name in self._progress_buffers:
            dtype = torch.float64 if name == "_last_displacement" else torch.int64
            setattr(self, name, torch.zeros(num_problems, device=self.device_cfg.device, dtype=dtype))
        self._reset_progress()
        self._candidate_action = torch.empty((num_problems, self.action_horizon, self.action_dim),
            device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        self._candidate_objective = torch.full((num_problems,), float("inf"),
            device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        self._candidate_valid = torch.zeros(num_problems, device=self.device_cfg.device, dtype=torch.bool)

    def _prepare_solve(self, action):
        action = action.view(self.config.num_problems, self.action_horizon, self.action_dim)
        self.domain.validate(action)
        for context in (self._line_search_context, self._initial_line_search_context):
            self.domain.validate_scales(context.line_search_scale)

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        self._prepare_solve(action)
        self._reset_progress()
        self._begin_functional_solve(action)
        return super().reinitialize(action, mask=mask,
            clear_optimizer_state=clear_optimizer_state, reset_num_iters=reset_num_iters)

    def optimize(self, seed_action):
        # Prepared initial values and their multipliers belong to one solve;
        # native optimize consumes its cached reinitialize state exactly once.
        if self._iteration_state is None:
            self._prepare_solve(seed_action)
            self._reset_progress()
            self._best.clear()
            self._begin_functional_solve(seed_action)
        self._al_blocks_remaining = self.outer_iters
        try:
            result = super().optimize(seed_action)
            if self._candidate_tracking:
                result = torch.where(self._candidate_valid[:, None, None], self._candidate_action, result)
            return result
        finally:
            self._al_blocks_remaining = 0

    def _dispatch(self, method_name, *args):
        self._check_request()
        result = super()._dispatch(method_name, *args)
        return self._finish_functional_block(result) if method_name == "_opt_iters" else result

    @property
    def solve_diagnostics(self):
        """Per-seed direction work and shared batch evaluation work.

        ``shared_batch_work`` repeats the same optimizer-wide counters in
        each seed record; it is counted once per optimizer, not summed over
        seeds. Host call durations include their internal synchronous work.
        Explicit stream waiting is measured separately after those calls.
        """
        values = {name: getattr(self, name).detach().cpu().tolist() for name in self._progress_buffers}
        reports = [{"seed_index": index, "iterations": values["_iterations"][index],
            "effective_updates": values["_effective_updates"][index],
            "last_maximum_displacement": values["_last_displacement"][index],
            "objective_updates": values["_objective_updates"][index],
            "line_search_evaluations": values["_line_evaluations"][index],
            "line_search_backtracks": values["_line_backtracks"][index],
            "direction_calls": self._direction_calls[index], **self._direction_work[index],
            "shared_batch_work": dict(self._batch_work),
            "last_direction": self._direction_reports[index]}
            for index in range(self.config.num_problems)]
        if self._candidate_tracking:
            duals = self._functional_constraints.snapshot()["seeds"]
            valid = self._candidate_valid.detach().cpu().tolist()
            objective = self._candidate_objective.detach().cpu().tolist()
            for index, report in enumerate(reports):
                report.update(functional_continuation=duals[index],
                    sampled_feasible_candidate=valid[index],
                    retained_candidate_objective=objective[index] if valid[index] else None)
        return reports

    @property
    def extra_workspace_bytes(self):
        tensors = [getattr(self, name) for name in self._progress_buffers+self._candidate_buffers]
        if self._functional_constraints is not None:
            owner = self._functional_constraints
            tensors.extend(value for value in (owner._multiplier_buffer, owner._penalty_buffer,
                owner._previous_residual, owner._updates, owner._penalty_capped) if value is not None)
        return sum(value.numel()*value.element_size() for value in tensors)

    def bind_domain(self, domain):
        self.domain = domain
        self._iteration_state = None
        self._reset_progress()
        self._line_search_strategy.domain = domain
        self._initial_line_search_strategy.domain = domain


class LayoutGoalManager(GoalManager):
    """Allow goalset padding only when all other solve-layout fields match."""

    def _get_padded_goalset_for_links(self, solve_state, current_solve_state,
                                    current_goal_buffer, links_goal_pose):
        if replace(solve_state, num_goalset=current_solve_state.num_goalset) != current_solve_state:
            return None
        return super()._get_padded_goalset_for_links(
            solve_state, current_solve_state, current_goal_buffer, links_goal_pose)


def bind_solver_components(solver, domain):
    """Replace native stages with the shared core before their first solve.

    SolverCore and MultiStageOptimizer reference the same stage list. Updating
    its entries preserves native rollout, goal and seed ownership while dropping
    unused quasi-Newton buffers and callbacks. Existing bindings only replace
    the current request's mathematical domain.
    """
    for index, optimizer in enumerate(solver.core.optimizers):
        if isinstance(optimizer, PhysicalGradientCore):
            optimizer.bind_domain(domain)
        else:
            solver.core.optimizers[index] = PhysicalGradientCore(optimizer.config,
                optimizer.get_all_rollout_instances(), domain=domain)
    manager = solver.core.goal_registry_manager
    if not isinstance(manager, LayoutGoalManager):
        solver.core.goal_registry_manager = LayoutGoalManager(solver.core.device_cfg)
# END all extensions are local to the FUNCTO-owned solver instances.
