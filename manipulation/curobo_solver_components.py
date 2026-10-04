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
from dataclasses import replace

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
        super().__init__(*args, **kwargs)
        # BEGIN scope graph checks independently of the CPU metrics rollout.
        self._device_checks = DeviceChecks(self.device_cfg.device)
        self.graph_capture_bytes = 0
        self.graph_snapshot_bytes = 0
        # END graph allocation is measured separately from known tensor storage.

    def finish_init(self):
        # BEGIN retain one native step, not one copy per configured iteration.
        # The installed core's unmodified _opt_iters loop still owns the exact
        # iteration count and recording order. Capturing that whole Python
        # loop creates a separate CUDA node sequence for every iteration and
        # excessive host metadata for real scenes. Reuse one native step's
        # executor instead; no cost, gradient or L-BFGS operation is rewritten.
        if self._deferred_cuda_graph:
            self._executors["_opt_step"] = create_graph_executor(
                capture_fn=super()._opt_step, device=self.device_cfg.device, use_cuda_graph=True)
            self._executors["_prepare_initial_iteration_state"] = create_graph_executor(
                capture_fn=self._prepare_initial_iteration_state,
                device=self.device_cfg.device, use_cuda_graph=True)
        # END metrics and the original Python iteration loop stay outside capture.

    def _opt_step(self, iteration_state):
        if not self.use_cuda_graph:
            return super()._opt_step(iteration_state)
        return self._replay_graph("_opt_step", iteration_state)

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
        return super().reinitialize(action, mask=mask,
            clear_optimizer_state=clear_optimizer_state, reset_num_iters=reset_num_iters)

    def optimize(self, seed_action):
        self._prepare_solve(seed_action)
        return super().optimize(seed_action)

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
        if not self.use_cuda_graph:
            return super()._dispatch(method_name, *args)
        # One fault boundary surrounds the original inner-iteration block.
        # Its native loop calls our _opt_step and records each replayed state;
        # the scalar mask is neither reset nor read between those steps.
        self._device_checks.reset()
        with self._device_checks.scope():
            result = (self._replay_graph(method_name, *args)
                      if method_name in self._executors
                      else super()._dispatch(method_name, *args))
        self._device_checks.raise_if_failed()
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
        return safe_bytes+self._device_checks.storage_bytes+self.graph_snapshot_bytes
    # END no new optimizer kernel, fusion, reduction order or iteration is used.

    def bind_domain(self, domain: JointPositionDomain):
        """Rebind after FUNCTO rebuilds validation workspaces on the same clock."""
        # BEGIN a captured closure must not keep the previous bounds/prefix.
        self.reset_cuda_graph()
        # END the next solve validates and binds its own safe seed again.
        self.domain = domain
        self._iteration_state = None
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
        # END CPU solvers retain the same original native eager implementation.
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
