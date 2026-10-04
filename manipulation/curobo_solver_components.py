"""FUNCTO-owned components for the native planning solvers.

cuRobo still owns L-BFGS, rollout evaluation and goal-buffer allocation. This
module composes its existing internal components at the FUNCTO solver boundary:
physical joint domains constrain every evaluated search point, and a goal buffer
is reused only when its problem/seed layout is unchanged. No cuRobo class,
factory registry or module function is modified process-wide.

These extensions deliberately depend on cuRobo's internal component interfaces.
They support the approximate-Wolfe L-BFGS configuration used by this planner,
with CUDA graphs disabled because domain validation synchronizes with Python.
"""

# BEGIN instance-local native solver composition owned entirely by FUNCTO.
from dataclasses import replace

from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.gradient.lbfgs import LBFGSOpt
from curobo._src.optim.gradient.line_search_strategy import (
    ApproxWolfeLineSearchStrategy,
    LineSearchType,
)
from curobo._src.solver.manager_goal import GoalManager

from .physical_joint_domain import JointPositionDomain


class PhysicalLineSearch(ApproxWolfeLineSearchStrategy):
    """Keep native Wolfe selection, restricting its points before any rollout."""

    def __init__(self, domain: JointPositionDomain):
        super().__init__()
        self.domain = domain

    def _prepare_search_points(self, x, step_direction, context):
        # Native preparation checks shapes and applies step scaling/terminal
        # fixing. Its temporary points are never evaluated. Retain its adjusted
        # direction, shorten it against the physical domain, then form the
        # actual points and matching direction consumed by Wolfe selection.
        _, direction = super()._prepare_search_points(x, step_direction, context)
        base = x.detach()
        direction = self.domain.limit_step(
            base, direction.reshape_as(base), context.line_search_scale)
        points = self.jit_get_x_set(direction, base, context.line_search_scale)
        return (
            points.view(x.shape[0], context.n_linesearch, context.opt_dim)
                  .detach().requires_grad_(True),
            direction.view(x.shape[0], 1, context.opt_dim),
        )


class PhysicalGradientCore(GradientOptCore):
    """Native gradient lifecycle with a legal initial best state and trials."""

    def __init__(self, *args, domain: JointPositionDomain, **kwargs):
        self.domain = domain
        super().__init__(*args, **kwargs)

    def _create_initial_iteration_state(self, action):
        self.domain.validate(action)
        state = super()._create_initial_iteration_state(action)
        # A zero-filled best action can violate the mechanical limits or the
        # fixed cubic entrance even if no evaluated trial is selected as best.
        state.best_action = action.clone()
        return state

    def update_num_problems(self, num_problems):
        if self._line_search_context is None:
            # Install both strategies before native resizing. The initialization
            # rollout and ordinary iterations must obey exactly the same domain.
            self._line_search_context = self._create_line_search_context()
            self._initial_line_search_context = self._create_line_search_context(
                compute_costs_and_gradients=self._compute_cost_constraint_and_gradient_initial)
            self._line_search_strategy = PhysicalLineSearch(self.domain)
            self._initial_line_search_strategy = PhysicalLineSearch(self.domain)
        super().update_num_problems(num_problems)

    def bind_domain(self, domain: JointPositionDomain):
        """Rebind after FUNCTO rebuilds validation workspaces on the same clock."""
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


def bind_solver_components(solver, domain: JointPositionDomain) -> None:
    """Compose only a FUNCTO-owned solver, before its first solve.

    Reuse the existing native rollouts and L-BFGS history owner. Assign the new
    core before resizing so native L-BFGS callbacks access that core's dimensions.
    Repeated binding updates the physical domain without discarding goal state.
    """
    if solver.core.config.use_cuda_graph:
        raise ValueError("FUNCTO physical domains require CUDA graphs to be disabled")
    for optimizer in solver.core.optimizers:
        if (type(optimizer) is not LBFGSOpt
                or optimizer.config.line_search_type != LineSearchType.APPROX_WOLFE
                or optimizer.use_cuda_graph):
            raise ValueError("FUNCTO physical domains require approximate-Wolfe L-BFGS without CUDA graphs")
        if isinstance(optimizer._core, PhysicalGradientCore):
            optimizer._core.bind_domain(domain)
            continue
        if optimizer._core._iteration_state is not None:
            raise ValueError("Bind FUNCTO optimizer components before native solving")
        core = PhysicalGradientCore(
            optimizer.config, optimizer.get_all_rollout_instances(),
            step_direction_fn=optimizer._get_step_direction_impl,
            on_reinitialize=optimizer._on_reinitialize,
            on_initial_state=optimizer._on_initial_state,
            on_resize=optimizer._on_resize,
            on_shift=optimizer._on_shift,
            domain=domain, use_cuda_graph=False,
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
