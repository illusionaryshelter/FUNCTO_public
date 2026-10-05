"""Exercise FUNCTO composition using native CPU L-BFGS and goal allocation."""

# BEGIN regression coverage for the behavior moved out of third-party source.
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.types.device_cfg import DeviceCfg
from curobo.types import GoalToolPose, JointState

from manipulation.curobo_solver_components import (
    LayoutGoalManager, PhysicalGradientCore, bind_solver_components,
)
from manipulation.physical_joint_domain import (
    CubicJointDomain, JointDomainInitializationError, JointPositionDomain,
)


class AnalyticRollout:
    """A differentiable objective beyond the limits; inspect actual evaluations."""

    def __init__(self, domain, horizon, offset=0.):
        self.domain, self.action_horizon = domain, horizon
        self.action_dim = domain.lower.numel()
        self.action_bound_lows = domain.lower.float()
        self.action_bound_highs = domain.upper.float()
        self.evaluations = []
        self.offset = offset

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def evaluate_action(self, action, use_cuda_graph=False):
        # This observes tensors at the rollout boundary, independently of
        # which search strategy produced them. Violations abort before cost.
        self.domain.validate(action)
        self.evaluations.append(action.detach().clone())
        costs = ((action-4.)**2).sum(dim=(-1, -2))+self.offset
        return SimpleNamespace(costs_and_constraints=SimpleNamespace(
            get_sum_cost_and_constraint=lambda sum_horizon: costs))


class NativeComponentTest(unittest.TestCase):
    device = DeviceCfg("cpu")

    def make_solver(self, domain, horizon, offset=0.):
        rollouts = [AnalyticRollout(domain, horizon, offset) for _ in range(2)]
        optimizer = LBFGSOpt(LBFGSOptCfg(
            device_cfg=self.device, num_iters=4, inner_iters=2, history=2,
            line_search_scale=[0., .2, 1.], step_scale=.98,
            use_cuda_kernel_step_direction=False, use_cuda_kernel_line_search=False,
        ), rollouts)
        solver = SimpleNamespace(core=SimpleNamespace(
            device_cfg=self.device, config=SimpleNamespace(use_cuda_graph=False),
            optimizers=[optimizer], goal_registry_manager=GoalManager(self.device)))
        return solver, optimizer, rollouts

    def test_both_rollouts_obey_domains_through_native_resize_and_reinitialize(self):
        lower, upper = torch.tensor([.1, -.8]), torch.tensor([1., .9])
        domains = [(JointPositionDomain(lower, upper), 1),
            (CubicJointDomain(lower, upper, torch.tensor([.4, .2]),
                             torch.tensor([.05, -.05]), .2, 8), 8)]
        for domain, horizon in domains:
            with self.subTest(horizon=horizon):
                solver, optimizer, rollouts = self.make_solver(domain, horizon)
                bind_solver_components(solver, domain)
                with patch("curobo._src.runtime.cuda_event_timers", False):
                    for count in (2, 1, 2):
                        optimizer.update_num_problems(count)
                        seeds = torch.tensor([.4, .2]).repeat(count, horizon, 1)
                        optimizer.reinitialize(seeds)
                        result = optimizer.optimize(seeds)
                        domain.validate(result)
                        self.assertGreater(float((result-seeds).abs().sum()), 0.)
                        self.assertEqual(tuple(result.shape), (count, horizon, 2))
                self.assertTrue(all(len(rollout.evaluations) >= 3 for rollout in rollouts))

    def test_initial_best_is_legal_even_when_native_cost_sentinel_is_smaller(self):
        domain = JointPositionDomain(torch.tensor([.1, .1]), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1, offset=1e20)
        bind_solver_components(solver, domain)
        seed = torch.full((1, 1, 2), .4)
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            result = optimizer.optimize(seed)
        torch.testing.assert_close(result, seed)

    def test_illegal_seed_never_reaches_either_rollout_and_binding_is_local(self):
        domain = JointPositionDomain(torch.tensor([.1, .1]), torch.ones(2))
        owned, optimizer, rollouts = self.make_solver(domain, 1)
        ordinary, native, _ = self.make_solver(domain, 1)
        bind_solver_components(owned, domain)
        self.assertIsInstance(optimizer._core, PhysicalGradientCore)
        self.assertIs(type(native._core), GradientOptCore)
        self.assertIs(type(ordinary.core.goal_registry_manager), GoalManager)
        with self.assertRaises(JointDomainInitializationError):
            optimizer.reinitialize(torch.zeros((1, 1, 2)))
        self.assertTrue(all(not rollout.evaluations for rollout in rollouts))
        manager = owned.core.goal_registry_manager
        bind_solver_components(owned, domain)
        self.assertIs(owned.core.goal_registry_manager, manager)

    # BEGIN actual displacement, per-row history and recovery acceptance contracts.
    def test_zero_displacement_preserves_history_and_references_current_gradient(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        core, history = optimizer._core, optimizer._qn
        state = core._create_initial_iteration_state(torch.full((1, 1, 2), .4))
        state.exploration_gradient.fill_(2.)
        history.set_reference(state.exploration_action.reshape_as(history.x_0),
                              state.exploration_gradient.reshape_as(history.grad_0))
        history.s.fill_(.5)
        history.y.fill_(1.)
        history.rho.fill_(1.)
        before = {name: getattr(history, name).clone() for name in ("s", "y", "rho", "alpha")}
        direction = core._guarded_step_direction(state)
        for name, value in before.items():
            torch.testing.assert_close(getattr(history, name), value, rtol=0, atol=0)
        torch.testing.assert_close(direction, -optimizer.config.initial_step_scale*state.exploration_gradient)
        torch.testing.assert_close(history.x_0, state.exploration_action.reshape_as(history.x_0))
        torch.testing.assert_close(history.grad_0, state.exploration_gradient.reshape_as(history.grad_0))
        self.assertEqual(core.solve_diagnostics[0]["history_skips"], 1)

    def test_negative_curvature_resets_only_affected_row_and_rebinds_reference(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer.update_num_problems(2)
        core, history = optimizer._core, optimizer._qn
        state = core._create_initial_iteration_state(torch.full((2, 1, 2), .4))
        state.exploration_gradient.copy_(torch.tensor([[[1., 1.]], [[3., 3.]]]))
        history.x_0.fill_(.2)
        history.grad_0.fill_(2.)
        direction = core._guarded_step_direction(state)
        self.assertTrue(torch.equal(history.s[:, 0], torch.zeros_like(history.s[:, 0])))
        self.assertTrue(torch.any(history.s[:, 1] != 0))
        torch.testing.assert_close(direction[0], -optimizer.config.initial_step_scale*state.exploration_gradient[0])
        torch.testing.assert_close(history.x_0, state.exploration_action.reshape_as(history.x_0))
        torch.testing.assert_close(history.grad_0, state.exploration_gradient.reshape_as(history.grad_0))
        self.assertEqual([row["history_resets"] for row in core.solve_diagnostics], [1, 0])

    def test_retired_seed_keeps_history_references_when_a_peer_is_evaluated(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer.update_num_problems(2)
        core, history = optimizer._core, optimizer._qn
        state = core._create_initial_iteration_state(torch.full((2, 1, 2), .4))
        state.exploration_gradient.fill_(3.)
        history.x_0.fill_(.2)
        history.grad_0.fill_(2.)
        core._exhausted[0] = True
        core._last_curvature[0] = .25
        before = {name: getattr(history, name).clone()
                  for name in ("s", "y", "rho", "alpha", "x_0", "grad_0")}
        direction = core._guarded_step_direction(state)
        for name, value in before.items():
            actual = getattr(history, name)
            index = (slice(None), 0) if actual.ndim == 4 else 0
            torch.testing.assert_close(actual[index], value[index], rtol=0, atol=0)
        torch.testing.assert_close(direction[0], torch.zeros_like(direction[0]))
        self.assertEqual(core.solve_diagnostics[0]["last_curvature"], .25)

    def test_legal_recovery_without_armijo_descent_is_rejected_and_seed_stops(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        seed = torch.ones((1, 1, 2))
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
        core = optimizer._core
        state = core._iteration_state
        core._stalled_steps.fill_(core.config.inner_iters)
        # A move into the position domain increases this objective. A legal
        # QP proposal alone must not be accepted as optimization progress.
        with patch.object(domain, "recovery_step", return_value=(-torch.full_like(seed, .1),
                {"success": True, "reason": "test_inward_but_ascending"})):
            core._recover_stalled(state)
        torch.testing.assert_close(state.exploration_action, seed)
        torch.testing.assert_close(optimizer._qn.x_0, seed.reshape_as(optimizer._qn.x_0))
        self.assertTrue(core.solve_diagnostics[0]["exhausted"])
        self.assertEqual(core.solve_diagnostics[0]["recovery_accepted"], 0)
        self.assertTrue(core._all_exhausted)

    def test_all_stalled_seeds_stop_before_fixed_iteration_budget(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer.config.num_iters = 100
        seed = torch.ones((1, 1, 2))
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            result = optimizer.optimize(seed)
        torch.testing.assert_close(result, seed)
        report = optimizer._core.solve_diagnostics[0]
        self.assertEqual(report["iterations"], optimizer.config.inner_iters)
        self.assertEqual(report["recovery_attempts"], 1)
        self.assertTrue(report["exhausted"])

    def test_recovery_recomputes_main_gradient_and_backtracks_oversized_descent(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        seed = torch.zeros((1, 1, 2))

        def objective(target, weight):
            def evaluate(action, use_cuda_graph=False):
                domain.validate(action)
                costs = weight*((action-target)**2).sum(dim=(-1, -2))
                return SimpleNamespace(costs_and_constraints=SimpleNamespace(
                    get_sum_cost_and_constraint=lambda sum_horizon: costs))
            return evaluate

        # Opposing initial/main gradients expose stale-reference use. The
        # main minimizer is so near that all original trial scales overshoot.
        with patch.object(rollouts[0], "evaluate_action", new=objective(.01, 10000.)), \
                patch.object(rollouts[1], "evaluate_action", new=objective(-.1, 1.)), \
                patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            core = optimizer._core
            state = core._iteration_state
            self.assertTrue(bool((state.exploration_gradient > 0).all()))
            core._stalled_steps.fill_(core.config.inner_iters)
            with patch.object(domain, "recovery_step", wraps=domain.recovery_step) as recovery:
                core._recover_stalled(state)
            self.assertTrue(bool((recovery.call_args.args[1] < 0).all()))
        report = core.solve_diagnostics[0]["recoveries"][0]
        self.assertTrue(report["accepted"])
        self.assertGreater(report["line_search_evaluations"], 1)
        self.assertLess(report["cost_after"], report["cost_before"])
        self.assertTrue(bool((state.exploration_action > 0).all()))
        torch.testing.assert_close(state.best_action, state.exploration_action)
        torch.testing.assert_close(state.best_cost, state.exploration_cost)
        torch.testing.assert_close(optimizer._qn.grad_0,
                                  state.exploration_gradient.reshape_as(optimizer._qn.grad_0))

    def test_boundary_recovery_moves_free_coordinate_and_keeps_retired_peer_fixed(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer.update_num_problems(2)
        optimizer.config.num_iters = 12
        seed = torch.tensor([[[1., 1.]], [[1., .2]]])
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            result = optimizer.optimize(seed)
        domain.validate(result)
        torch.testing.assert_close(result[0], seed[0])
        self.assertGreater(float(result[1, 0, 1]), float(seed[1, 0, 1]))
        reports = optimizer._core.solve_diagnostics
        self.assertEqual(reports[0]["iterations"], optimizer.config.inner_iters)
        self.assertGreater(reports[1]["iterations"], reports[0]["iterations"])
        self.assertGreater(reports[1]["recovery_accepted"], 0)
    # END recovered motion must decrease the same objective and preserve peers.

    # BEGIN exercise capture lifecycle and fail-closed behavior on CPU tensors.
    def test_deferred_bad_iteration_only_constructs_legal_search_points(self):
        domain = JointPositionDomain(torch.tensor([.1, .1]), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        core = optimizer._core
        seed = torch.full((1, 1, 2), .4)
        core._prepare_solve(seed)
        checks = core._device_checks
        with checks.scope():
            points, _ = core._line_search_strategy._prepare_search_points(
                torch.zeros_like(seed), torch.ones_like(seed), core._line_search_context)
        domain.validate(points.reshape(1, 3, 1, 2))
        with self.assertRaises(JointDomainInitializationError):
            checks.raise_if_failed()

    def test_capture_warmup_restores_history_and_clears_only_warmup_faults(self):
        domain = JointPositionDomain(torch.tensor([.1, .1]), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        core = optimizer._core
        records = core._capture_state()

        class MutatingExecutor:
            is_initialized = False

            def warmup(self, *args):
                self._graph_input = tuple(value.clone() for value in args)
                for _, _, tensor, _ in records:
                    tensor.fill_(1)
                domain.validate(torch.zeros((1, 1, 2)))
                self.is_initialized = True

            def __call__(self, value):
                for _, _, tensor, saved in records:
                    assert torch.equal(tensor, saved), "Warmup changed native closure state"
                return value

            def reset(self):
                self.is_initialized = False

        executor = MutatingExecutor()
        core._executors["_opt_step"] = executor
        core.use_cuda_graph = True
        value = core._create_initial_iteration_state(torch.full((1, 1, 2), .4))
        with patch("torch.cuda.memory_allocated", return_value=0):
            result = core._dispatch("_opt_iters", value)
        self.assertTrue(torch.equal(result.action, value.action))
        # The native executor's action-pointer shortcut misses this changed
        # field. The owned dispatch must copy it even though action aliases.
        changed = replace(result, step_direction=torch.full_like(result.step_direction, .25))
        replayed = core._dispatch("_opt_iters", changed)
        self.assertTrue(torch.equal(replayed.step_direction, changed.step_direction))
        self.assertTrue(executor.is_initialized)
        core.bind_domain(domain)
        self.assertFalse(executor.is_initialized)

    def test_step_replays_keep_faults_until_one_original_block_check(self):
        domain = JointPositionDomain(torch.tensor([.1, .1]), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        core = optimizer._core

        class FaultingExecutor:
            is_initialized = False
            calls = 0

            def warmup(self, value):
                self._graph_input = (value.clone(),)
                self.is_initialized = True

            def __call__(self, value):
                self.calls += 1
                # The later legal replay must not clear an earlier fault.
                domain.validate(torch.full_like(value.action, 0. if self.calls == 1 else .4))
                return value

            def reset(self):
                self.is_initialized = False

        executor = FaultingExecutor()
        core._executors["_opt_step"] = executor
        core.use_cuda_graph = True
        value = core._create_initial_iteration_state(torch.full((1, 1, 2), .4))
        with patch("torch.cuda.memory_allocated", return_value=0), \
                patch.object(core._device_checks, "raise_if_failed",
                             wraps=core._device_checks.raise_if_failed) as host_check:
            with self.assertRaises(JointDomainInitializationError):
                core._dispatch("_opt_iters", value)
            self.assertEqual(executor.calls, core.config.inner_iters)
            self.assertEqual(host_check.call_count, 1)
            core._dispatch("_opt_iters", value)
            self.assertEqual(executor.calls, 2*core.config.inner_iters)
            self.assertEqual(host_check.call_count, 2)

    def test_explicit_graph_request_rejects_cpu_without_changing_backend(self):
        domain = JointPositionDomain(torch.tensor([.1, .1]), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        original = optimizer._core
        with self.assertRaisesRegex(ValueError, "native CUDA L-BFGS"):
            bind_solver_components(solver, domain, use_optimizer_graph=True)
        self.assertIs(optimizer._core, original)

    def test_native_shift_keeps_buffers_and_shape_reset_invalidates_graph(self):
        domain = JointPositionDomain(torch.tensor([.1, .1]), torch.ones(2))
        solver, optimizer, rollouts = self.make_solver(domain, 4)
        bind_solver_components(solver, domain)
        core = optimizer._core
        cached = core._create_initial_iteration_state(torch.full((1, 4, 2), .4))
        core._iteration_state = cached
        history = optimizer._qn.s

        class CapturedExecutor:
            is_initialized = True

            def reset(self):
                self.is_initialized = False

        executor = CapturedExecutor()
        core._executors["_opt_step"] = executor
        core.use_cuda_graph = True
        optimizer.shift(0)
        self.assertTrue(executor.is_initialized)
        optimizer.shift(1)
        # The installed shift helper copies into its inputs before returning
        # the same tensors; its tuple assignment does not replace storage.
        self.assertTrue(executor.is_initialized)
        self.assertIs(optimizer._qn.s, history)
        self.assertIs(core._iteration_state, cached)

        resets = []
        for index, rollout in enumerate(rollouts):
            rollout.reset_shape = lambda index=index: resets.append(index)
        executor.is_initialized = True
        core.reset_shape()
        self.assertFalse(executor.is_initialized)
        self.assertEqual(resets, list(range(len(rollouts))))
        self.assertIs(core._iteration_state, cached)
    # END production CUDA capture itself is checked by a separate bounded probe.

    def goal(self, count, value):
        return GoalToolPose(tool_frames=["tool"],
            position=torch.full((1, 1, 1, count, 3), float(value)),
            quaternion=torch.tensor([1., 0., 0., 0.]).repeat(1, 1, 1, count, 1))

    def state(self, seeds, goals=1):
        return SolveState(SolveMode.SINGLE, batch_size=1, num_envs=1,
            num_seeds=seeds, num_trajopt_seeds=seeds, num_goalset=goals, tool_frames=["tool"])

    def test_goal_seed_indices_reallocate_for_2_1_2_and_reuse_same_layout(self):
        manager = LayoutGoalManager(self.device)
        current = JointState.from_position(torch.zeros((1, 2)))
        previous = None
        for seeds in (2, 1, 2):
            state = self.state(seeds)
            buffer, changed = manager.update_goal_buffer(
                state, self.goal(1, seeds), current_js=current)
            self.assertTrue(changed)
            self.assertIsNot(buffer, previous)
            self.assertEqual(manager.solve_state, state)
            self.assertEqual(buffer.idxs_link_pose.numel(), seeds)
            self.assertEqual(buffer.idxs_current_js.numel(), seeds)
            reused, changed = manager.update_goal_buffer(state, self.goal(1, seeds+1))
            self.assertFalse(changed)
            self.assertIs(reused, buffer)
            self.assertTrue(bool((reused.link_goal_poses.position == seeds+1).all()))
            previous = buffer

    def test_smaller_goalset_still_uses_native_padding(self):
        manager = LayoutGoalManager(self.device)
        buffer, _ = manager.update_goal_buffer(self.state(2, 3), self.goal(3, 1.))
        reused, changed = manager.update_goal_buffer(self.state(2, 1), self.goal(1, 2.))
        self.assertFalse(changed)
        self.assertIs(reused, buffer)
        self.assertEqual(reused.link_goal_poses.position.shape[-2], 3)
        self.assertTrue(bool((reused.link_goal_poses.position == 2.).all()))


if __name__ == "__main__":
    unittest.main()
# END no physical robot, scene substitution, or CUDA allocation is needed here.
