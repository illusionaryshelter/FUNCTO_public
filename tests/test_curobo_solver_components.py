"""Exercise FUNCTO composition using native CPU L-BFGS and goal allocation."""

# BEGIN regression coverage for the behavior moved out of third-party source.
from dataclasses import fields, replace
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
from manipulation.functional_augmented_lagrangian import FunctionalAugmentedLagrangian
from manipulation.local_geometry_model import LocalGeometryModels, LocalPairModel


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


# BEGIN real block records exercise both native rollout entry points on CPU.
class AnalyticGeometryRollout(AnalyticRollout):
    """A translating point against a unit sphere, with exact radial witnesses.

    The analytic query here supplies the test's actual base tangent. Production
    LocalPairModel and LocalGeometryModels own the scalar and frozen lifecycle;
    the unchanged native optimizer owns every cost and gradient callback.
    """

    def __init__(self, domain, owner, models):
        super().__init__(domain, 1)
        self.constraint_manager = self
        self.optimization_constraints = owner
        self.local_geometry_models = models
        self.used_records = []

    def evaluate_action(self, action, use_cuda_graph=False):
        self.domain.validate(action)
        self.evaluations.append(action.detach().clone())
        models = self.local_geometry_models
        values, record_ids = [], []
        quaternion = action.new_tensor([1., 0., 0., 0.])
        for row in range(len(action)):
            state = models.state(models.state_key(row, 0., .5, .5))
            if models.captures_row(row):
                radial = torch.tensor([float(action[row, 0, 0].detach()), .3, 0.], dtype=torch.float64)
                normal = tuple((radial/radial.norm()).tolist())
                state.pairs["sphere"] = LocalPairModel((0., 0., 0.), normal, normal, True, .1)
            model = state.pairs["sphere"]
            coordinate = action[row, 0, 0]
            point = torch.stack((coordinate, coordinate*0+.3, coordinate*0))
            values.append(model.value((point, quaternion), (torch.zeros_like(point), quaternion)))
            record_ids.append(id(model))
        if not models.capturing:
            self.used_records.append(tuple(record_ids))
        fixed = ((action-1.)**2).sum((-1, -2))
        costs = fixed+torch.stack(values)
        result = SimpleNamespace(costs_and_constraints=SimpleNamespace(
            get_sum_cost_and_constraint=lambda sum_horizon: costs))
        feasible = action[:, 0, 0].detach().square()+.3**2 >= 1.
        self.evidence = result, feasible, fixed.detach()
        return result

    def optimizer_evidence(self, result):
        original, feasible, fixed = self.evidence
        assert original is result
        return feasible, {}, fixed
# END hard feasibility remains separate from the sphere tangent objective.


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

    # BEGIN local geometry has its own lifecycle even with no functional keys.
    def make_geometry_solver(self, count=1):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        owner, models = FunctionalAugmentedLagrangian(()), LocalGeometryModels()
        rollouts = [AnalyticGeometryRollout(domain, owner, models) for _ in range(2)]
        optimizer = LBFGSOpt(LBFGSOptCfg(
            device_cfg=self.device, num_iters=4, inner_iters=2, history=2,
            line_search_scale=[0., .2, 1.], step_scale=.98,
            use_cuda_kernel_step_direction=False, use_cuda_kernel_line_search=False), rollouts)
        solver = SimpleNamespace(core=SimpleNamespace(
            device_cfg=self.device, config=SimpleNamespace(use_cuda_graph=False),
            optimizers=[optimizer], goal_registry_manager=GoalManager(self.device)))
        bind_solver_components(solver, domain)
        optimizer.update_num_problems(count)
        return optimizer, rollouts, owner, models

    def test_initial_and_regular_callbacks_share_frozen_models_across_particles(self):
        optimizer, rollouts, owner, models = self.make_geometry_solver(2)
        core = optimizer._core
        seed = torch.tensor([.25, .6]).view(2, 1, 1)
        core._begin_functional_solve(seed)
        self.assertEqual(owner.keys, ())
        # BEGIN the shared native layout requires exactly one capture query.
        self.assertEqual([len(rollout.evaluations) for rollout in rollouts], [1, 0])
        # END the initial callback consumes the ordinary rollout's records.
        trials = torch.tensor([.1, .3, .7, .2, .8, 1.2]).view(2, 3, 1)
        ordinary = core._compute_cost_constraint_and_gradient(trials)
        initial = core._compute_cost_constraint_and_gradient_initial(trials)
        for actual, expected in zip(ordinary, initial):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIs(rollouts[0].local_geometry_models, rollouts[1].local_geometry_models)
        ids = rollouts[0].used_records[-1]
        self.assertEqual(ids, rollouts[1].used_records[-1])
        self.assertEqual(len(set(ids[:3])), 1)
        self.assertEqual(len(set(ids[3:])), 1)
        self.assertNotEqual(ids[0], ids[3])
        self.assertFalse(models.capturing)

    def test_geometry_only_block_refresh_restarts_exactly_the_changed_seed(self):
        optimizer, rollouts, owner, _ = self.make_geometry_solver(2)
        core, history = optimizer._core, optimizer._qn
        seed = torch.tensor([.25, .6]).view(2, 1, 1)
        core._begin_functional_solve(seed)
        state = core._create_initial_iteration_state(seed)
        state.exploration_action[0].fill_(.5)
        state.current_iteration.fill_(2)
        core._iterations.fill_(2)
        for name in ("s", "y", "rho", "alpha"):
            getattr(history, name).fill_(3.)
        core._al_blocks_remaining = 2
        before_queries = len(rollouts[0].evaluations)
        with patch.object(core, "_restart_changed_objective", wraps=core._restart_changed_objective) as restart:
            core._finish_functional_block(state)
        # BEGIN one combined capture/evidence query, then the changed gradient.
        self.assertEqual(len(rollouts[0].evaluations) - before_queries, 2)
        # END original base evidence does not cause a third rollout.
        self.assertEqual(restart.call_count, 1)
        self.assertEqual(restart.call_args.args[-1].tolist(), [True, False])
        for name in ("s", "y", "rho", "alpha"):
            self.assertEqual(float(getattr(history, name)[:, 0].abs().sum()), 0.)
            self.assertTrue(bool((getattr(history, name)[:, 1] == 3.).all()))
        self.assertEqual(core._iterations.tolist(), [2, 2])
        self.assertEqual([row["outer_updates"] for row in owner.snapshot()["seeds"]], [0, 0])
        self.assertEqual(core._history_resets.tolist(), [1, 0])
        self.assertEqual(float(state.exploration_action[0]), .5)

    def test_unchanged_geometry_and_empty_functional_keys_keep_history(self):
        optimizer, _, _, _ = self.make_geometry_solver()
        core, history = optimizer._core, optimizer._qn
        seed = torch.full((1, 1, 1), .25)
        core._begin_functional_solve(seed)
        state = core._create_initial_iteration_state(seed)
        for name in ("s", "y", "rho", "alpha"):
            getattr(history, name).fill_(3.)
        before = {name: getattr(history, name).clone() for name in ("s", "y", "rho", "alpha")}
        core._al_blocks_remaining = 2
        with patch.object(core, "_restart_changed_objective", wraps=core._restart_changed_objective) as restart, \
                patch.object(core, "_compute_cost_constraint_and_gradient",
                             wraps=core._compute_cost_constraint_and_gradient) as gradients:
            self.assertIs(core._finish_functional_block(state), state)
        self.assertEqual(restart.call_count, 0)
        self.assertEqual(gradients.call_count, 0)
        for name, saved in before.items():
            torch.testing.assert_close(getattr(history, name), saved, rtol=0, atol=0)

    def test_interrupted_geometry_refresh_discards_only_its_pending_records(self):
        optimizer, rollouts, _, models = self.make_geometry_solver()
        core = optimizer._core
        seed = torch.full((1, 1, 1), .25)
        core._begin_functional_solve(seed)
        key = models.state_key(0, 0., .5, .5)
        original = models.state(key).pairs["sphere"]
        # BEGIN interrupt after the sole capture has populated pending records.
        evaluate = rollouts[0].evaluate_action
        def interrupted(*args, **kwargs):
            evaluate(*args, **kwargs)
            raise TimeoutError("capture deadline")
        with patch.object(rollouts[0], "evaluate_action", side_effect=interrupted):
            with self.assertRaisesRegex(TimeoutError, "capture deadline"):
                core._refresh_local_geometry(torch.full_like(seed, .7))
        # END the previous block survives a partially completed capture.
        self.assertFalse(models.capturing)
        self.assertIs(models.state(key).pairs["sphere"], original)
    # END refresh never changes hard evidence, native budgets or peer histories.

    # BEGIN block updates exercise the real CPU native optimizer, not a fake loop.
    # BEGIN the same fixture can place its fixed minimizer beyond a joint bound.
    def attach_functional_objective(self, rollouts, owner, residual, *, target=1.):
        for rollout in rollouts:
            rollout.constraint_manager = rollout
            rollout.optimization_constraints = owner
            rollout.objective_offset = 0.

            def evaluate(action, use_cuda_graph=False, rollout=rollout):
                rollout.domain.validate(action)
                fixed = ((action-target)**2).sum((-1, -2))+rollout.objective_offset
                signed = {"event": residual(action, owner.num_seeds)}
                costs = fixed+owner.terms(signed)["event"]
                result = SimpleNamespace(costs_and_constraints=SimpleNamespace(
                    get_sum_cost_and_constraint=lambda sum_horizon: costs))
                rollout.evidence = (result, signed, fixed)
                return result

            def evidence(result, rollout=rollout):
                source, signed, fixed = rollout.evidence
                self.assertIs(source, result)
                feasible = torch.isfinite(signed["event"]) & (signed["event"] <= 0)
                return feasible, {key: value.detach() for key, value in signed.items()}, fixed.detach()

            rollout.evaluate_action = evaluate
            rollout.optimizer_evidence = evidence
    # END all fixture variants retain the real production AL calculation.

    def test_original_100_steps_four_blocks_keep_seed_parameters_and_total_work(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner,
            lambda action, count: action.new_tensor([1., -1.]).repeat_interleave(action.shape[0]//count))
        bind_solver_components(solver, domain)
        optimizer.update_num_problems(2)
        optimizer.config.num_iters, optimizer.config.inner_iters = 100, 25
        seed = torch.ones((2, 1, 1))
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            result = optimizer.optimize(seed)
        self.assertEqual(tuple(result.shape), tuple(seed.shape))
        reports = optimizer._core.solve_diagnostics
        self.assertEqual([row["functional_continuation"]["outer_updates"] for row in reports], [3, 3])
        self.assertEqual([row["functional_continuation"]["penalty"] for row in reports], [200., 2.])
        self.assertEqual(reports[0]["iterations"], 100)
        self.assertEqual(reports[1]["iterations"], 25)
        self.assertEqual([row["sampled_feasible_candidate"] for row in reports], [False, True])
        self.assertEqual(reports[0]["recovery_attempts"], 4)
        self.assertEqual(reports[1]["recovery_attempts"], 1)
        self.assertEqual(optimizer.config.num_iters, 100)

    def test_changed_objective_clears_only_its_row_and_preserves_cumulative_counters(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer.update_num_problems(2)
        core, history = optimizer._core, optimizer._qn
        state = core._create_initial_iteration_state(torch.ones((2, 1, 1)))
        state.current_iteration.fill_(25)
        core._iterations.fill_(25)
        core._exhausted.fill_(True)
        core._stalled_steps.fill_(25)
        core._recovery_reports = [[{"accepted": False}], [{"accepted": False}]]
        for name in ("s", "y", "rho", "alpha"):
            getattr(history, name).fill_(3.)
        peer = {field.name: getattr(state, field.name)[1].clone() for field in fields(state)
                if isinstance(getattr(state, field.name), torch.Tensor)}
        core._restart_changed_objective(state, torch.zeros((2, 1, 1)),
            torch.tensor([7., 8.]), torch.tensor([[[2.]], [[3.]]]), torch.tensor([True, False]))
        for name in ("s", "y", "rho", "alpha"):
            self.assertEqual(float(getattr(history, name)[:, 0].abs().sum()), 0.)
            self.assertTrue(bool((getattr(history, name)[:, 1] == 3.).all()))
        for name, value in peer.items():
            torch.testing.assert_close(getattr(state, name)[1], value)
        torch.testing.assert_close(core._iterations, torch.tensor([25, 25]))
        torch.testing.assert_close(state.current_iteration, torch.tensor([25, 25], dtype=state.current_iteration.dtype))
        self.assertEqual(core._exhausted.tolist(), [False, True])
        self.assertEqual(core._stalled_steps.tolist(), [0, 25])
        self.assertEqual([len(rows) for rows in core._recovery_reports], [1, 1])
        self.assertEqual(float(state.best_cost[0]), 7.)
        self.assertEqual(float(history.grad_0[0]), 2.)

    def test_feasible_particle_survives_lower_augmented_cost_and_resets_for_new_solve(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner, lambda action, count: action[:, 0, 0])
        bind_solver_components(solver, domain)
        core = optimizer._core
        core._begin_functional_solve(torch.zeros((1, 1, 1)))
        first = torch.tensor([[[-.5], [.5], [1.]]])
        costs, _ = core._compute_cost_constraint_and_gradient(first)
        self.assertLess(float(costs[0, 1].detach()), float(costs[0, 0].detach()))
        torch.testing.assert_close(core._candidate_action, torch.tensor([[[-.5]]]))
        core._compute_cost_constraint_and_gradient(torch.tensor([[[.1], [.2], [.3]]]))
        torch.testing.assert_close(core._candidate_action, torch.tensor([[[-.5]]]))
        owner.update({"event": torch.tensor([1.])})
        core._compute_cost_constraint_and_gradient(torch.tensor([[[-.25], [.2], [.3]]]))
        torch.testing.assert_close(core._candidate_action, torch.tensor([[[-.25]]]))
        # BEGIN solve reset and capture warmup preserve the graded slot allocation.
        pointers = {name: getattr(core, name).data_ptr() for name in core._candidate_buffers}
        snapshot = core._capture_state()
        core._begin_functional_solve(torch.ones((1, 1, 1)))
        self.assertFalse(bool(core._candidate_valid.any()))
        self.assertEqual(pointers, {name: getattr(core, name).data_ptr() for name in core._candidate_buffers})
        self.assertEqual(owner.snapshot()["seeds"][0]["outer_updates"], 0)
        core._restore_capture_state(snapshot)
        self.assertEqual(core._candidate_valid.tolist(), [True])
        torch.testing.assert_close(core._candidate_action, torch.tensor([[[-.25]]]))
        # END CUDA capture owns these exact addresses, not replacement tensors.

    # BEGIN a lower objective cannot replace a feasible candidate with a failure.
    def test_candidate_feasibility_precedes_objective_independently_for_each_seed(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer.update_num_problems(2)
        core = optimizer._core
        actions = torch.tensor([.1, .2, .3, .4, .5, .6]).view(6, 1, 1)
        core._retain_sampled_candidate(actions,
            torch.tensor([True, True, False, False, True, False]),
            torch.tensor([5., 3., 4., -100., 20., -200.]))
        self.assertEqual(core._candidate_valid.tolist(), [True, True])
        torch.testing.assert_close(core._candidate_action.flatten(), torch.tensor([.2, .5]))
        # Infeasible cheap candidates cannot overwrite either incumbent, and
        # a feasible but more expensive candidate does not replace seed zero.
        core._retain_sampled_candidate(-actions,
            torch.tensor([False, True, False, False, False, False]),
            torch.tensor([-100., 50., -200., -1000., -2000., -3000.]))
        self.assertEqual(core._candidate_valid.tolist(), [True, True])
        torch.testing.assert_close(core._candidate_action.flatten(), torch.tensor([.2, .5]))
        core._retain_sampled_candidate(actions, torch.ones(6, dtype=torch.bool),
            torch.tensor([6., 4., 2., 22., 21., 23.]))
        torch.testing.assert_close(core._candidate_action.flatten(), torch.tensor([.3, .5]))
        torch.testing.assert_close(core._candidate_objective, torch.tensor([2., 20.]))

    def test_candidate_retention_rejects_nonfinite_values_and_failed_hard_evidence(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer.update_num_problems(2)
        core = optimizer._core
        core._retain_sampled_candidate(
            torch.tensor([float("nan"), .2, .3, .4, .5, .6]).view(6, 1, 1),
            torch.tensor([True, True, False, True, True, False]),
            torch.tensor([0., float("inf"), -100., 2., float("nan"), -100.]))
        self.assertEqual(core._candidate_valid.tolist(), [False, True])
        torch.testing.assert_close(core._candidate_action[1], torch.tensor([[.4]]))
        self.assertTrue(bool(torch.isinf(core._candidate_objective[0])))

    def test_native_result_retains_common_sampled_feasibility(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner, lambda action, count: action[:, 0, 0])
        bind_solver_components(solver, domain)
        optimizer.config.num_iters = optimizer.config.inner_iters
        seed = torch.full((1, 1, 1), -.5)
        with patch("curobo._src.runtime.cuda_event_timers", False):
            result = optimizer.optimize(seed)
        self.assertEqual(tuple(result.shape), tuple(seed.shape))
        self.assertTrue(bool((result <= 0).all()))
        torch.testing.assert_close(result, optimizer._core._candidate_action)
        report = optimizer._core.solve_diagnostics[0]
        self.assertTrue(report["sampled_feasible_candidate"])
        self.assertIsNotNone(report["retained_candidate_objective"])
        self.assertEqual(report["iterations"], optimizer.config.num_iters)
        self.assertEqual(owner.snapshot()["seeds"][0]["outer_updates"], 0)
    # END final native confirmation remains outside every optimizer callback.

    def test_block_update_measures_actual_continuation_instead_of_last_trial(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner, lambda action, count: action[:, 0, 0])
        bind_solver_components(solver, domain)
        core = optimizer._core
        seed = torch.full((1, 1, 1), .25)
        core._begin_functional_solve(seed)
        state = core._create_initial_iteration_state(seed)
        core._compute_cost_constraint_and_gradient(torch.tensor([[[.9], [1.], [1.1]]]))
        core._al_blocks_remaining = 2
        core._finish_functional_block(state)
        self.assertEqual(owner.snapshot()["seeds"][0]["multipliers"]["event"], .5)
        torch.testing.assert_close(state.exploration_action, seed)
        torch.testing.assert_close(state.exploration_cost, torch.tensor([.8125]))
        torch.testing.assert_close(state.exploration_gradient, torch.tensor([[[-.5]]]))
        core._finish_functional_block(state)
        self.assertEqual(owner.snapshot()["seeds"][0]["outer_updates"], 1)

    def test_native_result_uses_retained_sampled_candidate_without_an_extra_seed(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner, lambda action, count: action[:, 0, 0])
        bind_solver_components(solver, domain)
        optimizer.config.num_iters = optimizer.config.inner_iters
        seed = torch.full((1, 1, 1), -.5)
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            result = optimizer.optimize(seed)
        self.assertEqual(tuple(result.shape), tuple(seed.shape))
        self.assertTrue(bool((result <= 0).all()))
        torch.testing.assert_close(result, optimizer._core._candidate_action)
        self.assertTrue(optimizer._core.solve_diagnostics[0]["sampled_feasible_candidate"])
        self.assertEqual(owner.snapshot()["seeds"][0]["outer_updates"], 0)

    # BEGIN direct consecutive calls must not depend on an outer reinitialize.
    def test_consecutive_optimize_clears_old_phr_best_before_new_seed_evaluation(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner,
            lambda action, count: torch.ones_like(action[:, 0, 0]))
        bind_solver_components(solver, domain)
        core = optimizer._core
        names = ("cost", "action", "iteration", "current_iteration", "converged")
        pointers = {name: getattr(core._best, name).data_ptr() for name in names}
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.optimize(torch.full((1, 1, 1), .5))
            self.assertEqual(owner.snapshot()["seeds"][0]["outer_updates"], 1)
            self.assertLess(float(core._best.cost[0]), 100.)
            for rollout in rollouts:
                rollout.objective_offset = 100.
            initial = core._run_initial_evaluation
            entries = []

            def inspect_entry(state):
                entries.append(True)
                # The old cost is below every new objective value. Keeping it
                # would prevent ordinary native best tracking from improving.
                self.assertGreater(float(state.best_cost[0]), 100.)
                self.assertEqual(int(state.current_iteration[0]), 0)
                self.assertEqual(int(core._iterations[0]), 0)
                self.assertEqual(owner.snapshot()["seeds"][0]["outer_updates"], 0)
                self.assertEqual(float(optimizer._qn.s.abs().sum()), 0.)
                return initial(state)

            with patch.object(core, "_run_initial_evaluation", side_effect=inspect_entry):
                result = optimizer.optimize(torch.full((1, 1, 1), -.5))
        self.assertEqual(entries, [True])
        self.assertGreaterEqual(float(core._best.cost[0]), 100.)
        self.assertEqual(tuple(result.shape), (1, 1, 1))
        self.assertGreater(float(result[0, 0, 0]), -.5)
        self.assertEqual(pointers, {name: getattr(core._best, name).data_ptr() for name in names})
    # END within-solve block tests separately prove cumulative clocks are kept.
    # END no dense metric or extra native solve is used by candidate retention.

    # BEGIN actual CPU native steps exercise request cancellation inside a block.
    def test_request_guard_stops_after_one_actual_step_inside_native_inner_block(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        core = optimizer._core
        seed = torch.zeros((1, 1, 2))
        def guard():
            if int(core._iterations[0]) >= 1:
                raise TimeoutError("same request deadline")
        core.request_guard = guard
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            with self.assertRaisesRegex(TimeoutError, "same request deadline"):
                optimizer.optimize(seed)
        self.assertEqual(core.config.inner_iters, 2)
        self.assertEqual(int(core._iterations[0]), 1)
        self.assertEqual(core._recovery_reports, [[]])

    def test_expired_forward_never_starts_backward_for_either_native_callback(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        core = optimizer._core
        for initial in (False, True):
            with self.subTest(initial=initial):
                expired, backward = [False], []
                rollout = rollouts[int(initial)]
                original = rollout.evaluate_action
                def forward(action, **kwargs):
                    result = original(action, **kwargs)
                    costs = result.costs_and_constraints.get_sum_cost_and_constraint(True)
                    costs.register_hook(lambda gradient: backward.append(True) or gradient)
                    expired[0] = True
                    return result
                def guard():
                    if expired[0]:
                        raise TimeoutError("forward consumed deadline")
                core.request_guard = guard
                x = torch.full((core.config.num_problems, core.config.num_particles, core.opt_dim), .4)
                callback = (core._compute_cost_constraint_and_gradient_initial if initial
                            else core._compute_cost_constraint_and_gradient)
                with patch.object(rollout, "evaluate_action", side_effect=forward):
                    with self.assertRaisesRegex(TimeoutError, "forward consumed deadline"):
                        callback(x)
                self.assertEqual(backward, [])

    def test_completed_cuda_boundary_waits_only_current_stream_before_guard(self):
        # Mock the CUDA interface only: no CUDA tensor or environment is needed
        # to prove that elapsed-device work precedes the deadline observation.
        core = object.__new__(PhysicalGradientCore)
        core.device_cfg = SimpleNamespace(device=torch.device("cuda:0"))
        events = []
        core.request_guard = lambda: events.append("guard")
        stream = SimpleNamespace(synchronize=lambda: events.append("stream_done"))
        with patch("torch.cuda.current_stream", return_value=stream) as current, \
             patch("torch.cuda.is_current_stream_capturing", return_value=False), \
             patch("torch.cuda.synchronize", side_effect=AssertionError("global device synchronization")):
            core._check_request()
            core._check_request(wait_for_device=True)
        self.assertEqual(events, ["guard", "stream_done", "guard"])
        current.assert_called_once_with(torch.device("cuda:0"))

    def test_deadline_crossed_by_backward_rejects_completed_cost_callback(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        core = optimizer._core
        expired = [False]
        original = rollouts[0].evaluate_action
        def forward(action, **kwargs):
            result = original(action, **kwargs)
            costs = result.costs_and_constraints.get_sum_cost_and_constraint(True)
            def backward(gradient):
                expired[0] = True
                return gradient
            costs.register_hook(backward)
            return result
        def guard():
            if expired[0]:
                raise TimeoutError("backward consumed deadline")
        core.request_guard = guard
        x = torch.full((core.config.num_problems, core.config.num_particles, core.opt_dim), .4)
        with patch.object(rollouts[0], "evaluate_action", side_effect=forward):
            with self.assertRaisesRegex(TimeoutError, "backward consumed deadline"):
                core._compute_cost_constraint_and_gradient(x)
        self.assertTrue(expired[0])

    def test_guarded_cost_and_gradient_preserve_both_native_callbacks(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        core = optimizer._core
        x = torch.full((core.config.num_problems, core.config.num_particles, core.opt_dim), .4)
        checked = []
        core.request_guard = lambda: checked.append(True)
        for name in ("_compute_cost_constraint_and_gradient", "_compute_cost_constraint_and_gradient_initial"):
            expected = getattr(GradientOptCore, name)(core, x)
            actual = getattr(core, name)(x)
            for before, after in zip(expected, actual):
                torch.testing.assert_close(before, after, rtol=0., atol=0.)
        self.assertEqual(len(checked), 6)

    def test_recovery_rejects_a_qp_finishing_after_deadline_before_any_trial(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        seed = torch.ones((1, 1, 2))
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
        core = optimizer._core
        core._stalled_steps.fill_(core.config.inner_iters)
        expired = [False]
        def proposal(*args):
            expired[0] = True
            return -torch.full_like(seed, .1), {"success": True, "reason": "synthetic QP"}
        def guard():
            if expired[0]:
                raise TimeoutError("QP consumed deadline")
        core.request_guard = guard
        with patch.object(domain, "recovery_step", side_effect=proposal), \
             patch.object(core, "_compute_cost_constraint_and_gradient",
                          wraps=core._compute_cost_constraint_and_gradient) as evaluations:
            with self.assertRaisesRegex(TimeoutError, "QP consumed deadline"):
                core._recover_stalled(core._iteration_state)
        # Current/best references were re-evaluated; no backtracking trial ran.
        self.assertEqual(evaluations.call_count, 2)
        self.assertEqual(core._recovery_reports, [[]])
    def test_recovery_deadline_after_first_trial_prevents_another_backtrack(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        seed = torch.ones((1, 1, 2))
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
        core = optimizer._core
        core._stalled_steps.fill_(core.config.inner_iters)
        expired, count = [False], [0]
        original = core._compute_cost_constraint_and_gradient
        def evaluate(x):
            result = original(x)
            count[0] += 1
            expired[0] = count[0] >= 3
            return result
        def guard():
            if expired[0]:
                raise TimeoutError("first recovery trial consumed deadline")
        core.request_guard = guard
        with patch.object(domain, "recovery_step", return_value=(-torch.full_like(seed, .1),
                {"success": True, "reason": "inward ascending counterexample"})), \
             patch.object(core, "_compute_cost_constraint_and_gradient", side_effect=evaluate):
            with self.assertRaisesRegex(TimeoutError, "first recovery trial consumed deadline"):
                core._recover_stalled(core._iteration_state)
        self.assertEqual(count[0], 3)
        self.assertEqual(core._recovery_reports, [[]])

    # END deadline failures never become completed optimization candidates.

    def test_both_rollouts_obey_domains_through_native_resize_and_reinitialize(self):
        lower, upper = torch.tensor([.1, -.8]), torch.tensor([1., .9])
        # BEGIN explicit derivative limits accompany the named position intervals.
        bounds = torch.stack((torch.stack((lower, upper)),
            torch.tensor([[-10., -10.], [10., 10.]]),
            torch.tensor([[-100., -100.], [100., 100.]]),
            torch.tensor([[-1000., -1000.], [1000., 1000.]])))
        domains = [(JointPositionDomain(lower, upper), 1),
            (CubicJointDomain(bounds, torch.tensor([.4, .2]),
                             torch.tensor([.05, -.05]), .2, 8), 8)]
        # END both rollout copies consume the same continuous physical domain.
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

    # BEGIN partial block tails recover without changing AL or retirement cadence.
    def test_first_update_then_24_stalls_recovers_before_the_first_al_update(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner,
            lambda action, count: action[:, 0, 0]*0+1., target=4.)
        bind_solver_components(solver, domain)
        optimizer.config.num_iters, optimizer.config.inner_iters = 50, 25
        core = optimizer._core
        seed = torch.tensor([[[.9, 0.]]])
        boundaries = []
        recover, update = core._recover_stalled, owner.update

        def recover_at_boundary(state):
            boundaries.append(("recovery", core._iterations.tolist(),
                core._stalled_steps.tolist(), core._effective_updates.tolist()))
            return recover(state)

        def update_at_boundary(signed):
            boundaries.append(("AL", core._iterations.tolist(),
                core._stalled_steps.tolist(), core._effective_updates.tolist()))
            return update(signed)

        with patch("curobo._src.runtime.cuda_event_timers", False), \
                patch.object(core, "_recover_stalled", side_effect=recover_at_boundary), \
                patch.object(owner, "update", side_effect=update_at_boundary):
            optimizer.reinitialize(seed)
            result = optimizer.optimize(seed)
        # These counters come from the actual native 25-step block. Recovery
        # sees its one effective move and 24 subsequent ineffective steps.
        self.assertEqual(boundaries[0], ("recovery", [25], [24], [1]))
        self.assertEqual(boundaries[1], ("AL", [25], [0], [2]))
        self.assertEqual(sum(row[0] == "AL" for row in boundaries), 1)
        report = core.solve_diagnostics[0]
        first = report["recoveries"][0]
        self.assertEqual(first["stalled_steps_before"], 24)
        self.assertTrue(first["accepted"])
        self.assertFalse(first["retired"])
        self.assertEqual(report["iterations"], 50)
        self.assertEqual(report["functional_continuation"]["outer_updates"], 1)
        torch.testing.assert_close(result, torch.ones_like(seed), rtol=0, atol=0)

    def test_partial_failed_recovery_refreshes_main_gradient_and_keeps_peer_state(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer.update_num_problems(2)
        core, history = optimizer._core, optimizer._qn
        seed = torch.ones((2, 1, 2))

        def initial_objective(action, use_cuda_graph=False):
            domain.validate(action)
            costs = ((action+4.)**2).sum((-1, -2))
            return SimpleNamespace(costs_and_constraints=SimpleNamespace(
                get_sum_cost_and_constraint=lambda sum_horizon: costs))

        with patch("curobo._src.runtime.cuda_event_timers", False), \
                patch.object(rollouts[1], "evaluate_action", side_effect=initial_objective):
            optimizer.reinitialize(seed)
        state = core._iteration_state
        self.assertTrue(bool((state.exploration_gradient > 0).all()))
        core._stalled_steps[0] = 1
        for name in ("s", "y", "rho", "alpha"):
            getattr(history, name).fill_(3.)
        peer_state = {field.name: getattr(state, field.name)[1].clone() for field in fields(state)
            if isinstance(getattr(state, field.name), torch.Tensor)}
        peer_history = {name: getattr(history, name).clone() for name in
            ("s", "y", "rho", "alpha", "x_0", "grad_0", "step_q_buffer")}
        # The supplied inward direction is legal but increases the main
        # objective. Its rejection must not retire a one-step block tail.
        with patch.object(domain, "recovery_step", return_value=(-torch.full_like(seed[:1], .1),
                {"success": True, "reason": "inward_ascending_trial"})):
            core._recover_stalled(state)
        torch.testing.assert_close(state.exploration_action, seed, rtol=0, atol=0)
        torch.testing.assert_close(state.exploration_gradient[0], torch.full_like(seed[0], -6.))
        torch.testing.assert_close(state.exploration_cost[0], torch.tensor(18.))
        torch.testing.assert_close(state.step_direction[0],
            -optimizer.config.initial_step_scale*state.exploration_gradient[0])
        torch.testing.assert_close(history.x_0[0], state.exploration_action[0].reshape_as(history.x_0[0]))
        torch.testing.assert_close(history.grad_0[0], state.exploration_gradient[0].reshape_as(history.grad_0[0]))
        for name in ("s", "y", "rho", "alpha"):
            self.assertEqual(float(getattr(history, name)[:, 0].abs().sum()), 0.)
        for name, saved in peer_state.items():
            torch.testing.assert_close(getattr(state, name)[1], saved, rtol=0, atol=0)
        for name, saved in peer_history.items():
            index = (slice(None), 1) if saved.ndim == 4 else 1
            torch.testing.assert_close(getattr(history, name)[index], saved[index], rtol=0, atol=0)
        self.assertEqual(core._exhausted.tolist(), [False, False])
        self.assertEqual(core._stalled_steps.tolist(), [1, 0])
        self.assertEqual(core._effective_updates.tolist(), [0, 0])
        self.assertEqual(core._history_resets.tolist(), [1, 0])
        recovery = core.solve_diagnostics[0]["recoveries"][0]
        self.assertFalse(recovery["accepted"])
        self.assertFalse(recovery["retired"])
        self.assertEqual(recovery["stalled_steps_before"], 1)
        # Native steps remain available. Once the original complete-stall
        # condition is reached, its genuine constrained stationary point retires.
        core._dispatch("_opt_iters", state)
        self.assertEqual(core._iterations.tolist(), [2, 2])
        self.assertTrue(core.solve_diagnostics[0]["recoveries"][-1]["retired"])

    def test_partial_failed_recovery_keeps_feasible_trials_outside_armijo_selection(self):
        domain = JointPositionDomain(-torch.ones(1), torch.ones(1))
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner, lambda action, count: action[:, 0, 0])
        bind_solver_components(solver, domain)
        seed = torch.full((1, 1, 1), .5)
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
        core, state = optimizer._core, optimizer._core._iteration_state
        self.assertFalse(bool(core._candidate_valid.any()))
        core._stalled_steps.fill_(1)
        # At .5 the augmented quadratic is stationary. The negative trial
        # satisfies the original event but raises that quadratic, so the two
        # independent production decisions must retain it without moving base.
        with patch.object(domain, "recovery_step", return_value=(-torch.full_like(seed, .6),
                {"success": True, "reason": "feasible_non_armijo_trial"})):
            core._recover_stalled(state)
        torch.testing.assert_close(state.exploration_action, seed, rtol=0, atol=0)
        self.assertFalse(bool(core._exhausted.any()))
        self.assertEqual(core._effective_updates.tolist(), [0])
        self.assertEqual(core._candidate_valid.tolist(), [True])
        self.assertLess(float(core._candidate_action[0, 0, 0]), 0.)
        torch.testing.assert_close(core._candidate_objective,
            ((core._candidate_action-1.)**2).sum((-1, -2)))
        self.assertFalse(core.solve_diagnostics[0]["recoveries"][0]["accepted"])
    # END actual trial evidence and original stopping thresholds remain independent.

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
