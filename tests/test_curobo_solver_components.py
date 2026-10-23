"""CPU regressions for constrained native gradients, task blocks and goal layout."""

# BEGIN regression coverage for the behavior moved out of third-party source.
from dataclasses import fields
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
    LayoutGoalManager, PhysicalGradientCore, PhysicalLineSearch, bind_solver_components,
)
from manipulation.physical_joint_domain import (
    CubicJointDomain, JointDomainInitializationError, JointPositionDomain,
)
from manipulation.functional_augmented_lagrangian import FunctionalAugmentedLagrangian
from manipulation.local_geometry_model import LocalGeometryModels, LocalPairModel
from manipulation import physical_joint_qp


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
    the native gradient core owns every cost and gradient callback.
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


# BEGIN the installed core uses one physical direction and one ordinary line search.
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
        optimizer = solver.core.optimizers[0]
        optimizer.update_num_problems(count)
        return optimizer, rollouts, owner, models


    def test_initial_and_regular_callbacks_share_frozen_models_across_particles(self):
        optimizer, rollouts, owner, models = self.make_geometry_solver(2)
        core = optimizer
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
        seed = torch.tensor([.25, .6]).view(2, 1, 1)
        optimizer._begin_functional_solve(seed)
        state = optimizer._create_initial_iteration_state(seed)
        state.exploration_action[0].fill_(.5)
        state.current_iteration.fill_(2)
        optimizer._iterations.fill_(2)
        peer = {field.name: getattr(state, field.name)[1].clone() for field in fields(state)
                if isinstance(getattr(state, field.name), torch.Tensor)}
        optimizer._al_blocks_remaining = 2
        before_queries = len(rollouts[0].evaluations)
        with patch.object(optimizer, "_restart_changed_objective", wraps=optimizer._restart_changed_objective) as restart:
            optimizer._finish_functional_block(state)
        # One base capture supplies geometry and hard evidence; the second
        # query differentiates the changed frozen objective at that same base.
        self.assertEqual(len(rollouts[0].evaluations)-before_queries, 2)
        self.assertEqual(restart.call_count, 1)
        self.assertEqual(restart.call_args.args[-1].tolist(), [True, False])
        self.assertEqual(optimizer._iterations.tolist(), [2, 2])
        self.assertEqual(optimizer._objective_updates.tolist(), [1, 0])
        self.assertEqual(optimizer._direction_calls, [1, 0])
        self.assertEqual([row["outer_updates"] for row in owner.snapshot()["seeds"]], [0, 0])
        self.assertEqual(float(state.exploration_action[0]), .5)
        for name, saved in peer.items():
            torch.testing.assert_close(getattr(state, name)[1], saved, rtol=0, atol=0)

    def test_unchanged_geometry_and_empty_functional_keys_preserve_the_objective(self):
        optimizer, _, _, _ = self.make_geometry_solver()
        seed = torch.full((1, 1, 1), .25)
        optimizer._begin_functional_solve(seed)
        state = optimizer._create_initial_iteration_state(seed)
        saved = {field.name: getattr(state, field.name).clone() for field in fields(state)
                 if isinstance(getattr(state, field.name), torch.Tensor)}
        optimizer._al_blocks_remaining = 2
        with patch.object(optimizer, "_restart_changed_objective", wraps=optimizer._restart_changed_objective) as restart, \
                patch.object(optimizer, "_compute_cost_constraint_and_gradient",
                             wraps=optimizer._compute_cost_constraint_and_gradient) as gradients:
            self.assertIs(optimizer._finish_functional_block(state), state)
        restart.assert_not_called()
        gradients.assert_not_called()
        self.assertEqual(optimizer._objective_updates.tolist(), [0])
        for name, value in saved.items():
            torch.testing.assert_close(getattr(state, name), value, rtol=0, atol=0)

    def test_interrupted_geometry_refresh_discards_only_its_pending_records(self):
        optimizer, rollouts, _, models = self.make_geometry_solver()
        core = optimizer
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


    def test_original_100_steps_four_blocks_keep_seed_parameters_and_total_work(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner,
            lambda action, count: action.new_tensor([1., -1.]).repeat_interleave(action.shape[0]//count))
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        optimizer.update_num_problems(2)
        optimizer.config.num_iters, optimizer.config.inner_iters = 100, 25
        seed = torch.ones((2, 1, 1))
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            result = optimizer.optimize(seed)
        self.assertEqual(tuple(result.shape), tuple(seed.shape))
        reports = optimizer.solve_diagnostics
        self.assertEqual([row["functional_continuation"]["outer_updates"] for row in reports], [3, 3])
        self.assertEqual([row["functional_continuation"]["penalty"] for row in reports], [200., 2.])
        self.assertEqual(reports[0]["iterations"], 100)
        self.assertEqual(reports[1]["iterations"], 100)
        self.assertEqual([row["sampled_feasible_candidate"] for row in reports], [False, True])
        self.assertEqual(optimizer.config.num_iters, 100)


    def test_changed_objective_restarts_its_row_and_preserves_cumulative_counters(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, _, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        optimizer.update_num_problems(2)
        state = optimizer._create_initial_iteration_state(torch.ones((2, 1, 1)))
        state.current_iteration.fill_(25)
        optimizer._iterations.fill_(25)
        peer = {field.name: getattr(state, field.name)[1].clone() for field in fields(state)
                if isinstance(getattr(state, field.name), torch.Tensor)}
        optimizer._restart_changed_objective(state, torch.zeros((2, 1, 1)),
            torch.tensor([7., 8.]), torch.tensor([[[2.]], [[3.]]]), torch.tensor([True, False]))
        for name, value in peer.items():
            torch.testing.assert_close(getattr(state, name)[1], value, rtol=0, atol=0)
        self.assertEqual(optimizer._iterations.tolist(), [25, 25])
        self.assertEqual(state.current_iteration.tolist(), [25, 25])
        self.assertEqual(optimizer._objective_updates.tolist(), [1, 0])
        self.assertEqual(optimizer._direction_calls, [1, 0])
        self.assertEqual(float(state.best_cost[0]), 7.)
        self.assertEqual(float(state.exploration_gradient[0]), 2.)
        self.assertLess(float(state.step_direction[0]), 0.)

    def test_feasible_particle_survives_lower_augmented_cost_and_resets_for_new_solve(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner, lambda action, count: action[:, 0, 0])
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        core = optimizer
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
        # A new solve resets candidate validity and multipliers in the same
        # allocated seed layout; the previous feasible action owns no new score.
        pointers = {name: getattr(core, name).data_ptr() for name in core._candidate_buffers}
        core._begin_functional_solve(torch.ones((1, 1, 1)))
        self.assertFalse(bool(core._candidate_valid.any()))
        self.assertEqual(pointers, {name: getattr(core, name).data_ptr() for name in core._candidate_buffers})
        self.assertEqual(owner.snapshot()["seeds"][0]["outer_updates"], 0)

    def test_candidate_feasibility_precedes_objective_independently_for_each_seed(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        optimizer.update_num_problems(2)
        core = optimizer
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
        optimizer = solver.core.optimizers[0]
        optimizer.update_num_problems(2)
        core = optimizer
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
        optimizer = solver.core.optimizers[0]
        optimizer.config.num_iters = optimizer.config.inner_iters
        seed = torch.full((1, 1, 1), -.5)
        with patch("curobo._src.runtime.cuda_event_timers", False):
            result = optimizer.optimize(seed)
        self.assertEqual(tuple(result.shape), tuple(seed.shape))
        self.assertTrue(bool((result <= 0).all()))
        torch.testing.assert_close(result, optimizer._candidate_action)
        report = optimizer.solve_diagnostics[0]
        self.assertTrue(report["sampled_feasible_candidate"])
        self.assertIsNotNone(report["retained_candidate_objective"])
        self.assertEqual(report["iterations"], optimizer.config.num_iters)
        self.assertEqual(owner.snapshot()["seeds"][0]["outer_updates"], 0)


    def test_block_update_measures_actual_continuation_instead_of_last_trial(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner, lambda action, count: action[:, 0, 0])
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        core = optimizer
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


    def test_consecutive_optimize_clears_old_phr_best_before_new_seed_evaluation(self):
        domain = JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2)
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        owner = FunctionalAugmentedLagrangian(("event",))
        self.attach_functional_objective(rollouts, owner,
            lambda action, count: torch.ones_like(action[:, 0, 0]))
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        core = optimizer
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
                return initial(state)

            with patch.object(core, "_run_initial_evaluation", side_effect=inspect_entry):
                result = optimizer.optimize(torch.full((1, 1, 1), -.5))
        self.assertEqual(entries, [True])
        self.assertGreaterEqual(float(core._best.cost[0]), 100.)
        self.assertEqual(tuple(result.shape), (1, 1, 1))
        self.assertGreater(float(result[0, 0, 0]), -.5)
        self.assertEqual(pointers, {name: getattr(core._best, name).data_ptr() for name in names})


    def test_request_guard_stops_after_one_actual_step_inside_native_inner_block(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        core = optimizer
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


    def test_expired_forward_never_starts_backward_for_either_native_callback(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        core = optimizer
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
        core._batch_work = {"device_wait_seconds": 0.}
        events = []
        core.request_guard = lambda: events.append("guard")
        stream = SimpleNamespace(synchronize=lambda: events.append("stream_done"))
        with patch("torch.cuda.current_stream", return_value=stream) as current, \
             patch("torch.cuda.synchronize", side_effect=AssertionError("global device synchronization")):
            core._check_request()
            core._check_request(wait_for_device=True)
        self.assertEqual(events, ["guard", "stream_done", "guard"])
        current.assert_called_once_with(torch.device("cuda:0"))


    def test_deadline_crossed_by_backward_rejects_completed_cost_callback(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, optimizer, rollouts = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        core = optimizer
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
        optimizer = solver.core.optimizers[0]
        core = optimizer
        x = torch.full((core.config.num_problems, core.config.num_particles, core.opt_dim), .4)
        checked = []
        core.request_guard = lambda: checked.append(True)
        for name in ("_compute_cost_constraint_and_gradient", "_compute_cost_constraint_and_gradient_initial"):
            expected = getattr(GradientOptCore, name)(core, x)
            actual = getattr(core, name)(x)
            for before, after in zip(expected, actual):
                torch.testing.assert_close(before, after, rtol=0., atol=0.)
        self.assertEqual(len(checked), 6)


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
                optimizer = solver.core.optimizers[0]
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
        optimizer = solver.core.optimizers[0]
        seed = torch.full((1, 1, 2), .4)
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            result = optimizer.optimize(seed)
        torch.testing.assert_close(result, seed)
        torch.testing.assert_close(optimizer._best.cost, torch.full_like(optimizer._best.cost, 1e20))


    def test_illegal_seed_never_reaches_either_rollout_and_binding_is_local(self):
        domain = JointPositionDomain(torch.tensor([.1, .1]), torch.ones(2))
        owned, optimizer, rollouts = self.make_solver(domain, 1)
        ordinary, native, _ = self.make_solver(domain, 1)
        old_stages = owned.core.optimizers
        bind_solver_components(owned, domain)
        optimizer = owned.core.optimizers[0]
        self.assertIs(owned.core.optimizers, old_stages)
        self.assertIsInstance(optimizer, PhysicalGradientCore)
        self.assertIs(type(native._core), GradientOptCore)
        self.assertIs(type(ordinary.core.goal_registry_manager), GoalManager)
        with self.assertRaises(JointDomainInitializationError):
            optimizer.reinitialize(torch.zeros((1, 1, 2)))
        self.assertTrue(all(not rollout.evaluations for rollout in rollouts))
        manager = owned.core.goal_registry_manager
        bind_solver_components(owned, domain)
        self.assertIs(owned.core.goal_registry_manager, manager)


    # BEGIN ordinary directions and Armijo decisions operate independently per seed.
    def test_first_physical_direction_moves_free_coordinate_with_a_stationary_peer(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, _, _ = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        optimizer.update_num_problems(2)
        seed = torch.tensor([[[1., 1.]], [[1., .2]]])
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            state = optimizer._iteration_state
            torch.testing.assert_close(state.step_direction[0], torch.zeros_like(seed[0]), rtol=0, atol=0)
            self.assertEqual(float(state.step_direction[1, 0, 0]), 0.)
            self.assertGreater(float(state.step_direction[1, 0, 1]), 0.)
            initial_cost = state.exploration_cost.clone()
            result = optimizer._opt_step(state)
        domain.validate(result.exploration_action)
        torch.testing.assert_close(result.exploration_action[0], seed[0], rtol=0, atol=0)
        self.assertEqual(float(result.exploration_action[1, 0, 0]), 1.)
        self.assertGreater(float(result.exploration_action[1, 0, 1]), .2)
        self.assertLess(float(result.exploration_cost[1]), float(initial_cost[1]))
        self.assertEqual(optimizer._iterations.tolist(), [1, 1])
        self.assertEqual(optimizer._effective_updates.tolist(), [0, 1])

    def test_armijo_backtracks_one_seed_and_keeps_the_peers_first_accepted_state(self):
        from curobo._src.optim.gradient.line_search_context import LineSearchContext
        from curobo._src.optim.optimization_iteration_state import OptimizationIterationState

        weights = torch.tensor([1e6, 1.], dtype=torch.float64).view(2, 1, 1)
        base = torch.tensor([.003, .2], dtype=torch.float64).view(2, 1, 1)
        costs, gradient = (.5*weights*base.square()).flatten(), weights*base
        calls = []

        def evaluate(actions):
            calls.append(actions.detach().clone())
            return .5*weights*actions.square(), weights*actions

        state = OptimizationIterationState(action=base.clone(), cost=costs.clone(), gradient=gradient.clone(),
            exploration_action=base.clone(), exploration_gradient=gradient.clone(), exploration_cost=costs.clone(),
            step_direction=torch.full_like(base, -.1), best_action=base.clone(), best_cost=costs.clone(),
            best_iteration=torch.zeros(2, dtype=torch.int32), current_iteration=torch.zeros(2, dtype=torch.int32),
            converged=torch.zeros(2, dtype=torch.bool))
        context = LineSearchContext(device_cfg=DeviceCfg("cpu", dtype=torch.float64),
            line_search_scale=[0., .1, .3, .7, 1.], line_search_c_1=1e-5, line_search_c_2=.9,
            num_problems=2, opt_dim=1, action_horizon=1, action_dim=1, step_scale=1.,
            fix_terminal_action=False, action_horizon_step_max=torch.tensor([.1], dtype=torch.float64),
            use_cuda_kernel_line_search=False, compute_costs_and_gradients=evaluate,
            convergence_iteration=0, cost_delta_threshold=0., cost_relative_threshold=0.)
        search = PhysicalLineSearch(JointPositionDomain(-torch.ones(1), torch.ones(1)))
        result = search.search(state, context)
        # The first seed's smallest original move overshoots its quadratic;
        # halving the direction admits -.002. The peer accepts .1 immediately.
        torch.testing.assert_close(result.action.flatten(), torch.tensor([-.002, .1], dtype=torch.float64))
        torch.testing.assert_close(result.gradient.flatten(), torch.tensor([-2000., .1], dtype=torch.float64))
        torch.testing.assert_close(result.cost, (.5*weights*result.action.square()).flatten())
        torch.testing.assert_close(result.best_action, result.action)
        torch.testing.assert_close(result.best_cost, result.cost)
        self.assertEqual(result.current_iteration.tolist(), [1, 1])
        self.assertEqual(search.backtracks.tolist(), [1, 0])
        self.assertEqual(search.evaluations, len(calls))
        self.assertEqual(len(calls), 2)
    # END one iteration owns all of its backtracking function evaluations.

    # BEGIN the same request guard bounds forward, host QP and backtracking work.
    def test_expired_request_stops_before_either_rollout_forward(self):
        domain = JointPositionDomain(-torch.ones(1), torch.ones(1))
        solver, _, rollouts = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]

        def expired():
            raise TimeoutError("before forward")

        optimizer.request_guard = expired
        actions = torch.zeros(1, optimizer.config.num_particles, optimizer.opt_dim)
        for callback in (optimizer._compute_cost_constraint_and_gradient,
                         optimizer._compute_cost_constraint_and_gradient_initial):
            with self.assertRaisesRegex(TimeoutError, "before forward"):
                callback(actions)
        self.assertTrue(all(not rollout.evaluations for rollout in rollouts))

    def test_qp_finishing_after_deadline_cannot_start_an_objective_trial(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, _, rollouts = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        seed = torch.zeros(1, 1, 2)
        state = optimizer._create_initial_iteration_state(seed)
        state.exploration_gradient.fill_(-1.)
        completed = []
        solve = domain.descent_step

        def finishing_qp(*args, **kwargs):
            result = solve(*args, **kwargs)
            completed.append(True)
            return result

        def guard():
            if completed:
                raise TimeoutError("QP consumed deadline")

        optimizer.request_guard = guard
        with patch.object(domain, "descent_step", side_effect=finishing_qp):
            with self.assertRaisesRegex(TimeoutError, "QP consumed deadline"):
                optimizer._physical_step_direction(state)
        self.assertEqual(completed, [True])
        self.assertTrue(all(not rollout.evaluations for rollout in rollouts))
        torch.testing.assert_close(state.exploration_action, seed, rtol=0, atol=0)

    def test_deadline_after_first_joint_qp_keeps_its_count_and_stops_the_second_joint(self):
        domain = JointPositionDomain(-torch.ones(2), torch.ones(2))
        solver, _, rollouts = self.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        state = optimizer._create_initial_iteration_state(torch.zeros(1, 1, 2))
        state.exploration_gradient.fill_(-1.)
        completed = []
        native = physical_joint_qp.quadprog.solve_qp

        def solve(*args):
            result = native(*args)
            completed.append(True)
            return result

        def guard():
            if completed:
                raise TimeoutError("first joint consumed deadline")

        optimizer.request_guard = guard
        with patch.object(physical_joint_qp.quadprog, "solve_qp", side_effect=solve):
            with self.assertRaisesRegex(TimeoutError, "first joint consumed deadline"):
                optimizer._physical_step_direction(state)
        self.assertEqual(completed, [True])
        self.assertEqual(optimizer._direction_work[0]["qp_calls"], 1)
        self.assertGreater(optimizer._direction_work[0]["qp_solve_seconds"], 0.)
        self.assertEqual(optimizer._direction_reports[0]["reason"], "direction_evaluation_interrupted")
        self.assertEqual(optimizer._direction_reports[0]["qp_calls"], 1)
        self.assertTrue(all(not rollout.evaluations for rollout in rollouts))

    def test_completed_trial_exhausting_deadline_prevents_another_backtrack(self):
        domain = JointPositionDomain(-torch.ones(1), torch.ones(1))
        solver, _, rollouts = self.make_solver(domain, 1)

        def quadratic(action, use_cuda_graph=False):
            domain.validate(action)
            costs = (5e5*action.square()).sum((-1, -2))
            return SimpleNamespace(costs_and_constraints=SimpleNamespace(
                get_sum_cost_and_constraint=lambda sum_horizon: costs))

        for rollout in rollouts:
            rollout.evaluate_action = quadratic
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(torch.full((1, 1, 1), .003))
            context = optimizer._line_search_context
            evaluate = context.compute_costs_and_gradients
            completed = []

            def completed_trial(actions):
                result = evaluate(actions)
                completed.append(True)
                return result

            def guard():
                if completed:
                    raise TimeoutError("first trial consumed deadline")

            optimizer.request_guard = guard
            with patch.object(context, "compute_costs_and_gradients", side_effect=completed_trial):
                with self.assertRaisesRegex(TimeoutError, "first trial consumed deadline"):
                    optimizer._opt_step(optimizer._iteration_state)
        self.assertEqual(completed, [True])
        self.assertEqual(optimizer._iterations.tolist(), [0])
    # END unfinished directions and iterations do not publish completed candidates.

    def test_rebinding_domain_discards_cached_iteration_and_keeps_native_layout(self):
        domain = JointPositionDomain(torch.tensor([.1, .1]), torch.ones(2))
        solver, _, rollouts = self.make_solver(domain, 4)
        bind_solver_components(solver, domain)
        optimizer = solver.core.optimizers[0]
        cached = optimizer._create_initial_iteration_state(torch.full((1, 4, 2), .4))
        optimizer._iteration_state = cached
        optimizer._iterations.fill_(2)
        replacement = JointPositionDomain(domain.lower.clone(), domain.upper.clone())
        bind_solver_components(solver, replacement)
        self.assertIs(solver.core.optimizers[0], optimizer)
        self.assertIsNone(optimizer._iteration_state)
        self.assertIs(optimizer.domain, replacement)
        self.assertIs(optimizer._line_search_strategy.domain, replacement)
        self.assertIs(optimizer._initial_line_search_strategy.domain, replacement)
        self.assertEqual(optimizer._iterations.tolist(), [0])
        self.assertEqual(optimizer.get_all_rollout_instances(), rollouts)

    def test_equal_initial_trial_costs_still_allow_smaller_armijo_descent(self):
        from curobo._src.optim.gradient.line_search_context import LineSearchContext
        from curobo._src.optim.optimization_iteration_state import OptimizationIterationState

        evaluations = []

        def evaluate(values):
            x = values.detach().requires_grad_(True)
            costs = -x*(x-.125)*(x-.25)*(x-.5)*(x-1.)
            gradient, = torch.autograd.grad(costs.sum(), x)
            evaluations.append(costs.detach().clone())
            return costs, gradient

        base = torch.zeros((1, 1, 1), dtype=torch.float64)
        gradient = torch.full_like(base, -.015625)
        state = OptimizationIterationState(action=base.clone(), cost=torch.zeros(1, dtype=base.dtype),
            gradient=gradient.clone(), exploration_action=base.clone(), exploration_gradient=gradient.clone(),
            exploration_cost=torch.zeros(1, dtype=base.dtype), step_direction=torch.ones_like(base),
            best_action=base.clone(), best_cost=torch.zeros(1, dtype=base.dtype),
            best_iteration=torch.zeros(1, dtype=torch.int32), current_iteration=torch.zeros(1, dtype=torch.int32),
            converged=torch.zeros(1, dtype=torch.bool))
        context = LineSearchContext(device_cfg=DeviceCfg("cpu", dtype=torch.float64),
            line_search_scale=[0., .125, .25, .5, 1.], line_search_c_1=1e-5, line_search_c_2=.9,
            num_problems=1, opt_dim=1, action_horizon=1, action_dim=1, step_scale=1.,
            fix_terminal_action=False, action_horizon_step_max=torch.ones(1),
            use_cuda_kernel_line_search=False, compute_costs_and_gradients=evaluate,
            convergence_iteration=0, cost_delta_threshold=0., cost_relative_threshold=0.)
        search = PhysicalLineSearch(JointPositionDomain(-torch.ones(1)*2, torch.ones(1)*2))
        result = search.search(state, context)
        # These binary-exact roots give equal first-batch values even though
        # the derivative at zero is negative and x=1/16 decreases the scalar.
        torch.testing.assert_close(evaluations[0], torch.zeros_like(evaluations[0]), rtol=0, atol=0)
        self.assertEqual(float(result.action), .0625)
        self.assertEqual(float(result.cost), -.00030040740966796875)
        self.assertEqual(search.backtracks.tolist(), [1])
        self.assertEqual(result.current_iteration.tolist(), [1])

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


# END objective changes, sampled candidates and cumulative work have separate roles.

if __name__ == "__main__":
    unittest.main()
