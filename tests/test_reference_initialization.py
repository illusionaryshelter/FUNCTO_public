"""Complete curve initialization through real CPU cubic and native L-BFGS code.

The two-coordinate Cartesian FK is declared analytically. Original task managers,
continuous joint-domain checks and native optimizer state are production code;
GPU robot FK and full observed-scene qualification are separate integration tests.
"""
# BEGIN exercise actual coefficient updates, task evidence and request lifecycle.
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.state.state_robot import RobotState
from curobo.types import DeviceCfg, JointState

from manipulation.cubic_trajectory import CubicTrajectoryStep
from manipulation.curobo_solver_components import bind_solver_components
from manipulation.physical_joint_domain import CubicJointDomain
from manipulation.planning import _CuroboPlanner, _ProgressMap, PlanningBudget
from test_event_allowed_set_ik import event_problem
from test_scene_query_performance import manager_fixture


class CurveRollout:
    """Analytic tool translations use the exact production cubic derivatives."""
    def __init__(self, fixture, domain, step, start):
        self.fixture, self.domain, self.step, self.start = fixture, domain, step, start
        self.constraint_manager = fixture.manager
        self.metrics_constraint_manager = fixture.manager
        self._cost_manager_list = [fixture.manager]
        self.action_horizon, self.action_dim = step.action_horizon, 2
        self.action_bound_lows, self.action_bound_highs = domain.lower.float(), domain.upper.float()
        self.evaluations = []

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def evaluate_action(self, action, **kwargs):
        return self._evaluate(action)

    def compute_metrics_from_action(self, action):
        return self._evaluate(action)

    def _evaluate(self, action):
        self.domain.validate(action)
        self.evaluations.append(action.detach().clone())
        joint = JointState.zeros((len(action), self.step.horizon, 2), DeviceCfg("cpu"),
                                 joint_names=list(self.fixture.problem.joint_names))
        self.step.sample(action, self.start, torch.zeros(len(action), dtype=torch.int32),
                         joint, valid_count=self.step.horizon)
        state = RobotState(joint_state=joint, cuda_robot_model_state=self.fixture.kinematics(joint))
        # Cartesian FK makes position tracking directly comparable with the
        # declared reference; trapezoid weights preserve the physical clock.
        progress = torch.linspace(0., 1., self.step.horizon)[None]
        reference, _ = self.fixture.problem.sample_reference(progress)
        error = (state.tool_poses.position[:, :, 0]-reference).square().sum(-1)
        weights = torch.full_like(error, 1./(self.step.horizon-1))
        weights[:, [0, -1]] *= .5
        ordinary = CostCollection()
        ordinary.add((error*weights)[..., None], "analytic_tool_reference")
        terms = self.fixture.manager.compute_costs(state)
        return SimpleNamespace(state=state, actions=action,
            costs_and_constraints=CostsAndConstraints(costs=ordinary, constraints=terms))


def make_planner(fixture, count=2, iterations=8):
    device, controls = DeviceCfg("cpu"), 14
    step = CubicTrajectoryStep(controls, 1, .125, device=device.device)
    start = JointState.from_position(torch.zeros(1, 2), joint_names=list(fixture.problem.joint_names))
    start.velocity = torch.tensor([[.02, -.03]])
    domain = CubicJointDomain(torch.full((2,), -2.), torch.full((2,), 2.),
                              start.position[0], start.velocity[0], .125, controls)
    rollout = CurveRollout(fixture, domain, step, start)
    stage = LBFGSOpt(LBFGSOptCfg(device_cfg=device, num_iters=iterations, inner_iters=2,
        history=3, line_search_scale=[0., .1, .5, 1.], initial_step_scale=.001,
        use_cuda_kernel_step_direction=False, use_cuda_kernel_line_search=False), [rollout, rollout])
    shell = SimpleNamespace(core=SimpleNamespace(device_cfg=device,
        config=SimpleNamespace(use_cuda_graph=False), optimizers=[stage],
        goal_registry_manager=GoalManager(device)))
    bind_solver_components(shell, domain)
    stage.update_num_problems(count)
    optimizer = MultiStageOptimizer([stage])
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.torch, planner.JointState, planner.start = torch, JointState, start
    planner.check_request = Mock()
    stage._core.request_guard = planner.check_request
    planner.problem = fixture.problem
    planner.joint_names = fixture.problem.joint_names
    planner.physical_domain, planner._manager_type = domain, fixture.configuration
    planner.functional_constraints = fixture.manager.optimization_constraints
    planner.budget = PlanningBudget(knots=12, trajectory_seeds=count, trajectory_iterations=iterations,
        interpolation_steps=1, maximum_trajectory_dt=.125, interpolation_dt=.125)
    planner.formal_clock = {"duration": 2., "valid_count": 17, "interpolation_dt": .125, "native_dt": .125}
    planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    planner.finger_trajectory = None
    planner.reference_initialization = {"status": "collecting_proposals", "scene_constraints": "not_evaluated"}
    planner.initial_seed_strategies = ["declared_numeric_proposal"]*count
    planner.initialization_candidate = None
    bounds = torch.tensor([[-2., -2.], [2., 2.]])
    limits = SimpleNamespace(joint_names=list(planner.joint_names), position=bounds,
        velocity=bounds, acceleration=torch.tensor([[-100., -100.], [100., 100.]]))
    planner._verification_native = SimpleNamespace(get_cost=lambda _: SimpleNamespace(config=SimpleNamespace(joint_limits=limits)))

    def interpolate(joint):
        output = JointState.zeros((count, 17, 2), device, joint_names=list(planner.joint_names))
        step.sample(joint.knot, start, torch.zeros(count, dtype=torch.int32), output, valid_count=17)
        return output, torch.full((count,), 17), False

    planner.trajopt = SimpleNamespace(optimizer=optimizer, optimizer_rollouts=[rollout],
        core=SimpleNamespace(optimizers=[stage]), action_horizon=controls, action_dim=2,
        interpolation_steps=1, disable_tool_pose_tracking=Mock(), disable_joint_position_tracking=Mock(),
        _prepare_goal_buffer=Mock(return_value=(object(), True)), _update_rollout_params=Mock(),
        get_interpolated_trajectory=interpolate)
    seeds = torch.zeros((1, count, controls, 2))
    seeds[0, :, :, 0] = torch.arange(count)[:, None]*.05
    return planner, seeds, rollout, stage


@pytest.mark.parametrize("count", [1, 2])
def test_actual_native_curve_updates_preserve_clock_entry_and_original_event_evidence(manager_fixture, count):
    with manager_fixture("trajectory_optimizer", required=(.5,), capacity=4, task=event_problem()) as fixture:
        planner, seed, rollout, stage = make_planner(fixture, count)
        fixture.scene.evaluate = Mock(side_effect=AssertionError("Initialization queried scene geometry"))
        with patch("curobo._src.runtime.cuda_event_timers", False):
            refined = planner._refine_reference_seeds(seed)
        record = planner.reference_initialization
        assert record["status"] == "completed"
        assert record["iteration_allocation"] == {"total": 8, "inner_block": 2, "initialization": 4, "main": 4}
        assert stage.config.num_iters == 8
        assert refined.shape == seed.shape and not torch.equal(seed, refined)
        assert fixture.manager.local_geometry_models is not None
        for initial, final in zip(record["coarse"]["seeds"], record["refined"]["seeds"]):
            assert final["fixed_objective"] < initial["fixed_objective"]
            assert final["functional_inequalities"].keys() == dict.fromkeys(planner.problem.functional_constraint_keys).keys()
            assert "process_scene" not in final["optimizer_constraints"]
            np.testing.assert_allclose(final["trajectory"]["velocity"][0], planner.start.velocity[0].numpy(), atol=1e-8)
            assert final["trajectory"]["times"] == initial["trajectory"]["times"] == list(np.arange(17)*.125)
            np.testing.assert_array_equal(final["controls"], refined[0, final["seed_index"]].numpy())
        # Independent task reevaluation uses the handed-off coefficients. The
        # last line-search sample is not used as the returned curve's evidence.
        fixture.configuration.set_scene_constraints(False)
        actual = rollout.evaluate_action(refined[0])
        _, signed, objective = fixture.manager.optimizer_evidence(actual)
        for index, final in enumerate(record["refined"]["seeds"]):
            assert final["fixed_objective"] == pytest.approx(float(objective[index]))
            assert final["functional_inequalities"] == {key: pytest.approx(float(value[index])) for key, value in signed.items()}
        fixture.configuration.set_scene_constraints(True)


def test_single_native_block_is_reserved_for_main_solve(manager_fixture):
    with manager_fixture("trajectory_optimizer", required=(.5,), capacity=4, task=event_problem()) as fixture:
        planner, seed, _, stage = make_planner(fixture, count=1, iterations=2)
        with patch.object(planner.trajopt.optimizer, "optimize", side_effect=AssertionError("No initializer block")):
            refined = planner._refine_reference_seeds(seed)
        torch.testing.assert_close(refined, seed, rtol=0, atol=0)
        assert planner.main_trajectory_iterations == 2
        assert planner.reference_initialization["status"] == "no_initialization_iteration_block"
        assert stage.config.num_iters == 2 and fixture.manager.local_geometry_models is not None


def test_interruption_preserves_completed_coarse_curve_and_restores_original_scope(manager_fixture):
    with manager_fixture("trajectory_optimizer", required=(.5,), capacity=4, task=event_problem()) as fixture:
        planner, seed, _, stage = make_planner(fixture)
        with patch.object(planner.trajopt.optimizer, "optimize", side_effect=TimeoutError("declared deadline")), \
             patch("curobo._src.runtime.cuda_event_timers", False):
            with pytest.raises(TimeoutError, match="declared deadline"):
                planner._refine_reference_seeds(seed)
        record = planner.reference_initialization
        assert record["status"] == "interrupted" and "coarse" in record and "refined" not in record
        assert planner.initialization_candidate is not None
        assert stage.config.num_iters == 8 and fixture.manager.local_geometry_models is not None


def test_continuous_derivative_violation_remains_evidence_and_does_not_block_handoff(manager_fixture):
    with manager_fixture("trajectory_optimizer", required=(.5,), capacity=4, task=event_problem()) as fixture:
        planner, seed, _, _ = make_planner(fixture, count=1, iterations=2)
        planner._verification_native.get_cost("cspace").config.joint_limits.acceleration.fill_(0.)
        refined = planner._refine_reference_seeds(seed)
        final = planner.reference_initialization["refined"]["seeds"][0]
        assert final["continuous_residuals"]["continuous_acceleration"] > 0
        torch.testing.assert_close(refined, seed, rtol=0, atol=0)
# END initialization evidence describes the actual curve without execution authority.


# BEGIN actual native rollout indices distinguish one curve from line-search particles.
def test_snapshot_uses_native_metric_goal_indices_and_same_query_manager(manager_fixture):
    from curobo._src.rollout.rollout_robot import RobotRollout
    with manager_fixture("trajectory_optimizer", required=(.5,), capacity=4, task=event_problem()) as fixture:
        planner, seeds, analytic, _ = make_planner(fixture, count=2)
        fixture.configuration.set_scene_constraints(False)
        fixture.manager.optimization_constraints.bind(seeds[0], num_seeds=2)
        native = RobotRollout.__new__(RobotRollout)
        native._batch_size = 2
        native._use_cuda_graph = False
        native.update_batch_size = Mock()
        native.start_state = planner.start
        native.config = SimpleNamespace(convergence_cfg=None)
        native.metrics_constraint_manager = fixture.manager
        goal_fields = dict(seed_goal_js=None, idxs_seed_goal_js=None,
                           seed_enable_implicit_goal_js=None, idxs_env=None)
        native._metrics_goal = SimpleNamespace(idxs_current_js=torch.zeros(2, dtype=torch.int32), **goal_fields)
        native._num_particles_goal = SimpleNamespace(idxs_current_js=torch.zeros(8, dtype=torch.int32), **goal_fields)

        def forward(start, action, indices, *args, **kwargs):
            joint = JointState.zeros((len(action), analytic.step.horizon, 2), DeviceCfg("cpu"),
                                     joint_names=list(planner.joint_names))
            analytic.step.sample(action, start, indices, joint, valid_count=analytic.step.horizon)
            return RobotState(joint_state=joint, cuda_robot_model_state=fixture.kinematics(joint))

        native.transition_model = native.metrics_transition_model = SimpleNamespace(forward=forward)
        native._compute_costs_and_constraints_metrics_impl = lambda state: CostsAndConstraints(
            constraints=fixture.manager.compute_costs(state))
        # The optimizer's expanded path cannot evaluate one returned curve per
        # seed. This reproduces the real GPU failure without a CUDA allocation.
        with pytest.raises(ValueError, match="one measured entrance index"):
            native.evaluate_action(seeds[0])
        planner.trajopt.optimizer_rollouts = [native]
        with patch.object(native, "evaluate_action", side_effect=AssertionError("Wrong particle layout")):
            record, curves = planner._initialization_snapshot(seeds[0])
        assert len(curves) == len(record["seeds"]) == 2
        assert all("process_scene" not in row["optimizer_constraints"] for row in record["seeds"])
        assert native._num_particles_goal.idxs_current_js.numel() == 8
        assert native._metrics_goal.idxs_current_js.numel() == 2
# END the original native goal layouts remain unchanged after snapshot evaluation.


# BEGIN native graph reset invalidates ordinary-rollout goals before rebinding.
def test_initialization_rebinds_goal_clock_after_native_graph_reset(manager_fixture):
    from types import MethodType
    from curobo._src.rollout.goal_registry import GoalRegistry
    from curobo._src.rollout.rollout_robot import RobotRollout

    with manager_fixture("trajectory_optimizer", required=(.5,), capacity=4, task=event_problem()) as fixture:
        planner, seeds, initial, stage = make_planner(fixture, count=2)
        regular = CurveRollout(fixture, planner.physical_domain, initial.step, planner.start)
        regular.start_state = None
        regular._num_particles_goal = regular._metrics_goal = None
        regular._compute_metrics_from_state_executor = regular._compute_metrics_from_action_executor = None
        # Use the installed lifecycle: reset_cuda_graph calls reset_shape and
        # deletes both goal registries even when no CUDA graph was captured.
        regular.reset_shape = MethodType(RobotRollout.reset_shape, regular)
        regular.reset_cuda_graph = MethodType(RobotRollout.reset_cuda_graph, regular)
        regular.update_params = MethodType(RobotRollout.update_params, regular)
        stage._core.rollout_fn = regular
        stage._core._rollout_list = [regular, initial]
        planner.trajopt.optimizer_rollouts = [regular, initial]
        checked_clocks = []

        def evaluate_regular(actions, **kwargs):
            goal = regular._num_particles_goal
            output = JointState.zeros((len(actions), initial.step.horizon, 2), DeviceCfg("cpu"))
            # The actual production transition checks the explicit per-seed
            # clock and entrance indices before evaluating any curve sample.
            initial.step.forward(planner.start, actions, output,
                start_state_idx=None if goal is None else goal.idxs_current_js,
                goal_state=None if goal is None else goal.seed_goal_js,
                goal_state_idx=None if goal is None else goal.idxs_seed_goal_js,
                use_implicit_goal_state=None if goal is None else goal.seed_enable_implicit_goal_js)
            checked_clocks.append(goal.seed_goal_js.dt.clone())
            return regular._evaluate(actions)

        regular.evaluate_action = evaluate_regular

        def prepare_goal(state, pose, start, implicit, endpoint):
            goal = GoalRegistry(batch_size=1, num_seeds=2, current_js=start, seed_goal_js=endpoint,
                idxs_current_js=torch.zeros((2, 1), dtype=torch.int32))
            return goal, True

        planner.trajopt._prepare_goal_buffer = Mock(side_effect=prepare_goal)
        planner.trajopt._update_rollout_params = Mock(
            side_effect=lambda goal: regular.update_params(goal, num_particles=stage.config.num_particles))

        # Negative control reproduces the failed order with native registries.
        endpoint = JointState.from_position(seeds[:, :, -1].clone())
        endpoint.dt = torch.full((1, 2), planner.budget.maximum_trajectory_dt)
        goal, _ = prepare_goal(None, None, planner.start, False, endpoint)
        regular.update_params(goal, num_particles=stage.config.num_particles)
        assert regular._num_particles_goal.seed_goal_js.dt is not None
        stage.reset_cuda_graph()
        assert regular._num_particles_goal is regular._metrics_goal is None
        with pytest.raises(ValueError, match="request's explicit goal clock"):
            regular.evaluate_action(seeds[0])

        # Coarse snapshots and initial gradients use the separate initial
        # rollout; success additionally requires the ordinary line search to
        # receive the rebound goal after all stage graph resets have finished.
        with patch("curobo._src.runtime.cuda_event_timers", False):
            refined = planner._refine_reference_seeds(seeds)
        assert planner.reference_initialization["status"] == "completed"
        assert checked_clocks and not torch.equal(refined, seeds)
        assert all(torch.equal(value, endpoint.dt) for value in checked_clocks)
        assert regular._num_particles_goal is regular._metrics_goal is None
# END the next complete solve must bind its own goals after the final reset.
