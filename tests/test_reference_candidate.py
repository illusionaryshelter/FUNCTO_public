"""CPU contracts for preserving supplied curves at the shared delivery boundary.

Native geometry numerics remain integration coverage. These tests substitute
only its completed metric evidence, leaving request ordering, clock binding,
Hermite/domain checks and the early-return policy in production code.
"""
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from manipulation.planning import (
    JointTrajectory, PlanningBudget, _CuroboPlanner, _ProgressMap,
    _candidate_sample_times, _phase_aligned_parameters,
)
from manipulation.planning_problem import PlanningEvent
from test_motion_backend_contract import Native, Scene, problem, run


# BEGIN load the production nested adapter without constructing CUDA resources.
def process_optimizer_class():
    import ast
    import inspect
    import manipulation.planning as planning
    from curobo.trajectory_optimizer import TrajectoryOptimizer

    tree = ast.parse(inspect.getsource(planning))
    definition = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.ClassDef) and node.name == "ProcessTrajectoryOptimizer")
    scope = {"TrajectoryOptimizer": TrajectoryOptimizer}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
                 planning.__file__, "exec"), scope)
    return scope["ProcessTrajectoryOptimizer"]
# END execute the actual adapter and installed superclass, not a copied repair.


def supplied_curve():
    times = np.array([0., .13, .71, 1.])
    return JointTrajectory(("q",), times, (.1*times**2)[:, None], (.2*times)[:, None],
        np.array([0., .3, .6, 1.]), np.array([0., .5, .5, 1.]), (), (), {})


def candidate_fixture(*, accepted=True, components=None, native_outcomes=()):
    native = Native(native_outcomes)
    native.problem, native.scene, native.joint_names = problem(), Scene(), ("q",)
    native.start = SimpleNamespace(position=torch.zeros((1, 1)), velocity=torch.zeros((1, 1)))
    for name in ("bind_reference_candidate", "evaluate_candidate", "_candidate_domain_validation"):
        setattr(native, name, MethodType(getattr(_CuroboPlanner, name), native))

    def completed_metrics(curve, scene):
        native.calls.append("candidate")
        return {"accepted": accepted, "native_success": accepted,
            "residuals": {"interpolated:process_scene": 0. if accepted else 1.},
            "scene_components": components or {}}, np.array([0., 1.]), 4
    native._candidate_metrics = Mock(side_effect=completed_metrics)
    return native


def test_supplied_curve_passes_without_any_fit_ik_or_optimization():
    reference = supplied_curve()
    originals = [value.copy() for value in (reference.times, reference.position, reference.velocity)]
    native = candidate_fixture()
    result, backend = run(native, joint_reference=reference)
    try:
        assert result.success, result.diagnostics
        assert native.calls == ["candidate"]
        for actual, original in zip((result.trajectory.times, result.trajectory.position, result.trajectory.velocity), originals):
            np.testing.assert_array_equal(actual, original)
        assert result.diagnostics["joint_reference"]["source_curve_preserved"]
        counts = result.request_ledger["consumed_planning"]
        assert counts["ik_batches"] == counts["trajectory_solves"] == 0
    finally:
        backend.close()


def test_missing_observation_scope_returns_original_without_spatial_repair():
    gaps = {"unknown/forbidden/robot": {"maximum": .004, "time_s": .71, "progress": .6}}
    native = candidate_fixture(accepted=False, components=gaps)
    reference = supplied_curve()
    result, backend = run(native, joint_reference=reference)
    try:
        assert not result.success and result.execution_binding is None
        assert result.diagnostics["failure"] == "reference_outside_observation_scope"
        assert result.diagnostics["reference_scope_gaps"] == gaps
        assert native.calls == ["candidate"]
        np.testing.assert_array_equal(result.diagnostics["diagnostic_trajectory"].position, reference.position)
    finally:
        backend.close()


def test_supported_collision_enters_existing_finite_repair_only_after_original_check():
    native = candidate_fixture(accepted=False, components={"collision/robot/object": {"maximum": .01}},
                               native_outcomes=(False, True))
    result, backend = run(native, joint_reference=supplied_curve())
    try:
        assert result.success, result.diagnostics
        assert native.calls == ["candidate", "seed", "reference", "solve", "export"]
        assert not result.diagnostics["original_reference_candidate"]["accepted"]
    finally:
        backend.close()


def test_phase_mapping_keeps_moving_plateau_and_first_arrival_of_added_hold():
    source = _ProgressMap(np.array([0., .3, .6, 1.]), np.array([0., .5, .5, 1.]))
    target = _ProgressMap(np.array([0., .2, .7, .9, 1.]), np.array([0., .5, .5, 1., 1.]))
    np.testing.assert_allclose(_phase_aligned_parameters(source, target, np.array([.3, .45, .6, 1.])),
                               [.2, .45, .7, .9], rtol=0, atol=2e-16)
    with pytest.raises(Exception, match="plateau motion"):
        _phase_aligned_parameters(source, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), np.array([.45]))


def test_nonuniform_query_grid_retains_every_original_piece_and_bounds_spacing():
    reference = supplied_curve()
    parameters = _candidate_sample_times(reference, PlanningBudget())
    assert set(reference.times) <= set(parameters)
    assert np.diff(parameters).max() <= float(np.float32(.025))
    assert parameters[0] == 0. and parameters[-1] == reference.duration
    q, dq, ddq, _ = reference.sample(parameters)
    np.testing.assert_allclose(q[:, 0], .1*parameters**2, atol=1e-15)
    np.testing.assert_allclose(dq[:, 0], .2*parameters, atol=1e-15)
    np.testing.assert_allclose(ddq[:, 0], .2, atol=1e-14)


def test_completed_native_metrics_use_same_domain_boundary_without_requerying_scene():
    native = candidate_fixture()
    native._candidate_metrics.side_effect = AssertionError("Completed metrics must not be recomputed")
    curve, attempt, points = native.evaluate_candidate(supplied_curve(),
        native_report={"accepted": True, "residuals": {"interpolated:process_scene": 0.}})
    assert attempt["accepted"] and points == 0
    assert native.last_validation["cubic_derivatives"]["acceleration"]["maximum_bound_excess"] == 0
    assert curve is native.delivery_candidate


def test_native_metric_failure_clears_explicit_candidate_context():
    from curobo.types import JointState
    from curobo._src.solver.manager_goal import GoalManager
    from curobo._src.rollout.rollout_robot import RobotRollout
    from curobo._src.transition.robot_state_transition import RobotStateTransition
    from curobo._src.types.device_cfg import DeviceCfg
    native = candidate_fixture()
    native.anchors = np.array([0., 1.])
    native.device_cfg = DeviceCfg("cpu")
    native.joint_names = ("first", "second")
    native.torch, native.JointState, native.check_request = torch, JointState, Mock()
    native.start = JointState.from_position(torch.zeros((1, 2)), joint_names=list(native.joint_names))
    source = supplied_curve()
    reference = JointTrajectory(("second", "first"), source.times,
        np.c_[source.position, 2*source.position], np.c_[source.velocity, 2*source.velocity],
        source.progress_u, source.progress_s, (), (), {})
    # BEGIN exercise the installed native buffer contract, including seed axes.
    goals = GoalManager(DeviceCfg("cpu"))
    def prepare(**kwargs):
        return goals.update_goal_buffer(kwargs["solve_state"], goal_tool_poses=kwargs["goal_tool_poses"],
            current_js=kwargs["current_state"], seed_goal_js=kwargs["seed_goal_state"],
            goal_js=kwargs["goal_state"], use_implicit_goal=kwargs["use_implicit_goal"])
    # END only geometry execution below is replaced by the failing test sentry.
    manager = SimpleNamespace(_process_mode="trajectory_metrics", _candidate_query=None)
    # BEGIN the real augmented-state adapter verifies native scalar/state layouts.
    transition = RobotStateTransition.__new__(RobotStateTransition)
    transition.robot_dynamics = None
    transition._empty_joint_torque = None
    transition.num_dof, transition.config = 2, SimpleNamespace(device_cfg=native.device_cfg, horizon=81)
    transition.batch_size, transition.action_dim = 1, 2
    transition.robot_model = SimpleNamespace(compute_kinematics=lambda state, **kwargs: SimpleNamespace())
    def metric_sentry(state):
        assert state.joint_torque.shape == state.joint_state.position.shape
        assert state.joint_state.joint_names == list(native.joint_names)
        for name in ("position", "velocity", "acceleration", "jerk"):
            value = getattr(state.joint_state, name)
            assert value.shape[-1] == 2 and value.is_contiguous()
            value.view(-1)  # The installed cspace Warp adapter uses this exact operation.
        raise RuntimeError("metric sentry")
    # END GPU kinematics/cost kernels alone are outside this CPU API check.
    rollout = SimpleNamespace(_cost_manager_list=[manager], metrics_constraint_manager=manager,
        transition_model=transition, compute_metrics_from_state=metric_sentry)
    # The real rollout update multiplies the registry's batch and seed counts.
    # A goal-free registry's default -1 must not reach native state allocation.
    bound_rollout = RobotRollout.__new__(RobotRollout)
    bound_rollout.transition_model, bound_rollout._cost_manager_list = transition, []
    bound_rollout.start_state = bound_rollout._num_particles_goal = bound_rollout._metrics_goal = None
    bound_rollout._batch_size = 1
    # BEGIN production adapter owns metadata for both creation and resident reuse.
    optimizer_class = process_optimizer_class()
    native.trajopt = optimizer_class.__new__(optimizer_class)
    native.trajopt.core = SimpleNamespace(disable_tool_pose_tracking=Mock(), disable_joint_position_tracking=Mock(),
        prepare_goal_buffer=prepare,
        update_rollout_params=lambda goal, include_auxiliary: bound_rollout.update_params(goal, num_particles=1),
        additional_metrics_rollouts={"interpolated_rollout": rollout})
    # END candidate code itself no longer patches the returned native registry.
    for _ in range(2):
        with pytest.raises(RuntimeError, match="metric sentry"):
            _CuroboPlanner._candidate_metrics(native, reference, Scene())
        assert manager._candidate_query is None
        assert goals._goal_buffer.batch_size == 1
        assert goals._goal_buffer.seed_goal_js.position.shape == (1, 1, 2)
        assert goals._goal_buffer.seed_goal_js.dt.shape == (1, 1)
        torch.testing.assert_close(goals._goal_buffer.seed_goal_js.position, torch.tensor([[[.2, .1]]]))


# BEGIN native direct-solve control flow shares the same buffer admission.
@pytest.mark.parametrize("supply_trajectory", [False, True])
def test_native_direct_solve_binds_batch_on_creation_reuse_and_changed_seed_count(supply_trajectory):
    from curobo.types import JointState
    from curobo._src.solver.manager_goal import GoalManager
    from curobo._src.solver.solver_core import SolverCore
    from curobo._src.solver.solve_mode import SolveMode
    from curobo._src.solver.solve_state import SolveState
    from curobo._src.rollout.rollout_robot import RobotRollout
    from curobo._src.transition.robot_state_transition import RobotStateTransition
    from curobo._src.types.device_cfg import DeviceCfg

    device = DeviceCfg("cpu")
    transition = RobotStateTransition.__new__(RobotStateTransition)
    transition.config = SimpleNamespace(device_cfg=device, horizon=3)
    transition.action_dim, transition.num_dof, transition.batch_size = 2, 2, 0
    transition.robot_dynamics = None
    transition.robot_model = SimpleNamespace(update_batch_size=Mock())
    transition._rollout_step_fn = SimpleNamespace(update_batch_size=Mock())
    rollout = RobotRollout.__new__(RobotRollout)
    rollout.transition_model, rollout._cost_manager_list = transition, []
    rollout.start_state = rollout._num_particles_goal = rollout._metrics_goal = None
    rollout._batch_size = None
    rollout.compute_state_from_action = lambda action: SimpleNamespace(joint_state=JointState.from_position(action))

    core = SolverCore.__new__(SolverCore)
    core.config, core.device_cfg = SimpleNamespace(use_cuda_graph=False), device
    core.goal_registry_manager = GoalManager(device)
    core.metrics_rollout = core.auxiliary_rollout = rollout
    core.additional_metrics_rollouts = {"interpolated_rollout": rollout}
    core._task_initialized = False
    # Only numerical optimization and CUDA timing are replaced. Real native
    # core preparation, registry mutation, rollout copies and state allocation
    # must finish before this boundary is reached.
    core.optimizer = SimpleNamespace(optimizers=[SimpleNamespace(config=SimpleNamespace(), enable=Mock())],
        reset_shape=Mock(), update_num_problems=Mock(),
        update_rollout_params=lambda goal: rollout.update_params(goal, num_particles=1),
        reinitialize=Mock(side_effect=RuntimeError("optimization boundary")))
    optimizer_class = process_optimizer_class()
    optimizer = optimizer_class.__new__(optimizer_class)
    optimizer.core, optimizer._seed_dt_buffer = core, None
    optimizer.config = SimpleNamespace(minimum_trajectory_dt=.1, maximum_trajectory_dt=.1)
    optimizer.compute_trajectory_dt = lambda state, **kwargs: torch.full((state.position.shape[0],), .1)
    current = JointState.from_position(torch.zeros((1, 2)))
    previous, previous_count = None, None
    timer = SimpleNamespace(start=lambda: None)
    initialized = torch.cuda.is_initialized()
    with patch("curobo._src.solver.solver_trajopt.CudaEventTimer", return_value=timer), \
            patch("torch.cuda.init", side_effect=AssertionError("CPU contract must not allocate CUDA")):
        for index, count in enumerate((1, 1, 2, 2, 1)):
            action = torch.full((count, 3, 2), float(index))
            core.prepare_trajectory_seeds = lambda *args, **kwargs: action
            state = SolveState(solve_type=SolveMode.BATCH, num_trajopt_seeds=count,
                               batch_size=1, num_envs=1, num_goalset=1, tool_frames=[])
            with pytest.raises(RuntimeError, match="optimization boundary"):
                optimizer._solve_impl(current, state, None, count,
                    seed_traj=action if supply_trajectory else None, dt=torch.full((1, count), .1),
                    finetune_attempts=0, finetune_dt_scale=1.)
            goal = core.goal_registry_manager._goal_buffer
            assert goal is core._goal_buffer and goal.batch_size == 1 and goal.num_seeds == count
            assert (goal is previous) == (count == previous_count)
            assert goal.goal_js is None and goal.link_goal_poses is None
            assert goal.seed_goal_js.position.shape == (1, count, 2)
            assert goal.seed_goal_js.dt.shape == (1, count)
            torch.testing.assert_close(goal.seed_goal_js.position, action[:, -1][None])
            assert rollout._metrics_goal.batch_size == rollout._num_particles_goal.batch_size == 1
            assert transition.state_seq.position.shape == (count, 3, 2)
            previous, previous_count = goal, count
    assert torch.cuda.is_initialized() == initialized
# END no prior candidate evaluation is required to initialize a direct solve.


def test_process_manager_consumes_nonuniform_times_and_exact_source_anchor_states():
    from curobo.types import JointState
    from test_manipulation_planning import NativeCostManagerOwnershipTests
    owner = NativeCostManagerOwnershipTests()
    owner.setUp()
    try:
        with owner.anchor_self_collision_fixture("trajectory_metrics", required=(.55, .65)) as fixture:
            q = torch.tensor([[[-1., 0.], [2., 0.], [1., 0.]]]).repeat(2, 1, 1)
            js = JointState.from_position(q, joint_names=["first_joint", "second_joint"])
            js.velocity, js.dt = torch.zeros_like(q), torch.ones(2)
            fixture.state.joint_state = js
            fixture.state.cuda_robot_model_state = fixture.kinematics(js)
            seen = []

            def process(p, r, progress, joints, names, *, trajectory_time, **kwargs):
                seen.append((trajectory_time.clone(), joints.clone()))
                return {"process": torch.zeros_like(progress)}
            fixture.problem.residuals = process
            fixture.manager._candidate_query = {
                "scene": fixture.scene, "mapping": _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                "duration": 2., "times": torch.tensor([[0., .3, 2.]]).repeat(2, 1),
                "progress": torch.tensor([[0., .15, 1.]]).repeat(2, 1),
                "anchor_u": torch.tensor([[.55, .65]]).repeat(2, 1),
                "anchor_s": torch.tensor([[.55, .65]]).repeat(2, 1),
                "anchor_q": torch.tensor([[[4., 0.], [5., 0.]]]).repeat(2, 1, 1),
                "anchor_dq": torch.zeros((2, 2, 2)), "scene_components": {}}
            fixture.manager.compute_costs(fixture.state)
            torch.testing.assert_close(seen[0][0], torch.tensor([[0., .3, 1.1, 1.3, 2.]]).repeat(2, 1))
            torch.testing.assert_close(seen[0][1][..., 0], torch.tensor([[-1., 2., 4., 5., 1.]]).repeat(2, 1))
            fixture.manager._candidate_query = None
            fixture.manager.compute_costs(fixture.state)
            assert float(seen[1][0][0, 1]) == 1.
    finally:
        owner.doCleanups()


def test_terminal_evidence_after_last_tick_is_typed_unchecked_without_native_allocation():
    from manipulation.planning import MotionBackend
    request = problem()
    request.events = (PlanningEvent("terminal", "skill", 1., 197, "verification", "recorded terminal",
                                   "actual evidence", True),)
    duration = 9.850000000000293
    reference = JointTrajectory(("q",), np.array([0., duration]), np.zeros((2, 1)), np.zeros((2, 1)),
        np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    scene = Scene()
    scene.gripper = SimpleNamespace(control_timestep_s=.002)
    backend = MotionBackend({}, robot_validator=object())
    with patch("manipulation.planning._CuroboPlanner", side_effect=AssertionError("must reject before native allocation")) as factory:
        result = backend.plan(request, scene, joint_reference=reference)
    assert not result.success and result.execution_binding is None
    assert result.diagnostics["failure"] == "reference_clock_outside_curve"
    issue = result.diagnostics["reference_clock_issues"][0]
    assert issue["device_step"] == 4926 and issue["device_time_s"] == 9.852
    assert issue["curve_duration_s"] == duration and issue["status"] == "outside_curve_unchecked"
    assert result.request_ledger["consumed_planning"]["initializations"] == 0
    factory.assert_not_called()
    np.testing.assert_array_equal(reference.times, [0., duration])
    backend.close()


def test_native_pair_replacement_keeps_complete_kinematic_policy_immutable():
    from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg
    complete = SelfCollisionKinematicsCfg(num_spheres=3, sphere_padding=torch.zeros(3),
        collision_pairs=torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int16))
    def cost():
        return SimpleNamespace(config=SimpleNamespace(self_collision_kin_config=complete), enabled=True,
            enable_cost=Mock(), disable_cost=Mock(), _batch_size=2)
    regular, anchor = cost(), cost()
    manager = SimpleNamespace(get_cost=lambda _: regular, _anchor_self_collision=anchor, _batch_size=2)
    rollout = SimpleNamespace(_cost_manager_list=[manager])
    solver = SimpleNamespace(core=SimpleNamespace(get_all_rollout_instances=lambda: [rollout]))
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.torch, planner.ik = torch, solver
    planner.trajopt = SimpleNamespace(core=solver.core, additional_metrics_rollouts={"dense": rollout})
    planner._verification_native = manager
    planner.scene = SimpleNamespace(native_self_collision_pairs=[[0, 2]])
    planner._bind_native_self_collision_pairs()
    torch.testing.assert_close(complete.collision_pairs, torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int16))
    for item in (regular, anchor):
        torch.testing.assert_close(item.config.self_collision_kin_config.collision_pairs, torch.tensor([[0, 2]], dtype=torch.int16))
        assert item._batch_size is None
    assert manager._batch_size is None


def test_regular_samples_and_exact_anchors_share_one_metric_buffer_capacity():
    native = candidate_fixture()
    native.anchors = np.linspace(0., 1., 64)
    reference = JointTrajectory(("q",), np.array([0., 12.]), np.zeros((2, 1)), np.zeros((2, 1)),
        np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    assert len(_candidate_sample_times(reference, native.budget)) == 481
    with pytest.raises(Exception, match="samples and exact anchors"):
        _CuroboPlanner._candidate_metrics(native, reference, Scene())
