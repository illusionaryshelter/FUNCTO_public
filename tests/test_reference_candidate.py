# BEGIN tests migrated to native motion attempts and retained numerical contracts.
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
    _phase_aligned_parameters,
)
from manipulation.planning_problem import PlanningEvent
from test_motion_backend_contract import Native, Scene, problem, run


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


def supplied_curve():
    times = np.array([0., .13, .71, 1.])
    return JointTrajectory(("q",), times, (.1*times**2)[:, None], (.2*times)[:, None],
        np.array([0., .3, .6, 1.]), np.array([0., .5, .5, 1.]), (), (), {})


def test_phase_mapping_keeps_moving_plateau_and_first_arrival_of_added_hold():
    source = _ProgressMap(np.array([0., .3, .6, 1.]), np.array([0., .5, .5, 1.]))
    target = _ProgressMap(np.array([0., .2, .7, .9, 1.]), np.array([0., .5, .5, 1., 1.]))
    np.testing.assert_allclose(_phase_aligned_parameters(source, target, np.array([.3, .45, .6, 1.])),
                               [.2, .45, .7, .9], rtol=0, atol=2e-16)
    with pytest.raises(Exception, match="plateau motion"):
        _phase_aligned_parameters(source, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), np.array([.45]))


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
# END native attempt-planner test migration.
