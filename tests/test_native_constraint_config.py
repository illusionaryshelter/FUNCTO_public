"""CPU integration of planner configuration, native state bounds and retention."""

# BEGIN use production configuration binding and the installed native factory.
from copy import deepcopy
from types import SimpleNamespace

import torch
import warp as wp

from curobo import runtime
from curobo.content import get_task_configs_path
from curobo.types import DeviceCfg, JointState
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.state.state_robot import RobotState
from curobo._src.util.config_io import join_path, load_yaml
from curobo._src.util.warp import init_warp

from manipulation.curobo_solver_components import PhysicalGradientCore
from manipulation.planning import _bind_optimizer_constraint_config
from test_scene_query_performance import manager_fixture


def test_bound_configuration_drives_native_constraints_and_sampled_retention(
        manager_fixture, monkeypatch, tmp_path):
    """Every named native bound excludes its violating state from retention.

    The two-joint states are algebraic test data. CPU analytic kinematics
    supplies the existing manager interface; the actual native state-bound
    cost, rollout factory, evidence consumer and retention method execute.
    Effort is explicitly supplied here, so this test does not establish that
    any particular robot configuration computes inverse dynamics.
    """
    device = DeviceCfg("cpu")
    monkeypatch.setattr(wp.config, "kernel_cache_dir", str(tmp_path / "warp-cache"))
    init_warp(device_cfg=device)
    monkeypatch.setattr(runtime, "cuda_streams", False)
    # Native registration passes a device argument. Keep the shared CPU
    # fixture's stream substitution local to this actual factory invocation.
    monkeypatch.setattr(
        "curobo._src.rollout.cost_manager.cost_manager_robot.create_cuda_stream_pair",
        lambda *args: (None, None))

    optimizer = deepcopy(load_yaml(join_path(
        get_task_configs_path(), "trajopt/lbfgs_bspline_trajopt.yml")))
    metrics = deepcopy(load_yaml(join_path(get_task_configs_path(), "metrics_base.yml")))
    optimizer["rollout"]["cost_cfg"]["process_mode"] = "reference_objective"
    for config, mode in ((optimizer, "trajectory_optimizer"), (metrics, "trajectory_metrics")):
        constraint = config["rollout"]["constraint_cfg"]
        # The declared two-joint fixture has no scene or robot sphere assets.
        # Production scene ownership makes the same removal before binding.
        constraint.pop("scene_collision_cfg")
        constraint.pop("self_collision_cfg")
        constraint["process_mode"] = mode
    original_cost = deepcopy(optimizer["rollout"]["cost_cfg"])
    original_bound = deepcopy(metrics["rollout"]["constraint_cfg"]["cspace_cfg"])

    _bind_optimizer_constraint_config(optimizer, metrics)

    optimizer_cost = optimizer["rollout"]["cost_cfg"]
    metric_cost = metrics["rollout"]["cost_cfg"]
    optimizer_bound = optimizer["rollout"]["constraint_cfg"]["cspace_cfg"]
    metric_bound = metrics["rollout"]["constraint_cfg"]["cspace_cfg"]
    assert optimizer_cost == metric_cost == original_cost
    assert optimizer_bound == metric_bound == original_bound
    # Native manager initialization binds and mutates its own configuration.
    # Nested weights must therefore remain independent between the managers.
    assert metric_cost["cspace_cfg"]["weight"] is not optimizer_cost["cspace_cfg"]["weight"]
    assert optimizer_bound["weight"] is not metric_bound["weight"]
    assert optimizer_bound["activation_distance"] == [0.] * 5

    with manager_fixture("trajectory_optimizer", required=(.5,), capacity=4) as fixture:
        fixture.configuration.set_scene_constraints(False)
        rollout = RobotRolloutCfg.create_with_component_types(
            deepcopy(optimizer["rollout"]), robot_cfg=None, device_cfg=device,
            cost_manager_config_instance_type=fixture.configuration)
        torch.testing.assert_close(rollout.cost_cfg.cspace_cfg.weight,
            torch.tensor(original_cost["cspace_cfg"]["weight"]), rtol=0, atol=0)
        torch.testing.assert_close(rollout.constraint_cfg.cspace_cfg.weight,
            torch.tensor(original_bound["weight"]), rtol=0, atol=0)
        torch.testing.assert_close(rollout.cost_cfg.cspace_cfg.activation_distance,
            torch.tensor(original_cost["cspace_cfg"]["activation_distance"]), rtol=0, atol=0)

        names = ["first_joint", "second_joint"]
        bounds = torch.tensor([[-20., -20.], [20., 20.]])
        limits = JointLimits(names, bounds, bounds, bounds, bounds, bounds, device)
        robot = SimpleNamespace(tool_frames=["tool"], config=object(), total_spheres=0,
            get_self_collision_config=lambda: SimpleNamespace(num_spheres=0))
        transition = SimpleNamespace(robot_model=robot, interpolation_steps=1,
            action_dim=2, teleport_mode=False, joint_names=names,
            get_state_bounds=lambda: limits)
        manager = rollout.constraint_cfg.class_type(device)
        manager.initialize_from_config(rollout.constraint_cfg, transition)
        position, velocity, acceleration, jerk, effort = [torch.zeros(6, 3, 2) for _ in range(5)]
        # Batch zero satisfies every bound. Each other row violates one
        # different native state field at one physical sample and joint.
        for row, values in enumerate((position, velocity, acceleration, jerk, effort), start=1):
            values[row, 1, 0] = 21.
        joints = JointState(position=position, velocity=velocity,
            acceleration=acceleration, jerk=jerk, joint_names=names, dt=torch.ones(6))
        state = RobotState(joint_state=joints,
            cuda_robot_model_state=fixture.kinematics(joints), joint_torque=effort)
        constraints = manager.compute_costs(state)
        native_bound = constraints.values[constraints.names.index("cspace")]
        assert (native_bound.flatten(1).sum(-1) > 0).tolist() == [False] + [True] * 5
        assert manager.get_cost("cspace").config.activation_distance.eq(0).all()

        ordinary = CostCollection()
        declared_cost = torch.arange(1., 7.)[:, None, None].expand(6, 3, 1).clone()
        ordinary.add(declared_cost, "declared_ordinary_objective")
        trajectories = SimpleNamespace(costs_and_constraints=CostsAndConstraints(
            costs=ordinary, constraints=constraints))
        feasible, _, objective = manager.optimizer_evidence(trajectories)
        # Bound penalties affect eligibility. The candidate ranking retains
        # exactly the ordinary cost, including for an ineligible candidate.
        torch.testing.assert_close(objective, declared_cost.sum((1, 2)), rtol=0, atol=0)
        owner = SimpleNamespace(config=SimpleNamespace(num_problems=6, num_particles=1),
            action_horizon=3, action_dim=2, _candidate_action=torch.zeros_like(position),
            _candidate_objective=torch.full((6,), float("inf")),
            _candidate_valid=torch.zeros(6, dtype=torch.bool))
        PhysicalGradientCore._retain_sampled_candidate(owner, position, feasible, objective)
        expected = [True, False, False, False, False, False]
        assert feasible.tolist() == owner._candidate_valid.tolist() == expected
        assert owner._candidate_objective[0] == declared_cost[0].sum()
        assert torch.isinf(owner._candidate_objective[1:]).all()
    assert not torch.cuda.is_initialized()
# END native state-bound evidence reaches the actual production retention path.
