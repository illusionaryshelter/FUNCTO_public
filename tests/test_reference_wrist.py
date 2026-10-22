"""Actual ZYZ tool motion through bounded enrichment and native refinement.

The mechanism is Rz(q1) Ry(q2) Rz(q3), with a fixed tool offset. The declared
IK fixture returns mathematically equivalent configurations; SciPy independently
checks the actual returned polynomial. No task-scene qualification is asserted.
"""
# BEGIN a physical tool calculation replaces synthetic snapshot scores.
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints
from curobo._src.solver.manager_goal import GoalManager
from curobo.types import DeviceCfg, JointState

from manipulation.cubic_trajectory import CubicTrajectoryStep
from manipulation.curobo_solver_components import bind_solver_components
from manipulation.physical_joint_domain import CubicJointDomain, JointPositionDomain
from manipulation.planning import _CuroboPlanner, PlanningBudget
from manipulation.planning_problem import quaternion_multiply, quaternion_rotate
from manipulation.reference_tracking import physical_time_mean_weights
import test_manipulation_planning as fixtures


class WristRollout:
    """Differentiable declared FK, with production sampling and physical domain."""
    def __init__(self, planner):
        self.planner, self.domain = planner, planner.physical_domain
        self.step = CubicTrajectoryStep(18, 8, .0625, device="cpu")
        self.action_horizon, self.action_dim = 18, 3
        self.action_bound_lows, self.action_bound_highs = self.domain.lower, self.domain.upper
        self.constraint_manager = None
        self.times = torch.arange(self.step.horizon)[None]*.0625
        self.weights = physical_time_mean_weights(self.times, torch.tensor([self.step.horizon]))
        self.offset = torch.tensor([.09, .04, .08])
        self.target_quaternion = self.quaternion(planner.start.position)
        self.target_position = quaternion_rotate(self.target_quaternion, self.offset[None])

    @staticmethod
    def quaternion(q):
        axes = q.new_tensor([[0., 0., 1.], [0., 1., 0.], [0., 0., 1.]])
        parts = torch.cat((torch.cos(q[..., None]/2), torch.sin(q[..., None]/2)*axes), -1)
        return quaternion_multiply(quaternion_multiply(parts[..., 0, :], parts[..., 1, :]), parts[..., 2, :])

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def evaluate_action(self, action, **kwargs):
        self.domain.validate(action)
        joint = JointState.zeros((len(action), self.step.horizon, 3), DeviceCfg("cpu"),
                                 joint_names=list(self.planner.joint_names))
        self.step.sample(action, self.planner.start, torch.zeros(len(action), dtype=torch.int32),
                         joint, valid_count=self.step.horizon)
        orientation = self.quaternion(joint.position)
        position = quaternion_rotate(orientation, self.offset.expand(*orientation.shape[:-1], 3))
        conjugate = self.target_quaternion*orientation.new_tensor([1., -1., -1., -1.])
        relative = quaternion_multiply(orientation, conjugate)
        angle = 2*torch.atan2(torch.linalg.vector_norm(relative[..., 1:], dim=-1), relative[..., 0].abs())
        # Match the native complete-tool scalar: position uses one half, and
        # orientation uses the true SO(3) angle squared. The analytic backend
        # exposes that scalar directly to Torch's differentiation engine.
        point = (.5*10000.*(position-self.target_position).square().sum(-1)+500.*angle.square())*self.weights
        ordinary, constraints = CostCollection(), CostCollection()
        ordinary.add(point[..., None], "continuous_reference")
        constraints.add(point[..., None]*0., "declared_empty_task_constraints")
        return SimpleNamespace(state=SimpleNamespace(joint_state=joint), actions=action,
            costs_and_constraints=CostsAndConstraints(costs=ordinary, constraints=constraints))

    compute_metrics_from_action = evaluate_action

    def snapshot(self, actions):
        with torch.no_grad():
            evaluated = self.evaluate_action(actions)
        points = evaluated.costs_and_constraints.costs.values[0][..., 0]
        rows = [{"seed_index": i, "sampled_scope_feasible": True,
            "fixed_objective": float(row.sum()), "optimization_objective": float(row.sum()),
            "reference_times_s": self.times[0].tolist(), "reference_point_costs": row.tolist(),
            "controls": actions[i].tolist()} for i, row in enumerate(points)]
        return {"geometry_scope": "analytic_declared_mechanism", "seeds": rows}, list(actions.clone())


def wrist_planner():
    qa = np.array([-np.pi/2, .6, -np.pi/2], dtype=np.float32)
    qb = -qa
    pools = [np.array([qa if i < 2 else qb]) for i in range(18)]
    planner, _, _ = fixtures.ReferenceSeedDiversityTests.planner_for(
        pools[:6], [np.ones(1, bool)]*6, seed_count=1)
    planner.budget = PlanningBudget(knots=16, ik_seeds=1, trajectory_seeds=1,
        trajectory_iterations=100, interpolation_steps=8, maximum_trajectory_dt=.0625,
        interpolation_dt=.0625)
    planner.start = JointState.from_position(torch.from_numpy(qa)[None], joint_names=list(planner.joint_names))
    planner.start.velocity = torch.zeros_like(planner.start.position)
    planner.device_cfg = SimpleNamespace(to_device=lambda x: torch.as_tensor(x, dtype=torch.float32))
    planner.JointState, planner.seed_parameter = JointState, torch.zeros((1, 1))
    bounds = torch.tensor([[-4., 4.], [-10., 10.], [-50., 50.], [-500., 500.]])[:, :, None].expand(-1, -1, 3)
    planner.physical_domain = CubicJointDomain(bounds, planner.start.position, planner.start.velocity, .5, 18)
    planner.position_domain = JointPositionDomain(planner.physical_domain.lower, planner.physical_domain.upper)
    planner.formal_clock = {"duration": 10., "knot_dt": .5, "native_dt": .0625,
                            "interpolation_dt": .0625, "valid_count": 161}
    limits = SimpleNamespace(joint_names=list(planner.joint_names), velocity=bounds[1])
    planner._verification_native = SimpleNamespace(get_cost=lambda _: SimpleNamespace(config=SimpleNamespace(joint_limits=limits)))
    calls = []

    def solve(goal, **kwargs):
        # Original observations reproduce the published branch-switch example.
        # Additional calls solve the same pose using their supplied exact IK
        # warm start, so their outcome follows a numerical input, not time/name.
        assert np.float32(goal) == np.float32(float(planner.seed_parameter.item())*planner.formal_clock["duration"])
        value = pools[len(calls)][0] if len(calls) < 18 else kwargs["seed_config"][0, 0].numpy()
        calls.append(float(planner.seed_parameter.item()))
        return SimpleNamespace(js_solution=SimpleNamespace(position=torch.from_numpy(value.copy())[None, None]),
            success=torch.ones((1, 1), dtype=torch.bool), position_error=torch.zeros((1, 1)),
            rotation_error=torch.zeros((1, 1)))

    planner.ik.solve_pose = solve
    rollout = WristRollout(planner)
    device = DeviceCfg("cpu")
    stage = LBFGSOpt(LBFGSOptCfg(device_cfg=device, num_iters=100, inner_iters=25,
        history=10, line_search_scale=[0., .1, .5, 1.], initial_step_scale=.001,
        use_cuda_kernel_step_direction=False, use_cuda_kernel_line_search=False), [rollout, rollout])
    shell = SimpleNamespace(core=SimpleNamespace(device_cfg=device,
        config=SimpleNamespace(use_cuda_graph=False), optimizers=[stage], goal_registry_manager=GoalManager(device)))
    bind_solver_components(shell, planner.physical_domain)
    stage.update_num_problems(1)
    planner.trajopt = SimpleNamespace(optimizer=MultiStageOptimizer([stage]), optimizer_rollouts=[rollout],
        core=SimpleNamespace(optimizers=[stage]), action_horizon=18, action_dim=3,
        interpolation_steps=8, disable_tool_pose_tracking=Mock(), disable_joint_position_tracking=Mock(),
        _prepare_goal_buffer=Mock(return_value=(object(), True)), _update_rollout_params=Mock())
    planner.functional_constraints = SimpleNamespace(bind=lambda *a, **k: None, reset=lambda: None)
    planner._manager_type = SimpleNamespace(set_geometry_scope=lambda scope: None)
    planner._initialization_snapshot = rollout.snapshot
    planner._controls_curve = lambda action: action.clone()
    # Replace only the original fixture's no-op refinement with the real
    # production lifecycle; the native optimizer evaluates the scalar above.
    del planner._refine_reference_seeds
    return planner, rollout, calls, qa


def independent_error(planner, actions, qa):
    times = np.linspace(0., 1., 8001)
    q = planner.physical_domain.evaluate(actions, times)
    rotation = Rotation.from_euler("ZYZ", q)
    target = Rotation.from_euler("ZYZ", qa)
    angle = (rotation*target.inv()).magnitude()
    position = np.linalg.norm(rotation.apply([.09, .04, .08])-target.apply([.09, .04, .08]), axis=-1)
    return dict(maximum_rotation_rad=float(angle.max()), maximum_position_m=float(position.max()),
                mean_energy=float(np.trapezoid(.5*10000.*position**2+500.*angle**2, times)))


def test_equivalent_ik_switch_is_evaluated_and_refined_in_actual_tool_space():
    planner, rollout, calls, qa = wrist_planner()
    with patch("curobo._src.runtime.cuda_event_timers", False):
        result, success = planner.reference_seeds()
    record = planner.reference_initialization
    coarse = torch.tensor(record["coarse"]["seeds"][0]["controls"])
    before, after = independent_error(planner, coarse, qa), independent_error(planner, result[0, 0], qa)
    print({"coarse": before, "refined": after, "additional_times": calls[18:],
           "selection": record["enrichment"]["selected_sources"]})
    assert before["maximum_rotation_rad"] > 3.13
    assert after["mean_energy"] < before["mean_energy"]
    assert after["maximum_rotation_rad"] < before["maximum_rotation_rad"]
    assert after["maximum_position_m"] < before["maximum_position_m"]
    assert len(calls) <= 20 and success == [1]*len(calls)
    planner.physical_domain.validate(result)
    np.testing.assert_allclose(planner.physical_domain.evaluate(result[0, 0], [0.])[0], qa, atol=0, rtol=0)
    np.testing.assert_allclose(planner.physical_domain.evaluate(result[0, 0], [0.], derivative=1), 0., atol=0, rtol=0)
    assert result.shape == (1, 1, 18, 3) and planner.formal_clock["duration"] == 10.
# END success flags alone never establish the between-query tool motion.
