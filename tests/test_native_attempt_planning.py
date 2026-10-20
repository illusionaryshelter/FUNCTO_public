"""CPU regressions for nominal objectives and basic-motion acceptance.

The actual cuRobo collections and rollout dispatch execute on CPU. Analytic
FK replaces only the GPU geometry kernel; these tests do not claim robot
reachability, task success, or physical collision accuracy.
"""

# BEGIN native objective/constraint separation and bounded cubic regression.
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.cubic_seed import CubicSeed, cubic_position_report
from manipulation.planning import (
    MotionPlanningError, PlanningBudget, _CuroboPlanner, _ProgressMap,
    _minimum_cost_candidate, _process_manager_type,
)


def test_native_rollout_keeps_nonzero_support_task_cost_without_rejection():
    from curobo.types import DeviceCfg, JointState
    from curobo._src.rollout.rollout_robot import RobotRollout
    from curobo._src.state.state_robot import RobotState

    device = DeviceCfg("cpu")
    q = torch.full((2, 3, 3), .1, requires_grad=True)
    joints = JointState.from_position(q, joint_names=["x", "y", "z"])
    joints.velocity, joints.dt = torch.zeros_like(q), torch.full((2,), .5)

    def kinematics(joints):
        position = joints.position
        rotation = position.new_zeros((*position.shape[:2], 1, 4))
        rotation[..., 0] = 1.
        return SimpleNamespace(
            tool_poses=SimpleNamespace(tool_frames=["tool"], position=position[..., None, :], quaternion=rotation),
            tool_jacobians=position.new_zeros((*position.shape[:2], 1, 6, 3)),
            robot_spheres=torch.cat((position, torch.full_like(position[..., :1], .01)), -1)[..., None, :])

    problem = SimpleNamespace(tool_link="tool", mechanisms=(), stop_progress=(),
        transform_tool_pose=lambda p, r: (p, r), transform_robot_spheres=lambda value: value,
        residuals=lambda p, *args, **kwargs: {"support:0": torch.relu(p[..., 2]-.001)},
        optimizer_preferences=lambda p, *args, **kwargs: (p-1.).square().sum(-1))
    scene = SimpleNamespace(gripper=None, collision=0.,
        optimizer_preferences=lambda progress, *args: torch.zeros_like(progress))
    scene.clone_for_rollout = lambda: scene
    # BEGIN one mixed scene query retains geometric and non-geometric names.
    scene.query_count = 0
    def scene_components(spheres, p, r, progress, joints, *args, **kwargs):
        scene.query_count += 1
        return {"collision/robot/background": torch.full_like(progress, scene.collision),
                "collision/robot/object": torch.zeros_like(progress),
                "gripper/controller": torch.relu(joints[..., 0]-.2),
                "unsupported/release_anchor/object/0": torch.zeros_like(progress)}
    scene.components = scene_components
    # END all components use the same ordered regular and anchor samples.
    model = SimpleNamespace(tool_frames=["tool"], config=object(), total_spheres=1,
                            get_self_collision_config=lambda: None)
    transition = SimpleNamespace(joint_names=["x", "y", "z"], robot_model=model, interpolation_steps=1)
    state = RobotState(joint_state=joints, cuda_robot_model_state=kinematics(joints))
    with patch("curobo.kinematics.Kinematics", return_value=SimpleNamespace(compute_kinematics=kinematics)), \
         patch("curobo._src.rollout.cost_manager.cost_manager_robot.synchronize_cuda_streams"), \
         patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU test must not initialize CUDA")):
        factory = _process_manager_type(problem, scene,
            _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), np.array([0., 1.]),
            PlanningBudget(constraint_weight=1.), clock={"duration": 1.}, seed_parameter=torch.zeros((1, 1)))
        config = lambda mode: factory.create({"process_mode": mode}, device_cfg=device)
        rollout = RobotRollout.__new__(RobotRollout)
        rollout.config = SimpleNamespace(cost_cfg=config("trajectory_cost"),
            constraint_cfg=config("trajectory_constraint"), hybrid_cost_constraint_cfg=None)
        rollout._num_particles_goal = None
        for target, cfg in (("cost_manager", rollout.config.cost_cfg),
                            ("constraint_manager", rollout.config.constraint_cfg)):
            manager = cfg.class_type(device)
            manager.initialize_from_config(cfg, transition)
            setattr(rollout, target, manager)
        result = rollout._compute_costs_and_constraints_impl(state)
        assert result.costs.names == ["nominal_task"]
        assert result.constraints.names == ["collision/robot/background", "collision/robot/object",
            "gripper/controller", "unsupported/release_anchor/object/0"]
        assert scene.query_count == 1
        assert bool(result.get_feasible(sum_horizon=True).all())
        objective = result.get_sum_cost_and_constraint(sum_horizon=True)
        assert bool((objective > 0).all())
        objective.sum().backward()
        assert bool(torch.isfinite(q.grad).all()) and bool((q.grad.abs() > 0).any())
        # The same ordinary scene collision remains a physical rejection.
        scene.collision = .02
        collision = rollout._compute_costs_and_constraints_impl(state)
        assert not bool(collision.get_feasible(sum_horizon=True).any())

        # BEGIN component splitting preserves aggregate values and derivatives.
        # Two geometry terms and two fixed/controller terms share one query;
        # compare the native collection against their former summed formula.
        captured = {}
        def mixed_components(spheres, p, r, progress, joints, *args, **kwargs):
            terms = {"collision/robot/background": joints[..., 0].square(),
                     "collision/robot/object": 2*joints[..., 1].square(),
                     "gripper/controller": 3*joints[..., 2].square(),
                     "unsupported/release_anchor/object/0": torch.ones_like(progress)}
            captured["old_sum"] = sum(terms.values()).sum()
            return terms
        scene.components = mixed_components
        mixed = rollout._compute_costs_and_constraints_impl(state)
        separated_sum = sum(value.sum() for value in mixed.constraints.values)
        torch.testing.assert_close(separated_sum, captured["old_sum"])
        separated_gradient, = torch.autograd.grad(separated_sum, q, retain_graph=True)
        summed_gradient, = torch.autograd.grad(captured["old_sum"], q)
        torch.testing.assert_close(separated_gradient, summed_gradient)
        # END no extra query or new physical threshold is needed to classify.


def test_candidate_selection_uses_total_nominal_cost_and_rejects_nonfinite_values():
    assert _minimum_cost_candidate([80., 2.], [True, True]) == 1
    assert _minimum_cost_candidate([.1, 2.], [False, True]) == 1
    assert _minimum_cost_candidate([np.nan, 2.], [True, True]) == 1
    assert _minimum_cost_candidate([1., np.inf], [True, True]) == 0
    with pytest.raises(MotionPlanningError, match="no finite objective"):
        _minimum_cost_candidate([np.nan, np.inf], [True, True])


def test_native_result_accepts_task_cost_but_keeps_joint_and_finite_checks():
    # BEGIN selection consumes real finite curves before comparing task scores.
    from test_manipulation_planning import selection_fixture
    states = np.zeros((2, 9, 3))
    planner, native = selection_fixture(states, states, [10., 1.])
    # END nominal task error remains a preference, not an execution veto.
    _, report = planner.select_candidate(native)
    assert report["accepted"] and report["selected_seed_index"] == 1
    assert report["native_metrics"]["rollout"]["costs"]["nominal_task"] == pytest.approx(1.)
    native.metrics.costs_and_constraints.constraints.values[0][1] = 1.
    _, report = planner.select_candidate(native)
    assert report["accepted"] and report["selected_seed_index"] == 0
    native.metrics.costs_and_constraints.costs.values[0][0] = float("nan")
    _, report = planner.select_candidate(native)
    assert not report["accepted"]


def test_goal_free_native_result_keeps_missing_goalset_when_selecting_a_seed():
    from manipulation.planning import _trajectory_candidate
    from test_manipulation_planning import PlannerCandidateSelectionTests
    result = PlannerCandidateSelectionTests.native_candidates(native_violations=(0., 0.))
    # Run the installed native metric processor. An empty convergence
    # collection must produce no pose goalset, exactly as in the GPU solve.
    result._process_metrics()
    assert result.goalset_index is None
    selected = _trajectory_candidate(result, 1)
    assert selected.goalset_index is None
    assert selected.num_seeds == 1 and bool(selected.success.all())
    torch.testing.assert_close(selected.solution, result.solution[:, 1:2])
    torch.testing.assert_close(selected.total_cost_reshaped, result.total_cost_reshaped[:, 1:2])
    assert selected.get_interpolated_plan().position.shape == (1, 1, 5, 3)
    torch.testing.assert_close(selected.get_interpolated_plan().dt, torch.tensor([[.11]]))


def test_ompl_connection_preserves_release_history_for_later_single_state_queries():
    from unittest.mock import Mock
    from curobo.types import DeviceCfg, JointState
    from manipulation.online_geometry import FrozenESDF, FrozenPrimitives, OnlinePlanningScene
    from manipulation.planning import JointTrajectory
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import Attachment, CollisionOwner, ObservedSpace

    background = FrozenESDF(np.full((3, 3, 3), 10., np.float32), origin=[-2.]*3,
        voxel_size=2., owner_ids=("background",), source="empty ordinary model")
    owner = CollisionOwner("payload", FrozenPrimitives(primitives=[{"kind": "sphere", "dimensions": [.02]}],
        owner_ids=("payload",), source="payload model"), PosePath(np.array([0., 1.]),
        np.stack((np.eye(4), np.eye(4))), "current owner"), "current owner", True,
        local_spheres=[[0., 0., 0., .02]], sphere_materials=("body",), attachments=(
            Attachment((0., .5), [0., 0., 0.], [1., 0., 0., 0.], "hold", "release"),))
    coverage = ObservedSpace(np.zeros((3, 3, 3), bool), origin=[-2.]*3, voxel_size=2.,
        observation_ids=("analytic",), reference_frame="world")
    scene = OnlinePlanningScene(background=background, owners=(owner,), coverage=coverage,
        reference_frame="world", robot_sphere_materials=("body",),
        robot_geometry_source="analytic robot", has_gripper=False)
    body = np.array([[1., 1., 1., .02]])
    # BEGIN register the same Cartesian kinematics used by the planner below.
    # The robot body is fixed on base; x/y/z move only the tool and payload.
    # A zero-DOF native fixture would contradict the declared planning joints.
    urdf = '<robot name="cartesian"><link name="base"/>'
    for name, parent, child, axis in (("x", "base", "lx", "1 0 0"),
            ("y", "lx", "ly", "0 1 0"), ("z", "ly", "tool", "0 0 1")):
        urdf += (f'<link name="{child}"/><joint name="{name}" type="prismatic">'
            f'<parent link="{parent}"/><child link="{child}"/><axis xyz="{axis}"/>'
            '<limit lower="-2" upper="2" effort="10" velocity="10"/></joint>')
    urdf += '</robot>'
    body_pose = np.eye(4); body_pose[:3, 3] = body[0, :3]
    scene.bind_online_robot(
        robot_sphere_geometry=[{"link_name": "base", "sphere_indices": [0],
            "centers_local": body[:, :3].tolist(), "radii": body[:, 3].tolist()}],
        native_robot_model={"urdf_xml": urdf, "geometries": [{"name": "body", "frame": "base",
            "kind": "sphere", "dimensions": [.02], "T_frame_geometry": body_pose.tolist(),
            "source": "explicit fixed robot body"}]}, native_joint_names=("x", "y", "z"),
        native_self_pairs=[], robot_model={"T_reference_base": np.eye(4).tolist()})
    # END release-history checks still use the original independent owner.
    assert not scene.query_state(body, np.eye(4), .75,
        joint_position=np.zeros(3), joint_names=("x", "y", "z"))["passed"]

    def kinematics(joints):
        q = joints.position
        rotation = q.new_zeros((*q.shape[:2], 1, 4)); rotation[..., 0] = 1.
        return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=["tool"],
            position=q[..., None, :], quaternion=rotation),
            robot_spheres=q.new_tensor(body).expand(*q.shape[:2], 1, 4).clone())

    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.torch, planner.device_cfg, planner.JointState = torch, DeviceCfg("cpu"), JointState
    planner.joint_names, planner.scene = ("x", "y", "z"), scene
    planner.problem = SimpleNamespace(tool_link="tool", transform_tool_pose=lambda p, r: (p, r),
                                      transform_robot_spheres=lambda value: value)
    planner.check_request = Mock()
    planner.budget = PlanningBudget(search_seconds=1.)
    planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    planner.formal_clock = {"duration": 1.}
    planner._verification_fk = SimpleNamespace(compute_kinematics=kinematics)
    planner._RobotState = lambda **kwargs: SimpleNamespace(**kwargs,
        robot_spheres=kwargs["cuda_robot_model_state"].robot_spheres)
    limits = SimpleNamespace(joint_names=list(planner.joint_names),
                              position=torch.tensor([[-2.]*3, [2.]*3]))
    planner._verification_native = SimpleNamespace(get_cost={
        "cspace": SimpleNamespace(config=SimpleNamespace(joint_limits=limits))}.get)
    planner.seed_curve = CubicSeed(torch.zeros((1, 3)), torch.zeros((1, 3)), .1, 8)
    trajectory = JointTrajectory(planner.joint_names, np.array([0., 1.]),
        np.array([[0., 0., 0.], [.4, 0., 0.]]), np.array([[.4, 0., 0.], [.4, 0., 0.]]),
        np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    seeds, report = planner.ompl_seed(trajectory)
    assert seeds is not None, report
    assert len(report["stages"]) == 2 and report["state_queries"] > 0
    assert scene._fixed_release_poses == {}, "Full optimization must retain variable release poses"


def test_cubic_seed_fits_reference_without_certifying_it_and_preserves_entrance():
    from scipy.interpolate import CubicHermiteSpline
    from test_cubic_trajectory import independent_curve
    q0, dq0 = torch.tensor([[.2, -.3]]), torch.tensor([[.1, -.2]])
    seed = CubicSeed(q0, dq0, .4, 8)
    u = np.linspace(0., 1., 20)
    target = np.column_stack((2*u, -u))
    controls, report = seed.fit(u, target)
    assert report["method"] == "least_squares" and torch.isfinite(controls).all()
    fitted = independent_curve(q0[0].double().numpy(), dq0[0].double().numpy(),
                               .4, controls.double().numpy())
    np.testing.assert_allclose(fitted(0.), q0[0].numpy(), rtol=0., atol=1e-12)
    np.testing.assert_allclose(fitted(0., 1)/.4, dq0[0].numpy(), rtol=0., atol=1e-12)
    # Actual delivered motion, unlike a numerical guess, is checked against
    # explicitly supplied physical limits at its analytic interior extrema.
    curve = CubicHermiteSpline([0., 1.], [[0.], [0.]], [[5.], [-5.]])
    report = cubic_position_report(curve, np.array([-1.]), np.array([1.]))
    assert report["maximum_bound_excess"] == pytest.approx(.25)
# END tests exercise semantic decisions, native dispatch and independent algebra.
