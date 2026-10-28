"""CPU checks of native manager APIs and original static proposal evidence.

Installed cuRobo managers and cost collections expose the actual event-cost
append method without invoking the parent's CUDA stream synchronization.
Analytical FK supplies independent values and Jacobians; original task and
scene-query consumers process them. No library function is replaced.
"""

# BEGIN native pose-prior integration tests using maintained manager APIs.
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]


from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.metrics import CostCollection
from curobo._src.types.device_cfg import DeviceCfg
from manipulation.planning import _ConstrainedPlanner, _ProgressMap
from manipulation.planning_ik import rank_ik_candidates as original_rank
from manipulation.planning_problem import MotionPhase, PlanningProblem, PoseEventRequirement, PosePath
from manipulation.preference_planner.native_setup import NativeSetup, _ik_manager_type, rank_pose_ik_candidates
from manipulation.preference_planner.objective import pose_prior_cost
from manipulation.preference_planner.pose_roles import bind_pose_objectives

torch.set_num_threads(1)


def make_problem(roles=("prior",), *, targets=None, progresses=None, yaw=None):
    """Four analytical coordinates encode xyz translation and world z rotation."""
    path = PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))),
                    "analytical source reference")
    events = []
    for index, _ in enumerate(roles):
        target = np.eye(4)
        target[0, 3] = 0. if targets is None else targets[index]
        events.append(PoseEventRequirement(.5 if progresses is None else progresses[index],
            np.eye(4), target, .01, .1, {"origin": "declared analytical target"},
            yaw_axis=None if yaw is None else np.array([0., 0., 1.]),
            yaw_tolerance_rad=yaw))
    problem = PlanningProblem(("x", "y", "z", "theta"), np.zeros(4), np.array([.03, 0., 0., 0.]),
        "tool", "world", np.eye(4), np.eye(4), (MotionPhase("reference", path),),
        pose_events=tuple(events))
    declarations = [{"goal_id": ["pose_events", index], "role": role,
                     "scales": {name: (.04 if name == "position" else .2)
                                for name in item.component_tolerances},
                     "scale_source": "Explicit analytical normalization for native API checks"}
                    for index, (item, role) in enumerate(zip(events, roles))]
    return bind_pose_objectives(problem, declarations)


def analytical_state(q):
    """Independent xyz/z-axis kinematics and its world geometric Jacobian."""
    angle = q[..., 3]
    zero = torch.zeros_like(angle)
    quaternion = torch.stack((torch.cos(angle / 2), zero, zero, torch.sin(angle / 2)), -1)[..., None, :]
    jacobian = q.new_zeros((*q.shape[:2], 1, 6, 4))
    jacobian[..., 0, 0] = jacobian[..., 1, 1] = jacobian[..., 2, 2] = 1.
    jacobian[..., 5, 3] = 1.
    tool = SimpleNamespace(tool_frames=["tool"], position=q[..., None, :3], quaternion=quaternion)
    native = SimpleNamespace(tool_jacobians=jacobian)
    return SimpleNamespace(joint_state=SimpleNamespace(position=q, shape=q.shape),
        tool_poses=tool, cuda_robot_model_state=native,
        robot_spheres=q.new_zeros((*q.shape[:2], 1, 4)))


def manager(problem, mode, *, weight=7.5):
    """Instantiate the real installed manager with an empty native registry."""
    budget = SimpleNamespace(constraint_weight=weight)
    cfg_type = _ik_manager_type(problem, budget, clock={"duration": 2.},
                               seed_parameter=torch.tensor([[.5]], dtype=torch.float64))
    device = DeviceCfg("cpu", dtype=torch.float64)
    configuration = cfg_type.create({"process_mode": mode}, device_cfg=device)
    instance = configuration.class_type(device)
    robot_model = SimpleNamespace(get_self_collision_config=lambda: None, tool_frames=["tool"],
                                  total_spheres=0, compute_jacobian=False)
    transition = SimpleNamespace(robot_model=robot_model, joint_names=problem.joint_names)
    instance.initialize_from_config(configuration, transition)
    assert isinstance(instance, RobotCostManager)
    assert robot_model.compute_jacobian
    return instance, cfg_type


def costs(instance, q):
    # cuRobo's parent compute_costs synchronizes CUDA streams even with an
    # empty registry and CPU DeviceCfg. Exercise the actual task append
    # method directly; production calls it immediately after that parent.
    result = instance._add_event_costs(analytical_state(q), CostCollection())
    assert isinstance(result, CostCollection)
    return dict(zip(result.names, result.values))


def test_real_native_optimizer_attracts_prior_without_metric_rejection():
    problem = make_problem()
    optimizer, optimizer_cfg = manager(problem, "seed_optimizer")
    metric, metric_cfg = manager(problem, "seed_metrics")
    goals = (("pose_events", 0),)
    optimizer_cfg.bind_ik_goals(goals)
    metric_cfg.bind_ik_goals(goals)
    q = torch.tensor([[[.3, 0., 0., .2]]], dtype=torch.float64, requires_grad=True)
    optimized = costs(optimizer, q)
    measured = costs(metric, q)
    assert set(optimized) == {"event_allowed_set", "pose_prior"}
    assert set(measured) == {"event_allowed_set"}
    torch.testing.assert_close(measured["event_allowed_set"], torch.zeros((1, 1, 1), dtype=q.dtype))
    torch.testing.assert_close(optimized["event_allowed_set"], measured["event_allowed_set"])
    expected = 7.5 * ((q[..., 0] / .04).square() + (q[..., 3] / .2).square())
    torch.testing.assert_close(optimized["pose_prior"][..., 0], expected)
    derivative, = torch.autograd.grad(optimized["pose_prior"].sum(), q)
    torch.testing.assert_close(derivative[..., 0], 2 * 7.5 * q[..., 0] / .04**2)
    torch.testing.assert_close(derivative[..., 3], 2 * 7.5 * q[..., 3] / .2**2)


@pytest.mark.parametrize("mode", ["seed_optimizer", "seed_metrics"])
def test_required_event_keeps_original_allowed_set_units_and_normalization(mode):
    problem = make_problem(("required",))
    current, current_cfg = manager(problem, mode)
    current_cfg.bind_ik_goals((("pose_events", 0),))
    q = torch.tensor([[[.3, 0., 0., .2]]], dtype=torch.float64)
    actual = costs(current, q)
    # The declared .01 m / .1 rad allowed set leaves .29 m / .1 rad
    # violations. The original optimizer normalizes and squares them;
    # original seed_metrics retains their unsquared physical quantities.
    expected = 7.5 * ((.29 / .01)**2 + (.1 / .1)**2 if mode == "seed_optimizer" else .29 + .1)
    torch.testing.assert_close(actual["event_allowed_set"], q.new_full((1, 1, 1), expected))
    if mode == "seed_optimizer":
        torch.testing.assert_close(actual["pose_prior"], torch.zeros_like(actual["pose_prior"]))


def test_mixed_same_time_roles_use_original_hard_set_and_selected_prior():
    problem = make_problem(("prior", "required"), targets=(0., .3))
    instance, configuration = manager(problem, "seed_optimizer")
    q = torch.tensor([[[.3, 0., 0., .05]]], dtype=torch.float64)
    configuration.bind_ik_goals((("pose_events", 0), ("pose_events", 1)))
    combined = costs(instance, q)
    torch.testing.assert_close(combined["event_allowed_set"], torch.zeros_like(combined["event_allowed_set"]))
    assert combined["pose_prior"].item() > 0.
    configuration.bind_ik_goals((("pose_events", 1),))
    required_only = costs(instance, q)
    assert required_only["pose_prior"].item() == 0.


def test_free_yaw_remains_free_in_the_real_native_prior_gradient():
    problem = make_problem(yaw=np.pi)
    instance, configuration = manager(problem, "seed_optimizer")
    configuration.bind_ik_goals((("pose_events", 0),))
    q = torch.tensor([[[0., 0., 0., 1.2]]], dtype=torch.float64, requires_grad=True)
    values = costs(instance, q)
    assert values["pose_prior"].item() == 0.
    derivative, = torch.autograd.grad(values["pose_prior"].sum(), q)
    torch.testing.assert_close(derivative, torch.zeros_like(q), rtol=0., atol=0.)


def test_rebind_clears_selected_goals_and_consumes_the_new_local_collection():
    full = make_problem(("prior", "prior"), targets=(.8, .2), progresses=(.1, .5))
    instance, configuration = manager(full, "seed_optimizer")
    configuration.bind_ik_goals((("pose_events", 0),))
    full.start_progress = .4
    block, record = full.bound_motion_block()
    assert record["goal_identities"]["pose_events"] == [1]
    configuration.bind_request(block, SimpleNamespace(constraint_weight=3.))
    q = torch.tensor([[[.3, 0., 0., 0.]]], dtype=torch.float64)
    assert costs(instance, q) == {}
    configuration.bind_ik_goals((("pose_events", 0),))
    values = costs(instance, q)
    assert values["pose_prior"].item() == pytest.approx(3. * ((.3 - .2) / .04)**2)


def test_unbound_or_ordinary_manager_adds_no_event_terms():
    problem = make_problem()
    q = torch.zeros((1, 1, 4), dtype=torch.float64)
    instance, configuration = manager(problem, "seed_optimizer")
    assert costs(instance, q) == {}
    ordinary, ordinary_cfg = manager(problem, "")
    ordinary_cfg.bind_ik_goals((("pose_events", 0),))
    assert costs(ordinary, q) == {}


def test_append_preserves_the_native_cost_collection_and_its_existing_terms():
    instance, configuration = manager(make_problem(), "seed_optimizer")
    configuration.bind_ik_goals((("pose_events", 0),))
    q = torch.tensor([[[.3, 0., 0., 0.]]], dtype=torch.float64)
    collection = CostCollection()
    native = q.new_full((1, 1, 1), 17.)
    collection.add(native, "existing_native_component")
    result = instance._add_event_costs(analytical_state(q), collection)
    assert result is collection
    assert result.names == ["existing_native_component", "event_allowed_set", "pose_prior"]
    assert result.values[0] is native


class AnalyticalScene:
    """A declared scalar scene component exercises the real query dispatcher."""

    required_progress_anchors = ()
    gripper = None

    def __init__(self, calls, component):
        self.calls, self.component = calls, component

    def evaluate(self, spheres, p, r, progress, q, names, *, trajectory_time,
                 geometry_evidence_out, geometry_scope, components_out, fk_session=None):
        self.calls["scene"] += 1
        value = torch.zeros_like(progress)
        if self.component is not None:
            value = (q[..., 0] > .6).to(q.dtype)
            components_out[self.component] = value
        return value


def static_planner(*, role="prior", component=None, count=3):
    """Run NativeSetup ranking through original tensor task and scene APIs."""
    planner = NativeSetup()
    planner.problem = make_problem((role,), targets=(.8,))
    planner.torch = torch
    planner.joint_names = planner.problem.joint_names
    planner.budget = SimpleNamespace(ik_seeds=count, interpolation_dt=.01, constraint_weight=7.5)
    planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
    planner.JointState = planner._RobotState = SimpleNamespace
    planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    planner.formal_clock = {"duration": 2.}
    calls = {"fk": 0, "scene": 0, "native": 0}
    planner.scene = AnalyticalScene(calls, component)

    def forward(js):
        calls["fk"] += 1
        return analytical_state(js.position)

    def native(state):
        calls["native"] += 1
        return SimpleNamespace(names=["cspace"], values=[torch.zeros_like(state.joint_state.position[..., 0])])

    planner._verification_fk = SimpleNamespace(compute_kinematics=forward)
    planner._verification_native = SimpleNamespace(compute_costs=native)
    return planner, calls


def test_static_ranking_reuses_each_fk_once_and_prefers_prior_before_continuity():
    planner, calls = static_planner(count=2)
    candidates = [[.1, 0., 0., 0.], [.7, 0., 0., 0.]]
    selected, reports = planner._rank_ik_candidates(candidates, [True, False],
        np.zeros((1, 4)), .5, (("pose_events", 0),))
    assert selected.tolist() == [1, 0]
    assert calls == {"fk": 2, "scene": 2, "native": 2}
    assert all(row["task_satisfied"] and row["model_valid"] for row in reports)
    assert not reports[1]["allowed_set_ik_success"]
    for candidate, report in zip(candidates, reports):
        p = torch.tensor([[candidate[:3]]], dtype=torch.float64)
        q = torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64)
        progress = torch.tensor([[.5]], dtype=torch.float64)
        expected = 7.5 * pose_prior_cost(planner.problem, p, q, progress, progress * 2.,
                                       goal_ids=(("pose_events", 0),)).sum().item()
        assert report["pose_prior_cost"] == expected


@pytest.mark.parametrize("component", ["collision/registered/entity", "scene_update/background/robot"])
def test_original_static_qualification_precedes_lower_prior_cost(component):
    planner, calls = static_planner(component=component, count=2)
    selected, reports = planner._rank_ik_candidates(
        [[.1, 0., 0., 0.], [.7, 0., 0., 0.]], [False, True],
        np.zeros((1, 4)), .5, (("pose_events", 0),))
    assert selected.tolist() == [0, 1]
    assert reports[1]["pose_prior_cost"] < reports[0]["pose_prior_cost"]
    assert not reports[1]["model_valid"]
    assert component in reports[1]["scene_components"]
    assert calls == {"fk": 2, "scene": 2, "native": 2}


def test_static_duplicates_nonfinite_inputs_and_seed_limit_keep_original_semantics():
    planner, calls = static_planner(count=2)
    candidates = [[.1, 0., 0., 0.], [.100002, 0., 0., 0.], [.3, 0., 0., 0.],
                  [np.nan, 0., 0., 0.]]
    selected, reports = planner._rank_ik_candidates(candidates, [True] * 4,
        np.zeros((1, 4)), .5, (("pose_events", 0),))
    assert selected.tolist() == [2, 1]
    assert reports[3]["invalid_joint_indices"] == [0]
    assert calls == {"fk": 3, "scene": 3, "native": 3}


def test_required_only_ranking_matches_original_evidence_and_continuity_order():
    planner, _ = static_planner(role="required", count=3)
    candidates = np.array([[.1, 0., 0., 0.], [.7, 0., 0., 0.], [.8, 0., 0., 0.]])
    previous = np.array([[.11, 0., 0., 0.], [.79, 0., 0., 0.]])
    selected, reports = planner._rank_ik_candidates(candidates, [False] * 3,
        previous, .5, (("pose_events", 0),))
    expected = original_rank(candidates, reports, previous, count=3,
                             joint_tolerance=planner.problem.numerical.joint)
    np.testing.assert_array_equal(selected, expected)
    assert all(row["pose_prior_cost"] == 0. for row in reports)


def test_equal_prior_cost_keeps_nearest_predecessor_then_source_index():
    reports = [{"task_satisfied": True, "model_valid": True, "static_model_query": "evaluated",
                "pose_prior_cost": 3., "task_violations": {}, "native_violations": {}, "scene_components": {}}
               for _ in range(3)]
    selected = rank_pose_ik_candidates([[.1], [.9], [1.1]], reports, [[0.], [1.]],
                                      count=3, joint_tolerance=1e-5)
    # The first predecessor distance is .1; .9 is slightly closer in binary
    # floating-point and keeps the original numerical distance interpretation.
    assert selected.tolist() == original_rank([[.1], [.9], [1.1]], reports, [[0.], [1.]],
        count=3, joint_tolerance=1e-5).tolist()


def test_mixin_uses_original_constructor_cleanup_on_initial_binding_error():
    class Planner(NativeSetup, _ConstrainedPlanner):
        pass

    class BindingError(Exception):
        pass

    class Validator:
        def validate_planning_binding(self, problem, gripper):
            raise BindingError("Analytical binding failure before native allocation")

    prior = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    assert Planner._initialize is NativeSetup._initialize
    with pytest.raises(BindingError, match="before native allocation"):
        Planner(None, SimpleNamespace(gripper=None), None, None, "cpu", robot_validator=Validator())
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == prior
# END existing runtime code and installed library files remain unchanged.
