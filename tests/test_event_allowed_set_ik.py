"""Analytic point-IK tests of the actual task manager and event graph."""

# BEGIN use real task declarations with a two-coordinate analytic FK fixture.
import numpy as np
import pytest
import torch

from manipulation.planning_problem import MotionPhase, PlanningProblem, PoseEventRequirement, PosePath
from manipulation.planning_ik import select_ik_chains
from test_scene_query_performance import manager_fixture


def event_problem():
    target = np.eye(4)
    target[0, 3] = .4
    reference = PosePath(np.array([0., 1.]), np.stack((np.eye(4), target)), "analytic reference")
    goal = PoseEventRequirement(.5, np.eye(4), target, .05, .02, "declared allowed set",
                                yaw_axis=[0., 0., 1.], yaw_tolerance_rad=np.pi)
    return PlanningProblem(("first_joint", "second_joint"), np.zeros(2), np.zeros(2),
        "tool", "world", np.eye(4), np.eye(4), (MotionPhase("source", reference),),
        pose_events=(goal,))


def state_at(fixture, q):
    from curobo.types import JointState
    from curobo._src.state.state_robot import RobotState
    joints = JointState.from_position(q, joint_names=list(fixture.problem.joint_names))
    joints.velocity = torch.zeros_like(q)
    joints.dt = torch.ones(1, dtype=q.dtype)
    return RobotState(joint_state=joints, cuda_robot_model_state=fixture.kinematics(joints))


def test_event_manager_drives_toward_allowed_set_and_has_zero_cost_inside(manager_fixture):
    task = event_problem()
    with manager_fixture("seed_optimizer", required=(), capacity=4, task=task) as fixture:
        fixture.configuration.bind_ik_goals((("pose_events", 0),))
        q = torch.tensor([[[.1, 0.]]], dtype=torch.float64, requires_grad=True)
        costs = fixture.manager.compute_costs(state_at(fixture, q))
        value = costs.values[costs.names.index("event_allowed_set")]
        gradient = torch.autograd.grad(value.sum(), q)[0]
        # 0.30 m error minus a 0.05 m allowed radius gives normalized 5.
        assert value.item() == pytest.approx(777.*25.)
        assert gradient[0, 0, 0] == pytest.approx(-777.*2.*5./.05)
        inside = torch.tensor([[[.43, 0.]]], dtype=torch.float64, requires_grad=True)
        costs = fixture.manager.compute_costs(state_at(fixture, inside))
        value = costs.values[costs.names.index("event_allowed_set")]
        assert value.item() == 0
        assert torch.autograd.grad(value.sum(), inside)[0].count_nonzero() == 0
        fixture.configuration.bind_ik_goals(())
        assert "event_allowed_set" not in fixture.manager.compute_costs(state_at(fixture, q)).names


def test_event_metrics_and_projection_share_the_allowed_set(manager_fixture):
    task = event_problem()
    with manager_fixture("seed_metrics", required=(), capacity=4, task=task) as fixture:
        fixture.configuration.bind_ik_goals((("pose_events", 0),))
        q = torch.tensor([[[.43, 0.]], [[.46, 0.]]], dtype=torch.float64)
        costs = fixture.manager.compute_costs(state_at(fixture, q))
        value = costs.values[costs.names.index("event_allowed_set")].flatten()/777.
        np.testing.assert_allclose(value.numpy(), [0., .01], atol=1e-15)


def candidate(**changes):
    return dict(task_satisfied=True, model_valid=True, native_pose_ik_success=False,
                candidate_source="event_allowed_set_ik", **changes)


def test_event_graph_applies_signed_speed_bounds_and_requires_every_event():
    reports = [[candidate(), candidate()], [candidate()]]
    # The close first state -.1 is unreachable with a nonnegative joint rate;
    # .2 is reachable, and its connection to .5 remains within +.4 per second.
    result = select_ik_chains([[[-.1], [.2]], [[.5]]], reports, [[0.]], [1., 2.],
        [[0.], [.4]], count=2, joint_tolerance=1e-8, query_kind="functional")
    assert len(result) == 1 and result[0]["candidate_indices"] == [1, 0]
    reports[1][0]["task_satisfied"] = False
    assert select_ik_chains([[[-.1], [.2]], [[.5]]], reports, [[0.]], [1., 2.],
        [[0.], [.4]], count=2, joint_tolerance=1e-8, query_kind="functional") == []
# END these tests establish point computation and connection necessity, not path acceptance.


# BEGIN a reference influences the finite event shortlist before velocity energy.
def test_reference_distance_precedes_shortlist_truncation_and_preserves_static_tiers():
    reports = [[candidate(), candidate()]]
    args = ([[[0.], [.8]]], reports, [[0.]], [1.], [[-1.], [1.]])
    chosen = select_ik_chains(*args, count=1, joint_tolerance=1e-8,
        query_kind="functional", reference_positions=[[.8]])
    assert chosen[0]["candidate_indices"] == [1]
    assert chosen[0]["selection_cost"]["normalized_reference_distance"] == 0.
    reports[0][1]["model_valid"] = False
    chosen = select_ik_chains(*args, count=1, joint_tolerance=1e-8,
        query_kind="functional", reference_positions=[[.8]])
    assert chosen[0]["candidate_indices"] == [0]


def test_event_shortlists_and_actual_fit_losses_belong_to_each_reference_seed():
    from types import SimpleNamespace
    from manipulation.planning import _CuroboPlanner

    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.joint_names = ("q",)
    planner.start = SimpleNamespace(position=torch.zeros((1, 1)))
    planner.formal_clock = {"duration": 2.}
    planner.problem = SimpleNamespace(numerical=SimpleNamespace(joint=1e-8))
    planner.device_cfg = SimpleNamespace(to_device=torch.as_tensor)
    planner._seed_velocity_bounds = lambda: np.array([[-1.], [1.]])
    pool = {u: {"candidates": [{"joint_position": [value], "report": candidate()}
                                for value in (-.2, .2)]} for u in (.5, 1.)}
    planner._build_goal_ik_candidates = lambda seeds: pool
    planner.initial_seed_strategies = ["first_reference", "second_reference"]
    planner.reference_seed_fits = [{}, {}]
    calls = []

    def fit(progress, reference, prior_controls, velocity_bounds, *, event_progress, event_positions):
        # This fixture isolates pairing and selection; the physical QP itself
        # is exercised by test_physical_joint_qp with exact curve constraints.
        reference = np.asarray(reference)
        loss = float(np.sqrt(np.mean((event_positions-reference[1:])**2)))
        calls.append((reference.copy(), event_positions.copy()))
        return torch.full_like(prior_controls, float(event_positions[-1, 0])), {
            "success": True, "reference_rms_joint_units": loss}

    planner.physical_domain = SimpleNamespace(fit_reference=fit)
    result = planner._fit_event_seeds(torch.zeros((1, 2, 3, 1)), np.array([0., .5, 1.]),
        [np.array([[0.], [.2], [.2]]), np.array([[0.], [-.2], [-.2]])])
    assert len(calls) == 4
    assert {row["reference_seed_index"] for row in planner.goal_ik_chains} == {0, 1}
    assert sum(row["applied"] for row in planner.goal_ik_event_fits) == 2
    torch.testing.assert_close(result[0, 0], torch.full((3, 1), .2))
    torch.testing.assert_close(result[0, 1], torch.full((3, 1), -.2))
    assert all(fit["event_fit"]["reference_rms_joint_units"] == 0. for fit in planner.reference_seed_fits)
# END shortlisted fitting is finite and leaves complete trajectory qualification separate.
