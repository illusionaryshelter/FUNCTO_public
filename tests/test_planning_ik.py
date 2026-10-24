"""Functional IK ordering and original task/geometry evidence without GPU allocation."""

# BEGIN one functional ordering preserves measured failures and unknown verdicts.
from unittest.mock import Mock

import numpy as np
import pytest

from manipulation.planning_ik import rank_ik_candidates


def report(**changes):
    return {"task_satisfied": True, "model_valid": True,
            "static_model_query": "evaluated", "task_violations": {}, "native_violations": {},
            "scene_components": {}, "candidate_source": "event_allowed_set_ik", **changes}


def select(values, reports, *, count=1, tolerance=1e-5):
    values = np.asarray(values, float)
    return rank_ik_candidates(values, reports, np.zeros((1, values.shape[1])),
                              count=count, joint_tolerance=tolerance).tolist()


def test_task_then_model_evidence_precedes_numerical_distance():
    rows = [report(task_satisfied=False, task_violations={"coordinate:0": .1}),
            report(model_valid=False, native_violations={"cspace": .1}), report()]
    assert select([[.01], [.1], [.8]], rows, count=3) == [2, 1, 0]


def test_allowed_set_membership_does_not_require_solver_success_flag():
    rows = [report(allowed_set_ik_success=False), report(allowed_set_ik_success=True)]
    assert select([[.1], [2.]], rows) == [0]


def test_duplicate_representative_uses_original_functional_evidence():
    rows = [report(task_satisfied=False, task_violations={"event": .1}), report(), report()]
    assert select([[.1], [.100002], [.8]], rows, count=3) == [1, 2]


def test_nearest_of_supplied_predecessors_breaks_equal_qualification():
    values = np.array([[.1], [2.1]])
    selected = rank_ik_candidates(values, [report(), report()], np.array([[0.], [2.05]]),
        count=1, joint_tolerance=1e-5)
    assert selected.tolist() == [1]


def test_nonfinite_values_produce_no_invented_candidate():
    assert select([[np.nan], [np.inf]], [report(), report()], count=2) == []


def test_task_and_model_verdicts_are_explicit():
    with pytest.raises(ValueError, match="explicit booleans"):
        select([[0.]], [{}])


@pytest.mark.parametrize("component", ["collision/robot/self", "collision/entity/background",
    "contact/declared_relation", "initial_support/declared_relation", "gripper/controller"])
def test_registered_definite_failure_ranks_after_qualified_member(component):
    rows = [report(model_valid=False, scene_components={component: {"maximum": .01}}), report()]
    assert select([[.01], [.8]], rows) == [1]


@pytest.mark.parametrize("unknown", [
    {"scene_components": {"unresolved/robot/self": {"maximum": 1.}}},
    {"scene_components": {"unknown/forbidden/entity": {"maximum": 1.}}},
    {"scene_components": {"conditional/hand_geometry": {"maximum": 1.}}},
    {"scene_components": {"relationship/missing_release/entity/0": {"maximum": 1.}}},
    {"scene_components": {"scene_update/background/robot": {"maximum": 1.}}},
    {"scene_components": {"collision/robot/self": {"maximum": float("nan")}}},
    {"native_violations": {"cspace": float("inf")}},
    {},
])
def test_unestablished_geometry_is_distinct_from_registered_collision(unknown):
    known = report(model_valid=False, scene_components={"collision/robot/self": {"maximum": .01}})
    unresolved = report(model_valid=False, **unknown)
    assert select([[.01], [.8]], [known, unresolved]) == [1]
    assert select([[.01], [.8]], [unresolved, report()]) == [1]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_task_values_remain_unestablished(value):
    unresolved = report(task_satisfied=False, task_violations={"coordinate:0": value})
    known = report(task_satisfied=False, task_violations={"coordinate:0": .1})
    assert select([[.01], [.8]], [known, unresolved]) == [1]


def test_known_collision_and_unknown_relation_retain_original_components():
    row = report(model_valid=False, scene_components={
        "collision/robot/self": {"maximum": .01}, "relationship/missing_release/entity/0": {"maximum": 1.}})
    assert select([[.01], [.8]], [row, report(model_valid=False)]) == [1]
    assert set(row["scene_components"]) == {"collision/robot/self", "relationship/missing_release/entity/0"}
# END these proposal reports do not certify a trajectory or a later branch scene.


# BEGIN production IK screening uses the same native and online residuals.
def _static_planner(monkeypatch, *, self_failure=False, online_failure=False):
    """Exercise real planner screening without allocating native GPU IK."""
    from types import SimpleNamespace
    import torch
    from manipulation import planning

    planner = planning._ConstrainedPlanner.__new__(planning._ConstrainedPlanner)
    planner.torch = torch
    planner.joint_names = ("x", "y", "z")
    planner.budget = SimpleNamespace(ik_seeds=2, interpolation_dt=.01)
    planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
    planner.JointState = planner._RobotState = SimpleNamespace
    planner.formal_clock = {"duration": 2.}
    planner.progress_map = planning._ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    planner.problem = SimpleNamespace(tool_link="tool", numerical=SimpleNamespace(joint=1e-5),
        transform_tool_pose=lambda p, r: (p, r), transform_robot_spheres=lambda value: value,
        residuals=lambda p, r, progress, *args, **kwargs: {"process": torch.zeros_like(progress),
            # A static candidate cannot reject the trajectory's stop dynamics.
            "stop:test": torch.ones_like(progress)})
    planner.scene = SimpleNamespace(gripper=None)
    calls = {"online": [], "native": []}

    def forward(js):
        return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=["tool"],
            position=js.position[:, :, None],
            quaternion=torch.tensor([[[[1., 0., 0., 0.]]]], dtype=js.position.dtype)),
            robot_spheres=torch.zeros((1, 1, 1, 4), dtype=js.position.dtype))

    def native(state):
        q = state.joint_state.position
        calls["native"].append(q.clone())
        rejected = self_failure and float(q[0, 0, 0]) > .5
        return SimpleNamespace(names=["self_collision"], values=[q.new_full(q.shape[:2], float(rejected))])

    planner._verification_fk = SimpleNamespace(compute_kinematics=forward)
    planner._verification_native = SimpleNamespace(get_cost=lambda _: None, compute_costs=native)

    def online(scene, problem, progress_map, spheres, p, r, progress, q, names, *,
               trajectory_time, normalized_time, component_diagnostics):
        calls["online"].append((float(trajectory_time.item()), float(normalized_time.item())))
        rejected = online_failure and float(q[0, 0, 0]) > .5
        # BEGIN the same-call fixture distinguishes support from collision.
        if rejected:
            component_diagnostics["scene_update/background/robot"] = {
                "maximum": 1., "time_s": float(trajectory_time.item()), "progress": float(progress.item())}
        # END no extra static scene query supplies this reason.
        return torch.full_like(progress, float(rejected))

    monkeypatch.setattr(planning, "_scene_boundary_evaluate", online)
    return planner, calls


def test_finite_functional_proposals_all_receive_the_original_checks(monkeypatch):
    planner, calls = _static_planner(monkeypatch)
    selected, reports = planner._rank_ik_candidates(
        [[.1, 0., 0.], [.7, 0., 0.]], [True, False], [[0., 0., 0.]], .4, (("pose_events", 0),))
    assert selected.tolist() == [0, 1]
    assert all(row["task_satisfied"] and row["model_valid"] for row in reports)
    assert not reports[1]["allowed_set_ik_success"]
    assert calls["online"] == [(.8, .4)]*2
    assert len(calls["native"]) == 2


def test_static_functional_screening_uses_the_common_hard_contract(monkeypatch):
    planner, _ = _static_planner(monkeypatch)
    received = []
    original = planner.problem.residuals

    def residuals(*args, **kwargs):
        received.append(kwargs)
        return original(*args, **kwargs)

    planner.problem.residuals = residuals
    planner._rank_ik_candidates([[.1, 0., 0.]], [False], [[0., 0., 0.]], .4,
                               (("pose_events", 0),))
    assert len(received) == 1
    assert received[0]["direction_mode"] == "point"
    assert "for_acceptance" not in received[0]


def test_functional_static_pool_keeps_allowed_pose_when_nominal_center_ik_failed(monkeypatch):
    planner, _ = _static_planner(monkeypatch)
    selected, reports = planner._rank_ik_candidates(
        [[.7, 0., 0.], [.1, 0., 0.]], [True, False], [[0., 0., 0.]], .4,
        (("pose_events", 0),))
    assert selected.tolist() == [1, 0]
    assert reports[1]["task_satisfied"] and reports[1]["model_valid"]
    assert not reports[1]["allowed_set_ik_success"]


@pytest.mark.parametrize("failure", ["self_failure", "online_failure"])
def test_either_native_self_or_online_residual_rejects_static_model(monkeypatch, failure):
    planner, calls = _static_planner(monkeypatch, **{failure: True})
    selected, reports = planner._rank_ik_candidates(
        [[.1, 0., 0.], [.7, 0., 0.]], [True, True], [[0., 0., 0.]], .4, (("pose_events", 0),))
    assert selected.tolist() == [0, 1]
    assert reports[0]["model_valid"] and not reports[1]["model_valid"]
    # BEGIN rejection reasons reuse the original two state queries.
    assert len(calls["online"]) == len(calls["native"]) == 2
    assert reports[1]["task_violations"] == {}
    if failure == "self_failure":
        assert reports[1]["native_violations"] == {"self_collision": 1.}
        assert reports[1]["scene_components"] == {}
    else:
        assert reports[1]["native_violations"] == {}
        assert reports[1]["scene_components"] == {"scene_update/background/robot": {
            "maximum": 1., "time_s": .8, "progress": .4}}
    # END the failure category cannot change rank or static acceptance.
# END source-mesh audit margins are not inputs to production IK screening.


# BEGIN invalid proposals are explained before geometry access.
def test_nonfinite_ik_proposal_reports_columns_without_fk_or_scene_query(monkeypatch):
    planner, calls = _static_planner(monkeypatch)
    selected, reports = planner._rank_ik_candidates(
        [[float("nan"), 0., float("inf")]], [False], [[0., 0., 0.]], .4, (("pose_events", 0),))
    assert selected.size == 0
    assert reports[0]["invalid_joint_indices"] == [0, 2]
    assert not reports[0]["task_satisfied"] and not reports[0]["model_valid"]
    assert calls == {"online": [], "native": []}
# END no invented task/model evidence accompanies an invalid joint state.


# BEGIN mixed finite/non-finite task entries retain the actual rejection reason.
def test_task_reason_does_not_hide_negative_infinity_behind_finite_maximum(monkeypatch):
    import torch
    planner, calls = _static_planner(monkeypatch)
    planner.problem.residuals = lambda *args, **kwargs: {"process": torch.tensor([[float("-inf"), 0.]])}
    _, reports = planner._rank_ik_candidates([[.1, 0., 0.]], [True], [[0., 0., 0.]], .4, (("pose_events", 0),))
    assert not reports[0]["task_satisfied"]
    assert reports[0]["task_violations"] == {"process": float("-inf")}
    assert len(calls["online"]) == len(calls["native"]) == 1
# END no scalar maximum conceals unavailable task data.


def test_true_pose_pool_uses_event_clock_and_retains_only_the_existing_two_ik_outputs():
    from types import SimpleNamespace
    import torch
    from manipulation import planning
    from manipulation.planning_problem import PoseEventRequirement

    planner = planning._ConstrainedPlanner.__new__(planning._ConstrainedPlanner)
    # BEGIN the functional IK fixture declares the same submitted-work ledger.
    planner.budget = SimpleNamespace(ik_seeds=2, max_validation_points=32, maximum_trajectory_solves=1)
    planner.request_work = {"ik_queries": 0}
    planner.request_work_baseline = {"ik_queries": 0}
    # END both original event calls consume the common candidate-solve count.
    planner.joint_names = ("x", "y", "z")
    # BEGIN the fixture supplies actual typed pose declarations to the proposal API.
    planner.problem = SimpleNamespace(start_progress=0., end_progress=1., supports=(), coordinate_events=(), pose_events=[
        PoseEventRequirement(s, np.eye(4), np.eye(4), .01, .02, "declared event") for s in (.4, .4, .8)])
    # END complete poses have exactly one nominal yaw proposal.
    planner.problem.goal_requirements_at = lambda phase: tuple(
        ("pose_events", index) for index, goal in enumerate(planner.problem.pose_events) if goal.progress == phase)
    planner.trajopt = SimpleNamespace(action_horizon=6)
    planner.progress_map = planning._ProgressMap(np.array([0., .2, 1.]), np.array([0., .4, 1.]))
    planner.formal_clock = {"duration": 3.}
    planner.seed_parameter = torch.zeros((1, 1), dtype=torch.float64)
    planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
    planner.start = SimpleNamespace(position=torch.zeros((1, 3), dtype=torch.float64), dt=torch.tensor([0.]))
    from manipulation.physical_joint_domain import CubicJointDomain
    physical_bounds = torch.tensor([[-2., 2.], [-2., 2.], [-20., 20.], [-200., 200.]])
    planner.physical_domain = CubicJointDomain(physical_bounds[:, :, None].expand(-1, -1, 3),
        planner.start.position, torch.zeros((1, 3)), 3./10., 6)
    planner._functional_goal = lambda target: target
    planner._manager_type = SimpleNamespace(bind_ik_goals=Mock())
    planner.check_request = Mock()
    queries = []

    def solve(target, **kwargs):
        queries.append((target.progress, float(planner.seed_parameter.item()), kwargs))
        return SimpleNamespace(js_solution=SimpleNamespace(position=torch.tensor(
            [[[.1, 0., 0.], [.7, 0., 0.]]], dtype=torch.float64)), success=torch.ones((1, 2), dtype=torch.bool),
            goalset_index=torch.zeros((1, 2, 1), dtype=torch.long))

    planner.ik = SimpleNamespace(config=SimpleNamespace(num_seeds=2), solve_pose=solve, get_active_js=lambda value: value,
        disable_tool_pose_tracking=Mock(), enable_tool_pose_tracking=Mock())
    planner._rank_ik_candidates = lambda values, success, previous, u, goals: (
        np.array([1, 0]), [{"static": "first"}, {"static": "second"}])
    check_search = Mock()
    pool = planner._build_goal_ik_candidates(torch.zeros((1, 1, 6, 3), dtype=torch.float64),
                                             check_search=check_search)
    assert len(queries) == len(pool) == 2
    assert [row[0] for row in queries] == [.4, .8]
    assert queries[0][1] == .2 and queries[1][1] == pytest.approx(11/15)
    assert all(row[2]["return_seeds"] == 2 and row[2]["seed_config"].shape == (1, 1, 3) for row in queries)
    assert pool[.2]["goal_ids"] == [["pose_events", 0], ["pose_events", 1]]
    assert pool[.2]["time_s"] == pytest.approx(.6)
    assert pool[.2]["candidates"][0]["joint_position"] == [.7, 0., 0.]
    assert len(pool[.2]["candidates"]) == 2
    assert planner.ik.disable_tool_pose_tracking.call_count == 2
    assert planner.ik.enable_tool_pose_tracking.call_count == 2
    assert planner.request_work["ik_queries"] == 4
    assert all(float(row[2]["current_state"].dt) == 0. for row in queries)
    check_search.assert_called()
# END mocked queries test orchestration, not CUDA geometry or actual IK convergence.
