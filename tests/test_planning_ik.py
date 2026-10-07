"""CPU counterexamples for IK policies without native solver allocation."""

# BEGIN qualification, whole-chain continuity and explicit recovery are separate contracts.
from itertools import product

import numpy as np
import pytest

from manipulation.planning_ik import rank_ik_candidates, select_ik_chains


def report(**changes):
    return {"task_satisfied": True, "model_valid": True, "native_pose_ik_success": True,
            "candidate_source": "native_ik", "environment_clearance_m": .001,
            "self_clearance_m": .002, "joint_limit_margin_fraction": .2, **changes}


def select(values, reports, *, count=1, tolerance=1e-5, query_kind="reference"):
    values = np.asarray(values, float)
    return rank_ik_candidates(values, reports, np.zeros((1, values.shape[1])),
                              count=count, joint_tolerance=tolerance, query_kind=query_kind).tolist()


def chains(values, reports=None, *, count=2, times=None, velocity=None, tolerance=1e-5):
    pools = [np.asarray(pool, float) for pool in values]
    joints = pools[0].shape[1]
    if reports is None:
        reports = [[report() for _ in pool] for pool in pools]
    return select_ik_chains(pools, reports, np.zeros((1, joints)),
        np.arange(1, len(pools)+1, dtype=float) if times is None else times,
        np.array([[-1.]*joints, [1.]*joints]) if velocity is None else velocity,
        count=count, joint_tolerance=tolerance)


def test_extra_clearance_cannot_buy_a_large_continuity_loss():
    assert select([[.001], [2.]], [report(environment_clearance_m=.00005), report()]) == [0]


@pytest.mark.parametrize("first,second", [
    (report(task_satisfied=False, environment_clearance_m=100.), report(model_valid=False)),
    (report(model_valid=False, environment_clearance_m=100.), report()),
    (report(native_pose_ik_success=False, environment_clearance_m=100.), report()),
])
def test_same_query_qualification_precedes_any_continuity_or_clearance_advantage(first, second):
    assert select([[0.], [2.]], [first, second]) == [1]


def test_functional_allowed_set_does_not_require_nominal_center_ik_success():
    reports = [report(native_pose_ik_success=False), report()]
    assert select([[.1], [2.]], reports, query_kind="reference") == [1]
    assert select([[.1], [2.]], reports, query_kind="functional") == [0]


@pytest.mark.parametrize("first", [
    report(self_clearance_m=.0001, joint_limit_margin_fraction=.9),
    report(joint_limit_margin_fraction=.01),
])
def test_quality_breaks_only_equal_continuity_ties(first):
    assert select([[-1.], [1.]], [first, report()]) == [1]


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), -float("inf")])
def test_unknown_distance_never_masquerades_as_infinite_clearance(missing):
    assert select([[-1.], [1.]], [report(environment_clearance_m=missing),
                                 report(environment_clearance_m=-.01)]) == [1]


def test_duplicate_representative_retains_the_better_evidence_before_deduplication():
    values = [[0.], [.25e-5], [.5], [float("nan")]]
    reports = [report(model_valid=False), report(), report(), report(environment_clearance_m=100.)]
    assert select(values, reports, count=4) == [1, 2]
    assert select(values, reports, count=1) == [1]


def test_nearest_previous_state_precedes_extra_clearance():
    values = np.array([[1.], [1.5]])
    selected = rank_ik_candidates(values, [report(environment_clearance_m=100.), report()],
                                  [[0.], [1.6]], count=2, joint_tolerance=1e-5,
                                  query_kind="reference")
    assert selected.tolist() == [1, 0]


def test_all_broken_values_produce_no_candidate_instead_of_a_fabricated_state():
    assert select([[float("nan")], [float("inf")]], [report(), report()], count=2) == []
    assert chains([[[float("nan")]], [[float("inf")]]]) == []


def test_validity_must_be_explicit_not_a_missing_default():
    with pytest.raises(ValueError, match="explicit booleans"):
        select([[0.]], [{"task_satisfied": True, "model_valid": True}])
    with pytest.raises(ValueError, match="explicit candidate_source"):
        chains([[[.1]]], [[{key: value for key, value in report().items()
                           if key != "candidate_source"}]])
    with pytest.raises(ValueError, match="cannot claim native"):
        chains([[[.1]]], [[report(candidate_source="carried_previous")]])


def test_complete_chain_looks_past_the_nearest_first_query_and_keeps_two_histories():
    selected = chains([[[0.], [4.]], [[5.]]])
    assert [row["candidate_indices"] for row in selected] == [[1, 0], [0, 0]]
    assert selected[0]["selection_cost"]["normalized_velocity_energy"] == pytest.approx(8.5)
    assert selected[1]["selection_cost"]["normalized_velocity_energy"] == pytest.approx(12.5)
    assert all(row["reference_success"] and not row["recovery_queries"] for row in selected)


def test_chain_success_cannot_be_replaced_by_a_stationary_failed_reference():
    selected = chains([[[.5], [0.]], [[1.], [0.]]],
        [[report(), report(native_pose_ik_success=False, environment_clearance_m=100.)]]*2)
    assert len(selected) == 1
    assert selected[0]["candidate_indices"] == [0, 0]
    assert selected[0]["reference_success"]


# BEGIN missing task/model evidence cannot discard a successful reference proposal.
def test_task_and_source_screening_are_independent_of_reference_pose_success():
    selected = chains([[[.1], [.5]]],
        [[report(model_valid=False), report(native_pose_ik_success=False)]])
    assert len(selected) == 1 and selected[0]["candidate_indices"] == [0]
    assert selected[0]["reference_success"]
    assert selected[0]["selection_cost"]["static_model_failure_count"] == 1


def test_static_failures_rank_chains_without_erasing_the_only_native_solution():
    selected = chains([[[.1], [.5]], [[.2], [.6]]],
        [[report(task_satisfied=False, model_valid=False), report()]]*2, count=4)
    assert selected[0]["candidate_indices"] == [1, 1]
    assert len(selected) == 4
    only = chains([[[.1]], [[.2]]], [[report(task_satisfied=False, model_valid=False)]]*2)
    assert only[0]["candidate_indices"] == [0, 0]
    assert only[0]["selection_cost"]["static_task_failure_count"] == 2
    assert only[0]["selection_cost"]["static_model_failure_count"] == 2
# END whole-curve task and scene acceptance is not asserted by these proposals.


def test_failed_and_carried_queries_cannot_supply_reference_anchors():
    selected = chains([[[.1], [.9]], [[.1]]],
        [[report(native_pose_ik_success=False), report(native_pose_ik_success=False)],
         [report(native_pose_ik_success=False, candidate_source="carried_previous", model_valid=False)]])
    assert selected == []


def test_velocity_bounds_scale_named_joints_before_comparing_postures():
    selected = chains([[[1., 0.], [0., .5]]], velocity=[[-10., -1.], [5., 1.]])
    assert selected[0]["candidate_indices"] == [0]
    assert selected[0]["selection_cost"]["normalized_velocity_energy"] == pytest.approx(.005)


def test_nonuniform_physical_clock_uses_velocity_integral_not_query_count():
    selected = chains([[[1.]], [[2.]]], times=[.1, 2.])
    expected = (.1*(1/.1)**2+1.9*(1/1.9)**2)/2
    assert selected[0]["selection_cost"]["normalized_velocity_energy"] == pytest.approx(expected)


def test_tiny_extra_clearance_does_not_change_the_smoother_complete_chain():
    selected = chains([[[.1], [1.]], [[.2], [1.1]]],
        [[report(), report(self_clearance_m=.002000551)]]*2)
    assert selected[0]["candidate_indices"] == [0, 0]


def test_numerical_duplicates_do_not_manufacture_distinct_whole_chains():
    selected = chains([[[.1], [.100002]], [[.2], [.200002]]])
    assert len(selected) == 1
    selected = chains([[[.1], [.10004]], [[.2], [.20004]]])
    assert len(selected) == 2


def test_bounded_dynamic_program_matches_independent_exhaustive_small_graph():
    rng = np.random.default_rng(20261005)
    pools = rng.normal(size=(4, 3, 2))
    times = np.array([.3, .7, 1.4, 2.])
    bounds = np.array([[-2., -3.], [1., 2.]])
    scale = np.array([2., 3.])
    expected = []
    for path in product(range(3), repeat=4):
        postures = np.stack([pools[query, index] for query, index in enumerate(path)])
        changes = np.diff(np.r_[np.zeros((1, 2)), postures], axis=0)
        intervals = np.diff(np.r_[0., times])
        energy = np.sum(changes**2/scale**2/intervals[:, None])/(times[-1]*2)
        expected.append((float(energy), path))
    expected.sort()
    actual = chains(pools, times=times, velocity=bounds, count=3)
    assert [tuple(row["candidate_indices"]) for row in actual] == [path for _, path in expected[:3]]
    np.testing.assert_allclose([row["selection_cost"]["normalized_velocity_energy"] for row in actual],
                               [energy for energy, _ in expected[:3]], rtol=1e-14)


@pytest.mark.parametrize("times", [[0., 1.], [1., 1.], [2., 1.], [1., float("nan")]])
def test_invalid_clock_is_rejected_instead_of_retimed(times):
    with pytest.raises(ValueError, match="increasing physical times"):
        chains([[[.1]], [[.2]]], times=times)
# END these candidates are initialization evidence, never a checked continuous trajectory.


# BEGIN production IK screening uses the same native and online residuals.
def _static_planner(monkeypatch, *, self_failure=False, online_failure=False):
    """Exercise real planner screening without allocating native GPU IK."""
    from types import SimpleNamespace
    import torch
    from manipulation import planning

    planner = planning._CuroboPlanner.__new__(planning._CuroboPlanner)
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


def test_actual_task_and_online_checks_rank_reference_ik_without_hardening_its_pose(monkeypatch):
    planner, calls = _static_planner(monkeypatch)
    selected, reports = planner._rank_ik_candidates(
        [[.1, 0., 0.], [.7, 0., 0.]], [True, False], [[0., 0., 0.]], .4)
    assert selected.tolist() == [0, 1]
    assert all(row["task_satisfied"] and row["model_valid"] for row in reports)
    assert not reports[1]["native_pose_ik_success"]
    assert calls["online"] == [(.8, .4), (.8, .4)]
    assert len(calls["native"]) == 2


def test_functional_static_pool_keeps_allowed_pose_when_nominal_center_ik_failed(monkeypatch):
    planner, _ = _static_planner(monkeypatch)
    selected, reports = planner._rank_ik_candidates(
        [[.7, 0., 0.], [.1, 0., 0.]], [True, False], [[0., 0., 0.]], .4,
        (("pose_events", 0),))
    assert selected.tolist() == [1, 0]
    assert reports[1]["task_satisfied"] and reports[1]["model_valid"]
    assert not reports[1]["native_pose_ik_success"]


@pytest.mark.parametrize("failure", ["self_failure", "online_failure"])
def test_either_native_self_or_online_residual_rejects_static_model(monkeypatch, failure):
    planner, calls = _static_planner(monkeypatch, **{failure: True})
    selected, reports = planner._rank_ik_candidates(
        [[.1, 0., 0.], [.7, 0., 0.]], [True, True], [[0., 0., 0.]], .4)
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
        [[float("nan"), 0., float("inf")]], [False], [[0., 0., 0.]], .4)
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
    _, reports = planner._rank_ik_candidates([[.1, 0., 0.]], [True], [[0., 0., 0.]], .4)
    assert not reports[0]["task_satisfied"]
    assert reports[0]["task_violations"] == {"process": float("-inf")}
    assert len(calls["online"]) == len(calls["native"]) == 1
# END no scalar maximum conceals unavailable task data.


def test_true_pose_pool_uses_event_clock_and_retains_only_the_existing_two_ik_outputs():
    from types import SimpleNamespace
    import torch
    from manipulation import planning

    planner = planning._CuroboPlanner.__new__(planning._CuroboPlanner)
    planner.budget = SimpleNamespace(ik_seeds=2)
    planner.joint_names = ("x", "y", "z")
    planner.problem = SimpleNamespace(start_progress=0., pose_events=[
        SimpleNamespace(progress=.4), SimpleNamespace(progress=.4), SimpleNamespace(progress=.8)])
    planner.problem.goal_requirements_at = lambda phase: tuple(
        ("pose_events", index) for index, goal in enumerate(planner.problem.pose_events) if goal.progress == phase)
    planner.trajopt = SimpleNamespace(action_horizon=6)
    planner.progress_map = planning._ProgressMap(np.array([0., .2, 1.]), np.array([0., .4, 1.]))
    planner.formal_clock = {"duration": 3.}
    planner.seed_parameter = torch.zeros((1, 1), dtype=torch.float64)
    planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
    planner.start = SimpleNamespace(position=torch.zeros((1, 3), dtype=torch.float64))
    from manipulation.physical_joint_domain import CubicJointDomain
    planner.physical_domain = CubicJointDomain(torch.full((3,), -2.), torch.full((3,), 2.),
        planner.start.position, torch.zeros((1, 3)), 3./10., 6)
    planner._functional_goal = lambda target: target
    queries = []

    def solve(target, **kwargs):
        queries.append((target.progress, float(planner.seed_parameter.item()), kwargs))
        return SimpleNamespace(js_solution=SimpleNamespace(position=torch.tensor(
            [[[.1, 0., 0.], [.7, 0., 0.]]], dtype=torch.float64)), success=torch.ones((1, 2), dtype=torch.bool))

    planner.ik = SimpleNamespace(solve_pose=solve, get_active_js=lambda value: value)
    planner._rank_ik_candidates = lambda values, success, previous, u, goals: (
        np.array([1, 0]), [{"static": "first"}, {"static": "second"}])
    pool = planner._build_goal_ik_candidates(torch.zeros((1, 1, 6, 3), dtype=torch.float64))
    assert len(queries) == len(pool) == 2
    assert [row[0] for row in queries] == [.4, .8]
    assert queries[0][1] == .2 and queries[1][1] == pytest.approx(11/15)
    assert all(row[2]["return_seeds"] == 2 and row[2]["seed_config"].shape == (1, 1, 3) for row in queries)
    assert pool[.2]["goal_ids"] == [["pose_events", 0], ["pose_events", 1]]
    assert pool[.2]["time_s"] == pytest.approx(.6)
    assert pool[.2]["candidates"][0]["joint_position"] == [.7, 0., 0.]
    assert len(pool[.2]["candidates"]) == 2
# END mocked queries test orchestration, not CUDA geometry or actual IK convergence.
