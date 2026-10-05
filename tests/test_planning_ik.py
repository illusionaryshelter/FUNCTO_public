"""CPU counterexamples for IK priorities without native solver allocation."""

# BEGIN feasibility, missing evidence and finite diversity are independent contracts.
import numpy as np
import pytest

from manipulation.planning_ik import rank_ik_candidates


def report(**changes):
    return {"task_satisfied": True, "source_valid": True,
            "environment_clearance_m": .001, "self_clearance_m": .002,
            "joint_limit_margin_fraction": .2, **changes}


def select(values, reports, *, count=1, tolerance=1e-5):
    values = np.asarray(values, float)
    return rank_ik_candidates(values, reports, np.zeros((1, values.shape[1])),
                              count=count, joint_tolerance=tolerance).tolist()


def test_environment_clearance_precedes_large_continuity_advantage():
    assert select([[.001], [2.]], [report(environment_clearance_m=.00005), report()]) == [1]


@pytest.mark.parametrize("first,second", [
    (report(task_satisfied=False, environment_clearance_m=100.), report(source_valid=False)),
    (report(source_valid=False, environment_clearance_m=100.), report()),
    (report(self_clearance_m=.0001, joint_limit_margin_fraction=.9), report()),
    (report(joint_limit_margin_fraction=.01), report()),
])
def test_higher_priority_cannot_be_bought_with_lower_priority_quality(first, second):
    assert select([[0.], [2.]], [first, second]) == [1]


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), -float("inf")])
def test_unknown_distance_never_masquerades_as_infinite_clearance(missing):
    assert select([[0.], [2.]], [report(environment_clearance_m=missing),
                                report(environment_clearance_m=-.01)]) == [1]


def test_duplicate_representative_retains_the_better_evidence_before_deduplication():
    values = [[0.], [.25e-5], [.5], [float("nan")]]
    reports = [report(source_valid=False), report(), report(), report(environment_clearance_m=100.)]
    assert select(values, reports, count=4) == [1, 2]
    assert select(values, reports, count=1) == [1]


def test_nearest_previous_branch_breaks_only_equal_quality_ties():
    values = np.array([[1.], [1.5]])
    selected = rank_ik_candidates(values, [report(), report()], [[0.], [1.6]],
                                  count=2, joint_tolerance=1e-5)
    assert selected.tolist() == [1, 0]


def test_all_broken_values_produce_no_candidate_instead_of_a_fabricated_state():
    assert select([[float("nan")], [float("inf")]], [report(), report()], count=2) == []


def test_validity_must_be_explicit_not_a_missing_default():
    with pytest.raises(ValueError, match="explicit booleans"):
        select([[0.]], [{"task_satisfied": True}])
# END selection offers finite initializers, never whole-trajectory feasibility.


# BEGIN production IK integration consumes explicit fake static-query contracts on CPU.
def _static_planner(monkeypatch, *, complete=True, self_failure=False, release_missing=False,
                    future_hand=False, clearance_limitations=()):
    """Exercise real planner screening, without pretending to run native robot IK."""
    from types import SimpleNamespace
    import torch
    from manipulation import planning

    planner = planning._CuroboPlanner.__new__(planning._CuroboPlanner)
    planner.torch = torch
    planner.joint_names = ("x", "y", "z")
    planner.budget = SimpleNamespace(ik_seeds=2, interpolation_dt=.01)
    planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
    planner.JointState = SimpleNamespace
    planner.formal_clock = {"duration": 2.}
    planner.progress_map = planning._ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    planner.problem = SimpleNamespace(tool_link="tool", numerical=SimpleNamespace(joint=1e-5),
        transform_tool_pose=lambda p, r: (p, r), transform_robot_spheres=lambda value: value,
        residuals=lambda p, r, progress, *args, **kwargs: {"process": torch.zeros_like(progress),
            # A static API placeholder cannot certify or reject real stop dynamics.
            "stop:test": torch.ones_like(progress)})
    planner.trajopt = SimpleNamespace(transition_model=SimpleNamespace(
        action_bound_lows=torch.full((3,), -1., dtype=torch.float64),
        action_bound_highs=torch.full((3,), 1., dtype=torch.float64)))
    calls = {"source": [], "self": [], "coverage": []}

    def forward(js):
        return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=["tool"],
            position=js.position[:, :, None],
            quaternion=torch.tensor([[[[1., 0., 0., 0.]]]], dtype=js.position.dtype)),
            robot_spheres=torch.zeros((1, 1, 1, 4), dtype=js.position.dtype))

    def own(q, names, times, *, diagnostics):
        calls["self"].append(q.clone())
        rejected = self_failure and float(q[0, 0, 0]) > .5
        diagnostics.update(worst_state={"minimum_required_clearance_m": -.01 if rejected else .02},
                           unmodelled_geometry_names=["finger"] if future_hand else [])
        return torch.full_like(times, float(rejected))

    def coverage(q, names, times):
        calls["coverage"].append(q.clone())
        return [{"attempt_allowed": True, "passed": not future_hand}]

    scene = SimpleNamespace(check_robot_self_collision=own, check_robot_coverage=coverage)
    scene.clone_for_rollout = lambda: SimpleNamespace(**scene.__dict__)
    planner._verification_scene = scene
    planner._verification_fk = SimpleNamespace(compute_kinematics=forward)

    def source(scene, problem, progress_map, spheres, p, r, progress, q, names, *, trajectory_time,
               normalized_time, source_geometry, diagnostics, source_components_out):
        assert source_geometry
        calls["source"].append((float(trajectory_time.item()), float(normalized_time.item())))
        diagnostics.update(unpermitted_clearance={"minimum_recorded_distance_m": float(q[0, 0, 0])*.001,
            "complete_query": complete, "distance_complete": False,
            "limitations": ["observed_tsdf_local_components_not_global_solid_distance", *clearance_limitations]},
            unmodelled_geometry_names=["finger"] if future_hand else [])
        source_components_out["source_world_collision"] = torch.zeros_like(progress)
        source_components_out["unknown/release_anchor/object/0"] = torch.full_like(progress, float(release_missing))
        source_components_out["unknown/future_hand"] = torch.full_like(progress, float(future_hand))
        return source_components_out["source_world_collision"]

    monkeypatch.setattr(planning, "_scene_boundary_evaluate", source)
    return planner, calls


def test_actual_task_and_source_checks_rank_reference_ik_without_hardening_its_pose(monkeypatch):
    planner, calls = _static_planner(monkeypatch)
    selected, reports = planner._rank_ik_candidates(
        [[.1, 0., 0.], [.7, 0., 0.]], [True, False], [[0., 0., 0.]], .4)
    assert selected.tolist() == [1, 0]
    assert all(row["task_satisfied"] and row["source_valid"] for row in reports)
    assert not reports[1]["native_pose_ik_success"]
    assert "stop:test" not in reports[1]["task_residuals"]
    assert "stop_dynamics" in reports[1]["not_checked"]
    assert calls["source"] == [(.8, .4), (.8, .4)]
    assert len(calls["self"]) == len(calls["coverage"]) == 2


def test_static_source_self_rejection_cannot_be_overruled_by_environment_clearance(monkeypatch):
    planner, _ = _static_planner(monkeypatch, self_failure=True)
    selected, reports = planner._rank_ik_candidates(
        [[.1, 0., 0.], [.7, 0., 0.]], [True, True], [[0., 0., 0.]], .4)
    assert selected.tolist() == [0, 1]
    assert not reports[1]["source_valid"] and reports[1]["environment_clearance_m"] is None


@pytest.mark.parametrize("complete,missing_release", [(False, False), (True, True)])
def test_incomplete_distance_or_unbound_release_cannot_supply_a_ranked_margin(monkeypatch, complete, missing_release):
    planner, _ = _static_planner(monkeypatch, complete=complete, release_missing=missing_release)
    _, reports = planner._rank_ik_candidates([[.7, 0., 0.]], [True], [[0., 0., 0.]], .4)
    assert reports[0]["environment_clearance_m"] is None
    if missing_release:
        assert not reports[0]["source_valid"]


def test_future_hand_limitations_survive_modeled_static_acceptance(monkeypatch):
    planner, _ = _static_planner(monkeypatch, future_hand=True)
    _, reports = planner._rank_ik_candidates([[.7, 0., 0.]], [True], [[0., 0., 0.]], .4)
    assert reports[0]["source_valid"]
    assert not reports[0]["full_robot_coverage_certified"]
    assert reports[0]["unmodelled_geometry_names"] == ["finger"]


@pytest.mark.parametrize("limitation", ["finite_patch_unpermitted_subset_distance_unmeasured",
                                        "surface_permission_not_resolved_by_finite_patch"])
def test_unmeasured_unpermitted_subset_cannot_be_ranked_by_another_distant_pair(monkeypatch, limitation):
    """A completed collision check need not measure every unpermitted distance."""
    planner, _ = _static_planner(monkeypatch, clearance_limitations=(limitation,))
    _, reports = planner._rank_ik_candidates([[.7, 0., 0.]], [True], [[0., 0., 0.]], .4)
    assert reports[0]["source_valid"]
    assert reports[0]["environment_clearance_m"] is None
    assert limitation in reports[0]["source_clearance"]["limitations"]


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
