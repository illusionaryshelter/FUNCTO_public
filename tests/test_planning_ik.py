"""CPU counterexamples for IK policies without native solver allocation."""

# BEGIN native evidence, coarse-chain continuity and trajectory qualification are separate.
from itertools import product
from unittest.mock import Mock

import numpy as np
import pytest

from manipulation.planning_ik import rank_ik_candidates, select_ik_chains, select_ik_tracking_candidate


def report(**changes):
    return {"task_satisfied": True, "model_valid": True, "native_pose_ik_success": True,
            "static_model_query": "evaluated", "task_violations": {}, "native_violations": {},
            "scene_components": {},
            "candidate_source": "native_ik", "environment_clearance_m": .001,
            "self_clearance_m": .002, "joint_limit_margin_fraction": .2, **changes}


def select(values, reports, *, count=1, tolerance=1e-5, query_kind="reference"):
    values = np.asarray(values, float)
    return rank_ik_candidates(values, reports, np.zeros((1, values.shape[1])),
                              count=count, joint_tolerance=tolerance, query_kind=query_kind).tolist()


def chains(values, reports=None, *, count=2, times=None, velocity=None,
           positions=None, pose_costs=None, tolerance=1e-5):
    pools = [np.asarray(pool, float) for pool in values]
    joints = pools[0].shape[1]
    if reports is None:
        reports = [[report() for _ in pool] for pool in pools]
    return select_ik_chains(pools, reports, np.zeros((1, joints)),
        np.arange(1, len(pools)+1, dtype=float) if times is None else times,
        np.array([[-1.]*joints, [1.]*joints]) if velocity is None else velocity,
        position_limits=np.array([[-10.]*joints, [10.]*joints]) if positions is None else positions,
        pose_costs_by_query=[np.zeros(len(pool)) for pool in pools] if pose_costs is None else pose_costs,
        count=count, joint_tolerance=tolerance)


def test_extra_clearance_cannot_buy_a_large_continuity_loss():
    assert select([[.001], [2.]], [report(environment_clearance_m=.00005), report()]) == [0]


@pytest.mark.parametrize("first,second,reference,functional", [
    (report(task_satisfied=False, task_violations={"coordinate:0": .1},
            environment_clearance_m=100.),
     report(model_valid=False, native_violations={"cspace": .1}), 0, 1),
    (report(model_valid=False, native_violations={"cspace": .1}, environment_clearance_m=100.), report(), 0, 1),
    (report(native_pose_ik_success=False, environment_clearance_m=100.), report(), 1, 0),
])
def test_reference_numerical_continuity_and_functional_qualification_have_separate_priorities(first, second, reference, functional):
    assert select([[0.], [2.]], [first, second]) == [reference]
    assert select([[0.], [2.]], [first, second], query_kind="functional") == [functional]


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


def test_duplicate_representative_obeys_the_declared_numerical_or_functional_role():
    values = [[0.], [.25e-5], [.5], [float("nan")]]
    reports = [report(model_valid=False, native_violations={"cspace": .1}),
               report(), report(), report(environment_clearance_m=100.)]
    assert select(values, reports, count=4) == [0, 2]
    assert select(values, reports, count=1) == [0]
    assert select(values, reports, count=4, query_kind="functional") == [1, 2]


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
    with pytest.raises(ValueError, match="native_ik source"):
        chains([[[.1]]], [[{key: value for key, value in report().items()
                           if key != "candidate_source"}]])
    with pytest.raises(ValueError, match="native_ik source"):
        chains([[[.1]]], [[report(candidate_source="carried_previous")]])


def test_complete_chain_looks_past_the_nearest_first_query_and_keeps_two_histories():
    selected = chains([[[0.], [4.]], [[5.]]])
    assert [row["candidate_indices"] for row in selected] == [[1, 0], [0, 0]]
    assert selected[0]["selection_cost"]["normalized_velocity_energy"] == pytest.approx(8.5)
    assert selected[1]["selection_cost"]["normalized_velocity_energy"] == pytest.approx(12.5)
    assert all(row["reference_success"] and not row["interpolated_queries"] for row in selected)


def test_successful_chain_precedes_legal_approximate_alternative_without_deleting_it():
    selected = chains([[[.5], [0.]], [[1.], [0.]]],
        [[report(), report(native_pose_ik_success=False, environment_clearance_m=100.)]]*2)
    assert len(selected) == 2
    assert selected[0]["candidate_indices"] == [0, 0]
    assert selected[0]["reference_success"]
    assert selected[1]["selection_cost"]["nonconverged_candidate_count"] == 1
    assert not selected[1]["reference_success"]
    assert "APPROXIMATE_PROPOSAL" in selected[1]["query_statuses"]


# BEGIN absent or uncertain evidence is distinct from a measured violation.
def test_unestablished_model_and_native_reference_success_have_separate_counts():
    selected = chains([[[.1], [.5]]],
        [[report(model_valid=False), report(native_pose_ik_success=False)]])
    assert len(selected) == 2 and selected[0]["candidate_indices"] == [0]
    assert selected[0]["reference_success"]
    assert selected[0]["selection_cost"]["static_model_failed_count"] == 0
    assert selected[0]["selection_cost"]["static_model_not_established_count"] == 1


def test_static_failures_rank_chains_without_erasing_the_only_native_solution():
    # Symmetric paths have equal pose energy and equal velocity energy, so
    # the existing static evidence can decide their otherwise exact tie.
    selected = chains([[[-.1], [.1]], [[-.2], [.2]]],
        [[report(task_satisfied=False, model_valid=False, task_violations={"coordinate:0": .1},
                 native_violations={"cspace": .2}), report()]]*2, count=4)
    assert selected[0]["candidate_indices"] == [1, 1]
    assert len(selected) == 4
    only = chains([[[.1]], [[.2]]], [[report(task_satisfied=False, model_valid=False,
        task_violations={"coordinate:0": .1}, native_violations={"cspace": .2})]]*2)
    assert only[0]["candidate_indices"] == [0, 0]
    assert only[0]["selection_cost"]["static_task_failed_count"] == 2
    assert only[0]["selection_cost"]["static_model_failed_count"] == 2
# END whole-curve task and scene acceptance is not asserted by these proposals.


def test_legal_failed_states_are_coarse_nodes_with_original_success_evidence():
    rows = [[report(native_pose_ik_success=False), report(native_pose_ik_success=False)],
            [report(native_pose_ik_success=False)]]
    selected = chains([[[.1], [.9]], [[.2]]], rows, pose_costs=[[.2, .1], [.3]])
    assert selected[0]["candidate_indices"] == [0, 0]
    assert selected[0]["selection_cost"]["failed_pose_energy"] == pytest.approx(.5)
    assert selected[0]["selection_cost"]["successful_pose_energy"] == 0.
    assert selected[0]["query_statuses"] == ["APPROXIMATE_PROPOSAL"]*2
    assert selected[0]["interpolated_queries"] == []
    assert not selected[0]["reference_success"]
    assert all(not item["native_pose_ik_success"] for row in rows for item in row)


def test_position_domain_and_finite_energy_define_nodes_without_a_soft_pose_threshold():
    values = [[[float("nan")], [-1.], [1.], [1.01], [.5]], [[.25]]]
    rows = [[report(native_pose_ik_success=False) for _ in pool] for pool in values]
    selected = chains(values, rows, positions=[[-1.], [1.]],
        pose_costs=[[0., 1e8, 1e8, 0., float("inf")], [1e8]], count=2,
        velocity=[[-.01], [.01]], times=[.1, .2])
    assert {tuple(item["candidate_indices"]) for item in selected} == {(1, 0), (2, 0)}
    assert all(item["query_statuses"] == ["APPROXIMATE_PROPOSAL"]*2 for item in selected)
    # These are coarse numerical proposals. Their large pose/velocity
    # errors remain for the tool-space optimizer and continuous validation.
    assert all(item["selection_cost"]["failed_pose_energy"] == 2e8 for item in selected)


def test_qualified_successful_continuity_ranks_before_pose_refinement():
    selected = chains([[[.1], [2.]]], [[report(), report(model_valid=False)]],
                      pose_costs=[[1., .1]])
    assert selected[0]["candidate_indices"] == [0]
    assert selected[0]["selection_cost"]["successful_pose_energy"] == 1.
    assert selected[1]["candidate_indices"] == [1]
    assert selected[1]["selection_cost"]["static_model_not_established_count"] == 1


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
    costs = rng.uniform(size=(4, 3))
    solved = rng.random(size=(4, 3)) > .4
    # BEGIN exhaustive enumeration uses independently declared evidence states.
    # Codes 0/1/2 mean passed, definite violation and unestablished. They are
    # encoded as the same raw components the production producer supplies.
    task_states = rng.integers(0, 3, size=(4, 3))
    model_states = rng.integers(0, 3, size=(4, 3))
    rows = [[report(native_pose_ik_success=bool(solved[query, index]),
        task_satisfied=bool(task_states[query, index] == 0),
        model_valid=bool(model_states[query, index] == 0),
        task_violations={"coordinate:0": .1} if task_states[query, index] == 1 else {},
        native_violations={"cspace": .1} if model_states[query, index] == 1 else {})
        for index in range(3)] for query in range(4)]
    # END enumeration below never calls the production prefix interpreter.
    expected = []
    for path in product(range(3), repeat=4):
        postures = np.stack([pools[query, index] for query, index in enumerate(path)])
        changes = np.diff(np.r_[np.zeros((1, 2)), postures], axis=0)
        intervals = np.diff(np.r_[0., times])
        energy = np.sum(changes**2/scale**2/intervals[:, None])/(times[-1]*2)
        failures = sum(not solved[query, index] for query, index in enumerate(path))
        failed_energy = sum(costs[query, index] for query, index in enumerate(path)
                            if not solved[query, index])
        successful_energy = sum(costs[query, index] for query, index in enumerate(path)
                                if solved[query, index])
        evidence_counts = (sum(task_states[q, i] == 1 for q, i in enumerate(path)),
                           sum(model_states[q, i] == 1 for q, i in enumerate(path)),
                           sum(task_states[q, i] == 2 for q, i in enumerate(path)),
                           sum(model_states[q, i] == 2 for q, i in enumerate(path)))
        expected.append(((failures, float(energy), float(failed_energy),
                          float(successful_energy), *evidence_counts), path))
    expected.sort()
    actual = chains(pools, rows, times=times, velocity=bounds, pose_costs=costs, count=3)
    assert [tuple(row["candidate_indices"]) for row in actual] == [path for _, path in expected[:3]]
    np.testing.assert_allclose([row["selection_cost"]["normalized_velocity_energy"] for row in actual],
                               [score[1] for score, _ in expected[:3]], rtol=1e-14)
    np.testing.assert_allclose([row["selection_cost"]["failed_pose_energy"] for row in actual],
                               [score[2] for score, _ in expected[:3]], rtol=1e-14)
    np.testing.assert_allclose([row["selection_cost"]["successful_pose_energy"] for row in actual],
                               [score[3] for score, _ in expected[:3]], rtol=1e-14)


@pytest.mark.parametrize("times", [[0., 1.], [1., 1.], [2., 1.], [1., float("nan")]])
def test_invalid_clock_is_rejected_instead_of_retimed(times):
    with pytest.raises(ValueError, match="increasing physical times"):
        chains([[[.1]], [[.2]]], times=times)
# END these candidates are initialization evidence, never a checked continuous trajectory.


# BEGIN all reference consumers preserve success and numerical continuity first.
def policy_first_indices(values, rows, *, pose_costs=None):
    """Compare the three consumers using identical, explicitly supplied evidence."""
    values = np.asarray(values, float)
    energy = np.zeros(len(values)) if pose_costs is None else np.asarray(pose_costs, float)
    ranked = select(values, rows, count=len(values))
    tracked = select_ik_tracking_candidate(values, list(reversed(ranked)), energy,
        np.zeros((1, values.shape[1])), reports=rows)
    selected = chains([values], [rows], pose_costs=[energy], count=len(values))
    return ranked[0], tracked, selected[0]["candidate_indices"][0]


# BEGIN all reference consumers share continuity before pose-energy diagnostics.
@pytest.mark.parametrize("solved", [True, False])
def test_reference_pose_energy_cannot_override_continuity_for_any_native_status(solved):
    rows = [report(native_pose_ik_success=solved)]*2
    assert policy_first_indices([[.01], [.8]], rows, pose_costs=[1., 1e-12]) == (0, 0, 0)


@pytest.mark.parametrize("solved", [True, False])
def test_duplicate_reference_representative_uses_continuity_before_pose_energy(solved):
    rows = [[report(native_pose_ik_success=solved)]*2]
    selected = chains([[[.1], [.100002]]], rows, pose_costs=[[1., 1e-12]])
    assert len(selected) == 1
    assert selected[0]["candidate_indices"] == [0]
# END success and functional allowed-set evidence retain their original roles.


@pytest.mark.parametrize("component", ["collision/robot/self", "collision/entity/background",
    "contact/declared_relation", "initial_support/declared_relation", "gripper/controller"])
def test_definite_model_report_does_not_override_numerical_reference_continuity(component):
    rows = [report(model_valid=False, scene_components={component: {"maximum": .01}}), report()]
    assert policy_first_indices([[.01], [.8]], rows, pose_costs=[1e-12, 1e-6]) == (0, 0, 0)
    assert select([[.01], [.8]], rows, query_kind="functional") == [1]


@pytest.mark.parametrize("unknown", [
    {"static_model_query": "not_run_native_reference_ik_failed", "native_pose_ik_success": False},
    {"scene_components": {"unresolved/robot/self": {"maximum": 1.}}},
    {"scene_components": {"unknown/forbidden/entity": {"maximum": 1.}}},
    {"scene_components": {"conditional/hand_geometry": {"maximum": 1.}}},
    {"scene_components": {"relationship/missing_release/entity/0": {"maximum": 1.}}},
    {"scene_components": {"scene_update/background/robot": {"maximum": 1.}}},
    {"scene_components": {"unclassified_evidence": {"maximum": 1.}}},
    {"scene_components": {"collision/robot/self": {"maximum": float("nan")}}},
    {"native_violations": {"cspace": float("inf")}},
    {},
])
def test_unknown_model_never_becomes_a_definite_collision_or_failure(unknown):
    rows = [report(model_valid=False,
                   scene_components={"collision/robot/self": {"maximum": .01}}),
            report(model_valid=False, **unknown)]
    assert policy_first_indices([[.01], [.8]], rows) == (0, 0, 0)
    selected = chains([[[.8]]], [[rows[1]]])[0]
    assert selected["selection_cost"]["static_model_failed_count"] == 0
    assert selected["selection_cost"]["static_model_not_established_count"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_task_value_retains_unestablished_state(value):
    selected = chains([[[.1]]], [[report(task_satisfied=False,
        task_violations={"coordinate:0": value})]])[0]
    assert selected["selection_cost"]["static_task_failed_count"] == 0
    assert selected["selection_cost"]["static_task_not_established_count"] == 1


def test_known_collision_and_unknown_relation_keep_the_same_original_report():
    row = report(model_valid=False, scene_components={
        "collision/robot/self": {"maximum": .01},
        "unresolved/coverage/entity": {"maximum": 1.}})
    selected = chains([[[.1]]], [[row]])[0]
    assert selected["selection_cost"]["static_model_failed_count"] == 1
    assert selected["selection_cost"]["static_model_not_established_count"] == 0
    assert set(row["scene_components"]) == {"collision/robot/self", "unresolved/coverage/entity"}


def test_task_then_model_failure_priority_belongs_to_functional_queries():
    rows = [report(task_satisfied=False, task_violations={"coordinate:0": .1}),
            report(model_valid=False, native_violations={"cspace": .1})]
    assert policy_first_indices([[.01], [.8]], rows) == (0, 0, 0)
    assert select([[.01], [.8]], rows, query_kind="functional") == [1]


def test_unknown_verdict_only_breaks_equal_numerical_reference_ties():
    rows = [report(task_satisfied=False, model_valid=False), report()]
    assert policy_first_indices([[.01], [.8]], rows) == (0, 0, 0)
    assert policy_first_indices([[-.8], [.8]], rows) == (1, 1, 1)


def test_unqueried_reference_counts_neither_static_failure_and_keeps_failed_status():
    row = report(task_satisfied=False, model_valid=False, native_pose_ik_success=False,
        static_model_query="not_run_native_reference_ik_failed")
    selected = chains([[[.1]]], [[row]])[0]
    assert selected["selection_cost"] == {
        "static_task_failed_count": 0, "static_model_failed_count": 0,
        "nonconverged_candidate_count": 1,
        "static_task_not_established_count": 1, "static_model_not_established_count": 1,
        "failed_pose_energy": 0., "normalized_velocity_energy": pytest.approx(.01),
        "successful_pose_energy": 0.}
    assert selected["query_statuses"] == ["APPROXIMATE_PROPOSAL"]


def test_numerical_chain_retains_original_pool_indices_and_interpolated_status():
    values = [[[float("nan")], [.1]], [[float("nan")]], [[.3]], [[.4]]]
    rows = [[report(), report()], [report()], [report(native_pose_ik_success=False)], [report()]]
    selected = chains(values, rows)[0]
    assert selected["candidate_indices"] == [1, None, 0, 0]
    assert selected["anchor_query_indices"] == [0, 2, 3]
    assert selected["query_statuses"] == ["SOLVED", "INTERPOLATED_PROPOSAL", "APPROXIMATE_PROPOSAL", "SOLVED"]
    assert not selected["reference_success"]
# END no observation mask limits the independent full-reference objective.


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


def test_actual_task_and_online_checks_rank_reference_ik_without_hardening_its_pose(monkeypatch):
    planner, calls = _static_planner(monkeypatch)
    selected, reports = planner._rank_ik_candidates(
        [[.1, 0., 0.], [.7, 0., 0.]], [True, False], [[0., 0., 0.]], .4)
    assert selected.tolist() == [0, 1]
    # BEGIN reference failures keep unqueried status; functional queries differ below.
    assert reports[0]["task_satisfied"] and reports[0]["model_valid"]
    assert reports[1]["static_model_query"] == "not_run_native_reference_ik_failed"
    assert not reports[1]["native_pose_ik_success"]
    assert calls["online"] == [(.8, .4)]
    assert len(calls["native"]) == 1
    # END no model evidence is invented for a skipped reference proposal.


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
    from manipulation.planning_problem import PoseEventRequirement

    planner = planning._ConstrainedPlanner.__new__(planning._ConstrainedPlanner)
    # BEGIN the functional IK fixture declares the same submitted-work ledger.
    planner.budget = SimpleNamespace(ik_seeds=2, max_validation_points=32, maximum_trajectory_solves=1)
    planner.request_work = {"ik_queries": 0}
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
    planner.start = SimpleNamespace(position=torch.zeros((1, 3), dtype=torch.float64))
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

    planner.ik = SimpleNamespace(solve_pose=solve, get_active_js=lambda value: value,
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
    check_search.assert_called()
# END mocked queries test orchestration, not CUDA geometry or actual IK convergence.
