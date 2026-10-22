"""Bounded reference enrichment with declared IK outputs and real cubic fits.

Ordinary reference contributions and curve verdicts are explicit test data.
Production code owns query scheduling, source identities, budgets, pool
selection, successful-observation fitting and final old/new curve selection.
These tests allocate no robot or GPU solver.
"""
# BEGIN one original seed pass bounds numerical enrichment and retains evidence.
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from manipulation.planning import PlanningWorkExhausted
import test_manipulation_planning as fixtures


def fixture(*, count=2, additional_success=True, times=None, costs=None,
            old_feasible=True, new_feasible=True, old_fixed=10., new_fixed=20.,
            old_optimization=15., new_optimization=25.):
    """Build original pools through production, then expose extra native queries."""
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    planner, _, _ = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags, seed_count=count)
    seeds, _ = planner.reference_seeds()
    layers, chains, entry = planner._refine_reference_seeds.call_args.kwargs["enrichment"]
    duration = planner.formal_clock["duration"]
    parameters = np.array([0., .1875, .3125, 1.]) if times is None else np.asarray(times)
    contribution = [1e6, 30., 20., 1.] if costs is None else list(costs)
    coarse = {"geometry_scope": "robot", "seeds": [{
        "seed_index": index, "sampled_scope_feasible": old_feasible,
        "fixed_objective": old_fixed, "optimization_objective": old_optimization,
        "reference_times_s": (parameters*duration).tolist(), "reference_point_costs": list(contribution)
    } for index in range(count)]}
    calls = []

    def native(goal, **kwargs):
        u = float(planner.seed_parameter.item())
        calls.append({"u": u, "warm": kwargs["seed_config"].clone(),
                      "return_seeds": kwargs["return_seeds"]})
        values = torch.tensor([[[.7+.02*row, .1*u, 0.] for row in range(planner.budget.ik_seeds)]],
                              dtype=planner.start.position.dtype)
        success = torch.full((1, planner.budget.ik_seeds), additional_success, dtype=torch.bool)
        return SimpleNamespace(js_solution=SimpleNamespace(position=values), success=success,
            position_error=torch.full(success.shape, 1e-6 if additional_success else .2),
            rotation_error=torch.full(success.shape, 1e-5 if additional_success else .3))

    planner.ik.solve_pose = native
    # Only the scalar comparison is declared. The coefficients sent to this
    # boundary come from the actual production masked physical-domain QP.
    def snapshot(actions):
        rows = []
        for index, action in enumerate(actions):
            rows.append({"seed_index": index, "sampled_scope_feasible": new_feasible,
                "fixed_objective": new_fixed, "optimization_objective": new_optimization,
                "controls": action.tolist()})
        return {"geometry_scope": "robot", "seeds": rows}, [row.clone() for row in actions]

    planner._initialization_snapshot = Mock(side_effect=snapshot)
    planner._controls_curve = Mock(side_effect=lambda row: row.clone())
    return planner, seeds, coarse, layers, chains, entry, calls


def run(data, *, initialization=50):
    planner, seeds, coarse, layers, chains, entry, _ = data
    return planner._enrich_reference_seeds(seeds, coarse, layers, chains, entry,
        initialization=initialization, total=100)


def test_one_pass_uses_weighted_original_grid_and_at_most_two_calls_per_seed():
    data = fixture()
    planner, seeds, coarse, layers, chains, _, calls = data
    original_log = deepcopy(planner.reference_ik_queries)
    before_work = planner.request_work["ik_queries"]
    selected, _, _ = run(data)
    assert [row["u"] for row in calls] == pytest.approx([.1875, .25, .3125, .375], rel=0, abs=1e-15)
    assert len(calls) == 2*planner.budget.trajectory_seeds
    assert all(row["return_seeds"] == planner.budget.ik_seeds for row in calls)
    assert planner.request_work["ik_queries"]-before_work == len(calls)*planner.budget.ik_seeds
    report = planner.reference_initialization["enrichment"]
    assert report["calls_started"] == 4 and report["candidate_solves"] == 16
    assert report["maximum_new_calls"] == 4
    assert planner.reference_ik_queries[:len(original_log)] == original_log
    assert [row["kind"] for row in planner.reference_ik_queries[6:]] == [
        "additional", "right_neighbor", "additional", "right_neighbor"]
    assert [row["query_index"] for row in planner.reference_ik_queries] == list(range(10))
    assert len(planner.reference_ik_distinct_counts) == len(planner.reference_ik_queries)
    assert all(row["native_success_count"] == 4 for row in planner.reference_ik_queries[6:])
    for offset in (0, 2):
        additional = planner.reference_ik_queries[6+offset]
        right = planner.reference_ik_queries[7+offset]
        source = additional["warm_start_source"]
        expected = original_log[source["query_index"]]["candidates"][source["candidate_index"]]["joint_position"]
        np.testing.assert_array_equal(calls[offset]["warm"][0, 0].numpy(), expected)
        assert right["warm_start_source"]["query_index"] == additional["query_index"]
        index = right["warm_start_source"]["candidate_index"]
        np.testing.assert_array_equal(calls[offset+1]["warm"][0, 0].numpy(),
            additional["candidates"][index]["joint_position"])
    assert all(len(layer["values"]) <= planner.budget.ik_seeds for layer in layers.values())
    for chain in planner.reference_ik_chains:
        assert chain["query_parameters"] == sorted(layers)
        for state, source in zip(chain["joint_positions"], chain["candidate_sources"]):
            row = planner.reference_ik_queries[source["query_index"]]["candidates"][source["candidate_index"]]
            np.testing.assert_array_equal(state, row["joint_position"])
    planner._initialization_snapshot.assert_called_once()
    # A worse refit leaves both original coefficient arrays byte-identical.
    torch.testing.assert_close(selected, seeds, rtol=0, atol=0)
    assert report["selected_sources"] == ["original", "original"]
    assert coarse["seeds"][0]["reference_point_costs"] == [1e6, 30., 20., 1.]


def test_failed_additional_queries_keep_reports_without_repeating_global_fit_or_curve_evaluation():
    data = fixture(additional_success=False)
    planner, seeds, coarse, _, _, _, calls = data
    with patch.object(planner, "_fit_reference_chains", wraps=planner._fit_reference_chains) as fit:
        selected, snapshot, _ = run(data)
    assert [row["u"] for row in calls] == pytest.approx([.1875, .3125], rel=0, abs=1e-15)
    assert all(row["kind"] == "additional" for row in planner.reference_ik_queries[6:])
    assert all(row["native_success_count"] == 0 for row in planner.reference_ik_queries[6:])
    fit.assert_not_called()
    planner._initialization_snapshot.assert_not_called()
    assert selected is seeds and snapshot is coarse
    assert planner.reference_initialization["enrichment"]["status"] == "no_new_successful_observation"


def test_terminal_time_is_available_but_has_no_right_original_query():
    data = fixture(count=1, times=[0., 1.], costs=[1e6, 10.])
    run(data)
    assert [row["u"] for row in data[-1]] == [1.]


@pytest.mark.parametrize("mode", ["zero_iterations", "no_new_grid_time", "count_exhausted", "time_exhausted"])
def test_exhausted_optional_work_preserves_original_curves(mode):
    data = fixture()
    planner, seeds, coarse, layers, _, _, calls = data
    initialization = 0 if mode == "zero_iterations" else 50
    if mode == "no_new_grid_time":
        for row in coarse["seeds"]:
            row["reference_times_s"] = (np.r_[0., sorted(layers)]*planner.formal_clock["duration"]).tolist()
            row["reference_point_costs"] = list(range(len(row["reference_times_s"])))
    if mode == "count_exhausted":
        planner.request_work["ik_queries"] = (planner.budget.max_validation_points
            *planner.budget.ik_seeds*planner.budget.maximum_trajectory_solves)
    if mode == "time_exhausted":
        planner.remaining_request_seconds = lambda: 0.
    before = planner.request_work["ik_queries"]
    selected, snapshot, _ = run(data, initialization=initialization)
    assert not calls and selected is seeds and snapshot is coarse
    assert planner.request_work["ik_queries"] == before
    planner._initialization_snapshot.assert_not_called()


def test_one_remaining_query_slot_is_shared_with_right_neighbor_and_other_seeds():
    data = fixture()
    planner = data[0]
    limit = planner.budget.max_validation_points*planner.budget.ik_seeds*planner.budget.maximum_trajectory_solves
    planner.request_work["ik_queries"] = limit-planner.budget.ik_seeds
    run(data)
    assert [row["u"] for row in data[-1]] == pytest.approx([.1875], rel=0, abs=1e-15)
    assert planner.request_work["ik_queries"] == limit
    assert planner.reference_initialization["enrichment"]["calls_started"] == 1


def test_exception_after_native_submission_preserves_actual_candidate_work():
    data = fixture()
    planner = data[0]
    before = planner.request_work["ik_queries"]
    planner.ik.solve_pose = Mock(side_effect=RuntimeError("native query failed after submission"))
    with pytest.raises(RuntimeError, match="after submission"):
        run(data)
    assert planner.request_work["ik_queries"] == before+planner.budget.ik_seeds
    assert planner.reference_initialization["enrichment"]["calls_started"] == 1
    assert planner.reference_initialization["enrichment"]["candidate_solves"] == planner.budget.ik_seeds
    assert len(planner.reference_ik_queries) == 6


def test_required_ik_exhaustion_raises_without_starting_native_work():
    data = fixture(count=1)
    planner = data[0]
    limit = planner.budget.max_validation_points*planner.budget.ik_seeds*planner.budget.maximum_trajectory_solves
    planner.request_work["ik_queries"] = limit
    with pytest.raises(PlanningWorkExhausted):
        planner._solve_ik(.5, planner.start.position[:, None])
    assert not data[-1] and planner.request_work["ik_queries"] == limit


@pytest.mark.parametrize("old_feasible,new_feasible,old_fixed,new_fixed,old_optimization,new_optimization,accepted", [
    (True, False, 10., 1., 15., 1., False),
    (False, True, 1., 100., 1., 100., True),
    (True, True, 10., 10., 1., .1, False),
    (True, True, 10., 5., 1., 100., True),
    (False, False, 1., 100., 10., 5., True),
    (False, False, 100., 1., 5., 10., False),
])
def test_same_scope_curve_qualification_and_objective_choose_each_actual_refit(
        old_feasible, new_feasible, old_fixed, new_fixed, old_optimization, new_optimization, accepted):
    data = fixture(count=1, old_feasible=old_feasible, new_feasible=new_feasible,
        old_fixed=old_fixed, new_fixed=new_fixed,
        old_optimization=old_optimization, new_optimization=new_optimization)
    selected, _, _ = run(data)
    planner, original = data[:2]
    evaluated = planner._initialization_snapshot.call_args.args[0][None]
    torch.testing.assert_close(selected, evaluated if accepted else original, rtol=0, atol=0)
    assert not torch.equal(evaluated, original)
    assert planner.reference_initialization["enrichment"]["selected_sources"] == [
        "enriched" if accepted else "original"]
# END budgets and numerical selection never assert complete task-scene qualification.
