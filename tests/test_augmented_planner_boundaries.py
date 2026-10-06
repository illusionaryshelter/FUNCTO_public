"""CPU checks for shared AL/map boundaries and the original planning budget."""

# BEGIN real public-loop tests with scripted native results; no CUDA or robot work.
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import manipulation.planning as planning
import test_manipulation_planning as planning_fixtures
from manipulation.planning import PlanningBudget, PlanningWorkExhausted, _ProgressMap
from manipulation.planning_problem import MotionPhase


def scripted_boundary(flags, residuals, *, map_changed=False):
    """Expose AL generations at each actual call through plan_trajectory()."""
    contract = planning_fixtures.PlannerRetryContractTests()
    planner, native, trajectories = contract.scripted_planner(flags, residuals)
    planner.scene = SimpleNamespace(describe=lambda: {"scope": "CPU boundary fixture"})
    generation, seen = [0], []
    original_optimize = planner.optimize.side_effect

    def optimize(seed):
        seen.append(generation[0])
        return original_optimize(seed)

    def update(candidate):
        # The selected result, rather than the seed or another branch, owns
        # the multiplier update. No numerical convergence claim is mocked.
        assert candidate is native[len(seen)-1]
        generation[0] += 1
        return {"changed": True, "generation": generation[0]}

    planner.optimize.side_effect = optimize
    planner.update_functional_constraints = Mock(side_effect=update)
    planner.refine_background_for_seed = Mock(
        side_effect=lambda seed, **kwargs: {"updated": bool(map_changed and len(seen) == 1)})
    return contract, planner, native, trajectories, seen


@pytest.mark.parametrize("map_changed", (False, True))
def test_multiplier_change_reaches_next_solve_with_one_shared_rebuild(map_changed):
    contract, planner, native, _, seen = scripted_boundary(
        [False, True], [.1, 0.], map_changed=map_changed)
    budget = PlanningBudget(maximum_trajectory_solves=2, trajectory_iterations=100)
    planner.budget = budget
    result = contract.run_planner(planner, budget)

    assert result.success, result.diagnostics
    assert seen == [0, 1]
    assert planner.optimize.call_count == 2
    assert planner.refine_validation.call_count == 1
    assert planner.refine_background_for_seed.call_count == 2
    deferred = planner.refine_background_for_seed.call_args
    assert deferred.kwargs["rebuild"] is False
    torch.testing.assert_close(deferred.args[0], native[0].solution)
    assert result.diagnostics["attempts"][0]["functional_augmented_lagrangian"]["changed"]
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 2
    assert result.request_ledger["planning_budget"]["trajectory_iterations"] == 100
    planner.ompl_seed.assert_not_called()


def test_persistent_functional_violation_uses_only_four_original_solves():
    contract, planner, _, _, seen = scripted_boundary([False]*4, [.1]*4)
    budget = PlanningBudget(maximum_trajectory_solves=4, trajectory_iterations=100)
    planner.budget = budget
    result = contract.run_planner(planner, budget)

    assert not result.success
    assert result.execution_binding is None
    assert seen == [0, 1, 2, 3]
    assert planner.optimize.call_count == 4
    assert planner.refine_validation.call_count == 3
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 4
    assert result.request_ledger["planning_budget"]["maximum_trajectory_solves"] == 4
    assert result.request_ledger["planning_budget"]["trajectory_iterations"] == 100
    planner.ompl_seed.assert_not_called()


# BEGIN an independent searched seed also merges map changes before one rebuild.
def test_searched_seed_map_and_multiplier_reset_share_one_rebuild():
    contract, planner, _, _, _ = scripted_boundary([False, True], [.1, 0.])
    planner.update_functional_constraints = Mock(return_value={"changed": False})
    planner.refine_background_for_seed = Mock(side_effect=[
        {"updated": False}, {"updated": False}, {"updated": True}])
    path_seed = torch.ones((1, 1, 4, 3))
    planner.ompl_seed.return_value = path_seed, {"success": True}
    budget = PlanningBudget(maximum_trajectory_solves=2, search_seconds=1.)
    planner.budget = budget
    result = contract.run_planner(planner, budget)

    assert result.success, result.diagnostics
    assert planner.refine_background_for_seed.call_count == 3
    assert planner.refine_background_for_seed.call_args.kwargs == {"rebuild": False}
    torch.testing.assert_close(planner.refine_background_for_seed.call_args.args[0], path_seed)
    planner.reset_functional_constraints.assert_called_once_with()
    assert planner.refine_validation.call_count == 1
    assert planner.refine_validation.call_args.kwargs == {"phase": "feasibility"}
    assert planner.optimize.call_count == 2
    assert result.diagnostics["used_ompl"]
# END the next loop cannot perform another map-triggered reconstruction.


def test_quality_rejection_keeps_exact_previously_qualified_curve_and_binding():
    contract, planner, _, trajectories, _ = scripted_boundary([True, False], [0., .1])
    budget = PlanningBudget(maximum_trajectory_solves=2, trajectory_iterations=100)
    planner.budget = budget
    result = contract.run_planner(planner, budget)

    assert result.success, result.diagnostics
    assert result.trajectory is trajectories[0]
    assert result.execution_binding["candidate_id"] == "candidate-0"
    assert result.execution_binding["validation"]["measurements"]["candidate_index"] == 0
    assert planner.optimize_phases == ["feasibility", "quality"]
    assert result.diagnostics["returned_candidate"]["phase"] == "feasibility"
    assert planner.refine_background_for_seed.call_count == 1
    assert len(planner.bound_candidates) == 1
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 2
    planner.ompl_seed.assert_not_called()


def test_qualified_final_candidate_is_bound_without_optional_al_query_work():
    """Completed hard validation must precede work needed only by a later solve."""
    contract, planner, _, trajectories, _ = scripted_boundary([True], [0.])
    planner.update_functional_constraints.side_effect = PlanningWorkExhausted(
        "optional multiplier measurement would consume the remaining query budget")
    budget = PlanningBudget(maximum_trajectory_solves=1)
    planner.budget = budget
    result = contract.run_planner(planner, budget)

    assert result.success, result.diagnostics
    assert result.trajectory is trajectories[0]
    assert result.execution_binding["candidate_id"] == "candidate-0"
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 1
    planner.update_functional_constraints.assert_not_called()
    planner.refine_validation.assert_not_called()


def test_same_controls_revalidate_after_another_candidate_adds_history():
    """A verdict for earlier anchors cannot cover a newly discovered failure time."""
    contract = planning_fixtures.PlannerRetryContractTests()
    planner, native, curves = contract.scripted_planner([True, True], [0., .2])
    for name in ("joint_names", "times", "position", "velocity"):
        setattr(curves[1], name, deepcopy(getattr(curves[0], name)))
    planner._observed_verdicts = []
    planner.last_optimizer_diagnostics = [{"seed_index": 0, "unchanged_from_seed": False,
                                           "maximum_control_update": .1, "cores": []}]
    _, _, first = planner.select_candidate(native[0], origin={"kind": "initial_seed"})
    assert first["accepted"]
    native[1].solution.copy_(native[0].solution)

    _, _, repeated = planner.select_candidate(native[1])
    assert repeated["accepted"] and repeated["previously_considered"]
    assert planner.export.call_count == 1

    planner.witnesses.add_validation({"u": np.array([.23]), "violations": np.array([.1])})
    _, _, rechecked = planner.select_candidate(native[1])
    assert planner.export.call_count == 2
    assert rechecked["independent_validation"] == "completed"
    assert not rechecked["accepted"]
    assert .23 in planner.witnesses.parameters


def test_rejected_observation_rechecks_independent_failures_for_current_selection():
    """A rejection needs its actual per-time residuals when selected later."""
    contract = planning_fixtures.PlannerRetryContractTests()
    planner, native, _ = contract.scripted_planner([True], [.2])
    planner._observed_verdicts = []
    planner.last_optimizer_diagnostics = [{"seed_index": 0, "unchanged_from_seed": False,
                                           "maximum_control_update": .1, "cores": []}]
    _, _, observed = planner.select_candidate(native[0], origin={"kind": "initial_seed"})
    assert not observed["accepted"]
    assert planner._observed_verdicts == []
    planner.witnesses.set_current({}, duration=1.)

    _, _, selected = planner.select_candidate(native[0])
    assert not selected["accepted"]
    assert planner.export.call_count == 2
    np.testing.assert_array_equal(planner.witnesses.current_parameters, [.5])


def test_refuted_archive_cannot_return_when_final_candidate_is_a_different_curve():
    """A prior observer's completed rejection must survive later seed selection."""
    contract, planner, _, trajectories, _ = scripted_boundary([True, False], [0., .1])
    original_optimize = planner.optimize.side_effect

    def optimize(seed):
        result = original_optimize(seed)
        if planner.phase == "quality":
            planner._candidate_rejected(deepcopy(trajectories[0]),
                {"native_success": False, "accepted": False,
                 "reason": "same archived curve fails an added historical query"})
        return result

    planner.optimize.side_effect = optimize
    budget = PlanningBudget(maximum_trajectory_solves=2)
    planner.budget = budget
    result = contract.run_planner(planner, budget)

    assert not result.success
    assert result.trajectory is None and result.execution_binding is None
    assert len(planner.bound_candidates) == 1
    assert result.diagnostics["invalidated_feasible_candidate"]["reason"] == "same_curve_failed_additional_validation"
    assert "best_feasible_quality" not in result.diagnostics
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 2


def test_inactive_historical_failure_is_still_checked_and_rejects_export(monkeypatch):
    """A historical time outside the optimizer set remains a real geometry query."""
    problem = planning_fixtures.cartesian_problem(
        (MotionPhase("free", planning_fixtures.constant_path()),))
    planner, native = planning_fixtures.export_fixture(
        problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
        np.zeros((5, 3)), np.zeros((5, 3)), .25)
    planner.budget = replace(planner.budget, max_refinement_points=16)
    history = np.linspace(.23, .73, 21)
    planner.witnesses.add_validation({"u": history, "violations": np.ones(len(history))})
    planner.witnesses.set_current({}, duration=1.)
    assert planner.witnesses.current_parameters.size == 0
    rejected_time = float(history[0])
    source_times, sampler_arguments = [], []
    original_sampler = planning._export_sample_times

    def sampling(*args, **kwargs):
        sampler_arguments.append(kwargs)
        return original_sampler(*args, **kwargs)

    def source_check(joints, names, times, *, diagnostics):
        source_times.extend(times.detach().cpu().numpy().reshape(-1).tolist())
        diagnostics.update(checked_states=times.numel(), unmodelled_geometry_names=[],
                           worst_state={"trajectory_time_s": rejected_time})
        return (times == rejected_time).to(times)

    monkeypatch.setattr(planning, "_export_sample_times", sampling)
    planner._verification_scene.check_robot_self_collision = source_check
    trajectory, count = planner.export(native)

    assert len(sampler_arguments) == 1
    assert sampler_arguments[0]["budget"] is planner.budget
    assert set(history) <= set(sampler_arguments[0]["anchors"])
    assert set(history) <= set(source_times)
    assert len(source_times) == count
    assert trajectory.residuals["source_self_collision"] == 1.
    assert not all(value <= 0. for value in trajectory.residuals.values())
    np.testing.assert_array_equal(planner.witnesses.parameters, history)
# END scripted checks assert boundary/accounting contracts, not robot feasibility.
