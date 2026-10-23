"""CPU regressions for numerical IK proposals and independent success evidence.

The analytic planner fixture supplies native-shaped positions and residuals;
the production physical initializer preserves the measured entrance and bound clock.
These tests exercise proposal construction, not robot reachability or final
task/scene qualification.
"""

# BEGIN finite position-domain proposals remain distinct from native pose success.
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.planning_ik import select_ik_chains
import test_manipulation_planning as fixtures


def _report(solved):
    return {"task_satisfied": solved, "model_valid": solved,
            "static_model_query": "evaluated" if solved else "not_run_native_reference_ik_failed",
            "task_violations": {}, "native_violations": {}, "scene_components": {},
            "native_pose_ik_success": solved, "candidate_source": "native_ik"}


def test_domain_invalid_gap_uses_available_node_times_and_measured_entrance():
    """States outside the declared position domain cannot become graph nodes."""
    times = np.array([.1, .5, 1., 2., 3.])
    pools = [np.array([[value]]) for value in (100., 1., -100., 2., 100.)]
    reports = [[_report(index in (1, 3))] for index in range(len(times))]
    chain, = select_ik_chains(pools, reports, [[0.]], times, [[-1.], [1.]],
        position_limits=[[-2.], [2.]], pose_costs_by_query=[[0.]]*5,
        count=2, joint_tolerance=1e-5)
    np.testing.assert_allclose(np.asarray(chain["joint_positions"])[:, 0], [.2, 1., 4./3., 2., 2.])
    assert chain["candidate_indices"] == [None, 0, None, 0, None]
    assert chain["anchor_query_indices"] == [1, 3]
    assert chain["query_statuses"] == ["INTERPOLATED_PROPOSAL", "SOLVED",
        "INTERPOLATED_PROPOSAL", "SOLVED", "INTERPOLATED_PROPOSAL"]
    assert chain["interpolated_queries"] == [0, 2, 4]
    assert chain["selection_cost"]["normalized_velocity_energy"] == pytest.approx((2.+1./1.5)/3.)


def test_domain_invalid_native_states_do_not_change_tracking_or_coarse_branches():
    """Different illegal output coordinates preserve the same admitted proposals."""
    outcomes = []
    invalid_queries = (0, 2, 5)
    for invalid_value in (-100., 100.):
        _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
        for query in invalid_queries:
            pools[query] = np.full_like(pools[query], invalid_value)
            flags[query] = np.zeros_like(flags[query])
        planner, initialize, warm_starts = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
        seeds, _ = planner.reference_seeds()
        outcomes.append((seeds, warm_starts))
        initialize.assert_called_once()
        assert len(planner.reference_ik_chains) == 2
        for query, last_proposal in enumerate((None, None, 1, 1, 3, 4)):
            warm = warm_starts[query].numpy()
            assert warm.shape == (1, 1, 3)
            if last_proposal is None:
                np.testing.assert_array_equal(warm[0], planner.start.position.numpy())
            else:
                assert any(np.array_equal(warm[0, 0], row)
                           for row in pools[last_proposal][flags[last_proposal]])
            assert planner.reference_ik_queries[query]["warm_start_from_query"] == last_proposal
        for chain in planner.reference_ik_chains:
            assert chain["anchor_query_indices"] == [1, 3, 4]
            assert chain["interpolated_queries"] == list(invalid_queries)
            assert np.abs(chain["joint_positions"]).max() < 2.
        entrance = planner.physical_domain.evaluate(seeds[0], np.array([0.]))[:, 0]
        velocity = planner.physical_domain.evaluate(seeds[0], np.array([0.]), 1)[:, 0]
        np.testing.assert_allclose(entrance, np.broadcast_to(planner.start.position.numpy(), (2, 3)))
        np.testing.assert_allclose(velocity, np.broadcast_to(planner.start.velocity.numpy(), (2, 3)))
    torch.testing.assert_close(outcomes[0][0], outcomes[1][0], rtol=0., atol=0.)
    for first, second in zip(outcomes[0][1], outcomes[1][1]):
        torch.testing.assert_close(first, second, rtol=0., atol=0.)


def test_legal_nonconverged_outputs_supply_tracking_and_control_proposals():
    """Finite failed states retain their status while proposing free coefficients."""
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    flags = [np.zeros_like(row) for row in flags]
    planner, initialize, warm_starts = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
    with patch.object(planner.physical_domain, "fit_reference",
                      wraps=planner.physical_domain.fit_reference) as fit:
        seeds, counts = planner.reference_seeds()
    fit.assert_not_called()
    assert counts == [0]*6
    initialize.assert_called_once()
    assert len(planner.reference_ik_chains) == 2
    assert bool(torch.isfinite(seeds).all())
    np.testing.assert_array_equal(initialize.call_args.args[0][0].numpy(),
        [chain["joint_positions"] for chain in planner.reference_ik_chains])
    assert planner.initial_seed_strategies == ["reference_ik_control_polygon"]*2
    planner._refine_reference_seeds.assert_called_once()
    for query, record in enumerate(planner.reference_ik_queries):
        assert record["warm_start_from_query"] == (None if query == 0 else query-1)
        assert all(not row["report"]["native_pose_ik_success"] for row in record["candidates"])
        assert warm_starts[query].shape == (1, 1, 3)
        if query:
            assert any(np.array_equal(warm_starts[query].numpy()[0, 0], row)
                       for row in pools[query-1] if np.abs(row).max() <= 2.)
    for chain in planner.reference_ik_chains:
        assert chain["query_statuses"] == ["APPROXIMATE_PROPOSAL"]*6
        assert chain["interpolated_queries"] == []
        assert not chain["reference_success"]
        assert chain["selection_cost"]["nonconverged_candidate_count"] == 6
        assert np.abs(chain["joint_positions"]).max() < 2.


@pytest.mark.parametrize("failed_check", ["task_satisfied", "model_valid"])
def test_native_success_and_unestablished_static_verdicts_remain_separate(failed_check):
    """Stored False values without a violation do not fabricate measured failures."""
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    planner, initialize, warm_starts = fixtures.ReferenceSeedDiversityTests.planner_for(
        pools, flags, report_for=lambda row, success, parameter: {failed_check: False})
    seeds, counts = planner.reference_seeds()
    assert counts == [3]*6
    assert planner.reference_ik_distinct_counts == [3]*6
    assert len(planner.reference_ik_chains) == 2
    initialize.assert_called_once()
    assert bool(torch.isfinite(seeds).all())
    for query, record in enumerate(planner.reference_ik_queries):
        assert record["reference_status"] == "SOLVED"
        assert all(not row["report"][failed_check] for row in record["candidates"])
        warm = warm_starts[query].numpy()
        assert warm.shape == (1, 1, 3)
        if query == 0:
            np.testing.assert_array_equal(warm[0], planner.start.position.numpy())
        else:
            assert any(np.array_equal(warm[0, 0], row) for row in pools[query-1][flags[query-1]])
    failure_count = "static_task_not_established_count" if failed_check == "task_satisfied" else "static_model_not_established_count"
    assert all(chain["selection_cost"][failure_count] == len(pools) for chain in planner.reference_ik_chains)
    assert all(chain["selection_cost"]["static_task_failed_count"] == 0
               and chain["selection_cost"]["static_model_failed_count"] == 0
               for chain in planner.reference_ik_chains)


def test_broken_native_values_keep_json_safe_failure_and_convergence_evidence():
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    planner, _, _ = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
    planner.ik.solve_pose = lambda *args, **kwargs: SimpleNamespace(
        js_solution=SimpleNamespace(position=torch.full((1, 4, 3), float("nan"))),
        success=torch.zeros((1, 4), dtype=torch.bool),
        position_error=torch.full((1, 4), float("inf")),
        rotation_error=torch.full((1, 4), .3))
    seeds, counts = planner.reference_seeds()
    assert counts == [0]*6 and planner.reference_ik_chains == []
    assert bool(torch.isfinite(seeds).all())
    json.dumps(planner.reference_ik_queries, allow_nan=False)
    candidate = planner.reference_ik_queries[0]["candidates"][0]
    assert candidate["joint_position"] == [None]*3
    assert candidate["report"]["nominal_position_error_m"] is None
    assert candidate["report"]["nominal_rotation_error_rad"] == pytest.approx(.3)
# END native success, numerical usability and continuous task acceptance remain distinct.


# BEGIN complete numerical chains enter one physical initialization independently of success.
@pytest.mark.parametrize("solved", [(), (2,), (0, 1), (0, 2, 4), tuple(range(6))])
def test_every_legal_proposal_enters_control_polygon_with_fixed_clock_and_entry(solved):
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    for index in range(len(flags)):
        if index not in solved:
            flags[index] = np.zeros_like(flags[index])
    planner, initialize, _ = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
    domain = planner.physical_domain
    original_map, original_clock = domain._curve_map, dict(planner.formal_clock)
    q0, dq0 = planner.start.position.clone(), planner.start.velocity.clone()
    with patch.object(domain, "fit_reference", wraps=domain.fit_reference) as fit:
        seeds, _ = planner.reference_seeds()
    assert domain._curve_map is original_map and planner.formal_clock == original_clock
    assert seeds.shape == (1, 2, 6, 3)
    assert bool(domain.feasible(seeds).all())
    np.testing.assert_array_equal(domain.evaluate(seeds[0], [0.])[:, 0], q0.expand(2, -1).numpy())
    np.testing.assert_allclose(domain.evaluate(seeds[0], [0.], 1)[:, 0],
                               dq0.expand(2, -1).numpy(), rtol=0, atol=1e-14)
    initialize.assert_called_once()
    np.testing.assert_array_equal(initialize.call_args.args[0][0].numpy(),
        [chain["joint_positions"] for chain in planner.reference_ik_chains])
    assert planner.request_work["ik_queries"] == 6*planner.budget.ik_seeds
    planner._refine_reference_seeds.assert_called_once()
    fit.assert_not_called()
    for chain in planner.reference_ik_chains:
        expected = ["SOLVED" if index in solved else "APPROXIMATE_PROPOSAL" for index in range(6)]
        assert chain["query_statuses"] == expected


def test_changed_legal_failed_states_change_numerical_proposals_without_claiming_success():
    proposals, tracking = [], []
    for failed_value in (-.8, .8):
        pools = [np.array([[.1+.01*index, .02*index, 0.]]) for index in range(6)]
        flags = [np.array([index in (0, 2, 4)]) for index in range(6)]
        for index in (1, 3, 5):
            pools[index] = np.full((1, 3), failed_value)
        planner, initialize, warm = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags, seed_count=1)
        planner.reference_seeds()
        proposals.append(initialize.call_args.args[0].clone())
        tracking.append(warm)
        assert planner.reference_ik_chains[0]["query_statuses"] == [
            "SOLVED", "APPROXIMATE_PROPOSAL", "SOLVED", "APPROXIMATE_PROPOSAL", "SOLVED", "APPROXIMATE_PROPOSAL"]
    assert not torch.equal(proposals[0], proposals[1])
    torch.testing.assert_close(proposals[0][:, :, [0, 2, 4]], proposals[1][:, :, [0, 2, 4]], rtol=0, atol=0)
    assert not torch.equal(tracking[0][2], tracking[1][2])
# END proposal values and native convergence evidence remain separate inputs.


# BEGIN every native query retains the original cumulative candidate quota.
def test_required_ik_exhaustion_stops_before_native_submission():
    from manipulation.planning import PlanningWorkExhausted
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    planner, _, calls = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
    limit = planner.budget.max_validation_points*planner.budget.ik_seeds*planner.budget.maximum_trajectory_solves
    planner.request_work["ik_queries"] = limit
    with pytest.raises(PlanningWorkExhausted):
        planner._solve_ik(.5, planner.start.position[:, None])
    assert not calls and planner.request_work["ik_queries"] == limit


def test_failed_native_submission_retains_the_consumed_candidate_count():
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    planner, _, _ = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
    with patch.object(planner.ik, "solve_pose", side_effect=RuntimeError("submitted query failed")) as solve:
        with pytest.raises(RuntimeError, match="submitted query failed"):
            planner.reference_seeds()
    solve.assert_called_once()
    assert planner.request_work["ik_queries"] == planner.budget.ik_seeds
    assert planner.reference_ik_queries == []
# END failure after submission does not replenish the request budget.


# BEGIN supplied Hermite curves propose coefficients through the existing clock map.
def test_joint_reference_is_sampled_once_in_active_order_before_complete_optimization():
    from manipulation.planning import JointTrajectory
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    planner, initialize, calls = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
    planner.source_progress_map = planner.progress_map
    planner.reference_time_map = None
    times = np.array([0., .4, 1.2, 2.])
    source_q = np.stack((.1*times**2, -.02*times**2, .03*times**2), -1)
    source_dq = np.stack((.2*times, -.04*times, .06*times), -1)
    source = JointTrajectory(tuple(reversed(planner.joint_names)), times, source_q, source_dq,
        np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    originals = [value.copy() for value in (source.times, source.position, source.velocity)]
    with patch.object(planner.physical_domain, "fit_reference", wraps=planner.physical_domain.fit_reference) as fit:
        seeds, success = planner.reference_seeds(joint_reference=source)
    expected_times = planner.physical_domain._curve_map.reference_parameters*source.duration
    expected_controls = source.sample(expected_times)[0][:, ::-1]
    initialize.assert_called_once()
    np.testing.assert_allclose(initialize.call_args.args[0][0].numpy(),
        np.broadcast_to(expected_controls, (2, *expected_controls.shape)), atol=1e-14)
    fit.assert_not_called()
    planner._refine_reference_seeds.assert_called_once()
    assert success == [] and calls == [] and planner.request_work["ik_queries"] == 0
    assert planner.initial_seed_strategies == ["recorded_joint_control_polygon"]*2
    assert bool(planner.physical_domain.feasible(seeds).all())
    for actual, original in zip((source.times, source.position, source.velocity), originals):
        np.testing.assert_array_equal(actual, original)


def test_joint_reference_sampling_preserves_bound_clock_and_motion_inside_a_pause():
    from manipulation.planning import JointTrajectory, _ProgressMap
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    planner, initialize, calls = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
    planner.source_progress_map = _ProgressMap(np.array([0., .25, .5, 1.]),
                                               np.array([0., .5, .5, 1.]))
    planner.reference_time_map = _ProgressMap(np.array([0., .4, 1.]), np.array([0., .8, 1.]))
    planner.progress_map = _ProgressMap(np.array([0., .125, .25, .4, 1.]),
                                        np.array([0., .5, .5, .8, 1.]))
    # Only endpoint Hermite breaks are needed for this exact quadratic. Its
    # source time map separately records nontrivial motion during a pause.
    times = np.array([0., 2.])
    source = JointTrajectory(planner.joint_names, times,
        np.broadcast_to((.1*times**2)[:, None], (2, 3)).copy(),
        np.broadcast_to((.2*times)[:, None], (2, 3)).copy(),
        np.array([0., .2, .8, 1.]), np.array([0., .5, .5, 1.]), (), (), {})
    planner.reference_seeds(joint_reference=source)
    native_u = planner.physical_domain._curve_map.reference_parameters
    original_u = np.interp(native_u, [0., .4, 1.], [0., .8, 1.])
    expected_source_u = np.interp(original_u, [0., .25, .5, 1.], [0., .2, .8, 1.])
    expected = .1*(2.*expected_source_u)**2
    np.testing.assert_allclose(initialize.call_args.args[0][0, 0].numpy(),
        np.broadcast_to(expected[:, None], (6, 3)), atol=1e-14)
    assert calls == []
# END the supplied candidate remains distinct from its new numerical proposal.
