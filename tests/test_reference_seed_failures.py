"""CPU regressions for numerical IK proposals and independent success evidence.

The analytic planner fixture supplies native-shaped positions and residuals;
the production coarse fitter preserves the measured entrance and bound clock.
These tests exercise proposal construction, not robot reachability or final
task/scene qualification.
"""

# BEGIN finite position-domain proposals remain distinct from native pose success.
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning_ik import select_ik_chains
import test_manipulation_planning as fixtures


def _report(solved):
    return {"task_satisfied": True, "model_valid": True,
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
        planner, constant, warm_starts = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
        seeds, _ = planner.reference_seeds()
        outcomes.append((seeds, warm_starts))
        constant.assert_not_called()
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


def test_legal_nonconverged_outputs_supply_tracking_and_approximate_coarse_nodes():
    """An entirely nonconverged sequence can still carry numerical path information."""
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    flags = [np.zeros_like(row) for row in flags]
    planner, constant, warm_starts = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
    seeds, counts = planner.reference_seeds()
    assert counts == [0]*6
    constant.assert_not_called()
    assert len(planner.reference_ik_chains) == 2
    assert bool(torch.isfinite(seeds).all())
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
def test_native_success_and_static_failure_evidence_survive_numerical_initialization(failed_check):
    """Numerical proposals retain the evidence that whole-curve qualification consumes."""
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    planner, constant, warm_starts = fixtures.ReferenceSeedDiversityTests.planner_for(
        pools, flags, report_for=lambda row, success, parameter: {failed_check: False})
    seeds, counts = planner.reference_seeds()
    assert counts == [3]*6
    assert planner.reference_ik_distinct_counts == [3]*6
    assert len(planner.reference_ik_chains) == 2
    constant.assert_not_called()
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
    failure_count = "static_task_unqualified_count" if failed_check == "task_satisfied" else "static_model_unqualified_count"
    assert all(chain["selection_cost"][failure_count] == len(pools) for chain in planner.reference_ik_chains)


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
