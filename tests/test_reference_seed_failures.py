"""CPU counterexamples for failed-reference isolation and successful anchors.

The existing analytic planner fixture supplies declared native IK evidence;
the real cubic fitter retains measured q0/dq0 and physical velocity bounds.
These tests make no claim about native robot reachability or source geometry.
"""

# BEGIN failed IK evidence cannot become the next numerical seed or a joint target.
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


def test_failed_gap_uses_actual_successful_anchor_times_and_measured_entrance():
    """A huge failed posture cannot add a vertex or shorten a real edge time."""
    times = np.array([.1, .5, 1., 2., 3.])
    pools = [np.array([[value]]) for value in (100., 1., -100., 2., 100.)]
    reports = [[_report(index in (1, 3))] for index in range(len(times))]
    chain, = select_ik_chains(pools, reports, [[0.]], times, [[-1.], [1.]],
                             count=2, joint_tolerance=1e-5)
    np.testing.assert_allclose(np.asarray(chain["joint_positions"])[:, 0], [.2, 1., 4./3., 2., 2.])
    assert chain["candidate_indices"] == [None, 0, None, 0, None]
    assert chain["anchor_query_indices"] == [1, 3]
    assert chain["query_statuses"] == ["FALLBACK_SEED", "SOLVED", "FALLBACK_SEED", "SOLVED", "FALLBACK_SEED"]
    assert chain["selection_cost"]["normalized_velocity_energy"] == pytest.approx((2.+1./1.5)/3.)


def test_extreme_failed_poses_do_not_change_warm_starts_or_fitted_seed_branches():
    """Front, interior and trailing failures change evidence, never motion."""
    outcomes = []
    failed = (0, 2, 5)
    for failed_value in (-100., 100.):
        _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
        for query in failed:
            pools[query] = np.full_like(pools[query], failed_value)
            flags[query] = np.zeros_like(flags[query])
        planner, constant, warm_starts = fixtures.ReferenceSeedDiversityTests.planner_for(pools, flags)
        seeds, _ = planner.reference_seeds()
        outcomes.append((seeds, warm_starts))
        constant.assert_not_called()
        assert len(planner.reference_ik_chains) == 2
        for query, last_success in enumerate((None, None, 1, 1, 3, 4)):
            expected = (planner.start.position.numpy() if last_success is None
                        else pools[last_success][flags[last_success]])
            assert {tuple(row) for row in warm_starts[query].numpy()[0]} == {tuple(row) for row in expected}
            record = planner.reference_ik_queries[query]
            assert record["warm_start_from_query"] == last_success
            assert record["reference_status"] == ("FAILED_REFERENCE" if query in failed else "SOLVED")
            assert record["seed_status"] == ("FALLBACK_SEED" if query in failed else "SOLVED")
        for chain in planner.reference_ik_chains:
            assert chain["anchor_query_indices"] == [1, 3, 4]
            assert all(chain["candidate_indices"][query] is None for query in failed)
            assert np.abs(chain["joint_positions"]).max() < 2.
        # Static source checks on a prior query cannot certify a later query;
        # exactly the current native output pool is screened each time.
        assert planner._rank_ik_candidates.call_count == len(pools)
        entrance = planner.physical_domain.evaluate(seeds[0], np.array([0.]))[:, 0]
        velocity = planner.physical_domain.evaluate(seeds[0], np.array([0.]), 1)[:, 0]
        np.testing.assert_allclose(entrance, np.broadcast_to(planner.start.position.numpy(), (2, 3)))
        np.testing.assert_allclose(velocity, np.broadcast_to(planner.start.velocity.numpy(), (2, 3)))
    torch.testing.assert_close(outcomes[0][0], outcomes[1][0], rtol=0., atol=0.)
    for first, second in zip(outcomes[0][1], outcomes[1][1]):
        torch.testing.assert_close(first, second, rtol=0., atol=0.)


@pytest.mark.parametrize("failed_check", ["task_satisfied", "model_valid"])
def test_pose_convergence_without_static_feasibility_is_not_a_solved_reference(failed_check):
    _, _, pools, flags = fixtures.ReferenceSeedDiversityTests.two_tracks()
    planner, constant, warm_starts = fixtures.ReferenceSeedDiversityTests.planner_for(
        pools, flags, report_for=lambda row, success, parameter: {failed_check: False})
    seeds, counts = planner.reference_seeds()
    assert counts == [3]*6  # Native pose convergence is retained separately.
    assert planner.reference_ik_distinct_counts == [0]*6
    assert planner.reference_ik_chains == []
    assert planner.initial_seed_strategies == ["measured_state_constant"]*2
    assert constant.call_count == 2
    assert bool(torch.isfinite(seeds).all())
    assert all(torch.equal(value[:, 0], planner.start.position) for value in warm_starts)
    assert all(row["reference_status"] == "FAILED_REFERENCE" for row in planner.reference_ik_queries)


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
# END interpolated fallback samples make no claim about the failed target pose.
