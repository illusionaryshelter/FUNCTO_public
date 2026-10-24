"""Numerical local-IK status, one-fit initialization and original clock semantics."""

# BEGIN finite numerical states remain usable independently of native pose success.
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.planning import MotionPlanningError, PlanningWorkExhausted
import test_manipulation_planning as fixtures


@pytest.mark.parametrize("solved", [(), (2,), (0, 1), (0, 2, 4), tuple(range(6))])
def test_finite_failed_outputs_keep_their_values_and_supply_the_next_predecessor(solved):
    flags = np.array([i in solved for i in range(6)])
    samples = np.array([[.1+.01*i, -.02*i, .03*i] for i in range(6)])
    planner, fitting, calls = fixtures.local_reference_fixture(samples, flags)
    original_map = planner.physical_domain._curve_map
    original_clock = dict(planner.formal_clock)
    seeds, counts = planner.reference_seeds()
    assert counts == flags.astype(int).tolist()
    assert seeds.shape == (1, 1, 6, 3) and fitting.call_count == 1
    np.testing.assert_array_equal(fitting.call_args.args[1], samples)
    for index, record in enumerate(planner.reference_ik_queries):
        assert record["reference_status"] == ("SOLVED" if flags[index] else "APPROXIMATE_PROPOSAL")
        if index:
            np.testing.assert_array_equal(calls[index]["q"][0], samples[index-1])
    assert planner.physical_domain._curve_map is original_map
    assert planner.formal_clock == original_clock
    assert planner.request_work["ik_queries"] == 6


def test_finite_out_of_domain_reference_remains_an_initializer_for_original_constraints():
    samples = np.full((6, 3), 3.)
    planner, fitting, calls = fixtures.local_reference_fixture(samples, np.zeros(6, dtype=bool))
    seeds, counts = planner.reference_seeds()
    assert counts == [0]*6 and len(calls) == 6
    np.testing.assert_array_equal(fitting.call_args.args[1], samples)
    constraints, _ = planner.physical_domain.signed_constraints(seeds[0, 0].numpy())
    assert np.max(constraints) > 0.
    assert planner.reference_ik_queries[0]["reference_status"] == "APPROXIMATE_PROPOSAL"


def test_nonfinite_native_state_records_json_safe_failure_without_fitting_or_fill():
    planner, fitting, calls = fixtures.local_reference_fixture()
    planner.reference_ik.solve_pose = lambda *args, **kwargs: SimpleNamespace(
        js_solution=SimpleNamespace(position=torch.full((1, 1, 3), float("nan"))),
        success=torch.zeros((1, 1), dtype=torch.bool),
        position_error=torch.full((1, 1), float("inf")), rotation_error=torch.full((1, 1), .3))
    with pytest.raises(MotionPlanningError, match="nonfinite joint state"):
        planner.reference_seeds()
    record, = planner.reference_ik_queries
    assert record["joint_position"] == [None]*3
    assert record["reference_status"] == "NONFINITE_STATE"
    assert record["nominal_position_error_m"] is None
    assert record["nominal_rotation_error_rad"] == pytest.approx(.3)
    json.dumps(record, allow_nan=False)
    fitting.assert_not_called()
    assert planner.request_work["ik_queries"] == 1


def test_required_ik_exhaustion_stops_before_native_submission():
    planner, _, calls = fixtures.local_reference_fixture()
    limit = planner.budget.max_validation_points*planner.budget.ik_seeds*planner.budget.maximum_trajectory_solves
    planner.request_work["ik_queries"] = limit
    with pytest.raises(PlanningWorkExhausted):
        planner.reference_seeds()
    assert not calls and planner.request_work["ik_queries"] == limit


def test_failed_native_submission_retains_one_consumed_candidate():
    planner, fitting, _ = fixtures.local_reference_fixture()
    with patch.object(planner.reference_ik, "solve_pose", side_effect=RuntimeError("submitted query failed")) as solve:
        with pytest.raises(RuntimeError, match="submitted query failed"):
            planner.reference_seeds()
    solve.assert_called_once()
    assert planner.request_work["ik_queries"] == 1
    assert planner.reference_ik_queries == []
    fitting.assert_not_called()


def test_functional_batch_uses_its_real_shape_and_original_absolute_ledger():
    planner, _, _ = fixtures.local_reference_fixture()
    planner.request_work_baseline["ik_queries"] = 10000
    planner.request_work["ik_queries"] = 10000
    planner._solve_ik(.5, planner.start.position[:, None],
        solver=planner.ik, current_state=planner.start)
    assert planner.request_work["ik_queries"] == 10004
    call = planner.ik.solve_pose.call_args
    assert call.kwargs["return_seeds"] == 4
    assert float(call.kwargs["current_state"].dt) == 0.
# END finite approximations never acquire task or scene acceptance.

# BEGIN supplied Hermite curves propose coefficients through the existing clock map.
def test_joint_reference_is_sampled_once_in_active_order_before_complete_optimization():
    from manipulation.planning import JointTrajectory
    planner, fitting, calls = fixtures.local_reference_fixture()
    times = np.array([0., .4, 1.2, 2.])
    source_q = np.stack((.1*times**2, -.02*times**2, .03*times**2), -1)
    source_dq = np.stack((.2*times, -.04*times, .06*times), -1)
    source = JointTrajectory(tuple(reversed(planner.joint_names)), times, source_q, source_dq,
        np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    originals = [value.copy() for value in (source.times, source.position, source.velocity)]
    seeds, success = planner.reference_seeds(joint_reference=source)
    expected_times = planner.physical_domain._curve_map.reference_parameters*source.duration
    expected_controls = source.sample(expected_times)[0][:, ::-1]
    assert fitting.call_count == 1
    np.testing.assert_allclose([call.args[1] for call in fitting.call_args_list],
        expected_controls[None], atol=1e-14)
    assert success == [] and calls == [] and planner.request_work["ik_queries"] == 0
    assert planner.initial_seed_strategies == ["recorded_joint_samples"]
    for actual, original in zip((source.times, source.position, source.velocity), originals):
        np.testing.assert_array_equal(actual, original)


def test_joint_reference_sampling_preserves_bound_clock_and_motion_inside_a_pause():
    from manipulation.planning import JointTrajectory, _ProgressMap
    planner, fitting, calls = fixtures.local_reference_fixture()
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
    expected_source_u = np.interp(native_u, [0., .125, .25, .4, 1.], [0., .2, .8, .92, 1.])
    expected = .1*(2.*expected_source_u)**2
    np.testing.assert_allclose(fitting.call_args_list[0].args[1],
        np.broadcast_to(expected[:, None], (6, 3)), atol=1e-14)
    assert calls == []
# END the supplied candidate remains distinct from its new numerical proposal.


# BEGIN an added reference terminal interval retains source-curve parameterization.
def test_joint_reference_uses_original_pause_phase_before_added_terminal_interval():
    from manipulation.planning import JointTrajectory, _ProgressMap, reference_progress_clock
    planner, fitting, calls = fixtures.local_reference_fixture()
    source_map = _ProgressMap(np.array([0., .2, .8, 1.]), np.array([0., .5, .5, 1.]))
    planner.source_progress_map = source_map
    planner.progress_map, planner.reference_time_map = reference_progress_clock(
        source_map, planner.budget.knots, stop_progress=(), anchors=source_map.s)
    # The source quadratic continues moving during its own progress pause.
    # It has no terminal pause; the scheduling prior contributes that interval.
    times = np.array([0., 2.])
    source = JointTrajectory(planner.joint_names, times,
        np.broadcast_to((.1*times**2)[:, None], (2, 3)).copy(),
        np.broadcast_to((.2*times)[:, None], (2, 3)).copy(),
        source_map.u, source_map.s, (), (), {})
    planner.reference_seeds(joint_reference=source)
    parameters = planner.physical_domain._curve_map.reference_parameters
    expected_time = 2*np.interp(parameters, planner.reference_time_map.u,
                               planner.reference_time_map.s)
    expected = np.broadcast_to((.1*expected_time**2)[:, None], (6, 3))
    np.testing.assert_allclose(fitting.call_args_list[0].args[1], expected, atol=1e-14)
    assert len(planner.progress_map.anchor_parameters([1.])) == 2
    assert fitting.call_count == 1
    assert calls == []
# END the complete source polynomial supplies one initial coefficient fit.


# BEGIN cancellation preserves already completed native-query evidence.
def test_reference_query_cancellation_retains_completed_work_and_stops_before_fit():
    from manipulation.planning import MotionPlanningError
    planner,fitting,calls=fixtures.local_reference_fixture()
    cancelled=MotionPlanningError('cancelled','declared cancellation')
    planner.check_request.side_effect=[None,cancelled]
    with pytest.raises(MotionPlanningError) as raised:
        planner.reference_seeds()
    assert raised.value is cancelled
    assert len(calls)==len(planner.reference_ik_queries)==1
    assert planner.request_work['ik_queries']==1
    fitting.assert_not_called()
# END request interruption cannot reset submitted native work.
