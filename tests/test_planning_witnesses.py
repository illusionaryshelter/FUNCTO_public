"""CPU regressions for failure times retained across candidates and refinements."""

# BEGIN attempt-wide witness and pre-fold native constraint regressions.
from copy import deepcopy

import numpy as np
import pytest
import torch

from manipulation.planning_diagnostics import (
    AttemptWitnesses, capture_component_failures,
)


def native_trace(*times, normalized=False, label="interpolated"):
    key = "normalized_time" if normalized else "time_s"
    return {label: {"source_trace": {"status": "recorded", "component_failures": [
        {key: value, "component": "source_world_collision", "raw_residual": .01}
        for value in times]}}}


# BEGIN exchange tests distinguish permanent evidence from bounded optimizer work.
def select(witnesses, *, fixed=(), regular=(), maximum=16):
    return witnesses.select_parameters(fixed_parameters=fixed, regular_parameters=regular,
                                       max_points=maximum)


def test_only_selected_candidate_controls_queries_while_all_evidence_survives():
    witnesses = AttemptWitnesses()
    first_seed = native_trace(.4, 1., label="rollout")
    first_seed["interpolated"] = native_trace(.6)["interpolated"]
    second_seed = native_trace(1.4)
    witnesses.add_native(first_seed, duration=2.)
    witnesses.add_native(second_seed, duration=2.)
    witnesses.add_validation({"u": [.1, .2, .8, .9], "violations": [0., .03, .04, 0.]})
    np.testing.assert_array_equal(witnesses.parameters, [.2, .3, .5, .7, .8])
    np.testing.assert_array_equal(witnesses.current_parameters, [])
    witnesses.set_current(first_seed, duration=2.,
        validation={"u": [.1, .8], "violations": [0., .04]})
    np.testing.assert_array_equal(select(witnesses, fixed=(.5,), regular=(0., .2, 1.)), [.3, .8])
    # Later observer/other-seed evidence cannot overwrite the continuation
    # selected above. Its full history still belongs to final export checking.
    witnesses.add_native(native_trace(.2, normalized=True), duration=2.)
    witnesses.add_validation({"u": [.1, .6], "violations": [.02, 0.]})
    np.testing.assert_array_equal(witnesses.current_parameters, [.2, .3, .5, .8])
    np.testing.assert_array_equal(select(witnesses, fixed=(.5,), regular=(0., .2, 1.)), [.3, .8])
    np.testing.assert_array_equal(witnesses.parameters, [.1, .2, .3, .5, .7, .8])


def test_capture_parameters_prevent_float32_endpoint_time_drift():
    witnesses = AttemptWitnesses()
    native_time = float(np.float32(2.4))
    metrics = native_trace(native_time)
    metrics["interpolated"]["source_trace"]["component_failures"][0]["normalized_time"] = 1.
    witnesses.set_current(metrics, duration=2.4)
    np.testing.assert_array_equal(witnesses.parameters, [1.])
    np.testing.assert_array_equal(select(witnesses, regular=(0., 1.), maximum=0), [])
    assert witnesses.last_selection["pending_parameters"] == []


def test_explicit_native_joint_times_survive_missing_source_trace_without_index_guessing():
    witnesses = AttemptWitnesses()
    metrics = {"rollout": {"source_trace": {"status": "unavailable"},
        "violating_constraint_samples": [
            {"constraint": "cspace", "index": [2, 0], "normalized_time": .4},
            {"constraint": "process_scene", "index": [9, 0]}]},
        "interpolated": {"source_trace": {"status": "unmatched"},
            "violating_constraint_samples": [{"constraint": "cspace", "normalized_time": .6}]}}
    witnesses.set_current(metrics, duration=2.)
    np.testing.assert_array_equal(witnesses.parameters, [.4, .6])
    np.testing.assert_array_equal(select(witnesses), [.4, .6])


def test_twenty_one_current_failures_rotate_through_sixteen_slots_without_loss():
    witnesses = AttemptWitnesses()
    # This is the actual 24 dense times minus their three 81-grid queries;
    # comparison uses their native float32 values without epsilon merging.
    dense = np.arange(482, 506, dtype=np.float32)/np.float32(640)
    regular = np.arange(81, dtype=np.float32)/np.float32(80)
    metrics = native_trace(*dense, normalized=True)
    witnesses.set_current(metrics, duration=16.)
    first = select(witnesses, regular=regular)
    first_report = witnesses.last_selection
    assert len(first) == 16 and len(first_report["pending_parameters"]) == 5
    assert first_report["eligible_count"] == 21
    # Reselecting the same candidate must not reset fairness or erase history.
    witnesses.set_current(metrics, duration=16.)
    second = select(witnesses, regular=regular)
    assert len(second) == 16
    assert set(first_report["pending_parameters"]) <= set(second)
    assert len(set(first) | set(second)) == 21
    np.testing.assert_array_equal(witnesses.parameters, dense.astype(float))
    assert witnesses.last_selection["slack_certified"] is False


def test_historical_dense_collisions_do_not_displace_current_fixed_functional_queries():
    witnesses = AttemptWitnesses()
    dense = np.arange(482, 506, dtype=np.float32)/np.float32(640)
    witnesses.add_native(native_trace(*dense, normalized=True), duration=16.)
    goals = np.asarray([.836125, .95], dtype=np.float32).astype(float)
    current = native_trace(*goals, normalized=True)
    for row in current["interpolated"]["source_trace"]["component_failures"]:
        row["component"] = "skill_process"
    witnesses.set_current(current, duration=16.)
    regular = np.arange(81, dtype=np.float32)/np.float32(80)
    np.testing.assert_array_equal(select(witnesses, fixed=goals, regular=regular), [])
    assert len(witnesses.parameters) == 26
    assert witnesses.last_selection["covered_parameters"] == goals.tolist()
    assert witnesses.last_selection["pending_parameters"] == []
    # The 21 old off-regular times are still validation obligations; they do
    # not become currently failing merely because the scheduler remembers them.
    assert len(set(dense.astype(float))-set(regular.astype(float))) == 21


def test_only_exact_duplicate_times_share_query_and_returned_evidence_is_owned():
    witnesses = AttemptWitnesses()
    next_time = float(np.nextafter(np.float32(.5), np.float32(1.)))
    witnesses.set_current(native_trace(.5, .5, next_time, normalized=True), duration=2.)
    np.testing.assert_array_equal(witnesses.parameters, [.5, next_time])
    # An exact fixed anchor covers .5 but cannot certify the adjacent time.
    np.testing.assert_array_equal(select(witnesses, fixed=(.5,), maximum=1), [next_time])
    witnesses.parameters[:] = 0.
    witnesses.current_parameters[:] = 0.
    report = witnesses.last_selection
    report["selected_parameters"].clear()
    report["selection_counts"][0]["count"] = 500
    np.testing.assert_array_equal(witnesses.parameters, [.5, next_time])
    np.testing.assert_array_equal(witnesses.current_parameters, [.5, next_time])
    assert witnesses.last_selection["selected_parameters"] == [next_time]
    assert witnesses.last_selection["selection_counts"][0]["count"] == 1


def test_new_candidate_can_retire_queries_without_erasing_history_and_recurrence_returns():
    witnesses = AttemptWitnesses()
    witnesses.set_current(native_trace(.2, .4, normalized=True), duration=1.)
    np.testing.assert_array_equal(select(witnesses), [.2, .4])
    # A current pass supplies no signed clearance. Removing its optimization
    # query is scheduling only; the history must still reach independent export.
    witnesses.set_current({}, duration=1.)
    assert witnesses.last_selection is None
    np.testing.assert_array_equal(select(witnesses), [])
    np.testing.assert_array_equal(witnesses.parameters, [.2, .4])
    witnesses.set_current(native_trace(.4, normalized=True), duration=1.)
    np.testing.assert_array_equal(select(witnesses), [.4])
    assert witnesses.last_selection["selection_counts"] == [{"parameter": .4, "count": 2}]


def test_mixed_residual_units_do_not_change_selection_or_hide_zero_capacity_pending():
    witnesses = AttemptWitnesses()
    metrics = native_trace(.2, .4, .6, normalized=True)
    rows = metrics["interpolated"]["source_trace"]["component_failures"]
    for row, component, value in zip(rows, ("skill_process", "source_world_collision", "cspace"),
                                   (1e-20, 1., 1e20)):
        row.update(component=component, raw_residual=value)
    witnesses.set_current(metrics, duration=1.)
    np.testing.assert_array_equal(select(witnesses, maximum=0), [])
    assert witnesses.last_selection["pending_parameters"] == [.2, .4, .6]
    assert all(row["count"] == 0 for row in witnesses.last_selection["selection_counts"])
    np.testing.assert_array_equal(select(witnesses, maximum=1), [.2])
    np.testing.assert_array_equal(select(witnesses, maximum=1), [.4])


def test_nonfinite_hard_residuals_keep_their_failure_time_and_bad_time_admission_is_atomic():
    witnesses = AttemptWitnesses()
    witnesses.add_validation({"u": [.1, .2, .3, .4], "violations": [np.nan, np.inf, -np.inf, -1.]})
    np.testing.assert_array_equal(witnesses.parameters, [.1, .2, .3])
    with pytest.raises(ValueError, match="normalized physical times"):
        witnesses.add_native(native_trace(.5, 1.01, normalized=True), duration=1.)
    np.testing.assert_array_equal(witnesses.parameters, [.1, .2, .3])
    with pytest.raises(ValueError, match="one hard residual"):
        witnesses.add_validation({"u": [.4, .5], "violations": [1.]})
    np.testing.assert_array_equal(witnesses.parameters, [.1, .2, .3])


def test_failed_current_or_selection_update_leaves_history_priority_and_fairness_unchanged():
    witnesses = AttemptWitnesses()
    witnesses.set_current(native_trace(.2, .4, normalized=True), duration=1.)
    np.testing.assert_array_equal(select(witnesses, maximum=1), [.2])
    previous = witnesses.last_selection
    with pytest.raises(ValueError, match="one hard residual"):
        witnesses.set_current(native_trace(.8, normalized=True), duration=1.,
            validation={"u": [.6, .7], "violations": [1.]})
    with pytest.raises(ValueError, match="normalized physical times"):
        select(witnesses, regular=(1.1,), maximum=1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        select(witnesses, maximum=True)
    assert witnesses.last_selection == previous
    np.testing.assert_array_equal(witnesses.parameters, [.2, .4])
    np.testing.assert_array_equal(witnesses.current_parameters, [.2, .4])
    np.testing.assert_array_equal(select(witnesses, maximum=1), [.4])
# END omitted optimizer queries remain historical validation obligations.


def test_raw_process_and_self_capture_retains_real_anchor_time_before_folding():
    parameters = torch.tensor([[0., .25, .8, 1.], [0., .25, .8, 1.]])
    times = parameters*2.4
    progress = parameters.clone()
    progress[:, 2:] = 1.  # A progress plateau must retain both physical times.
    joints = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    position = torch.zeros((2, 4, 3))
    quaternion = torch.zeros((2, 4, 4))
    components = {"skill_process": torch.tensor([[0., 0., .02, 0.], [0., 0., 0., 0.]]),
        "source_self_collision": torch.tensor([[[0.], [0.], [0.], [0.]],
                                                [[0.], [0.], [0.], [float("nan")]]])}
    before = deepcopy(components)
    rows = capture_component_failures(components, times, progress, joints, position, quaternion,
                                     normalized_time=parameters)
    assert [(row["component"], row["batch_index"], row["ordered_query_index"])
            for row in rows] == [("skill_process", 0, 2), ("source_self_collision", 1, 3)]
    assert rows[0]["normalized_time"] == float(parameters[0, 2])
    assert rows[0]["time_s"] == float(times[0, 2])
    assert rows[1]["normalized_time"] == 1. and np.isnan(rows[1]["raw_residual"])
    assert rows[0]["progress"] == rows[1]["progress"] == 1.
    assert rows[0]["joint_position"] == [4., 5.]
    assert rows[1]["joint_position"] == [14., 15.]
    for name, value in components.items():
        torch.testing.assert_close(value, before[name], equal_nan=True)
    witnesses = AttemptWitnesses()
    witnesses.add_native({"interpolated": {"source_trace": {
        "status": "recorded", "component_failures": rows}}}, duration=2.4)
    np.testing.assert_array_equal(witnesses.parameters, [float(parameters[0, 2]), 1.])
# END witnesses preserve query identity without allocating native solver workspaces.
