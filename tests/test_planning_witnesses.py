"""CPU regressions for failure times retained across candidates and refinements."""

# BEGIN attempt-wide witness and pre-fold native constraint regressions.
from copy import deepcopy

import numpy as np
import pytest
import torch

from manipulation.planning_diagnostics import (
    AttemptWitnesses, WitnessCapacityError, capture_component_failures,
)


def native_trace(*times, normalized=False, label="interpolated"):
    key = "normalized_time" if normalized else "time_s"
    return {label: {"source_trace": {"status": "recorded", "component_failures": [
        {key: value, "component": "source_world_collision", "raw_residual": .01}
        for value in times]}}}


def merge(witnesses, *, existing=(), fixed=(), regular=(), maximum=16):
    return witnesses.merged_parameters(fixed_parameters=fixed, existing_parameters=existing,
        regular_parameters=regular, max_points=maximum)


def test_different_seed_native_and_completed_export_failures_all_survive():
    witnesses = AttemptWitnesses()
    first_seed = native_trace(.4, 1., label="rollout")
    first_seed["interpolated"] = native_trace(.6)["interpolated"]
    second_seed = native_trace(1.4)
    witnesses.add_native(first_seed, duration=2.)
    witnesses.add_native(second_seed, duration=2.)
    witnesses.add_validation({"u": [.1, .2, .8, .9], "violations": [0., .03, .04, 0.]})
    np.testing.assert_array_equal(witnesses.parameters, [.2, .3, .5, .7, .8])
    active = merge(witnesses, fixed=(.5,), regular=(0., .2, 1.))
    np.testing.assert_array_equal(active, [.3, .7, .8])
    # The next attempt moves the failure; neither earlier seed nor the
    # preceding independently checked curve may lose its refinement time.
    witnesses.add_validation({"u": [.1, .6], "violations": [.02, 0.]})
    np.testing.assert_array_equal(merge(witnesses, existing=active, fixed=(.5,), regular=(0., .2, 1.)),
                                  [.1, .3, .7, .8])


def test_capture_parameters_prevent_float32_endpoint_time_drift():
    witnesses = AttemptWitnesses()
    native_time = float(np.float32(2.4))
    metrics = native_trace(native_time)
    metrics["interpolated"]["source_trace"]["component_failures"][0]["normalized_time"] = 1.
    witnesses.add_native(metrics, duration=2.4)
    np.testing.assert_array_equal(witnesses.parameters, [1.])
    np.testing.assert_array_equal(merge(witnesses, regular=(0., 1.), maximum=0), [])


def test_explicit_native_joint_times_survive_missing_source_trace_without_index_guessing():
    witnesses = AttemptWitnesses()
    metrics = {"rollout": {"source_trace": {"status": "unavailable"},
        "violating_constraint_samples": [
            {"constraint": "cspace", "index": [2, 0], "normalized_time": .4},
            {"constraint": "process_scene", "index": [9, 0]}]},
        "interpolated": {"source_trace": {"status": "unmatched"},
            "violating_constraint_samples": [{"constraint": "cspace", "normalized_time": .6}]}}
    witnesses.add_native(metrics, duration=2.)
    np.testing.assert_array_equal(witnesses.parameters, [.4, .6])


def test_capacity_failure_never_discards_a_witness_or_mutates_active_parameters():
    witnesses = AttemptWitnesses()
    witnesses.add_validation({"u": [.2, .4, .6, .8], "violations": [1., 1., 1., 1.]})
    existing = np.array([.1, .2])
    before = existing.copy()
    with pytest.raises(WitnessCapacityError, match="requires 4 points, allowed 3"):
        merge(witnesses, existing=existing, fixed=(.4,), maximum=3)
    np.testing.assert_array_equal(existing, before)
    np.testing.assert_array_equal(witnesses.parameters, [.2, .4, .6, .8])
    np.testing.assert_array_equal(merge(witnesses, existing=existing, fixed=(.4,), maximum=4),
                                  [.1, .2, .6, .8])


def test_only_exact_duplicate_times_share_a_query_and_existing_queries_are_monotone():
    witnesses = AttemptWitnesses()
    next_time = float(np.nextafter(np.float32(.5), np.float32(1.)))
    witnesses.add_native(native_trace(.5, .5, next_time, normalized=True), duration=2.)
    np.testing.assert_array_equal(witnesses.parameters, [.5, next_time])
    # An exact fixed anchor covers .5 but cannot certify the adjacent time.
    np.testing.assert_array_equal(merge(witnesses, fixed=(.5,), maximum=1), [next_time])
    # A later supplied regular grid cannot erase previously active anchors.
    np.testing.assert_array_equal(merge(witnesses, existing=(.25,), fixed=(.5,), regular=(.25,), maximum=2),
                                  [.25, next_time])
    snapshot = witnesses.parameters
    snapshot[:] = 0.
    np.testing.assert_array_equal(witnesses.parameters, [.5, next_time])


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
