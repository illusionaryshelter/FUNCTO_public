"""CPU checks of selected native metrics promoted to exact Hermite queries."""

# BEGIN native metric layout, physical-time selection and capacity regressions.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning import _dense_collision_witnesses


def selected_result(values, *, collision, count=9, scene_update=None,
                    ordinary_values=None, ordinary_collision=None):
    horizon = len(next(iter(values.values())))
    constraints = SimpleNamespace(names=list(values), values=[
        torch.tensor(value, dtype=torch.float32).reshape(1, horizon, 1) for value in values.values()])
    # BEGIN fixtures supply actual ordinary evidence, never an implicit fallback.
    if ordinary_values is None:
        ordinary_values = {name: [0.]*3 for name in values}
    ordinary_horizon = len(next(iter(ordinary_values.values())))
    ordinary = SimpleNamespace(names=list(ordinary_values), values=[
        torch.tensor(value, dtype=torch.float32).reshape(1, ordinary_horizon, 1)
        for value in ordinary_values.values()])
    ordinary_collision = ([0.]*ordinary_horizon if ordinary_collision is None else ordinary_collision)
    # BEGIN explicit zero coverage preserves collision-only witness fixtures.
    ordinary_debug = torch.stack((torch.tensor(ordinary_collision, dtype=torch.float32),
                                  torch.zeros(ordinary_horizon), torch.zeros(ordinary_horizon)), -1)[None]
    # END all native evidence uses the one current three-channel contract.
    # END hard scene totals do not stand in for the collision channel.
    return SimpleNamespace(batch_size=1, num_seeds=1,
        metrics=SimpleNamespace(debug=ordinary_debug,
            costs_and_constraints=SimpleNamespace(constraints=ordinary)),
        interpolated_last_tstep=torch.tensor([[count]]),
        interpolated_trajectory=SimpleNamespace(position=torch.zeros(1, 1, horizon, 2)),
        # BEGIN the fixture declares collision evidence independently of hard totals.
        interpolated_metrics=SimpleNamespace(debug=torch.stack((
            torch.tensor(collision, dtype=torch.float32),
            torch.zeros(horizon) if scene_update is None else torch.tensor(scene_update, dtype=torch.float32),
            torch.zeros(horizon)), dim=-1).reshape(1, horizon, 3),
            costs_and_constraints=SimpleNamespace(constraints=constraints)))
        # END no production fallback from process_scene is exercised by these tests.


def select(result, *, anchors=(), capacity=8, regular_horizon=3):
    return _dense_collision_witnesses(result, regular_horizon=regular_horizon,
        anchor_parameters=anchors, capacity=capacity)


def test_padding_folded_anchor_terms_and_other_constraints_never_become_times():
    collision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 2000]
    result = selected_result({
        "process_scene": collision,
        "anchor_self_collision": [1000]*11,
        "skill_process": [1000]*11}, collision=collision)
    assert select(result)["parameters"] == []
    # A full prefix can place folded anchors directly at its final index.
    result.interpolated_last_tstep.fill_(11)
    result.interpolated_metrics.costs_and_constraints.constraints.values[0][0, 9] = 0
    result.interpolated_metrics.debug[0, 9] = 0
    assert select(result)["parameters"] == []


def test_already_detected_interval_does_not_take_another_recovery_check():
    # BEGIN an ordinary positive witness supersedes the old adjacent-peak rule.
    collision = [0, 0, 2, 3, 100, 4, 0, 0, 0, 1000, 2000]
    result = selected_result({"process_scene": collision}, collision=collision,
        ordinary_values={"process_scene": [0., 100., 0.]}, ordinary_collision=[0., 100., 0.])
    assert select(result)["parameters"] == []
    assert select(result, anchors=(.625,))["parameters"] == []
    # END this interval has detection coverage even though its collision remains.


# BEGIN recovery distinguishes each sampled interval and actual collision channel.
def test_constant_positive_collision_has_no_missing_detection_interval():
    result = selected_result({"process_scene": [1.]*9}, collision=[1.]*9,
        ordinary_values={"process_scene": [1.]*3}, ordinary_collision=[1.]*3)
    assert select(result)["parameters"] == []
    assert select(result)["peak_count"] == 0


def test_known_interval_does_not_suppress_separate_missed_interval_or_channel():
    collision = [1., 1., 0., 0., 0., 2., 0., 0., 0.]
    result = selected_result({"process_scene": collision,
        "self_collision": [0., 0., 0., 3., 0., 0., 0., 0., 0.]}, collision=collision,
        ordinary_values={"process_scene": [1., 0., 0.], "self_collision": [0.]*3},
        ordinary_collision=[1., 0., 0.])
    assert select(result)["parameters"] == [.375, .625]


def test_nonnested_ordinary_positive_is_matched_by_its_actual_parameter():
    collision = [0., 0., 2., 3., 0., 0., 0., 0., 0.]
    result = selected_result({"process_scene": collision}, collision=collision,
        ordinary_values={"process_scene": [0., 1., 0., 0.]},
        ordinary_collision=[0., 1., 0., 0.])
    # u == 1/3 lies in the dense positive interval [.25, .375], although
    # neither positive dense time belongs to the four-point ordinary grid.
    assert select(result, regular_horizon=4)["parameters"] == []
    result.metrics.debug[0, 1, 0] = 0.
    result.metrics.debug[0, 2, 0] = 1.
    assert select(result, regular_horizon=4)["parameters"] == [.375]


@pytest.mark.parametrize("ordinary_value", [0., float("nan"), float("inf"), -float("inf")])
def test_nonpositive_or_nonfinite_ordinary_value_is_not_a_detection_witness(ordinary_value):
    collision = [0., 0., 2., 3., 4., 2., 0., 0., 0.]
    result = selected_result({"process_scene": collision}, collision=collision,
        ordinary_values={"process_scene": [0., ordinary_value, 0.]},
        ordinary_collision=[0., ordinary_value, 0.])
    assert select(result)["parameters"] == [.375]


def test_folded_ordinary_last_column_does_not_claim_endpoint_detection():
    collision = [0., 0., 0., 0., 0., 0., 2., 3., 4., 0., 1000.]
    result = selected_result({"process_scene": collision}, collision=collision,
        ordinary_values={"process_scene": [0., 0., 1000.]},
        ordinary_collision=[0., 0., 1000.])
    assert select(result)["parameters"] == [.875]


def test_folded_dense_tail_cannot_extend_an_interval_to_a_later_ordinary_positive():
    # BEGIN a folded endpoint must be excluded before interval segmentation.
    collision = [0., 0., 0., 0., 0., 0., 0., 2., 1000.]
    ordinary = [0.]*11
    ordinary[9] = 1.  # u == .9 is after the only real dense positive at .875.
    result = selected_result({"process_scene": collision}, collision=collision,
        ordinary_values={"process_scene": ordinary}, ordinary_collision=ordinary)
    assert select(result, regular_horizon=11)["parameters"] == [.875]
    # END an unrelated known positive cannot erase this missed interval.


@pytest.mark.parametrize("fault", ["missing", "channel", "layout"])
def test_missing_or_misaligned_ordinary_evidence_is_an_explicit_error(fault):
    collision = [0., 1., 0., 0., 0., 0., 0., 0., 0.]
    result = selected_result({"process_scene": collision}, collision=collision)
    if fault == "missing":
        del result.metrics
    elif fault == "channel":
        result.metrics.costs_and_constraints.constraints.names = ["skill_process"]
    else:
        result.metrics.costs_and_constraints.constraints.values[0] = torch.zeros(1, 4, 1)
        result.metrics.debug = torch.zeros(1, 4, 3)
    with pytest.raises(ValueError, match="ordinary.*(evidence|layout)"):
        select(result)
# END routing classifications never relax nonfinite or positive hard residuals.


def test_native_self_peaks_share_capacity_without_using_folded_anchor_self():
    collision = [0, 2, 0, 1, 0, 0, 0, 0, 0]
    result = selected_result({"process_scene": collision,
                              "self_collision": [0, 0, 0, 0, 0, 900, 0, 0, 0],
                              "anchor_self_collision": [0, 0, 0, 0, 0, 0, 0, 8000, 0]}, collision=collision)
    report = select(result, capacity=2)
    assert report["parameters"] == [.125, .625]
    assert report["pending_parameters"] == [.375]
    assert report["peak_count"] == 3
    assert select(result, capacity=0)["parameters"] == []


def test_deduplication_is_exact_native_time_not_an_epsilon_window():
    collision = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    result = selected_result({"process_scene": collision}, collision=collision)
    adjacent = float(np.nextafter(np.float32(.125), np.float32(1)))
    assert select(result, anchors=(adjacent,))["parameters"] == [.125]
    assert select(result, anchors=(.125,))["parameters"] == []
    result.num_seeds = 2
    with pytest.raises(ValueError, match="selected native candidate"):
        select(result)
# END no geometry callback or new validation budget is consumed by selection.


# BEGIN support peaks neither create nor suppress same-query collision witnesses.
def test_mixed_support_and_collision_uses_only_collision_channel_for_times():
    collision = [0, 0, 2, 3, 100, 4, 0, 0, 0, 1000, 2000]
    support = [100000]*11
    result = selected_result({"process_scene": np.asarray(collision)+support},
        collision=collision, scene_update=support)
    assert select(result)["parameters"] == [.625]
    result.interpolated_metrics.debug[..., 0].zero_()
    assert select(result)["parameters"] == []
# END the hard scene total remains positive in both cases.
