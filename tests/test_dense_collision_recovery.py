"""CPU checks of selected native metrics promoted to exact Hermite queries."""

# BEGIN native metric layout, physical-time selection and capacity regressions.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning import _dense_collision_witnesses


def selected_result(values, *, collision, count=9, scene_update=None):
    horizon = len(next(iter(values.values())))
    constraints = SimpleNamespace(names=list(values), values=[
        torch.tensor(value, dtype=torch.float32).reshape(1, horizon, 1) for value in values.values()])
    return SimpleNamespace(batch_size=1, num_seeds=1,
        interpolated_last_tstep=torch.tensor([[count]]),
        interpolated_trajectory=SimpleNamespace(position=torch.zeros(1, 1, horizon, 2)),
        # BEGIN the fixture declares collision evidence independently of hard totals.
        interpolated_metrics=SimpleNamespace(debug=torch.stack((
            torch.tensor(collision, dtype=torch.float32),
            torch.zeros(horizon) if scene_update is None else torch.tensor(scene_update, dtype=torch.float32)),
            dim=-1).reshape(1, horizon, 2),
            costs_and_constraints=SimpleNamespace(constraints=constraints)))
        # END no production fallback from process_scene is exercised by these tests.


def select(result, *, anchors=(), capacity=8):
    return _dense_collision_witnesses(result, regular_horizon=3,
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


def test_existing_regular_peak_does_not_hide_an_off_grid_peak_in_same_interval():
    collision = [0, 0, 2, 3, 100, 4, 0, 0, 0, 1000, 2000]
    result = selected_result({"process_scene": collision}, collision=collision)
    assert select(result)["parameters"] == [.625]
    assert select(result, anchors=(.625,))["parameters"] == [.375]


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
