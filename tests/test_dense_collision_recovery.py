"""CPU checks of selected native metrics promoted to exact Hermite queries."""

# BEGIN native metric layout, physical-time selection and capacity regressions.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning import _dense_collision_witnesses


def selected_result(values, *, count=9):
    horizon = len(next(iter(values.values())))
    constraints = SimpleNamespace(names=list(values), values=[
        torch.tensor(value, dtype=torch.float32).reshape(1, horizon, 1) for value in values.values()])
    return SimpleNamespace(batch_size=1, num_seeds=1,
        interpolated_last_tstep=torch.tensor([[count]]),
        interpolated_trajectory=SimpleNamespace(position=torch.zeros(1, 1, horizon, 2)),
        interpolated_metrics=SimpleNamespace(costs_and_constraints=SimpleNamespace(constraints=constraints)))


def select(result, *, anchors=(), capacity=8):
    return _dense_collision_witnesses(result, regular_horizon=3,
        anchor_parameters=anchors, capacity=capacity)


def test_padding_folded_anchor_terms_and_other_constraints_never_become_times():
    result = selected_result({
        "process_scene": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 2000],
        "anchor_self_collision": [1000]*11,
        "skill_process": [1000]*11})
    assert select(result)["parameters"] == []
    # A full prefix can place folded anchors directly at its final index.
    result.interpolated_last_tstep.fill_(11)
    result.interpolated_metrics.costs_and_constraints.constraints.values[0][0, 9] = 0
    assert select(result)["parameters"] == []


def test_existing_regular_peak_does_not_hide_an_off_grid_peak_in_same_interval():
    result = selected_result({"process_scene": [0, 0, 2, 3, 100, 4, 0, 0, 0, 1000, 2000]})
    assert select(result)["parameters"] == [.625]
    assert select(result, anchors=(.625,))["parameters"] == [.375]


def test_native_self_peaks_share_capacity_without_using_folded_anchor_self():
    result = selected_result({"process_scene": [0, 2, 0, 1, 0, 0, 0, 0, 0],
                              "self_collision": [0, 0, 0, 0, 0, 900, 0, 0, 0],
                              "anchor_self_collision": [0, 0, 0, 0, 0, 0, 0, 8000, 0]})
    report = select(result, capacity=2)
    assert report["parameters"] == [.125, .625]
    assert report["pending_parameters"] == [.375]
    assert report["peak_count"] == 3
    assert select(result, capacity=0)["parameters"] == []


def test_deduplication_is_exact_native_time_not_an_epsilon_window():
    result = selected_result({"process_scene": [0, 1, 0, 0, 0, 0, 0, 0, 0]})
    adjacent = float(np.nextafter(np.float32(.125), np.float32(1)))
    assert select(result, anchors=(adjacent,))["parameters"] == [.125]
    assert select(result, anchors=(.125,))["parameters"] == []
    result.num_seeds = 2
    with pytest.raises(ValueError, match="selected native candidate"):
        select(result)
# END no geometry callback or new validation budget is consumed by selection.
