"""Requested maximum spacings produce one unchanged exact physical clock."""

# BEGIN decimal requests and actual device grids have distinct responsibilities.
from types import SimpleNamespace

import numpy as np
import pytest

from manipulation.planning import (PlanningBudget, _derived_spline_grid,
                                   _fixed_spline_grid, _initial_timing)


@pytest.mark.parametrize("requested,count", [(.01, 1537), (.02, 769), (.025, 385),
                                            (.05, 193), (.1, 97)])
def test_decimal_maximum_spacing_is_derived_before_capacity_admission(requested, count):
    budget = PlanningBudget(interpolation_dt=requested, interpolation_buffer_size=2048)
    grid = _derived_spline_grid(intervals=budget.knots + 4,
        native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
        interpolation_dt=budget.interpolation_dt)
    assert grid["valid_count"] == count
    assert grid["interpolation_dt"] <= float(np.float32(requested))
    assert grid["duration"] == (count - 1) * grid["interpolation_dt"]
    assert grid["duration"] == 12 * grid["knot_dt"]
    assert grid["knot_dt"] == 4 * grid["native_dt"]


def test_small_spacing_reports_real_buffer_capacity_and_can_fit_shorter_clock():
    with pytest.raises(ValueError, match="Interpolation buffer"):
        PlanningBudget(interpolation_dt=.005, interpolation_buffer_size=2048)
    budget = PlanningBudget(maximum_trajectory_dt=.1, interpolation_dt=.005,
                            interpolation_buffer_size=2048)
    grid = _derived_spline_grid(intervals=12, native_dt=budget.maximum_trajectory_dt,
        interpolation_steps=4, interpolation_dt=budget.interpolation_dt)
    assert grid["valid_count"] == 1537


@pytest.mark.parametrize("requested", [.01, .02, .025])
def test_source_binding_retains_duration_and_passes_strict_actual_grid(requested):
    source = SimpleNamespace(source_time_map={"source_times": [0., 1., 4.001],
                                              "progress": [0., .7, 1.]})
    budget = PlanningBudget(interpolation_dt=requested, interpolation_buffer_size=2048)
    selected, mapping = _initial_timing(source, budget)
    grid = _fixed_spline_grid(intervals=selected.knots + 4,
        native_dt=selected.maximum_trajectory_dt, interpolation_steps=selected.interpolation_steps,
        interpolation_dt=selected.interpolation_dt)
    assert 4.001 <= grid["duration"] < 4.001002
    assert grid["interpolation_dt"] <= float(np.float32(requested))
    np.testing.assert_array_equal(mapping.s, [0., .7, 1.])
    np.testing.assert_array_equal(mapping.u, np.array([0., 1., 4.001]) / 4.001)


def test_actual_grid_validator_still_rejects_an_inconsistent_clock():
    with pytest.raises(ValueError, match="Exact output grid"):
        _fixed_spline_grid(intervals=12, native_dt=.2, interpolation_steps=4,
                           interpolation_dt=.01)
# END no test changes the bound event clock, polynomial or collision model.
