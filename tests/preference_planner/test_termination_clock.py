"""Exercise real scheduling APIs and frozen terminal-time continuation.

These CPU tests cover data roles at the preference-planner clock boundary.  No
geometry, optimizer, simulation query or execution decision is performed.
"""

from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]


from manipulation.cubic_constraints import CubicAffineMap
from manipulation.planning import (
    PlanningBudget, _ProgressMap, _bind_event_clock, _continuation_timing,
    _fixed_spline_grid, _initial_timing, _phase_aligned_parameters,
    reference_progress_clock,
)
from manipulation.planning_problem import MotionPhase, PlanningEvent, PlanningProblem, PosePath
from manipulation.preference_planner.terminal import TerminalHold
from manipulation.preference_planner.termination_clock import terminal_clock


def clock_problem():
    """A generic request retains increasing source time and skill progress."""
    poses = np.stack((np.eye(4), np.eye(4), np.eye(4)))
    poses[1, 0, 3] = .1
    poses[2, 0, 3] = .2
    path = PosePath(np.array([0., .5, 1.]), poses, "explicit clock test reference")
    event = PlanningEvent("actuator", "reference", .91, 7, "close", "explicit command")
    return PlanningProblem(("arm",), [0.], [.03], "tool", "world", np.eye(4), np.eye(4),
        (MotionPhase("reference", path),), events=(event,),
        source_timestamps=np.array([0., 2., 4., 10.]),
        source_progress=np.array([0., .2, .4, 1.]))


def fresh_clock(problem):
    """Construct the same already-bound data produced by the base planner."""
    selected, source = _initial_timing(problem,
        PlanningBudget(knots=16, interpolation_buffer_size=768))
    grid = _fixed_spline_grid(intervals=selected.knots+4,
        native_dt=selected.maximum_trajectory_dt,
        interpolation_steps=selected.interpolation_steps,
        interpolation_dt=selected.interpolation_dt)
    before, warp = reference_progress_clock(source, selected.knots,
        stop_progress=problem.stop_progress, anchors=problem.anchors)
    mapping = _bind_event_clock(problem, before, grid["duration"], .002)
    warp = _ProgressMap(_phase_aligned_parameters(before, mapping, warp.u), warp.s)
    return selected, grid, mapping, warp


def test_fresh_request_retains_actual_source_time_warp_and_internal_plateau():
    problem = clock_problem()
    budget, grid, _, _ = fresh_clock(problem)
    # The lower-level monotone clock API explicitly supports progress
    # plateaus, independently of PlanningProblem's stricter source map.
    # This test checks the consumer's coordinate distinction at that API.
    input_clock = _ProgressMap(np.array([0., .2, .4, 1.]), np.array([0., .5, .5, 1.]))
    mapping, warp = reference_progress_clock(input_clock, budget.knots,
        stop_progress=(), anchors=input_clock.s)
    before_u, before_s = mapping.u.copy(), mapping.s.copy()
    u, z, source = terminal_clock(problem=problem, budget=budget, progress_map=mapping,
        reference_time_map=warp, duration=grid["duration"], continuation=None,
        incoming_terminal_contract=None)
    np.testing.assert_array_equal(u, warp.u)
    np.testing.assert_array_equal(z, warp.s)
    internal = mapping.anchor_parameters([.5])[:, 0]
    assert len(internal) == 2
    assert np.interp(internal[0], u, z) < np.interp(internal[1], u, z) < 1.
    np.testing.assert_array_equal(mapping.u, before_u)
    np.testing.assert_array_equal(mapping.s, before_s)
    assert "source-time warp" in source


def test_explicit_rebind_recovers_actual_warp_without_rebinding_events():
    problem = clock_problem()
    budget = PlanningBudget(knots=16, interpolation_buffer_size=768)
    continuation = {"rebind": {"kind": "remaining", "source": "explicit request"},
        "scene_anchors": np.array([0., .5, 1.]), "control_timestep_s": .002,
        "request_time_origin_s": 1.5, "request_time_origin_step": 750,
        "history": [], "goal_identities": None, "timing_events": []}
    selected, mapping = _continuation_timing(problem, budget, continuation)
    grid = _fixed_spline_grid(intervals=selected.knots+4,
        native_dt=selected.maximum_trajectory_dt,
        interpolation_steps=selected.interpolation_steps,
        interpolation_dt=selected.interpolation_dt)
    records = {key: dict(value) for key, value in mapping.event_clock.items()}
    u, z, source = terminal_clock(problem=problem, budget=selected, progress_map=mapping,
        reference_time_map=None, duration=grid["duration"], continuation=continuation,
        incoming_terminal_contract={"requested_hold_start_s": .01})
    fresh_budget, fresh_grid, _, fresh_warp = fresh_clock(problem)
    assert selected.knots == fresh_budget.knots
    assert grid["duration"] == fresh_grid["duration"]
    np.testing.assert_array_equal(u, fresh_warp.u)
    np.testing.assert_array_equal(z, fresh_warp.s)
    assert mapping.event_clock == records
    assert "explicit rebind" in source


def test_ordinary_continuation_carries_requested_time_not_implied_or_progress_time():
    # A prior cubic's requested .935 suffix implies .9 on its old grid.
    # Its semantic requested time is retained even on a new grid.  Skill
    # progress is deliberately already one, so it cannot choose this time.
    contract = {"requested_hold_start_s": 9.35, "implied_constant_start_s": 9.,
                "hold_end_s": 10., "positive_length_hold": True}
    continuation = {"rebind": None, "elapsed": 4.}
    progress = _ProgressMap(np.array([0., .2, 1.]), np.array([.8, 1., 1.]))
    u, z, source = terminal_clock(problem=None, budget=None, progress_map=progress,
        reference_time_map=None, duration=6., continuation=continuation,
        incoming_terminal_contract=contract)
    assert u[np.flatnonzero(z == 1.)[0]] * 6. == 9.35 - 4.
    assert u[np.flatnonzero(z == 1.)[0]] > progress.anchor_parameters([1.])[0, 0]
    assert "terminal-only encoding" in source
    curve = CubicAffineMap([0.], [.03], 21, .3)
    hold = TerminalHold.from_clock(curve, u, z, source=source)
    assert hold.report["requested_hold_start_s"] == 9.35 - 4.
    assert hold.report["implied_constant_start_s"] <= hold.report["requested_hold_start_s"]


def test_continuation_already_in_hold_has_constant_entire_interval():
    u, z, _ = terminal_clock(problem=None, budget=None, progress_map=None,
        reference_time_map=None, duration=1., continuation={"rebind": None, "elapsed": 9.},
        incoming_terminal_contract={"requested_hold_start_s": 8.835, "hold_end_s": 9.3})
    np.testing.assert_array_equal(u, [0., 1.])
    np.testing.assert_array_equal(z, [1., 1.])


def test_zero_length_prior_hold_stays_empty_at_same_physical_endpoint():
    u, z, source = terminal_clock(problem=None, budget=None, progress_map=None,
        reference_time_map=None, duration=6., continuation={"rebind": None, "elapsed": 4.},
        incoming_terminal_contract={"requested_hold_start_s": 10., "hold_end_s": 10.})
    curve = CubicAffineMap([0.], [.03], 21, .3)
    hold = TerminalHold.from_clock(curve, u, z, source=source)
    assert not hold.report["positive_length_hold"]
    assert hold.matrix.shape == (21, 21)
