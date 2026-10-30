"""Reference scheduling, command coverage and free robot terminal states."""

# BEGIN an independent scalar schedule preserves source and actuator semantics.
import numpy as np
import pytest
from scipy.interpolate import BSpline

from manipulation.cubic_trajectory import CubicBasis
from manipulation.gripper_trajectory import FingerTrajectory, stationary_finger_command
from manipulation.planning import (
    PlanningBudget, _ProgressMap, _bind_event_clock, _continuation_timing,
    _fixed_spline_grid, _initial_timing, reference_progress_clock,
)
from manipulation.planning_problem import MotionPhase, PlanningEvent, PlanningProblem, PosePath


def command_problem():
    """A named actuator declaration with a late command and no robot stop."""
    path = PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))),
                    "constant proposal for an unconstrained arm")
    events = (PlanningEvent("begin", "reference", 0., 0, "close", "explicit command"),
              PlanningEvent("finish", "reference", .99, 1, "open", "explicit command"))
    return PlanningProblem(("arm",), [0.], [.03], "tool", "world", np.eye(4), np.eye(4),
        (MotionPhase("reference", path),), events=events,
        source_timestamps=np.array([0., 10.]), source_progress=np.array([0., 1.]))


@pytest.mark.parametrize("count", [6, 8, 12])
def test_source_vertices_and_plateau_phase_survive_independent_reference_clock(count):
    source = _ProgressMap(np.array([0., .3, .55, .8, 1.]),
                          np.array([0., .6, .6, 1.7, 2.]))
    mapping, warp = reference_progress_clock(source, count, stop_progress=(), anchors=source.s)
    # Source times inside an original progress pause remain distinct. The
    # added terminal reference interval has exactly the retained duration.
    times = np.interp(source.u, warp.s, warp.u)
    times[0], times[-1] = 0., (count+3)/(count+4)
    np.testing.assert_allclose(np.interp(times, mapping.u, mapping.s), source.s, atol=1e-14)
    np.testing.assert_allclose(mapping.anchor_parameters([.6])[:, 0], times[[1, 2]], atol=1e-14)
    np.testing.assert_array_equal(mapping.anchor_parameters([2.])[:, 0], [(count+3)/(count+4), 1.])
    assert mapping.anchor_parameters([0.]).shape == (1, 2)
    assert np.all(np.diff(mapping.u.astype(np.float32)) > 0)


def test_declared_stop_preserves_original_scalar_cubic_and_linearization_bound():
    count = 15
    # These coefficients independently define the retained scalar schedule.
    # No robot state enters this oracle or the scheduling function.
    coefficients = np.r_[np.arange(-1, 3)/16., np.arange(3, 9)/16.,
                          [.5]*3, .5+np.arange(1, 7)/12., np.ones(3)]
    exact = BSpline(np.arange(-3., count+8.), coefficients, 3)
    source = _ProgressMap(np.array([0., 1.]), np.array([0., 2.]))
    mapping, warp = reference_progress_clock(source, count,
        stop_progress=(1.,), anchors=(0., 1., 2.))
    u = np.linspace(0., 1., 20001)
    error = np.abs(np.interp(u, warp.u, warp.s)-exact(u*(count+4)))
    assert error.max() <= 1e-6+1e-14
    np.testing.assert_allclose(mapping.anchor_parameters([1.]),
                               [[9/19, 1.], [10/19, 1.]], atol=2e-15)


def test_late_command_has_original_bounded_window_while_arm_terminal_remains_free():
    problem = command_problem()
    budget, source = _initial_timing(problem, PlanningBudget(knots=16, interpolation_buffer_size=768))
    grid = _fixed_spline_grid(intervals=budget.knots+4, native_dt=budget.maximum_trajectory_dt,
        interpolation_steps=budget.interpolation_steps, interpolation_dt=budget.interpolation_dt)

    def fingers(mapping):
        commands = []
        for row in mapping.event_clock.values():
            command = {**row["event"], **{k: v for k, v in row.items() if k != "event"}}
            command["finger_positions"] = [0.] if command["kind"] == "close" else [.038]
            commands.append(command)
        return FingerTrajectory.build(joint_names=("finger",), measured_initial_position=[0.],
            measured_initial_velocity=[0.],
            incoming_command=stationary_finger_command(joint_names=("finger",), position=[0.],
                timestamp_s=0., source="Declared stationary synthetic command", feedback_id="measured entrance"), commands=commands, duration=grid["duration"],
            executable_until=grid["duration"], timestep=.002,
            position_limits=[[0.], [.04]], velocity_limits=[[-.2], [.2]],
            acceleration_limits=[[-15.], [15.]], source="explicit bounded actuator reference",
            feedback_id="measured entrance", timestamp_s=0.)

    # The raw source map leaves less than the mean-speed lower bound. This
    # fixture catches replacing the compiled reference clock with that map.
    raw = _bind_event_clock(problem, source, grid["duration"], .002)
    with pytest.raises(ValueError, match="No C1 finger command"):
        fingers(raw)
    mapping, _ = reference_progress_clock(source, budget.knots,
        stop_progress=problem.stop_progress, anchors=problem.anchors)
    mapping = _bind_event_clock(problem, mapping, grid["duration"], .002)
    hand = fingers(mapping)
    command = hand.segments[-1]
    assert command["ramp_end_time_s"] < grid["duration"]
    assert command["ramp_end_time_s"]-command["start_time_s"] >= .285
    assert set(mapping.event_clock) == set(raw.event_clock)
    assert len(mapping.anchor_parameters([1.])) == 2

    # A nonzero last free coefficient gives terminal robot velocity and
    # acceleration on the same physical clock and progress plateau. There
    # is no structural terminal hold in the joint basis or hand schedule.
    basis = CubicBasis(budget.knots+5, grid["knot_dt"])
    controls = np.zeros((basis.knots, 1)); controls[-1] = .1
    times, q, dq = basis.hermite_nodes(controls, problem.start_q, problem.start_dq)
    acceleration, velocity_term = basis.weights(np.array([1.]), derivative=2)
    ddq = acceleration @ controls + velocity_term[:, None] * problem.start_dq
    assert times[-1] == hand.duration
    assert abs(dq[-1, 0]) > 0.
    assert abs(ddq[0, 0]) > 0.
    np.testing.assert_array_equal(q[0], problem.start_q)
    np.testing.assert_array_equal(dq[0], problem.start_dq)


def test_explicit_rebind_rebuilds_reference_schedule_with_same_event_ticks():
    problem = command_problem()
    budget = PlanningBudget(knots=16, interpolation_buffer_size=768)
    selected, source = _initial_timing(problem, budget)
    anchors = np.unique(np.r_[problem.anchors, .5])
    expected, _ = reference_progress_clock(source, selected.knots,
        stop_progress=problem.stop_progress, anchors=anchors)
    grid = _fixed_spline_grid(intervals=selected.knots+4, native_dt=selected.maximum_trajectory_dt,
        interpolation_steps=selected.interpolation_steps, interpolation_dt=selected.interpolation_dt)
    expected = _bind_event_clock(problem, expected, grid["duration"], .002)
    _, actual = _continuation_timing(problem, budget, {
        "rebind": {}, "scene_anchors": [.5], "control_timestep_s": .002,
        "request_time_origin_s": 0., "request_time_origin_step": 0,
        "history": (), "goal_identities": None, "timing_events": (),
    })
    np.testing.assert_array_equal(actual.u, expected.u)
    np.testing.assert_array_equal(actual.s, expected.s)
    for identity in expected.event_clock:
        assert actual.event_clock[identity] == expected.event_clock[identity]
# END reference scheduling and robot polynomial boundary conditions stay separate.
