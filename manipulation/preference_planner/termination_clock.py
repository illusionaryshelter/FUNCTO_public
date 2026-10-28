"""Bind the arm's terminal interval through the existing request clocks.

Fresh requests and explicit rebinding use the source-time schedule.  An
ordinary continuation retains the terminal time from its prior formal
validation.  The returned arrays are consumed only by ``TerminalHold``;
they do not replace the reference, skill-progress or actuator schedule.
"""

import numpy as np

from manipulation.planning import (
    _initial_timing, _phase_aligned_parameters, reference_progress_clock,
)


def terminal_clock(*, problem, budget, progress_map, reference_time_map,
                   duration, continuation, incoming_terminal_contract):
    """Return normalized time, terminal parameter and their explicit source.

    ``duration`` is this request's already bound physical duration in seconds.
    ``continuation`` is the original planner's validated continuation record,
    including its elapsed prior-curve time and explicit rebind declaration.
    The incoming contract comes from that prior candidate's
    ``validation.export.reference_termination``.  Its requested start belongs
    to the prior curve's local clock; its implied polynomial start is a
    representation consequence and does not replace that requested time.

    For an ordinary continuation, only the terminal interval is needed.
    A three-vertex monotone parameter encodes its unchanged physical start;
    its interior values are not claimed to reconstruct the prior source-time
    warp.  Only the interval where this parameter equals one is consumed.
    The actual reference and events retain the original continuation clock.
    """
    if continuation is None:
        return (reference_time_map.u, reference_time_map.s,
                "fresh bound source-time warp; z=1 marks reference completion")

    if continuation["rebind"] is not None:
        # The library's explicit-rebind path constructs this same source
        # schedule and then discards its independent warp.  Reconstruct that
        # scalar data with the same existing APIs; actual progress_map already
        # contains the bound device ticks, so no actuator is rebound here.
        selected, source = _initial_timing(problem, budget)
        anchors = np.unique(np.r_[problem.anchors, continuation["scene_anchors"]])
        anchors = anchors[anchors >= problem.start_progress]
        before_ticks, warp = reference_progress_clock(source, selected.knots,
            stop_progress=problem.stop_progress, anchors=anchors)
        bound_u = _phase_aligned_parameters(before_ticks, progress_map, warp.u)
        return (bound_u, warp.s,
                "explicit rebind source-time warp aligned to the existing bound device clock")

    start = (float(incoming_terminal_contract["requested_hold_start_s"])
             - float(continuation["elapsed"]))
    # An already active hold begins at this measured entrance.  TerminalHold
    # then expresses its compatibility with measured q0/dq0 in the same curve.
    # A zero-length suffix at the physical endpoint adds no hold equation.
    if start <= 0.:
        u, parameter = np.array([0., 1.]), np.array([1., 1.])
    elif start == duration:
        u, parameter = np.array([0., 1.]), np.array([0., 1.])
    else:
        u, parameter = np.array([0., start / duration, 1.]), np.array([0., 1., 1.])
    return (u, parameter,
            "prior formal requested terminal time minus elapsed curve time; terminal-only encoding")
