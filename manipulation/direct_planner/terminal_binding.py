"""Bind the original cubic's constant suffix with sparse shared coefficients.

Inputs are the maintained CubicAffineMap, its already-bound physical/source
clock, and clock provenance. The result is the existing TerminalHold value
object: its expansion, orthogonal projection and derivative pullback remain
unchanged. This module introduces neither a trajectory family nor a new clock.
"""

# BEGIN sparse affine binding of the original complete cubic coefficients.
import numpy as np

from manipulation.preference_planner.terminal import TerminalHold


def bind_terminal_hold(curve_map, clock_u, clock_z, *, source):
    """Return the original TerminalHold contract with orthonormal sparse columns.

    On original interval k, constancy requires complete coefficients c[k:]
    to agree. CubicAffineMap represents c=q0+M@(a-q0)+h*v*dq0. Therefore the
    first two possible constant intervals need the measured-velocity affine
    term; later intervals only identify a suffix of free position coefficients.

    The clock is normalized physical time u versus original reference time z,
    not task progress. A stop inside a polynomial makes that whole polynomial
    constant, exactly as the maintained TerminalHold.from_clock contract does.
    """
    u, z = np.asarray(clock_u, float), np.asarray(clock_z, float)
    if (u.ndim != 1 or z.shape != u.shape or len(u) < 2
            or not np.isfinite(u).all() or not np.isfinite(z).all()
            or u[0] != 0. or u[-1] != 1. or z[-1] != 1.
            or np.any(np.diff(u) <= 0.) or np.any(np.diff(z) < 0.)):
        raise ValueError("Terminal motion needs the bound monotone source-time clock")
    first = int(np.flatnonzero(z == 1.)[0])
    requested_u = float(u[first])
    breaks = np.arange(curve_map.intervals + 1, dtype=float) / curve_map.intervals
    interval = int(np.searchsorted(breaks, requested_u, side="right") - 1)
    rows = curve_map.knots
    offset = np.broadcast_to(curve_map.start_q, (rows, len(curve_map.start_q))).copy()

    if requested_u == 1.:
        # A single endpoint has no positive-length constant interval.
        matrix = np.eye(rows)
    elif interval == 0:
        if np.any(curve_map.start_dq != 0.):
            raise ValueError("A constant first cubic interval conflicts with measured nonzero entrance velocity")
        matrix = np.empty((rows, 0))
    elif interval == 1:
        # c1=c2 forces every free coefficient to q0+h*dq0/3. This is an
        # affine singleton, even though the initial nonzero velocity remains.
        offset += curve_map.knot_dt * curve_map.start_dq / 3.
        matrix = np.empty((rows, 0))
    else:
        independent = interval - 2
        matrix = np.zeros((rows, independent + 1))
        matrix[:independent, :independent] = np.eye(independent)
        # Disjoint columns are orthogonal. Unit length keeps reduce=E.T an
        # actual Euclidean projection; an unnormalised shared column would
        # sum its member coefficients instead of projecting their mean.
        matrix[independent:, independent] = 1. / np.sqrt(rows - independent)

    duration = curve_map.intervals * curve_map.knot_dt
    report = {
        "source": str(source),
        "clock_coordinate": "normalized original reference time z; independent of skill progress",
        "requested_hold_start_u": requested_u,
        "requested_hold_start_s": requested_u * duration,
        "implied_constant_start_u": float(breaks[interval]),
        "implied_constant_start_s": interval * curve_map.knot_dt,
        "hold_end_s": duration,
        "positive_length_hold": requested_u < 1.,
        "first_constant_polynomial_interval": interval if requested_u < 1. else None,
        "full_control_rows": int(rows),
        "reduced_control_rows": int(matrix.shape[1]),
        "eliminated_control_rows": int(rows - matrix.shape[1]),
        "zero_free_variables": matrix.shape[1] == 0,
        "continuity": "original cubic C2 at the constant-suffix join",
        "meaning": "all planned arm joints remain constant after reference completion",
        "parameterization": "orthonormal sparse complete-coefficient equality binding",
    }
    matrix.flags.writeable = offset.flags.writeable = False
    return TerminalHold(matrix, offset, report)
# END original trajectory, event clock and TerminalHold operations are retained.
