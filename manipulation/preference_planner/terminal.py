"""Eliminate motion after the supplied reference's terminal time interval.

The planner keeps FUNCTO's original uniform cubic, measured entrance,
physical constraints and export.  SciPy's null-space and least-squares APIs
only eliminate the affine coefficient equations for a constant suffix.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space
import torch


@dataclass(frozen=True)
class TerminalHold:
    """An affine subspace of one original cubic's position coefficients.

    ``full = offset + matrix @ reduced`` uses [control, named joint] arrays.
    The matrix has orthonormal columns, so ``reduce`` is the Euclidean
    least-squares projection of an initial coefficient estimate.  The
    represented trajectory, rather than an execution-time correction, is
    constant throughout the declared terminal interval.
    """

    matrix: np.ndarray
    offset: np.ndarray
    report: dict

    @classmethod
    def from_clock(cls, curve_map, clock_u, clock_z, *, source):
        """Bind the z=1 suffix of a source-time schedule.

        ``clock_u`` is normalized physical time, and ``clock_z`` is the
        original reference's normalized *time parameter*.  It is not the
        skill progress coordinate, which may remain constant while the
        supplied source trajectory continues moving.  The caller supplies
        the clock already used to bind the reference and actuator events.

        On cubic interval i, the complete B-spline coefficients c_i..c_(i+3)
        must all agree for the polynomial to be constant.  Applying this to
        every interval touched by the terminal suffix gives one affine
        linear system.  The original curve map already eliminates q0/dq0.

        If the requested hold starts inside an original polynomial segment,
        constancy on that positive-length subinterval implies constancy of
        the whole polynomial.  The report records this exact implication;
        the clock is not rounded or changed.  Adding a new curve breakpoint
        would instead define a different trajectory representation.
        """
        u, z = np.asarray(clock_u, float), np.asarray(clock_z, float)
        if (u.ndim != 1 or z.shape != u.shape or len(u) < 2
                or u[0] != 0. or u[-1] != 1. or z[-1] != 1.
                or np.any(np.diff(u) <= 0.) or np.any(np.diff(z) < 0.)):
            raise ValueError("Terminal motion needs the bound monotone source-time clock")
        first = int(np.flatnonzero(z == 1.)[0])
        requested_u = float(u[first])
        breaks = np.arange(curve_map.intervals + 1, dtype=float) / curve_map.intervals
        interval = int(np.searchsorted(breaks, requested_u, side="right") - 1)

        # Complete coefficient differences retain the exact entrance offset:
        # c = q0 + M @ (actions-q0) + h*v*dq0.  Equal coefficients in the
        # suffix therefore give A@(actions-q0) = -h*diff(v)*dq0.
        # A reference ending exactly at the physical endpoint has no hold
        # interval.  Its empty equation system yields the identity subspace.
        suffix = slice(interval, None) if requested_u < 1. else slice(0, 0)
        equations = np.diff(curve_map.matrix[suffix], axis=0)
        right = (-np.diff(curve_map.velocity_coefficients[suffix])[:, None]
                 * curve_map.knot_dt * curve_map.start_dq)
        if interval == 0 and np.any(curve_map.start_dq != 0.):
            raise ValueError("A constant first cubic interval conflicts with measured nonzero entrance velocity")
        matrix = null_space(equations)
        shift = np.linalg.lstsq(equations, right, rcond=None)[0]
        offset = curve_map.start_q + shift
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
            "full_control_rows": int(curve_map.knots),
            "reduced_control_rows": int(matrix.shape[1]),
            "eliminated_control_rows": int(curve_map.knots - matrix.shape[1]),
            "zero_free_variables": matrix.shape[1] == 0,
            "continuity": "original cubic C2 at the constant-suffix join",
            "meaning": "all planned arm joints remain constant after reference completion",
        }
        matrix.flags.writeable = offset.flags.writeable = False
        return cls(matrix, offset, report)

    def expand(self, reduced):
        """Return original coefficients, retaining Torch autograd if supplied.

        Leading candidate axes are permitted.  The trailing axes are
        [reduced control, joint], in the original curve's joint order.
        """
        if isinstance(reduced, torch.Tensor):
            matrix = reduced.new_tensor(self.matrix)
            offset = reduced.new_tensor(self.offset)
            return matrix @ reduced + offset
        return np.einsum("nk,...kj->...nj", self.matrix, np.asarray(reduced)) + self.offset

    def reduce(self, full):
        """Project an initial coefficient estimate onto this affine subspace.

        This operation constructs a numerical initial estimate.  Its changed
        trajectory must be recorded and evaluated by the same planner as any
        other estimate; projection carries no task or collision conclusion.
        Empty output columns are valid when the declared interval and measured
        entrance uniquely determine the whole original cubic.
        """
        return np.einsum("nk,...nj->...kj", self.matrix, np.asarray(full) - self.offset)

    def pullback(self, jacobian):
        """Map original C-order coefficient Jacobian columns to reduced ones.

        Input has shape [condition row, full control * joint].  The output
        preserves the original condition values and row order, and applies
        only the derivative of the affine coefficient substitution.
        """
        jacobian = np.asarray(jacobian)
        full_rows, joints = self.offset.shape
        values = jacobian.reshape(len(jacobian), full_rows, joints)
        return np.einsum("rnj,nk->rkj", values, self.matrix).reshape(
            len(jacobian), self.matrix.shape[1] * joints)
