"""Original continuous cubic bounds with direct local sparse derivatives.

The original CubicAffineMap owns extrema, tie selection and time queries.
This adapter changes only Jacobian storage: one row affects one named joint
and the actual polynomial/terminal coefficient columns of its segment.
"""

from dataclasses import dataclass

import numpy as np

from .sparse_rows import SparseRows


@dataclass
class PhysicalPoint:
    """One exact curve's continuous signed values and local numeric slots."""

    values: np.ndarray
    _jacobian_values: np.ndarray

    def jacobian_values(self):
        return self._jacobian_values


class SparsePhysicalRows:
    """Bind all original physical rows to reduced controls once per solve.

    Row order is derivative, original segment, joint, then bound side;
    acceleration retains its endpoint axis before the side. A fixed sparsity
    union covers every extremum that can become active within each segment.
    """

    def __init__(self, full_domain, terminal):
        self.curve = full_domain._curve_map
        self.bounds = full_domain._constraint_bounds
        self.terminal = terminal
        self.joints = len(self.curve.start_q)
        self.variables = terminal.matrix.shape[1]*self.joints
        polynomial = np.einsum('psk,kr->psr', self.curve.polynomial_weights, terminal.matrix)
        self.blocks, columns, keys = [], [], []
        for derivative in range(4):
            for segment in range(self.curve.intervals):
                reduced = np.flatnonzero(np.any(polynomial[:4-derivative, segment] != 0., axis=0))
                # The original evaluator uses the right polynomial at a knot.
                # Include its exact endpoint rows as well as the segment union.
                endpoint_weights, _ = self.curve.weights_at_knot_times(
                    np.array([segment, segment+1.]), derivative)
                if derivative < 3:
                    endpoint = endpoint_weights @ terminal.matrix
                    reduced = np.union1d(reduced, np.flatnonzero(np.any(endpoint != 0., axis=0)))
                projection = terminal.matrix[:, reduced]
                for joint in range(self.joints):
                    for endpoint in range(2 if derivative == 2 else 1):
                        for side in range(2):
                            columns.append(reduced*self.joints+joint)
                            keys.append(('physical', derivative, segment, joint, endpoint, side))
                            self.blocks.append((derivative, segment, joint, endpoint, side, projection))
        self.row_keys = tuple(keys)
        self.pattern = SparseRows(columns, self.variables)
        self.columns_by_row = self.pattern.columns_by_row

    def evaluate(self, full_controls, *, check_request):
        """Reuse native extrema and fill only each row's declared numeric slots.

        Native basis queries return coefficient rows, not an all-joint
        constraint Jacobian. Each local row is pulled through its prebound
        terminal projection and written directly into its sparse output slice.
        """
        controls = np.asarray(full_controls, dtype=float)
        values = np.empty(len(self.row_keys))
        nonzeros = np.empty(self.pattern.nnz)
        offset = 0
        segments = np.arange(self.curve.intervals)[:, None, None]
        for derivative in range(4):
            check_request()
            if derivative < 2:
                extrema, times = self.curve._segment_extrema(controls, derivative)
            elif derivative == 2:
                times = np.broadcast_to(segments[..., None]+np.array([0., 1.])[None, None, :, None],
                    (self.curve.intervals, self.joints, 2, 2))
            else:
                times = np.broadcast_to(segments+.5, (self.curve.intervals, self.joints, 2))
            indices = np.indices(times.shape)
            joints, sides = indices[1].ravel(), indices[-1].ravel()
            unique_times, time_indices = np.unique(times.ravel(), return_inverse=True)
            weights, velocity = self.curve.weights_at_knot_times(unique_times, derivative)
            for local, (joint, side, time_index) in enumerate(zip(joints, sides, time_indices)):
                row = offset+local
                sign = 2.*side-1.
                if derivative < 2:
                    value = extrema.ravel()[local]
                else:
                    value = (weights[time_index] @ (controls[:, joint]-self.curve.start_q[joint])
                             + velocity[time_index]*self.curve.start_dq[joint])
                values[row] = sign*(value-self.bounds[derivative, side, joint])
                nonzeros[self.pattern.slices[row]] = sign*(weights[time_index] @ self.blocks[row][-1])
            offset += len(joints)
        check_request()
        return PhysicalPoint(values, nonzeros)
