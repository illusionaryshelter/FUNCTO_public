"""Fixed row-local Jacobian storage for the planner's first-order solver.

Rows name original scalar conditions. Each row declares its possible reduced
control columns before optimization; values at zero derivative do not alter
that declaration. Producers fill this row-major numeric vector directly.
CasADi's column-major permutation is compiled once by the solver adapter.
"""

import numpy as np


class SparseRows:
    """Store one immutable CSR-style pattern without a dense Jacobian.

    ``columns_by_row`` is the structural dependency union in C-order flattened
    ``[reduced_control,named_joint]`` coordinates. Within each row columns are
    ascending and unique. ``slices`` address the producer's numeric output.
    """

    def __init__(self, columns_by_row, variables):
        source_columns = tuple(np.asarray(columns, dtype=np.int64) for columns in columns_by_row)
        self.variables = int(variables)
        self.indptr = np.r_[0, np.cumsum([len(columns) for columns in source_columns], dtype=np.int64)]
        self.columns = (np.concatenate(source_columns) if source_columns
                        else np.empty(0, dtype=np.int64))
        self.rows = np.repeat(np.arange(len(source_columns), dtype=np.int64),
                              np.diff(self.indptr))
        self.slices = tuple(slice(int(a), int(b)) for a, b in zip(self.indptr[:-1], self.indptr[1:]))
        # Row access is a view of the single owned column buffer, including an
        # empty task inventory. Repeated event scopes do not retain extra copies.
        self.columns_by_row = tuple(self.columns[target] for target in self.slices)
        self.nnz = len(self.columns)
        self.shape = len(self.columns_by_row), self.variables
        # The solver's CCS order is a permutation of these already unique slots.
        self.ccs_order = np.lexsort((self.rows, self.columns))
        for value in (self.indptr, self.columns, self.rows, self.ccs_order, *self.columns_by_row):
            value.flags.writeable = False

    @property
    def storage_bytes(self):
        """Resident index arrays; numeric values are owned by a current point."""
        return int(self.indptr.nbytes + self.columns.nbytes + self.rows.nbytes + self.ccs_order.nbytes)


def coefficient_columns(coefficient_rows, joint_columns, joints):
    """Return the structural product of fixed cubic rows and named joints.

    Coefficient rows already include terminal substitution. Exact nonzero
    constants define their dependence; trial derivative values are never used.
    """
    rows = np.asarray(coefficient_rows)
    active = np.flatnonzero(np.any(rows != 0., axis=0))
    joint_columns = np.asarray(joint_columns, dtype=np.int64)
    return (active[:, None]*int(joints)+joint_columns[None]).ravel()
