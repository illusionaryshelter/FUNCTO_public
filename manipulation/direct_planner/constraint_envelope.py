"""Exact maxima of declared scalar inequality groups and their active rows.

For every group, all original conditions g_i <= 0 are equivalent to
max_i(g_i) <= 0. Values remain attached to their original rows. The sparse
derivative selects the winning row and retains the union of possible columns.
"""

# BEGIN grouping changes the numerical row layout, not the feasible set.
import numpy as np

from .sparse_rows import SparseRows


class ConstraintEnvelope:
    """Bind explicit group identities over an immutable source row pattern."""

    def __init__(self, source_pattern, group_keys):
        self.source_pattern = source_pattern
        groups = {}
        for row, key in enumerate(group_keys):
            groups.setdefault(tuple(key), []).append(row)
        self.row_keys = tuple(groups)
        self.rows = tuple(np.asarray(rows, dtype=np.int64) for rows in groups.values())
        self.group_by_row = np.empty(len(group_keys), dtype=np.int64)
        columns = []
        for group, rows in enumerate(self.rows):
            self.group_by_row[rows] = group
            columns.append(np.unique(np.concatenate([source_pattern.columns_by_row[row] for row in rows])))
        self.pattern = SparseRows(columns, source_pattern.variables)
        self.output_slot = np.empty(source_pattern.nnz, dtype=np.int64)
        for row, target in enumerate(self.group_by_row):
            original = source_pattern.slices[row]
            self.output_slot[original] = self.pattern.indptr[target]+np.searchsorted(
                self.pattern.columns_by_row[target], source_pattern.columns_by_row[row])

    def select(self, original):
        """Return exact maxima and the first original maximizing row at ties."""
        maxima = np.full(len(self.rows), -np.inf)
        np.maximum.at(maxima, self.group_by_row, original)
        winners = np.full(len(self.rows), len(original), dtype=np.int64)
        winning_rows = np.flatnonzero(original == maxima[self.group_by_row])
        np.minimum.at(winners, self.group_by_row[winning_rows], winning_rows)
        return maxima, winners

    def jacobian_values(self, original, winners):
        """Map the active branch into its fixed union of coefficient columns."""
        rows = self.source_pattern.rows
        active = rows == winners[self.group_by_row[rows]]
        values = np.zeros(self.pattern.nnz)
        values[self.output_slot[active]] = original[active]
        return values
# END complete source values and witness identities remain with their producer.
