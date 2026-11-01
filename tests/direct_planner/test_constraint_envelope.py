"""Feasibility and active derivatives of explicit inequality groups."""

# BEGIN independent dense conditions exercise mixed columns and winner changes.
import numpy as np

from manipulation.direct_planner.constraint_envelope import ConstraintEnvelope
from manipulation.direct_planner.sparse_rows import SparseRows


def test_grouped_values_keep_all_conditions_and_active_derivatives():
    matrix = np.array([[1., 0., 0.], [0., -2., 0.], [.4, .8, 0.], [0., 0., -1.]])
    offset = np.array([-.1, -.3, -.4, -.2])
    source = SparseRows([np.flatnonzero(row) for row in matrix], 3)
    grouping = ConstraintEnvelope(source, [('a',), ('b',), ('a',), ('b',)])
    for x in np.random.default_rng(17).uniform(-.5, .5, (30, 3)):
        original = matrix@x+offset
        values, winners = grouping.select(original)
        np.testing.assert_array_equal(values, [max(original[0], original[2]), max(original[1], original[3])])
        assert np.all(values <= 0.) == np.all(original <= 0.)
        slots = grouping.jacobian_values(matrix[source.rows, source.columns], winners)
        actual = np.zeros(grouping.pattern.shape)
        actual[grouping.pattern.rows, grouping.pattern.columns] = slots
        expected = np.empty_like(actual)
        for column in range(3):
            delta = np.eye(3)[column]*1e-6
            expected[:, column] = (grouping.select(matrix@(x+delta)+offset)[0]
                - grouping.select(matrix@(x-delta)+offset)[0])/2e-6
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=2e-8)


def test_equal_maxima_retain_first_source_identity_and_empty_group_layout():
    source = SparseRows(([0], [1], [0, 1]), 2)
    grouping = ConstraintEnvelope(source, [('pair',)]*3)
    values, winners = grouping.select(np.array([.3, .3, -.1]))
    np.testing.assert_array_equal(values, [.3])
    np.testing.assert_array_equal(winners, [0])
    np.testing.assert_array_equal(grouping.jacobian_values(np.array([2., 3., 4., 5.]), winners), [2., 0.])
    empty = ConstraintEnvelope(SparseRows((), 2), ())
    assert empty.pattern.shape == (0, 2)
    assert all(value.size == 0 for value in empty.select(np.empty(0)))
# END maxima preserve original row evidence and the exact conjunction.
