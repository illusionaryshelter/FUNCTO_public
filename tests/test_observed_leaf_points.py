"""Original-leaf reuse preserves the scalar field's actual point membership."""

# BEGIN precomputation must retain both floating-point values and node scope.
from dataclasses import replace

import numpy as np
import pytest

from manipulation.implicit_distance import ObservedField


@pytest.mark.parametrize("kind", ["mixed", "negative", "zero", "boundary"])
def test_original_leaf_points_match_uncached_scalar_and_are_immutable(kind):
    rng = np.random.default_rng(318)
    values = rng.normal(size=(13, 11, 10))
    if kind == "negative":
        values = -np.ones_like(values)
    elif kind == "zero":
        values = np.zeros_like(values)
    elif kind == "boundary":
        values = np.abs(values)
        values[::2] = 0.
    field = ObservedField(values, np.ones_like(values, bool),
                          origin=[-.37, .11, -.06], voxel_size=.013)
    leaves = [node for node in field.nodes if node.cell >= 0]
    # Include leaves on both sides of the bounded construction batch.
    indices = np.unique(np.r_[np.arange(min(32, len(leaves))),
                              np.arange(max(0, len(leaves) - 32), len(leaves))])
    for index in indices:
        leaf = leaves[index]
        expected = field.occupied_points(replace(leaf, points=None))
        actual = field.occupied_points(leaf)
        np.testing.assert_array_equal(actual, expected)
        assert actual is leaf.points and not actual.flags.writeable
    point_storage = sum(node.points.nbytes + 128 for node in leaves)
    assert field.storage_nbytes >= point_storage + field.values.nbytes


def test_subdivided_node_does_not_inherit_original_leaf_points(monkeypatch):
    values = np.broadcast_to(np.array([-.01, .01])[:, None, None], (2, 2, 2))
    field = ObservedField(values, np.ones_like(values, bool),
                          origin=[-.01] * 3, voxel_size=.02)
    leaf = next(node for node in field.nodes if node.cell >= 0)
    children = field.split(leaf)
    assert children and all(child.points is None for child in children)
    for child in children:
        points = field.occupied_points(child)
        assert ((points >= child.lower) & (points <= child.upper)).all()
    with monkeypatch.context() as context:
        def unexpected(*args, **kwargs):
            raise AssertionError("Original leaves must reuse their completed point array")
        context.setattr(field, "cell_value", unexpected)
        assert field.occupied_points(leaf) is leaf.points
# END dynamic query nodes still derive points from their own spatial bounds.
