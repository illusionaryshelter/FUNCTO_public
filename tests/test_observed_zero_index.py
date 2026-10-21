"""Surface directions retain the original supported implicit occupancy."""

# BEGIN direction lookup searches zero-bearing cells independently of depth.
import numpy as np
import pytest

from manipulation.implicit_distance import ObservedField, query_distance


@pytest.mark.parametrize("x", [.15, 1.2])
def test_supported_surface_remains_available_from_deep_and_shallow_points(x):
    values = np.broadcast_to((np.arange(15) * .1 - 1.25)[:, None, None], (15, 3, 3)).copy()
    field = ObservedField(values, np.ones_like(values, bool), origin=np.zeros(3), voxel_size=.1)
    cells = field.cells.copy()
    point = np.array([x, .1, .1])
    result = field.local_zero(point)
    assert result is not None
    distance, zero, normal = result
    np.testing.assert_allclose(zero, [1.25, .1, .1], atol=1e-8)
    np.testing.assert_allclose(normal, [1., 0., 0.], atol=1e-12)
    assert abs(distance - (1.25 - x)) < 1e-8
    np.testing.assert_array_equal(field.cells, cells)
    np.testing.assert_array_equal(field.values, values.astype(np.float32))
    assert len(field.cells) == 52 and len(field._zero_cell_ids) == 4
    assert query_distance(field, np.eye(4), field, np.eye(4))["status"] == "intersection"


@pytest.mark.parametrize("value,occupied", [(-1., True), (0., True), (1., False)])
def test_constant_fields_have_no_invented_surface_normal(value, occupied):
    field = ObservedField(np.full((2, 2, 2), value), np.ones((2, 2, 2), bool),
                          origin=np.zeros(3), voxel_size=1.)
    assert (field.root is not None) == occupied
    assert field.local_zero(np.full(3, .5)) is None


def test_zero_on_a_supported_face_is_retained_but_unknown_samples_are_not():
    values = np.broadcast_to(np.array([-.1, 0.])[:, None, None], (2, 2, 2)).copy()
    observed = np.ones_like(values, bool)
    field = ObservedField(values, observed, origin=np.zeros(3), voxel_size=1.)
    assert field.local_zero(np.array([.2, .5, .5])) is not None
    observed[1, 0, 0] = False
    incomplete = ObservedField(values, observed, origin=np.zeros(3), voxel_size=1.)
    assert incomplete.local_zero(np.array([.2, .5, .5])) is None
    assert incomplete.root is None
# END occupied volume and missing observations keep their original meanings.
