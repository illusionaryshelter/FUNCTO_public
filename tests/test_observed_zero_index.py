"""Finite zero samples retain the original supported implicit occupancy."""

import numpy as np
import pytest

from manipulation.implicit_distance import ObservedField, query_distance
from manipulation.sampled_geometry_objective import MaterialSamples


def test_original_zero_edges_remain_available_without_changing_occupied_cells():
    values = np.broadcast_to((np.arange(15) * .1 - 1.25)[:, None, None], (15, 3, 3)).copy()
    field = ObservedField(values, np.ones_like(values, bool), origin=np.zeros(3), voxel_size=.1)
    cells = field.cells.copy()
    points = MaterialSamples(field).zero_points
    assert len(points) > 0
    np.testing.assert_allclose(points[:, 0], 1.25, atol=1e-8)
    sampled, supported, gradient, _ = field.sample(points)
    assert supported.all()
    np.testing.assert_allclose(sampled, 0., atol=1e-8)
    np.testing.assert_allclose(gradient, np.broadcast_to([1., 0., 0.], gradient.shape), atol=1e-7)
    np.testing.assert_array_equal(field.cells, cells)
    np.testing.assert_array_equal(field.values, values.astype(np.float32))
    assert query_distance(field, np.eye(4), field, np.eye(4))["status"] == "intersection"


@pytest.mark.parametrize("value,occupied", [(-1., True), (0., True), (1., False)])
def test_constant_fields_keep_original_occupancy_and_zero_spatial_gradient(value, occupied):
    field = ObservedField(np.full((2, 2, 2), value), np.ones((2, 2, 2), bool),
                          origin=np.zeros(3), voxel_size=1.)
    assert (field.root is not None) == occupied
    _, supported, gradient, _ = field.sample(np.full((1, 3), .5))
    assert supported.all() and np.count_nonzero(gradient) == 0
    assert bool(len(MaterialSamples(field).zero_points)) == (value == 0.)


def test_zero_on_a_supported_face_is_retained_and_unknown_samples_remain_absent():
    values = np.broadcast_to(np.array([-.1, 0.])[:, None, None], (2, 2, 2)).copy()
    observed = np.ones_like(values, bool)
    field = ObservedField(values, observed, origin=np.zeros(3), voxel_size=1.)
    assert len(MaterialSamples(field).zero_points) == 4
    observed[1, 0, 0] = False
    incomplete = ObservedField(values, observed, origin=np.zeros(3), voxel_size=1.)
    assert len(MaterialSamples(incomplete).zero_points) == 0 and incomplete.root is None
