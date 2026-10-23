"""Boundary-only zero cells retain surfaces without starving volume queries."""

# BEGIN exact trilinear sign structure replaces endless zero-width refinement.
import coal
import numpy as np
import pytest

from manipulation.implicit_distance import ObservedField, query_distance
from manipulation.native_field_distance import query_native_field
from manipulation.observed_collision import query_region_intersection


def plane():
    values = np.broadcast_to(np.array([-.05, 0., .05])[:, None, None], (3, 2, 2)).copy()
    return ObservedField(values, np.ones_like(values, bool),
                         origin=[-.05, .975, -.025], voxel_size=.05)


@pytest.mark.parametrize("native_first", [False, True])
def test_zero_face_does_not_starve_a_volume_intersection(monkeypatch, native_first):
    field = plane()
    original = field.split
    count = 0

    def counted(node):
        nonlocal count
        count += 1
        assert count <= 16, "The zero-volume branch must not refine indefinitely"
        return original(node)

    monkeypatch.setattr(field, "split", counted)
    if native_first:
        pose = np.eye(4)
        pose[:3, 3] = [-.005, 1., 0.]
        result = query_native_field(coal.Box(.02, .02, .02), pose, field, np.eye(4))
    else:
        moving = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                               origin=[-.015, .99, -.01], voxel_size=.02)
        result = query_distance(moving, np.eye(4), field, np.eye(4))
    assert result["status"] == "intersection"


def test_boundary_only_cell_remains_unresolved_and_available_to_contact_queries():
    values = np.broadcast_to(np.array([0., .05])[:, None, None], (2, 2, 2)).copy()
    field = ObservedField(values, np.ones_like(values, bool), origin=np.zeros(3), voxel_size=.05)
    result = query_native_field(coal.Box(.02, .02, .02), np.eye(4), field, np.eye(4))
    assert result["status"] == "unresolved" and result["intersection"] is None
    contact = query_region_intersection(field, np.eye(4),
        np.vstack((np.eye(3), -np.eye(3))), np.ones(6))
    assert contact["status"] == "contact"
    from manipulation.sampled_geometry_objective import MaterialSamples
    assert len(MaterialSamples(field).zero_points) > 0


def test_constant_zero_cell_keeps_its_volume():
    field = ObservedField(np.zeros((2, 2, 2)), np.ones((2, 2, 2), bool),
                          origin=[-.005] * 3, voxel_size=.01)
    assert not field._boundary_only_cells.any()
    assert query_native_field(coal.Box(.02, .02, .02), np.eye(4), field, np.eye(4))["status"] == "intersection"
# END neither a new tolerance nor a changed TSDF field is used.
