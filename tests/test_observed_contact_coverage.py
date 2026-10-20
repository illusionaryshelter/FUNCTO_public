"""Surface-contact and closed observed-space counterexamples."""

# BEGIN decisions concern supported implicit sets, not their bounding proxies.
from types import SimpleNamespace

import numpy as np

from manipulation.implicit_distance import ObservedField
from manipulation.observed_collision import query_region_intersection
from manipulation.observed_coverage import query_coverage


def field(values, origin=(0., 0., 0.), h=.01):
    values = np.asarray(values, float)
    return ObservedField(values, np.ones(values.shape, bool), origin=origin, voxel_size=h)


def region(lower, upper):
    return np.r_[np.eye(3), -np.eye(3)], np.r_[upper, -np.asarray(lower)]


def test_interior_negative_values_are_not_surface_contact():
    observed = field(np.broadcast_to(np.array([-.01, .01]), (2, 2, 2)),
                     origin=(-.01, -.01, -.01), h=.02)
    normals, bounds = region([-.005, -.005, -.003], [.005, .005, -.002])
    assert query_region_intersection(observed, np.eye(4), normals, bounds)["status"] == "missing_contact"


def test_opposite_sign_supported_segment_proves_surface_contact():
    observed = field(np.broadcast_to(np.array([-.01, .01]), (2, 2, 2)),
                     origin=(-.01, -.01, -.01), h=.02)
    normals, bounds = region([-.02]*3, [.02]*3)
    result = query_region_intersection(observed, np.eye(4), normals, bounds)
    assert result["status"] == "contact"
    assert result["evidence"] == "opposite_sign_supported_segment"


def test_zero_sample_surface_contact_and_missing_observation():
    observed = field(np.zeros((2, 2, 2)))
    normals, bounds = region([-.02]*3, [.02]*3)
    assert query_region_intersection(observed, np.eye(4), normals, bounds)["status"] == "contact"
    missing = ObservedField(np.zeros((2, 2, 2)), np.zeros((2, 2, 2), bool),
                            origin=[0]*3, voxel_size=.01)
    assert query_region_intersection(missing, np.eye(4), normals, bounds)["status"] == "missing_contact"


def test_union_of_measured_voxels_and_allowed_region_covers_seam():
    observed = field(-np.ones((2, 2, 2)), origin=(-.005, 0., 0.), h=.01)
    coverage = SimpleNamespace(_known=np.ones((2, 2, 1), bool),
                               origin=np.array([-.005, 0., 0.]), voxel_size=.01)
    regions = [dict(lower_m=[0., -.01, -.01], upper_m=[.02, .02, .02])]
    assert query_coverage(observed, np.eye(4), coverage, regions)["status"] == "covered"


def test_supported_zero_point_on_closed_voxel_face_is_not_outside():
    values = np.ones((2, 2, 2))
    values[0, 0, 0] = 0.
    observed = field(values, origin=(.005, 0., 0.))
    coverage = SimpleNamespace(_known=np.ones((1, 1, 1), bool),
                               origin=np.zeros(3), voxel_size=.01)
    result = query_coverage(observed, np.eye(4), coverage, [])
    # The represented set is one original zero sample on a closed voxel face.
    # Arithmetic uncertainty may remain, but this is no outside-space witness.
    assert result["status"] != "unknown_witness"
# END coverage and contact have independent evidence semantics.
