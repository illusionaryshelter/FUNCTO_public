"""Sign provenance stays local while native surface distances may propagate."""

# BEGIN CPU regressions for independent source and distance-query lattices.
import numpy as np

from manipulation.native_voxel_sources import _mask_source_sign_bounds


def test_negative_index_alias_and_positive_exterior_keep_only_nearest_site():
    # Source cells occupy [-1, 1). Expanded query centres are -2.5 ... 2.5.
    # In particular, the native int(-0.5) lookup at x=-1.5 aliases source
    # voxel zero; a positive source sign there is not local observation.
    bits = np.full((6, 1, 1), 7, dtype=np.uint8)
    actual = _mask_source_sign_bounds(bits,
        source_center=[0., 0., 0.], source_shape_xyz=[2, 2, 2], source_voxel_size=1.,
        query_center=[0., 0., 0.], query_voxel_size=1.)
    assert actual is bits
    np.testing.assert_array_equal(actual[:, 0, 0], [1, 1, 7, 7, 1, 1])


def test_source_lower_boundary_is_supported_and_upper_boundary_is_not():
    bits = np.full((5, 1, 1), 7, dtype=np.uint8)
    _mask_source_sign_bounds(bits,
        source_center=[0., 0., 0.], source_shape_xyz=[2, 2, 2], source_voxel_size=1.,
        query_center=[0., 0., 0.], query_voxel_size=.5)
    np.testing.assert_array_equal(bits[:, 0, 0], [7, 7, 7, 7, 1])


def test_query_center_and_xyz_shape_are_independent_of_the_source_grid():
    # The TSDF occupies x=[9,11), y=[-5,-1), z=[18,24). Its query lattice
    # has a different centre, spacing and shape. Only x={10,10.5}, y=-4
    # survives; z={23,23.5} remains inside the unchanged source box.
    bits = np.full((4, 1, 2), 7, dtype=np.uint8)
    _mask_source_sign_bounds(bits,
        source_center=[10., -3., 21.], source_shape_xyz=[2, 4, 6], source_voxel_size=1.,
        query_center=[10.75, -4., 23.25], query_voxel_size=.5)
    expected = np.ones((4, 1, 2), dtype=np.uint8)
    expected[:2] = 7
    np.testing.assert_array_equal(bits, expected)


def test_every_axis_restricts_sign_without_changing_interior_or_site_bits():
    # Include every native flag combination, including no nearest-site
    # source. Masking must never create bit 1 or alter any interior flag.
    bits = np.arange(27, dtype=np.uint8).reshape(3, 3, 3) % 8
    original = bits.copy()
    _mask_source_sign_bounds(bits,
        source_center=[0., 0., 0.], source_shape_xyz=[1, 1, 1], source_voxel_size=1.,
        query_center=[0., 0., 0.], query_voxel_size=1.)
    expected = original & np.uint8(1)
    expected[1, 1, 1] = original[1, 1, 1]
    np.testing.assert_array_equal(bits, expected)
    np.testing.assert_array_equal(bits & 1, original & 1)


def test_wholly_supported_lattice_retains_positive_and_negative_signs():
    bits = np.array([0, 1, 2, 3, 6, 7, 3, 7], dtype=np.uint8).reshape(2, 2, 2)
    original = bits.copy()
    _mask_source_sign_bounds(bits,
        source_center=[1., 2., 3.], source_shape_xyz=[4, 4, 4], source_voxel_size=.25,
        query_center=[1., 2., 3.], query_voxel_size=.25)
    np.testing.assert_array_equal(bits, original)
# END no test invokes Mapper, CUDA, geometry queries, or a new sensor observation.
