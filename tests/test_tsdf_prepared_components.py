"""Small checks for exact owner-pose cache identity and native-map residency.

The world-pose oracle is the original individual NumPy matrix product, and
distance evidence comes from the public Coal overload. No cache implementation
or tree certificate is used to manufacture either expected result.
"""

# BEGIN one tiny existing source fixture covers point, segment and volume poses.
import ctypes
import gc
import weakref

import coal
import numpy as np

from manipulation.tsdf_geometry import ObservedTsdfGeometry
from test_coal_scene_batch import raw_distance, request, result_bits, transform
from test_tsdf_query_optimization import component_record, geometry, world_pose


def register_one(source, prepared, identity, owner_pose):
    """Register only an explicitly requested leaf, retaining the original @."""
    component = source.collision_component(identity)
    expected = owner_pose @ component[3]
    prepared.register(np.array([identity], dtype=np.intp), [component[2]],
                      expected[None])
    return component, expected


def test_cache_uses_dtype_exact_pose_bits_and_only_the_most_recent_pose(geometry):
    owner = np.eye(4)
    first = geometry.prepared_components(owner)
    assert geometry.prepared_components(owner.copy()) is first
    # BEGIN layout is part of the original NumPy multiplication's inputs.
    fortran = geometry.prepared_components(np.asfortranarray(owner))
    assert fortran is not first
    assert geometry.prepared_components(np.asfortranarray(owner)) is fortran
    aligned = geometry.prepared_components(owner)
    unaligned_owner = np.ndarray((4, 4), dtype=np.float64, buffer=bytearray(129), offset=1)
    unaligned_owner[:] = owner
    assert unaligned_owner.strides == owner.strides and not unaligned_owner.flags.aligned
    unaligned = geometry.prepared_components(unaligned_owner)
    assert unaligned is not aligned
    assert geometry.prepared_components(unaligned_owner) is unaligned
    # END identical numeric bytes cannot erase stride or alignment differences.
    # Distinguish dtype even when every represented numeric value is identical.
    single = geometry.prepared_components(owner.astype(np.float32))
    assert single is not first
    assert geometry.prepared_components(owner.astype(np.float32)) is single
    restored = geometry.prepared_components(owner)
    assert restored is not first and restored is not single
    signed_zero = owner.copy()
    signed_zero[0, 1] = -0.
    changed = geometry.prepared_components(signed_zero)
    assert changed is not restored
    one_bit = signed_zero.copy()
    one_bit[0, 3] = np.nextafter(0., 1.)
    assert geometry.prepared_components(one_bit) is not changed
    # Input arrays are not retained as writable cache keys. Mutating an input
    # matrix must invalidate its former prepared target instance.
    latest = geometry.prepared_components(owner)
    owner[1, 3] = .125
    assert geometry.prepared_components(owner) is not latest


def test_strided_owner_keeps_its_original_distinct_numpy_product(geometry):
    # BEGIN reproduce the installed NumPy path difference with one rigid pose.
    # The local operand is the fixture's real oblique segment frame. The two
    # products below differ at [1,1] and [2,0] in the recovery environment,
    # despite identical dtype, shape and C-order bytes in the owner matrices.
    owner = np.array([
        [.8969947235604496, -.20417526955723792, -.3920623996329433, .111],
        [.08488185363615508, .9499885294792384, -.30052764395503556, -.234],
        [.43381509520532235, .23629272768983714, .8694654737325075, .345],
        [0., 0., 0., 1.]], dtype=np.float64)
    strided_owner = np.empty((8, 8), dtype=np.float64)[::2, ::2]
    strided_owner[:] = owner
    assert strided_owner.tobytes(order="C") == owner.tobytes(order="C")
    assert strided_owner.strides != owner.strides
    first = geometry.prepared_components(owner)
    _, ordinary_product = register_one(geometry, first, 1, owner)
    second = geometry.prepared_components(strided_owner)
    assert second is not first
    _, strided_product = register_one(geometry, second, 1, strided_owner)
    assert ordinary_product.tobytes() != strided_product.tobytes()
    assert first.pose(1).tobytes() == ordinary_product.tobytes()
    assert second.pose(1).tobytes() == strided_product.tobytes()
    assert geometry.prepared_components(strided_owner) is second
    # END cache hits retain the particular layout's original multiplication.


def test_pose_switch_keeps_borrowed_nonidentity_component_bits_and_evidence(geometry):
    owner = world_pose(.31, [.2, -.3, .4])
    first = geometry.prepared_components(owner)
    actor, actor_pose, options = coal.Box(.03, .04, .05), np.eye(4), request()
    # The existing oblique segment has both local translation and rotation.
    component, expected = register_one(geometry, first, 1, owner)
    assert not np.array_equal(component[3], np.eye(4))
    assert first.pose(1).tobytes() == expected.tobytes()
    original = raw_distance(actor, actor_pose, component[2], expected, options)
    next_owner = owner.copy()
    next_owner[0, 3] = -.15
    second = geometry.prepared_components(next_owner)
    _, next_expected = register_one(geometry, second, 1, next_owner)
    assert second.pose(1).tobytes() == next_expected.tobytes()
    assert first.pose(1).tobytes() == expected.tobytes()
    ids = np.array([1], dtype=np.intp)
    old_batch = first.scan(actor, transform(actor_pose), ids, options, 8)
    assert result_bits(old_batch.prefix(0, 1e-6)[2]) == result_bits(original)
    returned = geometry.prepared_components(owner)
    assert returned is not first and returned is not second
    _, again = register_one(geometry, returned, 1, owner)
    assert again.tobytes() == first.pose(1).tobytes()


def test_rounding_and_numpy_error_policy_changes_invalidate_world_pose_cache(geometry):
    library = ctypes.CDLL(None)
    previous = library.fegetround()
    owner = world_pose(.31, [.111, -.234, .345])
    # Build the immutable local primitive before changing the matrix-product
    # environment; only the original owner @ local chain varies in this test.
    local = geometry.collision_component(1)[3]
    instances, products = [], []
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            assert library.fesetround(mode) == 0
            target = geometry.prepared_components(owner)
            assert all(target is not previous_target for previous_target in instances)
            _, original = register_one(geometry, target, 1, owner)
            assert target.pose(1).tobytes() == (owner @ local).tobytes() == original.tobytes()
            assert geometry.prepared_components(owner.copy()) is target
            assert library.fegetround() == mode
            instances.append(target)
            products.append(original.tobytes())
        assert len(set(products)) > 1
    finally:
        assert library.fesetround(previous) == 0
    ordinary = geometry.prepared_components(owner)
    policy = np.geterr()
    alternative = "raise" if policy["over"] != "raise" else "ignore"
    with np.errstate(over=alternative):
        strict = geometry.prepared_components(owner)
        assert strict is not ordinary
        assert geometry.prepared_components(owner) is strict
    assert geometry.prepared_components(owner) is not strict


def test_replacement_geometry_has_its_own_table_and_retired_source_does_not_dangle():
    first = ObservedTsdfGeometry(component_record([[[.1, .2, .3]]]))
    replacement = ObservedTsdfGeometry(component_record([[[.4, .5, .6]]]))
    owner = np.eye(4)
    old_target = first.prepared_components(owner)
    new_target = replacement.prepared_components(owner)
    old_component, old_pose = register_one(first, old_target, 0, owner)
    new_component, new_pose = register_one(replacement, new_target, 0, owner)
    assert old_component[2] is not new_component[2]
    assert old_pose.tobytes() != new_pose.tobytes()
    actor, options = coal.Sphere(.01), request()
    original = raw_distance(actor, owner, old_component[2], old_pose, options)
    reference = weakref.ref(first)
    del first, old_component
    gc.collect()
    assert reference() is None
    batch = old_target.scan(actor, transform(owner), np.array([0], dtype=np.intp), options, 8)
    assert result_bits(batch.prefix(0, 1e-6)[2]) == result_bits(original)
    assert new_target.pose(0).tobytes() == new_pose.tobytes()


def test_prepared_reserve_is_admitted_before_lazily_constructing_leaf_shapes(geometry):
    admitted = geometry.storage_nbytes
    before = geometry.pending_storage_nbytes
    table_bytes = geometry._component_table.storage_nbytes
    preparation = geometry.component_preparation_storage_nbytes
    assert preparation > table_bytes
    assert before == geometry.collision_cache_storage_nbytes + preparation-table_bytes
    target = geometry.prepared_components(np.eye(4))
    assert geometry.storage_nbytes == admitted
    assert geometry.pending_storage_nbytes == before-target.storage_nbytes
    assert not geometry._collision_cache
    ids = np.arange(sum(geometry._counts), dtype=np.intp)
    np.testing.assert_array_equal(target.missing(ids), ids)
    pending = geometry.pending_storage_nbytes
    register_one(geometry, target, 1, np.eye(4))
    assert geometry.pending_storage_nbytes == pending-4096
    assert geometry.storage_nbytes == admitted
    np.testing.assert_array_equal(target.missing(ids), ids[ids != 1])
# END only a handful of source leaves and native results are allocated.
