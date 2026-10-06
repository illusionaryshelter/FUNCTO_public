"""Small CPU checks for ordered independent distances against public Coal.

The production zero-surface consumer reads the distance and two nearest points.
Those fields are compared byte for byte, including signed zero; unconsumed
triangle identifiers and possibly undefined normals on degenerate meshes are
not invented as new acceptance conditions by this performance change.
"""

# BEGIN independent batch execution keeps native evidence and failure order.
import ctypes
import gc
from pathlib import Path
import struct
import sys
import warnings

import coal
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation._coal_scene_batch import ordered_distances


def transform(translation=(0., 0., 0.)):
    return coal.Transform3s(np.eye(3), np.array(translation, dtype=np.float64))


def raw(first, first_transform, second, second_transform, request):
    result = coal.DistanceResult()
    coal.distance(first, first_transform, second, second_transform, request, result)
    return result


def evidence(result):
    points = [np.asarray(result.getNearestPoint1()), np.asarray(result.getNearestPoint2())]
    return (struct.pack("=d", float(result.min_distance)),
            *((point.dtype.str, point.shape, point.tobytes()) for point in points))


def surface():
    """A fully built OBBRSS mesh exercises the production shared-shape path."""
    vertices = np.array([[-.05, -.05, 0.], [.05, -.05, 0.],
                         [.05, .05, 0.], [-.05, .05, 0.]], dtype=np.float64)
    mesh = coal.BVHModelOBBRSS()
    mesh.beginModel(2, 4)
    mesh.addTriangle(*vertices[[0, 1, 2]])
    mesh.addTriangle(*vertices[[0, 2, 3]])
    mesh.endModel()
    return mesh


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_shared_bvh_and_native_shapes_match_original_bits_and_remain_immutable(workers):
    box, sphere, mesh = coal.Box(.04, .06, .08), coal.Sphere(.015), surface()
    first = [box, box, sphere, box, box, sphere, box, box]
    second = [mesh, mesh, mesh, mesh, box, box, mesh, mesh]
    first_poses = [transform(p) for p in ((-.2, 0., .1), (0., 0., .04),
        (.02, 0., .004), (0., 0., .01), (.04, 0., 0.), (.03, 0., 0.),
        (.2, -0., +0.), (.2, -0., +0.))]
    second_poses = [transform()] * len(first)
    request = coal.DistanceRequest()
    request_signature = request.saveToString()
    shape_signatures = [shape.saveToString() for shape in (box, sphere, mesh)]
    expected = [evidence(raw(*args, request))
                for args in zip(first, first_poses, second, second_poses)]
    batch = ordered_distances(first, first_poses, second, second_poses, request, workers)
    assert len(batch) == len(first) and batch.worker_count == workers
    assert batch.storage_nbytes > 0
    # Consumption order is the caller's choice; completion order is unobservable.
    for index in reversed(range(len(first))):
        assert evidence(batch.result(index)) == expected[index]
    old_result = batch.result(0)
    old_result.clear()
    assert evidence(batch.result(0)) == expected[0]
    assert request.saveToString() == request_signature
    assert [shape.saveToString() for shape in (box, sphere, mesh)] == shape_signatures
    # Holding the batch, rather than the original mutable Python lists, owns
    # every geometry used by native result pointers during later consumption.
    first.clear()
    second.clear()
    del box, sphere, mesh
    gc.collect()
    assert evidence(batch.result(0)) == expected[0]


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_non_degenerate_normal_bits_match_fresh_coal_calls(workers):
    first, second = coal.Box(.03, .04, .05), coal.Sphere(.01)
    poses = [transform((.2, -.1, .03)), transform((-.2, .03, -.04))]
    request = coal.DistanceRequest()
    expected = [raw(first, transform(), second, pose, request) for pose in poses]
    batch = ordered_distances([first]*2, [transform()]*2, [second]*2, poses, request, workers)
    for index, original in enumerate(expected):
        actual = batch.result(index)
        assert evidence(actual) == evidence(original)
        assert np.asarray(actual.normal).tobytes() == np.asarray(original.normal).tobytes()


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_python_conversion_and_native_failures_wait_for_the_requested_index(workers):
    box, empty_mesh = coal.Box(.1, .1, .1), coal.BVHModelOBBRSS()
    request = coal.DistanceRequest()
    identity, moved = transform(), transform((.3, 0., 0.))
    with pytest.raises(Exception) as expected_native:
        raw(box, identity, empty_mesh, moved, request)
    batch = ordered_distances([box, object(), box, box], [identity]*4,
                              [box, box, empty_mesh, box], [moved]*4, request, workers)
    expected = evidence(raw(box, identity, box, moved, request))
    assert evidence(batch.result(0)) == expected
    assert evidence(batch.result(3)) == expected
    for _ in range(2):
        with pytest.raises(TypeError):
            batch.result(1)
        with pytest.raises(type(expected_native.value)) as found:
            batch.result(2)
        assert str(found.value) == str(expected_native.value)


def test_python_sequence_error_preserves_original_exception_until_consumption():
    class BrokenSequence:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            if index == 1:
                raise LookupError("later descriptor deliberately failed")
            return transform((.2, 0., 0.))

    sphere, request = coal.Sphere(.01), coal.DistanceRequest()
    batch = ordered_distances([sphere]*3, [transform()]*3, [sphere]*3,
                              BrokenSequence(), request, 2)
    assert evidence(batch.result(0)) == evidence(batch.result(2))
    for _ in range(2):
        with pytest.raises(LookupError, match="later descriptor deliberately failed"):
            batch.result(1)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_each_worker_inherits_and_restores_caller_rounding_mode(workers):
    library = ctypes.CDLL(None)
    previous = library.fegetround()
    actor, target, request = coal.Box(.03, .04, .05), surface(), coal.DistanceRequest()
    actor_poses = [transform((.2, -.03, .07)), transform((-.1, .02, .08)),
                   transform((.04, .03, .11)), transform((.15, -.02, .13))]
    target_poses = [transform()]*4
    try:
        # Linux/x86 FE_TONEAREST, FE_DOWNWARD, FE_UPWARD and FE_TOWARDZERO.
        # All transforms are already constructed: only the native query's
        # floating-point environment changes, exactly as in the serial oracle.
        for mode in (0, 0x400, 0x800, 0xC00):
            assert library.fesetround(mode) == 0
            expected = [evidence(raw(actor, pose, target, target_poses[index], request))
                        for index, pose in enumerate(actor_poses)]
            batch = ordered_distances([actor]*4, actor_poses, [target]*4,
                                      target_poses, request, workers)
            assert library.fegetround() == mode
            for index in range(4):
                assert evidence(batch.result(index)) == expected[index]
            assert library.fegetround() == mode
    finally:
        assert library.fesetround(previous) == 0


def test_empty_large_batch_and_request_history_contract():
    sphere, request = coal.Sphere(.01), coal.DistanceRequest()
    empty = ordered_distances([], [], [], [], request, 4)
    assert len(empty) == 0 and empty.worker_count == 0
    with pytest.raises(IndexError):
        empty.result(0)
    # The old prefix-scan's 4096 leaf cap is not a new scene-inventory limit.
    count = 4097
    first, second = transform(), transform((.3, -0., +0.))
    batch = ordered_distances([sphere]*count, [first]*count, [sphere]*count,
                              [second]*count, request, 4)
    assert len(batch) == count
    assert evidence(batch.result(count-1)) == evidence(raw(sphere, first, sphere, second, request))
    with pytest.raises(ValueError, match="equal lengths"):
        ordered_distances([sphere], [], [sphere], [second], request)
    for workers in (0, 5):
        with pytest.raises(ValueError, match="between 1 and 4"):
            ordered_distances([], [], [], [], request, workers)
    request.gjk_initial_guess = coal.GJKInitialGuess.CachedGuess
    with pytest.raises(ValueError, match="without cached GJK history"):
        ordered_distances([sphere], [first], [sphere], [second], request)
    request.gjk_initial_guess = coal.GJKInitialGuess.DefaultGuess
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        request.enable_cached_gjk_guess = True
    with pytest.raises(ValueError, match="without cached GJK history"):
        ordered_distances([sphere], [first], [sphere], [second], request)
# END no approximate comparison or new finite-witness rejection masks changes.
