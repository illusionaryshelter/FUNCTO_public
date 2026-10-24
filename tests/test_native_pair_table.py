"""Small CPU contracts for the shared cold native robot-pair kernel.

All geometries below are analytic mathematical fixtures for robot primitive
queries. No task object, environment mesh, saved scene, GPU, or simulator state
is constructed. Independent Decimal support functions check the returned
full-entity separation certificate, not Coal's approximate distance estimate.
"""

# BEGIN shared numeric-pair kernel: cold library reference and outward bounds.
import ctypes
from decimal import Decimal, localcontext
import gc
import math
from pathlib import Path
import sys

import coal
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation._coal_scene_batch import NativePairTable


def _pose(translation=(0., 0., 0.), angle=0.):
    """A world-from-primitive pose; x/y rotate about the common world z axis."""
    c, s = math.cos(angle), math.sin(angle)
    result = np.eye(4)
    result[:3, :3] = ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))
    result[:3, 3] = translation
    return result


def _request():
    request = coal.DistanceRequest()
    request.enable_signed_distance = True
    return request


def _raw_row(first, first_pose, second, second_pose, request):
    """Use Coal's public fresh-solver call, independently of the table kernel."""
    result = coal.DistanceResult()
    coal.distance(first, coal.Transform3s(first_pose[:3, :3], first_pose[:3, 3]),
                  second, coal.Transform3s(second_pose[:3, :3], second_pose[:3, 3]),
                  request, result)
    return np.r_[float(result.min_distance), result.getNearestPoint1(),
                 result.getNearestPoint2(), result.normal]


def _raw_reference(shapes, pairs, poses, request):
    return np.asarray([[_raw_row(shapes[a], state[a], shapes[b], state[b], request)
                        for a, b in pairs] for state in poses], dtype=np.float64)


def _fixture():
    """Repeated shapes, both endpoint orders, and separated/penetrating poses."""
    shapes = [coal.Box(.125, .25, .125), coal.Cylinder(.0625, .25),
              coal.Box(.25, .125, .125)]
    pairs = np.asarray([(0, 1), (1, 0), (0, 2), (2, 0)], dtype=np.intp)
    poses = np.asarray([
        [_pose(), _pose((.5, .125, 0.), .2), _pose((-.5, .125, .125), -.3)],
        [_pose((.125, 0., 0.), -.1), _pose((.15, .02, .01), .4),
         _pose((-.1, -.15, .03), .1)],
        [_pose((-.1, .02, .1), .3), _pose((.2, -.2, .1), -.2),
         _pose((-.5, .15, -.1), .4)],
    ], dtype=np.float64)
    return shapes, pairs, poses


def _decimal(value):
    return Decimal.from_float(float(value))


def _projection(shape, pose, normal):
    """Independent 100-digit support of an exact stored box or cylinder.

    Matrix entries, primitive dimensions and the proposed normal are treated
    as their exact binary64 input values in one common reference frame.
    """
    n = tuple(map(_decimal, normal))
    local = [sum(n[row] * _decimal(pose[row, axis]) for row in range(3))
             for axis in range(3)]
    center = sum(n[row] * _decimal(pose[row, 3]) for row in range(3))
    if isinstance(shape, coal.Box):
        radius = sum(abs(value) * _decimal(half)
                     for value, half in zip(local, shape.halfSide))
    elif isinstance(shape, coal.Cylinder):
        radius = (_decimal(shape.radius) * (local[0]**2 + local[1]**2).sqrt()
                  + _decimal(shape.halfLength) * abs(local[2]))
    else:
        raise TypeError("The independent fixture proof covers boxes and cylinders")
    return center-radius, center+radius


def _assert_support_lower_bounds(shapes, pairs, poses, numbers, mask):
    positive = 0
    with localcontext() as context:
        context.prec = 100
        for state_id, state in enumerate(poses):
            for pair_id, (a, b) in enumerate(pairs):
                row = numbers[state_id, pair_id]
                if not mask[state_id, pair_id]:
                    assert row[11] == 0.
                    continue
                normal = row[7:10]
                norm = sum(_decimal(value)**2 for value in normal).sqrt()
                if norm == 0:
                    assert row[11] == 0.
                    continue
                first = _projection(shapes[a], state[a], normal)
                second = _projection(shapes[b], state[b], normal)
                exact_gap = max(Decimal(0), second[0]-first[1], first[0]-second[1])
                assert 0. <= row[11] and math.isfinite(row[11])
                assert _decimal(row[11]) <= exact_gap/norm
                positive += row[11] > 0.
    return positive


def test_raw_rows_accept_strides_and_own_output_after_later_queries():
    shapes, pairs, poses = _fixture()
    # Non-contiguous descriptors must be copied using their actual strides.
    storage = np.empty((6, 6, 8, 8), dtype=np.float64)
    strided_poses = storage[::2, ::2, ::2, ::2]
    strided_poses[:] = poses
    pair_storage = np.empty((8, 4), dtype=np.intp)
    strided_pairs = pair_storage[::2, ::2]
    strided_pairs[:] = pairs
    mask_storage = np.ones((6, 8), dtype=np.bool_)
    mask = mask_storage[::2, ::2]
    table = NativePairTable(shapes, strided_pairs)
    request = _request()
    actual = table.query(strided_poses, mask, request)
    np.testing.assert_array_equal(actual[..., :10], _raw_reference(shapes, pairs, poses, request))
    assert actual.shape == (3, 4, 12) and actual.dtype == np.float64
    assert actual.flags.owndata and actual.flags.c_contiguous
    saved = actual.copy()
    strided_poses[:] = poses[::-1]
    table.query(strided_poses, mask, request)
    np.testing.assert_array_equal(actual, saved)
    assert table.storage_nbytes > 0


@pytest.mark.parametrize("states,have_pair", [(0, True), (3, False), (0, False)])
def test_empty_axes_keep_the_numeric_schema(states, have_pair):
    shapes = [coal.Box(.125, .125, .125), coal.Cylinder(.0625, .25)]
    pairs = np.asarray([(0, 1)] if have_pair else [], dtype=np.intp).reshape(-1, 2)
    table = NativePairTable(shapes, pairs)
    poses = np.broadcast_to(np.eye(4), (states, 2, 4, 4)).copy()
    actual = table.query(poses, np.ones((states, len(pairs)), dtype=np.bool_), _request())
    assert actual.shape == (states, len(pairs), 12)
    assert actual.dtype == np.float64 and actual.flags.owndata


def test_pair_direction_shape_ownership_and_cold_history_are_independent():
    shapes, pairs, poses = _fixture()
    expected = _raw_reference(shapes, pairs, poses, _request())
    input_shapes, input_pairs = list(shapes), pairs.copy()
    table = NativePairTable(input_shapes, input_pairs)
    stored_bytes = table.storage_nbytes
    # The table owns individual shapes and index values, not a mutable list.
    input_shapes[:] = [None] * len(input_shapes)
    input_pairs[:] = 0
    del shapes, input_shapes, input_pairs
    gc.collect()
    mask = np.ones((len(poses), len(pairs)), dtype=np.bool_)
    for order in (np.arange(len(poses)), np.arange(len(poses))[::-1], np.arange(len(poses))):
        actual = table.query(poses[order], mask, _request())
        np.testing.assert_array_equal(actual[..., :10], expected[order])
    assert table.storage_nbytes == stored_bytes


def test_mask_changes_only_requested_certificates_and_bounds_are_conservative():
    shapes, pairs, poses = _fixture()
    table, request = NativePairTable(shapes, pairs), _request()
    all_mask = np.ones((len(poses), len(pairs)), dtype=np.bool_)
    mask = all_mask.copy()
    mask[0, 1::2] = False
    mask[1, :] = False
    full = table.query(poses, all_mask, request)
    masked = table.query(poses, mask, request)
    np.testing.assert_array_equal(masked[..., :11], full[..., :11])
    np.testing.assert_array_equal(masked[..., 11][mask], full[..., 11][mask])
    assert _assert_support_lower_bounds(shapes, pairs, poses, full, all_mask) > 0
    assert _assert_support_lower_bounds(shapes, pairs, poses, masked, mask) > 0


def test_analytic_separation_and_penetration_normal_keep_value_gradient_contract():
    shapes = [coal.Box(.25, .5, .5), coal.Box(.25, .5, .5)]
    pairs = np.asarray([(0, 1)], dtype=np.intp)
    table = NativePairTable(shapes, pairs)
    # These dyadic coordinates have an exact planar contact geometry. h is
    # only a finite-difference increment for this regression, not a model or
    # collision tolerance. The unique minimum translation lies along x.
    h = 2.**-18
    translations = (.5, .125-h, .125, .125+h)
    poses = np.asarray([[_pose(), _pose((x, 0., 0.))] for x in translations])
    request = _request()
    actual = table.query(poses, np.ones((4, 1), dtype=np.bool_), request)[:, 0]
    np.testing.assert_array_equal(actual[:, :10],
                                  _raw_reference(shapes, pairs, poses, request)[:, 0])
    assert 0. < actual[0, 11] <= .25
    # Coal's public signed-distance estimate carries the request's GJK/EPA
    # precision; exact dyadic geometry does not make the solver exact. The
    # independent cold reference above still requires bit-for-bit preservation.
    distance_precision = max(request.gjk_tolerance, request.epa_tolerance)
    np.testing.assert_allclose(actual[1:, 0], np.asarray(translations[1:])-.25,
                               rtol=0., atol=distance_precision)
    assert np.all(actual[1:, 11] == 0.)
    np.testing.assert_allclose(actual[2, 7:10], (1., 0., 0.), rtol=0., atol=1e-14)
    gradient_of_penetration = -(actual[3, 0]-actual[1, 0])/(2*h)
    assert gradient_of_penetration == pytest.approx(-actual[2, 7], rel=0., abs=1e-9)


def test_cold_request_contract_and_nonfinite_pose_error():
    shapes, pairs, poses = _fixture()
    table, request = NativePairTable(shapes, pairs), _request()
    mask = np.ones((len(poses), len(pairs)), dtype=np.bool_)
    request.gjk_initial_guess = coal.GJKInitialGuess.CachedGuess
    with pytest.raises(ValueError, match="cold|cached|history"):
        table.query(poses, mask, request)
    broken = poses.copy()
    broken[1, 0, 0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        table.query(broken, mask, _request())


def test_caller_rounding_is_restored_and_interval_bounds_cover_each_mode():
    # Linux/x86 modes are the same platform contract already exercised by the
    # repository's native bridge regressions. Only the current test thread's
    # rounding mode changes, and finally restores it even on assertion failure.
    library = ctypes.CDLL(None)
    library.fegetround.restype = ctypes.c_int
    library.fesetround.argtypes = (ctypes.c_int,)
    library.fesetround.restype = ctypes.c_int
    previous = library.fegetround()
    shapes, pairs, poses = _fixture()
    table, request = NativePairTable(shapes, pairs), _request()
    mask = np.ones((len(poses), len(pairs)), dtype=np.bool_)
    # Ellipsoid distance is a valid Coal operation, while this bridge's
    # registered projection interface does not implement its enclosure. This
    # fixture reaches an exception inside the interval scope, unlike a malformed
    # NumPy input which is rejected before entering that scope.
    unsupported = NativePairTable([coal.Ellipsoid(.1, .2, .3), coal.Sphere(.1)],
                                  np.asarray([(0, 1)], dtype=np.intp))
    unsupported_poses = np.asarray([[_pose(), _pose((1., 0., 0.))]])
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            assert library.fesetround(mode) == 0
            expected = _raw_reference(shapes, pairs, poses, request)
            actual = table.query(poses, mask, request)
            assert library.fegetround() == mode
            np.testing.assert_array_equal(actual[..., :10], expected)
            assert _assert_support_lower_bounds(shapes, pairs, poses, actual, mask) > 0
            with pytest.raises(RuntimeError, match="state 0, pair 0: .*projection"):
                unsupported.query(unsupported_poses, np.ones((1, 1), dtype=np.bool_), request)
            assert library.fegetround() == mode
        broken = poses.copy()
        broken[0, 0, 0, 3] = np.nan
        with pytest.raises(ValueError):
            table.query(broken, mask, request)
        assert library.fegetround() == mode
    finally:
        assert library.fesetround(previous) == 0
# END one maintained kernel serves numeric scenes and public report construction.
