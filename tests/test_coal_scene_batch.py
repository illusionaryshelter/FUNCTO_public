"""Small CPU equivalence checks for the exact ordered Coal distance bridge.

The baseline is the public coal.distance overload already used by query_scene,
with original NumPy-composed world poses and a fresh result for every pair.
"""

# BEGIN raw bitwise witnesses and near-contact stop order across the native bridge.
from pathlib import Path
import sys
import ctypes
import gc
import warnings

import coal
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation._coal_scene_batch import MAX_BATCH_SIZE, ComponentTable, PreparedComponents
from manipulation.tsdf_geometry import coal_component


def pose(translation, rotation=(0., 0., 0.)):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = translation
    return result


def transform(value):
    return coal.Transform3s(value[:3, :3], value[:3, 3])


def request():
    result = coal.DistanceRequest()
    result.enable_signed_distance = True
    return result


def raw_distance(first, first_pose, second, second_pose, options):
    result = coal.DistanceResult()
    coal.distance(first, transform(first_pose), second, transform(second_pose), options, result)
    return result


def result_bits(result):
    # Bytes compare signed zero and every represented bit of the original raw
    # distance, both closest points, and normal. No numerical tolerance is used.
    return np.r_[result.min_distance, np.asarray(result.getNearestPoint1()),
                 np.asarray(result.getNearestPoint2()), np.asarray(result.normal)].astype(np.float64).tobytes()


def prepared(shapes, poses):
    """Register original NumPy products without reproducing the native scan."""
    table = ComponentTable(len(shapes))
    result = PreparedComponents(table)
    identities = np.arange(len(shapes), dtype=np.intp)
    result.register(identities, shapes, poses)
    return result, identities


def scan(actor, actor_pose, shapes, poses, options, stop, workers=1):
    target, identities = prepared(shapes, poses)
    batch = target.scan(actor, transform(actor_pose), identities, options, workers)
    # BEGIN exercise all eight lanes when the fixture contains enough jobs.
    assert batch.worker_count == min(workers, len(shapes))
    # END a request for eight workers alone is not evidence that eight ran.
    return batch.prefix(0, stop)


def assert_matches_original(actor, actor_pose, shapes, poses, options, stop, workers=1):
    baseline = []
    for shape, target_pose in zip(shapes, poses):
        value = raw_distance(actor, actor_pose, shape, target_pose, options)
        baseline.append(value)
        if value.min_distance <= stop:
            break
    separated = len(baseline) - int(bool(baseline) and baseline[-1].min_distance <= stop)
    minimum = min(range(separated), key=lambda index: baseline[index].min_distance) if separated else -1
    actual = scan(actor, actor_pose, shapes, poses, options, stop, workers)
    assert actual[0] == separated
    assert actual[1] == minimum
    if minimum < 0:
        assert actual[2] is None
    else:
        assert result_bits(actual[2]) == result_bits(baseline[minimum])
    if separated == len(baseline):
        assert actual[3] == -1 and actual[4] is None
    else:
        assert actual[3] == separated
        assert result_bits(actual[4]) == result_bits(baseline[-1])
    return actual


@pytest.mark.parametrize("workers", [1, 8])
def test_separated_ties_and_owned_result_survive_later_calls(workers):
    actor, target = coal.Box(.04, .06, .08), coal.Box(.08, .02, .06)
    actor_pose = pose([.1, -.2, .3], [.12, -.09, .23])
    owner = pose([.4, .3, -.1], [.07, .11, -.03])
    targets = np.stack([owner @ pose([offset, .02, -.03])
                        for offset in (1.5, 1., 1., 1.8, 2., 1.7, 1.3, 2.2)])
    options = request()
    signature = options.saveToString()
    shapes_before = [shape.saveToString() for shape in (actor, target)]
    original = assert_matches_original(actor, actor_pose, [target]*len(targets), targets, options, 1e-6, workers)
    assert original[1] == 1
    bits = result_bits(original[2])
    # Changing transforms then returning to the initial state must not leak the
    # previous native solver's support guesses or mutate an earlier return value.
    for offset in (.5, -.1, .2):
        moved = targets.copy()
        moved[:, 1, 3] += offset
        assert_matches_original(actor, actor_pose, [target]*len(targets), moved, options, 1e-6, workers)
    assert_matches_original(actor, actor_pose, [target]*len(targets), targets, options, 1e-6, workers)
    assert result_bits(original[2]) == bits
    assert options.saveToString() == signature
    assert [shape.saveToString() for shape in (actor, target)] == shapes_before


@pytest.mark.parametrize("translation,stop", [(.08, 1e-6), (.105, .006)])
@pytest.mark.parametrize("workers", [1, 8])
def test_first_penetration_or_positive_measurement_band_stops_before_invalid_tail(translation, stop, workers):
    shape = coal.Box(.1, .1, .1)
    actor_pose = np.eye(4)
    targets = np.stack([pose([1., 0., 0.]), pose([translation, 0., 0.]), pose([2., 0., 0.])])
    options = request()
    expected = assert_matches_original(shape, actor_pose, [shape]*3, targets, options, stop, workers)
    assert expected[0] == 1 and expected[3] == 1
    # Speculative registration may encounter the bad suffix. Its exception must
    # wait until the caller accepts the near item and requests that suffix.
    targets[2] = np.nan
    target, identities = prepared([shape, shape, object()], targets)
    batch = target.scan(shape, transform(actor_pose), identities, options, workers)
    actual = batch.prefix(0, stop)
    assert actual[:2] == expected[:2] and actual[3] == expected[3]
    assert result_bits(actual[2]) == result_bits(expected[2])
    assert result_bits(actual[4]) == result_bits(expected[4])
    assert batch.physical_query_count == 2
    with pytest.raises(TypeError):
        batch.prefix(2, stop)


def test_near_first_has_no_separated_representative_and_exact_threshold_is_inclusive():
    shape = coal.Sphere(.01)
    poses = np.stack([pose([.04, 0., -0.]), pose([.3, 0., 0.])])
    options = request()
    raw = raw_distance(shape, np.eye(4), shape, poses[0], options)
    result = scan(shape, np.eye(4), [shape, object()], poses, options, raw.min_distance, 8)
    assert result[:3] == (0, -1, None)
    assert result[3] == 0 and result_bits(result[4]) == result_bits(raw)


def test_point_segment_surface_and_solid_use_the_original_coal_primitives():
    pieces = [np.array([[0., 0., 0.]]),
              np.array([[0., 0., 0.], [.03, .02, .01]]),
              np.array([[0., 0., 0.], [.03, 0., .01], [0., .04, .02]]),
              np.array([[0., 0., 0.], [.03, 0., 0.], [0., .04, 0.], [0., 0., .02]])]
    rows = [coal_component(points) for points in pieces]
    owner = pose([.2, -.3, .4], [.2, -.1, .05])
    targets = np.stack([owner @ local for _, local in rows])
    actor_pose = pose([-.1, .2, .1], [.3, .1, -.2])
    for actor in (coal.Box(.03, .02, .04), coal.Cylinder(.02, .04), rows[-1][0]):
        # A batch reports only its winning result. Singleton checks ensure that
        # each lower-dimensional primitive is itself compared byte for byte,
        # rather than being hidden as a nonwinning entry in the mixed batch.
        for index, (shape, _) in enumerate(rows):
            assert_matches_original(actor, actor_pose, [shape], targets[index:index+1], request(), 1e-6)
        assert_matches_original(actor, actor_pose, [shape for shape, _ in rows], targets, request(), 1e-6)


def test_empty_strided_batch_and_maximum_boundary():
    shape = coal.Sphere(.01)
    actor_pose = np.eye(4)
    assert scan(shape, actor_pose, [], np.empty((0, 4, 4)), request(), 1e-6, 8) == (0, -1, None, -1, None)
    values = np.stack([pose([.2+index*.01, 0., 0.]) for index in range(6)])
    assert_matches_original(shape, actor_pose, [shape]*3, values[::2], request(), 1e-6)
    repeated = np.broadcast_to(values[0], (MAX_BATCH_SIZE, 4, 4))
    # BEGIN the native default must dispatch eight lanes without a worker argument.
    targets, ids = prepared([shape]*MAX_BATCH_SIZE, repeated)
    batch = targets.scan(shape, transform(actor_pose), ids, request())
    assert batch.worker_count == 8
    result = batch.prefix(0, 1e-6)
    # END the same maximal tiny-sphere fixture verifies default dispatch and bytes.
    assert result[:2] == (MAX_BATCH_SIZE, 0) and result[3:] == (-1, None)
    expected = raw_distance(shape, actor_pose, shape, repeated[0], request())
    assert result_bits(result[2]) == result_bits(expected)
    with pytest.raises(ValueError, match="at most 4096"):
        scan(shape, actor_pose, [shape]*(MAX_BATCH_SIZE+1),
             np.broadcast_to(values[0], (MAX_BATCH_SIZE+1, 4, 4)), request(), 1e-6)


def test_only_visited_invalid_inputs_raise_and_float32_is_not_silently_cast():
    shape = coal.Sphere(.01)
    poses = np.stack([pose([.2, 0., 0.]), pose([.3, 0., 0.])])
    with pytest.raises(TypeError):
        scan(shape, np.eye(4), [shape, object()], poses, request(), 1e-6)
    invalid = poses.copy()
    invalid[1, 0, 3] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        scan(shape, np.eye(4), [shape]*2, invalid, request(), 1e-6)
    with pytest.raises(ValueError, match="native float64"):
        scan(shape, np.eye(4), [shape]*2, poses.astype(np.float32), request(), 1e-6)


@pytest.mark.parametrize("workers", [1, 8])
def test_native_error_after_near_is_delayed_and_resuming_uses_original_suffix(workers):
    shape, broken, options = coal.Box(.1, .1, .1), coal.BVHModelOBBRSS(), request()
    actor_pose = np.eye(4)
    poses = np.stack([pose([1., 0., 0.]), pose([.08, 0., 0.]),
                      pose([.5, 0., 0.]), pose([2., 0., 0.])])
    with pytest.raises(Exception) as baseline_error:
        raw_distance(shape, actor_pose, broken, poses[2], options)
    targets, ids = prepared([shape, shape, broken, shape], poses)
    batch = targets.scan(shape, transform(actor_pose), ids, options, workers)
    assert len(batch) == 4 and batch.physical_query_count == 4
    first = batch.prefix(0, 1e-6)
    assert first[0] == 1 and first[3] == 1
    for _ in range(2):
        with pytest.raises(type(baseline_error.value)) as actual_error:
            batch.prefix(2, 1e-6)
        assert str(actual_error.value) == str(baseline_error.value)
    suffix = batch.prefix(3, 1e-6)
    assert suffix[:2] == (1, 0) and suffix[3:] == (-1, None)
    assert result_bits(suffix[2]) == result_bits(raw_distance(shape, actor_pose, shape, poses[3], options))


def test_prepared_slots_copy_pose_bits_and_own_shapes_after_inputs_are_released():
    shape = coal.Sphere(.01)
    poses = np.stack([pose([.2, -.0, 0.]), pose([.3, 0., 0.])])
    originals = poses.copy()
    targets, ids = prepared([shape]*2, poses)
    assert targets.missing(ids).size == 0
    assert targets.pose(0).tobytes() == originals[0].tobytes()
    poses[:] = np.nan
    exposed = targets.pose(0)
    exposed[:] = np.nan
    assert targets.pose(0).tobytes() == originals[0].tobytes()
    actor, actor_pose, options = coal.Sphere(.02), np.eye(4), request()
    expected = raw_distance(actor, actor_pose, shape, originals[0], options)
    del shape
    gc.collect()
    batch = targets.scan(actor, transform(actor_pose), ids, options, 8)
    del targets
    gc.collect()
    actual = batch.prefix(0, 1e-6)
    assert result_bits(actual[2]) == result_bits(expected)
    saved = result_bits(actual[2])
    actual[2].clear()
    assert result_bits(batch.prefix(0, 1e-6)[2]) == saved


def test_sparse_source_ids_and_an_unprepared_suffix_keep_original_positions():
    shape, options = coal.Sphere(.01), request()
    table = ComponentTable(5)
    targets = PreparedComponents(table)
    ids = np.array([1, 3], dtype=np.intp)
    poses = np.stack([pose([.03, 0., 0.]), pose([.3, 0., 0.])])
    targets.register(ids, [shape]*2, poses)
    batch = targets.scan(shape, transform(np.eye(4)), ids, options, 8)
    assert batch.prefix(0, .02)[:3] == (0, -1, None)
    later = batch.prefix(1, .02)
    assert later[:2] == (1, 0)
    assert result_bits(later[2]) == result_bits(raw_distance(shape, np.eye(4), shape, poses[1], options))
    missing = targets.scan(shape, transform(np.eye(4)), np.array([1, 4], dtype=np.intp), options, 8)
    assert missing.prefix(0, .02)[3] == 0
    with pytest.raises(ValueError, match="not been prepared"):
        missing.prefix(1, .02)


@pytest.mark.parametrize("workers", [1, 8])
def test_component_workers_inherit_caller_rounding_and_restore_it(workers):
    library = ctypes.CDLL(None)
    previous = library.fegetround()
    actor, target = coal.Box(.03, .04, .05), coal.Capsule(.015, .04)
    actor_pose = pose([.02, -.03, .04], [.12, -.09, .23])
    poses = np.stack([pose([.3+index*.02, -.07, .11], [.07, .11, -.03]) for index in range(8)])
    options = request()
    targets, ids = prepared([target]*len(poses), poses)
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            assert library.fesetround(mode) == 0
            expected = [raw_distance(actor, actor_pose, target, target_pose, options) for target_pose in poses]
            batch = targets.scan(actor, transform(actor_pose), ids, options, workers)
            assert batch.worker_count == workers
            assert library.fegetround() == mode
            # BEGIN consume suffixes so later worker results are also checked.
            for start in range(len(poses)):
                minimum = min(range(start, len(poses)), key=lambda index: expected[index].min_distance)
                found = batch.prefix(start, 1e-6)
                assert found[1] == minimum-start
                assert result_bits(found[2]) == result_bits(expected[minimum])
            # END all comparisons still use original public Coal evidence.
            assert library.fegetround() == mode
    finally:
        assert library.fesetround(previous) == 0


def test_component_scan_rejects_cached_history_and_invalid_worker_counts():
    shape, options = coal.Sphere(.01), request()
    targets, ids = prepared([shape], np.stack([pose([.2, 0., 0.])]))
    for workers in (0, 9):
        with pytest.raises(ValueError):
            targets.scan(shape, transform(np.eye(4)), ids, options, workers)
    options.gjk_initial_guess = coal.GJKInitialGuess.CachedGuess
    with pytest.raises(ValueError, match="cached GJK"):
        targets.scan(shape, transform(np.eye(4)), ids, options, 8)
    options.gjk_initial_guess = coal.GJKInitialGuess.DefaultGuess
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        options.enable_cached_gjk_guess = True
    with pytest.raises(ValueError, match="cached GJK"):
        targets.scan(shape, transform(np.eye(4)), ids, options, 8)
# END registered data and parallel execution retain original raw prefix consumption.
