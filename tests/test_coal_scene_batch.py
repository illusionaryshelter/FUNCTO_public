"""Small CPU equivalence checks for the exact ordered Coal distance bridge.

The baseline is the public coal.distance overload already used by query_scene,
with original NumPy-composed world poses and a fresh result for every pair.
"""

# BEGIN raw bitwise witnesses and near-contact stop order across the native bridge.
from pathlib import Path
import sys

import coal
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation._coal_scene_batch import MAX_BATCH_SIZE, scan_separated_prefix
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


def assert_matches_original(actor, actor_pose, shapes, poses, options, stop):
    baseline = []
    for shape, target_pose in zip(shapes, poses):
        value = raw_distance(actor, actor_pose, shape, target_pose, options)
        baseline.append(value)
        if value.min_distance <= stop:
            break
    separated = len(baseline) - int(bool(baseline) and baseline[-1].min_distance <= stop)
    minimum = min(range(separated), key=lambda index: baseline[index].min_distance) if separated else -1
    actual = scan_separated_prefix(actor, transform(actor_pose), shapes, poses, options, stop)
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


def test_separated_ties_and_owned_result_survive_later_calls():
    actor, target = coal.Box(.04, .06, .08), coal.Box(.08, .02, .06)
    actor_pose = pose([.1, -.2, .3], [.12, -.09, .23])
    owner = pose([.4, .3, -.1], [.07, .11, -.03])
    targets = np.stack([owner @ pose([offset, .02, -.03]) for offset in (1.5, 1., 1., 1.8)])
    options = request()
    signature = options.saveToString()
    original = assert_matches_original(actor, actor_pose, [target]*4, targets, options, 1e-6)
    assert original[1] == 1
    bits = result_bits(original[2])
    # Changing transforms then returning to the initial state must not leak the
    # previous native solver's support guesses or mutate an earlier return value.
    for offset in (.5, -.1, .2):
        moved = targets.copy()
        moved[:, 1, 3] += offset
        assert_matches_original(actor, actor_pose, [target]*4, moved, options, 1e-6)
    assert_matches_original(actor, actor_pose, [target]*4, targets, options, 1e-6)
    assert result_bits(original[2]) == bits
    assert options.saveToString() == signature


@pytest.mark.parametrize("translation,stop", [(.08, 1e-6), (.105, .006)])
def test_first_penetration_or_positive_measurement_band_stops_before_invalid_tail(translation, stop):
    shape = coal.Box(.1, .1, .1)
    actor_pose = np.eye(4)
    targets = np.stack([pose([1., 0., 0.]), pose([translation, 0., 0.]), pose([2., 0., 0.])])
    options = request()
    expected = assert_matches_original(shape, actor_pose, [shape]*3, targets, options, stop)
    assert expected[0] == 1 and expected[3] == 1
    # Neither an invalid shape nor nonfinite transform after the near result is
    # examined. Eager full-sequence extraction would fail this policy boundary.
    targets[2] = np.nan
    actual = scan_separated_prefix(shape, transform(actor_pose), [shape, shape, object()],
                                   targets, options, stop)
    assert actual[:2] == expected[:2] and actual[3] == expected[3]
    assert result_bits(actual[2]) == result_bits(expected[2])
    assert result_bits(actual[4]) == result_bits(expected[4])


def test_near_first_has_no_separated_representative_and_exact_threshold_is_inclusive():
    shape = coal.Sphere(.01)
    poses = np.stack([pose([.04, 0., -0.]), pose([.3, 0., 0.])])
    options = request()
    raw = raw_distance(shape, np.eye(4), shape, poses[0], options)
    result = scan_separated_prefix(shape, transform(np.eye(4)), [shape, object()],
                                   poses, options, raw.min_distance)
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
    assert scan_separated_prefix(shape, transform(actor_pose), [], np.empty((0, 4, 4)),
                                 request(), 1e-6) == (0, -1, None, -1, None)
    values = np.stack([pose([.2+index*.01, 0., 0.]) for index in range(6)])
    assert_matches_original(shape, actor_pose, [shape]*3, values[::2], request(), 1e-6)
    repeated = np.broadcast_to(values[0], (MAX_BATCH_SIZE, 4, 4))
    result = scan_separated_prefix(shape, transform(actor_pose), [shape]*MAX_BATCH_SIZE,
                                   repeated, request(), 1e-6)
    assert result[:2] == (MAX_BATCH_SIZE, 0) and result[3:] == (-1, None)
    expected = raw_distance(shape, actor_pose, shape, repeated[0], request())
    assert result_bits(result[2]) == result_bits(expected)
    with pytest.raises(ValueError, match="at most 4096"):
        scan_separated_prefix(shape, transform(actor_pose), [shape]*(MAX_BATCH_SIZE+1),
                              np.broadcast_to(values[0], (MAX_BATCH_SIZE+1, 4, 4)), request(), 1e-6)


def test_only_visited_invalid_inputs_raise_and_float32_is_not_silently_cast():
    shape = coal.Sphere(.01)
    poses = np.stack([pose([.2, 0., 0.]), pose([.3, 0., 0.])])
    with pytest.raises(TypeError):
        scan_separated_prefix(shape, transform(np.eye(4)), [shape, object()], poses, request(), 1e-6)
    invalid = poses.copy()
    invalid[1, 0, 3] = np.nan
    with pytest.raises(ValueError, match="visited target transform"):
        scan_separated_prefix(shape, transform(np.eye(4)), [shape]*2, invalid, request(), 1e-6)
    with pytest.raises(ValueError, match="native float64"):
        scan_separated_prefix(shape, transform(np.eye(4)), [shape]*2, poses.astype(np.float32), request(), 1e-6)
# END the bridge preserves raw results and stops exactly where Python resumes policy.
