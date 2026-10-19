"""Small CPU counterexamples for ordered state-local surface prefetching."""

# BEGIN the serial oracle retains the previous per-encounter Coal/cache path.
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import coal
import numpy as np
import pytest

from manipulation import scene_surface_batch as surface_module
from manipulation.robot_geometry_check import _scene_component_pairs
from manipulation.scene_surface_batch import SurfaceDistanceBatch, surface_batch_workspace_bytes
from manipulation.tsdf_geometry import ObservedTsdfGeometry, tsdf_geometry_from_samples
from test_independent_scene_validation import binding, checker
from test_tsdf_query_optimization import component_record


def assert_report_bytes_equal(actual, expected):
    """Compare independent audit reports, including signed-zero float bytes."""
    assert type(actual) is type(expected)
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            assert_report_bytes_equal(actual[name], expected[name])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_report_bytes_equal(first, second)
    elif isinstance(actual, float):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    else:
        assert actual == expected


class SerialSurfaceQueries:
    """Independent original-order reference, with no prefetch or shadow cache."""

    def __init__(self, actors, targets, decisions, request, cache):
        self.request, self.cache = request, cache
        self.calls = 0

    def consume(self, actor, target):
        surface = target["shape"].surface_shape
        key = id(actor["shape"]), id(surface)
        signature = (self.request.saveToString(), actor["pose"].dtype.str,
            actor["pose"].tobytes(), target["pose"].dtype.str, target["pose"].tobytes())
        cached = self.cache.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1:]
        result = coal.DistanceResult()
        transforms = [coal.Transform3s(row["pose"][:3, :3], row["pose"][:3, 3])
                      for row in (actor, target)]
        coal.distance(actor["shape"], transforms[0], surface, transforms[1], self.request, result)
        self.calls += 1
        value = float(result.min_distance)
        if not np.isfinite(value):
            raise RuntimeError("Observed TSDF surface distance returned nonfinite evidence")
        first, second = np.asarray(result.getNearestPoint1()).copy(), np.asarray(result.getNearestPoint2()).copy()
        if self.request.saveToString() == signature[0]:
            self.cache[key] = signature, value, first, second
        else:
            self.cache.pop(key, None)
        return value, first, second


def field():
    triangle = np.array([[0., -.2, -.2], [0., .2, -.2], [0., 0., .2]])
    return ObservedTsdfGeometry(component_record([triangle]))


def body(owner, name, shape, x=0., *, layer=None):
    pose = np.eye(4)
    pose[0, 3] = x
    return {"owner": owner, "name": name, "shape": shape, "pose": pose,
        "vertices": shape if isinstance(shape, ObservedTsdfGeometry) else None,
        "source": "small exact scene fixture", "layer": layer}


def evidence_bytes(row):
    return tuple(np.asarray(value, dtype=np.float64).tobytes() for value in row)


@pytest.mark.parametrize("positions,expected_misses", [([1., 2., 1.], 3), ([1., 1., 2.], 2)])
@pytest.mark.parametrize("warm", [False, True])
def test_shadow_cache_keeps_last_signature_and_writes_only_when_consumed(positions, expected_misses, warm):
    shape, target = coal.Box(.1, .1, .1), body("map", "surface", field())
    actors = [body("robot", str(index), shape, x) for index, x in enumerate(positions)]
    request, actual_cache, expected_cache = coal.DistanceRequest(), {}, {}
    reference = SerialSurfaceQueries(actors, [target], {}, request, expected_cache)
    if warm:
        reference.consume(actors[0], target)
        actual_cache.update(deepcopy(expected_cache))
    before = deepcopy(actual_cache)
    batch = SurfaceDistanceBatch(actors, [target], {}, request, actual_cache)
    assert batch.scheduled_query_count == expected_misses-int(warm)
    assert batch.cache_hit_count == len(actors)-batch.scheduled_query_count
    assert actual_cache.keys() == before.keys()
    for key in before:
        assert actual_cache[key][0] == before[key][0]
        assert evidence_bytes(actual_cache[key][1:]) == evidence_bytes(before[key][1:])
    for actor in actors:
        assert evidence_bytes(batch.consume(actor, target)) == evidence_bytes(reference.consume(actor, target))
        for key in actual_cache:
            assert actual_cache[key][0] == expected_cache[key][0]
            assert evidence_bytes(actual_cache[key][1:]) == evidence_bytes(expected_cache[key][1:])
    assert batch.storage_nbytes < max(1, batch.scheduled_query_count)*16384


def test_non_surface_pair_marks_visited_before_a_reverse_surface_encounter():
    target = body("map", "surface", field())
    actor = body("payload", "box", coal.Box(.1, .1, .1), 1.)
    batch = SurfaceDistanceBatch([target, actor], [actor, target], {}, coal.DistanceRequest(), {})
    assert batch.scheduled_query_count == batch.cache_hit_count == 0
    assert batch.native_worker_count == batch.storage_nbytes == 0


def test_layer_filter_precedes_pair_deduplication():
    geometry = field()
    coarse = body("map", "surface", geometry, 2., layer="coarse")
    fine = body("map", "surface", geometry, 0., layer="fine")
    actor = body("robot", "box", coal.Box(.1, .1, .1), 1.)
    decisions = {("robot", "box", "map"): {"layer": "fine"}}
    request, cache = coal.DistanceRequest(), {}
    batch = SurfaceDistanceBatch([actor], [coarse, fine], decisions, request, cache)
    reference = SerialSurfaceQueries([actor], [fine], decisions, request, {})
    assert batch.scheduled_query_count == 1
    assert evidence_bytes(batch.consume(actor, fine)) == evidence_bytes(reference.consume(actor, fine))


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_complete_scene_evidence_and_hint_sequence_match_serial(checker, monkeypatch, workers):
    values = np.ones((2, 2, 2))
    values[:, :, 0] = -1.
    model = binding()
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(values,
        np.ones(values.shape, bool), origin=[-.03, -.03, -.06], voxel_size=.06,
        name="surface", source="small occupied and zero geometry")
    robot = checker.query(joint_names=checker.joint_names, joint_positions=[0., 0.],
                          T_reference_base=np.eye(4), pairs=())

    def run(batch_type):
        checker.bind_scene(model)
        reports, hints = [], []
        with monkeypatch.context() as patch:
            patch.setattr(surface_module, "SurfaceDistanceBatch", batch_type)
            for x in (1., 0., 0., .2):
                pose = np.eye(4)
                pose[0, 3] = x
                reports.append(checker.query_scene(robot, tool_pose=np.eye(4), progress=.5,
                    finger_positions=[0., 0.], state_kind="actual", owner_poses={"target": pose}))
                hints.append([(key, saved[1]) for key, saved in checker._scene_component_hints.items()])
        return reports, hints

    expected = run(SerialSurfaceQueries)
    monkeypatch.setattr(surface_module, "SURFACE_QUERY_WORKERS", workers)
    actual = run(SurfaceDistanceBatch)
    assert_report_bytes_equal(actual, expected)
    assert checker._scene_surface_distance_cache
    checker.bind_scene(model)
    assert checker._scene_surface_distance_cache == checker._scene_component_hints == {}


@pytest.mark.parametrize("later_failure", ["prepare", "native"])
def test_later_surface_error_cannot_replace_earlier_component_error(monkeypatch, later_failure):
    actor = body("robot", "box", coal.Box(.1, .1, .1), 1.)
    first, later = body("map", "first", field()), body("other", "later", field())
    request, cache = coal.DistanceRequest(), {}
    original_batch = surface_module.ordered_distances
    later_error = RuntimeError("later native surface failed")
    if later_failure == "prepare":
        later["pose"] = None
    else:
        def batch_with_error(*args):
            native = original_batch(*args)
            def result(index):
                if index == 1:
                    raise later_error
                return native.result(index)
            return SimpleNamespace(result=result)
        monkeypatch.setattr(surface_module, "ordered_distances", batch_with_error)
    surfaces = SurfaceDistanceBatch([actor], [first, later], {}, request, cache)
    assert not cache
    earlier_error = RuntimeError("earlier signed component failed")
    monkeypatch.setattr(first["shape"], "component_indices", Mock(side_effect=earlier_error))
    pairs = _scene_component_pairs(actor, [first, later], {}, 1e-6, [], request,
        set(), set(), surfaces, hints={}, statistics={})
    with pytest.raises(RuntimeError) as caught:
        list(pairs)
    assert caught.value is earlier_error
    assert set(cache) == {(id(actor["shape"]), id(first["shape"].surface_shape))}
    with pytest.raises((AttributeError, RuntimeError)) as caught:
        surfaces.consume(actor, later)
    if later_failure == "native":
        assert caught.value is later_error
    assert len(cache) == 1


def test_surface_reservation_is_added_to_independent_scene_workspace(checker, monkeypatch):
    values = np.ones((2, 2, 2))
    values[:, :, 0] = -1.
    model = binding()
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(values,
        np.ones(values.shape, bool), origin=[1., 1., 1.], voxel_size=.01,
        name="surface", source="bounded memory fixture")
    checker.bind_scene(model)
    admitted = checker.scene_query_workspace_nbytes
    reserve = surface_batch_workspace_bytes(2, checker._request)
    with monkeypatch.context() as patch:
        patch.setattr(surface_module, "surface_batch_workspace_bytes", lambda *args: 0)
        checker.bind_scene(model)
    assert admitted-checker.scene_query_workspace_nbytes == reserve
# END no new numerical tolerance or alternate acceptance backend is involved.
