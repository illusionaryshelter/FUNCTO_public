"""CPU differential checks for TSDF candidate filtering and local Coal reuse."""

# BEGIN compare optimized queries with the original uncached component path.
from itertools import product
from pathlib import Path
import sys

import coal
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation.collision_records import encode_array
from manipulation.robot_geometry_check import _component_rows, _transform_bounds
from manipulation import tsdf_geometry
from manipulation.tsdf_geometry import ObservedTsdfGeometry, coal_component


def component_record(surface, negative=(), *, cells=()):
    """Encode explicit analytic pieces, independently of TSDF reconstruction."""
    record = {"kind": "tsdf", "origin": [40., 0., 0.],
              "shape_xyz": [3, 2, 2], "voxel_size": .5,
              "negative_cell_flat_indices": encode_array(np.asarray(cells, dtype=np.int64))}
    for kind, pieces in (("surface", surface), ("negative", negative)):
        points = [np.asarray(piece, dtype=np.float64).reshape(-1, 3) for piece in pieces]
        record[kind + "_vertices"] = encode_array(
            np.concatenate(points) if points else np.empty((0, 3), dtype=np.float64))
        record[kind + "_offsets"] = encode_array(
            np.r_[0, np.cumsum([len(piece) for piece in points])].astype(np.int64))
    return record


@pytest.fixture
def geometry():
    # Separate x ranges make the same within-kind index easy to distinguish.
    # The oblique segment requires both a translation and a rotated Coal frame.
    pieces = [np.array([[.1, .2, -.1]]),
              np.array([[3., .1, -.1], [3.4, .4, .1]]),
              np.array([[6., .1, .3], [6.4, .1, .3], [6.1, .5, .3]]),
              np.array([[9., .1, -.2], [9.4, .1, -.2],
                        [9., .5, -.2], [9., .1, .2]])]
    return ObservedTsdfGeometry(component_record(
        pieces, [piece + [20., 0., 0.] for piece in pieces], cells=(0, 1)))


def original_components(target, bounds=None, *, surface=True, negative=True):
    """Independently scan every original box; never use the new tree's answer."""
    if not len(target._lower) or not (surface or negative):
        return
    if bounds is None:
        candidates = range(len(target._lower))
    else:
        lower, upper = np.asarray(bounds, dtype=np.float64)
        candidates = [index for index in range(len(target._lower))
                      if np.all(target._upper[index] >= lower)
                      and np.all(target._lower[index] <= upper)]
    surfaces, negatives, _ = target._counts
    corners = np.asarray(list(product((0, 1), repeat=3)))
    for index in candidates:
        if index < surfaces:
            if surface:
                yield "surface", int(index), target.surface_vertices[
                    target.surface_offsets[index]:target.surface_offsets[index + 1]]
        elif index < surfaces + negatives:
            if negative:
                local = index - surfaces
                yield "negative", int(local), target.negative_vertices[
                    target.negative_offsets[local]:target.negative_offsets[local + 1]]
        elif negative:
            yield "negative_cell", int(index - surfaces - negatives), (
                target._lower[index] + corners * target.voxel_size)


@pytest.mark.parametrize("surface,negative", tuple(product((False, True), repeat=2)))
def test_batched_filter_retains_original_ids_order_vertices_and_closed_bounds(geometry, surface, negative):
    rng = np.random.default_rng(912)
    centres = rng.uniform([-2., -.5, -.5], [43., 1., 1.], size=(35, 3))
    widths = rng.uniform(0., 2., size=(35, 3))
    bounds = [None, ([-100.] * 3, [-99.] * 3), ([-100.] * 3, [100.] * 3)]
    bounds.extend(zip(centres - widths, centres + widths))
    # Zero-size corners and faces exercise inclusive comparisons. nextafter
    # separately checks that the optimization did not add any spatial epsilon.
    for lower, upper in zip(geometry._lower, geometry._upper):
        bounds.extend(((lower, lower), (upper, upper), (lower, upper)))
        face = lower.copy()
        face[0] = upper[0]
        bounds.append((face, upper))
        outside = upper.copy()
        outside[0] = np.nextafter(outside[0], np.inf)
        bounds.append((outside, outside))
    for selection in bounds:
        expected = list(original_components(geometry, selection, surface=surface, negative=negative))
        actual = list(geometry.iter_components(selection, surface=surface, negative=negative))
        assert [(kind, index) for kind, index, _ in actual] == [
            (kind, index) for kind, index, _ in expected]
        for (_, _, found), (_, _, wanted) in zip(actual, expected):
            np.testing.assert_array_equal(found, wanted)


def test_empty_inventory_has_no_filter_or_collision_candidates():
    empty = ObservedTsdfGeometry(component_record(()))
    assert list(empty.iter_components()) == []
    assert list(empty.iter_components(([-1.] * 3, [1.] * 3))) == []
    assert list(empty.iter_collision_components()) == []


# BEGIN reserve every unique source component without eager native shapes.
@pytest.mark.parametrize("count", (7, 37))
def test_snapshot_cache_reserve_covers_the_complete_inventory(count):
    record = component_record((), cells=np.arange(count, dtype=np.int64))
    record["shape_xyz"] = [count + 1, 2, 2]
    target = ObservedTsdfGeometry(record)
    assert target._collision_cache_capacity == count
    assert target.collision_cache_storage_nbytes == target.pending_storage_nbytes == count * 4096
    assert target.storage_nbytes >= count * 4096
    assert not target._collision_cache  # Pre-admission reserves are not eager Coal allocations.
    # BEGIN the last source component belongs to the same admitted inventory.
    np.testing.assert_array_equal(target.negative_cell_flat_indices, np.arange(count))
    last_point = np.array([40.+(count-.5)*.5, .25, .25])
    last = next(target.iter_components((last_point, last_point)))
    assert last[:2] == ("negative_cell", count-1)
    # END no fixed nearest subset replaces the complete source field.


def test_large_inventory_admission_reserves_beyond_the_former_lru_capacity():
    # Metadata admission is enough to check the capacity boundary. Do not build
    # hundreds of thousands of shapes or tree nodes merely to test arithmetic.
    count = 262145
    record = component_record((), cells=np.arange(count, dtype=np.int64))
    record["shape_xyz"] = [count+1, 2, 2]
    assert ObservedTsdfGeometry.initial_allocation_bound(record) > count*4096
# END only the tiny fixtures above allocate a real spatial tree.


def test_cache_reuses_only_local_geometry_and_keeps_reserved_storage_constant(geometry, monkeypatch):
    calls = []

    def tracked(vertices):
        calls.append(vertices.copy())
        return coal_component(vertices)

    monkeypatch.setattr(tsdf_geometry, "coal_component", tracked)
    admitted = geometry.storage_nbytes
    cold = list(geometry.iter_collision_components())
    warm = list(geometry.iter_collision_components())
    assert len(calls) == len(cold) == sum(geometry._counts)
    assert geometry.storage_nbytes == admitted
    assert {(kind, index) for kind, index, *_ in cold} == {
        (kind, index) for kind, index, _ in original_components(geometry)}
    for first, second, (_, _, vertices) in zip(cold, warm, original_components(geometry)):
        assert first[:2] == second[:2]
        assert all(before is after for before, after in zip(first[2:], second[2:]))
        shape, local, points = first[2:]
        assert not local.flags.writeable and not points.flags.writeable
        np.testing.assert_allclose(points @ local[:3, :3].T + local[:3, 3],
                                   vertices, rtol=0., atol=1e-14)
        assert shape is not None


def test_interleaved_states_never_evict_or_reconstruct_source_geometry(monkeypatch):
    target = ObservedTsdfGeometry(component_record([[[x, 0., 0.]] for x in (0., 10., 20.)]))
    calls = []

    def tracked(vertices):
        calls.append(vertices.copy())
        return coal_component(vertices)

    monkeypatch.setattr(tsdf_geometry, "coal_component", tracked)

    def select(index):
        point = np.array([10. * index, 0., 0.])
        return next(target.iter_collision_components((point, point)))

    zero, one = select(0), select(1)
    assert select(0)[2] is zero[2]
    two = select(2)
    assert list(target._collision_cache) == [("surface", 0), ("surface", 1), ("surface", 2)]
    again = select(1)
    assert len(calls) == 3 and len(target._collision_cache) == 3
    assert target.pending_storage_nbytes == 0
    assert again[2] is one[2]
    np.testing.assert_array_equal(again[3], one[3])
    np.testing.assert_array_equal(again[4], one[4])
    # Both previous and repeated rows borrow the same map-owned native object.
    for row in (zero, one, two, again):
        assert row[2].radius == 0.
        np.testing.assert_array_equal(row[4], np.zeros((1, 3)))


# BEGIN warm negative cells never repeat their eight-corner construction.
def test_warm_cells_stay_resident_after_other_components_are_used(monkeypatch):
    target = ObservedTsdfGeometry(component_record([[[0., 0., 0.]]], cells=(0, 1)))
    vertices = target._component_vertices
    prepared = []

    def tracked(kind, index):
        prepared.append((kind, index))
        return vertices(kind, index)

    monkeypatch.setattr(target, "_component_vertices", tracked)

    def select(point):
        # Interior points select exactly one negative cell; the separate
        # zero-dimensional surface exercises an intervening different component.
        return next(target.iter_collision_components((point, point)))

    first = select(np.array([40.25, .25, .25]))
    second = select(np.array([40.75, .25, .25]))
    warm = select(np.array([40.25, .25, .25]))
    assert prepared == [("negative_cell", 0), ("negative_cell", 1)]
    assert first[:2] == warm[:2] == ("negative_cell", 0)
    assert all(before is after for before, after in zip(first[2:], warm[2:]))

    select(np.zeros(3))
    assert list(target._collision_cache) == [("negative_cell", 0), ("negative_cell", 1), ("surface", 0)]
    rebuilt = select(np.array([40.75, .25, .25]))
    assert prepared == [("negative_cell", 0), ("negative_cell", 1),
                        ("surface", 0)]
    assert rebuilt[:2] == second[:2]
    assert rebuilt[2] is second[2]
    np.testing.assert_array_equal(rebuilt[3], second[3])
    np.testing.assert_array_equal(rebuilt[4], second[4])
    # BEGIN native reuse preserves actual signed distances, contacts and witnesses.
    # Check both an intersecting body and a separated body using fresh native
    # requests. Equal vertices alone would not test the reconstructed Coal
        # shape or its local placement consumed by the source collision checker.
    for centre in ([40.75, .25, .25], [42., .25, .25]):
        expected = native_evidence(second[2], second[3], np.asarray(centre))
        actual = native_evidence(rebuilt[2], rebuilt[3], np.asarray(centre))
        np.testing.assert_allclose(actual[0], expected[0], rtol=0., atol=1e-12)
        assert actual[1] == expected[1]
        np.testing.assert_allclose(actual[2], expected[2], rtol=0., atol=1e-12)
    # END cache residency does not enter the scene's geometric verdict.
# END other state queries cannot invalidate an immutable cell's geometry.


def test_cache_identity_separates_snapshots_and_component_kinds(geometry):
    first = {row[:2]: row[2:] for row in geometry.iter_collision_components()}
    for index in range(4):
        assert first["surface", index][0] is not first["negative", index][0]
    original = ObservedTsdfGeometry(component_record([[[0., 0., 0.]]]))
    replacement = ObservedTsdfGeometry(component_record([[[1., 2., 3.]]]))
    one = next(original.iter_collision_components())
    two = next(replacement.iter_collision_components())
    assert one[:2] == two[:2] == ("surface", 0)
    assert one[2] is not two[2]
    np.testing.assert_array_equal(one[3][:3, 3], [0., 0., 0.])
    np.testing.assert_array_equal(two[3][:3, 3], [1., 2., 3.])


def world_pose(angle, translation):
    """A proper rotation with a translation, including oblique local segments."""
    cosine, sine = np.cos(angle), np.sin(angle)
    pose = np.eye(4)
    pose[:3, :3] = [[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]]
    pose[:3, 3] = translation
    return pose


def native_evidence(shape, pose, actor_centre):
    """Use fresh requests/results while retaining every raw native witness."""
    actor = coal.Sphere(.08)
    actor_transform = coal.Transform3s(np.eye(3), actor_centre)
    transform = coal.Transform3s(pose[:3, :3], pose[:3, 3])
    request, result = coal.DistanceRequest(), coal.DistanceResult()
    request.enable_signed_distance = True
    coal.distance(actor, actor_transform, shape, transform, request, result)
    collision_request, collision = coal.CollisionRequest(), coal.CollisionResult()
    collision_request.enable_contact, collision_request.num_max_contacts = True, 64
    coal.collide(actor, actor_transform, shape, transform, collision_request, collision)
    contacts = [np.r_[contact.pos, contact.normal, contact.penetration_depth]
                for contact in collision.getContacts()]
    return (np.r_[result.min_distance, result.getNearestPoint1(), result.getNearestPoint2(), result.normal],
            collision.isCollision(), np.asarray(contacts).reshape(-1, 7))


@pytest.mark.parametrize("bounded", (False, True))
def test_cached_component_rows_match_fresh_native_queries_after_pose_changes(geometry, bounded):
    # Query the same snapshot at several world poses. Each native shape sees
    # both separated and intersecting actors, so prior query state cannot leak.
    for pose in (np.eye(4), world_pose(.73, [1.3, -.4, .2]),
                 world_pose(-.29, [-.7, .5, -.3]), np.eye(4)):
        row = {"owner": "target", "name": "analytic_map", "shape": geometry,
               "vertices": geometry, "pose": pose, "layer": "fine", "source": "analytic fixture"}
        bounds = (np.array([-1., -1., -1.]), np.array([25., 2., 2.])) if bounded else None
        margin = .03
        local_bounds = None if bounds is None else _transform_bounds(
            (bounds[0] - margin, bounds[1] + margin), np.linalg.inv(pose))
        expected = list(original_components(geometry, local_bounds))
        actual = list(_component_rows(row, bounds, margin))
        assert [item["tsdf_component_id"] for item in actual] == [
            (kind, index) for kind, index, _ in expected]
        for item, (_, _, vertices) in zip(actual, expected):
            fresh_shape, fresh_local = coal_component(vertices)
            fresh_pose = pose @ fresh_local
            np.testing.assert_array_equal(item["pose"], fresh_pose)
            np.testing.assert_array_equal(item["vertices"]["tsdf_component"],
                                          (vertices - fresh_local[:3, 3]) @ fresh_local[:3, :3])
            for key in ("owner", "name", "layer", "source"):
                assert item[key] == row[key]
            centre = vertices.mean(axis=0) @ pose[:3, :3].T + pose[:3, 3]
            for offset in ([1., .37, .61], [.017, -.011, .023]):
                wanted = native_evidence(fresh_shape, fresh_pose, centre + offset)
                found = native_evidence(item["shape"], item["pose"], centre + offset)
                np.testing.assert_allclose(found[0], wanted[0], rtol=0., atol=1e-12)
                assert found[1] == wanted[1]
                np.testing.assert_allclose(found[2], wanted[2], rtol=0., atol=1e-12)
# END candidate membership and all native geometry evidence retain original semantics.


# BEGIN surface reuse requires every native input and never shares report arrays.
@pytest.mark.parametrize("change", ("none", "pose_bit", "signed_zero", "request", "shape"))
def test_surface_cache_keeps_exact_witnesses_and_invalidates_changed_inputs(geometry, monkeypatch, change):
    from copy import deepcopy
    from manipulation.robot_geometry_check import _scene_component_pairs
    # BEGIN surface calls now enter one state-local native batch.
    from manipulation import scene_surface_batch
    from manipulation.scene_surface_batch import SurfaceDistanceBatch

    actor = {"owner": "robot", "name": "body", "shape": coal.Box(.1, .1, .1),
        "vertices": None, "pose": world_pose(.13, [-2., .2, .3]), "source": "native solid"}
    target = {"owner": "target", "name": "map", "shape": geometry, "vertices": geometry,
        "pose": np.eye(4), "source": "observed field", "layer": None}
    request, cache, calls = coal.DistanceRequest(), {}, []
    original = scene_surface_batch.ordered_distances

    def counted(*args):
        calls.extend([1] * len(args[0]))
        return original(*args)

    monkeypatch.setattr(scene_surface_batch, "ordered_distances", counted)

    def sample(retained):
        rows = []
        surfaces = SurfaceDistanceBatch([actor], [target], {}, request, retained)
        pieces = list(_scene_component_pairs(actor, [target], {}, 1e-6, rows,
            request, set(), set(), surfaces, hints={}, statistics={}))
        # END signed-component enumeration remains the original independent path.
        assert pieces  # Reusing a surface row never skips signed-piece enumeration.
        assert len(rows) == 1
        return rows[0]

    first = sample(cache)
    expected = deepcopy(first)
    first["witness_first"][0] = 12345.
    assert sample(cache) == expected and len(calls) == 1
    if change == "pose_bit":
        actor["pose"][0, 3] = np.nextafter(actor["pose"][0, 3], -np.inf)
    elif change == "signed_zero":
        target["pose"][0, 3] = -0.
    elif change == "request":
        request.gjk_tolerance *= 2.
    elif change == "shape":
        actor["shape"] = coal.Box(.12, .1, .1)
    before = len(calls)
    found = sample(cache)
    assert len(calls)-before == int(change != "none")
    fresh = sample({})
    assert list(found) == list(fresh)
    assert found == fresh
    # Dictionary equality alone would hide a changed sign bit in floating zero.
    for name in ("signed_distance_m", "witness_first", "witness_second"):
        assert np.asarray(found[name], dtype=np.float64).tobytes() == np.asarray(
            fresh[name], dtype=np.float64).tobytes()
    assert len(cache) == (2 if change == "shape" else 1)
# END misses preserve a fresh native query and cached rows expose owned lists only.


# BEGIN native request mutation and failed results can never seed reuse.
@pytest.mark.parametrize("failure", ("request_mutated", "nonfinite"))
def test_surface_cache_does_not_reuse_stateful_or_nonfinite_calls(geometry, monkeypatch, failure):
    from manipulation.robot_geometry_check import _scene_component_pairs
    # BEGIN fault injection targets the state-local surface batch only.
    from manipulation import scene_surface_batch
    from manipulation.scene_surface_batch import SurfaceDistanceBatch

    actor = {"owner": "robot", "name": "body", "shape": coal.Box(.1, .1, .1),
        "vertices": None, "pose": world_pose(.13, [-2., .2, .3]), "source": "native solid"}
    target = {"owner": "target", "name": "map", "shape": geometry, "vertices": geometry,
        "pose": np.eye(4), "source": "observed field", "layer": None}
    request, cache, calls = coal.DistanceRequest(), {}, []
    original = scene_surface_batch.ordered_distances

    def altered(*args):
        calls.extend([1] * len(args[0]))
        batch = original(*args)
        if failure == "request_mutated":
            request.gjk_tolerance *= 2.
        class AlteredBatch:
            def result(self, index):
                result = batch.result(index)
                if failure == "nonfinite":
                    result.min_distance = float("nan")
                return result
        return AlteredBatch()

    monkeypatch.setattr(scene_surface_batch, "ordered_distances", altered)
    for _ in range(2):
        surfaces = SurfaceDistanceBatch([actor], [target], {}, request, cache)
        samples = _scene_component_pairs(actor, [target], {}, 1e-6, [],
            request, set(), set(), surfaces, hints={}, statistics={})
        if failure == "nonfinite":
            with pytest.raises(RuntimeError, match="nonfinite evidence"):
                list(samples)
        else:
            list(samples)
        assert not cache
    assert len(calls) == 2
    # END the request guard and nonfinite rejection still occur at consumption.
# END later queries must repeat the real native call and retain its failures.


# BEGIN preparing a batch must preserve the original decisive-failure boundary.
@pytest.mark.parametrize("stop_after_first", (False, True))
def test_component_batch_defers_later_materialization_error_until_consumed(monkeypatch, stop_after_first):
    from manipulation.robot_geometry_check import _scene_component_pairs
    # BEGIN the no-surface fixture still uses the sole current batch interface.
    from manipulation.scene_surface_batch import SurfaceDistanceBatch

    field = ObservedTsdfGeometry(component_record((), [
        np.array([[0., 0., 0.]]), np.array([[.01, 0., 0.]])]))
    actor = {"owner": "robot", "name": "body", "shape": coal.Box(.1, .1, .1),
        "vertices": None, "pose": np.eye(4), "source": "complete native box"}
    target = {"owner": "target", "name": "points", "shape": field, "vertices": field,
        "pose": np.eye(4), "source": "observed points", "layer": None}
    materialize = tsdf_geometry.coal_component
    calls = []

    def fail_second(points):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("later component preparation failed")
        return materialize(points)

    monkeypatch.setattr(tsdf_geometry, "coal_component", fail_second)
    stopped = set()
    request = coal.DistanceRequest()
    surfaces = SurfaceDistanceBatch([actor], [target], {}, request, {})
    pairs = _scene_component_pairs(actor, [target], {}, 1e-6, [], request,
                                   set(), stopped, surfaces, hints={}, statistics={})
    # END the component failure and stop boundary are unchanged.
    first, second, key = next(pairs)
    assert first is actor
    assert second["tsdf_component_id"] == ("negative", 0)
    assert second["_prequeried_distance"].min_distance <= 1e-6
    if stop_after_first:
        stopped.add(key)
        assert list(pairs) == []
    else:
        with pytest.raises(RuntimeError, match="later component preparation failed"):
            next(pairs)
# END contact permission decides whether the later component is ever consumed.
