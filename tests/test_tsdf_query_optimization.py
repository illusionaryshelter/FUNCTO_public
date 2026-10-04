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
    """Preserve the pre-optimization Python AABB filter as a differential oracle."""
    if target._tree is None or not (surface or negative):
        return
    if bounds is None:
        candidates = range(len(target._centres))
    else:
        lower, upper = np.asarray(bounds, dtype=np.float64)
        centre = (lower + upper) / 2
        radius = float(np.linalg.norm(upper - lower) / 2) + target._radius
        candidates = target._tree.query_ball_point(centre, radius, return_sorted=True)
        candidates = [index for index in candidates
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


# BEGIN budget large inventories without allocating their native shapes.
@pytest.mark.parametrize("count,expected_capacity", ((16385, 16385), (262145, 262144)))
def test_snapshot_cache_reserve_covers_inventory_up_to_the_finite_cap(count, expected_capacity):
    record = component_record((), cells=np.arange(count, dtype=np.int64))
    record["shape_xyz"] = [count + 1, 2, 2]
    target = ObservedTsdfGeometry(record)
    assert target._collision_cache_capacity == expected_capacity
    assert target.storage_nbytes >= expected_capacity * 4096
    assert not target._collision_cache  # Pre-admission reserves are not eager Coal allocations.
# END oversized inventories keep bounded residency and the normal eviction path.


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


def test_cache_eviction_preserves_live_rows_and_reconstructs_evicted_geometry(monkeypatch):
    target = ObservedTsdfGeometry(component_record([[[x, 0., 0.]] for x in (0., 10., 20.)]))
    target._collision_cache_capacity = 2
    calls = []

    def tracked(vertices):
        calls.append(vertices.copy())
        return coal_component(vertices)

    monkeypatch.setattr(tsdf_geometry, "coal_component", tracked)

    def select(index):
        point = np.array([10. * index, 0., 0.])
        return next(target.iter_collision_components((point, point)))

    zero, one = select(0), select(1)
    assert select(0)[2] is zero[2]  # A hit makes zero the most recently used.
    two = select(2)
    assert list(target._collision_cache) == [("surface", 0), ("surface", 2)]
    again = select(1)
    assert len(calls) == 4 and len(target._collision_cache) == 2
    assert again[2] is not one[2]
    np.testing.assert_array_equal(again[3], one[3])
    np.testing.assert_array_equal(again[4], one[4])
    # Eviction drops cache ownership; a previously yielded row remains usable.
    for row in (zero, one, two, again):
        assert row[2].radius == 0.
        np.testing.assert_array_equal(row[4], np.zeros((1, 3)))


# BEGIN a warm negative-cell cache must skip the otherwise repeated eight corners.
def test_warm_cache_skips_vertices_and_eviction_rebuilds_exact_negative_cell(monkeypatch):
    target = ObservedTsdfGeometry(component_record([[[0., 0., 0.]]], cells=(0, 1)))
    target._collision_cache_capacity = 2
    vertices = target._component_vertices
    prepared = []

    def tracked(kind, index):
        prepared.append((kind, index))
        return vertices(kind, index)

    monkeypatch.setattr(target, "_component_vertices", tracked)

    def select(point):
        # Interior points select exactly one negative cell; the separate
        # zero-dimensional surface is used only to force normal LRU eviction.
        return next(target.iter_collision_components((point, point)))

    first = select(np.array([40.25, .25, .25]))
    second = select(np.array([40.75, .25, .25]))
    warm = select(np.array([40.25, .25, .25]))
    assert prepared == [("negative_cell", 0), ("negative_cell", 1)]
    assert first[:2] == warm[:2] == ("negative_cell", 0)
    assert all(before is after for before, after in zip(first[2:], warm[2:]))

    select(np.zeros(3))
    assert list(target._collision_cache) == [("negative_cell", 0), ("surface", 0)]
    rebuilt = select(np.array([40.75, .25, .25]))
    assert prepared == [("negative_cell", 0), ("negative_cell", 1),
                        ("surface", 0), ("negative_cell", 1)]
    assert rebuilt[:2] == second[:2]
    assert rebuilt[2] is not second[2]
    np.testing.assert_array_equal(rebuilt[3], second[3])
    np.testing.assert_array_equal(rebuilt[4], second[4])
# END eviction changes residency only; it retains the original corner arithmetic.


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
