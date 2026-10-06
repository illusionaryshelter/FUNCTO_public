"""Small CPU checks for conservative component-tree search and feasible bounds.

The oracles are closed NumPy AABB comparisons, analytic primitive separation,
and the unchanged public Coal leaf query. These tests exercise geometry that
can be pruned, complete near bands, stable IDs, and native-state isolation.
"""

# BEGIN spatial certificates never replace near-contact or raw-leaf decisions.
import ctypes
import gc
from pathlib import Path
import sys

import coal
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation._coal_scene_batch import (MAX_PROJECTION_DIRECTIONS, ComponentAABBTree,
                                           certified_distance_upper_bound)
from manipulation.tsdf_geometry import coal_component


def pose(translation=(0., 0., 0.), rotation=(0., 0., 0.)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    value[:3, 3] = translation
    return value


def transform(value):
    return coal.Transform3s(value[:3, :3], value[:3, 3])


def distance(first, first_pose, second, second_pose):
    request = coal.DistanceRequest()
    request.enable_signed_distance = True
    result = coal.DistanceResult()
    coal.distance(first, transform(first_pose), second, transform(second_pose), request, result)
    return result


def upper_bound(first, first_pose, second, second_pose, result):
    return certified_distance_upper_bound(first, transform(first_pose), second,
                                           transform(second_pose), result)


def tree_from_bounds(lower, upper, prunable=None):
    lower, upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
    if prunable is None:
        prunable = np.ones(len(lower), dtype=np.bool_)
    return ComponentAABBTree(lower, upper, np.asarray(prunable, dtype=np.bool_))


def test_closed_overlap_ids_match_independent_full_scan_and_own_their_inputs():
    lower = np.array([[0., 0., 0.], [1., -1., 0.], [2., 0., 0.], [-1., 0., 0.]])
    upper = lower + [.5, .5, .5]
    tree = tree_from_bounds(lower, upper, [True, False, True, False])
    for lo, hi in [([.5, 0., 0.], [1., 1., 1.]), ([-1., -2., -2.], [3., 2., 2.]),
                   ([7., 7., 7.], [8., 8., 8.]), ([2., 0., 0.], [2., 0., 0.])]:
        lo, hi = np.array(lo), np.array(hi)
        expected = np.flatnonzero(np.all(upper >= lo, axis=1) & np.all(lower <= hi, axis=1))
        actual = tree.overlap_ids(lo, hi)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.dtype(np.intp)
    lower[:] = 100.
    upper[:] = 101.
    np.testing.assert_array_equal(tree.overlap_ids(np.array([-1., -2., -2.]), np.array([3., 2., 2.])),
                                  np.arange(4, dtype=np.intp))
    assert tree.storage_nbytes == 4096 + 256 * 4


def test_certified_upper_and_tree_prune_far_nodes_keep_equal_ties_and_contact_band():
    actor = coal.Box(1., 1., 1.)
    target = coal.Box(1., 1., 1.)
    centres = np.array([[2., 0., 0.], [9., 0., 0.], [2., 0., 0.], [.9, 0., 0.], [20., 0., 0.]])
    tree = tree_from_bounds(centres - .5, centres + .5, [True, True, True, True, False])
    ids = np.arange(len(centres), dtype=np.intp)
    query = tree.query(actor, transform(pose()), transform(pose()), ids, .02)
    result = distance(actor, pose(), target, pose(centres[0]))
    upper = upper_bound(actor, pose(), target, pose(centres[0]), result)
    assert 1. <= upper <= 1. + 1e-12
    retained, stats = query.retained_ids(upper)
    np.testing.assert_array_equal(retained, [0, 2, 3, 4])
    assert stats["candidate_count"] == 5 and stats["bound_pruned_count"] == 1
    assert stats["retained_count"] == 4 and stats["node_bound_count"] > 0
    # Even a smaller caller bound never removes the complete contact band.
    retained, _ = query.retained_ids(0.)
    assert 3 in retained and 4 in retained
    np.testing.assert_array_equal(query.retained_ids(np.inf)[0], ids)


def test_seed_is_only_current_candidate_and_query_keeps_tree_alive():
    lower = np.array([[1., 0., 0.], [3., 0., 0.], [6., 0., 0.]])
    tree = tree_from_bounds(lower, lower + .2)
    query = tree.query(coal.Box(.1, .1, .1), transform(pose()), transform(pose()),
                       np.array([0, 2], dtype=np.intp), 1e-6)
    assert query.seed_id(2) == 2
    assert query.seed_id(1) == 0
    assert query.seed_id(-1) == 0
    assert MAX_PROJECTION_DIRECTIONS == 64
    assert query.workspace_nbytes == 4096 + 96 * 3 + MAX_PROJECTION_DIRECTIONS * 512
    del tree
    gc.collect()
    np.testing.assert_array_equal(query.retained_ids(np.inf)[0], [0, 2])


def test_local_candidate_masks_do_not_visit_the_rest_of_the_map_or_leak_between_queries():
    lower = np.zeros((64, 3))
    lower[:, 0] = np.arange(64, dtype=np.float64)
    tree = tree_from_bounds(lower, lower + .1)
    shape = coal.Box(.1, .1, .1)
    first = tree.query(shape, transform(pose()), transform(pose()), np.array([63], dtype=np.intp), .001)
    second = tree.query(shape, transform(pose()), transform(pose()), np.array([0], dtype=np.intp), .001)
    assert first.seed_id() == 63
    assert second.seed_id(63) == 0
    selected, stats = first.retained_ids(100.)
    np.testing.assert_array_equal(selected, [63])
    # The actor is closest to the excluded region. Searching that region would
    # visit nearly every leaf even though the current query has one candidate.
    assert stats["node_bound_count"] < 20
    np.testing.assert_array_equal(second.retained_ids(0.)[0], [0])


def test_cached_partial_certificates_reuse_work_and_resume_for_a_larger_cutoff():
    lower = np.array([[.7, 0., 0.], [2., 0., 0.], [0., .2, 0.], [.9, .9, 0.]])
    tree = tree_from_bounds(lower, lower + .1)
    shape = coal.Cylinder(.3, .2)
    ids = np.arange(4, dtype=np.intp)

    def query():
        return tree.query(shape, transform(pose()), transform(pose()), ids, .01)

    search = query()
    initial, initial_stats = search.retained_ids(.05)
    repeated, repeated_stats = search.retained_ids(.05)
    np.testing.assert_array_equal(repeated, initial)
    assert initial_stats["node_bound_count"] > 0
    assert repeated_stats["node_bound_count"] == 0
    # Seed ordering requests complete bounds. It must resume partial entries
    # left by threshold search instead of treating them as finished maxima.
    assert search.seed_id() == query().seed_id()
    # The previous pass may have stopped after just one successful direction.
    # A larger cutoff must resume those certificates, producing exactly the
    # same retained leaf IDs as a fresh complete traversal at that cutoff.
    larger, _ = search.retained_ids(1.5)
    np.testing.assert_array_equal(larger, query().retained_ids(1.5)[0])
    smaller, stats = search.retained_ids(.01)
    np.testing.assert_array_equal(smaller, query().retained_ids(.01)[0])
    assert stats["node_bound_count"] == 0


@pytest.mark.parametrize("actor_kind", ["box", "cylinder", "convex"])
def test_rotated_original_components_retain_coal_minimum_and_every_near_leaf(actor_kind):
    # Each tetrahedron's native local pose is identity. Thus these tests use
    # exactly the original owner transform covered by the node certificate.
    offsets = np.array([[.16, 0., 0.], [.4, .08, .01], [.8, -.1, .05], [.18, .01, .02]])
    template = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .025, 0.], [0., 0., .03]])
    pieces = [template + offset for offset in offsets]
    shapes = [coal_component(points)[0] for points in pieces]
    tree = tree_from_bounds([p.min(0) for p in pieces], [p.max(0) for p in pieces])
    actor = {"box": coal.Box(.08, .07, .06), "cylinder": coal.Cylinder(.04, .06),
             "convex": coal_component(np.array([[-.04, -.03, -.03], [.04, -.03, -.03],
                                                  [0., .04, -.03], [0., 0., .04]]))[0]}[actor_kind]
    owner = pose([.1, -.08, .3], [.24, -.13, .37])
    actor_pose = owner @ pose([0., 0., 0.], [.05, .12, -.06])
    results = [distance(actor, actor_pose, shape, owner) for shape in shapes]
    minimum = min(range(len(results)), key=lambda i: results[i].min_distance)
    query = tree.query(actor, transform(actor_pose), transform(owner), np.arange(4, dtype=np.intp), .01)
    seed = query.seed_id(minimum)
    upper = upper_bound(actor, actor_pose, shapes[seed], owner, results[seed])
    retained, stats = query.retained_ids(upper)
    assert minimum in retained
    assert stats["bound_pruned_count"] >= 1
    for i, result in enumerate(results):
        if result.min_distance <= .01:
            assert i in retained
        if i not in retained:
            assert result.min_distance > .01
            assert result.min_distance > results[minimum].min_distance - 1e-9


def test_cylinder_corner_separation_uses_analytic_circular_support():
    actor = coal.Cylinder(1., 1.)
    tree = tree_from_bounds([[.8, .8, -.1], [-.1, -.1, -.1]],
                            [[.9, .9, .1], [.1, .1, .1]])
    query = tree.query(actor, transform(pose()), transform(pose()), np.arange(2, dtype=np.intp), .1)
    retained, stats = query.retained_ids(.1)
    # Both coordinate AABBs overlap, but the first box is at least sqrt(1.28)-1
    # away from the circular cylinder. The second intersects its interior.
    np.testing.assert_array_equal(retained, [1])
    assert stats["bound_pruned_count"] == 1


def test_points_segments_meshes_and_unknown_primitive_preserve_query_availability():
    pieces = [np.array([[0., 0., 0.]]), np.array([[0., 0., 0.], [.04, .03, .02]]),
              np.array([[0., 0., 0.], [.04, 0., 0.], [0., .03, 0.]])]
    actor_pose = pose([-.3, -.2, -.1], [.1, -.07, .02])
    owner = pose([.1, .2, .3], [-.13, .04, .17])
    target = coal.Box(.02, .03, .04)
    tree = tree_from_bounds([[1., 0., 0.]], [[1.1, .1, .1]])
    for points in pieces:
        actor, local = coal_component(points)
        actual_pose = actor_pose @ local
        query = tree.query(actor, transform(actual_pose), transform(owner), np.array([0], dtype=np.intp), .001)
        assert query.seed_id() == 0
        result = distance(actor, actual_pose, target, owner)
        upper = upper_bound(actor, actual_pose, target, owner, result)
        assert np.isfinite(upper) and upper >= result.min_distance - 1e-9
        # Reverse order also verifies feasible points on original low-dimension targets.
        reverse = distance(target, owner, actor, actual_pose)
        assert np.isfinite(upper_bound(target, owner, actor, actual_pose, reverse))
    cone = coal.Cone(.05, .1)
    query = tree.query(cone, transform(actor_pose), transform(owner), np.array([0], dtype=np.intp), .001)
    np.testing.assert_array_equal(query.retained_ids(0.)[0], [0])
    result = distance(cone, actor_pose, target, owner)
    assert np.isinf(upper_bound(cone, actor_pose, target, owner, result))


def test_empty_tree_uncertified_only_tree_and_invalid_interfaces():
    empty = tree_from_bounds(np.empty((0, 3)), np.empty((0, 3)))
    assert empty.size == 0
    assert empty.overlap_ids(np.zeros(3), np.ones(3)).size == 0
    query = empty.query(coal.Box(.1, .1, .1), transform(pose()), transform(pose()),
                        np.empty(0, dtype=np.intp), .001)
    assert query.seed_id() == -1
    assert query.retained_ids(.1)[0].size == 0
    tree = tree_from_bounds([[2., 0., 0.]], [[3., 1., 1.]], [False])
    query = tree.query(coal.Box(.1, .1, .1), transform(pose()), transform(pose()),
                       np.array([0], dtype=np.intp), .001)
    assert query.seed_id() == 0
    np.testing.assert_array_equal(query.retained_ids(.1)[0], [0])
    with pytest.raises(ValueError, match="native float64"):
        ComponentAABBTree(np.zeros((1, 3), np.float32), np.ones((1, 3)), np.ones(1, dtype=bool))
    with pytest.raises(ValueError, match="unique, sorted"):
        tree.query(coal.Box(.1, .1, .1), transform(pose()), transform(pose()), np.array([0, 0], dtype=np.intp), .001)
    with pytest.raises(ValueError, match="upper_bound"):
        query.retained_ids(float("nan"))


def test_rounding_mode_restored_for_queries_bounds_and_validation_errors():
    library = ctypes.CDLL(None)
    original = library.fegetround()
    actor, target = coal.Box(.1, .1, .1), coal.Cylinder(.02, .03)
    actor_pose, target_pose = pose(), pose([.2, .03, .01])
    result = distance(actor, actor_pose, target, target_pose)
    tree = tree_from_bounds([[.18, .01, -.01]], [[.22, .05, .03]])
    try:
        # Linux/x86 fenv constants, the platform on which this native module is
        # compiled and exercised. Each request must restore the caller's mode.
        for mode in (0, 0x400, 0x800, 0xC00):
            assert library.fesetround(mode) == 0
            query = tree.query(actor, transform(actor_pose), transform(pose()), np.array([0], dtype=np.intp), .001)
            assert library.fegetround() == mode
            query.seed_id()
            assert library.fegetround() == mode
            query.retained_ids(.2)
            assert library.fegetround() == mode
            value = upper_bound(actor, actor_pose, target, target_pose, result)
            assert np.isfinite(value)
            assert library.fegetround() == mode
            with pytest.raises(ValueError):
                query.retained_ids(-1.)
            assert library.fegetround() == mode
    finally:
        library.fesetround(original)


# BEGIN real-face proposals must tighten a valid certificate, never raw evidence.
def _large_face_hull(*, triangles=True, thin=False):
    """A front triangle whose four closest vertices lie on its back face."""
    points = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.],
                       [-.02, -.02, -.03], [-.02, .02, -.03],
                       [.02, -.02, -.03], [.02, .02, -.03]])
    if thin:
        points[:, 1] *= 1e-5
    native = coal.StdVec_Vec3s()
    for point in points:
        native.append(point)
    return coal.ConvexBase.convexHull(native, triangles, "Qt")


def _raw_bits(result):
    """Keep every returned binary64 field, including signed zeros."""
    return np.r_[result.min_distance, result.getNearestPoint1(),
                 result.getNearestPoint2(), result.normal].astype(np.float64).tobytes()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("rotated", [False, True])
def test_native_faces_recover_large_face_witness_and_keep_generic_certificate(reverse, rotated):
    actor, generic, target = _large_face_hull(), _large_face_hull(triangles=False), coal.Sphere(0.)
    owner = pose([.13, -.27, .41], [.31, -.19, .23]) if rotated else pose()
    target_pose = owner @ pose([0., 0., .001])
    first, first_pose, second, second_pose = (target, target_pose, actor, owner) if reverse else (
        actor, owner, target, target_pose)
    raw = distance(first, first_pose, second, second_pose)
    before = _raw_bits(raw)
    value = upper_bound(first, first_pose, second, second_pose, raw)
    old = (upper_bound(target, target_pose, generic, owner, raw) if reverse else
           upper_bound(generic, owner, target, target_pose, raw))
    # The analytic separation is one millimetre. The no-topology convex body
    # still uses the original generic proposal and exposes its 31-mm weakness.
    # Rigid matrices from SciPy have ordinary binary64 orthogonality error;
    # this test tolerance compares coordinates, not a production safety band.
    assert .001 - 1e-12 <= value <= .001 + 1e-10
    assert old >= .03 and value <= old
    assert _raw_bits(raw) == before


@pytest.mark.parametrize("point,expected", [
    ([0., 0., .001], .001),
    ([0., -1.001, .001], np.sqrt(2.) * .001),
    ([1.001, -1.001, .001], np.sqrt(3.) * .001),
])
def test_face_edge_and_vertex_proposals_remain_inside_original_hull(point, expected):
    actor, generic, target = _large_face_hull(), _large_face_hull(triangles=False), coal.Sphere(0.)
    target_pose = pose(point)
    raw = distance(actor, pose(), target, target_pose)
    value = upper_bound(actor, pose(), target, target_pose, raw)
    # The three minima lie at the face centre, its bottom edge, and its bottom
    # right vertex. They follow from z <= 0, y >= -1 and x <= 1 for this hull.
    assert expected - 1e-12 <= value <= expected + 1e-8
    assert value <= upper_bound(generic, pose(), target, target_pose, raw)


def test_thin_faces_and_two_convex_endpoints_cannot_weaken_generic_upper():
    target = coal.Sphere(0.)
    thin, generic_thin = _large_face_hull(thin=True), _large_face_hull(triangles=False, thin=True)
    raw = distance(thin, pose(), target, pose([0., 0., .001]))
    value = upper_bound(thin, pose(), target, pose([0., 0., .001]), raw)
    # Coal can choose the front bottom-edge midpoint [0, -1e-5, 0] rather than
    # the analytic minimizer [0, 0, 0] on this thin face. The certificate makes
    # its proposal feasible; it does not correct the native distance solver's
    # witness. Both that midpoint and the opposite front vertex give the
    # explicit feasible upper hypot(.001, 1e-5), while z <= 0 proves .001 as
    # a lower bound. The 1e-12 allowance only compares independently evaluated
    # floating-point coordinates, not a production collision tolerance.
    feasible_front_upper = np.hypot(.001, 1e-5)
    assert .001 <= value <= feasible_front_upper + 1e-12
    assert value <= upper_bound(generic_thin, pose(), target, pose([0., 0., .001]), raw)
    actor, generic = _large_face_hull(), _large_face_hull(triangles=False)
    moved = pose([0., 0., .1])
    raw = distance(actor, pose(), actor, moved)
    # The first front face z=0 and the second back face z=.07 meet the analytic
    # separation. Both endpoint alternatives must remain certified separately.
    value = upper_bound(actor, pose(), actor, moved, raw)
    assert .07 - 1e-12 <= value <= .07 + 1e-10
    assert value <= upper_bound(generic, pose(), generic, moved, raw)


def test_an_incorrect_finite_witness_is_only_a_proposal_and_never_a_proof():
    actor, generic, target = _large_face_hull(), _large_face_hull(triangles=False), coal.Sphere(0.)
    target_pose = pose([0., 0., .001])
    # Deliberately supply another solid's valid Coal result: its first witness
    # is far outside this actor. No mutable binding internals are needed, and
    # the certificate must still construct a point in the requested actor.
    unrelated = distance(coal.Box(.1, .1, .1), pose([.3, -.2, .8]), target, target_pose)
    before = _raw_bits(unrelated)
    value = upper_bound(actor, pose(), target, target_pose, unrelated)
    assert np.isfinite(value) and value >= .001
    assert value <= upper_bound(generic, pose(), target, target_pose, unrelated)
    assert _raw_bits(unrelated) == before
    # Triangle topology cannot turn an uninitialized native result into
    # distance or contact evidence.
    assert np.isinf(upper_bound(actor, pose(), target, target_pose, coal.DistanceResult()))


def test_tighter_face_upper_keeps_equal_minima_and_entire_measurement_band():
    actor, target = _large_face_hull(), coal.Sphere(0.)
    centres = np.array([[0., 0., .001], [0., 0., .001], [0., 0., .02], [0., 0., .0015]])
    # These synthetic tree leaves exactly enclose the complete zero-radius
    # native spheres at their given world translations. This does not change
    # source TSDF point/segment pruning admission or its frame reconstruction.
    tree = tree_from_bounds(centres, centres)
    query = tree.query(actor, transform(pose()), transform(pose()), np.arange(4, dtype=np.intp), .002)
    raw = distance(actor, pose(), target, pose(centres[0]))
    value = upper_bound(actor, pose(), target, pose(centres[0]), raw)
    retained, _ = query.retained_ids(value)
    np.testing.assert_array_equal(retained, [0, 1, 3])


def test_real_face_proposals_restore_every_supported_caller_rounding_mode():
    library = ctypes.CDLL(None)
    original = library.fegetround()
    actor, target = _large_face_hull(), coal.Sphere(0.)
    actor_pose, target_pose = pose(), pose([0., 0., .001])
    raw = distance(actor, actor_pose, target, target_pose)
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            assert library.fesetround(mode) == 0
            value = upper_bound(actor, actor_pose, target, target_pose, raw)
            assert .001 <= value <= .001 + 1e-10
            assert library.fegetround() == mode
    finally:
        library.fesetround(original)
# END neither topology proposals nor their tightness replace cold leaf queries.
# END only certified far branches are omitted; raw contact-capable leaves survive.
