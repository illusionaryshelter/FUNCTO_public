"""Independent current-pose and stable-order checks for source component hints.

The oracle calls the original public Coal distance overload on every original
component in its original order. It never consults the production tree, hint
selection, retained IDs, or a previous state's floating-point result.
"""

# BEGIN a small real scene isolates hint lifecycle from optimizer trajectories.
import coal
import numpy as np
import pytest

from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker
from manipulation.tsdf_geometry import ObservedTsdfGeometry, tsdf_geometry_from_samples
from test_independent_scene_validation import binding, rigid
from test_robot_geometry_check import URDF


def _transform(pose):
    return coal.Transform3s(pose[:3, :3], pose[:3, 3])


def _evidence_bits(distance, first, second, normal):
    """Do not round distances, signed zeros, nearest points, or normals."""
    return np.r_[distance, first, second, normal].astype(np.float64).tobytes()


@pytest.fixture
def hint_scene(tmp_path):
    path = tmp_path / "hint-scene.urdf"
    path.write_text(URDF)
    checker = RobotGeometryChecker(path, base_frame="base", geometries=[
        NativePrimitive("pad", "base", "cylinder", (.05, .1), rigid(.049),
                        "complete native cylinder for stable point ties")])
    observed = np.zeros((3, 3, 1), dtype=bool)
    observed[0, 0, 0] = observed[0, 2, 0] = True
    observed[2, 0, 0] = observed[2, 2, 0] = True
    model = binding()
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(
        -np.ones(observed.shape), observed, origin=[-.045, -.045, .049],
        voxel_size=.045, name="four_ordered_points", source="four separated source points")
    checker.bind_scene(model)
    robot = checker.query(joint_names=checker.joint_names, joint_positions=[0., 0.],
                          T_reference_base=np.eye(4), pairs=())
    return checker, model, robot


def _assert_current_raw_evidence(checker, robot, shift):
    """Compare one complete scene result to fresh ordered native leaf calls."""
    owner_pose = np.eye(4)
    owner_pose[0, 3] = shift
    geometry = checker._scene_shapes[0]["shape"]
    actor = checker._geometries["pad"][1]
    actor_pose = np.asarray(robot["geometry_poses"]["pad"])
    expected = None
    # All four points remain strictly inside the cylinder's enclosing AABB for
    # these shifts. Thus the exhaustive oracle is the exact production domain.
    for kind, index, shape, local, _ in geometry.iter_collision_components():
        result = coal.DistanceResult()
        coal.distance(actor, _transform(actor_pose), shape, _transform(owner_pose @ local),
                      checker._request, result)
        if expected is None or result.min_distance < expected[2].min_distance:
            expected = (kind, index, result)
    actual = checker.query_scene(robot, tool_pose=np.eye(4), progress=.5,
        finger_positions=[0., 0.], state_kind="actual", owner_poses={"target": owner_pose})
    assert actual["passed"] and not actual["query_gaps"]
    rows = [row for row in actual["pairs"] if "target_tsdf_component" in row]
    assert len(rows) == 1
    row = rows[0]
    assert row["target_tsdf_component"] == list(expected[:2])
    native = expected[2]
    assert _evidence_bits(row["signed_distance_m"], row["witness_first"], row["witness_second"],
                          row["normal_first_to_second"]) == _evidence_bits(
        native.min_distance, native.getNearestPoint1(), native.getNearestPoint2(), native.normal)
    return actual, row


def test_prior_hint_is_requeried_at_the_current_pose_and_ties_keep_original_order(hint_scene):
    checker, _, robot = hint_scene
    initial, first = _assert_current_raw_evidence(checker, robot, 0.)
    assert first["target_tsdf_component"] == ["negative", 0]
    distances = [first["signed_distance_m"]]
    for shift in (.001, -.001):
        actual, row = _assert_current_raw_evidence(checker, robot, shift)
        distances.append(row["signed_distance_m"])
        assert sum(item["prior_hint_query_count"] for item in actual["component_searches"]) == 1
    assert distances[0] != distances[1]
    # The preceding negative x shift makes a later point win. Returning to the
    # symmetric state must select ID 0, even if the later ID was queried first.
    restored, row = _assert_current_raw_evidence(checker, robot, 0.)
    assert row["target_tsdf_component"] == ["negative", 0]
    assert sum(item["prior_hint_query_count"] for item in restored["component_searches"]) == 1
    assert sum(item["prior_hint_query_count"] for item in initial["component_searches"]) == 0


@pytest.mark.parametrize("invalid", [-1, 4, 100003])
def test_invalid_prior_id_never_indexes_a_different_component_or_supplies_old_distance(hint_scene, invalid):
    checker, _, robot = hint_scene
    _assert_current_raw_evidence(checker, robot, 0.)
    assert checker._scene_component_hints
    for key in checker._scene_component_hints:
        previous_geometry, _ = checker._scene_component_hints[key]
        checker._scene_component_hints[key] = (previous_geometry, invalid)
    actual, _ = _assert_current_raw_evidence(checker, robot, -.001)
    assert sum(item["prior_hint_query_count"] for item in actual["component_searches"]) == 0


def test_binding_a_new_source_epoch_discards_the_previous_hint(hint_scene):
    checker, model, robot = hint_scene
    _assert_current_raw_evidence(checker, robot, -.001)
    assert checker._scene_component_hints
    old_hints = checker._scene_component_hints.copy()
    checker.bind_scene(model)
    assert checker._scene_component_hints == {}
    # A same-named replacement has its own source object. Even a stale hint
    # accidentally presented to that new binding cannot select its leaf ID.
    assert all(previous is not checker._scene_shapes[0]["shape"]
               for previous, _ in old_hints.values())
    checker._scene_component_hints.update(old_hints)
    actual, row = _assert_current_raw_evidence(checker, robot, 0.)
    assert row["target_tsdf_component"] == ["negative", 0]
    assert sum(item["prior_hint_query_count"] for item in actual["component_searches"]) == 0


@pytest.mark.parametrize("stop_at_first", [False, True])
def test_a_later_hint_preparation_error_waits_for_its_original_position(monkeypatch, stop_at_first):
    from manipulation.robot_geometry_check import _scene_component_pairs

    record = tsdf_geometry_from_samples(-np.ones((2, 1, 1)), np.ones((2, 1, 1), bool),
        origin=[0., 0., 0.], voxel_size=.01, name="points", source="two original observed points")
    geometry = ObservedTsdfGeometry(record["solids"][0])
    actor = {"owner": "robot", "name": "body", "shape": coal.Box(.1, .1, .1),
             "vertices": None, "pose": np.eye(4), "source": "complete native box"}
    target = {"owner": "target", "name": "points", "shape": geometry, "vertices": geometry,
              "pose": np.eye(4), "source": "observed points", "layer": None}
    original = geometry.collision_component
    prepared = []

    def fail_later_hint(identity):
        prepared.append(identity)
        if identity == 1:
            raise RuntimeError("later hinted component cannot be prepared")
        return original(identity)

    monkeypatch.setattr(geometry, "collision_component", fail_later_hint)
    hints = {("robot", "body", "target", "points", None): (geometry, 1)}
    stopped = set()
    pairs = _scene_component_pairs(actor, [target], {}, 1e-6, [], coal.DistanceRequest(),
        set(), stopped, {}, hints=hints, statistics={})
    _, component, key = next(pairs)
    assert prepared == [1, 0]
    assert component["tsdf_component_id"] == ("negative", 0)
    assert component["_prequeried_distance"].min_distance <= 1e-6
    if stop_at_first:
        # The caller has rejected the earlier physical pair. An attempted later
        # seed must not replace that first failure with its preparation error.
        stopped.add(key)
        assert list(pairs) == []
    else:
        with pytest.raises(RuntimeError, match="later hinted component cannot be prepared"):
            next(pairs)
# END the oracle never treats a prior distance as current geometric evidence.
