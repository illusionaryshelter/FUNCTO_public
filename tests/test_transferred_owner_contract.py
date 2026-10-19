"""A continuous reference cannot silently become sampled collision evidence."""

# BEGIN preserve the owner serializer's exact native-path boundary.
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manipulation.planning_problem import PosePath, TransferredPosePath
from manipulation.planning_scene import CollisionOwner


def pose(position, rotation):
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    return result


def owner(path):
    query = SimpleNamespace(owner_ids=frozenset(("observed_part",)))
    return CollisionOwner("observed_part", query, path, "measured geometry", True)


def test_transferred_path_cannot_lose_its_continuous_rule_in_owner_serialization():
    keys = np.array([0., .13, .57, 1.])
    source = PosePath(keys, np.array([
        pose([s, .1*s, .2], [.4*s, -.7*s, 1.4*s]) for s in keys]),
        source="nonuniform measured material motion",
        attachment=pose([.31, -.17, .08], [.4, -.2, .7]))
    path = TransferredPosePath(source, pose([-.4, .2, .6], [.3, -.5, .1]), [
        {"progress": .29, "pose": pose([.4, -.3, .1], [.8, .2, -1.])},
        {"progress": .73, "pose": pose([-.2, .7, .4], [-.4, 1.1, .9])},
    ], source="independent entry and functional events")

    # These are precisely the path fields the collision-owner snapshot saves.
    # A round trip through that representation changes the physical curve,
    # even though every saved knot still matches and there is no parent path.
    restored = PosePath(path.progress, path.poses, source=path.source,
                        attachment=path.attachment)
    query = np.array([.04, .24, .51, .9])
    discrepancy = np.linalg.norm(path.sample_numpy(query)[:, :3, 3]
                                 - restored.sample_numpy(query)[:, :3, 3], axis=1)
    assert discrepancy.max() > .02
    assert path.parent is None
    with pytest.raises(ValueError, match="native PosePath serialization"):
        owner(path)


def test_native_world_path_with_fixed_offset_retains_owner_support():
    path = PosePath([0., 1.], np.array([
        pose([.1, .2, .3], [0., 0., 0.]),
        pose([.2, .3, .4], [0., 0., .7])]),
        source="observed world motion",
        attachment=pose([.03, -.02, .01], [.1, -.2, .3]))
    assert owner(path).pose_path is path
# END no new collision serialization or weaker reconstructed path is introduced.
