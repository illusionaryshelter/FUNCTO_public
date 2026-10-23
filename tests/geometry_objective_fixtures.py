"""Small original-field and rigid-relation fixtures for geometry tests."""

import numpy as np
import torch

from manipulation.observed_mapping import FrozenObservedField
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner
from test_native_scene_contract import make_scene


def _quaternion(angles):
    """Smooth unit quaternions for three independent rotation components."""
    half = angles / 2
    c, s = torch.cos(half), torch.sin(half)
    return torch.stack((c[..., 0]*c[..., 1]*c[..., 2]+s[..., 0]*s[..., 1]*s[..., 2],
                        s[..., 0]*c[..., 1]*c[..., 2]-c[..., 0]*s[..., 1]*s[..., 2],
                        c[..., 0]*s[..., 1]*c[..., 2]+s[..., 0]*c[..., 1]*s[..., 2],
                        c[..., 0]*c[..., 1]*s[..., 2]-s[..., 0]*s[..., 1]*c[..., 2]), -1)


def _plane(name="observed_plane"):
    values = np.broadcast_to(np.array([-.04, .01, .06])[:, None, None], (3, 2, 2)).copy()
    return FrozenObservedField(values, np.ones_like(values, bool), weights=np.ones_like(values),
        origin=[-.04, .975, -.025], voxel_size=.05, name=name, source="RGB-D")


def _moving_scene(tmp_path):
    scene = make_scene(tmp_path, fixed=("first", "second"))
    scene.background = _plane()
    field = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[-.01]*3, voxel_size=.02,
        name="material", source="RGB-D")
    scene.owners = (CollisionOwner("material", field,
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared path"),
        "observed material", moving=True,
        attachments=(Attachment((0., 1.), [0., 0., 0.], [1., 0., 0., 0.], "declared", None),)),)
    scene.unknown_policy = {"conditional_regions": [], "outside_regions": "allow_attempt"}
    return scene
