"""Existing entity models and fresh RGB-D share the same planning interface."""

# BEGIN regression coverage for geometry reuse independent of current visibility.
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch

from manipulation.observed_mapping import FrozenObservedField
from manipulation.observed_robot import ObservedBody
from manipulation.planning_observation import build_scene_from_rgbd
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner, ObservedSpace
from manipulation.robot_self_filter import ENVIRONMENT


def field(name):
    return FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[0., 0., 0.], voxel_size=.002,
        name=name, source="saved observation")


def retained_owner():
    poses = np.stack((np.eye(4), np.eye(4)))
    return CollisionOwner("entity", field("entity"), PosePath(np.array([0., 1.]), poses, "source"),
        "saved entity", attachments=(Attachment((0., 1.), [.1, 0., 0.],
            [1., 0., 0., 0.], "declared_attachment", None),))


def arguments(owner):
    return dict(mapper_cfg=None, depth_error_m=.0001, task_owner_ids=("entity",),
        reference_frame="world", robot_bodies=(ObservedBody("link", "link", None, "link", "source"),),
        robot_geometry_source="observed robot", observed_owners=(owner,),
        unknown_policy={"conditional_regions": [], "outside_regions": "allow_attempt"},
        max_mapping_bytes=1000, has_gripper=False)


def coverage():
    return ObservedSpace(np.ones((2, 2, 2), bool), origin=[0., 0., 0.],
        voxel_size=.01, observation_ids=("source",), reference_frame="world")


def test_existing_entity_and_background_need_no_images_or_mapper():
    owner, background, space = retained_owner(), field("background"), coverage()
    with patch("manipulation.observed_mapping.integrate_entity",
               side_effect=AssertionError("Existing geometry must not be integrated again")):
        scene = build_scene_from_rgbd({}, {}, background=background, coverage=space, **arguments(owner))
    assert scene.owners[0] is owner
    assert scene.owners[0].query is owner.query
    assert scene.background is background
    assert scene.coverage is space
    assert scene.observation_report["maximum_serial_mapping_bytes"] == 0
    # Complete occlusion changes no geometry input. Candidate tool motion still
    # moves the original field by the declared fixed attachment.
    position = torch.tensor([[[.2, .3, .4]]], dtype=torch.float64)
    quaternion = torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64)
    p, q = owner.pose(position, quaternion, torch.tensor([[.5]], dtype=torch.float64), (None,))
    np.testing.assert_allclose(p.numpy(), [[[.3, .3, .4]]])
    np.testing.assert_array_equal(q.numpy(), quaternion.numpy())


@dataclass
class MapperConfig:
    """Only native configuration fields consumed by this orchestration test."""
    num_cameras: int = 1
    enable_static: bool = False
    decay_factor: float = 1.
    frustum_decay_factor: float = 1.
    depth_minimum_distance: float = .05
    depth_maximum_distance: float = 5.
    voxel_size: float = .01


def test_new_entity_maps_its_pixels_while_existing_entity_is_passed_unchanged():
    owner = retained_owner()
    target = np.zeros((3, 3), bool)
    target[0, 0] = True
    original_entity = np.zeros((3, 3), bool)
    original_entity[1, 1] = True
    background = ~(target | original_entity)
    depth = np.ones((3, 3))
    view = {"image": np.zeros((3, 3, 3), np.uint8), "depth": depth, "K": np.eye(3),
        "T_output_camera": np.eye(4), "reference_frame": "world",
        "self_filter": {"labels": np.full((3, 3), ENVIRONMENT, np.uint8),
                        "free_ray_depth_m": depth, "source": "capture"},
        "owner_masks": {"target": target}, "background_mask": background}
    built = {}

    def integrate(observations, *, name, **kwargs):
        built[name] = list(observations)[0]["mask"].copy()
        return field(name), {"source": "new observed partition"}

    native = ModuleType("curobo.perception")
    native.MapperCfg = MapperConfig
    args = arguments(owner)
    args.update(mapper_cfg=MapperConfig(), task_owner_ids=("entity", "target"))
    with patch.dict("sys.modules", {"curobo.perception": native}), \
            patch("manipulation.planning_observation._mapper_memory_bound", return_value=1), \
            patch("manipulation.planning_observation._owner_mapper_config", return_value=args["mapper_cfg"]), \
            patch("manipulation.observed_mapping.integrate_entity", side_effect=integrate):
        scene = build_scene_from_rgbd({"old_capture": view},
            {"target": {"T_output_owner": np.eye(4), "source": "target observation"}},
            coverage=coverage(), **args)
    assert set(built) == {"target", "observed_background"}
    np.testing.assert_array_equal(built["target"], target)
    np.testing.assert_array_equal(built["observed_background"], background)
    assert not built["observed_background"][1, 1]
    assert next(row for row in scene.owners if row.owner_id == "entity") is owner
    assert scene.observation_report["endpoint_partition"]["old_capture"]["unmapped_pixels"] == 1
# END masks reconstruct only the entities named by the new observation input.
