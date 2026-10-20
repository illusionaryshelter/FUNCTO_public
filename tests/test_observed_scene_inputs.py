"""Existing entity models and fresh RGB-D share the same planning interface."""

# BEGIN regression coverage for geometry reuse independent of current visibility.
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.observed_mapping import FrozenObservedField
from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker
from manipulation.planning_observation import build_scene_from_rgbd
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner, ObservedSpace
from manipulation.robot_self_filter import ENVIRONMENT, SELF


def field(name):
    return FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[0., 0., 0.], voxel_size=.002,
        name=name, source="saved observation")


def retained_owner():
    poses = np.stack((np.eye(4), np.eye(4)))
    return CollisionOwner("entity", field("entity"), PosePath(np.array([0., 1.]), poses, "source"),
        "saved entity", attachments=(Attachment((0., 1.), [.1, 0., 0.],
            [1., 0., 0., 0.], "declared_attachment", None),))


def arguments(owner, tmp_path):
    # BEGIN known robot geometry is independent of these mapping inputs.
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="mapping_fixture"><link name="base"/></robot>')
    robot = RobotGeometryChecker(urdf, base_frame="base", geometries=(
        NativePrimitive("body", "base", "box", (.01, .01, .01), np.eye(4), "known robot"),))
    return dict(mapper_cfg=None, depth_error_m=.0001, task_owner_ids=("entity",),
        reference_frame="world", robot_model=robot, fixed_geometry_names=(), observed_owners=(owner,),
        unknown_policy={"conditional_regions": [], "outside_regions": "allow_attempt"},
        max_mapping_bytes=1000, has_gripper=False)
    # END the same source checker is handed directly to the scene.


def coverage():
    return ObservedSpace(np.ones((2, 2, 2), bool), origin=[0., 0., 0.],
        voxel_size=.01, observation_ids=("source",), reference_frame="world")


def test_existing_entity_and_background_need_no_images_or_mapper(tmp_path):
    owner, background, space = retained_owner(), field("background"), coverage()
    with patch("manipulation.observed_mapping.integrate_entity",
               side_effect=AssertionError("Existing geometry must not be integrated again")):
        scene = build_scene_from_rgbd({}, {}, background=background, coverage=space,
                                     **arguments(owner, tmp_path))
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


def test_new_entity_maps_its_pixels_while_existing_entity_is_passed_unchanged(tmp_path):
    # BEGIN the public mapper consumes the protected original static partition.
    owner = retained_owner()
    target = np.zeros((7, 7), bool)
    target[0, 0] = True
    original_entity = np.zeros((7, 7), bool)
    original_entity[3, 3] = True
    background = ~(target | original_entity)
    depth = np.ones((7, 7))
    view = {"image": np.zeros((7, 7, 3), np.uint8), "depth": depth, "K": np.eye(3),
        "T_output_camera": np.eye(4), "reference_frame": "world",
        "self_filter": {"labels": np.full((7, 7), ENVIRONMENT, np.uint8),
                        "free_ray_depth_m": depth, "source": "capture"},
        "owner_masks": {"target": target}, "background_mask": background}
    built = {}

    def integrate(observations, *, name, **kwargs):
        built[name] = list(observations)[0]["mask"].copy()
        return field(name), {"source": "new observed partition"}

    native = ModuleType("curobo.perception")
    native.MapperCfg = MapperConfig
    args = arguments(owner, tmp_path)
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
    expected_background = background.copy()
    expected_background[:2, :2] = False
    expected_background[2:5, 2:5] = False
    np.testing.assert_array_equal(built["observed_background"], expected_background)
    np.testing.assert_array_equal(view["background_mask"], background)
    np.testing.assert_array_equal(view["self_filter"]["free_ray_depth_m"], depth)
    assert built["observed_background"][6, 6]
    assert not built["observed_background"][3, 3]
    assert next(row for row in scene.owners if row.owner_id == "entity") is owner
    assert scene.observation_report["endpoint_partition"]["old_capture"]["unmapped_pixels"] == 12
    # END one retained pixel and eleven boundary pixels stay out of the new maps.
    # BEGIN the actual saved inputs reproduce the same one-time boundary.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
    from run_recorded_skill import save_observation_partitions
    from manipulation.planning_observation import _partition_masks, _self_filtered_views
    stored = save_observation_partitions(scene, tmp_path)["old_capture"]
    original_target = np.load(stored["owner_masks"]["target"])
    original_background = np.load(stored["background_candidate_mask_path"])
    admitted_background = np.load(stored["background_mask_path"])
    np.testing.assert_array_equal(original_target, target)
    np.testing.assert_array_equal(original_background, background)
    np.testing.assert_array_equal(admitted_background, expected_background)
    reused = {**view, "owner_masks": {"target": original_target},
              "background_mask": original_background}
    for _ in range(3):
        normalized = _self_filtered_views({"old_capture": reused}, "world", (.05, 5.))
        partition = _partition_masks(normalized, ("target",))["old_capture"]
        np.testing.assert_array_equal(partition[None], admitted_background)
        np.testing.assert_array_equal(partition["target"], target)
    # Snapshots are independent of the caller's later array edits.
    target[0, 0] = False
    background[6, 6] = False
    assert scene.observation_partition_masks["old_capture"]["owner_masks"]["target"][0, 0]
    assert scene.observation_partition_masks["old_capture"]["background_candidate"][6, 6]
    # END reusing semantic inputs neither grows the ring nor invokes perception.
# END masks reconstruct only the entities named by the new observation input.


# BEGIN saved semantic inputs retain pixels excluded from this mapping pass.
@pytest.mark.parametrize("reuse_background", [False, True])
@pytest.mark.parametrize("supply_background", [False, True])
def test_original_partitions_survive_self_filter_and_background_reuse(
        tmp_path, reuse_background, supply_background):
    target = np.zeros((5, 5), bool)
    target[2, 2] = target[2, 3] = True
    labels = np.full((5, 5), ENVIRONMENT, np.uint8)
    labels[2, 3] = SELF
    candidate = ~target
    candidate[4, 4] = False
    depth = np.ones((5, 5))
    view = {"image": np.zeros((5, 5, 3), np.uint8), "depth": depth, "K": np.eye(3),
        "T_output_camera": np.eye(4), "reference_frame": "world",
        "self_filter": {"labels": labels, "free_ray_depth_m": np.where(labels == SELF, 0., depth),
                        "source": "capture"},
        "owner_masks": {"target": target},
        "background_mask": candidate if supply_background else None}
    native = ModuleType("curobo.perception")
    native.MapperCfg = MapperConfig
    args = arguments(retained_owner(), tmp_path)
    args.update(mapper_cfg=MapperConfig(), observed_owners=(), task_owner_ids=("target",))
    fused = {}

    def integrate(observations, *, name, **kwargs):
        fused[name] = list(observations)[0]["mask"].copy()
        return field(name), {"source": "new observed partition"}

    saved_background = field("background") if reuse_background else None
    with patch.dict("sys.modules", {"curobo.perception": native}), \
            patch("manipulation.planning_observation._mapper_memory_bound", return_value=1), \
            patch("manipulation.planning_observation._owner_mapper_config", return_value=args["mapper_cfg"]), \
            patch("manipulation.observed_mapping.integrate_entity", side_effect=integrate):
        scene = build_scene_from_rgbd({"capture": view},
            {"target": {"T_output_owner": np.eye(4), "source": "target observation"}},
            background=saved_background, coverage=coverage(), **args)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
    from run_recorded_skill import save_observation_partitions
    stored = save_observation_partitions(scene, tmp_path)["capture"]
    np.testing.assert_array_equal(np.load(stored["owner_masks"]["target"]), target)
    assert not fused["target"][2, 3]
    if supply_background or not reuse_background:
        expected = candidate if supply_background else ~target
        np.testing.assert_array_equal(np.load(stored["background_candidate_mask_path"]), expected)
    else:
        assert stored["background_candidate_mask_path"] is None
    if reuse_background:
        assert scene.background is saved_background
        assert "observed_background" not in fused
        assert stored["background_mask_path"] is None
    else:
        np.testing.assert_array_equal(
            np.load(stored["background_mask_path"]), fused["observed_background"])
# END original masks and geometry reuse have independent storage semantics.
