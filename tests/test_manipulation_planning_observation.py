"""Small CPU counterexamples for current-scene RGB-D evidence handling."""

# BEGIN pixel ownership and observed-free volume tests without CUDA execution.
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from manipulation.planning_observation import (
    _add_known_solid_voxels, _known_free_voxels, _known_solid_evidence,
    _mapper_memory_bound, _owner_mapper_config, _partition_masks, _require_audit_mapping_memory,
    _require_native_voxel_layout, _views,
    build_audit_scene_from_rgbd, observed_space_from_rgbd,
)
from manipulation.planning_scene import ObservedSpace


def view(depth=None):
    depth = np.full((96, 96), 2.) if depth is None else np.asarray(depth, float)
    height, width = depth.shape
    return {"depth": depth, "K": np.array([[60., 0., (width-1)/2], [0., 60., (height-1)/2], [0., 0., 1.]]),
            "T_output_camera": np.eye(4), "reference_frame": "world",
            "image": np.zeros((*depth.shape, 3), np.uint8), "owner_masks": {},
            "robot_mask": np.zeros(depth.shape, bool)}


def solid_fixture():
    """A declared synthetic physical cube whose front actually occludes depth."""
    import trimesh
    record = view(np.ones((96, 96)))
    mask = np.zeros((96, 96), bool)
    mask[25:71, 25:71] = True
    record["owner_masks"] = {"part": mask}
    pose = np.eye(4)
    pose[2, 3] = 1.5
    frames = {"part": {"T_output_owner": pose, "source": "current visible pose fit",
                       "state": "current_observed", "reference_frame": "world",
                       "observation_ids": ["current"], "max_point_error_m": 0.}}
    solids = {"body": {"mesh": trimesh.creation.box(extents=[1., 1., 1.]),
                        "occupancy_kind": "verified_physical_solid", "geometry_source": "synthetic actual solid",
                        "max_geometry_error_m": 0., "body_kind": "owner", "body_id": "part",
                        "T_body_solid": np.eye(4)}}
    return record, frames, solids


class PlanningObservationTest(unittest.TestCase):
    def test_full_sphere_in_ray_prefix_is_known_but_surface_and_hidden_back_are_not(self):
        space = observed_space_from_rgbd({"camera0_now": view()}, origin=[-.5, -.5, .2], shape=(45, 21, 21),
                                         voxel_size=.05, depth_error_m=.01, reference_frame="world", depth_range_m=(.1, 3.))
        balls = torch.tensor([[[[0., 0., 1., .03], [0., 0., 2., .03], [0., 0., 2.3, .03]]]], dtype=torch.float64)
        costs = space.violation(balls)[0, 0]
        self.assertEqual(costs[0].item(), 0.)
        self.assertGreater(costs[1].item(), 0.)
        self.assertGreater(costs[2].item(), 0.)

    def test_one_invalid_foreground_ray_keeps_the_whole_projected_cell_unknown(self):
        record = view()
        record["depth"][47, 47] = np.nan
        normalized = _views({"current": record}, "world", (.1, 3.))
        known = _known_free_voxels(normalized, origin=[0., 0., 1.], shape=(2, 2, 2),
                                   voxel_size=.01, depth_error_m=0.)
        self.assertFalse(known[0, 0, 0])
        # The valid surrounding depths never fill a missing ray.
        self.assertTrue(np.isnan(record["depth"][47, 47]))

    def test_other_view_can_supply_missing_evidence_without_filling_occlusion(self):
        first, second = view(), view()
        first["depth"][:] = 0.
        second["T_output_camera"][0, 3] = .1
        normalized = _views({"missing": first, "valid_other_camera": second}, "world", (.1, 3.))
        known = _known_free_voxels(normalized, origin=[0., 0., 1.], shape=(2, 2, 2),
                                   voxel_size=.01, depth_error_m=.01)
        self.assertTrue(known.all())
        second["depth"][:] = .6
        blocked = _views({"missing": first, "foreground": second}, "world", (.1, 3.))
        self.assertFalse(_known_free_voxels(blocked, origin=[0., 0., 1.], shape=(2, 2, 2),
                                           voxel_size=.01, depth_error_m=.01).any())

    def test_field_of_view_and_metric_error_are_conservative(self):
        normalized = _views({"current": view()}, "world", (.1, 3.))
        outside = _known_free_voxels(normalized, origin=[1., 0., 1.], shape=(2, 2, 2),
                                     voxel_size=.01, depth_error_m=0.)
        self.assertFalse(outside.any())
        near = _known_free_voxels(normalized, origin=[0., 0., 1.98], shape=(2, 2, 2),
                                  voxel_size=.01, depth_error_m=.02)
        self.assertFalse(near.any())

    def test_owner_partition_removes_exact_pixels_and_keeps_geometry_inside_their_aabb(self):
        record = view()
        owner, other = np.zeros((96, 96), bool), np.zeros((96, 96), bool)
        owner[40, 40], owner[50, 50], other[45, 45] = True, True, True
        record["owner_masks"] = {"part": owner, "other": other}
        record["robot_mask"][30, 30] = True
        normalized = _views({"current": record}, "world", (.1, 3.))
        masks = _partition_masks(normalized, ("part", "other"))["current"]
        self.assertFalse(masks[None][40, 40])
        self.assertFalse(masks[None][30, 30])
        self.assertTrue(masks[None][44, 44])
        self.assertTrue(masks["other"][45, 45])
        self.assertEqual(int(masks["part"].sum()), 2)
        np.testing.assert_array_equal(record["depth"], np.full((96, 96), 2.))
        record["owner_masks"]["other"][40, 40] = True
        with self.assertRaisesRegex(ValueError, "disjoint"):
            _partition_masks(_views({"current": record}, "world", (.1, 3.)), ("part", "other"))

    def test_missing_owner_depth_and_wrong_calibration_cannot_create_geometry(self):
        record = view()
        record["owner_masks"] = {"part": np.zeros((96, 96), bool)}
        with self.assertRaisesRegex(ValueError, "current valid depth"):
            _partition_masks(_views({"current": record}, "world", (.1, 3.)), ("part",))
        record["K"][0, 1] = 1.
        with self.assertRaisesRegex(ValueError, "zero-skew"):
            _views({"current": record}, "world", (.1, 3.))
        record["K"][0, 1] = 0.
        record["reference_frame"] = "unregistered_camera"
        with self.assertRaisesRegex(ValueError, "planning frame"):
            _views({"current": record}, "world", (.1, 3.))

    def test_native_mapper_floor_is_budgeted_before_any_device_allocation(self):
        from curobo.perception import MapperCfg
        cfg = MapperCfg(extent_meters_xyz=(.4, .4, .4), voxel_size=.02,
                        extent_esdf_meters_xyz=(.4, .4, .4), esdf_voxel_size=.025,
                        image_height=96, image_width=96, device="cuda:0")
        self.assertEqual(cfg.max_blocks, 10_000)
        self.assertGreater(_mapper_memory_bound(cfg), 2*10_000*8**3*4)
        with self.assertRaisesRegex(ValueError, "declared budget"):
            build_audit_scene_from_rgbd({"current": view()}, {}, mapper_cfg=cfg, depth_error_m=.01,
                                  owner_voxel_sizes_m={},
                                  reference_frame="world", robot_sphere_materials=(None,),
                                  robot_geometry_source="explicit_model", has_gripper=False,
                                  max_mapping_bytes=1024)

    # BEGIN refined owner grids retain all current depth evidence and CPU admission.
    @staticmethod
    def refined_owner_fixture():
        from curobo.perception import MapperCfg
        first, second = view(), view()
        for record in (first, second):
            record["depth"][:] = 1.
            record["owner_masks"] = {"part": np.zeros((96, 96), bool)}
        first["owner_masks"]["part"][[42, 48, 52], [43, 49, 54]] = True
        second["owner_masks"]["part"][[41, 50, 55], [45, 52, 57]] = True
        first["depth"][42, 43] = .93
        second["depth"][55, 57] = 1.08
        # A labelled invalid depth must remain unknown, while the most distant
        # valid point from the other view must remain inside the owner grid.
        second["owner_masks"]["part"][30, 30] = True
        second["depth"][30, 30] = np.nan
        second["T_output_camera"][:3, 3] = [.04, -.02, .03]
        pose = np.eye(4)
        angle = .63
        pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        pose[:3, 3] = [.02, -.01, 1.]
        cfg = MapperCfg(extent_meters_xyz=(.5, .5, .5), voxel_size=.02,
            extent_esdf_meters_xyz=(.5, .5, .5), esdf_voxel_size=.025,
            grid_center=torch.tensor([0., 0., 1.]),
            image_height=96, image_width=96, device="cuda:0")
        return {"first": first, "second": second}, {"part": {
            "T_output_owner": pose, "source": "Current calibrated two-view owner pose"}}, cfg

    def test_refined_owner_grid_encloses_every_valid_label_from_both_rotated_views(self):
        views, frames, cfg = self.refined_owner_fixture()
        normalized = _views(views, "world", (.1, 3.))
        masks = _partition_masks(normalized, frames)
        pose, voxel, error = frames["part"]["T_output_owner"], .005, .003
        refined = _owner_mapper_config(cfg, normalized, masks, "part", pose, voxel, error)
        points = []
        for record in views.values():
            for row, column in np.argwhere(record["owner_masks"]["part"] & np.isfinite(record["depth"])):
                camera = np.linalg.solve(record["K"], [column, row, 1.])*record["depth"][row, column]
                world = record["T_output_camera"][:3, :3]@camera+record["T_output_camera"][:3, 3]
                points.append(pose[:3, :3].T@(world-pose[:3, 3]))
        points = np.asarray(points)
        self.assertEqual(len(points), 6)
        padding = 4*voxel+error
        lower = refined.grid_center.numpy()-np.asarray(refined.extent_meters_xyz)/2
        upper = refined.grid_center.numpy()+np.asarray(refined.extent_meters_xyz)/2
        np.testing.assert_allclose(lower, points.min(axis=0)-padding, atol=1e-8)
        np.testing.assert_allclose(upper, points.max(axis=0)+padding, atol=1e-8)
        self.assertTrue(np.all(points >= lower+padding-1e-8))
        self.assertTrue(np.all(points <= upper-padding+1e-8))
        self.assertEqual(refined.voxel_size, voxel)
        self.assertEqual(refined.esdf_voxel_size, voxel)
        self.assertEqual(refined.truncation_distance, 2*voxel)
        self.assertEqual(cfg.voxel_size, .02)
        self.assertEqual(cfg.esdf_voxel_size, .025)
        np.testing.assert_array_equal(cfg.grid_center.numpy(), [0., 0., 1.])
        self.assertTrue(np.isnan(views["second"]["depth"][30, 30]))

    def test_refinement_preserves_background_free_space_and_budgets_all_partitions(self):
        views, frames, cfg = self.refined_owner_fixture()
        spaces, configs, budgets = [], [], []

        def capture_space(*args, **kwargs):
            result = ObservedSpace(*args, **kwargs)
            spaces.append(result)
            return result

        def capture_budget(config):
            configs.append(config)
            return _mapper_memory_bound(config)

        def stop_before_native(estimated, device, **limits):
            budgets.append(estimated)
            self.assertEqual(limits, {"max_host_rss_bytes": 8*1024**3,
                                      "max_device_used_bytes": 6*1024**3})
            raise ValueError("CPU admission stop")

        options = dict(mapper_cfg=cfg, depth_error_m=.003, reference_frame="world",
            robot_sphere_materials=(None,), robot_geometry_source="Declared synthetic robot",
            has_gripper=False, max_mapping_bytes=1024**3)
        with patch("manipulation.planning_observation.ObservedSpace", side_effect=capture_space), \
                patch("manipulation.planning_observation._mapper_memory_bound", side_effect=capture_budget), \
                patch("manipulation.planning_observation._require_audit_mapping_memory", side_effect=stop_before_native), \
                patch("torch.cuda.mem_get_info", side_effect=AssertionError("CPU audit touched CUDA")), \
                patch("curobo.perception.Mapper") as mapper:
            for resolution in (.01, .005):
                with self.assertRaisesRegex(ValueError, "CPU admission stop"):
                    build_audit_scene_from_rgbd(views, frames, owner_voxel_sizes_m={"part": resolution}, **options)
            mapper.assert_not_called()
        self.assertEqual(len(configs), 4)
        self.assertIs(configs[0], cfg)
        self.assertIs(configs[2], cfg)
        self.assertEqual(configs[3].voxel_size, .005)
        # Native admission happens immediately before the next Mapper, so the
        # first check covers only the background; later retained fields already
        # count in live RSS/device usage rather than being charged twice.
        self.assertEqual(budgets, [_mapper_memory_bound(cfg)]*2)
        np.testing.assert_array_equal(spaces[0].known_free, spaces[1].known_free)
        np.testing.assert_array_equal(spaces[0].origin, spaces[1].origin)
        self.assertEqual(spaces[0].voxel_size, spaces[1].voxel_size)
        # Each allocation fits separately; their combined allocation does not.
        # The real builder must reject their sum before live-device admission.
        individual = [_mapper_memory_bound(config) for config in configs[2:]]
        with patch("manipulation.planning_observation._require_audit_mapping_memory") as memory, \
                patch("curobo.perception.Mapper") as mapper:
            with self.assertRaisesRegex(ValueError, "declared budget"):
                build_audit_scene_from_rgbd(views, frames, owner_voxel_sizes_m={"part": .005},
                    **{**options, "max_mapping_bytes": max(individual)})
            memory.assert_not_called()
            mapper.assert_not_called()

    def test_refinement_rejects_invalid_resolution_or_empty_evidence_before_cuda(self):
        views, frames, cfg = self.refined_owner_fixture()
        options = dict(mapper_cfg=cfg, depth_error_m=.003, reference_frame="world",
            robot_sphere_materials=(None,), robot_geometry_source="Declared synthetic robot",
            has_gripper=False, max_mapping_bytes=1024**3)
        with patch("torch.cuda.mem_get_info", side_effect=AssertionError("CPU rejection touched CUDA")) as memory, \
                patch("curobo.perception.Mapper") as mapper:
            for resolution in (0., -.001, float("nan"), float("inf"), True, "0.005", [0.005]):
                with self.subTest(resolution=resolution), self.assertRaisesRegex(ValueError, "positive metres"):
                    build_audit_scene_from_rgbd(views, frames, owner_voxel_sizes_m={"part": resolution}, **options)
            for sizes in (None, .005, {}, {"part": .005, "extra": .01}, {"other": .005}):
                with self.subTest(sizes=sizes), self.assertRaisesRegex(ValueError, "exactly every"):
                    build_audit_scene_from_rgbd(views, frames, owner_voxel_sizes_m=sizes, **options)
            for record in views.values():
                record["depth"][record["owner_masks"]["part"]] = 0.
            with self.assertRaisesRegex(ValueError, "current valid depth"):
                build_audit_scene_from_rgbd(views, frames, owner_voxel_sizes_m={"part": .005}, **options)
            memory.assert_not_called()
            mapper.assert_not_called()
    # END owner precision does not change background evidence or bypass budgets.

    # BEGIN resource limits must fail before native mapping allocation.
    def test_mapping_total_limits_and_missing_hand_fail_without_cuda_access(self):
        from curobo.perception import MapperCfg
        cfg = MapperCfg(extent_meters_xyz=(.4, .4, .4), voxel_size=.02,
                        extent_esdf_meters_xyz=(.4, .4, .4), esdf_voxel_size=.025,
                        image_height=96, image_width=96, device="cuda:0")
        options = dict(mapper_cfg=cfg, depth_error_m=.01, reference_frame="world",
                       robot_sphere_materials=(None,), robot_geometry_source="explicit_model",
                       owner_voxel_sizes_m={})
        with patch("torch.cuda.mem_get_info", side_effect=AssertionError("CPU rejection touched CUDA")) as memory, \
                patch("curobo.perception.Mapper") as mapper:
            for value in (0, True, -1, 6*1024**3+1):
                with self.subTest(mapping=value), self.assertRaisesRegex(ValueError, "both declared total"):
                    build_audit_scene_from_rgbd({"current": view()}, {}, has_gripper=False,
                                          max_mapping_bytes=value, **options)
            with self.assertRaisesRegex(ValueError, "twelve GiB"):
                build_audit_scene_from_rgbd({"current": view()}, {}, has_gripper=False,
                                      max_host_rss_bytes=12*1024**3+1, **options)
            with self.assertRaisesRegex(ValueError, "eight GiB"):
                build_audit_scene_from_rgbd({"current": view()}, {}, has_gripper=False,
                                      max_device_used_bytes=8*1024**3+1, **options)
            with self.assertRaisesRegex(ValueError, "current feedback"):
                build_audit_scene_from_rgbd({"current": view()}, {}, has_gripper=True, **options)
            memory.assert_not_called()
            mapper.assert_not_called()

    def test_live_free_memory_must_cover_all_mapping_buffers_and_transient_reserve(self):
        from curobo.perception import MapperCfg
        cfg = MapperCfg(extent_meters_xyz=(.4, .4, .4), voxel_size=.02,
                        extent_esdf_meters_xyz=(.4, .4, .4), esdf_voxel_size=.025,
                        image_height=96, image_width=96, device="cuda:0")
        estimated = _mapper_memory_bound(cfg)
        required = estimated+64*1024**2
        with patch("torch.cuda.mem_get_info", return_value=(required-1, required-1)) as memory, \
                patch("manipulation.planning_observation.require_host_memory"), \
                patch("curobo.perception.Mapper") as mapper:
            with self.assertRaisesRegex(ValueError, "gpu_native_reserve"):
                build_audit_scene_from_rgbd({"current": view()}, {}, mapper_cfg=cfg, depth_error_m=.01,
                                      owner_voxel_sizes_m={},
                                      reference_frame="world", robot_sphere_materials=(None,),
                                      robot_geometry_source="explicit_model", has_gripper=False)
            memory.assert_called_once_with(torch.device("cuda:0"))
            mapper.assert_not_called()
        # Exactly the native requirement plus the global free reserve is
        # admitted. No real CUDA context is needed for this policy boundary.
        with patch("torch.cuda.mem_get_info", return_value=(required+2*1024**3, required+2*1024**3)), \
                patch("manipulation.planning_observation.require_host_memory"):
            self.assertEqual(_require_audit_mapping_memory(estimated, "cuda:0"), required)
    # END mapping resource-limit counterexamples.

    # BEGIN independent owner spacing and total-resource policy counterexamples.
    def test_two_owner_spacings_keep_every_label_pose_and_background_evidence(self):
        views, frames, cfg = self.refined_owner_fixture()
        other_pose = np.eye(4)
        other_pose[:3, 3] = [-.2, .1, .7]
        frames["large_part"] = {"T_output_owner": other_pose,
                                 "source": "Independent calibrated owner pose"}
        for record in views.values():
            mask = np.zeros((96, 96), bool)
            mask[[60, 70], [20, 30]] = True
            record["owner_masks"]["large_part"] = mask
            record["depth"][70, 30] = 2.8
        saved_depth = {key: record["depth"].copy() for key, record in views.items()}
        saved_masks = {key: {owner: mask.copy() for owner, mask in record["owner_masks"].items()}
                       for key, record in views.items()}
        selected = {}

        def capture_owner(config, normalized, masks, identity, pose, size, error):
            result = _owner_mapper_config(config, normalized, masks, identity, pose, size, error)
            selected[identity] = result
            return result

        with patch("manipulation.planning_observation._owner_mapper_config", side_effect=capture_owner), \
                patch("manipulation.planning_observation._require_audit_mapping_memory",
                      side_effect=ValueError("CPU admission stop")), \
                patch("torch.cuda.mem_get_info", side_effect=AssertionError("CPU audit touched CUDA")), \
                patch("curobo.perception.Mapper") as mapper:
            with self.assertRaisesRegex(ValueError, "CPU admission stop"):
                build_audit_scene_from_rgbd(views, frames, mapper_cfg=cfg, depth_error_m=.003,
                    owner_voxel_sizes_m={"part": .003, "large_part": .015},
                    reference_frame="world", robot_sphere_materials=(None,),
                    robot_geometry_source="Explicit synthetic robot", has_gripper=False,
                    max_mapping_bytes=2*1024**3)
            mapper.assert_not_called()
        self.assertEqual(selected["part"].voxel_size, .003)
        self.assertEqual(selected["large_part"].voxel_size, .015)
        self.assertEqual(cfg.esdf_voxel_size, .025)
        for identity, config in selected.items():
            pose = frames[identity]["T_output_owner"]
            lower = config.grid_center.numpy()-np.asarray(config.get_actual_extent())/2
            upper = config.grid_center.numpy()+np.asarray(config.get_actual_extent())/2
            for record in views.values():
                for row, col in np.argwhere(record["owner_masks"][identity] & np.isfinite(record["depth"])):
                    camera = np.linalg.solve(record["K"], [col, row, 1.])*record["depth"][row, col]
                    world = record["T_output_camera"][:3, :3]@camera+record["T_output_camera"][:3, 3]
                    point = pose[:3, :3].T@(world-pose[:3, 3])
                    self.assertTrue(np.all((lower <= point) & (point <= upper)))
        for name, record in views.items():
            np.testing.assert_array_equal(record["depth"], saved_depth[name])
            for identity, mask in record["owner_masks"].items():
                np.testing.assert_array_equal(mask, saved_masks[name][identity])

    def test_more_than_one_gib_mapping_is_admitted_under_explicit_total_limits(self):
        from curobo.perception import MapperCfg
        cfg = MapperCfg(extent_meters_xyz=(.4, .4, .4), voxel_size=.02,
            extent_esdf_meters_xyz=(.4, .4, .4), esdf_voxel_size=.025,
            image_height=96, image_width=96, device="cuda:0")
        estimate = 1024**3+1
        # A mocked capacity estimate exercises >1 GiB policy without allocating
        # the memory. Real normalization, coverage and static gates still run.
        with patch("manipulation.planning_observation._mapper_memory_bound", return_value=estimate), \
                patch("manipulation.planning_observation._require_audit_mapping_memory",
                      side_effect=ValueError("reached live admission")) as admission, \
                patch("curobo.perception.Mapper") as mapper:
            with self.assertRaisesRegex(ValueError, "reached live admission"):
                build_audit_scene_from_rgbd({"current": view()}, {}, owner_voxel_sizes_m={},
                    mapper_cfg=cfg, depth_error_m=.01, reference_frame="world",
                    robot_sphere_materials=(None,), robot_geometry_source="Explicit robot",
                    has_gripper=False, max_mapping_bytes=2*1024**3,
                    max_host_rss_bytes=12*1024**3, max_device_used_bytes=8*1024**3)
            admission.assert_called_once_with(estimate, "cuda:0",
                max_host_rss_bytes=12*1024**3, max_device_used_bytes=8*1024**3)
            mapper.assert_not_called()

    def test_mapping_counts_existing_device_use_without_borrowing_torch_cache(self):
        gib = 1024**3
        estimate = gib
        with patch("torch.cuda.mem_get_info", return_value=(6*gib, 12*gib)), \
                patch("torch.cuda.memory_reserved", side_effect=AssertionError("Native map borrowed Torch cache")), \
                patch("manipulation.tsdf_geometry.ObservedTsdfGeometry.pending_storage_total_nbytes", return_value=12345), \
                patch("manipulation.planning_observation.require_host_memory") as host:
            with self.assertRaisesRegex(ValueError, "gpu_device_used"):
                _require_audit_mapping_memory(estimate, "cuda:0",
                    max_host_rss_bytes=12*gib, max_device_used_bytes=6*gib)
            required = _require_audit_mapping_memory(estimate, "cuda:0",
                max_host_rss_bytes=12*gib, max_device_used_bytes=8*gib)
            self.assertEqual(required, estimate+64*1024**2)
            host.assert_called_with(required, pending_existing_bytes=12345, max_rss_bytes=12*gib)

    def test_mapping_host_refusal_precedes_device_access(self):
        with patch("manipulation.planning_observation.require_host_memory",
                   side_effect=ValueError("host total exceeded")) as host, \
                patch("manipulation.tsdf_geometry.ObservedTsdfGeometry.pending_storage_total_nbytes", return_value=0), \
                patch("torch.cuda.mem_get_info", side_effect=AssertionError("Host refusal touched CUDA")):
            with self.assertRaisesRegex(ValueError, "host total exceeded"):
                _require_audit_mapping_memory(1024, "cuda:0", max_host_rss_bytes=12*1024**3,
                                        max_device_used_bytes=8*1024**3)
            host.assert_called_once_with(1024+64*1024**2,
                pending_existing_bytes=0, max_rss_bytes=12*1024**3)

    def test_dense_tsdf_record_limit_is_checked_before_any_gpu_mapping(self):
        from curobo.perception import MapperCfg
        # TSDF and ESDF are distinct native grids. This source has more than
        # eight million TSDF samples despite an inexpensive coarse ESDF, so its
        # float32 independent-evidence record would exceed the existing cap.
        cfg = MapperCfg(extent_meters_xyz=(1.1, 1.1, 1.1), voxel_size=.005,
            extent_esdf_meters_xyz=(1.1, 1.1, 1.1), esdf_voxel_size=.05,
            image_height=96, image_width=96, device="cuda:0")
        with patch("manipulation.planning_observation._require_audit_mapping_memory") as admission, \
                patch("manipulation.planning_observation._known_free_voxels") as coverage, \
                patch("curobo.perception.Mapper") as mapper:
            with self.assertRaisesRegex(ValueError, "32 MiB collision-array"):
                build_audit_scene_from_rgbd({"current": view()}, {}, owner_voxel_sizes_m={},
                    mapper_cfg=cfg, depth_error_m=.01, reference_frame="world",
                    robot_sphere_materials=(None,), robot_geometry_source="Explicit robot",
                    has_gripper=False, max_mapping_bytes=4*1024**3)
            admission.assert_not_called()
            coverage.assert_not_called()
            mapper.assert_not_called()
    # END total limits, not a hidden one-GiB cap, decide live allocation admission.

    def test_pixel_budget_rejects_large_view_before_float64_conversion(self):
        record = view()
        # A zero-stride view is only one scalar in memory. Normalizing it to
        # float64 before checking dimensions would allocate an unwanted 72 MB.
        record["depth"] = np.broadcast_to(np.float32(1.), (3000, 3000))
        with self.assertRaisesRegex(ValueError, "eight-million-pixel"):
            _views({"too_large": record}, "world", (.1, 3.))

    # BEGIN whole-cell physical occupancy evidence and contradictory-input tests.
    def test_rgbd_solids_preserve_original_free_grid(self):
        record, frames, solids = solid_fixture()
        options = dict(origin=[-.6, -.6, .7], shape=(33, 25, 25), voxel_size=.05,
                       depth_error_m=.01, reference_frame="world", depth_range_m=(.1, 3.))
        original = observed_space_from_rgbd({"current": record}, **options)
        occupied = observed_space_from_rgbd({"current": record}, known_solids=solids,
                                            owner_frames=frames, **options)
        np.testing.assert_array_equal(occupied.known_free, original.known_free)
        self.assertFalse(occupied.known_free[16, 12, 12])
        with self.assertRaises(ValueError):
            occupied.known_free[:] = True
        self.assertEqual(occupied.violation(torch.tensor([[[[0., 0., 1.5, .04]]]])).item(), 0.)

    def test_native_rgbd_builder_preserves_free_before_device_allocation(self):
        # Exercise the actual mapper builder up to its device admission gate.
        # Only native mapping is blocked; rays, solid evidence and ObservedSpace
        # construction run normally, so this checks their production wiring.
        from curobo.perception import MapperCfg
        record, frames, solids = solid_fixture()
        cfg = MapperCfg(extent_meters_xyz=(1.4, 1.4, 1.4), voxel_size=.05,
                        extent_esdf_meters_xyz=(1.4, 1.4, 1.4), esdf_voxel_size=.05,
                        grid_center=torch.tensor([0., 0., 1.5]),
                        image_height=96, image_width=96, device="cuda:0")
        spaces = []

        def capture_space(*args, **kwargs):
            result = ObservedSpace(*args, **kwargs)
            spaces.append(result)
            return result

        with patch("manipulation.planning_observation.ObservedSpace", side_effect=capture_space), \
                patch("manipulation.planning_observation._require_audit_mapping_memory",
                      side_effect=ValueError("stop before native mapping")), \
                patch("curobo.perception.Mapper") as mapper:
            with self.assertRaisesRegex(ValueError, "stop before native mapping"):
                build_audit_scene_from_rgbd({"current": record}, frames, mapper_cfg=cfg, depth_error_m=.01,
                    owner_voxel_sizes_m={"part": .05},
                    reference_frame="world", robot_sphere_materials=(None,),
                    robot_geometry_source="synthetic_robot", has_gripper=False,
                    known_solids=solids, max_mapping_bytes=1024**3)
            mapper.assert_not_called()
        self.assertEqual(len(spaces), 1)
        space = spaces[0]
        original = observed_space_from_rgbd({"current": record}, origin=space.origin,
            shape=space.known_free.shape, voxel_size=space.voxel_size, depth_error_m=.01,
            reference_frame="world", depth_range_m=(cfg.depth_minimum_distance, cfg.depth_maximum_distance))
        np.testing.assert_array_equal(space.known_free, original.known_free)
        self.assertGreater(len(space.occupancy_evidence["solids"]), 0)

    def test_current_physical_solid_adds_its_interior_but_not_hidden_air_or_backspace(self):
        record, frames, solids = solid_fixture()
        options = dict(origin=[-.6, -.6, .7], shape=(33, 25, 25), voxel_size=.05,
                       depth_error_m=.01, reference_frame="world", depth_range_m=(.1, 3.))
        before = observed_space_from_rgbd({"current": record}, **options)
        after = observed_space_from_rgbd({"current": record}, known_solids=solids,
                                         owner_frames=frames, **options)
        balls = torch.tensor([[[[0., 0., 1.5, .04], [.58, 0., 1.5, .02], [0., 0., 2.2, .02]]]], dtype=torch.float64)
        self.assertGreater(before.violation(balls)[0, 0, 0].item(), 0.)
        costs = after.violation(balls)[0, 0]
        self.assertEqual(costs[0].item(), 0.)
        self.assertGreater(costs[1].item(), 0.)
        self.assertGreater(costs[2].item(), 0.)
        self.assertEqual(set(after.occupancy_evidence["owners"]), {"part"})
        self.assertEqual(after.occupancy_evidence["solids"][0]["occupancy_kind"], "verified_physical_solid")

    def test_a_watertight_cavity_stays_unknown_and_whole_cell_error_margin_is_required(self):
        import trimesh
        record, frames, solids = solid_fixture()
        inner = trimesh.creation.box(extents=[.4, .4, .4])
        inner.invert()
        solids["body"]["mesh"] = trimesh.util.concatenate((solids["body"]["mesh"], inner))
        self.assertTrue(solids["body"]["mesh"].is_volume)
        space = observed_space_from_rgbd({"current": record}, known_solids=solids, owner_frames=frames,
                                         origin=[-.45, -.45, 1.05], shape=(31, 31, 31), voxel_size=.03,
                                         depth_error_m=0., reference_frame="world", depth_range_m=(.1, 3.))
        balls = torch.tensor([[[[0., 0., 1.5, .02], [.33, 0., 1.5, .02]]]], dtype=torch.float64)
        self.assertGreater(space.violation(balls)[0, 0, 0].item(), 0.)
        self.assertEqual(space.violation(balls)[0, 0, 1].item(), 0.)

        record, frames, solids = solid_fixture()
        views = _views({"current": record}, "world", (.1, 3.))
        cells = np.zeros((2, 2, 2), bool)
        records, _ = _known_solid_evidence(solids, frames, None, views, "world")
        _add_known_solid_voxels(cells, records, origin=[.47, 0., 1.5], voxel_size=.02)
        self.assertTrue(cells[0, 0, 0])
        self.assertFalse(cells[0, 0, 1])  # This cell crosses the real x=.5 surface.
        solids["body"]["max_geometry_error_m"] = .02
        records, _ = _known_solid_evidence(solids, frames, None, views, "world")
        uncertain = np.zeros((2, 2, 2), bool)
        _add_known_solid_voxels(uncertain, records, origin=[.47, 0., 1.5], voxel_size=.02)
        self.assertFalse(uncertain[0, 0, 0])

    def test_free_ray_and_occupied_solid_proofs_must_not_contradict(self):
        record, frames, solids = solid_fixture()
        # Here the current depth explicitly sees beyond the claimed solid.
        # Both proofs include full-cell/error margins before detecting conflict.
        record["depth"][:] = 3.
        with self.assertRaisesRegex(ValueError, "contradicts proved-free"):
            observed_space_from_rgbd({"current": record}, known_solids=solids, owner_frames=frames,
                                      origin=[-.1, -.1, 1.4], shape=(5, 5, 5), voxel_size=.05,
                                      depth_error_m=.01, reference_frame="world", depth_range_m=(.1, 4.))

    def test_closed_collision_hulls_partial_meshes_and_future_poses_are_not_occupied_evidence(self):
        record, frames, solids = solid_fixture()
        views = _views({"current": record}, "world", (.1, 3.))
        solids["body"]["occupancy_kind"] = "collision_outer_proxy"
        with self.assertRaisesRegex(ValueError, "independently verified"):
            _known_solid_evidence(solids, frames, None, views, "world")
        solids["body"]["occupancy_kind"] = "verified_physical_solid"
        frames["part"]["state"] = "predicted_future"
        with self.assertRaisesRegex(ValueError, "current observed"):
            _known_solid_evidence(solids, frames, None, views, "world")
        frames["part"]["state"] = "current_observed"
        frames["part"]["observation_ids"] = ["old_snapshot"]
        with self.assertRaisesRegex(ValueError, "current snapshot"):
            _known_solid_evidence(solids, frames, None, views, "world")
        frames["part"]["observation_ids"] = ["current"]
        solids["body"]["mesh"].update_faces(np.arange(11))
        with self.assertRaisesRegex(ValueError, "closed"):
            _known_solid_evidence(solids, frames, None, views, "world")

    def test_solid_face_budget_is_checked_before_topology_or_proximity_work(self):
        record, frames, solids = solid_fixture()
        mesh = solids["body"]["mesh"]
        mesh.faces = np.tile(mesh.faces, (4167, 1))
        views = _views({"current": record}, "world", (.1, 3.))
        with self.assertRaisesRegex(ValueError, "geometry budget"):
            _known_solid_evidence(solids, frames, None, views, "world")
    # END whole-cell physical occupancy evidence and contradictory-input tests.


# BEGIN native float grid parameters must index the exact completed field.
class NativeVoxelLayoutTest(unittest.TestCase):
    @staticmethod
    def native_fixture(shape, requested_size, name="observed_background"):
        """Run actual native metadata construction, isolating only pose kernels.

        The identity pose has no effect on grid-size arithmetic. Replacing its
        helper avoids compiling Warp transforms in a CPU regression, while
        VoxelData computes and stores its real float32 params and field view.
        """
        from curobo._src.geom.data.data_voxel import VoxelData
        from curobo._src.geom.types import VoxelGrid
        from curobo._src.types.device_cfg import DeviceCfg

        class IdentityPose:
            @classmethod
            def from_batch_list(cls, rows, *, device_cfg):
                assert rows == [[0., 0., 0., 1., 0., 0., 0.]]
                assert device_cfg.device.type == "cpu"
                return cls()

            def inverse(self):
                return self

            def get_pose_vector(self):
                return torch.tensor([[0., 0., 0., 1., 0., 0., 0.]], dtype=torch.float32)

        size = torch.tensor(requested_size, dtype=torch.float32).item()
        field = torch.full(shape, 10000., dtype=torch.float16)
        field.reshape(-1)[0] = -.01
        voxel = VoxelGrid(name=name, pose=[0., 0., 0., 1., 0., 0., 0.],
            dims=[count*size for count in shape], voxel_size=size,
            feature_tensor=field, feature_dtype=field.dtype)
        with patch("curobo._src.geom.data.data_voxel.Pose", IdentityPose):
            storage = VoxelData.create_from_voxel_grids([voxel], DeviceCfg("cpu"))
        assert storage.features.data_ptr() == field.data_ptr()
        return voxel, SimpleNamespace(data=SimpleNamespace(voxels=storage))

    # BEGIN regression for native #699 rounding and unchanged input storage.
    def test_near_integer_dimensions_use_native_rounding_without_metadata_repair(self):
        cases = [((3, 2, 2), .0015, (2, 2, 2)),
                 ((2, 3, 2), .0015, (2, 2, 2)),
                 ((2, 2, 3), .0015, (2, 2, 2)),
                 ((49, 2, 2), .015, (48, 2, 2)),
                 ((120, 120, 100), .01, (119, 119, 100))]
        for shape, size, truncated in cases:
            with self.subTest(shape=shape, size=size):
                voxel, native = self.native_fixture(shape, size)
                field = voxel.feature_tensor.clone()
                params = native.data.voxels.params.clone()
                self.assertEqual(tuple(params[0, 0, :3].to(torch.int32).tolist()), truncated)
                _require_native_voxel_layout(voxel, native)
                torch.testing.assert_close(voxel.feature_tensor, field, rtol=0., atol=0.)
                torch.testing.assert_close(native.data.voxels.params, params, rtol=0., atol=0.)

    def test_native_roundf_ties_and_inconsistent_layouts(self):
        voxel, native = self.native_fixture((3, 2, 2), .0015)
        # The guard must match Warp roundf rather than ties-to-even torch.round.
        native.data.voxels.params[0, 0, 0] = 2.5
        _require_native_voxel_layout(voxel, native)
        for value in (2.499, 3.5, 0., -1., float("nan"), float("inf")):
            with self.subTest(value=value):
                native.data.voxels.params[0, 0, 0] = value
                with self.assertRaises(ValueError):
                    _require_native_voxel_layout(voxel, native)
    # END the guard validates actual indexing, not a historical truncation rule.

    def test_correct_native_integer_layout_allows_positive_roundoff(self):
        # Unequal XYZ dimensions include the current two owner-grid sizes and
        # background size. They exercise general metadata, not task identities.
        for shape, size in (((108, 111, 115), .0015), ((95, 135, 93), .0015), ((94, 67, 80), .015)):
            with self.subTest(shape=shape):
                voxel, native = self.native_fixture(shape, size)
                params = native.data.voxels.params.clone()
                _require_native_voxel_layout(voxel, native)
                self.assertEqual(tuple(params[0, 0, :3].to(torch.int32).tolist()), shape)
                torch.testing.assert_close(native.data.voxels.params, params, rtol=0., atol=0.)
                self.assertEqual(voxel.feature_tensor.shape, torch.Size(shape))
                if shape == (95, 135, 93):
                    self.assertEqual(params[0, 0, 2].item(), 93.00000762939453)

    def test_rgbd_builder_rejects_layout_before_query_or_independent_export(self):
        from curobo.perception import MapperCfg
        from curobo.types import DeviceCfg
        voxel, native = self.native_fixture((3, 2, 2), .0015)
        # BEGIN an actually inconsistent rounded layout must still fail closed.
        native.data.voxels.params[0, 0, 0] = 2.0
        # END a tiny float32 error near 3 is now handled by the repaired kernel.
        cfg = MapperCfg(extent_meters_xyz=(.06, .06, .06), voxel_size=.015,
            extent_esdf_meters_xyz=(.06, .06, .06), esdf_voxel_size=.015,
            image_height=96, image_width=96, device="cuda:0")
        with patch("manipulation.planning_observation._require_audit_mapping_memory"), \
                patch("curobo.perception.Mapper") as mapper, \
                patch("curobo.types.DeviceCfg", return_value=DeviceCfg("cpu")), \
                patch("curobo.types.CameraObservation", side_effect=lambda **values: SimpleNamespace(**values)), \
                patch("curobo.types.Pose.from_matrix", return_value=SimpleNamespace()), \
                patch("curobo._src.geom.collision.collision_scene.SceneCollision.from_config", return_value=native) as create, \
                patch("manipulation.planning_observation.CuroboSphereQuery") as query, \
                patch("manipulation.tsdf_geometry.build_tsdf_geometry") as export:
            mapper.return_value.get_stats.return_value = {
                "allocation_failures": 0, "num_allocated": 0, "active_blocks": 0, "free_count": 0}
            mapper.return_value.compute_esdf.return_value = voxel
            with self.assertRaisesRegex(ValueError, "native_rounded_shape_xyz=\\(2, 2, 2\\)"):
                build_audit_scene_from_rgbd({"current": view()}, {}, mapper_cfg=cfg, depth_error_m=.001,
                    owner_voxel_sizes_m={},
                    reference_frame="world", robot_sphere_materials=(None,),
                    robot_geometry_source="Explicit CPU robot fixture", has_gripper=False,
                    max_mapping_bytes=1024**3)
            mapper.return_value.compute_esdf.assert_called_once()
            create.assert_called_once()
            query.assert_not_called()
            export.assert_not_called()
# END no mismatched native world is silently reinterpreted by its consumers.


# END pixel ownership and observed-free volume tests without CUDA execution.

# BEGIN production tests admit one fixed model without source reconstruction.
class OnlineObservationContractTest(unittest.TestCase):
    # BEGIN exercise actual admission signatures with measured-resource fixtures.
    def test_online_mapping_admission_uses_real_checks_without_lazy_reservations(self):
        from manipulation.planning_observation import _require_mapping_memory
        from manipulation.resource_admission import ResourceLimitError
        process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1024**3),
                                  children=lambda **kwargs: [])
        with patch('psutil.Process', return_value=process), \
             patch('psutil.virtual_memory', return_value=SimpleNamespace(available=16*1024**3)), \
             patch('torch.cuda.mem_get_info', return_value=(11*1024**3, 12*1024**3)) as device:
            self.assertEqual(_require_mapping_memory(1024**3, 'cuda:0'), 1088*1024**2)
            device.assert_called_once()
            with self.assertRaises(ResourceLimitError):
                _require_mapping_memory(1024**3, 'cuda:0', max_host_rss_bytes=2*1024**3)
            self.assertEqual(device.call_count, 1)
    # END a host refusal must occur before device admission or native allocation.

    def test_registered_sphere_radius_expands_owner_support_before_mapping(self):
        views,frames,cfg=PlanningObservationTest.refined_owner_fixture()
        normalized=_views(views,'world',(.1,3.))
        masks=_partition_masks(normalized,frames)
        pose=frames['part']['T_output_owner']
        small=_owner_mapper_config(cfg,normalized,masks,'part',pose,.005,.003)
        supported=_owner_mapper_config(cfg,normalized,masks,'part',pose,.005,.003,query_radius_m=.072)
        np.testing.assert_allclose(np.asarray(supported.extent_esdf_meters_xyz)-small.extent_esdf_meters_xyz,[.144]*3,atol=1e-12)
        np.testing.assert_array_equal(supported.grid_center,small.grid_center)
        self.assertEqual(supported.voxel_size,small.voxel_size)

    def test_native_samples_keep_xyz_order_positive_free_sign_and_physical_centres(self):
        from curobo.perception import MapperCfg
        from curobo.types import DeviceCfg
        from manipulation.planning_observation import _map_online_partition
        cfg=MapperCfg(extent_meters_xyz=(.06,.08,.1),voxel_size=.02,
            extent_esdf_meters_xyz=(.06,.08,.1),esdf_voxel_size=.02,
            grid_center=torch.tensor([.2,.3,1.]),image_height=96,image_width=96,device='cuda:0')
        values=torch.tensor(np.broadcast_to(np.array([-.02,0.,.02])[:,None,None],(3,4,5)).copy(),dtype=torch.float16)
        voxel=SimpleNamespace(feature_tensor=values,pose=[.2,.3,1.,1.,0.,0.,0.],voxel_size=.02)
        normalized=_views({'A':view()},'world',(.1,3.))
        masks=_partition_masks(normalized,{})
        bits=np.full((3,4,5),3,dtype=np.uint8)
        with patch('curobo.perception.Mapper') as mapper, \
             patch('curobo.types.DeviceCfg',return_value=DeviceCfg('cpu')), \
             patch('curobo.types.CameraObservation',side_effect=lambda **kw:SimpleNamespace(**kw)), \
             patch('curobo.types.Pose.from_matrix',return_value=SimpleNamespace()), \
             patch('manipulation.native_voxel_sources.export_source_bits',return_value=bits) as sources, \
             patch('manipulation.tsdf_geometry.build_tsdf_geometry',side_effect=AssertionError('online map exported source solid')), \
             patch('curobo._src.geom.collision.collision_scene.SceneCollision.from_config',side_effect=AssertionError('online map created second collision scene')):
            mapper.return_value.get_stats.return_value={'allocation_failures':0,'num_allocated':0,'active_blocks':0,'free_count':0}
            mapper.return_value.compute_esdf.return_value=voxel
            query,report=_map_online_partition(normalized,masks,None,cfg,np.eye(4))
        np.testing.assert_allclose(query.origin,[.18,.27,.96],atol=1e-12)
        np.testing.assert_array_equal(query.distance,values.float().numpy())
        samples=torch.tensor([[.18,.29,1.],[.2,.29,1.],[.22,.29,1.]],dtype=torch.float64)
        np.testing.assert_allclose(query.signed_distance(samples).numpy(),values.float().numpy()[:,1,2],atol=1e-12)
        self.assertEqual(report['shape_xyz'],[3,4,5])
        self.assertEqual(sources.call_args.kwargs['max_cells'],32*1024**2//4)

    def test_online_builder_maps_one_selected_field_and_uses_complete_robot_radius(self):
        from manipulation.planning_observation import build_scene_from_rgbd
        from manipulation.online_geometry import FrozenESDF,OnlinePlanningScene
        views,frames,cfg=PlanningObservationTest.refined_owner_fixture()
        captured={}
        def make(normalized,masks,identity,config,inverse):
            captured[identity]=config
            shape=np.ceil(np.asarray(config.extent_esdf_meters_xyz)/config.esdf_voxel_size).astype(int)
            origin=config.grid_center.numpy()-(shape-1)*config.esdf_voxel_size/2
            values=np.ones(tuple(shape),dtype=np.float32)
            query=FrozenESDF(values,origin=origin,voxel_size=config.esdf_voxel_size,
                site_valid=np.zeros_like(values,dtype=bool),observed_sign=np.zeros_like(values,dtype=bool),
                owner_ids=('observed_background' if identity is None else identity,),source='mock current field')
            return query,{'shape_xyz':shape.tolist()}
        fine={None:{'center_m':[0.,0.,1.],'extent_m':[.2,.2,.2],'voxel_size_m':.01,
            'measurement_allowance_m':.003,'measurement_evidence':'measured depth bound','source':'declared operation bounds'}}
        body=[{'body_id':'link','sphere_indices':[0],'local_spheres':[[0.,0.,0.,.04]],'geometry_names':['body']}]
        with patch('manipulation.planning_observation._map_online_partition',side_effect=make) as mapper, \
             patch('manipulation.planning_observation._require_mapping_memory'), \
             patch('torch.cuda.mem_get_info',side_effect=AssertionError('CPU contract touched CUDA')):
            result=build_scene_from_rgbd(views,frames,mapper_cfg=cfg,depth_error_m=.003,
                owner_voxel_sizes_m={'part':.005},reference_frame='world',robot_sphere_materials=(None,),
                robot_geometry_source='declared native robot',has_gripper=False,max_mapping_bytes=1024**3,
                fine_regions=fine,robot_collision_bodies=body,
                unknown_policy={'conditional_regions':[],'outside_regions':'forbid'})
        self.assertIsInstance(result,OnlinePlanningScene)
        self.assertEqual(mapper.call_count,2)
        self.assertEqual(captured[None].esdf_voxel_size,.01)
        normalized=_views(views,'world',(.1,3.))
        expected=_owner_mapper_config(cfg,normalized,_partition_masks(normalized,frames),'part',frames['part']['T_output_owner'],.005,.003,query_radius_m=.04)
        np.testing.assert_allclose(captured['part'].extent_esdf_meters_xyz,expected.extent_esdf_meters_xyz)
        self.assertFalse(hasattr(result.background,'refinement_source'))
# END these CPU tests never allocate native mapping or simulator state.

if __name__ == "__main__":
    unittest.main()
