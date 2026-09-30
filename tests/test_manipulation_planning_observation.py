"""Small CPU counterexamples for current-scene RGB-D evidence handling."""

# BEGIN pixel ownership and observed-free volume tests without CUDA execution.
import unittest
from unittest.mock import patch

import numpy as np
import torch

from manipulation.planning_observation import (
    _add_known_solid_voxels, _known_free_voxels, _known_solid_evidence,
    _mapper_memory_bound, _partition_masks, _require_mapping_memory, _views,
    build_scene_from_rgbd, observed_space_from_rgbd,
)


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
            build_scene_from_rgbd({"current": view()}, {}, mapper_cfg=cfg, depth_error_m=.01,
                                  reference_frame="world", robot_sphere_materials=(None,),
                                  robot_geometry_source="explicit_model", has_gripper=False,
                                  max_mapping_bytes=1024)

    # BEGIN resource limits must fail before native mapping allocation.
    def test_mapping_hard_limit_and_missing_hand_fail_without_cuda_access(self):
        from curobo.perception import MapperCfg
        cfg = MapperCfg(extent_meters_xyz=(.4, .4, .4), voxel_size=.02,
                        extent_esdf_meters_xyz=(.4, .4, .4), esdf_voxel_size=.025,
                        image_height=96, image_width=96, device="cuda:0")
        options = dict(mapper_cfg=cfg, depth_error_m=.01, reference_frame="world",
                       robot_sphere_materials=(None,), robot_geometry_source="explicit_model")
        with patch("torch.cuda.mem_get_info", side_effect=AssertionError("CPU rejection touched CUDA")) as memory, \
                patch("curobo.perception.Mapper") as mapper:
            with self.assertRaisesRegex(ValueError, "one GiB"):
                build_scene_from_rgbd({"current": view()}, {}, has_gripper=False,
                                      max_mapping_bytes=1024**3+1, **options)
            with self.assertRaisesRegex(ValueError, "current feedback"):
                build_scene_from_rgbd({"current": view()}, {}, has_gripper=True, **options)
            memory.assert_not_called()
            mapper.assert_not_called()

    def test_live_free_memory_must_cover_all_mapping_buffers_and_transient_reserve(self):
        from curobo.perception import MapperCfg
        cfg = MapperCfg(extent_meters_xyz=(.4, .4, .4), voxel_size=.02,
                        extent_esdf_meters_xyz=(.4, .4, .4), esdf_voxel_size=.025,
                        image_height=96, image_width=96, device="cuda:0")
        estimated = _mapper_memory_bound(cfg)
        required = estimated+64*1024**2
        with patch("torch.cuda.mem_get_info", return_value=(required-1, 12*1024**3)) as memory, \
                patch("curobo.perception.Mapper") as mapper:
            with self.assertRaisesRegex(ValueError, "Insufficient free device memory"):
                build_scene_from_rgbd({"current": view()}, {}, mapper_cfg=cfg, depth_error_m=.01,
                                      reference_frame="world", robot_sphere_materials=(None,),
                                      robot_geometry_source="explicit_model", has_gripper=False)
            memory.assert_called_once_with(torch.device("cuda:0"))
            mapper.assert_not_called()
        # The equality boundary itself is allowed; no native mapper or real
        # CUDA context is needed to prove the finite allocation policy.
        with patch("torch.cuda.mem_get_info", return_value=(required, 12*1024**3)):
            self.assertEqual(_require_mapping_memory(estimated, "cuda:0"), required)
    # END mapping resource-limit counterexamples.

    def test_pixel_budget_rejects_large_view_before_float64_conversion(self):
        record = view()
        # A zero-stride view is only one scalar in memory. Normalizing it to
        # float64 before checking dimensions would allocate an unwanted 72 MB.
        record["depth"] = np.broadcast_to(np.float32(1.), (3000, 3000))
        with self.assertRaisesRegex(ValueError, "eight-million-pixel"):
            _views({"too_large": record}, "world", (.1, 3.))

    # BEGIN whole-cell physical occupancy evidence and contradictory-input tests.
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


if __name__ == "__main__":
    unittest.main()
# END pixel ownership and observed-free volume tests without CUDA execution.
