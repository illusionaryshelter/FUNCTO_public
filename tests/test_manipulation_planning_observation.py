"""Small CPU counterexamples for current-scene RGB-D evidence handling."""

import unittest

from types import SimpleNamespace

from unittest.mock import MagicMock, patch

import numpy as np

import torch

from manipulation.planning_observation import (_known_free_voxels, _owner_mapper_config,
    _partition_masks, _self_filtered_views, _self_filter_report, _mapper_intrinsics)

from manipulation.planning_scene import ObservedSpace

def view(depth=None):
    depth = np.full((96, 96), 2.) if depth is None else np.asarray(depth, float)
    height, width = depth.shape
    return {"depth": depth, "K": np.array([[60., 0., (width-1)/2], [0., 60., (height-1)/2], [0., 0., 1.]]),
            "T_output_camera": np.eye(4), "reference_frame": "world",
            "image": np.zeros((*depth.shape, 3), np.uint8), "owner_masks": {},
            "robot_mask": np.zeros(depth.shape, bool),
            "self_filter": environment_evidence(depth)}

def environment_evidence(depth):
    from manipulation.robot_self_filter import ENVIRONMENT, AMBIGUOUS_SHADOW
    valid = np.isfinite(depth) & (depth > 0)
    return {"labels": np.where(valid, ENVIRONMENT, AMBIGUOUS_SHADOW).astype(np.uint8),
            "free_ray_depth_m": np.where(valid, depth, 0.),
            "source": "Synthetic calibrated environment-only observation"}

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


def observed_owner_fixture():
    """One labelled front surface and the corresponding native map settings."""
    from curobo.perception import MapperCfg
    record, frames, _ = solid_fixture()
    cfg = MapperCfg(extent_meters_xyz=(1., 1., 1.),
        extent_esdf_meters_xyz=(1., 1., 1.), image_height=96, image_width=96,
        voxel_size=.01, esdf_voxel_size=.01, device='cpu')
    return {'camera': record}, frames, cfg

class OnlineObservationContractTest(unittest.TestCase):
    def test_mapper_intrinsics_preserves_integer_center_rays_and_nearest_depth_reads(self):
        matrix = np.array([[701., 0., 319.5], [0., 533., 239.5], [0., 0., 1.]])
        original = matrix.copy()
        matrix.setflags(write=False)
        native = _mapper_intrinsics(matrix)
        pixels = np.array([[12., 31.], [87., 65.], [318., 239.]])
        external_rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(matrix).T
        native_rays = np.c_[pixels+.5, np.ones(len(pixels))] @ np.linalg.inv(native).T
        np.testing.assert_allclose(native_rays, external_rays, rtol=0., atol=1e-15)
        # Probe both sides of each positive pixel's interior. Casting the
        # converted native projection must select the external nearest centre.
        for displacement in (-.49, 0., .49):
            rays = np.c_[pixels+displacement, np.ones(len(pixels))] @ np.linalg.inv(matrix).T
            projected = rays @ native.T
            selected = (projected[:, :2]/projected[:, 2:]).astype(int)
            np.testing.assert_array_equal(selected, pixels.astype(int))
        native[0, 2] = 0.
        np.testing.assert_array_equal(matrix, original)

    def test_unknown_rays_never_prove_even_the_prefix_free(self):
        from manipulation.robot_self_filter import SELF, AMBIGUOUS_SHADOW
        records = {}
        for name, label in (("self", SELF), ("shadow", AMBIGUOUS_SHADOW)):
            record = view()
            record["self_filter"]["labels"][:] = label
            record["self_filter"]["free_ray_depth_m"][:] = 0.
            records[name] = record
        normalized = _self_filtered_views(records, "world", (.1, 3.))
        options = dict(origin=[0., 0., 1.], shape=(2, 2, 2), voxel_size=.01, depth_error_m=.01)
        rays = {name: row["self_filter"]["free_ray_depth_m"] for name, row in normalized.items()}
        self.assertFalse(_known_free_voxels(normalized, ray_depths=rays, **options).any())
        # A separate clear observation may supply evidence; excluded rays may not.
        normalized.update(_self_filtered_views({"clear": view()}, "world", (.1, 3.)))
        rays["clear"] = normalized["clear"]["self_filter"]["free_ray_depth_m"]
        self.assertTrue(_known_free_voxels(normalized, ray_depths=rays, **options).all())

    def test_partition_clips_both_excluded_states_and_keeps_semantic_conflicts(self):
        from manipulation.robot_self_filter import SELF, AMBIGUOUS_SHADOW
        record = view()
        record["self_filter"]["labels"][30, 30] = SELF
        record["self_filter"]["labels"][31, 31] = AMBIGUOUS_SHADOW
        record["self_filter"]["free_ray_depth_m"][[30, 31], [30, 31]] = 0.
        owner = np.zeros(record["depth"].shape, bool)
        owner[[30, 31, 32], [30, 31, 32]] = True
        record["owner_masks"] = {"part": owner}
        normalized = _self_filtered_views({"camera": record}, "world", (.1, 3.))
        # BEGIN all accepted endpoints keep their supplied owner or background.
        masks = _partition_masks(normalized, ("part",))["camera"]
        self.assertEqual(int(masks["part"].sum()), 1)
        self.assertFalse(masks[None][30, 30])
        self.assertFalse(masks[None][31, 31])
        self.assertTrue(masks[None][33, 33])
        self.assertTrue(masks[None][29, 29])
        self.assertTrue(masks[None][30, 31])
        self.assertTrue(masks[None][34, 34])
        # END SELF/shadow and owner masks never suppress adjacent environment.
        report = _self_filter_report(normalized)["camera"]["owners"]["part"]
        self.assertEqual(report["excluded_self_pixels"], 1)
        self.assertEqual(report["excluded_ambiguous_shadow_pixels"], 1)
        np.testing.assert_array_equal(record["depth"], np.full((96, 96), 2.))
        # A semantic conflict cannot be hidden by subsequent self filtering.
        other = np.zeros_like(owner)
        other[30, 30] = True
        normalized["camera"]["owner_masks"]["other"] = other
        with self.assertRaisesRegex(ValueError, "disjoint"):
            _partition_masks(normalized, ("part", "other"))
        del normalized["camera"]["owner_masks"]["other"]
        normalized["camera"]["owner_masks"]["part"][32, 32] = False
        with self.assertRaisesRegex(ValueError, "after self filtering"):
            _partition_masks(normalized, ("part",))

    def test_unlabelled_adjacent_obstacle_stays_occupied_without_inventing_an_owner(self):
        record = view()
        mask = np.zeros(record['depth'].shape, bool)
        mask[40:43, 40:43] = True
        record['owner_masks'] = {'part': mask}
        original_depth = record['depth'].copy()
        original_rays = record['self_filter']['free_ray_depth_m'].copy()
        original_labels = record['self_filter']['labels'].copy()
        normalized = _self_filtered_views({'camera': record}, 'world', (.1, 3.))
        mapped = _partition_masks(normalized, ('part',))['camera']
        # An unlabelled adjacent endpoint may be leaked object material or a
        # thin independent obstacle. The planner cannot decide that identity
        # from adjacency and must retain the measured obstacle in either case.
        self.assertTrue(mapped[None][43, 43])
        self.assertFalse(mapped['part'][43, 43])
        self.assertTrue(mapped[None][44, 44])
        np.testing.assert_array_equal(mapped['part'], mask)
        np.testing.assert_array_equal(record['depth'], original_depth)
        np.testing.assert_array_equal(record['self_filter']['free_ray_depth_m'], original_rays)
        np.testing.assert_array_equal(record['self_filter']['labels'], original_labels)
        np.testing.assert_array_equal(normalized['camera']['self_filter']['free_ray_depth_m'], original_rays)
        self.assertFalse(mapped[None].flags.writeable)
        np.testing.assert_array_equal(mapped[None] | mapped['part'], np.ones_like(mask))
        self.assertFalse(np.any(mapped[None] & mapped['part']))

    def test_adjacent_owners_keep_identity_and_do_not_erase_other_endpoints(self):
        record = view()
        first, second = np.zeros(record['depth'].shape, bool), np.zeros(record['depth'].shape, bool)
        first[40, 40] = True
        second[40, 41] = True
        record['owner_masks'] = {'first': first, 'second': second}
        normalized = _self_filtered_views({'camera': record}, 'world', (.1, 3.))
        mapped = _partition_masks(normalized, ('first', 'second'))['camera']
        np.testing.assert_array_equal(mapped['first'], first)
        np.testing.assert_array_equal(mapped['second'], second)
        self.assertFalse(mapped[None][40, 40])
        self.assertFalse(mapped[None][40, 41])
        self.assertTrue(mapped[None][39, 40])
        self.assertTrue(mapped[None][39, 41])
        # Integer counts prove both endpoint conservation and non-overlap.
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in mapped.values()),
                                      np.ones_like(first, dtype=int))
        # Conflicting raw semantic evidence still fails before any uncertainty
        # treatment can hide the overlap.
        record['owner_masks']['second'][40, 40] = True
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            _partition_masks(normalized, ('first', 'second'))

    def test_invalid_depth_and_robot_exclusions_do_not_erase_adjacent_environment(self):
        from manipulation.robot_self_filter import SELF, AMBIGUOUS_SHADOW, ENVIRONMENT
        record = view()
        record['depth'][20, 20] = 0.
        record['self_filter'] = environment_evidence(record['depth'])
        record['self_filter']['labels'][30, 30] = SELF
        record['self_filter']['labels'][50, 50] = AMBIGUOUS_SHADOW
        record['self_filter']['free_ray_depth_m'][[30, 50], [30, 50]] = 0.
        owner = np.zeros(record['depth'].shape, bool)
        owner[[20, 30, 40, 50], [20, 30, 40, 50]] = True
        record['owner_masks'] = {'part': owner}
        normalized = _self_filtered_views({'camera': record}, 'world', (.1, 3.))
        mapped = _partition_masks(normalized, ('part',))['camera']
        for row in (20, 30, 50):
            self.assertFalse(mapped[None][row, row])
            self.assertFalse(mapped['part'][row, row])
            self.assertTrue(mapped[None][row+1, row+1])
        self.assertTrue(mapped[None][41, 41])
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in mapped.values()),
            (normalized['camera']['self_filter']['labels'] == ENVIRONMENT).astype(int))

    def test_online_evidence_is_required_and_cannot_grant_excluded_rays(self):
        from manipulation.robot_self_filter import AMBIGUOUS_SHADOW
        record = view()
        del record["self_filter"]
        with self.assertRaisesRegex(ValueError, "three-state self_filter"):
            _self_filtered_views({"camera": record}, "world", (.1, 3.))
        # An old robot_mask remains present and must not enable a fallback.
        record["self_filter"] = environment_evidence(record["depth"])
        record["self_filter"]["labels"][30, 30] = AMBIGUOUS_SHADOW
        with self.assertRaises(ValueError):
            _self_filtered_views({"camera": record}, "world", (.1, 3.))
        record["self_filter"]["free_ray_depth_m"][30, 30] = 0.
        record["self_filter"]["labels"][31, 31] = 255
        with self.assertRaises(ValueError):
            _self_filtered_views({"camera": record}, "world", (.1, 3.))

    def test_mapper_range_clips_evidence_without_mutating_measured_classification(self):
        from manipulation.robot_self_filter import ENVIRONMENT, AMBIGUOUS_SHADOW
        record = view()
        normalized = _self_filtered_views({"camera": record}, "world", (.1, 1.5))["camera"]
        self.assertFalse(normalized["depth"].any())
        self.assertFalse(normalized["self_filter"]["free_ray_depth_m"].any())
        self.assertTrue((normalized["self_filter"]["labels"] == AMBIGUOUS_SHADOW).all())
        self.assertTrue((record["self_filter"]["labels"] == ENVIRONMENT).all())
        self.assertTrue((record["depth"] == 2.).all())

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

    def test_registered_sphere_radius_expands_owner_support_before_mapping(self):
        views, frames, cfg = observed_owner_fixture()
        normalized = _self_filtered_views(views, 'world', (.1, 3.))
        masks = _partition_masks(normalized, frames)
        pose = frames['part']['T_output_owner']
        small=_owner_mapper_config(cfg,normalized,masks,'part',pose,.005,.003)
        supported=_owner_mapper_config(cfg,normalized,masks,'part',pose,.005,.003,query_radius_m=.072)
        np.testing.assert_allclose(np.asarray(supported.extent_esdf_meters_xyz)-small.extent_esdf_meters_xyz,[.144]*3,atol=1e-12)
        np.testing.assert_array_equal(supported.grid_center,small.grid_center)
        self.assertEqual(supported.voxel_size,small.voxel_size)

    # BEGIN the production observation adapter consumes only the native ESDF.
    def test_mapping_freezes_native_esdf_without_building_source_geometry(self):
        from manipulation.planning_observation import _map_online_partition
        record = view(np.ones((3, 3)))
        distances = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)-5.
        voxel = SimpleNamespace(feature_tensor=distances, voxel_size=.01,
                                pose=[.1, .2, .3, 1., 0., 0., 0.])
        mapper = MagicMock()
        mapper.get_stats.return_value = {'allocation_failures': 0, 'num_allocated': 2,
                                        'active_blocks': 1, 'free_count': 1}
        mapper.compute_esdf.return_value = voxel
        # Native Pose invokes a compiled transform kernel; this adapter test
        # isolates geometry freezing and verifies the supplied matrix itself.
        with patch('curobo.perception.Mapper', return_value=mapper), \
             patch('curobo.types.Pose.from_matrix', side_effect=lambda matrix: matrix) as pose:
            field, report = _map_online_partition({'camera': record},
                {'camera': {None: np.ones((3, 3), bool)}}, None,
                SimpleNamespace(device='cpu'), np.eye(4))
        mapper.integrate.assert_called_once()
        np.testing.assert_array_equal(pose.call_args.args[0].numpy(), np.eye(4)[None])
        mapper.compute_esdf.assert_called_once_with()
        mapper.build_tsdf_geometry.assert_not_called()
        np.testing.assert_array_equal(field.distance, distances.numpy())
        np.testing.assert_allclose(field.origin, [.095, .19, .285])
        self.assertEqual(report['shape_xyz'], [2, 3, 4])
    # END sign, axis order and sample-centre placement survive native freezing.


if __name__ == "__main__":
    unittest.main()
