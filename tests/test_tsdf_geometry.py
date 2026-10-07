"""CPU counterexamples for one TSDF zero surface and its observed negative side."""

# BEGIN affine TSDF tests use analytic geometry, without CUDA or simulation.
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import coal
import numpy as np
import torch

from manipulation.collision_records import MAX_ARRAY_BYTES, decode_array
from manipulation.tsdf_geometry import (
    ObservedTsdfGeometry, build_tsdf_geometry, coal_component, tsdf_geometry_from_samples,
)


def inventory(sdf, observed=None, *, origin=(0., 0., 0.), size=1.):
    return tsdf_geometry_from_samples(np.asarray(sdf),
        np.ones(np.shape(sdf), dtype=bool) if observed is None else observed,
        origin=origin, voxel_size=size, name="test_map", source="analytic CPU TSDF")


def geometry(sdf, observed=None, **kwargs):
    return ObservedTsdfGeometry(inventory(sdf, observed, **kwargs)["solids"][0])


def pose_at(point):
    pose = np.eye(4)
    pose[:3, 3] = point
    return pose


def distance(first, first_pose, second, second_pose):
    request = coal.DistanceRequest()
    request.enable_signed_distance = True
    result = coal.DistanceResult()
    coal.distance(first, coal.Transform3s(first_pose[:3, :3], first_pose[:3, 3]),
                  second, coal.Transform3s(second_pose[:3, :3], second_pose[:3, 3]),
                  request, result)
    return float(result.min_distance)


def component_distances(target, actor, actor_pose, *, bounds=None, surface=False, negative=True):
    result = []
    for kind, index, vertices in target.iter_components(bounds, surface=surface, negative=negative):
        shape, component_pose = coal_component(vertices)
        result.append((kind, index, distance(actor, actor_pose, shape, component_pose)))
    return result


class ObservedTsdfGeometryTests(unittest.TestCase):
    def test_source_array_budget_rejects_before_block_export_or_dense_allocation(self):
        count = MAX_ARRAY_BYTES // np.dtype(np.float32).itemsize + 1
        export_blocks = Mock()
        mapper = SimpleNamespace(config=SimpleNamespace(enable_static=False, grid_shape=(1, 1, count)),
                                 tsdf=SimpleNamespace(export_blocks=export_blocks))
        # Broadcast views exercise the same CPU admission without allocating
        # an oversized sample/validity array merely to test its rejection.
        values = np.broadcast_to(np.array(-1., dtype=np.float32), (count, 1, 1))
        observed = np.broadcast_to(np.array(True), values.shape)
        with patch("manipulation.tsdf_geometry.np.zeros") as allocator:
            with self.assertRaisesRegex(ValueError, f"{MAX_ARRAY_BYTES // 1024**2} MiB"):
                build_tsdf_geometry(mapper, "too_large", "explicit oversized sample grid")
            with self.assertRaisesRegex(ValueError, f"{MAX_ARRAY_BYTES // 1024**2} MiB"):
                tsdf_geometry_from_samples(values, observed, origin=(0., 0., 0.),
                    voxel_size=.001, name="too_large", source="broadcast sample grid")
        export_blocks.assert_not_called()
        allocator.assert_not_called()

    def test_tilted_plane_zero_sections_and_negative_pieces_share_one_halfspace(self):
        origin = np.array([.17, -.2, .3])
        size = .02
        normal = np.array([1., 2., 3.])
        points = origin + np.moveaxis(np.indices((3, 3, 3)), 0, -1) * size
        offset = float(normal @ (origin + np.array([.021, .019, .023])))
        target = geometry(points @ normal - offset, origin=origin, size=size)
        surfaces = list(target.iter_components(surface=True, negative=False))
        self.assertTrue(surfaces)
        for _, _, vertices in surfaces:
            np.testing.assert_allclose(vertices @ normal, offset, atol=3e-15, rtol=0)
        for _, _, vertices in target.iter_components(surface=False, negative=True):
            self.assertLessEqual(float(np.max(vertices @ normal - offset)), 3e-15)
        self.assertIsNotNone(target.surface_shape)
        self.assertGreater(target.storage_nbytes, target.surface_vertices.nbytes + target.negative_vertices.nbytes)

    def test_full_negative_cells_stop_at_sample_centres_not_half_voxel_faces(self):
        target = geometry(-np.ones((3, 2, 2)), origin=(4., 5., 6.), size=.1)
        np.testing.assert_allclose(target.bounds, [[4., 5., 6.], [4.2, 5.1, 6.1]], atol=1e-14)
        self.assertEqual(len(target.negative_cell_flat_indices), 2)
        self.assertEqual(target.surface_offsets.tolist(), [0])
        self.assertEqual(target.negative_offsets.tolist(), [0])
        self.assertEqual({kind for kind, _, _ in target.iter_components()}, {"negative_cell"})
        self.assertFalse(list(target.iter_components(([3.96, 5., 6.], [3.99, 5.1, 6.1]))))

    def test_unobserved_values_cannot_fill_a_hole_and_isolated_evidence_stays_a_point(self):
        observed = np.zeros((2, 2, 2), dtype=bool)
        observed[0, 0, 0] = observed[1, 1, 1] = True
        first = np.full((2, 2, 2), np.nan)
        first[0, 0, 0], first[1, 1, 1] = -1., 0.
        second = first.copy()
        second[~observed] = -100000.
        one, two = inventory(first, observed), inventory(second, observed)
        self.assertEqual(one, two)
        target = ObservedTsdfGeometry(one["solids"][0])
        components = list(target.iter_components())
        self.assertEqual([(kind, len(points)) for kind, _, points in components],
                         [("surface", 1), ("negative", 1)])
        self.assertFalse(list(target.iter_components(([.2, .2, .2], [.8, .8, .8]))))

    def test_partial_cell_keeps_only_tetrahedra_whose_four_samples_are_observed(self):
        # This is exactly one supported tetrahedron, 000--100--110--111.
        observed = np.zeros((2, 2, 2), dtype=bool)
        for index in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)):
            observed[index] = True
        target = geometry(-np.ones((2, 2, 2)), observed)
        components = list(target.iter_components(surface=False, negative=True))
        self.assertEqual(len(components), 1)
        vertices = components[0][2]
        self.assertEqual(len(vertices), 4)
        self.assertTrue(np.all(vertices[:, 0] >= vertices[:, 1]))
        self.assertTrue(np.all(vertices[:, 1] >= vertices[:, 2]))
        # A body near 010 has the same cell AABB, but is in the unknown portion.
        actor = coal.Sphere(.02)
        distances = component_distances(target, actor, pose_at([.02, .95, .02]))
        self.assertTrue(all(value > 0 for _, _, value in distances))

    def test_negative_containment_is_detected_despite_positive_surface_distance(self):
        _, _, z = np.indices((2, 2, 2))
        target = geometry(z - .8)
        actor, actor_pose = coal.Sphere(.02), pose_at([.2, .2, .2])
        self.assertGreater(distance(actor, actor_pose, target.surface_shape, np.eye(4)), .5)
        self.assertTrue(any(value < 0 for _, _, value in component_distances(target, actor, actor_pose)))

    def test_thin_negative_fragment_is_not_replaced_by_a_centroid_or_sample_test(self):
        _, _, z = np.indices((2, 2, 2))
        target = geometry(z - .5)
        actor, actor_pose = coal.Box(.02, .02, .02), pose_at([.5, .5, .505])
        lower, upper = np.array([.49, .49, .495]), np.array([.51, .51, .515])
        # The body centre is positive; none of the eight TSDF sample locations
        # is in this body. Its lower 5 mm still intersects observed material.
        self.assertGreater(actor_pose[2, 3] - .5, 0)
        corners = np.moveaxis(np.indices((2, 2, 2)), 0, -1).reshape(-1, 3)
        self.assertFalse(np.any(np.all((corners >= lower) & (corners <= upper), axis=1)))
        values = component_distances(target, actor, actor_pose, bounds=(lower, upper))
        self.assertTrue(any(value < 0 for _, _, value in values))
        # On the positive side there is no half-cell occupancy expansion.
        self.assertFalse(list(target.iter_components(([.49, .49, .51], [.51, .51, .53]),
                                                     surface=False, negative=True)))

    def test_points_segments_and_zero_plateau_keep_their_true_dimension(self):
        actor = coal.Sphere(.1)
        for points, centre, expected in (([[0., 0., 0.]], [.5, 0., 0.], .4),
                                          ([[0., 0., 0.], [1., 0., 0.]], [.5, .3, 0.], .2)):
            shape, component_pose = coal_component(points)
            self.assertAlmostEqual(distance(actor, pose_at(centre), shape, component_pose), expected, places=6)
        target = geometry(np.zeros((2, 2, 2)))
        self.assertFalse(list(target.iter_components(surface=False, negative=True)))
        values = component_distances(target, coal.Sphere(.01), pose_at([.3, .2, .1]),
                                     surface=True, negative=False)
        self.assertTrue(any(value < 0 for _, _, value in values))

    def test_public_block_export_uses_weights_xyz_order_centres_and_grid_clipping(self):
        # Two B=2 blocks span three x samples. The fourth padded x sample is
        # deliberately negative and must not escape the configured map bounds.
        payload = np.zeros((2, 8, 2), dtype=np.float16)
        for block in range(2):
            for z in range(2):
                for y in range(2):
                    for x in range(2):
                        world_x = 9. + 2 * block + x
                        value = world_x - 10.25 if world_x <= 11 else -100.
                        payload[block, z * 4 + y * 2 + x] = [2 * value, 2.]
        config = SimpleNamespace(enable_static=False, grid_shape=(2, 2, 3), voxel_size=1.,
                                 grid_center=torch.tensor([10., 20., 30.]), minimum_tsdf_weight=.1)
        exported = {"active_block_coords": torch.tensor([[-1, 0, 0], [0, 0, 0]]),
                    "block_data": torch.from_numpy(payload)}
        mapper = SimpleNamespace(config=config, tsdf=SimpleNamespace(block_size=2, export_blocks=lambda: exported))
        result = build_tsdf_geometry(mapper, "weighted", "explicit weighted samples")
        record = result["solids"][0]
        np.testing.assert_array_equal(record["origin"], [9., 19.5, 29.5])
        np.testing.assert_allclose(decode_array(record["surface_vertices"])[:, 0], 10.25, atol=1e-14)
        self.assertEqual(result["field_summary"]["observed_samples"], 12)
        target = ObservedTsdfGeometry(record)
        self.assertLessEqual(target.bounds[1][0], 10.25)
        np.testing.assert_array_equal(exported["block_data"].numpy(), payload)


if __name__ == "__main__":
    unittest.main()
# END negative material and zero distance are checked independently of ESDF seeds.
