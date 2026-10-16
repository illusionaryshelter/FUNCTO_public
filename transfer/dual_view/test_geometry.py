"""Small CPU-only tests for frozen-query and actual-region geometry contracts."""

import unittest
from copy import deepcopy

import numpy as np
from scipy.sparse import csr_matrix

from .geometry import (build_target_surface, consensus_region, geometry_support,
                       prepare_source, project_oriented, region_for_transform)


def _observation(depth, entity=None):
    """A synthetic front-facing plane using exact binary pixel spacing."""
    y, x = np.nonzero(depth > 0)
    pixels = np.column_stack((x, y))
    k = np.array([[256., 0., 6.], [0., 256., 6.], [0., 0., 1.]])
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T
    xyz = rays * depth[y, x, None]
    labels = np.ones(len(xyz), bool) if entity is None else entity[y, x]
    return dict(xyz=xyz, pixels=pixels, entity_member=labels,
                K=k, T_world_camera_cv=np.eye(4))


def _surface(depth_a=None, depth_b=None):
    depth_a = np.ones((13, 13)) if depth_a is None else depth_a
    depth_b = depth_a.copy() if depth_b is None else depth_b
    views = []
    for name, depth in zip(("A", "B"), (depth_a, depth_b)):
        obs = _observation(depth)
        views.append(dict(id=name, depth_m=depth, K=obs["K"],
            T_world_camera_cv=obs["T_world_camera_cv"], observation=obs))
    return build_target_surface(views)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.surface = _surface()
        self.point = np.array([[0., 0., 1.]])
        self.normal = np.array([[0., 0., -1.]])

    def test_owner_still_needs_valid_plane_and_source_normal(self):
        supported = project_oriented(self.surface, self.point, self.normal, 0)
        self.assertTrue(supported["supported"][0])
        invalid = deepcopy(self.surface)
        invalid["observation_valid"][supported["observation_indices"][0]] = False
        projected = project_oriented(invalid, self.point, self.normal, 0)
        self.assertTrue(projected["position_supported"][0])
        self.assertFalse(projected["supported"][0])
        projected = project_oriented(self.surface, self.point,
                                      np.full((1, 3), np.nan), 0)
        self.assertTrue(projected["position_supported"][0])
        self.assertFalse(projected["supported"][0])
        self.assertTrue(projected["unknown"][0])

    def test_reverse_normal_is_unknown_without_hard_conflict(self):
        result = project_oriented(self.surface, self.point, -self.normal, 0)
        self.assertTrue(result["position_supported"][0])
        self.assertFalse(result["oriented_supported"][0])
        self.assertFalse(result["conflict"][0])
        self.assertTrue(result["unknown"][0])

    def test_all_positive_original_pixel_corners_need_entity_labels(self):
        corner = self.surface["pixel_to_observation"][0][7, 7]
        self.surface["observation_entity_member"][corner] = False
        between = np.array([[.5 / 256, .5 / 256, 1.]])
        result = project_oriented(self.surface, between, self.normal, 0)
        self.assertTrue(result["position_supported"][0])
        self.assertFalse(result["entity_supported"][0])
        self.assertFalse(result["supported"][0])
        # At the integer pixel the missing diagonal contributes zero weight.
        exact = project_oriented(self.surface, self.point, self.normal, 0)
        self.assertTrue(exact["supported"][0])

    def test_conflict_survives_missing_normal_entity_and_other_view_support(self):
        surface = _surface(depth_b=np.full((13, 13), 1.04))
        result = geometry_support(surface, self.point, self.normal)
        self.assertTrue(result["support"][0])
        self.assertTrue(result["conflict"][0])
        self.assertEqual(result["view_support"].shape, (1, 2))
        surface["observation_entity_member"][:] = False
        result = geometry_support(surface, self.point, np.full((1, 3), np.nan))
        self.assertFalse(result["support"][0])
        self.assertTrue(result["conflict"][0])
        self.assertFalse(result["unknown"][0])


class SourceTests(unittest.TestCase):
    def test_complete_depth_normals_and_frozen_missing_query(self):
        depth = np.ones((13, 13))
        depth[:3, :3] = 0
        depth[0, 0] = 1
        source = _observation(depth)
        xy = source["pixels"]
        source["region_mask"] = ((xy[:, 0] >= 4) & (xy[:, 0] <= 8)
                                  & (xy[:, 1] >= 4) & (xy[:, 1] <= 8))
        regional = np.flatnonzero(source["region_mask"])
        source["entity_member"][regional[0]] = False
        query_ids = np.r_[regional[:6], 0]
        result = prepare_source(source, depth, query_indices=query_ids,
            query_region=source["region_mask"][query_ids])
        self.assertEqual(result["reasons"], [])
        np.testing.assert_array_equal(result["query_indices"], query_ids)
        self.assertEqual(result["diagnostics"]["fixed_query_denominator"], 7)
        self.assertFalse(result["query_normal_valid"][-1])
        self.assertTrue(np.isnan(result["query_normals"][-1]).all())
        self.assertTrue(result["query_normal_valid"][:-1].all())
        self.assertFalse(np.isin(result["check_indices"], query_ids).any())
        self.assertLessEqual(result["diagnostics"]["check_region_count"], 32)
        self.assertLessEqual(result["diagnostics"]["check_context_count"], 32)
        np.testing.assert_allclose(result["normal"], [0, 0, -1])


class RegionTests(unittest.TestCase):
    def setUp(self):
        self.surface = _surface()
        self.source_pack = dict(center=np.array([0., 0., 1.]), normal=np.array([0., 0., -1.]))
        self.transform = (1., np.eye(3), np.zeros(3))

    def test_nearest_actual_seed_and_supported_center(self):
        mode, reason = region_for_transform(self.surface, self.source_pack, self.transform)
        self.assertIsNone(reason)
        np.testing.assert_array_equal(mode["center"], self.source_pack["center"])
        self.assertGreaterEqual(mode["diagnostics"]["neighborhood_point_count"], 6)
        self.assertEqual(mode["diagnostics"]["neighborhood_rank"], 2)
        # A center outside the entity cannot be snapped to neighboring labels.
        for lookup in self.surface["pixel_to_observation"]:
            self.surface["observation_entity_member"][lookup[6, 6]] = False
        mode, reason = region_for_transform(self.surface, self.source_pack, self.transform)
        self.assertIsNone(mode)
        self.assertEqual(reason, "mapped_center_without_oriented_entity_support")

    def test_exact_nearest_tie_requires_connected_fixed_plane(self):
        self.source_pack["center"] = np.array([.5 / 256, 0., 1.])
        mode, reason = region_for_transform(self.surface, self.source_pack, self.transform)
        self.assertIsNone(reason)
        self.assertEqual(mode["diagnostics"]["nearest_seed_exact_tie_count"], 2)
        n = len(self.surface["xyz"])
        self.surface["adjacency"] = csr_matrix((n, n), dtype=bool)
        mode, reason = region_for_transform(self.surface, self.source_pack, self.transform)
        self.assertIsNone(mode)
        self.assertEqual(reason, "center_projection_ambiguous")

    def test_consensus_consumes_generator_and_never_expands_intersection(self):
        xyz = self.surface["xyz"]
        left = xyz[:, 0] <= 2 / 256
        right = xyz[:, 0] >= -2 / 256
        expected = left & right
        modes = (dict(mask=mask) for mask in (left, right))
        mode, reason = consensus_region(self.surface, modes,
            np.array([[-2 / 256, 0, 1], [2 / 256, 0, 1]]),
            np.array([[0, 0, -1], [0, 0, -1]]))
        self.assertIsNone(reason)
        np.testing.assert_array_equal(mode["common_mask"], expected)
        self.assertFalse(np.any(mode["mask"] & ~expected))
        self.assertTrue(expected[mode["sample_indices"]].all())
        self.assertEqual(mode["diagnostics"]["common_hypothesis_count"], 2)

    def test_consensus_retains_unselected_components(self):
        xyz = self.surface["xyz"]
        common = np.abs(xyz[:, 0]) >= 2 / 256
        mode, reason = consensus_region(self.surface, [dict(mask=common)],
            np.array([[0, 0, 1]]), np.array([[0, 0, -1]]))
        self.assertIsNone(reason)
        self.assertEqual(mode["diagnostics"]["common_component_count"], 2)
        self.assertLess(mode["mask"].sum(), mode["common_mask"].sum())

    def test_nonempty_but_collinear_intersection_is_insufficient(self):
        common = self.surface["xyz"][:, 1] == 0
        mode, reason = consensus_region(self.surface, [dict(mask=common)],
            np.array([[0, 0, 1]]), np.array([[0, 0, -1]]))
        self.assertIsNone(mode)
        self.assertEqual(reason, "common_region_not_sampleable")


if __name__ == "__main__":
    unittest.main()
