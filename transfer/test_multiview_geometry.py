"""Small CPU regressions for exact observed identities and surface association.

The fixtures construct only calibrated RGB-D arrays. No image model, renderer,
ground-truth object geometry, GPU allocation, or repository dataset is needed.
"""

import unittest
from unittest import mock

import numpy as np

from transfer.core import TransferConfig, _pixel_components
from transfer.multiview_geometry import (associate_view, build_candidate_bank,
                                         _association_planes)


# BEGIN independent multiview geometry regressions.
def _target(view_id, depth=None, *, mask=None, K=None, T=None):
    """Backproject exact pixel identities using the production camera convention."""
    depth = np.ones((9, 9), dtype=float) if depth is None else np.asarray(depth)
    height, width = depth.shape
    mask = np.ones(depth.shape, bool) if mask is None else np.asarray(mask)
    K = (np.array([[1000., 0., (width - 1) / 2],
                   [0., 1000., (height - 1) / 2], [0., 0., 1.]])
         if K is None else np.asarray(K))
    T = np.eye(4) if T is None else np.asarray(T)
    y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    pixels = np.column_stack((x, y)).astype(np.int64)
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    xyz = (rays * depth[y, x, None]) @ T[:3, :3].T + T[:3, 3]
    return {"id": view_id, "observation": {"xyz": xyz, "pixels": pixels,
            "K": K, "T_world_camera_cv": T, "object_mask": mask.copy()},
            "depth_m": depth, "selected_mask": mask.copy(),
            "crop_xyxy": np.array([0, 0, width, height])}


def _point_target(view_id, xyz):
    """Observe a chosen world point at the center of a different calibrated view."""
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(xyz) - [0., 0., 1.]
    target = _target(view_id, np.ones((1, 1)), T=transform)
    # Preserve the supplied observed world coordinate exactly: no arithmetic
    # rounding is allowed to merge or move candidates in the test bank.
    target["observation"]["xyz"] = np.asarray(xyz, dtype=float)[None].copy()
    return target


def _foreign_candidate(target, xyz, *, radius=.03):
    other = _point_target("z_foreign", xyz)
    bank = build_candidate_bank([other, target])
    candidate = int(bank["observation_candidate_index"][-1])
    association = associate_view(bank, target, sampling_radius_m=radius)
    return bank, candidate, association, other


def _state(association, index):
    return association["state_names"][int(association["state"][index])]


class TestCandidateBank(unittest.TestCase):
    def test_canonical_view_order_exact_union_and_all_provenance(self):
        a = _target("a", np.array([[1., 1.1]]))
        b = _target("b", np.array([[1., 1.2]]))
        first = build_candidate_bank([b, a])
        second = build_candidate_bank([a, b])
        for key in first:
            if key == "view_ids":
                self.assertEqual(first[key], ["a", "b"])
            else:
                np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(len(first["xyz"]), 3)
        np.testing.assert_array_equal(first["view_offsets"], [0, 2, 4])
        np.testing.assert_array_equal(first["observation_view_index"], [0, 0, 1, 1])
        np.testing.assert_array_equal(first["observation_local_index"], [0, 1, 0, 1])
        np.testing.assert_array_equal(first["xyz"][first["observation_candidate_index"]],
                                      first["observation_xyz"])
        np.testing.assert_array_equal(first["observation_xyz"][first["candidate_first_observation_index"]],
                                      first["xyz"])
        shared = first["observation_candidate_index"][0]
        self.assertEqual(shared, first["observation_candidate_index"][2])
        for target in (a, b):
            association = associate_view(first, target)
            self.assertTrue(association["owner"][shared])
            self.assertTrue(association["supported"][shared])

    def test_no_rounding_or_same_pixel_cross_camera_identity(self):
        a = _point_target("a", [0., 0., 1.])
        b = _point_target("b", [0., 0., np.nextafter(1., 2.)])
        bank = build_candidate_bank([a, b])
        self.assertEqual(len(bank["xyz"]), 2)
        np.testing.assert_array_equal(a["observation"]["pixels"], b["observation"]["pixels"])
        np.testing.assert_array_equal(bank["observation_view_index"], [0, 1])
        self.assertNotEqual(bank["observation_candidate_index"][0],
                            bank["observation_candidate_index"][1])

    def test_empty_observations_and_invalid_identity_contract(self):
        a = _target("a", np.full((2, 2), np.nan))
        b = _target("b", np.full((2, 2), np.nan))
        bank = build_candidate_bank([a, b])
        self.assertEqual(bank["xyz"].shape, (0, 3))
        result = associate_view(bank, a)
        self.assertEqual(result["pixels"].shape, (0, 2))
        self.assertEqual(result["diagnostics"]["candidate_count"], 0)
        for targets in ([a], [a, a], [a, b, a]):
            with self.assertRaises(ValueError):
                build_candidate_bank(targets)
        malformed = _target("c")
        malformed["observation"]["pixels"][1] = malformed["observation"]["pixels"][0]
        with self.assertRaises(ValueError):
            build_candidate_bank([a, malformed])


class TestViewAssociation(unittest.TestCase):
    def test_owner_nonidentity_calibration_roundtrip_and_no_plane_requirement(self):
        K = np.array([[700., 13., 3.25], [0., 450., 4.75], [0., 0., 1.]])
        T = np.array([[0., -1., 0., .3], [1., 0., 0., -.2],
                      [0., 0., 1., .5], [0., 0., 0., 1.]])
        # A single measured point cannot define a plane, but remains directly
        # supported in its owner camera, including at the image border.
        mask = np.zeros((9, 9), bool)
        mask[0, 0] = True
        target = _target("a", mask=mask, K=K, T=T)
        other = _point_target("b", [4., 2., 1.])
        bank = build_candidate_bank([target, other])
        index = int(bank["observation_candidate_index"][0])
        with mock.patch("transfer.multiview_geometry._association_planes", wraps=_association_planes) as planes:
            result = associate_view(bank, target)
        self.assertTrue(result["owner"][index])
        self.assertTrue(result["supported"][index])
        self.assertEqual(_state(result, index), "owner_observed")
        np.testing.assert_array_equal(result["pixels"][index], [0., 0.])
        self.assertEqual(planes.call_count, 0)

    def test_slanted_plane_uses_plane_distance_not_nearest_optical_depth(self):
        base = _target("a")
        K = base["observation"]["K"]
        normal = np.array([1., 0., .055])
        normal /= np.linalg.norm(normal)
        yy, xx = np.indices((9, 9))
        rays = np.stack(((xx - 4) / 1000, (yy - 4) / 1000, np.ones((9, 9))), axis=-1)
        depth = normal[2] / (rays @ normal)
        target = _target("a", depth, K=K)
        pixel = np.array([4.4, 4.3])
        ray = np.linalg.inv(K) @ np.r_[pixel, 1.]
        q = ray * (normal[2] / (ray @ normal))
        bank, index, result, _ = _foreign_candidate(target, q)
        self.assertFalse(result["owner"][index])
        self.assertTrue(result["supported"][index])
        self.assertEqual(_state(result, index), "co_observed")
        np.testing.assert_allclose(result["pixels"][index], pixel, atol=1e-12, rtol=0)
        points, _, spacing, _, lookup = _association_planes(
            depth, target["selected_mask"], K, np.eye(4), .03, TransferConfig())
        nearest = lookup[4, 4]
        h = TransferConfig().surface_height_spacing_factor * spacing[nearest]
        self.assertGreater(abs(q[2] - points[nearest, 2]), h)
        np.testing.assert_array_equal(bank["xyz"][index], q)

    def test_all_four_corners_require_depth_and_sam_support(self):
        q = np.array([.0004, .0004, 1.])  # projection (4.4, 4.4), nearest (4,4)
        for defect, expected in (("depth", "invalid_depth"),
                                 ("sam", "outside_selected_mask")):
            with self.subTest(defect=defect):
                depth = np.ones((9, 9))
                mask = np.ones((9, 9), bool)
                if defect == "depth":
                    depth[5, 5] = np.nan
                else:
                    mask[5, 5] = False
                target = _target("a", depth, mask=mask)
                bank, index, result, other = _foreign_candidate(target, q)
                self.assertFalse(result["supported"][index])
                self.assertEqual(_state(result, index), expected)
                self.assertTrue(associate_view(bank, other)["supported"][index])

    def test_depth_edge_cannot_extend_nearest_pixel_plane(self):
        depth = np.ones((9, 9))
        depth[:, 5:] = 1.1
        target = _target("a", depth)
        q = np.array([.0004, .0004, 1.])
        _, index, result, _ = _foreign_candidate(target, q)
        self.assertFalse(result["supported"][index])
        self.assertEqual(_state(result, index), "corner_surface_mismatch")

    def test_front_and_occluded_surface_conflicts_never_remove_owner(self):
        target = _target("a")
        for depth, expected in ((.99, "front_of_observed_surface"),
                                 (1.01, "behind_observed_surface")):
            with self.subTest(depth=depth):
                bank, index, result, other = _foreign_candidate(target, [0., 0., depth])
                self.assertFalse(result["supported"][index])
                self.assertEqual(_state(result, index), expected)
                self.assertFalse(result["diagnostics"]["hard_free_space_veto"])
                self.assertTrue(associate_view(bank, other)["supported"][index])

    def test_subresolution_offset_is_explicitly_associated_and_radius_is_bounded(self):
        target = _target("a")
        _, index, result, _ = _foreign_candidate(target, [0., 0., 1.001])
        self.assertEqual(_state(result, index), "co_observed")
        _, index, result, _ = _foreign_candidate(target, [0., 0., 1.04])
        self.assertEqual(_state(result, index), "outside_local_radius")

    def test_outside_and_behind_camera_are_neutral(self):
        target = _target("a")
        for xyz, expected in (([1., 0., 1.], "outside_image"),
                               ([0., 0., -1.], "behind_camera"),
                               ([0., 0., 0.], "invalid_projection")):
            with self.subTest(xyz=xyz):
                bank, index, result, other = _foreign_candidate(target, xyz)
                self.assertFalse(result["supported"][index])
                self.assertEqual(_state(result, index), expected)
                self.assertTrue(associate_view(bank, other)["supported"][index])

    def test_full_depth_neighbors_survive_sam_computation_crop(self):
        mask = np.zeros((9, 9), bool)
        mask[4:6, 4:6] = True
        target = _target("a", mask=mask)
        _, index, result, _ = _foreign_candidate(target, [.0004, .0004, 1.])
        self.assertEqual(_state(result, index), "co_observed")
        # Four SAM points alone cannot meet the inherited six-point support
        # gate. Passing proves surrounding non-SAM depth remains available.
        self.assertGreater(result["diagnostics"]["plane_computation_point_count"], int(mask.sum()))

    def test_collinear_local_observation_is_unknown(self):
        depth = np.full((9, 9), np.nan)
        depth[4] = 1.
        target = _target("a", depth)
        _, index, result, _ = _foreign_candidate(target, [.0004, 0., 1.])
        self.assertFalse(result["supported"][index])
        self.assertEqual(_state(result, index), "invalid_local_plane")

    def test_observation_gap_preserves_native_per_view_connectivity(self):
        mask = np.ones((9, 9), bool)
        mask[4] = False
        target = _target("a", mask=mask)
        other = _target("b", mask=mask)
        other["observation"]["xyz"] = other["observation"]["xyz"] + [1., 0., 0.]
        bank = build_candidate_bank([other, target])
        for view_index, view in enumerate((target, other)):
            ids = bank["observation_view_index"] == view_index
            np.testing.assert_array_equal(bank["observation_local_index"][ids],
                                          np.arange(len(view["observation"]["xyz"])))
            components = _pixel_components(view["observation"]["pixels"])
            self.assertEqual(len(np.unique(components)), 2)
        self.assertEqual(len(bank["xyz"]), 2 * len(target["observation"]["xyz"]))


if __name__ == "__main__":
    unittest.main()
# END independent multiview geometry regressions.
