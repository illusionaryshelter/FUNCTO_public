"""CPU checks for observed two-surface frames, independent of visual models.

Analytic perspective depth supplies known physical planes and one rigid motion.
The successful dense case uses the real plane extractor. Other cases inject
only coarse candidate masks: depth, erosion, metric refitting and frame
construction still execute. These checks do not certify VLM surface semantics,
material identity, collision clearance or physical grasp stability.
"""

# BEGIN independently specified observed-surface geometry checks.
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.surface_geometry import observe_orthogonal_surface_frame


EXTRACTOR = "manipulation.surface_geometry._plane_candidates"
SIDE_SOURCE = "Synthetic opaque convex exterior faces visible to the calibrated camera"


def ray_point(depth, K, pixel):
    """Independent analytic pinhole convention for the expected anchor point."""
    u, v = np.asarray(pixel, float)
    z = depth[int(np.rint(v)), int(np.rint(u))]
    return np.linalg.solve(K, [u, v, 1.0]) * z


def observed_corner(angle_degrees=90.0, rigid=None, K=None):
    """Ray-cast a convex intersection of two exterior planar half-spaces.

    The corner is defined in Euclidean coordinates, not by the frame function
    being tested. Both planes are visible. Their intersection is deeper than
    the camera and each ray enters the solid at the farther half-space plane.
    """
    if K is None:
        K = np.array([[140.0, 0.0, 47.5], [0.0, 140.0, 47.5], [0.0, 0.0, 1.0]])
    else:
        K = np.asarray(K, float).copy()
    half_angle = np.deg2rad(angle_degrees / 2)
    sine, cosine = np.sin(half_angle), np.cos(half_angle)
    normals = np.array([[sine, 0.0, -cosine], [-sine, 0.0, -cosine]])
    corner = np.array([0.0, 0.0, 1.0])
    if rigid is not None:
        normals = normals @ rigid[:3, :3].T
        corner = rigid[:3, :3] @ corner + rigid[:3, 3]
    rows, columns = np.mgrid[:96, :96]
    rays = np.stack((columns, rows, np.ones_like(rows)), axis=-1) @ np.linalg.inv(K).T
    distances = (normals @ corner) / (rays @ normals.T)
    if not np.all(distances > 0):
        raise AssertionError("Synthetic camera must view both faces from their exterior")
    face_index = np.argmax(distances, axis=-1)
    owner = np.zeros((96, 96), bool)
    owner[8:88, 8:88] = True
    depth = np.where(owner, np.max(distances, axis=-1), 0.0)
    masks = [owner & (face_index == index) for index in range(2)]
    return {
        "depth": depth, "K": K, "owner": owner, "masks": masks,
        "normals": normals, "corner": corner,
        "axis_pixel": np.array([65.0, 49.0]),
        "anchor_pixel": np.array([30.0, 49.0]),
    }


def coarse_candidates(data, masks=None):
    """Supply coarse membership without doing the tested SVD normal estimate."""
    masks = data["masks"] if masks is None else masks
    result = []
    for index, mask in enumerate(masks):
        rows, columns = np.nonzero(mask)
        result.append({
            "candidate_id": 7 + 5 * index, "mask": mask.copy(),
            "origin": data["corner"].copy(),
            "normal": data["normals"][index % 2].copy(),
            # This arbitrary tangent has no role in the intended observation.
            "first_axis": np.array([0.0, 1.0, 0.0]),
            "bbox": [int(columns.min()), int(rows.min()),
                     int(columns.max() + 1), int(rows.max() + 1)],
            "plane_rms_m": 0.0, "sample_inlier_count": int(mask.sum()),
            "observed_pixel_count": int(mask.sum()),
        })
    return result


def observe(data, **overrides):
    options = {
        "depth": data["depth"], "K": data["K"], "owner_mask": data["owner"],
        "axis_pixel": data["axis_pixel"], "anchor_pixel": data["anchor_pixel"],
        "surface_side_source": SIDE_SOURCE,
    }
    options.update(overrides)
    return observe_orthogonal_surface_frame(**options)


class SurfaceGeometryTest(unittest.TestCase):
    def test_real_dense_plane_extraction_recovers_analytic_surface_directions(self):
        data = observed_corner()
        result = observe(data)
        frame = result["frame"]
        np.testing.assert_allclose(frame[:3, 0], data["normals"][0], atol=1e-9, rtol=0)
        np.testing.assert_allclose(-frame[:3, 1], data["normals"][1], atol=1e-9, rtol=0)
        np.testing.assert_allclose(frame[:3, 2], [0, -1, 0], atol=1e-9, rtol=0)
        np.testing.assert_allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(frame[:3, :3]), 1.0, places=12)
        np.testing.assert_allclose(frame[:3, 3], ray_point(data["depth"], data["K"], data["anchor_pixel"]),
                                   atol=1e-12, rtol=0)
        self.assertGreaterEqual(result["diagnostics"]["candidate_count"], 2)
        for role, index in (("axis", 0), ("anchor", 1)):
            plane = result["planes"][role]
            np.testing.assert_allclose(plane["normal"], data["normals"][index], atol=1e-9)
            self.assertGreater(plane["core_pixel_count"], 100)
            self.assertLess(plane["plane_rms_m"], 1e-10)
            self.assertFalse(np.any(plane["core_mask"] & ~data["owner"]))

    def test_known_rigid_motion_is_recovered_without_world_axes_or_pca_tangents(self):
        source = observed_corner()
        motion = np.eye(4)
        motion[:3, :3] = Rotation.from_rotvec([0.11, -0.08, 0.17]).as_matrix()
        motion[:3, 3] = [0.06, -0.02, 0.12]
        original_anchor = ray_point(source["depth"], source["K"], source["anchor_pixel"])
        moved_anchor = motion[:3, :3] @ original_anchor + motion[:3, 3]
        target_K = source["K"].copy()
        target_pixel = np.array([32.0, 49.0])
        # Choose calibration so this independently transformed material sample
        # is exactly a pixel center; no rounding error can masquerade as motion.
        target_K[:2, 2] = target_pixel - np.diag(target_K)[:2] * moved_anchor[:2] / moved_anchor[2]
        target = observed_corner(rigid=motion, K=target_K)
        target["anchor_pixel"] = target_pixel
        original_axis = ray_point(source["depth"], source["K"], source["axis_pixel"])
        moved_axis = motion[:3, :3] @ original_axis + motion[:3, 3]
        projected = target_K @ moved_axis
        target["axis_pixel"] = np.rint(projected[:2] / projected[2])
        with patch(EXTRACTOR, return_value=(coarse_candidates(source), 6400)):
            first = observe(source)["frame"]
        with patch(EXTRACTOR, return_value=(coarse_candidates(target), 6400)):
            second = observe(target)["frame"]
        np.testing.assert_allclose(second, motion @ first, atol=1e-10, rtol=0)
        mapping = second @ np.linalg.inv(first)
        np.testing.assert_allclose(mapping, motion, atol=1e-10, rtol=0)
        # This independently placed tool pose checks the downstream mapping,
        # including the lever arm; comparing orientations alone would miss it.
        tool = np.eye(4)
        tool[:3, :3] = Rotation.from_rotvec([-0.23, 0.18, 0.09]).as_matrix()
        tool[:3, 3] = [-0.08, 0.04, 0.92]
        np.testing.assert_allclose(mapping @ tool, motion @ tool, atol=1e-10, rtol=0)

    def test_anchor_uses_its_original_depth_even_when_it_is_off_the_fitted_plane(self):
        data = observed_corner()
        u, v = data["anchor_pixel"].astype(int)
        data["depth"][v, u] += 0.0013
        expected = ray_point(data["depth"], data["K"], data["anchor_pixel"])
        with patch(EXTRACTOR, return_value=(coarse_candidates(data), 6400)):
            result = observe(data)
        # Independent solve(K, ray) and inverse-matrix multiplication differ
        # at floating-point roundoff, far below the deliberate 1.3 mm offset.
        np.testing.assert_allclose(result["frame"][:3, 3], expected, atol=1e-12, rtol=0)
        plane = result["planes"]["anchor"]
        self.assertGreater(abs((expected - plane["origin"]) @ plane["normal"]), 0.0005)
        self.assertGreater(np.linalg.norm(expected - plane["origin"]), 0.02)

    def test_candidate_order_unsigned_normals_and_pca_tangents_do_not_set_roles(self):
        data = observed_corner()
        candidates = coarse_candidates(data)
        with patch(EXTRACTOR, return_value=(candidates, 6400)):
            expected = observe(data)
        changed = deepcopy(candidates[::-1])
        for candidate in changed:
            candidate["normal"] *= -1
            candidate["first_axis"] = np.array([0.9, -0.1, 0.7])
        with patch(EXTRACTOR, return_value=(changed, 6400)):
            actual = observe(data)
        np.testing.assert_array_equal(actual["frame"], expected["frame"])
        self.assertEqual(actual["planes"]["axis"]["candidate_id"], 7)
        self.assertEqual(actual["planes"]["anchor"]["candidate_id"], 12)

    def test_explicit_role_swap_is_respected_rather_than_guessed_from_image_position(self):
        data = observed_corner()
        with patch(EXTRACTOR, return_value=(coarse_candidates(data), 6400)):
            result = observe(data, axis_pixel=data["anchor_pixel"], anchor_pixel=data["axis_pixel"])
        np.testing.assert_allclose(result["frame"][:3, 0], data["normals"][1], atol=1e-10)
        np.testing.assert_allclose(result["frame"][:3, 3], ray_point(data["depth"], data["K"], data["axis_pixel"]),
                                   atol=1e-12, rtol=0)
        # Geometry cannot recognize a linguistic top/side swap. It must not
        # silently reorder valid roles to make a familiar-looking tool frame.
        self.assertEqual(result["planes"]["axis"]["candidate_id"], 12)

    def test_small_nonorthogonality_is_retained_in_planes_and_explicitly_reported(self):
        data = observed_corner(angle_degrees=88)
        with patch(EXTRACTOR, return_value=(coarse_candidates(data), 6400)):
            result = observe(data)
        self.assertAlmostEqual(result["diagnostics"]["normal_angle_rad"], np.deg2rad(88), places=11)
        self.assertAlmostEqual(result["diagnostics"]["orthogonality_error_rad"], np.deg2rad(2), places=11)
        self.assertAlmostEqual(result["diagnostics"]["orthogonalization_angle_rad"], np.deg2rad(2), places=11)
        np.testing.assert_allclose(result["planes"]["anchor"]["normal"], data["normals"][1], atol=1e-11)
        self.assertGreater(np.linalg.norm(result["planes"]["anchor"]["normal"] + result["frame"][:3, 1]), 0.03)

    def test_nonorthogonal_surfaces_are_rejected_without_silently_repairing_the_angle(self):
        data = observed_corner(angle_degrees=70)
        with patch(EXTRACTOR, return_value=(coarse_candidates(data), 6400)):
            with self.assertRaises(ValueError):
                observe(data)

    def test_same_missing_or_ambiguous_surface_membership_is_rejected(self):
        data = observed_corner()
        candidates = coarse_candidates(data)
        duplicate = deepcopy(candidates[0])
        duplicate["candidate_id"] = 33
        cases = (
            (candidates, {"anchor_pixel": data["axis_pixel"]}),
            ([], {}),
            (candidates[:1], {}),
            (candidates + [duplicate], {}),
        )
        for proposal, overrides in cases:
            with self.subTest(candidate_count=len(proposal), overrides=overrides):
                with patch(EXTRACTOR, return_value=(proposal, 6400)):
                    with self.assertRaises(ValueError):
                        observe(data, **overrides)

    def test_query_on_coarse_boundary_cannot_bypass_interior_evidence(self):
        data = observed_corner()
        with patch(EXTRACTOR, return_value=(coarse_candidates(data), 6400)):
            with self.assertRaises(ValueError):
                observe(data, axis_pixel=[87, 49])

    def test_overlap_with_unselected_candidate_is_removed_before_support_count(self):
        data = observed_corner()
        candidates = coarse_candidates(data)
        u, v = data["axis_pixel"].astype(int)
        overlapping = candidates[0]["mask"].copy()
        overlapping[v-2:v+3, u-2:u+3] = False
        extra = coarse_candidates(data, [overlapping])[0]
        extra["candidate_id"] = 44
        # The axis query belongs uniquely to candidate 7, but only a 5x5
        # exclusive patch remains. Ignoring the other plane would falsely pass.
        with patch(EXTRACTOR, return_value=(candidates + [extra], 6400)):
            with self.assertRaises(ValueError):
                observe(data)

    def test_collinear_core_is_rejected_even_when_it_has_many_pixels(self):
        data = observed_corner()
        u = int(data["axis_pixel"][0])
        strip = np.zeros_like(data["owner"])
        strip[10:86, u-1:u+2] = True
        candidates = coarse_candidates(data, [strip, data["masks"][1]])
        with patch(EXTRACTOR, return_value=(candidates, 6400)):
            with self.assertRaises(ValueError):
                observe(data)

    def test_near_grazing_observed_face_cannot_supply_a_reliable_outward_sign(self):
        data = observed_corner()
        data["depth"][:] = 0
        data["owner"][:] = False
        masks = [np.zeros_like(data["owner"]) for _ in range(2)]
        masks[0][20:76, 45:52] = True
        masks[1][20:76, 16:37] = True
        normals = np.array([[1.0, 0.0, -0.035], [-0.035, 0.0, -1.0]])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        rows, columns = np.mgrid[:96, :96]
        rays = np.stack((columns, rows, np.ones_like(rows)), axis=-1) @ np.linalg.inv(data["K"]).T
        for index, mask in enumerate(masks):
            offset = -0.005 if index == 0 else -1.0
            data["depth"][mask] = offset / (rays[mask] @ normals[index])
            data["owner"] |= mask
        data.update(masks=masks, normals=normals, axis_pixel=np.array([48, 49]), anchor_pixel=np.array([26, 49]))
        with patch(EXTRACTOR, return_value=(coarse_candidates(data), int(data["owner"].sum()))):
            with self.assertRaisesRegex(ValueError, "near-grazing"):
                observe(data)

    def test_nonplanar_depth_cannot_pass_merely_because_candidate_mask_was_supplied(self):
        data = observed_corner()
        rows, _ = np.mgrid[:96, :96]
        data["depth"][data["masks"][0]] += (0.04 * np.sin(rows / 3.0))[data["masks"][0]]
        with patch(EXTRACTOR, return_value=(coarse_candidates(data), 6400)):
            with self.assertRaisesRegex(ValueError, "distance tolerance"):
                observe(data)

    def test_invalid_depth_calibration_query_and_provenance_fail_before_extraction(self):
        data = observed_corner()
        cases = [
            {"surface_side_source": value} for value in (None, "", "   ", 9)
        ] + [
            {"orthogonality_tolerance_rad": value}
            for value in (-0.01, np.nan, np.inf, np.pi/2, True, [0.1])
        ] + [
            {"depth": np.zeros((3, 3, 3))}, {"depth": np.empty((0, 0))},
            {"K": np.zeros((3, 3))}, {"owner_mask": np.zeros((2, 2), bool)},
            {"owner_mask": np.zeros_like(data["owner"])},
            {"axis_pixel": [0, 0]}, {"anchor_pixel": [96, 49]},
            {"anchor_pixel": [30, np.nan]}, {"anchor_pixel": [30, 49, 0]},
        ]
        for overrides in cases:
            with self.subTest(fields=list(overrides)):
                with patch(EXTRACTOR) as extract:
                    with self.assertRaises(ValueError):
                        observe(data, **overrides)
                    extract.assert_not_called()
        for invalid in (0.0, -0.1, np.nan, np.inf):
            broken = data["depth"].copy()
            broken[49, 30] = invalid
            with self.subTest(anchor_depth=invalid), patch(EXTRACTOR) as extract:
                with self.assertRaises(ValueError):
                    observe(data, depth=broken)
                extract.assert_not_called()

    def test_inputs_and_extractor_evidence_are_not_modified_or_aliased(self):
        data = observed_corner()
        candidates = coarse_candidates(data)
        original_data, original_candidates = deepcopy(data), deepcopy(candidates)
        with patch(EXTRACTOR, return_value=(candidates, 6400)):
            result = observe(data)
        for key in ("depth", "K", "owner", "axis_pixel", "anchor_pixel"):
            np.testing.assert_array_equal(data[key], original_data[key])
        result["planes"]["axis"]["normal"][:] = 0
        result["planes"]["axis"]["core_mask"][:] = False
        for actual, original in zip(candidates, original_candidates):
            np.testing.assert_array_equal(actual["mask"], original["mask"])
            np.testing.assert_array_equal(actual["normal"], original["normal"])


if __name__ == "__main__":
    unittest.main()
# END independently specified observed-surface geometry checks.
