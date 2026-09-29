"""CPU-only observation tests; no asset, model, renderer or ground truth input."""

# BEGIN protruding-line observation and explicit-anchor invariants.
from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.observations import backproject
from manipulation.protruding_geometry import (_line_axis, observe_protruding_line,
                                             protruding_frames)


def observation():
    """A visible flat-backed U shape, with short arms at varying protrusion."""
    depth = np.ones((80, 180), float)
    K = np.array([[900., 0., 90.], [0., 900., 40.], [0., 0., 1.]])
    region = np.zeros_like(depth, bool)
    region[25:34, 20:160] = True
    depth[25:34, 20:160] = .96
    for row in range(34, 61):
        region[row, 20:29] = region[row, 151:160] = True
        depth[row, 20:29] = depth[row, 151:160] = .96 + .04*(row-33)/27
    plane = {"origin": np.array([0., 0., 1.]), "normal": np.array([0., 0., 1.]),
             "mask": ~region, "plane_rms_m": 0.}
    return depth, K, region, plane


def same_frame_set(test, first, second, atol=1e-10):
    errors = np.linalg.norm(first[:, None]-second[None], axis=(2, 3))
    test.assertTrue(np.all(np.min(errors, axis=0) <= atol), errors)
    test.assertTrue(np.all(np.min(errors, axis=1) <= atol), errors)


class ProtrudingGeometryTest(unittest.TestCase):
    def test_front_band_excludes_arms_and_preserves_explicit_offcentre_anchor(self):
        depth, K, region, plane = observation()
        geometry = observe_protruding_line(depth, K, region, plane)
        self.assertTrue(np.all(~geometry["front_mask"] | region))
        self.assertFalse(geometry["front_mask"][55, 24])
        np.testing.assert_allclose(geometry["outward_normal"], [0, 0, -1])
        self.assertGreater(abs(geometry["line_axis"][0]), .999)
        anchor = np.array([65., 29.])
        frames, diagnostic = protruding_frames(depth, K, anchor, geometry)
        measured = backproject(depth, K, anchor[None])[0]
        np.testing.assert_allclose(frames[:, :3, 3], np.repeat(measured[None], 2, axis=0))
        self.assertNotAlmostEqual(measured[0], 0.)
        for frame in frames:
            np.testing.assert_allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(frame[:3, :3]), 1.)
        np.testing.assert_allclose(frames[1, :3, :3], frames[0, :3, :3] @ np.diag([-1., 1., -1.]))
        self.assertIn("unverified", diagnostic["orientation_candidates"])

    def test_unsigned_plane_and_line_signs_preserve_candidate_set(self):
        depth, K, region, plane = observation()
        first = observe_protruding_line(depth, K, region, plane)
        plane["normal"] *= -1
        second = observe_protruding_line(depth, K, region, plane)
        second["line_axis"] *= -1
        frames1, _ = protruding_frames(depth, K, [65, 29], first)
        frames2, _ = protruding_frames(depth, K, [65, 29], second)
        same_frame_set(self, frames1, frames2)

    def test_line_fit_is_point_order_independent_and_se3_equivariant(self):
        depth, K, region, plane = observation()
        geometry = observe_protruding_line(depth, K, region, plane)
        points = backproject(depth, K, mask=geometry["fit_mask"])
        normal = geometry["outward_normal"]
        axis, _ = _line_axis(points, normal)
        shuffled, _ = _line_axis(points[np.random.default_rng(5).permutation(len(points))], normal)
        self.assertAlmostEqual(abs(float(axis @ shuffled)), 1., places=12)
        rotation = Rotation.from_euler("xyz", [.24, -.17, .38]).as_matrix()
        translation = np.array([.07, -.04, .12])
        changed, _ = _line_axis(points @ rotation.T + translation, rotation @ normal)
        self.assertAlmostEqual(abs(float(changed @ (rotation @ axis))), 1., places=12)

        frames, _ = protruding_frames(depth, K, [65, 29], geometry)
        transformed_anchor = rotation @ frames[0, :3, 3] + translation
        changed_geometry = deepcopy(geometry)
        changed_geometry["line_axis"] = rotation @ geometry["line_axis"]
        changed_geometry["outward_normal"] = rotation @ normal
        changed_geometry["plane_origin"] = rotation @ geometry["plane_origin"] + translation
        changed_geometry["front_mask"] = np.ones((11, 11), bool)
        # A calibrated ray through an integer pixel represents the same
        # measured anchor after changing the observation coordinate frame.
        x, y, z = transformed_anchor
        changed_K = np.array([[900., 0., 5-900*x/z], [0., 900., 5-900*y/z], [0., 0., 1.]])
        changed_frames, _ = protruding_frames(np.full((11, 11), z), changed_K, [5, 5], changed_geometry)
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, translation
        same_frame_set(self, transform @ frames, changed_frames)

    def test_depth_outlier_and_holes_never_become_completed_geometry(self):
        depth, K, region, plane = observation()
        depth[29, 70] = .70
        region[28:31, 95:98] = False
        # The hole is unobserved rather than newly claimed as backing plane.
        geometry = observe_protruding_line(depth, K, region, plane)
        self.assertFalse(geometry["front_mask"][29, 70])
        self.assertFalse(geometry["front_mask"][28:31, 95:98].any())
        with self.assertRaises(ValueError):
            protruding_frames(depth, K, [70, 29], geometry)

    def test_backing_arm_and_invalid_depth_cannot_replace_anchor(self):
        depth, K, region, plane = observation()
        geometry = observe_protruding_line(depth, K, region, plane)
        for anchor in ([90, 50], [24, 55], [-1, 29]):
            with self.subTest(anchor=anchor), self.assertRaises(ValueError):
                protruding_frames(depth, K, anchor, geometry)
        depth[29, 65] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid depth"):
            protruding_frames(depth, K, [65, 29], geometry)

    # BEGIN anchor evidence is distinct from the conservative axis-fit subset.
    def test_observed_boundary_anchor_is_preserved_without_entering_direction_fit(self):
        depth, K, region, plane = observation()
        anchor = [65, 25]
        geometry = observe_protruding_line(depth, K, region, plane)
        self.assertFalse(geometry["fit_mask"][25, 65])
        self.assertTrue(geometry["front_mask"][25, 65])
        self.assertTrue(np.all(~geometry["fit_mask"] | geometry["front_mask"]))
        frames, _ = protruding_frames(depth, K, anchor, geometry)
        point = backproject(depth, K, np.array([anchor]))[0]
        np.testing.assert_allclose(frames[:, :3, 3], np.repeat(point[None], 2, axis=0))
        # The adjacent pixel observes the backing, so preserving this anchor
        # cannot be implemented by requiring all neighbours to share its depth.
        self.assertGreater(abs(float(depth[24, 65]-depth[25, 65])), .03)

    def test_wrong_depth_boundary_and_unconnected_island_are_not_anchor_surface(self):
        for bad_depth in (1., .70):
            with self.subTest(bad_depth=bad_depth):
                depth, K, region, plane = observation()
                depth[25, 65] = bad_depth
                geometry = observe_protruding_line(depth, K, region, plane)
                self.assertFalse(geometry["front_mask"][25, 65])
                with self.assertRaises(ValueError):
                    protruding_frames(depth, K, [65, 25], geometry)
        depth, K, region, plane = observation()
        region[10:13, 65:68] = True
        plane["mask"][10:13, 65:68] = False
        depth[10:13, 65:68] = .96
        geometry = observe_protruding_line(depth, K, region, plane)
        self.assertFalse(geometry["front_mask"][10:13, 65:68].any())
    # END anchor evidence is distinct from the conservative axis-fit subset.

    def test_direction_and_observation_failures_are_explicit(self):
        t = np.linspace(-.1, .1, 100)
        parallel = np.column_stack((np.zeros_like(t), np.zeros_like(t), t))
        with self.assertRaisesRegex(ValueError, "parallel"):
            _line_axis(parallel, [0, 0, 1])
        circle = np.column_stack((.1*np.cos(t*30), .1*np.sin(t*30), np.zeros_like(t)))
        with self.assertRaisesRegex(ValueError, "distinct line"):
            _line_axis(circle, [0, 0, 1])
        depth, K, region, plane = observation()
        with self.assertRaisesRegex(ValueError, "outward side"):
            observe_protruding_line(np.ones_like(depth), K, region, plane)
        region[:] = False
        region[29, 20:160] = True
        with self.assertRaisesRegex(ValueError, "Insufficient interior"):
            observe_protruding_line(depth, K, region, plane)
        with self.assertRaisesRegex(ValueError, "Insufficient observed points"):
            _line_axis(np.zeros((3, 3)), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
# END protruding-line observation and explicit-anchor invariants.
