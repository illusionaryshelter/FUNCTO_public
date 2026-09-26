"""Protect pose conventions in the offline diagnostic renderer.

These tests use analytic rigid transforms, not image snapshots. They catch the
function-point pivot, Euler convention and nonzero rotation-center mistakes
that could make a plausible-looking visualization geometrically misleading.
"""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from visualize_run import move_points, project_points, target_to_camera


class VisualizationGeometryTest(unittest.TestCase):
    def test_known_quarter_turn_about_function_point(self):
        start = np.array([[2., 3., 4.], [0., 0., 0.]])
        end = np.array([[5., 6., 7.], [0., 0., np.pi / 2]])
        points = np.array([[2., 3., 4.], [3., 3., 4.], [2., 4., 4.]])
        expected = np.array([[5., 6., 7.], [5., 7., 7.], [4., 6., 7.]])
        np.testing.assert_allclose(move_points(points, start, end), expected, atol=1e-12)

    def test_absolute_replay_matches_sequential_transforms_and_preserves_rigidity(self):
        rng = np.random.default_rng(3)
        points = rng.normal(size=(15, 3))
        poses = rng.normal(size=(12, 2, 3))
        sequential = points.copy()
        # A separate column-vector homogeneous representation acts as the
        # oracle for the row-vector implementation used by the renderer.
        transforms = []
        for position, angles in poses:
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
            transform[:3, 3] = position
            transforms.append(transform)
        for i in range(1, len(poses)):
            step = transforms[i] @ np.linalg.inv(transforms[i - 1])
            homogeneous = np.column_stack((sequential, np.ones(len(points))))
            sequential = (step @ homogeneous.T).T[:, :3]
            actual = move_points(points, poses[0], poses[i])
            np.testing.assert_allclose(actual, sequential, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(actual[1:] - actual[0], axis=1),
                                       np.linalg.norm(points[1:] - points[0], axis=1), atol=1e-12)
        np.testing.assert_allclose(move_points(points, poses[0], poses[0]), points, atol=1e-12)

    def test_target_inverse_and_projection_include_nonzero_rotation_center(self):
        intrinsic = np.array([[500., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        pixels = np.array([[320., 240.], [420., 300.], [220., 100.]])
        depths = np.array([0.4, 0.7, 0.9])
        camera = (np.linalg.inv(intrinsic) @ np.column_stack((pixels, np.ones(3))).T).T * depths[:, None]
        center = np.array([1., -2., 0.1])
        centroid = np.array([0.2, 0.3, -0.1])
        rotation = Rotation.from_euler("xyz", [0.3, -0.8, 1.1]).as_matrix()
        target = np.array([rotation @ (point - center) + center - centroid for point in camera])
        frame = dict(rotation_center=center, rotation_matrix=rotation, centroid=centroid)
        restored = target_to_camera(target, frame)
        np.testing.assert_allclose(restored, camera, atol=1e-12)
        np.testing.assert_allclose(project_points(restored, intrinsic), pixels, atol=1e-10)
        with self.assertRaisesRegex(ValueError, "behind the camera"):
            project_points(np.array([[1., 1., -1.]]), intrinsic)


if __name__ == "__main__":
    unittest.main()
