"""CPU counterexamples for complete-cell, nearest-pixel free-space evidence."""

# BEGIN independent complete-cell projection and rectangle-minimum regressions.
import itertools
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation import planning_observation as observation


def calibrated_view(depth, *, focal=(20., 20.), principal=None, transform=None):
    """Construct only the calibrated evidence consumed by the private adapter."""
    depth = np.asarray(depth, dtype=float)
    height, width = depth.shape
    cx, cy = principal if principal is not None else ((width - 1) / 2, (height - 1) / 2)
    return {"depth": depth,
            "K": np.array([[focal[0], 0., cx], [0., focal[1], cy], [0., 0., 1.]]),
            "T_output_camera": np.eye(4) if transform is None else transform}


def known(record, origin, *, spacing=.01, error=0., rays=None):
    """Return the first full cube; the public grid contract requires two cells."""
    return observation._known_free_voxels(
        {"camera": record}, ray_depths={"camera": record["depth"] if rays is None else rays},
        origin=origin, shape=(2, 2, 2), voxel_size=spacing, depth_error_m=error)[0, 0, 0]


def scalar_reference(records, rays, *, origin, shape, spacing, error):
    """Slow mathematical oracle, independent of chunking and minimum filters.

    Every vertex is transformed separately, and every integer pixel in the
    projection rectangle is inspected directly. Random comparisons avoid exact
    raster boundaries; explicit tests below cover the outward error guard there.
    """
    answer = np.zeros(shape, dtype=bool)
    for z, y, x in np.ndindex(shape):
        center = np.asarray(origin) + spacing * np.array([x, y, z])
        vertices = [center + spacing * np.array(signs) / 2
                    for signs in itertools.product((-1., 1.), repeat=3)]
        for name, record in records.items():
            transform, matrix = record["T_output_camera"], record["K"]
            camera = np.array([transform[:3, :3].T @ (vertex - transform[:3, 3])
                               for vertex in vertices])
            if min(point[2] for point in camera) <= 0.:
                continue
            pixels = np.array([[matrix[0, 0] * point[0] / point[2] + matrix[0, 2],
                                matrix[1, 1] * point[1] / point[2] + matrix[1, 2]]
                               for point in camera])
            lower, upper = pixels.min(axis=0), pixels.max(axis=0)
            height, width = rays[name].shape
            if np.any(lower < 0.) or np.any(upper > [width - 1, height - 1]):
                continue
            x0, y0 = np.rint(lower).astype(int)
            x1, y1 = np.rint(upper).astype(int)
            minimum = min(float(rays[name][row, col])
                          for row in range(y0, y1 + 1) for col in range(x0, x1 + 1))
            if max(point[2] for point in camera) + error < minimum:
                answer[z, y, x] = True
                break
    return answer


class CompleteVoxelProjectionTest(unittest.TestCase):
    def test_complete_cube_near_image_edge_is_free_but_outside_cube_is_unknown(self):
        record = calibrated_view(np.full((33, 49), 2.))
        # The entire footprint is inside the first/last image columns. A large
        # circle or padded square would unnecessarily include off-image pixels.
        for direction in (-1., 1.):
            with self.subTest(direction=direction):
                self.assertTrue(known(record, [direction * 1.16, 0., 1.]))
                self.assertFalse(known(record, [direction * 1.20, 0., 1.]))

    def test_continuous_outside_projection_is_rejected_even_when_rounding_inside(self):
        for cx, x in ((0., -.2), (12., .2)):
            with self.subTest(cx=cx):
                record = calibrated_view(np.full((13, 13), 2.),
                                         focal=(1., 1.), principal=(cx, 6.))
                # Every horizontal coordinate rounds into column 0 or 12,
                # but the cube is wholly outside the continuous image domain.
                self.assertFalse(known(record, [x, 0., 1.]))

    def test_narrow_rectangle_excludes_external_hole_but_checks_its_interior(self):
        record = calibrated_view(np.full((49, 49), 2.), focal=(60., 6.))
        rays = record["depth"].copy()
        rays[26, 24] = 0.
        self.assertTrue(known(record, [0., 0., 1.], spacing=.1, rays=rays))
        # This pixel is between the corner projections; testing only vertices
        # would miss it. A zero prefix remains unknown despite valid neighbours.
        rays[24, 24] = 0.
        self.assertFalse(known(record, [0., 0., 1.], spacing=.1, rays=rays))

    def test_rotated_cell_uses_farthest_vertex_depth_and_original_error(self):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("y", 45., degrees=True).as_matrix()
        center = transform[:3, :3] @ np.array([0., 0., 1.])
        record = calibrated_view(np.full((49, 49), 1.15), transform=transform)
        self.assertTrue(known(record, center, spacing=.2))
        # Center z=1 is free, but the far corner has z=1+sqrt(2)/10.
        self.assertFalse(known(record, center, spacing=.2, error=.01))
        record["depth"][:] = 1.12
        self.assertFalse(known(record, center, spacing=.2))

    def test_camera_plane_crossing_never_proves_free(self):
        record = calibrated_view(np.full((49, 49), 2.), focal=(.1, .1))
        self.assertFalse(known(record, [0., 0., .03], spacing=.1))
        self.assertFalse(known(record, [0., 0., -.03], spacing=.1))

    def test_depth_equality_with_farthest_vertex_plus_error_is_not_free(self):
        # Binary-exact constants isolate strict inequality from decimal error.
        record = calibrated_view(np.full((49, 49), 1.25))
        self.assertFalse(known(record, [0., 0., 1.], spacing=.25, error=.125))
        record["depth"][:] = 1.25 + 1e-8
        self.assertTrue(known(record, [0., 0., 1.], spacing=.25, error=.125))

    def test_large_coordinate_cancellation_cannot_erase_cell_depth_extent(self):
        transform = np.eye(4)
        transform[:3, 3] = 1e15
        record = calibrated_view(np.full((49, 49), 1.04), transform=transform)
        # At this coordinate magnitude +/- .05 metre corner offsets round
        # away. A center-only calculation would falsely claim z=1 < 1.04,
        # although the conceptual complete cube reaches z=1.05. The arithmetic
        # guard must reject the insufficiently resolved geometry.
        self.assertFalse(known(record, [1e15, 1e15, 1e15 + 1.], spacing=.1))

    def test_half_pixel_boundary_includes_both_neighbours_with_roundoff_guard(self):
        record = calibrated_view(np.full((13, 13), 2.), focal=(1., 1.), principal=(5.5, 6.))
        rays = record["depth"].copy()
        rays[:, 5] = 0.
        # The left face has u=5.5 exactly. The numerical uncertainty guard
        # includes both adjacent pixels, regardless of ties-to-even selecting 6.
        self.assertFalse(known(record, [.05, 0., 1.], spacing=.1, rays=rays))
        self.assertFalse(known(record, [.05 - 1e-6, 0., 1.], spacing=.1, rays=rays))
        self.assertTrue(known(record, [.05 + 1e-6, 0., 1.], spacing=.1, rays=rays))

    def test_two_partial_views_cannot_jointly_claim_a_complete_cube(self):
        first = calibrated_view(np.full((49, 49), 2.), focal=(60., 60.))
        second = calibrated_view(np.full((49, 49), 2.), focal=(60., 60.))
        records = {"first": first, "second": second}
        rays = {name: record["depth"].copy() for name, record in records.items()}
        rays["first"][23, 23] = 0.
        rays["second"][25, 25] = 0.
        result = observation._known_free_voxels(
            records, ray_depths=rays, origin=[0., 0., 1.], shape=(2, 2, 2),
            voxel_size=.1, depth_error_m=0.)
        self.assertFalse(result[0, 0, 0])
        rays["second"][:] = 2.
        result = observation._known_free_voxels(
            records, ray_depths=rays, origin=[0., 0., 1.], shape=(2, 2, 2),
            voxel_size=.1, depth_error_m=0.)
        self.assertTrue(result[0, 0, 0])

    def test_vectorized_projection_matches_independent_scalar_oracle(self):
        rng = np.random.default_rng(721)
        for case in range(8):
            records, rays = {}, {}
            for index in range(2):
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_rotvec(rng.uniform(-.16, .16, 3)).as_matrix()
                transform[:3, 3] = rng.uniform(-.08, .08, 3)
                record = calibrated_view(rng.uniform(1.05, 1.5, (29, 37)),
                                         focal=(24.7, 17.3), transform=transform)
                name = str(index)
                records[name] = record
                rays[name] = record["depth"].copy()
                rays[name][rng.random((29, 37)) < .02] = 0.
            options = dict(origin=[-.19, -.11, .84], shape=(4, 3, 5), spacing=.055, error=.017)
            expected = scalar_reference(records, rays, **options)
            actual = observation._known_free_voxels(
                records, ray_depths=rays, origin=options["origin"], shape=options["shape"],
                voxel_size=options["spacing"], depth_error_m=options["error"])
            with self.subTest(case=case):
                np.testing.assert_array_equal(actual, expected)


class RectangleMinimumTest(unittest.TestCase):
    def test_rectangle_minima_equal_direct_slices_for_all_small_extents(self):
        rng = np.random.default_rng(218)
        for shape in ((1, 1), (1, 17), (17, 1), (7, 10), (18, 19)):
            depth = rng.uniform(.1, 3., shape)
            depth[0, 0] = 0.
            height, width = shape
            lower, upper = [], []
            # Every width/height includes size 1, odd/even sizes, and both
            # sides of each power-of-two threshold used by the acceleration.
            for h in range(1, height + 1):
                for w in range(1, width + 1):
                    for y, x in ((0, 0), (height - h, width - w),
                                 (int(rng.integers(height - h + 1)),
                                  int(rng.integers(width - w + 1)))):
                        lower.append((x, y))
                        upper.append((x + w - 1, y + h - 1))
            expected = np.array([depth[y0:y1 + 1, x0:x1 + 1].min()
                                 for (x0, y0), (x1, y1) in zip(lower, upper)])
            actual = observation._rectangle_min_depth(depth, np.array(lower), np.array(upper))
            with self.subTest(shape=shape):
                np.testing.assert_array_equal(actual, expected)

# END tests never modify ray evidence, permissions, spheres, or planner settings.
