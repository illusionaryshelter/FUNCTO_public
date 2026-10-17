"""Exact nearest-observation regressions for generic synthetic 3-D inputs.

The oracle is SciPy's original unbounded query on the identical tree. This
checks its observable tie choice and floating-point distances without adding
a new canonical ordering, using a tolerance, or relying on a dataset fixture.
"""

import math
import unittest
from unittest import mock
import warnings

import numpy as np
from scipy.sparse import issparse
from scipy.spatial import cKDTree

from transfer import multiview_geometry as geometry
from transfer.test_multiview_geometry import calibrated_view


# BEGIN bounded-search regressions from independent adversarial review.
class TestRadiusBoundedNearest(unittest.TestCase):
    def assert_original_eligible(self, tree, queries, radius):
        """Require identical admission, indices and distance bits inside r."""
        original_distance, original_index = tree.query(queries, workers=1)
        distance, index = geometry._nearest_within_radius(tree, queries, radius)
        eligible = np.isfinite(original_distance) & (original_distance <= radius)
        actual = np.isfinite(distance) & (distance <= radius)
        np.testing.assert_array_equal(actual, eligible)
        np.testing.assert_array_equal(index[eligible], original_index[eligible])
        np.testing.assert_array_equal(distance[eligible].view(np.uint64),
                                      original_distance[eligible].view(np.uint64))
        return distance, index

    def test_default_leafsize_retains_original_equal_distance_choice(self):
        # The unrelated far points determine KD-tree subdivision and heap
        # order. On SciPy 1.15.3, unbounded k=1 chooses point 37, while k=1
        # bounded at nextafter(sqrt(2), inf) chooses point 25. Both are distance
        # one from the query; removing either changes observation provenance.
        # The oracle remains the original query, so a library update need not
        # preserve a hard-coded internal tie index to satisfy this contract.
        points = np.array([
            [3, 4, 2], [3, 3, 0], [-3, -3, -1], [0, -4, 1], [1, -1, 1],
            [3, 0, -2], [3, 3, -4], [-4, -3, -3], [-3, 0, -3], [-2, 0, 2],
            [4, 0, 3], [3, -3, 0], [0, 1, -3], [-3, -1, 0], [-2, -4, 0],
            [1, -2, -2], [2, 2, 0], [1, 3, 0], [3, 1, 2], [-3, 4, 1],
            [1, -4, 4], [3, 1, 2], [3, -3, -3], [-1, 1, -3], [-2, 0, 0],
            [0, 0, -1], [0, 1, 0], [2, 1, -1], [3, 0, -4], [0, 2, 1],
            [0, 3, 2], [-4, 0, 4], [2, 3, 2], [-4, 0, 0], [-3, 4, 0],
            [0, -3, -2], [4, -3, -3], [0, 1, -2], [2, 2, 4], [2, 4, -4],
            [3, -3, 4], [3, 3, 4], [-4, 0, -1], [-4, 0, -3], [-1, 3, 1],
            [4, -3, -4], [-4, 3, -1], [-1, 4, 1], [3, -2, -3], [-1, 1, 0],
            [-1, 1, 0], [2, 0, -1], [4, -1, -1], [-1, 4, 2], [-1, 4, -1],
        ], dtype=float)
        queries = np.array([[0., 0., -2.]])
        distance, _ = self.assert_original_eligible(cKDTree(points), queries, math.sqrt(2.))
        np.testing.assert_array_equal(distance, [1.])

    def test_duplicate_and_lattice_ties_across_tree_subdivisions(self):
        rng = np.random.default_rng(76031)
        for leafsize in (1, 2, 4, 8, 16, 32, 64):
            for repeat in range(24):
                points = rng.integers(-4, 5, size=(73, 3)).astype(float)
                points[-8:] = points[:8]
                queries = rng.integers(-4, 5, size=(29, 3)).astype(float)
                if repeat % 2:
                    queries += .5
                radius = float(rng.choice([1., math.sqrt(2.), math.sqrt(3.), 2., 4.]))
                with self.subTest(leafsize=leafsize, repeat=repeat):
                    self.assert_original_eligible(cKDTree(points, leafsize=leafsize),
                                                  queries, radius)

    def test_subnormal_radius_retains_coincident_and_underflowed_distances(self):
        for radius in (np.nextafter(0., 1.), 1e-300, 1e-200,
                       math.sqrt(np.nextafter(0., 1.)), 1e-160):
            for points in (np.zeros((1, 3)), np.zeros((40, 3)),
                           np.array([[1e-200, 0., 0.], [1e-170, 0., 0.]])):
                with self.subTest(radius=radius, count=len(points)):
                    distance, _ = self.assert_original_eligible(
                        cKDTree(points), np.zeros((1, 3)), radius)
                    # Preserve the original numerical contract: squared
                    # underflow can make a nonzero displacement return zero.
                    np.testing.assert_array_equal(distance, [0.])

    def test_radius_and_adjacent_floats_preserve_inclusive_admission(self):
        tree = cKDTree(np.zeros((1, 3)))
        for radius in (1e-160, 1e-100, .03, 1., 1e100, 1e150):
            lengths = np.array([np.nextafter(radius, 0.), radius,
                                np.nextafter(radius, np.inf)])
            axial = np.column_stack((lengths, np.zeros((3, 2))))
            diagonal = np.column_stack((lengths / math.sqrt(2.),
                                        lengths / math.sqrt(2.), np.zeros(3)))
            with self.subTest(radius=radius):
                self.assert_original_eligible(tree, np.vstack((axial, diagonal)), radius)

    def test_large_radius_square_overflow_has_no_warning_or_distance_change(self):
        tree = cKDTree(np.array([[0., 0., 0.], [1e150, 0., 0.]]))
        queries = np.array([[.5e150, 0., 0.], [1e153, 0., 0.]])
        for radius in (1e154, 1e155, np.finfo(float).max):
            with self.subTest(radius=radius), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                self.assert_original_eligible(tree, queries, radius)

    def test_empty_queries_empty_tree_and_single_neighbor(self):
        empty = np.empty((0, 3))
        point = np.zeros((1, 3))
        queries = np.array([[0., 0., 0.], [.125, 0., 0.], [2., 0., 0.]])
        for points, requested in ((point, empty), (empty, queries),
                                  (point, queries), (empty, empty)):
            with self.subTest(tree_size=len(points), queries=len(requested)):
                self.assert_original_eligible(cKDTree(points), requested, .5)
        distance, index = geometry._nearest_within_radius(cKDTree(point), queries[-1:], .5)
        np.testing.assert_array_equal(distance, [np.inf])
        np.testing.assert_array_equal(index, [1])


class TestRadiusBoundedSurfaceIntegration(unittest.TestCase):
    def assert_exact_surface(self, first, second):
        """Check full nested surface values, including CSR graph representation."""
        if isinstance(first, np.ndarray):
            self.assertEqual(first.dtype, second.dtype)
            np.testing.assert_array_equal(first, second)
            if first.dtype.kind == "f":
                np.testing.assert_array_equal(np.signbit(first), np.signbit(second))
        elif issparse(first):
            self.assertEqual(first.format, second.format)
            self.assertEqual(first.shape, second.shape)
            for field in ("data", "indices", "indptr"):
                self.assert_exact_surface(getattr(first, field), getattr(second, field))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_exact_surface(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for before, after in zip(first, second):
                self.assert_exact_surface(before, after)
        else:
            self.assertEqual(first, second)

    def test_complete_surface_matches_original_query_for_generic_calibrated_views(self):
        # Patch only the query helper to the original SciPy call. Both builds
        # still execute production plane checks, mutuality, projection gates,
        # representative choice and adjacency, so integration is exercised.
        empty_depth = np.full((9, 9), np.nan)
        shifted_camera = np.eye(4)
        shifted_camera[0, 3] = .002
        scenarios = (
            (calibrated_view("a"), calibrated_view("b")),
            (calibrated_view("a"), calibrated_view("b", T=shifted_camera)),
            (calibrated_view("a"), calibrated_view("b", depth=np.full((9, 9), 2.))),
            (calibrated_view("a"), calibrated_view("b", depth=empty_depth)),
            (calibrated_view("a", depth=empty_depth), calibrated_view("b", depth=empty_depth)),
        )
        for number, views in enumerate(scenarios):
            with self.subTest(scenario=number):
                optimized = geometry.build_target_surface(views)
                with mock.patch.object(geometry, "_nearest_within_radius",
                                       side_effect=lambda tree, points, radius: tree.query(points, workers=1)):
                    original = geometry.build_target_surface(views)
                self.assert_exact_surface(original, optimized)
# END bounded-search regressions; no task or dataset data are loaded.


if __name__ == "__main__":
    unittest.main()
