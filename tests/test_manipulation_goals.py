"""Geometric goals must constrain object extent, not merely its centre."""

import itertools
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.goals import (InteriorRegion, SupportSurface, check_inside,
                                check_on, place_inside, place_on)


def box(half):
    return np.asarray(list(itertools.product(*[(-x, x) for x in half])))


class GeometricGoalsTest(unittest.TestCase):
    def test_tilted_surface_preserves_orientation_and_whole_footprint(self):
        rotation = Rotation.from_euler("xyz", [0.3, -0.4, 0.2]).as_matrix()
        surface = SupportSurface(np.array([1., 2., 3.]), rotation[:, :2], rotation[:, 2],
                                 np.array([[-.5, -.4], [.5, -.4], [.5, .4], [-.5, .4]]))
        body = box([.1, .05, .08])
        preferred = np.eye(4)
        preferred[:3, :3] = rotation
        preferred[:3, 3] = surface.origin + rotation @ [.49, .39, 1.]
        pose, report = place_on(preferred, body, surface)
        np.testing.assert_allclose(pose[:3, :3], rotation)
        np.testing.assert_allclose(rotation.T @ (pose[:3, 3]-surface.origin), [.4, .35, .08], atol=1e-8)
        self.assertLess(report["support_gap_m"], 1e-8)
        self.assertLess(check_on(pose, body, surface)["boundary_violation_m"], 1e-8)

    def test_object_too_large_for_support_is_rejected(self):
        surface = SupportSurface(np.zeros(3), np.eye(3)[:, :2], np.array([0., 0., 1.]),
                                 np.array([[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]]))
        with self.assertRaisesRegex(ValueError, "does not fit"):
            place_on(np.eye(4), box([.2, .2, .1]), surface)

    def test_containment_checks_extent_instead_of_center(self):
        region = InteriorRegion.from_vertices(box([.2, .3, .4]), "known free cavity")
        too_large = box([.25, .1, .1])
        self.assertGreater(check_inside(np.eye(4), too_large, region)["interior_violation_m"], 0)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            place_inside(np.eye(4), too_large, region)
        pose = np.eye(4)
        pose[:3, 3] = [.5, 0, 0]
        pose, report = place_inside(pose, box([.1, .1, .1]), region, margin=.01)
        self.assertAlmostEqual(pose[0, 3], .09, places=7)
        self.assertLess(report["interior_violation_m"], 1e-8)

    def test_invalid_geometry_and_reflections_are_rejected(self):
        with self.assertRaises(ValueError):
            InteriorRegion.from_vertices(np.zeros((4, 3)), "degenerate")
        region = InteriorRegion.from_vertices(box([1., 1., 1.]), "known cavity")
        reflection = np.diag([-1., 1., 1., 1.])
        with self.assertRaisesRegex(ValueError, "SO\\(3\\)"):
            place_inside(reflection, box([.1, .1, .1]), region)

    def test_open_or_empty_halfspaces_are_not_a_container(self):
        # Four walls do not enclose a volume: there is no floor or ceiling.
        open_region = [[1, 0, 0, -1], [-1, 0, 0, -1],
                       [0, 1, 0, -1], [0, -1, 0, -1]]
        with self.assertRaisesRegex(ValueError, "bounded"):
            InteriorRegion(open_region, "open walls")
        # Opposite x inequalities contradict one another.
        empty = open_region + [[0, 0, 1, -1], [0, 0, -1, -1], [1, 0, 0, 2]]
        with self.assertRaisesRegex(ValueError, "nonempty"):
            InteriorRegion(empty, "contradictory evidence")


if __name__ == "__main__":
    unittest.main()
