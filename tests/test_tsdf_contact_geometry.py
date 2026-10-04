"""Complete contact semantics for affine TSDF surfaces and negative components."""

# BEGIN contact uses complete finite components and actual observed zero sets.
import unittest

import numpy as np
import torch

from manipulation.contact_geometry import (
    component_component_overlap_within_region,
    convex_component_intersects_region,
    convex_component_overlap_within_region,
    make_contact_geometry,
)
from manipulation.tsdf_geometry import tsdf_geometry_from_samples


def region(lower, upper):
    """An axis-aligned finite permission region in the target's coordinates."""
    return np.r_[np.eye(3), -np.eye(3)], np.r_[upper, -np.asarray(lower)]


def policy(lower, upper, *, allowance=0.):
    """One active material face; its thin permitted region is explicit."""
    normals, bounds = region(lower, upper)
    height = (lower[2]+upper[2])/2
    points = [[x, y, height] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])]
    return {"active": torch.ones((1, 1), dtype=torch.bool),
            "normal_valid": torch.ones((1, 1), dtype=torch.bool),
            "sphere_indices": (0,), "allowance": allowance,
            "normals": torch.tensor(normals[None, None], dtype=torch.float64),
            "bounds": torch.tensor(bounds[None, None], dtype=torch.float64),
            "feature_points": torch.tensor([[points]], dtype=torch.float64)}


def contact_geometry(values, observed=None):
    if observed is None:
        observed = np.ones(values.shape, dtype=bool)
    return make_contact_geometry(tsdf_geometry_from_samples(values, observed,
        origin=[0., 0., 0.], voxel_size=1., name="target", source="observed test TSDF"))


class TsdfCompleteContactTests(unittest.TestCase):
    def test_surface_crossing_region_does_not_need_a_vertex_inside(self):
        triangle = np.array([[-2., -1., 0.], [2., -1., 0.], [0., 2., 0.]])
        permission = region([-.1, -.1, -.01], [.1, .1, .01])
        normals, bounds = permission
        self.assertFalse(np.any((triangle @ normals.T <= bounds).all(axis=1)))
        evidence = convex_component_intersects_region(triangle, permission)
        self.assertTrue(evidence["completed"])
        self.assertTrue(evidence["intersects"])

    def test_permission_covers_complete_clipped_overlap(self):
        triangle = np.array([[-2., -1., 0.], [2., -1., 0.], [0., 2., 0.]])
        permission = region([-.1, -.1, -.01], [.1, .1, .01])
        clipped = convex_component_overlap_within_region(permission, triangle, permission)
        self.assertTrue(clipped["contained"])
        # Identical source clipping planes imply containment, including the
        # boundary. Moving a source plane outside must still reject its full
        # overlap; floating-point handling cannot enlarge the allowed patch.
        protruding = convex_component_overlap_within_region(
            region([-.1, -.1, -.01], [.100001, .1, .01]), triangle, permission)
        self.assertFalse(protruding["contained"])
        whole = convex_component_overlap_within_region(
            region([-3., -3., -1.], [3., 3., 1.]), triangle, permission)
        self.assertTrue(whole["completed"])
        self.assertFalse(whole["contained"])
        self.assertFalse(whole["intersection_empty"])

    def test_component_intersection_and_measured_point_band_are_complete(self):
        first = np.array([[-1., 0., 0.], [1., 0., 0.]])
        second = np.array([[0., -1., 0.], [0., 1., 0.]])
        permission = region([-.01]*3, [.01]*3)
        crossing = component_component_overlap_within_region(first, second, permission)
        self.assertTrue(crossing["contained"])
        self.assertFalse(crossing["intersection_empty"])
        # Two measured point bands overlap in a box larger than the allowed
        # patch. A single allowed witness at the origin cannot permit it.
        measured = component_component_overlap_within_region(
            np.zeros((1, 3)), np.zeros((1, 3)), permission,
            first_margin_m=.02, second_margin_m=.02)
        self.assertTrue(measured["completed"])
        self.assertFalse(measured["contained"])

    def test_affine_surface_required_contact_and_distance_gradient(self):
        values = np.indices((2, 2, 2))[2].astype(float)-.5
        geometry = contact_geometry(values)
        touching = policy([.2, .2, .49], [.6, .6, .51], allowance=.01)
        self.assertEqual(geometry.required_contact(touching).item(), 0.)
        balls = torch.tensor([[[[.4, .3, .6, .2]]]], dtype=torch.float64, requires_grad=True)
        guidance = policy([0., 0., 0.], [1., 1., 1.])
        value, enabled = geometry.query(balls, [guidance])
        self.assertTrue(enabled.item())
        self.assertAlmostEqual(value.item(), .1, places=12)
        value.sum().backward()
        torch.testing.assert_close(balls.grad[0, 0, 0],
            torch.tensor([0., 0., -1., 1.], dtype=torch.float64), atol=1e-12, rtol=0.)

    def test_negative_cell_cutoff_never_establishes_required_contact(self):
        geometry = contact_geometry(-np.ones((2, 2, 2)))
        cutoff = policy([-.01, .2, .2], [.01, .8, .8], allowance=.01)
        self.assertGreater(geometry.required_contact(cutoff).item(), 0.)
        self.assertTrue(geometry.last_required_report["target_surface_inventory_empty"])
        balls = torch.tensor([[[[.05, .5, .5, .1]]]], dtype=torch.float64)
        value, _ = geometry.query(balls, [cutoff])
        self.assertGreater(value.item(), .1)

    def test_unknown_neighbors_do_not_expand_a_negative_point(self):
        values = -np.ones((2, 2, 2))
        observed = np.zeros(values.shape, dtype=bool)
        observed[0, 0, 0] = True
        geometry = contact_geometry(values, observed)
        near = policy([-.5]*3, [.5]*3)
        balls = torch.tensor([[[[.3, 0., 0., .1]]]], dtype=torch.float64)
        value, _ = geometry.query(balls, [near])
        self.assertEqual(value.item(), 0.)
        self.assertGreater(geometry.required_contact(near).item(), 0.)

    def test_an_observed_zero_point_can_establish_required_contact(self):
        values = np.zeros((1, 1, 1))
        geometry = contact_geometry(values)
        self.assertEqual(geometry.required_contact(policy([-.1]*3, [.1]*3)).item(), 0.)
        self.assertEqual(len(geometry.surface_components), 1)


if __name__ == "__main__":
    unittest.main()
# END unknown cutoffs block only where observed evidence actually exists.
