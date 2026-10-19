"""Complete finite contact intersections, gradients and admission bounds."""

# BEGIN partial contact does not require whole-pad or whole-sphere containment.
from dataclasses import replace
import unittest

import numpy as np
import torch

from manipulation.contact_geometry import ContactCellGeometry, make_contact_geometry, region_slack
from manipulation.tsdf_geometry import tsdf_geometry_from_samples
from manipulation.planning_scene import CollisionOwner


def box(name, centre, dimensions):
    return {"name": name, "kind": "cuboid", "pose": [*centre, 1., 0., 0., 0.],
            "dimensions": dimensions, "source": "complete explicit test solid"}




class CompleteContactTests(unittest.TestCase):





    # BEGIN observed TSDF evidence retains its actual spatial dimension.
    def test_isolated_observed_point_and_empty_union_are_explicit(self):
        field = np.zeros((2, 2, 2))
        field[0, 1, 1] = -.01
        observed = np.zeros_like(field, dtype=bool)
        observed[0, 1, 1] = True
        geometry = tsdf_geometry_from_samples(field, observed,
            origin=[-.005]*3, voxel_size=.01, name="observed", source="explicit observed TSDF")
        cells = make_contact_geometry(geometry)
        np.testing.assert_allclose(cells.centres, [[-.005, .005, .005]])
        self.assertEqual(cells.radii.tolist(), [0.])
        self.assertEqual(len(cells.surface_components), 0)
        empty = ContactCellGeometry({"solids": [], "query_gaps": []})
        output, enabled = empty.query(torch.zeros((1, 1, 1, 4), dtype=torch.float64), [])
        self.assertEqual(output.item(), 0.)
        self.assertFalse(enabled.item())
    # END no negative sample becomes an ESDF-threshold occupied cube.

    def test_required_overlap_slack_has_the_dual_gradient(self):
        normals = torch.tensor([[[1., 0., 0.], [-1., 0., 0.]]], dtype=torch.float64)
        bounds = torch.tensor([[-.01, -.01]], dtype=torch.float64, requires_grad=True)
        value = region_slack(normals, bounds)
        value.sum().backward()
        self.assertAlmostEqual(value.item(), .01)
        torch.testing.assert_close(bounds.grad, torch.tensor([[-.5, -.5]], dtype=torch.float64))



if __name__ == "__main__":
    unittest.main()
# END every permitted pair has a complete bounded geometric witness.
