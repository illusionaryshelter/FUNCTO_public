"""Complete finite contact intersections, gradients and admission bounds."""

# BEGIN partial contact does not require whole-pad or whole-sphere containment.
from dataclasses import replace
import unittest

import numpy as np
import torch

from manipulation.contact_geometry import ContactCellGeometry, make_contact_geometry, region_slack
from manipulation.tsdf_geometry import tsdf_geometry_from_samples
from test_manipulation_planning_scene import BoxQuery, contact_rule, inputs, pose_path, scene
from manipulation.planning_scene import CollisionOwner


def box(name, centre, dimensions):
    return {"name": name, "kind": "cuboid", "pose": [*centre, 1., 0., 0., 0.],
            "dimensions": dimensions, "source": "complete explicit test solid"}


def material_scene(extra=(), **rule_changes):
    query = BoxQuery("surface")
    query.independent_geometry = {"solids": [box("local_bar", [0., 0., -.005], [.006, .006, .01]), *extra], "query_gaps": []}
    target = CollisionOwner("surface", query, pose_path(), "complete observed local solids", True)
    rule = contact_rule(feature_points=[[-.05, -.05, -.01], [.05, -.05, -.01],
        [.05, .05, -.01], [-.05, .05, -.01]], patch_half_size=[.004, .004], **rule_changes)
    return scene((target,), materials=("pad",), contacts=(rule,))


class CompleteContactTests(unittest.TestCase):
    def test_partial_sphere_cap_and_large_pad_are_permitted(self):
        world = material_scene()
        args = inputs(positions=[[0., 0., .009]], sphere_rows=[[[0., 0., .009, .01]]])
        self.assertEqual(world.evaluate(*args).item(), 0.)
        report = world._contact_cells["surface"].last_report
        self.assertEqual(report["candidate_pairs"], 1)
        self.assertEqual(report["permitted_pairs"], 1)
        # Both the .1 m material face and .02 m sphere projection extend
        # beyond the .008 m patch; only the actual local overlap is permitted.
        self.assertEqual(world.contacts[0].patch_half_size[0], .004)

    def test_second_side_or_back_overlap_cannot_borrow_local_permission(self):
        args = inputs(positions=[[0., 0., .009]], sphere_rows=[[[0., 0., .009, .01]]])
        for obstacle in (box("outside_patch", [.0045, 0., -.0005], [.001, .002, .001]),
                         box("finger_back", [0., 0., .018], [.002, .002, .004])):
            with self.subTest(obstacle=obstacle["name"]):
                world = material_scene((obstacle,))
                self.assertGreater(world.evaluate(*args).item(), 0.)
                self.assertEqual(world._contact_cells["surface"].last_report["candidate_pairs"], 2)

    def test_unpermitted_depth_gradient_matches_complete_box_distance(self):
        world = material_scene()
        def cost(z, gradient=False):
            args = inputs(positions=[[0., 0., z]], sphere_rows=[[[0., 0., z, .01]]])
            args[0].requires_grad_(gradient)
            result = world.evaluate(*args).sum()
            return result, args[0]
        result, balls = cost(.006, True)
        result.backward()
        step = 1e-6
        difference = (cost(.006+step)[0]-cost(.006-step)[0])/(2*step)
        self.assertGreater(result.item(), .0039)
        self.assertAlmostEqual(balls.grad[0, 0, 0, 2].item(), difference.item(), places=8)

    def test_spatial_culling_retains_all_near_cells(self):
        remote = tuple(box(f"remote_{i}", [10.+i, 0., 0.], [.001]*3) for i in range(300))
        world = material_scene(remote)
        args = inputs(positions=[[0., 0., .009]], sphere_rows=[[[0., 0., .009, .01]]])
        self.assertEqual(world.evaluate(*args).item(), 0.)
        report = world._contact_cells["surface"].last_report
        self.assertEqual(report["raw_pair_product"], 301)
        self.assertEqual(report["candidate_pairs"], 1)

    def test_pair_budget_is_checked_before_neighbor_list_allocation(self):
        world = material_scene((box("second", [0., 0., -.005], [.006]*3),))
        cells = world._contact_cells["surface"]
        cells.max_candidate_pairs = 1
        actual = cells.tree
        calls = []
        class Tree:
            def query_ball_point(self, *args, **kwargs):
                calls.append(kwargs.get("return_length", False))
                if not kwargs.get("return_length", False):
                    raise AssertionError("Unadmitted neighbor lists were allocated")
                return actual.query_ball_point(*args, **kwargs)
        cells.tree = Tree()
        args = inputs(positions=[[0., 0., .009]], sphere_rows=[[[0., 0., .009, .01]]])
        with self.assertRaisesRegex(ValueError, "candidate pairs, budget 1"):
            world.evaluate(*args)
        self.assertEqual(calls, [True])

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

    def test_required_contact_needs_target_occupancy_in_the_finite_region(self):
        args = inputs(positions=[[0., 0., .009]], sphere_rows=[[[0., 0., .009, .01]]])
        for solids in ([], [box("far", [.2, 0., -.005], [.002]*3)],
                       [box("near_but_outside_patch", [.02, 0., -.005], [.002]*3)]):
            with self.subTest(solids=solids):
                world = material_scene(required=True)
                geometry = {"solids": solids, "query_gaps": []}
                world.owners[0].query.independent_geometry = geometry
                world._contact_cells["surface"] = ContactCellGeometry(geometry)
                self.assertGreater(world.components(*args)["contact/pad_to_surface"].item(), 0.)
                world.contacts = (replace(world.contacts[0], required=False),)
                self.assertEqual(world.evaluate(*args).item(), 0.)
        self.assertEqual(material_scene(required=True).evaluate(*args).item(), 0.)


if __name__ == "__main__":
    unittest.main()
# END every permitted pair has a complete bounded geometric witness.
