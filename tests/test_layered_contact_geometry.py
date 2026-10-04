"""Body-selected contact uses one complete native inventory per rule/state."""

# BEGIN coarse and fine contact inventories must never lend each other material.
from copy import copy, deepcopy
from itertools import product
import unittest
from unittest.mock import patch

import torch

from manipulation.contact_geometry import ContactCellGeometry, LayeredContactCellGeometry


def box(name, centre, dimensions):
    return {"name": name, "kind": "cuboid", "pose": [*centre, 1., 0., 0., 0.],
            "dimensions": list(dimensions), "source": "complete test box"}


def geometry(solids=(), gaps=()):
    return {"solids": list(solids), "query_gaps": list(gaps)}


def layered(coarse, fine):
    # Metadata is retained but deliberately not consumed by this adapter:
    # the caller has already made the body selection before each query.
    return {"layers": {"coarse": coarse, "fine": fine},
            "fine_region": {"selection_owned_by": "PlanningScene", "arbitrary_metadata": [1, 2]}}


def policy(shape, indices=(0,), centre=(0., 0., 0.), half=(.005, .01, .01), allowance=0.):
    dtype = torch.float64
    axes = torch.eye(3, dtype=dtype)
    normals = torch.cat((axes, -axes), dim=0)
    origin = torch.tensor(centre, dtype=dtype)
    extents = torch.tensor(half, dtype=dtype)
    bounds = torch.cat((origin + extents, -origin + extents))
    corners = torch.tensor(list(product((-1., 1.), repeat=3)), dtype=dtype)*extents + origin
    return {"sphere_indices": indices, "active": torch.ones(shape, dtype=torch.bool),
            "normal_valid": torch.ones(shape, dtype=torch.bool),
            "normals": normals.expand(*shape, 6, 3).clone(),
            "bounds": bounds.expand(*shape, 6).clone(),
            "feature_points": corners.expand(*shape, 8, 3).clone(), "allowance": allowance}


class LayeredContactTests(unittest.TestCase):
    def test_opposite_layer_costs_follow_mask_and_selected_gradient(self):
        near = geometry([box("near", [0., 0., 0.], [.2]*3)])
        far = geometry([box("far", [1., 0., 0.], [.2]*3)])
        mask = torch.tensor([[[False]], [[True]]])
        for coarse, fine, expected in ((near, far, [.01, 0.]), (far, near, [0., .01])):
            with self.subTest(expected=expected):
                cells = LayeredContactCellGeometry(layered(coarse, fine))
                spheres = torch.tensor([[[[.14, 0., 0., .05]]]]*2, dtype=torch.float64,
                                       requires_grad=True)
                cost, enabled = cells.query(spheres, [policy((2, 1))], mask)
                torch.testing.assert_close(cost.flatten(), torch.tensor(expected, dtype=torch.float64))
                self.assertTrue(enabled.all())
                cost.sum().backward()
                expected_dx = torch.tensor([-1. if value else 0. for value in expected], dtype=torch.float64)
                torch.testing.assert_close(spheres.grad[..., 0].flatten(), expected_dx)
                for name in ("coarse", "fine"):
                    self.assertEqual(cells.last_report["layers"][name]["enabled_spheres"], 1)

    def test_finite_permission_uses_only_the_selected_layer(self):
        permitted = geometry([box("permitted", [0., 0., -.005], [.006, .006, .01])])
        outside = geometry([box("outside_patch", [.0045, 0., -.0005], [.001, .002, .001])])
        spheres = torch.tensor([[[[0., 0., .009, .01]]]]*2, dtype=torch.float64)
        rule = policy((2, 1), centre=(0., 0., -.001), half=(.004, .004, .001), allowance=.002)
        mask = torch.tensor([[[False]], [[True]]])
        for coarse, fine, permitted_state in ((permitted, outside, 0), (outside, permitted, 1)):
            with self.subTest(permitted_state=permitted_state):
                cells = LayeredContactCellGeometry(layered(coarse, fine))
                cost, enabled = cells.query(spheres, [rule], mask)
                self.assertEqual(cost[permitted_state].item(), 0.)
                self.assertGreater(cost[1-permitted_state].item(), 0.)
                self.assertTrue(enabled.all())
                permitted_name = "coarse" if permitted_state == 0 else "fine"
                self.assertEqual(cells.last_report["layers"][permitted_name]["permitted_pairs"], 1)

    def test_different_rules_can_select_different_layers_in_one_state(self):
        near = geometry([box("near", [0., 0., 0.], [.2]*3)])
        cells = LayeredContactCellGeometry(layered(near, geometry()))
        spheres = torch.tensor([[[[.14, 0., 0., .05], [.14, 0., 0., .05]]]], dtype=torch.float64)
        cost, enabled = cells.query(spheres, [policy((1, 1), (0,)), policy((1, 1), (1,))],
                                    torch.tensor([[[False, True]]]))
        torch.testing.assert_close(cost, torch.tensor([[[.01, 0.]]], dtype=torch.float64))
        self.assertTrue(enabled.all())

    def test_required_contact_cannot_borrow_occupancy_from_the_other_layer(self):
        occupied = geometry([box("occupied", [0., 0., 0.], [.002]*3)])
        empty = geometry()
        mask = torch.tensor([[[False]], [[True]]])
        for coarse, fine, expected in ((occupied, empty, [0., 1.]), (empty, occupied, [1., 0.])):
            with self.subTest(expected=expected):
                cells = LayeredContactCellGeometry(layered(coarse, fine))
                residual = cells.required_contact(policy((2, 1)), mask)
                torch.testing.assert_close(residual.flatten(), torch.tensor(expected, dtype=torch.float64))
                # Each native solver receives only its own state's activity;
                # occupied material is verified by the existing complete LP.
                for name, layer in cells.layers.items():
                    report = cells.last_required_report["layers"][name]
                    self.assertEqual(report["active_states"], 1)
                    self.assertEqual(report["candidate_cells"], len(layer.centres))

    def test_required_contact_preserves_selected_highs_dual_gradient(self):
        occupied = geometry([box("occupied", [0., 0., 0.], [.002]*3)])
        separated = geometry([box("separated", [.02, 0., 0.], [.02]*3)])
        cells = LayeredContactCellGeometry(layered(occupied, separated))
        rule = policy((2, 1))
        rule["bounds"].requires_grad_(True)
        residual = cells.required_contact(rule, torch.tensor([[[False]], [[True]]]))
        torch.testing.assert_close(residual.flatten(), torch.tensor([0., .0025], dtype=torch.float64))
        residual.sum().backward()
        expected = torch.zeros_like(rule["bounds"])
        expected[1, 0, 0] = -.5
        torch.testing.assert_close(rule["bounds"].grad, expected)

    def test_mixed_policy_layers_are_rejected_even_for_inactive_states(self):
        cells = LayeredContactCellGeometry(layered(geometry(), geometry()))
        spheres = torch.ones((1, 1, 2, 4), dtype=torch.float64)
        mask = torch.tensor([[[False, True]]])
        for active in (True, False):
            with self.subTest(active=active):
                rule = policy((1, 1), (0, 1))
                rule["active"].fill_(active)
                with patch.object(cells.layers["coarse"], "query") as coarse_query:
                    with self.assertRaisesRegex(ValueError, "mixed coarse/fine"):
                        cells.query(spheres, [rule], mask)
                    coarse_query.assert_not_called()
                with patch.object(cells.layers["coarse"], "required_contact") as coarse_required:
                    with self.assertRaisesRegex(ValueError, "mixed coarse/fine"):
                        cells.required_contact(rule, mask)
                    coarse_required.assert_not_called()

    def test_mask_shape_dtype_and_rule_membership_are_explicit(self):
        cells = LayeredContactCellGeometry(layered(geometry(), geometry()))
        spheres = torch.ones((1, 1, 1, 4), dtype=torch.float64)
        for mask in (torch.zeros((1, 1, 1)), torch.zeros((1, 1), dtype=torch.bool),
                     torch.zeros((1, 2, 1), dtype=torch.bool)):
            with self.subTest(mask_shape=mask.shape, dtype=mask.dtype):
                with self.assertRaisesRegex(ValueError, "fine_mask"):
                    cells.query(spheres, [policy((1, 1))], mask)
        mask = torch.zeros((1, 1, 1), dtype=torch.bool)
        for indices in ((), (-1,), (1,), (True,)):
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, "sphere_indices"):
                    cells.required_contact(policy((1, 1), indices), mask)

    def test_inactive_required_report_is_cleared_on_each_call(self):
        occupied = geometry([box("occupied", [0., 0., 0.], [.002]*3)])
        cells = LayeredContactCellGeometry(layered(occupied, occupied))
        rule = policy((1, 1))
        cells.required_contact(rule, torch.zeros((1, 1, 1), dtype=torch.bool))
        self.assertIsNotNone(cells.last_required_report["layers"]["coarse"])
        cells.required_contact(rule, torch.ones((1, 1, 1), dtype=torch.bool))
        self.assertIsNone(cells.last_required_report["layers"]["coarse"])
        self.assertIsNotNone(cells.last_required_report["layers"]["fine"])

    def test_source_inventory_diagnostics_and_rollout_copy_remain_separate(self):
        source = layered(geometry([box("coarse", [0., 0., 0.], [.2]*3)]), geometry())
        original = deepcopy(source)
        cells = LayeredContactCellGeometry(source)
        source["layers"]["coarse"]["solids"].clear()
        source["fine_region"]["arbitrary_metadata"].append(3)
        self.assertEqual(cells.geometry, original)
        self.assertEqual(cells.layer_cell_counts, {"coarse": 1, "fine": 0})
        self.assertFalse(hasattr(cells, "centres"))
        self.assertTrue(cells.complete)
        self.assertEqual(cells.gaps, [])
        self.assertEqual(cells.workspace_bytes(), sum(layer.workspace_bytes() for layer in cells.layers.values()))
        self.assertEqual(cells.pair_chunk, 2*ContactCellGeometry.pair_chunk)
        self.assertEqual(cells.max_candidate_pairs, 2*ContactCellGeometry.max_candidate_pairs)
        self.assertEqual(cells.cell_half_diagonal_m, 0.)
        cells.last_report = {"parent": True}
        cells.layers["coarse"].last_report = {"parent": True}
        clone = copy(cells)
        self.assertIsNone(clone.last_report)
        self.assertIsNone(clone.layers["coarse"].last_report)
        self.assertEqual(cells.layers["coarse"].last_report, {"parent": True})
        self.assertIs(clone.layers["coarse"].centres, cells.layers["coarse"].centres)
        self.assertIs(clone.layers["coarse"].tree, cells.layers["coarse"].tree)

    def test_partial_layer_is_disclosed_and_cannot_authorize_contact(self):
        cells = LayeredContactCellGeometry(layered(geometry(), geometry(gaps=("incomplete",))))
        self.assertFalse(cells.complete)
        self.assertEqual(cells.gaps, ["fine: incomplete"])
        mask = torch.zeros((1, 1, 1), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "partial"):
            cells.query(torch.ones((1, 1, 1, 4), dtype=torch.float64), [policy((1, 1))], mask)
        with self.assertRaisesRegex(ValueError, "partial"):
            cells.required_contact(policy((1, 1)), mask)

    def test_original_single_layer_query_keeps_its_cost_and_gradient(self):
        cells = ContactCellGeometry(geometry([box("original", [0., 0., 0.], [.2]*3)]))
        spheres = torch.tensor([[[[.14, 0., 0., .05]]]], dtype=torch.float64, requires_grad=True)
        cost, enabled = cells.query(spheres, [policy((1, 1))])
        self.assertAlmostEqual(cost.item(), .01)
        self.assertTrue(enabled.item())
        cost.sum().backward()
        self.assertAlmostEqual(spheres.grad[0, 0, 0, 0].item(), -1.)


if __name__ == "__main__":
    unittest.main()
# END finite permission and required occupancy use the same selected native layer.
