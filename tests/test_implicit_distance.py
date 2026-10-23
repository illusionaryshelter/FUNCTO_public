"""Analytic and boundary checks for direct observed-set distance queries."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.implicit_distance import ObservedField, PairPermission, query_distance
from manipulation.observed_mapping import FrozenObservedField


class ImplicitDistanceTests(unittest.TestCase):
    def setUp(self):
        axis = np.linspace(-.05, .05, 7)
        points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1)
        self.values = np.max(np.abs(points), -1) - .04
        self.field = ObservedField(self.values, np.ones(self.values.shape, bool),
                                   origin=[-.05] * 3, voxel_size=axis[1] - axis[0])

    # BEGIN task collision uses nonpenetration without a positive gap.
    def query(self, shift):
        pose = np.eye(4)
        pose[0, 3] = shift
        return query_distance(self.field, np.eye(4), self.field, pose)

    def test_positive_gaps_below_previous_clearance_are_clear(self):
        for gap in (.0015, .0001, 1e-7):
            self.assertEqual(self.query(.08 + gap)["status"], "clear")

    def test_batched_initial_bounds_match_scalar_refinement(self):
        """Mixed slopes and translated coordinates retain outward bounds."""
        from manipulation.implicit_distance import _Node
        rng = np.random.default_rng(841)
        values = rng.normal(size=(13, 12, 11))
        field = ObservedField(values, np.ones(values.shape, bool),
                              origin=[1.7, -.42, .003], voxel_size=.004)
        for node in field.nodes:
            if node.cell < 0:
                continue
            origin = field.cell_origins[node.cell]
            scalar = field.tighten(_Node(origin, origin + field.voxel_size,
                                         cell=node.cell))
            np.testing.assert_array_equal(node.lower, scalar.lower)
            np.testing.assert_array_equal(node.upper, scalar.upper)

    def test_intersection_requires_a_common_supported_point(self):
        field = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                              origin=[0.] * 3, voxel_size=.01)
        result = query_distance(field, np.eye(4), field, np.eye(4))
        self.assertEqual(result["status"], "intersection")
        for field, shift in ((field, 0.), (field, 0.)):
            point = np.asarray(result["witness_first"]) - [shift, 0., 0.]
            value, valid, _, error = field.sample(point)
            self.assertTrue(valid)
            self.assertLess(value, -error)

    def test_rigid_change_of_reference_preserves_classification(self):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", [.7, -.4, .2]).as_matrix()
        pose[:3, 3] = [1.5, -.8, .9]
        other = np.eye(4)
        other[0, 3] = .09
        result = query_distance(self.field, pose, self.field, pose @ other)
        self.assertEqual(result["status"], "clear")

    def test_zero_samples_remain_closed_occupancy(self):
        field = ObservedField(np.zeros((2, 2, 2)), np.ones((2, 2, 2), bool),
                              origin=[0.] * 3, voxel_size=.01)
        pose = np.eye(4)
        pose[0, 3] = .011
        result = query_distance(field, np.eye(4), field, pose)
        self.assertEqual(result["status"], "clear")
        self.assertEqual(query_distance(field, np.eye(4), field, np.eye(4))["status"],
                         "intersection")

    def test_closed_supported_cell_border(self):
        observed = np.ones((3, 2, 2), bool)
        observed[2] = False
        field = ObservedField(-np.ones((3, 2, 2)), observed, origin=[0.] * 3, voxel_size=1.)
        self.assertTrue(field.sample(np.array([1., .5, .5]))[1])
        self.assertFalse(field.sample(np.array([1.01, .5, .5]))[1])

    def test_unknown_never_becomes_positive_free_values(self):
        field = ObservedField(np.ones((2, 2, 2)), np.zeros((2, 2, 2), bool),
                              origin=[0.] * 3, voxel_size=1.)
        value, valid, _, _ = field.sample(np.array([.5] * 3))
        self.assertFalse(valid)
        self.assertTrue(np.isnan(value))
        result = query_distance(field, np.eye(4), self.field, np.eye(4))
        self.assertEqual(result["scope"], "represented_observed_occupancy")

    def test_serialization_retains_raw_samples_weights_and_support(self):
        observed = np.ones(self.values.shape, bool)
        observed[0, 0, 0] = False
        field = FrozenObservedField(self.values, observed, weights=observed.astype(float) * 3,
                                    origin=[-.05] * 3, voxel_size=.1 / 6,
                                    name="entity", source="calibrated_rgbd")
        restored = FrozenObservedField.from_record(field.record())
        for name in ("values", "weights", "observed", "origin"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(field, name))
        self.assertFalse(restored.values.flags.writeable)

    def test_partial_permission_does_not_skip_the_rest_of_the_pair(self):
        normals = np.r_[np.eye(3), -np.eye(3)]
        half = (normals, np.array([0., 1., 1., 1., 1., 1.]))
        region = PairPermission([(half, half)])
        result = query_distance(self.field, np.eye(4), self.field, np.eye(4),
                                permission=region)
        self.assertEqual(result["status"], "intersection")
        self.assertGreater(result["witness_first"][0], 0.)

    def test_fully_permitted_pair_is_removed_from_the_query_domain(self):
        normals = np.r_[np.eye(3), -np.eye(3)]
        box = (normals, np.ones(6))
        result = query_distance(self.field, np.eye(4), self.field, np.eye(4),
                                permission=PairPermission([(box, box)]))
        self.assertEqual(result["status"], "clear")
        self.assertEqual(result["queries"], 0)

    def test_penetration_evidence_and_fixed_material_objective_are_separate(self):
        import torch
        from manipulation.observed_collision import pair_residual
        from manipulation.sampled_geometry_objective import MaterialSamples, pair_objective
        first = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        second = torch.tensor([.07, 0., 0.], dtype=torch.float64)
        quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
        target = np.eye(4)
        target[0, 3] = .07
        residual, record = pair_residual(
            self.field, (first, quaternion), np.eye(4),
            self.field, (second, quaternion), target)
        self.assertEqual(record["status"], "intersection")
        self.assertEqual(residual.item(), 1.)
        # BEGIN the finite original-material scalar supplies its own derivative.
        samples = MaterialSamples(self.field)
        residual.backward()
        self.assertEqual(torch.count_nonzero(first.grad).item(), 0)
        first.grad.zero_()
        pair_objective(samples, (first, quaternion), samples, (second, quaternion)).backward()
        # Descent moves the left box away from the box translated to its right.
        self.assertGreater(first.grad[0].item(), 0.)
        self.assertTrue(torch.isfinite(first.grad).all())
        # END the hard value and its zero gradient remain classification evidence.

    def test_touching_supported_faces_do_not_establish_volume_overlap(self):
        for value in (-1., 0.):
            field = ObservedField(np.full((2, 2, 2), value), np.ones((2, 2, 2), bool),
                                  origin=[0.] * 3, voxel_size=.01)
            pose = np.eye(4)
            pose[0, 3] = .01
            result = query_distance(field, np.eye(4), field, pose)
            self.assertEqual(result["status"], "unresolved")
            self.assertIsNone(result["intersection"])

    def test_shared_negative_corner_does_not_end_a_thick_overlap_search(self):
        values = np.ones((2, 2, 2))
        values[0, 0, 0] = -1.
        field = ObservedField(values, np.ones_like(values, bool), origin=[0.] * 3, voxel_size=1.)
        result = query_distance(field, np.eye(4), field, np.eye(4))
        self.assertEqual(result["status"], "intersection")
        point = np.asarray(result["witness_first"])
        self.assertTrue(np.all(point > 0) and np.all(point < 1))
        self.assertLess(field.sample(point)[0], 0.)

    def test_touching_cell_does_not_hide_another_cell_penetration(self):
        values = -np.ones((5, 4, 2))
        observed = np.zeros_like(values, bool)
        observed[:2, 2:4] = True
        observed[3:5, :2] = True
        first = ObservedField(values, observed, origin=[-.02, -.01, -.005], voxel_size=.01)
        second = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                               origin=[-.04, -.07, -.04], voxel_size=.08)
        result = query_distance(first, np.eye(4), second, np.eye(4))
        self.assertEqual(result["status"], "intersection")
        self.assertGreater(result["witness_first"][0], .01)
    # END positive collision values require represented volume overlap.


if __name__ == "__main__":
    unittest.main()
