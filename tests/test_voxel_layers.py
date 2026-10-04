"""Adversarial complete-body, source-support and selected-gradient tests."""
import unittest

import numpy as np
import torch

from manipulation.voxel_layers import FineRegion, LayeredVoxelQuery, validate_body_groups


def region(bits=None, free=None):
    shape = (21, 21, 21)
    bits = np.full(shape, 3, np.uint8) if bits is None else bits
    free = np.zeros(shape, bool) if free is None else free
    return {"pose": [0., 0., 0., 1., 0., 0., 0.], "shape_xyz": list(shape),
            "voxel_size_m": .01, "numerical_support_m": .02,
            "measurement_allowance_m": .001, "measurement_evidence": "independent calibration sample",
            "source": "current snapshot source sign and full-cell depth rays",
            "source_bits_hex": bits.tobytes().hex(),
            "raw_free_packbits_hex": np.packbits(free).tobytes().hex()}


GROUPS = ({"body_id": "arm", "sphere_indices": [0, 1], "geometry_names": ["native_arm"]},
          {"body_id": "tool", "sphere_indices": [2], "geometry_names": ["native_tool"]})


class TestFineRegion(unittest.TestCase):
    def setUp(self):
        self.spheres = torch.tensor([[[[0., 0., 0., .01], [.03, 0., 0., .01],
                                       [0., .02, 0., .01]]]], dtype=torch.float64)

    def test_complete_body_crossing_selects_coarse_only_for_that_body(self):
        fine = FineRegion(region())
        self.assertEqual(fine.select(self.spheres, GROUPS).sphere_fine.tolist(), [[[True, True, True]]])
        self.spheres[..., 1, 0] = .08
        selected = fine.select(self.spheres, GROUPS)
        self.assertEqual(selected.sphere_fine.tolist(), [[[False, False, True]]])
        self.assertEqual(selected.record()["body_ids"], ["arm", "tool"])

    def test_radius_and_measured_margin_are_in_envelope(self):
        self.spheres[..., 1, 0] = .0739
        fine = FineRegion(region())
        self.assertTrue(fine.select(self.spheres, GROUPS).body_fine[0, 0, 0])
        self.spheres[..., 1, 0] = .0741
        self.assertFalse(fine.select(self.spheres, GROUPS).body_fine[0, 0, 0])

    def test_missing_sign_requires_raw_free_and_conflict_rejects(self):
        bits = np.full((21, 21, 21), 3, np.uint8)
        free = np.zeros_like(bits, bool)
        # This sample is in arm's complete AABB between its two balls.
        bits[12, 10, 10] = 1
        self.assertFalse(FineRegion(region(bits, free)).select(self.spheres, GROUPS).body_fine[0, 0, 0])
        free[12, 10, 10] = True
        self.assertTrue(FineRegion(region(bits, free)).select(self.spheres, GROUPS).body_fine[0, 0, 0])
        bits[12, 10, 10] = 7
        self.assertFalse(FineRegion(region(bits, free)).select(self.spheres, GROUPS).body_fine[0, 0, 0])
        bits[12, 10, 10] = 2
        self.assertFalse(FineRegion(region(bits, free)).select(self.spheres, GROUPS).body_fine[0, 0, 0])

    def test_prefix_count_matches_direct_inclusive_boxes(self):
        rng = np.random.default_rng(913)
        invalid = rng.random((21, 21, 21)) < .2
        fine = FineRegion(region(np.where(invalid, 0, 3).astype(np.uint8)))
        low = rng.integers(0, 18, (100, 3))
        high = np.minimum(low+rng.integers(0, 5, (100, 3)), 20)
        expected = [int(invalid[tuple(slice(a, b+1) for a, b in zip(lo, hi))].sum())
                    for lo, hi in zip(low, high)]
        self.assertEqual(fine._box_count(torch.tensor(low), torch.tensor(high)).tolist(), expected)

    def test_translated_grid_same_decision(self):
        record = region()
        record["pose"] = [1., 2., 3., 1., 0., 0., 0.]
        transformed = self.spheres.clone()
        transformed[..., :3] += torch.tensor([1., 2., 3.])
        expected = FineRegion(region()).select(self.spheres, GROUPS)
        actual = FineRegion(record).select(transformed, GROUPS)
        self.assertTrue(torch.equal(actual.body_fine, expected.body_fine))
        self.assertTrue(torch.allclose(actual.influence_aabb, expected.influence_aabb))

    def test_mapper_contract_rejects_rotated_grid(self):
        record = region()
        record["pose"][3:] = [2**-.5, 0., 0., 2**-.5]
        with self.assertRaises(ValueError):
            FineRegion(record)

    def test_float32_support_boundary_cannot_round_an_outside_body_inward(self):
        record = region()
        record.update(voxel_size_m=float(np.float32(.0005)), numerical_support_m=.0025,
                      measurement_allowance_m=.00010263787690294776)
        spheres = torch.tensor([[[[.002448305021971464, 0., 0., .00019905743829440325]]]], dtype=torch.float32)
        body = ({"body_id": "edge", "sphere_indices": [0], "geometry_names": ["native_edge"]},)
        selected = FineRegion(record).select(spheres, body)
        self.assertFalse(selected.body_fine.item())
        exact_high = float(spheres[0, 0, 0, 0])+float(spheres[0, 0, 0, 3])+record["measurement_allowance_m"]
        self.assertGreaterEqual(selected.influence_aabb[0, 0, 0, 1, 0].item(), exact_high)

    def test_huge_integer_dimensions_cannot_overflow_the_cell_budget(self):
        record = region()
        record["shape_xyz"] = [2**32]*3
        with self.assertRaises(ValueError):
            FineRegion(record)

    def test_complete_partition_required(self):
        with self.assertRaises(ValueError):
            validate_body_groups(GROUPS[:1], 3)
        with self.assertRaises(ValueError):
            validate_body_groups((GROUPS[0], GROUPS[0]), 3)

    def test_saved_support_verification_rejects_edited_indices_without_reselection(self):
        fine = FineRegion(region())
        selected = fine.select(self.spheres, GROUPS)
        verified = fine.validate_selection_envelope(selected.influence_aabb,
            selected.index_lower, selected.index_upper)
        self.assertTrue(verified["indices_match"].all())
        self.assertTrue(verified["fine_supported"].all())
        edited = selected.index_lower.clone()
        edited[..., 0] += 1
        verified = fine.validate_selection_envelope(selected.influence_aabb, edited, selected.index_upper)
        self.assertFalse(verified["indices_match"].any())

    def test_uncertified_or_invalid_record_rejected(self):
        for key, value in (("source_bits_hex", "00"), ("measurement_evidence", ""),
                           ("measurement_allowance_m", -.1), ("numerical_support_m", 0.)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                FineRegion({**region(), key: value})


class NativeStub:
    owner_ids = frozenset(("obstacle",))
    source = "test native field"
    independent_geometry = {"solids": [], "query_gaps": []}

    def __init__(self, value, derivative):
        self.value, self.derivative = value, derivative

    def clone_for_rollout(self):
        return NativeStub(self.value, self.derivative)

    def __call__(self, spheres, *, key):
        return self.value+self.derivative*spheres[..., 0]


class TestLayeredQuery(unittest.TestCase):
    def test_measured_allowance_rounds_outward_and_selection_contains_actual_query(self):
        query = LayeredVoxelQuery(NativeStub(0., 1.), NativeStub(0., 1.), region())
        spheres = torch.tensor([[[[.01, 0., 0., .015843675]]]], requires_grad=True)
        inflated = query.inflated_spheres(spheres)
        exact = float(spheres[0, 0, 0, 3].detach())+query.region.margin
        self.assertGreaterEqual(float(inflated[0, 0, 0, 3].detach()), exact)
        selected = query.region.select(spheres,
            ({"body_id": "whole", "sphere_indices": [0], "geometry_names": ["native"]},))
        self.assertGreaterEqual(selected.influence_aabb[0, 0, 0, 1, 0].item(),
                                float(spheres[0, 0, 0, 0].detach())+float(inflated[0, 0, 0, 3].detach()))
        inflated[..., 3].sum().backward()
        self.assertEqual(spheres.grad[..., 3].item(), 1.)

    def test_shared_choice_selects_one_value_and_its_gradient(self):
        spheres = torch.tensor([[[[0., 0., 0., .01], [.08, 0., 0., .01],
                                  [0., .02, 0., .01]]]], requires_grad=True)
        query = LayeredVoxelQuery(NativeStub(10., 2.), NativeStub(0., 3.), region())
        selected = query.region.select(spheres, GROUPS)
        values = query(spheres, key="actor", selection=selected)
        self.assertTrue(torch.allclose(values, torch.tensor([[[10., 10.16, 0.]]])))
        values.sum().backward()
        self.assertEqual(spheres.grad[..., 0].tolist(), [[[2., 2., 3.]]])
        cloned = query.clone_for_rollout()
        self.assertIsNot(query.coarse, cloned.coarse)
        self.assertIs(query.region, cloned.region)


if __name__ == "__main__":
    unittest.main()
