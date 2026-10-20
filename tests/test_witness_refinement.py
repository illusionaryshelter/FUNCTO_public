"""Small CPU checks for source-safe overlap and bounded witness remapping."""
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from manipulation.planning_observation import _FrozenBackgroundRefinement
from manipulation.planning_scene import ObservedSpace, PlanningScene
from manipulation.voxel_layers import FineRegion, LayeredVoxelQuery, MAX_FINE_CELLS
from test_voxel_layers import NativeStub, region


BODY = ({"body_id": "whole", "sphere_indices": [0, 1], "geometry_names": ["native"]},)


def supported_query():
    query = LayeredVoxelQuery(NativeStub(10., 2.), NativeStub(1., 3.), region())
    query.region.prepare_guidance_boxes(np.array([[[-.02, -.02, -.02], [.02, .02, .02]]]))
    return query


class SourceSafeOverlapTests(unittest.TestCase):
    def test_complete_body_weight_and_true_overlap_gradient(self):
        query = supported_query()
        spheres = torch.tensor([[[[.06, 0., 0., .01], [.04, 0., 0., .01]]]],
                               dtype=torch.float64, requires_grad=True)
        weights, bodies = query.region.guidance_weights(spheres, BODY)
        self.assertTrue(0 < bodies.item() < 1)
        self.assertEqual(weights[0, 0, 0].item(), weights[0, 0, 1].item())
        value = query.background_guidance(spheres, key="cpu", groups=BODY).sum()
        gradient, = torch.autograd.grad(value, spheres)
        step = 1e-7
        plus, minus = spheres.detach().clone(), spheres.detach().clone()
        plus[0, 0, 0, 0] += step
        minus[0, 0, 0, 0] -= step
        difference = (query.background_guidance(plus, key="cpu", groups=BODY).sum()
                      - query.background_guidance(minus, key="cpu", groups=BODY).sum())/(2*step)
        self.assertAlmostEqual(gradient[0, 0, 0, 0].item(), difference.item(), places=5)
        # The derivative through the overlap matters, not just field blending.
        self.assertGreater(abs(gradient[0, 0, 0, 0].item()), 10.)

    def test_zero_weight_and_gradient_outside_supported_box(self):
        query = supported_query()
        spheres = torch.tensor([[[[0., 0., 0., .01], [.09, 0., 0., .01]]]],
                               dtype=torch.float64, requires_grad=True)
        weights, _ = query.region.guidance_weights(spheres, BODY)
        self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))
        derivative, = torch.autograd.grad(weights.sum(), spheres)
        self.assertTrue(torch.equal(derivative, torch.zeros_like(derivative)))
        self.assertFalse(query.region.select(spheres, BODY).body_fine.item())

    def test_positive_weight_implies_complete_source_support(self):
        query = supported_query()
        generator = torch.Generator().manual_seed(711)
        centers = (torch.rand(1, 400, 2, 3, generator=generator, dtype=torch.float64)-.5)*.18
        radii = torch.full((1, 400, 2, 1), .005, dtype=torch.float64)
        spheres = torch.cat((centers, radii), -1)
        _, weights = query.region.guidance_weights(spheres, BODY)
        selected = query.region.select(spheres, BODY)
        self.assertTrue(weights.gt(0).any())
        self.assertTrue((~weights.gt(0) | selected.body_fine).all())

    def test_missing_sign_does_not_create_supported_cuboid(self):
        bits = np.full((21, 21, 21), 3, np.uint8)
        bits[10, 10, 10] = 1
        fine = FineRegion(region(bits))
        report = fine.prepare_guidance_boxes(np.array([[[-.02]*3, [.02]*3]]))
        self.assertEqual(report["cuboids"], 0)
        self.assertEqual(report["unsupported_witnesses"], 1)

    # BEGIN exact cuboid reuse must leave capacity for later distinct support.
    def test_different_boundary_witnesses_with_identical_boxes_do_not_use_capacity(self):
        bits = np.full((21, 21, 21), 3, np.uint8)
        bits[10] = 1  # Separate supported left/right regions with an unknown plane.
        fine = FineRegion(region(bits))
        guard = 32*np.finfo(np.float32).eps*(1.+float((np.asarray(fine.shape)*fine.size).max()))
        # Each left witness is strictly supported, but lies inside the guard
        # of the same maximal box's upper x face. They deliberately differ;
        # removing repeated input envelopes cannot solve this capacity fault.
        left = np.array([[[-.06, -.01, -.01], [-.02-guard*fraction, .01, .01]]
                         for fraction in np.linspace(.25, .75, 17)])
        right = np.array([[[.03, -.01, -.01], [.06, .01, .01]]])
        report = fine.prepare_guidance_boxes(np.concatenate((left, right)), maximum_boxes=2)
        self.assertEqual(report["source_supported_witnesses"], 18)
        self.assertEqual(report["cuboids"], 2)
        self.assertEqual(report["duplicate_cuboid_candidates"], 16)
        self.assertTrue(fine.guidance_boxes[0, 1, 0] < 0 < fine.guidance_boxes[1, 0, 0])
        self.assertTrue(np.all(right[0, 0] > fine.guidance_boxes[1, 0]))
        self.assertTrue(np.all(right[0, 1] < fine.guidance_boxes[1, 1]))
        self.assertTrue(fine._envelope_support(torch.from_numpy(fine.guidance_boxes.copy()))[0].all())

    def test_covered_narrower_witness_can_expand_to_a_different_box(self):
        bits = np.full((21, 21, 21), 3, np.uint8)
        bits[10, 15] = 1
        fine = FineRegion(region(bits))
        envelopes = np.array([[[-.06, -.01, -.01], [-.04, .07, .01]],
                              [[-.06, -.01, -.01], [-.04, .01, .01]]])
        report = fine.prepare_guidance_boxes(envelopes, maximum_boxes=2)
        self.assertEqual(report["cuboids"], 2)
        first, second = fine.guidance_boxes
        self.assertTrue(np.all(envelopes[1, 0] > first[0]))
        self.assertTrue(np.all(envelopes[1, 1] < first[1]))
        # The narrow witness's x expansion crosses x=0 below the unknown
        # strip at y=.05; the initial tall witness cannot cross that strip.
        self.assertLess(first[1, 0], 0.)
        self.assertGreater(second[1, 0], 0.)
        self.assertLess(second[1, 1], first[1, 1])
        self.assertTrue(fine._envelope_support(torch.from_numpy(fine.guidance_boxes.copy()))[0].all())
    # END final coordinate equality preserves every distinct candidate cuboid.

    # BEGIN the original selector needs exactly one interpolation sample per side.
    def test_supported_sample_boundary_is_not_eroded_twice(self):
        bits = np.full((21, 21, 21), 3, np.uint8)
        bits[10] = 1
        fine = FineRegion(region(bits))
        groups = ({"body_id": "probe", "sphere_indices": [0], "geometry_names": ["native"]},)
        spheres = torch.tensor([[[[-.04, 0., 0., .014]]]], dtype=torch.float64)
        selection = fine.select(spheres, groups)
        self.assertTrue(selection.body_fine.item())
        report = fine.prepare_guidance_boxes(selection.influence_aabb.reshape(-1, 2, 3).numpy())
        self.assertEqual(report["supported_uncovered_envelopes"], 0)
        self.assertTrue(fine.guidance_weights(spheres, groups)[1].item() > 0)
        # This envelope reaches x=-.025: it fits the strict - .02 source
        # boundary, but the former second erosion incorrectly ended at -.03.
        self.assertGreater(fine.guidance_boxes[0, 1, 0], -.025)
        self.assertTrue(fine._envelope_support(torch.from_numpy(fine.guidance_boxes.copy()))[0].all())

        # Probe both sides of every guarded face using actual float32/float64
        # sphere arithmetic. Positive guidance must always retain the exact
        # source selector; crossing the missing-sign plane must stay coarse.
        for dtype in (torch.float32, torch.float64):
            box = fine.guidance_boxes[0]
            probes = []
            for axis in range(3):
                for side in (0, 1):
                    sign = 1 if side == 0 else -1
                    for offset in (-1e-5, -1e-7, 0., 1e-7, 1e-5):
                        center = box.mean(0)
                        center[axis] = box[side, axis]+sign*(fine.margin+offset)
                        probes.append([*center, 0.])
            probes.append([-.015, 0., 0., .01])
            sample = torch.tensor(probes, dtype=dtype)[None, :, None, :]
            selected = fine.select(sample, groups).body_fine
            weights = fine.guidance_weights(sample, groups)[1]
            self.assertTrue(weights.gt(0).any())
            self.assertTrue((~weights.gt(0) | selected).all())
            self.assertFalse(selected[0, -1].item())
            self.assertEqual(weights[0, -1].item(), 0.)
    # END strict support is retained at faces and across unsupported observations.

    def test_overlap_value_and_derivative_vanish_at_boundary(self):
        query = supported_query()
        boundary = query.region.guidance_boxes[0, 1, 0]-.011
        weights, derivatives = [], []
        for offset in (-1e-7, 0., 1e-7):
            spheres = torch.tensor([[[[boundary+offset, 0., 0., .01], [0., 0., 0., .01]]]],
                                   dtype=torch.float64, requires_grad=True)
            _, weight = query.region.guidance_weights(spheres, BODY)
            derivative, = torch.autograd.grad(weight.sum(), spheres)
            weights.append(weight.item()); derivatives.append(derivative.abs().max().item())
        self.assertLess(max(weights), 1e-9)
        self.assertLess(max(derivatives), .001)


class BoundedRefinementTests(unittest.TestCase):
    def source(self):
        views = {"camera": {"image": np.zeros((3, 3, 3), np.uint8),
            "depth": np.ones((3, 3)), "K": np.eye(3), "T_output_camera": np.eye(4)}}
        masks = {"camera": {None: np.ones((3, 3), bool)}}
        source = _FrozenBackgroundRefinement(views, masks,
            SimpleNamespace(device="cuda:0"),
            {"voxel_size_m": .01, "measurement_allowance_m": .001,
             "measurement_evidence": "calibration", "source": "raw depth"}, .001,
            # BEGIN refinements retain the explicit original request limits.
            max_mapping_bytes=1024**3, max_host_rss_bytes=8*1024**3,
            max_device_used_bytes=6*1024**3)
            # END no refinement silently restores an unrelated default budget.
        return source, views, masks

    def witnesses(self, extent=.02):
        return [{"body_id": "whole", "batch_index": 0, "time_s": 1.,
                 "coarse_cost": 1., "positive_spheres": 2,
                 "envelope_world_m": [[-extent]*3, [extent]*3]}]

    def test_snapshot_is_independent_and_read_only(self):
        source, views, masks = self.source()
        views["camera"]["depth"][:] = 3
        masks["camera"][None][:] = False
        self.assertTrue((source.views["camera"]["depth"] == 1).all())
        self.assertTrue(source.masks["camera"][None].all())
        with self.assertRaises(ValueError):
            source.views["camera"]["depth"][0, 0] = 2

    def test_oversized_union_refused_before_mapper_or_device_access(self):
        source, _, _ = self.source()
        with patch("manipulation.planning_observation._map_partition") as mapper, \
             patch("manipulation.planning_observation._require_mapping_memory") as device:
            replacement, report = source.refine(supported_query(), self.witnesses(5.))
        self.assertIsNone(replacement)
        self.assertEqual(report["status"], "local_cell_budget_refused")
        self.assertGreater(report["requested_cells"], MAX_FINE_CELLS)
        mapper.assert_not_called(); device.assert_not_called()

    def test_mapping_byte_budget_refused_before_allocation(self):
        source, _, _ = self.source()
        with patch("manipulation.planning_observation._fine_mapper_config", return_value=source.config), \
             patch("manipulation.planning_observation._mapper_memory_bound", return_value=1024**3+1), \
             patch("manipulation.planning_observation._map_partition") as mapper:
            # Mapping admission is needed only when the requested envelope
            # genuinely enlarges the already-mapped physical region.
            replacement, report = source.refine(supported_query(), self.witnesses(.2))
        self.assertIsNone(replacement)
        self.assertEqual(report["status"], "mapping_byte_budget_refused")
        mapper.assert_not_called()

    # BEGIN expanded maps inherit the request budget instead of a one-GiB cap.
    def test_expansion_above_one_gib_reaches_same_request_total_admission(self):
        from manipulation.resource_admission import ResourceLimitError
        _, views, masks = self.source()
        source = _FrozenBackgroundRefinement(views, masks, SimpleNamespace(device="cuda:0"),
            {"voxel_size_m": .01, "measurement_allowance_m": .001,
             "measurement_evidence": "calibration", "source": "raw depth"}, .001,
            max_mapping_bytes=2*1024**3, max_host_rss_bytes=12*1024**3,
            max_device_used_bytes=8*1024**3)
        estimate = 1024**3+1
        with patch("manipulation.planning_observation._fine_mapper_config", return_value=source.config), \
                patch("manipulation.planning_observation._mapper_memory_bound", return_value=estimate), \
                patch("manipulation.planning_observation._require_mapping_memory",
                      side_effect=ResourceLimitError("host_process_rss", 13*1024**3,
                                                     12*1024**3)) as admission, \
                patch("manipulation.planning_observation._map_partition") as mapper:
            replacement, report = source.refine(supported_query(), self.witnesses(.2))
        self.assertIsNone(replacement)
        self.assertEqual(report["status"], "live_mapping_memory_refused")
        self.assertEqual(report["max_mapping_bytes"], 2*1024**3)
        admission.assert_called_once_with(estimate, "cuda:0",
            max_host_rss_bytes=12*1024**3, max_device_used_bytes=8*1024**3)
        mapper.assert_not_called()
    # END refusal preserves the existing scene and its already allocated fields.

    # BEGIN reproduce native float32 spacing without allocating a native map.
    def native_query(self, shape, pose):
        shape = tuple(shape)
        record = region()
        record.update(shape_xyz=list(shape), pose=[*pose, 1., 0., 0., 0.],
            voxel_size_m=float(np.float32(.002)), numerical_support_m=.01,
            measurement_allowance_m=.0001,
            source_bits_hex=np.full(shape, 3, np.uint8).tobytes().hex(),
            raw_free_packbits_hex=np.packbits(np.zeros(shape, bool)).tobytes().hex())
        query = LayeredVoxelQuery(NativeStub(10., 2.), NativeStub(1., 3.), record)
        query.mapping_report = {}
        return query

    def test_a004_native_grids_do_not_grow_or_remap_for_contained_witnesses(self):
        source, _, _ = self.source()
        source.request["voxel_size_m"] = .002
        # Exact archived A004 shapes/poses. In particular, the old formula
        # changed (170,140,127)->(171,141,128)->(172,142,129) even when no
        # padded witness extended the physical native bounds at all.
        cases = (((169, 139, 125), .9954607486724854),
                 ((170, 140, 127), .996846616268158),
                 ((171, 141, 128), .996846616268158),
                 ((172, 142, 129), .996846616268158))
        for shape, z in cases:
            with self.subTest(shape=shape):
                pose = np.array([-.06572640687227249, -.14608871936798096, z])
                query = self.native_query(shape, pose)
                witnesses = self.witnesses()
                witnesses[0]["envelope_world_m"] = np.stack((pose-.005, pose+.005)).tolist()
                for _ in range(3):
                    with patch("manipulation.planning_observation._fine_mapper_config") as config, \
                         patch("manipulation.planning_observation._map_partition") as mapper, \
                         patch("manipulation.planning_observation._require_mapping_memory") as device:
                        replacement, report = source.refine(query, witnesses)
                    self.assertTrue(report["mapping_reused"])
                    self.assertEqual(report["requested_cells"], math.prod(shape))
                    self.assertEqual(report["shape_xyz"], list(shape))
                    self.assertEqual(report["center_m"], pose.tolist())
                    config.assert_not_called(); mapper.assert_not_called(); device.assert_not_called()
                    if replacement is not None:
                        self.assertEqual(replacement.region.record, query.region.record)
                        query = replacement

    def test_real_expansion_preserves_other_axes_and_covers_quantized_native_bounds(self):
        source, _, _ = self.source()
        source.request["voxel_size_m"] = .002
        shape = (41, 43, 45)
        pose = np.array([-.06572640687227249, -.14608871936798096, .996846616268158])
        # A genuine 0.1 micrometre growth must not be erased by the previous
        # size*.01 allclose tolerance. A larger case exercises center rounding.
        for growth in (1e-7, .0031):
            with self.subTest(growth=growth):
                query = self.native_query(shape, pose)
                half = np.asarray(shape)*query.region.size/2
                padding = query.region.support+6*query.region.size
                envelope = np.stack((pose-.002, pose+.002))
                envelope[1, 2] = pose[2]+half[2]-padding+growth
                witnesses = self.witnesses()
                witnesses[0]["envelope_world_m"] = envelope.tolist()

                def mapped(*args):
                    # Match the installed Mapper's independent rules: Python
                    # spacing determines ceil(shape), native float32 spacing
                    # and center determine the published physical geometry.
                    request = args[-1]
                    native_shape = tuple(math.ceil(value/request["voxel_size_m"])
                                         for value in request["extent_m"])
                    native_pose = np.asarray(request["center_m"], dtype=np.float32).astype(float)
                    replacement = self.native_query(native_shape, native_pose)
                    return replacement.fine, {}, replacement.region.record

                with patch("manipulation.planning_observation._fine_mapper_config",
                           return_value=source.config) as config, \
                     patch("manipulation.planning_observation._mapper_memory_bound", return_value=1000), \
                     patch("manipulation.planning_observation._map_partition", side_effect=mapped) as mapper, \
                     patch("manipulation.planning_observation._require_mapping_memory"):
                    replacement, report = source.refine(query, witnesses)
                    self.assertIsNotNone(replacement)
                    self.assertFalse(report["mapping_reused"])
                    self.assertEqual(replacement.region.shape[:2], shape[:2])
                    self.assertGreater(replacement.region.shape[2], shape[2])
                    np.testing.assert_array_equal(replacement.region.pose[:2], pose[:2])
                    new_half = np.asarray(replacement.region.shape)*replacement.region.size/2
                    required_low = np.minimum(pose-half, envelope[0]-padding)
                    required_high = np.maximum(pose+half, envelope[1]+padding)
                    self.assertTrue(np.all(replacement.region.pose[:3]-new_half <= required_low))
                    self.assertTrue(np.all(replacement.region.pose[:3]+new_half >= required_high))
                    self.assertEqual(list(replacement.region.shape), report["shape_xyz"])
                    # Repeating this same newly-contained witness must reuse
                    # the just-published field, with no second Mapper/config.
                    _, repeated = source.refine(replacement, witnesses)
                    self.assertTrue(repeated["mapping_reused"])
                    self.assertEqual(repeated["shape_xyz"], report["shape_xyz"])
                    self.assertEqual(repeated["center_m"], report["center_m"])
                    mapper.assert_called_once(); config.assert_called_once()
    # END real coverage drives growth; nominal/native conversion cannot do so.

    def test_new_supported_witness_in_same_roi_adds_box_without_remapping(self):
        source, _, _ = self.source()
        shape = (41, 41, 41)
        bits = np.full(shape, 3, np.uint8)
        # A missing-sign plane separates two source-supported components.
        bits[20] = 1
        record = region()
        record.update(shape_xyz=list(shape), source_bits_hex=bits.tobytes().hex(),
                      raw_free_packbits_hex=np.packbits(np.zeros(shape, bool)).tobytes().hex())
        query = LayeredVoxelQuery(NativeStub(10., 2.), NativeStub(1., 3.), record)
        query.region.prepare_guidance_boxes(np.array([[[-.08, -.02, -.02], [-.06, .02, .02]]]))
        query.mapping_report = {}
        old = query.region.guidance_boxes.copy()
        witnesses = self.witnesses()
        witnesses[0]["envelope_world_m"] = [[.06, -.02, -.02], [.08, .02, .02]]
        with patch("manipulation.planning_observation._fine_mapper_config", return_value=source.config), \
             patch("manipulation.planning_observation._mapper_memory_bound", return_value=1000), \
             patch("manipulation.planning_observation._map_partition") as mapper, \
             patch("manipulation.planning_observation._require_mapping_memory") as device:
            replacement, report = source.refine(query, witnesses)
        self.assertIsNotNone(replacement)
        self.assertTrue(report["mapping_reused"])
        self.assertEqual(len(replacement.region.guidance_boxes), 2)
        self.assertTrue(np.array_equal(replacement.region.guidance_boxes[0], old[0]))
        self.assertEqual(len(query.region.guidance_boxes), 1)
        mapper.assert_not_called(); device.assert_not_called()

    def test_scene_replacement_retains_policy_and_original_query(self):
        query = supported_query()
        replacement = supported_query()
        class Source:
            storage_bytes = 0
            def refine(self, current, witnesses):
                self.current = current
                return replacement, {"status": "refined"}
        query.refinement_source = Source()
        scene = PlanningScene(background=query, owners=(),
            coverage=ObservedSpace(np.ones((5, 5, 5), bool), origin=[-1.]*3,
                voxel_size=.5, observation_ids=["camera"], reference_frame="world"),
            reference_frame="world", robot_sphere_materials=(None, None),
            robot_geometry_source="complete", robot_collision_bodies=BODY, has_gripper=False)
        spheres = torch.tensor([[[[.06, 0., 0., .01], [.04, 0., 0., .01]]]], dtype=torch.float64)
        witnesses = scene.background_refinement_witnesses(spheres, torch.zeros(1, 1, 3),
            torch.tensor([[[1., 0., 0., 0.]]]), torch.zeros(1, 1), trajectory_time=torch.ones(1, 1))
        self.assertEqual(len(witnesses), 1)
        self.assertEqual(witnesses[0]["positive_spheres"], 2)
        changed, report = scene.refine_background(witnesses)
        self.assertEqual(report["status"], "refined")
        self.assertIs(scene.background, query)
        self.assertIs(changed.background, replacement)
        self.assertTrue(np.array_equal(changed.coverage.known_free, scene.coverage.known_free))
        self.assertEqual(changed.robot_collision_bodies, scene.robot_collision_bodies)


if __name__ == "__main__":
    unittest.main()
