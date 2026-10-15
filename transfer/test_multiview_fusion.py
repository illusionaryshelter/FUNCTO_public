"""End-to-end CPU checks for observed geometry plus complete-response fusion.

These tests use tiny fixed FP32 feature maps, actual source descriptor sampling,
actual camera projection/plane association, and the complete fusion function.
Only call-observation spies are patched; no model, renderer, or GPU is loaded.
"""

import copy
import unittest
from unittest import mock

import numpy as np

from transfer import multiview_features as fusion
from transfer.segmentation import prepare_observation


# BEGIN integrated multiview feature/geometry contract regressions.
def _view(view_id, *, depth_value=1., transform=None, source=False):
    """Prepare one calibrated observed patch through the real SAM boundary."""
    image = np.zeros((20, 24, 3), dtype=np.uint8)
    depth = np.full(image.shape[:2], depth_value)
    mask = np.zeros(depth.shape, bool)
    mask[6:14, 8:16] = True
    K = np.array([[600., 7., 11.5], [0., 650., 9.5], [0., 0., 1.]])
    T = np.eye(4) if transform is None else np.asarray(transform)
    evidence = None
    if source:
        evidence = np.zeros(depth.shape, bool)
        evidence[8:12, 10:14] = True
    detections = [{"label": "container", "score": .8,
                   "bbox": [8., 6., 16., 14.], "mask": mask}]
    prepared = prepare_observation(image, depth, K, T, detections,
        prompt="container", source_region_mask=evidence)
    assert prepared["prepared"]
    return {"id": view_id, "observation": prepared["observation"],
            "depth_m": depth, "selected_mask": mask,
            "crop_xyxy": prepared["crop_xyxy"],
            "feature": np.zeros((1, 1, 8, 8), dtype=np.float32)}


def _empty_view(view_id):
    target = _view(view_id)
    target["depth_m"][:] = np.nan
    target["observation"]["xyz"] = np.empty((0, 3))
    target["observation"]["pixels"] = np.empty((0, 2), dtype=np.int64)
    for field in ("entity_member", "region_mask", "evidence_only"):
        target["observation"][field] = np.empty(0, dtype=bool)
    return target


def _missing_view(view_id):
    """Keep the camera/full depth while explicitly declaring no SAM observation."""
    target = _empty_view(view_id)
    target["depth_m"][:] = 1.
    target["selected_mask"][:] = False
    target["observation"]["object_mask"][:] = False
    target["feature"] = None
    return target


def _run(source, source_map, views, *, count=7):
    return fusion.fuse_correspondences(source["observation"], source_map,
        source["crop_xyxy"], views, count=count)


class TestCompleteMultiviewFusion(unittest.TestCase):
    def test_real_source_sampling_native_target_inverse_and_view_order(self):
        import torch
        from local_models.marco import _geometry, resize_scales, RESOLUTION

        T = np.array([[0., -1., 0., .3], [1., 0., 0., -.2],
                      [0., 0., 1., .5], [0., 0., 0., 1.]])
        source = _view("source", source=True, transform=T)
        a = _view("a", transform=T)
        second_pose = T.copy()
        second_pose[:3, 3] += T[:3, :3] @ [.0043, 0., 0.]
        b = _view("b", transform=second_pose)
        rng = np.random.default_rng(713)
        source_map = rng.normal(size=(1, 3, 8, 8)).astype(np.float32)
        a["feature"] = rng.normal(size=(1, 3, 8, 8)).astype(np.float32)
        b["feature"] = rng.normal(size=(1, 3, 8, 8)).astype(np.float32)
        with mock.patch.object(fusion, "sample_descriptors", wraps=fusion.sample_descriptors) as sampler:
            with mock.patch.object(fusion, "rank_sampling_plan", wraps=fusion.rank_sampling_plan) as plans:
                first = _run(source, source_map, [b, a])
        second = _run(source, source_map, [a, b])
        for name, value in first.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, second[name], err_msg=name)
        for name, value in first["bank"].items():
            np.testing.assert_array_equal(value, second["bank"][name], err_msg=name)
        self.assertEqual(sampler.call_count, 1)
        source_pixels = source["observation"]["pixels"][first["raw_source_idx"]]
        np.testing.assert_array_equal(sampler.call_args.args[1],
            source_pixels - source["crop_xyxy"][:2])
        self.assertTrue(np.all(source["crop_xyxy"][:2] > 0))

        # The independent native output map must invert the coordinates sent
        # into rank interpolation. This catches using the source input Wf/840
        # convention, swapping xy/HW, or forgetting the crop-origin addition.
        for view_index, view in enumerate((a, b)):
            feature_xy = plans.call_args_list[view_index].args[0]
            resized = _geometry().scaling_coordinates(torch.from_numpy(feature_xy),
                view["feature"].shape[-2:], (RESOLUTION, RESOLUTION)).numpy()
            left, top, right, bottom = view["crop_xyxy"]
            original = resized / resize_scales((right-left, bottom-top)) + [left, top]
            np.testing.assert_allclose(original, first["candidate_projection_pixels"][:, view_index],
                                       atol=1e-12, rtol=0)
        self.assertGreater(first["diagnostics"]["candidate_supported_by_both_views"], 0)

    def test_actual_occlusion_unknown_is_zero_and_denominator_stays_two(self):
        source = _view("source", source=True)
        a, b = _view("a"), _view("b", depth_value=.99)
        # Every raw dot is negative, yet the ordered response remains usable.
        # The other camera's constant response gives exactly zero rank evidence.
        source_map = np.ones((1, 1, 8, 8), dtype=np.float32)
        a["feature"] = np.arange(64, dtype=np.float32).reshape(1, 1, 8, 8) - 100
        b["feature"].fill(17.)
        result = _run(source, source_map, [a, b])
        self.assertTrue(result["raw_valid"].all())
        self.assertTrue(np.all(result["raw_dot_ranges"][:, 0, 1] < 0))
        np.testing.assert_array_equal(result["raw_view_supported"][:, 0], True)
        np.testing.assert_array_equal(result["raw_view_supported"][:, 1], False)
        np.testing.assert_array_equal(result["raw_view_scores"][:, 1], 0.)
        np.testing.assert_allclose(result["raw_scores"], result["raw_view_scores"][:, 0] / 2,
                                   atol=0, rtol=0)
        state_names = result["diagnostics"]["views"][1]["state_names"]
        behind_code = state_names.index("behind_observed_surface")
        self.assertTrue(np.any(result["candidate_view_states"][:, 1] == behind_code))
        # Identical source descriptors all select this one real candidate; only
        # the first original query survives duplicate-target resolution.
        self.assertEqual(len(result["src_anchor_idx"]), 1)
        self.assertEqual(result["src_anchor_idx"][0], result["raw_source_idx"][0])
        np.testing.assert_array_equal(result["raw_fitting_row"], [0, -1, -1, -1, -1, -1, -1])

    def test_exact_candidate_union_preserves_fitting_to_observation_mapping(self):
        source = _view("source", source=True)
        a, b = _view("a"), _view("b")
        rng = np.random.default_rng(20)
        source_map = rng.normal(size=(1, 3, 8, 8)).astype(np.float32)
        a["feature"] = rng.normal(size=(1, 3, 8, 8)).astype(np.float32)
        b["feature"] = rng.normal(size=(1, 3, 8, 8)).astype(np.float32)
        result = _run(source, source_map, [a, b])
        bank = result["bank"]
        self.assertEqual(len(bank["xyz"]), 64)
        self.assertEqual(len(bank["observation_xyz"]), 128)
        self.assertGreater(len(result["src_anchor_idx"]), 0)
        np.testing.assert_array_equal(bank["observation_candidate_index"][:64],
                                      bank["observation_candidate_index"][64:])
        retained = np.flatnonzero(result["raw_retained"])
        fitting_order = retained[np.argsort(result["raw_fitting_row"][retained])]
        np.testing.assert_array_equal(result["src_anchor_idx"], result["raw_source_idx"][fitting_order])
        np.testing.assert_array_equal(result["tgt_anchor_idx"], result["raw_target_idx"][fitting_order])
        for name in ("src_anchor_idx", "tgt_anchor_idx"):
            self.assertEqual(len(np.unique(result[name])), len(result[name]))
        candidate_ids = result["tgt_anchor_idx"]
        observation_ids = bank["candidate_first_observation_index"][candidate_ids]
        np.testing.assert_array_equal(bank["xyz"][candidate_ids], bank["observation_xyz"][observation_ids])

    def test_distinct_world_locations_with_equal_positive_peaks_abstain(self):
        source = _view("source", source=True)
        a = _view("a")
        other_pose = np.eye(4)
        other_pose[0, 3] = 1.
        b = _view("b", transform=other_pose)
        source_map = np.ones((1, 1, 8, 8), dtype=np.float32)
        a["feature"] = np.arange(64, dtype=np.float32).reshape(1, 1, 8, 8)
        b["feature"] = a["feature"].copy()
        result = _run(source, source_map, [a, b])
        self.assertTrue(np.all(result["raw_scores"] > 0))
        self.assertTrue(np.all(result["raw_tie_count"] >= 2))
        self.assertFalse(result["raw_valid"].any())
        self.assertTrue(np.all(result["raw_reasons"] == "ambiguous_fused_maximum"))
        self.assertTrue(np.all(result["raw_target_idx"] == -1))
        self.assertEqual(len(result["src_anchor_idx"]), 0)

    def test_empty_source_and_target_evidence_complete_without_fallback(self):
        source_map = np.ones((1, 1, 8, 8), dtype=np.float32)
        for empty_source, empty_targets in ((True, False), (False, True), (True, True)):
            with self.subTest(empty_source=empty_source, empty_targets=empty_targets):
                source = _view("source", source=True)
                if empty_source:
                    source["observation"]["region_mask"][:] = False
                views = ([_empty_view("a"), _empty_view("b")] if empty_targets
                         else [_view("a"), _view("b")])
                result = _run(source, source_map, views)
                self.assertEqual(len(result["src_anchor_idx"]), 0)
                self.assertEqual(result["diagnostics"]["no_source_region"], empty_source)
                self.assertEqual(result["diagnostics"]["no_observed_candidate"], empty_targets)
                if empty_targets:
                    self.assertTrue(np.all(result["raw_target_idx"] == -1))
                    self.assertTrue(np.all(result["raw_reasons"] == "no_observed_candidate"))
                if empty_source:
                    self.assertEqual(len(result["raw_source_idx"]), 0)

    # BEGIN explicit missing-view evidence without synthesized feature maps.
    def test_missing_feature_equals_empty_zero_feature_and_keeps_fixed_two(self):
        source = _view("source", source=True)
        source_map = np.ones((1, 1, 8, 8), dtype=np.float32)
        a = _view("a")
        a["feature"] = np.arange(64, dtype=np.float32).reshape(1, 1, 8, 8)
        missing = _missing_view("b")
        zero_feature = copy.deepcopy(missing)
        zero_feature["feature"] = np.zeros((1, 1, 8, 8), dtype=np.float32)
        missing["crop_xyxy"] = None  # No SAM instance means no model crop exists.
        # A real missing view should not invoke either a feature interpolation
        # plan or a plane fit. The observed view owns every candidate directly.
        with mock.patch.object(fusion, "rank_sampling_plan", wraps=fusion.rank_sampling_plan) as plans:
            with mock.patch("transfer.multiview_geometry._association_planes") as planes:
                actual = _run(source, source_map, [a, missing])
        expected = _run(source, source_map, [a, zero_feature])
        self.assertEqual(plans.call_count, 1)
        self.assertEqual(planes.call_count, 0)
        for name, value in actual.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, expected[name], err_msg=name)
        self.assertTrue(actual["raw_valid"].all())
        self.assertEqual(actual["diagnostics"]["fixed_view_denominator"], 2)
        np.testing.assert_array_equal(actual["raw_view_scores"][:, 1], 0.)
        np.testing.assert_array_equal(actual["candidate_view_supported"][:, 1], False)
        np.testing.assert_allclose(actual["raw_scores"], actual["raw_view_scores"][:, 0] / 2,
                                   atol=0, rtol=0)
        # The absent SAM result must not erase the actual camera projection or
        # replace its association status with a copied observed-camera status.
        self.assertTrue(np.isfinite(actual["candidate_projection_pixels"][:, 1]).all())
        names = actual["diagnostics"]["views"][1]["state_names"]
        self.assertTrue(np.all(actual["candidate_view_states"][:, 1]
                               == names.index("outside_selected_mask")))

    def test_both_views_explicitly_missing_reject_without_fabricated_maps(self):
        source = _view("source", source=True)
        source_map = np.ones((1, 1, 8, 8), dtype=np.float32)
        with mock.patch.object(fusion, "rank_sampling_plan", wraps=fusion.rank_sampling_plan) as plans:
            result = _run(source, source_map, [_missing_view("a"), _missing_view("b")])
        self.assertEqual(plans.call_count, 0)
        self.assertTrue(result["diagnostics"]["no_observed_candidate"])
        self.assertEqual(result["diagnostics"]["fixed_view_denominator"], 2)
        self.assertFalse(result["raw_valid"].any())
        self.assertTrue(np.all(result["raw_reasons"] == "no_observed_candidate"))
        self.assertEqual(len(result["src_anchor_idx"]), 0)
        self.assertEqual(result["candidate_projection_pixels"].shape, (0, 2, 2))

    def test_missing_feature_rejects_nonempty_observation_or_selected_mask(self):
        source = _view("source", source=True)
        source_map = np.ones((1, 1, 8, 8), dtype=np.float32)
        invalid_observation = _view("b")
        invalid_observation["feature"] = None
        invalid_mask = _missing_view("b")
        invalid_mask["selected_mask"][9, 11] = True
        invalid_mask["observation"]["object_mask"][9, 11] = True
        for invalid in (invalid_observation, invalid_mask):
            with self.subTest(observation_count=len(invalid["observation"]["xyz"])):
                with self.assertRaises(ValueError):
                    _run(source, source_map, [_view("a"), invalid])
    # END explicit missing-view evidence without synthesized feature maps.


if __name__ == "__main__":
    unittest.main()
# END integrated multiview feature/geometry contract regressions.
