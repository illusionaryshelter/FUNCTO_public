"""CPU regressions for SAM instance selection and unaltered source evidence.

All model calls are mocked. These tests exercise the input/selection boundary,
not model quality, and neither import a model nor allocate a CUDA device.
"""

# BEGIN original-pixel segmentation evidence regressions.
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from transfer.segmentation import (detect_image, load_detections,
                                   prepare_observation, store_detections)


class TestSegmentationPreparation(unittest.TestCase):
    def setUp(self):
        self.rgb = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        self.depth = np.full((8, 10), 2.0)
        self.K = np.array([[10., 0., 4.], [0., 20., 3.], [0., 0., 1.]])
        # Proper nonidentity camera pose checks that no world-axis assumption
        # slips into backprojection or the source evidence center.
        self.T = np.array([[0., -1., 0., .3], [1., 0., 0., -.2],
                           [0., 0., 1., .5], [0., 0., 0., 1.]])
        self.prompt = "container"

    def mask(self, *pixels):
        result = np.zeros(self.depth.shape, dtype=bool)
        for x, y in pixels:
            result[y, x] = True
        return result

    def detection(self, mask, score=.8, bbox=None):
        return {"label": self.prompt, "score": score,
                "bbox": [0., 0., 10., 8.] if bbox is None else bbox, "mask": mask}

    def prepare(self, detections, **kwargs):
        return prepare_observation(self.rgb, self.depth, self.K, self.T, detections,
                                   prompt=self.prompt, **kwargs)

    def test_source_selects_valid_evidence_coverage_before_score(self):
        evidence = self.mask((1, 1), (2, 1), (3, 1), (4, 1))
        self.depth[1, 4] = np.nan
        detections = [self.detection(self.mask((1, 1), (4, 1)), .99),
                      self.detection(self.mask((2, 1), (3, 1)), .1)]
        result = self.prepare(detections, source_region_mask=evidence)
        self.assertTrue(result["prepared"])
        self.assertEqual(result["selection"]["selected_index"], 1)
        self.assertEqual([r["source_evidence_covered_pixels"] for r in result["selection"]["candidates"]], [1, 2])
        self.assertEqual(result["diagnostics"]["source_evidence_invalid_depth_pixels"], 1)

    def test_partial_source_mask_retains_all_valid_evidence_and_original_sam(self):
        sam = self.mask((1, 1), (2, 1), (9, 7))
        evidence = self.mask((2, 1), (3, 1), (4, 1), (4, 2))
        original_sam, original_evidence = sam.copy(), evidence.copy()
        result = self.prepare([self.detection(sam)], source_region_mask=evidence,
                              crop_margin_fraction=0)
        observation = result["observation"]
        np.testing.assert_array_equal(result["source_evidence_mask"], original_evidence)
        np.testing.assert_array_equal(observation["object_mask"], original_sam)
        np.testing.assert_array_equal(sam, original_sam)
        np.testing.assert_array_equal(evidence, original_evidence)
        self.assertEqual(int(observation["region_mask"].sum()), 4)
        self.assertEqual(int(observation["entity_member"].sum()), 3)
        self.assertEqual(int(observation["evidence_only"].sum()), 3)
        y, x = np.nonzero(sam | evidence)
        np.testing.assert_array_equal(observation["pixels"], np.column_stack((x, y)))
        # The false-positive point at (9,7) must remain in inference; no hidden
        # owner mask can clean the selected SAM domain at this boundary.
        self.assertIn([9, 7], observation["pixels"].tolist())
        expected_camera = np.column_stack(((x-4)*.2, (y-3)*.1, np.full(len(x), 2.)))
        expected_world = np.column_stack((.3-expected_camera[:, 1],
                                          -.2+expected_camera[:, 0],
                                          .5+expected_camera[:, 2]))
        np.testing.assert_allclose(observation["xyz"], expected_world, atol=1e-14)
        np.testing.assert_allclose(result["diagnostics"]["source_evidence_center"],
                                   expected_world[evidence[y, x]].mean(0), atol=1e-14)
        np.testing.assert_array_equal(result["crop_xyxy"], [1, 1, 10, 8])

    def test_source_crop_preserves_original_evidence_even_without_its_depth(self):
        sam = self.mask((1, 1), (2, 1))
        evidence = self.mask((2, 1), (8, 6))
        self.depth[6, 8] = 0
        result = self.prepare([self.detection(sam)], source_region_mask=evidence,
                              crop_margin_fraction=0)
        np.testing.assert_array_equal(result["crop_xyxy"], [1, 1, 9, 7])
        self.assertEqual(int(result["observation"]["region_mask"].sum()), 1)
        self.assertEqual(int(result["source_evidence_mask"].sum()), 2)

    def test_target_selection_does_not_consult_depth_to_replace_top_score(self):
        self.depth[1, 1] = np.nan
        result = self.prepare([self.detection(self.mask((1, 1)), .9),
                               self.detection(self.mask((5, 5)), .8)])
        self.assertFalse(result["prepared"])
        self.assertEqual(result["selection"]["selected_index"], 0)
        self.assertEqual(result["reasons"], ["selected_instance_without_valid_depth"])
        self.assertIsNone(result["observation"])

    def test_ties_use_score_then_original_index(self):
        evidence = self.mask((1, 1), (2, 1))
        candidates = [self.detection(self.mask((1, 1)), .8),
                      self.detection(self.mask((2, 1)), .9),
                      self.detection(self.mask((2, 1)), .9)]
        source = self.prepare(candidates, source_region_mask=evidence)
        target = self.prepare(candidates)
        self.assertEqual(source["selection"]["selected_index"], 1)
        self.assertEqual(target["selection"]["selected_index"], 1)

    def test_source_no_overlap_is_explicit_failure_without_box_prior(self):
        evidence = self.mask((8, 6))
        # The saved detector box encloses the source evidence, but only the
        # actual predicted mask can establish a source instance overlap.
        result = self.prepare([self.detection(self.mask((1, 1)))], source_region_mask=evidence)
        self.assertFalse(result["prepared"])
        self.assertEqual(result["reasons"], ["source_instance_without_evidence_overlap"])
        np.testing.assert_array_equal(result["source_evidence_mask"], evidence)

    def test_empty_detections_and_missing_source_depth_remain_failure_data(self):
        result = self.prepare([])
        self.assertEqual(result["reasons"], ["no_sam_detections"])
        evidence = self.mask((1, 1))
        self.depth[1, 1] = np.inf
        result = self.prepare([self.detection(evidence)], source_region_mask=evidence)
        self.assertEqual(result["reasons"], ["source_evidence_without_valid_depth"])
        self.assertEqual(result["diagnostics"]["source_evidence_original_pixels"], 1)

    def test_target_preserves_holes_and_raw_mask_but_crops_valid_depth_only(self):
        sam = self.mask((1, 1), (2, 1), (3, 1), (1, 2), (3, 2),
                        (1, 3), (2, 3), (3, 3), (9, 7))
        self.depth[7, 9] = 0
        result = self.prepare([self.detection(sam)], crop_margin_fraction=0)
        observation = result["observation"]
        np.testing.assert_array_equal(observation["object_mask"], sam)
        np.testing.assert_array_equal(result["crop_xyxy"], [1, 1, 4, 4])
        self.assertNotIn([2, 2], observation["pixels"].tolist())
        self.assertNotIn([9, 7], observation["pixels"].tolist())
        self.assertTrue(observation["entity_member"].all())
        self.assertFalse(observation["region_mask"].any())
        self.assertFalse(observation["evidence_only"].any())
        self.assertIsNone(result["source_evidence_mask"])

    def test_bad_contracts_raise_instead_of_becoming_segmentation_failures(self):
        candidate = self.detection(self.mask((1, 1)))
        for change in ({"mask": candidate["mask"].astype(np.uint8)}, {"score": np.nan},
                       {"bbox": [0, 0, 2]}, {"label": "different literal prompt"}):
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    self.prepare([{**candidate, **change}])
        with self.assertRaises(ValueError):
            self.prepare([candidate], source_region_mask=np.zeros((3, 3), bool))
        bad_pose = self.T.copy();bad_pose[:3, :3] *= 2
        with self.assertRaises(ValueError):
            prepare_observation(self.rgb, self.depth, self.K, bad_pose, [candidate], prompt=self.prompt)

    def test_ground_truth_keywords_are_not_accepted(self):
        with self.assertRaises(TypeError):
            self.prepare([], owner_mask=np.ones(self.depth.shape, bool))


class TestSegmentationStorage(unittest.TestCase):
    def test_all_native_candidates_roundtrip_without_selection_or_mask_edits(self):
        rgb = np.zeros((6, 7, 3), np.uint8)
        first = np.zeros((6, 7), bool);first[1:5, 1:6] = True;first[2, 3] = False
        second = np.eye(6, 7, dtype=bool)
        candidates = [{"label": "object", "score": score, "bbox": [-.2, .1, 6.7, 5.9], "mask": mask}
                      for score, mask in ((.01, first), (.99, second))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence"
            metadata = store_detections(path, rgb, "object", candidates)
            self.assertEqual(metadata["candidate_count"], 2)
            restored = load_detections(path)
            for actual, expected in zip(restored, candidates):
                self.assertEqual(actual["score"], expected["score"])
                self.assertEqual(actual["bbox"], expected["bbox"])
                np.testing.assert_array_equal(actual["mask"], expected["mask"])
            with self.assertRaises(FileExistsError):
                store_detections(path, rgb, "object", candidates)

    def test_empty_model_output_has_a_loadable_explicit_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty"
            record = store_detections(path, np.zeros((3, 4, 3), np.uint8), "object", [])
            self.assertEqual(record["status"], "no_detections")
            self.assertEqual(load_detections(path), [])

    def test_detect_uses_full_rgb_single_prompt_and_propagates_worker_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = np.arange(5*9*3, dtype=np.uint8).reshape(5, 9, 3)
            source = directory / "input.png";Image.fromarray(image).save(source)
            def fake_detect(rgb, labels):
                np.testing.assert_array_equal(np.asarray(rgb), image)
                self.assertEqual(labels, ["cabinet"])
                return []
            with mock.patch("local_models.sam31.detect", side_effect=fake_detect) as detector:
                output = directory / "run"
                record = detect_image(source, "cabinet", output)
                self.assertEqual(record["status"], "no_detections")
                detector.assert_called_once()
                with self.assertRaises(FileExistsError):
                    detect_image(source, "cabinet", output)
                detector.assert_called_once()
            with mock.patch("local_models.sam31.detect", side_effect=RuntimeError("worker stopped")):
                with self.assertRaisesRegex(RuntimeError, "worker stopped"):
                    detect_image(source, "cabinet", directory / "failed")

    def test_corrupt_saved_indices_are_not_silently_reordered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            record = store_detections(path, np.zeros((3, 4, 3), np.uint8), "object",
                [{"label": "object", "score": .8, "bbox": [0, 0, 4, 3], "mask": np.ones((3, 4), bool)}])
            record["candidates"][0]["index"] = 7
            (path / "candidates.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_detections(path)


# BEGIN end-to-end CPU checks: frozen SAM evidence through actual direct core.
class TestSegmentedExperimentIntegration(unittest.TestCase):
    """Mock descriptor decoding only; run real preparation, fitting and metrics."""

    def fixture(self, directory):
        root, detections = directory / "data", directory / "detections"
        root.mkdir()
        height, width = 64, 72
        rgb = np.arange(height*width*3, dtype=np.uint32).reshape(height, width, 3).astype(np.uint8)
        K = np.array([[200., 0., 36.], [0., 200., 32.], [0., 0., 1.]])
        evidence = np.zeros((height, width), bool);evidence[30:38, 32:40] = True
        source_entity = np.zeros_like(evidence);source_entity[16:46, 20:52] = True
        target_entity = np.zeros_like(evidence);target_entity[18:50, 24:60] = True
        source_sam = source_entity.copy();source_sam[:, 36:38] = False
        source_decoy = np.zeros_like(evidence);source_decoy[30, 32] = True
        target_decoy = np.zeros_like(evidence);target_decoy[20:23, 26:29] = True
        for role, entity, shift in (("source", source_entity, [0., 0., 0.]),
                                     ("target", target_entity, [.04, -.02, 0.])):
            path = root / role;path.mkdir()
            Image.fromarray(rgb).save(path / "rgb.png")
            np.save(path / "depth_m.npy", np.ones((height, width)), allow_pickle=False)
            T = np.eye(4);T[:3, 3] = shift
            (path / "calibration.json").write_text(json.dumps({"K": K.tolist(), "T_world_camera_cv": T.tolist()}))
            support = np.where(entity, 0, -1).astype(np.int32)
            np.savez_compressed(path / "truth.npz", entity_mask=entity, support_id_image=support)
        for role, masks, scores in (("source", [source_decoy, source_sam], [.99, .7]),
                                    ("target", [target_decoy, target_entity], [.6, .9])):
            candidates = [{"label": "container", "score": score, "bbox": [0, 0, width, height], "mask": mask}
                          for mask, score in zip(masks, scores)]
            store_detections(detections / role, rgb, "container", candidates)
        np.save(root / "evidence.npy", evidence, allow_pickle=False)
        y, x = np.nonzero(evidence)
        reference = np.array([(x.mean()-36)/200+.04, (y.mean()-32)/200-.02, 1.])
        (root / "pair_truth.json").write_text(json.dumps({"preferred_support_id": 0,
            "target_reference_evidence_center_world": reference.tolist()}))
        description = {"views": {role: {"path": role, "prompt": "container"} for role in ("source", "target")},
            "pairs": [{"id": "one", "group": "synthetic", "variation": "pose", "source": "source", "target": "target",
                       "evidence": "evidence.npy", "pair_truth": "pair_truth.json"}]}
        (root / "manifest.json").write_text(json.dumps(description))
        return root, detections, evidence, description

    def execute_automatic(self, root, detections, run, evidence):
        from transfer import segmented_experiment as runner
        original_archive, original_read_text = runner.archive, Path.read_text
        def guarded_archive(path):
            if Path(path).name == "truth.npz":
                raise AssertionError("Automatic stage attempted to read target/entity truth")
            return original_archive(path)
        def guarded_read_text(path, *args, **kwargs):
            if path.name == "pair_truth.json":
                raise AssertionError("Automatic stage attempted to read held-out reference geometry")
            return original_read_text(path, *args, **kwargs)
        with mock.patch.object(runner, "archive", side_effect=guarded_archive), \
                mock.patch.object(Path, "read_text", guarded_read_text), redirect_stdout(io.StringIO()):
            runner.prepare_suite(root, detections, run, arms=("sam31",))
            pair_directory = run / "sam31" / "one"
            source = original_archive(pair_directory / "source_observation.npz")
            target = original_archive(pair_directory / "target_observation.npz")
            inputs = json.loads((pair_directory / "inputs.json").read_text())
            source_offset = np.asarray(inputs["source_crop_xyxy"][:2])
            target_offset = np.asarray(inputs["target_crop_xyxy"][:2])
            self.assertTrue(np.any(source_offset != target_offset))
            self.assertGreater(int(source["evidence_only"].sum()), 0)
            self.assertEqual(int(source["region_mask"].sum()), int(evidence.sum()))
            np.savez_compressed(pair_directory / "features.npz",
                                feature1=np.zeros((1, 1, 1, 1), np.float32),
                                feature2=np.zeros((1, 1, 1, 1), np.float32))
            def correspondence(source_chart, target_chart, region, features):
                np.testing.assert_array_equal(source_chart["pixels"], source["pixels"]-source_offset)
                np.testing.assert_array_equal(target_chart["pixels"], target["pixels"]-target_offset)
                np.testing.assert_array_equal(region, source["region_mask"])
                source_indices = np.flatnonzero(region)
                lookup = {tuple(p): i for i, p in enumerate(target["pixels"].tolist())}
                target_indices = np.asarray([lookup[tuple(p)] for p in source["pixels"][source_indices]], dtype=np.int64)
                count = len(source_indices)
                return {"src_anchor_idx": source_indices, "tgt_anchor_idx": target_indices,
                    "scores": np.ones(count), "raw_source_idx": source_indices.copy(),
                    "raw_target_idx": target_indices.copy(),
                    "raw_target_pixels": target_chart["pixels"][target_indices].astype(float),
                    "raw_valid": np.ones(count, bool), "raw_scores": np.ones(count),
                    "matching_rule": "independently specified synthetic pixel identity"}
            with mock.patch("transfer.features.native_region_pairs", side_effect=correspondence):
                runner.predict_suite(root, run, arms=("sam31",))
            saved = original_archive(pair_directory / "native_region_pairs.npz")
            expected = source["pixels"][np.flatnonzero(source["region_mask"])]
            np.testing.assert_array_equal(saved["raw_target_pixels"], expected)
        prediction = original_archive(pair_directory / "direct_local.npz")
        self.assertTrue(bool(prediction["accepted"]))
        self.assertGreater(int(prediction["mode_count"]), 0)
        preparation = json.loads((pair_directory / "preparation.json").read_text())
        self.assertEqual(preparation["source"]["selection"]["selected_index"], 1)
        self.assertEqual(preparation["target"]["selection"]["selected_index"], 1)
        return prediction

    def test_changing_target_truth_cannot_change_preparation_or_real_core_prediction(self):
        from transfer import segmented_experiment as runner
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, detections, evidence, description = self.fixture(directory)
            first_run, second_run = directory / "first", directory / "second"
            first = self.execute_automatic(root, detections, first_run, evidence)
            with mock.patch.object(runner, "render_comparison"), redirect_stdout(io.StringIO()):
                runner.evaluate_suite(root, first_run, arms=("sam31",))
            original_metrics = json.loads((first_run / "sam31/one/metrics.json").read_text())
            self.assertTrue(original_metrics["top1_preferred_support_hit"])
            # Both entity labels and reference geometry change drastically.
            # Preparation and actual RANSAC/neighborhood prediction must remain
            # byte-for-byte identical while held-out measurements must respond.
            np.savez_compressed(root / "target/truth.npz", entity_mask=np.zeros(evidence.shape, bool),
                                support_id_image=np.full(evidence.shape, -1, np.int32))
            (root / "pair_truth.json").write_text(json.dumps({"preferred_support_id": 0,
                "target_reference_evidence_center_world": [3., 4., 5.]}))
            second = self.execute_automatic(root, detections, second_run, evidence)
            self.assertEqual(set(first), set(second))
            for key in first:
                np.testing.assert_array_equal(first[key], second[key], err_msg=key)
            for role in ("source", "target"):
                a = runner.archive(first_run / "sam31/one" / f"{role}_observation.npz")
                b = runner.archive(second_run / "sam31/one" / f"{role}_observation.npz")
                self.assertEqual(set(a), set(b))
                for key in a: np.testing.assert_array_equal(a[key], b[key])
            with mock.patch.object(runner, "render_comparison"), redirect_stdout(io.StringIO()):
                runner.evaluate_suite(root, second_run, arms=("sam31",))
            changed_metrics = json.loads((second_run / "sam31/one/metrics.json").read_text())
            self.assertTrue(changed_metrics["accepted"])
            self.assertFalse(changed_metrics["top1_preferred_support_hit"])
            self.assertGreater(changed_metrics["center_error_m"], 1.)

    def test_preparation_failures_keep_the_full_denominator_and_selected_mask_metrics(self):
        from transfer import segmented_experiment as runner
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root, detections, evidence, description = self.fixture(directory)
            # A selected, occupied SAM mask with unusable depth is a depth
            # failure, not an empty segmentation. Its mask metrics still exist.
            np.save(root / "target/depth_m.npy", np.full(evidence.shape, np.nan), allow_pickle=False)
            description["views"]["no_detection"] = {"path": "target", "prompt": "container"}
            description["pairs"].append({**description["pairs"][0], "id": "empty", "target": "no_detection"})
            (root / "manifest.json").write_text(json.dumps(description))
            rgb = np.array(Image.open(root / "target/rgb.png"))
            store_detections(detections / "no_detection", rgb, "container", [])
            run = directory / "run"
            with redirect_stdout(io.StringIO()), mock.patch.object(runner, "render_comparison"):
                runner.prepare_suite(root, detections, run, arms=("sam31",))
                runner.predict_suite(root, run, arms=("sam31",))
                runner.evaluate_suite(root, run, arms=("sam31",))
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(summary["pairs_per_arm"], 2)
            self.assertEqual(len(summary["rows"]), 2)
            self.assertTrue(all(not row["accepted"] and not row["prediction_executed"] for row in summary["rows"]))
            reports = {name: json.loads((run / "sam31" / name / "report.json").read_text()) for name in ("one", "empty")}
            self.assertIn("target:selected_instance_without_valid_depth", reports["one"]["prediction"]["reasons"])
            self.assertIn("target:no_sam_detections", reports["empty"]["prediction"]["reasons"])
            self.assertEqual(reports["one"]["mask_metrics"]["target"]["selected_pixels"], 32*36)
            self.assertEqual(reports["one"]["mask_metrics"]["target"]["entity_iou"], 1.)
            self.assertEqual(reports["empty"]["mask_metrics"]["target"]["selected_pixels"], 0)
# END actual-core experiment isolation and complete-denominator regressions.


if __name__ == "__main__":
    unittest.main()
# END original-pixel segmentation evidence regressions.
