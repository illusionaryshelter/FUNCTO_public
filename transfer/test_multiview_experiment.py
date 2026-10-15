"""Small contracts for frozen experiment artifacts and held-out evaluation.

These tests use temporary typed arrays, never models, fixture scenes, or real
experiment results. They verify the boundary that keeps truth out of inference.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

from transfer import multiview_experiment as experiment, multiview_registration_cases as fixture


# BEGIN independent experiment artifact and truth-isolation contracts.
class MultiviewExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="multiview-experiment-test-")
        self.folder = Path(self.temporary.name)
        self.manifest = self.folder / "manifest.json"
        self.case = {"id": "sample", "source_input_dir": "missing-input",
            "source_observation_dir": "missing-observation",
            "targets": [{"id": view, "input_dir": "missing-input", "observation_dir": "missing-observation",
                         "missing_target": False, "missing_target_reasons": [], "missing_target_stage": None}
                        for view in ("A", "B")],
            "evaluation": {"source_truth": "DO-NOT-OPEN-source.npz", "pair_truth": "DO-NOT-OPEN-pair.json",
                           "target_truth": {"A": "DO-NOT-OPEN-a.npz", "B": "DO-NOT-OPEN-b.npz"}}}
        self.write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({"schema": "multiview_grounding_v1", "cases": [self.case]}))

    def test_manifest_does_not_open_declared_truth(self):
        with mock.patch.object(experiment, "archive", side_effect=AssertionError("must not read truth")):
            path, description = experiment.read_manifest(self.manifest)
        self.assertEqual(path, self.manifest)
        self.assertEqual(description["cases"][0]["id"], "sample")

    def test_evaluation_waits_for_complete_prediction_matrix(self):
        with mock.patch.object(experiment, "archive", side_effect=AssertionError("truth read before barrier")):
            with self.assertRaisesRegex(FileNotFoundError, "Finish every declared prediction"):
                experiment.evaluate_suite(self.manifest, self.folder / "results")

    def test_duplicate_view_identity_is_rejected(self):
        self.case["targets"][1]["id"] = "A"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "unique"):
            experiment.read_manifest(self.manifest)

    def test_fusion_array_and_provenance_roundtrip(self):
        result = {"bank": {"xyz": np.arange(9.).reshape(3, 3), "view_ids": ["A", "B"],
                           "view_offsets": np.array([0, 2, 4])},
            "src_anchor_idx": np.array([1, 2]), "tgt_anchor_idx": np.array([0, 2]),
            "scores": np.array([.2, .5]), "raw_reasons": np.array(["ok", "no_support"]),
            "diagnostics": {"method": "test"}}
        experiment._save_fusion(self.folder, result, {"fusion_ms": 0.})
        restored = experiment.load_fusion(self.folder)
        for key in ("src_anchor_idx", "tgt_anchor_idx", "scores", "raw_reasons"):
            np.testing.assert_array_equal(restored[key], result[key])
        np.testing.assert_array_equal(restored["bank"]["xyz"], result["bank"]["xyz"])
        np.testing.assert_array_equal(restored["bank"]["view_offsets"], result["bank"]["view_offsets"])
        self.assertEqual(restored["bank"]["view_ids"], ["A", "B"])
        self.assertTrue(restored["frozen"])
        self.assertIs(restored["target_truth_used"], False)

    def test_nonuniform_truth_reports_mapping_error_without_false_scale(self):
        points = np.array([[0., 0., 0.], [1., 2., 3.]])
        mapping = np.diag([1.2, .8, 1., 1.])
        metrics = experiment.transform_metrics(1., np.eye(3), np.zeros(3), points, mapping, False)
        self.assertIsNone(metrics["scale_relative_error"])
        self.assertIsNone(metrics["rotation_error_deg"])
        self.assertIsNone(metrics["translation_parameter_error_m"])
        self.assertGreater(metrics["source_points_transform_rmse_m"], 0.)

    def test_transform_metrics_accepts_frozen_json_arrays_without_refitting(self):
        points = np.array([[0., 0., 0.], [1., 2., 3.], [.3, -.2, .7]])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        translation = np.array([.2, -.3, .4])
        mapping = np.eye(4)
        mapping[:3, :3] = 1.25 * rotation
        mapping[:3, 3] = translation
        frozen = json.loads(json.dumps({"scale": 1.25, "rotation": rotation.tolist(), "translation": translation.tolist()}))
        with mock.patch("transfer.core.prepare_anchors", side_effect=AssertionError("evaluation must not refit")):
            metrics = experiment.transform_metrics(frozen["scale"], frozen["rotation"], frozen["translation"], points, mapping, True)
        self.assertLess(metrics["source_points_transform_rmse_m"], 1e-12)
        self.assertLess(metrics["scale_relative_error"], 1e-12)
        self.assertLess(metrics["rotation_error_deg"], 1e-12)
        self.assertLess(metrics["translation_parameter_error_m"], 1e-12)

    def test_depth_artifact_mismatch_is_rejected_without_snapping(self):
        depth = np.ones((3, 3))
        observation = {"pixels": np.array([[1, 1]]), "xyz": np.array([[1., 1., 1.]]),
                       "K": np.eye(3), "T_world_camera_cv": np.eye(4)}
        experiment._check_observed_depth(observation, depth)
        different = depth.copy()
        different[1, 1] = 2.
        with self.assertRaises(AssertionError):
            experiment._check_observed_depth(observation, different)

    def _missing_target_inputs(self):
        """Write one observed view plus one explicit segmentation failure."""
        mask = np.array([[True, False], [False, False]])
        observation = {"xyz": np.array([[0., 0., 1.]]), "pixels": np.array([[0, 0]]),
            "object_mask": mask, "entity_member": np.ones(1, dtype=bool),
            "region_mask": np.ones(1, dtype=bool), "evidence_only": np.zeros(1, dtype=bool),
            "K": np.eye(3), "T_world_camera_cv": np.eye(4)}
        full_source = self.folder / "source-rgbd"
        full_source.mkdir()
        np.save(full_source / "depth_m.npy", np.ones((2, 2)))
        calibration = {"K": np.eye(3).tolist(), "T_world_camera_cv": np.eye(4).tolist()}
        (full_source / "calibration.json").write_text(json.dumps(calibration))
        self.case["source_observation_dir"] = str(full_source)
        for view_index, target in enumerate(self.case["targets"]):
            directory, full = self.folder / f"inputs-{target['id']}", self.folder / f"rgbd-{target['id']}"
            directory.mkdir()
            full.mkdir()
            np.save(full / "depth_m.npy", np.ones((2, 2)))
            pose = np.eye(4)
            pose[0, 3] = view_index * .2
            (full / "calibration.json").write_text(json.dumps({"K": np.eye(3).tolist(), "T_world_camera_cv": pose.tolist()}))
            target.update(input_dir=str(directory), observation_dir=str(full))
            missing = view_index == 1
            target.update(missing_target=missing, missing_target_reasons=["no_sam_detections"] if missing else [],
                          missing_target_stage="segmentation_preparation" if missing else None)
            prepared = {"prepared": not missing, "source": {"prepared": True, "reasons": []},
                "target": {"prepared": not missing, "reasons": target["missing_target_reasons"]}}
            (directory / "preparation.json").write_text(json.dumps(prepared))
            metadata = {"source_crop_xyxy": [0, 0, 2, 2]}
            np.savez(directory / "source_observation.npz", **observation)
            if not missing:
                self.case["source_input_dir"] = str(directory)
                metadata["target_crop_xyxy"] = [0, 0, 2, 2]
                np.savez(directory / "target_observation.npz", **observation)
                np.save(directory / "target_selected_mask.npy", mask)
                np.savez(directory / "features.npz", feature1=np.ones((1, 1, 2, 2), dtype=np.float32),
                         feature2=np.ones((1, 1, 2, 2), dtype=np.float32))
            (directory / "inputs.json").write_text(json.dumps(metadata))
        self.write_manifest()

    def test_explicit_missing_target_retains_empty_view_and_original_camera(self):
        self._missing_target_inputs()
        path, description = experiment.read_manifest(self.manifest)
        source, feature, crop, targets = experiment.load_inputs(path, description["cases"][0], features=True)
        missing = targets[1]
        self.assertEqual(missing["id"], "B")
        self.assertTrue(missing["missing_target"])
        self.assertEqual(missing["missing_target_reasons"], ["no_sam_detections"])
        self.assertEqual(missing["observation"]["xyz"].shape, (0, 3))
        self.assertEqual(missing["observation"]["pixels"].shape, (0, 2))
        self.assertFalse(missing["selected_mask"].any())
        self.assertEqual(missing["selected_mask"].shape, (2, 2))
        self.assertIsNone(missing["feature"])
        self.assertIsNone(missing["crop_xyxy"])
        self.assertEqual(missing["observation"]["T_world_camera_cv"][0, 3], .2)
        np.testing.assert_array_equal(missing["depth_m"], np.ones((2, 2)))
        self.assertEqual(len(source["xyz"]), 1)
        self.assertIsNotNone(feature)

    def test_missing_target_requires_matching_preparation_declaration(self):
        self._missing_target_inputs()
        self.case["targets"][1].update(missing_target=False, missing_target_reasons=[], missing_target_stage=None)
        with self.assertRaisesRegex(ValueError, "disagrees with its frozen preparation"):
            experiment.load_inputs(self.manifest, self.case)

    def test_missing_target_does_not_hide_a_missing_source_artifact(self):
        self._missing_target_inputs()
        (Path(self.case["targets"][1]["input_dir"]) / "source_observation.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            experiment.load_inputs(self.manifest, self.case)

    def test_source_preparation_is_independent_of_its_paired_target_status(self):
        self._missing_target_inputs()
        self.case["source_input_dir"] = self.case["targets"][1]["input_dir"]
        source, feature, _, targets = experiment.load_inputs(self.manifest, self.case)
        self.assertEqual(len(source["xyz"]), 1)
        self.assertIsNone(feature)
        self.assertTrue(targets[1]["missing_target"])
        # There is no feature cache at this explicit source path. The loader
        # must report that fact rather than silently switch to another pair.
        with self.assertRaises(FileNotFoundError):
            experiment.load_inputs(self.manifest, self.case, features=True)

    def test_missing_target_feature_stage_records_skip_without_model_call(self):
        supplement = self.folder / "cabinet_b"
        directory = supplement / "inputs/sam31/cabinet_yaw_a_to_b"
        directory.mkdir(parents=True)
        (directory / "preparation.json").write_text(json.dumps({"prepared": False,
            "source": {"prepared": True}, "target": {"prepared": False, "reasons": ["no_sam_detections"]}}))
        with mock.patch("transfer.segmented_experiment.extract_suite", side_effect=AssertionError("must not call MARCO")):
            fixture.extract_features(self.folder, self.folder)
        record = json.loads((supplement / "marco-timing.json").read_text())
        self.assertEqual(record["status"], "skipped_no_target_observation")
        self.assertEqual(record["model_calls"], 0)
        self.assertEqual(record["fresh_pair_call_ms"], 0.)
        self.assertFalse((directory / "features.npz").exists())


if __name__ == "__main__":
    unittest.main()
# END independent experiment artifact and truth-isolation contracts.
