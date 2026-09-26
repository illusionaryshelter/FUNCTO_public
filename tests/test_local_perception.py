"""CPU contract checks for local SAM detection and FUNCTO detection artifacts."""

# BEGIN local SAM regression checks: synthetic worker arrays never load a model.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from cam_to_target_trans import demo_test_deteciton
from local_models.sam31 import detect


class SamDetectionContractTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (10, 6))

    def test_empty_mask_is_removed_without_mixing_instance_fields(self):
        """An erased middle instance must not shift another label or box onto it."""
        masks = np.zeros((3, 6, 10), dtype=bool)
        masks[0, 1:3, 1:4] = True
        masks[2, 3:5, 6:9] = True
        arrays = {
            "labels": np.array(["cup", "cup", "bowl"]),
            "scores": np.array([0.7, 0.99, 0.8], dtype=np.float32),
            "boxes": np.array([[1, 1, 4, 3], [0, 0, 10, 6], [6, 3, 9, 5]], dtype=np.float32),
            "masks": masks,
        }
        with patch("local_models.runtime.run_worker", return_value=arrays):
            detections = detect(self.image, ["cup", "bowl"])

        self.assertEqual([item["label"] for item in detections], ["cup", "bowl"])
        for item, original_index in zip(detections, [0, 2]):
            self.assertEqual(item["score"], float(arrays["scores"][original_index]))
            self.assertEqual(item["bbox"], arrays["boxes"][original_index].tolist())
            np.testing.assert_array_equal(item["mask"], masks[original_index])
        # Filtering is read-only: callers or other consumers may reuse arrays.
        self.assertEqual(arrays["labels"].tolist(), ["cup", "cup", "bowl"])

    def test_all_empty_and_no_native_detections_both_return_empty(self):
        """A fully occluded label is equivalent to having no usable instance."""
        for count in (0, 2):
            with self.subTest(count=count):
                arrays = {
                    "labels": np.asarray(["cup"] * count, dtype=str),
                    "scores": np.ones(count, dtype=np.float32),
                    "boxes": np.zeros((count, 4), dtype=np.float32),
                    "masks": np.zeros((count, 6, 10), dtype=bool),
                }
                with patch("local_models.runtime.run_worker", return_value=arrays):
                    self.assertEqual(detect(self.image, ["cup"]), [])

    def test_missing_target_fails_before_writing_any_role_artifacts(self):
        """A high-score target with no occupied pixels cannot complete detection."""
        masks = np.zeros((2, 6, 10), dtype=bool)
        masks[0, 1:5, 2:8] = True
        arrays = {
            "labels": np.array(["cup", "bowl"]),
            "scores": np.array([0.8, 0.99], dtype=np.float32),
            "boxes": np.array([[2, 1, 8, 5], [0, 0, 10, 6]], dtype=np.float32),
            "masks": masks,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rgb").mkdir()
            self.image.save(root / "rgb" / "00000.jpg")
            with patch("local_models.runtime.run_worker", return_value=arrays):
                with self.assertRaisesRegex(RuntimeError, "no target.*bowl"):
                    demo_test_deteciton(root, "cup", "bowl")
            self.assertEqual(list((root / "detection_output").iterdir()), [])

    def test_roles_use_highest_score_and_save_matching_boxes_and_masks(self):
        """Keep the downstream pixel-box and binary-mask file contracts intact."""
        masks = np.zeros((3, 6, 10), dtype=bool)
        masks[0, 0:2, 0:2] = True
        masks[1, 1:4, 2:5] = True
        masks[2, 3:5, 6:9] = True
        arrays = {
            "labels": np.array(["cup", "cup", "bowl"]),
            "scores": np.array([0.6, 0.9, 0.8], dtype=np.float32),
            "boxes": np.array([[0, 0, 2, 2], [2, 1, 5, 4], [6, 3, 9, 5]], dtype=np.float32),
            "masks": masks,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rgb").mkdir()
            self.image.save(root / "rgb" / "00000.jpg")
            with patch("local_models.runtime.run_worker", return_value=arrays):
                demo_test_deteciton(root, "cup", "bowl")
            output = root / "detection_output"
            self.assertEqual(len(list(output.iterdir())), 8)
            for role, original_index in [("tool", 1), ("target", 2)]:
                self.assertEqual(
                    json.loads((output / f"{role}.json").read_text()),
                    arrays["boxes"][original_index].tolist(),
                )
                saved_mask = np.load(output / f"{role}_mask.npy", allow_pickle=False)
                self.assertEqual(saved_mask.dtype, np.bool_)
                np.testing.assert_array_equal(saved_mask, masks[original_index])
                with Image.open(output / f"{role}_mask.png") as png:
                    np.testing.assert_array_equal(np.asarray(png), saved_mask.astype(np.uint8) * 255)


if __name__ == "__main__":
    unittest.main()
# END local SAM regression checks.
