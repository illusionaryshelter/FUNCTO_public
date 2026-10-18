"""CPU single-image/pair equivalence without loading a pretrained model."""

# BEGIN one real image retains MARCO's maintained native preprocessing.
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from local_models import marco


class MarcoSingleImageTests(unittest.TestCase):
    def test_native_single_preprocessing_equals_each_pair_component(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.png", Path(directory) / "second.png"
            Image.fromarray(np.arange(11 * 17 * 3, dtype=np.uint16).reshape(11, 17, 3).astype(np.uint8)).save(first)
            Image.fromarray(np.arange(19 * 9 * 3, dtype=np.uint16).reshape(19, 9, 3).astype(np.uint8)).save(second)
            native = marco._preprocessing()
            pair = native.preprocess_data(str(first), str(second), [[0., 0.]], device="cpu")["samples"]
            for index, path in enumerate((first, second)):
                single = marco.preprocess_image(path, device="cpu")
                np.testing.assert_array_equal(single.numpy(), pair[:, index:index + 1].numpy())

    def test_worker_does_not_duplicate_source_or_change_pair_feature_component(self):
        import torch
        from torch.nn import functional as F

        native = marco._preprocessing()
        count = []

        class PerImageBackbone:
            softmax_temp = marco.TEMPERATURE

            def eval(self):
                return self

            def preprocess_data(self, source, target, points, **kwargs):
                return native.preprocess_data(source, target, points, inference_res=marco.RESOLUTION, device="cpu")

            def forward_backbone(self, samples):
                count.append(samples.shape[1])
                flat = samples.reshape(-1, *samples.shape[2:])
                return F.adaptive_avg_pool2d(flat, (4, 4)).reshape(1, samples.shape[1], 3, 4, 4)

        original_preprocess = marco.preprocess_image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.fromarray(np.arange(13 * 7 * 3, dtype=np.uint16).reshape(13, 7, 3).astype(np.uint8)).save(path)
            single_output, pair_output = Path(directory) / "single.npz", Path(directory) / "pair.npz"
            with patch.object(torch.hub, "load", return_value=PerImageBackbone()), \
                 patch.object(marco, "FEATURE_SHAPE", (1, 3, 4, 4)), \
                 patch.object(marco, "preprocess_image", side_effect=lambda value: original_preprocess(value, device="cpu")):
                marco.worker_main([path], single_output)
                marco.worker_main([path, path], pair_output)
            self.assertEqual(count, [1, 2])
            with np.load(single_output, allow_pickle=False) as single, np.load(pair_output, allow_pickle=False) as pair:
                self.assertEqual(single.files, ["feature"])
                np.testing.assert_array_equal(single["feature"], pair["feature1"])
                np.testing.assert_array_equal(single["feature"], pair["feature2"])

    def test_file_single_worker_command_has_one_input_and_does_not_overwrite(self):
        from transfer.features import extract_image

        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "image.png", Path(directory) / "map.npz"
            Image.new("RGB", (3, 4)).save(source)
            with patch("transfer.features.subprocess.run") as run:
                extract_image(source, output)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("worker") + 1:], [str(source), str(output)])
            output.touch()
            with patch("transfer.features.subprocess.run") as run:
                with self.assertRaises(FileExistsError):
                    extract_image(source, output)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
# END model quality and CUDA equivalence remain actual supervised GPU checks.
