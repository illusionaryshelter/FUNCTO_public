"""Offline contracts for fresh FUNCTO runs and Gemini visual selections.

Only tiny synthetic images/artifacts are used. Gemini is mocked at its public
request boundary; no GPU model, API, optimizer, or full pipeline is executed.
"""

# BEGIN regression checks for run isolation and visual-prompt migration.
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import main as pipeline
from utils_IL import vp_utils


class RunArtifactsContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="functo-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for name in ("rgb", "depth"):
            (self.source / name).mkdir()
        Image.new("RGB", (8, 8)).save(self.source / "rgb/00000.jpg")
        Image.fromarray(np.full((8, 8), 500, dtype=np.uint16)).save(
            self.source / "depth/00000.png"
        )
        (self.source / "intrinsic.json").write_text('{"width": 8, "height": 8}')
        (self.source / "config.json").write_text('{"input": "synthetic"}')
        self.destination = self.root / "run"

    @staticmethod
    def write_json(root, relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def valid_artifacts(self):
        """Construct schema-valid data, not a simulated successful model run."""
        pipeline.prepare_run(self.source, self.destination)
        detection = self.destination / "detection_output"
        detection.mkdir()
        for role in ("tool", "target"):
            mask = np.zeros((8, 8), dtype=bool)
            mask[1:4, 1:4] = True
            np.save(detection / f"{role}_mask.npy", mask)
            self.write_json(self.destination, f"detection_output/{role}.json", [1, 1, 4, 4])
        self.write_json(
            self.destination, "func_point_transfer_output/test_init_frame_func_point_out.json", [2, 2]
        )
        self.write_json(
            self.destination, "grasp_transfer_output/test_init_frame_grasp_point_out.json", [3, 2]
        )
        self.write_json(self.destination, "cam_to_target_trans_output/cam_to_target_trans.json", {
            "rotation_matrix": np.eye(3).tolist(), "rotation_center": [0, 0, 0],
            "centroid": [0, 0, 0.5], "target_dim": [0.1, 0.1, 0.1],
            "box_points": [[0, 0, 0]] * 8,
        })
        self.write_json(self.destination, "tool_pose_transfer_output/test_tool_track.json", {
            "function_init": [0, 0, 0], "center_init": [0.1, 0, 0], "grasp_init": [0, 0.1, 0],
            "function_func": [0, 0, 0.1], "center_func": [0.1, 0, 0.1], "grasp_func": [0, 0.1, 0.1],
        })
        trajectory = self.destination / "tool_traj_transfer_output"
        self.write_json(self.destination, "tool_traj_transfer_output/test_tool_trajectory.json", [
            [[0, 0, 0], [0, 0, 0]], [[0, 0, 0.1], [0, 0, 0]],
        ])
        (trajectory / "test_tool_traj_pc.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
            "property float y\nproperty float z\nend_header\n0 0 0\n"
        )

    def test_prepare_run_copies_observation_but_no_previous_predictions(self):
        self.write_json(self.source, "tool_traj_transfer_output/test_tool_trajectory.json", ["stale"])
        self.write_json(self.source, "func_point_transfer_output/test_init_frame_func_point_out.json", [999, 999])
        pipeline.prepare_run(self.source, self.destination)
        actual = {
            str(path.relative_to(self.destination))
            for path in self.destination.rglob("*") if path.is_file()
        }
        expected = {"rgb/00000.jpg", "depth/00000.png", "intrinsic.json", "config.json"}
        self.assertEqual(actual, expected)
        for relative in expected:
            self.assertEqual((self.source / relative).read_bytes(), (self.destination / relative).read_bytes())

    def test_nonempty_run_directory_is_rejected_without_overwriting(self):
        self.destination.mkdir()
        sentinel = self.destination / "previous-run.json"
        sentinel.write_text("previous result")
        with self.assertRaises(FileExistsError):
            pipeline.prepare_run(self.source, self.destination)
        self.assertEqual(sentinel.read_text(), "previous result")
        self.assertFalse((self.destination / "rgb").exists())

    def test_missing_depth_fails_before_creating_a_run(self):
        (self.source / "depth/00000.png").unlink()
        with self.assertRaises(FileNotFoundError):
            pipeline.prepare_run(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_complete_finite_artifacts_satisfy_the_schema(self):
        self.valid_artifacts()
        result = pipeline.validate_outputs(self.destination)
        self.assertEqual(result["trajectory_shape"], [2, 2, 3])
        self.assertEqual(result["points_2d"], {"function": [2.0, 2.0], "grasp": [3.0, 2.0]})

    def test_nonfinite_trajectory_is_rejected(self):
        self.valid_artifacts()
        self.write_json(self.destination, "tool_traj_transfer_output/test_tool_trajectory.json", [
            [[0, 0, 0], [0, 0, 0]], [[float("nan"), 0, 0], [0, 0, 0]],
        ])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pipeline.validate_outputs(self.destination)

    # BEGIN target-frame regressions: orthogonality alone does not imply SO(3).
    def test_reflection_is_rejected_even_though_it_is_orthonormal(self):
        self.valid_artifacts()
        relative = "cam_to_target_trans_output/cam_to_target_trans.json"
        frame = json.loads((self.destination / relative).read_text())
        frame["rotation_matrix"] = np.diag([1.0, 1.0, -1.0]).tolist()
        self.write_json(self.destination, relative, frame)
        with self.assertRaisesRegex(ValueError, "reflection"):
            pipeline.validate_outputs(self.destination)

    def test_invalid_frame_vectors_and_nonpositive_dimensions_are_rejected(self):
        self.valid_artifacts()
        relative = "cam_to_target_trans_output/cam_to_target_trans.json"
        valid = json.loads((self.destination / relative).read_text())
        cases = (
            ("centroid", [0.0, 0.0], "Invalid target frame vector"),
            ("rotation_center", [0.0, 0.0, float("inf")], "Invalid target frame vector"),
            ("target_dim", [0.1, float("nan"), 0.1], "Invalid target frame vector"),
            ("target_dim", [0.1, -0.1, 0.1], "positive"),
            ("target_dim", [0.1, 0.0, 0.1], "positive"),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value):
                frame = dict(valid)
                frame[key] = value
                self.write_json(self.destination, relative, frame)
                with self.assertRaisesRegex(ValueError, message):
                    pipeline.validate_outputs(self.destination)
    # END target-frame regression checks.

    def test_main_help_works_outside_repository_without_api_credentials(self):
        environment = os.environ.copy()
        environment.pop("OPENAI_API_KEY", None)
        environment.pop("SUB2API_API_KEY", None)
        environment["PYTHONNOUSERSITE"] = "1"
        result = subprocess.run(
            [sys.executable, str(pipeline.PROJECT_ROOT / "main.py"), "--help"],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-dir", result.stdout)


class VisualPromptContractTest(unittest.TestCase):
    def setUp(self):
        self.demo = Image.new("RGB", (8, 8), "red")
        self.test = Image.new("RGB", (8, 8), "blue")
        self.candidates = {"grasped": np.asarray([[1, 2], [3, 4]]), "unattached": None}
        self.task = {"object_grasped": "tool", "task_instruction": "use the tool"}

    def test_empty_grasped_object_preserves_the_nullable_motion_contract(self):
        context, raw, image = vp_utils.request_motion(
            {"object_grasped": "", "task_instruction": "observe the scene"},
            self.test, self.test, {"grasped": None}, prompts={}, debug=True,
            grasp=True, loaded_context={"grasp_keypoint": ""},
        )
        self.assertIsNone(context["keypoints_2d"]["function"])
        self.assertEqual(raw, {"grasp_keypoint": ""})
        self.assertIsInstance(image, Image.Image)

    def test_function_request_retains_demo_test_order_and_system_prompt(self):
        with patch.object(vp_utils, "Gemini") as client:
            client.return_value.chat_json.return_value = {"function_keypoint": "P2"}
            context, mapped, log_image = vp_utils.keypoint_transfer(
                self.task, self.test, [self.demo, self.test], self.candidates,
                prompts="Original function prompt", debug=False,
            )
        client.return_value.chat_json.assert_called_once_with(
            f"Task: {self.task}", [self.demo, self.test], system_prompt="Original function prompt"
        )
        self.assertEqual(context, {"function_keypoint": "P2"})
        np.testing.assert_array_equal(mapped["selected_keypoint"], [3, 4])
        self.assertIsNone(log_image)

    def test_grasp_field_and_one_based_candidate_index_are_preserved(self):
        context = {"function_keypoint": "P2", "grasp_keypoint": "P1"}
        _, mapped, _ = vp_utils.keypoint_transfer(
            self.task, self.test, self.test, self.candidates, prompts="grasp prompt",
            loaded_context=context, grasp=True,
        )
        np.testing.assert_array_equal(mapped["selected_keypoint"], [1, 2])

    def test_invalid_candidate_labels_do_not_silently_index_other_points(self):
        for label in ("P0", "P3", "P-1", "Q1", 1, None):
            with self.subTest(label=label), self.assertRaises(ValueError):
                vp_utils._selected_keypoint({"function_keypoint": label}, self.candidates)

    def test_pose_index_is_zero_based_and_bool_or_out_of_range_is_rejected(self):
        with patch.object(vp_utils, "Gemini") as client:
            client.return_value.chat_json.return_value = {"selected_idx": 1}
            context = vp_utils.pose_selection(self.task, [self.demo, self.test], "pose prompt")
            self.assertEqual(context["selected_idx"], 1)
            client.return_value.chat_json.assert_called_once_with(
                f"Task: {self.task}", [self.demo, self.test], system_prompt="pose prompt"
            )
            for index in (True, -1, 2, "1"):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    client.return_value.chat_json.return_value = {"selected_idx": index}
                    vp_utils.pose_selection(self.task, [self.demo, self.test], "pose prompt")

    def test_examples_precede_current_images_and_keep_their_responses(self):
        with patch.object(vp_utils, "Gemini") as client:
            client.return_value.chat_json.return_value = {"function_keypoint": "P1"}
            vp_utils._request_context(
                ["Current task"], [self.test], "system", [self.demo], ['{"function_keypoint":"P2"}']
            )
        args, kwargs = client.return_value.chat_json.call_args
        self.assertEqual(args[1], [self.demo, self.test])
        self.assertIn('Example image 1: {"function_keypoint":"P2"}', args[0])
        self.assertTrue(args[0].endswith("Current task"))
        self.assertEqual(kwargs, {"system_prompt": "system"})

    def test_incomplete_demonstration_pair_is_rejected_before_api_use(self):
        with patch.object(vp_utils, "Gemini") as client:
            with self.assertRaisesRegex(ValueError, "supplied together"):
                vp_utils._request_context("task", self.test, "system", [self.demo], None)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
# END regression checks for run isolation and visual-prompt migration.
