"""Exercise real CLI output status when valid trajectories exceed display bounds."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from manipulate import main
from manipulation.pipeline import write_json


class ManipulationCLITest(unittest.TestCase):
    # BEGIN compiled-skill CLI and portable cached-array replay regression.
    def test_grasp_program_renders_and_saved_npz_reference_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            poses = np.repeat(np.eye(4)[None], 5, axis=0)
            poses[:, 0, 3] = np.arange(5) * .01
            np.savez(root / "demo.npz", T_carrier_eef=poses,
                     frame_indices=np.arange(10, 15), timestamps=np.arange(5) * .05)
            program = {"skill_kind": "grasp", "demo": {
                "trajectory_file": "demo.npz", "protected_start_frame": 11,
                "tool_id": "gripper", "tool_frame": "eef",
                "gripper_events": [{"event_id": "close", "frame_index": 11, "command": "close"}]},
                "scene": {"reference_frame": "world", "carrier_id": "object",
                          "carrier_pose": np.eye(4), "eef_pose": np.eye(4),
                          "region_transform": np.eye(4), "tool_id": "gripper", "tool_frame": "eef"}}
            write_json(root / "program.json", program)
            output = root / "run"
            with patch("sys.argv", ["manipulate.py", "--prepared", str(root / "program.json"),
                                    "--output-dir", str(output)]):
                main()
            result = json.loads((output / "trajectory.json").read_text())
            self.assertEqual(result["holding_status"], "requires_verification")
            self.assertEqual(result["status"], "candidate")
            self.assertTrue((output / "trajectory.png").is_file())
            cached = json.loads((output / "observations.json").read_text())
            self.assertEqual(cached["demo"]["trajectory_file"], str(root / "demo.npz"))
            replay = root / "replay"
            with patch("sys.argv", ["manipulate.py", "--prepared", str(output / "observations.json"),
                                    "--output-dir", str(replay), "--no-render"]):
                main()
            replayed = json.loads((replay / "trajectory.json").read_text())
            np.testing.assert_array_equal(replayed["eef_poses"], result["eef_poses"])
    # END compiled-skill CLI regression.

    def test_render_failure_does_not_claim_overall_success(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 0, 3] = [0., .1, .2]
        observation = {
            "task": {"relation": "transfer", "interaction": "none", "motion": "free"},
            "reference_id": "table", "reference_frame": "world", "demo_poses": poses,
            "scene": {"initial_pose": poses[0], "goal_pose": poses[-1],
                      "reference_transform": np.eye(4),
                      "body_points": [[0, 0, 0], [.02, 0, 0], [0, .02, 0]]},
        }
        stages = []
        for index in range(13):
            stage = deepcopy(observation)
            stage.update(carrier_id=f"object-{index}", body_frame_id=f"material-{index}")
            stages.append(stage)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "inputs.json", root / "run"
            write_json(source, {"stages": stages})
            with patch("sys.argv", ["manipulate.py", "--prepared", str(source),
                                    "--output-dir", str(output)]):
                with self.assertRaisesRegex(ValueError, "twelve stages"):
                    main()
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["trajectory_status"], "passed")
            self.assertEqual(report["visualization_status"], "failed")
            self.assertIn("Visualization ValueError", report["error"])
            trajectory = json.loads((output / "trajectory.json").read_text())
            self.assertEqual(len(trajectory["stages"]), 13)


if __name__ == "__main__":
    unittest.main()
