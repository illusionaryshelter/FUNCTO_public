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
