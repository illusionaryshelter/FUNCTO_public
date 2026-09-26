"""Bridge regression with independent tool targets and a simulated legacy chain.

Perception, CUDA and optimization boundaries are mocked. The pure frame helper
is loaded directly from its real source definition, without importing that
legacy module's heavyweight dependencies or duplicating its implementation.
"""

# BEGIN independent-goal and body-frame bridge regression tests.
import ast
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.tool_use import run_tool_use
from utils_IL.params import CAM_MATRIX


def legacy_frame_function():
    path = Path(__file__).resolve().parents[1] / "tool_traj_transfer.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "compute_function_plane_frame")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"np": np, "R_scipy": Rotation}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["compute_function_plane_frame"]


def module(name, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    return result


class ToolUseBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compute_frame = staticmethod(legacy_frame_function())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.demo, self.scene, self.output = (self.root / name for name in ("demo", "scene", "output"))
        for directory in (self.demo, self.scene):
            directory.mkdir()
            calibration = {"width": 1280, "height": 720, "depth_unit_m": 0.001,
                           "intrinsic_matrix": CAM_MATRIX.flatten(order="F").tolist()}
            (directory / "intrinsic.json").write_text(json.dumps(calibration))
        files = {
            "key_point_track_output/keyframe_idx.json": {"func": 1},
            "tool_pose_track_output/tool_track.json": {},
            "func_point_track_output/func_point_2d_init.json": {},
            "grasp_det_output/grasp_points.json": {},
            "cam_to_target_trans_output/cam_to_target_trans.json": {},
            "detection_output/tool.json": {},
        }
        for relative, value in files.items():
            path = self.demo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        np.save(self.demo / "detection_output/tool_mask.npy", np.ones((2, 2), bool))
        self.task = {"relation": "tool_use", "interaction": "tool", "motion": "free",
                     "carrier": {"scene_label": "cup", "demo_label": "cup"},
                     "reference": {"scene_label": "bowl", "demo_label": "bowl"},
                     "goal_description": "The visible demonstration pours into the bowl."}
        self.start_angles, self.goal_angles = np.array([.2, -.4, .3]), np.array([.4, -.3, 1.1])
        self.start_position, self.goal_position = np.array([.4, -.2, .7]), np.array([.6, .3, .5])
        self.start_rotation = Rotation.from_euler("xyz", self.start_angles).as_matrix()
        self.goal_rotation = Rotation.from_euler("xyz", self.goal_angles).as_matrix()
        self.local_points = np.array([[.02, .03, .04], [.04, .03, .04],
                                      [.02, .06, .04], [.02, .03, .08]])
        self.cloud = types.SimpleNamespace(points=self.local_points @ self.start_rotation.T + self.start_position)
        self.target = {
            "function_func": self.goal_position.tolist(),
            "center_func": (self.goal_position + self.goal_rotation[:, 0] * .12).tolist(),
            "grasp_func": (self.goal_position + self.goal_rotation[:, 1] * .15).tolist(),
        }
        # A small accepted optimization residual must remain visible. A bridge
        # that takes its goal from the output would erase it and fail this test.
        self.contact_position = self.goal_position + [2e-7, 0, 0]
        self.contact_angles = self.goal_angles.copy()
        self.mutate_target_after_optimization = False
        self.frame_helper = Mock(side_effect=self.compute_frame)
        self.client = Mock(chat_json=Mock(return_value={"action": "pour"}))

        def prepare_run(source, destination):
            destination.mkdir(parents=True)

        def pose_transfer(demo, destination, *args):
            path = destination / "tool_pose_transfer_output/test_tool_track.json"
            path.parent.mkdir()
            path.write_text(json.dumps(self.target))

        def trajectory_transfer(demo, destination):
            old = np.array([[self.start_position, self.start_angles],
                            [self.contact_position, self.contact_angles],
                            [self.goal_position + [0, .2, .1], self.goal_angles + [0, 0, .2]]])
            path = destination / "tool_traj_transfer_output/test_tool_trajectory.json"
            path.parent.mkdir()
            path.write_text(json.dumps(old.tolist()))
            if self.mutate_target_after_optimization:
                changed = {key: (np.asarray(value) + 10).tolist() for key, value in self.target.items()}
                (destination / "tool_pose_transfer_output/test_tool_track.json").write_text(json.dumps(changed))

        self.optimizer = Mock(side_effect=trajectory_transfer)
        self.fake_torch = module("torch", manual_seed=Mock(), set_num_threads=Mock(),
                                 cuda=types.SimpleNamespace(is_available=Mock(return_value=False)))
        self.modules = {
            "matplotlib": module("matplotlib", use=Mock()),
            "torch": self.fake_torch,
            "open3d": module("open3d", utility=types.SimpleNamespace(random=types.SimpleNamespace(seed=Mock()))),
            "main": module("main", prepare_run=prepare_run,
                           validate_outputs=Mock(return_value={"trajectory_shape": [3, 2, 3]})),
            "cam_to_target_trans": module("cam_to_target_trans", cam_to_target_trans=Mock(),
                                           demo_test_deteciton=Mock()),
            "func_point_transfer": module("func_point_transfer", func_point_transfer=Mock()),
            "grasp_point_transfer": module("grasp_point_transfer", grasp_point_transfer=Mock()),
            "tool_pose_transfer": module("tool_pose_transfer", tool_pose_transfer=Mock(side_effect=pose_transfer)),
            "tool_traj_transfer": module("tool_traj_transfer", compute_function_plane_frame=self.frame_helper,
                                          pc_viz=Mock(return_value=(None, self.cloud)),
                                          tool_trajectory_transfer=self.optimizer),
        }

    def run_bridge(self):
        with patch.dict("sys.modules", self.modules):
            return run_tool_use(self.demo, self.scene, self.task, "Repeat the demonstration",
                                self.output, client=self.client)

    def test_independent_goal_and_fixed_function_body_coordinates(self):
        result = self.run_bridge()
        expected = np.eye(4)
        expected[:3, :3], expected[:3, 3] = self.goal_rotation, self.goal_position
        np.testing.assert_allclose(result["goal_pose"], expected, atol=1e-12)
        np.testing.assert_allclose(result["body_points"], self.local_points, atol=1e-12)
        reconstructed = result["body_points"] @ result["poses"][0, :3, :3].T + result["poses"][0, :3, 3]
        np.testing.assert_allclose(reconstructed, self.cloud.points, atol=1e-12)
        self.assertEqual(result["goal_frame_index"], 1)
        self.assertEqual(result["reference_frame"], "legacy_target")
        self.assertAlmostEqual(result["goal_checks"]["goal_position_error_m"], 2e-7, places=12)
        self.assertLess(result["goal_checks"]["goal_rotation_error_rad"], 1e-12)
        self.assertEqual(result["goal_checks"]["trajectory_shape"], [3, 2, 3])
        self.frame_helper.assert_called_once()
        self.assertIn(self.task["goal_description"], self.client.chat_json.call_args.args[0])
        saved = json.loads((self.output / "trajectory.json").read_text())
        np.testing.assert_allclose(saved["goal_pose"], expected, atol=1e-12)
        self.assertFalse(np.array_equal(result["goal_pose"], result["poses"][1]))

    def test_target_is_captured_before_optimization(self):
        self.mutate_target_after_optimization = True
        result = self.run_bridge()
        np.testing.assert_allclose(result["goal_pose"][:3, 3], self.goal_position, atol=1e-12)
        self.assertAlmostEqual(result["goal_checks"]["goal_position_error_m"], 2e-7, places=12)

    def test_position_miss_cannot_pass_by_redefining_goal_from_result(self):
        self.contact_position = self.goal_position + [.01, 0, 0]
        with self.assertRaisesRegex(ValueError, "independent goal.*position"):
            self.run_bridge()
        self.assertFalse((self.output / "trajectory.json").exists())

    def test_rotation_miss_is_checked_as_rotation_not_euler_component_equality(self):
        self.contact_angles = self.goal_angles + [0, 0, .03]
        with self.assertRaisesRegex(ValueError, "independent goal.*rotation"):
            self.run_bridge()
        self.assertFalse((self.output / "trajectory.json").exists())

    def test_equivalent_euler_branch_preserves_the_same_goal_rotation(self):
        self.contact_angles = self.goal_angles + [0, 0, 2 * np.pi]
        result = self.run_bridge()
        self.assertLess(result["goal_checks"]["goal_rotation_error_rad"], 1e-12)

    def test_degenerate_goal_rejected_before_optimization(self):
        self.target["grasp_func"] = self.target["center_func"].copy()
        with self.assertRaisesRegex(ValueError, "noncollinear"):
            self.run_bridge()
        self.optimizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
# END independent-goal and body-frame bridge regression tests.
