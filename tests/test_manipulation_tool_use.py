"""Compiled tool interface and its real functional-event geometry core.

Only the numerical optimizer is injected where noted. Functional frames,
two-sided target grounding, timeline conversion and independent output checks
remain real. No legacy annotations, vision weights or network calls are used.
"""

# BEGIN compiled tool-program bridge tests.
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.pipeline import write_json
from manipulation.tool_use import prepare_tracked_demo, run_tool_use


def pose(angles=(0., 0., 0.), translation=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    result[:3, 3] = translation
    return result


def fixture():
    frames = np.arange(11) * 3 + 7
    source = np.array([pose((.2, -.1, .3), (.4 + .002*i, -.2 + .0005*i, .7))
                       for i in range(len(frames))])
    cd, cn = pose((.1, .2, -.3), (.03, -.02, .01)), pose((-.2, .1, .4), (-.02, .03, .04))
    rd, rn = pose((.3, -.2, .1), (.1, .04, -.03)), pose((-.4, .1, -.2), (-.06, .02, .1))
    reference_pose = pose((-.2, .3, -.1), (.6, -.4, .2))
    event = {"event_id": "functional_effect", "frame_index": int(frames[7]),
             "carrier_frame": cd, "reference_frame": rd,
             "source": "compiled observed part frames and effect event",
             "position_tolerance_m": 1e-6, "rotation_tolerance_rad": 1e-6}
    program = {"skill_kind": "manipulation", "task": {"relation": "tool_use", "motion": "free"},
               "demo": {"frame_indices": frames, "timestamps": 1. + np.arange(11) * .05,
                        "T_reference_carrier": source, "functional_events": [event]}}
    mapping = rn @ np.linalg.inv(rd)
    initial = reference_pose @ mapping @ source[0] @ cd @ np.linalg.inv(cn)
    scene = {"initial_pose": initial, "reference_frame": "observed_camera", "reference_pose": reference_pose,
             "carrier_id": "new_functional_object", "reference_id": "current_target",
             "body_points": [[x, y, z] for x in (-.03, .03) for y in (-.02, .02) for z in (0., .08)],
             "functional_bindings": {"functional_effect": {
                 "carrier_id": "new_functional_object", "reference_id": "current_target",
                 "carrier_frame": cn, "reference_frame": rn, "source": "current observed functional regions"}}}
    reference = np.array([pose((.4, -.2, .1+.01*i), (.1+.003*i, -.3, .2)) for i in range(11)])
    carrier = reference @ source
    tracking = {"frame_indices": frames.copy(), "body_poses": {"carrier": carrier, "reference": reference},
                "poses": reference[0] @ source, "tracker": "analytic fixed-owner observation fixture"}
    return program, scene, tracking


class TrackedProgramTest(unittest.TestCase):
    def test_relative_conversion_removes_reference_once_without_mutating_program(self):
        program, _, tracking = fixture()
        expected = program["demo"]["T_reference_carrier"].copy()
        # Existing stale motion must not override explicitly supplied new
        # observations. Functional frames and event identities remain intact.
        program["demo"]["T_reference_carrier"][:, :3, 3] += 9.
        before = deepcopy(program)
        observed_before = deepcopy(tracking)
        prepared = prepare_tracked_demo(program, tracking)
        np.testing.assert_allclose(prepared["demo"]["T_reference_carrier"], expected, atol=1e-14)
        np.testing.assert_array_equal(program["demo"]["T_reference_carrier"], before["demo"]["T_reference_carrier"])
        np.testing.assert_array_equal(tracking["poses"], observed_before["poses"])
        np.testing.assert_array_equal(prepared["demo"]["frame_indices"], before["demo"]["frame_indices"])
        np.testing.assert_array_equal(prepared["demo"]["timestamps"], before["demo"]["timestamps"])
        self.assertEqual(prepared["demo"]["functional_events"][0]["frame_index"], 28)
        prepared["demo"]["functional_events"][0]["carrier_frame"][:] = 0
        np.testing.assert_array_equal(program["demo"]["functional_events"][0]["carrier_frame"],
                                      before["demo"]["functional_events"][0]["carrier_frame"])

    def test_sparse_frames_are_supported_but_mismatched_ids_and_gauges_reject(self):
        program, _, tracking = fixture()
        wrong_ids = deepcopy(tracking); wrong_ids["frame_indices"] += 1
        with self.assertRaisesRegex(ValueError, "exactly match.*frame_indices"):
            prepare_tracked_demo(program, wrong_ids)
        missing_ids = deepcopy(tracking); missing_ids["frame_indices"] = missing_ids["frame_indices"][:-1]
        with self.assertRaisesRegex(ValueError, "exactly match.*frame_indices"):
            prepare_tracked_demo(program, missing_ids)
        wrong_gauge = deepcopy(tracking)
        wrong_gauge["poses"] = program["demo"]["T_reference_carrier"].copy()
        with self.assertRaisesRegex(ValueError, "initial-reference gauge"):
            prepare_tracked_demo(program, wrong_gauge)
        wrong_count = deepcopy(tracking)
        wrong_count["body_poses"]["carrier"] = wrong_count["body_poses"]["carrier"][:-1]
        with self.assertRaisesRegex(ValueError, "complete compiled frame_indices"):
            prepare_tracked_demo(program, wrong_count)
        wrong_time = deepcopy(tracking); wrong_time["timestamps"] = program["demo"]["timestamps"] + .1
        with self.assertRaisesRegex(ValueError, "timestamps disagree"):
            prepare_tracked_demo(program, wrong_time)

    def test_missing_compiled_functional_fields_are_explicit_not_reinferred(self):
        program, _, tracking = fixture()
        del program["demo"]["T_reference_carrier"]
        prepared = prepare_tracked_demo(program, tracking)
        self.assertEqual(prepared["demo"]["T_reference_carrier"].shape, (11, 4, 4))
        del program["demo"]["functional_events"][0]["carrier_frame"]
        with self.assertRaisesRegex(ValueError, "functional_events\\[0\\].*carrier_frame"):
            prepare_tracked_demo(program, tracking)


class CompiledToolBridgeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.demo, self.scene, self.output = [self.root / name for name in ("demo", "scene", "output")]
        self.demo.mkdir(); self.scene.mkdir()
        self.program, self.current, self.tracking = fixture()
        self.program_path = self.demo / "interaction_program.json"
        self.scene_path = self.scene / "interaction_scene.json"
        self.save_inputs()
        self.corrupt_position = 0.
        self.corrupt_rotation = 0.
        self.optimizer = Mock(side_effect=self.solve)

    def save_inputs(self):
        write_json(self.program_path, self.program)
        write_json(self.scene_path, self.current)

    def solve(self, reference, initial, function_keyframes):
        # The known reference already satisfies the fixture's target. This
        # injected solver boundary preserves the real transfer wrapper's
        # equality and increment validation without launching IPOPT here.
        output = np.asarray(reference).copy()
        output[0] = initial
        self.assertEqual(len(reference), 11)
        for goal in function_keyframes:
            output[goal["index"]] = goal["pose"]
            output[goal["index"], 0, 0] += self.corrupt_position
            output[goal["index"], 1, 2] += self.corrupt_rotation
        return output

    def run_bridge(self, tracking=None):
        with patch("manipulation.function_motion.optimize_trajectory", self.optimizer):
            return run_tool_use(self.demo, self.scene, self.program["task"],
                                "Apply the observed functional edge to the surface", self.output,
                                tracking=tracking)

    def expected_goal(self):
        event = self.program["demo"]["functional_events"][0]
        binding = self.current["functional_bindings"]["functional_effect"]
        return (self.current["reference_pose"] @ binding["reference_frame"]
                @ np.linalg.inv(event["reference_frame"])
                @ self.program["demo"]["T_reference_carrier"][7] @ event["carrier_frame"]
                @ np.linalg.inv(binding["carrier_frame"]))

    def test_compiled_source_calls_shared_function_core_and_preserves_internal_event(self):
        before = self.program_path.read_text()
        result = self.run_bridge()
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["output_type"], "object_trajectory")
        self.assertNotIn("eef_poses", result)
        self.assertNotIn("holding", result)
        np.testing.assert_allclose(result["poses"][0], self.current["initial_pose"], atol=1e-13)
        np.testing.assert_allclose(result["poses"][7], self.expected_goal(), atol=1e-13)
        self.assertEqual(result["function_checks"][0]["frame_index"], 28)
        self.assertEqual(result["function_checks"][0]["constraint_kind"], "functional_frame")
        self.assertGreater(np.linalg.norm(result["poses"][-1]-self.expected_goal()), .001)
        np.testing.assert_array_equal(result["frame_indices"], self.program["demo"]["frame_indices"])
        self.optimizer.assert_called_once()
        self.assertEqual(self.program_path.read_text(), before)
        self.assertEqual(result["input_evidence"]["motion_source"], "compiled demonstration")
        saved = json.loads((self.output / "trajectory.json").read_text())
        np.testing.assert_allclose(saved["function_checks"][0]["goal_pose"], self.expected_goal(), atol=1e-13)

    def test_only_compiled_json_is_read_without_legacy_landmarks_or_named_action_dispatch(self):
        # Make obsolete files deliberately invalid. Opening them would fail,
        # while their existence must have no influence on this production path.
        for relative in ("tool_pose_track_output/tool_track.json", "key_point_track_output/keyframe_idx.json",
                         "func_point_track_output/func_point_2d_init.json", "grasp_det_output/grasp_points.json"):
            path = self.demo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not valid JSON and must never be read")
        allowed = {self.program_path, self.scene_path}
        original_read = Path.read_text
        read_paths = []
        def read(path, *args, **kwargs):
            self.assertIn(path, allowed)
            read_paths.append(path)
            return original_read(path, *args, **kwargs)
        with patch.object(Path, "read_text", autospec=True, side_effect=read):
            result = self.run_bridge()
        self.assertEqual(set(read_paths), allowed)
        self.assertEqual(result["status"], "candidate")
        self.assertFalse((self.output / "tracked_demo").exists())
        self.assertFalse((self.output / "tool_use").exists())

    def test_supplied_tracking_replaces_only_motion_in_memory(self):
        expected = self.expected_goal()
        self.program["demo"]["T_reference_carrier"][:, :3, 3] += 20.
        self.save_inputs()
        before = self.program_path.read_text()
        result = self.run_bridge(self.tracking)
        np.testing.assert_allclose(result["poses"][7], expected, atol=1e-13)
        self.assertEqual(result["input_evidence"]["motion_source"], "current tracked demonstration")
        self.assertEqual(self.program_path.read_text(), before)

    def test_optimizer_cannot_report_pass_after_missing_function_position_or_direction(self):
        for position, angle in ((.01, 0.), (0., .03)):
            self.corrupt_position, self.corrupt_rotation = position, angle
            with self.subTest(position=position, angle=angle), self.assertRaisesRegex(
                    ValueError, "independent equality"):
                self.run_bridge()
            self.assertFalse((self.output / "trajectory.json").exists())

    def test_missing_compilation_metadata_and_event_timeline_fail_before_solver(self):
        for mutate, message in (
            (lambda: self.program["demo"].pop("functional_events"), "functional_events"),
            (lambda: self.program["demo"].pop("T_reference_carrier"), "T_reference_carrier"),
            (lambda: self.program["demo"]["functional_events"][0].update(frame_index=8), "original demonstration frame"),
            (lambda: self.current.pop("reference_pose"), "reference_pose"),
        ):
            self.program, self.current, self.tracking = fixture()
            mutate(); self.save_inputs()
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.run_bridge()
        self.optimizer.assert_not_called()
        self.program_path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing compiled interaction program"):
            self.run_bridge()

    def test_requested_or_compiled_constraints_are_not_erased_and_output_is_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, "requested constrained motion"):
            run_tool_use(self.demo, self.scene, {"relation": "tool_use", "motion": "planar"},
                         "Retain the plane", self.output)
        self.program["task"]["motion"] = "prismatic"; self.save_inputs()
        with self.assertRaisesRegex(ValueError, "constrained motion"):
            self.run_bridge()
        self.program, self.current, self.tracking = fixture(); self.save_inputs()
        self.run_bridge()
        saved = (self.output / "trajectory.json").read_text()
        with self.assertRaises(FileExistsError):
            self.run_bridge()
        self.assertEqual((self.output / "trajectory.json").read_text(), saved)


if __name__ == "__main__":
    unittest.main()
# END compiled tool-program bridge tests.
