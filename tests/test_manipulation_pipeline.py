"""Whole-stage and state-handoff tests with independently specified geometry.

These are numerical manipulation cases, not evidence of real model perception
or physical robot execution. Fixtures need neither function nor grasp points.
"""

from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.pipeline import run_prepared, run_sequence, run_stage, write_json


def box(half):
    return np.asarray(list(itertools.product(*[(-x, x) for x in half])))


def pose(position=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, 3] = position
    return result


def stage(relation="transfer", motion="free", interaction="none"):
    initial, goal = pose([0., 0., .4]), pose([.2, .1, .2])
    trajectory = np.repeat(initial[None], 12, axis=0)
    trajectory[:, :3, 3] = np.linspace(initial[:3, 3], goal[:3, 3], len(trajectory))
    return {"task": {"relation": relation, "motion": motion, "interaction": interaction},
            "carrier_id": "object", "reference_id": "support", "reference_frame": "world",
            "body_frame_id": "object/material", "demo_poses": trajectory,
            "evidence": {"source": "analytic fixture; no perception inference"},
            "scene": {"initial_pose": initial, "reference_transform": np.eye(4),
                      "body_points": box([.04, .03, .02]), "goal_pose": goal}}


def support():
    return {"origin": [0., 0., 0.], "normal": [0., 0., 1.], "basis": [[1., 0.], [0., 1.], [0., 0.]],
            "boundary": [[-.3, -.3], [.3, -.3], [.3, .3], [-.3, .3]], "source": "known patch"}


class ManipulationPipelineTest(unittest.TestCase):
    def test_free_transfer_needs_no_tool_or_grasp_and_preserves_pose_contract(self):
        observed = stage()
        result = run_stage(observed)
        np.testing.assert_allclose(result["poses"][0], observed["scene"]["initial_pose"])
        np.testing.assert_allclose(result["poses"][-1], observed["scene"]["goal_pose"])
        self.assertEqual(result["output_type"], "object_trajectory")
        self.assertIsNone(result["interaction_geometry"])
        self.assertNotIn("function", result)

    def test_auto_free_carry_does_not_become_a_physical_rail(self):
        observed = stage(motion="auto")
        observed["scene"]["goal_pose"] = pose([.1, .4, .25])
        result = run_stage(observed)
        self.assertEqual(result["model"]["kind"], "free")
        np.testing.assert_allclose(result["poses"][-1][:3, 3], [.1, .4, .25])

    def test_push_slide_without_grasp_and_incompatible_goal(self):
        observed = stage("articulation", "prismatic", "push")
        observed["demo_poses"] = np.array([pose([x, 0, 0]) for x in np.linspace(.3, 0, 20)])
        observed["scene"]["initial_pose"] = pose([.25, 0, 0])
        observed["scene"]["goal_pose"] = pose()
        result = run_stage(observed)
        np.testing.assert_allclose(result["poses"][:, 1:3, 3], 0, atol=1e-10)
        self.assertIsNone(result["interaction_geometry"])
        observed["scene"]["goal_pose"][1, 3] = .01
        with self.assertRaises(ValueError):
            run_stage(observed)

    def test_arbitrary_hinge_endpoint_and_arc(self):
        axis = np.array([1., 2., -1.]); axis /= np.linalg.norm(axis)
        center, offset = np.array([.3, -.1, .2]), np.array([.1, .2, .3])
        rotations = Rotation.from_rotvec(np.linspace(0., 1.4, 20)[:, None]*axis).as_matrix()
        trajectory = np.repeat(np.eye(4)[None], len(rotations), axis=0)
        trajectory[:, :3, :3] = rotations
        trajectory[:, :3, 3] = center + np.einsum("nij,j->ni", rotations, offset)
        observed = stage("articulation", "revolute", "push")
        observed["demo_poses"] = trajectory
        observed["scene"]["initial_pose"], observed["scene"]["goal_pose"] = trajectory[0], trajectory[-1]
        result = run_stage(observed)
        np.testing.assert_allclose(result["poses"], trajectory, atol=1e-8)

    def test_support_estimate_is_adjusted_but_exact_goal_is_not(self):
        observed = stage("on")
        observed["scene"]["support"] = support()
        with self.assertRaisesRegex(ValueError, "Exact goal"):
            run_stage(observed)
        observed["scene"]["preferred_goal_pose"] = observed["scene"].pop("goal_pose")
        result = run_stage(observed)
        self.assertAlmostEqual(result["poses"][-1, 2, 3], .02)
        self.assertLess(result["goal_checks"]["support_gap_m"], 1e-8)

    def test_inside_requires_real_interior_and_respects_exact_goal(self):
        observed = stage("inside")
        with self.assertRaisesRegex(ValueError, "free interior"):
            run_stage(observed)
        observed["scene"]["interior"] = {"vertices": box([.1, .1, .1]), "source": "known free cavity"}
        with self.assertRaisesRegex(ValueError, "Exact goal"):
            run_stage(observed)
        observed["scene"]["preferred_goal_pose"] = observed["scene"].pop("goal_pose")
        result = run_stage(observed)
        self.assertLess(result["goal_checks"]["interior_violation_m"], 1e-8)

    def test_exact_support_goal_obeys_margin_without_moving_it(self):
        observed = stage("on")
        observed["scene"].update(support=support(), goal_pose=pose([.25, 0, .02]), margin=.02)
        # The object's rightmost point is x=.29: geometrically inside, but
        # only 1 cm from an edge when this task requests 2 cm of clearance.
        with self.assertRaisesRegex(ValueError, "Exact goal"):
            run_stage(observed)
        observed["scene"]["goal_pose"] = pose([.23, 0, .02])
        result = run_stage(observed)
        self.assertAlmostEqual(result["goal_checks"]["minimum_boundary_clearance_m"], .03)
        np.testing.assert_allclose(result["goal_pose"], observed["scene"]["goal_pose"])

    def test_invalid_margin_cannot_bypass_exact_goal_validation(self):
        for margin in (float("nan"), float("inf"), -.01, "0", [0], True):
            observed = stage("on")
            observed["scene"].update(support=support(), goal_pose=pose([0, 0, .02]), margin=margin)
            with self.subTest(margin=margin), self.assertRaisesRegex(ValueError, "Geometry margin"):
                run_stage(observed)

    def test_entry_corridor_checks_intermediate_nodes_and_margin(self):
        observed = stage()
        observed["scene"].update(
            entry_region={"vertices": box([.3, .3, .5]), "source": "known free entry corridor"},
            entry_start_index=0)
        observed["demo_poses"][5, 0, 3] = .4
        with self.assertRaisesRegex(ValueError, "entry corridor"):
            run_stage(observed)
        observed["demo_poses"][5, 0, 3] = .25
        observed["scene"]["margin"] = .02
        with self.assertRaisesRegex(ValueError, "entry corridor"):
            run_stage(observed)
        observed["scene"]["margin"] = .005
        result = run_stage(observed)
        self.assertEqual(result["goal_checks"]["entry_corridor_checked_nodes"], 12)
        self.assertAlmostEqual(result["goal_checks"]["entry_corridor_minimum_clearance_m"], .01)
        self.assertIn("saved nodes only", result["goal_checks"]["entry_corridor_scope"])

    def test_reference_state_is_used_for_placement_and_passive_state_is_invalidated(self):
        opening = stage("articulation", "prismatic", "pull")
        opening.update(carrier_id="drawer", reference_id="cabinet", body_frame_id="drawer/material")
        opening["demo_poses"] = np.array([pose([x, 0, 0]) for x in np.linspace(0, .3, 12)])
        opening["scene"].update(initial_pose=pose(), goal_pose=pose([.3, 0, 0]))
        placement = stage("on")
        placement.update(reference_id="drawer", reference_body_frame_id="drawer/material")
        placement["scene"].update(reference_pose=pose(), support=support(), goal_pose=pose([0, 0, .02]))
        closing = deepcopy(opening)
        closing["demo_poses"] = opening["demo_poses"][::-1].copy()
        closing["scene"].update(initial_pose=pose([.3, 0, 0]), goal_pose=pose())
        result = run_sequence([opening, placement, closing])
        np.testing.assert_allclose(result["stages"][1]["poses"][-1, :3, 3], [.3, 0, .02])
        np.testing.assert_allclose(result["final_entity_poses"]["drawer"], pose(), atol=1e-8)
        self.assertNotIn("object", result["final_entity_poses"])
        self.assertEqual(result["unobserved_entity_states"]["object"]["status"], "needs_observation")
        with self.assertRaisesRegex(ValueError, "fresh registered observation"):
            run_sequence([opening, placement, closing, stage()])

    def test_repeated_carrier_body_frames_must_match(self):
        first, second = stage(), stage()
        second["body_frame_id"] = "different_crop_centroid"
        with self.assertRaisesRegex(ValueError, "fixed body frame"):
            run_sequence([first, second])

    def test_invalid_frame_indices_are_rejected(self):
        for value in (["x"]*12, [float("nan")]*12, list(range(-1, 11)), list(range(11))):
            observed = stage()
            observed["frame_indices"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Frame indices"):
                run_stage(observed)

    def test_prepared_entry_saves_result_and_failure_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            write_json(source, stage())
            result = run_prepared(source, root / "run")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(json.loads((root/"run/run.json").read_text())["status"], "passed")
            with self.assertRaises(FileExistsError):
                run_prepared(source, root/"run")
            source.write_text('{"demo_poses": NaN}')
            with self.assertRaises(ValueError):
                run_prepared(source, root/"bad")
            self.assertEqual(json.loads((root/"bad/run.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
