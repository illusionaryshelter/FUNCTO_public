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
        # BEGIN observed line is visible in diagnostics without becoming a rail.
        structure = result["motion_checks"]["observed_structure"]
        self.assertIn("prismatic", structure["compatible_candidates"])
        self.assertEqual(result["motion_checks"]["applied_constraint"]["kind"], "free")
        # END observed/applied line regression.

    # BEGIN whole-stage planar semantics and structure evidence are independent.
    def test_tilted_se2_candidate_is_retained_and_applied_only_when_requested(self):
        phase = np.linspace(0., 1., 25)
        local = np.repeat(np.eye(4)[None], len(phase), axis=0)
        local[:, :3, :3] = Rotation.from_euler("z", .8*phase).as_matrix()
        local[:, :3, 3] = np.column_stack((.2*phase, .1*np.sin(4*phase), .3*np.ones(len(phase))))
        placement = pose([.3, -.2, .1])
        placement[:3, :3] = Rotation.from_euler("xyz", [.4, -.6, .2]).as_matrix()
        trajectory = placement @ local
        observed = stage(motion="auto")
        observed["demo_poses"] = trajectory
        observed["frame_indices"] = np.arange(len(phase))*3
        observed["scene"].update(initial_pose=trajectory[0], goal_pose=trajectory[-1])

        unconstrained = run_stage(observed)
        self.assertEqual(unconstrained["model"]["kind"], "free")
        structure = unconstrained["motion_checks"]["observed_structure"]
        self.assertIn("planar", structure["compatible_candidates"])
        self.assertEqual(structure["preferred_description"], "planar")

        observed["task"]["motion"] = "planar"
        planar = run_stage(observed)
        self.assertEqual(planar["model"]["kind"], "planar")
        self.assertEqual(planar["motion_checks"]["applied_constraint"]["requested_motion"], "planar")
        np.testing.assert_allclose(planar["poses"], trajectory, atol=1e-10)
        np.testing.assert_array_equal(planar["frame_indices"], observed["frame_indices"])
        self.assertEqual(np.asarray(planar["model"]["coordinates"]).shape, (len(phase), 3))
        normal = placement[:3, 2]
        displacement = planar["poses"][:, :3, 3] - trajectory[0, :3, 3]
        np.testing.assert_allclose(displacement @ normal, 0., atol=1e-10)

        # A known planar task cannot become free when an observed frame tilts.
        observed["demo_poses"] = trajectory.copy()
        observed["demo_poses"][12, :3, :3] = (
            placement[:3, :3] @ Rotation.from_euler("x", .3).as_matrix()
            @ local[12, :3, :3])
        with self.assertRaisesRegex(ValueError, "incompatible"):
            run_stage(observed)

    def test_fixed_orientation_observations_can_support_a_planar_task(self):
        # Zero observed yaw does not disprove an SE(2) task; a sufficiently
        # spread 2-D path establishes a plane and the task requests allowed yaw.
        phase = np.linspace(0., 1., 25)
        trajectory = np.array([pose([.2*t, .1*np.sin(4*t), .3]) for t in phase])
        observed = stage(motion="planar")
        observed["demo_poses"] = trajectory
        goal = trajectory[-1].copy()
        goal[:3, :3] = Rotation.from_euler("z", .4).as_matrix()
        observed["scene"].update(initial_pose=trajectory[0], goal_pose=goal)
        result = run_stage(observed)
        self.assertEqual(result["model"]["kind"], "planar")
        structure = result["motion_checks"]["observed_structure"]
        self.assertIn("translation_plane", structure["compatible_candidates"])
        self.assertIn("planar", structure["compatible_candidates"])
        np.testing.assert_allclose(result["poses"][-1], goal, atol=1e-10)
        np.testing.assert_allclose(result["poses"][:, 2, 3], .3, atol=1e-10)

    def test_static_auto_object_observations_are_valid_but_structure_is_undetermined(self):
        observed = stage(motion="auto")
        initial = observed["scene"]["initial_pose"].copy()
        observed["demo_poses"] = np.repeat(initial[None], 8, axis=0)
        observed["scene"]["goal_pose"] = initial
        result = run_stage(observed)
        structure = result["motion_checks"]["observed_structure"]
        self.assertEqual(structure["status"], "insufficient_excitation")
        self.assertIsNone(structure["preferred_description"])
        np.testing.assert_allclose(result["poses"], observed["demo_poses"])

        observed["task"]["relation"] = "articulation"
        with self.assertRaisesRegex(ValueError, "excited"):
            run_stage(observed)

    def test_final_on_relation_does_not_constrain_the_airborne_carry_to_se2(self):
        observed = stage("on", "auto", "grasp")
        observed["scene"]["support"] = support()
        observed["scene"]["goal_pose"] = pose([.2, .1, .02])
        phase = np.linspace(0., 1., len(observed["demo_poses"]))
        observed["demo_poses"][:, :3, 3] = np.column_stack((
            .2*phase, .1*phase,
            .4-.38*phase + .08*np.sin(np.pi*phase)*np.sin(4*np.pi*phase)))
        result = run_stage(observed)
        self.assertEqual(result["model"]["kind"], "free")
        self.assertGreater(result["poses"][0, 2, 3], .3)
        self.assertAlmostEqual(result["poses"][-1, 2, 3], .02)
        self.assertLess(result["goal_checks"]["support_gap_m"], 1e-10)

    def test_articulation_screw_motion_cannot_downgrade_to_free(self):
        phase = np.linspace(0., 1.2, 25)
        trajectory = np.array([pose([0., 0., .1*t]) for t in phase])
        trajectory[:, :3, :3] = Rotation.from_euler("z", phase).as_matrix()
        observed = stage("articulation", "auto", "pull")
        observed["demo_poses"] = trajectory
        observed["scene"].update(initial_pose=trajectory[0], goal_pose=trajectory[-1])
        with self.assertRaisesRegex(ValueError, "Articulation"):
            run_stage(observed)
    # END planar semantics and structure evidence.

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

    # BEGIN inside transfer and optional known-volume certification regressions.
    def test_inside_transfers_bound_endpoint_without_unobserved_cavity_input(self):
        observed = stage("inside")
        reference, carrier = pose([.6, -.2, .1]), pose([-.1, .3, .2])
        reference[:3, :3] = Rotation.from_euler("z", .3).as_matrix()
        carrier[:3, :3] = Rotation.from_euler("z", -.2).as_matrix()
        expected = reference @ observed["demo_poses"][-1]
        observed["scene"].pop("goal_pose")
        observed["scene"].update(
            initial_pose=carrier @ observed["demo_poses"][0],
            reference_transform=reference, preferred_goal_pose=expected,
            goal_source="demonstrated endpoint bound through visible reference correspondences")
        result = run_stage(observed)
        np.testing.assert_allclose(result["poses"][0], observed["scene"]["initial_pose"])
        np.testing.assert_allclose(result["poses"][-1], expected)
        self.assertEqual(result["task"]["relation"], "inside")
        self.assertIsNone(result["goal_geometry"])
        self.assertIsNone(result["interaction_geometry"])
        checks = result["goal_checks"]
        self.assertEqual(checks["containment_status"], "not_verified")
        self.assertEqual(checks["goal_source"], observed["scene"]["goal_source"])
        self.assertNotIn("interior_violation_m", checks)
        self.assertNotIn("minimum_clearance_m", checks)
        self.assertNotIn("checked_points", checks)

    def test_inside_known_interior_respects_exact_goal_and_whole_object(self):
        observed = stage("inside")
        observed["scene"]["interior"] = {"vertices": box([.1, .1, .1]), "source": "known free cavity"}
        with self.assertRaisesRegex(ValueError, "Exact goal"):
            run_stage(observed)
        observed["scene"]["preferred_goal_pose"] = observed["scene"].pop("goal_pose")
        result = run_stage(observed)
        self.assertLess(result["goal_checks"]["interior_violation_m"], 1e-8)
        self.assertEqual(result["goal_checks"]["containment_status"], "verified_on_supplied_points")
        observed["scene"]["body_points"] = box([.2, .2, .2])
        with self.assertRaisesRegex(ValueError, "does not fit"):
            run_stage(observed)

    def test_inside_invalid_supplied_interior_is_not_ignored(self):
        observed = stage("inside")
        observed["scene"]["interior"] = {
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            "source": "coplanar observations do not enclose free volume"}
        with self.assertRaisesRegex(ValueError, "do not enclose"):
            run_stage(observed)

    def test_inside_unknown_cavity_cannot_silently_ignore_margin(self):
        observed = stage("inside")
        observed["scene"]["margin"] = .005
        with self.assertRaisesRegex(ValueError, "margin needs known"):
            run_stage(observed)
        observed["scene"].update(
            entry_region={"vertices": box([.3, .3, .5]), "source": "known entry corridor"},
            entry_start_index=0)
        result = run_stage(observed)
        checks = result["goal_checks"]
        self.assertEqual(checks["containment_status"], "not_verified")
        self.assertNotIn("minimum_clearance_m", checks)
        self.assertEqual(checks["entry_corridor_margin_m"], .005)
        self.assertEqual(checks["entry_corridor_checked_nodes"], len(observed["demo_poses"]))
    # END inside transfer and optional known-volume certification regressions.

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

        # BEGIN an unverified inside relation still cannot imply attachment.
        placement["task"]["relation"] = "inside"
        placement["scene"].pop("support")
        result = run_sequence([opening, placement, closing])
        inside_stage = result["stages"][1]
        np.testing.assert_allclose(inside_stage["poses"][-1, :3, 3], [.3, 0, .02])
        self.assertEqual(inside_stage["goal_checks"]["containment_status"], "not_verified")
        self.assertNotIn("object", result["final_entity_poses"])
        self.assertEqual(result["unobserved_entity_states"]["object"]["status"], "needs_observation")
        # END an unverified inside relation still cannot imply attachment.

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
