"""End-to-end CPU counterexamples for nominal-reference and task-set separation.

Each successful case supplies a concrete noncentral member and evaluates the
compiled original requirements. Grounding a seed alone is never the assertion
that proves task acceptance. No robot IK or physical execution is simulated.
"""

# BEGIN original allowed sets survive optional nominal reference generation.
from copy import deepcopy
from itertools import product
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from manipulation.program import ground_program
from manipulation.regions import SupportReferenceNotFound, ground_on
from manipulation.goals import SupportSurface
from tests.test_manipulation_planning_problem import (
    compile_inputs, components, grasp_input, manipulation_input, on_input, pose, prismatic_input,
)


class AllowedSetGroundingTest(unittest.TestCase):
    def evaluate_body(self, request, skill, body, progress):
        """Evaluate independent carrier geometry through the actual attachment."""
        tool = body @ skill["scene"]["holding"]["T_carrier_eef"]
        p, q = components(np.asarray([[tool]]))
        return request.evaluate(p, q, torch.tensor([[progress]], dtype=torch.float64)).item()

    def test_actual_owner_contract_is_explicit_and_independent_of_frozen_world_pose(self):
        grasp = grasp_input()
        request = compile_inputs(grasp)
        actual = request.pose_events[0].actual_relation
        self.assertEqual(set(actual), {"target_owner_id", "T_owner_target"})
        self.assertEqual(actual["target_owner_id"], grasp["scene"]["carrier_id"])
        np.testing.assert_array_equal(actual["T_owner_target"], grasp["scene"]["grasp_goal"]["T_carrier_eef"])
        np.testing.assert_allclose(request.pose_events[0].target,
                                   grasp["scene"]["carrier_pose"] @ actual["T_owner_target"])
        held = manipulation_input()
        request = compile_inputs(held)
        goal = request.pose_events[0]
        actual = goal.actual_relation
        self.assertEqual(actual["function_owner_id"], held["scene"]["carrier_id"])
        self.assertEqual(actual["target_owner_id"], held["scene"]["reference_id"])
        condition = next(row for row in request.conditions if row.get("condition_id") == actual["attachment_condition_id"])
        self.assertEqual(condition["kind"], "observed_attachment")
        np.testing.assert_allclose(np.asarray(condition["T_carrier_eef"]) @ goal.tool_from_function,
                                   actual["T_owner_function"])
        np.testing.assert_allclose(held["scene"]["reference_pose"] @ actual["T_owner_target"], goal.target)
        from dataclasses import replace
        for invalid in ({}, {"target_owner_id": "carrier"},
                        {"target_owner_id": "carrier", "T_owner_target": pose(), "use_initial_pose": True},
                        {"attachment_condition_id": "unowned"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(goal, actual_relation=invalid)

    def test_pose_centre_outside_model_and_limits_keeps_a_valid_allowed_member(self):
        skill = manipulation_input()
        program, scene = skill["program"], skill["scene"]
        program["task"] = {"relation": "articulation", "motion": "prismatic"}
        program["demo"]["functional_events"][0]["position_tolerance_m"] = .01
        scene["functional_bindings"]["function"]["reference_frame"] = pose((0., .006, 0.))
        scene["motion_model"] = {
            "kind": "prismatic", "reference_pose": pose(),
            "basis": Rotation.from_euler("y", np.pi/2).as_matrix(),
            "coordinate_limits": [[0., .035]], "source": "independent slider geometry",
        }
        grounded = ground_program(program, scene)
        self.assertEqual(grounded["motion_checks"]["reference_generation"]["status"], "failed")
        request = compile_inputs(skill)
        goal = request.pose_events[0]
        np.testing.assert_allclose(goal.target, pose((.04, .006, 0.)))
        self.assertEqual(goal.position_tolerance_m, .01)
        self.assertLess(self.evaluate_body(request, skill, pose((.034, 0., 0.)), goal.progress), 1e-12)
        self.assertGreater(self.evaluate_body(request, skill, goal.target, goal.progress), .004)

    def test_simultaneous_pose_centres_preserve_their_overlap_without_averaging(self):
        skill = manipulation_input()
        event = skill["program"]["demo"]["functional_events"][0]
        event["position_tolerance_m"] = .01
        other = deepcopy(event)
        other["event_id"] = "second_part"
        skill["program"]["demo"]["functional_events"].append(other)
        binding = deepcopy(skill["scene"]["functional_bindings"]["function"])
        binding["reference_frame"] = pose((.015, 0., 0.))
        skill["scene"]["functional_bindings"]["second_part"] = binding
        request = compile_inputs(skill)
        self.assertEqual(len(request.pose_events), 2)
        np.testing.assert_allclose([goal.target[0, 3] for goal in request.pose_events], [.04, .055])
        progress = request.pose_events[0].progress
        self.assertLess(self.evaluate_body(request, skill, pose((.0475, 0., 0.)), progress), 1e-12)
        self.assertGreater(self.evaluate_body(request, skill, pose((.04, 0., 0.)), progress), .004)

    def test_material_lever_arm_allows_distinct_body_origins_at_the_same_event(self):
        skill = manipulation_input()
        event = skill["program"]["demo"]["functional_events"][0]
        event.update(carrier_frame=pose((1., 0., 0.)),
                     position_tolerance_m=1e-5, rotation_tolerance_rad=.05)
        other = deepcopy(event)
        other["event_id"] = "second_part"
        skill["program"]["demo"]["functional_events"].append(other)
        for name, angle in (("function", -.04), ("second_part", .04)):
            nominal = pose(angles=(0., 0., angle))
            nominal[:3, 3] = [1.04, 0., 0.] - nominal[:3, :3] @ np.array([1., 0., 0.])
            binding = deepcopy(skill["scene"]["functional_bindings"]["function"])
            binding["carrier_frame"] = pose((1., 0., 0.))
            binding["reference_frame"] = nominal @ pose((-.04, 0., 0.))
            skill["scene"]["functional_bindings"][name] = binding
        request = compile_inputs(skill)
        self.assertLess(self.evaluate_body(request, skill, pose((.04, 0., 0.)),
                                            request.pose_events[0].progress), 1e-12)
        self.assertEqual(len(request.pose_events), 2)

    def test_actual_entry_may_be_a_noncentral_member_but_cannot_be_replaced(self):
        skill = manipulation_input()
        event = skill["program"]["demo"]["functional_events"][0]
        event.update(frame_index=100, position_tolerance_m=.01)
        binding = skill["scene"]["functional_bindings"]["function"]
        binding["reference_frame"] = pose((.006, 0., 0.))
        request = compile_inputs(skill)
        self.assertEqual(request.pose_events[0].progress, 0.)
        np.testing.assert_allclose(request.pose_events[0].target, pose((.006, 0., 0.)))
        self.assertLess(self.evaluate_body(request, skill, pose(), 0.), 1e-12)
        binding["reference_frame"] = pose((.02, 0., 0.))
        with self.assertRaisesRegex(ValueError, "allowed set excludes the observed"):
            compile_inputs(skill)

    def test_coordinate_band_intersects_limits_without_rewriting_its_centre(self):
        skill = prismatic_input()
        skill["program"]["demo"]["functional_events"][0]["coordinate_tolerance"] = .02
        skill["scene"]["functional_bindings"]["function"]["current_coordinates"] = [.31]
        request = compile_inputs(skill)
        goal = request.coordinate_events[0]
        self.assertEqual((goal.absolute_target, goal.tolerance), (.31, .02))
        self.assertLess(self.evaluate_body(request, skill, pose((.295, 0., 0.)), goal.progress), 1e-12)
        self.assertGreater(self.evaluate_body(request, skill, pose((.31, 0., 0.)), goal.progress), .009)

    def test_simultaneous_coordinate_intersection_and_immutable_actual_entry(self):
        skill = prismatic_input()
        event = skill["program"]["demo"]["functional_events"][0]
        event["coordinate_tolerance"] = .02
        other = deepcopy(event)
        other["event_id"] = "second_extent"
        skill["program"]["demo"]["functional_events"].append(other)
        binding = deepcopy(skill["scene"]["functional_bindings"]["function"])
        binding["current_coordinates"] = [.17]
        skill["scene"]["functional_bindings"]["second_extent"] = binding
        request = compile_inputs(skill)
        self.assertEqual([goal.absolute_target for goal in request.coordinate_events], [.14, .17])
        self.assertLess(self.evaluate_body(request, skill, pose((.155, 0., 0.)),
                                            request.coordinate_events[0].progress), 1e-12)
        binding["current_coordinates"] = [.19]
        with self.assertRaisesRegex(ValueError, "allowed sets do not intersect"):
            compile_inputs(skill)
        entry = prismatic_input()
        event = entry["program"]["demo"]["functional_events"][0]
        event.update(frame_index=100, coordinate_value=[0.], coordinate_tolerance=.02)
        binding = entry["scene"]["functional_bindings"]["function"]
        binding["current_coordinates"] = [.11]
        request = compile_inputs(entry)
        self.assertEqual(request.coordinate_events[0].absolute_target, .11)
        self.assertLess(self.evaluate_body(request, entry, pose((.1, 0., 0.)), 0.), 1e-12)
        binding["current_coordinates"] = [.14]
        with self.assertRaisesRegex(ValueError, "allowed set excludes the observed"):
            compile_inputs(entry)

    def test_finite_support_search_failure_preserves_an_unsampled_valid_yaw(self):
        skill = on_input()
        scene = skill["scene"]
        body = np.asarray(list(product((-.1, .1), (-.01, .01), (0., .02))))
        rotation = Rotation.from_euler("z", np.deg2rad(2.5)).as_matrix()
        surface = SupportSurface(np.array([.04, 0., 0.]), rotation[:, :2], rotation[:, 2],
                                 np.array([[-.100001, -.010001], [.100001, -.010001],
                                           [.100001, .010001], [-.100001, .010001]]),
                                 source="independent thin finite support at an unsampled yaw")
        scene["body_points"], scene["support"] = body, surface.to_dict()
        face = scene["functional_bindings"]["function"]["on_geometry"]
        face["carrier_region_points"] = body[body[:, 2] == 0]
        with self.assertRaises(SupportReferenceNotFound):
            ground_on(pose((.04, 0., 0.)), body, face["carrier_region_points"],
                      face["carrier_outward_normal"], surface, gap_tolerance=1e-6, normal_tolerance=1e-6)
        grounded = ground_program(skill["program"], scene)
        self.assertEqual(grounded["function_checks"][0]["reference_generation"]["status"], "failed")
        request = compile_inputs(skill)
        self.assertEqual(len(request.supports), 1)
        self.assertEqual(request.pose_events, ())
        np.testing.assert_array_equal(request.supports[0].surface.boundary, surface.boundary)
        allowed = pose((.04, 0., 0.), (0., 0., np.deg2rad(2.5)))
        self.assertLess(self.evaluate_body(request, skill, allowed, request.supports[0].progress), 1e-12)
        self.assertGreater(self.evaluate_body(request, skill, pose((.04, 0., 0.)),
                                               request.supports[0].progress), .003)
        face["carrier_outward_normal"] = [0., 0., 0.]
        with self.assertRaisesRegex(ValueError, "nonzero"):
            compile_inputs(skill)

    def test_inside_nominal_centre_outside_body_clearance_keeps_valid_goal_member(self):
        skill = manipulation_input()
        skill["program"]["task"]["relation"] = "inside"
        skill["program"]["demo"]["functional_events"][0]["position_tolerance_m"] = .01
        scene = skill["scene"]
        scene["body_points"] = np.asarray(list(product((-.002, .002), repeat=3)))
        scene["interior"] = {"vertices": list(product((-.05, .041), (-.02, .02), (-.02, .02))),
                             "source": "independent convex free interior"}
        request = compile_inputs(skill)
        self.assertEqual((len(request.interiors), len(request.pose_events)), (1, 1))
        goal = request.pose_events[0]
        np.testing.assert_allclose(goal.target, pose((.04, 0., 0.)))
        self.assertLess(self.evaluate_body(request, skill, pose((.035, 0., 0.)), goal.progress), 1e-12)
        self.assertGreater(self.evaluate_body(request, skill, goal.target, goal.progress), .0008)

    def test_exact_normal_face_failure_keeps_a_valid_small_tilt_member(self):
        skill = on_input()
        scene = skill["scene"]
        face = scene["functional_bindings"]["function"]["on_geometry"]
        # The requested small horizontal material patch has a lower protrusion
        # beside it. Exact normal alignment cannot support the patch; a permitted
        # 0.03 rad tilt lifts that protrusion without moving any face point beyond
        # the original 50 micrometre gap band. No surface or margin is enlarged.
        region = np.asarray(list(product((-.1, -.099), (-.01, .01), (0.,))))
        body = np.vstack((region, [[.1, -.01, -.005], [.1, .01, -.005],
                                  [-.1, -.01, .02], [.1, .01, .02]]))
        scene["body_points"] = body
        face.update(carrier_region_points=region, gap_tolerance=5e-5, normal_tolerance=.04)
        grounded = ground_program(skill["program"], scene)
        self.assertIn("exact-normal", grounded["function_checks"][0]["reference_generation"]["reason"])
        request = compile_inputs(skill)
        allowed = pose(angles=(0., -.03, 0.))
        allowed[2, 3] = -(body @ allowed[:3, :3].T)[:, 2].min()
        self.assertLess(self.evaluate_body(request, skill, allowed, request.supports[0].progress), 1e-12)
        self.assertEqual(request.supports[0].normal_tolerance, .04)
        self.assertEqual(request.supports[0].gap_tolerance, 5e-5)
        exact_normal = pose((0., 0., .005))
        self.assertGreater(self.evaluate_body(request, skill, exact_normal, request.supports[0].progress), .004)

    def test_missing_antiparallel_seed_branch_keeps_valid_support_members(self):
        skill = on_input()
        # The source preference reaches an upside-down pose. An unobserved
        # half-turn axis cannot initialize it, but this adds no task yaw target.
        skill["program"]["demo"]["T_reference_carrier"][3] = pose((.04, 0., 0.), (0., np.pi, 0.))
        grounded = ground_program(skill["program"], skill["scene"])
        self.assertIn("antiparallel_axis", grounded["function_checks"][0]["reference_generation"]["reason"])
        request = compile_inputs(skill)
        self.assertLess(self.evaluate_body(request, skill, pose((.04, 0., 0.)),
                                            request.supports[0].progress), 1e-12)
        # A supplied invalid branch remains a structural input error.
        face = skill["scene"]["functional_bindings"]["function"]["on_geometry"]
        face["antiparallel_axis"] = [0., 0., 1.]
        with self.assertRaisesRegex(ValueError, "perpendicular"):
            compile_inputs(skill)

    def test_actual_entry_support_keeps_the_declared_positive_face_gap_band(self):
        skill = on_input()
        skill["program"]["demo"]["functional_events"][0]["frame_index"] = 100
        scene = skill["scene"]
        scene["initial_pose"] = pose((0., 0., 5e-7))
        scene["eef_pose"] = scene["initial_pose"] @ scene["holding"]["T_carrier_eef"]
        request = compile_inputs(skill)
        self.assertEqual(request.supports[0].gap_tolerance, 1e-6)
        self.assertLess(self.evaluate_body(request, skill, scene["initial_pose"], 0.), 1e-12)
        scene["initial_pose"][2, 3] = 2e-6
        scene["eef_pose"] = scene["initial_pose"] @ scene["holding"]["T_carrier_eef"]
        with self.assertRaisesRegex(ValueError, "specified support face"):
            compile_inputs(skill)


if __name__ == "__main__":
    unittest.main()
# END supplied task geometry and tolerances remain authoritative downstream.
