"""CPU-only compilation checks for requirements and their actual consumers.

Fixtures supply independent geometry and use the real compiler/residual code;
no perception model, simulator, robot solver or GPU allocation is involved.
"""

# BEGIN required declarations cannot disappear behind a successful reference.
from copy import deepcopy
import unittest

import numpy as np
import torch

from test_manipulation_planning_problem import (
    compile_inputs, components, grasp_input, manipulation_input, on_input, prismatic_input,
)


class RequirementConsumptionTest(unittest.TestCase):
    def test_inventory_addresses_actual_goals_processes_and_effect_conditions(self):
        skill = prismatic_input()
        skill["coordinate_directions"] = [{"start_frame": 100, "end_frame": 106,
            "coordinate_index": 0, "kind": "net", "sign": 1, "minimum": .04,
            "tolerance": 1e-8, "source": "declared positive opening"}]
        request = compile_inputs(skill)
        rows = {row["requirement_id"]: row for row in request.requirement_consumption}
        self.assertIn(("coordinate_events", 0), rows)
        self.assertIn(("mechanisms", 0, "coordinate_limits"), rows)
        self.assertIn(("mechanisms", 0, "directions", 0), rows)
        self.assertEqual(rows[("coordinate_events", 0)]["consumers"],
                         ("PlanningProblem.residuals", "simulation_posthoc_pose_evaluation"))
        progress = torch.tensor([[0., request.coordinate_events[0].progress]], dtype=torch.float64)
        mechanism = request.mechanisms[0]
        body = mechanism.model.poses(np.array([[.1], [.11]]))
        eef = body @ np.linalg.inv(mechanism.tool_from_carrier)
        p, q = components(eef[None])
        # The named consumer must detect a real violation of this compiled
        # target, rather than merely reporting that a dictionary key exists.
        residuals = request.residuals(p, q, progress,
                                    trajectory_time=torch.tensor([[0., 1.]], dtype=torch.float64))
        self.assertGreater(float(residuals["function_coordinate:0"].max()), .02)
        for index, condition in enumerate(request.conditions):
            self.assertEqual(rows[("conditions", index)]["condition_id"], condition["condition_id"])
        support = compile_inputs(on_input())
        support_rows = {row["requirement_id"] for row in support.requirement_consumption}
        self.assertIn(("supports", 0), support_rows)
        self.assertNotIn(("pose_events", 0), support_rows)

    def test_reference_density_changes_neither_requirement_identities_nor_scopes(self):
        skill = grasp_input()
        original = compile_inputs(skill)
        dense = deepcopy(skill)
        demo = dense["program"]["demo"]
        # Insert an ordinary row between the first two original observations.
        # All event IDs, event seconds and the final source instant are intact.
        demo["frame_indices"] = np.insert(demo["frame_indices"], 1, 12)
        demo["timestamps"] = np.insert(demo["timestamps"], 1, 2.05)
        demo["T_carrier_eef"] = np.insert(demo["T_carrier_eef"], 1,
                                         demo["T_carrier_eef"][0], axis=0)
        compiled = compile_inputs(dense)
        select = lambda problem: [(row["requirement_id"], row["categories"],
                                   row["consumers"], row["progress_scope"])
                                  for row in problem.requirement_consumption]
        self.assertEqual(select(original), select(compiled))
        np.testing.assert_array_equal(original.anchors, compiled.anchors)
        self.assertEqual(compiled.pose_relations, ())

    def test_unknown_requirements_and_constraint_kinds_are_rejected(self):
        for mutate in (
            lambda skill: skill.update(requirements=[{"kind": "force"}]),
            lambda skill: skill["program"].update(requirements=[{"kind": "force"}]),
            lambda skill: skill["scene"].update(pose_relations=[]),
            lambda skill: skill["program"]["demo"]["functional_events"][0].update(constraint_kind="force"),
            lambda skill: skill["program"]["demo"]["functional_events"][0].update(force_tolerance=1.),
        ):
            skill = manipulation_input()
            mutate(skill)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "Unsupported"):
                compile_inputs(skill)
        skill = grasp_input()
        skill["scene"]["observation_metadata"] = {"frame_source": "retained provenance"}
        self.assertTrue(compile_inputs(skill).requirement_consumption)

    def test_other_constraint_families_cannot_silently_supply_unused_tolerances(self):
        for fixture, name, value in (
            (manipulation_input, "coordinate_value", [.1]),
            (manipulation_input, "coordinate_tolerance", .01),
            (prismatic_input, "position_tolerance_m", .01),
            (prismatic_input, "rotation_tolerance_rad", .01),
        ):
            skill = fixture()
            skill["program"]["demo"]["functional_events"][0][name] = value
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "without consumers"):
                compile_inputs(skill)

    def test_condition_inventory_separates_evidence_points_from_attachment_spans(self):
        for fixture in (grasp_input, manipulation_input):
            request = compile_inputs(fixture())
            rows = {row["requirement_id"]: row for row in request.requirement_consumption}
            for index, condition in enumerate(request.conditions):
                row = rows[("conditions", index)]
                self.assertTrue(row["source"])
                # BEGIN declaration is a model condition without a truth obligation.
                if condition["kind"] == "declared_attachment":
                    self.assertNotIn("actual_evidence_obligation", row["categories"])
                    self.assertEqual(row["evidence_progress"], 0.)
                    self.assertEqual(row["scene_rule_spans"], ((0., 1.),))
                    self.assertEqual(row["progress_scope"], (0., 1.))
                # END physical outcome evidence belongs to separate effect conditions.
                elif condition["kind"] == "predicted_release":
                    self.assertEqual(row["evidence_progress"], 1.)
                    self.assertEqual(row["scene_rule_boundary"], 1.)
                    self.assertEqual(row["scene_rule_spans"], ())
                elif condition["kind"] == "candidate_attachment":
                    self.assertEqual(row["scene_rule_spans"], ())
                    self.assertEqual(row["evidence_progress"], 1.)

    def test_execution_event_requires_a_real_consumer_and_preserves_evidence(self):
        skill = grasp_input()
        skill["execution_events"] = [{"event_id": "check", "frame_index": 23,
            "kind": "prerequisite", "source": {"sensor": "fixture"}}]
        with self.assertRaisesRegex(ValueError, "no evidence, stop or activation-time consumer"):
            compile_inputs(skill)
        skill["execution_events"][0]["requires_evidence"] = "tool_ready"
        compiled = compile_inputs(skill)
        event_index = next(i for i, event in enumerate(compiled.events) if event.event_id == "check")
        row = next(row for row in compiled.requirement_consumption
                   if row["requirement_id"] == ("events", event_index))
        self.assertIn("TrajectoryExecutor.execute", row["consumers"])
        self.assertEqual(row["requires_evidence"], "tool_ready")
        self.assertFalse(compiled.events[event_index].stop)
        # The input and report are independently owned, including nested source.
        skill["execution_events"][0]["source"]["sensor"] = "changed after compile"
        row["source"]["sensor"] = "changed report"
        self.assertEqual(compiled.events[event_index].source["sensor"], "fixture")

    def test_recorded_command_metadata_is_preserved_without_becoming_task_time(self):
        skill = grasp_input()
        baseline = compile_inputs(skill)
        demo = skill["program"]["demo"]
        # Match the existing recorder's event shape, including its nested
        # actuator provenance; no external recording is needed by this test.
        for index, event in enumerate(demo["gripper_events"]):
            row = int(np.flatnonzero(demo["frame_indices"] == event["frame_index"])[0])
            event.update(timestamp=float(demo["timestamps"][row]),
                record_kind="segment_initial_intent" if index == 0 else "command_change",
                source={"dataset": "trajectories.h5/actions/vector", "action_part": "right_gripper",
                        "action_slice": [6, 7], "raw_value": [-1. if index == 0 else 1.],
                        "meaning": "controller intention; not measured contact or holding"})
        recorded = deepcopy(demo["gripper_events"])
        compiled = compile_inputs(skill)
        commands = [event for event in compiled.events if event.kind in {"open", "close", "hold"}]
        self.assertEqual([event.source["original_event"] for event in commands], recorded)
        self.assertTrue(all(event.timing == () for event in commands))
        np.testing.assert_array_equal(compiled.anchors, baseline.anchors)
        np.testing.assert_array_equal(compiled.source_time_map["progress"], baseline.source_time_map["progress"])
        demo["gripper_events"][0]["source"]["raw_value"][0] = 0.
        self.assertEqual(commands[0].source["original_event"]["source"]["raw_value"], [-1.])
        demo["gripper_events"][0]["force_requirement"] = 10.
        with self.assertRaisesRegex(ValueError, "Unsupported gripper event fields"):
            compile_inputs(skill)

    def test_time_requirement_cannot_name_a_missing_goal_or_effect(self):
        for target, extra in (("arrival", {"goal_id": ["pose_events", 9]}),
                              ("effect", {"condition_id": "missing_effect"})):
            skill = grasp_input()
            skill["program"]["demo"]["gripper_events"][1]["timing"] = [{
                "check_target": target, "window_s": [0., 1.], "time_basis": "request_physical",
                "origin_time_s": 0., "source": "explicit deadline", **extra}]
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "uncompiled"):
                compile_inputs(skill)

    def test_mechanism_direction_cannot_extend_after_release(self):
        skill = prismatic_input()
        skill["coordinate_directions"] = [{"start_frame": 106, "end_frame": 112,
            "coordinate_index": 0, "kind": "net", "sign": 1, "minimum": .04,
            "tolerance": 1e-8, "source": "declared scoped opening"}]
        with self.assertRaisesRegex(ValueError, "controlled mechanism scope"):
            compile_inputs(skill)

    # BEGIN requirements after release must not disappear into source metadata.
    def test_explicit_execution_requirement_after_release_is_rejected(self):
        skill = manipulation_input()
        skill["execution_events"] = [{"event_id": "later_check", "frame_index": 112,
            "kind": "prerequisite", "source": "explicit current task", "requires_evidence": "next_observation"}]
        with self.assertRaisesRegex(ValueError, "current controlled skill interval"):
            compile_inputs(skill)

    def test_later_source_command_is_retained_but_its_declared_requirement_cannot_be_ignored(self):
        for requirement in ({"stop": True}, {"requires_evidence": "current_release_check"},
                            {"timing": [{"check_target": "activation", "window_s": [0., 1.],
                              "time_basis": "request_physical", "origin_time_s": 0., "source": "current task deadline"}]}):
            skill = manipulation_input()
            later = {"event_id": "source_later_close", "frame_index": 112, "command": "close",
                     "source": {"recording": "unchanged original", "raw_value": [1.]}}
            skill["program"]["demo"]["gripper_events"].append(later)
            compiled = compile_inputs(skill)
            self.assertNotIn("source_later_close", [event.event_id for event in compiled.events])
            self.assertEqual(compiled.phases[0].reference.source["gripper_events"][-1], later)
            later.update(requirement)
            with self.subTest(requirement=requirement), self.assertRaisesRegex(ValueError, "after release"):
                compile_inputs(skill)
    # END source annotations cannot hide an unconsumed current task obligation.

    def test_direction_source_and_consumption_records_do_not_alias_input(self):
        skill = prismatic_input()
        source = {"constraint": "positive opening"}
        skill["coordinate_directions"] = [{"start_frame": 100, "end_frame": 106,
            "coordinate_index": 0, "kind": "net", "sign": 1, "minimum": .04,
            "tolerance": 1e-8, "source": source}]
        compiled = compile_inputs(skill)
        source["constraint"] = "edited input"
        rows = compiled.requirement_consumption
        row = next(row for row in rows if row["requirement_id"] == ("mechanisms", 0, "directions", 0))
        row["source"]["constraint"] = "edited report"
        self.assertEqual(compiled.mechanisms[0].directions[0].source["constraint"], "positive opening")


if __name__ == "__main__":
    unittest.main()
# END CPU semantic coverage makes no robot-execution or performance claim.
