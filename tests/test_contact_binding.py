"""Task features and contact identities remain independent of collision models."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

from manipulation.contact_binding import build_material_contact_rules, bind_grasp_contact_rules
from manipulation.planning_scene import ContactRule


def contact_rule():
    return ContactRule(rule_id="local", actor="robot", robot_body="pad", target_owner="surface",
        interval=(0., 1.), material_id="inner", feature_kind="edge",
        feature_points=[[0., -.01, 0.], [0., .01, .0000004]], feature_normal=[0., 0., -1.],
        patch_origin=[0., 0., 0.], patch_axes=np.eye(3), patch_half_size=[.02, .02],
        gap_bounds=(-.001, .001), normal_tolerance_rad=.01, modeling_allowance_m=0.,
        geometry_error_m=0., source="source task feature")


class ContactBindingTests(unittest.TestCase):
    # BEGIN finite binding must be explicit and cannot partially mutate inputs.
    def test_finite_grasp_binding_is_explicit_atomic_and_independent(self):

        condition = {"kind": "grasp_contact_activation", "condition_id": "grasp:close:0",
            "carrier_id": "surface", "interval": [0., 1.], "arrival_goal_id": ["pose_events", 0],
            "activation_event": {"skill_id": "grasp", "event_id": "close", "occurrence": 0, "kind": "close"},
            "source": "explicit current arrival"}
        problem = SimpleNamespace(conditions=(deepcopy(condition),))
        rule = contact_rule()
        for invalid in (None, {}, {condition["condition_id"]: ["missing"]}):
            with self.assertRaises(ValueError):
                bind_grasp_contact_rules(problem, (rule,), invalid)
            self.assertEqual(problem.conditions, (condition,))
        supplied = {condition["condition_id"]: [rule.rule_id]}
        bind_grasp_contact_rules(problem, (rule,), supplied)
        supplied[condition["condition_id"]].clear()
        self.assertEqual(problem.conditions[0]["contact_rule_ids"], [rule.rule_id])
        self.assertNotIn("contact_rule_ids", condition)

    def test_finite_grasp_binding_cannot_reuse_one_rule_for_two_close_events(self):

        def condition(identity):
            return {"kind": "grasp_contact_activation", "condition_id": identity,
                "carrier_id": "surface", "interval": [0., 1.], "arrival_goal_id": ["pose_events", 0],
                "activation_event": {"skill_id": "grasp", "event_id": identity, "occurrence": 0, "kind": "close"},
                "source": "explicit current arrival"}
        problem = SimpleNamespace(conditions=(condition("first"), condition("second")))
        original = deepcopy(problem.conditions)
        rule = contact_rule()
        with self.assertRaisesRegex(ValueError, "multiple conditions"):
            bind_grasp_contact_rules(problem, (rule,), {key: [rule.rule_id] for key in ("first", "second")})
        self.assertEqual(problem.conditions, original)
    # END a successful association grants no actual execution latch.


    def test_declared_feature_points_and_body_identity_are_preserved(self):
        from dataclasses import asdict
        original = contact_rule()
        record = asdict(original)
        features = ("feature_kind", "feature_points", "feature_normal", "feature_joint_translation", "robot_body")
        material = {key: record.pop(key) for key in features}
        material["source"] = "preserved source feature"
        bound, = build_material_contact_rules({"materials": {"inner": material}}, [record])
        np.testing.assert_array_equal(bound.feature_points, original.feature_points)
        self.assertEqual(bound.robot_body, "pad")
        self.assertNotIn("sphere_indices", asdict(bound))


if __name__ == '__main__':
    unittest.main()
