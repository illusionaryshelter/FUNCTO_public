"""CPU robot-only material binding; no renderer, physics or object state."""

# BEGIN named material/cover/FK regressions.
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import sys
import tempfile
import unittest

import mujoco
import numpy as np

from manipulation.contact_binding import (calibrate_box_material_features, build_material_contact_rules,
                                         bind_grasp_contact_rules)

sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))
from robot_fixture import _geom_cover, bind_fixture_contact_materials


_ROBOT = """<mujoco><worldbody><body name="base"><site name="tcp"/>
  <body name="left" pos="-.01 0 0"><joint name="left_q" type="slide" axis="-1 0 0" range="0 .04"/>
    <geom name="left_pad" type="box" size=".004 .008 .008" mass=".1"/></body>
  <body name="right" pos=".01 0 0"><joint name="right_q" type="slide" axis="-1 0 0" range="-.04 0"/>
    <geom name="right_pad" type="box" size=".004 .008 .008" mass=".1"/></body>
</body></worldbody></mujoco>"""
_FACES = [dict(material_id="left_inner", native_geometry="left_pad", face_axis=0, face_sign=1,
               source="explicit test face"),
          dict(material_id="right_inner", native_geometry="right_pad", face_axis=0, face_sign=-1,
               source="explicit opposite test face")]


class ContactBindingTests(unittest.TestCase):
    # BEGIN finite binding must be explicit and cannot partially mutate inputs.
    def test_finite_grasp_binding_is_explicit_atomic_and_independent(self):
        from manipulation.planning_scene import ContactRule
        contact_rule = lambda: ContactRule("contact", "robot", "surface", (0., 1.), (0,), "bound test material")

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
        from manipulation.planning_scene import ContactRule
        contact_rule = lambda: ContactRule("contact", "robot", "surface", (0., 1.), (0,), "bound test material")

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

    def test_native_face_has_full_polygon_and_affine_named_motion(self):
        model = mujoco.MjModel.from_xml_string(_ROBOT)
        result = calibrate_box_material_features(model, tcp_site="tcp",
            joint_positions={"left_q": .023, "right_q": -.017},
            finger_joint_names=("left_q", "right_q"), material_faces=_FACES)
        left, right = result["left_inner"], result["right_inner"]
        self.assertEqual(np.asarray(left["feature_points"]).shape, (4, 3))
        np.testing.assert_allclose(left["feature_normal"], [1., 0., 0.])
        np.testing.assert_allclose(right["feature_normal"], [-1., 0., 0.])
        mapping = np.asarray(left["feature_joint_translation"])
        np.testing.assert_allclose(mapping[:, 0, 0], -1.)
        np.testing.assert_allclose(mapping[:, :, 1], 0.)
        self.assertAlmostEqual(left["native_thickness_m"], .008)
        self.assertLess(left["affine_fk_max_error_m"], 1e-12)

    def test_runtime_permutation_and_exact_cover_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            xml = directory/"robot.xml"
            xml.write_text(_ROBOT)
            model = mujoco.MjModel.from_xml_path(str(xml))
            records, spheres = [], {}
            for face in _FACES:
                geom = model.geom(face["native_geometry"]).id
                balls, record, _ = _geom_cover(model, geom, {"subdivisions": [1, 2, 2]}, directory/f"{geom}.npz")
                records.append(record)
                spheres[record["link"]] = balls
            fixture = {"sphere_count": 8, "robot_mjcf_path": str(xml), "tcp_site": "tcp",
                       "collision_geometries": records}
            robot = {"robot_cfg": {"kinematics": {"collision_link_names": ["left", "right"], "collision_spheres": spheres}}}
            feedback = {"joint_names": ["right_q", "left_q"], "start_q": [-.02, .01],
                        "finger_joint_names": ["left_q", "right_q"]}
            ordering = {"left": [4, 5, 6, 7], "right": [0, 1, 2, 3]}
            binding = bind_fixture_contact_materials(fixture, robot, ordering, feedback, _FACES)
            self.assertEqual(binding["robot_sphere_materials"], ["right_inner"]*4+["left_inner"]*4)
            self.assertEqual(binding["materials"]["left_inner"]["sphere_indices"], [4, 5, 6, 7])
            self.assertEqual(binding["native_geometry_materials"], {"left_pad": "left_inner", "right_pad": "right_inner"})
            declaration = dict(rule_id="local", actor="robot", material_id="left_inner", target_owner="observed",
                interval=(0., 1.), source="explicit observed interaction")
            rule, = build_material_contact_rules(binding, [declaration])
            self.assertEqual(asdict(rule)["sphere_indices"], (4, 5, 6, 7))
            self.assertEqual(set(asdict(rule)), {"rule_id", "actor", "target_owner", "interval", "sphere_indices", "source"})
            with self.assertRaisesRegex(ValueError, "Interaction declarations"):
                build_material_contact_rules(binding, [{**declaration, "gap_bounds": [-.001, .001]}])
            spheres["left"][0]["radius"] += .001
            with self.assertRaisesRegex(ValueError, "sphere cover"):
                bind_fixture_contact_materials(fixture, robot, ordering, feedback, _FACES)

    def test_missing_feedback_and_undeclared_material_axis_fail(self):
        model = mujoco.MjModel.from_xml_string(_ROBOT)
        with self.assertRaisesRegex(ValueError, "every isolated robot"):
            calibrate_box_material_features(model, tcp_site="tcp", joint_positions={"left_q": .01},
                finger_joint_names=("left_q", "right_q"), material_faces=_FACES)
        with self.assertRaisesRegex(ValueError, "native box face"):
            calibrate_box_material_features(model, tcp_site="tcp", joint_positions={"left_q": .01, "right_q": -.01},
                finger_joint_names=("left_q", "right_q"), material_faces=[dict(_FACES[0], face_axis=3)])


if __name__ == "__main__":
    unittest.main()
# END binding does not synthesize contact evidence or alter collision pairs.
