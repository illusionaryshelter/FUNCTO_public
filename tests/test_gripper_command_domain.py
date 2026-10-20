"""Pure algebra, XML and mock lifecycle checks; no native physics is run."""

# BEGIN regression checks for bounded independent command-domain calibration.
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


_PATH = Path(__file__).parent/"planning"/"probe_gripper_command_domain.py"
_SPEC = importlib.util.spec_from_file_location("command_domain_probe", _PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def _materials():
    """A nonzero native offset makes q-difference shortcuts observably wrong."""
    left = np.tile([.01, 0., 0.], (4, 1))
    right = np.tile([.025, 0., 0.], (4, 1))
    mapping = np.zeros((4, 3, 2))
    mapping[:, 0, 0] = -1
    second_mapping = np.zeros_like(mapping)
    second_mapping[:, 0, 1] = -1
    return {"first": {"feature_points": left, "feature_joint_translation": mapping,
                      "feature_normal": [1., 0., 0.]},
            "second": {"feature_points": right, "feature_joint_translation": second_mapping,
                       "feature_normal": [-1., 0., 0.]}}


class _Backend:
    """Named deterministic telemetry, deliberately not a dynamics model."""
    def __init__(self):
        self.timestep, self.sim_time = .002, 0.
        self.arm_joint_names, self.finger_joint_names = ("arm",), ("left", "right")
        self.joint_names = self.arm_joint_names+self.finger_joint_names
        self.actuator_names = ("arm_motor", "left_servo", "right_servo")
        self.joint_ranges = np.array([[-1., 1.], [0., .04], [-.04, 0.]])
        self.velocity_limits = dict.fromkeys(self.joint_names, .2)
        self.trial_id, self.feedback_source = "mock", "named mock feedback"
        self.q, self.dq = np.array([0., .02, -.02]), np.zeros(3)
        self.control = self.q.copy()
        self.hold_confirmed = True
        self.hold_steps = 2
        self.finger_speed = 0.
        self.native_failure = False
        # BEGIN offline diagnostics use a distinct, deliberately permuted order.
        self._qvel_ids = self._actuator_ids = np.array([2, 0, 1])
        self.native_data = SimpleNamespace(qacc=np.array([20., 30., 10.]),
            actuator_force=np.array([50., 60., 40.]))
        # END these values never appear in the mock runtime feedback.

    @property
    def finger_control(self):
        return self.control[1:].copy()

    @property
    def feedback_id(self):
        return f"{self.trial_id}@native_time={self.sim_time:.17g}"

    def reference_tool_pose(self, q):
        return np.eye(4)

    def proprioception(self):
        return {"joint_names": self.joint_names, "arm_joint_names": self.arm_joint_names,
                "finger_joint_names": self.finger_joint_names, "actuator_names": self.actuator_names,
                "trial_id": self.trial_id, "source": self.feedback_source,
                "feedback_id": self.feedback_id, "sim_time": self.sim_time,
                "timestamp_s": self.sim_time, "q": self.q.copy(), "dq": self.dq.copy(),
                # BEGIN model TCP is distinct from native dynamics diagnostics.
                "tool_pose": np.eye(4),
                "tool_pose_source": "forward_kinematics_from_measured_arm_q",
                "tool_pose_is_independent_observation": False,
                # END runtime feedback has neither qacc nor actuator_force.
                "actuator_ctrl": {"values": self.control.copy()}}

    def apply_control(self, q, dq, ddq, finger_q):
        self.control = np.r_[q, finger_q]
        result = {key: np.zeros(1) for key in probe.TORQUE_FIELDS}
        result.update(actuator_ctrl=self.control.copy(), reference_arm_q=np.asarray(q).copy(),
            reference_arm_dq=np.asarray(dq).copy(), reference_arm_ddq=np.asarray(ddq).copy(),
            arm_torque_saturated=np.zeros(1, bool))
        return result

    def step(self):
        self.sim_time += self.timestep
        self.q = self.control.copy()
        self.dq[1:] = self.finger_speed
        if self.native_failure:
            self.native_failure = False
            raise RuntimeError("native post-integration failure")

    def abort_hold(self, feedback, finger_positions, *, expected_sim_time, stop_velocity, max_steps):
        completed = 0
        try:
            for _ in range(min(self.hold_steps, max_steps)):
                self.apply_control(feedback["q"][:1], np.zeros(1), np.zeros(1), finger_positions)
                self.step()
                completed += 1
        except Exception as error:
            return {"physical_stop_confirmed": False, "physical_steps": completed, "reason": str(error)}
        return {"physical_stop_confirmed": self.hold_confirmed, "physical_steps": completed,
                "samples": [], "controls": []}


def _case_inputs(mode="static"):
    curve = SimpleNamespace(duration=1., sample=lambda time: (np.zeros(1), np.zeros(1), np.zeros(1), 0.))
    case = {"name": "mock_case", "mode": mode, "source_time_s": 0., "order": ["open", "close"]}
    fingers = SimpleNamespace(duration=.012, sample=lambda time: (np.array([.03, -.03]), np.zeros(2)))
    limits = SimpleNamespace(stop_arm_velocity=[.01], stop_finger_velocity=[.001, .001],
        arm_position_error=[.01], arm_velocity_error=[.1], tcp_position_error_m=.005,
        tcp_rotation_error_rad=.02)
    state = {"position_bounds": [[0., .04], [-.04, 0.]], "maximum_speed": [.2, .2]}
    return curve, case, fingers, limits, {"open": state, "close": state}


class CommandDomainTests(unittest.TestCase):
    def test_signed_native_endpoints_map_inward_by_stroke(self):
        rows = probe.candidate_targets({"open": [.04, -.04], "close": [0., 0.]},
                                      [[0., .04], [-.04, 0.]], [.025, .05])
        np.testing.assert_array_equal(rows[0]["targets"]["open"], [.039, -.039])
        np.testing.assert_array_equal(rows[0]["targets"]["close"], [.001, -.001])
        self.assertLess(rows[0]["fraction"], rows[1]["fraction"])
        for fractions in ([0.], [.5], [.05, .025], [.025, .025], [.01, .02, .03]):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                probe.candidate_targets({"open": [.04], "close": [0.]}, [[0., .04]], fractions)
        with self.assertRaises(ValueError):
            probe.candidate_targets({"open": [.039], "close": [0.]}, [[0., .04]], [.025])

    def test_gap_keeps_native_fixed_offset_and_sign(self):
        np.testing.assert_allclose(probe.material_gaps(_materials(), np.array([.02, -.02])), [.055, .055])
        np.testing.assert_allclose(probe.material_gaps(_materials(), np.array([-.02, .02])), [-.025, -.025])

    def test_xml_preserves_complete_robot_mechanics_and_rejects_cross_constraint(self):
        xml = '''<mujoco><option impratio="20" density="1.2" viscosity="2e-5" cone="elliptic"/>
        <worldbody><geom name="floor"/><body name="robot"><inertial mass="2"/>
        <joint name="joint" damping="100" frictionloss="1"/><geom name="pad" solref=".01 .5"/>
        <site name="tcp"/><camera name="camera"/></body><body name="object"><joint name="external"/></body></worldbody>
        <actuator><general name="servo" joint="joint" gainprm="1000"/><motor joint="external"/></actuator>
        <sensor><jointpos joint="joint"/></sensor></mujoco>'''
        result, names = probe.robot_xml(xml, base_link="robot", joint_names=["joint"], source_dir="/tmp")
        root = probe.ET.fromstring(result)
        self.assertEqual(root.find("option").attrib, probe.ET.fromstring(xml).find("option").attrib)
        self.assertEqual(root.find(".//joint").attrib["damping"], "100")
        self.assertEqual(root.find(".//geom").attrib["solref"], ".01 .5")
        self.assertEqual(len(root.find("worldbody")), 1)
        self.assertEqual(len(root.find("actuator")), 1)
        self.assertEqual(names["joint"], {"joint"})
        self.assertIsNone(root.find("sensor"))
        self.assertIsNone(root.find(".//camera"))
        with self.assertRaisesRegex(ValueError, "Cross-boundary"):
            probe.robot_xml(xml.replace("</mujoco>", '<equality><weld body1="robot" body2="object"/></equality></mujoco>'),
                            base_link="robot", joint_names=["joint"], source_dir="/tmp")

    def test_ledger_journal_precedes_step_and_counts_post_step_failure(self):
        with TemporaryDirectory() as tmp:
            journal = Path(tmp)/"ledger.jsonl"
            ledger = probe.StepLedger(limit=3, journal=journal)
            backend = _Backend()
            captured = []
            def failing_step():
                rows = [json.loads(row) for row in journal.read_text().splitlines()]
                self.assertEqual(rows[-1]["event"], "reserved_before_native_step")
                backend.sim_time += backend.timestep
                raise RuntimeError("after integration")
            with self.assertRaisesRegex(RuntimeError, "after integration"):
                ledger.step(backend, failing_step, lambda *args: captured.append(args))
            self.assertEqual((ledger.charged, ledger.confirmed), (1, 1))
            self.assertTrue(captured[0][0])
            ledger.step(backend, backend.step, lambda *args: None)
            self.assertEqual((ledger.charged, ledger.confirmed), (2, 2))
            with self.assertRaises(RuntimeError):
                ledger.reserve_case()

    def test_failed_precharge_sink_prevents_native_integration(self):
        ledger, backend = probe.StepLedger(), _Backend()
        with patch.object(ledger, "persist", side_effect=OSError("sink")):
            with self.assertRaises(OSError):
                ledger.step(backend, backend.step, lambda *args: None)
        self.assertEqual(backend.sim_time, 0.)
        self.assertEqual((ledger.charged, ledger.confirmed), (1, 0))

    def test_no_advance_still_charged_and_clock_jump_is_unknown(self):
        ledger, backend = probe.StepLedger(), _Backend()
        with self.assertRaises(RuntimeError):
            ledger.step(backend, lambda: None, lambda *args: None)
        self.assertEqual((ledger.charged, ledger.confirmed), (1, 0))
        with self.assertRaises(RuntimeError):
            ledger.step(backend, lambda: setattr(backend, "sim_time", .123), lambda *args: None)
        self.assertFalse(ledger.clock_exact)

    def test_source_commands_use_validated_finger_events_without_device_policy(self):
        original = [{"skill_id": "source", "event_id": kind, "occurrence": 0,
                     "kind": kind, "time": float(index), "progress": float(index)}
                    for index, kind in enumerate(("open", "close"))]
        fingers = SimpleNamespace(joint_names=("left", "right"), events=tuple(
            {**row, "finger_positions": [.04, -.04] if row["kind"] == "open" else [0., 0.]}
            for row in original))
        state = {"position_bounds": [[0., .04], [-.04, 0.]], "maximum_speed": [.2, .2]}
        limits = SimpleNamespace(finger_joint_names=fingers.joint_names,
            events=[{"skill_id": "source", "event_id": row["event_id"], **state} for row in original])
        chosen, protection = probe.source_commands(SimpleNamespace(events=original, finger_trajectory=fingers), limits)
        np.testing.assert_equal(chosen["open"], [.04, -.04])
        self.assertEqual(protection["close"], state)

    def test_finger_reference_retains_measured_nonzero_velocity(self):
        source = SimpleNamespace(joint_names=("left", "right"),
            position_limits=np.array([[0., -.04], [.04, 0.]]),
            velocity_limits=np.array([[-.2, -.2], [.2, .2]]),
            acceleration_limits=np.array([[-10., -10.], [10., 10.]]))
        candidate = {"targets": {"open": np.array([.039, -.039]), "close": np.array([.001, -.001])}}
        q0, dq0 = np.array([.0392, -.0392]), np.array([.00067, -.00067])
        fingers = probe.make_finger_reference(source, {"order": ["open", "close"]}, candidate, q0, dq0, .002, 0.)
        np.testing.assert_array_equal(fingers.sample(0.)[0], q0)
        np.testing.assert_array_equal(fingers.sample(0.)[1], dq0)
        self.assertEqual(fingers.events[1]["time"], 1.1)
        self.assertEqual(fingers.events[1]["event_id"], "1")
        np.testing.assert_allclose(fingers.sample(2.2)[0], candidate["targets"]["close"])

    def _run_mock_case(self, backend, *, mode="static", contacts=None, capture=None):
        ledger = probe.StepLedger(limit=20)
        with TemporaryDirectory() as tmp, patch.object(probe, "NORMAL_STEPS", 6), \
                patch.object(probe, "HOLD_STEPS", 4), \
                patch.object(probe, "native_self_contacts", side_effect=contacts or (lambda *args: ([], False))):
            context = patch.object(probe.np, "savez_compressed", side_effect=capture) if capture else patch.object(probe, "MAX_RSS_MIB", 10**9)
            with context:
                result = probe.bench_case(backend, *_case_inputs(mode), _materials(), ledger, Path(tmp)/"case")
            if (Path(tmp)/"case.npz").exists():
                with np.load(Path(tmp)/"case.npz", allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
            else:
                arrays = {}
            return result, ledger, arrays

    def test_success_records_normal_and_hold_with_same_time_control(self):
        # BEGIN unavailable sensor fields must stay outside online feedback.
        backend = _Backend()
        self.assertNotIn("qacc", backend.proprioception())
        self.assertNotIn("actuator_force", backend.proprioception())
        result, ledger, arrays = self._run_mock_case(backend)
        np.testing.assert_array_equal(arrays["diagnostic_qacc"], np.tile([10., 20., 30.], (9, 1)))
        np.testing.assert_array_equal(arrays["diagnostic_actuator_force"], np.tile([40., 50., 60.], (9, 1)))
        self.assertNotIn("qacc", arrays)
        self.assertNotIn("actuator_force", arrays)
        np.testing.assert_array_equal(result["maximum_absolute_diagnostic_actuator_force"], [40., 50., 60.])
        # END offline diagnostic records preserve their named order separately.
        self.assertTrue(result["bench_feasible"])
        self.assertEqual((result["normal_steps"], result["hold_confirmed_physics_steps"]), (6, 2))
        self.assertEqual((ledger.charged, ledger.confirmed), (8, 8))
        np.testing.assert_array_equal(arrays["q"][:, 1:]-arrays["actual_actuator_ctrl"][:, 1:], 0.)
        self.assertEqual(list(arrays["actual_phase"][-2:]), ["hold", "hold"])

    def test_post_step_failure_keeps_primary_and_exact_phase_ledger(self):
        backend = _Backend()
        backend.native_failure = True
        result, ledger, arrays = self._run_mock_case(backend)
        self.assertFalse(result["bench_feasible"])
        self.assertIn("native post-integration failure", result["primary_failure"]["reason"])
        self.assertEqual(result["normal_completed_loop_iterations"], 0)
        self.assertEqual(result["normal_steps"], 1)
        self.assertEqual(result["hold_confirmed_physics_steps"], 2)
        self.assertEqual(ledger.confirmed, len(arrays["q"])-1)

    def test_dynamic_endpoint_does_not_misuse_stop_threshold(self):
        backend = _Backend()
        backend.finger_speed = .005
        dynamic, _, _ = self._run_mock_case(backend, mode="dynamic")
        self.assertTrue(all(row["passed"] for row in dynamic["endpoints"]))
        self.assertTrue(all(not row["settled"] for row in dynamic["endpoints"]))
        static, _, _ = self._run_mock_case(_BackendWithFingerSpeed(.005))
        self.assertEqual(static["primary_failure"]["phase"], "endpoint")

    def test_unconfirmed_stop_or_active_self_contact_cannot_pass(self):
        backend = _Backend()
        backend.hold_confirmed = False
        result, _, _ = self._run_mock_case(backend)
        self.assertFalse(result["bench_feasible"])
        contact = {"geom1": "pad", "geom2": "arm", "distance_m": -.001,
                   "efc_address": 4, "contact_frame_force": np.array([1., 0., 0., 0., 0., 0.])}
        result, _, _ = self._run_mock_case(_Backend(), contacts=lambda *args: ([contact], False))
        self.assertFalse(result["bench_feasible"])
        self.assertGreater(result["self_contact"]["active_constraint_rows"], 0)

    def test_npz_sink_failure_retains_summary_and_disqualifies_evidence(self):
        result, ledger, _ = self._run_mock_case(_Backend(), capture=OSError("npz sink failed"))
        self.assertFalse(result["bench_feasible"])
        self.assertEqual(ledger.confirmed, 8)
        self.assertIn("npz sink failed", result["persistence_errors"][0])


class _BackendWithFingerSpeed(_Backend):
    def __init__(self, speed):
        super().__init__()
        self.finger_speed = speed


if __name__ == "__main__":
    unittest.main()
# END all tests use synthetic XML or deterministic mock telemetry only.
