"""CPU reference/clock contracts; no robot dynamics or contact are simulated."""

# BEGIN named C1 references preserve measured origins and original event clocks.
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from manipulation.gripper_trajectory import FingerTrajectory


def _arguments():
    return {
        "joint_names": ("finger_a", "finger_b"),
        "initial_position": [.03, -.025], "initial_velocity": [.0006, -.0003],
        "commands": [
            {"skill_id": "s", "event_id": "open", "occurrence": 0, "kind": "open", "time": 0., "progress": 0.,
             "finger_positions": [.035, -.034], "target_source": "explicit CPU test command"},
            {"skill_id": "s", "event_id": "close", "occurrence": 0, "kind": "close", "time": .3005, "progress": .6,
             "finger_positions": [.005, -.008], "target_source": "explicit CPU test command"},
            {"skill_id": "s", "event_id": "later", "occurrence": 0, "kind": "open", "time": .9, "progress": 1.,
             "finger_positions": [.01, -.012], "target_source": "conditional future command"}],
        "duration": 1., "executable_until": .8, "timestep": .002,
        "position_limits": [[0., -.04], [.04, 0.]],
        "velocity_limits": [[-.2, -.2], [.2, .2]],
        "acceleration_limits": [[-15., -15.], [15., 15.]],
        "source": "Declared CPU nominal reference; no physical tracking claim",
        "feedback_id": "same measured entrance", "timestamp_s": 42.,
    }


class FingerTrajectoryTests(unittest.TestCase):
    def test_measured_q_and_nonzero_dq_are_preserved_before_the_first_open_target(self):
        arguments = _arguments()
        reference = FingerTrajectory.build(**arguments)
        q, dq = reference.sample(0.)
        np.testing.assert_array_equal(q, arguments["initial_position"])
        np.testing.assert_array_equal(dq, arguments["initial_velocity"])
        self.assertFalse(np.array_equal(q, arguments["commands"][0]["finger_positions"]))
        self.assertFalse(np.isclose(q.sum(), 0.))
        self.assertIn("not measured", reference.describe()["scope"])
        self.assertIn("not a calibrated", reference.describe()["limits_scope"])

    def test_non_grid_event_is_not_moved_or_issued_early(self):
        reference = FingerTrajectory.build(**_arguments())
        event = reference.events[1]
        self.assertEqual(event["time"], .3005)
        self.assertEqual(event["activation_step"], 151)
        self.assertEqual(event["applied_time_s"], .302)
        self.assertEqual(event["application_delay_s"], .302 - .3005)
        for time_s in (.3005, .301, .302):
            np.testing.assert_array_equal(reference.sample(time_s)[0], [.035, -.034])
        after, _ = reference.sample(.304)
        self.assertLess(after[0], .035)
        self.assertGreater(after[1], -.034)

    def test_native_reference_bounds_and_c1_are_checked_without_a_c2_gate(self):
        reference = FingerTrajectory.build(**_arguments())
        self.assertTrue(reference.describe()["bounds_report"]["bounds_satisfied"])
        coefficients = reference._curve.c
        width = np.diff(reference.times)[:, None]
        left_dq = (3 * coefficients[0] * width + 2 * coefficients[1]) * width + coefficients[2]
        np.testing.assert_allclose(left_dq[:-1], reference.velocity[1:-1], atol=1e-12, rtol=0.)
        ddq_jumps = 2 * coefficients[1, 1:] - (6 * coefficients[0] * width + 2 * coefficients[1])[:-1]
        self.assertGreater(float(np.max(np.abs(ddq_jumps))), 0.)

    # BEGIN shifted ramps use the same absolute native knots as the final curve.
    def test_nonzero_start_rejects_a_locally_valid_ramp_that_exceeds_absolute_velocity(self):
        from scipy.interpolate import CubicHermiteSpline

        arguments = _arguments()
        arguments.update(initial_position=[.039, -.039], duration=14., executable_until=14.)
        arguments["commands"] = arguments["commands"][:2]
        arguments["commands"][0]["finger_positions"] = [.038, -.038]
        arguments["commands"][1].update(time=13.378, finger_positions=[.002, -.002])
        start = arguments["commands"][1]["time"]
        old_finish = 6824*arguments["timestep"]
        endpoints = np.array([[.038, -.038], [.002, -.002]])
        rates = np.zeros((2, 2))
        relative = CubicHermiteSpline([0., .27], endpoints, rates)
        absolute = CubicHermiteSpline([start, old_finish], endpoints, rates)
        # This is the counterexample: the same intended duration has a
        # smaller represented width on the saved absolute time axis.
        self.assertLess(old_finish-start, .27)
        self.assertLessEqual(float(np.abs(relative(.135, 1)).max()), .2)
        self.assertGreater(float(np.abs(absolute((start+old_finish)/2, 1)).max()), .2)

        reference = FingerTrajectory.build(**arguments)
        event, segment = reference.events[1], reference.segments[1]
        self.assertEqual(event["time"], start)
        self.assertEqual(event["activation_step"], 6689)
        self.assertEqual(event["applied_time_s"], 6689*arguments["timestep"])
        self.assertEqual(segment["ramp_end_time_s"], 6825*arguments["timestep"])
        self.assertEqual(segment["end_time_s"], arguments["duration"])
        np.testing.assert_array_equal(reference.position[0], arguments["initial_position"])
        np.testing.assert_array_equal(reference.velocity[0], arguments["initial_velocity"])
        self.assertTrue(reference.describe()["bounds_report"]["bounds_satisfied"])
        for row in reference.describe()["bounds_report"]["joint_extrema"].values():
            self.assertGreaterEqual(row["velocity"]["minimum"], -.2)
            self.assertLessEqual(row["velocity"]["maximum"], .2)
        selected = CubicHermiteSpline([start, segment["ramp_end_time_s"]], endpoints, rates)
        column = int(np.flatnonzero(reference.times == start)[0])
        np.testing.assert_array_equal(reference._curve.c[:, column], selected.c[:, 0])
        loaded = FingerTrajectory.from_mapping(reference.describe())
        np.testing.assert_array_equal(loaded.times, reference.times)
        np.testing.assert_array_equal(loaded._curve.c, reference._curve.c)

    def test_absolute_ramp_rejects_a_short_deadline_and_preserves_a_feasible_non_grid_deadline(self):
        from manipulation.gripper_trajectory import _segment

        q0, goal, dq0 = np.array([.038, -.038]), np.array([.002, -.002]), np.zeros(2)
        arguments = _arguments()
        limits = {label: np.asarray(arguments[label+"_limits"])
                  for label in ("position", "velocity", "acceleration")}
        options = (q0, dq0, goal)
        for end in (13.648, np.nextafter(13.648, -np.inf)):
            with self.subTest(end=end), self.assertRaisesRegex(ValueError, "No C1 finger command"):
                _segment(*options, 13.378, end, .002, arguments["joint_names"], limits)
        # The next grid tick is 13.65; a feasible earlier original deadline
        # must retain its exact value instead of moving to that later tick.
        deadline = 13.649
        finish = _segment(*options, 13.378, deadline, .002, arguments["joint_names"], limits)
        self.assertEqual(finish, deadline)
        self.assertLess(finish, 6825*.002)
    # END no epsilon, target change or extra event time makes a reference pass.

    # BEGIN a dependency barrier preserves the complete conditional curve.
    def test_barrier_limits_authorization_and_preserves_later_original_commands(self):
        reference = FingerTrajectory.build(**_arguments())
        self.assertFalse(reference.events[-1]["in_executable_prefix"])
        before = reference.sample(reference.executable_until)[0]
        for time_s in (.8, .9):
            q, dq = reference.sample(time_s)
            np.testing.assert_array_equal(q, before)
            np.testing.assert_array_equal(dq, [0., 0.])
        self.assertFalse(np.array_equal(reference.sample(.95)[0], before))
        np.testing.assert_allclose(reference.sample(1.)[0], [.01, -.012], atol=1e-15, rtol=0.)
        self.assertIn("later", [row["event_id"] for row in reference.segments])
        complete = _arguments(); complete["executable_until"] = complete["duration"]
        whole = FingerTrajectory.build(**complete)
        np.testing.assert_array_equal(reference._curve.c, whole._curve.c)
        np.testing.assert_array_equal(reference.times, whole.times)
    # END granting the suffix later cannot regenerate or replace its polynomial.

    def test_unreachable_deadline_and_late_native_activation_are_rejected(self):
        arguments = _arguments()
        arguments["commands"][1]["time"] = .79
        with self.assertRaisesRegex(ValueError, "before 0.9"):
            FingerTrajectory.build(**arguments)
        arguments["commands"][1]["time"] = .7995
        with self.assertRaisesRegex(ValueError, "activate before.*barrier"):
            FingerTrajectory.build(**arguments)

    def test_initial_velocity_is_not_zeroed_to_make_a_bound_pass(self):
        arguments = _arguments()
        arguments["initial_velocity"][0] = .21
        with self.assertRaisesRegex(ValueError, "initial finger q/dq"):
            FingerTrajectory.build(**arguments)

    # BEGIN device quantization preserves required command semantics.
    def test_repeated_native_tick_rejects_distinct_required_goals(self):
        arguments = _arguments()
        arguments["commands"][0]["time"] = .001
        arguments["commands"].insert(1, {**deepcopy(arguments["commands"][0]),
            "event_id": "superseding-open", "time": .0015, "finger_positions": [.036, -.035]})
        with self.assertRaisesRegex(ValueError, "required command sequence"):
            FingerTrajectory.build(**arguments)
        arguments["commands"][1]["finger_positions"] = arguments["commands"][0]["finger_positions"]
        reference = FingerTrajectory.build(**arguments)
        self.assertTrue(reference.events[0]["superseded"])
        self.assertFalse(reference.events[1]["superseded"])
        np.testing.assert_array_equal(reference.sample(0.)[1], arguments["initial_velocity"])

    def test_continuation_preserves_original_integer_tick_despite_float_subtraction(self):
        from manipulation.gripper_trajectory import _commands
        # In binary64 this relative intent is slightly later than local .2;
        # re-running ceil would incorrectly defer the command to local .3.
        original = .6000000000000001
        row = {"skill_id": "s", "event_id": "open", "occurrence": 0, "kind": "open", "progress": 1.,
            "time": original-.4, "finger_positions": [.02],
            "activation_time_s": .2, "bound_activation_step": 2,
            "request_activation_step": 6, "request_activation_time_s": original,
            "request_intent_time_s": original, "request_time_origin_step": 4,
            "device_timestep_s": .1}
        result = _commands([row], ("finger",), 1., 1., .1, np.array([[0.], [.04]]))
        self.assertEqual(result[0]["activation_step"], 2)
        self.assertEqual(result[0]["applied_time_s"], .2)
    # END equivalent repeated targets remain archived with their source identities.

    # BEGIN explicit retries retain independent ramps and survive exact reload.
    def test_repeated_source_command_uses_a_distinct_occurrence(self):
        arguments = _arguments()
        arguments["commands"][-1].update(event_id="open", occurrence=1)
        reference = FingerTrajectory.build(**arguments)
        repeated = [row for row in reference.events if row["event_id"] == "open"]
        self.assertEqual([row["occurrence"] for row in repeated], [0, 1])
        self.assertNotEqual(repeated[0]["applied_time_s"], repeated[1]["applied_time_s"])
        mapping = reference.describe()
        self.assertEqual(FingerTrajectory.from_mapping(mapping).describe(), mapping)
        changed = deepcopy(mapping)
        changed["segments"][-1]["occurrence"] = 0
        with self.assertRaisesRegex(ValueError, "introduce or duplicate"):
            FingerTrajectory.from_mapping(changed)

    def test_missing_duplicate_or_invalid_occurrence_cannot_identify_a_retry(self):
        for invalid in (None, -1, True, 0.5):
            arguments = _arguments()
            arguments["commands"][-1]["occurrence"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "occurrence"):
                FingerTrajectory.build(**arguments)
        arguments = _arguments()
        arguments["commands"][-1]["event_id"] = "open"
        with self.assertRaisesRegex(ValueError, "unique.*occurrences"):
            FingerTrajectory.build(**arguments)
        del arguments["commands"][-1]["occurrence"]
        with self.assertRaisesRegex(ValueError, "original timed event identities"):
            FingerTrajectory.build(**arguments)
    # END reloading cannot turn an applied source occurrence into a new command.

    def test_serialized_curve_roundtrip_is_exact_and_tampering_is_rejected(self):
        reference = FingerTrajectory.build(**_arguments())
        mapping = reference.describe()
        loaded = FingerTrajectory.from_mapping(mapping)
        np.testing.assert_array_equal(loaded._curve.c, reference._curve.c)
        for original, rebuilt in zip(reference.sample(np.linspace(0., 1., 401)), loaded.sample(np.linspace(0., 1., 401))):
            np.testing.assert_array_equal(original, rebuilt)
        changed = deepcopy(mapping)
        changed["events"][1]["applied_time_s"] = changed["events"][1]["time"]
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            FingerTrajectory.from_mapping(changed)
        changed = deepcopy(mapping)
        changed["position"][1][0] += 1e-5
        with self.assertRaises(ValueError):
            FingerTrajectory.from_mapping(changed)
        mapping["position"][0][0] += .001
        self.assertEqual(reference.position[0, 0], .03)

    def test_torch_uses_the_same_polynomial_and_has_the_same_time_derivative(self):
        reference = FingerTrajectory.build(**_arguments())
        times = torch.tensor([[.011, .151, .323, .551, .81]], dtype=torch.float64, requires_grad=True)
        q, dq = reference.sample_torch(times)
        expected_q, expected_dq = reference.sample(times.detach().numpy())
        np.testing.assert_allclose(q.detach().numpy(), expected_q, atol=1e-15, rtol=1e-13)
        np.testing.assert_allclose(dq.detach().numpy(), expected_dq, atol=1e-15, rtol=1e-13)
        q.sum().backward()
        np.testing.assert_allclose(times.grad.numpy(), expected_dq.sum(-1), atol=1e-15, rtol=1e-13)
        for invalid in (torch.tensor([-1.]), torch.tensor([1.01]), torch.tensor([float("nan")])):
            with self.assertRaises(ValueError):
                reference.sample_torch(invalid)

    def test_execution_binding_rejects_changed_feedback_clock_target_or_hardware(self):
        arguments = _arguments()
        reference = FingerTrajectory.build(**arguments)
        trajectory = SimpleNamespace(duration=1., executable_until=.8)
        feedback = {"joint_names": ["arm", "finger_b", "finger_a"],
                    "finger_joint_names": ["finger_a", "finger_b"],
                    "q": [1., -.025, .03], "dq": [0., -.0003, .0006],
                    "feedback_id": arguments["feedback_id"], "timestamp_s": 42.}
        options = dict(initial_feedback=feedback, timestep=.002,
                       position_limits=arguments["position_limits"], velocity_limits=arguments["velocity_limits"])
        self.assertTrue(reference.validate_execution(trajectory, arguments["commands"], **options)["bounds_satisfied"])
        feedback["dq"][2] = 0.
        with self.assertRaisesRegex(ValueError, "measured initial q/dq"):
            reference.validate_execution(trajectory, arguments["commands"], **options)
        feedback["dq"][2] = .0006
        wrong = deepcopy(arguments["commands"])
        wrong[1]["finger_positions"][0] += .001
        with self.assertRaisesRegex(ValueError, "every exact original"):
            reference.validate_execution(trajectory, wrong, **options)
        options["timestep"] = .001
        with self.assertRaisesRegex(ValueError, "clock and identity"):
            reference.validate_execution(trajectory, arguments["commands"], **options)
        options["timestep"] = .002
        options["velocity_limits"] = [[-.1, -.1], [.1, .1]]
        with self.assertRaisesRegex(ValueError, "hardware speed"):
            reference.validate_execution(trajectory, arguments["commands"], **options)
# END these references make no claim about future measured motion or Holding.


if __name__ == "__main__":
    unittest.main()
