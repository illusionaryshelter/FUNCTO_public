"""Finite CPU checks for active intent, measured entrance and native commands."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from manipulation.gripper_trajectory import (
    FingerTrajectory, bind_nominal_finger_geometry, capture_active_command, stationary_finger_command,
)
from manipulation.planning_scene import PrismaticGripper

from manipulation.planning_problem import event_occurrence_key


def event(identifier, kind, time, positions):
    return dict(skill_id="next", event_id=identifier, occurrence=0, kind=kind,
                time=time, progress=time / 2., finger_positions=positions)


def preceding_reference():
    """A prior closing ramp plus a later opening event that is not inherited."""
    return FingerTrajectory.build(joint_names=("a", "b"),
        measured_initial_position=[.03, -.025], measured_initial_velocity=[0., 0.],
        incoming_command=stationary_finger_command(joint_names=("a", "b"), position=[.03, -.025],
            timestamp_s=40., source="Declared stationary prior command", feedback_id="prior-feedback"),
        commands=[event("active", "close", 0., [.003, -.006]),
                  event("future", "open", 1., [.035, -.033])],
        duration=2., executable_until=2., timestep=.002,
        position_limits=[[0., -.04], [.04, 0.]],
        velocity_limits=[[-.2, -.2], [.2, .2]],
        acceleration_limits=[[-15., -15.], [15., 15.]],
        source="Explicit prior reference", feedback_id="prior-feedback", timestamp_s=40.)


def arguments(previous_time=.05):
    prior = preceding_reference()
    incoming = capture_active_command(prior, curve_time_s=previous_time,
        last_dispatch_time_s=previous_time-.002,
        boundary_timestamp_s=42., source="Explicit prior command archive and boundary sample")
    return dict(joint_names=prior.joint_names, incoming_command=incoming,
        measured_initial_position=[.021, -.004], measured_initial_velocity=[.0002, -.0003],
        commands=[event("later-close", "close", .08, [.004, -.008]),
                  event("release", "open", 1., [.035, -.033])],
        duration=2., executable_until=1.5, timestep=.002,
        position_limits=prior.position_limits, velocity_limits=prior.velocity_limits,
        acceleration_limits=prior.acceleration_limits, source="Explicit command continuation",
        feedback_id="current-feedback", timestamp_s=42.)


def feedback(reference):
    """Independent observed coordinates and a separate live-command sample."""
    return dict(finger_joint_names=list(reference.joint_names), joint_names=["arm", "b", "a"],
        feedback_id=reference.feedback_id, timestamp_s=reference.timestamp_s, sim_time=reference.timestamp_s,
        q=[1., reference.measured_initial_position[1], reference.measured_initial_position[0]],
        dq=[.1, reference.measured_initial_velocity[1], reference.measured_initial_velocity[0]],
        actuator_ctrl=dict(joint_names=["a", "arm", "b"],
            values=[reference.position[0, 0], 3., reference.position[0, 1]],
            sim_time=reference.timestamp_s, timestamp_s=reference.timestamp_s))


def validate(reference, observed):
    return reference.validate_execution(SimpleNamespace(duration=reference.duration,
        executable_until=reference.executable_until), reference.events,
        initial_feedback=observed, timestep=reference.timestep,
        position_limits=reference.position_limits, velocity_limits=reference.velocity_limits)


def planning_hand(reference, nominal_factory):
    """Bind the same real entrance while the model factory is explicit input."""
    return PrismaticGripper(controller="timed_reference", joint_names=reference.joint_names,
        current_positions=reference.measured_initial_position,
        joint_limits=reference.position_limits.T, joint_bounds=None,
        coupling_matrix=np.empty((0, 2)), coupling_offset=np.empty(0),
        commands=tuple(dict(skill_id=row["skill_id"], event_id=row["event_id"],
            occurrence=row["occurrence"], command=row["kind"], progress=row["progress"])
            for row in reference.events), source="Explicit command-separated planning hand",
        control_timestep_s=reference.timestep,
        finger_trajectory_factory=lambda duration, progress: reference,
        nominal_finger_trajectory_factory=nominal_factory)


class CommandReferenceTests(unittest.TestCase):
    def test_unfinished_active_intent_continues_until_new_event_with_nonzero_velocity(self):
        prior = preceding_reference()
        values = arguments()
        result = FingerTrajectory.build(**values)
        # The new event occurs before completion of the inherited ramp. Its
        # left and right pieces share the original nonzero command velocity.
        query = np.array([0., .02, .05, .08])
        expected_q, expected_dq = prior.sample(query + .05)
        actual_q, actual_dq = result.sample(query)
        np.testing.assert_allclose(actual_q, expected_q, rtol=0., atol=2e-17)
        np.testing.assert_allclose(actual_dq, expected_dq, rtol=0., atol=2e-16)
        self.assertTrue(np.any(actual_dq[-1] != 0.))
        self.assertEqual(result.segments[0]["kind"], "incoming_command")
        self.assertEqual(result.incoming_command["active_event"]["event_id"], "active")
        self.assertEqual([row["event_id"] for row in result.events], ["later-close", "release"])
        self.assertNotIn("future", [row["event_id"] for row in result.events])
        self.assertEqual(result.events[1]["applied_time_s"], 1.)
        self.assertFalse(result.events[1]["superseded"])
        np.testing.assert_array_equal(result.measured_initial_position, [.021, -.004])
        np.testing.assert_array_equal(result.measured_initial_velocity, [.0002, -.0003])

    def test_active_target_holds_after_its_ramp_and_t0_command_can_replace_it(self):
        values = arguments()
        values["commands"][0]["time"] = .5
        result = FingerTrajectory.build(**values)
        finish = values["incoming_command"]["remaining_ramp_s"]
        np.testing.assert_array_equal(result.sample([finish, .4, .5])[0],
                                      np.tile([.003, -.006], (3, 1)))
        np.testing.assert_array_equal(result.sample([finish, .4, .5])[1], np.zeros((3, 2)))
        # A command at zero replaces the active target while retaining q/dq.
        values["commands"][0]["time"] = 0.
        immediate = FingerTrajectory.build(**values)
        self.assertEqual(immediate.segments[0]["kind"], "command")
        np.testing.assert_array_equal(immediate.position[0], result.position[0])
        np.testing.assert_array_equal(immediate.velocity[0], result.velocity[0])

    def test_repeated_handover_and_new_schema_roundtrip_keep_active_source(self):
        first = FingerTrajectory.build(**arguments())
        restored = FingerTrajectory.from_mapping(first.describe())
        self.assertEqual(restored.describe(), first.describe())
        with self.assertRaisesRegex(ValueError, "new measured-entrance schema"):
            FingerTrajectory.from_mapping({**preceding_reference().describe(), "schema": "obsolete-command-schema"})
        # This boundary occurs while the preceding skill still carries an
        # inherited command, so the original active target must survive again.
        active = capture_active_command(restored, curve_time_s=.03,
            last_dispatch_time_s=.028,
            boundary_timestamp_s=43., source="Second explicit command boundary")
        self.assertEqual(active["active_event"]["event_id"], "active")
        self.assertGreater(active["remaining_ramp_s"], 0.)
        np.testing.assert_array_equal(active["target_position"], [.003, -.006])
        changed = first.describe()
        changed["incoming_command"]["target_position"][0] += .001
        with self.assertRaisesRegex(ValueError, "active source event"):
            FingerTrajectory.from_mapping(changed)

    def test_step_endpoint_does_not_activate_the_next_source_event(self):
        prior = preceding_reference()
        incoming = capture_active_command(prior, curve_time_s=1., last_dispatch_time_s=.998,
            boundary_timestamp_s=41., source="Completed control before the next activation tick")
        self.assertEqual(incoming["active_event"]["event_id"], "active")
        np.testing.assert_array_equal(incoming["target_position"], [.003, -.006])
        # Once the next step actually dispatches, that new event is active.
        next_command = capture_active_command(prior, curve_time_s=1.002, last_dispatch_time_s=1.,
            boundary_timestamp_s=41.002, source="Completed control after the activation tick")
        self.assertEqual(next_command["active_event"]["event_id"], "future")
        np.testing.assert_array_equal(next_command["target_position"], [.035, -.033])

    def test_live_command_and_actual_observation_are_checked_independently(self):
        reference = FingerTrajectory.build(**arguments())
        observed = feedback(reference)
        self.assertTrue(validate(reference, observed)["bounds_satisfied"])
        self.assertFalse(np.array_equal(reference.position[0], reference.measured_initial_position))
        changed = deepcopy(observed)
        changed["q"][1] += 1e-5
        with self.assertRaisesRegex(ValueError, "independent observations"):
            validate(reference, changed)
        changed = deepcopy(observed)
        changed["actuator_ctrl"]["values"][0] += 1e-5
        with self.assertRaisesRegex(ValueError, "live incoming actuator"):
            validate(reference, changed)
        changed = deepcopy(observed)
        changed["actuator_ctrl"]["timestamp_s"] += .002
        with self.assertRaisesRegex(ValueError, "live incoming actuator"):
            validate(reference, changed)

    def test_held_geometry_uses_measured_shape_and_original_release_clock(self):
        reference = FingerTrajectory.build(**arguments())
        clock = {event_occurrence_key(row): dict(time=row["time"],
            activation_time_s=row["applied_time_s"]) for row in reference.events}
        original = reference.describe()
        conditions = [dict(kind="declared_attachment", condition_id="declared-relation")]
        model = bind_nominal_finger_geometry(reference, conditions, clock, progress_interval=(0., 1.))
        self.assertEqual(model.valid_until_s, 1.)
        q, dq = model.sample([0., .4, 1.])
        np.testing.assert_array_equal(q, np.tile(reference.measured_initial_position, (3, 1)))
        np.testing.assert_array_equal(dq, np.zeros((3, 2)))
        self.assertEqual(reference.describe(), original)
        free = bind_nominal_finger_geometry(reference, [], clock, progress_interval=(0., 1.))
        np.testing.assert_array_equal(free.sample(0.)[0], reference.measured_initial_position)
        np.testing.assert_array_equal(free.sample(0.)[1], reference.measured_initial_velocity)
        self.assertFalse(np.array_equal(free.sample(0.)[0], reference.sample(0.)[0]))

    def test_planning_clock_binds_command_and_measured_geometry_separately(self):
        reference = FingerTrajectory.build(**arguments())
        conditions = [dict(kind="declared_attachment", condition_id="declared-relation")]
        clock = {event_occurrence_key(row): dict(time=row["time"],
            activation_time_s=row["applied_time_s"]) for row in reference.events}
        hand = planning_hand(reference, lambda duration, progress, command:
            bind_nominal_finger_geometry(command, conditions, clock, progress_interval=(0., 1.)))
        self.assertFalse(np.array_equal(reference.position[0], hand.current_positions))
        hand.bind_clock(reference.duration, SimpleNamespace())
        self.assertIs(hand.finger_trajectory, reference)
        np.testing.assert_array_equal(hand.nominal_finger_trajectory.position[0], hand.current_positions)
        hand.current_positions = hand.current_positions + [.001, 0.]
        with self.assertRaisesRegex(ValueError, "measured entrance"):
            hand.bind_clock(reference.duration, SimpleNamespace())

    def test_missing_motion_model_keeps_actual_entrance_and_unknown_future(self):
        reference = FingerTrajectory.build(**arguments())
        hand = planning_hand(reference, None)
        hand.bind_clock(reference.duration, SimpleNamespace())
        self.assertIsNone(hand.nominal_finger_trajectory)
        times = torch.tensor([[0., .2]], dtype=torch.float64)
        state = hand.state(torch.zeros_like(times), None, None, trajectory_time=times)
        np.testing.assert_array_equal(state.coordinates[0, 0].numpy(), reference.measured_initial_position)
        self.assertEqual(state.uncertain.tolist(), [[False, True]])
        self.assertTrue(bool(torch.all(state.half[0, 1] > 0.)))
        # Uncertain shape is retained as a model state, not turned into a
        # controller-bound violation or an invented position trajectory.
        self.assertEqual(float(state.controller_residual.max()), 0.)


if __name__ == "__main__":
    unittest.main()
