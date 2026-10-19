"""Small CPU counterexamples for a declared physical finger-geometry prefix."""

# BEGIN nominal geometry never changes original commands or measured evidence.
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.gripper_trajectory import (
    FingerTrajectory, NominalFingerGeometry, bind_nominal_finger_geometry,
)
from manipulation.planning_problem import event_occurrence_key
from manipulation.planning_scene import PrismaticGripper
from manipulation.robot_validation import RobotStateValidator
from test_gripper_trajectory import _arguments


def command_and_clock(*, held=False):
    arguments = _arguments()
    if held:
        # A commanded close can differ from the actual, contact-stopped q.
        # It retains the entrance's small nonzero measured dq.
        arguments["commands"] = arguments["commands"][1:]
    reference = FingerTrajectory.build(**arguments)
    clock = {event_occurrence_key(row): {"event": deepcopy(row), "time": row["time"],
        "activation_time_s": row["applied_time_s"]} for row in reference.events}
    return reference, clock


def activation(identity=("s", "close", 0), begin=.6):
    return {"kind": "grasp_contact_activation", "condition_id": repr(identity),
        "interval": [begin, 1.], "activation_event": dict(
            zip(("skill_id", "event_id", "occurrence"), identity))}


def holding(reference):
    return {"kind": "observed_attachment", "condition_id": "measured-holding",
        "source": {"verified": True, "robot_feedback_id": reference.feedback_id}}


def bind(reference, conditions, clock, interval=(0., 1.)):
    return bind_nominal_finger_geometry(reference, conditions, clock, progress_interval=interval)


def hand(model, *, command_factory=None, nominal_factory=None):
    result = PrismaticGripper(controller="binary" if command_factory is None else "timed_reference",
        joint_names=model.joint_names, current_positions=model.position[0],
        joint_limits=[[0., .04], [-.04, 0.]], joint_bounds=None,
        coupling_matrix=np.empty((0, 2)), coupling_offset=[], sphere_indices=(0, 1),
        local_spheres=[[0., 0., 0., .001]]*2,
        sphere_joint_translation=[[[1., 0.], [0., 0.], [0., 0.]], [[0., 1.], [0., 0.], [0., 0.]]],
        self_collision_pairs=(), commands=(), source="explicit CPU hand", control_timestep_s=.002,
        finger_trajectory_factory=command_factory, nominal_finger_trajectory_factory=nominal_factory)
    if nominal_factory is None:
        result.nominal_finger_trajectory = model
    return result


def test_free_geometry_keeps_original_polynomial_and_bound_activation_tick():
    reference, clock = command_and_clock()
    original = reference.describe()
    model = bind(reference, [activation()], clock)
    assert model.valid_until_s == .302 != .3005  # Original applied tick, not intention time.
    assert model.valid_until_s < reference.executable_until == .8
    assert model.duration == reference.duration == 1.
    for field in ("times", "position", "velocity"):
        np.testing.assert_array_equal(getattr(model, field), getattr(reference, field))
    assert reference.describe() == original
    assert np.all(model.valid_mask([0., .302]))
    assert not model.valid_mask(np.nextafter(.302, np.inf))


def test_earliest_contact_uses_named_occurrences_independent_of_condition_order():
    reference, clock = command_and_clock()
    earlier = ("other_skill", "jaw_action", 3)
    clock[earlier] = {"time": .21, "activation_time_s": .212}
    conditions = [activation(), activation(earlier, .4)]
    model = bind(reference, conditions, clock)
    assert model.valid_until_s == .212
    assert repr(earlier) in model.source
    assert bind(reference, list(reversed(conditions)), clock).describe() == model.describe()


def test_held_geometry_freezes_measured_q_without_discarding_actual_command_dq():
    reference, clock = command_and_clock(held=True)
    original = reference.describe()
    model = bind(reference, [holding(reference)], clock)
    assert model.valid_until_s == .9  # The bound open still belongs to the command suffix.
    q, dq = model.sample([0., .4, .9])
    np.testing.assert_array_equal(q, np.tile(reference.position[0], (3, 1)))
    assert not dq.any()
    assert np.any(reference.sample(0.)[1] != 0.)
    assert not np.array_equal(reference.sample(.8)[0], model.sample(.8)[0])
    assert reference.describe() == original
    assert "not measured dq" in model.source


def test_original_release_condition_without_occurrence_uses_actual_bound_effect():
    reference, clock = command_and_clock(held=True)
    release = {"kind": "predicted_release", "skill_id": "s", "event_id": "release-effect",
               "condition_id": "original-release", "progress": .75}
    clock[("s", "release-effect", 2)] = {"time": .7, "consumption_time_s": .7}
    model = bind(reference, [holding(reference), release], clock)
    assert model.valid_until_s == .7
    assert "('s', 'release-effect', 2)" in model.source


def test_future_block_contact_or_release_does_not_erase_the_current_geometry():
    arguments = _arguments()
    arguments["commands"] = []  # This current block has no future block's commands.
    reference = FingerTrajectory.build(**arguments)
    clock = {}
    release = {"kind": "predicted_release", "skill_id": "s", "event_id": "release-effect",
               "condition_id": "future-release", "progress": .9}
    # These retained source conditions belong to a later finite motion block.
    free = bind(reference, [activation(begin=.8)], {}, interval=(0., .5))
    held = bind(reference, [holding(reference), release], clock, interval=(0., .5))
    assert free.valid_until_s == held.valid_until_s == reference.duration
    # A contact already reached before this block has no measured Holding
    # geometry merely because its command occurrence disappeared from the clock.
    consumed = bind(reference, [activation(begin=.2)], {}, interval=(.3, .5))
    assert consumed.valid_until_s == 0.


def test_missing_current_occurrence_cannot_silently_remove_finger_collision_checks():
    reference, _ = command_and_clock(held=True)
    with pytest.raises(ValueError, match="original bound event occurrence"):
        bind(reference, [activation()], {})
    release = {"kind": "predicted_release", "skill_id": "s", "event_id": "release-effect",
               "condition_id": "current-release", "progress": .8}
    with pytest.raises(ValueError, match="original bound event occurrence"):
        bind(reference, [holding(reference), release], {})


def test_superseded_open_does_not_invent_a_held_geometry_transition():
    arguments = _arguments()
    arguments["commands"] = arguments["commands"][1:]
    arguments["commands"][-1]["time"] = .9001
    arguments["commands"].append({**arguments["commands"][-1], "event_id": "hold_same_tick",
                                   "kind": "hold", "time": .901})
    reference = FingerTrajectory.build(**arguments)
    assert reference.events[-2]["superseded"] is True
    model = bind(reference, [holding(reference)], {})
    assert model.valid_until_s == reference.duration


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixed_batch_cpu_and_torch_share_exact_validity_boundary(dtype):
    reference, _ = command_and_clock()
    model = NominalFingerGeometry.from_reference(reference, valid_until_s=.1, source="typed boundary fixture")
    endpoint = dtype(model.valid_until_s)
    times = np.array([0., endpoint, np.nextafter(endpoint, dtype(np.inf)), .5], dtype=dtype)
    if dtype is np.float32:
        assert float(endpoint) > model.valid_until_s
    expected = [True, True, False, False]
    assert model.valid_mask(times).tolist() == expected
    actual_hand = hand(model)
    query = torch.from_numpy(times[None])
    state = actual_hand.state(torch.zeros_like(query), None, None, trajectory_time=query)
    assert (~state.uncertain).tolist() == [expected]
    assert not state.half[0, :2].any()
    assert torch.all(state.half[0, 2:] > 0.)
    assert not state.controller_residual.any()  # A physical-time pause may retain progress zero.
    assert actual_hand.state(torch.zeros_like(query), None, None).uncertain.all()
    actual = torch.tensor([[reference.position[0]]*4], dtype=query.dtype)
    measured = actual_hand.state(torch.zeros_like(query), actual, model.joint_names,
                                 trajectory_time=query, measured=True)
    assert not measured.uncertain.any()
    assert torch.equal(measured.coordinates, actual)


@pytest.mark.parametrize("invalid", [np.nan, -.01, 1.01])
def test_invalid_physical_times_are_not_silently_changed_to_unknown_geometry(invalid):
    reference, _ = command_and_clock()
    model = NominalFingerGeometry.from_reference(reference, valid_until_s=.1, source="finite fixture")
    query = torch.tensor([[invalid]], dtype=torch.float64)
    with pytest.raises(ValueError, match="trajectory duration"):
        hand(model).state(torch.zeros_like(query), None, None, trajectory_time=query)
    with pytest.raises(ValueError, match="trajectory duration"):
        model.valid_mask(invalid)




def test_cloned_hand_binds_geometry_to_its_own_new_command_reference():
    reference, _ = command_and_clock()
    seed = NominalFingerGeometry.from_reference(reference, valid_until_s=.3, source="clone fixture")
    seen = []
    def commands(duration, mapping):
        arguments = _arguments()
        arguments.update(duration=duration, executable_until=duration)
        return FingerTrajectory.build(**arguments)
    def geometry(duration, mapping, command_reference):
        seen.append(command_reference)
        return NominalFingerGeometry.from_reference(command_reference, valid_until_s=duration,
                                                   source="explicit cloned binding")
    original = hand(seed, command_factory=commands, nominal_factory=geometry)
    original.bind_clock(1., None)
    copied = deepcopy(original)
    copied.bind_clock(2., None)
    assert seen[-1] is copied.finger_trajectory
    assert copied.nominal_finger_trajectory.duration == copied.finger_trajectory.duration == 2.
    assert original.nominal_finger_trajectory.duration == original.finger_trajectory.duration == 1.


def test_production_online_archive_preserves_polynomial_validity_and_source():
    from manipulation.online_geometry import OnlinePlanningScene
    from test_prismatic_self_collision import hand as native_hand, scene
    current = native_hand(nominal=True)
    current.nominal_finger_trajectory = replace(current.nominal_finger_trajectory, valid_until_s=.1)
    world = scene(current)
    binding = world.to_execution_binding()
    record = binding["gripper"]["nominal_model"]
    assert record["valid_until_s"] == .1
    restored = OnlinePlanningScene.from_binding(binding)
    assert restored.gripper.nominal_finger_trajectory.describe() == record
    times = torch.tensor([[0., .1, .2]], dtype=torch.float64)
    assert restored.gripper.state(torch.zeros_like(times), None, None,
                                  trajectory_time=times).uncertain.tolist() == [[False, False, True]]
    record.pop("valid_until_s")
    with pytest.raises(ValueError, match="polynomial, validity and source"):
        OnlinePlanningScene.from_binding(binding)
# END unknown contact suffixes retain the existing conditional semantics.
