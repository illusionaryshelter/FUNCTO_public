"""CPU execution-boundary tests for independent model and native solid checks."""

# BEGIN actual-state and whole-cubic preflight regressions.
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation.planning import JointTrajectory
from manipulation.robot_geometry_check import NativePrimitive, PairQuery, RobotGeometryChecker
from manipulation.robot_validation import RobotStateValidator
from test_robot_geometry_check import URDF


NAMES = ("left_slide", "right_slide")


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF)
    offset = np.eye(4)
    offset[1, 3] = .4
    return RobotGeometryChecker(path, base_frame="base", geometries=[
        NativePrimitive("first", "left", "box", (.04, .04, .04), np.eye(4), "native first solid"),
        NativePrimitive("second", "right", "box", (.04, .04, .04), offset, "native second solid")])


def validator(model, pairs=None):
    return RobotStateValidator(model, joint_names=NAMES, arm_joint_names=NAMES[:1], finger_joint_names=NAMES[1:],
        pairs=[PairQuery("first", "second", "required", "source native enabled pair")] if pairs is None else pairs,
        named_velocity_limits={name: 1. for name in NAMES}, reference_frame="world",
        T_reference_base=np.eye(4), tool_frame="tool", T_frame_tcp=np.eye(4),
        fk_translation_tolerance_m=1e-8, fk_rotation_tolerance_rad=1e-8, source="independent source fixture")


def feedback(arm=.03, finger=-.08, arm_speed=0.):
    tool = np.eye(4)
    tool[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    tool[:3, 3] = [0, 0, .3]
    return {"joint_names": NAMES, "arm_joint_names": NAMES[:1], "finger_joint_names": NAMES[1:],
            "q": np.array([arm, finger]), "dq": np.array([arm_speed, 0.]),
            "reference_frame": "world", "tool_link": "tool", "tool_transform": np.eye(4),
            "base_pose": np.eye(4), "tool_pose": tool,
            # Deliberately unrelated goals must never replace actual fingers.
            "actuator_ctrl": {"joint_names": NAMES, "values": [0., 0.]}}


def trajectory(positions, velocities, *, duration=1., barrier=None):
    events = () if barrier is None else ({"requires_evidence": "original evidence", "time": barrier},)
    return JointTrajectory(NAMES[:1], np.array([0., duration]), np.asarray(positions, float)[:, None],
                           np.asarray(velocities, float)[:, None], np.array([0., 1.]), np.array([0., 1.]),
                           events, (), {})


LIMITS = SimpleNamespace(joint_names=NAMES, arm_joint_names=NAMES[:1], finger_joint_names=NAMES[1:])


def test_actual_finger_state_is_independent_of_the_command_and_collision_is_mandatory(model):
    gate = validator(model)
    opened = feedback()
    assert gate.check_state(opened, stage="before_step", trajectory_time=0.)["passed"] is True
    closed = feedback(finger=0.)
    closed["actuator_ctrl"]["values"] = [.03, -.08]
    rejected = gate.check_state(closed, stage="after_step", trajectory_time=.1)
    assert rejected["passed"] is False
    assert rejected["distances_m"][0] < 0
    collision = next(row for row in rejected["violations"] if row["kind"] == "independent_robot_collision")
    assert collision["evidence"]["policy"] == "required"
    assert len(collision["evidence"]["witness_first"]) == 3


def test_pair_inventory_and_native_exclusion_are_explicit_and_preserved(model):
    with pytest.raises(ValueError, match="complete unique"):
        validator(model, [])
    with pytest.raises(ValueError, match="required/excluded"):
        validator(model, [PairQuery("first", "second", "diagnostic", "unclassified")])
    gate = validator(model, [PairQuery("first", "second", "excluded", "verified native welded-body rule")])
    result = gate.check_state(feedback(finger=0.), stage="after_step", trajectory_time=.1)
    assert result["passed"] is True
    assert result["distances_m"][0] < 0
    assert gate.describe()["pair_order"][0]["source"] == "verified native welded-body rule"
    assert gate.describe()["continuous_collision_certified"] is False


def test_numerical_contact_boundary_is_not_a_contact_permission(model):
    result = validator(model).check_state(feedback(finger=-.01000025), stage="before_step", trajectory_time=0.)
    assert result["passed"] is False
    assert result["violations"][0]["relation"] == "numerical_boundary"


@pytest.mark.parametrize("field,kind", [("base_pose", "base:translation_m"), ("tool_pose", "tcp:translation_m")])
def test_backend_base_and_tcp_are_checked_against_independent_fk(model, field, kind):
    state = feedback()
    state[field][2, 3] += .001
    result = validator(model).check_state(state, stage="initial_state", trajectory_time=0.)
    assert result["passed"] is False
    assert any(row["kind"] == "independent_model_fk" and row["name"] == kind for row in result["violations"])


def test_wrong_frame_metadata_or_missing_actual_finger_cannot_be_relabelled(model):
    gate = validator(model)
    wrong = feedback()
    wrong["tool_link"] = "left"
    with pytest.raises(ValueError, match="frame identity"):
        gate.check_state(wrong, stage="initial_state", trajectory_time=0.)
    incomplete = feedback()
    incomplete["joint_names"] = NAMES[:1]
    incomplete["q"] = incomplete["q"][:1]
    with pytest.raises(ValueError, match="model identity"):
        gate.check_state(incomplete, stage="initial_state", trajectory_time=0.)


def test_preflight_finds_interior_position_overshoot_without_endpoint_sampling(model):
    curve = trajectory([.05, .05], [.4, -.4])
    result = validator(model).preflight(curve, feedback(arm=.05, arm_speed=.4), LIMITS)
    assert result["passed"] is False
    failure = next(row for row in result["violations"] if row["kind"] == "reference_cubic_position")
    assert failure["time_s"] == pytest.approx(.5)
    assert failure["measured"] == pytest.approx(.15)


def test_preflight_finds_interior_speed_peak_when_endpoint_speeds_are_zero(model):
    curve = trajectory([0., .1], [0., 0.], duration=.1)
    result = validator(model).preflight(curve, feedback(arm=0.), LIMITS)
    failure = next(row for row in result["violations"] if row["kind"] == "reference_cubic_velocity")
    assert result["passed"] is False
    assert failure["time_s"] == pytest.approx(.05)
    assert failure["measured"] == pytest.approx(1.5)


def test_preflight_preserves_original_barrier_and_does_not_invent_future_fingers(model):
    curve = trajectory([.05, .05], [.4, -.4], barrier=.1)
    original = deepcopy(curve.events)
    result = validator(model).preflight(curve, feedback(arm=.05, arm_speed=.4), LIMITS)
    assert result["passed"] is True
    assert result["checked_interval_s"] == [0., .1]
    assert result["reference_cubic_extrema"]["left_slide"]["position"]["maximum"] == pytest.approx(.086)
    assert result["future_finger_states_assumed"] is False
    assert result["reference_collision_checked"] is False
    assert result["observed_scene_checked"] is False
    assert curve.events == original


def test_changed_executable_polynomial_and_unmatched_velocity_contract_fail(model):
    curve = trajectory([.03, .04], [0., 0.])
    curve._curve.c[0, 0, 0] += 1.
    with pytest.raises(ValueError, match="actual executable cubic"):
        validator(model).preflight(curve, feedback(), LIMITS)
    model.joint_velocity_limits["left_slide"] = .5
    with pytest.raises(ValueError, match="URDF's named velocity"):
        validator(model)


def test_source_pair_policy_keeps_world_weld_pair_and_individual_geom_masks():
    """A partly enabled body pair must retain its per-geom mask decisions."""
    import mujoco

    sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
    from robot_validation_fixture import _native_pair_policies

    native = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="fixed">
      <geom name="base_geom" type="box" size=".1 .1 .1" contype="3" conaffinity="3"/>
      <body name="first" pos="0 0 1"><joint name="first_q" type="hinge"/>
        <geom name="first_a" type="box" size=".1 .1 .1" contype="1" conaffinity="1"/>
        <geom name="first_b" type="box" size=".1 .1 .1" pos=".3 0 0" contype="2" conaffinity="2"/>
        <body name="child" pos="0 0 1"><joint name="child_q" type="hinge"/>
          <geom name="child_geom" type="box" size=".1 .1 .1"/>
        </body>
      </body>
      <body name="sibling" pos="1 0 0"><joint name="sibling_q" type="hinge"/>
        <geom name="sibling_geom" type="box" size=".1 .1 .1" contype="2" conaffinity="2"/>
      </body>
    </body></worldbody></mujoco>""")
    names = ("base_geom", "first_a", "first_b", "child_geom", "sibling_geom")
    policy = {(a, b): (identity, reason) for a, b, identity, reason in _native_pair_policies(native, names)}
    assert len(policy) == 10
    assert policy["base_geom", "first_a"][0] == "required"
    assert policy["first_a", "first_b"][0] == "excluded"
    assert policy["first_a", "child_geom"][0] == "excluded"
    assert policy["first_a", "sibling_geom"][0] == "excluded"
    assert "individual geom" in policy["first_a", "sibling_geom"][1]
    assert policy["first_b", "sibling_geom"][0] == "required"
# END actual-state and whole-cubic preflight regressions.
