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


# BEGIN legal model contact is distinct from filtering or blanket collision removal.
@pytest.mark.parametrize("finger", [-.01000025, 0.])
def test_explicit_mechanical_contact_preserves_boundary_and_overlap_evidence(model, finger):
    permitted = PairQuery("first", "second", "mechanical_contact", "Declared robot-local mechanical contact")
    gate = validator(model, [permitted])
    result = gate.check_state(feedback(finger=finger), stage="after_step", trajectory_time=.1)
    assert result["passed"] is True
    assert len(result["mechanical_contacts"]) == 1
    assert result["mechanical_contacts"][0]["policy"] == "mechanical_contact"
    assert result["mechanical_contacts"][0]["signed_distance_m"] == result["distances_m"][0]
    assert result["min_required_clearance_m"] is None
    assert gate.describe()["pair_order"][0]["source"] == permitted.source
    # An identical unpermitted pair still fails; closeness never grants permission.
    assert validator(model).check_state(feedback(finger=finger), stage="after_step", trajectory_time=.1)["passed"] is False


def test_mechanical_contact_cannot_exempt_joint_or_fk_checks(model):
    gate = validator(model, [PairQuery("first", "second", "mechanical_contact", "Robot mechanical contact")])
    state = feedback(finger=0., arm_speed=2.)
    state["tool_pose"][0, 3] += .001
    result = gate.check_state(state, stage="after_step", trajectory_time=.1)
    assert result["passed"] is False
    assert {row["kind"] for row in result["violations"]} == {"independent_joint_velocity", "independent_model_fk"}
    assert result["mechanical_contacts"][0]["signed_distance_m"] < 0
# END contact permission changes only the declared geometry-pair acceptance.


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


# BEGIN global native contact switches cannot masquerade as enabled pair policy.
@pytest.mark.parametrize("flag_name", ["mjDSBL_CONTACT", "mjDSBL_CONSTRAINT"])
def test_source_pair_policy_rejects_disabled_contact_physics(flag_name):
    import mujoco

    sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
    from robot_validation_fixture import _native_pair_policies

    native = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="first" type="box" size=".1 .1 .1"/>
      <body name="moving" pos="0 0 1"><joint name="q" type="hinge"/>
        <geom name="second" type="box" size=".1 .1 .1"/>
      </body>
    </worldbody></mujoco>""")
    assert _native_pair_policies(native, ("first", "second"))[0][2] == "required"
    native.opt.disableflags |= int(getattr(mujoco.mjtDisableBit, flag_name))
    with pytest.raises(ValueError, match="contact and constraint processing"):
        _native_pair_policies(native, ("first", "second"))
# END reject each disabled native contact switch independently.
# END actual-state and whole-cubic preflight regressions.

# BEGIN prefix extension preserves the same curve and does not recheck expired obligations.
def test_explicit_prefix_extension_still_checks_known_arm_with_unknown_fingers(model):
    from test_independent_scene_validation import binding
    scene = binding()
    scene["owners"][0]["geometry"]["solids"][0].update(
        dimensions=[.01, .01, .01], pose=[.095, .2, 0., 1., 0., 0., 0.])
    curve = trajectory([.03, .09], [0., 0.], barrier=.4)
    gate = validator(model)
    # BEGIN use progress after the same-curve q/dq/ddq sample components.
    gate.bind_scene(scene, progress_at_time=lambda t: float(curve.sample(t)[3]))
    # END no derivative can replace the original scene progress clock.
    prefix = gate.preflight(curve, feedback(), LIMITS)
    assert prefix["passed"] and not prefix["observed_scene_checked"]
    assert prefix["nominal_scene"]["unresolved_assumptions"][0]["geometry_names"] == ["second"]
    current = feedback(arm=float(curve._curve(.4)[0]), arm_speed=float(curve._curve(.4, 1)[0]))
    extension = gate.preflight(curve, current, LIMITS, start_time=.4, until=1.)
    assert extension["checked_interval_s"] == [.4, 1.]
    assert not extension["passed"]
    assert any(row["kind"] == "independent_scene_collision" for row in extension["violations"])
    assert curve.executable_until == .4


def test_scalar_preflight_new_interval_does_not_replay_old_peak(model):
    curve = trajectory([.03, .03], [.4, -.4])
    gate = validator(model)
    assert not gate.preflight(curve, feedback(arm_speed=.4), LIMITS)["passed"]
    actual = feedback(arm=float(curve._curve(.9)[0]), arm_speed=float(curve._curve(.9, 1)[0]))
    renewed = gate.preflight(curve, actual, LIMITS, start_time=.9, until=1.)
    assert renewed["passed"] and renewed["checked_interval_s"] == [.9, 1.]
    assert renewed["actual_state"]["trajectory_time"] == .9
# END no callback can silently expand or retime an authorized interval.

# BEGIN the real validator must survive the execution adapter without losing cause.
@pytest.mark.parametrize("failure", ["collision", "required_query_gap"])
def test_scene_failures_keep_named_physical_evidence_through_executor(model, failure):
    from manipulation.execution import TrajectoryExecutor
    from test_independent_scene_validation import binding

    scene = binding()
    scene["owners"][0]["geometry"]["solids"][0].update(
        dimensions=[.01, .01, .01], pose=[.03, .2, 0., 1., 0., 0., 0.])
    expected = "independent_scene_collision"
    if failure == "required_query_gap":
        scene["owners"][0]["geometry"] = {"solids": [], "query_gaps": ["unsupported_required_native_shape"]}
        expected = "independent_scene_query_incomplete"
    gate = validator(model)
    curve = trajectory([.03, .03], [0., 0.])
    # BEGIN use progress after the same-curve q/dq/ddq sample components.
    gate.bind_scene(scene, progress_at_time=lambda t: float(curve.sample(t)[3]))
    # END no derivative can replace the original scene progress clock.
    adapter = SimpleNamespace(geometry_checker=gate, _geometry_pair_count=1)
    for method, args, kwargs in (
        ("check_state", (feedback(),), {"stage": "actual", "trajectory_time": 0.}),
        ("preflight", (curve, feedback(), LIMITS), {}),
    ):
        result = TrajectoryExecutor._geometry_result(adapter, method, *args, **kwargs)
        assert not result["passed"]
        assert all(row["kind"] != "geometry_checker_error" and row["name"] for row in result["violations"])
        assert any(row["kind"] == expected for row in result["violations"])
        if failure == "required_query_gap":
            assert any(row.get("reason") == "unsupported_required_native_shape" for row in result["violations"])


def test_prefix_preflight_uses_actual_feedback_time_separately_from_authorized_start(model):
    curve = trajectory([.03, .03], [0., 0.])
    result = validator(model).preflight(curve, feedback(), LIMITS, start_time=.4, until=1., current_time=.5)
    assert result["passed"]
    assert result["checked_interval_s"] == [.4, 1.]
    assert result["actual_state"]["trajectory_time"] == .5
# END native device tick alignment does not relabel the actual contact-rule time.


# BEGIN shared planning solid policy needs no sphere proxy or GPU state.
def planning_hand(*, controller="binary", current=-.08, nominal=None):
    """Minimal physical hand contract; its command deliberately predicts contact."""
    return SimpleNamespace(controller=controller, joint_names=NAMES[1:],
        current_positions=np.array([current]), joint_limits=np.array([[-.1, 0.]]),
        nominal_finger_trajectory=nominal,
        finger_trajectory=SimpleNamespace(sample=lambda time: (np.array([0.]), np.array([0.]))))


def planning_problem(names=NAMES[:1], positions=(.03,)):
    return SimpleNamespace(joint_names=names, start_q=np.asarray(positions),
        reference_frame="world", base_pose=np.eye(4), tool_link="tool", tool_transform=np.eye(4))


@pytest.mark.parametrize("finger,passed,relation", [
    (-.08, True, "separated"), (0., False, "penetrating"),
    (-.01000025, False, "numerical_boundary"),
])
def test_planning_self_queries_native_solids_without_any_sphere_cost(model, finger, passed, relation):
    gate = validator(model)
    result = gate.check_planning_self([finger, .03], NAMES[::-1], time_s=.2,
                                     gripper=planning_hand(controller="position_tracking"))
    assert result["passed"] is passed
    assert result["residual"] == float(not passed)
    assert result["queried_pairs"] == 1
    assert result["minimum_pair"]["relation"] == relation
    assert result["joint_names"] == list(NAMES)
    assert result["joint_positions"] == [.03, finger]
    assert len(result["violations"]) == int(not passed)
    assert result["unmodelled_geometry_names"] == []


@pytest.mark.parametrize("policy", ["excluded", "mechanical_contact"])
def test_planning_native_pair_permissions_keep_joint_limits(model, policy):
    gate = validator(model, [PairQuery("first", "second", policy, "explicit source policy")])
    accepted = gate.check_planning_self([.03, 0.], NAMES, time_s=.2, gripper=None)
    assert accepted["passed"] and accepted["queried_pairs"] == 0
    assert accepted["minimum_pair"] is None and accepted["minimum_required_clearance_m"] is None
    rejected = gate.check_planning_self([.11, 0.], NAMES, time_s=.2, gripper=None)
    assert not rejected["passed"] and rejected["residual"] == 1.
    assert rejected["violations"] == []
    assert set(rejected["joint_limit_violations"]) == {"left_slide"}


@pytest.mark.parametrize("index,boundary", [(0, "upper"), (1, "lower")])
def test_planning_source_limits_admit_only_exact_active_dtype_endpoints(model, index, boundary):
    # Mechanical contact removes collision as a confounder; the independent
    # scalar joint domain must still reject each adjacent unrepresentable q.
    gate = validator(model, [PairQuery("first", "second", "mechanical_contact", "explicit source permission")])
    q = np.array([.03, -.08], dtype=np.float32)
    q[index] = np.float32(.1 if boundary == "upper" else -.1)
    accepted = gate.check_planning_self(q, NAMES, time_s=.2, gripper=None)
    assert accepted["passed"] and accepted["joint_limit_violations"] == {}
    assert accepted["joint_positions"] == q.astype(float).tolist()
    evidence = accepted["rounded_limit_representations"]
    assert len(evidence) == 1 and evidence[0]["joint_name"] == NAMES[index]
    assert evidence[0]["represented_boundary"] == boundary
    assert evidence[0]["position"] == float(q[index])
    assert evidence[0]["position"] != evidence[0][boundary]
    adjacent = q.copy()
    adjacent[index] = np.nextafter(q[index], np.float32(np.inf if boundary == "upper" else -np.inf))
    for values in (adjacent, q.astype(np.float64)):
        rejected = gate.check_planning_self(values, NAMES, time_s=.2, gripper=None)
        assert not rejected["passed"] and rejected["rounded_limit_representations"] == []
        assert set(rejected["joint_limit_violations"]) == {NAMES[index]}
    actual = feedback(arm=float(q[0]), finger=float(q[1]))
    assert any(row["kind"] == "independent_joint_position"
               for row in gate.check_state(actual, stage="actual", trajectory_time=.2)["violations"])


def test_planning_active_dtype_cannot_relax_independent_nominal_finger_limits(model):
    gate = validator(model, [PairQuery("first", "second", "mechanical_contact", "explicit source permission")])
    nominal = SimpleNamespace(joint_names=NAMES[1:], duration=1.,
        sample=lambda time: (np.array([float(np.float32(-.1))]), np.array([0.])))
    result = gate.check_planning_self(np.array([.03], dtype=np.float32), NAMES[:1], time_s=.2,
        gripper=planning_hand(controller="timed_reference", nominal=nominal))
    assert not result["passed"] and result["rounded_limit_representations"] == []
    assert set(result["joint_limit_violations"]) == {"right_slide"}


@pytest.mark.parametrize("controller", ["binary", "timed_reference"])
def test_planning_unknown_future_fingers_never_use_the_control_command(model, controller):
    gate = validator(model)
    hand = planning_hand(controller=controller)
    actual = gate.check_planning_self([.03], NAMES[:1], time_s=0., gripper=hand)
    assert actual["passed"] and actual["queried_pairs"] == 1
    assert actual["joint_positions"] == [.03, -.08]
    future = gate.check_planning_self([.03], NAMES[:1], time_s=.2, gripper=hand)
    assert future["passed"] and future["queried_pairs"] == 0
    assert future["unmodelled_geometry_names"] == ["second"]
    nominal = SimpleNamespace(joint_names=NAMES[1:], duration=1.,
        sample=lambda time: (np.array([-.08+.08*time]), np.array([.08])))
    hand.nominal_finger_trajectory = nominal
    middle = gate.check_planning_self([.03], NAMES[:1], time_s=.5, gripper=hand)
    assert middle["passed"] and middle["joint_positions"] == [.03, -.04]
    finish = gate.check_planning_self([.03], NAMES[:1], time_s=1., gripper=hand)
    assert not finish["passed"] and finish["queried_pairs"] == 1
    assert finish["unmodelled_geometry_names"] == []


def test_planning_nominal_finger_endpoint_preserves_exact_native_time_representation(model):
    from scipy.interpolate import CubicHermiteSpline
    from manipulation.gripper_trajectory import FingerTrajectory

    duration = 9*float(np.float32(.2))
    nominal = SimpleNamespace(joint_names=NAMES[1:], duration=duration,
        _curve=CubicHermiteSpline([0., duration], [[-.08], [-.04]], [[0.], [0.]]))
    # Use the production sampler's strict physical domain and polynomial;
    # actuator event generation is independent of this representation check.
    nominal.sample = lambda time: FingerTrajectory.sample(nominal, time)
    hand = planning_hand(controller="timed_reference", nominal=nominal)
    gate = validator(model)
    endpoint = np.float32(duration)
    assert float(endpoint) > duration
    result = gate.check_planning_self([.03], NAMES[:1], time_s=endpoint, gripper=hand)
    final_position = float(nominal.sample(duration)[0][0])
    assert result["passed"] and result["joint_positions"] == [.03, final_position]
    interior = gate.check_planning_self([.03], NAMES[:1], time_s=1, gripper=hand)
    assert interior["joint_positions"][1] == float(nominal.sample(1.)[0][0])
    assert interior["joint_positions"][1] != final_position
    for beyond in (np.nextafter(endpoint, np.float32(np.inf)), float(endpoint),
                   np.nextafter(duration, np.inf)):
        with pytest.raises(ValueError, match="original trajectory duration"):
            gate.check_planning_self([.03], NAMES[:1], time_s=beyond, gripper=hand)


def test_planning_unknown_finger_keeps_arm_against_fixed_installation(model):
    from itertools import combinations
    fixed_pose = np.eye(4)
    fixed_pose[:3, 3] = [.03, .2, 0.]
    second_pose = np.eye(4)
    second_pose[1, 3] = .4
    checker = RobotGeometryChecker(model.urdf_path, base_frame="base", geometries=[
        NativePrimitive("first", "left", "box", (.04, .04, .04), np.eye(4), "moving arm solid"),
        NativePrimitive("second", "right", "box", (.04, .04, .04), second_pose, "moving finger solid"),
        NativePrimitive("mount", "base", "box", (.01, .01, .01), fixed_pose, "fixed installation")])
    pairs = [PairQuery(a, b, "required", "native enabled pair")
             for a, b in combinations(checker.geometry_names, 2)]
    result = validator(checker, pairs).check_planning_self([.03], NAMES[:1], time_s=1., gripper=planning_hand())
    assert not result["passed"] and result["queried_pairs"] == 1
    assert result["unmodelled_geometry_names"] == ["second"]
    assert [(row["first"], row["second"]) for row in result["violations"]] == [("first", "mount")]


def test_planning_robot_without_hand_requires_complete_native_coordinates(model):
    gate = RobotStateValidator(model, joint_names=NAMES, arm_joint_names=NAMES, finger_joint_names=(),
        pairs=[PairQuery("first", "second", "required", "native enabled pair")],
        named_velocity_limits={name: 1. for name in NAMES}, reference_frame="world",
        T_reference_base=np.eye(4), tool_frame="tool", T_frame_tcp=np.eye(4),
        fk_translation_tolerance_m=1e-8, fk_rotation_tolerance_rad=1e-8, source="robot without hand")
    gate.validate_planning_binding(planning_problem(NAMES, [.03, -.08]), None)
    assert gate.check_planning_self([.03, -.08], NAMES, time_s=.2, gripper=None)["passed"]
    with pytest.raises(ValueError, match="every native joint"):
        gate.check_planning_self([.03], NAMES[:1], time_s=.2, gripper=None)


def test_planning_binding_preserves_calibration_joint_partition_and_measured_fingers(model):
    gate, problem, hand = validator(model), planning_problem(), planning_hand()
    gate.validate_planning_binding(problem, hand)
    for field in ("base_pose", "tool_transform"):
        wrong = deepcopy(problem)
        getattr(wrong, field)[0, 3] = .01
        with pytest.raises(ValueError, match="identity"):
            gate.validate_planning_binding(wrong, hand)
    wrong = deepcopy(hand)
    wrong.joint_limits[0, 0] = -.2
    with pytest.raises(ValueError, match="mechanical limits"):
        gate.validate_planning_binding(problem, wrong)
    hand.controller = "position_tracking"
    with pytest.raises(ValueError, match="active planning coordinates"):
        gate.validate_planning_binding(problem, hand)
    active = planning_problem(NAMES, [.03, -.08])
    gate.validate_planning_binding(active, hand)
    active.start_q[1] = -.04
    with pytest.raises(ValueError, match="measured entrance"):
        gate.validate_planning_binding(active, hand)


def test_planning_binding_records_owned_source_geometry_and_complete_policy(model):
    import json
    gate = validator(model)
    record = gate.planning_binding_record()
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["pair_order"][0]["policy"] == "required"
    assert "left_slide" in record["model"]["urdf_xml"]
    assert record["model"]["geometries"][0]["dimensions"] == [.04, .04, .04]
    assert record["numerics"] == model.numerics
    record["model"]["geometries"][0]["dimensions"][0] = 10.
    assert gate.planning_binding_record()["model"]["geometries"][0]["dimensions"][0] == .04
# END every planning decision uses the same native solid and mechanical policy.


# BEGIN execution checks the exact robot model saved with accepted planning inputs.
@pytest.mark.parametrize("changed", ["geometry", "pair_policy"])
def test_execution_scene_rejects_different_planned_source_solids_or_pair_policy(model, changed):
    from test_independent_scene_validation import binding
    gate, scene = validator(model), binding()
    scene["robot_self_collision"] = gate.planning_binding_record()
    if changed == "geometry":
        scene["robot_self_collision"]["model"]["geometries"][0]["dimensions"][0] = .08
    else:
        scene["robot_self_collision"]["pair_order"][0]["policy"] = "mechanical_contact"
    with pytest.raises(ValueError, match="source robot geometry and pair policy"):
        gate.bind_scene(scene, progress_at_time=lambda time: time)
    assert not hasattr(gate, "_scene_binding")


def test_execution_scene_rebinding_preserves_frozen_robot_identity_after_first_bind(model):
    from test_independent_scene_validation import binding
    gate, scene = validator(model), binding()
    identity = gate.planning_binding_record()
    scene["robot_self_collision"] = identity
    for _ in range(2):
        gate.bind_scene(scene, progress_at_time=lambda time: time)
        assert gate.describe()["scene_bound"] is True
        assert gate.planning_binding_record() == identity
    standalone = binding()
    standalone["robot_self_collision"] = None
    gate.bind_scene(standalone, progress_at_time=lambda time: time)
    assert gate._scene_binding["robot_self_collision"] is None
# END source equality is independent of mutable execution-scene bookkeeping.
