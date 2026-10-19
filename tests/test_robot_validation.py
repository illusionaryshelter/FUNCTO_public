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
    assert result["distances_m"] == []
    assert gate.describe()["pair_order"] == []
    assert gate.describe()["pair_policy"][0]["source"] == "verified native welded-body rule"
    assert gate.describe()["continuous_collision_certified"] is False


def test_numerical_contact_boundary_is_not_a_contact_permission(model):
    result = validator(model).check_state(feedback(finger=-.01000025), stage="before_step", trajectory_time=0.)
    assert result["passed"] is False
    assert result["violations"][0]["relation"] == "numerical_boundary"


# BEGIN legal model contact is distinct from filtering or blanket collision removal.
@pytest.mark.parametrize("finger", [-.01000025, 0.])
def test_explicit_mechanical_contact_retains_policy_without_querying_permitted_pair(model, finger):
    permitted = PairQuery("first", "second", "mechanical_contact", "Declared robot-local mechanical contact")
    gate = validator(model, [permitted])
    result = gate.check_state(feedback(finger=finger), stage="after_step", trajectory_time=.1)
    assert result["passed"] is True
    assert "mechanical_contacts" not in result
    assert result["distances_m"] == []
    assert gate.describe()["pair_order"] == []
    assert result["min_required_clearance_m"] is None
    assert gate.describe()["pair_policy"][0]["source"] == permitted.source
    # An identical unpermitted pair still fails; closeness never grants permission.
    assert validator(model).check_state(feedback(finger=finger), stage="after_step", trajectory_time=.1)["passed"] is False


def test_mechanical_contact_cannot_exempt_joint_or_fk_checks(model):
    gate = validator(model, [PairQuery("first", "second", "mechanical_contact", "Robot mechanical contact")])
    state = feedback(finger=0., arm_speed=2.)
    state["tool_pose"][0, 3] += .001
    result = gate.check_state(state, stage="after_step", trajectory_time=.1)
    assert result["passed"] is False
    assert {row["kind"] for row in result["violations"]} == {"independent_joint_velocity", "independent_model_fk"}
    assert "mechanical_contacts" not in result
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


def test_planning_robot_without_hand_requires_complete_native_coordinates(model):
    gate = RobotStateValidator(model, joint_names=NAMES, arm_joint_names=NAMES, finger_joint_names=(),
        pairs=[PairQuery("first", "second", "required", "native enabled pair")],
        named_velocity_limits={name: 1. for name in NAMES}, reference_frame="world",
        T_reference_base=np.eye(4), tool_frame="tool", T_frame_tcp=np.eye(4),
        fk_translation_tolerance_m=1e-8, fk_rotation_tolerance_rad=1e-8, source="robot without hand")
    gate.validate_planning_binding(planning_problem(NAMES, [.03, -.08]), None)
    with pytest.raises(ValueError, match="every native joint"):
        gate.validate_planning_binding(planning_problem(NAMES[:1], [.03]), None)


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


# END entrance identity stays independent of nominal task quality.


# BEGIN production runtime shares online fields, sphere layout and actual joints.
def online_binding(gate, *, distance=1., supported=True):
    from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
    from manipulation.planning_scene import ObservedSpace
    shape = (9, 9, 9)
    # BEGIN represent a genuinely unsupported boundary, not an empty remote map.
    # The left solid's measured sphere centre is x=.03. A field starting at
    # x=.04 has no sample there, but its first occupied cell and that sphere
    # overlap. An unrelated all-positive field at x=10 correctly has no such
    # scene-update obligation; the previous fixture did not test this failure.
    values = np.full(shape, distance, np.float32)
    if not supported:
        values[0, :, :] = 0.
    # BEGIN construct the explicit source whose boundary requires an update.
    from observed_geometry_fixtures import field_from_samples
    field = field_from_samples(values,
        origin=[-1., -1., -1.] if supported else [.04, -1., -1.],
        voxel_size=.25, observed=np.ones(shape, bool), owner_ids=("background",), source="online sensor source")
    # END source geometry and numerical support have independent meanings.
    # END unqueried numerical support remains independent of collision evidence.
    coverage = ObservedSpace(np.ones(shape, bool), origin=[-1., -1., -1.], voxel_size=.25,
        observation_ids=["observation"], reference_frame="world")
    # BEGIN complete native box covers and explicit source identities.
    # Both fixture boxes have 40 mm sides. The second box is translated 400 mm
    # in its link frame; its cover must include that native local transform.
    radius = float(np.sqrt(3.) * .02 + 1e-6)
    centers = [[0., 0., 0.], [0., .4, 0.]]
    bodies = [{'body_id': link, 'frame': link, 'sphere_indices': [index],
        'geometry_names': [name], 'local_spheres': [[*centers[index], radius]],
        'source': 'Complete circum-sphere of the explicit native test box'}
        for index, (link, name) in enumerate((('left', 'first'), ('right', 'second')))]
    scene = OnlinePlanningScene(background=field, owners=(), coverage=coverage,
        reference_frame="world", robot_sphere_materials=("left", "right"),
        robot_geometry_source="native sphere fixture", has_gripper=False,
        robot_collision_bodies=bodies)
    scene.bind_online_robot(robot_sphere_geometry=[
        {"link_name": link, "sphere_indices": [index], "centers_local": [centers[index]], "radii": [radius]}
        for index, link in enumerate(("left", "right"))],
        # BEGIN bind the same native solids and mechanical pair policy.
        native_robot_model=gate.checker.model_record(), native_joint_names=gate.joint_names,
        native_self_pairs=[{"first": pair.first, "second": pair.second,
            "margin": max(gate.checker.numerics["gjk_tolerance"], gate.checker.numerics["epa_tolerance"])}
            for pair in gate.required_pairs],
        # END observed field queries remain separate scene components.
        robot_model={"urdf_path": str(gate.checker.urdf_path), "base_frame": gate.checker.base_frame,
            "tool_frame": gate.tool_frame, "T_reference_base": gate.T_reference_base.tolist(),
            "T_frame_tcp": gate.T_frame_tcp.tolist()})
    # END native confirmation receives the same complete robot sphere inventory.
    return scene.to_execution_binding()


# BEGIN execution never rebuilds or queries a frozen world, even at preflight.
def test_preflight_and_actual_tick_query_required_robot_pairs_only(model, monkeypatch):
    from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
    gate = validator(model)
    curve = trajectory([.03, .03], [0., 0.])
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen-world queries are outside physical execution")
    monkeypatch.setattr(model, "bind_scene", forbidden)
    monkeypatch.setattr(model, "query_scene", forbidden)
    monkeypatch.setattr(OnlinePlanningScene, "from_binding", forbidden)
    monkeypatch.setattr(OnlinePlanningScene, "query_state", forbidden)
    monkeypatch.setattr(FrozenESDF, "signed_distance", forbidden)
    native_query = model.query
    calls = []
    def query(**kwargs):
        calls.append(deepcopy(kwargs))
        return native_query(**kwargs)
    monkeypatch.setattr(model, "query", query)
    assert not hasattr(gate, "bind_scene")
    result = gate.preflight(curve, feedback(), LIMITS)
    assert result["passed"] and not result["observed_scene_checked"]
    assert len(calls) == 1 and calls[0]["pairs"] == gate.required_pairs
    assert calls[0]["frames"] == ["tool"]
    assert result["reference_collision_checked"] is False
    assert len(result["actual_state"]["distances_m"]) == len(gate.describe()["pair_order"])
    gate.check_state(feedback(finger=-.08), stage="actual", trajectory_time=.1)
    assert calls[-1]["joint_positions"][1] == -.08
    assert calls[-1]["pairs"] == gate.required_pairs
    assert gate.preflight(curve, feedback(), LIMITS, start_time=.2, current_time=.2, until=.8)["passed"]


@pytest.mark.parametrize("failure", ["collision", "unsupported_field"])
def test_frozen_world_failures_remain_planning_verdicts_only(model, failure):
    from manipulation.execution import TrajectoryExecutor
    from manipulation.online_geometry import OnlinePlanningScene
    gate = validator(model)
    binding = online_binding(gate, distance=-.1 if failure == "collision" else 1.,
                             supported=failure != "unsupported_field")
    scene = OnlinePlanningScene.from_binding(binding)
    state = feedback()
    robot = model.query(joint_names=NAMES, joint_positions=state["q"],
                        T_reference_base=np.eye(4), pairs=(), frames=["left", "right"])
    spheres = np.empty((2, 4))
    for row in binding["robot_sphere_geometry"]:
        pose = np.asarray(robot["frame_poses"][row["link_name"]])
        ids = row["sphere_indices"]
        spheres[ids, :3] = np.asarray(row["centers_local"]) @ pose[:3, :3].T + pose[:3, 3]
        spheres[ids, 3] = row["radii"]
    planning = scene.query_state(spheres, state["tool_pose"], 0., joint_position=state["q"],
        joint_names=NAMES)
    if failure == "collision":
        assert not planning["passed"] and planning["components"]["collision/robot/background"] > 0
    else:
        assert planning["passed"]
        assert not any(name.startswith("scene_update/") for name in planning["components"])
    adapter = SimpleNamespace(geometry_checker=gate, _geometry_pair_count=len(gate.required_pairs))
    actual = TrajectoryExecutor._geometry_result(adapter, "check_state", state,
        stage="actual", trajectory_time=0.)
    assert actual["passed"] and "scene" not in actual
# END bad frozen observations cannot erase actual robot safety checks or certify the world.
