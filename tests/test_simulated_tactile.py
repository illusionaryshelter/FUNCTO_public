"""Small CPU native-physics checks of the explicitly declared tactile channel."""

# BEGIN current sensor facts do not consume a hidden object pose or offline grade.
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))
from simulated_tactile import SimulatedTactileObserver, _sampled_span_bounds


_XML = """<mujoco>
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <default><geom friction="1 .01 .001" condim="3"/><joint damping="3"/></default>
  <worldbody>
    <geom name="declared_owner_patch" type="box" size=".01 .02 .02"/>
    <body name="left_pad_body" pos="-.04 0 0">
      <joint name="left_q" type="slide" axis="1 0 0" range="0 .03"/>
      <geom name="left_pad" type="box" size=".008 .015 .015" mass=".05"/>
    </body>
    <body name="right_pad_body" pos=".04 0 0">
      <joint name="right_q" type="slide" axis="-1 0 0" range="0 .03"/>
      <geom name="right_pad" type="box" size=".008 .015 .015" mass=".05"/>
    </body>
  </worldbody>
  <actuator><position joint="left_q" kp="1000" ctrlrange="0 .03"/>
    <position joint="right_q" kp="1000" ctrlrange="0 .03"/></actuator>
</mujoco>"""


def _fixture():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = ("left_q", "right_q")
    simulation = SimpleNamespace(native_model=model, native_data=data, finger_joint_names=names)
    config = {"source": "Declared CPU two-pad sensor: native contact measurements and actual finger feedback",
        "owner_id": "declared_owner", "effect": {"condition_id": "task:pinch", "skill_id": "task", "event_id": "pinch"},
        "target_geometry_names": ["declared_owner_patch"],
        "contact_groups": [{"material_id": "left_material", "native_geometry_names": ["left_pad"]},
                           {"material_id": "right_material", "native_geometry_names": ["right_pad"]}],
        "finger_joint_names": list(names), "stable_window_s": .04, "maximum_sample_gap_s": .004,
        "minimum_normal_force_n": [.01, .01], "maximum_contact_point_span_m": .001,
        "maximum_contact_normal_span_rad": .02, "maximum_opposition_error_rad": .05,
        "maximum_finger_position_span": [.0005, .0005], "maximum_contacts_per_sample": 64}
    binding = {"problem": {"reference_frame": "world", "tool_link": "declared_actual_tcp", "conditions": [{**config["effect"], "carrier_id": "declared_owner",
        "effect_requirement": "stable_grasp_from_current_synchronized_evidence"}]},
        "scene": {"owners": [{"owner_id": "declared_owner"}],
            "native_geometry_materials": {"left_pad": "left_material", "right_pad": "right_material"},
            "contacts": [{"actor": "robot", "material_id": material, "target_owner": "declared_owner"}
                         for material in ("left_material", "right_material")]}}
    policy = {"source": "Explicit native tactile sample schedule", "period_s": .002,
              "maximum_age_s": 0., "required_fields": []}

    def feedback():
        return {"joint_names": names, "finger_joint_names": names, "q": data.qpos.copy(),
            "dq": data.qvel.copy(), "tool_pose": np.eye(4), "timestamp_s": float(data.time),
            "reference_frame": "world", "tool_link": "declared_actual_tcp",
            "feedback_id": f"actual native time {float(data.time):.9f}"}

    return simulation, config, binding, policy, feedback


def _sensor(simulation, config, binding, policy):
    return SimulatedTactileObserver(simulation, config, execution_binding=binding,
                                    observation_policy=policy, maximum_samples=501)


def test_actual_pinching_needs_a_current_full_window_and_lost_touch_erases_current_true():
    simulation, config, binding, policy, feedback = _fixture()
    sensor = _sensor(simulation, config, binding, policy)
    assert sensor(feedback(), 0.)["effects"]["task:pinch"]["value"] == "F"
    simulation.native_data.ctrl[:] = .027
    saw_unverified_contact = False
    for _ in range(180):
        mujoco.mj_step(simulation.native_model, simulation.native_data)
        mujoco.mj_forward(simulation.native_model, simulation.native_data)
        observation = sensor(feedback(), float(simulation.native_data.time))
        row = sensor.observations[-1]
        if min(row["normal_force_n"]) >= .01 and observation["effects"]["task:pinch"]["value"] != "T":
            saw_unverified_contact = True
    assert saw_unverified_contact
    assert observation["effects"]["task:pinch"]["value"] == "T"
    assert observation["effects"]["task:pinch"]["window"]["complete_window"]
    assert "scene_owner_poses" not in observation and "scene_condition_evidence" not in observation
    assert not sensor.describe()["owner_pose_read"]
    assert not sensor.describe()["attachment_observation_provided"]
    assert max(abs(value) for value in sensor.observations[-1]["inward_normal_tcp"][0][1:]) < 1e-10
    # The current fact must become false after release even though the same
    # sensor really established T earlier; this is no occurred-event latch.
    simulation.native_data.ctrl[:] = 0.
    for _ in range(50):
        mujoco.mj_step(simulation.native_model, simulation.native_data)
        mujoco.mj_forward(simulation.native_model, simulation.native_data)
        observation = sensor(feedback(), float(simulation.native_data.time))
    assert observation["effects"]["task:pinch"]["value"] == "F"
    assert any(row["effect"]["value"] == "T" for row in sensor.observations)


def test_existing_contact_has_unknown_history_and_sampling_gap_cannot_certify_stability():
    simulation, config, binding, policy, feedback = _fixture()
    simulation.native_data.ctrl[:] = .027
    for _ in range(180):
        mujoco.mj_step(simulation.native_model, simulation.native_data)
    mujoco.mj_forward(simulation.native_model, simulation.native_data)
    sensor = _sensor(simulation, config, binding, policy)
    first = sensor(feedback(), 0.)
    assert first["effects"]["task:pinch"]["value"] == "U"
    for _ in range(30):
        mujoco.mj_step(simulation.native_model, simulation.native_data)
    mujoco.mj_forward(simulation.native_model, simulation.native_data)
    second = sensor(feedback(), .06)
    assert second["effects"]["task:pinch"]["value"] == "U"
    assert not second["effects"]["task:pinch"]["window"]["continuous_sampling"]
    count = len(sensor.observations)
    assert sensor(feedback(), .06) == second
    assert len(sensor.observations) == count


def test_material_owner_effect_and_native_identity_mismatches_are_rejected():
    for change in (
        lambda config, binding: config.update(owner_id="another owner"),
        lambda config, binding: config["effect"].update(event_id="another event"),
        lambda config, binding: config["contact_groups"][0].update(material_id="unbound material"),
        lambda config, binding: config["contact_groups"][1].update(native_geometry_names=["left_pad"]),
        lambda config, binding: binding["scene"].update(contacts=[]),
    ):
        simulation, config, binding, policy, _ = _fixture()
        change(config, binding)
        with pytest.raises(ValueError):
            _sensor(simulation, config, binding, policy)


def test_robot_tcp_motion_is_measured_in_the_contact_frame_and_not_hidden_by_stable_names():
    simulation, config, binding, policy, feedback = _fixture()
    simulation.native_data.ctrl[:] = .027
    for _ in range(180):
        mujoco.mj_step(simulation.native_model, simulation.native_data)
    mujoco.mj_forward(simulation.native_model, simulation.native_data)
    sensor = _sensor(simulation, config, binding, policy)
    for index in range(30):
        mujoco.mj_step(simulation.native_model, simulation.native_data)
        mujoco.mj_forward(simulation.native_model, simulation.native_data)
        actual = feedback()
        # An explicitly varied robot TCP measurement makes the tactile points
        # slide in its frame. The same geometry names and forces are not T.
        actual["tool_pose"][1, 3] = .0002*index
        observation = sensor(actual, index*.002)
    effect = observation["effects"]["task:pinch"]
    assert effect["value"] == "F"
    assert max(effect["window"]["contact_point_witness_spans_m"]) > config["maximum_contact_point_span_m"]


def test_opposite_small_normal_changes_cannot_borrow_the_first_normal_as_a_span_proof():
    simulation, config, binding, policy, feedback = _fixture()
    config["maximum_contact_point_span_m"] = .01
    config["maximum_contact_normal_span_rad"] = .02
    simulation.native_data.ctrl[:] = .027
    for _ in range(180):
        mujoco.mj_step(simulation.native_model, simulation.native_data)
    mujoco.mj_forward(simulation.native_model, simulation.native_data)
    sensor = _sensor(simulation, config, binding, policy)
    # Every later normal remains within .015 rad of the first, yet two real
    # measurements differ by .03 rad and violate the original .02 rad span.
    for angle in (0., *([.015, -.015]*10)):
        mujoco.mj_step(simulation.native_model, simulation.native_data)
        mujoco.mj_forward(simulation.native_model, simulation.native_data)
        actual = feedback()
        actual["tool_pose"][:3, :3] = Rotation.from_rotvec([0., 0., angle]).as_matrix()
        observation = sensor(actual, float(simulation.native_data.time))
    effect = observation["effects"]["task:pinch"]
    assert effect["window"]["complete_window"]
    assert max(effect["window"]["contact_normal_witness_spans_rad"]) > .02
    assert max(effect["window"]["contact_normal_span_upper_bounds_rad"]) > .02
    assert effect["value"] == "F"


def test_coordinate_box_diameter_is_an_upper_bound_and_not_a_false_counterexample():
    points = np.asarray([[-1., 0., 0.], [1., 0., 0.], [0., -1., 0.], [0., 1., 0.]])[:, None]
    witness, upper = _sampled_span_bounds(points)
    assert witness[0] == 2.
    assert upper[0] == np.sqrt(8.)
    # A threshold between these values is unresolved by this bounded query;
    # the larger AABB diagonal is not a distance between two measured points.
    assert witness[0] < 2.5 < upper[0]
# END these are CPU sensor tests, not a full planned-grasp or production-vision claim.
