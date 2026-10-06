"""CPU coverage binding from current named feedback and a real native fixture."""

# BEGIN actual fixture/FK regressions without physics, rendering or CUDA.
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
from robot_coverage_fixture import build_robot_coverage
from robot_fixture import export_fixture


@pytest.fixture(scope="module")
def current_robot(tmp_path_factory):
    """Export real source solids, then observe another named robot posture.

    The HDF5 input belongs only to the existing fixture exporter. Coverage
    receives a later q/dq snapshot with no recording status or provenance;
    its TCP is calculated independently by MuJoCo robot-only kinematics.
    """
    root = tmp_path_factory.mktemp("current_robot_coverage")
    source = root / "robot.xml"
    source.write_text('''<mujoco><compiler angle="radian"/><asset/>
      <worldbody><body name="base" pos=".2 -.1 .4">
        <geom name="mount" type="box" size=".03 .03 .03"/>
        <body name="assembly">
        <body name="arm" pos="0 0 .2">
          <joint name="arm_slide" type="slide" axis="1 0 0" limited="true" range="-.5 .5"/>
          <geom name="arm_solid" type="box" size=".02 .02 .04"/>
          <body name="finger" pos="0 .1 .1">
            <joint name="finger_slide" type="slide" axis="0 1 0" limited="true" range="0 .04"/>
            <geom name="finger_solid" type="box" size=".005 .015 .03"/>
            <site name="tcp" pos=".02 0 .08"/>
          </body>
        </body>
        </body>
      </body></worldbody></mujoco>''')
    names = ["arm_slide", "finger_slide"]
    catalog = root / "catalog.json"
    catalog.write_text(json.dumps({"joints": [
        {"name": name, "qpos_count": 1, "qvel_count": 1,
         "qpos_start": index, "qvel_start": index}
        for index, name in enumerate(names)]}))
    episode = root / "robot_positions.h5"
    with h5py.File(episode, "w") as stream:
        stream.create_dataset("states/qpos", data=[[0., 0.], [.2, .01]])
    fixture = export_fixture(
        scene_path=source, catalog_path=catalog, episode_path=episode,
        root_body="base", collision_root="assembly", tcp_site="tcp",
        arm_joint_names=names[:1], finger_joint_names=names[1:],
        velocity_limits=dict.fromkeys(names, 1.), effort_limits=dict.fromkeys(names, 2.),
        cover_specifications={name: {"subdivisions": [1, 1, 1]}
                              for name in ("arm_solid", "finger_solid")},
        initial_frame=0, output=root / "fixture", binding_audit_path=catalog,
        max_acceleration=2., max_jerk=10., mechanical_contact_pairs=[])
    model = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    data = mujoco.MjData(model)
    q, dq = [.08, .023], [.12, -.015]
    for name, position, velocity in zip(names, q, dq):
        data.qpos[model.joint(name).qposadr[0]] = position
        data.qvel[model.joint(name).dofadr[0]] = velocity
    mujoco.mj_kinematics(model, data)
    tcp, base = np.eye(4), np.eye(4)
    tcp[:3, :3] = data.site_xmat[model.site("tcp").id].reshape(3, 3)
    tcp[:3, 3] = data.site_xpos[model.site("tcp").id]
    base[:3, :3] = data.xmat[model.body("base").id].reshape(3, 3)
    base[:3, 3] = data.xpos[model.body("base").id]
    feedback = {"joint_names": names, "start_q": q, "start_dq": dq,
        "reference_frame": "world", "base_pose": base.tolist(),
        "tool_link": "tcp", "tool_frame": "tcp", "tool_transform": np.eye(4).tolist(),
        "eef_pose": tcp.tolist(), "source": "Current named robot q/dq; independent native robot-only FK"}
    return fixture, feedback


def build(current_robot, *, changed=None, initialization_source="Explicit valid current model-state assumption"):
    fixture, feedback = current_robot
    current = deepcopy(feedback)
    if changed is not None:
        current.update(changed)
    # An empty original free grid cannot manufacture environmental evidence.
    coverage = SimpleNamespace(known_free=np.zeros((2, 2, 2), bool),
        origin=np.zeros(3), voxel_size=.1, reference_frame="world")
    oracle = build_robot_coverage(fixture, current, coverage,
        initialization_source=initialization_source, max_queries=2)
    return oracle, current, coverage


def test_current_snapshot_binds_real_coverage_without_recording_markers(current_robot):
    original = deepcopy(current_robot)
    oracle, feedback, coverage = build(current_robot)
    assert "status" not in feedback and "provenance" not in feedback
    assert feedback["start_q"] != current_robot[0]["initial_joint_positions"]
    expected = dict(zip(feedback["joint_names"], feedback["start_q"]))
    np.testing.assert_array_equal(oracle.baseline_joint_positions,
        [expected[name] for name in oracle.baseline_joint_names])
    assert oracle.initialization_source.endswith("measured snapshot: " + feedback["source"])
    assert oracle.check_state(joint_names=feedback["joint_names"],
        joint_positions=feedback["start_q"], entrance=True)["passed"]
    moved = list(feedback["start_q"])
    moved[0] += .1
    result = oracle.check_state(joint_names=feedback["joint_names"], joint_positions=moved)
    assert not result["passed"] and result["failure"] == "new_volume_without_free_evidence"
    assert not coverage.known_free.any()
    assert oracle.describe()["new_free_cells_added"] == 0
    assert current_robot == original


@pytest.mark.parametrize("names", [
    ["arm_slide", "foreign_joint"], ["arm_slide", "arm_slide"], ["finger_slide", "arm_slide"],
])
def test_wrong_incomplete_or_reordered_joint_identity_fails(current_robot, names):
    with pytest.raises(ValueError, match="complete named feedback"):
        build(current_robot, changed={"joint_names": names})


@pytest.mark.parametrize("key", ["start_q", "start_dq"])
@pytest.mark.parametrize("values", [[np.nan, .01], [0.], [0., np.inf]])
def test_q_and_dq_need_complete_finite_values(current_robot, key, values):
    with pytest.raises(ValueError, match="finite|named coordinates"):
        build(current_robot, changed={key: values})


def test_native_joint_limits_and_original_fk_tolerance_remain_required(current_robot):
    with pytest.raises(ValueError, match="native q0/FK"):
        build(current_robot, changed={"start_q": [.500001, .023]})
    for error in (2e-8, np.nan):
        pose = np.asarray(current_robot[1]["eef_pose"]).copy()
        pose[0, 3] += error
        with pytest.raises(ValueError, match="native q0/FK"):
            build(current_robot, changed={"eef_pose": pose.tolist()})


@pytest.mark.parametrize("key,value", [
    ("reference_frame", "different_frame"), ("tool_link", "another_link"),
    ("tool_frame", "another_tcp"), ("source", ""), ("source", None),
])
def test_current_frame_tool_identity_and_source_are_explicit(current_robot, key, value):
    with pytest.raises(ValueError, match="reference frame|tool identity|feedback source"):
        build(current_robot, changed={key: value})


@pytest.mark.parametrize("key", ["base_pose", "tool_transform"])
def test_exact_calibrated_base_and_tool_transform_are_preserved(current_robot, key):
    pose = np.asarray(current_robot[1][key]).copy()
    pose[0, 3] = np.nextafter(pose[0, 3], np.inf)
    with pytest.raises(ValueError, match="model placement|tool identity"):
        build(current_robot, changed={key: pose.tolist()})


def test_current_feedback_cannot_supply_its_own_initial_validity_assumption(current_robot):
    with pytest.raises(ValueError, match="valid initial-model assumption"):
        build(current_robot, initialization_source="")
# END actual state/model validation never creates observed free cells.
