"""Small real CPU contact integration: control, actual feedback and Coal scope.

This fixture runs native MuJoCo physics without rendering or GPU allocation.
It does not invoke the native trajectory planner or fabricate planning proof.
The supplied static target geometry is independent test input; native contacts
are evaluator evidence only, never a source of runtime contact permission.
"""

# BEGIN real contact complements mocked execution sequencing regressions.
from dataclasses import asdict
import resource

import mujoco
import numpy as np

from manipulation.planning_scene import ContactRule
from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker
from manipulation.robosuite_execution import RobosuiteExecutionBackend


_XML = """<mujoco>
  <compiler angle="radian"/>
  <option timestep=".002" gravity="0 0 0"/>
  <default><joint damping="1"/><geom friction=".8 .01 .001" solref=".004 1"/></default>
  <worldbody>
    <geom name="target_slab" type="box" pos="0 0 -.05" size=".3 .3 .05"/>
    <body name="base">
      <body name="arm" pos="0 0 .07">
        <joint name="arm_q" type="slide" axis="0 0 1" range="-.04 .04"/>
        <geom name="pad" type="box" size=".05 .05 .05" mass=".2"/>
        <site name="tcp" pos="0 0 -.05"/>
        <body name="left" pos="0 .15 .1">
          <joint name="left_q" type="slide" axis="0 1 0" range="0 .04"/>
          <geom name="left_pad" type="box" size=".01 .01 .01" mass=".02"/>
        </body>
        <body name="right" pos="0 -.15 .1">
          <joint name="right_q" type="slide" axis="0 1 0" range="-.04 0"/>
          <geom name="right_pad" type="box" size=".01 .01 .01" mass=".02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="arm_motor" joint="arm_q" ctrllimited="true" ctrlrange="-10 10"/>
    <position name="left_servo" joint="left_q" kp="100" ctrllimited="true" ctrlrange="0 .04"/>
    <position name="right_servo" joint="right_q" kp="100" ctrllimited="true" ctrlrange="-.04 0"/>
  </actuator>
</mujoco>"""

_URDF = """<robot name="cpu_press_fixture">
  <link name="base"/><link name="arm"/><link name="left"/><link name="right"/><link name="tcp"/>
  <joint name="arm_q" type="prismatic"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 .07"/><axis xyz="0 0 1"/>
    <limit lower="-.04" upper=".04" effort="10" velocity="1"/></joint>
  <joint name="left_q" type="prismatic"><parent link="arm"/><child link="left"/>
    <origin xyz="0 .15 .1"/><axis xyz="0 1 0"/>
    <limit lower="0" upper=".04" effort="10" velocity="1"/></joint>
  <joint name="right_q" type="prismatic"><parent link="arm"/><child link="right"/>
    <origin xyz="0 -.15 .1"/><axis xyz="0 1 0"/>
    <limit lower="-.04" upper="0" effort="10" velocity="1"/></joint>
  <joint name="tcp_fixed" type="fixed"><parent link="arm"/><child link="tcp"/>
    <origin xyz="0 0 -.05"/></joint>
</robot>"""


def _scene(rule):
    identity = np.eye(4).tolist()
    return {
        "reference_frame": "world", "native_geometry_materials": {"pad": "press_face"},
        "contacts": [asdict(rule)], "background": {"geometry": {"solids": [], "query_gaps": []}},
        "owners": [{"owner_id": "target", "source": "explicit static fixture target",
            "geometry_complete": True, "moving": False, "attachments": [], "native_geometry_names": [],
            "pose_interval": [0., 1.], "pose_path": {"progress": [0., 1.], "poses": [identity, identity],
                "source": "explicit static target transform", "attachment": identity,
                "coordinates": None, "motion_model": None},
            "geometry": {"solids": [{"name": "slab", "kind": "cuboid", "dimensions": [.6, .6, .1],
                "pose": [0., 0., -.05, 1., 0., 0., 0.], "source": "explicit fixture target dimensions"}],
                "query_gaps": []}}]}


def test_native_contact_is_permitted_locally_and_actual_state_remains_distinct(tmp_path, record_property):
    # BEGIN attribute resource growth to this fixture within a shared process.
    # ru_maxrss is a process lifetime high-water mark: prior tests may have
    # already exceeded this small fixture's budget. Record both measurements
    # and limit only the newly raised peak, retaining the machine-wide cap.
    baseline_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    # END attribute resource growth to this fixture within a shared process.
    from robosuite.utils.binding_utils import MjSim

    model = mujoco.MjModel.from_xml_string(_XML)
    sim = MjSim(model)
    sim.data._data.qpos[:] = [0., .01, -.01]
    sim.data._data.ctrl[:] = [0., .01, -.01]
    mujoco.mj_forward(model, sim.data._data)
    backend = RobosuiteExecutionBackend(sim, base_link="base", tcp_site="tcp",
        arm_joint_names=("arm_q",), finger_joint_names=("left_q", "right_q"),
        arm_kp=500., arm_kd=20., velocity_limits={"arm_q": 1., "left_q": 1., "right_q": 1.},
        source="CPU native contact fixture actual robot feedback")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(_URDF)
    geometry = RobotGeometryChecker(urdf, base_frame="base", geometries=[
        NativePrimitive("pad", "arm", "box", (.1, .1, .1), np.eye(4), "native pad dimensions"),
        NativePrimitive("left_pad", "left", "box", (.02, .02, .02), np.eye(4), "native left finger"),
        NativePrimitive("right_pad", "right", "box", (.02, .02, .02), np.eye(4), "native right finger")])
    contact = ContactRule(rule_id="press", actor="robot", target_owner="target", interval=(0., 1.),
        sphere_indices=(0,), material_id="press_face",
        feature_points=[[-.05, -.05, 0.], [.05, -.05, 0.], [.05, .05, 0.], [-.05, .05, 0.]],
        feature_normal=[0., 0., -1.], patch_origin=[0., 0., 0.], patch_axes=np.eye(3), patch_half_size=[.1, .1],
        gap_bounds=(-.002, .002), normal_tolerance_rad=.01, modeling_allowance_m=.003,
        geometry_error_m=.003, minimum_thickness_m=.1, source="independent finite pressing face and patch")
    geometry.bind_scene(_scene(contact))
    initial = backend.proprioception()
    peak_force, saw_contact = 0., False
    pad_id, slab_id = model.geom("pad").id, model.geom("target_slab").id
    for index in range(400):
        # A deliberately specified control reference, not planner output. The
        # controller cannot force the actual pad through the physical slab.
        goal = -.026*min((index+1)/150., 1.)
        backend.apply_control(np.asarray([goal]), np.asarray([.01, -.01]))
        backend.step()
        for contact_index in range(backend.native_data.ncon):
            actual_contact = backend.native_data.contact[contact_index]
            if {int(actual_contact.geom1), int(actual_contact.geom2)} != {pad_id, slab_id}:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, backend.native_data, contact_index, force)
            peak_force = max(peak_force, abs(float(force[0])))
            saw_contact = True
    actual = backend.proprioception()
    assert saw_contact and peak_force > .1
    assert actual["sim_time"] > initial["sim_time"]+.79
    assert actual["q"][0] > -.023 and abs(actual["q"][0]+.026) > .003
    state = geometry.query(joint_names=actual["joint_names"], joint_positions=actual["q"],
        T_reference_base=actual["base_pose"], pairs=(), frames=("tcp",))
    np.testing.assert_allclose(state["frame_poses"]["tcp"], actual["tool_pose"], atol=1e-10)
    accepted = geometry.query_scene(state, tool_pose=actual["tool_pose"], progress=1.,
        finger_positions=actual["q"][1:], state_kind="actual")
    assert accepted["passed"] and accepted["observed_scene_checked"]
    pad = next(row for row in accepted["pairs"] if row["actor_geometry"] == "pad")
    assert pad["permission"] == "press" and pad["contacts"]
    assert pad["contact_region_checks"][0]["contained"]
    # The physical contact itself grants no owner-wide exception. Removing
    # the separate material permission must reject exactly the same feedback.
    without_permission = _scene(contact)
    without_permission["contacts"] = []
    geometry.bind_scene(without_permission)
    rejected = geometry.query_scene(state, tool_pose=actual["tool_pose"], progress=1.,
        finger_positions=actual["q"][1:], state_kind="actual")
    assert not rejected["passed"] and rejected["violations"]
    held = backend.abort_hold(actual, np.asarray([.01, -.01]), expected_sim_time=actual["sim_time"],
        stop_velocity=np.asarray([.01, .01, .01]), max_steps=50)
    assert held["native_ctrl_written"] and held["physical_steps"] > 0
    assert held["physical_stop_confirmed"]
    assert all(abs(value) <= .01 for value in held["samples"][-1]["dq"])
    # Bound this fixture explicitly: three scalar robot joints, one static
    # target, 400 contact steps, no camera/rendering/native trajectory solver.
    # BEGIN fixture growth and total process limit remain separately visible.
    final_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    record_property("baseline_peak_mib", baseline_peak_mib)
    record_property("final_peak_mib", final_peak_mib)
    record_property("peak_increment_mib", final_peak_mib-baseline_peak_mib)
    assert final_peak_mib-baseline_peak_mib < 1024
    assert final_peak_mib < 16*1024
    # END fixture growth and total process limit remain separately visible.
# END real contact complements mocked execution sequencing regressions.
