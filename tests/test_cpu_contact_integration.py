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


# BEGIN share the exact native model with the formal planner/contact probe.
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
from contact_fixture import native_model_text
_XML, _URDF = native_model_text()
# END the original CPU test retains its 0.1 m square pad.


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
    # BEGIN direction evidence uses the calibrated physical mechanism channel.
    # This native fixture has exactly one prismatic arm coordinate along world
    # z; its measured q/dq are therefore the actual carrier coordinate/rate.
    # The observer's acquisition interval spans actual native samples and is
    # deliberately independent of the commanded goal and progress variable.
    direction_samples = []
    # END no object direction is inferred from an unconfirmed grasp or FK alone.
    pad_id, slab_id = model.geom("pad").id, model.geom("target_slab").id
    for index in range(400):
        # A deliberately specified control reference, not planner output. The
        # controller cannot force the actual pad through the physical slab.
        goal = -.026*min((index+1)/150., 1.)
        # BEGIN supply the actual derivatives of this declared linear probe.
        # The right-hand sample at the ramp/hold corner has zero speed. This
        # diagnostic has a velocity jump and makes no C1 planning claim.
        speed = -.026/(150*backend.timestep) if index+1 < 150 else 0.
        backend.apply_control(np.asarray([goal]), np.asarray([speed]),
                              np.zeros(1), np.asarray([.01, -.01]))
        # END measured contact still cannot grant runtime permission.
        backend.step()
        measured = backend.proprioception()
        direction_samples.append({"timestamp_s": float(measured["timestamp_s"]),
            "coordinate": float(measured["q"][0]), "rate": float(measured["dq"][0])})
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
    # BEGIN actual displacement/rate can refute a perfectly advanced command.
    assert min(row["rate"] for row in direction_samples) < -.001
    actual_net = -(direction_samples[-1]["coordinate"]-float(initial["q"][0]))
    assert actual_net < .025 < .026  # The commanded travel is not measured travel.
    direction_fact = {"identity": [0, 0], "value": "T" if actual_net >= .025 else "F",
        "source": "Calibrated world-z prismatic q sensor in the actual native mechanism",
        "acquisition_interval_s": [float(initial["timestamp_s"]), direction_samples[-1]["timestamp_s"]],
        "scope_complete": True, "measured_signed_displacement": actual_net,
        "observed_sample_count": len(direction_samples), "sampling_period_s": backend.timestep}
    assert direction_fact["value"] == "F"
    assert abs(direction_fact["acquisition_interval_s"][1]-direction_fact["acquisition_interval_s"][0]-.8) < 1e-10
    record_property("actual_direction_evidence", direction_fact)
    # END physical contact and source timing cannot turn this direction F into T.
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
    # BEGIN account for every completed native step, including the separate hold.
    # Record before checking the stop verdict: a rejected stop still consumes
    # its actual physics steps, while mj_forward queries do not advance time.
    record_property("native_physics_steps", 400+held["physical_steps"])
    # END the executed hold count, not its maximum budget, enters the ledger.
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


# BEGIN a finger position fault must not disable bounded native arm stopping.
def test_native_arm_stop_keeps_the_original_finger_position_fault_visible(record_property):
    """Three scalar coordinates and at most 250 CPU steps; no task execution.

    The declared initial fault places one finger just beyond its native range.
    Stopping may change the state only through native integration. It does not
    certify the failed motion or replace its invalid finger q by a legal q.
    """
    from robosuite.utils.binding_utils import MjSim

    model = mujoco.MjModel.from_xml_string('''
      <mujoco><option timestep=".002" gravity="0 0 0"/>
        <default><geom contype="0" conaffinity="0"/></default>
        <worldbody><body name="base"><body name="arm">
          <joint name="arm_q" type="hinge" axis="0 0 1" range="-2 2"/>
          <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
          <site name="tcp" pos=".1 0 0"/>
          <body name="left"><joint name="left_q" type="slide" axis="0 1 0" range="0 .04" damping="1"/>
            <geom type="sphere" size=".01" mass=".1"/></body>
          <body name="right"><joint name="right_q" type="slide" axis="0 1 0" range="-.04 0" damping="1"/>
            <geom type="sphere" size=".01" mass=".1"/></body>
        </body></body></worldbody>
        <actuator><motor name="arm_motor" joint="arm_q" ctrllimited="true" ctrlrange="-10 10"/>
          <position name="left_servo" joint="left_q" kp="1000" ctrllimited="true" ctrlrange="0 .04"/>
          <position name="right_servo" joint="right_q" kp="1000" ctrllimited="true" ctrlrange="-.04 0"/>
        </actuator>
      </mujoco>''')
    sim = MjSim(model)
    sim.data._data.qpos[:] = [.02, .0400003, -.04]
    sim.data._data.qvel[:] = [.4, 0., 0.]
    sim.data._data.ctrl[:] = [0., .04, -.04]
    mujoco.mj_forward(model, sim.data._data)
    backend = RobosuiteExecutionBackend(sim, base_link="base", tcp_site="tcp",
        arm_joint_names=("arm_q",), finger_joint_names=("left_q", "right_q"),
        arm_kp=1000., arm_kd=200., velocity_limits={"arm_q": 1., "left_q": 1., "right_q": 1.},
        source="Explicit isolated native fault-stop fixture")
    initial = backend.proprioception()
    result = backend.abort_hold(initial, np.array([.04, -.04]), expected_sim_time=initial["sim_time"],
        stop_velocity=np.array([.01, .001, .001]), max_steps=250)
    # BEGIN retain the exact completed native stop count independently of success.
    # This fixture has no preceding trajectory steps; setup mj_forward calls
    # only refresh derived state and must not inflate physical consumption.
    record_property("native_physics_steps", result["physical_steps"])
    # END the existing isolated_native_stop_steps diagnostic remains unchanged.
    assert initial["q"][1] > backend.joint_ranges[1, 1]
    assert initial["dq"][0] == .4
    assert result["prepared"] and result["native_ctrl_written"]
    assert 1 < result["physical_steps"] <= 250
    assert result["physical_stop_confirmed"]
    assert np.all(np.abs(result["samples"][-1]["dq"]) <= [.01, .001, .001])
    fault = result["joint_position_violation_samples"][0]
    assert fault["phase"] == "entry"
    assert fault["violations"][0]["joint_name"] == "left_q"
    assert fault["violations"][0]["measured"] == initial["q"][1]
    assert fault["violations"][0]["upper"] == .04
    assert all(control["actuator_ctrl"][1:] == [.04, -.04] for control in result["controls"])
    record_property("isolated_native_stop_steps", result["physical_steps"])
# END position-domain failure remains distinct from actual velocity convergence.


# BEGIN audit the recorded near-contact coordinates without a physics runtime.
def test_complete_solid_empty_intersection_resolves_coal_numerical_boundary(tmp_path):
    # The source-solid audit owns its own model. No production validator,
    # MuJoCo model, controller or simulation step is needed for these three
    # independently supplied joint configurations.
    urdf = tmp_path / "audit-robot.urdf"
    urdf.write_text(native_model_text(pad_width=.01)[1])
    checker = RobotGeometryChecker(urdf, base_frame="base", geometries=[
        NativePrimitive("pad", "arm", "box", (.01, .01, .1), np.eye(4), "explicit thin audit pad"),
        NativePrimitive("left_pad", "left", "box", (.02, .02, .02), np.eye(4), "explicit audit left finger"),
        NativePrimitive("right_pad", "right", "box", (.02, .02, .02), np.eye(4), "explicit audit right finger")])
    rule = ContactRule(rule_id="press", actor="robot", target_owner="target", interval=(0., 1.),
        sphere_indices=(0,), material_id="press_face",
        feature_points=[[-.005,-.005,0.],[.005,-.005,0.],[.005,.005,0.],[-.005,.005,0.]],
        feature_normal=[0.,0.,-1.], patch_origin=[0.,0.,0.], patch_axes=np.eye(3), patch_half_size=[.1,.1],
        gap_bounds=(-.0025,.0025), normal_tolerance_rad=.01, modeling_allowance_m=.003,
        geometry_error_m=.003, minimum_thickness_m=.1, source="unchanged formal probe finite contact permission")

    def check(arm_q, *, permit):
        binding = _scene(rule)
        if not permit:
            binding["contacts"] = []
        checker.bind_scene(binding)
        state = checker.query(joint_names=("arm_q", "left_q", "right_q"),
            joint_positions=[arm_q, .01, -.01], T_reference_base=np.eye(4), pairs=(), frames=("tcp",))
        return checker.query_scene(state, tool_pose=np.asarray(state["frame_poses"]["tcp"]), progress=1.,
            finger_positions=[.01, -.01], state_kind="actual")

    # This exact q was rejected after native step 434 in the formal planner
    # probe. The actual face has +0.178556 micrometres clearance, while Coal
    # reports a zero-distance numerical boundary. Complete convex halfspaces
    # resolve it as disjoint without any task permission or wider tolerance.
    separated = check(-.019999821444431916, permit=False)
    assert separated["passed"]
    pad = next(row for row in separated["pairs"] if row["actor_geometry"] == "pad")
    assert pad["signed_distance_m"] == 0.
    assert pad["relation"] == "numerical_boundary_with_proven_empty_intersection"
    assert pad["boundary_intersection_check"]["intersection_empty"]
    assert pad["permission"] is None

    # A true overlap still needs the complete local material permission; an
    # empty-intersection result cannot become a whole-pair contact exception.
    penetrating = check(-.0201, permit=False)
    assert not penetrating["passed"] and penetrating["violations"]
    permitted = check(-.0201, permit=True)
    assert permitted["passed"]
    pad = next(row for row in permitted["pairs"] if row["actor_geometry"] == "pad")
    assert pad["permission"] == "press"
# END physical separation and allowed local overlap remain separate query results.
