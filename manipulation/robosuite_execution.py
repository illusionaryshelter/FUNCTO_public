"""Native MuJoCo control through installed robosuite joint/gripper controllers.

Bind an already initialized MjSim without loading a recording or resetting its
state. Absolute arm q/dq/ddq from one trajectory sample and independent native
finger positions go to mature controllers once per native physics step. The
arm adapter adds analytic trajectory feedforward using the controller's own
native mass matrix; no second interpolator or full-scene inverse dynamics runs.
Only actuator ctrl is written after binding; FK uses separate MjData.
"""

# BEGIN native robosuite execution backend without recording dependencies.
from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import mujoco
import numpy as np

from .execution import _finite_array


class RobosuiteExecutionBackend:
    """One named robot control binding in an existing native simulation."""

    def __init__(self, sim, *, base_link, tcp_site, arm_joint_names, finger_joint_names,
                 arm_kp, arm_kd, velocity_limits, source):
        from robosuite.controllers.parts.generic.joint_pos import JointPositionController
        from robosuite.controllers.parts.gripper.simple_grip import SimpleGripController

        gains = np.asarray([arm_kp, arm_kd], dtype=float)
        if gains.shape != (2,) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("Joint-controller gains must be explicit finite positive scalars")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("A native feedback binding requires its explicit source")
        if any(isinstance(names, str) or not names or any(not isinstance(name, str) or not name.strip() for name in names)
               for names in (arm_joint_names, finger_joint_names)):
            raise ValueError("Native arm and finger joints require explicit ordered names")
        self._sim = sim
        self.native_model, self.native_data = sim.model._model, sim.data._data
        self._trial_id = str(uuid4())
        self.arm_joint_names, self.finger_joint_names = tuple(arm_joint_names), tuple(finger_joint_names)
        self.joint_names = self.arm_joint_names + self.finger_joint_names
        if not self.arm_joint_names or not self.finger_joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Arm and finger feedback names must be nonempty and disjoint")
        # BEGIN preserve the explicit named hardware velocity limits.
        if not isinstance(velocity_limits, dict) or set(velocity_limits) != set(self.joint_names):
            raise ValueError("Control binding requires hardware velocity limits for every named arm/finger joint")
        speed = _finite_array([velocity_limits[name] for name in self.joint_names],
                              (len(self.joint_names),), "Named hardware velocity limits")
        if np.any(speed <= 0):
            raise ValueError("Hardware velocity limits must be positive")
        self._velocity_limits = dict(zip(self.joint_names, speed.tolist()))
        # END preserve the explicit named hardware velocity limits.
        self._base_id = self.native_model.body(base_link).id
        self._site_id = self.native_model.site(tcp_site).id
        self._joint_ids = np.asarray([self.native_model.joint(name).id for name in self.joint_names])
        if not self.native_model.jnt_limited[self._joint_ids].all():
            raise ValueError("Every controlled scalar joint requires native position bounds")
        if not np.isin(self.native_model.jnt_type[self._joint_ids],
                       [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]).all():
            raise ValueError("Only explicit scalar robot feedback is supported")
        self._qpos_ids = self.native_model.jnt_qposadr[self._joint_ids].copy()
        self._qvel_ids = self.native_model.jnt_dofadr[self._joint_ids].copy()
        actuator_ids = []
        for joint_id in self._joint_ids:
            matches = np.flatnonzero((self.native_model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                     & (self.native_model.actuator_trnid[:, 0] == joint_id))
            if len(matches) != 1:
                raise ValueError("Every controlled scalar joint must have exactly one native joint actuator")
            actuator_ids.append(int(matches[0]))
        self._actuator_ids = np.asarray(actuator_ids)
        # BEGIN explicit native command identity, independent from measured q.
        self._actuator_names = tuple(self.native_model.actuator(index).name for index in actuator_ids)
        if any(not name for name in self._actuator_names):
            raise ValueError("Every controlled native actuator must have an explicit name")
        self._feedback_source = source
        # END explicit native command identity, independent from measured q.
        model, narm = self.native_model, len(self.arm_joint_names)
        self._arm_actuators, self._finger_actuators = self._actuator_ids[:narm], self._actuator_ids[narm:]
        if (not model.actuator_ctrllimited[self._actuator_ids].all()
                or not np.isfinite(model.actuator_ctrlrange[self._actuator_ids]).all()):
            raise ValueError("Explicit finite native actuator control bounds are required")
        # The mature arm controller emits torques, so its native transmission
        # must be unit-gear fixed-gain motors. Gripper GRIP controls retain the
        # original position-actuator gains and forces from the scene XML.
        if (np.any(model.actuator_gear[self._arm_actuators, 0] != 1)
                or np.any(model.actuator_gear[self._arm_actuators, 1:] != 0)
                or np.any(model.actuator_gaintype[self._arm_actuators] != mujoco.mjtGain.mjGAIN_FIXED)
                or np.any(model.actuator_dyntype[self._arm_actuators] != mujoco.mjtDyn.mjDYN_NONE)
                or np.any(model.actuator_gainprm[self._arm_actuators, 0] != 1)
                or np.any(model.actuator_biastype[self._arm_actuators] != mujoco.mjtBias.mjBIAS_NONE)):
            raise ValueError("Joint torque controller requires native unit-gain torque motors")
        # BEGIN physical finger targets through the existing unscaled GRIP.
        # A unit-gear native position actuator interprets ctrl in the named
        # joint's physical units. A velocity/motor actuator must not be
        # silently treated as a position servo by this generic interface.
        finger_ids = self._finger_actuators
        finger_gain = model.actuator_gainprm[finger_ids, 0]
        if (np.any(model.actuator_gear[finger_ids, 0] != 1)
                or np.any(model.actuator_gear[finger_ids, 1:] != 0)
                or np.any(model.actuator_gaintype[finger_ids] != mujoco.mjtGain.mjGAIN_FIXED)
                or np.any(model.actuator_dyntype[finger_ids] != mujoco.mjtDyn.mjDYN_NONE)
                or np.any(model.actuator_biastype[finger_ids] != mujoco.mjtBias.mjBIAS_AFFINE)
                or np.any(finger_gain <= 0)
                or np.any(model.actuator_biasprm[finger_ids, 0] != 0)
                or np.any(model.actuator_biasprm[finger_ids, 1] != -finger_gain)):
            raise ValueError("Explicit finger position targets require native unit-gear position actuators")
        self._dt = float(model.opt.timestep)
        if not np.isfinite(self._dt) or self._dt <= 0:
            raise ValueError("Native physics timestep must be finite and positive")
        # END physical finger targets through the existing unscaled GRIP.

        def indexes(start, stop):
            return {"joints": self._joint_ids[start:stop].tolist(),
                    "qpos": self._qpos_ids[start:stop].tolist(),
                    "qvel": self._qvel_ids[start:stop].tolist(),
                    "actuators": self._actuator_ids[start:stop].tolist()}

        # BEGIN native mass/bias compensation is part of the trajectory law.
        bounds = model.actuator_ctrlrange[self._arm_actuators]
        self._arm_controller = JointPositionController(self._sim, indexes(0, narm),
            actuator_range=(bounds[:, 0], bounds[:, 1]), ref_name=tcp_site,
            input_type="absolute", kp=float(arm_kp), kd=float(arm_kd), damping_ratio=1,
            policy_freq=1/self._dt, lite_physics=False, interpolator=None,
            use_torque_compensation=True)
        # END derivative feedforward uses this same compensated controller.
        bounds = model.actuator_ctrlrange[self._finger_actuators]
        # BEGIN resume raw native command state without binary action memory.
        # Contact can keep actual q away from ctrl. Initialization therefore
        # preserves restored ctrl, and never substitutes measured positions.
        restored_ctrl = self.native_data.ctrl[self._finger_actuators].copy()
        widths = bounds[:, 1] - bounds[:, 0]
        if (np.any(widths <= 0) or not np.isfinite(restored_ctrl).all()
                or np.any(restored_ctrl < bounds[:, 0]) or np.any(restored_ctrl > bounds[:, 1])):
            raise ValueError("Restored gripper ctrl must be inside finite native ranges")
        self._grip_controller = SimpleGripController(self._sim, indexes(narm, len(self.joint_names)),
            actuator_range=(bounds[:, 0], bounds[:, 1]), policy_freq=1/self._dt,
            use_action_scaling=False, interpolator=None)
        self._grip_controller.set_goal(restored_ctrl.copy())
        if not np.array_equal(self._grip_controller.run_controller(), restored_ctrl):
            raise ValueError("Unscaled GRIP must reproduce restored native position controls exactly")
        # END resume raw native command state without binary action memory.
        # BEGIN isolated robot FK for physical TCP tracking checks.
        self._prepare_reference_fk()
        # END isolated robot FK for physical TCP tracking checks.
        self._description = {
            "backend": "robosuite native MuJoCo trajectory-state and gripper controllers",
            "source": source, "trial_id": self._trial_id,
            "joint_names": list(self.joint_names), "arm_joint_names": list(self.arm_joint_names),
            "finger_joint_names": list(self.finger_joint_names),
            "base_link": base_link, "reference_frame": self.reference_frame,
            "tool_link": self.tool_link, "tool_transform": self.tool_transform.tolist(),
            "actuator_names": list(self.actuator_names), "native_timestep_s": self.timestep,
            "actuator_ctrlrange": model.actuator_ctrlrange[self._actuator_ids].tolist(),
            "actuator_forcelimited": model.actuator_forcelimited[self._actuator_ids].astype(bool).tolist(),
            "actuator_forcerange": model.actuator_forcerange[self._actuator_ids].tolist(),
            # BEGIN bind the complete trajectory-control contract by joint name.
            "arm_controller": "robosuite JointPositionController plus same-curve dq/ddq feedforward, native torque actuators",
            "arm_kp": float(arm_kp), "arm_kd": float(arm_kd),
            "effective_arm_kp": self._arm_controller.kp.tolist(),
            "effective_arm_kd": self._arm_controller.kd.tolist(),
            "arm_interpolator": None, "desired_velocity_feedforward": True,
            "desired_acceleration_feedforward": True,
            "controller_policy": {
                "mode": "joint_trajectory_q_dq_ddq",
                "arm_joint_names": list(self.arm_joint_names),
                "kp": self._arm_controller.kp.tolist(), "kd": self._arm_controller.kd.tolist(),
                "native_timestep_s": self.timestep,
                "reference_derivatives": ["q", "dq", "ddq"],
                "sampling_phase": "clamped_step_endpoint",
                "acceleration_knot_side": "right_internal_last_left"},
            # END this policy cannot be inferred from a q-only authorization.
            "gripper_controller": "robosuite SimpleGripController, use_action_scaling=False",
            "gripper_command_frequency_hz": 1/self.timestep,
            "initial_finger_position_ctrl": restored_ctrl.tolist(),
            "gripper_initialization": "Current native ctrl is preserved independently of measured finger q"}

    @property
    def timestep(self):
        return self._dt

    @property
    def sim_time(self):
        return float(self.native_data.time)

    @property
    def trial_id(self):
        return self._trial_id

    @property
    def feedback_source(self):
        return self._feedback_source

    @property
    def feedback_id(self):
        return f"{self._trial_id}@native_time={self.sim_time:.17g}"

    @property
    def reference_frame(self):
        # Native site_xpos/site_xmat and body xpos/xmat are in MuJoCo world.
        return "world"

    @property
    def tool_link(self):
        # The observed tool frame is this exact native site. No runner may
        # relabel another frame or attach a second, unapplied TCP transform.
        return self.native_model.site(self._site_id).name

    @property
    def tool_transform(self):
        return np.eye(4)

    @property
    def actuator_names(self):
        return self._actuator_names

    @property
    def joint_ranges(self):
        return self.native_model.jnt_range[self._joint_ids].copy()

    @property
    def velocity_limits(self):
        return dict(self._velocity_limits)

    @property
    def finger_ctrlrange(self):
        return self.native_model.actuator_ctrlrange[self._finger_actuators].copy()

    @property
    def finger_control(self):
        """Read the current native command, not measured finger position."""
        return self.native_data.ctrl[self._finger_actuators].copy()

    @property
    def actuator_control(self):
        return self.native_data.ctrl[self._actuator_ids].copy()

    def describe(self):
        return deepcopy(self._description)

    # BEGIN mandatory trajectory derivatives through the native arm controller.
    def apply_control(self, arm_q, arm_dq, arm_ddq, finger_positions):
        """Apply one named q/dq/ddq reference and native finger positions.

        Every arm vector follows ``arm_joint_names`` from this binding. The
        executor samples all three at its declared clamped step endpoint;
        none is substituted by measured velocity or finite differences here.
        Installed robosuite supplies the position/damping torque, mass matrix
        and bias at the same current state. Adding M*(Kd*dq_ref + ddq_ref)
        yields M*(ddq_ref + Kp*(q_ref-q) + Kd*(dq_ref-dq)) + bias, without
        changing gains or canceling unknown contact/passive forces.

        This method writes only actuator ctrl and does not integrate physics.
        Its raw torque components and references belong to the same call;
        saturation is applied only to their total, never to each component.
        """
        narm = len(self.arm_joint_names)
        arm_q = _finite_array(arm_q, (narm,), "Named arm reference positions")
        arm_dq = _finite_array(arm_dq, (narm,), "Named arm reference velocities")
        arm_ddq = _finite_array(arm_ddq, (narm,), "Named arm reference accelerations")
        finger_positions = _finite_array(finger_positions, (len(self.finger_joint_names),), "Native finger position goals")
        bounds = self.finger_ctrlrange
        if np.any(finger_positions < bounds[:, 0]) or np.any(finger_positions > bounds[:, 1]):
            raise ValueError("Finger position goals must remain inside native control bounds")
        self._grip_controller.set_goal(finger_positions.copy())
        self._arm_controller.set_goal(arm_q.copy())
        controller = self._arm_controller
        position_controller_torque = _finite_array(controller.run_controller(), (narm,), "Native position-controller torque")
        # run_controller has refreshed this complete state and only marks it
        # for the next update; reading its arrays performs no extra simulation
        # or forward/inverse dynamics call. Their order is our arm joint order.
        mass = _finite_array(controller.mass_matrix, (narm, narm), "Native arm mass matrix")
        kp = _finite_array(controller.kp, (narm,), "Native named arm proportional gains")
        kd = _finite_array(controller.kd, (narm,), "Native named arm damping gains")
        actual_q = _finite_array(controller.joint_pos, (narm,), "Controller current arm positions")
        actual_dq = _finite_array(controller.joint_vel, (narm,), "Controller current arm velocities")
        bias = _finite_array(controller.torque_compensation, (narm,), "Native arm bias torque")
        position_feedback = _finite_array(mass @ (kp * (arm_q-actual_q)), (narm,), "Arm position-feedback torque")
        velocity_feedback = _finite_array(mass @ (-kd * actual_dq), (narm,), "Arm velocity-feedback torque")
        velocity_feedforward = _finite_array(mass @ (kd * arm_dq), (narm,), "Arm velocity-feedforward torque")
        acceleration_feedforward = _finite_array(mass @ arm_ddq, (narm,), "Arm acceleration-feedforward torque")
        torque = _finite_array(position_controller_torque + velocity_feedforward + acceleration_feedforward,
                               (narm,), "Native trajectory-controller torque")
        bounds = self.native_model.actuator_ctrlrange[self._arm_actuators]
        bounded = np.clip(torque, bounds[:, 0], bounds[:, 1])
        finger_ctrl = np.asarray(self._grip_controller.run_controller(), dtype=float)
        if not np.array_equal(finger_ctrl, finger_positions):
            raise RuntimeError("Unscaled GRIP changed an explicit native finger position target")
        self.native_data.ctrl[self._arm_actuators] = bounded
        self.native_data.ctrl[self._finger_actuators] = finger_ctrl
        # BEGIN arm control clipping is separate from native finger force limits.
        return {"actuator_ctrl": self.actuator_control,
                "requested_arm_torque": torque.copy(), "arm_torque_saturated": bounded != torque,
                "arm_control_saturated_coordinates": int(np.count_nonzero(bounded != torque)),
                "reference_arm_q": arm_q.copy(), "reference_arm_dq": arm_dq.copy(),
                "reference_arm_ddq": arm_ddq.copy(),
                "arm_position_feedback_torque": position_feedback.copy(),
                "arm_velocity_feedback_torque": velocity_feedback.copy(),
                "arm_velocity_feedforward_torque": velocity_feedforward.copy(),
                "arm_acceleration_feedforward_torque": acceleration_feedforward.copy(),
                "arm_bias_torque": bias.copy()}
        # END count only the arm controller's clipped torque coordinates.
    # END no derivative-free fallback or mutation of physical q/dq/qacc.

    def step(self):
        """Integrate exactly one native step and refresh derived feedback."""
        self._sim.step()
        # mj_step leaves derived positions at its integration stage. Forward
        # refreshes native robot feedback and evaluator contacts without a step.
        mujoco.mj_forward(self.native_model, self.native_data)


    def _prepare_reference_fk(self):
        """Require a fixed, controlled TCP chain before creating scratch data.

        This arm-curve executor has no commanded finger *path*. A TCP that
        depends on fingers cannot use measured fingers to hide its tracking
        error, so it requires a separately defined reference and is rejected
        here. Uncontrolled or mocap-driven ancestors are likewise not silently
        taken from an environment snapshot as a moving robot-base reference.
        """
        model = self.native_model
        body, ancestors = int(model.site_bodyid[self._site_id]), set()
        while body:
            ancestors.add(body)
            body = int(model.body_parentid[body])
        ancestors.add(0)
        if self._base_id not in ancestors:
            raise ValueError("The executed TCP must descend from the calibrated robot base")
        chain = {int(index) for index, owner in enumerate(model.jnt_bodyid) if int(owner) in ancestors}
        arm_ids = set(self._joint_ids[:len(self.arm_joint_names)].tolist())
        if not chain <= arm_ids or any(model.body_mocapid[index] >= 0 for index in ancestors):
            raise ValueError("TCP reference needs a fixed base and only the explicitly controlled arm joints")
        self._reference_data = mujoco.MjData(model)
        self._reference_data.qpos[self._qpos_ids] = self.native_data.qpos[self._qpos_ids]


    def reference_tool_pose(self, arm_q):
        """Compute exact TCP FK of a sampled arm cubic without stepping it."""
        values = _finite_array(arm_q, (len(self.arm_joint_names),), "Reference arm positions")
        self._reference_data.qpos[self._qpos_ids[:len(values)]] = values
        mujoco.mj_kinematics(self.native_model, self._reference_data)
        pose = np.eye(4)
        pose[:3, :3] = self._reference_data.site_xmat[self._site_id].reshape(3, 3)
        pose[:3, 3] = self._reference_data.site_xpos[self._site_id]
        return pose


    # BEGIN bounded physical hold with actual feedback confirmation.
    def abort_hold(self, feedback, finger_positions, *, expected_sim_time, stop_velocity, max_steps):
        """Apply hold through existing controllers, then measure bounded settling.

        The arm goal is the failed sample's actual q. The last applied finger
        goal is retained, so an event rejected by supervision is never issued
        by fault handling. Hold telemetry is separate from the failed curve.
        No state reset or claimed zero velocity substitutes for feedback.
        """
        result = {"prepared": False, "native_ctrl_written": False, "physical_steps": 0,
                  "physical_stop_confirmed": False, "samples": [], "controls": []}
        try:
            if (tuple(feedback.get("joint_names", ())) != self.joint_names
                    or tuple(feedback.get("arm_joint_names", ())) != self.arm_joint_names
                    or tuple(feedback.get("finger_joint_names", ())) != self.finger_joint_names
                    or feedback.get("trial_id") != self._trial_id
                    or feedback.get("source") != self._feedback_source):
                raise ValueError("Cannot hold from unidentified feedback")
            current_time = float(self.native_data.time)
            clocks = np.asarray([feedback.get("sim_time"), feedback.get("timestamp_s"), current_time], dtype=float)
            if (not np.isfinite(clocks).all() or np.max(np.abs(clocks-expected_sim_time)) > 1e-8
                    or feedback.get("feedback_id") != f"{self._trial_id}@native_time={current_time:.17g}"):
                raise ValueError("Cannot hold from stale or discontinuous native feedback time")
            q = _finite_array(feedback["q"], (len(self.joint_names),), "Hold feedback positions")
            _finite_array(feedback["dq"], q.shape, "Hold feedback velocities")
            limits = _finite_array(stop_velocity, q.shape, "Named physical stop velocities")
            if np.any(limits < 0) or type(max_steps) is not int or not 1 <= max_steps <= 10000:
                raise ValueError("Physical hold requires finite nonnegative criteria and a bounded step count")
            ranges = self.joint_ranges
            if np.any(q < ranges[:, 0]) or np.any(q > ranges[:, 1]):
                raise ValueError("Out-of-range actual coordinates cannot become a new hold goal")
            goal = q[:len(self.arm_joint_names)].copy()
            # BEGIN holds explicitly record their own complete reference.
            result.update(prepared=True, arm_position_goal=goal.tolist(),
                arm_velocity_goal=np.zeros_like(goal).tolist(),
                arm_acceleration_goal=np.zeros_like(goal).tolist(),
                arm_joint_names=list(self.arm_joint_names),
                finger_joint_names=list(self.finger_joint_names),
                actuator_names=list(self.actuator_names),
                finger_position_goal=np.asarray(finger_positions).tolist(), stop_velocity=limits.tolist())
            # END reference derivatives do not overwrite actual measured dq.
            for index in range(max_steps):
                # BEGIN physical hold has explicit zero desired derivatives.
                # Actual nonzero velocity remains in the measured damping
                # term; zeroing the reference does not reset native state.
                control = self.apply_control(goal, np.zeros_like(goal), np.zeros_like(goal), finger_positions)
                # END hold uses the same strict trajectory-state interface.
                result["native_ctrl_written"] = True
                # BEGIN retain hold torque demands, components and saturation.
                # A bounded safety hold is a real control sequence too; its
                # telemetry must not collapse back to the old q-only record.
                result["controls"].append({"control_sim_time": self.sim_time,
                    **{key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in control.items()}})
                # END each row precedes exactly its following native hold step.
                self.step()
                actual = self.proprioception()
                result["physical_steps"] = index+1
                result["samples"].append(actual)
                measured_q = _finite_array(actual["q"], q.shape, "Hold actual positions")
                measured_dq = _finite_array(actual["dq"], q.shape, "Hold actual velocities")
                if np.any(measured_q < ranges[:, 0]) or np.any(measured_q > ranges[:, 1]):
                    raise ValueError("Physical joint range invalidated during hold")
                if abs(float(actual["sim_time"])-(current_time+(index+1)*self.timestep)) > 1e-8:
                    raise ValueError("Native clock became discontinuous during hold")
                if np.all(np.abs(measured_dq) <= limits):
                    result["physical_stop_confirmed"] = True
                    break
            if not result["physical_stop_confirmed"]:
                result["reason"] = "Hold budget exhausted before named stop velocity criteria were observed"
        except Exception as error:
            result["reason"] = f"{type(error).__name__}: {error}"
        return result
    # END physical hold result preserves unconfirmed and failed stops.


    def proprioception(self):
        """Return only explicitly named robot feedback and its native TCP."""
        base = np.eye(4)
        base[:3, :3] = self.native_data.xmat[self._base_id].reshape(3, 3)
        base[:3, 3] = self.native_data.xpos[self._base_id]
        tool = np.eye(4)
        tool[:3, :3] = self.native_data.site_xmat[self._site_id].reshape(3, 3)
        tool[:3, 3] = self.native_data.site_xpos[self._site_id]
        # BEGIN named actual state and separately named native command state.
        timestamp = float(self.native_data.time)
        return {"joint_names": self.joint_names,
                "reference_frame": self.reference_frame, "tool_link": self.tool_link,
                "tool_transform": self.tool_transform,
                "arm_joint_names": self.arm_joint_names, "finger_joint_names": self.finger_joint_names,
                "trial_id": self._trial_id, "feedback_id": f"{self._trial_id}@native_time={timestamp:.17g}",
                "source": self._feedback_source, "timestamp_s": timestamp,
                "q": self.native_data.qpos[self._qpos_ids].copy(),
                "dq": self.native_data.qvel[self._qvel_ids].copy(),
                # BEGIN preserve native dynamics in their independent named orders.
                # qacc is the acceleration from the refreshed native forward
                # state, not a finite difference of logged velocities. Force
                # belongs to actuators; position-servo ctrl is not a force.
                "qacc": self.native_data.qacc[self._qvel_ids].copy(),
                "actuator_names": self.actuator_names,
                "actuator_force": self.native_data.actuator_force[self._actuator_ids].copy(),
                # END raw dynamics remain separate from commands and limits.
                "base_pose": base, "tool_pose": tool, "sim_time": timestamp,
                "actuator_ctrl": {"joint_names": self.joint_names,
                    "actuator_names": self._actuator_names,
                    "values": self.native_data.ctrl[self._actuator_ids].tolist(),
                    "sim_time": timestamp, "timestamp_s": timestamp,
                    "semantics": "native actuator command state; never measured joint position"}}

# END native robosuite execution backend without recording dependencies.
