"""Native MuJoCo control through installed robosuite joint/gripper controllers.

Bind an already initialized MjSim without loading a recording or resetting its
state. Absolute arm positions and independent native finger positions go to
mature controllers once per native physics step, without a second interpolator.
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

        bounds = model.actuator_ctrlrange[self._arm_actuators]
        self._arm_controller = JointPositionController(self._sim, indexes(0, narm),
            actuator_range=(bounds[:, 0], bounds[:, 1]), ref_name=tcp_site,
            input_type="absolute", kp=float(arm_kp), kd=float(arm_kd), damping_ratio=1,
            policy_freq=1/self._dt, lite_physics=False, interpolator=None)
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
            "backend": "robosuite native MuJoCo joint-position and gripper controllers",
            "source": source, "trial_id": self._trial_id,
            "joint_names": list(self.joint_names), "arm_joint_names": list(self.arm_joint_names),
            "finger_joint_names": list(self.finger_joint_names),
            "base_link": base_link, "reference_frame": self.reference_frame,
            "tool_link": self.tool_link, "tool_transform": self.tool_transform.tolist(),
            "actuator_names": list(self.actuator_names), "native_timestep_s": self.timestep,
            "actuator_ctrlrange": model.actuator_ctrlrange[self._actuator_ids].tolist(),
            "actuator_forcelimited": model.actuator_forcelimited[self._actuator_ids].astype(bool).tolist(),
            "actuator_forcerange": model.actuator_forcerange[self._actuator_ids].tolist(),
            "arm_controller": "robosuite JointPositionController, absolute q, native torque actuators",
            "arm_kp": float(arm_kp), "arm_kd": float(arm_kd),
            "effective_arm_kp": self._arm_controller.kp.tolist(),
            "effective_arm_kd": self._arm_controller.kd.tolist(),
            "arm_interpolator": None, "desired_velocity_feedforward": False,
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

    def apply_control(self, arm_q, finger_positions):
        """Write one absolute arm goal and independent native finger targets.

        This method does not integrate physics. The caller records the exact
        returned controls before calling step(), so rejected measurements can
        never be mistaken for an unapplied physical command.
        """
        arm_q = _finite_array(arm_q, (len(self.arm_joint_names),), "Absolute arm position goal")
        finger_positions = _finite_array(finger_positions, (len(self.finger_joint_names),), "Native finger position goals")
        bounds = self.finger_ctrlrange
        if np.any(finger_positions < bounds[:, 0]) or np.any(finger_positions > bounds[:, 1]):
            raise ValueError("Finger position goals must remain inside native control bounds")
        self._grip_controller.set_goal(finger_positions.copy())
        self._arm_controller.set_goal(arm_q.copy())
        torque = _finite_array(self._arm_controller.run_controller(), (len(self.arm_joint_names),), "Native arm torque")
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
                "arm_control_saturated_coordinates": int(np.count_nonzero(bounded != torque))}
        # END count only the arm controller's clipped torque coordinates.

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


    def prepare_abort_hold(self, feedback, finger_positions, *, expected_sim_time):
        """Prepare mature-controller hold output without writing ctrl or stepping.

        Synchronous simulation stops on failure. This records a prospective
        damping/hold command for a caller, not a verified physical stop. The
        last *applied* finger goal is retained; a rejected event cannot reach
        the gripper through fault handling.
        """
        result = {"prepared": False, "native_ctrl_written": False, "physical_steps": 0,
                  "physical_stop_confirmed": False}
        try:
            if (tuple(feedback.get("joint_names", ())) != self.joint_names
                    or tuple(feedback.get("arm_joint_names", ())) != self.arm_joint_names
                    or tuple(feedback.get("finger_joint_names", ())) != self.finger_joint_names
                    or feedback.get("trial_id") != self._trial_id
                    or feedback.get("source") != self._feedback_source):
                raise ValueError("Cannot prepare a hold from unidentified feedback")
            current_time = float(self.native_data.time)
            clocks = np.asarray([feedback.get("sim_time"), feedback.get("timestamp_s"), current_time], dtype=float)
            if (not np.isfinite(clocks).all() or np.max(np.abs(clocks-expected_sim_time)) > 1e-8
                    or feedback.get("feedback_id") != f"{self._trial_id}@native_time={current_time:.17g}"):
                raise ValueError("Cannot prepare a hold from stale or discontinuous native feedback time")
            q = _finite_array(feedback["q"], (len(self.joint_names),), "Hold feedback positions")
            _finite_array(feedback["dq"], q.shape, "Hold feedback velocities")
            ranges = self.joint_ranges
            if np.any(q < ranges[:, 0]) or np.any(q > ranges[:, 1]):
                raise ValueError("Cannot turn out-of-range actual coordinates into a new hold goal")
            self._arm_controller.set_goal(q[:len(self.arm_joint_names)].copy())
            torque = _finite_array(self._arm_controller.run_controller(),
                                    (len(self.arm_joint_names),), "Prepared hold torque")
            bounds = self.native_model.actuator_ctrlrange[self._arm_actuators]
            torque = np.clip(torque, bounds[:, 0], bounds[:, 1])
            self._grip_controller.set_goal(finger_positions.copy())
            finger_ctrl = np.asarray(self._grip_controller.run_controller(), dtype=float)
            if not np.array_equal(finger_ctrl, finger_positions):
                raise ValueError("Prepared hold changed the last applied native finger goal")
            result.update(prepared=True, arm_position_goal=q[:len(self.arm_joint_names)].tolist(),
                          finger_position_goal=finger_positions.tolist(),
                          actuator_names=list(self._actuator_names),
                          prospective_actuator_ctrl=np.r_[torque, finger_ctrl].tolist())
        except Exception as error:
            result["reason"] = f"{type(error).__name__}: {error}"
        return result


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
