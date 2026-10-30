"""Native MuJoCo control through installed robosuite joint/gripper controllers.

Bind an already initialized MjSim without loading a recording or resetting its
state. Absolute arm q/dq/ddq from one trajectory sample and independent native
finger positions go to mature controllers once per native physics step. The
arm adapter adds analytic trajectory feedforward using the controller's own
native mass matrix; no second interpolator or full-scene inverse dynamics runs.
The backend owns the integral acceleration state and commits it only after a
confirmed native interval. Finger gravity updates the affine actuator bias
before original total force clipping; FK uses separate data.
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
                 arm_kp, arm_kd, velocity_limits, robot_gravity, source):
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
        # BEGIN fixed-robot gravity is an explicit independent model input.
        if tuple(robot_gravity.joint_names) != self.joint_names:
            raise ValueError("Robot gravity must use this backend's named encoder order")
        self._robot_gravity = robot_gravity
        # END the live task model never supplies finger compensation forces.
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
        # BEGIN one effective torque domain for this unit-transmission binding.
        # All active native limits act on the same N m coordinate because this
        # backend already requires exactly one unit-gain motor per arm joint.
        # Unused force-range fields are often [0, 0]; their flags determine
        # whether they participate. Fixed robot model limits are task invariant.
        effective = model.actuator_ctrlrange[self._arm_actuators].copy()
        arm_joints = self._joint_ids[:narm]
        for active, ranges in (
                (model.actuator_forcelimited[self._arm_actuators],
                 model.actuator_forcerange[self._arm_actuators]),
                (model.jnt_actfrclimited[arm_joints], model.jnt_actfrcrange[arm_joints])):
            selected = np.asarray(active, dtype=bool)
            effective[selected, 0] = np.maximum(effective[selected, 0], ranges[selected, 0])
            effective[selected, 1] = np.minimum(effective[selected, 1], ranges[selected, 1])
        if not np.isfinite(effective).all() or np.any(effective[:, 0] >= effective[:, 1]):
            raise ValueError("Active native arm torque limits require a finite nonempty intersection")
        self._arm_torque_bounds = effective
        # END the final output, conditional integral and saturation share L/U.
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
        # BEGIN each native attempt publishes its own boundary accounting.
        self._step_attempts = 0
        self._last_step_accounting = None
        # END later feedback failures cannot erase an already completed step.
        # END physical finger targets through the existing unscaled GRIP.

        def indexes(start, stop):
            return {"joints": self._joint_ids[start:stop].tolist(),
                    "qpos": self._qpos_ids[start:stop].tolist(),
                    "qvel": self._qvel_ids[start:stop].tolist(),
                    "actuators": self._actuator_ids[start:stop].tolist()}

        # BEGIN native mass/bias compensation is part of the trajectory law.
        bounds = self._arm_torque_bounds
        # BEGIN joint-space control has no site pose or Jacobian observation.
        self._arm_controller = JointPositionController(self._sim, indexes(0, narm),
            actuator_range=(bounds[:, 0], bounds[:, 1]), ref_name=None,
            input_type="absolute", kp=float(arm_kp), kd=float(arm_kd), damping_ratio=1,
            policy_freq=1/self._dt, lite_physics=False, interpolator=None,
            use_torque_compensation=True)
        # BEGIN acceleration-domain integral with the declared nominal poles.
        # The current device supplies positive scalar Kp/Kd. The repeated
        # slow pole construction requires Kd^2 >= 3*Kp; it is a fixed device
        # design, not a gain selected from task or contact outcomes.
        kp, kd = self._arm_controller.kp, self._arm_controller.kd
        discriminant = kd*kd - 3*kp
        if np.any(discriminant < 0):
            raise ValueError("Integral pole construction requires Kd squared >= 3 Kp")
        slow = (kd - np.sqrt(discriminant))/3
        self._arm_ki = slow*slow*(kd - 2*slow)
        if not np.isfinite(self._arm_ki).all() or np.any(self._arm_ki <= 0):
            raise ValueError("Integral pole construction must produce finite positive Ki")
        self._integral_acceleration = np.zeros(narm)
        self._integral_sim_time = self.sim_time
        self._pending_integral_interval = None
        self._control_rebase_requested = False
        # END z is backend state; q/dq feedback remains a sensor-only contract.
        # END ref_name=None disables the mature controller's unused site reads.
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
        # BEGIN one gravity law for ordinary control, recovery and stopping.
        self._finger_gain = model.actuator_gainprm[self._finger_actuators, 0].copy()
        self._finger_velocity_bias = model.actuator_biasprm[self._finger_actuators, 2].copy()
        self._finger_force_bounds = model.actuator_forcerange[self._finger_actuators].copy()
        if not model.actuator_forcelimited[self._finger_actuators].all():
            raise ValueError("Gravity-compensated fingers require their declared total force limits")
        self._update_finger_gravity(restored_ctrl)
        mujoco.mj_forward(model, self.native_data)
        # END initialization preserves native q/dq/time and the held command.
        # BEGIN model FK uses measured arm q, never a live TCP observation.
        self._prepare_reference_fk()
        # END the same calibrated model supplies reference and measured-q FK.
        self._description = {
            "backend": "robosuite native MuJoCo trajectory-state and gripper controllers",
            "source": source, "trial_id": self._trial_id,
            "joint_names": list(self.joint_names), "arm_joint_names": list(self.arm_joint_names),
            "finger_joint_names": list(self.finger_joint_names),
            "base_link": base_link, "reference_frame": self.reference_frame,
            "tool_link": self.tool_link, "tool_transform": self.tool_transform.tolist(),
            # BEGIN disclose model-derived poses separately from encoder feedback.
            "base_pose_source": "known_fixed_robot_model_calibration",
            "tool_pose_source": "forward_kinematics_from_measured_arm_q",
            "tool_pose_is_independent_observation": False,
            "proprioception_live_fields": ["named_joint_q", "named_joint_dq", "named_actuator_ctrl", "time"],
            # END no independent TCP, acceleration or force sensor is claimed.
            "actuator_names": list(self.actuator_names), "native_timestep_s": self.timestep,
            "actuator_ctrlrange": model.actuator_ctrlrange[self._actuator_ids].tolist(),
            "actuator_forcelimited": model.actuator_forcelimited[self._actuator_ids].astype(bool).tolist(),
            "actuator_forcerange": model.actuator_forcerange[self._actuator_ids].tolist(),
            # BEGIN publish the limits used by every arm output computation.
            "joint_actuator_forcelimited": model.jnt_actfrclimited[self._joint_ids].astype(bool).tolist(),
            "joint_actuator_forcerange": model.jnt_actfrcrange[self._joint_ids].tolist(),
            "effective_arm_torque_range": self._arm_torque_bounds.tolist(),
            "arm_torque_limit_source": "Intersection of ctrlrange, active actuator forcerange and active joint actuatorfrcrange under one-to-one unit transmission",
            # END ranges are model declarations, not measured force signals.
            # BEGIN bind the complete trajectory-control contract by joint name.
            "arm_controller": "robosuite JointPositionController plus same-curve dq/ddq feedforward and acceleration integral, native torque actuators",
            # BEGIN model terms are not measured object/contact forces.
            "arm_dynamics_source": "Known fixed robot model M(q) and bias C(q,dq)+g at measured robot q/dq; no target-pose or contact-force feedback",
            # END robot model terms retain their own source and units.
            "arm_kp": float(arm_kp), "arm_kd": float(arm_kd),
            "effective_arm_kp": self._arm_controller.kp.tolist(),
            "effective_arm_kd": self._arm_controller.kd.tolist(),
            "arm_interpolator": None, "desired_velocity_feedforward": True,
            "desired_acceleration_feedforward": True,
            "controller_policy": {
                "mode": "joint_trajectory_q_dq_ddq_integral",
                "arm_joint_names": list(self.arm_joint_names),
                "kp": self._arm_controller.kp.tolist(), "kd": self._arm_controller.kd.tolist(),
                "ki": self._arm_ki.tolist(),
                "native_timestep_s": self.timestep,
                "reference_derivatives": ["q", "dq", "ddq"],
                "sampling_phase": "step_start",
                "integral_update": "confirmed_native_step_torque_conditional",
                "torque_limits": self._arm_torque_bounds.tolist(),
                "acceleration_knot_side": "right_internal_last_left"},
            "integral_state": "Named acceleration state z [rad/s^2]; retained across continuous segments and stop; explicit rebase matches the next control output once",
            "integral_snapshot": "Completed native-step boundary: named z, native time and controller policy; saved separately from encoder feedback",
            # END this policy cannot be inferred from a q-only authorization.
            "gripper_controller": "Unscaled SimpleGripController position target plus fixed-robot gravity feedforward",
            "finger_gravity_control": {
                "source": self._robot_gravity.source,
                "robot_joint_names": list(self._robot_gravity.joint_names),
                "gravity_world_m_s2": self._robot_gravity.model.opt.gravity.tolist(),
                "force_law": "clip(Kp*(u-q)+velocity_bias*dq+g(q), original actuator forcerange)",
                "force_range": self._finger_force_bounds.tolist(),
                "application": "Same pre-step robot q; affine bias before native total force clipping",
                "model_scope": "Official fixed robot and installed end effector; reused across tasks"},
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
        # BEGIN the known robot installation defines the fixed model gauge.
        # Scratch FK uses this calibration; it is not a live body-pose sensor.
        return "world"
        # END the gauge does not disclose a task object's current pose.

    @property
    def tool_link(self):
        # BEGIN the named model site identifies the estimated TCP frame.
        # No runner may relabel it or attach an unapplied TCP transform.
        return self.native_model.site(self._site_id).name
        # END its current pose is computed from measured joint positions.

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

    # BEGIN explicit controller ownership and completed-boundary state APIs.
    def request_control_rebase(self):
        """Match the next actual control write to the current effective ctrl.

        Call only for explicit control takeover or reference rebasing. A
        continuous segment boundary needs no operation. The next successful
        apply consumes this request; repeated evaluations at that same time
        then retain z. A stop cancels a request that has not yet been applied.
        """
        self._control_rebase_requested = True

    def snapshot_control_state(self):
        """Return named integral state at a completed control boundary.

        A prepared, unexecuted control interval contains e/M/u which are not
        part of this boundary format. Native integration state is persisted
        by the caller with MuJoCo's existing getState API at this same time.
        """
        if self._pending_integral_interval is not None or self._control_rebase_requested:
            raise ValueError("Control snapshots require a completed step with no pending rebase")
        if self._integral_sim_time != self.sim_time:
            raise ValueError("Control snapshot time must equal the native integration boundary")
        return {"arm_joint_names": list(self.arm_joint_names),
                "integral_acceleration": self._integral_acceleration.tolist(),
                "sim_time": self._integral_sim_time,
                "controller_policy": deepcopy(self._description["controller_policy"])}

    def restore_control_state(self, state):
        """Restore z after the caller restores the matching native state.

        Names, time and policy define the saved state's coordinates and
        control law. A complete state resumes directly and does not request
        an output rebase; initial takeover from a legacy controller is an
        explicit request_control_rebase call instead.
        """
        if self._pending_integral_interval is not None or self._control_rebase_requested:
            raise ValueError("Restore control state before preparing a control interval")
        if (tuple(state["arm_joint_names"]) != self.arm_joint_names
                or state["controller_policy"] != self._description["controller_policy"]):
            raise ValueError("Control state must use this binding's named coordinates and policy")
        stamp = float(state["sim_time"])
        if not np.isfinite(stamp) or stamp != self.sim_time:
            raise ValueError("Control state time must equal restored native time")
        values = _finite_array(state["integral_acceleration"], (len(self.arm_joint_names),),
                               "Named restored integral acceleration")
        self._integral_acceleration = values.copy()
        self._integral_sim_time = stamp
    # END exact state restoration is independent of measured q/dq feedback.

    # BEGIN measured robot state and fixed model define the finger feedforward.
    def _update_finger_gravity(self, targets):
        """Update this interval's force; return pre-step model output terms.

        Actual post-step actuator forces and task contacts belong to separate
        physical records. They do not supply this model calculation.
        """
        narm = len(self.arm_joint_names)
        q = self.native_data.qpos[self._qpos_ids].copy()
        dq = self.native_data.qvel[self._qvel_ids].copy()
        gravity = self._robot_gravity.compute(q)[narm:]
        self.native_model.actuator_biasprm[self._finger_actuators, 0] = gravity
        servo = self._finger_gain*(targets-q[narm:]) + self._finger_velocity_bias*dq[narm:]
        requested = servo + gravity
        bounded = np.clip(requested, self._finger_force_bounds[:, 0], self._finger_force_bounds[:, 1])
        return {"finger_gravity_force": gravity, "finger_position_servo_force": servo,
                "finger_requested_force": requested, "finger_bounded_force": bounded,
                "finger_force_saturated": bounded != requested}
    # END native affine actuation clips this same total during physical stepping.

    # BEGIN mandatory trajectory derivatives through the native arm controller.
    def apply_control(self, arm_q, arm_dq, arm_ddq, finger_positions):
        """Apply one named q/dq/ddq reference and native finger positions.

        Every arm vector follows ``arm_joint_names`` from this binding. The
        executor samples all three at the current native interval's start;
        none is substituted by measured velocity or finite differences here.
        Installed robosuite supplies the position/damping torque, mass matrix
        and bias at the same current state. Adding M*(Kd*dq_ref + ddq_ref)
        yields M*(ddq_ref + Kp*(q_ref-q) + Kd*(dq_ref-dq)) + bias. The same
        final calculation adds M*z before total clipping; z is the retained
        acceleration integral, not an estimate from measured contact forces.

        This method writes only actuator ctrl and does not integrate physics.
        Its raw torque components and references belong to the same call;
        saturation is applied only to their total, never to each component.
        Integral accumulation occurs only in step after physical advancement.
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
        # BEGIN one composed arm output, with state used but not integrated.
        base_torque = _finite_array(position_controller_torque + velocity_feedforward + acceleration_feedforward,
                                    (narm,), "Native trajectory-controller base torque")
        bounds = self._arm_torque_bounds
        rebased = self._control_rebase_requested
        integral = self._integral_acceleration.copy()
        if rebased:
            previous = np.clip(self.native_data.ctrl[self._arm_actuators], bounds[:, 0], bounds[:, 1])
            integral = np.linalg.solve(mass, previous-base_torque)
        integral_torque = _finite_array(mass @ integral, (narm,), "Arm integral torque")
        torque = _finite_array(base_torque + integral_torque, (narm,), "Native trajectory-controller total torque")
        bounded = np.clip(torque, bounds[:, 0], bounds[:, 1])
        finger_ctrl = np.asarray(self._grip_controller.run_controller(), dtype=float)
        if not np.array_equal(finger_ctrl, finger_positions):
            raise RuntimeError("Unscaled GRIP changed an explicit native finger position target")
        # BEGIN update gravity at the unique actual control write.
        finger_control = self._update_finger_gravity(finger_ctrl)
        # END arm and finger model outputs retain their distinct units.
        self.native_data.ctrl[self._arm_actuators] = bounded
        self.native_data.ctrl[self._finger_actuators] = finger_ctrl
        self._integral_acceleration = integral
        self._integral_sim_time = self.sim_time
        self._control_rebase_requested = False
        # Keep the last command at this native time. Repeated writes replace
        # the held interval; only step() may integrate its actual duration.
        self._pending_integral_interval = {"sim_time": self.sim_time,
            "error": (arm_q-actual_q).copy(), "mass": mass.copy(), "requested_torque": torque.copy()}
        # END no second arm writer or integration on a function-call clock.
        # BEGIN arm control clipping is separate from native finger force limits.
        return {"actuator_ctrl": self.actuator_control, **finger_control,
                "requested_arm_torque": torque.copy(), "arm_torque_saturated": bounded != torque,
                "arm_control_saturated_coordinates": int(np.count_nonzero(bounded != torque)),
                "reference_arm_q": arm_q.copy(), "reference_arm_dq": arm_dq.copy(),
                "reference_arm_ddq": arm_ddq.copy(),
                "arm_position_feedback_torque": position_feedback.copy(),
                "arm_velocity_feedback_torque": velocity_feedback.copy(),
                "arm_velocity_feedforward_torque": velocity_feedforward.copy(),
                "arm_acceleration_feedforward_torque": acceleration_feedforward.copy(),
                "arm_bias_torque": bias.copy(),
                "arm_integral_torque": integral_torque.copy(),
                "arm_integral_state": integral.copy(),
                "arm_position_error": (arm_q-actual_q).copy(),
                "arm_integral_rebased": bool(rebased)}
        # END count only the arm controller's clipped torque coordinates.
    # END no derivative-free fallback or mutation of physical q/dq/qacc.

    # BEGIN account at the native integration boundary, before derived feedback.
    @property
    def last_step_accounting(self):
        """Return the latest attempt's facts independently of later sampling."""
        return deepcopy(self._last_step_accounting)

    def step(self):
        """Integrate one native step and retain its count even if refresh fails.

        One entered attempt is conservatively charged. A normal native return
        confirms its one completed step; an exceptional return can still be
        resolved by the native clock. A stationary exceptional clock confirms
        zero completed steps without refunding the attempted work. An unreadable
        or discontinuous clock leaves only a lower bound on completed steps.
        No accounting or derived-feedback error replaces a native exception.
        """
        self._step_attempts += 1
        report = {"attempt_index": self._step_attempts, "charged_steps": 1,
                  "physical_steps": 0, "step_accounting_exact": True,
                  "native_step_attempted": False, "native_step_returned": False,
                  "before_sim_time": None, "after_sim_time": None,
                  "clock_advance_confirmed": False, "clock_error": None}
        self._last_step_accounting = report
        try:
            before = float(self.native_data.time)
            report["before_sim_time"] = before
            if not np.isfinite(before):
                raise ValueError("Native integration requires a finite initial clock")
        except Exception as error:
            report["clock_error"] = f"{type(error).__name__}: {error}"
            raise
        try:
            report["native_step_attempted"] = True
            self._sim.step()
            report["native_step_returned"] = True
        except Exception as error:
            report["native_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            # Read accounting before mj_forward or proprioception. Both may
            # fail after integration and must not make that step disappear.
            try:
                after = float(self.native_data.time)
                report["after_sim_time"] = after
                finite = bool(np.isfinite(after))
                advanced = finite and abs(after-(before+self.timestep)) <= 1e-8
                stationary = finite and after == before
                report["clock_advance_confirmed"] = bool(advanced)
                if not finite:
                    report["clock_error"] = "Native clock became nonfinite during integration"
                elif not advanced and (report["native_step_returned"] or not stationary):
                    report["clock_error"] = "Native clock does not agree with the completed integration boundary"
            except Exception as error:
                advanced = stationary = False
                report["clock_error"] = f"{type(error).__name__}: {error}"
            report["physical_steps"] = int(report["native_step_returned"] or advanced)
            # A completed native call is known work, but a contradictory clock
            # cannot authorize an exact continuation time. Keep that completed
            # step as a lower bound while explicitly marking the inconsistency.
            report["step_accounting_exact"] = bool(advanced or (stationary and not report["native_step_returned"]))
            # BEGIN commit one integral update at the confirmed physical edge.
            # This runs even if native step raises after advancing its clock,
            # and before later FK/forward refresh can fail. The saved matrix,
            # error and request belong to the command held over this interval.
            report["integral_update_committed"] = False
            if advanced:
                interval = self._pending_integral_interval
                if interval is not None:
                    h = after-before
                    mass, requested = interval["mass"], interval["requested_torque"]
                    raw = mass @ (h*self._arm_ki*interval["error"])
                    bounds = self._arm_torque_bounds
                    accepted = np.clip(raw, np.minimum(bounds[:, 0]-requested, 0.),
                                       np.maximum(bounds[:, 1]-requested, 0.))
                    old_integral = self._integral_acceleration.copy()
                    try:
                        delta = np.linalg.solve(mass, accepted)
                    except Exception as error:
                        # The native exception remains primary if integration
                        # already reported one; preserve this separate control
                        # update failure in the same boundary record.
                        report["integral_update_error"] = f"{type(error).__name__}: {error}"
                        if "native_error" not in report:
                            raise
                    else:
                        self._integral_acceleration += delta
                        self._pending_integral_interval = None
                        report.update(integral_update_committed=True,
                            integral_interval_start=interval["sim_time"],
                            integral_dt_s=h,
                            integral_requested_torque_increment=raw.tolist(),
                            integral_accepted_torque_increment=accepted.tolist(),
                            integral_acceleration_before=old_integral.tolist(),
                            integral_acceleration_after=self._integral_acceleration.tolist())
                self._integral_sim_time = after
            # END completed physical time owns the new control state.
        # mj_step leaves derived positions at its integration stage. Forward
        # refreshes feedback without another integration. Its failure preserves
        # the completed native boundary record and remains the original error.
        try:
            mujoco.mj_forward(self.native_model, self.native_data)
        except Exception as error:
            report["feedback_refresh_error"] = f"{type(error).__name__}: {error}"
            raise
        return self.last_step_accounting
    # END native counts do not depend on a successful post-step sample.


    # BEGIN fixed device calibration and scratch FK share no live pose arrays.
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
        base_ancestors, body = {0}, self._base_id
        while body:
            base_ancestors.add(body)
            body = int(model.body_parentid[body])
        if any(int(owner) in base_ancestors for owner in model.jnt_bodyid):
            raise ValueError("Robot base calibration requires a fixed base with no joint-driven ancestors")
        self._reference_data = mujoco.MjData(model)
        self._reference_data.qpos[self._qpos_ids] = self.native_data.qpos[self._qpos_ids]
        # Base ancestors are static model transforms. Kinematics reads the
        # scratch state only; no live object/body/site pose supplies this cache.
        mujoco.mj_kinematics(model, self._reference_data)
        self._calibrated_base_pose = np.eye(4)
        self._calibrated_base_pose[:3, :3] = self._reference_data.xmat[self._base_id].reshape(3, 3)
        self._calibrated_base_pose[:3, 3] = self._reference_data.xpos[self._base_id]


    def reference_tool_pose(self, arm_q):
        """Compute model TCP FK of explicit arm q without observing or stepping."""
        values = _finite_array(arm_q, (len(self.arm_joint_names),), "Reference arm positions")
        self._reference_data.qpos[self._qpos_ids[:len(values)]] = values
        mujoco.mj_kinematics(self.native_model, self._reference_data)
        pose = np.eye(4)
        pose[:3, :3] = self._reference_data.site_xmat[self._site_id].reshape(3, 3)
        pose[:3, 3] = self._reference_data.site_xpos[self._site_id]
        return pose
    # END reference q and measured q use the same known device model.


    # BEGIN bounded physical hold with actual feedback confirmation.
    def abort_hold(self, feedback, finger_positions, *, expected_sim_time, stop_velocity, max_steps):
        """Apply hold through existing controllers, then measure bounded settling.

        The arm goal is the failed sample's actual q. The last applied finger
        goal is retained, so an event rejected by supervision is never issued
        by fault handling. Hold telemetry is separate from the failed curve.
        No state reset or claimed zero velocity substitutes for feedback.
        The current integral state is retained; output matching is cancelled
        so that the zero desired velocity immediately supplies braking.
        """
        # BEGIN a stop keeps z and never consumes a pending output rebase.
        # Entry supervision can fail before the requested first normal write;
        # the stop still requires its new damping force immediately.
        self._control_rebase_requested = False
        # END holding uses the same integral without cancelling its brake.
        # BEGIN fault stopping preserves position faults separately from speed convergence.
        result = {"prepared": False, "native_ctrl_written": False, "physical_steps": 0,
                  "charged_steps": 0, "step_accounting_exact": True, "step_accounting": [],
                  "physical_stop_confirmed": False, "samples": [], "controls": [],
                  "joint_position_bounds_satisfied": None, "joint_position_violation_samples": [],
                  "physical_stop_scope": "Measured named stop velocities; does not clear the original execution fault or certify restored joint position bounds"}
        # END a stopped actuator does not imply that its physical fault disappeared.
        try:
            # BEGIN an exhausted shared budget permits observation, not motion.
            # Read current named feedback for a zero-step stop assessment;
            # an earlier failed sample cannot stand in for fresh measured dq.
            if type(max_steps) is int and max_steps == 0:
                feedback = self.proprioception()
            # END neither a controller write nor an integration is needed here.
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
            measured_entry_dq = _finite_array(feedback["dq"], q.shape, "Hold feedback velocities")
            limits = _finite_array(stop_velocity, q.shape, "Named physical stop velocities")
            if np.any(limits < 0) or type(max_steps) is not int or not 0 <= max_steps <= 10000:
                raise ValueError("Physical hold requires finite nonnegative criteria and a bounded step count")
            ranges = self.joint_ranges
            # BEGIN only arm coordinates become new hold targets.
            # Fingers retain the last commanded native positions, independently
            # checked against ctrlrange by apply_control. Their actual soft-limit
            # excursion must remain a reported fault, but cannot suppress the
            # bounded stop of a still-moving, in-range arm. No actual is clipped.
            narm = len(self.arm_joint_names)

            def retain_position_faults(positions, *, phase, sim_time):
                outside = np.flatnonzero((positions < ranges[:, 0]) | (positions > ranges[:, 1]))
                result["joint_position_bounds_satisfied"] = not bool(outside.size)
                if outside.size:
                    result["joint_position_violation_samples"].append({
                        "phase": phase, "sim_time": float(sim_time),
                        "violations": [{"joint_name": self.joint_names[int(j)],
                            "measured": float(positions[j]), "lower": float(ranges[j, 0]),
                            "upper": float(ranges[j, 1])} for j in outside]})
                return outside

            outside = retain_position_faults(q, phase="entry", sim_time=current_time)
            # BEGIN zero remaining steps never synthesize a hold or a stop.
            if max_steps == 0:
                result["samples"].append(feedback)
                result["stop_velocity"] = limits.tolist()
                result["physical_stop_confirmed"] = bool(np.all(np.abs(measured_entry_dq) <= limits))
                result["reason"] = ("No hold steps allocated; current named stop velocities are satisfied"
                    if result["physical_stop_confirmed"] else
                    "No hold steps allocated; current named stop velocities are not satisfied")
                return result
            # END any finite position fault remains independent of this velocity verdict.
            if np.any(outside < narm):
                raise ValueError("Out-of-range actual arm coordinates cannot become a new hold goal")
            goal = q[:narm].copy()
            # END the normal executor still rejects every arm and finger limit violation.
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
                # BEGIN charge before attempting integration and retain its
                # boundary facts before any possibly failing sample capture.
                previous_attempt = self._step_attempts
                result["charged_steps"] += 1
                result["joint_position_bounds_satisfied"] = None
                step_error = None
                try:
                    self.step()
                except Exception as error:
                    step_error = error
                try:
                    accounting = self.last_step_accounting
                    if accounting is None or accounting["attempt_index"] != previous_attempt+1:
                        # A wrapper failed before reaching the native boundary.
                        # Keep the conservative charge and the original error;
                        # absent new evidence cannot establish zero motion.
                        result["step_accounting_exact"] = False
                        result["step_accounting_error"] = "No current native step accounting was published"
                    else:
                        result["step_accounting"].append(accounting)
                        result["physical_steps"] += accounting["physical_steps"]
                        result["step_accounting_exact"] &= accounting["step_accounting_exact"]
                except Exception as error:
                    result["step_accounting_exact"] = False
                    result["step_accounting_error"] = f"{type(error).__name__}: {error}"
                if step_error is not None:
                    raise step_error
                if result.get("step_accounting_error") is not None:
                    raise RuntimeError(result["step_accounting_error"])
                actual = self.proprioception()
                # END failed forward/proprioception cannot reduce consumed work.
                result["samples"].append(actual)
                # BEGIN report the latest position evidence even when velocity is invalid.
                # Unknown new q must not inherit an earlier in-range verdict.
                # A finite position fault is retained before dq validation so
                # that invalid velocity cannot erase the independently known q.
                result["joint_position_bounds_satisfied"] = None
                measured_q = _finite_array(actual["q"], q.shape, "Hold actual positions")
                outside = retain_position_faults(measured_q, phase="after_step",
                                                sim_time=actual["sim_time"])
                # END all subsequent stop checks refer to this actual sample.
                measured_dq = _finite_array(actual["dq"], q.shape, "Hold actual velocities")
                # BEGIN retain ongoing finger faults while completing arm deceleration.
                if np.any(outside < narm):
                    raise ValueError("Physical arm joint range invalidated during hold")
                # END speed convergence below never certifies position-domain recovery.
                # BEGIN an invalid native clock cannot confirm a physical stop.
                actual_time = float(actual["sim_time"])
                if (not np.isfinite(actual_time)
                        or abs(actual_time-(current_time+(index+1)*self.timestep)) > 1e-8):
                    raise ValueError("Native clock became discontinuous during hold")
                # END NaN cannot pass through an otherwise false comparison.
                if np.all(np.abs(measured_dq) <= limits):
                    result["physical_stop_confirmed"] = True
                    break
            if not result["physical_stop_confirmed"]:
                result["reason"] = "Hold budget exhausted before named stop velocity criteria were observed"
        except Exception as error:
            result["reason"] = f"{type(error).__name__}: {error}"
        return result
    # END physical hold result preserves unconfirmed and failed stops.


    # BEGIN runtime robot feedback excludes simulator pose/dynamics oracles.
    def proprioception(self):
        """Read encoders/commands/time and estimate TCP with calibrated FK.

        The tool pose is a model estimate from this sample's measured arm q,
        not an independent TCP observation. Acceleration and actuator force
        are not available sensors in this binding and are not exported.
        """
        timestamp = float(self.native_data.time)
        q = self.native_data.qpos[self._qpos_ids].copy()
        tool = self.reference_tool_pose(q[:len(self.arm_joint_names)])
        return {"joint_names": self.joint_names,
                "reference_frame": self.reference_frame, "tool_link": self.tool_link,
                "tool_transform": self.tool_transform,
                "arm_joint_names": self.arm_joint_names, "finger_joint_names": self.finger_joint_names,
                "trial_id": self._trial_id, "feedback_id": f"{self._trial_id}@native_time={timestamp:.17g}",
                "source": self._feedback_source, "timestamp_s": timestamp,
                "q": q,
                "dq": self.native_data.qvel[self._qvel_ids].copy(),
                "actuator_names": self.actuator_names,
                "base_pose": self._calibrated_base_pose.copy(), "tool_pose": tool, "sim_time": timestamp,
                "base_pose_source": "known_fixed_robot_model_calibration",
                "tool_pose_source": "forward_kinematics_from_measured_arm_q",
                "tool_pose_is_independent_observation": False,
                "actuator_ctrl": {"joint_names": self.joint_names,
                    "actuator_names": self._actuator_names,
                    "values": self.native_data.ctrl[self._actuator_ids].tolist(),
                    "sim_time": timestamp, "timestamp_s": timestamp,
                    "semantics": "native actuator command state; never measured joint position"}}
    # END no live body/site, qacc or actuator-force read enters proprioception.

# END native robosuite execution backend without recording dependencies.
