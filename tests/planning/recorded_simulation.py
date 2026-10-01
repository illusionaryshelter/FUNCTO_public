"""Execute a named joint curve with native controls in an isolated recording.

The full recorded integration state is read only during explicit setup. The
planner-facing methods return robot proprioception and RGB-D; object states
and solved contacts are available only to an explicitly supplied evaluator.
Accepted plans and authorized rejected-candidate diagnostics retain different
labels, and both stop at the original first evidence barrier.
"""

# BEGIN recorded scene restoration and physical control-only trajectory trial.
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time

import h5py
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_saved_trajectory(path, *, evaluation_mode):
    """Load the exported cubic without smoothing, retiming or relabelling it."""
    from manipulation.planning import JointTrajectory

    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    expected = {"accepted_plan": "planner_accepted",
                "diagnostic_rejected_candidate": "rejected_diagnostic_not_executable"}
    if evaluation_mode not in expected or metadata["classification"] != expected[evaluation_mode]:
        raise ValueError("Execution mode must retain the saved planner classification")
    with np.load(path, allow_pickle=False) as arrays:
        if str(arrays["classification"].item()) != expected[evaluation_mode]:
            raise ValueError("Saved trajectory arrays and metadata disagree on planner status")
        curve = JointTrajectory(tuple(arrays["joint_names"].tolist()), arrays["times"].copy(),
            arrays["position"].copy(), arrays["velocity"].copy(), arrays["progress_u"].copy(),
            arrays["progress_s"].copy(), tuple(metadata["events"]), tuple(metadata["conditions"]),
            metadata["residuals"])
    if curve.next_verification != metadata["next_verification"]:
        raise ValueError("The saved evidence barrier must be preserved exactly")
    return curve


class RecordedSimulation:
    """One restored native scene with explicitly identified robot feedback.

    ``native_model`` / ``native_data`` belong to camera and evaluator code,
    never to planning. The only execution writes are actuator controls; model
    timestep and all generalized positions remain native throughout a trial.
    """

    @classmethod
    def restore(cls, *, scene_path, catalog_path, episode_path, frame_index,
                base_link, tcp_site, arm_joint_names, finger_joint_names,
                gripper_type, policy_frequency, arm_kp, arm_kd):
        from robosuite.controllers.parts.generic.joint_pos import JointPositionController
        from robosuite.controllers.parts.gripper.simple_grip import SimpleGripController
        from robosuite.models.grippers import gripper_factory
        from robosuite.utils.binding_utils import MjSim

        gains = np.asarray([arm_kp, arm_kd], dtype=float)
        if type(frame_index) is not int or frame_index < 0 or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("Restore frame and joint-controller gains must be explicit and valid")
        obj = cls()
        catalog = json.loads(Path(catalog_path).read_text())
        state_spec = catalog["integration_state"]
        if state_spec["spec_name"] != "mjSTATE_INTEGRATION":
            raise ValueError("Only the recording's declared full integration-state format is supported")
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        if state_spec["spec_value"] != spec:
            raise ValueError("Recording and native MuJoCo integration-state definitions differ")
        obj.native_model = mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))
        obj._sim = MjSim(obj.native_model)
        obj.native_data = obj._sim.data._data
        expected_size = mujoco.mj_stateSize(obj.native_model, spec)
        if expected_size != state_spec["size"]:
            raise ValueError("Recorded integration-state size differs from the selected model")
        with h5py.File(episode_path, "r") as stream:
            state = np.asarray(stream["states/integration_state"][frame_index], dtype=float)
            recorded_time = float(stream["states/sim_time"][frame_index])
        if state.shape != (expected_size,) or not np.isfinite(state).all():
            raise ValueError("Selected recorded integration state must be finite and complete")
        mujoco.mj_setState(obj.native_model, obj.native_data, state, spec)
        mujoco.mj_forward(obj.native_model, obj.native_data)
        if not np.isclose(obj.native_data.time, recorded_time, atol=1e-10, rtol=0):
            raise ValueError("Restored integration time does not match the selected recorded frame")
        obj.arm_joint_names, obj.finger_joint_names = tuple(arm_joint_names), tuple(finger_joint_names)
        obj.joint_names = obj.arm_joint_names + obj.finger_joint_names
        if not obj.arm_joint_names or not obj.finger_joint_names or len(set(obj.joint_names)) != len(obj.joint_names):
            raise ValueError("Arm and finger feedback names must be nonempty and disjoint")
        obj._base_id = obj.native_model.body(base_link).id
        obj._site_id = obj.native_model.site(tcp_site).id
        obj._joint_ids = np.asarray([obj.native_model.joint(name).id for name in obj.joint_names])
        if not np.isin(obj.native_model.jnt_type[obj._joint_ids],
                       [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]).all():
            raise ValueError("Only explicit scalar robot feedback is supported")
        obj._qpos_ids = obj.native_model.jnt_qposadr[obj._joint_ids].copy()
        obj._qvel_ids = obj.native_model.jnt_dofadr[obj._joint_ids].copy()
        actuator_ids = []
        for joint_id in obj._joint_ids:
            matches = np.flatnonzero((obj.native_model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                     & (obj.native_model.actuator_trnid[:, 0] == joint_id))
            if len(matches) != 1:
                raise ValueError("Every controlled scalar joint must have exactly one native joint actuator")
            actuator_ids.append(int(matches[0]))
        obj._actuator_ids = np.asarray(actuator_ids)
        model, narm = obj.native_model, len(obj.arm_joint_names)
        obj._arm_actuators, obj._finger_actuators = obj._actuator_ids[:narm], obj._actuator_ids[narm:]
        if not model.actuator_ctrllimited[obj._actuator_ids].all():
            raise ValueError("Explicit finite native actuator control bounds are required")
        # The mature arm controller emits torques, so its native transmission
        # must be unit-gear fixed-gain motors. Gripper GRIP controls retain the
        # original position-actuator gains and forces from the scene XML.
        if (np.any(model.actuator_gear[obj._arm_actuators, 0] != 1)
                or np.any(model.actuator_gainprm[obj._arm_actuators, 0] != 1)
                or np.any(model.actuator_biastype[obj._arm_actuators] != mujoco.mjtBias.mjBIAS_NONE)):
            raise ValueError("Joint torque controller requires native unit-gain torque motors")
        obj._dt = float(model.opt.timestep)
        if not np.isfinite(policy_frequency) or policy_frequency <= 0:
            raise ValueError("The recorded gripper policy frequency must be explicit and positive")
        ratio = 1 / (policy_frequency * obj._dt)
        if not np.isclose(ratio, round(ratio), atol=1e-10, rtol=0):
            raise ValueError("Native physics steps must divide the declared gripper policy period")
        obj._grip_stride = int(round(ratio))

        def indexes(start, stop):
            return {"joints": obj._joint_ids[start:stop].tolist(),
                    "qpos": obj._qpos_ids[start:stop].tolist(),
                    "qvel": obj._qvel_ids[start:stop].tolist(),
                    "actuators": obj._actuator_ids[start:stop].tolist()}

        bounds = model.actuator_ctrlrange[obj._arm_actuators]
        obj._arm_controller = JointPositionController(obj._sim, indexes(0, narm),
            actuator_range=(bounds[:, 0], bounds[:, 1]), ref_name=tcp_site,
            input_type="absolute", kp=float(arm_kp), kd=float(arm_kd), damping_ratio=1,
            policy_freq=1/obj._dt, lite_physics=False)
        bounds = model.actuator_ctrlrange[obj._finger_actuators]
        obj._gripper_model = gripper_factory(gripper_type, idn="trajectory_trial")
        if obj._gripper_model.dof != 1 or len(obj._gripper_model.actuators) != len(obj.finger_joint_names):
            raise ValueError("The declared mature binary gripper model must match the native actuator count")
        # GRIP scales normalized actuator commands linearly into ctrlrange.
        # Invert the restored ctrl to resume Python action memory exactly; q
        # can lag commanded positions under contact and cannot replace ctrl.
        restored_ctrl = obj.native_data.ctrl[obj._finger_actuators].copy()
        widths = bounds[:, 1] - bounds[:, 0]
        if np.any(widths <= 0) or np.any(restored_ctrl < bounds[:, 0]) or np.any(restored_ctrl > bounds[:, 1]):
            raise ValueError("Restored gripper ctrl must be inside finite native ranges")
        obj._gripper_model.current_action = (restored_ctrl - bounds.mean(axis=1)) / (widths / 2)
        obj._grip_controller = SimpleGripController(obj._sim, indexes(narm, len(obj.joint_names)),
            actuator_range=(bounds[:, 0], bounds[:, 1]), policy_freq=float(policy_frequency))
        obj._grip_controller.set_goal(obj._gripper_model.current_action.copy())
        if not np.allclose(obj._grip_controller.run_controller(), restored_ctrl, atol=1e-12, rtol=0):
            raise ValueError("Mature GRIP scaling does not reproduce the restored native controls")
        obj.restore_report = {"scene_path": str(Path(scene_path).resolve()),
            "episode_path": str(Path(episode_path).resolve()), "frame_index": frame_index,
            "sim_time": recorded_time, "integration_state_spec": deepcopy(state_spec),
            "state_read_scope": "full state used only for simulator setup; planner sees named proprioception and RGB-D",
            "arm_controller": "robosuite JointPositionController, absolute q, native torque actuators",
            "arm_kp": float(arm_kp), "arm_kd": float(arm_kd),
            "effective_arm_kp": obj._arm_controller.kp.tolist(),
            "effective_arm_kd": obj._arm_controller.kd.tolist(),
            "desired_velocity_feedforward": False,
            "gripper_model": gripper_type, "gripper_controller": "robosuite SimpleGripController (GRIP)",
            "gripper_policy_frequency_hz": float(policy_frequency),
            "gripper_current_action": obj._gripper_model.current_action.tolist(),
            "gripper_initialization": "inverse native ctrlrange scaling of restored ctrl; not measured q",
            "python_arm_controller_history": "new joint-position goal from the exported curve; prior composite IK goal history is not restored"}
        return obj

    def proprioception(self):
        """Return only explicitly named robot feedback and its native TCP."""
        base = np.eye(4)
        base[:3, :3] = self.native_data.xmat[self._base_id].reshape(3, 3)
        base[:3, 3] = self.native_data.xpos[self._base_id]
        tool = np.eye(4)
        tool[:3, :3] = self.native_data.site_xmat[self._site_id].reshape(3, 3)
        tool[:3, 3] = self.native_data.site_xpos[self._site_id]
        return {"joint_names": self.joint_names,
                "q": self.native_data.qpos[self._qpos_ids].copy(),
                "dq": self.native_data.qvel[self._qvel_ids].copy(),
                "base_pose": base, "tool_pose": tool, "sim_time": float(self.native_data.time)}

    def capture_rgbd(self, camera):
        """Expose synchronized visible RGB-D and calibration, excluding labels."""
        captured = camera.capture(self.native_data)
        return {"rgb": captured["rgb"], "depth_m": captured["depth_m"],
                "calibration": deepcopy(camera.calibration), "sim_time": float(self.native_data.time)}

    def execute(self, trajectory, *, evaluation_mode, output, step_callback=None,
                max_physics_steps=30_000, start_position_tolerance=2e-6,
                start_velocity_tolerance=2e-6):
        """Follow the exact exported curve until its original evidence barrier.

        ``step_callback(model, data, record)`` is an evaluator/camera hook after
        each physical step. Its output is never fed to the planner here. The
        runner does not certify a grasp, bypass a barrier, or advance a skill.
        The last native step can finish less than one timestep after the
        barrier's clock time, but its commanded q/dq/progress is clamped to the
        barrier itself. No goal from the later curve is sent to an actuator.
        """
        if evaluation_mode not in ("accepted_plan", "diagnostic_rejected_candidate"):
            raise ValueError("Use an explicit accepted or rejected diagnostic evaluation mode")
        if tuple(trajectory.joint_names) != self.arm_joint_names:
            raise ValueError("Trajectory joint names/order must equal the explicitly controlled arm")
        stop = float(trajectory.executable_until)
        arrays = (trajectory.times, trajectory.position, trajectory.velocity,
                  trajectory.progress_u, trajectory.progress_s, trajectory._curve.c)
        if not all(np.isfinite(value).all() for value in arrays) or not 0 <= stop <= trajectory.duration:
            raise ValueError("Only a complete finite exported trajectory may be physically evaluated")
        count = int(np.ceil(stop/self._dt - 1e-10))
        if type(max_physics_steps) is not int or not 0 <= count <= max_physics_steps <= 100_000:
            raise ValueError("Trajectory exceeds the explicit bounded local physics-step budget")
        tolerances = np.asarray([start_position_tolerance, start_velocity_tolerance], float)
        if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
            raise ValueError("Explicit finite positive start-feedback tolerances are required")
        start_q, start_dq, _ = trajectory.sample(0.)
        actual = self.proprioception()
        narm = len(self.arm_joint_names)
        start_errors = np.array([np.max(np.abs(actual["q"][:narm]-start_q)),
                                 np.max(np.abs(actual["dq"][:narm]-start_dq))])
        if np.any(start_errors > tolerances):
            raise ValueError(f"Curve and restored feedback have different entrances: q={start_errors[0]}, dq={start_errors[1]}; tolerances={tolerances.tolist()}")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        started, sim_start = time.monotonic(), float(self.native_data.time)
        report = {"evaluation_mode": evaluation_mode,
            "planner_accepted": evaluation_mode == "accepted_plan", "status": "running",
            "restore": self.restore_report, "next_verification": trajectory.next_verification,
            "evidence_passed": False, "full_trajectory_duration_s": trajectory.duration,
            "planned_stop_s": stop, "physics_timestep_s": self._dt,
            "start_feedback_check": {"maximum_position_error": float(start_errors[0]),
                                     "maximum_velocity_error": float(start_errors[1]),
                                     "position_tolerance": float(tolerances[0]),
                                     "velocity_tolerance": float(tolerances[1]),
                                     "tolerance_scope": "float32 trajectory export roundoff only; does not authorize another episode entrance"},
            "barrier_discretization": "ceil native steps; final command q/dq/progress clamped to original barrier; physical clock may overrun by less than one timestep",
            "executed_steps": 0, "barrier_crossed": False,
            "gripper_command_timing": "original events sampled at declared policy ticks; at most one policy period later",
            "events": [], "superseded_gripper_events": [],
            "pending_verifications": [deepcopy(event) for event in trajectory.events if event["requires_evidence"] is not None],
            "unconsumed_events": deepcopy(trajectory.events)}
        command_events = sorted((deepcopy(event) for event in trajectory.events
                                 if event["kind"] in ("open", "close", "hold")), key=lambda event: event["time"])
        controls, samples, event_index = [], [], 0
        command = "hold"
        saturation_count = 0

        def record(elapsed, desired_q, desired_dq, progress):
            feedback = self.proprioception()
            return {"elapsed": float(elapsed), "sim_time": feedback["sim_time"],
                    "curve_time": min(float(elapsed), stop),
                    "desired_q": np.asarray(desired_q).copy(), "desired_dq": np.asarray(desired_dq).copy(),
                    "q": feedback["q"], "dq": feedback["dq"], "tool_pose": feedback["tool_pose"],
                    "progress": float(progress), "gripper_command": command}

        q, dq, progress = trajectory.sample(0.)
        samples.append(record(0., q, dq, progress))
        try:
            for index in range(count):
                elapsed = index * self._dt
                if index % self._grip_stride == 0:
                    due = []
                    while event_index < len(command_events) and command_events[event_index]["time"] <= elapsed + 1e-12:
                        due.append(command_events[event_index])
                        event_index += 1
                    if due:
                        # A 20 Hz binary controller can apply one intention at
                        # a tick. Preserve superseded source events explicitly;
                        # never report a command as executed if it was replaced
                        # before the mature formatter saw it.
                        report["superseded_gripper_events"].extend(deepcopy(due[:-1]))
                        command = due[-1]["kind"]
                        report["events"].append({**due[-1], "applied_at_s": elapsed})
                    action = self._gripper_model.format_action(np.array([{"open": -1., "close": 1., "hold": 0.}[command]]))
                    self._grip_controller.set_goal(action)
                next_time = (index + 1) * self._dt
                q, dq, progress = trajectory.sample(min(next_time, stop))
                self._arm_controller.set_goal(q.copy())
                torque = self._arm_controller.run_controller()
                bounds = self.native_model.actuator_ctrlrange[self._arm_actuators]
                bounded = np.clip(torque, bounds[:, 0], bounds[:, 1])
                saturation_count += int(np.count_nonzero(bounded != torque))
                self.native_data.ctrl[self._arm_actuators] = bounded
                self.native_data.ctrl[self._finger_actuators] = self._grip_controller.run_controller()
                controls.append(self.native_data.ctrl[self._actuator_ids].copy())
                self._sim.step()
                # mj_step leaves derived positions at its integration stage.
                # Forward refreshes sensor/FK evidence without another step.
                mujoco.mj_forward(self.native_model, self.native_data)
                current = record(next_time, q, dq, progress)
                samples.append(current)
                report["executed_steps"] = index + 1
                if not np.isfinite(current["q"]).all() or not np.isfinite(current["dq"]).all():
                    raise FloatingPointError("Native physical robot feedback became non-finite")
                if not np.isclose(self.native_data.time, sim_start + next_time, atol=1e-8, rtol=0):
                    raise RuntimeError("Native integration time reset or skipped a physical step")
                if step_callback is not None:
                    step_callback(self.native_model, self.native_data, current)
            report["status"] = "stopped_for_evidence" if trajectory.next_verification is not None else "curve_finished"
        except Exception as error:
            report.update(status="execution_error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            actual_q = np.asarray([row["q"] for row in samples])
            desired_q = np.asarray([row["desired_q"] for row in samples])
            final_elapsed = float(self.native_data.time - sim_start)
            applied_ids = {event["event_id"] for event in report["events"]}
            barrier_command_reached = report["executed_steps"] > 0 and samples[-1]["curve_time"] >= stop-1e-12
            report.update(elapsed_wall_s=time.monotonic()-started, elapsed_sim_s=final_elapsed,
                actual_sim_stop_s=final_elapsed, last_command_sample_s=samples[-1]["curve_time"] if controls else None,
                remaining_to_barrier_s=max(0., stop-final_elapsed),
                physical_clock_overrun_s=max(0., final_elapsed-stop),
                barrier_command_reached=barrier_command_reached,
                unapplied_gripper_events=[deepcopy(event) for event in command_events if event["event_id"] not in applied_ids],
                maximum_arm_tracking_error_rad=float(np.max(np.abs(actual_q[:, :narm]-desired_q))),
                actuator_saturated_coordinates=saturation_count,
                maximum_absolute_actual_arm_velocity=float(np.max(np.abs(np.asarray([row["dq"] for row in samples])[:, :narm]))),
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                unconsumed_events=[deepcopy(event) for event in trajectory.events if event["event_id"] not in applied_ids])
            np.savez_compressed(output/"physical-trajectory.npz", joint_names=np.asarray(self.joint_names),
                elapsed=np.asarray([row["elapsed"] for row in samples]),
                curve_time=np.asarray([row["curve_time"] for row in samples]),
                sim_time=np.asarray([row["sim_time"] for row in samples]),
                desired_q=desired_q, desired_dq=np.asarray([row["desired_dq"] for row in samples]),
                actual_q=actual_q, actual_dq=np.asarray([row["dq"] for row in samples]),
                actual_tool_pose=np.asarray([row["tool_pose"] for row in samples]),
                progress=np.asarray([row["progress"] for row in samples]), controls=np.asarray(controls),
                gripper_command=np.asarray([row["gripper_command"] for row in samples]))
            (output/"execution-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return report
# END recorded scene restoration and physical control-only trajectory trial.
