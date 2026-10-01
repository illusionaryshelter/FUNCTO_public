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
from dataclasses import dataclass
import json
from pathlib import Path
import resource
import sys
import time
from uuid import uuid4

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


# BEGIN independent named finger targets bound to the exported event clock.
@dataclass(frozen=True)
class FixedFingerTargets:
    """Explicit native position targets; neither robot feedback nor a curve.

    Each row identifies one original ``(skill_id, event_id, kind)``. Event
    times belong exclusively to the exported, retimed arm trajectory. The
    independent coordinates need not have equal magnitudes or opposite signs.
    ``source`` records where the commanded targets were explicitly chosen.
    """

    joint_names: tuple[str, ...]
    source: str
    events: tuple[dict, ...]

    def __post_init__(self):
        if isinstance(self.joint_names, str):
            raise ValueError("Finger target joint names must be an ordered sequence")
        names = tuple(self.joint_names)
        if (not names or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Finger target joint names must be nonempty and unique")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Finger targets require an explicit nonempty source")
        rows, identities = [], set()
        for row in self.events:
            if not isinstance(row, dict) or set(row) != {"skill_id", "event_id", "kind", "positions"}:
                raise ValueError("Each finger target requires only skill_id, event_id, kind and positions; times come from the curve")
            if any(not isinstance(row[key], str) or not row[key].strip()
                   for key in ("skill_id", "event_id")) or row["kind"] not in ("open", "close", "hold"):
                raise ValueError("Finger targets require explicit original command-event identities")
            identity = (row["skill_id"], row["event_id"])
            if identity in identities:
                raise ValueError(f"Duplicate finger target event: {identity}")
            positions = np.asarray(row["positions"], dtype=float)
            if positions.shape != (len(names),) or not np.isfinite(positions).all():
                raise ValueError("Every finger target must contain one finite position per named joint")
            rows.append({**deepcopy(row), "positions": positions.tolist()})
            identities.add(identity)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "events", tuple(rows))

    @classmethod
    def from_mapping(cls, value):
        """Read the required JSON input without deriving or defaulting goals."""
        if not isinstance(value, dict) or set(value) != {"joint_names", "source", "events"}:
            raise ValueError("Finger target input requires exactly joint_names, source and events")
        return cls(value["joint_names"], value["source"], value["events"])

    def bind(self, trajectory, *, finger_joint_names, ctrlrange):
        """Validate every original command and bind its existing retimed time."""
        if self.joint_names != tuple(finger_joint_names):
            raise ValueError("Finger target names/order must equal the native finger actuator binding")
        bounds = np.asarray(ctrlrange, dtype=float)
        if (bounds.shape != (len(self.joint_names), 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1])):
            raise ValueError("Finger target validation requires finite native actuator control ranges")
        targets = {(row["skill_id"], row["event_id"]): row for row in self.events}
        commands = [event for event in trajectory.events if event["kind"] in ("open", "close", "hold")]
        identities = [(event["skill_id"], event["event_id"]) for event in commands]
        if len(set(identities)) != len(identities) or set(identities) != set(targets):
            raise ValueError("Finger targets must match every original command event exactly, without missing or extra events")
        bound = []
        for event, identity in zip(commands, identities):
            target = targets[identity]
            positions = np.asarray(target["positions"], dtype=float)
            if event["kind"] != target["kind"]:
                raise ValueError(f"Finger target kind differs from its original event: {identity}")
            if (not np.isfinite(event["time"]) or not 0 <= event["time"] <= trajectory.duration
                    or np.any(positions < bounds[:, 0]) or np.any(positions > bounds[:, 1])):
                raise ValueError("Finger target time or position is outside the exported curve/native actuator range")
            bound.append({**deepcopy(event), "finger_positions": positions.tolist(),
                          "target_source": self.source})
        return sorted(bound, key=lambda event: event["time"])
# END independent named finger targets bound to the exported event clock.


class RecordedSimulation:
    """One restored native scene with explicitly identified robot feedback.

    ``native_model`` / ``native_data`` belong to camera and evaluator code,
    never to planning. The only execution writes are actuator controls; model
    timestep and all generalized positions remain native throughout a trial.
    """

    @classmethod
    def restore(cls, *, scene_path, catalog_path, episode_path, frame_index,
                base_link, tcp_site, arm_joint_names, finger_joint_names,
                arm_kp, arm_kd):
        from robosuite.controllers.parts.generic.joint_pos import JointPositionController
        from robosuite.controllers.parts.gripper.simple_grip import SimpleGripController
        from robosuite.utils.binding_utils import MjSim

        gains = np.asarray([arm_kp, arm_kd], dtype=float)
        if type(frame_index) is not int or frame_index < 0 or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("Restore frame and joint-controller gains must be explicit and valid")
        obj = cls()
        # BEGIN independent identity for each restored physical trial branch.
        obj._trial_id = str(uuid4())
        # END independent identity for each restored physical trial branch.
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
        # BEGIN explicit native command identity, independent from measured q.
        obj._actuator_names = tuple(obj.native_model.actuator(index).name for index in actuator_ids)
        if any(not name for name in obj._actuator_names):
            raise ValueError("Every controlled native actuator must have an explicit name")
        obj._feedback_source = (f"Trial {obj._trial_id}: named native robot q/dq and command ctrl from {Path(scene_path).resolve()}; "
                                f"restored {Path(episode_path).resolve()} frame {frame_index}")
        # END explicit native command identity, independent from measured q.
        model, narm = obj.native_model, len(obj.arm_joint_names)
        obj._arm_actuators, obj._finger_actuators = obj._actuator_ids[:narm], obj._actuator_ids[narm:]
        if (not model.actuator_ctrllimited[obj._actuator_ids].all()
                or not np.isfinite(model.actuator_ctrlrange[obj._actuator_ids]).all()):
            raise ValueError("Explicit finite native actuator control bounds are required")
        # The mature arm controller emits torques, so its native transmission
        # must be unit-gear fixed-gain motors. Gripper GRIP controls retain the
        # original position-actuator gains and forces from the scene XML.
        if (np.any(model.actuator_gear[obj._arm_actuators, 0] != 1)
                or np.any(model.actuator_gainprm[obj._arm_actuators, 0] != 1)
                or np.any(model.actuator_biastype[obj._arm_actuators] != mujoco.mjtBias.mjBIAS_NONE)):
            raise ValueError("Joint torque controller requires native unit-gain torque motors")
        # BEGIN physical finger targets through the existing unscaled GRIP.
        # A unit-gear native position actuator interprets ctrl in the named
        # joint's physical units. A velocity/motor actuator must not be
        # silently treated as a position servo by this generic interface.
        finger_ids = obj._finger_actuators
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
        obj._dt = float(model.opt.timestep)
        if not np.isfinite(obj._dt) or obj._dt <= 0:
            raise ValueError("Native physics timestep must be finite and positive")
        # END physical finger targets through the existing unscaled GRIP.

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
        # BEGIN resume raw native command state without binary action memory.
        # Contact can keep actual q away from ctrl. Initialization therefore
        # preserves restored ctrl, and never substitutes measured positions.
        restored_ctrl = obj.native_data.ctrl[obj._finger_actuators].copy()
        widths = bounds[:, 1] - bounds[:, 0]
        if (np.any(widths <= 0) or not np.isfinite(restored_ctrl).all()
                or np.any(restored_ctrl < bounds[:, 0]) or np.any(restored_ctrl > bounds[:, 1])):
            raise ValueError("Restored gripper ctrl must be inside finite native ranges")
        obj._grip_controller = SimpleGripController(obj._sim, indexes(narm, len(obj.joint_names)),
            actuator_range=(bounds[:, 0], bounds[:, 1]), policy_freq=1/obj._dt,
            use_action_scaling=False)
        obj._grip_controller.set_goal(restored_ctrl.copy())
        if not np.array_equal(obj._grip_controller.run_controller(), restored_ctrl):
            raise ValueError("Unscaled GRIP must reproduce restored native position controls exactly")
        # END resume raw native command state without binary action memory.
        obj.restore_report = {"scene_path": str(Path(scene_path).resolve()),
            "trial_id": obj._trial_id,
            "episode_path": str(Path(episode_path).resolve()), "frame_index": frame_index,
            "sim_time": recorded_time, "integration_state_spec": deepcopy(state_spec),
            "state_read_scope": "full state used only for simulator setup; planner sees named proprioception and RGB-D",
            "arm_controller": "robosuite JointPositionController, absolute q, native torque actuators",
            "arm_kp": float(arm_kp), "arm_kd": float(arm_kd),
            "effective_arm_kp": obj._arm_controller.kp.tolist(),
            "effective_arm_kd": obj._arm_controller.kd.tolist(),
            "desired_velocity_feedforward": False,
            # BEGIN explicit every-step position-command provenance.
            "gripper_controller": "robosuite SimpleGripController, use_action_scaling=False",
            "gripper_command_frequency_hz": 1/obj._dt,
            "gripper_restored_position_ctrl": restored_ctrl.tolist(),
            "gripper_initialization": "restored native position ctrl in named joint units; command state is independent of measured q",
            # END explicit every-step position-command provenance.
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
        # BEGIN named actual state and separately named native command state.
        timestamp = float(self.native_data.time)
        return {"joint_names": self.joint_names,
                "arm_joint_names": self.arm_joint_names, "finger_joint_names": self.finger_joint_names,
                "trial_id": self._trial_id, "feedback_id": f"{self._trial_id}@native_time={timestamp:.17g}",
                "source": self._feedback_source, "timestamp_s": timestamp,
                "q": self.native_data.qpos[self._qpos_ids].copy(),
                "dq": self.native_data.qvel[self._qvel_ids].copy(),
                "base_pose": base, "tool_pose": tool, "sim_time": timestamp,
                "actuator_ctrl": {"joint_names": self.joint_names,
                    "actuator_names": self._actuator_names,
                    "values": self.native_data.ctrl[self._actuator_ids].tolist(),
                    "sim_time": timestamp, "timestamp_s": timestamp,
                    "semantics": "native actuator command state; never measured joint position"}}
        # END named actual state and separately named native command state.

    def capture_rgbd(self, camera):
        """Expose synchronized visible RGB-D and calibration, excluding labels."""
        captured = camera.capture(self.native_data)
        return {"rgb": captured["rgb"], "depth_m": captured["depth_m"],
                "calibration": deepcopy(camera.calibration), "sim_time": float(self.native_data.time)}

    # BEGIN seven-arm execution with separate fixed finger goals and actual state.
    def execute(self, trajectory, *, finger_targets, initial_feedback,
                evaluation_mode, output, step_callback=None,
                max_physics_steps=30_000, start_position_tolerance=2e-6,
                start_velocity_tolerance=2e-6):
        """Follow the exact exported curve until its original evidence barrier.

        ``step_callback(model, data, record)`` is an evaluator/camera hook after
        each physical step. Its output is never fed to the planner here. The
        runner does not certify a grasp, bypass a barrier, or advance a skill.
        The last native step can finish less than one timestep after the
        barrier's clock time, but its commanded q/dq/progress is clamped to the
        barrier itself. Finger events use their original retimed times and
        are applied on the next native step, without changing the arm cubic.
        The complete measured entrance is checked independently of all goals.
        No goal from the later curve is sent to an actuator.
        """
        if evaluation_mode not in ("accepted_plan", "diagnostic_rejected_candidate"):
            raise ValueError("Use an explicit accepted or rejected diagnostic evaluation mode")
        if tuple(trajectory.joint_names) != self.arm_joint_names:
            raise ValueError("Trajectory joint names/order must equal the explicitly controlled arm")
        if not isinstance(finger_targets, FixedFingerTargets):
            raise ValueError("An explicit FixedFingerTargets input is required")
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
        for key, names in (("joint_names", self.joint_names), ("arm_joint_names", self.arm_joint_names),
                           ("finger_joint_names", self.finger_joint_names)):
            if tuple(initial_feedback[key]) != names:
                raise ValueError("Initial actual feedback must retain every native robot joint name and its order")
        if any(not isinstance(initial_feedback[key], str) or not initial_feedback[key].strip()
               for key in ("feedback_id", "source", "reference_frame", "tool_link")):
            raise ValueError("Initial actual feedback requires explicit identity, source and calibrated frame names")
        tool_transform = np.asarray(initial_feedback["tool_transform"], dtype=float)
        if (tool_transform.shape != (4, 4) or not np.isfinite(tool_transform).all()
                or not np.allclose(tool_transform[3], [0., 0., 0., 1.], atol=1e-9, rtol=0)):
            raise ValueError("Initial feedback requires its explicit finite calibrated link-from-TCP transform")
        if (not np.isfinite(initial_feedback["timestamp_s"])
                or abs(initial_feedback["timestamp_s"]-initial_feedback["sim_time"]) > 1e-9):
            raise ValueError("Initial feedback timestamp and native simulation time must agree")
        entrance_errors = {}
        for key, shape in (("q", (len(self.joint_names),)), ("dq", (len(self.joint_names),)),
                           ("base_pose", (4, 4)), ("tool_pose", (4, 4)), ("sim_time", ())):
            expected = np.asarray(initial_feedback[key], dtype=float)
            measured = np.asarray(actual[key], dtype=float)
            if (expected.shape != shape or measured.shape != shape
                    or not np.isfinite(expected).all() or not np.isfinite(measured).all()):
                raise ValueError("Initial actual feedback must contain complete finite robot state and poses")
            entrance_errors[key] = float(np.max(np.abs(measured-expected)))
        # Native/JSON feedback is double precision. The larger user-specified
        # arm-curve tolerance below covers only float32 trajectory export;
        # it must not excuse a different measured finger entrance or clock.
        if max(entrance_errors.values()) > 1e-9:
            raise ValueError(f"Restored full robot state differs from initial actual feedback: {entrance_errors}")
        narm = len(self.arm_joint_names)
        start_errors = np.array([np.max(np.abs(actual["q"][:narm]-start_q)),
                                 np.max(np.abs(actual["dq"][:narm]-start_dq))])
        if np.any(start_errors > tolerances):
            raise ValueError(f"Curve and restored feedback have different entrances: q={start_errors[0]}, dq={start_errors[1]}; tolerances={tolerances.tolist()}")
        finger_bounds = self.native_model.actuator_ctrlrange[self._finger_actuators]
        command_events = finger_targets.bind(trajectory, finger_joint_names=self.finger_joint_names,
                                             ctrlrange=finger_bounds)
        target_positions = self.native_data.ctrl[self._finger_actuators].copy()
        if (not np.isfinite(target_positions).all() or np.any(target_positions < finger_bounds[:, 0])
                or np.any(target_positions > finger_bounds[:, 1])):
            raise ValueError("Initial native finger command must be finite and inside the native control ranges")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        started, sim_start = time.monotonic(), float(self.native_data.time)
        report = {"evaluation_mode": evaluation_mode,
            "trial_id": self._trial_id,
            "planner_accepted": evaluation_mode == "accepted_plan", "status": "running",
            "restore": self.restore_report, "next_verification": trajectory.next_verification,
            "evidence_passed": False, "full_trajectory_duration_s": trajectory.duration,
            "planned_stop_s": stop, "physics_timestep_s": self._dt,
            "start_feedback_check": {"maximum_position_error": float(start_errors[0]),
                                     "maximum_velocity_error": float(start_errors[1]),
                                     "full_actual_feedback_errors": entrance_errors,
                                     "full_actual_feedback_tolerance": 1e-9,
                                     "initial_feedback_id": initial_feedback["feedback_id"],
                                     "position_tolerance": float(tolerances[0]),
                                     "velocity_tolerance": float(tolerances[1]),
                                     "tolerance_scope": "float32 trajectory export roundoff only; does not authorize another episode entrance"},
            "barrier_discretization": "ceil native steps; final command q/dq/progress clamped to original barrier; physical clock may overrun by less than one timestep",
            "executed_steps": 0, "barrier_crossed": False,
            "gripper_command_timing": "original retimed events applied at the next native step start; less than one native timestep late",
            "finger_targets": {"joint_names": finger_targets.joint_names, "source": finger_targets.source,
                               "bound_events": deepcopy(command_events)},
            "command_record_semantics": "sample zero uses curve q(0) and restored finger ctrl; each later command_q combines the arm goal sampled at min(step end, barrier) with the finger goal applied during that step",
            "arm_curve_dq_semantics": "arm cubic derivative at curve_time, recorded only; no desired finger velocity is invented for position-target steps",
            "actuator_control_semantics": "one native torque/position ctrl row per executed attempt, timestamped at step start; actual q/dq are separately recorded after each completed step",
            "events": [], "superseded_gripper_events": [],
            "pending_verifications": [deepcopy(event) for event in trajectory.events if event["requires_evidence"] is not None],
            "unconsumed_events": deepcopy(trajectory.events)}
        controls, samples, event_index = [], [], 0
        command = "restored_ctrl"
        saturation_count = 0

        def record(elapsed, desired_arm_q, arm_curve_dq, progress):
            feedback = self.proprioception()
            return {"elapsed": float(elapsed), "sim_time": feedback["sim_time"],
                    "curve_time": min(float(elapsed), stop),
                    "command_q": np.concatenate((np.asarray(desired_arm_q), target_positions)),
                    "arm_curve_dq": np.asarray(arm_curve_dq).copy(),
                    "q": feedback["q"], "dq": feedback["dq"], "tool_pose": feedback["tool_pose"],
                    "progress": float(progress), "gripper_command": command}

        q, dq, progress = trajectory.sample(0.)
        samples.append(record(0., q, dq, progress))
        try:
            for index in range(count):
                elapsed = index * self._dt
                due = []
                while event_index < len(command_events) and command_events[event_index]["time"] <= elapsed + 1e-12:
                    due.append(command_events[event_index])
                    event_index += 1
                if due:
                    # Distinct source events can fall in one native interval.
                    # Retain every identity, and never label overwritten goals
                    # as applied to a physical step.
                    report["superseded_gripper_events"].extend(deepcopy(due[:-1]))
                    command = due[-1]["kind"]
                    target_positions = np.asarray(due[-1]["finger_positions"], dtype=float)
                self._grip_controller.set_goal(target_positions.copy())
                next_time = (index + 1) * self._dt
                q, dq, progress = trajectory.sample(min(next_time, stop))
                self._arm_controller.set_goal(q.copy())
                torque = self._arm_controller.run_controller()
                bounds = self.native_model.actuator_ctrlrange[self._arm_actuators]
                bounded = np.clip(torque, bounds[:, 0], bounds[:, 1])
                saturation_count += int(np.count_nonzero(bounded != torque))
                finger_ctrl = np.asarray(self._grip_controller.run_controller(), dtype=float)
                if not np.array_equal(finger_ctrl, target_positions):
                    raise RuntimeError("Unscaled GRIP changed an explicit native finger position target")
                if not np.isfinite(bounded).all():
                    raise FloatingPointError("Arm controller produced a non-finite native torque")
                self.native_data.ctrl[self._arm_actuators] = bounded
                self.native_data.ctrl[self._finger_actuators] = finger_ctrl
                if due:
                    report["events"].append({**due[-1], "applied_at_s": elapsed,
                        "application_delay_s": max(0., elapsed-due[-1]["time"])})
                controls.append(self.native_data.ctrl[self._actuator_ids].copy())
                self._sim.step()
                # mj_step leaves derived positions at its integration stage.
                # Forward refreshes sensor/FK evidence without another step.
                mujoco.mj_forward(self.native_model, self.native_data)
                current = record(next_time, q, dq, progress)
                report["executed_steps"] = index + 1
                if not np.isfinite(current["q"]).all() or not np.isfinite(current["dq"]).all():
                    raise FloatingPointError("Native physical robot feedback became non-finite")
                samples.append(current)
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
            command_q = np.asarray([row["command_q"] for row in samples])
            final_elapsed = float(self.native_data.time - sim_start)
            applied_ids = {(event["skill_id"], event["event_id"]) for event in report["events"]}
            barrier_command_reached = stop == 0 or (report["executed_steps"] > 0 and samples[-1]["curve_time"] >= stop-1e-12)
            report.update(elapsed_wall_s=time.monotonic()-started, elapsed_sim_s=final_elapsed,
                actual_sim_stop_s=final_elapsed, last_command_sample_s=samples[-1]["curve_time"] if controls else None,
                remaining_to_barrier_s=max(0., stop-final_elapsed),
                physical_clock_overrun_s=max(0., final_elapsed-stop),
                barrier_command_reached=barrier_command_reached,
                unapplied_gripper_events=[deepcopy(event) for event in command_events
                                          if (event["skill_id"], event["event_id"]) not in applied_ids],
                maximum_arm_tracking_error_rad=float(np.max(np.abs(actual_q[:, :narm]-command_q[:, :narm]))),
                maximum_finger_position_tracking_error=float(np.max(np.abs(actual_q[:, narm:]-command_q[:, narm:]))),
                actuator_saturated_coordinates=saturation_count,
                maximum_absolute_actual_arm_velocity=float(np.max(np.abs(np.asarray([row["dq"] for row in samples])[:, :narm]))),
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                unconsumed_events=[deepcopy(event) for event in trajectory.events
                                   if (event["skill_id"], event["event_id"]) not in applied_ids])
            np.savez_compressed(output/"physical-trajectory.npz", joint_names=np.asarray(self.joint_names),
                command_joint_names=np.asarray(self.joint_names),
                arm_joint_names=np.asarray(self.arm_joint_names), finger_joint_names=np.asarray(self.finger_joint_names),
                elapsed=np.asarray([row["elapsed"] for row in samples]),
                curve_time=np.asarray([row["curve_time"] for row in samples]),
                sim_time=np.asarray([row["sim_time"] for row in samples]),
                command_q=command_q, arm_curve_dq=np.asarray([row["arm_curve_dq"] for row in samples]),
                actual_q=actual_q, actual_dq=np.asarray([row["dq"] for row in samples]),
                actual_tool_pose=np.asarray([row["tool_pose"] for row in samples]),
                progress=np.asarray([row["progress"] for row in samples]),
                actuator_names=np.asarray(self._actuator_names),
                actuator_ctrl=np.asarray(controls).reshape(-1, len(self.joint_names)),
                control_elapsed=np.arange(len(controls))*self._dt,
                control_sim_time=sim_start+np.arange(len(controls))*self._dt,
                gripper_command=np.asarray([row["gripper_command"] for row in samples]))
            if report["status"] in ("stopped_for_evidence", "curve_finished"):
                # This is a robot-only observation. A later bridge still needs
                # a current scene observation and independently verified
                # holding evidence before constructing an attached-object model.
                feedback = self.proprioception()
                name = "barrier-feedback.json" if trajectory.next_verification is not None else "final-feedback.json"
                feedback_path = (output/name).resolve()
                feedback.update(evidence_passed=False, next_verification=trajectory.next_verification,
                    reference_frame=initial_feedback["reference_frame"], tool_link=initial_feedback["tool_link"],
                    tool_transform=tool_transform.tolist(), initial_feedback_id=initial_feedback["feedback_id"])
                serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                for key, value in feedback.items()}
                feedback_path.write_text(json.dumps(serializable, indent=2, allow_nan=False)+"\n")
                report.update(robot_feedback_path=str(feedback_path), feedback_id=feedback["feedback_id"],
                              barrier_feedback_path=str(feedback_path) if trajectory.next_verification is not None else None)
            (output/"execution-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return report
    # END seven-arm execution with separate fixed finger goals and actual state.
# END recorded scene restoration and physical control-only trajectory trial.
