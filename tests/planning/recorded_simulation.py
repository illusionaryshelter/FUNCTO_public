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
from scipy.spatial.transform import Rotation

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


# BEGIN explicit physical execution limits, separate from solver roundoff.
def _finite_array(value, shape, label, *, nonnegative=False):
    result = np.asarray(value, dtype=float)
    if (result.shape != shape or not np.isfinite(result).all()
            or (nonnegative and np.any(result < 0))):
        raise ValueError(f"{label} must be a finite array of shape {shape}")
    return result.copy()


def _feedback_json(value):
    """Keep a failed report writable; raw nonfinite feedback remains in NPZ."""
    if isinstance(value, np.ndarray):
        return _feedback_json(value.tolist())
    if isinstance(value, dict):
        return {key: _feedback_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_feedback_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass(frozen=True)
class ExecutionLimits:
    """Measured-state limits supplied by the same physical planning contract.

    No physical tolerance is inferred from controller gains, numerical solver
    tolerances, actuator targets or a previously failed execution. Finger
    bounds describe permitted *actual* positions after each original event;
    targets may lie outside them when contact prevents complete closure.
    One arm/TCP tracking band applies to the path and to command-event gates.
    Event times are owned exclusively by ``JointTrajectory``.
    """

    source: str
    joint_names: tuple[str, ...]
    arm_joint_names: tuple[str, ...]
    finger_joint_names: tuple[str, ...]
    arm_position_error: object
    arm_velocity_error: object
    tcp_position_error_m: float
    tcp_rotation_error_rad: float
    stop_arm_velocity: object
    initial_fingers: dict
    events: tuple[dict, ...]

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Physical execution limits need explicit provenance")
        for field in ("joint_names", "arm_joint_names", "finger_joint_names"):
            value = getattr(self, field)
            if (not isinstance(value, (list, tuple)) or not value
                    or any(not isinstance(name, str) or not name.strip() for name in value)
                    or len(set(value)) != len(value)):
                raise ValueError("Execution limits require complete ordered unique joint names")
            object.__setattr__(self, field, tuple(value))
        if self.joint_names != self.arm_joint_names + self.finger_joint_names:
            raise ValueError("Execution arm and fingers must partition the complete named robot order")
        for field in ("arm_position_error", "arm_velocity_error", "stop_arm_velocity"):
            value = _finite_array(getattr(self, field), (len(self.arm_joint_names),), field, nonnegative=True)
            object.__setattr__(self, field, value.tolist())
        tcp = _finite_array([self.tcp_position_error_m, self.tcp_rotation_error_rad], (2,),
                            "Physical TCP tracking tolerances", nonnegative=True)
        if tcp[1] > np.pi:
            raise ValueError("A TCP angular tracking tolerance cannot exceed pi radians")
        object.__setattr__(self, "tcp_position_error_m", float(tcp[0]))
        object.__setattr__(self, "tcp_rotation_error_rad", float(tcp[1]))

        def finger_bounds(row, *, event):
            keys = {"position_bounds", "maximum_speed"}
            if event:
                keys.update(("skill_id", "event_id"))
            if not isinstance(row, dict) or set(row) != keys:
                raise ValueError("Each actual finger interval needs explicit bounds/speed and only its original event identity")
            if event and any(not isinstance(row[key], str) or not row[key].strip()
                             for key in ("skill_id", "event_id")):
                raise ValueError("Finger intervals require original skill/event identities")
            bounds = _finite_array(row["position_bounds"], (len(self.finger_joint_names), 2),
                                   "Actual finger position bounds")
            speed = _finite_array(row["maximum_speed"], (len(self.finger_joint_names),),
                                  "Actual finger speed limits", nonnegative=True)
            if np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError("Actual finger intervals must be ordered")
            return {**deepcopy(row), "position_bounds": bounds.tolist(), "maximum_speed": speed.tolist()}

        object.__setattr__(self, "initial_fingers", finger_bounds(self.initial_fingers, event=False))
        if not isinstance(self.events, (list, tuple)):
            raise ValueError("Finger intervals require an explicit event sequence")
        rows = tuple(finger_bounds(row, event=True) for row in self.events)
        identities = [(row["skill_id"], row["event_id"]) for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError("An original event cannot have duplicate actual finger intervals")
        object.__setattr__(self, "events", rows)

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Execution limits must explicitly supply every physical limit and its named binding")
        return cls(**deepcopy(value))

    def describe(self):
        return _feedback_json({key: deepcopy(getattr(self, key)) for key in self.__dataclass_fields__})

    def bind(self, trajectory, *, joint_names, arm_joint_names, finger_joint_names,
             native_joint_ranges, named_velocity_limits):
        """Bind native hardware ranges and original events, with no new clock."""
        if (self.joint_names != tuple(joint_names) or self.arm_joint_names != tuple(arm_joint_names)
                or self.finger_joint_names != tuple(finger_joint_names)
                or self.arm_joint_names != tuple(trajectory.joint_names)):
            raise ValueError("Execution limits, curve and native robot must have identical named joint order")
        ranges = _finite_array(native_joint_ranges, (len(self.joint_names), 2), "Native joint ranges")
        if np.any(ranges[:, 0] >= ranges[:, 1]):
            raise ValueError("Every executed native scalar joint needs an explicit ordered position range")
        if not isinstance(named_velocity_limits, dict) or set(named_velocity_limits) != set(self.joint_names):
            raise ValueError("Hardware velocity limits must name exactly the complete controlled robot")
        speed = _finite_array([named_velocity_limits[name] for name in self.joint_names],
                              (len(self.joint_names),), "Named hardware velocity limits")
        if np.any(speed <= 0):
            raise ValueError("Hardware velocity limits must be finite and positive")
        commands = [event for event in trajectory.events if event["kind"] in ("open", "close", "hold")]
        identities = [(event["skill_id"], event["event_id"]) for event in commands]
        rows = {(row["skill_id"], row["event_id"]): deepcopy(row) for row in self.events}
        if len(set(identities)) != len(identities) or set(identities) != set(rows):
            raise ValueError("Actual finger intervals must match every original command event exactly")
        narm = len(self.arm_joint_names)
        for row in (self.initial_fingers, *self.events):
            bounds = np.asarray(row["position_bounds"])
            if (np.any(bounds[:, 0] < ranges[narm:, 0]) or np.any(bounds[:, 1] > ranges[narm:, 1])
                    or np.any(np.asarray(row["maximum_speed"]) > speed[narm:])):
                raise ValueError("Actual finger intervals must remain within native position and hardware velocity limits")
        if np.any(np.asarray(self.stop_arm_velocity) > speed[:narm]):
            raise ValueError("Stop-speed limits cannot exceed the named hardware limits")
        return {"joint_ranges": ranges, "maximum_speed": speed,
                "initial_fingers": deepcopy(self.initial_fingers), "event_fingers": rows}


class ExecutionGuardError(RuntimeError):
    """Actual feedback invalidated the explicit execution contract."""
# END explicit physical execution limits, separate from solver roundoff.


class RecordedSimulation:
    """One restored native scene with explicitly identified robot feedback.

    ``native_model`` / ``native_data`` belong to camera and evaluator code,
    never to planning. The only execution writes are actuator controls; model
    timestep and all generalized positions remain native throughout a trial.
    """

    @classmethod
    def restore(cls, *, scene_path, catalog_path, episode_path, frame_index,
                base_link, tcp_site, arm_joint_names, finger_joint_names,
                arm_kp, arm_kd, velocity_limits):
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
        # BEGIN preserve the fixture's explicit hardware velocity limits.
        if not isinstance(velocity_limits, dict) or set(velocity_limits) != set(obj.joint_names):
            raise ValueError("Restore requires hardware velocity limits for every named arm/finger joint")
        speed = _finite_array([velocity_limits[name] for name in obj.joint_names],
                              (len(obj.joint_names),), "Named hardware velocity limits")
        if np.any(speed <= 0):
            raise ValueError("Hardware velocity limits must be positive")
        obj._velocity_limits = dict(zip(obj.joint_names, speed.tolist()))
        # END preserve the fixture's explicit hardware velocity limits.
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
        # BEGIN isolated robot FK for physical TCP tracking checks.
        obj._prepare_reference_fk()
        # END isolated robot FK for physical TCP tracking checks.
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

    # BEGIN native kinematics only; reference evaluation never changes physics.
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
    # END native kinematics only; reference evaluation never changes physics.

    # BEGIN measured execution checks and non-integrating abort preparation.
    def _execution_violations(self, feedback, reference_q, reference_dq, reference_tcp,
                              *, limits, binding, finger_state, expected_sim_time,
                              require_stop=False):
        """Return explicit named violations; neither clip nor repair feedback."""
        failures = []

        def add(kind, name, measured, **allowed):
            failures.append({"kind": kind, "name": name, "measured": measured, **allowed})

        for field, expected in (("joint_names", self.joint_names),
                                ("arm_joint_names", self.arm_joint_names),
                                ("finger_joint_names", self.finger_joint_names)):
            if tuple(feedback.get(field, ())) != expected:
                add("feedback_identity", field, feedback.get(field), expected=list(expected))
        for field, expected in (("trial_id", self._trial_id), ("source", self._feedback_source)):
            if feedback.get(field) != expected:
                add("feedback_identity", field, feedback.get(field), expected=expected)
        for field in ("sim_time", "timestamp_s"):
            value = feedback.get(field)
            if (not isinstance(value, (int, float, np.floating)) or not np.isfinite(value)
                    or abs(float(value)-expected_sim_time) > 1e-8):
                # This is the existing native clock's numerical allowance, not
                # an execution delay or a physical tracking tolerance.
                add("feedback_clock", field, value, expected=expected_sim_time)
        current_time = float(self.native_data.time)
        expected_id = f"{self._trial_id}@native_time={current_time:.17g}"
        if feedback.get("feedback_id") != expected_id:
            add("feedback_identity", "feedback_id", feedback.get("feedback_id"), expected=expected_id)
        actual = {}
        for field in ("q", "dq"):
            value = np.asarray(feedback.get(field), dtype=float)
            if value.shape != (len(self.joint_names),):
                add("feedback_shape", field, list(value.shape), expected=[len(self.joint_names)])
                continue
            actual[field] = value
            for index in np.flatnonzero(~np.isfinite(value)):
                add("nonfinite_feedback", f"{field}:{self.joint_names[index]}", float(value[index]))
        pose = np.asarray(feedback.get("tool_pose"), dtype=float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-9, rtol=0)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7, rtol=0)
                or abs(np.linalg.det(pose[:3, :3])-1.) > 1e-7):
            add("feedback_pose", "tool_pose", pose.tolist())
        if failures:
            return failures

        q, dq = actual["q"], actual["dq"]
        ranges, speed = binding["joint_ranges"], binding["maximum_speed"]
        for index in np.flatnonzero((q < ranges[:, 0]) | (q > ranges[:, 1])):
            add("native_joint_position", self.joint_names[index], float(q[index]),
                lower=float(ranges[index, 0]), upper=float(ranges[index, 1]))
        for index in np.flatnonzero(np.abs(dq) > speed):
            add("hardware_joint_velocity", self.joint_names[index], float(dq[index]),
                maximum_absolute=float(speed[index]))
        narm = len(self.arm_joint_names)
        for field, measured, expected, threshold in (
                ("arm_position_tracking", q[:narm], reference_q, np.asarray(limits.arm_position_error)),
                ("arm_velocity_tracking", dq[:narm], reference_dq, np.asarray(limits.arm_velocity_error))):
            for index in np.flatnonzero(np.abs(measured-expected) > threshold):
                add(field, self.arm_joint_names[index], float(measured[index]),
                    expected=float(expected[index]), maximum_error=float(threshold[index]))
        if require_stop:
            threshold = np.asarray(limits.stop_arm_velocity)
            for index in np.flatnonzero(np.abs(dq[:narm]) > threshold):
                add("planned_stop_velocity", self.arm_joint_names[index], float(dq[index]),
                    maximum_absolute=float(threshold[index]))
        finger_bounds = np.asarray(finger_state["position_bounds"])
        finger_speed = np.asarray(finger_state["maximum_speed"])
        for index in np.flatnonzero((q[narm:] < finger_bounds[:, 0]) | (q[narm:] > finger_bounds[:, 1])):
            add("actual_finger_position", self.finger_joint_names[index], float(q[narm+index]),
                lower=float(finger_bounds[index, 0]), upper=float(finger_bounds[index, 1]))
        for index in np.flatnonzero(np.abs(dq[narm:]) > finger_speed):
            add("actual_finger_velocity", self.finger_joint_names[index], float(dq[narm+index]),
                maximum_absolute=float(finger_speed[index]))
        position_error = float(np.linalg.norm(pose[:3, 3]-reference_tcp[:3, 3]))
        rotation_error = float(Rotation.from_matrix(reference_tcp[:3, :3].T @ pose[:3, :3]).magnitude())
        for field, error, bound in (("tcp_position_m", position_error, limits.tcp_position_error_m),
                                    ("tcp_rotation_rad", rotation_error, limits.tcp_rotation_error_rad)):
            if error > bound:
                add("tcp_tracking", field, error, maximum_error=bound)
        return failures

    def _prepare_abort_hold(self, feedback, finger_positions, binding, *, expected_sim_time):
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
            ranges = binding["joint_ranges"]
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
    # END measured execution checks and non-integrating abort preparation.

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
    def execute(self, trajectory, *, finger_targets, initial_feedback, execution_limits,
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
        No goal from the later curve is sent to an actuator. Physical tracking
        and actual finger intervals require an explicit ``ExecutionLimits``;
        rejection stops simulation before another event or physical step.
        """
        if evaluation_mode not in ("accepted_plan", "diagnostic_rejected_candidate"):
            raise ValueError("Use an explicit accepted or rejected diagnostic evaluation mode")
        if tuple(trajectory.joint_names) != self.arm_joint_names:
            raise ValueError("Trajectory joint names/order must equal the explicitly controlled arm")
        if not isinstance(finger_targets, FixedFingerTargets):
            raise ValueError("An explicit FixedFingerTargets input is required")
        if not isinstance(execution_limits, ExecutionLimits):
            raise ValueError("Explicit physical ExecutionLimits are required before execution")
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
        # BEGIN one physical contract shared by path checks and event gates.
        if not self.native_model.jnt_limited[self._joint_ids].all():
            raise ValueError("Execution requires explicit native ranges for every controlled scalar joint")
        limit_binding = execution_limits.bind(trajectory,
            joint_names=self.joint_names, arm_joint_names=self.arm_joint_names,
            finger_joint_names=self.finger_joint_names,
            native_joint_ranges=self.native_model.jnt_range[self._joint_ids],
            named_velocity_limits=self._velocity_limits)
        finger_state = deepcopy(limit_binding["initial_fingers"])
        # END one physical contract shared by path checks and event gates.
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
        # BEGIN physical guard provenance and actual-event evidence.
        report.update(execution_limits=execution_limits.describe(),
            named_hardware_velocity_limits=deepcopy(self._velocity_limits),
            native_joint_position_ranges=limit_binding["joint_ranges"].tolist(),
            execution_guard_scope="Actual robot tracking and declared hand intervals; not collision, contact or Holding certification",
            feedback_failure=None, event_gate_checks=[], abort_hold=None)
        # END physical guard provenance and actual-event evidence.
        controls, samples, event_index = [], [], 0
        command = "restored_ctrl"
        saturation_count = 0

        def record(elapsed, desired_arm_q, arm_curve_dq, progress, feedback=None):
            feedback = self.proprioception() if feedback is None else feedback
            return {"elapsed": float(elapsed), "sim_time": feedback["sim_time"],
                    "curve_time": min(float(elapsed), stop),
                    "command_q": np.concatenate((np.asarray(desired_arm_q), target_positions)),
                    "arm_curve_dq": np.asarray(arm_curve_dq).copy(),
                    "q": feedback["q"], "dq": feedback["dq"], "tool_pose": feedback["tool_pose"],
                    "reference_tool_pose": self.reference_tool_pose(desired_arm_q),
                    "feedback": deepcopy(feedback), "finger_state": deepcopy(finger_state),
                    "progress": float(progress), "gripper_command": command}

        # BEGIN reject measured violations before changing an event's command.
        def guard(feedback, *, curve_time, physical_elapsed, point, event=None):
            reference_q, reference_dq, _ = trajectory.sample(curve_time)
            reference_tcp = self.reference_tool_pose(reference_q)
            barrier = trajectory.next_verification
            requires_stop = (bool(event.get("stop", False)) if event is not None else
                             bool(barrier is not None and barrier.get("stop", False)
                                  and abs(curve_time-stop) <= 1e-12))
            failures = self._execution_violations(feedback, reference_q, reference_dq, reference_tcp,
                limits=execution_limits, binding=limit_binding, finger_state=finger_state,
                expected_sim_time=sim_start+physical_elapsed, require_stop=requires_stop)
            check = {"point": point, "curve_time_s": float(curve_time),
                     "physical_elapsed_s": float(physical_elapsed), "passed": not failures}
            if event is not None:
                check.update(skill_id=event["skill_id"], event_id=event["event_id"], kind=event["kind"])
                report["event_gate_checks"].append(deepcopy(check))
            if failures:
                report["status"] = "feedback_rejected"
                report["feedback_failure"] = {**check, "violations": failures,
                    "actual_feedback": deepcopy(feedback), "reference_arm_q": reference_q.tolist(),
                    "reference_arm_dq": reference_dq.tolist(), "reference_tool_pose": reference_tcp.tolist(),
                    "actual_finger_interval": deepcopy(finger_state),
                    "last_applied_finger_goal": target_positions.tolist(),
                    "rejected_event": None if event is None else deepcopy(event)}
                raise ExecutionGuardError(f"Actual feedback violates {point}: {failures[0]['kind']} / {failures[0]['name']}")
        # END reject measured violations before changing an event's command.

        q, dq, progress = trajectory.sample(0.)
        samples.append(record(0., q, dq, progress))
        try:
            guard(samples[-1]["feedback"], curve_time=0., physical_elapsed=0., point="initial_state")
            for index in range(count):
                elapsed = index * self._dt
                # Re-read once at each step boundary. The same snapshot gates
                # any due event; a failed gate cannot mutate a finger goal.
                before = self.proprioception()
                guard(before, curve_time=min(elapsed, stop), physical_elapsed=elapsed, point="before_step")
                due = []
                while event_index < len(command_events) and command_events[event_index]["time"] <= elapsed + 1e-12:
                    due.append(command_events[event_index])
                    event_index += 1
                next_command = command
                next_finger_positions = target_positions.copy()
                next_finger_state = finger_state
                if due:
                    # Distinct source events can fall in one native interval.
                    # Retain every identity, and never label overwritten goals
                    # as applied to a physical step.
                    guard(before, curve_time=float(due[-1]["time"]), physical_elapsed=elapsed,
                          point="command_event", event=due[-1])
                    next_command = due[-1]["kind"]
                    next_finger_positions = np.asarray(due[-1]["finger_positions"], dtype=float)
                    # The pre-event state belongs to the old interval. Only an
                    # actually applied command selects its declared new one.
                    next_finger_state = deepcopy(limit_binding["event_fingers"][(due[-1]["skill_id"], due[-1]["event_id"])])
                self._grip_controller.set_goal(next_finger_positions.copy())
                next_time = (index + 1) * self._dt
                q, dq, progress = trajectory.sample(min(next_time, stop))
                self._arm_controller.set_goal(q.copy())
                torque = self._arm_controller.run_controller()
                bounds = self.native_model.actuator_ctrlrange[self._arm_actuators]
                bounded = np.clip(torque, bounds[:, 0], bounds[:, 1])
                saturation_count += int(np.count_nonzero(bounded != torque))
                finger_ctrl = np.asarray(self._grip_controller.run_controller(), dtype=float)
                if not np.array_equal(finger_ctrl, next_finger_positions):
                    raise RuntimeError("Unscaled GRIP changed an explicit native finger position target")
                if not np.isfinite(bounded).all():
                    raise FloatingPointError("Arm controller produced a non-finite native torque")
                self.native_data.ctrl[self._arm_actuators] = bounded
                self.native_data.ctrl[self._finger_actuators] = finger_ctrl
                target_positions, finger_state, command = next_finger_positions, next_finger_state, next_command
                if due:
                    report["superseded_gripper_events"].extend(deepcopy(due[:-1]))
                    report["events"].append({**due[-1], "applied_at_s": elapsed,
                        "application_delay_s": max(0., elapsed-due[-1]["time"])})
                controls.append(self.native_data.ctrl[self._actuator_ids].copy())
                self._sim.step()
                # mj_step leaves derived positions at its integration stage.
                # Forward refreshes sensor/FK evidence without another step.
                mujoco.mj_forward(self.native_model, self.native_data)
                current = record(next_time, q, dq, progress)
                report["executed_steps"] = index + 1
                samples.append(current)
                # Preserve the completed physical sample, including independent
                # evaluator contacts, before rejecting its measured feedback.
                try:
                    guard(current["feedback"], curve_time=min(next_time, stop),
                          physical_elapsed=next_time, point="after_step")
                finally:
                    if step_callback is not None:
                        try:
                            step_callback(self.native_model, self.native_data, current)
                        except Exception as observer_error:
                            if report["feedback_failure"] is None:
                                raise
                            # Nonfinite feedback can also invalidate evaluator
                            # input. Retain the primary measured guard failure
                            # and its physical sample rather than masking it.
                            report["failed_sample_observer_error"] = f"{type(observer_error).__name__}: {observer_error}"
            report["status"] = "stopped_for_evidence" if trajectory.next_verification is not None else "curve_finished"
        except Exception as error:
            if report["status"] != "feedback_rejected":
                report["status"] = "execution_error"
            report["error"] = f"{type(error).__name__}: {error}"
            # Preparing controller output neither writes native ctrl nor runs
            # another physical step, and never applies the rejected event.
            failure_feedback = (report["feedback_failure"]["actual_feedback"]
                                if report["feedback_failure"] is not None else self.proprioception())
            report["abort_hold"] = self._prepare_abort_hold(failure_feedback, target_positions, limit_binding,
                expected_sim_time=sim_start+report["executed_steps"]*self._dt)
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
                reference_tool_pose=np.asarray([row["reference_tool_pose"] for row in samples]),
                actual_finger_position_bounds=np.asarray([row["finger_state"]["position_bounds"] for row in samples]),
                actual_finger_maximum_speed=np.asarray([row["finger_state"]["maximum_speed"] for row in samples]),
                progress=np.asarray([row["progress"] for row in samples]),
                actuator_names=np.asarray(self._actuator_names),
                actuator_ctrl=np.asarray(controls).reshape(-1, len(self.joint_names)),
                control_elapsed=np.arange(len(controls))*self._dt,
                control_sim_time=sim_start+np.arange(len(controls))*self._dt,
                gripper_command=np.asarray([row["gripper_command"] for row in samples]))
            if report["status"] in ("stopped_for_evidence", "curve_finished", "feedback_rejected", "execution_error"):
                # This is a robot-only observation. A later bridge still needs
                # a current scene observation and independently verified
                # holding evidence before constructing an attached-object model.
                feedback = self.proprioception()
                completed = report["status"] in ("stopped_for_evidence", "curve_finished")
                name = ("failure-feedback.json" if not completed else
                        "barrier-feedback.json" if trajectory.next_verification is not None else "final-feedback.json")
                feedback_path = (output/name).resolve()
                feedback.update(evidence_passed=False, next_verification=trajectory.next_verification,
                    reference_frame=initial_feedback["reference_frame"], tool_link=initial_feedback["tool_link"],
                    tool_transform=tool_transform.tolist(), initial_feedback_id=initial_feedback["feedback_id"])
                feedback["execution_status"] = report["status"]
                serializable = _feedback_json(feedback)
                feedback_path.write_text(json.dumps(serializable, indent=2, allow_nan=False)+"\n")
                report.update(robot_feedback_path=str(feedback_path), feedback_id=feedback["feedback_id"],
                              barrier_feedback_path=str(feedback_path) if completed and trajectory.next_verification is not None else None)
            (output/"execution-report.json").write_text(json.dumps(_feedback_json(report), indent=2, allow_nan=False)+"\n")
        return report
    # END seven-arm execution with separate fixed finger goals and actual state.
# END recorded scene restoration and physical control-only trajectory trial.
