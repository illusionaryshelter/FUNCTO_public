"""Execute a named arm curve against explicit actual-state contracts.

This module owns the original trajectory clock, command-event gates and first
verification barrier. A control backend owns actuator I/O and native stepping;
an explicitly supplied independent geometry checker owns model acceptance.
Recording restoration, datasets and planning-scene construction stay outside
this production execution boundary.
"""

# BEGIN production trajectory execution and required physical contracts.
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .gripper_trajectory import FingerTrajectory

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
    targets may lie outside them when contact prevents complete closure. These
    intervals are runtime protection limits, never a planning geometry sweep.
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
        if any(not isinstance(event.get("progress"), (int, float, np.floating))
               or not np.isfinite(event["progress"]) or event["progress"] < 0 for event in commands):
            raise ValueError("Actual hand intervals require each command's original finite progress anchor")
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
        # BEGIN actual protection intervals are no longer future geometry.
        # Their old monotone enlargement rule existed only to cover a planning
        # full-stroke union. Timed nominal geometry now has its own saved
        # reference. A new runtime interval is checked against real feedback
        # after its command activates; it never proves the actuator converged.
        # END runtime limits do not construct a hand or certify future motion.
        if np.any(np.asarray(self.stop_arm_velocity) > speed[:narm]):
            raise ValueError("Stop-speed limits cannot exceed the named hardware limits")
        return {"joint_ranges": ranges, "maximum_speed": speed,
                "initial_fingers": deepcopy(self.initial_fingers), "event_fingers": rows}


class ExecutionGuardError(RuntimeError):
    """Actual feedback invalidated the explicit execution contract."""
# END explicit physical execution limits, separate from solver roundoff.


class TrajectoryExecutor:
    """Follow the exported cubic using one concrete backend and geometry gate.

    No alternate event clock, interpolation or missing-checker fallback is
    provided. The backend exposes actual named robot state independently from
    actuator control. Geometry validation is mandatory for every execution
    mode, including explicitly labelled rejected-candidate diagnostics.
    """

    def __init__(self, backend, *, geometry_checker):
        self.backend = backend
        self.geometry_checker = geometry_checker
        if (not isinstance(getattr(geometry_checker, "name", None), str)
                or not geometry_checker.name.strip()
                or tuple(getattr(geometry_checker, "joint_names", ())) != backend.joint_names
                or any(not callable(getattr(geometry_checker, name, None))
                       for name in ("describe", "preflight", "check_state"))):
            raise ValueError("Execution requires a named independent geometry checker for the complete robot order")
        self._geometry_description = deepcopy(geometry_checker.describe())
        if (not isinstance(self._geometry_description, dict)
                or not isinstance(self._geometry_description.get("pair_order"), list)):
            raise ValueError("The independent geometry checker must declare its stable pair order")
        self._geometry_pair_count = len(self._geometry_description["pair_order"])

    def _geometry_result(self, method, *args, **kwargs):
        """Require an explicit verdict; checker errors reject physical control.

        A malformed result is a checker failure, never an omitted check. The
        caller retains this failure alongside the same actual feedback used
        for tracking checks, then stops before issuing another command.
        """
        try:
            result = getattr(self.geometry_checker, method)(*args, **kwargs)
            if (not isinstance(result, dict) or type(result.get("passed")) is not bool
                    or not isinstance(result.get("violations"), list)
                    or any(not isinstance(row, dict) or not {"kind", "name"} <= set(row)
                           for row in result["violations"])
                    or result["passed"] != (len(result["violations"]) == 0)):
                raise ValueError("A geometry verdict needs a boolean passed and consistent named violations")
            result = deepcopy(result)
            if method == "check_state":
                distances = _finite_array(result["distances_m"], (self._geometry_pair_count,),
                                          "Independent geometry distances in declared pair order")
                result["distances_m"] = distances.tolist()
                result["distance_query_completed"] = True
            return result
        except Exception as error:
            return {"passed": False, "distance_query_completed": False,
                "violations": [{"kind": "geometry_checker_error",
                "name": self.geometry_checker.name, "measured": f"{type(error).__name__}: {error}"}]}

    def _execution_violations(self, feedback, reference_q, reference_dq, reference_tcp,
                              *, limits, binding, finger_state, expected_sim_time,
                              require_stop=False):
        """Return explicit named violations; neither clip nor repair feedback."""
        backend = self.backend
        failures = []

        def add(kind, name, measured, **allowed):
            failures.append({"kind": kind, "name": name, "measured": measured, **allowed})

        # BEGIN actuator force has its own named native order.
        for field, expected in (("joint_names", backend.joint_names),
                                ("arm_joint_names", backend.arm_joint_names),
                                ("finger_joint_names", backend.finger_joint_names),
                                ("actuator_names", backend.actuator_names)):
            if tuple(feedback.get(field, ())) != expected:
                add("feedback_identity", field, feedback.get(field), expected=list(expected))
        # END do not infer actuator order from joint labels.
        for field, expected in (("trial_id", backend.trial_id), ("source", backend.feedback_source)):
            if feedback.get(field) != expected:
                add("feedback_identity", field, feedback.get(field), expected=expected)
        for field in ("sim_time", "timestamp_s"):
            value = feedback.get(field)
            if (not isinstance(value, (int, float, np.floating)) or not np.isfinite(value)
                    or abs(float(value)-expected_sim_time) > 1e-8):
                # This is the existing native clock's numerical allowance, not
                # an execution delay or a physical tracking tolerance.
                add("feedback_clock", field, value, expected=expected_sim_time)
        current_time = backend.sim_time
        if not np.isfinite(current_time) or abs(current_time-expected_sim_time) > 1e-8:
            add("feedback_clock", "native_data.time", current_time, expected=expected_sim_time)
        expected_id = backend.feedback_id
        if feedback.get("feedback_id") != expected_id:
            add("feedback_identity", "feedback_id", feedback.get("feedback_id"), expected=expected_id)
        actual = {}
        # BEGIN require finite native telemetry without inventing acceleration
        # or force-spike thresholds. Hardware/tracking guards remain separate.
        for field, names in (("q", backend.joint_names), ("dq", backend.joint_names),
                             ("qacc", backend.joint_names), ("actuator_force", backend.actuator_names)):
            value = np.asarray(feedback.get(field), dtype=float)
            if value.shape != (len(names),):
                add("feedback_shape", field, list(value.shape), expected=[len(names)])
                continue
            actual[field] = value
            for index in np.flatnonzero(~np.isfinite(value)):
                add("nonfinite_feedback", f"{field}:{names[index]}", float(value[index]))
        # END retain raw measurements; finite telemetry is not spike acceptance.
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
            add("native_joint_position", backend.joint_names[index], float(q[index]),
                lower=float(ranges[index, 0]), upper=float(ranges[index, 1]))
        for index in np.flatnonzero(np.abs(dq) > speed):
            add("hardware_joint_velocity", backend.joint_names[index], float(dq[index]),
                maximum_absolute=float(speed[index]))
        narm = len(backend.arm_joint_names)
        for field, measured, expected, threshold in (
                ("arm_position_tracking", q[:narm], reference_q, np.asarray(limits.arm_position_error)),
                ("arm_velocity_tracking", dq[:narm], reference_dq, np.asarray(limits.arm_velocity_error))):
            for index in np.flatnonzero(np.abs(measured-expected) > threshold):
                add(field, backend.arm_joint_names[index], float(measured[index]),
                    expected=float(expected[index]), maximum_error=float(threshold[index]))
        if require_stop:
            threshold = np.asarray(limits.stop_arm_velocity)
            for index in np.flatnonzero(np.abs(dq[:narm]) > threshold):
                add("planned_stop_velocity", backend.arm_joint_names[index], float(dq[index]),
                    maximum_absolute=float(threshold[index]))
        finger_bounds = np.asarray(finger_state["position_bounds"])
        finger_speed = np.asarray(finger_state["maximum_speed"])
        for index in np.flatnonzero((q[narm:] < finger_bounds[:, 0]) | (q[narm:] > finger_bounds[:, 1])):
            add("actual_finger_position", backend.finger_joint_names[index], float(q[narm+index]),
                lower=float(finger_bounds[index, 0]), upper=float(finger_bounds[index, 1]))
        for index in np.flatnonzero(np.abs(dq[narm:]) > finger_speed):
            add("actual_finger_velocity", backend.finger_joint_names[index], float(dq[narm+index]),
                maximum_absolute=float(finger_speed[index]))
        position_error = float(np.linalg.norm(pose[:3, 3]-reference_tcp[:3, 3]))
        rotation_error = float(Rotation.from_matrix(reference_tcp[:3, :3].T @ pose[:3, :3]).magnitude())
        for field, error, bound in (("tcp_position_m", position_error, limits.tcp_position_error_m),
                                    ("tcp_rotation_rad", rotation_error, limits.tcp_rotation_error_rad)):
            if error > bound:
                add("tcp_tracking", field, error, maximum_error=bound)
        return failures


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
        barrier itself. Finger events use the archived reference's original
        retimed times and native activation ticks, without changing the arm
        cubic. The same saved C1 finger reference is sampled at each arm query.
        The complete measured entrance is checked independently of all goals.
        No goal from the later curve is sent to an actuator. Physical tracking
        and actual finger intervals require an explicit ``ExecutionLimits``;
        rejection stops simulation before another event or physical step.
        """
        backend = self.backend
        if evaluation_mode not in ("accepted_plan", "diagnostic_rejected_candidate", "diagnostic_control_reference"):
            raise ValueError("Use an explicit planned-candidate or control-reference evaluation mode")
        if tuple(trajectory.joint_names) != backend.arm_joint_names:
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
        count = int(np.ceil(stop/backend.timestep - 1e-10))
        if type(max_physics_steps) is not int or not 0 <= count <= max_physics_steps <= 100_000:
            raise ValueError("Trajectory exceeds the explicit bounded local physics-step budget")
        tolerances = np.asarray([start_position_tolerance, start_velocity_tolerance], float)
        if not np.isfinite(tolerances).all() or np.any(tolerances < 0):
            raise ValueError("Start-feedback numerical tolerances must be finite and nonnegative; zero requires exact equality")
        start_q, start_dq, _ = trajectory.sample(0.)
        actual = backend.proprioception()
        for key, names in (("joint_names", backend.joint_names), ("arm_joint_names", backend.arm_joint_names),
                           ("finger_joint_names", backend.finger_joint_names)):
            if tuple(initial_feedback[key]) != names:
                raise ValueError("Initial actual feedback must retain every native robot joint name and its order")
        if any(not isinstance(initial_feedback[key], str) or not initial_feedback[key].strip()
               for key in ("feedback_id", "source", "reference_frame", "tool_link")):
            raise ValueError("Initial actual feedback requires explicit identity, source and calibrated frame names")
        tool_transform = np.asarray(initial_feedback["tool_transform"], dtype=float)
        if (tool_transform.shape != (4, 4) or not np.isfinite(tool_transform).all()
                or not np.allclose(tool_transform[3], [0., 0., 0., 1.], atol=1e-9, rtol=0)):
            raise ValueError("Initial feedback requires its explicit finite calibrated link-from-TCP transform")
        if (any(initial_feedback[key] != actual[key] for key in ("reference_frame", "tool_link"))
                or not np.allclose(tool_transform, actual["tool_transform"], atol=1e-9, rtol=0)):
            raise ValueError("Initial feedback must retain the backend's actual reference frame and native TCP binding")
        if (not np.isfinite(initial_feedback["timestamp_s"])
                or abs(initial_feedback["timestamp_s"]-initial_feedback["sim_time"]) > 1e-9):
            raise ValueError("Initial feedback timestamp and native simulation time must agree")
        entrance_errors = {}
        for key, shape in (("q", (len(backend.joint_names),)), ("dq", (len(backend.joint_names),)),
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
        narm = len(backend.arm_joint_names)
        start_errors = np.array([np.max(np.abs(actual["q"][:narm]-start_q)),
                                 np.max(np.abs(actual["dq"][:narm]-start_dq))])
        if np.any(start_errors > tolerances):
            raise ValueError(f"Curve and restored feedback have different entrances: q={start_errors[0]}, dq={start_errors[1]}; tolerances={tolerances.tolist()}")
        finger_bounds = backend.finger_ctrlrange
        command_events = finger_targets.bind(trajectory, finger_joint_names=backend.finger_joint_names,
                                             ctrlrange=finger_bounds)
        # BEGIN execute the exact archived finger reference, never a new ramp.
        finger_reference = trajectory.finger_trajectory
        if command_events and not isinstance(finger_reference, FingerTrajectory):
            raise ValueError("Original finger events require the exact saved planning FingerTrajectory")
        if finger_reference is not None and not isinstance(finger_reference, FingerTrajectory):
            raise ValueError("A supplied finger reference must be the validated original FingerTrajectory")
        finger_reference_report = None
        if finger_reference is not None:
            speeds = np.array([backend.velocity_limits[name] for name in backend.finger_joint_names])
            finger_reference_report = finger_reference.validate_execution(
                trajectory, command_events, initial_feedback=initial_feedback, timestep=backend.timestep,
                position_limits=backend.joint_ranges[narm:].T,
                velocity_limits=np.stack((-speeds, speeds)))
            command_events = deepcopy(list(finger_reference.events))
        # END missing event references have no position-step fallback.
        # BEGIN one physical contract shared by path checks and event gates.
        limit_binding = execution_limits.bind(trajectory,
            joint_names=backend.joint_names, arm_joint_names=backend.arm_joint_names,
            finger_joint_names=backend.finger_joint_names,
            native_joint_ranges=backend.joint_ranges,
            named_velocity_limits=backend.velocity_limits)
        finger_state = deepcopy(limit_binding["initial_fingers"])
        # END one physical contract shared by path checks and event gates.
        target_positions = backend.finger_control
        if (not np.isfinite(target_positions).all() or np.any(target_positions < finger_bounds[:, 0])
                or np.any(target_positions > finger_bounds[:, 1])):
            raise ValueError("Initial native finger command must be finite and inside the native control ranges")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        started, sim_start = time.monotonic(), float(backend.sim_time)
        report = {"evaluation_mode": evaluation_mode,
            "reference_origin": ("explicit_control_diagnostic" if evaluation_mode == "diagnostic_control_reference"
                                 else "planned_candidate"),
            "trial_id": backend.trial_id,
            "planner_accepted": evaluation_mode == "accepted_plan", "status": "running",
            "backend": backend.describe(), "next_verification": trajectory.next_verification,
            "evidence_passed": False, "full_trajectory_duration_s": trajectory.duration,
            "planned_stop_s": stop, "physics_timestep_s": backend.timestep,
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
            "gripper_command_timing": "original retimed event identities activate at their saved ceil-to-native-step tick; every subsequent position command samples the same archived finger reference and arm clock",
            "finger_targets": {"joint_names": finger_targets.joint_names, "source": finger_targets.source,
                               "bound_events": deepcopy(command_events)},
            "finger_reference": None if finger_reference is None else finger_reference.describe(),
            "finger_reference_bounds": finger_reference_report,
            "finger_reference_actual_tracking_certified": False,
            "command_record_semantics": "sample zero uses curve q(0) and restored finger ctrl; each later command_q combines the arm goal sampled at min(step end, barrier) with the finger goal applied during that step",
            "arm_curve_dq_semantics": "arm cubic derivative at curve_time, recorded independently of actual velocity",
            "finger_reference_semantics": "reference_finger_q/dq are the archived C1 nominal command at curve_time, not actual feedback; absent references are NaN and explicitly unprovided; physical contact can prevent target attainment without proving a tracking failure or Holding",
            "actuator_control_semantics": "one native torque/position ctrl row per executed attempt, timestamped at step start; actual q/dq are separately recorded after each completed step",
            # BEGIN preserve the two clocks and meanings needed for dynamics review.
            "actual_dynamics_semantics": "actual_qacc and actual_actuator_force use the initial refreshed native state and every completed post-step mj_forward state at elapsed/sim_time; qacc is native forward acceleration, not differentiated dq; force is native actuator output, not mixed-unit ctrl",
            "arm_torque_semantics": "requested_arm_torque and applied_arm_torque are pre-step controller demand and clipped arm ctrl in arm_actuator_names order at control_elapsed/control_sim_time; adjacent-row increments may be computed from these original arrays",
            "actuator_spike_assessment": "Raw native force and command arrays retained; no force/acceleration spike threshold or all-actuator pass is inferred",
            # END command clipping and dynamics remain distinct evidence.
            "events": [], "superseded_gripper_events": [],
            "pending_verifications": [deepcopy(event) for event in trajectory.events if event["requires_evidence"] is not None],
            "unconsumed_events": deepcopy(trajectory.events)}
        # BEGIN physical guard provenance and actual-event evidence.
        report.update(execution_limits=execution_limits.describe(),
            named_hardware_velocity_limits=deepcopy(backend.velocity_limits),
            native_joint_position_ranges=limit_binding["joint_ranges"].tolist(),
            execution_guard_scope="Actual robot tracking, declared hand intervals and independent named model checks; no scene, continuous-path or Holding certification",
            geometry_checker=deepcopy(self._geometry_description), geometry_preflight=None,
            geometry_check_counts={"initial_state": 0, "before_step": 0, "command_event": 0, "after_step": 0},
            last_geometry_check=None,
            feedback_failure=None, event_gate_checks=[], abort_hold=None)
        # END physical guard provenance and actual-event evidence.
        controls, requested_torques, saturated_torques, samples, event_index = [], [], [], [], 0
        command = "restored_ctrl"
        # BEGIN this counter excludes native finger actuator force limiting.
        arm_control_saturation_count = 0
        # END explicit arm control clipping scope.

        def record(elapsed, desired_arm_q, arm_curve_dq, progress, feedback=None):
            feedback = backend.proprioception() if feedback is None else feedback
            # BEGIN retain reference and actual fingers as separate evidence.
            reference_fingers = (finger_reference.sample(min(float(elapsed), stop))
                                 if finger_reference is not None else
                                 (np.full(len(backend.finger_joint_names), np.nan),
                                  np.full(len(backend.finger_joint_names), np.nan)))
            # END unknown desired velocity is not manufactured from a target.
            return {"elapsed": float(elapsed), "sim_time": feedback["sim_time"],
                    "curve_time": min(float(elapsed), stop),
                    "command_q": np.concatenate((np.asarray(desired_arm_q), target_positions)),
                    "arm_curve_dq": np.asarray(arm_curve_dq).copy(),
                    "reference_finger_q": reference_fingers[0].copy(),
                    "reference_finger_dq": reference_fingers[1].copy(),
                    "q": feedback["q"], "dq": feedback["dq"], "tool_pose": feedback["tool_pose"],
                    # BEGIN synchronized native dynamics, including failed samples.
                    "qacc": feedback["qacc"], "actuator_force": feedback["actuator_force"],
                    # END these are measured telemetry rather than curve derivatives.
                    "reference_tool_pose": backend.reference_tool_pose(desired_arm_q),
                    "feedback": deepcopy(feedback), "finger_state": deepcopy(finger_state),
                    "geometry_check": None,
                    "progress": float(progress), "gripper_command": command}

        # BEGIN reject measured violations before changing an event's command.
        def guard(feedback, *, curve_time, physical_elapsed, point, event=None, sample=None):
            reference_q, reference_dq, _ = trajectory.sample(curve_time)
            reference_tcp = backend.reference_tool_pose(reference_q)
            barrier = trajectory.next_verification
            requires_stop = (bool(event.get("stop", False)) if event is not None else
                             bool(barrier is not None and barrier.get("stop", False)
                                  and abs(curve_time-stop) <= 1e-12))
            failures = self._execution_violations(feedback, reference_q, reference_dq, reference_tcp,
                limits=execution_limits, binding=limit_binding, finger_state=finger_state,
                expected_sim_time=sim_start+physical_elapsed, require_stop=requires_stop)
            geometry = None
            if not failures:
                geometry = self._geometry_result("check_state", deepcopy(feedback),
                    stage=point, trajectory_time=float(curve_time))
                report["geometry_check_counts"][point] += 1
                report["last_geometry_check"] = deepcopy(geometry)
                failures.extend(geometry["violations"])
            if sample is not None:
                sample["geometry_check"] = deepcopy(geometry)
            check = {"point": point, "curve_time_s": float(curve_time),
                     "physical_elapsed_s": float(physical_elapsed), "passed": not failures}
            if event is not None:
                check.update(skill_id=event["skill_id"], event_id=event["event_id"], kind=event["kind"])
                check["geometry_check"] = deepcopy(geometry)
                report["event_gate_checks"].append(deepcopy(check))
            if failures:
                report["status"] = "feedback_rejected"
                report["feedback_failure"] = {**check, "violations": failures,
                    "actual_feedback": deepcopy(feedback), "reference_arm_q": reference_q.tolist(),
                    "reference_arm_dq": reference_dq.tolist(), "reference_tool_pose": reference_tcp.tolist(),
                    "actual_finger_interval": deepcopy(finger_state),
                    "geometry_check": deepcopy(geometry),
                    "last_applied_finger_goal": target_positions.tolist(),
                    "rejected_event": None if event is None else deepcopy(event)}
                raise ExecutionGuardError(f"Actual feedback violates {point}: {failures[0]['kind']} / {failures[0]['name']}")
        # END reject measured violations before changing an event's command.

        q, dq, progress = trajectory.sample(0.)
        samples.append(record(0., q, dq, progress))
        try:
            # Preflight checks the declared model/reference contract without
            # stepping or preparing an actuator goal. A diagnostic reference
            # receives exactly the same mandatory acceptance gate.
            preflight = self._geometry_result("preflight", trajectory, deepcopy(initial_feedback), execution_limits)
            report["geometry_preflight"] = deepcopy(preflight)
            if not preflight["passed"]:
                report["status"] = "feedback_rejected"
                report["feedback_failure"] = {"point": "geometry_preflight", "curve_time_s": 0.,
                    "physical_elapsed_s": 0., "passed": False, "violations": deepcopy(preflight["violations"]),
                    "actual_feedback": deepcopy(samples[-1]["feedback"]), "geometry_check": deepcopy(preflight),
                    "reference_arm_q": q.tolist(), "reference_arm_dq": dq.tolist(),
                    "reference_tool_pose": backend.reference_tool_pose(q).tolist(),
                    "actual_finger_interval": deepcopy(finger_state),
                    "last_applied_finger_goal": target_positions.tolist(), "rejected_event": None}
                raise ExecutionGuardError("Independent geometry preflight rejected the execution reference")
            guard(samples[-1]["feedback"], curve_time=0., physical_elapsed=0., point="initial_state", sample=samples[-1])
            for index in range(count):
                elapsed = index * backend.timestep
                # Re-read once at each step boundary. The same snapshot gates
                # any due event; a failed gate cannot mutate a finger goal.
                before = backend.proprioception()
                guard(before, curve_time=min(elapsed, stop), physical_elapsed=elapsed, point="before_step")
                due = []
                while event_index < len(command_events) and command_events[event_index]["activation_step"] <= index:
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
                    # The pre-event state belongs to the old interval. Only an
                    # actually applied command selects its declared new one.
                    next_finger_state = deepcopy(limit_binding["event_fingers"][(due[-1]["skill_id"], due[-1]["event_id"])])
                next_time = (index + 1) * backend.timestep
                q, dq, progress = trajectory.sample(min(next_time, stop))
                # BEGIN one synchronous 7+2 command query, with independent feedback.
                # A future event's ramp starts at its saved activation tick.
                # At that boundary its q/dq remain C1 with the prior hold, so
                # sampling this step's endpoint cannot issue an early closure.
                if finger_reference is not None:
                    next_finger_positions, _ = finger_reference.sample(min(next_time, stop))
                # END native GRIP receives the same named reference as planning.
                control = backend.apply_control(q, next_finger_positions)
                # BEGIN aggregate only the backend's named arm clipping count.
                arm_control_saturation_count += control["arm_control_saturated_coordinates"]
                # END native finger force limits are measured separately.
                target_positions, finger_state, command = next_finger_positions, next_finger_state, next_command
                if due:
                    report["superseded_gripper_events"].extend(deepcopy(due[:-1]))
                    report["events"].append({**due[-1], "applied_at_s": elapsed,
                        "application_delay_s": max(0., elapsed-due[-1]["time"])})
                controls.append(control["actuator_ctrl"].copy())
                requested_torques.append(control["requested_arm_torque"].copy())
                saturated_torques.append(control["arm_torque_saturated"].copy())
                backend.step()
                current = record(next_time, q, dq, progress)
                report["executed_steps"] = index + 1
                samples.append(current)
                # Preserve the completed physical sample, including independent
                # evaluator contacts, before rejecting its measured feedback.
                try:
                    guard(current["feedback"], curve_time=min(next_time, stop),
                          physical_elapsed=next_time, point="after_step", sample=current)
                finally:
                    if step_callback is not None:
                        try:
                            step_callback(backend.native_model, backend.native_data, current)
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
                                if report["feedback_failure"] is not None else backend.proprioception())
            report["abort_hold"] = backend.prepare_abort_hold(failure_feedback, target_positions,
                expected_sim_time=sim_start+report["executed_steps"]*backend.timestep)
            raise
        finally:
            actual_q = np.asarray([row["q"] for row in samples])
            command_q = np.asarray([row["command_q"] for row in samples])
            final_elapsed = float(backend.sim_time - sim_start)
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
                maximum_finger_command_position_difference=float(np.max(np.abs(actual_q[:, narm:]-command_q[:, narm:]))),
                # BEGIN preserve the arm-only meaning in the report field.
                arm_control_saturated_coordinates=arm_control_saturation_count,
                # END never describe this as all-actuator saturation.
                maximum_absolute_actual_arm_velocity=float(np.max(np.abs(np.asarray([row["dq"] for row in samples])[:, :narm]))),
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                unconsumed_events=[deepcopy(event) for event in trajectory.events
                                   if (event["skill_id"], event["event_id"]) not in applied_ids])
            np.savez_compressed(output/"physical-trajectory.npz", joint_names=np.asarray(backend.joint_names),
                command_joint_names=np.asarray(backend.joint_names),
                arm_joint_names=np.asarray(backend.arm_joint_names), finger_joint_names=np.asarray(backend.finger_joint_names),
                elapsed=np.asarray([row["elapsed"] for row in samples]),
                curve_time=np.asarray([row["curve_time"] for row in samples]),
                sim_time=np.asarray([row["sim_time"] for row in samples]),
                command_q=command_q, arm_curve_dq=np.asarray([row["arm_curve_dq"] for row in samples]),
                reference_finger_provided=np.asarray(finger_reference is not None),
                reference_finger_q=np.asarray([row["reference_finger_q"] for row in samples]),
                reference_finger_dq=np.asarray([row["reference_finger_dq"] for row in samples]),
                actual_q=actual_q, actual_dq=np.asarray([row["dq"] for row in samples]),
                # BEGIN keep every actual dynamics sample aligned with elapsed/sim_time.
                actual_qacc=np.asarray([row["qacc"] for row in samples]),
                actual_actuator_force=np.asarray([row["actuator_force"] for row in samples]),
                # END no filtering or finite-difference reconstruction of native outputs.
                actual_tool_pose=np.asarray([row["tool_pose"] for row in samples]),
                reference_tool_pose=np.asarray([row["reference_tool_pose"] for row in samples]),
                actual_finger_position_bounds=np.asarray([row["finger_state"]["position_bounds"] for row in samples]),
                actual_finger_maximum_speed=np.asarray([row["finger_state"]["maximum_speed"] for row in samples]),
                # A failed physical check or checker exception can prevent a
                # distance query. NaN plus checked=False explicitly preserves
                # this absence; an omitted query is never saved as a pass.
                geometry_checked=np.asarray([row["geometry_check"] is not None
                    and row["geometry_check"]["distance_query_completed"] for row in samples], dtype=bool),
                geometry_passed=np.asarray([row["geometry_check"] is not None
                    and row["geometry_check"]["passed"] for row in samples], dtype=bool),
                geometry_distances_m=np.asarray([row["geometry_check"]["distances_m"]
                    if row["geometry_check"] is not None and row["geometry_check"]["distance_query_completed"]
                    else np.full(self._geometry_pair_count, np.nan) for row in samples]),
                progress=np.asarray([row["progress"] for row in samples]),
                actuator_names=np.asarray(backend.actuator_names),
                actuator_ctrl=np.asarray(controls).reshape(-1, len(backend.joint_names)),
                requested_arm_torque=np.asarray(requested_torques).reshape(-1, narm),
                # BEGIN explicit applied arm torque has no gripper position units.
                arm_actuator_names=np.asarray(backend.actuator_names[:narm]),
                applied_arm_torque=np.asarray(controls).reshape(-1, len(backend.joint_names))[:, :narm],
                # END arm commands use the existing pre-step control timestamps.
                arm_torque_saturated=np.asarray(saturated_torques, dtype=bool).reshape(-1, narm),
                control_elapsed=np.arange(len(controls))*backend.timestep,
                control_sim_time=sim_start+np.arange(len(controls))*backend.timestep,
                gripper_command=np.asarray([row["gripper_command"] for row in samples]))
            if report["status"] in ("stopped_for_evidence", "curve_finished", "feedback_rejected", "execution_error"):
                # This is a robot-only observation. A later bridge still needs
                # a current scene observation and independently verified
                # holding evidence before constructing an attached-object model.
                feedback = backend.proprioception()
                completed = report["status"] in ("stopped_for_evidence", "curve_finished")
                name = ("failure-feedback.json" if not completed else
                        "barrier-feedback.json" if trajectory.next_verification is not None else "final-feedback.json")
                feedback_path = (output/name).resolve()
                feedback.update(evidence_passed=False, next_verification=trajectory.next_verification,
                    initial_feedback_id=initial_feedback["feedback_id"])
                feedback["execution_status"] = report["status"]
                serializable = _feedback_json(feedback)
                feedback_path.write_text(json.dumps(serializable, indent=2, allow_nan=False)+"\n")
                report.update(robot_feedback_path=str(feedback_path), feedback_id=feedback["feedback_id"],
                              barrier_feedback_path=str(feedback_path) if completed and trajectory.next_verification is not None else None)
            (output/"execution-report.json").write_text(json.dumps(_feedback_json(report), indent=2, allow_nan=False)+"\n")
        return report

# END production trajectory execution and required physical contracts.
