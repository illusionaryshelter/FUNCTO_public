"""Execute a named arm curve against explicit actual-state contracts.

This module owns the original trajectory clock, command-event gates and first
verification barrier. A control backend owns actuator I/O and native stepping;
an explicitly supplied independent geometry checker owns model acceptance.
Recording restoration, datasets and planning-scene construction stay outside
this production execution boundary.
"""

# BEGIN production trajectory execution and required physical contracts.
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .gripper_trajectory import FingerTrajectory, _activation
# BEGIN physical time checks share the planner's explicit occurrence contract.
from .planning_problem import (EventTimeConstraint, PoseEventRequirement, TimeBindingError,
    event_occurrence_key, resolve_event_window)
from .motion import _pose_array
# END an original name alone never identifies a retried action.


# BEGIN shared functional clock for runtime evidence and independent result grading.
def pose_goal_schedule(trajectory, progress, timestep):
    """Return the first original progress time and its strict native due tick.

    A terminal progress plateau starts at its first matching knot. The
    physical observation uses the first device tick covering that time;
    one-ULP future times cannot borrow the preceding tick. This function
    supplies timing only and never constructs an owner pose or goal witness.
    """
    matches = np.flatnonzero(np.abs(np.asarray(trajectory.progress_s)-progress) <= 1e-12)
    first_u = (float(trajectory.progress_u[matches[0]]) if matches.size else
               float(np.interp(progress, trajectory.progress_s, trajectory.progress_u)))
    scheduled = (first_u*trajectory.duration
                 if progress <= float(trajectory.progress_s[-1])+1e-12 else float("inf"))
    return scheduled, (_activation(scheduled, timestep)[1] if np.isfinite(scheduled) else float("inf"))
# END both consumers keep the original first-arrival clock without sharing observations.


def _timing_checks(event, check_target, time_s, history, *, goal_id=None, condition_id=None):
    """Compare one actual witness with its own original physical-time windows.

    The caller supplies a measured activation/arrival/acquisition timestamp.
    No reference event time or planning prediction can stand in for it here.
    Unresolved origins remain U and retain their precise missing requirement.
    """
    checks = []
    for index, value in enumerate(event.get("timing", ())):
        constraint = value if isinstance(value, EventTimeConstraint) else EventTimeConstraint(**value)
        if (constraint.check_target != check_target
                or (check_target == "arrival" and tuple(constraint.goal_id) != tuple(goal_id or ()))
                or (check_target == "effect" and constraint.condition_id != condition_id)):
            continue
        check = {"skill_id": event["skill_id"], "event_id": event["event_id"],
                 "occurrence": event["occurrence"], "check_target": check_target,
                 "constraint_index": index, "time_s": float(time_s), "source": deepcopy(constraint.source)}
        try:
            lower, upper = resolve_event_window(constraint, history)
            check.update(window_s=[lower, upper], value=("T" if
                (lower is None or time_s >= lower-1e-9) and (upper is None or time_s <= upper+1e-9) else "F"))
        except TimeBindingError as error:
            check.update(value="U", reason=str(error), origin_event=deepcopy(constraint.origin_event))
        checks.append(check)
    return checks


# BEGIN actual task accuracy is independent of engineering reference tracking.
def _actual_pose_goal(requirement, feedback, observation):
    """Measure one original functional allowed set from applicable actual poses.

    # BEGIN robot FK estimates are not independent Cartesian measurements.
    A declared fixed tool/world goal can use the robot TCP estimate from
    measured joints and the bound robot model/calibration. This does not
    certify physical Cartesian accuracy or replace a current owner observation.
    # END the original functional tolerances remain unchanged.
    An owner-relative target instead needs the current owner's
    observation. A carried material frame needs either its observed owner pose
    or a currently verified holding relation. Neither a frozen owner model nor
    a candidate attachment supplies a missing runtime measurement.
    """
    # BEGIN provenance separates joint-derived estimates from owner observations.
    measured = {"value": "U", "source": "Executor compares applicable current robot FK estimates and supplied owner observations with the original task allowed set",
        "actual_function_pose": None, "actual_target_pose": None,
        "position_error_m": None, "rotation_error_rad": None, "measurement_sources": []}
    # END this comparison does not claim independent Cartesian sensing.
    relation = requirement.actual_relation
    poses = {} if observation is None else observation.get("scene_owner_poses", {})
    conditions = {} if observation is None else observation.get("scene_condition_evidence", {})
    missing = []

    def owner_pose(identity):
        if identity not in poses:
            missing.append(f"No current observed pose for owner {identity}")
            return None
        pose = _pose_array(poses[identity], f"Current observed owner pose {identity}")
        measured["measurement_sources"].append({"kind": "current_owner_pose", "owner_id": identity,
            "source": observation["source"], "timestamp_s": float(observation["timestamp_s"]), "pose": pose.tolist()})
        return pose

    def actual_tool():
        # BEGIN retain the estimator provenance alongside the actual joint clock.
        pose = _pose_array(feedback["tool_pose"], "Robot TCP estimate from measured joints")
        measured["measurement_sources"].append({"kind": "actual_robot_tcp", "source": feedback["source"],
            "pose_source": feedback["tool_pose_source"],
            "independent_observation": feedback["tool_pose_is_independent_observation"],
            "timestamp_s": float(feedback["timestamp_s"]), "feedback_id": feedback["feedback_id"], "pose": pose.tolist()})
        # END FK agreement alone never supplies a Cartesian sensor measurement.
        return pose

    if relation is not None and "function_owner_id" in relation:
        if relation["function_owner_id"] in poses:
            function = owner_pose(relation["function_owner_id"]) @ np.asarray(relation["T_owner_function"])
        elif (relation.get("attachment_condition_id") is not None
              and conditions.get(relation["attachment_condition_id"]) is True):
            function = actual_tool() @ requirement.tool_from_function
            measured["measurement_sources"].append({"kind": "current_verified_attachment",
                "condition_id": relation["attachment_condition_id"], "source": observation["source"],
                "timestamp_s": float(observation["timestamp_s"]), "value": True})
        else:
            function = None
            missing.append("No current owner pose or verified actual holding relation for the material function frame")
    else:
        function = actual_tool() @ requirement.tool_from_function
    if relation is not None and "target_owner_id" in relation:
        owner = owner_pose(relation["target_owner_id"])
        target = None if owner is None else owner @ np.asarray(relation["T_owner_target"])
    else:
        target = requirement.target
    measured["actual_function_pose"] = None if function is None else function.tolist()
    measured["actual_target_pose"] = None if target is None else target.tolist()
    if missing:
        measured["reason"] = "; ".join(missing)
        return measured
    position = float(np.linalg.norm(function[:3, 3]-target[:3, 3]))
    rotation = float(Rotation.from_matrix(target[:3, :3].T @ function[:3, :3]).magnitude())
    measured.update(position_error_m=position, rotation_error_rad=rotation,
        value="T" if position <= requirement.position_tolerance_m and rotation <= requirement.rotation_tolerance_rad else "F")
    return measured
# END a completion label cannot replace these original metric comparisons.

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
        for event in commands:
            event_occurrence_key(event)
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
    # BEGIN native scalar types retain their JSON value, including NumPy bool.
    # Convert before descending so scalar tuples/arrays still use the same
    # feedback policy. Unsupported values remain errors; nothing is stringified.
    if isinstance(value, np.generic):
        value = value.item()
    # END finite values preserve their original boolean/numeric meaning.
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
    # BEGIN whole-device stop confirmation uses a declared hand criterion.
    stop_finger_velocity: object
    # END position goals never prove stopped fingers.
    initial_fingers: dict
    events: tuple[dict, ...]
    # BEGIN one explicit device observation cadence, separate from robot control.
    observation_policy: dict | None
    # END None declares no online observation channel for the bound static model.

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
        # BEGIN explicit measured finger stop threshold in named joint order.
        finger_stop = _finite_array(self.stop_finger_velocity, (len(self.finger_joint_names),),
                                    "stop_finger_velocity", nonnegative=True)
        object.__setattr__(self, "stop_finger_velocity", finger_stop.tolist())
        # END no stopping tolerance is inferred from allowed path velocity.
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
        # BEGIN validate freshness requirements before any command is issued.
        policy = self.observation_policy
        if policy is not None:
            if (not isinstance(policy, dict) or set(policy) != {
                    "source", "period_s", "maximum_age_s", "required_fields"}
                    or not isinstance(policy["source"], str) or not policy["source"].strip()
                    or not np.isfinite([policy["period_s"], policy["maximum_age_s"]]).all()
                    or policy["period_s"] <= 0 or policy["maximum_age_s"] < 0
                    or not isinstance(policy["required_fields"], (tuple, list))
                    or not set(policy["required_fields"]) <= {
                        "binding_valid", "scene_condition_evidence", "scene_owner_poses"}
                    or len(set(policy["required_fields"])) != len(policy["required_fields"])):
                raise ValueError("Observation policy needs explicit cadence, age bound, required fields and source")
            object.__setattr__(self, "observation_policy", deepcopy(policy))
        # END future effect requirements remain separate from current feedback needs.

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
        for event in commands:
            event_occurrence_key(event)
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
        if np.any(np.r_[self.stop_arm_velocity, self.stop_finger_velocity] > speed):
            raise ValueError("Stop-speed limits cannot exceed the named hardware limits")
        return {"joint_ranges": ranges, "maximum_speed": speed,
                "initial_fingers": deepcopy(self.initial_fingers), "event_fingers": rows}


class ExecutionGuardError(RuntimeError):
    """Actual feedback invalidated the explicit execution contract."""
# END explicit physical execution limits, separate from solver roundoff.


# BEGIN derive remaining device inputs from actual application, never elapsed time.
def continuation_execution_inputs(continuation, finger_targets, execution_limits, *, rebind=None):
    """Select remaining or explicitly retried original device policies.

    The request retains one immutable full device policy. A retry chooses a
    new occurrence of an existing original event, never new actuator targets
    or looser physical limits. The actual last applied interval is preserved
    independently of the candidate's event clock and occurrence numbering.
    """
    if not isinstance(finger_targets, FixedFingerTargets) or not isinstance(execution_limits, ExecutionLimits):
        raise ValueError("Continuation needs the original explicit device inputs")
    original = continuation["original_binding"]
    if continuation["candidate_id"] != original["candidate_id"]:
        raise ValueError("Continuation must retain its exact candidate identity")
    supplied = {"finger_targets": {"joint_names": list(finger_targets.joint_names),
        "source": finger_targets.source, "events": list(deepcopy(finger_targets.events))},
        "execution_limits": execution_limits.describe()}
    policy = continuation["device_policy"]
    # BEGIN continuation retains the complete controller and result-obligation policy.
    if (not isinstance(policy, dict) or set(policy) != {
            "finger_targets", "execution_limits", "controller_policy", "result_only_goal_ids"}
            or original.get("device_policy") != policy
            or supplied != {key: policy[key] for key in supplied}):
        raise ValueError("A continuation or retry cannot change the original full device policy")
    # END selecting remaining commands never upgrades a historical position-only controller.
    commands = {(row["skill_id"], row["event_id"]): row for row in policy["finger_targets"]["events"]}
    pending = {(row["skill_id"], row["event_id"]) for row in continuation["remaining_events"]
               if row["kind"] in ("open", "close", "hold")}
    if not pending <= set(commands):
        raise ValueError("Remaining commands must belong to the original request device policy")
    if rebind is not None:
        if not isinstance(rebind, dict) or rebind.get("kind") not in ("remaining", "retry") or not rebind.get("source"):
            raise ValueError("Device rebind needs an explicit remaining/retry decision and source")
        retries = rebind.get("retry_event_ids", ())
        if any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in retries):
            raise ValueError("Retried command identities must be original skill/event pairs")
        if rebind["kind"] == "remaining" and retries:
            raise ValueError("Remaining work cannot implicitly replay an applied command")
        retry_ids = {tuple(row) for row in retries}
        source_events = {(row["skill_id"], row["event_id"]) for row in continuation["request_ledger"]["original_events"]}
        if not retry_ids <= source_events:
            raise ValueError("Retried events must belong to the immutable original request")
        pending.update(retry_ids & set(commands))
    limits = deepcopy(policy["execution_limits"])
    limits["initial_fingers"] = deepcopy(continuation["active_finger_interval"])
    limits["events"] = [row for row in limits["events"] if (row["skill_id"], row["event_id"]) in pending]
    targets = tuple(row for row in policy["finger_targets"]["events"]
                    if (row["skill_id"], row["event_id"]) in pending)
    return (FixedFingerTargets(finger_targets.joint_names, finger_targets.source, targets),
            ExecutionLimits.from_mapping(limits))

# END same-request continuation does not replay or relabel a completed command.


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
        # BEGIN expose this invocation's existing diagnostics after an error.
        # The caller receives the same report object, without rereading a
        # potentially large archive or copying its complete geometry evidence.
        self.last_execution_report = None
        # END this is diagnostic state, never a new execution authorization.
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

        # BEGIN actuator command identity retains its own named native order.
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
        # BEGIN online supervision consumes only available named proprioception.
        # Native qacc and actuator_force belong to independent evaluation;
        # missing or invalid simulator diagnostics cannot stop robot execution.
        for field, names in (("q", backend.joint_names), ("dq", backend.joint_names)):
            value = np.asarray(feedback.get(field), dtype=float)
            if value.shape != (len(names),):
                add("feedback_shape", field, list(value.shape), expected=[len(names)])
                continue
            actual[field] = value
            for index in np.flatnonzero(~np.isfinite(value)):
                add("nonfinite_feedback", f"{field}:{names[index]}", float(value[index]))
        # END physical position, speed and tracking limits remain unchanged.
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


    def execute(self, result, *, finger_targets, initial_feedback, execution_limits,
                output, step_callback=None, observation_callback=None,
                max_physics_steps=30_000, start_position_tolerance=2e-6,
                start_velocity_tolerance=2e-6, max_hold_steps=250):
        """Follow the exact bound curve within its authorized execution scope.

        # BEGIN independent evidence collection has no control authority.
        ``step_callback(model, data, record)`` is an independent evaluator/camera
        sink for the initial state and each completed normal execution step.
        Its first exception is recorded and disables that sink; it cannot reject
        execution, trigger a hold, or mask an online guard failure. Its return
        value is never consumed. Holds have their own physical accounting.
        # END only the separately declared online observer supplies semantics.
        The
        ``observation_callback(feedback, curve_time)`` supplies current allowed
        scene/effect evidence. Only a true declared prerequisite can extend
        the authorized prefix, after independent checks of the same suffix.
        The last native step can finish less than one timestep after the
        barrier's clock time, but its commanded q/dq/progress is clamped to the
        barrier itself. Finger events use the archived reference's original
        bound times and native activation ticks, without changing the arm
        cubic. The same saved C1 finger reference is sampled at each arm query.
        The complete measured entrance is checked independently of all goals.
        No goal from the later curve is sent to an actuator. Physical tracking
        and actual finger intervals require an explicit ``ExecutionLimits``;
        rejection sends bounded device hold and records its measured result.
        # BEGIN one physical allowance includes every attempted native step.
        ``max_physics_steps`` bounds normal playback and both hold paths
        together. Failed or uncertain attempts remain charged; confirmed
        native advancement is recorded separately and owns physical time.
        # END a hold is not an additional allowance outside the request.
        """
        # BEGIN a rejected new binding cannot expose an earlier trial report.
        self.last_execution_report = None
        # END this invocation publishes its own object only after creation.
        # BEGIN formal candidate handoff: labels cannot authorize commands.
        from .execution_binding import (check_result, nominal_finger_sampler,
            validate_controller_policy, validate_result_only_goal_ids)
        binding = check_result(result, execution_limits=execution_limits)
        trajectory = result.trajectory
        if not callable(getattr(self.geometry_checker, "bind_scene", None)):
            raise ValueError("Execution requires independent geometry with the bound scene scope")
        self.geometry_checker.bind_scene(binding["scene"],
            progress_at_time=lambda t: float(trajectory.sample(t)[3]),
            nominal_fingers=nominal_finger_sampler(binding["scene"]))
        if type(max_hold_steps) is not int or not 1 <= max_hold_steps <= 10_000:
            raise ValueError("Emergency hold needs a positive bounded device step budget")
        if observation_callback is not None and not callable(observation_callback):
            raise ValueError("Current scene/effect observations require an explicit callable")
        if observation_callback is not None and execution_limits.observation_policy is None:
            raise ValueError("An observation callback needs its declared device cadence and freshness policy")
        backend = self.backend
        # END formal candidate handoff.
        if tuple(trajectory.joint_names) != backend.arm_joint_names:
            raise ValueError("Trajectory joint names/order must equal the explicitly controlled arm")
        if not isinstance(finger_targets, FixedFingerTargets):
            raise ValueError("An explicit FixedFingerTargets input is required")
        if not isinstance(execution_limits, ExecutionLimits):
            raise ValueError("Explicit physical ExecutionLimits are required before execution")
        # BEGIN bind calibration as well as q/dq to the actual robot entrance.
        planned = binding["problem"]
        for key in ("reference_frame", "tool_link"):
            if planned.get(key) != initial_feedback.get(key):
                raise ValueError(f"Planning and actual feedback have different {key}")
        for key in ("base_pose", "tool_transform"):
            expected = np.asarray(planned.get(key), dtype=float)
            actual_frame = np.asarray(initial_feedback.get(key), dtype=float)
            if (expected.shape != (4, 4) or actual_frame.shape != (4, 4)
                    or not np.isfinite(expected).all() or not np.isfinite(actual_frame).all()
                    or not np.allclose(expected, actual_frame, atol=1e-9, rtol=0)):
                raise ValueError(f"Planning and actual feedback have different {key} calibration")
        binding["execution_limits"] = execution_limits.describe()
        # BEGIN only an explicitly rebound controller policy authorizes this backend.
        policy = binding.get("device_policy")
        if not isinstance(policy, dict) or set(policy) != {
                "finger_targets", "execution_limits", "controller_policy", "result_only_goal_ids"}:
            raise ValueError("Execution requires an explicit complete device rebind; historical position-only authorization is invalid")
        declared_controller = validate_controller_policy(policy["controller_policy"], joint_names=backend.arm_joint_names)
        actual_controller = validate_controller_policy(backend.describe().get("controller_policy"), joint_names=backend.arm_joint_names)
        if (declared_controller != actual_controller
                or declared_controller["native_timestep_s"] != float(backend.timestep)):
            raise ValueError("The bound controller policy differs from the actual backend mode, gains or clock")
        result_goal_ids = {tuple(identity) for identity in validate_result_only_goal_ids(
            binding, trajectory, policy["result_only_goal_ids"])}
        # END there is no implicit q-only migration or global optional-check switch.
        original_targets = FixedFingerTargets.from_mapping(policy["finger_targets"])
        original_limits = ExecutionLimits.from_mapping(policy["execution_limits"])
        original_rows = {(row["skill_id"], row["event_id"]): row for row in original_targets.events}
        original_intervals = {(row["skill_id"], row["event_id"]): row for row in original_limits.events}
        current_limits = execution_limits.describe()
        if (finger_targets.joint_names != original_targets.joint_names or finger_targets.source != original_targets.source
                or any(row != original_rows.get((row["skill_id"], row["event_id"])) for row in finger_targets.events)
                or any(row != original_intervals.get((row["skill_id"], row["event_id"])) for row in execution_limits.events)
                or any(value != policy["execution_limits"][key] for key, value in current_limits.items()
                       if key not in {"events", "initial_fingers"})
                or current_limits["initial_fingers"] != binding.get("active_finger_interval", original_limits.initial_fingers)):
            raise ValueError("A new occurrence cannot change its original actuator policy or physical protection limits")
        # END candidate subsets never erase a previously consumed command's policy.
        # BEGIN the same request retains its cumulative native-step budget.
        if type(max_physics_steps) is not int or not 0 <= max_physics_steps <= 100_000:
            raise ValueError("Execution requires a bounded native physics-step budget")
        physics_budget = binding.get("physics_budget", max_physics_steps)
        prior_physics_steps = binding.get("consumed_physics_steps", 0)
        if (type(physics_budget) is not int or not 0 <= physics_budget <= 100_000
                or type(prior_physics_steps) is not int or prior_physics_steps < 0
                or binding.get("remaining_physics_steps", max(0, physics_budget-prior_physics_steps))
                   != max(0, physics_budget-prior_physics_steps)):
            raise ValueError("Continuation physics budget lost its original cumulative accounting")
        max_physics_steps = min(max_physics_steps, max(0, physics_budget-prior_physics_steps))
        binding.update(physics_budget=physics_budget, consumed_physics_steps=prior_physics_steps,
                       remaining_physics_steps=max(0, physics_budget-prior_physics_steps))
        # END smaller execution allowances cannot enlarge the original request.
        result.execution_binding = deepcopy(binding)
        # END feedback/backend agreement alone cannot validate an old world/TCP binding.
        stop = float(trajectory.executable_until)
        arrays = (trajectory.times, trajectory.position, trajectory.velocity,
                  trajectory.progress_u, trajectory.progress_s, trajectory._curve.c)
        if not all(np.isfinite(value).all() for value in arrays) or not 0 <= stop <= trajectory.duration:
            raise ValueError("Only a complete finite exported trajectory may be physically evaluated")
        prefix_count = int(np.ceil(stop/backend.timestep - 1e-10))
        if type(max_physics_steps) is not int or not 0 <= prefix_count <= max_physics_steps <= 100_000:
            raise ValueError("Trajectory exceeds the explicit bounded local physics-step budget")
        count = min(max_physics_steps, int(np.ceil(trajectory.duration/backend.timestep - 1e-10)))
        tolerances = np.asarray([start_position_tolerance, start_velocity_tolerance], float)
        if not np.isfinite(tolerances).all() or np.any(tolerances < 0):
            raise ValueError("Start-feedback numerical tolerances must be finite and nonnegative; zero requires exact equality")
        # BEGIN all arm references share one analytic position/velocity/acceleration query.
        start_q, start_dq, start_ddq, _ = trajectory.sample(0.)
        _finite_array(start_ddq, start_q.shape, "Initial arm reference acceleration")
        # END a missing or nonfinite derivative is never a position-only fallback.
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
        report = {"reference_origin": "bound_planning_result",
            "candidate_id": binding["candidate_id"],
            "execution_binding": binding,
            "trial_id": backend.trial_id,
            "planner_accepted": True, "status": "running",
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
            "barrier_discretization": "ceil native steps; final command q/dq/ddq/progress clamped to original barrier; physical clock may overrun by less than one timestep",
            "executed_steps": 0, "barrier_crossed": False,
            "gripper_command_timing": "original retimed event identities activate at their saved ceil-to-native-step tick; every subsequent position command samples the same archived finger reference and arm clock",
            "finger_targets": {"joint_names": finger_targets.joint_names, "source": finger_targets.source,
                               "bound_events": deepcopy(command_events)},
            "finger_reference": None if finger_reference is None else finger_reference.describe(),
            "finger_reference_bounds": finger_reference_report,
            "finger_reference_actual_tracking_certified": False,
            "command_record_semantics": "sample zero uses curve q(0) and restored finger ctrl; each later command_q combines the arm goal sampled at min(step end, barrier) with the finger goal applied during that step",
            "arm_curve_dq_semantics": "arm cubic derivative at curve_time, recorded independently of actual velocity",
            # BEGIN analytic reference acceleration is a command quantity.
            "arm_curve_ddq_semantics": "Same Hermite query as reference q/dq at curve_time; right internal-knot acceleration and last-piece left acceleration at the endpoint",
            # END reference acceleration jumps do not add a C2 requirement.
            "finger_reference_semantics": "reference_finger_q/dq are the archived C1 nominal command at curve_time, not actual feedback; absent references are NaN and explicitly unprovided; physical contact can prevent target attainment without proving a tracking failure or Holding",
            "actuator_control_semantics": "one native torque/position ctrl row per executed attempt, timestamped at step start; actual q/dq are separately recorded after each completed step",
            # BEGIN online archives contain robot observations and issued commands.
            "robot_pose_semantics": "base_pose is known fixed robot calibration; tool_pose/actual_tool_pose are model FK estimates from measured joint q, not independent Cartesian observations or certified physical accuracy",
            "arm_torque_semantics": "requested_arm_torque and applied_arm_torque are pre-step controller demand and clipped arm ctrl in arm_actuator_names order at control_elapsed/control_sim_time; adjacent-row increments may be computed from these original arrays",
            "actuator_spike_assessment": "Issued command arrays retained; native acceleration/force diagnostics belong only to independent evaluation and confer no online acceptance",
            # END simulator-only dynamics are absent from online feedback/archives.
            # BEGIN evidence sinks report loss without creating a robot failure.
            "independent_evidence_sink": {"configured": step_callback is not None,
                "completed": False, "disabled": False, "first_error": None,
                "successful_callbacks": 0, "attempted_callbacks": 0},
            # END completeness is finalized against the actual normal-step ledger.
            "events": [], "superseded_gripper_events": [],
            "pending_verifications": [deepcopy(event) for event in trajectory.events if event["requires_evidence"] is not None],
            "unconsumed_events": deepcopy(trajectory.events)}
        # BEGIN retain this same object through normal return and exceptional cleanup.
        self.last_execution_report = report
        # END finally updates the shared report with actual hold and continuation facts.
        # BEGIN physical guard provenance and actual-event evidence.
        report.update(execution_limits=execution_limits.describe(),
            named_hardware_velocity_limits=deepcopy(backend.velocity_limits),
            native_joint_position_ranges=limit_binding["joint_ranges"].tolist(),
            execution_guard_scope="Measured joint tracking, robot FK estimates and independently queried bound robot/scene geometry; FK consistency is a model check, and current owner/effect observations remain separate",
            geometry_checker=deepcopy(self._geometry_description), geometry_preflight=None,
            geometry_check_counts={"initial_state": 0, "before_step": 0, "command_event": 0, "after_step": 0},
            last_geometry_check=None,
            feedback_failure=None, event_gate_checks=[], abort_hold=None,
            dependency_hold=None, verified_events=[], dependency_outcomes=[], extended_preflights=[],
            evaluated_events=[], stopped_events=[], reached_events=[],
            completed_goal_ids=deepcopy(binding.get("completed_goal_ids", [])),
            goal_evidence=deepcopy(binding.get("goal_evidence", [])))
        # BEGIN one local allowance covers normal commands and both hold paths.
        normal_accounting = {"charged_steps": 0, "physical_steps": 0,
                             "step_accounting_exact": True}
        report["normal_step_accounting"] = normal_accounting
        report["last_confirmed_curve_time_s"] = 0.

        def step_accounting():
            """Separate conservative budget charges from confirmed movement."""
            phases = [normal_accounting, *(report[key] for key in ("dependency_hold", "abort_hold")
                                           if report[key] is not None)]
            charged = sum(row["charged_steps"] for row in phases)
            return {"charged_steps": charged,
                    "physical_steps": sum(row["physical_steps"] for row in phases),
                    "step_accounting_exact": all(row["step_accounting_exact"] for row in phases),
                    "execution_step_limit": max_physics_steps,
                    "remaining_charged_steps": max(0, max_physics_steps-charged),
                    "scope": "This execute invocation; normal playback and both hold paths"}

        def run_hold(key, feedback, *, expected_sim_time):
            """Allocate only the same invocation's unspent native attempts.

            Reserve the complete supplied hold allowance until its owned
            result is available. If the hold itself unexpectedly raises or
            loses its report, no unknown work becomes a free retry.
            """
            allowance = min(max_hold_steps, step_accounting()["remaining_charged_steps"])
            report[key] = {"prepared": None, "native_ctrl_written": None,
                "physical_stop_confirmed": False, "physical_steps": 0,
                "charged_steps": allowance, "step_accounting_exact": allowance == 0,
                "samples": [], "controls": [],
                "reason": "Hold result unavailable; its allotted attempts remain conservatively charged"}
            outcome = backend.abort_hold(feedback, target_positions,
                expected_sim_time=expected_sim_time,
                stop_velocity=np.r_[execution_limits.stop_arm_velocity, execution_limits.stop_finger_velocity],
                max_steps=allowance)
            if (not isinstance(outcome, dict)
                    or type(outcome.get("charged_steps")) is not int
                    or type(outcome.get("physical_steps")) is not int
                    or type(outcome.get("step_accounting_exact")) is not bool
                    or not 0 <= outcome["physical_steps"] <= outcome["charged_steps"] <= allowance):
                raise ValueError("Hold result must retain bounded charged and confirmed native steps")
            report[key] = outcome

        def normal_step(curve_time):
            """Charge before integration and retain its result before capture.

            The backend publishes accounting at the native integration
            boundary, before forward refresh. A later refresh/record error
            cannot erase that completed step or replace the primary error.
            """
            if step_accounting()["remaining_charged_steps"] <= 0:
                raise ExecutionGuardError("The total native physics-step budget is exhausted")
            previous = backend.last_step_accounting
            previous_attempt = 0 if previous is None else previous["attempt_index"]
            normal_accounting["charged_steps"] += 1
            primary = None
            try:
                backend.step()
            except Exception as error:
                primary = error
            try:
                outcome = backend.last_step_accounting
                if (not isinstance(outcome, dict)
                        or type(outcome.get("attempt_index")) is not int
                        or outcome["attempt_index"] != previous_attempt+1
                        or type(outcome.get("charged_steps")) is not int or outcome["charged_steps"] != 1
                        or type(outcome.get("physical_steps")) is not int or outcome["physical_steps"] not in (0, 1)
                        or type(outcome.get("step_accounting_exact")) is not bool):
                    raise ValueError("Backend must publish accounting for this exact native step attempt")
            except Exception as error:
                outcome = {"charged_steps": 1, "physical_steps": 0, "step_accounting_exact": False,
                           "reason": f"{type(error).__name__}: {error}"}
                if primary is None:
                    primary = error
                else:
                    report.setdefault("cleanup_errors", []).append({
                        "stage": "normal_step_accounting", "error": outcome["reason"]})
            normal_accounting["physical_steps"] += outcome["physical_steps"]
            normal_accounting["step_accounting_exact"] &= outcome["step_accounting_exact"]
            report["executed_steps"] = normal_accounting["physical_steps"]
            report["last_normal_step_accounting"] = deepcopy(outcome)
            if outcome["physical_steps"]:
                report["last_confirmed_curve_time_s"] = float(curve_time)
            if primary is not None:
                raise primary
        # END post-step telemetry never owns budget accounting or completed time.
        # BEGIN every functional pose goal keeps its original identity and clock.
        pose_goals = binding["problem"].get("pose_events", [])
        pose_goal_ids = binding.get("goal_identities", {}).get("pose_events", [])
        if len(pose_goals) != len(pose_goal_ids):
            raise ValueError("Actual pose goals require every original collection/index identity")
        pose_requirements, pose_goal_times, pose_goal_device_times = {}, {}, {}
        for row, index in zip(pose_goals, pose_goal_ids):
            identity = ("pose_events", index)
            if type(index) is not int or index < 0 or identity in pose_requirements:
                raise ValueError("Actual pose goals require unique original nonnegative indexes")
            requirement = PoseEventRequirement(**deepcopy(row))
            pose_requirements[identity] = requirement
            pose_goal_times[identity], pose_goal_device_times[identity] = pose_goal_schedule(
                trajectory, requirement.progress, backend.timestep)
        pose_goal_queries, evaluated_result_goals = set(), set()
        report["pose_goal_checks"] = []
        # BEGIN bind each contact permission to one original close and arrival.
        # These records are part of the already validated formal scene/problem;
        # neither a task name nor a close command invents a target or a pair.
        contact_gates, matched_permission_ids = {}, set()
        permission_rows = binding["scene"].get("robot_owner_permissions", [])
        permission_ids = [row["rule_id"] for row in permission_rows]
        if len(set(permission_ids)) != len(permission_ids):
            raise ValueError("Robot-owner contact permission identities must be unique")
        # BEGIN Holding authorization is rebuilt from this entrance, never a prior latch.
        # The caller supplies verified evidence; execution checks its identity,
        # clock and frozen attachment geometry without reading native object truth.
        holding_permissions = {}
        start_progress = float(trajectory.sample(0.)[3])
        for rule in permission_rows:
            condition_id = rule.get("holding_condition_id")
            if condition_id is None:
                continue
            conditions = [row for row in binding["problem"].get("conditions", ())
                          if row.get("condition_id") == condition_id]
            owners = [row for row in binding["scene"].get("owners", ())
                      if row.get("owner_id") == rule["target_owner"]]
            relations = [row for owner in owners for row in owner.get("attachments", ())
                         if row.get("condition_id") == condition_id
                         and tuple(row.get("interval", ())) == tuple(rule["interval"])]
            if (rule.get("arrival_goal_id") not in ([], ()) or rule.get("activation_event") != {}
                    or len(conditions) != 1 or len(owners) != 1 or len(relations) != 1):
                raise ValueError("A Holding permission needs one observed condition and matching owner attachment")
            condition, relation = conditions[0], relations[0]
            lower, upper = map(float, rule["interval"])
            if (condition.get("kind") != "observed_attachment"
                    or condition.get("carrier_id") != rule["target_owner"]
                    or condition.get("progress") != lower
                    or not lower <= start_progress <= upper
                    or (start_progress == upper and relation["release_condition_id"] is not None)):
                raise ValueError("A Holding permission must bind the currently active observed attachment")
            evidence = condition.get("source")
            provenance = evidence.get("source") if isinstance(evidence, Mapping) else None
            if (not isinstance(evidence, dict) or evidence.get("verified") is not True
                    or evidence.get("robot_feedback_id") != initial_feedback["feedback_id"]
                    or evidence.get("carrier_id") != rule["target_owner"]
                    or not ((isinstance(provenance, str) and provenance.strip())
                            or (isinstance(provenance, Mapping) and provenance))):
                raise ValueError("Holding permission evidence must identify this current robot feedback and owner")
            timestamp = evidence.get("timestamp_s")
            if (not isinstance(timestamp, (int, float)) or not np.isfinite(timestamp)
                    or abs(float(timestamp)-float(initial_feedback["timestamp_s"])) > 1e-9):
                raise ValueError("Holding permission evidence must use the current robot feedback timestamp")
            expected = np.linalg.inv(_pose_array(condition.get("T_carrier_eef"), "observed Holding transform"))
            actual = np.eye(4)
            actual[:3, 3] = relation["position_tool_owner"]
            actual[:3, :3] = Rotation.from_quat(np.asarray(relation["quaternion_tool_owner"])[[1, 2, 3, 0]]).as_matrix()
            if not np.allclose(actual, expected, atol=1e-7, rtol=0):
                raise ValueError("Holding permission and collision attachment transform disagree")
            if initial_feedback.get("scene_condition_evidence", {}).get(condition_id) is False:
                raise ValueError("Current feedback refutes the Holding permission condition")
            holding_permissions[rule["rule_id"]] = {
                "rule_id": rule["rule_id"], "condition_id": condition_id,
                "target_owner": rule["target_owner"], "interval": [lower, upper],
                "release_condition_id": relation["release_condition_id"],
                "feedback_id": initial_feedback["feedback_id"],
                "timestamp_s": float(initial_feedback["timestamp_s"]),
                "source": deepcopy(evidence)}
            matched_permission_ids.add(rule["rule_id"])
        # END false or stale evidence cannot be repaired by a software pair rule.
        for condition in binding["problem"].get("conditions", ()):
            if condition.get("kind") != "grasp_contact_activation":
                continue
            activation = condition["activation_event"]
            identity = event_occurrence_key(activation)
            arrival = tuple(condition["arrival_goal_id"])
            requirement = pose_requirements.get(arrival)
            commands = [event for event in command_events
                        if event_occurrence_key(event) == identity and event["kind"] == "close"]
            rules = [row for row in permission_rows
                     if row["target_owner"] == condition["carrier_id"]
                     and row["arrival_goal_id"] == condition["arrival_goal_id"]
                     and row["activation_event"] == activation and row["interval"] == condition["interval"]]
            if (activation.get("kind") != "close" or identity in contact_gates
                    or len(commands) != 1 or len(rules) != 1
                    or requirement is None or requirement.actual_relation is not None
                    or arrival in result_goal_ids
                    or requirement.progress != float(commands[0]["progress"])
                    or float(condition["interval"][0]) != float(commands[0]["progress"])
                    or rules[0]["rule_id"] in matched_permission_ids):
                raise ValueError("A contact activation needs one unchanged close, fixed perceived arrival and matching scene permission")
            matched_permission_ids.add(rules[0]["rule_id"])
            contact_gates[identity] = {"condition": deepcopy(condition),
                "permission": deepcopy(rules[0]), "requirement": requirement}
        if matched_permission_ids != set(permission_ids):
            raise ValueError("Every robot-owner permission needs its explicit arrival or current Holding activation")
        # Execution-local state deliberately does not inherit a prior trial's
        # latch or a completed-goal label. A new close needs a new FK witness.
        active_contact_permissions = frozenset(holding_permissions)
        report["grasp_contact_gate_checks"] = []
        report["grasp_contact_permission_changes"] = []
        report["active_robot_owner_permissions"] = sorted(active_contact_permissions)
        # BEGIN original release progress, rather than open intent, ends Holding permission.
        report["holding_contact_permission_changes"] = [
            {**deepcopy(row), "action": "activated", "curve_progress": start_progress}
            for row in holding_permissions.values()]

        def current_holding_permissions(curve_time):
            progress = float(trajectory.sample(curve_time)[3])
            return frozenset(identity for identity, row in holding_permissions.items()
                if row["interval"][0] <= progress and (progress < row["interval"][1]
                    or (progress == row["interval"][1] and row["release_condition_id"] is None)))
        # END this predicate grants no new Holding after the initial binding.
        # END source pair policy and current activation evidence stay distinct.
        # BEGIN retain the distinction between FK estimates and physical accuracy.
        report["pose_goal_scope"] = (
            "Original functional allowed sets evaluated from robot TCP FK estimates and explicitly required current owner/holding observations; "
            "scheduled first-arrival native sample or declared actual-arrival window. Explicit result-only goals record T/F/U without an online dependency; reference tracking and effect observations remain separate.")
        # END unchanged task tolerances do not become a Cartesian sensor claim.
        # END a source-progress crossing selects evaluation but cannot prove attainment.
        # BEGIN direction obligations retain original scope identities and actual evidence.
        report["completed_direction_ids"] = deepcopy(binding.get("completed_direction_ids", []))
        report["direction_evidence"] = deepcopy(binding.get("direction_evidence", []))
        report["direction_checks"] = []
        mechanisms = binding["problem"].get("mechanisms", [])
        direction_ids = binding.get("direction_identities", [])
        direction_requirements = {}
        if any(row.get("directions") for row in mechanisms):
            if len(direction_ids) != len(mechanisms):
                raise ValueError("Direction evidence needs the original mechanism/direction identities")
            for mechanism, identities in zip(mechanisms, direction_ids):
                if len(identities) != len(mechanism.get("directions", [])):
                    raise ValueError("Direction identities must cover every retained requirement")
                for requirement, identity in zip(mechanism.get("directions", []), identities):
                    if (not isinstance(identity, (tuple, list)) or len(identity) != 2
                            or any(type(index) is not int or index < 0 for index in identity)
                            or tuple(identity) in direction_requirements):
                        raise ValueError("Direction identity must be a unique original nonnegative index pair")
                    direction_requirements[tuple(identity)] = deepcopy(requirement)
        direction_starts = {}
        direction_end_times = {}
        for identity, requirement in direction_requirements.items():
            end = float(requirement["end"])
            matches = np.flatnonzero(np.abs(np.asarray(trajectory.progress_s)-end) <= 1e-12)
            end_u = (float(trajectory.progress_u[matches[-1]]) if matches.size else
                     float(np.interp(end, trajectory.progress_s, trajectory.progress_u)))
            direction_end_times[identity] = (end_u*trajectory.duration
                if end <= float(trajectory.progress_s[-1])+1e-12 else float("inf"))
        direction_boundary_polls = set()
        retried_directions = {tuple(row) for row in binding.get("retry_direction_ids", [])}
        for row in report["direction_evidence"]:
            key = tuple(row["identity"])
            if key not in retried_directions and row.get("scope_started_at_s") is not None:
                direction_starts[key] = float(row["scope_started_at_s"])
        report["direction_evidence_scope"] = (
            "Explicit allowed observer facts from actual coordinates/rates; acquisition intervals and "
            "declared observation cadence retained. Source progress selects the obligation only and "
            "never proves physical direction. Sampled evidence does not certify unobserved intersample motion.")
        # END completed reference playback cannot complete a direction requirement.
        # BEGIN actual occurrence history uses the original request physical clock.
        report["history"] = deepcopy(binding.get("request_ledger", {}).get("history", []))
        report["effect_history"] = deepcopy(binding.get("effect_history", []))
        timing_events = deepcopy(binding["timing_events"])
        report["timing_checks"] = deepcopy(binding["timing_checks"])
        active_timing_events = {}
        for event in timing_events:
            identity = event_occurrence_key(event)
            current = active_timing_events.get(identity[:2])
            if current is None or current["occurrence"] < identity[2]:
                active_timing_events[identity[:2]] = event
        request_origin = float(binding.get("request_time_origin_s", 0.))
        # END elapsed native time includes holds, unlike ordinary source timing.
        # END physical guard provenance and actual-event evidence.
        # BEGIN device feedback policy, declared before control I/O.
        report["feedback_policy"] = {
            "native_period_s": float(backend.timestep),
            "robot_checks": "Initial state and every native step boundary, before and after integration",
            "geometry_checks": "Every robot check and explicit event gate; same bound current scene",
            "observations": deepcopy(execution_limits.observation_policy),
            "timestamp_roundoff_s": 1e-8,
            "stale_required_feedback": "Reject current obligation and send bounded hold",
            "missing_future_effect": "Record U; block only a declared dependent operation",
            "delivery": "Observations retain acquisition timestamps; robot feedback remains synchronous"}
        report["feedback_policy"]["direction_boundary_observations"] = (
            "The same declared observer is also polled once at each actual native scope entrance/end; "
            "periodic phase is unchanged. A sensor that cannot supply current boundary evidence leaves U.")
        report["observation_checks"], report["object_changes"] = [], []
        # END physical contact evidence is distinct from absent effect observations.
        controls, requested_torques, saturated_torques, samples, event_index = [], [], [], [], 0
        # BEGIN retain each native controller component on the command clock.
        control_terms = {name: [] for name in (
            "reference_arm_q", "reference_arm_dq", "reference_arm_ddq",
            "arm_position_feedback_torque", "arm_velocity_feedback_torque",
            "arm_velocity_feedforward_torque", "arm_acceleration_feedforward_torque", "arm_bias_torque")}
        # END these values come from the actual issued controller calculation.
        command = "restored_ctrl"
        # BEGIN evidence dependencies authorize intervals on the same clock.
        barriers = sorted((deepcopy(event) for event in trajectory.events
                           if event["requires_evidence"] is not None), key=lambda row: row["time"])
        barrier_index = 0
        # END effect-only events do not appear in this dependency sequence.
        # BEGIN nonblocking evaluations and explicit stops keep their identities.
        ordinary_events = sorted((deepcopy(event) for event in trajectory.events
            if event["requires_evidence"] is None and event["kind"] not in ("open", "close", "hold")
            and (event["kind"] in ("effect", "function") or event.get("stop", False)
                 or any(item["check_target"] == "activation" for item in event.get("timing", ())))),
            key=lambda row: row["time"])
        ordinary_index = 0
        report["geometry_check_counts"]["event_gate"] = 0
        # END elapsed time consumes an evaluation opportunity, never a success fact.
        # BEGIN this counter excludes native finger actuator force limiting.
        arm_control_saturation_count = 0
        # END explicit arm control clipping scope.
        observation_cache, next_observation_poll = None, 0.

        # BEGIN current direction guards consume measured intervals on the native clock.
        def check_directions(observation, feedback, curve_time, elapsed):
            """Require current scoped proof, while leaving future/net-prefix unknowns open.

            An observer owns coordinate/rate measurement and its T/F/U decision.
            This boundary owns identity, acquisition freshness, interval coverage,
            and physical stopping. In particular, an old completed attempt or a
            planned coordinate derivative cannot provide a new actual witness.
            """
            now = float(feedback["timestamp_s"])
            facts = [] if observation is None else observation.get("directions", [])
            if not isinstance(facts, list):
                raise ValueError("Direction observations require an explicit fact list")
            by_id = {}
            for fact in facts:
                identity = fact.get("identity") if isinstance(fact, dict) else None
                if (not isinstance(identity, (list, tuple)) or len(identity) != 2
                        or any(type(index) is not int or index < 0 for index in identity)
                        or tuple(identity) not in direction_requirements or tuple(identity) in by_id):
                    raise ValueError("A direction fact must identify one retained original requirement")
                interval = np.asarray(fact.get("acquisition_interval_s"), dtype=float)
                if (fact.get("value") not in ("T", "F", "U")
                        or not isinstance(fact.get("source"), str) or not fact["source"].strip()
                        or type(fact.get("scope_complete")) is not bool
                        or interval.shape != (2,) or not np.isfinite(interval).all()
                        or interval[0] > interval[1] or interval[1] > float(observation["timestamp_s"])+1e-8):
                    raise ValueError("A direction fact needs T/F/U, source and its actual acquisition interval")
                by_id[tuple(identity)] = fact
            failures = []
            progress = float(trajectory.sample(curve_time)[3])
            policy = execution_limits.observation_policy
            for identity, requirement in direction_requirements.items():
                if list(identity) in report["completed_direction_ids"]:
                    continue
                active = progress >= float(requirement["start"])-1e-12
                if active and identity not in direction_starts:
                    # A suffix beginning inside a scope needs its saved actual
                    # beginning. Resetting this origin would erase earlier motion.
                    if (float(trajectory.progress_s[0]) <= requirement["start"]+1e-12
                            or identity in retried_directions):
                        direction_starts[identity] = now
                scope_start = direction_starts.get(identity)
                at_end = float(curve_time) >= direction_end_times[identity]-1e-12
                fact = by_id.get(identity)
                fresh = (fact is not None and policy is not None
                    and now-float(fact["acquisition_interval_s"][1]) <= policy["maximum_age_s"]+1e-8
                    and float(fact["acquisition_interval_s"][1]) >= sim_start-1e-8
                    and (scope_start is None or float(fact["acquisition_interval_s"][1]) >= scope_start-1e-8))
                row = None
                if fact is not None or active:
                    retained = fact if fact is not None else {"value": "U", "scope_complete": False,
                        "source": "Executor has no applicable actual direction observation",
                        "acquisition_interval_s": [now, now]}
                    row = {**deepcopy(retained), "identity": list(identity),
                        "candidate_id": binding["candidate_id"], "observed_at_s": now if observation is None else float(observation["timestamp_s"]),
                        "scope_started_at_s": scope_start, "scope_ended_at_s": now if at_end else None,
                        "observation_policy": deepcopy(policy), "feedback_id": feedback["feedback_id"]}
                # A monotone observer first acquires a baseline. Its first
                # finite interval is due at the next declared observation tick;
                # there is no displacement claim at a zero-length entrance.
                interval_due = (scope_start is None or policy is None or
                    now >= scope_start+float(policy["period_s"])-1e-8)
                required_now = active and (at_end or requirement["kind"] == "local_rate"
                    or (requirement["kind"] == "monotone" and interval_due))
                value = fact["value"] if fresh else "U"
                covered = False
                if fresh and value == "T" and at_end and fact["scope_complete"]:
                    declared_net_origin = (requirement["kind"] == "net"
                        and requirement.get("origin_coordinate") is not None and bool(requirement.get("origin_source")))
                    covered = (float(fact["acquisition_interval_s"][1]) >= now-1e-8 and
                        (declared_net_origin or (scope_start is not None
                         and float(fact["acquisition_interval_s"][0]) <= scope_start+1e-8)))
                if covered:
                    report["completed_direction_ids"].append(list(identity))
                if row is not None:
                    # Preserve the observer's raw scope claim separately from
                    # the executor's checked acquisition coverage. Continuation
                    # cannot promote an unaccepted raw T into completion.
                    row["completion_verified"] = covered
                    if row not in report["direction_evidence"]:
                        report["direction_evidence"].append(row)
                if not active:
                    continue
                # A partial net displacement can recover before its endpoint.
                # Monotone/rate F instead refutes an already applicable interval.
                violated = value == "F" and (requirement["kind"] != "net" or at_end)
                unresolved = required_now and (value != "T" or (at_end and not covered))
                check = {"identity": list(identity), "kind": requirement["kind"],
                    "curve_time_s": float(curve_time), "physical_elapsed_s": float(elapsed),
                    "required_now": required_now, "scope_complete": covered,
                    "value": "F" if violated else "U" if unresolved else value,
                    "evidence": deepcopy(fact), "scope_started_at_s": scope_start}
                if check not in report["direction_checks"]:
                    report["direction_checks"].append(check)
                if violated or unresolved:
                    failures.append({"kind": "actual_direction_violated" if violated else "actual_direction_unverified",
                        "name": "/".join(map(str, identity)), "evidence": check})
            if failures:
                report["status"] = "feedback_rejected"
                report["feedback_failure"] = {"point": "actual_direction", "curve_time_s": float(curve_time),
                    "physical_elapsed_s": float(elapsed), "actual_feedback": deepcopy(feedback),
                    "violations": failures, "last_applied_finger_goal": target_positions.tolist(), "rejected_event": None}
                raise ExecutionGuardError("Current actual coordinate direction is violated or unverified")
        # END future unknown direction facts never create an automatic entrance barrier.

        # BEGIN measured witnesses and deadlines retain occurrence-level identity.
        def effect_timestamp(identity, fact, observation):
            """Distinguish current predicate sampling from a witnessed past occurrence.

            A newly read assessment of an explicitly declared occurred-event
            fact can remain T. Its actual witness time and occurrence identity
            stay unchanged; the observation's fresh delivery time cannot be
            relabelled as a new physical event or used by a later retry.
            """
            acquired = float(observation["timestamp_s"])
            kind = fact.get("evidence_kind")
            conditions = [row for row in trajectory.conditions if row.get("condition_id") == identity]
            if (fact["value"] == "T" and any(row.get("evidence_kind") == "occurred_event" for row in conditions)
                    and kind != "occurred_event"):
                raise ValueError("An occurred-event T needs explicit witness semantics and acquisition time")
            if kind == "occurred_event":
                if len(conditions) != 1 or conditions[0].get("evidence_kind") != "occurred_event":
                    raise ValueError("Historical event truth needs its independently declared occurred-event condition")
                if fact["value"] != "T":
                    return acquired
                witnessed = fact.get("witness_timestamp_s")
                if (isinstance(witnessed, bool) or not isinstance(witnessed, (int, float, np.floating)) or not np.isfinite(witnessed)
                        or witnessed > acquired+1e-8):
                    raise ValueError("An occurred-event T needs its finite original witness acquisition time")
                identity_key = event_occurrence_key(fact.get("event", {}))
                expected_pair = (conditions[0].get("skill_id"), conditions[0].get("event_id"))
                available = [event_occurrence_key(row) for row in
                    (*binding.get("request_events", ()), *timing_events, *trajectory.events)
                    if (row["skill_id"], row["event_id"]) == expected_pair]
                if not available or identity_key != max(available, key=lambda key: key[2]):
                    raise ValueError("An occurred-event witness cannot be borrowed from another event or retry")
                return float(witnessed)
            if kind not in (None, "current_state") or "witness_timestamp_s" in fact or "event" in fact:
                raise ValueError("A historical effect witness requires an explicit occurred-event evidence kind")
            return acquired

        def witness(event, check_target, time_s, value, source, **details):
            """Append actual evidence without changing an existing first T origin."""
            row = {"skill_id": event["skill_id"], "event_id": event["event_id"],
                   "occurrence": event["occurrence"], "check_target": check_target,
                   "time_s": float(time_s), "value": value, "source": source, **deepcopy(details)}
            previous = [item for item in report["history"]
                        if event_occurrence_key(item) == event_occurrence_key(row)
                        and item["check_target"] == check_target and item["value"] == "T"
                        and item.get("condition_id") == row.get("condition_id")
                        and item.get("goal_id") == row.get("goal_id")]
            if value != "T" or not previous:
                if row not in report["history"]:
                    report["history"].append(row)
            return row

        def timed_witness(event, check_target, time_s, value, source, **details):
            checks = _timing_checks(event, check_target, time_s, report["history"],
                goal_id=details.get("goal_id"), condition_id=details.get("condition_id"))
            for check in checks:
                check["evidence_value"] = value
                if check not in report["timing_checks"]:
                    report["timing_checks"].append(check)
            # Geometry/effect truth is retained independently even when its
            # actual occurrence misses the declared clock obligation.
            witness(event, check_target, time_s, value, source, **details)
            return checks

        def reject_timing(feedback, elapsed, checks, *, event=None, point="event_time"):
            candidate_ids = {event_occurrence_key(row) for row in trajectory.events}
            # An actual future occurrence in this same candidate can supply
            # a relative origin later. Preserve this check as pending until
            # then; neither a prediction nor an old missing origin is used.
            checks = [row for row in checks if not (row["value"] == "U"
                and row.get("origin_event") is not None
                and event_occurrence_key(row["origin_event"]) in candidate_ids)]
            failures = [{"kind": "event_time_outside_window" if row["value"] == "F" else "event_time_unresolved",
                         "name": f"{row['skill_id']}/{row['event_id']}/{row['occurrence']}/{row['check_target']}",
                         "evidence": deepcopy(row)} for row in checks if row["value"] != "T"]
            if not failures:
                return
            report["status"] = "feedback_rejected"
            report["feedback_failure"] = {"point": point, "curve_time_s": min(float(elapsed), stop),
                "physical_elapsed_s": float(elapsed), "actual_feedback": deepcopy(feedback),
                "violations": failures, "last_applied_finger_goal": target_positions.tolist(),
                "rejected_event": deepcopy(event)}
            raise ExecutionGuardError("An actual event did not satisfy its original physical-time requirement")

        def resolve_pending_timing(feedback, elapsed):
            """Recheck retained actual witnesses once their future origin exists."""
            events = {event_occurrence_key(row): row for row in timing_events}
            for pending in list(report["timing_checks"]):
                if pending["value"] != "U" or pending.get("origin_event") is None:
                    continue
                event = events[event_occurrence_key(pending)]
                constraint = event["timing"][pending["constraint_index"]]
                facts = [row for row in report["history"]
                    if event_occurrence_key(row) == event_occurrence_key(event) and row["value"] == "T"
                    and row["check_target"] == constraint["check_target"]
                    and (constraint["check_target"] != "arrival"
                         or tuple(row.get("goal_id", ())) == tuple(constraint["goal_id"]))
                    and (constraint["check_target"] != "effect" or row.get("condition_id") == constraint["condition_id"])]
                if not facts:
                    continue
                fact = facts[0]
                checks = _timing_checks(event, fact["check_target"], fact["time_s"], report["history"],
                    goal_id=fact.get("goal_id"), condition_id=fact.get("condition_id"))
                for check in checks:
                    check["evidence_value"] = "T"
                    if check not in report["timing_checks"]:
                        report["timing_checks"].append(check)
                active = active_timing_events[event_occurrence_key(event)[:2]]
                if active["occurrence"] == event["occurrence"]:
                    reject_timing(feedback, elapsed, checks, event=event)

        def check_deadlines(feedback, elapsed):
            """An expired absent witness is incomplete U, never a fabricated F."""
            now = request_origin+float(elapsed)
            for event in active_timing_events.values():
                for index, item in enumerate(event.get("timing", ())):
                    constraint = EventTimeConstraint(**item)
                    try:
                        window = resolve_event_window(constraint, report["history"])
                    except TimeBindingError:
                        continue
                    if window[1] is None or now <= window[1]+1e-9:
                        continue
                    if any(row["constraint_index"] == index and row["value"] == "T"
                           and row.get("evidence_value") == "T"
                           and event_occurrence_key(row) == event_occurrence_key(event)
                           for row in report["timing_checks"]):
                        continue
                    check = {"skill_id": event["skill_id"], "event_id": event["event_id"],
                        "occurrence": event["occurrence"], "check_target": constraint.check_target,
                        "constraint_index": index, "time_s": now, "window_s": list(window),
                        "value": "U", "source": constraint.source,
                        "reason": "Original deadline expired without an applicable actual witness"}
                    report["timing_checks"].append(check)
                    reject_timing(feedback, elapsed, [check], event=event, point="expired_event_window")

        def observe_timed_effects(observation, feedback, elapsed):
            """Use acquisition time for effects; cached old binding truth is ineligible."""
            for event in active_timing_events.values():
                identities = {item["condition_id"] for item in event.get("timing", ())
                              if item["check_target"] == "effect"}
                for identity in identities:
                    evidence = observation.get("effects", {}).get(identity)
                    if evidence is None:
                        continue
                    acquisition = request_origin+effect_timestamp(identity, evidence, observation)-sim_start
                    if acquisition < request_origin-1e-9 and evidence.get("evidence_kind") != "occurred_event":
                        continue
                    indexes = [index for index, item in enumerate(event.get("timing", ()))
                               if item["check_target"] == "effect" and item["condition_id"] == identity]
                    if all(any(row["constraint_index"] == index and row["value"] == "T"
                               and row.get("evidence_value") == "T"
                               and event_occurrence_key(row) == event_occurrence_key(event)
                               for row in report["timing_checks"]) for index in indexes):
                        continue
                    checks = _timing_checks(event, "effect", acquisition, report["history"], condition_id=identity)
                    # A state observed before the opening of an effect window
                    # is not that window's completion witness. A later fresh
                    # observation may establish T without any invented wait.
                    if (not (evidence.get("evidence_kind") == "occurred_event" and evidence["value"] == "T")
                            and any("window_s" in row and row["window_s"][0] is not None
                                    and acquisition < row["window_s"][0]-1e-9 for row in checks)):
                        continue
                    checks = timed_witness(event, "effect", acquisition, evidence["value"], evidence["source"],
                        condition_id=identity, feedback_id=feedback["feedback_id"],
                        timestamp_s=observation["timestamp_s"])
                    if evidence["value"] == "T":
                        reject_timing(feedback, elapsed, checks, event=event)
        # END no predicted origin or elapsed reference node proves actual completion.

        # BEGIN actual functional goals consume their own geometry and evaluation time.
        def check_pose_goals(observation, feedback, curve_time, elapsed):
            """Evaluate original pose sets without accepting observer T labels.

            Without a declared arrival window, the first native observation at
            the original functional clock is the evaluation opportunity. Only
            execution obligations require that result to be T before advancing;
            explicitly result-only goals keep F/U without stopping the tail.
            An explicit arrival
            window instead permits a later actual witness inside that window;
            intermediate misses do not erase the remaining opportunity. No
            native step, source clock, sensor effect, or engineering tracking
            band can itself establish this task-accuracy result.
            """
            now = request_origin+float(feedback["timestamp_s"])-sim_start
            failures = []
            for identity, requirement in pose_requirements.items():
                if list(identity) in report["completed_goal_ids"] or identity in evaluated_result_goals:
                    continue
                scheduled = pose_goal_times[identity]
                arrivals = [(event, item) for event in active_timing_events.values()
                    for item in event.get("timing", ()) if item["check_target"] == "arrival"
                    and tuple(item["goal_id"]) == identity]
                windows, unresolved = [], []
                for event, item in arrivals:
                    try:
                        windows.append(resolve_event_window(EventTimeConstraint(**item), report["history"]))
                    except TimeBindingError as error:
                        unresolved.append(str(error))
                if arrivals:
                    lower = max((bounds[0] for bounds in windows if bounds[0] is not None), default=None)
                    upper = min((bounds[1] for bounds in windows if bounds[1] is not None), default=None)
                    if lower is not None and now < lower-1e-9:
                        continue
                    if unresolved and curve_time < scheduled-1e-12:
                        continue
                    required = ((upper is not None and now >= upper-1e-9)
                                or curve_time >= trajectory.duration-1e-12)
                    in_window = not unresolved and (upper is None or now <= upper+1e-9)
                else:
                    # Use the same strict native activation map as commands
                    # and continuation proof. Even a one-ULP future clock
                    # cannot borrow the preceding tick through an epsilon.
                    if float(elapsed) < pose_goal_device_times[identity]:
                        continue
                    required, in_window, lower, upper = True, True, None, None
                query = (identity, float(feedback["timestamp_s"]))
                if query in pose_goal_queries:
                    continue
                pose_goal_queries.add(query)
                measured = _actual_pose_goal(requirement, feedback, observation)
                raw = pose_goals[pose_goal_ids.index(identity[1])]
                evidence = {**measured, "identity": list(identity), "candidate_id": binding["candidate_id"],
                    "feedback_id": feedback["feedback_id"], "timestamp_s": float(feedback["timestamp_s"]),
                    "evaluated_curve_time_s": float(curve_time), "physical_elapsed_s": float(elapsed),
                    "scheduled_curve_time_s": scheduled, "device_due_time_s": pose_goal_device_times[identity],
                    "requirement": deepcopy(raw),
                    "completion_verified": False, "evaluation_window_s": [lower, upper] if arrivals else None,
                    "evaluation_scope": "result" if identity in result_goal_ids else "execution"}
                evidence["time_value"] = "U" if unresolved else "T" if in_window else "F"
                if unresolved:
                    evidence["time_reason"] = "; ".join(unresolved)
                evidence["completion_verified"] = evidence["value"] == "T" and in_window
                # Retain even a real late geometric T before the time guard
                # rejects it. A missed window does not make that geometry false.
                report["goal_evidence"].append(deepcopy(evidence))
                check = {"identity": list(identity), "value": evidence["value"], "required_now": required,
                    "required_for_execution": identity not in result_goal_ids,
                    "curve_time_s": float(curve_time), "physical_elapsed_s": float(elapsed),
                    "completion_verified": evidence["completion_verified"], "evidence": deepcopy(evidence)}
                report["pose_goal_checks"].append(check)
                if evidence["value"] == "T" and not unresolved:
                    for event in {event_occurrence_key(row[0]): row[0] for row in arrivals}.values():
                        checks = timed_witness(event, "arrival", now, "T", evidence["source"],
                            goal_id=list(identity), feedback_id=feedback["feedback_id"],
                            timestamp_s=feedback["timestamp_s"])
                        reject_timing(feedback, elapsed, checks, event=event)
                if evidence["completion_verified"]:
                    report["completed_goal_ids"].append(list(identity))
                # BEGIN result evaluation cannot invent a dependency on missing terminal owner feedback.
                if identity in result_goal_ids:
                    evaluated_result_goals.add(identity)
                if required and identity not in result_goal_ids and not evidence["completion_verified"]:
                    failures.append({"kind": "actual_pose_goal_violated" if evidence["value"] == "F"
                                     else "actual_pose_goal_unverified", "name": "/".join(map(str, identity)),
                                     "evidence": check})
                # END explicit online/arrival obligations retain the original rejection path.
            if failures:
                report["status"] = "feedback_rejected"
                report["feedback_failure"] = {"point": "actual_pose_goal", "curve_time_s": float(curve_time),
                    "physical_elapsed_s": float(elapsed), "actual_feedback": deepcopy(feedback),
                    "violations": failures, "last_applied_finger_goal": target_positions.tolist(), "rejected_event": None}
                raise ExecutionGuardError("Actual functional pose is outside its original allowed set or lacks current evidence")
        # END once-goal evaluation does not impose a new trajectory-wide pose path.

        # BEGIN bounded multi-rate observation consumption without timestamp rewriting.
        def observation_at(feedback, curve_time, elapsed):
            nonlocal observation_cache, next_observation_poll
            policy = execution_limits.observation_policy
            if policy is None:
                return None
            now = float(feedback["timestamp_s"])
            # Direction interval boundaries need an actual measurement even
            # when the native tick does not align with the periodic observer.
            # This is the same allowed channel, not interpolation of old facts;
            # its sensor may still return None or an unchanged acquisition time.
            progress = float(trajectory.sample(curve_time)[3])
            boundaries = []
            for identity, requirement in direction_requirements.items():
                if list(identity) in report["completed_direction_ids"]:
                    continue
                for side, reached in (("start", progress >= requirement["start"]-1e-12),
                                      ("end", curve_time >= direction_end_times[identity]-1e-12)):
                    key = (*identity, side)
                    if reached and key not in direction_boundary_polls:
                        boundaries.append(key)
            periodic = float(elapsed) >= next_observation_poll-1e-12
            if periodic or boundaries:
                due = next_observation_poll
                missed = max(0, int(np.floor((float(elapsed)-due+1e-12)/float(policy["period_s"])))) if periodic else 0
                if periodic:
                    next_observation_poll = due+(missed+1)*float(policy["period_s"])
                direction_boundary_polls.update(boundaries)
                observation = (None if observation_callback is None else
                               observation_callback(deepcopy(feedback), float(curve_time)))
                check = {"polled_at_s": now, "schedule_delay_s": max(0., float(elapsed)-due),
                         "missed_schedule_count": missed, "new_sample": observation is not None,
                         "poll_reason": "periodic_and_direction_boundary" if periodic and boundaries else
                             "periodic" if periodic else "direction_boundary",
                         "direction_boundaries": [list(value) for value in boundaries]}
                if observation is not None:
                    if (not isinstance(observation, dict) or not isinstance(observation.get("source"), str)
                            or not observation["source"].strip()
                            or not np.isfinite(observation.get("timestamp_s", np.nan))
                            or float(observation["timestamp_s"]) > now+1e-8):
                        raise ValueError("An observation needs its original finite acquisition time and source")
                    if (observation_cache is not None
                            and observation["timestamp_s"] < observation_cache["timestamp_s"]-1e-8):
                        raise ValueError("Observation acquisition time cannot move backward")
                    for field in ("binding_valid", "scene_condition_evidence", "scene_owner_poses", "effects"):
                        if field in observation and (type(observation[field]) is not bool if field == "binding_valid"
                                                    else not isinstance(observation[field], dict)):
                            raise ValueError(f"Observation field {field} has no explicit valid value")
                    # BEGIN all acquired T/F/U facts survive later verdict changes.
                    for identity, fact in observation.get("effects", {}).items():
                        if (not isinstance(fact, dict) or fact.get("value") not in ("T", "F", "U")
                                or not fact.get("source")):
                            raise ValueError("An effect needs explicit T/F/U evidence and provenance")
                        row = {"condition_id": identity, **deepcopy(fact),
                            "timestamp_s": effect_timestamp(identity, fact, observation),
                            "observed_at_s": float(observation["timestamp_s"]),
                            "time_s": request_origin+effect_timestamp(identity, fact, observation)-sim_start,
                            "candidate_id": binding["candidate_id"], "observation_source": observation["source"]}
                        if row not in report["effect_history"]:
                            report["effect_history"].append(row)
                    # END a latest-value snapshot never erases an earlier observation.
                    observation_cache = deepcopy(observation)
                    check["sample"] = deepcopy(observation)
                    check["acquired_at_s"] = float(observation["timestamp_s"])
                    check["age_s"] = max(0., now-float(observation["timestamp_s"]))
                    report["object_changes"].extend(deepcopy(observation.get("object_changes", [])))
                report["observation_checks"].append(check)
            fresh = (observation_cache is not None
                     and now-float(observation_cache["timestamp_s"]) <= policy["maximum_age_s"]+1e-8)
            missing = [key for key in policy["required_fields"]
                       if not fresh or key not in observation_cache]
            if missing:
                report["status"] = "feedback_rejected"
                report["feedback_failure"] = {"point": "observation_freshness",
                    "curve_time_s": float(curve_time), "physical_elapsed_s": float(elapsed),
                    "actual_feedback": deepcopy(feedback), "last_applied_finger_goal": target_positions.tolist(),
                    "rejected_event": None, "violations": [{"kind": "required_observation_unavailable",
                        "name": key, "last_observation": deepcopy(observation_cache),
                        "maximum_age_s": policy["maximum_age_s"]} for key in missing]}
                raise ExecutionGuardError("Current required observation is missing or expired")
            if observation_cache is not None and observation_cache.get("binding_valid") is False:
                # Expiry cannot erase an observed invalidation of this binding.
                return deepcopy(observation_cache)
            if fresh:
                current = deepcopy(observation_cache)
                if float(current["timestamp_s"]) < sim_start-1e-8:
                    current["effects"] = {key: {"value": "U",
                        "source": "Historical acquisition predates this actual occurrence binding"}
                        for key in current.get("effects", {})}
                    current["completed_goal_ids"] = []
                return current
            # Losing optional future-effect evidence does not invalidate the
            # static model. It also cannot leave an old T eligible for a gate.
            identities = {str(event["requires_evidence"]) for event in trajectory.events
                          if event["requires_evidence"] is not None}
            identities.update(condition["condition_id"] for condition in trajectory.conditions
                              if condition.get("effect_requirement") and condition.get("condition_id") is not None)
            if observation_cache is not None:
                identities.update(observation_cache.get("effects", {}))
            return {"source": "Declared observation freshness policy", "timestamp_s": now,
                    "effects": {identity: {"value": "U", "source": "No unexpired effect observation"}
                                for identity in identities}}
        # END missing optional effects never become missing mandatory robot feedback.

        def record(elapsed, desired_arm_q, arm_curve_dq, arm_curve_ddq, progress, feedback=None):
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
                    # BEGIN record the commanded polynomial derivative, never differentiated actual velocity.
                    "arm_curve_ddq": np.asarray(arm_curve_ddq).copy(),
                    # END q/dq/ddq retain exactly the same clipped reference clock.
                    "reference_finger_q": reference_fingers[0].copy(),
                    "reference_finger_dq": reference_fingers[1].copy(),
                    "q": feedback["q"], "dq": feedback["dq"], "tool_pose": feedback["tool_pose"],
                    # BEGIN estimator provenance accompanies the recorded robot pose.
                    "tool_pose_source": feedback["tool_pose_source"],
                    "tool_pose_is_independent_observation": feedback["tool_pose_is_independent_observation"],
                    # END native acceleration and force are independent evaluator data.
                    "reference_tool_pose": backend.reference_tool_pose(desired_arm_q),
                    "feedback": deepcopy(feedback), "finger_state": deepcopy(finger_state),
                    "geometry_check": None,
                    "progress": float(progress), "gripper_command": command}

        # BEGIN reuse one actual-arrival proof before every provisional contact query.
        def provisional_contact_permissions(feedback, *, curve_time, physical_elapsed, point,
                                            contact_event, permissions, failures):
            """Inspect an already validated sample without committing a contact latch.

            The caller supplies scalar/identity/clock failures first. In
            particular, a zero-tick close needs this proof before preflight's
            actual-state geometry check; the loop still obtains a fresh proof
            before it may write that close command.
            """
            contact_check = None
            if not failures and contact_event is not None:
                gate = contact_gates[event_occurrence_key(contact_event)]
                condition, permission = gate["condition"], gate["permission"]
                measured = _actual_pose_goal(gate["requirement"], feedback, None)
                current_progress = float(trajectory.sample(curve_time)[3])
                in_interval = (float(condition["interval"][0]) <= current_progress
                               <= float(condition["interval"][1]))
                due_now = float(physical_elapsed) >= float(contact_event["applied_time_s"])
                contact_check = {"condition_id": condition["condition_id"],
                    "rule_id": permission["rule_id"], "target_owner": permission["target_owner"],
                    "activation_event": deepcopy(condition["activation_event"]),
                    "arrival_goal_id": deepcopy(condition["arrival_goal_id"]),
                    "candidate_id": binding["candidate_id"], "feedback_id": feedback["feedback_id"],
                    "timestamp_s": float(feedback["timestamp_s"]), "actual_q": np.asarray(feedback["q"]).tolist(),
                    "point": point, "curve_time_s": float(curve_time),
                    "physical_elapsed_s": float(physical_elapsed), "measurement": measured,
                    "within_bound_interval": in_interval, "activation_due": due_now,
                    "passed": measured["value"] == "T" and in_interval and due_now,
                    "control_applied": False,
                    "source": "Frozen perceived interaction target and this sample's measured-joint FK; software contact authorization only, not current owner observation or Holding"}
                report["grasp_contact_gate_checks"].append(contact_check)
                if contact_check["passed"]:
                    # A new close never inherits permissions for another
                    # target or a previous occurrence, even inside this trial.
                    # BEGIN a new grasp latch does not revoke a separate observed attachment.
                    permissions = current_holding_permissions(curve_time) | {permission["rule_id"]}
                    # END only the old arrival-based latch is replaced by this close.
                else:
                    failures.append({"kind": "grasp_contact_arrival_rejected",
                        "name": condition["condition_id"], "evidence": deepcopy(contact_check)})
            return permissions, contact_check
        # END elapsed time selects a gate; only this measured pose can satisfy it.

        # BEGIN reject measured violations before changing an event's command.
        def guard(feedback, *, curve_time, physical_elapsed, point, event=None, sample=None,
                  contact_event=None, contact_permissions=None):
            # BEGIN remove expired Holding authorization before this state's geometry query.
            nonlocal active_contact_permissions
            holding_now = current_holding_permissions(curve_time)
            expired = (active_contact_permissions & holding_permissions.keys())-holding_now
            for identity in sorted(expired):
                report["holding_contact_permission_changes"].append({
                    **deepcopy(holding_permissions[identity]), "action": "expired",
                    "curve_progress": float(trajectory.sample(curve_time)[3]),
                    "observed_feedback_id": feedback["feedback_id"],
                    "observed_timestamp_s": float(feedback["timestamp_s"])})
            active_contact_permissions -= expired
            report["active_robot_owner_permissions"] = sorted(active_contact_permissions)
            # END open commands are deliberately absent from this attachment lifetime.
            # BEGIN consume allowed observations without native object truth.
            observation = observation_at(feedback, curve_time, physical_elapsed)
            if observation is not None:
                for effect in observation.get("effects", {}).values():
                    if (not isinstance(effect, dict) or effect.get("value") not in ("T", "F", "U")
                            or not effect.get("source")):
                        raise ValueError("An effect needs explicit T/F/U evidence and provenance")
                report["latest_observation"] = deepcopy(observation)
                if observation.get("binding_valid") is False:
                    report["status"] = "feedback_rejected"
                    report["feedback_failure"] = {"point": point, "curve_time_s": float(curve_time),
                        "physical_elapsed_s": float(physical_elapsed), "actual_feedback": deepcopy(feedback),
                        "violations": [{"kind": "current_scene_binding_invalid", "name": "bound_scene",
                                        "evidence": deepcopy(observation)}],
                        "last_applied_finger_goal": target_positions.tolist(),
                        "rejected_event": None if event is None else deepcopy(event)}
                    raise ExecutionGuardError("Current observation invalidated the bound scene; rebind from actual")
                # Completion IDs refer to the original request collections,
                # never a suffix list renumbered by an interrupted binding.
                for identity in observation.get("completed_goal_ids", []):
                    if (not isinstance(identity, (tuple, list)) or len(identity) != 2
                            or identity[0] not in ("pose_events", "coordinate_events", "supports")
                            or type(identity[1]) is not int or identity[1] < 0):
                        raise ValueError("Completed goal evidence needs an original collection/index identity")
                    key = tuple(identity)
                    previous = {tuple(item) for item in binding.get("completed_goal_ids", [])}
                    indexes = binding.get("goal_identities", {}).get(identity[0], [])
                    if key not in previous:
                        if identity[1] not in indexes:
                            raise ValueError("Completed goal identity does not belong to this original request")
                    # BEGIN pose completion always uses the actual functional calculation.
                    # The observer may report its own label for diagnostics,
                    # but it cannot bypass the original pose set or clock.
                    if identity[0] == "pose_events":
                        continue
                    # END other typed goal observers retain their explicit provenance.
                    identity = list(identity)
                    if identity not in report["completed_goal_ids"]:
                        # BEGIN an arrival obligation consumes actual completion time.
                        arrival_time = request_origin+float(observation["timestamp_s"])-sim_start
                        for arrival_event in active_timing_events.values():
                            if any(item["check_target"] == "arrival" and list(item["goal_id"]) == identity
                                   for item in arrival_event.get("timing", ())):
                                checks = timed_witness(arrival_event, "arrival", arrival_time, "T",
                                    observation["source"], goal_id=identity, feedback_id=feedback["feedback_id"],
                                    timestamp_s=observation["timestamp_s"])
                                reject_timing(feedback, physical_elapsed, checks, event=arrival_event)
                        # END a reference anchor cannot replace an observed goal arrival.
                        report["completed_goal_ids"].append(identity)
                        acquired_elapsed = float(observation["timestamp_s"])-sim_start
                        report["goal_evidence"].append({"identity": identity, "value": "T",
                            "source": observation["source"], "timestamp_s": observation["timestamp_s"],
                            "feedback_id": feedback["feedback_id"], "candidate_id": binding["candidate_id"],
                            "completion_verified": True, "evaluated_curve_time_s": min(acquired_elapsed, stop),
                            "physical_elapsed_s": acquired_elapsed,
                            "consumer_elapsed_s": float(physical_elapsed), "consumer_curve_time_s": float(curve_time),
                            "requirement": deepcopy(binding["problem"][identity[0]][indexes.index(identity[1])]),
                            "measurement_sources": [{"kind": "explicit_upstream_typed_goal_observation",
                                "source": observation["source"], "timestamp_s": observation["timestamp_s"]}]})
                for key in ("scene_condition_evidence", "scene_owner_poses"):
                    if key in observation:
                        feedback[key] = deepcopy(observation[key])
                observe_timed_effects(observation, feedback, physical_elapsed)
            check_directions(observation, feedback, curve_time, physical_elapsed)
            resolve_pending_timing(feedback, physical_elapsed)
            # END observations never overwrite measured robot coordinates.
            reference_q, reference_dq, _, _ = trajectory.sample(curve_time)
            reference_tcp = backend.reference_tool_pose(reference_q)
            barrier = barriers[barrier_index] if barrier_index < len(barriers) else None
            requires_stop = (bool(event.get("stop", False)) if event is not None else
                             bool(barrier is not None and barrier.get("stop", False)
                                  and abs(curve_time-stop) <= 1e-12))
            failures = self._execution_violations(feedback, reference_q, reference_dq, reference_tcp,
                limits=execution_limits, binding=limit_binding, finger_state=finger_state,
                expected_sim_time=sim_start+physical_elapsed, require_stop=requires_stop)
            # BEGIN actual arrival may authorize only this pending close's pair.
            # The temporary query mode is not a latch. It becomes execution
            # state only after the same command has passed every guard and its
            # actuator write has succeeded. A second read before that write
            # must prove arrival again, even when a goal was completed earlier.
            queried_permissions = (active_contact_permissions if contact_permissions is None
                                   else frozenset(contact_permissions))
            # BEGIN provisional overrides cannot retain a Holding rule beyond release.
            queried_permissions = (queried_permissions-holding_permissions.keys()) | holding_now
            # END explicit F attachment evidence remains a separate geometry violation.
            queried_permissions, contact_check = provisional_contact_permissions(feedback,
                curve_time=curve_time, physical_elapsed=physical_elapsed, point=point,
                contact_event=contact_event, permissions=queried_permissions, failures=failures)
            # END elapsed time merely selects a gate; it cannot satisfy it.
            geometry = None
            if not failures:
                geometry = self._geometry_result("check_state", deepcopy(feedback),
                    stage=point, trajectory_time=float(curve_time),
                    active_robot_owner_permissions=queried_permissions)
                report["geometry_check_counts"][point] += 1
                report["last_geometry_check"] = deepcopy(geometry)
                failures.extend(geometry["violations"])
            if sample is not None:
                sample["geometry_check"] = deepcopy(geometry)
            check = {"point": point, "curve_time_s": float(curve_time),
                     "physical_elapsed_s": float(physical_elapsed), "passed": not failures}
            if event is not None:
                check.update(skill_id=event["skill_id"], event_id=event["event_id"],
                             occurrence=event["occurrence"], kind=event["kind"])
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
            # BEGIN only independently valid measured feedback can prove a task pose.
            check_pose_goals(observation, feedback, curve_time, physical_elapsed)
            check_deadlines(feedback, physical_elapsed)
            # END original task accuracy is checked before another native command.
            return queried_permissions, contact_check
        # END reject measured violations before changing an event's command.

        # BEGIN evaluate due effects and check measured stops without adding waits.
        def consume_ordinary_events(feedback, elapsed):
            nonlocal ordinary_index
            while (ordinary_index < len(ordinary_events)
                   and min(float(elapsed), stop) >= ordinary_events[ordinary_index]["time"]-1e-12):
                event = ordinary_events[ordinary_index]
                if event.get("stop", False):
                    guard(feedback, curve_time=min(float(elapsed), stop), physical_elapsed=elapsed,
                          point="event_gate", event=event)
                    report["stopped_events"].append({**deepcopy(event),
                        "checked_at_s": float(elapsed), "feedback_id": feedback["feedback_id"],
                        "actual_arm_velocity": np.asarray(feedback["dq"])[:narm].tolist(),
                        "maximum_absolute_velocity": list(execution_limits.stop_arm_velocity)})
                if event["kind"] == "effect":
                    available = deepcopy(report.get("latest_observation", {}).get("effects", {}))
                    # Only a declared identity relation associates an effect
                    # predicate with an event. Similar names/times are not proof.
                    identities = [condition["condition_id"] for condition in trajectory.conditions
                        if condition.get("effect_requirement") and condition.get("condition_id") is not None
                        and condition.get("skill_id") == event["skill_id"]
                        and condition.get("event_id") == event["event_id"]]
                    evidence = {identity: deepcopy(available.get(identity,
                        {"value": "U", "source": "No current effect observation supplied"}))
                        for identity in identities}
                    # BEGIN every effect occurrence retains its actual observation fact.
                    observation = report.get("latest_observation", {})
                    for identity, fact in evidence.items():
                        acquired = effect_timestamp(identity, fact, observation) if observation else sim_start+elapsed
                        acquisition = request_origin+acquired-sim_start
                        # Historical facts remain in request history only. A
                        # pre-binding sample cannot certify this occurrence.
                        if acquisition < request_origin-1e-9 and fact.get("evidence_kind") != "occurred_event":
                            fact.update(value="U", source="No current occurrence observation supplied")
                        checks = _timing_checks(event, "effect", acquisition, report["history"], condition_id=identity)
                        if (not (fact.get("evidence_kind") == "occurred_event" and fact["value"] == "T")
                                and any("window_s" in row and row["window_s"][0] is not None
                                        and acquisition < row["window_s"][0]-1e-9 for row in checks)):
                            fact.update(value="U", source="No effect acquisition inside the declared window")
                        timed_witness(event, "effect", acquisition, fact["value"], fact["source"],
                            condition_id=identity, feedback_id=feedback["feedback_id"])
                    # END effect truth and task-time acceptance remain separate evidence.
                    values = [row["value"] for row in evidence.values()]
                    report["evaluated_events"].append({**deepcopy(event),
                        "evaluation_time_s": float(elapsed), "feedback_id": feedback["feedback_id"],
                        "effects": evidence, "evaluation": ("F" if "F" in values else
                            "T" if values and all(value == "T" for value in values) else "U"),
                        "source": "Bound effect predicates evaluated from explicit observed T/F/U evidence"})
                if event["kind"] == "function":
                    # A reached node records the clock occurrence only. Past
                    # goals still require their separate actual completion IDs.
                    report["reached_events"].append({**deepcopy(event),
                        "reached_time_s": float(elapsed), "feedback_id": feedback["feedback_id"]})
                if event["kind"] not in ("open", "close", "hold"):
                    checks = timed_witness(event, "activation", request_origin+float(elapsed), "T",
                        "Actual executor occurrence of the original non-command event", feedback_id=feedback["feedback_id"])
                    reject_timing(feedback, elapsed, checks, event=event)
                ordinary_index += 1
        # END unknown/false effects remain outcomes, not implicit barriers.

        # BEGIN consume current T/F/U before crossing a declared dependency.
        def release_dependencies(feedback, elapsed):
            nonlocal barrier_index, stop
            while barrier_index < len(barriers) and elapsed >= barriers[barrier_index]["time"]-1e-12:
                event = barriers[barrier_index]
                checks = _timing_checks(event, "activation", request_origin+float(elapsed), report["history"])
                reject_timing(feedback, elapsed, checks, event=event)
                available_effects = deepcopy(report.get("latest_observation", {}).get("effects", {}))
                evidence = available_effects.get(str(event["requires_evidence"]),
                    {"value": "U", "source": "No current prerequisite observation"})
                if evidence["value"] == "T":
                    # A fresh-enough state sample can still predate an effect's
                    # required acquisition window. Its raw T cannot authorize
                    # a dependent operation until that separate obligation has
                    # an actual, occurrence-matched in-window witness.
                    for timed_event in active_timing_events.values():
                        for constraint_index, constraint in enumerate(timed_event.get("timing", ())):
                            if (constraint["check_target"] != "effect"
                                    or constraint["condition_id"] != str(event["requires_evidence"])):
                                continue
                            witnessed = any(event_occurrence_key(row) == event_occurrence_key(timed_event)
                                and row["constraint_index"] == constraint_index and row["value"] == "T"
                                and row.get("evidence_value") == "T" for row in report["timing_checks"])
                            if not witnessed:
                                evidence = {"value": "U", "source": "Required effect has no actual acquisition in its original time window"}
                report["dependency_outcomes"].append({"event": deepcopy(event), "evidence": deepcopy(evidence),
                    "feedback_id": feedback["feedback_id"], "physical_elapsed_s": float(elapsed)})
                if evidence["value"] != "T":
                    report["status"] = "effect_rejected" if evidence["value"] == "F" else "stopped_for_evidence"
                    # BEGIN dependency stopping spends only the shared remainder.
                    run_hold("dependency_hold", feedback, expected_sim_time=sim_start+elapsed)
                    # END an exhausted allowance never grants an extra stop step.
                    return False
                report["verified_events"].append(deepcopy(event))
                # Only actual dependency evaluation activates this event;
                # the verified prerequisite remains separately sourced above.
                if event["kind"] not in ("open", "close", "hold"):
                    timed_witness(event, "activation", request_origin+float(elapsed), "T",
                        "Actual prerequisite gate accepted before entering its dependent interval",
                        feedback_id=feedback["feedback_id"])
                barrier_index += 1
                previous_stop = stop
                stop = (float(barriers[barrier_index]["time"]) if barrier_index < len(barriers)
                        else trajectory.duration)
                if int(np.ceil(stop/backend.timestep-1e-10)) > max_physics_steps:
                    raise ExecutionGuardError("Remaining authorized interval exceeds the original cumulative physics budget")
                # The curve and all clocks remain untouched. Query the newly
                # authorized nominal interval before another native command.
                preflight = self._geometry_result("preflight", trajectory, deepcopy(feedback),
                    execution_limits, start_time=previous_stop, until=stop,
                    current_time=min(float(elapsed), trajectory.duration),
                    active_robot_owner_permissions=active_contact_permissions)
                report["extended_preflights"].append({"until": stop, "verdict": deepcopy(preflight)})
                if not preflight["passed"]:
                    report["status"] = "feedback_rejected"
                    report["feedback_failure"] = {"point": "extended_geometry_preflight",
                        "curve_time_s": min(float(elapsed), trajectory.duration),
                        "physical_elapsed_s": float(elapsed), "actual_feedback": deepcopy(feedback),
                        "violations": deepcopy(preflight["violations"]), "geometry_check": deepcopy(preflight),
                        "last_applied_finger_goal": target_positions.tolist(), "rejected_event": None}
                    raise ExecutionGuardError("Independent geometry rejected the newly authorized interval")
                report["authorized_prefix_extended"] = True
                report["evidence_passed"] = True
                report["next_verification"] = (deepcopy(barriers[barrier_index])
                    if barrier_index < len(barriers) else None)
                report["planned_stop_s"] = stop
            return True
        # END unknown or false evidence causes a physical hold, never success.

        # BEGIN an independent evaluator cannot become an execution dependency.
        def emit_independent_evidence(sample, phase):
            sink = report["independent_evidence_sink"]
            if not sink["configured"] or sink["disabled"]:
                return
            sink["attempted_callbacks"] += 1
            try:
                # Keep evaluator mutations of its record out of subsequent
                # online event/goal consumption. Native model/data are exposed
                # exclusively as read-only evaluator inputs by this contract.
                step_callback(backend.native_model, backend.native_data, deepcopy(sample))
            except Exception as error:
                sink["disabled"] = True
                sink["first_error"] = {"type": type(error).__name__, "message": str(error),
                    "phase": phase, "curve_time": sample["curve_time"], "sim_time": sample["sim_time"]}
            else:
                sink["successful_callbacks"] += 1
        # END return values and failures cannot grant or deny robot commands.

        q, dq, ddq, progress = trajectory.sample(0.)
        samples.append(record(0., q, dq, ddq, progress))
        # BEGIN persistence failures must not replace a measured execution error.
        execution_error = None
        # END retain the primary exception separately from diagnostic sinks.
        try:
            # BEGIN retain one initial sample even when preflight later rejects.
            emit_independent_evidence(samples[-1], "initial_state")
            # END the runner must not separately duplicate this initial exposure.
            # BEGIN a close due at entry needs its actual proof before geometry.
            # Select the same last command as the zero-tick loop, without
            # consuming any event. Only this special entrance uses a temporary
            # permission, and both geometry consumers receive its same fresh
            # feedback. The execution latch remains empty until apply_control.
            initial_permissions = active_contact_permissions
            preflight_feedback = initial_feedback
            entry_commands = [event for event in command_events if event["activation_step"] == 0]
            entry_close = (entry_commands[-1] if entry_commands and entry_commands[-1]["kind"] == "close"
                and event_occurrence_key(entry_commands[-1]) in contact_gates else None)
            if entry_close is not None:
                preflight_feedback = samples[-1]["feedback"]
                reference_tcp = backend.reference_tool_pose(q)
                failures = self._execution_violations(preflight_feedback, q, dq, reference_tcp,
                    limits=execution_limits, binding=limit_binding, finger_state=finger_state,
                    expected_sim_time=sim_start, require_stop=bool(entry_close.get("stop", False)))
                initial_permissions, _ = provisional_contact_permissions(preflight_feedback,
                    curve_time=0., physical_elapsed=0., point="initial_contact_gate",
                    contact_event=entry_close, permissions=initial_permissions, failures=failures)
                if failures:
                    report["status"] = "feedback_rejected"
                    report["feedback_failure"] = {"point": "initial_contact_gate", "curve_time_s": 0.,
                        "physical_elapsed_s": 0., "passed": False, "violations": failures,
                        "actual_feedback": deepcopy(preflight_feedback), "geometry_check": None,
                        "reference_arm_q": q.tolist(), "reference_arm_dq": dq.tolist(),
                        "reference_tool_pose": reference_tcp.tolist(),
                        "actual_finger_interval": deepcopy(finger_state),
                        "last_applied_finger_goal": target_positions.tolist(), "rejected_event": deepcopy(entry_close)}
                    raise ExecutionGuardError(f"Actual feedback violates initial_contact_gate: {failures[0]['kind']} / {failures[0]['name']}")
            # END preflight still prepares the geometry context before any state guard.
            # Preflight checks the declared model/reference contract without
            # stepping or preparing an actuator goal. Formal planning success
            # does not bypass this independent current-scene acceptance gate.
            preflight = self._geometry_result("preflight", trajectory, deepcopy(preflight_feedback), execution_limits,
                active_robot_owner_permissions=initial_permissions)
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
            guard(samples[-1]["feedback"], curve_time=0., physical_elapsed=0., point="initial_state", sample=samples[-1],
                contact_permissions=initial_permissions)
            consume_ordinary_events(samples[-1]["feedback"], 0.)
            active = release_dependencies(samples[-1]["feedback"], 0.)
            for index in range(count):
                if not active:
                    break
                # BEGIN exhausted total capacity forbids even a new control write.
                if step_accounting()["remaining_charged_steps"] <= 0:
                    raise ExecutionGuardError("The total native physics-step budget is exhausted")
                # END hold attempts and normal playback use the same allowance.
                elapsed = index * backend.timestep
                # Re-read once at each step boundary. The same snapshot gates
                # any due event; a failed gate cannot mutate a finger goal.
                before = backend.proprioception()
                due = []
                while event_index < len(command_events) and command_events[event_index]["activation_step"] <= index:
                    due.append(command_events[event_index])
                    event_index += 1
                # BEGIN check arrival before entering the pending close mode.
                pending_close = (due[-1] if due and due[-1]["kind"] == "close"
                    and event_occurrence_key(due[-1]) in contact_gates else None)
                next_contact_permissions, contact_check = guard(before,
                    curve_time=min(elapsed, stop), physical_elapsed=elapsed, point="before_step",
                    contact_event=pending_close,
                    contact_permissions=(current_holding_permissions(min(elapsed, stop)) if due and due[-1]["kind"] == "close"
                                         and pending_close is None else None))
                # END a missed gate stops before sampling or writing the close ramp.
                next_command = command
                next_finger_positions = target_positions.copy()
                next_finger_state = finger_state
                if due:
                    # Distinct source events can fall in one native interval.
                    # Retain every identity, and never label overwritten goals
                    # as applied to a physical step.
                    # BEGIN target, permission and command share the activation tick.
                    guard(before, curve_time=float(due[-1]["applied_time_s"]), physical_elapsed=elapsed,
                          point="command_event", event=due[-1],
                          contact_permissions=next_contact_permissions)
                    checks = _timing_checks(due[-1], "activation", request_origin+elapsed, report["history"])
                    reject_timing(before, elapsed, checks, event=due[-1])
                    # END original intent time remains in the event ledger only.
                    next_command = due[-1]["kind"]
                    # The pre-event state belongs to the old interval. Only an
                    # actually applied command selects its declared new one.
                    next_finger_state = deepcopy(limit_binding["event_fingers"][(due[-1]["skill_id"], due[-1]["event_id"])])
                next_time = (index + 1) * backend.timestep
                q, dq, ddq, progress = trajectory.sample(min(next_time, stop))
                # BEGIN one synchronous 7+2 command query, with independent feedback.
                # A future event's ramp starts at its saved activation tick.
                # At that boundary its q/dq remain C1 with the prior hold, so
                # sampling this step's endpoint cannot issue an early closure.
                if finger_reference is not None:
                    next_finger_positions, _ = finger_reference.sample(min(next_time, stop))
                # END native GRIP receives the same named reference as planning.
                # BEGIN the backend receives the same endpoint q/dq/ddq as the saved reference.
                control = backend.apply_control(q, dq, ddq, next_finger_positions)
                # END actuator interpretation cannot change the candidate's clock or polynomial.
                # BEGIN only a successful command write commits/revokes a latch.
                if due and due[-1]["kind"] in ("open", "close"):
                    previous_permissions = sorted(active_contact_permissions)
                    active_contact_permissions = (next_contact_permissions if pending_close is not None
                                                  else current_holding_permissions(min(elapsed, stop)))
                    if contact_check is not None:
                        contact_check["control_applied"] = True
                    report["active_robot_owner_permissions"] = sorted(active_contact_permissions)
                    report["grasp_contact_permission_changes"].append({
                        "event": deepcopy(due[-1]), "feedback_id": before["feedback_id"],
                        "timestamp_s": float(before["timestamp_s"]), "applied_at_s": float(elapsed),
                        "previous_rule_ids": previous_permissions,
                        "active_rule_ids": sorted(active_contact_permissions)})
                # END open and a different close cannot borrow an earlier target.
                # BEGIN aggregate only the backend's named arm clipping count.
                arm_control_saturation_count += control["arm_control_saturated_coordinates"]
                # END native finger force limits are measured separately.
                target_positions, finger_state, command = next_finger_positions, next_finger_state, next_command
                if due:
                    report["superseded_gripper_events"].extend(deepcopy(due[:-1]))
                    report["events"].append({**due[-1], "applied_at_s": elapsed,
                        "application_delay_s": max(0., elapsed-due[-1]["time"])})
                    timed_witness(due[-1], "activation", request_origin+elapsed, "T",
                        "Native actuator control written for this exact original event occurrence",
                        feedback_id=before["feedback_id"])
                controls.append(control["actuator_ctrl"].copy())
                requested_torques.append(control["requested_arm_torque"].copy())
                saturated_torques.append(control["arm_torque_saturated"].copy())
                # BEGIN preserve native feedforward and feedback terms independently.
                for name in control_terms:
                    control_terms[name].append(_finite_array(control[name], (narm,), name))
                # END no term is reconstructed from post-step telemetry.
                # BEGIN native completion is counted before any post-step capture.
                normal_step(min(next_time, stop))
                current = record(next_time, q, dq, ddq, progress)
                # END failed record/FK cannot hide a completed physical step.
                samples.append(current)
                # Preserve the completed physical sample, including independent
                # evaluator contacts, before rejecting its measured feedback.
                try:
                    # BEGIN the arrival tick is checked before its next control write.
                    # A close due at this completed tick may use a provisional
                    # pair mode only after actual FK arrival passes. The next
                    # iteration re-reads and rechecks it before committing, so
                    # this state check cannot become a stale authorization.
                    upcoming = [event for event in command_events[event_index:]
                                if event["activation_step"] <= index+1]
                    upcoming_close = (upcoming[-1] if upcoming and upcoming[-1]["kind"] == "close"
                        and event_occurrence_key(upcoming[-1]) in contact_gates else None)
                    guard(current["feedback"], curve_time=min(next_time, stop),
                          physical_elapsed=next_time, point="after_step", sample=current,
                          contact_event=upcoming_close)
                    # END only an explicit pending close can request that mode.
                finally:
                    # BEGIN post-step evaluation cannot mask or create a guard failure.
                    emit_independent_evidence(current, "after_step")
                    # END the first sink failure disables only further collection.
                consume_ordinary_events(current["feedback"], next_time)
                active = release_dependencies(current["feedback"], next_time)
            if active:
                report["status"] = "curve_finished"
        except Exception as error:
            execution_error = error
            if report["status"] != "feedback_rejected":
                report["status"] = "execution_error"
            report["error"] = f"{type(error).__name__}: {error}"
            # BEGIN send bounded hold control and preserve separate feedback.
            # The last applied finger goal remains in force. No rejected
            # event, source reset or artificial zero velocity is introduced.
            # BEGIN cleanup failures retain the original execution exception.
            try:
                failure_feedback = (report["feedback_failure"]["actual_feedback"]
                                    if report["feedback_failure"] is not None else backend.proprioception())
                expected_steps = step_accounting()["physical_steps"]
                run_hold("abort_hold", failure_feedback,
                         expected_sim_time=sim_start+expected_steps*backend.timestep)
            except Exception as hold_error:
                report.setdefault("cleanup_errors", []).append({
                    "stage": "abort_hold", "error": f"{type(hold_error).__name__}: {hold_error}"})
            # END missing fresh feedback never becomes a fabricated hold origin.
            # END physical stop confirmation is distinct from loop exit.
            raise
        finally:
            # BEGIN publish budget facts independently of final telemetry capture.
            accounting = step_accounting()
            report["physics_step_accounting"] = accounting
            # BEGIN a missing post-step sample leaves independent evidence incomplete.
            sink = report["independent_evidence_sink"]
            normal = report["normal_step_accounting"]
            sink["completed"] = bool(sink["configured"] and not sink["disabled"]
                and normal["step_accounting_exact"]
                and sink["successful_callbacks"] == 1+normal["physical_steps"])
            # END this is collection coverage, never task or physical success.
            finalization_error = None
            try:
                final_feedback = deepcopy(backend.proprioception())
            except Exception as error:
                final_feedback = None
                finalization_error = error
                report.setdefault("cleanup_errors", []).append({
                    "stage": "final_feedback", "error": f"{type(error).__name__}: {error}"})
                if execution_error is None:
                    report["status"] = "execution_error"
                    report["error"] = f"{type(error).__name__}: {error}"
            # END unavailable actual feedback cannot suppress the step ledger.
            actual_q = np.asarray([row["q"] for row in samples])
            command_q = np.asarray([row["command_q"] for row in samples])
            # BEGIN an unreadable actual clock cannot erase charged native work.
            try:
                final_elapsed = float(backend.sim_time - sim_start)
                if not np.isfinite(final_elapsed) or final_elapsed < 0:
                    raise ValueError("Final native clock is nonfinite or precedes this execution")
            except Exception as error:
                final_elapsed = None
                accounting["step_accounting_exact"] = False
                report.setdefault("cleanup_errors", []).append({
                    "stage": "final_clock", "error": f"{type(error).__name__}: {error}"})
                if finalization_error is None:
                    finalization_error = error
                if execution_error is None:
                    report["status"] = "execution_error"
                    report["error"] = f"{type(finalization_error).__name__}: {finalization_error}"
            # END unknown elapsed time is never replaced by count*timestep.
            applied_ids = {event_occurrence_key(event) for event in report["events"]}
            consumed_ids = applied_ids | {event_occurrence_key(event)
                for event in report["verified_events"] if event["kind"] not in ("open", "close", "hold")}
            consumed_ids.update(event_occurrence_key(event)
                                for event in report["superseded_gripper_events"])
            consumed_ids.update(event_occurrence_key(event)
                                for key in ("evaluated_events", "stopped_events", "reached_events") for event in report[key])
            barrier_command_reached = stop == 0 or (report["executed_steps"] > 0 and samples[-1]["curve_time"] >= stop-1e-12)
            report.update(elapsed_wall_s=time.monotonic()-started, elapsed_sim_s=final_elapsed,
                actual_sim_stop_s=final_elapsed, last_command_sample_s=samples[-1]["curve_time"] if controls else None,
                remaining_to_barrier_s=None if final_elapsed is None else max(0., stop-final_elapsed),
                physical_clock_overrun_s=None if final_elapsed is None else max(0., final_elapsed-stop),
                barrier_command_reached=barrier_command_reached,
                barrier_crossed=any(samples[-1]["curve_time"] > event["time"]
                                    for event in report["verified_events"]),
                unapplied_gripper_events=[deepcopy(event) for event in command_events
                                          if event_occurrence_key(event) not in applied_ids],
                maximum_arm_tracking_error_rad=float(np.max(np.abs(actual_q[:, :narm]-command_q[:, :narm]))),
                maximum_finger_command_position_difference=float(np.max(np.abs(actual_q[:, narm:]-command_q[:, narm:]))),
                # BEGIN preserve the arm-only meaning in the report field.
                arm_control_saturated_coordinates=arm_control_saturation_count,
                # END never describe this as all-actuator saturation.
                maximum_absolute_actual_arm_velocity=float(np.max(np.abs(np.asarray([row["dq"] for row in samples])[:, :narm]))),
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                unconsumed_events=[deepcopy(event) for event in trajectory.events
                                   if event_occurrence_key(event) not in consumed_ids],
                pending_verifications=[deepcopy(event) for event in barriers[barrier_index:]])
            # BEGIN explicit effect outcomes and same-request continuation data.
            observed = report.get("latest_observation", {})
            effects = deepcopy(observed.get("effects", {}))
            required_effects = set()
            for condition in trajectory.conditions:
                identity = condition.get("condition_id")
                if identity is not None and condition.get("effect_requirement"):
                    required_effects.add(identity)
                    effects.setdefault(identity, {"value": "U", "source": "No current effect observation supplied"})
            for event in trajectory.events:
                if event.get("requires_evidence") is not None:
                    required_effects.add(str(event["requires_evidence"]))
                    effects.setdefault(str(event["requires_evidence"]),
                        {"value": "U", "source": "No current effect observation supplied"})
            report["effects"] = effects
            required_values = [effects[identity]["value"] for identity in required_effects]
            report["required_effect_ids"] = sorted(required_effects)
            report["effect_result"] = ("F" if "F" in required_values else
                "T" if required_values and all(value == "T" for value in required_values) else "U")
            # BEGIN completion of the cubic is independent of hard-time obligations.
            timing_values = []
            for event in active_timing_events.values():
                for index, constraint in enumerate(event.get("timing", ())):
                    checks = [row for row in report["timing_checks"]
                              if event_occurrence_key(row) == event_occurrence_key(event)
                              and row["constraint_index"] == index]
                    value = ("T" if any(row["value"] == "T" and row.get("evidence_value") == "T" for row in checks)
                             else "F" if any(row["value"] == "F" and row.get("evidence_value") == "T" for row in checks)
                             else "U")
                    timing_values.append(value)
                    if not checks:
                        report["timing_checks"].append({"skill_id": event["skill_id"],
                            "event_id": event["event_id"], "occurrence": event["occurrence"],
                            "check_target": constraint["check_target"], "constraint_index": index,
                            "time_s": None if final_elapsed is None else request_origin+final_elapsed,
                            "value": "U", "source": constraint["source"],
                            "reason": "Execution ended without an applicable actual witness"})
            report["timing_result"] = (None if not timing_values else "F" if "F" in timing_values
                else "T" if all(value == "T" for value in timing_values) else "U")
            # BEGIN report actual task accuracy separately from effects and tracking.
            pose_values = ["T" if list(identity) in report["completed_goal_ids"] else
                "F" if any(tuple(row["identity"]) == identity and row["required_now"] and row["value"] == "F"
                           for row in report["pose_goal_checks"]) else "U" for identity in pose_requirements]
            report["pose_goal_result"] = (None if not pose_values else "F" if "F" in pose_values
                else "T" if all(value == "T" for value in pose_values) else "U")
            # END an effect predicate cannot replace a missing owner pose.
            # BEGIN only online pose obligations can require another execution binding.
            execution_pose_values = [value for identity, value in zip(pose_requirements, pose_values)
                                     if identity not in result_goal_ids]
            report["execution_pose_goal_result"] = (None if not execution_pose_values else
                "F" if "F" in execution_pose_values else
                "T" if all(value == "T" for value in execution_pose_values) else "U")
            # END result-only U remains visible without authorizing a fake completion or forcing a new run.
            # BEGIN direction completion is an independent actual-evidence verdict.
            direction_values = ["T" if list(identity) in report["completed_direction_ids"] else
                "F" if any(tuple(row["identity"]) == identity and row["value"] == "F"
                           for row in report["direction_checks"]) else "U"
                for identity in direction_requirements]
            report["direction_result"] = (None if not direction_values else "F" if "F" in direction_values
                else "T" if all(value == "T" for value in direction_values) else "U")
            # END a finished cubic cannot hide unresolved scoped obligations.
            request_ledger = deepcopy(binding.get("request_ledger", {}))
            # BEGIN even an event-free actual trial requires its physical continuation.
            request_ledger["requires_continuation"] = True
            # END ledger-only planning retries cannot reset an executed skill's clock.
            request_ledger["history"] = deepcopy(report["history"])
            request_ledger["effect_history"] = deepcopy(report["effect_history"])
            request_ledger["timing_events"] = deepcopy(timing_events)
            request_ledger["timing_checks"] = deepcopy(report["timing_checks"])
            for key in ("goal_evidence", "completed_goal_ids", "direction_evidence", "completed_direction_ids"):
                rows = request_ledger.setdefault(key, [])
                rows.extend(deepcopy(row) for row in report[key] if row not in rows)
            report["request_ledger"] = request_ledger
            # END partial or late physical evidence is never rounded into success.
            report["continuation"] = {
                "candidate_id": binding["candidate_id"], "original_binding": binding,
                "request_ledger": request_ledger, "history": deepcopy(report["history"]),
                "effect_history": deepcopy(report["effect_history"]),
                "timing_checks": deepcopy(report["timing_checks"]),
                "timing_events": deepcopy(timing_events),
                "device_policy": deepcopy(binding["device_policy"]),
                "active_finger_interval": {key: deepcopy(finger_state[key])
                    for key in ("position_bounds", "maximum_speed")},
                "physics_budget": physics_budget,
                "device_timestep_s": float(backend.timestep),
                # BEGIN confirmed command time survives a failed post-step capture.
                "elapsed_physical_s": final_elapsed,
                "last_curve_time_s": report["last_confirmed_curve_time_s"],
                # END missing a state sample never rolls back the commanded curve.
                "request_time_origin_s": float(binding.get("request_time_origin_s", 0.)),
                "cumulative_elapsed_physical_s": (None if final_elapsed is None else
                    float(binding.get("request_time_origin_s", 0.))+final_elapsed),
                "applied_events": deepcopy(report["events"]),
                "verified_events": deepcopy(report["verified_events"]),
                "evaluated_events": deepcopy(report["evaluated_events"]),
                "stopped_events": deepcopy(report["stopped_events"]),
                "reached_events": deepcopy(report["reached_events"]),
                "completed_goal_ids": deepcopy(report["completed_goal_ids"]),
                "goal_evidence": deepcopy(report["goal_evidence"]),
                "completed_direction_ids": deepcopy(report["completed_direction_ids"]),
                "direction_evidence": deepcopy(report["direction_evidence"]),
                "superseded_events": deepcopy(report["superseded_gripper_events"]),
                "remaining_events": deepcopy(report["unconsumed_events"]),
                # BEGIN absent final feedback remains explicitly unavailable.
                "effects": effects, "actual_feedback": final_feedback,
                # END no earlier sample is relabelled as the current robot state.
                "object_changes": deepcopy(report["object_changes"]),
                "requires_rebind": (final_feedback is None or not accounting["step_accounting_exact"]
                    or report["status"] != "curve_finished"
                    or report["timing_result"] not in (None, "T")
                    or report["direction_result"] not in (None, "T")
                    or report["execution_pose_goal_result"] not in (None, "T")),
                "reason": "Remaining obligations require current actual/scene binding; applied events and original hard clock cannot be reset",
                # BEGIN budget charges and confirmed time have distinct contracts.
                "consumed_physics_steps": accounting["charged_steps"],
                "confirmed_physics_steps": accounting["physical_steps"],
                "step_accounting_exact": accounting["step_accounting_exact"],
                "cumulative_physics_steps": prior_physics_steps+accounting["charged_steps"],
                "remaining_physics_steps": max(0, physics_budget-prior_physics_steps-accounting["charged_steps"])}
                # END an uncertain attempt is never a free retry or invented time.
            # END finishing command playback never asserts Holding or release.
            # BEGIN each evidence sink is independent of the primary verdict.
            persistence_errors = []

            def persist(name, write):
                """Attempt every artifact and retain exact write failures in memory."""
                try:
                    write()
                except Exception as error:
                    persistence_errors.append(error)
                    report.setdefault("persistence_errors", []).append({
                        "artifact": name, "error": f"{type(error).__name__}: {error}"})
                    return False
                return True

            continuation_path = (output/"continuation.json").resolve()
            if persist(continuation_path.name, lambda: continuation_path.write_text(
                    json.dumps(_feedback_json(report["continuation"]), indent=2, allow_nan=False)+"\n")):
                report["continuation_path"] = str(continuation_path)
            persist("physical-trajectory.npz", lambda: np.savez_compressed(
                output/"physical-trajectory.npz", joint_names=np.asarray(backend.joint_names),
                command_joint_names=np.asarray(backend.joint_names),
                arm_joint_names=np.asarray(backend.arm_joint_names), finger_joint_names=np.asarray(backend.finger_joint_names),
                elapsed=np.asarray([row["elapsed"] for row in samples]),
                curve_time=np.asarray([row["curve_time"] for row in samples]),
                sim_time=np.asarray([row["sim_time"] for row in samples]),
                command_q=command_q, arm_curve_dq=np.asarray([row["arm_curve_dq"] for row in samples]),
                # BEGIN analytic derivatives and torque terms preserve their separate sample clocks.
                arm_curve_ddq=np.asarray([row["arm_curve_ddq"] for row in samples]),
                **{name: np.asarray(values).reshape(-1, narm) for name, values in control_terms.items()},
                # END command terms have one row per native control write; state samples include time zero.
                reference_finger_provided=np.asarray(finger_reference is not None),
                reference_finger_q=np.asarray([row["reference_finger_q"] for row in samples]),
                reference_finger_dq=np.asarray([row["reference_finger_dq"] for row in samples]),
                actual_q=actual_q, actual_dq=np.asarray([row["dq"] for row in samples]),
                # BEGIN record robot FK provenance, without native dynamics diagnostics.
                actual_tool_pose=np.asarray([row["tool_pose"] for row in samples]),
                tool_pose_source=np.asarray([row["tool_pose_source"] for row in samples]),
                tool_pose_is_independent_observation=np.asarray(
                    [row["tool_pose_is_independent_observation"] for row in samples], dtype=bool),
                # END actual joint sampling and Cartesian estimation stay distinct.
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
                gripper_command=np.asarray([row["gripper_command"] for row in samples])))
            # BEGIN feedback artifacts require a captured current sample.
            if final_feedback is not None and report["status"] in (
                    "stopped_for_evidence", "curve_finished", "feedback_rejected", "execution_error", "effect_rejected"):
            # END an unavailable sample remains absent, while both ledgers persist.
                # This is a robot-only observation. A later bridge still needs
                # a current scene observation and independently verified
                # holding evidence before constructing an attached-object model.
                feedback = deepcopy(report["continuation"]["actual_feedback"])
                completed = report["status"] in ("stopped_for_evidence", "curve_finished")
                name = ("failure-feedback.json" if not completed else
                        "barrier-feedback.json" if trajectory.next_verification is not None else "final-feedback.json")
                feedback_path = (output/name).resolve()
                feedback.update(evidence_passed=report["evidence_passed"], next_verification=report["next_verification"],
                    initial_feedback_id=initial_feedback["feedback_id"])
                feedback["execution_status"] = report["status"]
                if persist(name, lambda: feedback_path.write_text(
                        json.dumps(_feedback_json(feedback), indent=2, allow_nan=False)+"\n")):
                    report.update(robot_feedback_path=str(feedback_path), feedback_id=feedback["feedback_id"],
                                  barrier_feedback_path=str(feedback_path) if completed and trajectory.next_verification is not None else None)
            persist("execution-report.json", lambda: (output/"execution-report.json").write_text(
                json.dumps(_feedback_json(report), indent=2, allow_nan=False)+"\n"))
            # BEGIN final capture errors precede later persistence errors.
            if execution_error is None:
                if finalization_error is not None:
                    raise finalization_error
                if persistence_errors:
                    raise persistence_errors[0]
            # END cleanup cannot replace a primary execution exception.
            # END failed persistence never reports success or hides an execution error.
        return report

# END production trajectory execution and required physical contracts.
