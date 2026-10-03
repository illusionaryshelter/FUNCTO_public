"""Owned planning evidence handed to execution with the exact candidate.

This is serialization and identity checking for ``PlanningResult``, not a
second request API. A candidate ID identifies an observation/clock binding;
it is not a security credential or a substitute for independent preflight.
"""

# BEGIN exact candidate, scene and validation handoff.
from dataclasses import fields, is_dataclass
from collections.abc import Mapping
from copy import deepcopy
from uuid import uuid4

import numpy as np


def plain(value):
    """Copy declared input fields without retaining solver/cache objects.

    Unbounded intervals are explicit JSON data; NaN never represents a
    missing measurement. Private dataclass caches are deliberately excluded.
    """
    if is_dataclass(value):
        return {item.name: plain(getattr(value, item.name)) for item in fields(value)
                if not item.name.startswith("_")}
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if np.isnan(value):
            raise ValueError("A formal execution binding cannot contain NaN")
        return value if np.isfinite(value) else {"unbounded": "+" if value > 0 else "-"}
    raise TypeError(f"Execution binding contains an undeclared input type: {type(value).__name__}")


def curve_snapshot(trajectory):
    """Keep the polynomial, commands and scope, including the full suffix."""
    return plain({"joint_names": trajectory.joint_names, "times": trajectory.times,
        "position": trajectory.position, "velocity": trajectory.velocity,
        "coefficients": trajectory._curve.c,
        "progress_u": trajectory.progress_u, "progress_s": trajectory.progress_s,
        "events": trajectory.events, "conditions": trajectory.conditions,
        "duration": trajectory.duration, "executable_until": trajectory.executable_until,
        "next_verification": trajectory.next_verification,
        "fingers": None if trajectory.finger_trajectory is None else
            trajectory.finger_trajectory.describe()})


def bind_candidate(trajectory, *, problem, scene, validation, native_report):
    """Bind only a natively accepted, fully checked exported curve.

    Independent solid geometry and actual tracking are still executor work.
    Their absence cannot be relabelled as a completed planning query.
    """
    violations = np.asarray(validation["violations"], dtype=float)
    queries = np.asarray(validation["u"], dtype=float)
    if (native_report.get("accepted") is not True or not violations.size
            or queries.shape != violations.shape or not np.isfinite(queries).all()
            or not np.isfinite(violations).all() or np.any(violations > 0)
            or np.any(queries < 0) or np.any(queries > 1)):
        raise ValueError("Execution binding requires completed accepted native/export validation")
    candidate_id = str(uuid4())
    inputs = {"problem": plain(problem), "scene": plain(scene.to_execution_binding())}
    return {"candidate_id": candidate_id, "curve": curve_snapshot(trajectory),
        **deepcopy(inputs),
        "validation": {"candidate_id": candidate_id, "completed": True,
            "inputs": inputs,
            "scope": "native constraints and exported full curve; independent execution preflight pending",
            "export": plain(validation), "native": plain(native_report)},
        "execution_limits": None}


def check_result(result, *, execution_limits=None):
    """Reject a bare curve, stale report, changed clock or candidate mutation."""
    from .planning import PlanningResult

    if not isinstance(result, PlanningResult) or result.success is not True or result.trajectory is None:
        raise ValueError("Execution requires a successful formal PlanningResult")
    binding = result.execution_binding
    if (not isinstance(binding, dict) or not isinstance(binding.get("candidate_id"), str)
            or not binding["candidate_id"].strip()):
        raise ValueError("PlanningResult has no candidate/input execution binding")
    evidence = binding.get("validation", {})
    if (evidence.get("candidate_id") != binding["candidate_id"]
            or evidence.get("completed") is not True
            or evidence.get("native", {}).get("accepted") is not True
            or not isinstance(binding.get("scene"), dict)
            or not isinstance(binding.get("problem"), dict)):
        raise ValueError("Candidate and completed validation evidence do not agree")
    violations = np.asarray(evidence.get("export", {}).get("violations", []), dtype=float)
    queries = np.asarray(evidence.get("export", {}).get("u", []), dtype=float)
    if (not violations.size or queries.shape != violations.shape
            or not np.isfinite(violations).all() or np.any(violations > 0)
            or not np.isfinite(queries).all() or np.any(queries < 0) or np.any(queries > 1)):
        raise ValueError("Candidate export validation is missing or rejected")
    if binding.get("curve") != curve_snapshot(result.trajectory):
        raise ValueError("Candidate polynomial, command clock or executable scope changed after validation")
    if evidence.get("inputs") != {"problem": binding["problem"], "scene": binding["scene"]}:
        raise ValueError("Candidate scene or problem differs from the validated input binding")
    if execution_limits is not None:
        declared = plain(execution_limits.describe())
        if binding.get("execution_limits") is not None and binding["execution_limits"] != declared:
            raise ValueError("Candidate and executor use different physical execution limits")
    return deepcopy(binding)


# BEGIN explicit device rebinding preserves the already checked mathematical candidate.
def validate_controller_policy(value, *, joint_names):
    """Validate the one supported named trajectory-control interpretation."""
    keys = {"mode", "arm_joint_names", "kp", "kd", "native_timestep_s",
            "reference_derivatives", "sampling_phase", "acceleration_knot_side"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("An execution controller policy must explicitly declare every trajectory-control field")
    result = plain(value)
    names = tuple(joint_names)
    if (result["mode"] != "joint_trajectory_q_dq_ddq"
            or not isinstance(result["arm_joint_names"], list)
            or tuple(result["arm_joint_names"]) != names
            or result["reference_derivatives"] != ["q", "dq", "ddq"]
            or result["sampling_phase"] != "clamped_step_endpoint"
            or result["acceleration_knot_side"] != "right_internal_last_left"):
        raise ValueError("Execution requires the bound named q/dq/ddq endpoint controller policy")
    for key in ("kp", "kd"):
        try:
            gains = np.asarray(result[key], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Controller gains must be finite positive values in the complete arm joint order") from error
        if gains.shape != (len(names),) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("Controller gains must be finite positive values in the complete arm joint order")
    dt = result["native_timestep_s"]
    if isinstance(dt, bool) or not isinstance(dt, (float, int)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("The controller policy needs its finite positive native timestep")
    return result


def validate_result_only_goal_ids(binding, trajectory, values):
    """Select terminal result obligations without weakening online consumers.

    Every selected goal keeps its original identity and mathematical allowed
    set. Only a goal at the final skill progress can be a result-only goal;
    an explicit actual-arrival window always remains an online obligation.
    Process, current-feedback and effect-dependency checks are separate and
    are never disabled by this list.
    """
    if not isinstance(values, (list, tuple)):
        raise ValueError("Result-only goals require an explicit list of original goal identities")
    identities = binding.get("goal_identities", {})
    selected = []
    for value in values:
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or value[0] not in ("pose_events", "coordinate_events", "supports")
                or type(value[1]) is not int or value[1] < 0):
            raise ValueError("A result-only goal must name an original typed goal identity")
        identity = list(value)
        indexes = identities.get(value[0], [])
        if identity in selected:
            raise ValueError("Result-only goal identities must be unique and belong to this original binding")
        # A continuation keeps the full immutable policy even after verified
        # original goals have been removed from its remaining requirement list.
        if identity in binding.get("completed_goal_ids", []):
            selected.append(identity)
            continue
        if value[1] not in indexes:
            raise ValueError("Result-only goal identities must be unique and belong to this original binding")
        requirements = binding["problem"].get(value[0], [])
        if len(requirements) != len(indexes):
            raise ValueError("Result-only goal identities must retain their complete bound requirements")
        requirement = requirements[indexes.index(value[1])]
        if requirement.get("progress") != float(trajectory.progress_s[-1]):
            raise ValueError("Only a terminal goal without a remaining progress suffix can be result-only")
        for event in (*trajectory.events, *binding.get("timing_events", ())):
            if any(item.get("check_target") == "arrival" and tuple(item.get("goal_id") or ()) == tuple(identity)
                   for item in event.get("timing", ())):
                raise ValueError("An actual-arrival obligation cannot be changed into a result-only goal")
        selected.append(identity)
    return selected


def rebind_execution_result(result, *, controller_policy, result_only_goal_ids,
                            finger_targets, execution_limits, source):
    """Make a new explicit execution binding from unchanged formal evidence.

    This is a device-contract change at a separately declared entrance, not
    continuation of an executed world. The original curve, planning inputs,
    checks and cumulative request ledger remain unchanged. A completed native
    solve is not rerun or relabelled as an actual tracking measurement. The
    executor still has to verify current actual feedback and run its complete
    independent preflight before the first control write.
    """
    from .execution import ExecutionLimits, FixedFingerTargets
    from .planning import PlanningResult

    # The source reader verifies the archived mathematical binding without
    # granting the old control mode permission to run on the new backend.
    binding = check_result(result)
    if (not ((isinstance(source, str) and source.strip()) or (isinstance(source, dict) and source))
            or not isinstance(finger_targets, FixedFingerTargets)
            or not isinstance(execution_limits, ExecutionLimits)):
        raise ValueError("Execution rebinding needs explicit provenance and the complete typed device inputs")
    ledger = binding.get("request_ledger", {})
    if result.request_ledger is not None and result.request_ledger != ledger:
        raise ValueError("Execution rebinding cannot choose between inconsistent cumulative request ledgers")
    if ledger.get("requires_continuation") is True or binding.get("consumed_physics_steps", 0):
        raise ValueError("An executed request must continue from its actual world instead of device-only rebinding")
    policy = validate_controller_policy(controller_policy, joint_names=result.trajectory.joint_names)
    result_ids = validate_result_only_goal_ids(binding, result.trajectory, result_only_goal_ids)
    limits = execution_limits.describe()
    previous_limits = binding.get("execution_limits")
    if previous_limits is not None:
        previous_limits = ExecutionLimits.from_mapping(previous_limits).describe()
        if any(previous_limits[key] != limits[key] for key in limits
               if key not in {"source", "observation_policy"}):
            raise ValueError("Device rebinding cannot change the original physical protection limits")
        old_observation, new_observation = previous_limits["observation_policy"], limits["observation_policy"]
        if old_observation != new_observation and not (
                new_observation is None and old_observation is not None
                and old_observation["required_fields"] == []):
            raise ValueError("Device rebinding can remove only an optional observation channel, not required feedback")
    targets = {"joint_names": list(finger_targets.joint_names), "source": finger_targets.source,
               "events": list(deepcopy(finger_targets.events))}
    previous_policy = binding.get("device_policy")
    if previous_policy is not None and previous_policy.get("finger_targets") != targets:
        raise ValueError("Device rebinding cannot replace the original finger command policy")
    source_id = binding["candidate_id"]
    identity = str(uuid4())
    binding.update(candidate_id=identity, execution_limits=limits,
        device_policy={"finger_targets": targets, "execution_limits": limits,
                       "controller_policy": policy, "result_only_goal_ids": result_ids},
        execution_rebinding={"source": plain(source), "source_candidate_id": source_id,
            "source_controller_policy": None if previous_policy is None else previous_policy.get("controller_policy"),
            "source_execution_limits": previous_limits,
            "scope": "Unchanged formal curve/problem/scene evidence; new device contract requires actual entrance and independent preflight"})
    binding["validation"]["source_candidate_id"] = source_id
    binding["validation"]["candidate_id"] = identity
    # The ledger is copied exactly, including every planning attempt and any
    # original timing facts. No execution history is erased or manufactured.
    return PlanningResult(result.success, result.trajectory, deepcopy(result.diagnostics),
                          binding, deepcopy(ledger) if "request_ledger" in binding else None)
# END no implicit controller migration or new solver/physical budget.


def nominal_finger_sampler(scene_binding):
    """Reconstruct only a separately supplied nominal geometry polynomial."""
    from scipy.interpolate import CubicHermiteSpline

    hand = scene_binding.get("gripper")
    model = None if hand is None else hand.get("nominal_model")
    if model is None:
        return None
    t, q, dq = (np.asarray(model[key], dtype=float) for key in ("times", "position", "velocity"))
    if (t.ndim != 1 or len(t) < 2 or q.shape != dq.shape
            or q.shape != (len(t), len(hand["joint_names"]))
            or not all(np.isfinite(a).all() for a in (t, q, dq))
            or t[0] != 0 or np.any(np.diff(t) <= 0)):
        raise ValueError("Independent nominal finger model is incomplete")
    curve = CubicHermiteSpline(t, q, dq, extrapolate=False)

    def sample(time):
        if not np.isfinite(time) or not t[0] <= time <= t[-1]:
            raise ValueError("Nominal geometry query exceeds its bound clock")
        return curve(time), curve(time, 1)

    return sample
# END exact candidate, scene and validation handoff.
