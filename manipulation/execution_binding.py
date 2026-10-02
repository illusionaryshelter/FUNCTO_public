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
