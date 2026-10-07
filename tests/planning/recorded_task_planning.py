"""Record one production request without invoking additional planning checks.

Only the planner owns candidate evaluation and its single delivery acceptance.
This helper saves the returned exact curve and bounded diagnostics; a failed
candidate remains a visualization artifact without execution authorization.
"""

# BEGIN passive recording for the resident motion backend.
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np

MAX_CURVE_METADATA_BYTES = 512 * 1024**2
# END writers cannot emit metadata that the matching reader will refuse by size.


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


# BEGIN diagnostic documents keep nonfinite evidence and complete file writes.
def _diagnostic_value(value):
    """Convert owned CPU evidence without silently dropping NaN/Inf failures."""
    if isinstance(value, np.ndarray):
        return _diagnostic_value(value.tolist())
    if isinstance(value, np.generic):
        return _diagnostic_value(value.item())
    if isinstance(value, dict):
        return {key: _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_diagnostic_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _write_diagnostic_json(path, document):
    """Replace a complete JSON document; a failed write keeps the last one."""
    path = Path(path)
    rendered = json.dumps(_diagnostic_value(document), default=_json_value,
                          indent=2, allow_nan=False)+"\n"
    temporary = path.with_name(path.name+".tmp")
    try:
        temporary.write_text(rendered)
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def save_curve(path, curve, *, classification, execution_limits, fk=None, validation_status="completed",
               joint_limits=None, planning_result=None):
    """Preserve the exact exported cubic, event identities and optional FK.

    ``fk`` contains independently evaluated native FK of this same exported
    curve, in the problem reference frame. Native planner callers also supply
    their effective bounds; the shared production calculus then reports both
    the full curve and its original executable prefix. This recorder never
    changes classification based on an additional report.
    """
    # BEGIN accepted archives retain their exact formal authorization.
    from manipulation.execution_binding import check_result
    from manipulation.execution import ExecutionLimits
    binding = None
    if classification == "planner_accepted":
        if planning_result is None or planning_result.trajectory is not curve:
            raise ValueError("Accepted archives require this curve's formal PlanningResult")
        limits = ExecutionLimits.from_mapping(execution_limits)
        binding = check_result(planning_result, execution_limits=limits)
        binding["execution_limits"] = limits.describe()
        planning_result.execution_binding = deepcopy(binding)
    elif planning_result is not None:
        raise ValueError("Diagnostic archives cannot carry executable result authorization")
    # END rejected curves remain diagnostic data only.
    arrays = dict(joint_names=np.asarray(curve.joint_names), times=curve.times,
                  position=curve.position, velocity=curve.velocity,
                  progress_u=curve.progress_u, progress_s=curve.progress_s,
                  classification=np.asarray(classification))
    # BEGIN archive the already planned hand curve without regenerating it.
    fingers = curve.finger_trajectory
    if fingers is not None:
        arrays.update(finger_joint_names=np.asarray(fingers.joint_names),
                      finger_times=fingers.times, finger_position=fingers.position,
                      finger_velocity=fingers.velocity)
    # END commanded and actual finger positions remain separate artifacts.
    if fk is not None:
        arrays.update(fk)
    # BEGIN reuse production polynomial checks with the actual named limits.
    derivative_metadata = {}
    if joint_limits is not None:
        from manipulation.planning import cubic_derivative_report
        derivative_metadata = {
            "verification_joint_limits": joint_limits,
            "cubic_derivatives": {
                "full_curve": cubic_derivative_report(
                    curve, joint_limits["acceleration"], joint_names=joint_limits["joint_names"]),
                "executable_prefix": cubic_derivative_report(
                    curve, joint_limits["acceleration"], joint_names=joint_limits["joint_names"],
                    until=curve.executable_until),
            },
        }
    # END full/prefix diagnostics do not relabel or resample the saved curve.
    # BEGIN admit the complete metadata before creating either archive file.
    # ensure_ascii makes the character count the exact UTF-8 byte count without
    # allocating a second encoded copy of this potentially large document.
    rendered = json.dumps(
        {"classification": classification, "validation_status": validation_status,
         "execution_binding": binding,
         "execution_limits": deepcopy(execution_limits),
         "finger_trajectory": None if fingers is None else fingers.describe(),
         "satisfied_dependencies": list(curve.satisfied_dependencies),
         "events": curve.events,
         "conditions": curve.conditions, "residuals": curve.residuals,
         "duration_seconds": curve.duration,
         "executable_until_seconds": curve.executable_until,
         "next_verification": curve.next_verification, **derivative_metadata},
        default=_json_value, indent=2, allow_nan=False, ensure_ascii=True)+"\n"
    if len(rendered) > MAX_CURVE_METADATA_BYTES:
        raise ValueError("Trajectory metadata exceeds the bounded 512 MiB archive size")
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(rendered, encoding="utf-8")
    # END every geometry and validation field is retained without truncation.


def plan_and_record(problem, scene, robot, output, *, robot_validator, budget, provenance,
                    execution_limits, backend, joint_reference=None, continuation=None, rebind=None, request_ledger=None):
    """Call the bound backend once; persistence cannot initiate another solve.

    The caller owns the backend lifetime so successive requests can reuse its
    native models. Diagnostic curves have no formal binding and cannot enter
    the simulation executor. No monkeypatch, observer-side export, source
    geometry query, or candidate-sized tensor history is installed here.
    """
    import torch
    from manipulation import planning

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"scope": "resident_motion_backend", "status": "running", "success": False,
              "execution_tested": False, "budget": asdict(budget),
              "provenance": deepcopy(provenance), "execution_limits": deepcopy(execution_limits),
              "phase_count": problem.phase_count, "candidates": [], "observer_errors": []}
    try:
        result = planning.plan_trajectory(problem, scene, robot, robot_validator=robot_validator,
            budget=budget, backend=backend, joint_reference=joint_reference, continuation=continuation, rebind=rebind,
            request_ledger=request_ledger)
        # The backend hands off at most one owned CPU diagnostic curve. Remove
        # this non-JSON object before binding/report serialization; do not
        # modify its residuals or reevaluate it to populate an archive.
        diagnostic = result.diagnostics.pop("diagnostic_trajectory", None)
        report.update(status="completed", success=result.success, diagnostics=result.diagnostics)
        if result.request_ledger is not None:
            ledger_path = output / "request-ledger.json"
            _write_diagnostic_json(ledger_path, result.request_ledger)
            report["request_ledger_path"] = str(ledger_path.resolve())
        bounds = result.diagnostics.get("verification_joint_limits")
        if result.trajectory is not None:
            save_curve(output / "trajectory.npz", result.trajectory,
                classification="planner_accepted", execution_limits=execution_limits,
                joint_limits=bounds, planning_result=result)
        elif diagnostic is not None:
            entry = {"path": "diagnostic-trajectory.npz", "classification": "rejected_diagnostic_not_executable",
                     "validation_status": "not_completed", "curve_saved": False,
                     "attempt": {"failure": result.diagnostics.get("failure"),
                                 "residuals": diagnostic.residuals}}
            report["candidates"].append(entry)
            try:
                save_curve(output / entry["path"], diagnostic, classification=entry["classification"],
                    execution_limits=execution_limits, validation_status=entry["validation_status"])
                entry["curve_saved"] = True
            except Exception as error:
                entry["diagnostic_error"] = f"{type(error).__name__}: {error}"
        return result
    except KeyboardInterrupt:
        report.update(status="interrupted", interruption_reason="Explicit operator interrupt")
        (output / "exception.txt").write_text(traceback.format_exc())
        raise
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (output / "exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if torch.cuda.is_initialized():
            report.update(peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
        _write_diagnostic_json(output / "planning-report.json", report)
# END record only the production request's own result and authorization.
