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
from manipulation.evidence_archive import externalize_evidence, referenced_evidence_bytes

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


def _write_json_value(path, document):
    """Replace one complete value document using the existing numeric rules."""
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


# BEGIN immutable completed evidence is archived once per result directory.
def _write_diagnostic_json(path, document):
    """Write a report and relative references to its full shared evidence."""
    path = Path(path)
    _write_json_value(path, externalize_evidence(document, path.parent, write_record=_write_json_value))
# END every referenced file contains the complete original record.


def save_curve(path, curve, *, classification, execution_limits, fk=None, validation_status="completed",
               joint_limits=None, planning_result=None, source=None):
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
        # BEGIN check_result already returns an owned mutable binding header.
        planning_result.execution_binding = binding
        # END immutable completed evidence remains shared with its diagnostics.
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
        # BEGIN reuse only the exact curve and limits checked by this binding.
        full = (binding["validation"]["export"]["cubic_derivatives"]
            if binding is not None and joint_limits == binding.get("verification_joint_limits")
            else cubic_derivative_report(curve, joint_limits["acceleration"],
                                         joint_names=joint_limits["joint_names"]))
        prefix = (full if curve.executable_until == curve.duration else
            cubic_derivative_report(curve, joint_limits["acceleration"],
                joint_names=joint_limits["joint_names"], until=curve.executable_until))
        # END a genuinely different interval still needs its own extrema.
        derivative_metadata = {
            "verification_joint_limits": joint_limits,
            "cubic_derivatives": {
                "full_curve": full,
                "executable_prefix": prefix,
            },
        }
    # END full/prefix diagnostics do not relabel or resample the saved curve.
    # BEGIN admit the complete metadata before creating either archive file.
    # ensure_ascii makes the character count the exact UTF-8 byte count without
    # allocating a second encoded copy of this potentially large document.
    # BEGIN the curve metadata names the same complete evidence files as its report.
    metadata = {"classification": classification, "validation_status": validation_status,
         # BEGIN source names the candidate that owns this archive's polynomial.
         "source": source,
         # END qualification remains in the independent execution binding.
         "execution_binding": binding,
         "execution_limits": deepcopy(execution_limits),
         "finger_trajectory": None if fingers is None else fingers.describe(),
         "satisfied_dependencies": list(curve.satisfied_dependencies),
         "events": curve.events,
         "conditions": curve.conditions, "residuals": curve.residuals,
         "duration_seconds": curve.duration,
         "executable_until_seconds": curve.executable_until,
         "next_verification": curve.next_verification, **derivative_metadata}
    metadata = externalize_evidence(metadata, path.parent, write_record=_write_json_value)
    rendered = json.dumps(metadata,
        default=_json_value, indent=2, allow_nan=False, ensure_ascii=True)+"\n"
    # END numerical arrays and acceptance data remain unchanged by file sharing.
    # BEGIN the original archive bound includes each unique complete side record.
    if len(rendered) + referenced_evidence_bytes(metadata, path.parent) > MAX_CURVE_METADATA_BYTES:
        raise ValueError("Trajectory metadata exceeds the bounded 512 MiB archive size")
    # END sharing avoids duplicate storage, not accounting for retained data.
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
                     # BEGIN archive the production curve's actual evidence status.
                     "validation_status": result.diagnostics.get("diagnostic_validation_status", "not_completed"),
                     "source": result.diagnostics.get("diagnostic_candidate_source"), "curve_saved": False,
                     # END a checked rejected seed remains distinct from an optimizer trial.
                     "attempt": {"failure": result.diagnostics.get("failure"),
                                 "residuals": diagnostic.residuals}}
            report["candidates"].append(entry)
            try:
                save_curve(output / entry["path"], diagnostic, classification=entry["classification"],
                    execution_limits=execution_limits, validation_status=entry["validation_status"],
                    source=entry["source"])
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
