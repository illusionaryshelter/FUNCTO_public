"""Run the current planner and retain inspectable evidence for a recorded task.

This test helper accepts already constructed production inputs. It does not
choose a dataset, manufacture an observation, weaken a constraint, or execute
a rejected curve. Failed native candidates remain explicitly diagnostic. The
caller owns the observed scene, robot binding and separate physics evaluator.
"""

# BEGIN current-planner invocation with scoped diagnostic curve recording.
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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


class _ExportValidationRecorder:
    """Own one export's bounded CPU summaries, never an acceptance decision.

    Source-world reports already retain one worst state over all completed
    chunks. Source-self reports retain one worst state per chunk; this recorder
    keeps only the overall first worst state, using the production rank. No
    trajectory-sized inventory of pair reports or native tensor is retained.
    """

    def __init__(self, path, *, native_result, seed_index, observer_errors):
        self.path, self.observer_errors = Path(path), observer_errors
        self.document = {
            "schema": "functo.export_validation_progress.1",
            "classification": "partial_validation_diagnostic_not_executable",
            "native_result": native_result, "seed_index": seed_index,
            "status": "started", "validation_complete": False,
            "completed_batches": 0, "planned_points": None, "checked_points": 0,
            "curve_duration_s": None, "checked_time_interval_s": None,
            "source_world_collision": {},
            "source_self_collision": {"checked_states": 0,
                "unmodelled_geometry_names": [], "worst_state": None},
            "process_constraints": {"status": "not_archived",
                "reason": "Full chronological process residuals are archived only after export returns"},
            "native_joint_constraints": {"status": "not_archived",
                "reason": "Partial evidence does not add native tensor reductions or host copies"},
            "exception": None, "budget_exhausted": False,
            "scope": "Completed finite batches only; checked time endpoints do not certify the continuous interval. "
                "World checked_states includes exact/left/right modes in original residual order, not unique Coal calls. "
                "NaN/Infinity are explicit string values. No record authorizes execution.",
        }
        self._write()

    def _write(self):
        try:
            _write_diagnostic_json(self.path, self.document)
        except Exception as error:
            # Keep error storage bounded even if every later disk write fails.
            message = f"Export progress {self.path.name}: {type(error).__name__}: {error}"
            if message not in self.observer_errors:
                self.observer_errors.append(message)

    def observe(self, progress):
        """Copy this batch before later queries reuse their mutable reports."""
        try:
            values = deepcopy(progress)
            current = values.pop("source_self_collision_batch")
            if current is not None:
                self.document["completed_batches"] += 1
                retained = self.document["source_self_collision"]
                retained["checked_states"] += current["checked_states"]
                retained["unmodelled_geometry_names"] = sorted(set(
                    retained["unmodelled_geometry_names"]) | set(current["unmodelled_geometry_names"]))
                previous, candidate = retained["worst_state"], current["worst_state"]
                def rank(witness):
                    clearance = witness["minimum_required_clearance_m"]
                    return witness["residual"], -float("inf") if clearance is None else -clearance
                if candidate is not None and (previous is None or rank(candidate) > rank(previous)):
                    retained["worst_state"] = candidate
            self.document.update(values, status="partial")
            self._write()
        except Exception as error:
            message = f"Export progress {self.path.name}: {type(error).__name__}: {error}"
            if message not in self.observer_errors:
                self.observer_errors.append(message)

    def finish(self, *, error=None, residuals=None):
        """Completion means export returned, independently of native success."""
        from manipulation.planning import PlanningWorkExhausted

        try:
            if error is None:
                self.document.update(status="completed", validation_complete=True,
                    classification="completed_validation_diagnostic_not_executable",
                    full_validation_residuals=deepcopy(residuals),
                    process_constraints={"status": "completed"},
                    native_joint_constraints={"status": "completed_in_full_validation_residuals"})
            else:
                self.document.update(status="partial", validation_complete=False,
                    exception={"type": type(error).__name__, "message": str(error)},
                    budget_exhausted=isinstance(error, PlanningWorkExhausted))
            self._write()
        except Exception as diagnostic_error:
            message = f"Export finish {self.path.name}: {type(diagnostic_error).__name__}: {diagnostic_error}"
            if message not in self.observer_errors:
                self.observer_errors.append(message)
# END recorder I/O and metadata failures never become planning failures.


# BEGIN preserve native representations before candidate gathering/export.
def _cpu_array(value):
    """Detach an owned CPU copy; recording cannot alias reusable solver buffers."""
    import torch

    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Diagnostic arrays must not contain Python objects")
    return array.copy()


def save_native_result(path, result, joint_names):
    """Save all returned seeds before gather drops their native intermediates.

    Shapes remain native. In particular, ``solution`` contains B-spline
    controls, whereas ``js_solution`` contains the rollout. The interpolated
    buffer includes padding; only ``interpolated_last_tstep`` defines each
    seed's valid prefix. No time array is invented from another layer's dt.
    """
    path = Path(path)
    arrays = {"active_joint_names": np.asarray(joint_names)}
    metadata = {"batch_size": int(result.batch_size), "num_seeds": int(result.num_seeds),
                "classification": "native_diagnostic_not_executable",
                "control_point_semantics": "solution is not a joint trajectory",
                "states": {}}
    for name in ("solution", "optimized_seeds", "success", "seed_rank", "seed_cost",
                 "interpolated_last_tstep"):
        value = getattr(result, name)
        if value is not None:
            arrays[name] = _cpu_array(value)
    for name in ("js_solution", "interpolated_trajectory"):
        state = getattr(result, name)
        metadata["states"][name] = None if state is None else {"joint_names": state.joint_names}
        if state is None:
            continue
        for field in ("position", "velocity", "acceleration", "jerk", "dt", "knot_dt"):
            value = getattr(state, field)
            if value is not None:
                arrays[f"{name}.{field}"] = _cpu_array(value)
    metadata["arrays"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                          for key, value in arrays.items()}
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, default=_json_value,
                                                   indent=2, allow_nan=False)+"\n")


# BEGIN preserve the live verifier's effective named scalar bounds.
def verification_joint_limits(planner):
    """Copy constructed native bounds without guessing from robot YAML.

    Config dataclasses can still contain ``joint_limits=None`` because each
    live cost receives its bounds from the native transition model. Read that
    actual cost instance, preserve its joint order/dtype, and require every
    active joint exactly once before any curve report uses the matrices.
    """
    cost = planner._verification_native.get_cost("cspace")
    if cost is None or not cost.enabled:
        raise ValueError("Effective bounds require the enabled native verification cspace cost")
    limits = cost.config.joint_limits
    names = tuple(limits.joint_names)
    if (not names or len(set(names)) != len(names) or len(set(planner.joint_names)) != len(planner.joint_names)
            or set(names) != set(planner.joint_names)):
        raise ValueError("Effective native limits must name every active joint exactly once")
    result = {"source": "live planner._verification_native.get_cost('cspace').config.joint_limits",
              "joint_names": list(names)}
    for label in ("position", "velocity", "acceleration", "jerk"):
        array = _cpu_array(getattr(limits, label))
        if (array.shape != (2, len(names)) or not np.isfinite(array).all()
                or np.any(array[0] > array[1])):
            raise ValueError(f"Effective native {label} bounds require finite ordered lower/upper rows")
        result[label] = array
    return result
# END no parsed-input fallback for effective native limits.


def save_native_configuration(path, planner):
    """Record constructed configs, without traversing live solver objects.

    cuRobo configuration dataclasses contain tensors and Python class types;
    save tensors in NPZ and classes by qualified name. The repeated native
    robot_config is represented by the checked input robot dictionary below.
    Its parsed runtime objects are intentionally outside this config record;
    the live verifier's four effective scalar bound matrices are saved below
    as a separate, explicitly named field.
    Unsupported values fail with their precise field path, rather than a
    repr string which could falsely suggest the configuration was preserved.
    """
    import torch

    arrays = {}

    def encode(value, key):
        if isinstance(value, Enum):
            return {"enum": f"{type(value).__module__}.{type(value).__qualname__}.{value.name}"}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (np.ndarray, torch.Tensor)):
            arrays[key] = _cpu_array(value)
            return {"npz_key": key, "shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)}
        if isinstance(value, (Path, torch.device, torch.dtype)):
            return str(value)
        if isinstance(value, type):
            return {"class": f"{value.__module__}.{value.__qualname__}"}
        if is_dataclass(value):
            return {field.name: ({"reference": "bounded_robot_input",
                                  "scope": "parsed robot internals not serialized"}
                                 if field.name == "robot_config" else
                                 encode(getattr(value, field.name), f"{key}.{field.name}"))
                    for field in fields(value)}
        if isinstance(value, dict) and all(isinstance(name, str) for name in value):
            return {name: encode(item, f"{key}.{name}") for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item, f"{key}.{index}") for index, item in enumerate(value)]
        raise TypeError(f"Unsupported native config {key}: {type(value).__name__}")

    values = {"ik": planner.ik.config, "trajopt": planner.trajopt.config,
              "bounded_robot_input": planner.robot_dict,
              "active_joint_names": planner.joint_names,
              "transition_horizon": planner.trajopt.horizon,
              "action_horizon": planner.trajopt.action_horizon,
              "progress_u": planner.progress_map.u, "progress_s": planner.progress_map.s,
              # BEGIN distinguish the formal execution clock from raw rollouts.
              "formal_clock": planner.formal_clock,
              # END every hand reference uses this same native output clock.
              # BEGIN retain runtime bounds in addition to existing configs.
              "verification_joint_limits": verification_joint_limits(planner)}
              # END native names and all four matrices keep their actual dtype.
    document = encode(values, "configuration")
    # JSON serialization happens before either file is written: NaN or an
    # unsupported configuration cannot leave an apparently complete record.
    rendered = json.dumps(document, indent=2, allow_nan=False)+"\n"
    path = Path(path)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    path.with_suffix(".json").write_text(rendered)
# END preserve native representations before candidate gathering/export.


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
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(
        {"classification": classification, "validation_status": validation_status,
         "execution_binding": binding,
         "execution_limits": deepcopy(execution_limits),
         "finger_trajectory": None if fingers is None else fingers.describe(),
         "events": curve.events,
         "conditions": curve.conditions, "residuals": curve.residuals,
         "duration_seconds": curve.duration,
         "executable_until_seconds": curve.executable_until,
         "next_verification": curve.next_verification, **derivative_metadata},
        default=_json_value, indent=2, allow_nan=False)+"\n")


# BEGIN rejected curves retain the exact scene and query scope without authorization.
def save_collision_snapshot(path, planner, curve, *, curve_path, validation_status, query_capture):
    """Archive this candidate's owned inputs for independent CPU diagnostics.

    Reuse the production scene serializer, including observed target geometry,
    background, pair permissions and the already bound nominal finger curve.
    The full polynomial and event clock identify the saved curve exactly.
    Neither this file nor a later independent pass grants execution authority.
    An unfinished export has no completed validation report: the planner may
    still hold ``last_validation`` from an entirely different native seed.
    """
    from manipulation.execution_binding import curve_snapshot, plain

    # BEGIN native rejection deliberately omits independent export validation.
    if validation_status not in {"completed", "not_completed", "skipped_native_rejection"}:
        raise ValueError("Collision snapshots require an explicit export validation status")
    # END skipped validation cannot inherit another seed's last_validation.
    # BEGIN one authoritative copy of potentially large query records per file.
    # The scoped observer already owns the exact production records, including
    # partial failed exports. Validation keeps metrics, bounds and violations;
    # scene_query_capture keeps the actual query arrays and layer decisions.
    validation = (plain({key: value for key, value in planner.last_validation.items()
                         if key != "scene_layer_queries"})
                  if validation_status == "completed" else None)
    # END no query payload is duplicated inside the validation dictionary.
    document = {"classification": "diagnostic_collision_snapshot_not_executable",
        "curve_path": Path(curve_path).name, "curve": curve_snapshot(curve),
        "problem": plain(planner.problem),
        "scene": plain(planner._verification_scene.to_execution_binding()),
        "formal_clock": plain(planner.formal_clock),
        "validation_status": validation_status, "validation": validation,
        "scene_query_capture": plain(query_capture),
        "scope": "same candidate, verification scene and actual sampled query inputs; independent solid audit pending"}
    # Render first, so unsupported values cannot leave a plausible partial JSON
    # record. The caller records a failed write explicitly beside this curve.
    rendered = json.dumps(document, indent=2, allow_nan=False)+"\n"
    Path(path).write_text(rendered)
# END diagnostics preserve scene identity without creating an execution binding.


# BEGIN recorded planning carries the required source-solid validator unchanged.
def plan_and_record(problem, scene, robot, output, *, robot_validator, budget, provenance, execution_limits,
                    continuation=None, rebind=None, request_ledger=None):
    """Call ``plan_trajectory`` once and preserve its measured outcome.

    The scoped observer copies checked continuations and exact native curves
    whose final verification could not finish. Their validation statuses are
    distinct. It never changes selection or lets a rejected diagnostic curve
    escape as ``PlanningResult.trajectory``. All CUDA work stays within the
    caller's explicit single-process resource budget.
    """
    import torch
    from curobo.kinematics import Kinematics
    from curobo.types import JointState
    from manipulation import planning

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"scope": "current FUNCTO joint planning on explicitly supplied task inputs",
              "status": "running", "success": False, "execution_tested": False,
              "budget": asdict(budget), "provenance": deepcopy(provenance),
              "execution_limits": deepcopy(execution_limits),
              "skill_ids": sorted({event.skill_id for event in problem.events}),
              "phase_count": problem.phase_count,
              "required_source_anchors": len(problem.anchors),
              "candidates": [], "native_results": [], "effective_configurations": [],
              # BEGIN native and partial reports have independent file identities.
              "optimizer_inputs": [], "export_validations": [], "observer_errors": [],
              # END failed exports cannot erase metrics recorded before them.
              # BEGIN expose missing scene evidence even when construction fails.
              "initialization": {"status": "not_started"},
              "collision_snapshot_scope": "each retained candidate; unselected native seeds remain native-result diagnostics"}
              # END a run without an initialized planner cannot claim a snapshot.
    original = planning._CuroboPlanner.select_candidate
    original_export = planning._CuroboPlanner.export
    # BEGIN reuse the production's single per-seed metrics calculation.
    original_native_diagnostics = planning._CuroboPlanner.native_diagnostics
    native_metrics_document, current_native_seed = None, None
    export_progress_records = {}
    # END no diagnostic invokes a second native metrics pass.
    original_curve_init = planning.JointTrajectory.__post_init__
    native_trace = None
    configuration_index = 0
    # BEGIN retain an owned CPU copy before the native planner is released.
    selected_joint_limits = None
    # END no live solver reference is retained by the recording adapter.
    # BEGIN query copies belong to one selection round and the exact curve object.
    export_query_captures = {}
    # END no prior seed's successful validation fills a failed candidate record.

    def record_stage(stage):
        try:
            (output/"progress.json").write_text(json.dumps(stage)+"\n")
            print(json.dumps(stage), flush=True)
        except OSError as error:
            report.setdefault("observer_errors", []).append(f"Stage record: {type(error).__name__}: {error}")

    def stage_observer(method, name):
        """Expose coarse native stages without polling or changing solver work."""
        def measured(planner, *args, **kwargs):
            stage = {"stage": name, "state": "started",
                     "elapsed_seconds": time.monotonic()-started}
            # BEGIN report initialization failure before any candidate can exist.
            if name == "_initialize":
                report["initialization"] = {"status": "started"}
            # END stage failures are evidence gaps, not empty successful scenes.
            record_stage(stage)
            # BEGIN copy the exact optimizer input before native buffer reuse.
            if name == "optimize":
                path = output/f"optimizer-input-{len(report['optimizer_inputs']):02d}.npz"
                entry = {"path": path.name, "saved": False}
                try:
                    np.savez_compressed(path, seed_controls=_cpu_array(args[0]),
                        joint_names=np.asarray(planner.joint_names),
                        start_q=_cpu_array(planner.start.position),
                        start_dq=_cpu_array(planner.start.velocity),
                        progress_u=planner.progress_map.u.copy(),
                        progress_s=planner.progress_map.s.copy())
                    entry["saved"] = True
                except Exception as error:
                    entry["error"] = f"{type(error).__name__}: {error}"
                report["optimizer_inputs"].append(entry)
            # END copy the exact optimizer input before native buffer reuse.
            # BEGIN preserve a failed construction stage without changing its exception.
            try:
                result = method(planner, *args, **kwargs)
            except Exception as error:
                stage.update(state="failed", elapsed_seconds=time.monotonic()-started,
                             error=f"{type(error).__name__}: {error}")
                if name == "_initialize":
                    report["initialization"] = {"status": "failed", "error": stage["error"],
                        "collision_snapshot_unavailable": "No fully initialized candidate verification scene"}
                record_stage(stage)
                raise
            if name == "_initialize":
                report["initialization"] = {"status": "completed"}
            # END archive recording never turns a construction failure into a plan.
            # BEGIN persist the measured entrance before potentially long IK.
            # This copies the production verdict once; interrupted later work
            # must not erase its evidence or cause a second geometry query.
            if name == "initial_state_diagnostics":
                try:
                    (output/"initial-state-diagnostics.json").write_text(json.dumps(
                        result, default=_json_value, indent=2, allow_nan=False)+"\n")
                except Exception as error:
                    report.setdefault("observer_errors", []).append(
                        f"Entrance record: {type(error).__name__}: {error}")
            # END entrance recording does not alter its production verdict.
            # BEGIN save effective settings after construction and each solve.
            # Refinement can replace the optimizer. Capture the instance that
            # actually produced the next result, not only its input YAML.
            if name in ("_initialize", "optimize"):
                nonlocal configuration_index
                path = output/f"effective-config-{configuration_index:02d}"
                configuration_index += 1
                entry = {"path": path.with_suffix(".json").name, "stage": name, "saved": False}
                try:
                    save_native_configuration(path, planner)
                    entry["saved"] = True
                except Exception as error:
                    entry["error"] = f"{type(error).__name__}: {error}"
                report["effective_configurations"].append(entry)
            # END save effective settings after construction and each solve.
            stage.update(state="completed", elapsed_seconds=time.monotonic()-started)
            record_stage(stage)
            return result
        return measured

    def preserve_candidate(planner, curve, attempt, *, validation_status):
        """Keep a constructed native curve without changing its acceptance."""
        nonlocal selected_joint_limits
        selected_joint_limits = None
        index = len(report["candidates"])
        # BEGIN selection observers do not yet own a formal result binding.
        classification = ("validated_candidate_pending_result" if attempt["accepted"]
                          else "rejected_diagnostic_not_executable")
        # END only the final returned result is saved as planner_accepted.
        path = output/f"candidate-{index:02d}.npz"
        entry = {"path": path.name, "classification": classification,
                 "attempt": deepcopy(attempt), "validation_status": validation_status,
                 "curve_saved": False, "native_result": native_trace}
        # BEGIN link only this exact curve's independently persisted progress.
        partial = export_progress_records.get(id(curve))
        if partial is not None and partial[0] is curve:
            entry["export_validation_path"] = partial[1]
        if native_trace is not None:
            entry["native_metrics_path"] = Path(native_trace).stem+"-metrics.json"
        # END paths identify diagnostic evidence, not execution authority.
        # BEGIN each saved curve names its own full observed scene and exact modes.
        snapshot_path = path.with_name(path.stem+"-scene.json")
        snapshot = {"path": snapshot_path.name, "saved": False}
        entry["collision_snapshot"] = snapshot
        capture = export_query_captures.get(id(curve))
        query_capture = ({"status": "not_recorded", "calls": [],
                          "errors": ["No observation belongs to this exact exported curve"]}
                         if capture is None or capture[0] is not curve else capture[1])
        try:
            save_collision_snapshot(snapshot_path, planner, curve, curve_path=path,
                validation_status=validation_status, query_capture=query_capture)
            snapshot.update(saved=True, query_capture_status=query_capture["status"])
        except Exception as error:
            snapshot["error"] = f"{type(error).__name__}: {error}"
        # END snapshot failure is explicit and cannot change planner acceptance.
        try:
            # BEGIN retain the exact verifier that produced this candidate.
            selected_joint_limits = verification_joint_limits(planner)
            # END no previous candidate's limits can fill a missing record.
            # Save CPU evidence before optional display FK. A visualization
            # failure must not change the production candidate or its checks.
            save_curve(path, curve, classification=classification, validation_status=validation_status,
                       execution_limits=execution_limits, joint_limits=selected_joint_limits)
            entry["curve_saved"] = True
            entry["analytic_derivatives_saved"] = True
            # A private native FK buffer leaves the optimizer's reusable
            # buffers untouched. Preserve the exported curve's named order.
            fk = Kinematics(planner.ik.kinematics.config,
                            compute_spheres=False, compute_com=False)
            query = planner.device_cfg.to_device(curve.position)[None].contiguous()
            with torch.no_grad():
                state = fk.compute_kinematics(JointState.from_position(
                    query, joint_names=list(curve.joint_names)))
                link_p, link_q = state.tool_poses.position, state.tool_poses.quaternion
                tool_index = fk.tool_frames.index(problem.tool_link)
                position, quaternion = problem.transform_tool_pose(
                    link_p[:, :, tool_index], link_q[:, :, tool_index])
                sampled = {"eef_position": position[0].cpu().numpy().copy(),
                           "eef_quaternion_wxyz": quaternion[0].cpu().numpy().copy()}
            save_curve(path, curve, classification=classification, fk=sampled,
                       validation_status=validation_status, execution_limits=execution_limits,
                       joint_limits=selected_joint_limits)
            entry["display_fk_saved"] = True
        except Exception as error:
            entry["diagnostic_error"] = f"{type(error).__name__}: {error}"
            entry["diagnostic_traceback"] = traceback.format_exc()
        report["candidates"].append(entry)

    def observed_export(planner, native_result):
        # Production constructs its exact cubic before bounded verification.
        # Preserve that same object if validation cannot finish; do not retry
        # validation with weaker settings or return it as a successful plan.
        constructed = []
        # BEGIN each export owns a fresh partial record, even when it fails early.
        progress_path = output/f"export-{len(report['export_validations']):02d}-validation.json"
        progress = _ExportValidationRecorder(progress_path, native_result=native_trace,
            seed_index=current_native_seed, observer_errors=report["observer_errors"])
        progress_entry = {"path": progress_path.name, "native_result": native_trace,
                          "seed_index": current_native_seed, "validation_complete": False}
        report["export_validations"].append(progress_entry)
        # END per-batch atomic files survive errors before the final run report.
        # BEGIN copy production's own expanded query record after each chunk.
        query_capture = {"status": "not_recorded", "calls": [], "errors": []}
        original_boundary_query = planning._scene_boundary_evaluate

        def capture_boundary(*args, **kwargs):
            if "layer_selection_out" not in kwargs:
                kwargs["layer_selection_out"] = {}
            value = original_boundary_query(*args, **kwargs)
            try:
                query_capture["calls"].append(deepcopy(kwargs["layer_selection_out"]))
            except Exception as error:
                query_capture["errors"].append(f"{type(error).__name__}: {error}")
            query_capture["status"] = "error" if query_capture["errors"] else "completed"
            return value
        # END a release-history scene clone keeps its own bound evaluate method.

        def capture(curve):
            original_curve_init(curve)
            constructed.append(curve)
            export_progress_records[id(curve)] = (curve, progress_path.name)

        try:
            # BEGIN the recorder is scoped to this export's private verification scene.
            with patch.object(planning.JointTrajectory, "__post_init__", new=capture), \
                    patch.object(planning, "_scene_boundary_evaluate", new=capture_boundary), \
                    patch.object(planner, "_export_progress_observer", new=progress.observe, create=True):
                exported = original_export(planner, native_result)
            export_query_captures[id(exported[0])] = (exported[0], query_capture)
            progress.finish(residuals=exported[0].residuals)
            progress_entry["validation_complete"] = True
            return exported
            # END other candidate scenes and the optimizer remain unmodified.
        # BEGIN preserve the original error category, including request exhaustion.
        except BaseException as error:
            progress.finish(error=error)
            progress_entry["exception"] = {"type": type(error).__name__, "message": str(error)}
            # An explicit interrupt retains already persisted CPU progress but
            # does not initiate another display FK after the stop request.
            if isinstance(error, Exception) and len(constructed) == 1:
                try:
                    # BEGIN unfinished validation retains only its own observed query inputs.
                    export_query_captures[id(constructed[0])] = (constructed[0], query_capture)
                    # END save_collision_snapshot deliberately omits stale last_validation.
                    seed_metrics = (None if native_metrics_document is None else next(
                        (item["metrics"] for item in native_metrics_document["seeds"]
                         if item["seed_index"] == current_native_seed), None))
                    preserve_candidate(planner, constructed[0],
                        {"accepted": False, "seed_index": current_native_seed,
                         "native_success": None if seed_metrics is None else seed_metrics["success"],
                         "native_metrics": deepcopy(seed_metrics),
                         "export_error": str(error), "export_error_type": type(error).__name__},
                        validation_status="not_completed")
                except Exception as diagnostic_error:
                    report.setdefault("observer_errors", []).append(
                        f"Rejected curve record: {type(diagnostic_error).__name__}: {diagnostic_error}")
            raise
        # END partial witnesses never acquire a completed validation status.

    # BEGIN persist the existing native diagnostic before the corresponding export.
    def observed_native_diagnostics(planner, native_result, seed_index=None):
        nonlocal current_native_seed
        metrics = original_native_diagnostics(planner, native_result, seed_index=seed_index)
        current_native_seed = seed_index
        try:
            if native_metrics_document is None or seed_index is None:
                raise ValueError("Native metrics require their current result and explicit seed identity")
            entry = {"seed_index": int(seed_index), "metrics": deepcopy(metrics),
                     "positive_constraint_names": {}, "nonfinite_metric_names": []}
            for label in ("rollout", "interpolated"):
                if label not in metrics:
                    continue
                entry["positive_constraint_names"][label] = [name for name, value in
                    metrics[label]["constraints"].items() if np.isfinite(value) and value > 0.]
                for kind in ("constraints", "convergence"):
                    entry["nonfinite_metric_names"].extend(f"{label}.{kind}.{name}"
                        for name, value in metrics[label][kind].items() if not np.isfinite(value))
            native_metrics_document["seeds"].append(entry)
            native_metrics_document["acceptance_semantics"] = {
                "constraints": "Native summed constraints over the full horizon must be <= 0; "
                    "archived values are per-name maxima, so positive names identify violated nonnegative terms",
                "constraint_threshold": 0.,
                "convergence": "Native position/orientation convergence uses only the last timestep and strict <. "
                    "Archived convergence values are whole-horizon maxima, not final-timestep failure flags",
                "position_tolerance": float(planner.trajopt.config.position_tolerance),
                "orientation_tolerance": float(planner.trajopt.config.orientation_tolerance),
                "success": "rollout feasibility AND enabled terminal convergence AND interpolated feasibility",
            }
            metrics_path = output/(Path(native_trace).stem+"-metrics.json")
            _write_diagnostic_json(metrics_path, native_metrics_document)
        except Exception as error:
            report["observer_errors"].append(f"Native metrics record: {type(error).__name__}: {error}")
        return metrics
    # END the recorder never repeats native evaluation or mutates its result.

    # BEGIN recording preserves seed/block selection options and result identity.
    def observed(planner, native_result, **selection_options):
        # BEGIN retain every native seed before production gathers a candidate.
        nonlocal native_trace, native_metrics_document, current_native_seed
        # BEGIN bound retained query memory to the existing native seed count.
        export_query_captures.clear()
        export_progress_records.clear()
        # END snapshots from a previous optimization round are already on disk.
        path = output/f"native-result-{len(report['native_results']):02d}.npz"
        native_trace = path.name
        current_native_seed = None
        native_metrics_document = {
            "schema": "functo.native_candidate_metrics.1",
            "classification": "native_diagnostic_not_executable",
            "native_result": path.name, "expected_seeds": int(native_result.num_seeds),
            "seeds": [], "nonfinite_encoding": "NaN, Infinity and -Infinity are explicit strings",
        }
        entry = {"path": path.name, "saved": False,
                 "metrics_path": path.stem+"-metrics.json",
                 "optimizer_input": report["optimizer_inputs"][-1]["path"]
                 if report["optimizer_inputs"] else None,
                 "effective_configuration": report["effective_configurations"][-1]["path"]
                 if report["effective_configurations"] else None}
        try:
            save_native_result(path, native_result, planner.joint_names)
            entry["saved"] = True
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
        report["native_results"].append(entry)
        # END retain every native seed before production gathers a candidate.
        # The same selector now observes initial seeds and optimizer blocks.
        # Forward its candidate subset, first-feasible policy and origin as
        # supplied; the recorder has no authority to broaden that selection.
        selection = original(planner, native_result, **selection_options)
        _, curve, attempt = selection
        # BEGIN archive the production selection's actual independent-check status.
        preserve_candidate(planner, curve, attempt,
                           validation_status=attempt["independent_validation"])
        # END a cheap native-rejected curve has no completed validation report.
        return selection
    # END selector exceptions propagate without substitution or a second call.

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(planning._CuroboPlanner, "select_candidate", new=observed))
            stack.enter_context(patch.object(planning._CuroboPlanner, "export", new=observed_export))
            stack.enter_context(patch.object(planning._CuroboPlanner, "native_diagnostics",
                                             new=observed_native_diagnostics))
            # BEGIN record the entrance at its original evaluation point.
            for name in ("_initialize", "initial_state_diagnostics", "reference_seeds", "optimize"):
                method = getattr(planning._CuroboPlanner, name)
                stack.enter_context(patch.object(planning._CuroboPlanner, name,
                                                 new=stage_observer(method, name)))
            # END the observer does not repeat or bypass any entrance check.
            # BEGIN continuation consumes the original ledger and request budget.
            result = planning.plan_trajectory(problem, scene, robot, robot_validator=robot_validator, budget=budget,
                continuation=continuation, rebind=rebind, request_ledger=request_ledger)
            # END recording never substitutes the robot's physical pair policy.
            # END a new actual binding is never an implicit budget reset.
        report.update(status="completed", success=result.success,
                      diagnostics=result.diagnostics)
        # BEGIN failed planning attempts retain the same reusable work ledger.
        report["request_ledger_path"] = None
        if result.request_ledger is not None:
            ledger_path = output / "request-ledger.json"
            ledger_path.write_text(json.dumps(result.request_ledger, default=_json_value,
                                             indent=2, allow_nan=False)+"\n")
            report["request_ledger_path"] = str(ledger_path.resolve())
        else:
            report["request_ledger_unavailable"] = "Request identity was rejected before a valid work ledger could be initialized"
        # END a CLI restart cannot silently reset spent initialization/solve work.
        if result.trajectory is not None:
            # BEGIN accepted artifacts retain their own effective native bounds.
            if selected_joint_limits is None:
                raise ValueError("Accepted curve recording requires its actual native verifier bounds")
            # END recording failure cannot silently omit the derivative evidence.
            save_curve(output/"trajectory.npz", result.trajectory,
                       classification="planner_accepted", execution_limits=execution_limits,
                       joint_limits=selected_joint_limits, planning_result=result)
        return result
    except KeyboardInterrupt:
        report.update(status="interrupted", interruption_reason="Explicit operator interrupt")
        # BEGIN recording an interruption cannot replace the interruption.
        try:
            (output/"exception.txt").write_text(traceback.format_exc())
        except Exception as error:
            report["observer_errors"].append(f"Exception record: {type(error).__name__}: {error}")
        # END already persisted per-batch CPU evidence remains authoritative.
        raise
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        # BEGIN preserve the original failure if the diagnostic filesystem fails.
        try:
            (output/"exception.txt").write_text(traceback.format_exc())
        except Exception as diagnostic_error:
            report["observer_errors"].append(
                f"Exception record: {type(diagnostic_error).__name__}: {diagnostic_error}")
        # END this write has no authority to change the planner's exception.
        raise
    finally:
        # BEGIN final telemetry cannot mask an exception or revoke a result.
        report["elapsed_seconds"] = time.monotonic()-started
        try:
            report.update(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                          peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
        except Exception as error:
            report["observer_errors"].append(f"Peak resource record: {type(error).__name__}: {error}")
        try:
            _write_diagnostic_json(output/"planning-report.json", report)
        except Exception as error:
            report["observer_errors"].append(f"Final planning report: {type(error).__name__}: {error}")
        # END a missing final report still fails the external execution gate.
# END current-planner invocation with scoped diagnostic curve recording.
