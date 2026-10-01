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
               joint_limits=None):
    """Preserve the exact exported cubic, event identities and optional FK.

    ``fk`` contains independently evaluated native FK of this same exported
    curve, in the problem reference frame. Native planner callers also supply
    their effective bounds; the shared production calculus then reports both
    the full curve and its original executable prefix. This recorder never
    changes classification based on an additional report.
    """
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
         "execution_limits": deepcopy(execution_limits),
         "finger_trajectory": None if fingers is None else fingers.describe(),
         "events": curve.events,
         "conditions": curve.conditions, "residuals": curve.residuals,
         "duration_seconds": curve.duration,
         "executable_until_seconds": curve.executable_until,
         "next_verification": curve.next_verification, **derivative_metadata},
        default=_json_value, indent=2, allow_nan=False)+"\n")


def plan_and_record(problem, scene, robot, output, *, budget, provenance, execution_limits):
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
              "optimizer_inputs": []}
    original = planning._CuroboPlanner.select_candidate
    original_export = planning._CuroboPlanner.export
    original_curve_init = planning.JointTrajectory.__post_init__
    native_trace = None
    configuration_index = 0
    # BEGIN retain an owned CPU copy before the native planner is released.
    selected_joint_limits = None
    # END no live solver reference is retained by the recording adapter.

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
            result = method(planner, *args, **kwargs)
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
        classification = ("planner_accepted" if attempt["accepted"]
                          else "rejected_diagnostic_not_executable")
        path = output/f"candidate-{index:02d}.npz"
        entry = {"path": path.name, "classification": classification,
                 "attempt": deepcopy(attempt), "validation_status": validation_status,
                 "curve_saved": False, "native_result": native_trace}
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

        def capture(curve):
            original_curve_init(curve)
            constructed.append(curve)

        try:
            with patch.object(planning.JointTrajectory, "__post_init__", new=capture):
                return original_export(planner, native_result)
        except ValueError as error:
            if len(constructed) == 1:
                try:
                    preserve_candidate(planner, constructed[0],
                        {"accepted": False, "native_success": bool(native_result.success.all()),
                         "export_error": str(error)}, validation_status="not_completed")
                except Exception as diagnostic_error:
                    report.setdefault("observer_errors", []).append(
                        f"Rejected curve record: {type(diagnostic_error).__name__}: {diagnostic_error}")
            raise

    def observed(planner, native_result):
        # BEGIN retain every native seed before production gathers a candidate.
        nonlocal native_trace
        path = output/f"native-result-{len(report['native_results']):02d}.npz"
        native_trace = path.name
        entry = {"path": path.name, "saved": False,
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
        selected, curve, attempt = original(planner, native_result)
        preserve_candidate(planner, curve, attempt, validation_status="completed")
        return selected, curve, attempt

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(planning._CuroboPlanner, "select_candidate", new=observed))
            stack.enter_context(patch.object(planning._CuroboPlanner, "export", new=observed_export))
            for name in ("_initialize", "reference_seeds", "optimize"):
                method = getattr(planning._CuroboPlanner, name)
                stack.enter_context(patch.object(planning._CuroboPlanner, name,
                                                 new=stage_observer(method, name)))
            result = planning.plan_trajectory(problem, scene, robot, budget=budget)
        report.update(status="completed", success=result.success,
                      diagnostics=result.diagnostics)
        if result.trajectory is not None:
            # BEGIN accepted artifacts retain their own effective native bounds.
            if selected_joint_limits is None:
                raise ValueError("Accepted curve recording requires its actual native verifier bounds")
            # END recording failure cannot silently omit the derivative evidence.
            save_curve(output/"trajectory.npz", result.trajectory,
                       classification="planner_accepted", execution_limits=execution_limits,
                       joint_limits=selected_joint_limits)
        return result
    except KeyboardInterrupt:
        report.update(status="interrupted", interruption_reason="Explicit operator interrupt")
        (output/"exception.txt").write_text(traceback.format_exc())
        raise
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (output/"exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_torch_reserved_bytes=torch.cuda.max_memory_reserved())
        (output/"planning-report.json").write_text(json.dumps(
            report, default=_json_value, indent=2, allow_nan=False)+"\n")
# END current-planner invocation with scoped diagnostic curve recording.
