"""Run the current planner and retain inspectable evidence for a recorded task.

This test helper accepts already constructed production inputs. It does not
choose a dataset, manufacture an observation, weaken a constraint, or execute
a rejected curve. Failed native candidates remain explicitly diagnostic. The
caller owns the observed scene, robot binding and separate physics evaluator.
"""

# BEGIN current-planner invocation with scoped diagnostic curve recording.
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict
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


def save_curve(path, curve, *, classification, fk=None):
    """Preserve the exact exported cubic, event identities and optional FK.

    ``fk`` contains independently evaluated native FK of this same exported
    curve, in the problem reference frame. No reference path is substituted
    when the planner fails or when FK has not been evaluated.
    """
    arrays = dict(joint_names=np.asarray(curve.joint_names), times=curve.times,
                  position=curve.position, velocity=curve.velocity,
                  progress_u=curve.progress_u, progress_s=curve.progress_s,
                  classification=np.asarray(classification))
    if fk is not None:
        arrays.update(fk)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(
        {"classification": classification, "events": curve.events,
         "conditions": curve.conditions, "residuals": curve.residuals,
         "duration_seconds": curve.duration,
         "executable_until_seconds": curve.executable_until,
         "next_verification": curve.next_verification},
        default=_json_value, indent=2, allow_nan=False)+"\n")


def plan_and_record(problem, scene, robot, output, *, budget, provenance):
    """Call ``plan_trajectory`` once and preserve its measured outcome.

    The scoped observer copies each selected continuation *after* the normal
    candidate checks. It neither changes selection nor lets a diagnostic
    candidate escape as ``PlanningResult.trajectory``. All CUDA work remains
    within the caller's explicit single-process resource budget.
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
              "skill_ids": sorted({event.skill_id for event in problem.events}),
              "phase_count": problem.phase_count,
              "required_source_anchors": len(problem.anchors),
              "candidates": []}
    original = planning._CuroboPlanner.select_candidate

    def stage_observer(method, name):
        """Expose coarse native stages without polling or changing solver work."""
        def measured(planner, *args, **kwargs):
            stage = {"stage": name, "state": "started",
                     "elapsed_seconds": time.monotonic()-started}
            (output/"progress.json").write_text(json.dumps(stage)+"\n")
            print(json.dumps(stage), flush=True)
            result = method(planner, *args, **kwargs)
            stage.update(state="completed", elapsed_seconds=time.monotonic()-started)
            (output/"progress.json").write_text(json.dumps(stage)+"\n")
            print(json.dumps(stage), flush=True)
            return result
        return measured

    def observed(planner, native_result):
        selected, curve, attempt = original(planner, native_result)
        index = len(report["candidates"])
        classification = ("planner_accepted" if attempt["accepted"]
                          else "rejected_diagnostic_not_executable")
        path = output/f"candidate-{index:02d}.npz"
        entry = {"path": path.name, "classification": classification,
                 "attempt": deepcopy(attempt), "curve_saved": False}
        try:
            # Save CPU evidence before optional display FK. A visualization
            # failure must not change the production candidate or its checks.
            save_curve(path, curve, classification=classification)
            entry["curve_saved"] = True
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
            save_curve(path, curve, classification=classification, fk=sampled)
            entry["display_fk_saved"] = True
        except Exception as error:
            entry["diagnostic_error"] = f"{type(error).__name__}: {error}"
            entry["diagnostic_traceback"] = traceback.format_exc()
        report["candidates"].append(entry)
        return selected, curve, attempt

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(planning._CuroboPlanner, "select_candidate", new=observed))
            for name in ("_initialize", "reference_seeds", "optimize"):
                method = getattr(planning._CuroboPlanner, name)
                stack.enter_context(patch.object(planning._CuroboPlanner, name,
                                                 new=stage_observer(method, name)))
            result = planning.plan_trajectory(problem, scene, robot, budget=budget)
        report.update(status="completed", success=result.success,
                      diagnostics=result.diagnostics)
        if result.trajectory is not None:
            save_curve(output/"trajectory.npz", result.trajectory,
                       classification="planner_accepted")
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
