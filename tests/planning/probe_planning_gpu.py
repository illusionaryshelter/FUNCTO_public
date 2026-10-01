"""Bounded V2 integration probe; the root agent runs GPU work serially.

This is a synthetic open-space process test, not a perception/scene or robot
execution test. A bent necessary suffix must be generated through the custom
constraint while native terminal pose tracking is disabled. Failed candidates
remain diagnostic artifacts: neither a tolerance sensitivity table nor a
smooth-looking curve can promote them to executable plans. No packages or
environment settings are installed or changed by this script.
"""

# BEGIN bounded full-process GPU integration probe.
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import traceback
from unittest.mock import patch

import numpy as np


# BEGIN bounded, read-only diagnostics of the exact saved candidate cubic.
def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported diagnostic value: {type(value).__name__}")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, default=_json, indent=2, allow_nan=False) + "\n")


def _distribution(values):
    """Unweighted query-sample statistics, with no invented pass threshold."""
    values = np.asarray(values, float)
    result = {"sample_count": len(values), "all_finite": bool(np.isfinite(values).all())}
    if not len(values) or not result["all_finite"]:
        return result
    result.update(mean=values.mean(0), standard_deviation=values.std(0),
                  rms=np.sqrt(np.mean(values**2, axis=0)),
                  absolute_p95=np.quantile(np.abs(values), .95, axis=0),
                  absolute_p99=np.quantile(np.abs(values), .99, axis=0),
                  maximum_absolute=np.max(np.abs(values), axis=0))
    return result


def _diagnose_curve(planner, problem, path, output):
    """Inspect one frozen saved cubic after the production solve has ended.

    FK and the original native STATE/self-collision checks run in chunks of
    256. Every saved knot, midpoint and original clock anchor is retained;
    exceeding 4096 samples rejects this diagnostic instead of thinning it.
    The source candidate, its acceptance and its timing are never changed.
    """
    import torch
    from manipulation.planning import JointTrajectory, _ProgressMap
    from manipulation.planning_problem import _pose_error

    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as saved:
        curve = JointTrajectory(tuple(saved["joint_names"].tolist()), saved["times"].copy(),
            saved["position"].copy(), saved["velocity"].copy(), saved["progress_u"].copy(),
            saved["progress_s"].copy(), tuple(metadata["events"]), tuple(metadata["conditions"]),
            metadata["residuals"])
    if curve.joint_names != planner.joint_names or len(curve.times) > 4096:
        raise ValueError("Diagnostic candidate must retain the actual bounded native joint order")
    # This probe has fixed source references and no endpoint families/events.
    # Chunking those simple references does not remove a required FK anchor.
    if (problem.events or problem.conditions or any(phase.endpoint_family is not None
            or phase.frame_anchor is not None or phase.reference.motion_model is not None
            for phase in problem.phases)):
        raise ValueError("The open-scene diagnostic only supports this explicit fixed-reference experiment")
    progress_map = _ProgressMap(curve.progress_u.copy(), curve.progress_s.copy())
    anchor_pairs = progress_map.anchor_parameters(problem.anchors)
    query = np.unique(np.r_[curve.times, (curve.times[:-1] + curve.times[1:]) / 2,
                            anchor_pairs[:, 0] * curve.duration])
    if len(query) > 4096:
        raise ValueError("All original knots, midpoints and source anchors exceed 4096 diagnostic samples")
    q, dq, progress = curve.sample(query)
    ddq, jerk = curve._curve(query, 2), curve._curve(query, 3)
    arrays = {"times": query, "progress": progress, "q": q, "dq": dq, "ddq": ddq, "jerk": jerk,
              "joint_names": np.asarray(curve.joint_names), "anchor_parameters": anchor_pairs,
              "original_knot_times": curve.times, "original_knot_q": curve.position,
              "original_knot_dq": curve.velocity, "progress_u": curve.progress_u, "progress_s": curve.progress_s}
    chunks = {}

    def retain(key, value):
        # Copy before another native query can reuse its output buffers.
        chunks.setdefault(key, []).append(value[0].detach().cpu().numpy().copy())

    with torch.no_grad():
        for start in range(0, len(query), 256):
            interval = slice(start, min(start + 256, len(query)))
            fields = {key: planner.device_cfg.to_device(value[None, interval]).contiguous()
                      for key, value in (("position", q), ("velocity", dq), ("acceleration", ddq), ("jerk", jerk))}
            joint_state = planner.JointState(**fields,
                dt=planner.device_cfg.to_device([float(curve.times[1] - curve.times[0])]),
                joint_names=list(curve.joint_names))
            state = planner._verification_fk.compute_kinematics(joint_state)
            tool_index = state.tool_poses.tool_frames.index(problem.tool_link)
            position, quaternion = problem.transform_tool_pose(
                state.tool_poses.position[:, :, tool_index], state.tool_poses.quaternion[:, :, tool_index])
            s = planner.device_cfg.to_device(progress[None, interval])
            retain("tcp_position", position)
            retain("tcp_quaternion_wxyz", quaternion)
            for key, value in fields.items():
                retain(f"native_input_{key}", value)
            components = problem.residuals(position, quaternion, s, fields["position"], curve.joint_names,
                                           joint_velocity=fields["velocity"])
            components["scene"] = planner._verification_scene.evaluate(
                problem.transform_robot_spheres(state.robot_spheres), position, quaternion, s,
                fields["position"], curve.joint_names)
            native = planner._verification_native.compute_costs(
                planner._RobotState(joint_state=joint_state, cuda_robot_model_state=state))
            components.update({f"native:{name}": value for name, value in zip(native.names, native.values)})
            for name, value in components.items():
                retain(f"residual:{name}", value)
            for index, phase in enumerate(problem.phases):
                target_p, target_q = problem._phase_reference(index, position, quaternion, s)
                error = _pose_error(position, quaternion, target_p, target_q)
                inside = (s >= index - 1e-7) & (s <= index + 1 + 1e-7)
                active = inside if phase.kind == "pose" else (s - index - 1).abs() <= 1e-7
                if phase.kind == "free" and not phase.require_end:
                    active = torch.zeros_like(active)
                prefix = f"phase_{index}"
                retain(f"{prefix}_reference_position", target_p)
                retain(f"{prefix}_reference_quaternion_wxyz", target_q)
                retain(f"{prefix}_error_target_local", error)
                retain(f"{prefix}_inside", inside)
                retain(f"{prefix}_constraint_active", active)
    arrays.update({key: np.concatenate(values, axis=0) for key, values in chunks.items()})

    # Inspect continuity from the actual polynomial coefficients. Finite
    # differences of densely sampled q would introduce a new noise source.
    coefficients, widths = curve._curve.c, np.diff(curve.times)[:, None]
    acceleration_left = 6 * coefficients[0] * widths + 2 * coefficients[1]
    velocity_left = 3 * coefficients[0] * widths**2 + 2 * coefficients[1] * widths + coefficients[2]
    arrays.update(breakpoint_times=curve.times[1:-1],
        velocity_jumps=coefficients[2, 1:] - velocity_left[:-1],
        acceleration_jumps=2 * coefficients[1, 1:] - acceleration_left[:-1],
        jerk_jumps=6 * np.diff(coefficients[0], axis=0))
    report = {"source_candidate": str(path), "source_classification": metadata["classification"],
              "source_validation_status": metadata["validation_status"], "reclassified": False,
              "physical_execution_authorized": False, "sample_count": len(query), "fk_chunk_size": 256,
              "sampling": "all original knots, their midpoints and every original source anchor; no resampling of the curve",
              "statistics": "unweighted query-sample statistics, not fractions of elapsed time",
              "collision_scope": "same native cuRobo robot sphere model at sampled configurations; no continuous/scene/physics proof",
              "original_numerical_tolerances": asdict(problem.numerical),
              "derivatives": {key: _distribution(arrays[key]) for key in
                              ("dq", "ddq", "jerk", "velocity_jumps", "acceleration_jumps", "jerk_jumps")},
              "residuals": {}, "joint_limits": {}, "phases": []}
    for key, value in arrays.items():
        if key.startswith("residual:"):
            flat = value.reshape(len(query), -1)
            finite = np.isfinite(flat).all()
            report["residuals"][key[9:]] = {
                "all_finite": bool(finite), "maximum": float(flat.max()) if finite else None,
                "positive_sample_count": int(np.any(flat > 0, axis=1).sum()),
                "nonfinite_sample_count": int((~np.isfinite(flat).all(axis=1)).sum()), "array_key": key}
    cspace = planner._verification_native.get_cost("cspace")
    limits = cspace.config.joint_limits
    columns = [limits.joint_names.index(name) for name in curve.joint_names]
    for kind in ("position", "velocity", "acceleration", "jerk"):
        bounds = getattr(limits, kind)[:, columns].detach().cpu().numpy().copy()
        values = arrays[f"native_input_{kind}"]
        excess = np.maximum(np.maximum(bounds[0] - values, values - bounds[1]), 0.)
        arrays[f"native_{kind}_limits"] = bounds
        arrays[f"native_{kind}_limit_excess"] = excess
        report["joint_limits"][kind] = {"lower": bounds[0], "upper": bounds[1],
            "maximum_excess_per_joint": excess.max(0), "positive_sample_count_per_joint": (excess > 0).sum(0)}
    for index, phase in enumerate(problem.phases):
        prefix = f"phase_{index}"
        error = arrays[f"{prefix}_error_target_local"]
        active = arrays[f"{prefix}_constraint_active"]
        inside = arrays[f"{prefix}_inside"]
        constrained = error[active]
        row = {"name": phase.name, "kind": phase.kind, "require_end": phase.require_end,
               "pose_bounds": phase.pose_bounds, "reference_source": phase.reference.source,
               "constraint_active_samples": int(active.sum()),
               "constrained_signed_xyz_rotvec": _distribution(constrained),
               "whole_phase_reference_comparison": _distribution(error[inside]),
               "free_interior_constrained": False if phase.kind == "free" else None,
               "position_slack_sensitivity": [],
               "sensitivity_scope": "same frozen curve and original active mask; hypothetical per-axis numerical slack only; no acceptance decision"}
        if len(constrained) and np.isfinite(constrained).all():
            position_error = np.linalg.norm(constrained[:, :3], axis=1)
            peak = int(np.flatnonzero(active)[position_error.argmax()])
            corners = index + phase.reference.progress[1:-1]
            row.update(position_error_norm_m=_distribution(position_error),
                       peak_position_error={"time_s": float(query[peak]), "progress": float(progress[peak]),
                           "euclidean_m": float(position_error.max()), "signed_local_xyz_m": error[peak, :3],
                           "nearest_internal_corner_progress_distance": (float(np.min(np.abs(corners-progress[peak])))
                                                                        if len(corners) else None)})
            rotation_excess = np.maximum(np.maximum(
                phase.pose_bounds[3:, 0] - constrained[:, 3:] - problem.numerical.rotation_rad,
                constrained[:, 3:] - phase.pose_bounds[3:, 1] - problem.numerical.rotation_rad), 0.)
            row["unchanged_rotation_maximum_excess_rad"] = rotation_excess.max(0)
            for millimetres in (.1, .5, 1., 2., 5.):
                slack = millimetres / 1000.
                excess = np.maximum(np.maximum(phase.pose_bounds[:3, 0] - constrained[:, :3] - slack,
                                               constrained[:, :3] - phase.pose_bounds[:3, 1] - slack), 0.)
                row["position_slack_sensitivity"].append({"per_component_slack_mm": millimetres,
                    "outside_fraction_per_xyz_component": (excess > 0).mean(0),
                    "outside_fraction_any_xyz_component": float(np.any(excess > 0, axis=1).mean()),
                    "maximum_excess_per_xyz_component_m": excess.max(0),
                    "maximum_excess_m": float(excess.max())})
        report["phases"].append(row)
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    report["arrays"] = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()}
    _write_json(output.with_suffix(".json"), report)
    return {"source_candidate": path.name, "diagnostic": str(output.with_suffix(".json")),
            "arrays": str(output.with_suffix(".npz")), "classification": metadata["classification"],
            "reclassified": False, "sample_count": len(query)}
# END bounded, read-only diagnostics of the exact saved candidate cubic.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSON report path; detailed artifacts use its same-stem directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix != ".json" or output.exists():
        raise ValueError("Provide a new .json report path; previous evidence is never overwritten")
    artifacts = output.with_suffix("")
    artifacts.mkdir(parents=True, exist_ok=False)
    # BEGIN resolve this checkout without a workspace or date-directory dependency.
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project))
    # END project-local imports.

    import torch
    from scipy.spatial.transform import Rotation
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    from manipulation import planning
    from manipulation.planning import PlanningBudget
    from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath
    from recorded_task_planning import plan_and_record

    class DeclaredOpenScene:
        """An explicit synthetic open world, with no mutable query buffers."""
        reference_frame = "synthetic_robot_base"
        required_progress_anchors = np.array([], dtype=float)
        required_robot_links = ()

        def __init__(self, sphere_count):
            self.robot_sphere_materials = (None,)*sphere_count

        def clone_for_rollout(self):
            return DeclaredOpenScene(len(self.robot_sphere_materials))

        def validate_problem(self, problem):
            if problem.events or problem.conditions or any(phase.metadata for phase in problem.phases):
                raise ValueError("This declared open experiment cannot validate interaction evidence or gripper commands")

        def estimated_workspace_bytes(self, batch, horizon):
            return 0

        def validate_robot_geometry(self, current_spheres_output, link_poses_output):
            """This explicit empty fixture contains no occupied robot evidence."""

        def optimizer_preferences(self, progress, joint_position, joint_names):
            return progress*0.

        def describe(self):
            return {"scope": "explicit synthetic open scene", "production_environment_validation": False}

        def with_fixed_endpoints(self, position, quaternion, progress):
            return self.clone_for_rollout()

        def project_residual(self, progress, joint_position, joint_names):
            return joint_position[..., :0]

        def evaluate(self, robot_spheres, position, quaternion, progress,
                     joint_position=None, joint_names=None):
            return torch.zeros_like(progress)

        # BEGIN expose the same explicit empty scene to entrance diagnostics.
        def components(self, robot_spheres, position, quaternion, progress,
                       joint_position=None, joint_names=None):
            """No scene constraints exist in this declared synthetic empty world."""
            return {}
        # END expose the same explicit empty scene to entrance diagnostics.

    # BEGIN apply the same explicit local resource cap as the other robot checks.
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("This explicit integration check requires CUDA")
    torch.cuda.set_per_process_memory_fraction(
        2*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    # END bounded local GPU allocation.
    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
    initial = kinematics.default_joint_state
    state = kinematics.compute_kinematics(initial)
    position = state.tool_poses.position[0, 0, 0].detach().cpu().numpy()
    quaternion = state.tool_poses.quaternion[0, 0, 0].detach().cpu().numpy()
    start_pose = np.eye(4)
    start_pose[:3, 3] = position
    start_pose[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    arrive = start_pose.copy()
    arrive[0, 3] += .015
    middle = arrive.copy()
    middle[1, 3] += .018
    finish = arrive.copy()
    finish[0, 3] += .018
    free = MotionPhase("synthetic_approach", PosePath(np.array([0., 1.]),
                       np.stack((start_pose, arrive)), "synthetic calibrated reach"), kind="free")
    suffix = MotionPhase("necessary_bent_suffix", PosePath(np.array([0., .5, 1.]),
                         np.stack((arrive, middle, finish)), "synthetic necessary intermediate movement"))
    problem = PlanningProblem(tuple(kinematics.joint_names),
                              initial.position.detach().cpu().numpy().reshape(-1),
                              np.zeros(len(kinematics.joint_names)), kinematics.tool_frames[0],
                              "synthetic_robot_base", np.eye(4), np.eye(4), (free, suffix))
    # BEGIN preserve original planning inputs and inspect failures without changing them.
    budget = PlanningBudget(search_seconds=0.)
    scene = DeclaredOpenScene(kinematics.total_spheres)
    source_case = {"scope": "synthetic Franka bent-path experiment; no recorded task or execution",
        "probe_source": str(Path(__file__).resolve()), "robot_configuration": "franka.yml",
        "joint_names": problem.joint_names, "start_q": problem.start_q, "start_dq": problem.start_dq,
        "reference_frame": problem.reference_frame, "base_pose": problem.base_pose,
        "tool_link": problem.tool_link, "tool_transform": problem.tool_transform,
        "numerical": asdict(problem.numerical), "budget": asdict(budget),
        "events": [], "conditions": [], "scene": scene.describe(),
        "phases": [{"name": phase.name, "kind": phase.kind, "require_end": phase.require_end,
            "pose_bounds": phase.pose_bounds, "reference_progress": phase.reference.progress,
            "reference_poses": phase.reference.poses, "reference_source": phase.reference.source,
            "reference_attachment": phase.reference.attachment} for phase in problem.phases]}
    _write_json(artifacts / "source-case.json", source_case)
    (artifacts / "probe-source.py").write_text(Path(__file__).read_text())
    report = {"status": "planning", "success": False,
              "scope": "synthetic open-scene full-process optimization; no robot execution",
              "execution": "native eager", "source_case": str(artifacts / "source-case.json"),
              "execution_limits": None, "candidate_diagnostics": [], "diagnostic_errors": []}
    live = {}
    original_initialize = planning._CuroboPlanner._initialize

    def remember_planner(planner, *arguments, **keywords):
        value = original_initialize(planner, *arguments, **keywords)
        live["planner"] = planner
        return value

    try:
        # The hook only retains a reference; the existing recorder still owns
        # native/candidate/config preservation and restores every scoped hook.
        with patch.object(planning._CuroboPlanner, "_initialize", new=remember_planner):
            result = plan_and_record(problem, scene, "franka.yml", artifacts / "planning", budget=budget,
                provenance={"source_case": str(artifacts / "source-case.json"), "scope": source_case["scope"]},
                execution_limits=None)
        report.update(status="completed", success=result.success, diagnostics=result.diagnostics)
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (artifacts / "exception.txt").write_text(traceback.format_exc())
    finally:
        planning_report = artifacts / "planning" / "planning-report.json"
        if "planner" in live and planning_report.is_file():
            recorded = json.loads(planning_report.read_text())
            candidates = recorded["candidates"]
            if len(candidates) > budget.maximum_trajectory_solves * budget.trajectory_seeds:
                report["diagnostic_errors"].append("Saved candidate count exceeds the original finite solve budget")
            else:
                for index, candidate in enumerate(candidates):
                    if not candidate["curve_saved"]:
                        continue
                    try:
                        entry = _diagnose_curve(live["planner"], problem,
                            artifacts / "planning" / candidate["path"], artifacts / f"curve-diagnostic-{index:02d}")
                        entry["native_result"] = candidate["native_result"]
                        report["candidate_diagnostics"].append(entry)
                    except Exception as error:
                        report["diagnostic_errors"].append(f"{candidate['path']}: {type(error).__name__}: {error}")
                        (artifacts / f"diagnostic-exception-{index:02d}.txt").write_text(traceback.format_exc())
        report.update(peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _write_json(output, report)
        print(json.dumps({"status": report["status"], "success": report["success"], "report": str(output),
                          "candidate_diagnostics": len(report["candidate_diagnostics"]),
                          "diagnostic_errors": report["diagnostic_errors"]}, allow_nan=False), flush=True)
    # END preserve original planning inputs and inspect failures without changing them.
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# END bounded full-process GPU integration probe.
