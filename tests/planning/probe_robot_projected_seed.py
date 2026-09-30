"""Exercise native robot-model projected OMPL search and full native TrajOpt.

The search is invoked explicitly to test the production integration.  This
does not claim that the initial optimizer failed or that an observed physical
world has been validated.  Geometry is the declared empty research scene;
native robot self-collision and every process constraint remain active.
"""

# BEGIN bounded actual-robot projected-seed integration experiment.
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

# BEGIN resolve imports from this FUNCTO checkout, independent of its directory name.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# END project-local imports.
from probe_process_planner import NoObstacleExperiment


def run(output, search_seconds):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"scope": "native Franka-model FK, projected OMPL, complete native trajectory optimization",
              "search_reason": "explicit integration probe, regardless of baseline optimizer success",
              "production_environment_validation": False, "status": "running"}
    planner = None
    try:
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import JointState
        from manipulation.planning import PlanningBudget, _CuroboPlanner
        from manipulation.planning_problem import PlanningProblem, MotionPhase, PosePath, NumericalTolerance

        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        fk = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
        initial = fk.default_joint_position.reshape(-1).clone()
        finish = initial.clone()
        finish[0] += .04
        finish[1] -= .015
        state = fk.compute_kinematics(JointState.from_position(
            torch.stack((initial, finish))[None].contiguous(), joint_names=fk.joint_names))
        position = state.tool_poses.position[0, :, 0].detach().cpu().numpy()
        quaternion = state.tool_poses.quaternion[0, :, 0].detach().cpu().numpy()
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[:, :3, 3] = position
        poses[:, :3, :3] = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
        problem = PlanningProblem(tuple(fk.joint_names), initial.cpu().numpy(), np.zeros(len(initial)),
                                  "panda_hand", "robot_base", np.eye(4), np.eye(4),
                                  (MotionPhase("required_local_motion", PosePath(np.array([0., 1.]), poses,
                                                "explicit small native-FK endpoint process")),),
                                  numerical=NumericalTolerance(position_m=.001, rotation_rad=.005))
        budget = PlanningBudget(ik_seeds=4, trajectory_seeds=1, ik_iterations=80,
                                knots=12, interpolation_buffer_size=768,
                                trajectory_iterations=150, search_seconds=search_seconds,
                                maximum_trajectory_solves=3)
        report["budget"] = asdict(budget)
        planner = _CuroboPlanner(problem, NoObstacleExperiment(fk.total_spheres), "franka.yml", budget, "cuda:0")
        report["robot_metadata"] = planner.robot_metadata
        seeds, successes = planner.reference_seeds()
        report["reference_ik_success_counts"] = successes
        baseline = planner.optimize(seeds)
        candidate, points = planner.export(baseline)
        report["baseline"] = {"native_success": bool(baseline.success.all().item()),
                              "validation_points": points, "residuals": candidate.residuals}

        search_started = time.monotonic()
        seed, progress_map, search_report = planner.ompl_seed(seeds, candidate)
        report["ompl"] = search_report
        report["search_elapsed_seconds"] = time.monotonic()-search_started
        report["success"] = False
        if seed is None:
            report["failure"] = "projected_search_did_not_return_a_complete_seed"
        else:
            projected = any(stage.get("space") == "projected" and stage.get("equalities") == 6
                            and stage.get("samples", 0) >= 2
                            for branch in search_report["branches"] for stage in branch)
            if not projected:
                raise ValueError("This probe requires a real six-equality native projected-space path")
            planner.replace_schedule(progress_map)
            report["trajectory_attempts"] = []
            for iteration in range(budget.maximum_trajectory_solves):
                result = planner.optimize(seed)
                trajectory, points = planner.export(result)
                native_success = bool(result.success.all().item())
                valid = all(np.isfinite(value) and value <= 0 for value in trajectory.residuals.values())
                report["trajectory_attempts"].append({
                    "native_success": native_success, "native_metrics": planner.native_diagnostics(result),
                    "validation_points": points, "residuals": trajectory.residuals,
                    "measurements": planner.last_validation["measurements"]})
                if native_success and valid:
                    report.update(success=True, duration=trajectory.duration)
                    np.savez(output/"trajectory.npz", times=trajectory.times, position=trajectory.position,
                             velocity=trajectory.velocity, progress_u=trajectory.progress_u, progress_s=trajectory.progress_s)
                    break
                parameters = planner.refinement_samples()
                if parameters is None or iteration+1 == budget.maximum_trajectory_solves:
                    break
                seed = result.solution.detach().clone()
                planner.replace_schedule(planner.progress_map, parameters)
            if not report["success"]:
                report["failure"] = "complete_native_optimization_or_final_curve_rejected"
        report.update(status="completed", torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (output/"exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        if planner is not None:
            planner.close()
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-seconds", type=float, default=15.)
    args = parser.parse_args()
    run(args.output, args.search_seconds)
# END bounded actual-robot projected-seed integration experiment.
