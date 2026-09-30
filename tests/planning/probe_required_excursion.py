"""Compare equal endpoint problems with and without a required middle motion.

This is an explicitly empty synthetic world and a rigid Franka tool. Both
requests use the same actual robot, starting state, endpoint and numerical
budget. Only the required intermediate reference pose changes. This probes
generation-time process behavior; it is not perception or contact validation.
"""

# BEGIN paired real-robot process-generation experiment.
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


def run(output, knots):
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {"scope": "paired synthetic equal-endpoint robot processes",
              "production_environment_validation": False, "status": "running", "cases": {}}
    try:
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import JointState
        from manipulation.planning_problem import PlanningProblem, MotionPhase, PosePath, NumericalTolerance
        from manipulation.planning import PlanningBudget, plan_trajectory

        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        fk = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
        q0 = fk.default_joint_position.reshape(-1).clone()
        qm = q0.clone()
        qm[0] += .12
        qm[1] -= .02
        state = fk.compute_kinematics(JointState.from_position(
            torch.stack((q0, qm, q0))[None].contiguous(), joint_names=fk.joint_names))
        p = state.tool_poses.position[0, :, 0].detach().cpu().numpy()
        r = state.tool_poses.quaternion[0, :, 0].detach().cpu().numpy()
        excursion = np.repeat(np.eye(4)[None], 3, axis=0)
        excursion[:, :3, 3] = p
        excursion[:, :3, :3] = Rotation.from_quat(r[:, [1, 2, 3, 0]]).as_matrix()
        constant = np.repeat(excursion[:1], 3, axis=0)
        budget = PlanningBudget(ik_seeds=4, trajectory_seeds=1, ik_iterations=80,
                                knots=knots, interpolation_buffer_size=1024,
                                trajectory_iterations=150, search_seconds=0.)
        report["budget"] = asdict(budget)
        report["required_midpoint_displacement_m"] = float(np.linalg.norm(p[1]-p[0]))
        report["identical_initial_and_final_pose"] = bool(np.array_equal(constant[[0, 2]], excursion[[0, 2]]))
        cases = {}
        for name, poses in (("hold", constant), ("excursion", excursion)):
            problem = PlanningProblem(tuple(fk.joint_names), q0.cpu().numpy(), np.zeros(len(q0)),
                "panda_hand", "robot_base", np.eye(4), np.eye(4),
                (MotionPhase("required_motion", PosePath(np.array([0., .5, 1.]), poses,
                                                         "explicit paired synthetic target")),),
                numerical=NumericalTolerance(position_m=.001, rotation_rad=.005))
            result = plan_trajectory(problem, NoObstacleExperiment(fk.total_spheres), "franka.yml", budget=budget)
            row = {"success": result.success, "diagnostics": result.diagnostics}
            if result.trajectory is not None:
                curve = result.trajectory
                query = np.array([0., .5, 1.])*curve.duration
                q, dq, progress = curve.sample(query)
                sample = torch.as_tensor(q[None], dtype=torch.float32, device="cuda:0").contiguous()
                state = fk.compute_kinematics(JointState.from_position(sample, joint_names=fk.joint_names))
                position = state.tool_poses.position[:, :, 0].detach().clone()
                quaternion = state.tool_poses.quaternion[:, :, 0].detach().clone()
                row["midpoint_displacement_m"] = float(torch.linalg.vector_norm(position[0, 1]-position[0, 0]).cpu())
                row["endpoint_displacement_m"] = float(torch.linalg.vector_norm(position[0, 2]-position[0, 0]).cpu())
                row["duration"] = curve.duration
                np.savez(output/f"{name}.npz", times=curve.times, position=curve.position, velocity=curve.velocity,
                         progress_u=curve.progress_u, progress_s=curve.progress_s)
                cases[name] = (problem, position, quaternion, torch.as_tensor(progress[None], dtype=torch.float32, device="cuda:0"), sample)
            report["cases"][name] = row
        if len(cases) == 2:
            loop_problem = cases["excursion"][0]
            _, hold_p, hold_r, hold_s, hold_q = cases["hold"]
            # The hold shares both endpoint poses, but fails the actual
            # excursion at its explicit source anchor. This is independent
            # of the native disabled ToolPoseCost convergence fields.
            crossed = loop_problem.evaluate(hold_p, hold_r, hold_s, hold_q, tuple(fk.joint_names))
            report["hold_on_excursion_violation"] = crossed.detach().cpu().numpy().tolist()
            report["paired_behavior_passed"] = bool(crossed[0, 1] > .01)
        report.update(status="completed", success=(len(cases) == 2 and report.get("paired_behavior_passed", False)),
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (output/"exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--knots", type=int, default=24)
    args = parser.parse_args()
    run(args.output, args.knots)
# END paired real-robot process-generation experiment.
