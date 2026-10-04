"""Compare equal endpoint problems with and without a required middle motion.

This is an explicitly empty synthetic world with declared fixed finger geometry.
Both requests use the same source robot, starting state, endpoint and numerical
budget. Only the independently required intermediate pose changes. This probes
generation-time process behavior; it is not perception or contact validation.
"""

# BEGIN paired real-robot process-generation experiment.
import argparse
from copy import deepcopy
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
from probe_process_planner import no_obstacle_scene


def run(output, knots, *, fixture_path):
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = {"scope": "paired synthetic equal-endpoint robot processes",
              "production_environment_validation": False, "status": "running", "cases": {}}
    try:
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from manipulation.planning_problem import (PlanningProblem, MotionPhase, PosePath,
            NumericalTolerance, PoseEventRequirement)
        from manipulation.planning import PlanningBudget, plan_trajectory
        from probe_robot_fixture import load_probe_fixture, fixed_probe_hand

        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        # BEGIN paired cases retain the same complete source model and posture.
        fixture, robot, validator = load_probe_fixture(fixture_path, reference_frame="robot_base")
        full_positions = np.asarray(robot["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"])
        device = DeviceCfg("cuda:0")
        fk = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
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
        excursion = excursion @ validator.T_frame_tcp
        # END source frame transforms are shared by both compared requirements.
        constant = np.repeat(excursion[:1], 3, axis=0)
        budget = PlanningBudget(ik_seeds=4, trajectory_seeds=1, ik_iterations=80, max_robot_spheres=fk.total_spheres,
                                knots=knots, interpolation_buffer_size=1024,
                                trajectory_iterations=150, search_seconds=0.)
        report["budget"] = asdict(budget)
        report["required_midpoint_displacement_m"] = float(np.linalg.norm(p[1]-p[0]))
        report["identical_initial_and_final_pose"] = bool(np.array_equal(constant[[0, 2]], excursion[[0, 2]]))
        cases = {}
        for name, poses in (("hold", constant), ("excursion", excursion)):
            # BEGIN midpoint and endpoint are declared task goals, not source locks.
            problem = PlanningProblem(tuple(fk.joint_names), q0.cpu().numpy(), np.zeros(len(q0)),
                fixture["tool_link"], "robot_base", np.eye(4), validator.T_frame_tcp,
                (MotionPhase("source_reference", PosePath(np.array([0., .5, 1.]), poses,
                                                          "paired synthetic reference")),),
                numerical=NumericalTolerance(position_m=.001, rotation_rad=.005),
                pose_events=tuple(PoseEventRequirement(progress, np.eye(4), target, .001, .005,
                    "Independent paired experiment midpoint/return goal")
                    for progress, target in ((.5, poses[1]), (1., poses[-1]))),
                source_timestamps=np.array([0., 1., 2.]), source_progress=np.array([0., .5, 1.]))
            # END ordinary poses between these real goals remain unconstrained.
            # BEGIN each case binds its own hand clock over the shared source geometry.
            hand, owners, hand_report = fixed_probe_hand(fixture, robot, problem, device=device,
                                                        full_current_positions=full_positions)
            scene = no_obstacle_scene(fk.total_spheres, hand=hand, installation_owners=owners,
                                      geometry_source=fixture["robot_geometry_source"])
            result = plan_trajectory(problem, scene, robot, robot_validator=validator, budget=budget)
            # END no fixture or nominal finger evidence is silently omitted.
            row = {"success": result.success, "diagnostics": result.diagnostics,
                   "fixed_hand_hypothesis": hand_report}
            if result.trajectory is not None:
                curve = result.trajectory
                # BEGIN query the saved clock at the true goal progress values.
                query = np.interp([0., .5, 1.], curve.progress_s, curve.progress_u)*curve.duration
                query[0] = 0.
                # END half the duration is not assumed to be half the progress.
                # BEGIN preserve the goal clock in the q/dq/ddq sample layout.
                q, dq, _, progress = curve.sample(query)
                # END only the requested functional times define these goals.
                sample = torch.as_tensor(q[None], dtype=torch.float32, device="cuda:0").contiguous()
                state = fk.compute_kinematics(JointState.from_position(sample, joint_names=fk.joint_names))
                position = state.tool_poses.position[:, :, 0].detach().clone()
                quaternion = state.tool_poses.quaternion[:, :, 0].detach().clone()
                # BEGIN paired diagnostics use the same source TCP as the goals.
                position, quaternion = problem.transform_tool_pose(position, quaternion)
                # END an offset tool frame cannot silently become the graded TCP.
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
            # excursion at its independently required midpoint. This checks
            # the material-frame goal directly, apart from native metrics.
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
    parser.add_argument("--fixture", type=Path, required=True,
                        help="Verified native robot manifest supplying source solids and pair policy")
    parser.add_argument("--knots", type=int, default=24)
    args = parser.parse_args()
    run(args.output, args.knots, fixture_path=args.fixture)
# END paired real-robot process-generation experiment.
