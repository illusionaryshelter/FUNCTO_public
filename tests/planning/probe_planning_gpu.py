"""Bounded constrained motion probe: nominal goals remain independent of references.

Run in the existing recovery environment with this checkout's cuRobo on
PYTHONPATH. CUDA execution is serial; Torch's allocator is capped at 2 GiB.
This allocator cap does not bound other native allocations. This declared empty
synthetic world checks constrained planning only; it supplies no observed-scene,
independent Coal, actuator, or physical contact certification.
"""

# BEGIN bounded native goal/reference integration and candidate diagnostics.
import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True,
                        help="Verified native robot manifest supplying source solids and pair policy")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import DeviceCfg
    from manipulation.execution_binding import plain
    from manipulation.planning import PlanningBudget, plan_trajectory
    from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath, PoseEventRequirement
    from manipulation.planning_scene import ObservedSpace
    from manipulation.online_geometry import OnlinePlanningScene
    from probe_robot_fixture import load_probe_fixture, fixed_probe_hand, empty_background

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Native PRD probe requires accessible CUDA; no CPU substitute is reported")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(2*1024**3/total, 1.))
    torch.cuda.reset_peak_memory_stats()
    # BEGIN every variant uses the same explicitly supplied source robot.
    fixture, robot, validator = load_probe_fixture(args.fixture, reference_frame="synthetic_base")
    full_positions = np.asarray(robot["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"])
    device = DeviceCfg("cuda:0")
    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
    actual = kinematics.default_joint_state
    state = kinematics.compute_kinematics(actual)
    base = np.eye(4)
    base[:3, 3] = state.tool_poses.position[0, 0, 0].detach().cpu().numpy()
    wxyz = state.tool_poses.quaternion[0, 0, 0].detach().cpu().numpy()
    base[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
    base = base @ validator.T_frame_tcp
    joint_names = tuple(kinematics.joint_names)
    start_q = actual.position.detach().cpu().numpy().reshape(-1)
    tool_link, sphere_count = fixture["tool_link"], kinematics.total_spheres
    # END actual source names and TCP transforms replace built-in robot assumptions.
    del state, actual, kinematics
    gc.collect()
    # BEGIN retain the existing bounded integration work budget.
    # The three synthetic variants exercise the shared nominal objective and curve
    # export. Nonzero task errors remain diagnostics, not acceptance thresholds.
    budget = PlanningBudget(max_validation_points=512, max_task_anchors=32, max_robot_spheres=sphere_count,
                            search_seconds=3., maximum_trajectory_solves=2)
    # END keep the complete budget in every probe report.
    rows = []
    for name, reference_bend, goal_offset in (("straight", 0., .015),
            ("bent_same_goal", .02, .015), ("changed_goal", .02, .03)):
        goal = base.copy()
        goal[0, 3] += goal_offset
        # Both true goals are independently bound. The ordinary reference
        # retains the same nominal endpoint in the changed-goal case.
        reference_end = base.copy()
        reference_end[0, 3] += .015
        middle = reference_end.copy()
        middle[1, 3] += reference_bend
        source = f"Synthetic reference variant: {name}"
        path = PosePath(np.array([0., .5, 1.]), np.stack((base, middle, reference_end)), source)
        problem = PlanningProblem(joint_names, start_q, np.zeros_like(start_q), tool_link,
            "synthetic_base", np.eye(4), validator.T_frame_tcp, (MotionPhase(name, path),),
            pose_events=(PoseEventRequirement(1., np.eye(4), goal, 1e-4, 1e-3,
                "Explicit synthetic terminal reach goal"),),
            source_timestamps=np.array([0., 1., 2.]), source_progress=np.array([0., .5, 1.]))
        # BEGIN nominal fixed geometry is explicit and source installations stay present.
        hand, owners, hand_report = fixed_probe_hand(fixture, robot, problem, device=device,
                                                    full_current_positions=full_positions)
        coverage = ObservedSpace(np.ones((20, 20, 20), bool), origin=[-2., -2., -2.], voxel_size=.2,
            reference_frame="synthetic_base", observation_ids=("declared-synthetic-empty-box",))
        scene = OnlinePlanningScene(reference_frame="synthetic_base", background=empty_background(coverage),
            owners=owners, coverage=coverage, robot_sphere_materials=(None,)*sphere_count, gripper=hand,
            robot_geometry_source=fixture["robot_geometry_source"])
        started = time.monotonic()
        result = plan_trajectory(problem, scene, robot, robot_validator=validator, budget=budget)
        # END this probe evaluates planning only; no physical world is executed.
        row = {"case": name, "success": result.success, "elapsed_s": time.monotonic()-started,
               "goal": goal.tolist(), "reference": path.poses.tolist(), "diagnostics": result.diagnostics,
               "execution_binding": result.execution_binding, "fixed_hand_hypothesis": hand_report}
        (args.output/f"{name}.json").write_text(json.dumps(plain(row), indent=2, allow_nan=False)+"\n")
        rows.append({"case": name, "success": result.success, "elapsed_s": row["elapsed_s"]})
        del result, scene, problem
        gc.collect()
        torch.cuda.empty_cache()
    report = {"scope": __doc__, "cases": rows, "budget": asdict(budget),
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
              "physical_execution_tested": False}
    (args.output/"report.json").write_text(json.dumps(plain(report), indent=2, allow_nan=False)+"\n")
    print(json.dumps(report))
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
# END no exported diagnostic candidate is relabelled as executable.
