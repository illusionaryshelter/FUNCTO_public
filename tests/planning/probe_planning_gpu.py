"""Bounded native PRD probe: real goals remain independent of references.

Run in the existing recovery environment with this checkout's cuRobo on
PYTHONPATH. CUDA execution is serial; Torch's allocator is capped at 2 GiB.
This allocator cap does not bound other native allocations. This declared empty
synthetic world checks native planning only; it supplies no observed-scene,
independent Coal, actuator, or physical contact certification.
"""

# BEGIN bounded native goal/reference integration with formal result evidence.
import argparse
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


class EmptyBackground:
    """An explicitly empty synthetic fixture, never a perception substitute."""
    owner_ids = ()
    source = "Declared synthetic empty world for native planner regression"
    independent_geometry = {"solids": [], "query_gaps": []}

    def __call__(self, spheres, *, key):
        return spheres[..., 0]*0.

    def clone_for_rollout(self):
        return EmptyBackground()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from manipulation.execution_binding import plain
    from manipulation.planning import PlanningBudget, plan_trajectory
    from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath, PoseEventRequirement
    from manipulation.planning_scene import ObservedSpace, PlanningScene

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Native PRD probe requires accessible CUDA; no CPU substitute is reported")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(2*1024**3/total, 1.))
    torch.cuda.reset_peak_memory_stats()
    robot = "franka.yml"
    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(robot))
    actual = kinematics.default_joint_state
    state = kinematics.compute_kinematics(actual)
    base = np.eye(4)
    base[:3, 3] = state.tool_poses.position[0, 0, 0].detach().cpu().numpy()
    wxyz = state.tool_poses.quaternion[0, 0, 0].detach().cpu().numpy()
    base[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
    joint_names = tuple(kinematics.joint_names)
    start_q = actual.position.detach().cpu().numpy().reshape(-1)
    tool_link, sphere_count = kinematics.tool_frames[0], kinematics.total_spheres
    del state, actual, kinematics
    gc.collect()
    # BEGIN explicit integration budget; task precision and device policy stay fixed.
    # The archived 1 s / 256-query run exhausted its finite OMPL connection
    # budget for the changed goal. This probe grants bounded additional search
    # work; it does not change production defaults or relax an acceptance gate.
    budget = PlanningBudget(max_validation_points=512, max_refinement_points=32,
                            search_seconds=3., maximum_trajectory_solves=3)
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
            "synthetic_base", np.eye(4), np.eye(4), (MotionPhase(name, path),),
            pose_events=(PoseEventRequirement(1., np.eye(4), goal, 1e-4, 1e-3,
                "Explicit synthetic terminal reach goal"),),
            source_timestamps=np.array([0., 1., 2.]), source_progress=np.array([0., .5, 1.]))
        scene = PlanningScene(reference_frame="synthetic_base", background=EmptyBackground(), owners=(),
            coverage=ObservedSpace(np.ones((20, 20, 20), bool), origin=[-2., -2., -2.], voxel_size=.2,
                                   reference_frame="synthetic_base", observation_ids=("declared-synthetic-empty-box",)),
            robot_sphere_materials=(None,)*sphere_count, has_gripper=False,
            robot_geometry_source="Native franka.yml robot geometry; explicit empty synthetic fixture")
        started = time.monotonic()
        result = plan_trajectory(problem, scene, robot, budget=budget)
        row = {"case": name, "success": result.success, "elapsed_s": time.monotonic()-started,
               "goal": goal.tolist(), "reference": path.poses.tolist(), "diagnostics": result.diagnostics,
               "execution_binding": result.execution_binding}
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
