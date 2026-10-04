"""Bounded complete planning and entrance checks with two exact source boxes.

The model is shared with the anchor regression. No physical robot, contact
task, observation reconstruction or independent environment claim is involved.
Run serially under the existing external RAM/device monitor.
"""

# BEGIN complete public API regression with explicit source-solid evidence.
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from probe_anchor_self_collision_gpu import _robot_config, _source_scene


def run(output):
    import torch
    from manipulation.execution_binding import plain
    from manipulation.planning import PlanningBudget, _CuroboPlanner, plan_trajectory
    from manipulation.planning_problem import MotionPhase, PlanningProblem, PoseEventRequirement, PosePath

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(
        min(.5, 2*1024**3/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    robot = _robot_config(output)
    budget = PlanningBudget(ik_seeds=2, trajectory_seeds=1, knots=6, ik_iterations=40,
        trajectory_iterations=100, maximum_trajectory_dt=.1, maximum_trajectory_solves=1,
        search_seconds=0., max_robot_spheres=8, max_validation_points=128,
        max_refinement_points=8, planning_seconds=90.)
    report = {"scope": __doc__, "budget": asdict(budget), "cases": [], "passed": False}
    started = time.monotonic()

    def problem_at(start, end):
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[:, 0, 3] = [start, end]
        return PlanningProblem(("slide",), np.array([start]), np.zeros(1), "slider", "robot_base",
            np.eye(4), np.eye(4), (MotionPhase("source_box_reach", PosePath(
                np.array([0., 1.]), poses, "explicit two-box slider reference")),),
            pose_events=(PoseEventRequirement(1., np.eye(4), poses[-1], 1e-4, 1e-3,
                "explicit source-box slider terminal goal"),),
            source_timestamps=np.array([0., 1.]), source_progress=np.array([0., 1.]))

    try:
        # These cases use identical complete models, changing only the actual
        # named state. The first has proxy overlap but 30 mm solid clearance.
        for name, q0, expected in (("proxy_overlap_source_separated", .1, True),
                                   ("true_source_collision", .03, False)):
            problem = problem_at(q0, q0)
            scene = _source_scene(robot, problem)
            planner = _CuroboPlanner(problem, scene, robot, budget, "cuda:0",
                                     robot_validator=scene.robot_validator)
            try:
                diagnostics = planner.initial_state_diagnostics()
                report["cases"].append({"case": name, "expected_passed": expected,
                    "passed": diagnostics["passed"] == expected, "initial_state": diagnostics,
                    "analytic_sphere_penetration_m": max(0., .08-abs(q0-.03))})
                if diagnostics["passed"] != expected:
                    raise AssertionError(f"Source-solid entrance decision was incorrect: {name}")
            finally:
                planner.close()
            del planner, scene
            gc.collect()
            torch.cuda.empty_cache()

        # A short reach on the other side of the fixed box exercises the full
        # public request, native success formula and final cubic acceptance.
        problem = problem_at(-.3, -.28)
        scene = _source_scene(robot, problem)
        result = plan_trajectory(problem, scene, robot, robot_validator=scene.robot_validator, budget=budget)
        row = {"case": "complete_public_reach", "passed": result.success,
            "diagnostics": result.diagnostics, "execution_binding": result.execution_binding,
            "request_ledger": result.request_ledger}
        if result.trajectory is not None:
            curve = result.trajectory
            row.update(duration_s=curve.duration, residuals=curve.residuals)
            np.savez(output/"trajectory.npz", times=curve.times, position=curve.position,
                     velocity=curve.velocity, progress_u=curve.progress_u, progress_s=curve.progress_s)
        report["cases"].append(row)
        if not result.success:
            raise AssertionError("The clear two-box public reach was not accepted")
        report["passed"] = True
    finally:
        report.update(elapsed_s=time.monotonic()-started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (output/"report.json").write_text(json.dumps(plain(report), indent=2, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output.resolve()))
# END a successful source gate does not bypass any other native constraint.
