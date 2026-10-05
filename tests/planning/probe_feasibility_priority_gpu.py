"""One bounded two-phase native planning check in a declared empty scene.

The robot and installation solids come from an explicit verified fixture. The
stationary reference and small allowed goal offset are synthetic. This checks
solver construction, typed goal costs, source checks, quality rebuild and final
binding; it makes no recorded-task, perception or physical-execution claim.
"""

# BEGIN serial native integration with one IK and one trajectory seed.
import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import DeviceCfg
    from manipulation.execution_binding import plain
    from manipulation.planning import PlanningBudget, plan_trajectory
    from manipulation.planning_problem import (MotionPhase, NumericalTolerance,
        PlanningProblem, PoseEventRequirement, PosePath)
    from probe_robot_fixture import load_probe_fixture, fixed_probe_hand
    from probe_process_planner import no_obstacle_scene

    if not torch.cuda.is_available():
        raise RuntimeError("This native phase-switch check requires accessible CUDA")
    torch.set_num_threads(1)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.5, 2*1024**3/total), 0)
    torch.cuda.reset_peak_memory_stats(0)
    fixture, robot, validator = load_probe_fixture(arguments.fixture, reference_frame="robot_base")
    device = DeviceCfg("cuda:0")
    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
    initial = kinematics.default_joint_state
    state = kinematics.compute_kinematics(initial)
    index = state.tool_poses.tool_frames.index(fixture["tool_link"])
    reference = np.eye(4)
    reference[:3, 3] = state.tool_poses.position[0, 0, index].detach().cpu().numpy()
    quaternion = state.tool_poses.quaternion[0, 0, index].detach().cpu().numpy()
    reference[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    reference = reference @ validator.T_frame_tcp
    target = reference.copy()
    target[0, 3] += .0002
    problem = PlanningProblem(tuple(kinematics.joint_names),
        initial.position.detach().cpu().numpy().reshape(-1), np.zeros(len(kinematics.joint_names)),
        fixture["tool_link"], "robot_base", np.eye(4), validator.T_frame_tcp,
        (MotionPhase("synthetic_phase_check", PosePath(np.array([0., 1.]),
            np.stack((reference, reference)), "Explicit stationary synthetic reference")),),
        numerical=NumericalTolerance(position_m=.001, rotation_rad=.005),
        pose_events=(PoseEventRequirement(1., np.eye(4), target, .001, .005,
            "Synthetic target whose original tolerance contains the starting pose"),),
        source_timestamps=np.array([0., 1.]), source_progress=np.array([0., 1.]))
    hand, owners, hand_report = fixed_probe_hand(fixture, robot, problem, device=device,
        full_current_positions=np.asarray(robot["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"]))
    scene = no_obstacle_scene(kinematics.total_spheres, hand=hand,
        installation_owners=owners, geometry_source=fixture["robot_geometry_source"])
    budget = PlanningBudget(ik_seeds=1, trajectory_seeds=1, knots=6,
        ik_iterations=20, trajectory_iterations=25, max_robot_spheres=kinematics.total_spheres,
        max_validation_points=512, max_refinement_points=32,
        maximum_trajectory_solves=2, search_seconds=0., planning_seconds=120.)
    del state, initial, kinematics
    gc.collect()
    result = plan_trajectory(problem, scene, robot, robot_validator=validator, budget=budget)
    phases = [row["phase"] for row in result.diagnostics["attempts"]]
    report = {"scope": __doc__, "success": result.success, "phases": phases,
        "budget": asdict(budget), "diagnostics": result.diagnostics,
        "fixed_hand_hypothesis": hand_report,
        "execution_binding": result.execution_binding,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "physical_execution_tested": False}
    (arguments.output/"report.json").write_text(json.dumps(plain(report), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"success": result.success, "phases": phases,
        "failure": result.diagnostics.get("failure"), "reason": result.diagnostics.get("reason"),
        "elapsed_s": result.diagnostics["elapsed_seconds"], "peak_rss_mib": report["peak_rss_mib"]}))
    if not result.success or phases != ["feasibility", "quality"]:
        raise RuntimeError("The bounded native check did not return a bound curve after both phases")


if __name__ == "__main__":
    main()
# END external resource supervision also bounds allocations outside Torch.
