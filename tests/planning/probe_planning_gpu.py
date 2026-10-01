"""Bounded V2 integration probe; the root agent runs GPU work serially.

This is a synthetic open-space process test, not a perception/scene or robot
execution test.  A bent necessary suffix must be generated through the custom
constraint while native terminal pose tracking is disabled.  No packages or
environment settings are installed or changed by this script.
"""

# BEGIN bounded full-process GPU integration probe.
import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # BEGIN resolve this checkout without a workspace or date-directory dependency.
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project))
    # END project-local imports.

    import torch
    from scipy.spatial.transform import Rotation
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    from manipulation.planning import PlanningBudget, plan_trajectory
    from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath

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
    result = plan_trajectory(problem, DeclaredOpenScene(kinematics.total_spheres), "franka.yml",
                             budget=PlanningBudget(search_seconds=0.))
    report = {"success": result.success, "diagnostics": result.diagnostics,
              "scope": "synthetic open-scene full-process optimization; no robot execution",
              "execution": "native eager",
              "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if result.trajectory is not None:
        trajectory = result.trajectory
        report["joint_change_norm"] = float(np.linalg.norm(trajectory.position[-1]-trajectory.position[0]))
        report["duration_seconds"] = trajectory.duration
        np.savez(args.output.with_suffix(".npz"), times=trajectory.times,
                 position=trajectory.position, velocity=trajectory.velocity,
                 progress_u=trajectory.progress_u, progress_s=trajectory.progress_s)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# END bounded full-process GPU integration probe.
