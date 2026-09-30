"""Bounded full-robot avoidance with the production planner and native queries.

All geometry here is explicitly synthetic. The Panda fingers are mechanically
locked by its supplied robot configuration, so this experiment tests a rigid
tool and makes no claim about actuated closing or observed real-world safety.
"""

# BEGIN synthetic robot collision-generation integration experiment.
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

# BEGIN resolve imports from this FUNCTO checkout, independent of its directory name.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# END project-local imports.


def run(output, knots=8):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"scope": "synthetic known world, full arm, rigid locked tool",
              "production_environment_validation": False, "status": "running"}
    try:
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
        from curobo._src.geom.types import Cuboid, SceneCfg
        from manipulation.planning_scene import CuroboSphereQuery, ObservedSpace, PlanningScene
        from manipulation.planning_problem import PlanningProblem, MotionPhase, PosePath, NumericalTolerance
        from manipulation import planning

        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        config = DeviceCfg("cuda:0")
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
        initial = kinematics.default_joint_position.reshape(-1).clone()
        middle, entrance, finish = initial.clone(), initial.clone(), initial.clone()
        middle[0] += .75
        entrance[0] += 1.5
        finish[0] += 1.6
        finish[1] -= .02
        joints = torch.stack((initial, middle, entrance, finish))[None].contiguous()
        state = kinematics.compute_kinematics(JointState.from_position(joints, joint_names=kinematics.joint_names))
        balls = state.robot_spheres[0].detach().cpu().numpy()
        p = state.tool_poses.position[0, :, 0].detach().cpu().numpy()
        r = state.tool_poses.quaternion[0, :, 0].detach().cpu().numpy()

        # Choose a small obstacle on the direct arm sweep, with explicit
        # clearance at the measured start and both required suffix endpoints.
        # This constructs a synthetic counterexample; it does not move an
        # observed real obstacle in response to an inconvenient prediction.
        half = np.array([.012, .012, .012])
        choices = []
        for index, center in enumerate(balls[1, :, :3]):
            signed_axes = np.abs(balls[..., :3]-center)-half
            signed = np.linalg.norm(np.maximum(signed_axes, 0), axis=-1) + np.minimum(signed_axes.max(-1), 0)
            clearance = signed-balls[..., 3]
            clearance = np.where(balls[..., 3] > 0, clearance, np.inf)
            end_clearance = float(clearance[[0, 2, 3]].min())
            if end_clearance > .015 and clearance[1].min() < -.015:
                choices.append((end_clearance, index, center.copy(), clearance.min(-1).tolist()))
        if not choices:
            raise ValueError("This declared synthetic construction has no clear endpoint obstacle placement")
        _, index, center, clearance = max(choices, key=lambda item: item[0])
        report["obstacle"] = {"center": center.tolist(), "dimensions": (2*half).tolist(),
                              "witness_robot_sphere": index, "direct_path_clearance_m": clearance}
        native = SceneCollision.from_config(SceneCollisionCfg(
            device_cfg=config, scene_model=SceneCfg(cuboid=[Cuboid(
                name="synthetic_obstacle", pose=[*center, 1., 0., 0., 0.], dims=(2*half).tolist())]),
            cache={"cuboid": 1}))
        background = CuroboSphereQuery(native, owner_ids=("synthetic_obstacle",),
                                       source="fully specified synthetic cuboid")
        coverage = ObservedSpace(np.ones((81, 81, 81), bool), origin=[-2., -2., -2.], voxel_size=.05,
                                 observation_ids=("explicit synthetic known-space grid",), reference_frame="robot_base")
        scene = PlanningScene(background=background, owners=(), coverage=coverage, reference_frame="robot_base",
                              robot_sphere_materials=(None,)*kinematics.total_spheres,
                              robot_geometry_source="supplied Franka spheres with mechanically locked fingers",
                              has_gripper=False)
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, :3, 3] = p[[0, 2, 3]]
        poses[:, :3, :3] = Rotation.from_quat(r[[0, 2, 3]][:, [1, 2, 3, 0]]).as_matrix()
        problem = PlanningProblem(tuple(kinematics.joint_names), initial.cpu().numpy(), np.zeros(len(initial)),
                                  "panda_hand", "robot_base", np.eye(4), np.eye(4),
                                  (MotionPhase("free_approach", PosePath(np.array([0., 1.]), poses[:2], "synthetic target"), "free"),
                                   MotionPhase("required_suffix", PosePath(np.array([0., 1.]), poses[1:], "synthetic necessary local path"))),
                                  numerical=NumericalTolerance(position_m=.001, rotation_rad=.005))
        budget = planning.PlanningBudget(ik_seeds=4, trajectory_seeds=2, ik_iterations=80,
                                         knots=knots, interpolation_buffer_size=768,
                                         trajectory_iterations=150, maximum_trajectory_solves=3, search_seconds=3.)
        report["knots"] = knots
        original = planning._CuroboPlanner.optimize
        report["generation_attempts"] = []

        def measured(planner, seed):
            saved = seed.detach().clone()
            result = original(planner, seed)
            # Evaluate a cloned initializer on the live particle layout to
            # expose the actual collision objective entering generation.
            particles = planner.trajopt.optimizer.optimizers[0].config.num_particles
            x = saved.reshape(-1, saved.shape[-2], saved.shape[-1]).repeat_interleave(particles, 0).requires_grad_(True)
            rollout = planner.trajopt.optimizer_rollouts[0]
            evaluated = rollout.evaluate_action(x, use_cuda_graph=False)
            terms = evaluated.costs_and_constraints.constraints
            value = terms.values[terms.names.index("process_scene")]
            gradient, = torch.autograd.grad(value.sum(), x)
            report["generation_attempts"].append({"scene_objective_at_initializer": float(value.sum().detach().cpu()),
                                                   "scene_gradient_max": float(gradient.abs().max().detach().cpu())})
            return result

        # BEGIN restore collision diagnostics after every planner invocation.
        # The same imported helper may be reused after a native or diagnostic
        # exception; neither path may leave its observer on the shared class.
        with patch.object(planning._CuroboPlanner, "optimize", new=measured):
            result = planning.plan_trajectory(problem, scene, "franka.yml", budget=budget)
        # END scoped collision diagnostics.
        report.update(status="completed", success=result.success, diagnostics=result.diagnostics)
        if result.trajectory is not None:
            curve = result.trajectory
            np.savez(output / "trajectory.npz", times=curve.times, position=curve.position, velocity=curve.velocity,
                     progress_u=curve.progress_u, progress_s=curve.progress_s)
            report["duration"] = curve.duration
        report.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (output / "exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--knots", type=int, default=8)
    args = parser.parse_args()
    run(args.output, args.knots)
# END synthetic robot collision-generation integration experiment.
