"""Bounded V2 process-manager integration on a synthetic obstacle-free robot.

This is a numerical API test, not a current-scene planning or execution claim.
It imports the production planner and limits all seed/horizon allocations.
"""

# BEGIN bounded explicit GPU integration experiment.
import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
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


# BEGIN use the production scene contract for this explicit empty fixture.
def no_obstacle_scene(sphere_count, *, hand, installation_owners, geometry_source):
    """Build a declared synthetic empty world, with no perception claim."""
    from manipulation.planning_scene import ObservedSpace
    from manipulation.online_geometry import OnlinePlanningScene
    from probe_robot_fixture import empty_background

    coverage = ObservedSpace(np.ones((20, 20, 20), bool), origin=[-2., -2., -2.], voxel_size=.2,
        reference_frame="robot_base", observation_ids=("declared synthetic empty box",))
    return OnlinePlanningScene(background=empty_background(coverage), owners=installation_owners,
        coverage=coverage, reference_frame="robot_base", robot_sphere_materials=(None,)*sphere_count,
        robot_geometry_source=geometry_source, gripper=hand,
        unknown_policy={"outside_regions": "forbid", "conditional_regions": []})
# END there is no partial mock of the changing planner/scene interface.


def run(output, debug_gradient, initial_joint_velocity=0., *, fixture_path):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"scope": "synthetic no-obstacle process-manager API integration",
              "production_environment_validation": False, "status": "running"}
    try:
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from manipulation.planning_problem import (PlanningProblem, MotionPhase, PosePath,
            NumericalTolerance, PoseEventRequirement, PoseProcessRequirement)
        from manipulation.planning import PlanningBudget, plan_trajectory
        from manipulation import planning as planner_module
        from probe_robot_fixture import load_probe_fixture, fixed_probe_hand

        if not torch.cuda.is_available():
            raise RuntimeError("Explicit GPU access is required for this bounded experiment")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        # BEGIN source model and nominal fixed fingers are explicit probe inputs.
        fixture, robot, validator = load_probe_fixture(fixture_path, reference_frame="robot_base")
        full_positions = np.asarray(robot["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"])
        device = DeviceCfg("cuda:0")
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
        initial = kinematics.default_joint_position.reshape(-1).clone()
        q1, q2 = initial.clone(), initial.clone()
        q1[0] += .08
        q1[1] -= .03
        q2[0] += .12
        q2[1] -= .04
        state = kinematics.compute_kinematics(JointState.from_position(
            torch.stack((initial, q1, q2))[None].contiguous(), joint_names=kinematics.joint_names))
        p = state.tool_poses.position[0, :, 0].detach().cpu().numpy()
        r = state.tool_poses.quaternion[0, :, 0].detach().cpu().numpy()
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, :3, 3] = p
        poses[:, :3, :3] = Rotation.from_quat(r[:, [1, 2, 3, 0]]).as_matrix()
        poses = poses @ validator.T_frame_tcp
        # END the same verified TCP transform defines reference and planning poses.
        # BEGIN measured nonzero boundary-velocity regression.
        measured_velocity = np.zeros(len(initial))
        measured_velocity[0] = initial_joint_velocity
        report["measured_initial_velocity"] = measured_velocity.tolist()
        # BEGIN the suffix is an explicit synthetic process, separate from its seed.
        suffix = PosePath(np.array([0., 1.]), poses[1:], "synthetic required local relation")
        problem = PlanningProblem(
            tuple(kinematics.joint_names), initial.cpu().numpy(), measured_velocity,
            fixture["tool_link"], "robot_base", np.eye(4), validator.T_frame_tcp,
            (MotionPhase("approach_reference", PosePath(np.array([0., 1.]), poses[:2], "synthetic FK reference")),
             MotionPhase("local_reference", suffix)),
            numerical=NumericalTolerance(position_m=.001, rotation_rad=.005),
            pose_events=(PoseEventRequirement(2., np.eye(4), poses[-1], .001, .005,
                "Independent synthetic terminal reach goal"),),
            pose_relations=(PoseProcessRequirement(1., 2., suffix, np.zeros((6, 2)),
                "Explicit fixture requirement to follow the local relation after free approach"),),
            source_timestamps=np.array([0., 1., 2.]), source_progress=np.array([0., 1., 2.]))
        # END source reference samples alone impose no pose obligation.
        # END measured nonzero boundary-velocity regression.
        budget = PlanningBudget(ik_seeds=4, trajectory_seeds=1, max_robot_spheres=kinematics.total_spheres,
                                trajectory_iterations=125, search_seconds=0.)
        report["budget"] = asdict(budget)
        # BEGIN preserve complete installation owners and fixed-geometry evidence.
        hand, owners, hand_report = fixed_probe_hand(fixture, robot, problem, device=device,
                                                    full_current_positions=full_positions)
        scene = no_obstacle_scene(kinematics.total_spheres, hand=hand, installation_owners=owners,
                                  geometry_source=fixture["robot_geometry_source"])
        report["fixed_hand_hypothesis"] = hand_report
        # END an empty synthetic background does not erase source robot fixtures.
        # BEGIN keep optional diagnostics within one planner invocation.
        if debug_gradient:
            original_optimize = planner_module._ConstrainedPlanner.optimize

            def measured_optimize(planner, seed):
                result = original_optimize(planner, seed)
                traces = []
                for rollout in planner.trajopt.optimizer_rollouts:
                    # The live optimizer retains one goal index per line-search
                    # particle, so diagnostic actions must preserve that axis.
                    particles = planner.trajopt.optimizer.optimizers[0].config.num_particles
                    x = seed.detach().reshape(-1, seed.shape[-2], seed.shape[-1]).repeat_interleave(
                        particles, dim=0).clone().requires_grad_(True)
                    evaluated = rollout.evaluate_action(x, use_cuda_graph=False)
                    collection = evaluated.costs_and_constraints
                    row = {"rollout": rollout.rollout_instance_name,
                           "manager": type(rollout.constraint_manager).__name__,
                           "process_mode": rollout.constraint_manager._process_mode,
                           "terms": {}}
                    for label, terms in (("cost", collection.costs), ("constraint", collection.constraints)):
                        for name, value in zip(terms.names, terms.values):
                            grad = torch.autograd.grad(value.sum(), x, retain_graph=True, allow_unused=True)[0] if value.requires_grad else None
                            row["terms"][f"{label}:{name}"] = {
                                "sum": float(value.detach().sum().cpu()),
                                "requires_grad": value.requires_grad,
                                "gradient_present": grad is not None,
                                "gradient_nonfinite": None if grad is None else int((~torch.isfinite(grad)).sum().cpu()),
                                "finite_gradient_max": None if grad is None else float(torch.nan_to_num(grad).abs().max().cpu())}
                    traces.append(row)
                report["gradient_trace"] = traces
                report["max_control_change"] = float((result.solution.reshape_as(seed)-seed).abs().max().detach().cpu())
                return result

        # A native solve or diagnostic callback may raise. ExitStack restores
        # the prior method before the outer error handler records the failure.
        with ExitStack() as instrumentation:
            if debug_gradient:
                instrumentation.enter_context(patch.object(
                    planner_module._ConstrainedPlanner, "optimize", new=measured_optimize))
            result = plan_trajectory(problem, scene, robot, robot_validator=validator, budget=budget)
        # END scoped optional diagnostics.
        report.update(status="completed", success=result.success, diagnostics=result.diagnostics)
        if result.trajectory is not None:
            t = result.trajectory
            np.savez(output / "trajectory.npz", times=t.times, position=t.position, velocity=t.velocity,
                     progress_u=t.progress_u, progress_s=t.progress_s)
            report["duration"] = t.duration
            # BEGIN independent boundary-velocity check on the actual curve.
            _, actual_velocity, _, _ = t.sample(0.)
            report["exported_initial_velocity"] = actual_velocity.tolist()
            report["initial_velocity_max_error"] = float(np.max(np.abs(actual_velocity-measured_velocity)))
            if report["initial_velocity_max_error"] > problem.numerical.velocity:
                raise AssertionError("The accepted curve changed the measured initial velocity")
            # END independent boundary-velocity check on the actual curve.
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
    parser.add_argument("--fixture", type=Path, required=True,
                        help="Verified native robot manifest supplying source solids and pair policy")
    parser.add_argument("--debug-gradient", action="store_true")
    parser.add_argument("--initial-joint-velocity", type=float, default=0.,
                        help="Synthetic measured velocity of the first source arm joint in rad/s")
    args = parser.parse_args()
    run(args.output, args.debug_gradient, args.initial_joint_velocity, fixture_path=args.fixture)
# END bounded explicit GPU integration experiment.
