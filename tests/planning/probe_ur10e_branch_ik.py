"""One bounded native UR10e IK audit before any paired planning experiment.

The robot, measured synthetic q0 and tool frame come from the installed model.
Only one eight-seed IK call is made. Actual collision-sphere geometry separates
physical branches from different 2*pi joint representations. Native Jacobians
are checked at finite nearby configurations; this does not certify the entire
pose-tolerance tube or prove that every future path must change branch.
No returned IK joint solution is supplied to a production trajectory planner.
"""

# BEGIN independent bounded six-DOF branch and Jacobian evidence.
import argparse
import json
from pathlib import Path
import resource
import time
import traceback
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

# BEGIN independent IK diagnostics own their three small numerical helpers.
def pose_matrices(position, quaternion):
    """Convert native wxyz FK arrays independently through NumPy/SciPy."""
    position, quaternion = np.asarray(position), np.asarray(quaternion)
    result = np.broadcast_to(np.eye(4), (*position.shape[:-1], 4, 4)).copy()
    result[..., :3, 3] = position
    result[..., :3, :3] = Rotation.from_quat(quaternion.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(
        *position.shape[:-1], 3, 3)
    return result


def error(first, second):
    """Measure physical pose error without optimizer clipping or weights."""
    return {"position_m": float(np.linalg.norm(first[:3, 3]-second[:3, 3])),
            "rotation_rad": float(Rotation.from_matrix(first[:3, :3].T@second[:3, :3]).magnitude())}


def safe_json(value):
    """Keep nonfinite diagnostic evidence without emitting invalid JSON."""
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value
# END the removed endpoint-family experiment is not an IK dependency.


ROBOT = "ur10e.yml"
IK_SEEDS = 8
IK_ITERATIONS = 200
GEOMETRY_EQUIVALENCE_M = 1e-5
MINIMUM_USEFUL_SPHERE_SEPARATION_M = .02
MINIMUM_JACOBIAN_SINGULAR_VALUE = 1e-3
NEARBY_JOINT_OFFSET_RAD = .001
MAX_LOCAL_FK_CONFIGURATIONS = (IK_SEEDS+1)*13


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "success": False,
              "scope": "native six-DOF IK candidates and finite local Jacobian evidence",
              "robot": ROBOT, "physical_grasp_validation": False,
              "whole_trajectory_validation": False, "witness_joint_injection": False,
              "limits": {"native_ik_calls": 1, "ik_seeds": IK_SEEDS,
                         "lbfgs_iterations": IK_ITERATIONS, "max_local_fk_configurations": MAX_LOCAL_FK_CONFIGURATIONS,
                         "torch_memory_cap_bytes": 2*1024**3}}
    solver = None
    try:
        import torch
        from curobo.content import get_robot_configs_path
        from curobo._src.util.config_io import join_path, load_yaml
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, GoalToolPose, JointState

        if not torch.cuda.is_available():
            raise RuntimeError("This bounded native probe requires explicit serial GPU execution")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        device = DeviceCfg("cuda:0")
        configuration = load_yaml(join_path(get_robot_configs_path(), ROBOT))["robot_cfg"]["kinematics"]
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(ROBOT), compute_jacobian=True)
        current = kinematics.default_joint_position.reshape(-1).clone()
        if len(current) != 6 or len(kinematics.tool_frames) != 1:
            raise ValueError("The installed fixture must retain six native joints and one configured tool")
        tool_link = kinematics.tool_frames[0]

        def fk(values):
            q = device.to_device(np.asarray(values))[None].contiguous()
            state = kinematics.compute_kinematics(JointState.from_position(q, joint_names=kinematics.joint_names))
            tools = pose_matrices(state.tool_poses.position[0, :, 0].detach().cpu().numpy().copy(),
                                  state.tool_poses.quaternion[0, :, 0].detach().cpu().numpy().copy())
            return (tools, state.robot_spheres[0].detach().cpu().numpy().copy(),
                    state.tool_jacobians[0, :, 0].detach().cpu().numpy().copy())

        current_q = current.detach().cpu().numpy().copy()
        reference_q = np.broadcast_to(current_q, (4, 6)).copy()
        reference_q[:, 0] += np.linspace(0., .45, 4)
        reference_tools, reference_spheres, reference_jacobians = fk(reference_q)
        active = reference_spheres[0, :, 3] > 0
        limits = kinematics.get_joint_limits().position.detach().cpu().numpy().copy()
        report["model"] = {"joint_names": list(kinematics.joint_names), "base_link": configuration["base_link"],
                           "tool_link": tool_link, "current_q": current_q.tolist(), "joint_limits": limits.tolist(),
                           "total_spheres": kinematics.total_spheres, "active_spheres": int(active.sum()),
                           "reference_definition": "native FK of q0 with first joint increment 0..0.45 rad at four nodes",
                           "reference_joint_values_are_planner_seeds": False}
        report["reference_jacobian_singular_values"] = np.linalg.svd(reference_jacobians, compute_uv=False).tolist()

        ik_configuration = InverseKinematicsCfg.create(
            robot=ROBOT, optimizer_configs=["ik/lbfgs_ik.yml"], device_cfg=device,
            num_seeds=IK_SEEDS, seed_solver_num_seeds=IK_SEEDS, max_batch_size=1, max_goalset=1,
            use_cuda_graph=False, position_tolerance=.0001, orientation_tolerance=.001,
            override_optimizer_num_iters={"lbfgs": IK_ITERATIONS}, random_seed=123)
        ik_configuration.exit_early = False
        solver = InverseKinematics(ik_configuration)
        # BEGIN use the production q0 warm start and native internal sampler.
        # SeedIK owns a separate Halton stream from the main seed_manager.
        # Supplying only q0 lets the unmodified native call append seven seeds
        # in exactly its usual order. This observer copies the actual generated
        # seeds for evidence without advancing either sampler or changing data.
        provided_seeds = current[None, None].contiguous()
        generated_seed_batches = []
        generate_native_seeds = solver.seed_ik_solver._generate_seed_configs

        def record_native_seeds(*args, **kwargs):
            generated = generate_native_seeds(*args, **kwargs)
            generated_seed_batches.append(generated.detach().cpu().numpy().copy())
            return generated

        target = reference_tools[0]
        quaternion = Rotation.from_matrix(target[:3, :3]).as_quat()[[3, 0, 1, 2]]
        goal = GoalToolPose(tool_frames=[tool_link],
                           position=device.to_device(target[None, None, None, None, :3, 3]).contiguous(),
                           quaternion=device.to_device(quaternion[None, None, None, None]).contiguous())
        with patch.object(solver.seed_ik_solver, "_generate_seed_configs", new=record_native_seeds):
            result = solver.solve_pose(goal, current_state=JointState.from_position(
                current[None], joint_names=kinematics.joint_names),
                seed_config=provided_seeds, return_seeds=IK_SEEDS)
        if len(generated_seed_batches) != 1 or generated_seed_batches[0].shape != (1, IK_SEEDS, 6):
            raise ValueError("The single native IK call did not generate exactly one eight-seed batch")
        seeds = generated_seed_batches[0][0]
        # END use the production q0 warm start and native internal sampler.
        values = solver.get_active_js(result.js_solution).position[0].detach().cpu().numpy().copy()
        success = result.success[0].detach().cpu().numpy().copy()
        feasible = result.feasible[0].detach().cpu().numpy().copy()
        if not np.isfinite(values).all():
            raise ValueError("Native IK returned nonfinite candidates")
        tools, spheres, jacobians = fk(values)
        singular_values = np.linalg.svd(jacobians, compute_uv=False)
        geometry_distance = np.linalg.norm(spheres[:, None, active, :3]-spheres[None, :, active, :3], axis=-1).max(-1)
        distance_from_current = np.linalg.norm(spheres[:, active, :3]-reference_spheres[0, active, :3], axis=-1).max(-1)
        pose_errors = [error(tool, target) for tool in tools]
        physically_converged = np.array([item["position_m"] <= .0001 and item["rotation_rad"] <= .001
                                         for item in pose_errors])
        usable = success & physically_converged & (singular_values[:, -1] > MINIMUM_JACOBIAN_SINGULAR_VALUE)
        current_geometry_covered = bool(np.any(usable & (distance_from_current <= GEOMETRY_EQUIVALENCE_M)))
        order = np.argsort(np.linalg.norm(values-current_q, axis=-1), kind="stable")
        representatives = []
        for index in order:
            if usable[index] and all(geometry_distance[index, previous] > GEOMETRY_EQUIVALENCE_M
                                     for previous in representatives):
                representatives.append(int(index))
        alternatives = [index for index in representatives
                        if distance_from_current[index] >= MINIMUM_USEFUL_SPHERE_SEPARATION_M]
        alternatives.sort(key=lambda index: -distance_from_current[index])
        alternatives = alternatives[:2]
        report["native_ik"] = {
            "exit_early": bool(solver.config.exit_early), "seed_config": seeds.tolist(),
            "provided_seed_config": provided_seeds[0].detach().cpu().numpy().tolist(),
            "seed_policy": "q0 warm start plus seven seeds generated by the native SeedIK Halton sampler",
            "native_sampler_seed": solver.seed_ik_solver.config.sampler_seed,
            "native_seed_generation_calls": len(generated_seed_batches),
            "success": success.tolist(), "feasible": feasible.tolist(),
            "native_position_error": result.position_error[0].detach().cpu().numpy().tolist(),
            "native_rotation_error": result.rotation_error[0].detach().cpu().numpy().tolist(),
            "actual_tool_pose_errors": pose_errors, "joint_position": values.tolist(),
            "raw_joint_distance_from_q0_rad": np.linalg.norm(values-current_q, axis=-1).tolist(),
            # Wrapped differences are diagnostics only. Native joint values,
            # bounds, seeds and commands are never wrapped or shortened.
            "wrapped_joint_distance_from_q0_rad": np.linalg.norm(
                np.arctan2(np.sin(values-current_q), np.cos(values-current_q)), axis=-1).tolist(),
            "maximum_active_sphere_displacement_from_q0_m": distance_from_current.tolist(),
            "pairwise_maximum_active_sphere_displacement_m": geometry_distance.tolist(),
            "jacobian_singular_values": singular_values.tolist(),
            "geometry_equivalence_tolerance_m": GEOMETRY_EQUIVALENCE_M,
            "current_geometry_has_native_successful_representative": current_geometry_covered,
            "physical_branch_representatives": representatives, "selected_alternative_indices": alternatives,
        }
        centers = np.concatenate((current_q[None], values[representatives]))
        offsets = np.concatenate((np.zeros((1, 6)), np.eye(6)*NEARBY_JOINT_OFFSET_RAD,
                                  -np.eye(6)*NEARBY_JOINT_OFFSET_RAD))
        nearby = (centers[:, None]+offsets[None]).reshape(-1, 6)
        nearby = nearby[np.all((nearby >= limits[0]) & (nearby <= limits[1]), axis=-1)]
        if len(nearby) > MAX_LOCAL_FK_CONFIGURATIONS:
            raise ValueError("Finite local Jacobian audit exceeded its declared workspace")
        _, _, nearby_jacobians = fk(nearby)
        nearby_singular_values = np.linalg.svd(nearby_jacobians, compute_uv=False)
        nearby_nonsingular = bool(np.all(nearby_singular_values[:, -1] > MINIMUM_JACOBIAN_SINGULAR_VALUE))
        report["local_jacobian_audit"] = {
            "configurations": len(nearby), "axis_offsets_rad": NEARBY_JOINT_OFFSET_RAD,
            "raw_geometric_jacobian_units": "linear rows m/rad, angular rows rad/rad; mixed units are retained",
            "minimum_singular_value": float(nearby_singular_values[:, -1].min()),
            "numerical_minimum_threshold": MINIMUM_JACOBIAN_SINGULAR_VALUE,
            "all_sampled_configurations_nonsingular": nearby_nonsingular,
            "scope": "finite native configurations near q0 and successful physical branches; no whole-tube claim",
        }
        np.savez(output/"native_ik_evidence.npz", current_q=current_q, seed_config=seeds, reference_q=reference_q,
                 reference_tools=reference_tools, reference_spheres=reference_spheres,
                 reference_jacobians=reference_jacobians, candidate_q=values, candidate_tools=tools,
                 candidate_spheres=spheres, candidate_jacobians=jacobians,
                 candidate_success=success, selected_alternative_indices=np.asarray(alternatives, dtype=int))
        report.update(status="completed", success=current_geometry_covered and bool(alternatives) and nearby_nonsingular)
        if not report["success"]:
            report["failure"] = "bounded_native_ik_did_not_establish_a_distinct_regular_physical_branch"
        return report
    except Exception as exception:
        report.update(status="error", error=repr(exception), traceback=traceback.format_exc())
        return report
    finally:
        if solver is not None:
            solver.destroy()
        report["elapsed_seconds"] = time.monotonic()-started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        except (NameError, RuntimeError):
            pass
        (output/"report.json").write_text(json.dumps(safe_json(report), indent=2, allow_nan=False)+"\n")
        print(json.dumps(safe_json(report), allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.output)
    raise SystemExit(0 if result["success"] else 1)
# END independent bounded six-DOF branch and Jacobian evidence.
