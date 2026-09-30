"""Bounded native Franka experiment for a future obstacle changing grasp q.

The two production solves use the same measured q0, fixed grasp EEF, rigid
candidate attachment and complete future pose requirements. Only a cuboid
obstacle changes. A separate native IK search may witness another redundant
posture, but its joints are never supplied to either production planner.

This is synthetic conditional geometry with mechanically locked fingers. It
does not validate real grasping, observed Holding, moving closure, execution,
or completeness of IK/trajectory search. An unavailable witness or failed
production solve is reported as limited coverage, not physical impossibility.
"""

# BEGIN bounded paired future-constraint integration experiment.
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from probe_on_endpoint_family import (
    box_clearance, error, first_time_at_progress, pose_matrices, safe_json,
)


IK_SEEDS = 8
MAX_ALTERNATIVE_BRANCHES = 2
FUTURE_NODES = 4
MAX_FIXTURE_IK_SOLVES = 1+MAX_ALTERNATIVE_BRANCHES*(FUTURE_NODES-1)
MAX_FOREARM_SPHERES = 32
MINIMUM_BRANCH_DISTANCE_RAD = .2
MINIMUM_GRASP_CHANGE_RAD = .05
SURFACE_PENETRATIONS_M = (.0035, .0055)
REQUIRED_OBSTACLE_CLEARANCE_M = .012
MAX_CUBOID_CENTERS = (FUTURE_NODES-1)*MAX_FOREARM_SPHERES*MAX_ALTERNATIVE_BRANCHES*len(SURFACE_PENETRATIONS_M)


# BEGIN finite shallow sphere-surface obstacle construction.
def surface_obstacle_center(sphere, alternative_sphere, half, penetration):
    """Place one cuboid along the ray away from a corresponding alternative.

    Solve the exact analytic sphere/cuboid distance for the requested shallow
    overlap. The cuboid keeps its original dimensions and axis alignment.
    A zero displacement has no declared direction and supplies no candidate.
    Root evaluations only locate this one candidate; they do not search over
    obstacle directions, sizes, robot joints or additional scene variants.
    """
    direction = np.asarray(sphere[:3], float)-alternative_sphere[:3]
    separation = float(np.linalg.norm(direction))
    if separation == 0.:
        return None
    direction /= separation
    origin = np.asarray(sphere[:3], float)
    # This upper distance puts the selected sphere outside the box by at
    # least penetration; the ray origin lies inside the box and sphere.
    upper = float(sphere[3]+np.linalg.norm(half)+penetration)

    def residual(distance):
        return float(box_clearance(sphere, origin+distance*direction, half))+penetration

    distance = brentq(residual, 0., upper, xtol=1e-12, maxiter=64)
    return origin+distance*direction
# END finite shallow sphere-surface obstacle construction.


def make_problem(joint_names, current_q, tools):
    """A fixed grasp boundary followed by an unchanged necessary carry path.

    The initial EEF already equals the nominal grasp pose. A free connection
    still permits a different redundant configuration at the grasp boundary;
    the measured initial joints stay fixed. This isolates that boundary choice
    without requiring another approach/contact construction.
    """
    from manipulation.planning_problem import (
        MotionPhase, NumericalTolerance, PlanningEvent, PlanningProblem, PosePath,
    )

    tool_body = np.eye(4)
    tool_body[:3, 3] = [0., 0., .22]
    attachment = np.linalg.inv(tool_body)
    bodies = tools @ tool_body
    initial = bodies[0].copy()
    source = "declared synthetic common grasp and future carry reference"
    hold_id = "grasp:pending_actual_holding"
    common = {"skill_id": "grasp", "carrier_id": "branch_body",
              "owner_initial_pose": initial.tolist()}
    approach = MotionPhase("free_before_grasp", PosePath(np.array([0., 1.]),
                           np.stack((tools[0], tools[0])), source), kind="free",
                           metadata={**deepcopy(common), "mode": "free_pregrasp"})
    carry = MotionPhase("necessary_future_carry", PosePath(np.linspace(0., 1., len(tools)),
                        bodies, source, attachment=attachment),
                        metadata={**deepcopy(common), "skill_id": "carry", "mode": "held",
                                  "attachment": attachment.tolist(), "attachment_interval": [1., 2.],
                                  "attachment_condition_id": hold_id, "release_condition_id": None,
                                  "conditional": True})
    evidence = "actual stable grasp feedback is not supplied by this synthetic fixture"
    problem = PlanningProblem(
        tuple(joint_names), current_q, np.zeros(len(current_q)), "panda_hand", "robot_base",
        np.eye(4), np.eye(4), (approach, carry),
        events=(PlanningEvent("verify_holding", "grasp", 1., 0, "verification", source, evidence, True),),
        numerical=NumericalTolerance(position_m=.001, rotation_rad=.005),
        conditions=({"kind": "candidate_attachment", "condition_id": hold_id,
                     "carrier_id": "branch_body", "progress": 1.,
                     "T_carrier_eef": attachment.tolist(), "requires_evidence": evidence},))
    return problem, initial, tool_body


def run(output, knots):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {
        "status": "running", "success": False,
        "scope": "synthetic native Franka joint optimization; paired future-obstacle experiment",
        "physical_grasp_validation": False, "moving_gripper_validation": False,
        "production_environment_validation": False, "joint_branch_selection_validation": False,
        "fixture_limits": {"ik_seeds": IK_SEEDS, "alternative_branches": MAX_ALTERNATIVE_BRANCHES,
                           "future_reference_nodes": FUTURE_NODES,
                           "fixture_ik_solve_calls": MAX_FIXTURE_IK_SOLVES,
                           "forearm_spheres": MAX_FOREARM_SPHERES,
                           "cuboid_centers": MAX_CUBOID_CENTERS,
                           "centers_per_node_sphere_branch": len(SURFACE_PENETRATIONS_M),
                           "sphere_penetration_targets_m": list(SURFACE_PENETRATIONS_M),
                           "complete_planner_calls": 2, "trajectory_seeds": 2,
                           "torch_memory_cap_bytes": 2*1024**3},
        "witness_joint_injection": False,
    }
    fixture_ik = None
    try:
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, GoalToolPose, JointState
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
        from curobo._src.geom.types import Cuboid, SceneCfg
        from manipulation.planning import PlanningBudget, plan_trajectory
        from manipulation.planning_problem import PosePath
        from manipulation.planning_scene import (
            Attachment, CollisionOwner, CuroboSphereQuery, ObservedSpace, PlanningScene,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("This bounded probe requires explicit serial GPU execution")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        device = DeviceCfg("cuda:0")
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
        current = kinematics.default_joint_position.reshape(-1).clone()
        reference_q = current[None].repeat(FUTURE_NODES, 1)
        reference_q[:, 0] += torch.linspace(0., .45, FUTURE_NODES, device=current.device)

        def fk(configurations):
            values = torch.as_tensor(np.asarray(configurations), device=current.device, dtype=current.dtype)
            state = kinematics.compute_kinematics(JointState.from_position(
                values[None].contiguous(), joint_names=kinematics.joint_names))
            return (pose_matrices(state.tool_poses.position[0, :, 0].detach().cpu().numpy().copy(),
                                  state.tool_poses.quaternion[0, :, 0].detach().cpu().numpy().copy()),
                    state.robot_spheres[0].detach().cpu().numpy().copy())

        tools, _ = fk(reference_q.cpu().numpy())
        problem, initial_body, tool_body = make_problem(kinematics.joint_names, current.cpu().numpy(), tools)
        coverage = ObservedSpace(np.ones((81, 81, 81), bool), origin=[-2., -2., -2.], voxel_size=.05,
                                 observation_ids=("explicit synthetic known-space grid",), reference_frame="robot_base")
        half = np.array([.009, .009, .009])

        def scene(center):
            def query(name, pose, dimensions):
                collision = SceneCollision.from_config(SceneCollisionCfg(
                    device_cfg=device, scene_model=SceneCfg(cuboid=[Cuboid(name=name, pose=pose, dims=dimensions)]),
                    cache={"cuboid": 1}))
                return CuroboSphereQuery(collision, owner_ids=(name,), source="fully declared synthetic cuboid")

            background = query("future_obstacle", [*center, 1., 0., 0., 0.], (2*half).tolist())
            body_query = query("branch_body", [0., 0., .005, 1., 0., 0., 0.], [.01, .01, .01])
            owner = CollisionOwner(
                "branch_body", body_query,
                PosePath(np.array([0., 1.]), np.stack((initial_body, initial_body)), "synthetic initial carrier pose"),
                "fully declared small synthetic body", True,
                local_spheres=[[0., 0., .005, np.sqrt(3)*.005]],
                attachments=(Attachment((1., 2.), tool_body[:3, 3], np.array([1., 0., 0., 0.]),
                                        "grasp:pending_actual_holding", None),))
            return PlanningScene(background=background, owners=(owner,), coverage=coverage,
                                 reference_frame="robot_base", robot_sphere_materials=(None,)*kinematics.total_spheres,
                                 robot_geometry_source="supplied Franka spheres with mechanically locked fingers",
                                 has_gripper=False)

        budget = PlanningBudget(ik_seeds=IK_SEEDS, trajectory_seeds=2, knots=knots,
                                 ik_iterations=80, trajectory_iterations=200,
                                 interpolation_buffer_size=1024, maximum_trajectory_solves=2, search_seconds=2.)
        report["budget"] = asdict(budget)
        torch.manual_seed(budget.random_seed)
        baseline = plan_trajectory(problem, scene(np.array([1.6, 1.6, 1.6])), "franka.yml", budget=budget)
        report["baseline"] = {"success": baseline.success, "diagnostics": baseline.diagnostics}
        if not baseline.success:
            report.update(status="inconclusive", failure="default_full_window_baseline_not_verified")
            return report
        curve = baseline.trajectory
        source_progress = np.linspace(1., 2., FUTURE_NODES)
        times = np.array([first_time_at_progress(curve, value) for value in source_progress])
        baseline_q = curve._curve(times)
        baseline_tools, baseline_spheres = fk(baseline_q)
        _, current_spheres = fk(current.cpu().numpy()[None])
        np.savez(output/"baseline.npz", times=curve.times, position=curve.position, velocity=curve.velocity,
                 progress_u=curve.progress_u, progress_s=curve.progress_s)

        # BEGIN evaluate every budgeted native IK branch before returning.
        fixture_configuration = InverseKinematicsCfg.create(
            robot="franka.yml", optimizer_configs=["ik/lbfgs_ik.yml"], device_cfg=device,
            num_seeds=IK_SEEDS, seed_solver_num_seeds=IK_SEEDS, max_batch_size=1, max_goalset=1,
            use_cuda_graph=False, position_tolerance=.0001, orientation_tolerance=.001,
            override_optimizer_num_iters={"lbfgs": 100}, random_seed=budget.random_seed)
        # The native default exits when this one batch has ANY successful
        # seed. Here q0 already reaches the grasp, so that shortcut would skip
        # LBFGS for the other seven candidates whose coverage is being tested.
        fixture_configuration.exit_early = False
        fixture_ik = InverseKinematics(fixture_configuration)
        report["fixture_ik_policy"] = {"exit_early": False, "num_seeds": IK_SEEDS,
                                        "lbfgs_iterations": 100,
                                        "position_tolerance": .0001, "orientation_tolerance": .001,
                                        "initial_seed_counts": {"unmodified_center": 1,
                                                                "local_gaussian": 3, "global_halton": 4},
                                        "local_joint_range_fraction": .05,
                                        "local_center": "measured q0, then preceding native IK continuation"}
        report["fixture_ik_candidates"] = []
        # END evaluate every budgeted native IK branch before returning.
        measured = JointState.from_position(current[None], joint_names=kinematics.joint_names)
        fixture_calls = 0

        def solve_ik(tool_pose, previous=None):
            nonlocal fixture_calls
            if fixture_calls >= MAX_FIXTURE_IK_SOLVES:
                raise RuntimeError("Explicit finite fixture IK-call budget exhausted")
            fixture_calls += 1
            quaternion = Rotation.from_matrix(tool_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            goal = GoalToolPose(tool_frames=["panda_hand"],
                                position=device.to_device(tool_pose[None, None, None, None, :3, 3]).contiguous(),
                                quaternion=device.to_device(quaternion[None, None, None, None]).contiguous())
            # BEGIN bounded local/global initialization through native samplers.
            # The local center is the measured configuration for the grasp,
            # then the preceding native IK result for continuation. No target
            # joint solution or target Jacobian determines any perturbation.
            center = current if previous is None else device.to_device(previous)
            lower = fixture_ik.transition_model.action_bound_lows.reshape(-1)
            upper = fixture_ik.transition_model.action_bound_highs.reshape(-1)
            gaussian = fixture_ik.seed_manager.action_sample_generator.get_gaussian_samples(3, variance=1.)
            local = torch.clamp(center[None]+.05*(upper-lower)*gaussian, min=lower, max=upper)
            global_seeds = fixture_ik.seed_manager.generate_random_actions(1, 4)[0]
            initial_seeds = torch.cat((center[None], local, global_seeds), dim=0)[None].contiguous()
            # Supplying all eight seeds prevents either native stage from
            # appending hidden extra candidates or changing this fixed mix.
            result = fixture_ik.solve_pose(goal, current_state=measured, return_seeds=IK_SEEDS,
                                           seed_config=initial_seeds)
            # END bounded local/global initialization through native samplers.
            values = fixture_ik.get_active_js(result.js_solution).position.detach().cpu().numpy()[0].copy()
            valid = result.success.detach().cpu().numpy()[0].copy()
            # BEGIN retain all candidate diagnostics, including failed seeds.
            prior = current.cpu().numpy() if previous is None else previous
            report["fixture_ik_candidates"].append({
                "call": fixture_calls, "native_success": valid.tolist(),
                "seed_config": initial_seeds.detach().cpu().numpy()[0].tolist(),
                "native_feasible": result.feasible.detach().cpu().numpy()[0].tolist(),
                "native_position_error": result.position_error.detach().cpu().numpy()[0].tolist(),
                "native_rotation_error": result.rotation_error.detach().cpu().numpy()[0].tolist(),
                "joint_distance_from_initializer_rad": np.linalg.norm(values-prior, axis=-1).tolist(),
                "joint_distance_from_baseline_grasp_rad": np.linalg.norm(values-baseline_q[0], axis=-1).tolist(),
                "finite_joint_values": np.isfinite(values).all(-1).tolist(),
            })
            # END retain all candidate diagnostics, including failed seeds.
            return values[valid & np.isfinite(values).all(-1)]

        alternatives = solve_ik(tools[0])
        distances = np.linalg.norm(alternatives-baseline_q[0], axis=-1)
        eligible = np.flatnonzero(distances >= MINIMUM_BRANCH_DISTANCE_RAD)
        eligible = eligible[np.argsort(-distances[eligible])][:MAX_ALTERNATIVE_BRANCHES]
        branches = []
        for index in eligible:
            sequence = [alternatives[index]]
            for target in tools[1:]:
                options = solve_ik(target, sequence[-1])
                if not len(options):
                    break
                sequence.append(options[np.argmin(np.linalg.norm(options-sequence[-1], axis=-1))])
            if len(sequence) == FUTURE_NODES:
                branch_q = np.asarray(sequence)
                branch_tools, branch_spheres = fk(branch_q)
                branches.append((branch_q, branch_tools, branch_spheres))
        fixture_ik.destroy()
        fixture_ik = None
        report["fixture_search"] = {"ik_solve_calls": fixture_calls,
                                    "native_successful_grasp_options": len(alternatives),
                                    "distinct_grasp_options": len(eligible),
                                    "successful_sampled_continuations": len(branches),
                                    "minimum_branch_distance_rad": MINIMUM_BRANCH_DISTANCE_RAD,
                                    "whole_curve_witness_verified": False}
        if not branches:
            report.update(status="inconclusive", failure="bounded_native_ik_did_not_cover_a_distinct_future_branch")
            return report

        forearm = np.concatenate([kinematics.kinematics_config.get_sphere_index_from_link_name(name).cpu().numpy()
                                  for name in ("panda_link3", "panda_link4", "panda_link5")])
        forearm = np.unique(forearm)[:MAX_FOREARM_SPHERES]
        # Include the actual FK carrier poses as well as their common nominal
        # reference: finite IK pose tolerances must not hide a payload contact.
        body_poses = np.stack((tools, baseline_tools, *(branch[1] for branch in branches)))
        body_spheres = np.zeros((*body_poses.shape[:-2], 1, 4))
        body_spheres[..., 0, :3] = body_poses[..., :3, 3]+body_poses[..., :3, :3] @ np.array([0., 0., .225])
        body_spheres[..., 0, 3] = np.sqrt(3)*.005
        # BEGIN independently check each finite, shallow surface candidate.
        choices, candidate_diagnostics = [], []
        maximum_center_separation = 0.
        zero_separation_pairs = 0
        for node in range(1, FUTURE_NODES):
            for sphere_index in forearm:
                if baseline_spheres[node, sphere_index, 3] <= 0:
                    continue
                for branch_index, (_, _, branch_spheres) in enumerate(branches):
                    sphere = baseline_spheres[node, sphere_index]
                    alternative = branch_spheres[node, sphere_index]
                    separation = float(np.linalg.norm(sphere[:3]-alternative[:3]))
                    maximum_center_separation = max(maximum_center_separation, separation)
                    if separation == 0.:
                        zero_separation_pairs += 1
                        continue
                    for penetration in SURFACE_PENETRATIONS_M:
                        center = surface_obstacle_center(sphere, alternative, half, penetration)
                        current_clearance = float(box_clearance(current_spheres, center, half).min())
                        grasp_clearance = float(box_clearance(baseline_spheres[:1], center, half).min())
                        body_clearance = float(box_clearance(body_spheres[[0, 1, branch_index+2]], center, half).min())
                        future_clearance = float(box_clearance(baseline_spheres[1:], center, half).min())
                        alternative_clearance = float(box_clearance(branch_spheres, center, half).min())
                        protected = min(current_clearance, grasp_clearance, body_clearance)
                        shallow = .003 <= -future_clearance <= .006
                        separated = min(protected, alternative_clearance) > REQUIRED_OBSTACLE_CLEARANCE_M
                        candidate = {
                            "node": node, "sphere_index": int(sphere_index), "branch_index": branch_index,
                            "center": center.tolist(), "corresponding_sphere_center_separation_m": separation,
                            "target_sphere_penetration_m": penetration,
                            "actual_sphere_clearance_m": float(box_clearance(sphere, center, half)),
                            "baseline_future_sampled_clearance_m": future_clearance,
                            "measured_q0_clearance_m": current_clearance,
                            "baseline_grasp_clearance_m": grasp_clearance,
                            "common_body_sampled_clearance_m": body_clearance,
                            "alternative_sampled_clearance_m": alternative_clearance,
                            "protected_geometry_clearance_m": protected,
                            "minimum_protected_and_alternative_clearance_m": min(protected, alternative_clearance),
                            "baseline_shallow_penetration_verified": shallow,
                            "all_required_clearances_verified": separated,
                        }
                        candidate_diagnostics.append(candidate)
                        if shallow and separated:
                            choices.append(candidate)
        protected_candidates = [item for item in candidate_diagnostics
                                if item["protected_geometry_clearance_m"] > REQUIRED_OBSTACLE_CLEARANCE_M
                                and item["baseline_shallow_penetration_verified"]]
        report["obstacle_search"] = {
            "scope": "finite surface centers and supplied FK nodes; not a whole-curve witness",
            "maximum_candidate_centers": MAX_CUBOID_CENTERS,
            "evaluated_candidate_centers": len(candidate_diagnostics),
            "zero_separation_pairs": zero_separation_pairs,
            "accepted_candidate_centers": len(choices),
            "required_clearance_m": REQUIRED_OBSTACLE_CLEARANCE_M,
            "payload_checks": "nominal reference, actual baseline FK and selected alternative FK nodes",
            "maximum_corresponding_sphere_center_separation_m": maximum_center_separation,
            "maximum_tested_alternative_clearance_m": max(
                (item["alternative_sampled_clearance_m"] for item in candidate_diagnostics), default=None),
            "maximum_tested_minimum_clearance_m": max(
                (item["minimum_protected_and_alternative_clearance_m"] for item in candidate_diagnostics), default=None),
            "maximum_alternative_clearance_with_shallow_baseline_and_protected_geometry_m": max(
                (item["alternative_sampled_clearance_m"] for item in protected_candidates), default=None),
            "candidates": candidate_diagnostics,
        }
        # END independently check each finite, shallow surface candidate.
        if not choices:
            report.update(status="inconclusive", failure="finite_fixture_search_found_no_future_only_branch_obstacle")
            return report
        chosen = max(choices, key=lambda value: value["alternative_sampled_clearance_m"])
        clearance = chosen["alternative_sampled_clearance_m"]
        node, sphere_index, branch_index = chosen["node"], chosen["sphere_index"], chosen["branch_index"]
        center = np.asarray(chosen["center"])
        branch_q, branch_tools, _ = branches[branch_index]
        report["obstacle"] = {"center": center.tolist(), "dimensions": (2*half).tolist(),
                              "blocked_baseline_progress": float(source_progress[node]),
                              "blocked_forearm_sphere": sphere_index,
                              "measured_q0_clearance_m": float(box_clearance(current_spheres, center, half).min()),
                              "baseline_grasp_clearance_m": float(box_clearance(baseline_spheres[:1], center, half).min()),
                              "common_body_sampled_clearance_m": float(box_clearance(
                                  body_spheres[[0, 1, branch_index+2]], center, half).min()),
                              "baseline_future_clearance_m": float(box_clearance(baseline_spheres, center, half).min()),
                              "alternative_sampled_clearance_m": clearance,
                              "target_sphere_penetration_m": chosen["target_sphere_penetration_m"],
                              "baseline_shallow_penetration_verified": chosen["baseline_shallow_penetration_verified"],
                              "all_required_clearances_verified": chosen["all_required_clearances_verified"]}
        report["alternative_witness"] = {"joint_position": branch_q.tolist(),
                                          "grasp_joint_distance_rad": float(np.linalg.norm(branch_q[0]-baseline_q[0])),
                                          "grasp_eef_error": error(branch_tools[0], tools[0]),
                                          "whole_required_curve_verified": False,
                                          "supplied_as_production_seed": False}
        # A fresh production planner receives exactly the same problem and
        # seed policy. The IK witness above influences only obstacle selection.
        torch.manual_seed(budget.random_seed)
        blocked = plan_trajectory(problem, scene(center), "franka.yml", budget=budget)
        report["with_future_obstacle"] = {"success": blocked.success, "diagnostics": blocked.diagnostics}
        if not blocked.success:
            report.update(status="inconclusive", failure="default_initializers_did_not_produce_a_verified_obstacle_curve")
            return report
        new_curve = blocked.trajectory
        new_q = new_curve._curve(first_time_at_progress(new_curve, 1.))
        new_tool, _ = fk(new_q[None])
        change = float(np.max(np.abs(new_q-baseline_q[0])))
        report["comparison"] = {"baseline_grasp_q": baseline_q[0].tolist(), "obstacle_grasp_q": new_q.tolist(),
                                 "maximum_grasp_joint_change_rad": change,
                                 "minimum_change_for_positive_rad": MINIMUM_GRASP_CHANGE_RAD,
                                 "grasp_eef_change": error(new_tool[0], baseline_tools[0]),
                                 "same_fixed_grasp_requirement": True, "same_attachment": True,
                                 "conditional_executable_until": new_curve.executable_until}
        np.savez(output/"with_future_obstacle.npz", times=new_curve.times, position=new_curve.position,
                 velocity=new_curve.velocity, progress_u=new_curve.progress_u, progress_s=new_curve.progress_s)
        positive = change >= MINIMUM_GRASP_CHANGE_RAD
        report.update(status="completed", success=positive, joint_branch_selection_validation=positive)
        if not positive:
            report["failure"] = "verified_obstacle_curve_did_not_demonstrate_a_changed_grasp_configuration"
        return report
    except Exception as exception:
        report.update(status="error", error=repr(exception), traceback=traceback.format_exc())
        return report
    finally:
        if fixture_ik is not None:
            fixture_ik.destroy()
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
    parser.add_argument("--knots", type=int, choices=(12, 16, 24), default=16)
    args = parser.parse_args()
    result = run(args.output, args.knots)
    raise SystemExit(0 if result["success"] else 1)
# END bounded paired future-constraint integration experiment.
