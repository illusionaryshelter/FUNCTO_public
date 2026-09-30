"""Bounded paired UR10e test of a future obstacle changing physical grasp IK.

The audited native six-joint model, current state, grasp pose, conditional
attachment and future pose requirements are shared by two production calls.
Only one scene cuboid moves. Independent native IK results select its location;
none of those joint values enter either production initializer.

The payload is an explicitly synthetic rigid body, not supplied UR10e hardware.
Native full-curve acceptance establishes the returned motion from q0. Discrete
alternative FK nodes and finite Jacobian checks do not certify a whole branch
or prove that every admissible motion must switch branches.
"""

# BEGIN bounded six-DOF paired future-obstacle integration evidence.
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import resource
import time
import traceback
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from probe_future_branch_coupling import surface_obstacle_center
from probe_on_endpoint_family import box_clearance, error, first_time_at_progress, pose_matrices, safe_json
from probe_ur10e_branch_ik import (
    IK_ITERATIONS, IK_SEEDS, MINIMUM_JACOBIAN_SINGULAR_VALUE,
    MINIMUM_USEFUL_SPHERE_SEPARATION_M, ROBOT,
)


FUTURE_NODES = 4
MAX_ALTERNATIVE_BRANCHES = 2
MAX_FIXTURE_IK_CALLS = MAX_ALTERNATIVE_BRANCHES*(FUTURE_NODES-1)
MAX_ARM_SPHERES = 32
MAX_CUBOID_CENTERS = (FUTURE_NODES-1)*MAX_ARM_SPHERES*MAX_ALTERNATIVE_BRANCHES
MAX_SCENE_AUDIT_POINTS = 1028
SURFACE_PENETRATION_M = .0055
REQUIRED_CLEARANCE_M = .012
MINIMUM_GRASP_CHANGE_RAD = .05


def make_problem(joint_names, current_q, tools, tool_link, base_link):
    """Keep the grasp fixed while leaving its joint configuration variable."""
    from manipulation.planning_problem import (
        MotionPhase, NumericalTolerance, PlanningEvent, PlanningProblem, PosePath,
    )

    tool_body = np.eye(4)
    tool_body[:3, 3] = [0., 0., .22]
    attachment = np.linalg.inv(tool_body)
    bodies = tools@tool_body
    source = "declared synthetic UR10e common grasp and future carry reference"
    hold_id = "grasp:pending_actual_holding"
    metadata = {"skill_id": "grasp", "carrier_id": "branch_body",
                "owner_initial_pose": bodies[0].tolist()}
    approach = MotionPhase(
        "free_before_grasp", PosePath(np.array([0., 1.]), np.stack((tools[0], tools[0])), source),
        kind="free", metadata={**deepcopy(metadata), "mode": "free_pregrasp"})
    carry = MotionPhase(
        "necessary_future_carry", PosePath(np.linspace(0., 1., FUTURE_NODES), bodies, source,
                                          attachment=attachment),
        metadata={**deepcopy(metadata), "skill_id": "carry", "mode": "held",
                  "attachment": attachment.tolist(), "attachment_interval": [1., 2.],
                  "attachment_condition_id": hold_id, "release_condition_id": None, "conditional": True})
    evidence = "actual stable grasp feedback is not supplied by this synthetic fixture"
    problem = PlanningProblem(
        tuple(joint_names), current_q, np.zeros(len(current_q)), tool_link, base_link,
        np.eye(4), np.eye(4), (approach, carry),
        events=(PlanningEvent("verify_holding", "grasp", 1., 0, "verification", source, evidence, True),),
        numerical=NumericalTolerance(position_m=.001, rotation_rad=.005),
        conditions=({"kind": "candidate_attachment", "condition_id": hold_id, "carrier_id": "branch_body",
                     "progress": 1., "T_carrier_eef": attachment.tolist(), "requires_evidence": evidence},))
    return problem, bodies[0], tool_body


def save_curve(path, curve):
    """Preserve the actual production output without a witness-based rewrite."""
    np.savez(path, times=curve.times, position=curve.position, velocity=curve.velocity,
             progress_u=curve.progress_u, progress_s=curve.progress_s)


def run(output, audit):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {
        "status": "running", "success": False,
        "scope": "synthetic native UR10e paired complete-window planning",
        "physical_grasp_validation": False, "moving_gripper_validation": False,
        "production_environment_validation": False, "joint_branch_selection_validation": False,
        "witness_joint_injection": False, "joint_wrapping_or_limit_changes": False,
        "source_ik_audit": str(audit),
        "fixture_limits": {"ik_seeds": IK_SEEDS, "ik_iterations": IK_ITERATIONS,
                           "source_audit_ik_calls": 1, "additional_fixture_ik_calls": MAX_FIXTURE_IK_CALLS,
                           "alternative_branches": MAX_ALTERNATIVE_BRANCHES, "future_reference_nodes": FUTURE_NODES,
                           "arm_spheres": MAX_ARM_SPHERES, "cuboid_centers": MAX_CUBOID_CENTERS,
                           "scene_audit_points": MAX_SCENE_AUDIT_POINTS,
                           "surface_penetration_m": SURFACE_PENETRATION_M,
                           "complete_planner_calls": 2, "trajectory_seeds": 2,
                           "torch_memory_cap_bytes": 2*1024**3},
        "actual_complete_planner_calls": 0, "actual_fixture_ik_calls": 0,
        "initial_seed_geometry": {}, "fixture_ik_candidates": [],
    }
    fixture_ik = None
    try:
        source_report = json.loads((audit/"report.json").read_text())
        if (not source_report["success"] or source_report["robot"] != ROBOT
                or source_report["limits"]["ik_seeds"] != IK_SEEDS
                or source_report["limits"]["lbfgs_iterations"] != IK_ITERATIONS):
            raise ValueError("A successful audit of this unchanged native model and IK budget is required")
        with np.load(audit/"native_ik_evidence.npz") as source:
            audited_current = source["current_q"].copy()
            audited_tools = source["reference_tools"].copy()
            audited_candidates = source["candidate_q"].copy()
            alternative_indices = source["selected_alternative_indices"].copy()
        if (audited_current.shape != (6,) or audited_candidates.shape != (IK_SEEDS, 6)
                or not 1 <= len(alternative_indices) <= MAX_ALTERNATIVE_BRANCHES
                or alternative_indices.tolist() != source_report["native_ik"]["selected_alternative_indices"]):
            raise ValueError("Source audit arrays do not match its finite six-joint branch evidence")

        import torch
        from curobo.content import get_robot_configs_path
        from curobo._src.util.config_io import join_path, load_yaml
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, GoalToolPose, JointState
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
        from curobo._src.geom.types import Cuboid, SceneCfg
        import manipulation.planning as planning
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
        configuration = load_yaml(join_path(get_robot_configs_path(), ROBOT))["robot_cfg"]["kinematics"]
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(ROBOT), compute_jacobian=True)
        current = kinematics.default_joint_position.reshape(-1).clone()
        current_q = current.detach().cpu().numpy().copy()
        tool_link, base_link = kinematics.tool_frames[0], configuration["base_link"]
        if (not np.array_equal(current_q, audited_current) or len(kinematics.tool_frames) != 1
                or tool_link != source_report["model"]["tool_link"]
                or base_link != source_report["model"]["base_link"]
                or list(kinematics.joint_names) != source_report["model"]["joint_names"]):
            raise ValueError("The installed robot/current/tool layout changed since the native audit")

        def fk(values):
            q = device.to_device(np.asarray(values))[None].contiguous()
            state = kinematics.compute_kinematics(JointState.from_position(q, joint_names=kinematics.joint_names))
            tools = pose_matrices(state.tool_poses.position[0, :, 0].detach().cpu().numpy().copy(),
                                  state.tool_poses.quaternion[0, :, 0].detach().cpu().numpy().copy())
            return (tools, state.robot_spheres[0].detach().cpu().numpy().copy(),
                    state.tool_jacobians[0, :, 0].detach().cpu().numpy().copy())

        reference_q = np.broadcast_to(current_q, (FUTURE_NODES, 6)).copy()
        reference_q[:, 0] += np.linspace(0., .45, FUTURE_NODES)
        tools, reference_spheres, reference_jacobians = fk(reference_q)
        if not np.allclose(tools, audited_tools, atol=1e-6, rtol=0):
            raise ValueError("Actual native future FK differs from the staged IK audit")
        active = reference_spheres[0, :, 3] > 0
        problem, initial_body, tool_body = make_problem(
            kinematics.joint_names, current_q, tools, tool_link, base_link)
        report["model"] = {**source_report["model"],
                           "reference_joint_values_are_planner_seeds": False}
        report["payload"] = {"source": "explicit conditional synthetic body; not native robot payload geometry",
                             "tool_to_body": tool_body.tolist(), "local_box_center": [0., 0., .005],
                             "box_dimensions_m": [.01, .01, .01],
                             "requires_actual_holding_evidence": True}
        report["reference_jacobian_singular_values"] = np.linalg.svd(reference_jacobians, compute_uv=False).tolist()
        coverage = ObservedSpace(
            np.ones((81, 81, 81), bool), origin=[-2., -2., -2.], voxel_size=.05,
            observation_ids=("explicit synthetic known-space grid",), reference_frame=base_link)
        half = np.full(3, .009)

        def make_scene(center):
            def query(name, pose, dimensions):
                collision = SceneCollision.from_config(SceneCollisionCfg(
                    device_cfg=device, scene_model=SceneCfg(cuboid=[Cuboid(name=name, pose=pose, dims=dimensions)]),
                    cache={"cuboid": 1}))
                return CuroboSphereQuery(collision, owner_ids=(name,), source="fully declared synthetic cuboid")

            background = query("future_obstacle", [*center, 1., 0., 0., 0.], (2*half).tolist())
            owner = CollisionOwner(
                "branch_body", query("branch_body", [0., 0., .005, 1., 0., 0., 0.], [.01, .01, .01]),
                PosePath(np.array([0., 1.]), np.stack((initial_body, initial_body)), "synthetic initial carrier pose"),
                "fully declared small synthetic body", True,
                local_spheres=[[0., 0., .005, np.sqrt(3)*.005]],
                attachments=(Attachment((1., 2.), tool_body[:3, 3], np.array([1., 0., 0., 0.]),
                                        "grasp:pending_actual_holding", None),))
            return PlanningScene(
                background=background, owners=(owner,), coverage=coverage, reference_frame=base_link,
                robot_sphere_materials=(None,)*kinematics.total_spheres,
                robot_geometry_source="unchanged native UR10e collision spheres; no configured moving gripper",
                has_gripper=False)

        # BEGIN budget the unchanged native boundary initialization and search.
        # The whole search deadline also covers four sequential 8-seed,
        # 200-iteration boundary IK calls before any OMPL branch is entered.
        budget = planning.PlanningBudget(
            ik_seeds=IK_SEEDS, trajectory_seeds=2, knots=24, ik_iterations=IK_ITERATIONS,
            trajectory_iterations=200, interpolation_buffer_size=1024,
            maximum_trajectory_solves=2, search_seconds=15.)
        # END budget the unchanged native boundary initialization and search.
        report["budget"] = asdict(budget)
        report["knots_reason"] = "24 native knots retain the original raw winding branches and provide bounded approach time"
        active_stage = None
        original_planner = planning._CuroboPlanner

        class ObservedPlanner(original_planner):
            # BEGIN observe actual native boundary initialization duration.
            def _boundary_candidates(self, *args, **kwargs):
                boundary_started = time.monotonic()
                completed, candidates = False, None
                try:
                    candidates = super()._boundary_candidates(*args, **kwargs)
                    completed = True
                    return candidates
                finally:
                    report.setdefault("boundary_initialization", {})[active_stage] = {
                        "elapsed_seconds": time.monotonic()-boundary_started,
                        "returned_normally": completed,
                        "returned_none": candidates is None if completed else None,
                        "required_boundary_ik_calls": len(self.anchors)-1,
                        "anchors": self.anchors.tolist(),
                    }
            # END observe actual native boundary initialization duration.

            def reference_seeds(self):
                # This observer sees free control configurations, not an
                # accepted curve. Return the original tensor and counts.
                result = super().reference_seeds()
                seed_q = result[0][0].detach().cpu().numpy().copy()
                if seed_q.shape != (2, self.trajopt.action_horizon, 6):
                    raise ValueError("Observed production seeds changed their declared bounded shape")
                _, seed_spheres, _ = fk(seed_q.reshape(-1, 6))
                seed_spheres = seed_spheres.reshape(2, seed_q.shape[1], kinematics.total_spheres, 4)
                u = (np.arange(seed_q.shape[1])+3.)/(seed_q.shape[1]+4.)
                report["initial_seed_geometry"][active_stage] = {
                    "scope": "actual returned native free controls; not an accepted grasp or curve",
                    "joint_position": seed_q.tolist(), "control_u": u.tolist(),
                    "control_progress": np.interp(u, self.progress_map.u, self.progress_map.s).tolist(),
                    "strategies": list(self.initial_seed_strategies),
                    "native_ik_success_counts": list(result[1]),
                    "native_ik_distinct_counts": list(self.reference_ik_distinct_counts),
                    "maximum_active_sphere_displacement_from_q0_m": np.linalg.norm(
                        seed_spheres[:, :, active, :3]-reference_spheres[0, active, :3], axis=-1).max(-1).tolist(),
                    "pairwise_maximum_active_sphere_displacement_m": np.linalg.norm(
                        seed_spheres[0][:, active, :3]-seed_spheres[1][:, active, :3], axis=-1).max(-1).tolist(),
                    "pairwise_maximum_raw_joint_delta_rad": np.abs(seed_q[0]-seed_q[1]).max(-1).tolist(),
                }
                return result

        def solve_production(label, scene):
            nonlocal active_stage
            if report["actual_complete_planner_calls"] >= 2:
                raise RuntimeError("The two-call complete-planning budget is exhausted")
            report["actual_complete_planner_calls"] += 1
            active_stage = label
            torch.manual_seed(budget.random_seed)
            with patch.object(planning, "_CuroboPlanner", ObservedPlanner):
                return planning.plan_trajectory(problem, scene, ROBOT, budget=budget)

        baseline_center = np.array([1.6, 1.6, 1.6])
        baseline = solve_production("baseline", make_scene(baseline_center))
        report["baseline"] = {"success": baseline.success, "diagnostics": baseline.diagnostics}
        if not baseline.success:
            report.update(status="inconclusive", failure="default_full_window_baseline_not_verified")
            return report
        curve = baseline.trajectory
        save_curve(output/"baseline.npz", curve)
        source_progress = np.linspace(1., 2., FUTURE_NODES)
        source_times = np.array([first_time_at_progress(curve, value) for value in source_progress])
        baseline_q = curve._curve(source_times)
        baseline_tools, baseline_spheres, baseline_jacobians = fk(baseline_q)
        all_times = np.unique(np.concatenate((curve.times, source_times)))
        if len(all_times) > MAX_SCENE_AUDIT_POINTS:
            raise ValueError("The independent baseline scene audit exceeded its declared point budget")
        all_progress = np.interp(all_times/curve.duration, curve.progress_u, curve.progress_s)
        all_q = curve._curve(all_times)
        all_tools, all_spheres, _ = fk(all_q)
        report["baseline"]["grasp_joint_position"] = baseline_q[0].tolist()
        report["baseline"]["grasp_error"] = error(baseline_tools[0], tools[0])
        report["baseline"]["future_node_jacobian_singular_values"] = np.linalg.svd(
            baseline_jacobians, compute_uv=False).tolist()

        # Continue each audited physical branch with one unmodified previous
        # state plus seven native seeds. This is independent fixture evidence.
        ik_configuration = InverseKinematicsCfg.create(
            robot=ROBOT, optimizer_configs=["ik/lbfgs_ik.yml"], device_cfg=device,
            num_seeds=IK_SEEDS, seed_solver_num_seeds=IK_SEEDS, max_batch_size=1, max_goalset=1,
            use_cuda_graph=False, position_tolerance=.0001, orientation_tolerance=.001,
            override_optimizer_num_iters={"lbfgs": IK_ITERATIONS}, random_seed=budget.random_seed)
        ik_configuration.exit_early = False
        fixture_ik = InverseKinematics(ik_configuration)
        measured = JointState.from_position(current[None], joint_names=kinematics.joint_names)

        def solve_continuation(target, previous, branch_index, node):
            if report["actual_fixture_ik_calls"] >= MAX_FIXTURE_IK_CALLS:
                raise RuntimeError("The finite independent IK-continuation budget is exhausted")
            report["actual_fixture_ik_calls"] += 1
            quaternion = Rotation.from_matrix(target[:3, :3]).as_quat()[[3, 0, 1, 2]]
            goal = GoalToolPose(tool_frames=[tool_link],
                               position=device.to_device(target[None, None, None, None, :3, 3]).contiguous(),
                               quaternion=device.to_device(quaternion[None, None, None, None]).contiguous())
            generated_batches = []
            original = fixture_ik.seed_ik_solver._generate_seed_configs

            def observe(*args, **kwargs):
                generated = original(*args, **kwargs)
                generated_batches.append(generated.detach().cpu().numpy().copy())
                return generated

            with patch.object(fixture_ik.seed_ik_solver, "_generate_seed_configs", new=observe):
                result = fixture_ik.solve_pose(
                    goal, current_state=measured, seed_config=device.to_device(previous[None, None]).contiguous(),
                    return_seeds=IK_SEEDS)
            if len(generated_batches) != 1 or generated_batches[0].shape != (1, IK_SEEDS, 6):
                raise ValueError("Independent continuation changed its native eight-seed call shape")
            values = fixture_ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy().copy()
            success = result.success[0].detach().cpu().numpy().copy()
            if not np.isfinite(values).all():
                raise ValueError("Independent continuation returned nonfinite native joint values")
            candidate_tools, _, jacobians = fk(values)
            errors = [error(tool, target) for tool in candidate_tools]
            singular_values = np.linalg.svd(jacobians, compute_uv=False)
            valid = success & np.array([value["position_m"] <= .0001 and value["rotation_rad"] <= .001
                                        for value in errors])
            valid &= singular_values[:, -1] > MINIMUM_JACOBIAN_SINGULAR_VALUE
            eligible = np.flatnonzero(valid)
            selected = int(eligible[np.argmin(np.linalg.norm(values[eligible]-previous, axis=-1))]) if len(eligible) else None
            report["fixture_ik_candidates"].append({
                "call": report["actual_fixture_ik_calls"], "source_branch_index": int(branch_index), "future_node": node,
                "seed_config": generated_batches[0][0].tolist(), "provided_previous_q": previous.tolist(),
                "native_sampler_seed": fixture_ik.seed_ik_solver.config.sampler_seed,
                "native_success": success.tolist(), "native_feasible": result.feasible[0].detach().cpu().numpy().tolist(),
                "native_position_error": result.position_error[0].detach().cpu().numpy().tolist(),
                "native_rotation_error": result.rotation_error[0].detach().cpu().numpy().tolist(),
                "joint_position": values.tolist(), "actual_pose_errors": errors,
                "jacobian_singular_values": singular_values.tolist(), "selected_candidate_index": selected,
            })
            return None if selected is None else values[selected]

        branches, continuation_reports = [], []
        for index in alternative_indices:
            sequence = [audited_candidates[index].copy()]
            for node, target in enumerate(tools[1:], start=1):
                selected = solve_continuation(target, sequence[-1], index, node)
                if selected is None:
                    break
                sequence.append(selected)
            branch_q = np.asarray(sequence)
            branch_tools, branch_spheres, branch_jacobians = fk(branch_q)
            continuation_reports.append({
                "source_audit_index": int(index), "completed_nodes": len(sequence),
                "joint_position": branch_q.tolist(),
                "maximum_neighbor_raw_joint_delta_rad": float(np.abs(np.diff(branch_q, axis=0)).max()) if len(sequence) > 1 else None,
                "jacobian_singular_values": np.linalg.svd(branch_jacobians, compute_uv=False).tolist(),
                "whole_curve_witness_verified": False, "supplied_as_production_seed": False,
            })
            if len(sequence) == FUTURE_NODES:
                branches.append((branch_q, branch_tools, branch_spheres))
        fixture_ik.destroy()
        fixture_ik = None
        report["fixture_continuations"] = continuation_reports
        report["successful_sampled_continuations"] = len(branches)
        if not branches:
            report.update(status="inconclusive", failure="bounded_native_ik_did_not_cover_a_complete_sampled_alternative")
            return report

        arm = np.unique(np.concatenate([
            kinematics.kinematics_config.get_sphere_index_from_link_name(name).cpu().numpy()
            for name in ("upper_arm_link", "forearm_link")]))
        if len(arm) > MAX_ARM_SPHERES:
            raise ValueError("Native arm geometry exceeded the declared fixture sphere budget")
        choices, candidate_reports = [], []
        for branch_index, (_, branch_tools, branch_spheres) in enumerate(branches):
            # Before grasp the payload stays at its declared observed pose.
            # After grasp use actual baseline FK and actual alternative FK.
            body_poses = np.concatenate((tools, all_tools[all_progress >= 1.], branch_tools))
            body_spheres = np.empty((len(body_poses), 4))
            body_spheres[:, :3] = body_poses[:, :3, 3]+body_poses[:, :3, :3]@np.array([0., 0., .225])
            body_spheres[:, 3] = np.sqrt(3)*.005
            for node in range(1, FUTURE_NODES):
                for sphere_index in arm:
                    sphere = baseline_spheres[node, sphere_index]
                    if sphere[3] <= 0:
                        continue
                    center = surface_obstacle_center(
                        sphere, branch_spheres[node, sphere_index], half, SURFACE_PENETRATION_M)
                    if center is None:
                        continue
                    approach_clearance = float(box_clearance(all_spheres[all_progress <= 1.], center, half).min())
                    grasp_clearance = float(box_clearance(baseline_spheres[:1], center, half).min())
                    body_clearance = float(box_clearance(body_spheres, center, half).min())
                    alternative_clearance = float(box_clearance(branch_spheres, center, half).min())
                    future_clearance = float(box_clearance(baseline_spheres[1:], center, half).min())
                    shallow = .003 <= -future_clearance <= .006
                    protected = min(approach_clearance, grasp_clearance, body_clearance, alternative_clearance)
                    candidate = {
                        "node": node, "sphere_index": int(sphere_index), "branch_index": branch_index,
                        "center": center.tolist(), "actual_selected_sphere_clearance_m": float(box_clearance(sphere, center, half)),
                        "baseline_future_node_clearance_m": future_clearance,
                        "baseline_approach_sampled_clearance_m": approach_clearance,
                        "baseline_grasp_clearance_m": grasp_clearance, "payload_sampled_clearance_m": body_clearance,
                        "alternative_sampled_clearance_m": alternative_clearance,
                        "minimum_protected_clearance_m": protected,
                        "shallow_baseline_penetration": shallow,
                        "required_clearances_pass": protected > REQUIRED_CLEARANCE_M,
                    }
                    candidate_reports.append(candidate)
                    if shallow and protected > REQUIRED_CLEARANCE_M:
                        choices.append(candidate)
        report["obstacle_search"] = {
            "scope": "finite analytic cuboid centers; sampled alternatives are not whole-curve witnesses",
            "arm_links": ["upper_arm_link", "forearm_link"], "arm_spheres": arm.tolist(),
            "evaluated_centers": len(candidate_reports), "accepted_centers": len(choices),
            "maximum_centers": MAX_CUBOID_CENTERS, "required_clearance_m": REQUIRED_CLEARANCE_M,
            "candidate_selection": "earliest future reference node, then largest sampled alternative clearance",
            "candidates": candidate_reports,
        }
        if len(candidate_reports) > MAX_CUBOID_CENTERS:
            raise ValueError("Finite obstacle construction exceeded its declared center budget")
        if not choices:
            report.update(status="inconclusive", failure="finite_search_found_no_future_only_branch_obstacle")
            return report
        chosen = min(choices, key=lambda value: (value["node"], -value["alternative_sampled_clearance_m"]))
        center = np.asarray(chosen["center"])
        report["obstacle"] = {
            **chosen, "dimensions_m": (2*half).tolist(), "target_penetration_m": SURFACE_PENETRATION_M,
            "blocked_baseline_progress": float(source_progress[chosen["node"]]),
            "baseline_obstacle_center": baseline_center.tolist(),
        }

        # The original scene's native collision query must independently
        # reject the old future curve. No clearance threshold is relaxed.
        obstacle_scene = make_scene(center)
        audit_scene = obstacle_scene.clone_for_rollout()
        quaternions = Rotation.from_matrix(all_tools[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        components = audit_scene.components(
            device.to_device(all_spheres[None]), device.to_device(all_tools[None, :, :3, 3]),
            device.to_device(quaternions[None]), device.to_device(all_progress[None]),
            device.to_device(all_q[None]), tuple(kinematics.joint_names))
        component_arrays = {name: value.detach().cpu().numpy()[0] for name, value in components.items()}
        old_future_violation = float(component_arrays["collision/robot/background"][all_progress > 1.].max())
        old_approach_violation = float(component_arrays["collision/robot/background"][all_progress <= 1.].max())
        report["old_curve_in_new_scene"] = {
            "points": len(all_times), "original_scene_components_max": {
                name: float(value.max()) for name, value in component_arrays.items()},
            "future_robot_background_violation": old_future_violation,
            "approach_robot_background_violation": old_approach_violation,
            "original_scene_metric_rejects_old_future": old_future_violation > 0,
        }
        if old_future_violation <= 0 or old_approach_violation > 0:
            report.update(status="inconclusive", failure="original_scene_metric_did_not_isolate_the_old_future_collision")
            return report

        blocked = solve_production("with_future_obstacle", obstacle_scene)
        report["with_future_obstacle"] = {"success": blocked.success, "diagnostics": blocked.diagnostics}
        if not blocked.success:
            report.update(status="inconclusive", failure="default_initializers_did_not_produce_a_verified_obstacle_curve")
            return report
        new_curve = blocked.trajectory
        save_curve(output/"with_future_obstacle.npz", new_curve)
        new_q = new_curve._curve(first_time_at_progress(new_curve, 1.))
        new_tools, new_spheres, new_jacobians = fk(new_q[None])
        delta = new_q-baseline_q[0]
        # Periodic differences are diagnostic only; no seed, native bound or
        # commanded position is mapped to another winding representation.
        wrapped_delta = np.arctan2(np.sin(delta), np.cos(delta))
        raw_change = float(np.abs(delta).max())
        wrapped_change = float(np.abs(wrapped_delta).max())
        geometry_change = float(np.linalg.norm(
            new_spheres[0, active, :3]-baseline_spheres[0, active, :3], axis=-1).max())
        positive = (raw_change >= MINIMUM_GRASP_CHANGE_RAD and wrapped_change >= MINIMUM_GRASP_CHANGE_RAD
                    and geometry_change >= MINIMUM_USEFUL_SPHERE_SEPARATION_M)
        report["comparison"] = {
            "baseline_grasp_q": baseline_q[0].tolist(), "obstacle_grasp_q": new_q.tolist(),
            "maximum_raw_grasp_joint_change_rad": raw_change, "maximum_wrapped_grasp_joint_change_rad": wrapped_change,
            "maximum_active_sphere_grasp_displacement_m": geometry_change,
            "minimum_wrapped_joint_change_for_positive_rad": MINIMUM_GRASP_CHANGE_RAD,
            "minimum_sphere_change_for_positive_m": MINIMUM_USEFUL_SPHERE_SEPARATION_M,
            "grasp_eef_change": error(new_tools[0], baseline_tools[0]), "obstacle_grasp_error": error(new_tools[0], tools[0]),
            "obstacle_grasp_jacobian_singular_values": np.linalg.svd(new_jacobians, compute_uv=False).tolist(),
            "same_fixed_grasp_requirement": True, "same_attachment": True, "shared_problem_instance": True,
            "only_scene_difference": "location of the same 18 mm cuboid",
            "both_complete_curves_verified_from_original_q0": True,
            "conditional_executable_until": new_curve.executable_until,
        }
        report.update(status="completed", success=positive, joint_branch_selection_validation=positive)
        if not positive:
            report["failure"] = "verified_future_obstacle_curve_did_not_change_the_physical_grasp_branch"
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
    parser.add_argument("--audit", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.audit)
    raise SystemExit(0 if result["success"] else 1)
# END bounded six-DOF paired future-obstacle integration evidence.
