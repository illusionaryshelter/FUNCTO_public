"""Bounded native-robot test of an on endpoint family and its release frame.

Run each mode in its own process. Both use the same locked-finger Franka,
measured synthetic joint state, rigid candidate relation, finite support patch
and hand-local obstacle. The family may relocate the support endpoint; the
fixed mode requires the original full poses, whose hand sphere is provably
inside that obstacle. All FK and collision queries use installed cuRobo.

This is synthetic, conditional geometry. The small carried body is a declared
rigid proxy, the support is a geometric patch, and release is a conditional
mode change. It does not validate actuated closing, physical support/contact,
observed scene completion, actual Holding or real robot execution.
"""

# BEGIN bounded shared-endpoint and release-frame GPU integration experiment.
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

# BEGIN resolve imports from this FUNCTO checkout, independent of its directory name.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# END project-local imports.


def pose_matrices(position, quaternion):
    """Independent NumPy/SciPy conversion of native wxyz FK values."""
    position, quaternion = np.asarray(position), np.asarray(quaternion)
    result = np.broadcast_to(np.eye(4), (*position.shape[:-1], 4, 4)).copy()
    result[..., :3, 3] = position
    result[..., :3, :3] = Rotation.from_quat(quaternion.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(
        *position.shape[:-1], 3, 3)
    return result


def box_clearance(spheres, center, half):
    """Analytic sphere/cuboid witness independent of the native query kernel."""
    distance = np.abs(spheres[..., :3]-center)-half
    signed = np.linalg.norm(np.maximum(distance, 0.), axis=-1)+np.minimum(distance.max(-1), 0.)
    return np.where(spheres[..., 3] > 0, signed-spheres[..., 3], np.inf)


def first_time_at_progress(curve, value):
    """Find first arrival, including an event at either native end plateau."""
    same = np.flatnonzero(curve.progress_s == value)
    if len(same):
        return float(curve.progress_u[same[0]]*curve.duration)
    right = np.searchsorted(curve.progress_s, value)
    fraction = (value-curve.progress_s[right-1])/(curve.progress_s[right]-curve.progress_s[right-1])
    return float((curve.progress_u[right-1]+fraction*(curve.progress_u[right]-curve.progress_u[right-1]))*curve.duration)


def error(first, second):
    """Physical pose error, independent of optimizer clipping or cost weight."""
    return {"position_m": float(np.linalg.norm(first[:3, 3]-second[:3, 3])),
            "rotation_rad": float(Rotation.from_matrix(first[:3, :3].T@second[:3, :3]).magnitude())}


def make_problem(joint_names, current_q, tools, mode):
    """Declare the same conditional process, changing only endpoint freedom.

    tools contains the native start, nominal midpoint, support goal and
    retreat goal. No IK branch, robot geometry or observed evidence is invented
    here. A finite planar body makes the on relation's tangent/yaw freedom
    explicit; the rest of the required motion follows the endpoint family.
    """
    from manipulation.goals import SupportSurface
    from manipulation.planning_problem import (
        MotionPhase, NumericalTolerance, PlanningEvent, PlanningProblem,
        PosePath, SupportRequirement,
    )

    body_goal = np.eye(4)
    # This isolated proxy keeps a rigid relation to the hand without pretending
    # that a collision permission proves a real finger grasp/contact patch.
    body_goal[:3, 3] = tools[2, :3, 3]+tools[2, :3, :3]@np.array([0., 0., .22])
    attachment = np.linalg.inv(body_goal)@tools[2]
    tool_body = np.linalg.inv(attachment)
    bodies = tools[:3]@tool_body
    source = "declared synthetic on-family integration fixture"
    owner_id, hold_id, release_id = "probe_body", "placement:pending_holding", "placement:actual_release"
    meta = {"skill_id": "placement", "carrier_id": owner_id,
            "owner_initial_pose": bodies[0].tolist(), "mode": "held",
            "attachment": attachment.tolist(), "attachment_interval": [0., 2.],
            "attachment_condition_id": hold_id, "release_condition_id": release_id,
            "conditional": True}
    phases = (
        MotionPhase("on_motion", PosePath(np.array([0., .5, 1.]), bodies, source, attachment=attachment),
                    endpoint_family={"move_start": False, "end": "variable"} if mode == "family" else None,
                    metadata=deepcopy(meta)),
        MotionPhase("held_after_function", PosePath(np.array([0., 1.]), np.stack((bodies[-1], bodies[-1])),
                                                    source, attachment=attachment),
                    endpoint_family={"move_start": True, "end": "unanchored"} if mode == "family" else None,
                    metadata=deepcopy(meta)),
        MotionPhase("released_retreat", PosePath(np.array([0., 1.]), tools[2:], source),
                    frame_anchor={"progress": 2., "reference_pose": tools[2]} if mode == "family" else None,
                    metadata={"skill_id": "placement", "carrier_id": owner_id,
                              "owner_initial_pose": bodies[0].tolist(), "mode": "released_retreat",
                              "release_progress": 2.}),
    )
    body_points = np.array([[x, y, z] for x in (-.005, .005) for y in (-.005, .005) for z in (0., .01)])
    surface = SupportSurface(
        origin=body_goal[:3, 3], normal=np.array([0., 0., 1.]),
        basis=np.array([[1., 0.], [0., 1.], [0., 0.]]),
        boundary=np.array([[-.7, -.7], [.7, -.7], [.7, .7], [-.7, .7]]),
        source="explicit finite synthetic support patch; no unobserved table volume")
    support = SupportRequirement(1., tool_body, body_points, body_points[body_points[:, 2] == 0],
                                 np.array([0., 0., -1.]), surface, .001, .005, 0., source)
    conditions = (
        {"kind": "candidate_attachment", "condition_id": hold_id, "carrier_id": owner_id,
         "progress": 0., "T_carrier_eef": attachment.tolist(),
         "requires_evidence": "actual stable grasp feedback is not supplied by this synthetic fixture"},
        {"kind": "predicted_release", "condition_id": release_id, "carrier_id": owner_id,
         "progress": 2., "T_carrier_eef": attachment.tolist(),
         "requires_evidence": "actual release feedback is not supplied by this synthetic fixture"},
    )
    events = (
        PlanningEvent("holding", "placement", 0., 0, "verification", source,
                      conditions[0]["requires_evidence"], True),
        PlanningEvent("on", "placement", 1., 2, "function", source),
        PlanningEvent("release", "placement", 2., 3, "verification", source,
                      conditions[1]["requires_evidence"], True),
    )
    problem = PlanningProblem(tuple(joint_names), current_q, np.zeros(len(current_q)), "panda_hand",
                              "robot_base", np.eye(4), np.eye(4), phases, events=events,
                              numerical=NumericalTolerance(position_m=.001, rotation_rad=.005),
                              conditions=conditions, supports=(support,))
    return problem, bodies, attachment


def safe_json(value):
    """Retain numerical failure evidence in valid JSON instead of losing it."""
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def run(output, mode, knots):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "mode": mode,
              "scope": "native Franka FK and joint optimization in synthetic conditional geometry",
              "production_environment_validation": False, "moving_gripper_validation": False,
              "joint_branch_selection_validation": False}
    try:
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
        from curobo._src.geom.types import Cuboid, SceneCfg
        from manipulation.motion import transfer_motion
        from manipulation.planning import PlanningBudget, plan_trajectory
        from manipulation.planning_problem import PosePath
        from manipulation.planning_scene import Attachment, CollisionOwner, CuroboSphereQuery, ObservedSpace, PlanningScene

        if not torch.cuda.is_available():
            raise RuntimeError("This bounded integration probe requires explicit GPU execution")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        device = DeviceCfg("cuda:0")
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
        current = kinematics.default_joint_position.reshape(-1).clone()
        nominal, alternative = current.clone(), current.clone()
        nominal[0] += .35
        alternative[0] -= .3
        midpoint = (current+nominal)/2
        nominal_retreat, alternative_retreat = nominal.clone(), alternative.clone()
        nominal_retreat[1] -= .025
        alternative_retreat[1] -= .025
        configurations = torch.stack((current, midpoint, nominal, nominal_retreat,
                                       alternative, alternative_retreat))[None].contiguous()
        state = kinematics.compute_kinematics(JointState.from_position(configurations,
                                                                      joint_names=kinematics.joint_names))
        spheres = state.robot_spheres[0].detach().cpu().numpy().copy()
        all_tools = pose_matrices(state.tool_poses.position[0, :, 0].detach().cpu().numpy(),
                                   state.tool_poses.quaternion[0, :, 0].detach().cpu().numpy())
        hand_indices = kinematics.kinematics_config.get_sphere_index_from_link_name("panda_hand").cpu().numpy()
        half = np.array([.009, .009, .009])
        choices = []
        for index in hand_indices:
            if spheres[2, index, 3] <= 0:
                continue
            direction = spheres[4, index, :3]-spheres[2, index, :3]
            direction /= np.linalg.norm(direction)
            center = spheres[2, index, :3]-.003*direction
            clearance = box_clearance(spheres, center, half)
            witness_clear = float(clearance[[0, 4, 5]].min())
            local_radius = np.linalg.norm(spheres[2, index, :3]-all_tools[2, :3, 3])
            # Full-pose tolerances bound movement of this hand-fixed sphere.
            # A penetration deeper than this cannot be fixed by changing an
            # arm branch while the hand pose remains inside its allowed band.
            bound = np.sqrt(3)*.001+2*np.sin(np.sqrt(3)*.005/2)*local_radius
            penetration = float(-clearance[2, index])
            if witness_clear > .015 and penetration > bound+.01:
                choices.append((witness_clear, int(index), center.copy(), clearance.min(-1), penetration, bound))
        if not choices:
            raise ValueError("The declared synthetic construction has no certified fixed-hand endpoint obstacle")
        _, index, center, clearance, penetration, displacement_bound = max(choices, key=lambda item: item[0])
        report["obstacle"] = {"center": center.tolist(), "dimensions": (2*half).tolist(),
                              "rigid_hand_sphere_index": index, "configuration_clearance_m": clearance.tolist(),
                              "nominal_hand_penetration_m": penetration,
                              "maximum_sphere_shift_under_fixed_pose_tolerances_m": float(displacement_bound),
                              "fixed_pose_obstacle_contradiction": bool(penetration > displacement_bound)}
        problem, bodies, attachment = make_problem(kinematics.joint_names, current.cpu().numpy(), all_tools[:4], mode)
        tool_body = np.linalg.inv(attachment)
        alternate_body = all_tools[4]@tool_body
        p = torch.as_tensor(all_tools[4:5, :3, 3][None], dtype=torch.float64)
        r = torch.as_tensor(Rotation.from_matrix(all_tools[4:5, :3, :3]).as_quat()[:, [3, 0, 1, 2]][None], dtype=torch.float64)
        support_witness = float(problem.supports[0].evaluate(p, r, problem.numerical).max())
        if support_witness != 0:
            raise ValueError("The native alternative endpoint does not satisfy the declared support relation")
        report["alternative_endpoint_witness"] = {
            "joint_position": alternative.cpu().tolist(), "support_residual": support_witness,
            "body_pose": alternate_body.tolist(), "whole_required_curve_verified": False}

        def query(name, pose, dimensions):
            native = SceneCollision.from_config(SceneCollisionCfg(
                device_cfg=device, scene_model=SceneCfg(cuboid=[Cuboid(name=name, pose=pose, dims=dimensions)]),
                cache={"cuboid": 1}))
            return CuroboSphereQuery(native, owner_ids=(name,), source="fully declared synthetic cuboid")

        background = query("endpoint_obstacle", [*center, 1., 0., 0., 0.], (2*half).tolist())
        body_query = query("probe_body", [0., 0., .005, 1., 0., 0., 0.], [.01, .01, .01])
        relation = Attachment((0., 2.), tool_body[:3, 3],
                              Rotation.from_matrix(tool_body[:3, :3]).as_quat()[[3, 0, 1, 2]],
                              "placement:pending_holding", "placement:actual_release")
        owner = CollisionOwner("probe_body", body_query,
                               PosePath(np.array([0., 1.]), np.stack((bodies[0], bodies[0])), "synthetic initial body pose"),
                               "fully declared synthetic cube", True,
                               local_spheres=[[0., 0., .005, np.sqrt(3)*.005]], attachments=(relation,))
        coverage = ObservedSpace(np.ones((81, 81, 81), bool), origin=[-2., -2., -2.], voxel_size=.05,
                                 observation_ids=("explicit synthetic known-space grid",), reference_frame="robot_base")
        scene = PlanningScene(background=background, owners=(owner,), coverage=coverage, reference_frame="robot_base",
                              robot_sphere_materials=(None,)*kinematics.total_spheres,
                              robot_geometry_source="supplied Franka spheres with mechanically locked fingers",
                              has_gripper=False)
        budget = PlanningBudget(ik_seeds=4, trajectory_seeds=2, knots=knots, ik_iterations=80,
                                 trajectory_iterations=200, interpolation_buffer_size=1024 if knots == 24 else 768,
                                 maximum_trajectory_solves=2, search_seconds=2.)
        report["budget"] = asdict(budget)
        result = plan_trajectory(problem, scene, "franka.yml", budget=budget)
        report.update(status="completed", success=result.success, diagnostics=result.diagnostics)
        if result.trajectory is not None:
            curve = result.trajectory
            np.savez(output/"trajectory.npz", times=curve.times, position=curve.position, velocity=curve.velocity,
                     progress_u=curve.progress_u, progress_s=curve.progress_s)
            progress = np.array([0., .5, 1., 1.5, 2., 2.5, 3.])
            times = np.array([first_time_at_progress(curve, value) for value in progress])
            q, _, sampled_progress = curve.sample(times)
            state = kinematics.compute_kinematics(JointState.from_position(
                device.to_device(q[None]).contiguous(), joint_names=list(curve.joint_names)))
            actual = pose_matrices(state.tool_poses.position[0, :, 0].detach().cpu().numpy(),
                                   state.tool_poses.quaternion[0, :, 0].detach().cpu().numpy())
            actual_body = actual@tool_body
            expected_bodies = transfer_motion(bodies, bodies[0], goal_pose=actual_body[2])["poses"]
            expected_middle = expected_bodies[1]@attachment
            release_delta = actual[4]@np.linalg.inv(all_tools[2])
            expected_retreat = release_delta@all_tools[3]
            report["independent_curve_checks"] = {
                "sampled_progress": sampled_progress.tolist(),
                "endpoint_tangent_displacement_m": float(np.linalg.norm(actual_body[2, :2, 3]-bodies[-1, :2, 3])),
                "endpoint_yaw_change_rad": float(Rotation.from_matrix(bodies[-1, :3, :3].T@actual_body[2, :3, :3]).as_rotvec()[2]),
                "required_midpoint_vs_moved_family": error(actual[1], expected_middle),
                "required_midpoint_vs_nominal": error(actual[1], all_tools[1]),
                "release_vs_function_anchor": error(actual[4], actual[2]),
                "retreat_end_vs_actual_release_transform": error(actual[6], expected_retreat),
                "retreat_end_vs_nominal": error(actual[6], all_tools[3]),
                "executable_until": curve.executable_until,
                "all_future_attachment_evidence_supplied": False,
            }
            # These are probe assertions, independent of the production success
            # flag. A success without meaningful endpoint movement does not
            # establish that the intended freedom was actually exercised.
            checks = report["independent_curve_checks"]
            if mode == "family" and (checks["endpoint_tangent_displacement_m"] <= .005
                                     or checks["required_midpoint_vs_moved_family"]["position_m"] > .002
                                     or checks["retreat_end_vs_actual_release_transform"]["position_m"] > .002):
                raise AssertionError("The accepted native curve did not demonstrate the required shared endpoint freedom")
            if mode == "fixed":
                raise AssertionError("A fixed-pose result contradicted the certified rigid-hand obstacle overlap")
        report.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exception:
        report.update(status="error", error=f"{type(exception).__name__}: {exception}")
        (output/"exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report = safe_json(report)
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("family", "fixed"), default="family")
    parser.add_argument("--knots", type=int, choices=(8, 12, 16, 24), default=12)
    arguments = parser.parse_args()
    run(arguments.output, arguments.mode, arguments.knots)
# END bounded shared-endpoint and release-frame GPU integration experiment.
