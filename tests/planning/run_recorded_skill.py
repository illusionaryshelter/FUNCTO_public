"""Test one observed skill with current joint planning and explicit robot data.

Every episode, candidate, camera and robot choice comes from the CLI inputs.
The selected historical result is not read: the compiled source program and
its observed geometric binding are rebuilt by today's planning interface.
The RGB-D field remains partial and all current unknown-space checks remain
enabled. Physics execution consumes the saved curve in a separate process.
"""

# BEGIN observed skill integration using existing production scene contracts.
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


# BEGIN one actual-finger interval contract shared by planning and execution.
def event_finger_bounds(execution_limits, events):
    """Return retiming-independent actual bounds at original progress events.

    At an event progress both pre-command and post-command states can occur,
    so the query covers their union. The post-event interval must include the
    whole physical closing transition; position targets are never substituted
    for this measured-state contract. Bounds persist across progress plateaus.
    """
    import torch

    rows = {(row["skill_id"], row["event_id"]): row for row in execution_limits.events}
    commands = sorted((event for event in events if event.kind in {"open", "close", "hold"}),
                      key=lambda event: event.progress)
    identities = [(event.skill_id, event.event_id) for event in commands]
    if len(set(identities)) != len(identities) or set(identities) != set(rows):
        raise ValueError("Actual finger bounds must match the original planner command events exactly")
    initial = np.asarray(execution_limits.initial_fingers["position_bounds"])
    schedule = [(float(event.progress), np.asarray(rows[identity]["position_bounds"]))
                for event, identity in zip(commands, identities)]

    def bounds(progress):
        shape = (*progress.shape, len(execution_limits.finger_joint_names))
        lower = torch.as_tensor(initial[:, 0], dtype=progress.dtype, device=progress.device).expand(shape)
        upper = torch.as_tensor(initial[:, 1], dtype=progress.dtype, device=progress.device).expand(shape)
        for boundary, interval in schedule:
            lo = torch.as_tensor(interval[:, 0], dtype=progress.dtype, device=progress.device)
            hi = torch.as_tensor(interval[:, 1], dtype=progress.dtype, device=progress.device)
            exact = (progress == boundary)[..., None]
            after = (progress > boundary)[..., None]
            lower = torch.where(exact, torch.minimum(lower, lo), torch.where(after, lo, lower))
            upper = torch.where(exact, torch.maximum(upper, hi), torch.where(after, hi, upper))
        return lower, upper

    return bounds


def position_controlled_hand(robot, feedback, problem, *, device, execution_limits, velocity_limits):
    """Calibrate the affine hand from the supplied native robot, not a name.

    Native axes and sphere ordering are measured by FK. Independent position
    actuators have no invented equal-width mechanical coupling. Native FK
    locks fingers at measured q; the scene replaces their proxies with the
    actual intervals supervised by the same execution-limit input.
    """
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    from manipulation.planning_scene import PrismaticGripper

    unlocked = deepcopy(robot)
    unlocked["robot_cfg"]["kinematics"]["lock_joints"] = None
    native = Kinematics(KinematicsCfg.from_robot_yaml_file(unlocked, device_cfg=device))
    names = tuple(native.joint_names)
    measured = dict(zip(feedback["joint_names"], feedback["start_q"]))
    finger_names = tuple(feedback["finger_joint_names"])
    columns = [names.index(name) for name in finger_names]
    if len(columns) != 2:
        raise ValueError("This recorded parallel-jaw test requires two explicitly bound native fingers")
    parameters = native.config.kinematics_config
    indices = tuple(int(i) for link in feedback["finger_link_names"]
                    for i in parameters.get_sphere_index_from_link_name(link).cpu().tolist())
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("The actual native finger links must have disjoint nonempty sphere sets")
    limits = parameters.joint_limits.position[:, columns].T.cpu().numpy().copy()
    # Bind all ranges by identity before asking a planner to consume a narrower
    # actual hand interval. No numerical solver tolerance becomes a physical
    # tracking allowance, and no allowed interval is inferred from ctrl.
    full_columns = [names.index(name) for name in feedback["joint_names"]]
    binding = execution_limits.bind(
        SimpleNamespace(joint_names=problem.joint_names, events=[asdict(event) for event in problem.events]),
        joint_names=feedback["joint_names"], arm_joint_names=feedback["arm_joint_names"],
        finger_joint_names=finger_names,
        native_joint_ranges=parameters.joint_limits.position[:, full_columns].T.cpu().numpy(),
        named_velocity_limits=velocity_limits)
    actual_fingers = np.asarray([measured[name] for name in finger_names])
    actual_dq = dict(zip(feedback["joint_names"], feedback["start_dq"]))
    initial_bounds = np.asarray(binding["initial_fingers"]["position_bounds"])
    if (np.any(actual_fingers < initial_bounds[:, 0]) or np.any(actual_fingers > initial_bounds[:, 1])
            or np.any(np.abs([actual_dq[name] for name in finger_names])
                      > np.asarray(binding["initial_fingers"]["maximum_speed"]))):
        raise ValueError("Measured initial fingers violate the shared execution bounds")
    initial = np.array([measured[name] for name in names])
    values = np.repeat(initial[None], 3, axis=0)
    # Use each joint's in-range lower/upper coordinates, without assuming its
    # sign or silently clamping the measured state. Affinity is checked below.
    values[:, columns] = limits[:, 0]
    for index, column in enumerate(columns):
        values[index+1, column] = limits[index, 1]
    query = device.to_device(values)[None].contiguous()
    with torch.no_grad():
        state = native.compute_kinematics(JointState.from_position(query, joint_names=list(names)))
        balls = problem.transform_robot_spheres(state.robot_spheres)[0].cpu().numpy().copy()
        tool_index = native.tool_frames.index(problem.tool_link)
        tool_p, tool_q = problem.transform_tool_pose(
            state.tool_poses.position[:, :, tool_index], state.tool_poses.quaternion[:, :, tool_index])
        p, q = tool_p[0].cpu().numpy().copy(), tool_q[0].cpu().numpy().copy()
    local = np.stack([Rotation.from_quat(r[[1, 2, 3, 0]]).inv().apply(row[:, :3]-x)
                      for row, x, r in zip(balls, p, q)])[:, indices]
    spans = limits[:, 1]-limits[:, 0]
    if np.any(spans <= 0):
        raise ValueError("Calibrated prismatic fingers need positive native mechanical ranges")
    affine = np.stack([(local[index+1]-local[0])/spans[index]
                       for index in range(len(columns))], axis=-1)
    zero = local[0]-np.einsum("ncj,j->nc", affine, limits[:, 0])
    local_spheres = np.c_[zero, balls[0, indices, 3]]
    with torch.no_grad():
        state = native.compute_kinematics(JointState.from_position(
            device.to_device(initial)[None, None], joint_names=list(names)))
        current = problem.transform_robot_spheres(state.robot_spheres)[0, 0].cpu().numpy().copy()
        tool_p, tool_q = problem.transform_tool_pose(
            state.tool_poses.position[:, :, tool_index], state.tool_poses.quaternion[:, :, tool_index])
        p, q = tool_p[0, 0].cpu().numpy().copy(), tool_q[0, 0].cpu().numpy().copy()
    current_local = Rotation.from_quat(q[[1, 2, 3, 0]]).inv().apply(current[list(indices), :3]-p)
    predicted = zero+np.einsum("ncj,j->nc", affine, initial[columns])
    error = float(np.max(np.abs(current_local-predicted)))
    if error > 2e-6:
        raise ValueError(f"Native finger geometry is not affine in the declared TCP: {error} m")
    pairs = native.get_self_collision_config().collision_pairs.cpu().numpy()
    self_pairs = tuple(tuple(int(i) for i in pair) for pair in pairs if set(pair)&set(indices))
    hand = PrismaticGripper(
        controller="position_bounds", joint_names=finger_names,
        current_positions=initial[columns], joint_limits=limits,
        joint_bounds=event_finger_bounds(execution_limits, problem.events),
        coupling_matrix=np.empty((0, len(columns))), coupling_offset=np.empty(0),
        sphere_indices=indices, local_spheres=local_spheres,
        sphere_joint_translation=affine, self_collision_pairs=self_pairs,
        commands=tuple({"event_id": event.event_id, "skill_id": event.skill_id,
                        "command": event.kind, "progress": event.progress}
                       for event in problem.events if event.kind in {"open", "close", "hold"}),
        source=f"Native robot FK and supervised independent position control: {execution_limits.source}")
    report = {"affine_fk_max_error_m": error, "native_sphere_count": native.total_spheres,
                  "finger_sphere_indices": list(indices), "current_fingers": initial[columns].tolist(),
                  "mechanical_limits": limits.tolist(), "controller": "position_bounds",
                  "execution_limits": execution_limits.describe(),
                  "enabled_self_collision_pairs": len(self_pairs),
                  "contact_permissions": "No current finite material-patch permission has been certified"}
    # BEGIN bind fixed installation pair policy to the actual native sphere ordering.
    sphere_indices_by_link = {
        link: [int(index) for index in parameters.get_sphere_index_from_link_name(link).cpu().tolist()]
        for link in robot["robot_cfg"]["kinematics"]["collision_link_names"]}
    return hand, report, {"robot_spheres": current, "tool_position": p, "tool_quaternion": q,
                         "sphere_indices_by_link": sphere_indices_by_link}
# END one actual-finger interval contract shared by planning and execution.
    # END the binary hand contract and its measured FK remain unchanged.


def run(args):
    from prepare_recorded_task import prepare_candidate
    from recorded_observation import render_robot_mask
    from recorded_task_planning import plan_and_record
    from recorded_simulation import ExecutionLimits
    from manipulation.planning import PlanningBudget
    from manipulation.planning_observation import build_scene_from_rgbd
    from curobo.perception import MapperCfg
    from curobo.types import DeviceCfg
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "preparing", "segment_id": args.segment,
              "candidate_id": args.candidate, "planning_called": False,
              "execution_tested": False, "scope": "Observed independent skill snapshot",
              "manifest_path": str(args.manifest.resolve()),
              "feedback_path": str(args.feedback.resolve()),
              "robot_fixture_path": str(args.robot_fixture.resolve())}
    started = time.monotonic()
    try:
        manifest = json.loads(args.manifest.read_text())
        root = Path(manifest["workspace_root"])
        segment = next(s for s in manifest["segments"] if s["id"] == args.segment)
        candidate = next(c for c in segment["grounding"]["candidates"] if c["id"] == args.candidate)
        program = json.loads((root/segment["program_path"]).read_text())
        scene_input = json.loads((root/candidate["scene_path"]).read_text())
        fixture = json.loads(args.robot_fixture.read_text())
        feedback = json.loads(args.feedback.read_text())
        execution_limits = ExecutionLimits.from_mapping(json.loads(args.execution_limits.read_text()))
        report["execution_limits_path"] = str(args.execution_limits.resolve())
        measured = dict(zip(feedback["joint_names"], feedback["start_q"]))
        velocities = dict(zip(feedback["joint_names"], feedback["start_dq"]))
        active = tuple(feedback["arm_joint_names"])
        planning_feedback = {**feedback, "joint_names": list(active),
                             "start_q": [measured[name] for name in active],
                             "start_dq": [velocities[name] for name in active]}
        # BEGIN pass the requested approach boundary before reference grounding.
        problem, provenance = prepare_candidate(
            program, scene_input, planning_feedback, args.segment,
            pregrasp_frame=args.pregrasp_frame)
        # END pass the requested approach boundary before reference grounding.
        # The supplied camera calibration is explicitly world-from-camera.
        # Another output frame needs its real calibrated transform, never a
        # relabelled world observation at this narrow recorded-test boundary.
        if problem.reference_frame != "world":
            raise ValueError("Recorded world camera calibration requires a world-frame planning problem")
        robot = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())
        # BEGIN verify the physical policy and separation certificates before GPU use.
        # The fixture distinguishes native exclusions from queries completed
        # by an all-configuration geometry proof. Recompute that contract on
        # load; an edited ignore list is not a valid certificate.
        from robot_fixture import validate_fixture_collision_contract, create_fixed_installation_owners
        report["robot_collision_contract"] = validate_fixture_collision_contract(fixture, robot)
        report["robot_collision_contract_validated"] = True
        # END verify the physical policy and separation certificates before GPU use.
        sphere_count = sum(len(rows) for rows in robot["robot_cfg"]["kinematics"]["collision_spheres"].values())
        if not 1 <= sphere_count <= 512 or sphere_count != len(fixture["sphere_materials"]):
            raise ValueError("Actual native robot geometry/materials must agree within the 512-sphere local budget")
        robot["robot_cfg"]["kinematics"]["lock_joints"] = {
            name: measured[name] for name in feedback["finger_joint_names"]}
        if not torch.cuda.is_available():
            raise RuntimeError("This explicit local planner test requires CUDA access")
        if args.torch_memory_mib not in (2048, 4096) or args.scene_workspace_mib not in (1024, 2048):
            raise ValueError("Use only the explicitly bounded local Torch and scene allocations")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(
            args.torch_memory_mib*1024**2/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        report["resource_limits"] = {"torch_allocator_bytes": args.torch_memory_mib*1024**2,
                                     "scene_workspace_bytes": args.scene_workspace_mib*1024**2,
                                     "mapping_bytes": 1024**3}
        device = DeviceCfg("cuda:0")
        hand, hand_report, entrance_fk = position_controlled_hand(
            robot, feedback, problem, device=device, execution_limits=execution_limits,
            velocity_limits=fixture["velocity_limits"])
        report["gripper"] = hand_report
        carrier_id = scene_input["carrier_id"]
        carrier_phases = [phase for phase in problem.phases
                          if phase.metadata.get("carrier_id") == carrier_id]
        if not carrier_phases:
            raise ValueError("The selected skill must bind its explicit carrier in the current problem")
        owner_pose = np.asarray(carrier_phases[0].metadata["owner_initial_pose"])
        supplied_masks = dict(args.owner_mask)
        if len(supplied_masks) != len(args.owner_mask) or set(supplied_masks) != {
                view["id"] for view in segment["snapshot"]["views"]}:
            raise ValueError("Supply exactly one current carrier owner-mask path for every identified view")
        report["carrier_observation"] = {
            "owner_id": carrier_id, "initial_pose": owner_pose.tolist(),
            "mask_paths": {key: str(Path(path).resolve()) for key, path in supplied_masks.items()},
            "geometry_path": None if args.owner_geometry is None else str(args.owner_geometry.resolve()),
            "reference_partition": "All visible non-carrier non-robot geometry remains in observed background"}
        observations = {}
        for view in segment["snapshot"]["views"]:
            calibration = json.loads((root/view["calibration_path"]).read_text())
            rgb = np.array(Image.open(root/view["rgb_path"]).convert("RGB"))
            depth = np.load(root/view["depth_path"], allow_pickle=False)
            rendered = render_robot_mask(
                scene_xml=Path(fixture["source_scene_xml"]), robot_root=fixture["robot_visual_root"],
                joint_map=[(name, name, 1.) for name in feedback["joint_names"]],
                joint_positions=measured, T_world_robot_root=(np.asarray(feedback["base_pose"])
                    @ np.asarray(fixture["T_base_visual_root"])),
                rgb=rgb, depth_m=depth, calibration=calibration,
                observation_id=view["id"], feedback_source=feedback["source"],
                depth_tolerance_m=args.robot_depth_tolerance,
                depth_tolerance_source=args.robot_depth_tolerance_source)
            view_output = args.output/"observations"/view["id"]
            view_output.mkdir(parents=True)
            np.save(view_output/"robot_mask.npy", rendered["robot_mask"])
            np.save(view_output/"unknown_boundary.npy", rendered["unknown_boundary"])
            (view_output/"robot-mask-report.json").write_text(json.dumps(rendered["report"], indent=2)+"\n")
            owner_mask = np.load(supplied_masks[view["id"]], allow_pickle=False)
            observations[view["id"]] = {
                "image": rgb, "depth": depth, "K": np.asarray(calibration["K"]),
                "T_output_camera": np.asarray(calibration["T_world_camera_cv"]),
                "reference_frame": problem.reference_frame,
                "robot_mask": rendered["robot_mask"],
                "owner_masks": {carrier_id: owner_mask}}
        # The finite grid is an explicit test resource limit. Space outside it
        # stays unknown; narrowing it never certifies a removed obstacle.
        height, width = next(iter(observations.values()))["depth"].shape
        config = MapperCfg(extent_meters_xyz=tuple(args.extent), voxel_size=args.voxel_size,
                           extent_esdf_meters_xyz=tuple(args.extent), esdf_voxel_size=args.voxel_size,
                           grid_center=torch.tensor(args.center), truncation_distance=2*args.voxel_size,
                           image_height=height, image_width=width, depth_minimum_distance=.05,
                           depth_maximum_distance=5., device="cuda:0")
        scene = build_scene_from_rgbd(
            observations, {carrier_id: {
                "T_output_owner": owner_pose,
                "source": str(root/candidate["scene_path"])}}, mapper_cfg=config,
            depth_error_m=args.depth_error, reference_frame=problem.reference_frame,
            robot_sphere_materials=tuple(fixture["sphere_materials"]),
            robot_geometry_source=fixture["robot_geometry_source"], gripper=hand,
            max_mapping_bytes=1024**3)
        report["observation_build"] = scene.observation_report
        if args.owner_geometry is not None:
            # Replace only this explicitly partitioned carrier's partial query
            # by its declared complete offline geometry. Its current pose and
            # conditional attachment still come from this planning problem;
            # no reference, unlabelled surface or unknown volume is removed.
            from recorded_owner_geometry import create_box_owner
            from manipulation.planning_scene import PlanningScene
            prepared = json.loads(args.owner_geometry.read_text())
            if prepared["owner_id"] != carrier_id:
                raise ValueError("Complete geometry must belong to this explicit current carrier")
            owner = create_box_owner(prepared, device_cfg=device)
            observed_report = scene.observation_report
            scene = PlanningScene(
                background=scene.background, owners=(owner,), coverage=scene.coverage,
                reference_frame=scene.reference_frame,
                robot_sphere_materials=scene.robot_sphere_materials,
                robot_geometry_source=scene.robot_geometry_source, gripper=hand)
            scene.observation_report = observed_report
            report["complete_carrier_geometry"] = prepared["provenance"]
        # BEGIN retain self-filtered installation as complete static collision owners.
        # These primitives use the same calibrated base as FK. They have no
        # moving sphere cover, so a mount extending beyond the observed grid
        # does not become another unknown-space motion actor. Its native pair
        # exclusions affect only robot/this-installation queries; payload and
        # unlabelled background collision checks keep their original meaning.
        fixed_owners, fixed_report = create_fixed_installation_owners(
            fixture, device_cfg=device, T_output_base=feedback["base_pose"],
            sphere_indices_by_link=entrance_fk["sphere_indices_by_link"])
        if fixed_owners:
            from manipulation.planning_scene import PlanningScene
            observed_report = scene.observation_report
            scene = PlanningScene(
                background=scene.background, owners=(*scene.owners, *fixed_owners), coverage=scene.coverage,
                reference_frame=scene.reference_frame, robot_sphere_materials=scene.robot_sphere_materials,
                robot_geometry_source=scene.robot_geometry_source, gripper=hand, contacts=scene.contacts,
                numerical_position_tolerance_m=scene.numerical_position_tolerance_m,
                numerical_rotation_tolerance_rad=scene.numerical_rotation_tolerance_rad)
            scene.observation_report = observed_report
        report["fixed_robot_installation"] = fixed_report
        # END observed depth and the existing unknown-space policy are not widened.
        scene.validate_problem(problem)
        # Independent entrance diagnostics use the same measured FK as the
        # native hand calibration. They expose pre-existing scene violations;
        # they do not select a candidate or waive an optimization constraint.
        with torch.no_grad():
            terms = scene.components(
                device.to_device(entrance_fk["robot_spheres"])[None, None],
                device.to_device(entrance_fk["tool_position"])[None, None],
                device.to_device(entrance_fk["tool_quaternion"])[None, None],
                device.to_device([[0.]]), device.to_device(problem.start_q)[None, None],
                problem.joint_names)
            report["initial_scene_components"] = {name: float(value.max()) for name, value in terms.items()}
        budget = PlanningBudget(knots=args.knots, interpolation_buffer_size=args.interpolation_buffer_size,
                                max_robot_spheres=512, ik_seeds=args.ik_seeds,
                                trajectory_seeds=args.trajectory_seeds,
                                ik_iterations=80, trajectory_iterations=100,
                                max_validation_points=args.max_validation_points,
                                max_refinement_points=args.max_refinement_points,
                                max_scene_workspace_bytes=args.scene_workspace_mib*1024**2,
                                maximum_trajectory_solves=2, search_seconds=3.)
        report["planning_called"] = True
        result = plan_and_record(problem, scene, robot, args.output/"planning", budget=budget,
                                 execution_limits=execution_limits.describe(),
                                 provenance={**provenance, "robot_fixture": str(args.robot_fixture),
                                             "current_views": list(observations),
                                             "old_selection_comparison_only": segment["grounding"]["status"]})
        report.update(status="completed", planning_success=result.success)
    except KeyboardInterrupt:
        report.update(status="interrupted", interruption_reason="Explicit operator interrupt")
        (args.output/"exception.txt").write_text(traceback.format_exc())
        raise
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (args.output/"exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        (args.output/"run-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "robot-fixture", "feedback", "execution-limits", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--candidate", required=True)
    # BEGIN recorded callers select source evidence, never a dataset constant.
    parser.add_argument("--pregrasp-frame", type=int,
                        help="Original source approach frame; required for grasp planning")
    # END recorded callers select source evidence, never a dataset constant.
    parser.add_argument("--owner-mask", nargs=2, action="append", required=True,
                        metavar=("VIEW_ID", "MASK_PATH"))
    parser.add_argument("--owner-geometry", type=Path)
    parser.add_argument("--center", type=float, nargs=3, required=True)
    parser.add_argument("--extent", type=float, nargs=3, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--depth-error", type=float, required=True)
    parser.add_argument("--robot-depth-tolerance", type=float, required=True)
    parser.add_argument("--robot-depth-tolerance-source", required=True)
    parser.add_argument("--knots", type=int, default=16)
    parser.add_argument("--interpolation-buffer-size", type=int, default=768)
    parser.add_argument("--ik-seeds", type=int, default=4)
    parser.add_argument("--trajectory-seeds", type=int, default=2)
    parser.add_argument("--max-validation-points", type=int, default=2048)
    parser.add_argument("--max-refinement-points", type=int, default=64)
    parser.add_argument("--scene-workspace-mib", type=int, choices=(1024, 2048), default=1024)
    parser.add_argument("--torch-memory-mib", type=int, choices=(2048, 4096), default=2048)
    run(parser.parse_args())
# END observed skill integration using existing production scene contracts.
