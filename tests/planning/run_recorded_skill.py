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


# BEGIN independent finger reference on the planner's exact physical clock.
def finger_trajectory_factory(problem, feedback, finger_targets, *, timestep, ctrlrange,
                              position_limits, velocity_limits, acceleration_limits):
    """Bind original events once the native formal duration/map are known.

    Actual-position protection bounds are deliberately absent from this
    constructor. They continue guarding measured feedback in the executor;
    they cannot predict a hand shape or replace the original position goals.
    The returned reference starts at measured q/dq and remains a command,
    including when physical contact prevents a target from being reached.
    """
    from manipulation.gripper_trajectory import FingerTrajectory

    names = tuple(feedback["finger_joint_names"])
    measured = dict(zip(feedback["joint_names"], feedback["start_q"]))
    speeds = dict(zip(feedback["joint_names"], feedback["start_dq"]))

    def build(duration_s, progress_map):
        events = []
        for identity, clock in progress_map.event_clock.items():
            # BEGIN consume the sole device-quantized source/event binding.
            from manipulation.planning_problem import event_occurrence_key
            event = deepcopy(clock["event"])
            if event_occurrence_key(event) != identity:
                raise ValueError("A finger reference needs the exact bound event occurrence metadata")
            events.append({**event, **{key: deepcopy(value) for key, value in clock.items() if key != "event"}})
            # END hand commands cannot independently invert or retime progress.
        barriers = [row["time"] for row in events if row["requires_evidence"] is not None]
        stop = min(barriers) if barriers else duration_s
        commands = finger_targets.bind(SimpleNamespace(events=events, duration=duration_s),
            finger_joint_names=names, ctrlrange=ctrlrange)
        velocity = np.asarray([velocity_limits[name] for name in names])
        acceleration = np.asarray([acceleration_limits[name] for name in names])
        return FingerTrajectory.build(joint_names=names,
            initial_position=[measured[name] for name in names],
            initial_velocity=[speeds[name] for name in names], commands=commands,
            duration=duration_s, executable_until=stop, timestep=timestep,
            position_limits=np.asarray([position_limits[name] for name in names]).T,
            velocity_limits=np.stack((-velocity, velocity)),
            acceleration_limits=np.stack((-acceleration, acceleration)),
            source=(f"Original explicit actuator targets: {finger_targets.source}; C1 command reference "
                    "under native planner acceleration bounds, not calibrated actual acceleration/tracking limits"),
            feedback_id=feedback["feedback_id"], timestamp_s=feedback["sim_time_s"])

    # BEGIN expose the already bound device tick to the planning clock owner.
    build.timestep = float(timestep)
    # END this is device metadata, never a per-skill retiming rule.
    return build
# END full mechanical intervals no longer select future hand geometry.


# BEGIN share unchanged native FK calibration with source-backed planning probes.
def position_controlled_hand(robot, feedback, problem, *, device, execution_limits,
                             position_limits, velocity_limits, trajectory_factory, execution_events=None):
    """Bind independent command reference after checking actual execution limits.

    The shared test adapter calibrates exactly the same native spheres and
    prismatic translations. This recorded boundary still owns the measured
    q/dq checks, execution policy, command identities and reference factory.
    """
    from manipulation.planning_scene import PrismaticGripper
    from probe_robot_fixture import calibrate_native_hand

    execution_events = problem.events if execution_events is None else execution_events
    measured = dict(zip(feedback["joint_names"], feedback["start_q"]))
    finger_names = tuple(feedback["finger_joint_names"])
    binding = execution_limits.bind(
        SimpleNamespace(joint_names=problem.joint_names, events=[asdict(event) for event in execution_events]),
        joint_names=feedback["joint_names"], arm_joint_names=feedback["arm_joint_names"],
        finger_joint_names=finger_names,
        native_joint_ranges=[position_limits[name] for name in feedback["joint_names"]],
        named_velocity_limits=velocity_limits)
    actual_fingers = np.asarray([measured[name] for name in finger_names])
    actual_dq = dict(zip(feedback["joint_names"], feedback["start_dq"]))
    initial_bounds = np.asarray(binding["initial_fingers"]["position_bounds"])
    if (np.any(actual_fingers < initial_bounds[:, 0]) or np.any(actual_fingers > initial_bounds[:, 1])
            or np.any(np.abs([actual_dq[name] for name in finger_names])
                      > np.asarray(binding["initial_fingers"]["maximum_speed"]))):
        raise ValueError("Measured initial fingers violate the shared execution bounds")
    geometry = calibrate_native_hand(robot, feedback, problem, device=device, position_limits=position_limits)
    hand = PrismaticGripper(
        controller="timed_reference", joint_names=finger_names,
        current_positions=geometry["current_positions"], joint_limits=geometry["joint_limits"],
        joint_bounds=None, finger_trajectory_factory=trajectory_factory,
        control_timestep_s=trajectory_factory.timestep,
        coupling_matrix=np.empty((0, len(finger_names))), coupling_offset=np.empty(0),
        sphere_indices=geometry["sphere_indices"], local_spheres=geometry["local_spheres"],
        sphere_joint_translation=geometry["sphere_joint_translation"],
        self_collision_pairs=geometry["self_collision_pairs"],
        commands=tuple({"event_id": event.event_id, "skill_id": event.skill_id,
                        "occurrence": event.occurrence, "command": event.kind, "progress": event.progress}
                       for event in execution_events if event.kind in {"open", "close", "hold"}),
        source=f"Native robot FK and synchronous independent position reference: {execution_limits.source}")
    report = {"affine_fk_max_error_m": geometry["affine_fk_max_error_m"],
        "native_sphere_count": len(geometry["robot_spheres"]),
        "finger_sphere_indices": list(geometry["sphere_indices"]),
        "current_fingers": geometry["current_positions"].tolist(),
        "mechanical_limits": geometry["joint_limits"].tolist(), "controller": "timed_reference",
        "geometry_scope": "same-time command reference; actual full named q/dq is separately supervised",
        "execution_limits": execution_limits.describe(),
        "enabled_self_collision_pairs": len(geometry["self_collision_pairs"]),
        "contact_permissions": "Bound separately from explicit native material faces and current owner-patch declarations"}
    return hand, report, {key: geometry[key] for key in
        ("robot_spheres", "tool_position", "tool_quaternion", "sphere_indices_by_link")}
# END execution semantics stay here; only measured native geometry is shared.


def run(args):
    from prepare_recorded_task import prepare_candidate
    from recorded_observation import render_robot_mask
    from recorded_task_planning import plan_and_record
    # BEGIN use the production execution contract shared with the executor.
    from manipulation.execution import ExecutionLimits, FixedFingerTargets
    # END production execution contract.
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
        # BEGIN preserve the same explicit snapshot identity used by execution.
        feedback["feedback_id"] = str(args.feedback.resolve())
        finger_targets = FixedFingerTargets.from_mapping(json.loads(args.finger_targets.read_text()))
        report["finger_targets_path"] = str(args.finger_targets.resolve())
        # END targets remain a separate input from measured feedback.
        execution_limits = ExecutionLimits.from_mapping(json.loads(args.execution_limits.read_text()))
        report["execution_limits_path"] = str(args.execution_limits.resolve())
        # BEGIN retain prior fixed targets and the last actually applied protection interval.
        continuation = None if args.continuation is None else json.loads(args.continuation.read_text())
        rebind = None if args.rebind is None else json.loads(args.rebind.read_text())
        request_ledger = None if args.request_ledger is None else json.loads(args.request_ledger.read_text())
        if args.request_ledger is not None and not isinstance(request_ledger, dict):
            raise ValueError("An explicitly supplied planning ledger cannot be null or reset to a new request")
        report["request_ledger_input"] = None if args.request_ledger is None else str(args.request_ledger.resolve())
        if rebind is not None and continuation is None:
            raise ValueError("An explicit same-goal rebind requires the previous execution ledger")
        if continuation is not None:
            from manipulation.execution import continuation_execution_inputs
            finger_targets, execution_limits = continuation_execution_inputs(
                continuation, finger_targets, execution_limits, rebind=rebind)
        # END unconsumed commands keep their original identities and values.
        measured = dict(zip(feedback["joint_names"], feedback["start_q"]))
        velocities = dict(zip(feedback["joint_names"], feedback["start_dq"]))
        active = tuple(feedback["arm_joint_names"])
        planning_feedback = {**feedback, "joint_names": list(active),
                             "start_q": [measured[name] for name in active],
                             "start_dq": [velocities[name] for name in active]}
        # BEGIN bind true goals without a reference waypoint.
        problem, provenance = prepare_candidate(
            program, scene_input, planning_feedback, args.segment)
        # END reference stages do not impose an approach boundary.
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
        # BEGIN planning and execution consume the same source-solid/pair adapter.
        # Build once before GPU allocation. The planner binds this independent
        # model to its own scene and formal clock; the recorder only passes it.
        from robot_validation_fixture import build_robot_validator
        robot_validator = build_robot_validator(fixture, reference_frame=problem.reference_frame,
            T_reference_base=problem.base_pose, tool_frame=problem.tool_link,
            T_frame_tcp=problem.tool_transform)
        report["independent_robot_model"] = robot_validator.describe()
        # END no sphere-cover certificate replaces a required source-solid pair.
        sphere_count = sum(len(rows) for rows in robot["robot_cfg"]["kinematics"]["collision_spheres"].values())
        # BEGIN admit the same finite refined-cover capacity as fixture and planner.
        # Full native geometry/material ownership remains exact. The existing
        # scene and whole-device memory checks still guard the actual rollout.
        if not 1 <= sphere_count <= 640 or sphere_count != len(fixture["sphere_materials"]):
            raise ValueError("Actual native robot geometry/materials must agree within the 640-sphere local budget")
        # END added capacity never changes a collision residual or permission.
        robot["robot_cfg"]["kinematics"]["lock_joints"] = {
            name: measured[name] for name in feedback["finger_joint_names"]}
        # BEGIN read native command timing/ranges from the selected robot model.
        # No world/object state enters planning through this hardware binding.
        import mujoco
        # The exported geometry-only fixture deliberately has no actuators.
        # Read only the named robot actuator metadata from its source model;
        # no MjData or current object state is constructed or queried here.
        native_model = mujoco.MjModel.from_xml_path(fixture["source_scene_xml"])
        finger_actuators = []
        for name in feedback["finger_joint_names"]:
            joint_id = native_model.joint(name).id
            selected = np.flatnonzero((native_model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (native_model.actuator_trnid[:, 0] == joint_id))
            if len(selected) != 1:
                raise ValueError("Each native finger requires its single explicit position actuator")
            finger_actuators.append(int(selected[0]))
        reference_factory = finger_trajectory_factory(problem, feedback, finger_targets,
            timestep=float(native_model.opt.timestep),
            ctrlrange=native_model.actuator_ctrlrange[finger_actuators].copy(),
            position_limits=fixture["position_limits"], velocity_limits=fixture["velocity_limits"],
            acceleration_limits={name: float(robot["robot_cfg"]["kinematics"]["cspace"]["max_acceleration"])
                                 for name in feedback["finger_joint_names"]})
        del native_model
        # END the finger factory binds only after the native formal clock exists.
        if not torch.cuda.is_available():
            raise RuntimeError("This explicit local planner test requires CUDA access")
        # BEGIN expose the production scene budget without raising Torch limits.
        if args.torch_memory_mib not in (2048, 4096) or args.scene_workspace_mib not in (1024, 2048, 3072, 6144):
            raise ValueError("Use only the explicitly bounded local Torch and scene allocations")
        # END process-tree and device-growth guards remain separate run limits.
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
            position_limits=fixture["position_limits"], velocity_limits=fixture["velocity_limits"],
            trajectory_factory=reference_factory)
        report["gripper"] = hand_report
        # BEGIN bind grasp arrival to the exact native hand/observed-owner pair.
        # Whole-hand grasp contact uses the problem's explicit arrival/close
        # condition. Finite material faces remain a separate optional policy
        # for tasks that explicitly need a particular local contact region.
        from robot_fixture import bind_fixture_contact_materials, bind_fixture_gripper_geometry
        from manipulation.contact_binding import build_material_contact_rules
        from manipulation.planning_scene import RobotOwnerContactPermission
        activations = [row for row in problem.conditions if row.get("kind") == "grasp_contact_activation"]
        contacts, pair_permissions = (), ()
        material_binding = {"robot_sphere_materials": [None]*sphere_count, "native_geometry_materials": {}}
        if activations:
            if args.contact_bindings is not None:
                raise ValueError("Pose-gated grasp uses its bound whole-hand permission, not obsolete finite grasp patches")
            hand_binding = bind_fixture_gripper_geometry(
                fixture, robot, entrance_fk["sphere_indices_by_link"], feedback["finger_joint_names"])
            pair_permissions = tuple(RobotOwnerContactPermission(
                rule_id=row["condition_id"], target_owner=row["carrier_id"],
                native_geometry_names=tuple(hand_binding["native_geometry_names"]),
                sphere_indices=tuple(hand_binding["sphere_indices"]), interval=tuple(row["interval"]),
                arrival_goal_id=tuple(row["arrival_goal_id"]), activation_event=row["activation_event"],
                source=f"{hand_binding['source']}; frozen perception arrival: {row['condition_id']}")
                for row in activations)
            report["contact_binding"] = {"policy": "measured_pose_arrival_before_close",
                "hand_geometry": hand_binding, "permissions": [asdict(rule) for rule in pair_permissions]}
        elif args.contact_bindings is not None:
            contact_inputs = json.loads(args.contact_bindings.read_text())
            if not isinstance(contact_inputs, dict) or set(contact_inputs) != {"material_faces", "contacts", "source"}:
                raise ValueError("Contact bindings need explicit material_faces, contacts and source")
            if not isinstance(contact_inputs["source"], str) or not contact_inputs["source"].strip():
                raise ValueError("Contact bindings require a current evidence source")
            material_binding = bind_fixture_contact_materials(
                fixture, robot, entrance_fk["sphere_indices_by_link"], feedback,
                contact_inputs["material_faces"])
            contacts = build_material_contact_rules(material_binding, contact_inputs["contacts"])
            report["contact_binding"] = {"policy": "explicit_finite_material_contacts",
                "path": str(args.contact_bindings.resolve()), "source": contact_inputs["source"],
                "materials": material_binding, "rule_ids": [rule.rule_id for rule in contacts]}
        else:
            report["contact_binding"] = {"policy": "no_robot_object_contact_permission"}
        # END permissions affect software collision legality, never native physical contact.
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
        # BEGIN bind the complete native covers and explicit request-local fine ROIs.
        # Each fixed link may own several geoms. Its entire sphere group and
        # every corresponding solid use one map; material labels are unrelated.
        body_geometries = {}
        for row in fixture["collision_geometries"]:
            body_geometries.setdefault(row["link"], []).append(row["geom"])
        collision_bodies = [{"body_id": link, "frame": link, "sphere_indices": indices,
                             "geometry_names": body_geometries[link],
                             "local_spheres": [[*sphere["center"], sphere["radius"]]
                                 for sphere in robot["robot_cfg"]["kinematics"]["collision_spheres"][link]],
                             "source": fixture["robot_geometry_source"]}
            for link, indices in entrance_fk["sphere_indices_by_link"].items()]
        fine_regions = {}
        if args.fine_regions is not None:
            requests = json.loads(args.fine_regions.read_text())
            for row in requests:
                identity = row["owner_id"]
                if identity in fine_regions:
                    raise ValueError("Only one bounded fine ROI is admitted per physical owner partition")
                fine_regions[identity] = {key: value for key, value in row.items() if key != "owner_id"}
        # END no simulator obstacle pose or task-specific crop enters the mapper.
        scene = build_scene_from_rgbd(
            observations, {carrier_id: {
                "T_output_owner": owner_pose,
                "source": str(root/candidate["scene_path"])}}, mapper_cfg=config,
            depth_error_m=args.depth_error, reference_frame=problem.reference_frame,
            robot_sphere_materials=tuple(material_binding["robot_sphere_materials"]),
            robot_geometry_source=fixture["robot_geometry_source"], gripper=hand,
            # BEGIN keep contact identities and finite permissions on the observed scene.
            contacts=contacts, native_geometry_materials=material_binding["native_geometry_materials"],
            # BEGIN both independent and planner models retain the exact pair policy.
            robot_owner_permissions=pair_permissions,
            # END a current measured arrival is still required by execution.
            owner_voxel_size_m=args.owner_voxel_size,
            fine_regions=fine_regions, robot_collision_bodies=collision_bodies,
            max_mapping_bytes=1024**3)
            # END the RGB-D partition remains partial.
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
                robot_collision_bodies=scene.robot_collision_bodies,
                # BEGIN preserve contact contracts when substituting explicit owner geometry.
                robot_geometry_source=scene.robot_geometry_source, gripper=hand,
                contacts=scene.contacts, native_geometry_materials=scene.native_geometry_materials,
                # BEGIN replacing owner geometry preserves the declared grasp policy.
                robot_owner_permissions=scene.robot_owner_permissions,
                # END the owner identity remains unchanged.
                numerical_position_tolerance_m=scene.numerical_position_tolerance_m,
                numerical_rotation_tolerance_rad=scene.numerical_rotation_tolerance_rad)
                # END the replacement does not relax any robot material permission.
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
                robot_collision_bodies=scene.robot_collision_bodies,
                robot_geometry_source=scene.robot_geometry_source, gripper=hand, contacts=scene.contacts,
                # BEGIN native identity survives fixed-installation augmentation.
                native_geometry_materials=scene.native_geometry_materials,
                # BEGIN fixed installation cannot drop the bound hand/owner permission.
                robot_owner_permissions=scene.robot_owner_permissions,
                # END all installation checks remain separately active.
                # END both geometry checkers retain identical material bindings.
                numerical_position_tolerance_m=scene.numerical_position_tolerance_m,
                numerical_rotation_tolerance_rad=scene.numerical_rotation_tolerance_rad)
            scene.observation_report = observed_report
        report["fixed_robot_installation"] = fixed_report
        # END fixed installation remains independently collision checked.
        # BEGIN same-model baseline occupancy, separate from observed free F.
        from robot_coverage_fixture import build_robot_coverage
        from manipulation.planning_scene import PlanningScene
        coverage = build_robot_coverage(fixture, feedback, scene.coverage,
            initialization_source=args.initial_state_assumption_source,
            max_queries=(2*args.trajectory_seeds+2)*args.max_validation_points+1)
        observed_report = scene.observation_report
        scene = PlanningScene(background=scene.background, owners=scene.owners, coverage=scene.coverage,
            reference_frame=scene.reference_frame, robot_sphere_materials=scene.robot_sphere_materials,
            robot_collision_bodies=scene.robot_collision_bodies,
            robot_geometry_source=scene.robot_geometry_source, gripper=hand, contacts=scene.contacts,
            # BEGIN robot occupancy evidence preserves object-contact identity.
            native_geometry_materials=scene.native_geometry_materials,
            # BEGIN occupied-volume evidence does not change contact policy.
            robot_owner_permissions=scene.robot_owner_permissions,
            # END keep identical robot/owner pairs through every scene rebuild.
            # END occupancy does not grant or remove contact permissions.
            robot_coverage=coverage, numerical_position_tolerance_m=scene.numerical_position_tolerance_m,
            numerical_rotation_tolerance_rad=scene.numerical_rotation_tolerance_rad)
        scene.observation_report = observed_report
        report["robot_coverage"] = scene.robot_coverage.describe()
        np.savez_compressed(args.output/"observed-free.npz", known_free=scene.coverage.known_free,
            origin=scene.coverage.origin, voxel_size=np.asarray(scene.coverage.voxel_size),
            observation_ids=np.asarray(scene.coverage.observation_ids))
        report["robot_coverage"]["original_free_grid_path"] = "observed-free.npz"
        scene.validate_problem(problem)
        # Only native measured data is available here. The planner owns the
        # single entrance verdict after binding the formal finger clock.
        np.savez_compressed(args.output/"entrance-scene.npz",
            measured_robot_spheres=entrance_fk["robot_spheres"],
            joint_names=np.asarray(feedback["joint_names"]),
            actual_q=np.asarray(feedback["start_q"]), actual_dq=np.asarray(feedback["start_dq"]))
        report["entrance_geometry_diagnostic"] = {
            "path": "entrance-scene.npz", "sphere_count": len(entrance_fk["robot_spheres"]),
            "scope": "Measured native q0; the formally bound planner performs the only entrance verdict"}
        # END no uncertainty envelope is labelled as the current hand state.
        # BEGIN bind refined device geometry to an explicit unchanged-work budget.
        # A supplied request ledger must declare this same model capacity;
        # historical work is retained by the existing exact-ledger validation.
        budget = PlanningBudget(knots=args.knots, interpolation_buffer_size=args.interpolation_buffer_size,
                                max_robot_spheres=640, ik_seeds=args.ik_seeds,
                                trajectory_seeds=args.trajectory_seeds,
                                ik_iterations=80, trajectory_iterations=100,
                                max_validation_points=args.max_validation_points,
                                max_refinement_points=args.max_refinement_points,
                                max_scene_workspace_bytes=args.scene_workspace_mib*1024**2,
                                # BEGIN expose the existing cumulative work contract.
                                # The supplied ledger must match these values;
                                # flags cannot reset or enlarge it implicitly.
                                maximum_trajectory_solves=args.maximum_trajectory_solves,
                                planning_seconds=args.planning_seconds, search_seconds=3.)
        # END geometry capacity does not replenish planning time or solve counts.
        # END reviewed input revisions remain caller-owned.
        report["planning_called"] = True
        # BEGIN keep this exact source model throughout recording and planning.
        result = plan_and_record(problem, scene, robot, args.output/"planning",
                                 robot_validator=robot_validator, budget=budget,
                                 execution_limits=execution_limits.describe(),
                                 continuation=continuation, rebind=rebind, request_ledger=request_ledger,
                                 provenance={**provenance, "robot_fixture": str(args.robot_fixture),
                                             "current_views": list(observations),
                                             "old_selection_comparison_only": segment["grounding"]["status"]})
        # END source geometry and policy are not reconstructed by the recorder.
        report.update(status="completed", planning_success=result.success,
                      initial_state=result.diagnostics.get("initial_state"),
                      robot_coverage={**scene.robot_coverage.describe(),
                                      "original_free_grid_path": "observed-free.npz"})
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
    # BEGIN grasp permission comes from arrival/close, with local contacts separately explicit.
    for name in ("manifest", "robot-fixture", "feedback", "execution-limits", "finger-targets", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--contact-bindings", type=Path,
                        help="Explicit finite material contact policy for a task without pose-gated grasp contact")
    # END omitted local bindings grant no local contact allowance.
    parser.add_argument("--segment", required=True)
    parser.add_argument("--candidate", required=True)
    # BEGIN same-skill continuation consumes an explicit prior execution ledger.
    parser.add_argument("--continuation", type=Path,
                        help="Prior continuation.json; current feedback and scene inputs must be freshly rebound")
    parser.add_argument("--rebind", type=Path,
                        help="Explicit remaining/retry decision JSON preserving original IDs and physical deadlines")
    parser.add_argument("--request-ledger", type=Path,
                        help="Prior request-ledger.json, including failed planning work; execution history also requires --continuation")
    # END missing current facts are binding failures, not replay defaults.
    parser.add_argument("--owner-mask", nargs=2, action="append", required=True,
                        metavar=("VIEW_ID", "MASK_PATH"))
    parser.add_argument("--owner-geometry", type=Path)
    parser.add_argument("--center", type=float, nargs=3, required=True)
    parser.add_argument("--extent", type=float, nargs=3, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    # BEGIN current owner precision is declared before a planning result exists.
    parser.add_argument("--owner-voxel-size", type=float, required=True)
    # BEGIN freeze reviewed local bounds, resolution and measurement evidence.
    parser.add_argument("--fine-regions", type=Path,
                        help="JSON array of owner_id (null for background), local bounds and measured-allowance evidence")
    # END the same generic ROI input is accepted for any observed partition.
    # END the mapper retains every current labelled sample in the refined field.
    parser.add_argument("--depth-error", type=float, required=True)
    parser.add_argument("--robot-depth-tolerance", type=float, required=True)
    parser.add_argument("--robot-depth-tolerance-source", required=True)
    parser.add_argument("--initial-state-assumption-source", required=True,
                        help="Explicit provenance for accepting the frozen native robot model occupancy at measured q0")
    parser.add_argument("--knots", type=int, default=16)
    parser.add_argument("--interpolation-buffer-size", type=int, default=768)
    parser.add_argument("--ik-seeds", type=int, default=4)
    parser.add_argument("--trajectory-seeds", type=int, default=2)
    parser.add_argument("--max-validation-points", type=int, default=2048)
    parser.add_argument("--max-refinement-points", type=int, default=64)
    # BEGIN make the existing request work limits reviewable in run commands.
    parser.add_argument("--maximum-trajectory-solves", type=int, default=2)
    parser.add_argument("--planning-seconds", type=float, default=120.)
    # END the production PlanningBudget and ledger checks validate these inputs.
    # BEGIN explicit larger scene storage shares the production six-GiB cap.
    # BEGIN explicit shared CPU cache capacity remains separate from Torch's cap.
    parser.add_argument("--scene-workspace-mib", type=int, choices=(1024, 2048, 3072, 6144), default=1024)
    # END the default allocation request and all geometric checks remain unchanged.
    # END the default allocation remains unchanged.
    parser.add_argument("--torch-memory-mib", type=int, choices=(2048, 4096), default=2048)
    # BEGIN opt-in host profiling covers preparation, planning and persistence.
    parser.add_argument("--profile-runtime", action="store_true",
                        help="Record host function timings under output/runtime-profile; disabled by default")
    args = parser.parse_args()
    from manipulation.runtime_profiling import run_with_optional_profile
    run_with_optional_profile(run, args, enabled=args.profile_runtime,
                              output=args.output/"runtime-profile")
    # END ordinary runs do not construct a profiler or write timing artifacts.
# END observed skill integration using existing production scene contracts.
