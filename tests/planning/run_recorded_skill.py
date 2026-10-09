"""Plan one observed skill with a resident backend and fixed online geometry.

All task, camera, contact and unknown-region inputs come from the caller.
The saved accepted curve is executed separately against actual robot feedback.
"""

# BEGIN observed skill integration using existing production scene contracts.
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
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


# BEGIN each observed owner has an explicit CLI resolution, without aliases.
def owner_voxel_sizes(arguments, owner_ids):
    """Parse repeated owner/spacing pairs without silently replacing duplicates.

    The scene adapter repeats the strict mapping check for non-CLI callers.
    Resolving identities here also rejects mistakes before creating a CUDA
    hand model, so a missing or misspelled owner cannot incur GPU work.
    """
    result = {}
    for identity, raw_value in arguments:
        if identity in result:
            raise ValueError(f"Duplicate owner voxel size for {identity!r}")
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Each owner voxel size must be a finite positive metre value")
        result[identity] = value
    if set(result) != set(owner_ids):
        raise ValueError("Owner voxel sizes must name every observed owner exactly once")
    return result
# END no object name, dataset or extent selects an implicit resolution.


# BEGIN persist the production builder's actual semantic partition decision.
def save_observation_partitions(scene, output):
    """Save exact fusion masks without reconstructing ownership in the runner.

    The scene builder owns semantic boundary decisions. These arrays describe
    its fresh background Mapper input; excluded endpoints are not free-space
    evidence. Original depth and independently measured free rays stay in the
    observation records rather than being inferred from these masks.
    """
    result = {}
    for name, masks in scene.observation_partition_masks.items():
        folder = Path(output)/"observations"/name
        folder.mkdir(parents=True, exist_ok=True)
        row = {}
        for key, filename in (("background", "background-mask.npy"),
                              ("semantic_boundary", "semantic-boundary-mask.npy")):
            path = folder/filename
            np.save(path, masks[key])
            row[f"{key}_mask_path"] = str(path.resolve())
        row["semantic_boundary"] = deepcopy(scene.observation_report["semantic_boundary"][name])
        (folder/"partition-report.json").write_text(json.dumps(row, indent=2)+"\n")
        result[name] = row
    return result
# END diagnostics consume one decision, not a second segmentation algorithm.


# BEGIN typed initial support is a separate recorded declaration channel.
def load_contact_bindings(path):
    """Validate recorded contact input and construct independent support facts.

    Ordinary contact material still binds through the native-material adapter.
    Initial support carries observed plane evidence and an existing interval;
    its entrance geometry and height baseline belong to scene construction.
    """
    from manipulation.initial_support import InitialSupportRelation

    inputs = json.loads(Path(path).read_text())
    required = {"material_faces", "contacts", "source"}
    optional = {"grasp_activations", "holding_activations", "initial_support_relations"}
    if (not isinstance(inputs, dict) or not required <= set(inputs)
            or set(inputs)-required-optional):
        raise ValueError("Contact bindings need material_faces, contacts, source and optional "
                         "grasp_activations/holding_activations/initial_support_relations")
    if not isinstance(inputs["source"], str) or not inputs["source"].strip():
        raise ValueError("Contact bindings require a current evidence source")
    declarations = inputs.get("initial_support_relations", [])
    if not isinstance(declarations, list) or any(not isinstance(row, dict) for row in declarations):
        raise ValueError("Initial support declarations must be a list of typed relation records")
    return inputs, tuple(InitialSupportRelation(**row) for row in declarations)
# END no nominal penetration or source-component inventory is caller supplied.


# BEGIN explicit joint reference is guidance with no saved authorization.
def load_joint_reference(path, problem):
    """Read a bounded named q/dq recording without changing current task goals.

    All coordinates and timestamps come from the supplied record. Shape,
    ordering and finite-value checks prevent malformed guidance from entering
    a native seed fit; goal/domain acceptance still belongs to the backend.
    """
    from manipulation.planning import JointTrajectory
    if path is None:
        return None
    path = Path(path)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Joint reference exceeds its 16 MiB input bound")
    with np.load(path, allow_pickle=False) as record:
        if set(record.files) != {"joint_names", "times", "position", "velocity", "progress"}:
            raise ValueError("Joint reference requires names, times, position, velocity and progress")
        names = tuple(record["joint_names"].tolist())
        t, q, dq, progress = (np.asarray(record[key], float).copy()
                             for key in ("times", "position", "velocity", "progress"))
    if (names != problem.joint_names or t.ndim != 1 or not 2 <= len(t) <= 8192
            or t[0] != 0 or np.any(np.diff(t) <= 0)
            or q.shape != (len(t), len(names)) or dq.shape != q.shape or progress.shape != t.shape
            or np.any(np.diff(progress) <= 0) or progress[0] != problem.start_progress
            or progress[-1] != problem.phase_count
            or not all(np.isfinite(value).all() for value in (t, q, dq, progress))):
        raise ValueError("Joint reference requires finite increasing time/progress and exact named layout")
    return JointTrajectory(names, t, q, dq, t/t[-1], progress, (), (), {})
# END a recorded reference never substitutes for the current measured entrance.


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
                             position_limits, velocity_limits, trajectory_factory, fixture, execution_events=None):
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
    geometry = calibrate_native_hand(robot, feedback, problem, device=device,
                                    position_limits=position_limits, fixture=fixture)
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
        source=f"Native robot FK and synchronous independent position reference: {execution_limits.source}",
        self_collision_geometry=geometry['self_collision_geometry'])
    report = {"affine_fk_max_error_m": geometry["affine_fk_max_error_m"],
        "native_geometry_binding": geometry['native_geometry_binding'],
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


# BEGIN explicit snapshot, owner-mask and hand-permission input contracts.
def require_feedback_identity(feedback):
    """Keep the capture adapter's identity across planning and execution.

    A file path only names storage and can change when a capture is copied.
    It cannot replace the ID already carried by current Holding evidence.
    """
    identity = feedback.get("feedback_id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("The input adapter must supply an explicit current feedback_id")
    return identity


def recorded_owner_observation_inputs(problem, scene_input, views, mask_rows, *, source):
    """Require each selected physical owner in every current RGB-D view.

    Carrier coordinates come from the compiled problem. A reference object's
    coordinates come only from its explicit current scene pose. An invisible
    object still needs an empty boolean mask; a missing mask is not evidence.
    """
    from manipulation.motion import _pose_array
    from recorded_owner_geometry import _problem_binding

    carrier = scene_input["carrier_id"]
    initial, _ = _problem_binding(problem, carrier)
    frames = {carrier: {"T_output_owner": initial, "source": source}}
    if "reference_id" in scene_input or "reference_pose" in scene_input:
        reference = scene_input.get("reference_id")
        if (not isinstance(reference, str) or not reference.strip() or reference == carrier
                or "reference_pose" not in scene_input):
            raise ValueError("A reference partition requires its distinct reference_id and current reference_pose")
        frames[reference] = {"T_output_owner": _pose_array(scene_input["reference_pose"],
                            "current reference-owner pose"), "source": source}
    # BEGIN explicit static observations can participate without becoming goals.
    # These are current collision partitions, not additional task references or
    # moving attachments. Each owns its original RGB-D pixels and a declared
    # coordinate frame. The common mapper retains all remaining background.
    observations = scene_input.get("observation_owners", [])
    if not isinstance(observations, list):
        raise ValueError("Additional observation owners must be an explicit list")
    for record in observations:
        if not isinstance(record, dict) or set(record) != {"owner_id", "T_output_owner", "source"}:
            raise ValueError("An observation owner requires exactly owner_id, T_output_owner and source")
        identity, evidence = record["owner_id"], record["source"]
        if (not isinstance(identity, str) or not identity.strip()
                or identity in frames or identity in {"robot", "background", "observed_background"}
                or not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("Observation owners need distinct physical identities and explicit current sources")
        frames[identity] = {"T_output_owner": _pose_array(record["T_output_owner"],
                            "current observation-owner frame"), "source": evidence}
    # END the mask completeness check below applies equally to every partition.
    view_ids = [view["id"] for view in views]
    if len(set(view_ids)) != len(view_ids) or not view_ids:
        raise ValueError("Current RGB-D views require distinct explicit IDs")
    masks = {}
    for row in mask_rows:
        if len(row) != 3:
            raise ValueError("Each owner mask requires OWNER_ID VIEW_ID MASK_PATH")
        owner, view, path = row
        if (owner, view) in masks:
            raise ValueError("Every current owner/view mask must be supplied exactly once")
        masks[owner, view] = Path(path)
    if set(masks) != {(owner, view) for owner in frames for view in view_ids}:
        raise ValueError("Supply exactly one current owner mask for every explicit owner and view")
    return frames, masks


# END input identities never supply simulated object geometry or contact truth.


def run(args):
    from prepare_recorded_task import prepare_candidate
    from recorded_observation import classify_robot_observation
    from manipulation.robot_self_filter import SELF, ENVIRONMENT, AMBIGUOUS_SHADOW
    from recorded_task_planning import plan_and_record, _write_diagnostic_json
    # BEGIN use the production execution contract shared with the executor.
    from manipulation.execution import ExecutionLimits, FixedFingerTargets
    # END production execution contract.
    from manipulation.planning import MotionBackend, PlanningBudget
    from manipulation.planning_observation import build_scene_from_rgbd
    from curobo.perception import MapperCfg
    from curobo.types import DeviceCfg
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "preparing", "segment_id": args.segment,
              # BEGIN report backend scope even when preparation fails early.
              "backend_scope": "resident_motion_backend",
              "new_prd_production_acceptance": True,
              # END finite-rule input does not change the planner implementation.
              "candidate_id": args.candidate, "planning_called": False,
              "execution_tested": False, "scope": "Observed independent skill snapshot",
              "manifest_path": str(args.manifest.resolve()),
              "feedback_path": str(args.feedback.resolve()),
              "robot_fixture_path": str(args.robot_fixture.resolve())}
    started = time.monotonic()
    # BEGIN wall milestones use Unix time for process joins and monotonic durations locally.
    report["wallclock"] = {"scope": "Saved scene inputs through planning and artifact persistence; external perception is excluded", "events": {}}
    def mark_wallclock(name):
        report["wallclock"]["events"][name] = {
            "unix_time_ns": time.time_ns(), "elapsed_seconds": time.monotonic()-started}
    # END these passive samples do not synchronize CUDA or query geometry.
    try:
        # BEGIN include a retained execution session in this request's host cap.
        # The explicit root must be this process or a real ancestor; standalone
        # runs select themselves. Model/planner subprocesses cannot hide their
        # still-live simulation parent behind a child-only RSS measurement.
        from manipulation.resource_admission import configure_host_process_root
        report["host_process_scope"] = configure_host_process_root(
            os.getpid() if args.host_process_root_pid is None else args.host_process_root_pid)
        # END the same root is used by all nested source-geometry admissions.
        # BEGIN reject impossible module limits before any CUDA preparation.
        if not 0 < args.mapping_memory_mib <= min(args.max_host_rss_mib, args.max_device_used_mib):
            raise ValueError("Mapping capacity must be positive and fit both declared total resource limits")
        # END native maps repeat this admission for callers outside this CLI.
        # BEGIN identify the first saved scene input read, independently of process startup.
        mark_wallclock("scene_input_read_started")
        # END the external runner owns any earlier image acquisition/perception timing.
        manifest = json.loads(args.manifest.read_text())
        root = Path(manifest["workspace_root"])
        segment = next(s for s in manifest["segments"] if s["id"] == args.segment)
        candidate = next(c for c in segment["grounding"]["candidates"] if c["id"] == args.candidate)
        program = json.loads((root/segment["program_path"]).read_text())
        scene_input = json.loads((root/candidate["scene_path"]).read_text())
        fixture = json.loads(args.robot_fixture.read_text())
        feedback = json.loads(args.feedback.read_text())
        # BEGIN preserve the same explicit snapshot identity used by execution.
        require_feedback_identity(feedback)
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
        # BEGIN resolve all owner resolutions while preparation is still on CPU.
        owner_frames, supplied_masks = recorded_owner_observation_inputs(
            problem, scene_input, segment["snapshot"]["views"], args.owner_mask,
            source=str(root/candidate["scene_path"]))
        owner_resolutions = owner_voxel_sizes(args.owner_voxel_size, owner_frames)
        # END subsequent masks, map configurations and reports share these IDs.
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
        if args.torch_memory_mib not in (2048, 4096) or args.scene_gpu_workspace_mib not in (1024, 2048, 3072, 6144):
            raise ValueError("Use only the explicitly bounded local Torch and scene allocations")
        # END process-tree and device-growth guards remain separate run limits.
        # BEGIN use the user-authorized eight CPU threads in this local runner.
        # Native Coal batches join before Python resumes; their workers never
        # call Torch/BLAS, so these pools do not multiply inside a Coal batch.
        torch.set_num_threads(8)
        # END CPU pool size does not change the separate GPU allocation cap.
        torch.cuda.set_per_process_memory_fraction(
            args.torch_memory_mib*1024**2/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        report["resource_limits"] = {"torch_allocator_bytes": args.torch_memory_mib*1024**2,
                                     "torch_cpu_threads": torch.get_num_threads(),
                                     "scene_gpu_workspace_bytes": args.scene_gpu_workspace_mib*1024**2,
                                     "max_host_rss_bytes": args.max_host_rss_mib*1024**2,
                                     "max_device_used_bytes": args.max_device_used_mib*1024**2,
                                     "mapping_bytes": args.mapping_memory_mib*1024**2,
                                     "owner_voxel_sizes_m": owner_resolutions}
        device = DeviceCfg("cuda:0")
        # BEGIN source hand topology from the bound robot fixture, not feedback.
        # Continuous proprioception measures q/dq and names its joints. Static
        # collision-link membership belongs to the already validated fixture;
        # do not require historical feedback files to duplicate this metadata.
        hand, hand_report, entrance_fk = position_controlled_hand(
            robot, {**feedback, "finger_link_names": fixture["finger_link_names"]},
            problem, device=device, execution_limits=execution_limits,
            position_limits=fixture["position_limits"], velocity_limits=fixture["velocity_limits"],
            trajectory_factory=reference_factory, fixture=fixture)
        # END actual measured positions, velocities and identity are unchanged.
        report["gripper"] = hand_report
        # BEGIN finite contact inputs are required for contact activation.
        from robot_fixture import bind_fixture_contact_materials
        from manipulation.contact_binding import build_material_contact_rules
        from recorded_owner_geometry import bind_box_owner_materials
        activations = [row for row in problem.conditions
                       if row.get("kind") in {"grasp_contact_activation", "observed_attachment"}]
        contacts, pair_permissions, initial_support_relations = (), (), ()
        material_binding = {"robot_sphere_materials": [None]*sphere_count,
                            "native_geometry_materials": {}, "materials": {}, "owner_materials": {}}
        prepared = None if args.owner_geometry is None else json.loads(args.owner_geometry.read_text())
        if prepared is not None:
            if prepared["owner_id"] != scene_input["carrier_id"]:
                raise ValueError("Complete geometry must belong to this explicit current carrier")
            owner_binding = bind_box_owner_materials(prepared)
            prepared["sphere_materials"] = owner_binding["sphere_materials"]
            material_binding["owner_materials"][prepared["owner_id"]] = owner_binding["materials"]
            material_binding["native_geometry_materials"].update(owner_binding["native_geometry_materials"])
            # BEGIN retain native feature/cone audits independently of permission.
            # A normal-cone projection never changes the original functional
            # normal or its gap/angle thresholds in the compiled problem.
            report["owner_material_geometry"] = owner_binding
            # END complete local feature evidence survives even without contacts.
        report["contact_binding"] = {"policies": []}
        if args.contact_bindings is not None:
            # BEGIN the parser constructs support relations outside material rules.
            contact_inputs, initial_support_relations = load_contact_bindings(args.contact_bindings)
            # END the scene alone binds initial source geometry and height.
            if contact_inputs["material_faces"]:
                robot_binding = bind_fixture_contact_materials(
                    fixture, robot, entrance_fk["sphere_indices_by_link"], feedback,
                    contact_inputs["material_faces"])
                material_binding["robot_sphere_materials"] = robot_binding["robot_sphere_materials"]
                material_binding["materials"] = robot_binding["materials"]
                material_binding["native_geometry_materials"].update(robot_binding["native_geometry_materials"])
            contacts = build_material_contact_rules(material_binding, contact_inputs["contacts"])
            # BEGIN grasp permissions require explicit existing condition IDs.
            if any(row["kind"] == "grasp_contact_activation" for row in activations):
                from manipulation.contact_binding import bind_grasp_contact_rules
                bind_grasp_contact_rules(problem, contacts, contact_inputs.get("grasp_activations"))
            elif contact_inputs.get("grasp_activations"):
                raise ValueError("Finite grasp activations require compiled grasp conditions")
            if any(row["kind"] == "observed_attachment" for row in activations):
                from manipulation.contact_binding import bind_holding_contact_rules
                bind_holding_contact_rules(problem, contacts, contact_inputs.get("holding_activations"))
            elif contact_inputs.get("holding_activations"):
                raise ValueError("Finite Holding activations require current attachment obligations")
            # END a finite patch cannot silently become an ungated grasp rule.
            report["contact_binding"]["policies"].append("explicit_finite_material_contacts")
            # BEGIN diagnostics distinguish initial support from exact materials.
            if initial_support_relations:
                report["contact_binding"]["policies"].append("explicit_initial_planar_support")
            # END this label does not change collision activation or Holding facts.
            report["contact_binding"].update(path=str(args.contact_bindings.resolve()),
                source=contact_inputs["source"], materials=material_binding,
                rule_ids=[rule.rule_id for rule in contacts],
                initial_support_relation_ids=[row.relation_id for row in initial_support_relations])
        if any(row.get("kind") == "grasp_contact_activation" and "contact_rule_ids" not in row
               for row in problem.conditions):
            raise ValueError("Grasp activation requires explicit finite material contact bindings")
        # END no whole-hand permission is manufactured from a grasp condition.
        report["requirement_consumption"] = problem.requirement_consumption
        # END permissions affect software collision legality, never native physical contact.
        report["owner_observations"] = {
            "owners": {identity: {"initial_pose": frame["T_output_owner"].tolist(),
                "mask_paths": {view["id"]: str(supplied_masks[identity, view["id"]].resolve())
                               for view in segment["snapshot"]["views"]}}
                for identity, frame in owner_frames.items()},
            "geometry_path": None if args.owner_geometry is None else str(args.owner_geometry.resolve()),
            # BEGIN only the production partition decision admits background endpoints.
            "background_partition": "ENVIRONMENT endpoints outside semantic owners and the production semantic boundary mask",
            "background_rebuild": "Fresh Mapper and scene from original RGB-D for this request; no prior fused background is restored"}
            # END historical scene bindings are not observation inputs.
        observations = {}
        for view in segment["snapshot"]["views"]:
            calibration = json.loads((root/view["calibration_path"]).read_text())
            rgb = np.array(Image.open(root/view["rgb_path"]).convert("RGB"))
            depth = np.load(root/view["depth_path"], allow_pickle=False)
            # BEGIN classify native robot solids and their uncertain observation rays.
            filtered = classify_robot_observation(
                scene_xml=Path(fixture["source_scene_xml"]), robot_root=fixture["robot_visual_root"],
                joint_map=[(name, name, 1.) for name in feedback["joint_names"]],
                joint_positions=measured, T_world_robot_root=(np.asarray(feedback["base_pose"])
                    @ np.asarray(fixture["T_base_visual_root"])),
                observed_depth=depth, calibration=calibration,
                padding_scale=args.self_filter_padding_scale,
                padding_offset_m=args.self_filter_padding_offset_m,
                source=args.self_filter_source)
            labels = filtered["self_filter"]["labels"]
            view_output = args.output/"observations"/view["id"]
            view_output.mkdir(parents=True)
            np.save(view_output/"self-filter-labels.npy", labels)
            np.save(view_output/"free-ray-depth-m.npy", filtered["self_filter"]["free_ray_depth_m"])
            filter_report = {"provenance": filtered["provenance"],
                "observation_id": view["id"], "feedback_source": feedback["source"],
                "counts": {"SELF": int((labels == SELF).sum()),
                    "ENVIRONMENT": int((labels == ENVIRONMENT).sum()),
                    "AMBIGUOUS_SHADOW": int((labels == AMBIGUOUS_SHADOW).sum())}}
            (view_output/"self-filter-report.json").write_text(json.dumps(filter_report, indent=2)+"\n")
            # Save resolved masks for inspection; pass the original semantic masks
            # to the mapper so one contract checks ownership and counts removals.
            # Only ENVIRONMENT contributes either occupied endpoints or free rays.
            owner_masks, mask_report = {}, {}
            for index, identity in enumerate(owner_frames):
                supplied = np.load(supplied_masks[identity, view["id"]], allow_pickle=False)
                if supplied.dtype != np.bool_ or supplied.shape != depth.shape:
                    raise ValueError("Semantic owner masks must be aligned boolean images")
                visible = supplied & (labels == ENVIRONMENT)
                resolved_path = view_output / f"owner-{index:03d}-mask.npy"
                np.save(resolved_path, visible)
                owner_masks[identity] = supplied
                mask_report[identity] = {
                    # BEGIN distinguish unchanged segmentation from robot-filtered owner input.
                    "semantic_mask_path": str(supplied_masks[identity, view["id"]].resolve()),
                    "collision_mask_path": str(resolved_path.resolve()),
                    "semantic_mask_role": "Original segmentation input, retained without dilation",
                    "collision_mask_role": "Original segmentation intersected with ENVIRONMENT; the builder additionally enforces Mapper depth limits. Semantic boundary is never assigned to this owner",
                    # END the builder changes background admission, not owner identity.
                    "semantic_pixels": int(supplied.sum()),
                    "self_overlap_pixels": int((supplied & (labels == SELF)).sum()),
                    "ambiguous_shadow_overlap_pixels": int((supplied & (labels == AMBIGUOUS_SHADOW)).sum()),
                    "collision_owner_pixels": int(visible.sum())}
            (view_output/"owner-mask-report.json").write_text(json.dumps({
                # BEGIN semantic uncertainty is distinct from robot and ray evidence.
                "policy": "Owner masks retain ENVIRONMENT endpoints; the builder separately records background and semantic-boundary masks. Free rays retain their independent measured prefixes; original depth and semantic masks are unchanged",
                # END excluding an endpoint cannot extend its measured free ray.
                "robot_evidence": str((view_output/"self-filter-report.json").resolve()),
                "owners": mask_report}, indent=2)+"\n")
            observations[view["id"]] = {
                "image": rgb, "depth": depth, "K": np.asarray(calibration["K"]),
                "T_output_camera": np.asarray(calibration["T_world_camera_cv"]),
                "reference_frame": problem.reference_frame,
                "self_filter": filtered["self_filter"],
                "owner_masks": owner_masks}
            # END unknown robot shells and shadows remain unknown in both maps.
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
        # BEGIN install declared complete geometry before any observed map allocation.
        # The carrier keeps its semantic masks and current frame in observations:
        # those pixels still exclude it from background and supply free-ray evidence.
        # Only its redundant RGB-D distance field is omitted by the shared builder.
        complete_owners = ()
        if prepared is not None:
            from recorded_owner_geometry import create_box_owner
            complete_owners = (create_box_owner(
                prepared, contact_voxel_size_m=owner_resolutions[prepared["owner_id"]]),)
            report["complete_carrier_geometry"] = prepared["provenance"]
        # END primitive geometry, native registration and attachments stay unchanged.
        scene = build_scene_from_rgbd(
            observations, owner_frames, mapper_cfg=config,
            depth_error_m=args.depth_error, reference_frame=problem.reference_frame,
            robot_sphere_materials=tuple(material_binding["robot_sphere_materials"]),
            robot_geometry_source=fixture["robot_geometry_source"], gripper=hand,
            # BEGIN bind every finite rule to the final owners in one scene.
            # Native owner material validation runs during initial construction;
            # no partial carrier scene or temporary contact subset is created.
            complete_owners=complete_owners, contacts=contacts,
            # BEGIN initial support reaches the same production scene as materials.
            initial_support_relations=initial_support_relations,
            # END no recorded adapter substitutes a future placement ContactRule.
            # END owner-actor and robot-actor rules share the same validated scene.
            native_geometry_materials=material_binding["native_geometry_materials"],
            # BEGIN both independent and planner models retain the exact pair policy.
            robot_owner_permissions=pair_permissions,
            # END a current measured arrival is still required by execution.
            # BEGIN distinct physical owners share one total resource envelope.
            owner_voxel_sizes_m=owner_resolutions,
            fine_regions=fine_regions, robot_collision_bodies=collision_bodies,
            unknown_policy=json.loads(args.unknown_policy.read_text()),
            max_mapping_bytes=args.mapping_memory_mib*1024**2,
            max_host_rss_bytes=args.max_host_rss_mib*1024**2,
            max_device_used_bytes=args.max_device_used_mib*1024**2)
            # END module capacity cannot bypass the measured host/device limits.
        report["observation_build"] = scene.observation_report
        # BEGIN save the exact production masks before fixed-owner scene reconstruction.
        report["observation_partitions"] = save_observation_partitions(scene, args.output)
        # END no reconstructed scene needs to retain these diagnostic arrays.
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
            observed_report = scene.observation_report
            scene = type(scene)(
                background=scene.background, owners=(*scene.owners, *fixed_owners), coverage=scene.coverage,
                reference_frame=scene.reference_frame, unknown_policy=scene.unknown_policy,
                robot_sphere_materials=scene.robot_sphere_materials,
                robot_collision_bodies=scene.robot_collision_bodies,
                robot_geometry_source=scene.robot_geometry_source, gripper=hand, contacts=scene.contacts,
                # BEGIN installation augmentation preserves initial support identity.
                initial_support_relations=scene.initial_support_relations,
                # END reconstructing a scene cannot discard its support contract.
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
        # BEGIN persist observed-free evidence without source-volume coverage.
        np.savez_compressed(args.output/"observed-free.npz", known_free=scene.coverage.known_free,
            origin=scene.coverage.origin, voxel_size=np.asarray(scene.coverage.voxel_size),
            observation_ids=np.asarray(scene.coverage.observation_ids))
        report["unknown_policy"] = scene.unknown_policy
        # END conditional unknown permission is distinct from observed freedom.
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
        # A supplied ledger freezes the work/model dimensions. Physical memory
        # limits may change explicitly while the ledger retains all spent work.
        budget = PlanningBudget(knots=args.knots, interpolation_buffer_size=args.interpolation_buffer_size,
                                max_robot_spheres=640, ik_seeds=args.ik_seeds,
                                trajectory_seeds=args.trajectory_seeds,
                                # BEGIN recorded requests expose the existing bounded iteration budget.
                                ik_iterations=80, trajectory_iterations=args.trajectory_iterations,
                                # END native work and the persisted request ledger use the same value.
                                max_validation_points=args.max_validation_points,
                                max_task_anchors=args.max_task_anchors,
                                max_scene_gpu_workspace_bytes=args.scene_gpu_workspace_mib*1024**2,
                                max_host_rss_bytes=args.max_host_rss_mib*1024**2,
                                max_device_used_bytes=args.max_device_used_mib*1024**2,
                                # BEGIN expose the existing cumulative work contract.
                                # The supplied ledger must match these values;
                                # flags cannot reset or enlarge it implicitly.
                                maximum_trajectory_solves=args.maximum_trajectory_solves,
                                # BEGIN use the existing explicit search budget input.
                                planning_seconds=args.planning_seconds, search_seconds=args.search_seconds)
        # END this caller cannot bypass the total planning time or ledger contract.
        # END geometry capacity does not replenish planning time or solve counts.
        # END reviewed input revisions remain caller-owned.
        report["planning_called"] = True
        # BEGIN model initialization remains inside the planner's lazy bind_request.
        mark_wallclock("scene_preparation_completed")
        mark_wallclock("planning_call_started")
        # END constructing MotionBackend alone is not native initialization.
        # BEGIN the caller owns resident native models across motion requests.
        backend = MotionBackend(robot, robot_validator=robot_validator, budget=budget)
        try:
            result = plan_and_record(problem, scene, robot, args.output/"planning",
                robot_validator=robot_validator, budget=budget, backend=backend,
                execution_limits=execution_limits.describe(),
                joint_reference=load_joint_reference(args.joint_reference, problem),
                continuation=continuation, rebind=rebind, request_ledger=request_ledger,
                provenance={**provenance, "robot_fixture": str(args.robot_fixture),
                    "current_views": list(observations),
                    "old_selection_comparison_only": segment["grounding"]["status"]})
            # BEGIN record readiness before native cleanup, after artifact persistence.
            mark_wallclock("planning_result_saved")
            report["planner_wallclock"] = result.diagnostics.get("wallclock")
            # END a returned failure is timed without claiming an executable result.
        finally:
            backend.close()
        report.update(status="completed", planning_success=result.success,
                      initial_state=result.diagnostics.get("initial_state"),
                      failure=result.diagnostics.get("failure"))
        # END persistence cannot recreate a planner or initiate validation.
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
        _write_diagnostic_json(args.output/"run-report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # BEGIN grasp permission comes from arrival/close, with local contacts separately explicit.
    for name in ("manifest", "robot-fixture", "feedback", "execution-limits", "finger-targets", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--contact-bindings", type=Path,
                        help="Explicit finite robot/owner material contact policy; current Holding permissions remain separate")
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
    parser.add_argument("--owner-mask", nargs=3, action="append", required=True,
                        metavar=("OWNER_ID", "VIEW_ID", "MASK_PATH"))
    parser.add_argument("--owner-geometry", type=Path)
    parser.add_argument("--center", type=float, nargs=3, required=True)
    parser.add_argument("--extent", type=float, nargs=3, required=True)
    parser.add_argument("--voxel-size", type=float, required=True)
    # BEGIN current owner precision is declared before a planning result exists.
    parser.add_argument("--owner-voxel-size", nargs=2, action="append", required=True,
                        metavar=("OWNER_ID", "METRES"),
                        help="Repeat exactly once for each observed owner; no shared owner resolution")
    # BEGIN freeze reviewed local bounds, resolution and measurement evidence.
    parser.add_argument("--fine-regions", type=Path,
                        help="JSON array of owner_id (null for background), local bounds and measured-allowance evidence")
    # END the same generic ROI input is accepted for any observed partition.
    # END the mapper retains every current labelled sample in the refined field.
    parser.add_argument("--depth-error", type=float, required=True)
    # BEGIN explicit geometric uncertainty, independent of measured depth error.
    parser.add_argument("--self-filter-padding-offset-m", type=float, required=True)
    parser.add_argument("--self-filter-padding-scale", type=float, required=True)
    parser.add_argument("--self-filter-source", required=True)
    # END no rendered depth-equality threshold is accepted.
    parser.add_argument("--unknown-policy", type=Path, required=True,
                        help="Explicit bounded conditional unknown regions; outside them unknown is forbidden")
    parser.add_argument("--joint-reference", type=Path,
                        help="Optional named recorded q/dq curve; guidance has no execution authorization")
    parser.add_argument("--knots", type=int, default=16)
    parser.add_argument("--interpolation-buffer-size", type=int, default=768)
    parser.add_argument("--ik-seeds", type=int, default=4)
    parser.add_argument("--trajectory-seeds", type=int, default=2)
    # BEGIN a caller can vary work without changing constraints or native internals.
    parser.add_argument("--trajectory-iterations", type=int, default=100,
                        help="Trajectory iteration budget; existing PlanningBudget limit is 1..300")
    # END the default and PlanningBudget admission remain unchanged.
    parser.add_argument("--max-validation-points", type=int, default=2048)
    parser.add_argument("--max-task-anchors", type=int, default=256)
    # BEGIN make the existing request work limits reviewable in run commands.
    parser.add_argument("--maximum-trajectory-solves", type=int, default=2)
    parser.add_argument("--planning-seconds", type=float, default=120.)
    # BEGIN expose bounded OMPL time without changing the production default.
    parser.add_argument("--search-seconds", type=float, default=3.,
                        help="OMPL search time within the existing total planning budget")
    # END PlanningBudget rejects negative or nonfinite search durations.
    # END the production PlanningBudget and ledger checks validate these inputs.
    # BEGIN scene tensor capacity excludes all CPU geometry and its reservations.
    parser.add_argument("--scene-gpu-workspace-mib", type=int, choices=(1024, 2048, 3072, 6144), default=1024)
    # END this separate capacity is not the physical device-used limit.
    # BEGIN request-level physical limits remain explicit in standalone runs.
    parser.add_argument("--max-host-rss-mib", type=int, choices=(8192, 10240, 12288), default=8192)
    parser.add_argument("--max-device-used-mib", type=int, choices=(6144, 8192), default=6144)
    parser.add_argument("--mapping-memory-mib", type=int, default=1024,
                        help="Explicit combined map-buffer estimate cap, also checked against total live host/device budgets")
    parser.add_argument("--host-process-root-pid", type=int,
                        help="Existing ancestor owning the retained execution session; omitted for standalone planning")
    # END defaults preserve the original half-host/half-device limits.
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
