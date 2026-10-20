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


# BEGIN persist reusable semantic masks separately from fusion decisions.
def save_observation_partitions(scene, output):
    """Save original entity/background masks and the admitted fusion mask.

    Future --owner-mask inputs use the saved semantic entity files and
    --background-mask uses the saved background candidate. The fusion
    background records this run's filtering and one-pixel boundary; feeding
    it back as a candidate would incorrectly grow the excluded boundary.
    A reused background field produces no new fusion-background mask.
    """
    result = {}
    for name, masks in scene.observation_partition_masks.items():
        folder = Path(output)/"observations"/name
        folder.mkdir(parents=True, exist_ok=True)
        row = {'owner_masks': {}, 'background_candidate_mask_path': None,
               'background_mask_path': None,
               'mask_roles': {
                   'owner_masks': 'Original semantic entity masks; reusable --owner-mask inputs',
                   'background_candidate_mask_path': 'Original semantic background candidate; reusable --background-mask input',
                   'background_mask_path': 'Actual fusion mask after robot filtering and one boundary exclusion; not a semantic input',
               },
               'endpoint_partition':deepcopy(scene.observation_report['endpoint_partition'][name])}
        for index, (identity, mask) in enumerate(masks['owner_masks'].items()):
            path = folder/f"owner-{index:03d}-semantic-mask.npy"
            np.save(path, mask)
            row['owner_masks'][identity] = str(path.resolve())
        for key, filename, report_key in (
                ('background_candidate', 'background-candidate-mask.npy', 'background_candidate_mask_path'),
                ('background', 'background-mask.npy', 'background_mask_path')):
            if masks[key] is not None:
                path = folder/filename
                np.save(path, masks[key])
                row[report_key] = str(path.resolve())
        (folder/"partition-report.json").write_text(json.dumps(row, indent=2)+"\n")
        result[name] = row
    return result
# END saved original masks preserve ownership without another segmentation call.


# BEGIN typed initial support is a separate recorded declaration channel.
def load_contact_bindings(path):
    """Validate recorded contact input and construct independent support facts.

    Ordinary contact features bind through explicit source task declarations.
    Initial support carries observed plane evidence and an existing interval;
    its entrance geometry and height baseline belong to scene construction.
    """
    from manipulation.initial_support import InitialSupportRelation

    inputs = json.loads(Path(path).read_text())
    required = {"materials", "contacts", "source"}
    optional = {"grasp_activations", "holding_contacts", "initial_support_relations"}
    if (not isinstance(inputs, dict) or not required <= set(inputs)
            or set(inputs)-required-optional):
        raise ValueError("Contact bindings need materials, contacts, source and optional "
                         "grasp_activations/holding_contacts/initial_support_relations")
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
    # BEGIN geometry uses the already-created command and original bound clock.
    def nominal_geometry(duration_s, progress_map, command_reference):
        from manipulation.gripper_trajectory import bind_nominal_finger_geometry
        if command_reference.duration != duration_s:
            raise ValueError("Nominal geometry and command must share the original bound duration")
        return bind_nominal_finger_geometry(command_reference, problem.conditions,
            progress_map.event_clock, progress_interval=(progress_map.s[0], progress_map.s[-1]))
    # END measured Holding shape never overwrites the actual command q/dq.
    hand = PrismaticGripper(
        controller="timed_reference", joint_names=finger_names,
        current_positions=actual_fingers, joint_limits=[position_limits[name] for name in finger_names],
        joint_bounds=None, finger_trajectory_factory=trajectory_factory,
        nominal_finger_trajectory_factory=nominal_geometry,
        control_timestep_s=trajectory_factory.timestep,
        coupling_matrix=np.empty((0, len(finger_names))), coupling_offset=np.empty(0),
        commands=tuple({"event_id": event.event_id, "skill_id": event.skill_id,
                        "occurrence": event.occurrence, "command": event.kind, "progress": event.progress}
                       for event in execution_events if event.kind in {"open", "close", "hold"}),
        source=f"Measured fingers and synchronous command reference: {execution_limits.source}")
    report = {"current_fingers": actual_fingers.tolist(),
        "mechanical_limits": [position_limits[name] for name in finger_names],
        "controller": "timed_reference", "execution_limits": execution_limits.describe(),
        "geometry_scope": "Named nominal finger coordinates feed the complete robot FK; observations stay link-local"}
    return hand, report
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


def recorded_owner_observation_inputs(problem, scene_input, views, mask_rows, *, source, observed_owners=()):
    """Resolve masks only for entities selected for RGB-D reconstruction.

    Existing models retain their registered identity and motion. They do not
    participate in the new observation-mask contract, including under complete
    occlusion. New entity coordinates come from the explicit task binding.
    """
    from manipulation.motion import _pose_array
    from observed_robot_fixture import problem_owner_binding

    carrier = scene_input["carrier_id"]
    initial, _ = problem_owner_binding(problem, carrier)
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
    # BEGIN existing entities are geometry inputs, independently of visibility.
    existing = {owner.owner_id for owner in observed_owners}
    if not existing <= set(frames):
        raise ValueError("Existing models must identify entities in the task scene")
    frames = {identity: frame for identity, frame in frames.items() if identity not in existing}
    # END only entities being reconstructed need observation masks.
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


# BEGIN execution consumes an explicitly prepared current problem.
def run_prepared(args, *, problem, provenance):
    """Run the existing scene/solver pipeline on the supplied continuous path.

    Problem construction belongs to the caller. The actual feedback is checked
    again here, so an offline reference cannot replace current robot state.
    This function never reconstructs or resamples the supplied reference.
    """
    from manipulation.robot_self_filter import classify_depth
    from manipulation.robot_self_filter import SELF, ENVIRONMENT, AMBIGUOUS_SHADOW
    from recorded_task_planning import plan_and_record, _write_diagnostic_json
    # BEGIN use the production execution contract shared with the executor.
    from manipulation.execution import ExecutionLimits, FixedFingerTargets
    # END production execution contract.
    from manipulation.planning import MotionBackend, PlanningBudget
    from manipulation.planning_observation import (
        build_scene_from_rgbd, scene_task_owner_ids, observation_resolutions)
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
        # BEGIN report completed preparation boundaries during long field loads.
        print(f"[planning preparation] {name}: {time.monotonic()-started:.3f}s", flush=True)
        # END logging carries no planner state or decision.
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
        # BEGIN prepared requests must retain this snapshot's exact entrance.
        if (tuple(problem.joint_names) != active
                or not np.array_equal(problem.start_q, planning_feedback["start_q"])
                or not np.array_equal(problem.start_dq, planning_feedback["start_dq"])):
            raise ValueError("Prepared problem differs from current named robot q/dq")
        for field in ("tool_link", "reference_frame"):
            if getattr(problem, field) != feedback[field]:
                raise ValueError(f"Prepared problem disagrees with feedback {field}")
        for field in ("base_pose", "tool_transform"):
            if not np.allclose(getattr(problem, field), feedback[field], atol=1e-10, rtol=0):
                raise ValueError(f"Prepared problem disagrees with calibrated {field}")
        if {phase.metadata.get("skill_id") for phase in problem.phases} != {args.segment}:
            raise ValueError("Prepared problem must identify this current skill")
        # END the caller's exact PosePath object reaches plan_and_record.
        # The supplied camera calibration is explicitly world-from-camera.
        # Another output frame needs its real calibrated transform, never a
        # relabelled world observation at this narrow recorded-test boundary.
        if problem.reference_frame != "world":
            raise ValueError("Recorded world camera calibration requires a world-frame planning problem")
        # BEGIN load existing geometry before deciding which observations to build.
        from observed_robot_fixture import load_observed_owner
        from robot_validation_fixture import build_robot_validator, robot_self_filter_shapes
        mark_wallclock("existing_owner_load_started")
        observed_owners = (() if args.observed_owner is None else
                           (load_observed_owner(args.observed_owner),))
        mark_wallclock("existing_owner_load_completed")
        owner_frames, supplied_masks = recorded_owner_observation_inputs(
            problem, scene_input, segment["snapshot"]["views"], args.owner_mask,
            source=str(root/candidate["scene_path"]), observed_owners=observed_owners)
        task_owners = scene_task_owner_ids(scene_input)
        owner_resolutions = observation_resolutions((*owner_frames, *(owner.owner_id for owner in observed_owners)),
            task_owner_ids=task_owners, environment_voxel_size=args.voxel_size)
        owner_resolutions.update({owner.owner_id: owner.query.voxel_size for owner in observed_owners})
        background_masks = {name: Path(path) for name, path in args.background_mask}
        if len(background_masks) != len(args.background_mask):
            raise ValueError("Each background partition must name one observation")
        # END retained entities keep their original samples and resolution.
        # END subsequent masks, map configurations and reports share these IDs.
        robot = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())
        mark_wallclock("native_robot_load_started")
        # BEGIN known native geometry is independent of the RGB-D task fields.
        robot_validator = build_robot_validator(fixture,
            reference_frame=problem.reference_frame, T_reference_base=problem.base_pose,
            tool_frame=problem.tool_link, T_frame_tcp=problem.tool_transform)
        mark_wallclock("native_robot_load_completed")
        report["independent_robot_model"] = robot_validator.describe()
        robot["robot_cfg"]["kinematics"]["lock_joints"] = {
            name: measured[name] for name in feedback["finger_joint_names"]}
        reference_factory = finger_trajectory_factory(problem, feedback, finger_targets,
            timestep=fixture["control_timestep_s"], ctrlrange=fixture["finger_control_ranges"],
            position_limits=fixture["position_limits"], velocity_limits=fixture["velocity_limits"],
            acceleration_limits={name: float(robot["robot_cfg"]["kinematics"]["cspace"]["max_acceleration"])
                                 for name in feedback["finger_joint_names"]})
        entrance_fk = robot_validator.checker.query(joint_names=feedback["joint_names"],
            joint_positions=feedback["start_q"], T_reference_base=feedback["base_pose"], pairs=(),
            frames=(problem.tool_link,))
        # END current joint feedback positions every observed robot component.
        if not torch.cuda.is_available():
            raise RuntimeError("This explicit local planner test requires CUDA access")
        # BEGIN expose the production scene budget without raising Torch limits.
        if args.torch_memory_mib not in (2048, 4096) or args.scene_gpu_workspace_mib not in (1024, 2048, 3072, 6144):
            raise ValueError("Use only the explicitly bounded local Torch and scene allocations")
        # END process-tree and device-growth guards remain separate run limits.
        # BEGIN use the user-authorized eight CPU threads in this local runner.
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
        hand, hand_report = position_controlled_hand(robot, feedback, problem, device=device,
            execution_limits=execution_limits, position_limits=fixture["position_limits"],
            velocity_limits=fixture["velocity_limits"], trajectory_factory=reference_factory, fixture=fixture)
        report["gripper"] = hand_report
        contacts, initial_support_relations, holding_contacts = (), (), ()
        report["contact_binding"] = {"policies": []}
        if args.contact_bindings is not None:
            from manipulation.planning_scene import ContactRule
            from manipulation.holding_contact import bind_holding_contacts
            contact_inputs, initial_support_relations = load_contact_bindings(args.contact_bindings)
            contacts = tuple(ContactRule(**row) for row in contact_inputs["contacts"])
            if any(row.get("kind") == "grasp_contact_activation" for row in problem.conditions):
                from manipulation.contact_binding import bind_grasp_contact_rules
                bind_grasp_contact_rules(problem, contacts, contact_inputs["grasp_activations"])
            if any(row.get("kind") == "declared_attachment" for row in problem.conditions):
                holding_contacts = bind_holding_contacts(problem,
                    {"materials": contact_inputs["materials"]}, contact_inputs["holding_contacts"],
                    source=contact_inputs["source"])
            report["contact_binding"] = {"source": contact_inputs["source"],
                "rule_ids": [row.rule_id for row in contacts],
                "holding_contacts": [asdict(row) for row in holding_contacts],
                "initial_support_relation_ids": [row.relation_id for row in initial_support_relations]}
        report["requirement_consumption"] = problem.requirement_consumption
        # END permissions affect software collision legality, never native physical contact.
        # BEGIN describe each model by its actual construction input.
        report["owner_observations"] = {
            "new_entities": {identity: {"observation_pose": frame["T_output_owner"].tolist(),
                "mask_paths": {view["id"]: str(supplied_masks[identity, view["id"]].resolve())
                               for view in segment["snapshot"]["views"]}}
                for identity, frame in owner_frames.items()},
            "existing_entities": {owner.owner_id: owner.query.summary() for owner in observed_owners},
            "geometry_path": None if args.observed_owner is None else str(args.observed_owner.resolve()),
            "background_partition": {name: str(path) for name, path in background_masks.items()},
            "observation_source": segment["snapshot"]}
        capture_path = segment["snapshot"]["feedback_path"]
        capture = json.loads((root/capture_path).read_text())
        capture_fk = robot_validator.checker.query(joint_names=capture["joint_names"],
            joint_positions=capture["start_q"], T_reference_base=capture["base_pose"], pairs=(), frames=())
        filter_geometries = robot_self_filter_shapes(fixture, capture_fk["frame_poses"])
        # END image filtering uses encoder feedback at image capture, not a later task entrance.
        observations = {}
        for view in segment["snapshot"]["views"]:
            calibration = json.loads((root/view["calibration_path"]).read_text())
            rgb = np.array(Image.open(root/view["rgb_path"]).convert("RGB"))
            depth = np.load(root/view["depth_path"], allow_pickle=False)
            # BEGIN nominal native membership and padded shadow have separate roles.
            filtered = classify_depth(observed_depth=depth, K=np.asarray(calibration["K"]),
                T_world_camera=np.asarray(calibration["T_world_camera_cv"]),
                geometries=filter_geometries, padding_scale=args.self_padding_scale,
                padding_offset_m=args.self_padding_offset_m, source=args.self_filter_source)
            # END raw measured depth is unchanged by robot observation filtering.
            labels = filtered["self_filter"]["labels"]
            view_output = args.output/"observations"/view["id"]
            view_output.mkdir(parents=True)
            np.save(view_output/"self-filter-labels.npy", labels)
            np.save(view_output/"free-ray-depth-m.npy", filtered["self_filter"]["free_ray_depth_m"])
            filter_report = {"provenance": filtered["provenance"],
                "observation_id": view["id"], "feedback_source": capture["source"],
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
            # BEGIN source background partitions can exclude retained entities.
            if view["id"] in background_masks:
                observations[view["id"]]["background_mask"] = np.load(background_masks[view["id"]], allow_pickle=False)
            # END this input does not supply a new model or pose for a retained entity.
            # END unknown robot shells and shadows remain unknown in both maps.
        # The finite grid is an explicit test resource limit. Space outside it
        # stays unknown; narrowing it never certifies a removed obstacle.
        height, width = next(iter(observations.values()))["depth"].shape
        config = MapperCfg(extent_meters_xyz=tuple(args.extent), voxel_size=args.voxel_size,
                           extent_esdf_meters_xyz=tuple(args.extent), esdf_voxel_size=args.voxel_size,
                           grid_center=torch.tensor(args.center), truncation_distance=2*args.voxel_size,
                           image_height=height, image_width=width, depth_minimum_distance=.05,
                           depth_maximum_distance=5., device="cuda:0")
        fine_regions = {}
        if args.fine_regions is not None:
            requests = json.loads(args.fine_regions.read_text())
            for row in requests:
                identity = row["owner_id"]
                if identity in fine_regions:
                    raise ValueError("Only one bounded fine ROI is admitted per physical owner partition")
                fine_regions[identity] = {key: value for key, value in row.items() if key != "owner_id"}
        # END no simulator obstacle pose or task-specific crop enters the mapper.
        scene = build_scene_from_rgbd(observations, owner_frames, mapper_cfg=config,
            depth_error_m=args.depth_error, reference_frame=problem.reference_frame,
            robot_model=robot_validator.checker,
            fixed_geometry_names=tuple(row["geom"] for row in fixture["fixed_collision_geometries"]),
            gripper=hand,
            observed_owners=observed_owners, contacts=contacts, holding_contacts=holding_contacts,
            initial_support_relations=initial_support_relations, task_owner_ids=task_owners,
            fine_regions=fine_regions, unknown_policy=json.loads(args.unknown_policy.read_text()),
            max_mapping_bytes=args.mapping_memory_mib*1024**2,
            max_host_rss_bytes=args.max_host_rss_mib*1024**2,
            max_device_used_bytes=args.max_device_used_mib*1024**2)
        report["observation_build"] = scene.observation_report
        # BEGIN save the exact production masks before fixed-owner scene reconstruction.
        report["observation_partitions"] = save_observation_partitions(scene, args.output)
        # END no reconstructed scene needs to retain these diagnostic arrays.
        report["fixed_robot_installation"] = {
            "representation": "explicit native installation entities",
            "geometry_names": list(scene.fixed_geometry_names)}
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
            body_names=np.asarray(list(entrance_fk["geometry_poses"])),
            body_poses=np.asarray(list(entrance_fk["geometry_poses"].values())),
            joint_names=np.asarray(feedback["joint_names"]),
            actual_q=np.asarray(feedback["start_q"]), actual_dq=np.asarray(feedback["start_dq"]))
        report["entrance_geometry_diagnostic"] = {
            "path": "entrance-scene.npz", "body_count": len(robot_validator.checker.geometry_names),
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


def run(args):
    """Compile recorded program/scene inputs once, then run the shared request."""
    from prepare_recorded_task import prepare_candidate
    from recorded_task_planning import _write_diagnostic_json
    try:
        manifest = json.loads(args.manifest.read_text())
        root = Path(manifest["workspace_root"])
        segment = next(row for row in manifest["segments"] if row["id"] == args.segment)
        candidate = next(row for row in segment["grounding"]["candidates"] if row["id"] == args.candidate)
        feedback = json.loads(args.feedback.read_text())
        q = dict(zip(feedback["joint_names"], feedback["start_q"]))
        dq = dict(zip(feedback["joint_names"], feedback["start_dq"]))
        active = feedback["arm_joint_names"]
        selected = {**feedback, "joint_names": active,
                    "start_q": [q[name] for name in active], "start_dq": [dq[name] for name in active]}
        problem, provenance = prepare_candidate(
            json.loads((root/segment["program_path"]).read_text()),
            json.loads((root/candidate["scene_path"]).read_text()), selected, args.segment)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=False)
        _write_diagnostic_json(args.output/"run-report.json", dict(status="error",
            planning_called=False, failed_stage="compile_recorded_problem",
            error=f"{type(error).__name__}: {error}"))
        (args.output/"exception.txt").write_text(traceback.format_exc())
        raise
    return run_prepared(args, problem=problem, provenance=provenance)
# END both input producers use the same scene, planner and result writer.


def parse_args(argv=None):
    """Share the existing resource/scene CLI without a second option schema."""
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
    # BEGIN reconstruction masks and retained models are separate input choices.
    parser.add_argument("--owner-mask", nargs=3, action="append", default=[],
                        metavar=("OWNER_ID", "VIEW_ID", "MASK_PATH"))
    parser.add_argument("--background-mask", nargs=2, action="append", default=[],
                        metavar=("VIEW_ID", "MASK_PATH"))
    # END existing entity models do not require placeholder masks.
    parser.add_argument("--observed-owner", type=Path)
    parser.add_argument("--center", type=float, nargs=3, required=True)
    parser.add_argument("--extent", type=float, nargs=3, required=True)
    # BEGIN only the coarser environment spacing remains an experiment input.
    from manipulation.planning_observation import DEFAULT_ENVIRONMENT_VOXEL_SIZE_M
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_ENVIRONMENT_VOXEL_SIZE_M,
                        help="Environment grid spacing in metres; task owners use the library's 2 mm policy")
    # END no per-owner CLI override can silently coarsen a task object.
    # BEGIN freeze reviewed local bounds, resolution and measurement evidence.
    parser.add_argument("--fine-regions", type=Path,
                        help="JSON array of owner_id (null for background), local bounds and measured-allowance evidence")
    # END the same generic ROI input is accepted for any observed partition.
    # END the mapper retains every current labelled sample in the refined field.
    parser.add_argument("--depth-error", type=float, required=True)
    # BEGIN explicit geometric uncertainty, independent of measured depth error.
    parser.add_argument("--self-filter-source", required=True)
    # BEGIN explicit model shadow padding never expands nominal collision geometry.
    parser.add_argument("--self-padding-scale", type=float, required=True)
    parser.add_argument("--self-padding-offset-m", type=float, required=True)
    # END recorded runs state the same source padding used for their image filter.
    # END no rendered depth-equality threshold is accepted.
    # BEGIN the caller declares the unknown-space action explicitly.
    parser.add_argument("--unknown-policy", type=Path, required=True,
                        help="Explicit unknown policy: conditional_regions and outside_regions (forbid, require_observation, allow_attempt)")
    # END observation evidence is separate from the policy that uses it.
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
    # BEGIN explicit requests may use the authorized ten-GiB device envelope.
    parser.add_argument("--max-device-used-mib", type=int, choices=(6144, 8192, 10240), default=6144)
    # END the default request remains six GiB.
    parser.add_argument("--mapping-memory-mib", type=int, default=1024,
                        help="Explicit combined map-buffer estimate cap, also checked against total live host/device budgets")
    parser.add_argument("--host-process-root-pid", type=int,
                        help="Existing ancestor owning the retained execution session; omitted for standalone planning")
    # END defaults preserve the original half-host/half-device limits.
    parser.add_argument("--torch-memory-mib", type=int, choices=(2048, 4096), default=2048)
    # BEGIN opt-in host profiling covers preparation, planning and persistence.
    parser.add_argument("--profile-runtime", action="store_true",
                        help="Record host function timings under output/runtime-profile; disabled by default")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    from manipulation.runtime_profiling import run_with_optional_profile
    run_with_optional_profile(run, args, enabled=args.profile_runtime,
                              output=args.output/"runtime-profile")
    # END ordinary runs do not construct a profiler or write timing artifacts.
# END observed skill integration using existing production scene contracts.
