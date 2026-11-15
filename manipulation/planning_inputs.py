"""Bind current robot commands and observed owners for one planning request.

All functions consume explicit task, feedback and model declarations. Source
contacts remain skill knowledge, while current geometry is represented by
observed fields and calibrated robot kinematics.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

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


def finger_trajectory_factory(problem, feedback, finger_targets, *, incoming_command, timestep, ctrlrange,
                              position_limits, velocity_limits, acceleration_limits):
    """Bind original events once the native formal duration/map are known.

    Actual-position protection bounds are deliberately absent from this
    constructor. They continue guarding measured feedback in the executor;
    they cannot predict a hand shape or replace the original position goals.
    The returned reference continues the explicitly supplied active command.
    Measured q/dq remain separate entrance observations, including when contact
    prevents the physical fingers from reaching the issued target.
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
        # BEGIN separate active device command and measured finger entrance.
        return FingerTrajectory.build(joint_names=names, incoming_command=incoming_command,
            measured_initial_position=[measured[name] for name in names],
            measured_initial_velocity=[speeds[name] for name in names], commands=commands,
            duration=duration_s, executable_until=stop, timestep=timestep,
            position_limits=np.asarray([position_limits[name] for name in names]).T,
            velocity_limits=np.stack((-velocity, velocity)),
            acceleration_limits=np.stack((-acceleration, acceleration)),
            source=(f"Original explicit actuator targets: {finger_targets.source}; C1 command reference "
                    "under native planner acceleration bounds, not calibrated actual acceleration/tracking limits"),
            # BEGIN the public feedback clock is shared by real and simulated captures.
            feedback_id=feedback["feedback_id"], timestamp_s=feedback["timestamp_s"])
            # END the capture adapter owns the clock's original physical meaning.
        # END bound events retain their sole native clock.

    # BEGIN expose the already bound device tick to the planning clock owner.
    build.timestep = float(timestep)
    # END this is device metadata, never a per-skill retiming rule.
    return build


def position_controlled_hand(robot, feedback, problem, *, device, execution_limits,
                             position_limits, velocity_limits, trajectory_factory, fixture, execution_events=None):
    """Bind independent command reference after checking actual execution limits.

    The shared hand contract uses the registered complete robot FK for its
    prismatic coordinates. This boundary owns the measured q/dq checks,
    execution policy, command identities and synchronous reference factory.
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


def require_feedback_identity(feedback):
    """Keep the capture adapter's identity across planning and execution.

    A file path only names storage and can change when a capture is copied.
    It cannot replace the ID carried by the current named robot measurement
    and its explicitly sourced event/relation records.
    """
    identity = feedback.get("feedback_id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("The input adapter must supply an explicit current feedback_id")
    return identity


def owner_observation_inputs(problem, scene_input, views, mask_rows, *, source, observed_owners=()):
    """Resolve masks only for entities selected for RGB-D reconstruction.

    Existing models retain their registered identity and motion. They do not
    participate in the new observation-mask contract, including under complete
    occlusion. New entity coordinates come from the explicit task binding.
    """
    from manipulation.motion import _pose_array

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
            raise ValueError("Each owner mask requires OWNER_ID VIEW_ID MASK_VALUE")
        owner, view, path = row
        if (owner, view) in masks:
            raise ValueError("Every current owner/view mask must be supplied exactly once")
        masks[owner, view] = path
    if set(masks) != {(owner, view) for owner in frames for view in view_ids}:
        raise ValueError("Supply exactly one current owner mask for every explicit owner and view")
    return frames, masks

def load_observed_owner(path):
    """Restore the same pre-relationship field and the declared rigid motion."""
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import Attachment, CollisionOwner
    row = json.loads(Path(path).read_text())
    from .observed_mapping import FrozenObservedField
    field = FrozenObservedField.load(row['field_path'])
    pose = np.asarray(row['initial_pose'], float)
    return CollisionOwner(row['owner_id'], field,
        PosePath(np.array([0., 1.]), np.stack((pose, pose)), row['source']), row['source'],
        moving=bool(row['attachments']),
        attachments=tuple(Attachment(**relation) for relation in row['attachments']))


def problem_owner_binding(problem, owner_id):
    """Copy the exact current owner pose and deduplicated held spans.

    The inverse attachment maps owner-local points into the tool frame.
    Release IDs and progress come from the current compiler's held metadata,
    never a source frame guessed to be the executable release time.
    """
    from manipulation.motion import _pose_array
    from scipy.spatial.transform import Rotation
    phases = [phase for phase in problem.phases if phase.metadata.get("carrier_id") == owner_id]
    if not phases:
        raise ValueError("The explicit geometry owner is absent from the current planning problem")
    initial = _pose_array(phases[0].metadata["owner_initial_pose"], "current initial owner pose")
    attachments = []
    for phase in phases:
        metadata = phase.metadata
        if not np.allclose(initial, metadata["owner_initial_pose"], atol=1e-10, rtol=0):
            raise ValueError("A carrier's initial collision frame must remain the same across its phases")
        if metadata.get("mode") != "held":
            continue
        inverse = np.linalg.inv(_pose_array(metadata["attachment"], "compiled owner-from-tool attachment"))
        relation = {"interval": list(metadata["attachment_interval"]),
                    "position_tool_owner": inverse[:3, 3].tolist(),
                    "quaternion_tool_owner": Rotation.from_matrix(inverse[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist(),
                    "condition_id": metadata["attachment_condition_id"],
                    "release_condition_id": metadata["release_condition_id"]}
        same_span = [item for item in attachments if item["interval"] == relation["interval"]]
        if same_span:
            if same_span[0] != relation:
                raise ValueError("Repeated held phases disagree on their collision attachment")
        else:
            attachments.append(relation)
    return initial, attachments



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
