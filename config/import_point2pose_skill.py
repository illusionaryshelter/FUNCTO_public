"""Bind recorded Point2Pose observations to the existing portable skill format.

This adapter does not track, filter trajectories, infer events, or assess task
success. It retains the complete recorded interval and uses the existing
functional-frame, representation and bundle writers for all geometry.
"""

# BEGIN recorded RGB-D estimates enter the same fixed-frame representation.
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from manipulation.grasp import validate_timeline
from manipulation.motion import _pose_array, _poses_array
from manipulation.regions import source_region_frame
from manipulation.representation import build_representation
from manipulation.skill import _metadata, _source, save_grounding_skill
from transfer.multiview_experiment import _check_observation


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must identify the actual evidence source")
    return value


def _recording(directory, bindings):
    """Read every original row; retained old matrices never replace observations."""
    directory = Path(directory)
    report = json.loads((directory / "tracking_evidence.json").read_text())
    missing = []
    if report["status"] != "completed":
        return None, report, [dict(reason="tracking_recording_not_completed", status=report["status"],
                                  failed_frame=report.get("failed_frame"), error=report.get("error"))]
    with np.load(directory / "estimated_poses_by_frame.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    ids = data["object_ids"].tolist()
    if (not ids or any(not isinstance(owner, str) or not owner.strip() for owner in ids)
            or len(set(ids)) != len(ids) or ids != report["object_ids"]):
        raise ValueError("Tracking NPZ/report must name the same fixed object identities")
    if any(owner not in ids for owner in bindings.values()):
        raise ValueError("Every bound owner must identify an object in this recording")
    count = len(data["frame_indices"])
    rows = report["frames"]
    if (not count or len(rows) != count or report["processed_frames"] != count
            or not np.array_equal(data["tracker_frame_ids"], np.arange(count))
            or data["timestamp_s"].shape != (count,) or data["K"].shape != (count, 3, 3)):
        raise ValueError("Tracking report and arrays must retain the entire original frame sequence")
    for name in ("timestamp_available", "world_camera_available"):
        available = data[name]
        if available.dtype != np.bool_ or available.shape != (count,):
            raise ValueError(f"Invalid per-frame availability array: {name}")
        for index in np.flatnonzero(~available):
            missing.append(dict(frame_index=int(data["frame_indices"][index]), reason=name + "_missing"))
    # Missing real time/world is a missing observation, not an invitation to
    # fabricate a clock or compose poses with an identity camera transform.
    if missing:
        return data, report, missing
    frames, times = validate_timeline(data["frame_indices"], data["timestamp_s"])
    camera = _poses_array(data["T_world_camera_cv"], "recorded world camera poses")
    if len(camera) != count:
        raise ValueError("Camera poses must align with the original frame sequence")
    for name in ("T_world_B", "T_camera_B", "T_camera_initial_map"):
        if data[name].shape != (count, len(ids), 4, 4):
            raise ValueError(f"{name} must keep the original frame and object axes")
        _poses_array(data[name].reshape(-1, 4, 4), name)
    body = _poses_array(data["T_initial_camera_B"], "fixed initial body charts")
    if body.shape[0] != len(ids):
        raise ValueError("Fixed initial body charts must retain the object identity order")
    # These equalities check storage/coordinate consistency, not tracking
    # accuracy. They introduce no new registration or pose-quality threshold.
    if (not np.allclose(data["T_camera_B"], data["T_camera_initial_map"] @ body, atol=1e-7, rtol=0)
            or not np.allclose(data["T_world_B"], camera[:, None] @ data["T_camera_B"], atol=1e-7, rtol=0)):
        raise ValueError("Recorded body poses do not use the declared fixed B and camera transforms")
    for index, row in enumerate(rows):
        if (row["frame_index"] != int(frames[index]) or row["tracker_frame_id"] != index
                or row["frame_name"] != str(data["frame_names"][index])
                or not np.isclose(row["timestamp_s"], times[index], atol=1e-9, rtol=0)
                or row["camera_pose_available"] is not True
                or [item["object_index"] for item in row["objects"]] != list(range(len(ids)))):
            raise ValueError("Tracking evidence must align with every NPZ frame, time and object")
        for role, owner in bindings.items():
            status = row["objects"][ids.index(owner)]["status"]
            usable = status in {"frontend_condition_met", "warning"} or (index == 0 and status == "initialized")
            if not usable:
                missing.append(dict(frame_index=int(frames[index]), owner_id=owner, role=role,
                                    reason="no_current_source_observation", tracking_status=status))
    return data, report, missing


def import_point2pose_skill(tracking_directory, output, *, metadata, source, kind,
                           owner_bindings, binding_source, eef=None, function_frame=None,
                           approach_time=None, interaction_time=None, release_time=None,
                           events=(), event_source=None):
    """Save one observed interval in the existing version-2 skill bundle.

    ``source`` uses save_grounding_skill's visual fields only. Old body poses,
    L matrices and frame-construction records are rejected. Its point bank and
    camera calibration belong to source.frame_index/source.timestamp_s.

    ``owner_bindings`` maps carrier/reference to the runner's object_ids; a
    grasp requires carrier, manipulation also requires reference. Source owner
    role is carrier for grasp and reference for manipulation. ``binding_source``
    explains those identity assignments, separately from semantic metadata.

    Optional ``eef`` contains frame_indices, timestamps, T_world_eef, source and
    world_alignment_source. All samples must already share the recorded clock
    and world frame; this adapter does not interpolate or calibrate them.
    Optional ``function_frame`` contains T_world_function, frame_index,
    timestamp_s and source. Otherwise the existing source_region_frame builds
    the frame from the supplied region's measured points and source camera.

    Event records use their existing ``timestamp`` in original seconds. Any
    supplied event or boundary time requires event_source. Absent interaction
    time remains absent; saving a short interval does not make a complete skill.
    Missing/stale/failed tracking returns missing_source_observations and writes
    no bundle. Warning candidates retain every raw diagnostic in provenance.
    """
    meta = _metadata(metadata)
    expected_role = "carrier" if kind == "grasp" else "reference" if kind == "manipulation" else None
    if expected_role is None or meta["owner_role"] != expected_role:
        raise ValueError("Source owner role must match the represented functional motion")
    if (not isinstance(owner_bindings, dict) or "carrier" not in owner_bindings
            or not set(owner_bindings) <= {"carrier", "reference"}
            or expected_role not in owner_bindings):
        raise ValueError("Explicit carrier/reference owner bindings are required for this representation")
    bindings = {role: _text(owner, "bound owner ID") for role, owner in owner_bindings.items()}
    _text(binding_source, "binding_source")
    # BEGIN retain explicit visual-region geometry through the tracking import.
    allowed = {"image", "observation", "depth_m", "crop_xyxy", "timestamp_s", "frame_index", "owner_id", "region_geometry",
               "region_image_mask", "feature", "feature_model", "provenance", "missing"}
    # END no tracking pose determines a region's geometric model.
    if not isinstance(source, dict) or not set(source) <= allowed:
        raise ValueError("Source must contain only visual fields; old body poses, L and frame construction are forbidden")
    if source["owner_id"] != bindings[expected_role]:
        raise ValueError("Source region owner must match the explicit tracked owner binding")
    validate_timeline([source["frame_index"]], [source["timestamp_s"]])
    data, report, missing = _recording(tracking_directory, bindings)
    if missing:
        return dict(status="missing_source_observations", missing=missing, tracking_evidence=report)
    frames, times = data["frame_indices"], data["timestamp_s"]
    index = np.flatnonzero(frames == source["frame_index"])
    if len(index) != 1 or not np.isclose(times[index[0]], source["timestamp_s"], atol=1e-9, rtol=0):
        raise ValueError("Source image must use its original tracking frame and acquisition time")
    index = int(index[0])
    ids = data["object_ids"].tolist()
    source_body = data["T_world_B"][index, ids.index(source["owner_id"])]
    src = _source(dict(source, T_world_body_at_image=source_body))
    observation = src["observation"]
    if (not np.allclose(observation["K"], data["K"][index], atol=1e-9, rtol=0)
            or not np.allclose(observation["T_world_camera_cv"], data["T_world_camera_cv"][index], atol=1e-9, rtol=0)):
        raise ValueError("Visual source calibration must match the recorded source frame")
    _check_observation(observation, src["depth_m"],
                       {"K": observation["K"], "T_world_camera_cv": observation["T_world_camera_cv"]})
    if not observation["region_mask"].any():
        raise ValueError("A supplied source region needs its measured source-image points")
    world_function = None
    if function_frame is not None:
        validate_timeline([function_frame["frame_index"]], [function_frame["timestamp_s"]])
        if (function_frame["frame_index"] != src["frame_index"]
                or not np.isclose(function_frame["timestamp_s"], src["timestamp_s"], atol=1e-9, rtol=0)):
            raise ValueError("Provided world function frame must belong to the source image's frame and time")
        _text(function_frame["source"], "function frame source")
        world_function = _pose_array(function_frame["T_world_function"], "provided world function frame")
    geometry = source_region_frame(T_world_body=source_body, T_world_camera=observation["T_world_camera_cv"],
        points_world=observation["xyz"][observation["region_mask"]], T_world_function=world_function)
    src.update(L_body_region=geometry["L_body_region"], frame_construction=geometry["details"])
    eef_poses, eef_origin = None, None
    if eef is not None:
        eef_frames, eef_times = validate_timeline(eef["frame_indices"], eef["timestamps"])
        if not np.array_equal(eef_frames, frames) or not np.allclose(eef_times, times, atol=1e-9, rtol=0):
            raise ValueError("EEF observations must already align with every original tracking frame and time")
        eef_origin = dict(source=_text(eef["source"], "EEF source"),
                          world_alignment_source=_text(eef["world_alignment_source"], "EEF world alignment source"))
        eef_poses = _poses_array(eef["T_world_eef"], "recorded EEF observations")
        if len(eef_poses) != len(times):
            raise ValueError("EEF observations must cover the complete tracked interval")
    event_records = deepcopy(list(events))
    for time in (approach_time, interaction_time, release_time):
        if time is not None and (isinstance(time, (bool, np.bool_)) or not np.isscalar(time) or not np.isfinite(time)):
            raise ValueError("Boundary times must be finite original acquisition seconds")
    if event_records or any(time is not None for time in (approach_time, interaction_time, release_time)):
        _text(event_source, "event_source")
    for event in event_records:
        time = event["timestamp"]
        if isinstance(time, (bool, np.bool_)) or not np.isfinite(time) or not times[0] <= time <= times[-1]:
            raise ValueError("Events must retain explicit original times inside this interval")
        if "frame_index" in event:
            matches = np.flatnonzero(frames == event["frame_index"])
            if len(matches) != 1 or not np.isclose(times[matches[0]], time, atol=1e-9, rtol=0):
                raise ValueError("Event frame identity and acquisition time disagree")
    # Replace old motion/GT scope claims. Region input provenance is preserved
    # only as the caller's visual-region evidence, independently of P2P motion.
    provenance = dict(motion=dict(kind="point2pose_estimate", directory=str(Path(tracking_directory).resolve()),
                                  body_chart="frozen Point2Pose initial body chart B"),
        owner_bindings=dict(owners=bindings, source=binding_source),
        region=dict(kind="supplied_visual_region", visual_provenance=deepcopy(source.get("provenance", {})),
                    function_frame=None if function_frame is None else deepcopy(function_frame)),
        eef=eef_origin, events=dict(source=event_source, missing_interaction_time=interaction_time is None),
        tracking_evidence=report,
        accuracy_scope="Recorded estimates and diagnostics; no tracking-accuracy or execution-success claim")
    meta["provenance"] = deepcopy(provenance)
    src["provenance"] = deepcopy(provenance)
    tracks = {role: data["T_world_B"][:, ids.index(owner)] for role, owner in bindings.items()}
    representation = build_representation(timestamps=times, frame_indices=frames, kind=kind,
        T_world_carrier=tracks["carrier"], T_world_reference=tracks.get("reference"), T_world_eef=eef_poses,
        L_G=geometry["L_body_region"] if kind == "grasp" else None,
        L_R=geometry["L_body_region"] if kind == "manipulation" else None,
        approach_time=approach_time, interaction_time=interaction_time, release_time=release_time,
        events=event_records, provenance=provenance)
    manifest = save_grounding_skill(output, metadata=meta, source=src, representation=representation)
    return dict(status="saved_observation_segment", manifest=manifest, metadata=meta,
                source=src, representation=representation)
# END no tracker policy, task tolerance or missing event is synthesized here.
