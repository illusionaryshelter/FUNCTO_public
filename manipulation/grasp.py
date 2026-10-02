"""Consume independent grasp geometry without declaring a successful grasp.

Poses use column vectors, metres, and ``T_parent_child``. A demonstrated
``T_carrier_eef`` is an end-effector motion, even when the carrier never moves.
The current scene supplies an already checked correspondence from the demo
grasp region to its current counterpart. This module does not infer that
correspondence, transfer between different tools, or certify robot execution.

Source joint measurements and source controller actions remain evidence;
only explicit ``open`` / ``close`` / ``hold`` intentions are consumed here.
"""

# BEGIN independent grasp and measured-holding geometry consumers.
from copy import deepcopy

import numpy as np

from .motion import _inverse, _pose_array, _poses_array


def validate_timeline(frame_indices, timestamps):
    """Return independent arrays with original observation IDs and seconds.

    Frame IDs need not be consecutive or begin at zero. They are identifiers,
    not array offsets. Times must be finite and strictly increasing; this
    validation never replaces them with the geometry kernel's sample phase.
    A one-observation sequence is valid for an already established boundary.
    """
    # Validate before NumPy can coerce a mixed [False, 2] list to integers.
    if any(isinstance(value, (bool, np.bool_)) for value in frame_indices):
        raise ValueError("frame_indices must contain integer IDs, not booleans")
    frames = np.asarray(frame_indices)
    times = np.asarray(timestamps)
    if (frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu"
            or np.any(frames < 0) or np.any(frames[1:] <= frames[:-1])):
        raise ValueError("frame_indices must be nonnegative strictly increasing integer IDs")
    if (times.shape != frames.shape or times.dtype.kind not in "fiu"
            or not np.isfinite(times).all() or np.any(times[1:] <= times[:-1])):
        raise ValueError("timestamps must be finite strictly increasing seconds aligned with frame_indices")
    return frames.copy(), times.astype(float, copy=True)


def _name(record, key):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"A nonempty {key} is required")
    return value


def _gripper_events(value, frames):
    """Check explicit intentions without deriving commands from measured qpos.

    Observations own both boundaries; commands own ``[first, last)``. Requiring
    an observed source frame avoids silently snapping an event to a different
    time when a caller has decimated a trajectory. Retiming belongs to the
    robot backend and must carry these original event IDs with it.
    """
    if not isinstance(value, list) or not value:
        raise ValueError("gripper_events must contain explicit grasp intentions")
    frame_set = set(map(int, frames))
    events, ids, previous = [], set(), -1
    for event in value:
        if not isinstance(event, dict):
            raise ValueError("Each gripper event must be a record")
        event_id = _name(event, "event_id")
        frame = event.get("frame_index")
        command = event.get("command")
        if (isinstance(frame, (bool, np.bool_))
                or not isinstance(frame, (int, np.integer))
                or int(frame) not in frame_set or int(frame) == int(frames[-1])):
            raise ValueError("Gripper event frame_index must name a source observation in [first, last)")
        if event_id in ids or int(frame) <= previous:
            raise ValueError("Gripper event IDs must be unique and event frames strictly increasing")
        if command not in {"open", "close", "hold"}:
            raise ValueError("Gripper commands must be explicit open, close, or hold intentions")
        ids.add(event_id)
        previous = int(frame)
        events.append(deepcopy(event))
    if not any(event["command"] == "close" for event in events):
        raise ValueError("Independent grasp needs an explicit close intention")
    if events[-1]["command"] == "open":
        raise ValueError("Independent grasp must finish closing or holding, not releasing")
    return events


# BEGIN source grasp motion is an initializer, not a protected robot path.
def ground_grasp(demo, scene):
    """Bind the original same-tool reference and its independent grasp goal.

    Region correspondence maps the complete source motion into the current
    scene. The actual robot state belongs to the planner; no selected source
    frame creates an approach waypoint, Cartesian suffix lock, or stop. The
    terminal local grasp relation is a proposed goal, never observed Holding.
    ``protected_start_frame`` retains contact evidence identity only.
    """
    for key in ("tool_id", "tool_frame"):
        if _name(demo, key) != _name(scene, key):
            raise ValueError(f"Independent grasp requires the same {key}; cross-tool retargeting is not implied")
    relative = _poses_array(demo["T_carrier_eef"], "T_carrier_eef")
    frames, times = validate_timeline(demo["frame_indices"], demo["timestamps"])
    if len(relative) != len(frames):
        raise ValueError("T_carrier_eef must align with the original timeline")
    events = _gripper_events(demo["gripper_events"], frames)
    protected = demo["protected_start_frame"]
    if (isinstance(protected, (bool, np.bool_))
            or not isinstance(protected, (int, np.integer)) or protected not in frames):
        raise ValueError("protected_start_frame must name an original observation")
    if any(event["frame_index"] < protected and event["command"] == "close" for event in events):
        raise ValueError("The contact evidence boundary cannot follow a closing intention")
    carrier = _pose_array(scene["carrier_pose"], "carrier_pose")
    _pose_array(scene["eef_pose"], "eef_pose")
    mapping = _pose_array(scene["region_transform"], "region_transform")
    bound_relative = mapping @ relative
    return {
        "status": "candidate", "output_type": "eef_trajectory",
        "eef_poses": carrier @ bound_relative, "frame_indices": frames, "timestamps": times,
        "protected_start_frame": int(protected),
        "goal_pose": carrier @ bound_relative[-1],
        "candidate_T_carrier_eef": bound_relative[-1].copy(), "gripper_events": events,
        "tool_id": demo["tool_id"], "tool_frame": demo["tool_frame"],
        "holding_status": "requires_verification",
        "effect_requirement": {"frame_index": int(frames[-1]),
                               "requirement": "stable_grasp_from_current_synchronized_evidence",
                               "original_support_separation_required": False},
        "pose_convention": "column-vector world_from_eef; metres; 4x4 SE(3)",
        "path_scope": "bound source initialization reference; no path or execution feasibility claim",
    }
# END source grasp motion is an initializer, not a protected robot path.


def attached_eef_reference(carrier_poses, actual_T_carrier_eef):
    """Apply a caller-verified measured attachment during rigid holding only.

    The caller obtains ``actual_T_carrier_eef = inverse(T_world_carrier) @
    T_world_eef`` from synchronized current evidence after checking stability.
    This function only composes geometry; passing a matrix is not evidence that
    the object is held. Stop using it when that relation is no longer valid,
    and never extend it through the released retreat phase.
    """
    carrier = _poses_array(carrier_poses, "carrier_poses")
    actual = _pose_array(actual_T_carrier_eef, "actual_T_carrier_eef")
    return carrier @ actual


def retreat_from_release(demo_eef_poses, actual_release_pose):
    """Continue a free retreat from actual release using release-local motion.

    For the already bound independent retreat D, preserve each relative pose
    ``inverse(D[0]) @ D[t]`` and apply it at the actual release EEF A. Thus the
    first output is exactly A even if the actual grasp differed from the demo.
    This changes only the free retreat; no earlier object/function event is
    moved to make the join work. Geometry is continuous at release but must
    still pass the external robot backend's collision and timing checks.
    """
    demo = _poses_array(demo_eef_poses, "demo_eef_poses")
    actual = _pose_array(actual_release_pose, "actual_release_pose")
    output = actual @ _inverse(demo[0]) @ demo
    output[0] = actual
    return output
# END independent grasp and measured-holding geometry consumers.
