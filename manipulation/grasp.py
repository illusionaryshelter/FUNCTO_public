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

from .motion import _inverse, _pose_array, _poses_array, _tolerances
from .gripper_state import _source


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

    # BEGIN source end ownership is independent of device execution coverage.
    An explicitly owned command may use either source boundary. Requiring an
    observed frame avoids silently snapping an event after decimation. The
    shared physical clock must still provide a real activation tick and the
    required post-command coverage; accepting a source ID does not prove that.
    # END no last source observation is silently converted into a command.
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
                or int(frame) not in frame_set):
            raise ValueError("Gripper event frame_index must name an original source observation")
        if event_id in ids or int(frame) <= previous:
            raise ValueError("Gripper event IDs must be unique and event frames strictly increasing")
        if command not in {"open", "close", "hold"}:
            raise ValueError("Gripper commands must be explicit open, close, or hold intentions")
        ids.add(event_id)
        previous = int(frame)
        events.append(deepcopy(event))
    if not any(event["command"] == "close" for event in events):
        raise ValueError("Independent grasp needs an explicit close intention")
    # BEGIN an intention does not establish or refute the actual grasp effect.
    # A terminal open is retained like close/hold. A required Holding result
    # remains independently evaluated; source-command validation cannot infer
    # object attachment from the command name or silently delete the operation.
    # END actual effects, device coverage and dependencies remain separate.
    return events


# BEGIN generated grasp goals retain their original reference anchor.
def _grasp_goal(value):
    """Validate the shared explicit grasp relation and task precision contract."""
    required = {"T_carrier_eef", "position_tolerance_m", "rotation_tolerance_rad", "source"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("grasp_goal requires explicit T_carrier_eef, task position/rotation tolerances and source")
    if any(isinstance(value[name], (bool, np.bool_)) for name in
           ("position_tolerance_m", "rotation_tolerance_rad")):
        raise ValueError("Grasp task tolerances must be metric numbers, not booleans")
    position_tolerance, rotation_tolerance = _tolerances(
        value["position_tolerance_m"], value["rotation_tolerance_rad"])
    return {"T_carrier_eef": _pose_array(value["T_carrier_eef"], "grasp_goal.T_carrier_eef"),
            "position_tolerance_m": position_tolerance,
            "rotation_tolerance_rad": rotation_tolerance,
            "source": _source(value["source"], "Grasp task goal and allowed set")}


def _reference_prior(value):
    """Read the prior in the same carrier coordinates as the selected goal."""
    if not isinstance(value, dict) or not {"T_carrier_eef", "source"} <= set(value):
        raise ValueError("grasp_reference_prior requires T_carrier_eef and source")
    return {"T_carrier_eef": _pose_array(value["T_carrier_eef"], "grasp_reference_prior.T_carrier_eef"),
            "source": _source(value["source"], "Original grasp reference anchor")}


def bind_grasp_candidate(scene, T_reference_tool, *, source):
    """Return a scene whose new grasp goal also aligns its complete reference.

    ``T_reference_tool`` uses the scene's ``reference_frame`` and the existing
    ``tool_frame`` control point, not a generator's gripper base. Its producer
    must first perform that calibrated frame conversion. This binding only
    creates an action candidate; it establishes neither clearance nor Holding.

    Keep the original independent goal as ``grasp_reference_prior``. Grounding
    then moves every ordinary reference sample by the rigid displacement from
    that prior to the selected goal. An existing prior remains the anchor when
    another candidate is bound to an already bound scene, so changes never
    accumulate. The demonstration, functional correspondence, contact domains,
    current feedback and task tolerances are not altered.
    """
    if not isinstance(scene, dict):
        raise ValueError("A grasp candidate requires a bound scene record")
    for key in ("reference_frame", "tool_id", "tool_frame"):
        _name(scene, key)
    carrier = _pose_array(scene["carrier_pose"], "carrier_pose")
    target = _pose_array(T_reference_tool, "T_reference_tool")
    goal = _grasp_goal(scene.get("grasp_goal"))
    prior = (_reference_prior(scene["grasp_reference_prior"])
             if "grasp_reference_prior" in scene else
             {"T_carrier_eef": goal["T_carrier_eef"].copy(), "source": deepcopy(goal["source"])})
    # An exactly retained prior must not acquire floating-point drift through
    # inverse/compose. Nearby distinct candidates are never snapped to it.
    selected = (prior["T_carrier_eef"].copy()
                if np.array_equal(target, carrier @ prior["T_carrier_eef"])
                else _inverse(carrier) @ target)
    result = deepcopy(scene)
    result["grasp_reference_prior"] = prior
    result["grasp_goal"] = {**goal, "T_carrier_eef": selected,
                            "source": _source(source, "Generated grasp candidate")}
    return result
# END candidate binding never changes physical functional correspondence.


# BEGIN source grasp motion is an initializer, not a protected robot path.
def ground_grasp(demo, scene):
    """Bind the original same-tool reference and its independent grasp goal.

    Region correspondence maps the complete source motion into the current
    scene. The actual robot state belongs to the planner; no selected source
    frame creates an approach waypoint, Cartesian suffix lock, or stop. The
    grasp relation is explicitly supplied in ``scene.grasp_goal`` as the
    current carrier-local ``T_carrier_eef``, task ``position_tolerance_m`` and
    ``rotation_tolerance_rad``, and their ``source``. It is independent of
    ordinary reference samples and never observed Holding. Numerical solver
    tolerances do not supply missing task precision. ``protected_start_frame``
    retains contact evidence identity only.

    # BEGIN candidate alignment is explicit rather than inferred from a goal change.
    ``grasp_reference_prior`` opts into a generated candidate's full-reference
    alignment. For bound source samples D, original goal P and selected goal E
    in the output frame, the new reference is E @ inverse(P) @ D. This preserves
    the source's relative deviations even when D[-1] differs from P. Ordinary
    scenes without this record keep their independent goal/reference contract.
    # END reference alignment adds no waypoint, contact permission or command.
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
    # BEGIN independently bind the current grasp relation and its allowed set.
    grasp_goal = _grasp_goal(scene.get("grasp_goal"))
    # END neither the demonstration endpoint nor engineering epsilon fills this contract.
    mapping = _pose_array(scene["region_transform"], "region_transform")
    bound_relative = mapping @ relative
    # BEGIN relocate the complete ordinary reference around the explicit prior.
    prior = None
    if "grasp_reference_prior" in scene:
        prior = _reference_prior(scene["grasp_reference_prior"])
        if not np.array_equal(grasp_goal["T_carrier_eef"], prior["T_carrier_eef"]):
            bound_relative = (grasp_goal["T_carrier_eef"]
                              @ _inverse(prior["T_carrier_eef"]) @ bound_relative)
    # END no ordinary source sample is promoted to the candidate target.
    result = {
        "status": "candidate", "output_type": "eef_trajectory",
        "eef_poses": carrier @ bound_relative, "frame_indices": frames, "timestamps": times,
        "protected_start_frame": int(protected),
        "goal_pose": carrier @ grasp_goal["T_carrier_eef"], "grasp_goal": grasp_goal,
        "candidate_T_carrier_eef": grasp_goal["T_carrier_eef"].copy(), "gripper_events": events,
        "tool_id": demo["tool_id"], "tool_frame": demo["tool_frame"],
        "holding_status": "requires_verification",
        "effect_requirement": {"frame_index": int(frames[-1]),
                               "requirement": "stable_grasp_from_current_synchronized_evidence",
                               "original_support_separation_required": False},
        "pose_convention": "column-vector world_from_eef; metres; 4x4 SE(3)",
        "path_scope": "bound source initialization reference; no path or execution feasibility claim",
    }
    # BEGIN preserve the anchor provenance in generated reference artifacts.
    if prior is not None:
        result["grasp_reference_prior"] = prior
    return result
    # END the selected pose remains a candidate, never observed Holding.
# END source grasp motion is an initializer, not a protected robot path.


def attached_eef_reference(carrier_poses, actual_T_carrier_eef):
    """Apply the caller's declared attachment during modeled rigid holding.

    The caller obtains ``actual_T_carrier_eef = inverse(T_world_carrier) @
    T_world_eef`` from synchronized current evidence after checking stability.
    This function only composes geometry; passing a matrix is not evidence that
    the object is held. Stop using it when that relation is no longer valid,
    and never extrapolate it after release.
    """
    carrier = _poses_array(carrier_poses, "carrier_poses")
    actual = _pose_array(actual_T_carrier_eef, "actual_T_carrier_eef")
    return carrier @ actual


# BEGIN released-hand continuation is not part of the current held skill.
# No generic retreat helper remains; a later skill starts from new feedback.
# END attachment geometry stops when its observed relation ceases to apply.
# END independent grasp and measured-holding geometry consumers.
