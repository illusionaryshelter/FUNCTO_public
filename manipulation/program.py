"""Consume one compiled skill, preserving function events and tool phases.

This is geometry generation for an already selected InteractionProgram, not a
task planner or robot executor. Demonstration data may have offline GT origins;
``scene`` must be supplied by the caller's current perception/tool feedback.
All output poses share the explicitly named current reference frame. Source
frame IDs and timestamps are retained; backend retiming must carry the events.
"""

# BEGIN region-based skill program consumer.
from copy import deepcopy

import numpy as np

from .grasp import (attached_eef_reference, ground_grasp, retreat_from_release,
                    validate_timeline)
from .motion import (MotionModel, _errors, _inverse, _pose_array, _poses_array,
                     _tolerances, transfer_motion)
from .regions import ground_relation, relation_error


def _index(frame, frames, name):
    """Resolve an original frame ID without silently treating it as an index."""
    if isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer)):
        raise ValueError(f"{name} must be an original integer frame ID")
    found = np.flatnonzero(frames == frame)
    if len(found) != 1:
        raise ValueError(f"{name} must occur in the original observation timeline")
    return int(found[0])


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a number, not a boolean")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _intentions(events, frames):
    """Validate manipulation commands, which need not contain a new close."""
    if not isinstance(events, list):
        raise ValueError("gripper_events must be a list of explicit intentions")
    seen, previous = set(), -1
    for event in events:
        name = event.get("event_id")
        index = _index(event.get("frame_index"), frames, "gripper event")
        if (not isinstance(name, str) or not name or name in seen
                or index <= previous or index == len(frames) - 1):
            raise ValueError("Gripper events must be unique, ordered, and own actions in [first,last)")
        if event.get("command") not in {"open", "close", "hold"}:
            raise ValueError("Gripper intentions must be open, close or hold")
        seen.add(name)
        previous = index
    return deepcopy(events)


def _current_model(value):
    """Bind the existing FK model to evidenced current mechanism parameters.

    The caller supplies its reference pose, signed coordinate basis, and (for
    a hinge) axis point, all in the result reference frame. A source fit or an
    action name alone does not establish a new instance's mechanism. Optional
    coordinate limits have shape (dimension,2) and are checked on every node.
    """
    kind = value.get("kind")
    dimensions = {"prismatic": 1, "revolute": 1, "planar": 3, "translation_plane": 2}
    if kind not in dimensions or not value.get("source"):
        raise ValueError("A current constrained model needs a supported kind and evidence source")
    pose = _pose_array(value["reference_pose"], "Current mechanism reference pose")
    basis_pose = np.eye(4)
    basis_pose[:3, :3] = np.asarray(value["basis"], float)
    basis = _pose_array(basis_pose, "Current mechanism basis")[:3, :3]
    point = value.get("axis_point")
    if kind == "revolute":
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Current revolute model requires a finite axis_point")
    elif point is not None:
        raise ValueError("axis_point is only meaningful for a revolute model")
    ptol, rtol = _tolerances(value.get("position_tolerance", .005),
                             value.get("rotation_tolerance", .035))
    model = MotionModel(kind, np.zeros((1, dimensions[kind])), pose, basis, point,
                        ptol, rtol, {"source": value["source"], "scope": "caller-bound current mechanism"})
    limits = value.get("coordinate_limits")
    if limits is not None:
        limits = np.asarray(limits, float)
        if (limits.shape != (model.dimension, 2) or not np.isfinite(limits).all()
                or np.any(limits[:, 1] < limits[:, 0])):
            raise ValueError("coordinate_limits must be finite ordered bounds for every model coordinate")
    return model, limits


def _exact_coordinates(model, pose):
    coordinates = model.coordinates_of(pose[None])[0]
    p, r = _errors(pose[None], model.poses(coordinates))
    if p[0] > 1e-8 or r[0] > 1e-8:
        raise ValueError("A required function/start pose is incompatible with the current mechanism")
    return coordinates


def _event_transfer(reference, start, goals, function_frame, model_spec=None):
    """Transfer in a physical function frame, preserving every internal event.

    Free intervals reuse the existing SE(3) transfer kernel on a material
    functional frame, not on an arbitrary body origin. Constrained intervals
    stay in one caller-bound current FK model; we never fit a new axis after
    each event. A model projection is allowed only for the soft reference.
    Required event targets and the observed initial pose must be exactly on
    the model, and the final output is independently checked by the caller.
    """
    knots = [(0, start)] + [(index, goal) for index, goal in goals if index != 0]
    if goals and goals[0][0] == 0 and not np.allclose(goals[0][1], start, atol=1e-8, rtol=0):
        raise ValueError("The initial function event conflicts with the observed current pose")
    output = np.empty_like(reference)
    diagnostics = {"phase": "source sample index; physical timestamps retained separately"}
    if model_spec is None:
        # Right multiplication preserves the physical function origin and its
        # lever arm while the existing free kernel enforces each endpoint.
        frame = _pose_array(function_frame, "Current material function frame")
        function_reference = reference @ frame
        for position, (begin, initial) in enumerate(knots):
            end, goal = knots[position + 1] if position + 1 < len(knots) else (len(reference) - 1, None)
            result = transfer_motion(function_reference[begin:end + 1], initial @ frame,
                                     goal_pose=None if goal is None else goal @ frame)
            output[begin:end + 1] = result["poses"] @ _inverse(frame)
        diagnostics["model"] = {"kind": "free", "origin": "material functional frame"}
    else:
        model, limits = _current_model(model_spec)
        coordinates = model._coordinates_unchecked(reference)
        # Projection of noisy reference geometry is reported, not re-labelled
        # as evidence that the reference already lies on the mechanism.
        p, r = _errors(reference, model.poses(coordinates))
        diagnostics["reference_projection"] = {"max_position_m": float(p.max()),
                                                "max_rotation_rad": float(r.max())}
        angular = 0 if model.kind == "revolute" else 2 if model.kind == "planar" else None
        values = coordinates.copy()
        for position, (begin, initial) in enumerate(knots):
            end, goal = knots[position + 1] if position + 1 < len(knots) else (len(reference) - 1, None)
            initial_q = _exact_coordinates(model, initial)
            if angular is not None:
                initial_q[angular] += 2 * np.pi * np.round(
                    (coordinates[begin, angular] - initial_q[angular]) / (2 * np.pi))
            part = coordinates[begin:end + 1] - coordinates[begin] + initial_q
            if goal is not None:
                goal_q = _exact_coordinates(model, goal)
                if angular is not None:
                    goal_q[angular] += 2 * np.pi * np.round(
                        (part[-1, angular] - goal_q[angular]) / (2 * np.pi))
                part += np.linspace(0, 1, len(part))[:, None] * (goal_q - part[-1])
            values[begin:end + 1] = part
        if limits is not None and (np.any(values < limits[:, 0] - 1e-9)
                                   or np.any(values > limits[:, 1] + 1e-9)):
            raise ValueError("Transferred function trajectory exceeds the current mechanism limits")
        output = model.poses(values)
        model.coordinates = values
        diagnostics["model"] = model.to_dict()
        diagnostics["limit_status"] = "checked_at_nodes" if limits is not None else "not_supplied"
    if not np.allclose(output[0], start, atol=1e-7, rtol=0):
        raise ValueError("Function transfer changed the observed initial object pose")
    return output, diagnostics


def _functional_events(demo, scene, frames, poses, last_held, task):
    events = demo.get("functional_events")
    if not isinstance(events, list) or not events:
        raise ValueError("Manipulation requires explicit functional_events; release/end frames are not substitutes")
    current = scene.get("functional_bindings", {})
    reference_pose = _pose_array(scene["reference_pose"], "Current reference-owner pose")
    output, previous, names = [], -1, set()
    for event in events:
        name = event.get("event_id")
        index = _index(event.get("frame_index"), frames, "functional event")
        if (not isinstance(name, str) or not name or name in names
                or index < previous or index > last_held):
            raise ValueError("Functional events must be unique, ordered and occur within controlled object motion")
        if not event.get("source"):
            raise ValueError("Each functional event requires its semantic/geometric evidence source")
        binding = current.get(name)
        if not isinstance(binding, dict) or not binding.get("source"):
            raise ValueError(f"Functional event {name!r} requires an evidenced current region binding")
        # Region ownership is a semantic/perception assertion from the caller;
        # numeric transforms alone cannot identify a wrong, similar instance.
        if binding.get("carrier_id") != scene["carrier_id"] or binding.get("reference_id") != scene["reference_id"]:
            raise ValueError("Functional binding owners disagree with the selected current entities")
        relation = ground_relation(poses[index], event["carrier_frame"], binding["carrier_frame"],
                                   event["reference_frame"], binding["reference_frame"])
        # The semantic frames are owner-local. Embed their relation in the
        # declared output coordinates once, even when those are camera/world
        # rather than the target owner's own frame.
        relation["goal_pose"] = reference_pose @ relation["goal_pose"]
        relation["reference_binding"] = reference_pose @ relation["reference_binding"]
        if task["relation"] == "on":
            # For a surface relation, frame correspondence supplies a pose
            # preference. Only the supported face/normal/finite area are hard
            # requirements; an unobserved twist is not made a semantic fact.
            from .pipeline import _surface
            from .regions import ground_on
            face = binding["on_geometry"]
            surface = _surface(scene["support"])
            grounded = ground_on(relation["goal_pose"], scene["body_points"],
                                face["carrier_region_points"], face["carrier_outward_normal"],
                                surface, margin=scene.get("margin", 0.),
                                gap_tolerance=face["gap_tolerance"],
                                normal_tolerance=face["normal_tolerance"],
                                antiparallel_axis=face.get("antiparallel_axis"))
            relation["goal_pose"] = grounded["goal_pose"]
            relation["constraint_kind"] = "on_region"
        else:
            relation["constraint_kind"] = "functional_frame"
        output.append({**relation, "event_id": name, "index": index,
                       "frame_index": int(frames[index]), "binding": binding,
                       "source": event["source"],
                       "position_tolerance_m": (_positive(event["position_tolerance_m"], "event position tolerance")
                                                if relation["constraint_kind"] == "functional_frame" else None),
                       "rotation_tolerance_rad": (_positive(event["rotation_tolerance_rad"], "event rotation tolerance")
                                                  if relation["constraint_kind"] == "functional_frame" else None)})
        previous = index
        names.add(name)
    return output


def _object_reference(demo, scene, frames, source, last_held, task, initial):
    """Shared function/region consumer for object and complete EEF outputs."""
    if task.get("relation") not in {"transfer", "on", "inside", "articulation", "tool_use"}:
        raise ValueError("Unsupported functional relation")
    margin = scene.get("margin", 0.)
    if (isinstance(margin, (bool, np.bool_)) or not isinstance(margin, (int, float, np.number))
            or not np.isfinite(margin) or margin < 0):
        raise ValueError("Geometry margin must be finite and nonnegative")
    if task["relation"] == "inside" and margin > 0 and not any(
            name in scene for name in ("interior", "entry_region")):
        raise ValueError("An inside clearance margin requires observed free geometry")
    events = _functional_events(demo, scene, frames, source, last_held, task)
    # The soft reference uses the first active functional correspondence; all
    # required events have their own two-sided goals, so A_C cannot disappear
    # through initial-pose normalization, even with different local mappings.
    first = events[0]
    reference = first["reference_binding"] @ source @ _inverse(first["carrier_binding"])
    # Several regions can constrain one physical event. They must agree on
    # one candidate body pose; never average conflicting region transforms.
    # Every original relation is still checked independently after transfer.
    unique_goals = {}
    for event in events:
        index, goal = event["index"], event["goal_pose"]
        if index in unique_goals and not np.allclose(unique_goals[index], goal, atol=1e-8, rtol=0):
            raise ValueError("Same-frame functional regions require consistent pose candidates")
        unique_goals[index] = goal
    goal_pairs = list(unique_goals.items())
    motion = task.get("motion", "free")
    model_spec = scene.get("motion_model")
    if motion not in {"free", "auto"} and (model_spec is None or model_spec.get("kind") != motion):
        raise ValueError("Constrained motion requires a compatible evidenced current motion_model")
    if task["relation"] == "articulation" and (model_spec is None
            or model_spec.get("kind") not in {"prismatic", "revolute"}):
        raise ValueError("Articulation requires the current instance's supported mechanism model")
    function_frame = _pose_array(first["binding"]["carrier_frame"], "Current functional frame")
    if task["relation"] == "tool_use":
        if model_spec is not None or motion not in {"free", "auto"}:
            raise ValueError("The function optimizer cannot silently discard a current mechanism constraint")
        from .function_motion import transfer_function_motion
        solved = transfer_function_motion(reference[:last_held + 1] @ function_frame,
                                          initial @ function_frame,
                                          [{"index": index, "pose": goal @ function_frame}
                                           for index, goal in goal_pairs])
        objects = solved["poses"] @ _inverse(function_frame)
        motion_checks = solved["diagnostics"]
    else:
        objects, motion_checks = _event_transfer(reference[:last_held + 1], initial,
                                                 goal_pairs, function_frame, model_spec)
    checks = []
    for event in events:
        binding = event["binding"]
        pose = objects[event["index"]]
        if event["constraint_kind"] == "on_region":
            from .pipeline import _surface
            from .goals import check_on
            from .regions import check_region_on
            face = binding["on_geometry"]
            surface = _surface(scene["support"])
            error = check_region_on(pose, face["carrier_region_points"],
                                    face["carrier_outward_normal"], surface,
                                    margin=scene.get("margin", 0.),
                                    gap_tolerance=face["gap_tolerance"],
                                    normal_tolerance=face["normal_tolerance"])
            error["body_checks"] = check_on(pose, scene["body_points"], surface)
            if (not error["valid"] or error["body_checks"]["support_gap_m"] > 1e-7
                    or error["body_checks"]["minimum_boundary_clearance_m"] < scene.get("margin", 0.) - 1e-7):
                raise ValueError("Transferred function event violates the specified support face or full body bounds")
        else:
            error = relation_error(_inverse(_pose_array(scene["reference_pose"], "Reference pose")) @ pose,
                                   binding["carrier_frame"],
                                   binding["reference_frame"], event["functional_relation"])
            if error["position_error_m"] > event["position_tolerance_m"] or error["rotation_error_rad"] > event["rotation_tolerance_rad"]:
                raise ValueError(f"Transferred trajectory misses functional event {event['event_id']!r}")
        if task["relation"] == "inside":
            if "interior" in scene:
                from .pipeline import _interior
                from .goals import check_inside
                inside = check_inside(pose, scene["body_points"], _interior(scene["interior"]))
                if inside["minimum_clearance_m"] < scene.get("margin", 0.) - 1e-7:
                    raise ValueError("Function event violates supplied free interior geometry")
                error["containment"] = {"status": "verified_on_supplied_geometry", **inside}
            else:
                error["containment"] = {"status": "not_verified", "reason": "free interior geometry not supplied"}
        checks.append({"event_id": event["event_id"], "frame_index": event["frame_index"],
                       "constraint_kind": event["constraint_kind"],
                       "goal_pose": event["goal_pose"], **error})
    if "entry_region" in scene:
        # Preserve the existing geometry contract: this index addresses source
        # trajectory rows, unlike functional event frame IDs. Only controlled
        # object nodes exist here; released body poses require new observation.
        from .pipeline import _interior
        from .goals import check_inside
        begin = scene.get("entry_start_index")
        if type(begin) is not int or not 0 <= begin < len(objects):
            raise ValueError("entry_start_index must address a controlled object trajectory row")
        corridor = _interior(scene["entry_region"])
        clearance = min(check_inside(pose, scene["body_points"], corridor)["minimum_clearance_m"]
                        for pose in objects[begin:])
        if clearance < margin - 1e-7:
            raise ValueError("Transferred object motion leaves the supplied entry corridor")
        motion_checks["entry_corridor"] = {"minimum_clearance_m": float(clearance),
                                           "margin_m": float(margin),
                                           "checked_nodes": len(objects) - begin,
                                           "scope": "saved controlled nodes; not continuous collision certification"}
    return objects, checks, motion_checks, reference, first


def ground_object_program(program, scene):
    """Return only function-constrained body motion when no attachment is known.

    Human tool demonstrations may establish object motion before robot grasp
    binding exists. This geometry-only entry preserves their function events
    without fabricating an EEF trajectory. The complete skill consumer requires
    actual holding separately. It uses exactly the same object numerical core.
    """
    if program.get("skill_kind") != "manipulation":
        raise ValueError("An object-only function program must be manipulation")
    for key in ("carrier_id", "reference_id", "reference_frame"):
        if not isinstance(scene.get(key), str) or not scene[key]:
            raise ValueError(f"Current function geometry needs a nonempty {key}")
    if scene["carrier_id"] == scene["reference_id"]:
        raise ValueError("Function carrier and reference must be distinct")
    demo = program["demo"]
    frames, times = validate_timeline(demo["frame_indices"], demo["timestamps"])
    source = _poses_array(demo["T_reference_carrier"], "T_reference_carrier")
    if len(source) != len(frames):
        raise ValueError("Object function motion must cover the original timeline")
    last = len(frames) - 1 if demo.get("release_frame") is None else _index(
        demo["release_frame"], frames, "release_frame")
    initial = _pose_array(scene["initial_pose"], "Current carrier pose")
    objects, checks, motion_checks, _, _ = _object_reference(
        demo, scene, frames, source, last, program["task"], initial)
    result = {"status": "candidate", "output_type": "object_trajectory",
              "carrier_id": scene["carrier_id"], "reference_id": scene["reference_id"],
              "reference_frame": scene["reference_frame"], "poses": objects,
              "frame_indices": frames[:last + 1], "timestamps": times[:last + 1],
              "task": program["task"], "function_checks": checks,
              "motion_checks": motion_checks,
              "pose_convention": "column-vector reference_from_fixed_body; metres; 4x4 SE(3)",
              "path_scope": "function-constrained object reference; no inferred EEF or execution claim"}
    if "body_points" in scene:
        result["body_points"] = scene["body_points"]
    return result


def ground_program(program, scene):
    """Generate one grasp/manipulation geometry reference from a compiled demo.

    ``program`` contains ``skill_kind``, ``demo`` and (for manipulation) task
    ``relation``/``motion``. ``demo.functional_events`` explicitly name source
    frame IDs, two owner-local functional frames, their source, and tolerances.
    ``scene.functional_bindings[event_id]`` supplies the two current functional
    frames, owner IDs and evidence source. This does not assume congruent shapes.
    ``scene.reference_pose`` maps the current reference owner's local frame to
    ``scene.reference_frame`` (the common output frame); identity is explicit
    when those frames coincide. Initial poses and scene geometry use the latter.

    Manipulation additionally requires caller-verified ``scene.holding`` with
    measured ``T_carrier_eef`` and its evidence source, plus synchronized
    ``initial_pose`` and ``eef_pose``. ``release_frame`` is an observed source
    phase boundary, independent of the opening command and function event. A
    future release cannot already be observed: the returned retreat is anchored
    to its predicted boundary and must be rebound to actual feedback at runtime.
    """
    kind, demo = program["skill_kind"], program["demo"]
    frame_name = scene.get("reference_frame")
    if not isinstance(frame_name, str) or not frame_name:
        raise ValueError("The current scene must name its common reference_frame")
    if kind == "grasp":
        result = ground_grasp(demo, scene)
        result.update(skill_kind=kind, reference_frame=frame_name,
                      carrier_id=scene.get("carrier_id"),
                      pose_convention="column-vector reference_from_eef; metres; 4x4 SE(3)")
        return result
    if kind != "manipulation":
        raise ValueError("skill_kind must be grasp or manipulation")
    for key in ("tool_id", "tool_frame"):
        if not isinstance(demo.get(key), str) or not demo[key] or demo[key] != scene.get(key):
            raise ValueError(f"Manipulation requires the same demonstrated/current {key}")
    for key in ("carrier_id", "reference_id"):
        if not isinstance(scene.get(key), str) or not scene[key]:
            raise ValueError(f"Current manipulation needs a nonempty {key}")
    if scene["carrier_id"] == scene["reference_id"]:
        raise ValueError("Carrier and reference must be distinct rigid entities")
    frames, times = validate_timeline(demo["frame_indices"], demo["timestamps"])
    source = _poses_array(demo["T_reference_carrier"], "T_reference_carrier")
    source_eef = _poses_array(demo["T_carrier_eef"], "T_carrier_eef")
    if len(source) != len(frames) or len(source_eef) != len(frames):
        raise ValueError("Both source trajectories must cover the original observation timeline")
    commands = _intentions(demo["gripper_events"], frames)
    release = demo.get("release_frame")
    last_held = len(frames) - 1 if release is None else _index(release, frames, "release_frame")
    if release is None and any(event["command"] == "open" for event in commands):
        raise ValueError("An opening skill needs an explicit release boundary; command time is not that boundary")
    if release is not None and not any(event["command"] == "open" and event["frame_index"] <= release
                                      for event in commands):
        raise ValueError("The released phase needs an explicit opening intention before its boundary")
    holding = scene.get("holding", {})
    if holding.get("verified") is not True or not holding.get("source"):
        raise ValueError("Manipulation requires caller-verified actual holding, not a grasp candidate")
    actual = _pose_array(holding["T_carrier_eef"], "Actual T_carrier_eef")
    initial = _pose_array(scene["initial_pose"], "Current carrier pose")
    eef_initial = _pose_array(scene["eef_pose"], "Current EEF pose")
    if not np.allclose(initial @ actual, eef_initial, atol=1e-6, rtol=0):
        raise ValueError("Actual holding transform and current object/EEF observations are not synchronized")
    task = program["task"]
    if task.get("relation") not in {"transfer", "on", "inside", "articulation", "tool_use"}:
        raise ValueError("Unsupported functional relation")
    objects, checks, motion_checks, reference, first = _object_reference(
        demo, scene, frames, source, last_held, task, initial)
    eef = np.empty_like(source_eef)
    eef[:last_held + 1] = attached_eef_reference(objects, actual)
    if last_held < len(frames) - 1:
        # After release, the hand moves independently of the object. Do not
        # move a placed body to repair a hand join, or extrapolate an attachment.
        bound_hand = reference[last_held:] @ first["carrier_binding"] @ source_eef[last_held:]
        eef[last_held:] = retreat_from_release(bound_hand, eef[last_held])
    return {"status": "candidate", "skill_kind": kind, "output_type": "interaction_trajectory",
            "reference_frame": frame_name, "carrier_id": scene["carrier_id"],
            "reference_id": scene["reference_id"], "task": task,
            "object_poses": objects, "object_frame_indices": frames[:last_held + 1],
            "eef_poses": eef, "frame_indices": frames, "timestamps": times,
            "gripper_events": commands, "release_frame": release,
            "function_checks": checks, "motion_checks": motion_checks,
            "tool_id": demo["tool_id"], "tool_frame": demo["tool_frame"],
            "holding_source": holding["source"],
            "released_object_state": "requires_observation" if release is not None else "still_attached_reference",
            "retreat_anchor": "predicted_release_eef; rebind from actual release feedback" if release is not None else None,
            "pose_convention": "column-vector reference_from_body/eef; metres; 4x4 SE(3)",
            "path_scope": "geometry reference; robot IK, collision, forces and timing remain external"}
# END region-based skill program consumer.
