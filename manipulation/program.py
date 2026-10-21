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

from .grasp import attached_eef_reference, ground_grasp, validate_timeline
from .motion import (MotionModel, _errors, _inverse, _pose_array, _poses_array,
                     _tolerances, transfer_motion)
from .regions import ground_relation, relation_error


# BEGIN separate optional reference fitting from current task feasibility.
class ReferenceGenerationError(ValueError):
    """A source-profile fit failed after validating the independent goals."""


def _mechanism_entry_contract(model_spec, initial, *, require_lift=False):
    """Validate immutable actual entry without selecting a goal-set member.

    A full-frame target's centre need not lie on the mechanism or within its
    limits: another pose in the declared task allowed set may be feasible.
    The typed goal and physical mechanism remain separate solver obligations.
    Native scalar goals additionally admit exact interval-intersection checks
    where their signed coordinate meaning is explicitly supplied.
    """
    if model_spec is None:
        return None, None
    model, limits = _current_model(model_spec)
    start = _current_initial_coordinates(model_spec, model, initial, require_lift=require_lift)
    if limits is not None and (np.any(start < limits[:, 0]-1e-9) or np.any(start > limits[:, 1]+1e-9)):
        raise ValueError("The observed current start exceeds current mechanism limits")
    return model, limits


def _unfitted_reference(reference, model, error, coordinates=None):
    """Retain the bound source as a seed, with no feasibility claim about it."""
    output = reference.copy()
    if model is None:
        serialized = {"kind": "free", "origin": "bound source reference"}
    else:
        model.coordinates = (model._coordinates_unchecked(output) if coordinates is None
                             else np.asarray(coordinates, float).copy())
        output = model.poses(model.coordinates)
        serialized = model.to_dict()
    return output, {"model": serialized, "reference_generation": {
        "status": "failed", "reason": str(error),
        "seed": "bound source reference; not a task feasibility result"}}
# END separate optional reference fitting from current task feasibility.


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
        # BEGIN retain source-end commands for the shared physical clock.
        # The planner/device consumers check post-activation coverage. Source
        # validation only checks identity and declared ordering, not an
        # assumed absence of control intent at the final observation.
        if (not isinstance(name, str) or not name or name in seen or index <= previous):
            raise ValueError("Gripper events must be unique and ordered on original source observations")
        # END source ownership never certifies actual command execution.
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
    ``initial_coordinates`` with ``initial_coordinates_source`` binds a fresh
    actual lifted angular branch. A turn-sensitive request without this input
    requires exactly one angular member within its physical coordinate limits.
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


# BEGIN actual observed angular branch is independent of reference and target origin.
def _current_initial_coordinates(model_spec, model, pose, *, require_lift=False):
    """Preserve an evidenced lifted actual coordinate after checking its FK.

    Without an explicit observation, pose alone fixes only a principal angle.
    An original entry-delta target has its own separate immutable origin and
    must not be inferred from this fresh actual measurement during a rebind.
    """
    if "initial_coordinates" not in model_spec:
        if "initial_coordinates_source" in model_spec:
            raise ValueError("Actual coordinate evidence requires its initial_coordinates value")
        coordinates = _exact_coordinates(model, pose)
        angular = 0 if model.kind == "revolute" else 2 if model.kind == "planar" else None
        if angular is not None:
            limits = model_spec.get("coordinate_limits")
            if limits is not None:
                low, high = np.asarray(limits, float)[angular]
                first = int(np.ceil((low-coordinates[angular]-1e-9)/(2*np.pi)))
                last = int(np.floor((high-coordinates[angular]+1e-9)/(2*np.pi)))
                if first > last:
                    raise ValueError("The actual initial angle has no member within current coordinate limits")
                if first != last:
                    raise ValueError("Ambiguous actual angular branch requires observed initial_coordinates and initial_coordinates_source")
                coordinates[angular] += 2*np.pi*first
                model.diagnostics["initial_coordinate_convention"] = "unique actual angular branch within current physical limits"
            elif require_lift:
                raise ValueError("An actual angular turn obligation requires observed initial_coordinates and initial_coordinates_source")
        return coordinates
    coordinates = np.asarray(model_spec["initial_coordinates"])
    if (coordinates.shape != (model.dimension,) or coordinates.dtype.kind not in "iuf"
            or not np.isfinite(coordinates).all()):
        raise ValueError("Observed initial_coordinates must match the current mechanism dimension")
    source = _joint_source(model_spec.get("initial_coordinates_source"), "Observed initial_coordinates")
    p, r = _errors(pose[None], model.poses(coordinates))
    if p[0] > 1e-8 or r[0] > 1e-8:
        raise ValueError("Observed initial_coordinates disagree with the actual current pose")
    model.diagnostics["initial_coordinates_source"] = source
    model.diagnostics["initial_coordinate_convention"] = "explicit observed signed/lifted current coordinate"
    return coordinates.astype(float, copy=True)
# END same SE(3) does not erase supplied actual turns.


# BEGIN native coordinate events share the existing constrained transfer kernel.
def _event_transfer(reference, start, goals, function_frame, model_spec=None, *,
                    timestamps, coordinate_reference=None, coordinate_goals=None):
    """Transfer in a physical function frame, preserving every internal event.

    Free intervals reuse the existing SE(3) transfer kernel on a material
    functional frame, not on an arbitrary body origin. Constrained intervals
    stay in one caller-bound current FK model; we never fit a new axis after
    each event. A model projection is allowed only for the soft reference.
    This fitting attempt uses nominal event centres on the model. Failure to
    fit one such centre is a reference-generation failure; the separately
    retained typed goal allowed sets do not acquire exact pose equalities.

    Native joint events additionally supply their complete coordinate reference
    and indexed coordinate goals. These are not recovered from their FK poses:
    doing so would discard explicitly recorded revolute winding numbers.

    Native prismatic intervals between hard events retain the source's scalar
    progress profile while adapting its displacement to the new endpoints.
    The unanchored suffix after the final event retains source metric increments;
    this does not rescale the final controlled object motion before release.

    Endpoint corrections use the recorded timestamp fraction within each
    interval. Source sample density cannot redefine that fraction. Native
    prismatic progress scaling remains a source-coordinate operation, not a
    time correction; its pauses and reversals must retain their own meaning.
    """
    # BEGIN every time-weighted interval consumes the actual source timeline.
    _, times = validate_timeline(np.arange(len(reference)), timestamps)

    def interval_progress(begin, end):
        part = times[begin:end+1]
        return np.zeros(1) if begin == end else (part-part[0])/(part[-1]-part[0])
    # END a single-state suffix has no invented duration.
    knots = [(0, start)] + [(index, goal) for index, goal in goals if index != 0]
    if goals and goals[0][0] == 0 and not np.allclose(goals[0][1], start, atol=1e-8, rtol=0):
        raise ReferenceGenerationError("The nominal initial reference event differs from the actual entry member")
    output = np.empty_like(reference)
    diagnostics = {"phase": "recorded timestamp fraction within each functional interval; no execution timing inferred"}
    native = coordinate_reference is not None
    if native != (coordinate_goals is not None) or (native and model_spec is None):
        raise ValueError("Native coordinate reference/goals require one current mechanism model")
    if model_spec is None:
        # Right multiplication preserves the physical function origin and its
        # lever arm while the existing free kernel enforces each endpoint.
        frame = _pose_array(function_frame, "Current material function frame")
        function_reference = reference @ frame
        for position, (begin, initial) in enumerate(knots):
            end, goal = knots[position + 1] if position + 1 < len(knots) else (len(reference) - 1, None)
            result = transfer_motion(function_reference[begin:end + 1], initial @ frame,
                                     goal_pose=None if goal is None else goal @ frame,
                                     progress=interval_progress(begin, end))
            output[begin:end + 1] = result["poses"] @ _inverse(frame)
        diagnostics["model"] = {"kind": "free", "origin": "material functional frame"}
    else:
        model, limits = _current_model(model_spec)
        coordinates = (np.asarray(coordinate_reference, float).copy() if native
                       else model._coordinates_unchecked(reference))
        if (coordinates.shape != (len(reference), model.dimension)
                or not np.isfinite(coordinates).all()):
            raise ValueError("Coordinate reference must align with all reference poses")
        # Projection of noisy reference geometry is reported, not re-labelled
        # as evidence that the reference already lies on the mechanism.
        p, r = _errors(reference, model.poses(coordinates))
        if native and (p.max() > 1e-8 or r.max() > 1e-8):
            raise ValueError("Native coordinate reference and its FK poses disagree")
        if native and 0 in coordinate_goals and not np.allclose(
                coordinate_goals[0], coordinates[0], atol=1e-8, rtol=0):
            raise ValueError("The initial coordinate goal conflicts with the observed current pose")
        diagnostics["reference_projection"] = {"max_position_m": float(p.max()),
                                                "max_rotation_rad": float(r.max())}
        angular = 0 if model.kind == "revolute" else 2 if model.kind == "planar" else None
        values = coordinates.copy()
        # BEGIN native prismatic endpoint changes preserve demonstrated progress.
        if native and model.kind == "prismatic":
            diagnostics["coordinate_transfer"] = {
                "method": "piecewise_source_progress_affine",
                "intervals": [],
                "post_final_event": "unchanged source metric increments from the bound final event",
            }
        # END native prismatic progress diagnostics.
        for position, (begin, initial) in enumerate(knots):
            end, goal = knots[position + 1] if position + 1 < len(knots) else (len(reference) - 1, None)
            try:
                initial_q = ((coordinates[0].copy() if begin == 0 else coordinate_goals[begin].copy())
                             if native else _exact_coordinates(model, initial))
            except ValueError as error:
                raise ReferenceGenerationError("A nominal reference knot is outside the current mechanism") from error
            if angular is not None and not native:
                initial_q[angular] += 2 * np.pi * np.round(
                    (coordinates[begin, angular] - initial_q[angular]) / (2 * np.pi))
            part = coordinates[begin:end + 1] - coordinates[begin] + initial_q
            if goal is not None:
                try:
                    goal_q = coordinate_goals[end].copy() if native else _exact_coordinates(model, goal)
                except ValueError as error:
                    raise ReferenceGenerationError("A nominal reference goal is outside the current mechanism") from error
                if angular is not None and not native:
                    goal_q[angular] += 2 * np.pi * np.round(
                        (part[-1, angular] - goal_q[angular]) / (2 * np.pi))
                # BEGIN native prismatic progress instead of a time-linear drift.
                if native and model.kind == "prismatic":
                    source_delta = float(coordinates[end, 0] - coordinates[begin, 0])
                    target_delta = float(goal_q[0] - initial_q[0])
                    if not np.isfinite([source_delta, target_delta]).all():
                        raise ValueError("Prismatic endpoint displacements must be finite")
                    if source_delta == 0.:
                        # A return-to-start motion contains no net-progress
                        # denominator. Preserve its actual loop or pause when
                        # both targets agree; never invent a time drift to a
                        # different endpoint or erase the intervening motion.
                        # Current q comes from FK dot products; the same
                        # supplied coordinate can differ by a few arithmetic
                        # ULPs. This bound addresses float64 roundoff only,
                        # not camera noise or a permissible task displacement.
                        roundoff = 8 * np.finfo(float).eps * max(
                            1., abs(float(initial_q[0])), abs(float(goal_q[0])))
                        if abs(target_delta) > roundoff:
                            raise ReferenceGenerationError("Zero-net prismatic source interval cannot fit a nonzero displacement")
                        scale, mode = 1., "zero_net_source_motion_preserved"
                    elif np.array_equal(goal_q, part[-1]):
                        # Identity retargeting must not needlessly round the
                        # already translated source coordinates a second time.
                        scale, mode = 1., "source_metric_increments_preserved"
                    else:
                        scale, mode = target_delta / source_delta, "source_progress_scaled"
                        if not np.isfinite(scale):
                            raise ReferenceGenerationError("Prismatic source progress scale must be finite")
                        part = initial_q + scale * (coordinates[begin:end + 1] - coordinates[begin])
                        # Pin only floating-point endpoint arithmetic, not any
                        # interior sample or its order. A positive scale keeps
                        # every source increment's sign, including real pauses
                        # and reversals; a negative explicit goal reverses it.
                        part[0], part[-1] = initial_q, goal_q
                    diagnostics["coordinate_transfer"]["intervals"].append({
                        "begin_index": begin, "end_index": end, "method": mode,
                        "source_displacement": source_delta,
                        "target_displacement": target_delta, "progress_scale": scale,
                    })
                else:
                    # The revolute branch keeps its existing explicit unwrapped
                    # coordinate and winding contract. This timestamp-fraction
                    # correction introduces no modulo-angle reinterpretation.
                    part += interval_progress(begin, end)[:, None] * (goal_q - part[-1])
                # END native prismatic source-progress endpoint retargeting.
            values[begin:end + 1] = part
        if limits is not None and (np.any(values < limits[:, 0] - 1e-9)
                                   or np.any(values > limits[:, 1] + 1e-9)):
            raise ReferenceGenerationError("Transferred reference exceeds current mechanism limits")
        output = model.poses(values)
        model.coordinates = values
        diagnostics["model"] = model.to_dict()
        diagnostics["limit_status"] = "checked_at_nodes" if limits is not None else "not_supplied"
    if not np.allclose(output[0], start, atol=1e-7, rtol=0):
        raise ValueError("Function transfer changed the observed initial object pose")
    return output, diagnostics
# END native coordinate events share the existing constrained transfer kernel.


# BEGIN direct functional joint goals without manufactured region frames.
def _coordinate_vector(value, name):
    """One prismatic distance in metres or one unwrapped revolute angle."""
    vector = np.asarray(value)
    if (vector.shape != (1,) or vector.dtype.kind not in "iuf"
            or not np.isfinite(vector).all()):
        raise ValueError(f"{name} must be one finite numeric coordinate in a length-one vector")
    return vector.astype(float, copy=True)


def _joint_source(value, name):
    """Evidence may be a description or an existing structured source record."""
    if (not isinstance(value, (str, dict)) or not value
            or (isinstance(value, str) and not value.strip())):
        raise ValueError(f"{name} requires a nonempty evidence source")
    return deepcopy(value)


def _joint_reference(demo, scene, frames, last_held, task, initial):
    """Bind measured source coordinates and explicit current coordinate goals.

    Source ``motion_coordinates`` cover every original observation; only their
    controlled prefix through release generates a reference. Their evidence is in
    ``motion_coordinates_source``. Every event declares ``joint_coordinate``,
    ``coordinate_value`` at its original source frame and a positive
    ``coordinate_tolerance``. The current binding declares
    ``current_coordinates`` in the caller's current model, matching owner IDs
    and its own evidence source. Its ``coordinate_mode`` is ``absolute`` by
    default; ``entry_delta`` instead adds these values to explicitly supplied
    original-request ``entry_coordinates`` with ``entry_source`` evidence.
    The increment origin never follows a continuation's new actual state.
    Distance units are metres; angle units are
    unwrapped radians. The caller must bind the same signed coordinate sense
    on source and current mechanisms; fitting the source does not establish a
    current joint axis or a new instance's limits.

    The observed initial pose and any evidenced lifted coordinate define q0. The
    soft reference is q_ref=q_demo-q_demo[0]+q0. Native prismatic intervals
    between hard events then scale their source progress to the explicit
    current endpoint displacement; same-endpoint zero-net loops remain intact.
    After the final event, metric source increments remain unchanged.
    For revolute motion, a pose cannot reveal accumulated revolutions. Supply
    ``motion_model.initial_coordinates`` and its ``initial_coordinates_source``
    or physical limits selecting exactly one equivalent angle. Merely setting
    a reference pose does not observe the actual turn count. Original target
    origins and fresh current lifted coordinates remain independent inputs.
    Functional goals are evaluated directly by current FK, without a reference-owner
    functional frame or source/current body correspondence. Source hand motion
    and body poses are not inputs to this explicitly scalar route.
    """
    if task.get("relation") != "articulation":
        raise ValueError("Joint-coordinate functional events require an articulation relation")
    spec = scene.get("motion_model")
    if not isinstance(spec, dict) or spec.get("kind") not in {"prismatic", "revolute"}:
        raise ValueError("Joint-coordinate goals require an evidenced current articulation motion_model")
    if task.get("motion", "free") not in {"free", "auto", spec["kind"]}:
        raise ValueError("Constrained motion requires a compatible evidenced current motion_model")
    model, limits = _mechanism_entry_contract(spec, initial, require_lift=True)
    source_evidence = _joint_source(demo.get("motion_coordinates_source"), "motion_coordinates_source")
    raw = np.asarray(demo.get("motion_coordinates"))
    if raw.ndim == 1:
        raw = raw[:, None]
    if (raw.shape != (len(frames), 1) or raw.dtype.kind not in "iuf"
            or not np.isfinite(raw).all()):
        raise ValueError("motion_coordinates must be finite N or N-by-1 values on the original timeline")
    source_coordinates = raw.astype(float, copy=True)
    initial_coordinates = _current_initial_coordinates(spec, model, initial, require_lift=True)
    coordinates = source_coordinates-source_coordinates[0]+initial_coordinates
    reference = model.poses(coordinates)

    events = demo.get("functional_events")
    if not isinstance(events, list) or not events:
        raise ValueError("Manipulation requires explicit functional_events")
    bindings = scene.get("functional_bindings", {})
    if not isinstance(bindings, dict):
        raise ValueError("functional_bindings must map event IDs to current joint goals")
    names, previous, unique_goals, recorded = set(), -1, {}, []
    for event in events:
        if not isinstance(event, dict) or event.get("constraint_kind") != "joint_coordinate":
            raise ValueError("An articulation program cannot mix joint-coordinate and frame functional events")
        name = event.get("event_id")
        index = _index(event.get("frame_index"), frames, "functional event")
        if (not isinstance(name, str) or not name or name in names
                or index < previous or index > last_held):
            raise ValueError("Functional events must be unique, ordered and occur within controlled object motion")
        evidence = _joint_source(event.get("source"), "Functional joint event")
        tolerance = _positive(event.get("coordinate_tolerance"), "coordinate_tolerance")
        source_value = _coordinate_vector(event.get("coordinate_value"), "Source coordinate_value")
        if np.max(np.abs(source_value-source_coordinates[index])) > tolerance:
            raise ValueError("Source functional coordinate_value disagrees with its original observation")
        binding = bindings.get(name)
        if not isinstance(binding, dict):
            raise ValueError(f"Functional event {name!r} requires an evidenced current joint binding")
        binding_source = _joint_source(binding.get("source"), "Current joint binding")
        if binding.get("carrier_id") != scene["carrier_id"] or binding.get("reference_id") != scene["reference_id"]:
            raise ValueError("Functional binding owners disagree with the selected current entities")
        # BEGIN absolute and original-entry increment targets keep distinct identities.
        coordinate_mode = binding.get("coordinate_mode", "absolute")
        if coordinate_mode not in {"absolute", "entry_delta"}:
            raise ValueError("Current joint coordinate_mode must be absolute or entry_delta")
        raw_target = _coordinate_vector(binding.get("current_coordinates"), "Current functional coordinates")
        entry, entry_source = None, None
        if coordinate_mode == "entry_delta":
            entry = _coordinate_vector(binding.get("entry_coordinates"), "Original request entry coordinates")
            entry_source = _joint_source(binding.get("entry_source"), "Original request coordinate entry")
            target = entry+raw_target
        else:
            if "entry_coordinates" in binding or "entry_source" in binding:
                raise ValueError("Absolute coordinate goals do not declare an entry-delta origin")
            target = raw_target
        # END no continuation's new actual state resets an original increment target.
        # BEGIN nominal centres are preferences inside original scalar allowed sets.
        # Preserve every event and its target below. Only the common interval
        # selects a reference member; it never rewrites any task obligation.
        unique_goals.setdefault(index, target.copy())
        # END distinct centres may describe compatible simultaneous requirements.
        recorded.append({"event_id": name, "index": index, "frame_index": int(frames[index]),
                         "source_coordinate_value": source_value, "target_coordinates": target,
                         "coordinate_tolerance": tolerance, "source": evidence,
                         "binding_source": binding_source})
        recorded[-1].update(coordinate_mode=coordinate_mode, raw_target_coordinates=raw_target,
                            entry_coordinates=entry, entry_source=entry_source)
        names.add(name)
        previous = index
    # BEGIN intersect explicitly declared scalar bands and current physical limits.
    # These one-dimensional intersections are exact feasibility checks; no
    # reference centre or principal-angle surrogate can reject another member.
    for index, preferred in unique_goals.items():
        same_scope = [event for event in recorded if event["index"] == index]
        lower = max(float(event["target_coordinates"][0]-event["coordinate_tolerance"]) for event in same_scope)
        upper = min(float(event["target_coordinates"][0]+event["coordinate_tolerance"]) for event in same_scope)
        if limits is not None:
            lower, upper = max(lower, float(limits[0, 0])), min(upper, float(limits[0, 1]))
        if lower > upper:
            raise ValueError("Coordinate goal allowed sets do not intersect each other or current mechanism limits")
        if index == 0:
            if not lower <= initial_coordinates[0] <= upper:
                raise ValueError("The initial coordinate goal allowed set excludes the observed current pose")
            unique_goals[index] = initial_coordinates.copy()
        else:
            unique_goals[index] = np.asarray([min(upper, max(lower, float(preferred[0])))])
    # END clipping selects a seed member only; actual state and original goals stay intact.
    goals = [(index, model.poses(value)[0]) for index, value in unique_goals.items()]
    # BEGIN bind native goals before attempting the optional source-profile fit.
    try:
        objects, diagnostics = _event_transfer(
            reference[:last_held+1], initial, goals, None, spec,
            # BEGIN phase weights come from these exact controlled source times.
            timestamps=demo["timestamps"][:last_held+1],
            # END no goal, release or inserted frame creates another clock.
            coordinate_reference=coordinates[:last_held+1], coordinate_goals=unique_goals)
    except ReferenceGenerationError as error:
        objects, diagnostics = _unfitted_reference(
            reference[:last_held+1], model, error, coordinates[:last_held+1])
    # END bind native goals before attempting the optional source-profile fit.
    actual = np.asarray(diagnostics["model"]["coordinates"], float)
    checks = []
    for event in recorded:
        index, target = event["index"], event["target_coordinates"]
        goal = model.poses(target)[0]
        error = float(np.max(np.abs(actual[index]-target)))
        p, r = _errors(objects[index:index+1], goal[None])
        checks.append({key: value for key, value in event.items() if key != "index"})
        checks[-1].update(constraint_kind="joint_coordinate", goal_pose=goal,
                          status="bound_goal", reference_satisfies_goal=error <= event["coordinate_tolerance"],
                          reference_coordinates=actual[index].copy(), reference_coordinate_error=error,
                          coordinate_unit="m" if model.kind == "prismatic" else "rad",
                          position_error_m=float(p[0]), rotation_error_rad=float(r[0]))
    diagnostics.update(source_coordinates_source=source_evidence,
                       source_initial_coordinates=source_coordinates[0].copy(),
                       current_initial_coordinates=initial_coordinates.copy(),
                       initial_coordinate_convention=(
                           "explicit observed signed/lifted current coordinate"
                           if "initial_coordinates" in spec else
                           model.diagnostics.get("initial_coordinate_convention", "principal relative angle from the current reference pose; pre-existing revolutions are not inferred")
                           if model.kind == "revolute" else
                           "metric displacement from the current reference pose"),
                       current_initial_coordinates_source=deepcopy(spec.get("initial_coordinates_source")),
                       # BEGIN distinguish source reference from constrained retargeting.
                       coordinate_reference=(
                           "source coordinates shifted to observed current start; between hard events, "
                           "prismatic source progress is scaled to current endpoint displacement; "
                           "after the final event, source metric increments are retained"
                           if model.kind == "prismatic" else
                           "source unwrapped angular coordinates shifted to observed current start; "
                           "explicit winding is retained, with current functional targets enforced"))
                       # END explicit native coordinate-transfer semantics.
    return objects, checks, diagnostics


def _check_entry_corridor(objects, scene, margin, motion_checks):
    """Keep the existing optional observed-corridor check on both goal paths."""
    if "entry_region" not in scene:
        return
    from .pipeline import _interior
    from .goals import check_inside
    begin = scene.get("entry_start_index")
    if type(begin) is not int or not 0 <= begin < len(objects):
        raise ValueError("entry_start_index must address a controlled object trajectory row")
    corridor = _interior(scene["entry_region"])
    clearance = min(check_inside(pose, scene["body_points"], corridor)["minimum_clearance_m"]
                    for pose in objects[begin:])
    motion_checks["entry_corridor"] = {"minimum_clearance_m": float(clearance),
                                       "margin_m": float(margin),
                                       "reference_satisfies_corridor": bool(clearance >= margin-1e-7),
                                       "checked_nodes": len(objects)-begin,
                                       "scope": "saved controlled nodes; not continuous collision certification"}
# END direct functional joint goals without manufactured region frames.


def _functional_events(demo, scene, frames, poses, last_held, task, initial):
    events = demo.get("functional_events")
    if not isinstance(events, list) or not events:
        raise ValueError("Manipulation requires explicit functional_events; release/end frames are not substitutes")
    current = scene.get("functional_bindings", {})
    # BEGIN unordered support: an on relation has no measured target tangent
    # or material origin. Its finite support is already in the output frame.
    # Other relations retain their complete two-sided functional coordinates.
    on_relation = task["relation"] == "on"
    reference_pose = (None if on_relation else
                      _pose_array(scene["reference_pose"], "Current reference-owner pose"))
    carrier_mapping = None
    # END unordered support coordinate entrance.
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
        if on_relation:
            # BEGIN direct partial on relation. A_C maps source-carrier local
            # coordinates into the current carrier's fixed local coordinates.
            # Identity is explicit, with provenance in binding.source; it is
            # never inferred from similar labels or silently filled in here.
            if "carrier_mapping" not in binding:
                raise ValueError("On binding requires an explicit evidenced carrier_mapping")
            mapping = _pose_array(binding["carrier_mapping"], "On carrier_mapping")
            if carrier_mapping is None:
                carrier_mapping = mapping
            elif not np.allclose(mapping, carrier_mapping, atol=1e-8, rtol=0):
                raise ValueError("On events for one carrier require the same carrier_mapping")
            # P_t = X_0 A_C S_0^-1 S_t A_C^-1. Source-reference changes cancel,
            # while nonidentity carrier mappings retain their lever-arm effect.
            # This is a motion/pose preference; no shelf PCA axis participates.
            source_binding = initial @ mapping @ _inverse(poses[0])
            preferred = source_binding @ poses[index] @ _inverse(mapping)
            function_frame = mapping @ _pose_array(event["carrier_frame"], "Source material function frame")
            from .pipeline import _surface
            from .regions import SupportReferenceNotFound, ground_on
            face = binding["on_geometry"]
            surface = _surface(scene["support"])
            # The free tangential-position preference belongs to a material
            # function point, not an arbitrary carrier coordinate origin.
            # Normal alignment can rotate a translated body-local frame; if
            # its origin were held fixed, merely changing that coordinate
            # gauge would move the physical goal. Re-express all geometry in
            # the already evidenced material frame before placement, then
            # convert the candidate back. No target tangent is introduced.
            function_body = _inverse(function_frame)
            body_points = np.asarray(scene["body_points"], float)
            contact_points = np.asarray(face["carrier_region_points"], float)
            # BEGIN optional support initialization cannot delete the goal.
            # Catch only its dedicated nominal-member/search failure. Invalid
            # geometry and missing task inputs retain their original errors.
            # An unavailable preference branch or exact-normal member is only
            # a seed failure; SupportRequirement still consumes all geometry.
            reference_failure = None
            try:
                grounded = ground_on(preferred @ function_frame,
                                    body_points @ function_body[:3, :3].T + function_body[:3, 3],
                                    contact_points @ function_body[:3, :3].T + function_body[:3, 3],
                                    function_body[:3, :3] @ face["carrier_outward_normal"],
                                    surface, margin=scene.get("margin", 0.),
                                    gap_tolerance=face["gap_tolerance"],
                                    normal_tolerance=face["normal_tolerance"],
                                    antiparallel_axis=face.get("antiparallel_axis"))
            except SupportReferenceNotFound as error:
                reference_failure = {"status": "failed", "reason": str(error),
                                     "scope": "optional support reference initialization"}
                grounded = {"goal_pose": preferred @ function_frame,
                            "freedom": {"required": "specified face on finite support with opposing signed normals",
                                        "reference_generation": reference_failure}}
            # END no selected nominal member is promoted to a feasibility proof.
            relation = {"goal_pose": grounded["goal_pose"] @ function_body,
                        "carrier_binding": mapping, "reference_binding": source_binding,
                        "function_frame": function_frame,
                        "constraint_kind": "on_region", "freedom": grounded["freedom"]}
            if reference_failure is not None:
                relation["reference_generation"] = reference_failure
            # END direct partial on relation.
        else:
            relation = ground_relation(poses[index], event["carrier_frame"], binding["carrier_frame"],
                                       event["reference_frame"], binding["reference_frame"])
            # Full functional frames are owner-local; embed them once in the
            # output frame, including when it differs from the reference owner.
            relation["goal_pose"] = reference_pose @ relation["goal_pose"]
            relation["reference_binding"] = reference_pose @ relation["reference_binding"]
            relation["function_frame"] = _pose_array(binding["carrier_frame"], "Current functional frame")
            relation["constraint_kind"] = "functional_frame"
        output.append({**relation, "event_id": name, "index": index,
                       "frame_index": int(frames[index]), "binding": binding,
                       "source": event["source"],
                       "position_tolerance_m": (_positive(event["position_tolerance_m"], "event position tolerance")
                                                if relation["constraint_kind"] == "functional_frame" else None),
                       "rotation_tolerance_rad": (_positive(event["rotation_tolerance_rad"], "event rotation tolerance")
                                                  if relation["constraint_kind"] == "functional_frame" else None),
                       # BEGIN preserve explicit target-local yaw declarations through grounding.
                       **{key: deepcopy(event[key]) for key in ("yaw_axis", "yaw_tolerance_rad") if key in event}})
                       # END typed pose requirements own their validation and interpretation.
        previous = index
        names.add(name)
    return output


# BEGIN preparation and planning share the material pose-event geometry.
def _material_pose_requirement(event):
    """Bind one grounded functional event without promoting its reference path."""
    from .planning_problem import PoseEventRequirement
    material = event["function_frame"]
    return PoseEventRequirement(0., material, event["goal_pose"] @ material,
        event["position_tolerance_m"], event["rotation_tolerance_rad"], event["source"],
        yaw_axis=event.get("yaw_axis"), yaw_tolerance_rad=event.get("yaw_tolerance_rad"))
# END source recipe-frame consistency checks retain their independent full-frame meaning.


def _object_reference(demo, scene, frames, last_held, task, initial):
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
    # BEGIN joint-coordinate goals use FK directly, never surrogate region frames.
    declared_events = demo.get("functional_events")
    if isinstance(declared_events, list) and any(
            isinstance(event, dict) and event.get("constraint_kind") == "joint_coordinate"
            for event in declared_events):
        result = _joint_reference(demo, scene, frames, last_held, task, initial)
        _check_entry_corridor(result[0], scene, margin, result[2])
        return result
    # END joint-coordinate goals use FK directly, never surrogate region frames.
    # BEGIN only frame-defined references consume source body pose tracks.
    source = _poses_array(demo["T_reference_carrier"], "T_reference_carrier")
    if len(source) != len(frames):
        raise ValueError("Object function motion must cover the original timeline")
    # END scalar coordinates do not require an unused cabinet/body registration.
    events = _functional_events(demo, scene, frames, source, last_held, task, initial)
    # Full-frame relations use the first functional correspondence. For on,
    # reference_binding is instead the initial-state source-coordinate map;
    # its goals are constrained by finite support, without a target frame.
    first = events[0]
    reference = first["reference_binding"] @ source @ _inverse(first["carrier_binding"])
    # BEGIN retain simultaneous allowed sets without equating their centres.
    # The first nominal member initializes reference fitting only. Every
    # original relation remains independently represented in function_checks
    # and in the downstream typed goals. Triangle-inequality envelopes can
    # prove some contradictions without solving or shrinking the intersection.
    unique_goals = {}
    for event_index, event in enumerate(events):
        index, goal = event["index"], event["goal_pose"]
        if event["constraint_kind"] == "functional_frame":
            # BEGIN validate explicit orientation data before any allowed-set test.
            requirement = _material_pose_requirement(event)
            # END the same declaration reaches the later PlanningProblem.
            for previous_event in events[:event_index]:
                if previous_event["index"] != index or previous_event["constraint_kind"] != "functional_frame":
                    continue
                position_error, rotation_error = _errors(goal[None], previous_event["goal_pose"][None])
                # BEGIN the whole yaw family lies inside this conservative SO(3) ball.
                def radius(item):
                    yaw = 0. if item.get("yaw_axis") is None else item["yaw_tolerance_rad"]
                    return min(np.pi, item["rotation_tolerance_rad"]+yaw)
                translation_bound = sum(item["position_tolerance_m"]
                    + 2*np.linalg.norm(item["function_frame"][:3, 3])*np.sin(radius(item)/2)
                    for item in (event, previous_event))
                rotation_bound = radius(event)+radius(previous_event)
                # END this contradiction proof cannot shrink a permitted twist interval.
                if position_error[0] > translation_bound or rotation_error[0] > rotation_bound:
                    raise ValueError("Same-frame functional allowed sets are provably inconsistent")
            if index == 0:
                # BEGIN current entry is compared with the task's complete allowed set.
                errors = requirement.errors_numpy(initial)
                if any(errors[name] > bound for name, bound in requirement.component_tolerances.items()):
                    raise ValueError("The initial functional allowed set excludes the observed current pose")
                # END a different nominal yaw alone cannot reject an allowed entry.
        unique_goals.setdefault(index, initial.copy() if index == 0 else goal)
    # END compatible noncentral members are left to the shared goal consumers.
    goal_pairs = list(unique_goals.items())
    motion = task.get("motion", "free")
    model_spec = scene.get("motion_model")
    if motion not in {"free", "auto"} and (model_spec is None or model_spec.get("kind") != motion):
        raise ValueError("Constrained motion requires a compatible evidenced current motion_model")
    if task["relation"] == "articulation" and (model_spec is None
            or model_spec.get("kind") not in {"prismatic", "revolute"}):
        raise ValueError("Articulation requires the current instance's supported mechanism model")
    function_frame = first["function_frame"]
    # BEGIN goals/model are authoritative; source-profile fitting is optional.
    bound_model, _ = _mechanism_entry_contract(model_spec, initial)
    if task["relation"] == "tool_use":
        if model_spec is not None or motion not in {"free", "auto"}:
            raise ValueError("The function optimizer cannot silently discard a current mechanism constraint")
        from .function_motion import FunctionReferenceError, transfer_function_motion
        # The actual-entry goal has already been checked above. The fitting
        # kernel accepts only later nodes; its minimum length/index contract
        # must not reject a valid single-state controlled prefix at release.
        fitting_goals = [{"index": index, "pose": goal @ function_frame}
                         for index, goal in goal_pairs if index > 0]
        if not fitting_goals:
            objects = reference[:last_held+1].copy()
            motion_checks = {"reference_generation": {"status": "not_needed",
                "reason": "All functional goals are at the independently checked actual entry"}}
        else:
            try:
                solved = transfer_function_motion(reference[:last_held + 1] @ function_frame,
                                                  initial @ function_frame, fitting_goals)
                objects = solved["poses"] @ _inverse(function_frame)
                motion_checks = solved["diagnostics"]
            except FunctionReferenceError as error:
                objects, motion_checks = _unfitted_reference(reference[:last_held+1], None, error)
    else:
        try:
            objects, motion_checks = _event_transfer(reference[:last_held + 1], initial,
                # BEGIN the generic frame route shares the recorded-time phase.
                goal_pairs, function_frame, model_spec, timestamps=demo["timestamps"][:last_held+1])
                # END each functional interval normalizes its own original seconds.
        except ReferenceGenerationError as error:
            objects, motion_checks = _unfitted_reference(reference[:last_held+1], bound_model, error)
    # END goals/model are authoritative; source-profile fitting is optional.
    checks = []
    for event in events:
        binding = event["binding"]
        # BEGIN inspect the independent bound target, never use seed fit as qualification.
        pose = initial if event["index"] == 0 else event["goal_pose"]
        # END inspect the independent bound target, never use seed fit as qualification.
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
            error["freedom"] = deepcopy(event["freedom"])
            # BEGIN the declared face owns contact; other body samples need clearance.
            # The original gap band allows a small positive face height. Requiring
            # a different body sample to touch exactly would shrink that allowed set.
            body_world = np.asarray(scene["body_points"], float) @ pose[:3, :3].T+pose[:3, 3]
            minimum_height = float(np.min((body_world-surface.origin) @ surface.normal))
            error["body_checks"]["minimum_signed_height_m"] = minimum_height
            satisfies = (error["valid"] and minimum_height >= -1e-7
                         and error["body_checks"]["minimum_boundary_clearance_m"] >= scene.get("margin", 0.) - 1e-7)
            # END this matches the independent face-contact and body-clearance roles.
            error["checked_member_satisfies_support"] = bool(satisfies)
            if event["index"] == 0 and not satisfies:
                raise ValueError("Transferred function event violates the specified support face or full body bounds")
        else:
            error = relation_error(_inverse(_pose_array(scene["reference_pose"], "Reference pose")) @ pose,
                                   binding["carrier_frame"],
                                   binding["reference_frame"], event["functional_relation"])
            # BEGIN the nominal relation diagnostic and allowed-set result are separate.
            requirement = _material_pose_requirement(event)
            components = requirement.errors_numpy(pose)
            error["allowed_set_errors"] = {name: float(value) for name, value in components.items()}
            if any(components[name] > bound for name, bound in requirement.component_tolerances.items()):
                raise ValueError(f"Transferred trajectory misses functional event {event['event_id']!r}")
            # END full-reference yaw error remains an analysis value when yaw is free.
        if task["relation"] == "inside":
            if "interior" in scene:
                from .pipeline import _interior
                from .goals import check_inside
                inside = check_inside(pose, scene["body_points"], _interior(scene["interior"]))
                satisfies = inside["minimum_clearance_m"] >= scene.get("margin", 0.) - 1e-7
                if event["index"] == 0 and not satisfies:
                    raise ValueError("Function event violates supplied free interior geometry")
                error["containment"] = {"status": "checked_member_inside" if satisfies else "checked_member_outside",
                                        "scope": "one nominal member; the original interior and pose allowed sets remain required", **inside}
            else:
                error["containment"] = {"status": "not_verified", "reason": "free interior geometry not supplied"}
        checks.append({"event_id": event["event_id"], "frame_index": event["frame_index"],
                       "constraint_kind": event["constraint_kind"],
                       "status": "bound_goal",
                       "goal_pose": event["goal_pose"], **error})
        if "reference_generation" in event:
            checks[-1]["reference_generation"] = deepcopy(event["reference_generation"])
    # BEGIN share unchanged corridor semantics with direct joint-coordinate goals.
    # The index addresses source rows, not frame IDs; released body poses still
    # require new observations and are not included in this geometry check.
    _check_entry_corridor(objects, scene, margin, motion_checks)
    # END share unchanged corridor semantics with direct joint-coordinate goals.
    return objects, checks, motion_checks


def ground_object_program(program, scene):
    """Return only function-constrained body motion when no attachment is known.

    Human tool demonstrations may establish object motion before robot grasp
    binding exists. This geometry-only entry preserves their function events
    without fabricating an EEF trajectory. The complete skill consumer requires
    a declared attachment model separately. It uses the same object numerical core.
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
    last = len(frames) - 1 if demo.get("release_frame") is None else _index(
        demo["release_frame"], frames, "release_frame")
    initial = _pose_array(scene["initial_pose"], "Current carrier pose")
    objects, checks, motion_checks = _object_reference(
        demo, scene, frames, last, program["task"], initial)
    result = {"status": "candidate", "output_type": "object_trajectory",
              "carrier_id": scene["carrier_id"], "reference_id": scene["reference_id"],
              "reference_frame": scene["reference_frame"], "poses": objects,
              "frame_indices": frames[:last + 1], "timestamps": times[:last + 1],
              "source_frame_indices": frames, "source_timestamps": times,
              "task": program["task"], "function_checks": checks,
              "motion_checks": motion_checks,
              "pose_convention": "column-vector reference_from_fixed_body; metres; 4x4 SE(3)",
              "path_scope": "function-constrained object reference; no inferred EEF or execution claim"}
    if "body_points" in scene:
        result["body_points"] = scene["body_points"]
    return result


# BEGIN public joint-coordinate input documentation; holding/release logic stays shared.
# BEGIN ground complete source references without an independent approach contract.
def ground_program(program, scene):
    """Generate one grasp/manipulation geometry reference from a compiled demo.

    ``program`` contains ``skill_kind``, ``demo`` and (for manipulation) task
    ``relation``/``motion``. Every functional event names its source frame ID,
    carrier-local material function frame and evidence source. Full-frame
    relations additionally supply source/current reference functional frames,
    a current carrier frame and metric event tolerances. Their explicit
    ``scene.reference_pose`` embeds the current reference owner's local frame
    in ``scene.reference_frame`` (the common output frame).

    # BEGIN explicit current grasp goal, separate from source-reference motion.
    Grasp scenes require ``grasp_goal`` with current carrier-local
    ``T_carrier_eef``, positive task ``position_tolerance_m`` and
    ``rotation_tolerance_rad`` (strictly below pi), plus a nonempty ``source``.
    Region correspondence still transforms the ordinary demonstration;
    changing that reference cannot silently rewrite the independent grasp goal.
    # END no device/numerical tolerance or last reference sample supplies this input.

    Articulation events may instead declare ``constraint_kind=joint_coordinate``.
    Source ``motion_coordinates`` / ``motion_coordinates_source`` preserve the
    full observed signed scalar sequence; event ``coordinate_value`` and
    ``coordinate_tolerance`` describe its source function state. Current
    bindings provide ``current_coordinates`` with source/owner evidence in the
    already bound ``scene.motion_model``. These are absolute unless explicitly
    marked ``coordinate_mode=entry_delta`` with preserved ``entry_coordinates``
    and ``entry_source`` from the original request entrance.
    No reference pose, source body-pose sequence, surrogate region frame or
    source/current body correspondence is needed for this scalar route.

    An ``on`` relation instead supplies finite ``scene.support`` in the output
    frame and, per event, ``carrier_mapping`` (current-carrier-from-source-
    carrier), owner IDs, evidence source and ``on_geometry`` contact samples,
    signed normal and tolerances. It requires no current reference pose or
    complete reference functional frame. Source motion is aligned with the
    current initial carrier pose before the support constrains the event goal;
    its tangential position and yaw remain preferences, not observations.
    Every event for one carrier must use the same explicit carrier mapping.

    # BEGIN nominal attachment is a model, never simulator success evidence.
    Manipulation additionally requires ``scene.holding`` with the declared
    ``T_carrier_eef`` and its model source. ``initial_pose`` and robot
    ``eef_pose`` must use that same model; this equality checks coordinates,
    not whether the physical object is actually held. No verified flag or
    simulator object pose is required. ``release_frame`` is an observed source
    phase boundary, independent of the opening command and function event. The
    object and EEF references stop at this boundary. Complete source times and
    commands remain separate evidence; released-hand motion is not generated.
    Only the current tool identity and calibrated frame are required for held
    manipulation, because the declared attachment supplies its tool relation.
    # END physical holding success belongs to independent result assessment.

    Grasp source frames only initialize the current single-skill plan. Its
    bound terminal grasp relation is a goal; no approach frame is a waypoint.
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
    # END ground complete source references without an independent approach contract.
    if kind != "manipulation":
        raise ValueError("skill_kind must be grasp or manipulation")
    # BEGIN actual holding replaces obsolete source-hand/tool prerequisites.
    for key in ("tool_id", "tool_frame"):
        if not isinstance(scene.get(key), str) or not scene[key]:
            raise ValueError(f"Manipulation requires a nonempty current {key}")
    for key in ("carrier_id", "reference_id"):
        if not isinstance(scene.get(key), str) or not scene[key]:
            raise ValueError(f"Current manipulation needs a nonempty {key}")
    if scene["carrier_id"] == scene["reference_id"]:
        raise ValueError("Carrier and reference must be distinct rigid entities")
    frames, times = validate_timeline(demo["frame_indices"], demo["timestamps"])
    commands = _intentions(demo["gripper_events"], frames)
    release = demo.get("release_frame")
    last_held = len(frames) - 1 if release is None else _index(release, frames, "release_frame")
    if release is None and any(event["command"] == "open" for event in commands):
        raise ValueError("An opening skill needs an explicit release boundary; command time is not that boundary")
    if release is not None and not any(event["command"] == "open" and event["frame_index"] <= release
                                      for event in commands):
        raise ValueError("The released phase needs an explicit opening intention before its boundary")
    if any(event["frame_index"] > frames[last_held]
           and any(event.get(field) for field in ("timing", "requires_evidence", "stop"))
           for event in commands):
        raise ValueError("A gripper requirement lies after release, outside current controlled motion")
    # BEGIN a single declared attachment supplies the held-motion model.
    holding = scene.get("holding", {})
    if not holding.get("source"):
        raise ValueError("Manipulation requires the declared attachment model source")
    attachment = _pose_array(holding["T_carrier_eef"], "Declared T_carrier_eef")
    initial = _pose_array(scene["initial_pose"], "Current carrier pose")
    eef_initial = _pose_array(scene["eef_pose"], "Current EEF pose")
    if not np.allclose(initial @ attachment, eef_initial, atol=1e-6, rtol=0):
        raise ValueError("Carrier and EEF initial coordinates disagree with the declared attachment model")
    # END this is model consistency, not a measured object-to-target gate.
    task = program["task"]
    if task.get("relation") not in {"transfer", "on", "inside", "articulation", "tool_use"}:
        raise ValueError("Unsupported functional relation")
    objects, checks, motion_checks = _object_reference(
        demo, scene, frames, last_held, task, initial)
    eef = attached_eef_reference(objects, attachment)
    # Preserve the original command metadata even when its recording extends
    # beyond this skill's holding interval. A later source command is not an
    # executable event of the current controlled motion.
    controlled_commands = [event for event in commands if event["frame_index"] <= frames[last_held]]
    # END no source-hand dependence or automatic released-hand phase remains.
    return {"status": "candidate", "skill_kind": kind, "output_type": "interaction_trajectory",
            "reference_frame": frame_name, "carrier_id": scene["carrier_id"],
            "reference_id": scene["reference_id"], "task": task,
            "object_poses": objects, "object_frame_indices": frames[:last_held + 1],
            "eef_poses": eef, "frame_indices": frames[:last_held + 1], "timestamps": times[:last_held + 1],
            "source_frame_indices": frames, "source_timestamps": times,
            "source_gripper_events": commands,
            "gripper_events": controlled_commands, "release_frame": release,
            "function_checks": checks, "motion_checks": motion_checks,
            "tool_id": scene["tool_id"], "tool_frame": scene["tool_frame"],
            "holding_source": holding["source"],
            "released_object_state": "requires_observation" if release is not None else "still_attached_reference",
            "pose_convention": "column-vector reference_from_body/eef; metres; 4x4 SE(3)",
            "path_scope": "geometry reference; robot IK, collision, forces and timing remain external"}
# END public joint-coordinate input documentation; holding/release logic stays shared.
# END region-based skill program consumer.
