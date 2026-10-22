"""Fixed-frame demonstration motion and current-scene reference arithmetic.

All transforms map homogeneous column vectors, in metres.  B is a fixed
physical-body frame, F a body-attached functional frame and E the control tool
frame.  These functions neither infer task tolerances nor verify an attachment.
"""

# BEGIN PRD fixed-frame representation and reference generation.
from copy import deepcopy

import numpy as np

from .grasp import validate_timeline
from .motion import MotionModel, _errors, _inverse, _pose_array, _poses_array
from .planning_problem import PosePath


def _timeline(timestamps):
    times = np.asarray(timestamps)
    return validate_timeline(np.arange(len(times)), times)[1]


def _progress(times):
    return np.zeros(1) if len(times) == 1 else (times-times[0])/(times[-1]-times[0])


def _aligned_poses(value, times, name):
    poses = _poses_array(value, name)
    if len(poses) != len(times):
        raise ValueError(f"{name} must align with the original timestamps")
    return poses


def sample_poses(poses, timestamps, query_times):
    """Interpolate one actual pose track at original seconds, without clamping.

    Reuse PosePath's translation/Slerp interpolation.  A one-pose observation
    remains one observation: it cannot be extrapolated to a longer recording.
    """
    times = _timeline(timestamps)
    values = _aligned_poses(poses, times, "poses")
    query = np.asarray(query_times, dtype=float)
    if (not np.isfinite(query).all() or np.any(query < times[0])
            or np.any(query > times[-1])):
        raise ValueError("Pose query times must lie within the observed timeline")
    path = PosePath(_progress(times), values, source="original demonstration timestamps")
    progress = np.zeros_like(query) if len(times) == 1 else (query-times[0])/(times[-1]-times[0])
    return path.sample_numpy(progress)


def trajectory_summary(poses, timestamps, *, approach_time=None, interaction_time=None):
    """Read interaction pose and displacement direction from this same track.

    The returned direction describes the origin of the supplied pose, not the
    surface normal or another attached origin.  Missing events or zero net
    displacement do not discard the complete motion.
    """
    times = _timeline(timestamps)
    values = _aligned_poses(poses, times, "summary poses")
    return _sampled_summary(lambda time: sample_poses(values, times, time),
                            approach_time, interaction_time)


def _sampled_summary(sample, approach_time, interaction_time):
    """Read both summary values from the actual track's sampling operation."""
    result = {"approach_time": approach_time, "interaction_time": interaction_time,
              "missing": {}}
    if interaction_time is None:
        result["missing"]["interaction_pose"] = "No recorded interaction time"
    else:
        result["interaction_pose"] = sample(float(interaction_time))
    if approach_time is None or interaction_time is None:
        result["missing"]["approach_direction"] = "Both approach and interaction times are required"
        return result
    if float(approach_time) > float(interaction_time):
        raise ValueError("Approach time cannot follow interaction time")
    start = sample(float(approach_time))
    delta = result["interaction_pose"][:3, 3] - start[:3, 3]
    length = float(np.linalg.norm(delta))
    if length <= np.finfo(float).eps * max(1., np.linalg.norm(start[:3, 3]),
                                         np.linalg.norm(result["interaction_pose"][:3, 3])):
        result["missing"]["approach_direction"] = "The two relative positions coincide"
    else:
        result["approach_direction"] = delta / length
    return result


def _relative(parent, child):
    """Synchronized parent-from-child poses; no initial-pose normalization."""
    return np.stack([_inverse(p) @ c for p, c in zip(parent, child)])


def build_representation(*, timestamps, frame_indices, kind,
                         T_world_carrier=None, T_world_reference=None,
                         T_world_eef=None, L_G=None, L_R=None,
                         approach_time=None, interaction_time=None,
                         release_time=None, events=(), provenance=None):
    """Compile available synchronized observations into the PRD representation.

    ``carrier`` means the grasped body for grasp and the manipulated body for
    manipulation. ``reference`` is the physical body carrying F_R. L_G and L_R
    map functional coordinates into their respective fixed body coordinates.
    Raw tracks, original frame IDs/times and events remain in the result.  A
    missing source E does not block H_m; no event or pose is manufactured.
    """
    if kind not in {"grasp", "manipulation"}:
        raise ValueError("Representation kind must be grasp or manipulation")
    frames, times = validate_timeline(frame_indices, timestamps)
    result = {"kind": kind, "frame_indices": frames, "timestamps": times,
              "progress": _progress(times), "approach_time": approach_time,
              "interaction_time": interaction_time, "release_time": release_time,
              "events": deepcopy(list(events)), "provenance": deepcopy(provenance or {}),
              "summaries": {}, "missing": {}}
    for name, value in (("T_world_carrier", T_world_carrier),
                        ("T_world_reference", T_world_reference),
                        ("T_world_eef", T_world_eef)):
        if value is not None:
            result[name] = _aligned_poses(value, times, name)
    for name, value in (("L_G", L_G), ("L_R", L_R)):
        if value is not None:
            result[name] = _pose_array(value, name)
    for name, value in (("approach_time", approach_time),
                        ("interaction_time", interaction_time), ("release_time", release_time)):
        if value is not None and (not np.isfinite(value) or not times[0] <= value <= times[-1]):
            raise ValueError(f"{name} must be an original time inside this segment")
    if "T_world_carrier" in result and "T_world_eef" in result:
        result["C_D"] = _relative(result["T_world_carrier"], result["T_world_eef"])
    if kind == "grasp":
        if "C_D" in result and "L_G" in result:
            result["H_g"] = _inverse(result["L_G"]) @ result["C_D"]
        else:
            result["missing"]["H_g"] = "Grasp motion needs source carrier/EEF poses and L_G"
    else:
        if all(key in result for key in ("T_world_carrier", "T_world_reference", "L_R")):
            relative = _relative(result["T_world_reference"], result["T_world_carrier"])
            result["H_m"] = _inverse(result["L_R"]) @ relative
        else:
            result["missing"]["H_m"] = "Object motion needs source carrier/reference poses and L_R"
        if "C_D" not in result:
            result["missing"]["C_D"] = "Source carrier and EEF poses are not both available"
    for name in ("H_g", "H_m", "C_D"):
        if name in result:
            # BEGIN source H_m summaries share the consumer's continuous body motion.
            # H_m ends in an arbitrary body chart. Interpolating its translated
            # origin directly would change the physical path when that chart
            # is redefined. The relative displacement H(t) inv(H(0)) is chart
            # invariant; compose H(0) only after sampling it. H_g and C_D end in
            # the calibrated tool frame, so their existing semantics remain.
            if name == "H_m":
                initial = result[name][0]
                relative = result[name] @ _inverse(initial)
                result["summaries"][name] = _sampled_summary(
                    lambda time: sample_poses(relative, times, time) @ initial,
                    approach_time, interaction_time)
            else:
                result["summaries"][name] = trajectory_summary(
                    result[name], times, approach_time=approach_time, interaction_time=interaction_time)
            # END saved interaction poses and generated nominal goals use one path.
    return result


def _component_path(value, times, path_times, *, source, parent=None, attachment=None):
    """Interpolate one relation before composing its parent/right attachment.

    All components share the same physical-time interval. At release, each
    component is independently restricted to that interval; interpolating the
    already multiplied object/EEF samples would cut rotating lever-arm arcs.
    """
    poses = np.asarray(value)
    if poses.shape == (4, 4):
        values = np.repeat(_pose_array(value, source)[None], len(path_times), axis=0)
    else:
        values = sample_poses(value, times, path_times)
    return PosePath(_progress(path_times), values, source=source, parent=parent,
                    attachment=np.eye(4) if attachment is None else attachment)


# BEGIN nominal composition and geometric boundary declarations have distinct outputs.
def _right_path(parent, transform, source):
    """Sample the physical parent before applying a fixed attached relation."""
    return PosePath(parent.progress, np.repeat(np.eye(4)[None], len(parent.progress), axis=0),
                    source=source, parent=parent, attachment=transform)


def _require_same_pose(first, second, message):
    """Check geometric arithmetic, without inventing a physical task tolerance."""
    position, rotation = _errors(first[None], second[None])
    if max(position[0], rotation[0]) > 1e-8:
        raise ValueError(message)


def generate_reference(representation, *, T_world_function=None, J=None, A=None,
                       T_world_eef=None, T_world_body=None, motion_model=None,
                       source_coordinates=None, current_coordinates=None,
                       contact_motion=False, held_until_time=None):
    """Bind complete nominal source motion and declare entry/event boundaries.

    J maps current body coordinates B_N into demonstrated coordinates B_D;
    # BEGIN A is the declared rigid attachment model, not physical truth.
    A maps the modeled body B_N into tool E. Robot feedback E plus A determines
    the modeled body pose. If both poses are supplied, they must use this same
    model; the equality does not verify physical holding or target arrival.
    # END no simulator body measurement is required to bind held motion.
    Free object motion is bound in a fixed material frame, independently of
    the body's coordinate chart and the grasp. The original motion supplies
    the nominal interaction goal. Returned paths retain this complete nominal
    composition; their separately named boundaries enter BoundToolReference
    only after the planner has bound physical time and measured TCP velocity.

    Grasp motion requires actual E. A scalar mechanism instead uses its current
    model and current coordinate, preserving its physical manifold and winding.
    Production object/tool paths stop at release or an earlier held_until_time.
    Full source evidence and unadapted bound diagnostics remain separately named.
    Missing entry/interaction evidence cannot produce a boundary declaration.
    Reference geometry and its boundary data remain separate from task/scene
    qualification and the eventual planned joint trajectory.
    """
    frames, times = validate_timeline(representation["frame_indices"], representation["timestamps"])
    result = dict(frame_indices=frames, timestamps=times,
        events=deepcopy(representation.get("events", [])), missing={}, summaries={},
        boundary_checks={}, status="missing",
        scope="grounded nominal paths and explicit physical-time boundary declarations")
    kind = representation["kind"]
    if kind not in {"grasp", "manipulation"}:
        raise ValueError("Unknown representation kind")
    if A is not None and contact_motion:
        raise ValueError("Choose a rigid attachment model or time-varying contact, not both")
    extra = [representation.get(key) for key in ("approach_time", "interaction_time", "release_time")]
    extra += [held_until_time]
    extra = [float(value) for value in extra if value is not None]
    if any(not np.isfinite(value) or not times[0] <= value <= times[-1] for value in extra):
        raise ValueError("Reference event times must lie within the original timeline")
    full_times = np.unique(np.r_[times, extra])
    ends = [value for value in (held_until_time, representation.get("release_time")) if value is not None]
    end = min(ends) if ends else times[-1]
    path_times = full_times[full_times <= end]
    interaction = representation.get("interaction_time")
    for name, value in (("J", J), ("A", A), ("T_world_eef", T_world_eef), ("T_world_body", T_world_body)):
        if value is not None:
            result[name] = _pose_array(value, name)
    for name in ("H_g", "H_m", "C_D"):
        if name in representation:
            result[name] = _aligned_poses(representation[name], times, name)
    if T_world_function is not None:
        result["T_world_function"] = (_pose_array(T_world_function, "T_world_function")
            if np.asarray(T_world_function).shape == (4, 4)
            else _aligned_poses(T_world_function, times, "T_world_function"))
    if interaction is None:
        result["missing"]["interaction_time"] = "A recorded interaction time is required to bind the nominal goal"
        return result
    if interaction > end:
        result["missing"]["interaction_time"] = "Interaction lies after the controlled interval; no held reference can constrain it"
        return result
    event_progress = 0. if len(path_times) == 1 else (interaction-path_times[0])/(path_times[-1]-path_times[0])
    current_body, current_eef = result.get("T_world_body"), result.get("T_world_eef")
    if A is not None:
        if current_body is None and current_eef is not None:
            current_body = current_eef @ result["A"]
        elif current_eef is None and current_body is not None:
            current_eef = current_body @ _inverse(result["A"])
        if current_body is not None and current_eef is not None:
            _require_same_pose(current_body, current_eef @ result["A"],
                               "Current body, EEF and attachment A are inconsistent")

    def function_path(clock):
        return _component_path(result["T_world_function"], times, clock, source="current functional frame F")

    def diagnostic(name, path, clock):
        result["bound_" + name + "_timestamps"] = clock.copy()
        result["bound_" + name + "_poses"] = path.sample_numpy(_progress(clock))

    def publish(name, path, current, nominal_goal):
        # BEGIN geometric production preserves the nominal path until clock binding.
        source = {"generator": "generate_reference", "role": name,
                  "interaction_time": float(interaction), "reference_clock": path_times.tolist()}
        result[name + "_boundaries"] = {"initial_pose": np.asarray(current).copy(),
            "goals": [{"progress": float(event_progress), "pose": np.asarray(nominal_goal).copy(),
                       "source": source}], "source": source}
        result["boundary_checks"][name] = {"status": "declared_for_physical_time_binding",
            "initial_pose": np.asarray(current).copy(), "nominal_interaction": np.asarray(nominal_goal).copy(),
            "interaction_progress": float(event_progress)}
        # END no progress derivative is represented as measured physical velocity.
        result[name + "_path"] = path
        result[name + "_timestamps"] = path_times.copy()
        result[name + "_poses"] = path.sample_numpy(_progress(path_times))
        approach = representation.get("approach_time")
        if approach is not None and not path_times[0] <= approach <= path_times[-1]:
            approach = None
        def sample(time):
            progress = 0. if len(path_times) == 1 else (time-path_times[0])/(path_times[-1]-path_times[0])
            return path.sample_numpy(progress)
        result["summaries"][name] = _sampled_summary(sample, approach, interaction)

    if kind == "grasp":
        if "H_g" not in result or T_world_function is None:
            result["missing"]["eef_poses"] = "Grasp reference needs H_g and current F_G"
        elif current_eef is None:
            result["missing"]["eef_poses"] = "Grasp reference needs the actual current EEF pose"
        else:
            raw = _component_path(result["H_g"], times, path_times,
                                  source="bound source grasp relation", parent=function_path(path_times))
            diagnostic("eef", raw, path_times)
            publish("eef", raw, current_eef, raw.sample_numpy(event_progress))
            result["method"] = "nominal grasp relation with declared tool boundaries"
    else:
        raw_body = None
        if motion_model is not None:
            if not isinstance(motion_model, MotionModel) or motion_model.kind not in {"prismatic", "revolute"}:
                raise ValueError("Scalar mechanism reference needs a current prismatic/revolute MotionModel")
            if source_coordinates is None or current_coordinates is None:
                result["missing"]["object_poses"] = "Mechanism reference needs source scalar coordinates and current coordinate"
            else:
                coordinates = np.asarray(source_coordinates, dtype=float)
                if coordinates.shape == (len(times),):
                    coordinates = coordinates[:, None]
                current = np.asarray(current_coordinates, dtype=float).reshape(-1)
                if (coordinates.shape != (len(times), 1) or current.shape != (1,)
                        or not np.isfinite(coordinates).all() or not np.isfinite(current).all()):
                    raise ValueError("Scalar coordinates must be finite and align with the source timeline")
                result["source_coordinates"], result["current_coordinates"] = coordinates.copy(), current.copy()
                def mechanism_path(clock):
                    values = current + np.interp(clock, times, coordinates[:, 0])[:, None] - coordinates[0]
                    return PosePath(_progress(clock), motion_model.poses(values),
                        source="current mechanism FK from source scalar displacement",
                        motion_model=motion_model, coordinates=values)
                model_entry = motion_model.poses(current[None])[0]
                if current_body is not None:
                    _require_same_pose(current_body, model_entry, "Current pose conflicts with mechanism coordinates")
                current_body = model_entry
                if A is not None:
                    model_eef = current_body @ _inverse(result["A"])
                    if current_eef is not None:
                        _require_same_pose(current_eef, model_eef, "Current EEF conflicts with mechanism and attachment")
                    current_eef = model_eef
                raw_body = mechanism_path(path_times)
                result["coordinates"] = raw_body.coordinates.copy()
                diagnostic("object", mechanism_path(full_times), full_times)
                result["method"] = "current mechanism coordinates and FK"
        elif "H_m" not in result or T_world_function is None or J is None:
            result["missing"]["object_poses"] = "Object reference needs H_m, current F_R and body correspondence J"
        elif current_body is None:
            result["missing"]["object_poses"] = "Object reference needs actual body pose or actual EEF with attachment A"
        else:
            # H(t) right-composition depends on the arbitrarily chosen body
            # origin. Interpolate Q=H inv(H0) instead; Q survives source-chart
            # changes exactly. The material frame coincides with F at the
            # nominal event and is fixed to the current body through L_body.
            # Sampling Q BEFORE either right composition retains rotating arms.
            H0 = result["H_m"][0]
            Q = result["H_m"] @ _inverse(H0)
            Qc = sample_poses(Q, times, interaction)
            L_body = _inverse(result["J"]) @ _inverse(H0) @ _inverse(Qc)
            result["L_body_material"] = L_body
            def material_path(clock):
                return _component_path(Q, times, clock, source="source chart-invariant relative displacement",
                    parent=function_path(clock), attachment=_inverse(Qc))
            raw_material = material_path(path_times)
            raw_body = _right_path(raw_material, _inverse(L_body), "bound source body diagnostic")
            diagnostic("object", _right_path(material_path(full_times), _inverse(L_body),
                                              "full source binding diagnostic"), full_times)
            result["method"] = "nominal material-frame motion with declared body/tool boundaries"
        if raw_body is not None:
            publish("object", raw_body, current_body, raw_body.sample_numpy(event_progress))
            if A is not None:
                inverse_A = _inverse(result["A"])
                raw_eef = _right_path(raw_body, inverse_A, "bound held tool diagnostic")
                diagnostic("eef", raw_eef, path_times)
                publish("eef", raw_eef,
                        current_eef, raw_eef.sample_numpy(event_progress))
            elif contact_motion and "C_D" in result and J is not None and current_eef is not None:
                # Contact is an explicitly supplied time-varying source
                # relation, not a newly inferred attachment or pushing model.
                relation = _inverse(result["J"]) @ result["C_D"]
                raw_eef = _component_path(relation, times, path_times,
                    source="bound nonattached contact diagnostic", parent=raw_body)
                _require_same_pose(current_body @ relation[0], current_eef,
                                   "Actual EEF conflicts with the supplied source contact relation")
                diagnostic("eef", raw_eef, path_times)
                publish("eef", raw_eef, current_eef, raw_eef.sample_numpy(event_progress))
            else:
                result["missing"]["eef_poses"] = ("Nonattached contact needs C_D, J and actual EEF"
                    if contact_motion else "EEF reference needs a declared attachment model A")
    if "eef_poses" in result or "object_poses" in result:
        result["status"] = "partial" if result["missing"] else "reference"
    return result
# END complete nominal paths and observed/modelled entrance data retain their roles.
# END PRD fixed-frame representation and reference generation.
