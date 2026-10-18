"""Fixed-frame demonstration motion and current-scene reference arithmetic.

All transforms map homogeneous column vectors, in metres.  B is a fixed
physical-body frame, F a body-attached functional frame and E the control tool
frame.  These functions neither infer task tolerances nor verify an attachment.
"""

# BEGIN PRD fixed-frame representation and reference generation.
from copy import deepcopy

import numpy as np

from .grasp import validate_timeline
from .motion import MotionModel, _inverse, _pose_array, _poses_array
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
            result["summaries"][name] = trajectory_summary(
                result[name], times, approach_time=approach_time, interaction_time=interaction_time)
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


def generate_reference(representation, *, T_world_function=None, J=None, A=None,
                       motion_model=None, source_coordinates=None,
                       current_coordinates=None, contact_motion=False,
                       held_until_time=None):
    """Generate geometric references from explicit current relations only.

    J maps current body B_N into demonstrated B_D; A maps B_N into current E.
    For stable holding, E_ref=B_ref inv(A). The returned object_path/eef_path
    retain independent interpolation of F, H and the right-hand transforms;
    reported pose arrays are samples of these exact paths. Planning must use
    these paths rather than interpolate their already multiplied samples.
    ``contact_motion=True`` instead transports the time-varying source C_D for
    nonattached contact.  This is a reference, not a prediction of pushing.
    A known scalar mechanism uses current FK on q_now+q_D-q_D[0], with no cabinet
    registration or J requirement.  Source/target scalar units and positive
    direction must already agree.  No task goals or post-release retreat are
    inferred.  Missing independent inputs leave the computable result intact.
    """
    frames, times = validate_timeline(representation["frame_indices"], representation["timestamps"])
    result = {"frame_indices": frames, "timestamps": times, "events": deepcopy(representation.get("events", [])),
              "missing": {}, "summaries": {}, "status": "missing",
              "scope": "geometric reference; not planned or executed"}
    kind = representation["kind"]
    if kind not in {"grasp", "manipulation"}:
        raise ValueError("Unknown representation kind")
    if A is not None and contact_motion:
        raise ValueError("Choose actual rigid holding or time-varying contact, not both")
    # BEGIN preserve each composed relation between all reported samples.
    # Event times describe actual observed events, not invented source frames.
    # Keeping them as path knots also lets the planner retain exact event times.
    extra_times = [representation.get(key) for key in ("approach_time", "interaction_time", "release_time")]
    extra_times += [held_until_time]
    extra_times = [float(value) for value in extra_times if value is not None]
    if any(not np.isfinite(value) or not times[0] <= value <= times[-1] for value in extra_times):
        raise ValueError("Reference event times must lie within the original timeline")
    reference_times = np.unique(np.r_[times, extra_times])
    # Numeric relations remain separate evidence for saved reference artifacts.
    # Runtime PosePath objects are consumed directly by the planner adapter.
    for name, value in (("J", J), ("A", A)):
        if value is not None:
            result[name] = _pose_array(value, name)
    for name in ("H_g", "H_m", "C_D"):
        if name in representation:
            result[name] = _aligned_poses(representation[name], times, name)
    if T_world_function is not None:
        result["T_world_function"] = (_pose_array(T_world_function, "T_world_function")
            if np.asarray(T_world_function).shape == (4, 4)
            else _aligned_poses(T_world_function, times, "T_world_function"))

    def function_path(path_times):
        return _component_path(result["T_world_function"], times, path_times,
                               source="current functional frame F")

    if kind == "grasp":
        if "H_g" not in representation or T_world_function is None:
            result["missing"]["eef_poses"] = "Grasp reference needs H_g and current F_G"
        else:
            result["eef_path"] = _component_path(result["H_g"], times, reference_times,
                source="grasp relation H_g", parent=function_path(reference_times))
            result["eef_timestamps"] = reference_times.copy()
    else:
        object_path = None
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

                def object_path(path_times, attachment=None):
                    values = current + np.interp(path_times, times, coordinates[:, 0])[:, None] - coordinates[0]
                    return PosePath(_progress(path_times), motion_model.poses(values),
                        source="current mechanism FK from demonstrated scalar displacement",
                        motion_model=motion_model, coordinates=values,
                        attachment=np.eye(4) if attachment is None else attachment)

                result["coordinates"] = object_path(reference_times).coordinates.copy()
        elif "H_m" not in representation or T_world_function is None or J is None:
            result["missing"]["object_poses"] = "Object reference needs H_m, current F_R and source/current body correspondence J"
        else:
            def object_path(path_times, attachment=None):
                right = result["J"] if attachment is None else result["J"] @ attachment
                return _component_path(result["H_m"], times, path_times,
                    source="manipulation relation H_m with body correspondence J",
                    parent=function_path(path_times), attachment=right)

        if object_path is not None:
            result["object_path"] = object_path(reference_times)
            result["object_timestamps"] = reference_times.copy()
            if A is not None:
                end = held_until_time if held_until_time is not None else representation.get("release_time")
                held_times = reference_times if end is None else reference_times[reference_times <= float(end)]
                result["eef_path"] = object_path(held_times, attachment=_inverse(result["A"]))
                result["eef_timestamps"] = held_times.copy()
            elif contact_motion and "C_D" in representation and J is not None:
                # inv(J) is a constant left transform, so its product with
                # interpolated C_D equals interpolation of inv(J) @ C_D. The
                # parent object's right J therefore cancels for frame motion;
                # scalar mechanism motion retains its actual current FK.
                result["eef_path"] = _component_path(_inverse(result["J"]) @ result["C_D"],
                    times, reference_times, source="nonattached source contact relation inv(J) C_D",
                    parent=result["object_path"])
                result["eef_timestamps"] = reference_times.copy()
            else:
                result["missing"]["eef_poses"] = ("Nonattached contact needs source C_D and J" if contact_motion
                                                   else "EEF reference needs a confirmed current grasp relation A")
    for name in ("object", "eef"):
        if name + "_path" not in result:
            continue
        own_times = result[name + "_timestamps"]
        path = result[name + "_path"]
        result[name + "_poses"] = path.sample_numpy(_progress(own_times))
        approach, interaction = representation.get("approach_time"), representation.get("interaction_time")
        # An object interaction can follow release. It is not an attached EEF
        # interaction; retain its missing summary instead of extrapolating A.
        if interaction is not None and not own_times[0] <= interaction <= own_times[-1]:
            interaction = None
        if approach is not None and not own_times[0] <= approach <= own_times[-1]:
            approach = None
        def sample(time):
            progress = 0. if len(own_times) == 1 else (time-own_times[0])/(own_times[-1]-own_times[0])
            return path.sample_numpy(progress)

        result["summaries"][name] = _sampled_summary(sample, approach, interaction)
    if "eef_poses" in result or "object_poses" in result:
        result["status"] = "partial" if result["missing"] else "reference"
    # END arrays, summaries and planner paths use one composed reference.
    return result
# END PRD fixed-frame representation and reference generation.
