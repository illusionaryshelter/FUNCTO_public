"""Predicted body correspondence and geometry-anchored manipulation references.

D is the source skill's body chart, C the unchanged observed TSDF chart, E
the calibrated robot tool, F the grounded task region, and W the reference
world. T_X_Y maps Y coordinates to X, with translations in metres. Source
representation supplies complete motion and static task relations; current
RGB-D supplies occupied geometry and the observed support patch.
"""

# BEGIN reference transfer consumes explicit poses, task relations and RGB-D geometry.
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from .motion import _inverse, _pose_array
from .observed_support import estimate_occupied_volume_center, place_observed_on
from .representation import (
    _initial_rotation_offset, _right_path, _sampled_summary, generate_reference, sample_poses,
)


def predicted_body_correspondence(*, T_query_source_camera, T_world_query_camera,
                                  T_source_world_camera, T_source_world_body,
                                  T_world_field):
    """Express a predicted source body in the retained current field chart C.

    The prediction maps source optical camera Cr to query optical camera Cq.
    Its float32 rotation is represented on SO(3) with SciPy's existing matrix
    conversion. Source image body pose belongs to the saved skill; the two
    camera calibrations and the retained field chart belong to observations.
    No grasp-region coordinate frame enters this whole-body correspondence.
    """
    raw = np.asarray(T_query_source_camera, dtype=float)
    predicted = raw.copy()
    predicted[:3, :3] = Rotation.from_matrix(raw[:3, :3]).as_matrix()
    predicted = _pose_array(predicted, "predicted query-from-source camera pose")
    return (_inverse(_pose_array(T_world_field, "retained field chart"))
            @ _pose_array(T_world_query_camera, "query camera calibration")
            @ predicted
            @ _inverse(_pose_array(T_source_world_camera, "source camera calibration"))
            @ _pose_array(T_source_world_body, "source image body pose"))


def support_region_frame(functional_frame, surface):
    """Use the observed support normal and the grounded region's tangent axis.

    The region origin supplies the desired centre's planar location. Project
    its first axis onto the observed plane, retaining the grounded tangent
    direction. The support and task frame therefore share one normal while
    preserving the region's independently estimated position.
    """
    frame = _pose_array(functional_frame, "grounded functional region").copy()
    normal = np.asarray(surface.normal)
    normal = normal / np.linalg.norm(normal)
    first = frame[:3, 0] - normal * (normal @ frame[:3, 0])
    first = first / np.linalg.norm(first)
    frame[:3, :3] = np.column_stack((first, np.cross(normal, first), normal))
    return frame


def observed_manipulation_targets(representation, *, functional_frame, J, A,
                                  T_world_eef, field, surface, margin, precision_m):
    """Determine interaction/release targets before generating full motion.

    H_m(t*) is the source static body-to-functional task relation. Its initial
    and interaction poses supply the existing initial-relative RPY selection.
    At each declared event R_W_C = R_W_F D R_H_m(event) R_J. Centre XY comes
    from the grounded region; the current occupied set at that rotation fixes
    the support height and finite-patch fit. The result contains body-chart C
    targets and the translated centre chart G=[I,c_C].
    """
    frame = support_region_frame(functional_frame, surface)
    J = _pose_array(J, "source-from-current body correspondence")
    A = _pose_array(A, "declared tool-from-field attachment")
    E = _pose_array(T_world_eef, "current tool pose")
    initial = np.asarray(representation["H_m"])[0]
    interaction = _pose_array(representation["summaries"]["H_m"]["interaction_pose"],
                              "source static interaction relation")
    relative = np.asarray(representation["H_m"]) @ _inverse(initial)
    release_time = float(representation["release_time"])
    release = sample_poses(relative, representation["timestamps"], release_time) @ initial
    delta, rotation_source = _initial_rotation_offset(
        frame @ initial @ _inverse(interaction), frame,
        E @ A @ _inverse(J) @ _inverse(interaction),
        np.asarray(representation["is_initial_relative_rotation_rpy"], dtype=bool))
    center = estimate_occupied_volume_center(field)
    K = np.eye(4)
    K[:3, 3] = center["center_local"]
    start = float(representation["timestamps"][0])
    targets = []
    for timestamp, relation in ((float(representation["interaction_time"]), interaction),
                                (release_time, release)):
        preferred = np.eye(4)
        preferred[:3, :3] = frame[:3, :3] @ delta @ relation[:3, :3] @ J[:3, :3]
        preferred[:3, 3] = frame[:3, 3]
        body, placement = place_observed_on(preferred, field, surface,
            center_local=center["center_local"], margin=margin, precision_m=precision_m)
        targets.append(dict(timestamp_s=timestamp, progress=(timestamp-start)/(release_time-start),
            body_pose=body, center_pose=body @ K, source_static_relation=relation,
            observed_placement=placement))
    return dict(functional_frame=frame, center=center, entity_from_center=K,
                rotation_transfer=rotation_source, targets=targets)


def generate_observed_reference(representation, *, functional_frame, J, A,
                                T_world_eef, field, surface, margin, precision_m):
    """Construct geometric targets, transfer source motion, and derive TCP.

    Complete source H_m/C_D, events and release data enter generate_reference
    after targets have been determined. The nominal body path is right-anchored
    at the geometric release pose; its independently computed event targets
    become the ordinary TCP boundary declarations. BoundToolReference remains
    responsible for the measured entrance pose/twist and the selected clock.
    """
    geometry = observed_manipulation_targets(representation,
        functional_frame=functional_frame, J=J, A=A, T_world_eef=T_world_eef,
        field=field, surface=surface, margin=margin, precision_m=precision_m)
    reference = generate_reference(representation, T_world_function=geometry["functional_frame"],
        J=J, A=A, T_world_eef=T_world_eef)
    original = reference["object_path"]
    release = geometry["targets"][-1]["body_pose"]
    right = _inverse(original.sample_numpy(1.)) @ release
    source = dict(operation="observed occupied-set placement", geometry=geometry)
    body = _right_path(original, right, source)
    paths = dict(object=body, eef=_right_path(body, _inverse(reference["A"]),
        dict(operation="TCP from the same declared attachment", source=source)))
    reference["geometric_targets"] = geometry
    reference["geometric_anchor"] = dict(progress=1., target_pose=release,
        right_transform=right, original_object_path=original)
    for role, path in paths.items():
        reference[role+"_path"] = path
        reference[role+"_poses"] = path.sample_numpy(path.progress)
        boundaries = deepcopy(reference[role+"_boundaries"])
        boundaries["goals"] = [dict(progress=row["progress"],
            pose=row["body_pose"] if role == "object" else row["body_pose"] @ _inverse(A),
            source=dict(reference=boundaries["source"], geometric_target=row))
            for row in geometry["targets"]]
        reference[role+"_boundaries"] = boundaries
        clock = reference[role+"_timestamps"]
        summary = reference["summaries"][role]
        def sample(time):
            return path.sample_numpy((time-clock[0])/(clock[-1]-clock[0]))
        reference["summaries"][role] = _sampled_summary(
            sample, summary["approach_time"], summary["interaction_time"])
        reference["boundary_checks"][role]["nominal_interaction"] = path.sample_numpy(
            reference["boundary_checks"][role]["interaction_progress"])
    return reference
# END the reference component outputs the planner's existing tool-path interface.
