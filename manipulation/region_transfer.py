"""Bind a portable single-source skill to the existing dual-view computation.

Semantic selection happens before this module. It accepts observed arrays,
never an experiment manifest, a task-specific hint or a target region label.
Region acceptance remains the numerical pipeline's decision; the selected
candidate below only supplies a frame for reference-motion initialization.
"""

# BEGIN portable observed-region input and same-source frame propagation.
import numpy as np

from .motion import _inverse, _pose_array
from .regions import function_frame
from transfer.experiment import feature_observation
from transfer.joint_features import source_queries
from transfer.multiview_experiment import _check_observation
from transfer.segmentation import _calibration
from transfer.sim3 import _points, apply_sim3, rank


class RegionInputMissing(ValueError):
    """A declared input is absent; this is distinct from malformed evidence."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _time(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isfinite(value)):
        raise ValueError(f"{name} must be a finite acquisition time")
    return float(value)


def _feature(value, name):
    feature = np.asarray(value)
    if (feature.dtype != np.float32 or feature.ndim != 4 or feature.shape[0] != 1
            or min(feature.shape[1:]) < 1 or not np.isfinite(feature).all()):
        raise ValueError(f"{name} must be one finite FP32 [1,C,Hf,Wf] map")
    return feature


def _observation(observation, depth, K, camera, crop, name):
    """Check pixel/point/camera identity without changing any point ordering."""
    if not isinstance(observation, dict):
        raise ValueError(f"{name} must be an observed point bank")
    depth = np.asarray(depth)
    if depth.ndim != 2 or depth.dtype.kind not in "fiu" or min(depth.shape) < 1:
        raise ValueError(f"{name} needs a complete optical-depth image")
    K, camera = _calibration(K, camera)
    _check_observation(observation, depth, {"K": K, "T_world_camera_cv": camera})
    count = len(observation["xyz"])
    mask = np.asarray(observation["object_mask"])
    if mask.dtype != np.bool_ or mask.shape != depth.shape:
        raise ValueError(f"{name}.object_mask must use the original image shape")
    for field in ("entity_member", "region_mask", "evidence_only"):
        values = np.asarray(observation[field])
        if values.dtype != np.bool_ or values.shape != (count,):
            raise ValueError(f"{name}.{field} must be Boolean and point aligned")
    pixels = observation["pixels"]
    if not np.array_equal(observation["entity_member"], mask[pixels[:, 1], pixels[:, 0]]):
        raise ValueError(f"{name} entity labels disagree with the selected mask")
    if not np.array_equal(observation["evidence_only"],
                          observation["region_mask"] & ~observation["entity_member"]):
        raise ValueError(f"{name} source evidence labels disagree")
    bounds = np.asarray(crop)
    if bounds.shape != (4,) or bounds.dtype.kind not in "iu":
        raise ValueError(f"{name} crop must contain four integer image bounds")
    feature_observation(observation, bounds)
    return depth, K, camera


def prepare_inputs(source, views, *, same_state_source):
    """Build resident inputs from one schema-2 source and two selected views.

    Each view declares ``id``, ``entity_id``, original ``timestamp_s``, complete
    ``depth_m/K/T_world_camera_cv``, and its ``observation/feature/crop_xyxy``.
    Missing entity/feature evidence is represented by observation=None and
    explicit ``missing_reasons``; complete depth still enters the computation.
    ``same_state_source`` records why both views describe the same geometry,
    such as simultaneous acquisition or a frozen scene without a physics step.
    It is supplied evidence, not inferred from camera names or time proximity.

    ``source.owner_id`` names the source region's rigid body. View entity IDs
    name the selected semantic entity and do not identify a current rigid part.
    All large arrays remain shared and keep their original point identities.
    """
    state_source = _text(same_state_source, "same_state_source")
    if not isinstance(source, dict):
        raise ValueError("source must be the loaded skill's single source record")
    # BEGIN select geometry from the declared source region, never skill names.
    geometry = source.get('region_geometry')
    if geometry not in ('planar', 'surface'):
        raise ValueError('source.region_geometry must explicitly be planar or surface')
    # END input declaration; original observations and F remain unchanged.
    missing = [f"missing_source_{name}" for name in (
        "observation", "depth_m", "feature", "crop_xyxy", "L_body_region",
        "T_world_body_at_image") if source.get(name) is None]
    if missing:
        raise RegionInputMissing(missing)
    _text(source.get("owner_id"), "source.owner_id")
    _time(source.get("timestamp_s"), "source.timestamp_s")
    observation = source["observation"]
    depth, _, _ = _observation(observation, source["depth_m"], observation["K"],
        observation["T_world_camera_cv"], source["crop_xyxy"], "source")
    if not np.any(observation["region_mask"]):
        raise RegionInputMissing(["missing_source_region_points"])
    feature = _feature(source["feature"], "source.feature")
    if not isinstance(views, (list, tuple)) or len(views) != 2:
        raise ValueError("Exactly two current view records are required")
    prepared, feature_pairs, identities, ids = [], {}, set(), set()
    for view in views:
        if not isinstance(view, dict):
            raise ValueError("Each current view must be a record")
        name = _text(view.get("id"), "view.id")
        if name in ids:
            raise ValueError("Current view IDs must be distinct")
        ids.add(name)
        identities.add(_text(view.get("entity_id"), "view.entity_id"))
        _time(view.get("timestamp_s"), "view.timestamp_s")
        target_depth = np.asarray(view["depth_m"])
        if target_depth.ndim != 2 or target_depth.dtype.kind not in "fiu" or min(target_depth.shape) < 1:
            raise ValueError("A missing entity observation still requires full depth")
        K, camera = _calibration(view["K"], view["T_world_camera_cv"])
        record = dict(view, depth_m=target_depth, K=K, T_world_camera_cv=camera)
        target = view.get("observation")
        reasons = view.get("missing_reasons", [])
        if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("View missing_reasons must be a list of nonempty strings")
        if target is None:
            if not reasons:
                raise ValueError("A missing entity observation needs its actual reason")
            record.update(observation=None, crop_xyxy=None, missing_reasons=list(reasons))
        else:
            _observation(target, target_depth, K, camera, view["crop_xyxy"], name)
            if not np.all(target["entity_member"]):
                raise ValueError("Current point banks may only contain the selected entity")
            if view.get("feature") is None:
                # The view is explicitly unavailable for appearance, while its
                # complete depth remains in the same calibrated surface input.
                record.update(observation=None, crop_xyxy=None,
                              missing_reasons=list(reasons) + ["missing_target_feature"])
            else:
                current_feature = _feature(view["feature"], f"{name}.feature")
                if current_feature.shape != feature.shape:
                    raise ValueError("Source and current maps must share one feature profile")
                feature_pairs[name] = (feature, current_feature)
                record["missing_reasons"] = list(reasons)
        prepared.append(record)
    if len(identities) != 1:
        raise ValueError("Both views must refer to the same already selected entity")
    if not feature_pairs:
        raise RegionInputMissing(["no_current_entity_feature_observation"])
    query_indices, query_region = source_queries(observation)
    # BEGIN carry the geometry declaration through the numerical boundary.
    loaded = {"region_geometry": geometry, "source": observation, "source_depth": depth,
              "source_crop": source["crop_xyxy"], "source_rgb": source.get("image"),
              "query_indices": query_indices, "query_region": query_region,
              "views": prepared, "same_state_source": state_source,
              "source_timestamp_s": source["timestamp_s"],
              "target_entity_id": next(iter(identities))}
    # END no source-row or target-observation substitution.
    return loaded, feature_pairs


def select_reference_transform(result):
    """Choose the best existing hypothesis supporting the accepted primary.

    A tied common region does not certify a unique rotation. Its supported
    highest-score hypothesis with the smallest original ID initializes the
    reference without changing the accepted region or any numerical decision.
    """
    if not result.get("accepted", False):
        return None
    primary = [mode for mode in result["modes"] if mode["type"] == "primary"]
    if len(primary) != 1:
        raise ValueError("An accepted result must contain one primary region")
    supported = primary[0]["hypothesis_indices"]
    hypotheses = {item["hypothesis_id"]: item for item in result["diagnostics"]["hypotheses"]}
    if not supported or any(index not in hypotheses for index in supported):
        raise ValueError("The primary region lost its supporting hypothesis identities")
    candidates = [hypotheses[index] for index in supported]
    if any(not item["admitted_to_H"] or not np.isfinite(item["score"]) for item in candidates):
        raise ValueError("A primary reference cannot use an unadmitted hypothesis")
    chosen = min(candidates, key=lambda item: (-item["score"], item["hypothesis_id"]))
    scale, rotation, translation = chosen["transform"]
    rigid = np.eye(4)
    rigid[:3, :3], rigid[:3, 3] = rotation, translation
    rigid = _pose_array(rigid, "Chosen candidate rotation/translation")
    if isinstance(scale, (bool, np.bool_)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("The chosen candidate must have a positive finite scale")
    return {"hypothesis_id": chosen["hypothesis_id"], "score": float(chosen["score"]),
            "scale": float(scale), "rotation": rigid[:3, :3], "translation": rigid[:3, 3],
            "region_hypothesis_ids": list(supported),
            "finite_candidate_unique": bool(result["finite_candidate_unique"]),
            "paired_query_rows": np.asarray(chosen["paired_query_rows"], dtype=np.int64),
            "paired_surface_indices": np.asarray(chosen["paired_surface_indices"], dtype=np.int64),
            "selection": "highest_existing_score_then_smallest_hypothesis_id"}


def propagate_region_frame(source, transform, *, current_region_owner=None, same_state_source=None):
    """Map the source image-time frame; scale its origin, rotate its unit axes.

    ``current_region_owner`` optionally supplies owner_id, T_world_body,
    timestamp_s and source for the physical body carrying this region. It may
    be an active drawer inside the selected cabinet entity. Without that body
    pose, the current world frame is still available; no body-local frame or
    persistent moving-region attachment is manufactured. Binding the local
    frame additionally requires ``same_state_source`` to state why the owner
    pose and region observations describe this same geometry. Their original
    timestamps remain distinct when the declared evidence is a static scene.
    """
    source_body = _pose_array(source["T_world_body_at_image"], "Source body at region image time")
    local = _pose_array(source["L_body_region"], "Source body-local region frame")
    frame = source_body @ local
    rotation = np.asarray(transform["rotation"])
    current = np.eye(4)
    current[:3, :3] = rotation @ frame[:3, :3]
    current[:3, 3] = apply_sim3(frame[None, :3, 3], transform["scale"], rotation,
                                 transform["translation"])[0]
    current = _pose_array(current, "Current world region frame")
    return _region_frame_record(source, frame, current, transform["hypothesis_id"],
        "selected-candidate reference frame; not a unique-pose certificate",
        current_region_owner, same_state_source)


# BEGIN manipulation interaction regions use the accepted observed center.
def _region_frame_record(source, frame, current, hypothesis_id, scope,
                         current_region_owner, same_state_source):
    """Store world/source frames and bind an explicitly supplied current owner.

    Both frame constructors use this same SE(3) bookkeeping. The owner pose
    and its acquisition provenance have the same contract as before.
    """
    result = {"T_world_region": current, "T_source_world_region": frame,
              "source_owner_id": source["owner_id"],
              "source_timestamp_s": source["timestamp_s"],
              "hypothesis_id": hypothesis_id,
              "L_body_region": None, "current_owner_id": None,
              "scope": scope}
    if current_region_owner is not None:
        state_source = _text(same_state_source, "Region/owner same-state evidence")
        owner_id = _text(current_region_owner.get("owner_id"), "Current region owner ID")
        _text(current_region_owner.get("source"), "Current region owner pose source")
        _time(current_region_owner.get("timestamp_s"), "Current region owner pose time")
        pose = _pose_array(current_region_owner["T_world_body"], "Current region owner pose")
        result.update(L_body_region=_inverse(pose) @ current, current_owner_id=owner_id,
                      current_owner_pose_source=dict(current_region_owner),
                      current_owner_same_state_source=state_source)
    return result



def construct_interaction_region_frame(source, transform, primary, *,
                                       current_region_owner=None, same_state_source=None):
    """Construct a manipulation interaction frame from an accepted region.

    W denotes the current calibrated world. The origin is primary.center in W,
    the observed seed used by the accepted region (not its mask centroid).
    For declared planar geometry, z is the signed observed normal and x is
    the transferred source functional x projected into that tangent plane;
    y=z cross x. For declared surface geometry there is no single region
    normal: the source functional rotation is transferred unchanged.
    Source functional frames and represented motion are never rewritten.

    The accepted primary and its supporting transform are supplied together
    by run_region_transfer. This API changes reference construction only;
    region acceptance and candidate evidence retain their original values.
    """
    source_frame = (_pose_array(source["T_world_body_at_image"], "Source body at region image time")
                    @ _pose_array(source["L_body_region"], "Source body-local region frame"))
    rotated = np.asarray(transform["rotation"]) @ source_frame[:3, :3]
    geometry = source["region_geometry"]
    if geometry == "planar":
        normal = np.asarray(primary["normal"], dtype=float)
        normal = normal / np.linalg.norm(normal)
        tangent = rotated[:, 0] - normal * (normal @ rotated[:, 0])
        current = function_frame(primary["center"], tangent, normal)
        axes_source = "observed_normal_and_projected_transferred_functional_x"
    elif geometry == "surface":
        current = np.eye(4)
        current[:3, :3], current[:3, 3] = rotated, primary["center"]
        axes_source = "transferred_source_functional_rotation"
    else:
        raise ValueError("source.region_geometry must explicitly be planar or surface")
    result = _region_frame_record(source, source_frame,
        _pose_array(current, "Observed interaction region frame"), transform["hypothesis_id"],
        "accepted observed interaction-region reference; not a unique-pose certificate",
        current_region_owner, same_state_source)
    result["construction"] = dict(origin_source="accepted_primary.center",
        center_meaning="observed region seed, not mask centroid", axes_source=axes_source,
        region_geometry=geometry)
    return result
# END grasp source-frame propagation retains its original Sim(3) computation.


def run_region_transfer(source, views, *, skill_kind, same_state_source, current_region_owner=None):
    """Call the existing resident dual-view pipeline once and bind its frame.

    ``skill_kind`` is the existing representation.kind declaration. Grasp
    retains source-frame Sim(3) propagation. Manipulation constructs its
    interaction frame from the accepted primary center and declared geometry.
    The numerical region pipeline is identical for both declarations.
    The caller first runs dual_view.pipeline.prepare in its serial supervised
    GPU process. Missing inputs and an unsupported source plane retain explicit
    states; other malformed-input/runtime errors propagate. Returned loaded
    point banks and pipeline assets retain their separate index domains.
    When a current region-owner pose is supplied, ``same_state_source`` must
    also cover that pose, not merely the two images: either synchronized state
    records or explicit absence of owner motion between their acquisition times.
    """
    from transfer.dual_view import pipeline

    if skill_kind not in ("grasp", "manipulation"):
        raise ValueError("skill_kind must be grasp or manipulation")
    try:
        loaded, features = prepare_inputs(source, views, same_state_source=same_state_source)
    except RegionInputMissing as error:
        return {"status": "missing", "reasons": error.reasons,
                "reference_frame": None, "outputs": None}
    try:
        outputs = pipeline.run(loaded, features)
    except pipeline.InsufficientSource as error:
        return {"status": "rejected", "reasons": error.reasons,
                "reference_frame": None, "outputs": None, "loaded": loaded}
    result = outputs["result"]
    selected = select_reference_transform(result)
    frame = None
    if selected is not None:
        rows = selected["paired_query_rows"]
        target_ids = selected["paired_surface_indices"]
        if (rows.ndim != 1 or target_ids.shape != rows.shape
                or np.any(rows < 0) or np.any(rows >= len(loaded["query_indices"]))
                or np.any(target_ids < 0) or np.any(target_ids >= len(outputs["surface"]["xyz"]))):
            raise ValueError("Selected candidate pairing lost its source/target index domains")
        selected["source_point_indices"] = loaded["query_indices"][rows]
        # BEGIN select the frame contract from the declared representation kind.
        if skill_kind == "manipulation":
            primary = next(mode for mode in result["modes"] if mode["type"] == "primary")
            frame = construct_interaction_region_frame(source, selected, primary,
                current_region_owner=current_region_owner, same_state_source=same_state_source)
        else:
            frame = propagate_region_frame(source, selected, current_region_owner=current_region_owner,
                                           same_state_source=same_state_source)
        # END grasp retains the exact existing source origin and full rotation.
        frame["target_entity_id"] = loaded["target_entity_id"]
    return {"status": result["status"], "reasons": list(result["reasons"]),
            "reference_frame": frame, "selected_transform": selected,
            "outputs": outputs, "loaded": loaded}


def frame_from_deformation(coarse_source_points, deformed_source_points,
                           source_region_indices, coarse_frame):
    """Fit one rigid frame from OAReg's same-source-index deformation output.

    Inputs and output points already use the target-world coordinates. OAReg
    preserved the source order; q[i] is an estimated deformation, not target
    measurement i. Open3D fits p[i] -> q[i] with scaling disabled, then updates
    the already coarse-aligned frame once. No trajectory is deformed here.
    """
    source = _points(coarse_source_points, "Coarse source points")
    deformed = _points(deformed_source_points, "Deformed source points")
    indices = np.asarray(source_region_indices)
    if (source.shape != deformed.shape or indices.ndim != 1 or indices.dtype.kind not in "iu"
            or len(indices) < 3 or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(source))):
        raise ValueError("Deformation must preserve source shape and distinct source-region indices")
    if min(rank(source[indices])[0], rank(deformed[indices])[0]) < 2:
        raise ValueError("Local frame fitting requires noncollinear region points")
    import open3d as o3d

    first, second = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    first.points = o3d.utility.Vector3dVector(source[indices])
    second.points = o3d.utility.Vector3dVector(deformed[indices])
    correspondence = o3d.utility.Vector2iVector(np.column_stack((np.arange(len(indices)),) * 2))
    correction = o3d.pipelines.registration.TransformationEstimationPointToPoint(False).compute_transformation(
        first, second, correspondence)
    correction = _pose_array(correction, "OAReg local rigid correction")
    frame = correction @ _pose_array(coarse_frame, "Coarse region frame")
    predicted = source[indices] @ correction[:3, :3].T + correction[:3, 3]
    return {"T_world_region": frame, "T_local_correction": correction,
            "source_region_indices": indices.copy(),
            "rigid_fit_rms_m": float(np.sqrt(np.mean(np.sum((predicted - deformed[indices]) ** 2, axis=1)))),
            "scope": "local rigid approximation of deformed source points; not measured target correspondences"}
# END frame/point correspondence stays separate from task constraints and planning.
