"""Check rigid registration hypotheses against independently measured depth.

The source is an observed surface of a rigid owner, not a complete mesh. This
check applies only when that geometry is known to remain valid for the current
owner. A predicted surface in observed free space contradicts the hypothesis;
a predicted surface behind measured depth may be occluded and stays unknown.
Precise current contact masks are unnecessary. No simulator state, noise model
or motion planner is used.
"""

# BEGIN perception-only, same-geometry registration validation.
from collections.abc import Mapping

import cv2
import numpy as np

from .motion import _pose_array
from .observations import _intrinsics, backproject


# The existing correspondence path uses 10 mm as a consistency gate. Keep
# discontinuity exclusion fixed even when a caller changes the ray comparison
# gate; neither parameter estimates sensor variance or certifies safe motion.
_DEPTH_CONTINUITY_M = .01


def _name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _apply(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def _observation(record):
    # BEGIN owner membership does not imply complete visible-object coverage.
    """Validate metric depth and selected owner evidence, including empties.

    Empty masks or invalid depth express missing observation. They must not
    turn into positive evidence, nor be filled with invented surfaces. Input
    arrays are never modified. ``owner_mask`` may cover only observed surfaces
    of the bound rigid entity; neither source nor current masks assert complete
    owner coverage. Positive support is restricted to its eroded interior,
    while measured depth outside it may still contradict a predicted surface
    in observed free space. Current images need no functional-contact labels.
    """
    # END partial evidence has the same conservative ray semantics.
    depth = np.asarray(record["depth"], float)
    mask = np.asarray(record["owner_mask"])
    K = _intrinsics(record["K"])
    if depth.ndim != 2 or min(depth.shape) < 3:
        raise ValueError("Metric depth must be a two-dimensional image at least 3x3")
    if mask.shape != depth.shape or mask.dtype != np.bool_:
        raise ValueError("owner_mask must be a boolean image aligned with depth")
    kernel = np.ones((3, 3), np.uint8)
    core = cv2.erode(mask.astype(np.uint8), kernel,
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    valid = np.isfinite(depth) & (depth > 0)
    complete = cv2.erode(valid.astype(np.uint8), kernel,
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    low = cv2.erode(np.where(valid, depth, np.inf).astype(np.float32), kernel)
    high = cv2.dilate(np.where(valid, depth, -np.inf).astype(np.float32), kernel)
    # OpenCV may replace all-invalid extrema with finite float32 sentinels.
    # Subtract only in complete neighborhoods, avoiding irrelevant overflow.
    span = np.full(depth.shape, np.inf, np.float32)
    np.subtract(high, low, out=span, where=complete)
    stable = complete & (span <= _DEPTH_CONTINUITY_M)
    return {"depth": depth, "K": K, "core": core,
            "low": low, "high": high, "stable": stable}


def _ray_evidence(points_output, view, max_error):
    """Compare each nearest projected surface with a conservative measured ray.

    Z-buffering prevents multiple source points from adding votes at one pixel.
    The shortest depth in a continuous 3x3 neighborhood bounds free space;
    discontinuities, invalid depth and points behind observed surfaces remain
    unknown. Positive support additionally requires the selected owner mask.
    This is finite-resolution evidence, not a continuous visibility theorem.
    """
    camera = _apply(points_output, np.linalg.inv(view["T_output_camera"]))
    indices = np.flatnonzero(camera[:, 2] > 0)
    projected = camera[indices] @ view["K"].T
    uv = np.rint(projected[:, :2] / projected[:, 2, None])
    height, width = view["depth"].shape
    inside = (np.isfinite(uv).all(1) & (uv[:, 0] >= 1) & (uv[:, 0] < width-1)
              & (uv[:, 1] >= 1) & (uv[:, 1] < height-1))
    indices, pixels = indices[inside], uv[inside].astype(np.int64)
    order = np.argsort(camera[indices, 2], kind="stable")
    keys = pixels[:, 1]*width+pixels[:, 0]
    _, unique = np.unique(keys[order], return_index=True)
    chosen = order[unique]
    indices, pixels = indices[chosen], pixels[chosen]
    z = camera[indices, 2]
    x, y = pixels.T
    reliable = view["stable"][y, x]
    free = reliable & (z < view["low"][y, x]-max_error)
    near = (reliable & ~free & view["core"][y, x]
            & (np.abs(z-view["depth"][y, x]) <= max_error))
    labels = np.zeros(len(pixels), np.uint8)
    labels[near], labels[free] = 1, 2
    support = camera[indices[near]]
    singular = (np.linalg.svd(support-support.mean(0), compute_uv=False)
                if len(support) >= 3 else np.zeros(3))
    clearance = view["low"][y, x]-z
    summary = {
        "source_sample_count": len(points_output),
        "source_samples_outside_usable_image_or_behind_camera": int(len(points_output)-inside.sum()),
        "projected_source_samples": int(inside.sum()), "unique_projected_rays": len(pixels),
        "stable_depth_rays": int(reliable.sum()), "free_space_conflict_rays": int(free.sum()),
        "near_selected_owner_rays": int(near.sum()), "unknown_rays": int((labels == 0).sum()),
        "occluded_rays": int((reliable & (z > view["high"][y, x]+max_error)).sum()),
        "depth_boundary_or_invalid_rays": int((~reliable).sum()),
        "positive_support_noncollinear": bool(len(support) >= 3 and singular[1] > 1e-8),
        "largest_free_space_conflict_m": float(clearance[free].max()) if free.any() else 0.,
        "median_free_space_conflict_m": float(np.median(clearance[free])) if free.any() else 0.,
    }
    return {"summary": summary, "rays": {
        "pixels": pixels, "labels": labels, "predicted_depth_m": z,
        "observed_min_depth_m": view["low"][y, x], "source_indices": indices,
        "label_meaning": {"0": "unknown", "1": "owner_depth_support", "2": "free_space_conflict"}}}


def screen_registrations(source, views, transforms, *, reference_frame,
                         same_rigid_geometry, geometry_source, max_error=.01):
    """Screen an arbitrary nonempty set of same-owner rigid hypotheses.

    source: owner_id, depth (camera-z metres), K, owner_mask, T_camera_owner.
    views: nonempty view-ID -> {owner_id, reference_frame, depth, K, owner_mask,
                               T_output_camera}; views observe the same state.
    transforms: unique candidate-ID -> output-from-source-owner SE(3).

    ``owner_mask`` identifies visible evidence of the selected rigid entity;
    it need not cover its whole image footprint. Missing mask coverage adds no
    positive support and is never interpreted as evidence that the owner ends
    there. The caller establishes mask membership; this geometric check does
    not identify owners or merge independently moving parts into one body.

    Geometry applicability must be explicitly established by the caller, e.g.
    a known unchanged rigid instance. If false, return not_applicable with no
    selection; never silently assert equal geometry for a novel instance.
    Owner IDs identify bindings, not simulator assets. All hypotheses face all
    supplied views under the same gate, independent of candidate names/order.

    A clear conflict rejects a hypothesis. Otherwise this conservative entry
    requires noncollinear owner-depth support in every supplied view. Inadequate
    coverage remains unresolved. Select only one supported hypothesis with no
    unresolved competitors; ties stay ambiguous rather than ranked by length.
    The result is registration evidence, not task completion or execution safety.
    """
    _name(reference_frame, "reference_frame")
    source_id = _name(source["owner_id"], "source owner_id")
    _name(geometry_source, "geometry_source")
    if not isinstance(same_rigid_geometry, (bool, np.bool_)):
        raise ValueError("same_rigid_geometry must be an explicit boolean")
    if (isinstance(max_error, (bool, np.bool_)) or not np.isscalar(max_error)
            or not isinstance(max_error, (int, float, np.number))
            or not np.isfinite(max_error) or max_error <= 0):
        raise ValueError("max_error must be a finite positive metric distance")
    if not isinstance(views, Mapping) or not views:
        raise ValueError("At least one named current view is required")
    if not isinstance(transforms, Mapping) or not transforms:
        raise ValueError("At least one named registration hypothesis is required")
    checked = {}
    owner_ids = set()
    for name, record in views.items():
        _name(name, "view ID")
        owner_ids.add(_name(record["owner_id"], "current owner_id"))
        if record.get("reference_frame") != reference_frame:
            raise ValueError("Every camera must use the declared common reference_frame")
        checked[name] = {**_observation(record),
                         "T_output_camera": _pose_array(record["T_output_camera"], "T_output_camera")}
    if len(owner_ids) != 1:
        raise ValueError("All views must observe the same selected rigid owner")
    poses = {_name(name, "candidate ID"): _pose_array(value, "T_output_source_owner")
             for name, value in transforms.items()}
    observed = _observation(source)
    source_pose = _pose_array(source["T_camera_owner"], "source T_camera_owner")
    result = {
        "status": "not_applicable", "selected_candidate_id": None,
        "reference_frame": reference_frame, "source_owner_id": source_id,
        "target_owner_id": next(iter(owner_ids)), "geometry_source": geometry_source,
        "same_rigid_geometry": bool(same_rigid_geometry), "max_error_m": float(max_error),
        "depth_continuity_m": _DEPTH_CONTINUITY_M,
        "scope": "rigid registration versus measured rays; no robot, collision or complete inside certificate",
        "supported_candidate_ids": [], "rejected_candidate_ids": [], "unresolved_candidate_ids": [],
        "candidates": {},
    }
    if not same_rigid_geometry:
        result["reason"] = "Source surface geometry is not established for the current owner"
        return result
    y, x = np.nonzero(observed["core"] & observed["stable"])
    pixels = np.column_stack((x, y))
    points = backproject(observed["depth"], observed["K"], pixels)
    points = _apply(points, np.linalg.inv(source_pose))
    result["source_sample_count"] = len(points)
    for name, transform in poses.items():
        world = _apply(points, transform)
        evidence = {view_name: _ray_evidence(world, view, max_error)
                    for view_name, view in checked.items()}
        if any(row["summary"]["free_space_conflict_rays"] for row in evidence.values()):
            status = "rejected"
        elif all(row["summary"]["positive_support_noncollinear"] for row in evidence.values()):
            status = "supported"
        else:
            status = "unresolved"
        result[status + "_candidate_ids"].append(name)
        result["candidates"][name] = {"status": status, "T_output_source_owner": transform,
                                      "views": evidence}
    for key in ("supported_candidate_ids", "rejected_candidate_ids", "unresolved_candidate_ids"):
        result[key].sort()
    supported, unresolved = result["supported_candidate_ids"], result["unresolved_candidate_ids"]
    if unresolved:
        result["status"] = "unresolved"
    elif len(supported) == 1:
        result.update(status="selected", selected_candidate_id=supported[0])
    else:
        result["status"] = "ambiguous" if supported else "rejected"
    return result
# END perception-only, same-geometry registration validation.
