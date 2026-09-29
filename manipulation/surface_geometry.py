"""Observe a functional frame from two semantically selected visible surfaces.

Only aligned metric RGB-D, a visible owner mask and two existing semantic
queries are consumed. Plane membership supplies the directions; an observed
query supplies the origin. This does not recover complete object pose, assign
material identities to symmetric faces, or certify an executable grasp.
"""

# BEGIN observed two-surface geometry with explicit semantic queries.
import numpy as np
from scipy.ndimage import binary_erosion

from .observations import _intrinsics, _mask, backproject
from .regions import function_frame
from .support_grounding import (
    _DISTANCE_THRESHOLD,
    _MIN_SIDE_ALIGNMENT,
    _plane_candidates,
)


# Match the existing observed-plane estimator's minimum support. The relative
# singular-value bound only rejects a numerically collinear cloud; it is not a
# task-specific size prior or a claim about the accuracy of noisy depth.
_MIN_CORE_POINTS = 24
_MIN_SECOND_SPAN_M = 1e-8
_MIN_SECOND_SPAN_RATIO = 1e-5


def _query(value, name, depth, K, valid):
    """Retain the input ray and the existing nearest-pixel depth convention."""
    pixel = np.asarray(value, dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError(f"{name} must be a finite (u, v) pixel")
    # This checks bounds, own measured depth and owner membership. Rounding is
    # used only for image membership; it never replaces a subpixel query ray.
    point = backproject(depth, K, pixel[None], mask=valid)[0]
    column, row = np.rint(pixel).astype(np.int64)
    return pixel.copy(), (row, column), point


def _selected_candidate(candidates, index, name):
    """A nearest infinite plane is not evidence that a pixel belongs to it."""
    matches = [candidate for candidate in candidates if candidate["mask"][index]]
    if len(matches) != 1:
        raise ValueError(
            f"{name} must belong to exactly one observed plane candidate; "
            f"found {len(matches)}"
        )
    return matches[0]


def _refit_surface(depth, K, valid, candidates, selected, pixel, index, name):
    """Refit only nonoverlapping observed interior pixels of a selected face."""
    mask = selected["mask"] & valid
    candidate_count = int(mask.sum())
    others = np.zeros_like(valid)
    for candidate in candidates:
        if candidate is not selected:
            others |= candidate["mask"]
    exclusive = mask & ~others
    # A single erosion excludes observed silhouette, depth-hole and candidate
    # intersection boundaries. It never fills a hole or adds hidden geometry.
    core = binary_erosion(
        exclusive, structure=np.ones((3, 3), dtype=bool),
        iterations=1, border_value=0,
    )
    if not core[index]:
        raise ValueError(f"{name} lies outside its nonoverlapping eroded plane core")
    core_count = int(core.sum())
    if core_count < _MIN_CORE_POINTS:
        raise ValueError(f"{name} plane core has fewer than {_MIN_CORE_POINTS} observed points")
    points = backproject(depth, K, mask=core)
    origin = points.mean(axis=0)
    centered = points-origin
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    if singular[1] <= max(_MIN_SECOND_SPAN_M, singular[0] * _MIN_SECOND_SPAN_RATIO):
        raise ValueError(f"{name} plane core lacks noncollinear two-dimensional support")
    normal = axes[-1].copy()
    rms = float(np.sqrt(np.mean((centered @ normal)**2)))
    if not np.isfinite(rms) or rms > _DISTANCE_THRESHOLD:
        raise ValueError(f"{name} plane core exceeds the observed-plane distance tolerance")

    # The camera is the origin of this calibrated coordinate system. A visible
    # opaque exterior surface faces it; that explicit caller assumption signs
    # the fitted normal. Near-grazing surfaces cannot reliably establish side.
    to_camera = -origin / np.linalg.norm(origin)
    alignment = float(normal @ to_camera)
    if abs(alignment) <= _MIN_SIDE_ALIGNMENT:
        raise ValueError(f"{name} surface is near-grazing; its visible side is ambiguous")
    if alignment < 0:
        normal = -normal
    coarse_normal = np.asarray(selected["normal"], dtype=float)
    coarse_normal = coarse_normal / np.linalg.norm(coarse_normal)
    normal_change = float(np.arccos(np.clip(abs(coarse_normal @ normal), 0., 1.)))
    return {
        "candidate_id": int(selected["candidate_id"]),
        "core_mask": core,
        "origin": origin,
        "normal": normal,
        "plane_rms_m": rms,
        "singular_values_m": singular.tolist(),
        "core_pixel_count": core_count,
        "candidate_pixel_count": candidate_count,
        "exclusive_pixel_count": int(exclusive.sum()),
        "candidate_plane_rms_m": float(selected["plane_rms_m"]),
        "candidate_to_core_normal_angle_rad": normal_change,
        "camera_facing_alignment": abs(alignment),
        "query_pixel": pixel.tolist(),
    }


def observe_orthogonal_surface_frame(
        depth, K, owner_mask, axis_pixel, anchor_pixel, *,
        surface_side_source, orthogonality_tolerance_rad=np.deg2rad(5.0)):
    """Return one observed frame and both supporting surface measurements.

    All geometry uses camera coordinates and metres. ``axis_pixel`` identifies
    the semantic surface whose outward normal supplies functional x;
    ``anchor_pixel`` identifies the other semantic surface and supplies the
    original measured functional origin. The caller must use the same role
    convention for source and current observations. Neither role is inferred
    from world axes, an object category, a plane's size or PCA tangent axes.

    ``surface_side_source`` must be a nonempty textual declaration that the
    selected pixels show visible opaque exterior surfaces, with the source of
    that assumption. Signing toward the camera is conditional on this supplied
    physical interpretation; this function cannot establish opacity from depth.

    Existing 5 mm plane candidates serve only as coarse membership proposals.
    Each query must uniquely select a different candidate and remain inside its
    nonoverlapping, once-eroded 3-by-3 core. NumPy SVD refits those actual points.
    Each core needs at least 24 noncollinear points; ambiguous, near-grazing or
    insufficiently orthogonal observations raise without a geometric fallback.

    For signed unit normals a (axis surface) and b (anchor surface), x=a,
    z=normalize(b cross a), y=z cross x. The small accepted deviation of b from
    perpendicularity is removed and reported explicitly. The frame origin is
    exactly the anchor's own-depth backprojection, never a surface centroid,
    intersection, projected point or inferred complete-object centre.

    The result contains ``frame``, ``planes`` (``axis`` and ``anchor`` records
    with their actual ``core_mask``, fitted ``origin`` / ``normal`` and numerical
    diagnostics), and ``diagnostics``. Plane origins are computational fitting
    references only. Returning a frame does not resolve symmetric material-face
    identity, hidden geometry, collision clearance or physical grasp stability.
    """
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("Depth must be a nonempty two-dimensional array")
    K = _intrinsics(K)
    owner = _mask(owner_mask, depth.shape, "owner_mask")
    if not isinstance(surface_side_source, str) or not surface_side_source.strip():
        raise ValueError("surface_side_source must declare visible opaque exterior-surface evidence")
    if (isinstance(orthogonality_tolerance_rad, (bool, np.bool_))
            or not np.isscalar(orthogonality_tolerance_rad)):
        raise ValueError("orthogonality_tolerance_rad must be a finite scalar in [0, pi/2)")
    tolerance = float(orthogonality_tolerance_rad)
    if not np.isfinite(tolerance) or not 0 <= tolerance < np.pi / 2:
        raise ValueError("orthogonality_tolerance_rad must be a finite scalar in [0, pi/2)")
    valid = owner & np.isfinite(depth) & (depth > 0)
    if int(valid.sum()) < 2 * _MIN_CORE_POINTS:
        raise ValueError("Owner has insufficient observed depth for two supported surfaces")
    axis_query, axis_index, _ = _query(axis_pixel, "axis_pixel", depth, K, valid)
    anchor_query, anchor_index, anchor_point = _query(
        anchor_pixel, "anchor_pixel", depth, K, valid)
    candidates, sampled_count = _plane_candidates(depth, K, valid)
    axis_selected = _selected_candidate(candidates, axis_index, "axis_pixel")
    anchor_selected = _selected_candidate(candidates, anchor_index, "anchor_pixel")
    if axis_selected is anchor_selected:
        raise ValueError("axis_pixel and anchor_pixel must select two different observed planes")
    axis = _refit_surface(
        depth, K, valid, candidates, axis_selected, axis_query, axis_index, "axis_pixel")
    anchor = _refit_surface(
        depth, K, valid, candidates, anchor_selected, anchor_query, anchor_index, "anchor_pixel")
    first, second = axis["normal"], anchor["normal"]
    angle = float(np.arccos(np.clip(first @ second, -1., 1.)))
    orthogonality_error = abs(angle-np.pi/2)
    if orthogonality_error > tolerance:
        raise ValueError("Observed surface normals exceed orthogonality_tolerance_rad")
    cross = np.cross(second, first)
    cross /= np.linalg.norm(cross)
    frame = function_frame(anchor_point, first, cross)
    # Negative y is the normalized part of the measured anchor-surface normal
    # perpendicular to x. Report this actual correction, not a fitted ideal
    # angle that could hide a mismatch between the two observed surfaces.
    correction = float(np.arccos(np.clip(second @ -frame[:3, 1], -1., 1.)))
    diagnostics = {
        "sample_count": int(sampled_count),
        "candidate_count": len(candidates),
        "normal_angle_rad": angle,
        "orthogonality_error_rad": orthogonality_error,
        "orthogonalization_angle_rad": correction,
        "orthogonality_tolerance_rad": tolerance,
        "surface_side_source": surface_side_source.strip(),
        "plane_distance_threshold_m": _DISTANCE_THRESHOLD,
        "minimum_camera_facing_alignment": _MIN_SIDE_ALIGNMENT,
        "minimum_core_points": _MIN_CORE_POINTS,
        "mask_erosion_pixels": 1,
        "origin_source": "anchor_pixel original observed depth; no plane projection or centroid",
        "direction_source": "two uniquely selected observed surface normals; no PCA tangent axis",
        "scope": "visible functional-frame observation; not complete pose or executable grasp",
    }
    return {"frame": frame, "planes": {"axis": axis, "anchor": anchor},
            "diagnostics": diagnostics}
# END observed two-surface geometry with explicit semantic queries.
