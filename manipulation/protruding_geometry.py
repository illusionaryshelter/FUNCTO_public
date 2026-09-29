"""Observe a protruding straight region relative to an observed backing plane.

The caller has already selected the local interaction region and the appropriate
backing surface. Neither mask means the complete owner. All geometry is in the
same calibrated camera coordinates, in metres. The functions use exactly the
same definition for source and current observations and never assign a material
endpoint identity, invent a physical centre, or certify a grasp as executable.
"""

# BEGIN observed protruding-line geometry with an explicit matched anchor.
import numpy as np
from scipy.ndimage import binary_erosion, label

from .observations import _intrinsics, _mask, backproject


# These are measurement acceptance limits, not dimensions of a particular
# object. Five millimetres is the existing observed-plane tolerance. Resolving
# a protrusion requires twice that separation; a directional extent requires
# four times it. Smaller features require more precise evidence than this
# estimator accepts, rather than silently receiving an invented direction.
_SURFACE_TOLERANCE_M = 0.005
_MIN_PROTRUSION_M = 2 * _SURFACE_TOLERANCE_M
_MIN_LINE_EXTENT_M = 4 * _SURFACE_TOLERANCE_M
_MIN_POINTS = 12
_MIN_COMPONENT_PIXELS = 4
# A singular-value gap of four is a covariance-eigenvalue gap of sixteen.
_MIN_LINE_GAP = 4.0
# Projection only removes a bounded measurement discrepancy. A materially
# nonparallel observed line is rejected before orthogonalisation.
_MAX_LINE_NORMAL_COSINE = float(np.sin(np.deg2rad(10.0)))


def _vector(value, name, *, unit=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite three-vector")
    result = result.copy()
    if unit:
        length = np.linalg.norm(result)
        if length <= 1e-12:
            raise ValueError(f"{name} must be nonzero")
        result /= length
    return result


def _remove_isolated(mask):
    """Discard tiny observed islands; never fill a hole or join components."""
    components, _ = label(mask, structure=np.ones((3, 3), bool))
    counts = np.bincount(components.ravel())
    keep = counts >= _MIN_COMPONENT_PIXELS
    keep[0] = False
    return keep[components]


def _line_axis(points, outward):
    """Fit an unoriented line; the mean is only a numerical fitting origin."""
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < _MIN_POINTS
            or not np.isfinite(points).all()):
        raise ValueError("Insufficient observed points for a line direction")
    normal = _vector(outward, "outward_normal", unit=True)
    _, singular, axes = np.linalg.svd(points-points.mean(axis=0), full_matrices=False)
    if singular[0] <= 1e-12 or singular[0] < _MIN_LINE_GAP * singular[1]:
        raise ValueError("Observed front region has no sufficiently distinct line direction")
    raw = axes[0].copy()
    alignment = float(abs(raw @ normal))
    if alignment > _MAX_LINE_NORMAL_COSINE:
        raise ValueError("Observed line is not sufficiently parallel to the backing plane")
    axis = raw - (raw @ normal) * normal
    axis /= np.linalg.norm(axis)
    extent = float(np.ptp(points @ axis))
    if extent < _MIN_LINE_EXTENT_M:
        raise ValueError("Insufficient observed line extent at the accepted surface tolerance")
    return axis, {
        "singular_values_m": singular.tolist(),
        "line_gap": float(singular[0] / max(singular[1], 1e-12)),
        "raw_line_normal_abs_cosine": alignment,
        "observed_line_extent_m": extent,
    }


def observe_protruding_line(depth, K, region_mask, plane):
    """Return ``fit_mask``, observed ``front_mask``, directions and diagnostics.

    ``plane`` contains finite ``origin`` / unsigned ``normal``, its actual
    boolean observed ``mask`` and nonnegative ``plane_rms_m``. Selecting which
    surface backs this region is the caller's responsibility; the largest or
    closest plane is not automatically the correct one.

    The direction's ``fit_mask`` excludes one-pixel segmentation boundaries,
    large measured 3-D neighbour jumps and tiny islands. Its front band is
    within 5 mm of the 90th percentile of positive plane distances: a robust
    observed high end, not an extrapolated maximum. The accepted band must
    independently exhibit a long, dominant, plane-parallel direction.

    Correspondence uses ``front_mask``: original valid region pixels within
    that same metric band, in observed connected components that contain fit
    pixels. A boundary pixel can have valid own-surface depth even when a
    neighbour sees background, so erosion is not an anchor requirement. This
    separation never fills holes, invents pixels or changes a matched anchor.
    Neither mask reconstructs a complete owner, hidden geometry, material
    endpoint identity or physical symmetry.
    """
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("Depth must be a nonempty two-dimensional array")
    K = _intrinsics(K)
    region = _mask(region_mask, depth.shape, "region_mask")
    plane_mask = _mask(plane["mask"], depth.shape, "plane mask")
    origin = _vector(plane["origin"], "plane origin")
    normal = _vector(plane["normal"], "plane normal", unit=True)
    declared_rms = float(plane["plane_rms_m"])
    if not np.isfinite(declared_rms) or not 0 <= declared_rms <= _SURFACE_TOLERANCE_M:
        raise ValueError("Backing plane RMS exceeds the accepted surface tolerance")
    valid = np.isfinite(depth) & (depth > 0)
    plane_points = backproject(depth, K, mask=plane_mask & valid)
    if len(plane_points) < _MIN_POINTS:
        raise ValueError("Insufficient observed backing-plane points")
    measured_rms = float(np.sqrt(np.mean(((plane_points-origin) @ normal)**2)))
    plane_singular = np.linalg.svd(plane_points-plane_points.mean(axis=0), compute_uv=False)
    if measured_rms > _SURFACE_TOLERANCE_M or plane_singular[1] <= 1e-8:
        raise ValueError("Backing plane lacks independently observed planar support")

    # Erosion deliberately rejects one-pixel fragments instead of using a
    # boundary's potentially unrelated depth as evidence for a functional axis.
    core = binary_erosion(region & valid, structure=np.ones((3, 3), bool), border_value=0)
    yy, xx = np.nonzero(core)
    if len(xx) < _MIN_POINTS:
        raise ValueError("Insufficient interior region observations after boundary removal")
    points = backproject(depth, K, np.column_stack((xx, yy)))
    jumps = []
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)):
        neighbours = backproject(depth, K, np.column_stack((xx+dx, yy+dy)))
        jumps.append(np.linalg.norm(neighbours-points, axis=1))
    jumps = np.asarray(jumps)
    # Four ordinary observed neighbour spacings accommodates local perspective
    # sampling; 10 mm is the metric floor. Distances, unlike optical-z changes,
    # are invariant to the coordinate frame in which the observations are read.
    jump_limit = max(2 * _SURFACE_TOLERANCE_M, 4 * float(np.median(jumps)))
    core[yy, xx] = np.max(jumps, axis=0) <= jump_limit
    core = _remove_isolated(core)
    yy, xx = np.nonzero(core)
    if len(xx) < _MIN_POINTS:
        raise ValueError("Insufficient continuous region observations after depth-edge removal")
    points = backproject(depth, K, np.column_stack((xx, yy)))
    signed = (points-origin) @ normal
    median = float(np.median(signed))
    if abs(median) < _MIN_PROTRUSION_M:
        raise ValueError("Protrusion does not resolve the outward side of the backing plane")
    outward = normal if median > 0 else -normal
    heights = (points-origin) @ outward
    if np.quantile(heights, 0.1) < -_SURFACE_TOLERANCE_M:
        raise ValueError("Region observations occupy inconsistent sides of the backing plane")
    front_height = float(np.quantile(heights, 0.9))
    fit_mask = np.zeros_like(region)
    fit_mask[yy, xx] = np.abs(heights-front_height) <= _SURFACE_TOLERANCE_M
    fit_mask = _remove_isolated(fit_mask)
    fit_points = backproject(depth, K, mask=fit_mask)
    axis, line_diagnostics = _line_axis(fit_points, outward)

    # BEGIN separate robust direction fitting from measured anchor membership.
    # Only the original selected region is considered. Connectivity restricts
    # admissible surface pixels to the actually observed front structure; this
    # is not dilation, hole filling or an extrapolated physical bar outline.
    rows, cols = np.nonzero(region & valid)
    observed = backproject(depth, K, np.column_stack((cols, rows)))
    observed_heights = (observed-origin) @ outward
    observed_front = np.zeros_like(region)
    observed_front[rows, cols] = np.abs(observed_heights-front_height) <= _SURFACE_TOLERANCE_M
    components, _ = label(observed_front, structure=np.ones((3, 3), bool))
    connected_to_fit = np.unique(components[fit_mask])
    connected_to_fit = connected_to_fit[connected_to_fit != 0]
    front = observed_front & np.isin(components, connected_to_fit)
    # END separate robust direction fitting from measured anchor membership.
    diagnostics = {
        **line_diagnostics,
        "region_pixel_count": int(region.sum()),
        "interior_pixel_count": int(core.sum()),
        "fit_pixel_count": int(fit_mask.sum()),
        "front_pixel_count": int(front.sum()),
        "plane_declared_rms_m": declared_rms,
        "plane_observed_rms_m": measured_rms,
        "front_height_m": front_height,
        "surface_tolerance_m": _SURFACE_TOLERANCE_M,
        "depth_jump_limit_m": jump_limit,
        "anchor_mask_definition": "original valid observed front-band components containing reliable fit pixels",
        "axis_identity": "unoriented observed line; neither material endpoint is identified",
        "scope": "local interaction geometry; neither complete owner nor executable grasp",
    }
    return {"fit_mask": fit_mask, "front_mask": front, "line_axis": axis, "outward_normal": outward,
            "plane_origin": origin, "diagnostics": diagnostics}


def protruding_frames(depth, K, anchor_pixel, geometry):
    """Return ``(frames[2,4,4], diagnostics)`` at an explicit matched pixel.

    Use the very same RGB-D calibration that produced ``geometry``. The anchor
    must lie in its observed front mask, and its original depth is retained.
    It is never projected onto a plane/line, moved to a centroid, or called the
    physical centre. The orientations are z=+/-line, y=inward, x=y cross z.
    Both are geometric alternatives: the skill and motion backend must decide
    whether either (or both) is physically permissible and executable.
    """
    depth = np.asarray(depth, dtype=float)
    mask = _mask(geometry["front_mask"], depth.shape, "front_mask")
    pixel = np.asarray(anchor_pixel, dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("anchor_pixel must be one finite (u, v) pixel")
    # backproject enforces bounds, finite positive depth and actual mask
    # membership, preserving the caller's subpixel ray if supplied.
    anchor = backproject(depth, K, pixel[None], mask=mask)[0]
    outward = _vector(geometry["outward_normal"], "outward_normal", unit=True)
    axis = _vector(geometry["line_axis"], "line_axis", unit=True)
    if abs(float(axis @ outward)) > 1e-6:
        raise ValueError("Geometry line_axis and outward_normal must already be orthogonal")
    height = float((anchor-_vector(geometry["plane_origin"], "plane_origin")) @ outward)
    if abs(height-geometry["diagnostics"]["front_height_m"]) > _SURFACE_TOLERANCE_M:
        raise ValueError("Anchor depth does not belong to the observed front surface")
    frames = np.repeat(np.eye(4)[None], 2, axis=0)
    inward = -outward
    for frame, sign in zip(frames, (1., -1.)):
        z = sign * axis
        frame[:3, :3] = np.column_stack((np.cross(inward, z), inward, z))
        frame[:3, 3] = anchor
    diagnostics = {**geometry["diagnostics"], "anchor_pixel": pixel.tolist(),
                   "anchor_camera_m": anchor.tolist(),
                   "origin_definition": "explicit matched pixel with original measured depth",
                   "orientation_candidates": "two unsigned-line alternatives; execution symmetry unverified"}
    return frames, diagnostics
# END observed protruding-line geometry with an explicit matched anchor.
