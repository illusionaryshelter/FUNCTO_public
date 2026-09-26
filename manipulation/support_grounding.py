"""Select observed support geometry instead of assuming the largest plane.

The reference instance is already selected by the caller. Its RGB-D points
produce metric plane candidates; Gemini may select one candidate or abstain,
but cannot supply geometry. A returned patch stays inside the selected plane's
observed image mask at pixel resolution, without filling holes or hidden areas.
This establishes a finite geometric support region, not force stability.
"""

import json

import numpy as np

from .goals import SupportSurface
from .grounding import _candidate_sheet, _client, _image, _mask
from .observations import backproject


_DISTANCE_THRESHOLD = 0.005
_MAX_POINTS = 6000
_MAX_PLANES = 4


def _plane_candidates(depth, K, valid):
    """Consensus on at most 6000 points, followed by bounded full-mask checks.

    Candidate masks contain actual valid-depth pixels within the fitted plane
    tolerance. The masks are never replaced by a hull, box, or filled contour.
    Closely intersecting planes may share observed boundary pixels; this does
    not authorize extending either plane through unobserved regions.
    """
    import open3d as o3d

    flat = np.flatnonzero(valid)
    selected = flat[np.linspace(0, len(flat)-1, min(_MAX_POINTS, len(flat))).astype(int)]
    width = depth.shape[1]
    pixels = np.column_stack((selected % width, selected // width))
    remaining = backproject(depth, K, pixels)
    sample_count = len(remaining)
    minimum = max(24, int(np.ceil(0.03 * sample_count)))
    candidates = []
    for _ in range(_MAX_PLANES):
        if len(remaining) < minimum:
            break
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(remaining))
        _, indices = cloud.segment_plane(_DISTANCE_THRESHOLD, 3, 1000)
        if len(indices) < minimum:
            break
        points = remaining[indices]
        keep = np.ones(len(remaining), dtype=bool)
        keep[indices] = False
        remaining = remaining[keep]
        origin = points.mean(axis=0)
        _, singular, axes = np.linalg.svd(points-origin, full_matrices=False)
        if singular[1] <= max(1e-8, singular[0] * 1e-5):
            continue
        normal = axes[-1]
        mask = np.zeros_like(valid)
        # The image can be large; never construct its whole XYZ point cloud.
        for start in range(0, len(flat), _MAX_POINTS):
            batch = flat[start:start + _MAX_POINTS]
            xy = np.column_stack((batch % width, batch // width))
            xyz = backproject(depth, K, xy)
            in_plane = np.abs((xyz-origin) @ normal) <= _DISTANCE_THRESHOLD
            mask.flat[batch[in_plane]] = True
        yy, xx = np.nonzero(mask)
        if len(xx) < minimum:
            continue
        candidates.append({
            "candidate_id": len(candidates)+1, "mask": mask,
            "bbox": [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
            "origin": origin, "normal": normal, "first_axis": axes[0],
            "sample_inlier_count": len(points), "observed_pixel_count": len(xx),
            "plane_rms_m": float(np.sqrt(np.mean(((points-origin) @ normal)**2))),
        })
    return candidates, sample_count


def _covered_polygon(polygon, mask):
    """Check the polygon's whole rasterized area and densely sampled boundary.

    Boundary checks alone miss holes. Conversely, filled-polygon checks alone
    hide their subpixel rounding convention. We check both, at quarter-pixel
    boundary spacing, against an already one-pixel-eroded observed mask. This
    supplies a conservative pixel-resolution certificate; it cannot certify
    unmeasured subpixel holes or the physical quality of depth observations.
    """
    import cv2

    if len(polygon) < 3 or cv2.contourArea(polygon.astype(np.float32)) <= 1:
        return False, 0
    raster = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(raster, np.rint(polygon * 256).astype(np.int32), 1, shift=8)
    if np.any(raster.astype(bool) & ~mask):
        return False, int(raster.sum())
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        count = max(2, int(np.ceil(np.linalg.norm(second-first) * 4)) + 1)
        samples = first + np.linspace(0, 1, count)[:, None] * (second-first)
        pixels = np.rint(samples).astype(int)
        if (np.any(pixels < 0) or np.any(pixels[:, 0] >= mask.shape[1])
                or np.any(pixels[:, 1] >= mask.shape[0])
                or not mask[pixels[:, 1], pixels[:, 0]].all()):
            return False, int(raster.sum())
    return True, int(raster.sum())


def _observed_patch(mask):
    """Retain a full convex observation, or a conservative large local patch.

    For ordinary convex panels the full safe hull is retained, so a long shelf
    is not always shrunk to a small disk. If that hull crosses a hole, concavity,
    or gap, OpenCV's exact Euclidean distance transform supplies the largest
    inscribed disk center. Its radius is reduced by one pixel and a 64-vertex
    inscribed polygon is checked again. We never fill or close the input mask.
    """
    import cv2

    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    yy, xx = np.nonzero(core)
    if len(xx) < 6:
        raise ValueError("Selected support has no usable fully observed local area")
    polygon = cv2.convexHull(np.column_stack((xx, yy)).astype(np.int32)).reshape(-1, 2).astype(float)
    covered, count = _covered_polygon(polygon, core)
    method = "full observed convex patch"
    if not covered:
        distances = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        row, col = np.unravel_index(int(np.argmax(distances)), distances.shape)
        radius = float(distances[row, col]) - 1.0
        if radius <= 1:
            raise ValueError("Selected support has insufficient contiguous observed area")
        angles = np.arange(64) * (2 * np.pi / 64)
        polygon = np.array([col, row]) + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        covered, count = _covered_polygon(polygon, core)
        method = "conservative polygon in largest observed inscribed disk"
    if not covered:
        raise ValueError("Cannot construct a support patch inside its observed mask")
    return polygon, {
        "patch_method": method, "patch_pixel_count": count,
        "patch_area_pixels_squared": float(cv2.contourArea(polygon.astype(np.float32))),
        "patch_boundary_pixels": polygon.tolist(), "mask_erosion_pixels": 1,
        "coverage_check": "entire rasterized area and quarter-pixel boundary samples inside eroded observed mask",
    }


def infer_support(image, depth_m, K, reference_mask, facing_point, *, instruction, client=None):
    """Select a task support from visible reference-instance plane candidates.

    RGB and depth must be aligned, depth must be metric, and ``K`` must describe
    that image. All returned geometry uses camera coordinates. Even a single
    candidate requires semantic confirmation: its largest plane may be a side
    panel. Selection accepts exactly ``{"candidate_id": ID_or_null}``.
    ``facing_point`` is a point on the intended carrier side at the goal. The
    caller must not substitute its arbitrary initial position: a lifted object
    can start below a tabletop. This side observation does not infer gravity.
    """
    image = _image(image)
    mask = _mask(reference_mask, image, "reference_mask")
    depth = np.asarray(depth_m, float)
    facing = np.asarray(facing_point, float)
    if depth.shape != mask.shape:
        raise ValueError("Support depth and RGB dimensions must agree")
    if facing.shape != (3,) or not np.isfinite(facing).all():
        raise ValueError("Support-facing point must be finite XYZ")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Support selection requires a nonempty instruction")
    valid = mask & np.isfinite(depth) & (depth > 0)
    if valid.sum() < 24:
        raise ValueError("Reference has insufficient valid depth for an observed support plane")
    candidates, sample_count = _plane_candidates(depth, K, valid)
    if not candidates:
        raise ValueError("Reference has no sufficiently supported observed plane")
    prompt = (
        "Select the actual visible support surface for this placement instruction. "
        "Image 1 is the original scene. Image 2 contains numbered panels, each "
        "highlighting only observed RGB-D pixels on one candidate plane inside "
        "the already selected reference object. A large vertical side, cabinet "
        "door, decorative panel, or hidden surface is not automatically a support. "
        "Even if there is only one candidate, confirm it is the surface intended "
        "by the instruction. If none is appropriate or the evidence is ambiguous, "
        "select null. Do not invent a surface, axis, normal, depth, or dimensions. "
        "Return exactly {\"candidate_id\": integer_ID_or_null}. Allowed IDs: "
        + json.dumps([candidate["candidate_id"] for candidate in candidates])
        + "\nInstruction: " + json.dumps(instruction.strip())
    )
    selected = _client(client).chat_json(prompt, images=[image, _candidate_sheet(image, candidates)])
    if not isinstance(selected, dict) or set(selected) != {"candidate_id"}:
        raise ValueError("Support selection must contain exactly candidate_id, not generated geometry")
    number = selected["candidate_id"]
    if type(number) is not int or not 1 <= number <= len(candidates):
        raise ValueError("No unambiguous task support candidate was selected")
    chosen = candidates[number-1]
    origin, normal = chosen["origin"], chosen["normal"].copy()
    side = float(normal @ (facing-origin))
    if abs(side) < _DISTANCE_THRESHOLD:
        raise ValueError("Cannot determine the selected support side from the facing point")
    if side < 0:
        normal = -normal
    polygon, patch_diagnostics = _observed_patch(chosen["mask"])
    rays = np.column_stack((polygon, np.ones(len(polygon)))) @ np.linalg.inv(np.asarray(K, float)).T
    denominators = rays @ normal
    if np.any(np.abs(denominators) < 1e-10):
        raise ValueError("Selected support crosses the plane projection horizon")
    distance = float(origin @ normal) / denominators
    if not np.isfinite(distance).all() or np.any(distance <= 0):
        raise ValueError("Selected support patch does not project to positive camera depth")
    # On a single visible plane with positive depth, the inverse perspective
    # map preserves straight edges and convexity. Thus this image polygon maps
    # to one convex metric patch, without hulling separated 3-D observations.
    vertices = rays * distance[:, None]
    first = chosen["first_axis"]
    basis = np.column_stack((first, np.cross(normal, first)))
    surface = SupportSurface(origin, basis, normal, (vertices-origin) @ basis,
                             source="semantically_selected_observed_RGBD_support_patch")
    diagnostics = {
        "candidate_count": len(candidates), "selected_candidate_id": number,
        "sampled_point_count": sample_count, "plane_distance_threshold_m": _DISTANCE_THRESHOLD,
        "selected_roi_xyxy": chosen["bbox"], "selected_plane_observed_pixels": chosen["observed_pixel_count"],
        "plane_rms_m": chosen["plane_rms_m"], "geometry_source": surface.source,
        "candidate_evidence": [{key: candidate[key] for key in (
            "candidate_id", "bbox", "sample_inlier_count", "observed_pixel_count", "plane_rms_m")}
            for candidate in candidates],
        "scope": "visible mask-bounded planar geometry; hidden area and physical stability are not established",
        **patch_diagnostics,
    }
    return surface, diagnostics
