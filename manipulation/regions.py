"""Consume interaction regions without requiring three named semantic points.

Masks select visible metric evidence; owner masks only provide image context.
All rigid transforms use column vectors and metres. A functional frame has an
explicit origin, signed operational x axis and signed function-plane z normal.
Its axes describe the supplied functional geometry, not an inferred whole-object
shape registration. Partial relations must keep their unobserved freedoms outside
the full-frame API instead of silently inventing an axis.
"""

# BEGIN local interaction-region geometry and functional-frame consumers.
from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .goals import SupportSurface, _points, place_on
from .grounding import _image, match_points
from .motion import _inverse, _pose_array
from .observations import _mask, backproject, rigid_registration


def _vector(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return vector.copy()


def _unit(value, name):
    vector = _vector(value, name)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError(f"{name} must have nonzero finite length")
    return vector / length


def function_frame(origin, axis, plane_normal):
    """Construct an owner-local SE(3) frame from two observed signed directions.

    x follows the operational axis, z the function-plane normal, and y=z cross x.
    Axis and normal must be perpendicular; only floating-point residuals below
    1e-6 are removed. A missing/parallel/unsigned direction is not completed from
    a world axis. Region centroids or PCA axes are not introduced by this API.

    This follows MimicFunc A.2.2's center-to-function direction. The released
    ``compute_function_plane_frame`` uses the opposite x and y, an equivalent
    fixed local z-pi gauge only when all associated poses use it consistently.
    """
    point = _vector(origin, "origin")
    x, z = _unit(axis, "axis"), _unit(plane_normal, "plane_normal")
    cosine = float(x @ z)
    if abs(cosine) > 1e-6:
        raise ValueError("axis and plane_normal must be perpendicular, independently supported directions")
    z = _unit(z - cosine * x, "orthogonal plane_normal")
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    pose[:3, 3] = point
    return pose


def align_axis(preferred_rotation, local_axis, target_axis, *, antiparallel_axis=None):
    """Choose the rotation nearest a preference that maps one signed axis.

    The remaining twist is selected from ``preferred_rotation``, not measured
    from the single axis. Return one numerical rotation candidate; callers must
    retain that distinction when stating task requirements. For opposite axes,
    all perpendicular half-turn axes tie. The caller must supply an explicit
    geometric ``antiparallel_axis`` in the target/reference frame to break that
    ambiguity; no fixed world axis is substituted.
    """
    preferred = np.asarray(preferred_rotation, dtype=float)
    if preferred.shape != (3, 3):
        raise ValueError("preferred_rotation must have shape (3, 3)")
    frame = np.eye(4)
    frame[:3, :3] = preferred
    preferred = _pose_array(frame, "preferred_rotation")[:3, :3]
    source = preferred @ _unit(local_axis, "local_axis")
    target = _unit(target_axis, "target_axis")
    cross = np.cross(source, target)
    sine, cosine = float(np.linalg.norm(cross)), float(np.clip(source @ target, -1, 1))
    if sine > 1e-12:
        correction = Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()
    elif cosine > 0:
        correction = np.eye(3)
    else:
        if antiparallel_axis is None:
            raise ValueError("Opposite axes need an explicit geometric antiparallel_axis")
        branch = _unit(antiparallel_axis, "antiparallel_axis")
        if abs(float(branch @ source)) > 1e-6:
            raise ValueError("antiparallel_axis must be perpendicular to the aligned axes")
        branch = _unit(branch - (branch @ source) * source, "antiparallel_axis")
        correction = Rotation.from_rotvec(np.pi * branch).as_matrix()
    result = correction @ preferred
    if not np.allclose(result @ _unit(local_axis, "local_axis"), target, atol=1e-9, rtol=0):
        raise ValueError("Axis alignment did not preserve the requested signed direction")
    return result


def ground_relation(demo_pose, demo_carrier_frame, scene_carrier_frame,
                    demo_reference_frame, scene_reference_frame):
    """Preserve an explicit functional relation across different owner shapes.

    ``demo_pose`` is T_Rd_Cd. Each functional frame F is T_owner_function in
    that owner's *fixed* local coordinates, not an observation-camera pose.
    Let S=inv(F_Rd) D_d F_Cd. The event goal is D_n=F_Rn S inv(F_Cn).
    This is equivalent to A_R D_d inv(A_C), with A_X=F_Xn inv(F_Xd).
    No congruence of the complete objects or their region point sets is needed.

    The returned pose is a functional-event target, not a freely normalized
    trajectory reference: normalizing it by its initial pose would cancel A_C.
    A complete frame target must not silently replace a partially observed
    plane/axis relation; such callers first choose and label their free pose
    components and must validate the actual partial relation independently.
    """
    demo = _pose_array(demo_pose, "demo_pose")
    cd = _pose_array(demo_carrier_frame, "demo_carrier_frame")
    cn = _pose_array(scene_carrier_frame, "scene_carrier_frame")
    rd = _pose_array(demo_reference_frame, "demo_reference_frame")
    rn = _pose_array(scene_reference_frame, "scene_reference_frame")
    functional = _inverse(rd) @ demo @ cd
    return {"goal_pose": rn @ functional @ _inverse(cn),
            "functional_relation": functional,
            "carrier_binding": cn @ _inverse(cd),
            "reference_binding": rn @ _inverse(rd)}


def relation_error(pose, carrier_frame, reference_frame, functional_relation):
    """Independently check function-frame position and SO(3) angle residuals.

    Errors are metric diagnostics, with no built-in task acceptance threshold.
    They do not establish physical contact, grasp stability or task success.
    """
    body = _pose_array(pose, "pose")
    carrier = _pose_array(carrier_frame, "carrier_frame")
    reference = _pose_array(reference_frame, "reference_frame")
    expected = _pose_array(functional_relation, "functional_relation")
    actual = _inverse(reference) @ body @ carrier
    return {"position_error_m": float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])),
            "rotation_error_rad": float(Rotation.from_matrix(
                actual[:3, :3] @ expected[:3, :3].T).magnitude())}


# BEGIN consume partial support relations without inventing a full functional frame.
def _support_arguments(support, margin, gap_tolerance, normal_tolerance):
    if not isinstance(support, SupportSurface):
        raise ValueError("support must be an observed or supplied finite SupportSurface")
    values = (margin, gap_tolerance, normal_tolerance)
    if any(not np.isscalar(value) or not np.isfinite(value) for value in values):
        raise ValueError("Support margin and tolerances must be finite scalars")
    if margin < 0 or gap_tolerance < 0 or not 0 <= normal_tolerance < np.pi:
        raise ValueError("Support margin/gap tolerance must be nonnegative; normal tolerance must be in [0, pi)")


def check_region_on(pose, carrier_region_points, carrier_outward_normal, support,
                    *, gap_tolerance, normal_tolerance, margin=0.0):
    """Check the specifically declared contact face against a finite support.

    Every supplied owner-local contact-face sample must be within the support
    boundary and plane gap tolerance. Its signed outward normal must oppose the
    support normal. Unlike ``check_on``, a different, lower part of the carrier
    cannot satisfy this contact-face check. No convex hull is fabricated for the
    carrier region and no missing support geometry is completed.

    This validates supplied part geometry, not the semantic part name: if a
    caller labels the rim as the bottom, geometry cannot recover that label.
    Return diagnostics plus ``valid``; invalid numerical inputs raise, whereas
    a valid but unsatisfied geometric relation returns ``valid=False``.
    """
    _support_arguments(support, margin, gap_tolerance, normal_tolerance)
    body = _pose_array(pose, "pose")
    region = _points(carrier_region_points, "carrier_region_points", minimum=1)
    outward = body[:3, :3] @ _unit(carrier_outward_normal, "carrier_outward_normal")
    points = region @ body[:3, :3].T + body[:3, 3]
    local = points-support.origin
    gaps = local @ support.normal
    boundary = support._hull.equations
    signed_boundary = (local @ support.basis) @ boundary[:, :2].T + boundary[:, 2]
    violation = float(max(0., signed_boundary.max()+margin))
    normal_error = float(np.arccos(np.clip(outward @ -support.normal, -1., 1.)))
    gap_error = float(np.max(np.abs(gaps)))
    return {"valid": bool(gap_error <= gap_tolerance + 1e-10
                           and normal_error <= normal_tolerance + 1e-10
                           and violation <= 1e-7),
            "max_abs_gap_m": gap_error,
            "minimum_signed_gap_m": float(gaps.min()),
            "maximum_signed_gap_m": float(gaps.max()),
            "boundary_violation_m": violation,
            "normal_error_rad": normal_error,
            "gap_tolerance_m": float(gap_tolerance),
            "normal_tolerance_rad": float(normal_tolerance),
            "margin_m": float(margin),
            "checked_points": len(region), "geometry_source": support.source}


def ground_on(preferred_pose, body_points, carrier_region_points,
              carrier_outward_normal, support, *, margin=0.0,
              gap_tolerance, normal_tolerance, antiparallel_axis=None):
    """Instantiate a partial face-on-support relation using existing geometry.

    First rotate the specified carrier outward normal opposite the support
    normal, selecting the closest twist from the explicit pose preference.
    Then ``place_on`` moves the *whole supplied body geometry* into the finite
    support. Finally verify the declared functional contact face independently;
    the body's lowest point alone cannot make this API report success.

    The chosen twist and tangential location are numerical preferences, not
    measured task requirements. Their feasible range is bounded by the supplied
    support and body geometry. The output certifies this sampled geometric
    relation only, not hidden-shape coverage, force stability or collision-free
    robot execution. A missing antiparallel branch is rejected by ``align_axis``.
    """
    _support_arguments(support, margin, gap_tolerance, normal_tolerance)
    preferred = _pose_array(preferred_pose, "preferred_pose")
    preferred[:3, :3] = align_axis(preferred[:3, :3], carrier_outward_normal,
                                  -support.normal, antiparallel_axis=antiparallel_axis)
    goal, body_check = place_on(preferred, body_points, support, margin=margin)
    region_check = check_region_on(goal, carrier_region_points, carrier_outward_normal,
                                   support, gap_tolerance=gap_tolerance,
                                   normal_tolerance=normal_tolerance, margin=margin)
    if not region_check["valid"]:
        raise ValueError("Specified functional contact face does not satisfy support: "
                         f"gap={region_check['max_abs_gap_m']:.6g} m, "
                         f"normal={region_check['normal_error_rad']:.6g} rad, "
                         f"boundary={region_check['boundary_violation_m']:.6g} m")
    # Preserve the existing placement kernel's independent full-body check;
    # the small contact patch must not replace complete geometry coverage.
    return {"goal_pose": goal, "checks": {"body": body_check, "region": region_check},
            "freedom": {
                "required": "specified face on finite support with opposing signed normals",
                "twist": "closest preferred orientation; chosen free component, not observed",
                "tangential_position": "nearest preferred position satisfying supplied full-body bounds",
                "antiparallel_branch": "explicit geometric axis" if antiparallel_axis is not None else None}}
# END partial support-region consumption.


def bind_region(source_points, target_points, *, max_error=0.01):
    """Fit corresponding owner-local surface samples, preserving fit evidence.

    Source/target rows must already represent paired semantic-region locations.
    This optional estimator is useful for locally congruent geometry; it is not
    a requirement that functionally equivalent objects have congruent shapes.
    The existing robust rigid estimator rejects collinear and nonrigid consensus.
    A planar, noncollinear correspondence remains a valid source of full SE(3).
    """
    transform, diagnostics = rigid_registration(source_points, target_points, max_error=max_error)
    return {"transform": transform, "diagnostics": diagnostics}


def bind_observations(mask, depth, K, T_camera_owner):
    """Lift visible region pixels into their fixed owner frame, without mutation.

    Depth is registered metric camera-z depth. ``T_camera_owner`` is that
    observation's calibrated camera-from-owner pose; it need not be identity.
    Nonpositive/nonfinite depth contributes no surface point. An empty valid
    region fails rather than becoming a whole-owner observation. Hidden geometry
    is not reconstructed or filled. Returned pixels use original-image (u,v).
    """
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("depth must be a nonempty two-dimensional metric array")
    region = _mask(mask, depth.shape, "region mask")
    pose = _pose_array(T_camera_owner, "T_camera_owner")
    valid = region & np.isfinite(depth) & (depth > 0)
    rows, cols = np.nonzero(valid)
    if not len(rows):
        raise ValueError("Region contains no valid metric depth observations")
    pixels = np.column_stack((cols, rows)).astype(float)
    camera = backproject(depth, K, pixels, mask=region)
    local = (camera - pose[:3, 3]) @ pose[:3, :3]
    return {"owner_local_points": local, "pixels": pixels, "camera_points": camera}


def sample_region_pixels(mask, depth, count=24):
    """Choose bounded, dispersed source queries strictly inside a visible ROI.

    Erode a valid-depth ROI by one pixel when enough points remain, reducing
    boundary-depth contamination. Farthest-point sampling considers at most
    6,000 candidate pixels and stores no dense distance matrix. This controls
    descriptor workload; it neither invents a center semantic point nor expands
    a failed local mask to an owner mask. Returns ``(pixels_uv, valid_mask)``.
    """
    import cv2

    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("depth must be a nonempty two-dimensional metric array")
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 3:
        raise ValueError("count must be an integer of at least three")
    region = _mask(mask, depth.shape, "region mask")
    valid = region & np.isfinite(depth) & (depth > 0)
    interior = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if interior.sum() >= count:
        valid = interior
    y, x = np.nonzero(valid)
    if len(x) < 3:
        raise ValueError("Region has fewer than three valid depth pixels")
    candidates = np.column_stack((x, y)).astype(float)
    if len(candidates) > 6000:
        candidates = candidates[np.linspace(0, len(candidates)-1, 6000).astype(int)]
    selected = [int(np.argmin(np.sum((candidates-candidates.mean(0))**2, axis=1)))]
    distances = np.full(len(candidates), np.inf)
    while len(selected) < min(count, len(candidates)):
        distances = np.minimum(distances, np.sum((candidates-candidates[selected[-1]])**2, axis=1))
        index = int(np.argmax(distances))
        if distances[index] <= 0:
            break
        selected.append(index)
    return candidates[selected], valid.copy()


def _observation(record, name):
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} must be a region observation mapping")
    required = {"image", "mask", "owner_mask", "depth", "K", "T_camera_owner", "region_id", "owner_id"}
    if not required.issubset(record):
        raise ValueError(f"{name} is missing region observation fields: {sorted(required-set(record))}")
    for key in ("region_id", "owner_id"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{name}.{key} must be a nonempty string")
    image = _image(record["image"])
    shape = (image.height, image.width)
    region = _mask(record["mask"], shape, f"{name}.mask")
    owner = _mask(record["owner_mask"], shape, f"{name}.owner_mask")
    if np.any(region & ~owner):
        raise ValueError(f"{name} region mask extends outside its declared owner; resolve ownership before matching")
    depth = np.asarray(record["depth"], dtype=float)
    if depth.shape != shape:
        raise ValueError(f"{name} RGB, mask and registered depth dimensions must agree")
    pose = _pose_array(record["T_camera_owner"], f"{name}.T_camera_owner")
    return image, region, owner, depth, pose


def _context_crop(image, owner_mask, region_mask):
    """Keep owner context for descriptors while restricting their candidate ROI."""
    y, x = np.nonzero(owner_mask)
    padding = max(2, int(max(x.max()-x.min()+1, y.max()-y.min()+1) * .05))
    x0, y0 = max(0, int(x.min())-padding), max(0, int(y.min())-padding)
    x1 = min(image.width, int(x.max())+padding+1)
    y1 = min(image.height, int(y.max())+padding+1)
    return (image.crop((x0, y0, x1, y1)), region_mask[y0:y1, x0:x1],
            np.array([x0, y0]))


def match_regions(demo, scene, *, extractor=None, max_error=0.01):
    """Bind paired visible regions using owner context, masked matching and depth.

    Each mapping supplies ``image, mask, owner_mask, depth, K, T_camera_owner,
    region_id, owner_id``. Owner IDs identify the separately selected instances;
    this numeric API does not prove their semantic identity. The caller supplies
    a semantic pairing, and a region outside its declared owner is rejected.

    Existing SD/DINO correspondence is used unchanged. Only source-region pixels
    are sampled; only target-region pixels with valid depth are candidates. All
    matched points are transformed to the associated owner-local coordinates
    before rigid fitting. No GT mask, whole-owner fallback, mask intersection
    repair, or fixed function/center/grasp keypoint is introduced.
    """
    first, mask1, owner1, depth1, pose1 = _observation(demo, "demo")
    second, mask2, owner2, depth2, pose2 = _observation(scene, "scene")
    pixels, valid1 = sample_region_pixels(mask1, depth1)
    valid2 = mask2 & np.isfinite(depth2) & (depth2 > 0)
    if valid2.sum() < 3:
        raise ValueError("Scene region has fewer than three valid depth pixels")
    first_crop, first_mask, first_offset = _context_crop(first, owner1, valid1)
    second_crop, second_mask, second_offset = _context_crop(second, owner2, valid2)
    targets, matching = match_points(first_crop, second_crop, first_mask, second_mask,
                                     pixels-first_offset, extractor=extractor)
    targets = targets + second_offset
    scores = np.asarray(matching["scores"], dtype=float)
    selected, seen = [], set()
    for index in np.argsort(-scores, kind="stable"):
        key = tuple(np.rint(targets[index]).astype(int))
        # Preserve nearest-match diagnostics instead of assigning a convenient
        # unused pixel. Rigid consensus judges metric compatibility; descriptor
        # ties remain visible and are not a semantic-confidence certificate.
        if scores[index] <= 0 or key in seen:
            continue
        selected.append(int(index))
        seen.add(key)
    if len(selected) < 3:
        raise ValueError("Region correspondences have fewer than three distinct positive-score targets")
    source_pixels, target_pixels = pixels[selected], targets[selected]
    camera1 = backproject(depth1, demo["K"], source_pixels, mask=valid1)
    camera2 = backproject(depth2, scene["K"], target_pixels, mask=valid2)
    local1 = (camera1-pose1[:3, 3]) @ pose1[:3, :3]
    local2 = (camera2-pose2[:3, 3]) @ pose2[:3, :3]
    binding = bind_region(local1, local2, max_error=max_error)
    return {"transform": binding["transform"],
            "source_region_id": demo["region_id"], "target_region_id": scene["region_id"],
            "source_owner_id": demo["owner_id"], "target_owner_id": scene["owner_id"],
            "diagnostics": {"source_pixels": source_pixels, "target_pixels": target_pixels,
                            "source_local_points": local1, "target_local_points": local2,
                            "selected_query_indices": selected, "matching": matching,
                            "rigid_fit": binding["diagnostics"],
                            "crop_offsets_xy": [first_offset, second_offset],
                            "source": "paired semantic-region correspondences and calibrated metric RGB-D"}}
# END local interaction-region geometry and functional-frame consumers.
