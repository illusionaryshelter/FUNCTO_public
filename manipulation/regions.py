"""Consume source interaction regions and infer current functional geometry.

The known source interaction mask records the demonstrated local interaction.
Current owner masks bound visual search; current part masks are not inputs.
All rigid transforms use column vectors and metres. A full functional frame has
an explicit origin, signed operational x axis and function-plane z normal. A
plane recipe instead marks its tangent as a coordinate preference, so consumers
cannot mistake an unobserved functional angle for a hard geometric constraint.
Neither recipe estimates whole-object shape registration.
"""

# BEGIN local interaction-region geometry and functional-frame consumers.
from collections.abc import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .goals import SupportSurface, _points, place_on
from .grounding import _image, match_points, propose_query_regions
from .motion import _inverse, _pose_array
from .observations import _intrinsics, _mask, backproject, rigid_registration


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


def _context_crop(image, owner_mask, region_mask):
    """Keep owner context for descriptors while restricting their candidate ROI."""
    y, x = np.nonzero(owner_mask)
    padding = max(2, int(max(x.max()-x.min()+1, y.max()-y.min()+1) * .05))
    x0, y0 = max(0, int(x.min())-padding), max(0, int(y.min())-padding)
    x1 = min(image.width, int(x.max())+padding+1)
    y1 = min(image.height, int(y.max())+padding+1)
    return (image.crop((x0, y0, x1, y1)), region_mask[y0:y1, x0:x1],
            np.array([x0, y0]))


# BEGIN asymmetric source-functional-query to current-owner grounding.
def _owner_observation(record, name):
    """Validate observable owner data without requiring its functional answer."""
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} must be an owner observation mapping")
    required = {"image", "owner_mask", "depth", "K", "T_camera_owner", "owner_id"}
    if not required.issubset(record):
        raise ValueError(f"{name} is missing owner observation fields: {sorted(required-set(record))}")
    if not isinstance(record["owner_id"], str) or not record["owner_id"].strip():
        raise ValueError(f"{name}.owner_id must be a nonempty string")
    image = _image(record["image"])
    shape = (image.height, image.width)
    owner = _mask(record["owner_mask"], shape, f"{name}.owner_mask")
    depth = np.asarray(record["depth"], dtype=float)
    if depth.shape != shape:
        raise ValueError(f"{name} RGB, owner mask and registered depth dimensions must agree")
    _intrinsics(record["K"])
    if not np.any(owner & np.isfinite(depth) & (depth > 0)):
        raise ValueError(f"{name} owner has no valid depth")
    pose = _pose_array(record["T_camera_owner"], f"{name}.T_camera_owner")
    return image, owner, depth, pose


def _source_observation(demo):
    """Require source function evidence, including its geometric query roles.

    Function-direction witnesses may lie outside the true interaction patch,
    but must remain on the same selected source owner. The interaction mask is
    never expanded to make it encompass those witnesses. Small segmentation
    disagreements are reported instead of treating pixel containment as proof
    of owner identity; query ownership itself is checked by backprojection.
    """
    image, owner, depth, pose = _owner_observation(demo, "demo")
    required = {"mask", "region_id", "description", "query_pixels", "query_descriptions", "frame_recipe"}
    if not required.issubset(demo):
        raise ValueError(f"demo is missing source functional fields: {sorted(required-set(demo))}")
    for name in ("region_id", "description"):
        if not isinstance(demo[name], str) or not demo[name].strip():
            raise ValueError(f"demo.{name} must be a nonempty string")
    region = _mask(demo["mask"], (image.height, image.width), "demo interaction mask")
    if not np.any(region & owner):
        raise ValueError("Source interaction mask has no overlap with its selected owner")
    pixels = np.asarray(demo["query_pixels"])
    if (pixels.ndim != 2 or pixels.shape[1:] != (2,) or len(pixels) < 3
            or pixels.dtype.kind not in "iuf" or not np.isfinite(pixels).all()):
        raise ValueError("query_pixels must contain at least three finite numeric (u,v) queries")
    descriptions = demo["query_descriptions"]
    if (not isinstance(descriptions, (list, tuple)) or len(descriptions) != len(pixels)
            or any(not isinstance(value, str) or not value.strip() for value in descriptions)):
        raise ValueError("query_descriptions must explain every source query's geometric role")
    pixels = pixels.astype(float, copy=True)
    # Explicit queries retain their row identity; invalid depth and queries
    # outside the selected owner fail instead of being dropped or replaced.
    camera = backproject(depth, demo["K"], pixels, mask=owner)
    local = (camera-pose[:3, 3]) @ pose[:3, :3]
    return image, owner, region, depth, pose, pixels, local


def _recipe_frame(points, recipe):
    """Rebuild a function frame, independent of cross-instance point distances.

    For Q (N,3), o=Q.T@w, a=Q.T@axis_weights and b=Q.T@plane_weights.
    sum(w)=1 and both direction sums=0 make the construction SE(3)-equivariant.
    x=unit(a), z=unit(b cross x), y=z cross x. Different point distances and
    shapes are allowed. Zero or parallel directions cannot produce a frame.

    In ``plane`` mode the same arithmetic supplies a reproducible plane basis,
    but its x direction is merely a chosen tangent, never an observed functional
    angle. Its ordered normal is not automatically a physical outward normal;
    the source recipe must describe its sign evidence for the relevant caller.
    """
    if not isinstance(recipe, Mapping):
        raise ValueError("frame_recipe must be an explicit geometric recipe mapping")
    required = {"kind", "origin_weights", "axis_weights", "plane_weights", "source"}
    if not required.issubset(recipe):
        raise ValueError(f"frame_recipe is missing fields: {sorted(required-set(recipe))}")
    if recipe["kind"] not in ("frame", "plane"):
        raise ValueError("frame_recipe.kind must be 'frame' or 'plane'")
    if not isinstance(recipe["source"], str) or not recipe["source"].strip():
        raise ValueError("frame_recipe.source must explain functional geometry and normal-sign evidence")
    geometry = _points(points, "functional query points", minimum=3)
    weights = {}
    for key, expected_sum in (("origin_weights", 1.), ("axis_weights", 0.), ("plane_weights", 0.)):
        values = np.asarray(recipe[key])
        if (values.shape != (len(geometry),) or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all()):
            raise ValueError(f"frame_recipe.{key} must be a finite numeric weight for every query")
        values = values.astype(float, copy=True)
        if not np.isclose(values.sum(), expected_sum, atol=1e-9, rtol=0):
            raise ValueError(f"frame_recipe.{key} must sum to {expected_sum:g}")
        weights[key] = values
    origin = geometry.T @ weights["origin_weights"]
    a, b = geometry.T @ weights["axis_weights"], geometry.T @ weights["plane_weights"]
    x, hint = _unit(a, "recipe axis"), _unit(b, "recipe plane hint")
    cross = np.cross(hint, x)
    independence = float(np.linalg.norm(cross))
    if independence <= 1e-6:
        raise ValueError("Functional recipe directions are collinear or insufficiently independent")
    frame = function_frame(origin, x, cross / independence)
    provenance = ("observed functional query direction" if recipe["kind"] == "frame"
                  else "chosen ordered-sample tangent preference; not an observed functional angle")
    normalized = {"kind": recipe["kind"], **weights, "source": recipe["source"]}
    return frame, {"recipe": normalized,
                   "axis_length_m": float(np.linalg.norm(a)),
                   "plane_hint_length_m": float(np.linalg.norm(b)),
                   "direction_independence_sine": independence,
                   "x_axis_source": provenance,
                   "normal_source": "ordered recipe directions; physical outward sign requires declared source evidence",
                   "shape_assumption": "functional query correspondence; no local or whole-owner rigid congruence"}


def source_function_geometry(demo):
    """Validate and lift compiled source functional queries without inference.

    Source includes the true interaction mask, owner context, explicit queries,
    per-query descriptions and a geometry recipe. It need not contain fixed
    function/center/grasp landmarks. This computes the recipe's source frame;
    a program binder separately checks agreement with its compiled event frame.
    No visual model, current scene, API request or point matching is involved.
    """
    _, owner, region, _, _, pixels, points = _source_observation(demo)
    frame, diagnostics = _recipe_frame(points, demo["frame_recipe"])
    rows = np.rint(pixels).astype(int)
    in_interaction = (region & owner)[rows[:, 1], rows[:, 0]]
    normal_role = demo.get("normal_role")
    if normal_role is not None and (demo["frame_recipe"]["kind"] != "plane"
                                    or normal_role not in ("outward", "support")):
        raise ValueError("normal_role must be 'outward' or 'support' on a plane recipe")
    diagnostics.update(source_pixels=pixels,
                       query_descriptions=list(demo["query_descriptions"]),
                       region_id=demo["region_id"], owner_id=demo["owner_id"],
                       description=demo["description"],
                       interaction_query_indices=np.flatnonzero(in_interaction),
                       queries_outside_interaction_mask=np.flatnonzero(~in_interaction),
                       normal_role=normal_role,
                       interaction_evidence_scope="matched visible query samples; not a recovered complete contact face",
                       interaction_pixels_outside_owner=int(np.count_nonzero(region & ~owner)))
    return {"source_frame": frame, "geometry_kind": demo["frame_recipe"]["kind"],
            "normal_role": normal_role,
            "source_points": points, "source_origin": frame[:3, 3].copy(),
            "source_normal": frame[:3, 2].copy(), "diagnostics": diagnostics}


def match_functional_region(demo, current, *, client=None, extractor=None):
    """Infer current functional geometry from source evidence and owner RGB-D.

    Current input contains only image, owner_mask, depth, K, T_camera_owner and
    owner_id. It never supplies a part mask, functional frame or target queries.
    The VLM chooses coarse owner-local candidate cells from the source example;
    the existing dense matcher searches each proposed box intersected with the
    current owner and valid depth. Those query correspondences reconstruct the
    current functional frame under the same source recipe. Unlike rigid fitting,
    no cross-instance pairwise-distance equality is assumed.

    Owner-context crops are fixed for all queries. One descriptor pair is
    extracted and reused; each matcher invocation retains its bounded sampling.
    Repeated target pixels are not arbitrarily reassigned: recipe degeneracy,
    together with reported correspondence ambiguity, exposes missing geometry.
    A geometric frame candidate is not proof of semantic or physical success.
    """
    source = source_function_geometry(demo)
    first, owner1, region, depth1, _, pixels, _ = _source_observation(demo)
    second, owner2, depth2, pose2 = _owner_observation(current, "current")
    first_crop, first_owner, first_offset = _context_crop(first, owner1, owner1)
    second_crop, second_owner, second_offset = _context_crop(second, owner2, owner2)
    x, y = first_offset
    first_region = region[y:y+first_crop.height, x:x+first_crop.width]
    proposals = propose_query_regions(
        first_crop, second_crop, first_owner, second_owner, first_region,
        pixels-first_offset, description=demo["description"],
        query_descriptions=demo["query_descriptions"], client=client)
    boxes = np.asarray(proposals["boxes"])
    if (boxes.shape != (len(pixels), 4) or boxes.dtype.kind not in "iu"
            or np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] <= boxes[:, :2])
            or np.any(boxes[:, 2:] > [second_crop.width, second_crop.height])):
        raise ValueError("Coarse query boxes must be valid integer xyxy boxes for every query")
    x1, y1 = first_offset
    x2, y2 = second_offset
    first_valid = first_owner & np.isfinite(depth1[y1:y1+first_crop.height, x1:x1+first_crop.width])
    first_valid &= depth1[y1:y1+first_crop.height, x1:x1+first_crop.width] > 0
    second_depth = depth2[y2:y2+second_crop.height, x2:x2+second_crop.width]
    second_valid = second_owner & np.isfinite(second_depth) & (second_depth > 0)
    for index, (left, top, right, bottom) in enumerate(boxes):
        if not second_valid[top:bottom, left:right].any():
            raise ValueError(f"Coarse region for query {index} has no valid current owner depth")
    pair_cache = None

    def cached_pair(first_image, second_image):
        nonlocal pair_cache
        if pair_cache is None:
            actual = extractor
            if actual is None:
                from local_models.sd_dino import extract_pair
                actual = extract_pair
            pair_cache = actual(first_image, second_image)
        return pair_cache

    targets, matching = [], []
    for index, (left, top, right, bottom) in enumerate(boxes):
        candidates = np.zeros_like(second_valid)
        candidates[top:bottom, left:right] = second_valid[top:bottom, left:right]
        found, diagnostic = match_points(first_crop, second_crop, first_valid, candidates,
                                         (pixels[index:index+1]-first_offset), extractor=cached_pair)
        if not np.isfinite(diagnostic["scores"][0]) or diagnostic["scores"][0] <= 0:
            raise ValueError(f"Functional query {index} has no positive-score descriptor candidate")
        targets.append(found[0]+second_offset)
        matching.append(diagnostic)
    target_pixels = np.asarray(targets)
    camera_points = backproject(depth2, current["K"], target_pixels, mask=owner2)
    target_points = (camera_points-pose2[:3, 3]) @ pose2[:3, :3]
    target_frame, target_diagnostics = _recipe_frame(target_points, demo["frame_recipe"])
    diagnostics = dict(source["diagnostics"])
    # Per-query coarse windows require separate matcher calls. Retain duplicate
    # identities across those calls as evidence instead of inventing uniqueness.
    duplicate_groups = [np.flatnonzero(np.all(target_pixels == pixel, axis=1)).tolist()
                        for pixel in np.unique(target_pixels, axis=0)
                        if np.count_nonzero(np.all(target_pixels == pixel, axis=1)) > 1]
    diagnostics.update(target_geometry=target_diagnostics,
                       target_pixels=target_pixels, target_owner_id=current["owner_id"],
                       matching=matching, coarse_regions=proposals["diagnostics"],
                       coarse_boxes_xyxy=boxes,
                       crop_offsets_xy=[first_offset, second_offset],
                       descriptor_pair_extractions=1,
                       duplicate_target_groups=duplicate_groups,
                       matching_scope="source functional queries to coarse regions inside current owner",
                       transform_scope="functional-frame correspondence, not whole-owner rigid registration")
    return {**source, "target_frame": target_frame,
            "transform": target_frame @ _inverse(source["source_frame"]),
            "target_points": target_points, "target_origin": target_frame[:3, 3].copy(),
            "target_normal": target_frame[:3, 2].copy(), "diagnostics": diagnostics}
# END asymmetric source-functional-query grounding.
# END local interaction-region geometry and functional-frame consumers.
