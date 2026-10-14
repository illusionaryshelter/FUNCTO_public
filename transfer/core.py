"""Independent CPU experiment for transferring one visible planar convex patch.

The module consumes point correspondences, never images, object names, target
region labels, simulation geometry, or a robot state.  An ``accepted`` result
means the experiment's geometric checks passed; it is not a placement-safety
certificate.  Rejected candidates remain available only for diagnosis.

All geometry is in metres.  Source and target coordinates may use independent
frames.  Correspondence indices must be one-to-one; the matcher owns duplicate
resolution so repeated visual predictions cannot silently acquire extra weight.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.transform import Rotation


# BEGIN independent visible-region transfer experiment; no production imports.
@dataclass(frozen=True)
class TransferConfig:
    """One frozen, dimensionless set of experiment thresholds for every object.

    Relation scores and mode separation use median matched-anchor distance.
    RANSAC, surface neighborhoods, plane thickness and visible coverage use the
    whole observed object's RMS radius, shared even by the local-direct method.
    These are exploratory gates, not calibrated probabilities.
    """

    seed: int = 0
    ransac_iterations: int = 1500
    ransac_tolerance_ratio: float = 0.075
    min_correspondences: int = 6
    min_inlier_ratio: float = 0.2
    max_anchors: int = 32
    rank_relative_tolerance: float = 1e-6
    trim_fraction: float = 0.2
    alignment_weight: float = 0.12
    normal_weight: float = 0.035
    planarity_weight: float = 0.025
    max_relation_residual: float = 0.10
    mode_exclusion_ratio: float = 0.18
    min_mode_gap: float = 0.01
    normal_min_agreement: float = 0.65
    max_planarity_ratio: float = 0.15
    neighborhood_count: int = 35
    min_neighborhood_points: int = 6
    neighborhood_radius_ratio: float = 0.18
    min_region_points: int = 12
    plane_tolerance_ratio: float = 0.03
    coverage_distance_ratio: float = 0.06
    min_observed_fraction: float = 0.4
    max_source_hull_exclusion_fraction: float = 0.05
    max_projection_spacing_factor: float = 2.0
    max_projection_region_ratio: float = 0.5
    chunk_size: int = 2048

    def __post_init__(self):
        integer_names = ("seed", "ransac_iterations", "min_correspondences",
                         "max_anchors", "neighborhood_count",
                         "min_neighborhood_points", "min_region_points", "chunk_size")
        for name in integer_names:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if value < (0 if name == "seed" else 1):
                raise ValueError(f"Invalid {name}")
        if min(self.min_correspondences, self.max_anchors, self.min_neighborhood_points,
               self.min_region_points) < 3:
            raise ValueError("Geometry needs at least three points")
        if self.neighborhood_count < self.min_neighborhood_points:
            raise ValueError("neighborhood_count is smaller than its minimum support")
        for name, value in asdict(self).items():
            if name not in integer_names and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.trim_fraction < 1:
            raise ValueError("trim_fraction must lie in [0, 1)")
        for name in ("min_inlier_ratio", "normal_min_agreement", "min_observed_fraction",
                     "max_source_hull_exclusion_fraction"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("rank_relative_tolerance", "ransac_tolerance_ratio",
                     "neighborhood_radius_ratio", "mode_exclusion_ratio",
                     "plane_tolerance_ratio", "coverage_distance_ratio"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PreparedAnchors:
    """Shared registration for the global and two relational comparisons.

    Arrays reference the validated caller data where possible.  Callers must
    not mutate these arrays between methods.  Oracle preparation is explicit
    and cannot be passed off as automatic preparation.
    """

    src_xyz: np.ndarray
    tgt_xyz: np.ndarray
    src_anchor_idx: np.ndarray
    tgt_anchor_idx: np.ndarray
    input_src_anchor_idx: np.ndarray
    scale: object
    rotation: object
    translation: object
    diagnostics: dict
    reasons: tuple
    mode: str
    config: TransferConfig


def _points(value, name):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric [N,3] array")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite coordinates")
    return result


def _indices(value, size, name):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(raw < 0) or np.any(raw >= size):
        raise ValueError(f"{name} contains an out-of-range index")
    if len(np.unique(raw)) != len(raw):
        raise ValueError(f"{name} must contain unique indices")
    return np.asarray(raw, dtype=np.int64)


def _mask(value, size):
    raw = np.asarray(value)
    if raw.dtype != np.bool_ or raw.shape != (size,):
        raise ValueError("src_region_mask must be a boolean array with one entry per source point")
    return raw


def rms_size(points):
    """Rotation-equivariant RMS radius, independent of coordinate origin."""
    return float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))


def _rank(points, tolerance=1e-6):
    if len(points) < 2:
        return 0, 0.0
    singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
    if not len(singular) or singular[0] <= np.finfo(float).tiny:
        return 0, 0.0
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    condition = float(singular[-1] / singular[0]) if len(singular) == 3 else 0.0
    return rank, condition


def fps_indices(points, count):
    """Deterministic spatial sampling, stopping when unique geometry is exhausted.

    Excluding already selected indices is essential for duplicate-coordinate
    observations.  Choosing further copies of the same location would fabricate
    anchor count without adding any localization information.
    """
    points = _points(points, "points")
    count = min(int(count), len(points))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    distance = np.sum((points - points.mean(0)) ** 2, axis=1)
    current = int(np.argmax(distance))
    closest = np.full(len(points), np.inf)
    selected = np.zeros(len(points), dtype=bool)
    chosen = []
    tolerance = np.finfo(float).eps ** 2 * max(float(distance.max()), np.finfo(float).tiny)
    for _ in range(count):
        chosen.append(current)
        selected[current] = True
        closest = np.minimum(closest, np.sum((points - points[current]) ** 2, axis=1))
        closest[selected] = -np.inf
        current = int(np.argmax(closest))
        if closest[current] <= tolerance:
            break
    return np.asarray(chosen, dtype=np.int64)


def fit_sim3(src, tgt, rank_relative_tolerance=1e-6):
    """Umeyama fit for ``target = scale * source @ rotation.T + translation``.

    A rank-two triangle identifies a proper similarity transform.  This is
    distinct from uniqueness of arbitrary 3-D trilateration, which needs rank
    three or extra evidence from the visible candidate surface.
    """
    src, tgt = _points(src, "src"), _points(tgt, "tgt")
    if src.shape != tgt.shape or len(src) < 3:
        raise ValueError("Sim3 requires at least three matching 3-D points")
    if min(_rank(src, rank_relative_tolerance)[0], _rank(tgt, rank_relative_tolerance)[0]) < 2:
        raise ValueError("Sim3 anchors are coincident or collinear")
    source_mean, target_mean = src.mean(0), tgt.mean(0)
    x, y = src - source_mean, tgt - target_mean
    covariance = y.T @ x / len(x)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    correction[-1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = (u * correction) @ vt
    variance = float(np.mean(np.sum(x * x, axis=1)))
    scale = float(singular @ correction / variance)
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        raise ValueError("Sim3 must have a finite positive scale")
    translation = target_mean - scale * (rotation @ source_mean)
    if not np.isfinite(translation).all():
        raise ValueError("Sim3 translation is nonfinite")
    return scale, rotation, translation


def apply_sim3(points, scale, rotation, translation):
    return scale * (np.asarray(points) @ rotation.T) + translation


def prepare_anchors(src_xyz, tgt_xyz, src_anchor_idx, tgt_anchor_idx, *,
                    mode="automatic", config=None):
    """Validate and freeze the registration reused across comparison methods.

    ``automatic`` and ``direct`` run exactly the same RANSAC.  They differ only
    in the supplied correspondence set: all-object versus source-region-only.
    ``oracle`` fits the supplied known pairs without a matching/consensus gate.
    Invalid array contracts raise ValueError; insufficient geometry is recorded
    as a rejected preparation so batch experiments can keep negative results.
    """
    config = TransferConfig() if config is None else config
    if not isinstance(config, TransferConfig):
        raise ValueError("config must be TransferConfig")
    if mode not in {"automatic", "direct", "oracle"}:
        raise ValueError("Unknown anchor preparation mode")
    ps, pt = _points(src_xyz, "src_xyz"), _points(tgt_xyz, "tgt_xyz")
    ids = _indices(src_anchor_idx, len(ps), "src_anchor_idx")
    idt = _indices(tgt_anchor_idx, len(pt), "tgt_anchor_idx")
    if len(ids) != len(idt):
        raise ValueError("Source and target correspondence counts differ")
    stats = {"anchor_mode": mode, "input_correspondences": len(ids),
             "configuration": asdict(config)}

    def failed(reason):
        return PreparedAnchors(ps, pt, ids[:0], idt[:0], ids, None, None, None,
                               stats, (reason,), mode, config)

    required = 3 if mode == "oracle" else config.min_correspondences
    if len(ps) < 3 or len(pt) < 3 or len(ids) < required:
        return failed("insufficient_correspondences")
    source_radius, target_radius = rms_size(ps), rms_size(pt)
    if not np.isfinite([source_radius, target_radius]).all() or min(source_radius, target_radius) <= np.finfo(float).tiny:
        return failed("degenerate_anchors")
    a, b = ps[ids], pt[idt]
    tolerance = config.ransac_tolerance_ratio * target_radius
    stats.update(source_rms_radius=source_radius, target_rms_radius=target_radius,
                 ransac_tolerance_m=tolerance)
    if mode == "oracle":
        keep = np.ones(len(ids), dtype=bool)
    else:
        rng = np.random.default_rng(config.seed)
        keep = None
        quality = (-1, -np.inf)
        for _ in range(config.ransac_iterations):
            sample = rng.choice(len(ids), 3, replace=False)
            try:
                scale, rotation, translation = fit_sim3(a[sample], b[sample],
                                                        config.rank_relative_tolerance)
            except (ValueError, np.linalg.LinAlgError):
                continue
            ratio = scale / (target_radius / source_radius)
            if not 0.2 <= ratio <= 5.0:
                continue
            residual = np.linalg.norm(apply_sim3(a, scale, rotation, translation) - b, axis=1)
            inlier = residual <= tolerance
            count = int(inlier.sum())
            candidate_quality = (count, -float(np.median(residual[inlier])) if count else -np.inf)
            if candidate_quality > quality:
                quality, keep = candidate_quality, inlier
        if keep is None or keep.sum() < required:
            return failed("weak_correspondence_consensus")
    # Refit, then recompute residuals against the returned transform.  The old
    # prototype accidentally reported residuals from the penultimate fit.
    try:
        for _ in range(3):
            scale, rotation, translation = fit_sim3(a[keep], b[keep], config.rank_relative_tolerance)
            residual = np.linalg.norm(apply_sim3(a, scale, rotation, translation) - b, axis=1)
            proposed = residual <= tolerance
            if mode == "oracle" or np.array_equal(proposed, keep) or proposed.sum() < required:
                break
            keep = proposed
        scale, rotation, translation = fit_sim3(a[keep], b[keep], config.rank_relative_tolerance)
        residual = np.linalg.norm(apply_sim3(a, scale, rotation, translation) - b, axis=1)
    except (ValueError, np.linalg.LinAlgError):
        return failed("degenerate_anchors")
    final_keep = np.ones(len(ids), bool) if mode == "oracle" else residual <= tolerance
    if final_keep.sum() < required:
        return failed("weak_correspondence_consensus")
    stats.update(inliers=int(final_keep.sum()), inlier_ratio=float(final_keep.mean()),
                 median_inlier_3d_error_m=float(np.median(residual[final_keep])),
                 maximum_inlier_3d_error_m=float(residual[final_keep].max()),
                 sim3_scale=scale)
    reasons = []
    if mode != "oracle" and final_keep.mean() < config.min_inlier_ratio:
        reasons.append("weak_correspondence_consensus")
    input_ids = ids
    selected = fps_indices(a[final_keep], config.max_anchors)
    ids, idt = ids[final_keep][selected], idt[final_keep][selected]
    if len(ids) < 3 or min(_rank(ps[ids])[0], _rank(pt[idt])[0]) < 2:
        reasons.append("degenerate_anchors")
    stats["selected_anchors"] = len(ids)
    return PreparedAnchors(ps, pt, ids, idt, input_ids, scale, rotation, translation,
                           stats, tuple(reasons), mode, config)


def _anchor_lengths(source_anchors, target_anchors):
    a, b = _points(source_anchors, "source_anchors"), _points(target_anchors, "target_anchors")
    if a.shape != b.shape or len(a) < 3:
        raise ValueError("Relation signatures need at least three paired anchors")
    upper = np.triu_indices(len(a), 1)
    ls = float(np.median(np.linalg.norm(a[:, None] - a[None, :], axis=-1)[upper]))
    lt = float(np.median(np.linalg.norm(b[:, None] - b[None, :], axis=-1)[upper]))
    if min(ls, lt) <= np.finfo(float).tiny:
        raise ValueError("Anchors have no usable median extent")
    return a, b, ls, lt


def signature_scores(candidates, source_center, source_anchors, target_anchors, *,
                     trim_fraction=0.2, chunk_size=2048):
    """Trimmed normalized distance error on every visible candidate point.

    This is exactly Sim3-invariant only when correct anchors and the source
    center undergo one similarity transform.  Arbitrary anisotropic deformation
    is an empirical stress test.  Trimming does not protect the scale estimates
    and can discard the only anchors which distinguish two mirrored locations.
    """
    candidates = _points(candidates, "candidates")
    center = np.asarray(source_center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("source_center must be a finite 3-vector")
    if not 0 <= trim_fraction < 1 or chunk_size < 1:
        raise ValueError("Invalid signature sampling parameters")
    a, b, ls, lt = _anchor_lengths(source_anchors, target_anchors)
    keep = max(3, int(np.ceil((1 - trim_fraction) * len(a))))
    source_distances = np.linalg.norm(a - center, axis=1) / ls
    scores = np.empty(len(candidates), dtype=float)
    for begin in range(0, len(candidates), chunk_size):
        points = candidates[begin:begin + chunk_size]
        differences = np.abs(np.linalg.norm(points[:, None] - b[None], axis=-1) / lt
                             - source_distances)
        scores[begin:begin + len(points)] = np.partition(differences, keep - 1, axis=1)[:, :keep].mean(1)
    rank_s, condition_s = _rank(a)
    rank_t, condition_t = _rank(b)
    return scores, {"source_anchor_extent": ls, "target_anchor_extent": lt,
                    "source_anchor_rank": rank_s, "target_anchor_rank": rank_t,
                    "source_anchor_condition": condition_s, "target_anchor_condition": condition_t,
                    "trim_keep_count": keep}


def retained_anchor_rank(candidate, source_center, source_anchors, target_anchors,
                         trim_fraction=0.2):
    """Rank of the actual distance residual subset kept at one candidate.

    Sorting is stable only to make exact-tie diagnostics reproducible.  Tied
    subsets are not a proof that all equally scoring subsets share this rank.
    """
    a, b, ls, lt = _anchor_lengths(source_anchors, target_anchors)
    if not 0 <= trim_fraction < 1:
        raise ValueError("Invalid trim_fraction")
    difference = np.abs(np.linalg.norm(b - candidate, axis=1) / lt
                        - np.linalg.norm(a - source_center, axis=1) / ls)
    keep = max(3, int(np.ceil((1 - trim_fraction) * len(a))))
    selected = np.argsort(difference, kind="stable")[:keep]
    rs, cs = _rank(a[selected])
    rt, ct = _rank(b[selected])
    return {"source_rank": rs, "target_rank": rt, "source_condition": cs,
            "target_condition": ct, "kept_anchor_indices": selected.tolist()}


def spatial_mode_gap(points, scores, best_index, exclusion_radius):
    """Find the best spatially separated alternative over the entire input.

    Absence of an observed alternative is represented by None.  Dense samples
    around one minimum never hide a distant equally scoring minimum by consuming
    a fixed top-k budget.  This finite-observation test cannot detect an unseen
    or correspondence-consistent semantic symmetry.
    """
    points = _points(points, "points")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(points),) or not 0 <= best_index < len(points):
        raise ValueError("Invalid spatial mode input")
    if not np.isfinite(exclusion_radius) or exclusion_radius <= 0 or not np.isfinite(scores[best_index]):
        raise ValueError("Mode radius and selected score must be finite and valid")
    eligible = np.isfinite(scores) & (np.linalg.norm(points - points[best_index], axis=1) > exclusion_radius)
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return None, None
    alternate = int(indices[np.argmin(scores[indices])])
    return float(scores[alternate] - scores[best_index]), alternate


def _plane(points, tolerance):
    centered = points - points.mean(0)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 3 or singular[0] <= np.finfo(float).tiny or singular[1] <= tolerance * singular[0]:
        raise ValueError("Surface points do not span a plane")
    return basis[0], basis[1], basis[2], float(singular[2] / singular[1])


def _surface_geometry(points, tree, wanted_normal, extent, config):
    """Bounded-radius PCA on all target candidates, processed in small batches.

    A fixed k by itself can bridge two distant shelf levels.  Neighbours outside
    the declared radius are excluded even if k points cannot be collected.
    """
    normals = np.zeros_like(points)
    ratios = np.full(len(points), np.inf)
    agreements = np.zeros(len(points))
    valid = np.zeros(len(points), bool)
    radius = config.neighborhood_radius_ratio * extent
    count = min(config.neighborhood_count, len(points))
    for begin in range(0, len(points), config.chunk_size):
        query = points[begin:begin + config.chunk_size]
        distances, indices = tree.query(query, k=count, distance_upper_bound=radius, workers=1)
        if count == 1:
            distances, indices = distances[:, None], indices[:, None]
        usable = np.isfinite(distances)
        sizes = usable.sum(1)
        neighbors = points[np.minimum(indices, len(points) - 1)]
        means = (neighbors * usable[..., None]).sum(1) / np.maximum(sizes[:, None], 1)
        centered = (neighbors - means[:, None]) * usable[..., None]
        covariance = np.einsum("nki,nkj->nij", centered, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0)
        planar = (sizes >= config.min_neighborhood_points) & (
            eigenvalues[:, 1] > config.rank_relative_tolerance ** 2 * eigenvalues[:, 2])
        planar &= eigenvalues[:, 2] > np.finfo(float).tiny
        ratio = np.full(len(query), np.inf)
        ratio[planar] = np.sqrt(eigenvalues[planar, 0] / eigenvalues[planar, 1])
        normal = eigenvectors[:, :, 0]
        agreement = normal @ wanted_normal
        normal[agreement < 0] *= -1
        agreement = np.clip(np.abs(agreement), 0, 1)
        end = begin + len(query)
        normals[begin:end], ratios[begin:end], agreements[begin:end] = normal, ratio, agreement
        valid[begin:end] = planar & (ratio <= config.max_planarity_ratio) & (
            agreement >= config.normal_min_agreement)
    return normals, ratios, agreements, valid


def _inside_hull(xy, hull, tolerance):
    return np.all(xy @ hull.equations[:, :2].T + hull.equations[:, 2] <= tolerance, axis=1)


def _recover_patch(region, source_center, target_center, target_normal, points,
                   scale, rotation, extent, config):
    u, _, source_normal, _ = _plane(region, config.rank_relative_tolerance)
    predicted_normal = rotation @ source_normal
    if predicted_normal @ target_normal < 0:
        target_normal = -target_normal
    correction, _ = Rotation.align_vectors(target_normal[None], predicted_normal[None])
    local_rotation = correction.as_matrix() @ rotation
    predicted = target_center + scale * ((region - source_center) @ local_rotation.T)
    predicted -= np.outer((predicted - target_center) @ target_normal, target_normal)
    axis_u = local_rotation @ u
    axis_u -= target_normal * (axis_u @ target_normal)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(target_normal, axis_u)
    axes = np.column_stack((axis_u, axis_v))
    hull = ConvexHull((predicted - target_center) @ axes)
    tolerance = 1e-9 * extent
    mask = np.zeros(len(points), bool)
    for begin in range(0, len(points), config.chunk_size):
        delta = points[begin:begin + config.chunk_size] - target_center
        mask[begin:begin + len(delta)] = _inside_hull(delta @ axes, hull, tolerance) & (
            np.abs(delta @ target_normal) <= config.plane_tolerance_ratio * extent)
    # This measures visible support near the transferred source samples, not
    # free space, load capacity, or unobserved interior area.
    coverage = 0.0
    if mask.any():
        nearest = cKDTree(points[mask]).query(predicted, workers=1)[0]
        coverage = float(np.mean(nearest <= config.coverage_distance_ratio * extent))
    return mask, predicted, {"target_patch_points": int(mask.sum()),
                             "region_footprint_observed_fraction": coverage,
                             "target_plane_tolerance_m": config.plane_tolerance_ratio * extent,
                             "footprint_kind": "visible_planar_convex_source_hull"}


def infer_region(src_region_mask, prepared, *, method="relational"):
    """Transfer one source patch with a previously prepared registration.

    The relational comparisons share registration, target surface checks, and
    footprint recovery with global Sim3.  Relation-only removes the center's
    alignment prior; it still uses the shared Sim3 rotation/scale for recovery.
    ``direct_local`` uses only region correspondences for its preparation.
    """
    methods = {"global_sim3", "relational", "relation_only", "direct_local", "oracle_relational"}
    if method not in methods or not isinstance(prepared, PreparedAnchors):
        raise ValueError("Unknown method or invalid PreparedAnchors")
    required_mode = "oracle" if method == "oracle_relational" else "direct" if method == "direct_local" else "automatic"
    if prepared.mode != required_mode:
        raise ValueError(f"{method} requires {required_mode} anchor preparation")
    ps, pt, config = prepared.src_xyz, prepared.tgt_xyz, prepared.config
    source_mask = _mask(src_region_mask, len(ps))
    if method == "direct_local" and not source_mask[prepared.input_src_anchor_idx].all():
        raise ValueError("direct_local may use only source-region correspondences")
    diagnostics = dict(prepared.diagnostics)
    diagnostics.update(method=method, source_region_points=int(source_mask.sum()),
                       candidate_count=len(pt), region_center_definition="arithmetic_centroid")
    result = {"accepted": False, "reasons": list(prepared.reasons),
              "candidate_mask": np.zeros(len(pt), bool), "candidate_center": None,
              "candidate_surface": np.empty((0, 3)), "diagnostics": diagnostics,
              "src_anchor_idx": prepared.src_anchor_idx, "tgt_anchor_idx": prepared.tgt_anchor_idx,
              "sim3_scale": prepared.scale, "sim3_rotation": prepared.rotation,
              "sim3_translation": prepared.translation}
    if prepared.scale is None:
        return result
    if source_mask.sum() < config.min_region_points:
        result["reasons"].append("insufficient_source_region")
        return result
    region = ps[source_mask]
    source_center = region.mean(0)
    try:
        axis_u, axis_v, source_normal, source_planarity = _plane(region, config.rank_relative_tolerance)
    except (ValueError, np.linalg.LinAlgError):
        result["reasons"].append("source_region_not_approximately_planar")
        return result
    diagnostics["source_region_planarity_ratio"] = source_planarity
    diagnostics["source_center"] = source_center.tolist()
    if source_planarity > config.max_planarity_ratio:
        result["reasons"].append("source_region_not_approximately_planar")
    a, b = ps[prepared.src_anchor_idx], pt[prepared.tgt_anchor_idx]
    try:
        relation, relation_stats = signature_scores(pt, source_center, a, b,
            trim_fraction=config.trim_fraction, chunk_size=config.chunk_size)
    except ValueError:
        result["reasons"].append("degenerate_anchors")
        return result
    diagnostics.update(relation_stats)
    ls, lt = relation_stats["source_anchor_extent"], relation_stats["target_anchor_extent"]
    # Detect observed nonconvex annotations: unselected coplanar source samples
    # inside the proposed convex hull are evidence that filling it changes the
    # demonstrated region.  Missing/occluded samples cannot establish a hole.
    try:
        source_axes = np.column_stack((axis_u, axis_v))
        source_hull = ConvexHull((region - source_center) @ source_axes)
        inside = _inside_hull((ps - source_center) @ source_axes, source_hull, 1e-9 * ls)
        inside &= np.abs((ps - source_center) @ source_normal) <= config.plane_tolerance_ratio * diagnostics["source_rms_radius"]
        excluded = float(np.count_nonzero(inside & ~source_mask) / max(int(inside.sum()), 1))
        diagnostics["source_hull_exclusion_fraction"] = excluded
        if excluded > config.max_source_hull_exclusion_fraction:
            result["reasons"].append("source_region_not_convex_in_visible_support")
    except QhullError:
        result["reasons"].append("source_region_not_approximately_planar")
        return result
    coarse_center = apply_sim3(source_center, prepared.scale, prepared.rotation, prepared.translation)
    alignment = np.linalg.norm(pt - coarse_center, axis=1) / lt
    wanted_normal = prepared.rotation @ source_normal
    tree = cKDTree(pt)
    # Surface tests use the same whole-object extent for every method.  Using
    # local-direct anchor extent here would shrink its neighborhood and reject
    # an identical patch solely because its anchors cover a smaller area.
    surface_extent = diagnostics["target_rms_radius"]
    normals, flatness, agreement, eligible = _surface_geometry(pt, tree, wanted_normal, surface_extent, config)
    diagnostics["valid_surface_candidates"] = int(eligible.sum())
    diagnostics["local_surface_radius_m"] = config.neighborhood_radius_ratio * surface_extent
    if not eligible.any():
        result["reasons"].append("no_valid_local_surface")
        return result
    surface_cost = config.planarity_weight * flatness + config.normal_weight * (1 - agreement)
    raw_relation = np.where(eligible, relation, np.inf)
    relation_surface = np.where(eligible, relation + surface_cost, np.inf)
    combined = np.where(eligible, relation + config.alignment_weight * alignment + surface_cost, np.inf)
    raw_best = int(np.argmin(raw_relation))
    combined_best = int(np.argmin(combined))
    raw_gap, raw_alternative = spatial_mode_gap(pt, raw_relation, raw_best, config.mode_exclusion_ratio * lt)
    combined_gap, combined_alternative = spatial_mode_gap(pt, combined, combined_best, config.mode_exclusion_ratio * lt)
    diagnostics.update(raw_relation_gap=raw_gap, combined_gap=combined_gap,
                       raw_relation_alternate_index=raw_alternative,
                       combined_alternate_index=combined_alternative,
                       raw_relation_best_index=raw_best, combined_best_index=combined_best,
                       prior_resolved=bool(raw_gap is not None and raw_gap < config.min_mode_gap
                                           and (combined_gap is None or combined_gap >= config.min_mode_gap)))
    if method in {"global_sim3", "direct_local"}:
        # The baseline projects its predicted center to the closest compatible
        # observed surface.  It never selects a center using the relation score.
        center_index = int(np.argmin(np.where(eligible, alignment, np.inf)))
        # Recentring a patch at an arbitrarily distant visible point would hide
        # an unsupported global prediction: subsequent coverage is measured at
        # the new center.  Bound this correction by observed sampling spacing
        # and the demonstrated patch scale, without consulting target labels.
        valid_indices = np.flatnonzero(eligible)
        sample = valid_indices[np.linspace(0, len(valid_indices) - 1,
                                          min(config.chunk_size, len(valid_indices)), dtype=int)]
        neighbor_distance = tree.query(pt[sample], k=2, workers=1)[0][:, 1]
        positive_distance = neighbor_distance[np.isfinite(neighbor_distance) & (neighbor_distance > 0)]
        spacing = float(np.median(positive_distance)) if len(positive_distance) else 0.0
        projection_limit = max(config.max_projection_spacing_factor * spacing,
                               config.max_projection_region_ratio * prepared.scale * rms_size(region))
        projection_distance = float(np.linalg.norm(pt[center_index] - coarse_center))
        diagnostics.update(observed_surface_spacing_m=spacing,
                           center_projection_limit_m=projection_limit,
                           center_projection_distance_m=projection_distance)
        if projection_distance > projection_limit:
            result["reasons"].append("predicted_center_without_nearby_surface")
    else:
        selected_cost = relation_surface if method == "relation_only" else combined
        center_index = int(np.argmin(selected_cost))
        chosen_gap, _ = spatial_mode_gap(pt, selected_cost, center_index, config.mode_exclusion_ratio * lt)
        if chosen_gap is not None and chosen_gap < config.min_mode_gap:
            result["reasons"].append("ambiguous_spatial_modes")
        if relation[center_index] > config.max_relation_residual:
            result["reasons"].append("large_relation_residual")
    center = pt[center_index]
    retained = retained_anchor_rank(center, source_center, a, b, config.trim_fraction)
    diagnostics.update(candidate_center_index=center_index,
                       center_signature_score=float(relation[center_index]),
                       center_combined_score=float(combined[center_index]),
                       center_alignment_score=float(alignment[center_index]),
                       center_to_coarse_m=float(np.linalg.norm(center - coarse_center)),
                       coarse_center=coarse_center.tolist(),
                       region_normal_agreement=float(agreement[center_index]),
                       retained_source_rank=retained["source_rank"],
                       retained_target_rank=retained["target_rank"],
                       retained_source_condition=retained["source_condition"],
                       retained_target_condition=retained["target_condition"],
                       retained_anchor_indices=retained["kept_anchor_indices"])
    if raw_alternative is not None:
        diagnostics["alternate_retained_rank"] = retained_anchor_rank(
            pt[raw_alternative], source_center, a, b, config.trim_fraction)
    result["candidate_center"] = center.copy()
    try:
        mask, predicted, patch_stats = _recover_patch(region, source_center, center,
            normals[center_index], pt, prepared.scale, prepared.rotation, surface_extent, config)
    except (ValueError, QhullError, np.linalg.LinAlgError):
        result["reasons"].append("degenerate_region_footprint")
        return result
    diagnostics.update(patch_stats)
    result["candidate_mask"], result["candidate_surface"] = mask, predicted
    if mask.sum() < config.min_region_points or patch_stats["region_footprint_observed_fraction"] < config.min_observed_fraction:
        result["reasons"].append("little_visible_target_support")
    result["reasons"] = list(dict.fromkeys(result["reasons"]))
    result["accepted"] = not result["reasons"]
    return result


def transfer_region(src_xyz, tgt_xyz, src_region_mask, src_anchor_idx, tgt_anchor_idx,
                    method="relational", config=None):
    """Convenience entry point; no target truth or simulator metadata is accepted."""
    mode = "oracle" if method == "oracle_relational" else "direct" if method == "direct_local" else "automatic"
    prepared = prepare_anchors(src_xyz, tgt_xyz, src_anchor_idx, tgt_anchor_idx,
                               mode=mode, config=config)
    return infer_region(src_region_mask, prepared, method=method)
# END independent visible-region transfer experiment.
