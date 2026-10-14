"""Independent CPU experiment for locating and sampling an interaction region.

The source interaction mask supplies localization evidence, not a boundary to
copy.  A fixed metric neighborhood is sampled from actual target observations.
The module consumes correspondences and camera origins, never object names,
target region labels, hidden geometry, or a robot state.  An ``accepted`` result
means the localization evidence passed the experiment's checks; it does not
certify physical support, object-footprint containment, or placement safety.

All geometry is in metres.  Source and target coordinates may use independent
frames.  Correspondence indices must be one-to-one; the matcher owns duplicate
resolution so repeated visual predictions cannot silently acquire extra weight.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# BEGIN shared Sim(3) mathematics and explicit registration selection.
from .sim3 import _points, rank as _rank, rms_size, fit_sim3, apply_sim3
from .registration import register_sim3
# END shared registration imports.


# BEGIN independent visible-region transfer experiment; no production imports.
@dataclass(frozen=True)
class TransferConfig:
    """One frozen, dimensionless set of experiment thresholds for every object.

    Relation scores use median matched-anchor distance. RANSAC uses the
    observed object's RMS radius. Local planes use bounded pixel windows and
    the explicit sampling radius. Plane thickness uses measured local point
    spacing; connectivity uses the original eight-neighbor pixel topology.
    Sampling radius/spacing are separate metric call arguments,
    never inferred from container scale or the source annotation's footprint.
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
    min_mode_gap: float = 0.01
    normal_min_agreement: float = 0.65
    max_planarity_ratio: float = 0.15
    normal_pixel_radius: int = 2
    min_neighborhood_points: int = 6
    min_region_points: int = 12
    min_view_cosine: float = 0.05
    surface_height_spacing_factor: float = 2.0
    max_projection_spacing_factor: float = 2.0
    max_projection_sampling_ratio: float = 0.5
    chunk_size: int = 2048

    def __post_init__(self):
        integer_names = ("seed", "ransac_iterations", "min_correspondences",
                         "max_anchors", "normal_pixel_radius",
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
        if (2 * self.normal_pixel_radius + 1) ** 2 < self.min_neighborhood_points:
            raise ValueError("normal pixel window is smaller than its minimum support")
        for name, value in asdict(self).items():
            if name not in integer_names and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.trim_fraction < 1:
            raise ValueError("trim_fraction must lie in [0, 1)")
        for name in ("min_inlier_ratio", "normal_min_agreement", "min_view_cosine"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("rank_relative_tolerance", "ransac_tolerance_ratio",
                     "surface_height_spacing_factor"):
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


def prepare_anchors(src_xyz, tgt_xyz, src_anchor_idx, tgt_anchor_idx, *,
                    mode="automatic", config=None, registration_algorithm="ransac",
                    correspondence_scores=None):
    """Validate and freeze the registration reused across comparison methods.

    ``automatic`` and ``direct`` use the same selected registration algorithm.
    They differ only in the supplied pairs: all-object versus source-region-only.
    PROSAC additionally consumes match scores aligned with these pairs. The
    default RANSAC preserves the previous samples, arithmetic and decisions.
    ``oracle`` fits the supplied known pairs without a matching/consensus gate.
    Invalid array contracts raise ValueError; insufficient geometry is recorded
    as a rejected preparation so batch experiments can keep negative results.
    """
    config = TransferConfig() if config is None else config
    if not isinstance(config, TransferConfig):
        raise ValueError("config must be TransferConfig")
    if mode not in {"automatic", "direct", "oracle"}:
        raise ValueError("Unknown anchor preparation mode")
    # BEGIN algorithm names are explicit and never silently substituted.
    if registration_algorithm not in {"ransac", "prosac_lo", "teaserpp"}:
        raise ValueError("Unknown registration algorithm")
    if mode == "oracle" and registration_algorithm != "ransac":
        raise ValueError("Oracle mode does not select a robust registration backend")
    # END registration selection validation.
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
    # BEGIN explicit backend dispatch; shared region gates remain unchanged.
    if mode == "oracle":
        # Oracle fitting bypasses robust estimation exactly as before. Its
        # correspondences are supplied only by the explicit diagnostic caller.
        keep = np.ones(len(ids), dtype=bool)
        try:
            scale, rotation, translation = fit_sim3(a[keep], b[keep], config.rank_relative_tolerance)
            residual = np.linalg.norm(apply_sim3(a, scale, rotation, translation) - b, axis=1)
            scale, rotation, translation = fit_sim3(a[keep], b[keep], config.rank_relative_tolerance)
            residual = np.linalg.norm(apply_sim3(a, scale, rotation, translation) - b, axis=1)
        except (ValueError, np.linalg.LinAlgError):
            return failed("degenerate_anchors")
        final_keep = keep
    else:
        registration = register_sim3(a, b, algorithm=registration_algorithm,
            scores=correspondence_scores, tolerance_m=tolerance,
            size_ratio=target_radius / source_radius,
            iterations=config.ransac_iterations, seed=config.seed,
            rank_relative_tolerance=config.rank_relative_tolerance,
            min_correspondences=required)
        # Preserve all default diagnostic values for exact baseline comparison.
        # Other backends record their actual raw solution and common refit.
        if registration_algorithm != "ransac":
            stats["registration"] = registration.diagnostics
        if registration.reason is not None:
            return failed(registration.reason)
        scale, rotation, translation = (registration.scale, registration.rotation,
                                        registration.translation)
        residual, final_keep = registration.residuals, registration.inliers
    # END backend dispatch; final anchor selection and region logic are shared.
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


# BEGIN fixed metric neighborhoods replace copied source footprints.
def _plane(points, tolerance):
    centered = points - points.mean(0)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 3 or singular[0] <= np.finfo(float).tiny or singular[1] <= tolerance * singular[0]:
        raise ValueError("Surface points do not span a plane")
    return basis[0], basis[1], basis[2], float(singular[2] / singular[1])


def _camera_origin(value, name):
    raw = np.asarray(value)
    if raw.shape != (3,) or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite numeric 3-vector in its point-cloud frame")
    return np.asarray(raw, dtype=float)


def _pixels(value, size):
    """Require original observation coordinates, retaining missing-pixel gaps."""
    raw = np.asarray(value)
    if raw.shape != (size, 2) or raw.dtype.kind not in "iu":
        raise ValueError("target_pixels must be an integer [N,2] array aligned with target points")
    if len(np.unique(raw, axis=0)) != size:
        raise ValueError("target_pixels must contain one unique pixel per target point")
    return raw


def _sampling_parameters(radius, spacing, max_modes=None):
    for name, value in (("sampling_radius_m", radius), ("sampling_spacing_m", spacing)):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if max_modes is not None:
        if isinstance(max_modes, (bool, np.bool_)) or not isinstance(max_modes, (int, np.integer)) or max_modes < 1:
            raise ValueError("max_modes must be a positive integer")


def _surface_geometry(points, pixels, wanted_normal, camera_origin, sampling_radius_m, config):
    """Estimate local planes and their camera-facing orientation on all points.

    A normal is first oriented toward its own observing camera, then compared
    with the transformed source normal using a signed dot product. Opposite
    visible sides of a thin board therefore do not become equivalent merely
    because their unoriented planes are parallel. Grazing views cannot reliably
    determine this sign and are excluded explicitly. The plane support consists
    of observed points within a fixed pixel window and the physical sampling
    radius. Selecting nearest 3-D points instead can select only one pixel row
    under an oblique view, even on a perfectly observed two-dimensional plane.
    This is visibility evidence, not a gravity or physical-stability test.
    """
    normals = np.zeros_like(points)
    ratios = np.full(len(points), np.inf)
    agreements = np.zeros(len(points))
    view_cosines = np.zeros(len(points))
    spacings = np.zeros(len(points))
    valid = np.zeros(len(points), bool)
    tree = cKDTree(pixels)
    count = min((2 * config.normal_pixel_radius + 1) ** 2, len(points))
    for begin in range(0, len(points), config.chunk_size):
        query = points[begin:begin + config.chunk_size]
        pixel_distances, indices = tree.query(pixels[begin:begin + config.chunk_size],
            k=count, p=np.inf,
            distance_upper_bound=np.nextafter(float(config.normal_pixel_radius), np.inf), workers=1)
        if count == 1:
            pixel_distances, indices = pixel_distances[:, None], indices[:, None]
        neighbors = points[np.minimum(indices, len(points) - 1)]
        distances = np.linalg.norm(neighbors - query[:, None], axis=2)
        usable = np.isfinite(pixel_distances) & (distances <= sampling_radius_m * (1 + 1e-12))
        sizes = usable.sum(1)
        means = (neighbors * usable[..., None]).sum(1) / np.maximum(sizes[:, None], 1)
        centered = (neighbors - means[:, None]) * usable[..., None]
        eigenvalues, eigenvectors = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered))
        eigenvalues = np.maximum(eigenvalues, 0)
        planar = (sizes >= config.min_neighborhood_points) & (
            eigenvalues[:, 1] > config.rank_relative_tolerance ** 2 * eigenvalues[:, 2])
        planar &= eigenvalues[:, 2] > np.finfo(float).tiny
        ratio = np.full(len(query), np.inf)
        ratio[planar] = np.sqrt(eigenvalues[planar, 0] / eigenvalues[planar, 1])
        normal = eigenvectors[:, :, 0]
        view = camera_origin - query
        view_length = np.linalg.norm(view, axis=1)
        signed_view = np.einsum("ij,ij->i", normal, view) / np.maximum(view_length, np.finfo(float).tiny)
        normal[signed_view < 0] *= -1
        view_cosine = np.clip(np.abs(signed_view), 0, 1)
        agreement = np.clip(normal @ wanted_normal, -1, 1)
        # Short-range observed spacing controls only the plane's thickness.
        # It must not control connectivity: oblique RGB-D pixels can be close
        # along a row and much farther apart between neighboring rows.
        positive = np.sort(np.where(usable & (distances > 0), distances, np.inf), axis=1)
        spacing = np.median(positive[:, :min(4, count)], axis=1)
        spacing_valid = np.isfinite(spacing) & (spacing > 0)
        spacing = np.where(spacing_valid, spacing, 0.0)
        end = begin + len(query)
        normals[begin:end], ratios[begin:end], agreements[begin:end] = normal, ratio, agreement
        view_cosines[begin:end], spacings[begin:end] = view_cosine, spacing
        valid[begin:end] = planar & spacing_valid & (ratio <= config.max_planarity_ratio)
        valid[begin:end] &= (agreement >= config.normal_min_agreement)
        valid[begin:end] &= (view_cosine >= config.min_view_cosine) & (view_length > 0)
    return normals, ratios, agreements, view_cosines, spacings, valid


def _same_seed_plane(points, normals, seed_index, height_tolerance, config):
    """Compatibility with one frozen seed plane, never a transitive union.

    Chaining pairwise compatible neighborhoods can walk around a bevel onto a
    second shelf level. Every membership check here uses the original seed's
    plane and oriented normal, so that chain cannot expand the plane slab.
    """
    normal = normals[seed_index]
    delta = points - points[seed_index]
    height = delta @ normal
    tangent_distance = np.sqrt(np.maximum(np.sum(delta * delta, axis=1) - height * height, 0))
    compatible = (np.abs(height) <= height_tolerance)
    compatible &= normals @ normal >= config.normal_min_agreement
    return compatible, tangent_distance


def _pixel_components(pixels):
    """Label the exact eight-neighbor graph of the supplied observed pixels.

    A one-pixel Chebyshev ball contains at most nine unique pixels including
    self. Missing rows/columns retain their coordinates and therefore cannot
    acquire an edge merely because their 3-D observations are nearby.
    """
    count = min(9, len(pixels))
    distances, neighbors = cKDTree(pixels).query(pixels, k=count, p=np.inf,
        distance_upper_bound=np.nextafter(1.0, np.inf), workers=1)
    if count == 1:
        distances, neighbors = distances[:, None], neighbors[:, None]
    usable = np.isfinite(distances)
    rows = np.broadcast_to(np.arange(len(pixels))[:, None], neighbors.shape)[usable]
    columns = neighbors[usable]
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)),
                       shape=(len(pixels), len(pixels))).tocsr()
    return connected_components(graph, directed=False)[1]


def sample_neighborhood(points, eligible, normals, local_spacing, seed_index, *, target_pixels,
                        sampling_radius_m, sampling_spacing_m, config=None):
    """Sample actual observations from one fixed-radius, seed-connected patch.

    ``local_spacing`` has one positive finite value per eligible point. Radius
    is measured in the seed plane; its thickness uses a fixed multiple of the
    seed's observed spacing. Connectivity is the original eight-neighbor pixel
    graph restricted to eligible members of that seed plane and radius. Missing
    pixels retain their original coordinates, so an occluded pixel gap cannot
    be compressed away. No point is added inside a hole, and no convex hull or
    object footprint is constructed.

    The returned mask names observed neighborhood members. ``sample_indices``
    are point-cloud indices, beginning with the preferred seed and followed by
    deterministic farthest samples separated by at least sampling_spacing_m.
    """
    points = _points(points, "points")
    pixels = _pixels(target_pixels, len(points))
    # BEGIN reuse validated coordinates without weakening the public contract.
    return _sample_neighborhood_validated_coordinates(points, eligible, normals,
        local_spacing, seed_index, pixels=pixels, sampling_radius_m=sampling_radius_m,
        sampling_spacing_m=sampling_spacing_m, config=config)


def _sample_neighborhood_validated_coordinates(points, eligible, normals, local_spacing,
        seed_index, *, pixels, sampling_radius_m, sampling_spacing_m, config=None):
    """Sample with point/pixel arrays already checked by the immediate caller.

    This helper is scoped to the current call, not a cache keyed by array
    identity. All query-dependent normal, spacing and seed checks remain here.
    """
    eligible, normals = np.asarray(eligible), np.asarray(normals, dtype=float)
    local_spacing = np.asarray(local_spacing, dtype=float)
    config = TransferConfig() if config is None else config
    _sampling_parameters(sampling_radius_m, sampling_spacing_m)
    if not isinstance(config, TransferConfig):
        raise ValueError("config must be TransferConfig")
    if eligible.dtype != np.bool_ or eligible.shape != (len(points),):
        raise ValueError("eligible must be a point-aligned Boolean array")
    if normals.shape != points.shape or local_spacing.shape != (len(points),):
        raise ValueError("Normals and local spacings must align with target points")
    if (not np.isfinite(normals[eligible]).all()
            or not np.allclose(np.linalg.norm(normals[eligible], axis=1), 1.0, atol=1e-6)
            or not np.isfinite(local_spacing[eligible]).all()
            or np.any(local_spacing[eligible] <= 0)):
        raise ValueError("Eligible normals must be unit vectors and spacings finite and positive")
    if isinstance(seed_index, (bool, np.bool_)) or not isinstance(seed_index, (int, np.integer)):
        raise ValueError("seed_index must be an integer")
    if not 0 <= seed_index < len(points) or not eligible[seed_index]:
        raise ValueError("Seed must be an eligible target observation")
    spacing = float(local_spacing[seed_index])
    height_tolerance = config.surface_height_spacing_factor * spacing
    same_plane, tangent_distance = _same_seed_plane(points, normals, seed_index, height_tolerance, config)
    members = np.flatnonzero(eligible & same_plane & (tangent_distance <= sampling_radius_m * (1 + 1e-12)))
    neighborhood = points[members]
    seed_local = int(np.searchsorted(members, seed_index))
    # Unlike a metric radius graph, the pixel graph does not fragment a plane into
    # individual rows merely because the two pixel directions have unequal
    # physical spacing. The fixed seed-plane slab independently blocks jumps
    # to a different depth surface.
    components = _pixel_components(pixels[members])
    connected = components == components[seed_local]
    members = members[connected]
    neighborhood = points[members]
    seed_local = int(np.searchsorted(members, seed_index))
    # The sampling metric is physical distance, not a scaled source footprint.
    # There is no random seed and no surface projection of synthetic samples.
    chosen = [seed_local]
    selected = np.zeros(len(members), bool)
    selected[seed_local] = True
    closest = np.sum((neighborhood - neighborhood[seed_local]) ** 2, axis=1)
    closest[selected] = -np.inf
    while not selected.all():
        farthest = int(np.argmax(closest))
        if closest[farthest] < sampling_spacing_m ** 2 * (1 - 1e-12):
            break
        chosen.append(farthest)
        selected[farthest] = True
        closest = np.minimum(closest, np.sum((neighborhood - neighborhood[farthest]) ** 2, axis=1))
        closest[selected] = -np.inf
    mask = np.zeros(len(points), bool)
    mask[members] = True
    rank, condition = _rank(neighborhood, config.rank_relative_tolerance)
    return {"seed_index": int(seed_index), "center": points[seed_index].copy(),
            "normal": normals[seed_index].copy(), "mask": mask,
            "sample_indices": members[np.asarray(chosen, dtype=np.int64)],
            "diagnostics": {"observed_local_spacing_m": spacing,
                "surface_height_tolerance_m": height_tolerance,
                "connectivity": "observed_pixel_8_neighbors",
                "sampling_radius_m": float(sampling_radius_m),
                "sampling_spacing_m": float(sampling_spacing_m),
                "neighborhood_point_count": int(len(members)), "sample_count": len(chosen),
                "neighborhood_rank": rank, "neighborhood_condition": condition}}

# END sampling on coordinates validated at the public boundary.


def _surface_gap(points, normals, spacings, eligible, scores, best_index, config):
    """Search every eligible observation for a different seed-plane hypothesis.

    Output mode budget never limits this evidence check. In particular, many
    low-score points on the first level cannot hide a competing second level.
    """
    same, _ = _same_seed_plane(points, normals, best_index,
        config.surface_height_spacing_factor * spacings[best_index], config)
    candidates = np.flatnonzero(eligible & ~same & np.isfinite(scores))
    if not len(candidates):
        return None, None
    index = int(candidates[np.argmin(scores[candidates])])
    return float(scores[index] - scores[best_index]), index


def _overlapping_seed_neighborhood(points, pixels, normals, eligible, mode, radius, config):
    """Suppress overlapping centers relative to this one seed, not recursively.

    This is a geometric window-redundancy rule within one observed pixel
    component, not a proof that every hypothetical connected output mask
    intersects the primary mask around winding occlusions. Candidate centers
    must lie within two radii on this seed's fixed plane, connect to this seed
    inside that same bounded domain, and lie within one radius of an observed
    primary member. Disconnected coplanar observations never suppress each
    other. Distinct parallel levels independently fail the fixed-plane test.
    """
    same, tangent_distance = _same_seed_plane(points, normals, mode["seed_index"],
        mode["diagnostics"]["surface_height_tolerance_m"], config)
    possible = eligible & same & (tangent_distance <= 2 * radius * (1 + 1e-12))
    indices = np.flatnonzero(possible)
    suppressed = np.zeros(len(points), bool)
    if len(indices):
        components = _pixel_components(pixels[indices])
        seed_component = components[np.searchsorted(indices, mode["seed_index"])]
        indices = indices[components == seed_component]
        # Remove the small normal residual when measuring overlap in the seed
        # chart. All queried points must already satisfy its plane slab.
        normal, center = mode["normal"], mode["center"]
        observed = points[mode["mask"]] - center
        observed -= np.outer(observed @ normal, normal)
        query = points[indices] - center
        query -= np.outer(query @ normal, normal)
        distance = cKDTree(observed).query(query, workers=1)[0]
        suppressed[indices] = distance <= radius * (1 + 1e-12)
    suppressed[mode["mask"]] = True
    return suppressed


def infer_region(src_region_mask, prepared, *, target_pixels, source_camera_origin, target_camera_origin,
                 sampling_radius_m, sampling_spacing_m, max_modes, method="relational"):
    """Return localization preferences and fixed metric observed neighborhoods.

    The source mask is evidence for a preferred location and plane orientation;
    its outline is never transferred. Every method uses the same sampling
    parameters. Target pixels retain the original observed image topology;
    source/target camera origins must be supplied in the respective point-cloud
    frames. There is no unoriented-normal compatibility fallback.

    Different positions on the same plane remain location alternatives, which
    this neighborhood task permits. A close-scoring different surface is an
    unresolved surface hypothesis and rejects the result. All-domain ambiguity
    checks are independent of max_modes, which only bounds returned evidence.
    """
    methods = {"global_sim3", "relational", "relation_only", "direct_local", "oracle_relational"}
    if method not in methods or not isinstance(prepared, PreparedAnchors):
        raise ValueError("Unknown method or invalid PreparedAnchors")
    source_camera = _camera_origin(source_camera_origin, "source_camera_origin")
    target_camera = _camera_origin(target_camera_origin, "target_camera_origin")
    _sampling_parameters(sampling_radius_m, sampling_spacing_m, max_modes)
    required_mode = "oracle" if method == "oracle_relational" else "direct" if method == "direct_local" else "automatic"
    if prepared.mode != required_mode:
        raise ValueError(f"{method} requires {required_mode} anchor preparation")
    ps, pt, config = prepared.src_xyz, prepared.tgt_xyz, prepared.config
    pixels = _pixels(target_pixels, len(pt))
    source_mask = _mask(src_region_mask, len(ps))
    if method == "direct_local" and not source_mask[prepared.input_src_anchor_idx].all():
        raise ValueError("direct_local may use only source-region correspondences")
    diagnostics = dict(prepared.diagnostics)
    diagnostics.update(method=method, source_region_points=int(source_mask.sum()),
        candidate_count=len(pt), region_center_definition="arithmetic_centroid_of_source_evidence",
        sampling_radius_m=float(sampling_radius_m), sampling_spacing_m=float(sampling_spacing_m),
        max_modes=int(max_modes), source_camera_origin=source_camera.tolist(),
        target_camera_origin=target_camera.tolist(),
        output_meaning="localization preference and observed neighborhood; not a physical feasible set")
    result = {"accepted": False, "reasons": list(prepared.reasons), "modes": [],
              "diagnostics": diagnostics,
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
        _, _, source_normal, source_planarity = _plane(region, config.rank_relative_tolerance)
    except (ValueError, np.linalg.LinAlgError):
        result["reasons"].append("source_region_not_approximately_planar")
        return result
    diagnostics["source_region_planarity_ratio"] = source_planarity
    diagnostics["source_center"] = source_center.tolist()
    if source_planarity > config.max_planarity_ratio:
        result["reasons"].append("source_region_not_approximately_planar")
    source_view = source_camera - source_center
    source_view_length = float(np.linalg.norm(source_view))
    source_cosine = float(source_normal @ source_view / max(source_view_length, np.finfo(float).tiny))
    diagnostics["source_view_cosine"] = abs(source_cosine)
    if source_view_length <= 0 or abs(source_cosine) < config.min_view_cosine:
        result["reasons"].append("source_normal_orientation_unobservable")
        return result
    if source_cosine < 0:
        source_normal = -source_normal
    diagnostics["source_oriented_normal"] = source_normal.tolist()
    a, b = ps[prepared.src_anchor_idx], pt[prepared.tgt_anchor_idx]
    try:
        relation, relation_stats = signature_scores(pt, source_center, a, b,
            trim_fraction=config.trim_fraction, chunk_size=config.chunk_size)
    except ValueError:
        result["reasons"].append("degenerate_anchors")
        return result
    diagnostics.update(relation_stats)
    lt = relation_stats["target_anchor_extent"]
    coarse_center = apply_sim3(source_center, prepared.scale, prepared.rotation, prepared.translation)
    alignment = np.linalg.norm(pt - coarse_center, axis=1) / lt
    wanted_normal = prepared.rotation @ source_normal
    normals, flatness, agreement, view_cosines, spacings, eligible = _surface_geometry(
        pt, pixels, wanted_normal, target_camera, sampling_radius_m, config)
    diagnostics.update(valid_surface_candidates=int(eligible.sum()),
        local_surface_radius_m=float(sampling_radius_m),
        normal_pixel_radius=config.normal_pixel_radius,
        surface_connectivity="observed_pixel_8_neighbors",
        grazing_view_points=int(np.count_nonzero(view_cosines < config.min_view_cosine)),
        coarse_center=coarse_center.tolist())
    if not eligible.any():
        result["reasons"].append("no_valid_oriented_surface")
        return result
    surface_cost = config.planarity_weight * np.where(eligible, flatness, 0.0)
    surface_cost += config.normal_weight * (1 - agreement)
    raw_relation = np.where(eligible, relation, np.inf)
    relation_surface = np.where(eligible, relation + surface_cost, np.inf)
    combined = np.where(eligible, relation + config.alignment_weight * alignment + surface_cost, np.inf)
    raw_best, combined_best = int(np.argmin(raw_relation)), int(np.argmin(combined))
    raw_gap, raw_alternative = _surface_gap(pt, normals, spacings, eligible, raw_relation, raw_best, config)
    combined_gap, combined_alternative = _surface_gap(pt, normals, spacings, eligible, combined, combined_best, config)
    diagnostics.update(raw_relation_different_surface_gap=raw_gap,
        combined_different_surface_gap=combined_gap,
        raw_relation_alternate_surface_index=raw_alternative,
        combined_alternate_surface_index=combined_alternative,
        raw_relation_best_index=raw_best, combined_best_index=combined_best,
        prior_resolved=bool(raw_gap is not None and raw_gap < config.min_mode_gap
            and (combined_gap is None or combined_gap >= config.min_mode_gap)))
    if method in {"global_sim3", "direct_local"}:
        selected_cost = np.where(eligible, alignment, np.inf)
    else:
        selected_cost = relation_surface if method == "relation_only" else combined
    center_index = int(np.argmin(selected_cost))
    center = pt[center_index]
    surface_gap, surface_alternative = _surface_gap(pt, normals, spacings, eligible,
                                                   selected_cost, center_index, config)
    surface_ambiguous = surface_gap is not None and surface_gap < config.min_mode_gap
    diagnostics.update(selected_different_surface_gap=surface_gap,
        selected_alternate_surface_index=surface_alternative,
        surface_ambiguity=bool(surface_ambiguous),
        preferred_center_index=center_index,
        center_signature_score=float(relation[center_index]),
        center_combined_score=float(combined[center_index]),
        center_alignment_score=float(alignment[center_index]),
        center_to_coarse_m=float(np.linalg.norm(center - coarse_center)),
        region_normal_agreement=float(agreement[center_index]),
        target_view_cosine=float(view_cosines[center_index]))
    if surface_ambiguous:
        result["reasons"].append("surface_ambiguity")
    if method in {"global_sim3", "direct_local"}:
        projection_limit = max(config.max_projection_spacing_factor * spacings[center_index],
                               config.max_projection_sampling_ratio * sampling_radius_m)
        projection_distance = float(np.linalg.norm(center - coarse_center))
        diagnostics.update(observed_surface_spacing_m=float(spacings[center_index]),
            center_projection_limit_m=float(projection_limit), center_projection_distance_m=projection_distance)
        if projection_distance > projection_limit:
            result["reasons"].append("predicted_center_without_nearby_surface")
    elif relation[center_index] > config.max_relation_residual:
        result["reasons"].append("large_relation_residual")
    retained = retained_anchor_rank(center, source_center, a, b, config.trim_fraction)
    diagnostics.update(retained_source_rank=retained["source_rank"],
        retained_target_rank=retained["target_rank"],
        retained_source_condition=retained["source_condition"],
        retained_target_condition=retained["target_condition"],
        retained_anchor_indices=retained["kept_anchor_indices"])
    if raw_alternative is not None:
        diagnostics["alternate_retained_rank"] = retained_anchor_rank(
            pt[raw_alternative], source_center, a, b, config.trim_fraction)
    # Freeze comparisons against the primary seed. Secondary output windows
    # never redefine the primary plane or enlarge its sampling radius.
    primary_plane, primary_tangent = _same_seed_plane(pt, normals, center_index,
        config.surface_height_spacing_factor * spacings[center_index], config)
    distant_same_plane = eligible & primary_plane & (primary_tangent > 2 * sampling_radius_m)
    distant_indices = np.flatnonzero(distant_same_plane)
    position_gap, position_index = None, None
    if len(distant_indices):
        position_index = int(distant_indices[np.argmin(selected_cost[distant_indices])])
        position_gap = float(selected_cost[position_index] - selected_cost[center_index])
    diagnostics.update(same_plane_position_gap=position_gap, same_plane_position_index=position_index,
        position_preference_ambiguous=bool(position_gap is not None and position_gap < config.min_mode_gap))
    remaining = eligible.copy()
    next_seed = center_index
    for mode_number in range(max_modes):
        mode = _sample_neighborhood_validated_coordinates(
            pt, eligible, normals, spacings, next_seed, pixels=pixels,
            sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m, config=config)
        mode.update(score=float(selected_cost[next_seed]), relation_score=float(relation[next_seed]),
                    type="primary" if mode_number == 0 else (
                        "same_plane_localization" if primary_plane[next_seed] else "different_surface"))
        mode["diagnostics"].update(normal_agreement=float(agreement[next_seed]),
            view_cosine=float(view_cosines[next_seed]),
            raw_relation_min=float(relation[mode["mask"]].min()),
            raw_relation_max=float(relation[mode["mask"]].max()))
        result["modes"].append(mode)
        remaining &= ~_overlapping_seed_neighborhood(pt, pixels, normals, eligible, mode,
                                                     sampling_radius_m, config)
        if mode_number == 0 and surface_ambiguous and max_modes > 1:
            # A same-plane continuum must not consume the output budget and
            # hide the actual surface-ambiguity witness. This changes only
            # which diagnostic modes are stored, never their scores or gate.
            next_seed = surface_alternative
            continue
        if not remaining.any():
            break
        next_seed = int(np.argmin(np.where(remaining, selected_cost, np.inf)))
    # The primary remains first; subsequent diagnostic modes are score ordered.
    result["modes"][1:] = sorted(result["modes"][1:], key=lambda mode: (mode["score"], mode["seed_index"]))
    primary = result["modes"][0]
    if (primary["diagnostics"]["neighborhood_point_count"] < config.min_neighborhood_points
            or primary["diagnostics"]["neighborhood_rank"] < 2):
        result["reasons"].append("little_visible_target_support")
    diagnostics["returned_modes"] = len(result["modes"])
    diagnostics["sampling_scale_applied"] = False
    result["reasons"] = list(dict.fromkeys(result["reasons"]))
    result["accepted"] = not result["reasons"]
    return result


def transfer_region(src_xyz, tgt_xyz, src_region_mask, src_anchor_idx, tgt_anchor_idx,
                    method="relational", config=None, *, target_pixels, source_camera_origin, target_camera_origin,
                    sampling_radius_m, sampling_spacing_m, max_modes,
                    registration_algorithm="ransac", correspondence_scores=None):
    """Convenience entry; source evidence and explicit metric sampling only."""
    mode = "oracle" if method == "oracle_relational" else "direct" if method == "direct_local" else "automatic"
    prepared = prepare_anchors(src_xyz, tgt_xyz, src_anchor_idx, tgt_anchor_idx,
                               mode=mode, config=config,
                               registration_algorithm=registration_algorithm,
                               correspondence_scores=correspondence_scores)
    return infer_region(src_region_mask, prepared, method=method,
        target_pixels=target_pixels,
        source_camera_origin=source_camera_origin, target_camera_origin=target_camera_origin,
        sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m, max_modes=max_modes)
# END fixed metric neighborhoods replace copied source footprints.
# END independent visible-region transfer experiment.
