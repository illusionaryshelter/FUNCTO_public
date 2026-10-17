"""Two RGB-D images as an observed surface with explicit measurement provenance.

The complete positive depth images define geometry; entity segmentation only
labels existing observations. A surface element contains one measurement, or
one mutually supported pair from different cameras. Its coordinate is always
one actual measurement. No mesh, averaged point, voxel size, or hidden surface
is introduced. Unpaired elements are not evidence of physical exclusivity.

Local planes and their camera-facing signs reuse the existing single-view
estimator. Plane slabs are a discrete association model based on pixel spacing,
not calibrated sensor uncertainty or a free-space certificate.
"""

# BEGIN scalar nextafter handles an overflowing bound without NumPy warnings.
import math
# END floating-point bound dependency.

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .core import (TransferConfig, _pixels, _rank, _sampling_parameters,
                   _same_seed_plane, _surface_geometry)
from .segmentation import _calibration
from .sim3 import _points


STATE_NAMES = ("owner_observed", "co_observed", "behind_camera", "outside_image",
               "invalid_depth", "invalid_local_plane", "corner_surface_mismatch",
               "outside_local_radius", "front_of_observed_surface",
               "behind_observed_surface", "invalid_projection")
_STATE = {name: i for i, name in enumerate(STATE_NAMES)}


def _view_arrays(view):
    """Backproject a complete image and separately label its entity observations."""
    depth = np.asarray(view["depth_m"])
    if (depth.ndim != 2 or depth.dtype.kind not in "fiu" or min(depth.shape) < 1):
        raise ValueError("depth_m must be a nonempty numeric optical-depth image")
    K, T = _calibration(view["K"], view["T_world_camera_cv"])
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    pixels = np.column_stack((x, y)).astype(np.int64)
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    xyz = (rays * depth[y, x, None]) @ T[:3, :3].T + T[:3, 3]
    lookup = np.full(depth.shape, -1, dtype=np.int64)
    lookup[y, x] = np.arange(len(xyz))
    entity = np.zeros(len(xyz), bool)
    observation = view.get("observation")
    if observation is not None:
        points = _points(observation["xyz"], "entity observation xyz")
        xy = _pixels(observation["pixels"], len(points))
        if (np.any(xy < 0) or np.any(xy[:, 0] >= depth.shape[1])
                or np.any(xy[:, 1] >= depth.shape[0])):
            raise ValueError("Entity pixels must lie in the original depth image")
        ids = lookup[xy[:, 1], xy[:, 0]]
        if np.any(ids < 0):
            raise ValueError("Entity observations must have valid original depth")
        # Cached observations may store float32 coordinates. This is solely a
        # serialization-roundoff check, not a surface association tolerance.
        if not np.allclose(xyz[ids], points, rtol=1e-6, atol=1e-8):
            raise ValueError("Entity coordinates disagree with their calibrated depth image")
        labels = np.asarray(observation["entity_member"])
        if labels.dtype != np.bool_ or labels.shape != (len(points),):
            raise ValueError("entity_member must be Boolean and point aligned")
        entity[ids] = labels
    return xyz, pixels, entity, lookup, K, T, depth


def project_observations(surface, points, view_index, *, tolerance_m=0):
    """Classify points against one full depth image, retaining unknown reasons.

    A non-owner projection requires all enclosing pixels to describe the same
    oriented local plane. It reads the deterministic nearest original pixel;
    no search for a favorable depth or appearance value is permitted. Missing
    depth, occlusion, frame exits and unresolved planes are unknown. A point
    on the camera side of a supported plane slab is a model conflict, not a
    statistically certified collision/free-space observation.
    """
    points = _points(points, "projection points")
    if view_index not in (0, 1):
        raise ValueError("view_index must be 0 or 1")
    if (isinstance(tolerance_m, (bool, np.bool_)) or not np.isscalar(tolerance_m)
            or not np.isfinite(tolerance_m) or tolerance_m < 0):
        raise ValueError("tolerance_m must be finite and nonnegative")
    K, T = surface["K"][view_index], surface["T_world_camera_cv"][view_index]
    lookup = surface["pixel_to_observation"][view_index]
    height, width = lookup.shape
    observed, normals = surface["observation_xyz"], surface["observation_normals"]
    spacing, valid = surface["observation_spacing"], surface["observation_valid"]
    factor = surface["surface_height_spacing_factor"]
    agreement = surface["normal_min_agreement"]
    radius = surface["sampling_radius_m"]
    pixels = np.full((len(points), 2), np.nan)
    ids = np.full(len(points), -1, dtype=np.int64)
    states = np.full(len(points), _STATE["invalid_projection"], dtype=np.int8)
    # Chunking bounds the four-corner temporary arrays for full VGA images.
    for begin in range(0, len(points), 8192):
        stop = min(begin + 8192, len(points))
        camera = (points[begin:stop] - T[:3, 3]) @ T[:3, :3]
        homogeneous = camera @ K.T
        finite = np.isfinite(homogeneous).all(1) & (homogeneous[:, 2] != 0)
        uv = np.full((stop - begin, 2), np.nan)
        uv[finite] = homogeneous[finite, :2] / homogeneous[finite, 2, None]
        # Undo only floating-point roundoff of integer pixel centers. Without
        # this, an original border pixel could project just outside its frame.
        rounded = np.rint(uv)
        numeric = 64 * np.finfo(float).eps * max(height, width)
        uv = np.where(np.abs(uv - rounded) <= numeric, rounded, uv)
        pixels[begin:stop] = uv
        front = finite & (camera[:, 2] > 0)
        states[begin:stop][finite & ~front] = _STATE["behind_camera"]
        inside = front & np.all(uv >= 0, axis=1) & np.all(uv <= [width-1, height-1], axis=1)
        states[begin:stop][front & ~inside] = _STATE["outside_image"]
        active = begin + np.flatnonzero(inside)
        if not len(active):
            continue
        xy = np.rint(pixels[active]).astype(np.int64)
        near = lookup[xy[:, 1], xy[:, 0]]
        has_depth = near >= 0
        states[active[~has_depth]] = _STATE["invalid_depth"]
        active, near = active[has_depth], near[has_depth]
        ids[active] = near
        if not len(active):
            continue
        exact = np.all(np.abs(points[active] - observed[near]) <= (
            64 * np.finfo(float).eps * np.maximum(1, np.abs(observed[near]))), axis=1)
        states[active[exact]] = _STATE["owner_observed"]
        active, near = active[~exact], near[~exact]
        if not len(active):
            continue
        low, high = np.floor(pixels[active]).astype(int), np.ceil(pixels[active]).astype(int)
        corners = np.stack((low, np.column_stack((high[:, 0], low[:, 1])),
                            np.column_stack((low[:, 0], high[:, 1])), high), axis=1)
        corner_ids = lookup[corners[:, :, 1], corners[:, :, 0]]
        good = np.all(corner_ids >= 0, axis=1)
        states[active[~good]] = _STATE["invalid_depth"]
        active, near, corner_ids = active[good], near[good], corner_ids[good]
        good = valid[near] & np.all(valid[corner_ids], axis=1)
        states[active[~good]] = _STATE["invalid_local_plane"]
        active, near, corner_ids = active[good], near[good], corner_ids[good]
        if not len(active):
            continue
        normal = normals[near]
        thickness = factor * spacing[near]
        corner_delta = observed[corner_ids] - observed[near, None]
        good = np.all(np.abs(np.einsum("nki,ni->nk", corner_delta, normal)) <= thickness[:, None], axis=1)
        good &= np.all(np.einsum("nki,ni->nk", normals[corner_ids], normal) >= agreement, axis=1)
        states[active[~good]] = _STATE["corner_surface_mismatch"]
        active, near, normal, thickness = active[good], near[good], normal[good], thickness[good]
        delta = points[active] - observed[near]
        signed_height = np.einsum("ni,ni->n", delta, normal)
        in_slab = np.abs(signed_height) <= thickness + tolerance_m
        local = np.linalg.norm(delta, axis=1) <= radius + tolerance_m
        states[active[in_slab & local]] = _STATE["co_observed"]
        states[active[in_slab & ~local]] = _STATE["outside_local_radius"]
        states[active[~in_slab & (signed_height > 0)]] = _STATE["front_of_observed_surface"]
        states[active[~in_slab & (signed_height <= 0)]] = _STATE["behind_observed_surface"]
    supported = (states == _STATE["owner_observed"]) | (states == _STATE["co_observed"])
    conflict = states == _STATE["front_of_observed_surface"]
    return {"supported": supported, "conflict": conflict, "unknown": ~(supported | conflict),
            "pixels": pixels, "observation_indices": ids,
            "states": states, "state_names": list(STATE_NAMES)}


def _adjacency(surface, config):
    """Map original eight-neighbor image edges to compatible surface elements."""
    rows, columns = [], []
    xyz, normals = surface["observation_xyz"], surface["observation_normals"]
    valid, spacing = surface["observation_valid"], surface["observation_spacing"]
    mapping = surface["observation_to_element"]
    for lookup in surface["pixel_to_observation"]:
        height, width = lookup.shape
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            y0, y1 = max(0, -dy), min(height, height-dy)
            x0, x1 = max(0, -dx), min(width, width-dx)
            a = lookup[y0:y1, x0:x1].ravel()
            b = lookup[y0+dy:y1+dy, x0+dx:x1+dx].ravel()
            present = (a >= 0) & (b >= 0)
            a, b = a[present], b[present]
            delta = xyz[b] - xyz[a]
            thickness = config.surface_height_spacing_factor * np.minimum(spacing[a], spacing[b])
            compatible = valid[a] & valid[b]
            compatible &= np.einsum("ni,ni->n", normals[a], normals[b]) >= config.normal_min_agreement
            compatible &= np.linalg.norm(delta, axis=1) <= surface["sampling_radius_m"]
            compatible &= np.abs(np.einsum("ni,ni->n", delta, normals[a])) <= thickness
            compatible &= np.abs(np.einsum("ni,ni->n", delta, normals[b])) <= thickness
            a, b = mapping[a[compatible]], mapping[b[compatible]]
            distinct = a != b
            rows.append(a[distinct]); columns.append(b[distinct])
    row, column = np.concatenate(rows), np.concatenate(columns)
    graph = coo_matrix((np.ones(2*len(row), dtype=bool),
                        (np.r_[row, column], np.r_[column, row])),
                       shape=(len(surface["xyz"]), len(surface["xyz"]))).tocsr()
    graph.sort_indices()
    return graph


# BEGIN radius-bounded exact search with the original nearest-neighbor tie rule.
def _nearest_within_radius(tree, points, radius):
    """Return eligible nearest measurements without searching distant regions.

    SciPy squares a Euclidean query bound and compares squared distances
    strictly, whereas pairing tests the returned square root with ``<=``.
    A normal-range squared floor prevents underflow from rejecting coincident
    points. Two outward floating-point steps cover square-root and bound-square
    rounding; the caller still applies the unchanged physical radius.

    Pruning can reorder equal-distance KD-tree nodes. Querying two neighbors
    detects an ambiguous minimum; only those eligible queries repeat the
    original unbounded k=1 search on the same tree to preserve its chosen index.
    Equal rounded square roots conservatively trigger the same treatment even
    when the internal squared distances differ. Outside-radius measurements
    need no index identity because they cannot participate in a paired element.
    """
    base = max(float(radius), math.sqrt(np.finfo(float).tiny))
    upper = math.nextafter(math.nextafter(base, math.inf), math.inf)
    distances, indices = tree.query(points, k=2, distance_upper_bound=upper, workers=1)
    distance, nearest = distances[:, 0], indices[:, 0]
    ambiguous = (distance <= radius) & (distance == distances[:, 1])
    if np.any(ambiguous):
        distance[ambiguous], nearest[ambiguous] = tree.query(points[ambiguous], workers=1)
    return distance, nearest
# END bounded search; no point, physical threshold, or tie rule is replaced.


def build_target_surface(views, *, sampling_radius_m=.03, config=None):
    """Construct geometry from two complete depth frames, independent of features.

    Binary cross-view pairing requires mutual nearest measurements, compatible
    oriented planes and bidirectional full-depth footprint support. Each view
    contributes at most one member to an element; no transitive welding occurs.
    Representatives favor the less grazing actual measurement, with canonical
    view/pixel order breaking exact ties. Appearance never chooses geometry.
    """
    config = TransferConfig() if config is None else config
    if not isinstance(config, TransferConfig):
        raise ValueError("config must be TransferConfig")
    _sampling_parameters(sampling_radius_m, sampling_radius_m)
    views = list(views)
    if (len(views) != 2 or any(not isinstance(v.get("id"), str) or not v["id"] for v in views)
            or len({v["id"] for v in views}) != 2):
        raise ValueError("Require exactly two views with distinct nonempty string ids")
    views.sort(key=lambda v: v["id"])
    arrays = [_view_arrays(view) for view in views]
    sizes = np.asarray([len(a[0]) for a in arrays], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(sizes)]
    xyz = np.concatenate([a[0] for a in arrays])
    pixels = np.concatenate([a[1] for a in arrays])
    entity = np.concatenate([a[2] for a in arrays])
    normals, spacing, valid, view_cosines = [], [], [], []
    lookups = []
    for i, (points, xy, _, lookup, _, T, _) in enumerate(arrays):
        if len(points):
            normal, ratio, _, cosine, distance, _ = _surface_geometry(
                points, xy, np.zeros(3), T[:3, 3], sampling_radius_m, config)
            good = np.isfinite(ratio) & (ratio <= config.max_planarity_ratio)
            good &= np.isfinite(distance) & (distance > 0) & (cosine >= config.min_view_cosine)
            good &= np.linalg.norm(points - T[:3, 3], axis=1) > 0
        else:
            normal, cosine, distance, good = np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0, bool)
        normals.append(normal); spacing.append(distance); valid.append(good); view_cosines.append(cosine)
        lookups.append(np.where(lookup >= 0, lookup + offsets[i], -1))
    normals, spacing, valid = np.concatenate(normals), np.concatenate(spacing), np.concatenate(valid)
    cosine = np.concatenate(view_cosines)
    surface = {"observation_xyz": xyz, "observation_pixels": pixels,
        "observation_view_index": np.repeat(np.arange(2), sizes),
        "observation_normals": normals, "observation_spacing": spacing,
        "observation_valid": valid, "observation_entity_member": entity,
        "pixel_to_observation": lookups, "view_offsets": offsets,
        "view_ids": [v["id"] for v in views], "K": np.stack([a[4] for a in arrays]),
        "T_world_camera_cv": np.stack([a[5] for a in arrays]),
        "depth_m": [a[6] for a in arrays], "sampling_radius_m": float(sampling_radius_m),
        "surface_height_spacing_factor": float(config.surface_height_spacing_factor),
        "normal_min_agreement": float(config.normal_min_agreement)}
    pair_a, pair_b = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if np.all(sizes > 0):
        a, b = xyz[:offsets[1]], xyz[offsets[1]:]
        # BEGIN use the existing pairing radius during both exact searches.
        distance, nearest_b = _nearest_within_radius(cKDTree(b), a, sampling_radius_m)
        nearest_a = _nearest_within_radius(cKDTree(a), b, sampling_radius_m)[1]
        # Missing bounded neighbors use the tree size as an index sentinel.
        # Exclude them before indexing the reverse map; preserve ascending A
        # order and the original <= gate before checking mutual selection.
        pair_a = np.flatnonzero((nearest_b < len(b)) & (distance <= sampling_radius_m))
        pair_a = pair_a[nearest_a[nearest_b[pair_a]] == pair_a]
        # END bounded association; all later plane/projection gates are unchanged.
        pair_b = offsets[1] + nearest_b[pair_a]
        delta = xyz[pair_b] - xyz[pair_a]
        thickness = config.surface_height_spacing_factor * np.minimum(spacing[pair_a], spacing[pair_b])
        good = valid[pair_a] & valid[pair_b]
        good &= np.einsum("ni,ni->n", normals[pair_a], normals[pair_b]) >= config.normal_min_agreement
        good &= np.abs(np.einsum("ni,ni->n", delta, normals[pair_a])) <= thickness
        good &= np.abs(np.einsum("ni,ni->n", delta, normals[pair_b])) <= thickness
        pair_a, pair_b = pair_a[good], pair_b[good]
        ab = project_observations(surface, xyz[pair_a], 1)
        ba = project_observations(surface, xyz[pair_b], 0)
        good = ab["supported"] & ba["supported"]
        # A nearest 3-D measurement must also be inside the other's enclosing
        # pixel cell. This prevents pairing across a disjoint nearby image edge.
        for projection, other in ((ab, pair_b), (ba, pair_a)):
            uv, xy = projection["pixels"], pixels[other]
            good &= np.all(xy >= np.floor(uv), axis=1) & np.all(xy <= np.ceil(uv), axis=1)
        pair_a, pair_b = pair_a[good], pair_b[good]
    # Union only the disjoint accepted pairs. Every singleton remains present.
    parent = np.arange(len(xyz))
    parent[pair_b] = pair_a
    roots, mapping = np.unique(parent, return_inverse=True)
    representative = roots.copy()
    choose_b = cosine[pair_b] > cosine[pair_a]
    representative[mapping[pair_a[choose_b]]] = pair_b[choose_b]
    count = len(roots)
    members = np.zeros((count, 2), bool)
    members[mapping, surface["observation_view_index"]] = True
    element_entity = np.zeros(count, bool)
    np.logical_or.at(element_entity, mapping, entity)
    surface.update(xyz=xyz[representative], normals=normals[representative],
        spacing=spacing[representative], valid=valid[representative], entity_member=element_entity,
        representative_observation=representative, observation_to_element=mapping,
        member_views=members)
    observed = np.column_stack([project_observations(surface, surface["xyz"], i)["supported"] for i in range(2)])
    surface["observed_views"] = observed | members
    surface["adjacency"] = _adjacency(surface, config)
    surface["diagnostics"] = {"raw_observation_count": int(len(xyz)), "element_count": count,
        "paired_element_count": int(len(pair_a)), "entity_element_count": int(element_entity.sum()),
        "valid_element_count": int(surface["valid"].sum()),
        "unpaired_is_physical_exclusivity": False,
        "geometry_domain": "complete_valid_depth", "representative": "actual_less_grazing_measurement"}
    return surface


def check_points(surface, points, *, tolerance_m=0):
    """Return per-view and union diagnostics for fixed non-fitting source checks.

    Support and conflict can both be true when cameras disagree; unknown is
    reserved for points for which neither view supplies support or conflict.
    The caller must retain these counts and their fixed denominator separately.
    """
    views = [project_observations(surface, points, i, tolerance_m=tolerance_m) for i in range(2)]
    supported = np.column_stack([v["supported"] for v in views])
    conflict = np.column_stack([v["conflict"] for v in views])
    support, conflicts = supported.any(1), conflict.any(1)
    return {"support": support, "conflict": conflicts, "unknown": ~(support | conflicts),
            "view_support": supported, "view_conflict": conflict,
            "view_states": np.column_stack([v["states"] for v in views]),
            "state_names": list(STATE_NAMES)}


def sample_region(surface, eligible, seed, *, sampling_radius_m, sampling_spacing_m, config=None):
    """Sample actual elements in one fixed seed plane and observed connected patch.

    Pixel edges from both cameras may connect a patch across a gap visible in
    only one image. A hole unseen by both cannot acquire a synthetic edge or
    sample. The fixed seed-plane slab prevents chains around bevels or layers.
    """
    config = TransferConfig() if config is None else config
    if not isinstance(config, TransferConfig):
        raise ValueError("config must be TransferConfig")
    _sampling_parameters(sampling_radius_m, sampling_spacing_m)
    points, normals, spacing = surface["xyz"], surface["normals"], surface["spacing"]
    eligible = np.asarray(eligible)
    if eligible.dtype != np.bool_ or eligible.shape != (len(points),):
        raise ValueError("eligible must be a Boolean element-aligned mask")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed < len(points) or not eligible[seed] or not surface["valid"][seed]):
        raise ValueError("seed must name an eligible valid surface element")
    thickness = config.surface_height_spacing_factor * spacing[seed]
    same, tangent = _same_seed_plane(points, normals, seed, thickness, config)
    members = np.flatnonzero(eligible & surface["valid"] & surface["entity_member"] & same
                             & (tangent <= sampling_radius_m * (1 + 1e-12)))
    if not np.any(members == seed):
        raise ValueError("seed must belong to the target entity")
    graph = surface["adjacency"][members][:, members]
    labels = connected_components(graph, directed=False, return_labels=True)[1]
    members = members[labels == labels[np.searchsorted(members, seed)]]
    neighborhood = points[members]
    seed_local = int(np.searchsorted(members, seed))
    selected = np.zeros(len(members), bool)
    chosen = [seed_local]
    selected[seed_local] = True
    closest = np.sum((neighborhood - neighborhood[seed_local]) ** 2, axis=1)
    closest[selected] = -np.inf
    while not selected.all():
        farthest = int(np.argmax(closest))
        if closest[farthest] < sampling_spacing_m ** 2 * (1 - 1e-12):
            break
        chosen.append(farthest); selected[farthest] = True
        closest = np.minimum(closest, np.sum((neighborhood - neighborhood[farthest]) ** 2, axis=1))
        closest[selected] = -np.inf
    mask = np.zeros(len(points), bool); mask[members] = True
    rank, condition = _rank(neighborhood, config.rank_relative_tolerance)
    return {"seed_index": int(seed), "center": points[seed].copy(), "normal": normals[seed].copy(),
        "mask": mask, "sample_indices": members[np.asarray(chosen, dtype=np.int64)],
        "diagnostics": {"observed_local_spacing_m": float(spacing[seed]),
            "surface_height_tolerance_m": float(thickness),
            "sampling_radius_m": float(sampling_radius_m), "sampling_spacing_m": float(sampling_spacing_m),
            "connectivity": "two_original_pixel_graphs_mapped_to_elements",
            "neighborhood_point_count": int(len(members)), "sample_count": len(chosen),
            "neighborhood_rank": rank, "neighborhood_condition": condition}}
