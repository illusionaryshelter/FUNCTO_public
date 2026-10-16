"""Geometry contract for the independent third (dual-view) transfer method.

The old and joint single-view methods are read-only dependencies.  Complete
depth provides position evidence, including contradictions; entity masks and
oriented local planes determine whether a view supports a query.  Missing
normals never remove a frozen source query or hide a depth contradiction.
"""

import numpy as np
from scipy.sparse.csgraph import connected_components

from ..core import (TransferConfig, _indices, _mask, _plane, _same_seed_plane,
                    _sampling_parameters, _surface_geometry, fps_indices)
from ..multiview_geometry import (STATE_NAMES, _view_arrays, build_target_surface,
                                  project_observations, sample_region)
from ..sim3 import _points, apply_sim3, rms_size


def _config(config):
    """Use the existing frozen geometry parameters, without dataset switches."""
    config = TransferConfig() if config is None else config
    if not isinstance(config, TransferConfig):
        raise ValueError("config must be TransferConfig")
    return config


def _normal_valid(normals, count):
    """NaN/zero normals are unavailable evidence, not deletable query rows."""
    normals = np.asarray(normals, dtype=float)
    if normals.shape != (count, 3):
        raise ValueError("normals must be a point-aligned [N,3] array")
    valid = np.isfinite(normals).all(axis=1)
    valid &= np.isclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-6)
    return normals, valid


def prepare_source(source, depth_m, *, query_indices, query_region,
                   sampling_radius_m=.03, config=None):
    """Freeze queries and independent checks, reading normals from full depth.

    Independent checks preserve the previous source-only rule: up to 32 entity
    points inside and outside the demonstration, excluding every query.  The
    demonstration queries themselves remain legal outside a missed entity mask.
    Region/model insufficiency is returned in ``reasons``; malformed calibration
    or inconsistent depth coordinates is a contract error and raises ValueError.
    """
    geo = _config(config)
    _sampling_parameters(sampling_radius_m, sampling_radius_m)
    points = _points(source["xyz"], "source.xyz")
    region = _mask(source["region_mask"], len(points))
    entity = _mask(source["entity_member"], len(points))
    query_ids = _indices(query_indices, len(points), "query_indices").copy()
    query_region = _mask(query_region, len(query_ids)).copy()
    if (not np.array_equal(query_region, region[query_ids])
            or np.any(~query_region & ~entity[query_ids])):
        raise ValueError("Query provenance disagrees with source region/entity")
    unused = entity.copy()
    unused[query_ids] = False
    groups = []
    for membership in (region, ~region):
        ids = np.flatnonzero(unused & membership)
        groups.append(ids[fps_indices(points[ids], 32)] if len(ids) else ids)
    check_ids = np.concatenate(groups)

    # Reconstruct all positive depth pixels before estimating any local plane.
    # Segmenting this cloud first would invent normal gaps at the SAM boundary.
    xyz, _, _, lookup, _, pose, _ = _view_arrays(dict(
        depth_m=depth_m, K=source["K"],
        T_world_camera_cv=source["T_world_camera_cv"], observation=source))
    yy, xx = np.nonzero(lookup >= 0)
    full_pixels = np.column_stack((xx, yy))
    if len(xyz):
        normals, ratios, _, cosines, spacing, _ = _surface_geometry(
            xyz, full_pixels, np.zeros(3), pose[:3, 3], sampling_radius_m, geo)
        valid = np.isfinite(ratios) & (ratios <= geo.max_planarity_ratio)
        valid &= np.isfinite(spacing) & (spacing > 0)
        valid &= (cosines >= geo.min_view_cosine)
        valid &= np.linalg.norm(xyz - pose[:3, 3], axis=1) > 0
    else:
        normals, valid = np.empty((0, 3)), np.empty(0, bool)
    source_pixels = np.asarray(source["pixels"])
    observation_ids = lookup[source_pixels[:, 1], source_pixels[:, 0]]
    source_normals = normals[observation_ids].copy()
    source_valid = valid[observation_ids]
    source_normals[~source_valid] = np.nan

    center = points[region].mean(0) if region.any() else np.full(3, np.nan)
    normal, reasons = np.full(3, np.nan), []
    planarity = None
    if region.sum() < geo.min_region_points:
        reasons.append("insufficient_source_region_points")
    else:
        try:
            _, _, normal, planarity = _plane(points[region], geo.rank_relative_tolerance)
            if planarity > geo.max_planarity_ratio:
                reasons.append("source_region_not_approximately_planar")
            view = pose[:3, 3] - center
            length = float(np.linalg.norm(view))
            cosine = float(normal @ view / max(length, np.finfo(float).tiny))
            if length <= 0 or abs(cosine) < geo.min_view_cosine:
                reasons.append("source_normal_orientation_unobservable")
                normal[:] = np.nan
            elif cosine < 0:
                normal = -normal
        except (ValueError, np.linalg.LinAlgError):
            reasons.append("source_region_not_approximately_planar")
            normal[:] = np.nan
    size = rms_size(points[entity]) if entity.any() else 0.
    if not np.isfinite(size) or size <= np.finfo(float).tiny:
        reasons.append("degenerate_source_entity_geometry")
    diagnostics = dict(query_count=len(query_ids),
        region_query_count=int(query_region.sum()),
        context_query_count=int((~query_region).sum()),
        query_normal_valid_count=int(source_valid[query_ids].sum()),
        check_point_count=len(check_ids),
        check_region_count=int(region[check_ids].sum()),
        check_context_count=int((~region[check_ids]).sum()),
        check_normal_valid_count=int(source_valid[check_ids].sum()),
        check_query_region=region[check_ids].tolist(),
        source_region_points=int(region.sum()), source_planarity_ratio=planarity,
        full_depth_point_count=len(xyz), source_entity_rms_m=float(size),
        fixed_query_denominator=len(query_ids), normal_domain="complete_source_depth",
        quality_warnings=[] if len(check_ids) else ["no_independent_source_check_points"])
    return dict(query_indices=query_ids, query_region=query_region,
        query_xyz=points[query_ids].copy(), query_normals=source_normals[query_ids],
        query_normal_valid=source_valid[query_ids].copy(),
        check_indices=check_ids, check_xyz=points[check_ids].copy(),
        check_normals=source_normals[check_ids],
        check_normal_valid=source_valid[check_ids].copy(),
        center=center, normal=normal, size=float(size),
        diagnostics=diagnostics, reasons=reasons)


def project_oriented(surface, points, normals, view_index):
    """Add same-view plane, orientation and full entity-footprint eligibility.

    ``position_supported`` is the unchanged complete-depth classification with
    zero extra tolerance. ``supported`` requires all geometry/entity gates in
    this one view. All positive bilinear original-pixel corners need entity
    membership; a zero-weight corner is irrelevant. Source normal failure and
    missing entity labels do not change ``front_of_observed_surface`` conflicts.
    """
    points = _points(points, "projection points")
    normals, normal_valid = _normal_valid(normals, len(points))
    result = project_observations(surface, points, view_index, tolerance_m=0)
    nearest = result["observation_indices"]
    present = nearest >= 0
    plane_valid = np.zeros(len(points), bool)
    oriented = np.zeros(len(points), bool)
    entity = np.zeros(len(points), bool)
    plane_valid[present] = surface["observation_valid"][nearest[present]]
    active = np.flatnonzero(present & plane_valid & normal_valid)
    oriented[active] = np.einsum("ij,ij->i", normals[active],
        surface["observation_normals"][nearest[active]]) >= surface["normal_min_agreement"]
    active = np.flatnonzero(present)
    if len(active):
        uv = result["pixels"][active]
        low, high = np.floor(uv).astype(int), np.ceil(uv).astype(int)
        fraction = uv - low
        corners = np.stack((low, np.column_stack((high[:, 0], low[:, 1])),
            np.column_stack((low[:, 0], high[:, 1])), high), axis=1)
        fx, fy = fraction[:, 0], fraction[:, 1]
        weights = np.column_stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy))
        lookup = surface["pixel_to_observation"][view_index]
        ids = lookup[corners[:, :, 1], corners[:, :, 0]]
        labeled = np.zeros(ids.shape, bool)
        labeled[ids >= 0] = surface["observation_entity_member"][ids[ids >= 0]]
        entity[active] = np.all((weights <= 0) | labeled, axis=1)
    position = result.pop("supported")
    supported = position & plane_valid & oriented & entity
    result.update(position_supported=position, plane_valid=plane_valid,
        oriented_supported=oriented, entity_supported=entity, supported=supported,
        query_normal_valid=normal_valid, unknown=~(supported | result["conflict"]))
    return result


def geometry_support(surface, points, normals):
    """Keep both cameras' evidence; support and conflict may coexist."""
    views = [project_oriented(surface, points, normals, i) for i in range(2)]
    view_support = np.column_stack([view["supported"] for view in views])
    view_conflict = np.column_stack([view["conflict"] for view in views])
    support, conflict = view_support.any(1), view_conflict.any(1)
    return dict(support=support, conflict=conflict, unknown=~(support | conflict),
        view_support=view_support, view_conflict=view_conflict, views=views,
        view_states=np.column_stack([view["states"] for view in views]),
        state_names=list(STATE_NAMES))


def _sampleable(mode, config):
    """The neighborhood, rather than sparse display samples, supplies rank."""
    diagnostic = mode["diagnostics"]
    return (diagnostic["neighborhood_point_count"] >= max(6, config.min_neighborhood_points)
            and diagnostic["neighborhood_rank"] >= 2)


def _same_tied_patch(surface, eligible, tied, config):
    """Resolve IDs only after tied seeds share one fixed plane and graph patch."""
    if len(tied) <= 1:
        return True
    first = int(tied[0])
    same, _ = _same_seed_plane(surface["xyz"], surface["normals"], first,
        config.surface_height_spacing_factor * surface["spacing"][first], config)
    members = np.flatnonzero(eligible & same)
    if not np.isin(tied, members).all():
        return False
    labels = connected_components(surface["adjacency"][members][:, members],
        directed=False, return_labels=True)[1]
    tie_labels = labels[np.searchsorted(members, tied)]
    return bool(np.all(tie_labels == tie_labels[0]))


def region_for_transform(surface, source_pack, transform, *, sampling_radius_m=.03,
                         sampling_spacing_m=.005, config=None):
    """Project a supported mapped center once, then sample its nearest real seed.

    A farther seed is never tried to rescue an unsampleable nearest point. Exact
    distance ties across unproven support patches remain an explicit ambiguity;
    the inference module must retain that transform's competing score interval.
    """
    geo = _config(config)
    _sampling_parameters(sampling_radius_m, sampling_spacing_m)
    center = apply_sim3(source_pack["center"], *transform)
    normal = np.asarray(transform[1]) @ source_pack["normal"]
    support = geometry_support(surface, center[None], normal[None])
    if support["conflict"][0]:
        return None, "center_geometry_model_conflict"
    if not support["support"][0]:
        return None, "mapped_center_without_oriented_entity_support"
    eligible = surface["valid"] & surface["entity_member"]
    eligible = eligible & (surface["normals"] @ normal >= geo.normal_min_agreement)
    projection_limit = np.maximum(geo.max_projection_spacing_factor * surface["spacing"],
        geo.max_projection_sampling_ratio * sampling_radius_m)
    distance2 = np.sum((surface["xyz"] - center) ** 2, axis=1)
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        return None, "no_valid_surface_near_supported_mapping"
    nearest_distance2 = distance2[candidates].min()
    tied = candidates[distance2[candidates] == nearest_distance2]
    # Apply the displacement gate to the nearest observations, rather than
    # admitting a farther, coarser observation with a more permissive slab.
    tied = tied[nearest_distance2 <= projection_limit[tied] ** 2]
    if not len(tied):
        return None, "center_projection_exceeds_observed_limit"
    if not _same_tied_patch(surface, eligible, tied, geo):
        return None, "center_projection_ambiguous"
    seed = int(tied[0])
    mode = sample_region(surface, eligible, seed, sampling_radius_m=sampling_radius_m,
        sampling_spacing_m=sampling_spacing_m, config=geo)
    if not _sampleable(mode, geo):
        return None, "little_visible_target_support"
    mode.update(mapped_center=center, mapped_normal=normal)
    mode["diagnostics"].update(center_projection_distance_m=float(np.sqrt(nearest_distance2)),
        center_projection_limit_m=float(projection_limit[seed]),
        nearest_seed_exact_tie_count=len(tied))
    return mode, None


def consensus_region(surface, modes, centers, normals, *, sampling_radius_m=.03,
                     sampling_spacing_m=.005, config=None):
    """Sample only an actual intersection, preserving every connected component.

    Candidate seeds use sum of squared distances to all mapped centers, then
    stable surface ID. This chooses a reproducible display/sample representative
    and does not erase the remaining common components or identify a unique T.
    """
    geo = _config(config)
    _sampling_parameters(sampling_radius_m, sampling_spacing_m)
    centers = _points(centers, "consensus centers")
    normals, normal_valid = _normal_valid(normals, len(centers))
    if not len(centers) or not normal_valid.all():
        raise ValueError("Require one valid center and normal for each mode")
    eligible = surface["valid"] & surface["entity_member"]
    consumed = 0
    for mode in modes:
        if consumed >= len(centers):
            raise ValueError("More modes than consensus centers")
        normal = normals[consumed]
        eligible = eligible & _mask(mode["mask"], len(eligible))
        eligible &= surface["normals"] @ normal >= geo.normal_min_agreement
        consumed += 1
    if consumed != len(centers):
        raise ValueError("Fewer modes than consensus centers")
    members = np.flatnonzero(eligible)
    if not len(members):
        return None, "empty_common_observed_region"
    count, labels = connected_components(surface["adjacency"][members][:, members],
        directed=False, return_labels=True)
    distance2 = np.zeros(len(members))
    for center in centers:
        distance2 += np.sum((surface["xyz"][members] - center) ** 2, axis=1)
    for local in np.argsort(distance2, kind="stable"):
        seed = int(members[local])
        mode = sample_region(surface, eligible, seed, sampling_radius_m=sampling_radius_m,
            sampling_spacing_m=sampling_spacing_m, config=geo)
        if not _sampleable(mode, geo):
            continue
        # The full common mask is distinct from the selected connected seed
        # neighborhood. No caller may mistake display choice for uniqueness.
        mode.update(common_mask=eligible.copy(), common_indices=members,
                    common_component_labels=labels)
        mode["diagnostics"].update(common_point_count=len(members),
            common_component_count=int(count),
            common_component_sizes=np.bincount(labels, minlength=count).tolist(),
            selected_common_component=int(labels[local]),
            representative_center_squared_distance_sum=float(distance2[local]),
            common_hypothesis_count=consumed)
        return mode, None
    return None, "common_region_not_sampleable"
