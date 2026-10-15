"""Sample independently observed neighborhoods around one fused Sim(3) result.

Correspondence fusion and registration happen before this module is called.
Registration indexes the exact-coordinate candidate bank, whereas every output
mode indexes the concatenation of the original camera observations.  Pixels
never leave their own camera's connectivity graph.  An accepted result is a
localization preference with observed support, not a feasible placement set.
"""

import numpy as np
from scipy.spatial import cKDTree

from . import core
from .segmentation import _calibration


# BEGIN independent dual-view sampling on original camera pixel graphs.
def _index_array(value, count, upper, name):
    """Validate an index mapping; repeated destinations are allowed here."""
    raw = np.asarray(value)
    if (raw.shape != (count,) or raw.dtype.kind not in "iu"
            or np.any(raw < 0) or np.any(raw >= upper)):
        raise ValueError(f"{name} must be a valid integer mapping of length {count}")
    return raw.astype(np.int64, copy=False)


def _inputs(source, targets, prepared, bank):
    """Bind the two index domains to the exact observations used for fitting.

    An empty view is valid missing evidence, but it still has a declared ID,
    calibration, and empty point/pixel arrays.  The two-view contract never
    silently replaces a missing view with a copy of the other observation.
    """
    if not isinstance(prepared, core.PreparedAnchors) or prepared.mode != "direct":
        raise ValueError("Multiview neighborhoods require direct PreparedAnchors")
    if not isinstance(targets, (list, tuple)) or len(targets) != 2:
        raise ValueError("Exactly two declared target observations are required")
    ids = [target["id"] for target in targets]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 2:
        raise ValueError("Target observations require two distinct nonempty string IDs")
    ordered = sorted(targets, key=lambda target: target["id"])
    if list(bank["view_ids"]) != [target["id"] for target in ordered]:
        raise ValueError("Candidate-bank views must follow canonical target ID order")
    ps = core._points(source["xyz"], "source.xyz")
    source_mask = core._mask(source["region_mask"], len(ps))
    _, source_pose = _calibration(source["K"], source["T_world_camera_cv"])
    candidates = core._points(bank["xyz"], "candidate_bank.xyz")
    observed = core._points(bank["observation_xyz"], "candidate_bank.observation_xyz")
    if not np.array_equal(ps, prepared.src_xyz) or not np.array_equal(candidates, prepared.tgt_xyz):
        raise ValueError("Prepared registration belongs to different source or candidate coordinates")
    if not source_mask[prepared.input_src_anchor_idx].all():
        raise ValueError("Direct multiview registration may use only source-region queries")

    offsets = np.asarray(bank["view_offsets"])
    if (offsets.shape != (3,) or offsets.dtype.kind not in "iu" or offsets[0] != 0
            or offsets[-1] != len(observed) or np.any(np.diff(offsets.astype(np.int64)) < 0)):
        raise ValueError("view_offsets must delimit exactly two original observations")
    offsets = offsets.astype(np.int64, copy=False)
    view_indices = _index_array(bank["observation_view_index"], len(observed), 2,
                                "observation_view_index")
    local_indices = np.asarray(bank["observation_local_index"])
    if local_indices.shape != (len(observed),) or local_indices.dtype.kind not in "iu":
        raise ValueError("observation_local_index must align with original observations")
    to_candidate = _index_array(bank["observation_candidate_index"], len(observed),
                                len(candidates), "observation_candidate_index")
    first = _index_array(bank["candidate_first_observation_index"], len(candidates),
                         len(observed), "candidate_first_observation_index")
    if not np.array_equal(observed, candidates[to_candidate]):
        raise ValueError("Observation-to-candidate mapping changes measured coordinates")
    expected_first = np.full(len(candidates), len(observed), dtype=np.int64)
    np.minimum.at(expected_first, to_candidate, np.arange(len(observed)))
    if not np.array_equal(first, expected_first):
        raise ValueError("Each candidate must identify its first actual observation")
    if len(np.unique(candidates, axis=0)) != len(candidates):
        raise ValueError("The registration candidate bank must contain unique coordinates")

    views = []
    for view_index, target in enumerate(ordered):
        observation = target["observation"]
        points = core._points(observation["xyz"], "target.observation.xyz")
        pixels = core._pixels(observation["pixels"], len(points))
        _, pose = _calibration(observation["K"], observation["T_world_camera_cv"])
        begin, end = map(int, offsets[view_index:view_index + 2])
        if (end - begin != len(points) or not np.array_equal(observed[begin:end], points)
                or not np.all(view_indices[begin:end] == view_index)
                or not np.array_equal(local_indices[begin:end], np.arange(len(points)))):
            raise ValueError("Candidate-bank observation slices do not preserve original point identities")
        views.append({"id": target["id"], "points": points, "pixels": pixels,
                      "camera": pose[:3, 3], "begin": begin, "end": end})
    return ps, source_mask, source_pose[:3, 3], candidates, observed, first, view_indices, views


def _cross_view_duplicates(points, normals, eligible, view_indices, mode, *,
                           sampling_spacing_m, config):
    """Suppress another camera's redundant centers, without connecting masks.

    A center is redundant only when it lies on the selected seed's fixed
    oriented plane and within the existing metric sampling resolution of an
    actually observed member of that mode.  This does not weld point clouds,
    average positions, bridge missing pixels, or infer an unseen surface.
    """
    same_plane, _ = core._same_seed_plane(points, normals, mode["seed_index"],
        mode["diagnostics"]["surface_height_tolerance_m"], config)
    candidates = np.flatnonzero(eligible & same_plane & (view_indices != mode["view_index"]))
    suppressed = np.zeros(len(points), dtype=bool)
    if len(candidates):
        distance = cKDTree(points[mode["mask"]]).query(points[candidates], workers=1)[0]
        suppressed[candidates] = distance <= sampling_spacing_m * (1 + 1e-12)
    return suppressed


def infer_multiview_region(source, targets, prepared, candidate_bank, *,
                           sampling_radius_m=0.03, sampling_spacing_m=0.005, max_modes=3):
    """Use one already fitted transform and two independent pixel topologies.

    ``prepared.tgt_xyz`` indexes ``candidate_bank['xyz']``.  Returned
    ``tgt_anchor_idx``, mode masks, seeds, and samples instead index
    ``candidate_bank['observation_xyz']``; fitted candidate IDs remain explicit
    in diagnostics.  Calling this function never invokes a registration solver.
    Target feature maps, depths, and masks belong to the earlier fusion stage;
    this step consumes their frozen observed point/pixel identities only.
    """
    core._sampling_parameters(sampling_radius_m, sampling_spacing_m, max_modes)
    ps, source_mask, source_camera, candidates, points, first, view_indices, views = _inputs(
        source, targets, prepared, candidate_bank)
    config = prepared.config
    diagnostics = dict(prepared.diagnostics)
    diagnostics.update(method="multiview_direct_local", source_region_points=int(source_mask.sum()),
        candidate_count=len(points), registration_candidate_count=len(candidates),
        region_center_definition="arithmetic_centroid_of_source_evidence",
        sampling_radius_m=float(sampling_radius_m), sampling_spacing_m=float(sampling_spacing_m),
        max_modes=int(max_modes), source_camera_origin=source_camera.tolist(),
        target_camera_origins={view["id"]: view["camera"].tolist() for view in views},
        view_ids=[view["id"] for view in views],
        point_index_domain="candidate_bank.observation_xyz",
        registration_target_index_domain="candidate_bank.xyz",
        registration_target_candidate_indices=prepared.tgt_anchor_idx.tolist(),
        target_anchor_observation_rule="first actual observation in canonical view order",
        surface_connectivity="separate observed_pixel_8_neighbors for each camera",
        cross_view_mode_suppression="same seed plane and observed distance <= sampling_spacing_m",
        output_meaning="localization preference and observed neighborhood; not a physical feasible set")
    result = {"accepted": False, "reasons": list(prepared.reasons), "modes": [],
              "diagnostics": diagnostics, "src_anchor_idx": prepared.src_anchor_idx.copy(),
              "tgt_anchor_idx": first[prepared.tgt_anchor_idx],
              "sim3_scale": prepared.scale, "sim3_rotation": prepared.rotation,
              "sim3_translation": prepared.translation}
    if prepared.scale is None:
        return result
    if source_mask.sum() < config.min_region_points:
        result["reasons"].append("insufficient_source_region")
        return result
    region = ps[source_mask]
    source_center = region.mean(0)
    diagnostics["source_center"] = source_center.tolist()
    try:
        _, _, source_normal, source_planarity = core._plane(region, config.rank_relative_tolerance)
    except (ValueError, np.linalg.LinAlgError):
        result["reasons"].append("source_region_not_approximately_planar")
        return result
    diagnostics["source_region_planarity_ratio"] = source_planarity
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
    a, b = ps[prepared.src_anchor_idx], candidates[prepared.tgt_anchor_idx]
    try:
        relation, relation_stats = core.signature_scores(points, source_center, a, b,
            trim_fraction=config.trim_fraction, chunk_size=config.chunk_size)
    except ValueError:
        result["reasons"].append("degenerate_anchors")
        return result
    diagnostics.update(relation_stats)
    coarse_center = core.apply_sim3(source_center, prepared.scale, prepared.rotation, prepared.translation)
    wanted_normal = prepared.rotation @ source_normal
    alignment = np.linalg.norm(points - coarse_center, axis=1) / relation_stats["target_anchor_extent"]
    normals = np.zeros_like(points)
    flatness = np.full(len(points), np.inf)
    agreement = np.zeros(len(points))
    view_cosines = np.zeros(len(points))
    spacings = np.zeros(len(points))
    eligible = np.zeros(len(points), dtype=bool)
    per_view = []
    for view in views:
        begin, end = view["begin"], view["end"]
        if end > begin:
            values = core._surface_geometry(view["points"], view["pixels"], wanted_normal,
                view["camera"], sampling_radius_m, config)
            for destination, value in zip((normals, flatness, agreement, view_cosines, spacings, eligible), values):
                destination[begin:end] = value
        per_view.append({"id": view["id"], "observed_points": end - begin,
                         "valid_surface_candidates": int(eligible[begin:end].sum()),
                         "grazing_view_points": int(np.count_nonzero(
                             view_cosines[begin:end] < config.min_view_cosine))})
    diagnostics.update(valid_surface_candidates=int(eligible.sum()),
        local_surface_radius_m=float(sampling_radius_m), normal_pixel_radius=config.normal_pixel_radius,
        grazing_view_points=int(np.count_nonzero(view_cosines < config.min_view_cosine)),
        coarse_center=coarse_center.tolist(), per_view_surface_geometry=per_view)
    if not eligible.any():
        result["reasons"].append("no_valid_oriented_surface")
        return result

    # Direct-local localization uses distance to the same fitted center in
    # both cameras.  Relation scores remain diagnostics, as in the old direct
    # path, and do not introduce another optimization or model selection step.
    selected_cost = np.where(eligible, alignment, np.inf)
    center_index = int(np.argmin(selected_cost))
    surface_gap, surface_alternative = core._surface_gap(points, normals, spacings, eligible,
        selected_cost, center_index, config)
    surface_ambiguous = surface_gap is not None and surface_gap < config.min_mode_gap
    projection_limit = max(config.max_projection_spacing_factor * spacings[center_index],
                           config.max_projection_sampling_ratio * sampling_radius_m)
    projection_distance = float(np.linalg.norm(points[center_index] - coarse_center))
    diagnostics.update(selected_different_surface_gap=surface_gap,
        selected_alternate_surface_index=surface_alternative, surface_ambiguity=bool(surface_ambiguous),
        preferred_center_index=center_index, center_signature_score=float(relation[center_index]),
        center_alignment_score=float(alignment[center_index]), center_to_coarse_m=projection_distance,
        region_normal_agreement=float(agreement[center_index]), target_view_cosine=float(view_cosines[center_index]),
        observed_surface_spacing_m=float(spacings[center_index]),
        center_projection_limit_m=float(projection_limit), center_projection_distance_m=projection_distance)
    if surface_ambiguous:
        result["reasons"].append("surface_ambiguity")
    if projection_distance > projection_limit:
        result["reasons"].append("predicted_center_without_nearby_surface")
    retained = core.retained_anchor_rank(points[center_index], source_center, a, b, config.trim_fraction)
    diagnostics.update(retained_source_rank=retained["source_rank"], retained_target_rank=retained["target_rank"],
        retained_source_condition=retained["source_condition"], retained_target_condition=retained["target_condition"],
        retained_anchor_indices=retained["kept_anchor_indices"])
    primary_plane, primary_tangent = core._same_seed_plane(points, normals, center_index,
        config.surface_height_spacing_factor * spacings[center_index], config)
    distant = np.flatnonzero(eligible & primary_plane & (primary_tangent > 2 * sampling_radius_m))
    position_index = int(distant[np.argmin(selected_cost[distant])]) if len(distant) else None
    position_gap = float(selected_cost[position_index] - selected_cost[center_index]) if position_index is not None else None
    diagnostics.update(same_plane_position_gap=position_gap, same_plane_position_index=position_index,
        position_preference_ambiguous=bool(position_gap is not None and position_gap < config.min_mode_gap))

    remaining = eligible.copy()
    next_seed = center_index
    for mode_number in range(max_modes):
        view_index = int(view_indices[next_seed])
        view = views[view_index]
        begin, end = view["begin"], view["end"]
        local_seed = next_seed - begin
        local_mode = core.sample_neighborhood(view["points"], eligible[begin:end], normals[begin:end],
            spacings[begin:end], local_seed, target_pixels=view["pixels"],
            sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m, config=config)
        local_mask, local_samples = local_mode["mask"], local_mode["sample_indices"]
        global_mask = np.zeros(len(points), dtype=bool)
        global_mask[begin:end] = local_mask
        mode = dict(local_mode)
        mode.update(seed_index=next_seed, mask=global_mask, sample_indices=local_samples + begin,
            view_id=view["id"], view_index=view_index, local_seed_index=int(local_seed),
            sample_local_indices=local_samples.copy(), seed_pixel=view["pixels"][local_seed].copy(),
            sample_pixels=view["pixels"][local_samples].copy(),
            score=float(selected_cost[next_seed]), relation_score=float(relation[next_seed]),
            type="primary" if mode_number == 0 else (
                "same_plane_localization" if primary_plane[next_seed] else "different_surface"))
        mode["diagnostics"].update(normal_agreement=float(agreement[next_seed]),
            view_cosine=float(view_cosines[next_seed]), raw_relation_min=float(relation[global_mask].min()),
            raw_relation_max=float(relation[global_mask].max()))
        result["modes"].append(mode)
        # Reuse the old connectivity-aware suppression inside this camera.
        # Cross-camera suppression only removes duplicate mode centers; each
        # returned mask remains exactly one independently observed component.
        remaining[begin:end] &= ~core._overlapping_seed_neighborhood(view["points"], view["pixels"],
            normals[begin:end], eligible[begin:end], local_mode, sampling_radius_m, config)
        remaining &= ~_cross_view_duplicates(points, normals, eligible, view_indices, mode,
            sampling_spacing_m=sampling_spacing_m, config=config)
        if mode_number == 0 and surface_ambiguous and max_modes > 1:
            next_seed = surface_alternative
            continue
        if not remaining.any():
            break
        next_seed = int(np.argmin(np.where(remaining, selected_cost, np.inf)))
    result["modes"][1:] = sorted(result["modes"][1:], key=lambda mode: (mode["score"], mode["seed_index"]))
    primary = result["modes"][0]
    if (primary["diagnostics"]["neighborhood_point_count"] < config.min_neighborhood_points
            or primary["diagnostics"]["neighborhood_rank"] < 2):
        result["reasons"].append("little_visible_target_support")
    diagnostics.update(returned_modes=len(result["modes"]), sampling_scale_applied=False)
    result["reasons"] = list(dict.fromkeys(result["reasons"]))
    result["accepted"] = not result["reasons"]
    return result
# END independent dual-view sampling on original camera pixel graphs.
