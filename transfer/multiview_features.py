"""Preserve independent cached appearance peaks on a joint observed surface.

The output is a bipartite candidate graph, not a single fused match. Costs keep
one coordinate per camera and unavailable evidence is explicitly NaN. This
module neither calls MARCO nor reads transforms, task labels, or target truth.
The original evidence arrays continue to be used unchanged by the single-view
joint transfer route.
"""

import numpy as np

from .core import _pixels
from .multiview_geometry import project_observations


def _cached_view(view, surface, view_index, query_indices, query_region):
    """Map original entity pixels into the full-depth observation domain once."""
    observation, evidence = view.get("observation"), view.get("evidence")
    if observation is None or evidence is None:
        return None
    pixels = _pixels(observation["pixels"], len(observation["xyz"]))
    lookup = surface["pixel_to_observation"][view_index]
    if (np.any(pixels < 0) or np.any(pixels[:, 0] >= lookup.shape[1])
            or np.any(pixels[:, 1] >= lookup.shape[0])):
        raise ValueError("Cached target pixels lie outside their original depth frame")
    raw = lookup[pixels[:, 1], pixels[:, 0]]
    if np.any(raw < 0):
        raise ValueError("Cached target pixels have no full-depth observation")
    queries = np.asarray(evidence["query_indices"])
    if not np.array_equal(queries, query_indices):
        raise ValueError("Both cameras must use the same frozen source query identities")
    roles = np.asarray(evidence["query_region"])
    if roles.dtype != np.bool_ or not np.array_equal(roles, query_region):
        raise ValueError("Cached query roles disagree with the frozen source demonstration")
    cost, peaks = np.asarray(evidence["appearance_cost"]), np.asarray(evidence["peak_indices"])
    supported = np.asarray(evidence["supported_target"])
    if (cost.shape != (len(queries), len(pixels)) or cost.dtype.kind not in "fiu"
            or not np.isfinite(cost).all() or np.any((cost < 0) | (cost > 1))):
        raise ValueError("Cached appearance_cost must be finite [Q,Nentity] values in [0,1]")
    if (peaks.ndim != 2 or peaks.shape[0] != len(queries) or peaks.dtype.kind not in "iu"
            or np.any(peaks < -1) or np.any(peaks >= len(pixels))):
        raise ValueError("Cached peak_indices must index entity observations or use -1")
    if supported.dtype != np.bool_ or supported.shape != (len(pixels),):
        raise ValueError("supported_target must be a Boolean entity-observation mask")
    # A cached unsupported value of one is an old storage placeholder. It must
    # never become negative evidence or an allowed fitting edge in this path.
    available_peak = peaks >= 0
    if len(pixels):
        available_peak &= supported[np.maximum(peaks, 0)]
    original_for_raw = np.full(len(surface["observation_xyz"]), -1, dtype=np.int64)
    original_for_raw[raw] = np.arange(len(raw))
    return {"raw": raw, "cost": cost, "peaks": peaks, "supported": supported,
            "available_peak": available_peak, "original_for_raw": original_for_raw}


def build_correspondence_candidates(source, views, surface, *, query_indices, query_region):
    """Return per-query allowed elements and separate camera evidence coordinates.

    An edge exists only if that source query proposed the element in at least
    one camera. Another query's peak cannot enlarge its candidate set. Surface
    elements, rather than raw camera samples, form the unique assignment
    columns. For a camera cost, use its sole member observation if present;
    otherwise use a geometrically supported deterministic projected pixel.
    No favorable-neighbor search, cross-camera mean, or minimum is performed.
    """
    query_indices, query_region = np.asarray(query_indices), np.asarray(query_region)
    if (query_indices.ndim != 1 or query_indices.dtype.kind not in "iu"
            or len(np.unique(query_indices)) != len(query_indices)
            or np.any(query_indices < 0) or np.any(query_indices >= len(source["xyz"]))):
        raise ValueError("query_indices must be unique source observation indices")
    if query_region.dtype != np.bool_ or query_region.shape != query_indices.shape:
        raise ValueError("query_region must be a Boolean query-aligned mask")
    region = np.asarray(source["region_mask"])
    if region.dtype != np.bool_ or region.shape != (len(source["xyz"]),):
        raise ValueError("source region_mask must be Boolean and point aligned")
    if not np.array_equal(region[query_indices], query_region):
        raise ValueError("query_region disagrees with the original demonstration")
    views = sorted(list(views), key=lambda view: view["id"])
    if [v["id"] for v in views] != list(surface["view_ids"]):
        raise ValueError("Views and surface must have the same two canonical camera identities")
    cached = [_cached_view(view, surface, i, query_indices, query_region) for i, view in enumerate(views)]
    mapping = surface["observation_to_element"]
    proposed, row_ids, view_ids = [], [], []
    for i, cache in enumerate(cached):
        if cache is None:
            continue
        rows, columns = np.nonzero(cache["available_peak"])
        elements = mapping[cache["raw"][cache["peaks"][rows, columns]]]
        proposed.append(elements); row_ids.append(rows); view_ids.append(np.full(len(rows), i))
    candidate_ids = np.unique(np.concatenate(proposed)) if proposed else np.empty(0, dtype=np.int64)
    q, k = len(query_indices), len(candidate_ids)
    edge_views = np.zeros((q, k, 2), bool)
    for elements, rows, cameras in zip(proposed, row_ids, view_ids):
        edge_views[rows, np.searchsorted(candidate_ids, elements), cameras] = True
    appearance = np.full((q, k, 2), np.nan, dtype=np.float64)
    available = np.zeros((q, k, 2), bool)
    cost_observations = np.full((k, 2), -1, dtype=np.int64)
    for i, cache in enumerate(cached):
        if cache is None or not k:
            continue
        start, stop = surface["view_offsets"][i:i+2]
        # Pairing guarantees one member per camera, so this mapping is unique.
        member_raw = np.full(len(surface["xyz"]), -1, dtype=np.int64)
        member_raw[mapping[start:stop]] = np.arange(start, stop)
        raw = member_raw[candidate_ids]
        no_member = raw < 0
        if no_member.any():
            projection = project_observations(surface, surface["xyz"][candidate_ids[no_member]], i)
            projected_raw = np.where(projection["supported"], projection["observation_indices"], -1)
            raw[no_member] = projected_raw
        original = np.full(k, -1, dtype=np.int64)
        known = raw >= 0
        original[known] = cache["original_for_raw"][raw[known]]
        known &= original >= 0
        supported = np.zeros(k, bool)
        supported[known] = cache["supported"][original[known]]
        appearance[:, supported, i] = cache["cost"][:, original[supported]]
        available[:, supported, i] = True
        cost_observations[supported, i] = raw[supported]
    allowed = edge_views.any(axis=2)
    return {"query_indices": query_indices.astype(np.int64, copy=False),
        "query_region": query_region.copy(), "candidate_indices": candidate_ids,
        "allowed": allowed, "edge_views": edge_views,
        "appearance_cost": appearance, "appearance_available": available,
        "cost_observation_indices": cost_observations,
        "diagnostics": {"query_count": q, "candidate_element_count": k,
            "allowed_edge_count": int(allowed.sum()),
            "view_edge_counts": edge_views.sum(axis=(0, 1)).astype(int).tolist(),
            "view_available_cost_counts": available.sum(axis=(0, 1)).astype(int).tolist(),
            "appearance_domain": "original_per_view_cached_percentile_cost",
            "missing_appearance": "NaN_with_explicit_unavailable_mask"}}
