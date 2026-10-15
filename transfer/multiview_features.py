"""Fuse two independent MARCO response maps on observed world-space points.

This is an explicit alternative matching rule, not MARCO's native Gaussian
decoder and not a calibrated probability model. Every response is ranked over
its own fixed allowed feature cells *before* interpolation. Unknown views add
zero to a fixed two-view mean. Complete maps survive until this fusion; only
then does each source query select one target for the unchanged Sim(3) solvers.

No target annotation, object category, fitted transform, or registration result
is accepted here. All point identities and rejected raw queries are retained.
"""

import numpy as np
from scipy.stats import rankdata

from .core import TransferConfig, fps_indices
from .experiment import feature_observation
from .features import sample_descriptors


# BEGIN independent multiview rank fusion; existing native matcher unchanged.
def centered_rank_maps(logits, allowed):
    """Return tie-aware centered empirical ranks on a fixed feature-grid domain.

    Forbidden cells are zero placeholders, always excluded by the interpolation
    mask. Constant allowed responses have exactly zero evidence, even when the
    constant raw dot product is positive or negative. Ranks are unchanged by a
    strictly increasing transformation of the allowed raw responses.
    """
    values, mask = np.asarray(logits), np.asarray(allowed)
    if (values.ndim != 3 or values.dtype.kind not in 'fiu'
            or mask.dtype != np.bool_ or mask.shape != values.shape[1:]):
        raise ValueError('Require numeric [Q,Hf,Wf] responses and a Boolean [Hf,Wf] domain')
    if not np.isfinite(values).all():
        raise ValueError('MARCO responses must be finite')
    output = np.zeros(values.shape, dtype=np.float64)
    count = int(mask.sum())
    if count:
        ranks = rankdata(values[:, mask], method='average', axis=1)
        output[:, mask] = 2.0 * ((ranks - 0.5) / count) - 1.0
    return output


def rank_sampling_plan(feature_xy, allowed):
    """Precompute allowed-neighbor bilinear weights without clamping predictions.

    A prediction outside the closed feature-grid rectangle has no support.
    Safe clipped indices below are used only to gather arrays; their weights
    remain zero outside that rectangle or outside the fixed allowed domain.
    No nearest allowed cell replaces a missing interpolation neighborhood.
    """
    points, mask = np.asarray(feature_xy, dtype=float), np.asarray(allowed)
    if points.ndim != 2 or points.shape[1] != 2 or mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Require feature [N,2] coordinates and a Boolean grid')
    height, width = mask.shape
    if min(height, width) < 2:
        raise ValueError('Feature grid must have at least two cells on each axis')
    finite = np.isfinite(points).all(axis=1)
    inside = finite & np.all(points >= 0, axis=1) & np.all(points <= [width-1, height-1], axis=1)
    safe = np.where(inside[:, None], points, 0.0)
    lower = np.floor(safe).astype(np.int64)
    fraction = safe - lower
    indices, weights = [], []
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, y = lower[:, 0] + dx, lower[:, 1] + dy
        valid = inside & (x < width) & (y < height)
        x, y = np.minimum(x, width-1), np.minimum(y, height-1)
        weight = (fraction[:, 0] if dx else 1-fraction[:, 0])
        weight = weight * (fraction[:, 1] if dy else 1-fraction[:, 1])
        weight = weight * (valid & mask[y, x])
        indices.append(y*width+x)
        weights.append(weight)
    indices, weights = np.stack(indices, axis=1), np.stack(weights, axis=1)
    denominator = weights.sum(axis=1)
    supported = denominator > 0
    weights = np.divide(weights, denominator[:, None], out=np.zeros_like(weights),
                        where=supported[:, None])
    return {'indices': indices, 'weights': weights, 'supported': supported,
            'grid_shape': mask.shape}


def sample_rank_maps(ranks, plan):
    """Interpolate ranked evidence, retaining zero evidence for unknown points."""
    values = np.asarray(ranks, dtype=float)
    if values.ndim != 3 or values.shape[1:] != tuple(plan['grid_shape']):
        raise ValueError('Rank map shape differs from its interpolation plan')
    if not np.isfinite(values).all():
        raise ValueError('Rank evidence must be finite')
    flat = values.reshape(len(values), -1)
    # Accumulate one neighbor at a time instead of allocating [Q,N,4].
    output = np.zeros((len(values), len(plan['indices'])), dtype=np.float64)
    for corner in range(4):
        output += flat[:, plan['indices'][:, corner]] * plan['weights'][None, :, corner]
    return output


def select_fused_candidates(scores):
    """Select unique positive maxima; exact ties between distinct points abstain.

    Candidate IDs already refer to exact-XYZ-unique points. A positive maximum
    means above-centered ordinal evidence, not a probability of correctness.
    The unconstrained best index is saved for diagnostics even after abstention.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Fused evidence must be finite [Q,Ncandidate]')
    count = len(values)
    if not values.shape[1]:
        return {'best': np.full(count, -1, dtype=np.int64),
                'target': np.full(count, -1, dtype=np.int64), 'valid': np.zeros(count, bool),
                'score': np.zeros(count), 'ties': np.zeros(count, dtype=np.int64),
                'margin': np.zeros(count), 'reason': np.full(count, 'no_observed_candidate', dtype='<U40')}
    best = values.argmax(axis=1)
    maximum = values[np.arange(count), best]
    ties = np.sum(values == maximum[:, None], axis=1)
    valid = (maximum > 0) & (ties == 1)
    reasons = np.full(count, '', dtype='<U40')
    reasons[maximum <= 0] = 'no_positive_fused_evidence'
    reasons[(maximum > 0) & (ties > 1)] = 'ambiguous_fused_maximum'
    second = (np.partition(values, -2, axis=1)[:, -2]
              if values.shape[1] > 1 else maximum)
    return {'best': best, 'target': np.where(valid, best, -1), 'valid': valid,
            'score': maximum, 'ties': ties, 'margin': maximum-second, 'reason': reasons}


def unique_target_pairs(raw_source_idx, selection):
    """Retain the strongest query at each target; preserve the first query on ties."""
    sources = np.asarray(raw_source_idx)
    if (sources.ndim != 1 or sources.dtype.kind not in 'iu'
            or len(sources) != len(selection['target']) or len(np.unique(sources)) != len(sources)):
        raise ValueError('Raw source identities must be unique integer query IDs')
    kept = []
    target, valid, scores = selection['target'], selection['valid'], selection['score']
    for index in np.unique(target[valid]):
        candidates = np.flatnonzero(valid & (target == index))
        kept.append(int(candidates[np.argmax(scores[candidates])]))
    return np.asarray(kept, dtype=np.int64)


def _feature(value, name):
    """Validate cached feature tensors without loading or modifying a model."""
    feature = np.asarray(value)
    if (feature.dtype != np.float32 or feature.ndim != 4 or feature.shape[0] != 1
            or min(feature.shape[1:]) < 1 or min(feature.shape[-2:]) < 2
            or not np.isfinite(feature).all()):
        raise ValueError(f'{name} must be finite FP32 [1,C,Hf,Wf]')
    return feature


def fuse_correspondences(source, source_feature, source_crop_xyxy, targets, *,
                         config=None, sampling_radius_m=.03, count=160):
    """Build one frozen two-view correspondence set for all registration backends.

    ``targets`` contains exactly two dictionaries with unique string ``id`` and
    ``observation``, ``depth_m``, ``selected_mask``, ``crop_xyxy``, ``feature``.
    An explicitly unavailable view has no observed points, an empty selected
    mask and ``feature=None``; its crop may also be None. It supplies zero
    evidence while retaining its place in the fixed two-view denominator.
    The geometry module defines observed candidate provenance and cross-view
    surface association. It does not silently discard an owner-observed point
    because another discrete depth map disagrees at a subpixel projection.

    Returned fitting target indices address ``bank['xyz']``. Raw query records
    preserve invalid/ambiguous queries and query-to-fitting-row identities.
    ``bank['observation_xyz']`` is a different, explicitly mapped domain used by
    the final per-camera neighborhood sampler and held-out evaluation.
    """
    import torch
    from local_models.marco import RESOLUTION, QUERY_BATCH_SIZE, allowed_grid, resize_scales
    from .multiview_geometry import associate_view, build_candidate_bank

    cfg = TransferConfig() if config is None else config
    if not isinstance(cfg, TransferConfig):
        raise ValueError('config must be TransferConfig')
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('count must be a positive integer')
    if not np.isscalar(sampling_radius_m) or not np.isfinite(sampling_radius_m) or sampling_radius_m <= 0:
        raise ValueError('sampling_radius_m must be finite and positive')
    if len(targets) != 2:
        raise ValueError('This independent path requires exactly two target views')
    views = sorted(targets, key=lambda item: item['id'])
    bank = build_candidate_bank(views)
    source_map = _feature(source_feature, 'source_feature')
    region = np.asarray(source['region_mask'])
    if region.dtype != np.bool_ or region.shape != (len(source['xyz']),):
        raise ValueError('Source region must be a point-aligned Boolean mask')
    region_indices = np.flatnonzero(region)
    chosen = fps_indices(np.asarray(source['xyz'])[region_indices], count)
    source_ids = region_indices[chosen]
    source_chart = feature_observation(source, source_crop_xyxy)
    descriptors = (sample_descriptors(source_map, source_chart['pixels'][source_ids],
                                      source_chart['object_mask'].shape[::-1])
                   if len(source_ids) else np.empty((0, source_map.shape[1]), dtype=np.float32))
    n, candidates = len(source_ids), len(bank['xyz'])
    candidate_supported = np.zeros((candidates, 2), bool)
    projected = np.full((candidates, 2, 2), np.nan)
    states = np.zeros((candidates, 2), dtype=np.int16)
    prepared_views, view_diagnostics = [], []
    for view_index, view in enumerate(views):
        association = associate_view(bank, view, config=cfg, sampling_radius_m=sampling_radius_m)
        pixels = association['pixels']
        projected[:, view_index] = pixels
        states[:, view_index] = association['state']
        # A model-preparation failure is declared by the input loader, never
        # inferred from an exception or repaired by another prompt/crop. There
        # is no feature tensor or crop to sample for a genuinely empty view.
        if view['feature'] is None:
            if len(view['observation']['xyz']) or np.any(view['selected_mask']):
                raise ValueError('An unavailable feature view must have no observations and an empty selected mask')
            prepared_views.append((None, None, None, None))
            view_diagnostics.append({'id': view['id'], 'appearance_available': False,
                'unavailable_reasons': list(view.get('missing_target_reasons', [])),
                'unavailable_stage': view.get('missing_target_stage'),
                'allowed_feature_cells': 0, 'surface_association': association['diagnostics'],
                'state_names': association['state_names'], 'appearance_supported_candidates': 0,
                'geometry_supported_without_feature_neighbors': 0})
            continue
        feature = _feature(view['feature'], 'target_feature')
        if feature.shape[1] != source_map.shape[1]:
            raise ValueError('Source and target descriptors have different channel counts')
        chart = feature_observation(view['observation'], view['crop_xyxy'])
        mask = np.zeros_like(chart['object_mask'], dtype=bool)
        xy = chart['pixels']
        mask[xy[:, 1], xy[:, 0]] = True
        allowed = allowed_grid(mask, feature.shape[-2:])
        # Invert the native TARGET OUTPUT map, not the distinct source-sampler
        # Wf/840 convention. Both conventions intentionally match existing code.
        height, width = feature.shape[-2:]
        feature_xy = ((pixels - np.asarray(view['crop_xyxy'][:2]))
                      * resize_scales(mask.shape[::-1])
                      * np.array([width-1, height-1]) / (RESOLUTION-1))
        plan = rank_sampling_plan(feature_xy, allowed)
        supported = association['supported'] & plan['supported']
        candidate_supported[:, view_index] = supported
        prepared_views.append((torch.from_numpy(feature), allowed, plan, supported))
        view_diagnostics.append({'id': view['id'], 'appearance_available': True,
            'allowed_feature_cells': int(allowed.sum()),
            'surface_association': association['diagnostics'], 'state_names': association['state_names'],
            'appearance_supported_candidates': int(supported.sum()),
            'geometry_supported_without_feature_neighbors': int(np.sum(association['supported'] & ~plan['supported']))})

    selection_parts = {name: [] for name in ('best', 'target', 'valid', 'score', 'ties', 'margin', 'reason')}
    raw_view_scores = np.zeros((n, 2))
    per_view_best = np.full((n, 2), -1, dtype=np.int64)
    per_view_scores = np.zeros((n, 2))
    per_view_ties = np.zeros((n, 2), dtype=np.int64)
    raw_ranges = np.zeros((n, 2, 2))
    distinct_values = np.zeros((n, 2), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, n, QUERY_BATCH_SIZE):
            stop = min(start+QUERY_BATCH_SIZE, n)
            desc = torch.from_numpy(descriptors[start:stop])[None]
            scores_by_view = []
            for view_index, (feature, allowed, plan, supported) in enumerate(prepared_views):
                if feature is None:
                    scores_by_view.append(np.zeros((stop-start, candidates), dtype=np.float64))
                    per_view_ties[start:stop, view_index] = candidates
                    continue
                logits = torch.bmm(desc, feature.flatten(2)).reshape(stop-start, *feature.shape[-2:]).numpy()
                ranks = centered_rank_maps(logits, allowed)
                scores = sample_rank_maps(ranks, plan)
                scores[:, ~supported] = 0.0
                scores_by_view.append(scores)
                selected = select_fused_candidates(scores)
                per_view_best[start:stop, view_index] = selected['target']
                per_view_scores[start:stop, view_index] = selected['score']
                per_view_ties[start:stop, view_index] = selected['ties']
                if allowed.any():
                    allowed_values = logits[:, allowed]
                    raw_ranges[start:stop, view_index, 0] = allowed_values.min(axis=1)
                    raw_ranges[start:stop, view_index, 1] = allowed_values.max(axis=1)
                    distinct_values[start:stop, view_index] = [len(np.unique(row)) for row in allowed_values]
            fused = (scores_by_view[0] + scores_by_view[1]) / 2.0
            selected = select_fused_candidates(fused)
            for key in selection_parts:
                selection_parts[key].append(selected[key])
            if candidates:
                for view_index in range(2):
                    raw_view_scores[start:stop, view_index] = scores_by_view[view_index][
                        np.arange(stop-start), selected['best']]

    if n:
        selected = {key: np.concatenate(parts) for key, parts in selection_parts.items()}
    else:
        selected = select_fused_candidates(np.zeros((0, candidates)))
    keep = unique_target_pairs(source_ids, selected)
    retained = np.zeros(n, bool)
    retained[keep] = True
    fitting_row = np.full(n, -1, dtype=np.int64)
    fitting_row[keep] = np.arange(len(keep))
    best_supported = np.zeros((n, 2), bool)
    if candidates:
        best_supported[:] = candidate_supported[selected['best']]
    return {
        'bank': bank,
        'src_anchor_idx': source_ids[keep], 'tgt_anchor_idx': selected['target'][keep],
        'scores': selected['score'][keep], 'raw_source_idx': source_ids,
        'raw_target_idx': selected['target'], 'raw_best_target_idx': selected['best'],
        'raw_valid': selected['valid'], 'raw_scores': selected['score'],
        'raw_tie_count': selected['ties'], 'raw_score_margin': selected['margin'],
        'raw_reasons': selected['reason'], 'raw_retained': retained, 'raw_fitting_row': fitting_row,
        'raw_view_scores': raw_view_scores, 'raw_view_supported': best_supported,
        'per_view_best_candidate_idx': per_view_best, 'per_view_best_scores': per_view_scores,
        'per_view_best_tie_count': per_view_ties, 'raw_dot_ranges': raw_ranges,
        'raw_dot_distinct_values': distinct_values,
        'candidate_view_supported': candidate_supported, 'candidate_projection_pixels': projected,
        'candidate_view_states': states,
        'diagnostics': {
            'matching_rule': 'complete native raw-dot maps -> per-view centered midranks -> observed 3D candidate mean',
            'score_semantics': 'ordinal evidence; not probability or calibrated confidence',
            'view_ids': bank['view_ids'], 'view_count': 2, 'fixed_view_denominator': 2,
            'query_count': n, 'candidate_count': candidates,
            'valid_query_count': int(selected['valid'].sum()), 'retained_count': len(keep),
            'duplicate_target_discard_count': int(selected['valid'].sum()-len(keep)),
            'ambiguous_query_count': int(np.sum(selected['reason'] == 'ambiguous_fused_maximum')),
            'no_positive_evidence_count': int(np.sum(selected['reason'] == 'no_positive_fused_evidence')),
            'no_source_region': not bool(n), 'no_observed_candidate': not bool(candidates),
            'candidate_supported_by_both_views': int(np.sum(candidate_supported.all(axis=1))),
            'retained_supported_by_both_views': int(np.sum(best_supported[keep].all(axis=1))),
            'raw_dot_range_semantics': 'zero placeholders when raw_dot_distinct_values is zero; no measured response',
            'target_index_domain': 'bank.xyz exact-coordinate-unique candidate index',
            'no_target_truth': True, 'gaussian_used': False, 'softmax_used': False,
            'no_topk_candidate_truncation': True, 'views': view_diagnostics,
        },
    }
# END independent multiview rank fusion.
