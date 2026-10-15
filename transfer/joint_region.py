"""Independent multi-point, partial-assignment region transfer.

Every target entity point is considered against every retained correspondence
hypothesis; unchanged hard gates determine which scores need computation.
The search is exact over that explicitly finite hypothesis set;
proposal generation is bounded and is not a global correspondence certificate.
The interface receives observed points, source task evidence and appearance
costs only. Ground truth, asset names and scene meshes have no input channel.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .core import (TransferConfig, _camera_origin, _mask, _pixels, _plane,
                   _same_seed_plane, _sampling_parameters, _surface_geometry,
                   _sample_neighborhood_validated_coordinates)
from .sim3 import _points, apply_sim3, fit_sim3, rank, rms_size


@dataclass(frozen=True)
class JointConfig:
    """Uniform finite budgets; none are selected from evaluator measurements."""

    proposal_trials: int = 256
    max_hypotheses: int = 32
    null_cost: float = 0.5
    center_chunk: int = 256
    edge_chunk: int = 256
    seed: int = 0

    def __post_init__(self):
        for name in ('proposal_trials', 'max_hypotheses', 'center_chunk', 'edge_chunk', 'seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
                raise ValueError(f'Invalid {name}')
        if not np.isfinite(self.null_cost) or not 0 < self.null_cost <= 1:
            raise ValueError('null_cost must lie in (0,1]')


def graph_scores(centers, source_center, source_queries, target_points, assignment,
                 appearance_cost, *, config=None, center_mask=None):
    """Complete fixed-denominator graph score, evaluated in bounded blocks.

    Each node weighs 1/Q; every unordered source edge weighs 1/C(Q,2).
    An unmatched node pays null_cost once, its incident edges contribute zero.
    A matched degenerate target triangle pays maximal error, not a null reward.
    Triangle error is half the L1 distance between perimeter-normalized ordered
    sides [center-to-i, center-to-k, i-to-k]. It lies in [0,1].

    An optional Boolean center_mask applies an already established hard gate:
    excluded rows return infinity. Original center/edge block boundaries and
    floating-point reduction layout are preserved for every retained row.
    """
    cfg = JointConfig() if config is None else config
    q = len(source_queries)
    if q < 2:
        raise ValueError('The complete graph needs at least two source queries')
    # BEGIN apply exact hard gates without changing the graph's validation.
    if center_mask is not None:
        center_mask = np.asarray(center_mask)
        if center_mask.dtype != np.bool_ or center_mask.shape != (len(centers),):
            raise ValueError('center_mask must be Boolean and center aligned')
    # END masks only select centers; source nodes and edges remain unchanged.
    assigned = np.asarray(assignment)
    real = assigned >= 0
    unary = np.full(q, cfg.null_cost)
    unary[real] = appearance_cost[np.flatnonzero(real), assigned[real]]
    scores = np.full(len(centers), float(unary.mean()))
    if center_mask is not None:
        scores[~center_mask] = np.inf
    i, k = np.triu_indices(q, 1)
    total_edges = len(i)
    active = real[i] & real[k]
    i, k = i[active], k[active]
    if not len(i):
        return scores
    radii = np.linalg.norm(source_queries-source_center, axis=1)
    source_sides = np.stack((radii[i], radii[k],
                            np.linalg.norm(source_queries[i]-source_queries[k], axis=1)), axis=1)
    source_perimeter = source_sides.sum(1)
    if np.any(source_perimeter <= np.finfo(float).tiny):
        raise ValueError('Source queries contain a degenerate center triangle')
    source_shape = source_sides/source_perimeter[:, None]
    matched = target_points[np.maximum(assigned, 0)]
    bases = np.linalg.norm(matched[i]-matched[k], axis=1)
    # BEGIN preserve SciPy input validation even when every center is excluded.
    if center_mask is not None and len(centers):
        centers = np.asarray(centers)
        cdist(centers[:0], matched)
    # END zero rows validate shape/type without computing a distance matrix.
    for begin in range(0, len(centers), cfg.center_chunk):
        end = min(begin+cfg.center_chunk, len(centers))
        # BEGIN skip excluded rows while retaining the original reduction path.
        block = centers[begin:end]
        selected = None if center_mask is None else np.flatnonzero(center_mask[begin:end])
        if selected is not None:
            if not len(selected):
                continue
            block = block[selected]
            if len(selected) == 1 and end-begin > 1:
                # Advanced column indexing below normally makes a Fortran-
                # contiguous error matrix. With one row it becomes contiguous
                # on the reduction axis and NumPy may use pairwise summation.
                # A duplicate row preserves the old strided-axis arithmetic;
                # its result is discarded. A genuinely one-row original tail
                # keeps its original one-row reduction instead.
                block = np.repeat(block, 2, axis=0)
        distances = cdist(block, matched)
        error_sum = np.zeros(len(block))
        # END only retained rows enter the unchanged edge arithmetic below.
        for edge in range(0, len(i), cfg.edge_chunk):
            stop = min(edge+cfg.edge_chunk, len(i))
            a, b, base = distances[:, i[edge:stop]], distances[:, k[edge:stop]], bases[None, edge:stop]
            perimeter = a+b+base
            denominator = np.maximum(perimeter, np.finfo(float).tiny)
            error = (np.abs(a/denominator-source_shape[None, edge:stop, 0])
                     + np.abs(b/denominator-source_shape[None, edge:stop, 1])
                     + np.abs(base/denominator-source_shape[None, edge:stop, 2]))/2
            error[perimeter <= np.finfo(float).tiny] = 1
            error_sum += error.sum(1)
        # BEGIN write scores back at original indices for stable global ties.
        if selected is None:
            scores[begin:end] += error_sum/total_edges
        else:
            scores[begin+selected] += error_sum[:len(selected)]/total_edges
        # END original hypothesis-by-center ordering remains intact.
    return scores


def _valid_transform(transform, size_ratio):
    scale, rotation, translation = transform
    return (np.isfinite(scale) and scale > 0 and 0.2 <= scale/size_ratio <= 5
            and np.isfinite(rotation).all() and np.isfinite(translation).all()
            and np.allclose(rotation.T@rotation, np.eye(3), rtol=0, atol=1e-6)
            and np.linalg.det(rotation) > 0)


def multipoint_proposals(queries, target, peaks, appearance, *, tolerance, size_ratio,
                        initial_transforms=(), config=None, geometry_config=None):
    """Grow distance-consistent 6/12-point sets, then fit every point in a set.

    Two proposed pairs only establish a candidate scale for distance checks;
    they never define an output transform. Each extension must agree with ALL
    already selected pair distances within 2*tolerance, the triangle-inequality
    bound for two residuals no larger than tolerance. Each generated transform
    requires at least six distinct source AND target points and centered rank
    at least two. The size-12 fit is an additional proposal, not a replacement.
    Existing native-region robust estimates may enter the same proposal pool.
    """
    cfg = JointConfig() if config is None else config
    geo = TransferConfig() if geometry_config is None else geometry_config
    rows, columns = np.nonzero(peaks >= 0)
    targets = peaks[rows, columns]
    # A native decode may duplicate an appearance peak. Unique pairs prevent
    # duplicated observations from changing sampling probability or support.
    pairs = np.unique(np.column_stack((rows, targets)), axis=0)
    if not len(pairs):
        return []
    rows, targets = pairs.T
    priorities = appearance[rows, targets]
    source_dist = cdist(queries[rows], queries[rows])
    target_dist = cdist(target[targets], target[targets])
    rng = np.random.default_rng(cfg.seed)
    proposals = []

    def retain(transform, origin, support_size):
        if not _valid_transform(transform, size_ratio):
            return
        residual = np.linalg.norm(apply_sim3(queries[rows], *transform)-target[targets], axis=1)
        # Count source queries once even when several appearance peaks agree.
        best = np.full(len(queries), np.inf)
        np.minimum.at(best, rows, residual)
        inliers = best <= tolerance
        if inliers.sum() >= geo.min_correspondences:
            proposals.append({'transform': transform, 'origin': origin,
                              'proposal_support': None if support_size is None else int(support_size),
                              'quality': (-int(inliers.sum()), float(np.median(best[inliers]))),
                              'proposal_id': len(proposals)})

    for index, transform in enumerate(initial_transforms):
        retain(transform, f'native_consensus_{index}', None)
    order = np.argsort(priorities, kind='stable')
    for trial in range(cfg.proposal_trials):
        # The deterministic prefix grows to the whole peak population. Random
        # tie perturbations change the extension order, never the hard checks.
        prefix = min(len(order), max(12, 12+(len(order)-12)*(trial+1)//cfg.proposal_trials))
        if prefix < 2:
            break
        first, second = rng.choice(order[:prefix], 2, replace=False)
        distance = source_dist[first, second]
        if distance <= np.finfo(float).eps or targets[first] == targets[second]:
            continue
        scale = target_dist[first, second]/distance
        if not 0.2 <= scale/size_ratio <= 5:
            continue
        # BEGIN evaluate only compatibility rows actually used by this trial.
        # Every element uses the same dtype and arithmetic as the former full
        # matrix. No random draw, extension order, fit or admission changes.
        # Do not cap row accesses: the existing failed-fit `continue` may let
        # a degenerate size-12 set grow further, and that behavior is retained.
        def compatible_row(index):
            return np.abs(target_dist[index]-scale*source_dist[index]) <= 2*tolerance
        chosen = [int(first), int(second)]
        available = compatible_row(first) & compatible_row(second)
        # END on-demand rows replace only unused pairwise arithmetic.
        extension_order = np.argsort(priorities+0.05*rng.random(len(priorities)), kind='stable')
        for index in extension_order:
            if not available[index] or rows[index] in rows[chosen] or targets[index] in targets[chosen]:
                continue
            chosen.append(int(index))
            available &= compatible_row(index)
            if len(chosen) in (6, 12):
                try:
                    transform = fit_sim3(queries[rows[chosen]], target[targets[chosen]], geo.rank_relative_tolerance)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                retain(transform, f'multipoint_{len(chosen)}', len(chosen))
            if len(chosen) == 12:
                break
    # Equal transforms with different proposal identities are retained. No
    # correspondence-only deduplication discards a distinct effective domain.
    proposals.sort(key=lambda item: (*item['quality'], item['proposal_id']))
    # Native consensus has an explicit pool slot, so approximate peak support
    # ranking cannot silently remove the old evidence before the joint score.
    native = [item for item in proposals if item['origin'].startswith('native_')]
    other = [item for item in proposals if not item['origin'].startswith('native_')]
    return (native+other)[:cfg.max_hypotheses]


def partial_hypothesis(queries, target, candidate_ids, appearance, proposal, *,
                       tolerance, size_ratio, config, geometry_config):
    """One rectangular LAP, one fixed-support refit, then admission checks.

    Q-m_min dummy columns allow at least m_min real, one-to-one assignments.
    These forced real assignments still have to pass the residual/rank checks;
    dummy capacity does not manufacture six geometrically consistent pairs.
    Assignment minimizes a proposal surrogate, not the quadratic graph score.
    """
    q, minimum = len(queries), geometry_config.min_correspondences
    if q < minimum or len(candidate_ids) < minimum:
        return None
    distance = cdist(apply_sim3(queries, *proposal['transform']), target[candidate_ids])
    cost = appearance[:, candidate_ids]+np.minimum(distance/tolerance, 1)
    extended = np.column_stack((cost, np.full((q, q-minimum), config.null_cost)))
    row, col = linear_sum_assignment(extended)
    assignment = np.full(q, -1, dtype=np.int64)
    real = col < len(candidate_ids)
    assignment[row[real]] = candidate_ids[col[real]]
    real_ids = np.flatnonzero(assignment >= 0)
    p, t = queries[real_ids], target[assignment[real_ids]]
    initial = np.linalg.norm(apply_sim3(p, *proposal['transform'])-t, axis=1) <= tolerance
    if initial.sum() < minimum:
        return None
    try:
        transform = fit_sim3(p[initial], t[initial], geometry_config.rank_relative_tolerance)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not _valid_transform(transform, size_ratio):
        return None
    residual = np.linalg.norm(apply_sim3(p, *transform)-t, axis=1)
    final = residual <= tolerance
    if final.sum() < minimum or min(rank(p[final], geometry_config.rank_relative_tolerance)[0],
                                   rank(t[final], geometry_config.rank_relative_tolerance)[0]) < 2:
        return None
    # BEGIN preserve the proposal used to define I0 alongside the final fit.
    # The two transforms serve different audit roles; overwriting the first
    # would prevent reconstruction of the fixed initial-support decision.
    return {**proposal, 'initial_transform': proposal['transform'],
            'transform': transform, 'assignment': assignment,
            'initial_support': real_ids[initial], 'final_support': real_ids[final],
            'matched_count': len(real_ids), 'residuals': residual,
            'inlier_ratio': float(final.mean())}
    # END retain both mappings without changing fitting or selection arithmetic.


def infer_joint_region(source, target, evidence, *, query_region,
                       initial_transforms=(), sampling_radius_m=.03,
                       sampling_spacing_m=.005, config=None, geometry_config=None):
    """Return a primary region and full-domain surface/location witnesses.

    Target geometry is estimated once. Only normal agreement depends on each
    hypothesis rotation. Sampling checks are delayed until candidates are read
    in global score order, exactly preserving argmin over all valid candidates.
    The old center-projection formula bounds extra displacement relative to the
    estimated mapping; it provides no absolute bound against unknown truth.
    """
    cfg = JointConfig() if config is None else config
    geo = TransferConfig() if geometry_config is None else geometry_config
    ps, pt = _points(source['xyz'], 'source'), _points(target['xyz'], 'target')
    source_mask = _mask(source['region_mask'], len(ps))
    pixels = _pixels(target['pixels'], len(pt))
    _sampling_parameters(sampling_radius_m, sampling_spacing_m)
    source_camera = _camera_origin(source['T_world_camera_cv'][:3, 3], 'source_camera')
    target_camera = _camera_origin(target['T_world_camera_cv'][:3, 3], 'target_camera')
    query_ids = np.asarray(evidence['query_indices'])
    appearance, peaks = np.asarray(evidence['appearance_cost']), np.asarray(evidence['peak_indices'])
    query_region = np.asarray(query_region)
    if (query_ids.ndim != 1 or query_ids.dtype.kind not in 'iu' or len(np.unique(query_ids)) != len(query_ids)
            or np.any(query_ids < 0) or np.any(query_ids >= len(ps))
            or appearance.shape != (len(query_ids), len(pt)) or not np.isfinite(appearance).all()
            or np.any((appearance < 0) | (appearance > 1))
            or peaks.ndim != 2 or len(peaks) != len(query_ids) or peaks.dtype.kind not in 'iu'
            or np.any(peaks < -1) or np.any(peaks >= len(pt))
            or query_region.dtype != np.bool_ or query_region.shape != query_ids.shape):
        raise ValueError('Invalid query, appearance, or peak contract')
    if not np.array_equal(query_region, source_mask[query_ids]):
        raise ValueError('Query provenance disagrees with source task evidence')
    if not np.all(source['entity_member'][query_ids[~query_region]]) or not np.all(target['entity_member']):
        raise ValueError('Context and every target point must remain inside entity masks')
    diagnostic = {'method': 'joint_region', 'quality_warnings': [], 'source_region_points': int(source_mask.sum()),
                  'candidate_count': len(pt), 'query_count': len(query_ids), 'context_query_count': int((~query_region).sum()),
                  'target_truth_used': False, 'sampling_radius_m': sampling_radius_m,
                  'sampling_spacing_m': sampling_spacing_m, 'sampling_scale_applied': False}
    result = {'accepted': False, 'reasons': [], 'modes': [], 'diagnostics': diagnostic,
              'src_anchor_idx': np.empty(0, dtype=np.int64), 'tgt_anchor_idx': np.empty(0, dtype=np.int64),
              'sim3_scale': None, 'sim3_rotation': None, 'sim3_translation': None}

    def reject(reason):
        result['reasons'].append(reason)
        return result

    if source_mask.sum() < geo.min_region_points or len(pt) < geo.min_neighborhood_points:
        return reject('insufficient_observed_task_geometry')
    region = ps[source_mask]
    source_center = region.mean(0)
    try:
        _, _, normal, planarity = _plane(region, geo.rank_relative_tolerance)
    except (ValueError, np.linalg.LinAlgError):
        return reject('source_region_not_approximately_planar')
    if planarity > geo.max_planarity_ratio:
        return reject('source_region_not_approximately_planar')
    view = source_camera-source_center
    # BEGIN retain the original undefined-view rejection even at zero cutoff.
    view_length = float(np.linalg.norm(view))
    cosine = float(normal@view/max(view_length, np.finfo(float).tiny))
    if view_length <= 0 or abs(cosine) < geo.min_view_cosine:
        return reject('source_normal_orientation_unobservable')
    # END a zero camera-to-point vector cannot supply an oriented normal.
    if cosine < 0:
        normal = -normal
    queries = ps[query_ids]
    target_size, source_size = rms_size(pt), rms_size(ps[np.asarray(source['entity_member'], bool)])
    if min(target_size, source_size) <= np.finfo(float).tiny:
        return reject('degenerate_entity_geometry')
    tolerance, size_ratio = geo.ransac_tolerance_ratio*target_size, target_size/source_size
    proposals = multipoint_proposals(queries, pt, peaks, appearance, tolerance=tolerance,
        size_ratio=size_ratio, initial_transforms=initial_transforms, config=cfg, geometry_config=geo)
    candidate_ids = np.unique(peaks[peaks >= 0])
    hypotheses = []
    for proposal in proposals:
        hypothesis = partial_hypothesis(queries, pt, candidate_ids, appearance, proposal,
            tolerance=tolerance, size_ratio=size_ratio, config=cfg, geometry_config=geo)
        if hypothesis is not None:
            hypotheses.append(hypothesis)
    diagnostic.update(proposals_retained=len(proposals), hypotheses_admitted=len(hypotheses),
                      ransac_tolerance_m=tolerance, source_center=source_center.tolist())
    if not hypotheses:
        return reject('no_supported_partial_correspondence')
    # _surface_geometry's planes, flatness, view cosine and spacing do not
    # depend on wanted_normal. Reconstruct geometric validity independently,
    # then apply each hypothesis's signed normal agreement below.
    normals, flatness, _, view_cosines, spacings, _ = _surface_geometry(
        pt, pixels, normal, target_camera, sampling_radius_m, geo)
    # BEGIN match every view-validity condition from the shared surface helper.
    geometric = ((flatness <= geo.max_planarity_ratio) & (spacings > 0)
                 & (view_cosines >= geo.min_view_cosine)
                 & (np.linalg.norm(target_camera-pt, axis=1) > 0))
    # END only signed agreement remains hypothesis dependent.
    all_scores, eligible_masks, locations = [], [], []
    score_rows_evaluated = 0
    projection_limit = np.maximum(geo.max_projection_spacing_factor*spacings,
                                 geo.max_projection_sampling_ratio*sampling_radius_m)
    for hypothesis in hypotheses:
        transform = hypothesis['transform']
        center = apply_sim3(source_center, *transform)
        eligible = geometric & (normals@(transform[1]@normal) >= geo.normal_min_agreement)
        location = np.linalg.norm(pt-center, axis=1) <= projection_limit
        # BEGIN apply the unchanged hard gates before expensive graph scoring.
        center_mask = eligible & location
        scores = graph_scores(pt, source_center, queries, pt, hypothesis['assignment'], appearance,
                              config=cfg, center_mask=center_mask)
        all_scores.append(scores)
        # Count actual center rows, including the duplicate used solely to
        # preserve singleton floating-point reduction layout.
        block_starts = np.arange(0, len(pt), cfg.center_chunk)
        block_counts = np.add.reduceat(center_mask.astype(np.int64), block_starts)
        block_sizes = np.minimum(cfg.center_chunk, len(pt)-block_starts)
        score_rows_evaluated += int(center_mask.sum()+np.count_nonzero((block_counts == 1) & (block_sizes > 1)))
        # END location gates centers only, not the original sampling mask.
        eligible_masks.append(eligible)
        locations.append(center)
    values = np.asarray(all_scores)
    valid_flat = np.flatnonzero(np.isfinite(values.ravel()))
    ordered = valid_flat[np.argsort(values.ravel()[valid_flat], kind='stable')]
    # BEGIN distinguish the unchanged full search domain from work performed.
    diagnostic.update(full_domain_candidate_pairs=int(values.size),
                      graph_score_rows_evaluated=score_rows_evaluated,
                      surface_and_location_candidates=int(len(ordered)))
    # END the retired computed-count name no longer misstates execution work.
    if not len(ordered):
        return reject('no_valid_surface_near_supported_mapping')
    primary_plane, primary_tangent = None, None
    selected, competing = [], {}
    checked = 0
    for flat in ordered:
        h, seed = divmod(int(flat), len(pt))
        # BEGIN retain both original ambiguity meanings across the full domain.
        # Different surfaces and distant positions on the primary plane are
        # distinct warnings. Nearby windows cannot consume either witness slot.
        kind = 'primary'
        if primary_plane is not None:
            kind = ('different_surface' if not primary_plane[seed]
                    else 'same_plane_localization' if primary_tangent[seed] > 2*sampling_radius_m else None)
            if kind is None or kind in competing:
                continue
        # END hypothesis-independent comparison against the frozen primary plane.
        mode = _sample_neighborhood_validated_coordinates(pt, eligible_masks[h], normals, spacings, seed,
            pixels=pixels, sampling_radius_m=sampling_radius_m,
            sampling_spacing_m=sampling_spacing_m, config=geo)
        checked += 1
        if (mode['diagnostics']['neighborhood_point_count'] < geo.min_neighborhood_points
                or mode['diagnostics']['neighborhood_rank'] < 2):
            continue
        mode.update(score=float(values[h, seed]), type=kind,
                    hypothesis_index=h, proposal_id=hypotheses[h]['proposal_id'])
        selected.append((h, mode))
        if primary_plane is None:
            primary_plane, primary_tangent = _same_seed_plane(pt, normals, seed,
                geo.surface_height_spacing_factor*spacings[seed], geo)
        else:
            competing[kind] = mode
            if len(competing) == 2:
                break
    diagnostic['neighborhoods_checked'] = checked
    if not selected:
        return reject('little_visible_target_support')
    best_h, best_mode = selected[0]
    winner = hypotheses[best_h]
    final = winner['final_support']
    result.update(accepted=True, modes=[mode for _, mode in selected],
                  src_anchor_idx=query_ids[final], tgt_anchor_idx=winner['assignment'][final],
                  sim3_scale=winner['transform'][0], sim3_rotation=winner['transform'][1],
                  sim3_translation=winner['transform'][2])
    seed = best_mode['seed_index']
    alternate = competing.get('different_surface')
    position = competing.get('same_plane_localization')
    gap = alternate['score']-best_mode['score'] if alternate is not None else None
    position_gap = position['score']-best_mode['score'] if position is not None else None
    if winner['inlier_ratio'] < geo.min_inlier_ratio:
        diagnostic['quality_warnings'].append('low_correspondence_inlier_ratio')
    if gap is not None and gap < geo.min_mode_gap:
        diagnostic['quality_warnings'].append('surface_ambiguity')
    if position_gap is not None and position_gap < geo.min_mode_gap:
        diagnostic['quality_warnings'].append('position_preference_ambiguity')
    diagnostic.update(selected_score=best_mode['score'], preferred_center_index=seed,
        selected_different_surface_gap=gap, surface_ambiguity=gap is not None and gap < geo.min_mode_gap,
        same_plane_position_gap=position_gap,
        position_preference_ambiguous=position_gap is not None and position_gap < geo.min_mode_gap,
        center_projection_distance_m=float(np.linalg.norm(pt[seed]-locations[best_h])),
        center_projection_limit_m=float(projection_limit[seed]), coarse_center=locations[best_h].tolist(),
        inlier_ratio=winner['inlier_ratio'], matched_query_count=winner['matched_count'],
        query_coverage=winner['matched_count']/len(queries),
        region_support_count=int(query_region[final].sum()), context_support_count=int((~query_region[final]).sum()),
        support_center_distance_m=float(np.linalg.norm(queries[final].mean(0)-source_center)),
        selected_origin=winner['origin'], selected_proposal_support=winner['proposal_support'],
        selected_assignment=winner['assignment'].tolist(),
        selected_initial_support=winner['initial_support'].tolist(), selected_final_support=final.tolist(),
        returned_modes=len(selected))
    # All hypothesis metadata are frozen before any evaluator can open truth.
    diagnostic['hypotheses'] = [{key: value for key, value in item.items() if key != 'quality'}
                              for item in hypotheses]
    return result


def select_observation(results):
    """Select the lowest common score from an ordered collection of results.

    IDs are provenance only. Exact ties retain caller order; names such as A/B
    cannot affect the numeric score. A rejected observation cannot win. Callers
    must use the same source queries/configuration for every compared result.
    """
    candidates = [(float(item['diagnostics']['selected_score']), index)
                  for index, item in enumerate(results) if item['accepted']]
    return min(candidates)[1] if candidates else None
