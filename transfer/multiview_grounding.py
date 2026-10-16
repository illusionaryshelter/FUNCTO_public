"""Joint observed-surface inference with explicit missing edges and ambiguity.

The two target images have already been reconstructed into one observed-surface
index domain.  This module consumes that geometry and separate appearance
witnesses; it never reads target task labels, meshes, or evaluator measurements.
Finite proposal budgets limit recall.  Acceptance identifies one task region at
the declared sampling resolution, not a certificate of physical feasibility.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from . import multiview_geometry as geometry
from .core import (TransferConfig, _indices, _mask, _plane,
                   _sampling_parameters, fps_indices)
from .joint_region import JointConfig, graph_scores
from .segmentation import _calibration
from .sim3 import _points, apply_sim3, fit_sim3, rank, rms_size


# BEGIN reconstructed dual-view inference; the single-view joint path is unchanged.
@dataclass(frozen=True)
class MultiviewConfig:
    """Only CPU block sizes are adjustable; search decisions remain frozen.

    The non-init fields are serialized with the result so the search limits,
    absolute scale domain, and source-relative model tolerance are auditable.
    They are engineering assumptions, not parameters estimated from test truth.
    """

    center_chunk: int = 256
    edge_chunk: int = 256
    proposal_trials: int = field(default=256, init=False)
    proposal_fit_budget: int = field(default=256, init=False)
    max_hypotheses: int = field(default=32, init=False)
    pool_retained: int = field(default=10, init=False)
    check_points_per_group: int = field(default=32, init=False)
    seed: int = field(default=0, init=False)
    tolerance_ratio: float = field(default=0.075, init=False)
    minimum_scale: float = field(default=0.2, init=False)
    maximum_scale: float = field(default=5.0, init=False)

    def __post_init__(self):
        for name in ('center_chunk', 'edge_chunk'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')


def _valid_transform(transform, config):
    """Require a proper finite Sim(3) in the declared absolute scale range."""
    if not isinstance(transform, (tuple, list)) or len(transform) != 3:
        return False
    scale, rotation, translation = transform
    rotation, translation = np.asarray(rotation), np.asarray(translation)
    return (np.ndim(scale) == 0 and np.isfinite(scale)
            and config.minimum_scale <= scale <= config.maximum_scale
            and rotation.shape == (3, 3) and translation.shape == (3,)
            and np.isfinite(rotation).all() and np.isfinite(translation).all()
            and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            and np.linalg.det(rotation) > 0)


def _inputs(source, surface, evidence):
    """Keep source indices, query rows, local columns and surface IDs distinct."""
    ps, pt = _points(source['xyz'], 'source.xyz'), _points(surface['xyz'], 'surface.xyz')
    region = _mask(source['region_mask'], len(ps))
    entity = _mask(source['entity_member'], len(ps))
    _, pose = _calibration(source['K'], source['T_world_camera_cv'])
    query_ids = _indices(evidence['query_indices'], len(ps), 'query_indices')
    candidate_ids = _indices(evidence['candidate_indices'], len(pt), 'candidate_indices')
    query_region = _mask(evidence['query_region'], len(query_ids))
    # Demonstration pixels are independent evidence and survive a source SAM
    # miss. Only context queries, which have no demonstration label, require
    # entity membership. The entity alone still fixes the source RMS scale.
    if (not np.array_equal(query_region, region[query_ids])
            or np.any(~query_region & ~entity[query_ids])):
        raise ValueError('Query provenance disagrees with source region/entity membership')
    normals = np.asarray(surface['normals'], dtype=float)
    spacing = np.asarray(surface['spacing'], dtype=float)
    valid = _mask(surface['valid'], len(pt))
    target_entity = _mask(surface['entity_member'], len(pt))
    if (normals.shape != pt.shape or spacing.shape != (len(pt),)
            or not np.isfinite(normals[valid]).all()
            or not np.isfinite(spacing[valid]).all() or np.any(spacing[valid] <= 0)
            or not np.allclose(np.linalg.norm(normals[valid], axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Valid surfaces need unit normals and positive finite spacing')
    if not target_entity[candidate_ids].all():
        raise ValueError('Appearance candidate surfaces must have entity membership')
    view_ids = surface['view_ids']
    if (len(view_ids) != 2 or any(not isinstance(v, str) or not v for v in view_ids)
            or len(set(view_ids)) != 2):
        raise ValueError('Exactly two distinct target view IDs are required')
    shape = (len(query_ids), len(candidate_ids))
    allowed = np.asarray(evidence['allowed'])
    edge_views = np.asarray(evidence['edge_views'])
    available = np.asarray(evidence['appearance_available'])
    appearance = np.asarray(evidence['appearance_cost'], dtype=float)
    if (allowed.dtype != np.bool_ or allowed.shape != shape
            or edge_views.dtype != np.bool_ or edge_views.shape != shape + (2,)
            or available.dtype != np.bool_ or available.shape != shape + (2,)
            or appearance.shape != shape + (2,)
            or not np.array_equal(allowed, edge_views.any(axis=2))
            or np.any(edge_views & ~available)
            or not np.isfinite(appearance[available]).all()
            or np.any((appearance[available] < 0) | (appearance[available] > 1))
            or not np.isnan(appearance[~available]).all()):
        raise ValueError('Invalid per-query allowed edges or per-view appearance availability')
    observed = np.asarray(surface['observed_views'])
    members = np.asarray(surface['member_views'])
    if (observed.dtype != np.bool_ or observed.shape != (len(pt), 2)
            or members.dtype != np.bool_ or members.shape != (len(pt), 2)
            or np.any(available & ~observed[candidate_ids][None, :, :])):
        raise ValueError('Appearance must be supported by the declared target observation')
    return ps, pt, region, entity, pose[:3, 3], query_ids, candidate_ids, query_region


def multiview_proposals(queries, target, allowed, edge_views, *, source_size,
                        initial_transforms=(), config=None, geometry_config=None):
    """Grow A, B and union proposals without collapsing their appearance scores.

    Each pool has its own 85/85/86 growth and fit quotas and ten LAP seats.
    Size-six and size-twelve fits consume separate quota entries.  Empty or
    exhausted pools never donate work to another pool.  Pair compatibility is
    computed on demand, avoiding a square matrix over all candidate edges.
    """
    cfg = MultiviewConfig() if config is None else config
    geo = TransferConfig() if geometry_config is None else geometry_config
    masks = {'A': allowed & edge_views[:, :, 0], 'B': allowed & edge_views[:, :, 1],
             'union': allowed}
    edges = {name: np.nonzero(mask) for name, mask in masks.items()}
    quotas = {'A': 85, 'B': 85, 'union': 86}
    proposed = {name: [] for name in masks}
    native = {0: [], 1: []}
    diagnostic = {'grow_budget_by_pool': quotas.copy(), 'fit_budget_by_pool': quotas.copy(),
                  'grow_attempts': dict.fromkeys(masks, 0), 'fit_attempts': dict.fromkeys(masks, 0),
                  'feasible_proposals': dict.fromkeys(masks, 0), 'retained_by_pool': {},
                  'native_inputs': len(initial_transforms), 'native_retained': 0,
                  'finite_search_recall_guaranteed': False}
    next_id = 0

    def retain(transform, *, origin, pool, view_index=None, rows=(), columns=()):
        nonlocal next_id
        proposal_id, next_id = next_id, next_id + 1
        if not _valid_transform(transform, cfg):
            return
        tau = cfg.tolerance_ratio * source_size * float(transform[0])
        # Preserve pool seats and pool-local geometry ranking together: a
        # proposal generated by A is shortlisted on A's own allowed edges.
        # Every retained proposal subsequently receives the same union LAP.
        quality_rows, quality_cols = edges['union' if pool == 'native' else pool]
        residual = np.linalg.norm(apply_sim3(queries[quality_rows], *transform)-target[quality_cols], axis=1)
        best = np.full(len(queries), np.inf)
        np.minimum.at(best, quality_rows, residual)
        supported = best <= tau
        if supported.sum() < max(6, geo.min_correspondences):
            return
        # This cheap pool-local shortlist counts each source once, but is not
        # a one-to-one matching certificate. The subsequent LAP supplies that.
        quality = (-int(supported.sum()), float(np.mean((best[supported] / tau) ** 2)))
        item = {'transform': (float(transform[0]), np.asarray(transform[1], float),
                              np.asarray(transform[2], float)),
                'origin': origin, 'pool': pool, 'view_index': view_index,
                'proposal_id': proposal_id, 'proposal_support': len(rows) if len(rows) else None,
                'proposal_query_rows': np.asarray(rows, dtype=np.int64),
                'proposal_candidate_columns': np.asarray(columns, dtype=np.int64),
                'quality': quality, 'tolerance_m': tau}
        (native[view_index] if pool == 'native' else proposed[pool]).append(item)

    for item in initial_transforms:
        if (not isinstance(item, dict) or item.get('view_index') not in (0, 1)
                or not isinstance(item.get('origin'), str)):
            raise ValueError('Native proposals require transform, origin and view_index')
        retain(item['transform'], origin=item['origin'], pool='native', view_index=item['view_index'])

    rng = np.random.default_rng(cfg.seed)
    for trial in range(cfg.proposal_trials):
        pool = ('union', 'A', 'B')[trial % 3]
        diagnostic['grow_attempts'][pool] += 1
        rows, columns = edges[pool]
        if len(rows) < 6 or diagnostic['fit_attempts'][pool] >= quotas[pool]:
            continue
        first, second = rng.choice(len(rows), 2, replace=False)
        if rows[first] == rows[second] or columns[first] == columns[second]:
            continue
        source_distance = float(np.linalg.norm(queries[rows[first]]-queries[rows[second]]))
        if source_distance <= np.finfo(float).eps:
            continue
        scale = float(np.linalg.norm(target[columns[first]]-target[columns[second]]) / source_distance)
        if not cfg.minimum_scale <= scale <= cfg.maximum_scale:
            continue
        pair_tolerance = cfg.tolerance_ratio * source_size * scale

        def compatible(index):
            source_dist = np.linalg.norm(queries[rows]-queries[rows[index]], axis=1)
            target_dist = np.linalg.norm(target[columns]-target[columns[index]], axis=1)
            return np.abs(target_dist-scale*source_dist) <= 2*pair_tolerance

        chosen = [int(first), int(second)]
        available = compatible(first) & compatible(second)
        for index in rng.permutation(len(rows)):
            if (not available[index] or rows[index] in rows[chosen]
                    or columns[index] in columns[chosen]):
                continue
            chosen.append(int(index))
            available &= compatible(index)
            if len(chosen) in (6, 12):
                diagnostic['fit_attempts'][pool] += 1
                try:
                    transform = fit_sim3(queries[rows[chosen]], target[columns[chosen]],
                                         geo.rank_relative_tolerance)
                except (ValueError, np.linalg.LinAlgError):
                    transform = None
                if transform is not None:
                    retain(transform, origin=f'multipoint_{pool}_{len(chosen)}', pool=pool,
                           rows=rows[chosen], columns=columns[chosen])
            if len(chosen) == 12 or diagnostic['fit_attempts'][pool] >= quotas[pool]:
                break
    selected = []
    for view_index in (0, 1):
        native[view_index].sort(key=lambda item: (*item['quality'], item['proposal_id']))
        selected.extend(native[view_index][:1])
    diagnostic['native_retained'] = len(selected)
    for pool in ('A', 'B', 'union'):
        proposed[pool].sort(key=lambda item: (*item['quality'], item['proposal_id']))
        selected.extend(proposed[pool][:cfg.pool_retained])
        diagnostic['feasible_proposals'][pool] = len(proposed[pool])
        diagnostic['retained_by_pool'][pool] = min(len(proposed[pool]), cfg.pool_retained)
    diagnostic['total_new_proposal_fits'] = sum(diagnostic['fit_attempts'].values())
    diagnostic['proposals_retained'] = len(selected)
    return selected, diagnostic


def partial_hypothesis(queries, target, allowed, proposal, *, geometry_config=None,
                       config=None):
    """Maximum-cardinality allowed matching, one refit, and final nullification.

    Real costs lie in [0,1], while Q available dummy columns cost Q+1 each.
    Thus one extra real assignment outweighs the full possible residual sum.
    The initial proposal fixes tau: refitting a larger scale cannot enlarge its
    own tolerance.  Every returned nonnegative assignment is final support.
    """
    geo = TransferConfig() if geometry_config is None else geometry_config
    cfg = MultiviewConfig() if config is None else config
    q, k = allowed.shape
    minimum = max(6, geo.min_correspondences)
    if q < minimum or k < minimum:
        return None, 'insufficient_candidate_correspondences'
    tau = proposal['tolerance_m']
    distance = cdist(apply_sim3(queries, *proposal['transform']), target)
    permitted = allowed & (distance <= tau)
    real_cost = np.full((q, k), np.inf)
    real_cost[permitted] = (distance[permitted] / tau) ** 2
    row, column = linear_sum_assignment(np.column_stack((real_cost, np.full((q, q), q+1.0))))
    initial_assignment = np.full(q, -1, dtype=np.int64)
    real = column < k
    initial_assignment[row[real]] = column[real]
    initial_support = np.flatnonzero(initial_assignment >= 0)
    if len(initial_support) < minimum:
        return None, 'insufficient_initial_support'
    p, t = queries[initial_support], target[initial_assignment[initial_support]]
    try:
        transform = fit_sim3(p, t, geo.rank_relative_tolerance)
    except (ValueError, np.linalg.LinAlgError):
        return None, 'degenerate_initial_support'
    if not _valid_transform(transform, cfg):
        return None, 'invalid_refitted_transform'
    residual = np.linalg.norm(apply_sim3(p, *transform)-t, axis=1)
    retained = residual <= tau
    final_support = initial_support[retained]
    if (len(final_support) < minimum
            or min(rank(p[retained], geo.rank_relative_tolerance)[0],
                   rank(t[retained], geo.rank_relative_tolerance)[0]) < 2):
        return None, 'insufficient_final_support'
    assignment = initial_assignment.copy()
    assignment[initial_support[~retained]] = -1
    residuals = np.full(q, np.nan)
    residuals[initial_support] = residual
    return {**proposal, 'initial_transform': proposal['transform'], 'transform': transform,
            'initial_assignment': initial_assignment, 'assignment': assignment,
            'initial_support': initial_support, 'final_support': final_support,
            'residuals': residuals, 'matched_count': len(final_support),
            'inlier_ratio': len(final_support) / len(initial_support),
            'fit_error': float(np.sum((residual[retained] / tau) ** 2) / q)}, None


def _check_set(points, entity, region, query_ids, count):
    """Select fixed source-only region/context checks; fitting queries are excluded."""
    unused = entity.copy()
    unused[query_ids] = False
    groups = []
    for membership in (region, ~region):
        ids = np.flatnonzero(unused & membership)
        groups.append(ids[fps_indices(points[ids], count)] if len(ids) else ids)
    ids = np.concatenate(groups)
    return ids, region[ids]


def _appearance_coordinates(hypotheses, evidence, candidate_ids, observed_views):
    """Freeze one shared query domain per view before any Pareto comparison."""
    q = len(evidence['query_indices'])
    available = np.asarray(evidence['appearance_available'])
    appearance = np.asarray(evidence['appearance_cost'])
    domains, coordinates = [], np.zeros((len(hypotheses), 0))
    columns = []
    for view in range(2):
        common = np.ones(q, dtype=bool)
        for hypothesis in hypotheses:
            assignment = hypothesis['assignment']
            rows = np.flatnonzero(assignment >= 0)
            usable = np.zeros(q, dtype=bool)
            usable[rows] = (available[rows, assignment[rows], view]
                            & observed_views[candidate_ids[assignment[rows]], view])
            common &= usable
        rows = np.flatnonzero(common)
        domains.append(rows)
        if len(rows):
            columns.append([float(np.mean(appearance[rows, item['assignment'][rows], view]))
                            for item in hypotheses])
    if columns:
        coordinates = np.asarray(columns).T
    return domains, coordinates


def _nondominated(values):
    """Compare a single frozen coordinate system; all coordinates are minimized."""
    return np.asarray([i for i, row in enumerate(values)
                       if not np.any(np.all(values <= row, axis=1)
                                     & np.any(values < row, axis=1))], dtype=np.int64)


def _equivalent_regions(modes, *, sampling_spacing_m, normal_min_agreement):
    """All-pairs resolution check, never a transitive nearest-center chain."""
    for i, first in enumerate(modes):
        for second in modes[i+1:]:
            if (np.linalg.norm(first['center']-second['center']) > sampling_spacing_m
                    or first['normal'] @ second['normal'] < normal_min_agreement
                    or not first['mask'][second['seed_index']]
                    or not second['mask'][first['seed_index']]):
                return False
    return True


def infer_multiview_region(source, surface, evidence, *, initial_transforms=(),
                           sampling_radius_m=.03, sampling_spacing_m=.005,
                           config=None, geometry_config=None):
    """Infer one observed task region, or retain competing regions as ambiguous.

    ``assignment`` columns index evidence.candidate_indices; corresponding
    ``assignment_surface_ids``, output anchors, seeds, masks and samples index
    surface.xyz.  Each admitted hypothesis contributes exactly one region, the
    lowest geometric graph score whose connected neighborhood is sampleable.
    Per-view appearance is then compared on fixed domains shared by all those
    hypotheses.  No appearance average or hidden final scalarization is used.
    """
    cfg = MultiviewConfig() if config is None else config
    geo = TransferConfig() if geometry_config is None else geometry_config
    if not isinstance(cfg, MultiviewConfig) or not isinstance(geo, TransferConfig):
        raise ValueError('Expected MultiviewConfig and TransferConfig')
    _sampling_parameters(sampling_radius_m, sampling_spacing_m)
    ps, pt, region, entity, camera, query_ids, candidate_ids, query_region = _inputs(source, surface, evidence)
    diagnostic = {'method': 'multiview_reconstructed_joint', 'configuration': asdict(cfg),
                  'geometry_configuration': asdict(geo), 'quality_warnings': [],
                  'source_region_points': int(region.sum()), 'query_count': len(query_ids),
                  'context_query_count': int((~query_region).sum()),
                  'candidate_count': len(pt), 'appearance_candidate_count': len(candidate_ids),
                  'sampling_radius_m': sampling_radius_m, 'sampling_spacing_m': sampling_spacing_m,
                  'sampling_scale_applied': False, 'view_ids': list(surface['view_ids']),
                  'point_index_domain': 'surface.xyz', 'assignment_index_domain': 'evidence.candidate_indices',
                  'target_truth_used': False, 'hypotheses': [],
                  'output_meaning': 'observed localization at sampling resolution; not physical feasibility'}
    result = {'accepted': False, 'reasons': [], 'modes': [], 'diagnostics': diagnostic,
              'src_anchor_idx': np.empty(0, dtype=np.int64), 'tgt_anchor_idx': np.empty(0, dtype=np.int64),
              'sim3_scale': None, 'sim3_rotation': None, 'sim3_translation': None}

    def reject(reason):
        result['reasons'].append(reason)
        return result

    if region.sum() < geo.min_region_points or len(pt) < geo.min_neighborhood_points:
        return reject('insufficient_observed_task_geometry')
    source_center = ps[region].mean(0)
    try:
        _, _, normal, planarity = _plane(ps[region], geo.rank_relative_tolerance)
    except (ValueError, np.linalg.LinAlgError):
        return reject('source_region_not_approximately_planar')
    if planarity > geo.max_planarity_ratio:
        return reject('source_region_not_approximately_planar')
    view = camera-source_center
    view_length = float(np.linalg.norm(view))
    cosine = float(normal @ view / max(view_length, np.finfo(float).tiny))
    if view_length <= 0 or abs(cosine) < geo.min_view_cosine:
        return reject('source_normal_orientation_unobservable')
    if cosine < 0:
        normal = -normal
    source_size = rms_size(ps[entity])
    if not np.isfinite(source_size) or source_size <= np.finfo(float).tiny:
        return reject('degenerate_source_entity_geometry')
    queries, target = ps[query_ids], pt[candidate_ids]
    proposals, proposal_diagnostic = multiview_proposals(queries, target, evidence['allowed'],
        evidence['edge_views'], source_size=source_size, initial_transforms=initial_transforms,
        config=cfg, geometry_config=geo)
    diagnostic.update(proposal_diagnostic, source_center=source_center.tolist(),
                      source_entity_rms_m=source_size, source_oriented_normal=normal.tolist())
    check_ids, check_region = _check_set(ps, entity, region, query_ids, cfg.check_points_per_group)
    diagnostic.update(check_source_indices=check_ids.tolist(), check_query_region=check_region.tolist(),
                      check_point_count=len(check_ids), check_region_count=int(check_region.sum()),
                      check_context_count=int((~check_region).sum()))
    if not len(check_ids):
        diagnostic['quality_warnings'].append('no_independent_source_check_points')
    graph_config = JointConfig(center_chunk=cfg.center_chunk, edge_chunk=cfg.edge_chunk)
    zeros = np.zeros((len(query_ids), len(candidate_ids)))
    geometric = surface['valid'] & surface['entity_member']
    projection_limit = np.maximum(geo.max_projection_spacing_factor * surface['spacing'],
                                 geo.max_projection_sampling_ratio * sampling_radius_m)
    hypotheses, rejected = [], {}
    checked_regions = 0
    for proposal in proposals:
        hypothesis, reason = partial_hypothesis(queries, target, evidence['allowed'], proposal,
                                               geometry_config=geo, config=cfg)
        if hypothesis is None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        transform = hypothesis['transform']
        mapped_center = apply_sim3(source_center, *transform)
        eligible = geometric & (surface['normals'] @ (transform[1] @ normal) >= geo.normal_min_agreement)
        center_ids = np.flatnonzero(eligible & (np.linalg.norm(pt-mapped_center, axis=1) <= projection_limit))
        if not len(center_ids):
            rejected['no_valid_surface_near_supported_mapping'] = rejected.get(
                'no_valid_surface_near_supported_mapping', 0) + 1
            continue
        assignment = hypothesis['assignment']
        # graph_scores' appearance is exactly zero on real nodes. Subtract
        # its known dummy unary constant, leaving the existing pure triangle
        # geometry with its fixed Q choose 2 denominator and operation order.
        scores = graph_scores(pt[center_ids], source_center, queries, target, assignment,
                              zeros, config=graph_config)
        scores = np.maximum(scores - np.mean(assignment < 0) * graph_config.null_cost, 0)
        mode = None
        # Exactly one region is retained per hypothesis. Equal graph scores
        # use ascending surface ID through this stable order; later ambiguity
        # concerns this finite retained pool, not every physical task region.
        for local in np.argsort(scores, kind='stable'):
            seed = int(center_ids[local])
            candidate = geometry.sample_region(surface, eligible, seed,
                sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m, config=geo)
            checked_regions += 1
            if (candidate['diagnostics']['neighborhood_point_count'] < geo.min_neighborhood_points
                    or candidate['diagnostics']['neighborhood_rank'] < 2):
                continue
            mode = candidate
            mode.update(score=float(scores[local]), graph_score=float(scores[local]),
                        proposal_id=hypothesis['proposal_id'], type='diagnostic')
            break
        if mode is None:
            rejected['little_visible_target_support'] = rejected.get('little_visible_target_support', 0) + 1
            continue
        checks = geometry.check_points(surface, apply_sim3(ps[check_ids], *transform),
                                       tolerance_m=hypothesis['tolerance_m'])
        final = hypothesis['final_support']
        final_columns = assignment[final]
        final_views = evidence['edge_views'][final, final_columns]
        surface_ids = np.full(len(query_ids), -1, dtype=np.int64)
        surface_ids[final] = candidate_ids[final_columns]
        visible = surface['observed_views'][surface_ids[final]]
        members = surface['member_views'][surface_ids[final]]
        hypothesis.update(mode=mode, mapped_center=mapped_center,
            assignment_surface_ids=surface_ids, region_support_count=int(query_region[final].sum()),
            context_support_count=int((~query_region[final]).sum()), graph_score=mode['graph_score'],
            check_support_count=int(np.sum(checks['support'])),
            check_conflict_count=int(np.sum(checks['conflict'])),
            check_unknown_count=int(np.sum(checks['unknown'])), checks=checks,
            check_group_counts={name: {state: int(np.count_nonzero(np.asarray(checks[state])[group]))
                                     for state in ('support', 'conflict', 'unknown')}
                                for name, group in (('region', check_region), ('context', ~check_region))},
            final_edge_sources={'A_only': int(np.count_nonzero(final_views[:, 0] & ~final_views[:, 1])),
                                'B_only': int(np.count_nonzero(final_views[:, 1] & ~final_views[:, 0])),
                                'both': int(np.count_nonzero(final_views.all(axis=1)))},
            final_observed_view_counts={'A_only': int(np.count_nonzero(visible[:, 0] & ~visible[:, 1])),
                                       'B_only': int(np.count_nonzero(visible[:, 1] & ~visible[:, 0])),
                                       'both': int(np.count_nonzero(visible.all(axis=1)))},
            final_member_view_counts={'A_only': int(np.count_nonzero(members[:, 0] & ~members[:, 1])),
                                     'B_only': int(np.count_nonzero(members[:, 1] & ~members[:, 0])),
                                     'both': int(np.count_nonzero(members.all(axis=1)))})
        hypothesis['uses_exclusive_edges_from_both_views'] = bool(
            hypothesis['final_edge_sources']['A_only'] and hypothesis['final_edge_sources']['B_only'])
        hypotheses.append(hypothesis)
    diagnostic.update(hypotheses_admitted=len(hypotheses), hypothesis_rejections=rejected,
                      neighborhoods_checked=checked_regions, assignment_refit_upper_bound=len(proposals),
                      observed_views_meaning='surface support under declared projection rules; not physical certification')
    if not hypotheses:
        return reject('no_supported_task_region')
    domains, appearances = _appearance_coordinates(hypotheses, evidence, candidate_ids, surface['observed_views'])
    score_names = ['negative_region_support', 'negative_context_support', 'fit_error',
                   'graph_error', 'check_conflicts', 'negative_check_support']
    active_views = [view for view, domain in enumerate(domains) if len(domain)]
    score_names.extend(f'appearance_{surface["view_ids"][view]}' for view in active_views)
    values = np.asarray([[-h['region_support_count'], -h['context_support_count'], h['fit_error'],
                          h['graph_score'], h['check_conflict_count'], -h['check_support_count']]
                         for h in hypotheses], dtype=float)
    values = np.column_stack((values, appearances))
    nondominated = _nondominated(values)
    diagnostic.update(score_coordinates=score_names, appearance_common_query_rows=[d.tolist() for d in domains],
                      appearance_common_source_indices=[query_ids[d].tolist() for d in domains],
                      appearance_active_views=[surface['view_ids'][v] for v in active_views],
                      nondominated_hypothesis_indices=nondominated.tolist(),
                      region_equivalence_resolution_m=sampling_spacing_m)
    for view, domain in enumerate(domains):
        if len(domain) == 0:
            diagnostic['quality_warnings'].append(f'empty_common_appearance_domain:{surface["view_ids"][view]}')
        elif len(domain) < max(6, geo.min_correspondences):
            diagnostic['quality_warnings'].append(f'small_common_appearance_domain:{surface["view_ids"][view]}')
    modes = [hypotheses[i]['mode'] for i in nondominated]
    equivalent = _equivalent_regions(modes, sampling_spacing_m=sampling_spacing_m,
                                     normal_min_agreement=geo.normal_min_agreement)
    winner_index = None
    if equivalent:
        centers = np.asarray([mode['center'] for mode in modes])
        medoid = int(np.argmin(np.sum(cdist(centers, centers) ** 2, axis=1)))
        winner_index = int(nondominated[medoid])
    order = list(range(len(hypotheses)))
    if winner_index is not None:
        order.remove(winner_index)
        order.insert(0, winner_index)
    nondominated_set = set(nondominated.tolist())
    for mode_index, h in enumerate(order):
        hypothesis = hypotheses[h]
        mode = hypothesis.pop('mode')
        mode.update(hypothesis_index=h, type=('primary' if h == winner_index else
                    'nondominated_competitor' if h in nondominated_set else 'dominated_diagnostic'))
        hypothesis.update(mode_index=mode_index, score_vector=values[h], nondominated=h in nondominated_set)
        result['modes'].append(mode)
    # Missing residuals are NaN only in numerical working arrays. Frozen JSON
    # uses explicit nulls, because the common writer rejects nonfinite JSON.
    diagnostic['hypotheses'] = [
        {key: ([float(r) if np.isfinite(r) else None for r in value]
               if key == 'residuals' else value)
         for key, value in h.items() if key != 'quality'} for h in hypotheses]
    diagnostic.update(returned_modes=len(result['modes']), primary_hypothesis_index=winner_index,
                      region_ambiguous=not equivalent)
    if winner_index is None:
        return reject('ambiguous_nondominated_regions')
    winner = hypotheses[winner_index]
    final = winner['final_support']
    result.update(accepted=True, src_anchor_idx=query_ids[final],
                  tgt_anchor_idx=winner['assignment_surface_ids'][final],
                  sim3_scale=winner['transform'][0], sim3_rotation=winner['transform'][1],
                  sim3_translation=winner['transform'][2])
    if winner['inlier_ratio'] < geo.min_inlier_ratio:
        diagnostic['quality_warnings'].append('low_correspondence_inlier_ratio')
    if winner['check_conflict_count']:
        diagnostic['quality_warnings'].append('source_check_surface_conflicts')
    if winner['check_unknown_count']:
        diagnostic['quality_warnings'].append('source_check_geometry_unknown')
    if not winner['region_support_count']:
        diagnostic['quality_warnings'].append('no_region_query_correspondence_support')
    diagnostic.update(preferred_center_index=result['modes'][0]['seed_index'],
                      coarse_center=winner['mapped_center'].tolist(),
                      selected_assignment=winner['assignment'].tolist(),
                      selected_assignment_surface_ids=winner['assignment_surface_ids'].tolist(),
                      selected_final_support=final.tolist(), selected_origin=winner['origin'],
                      region_support_count=winner['region_support_count'],
                      context_support_count=winner['context_support_count'],
                      matched_query_count=winner['matched_count'], query_coverage=winner['matched_count']/len(query_ids),
                      inlier_ratio=winner['inlier_ratio'])
    return result
# END reconstructed dual-view inference.
