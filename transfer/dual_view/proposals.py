"""One finite 6/12-point Sim(3) proposal stream for the independent third scheme.

The complete target representative domain is scored before any seed is drawn.
Representatives only propose a transform: neither their multiplicity nor their
indices supply votes in the final continuous-world-query score. The two initial
pairs estimate scale only; a transform is never fitted from fewer than six pairs.
"""

from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from ..sim3 import fit_sim3
from .features import EmpiricalSupport


# BEGIN independent dual-view finite proposal implementation.
# BEGIN align prefix exploration with the two fits available per growth.
# Every growth may fit both six and twelve pairs. Half as many growths as
# fits guarantees the full-domain final prefix is reached before fit exhaustion.
GROWTH_BUDGET = 128
FIT_BUDGET = 256
# END fixed finite coverage budget; no additional fitting budget is introduced.
SEED = 0
MINIMUM_SCALE = .2
MAXIMUM_SCALE = 5.
COMPATIBILITY_RATIO = .075


def valid_transform(transform):
    """Apply the frozen positive-scale/proper-rotation registration contract."""
    if not isinstance(transform, (tuple, list)) or len(transform) != 3:
        return False
    scale, rotation, translation = transform
    rotation, translation = np.asarray(rotation), np.asarray(translation)
    return bool(np.ndim(scale) == 0 and np.isfinite(scale)
                and MINIMUM_SCALE <= scale <= MAXIMUM_SCALE
                and rotation.shape == (3, 3) and translation.shape == (3,)
                and np.isfinite(rotation).all() and np.isfinite(translation).all()
                and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
                and np.linalg.det(rotation) > 0)


def transform_key(transform):
    """Exact numerical identity, including equality of +0 and -0, without clustering."""
    scale, rotation, translation = transform
    return (float(scale), *np.asarray(rotation).ravel().tolist(),
            *np.asarray(translation).ravel().tolist())


def prefix_length(count, attempt):
    """The declared exponential prefix, with an exact full-domain final step."""
    if count == 0:
        return 0
    if attempt == GROWTH_BUDGET - 1:
        return int(count)
    return max(1, int(np.floor(count ** (attempt / (GROWTH_BUDGET - 1)))))


def rank_domain(source_pack, surface, appearance, *, chunk_size=4096):
    """Scan all valid entity representatives, retaining every known raw score.

    A block-sized sample avoids materializing per-view Q-by-N intermediates.
    Scores calibrate one fixed empirical reference distribution per source row.
    Their utility is retained as float32 for bounded-neighborhood refits, so a
    proposal never resamples the complete field. Integer inverse ranks support
    exact prefix membership; stable ties use ascending surface IDs.
    """
    started = perf_counter()
    rows = len(source_pack['query_xyz'])
    ids = np.flatnonzero(surface['valid'] & surface['entity_member'])
    if len(ids) >= np.iinfo(np.int32).max:
        raise MemoryError('target representative domain exceeds int32 indexing')
    scores = np.full((rows, len(ids)), -np.inf, dtype=np.float64)
    known_counts = np.zeros(rows, dtype=np.int64)
    # BEGIN independent review: retain the reasons for proposal-domain loss.
    # Availability belongs to a world representative and camera, whereas the
    # unknown-pair total below counts source-row/representative pairs. Keep
    # these different denominators explicit rather than multiplying view facts.
    view_state_counts = {view: {} for view in surface['view_ids']}
    # END proposal-domain diagnostic initialization.
    blocks = 0
    for begin in range(0, len(ids), chunk_size):
        chosen = ids[begin:begin + chunk_size]
        # BEGIN measured curved representatives retain their owner's error;
        # arbitrary candidate points receive their source error at evaluation.
        if source_pack['region_geometry'] == 'surface':
            sampled = appearance.sample(surface['xyz'][chosen], surface['normals'][chosen],
                query_position_error_m=surface['position_error_m'][chosen])
        else:
            sampled = appearance.sample(surface['xyz'][chosen], surface['normals'][chosen])
        # END the planar call and its numeric arguments are unchanged.
        known = np.asarray(sampled['known'], dtype=bool)
        values = np.asarray(sampled['scores'], dtype=float)
        if known.shape != (rows, len(chosen)) or values.shape != known.shape:
            raise ValueError('appearance full-domain sample must have shape [Q,N]')
        if not np.isfinite(values[known]).all():
            raise ValueError('known appearance scores must be finite')
        scores[:, begin:begin + len(chosen)] = np.where(known, values, -np.inf)
        known_counts += known.sum(axis=1)
        # BEGIN independent review: accumulate every camera's terminal state.
        reason_counts = sampled['reason_counts']
        if set(reason_counts) != set(view_state_counts):
            raise ValueError('appearance reason counts must name both declared views')
        for view, counts in reason_counts.items():
            if sum(counts.values()) != len(chosen):
                raise ValueError('appearance reason counts must cover every sampled representative')
            for reason, number in counts.items():
                view_state_counts[view][reason] = view_state_counts[view].get(reason, 0) + int(number)
        # END proposal-domain reason accumulation; ordering inputs are unchanged.
        blocks += 1
    order = []
    inverse = np.full((rows, len(ids)), -1, dtype=np.int32)
    for row in range(rows):
        ranked = np.argsort(-scores[row], kind='stable')[:known_counts[row]].astype(np.int32)
        order.append(ranked)
        inverse[row, ranked] = np.arange(len(ranked), dtype=np.int32)
    # BEGIN one shared observed reference domain for proposals and inference.
    known = np.isfinite(scores)
    empirical = EmpiricalSupport(scores, known)
    utility = np.zeros(scores.shape, dtype=np.float32)
    for row in range(rows):
        # Preserve unknown raw entries through calibration; zero here stores
        # no positive support edge and does not invent a raw feature response.
        utility[row] = np.where(known[row], empirical.utility(
            scores[row], query_rows=row), 0.)
    target = np.asarray(surface['xyz'][ids], dtype=float)
    domain = {'surface_ids': ids, 'xyz': target, 'order': order,
              'inverse': inverse, 'utility': utility,
              'empirical_support': empirical,
              'max_spacing_m': float(np.max(surface['spacing'][ids], initial=0.)),
              'tree': cKDTree(target) if len(target) else None}
    # END reuse fixed scores without per-transform target resampling.
    diagnostic = {
        'ranking_complete': True, 'ranking_blocks_completed': blocks,
        'ranking_query_count': rows, 'representative_count': len(ids),
        'known_representative_counts': known_counts.tolist(),
        'ranking_domain': 'all_valid_entity_surface_representatives',
        'ranking_unknown_pair_count': int(rows * len(ids) - known_counts.sum()),
        'ranking_view_state_counts': view_state_counts,
        'ranking_view_state_count_unit': 'representative_point_per_view',
        'ranking_unknown_pair_count_unit': 'source_query_representative_pair',
        'ranking_tie_break': 'ascending_surface_id',
        'ranking_seconds': perf_counter() - started,
        'ranking_work_qn': int(rows * len(ids)),
        'retained_domain_array_bytes': int(target.nbytes + inverse.nbytes
            + utility.nbytes + ids.nbytes + sum(row.nbytes for row in order)
            + sum(row.nbytes for row in empirical.references)),
    }
    domain.update(ranking_diagnostics=diagnostic,
                  input_binding=(source_pack, surface, appearance))
    return ids, order, inverse, diagnostic, domain


def build_proposals(source_pack, surface, appearance, *, initial_transforms=(),
                    prepared_domain=None, work_stats=None):
    """Retain native transforms and execute the one declared finite seed plan.

    Every growth attempt and every six/twelve-pair fitting call consumes its
    own budget, including failed attempts. Exact repeated transforms retain all
    provenance but need only one final score. Normal budget exhaustion is a
    completed finite estimator, never a continuous global-optimality claim.
    ``prepared_domain`` may be shared by native backends only while the exact
    prepared source, surface and appearance objects remain unchanged.
    """
    queries = np.asarray(source_pack['query_xyz'], dtype=float)
    region_rows = np.flatnonzero(source_pack['query_region'])
    source_size = float(source_pack['size'])
    if not np.isfinite(source_size) or source_size <= np.finfo(float).tiny:
        raise ValueError('source entity RMS must be positive and finite')
    proposals, seen = [], {}
    diagnostics = {'seed': SEED, 'growth_budget': GROWTH_BUDGET,
                   'proposal_fit_budget': FIT_BUDGET, 'growth_attempts': 0,
                   'proposal_fit_attempts': 0, 'proposal_fit_failures': 0,
                   'native_count': 0, 'native_invalid_count': 0,
                   'duplicate_transform_count': 0, 'native_records': [],
                   'minimum_fit_pairs': 6, 'maximum_fit_pairs': 12,
                   'scale_domain': [MINIMUM_SCALE, MAXIMUM_SCALE],
                   'compatibility_ratio': COMPATIBILITY_RATIO,
                   'full_domain_prefix_reached': False,
                   'global_optimality_certified': False}

    def retain(transform, provenance):
        key = transform_key(transform)
        if key in seen:
            proposals[seen[key]]['origins'].append(provenance)
            diagnostics['duplicate_transform_count'] += 1
            return
        seen[key] = len(proposals)
        proposals.append({'transform': (float(transform[0]),
                          np.asarray(transform[1], dtype=float).copy(),
                          np.asarray(transform[2], dtype=float).copy()),
                          'origins': [provenance], 'proposal_id': len(proposals)})

    for record in initial_transforms:
        diagnostics['native_count'] += 1
        transform = record['transform']
        provenance = {key: value for key, value in record.items() if key != 'transform'}
        valid = valid_transform(transform)
        diagnostics['native_records'].append(dict(provenance, valid_transform=valid))
        if valid:
            retain(transform, provenance)
        else:
            diagnostics['native_invalid_count'] += 1

    # BEGIN reuse only a domain bound to these exact immutable prepared inputs.
    # A task may evaluate two native backends on the same observations. Their
    # reference ranking is identical; a different task's domain is an error.
    if prepared_domain is None:
        ids, order, inverse, ranking, domain = rank_domain(source_pack, surface, appearance)
        ranking = dict(ranking, ranking_reused=False)
    else:
        binding = prepared_domain.get('input_binding', ())
        if len(binding) != 3 or not all(actual is expected for actual, expected in
                zip(binding, (source_pack, surface, appearance))):
            raise ValueError('prepared_domain belongs to different prepared inputs')
        domain = prepared_domain
        ids, order, inverse = domain['surface_ids'], domain['order'], domain['inverse']
        ranking = dict(domain['ranking_diagnostics'], ranking_reused=True,
            ranking_seconds=0.,
            shared_ranking_seconds=domain['ranking_diagnostics']['ranking_seconds'])
    # END task-local reuse; no feature or geometry result is recomputed here.
    diagnostics.update(ranking)
    target, tree = domain['xyz'], domain['tree']
    proposal_started = perf_counter()
    # BEGIN selected exact CUDA growth with original CPU RNG and Sim(3) fits.
    from .cuda_proposals import execute
    execute(queries, region_rows, source_size, ids, order, inverse, target, tree,
            diagnostics, retain, fit_sim3, valid_transform, work_stats)
    # END all fits and retained provenance keep the original attempt order.
    diagnostics.update(proposal_count=len(proposals),
        proposal_seconds=perf_counter() - proposal_started,
        proposal_search_complete=True,
        proposal_stop=('fit_budget_reached' if diagnostics['proposal_fit_attempts'] == FIT_BUDGET
                       else 'growth_budget_reached'))
    return proposals, diagnostics, domain
# END independent dual-view finite proposal implementation.
