"""Continuous-world-query inference for the independent third transfer scheme.

A single proper Sim(3) maps every frozen source row to one world position.
The two response fields are queried at that same position; every source row
remains in the score denominator. Geometry is a feasibility contract, not an
extra ranking coordinate. A returned region never silently identifies an
otherwise ambiguous interaction transform.
"""

import numpy as np

from ..core import _sampling_parameters
from ..sim3 import apply_sim3, fit_sim3, rank
from . import geometry
from .features import interval_score
from .proposals import build_proposals, transform_key, valid_transform


# BEGIN independent dual-view full-query scoring and region decision.
SCHEME = 'dual_view_continuous'
DISPLAY_DIAGNOSTIC_LIMIT = 6


def _finite_list(value):
    """Serialize missing scores as null, keeping the associated validity mask."""
    value = np.asarray(value)
    if value.ndim == 0:
        return float(value) if np.isfinite(value) else None
    return [_finite_list(row) for row in value]


def _measurement_pairs(surface, mapped, support):
    """Choose a supported actual depth measurement by residual, then camera ID.

    These pairs are used only for one deterministic geometry refit and anchor
    diagnostics. No feature maximization, nearest-neighbor search, or distinct
    target-capacity constraint changes the continuous final correspondence.
    """
    ids = np.column_stack([view['observation_indices'] for view in support['views']])
    distances = np.full(ids.shape, np.inf)
    for view in range(2):
        valid = support['view_support'][:, view] & (ids[:, view] >= 0)
        distances[valid, view] = np.linalg.norm(
            surface['observation_xyz'][ids[valid, view]] - mapped[valid], axis=1)
    chosen_view = np.argmin(distances, axis=1)
    rows = np.flatnonzero(support['support'])
    observation_ids = ids[rows, chosen_view[rows]]
    return rows, observation_ids, distances[rows, chosen_view[rows]]


def _support_summary(support):
    """Retain per-view depth states as well as combined support/conflict counts."""
    return {'support_count': int(support['support'].sum()),
            'conflict_count': int(support['conflict'].sum()),
            'unknown_count': int(support['unknown'].sum()),
            'support': support['support'].tolist(),
            'conflict': support['conflict'].tolist(),
            'view_support': support['view_support'].tolist(),
            'view_conflict': support['view_conflict'].tolist(),
            'view_states': support['view_states'].tolist(),
            'state_names': support['state_names']}


def _region_summary(mode):
    """Keep only small reconstructible region metadata, never a per-T N-mask."""
    return {'seed_index': mode['seed_index'], 'center': mode['center'].tolist(),
            'normal': mode['normal'].tolist(),
            'sample_indices': mode['sample_indices'].tolist(),
            'diagnostics': mode['diagnostics']}


def infer(source_pack, surface, appearance, *, initial_transforms=(),
          sampling_radius_m=.03, sampling_spacing_m=.005):
    """Return a finite-domain region decision without altering either old solver.

    All legal native, six-pair and twelve-pair transforms, and one supported
    refit per original transform, receive the same full-query scoring. An
    output-infeasible transform is excluded from H, except unresolved center
    projection branches, which must remain interval competitors. Potential
    winners W satisfy U >= max(L); no score margin or camera quota exists.
    """
    _sampling_parameters(sampling_radius_m, sampling_spacing_m)
    queries = np.asarray(source_pack['query_xyz'], dtype=float)
    query_region = np.asarray(source_pack['query_region'], dtype=bool)
    query_indices = np.asarray(source_pack['query_indices'], dtype=np.int64)
    source_normals = np.asarray(source_pack['query_normals'], dtype=float)
    count = len(queries)
    if queries.shape != (count, 3) or query_region.shape != (count,):
        raise ValueError('source queries and region membership must be aligned')
    if query_indices.shape != (count,) or source_normals.shape != queries.shape:
        raise ValueError('source query identities and normals must be aligned')
    if np.asarray(appearance.lower).shape != (count,) or np.asarray(appearance.upper).shape != (count,):
        raise ValueError('appearance bounds must cover every fixed source query')
    sampling = dict(sampling_radius_m=sampling_radius_m,
                    sampling_spacing_m=sampling_spacing_m)
    diagnostics = {'method': SCHEME, 'scheme': SCHEME, 'target_truth_used': False,
        'quality_warnings': ['finite_search_not_global'],
        'fixed_query_denominator': count, 'query_indices': query_indices.tolist(),
        'query_region': query_region.tolist(),
        'source': source_pack.get('diagnostics', {}),
        'query_count': count, 'region_query_count': int(query_region.sum()),
        'context_query_count': int((~query_region).sum()),
        'view_ids': list(surface['view_ids']), 'point_index_domain': 'surface.xyz',
        'score_definition': 'fixed_Q_mean_of_same_world_position_valid_view_max_raw_dot',
        'unknown_lower': np.asarray(appearance.lower).tolist(),
        'unknown_upper': np.asarray(appearance.upper).tolist(),
        'search_status': 'not_started', 'search_stage': 'input',
        'search_scope': 'native_teaser_ransac_plus_declared_6_12_point_sim3_estimator',
        'global_optimality_certified': False, 'region_identified': False,
        'transform_identified': False, 'sampling_radius_m': sampling_radius_m,
        'sampling_spacing_m': sampling_spacing_m, 'sampling_scale_applied': False,
        'output_meaning': 'finite-model observed localization, not semantic or action safety certification',
        'hypotheses': [], 'refit_attempts': 0, 'refit_invalid_count': 0,
        'refit_unavailable_count': 0,
        'display_diagnostic_limit': DISPLAY_DIAGNOSTIC_LIMIT,
        'display_limit_does_not_restrict_inference': True}
    result = {'scheme': SCHEME, 'status': 'not_started', 'accepted': False,
        'reasons': [], 'modes': [], 'diagnostics': diagnostics,
        'src_anchor_idx': np.empty(0, dtype=np.int64),
        'tgt_anchor_idx': np.empty(0, dtype=np.int64),
        'sim3_scale': None, 'sim3_rotation': None, 'sim3_translation': None,
        'region_identified': False, 'transform_identified': False,
        'diagnostic_hypotheses': diagnostics['hypotheses']}
    if not len(source_pack['check_xyz']):
        diagnostics['quality_warnings'].append('no_independent_source_check_points')
    if np.any(~np.asarray(source_pack['query_normal_valid'], dtype=bool)):
        diagnostics['quality_warnings'].append('some_fixed_source_query_normals_unavailable')
    if count == 0 or not query_region.any():
        result.update(status='missing_input', reasons=['no_fixed_source_region_queries'])
        diagnostics.update(decision_status='missing_input', source_model_applicable=False)
        return result
    if source_pack['reasons']:
        result.update(status='missing_input', reasons=list(source_pack['reasons']))
        diagnostics.update(decision_status='missing_input', source_model_applicable=False)
        return result
    diagnostics['source_model_applicable'] = True

    hypotheses, seen = [], {}

    def evaluate(transform, origins):
        """Score all Q, check independent geometry, and test one output region."""
        key = transform_key(transform)
        if key in seen:
            existing = hypotheses[seen[key]]
            existing['summary']['origins'].extend(origins)
            return existing
        mapped = apply_sim3(queries, *transform)
        normals = source_normals @ transform[1].T
        support = geometry.geometry_support(surface, mapped, normals)
        checks = geometry.geometry_support(surface,
            apply_sim3(source_pack['check_xyz'], *transform),
            np.asarray(source_pack['check_normals']) @ transform[1].T)
        center = apply_sim3(source_pack['center'], *transform)
        normal = transform[1] @ source_pack['normal']
        center_support = geometry.geometry_support(surface, center[None], normal[None])
        sampled = appearance.sample(mapped, normals, query_rows=np.arange(count))
        scores, known = np.asarray(sampled['scores']), np.asarray(sampled['known'], dtype=bool)
        if scores.shape != (count,) or known.shape != (count,):
            raise ValueError('paired appearance sample must preserve the fixed Q rows')
        interval = interval_score(scores, known, appearance.lower, appearance.upper)
        rows, observation_ids, residuals = _measurement_pairs(surface, mapped, support)
        source_rank = rank(queries[rows])[0]
        rejections = []
        if support['conflict'].any():
            rejections.append('query_geometry_model_conflict')
        if checks['conflict'].any():
            rejections.append('independent_check_geometry_model_conflict')
        if center_support['conflict'].any():
            rejections.append('center_geometry_model_conflict')
        if len(rows) < 6:
            rejections.append('fewer_than_six_oriented_query_supports')
        if source_rank < 2:
            rejections.append('supported_source_rank_below_two')
        if not query_region[rows].any():
            rejections.append('no_source_region_geometry_support')
        mode, reason = geometry.region_for_transform(surface, source_pack, transform, **sampling)
        ambiguous = reason == 'center_projection_ambiguous'
        if reason is not None and not ambiguous and reason not in rejections:
            rejections.append(reason)
        admitted = not rejections
        summary = {'hypothesis_id': len(hypotheses), 'origins': list(origins),
            'transform': [float(transform[0]), transform[1].tolist(), transform[2].tolist()],
            'lower': float(interval['lower']), 'upper': float(interval['upper']),
            'known_count': int(known.sum()), 'unknown_count': int((~known).sum()),
            'known_region_count': int(np.count_nonzero(known & query_region)),
            'known_context_count': int(np.count_nonzero(known & ~query_region)),
            'scores': _finite_list(scores), 'known': known.tolist(),
            'view_scores': _finite_list(sampled['view_scores']),
            'view_valid': np.asarray(sampled['view_valid'], dtype=bool).tolist(),
            'appearance_view_states': np.asarray(sampled['view_state']).tolist(),
            'appearance_reason_counts': sampled['reason_counts'],
            'geometry': _support_summary(support), 'checks': _support_summary(checks),
            'center_geometry': _support_summary(center_support),
            'supported_source_rank': source_rank,
            'region_geometry_count': int(np.count_nonzero(query_region[rows])),
            'paired_query_rows': rows.tolist(),
            'paired_observation_indices': observation_ids.tolist(),
            'paired_surface_indices': surface['observation_to_element'][observation_ids].tolist(),
            'paired_residuals_m': residuals.tolist(),
            'mapped_center': center.tolist(), 'mapped_normal': normal.tolist(),
            'admitted_to_H': admitted, 'output_available': mode is not None,
            'center_projection_ambiguous': ambiguous, 'output_reason': reason,
            'rejections': rejections, 'potential_winner': False,
            'region': None if mode is None else _region_summary(mode)}
        h = {'transform': transform, 'summary': summary, 'rows': rows,
             'observation_ids': observation_ids, 'center': center, 'normal': normal}
        seen[key] = len(hypotheses)
        hypotheses.append(h)
        diagnostics['hypotheses'].append(summary)
        return h

    def diagnostic_modes():
        """Rebuild a bounded display only; every H/W record remains in JSON."""
        choices = sorted((h for h in hypotheses if h['summary']['output_available']),
            key=lambda h: (not h['summary']['potential_winner'],
                           -h['summary']['lower'], h['summary']['hypothesis_id']))
        displayed = []
        for h in choices[:DISPLAY_DIAGNOSTIC_LIMIT]:
            mode, reason = geometry.region_for_transform(surface, source_pack, h['transform'], **sampling)
            if reason is not None:
                raise RuntimeError('deterministic region reconstruction changed its result')
            mode.update(type='potential_winner_diagnostic' if h['summary']['potential_winner']
                        else 'rejected_or_dominated_diagnostic',
                        hypothesis_index=h['summary']['hypothesis_id'],
                        lower=h['summary']['lower'], upper=h['summary']['upper'])
            displayed.append(mode)
        diagnostics['diagnostic_regions_available'] = len(choices)
        diagnostics['diagnostic_regions_displayed'] = len(displayed)
        return displayed

    try:
        diagnostics.update(search_stage='complete_representative_ranking_and_proposals',
                           search_status='running')
        proposals, proposal_diagnostics = build_proposals(source_pack, surface, appearance,
                                                        initial_transforms=initial_transforms)
        diagnostics['proposals'] = proposal_diagnostics
        diagnostics['search_stage'] = 'complete_query_scoring_and_output_feasibility'
        for proposal in proposals:
            transform = proposal['transform']
            h = evaluate(transform, proposal['origins'])
            rows, observation_ids = h['rows'], h['observation_ids']
            if len(rows) < 6 or rank(queries[rows])[0] < 2:
                diagnostics['refit_unavailable_count'] += 1
                continue
            diagnostics['refit_attempts'] += 1
            try:
                refined = fit_sim3(queries[rows], surface['observation_xyz'][observation_ids])
            except (ValueError, np.linalg.LinAlgError):
                diagnostics['refit_invalid_count'] += 1
                continue
            if not valid_transform(refined):
                diagnostics['refit_invalid_count'] += 1
                continue
            # Keep the original even when refitting reduces geometric residual.
            evaluate(refined, [{'origin': 'one_geometry_refit',
                'parent_proposal_id': proposal['proposal_id'],
                'parent_hypothesis_id': h['summary']['hypothesis_id'],
                'query_rows': rows.tolist(),
                'observation_indices': observation_ids.tolist()}])
        diagnostics.update(search_stage='interval_and_observed_region_decision',
                           evaluated_transform_count=len(hypotheses))
        admitted = [h for h in hypotheses if h['summary']['admitted_to_H']]
        diagnostics['hypotheses_admitted'] = len(admitted)
        winner_mode, unique_transform = None, None
        if not admitted:
            result.update(status='rejected_no_supported_region_in_H',
                          reasons=['no_feasible_hypothesis_in_declared_search'])
        else:
            best_lower = max(h['summary']['lower'] for h in admitted)
            winners = [h for h in admitted if h['summary']['upper'] >= best_lower]
            for h in winners:
                h['summary']['potential_winner'] = True
            diagnostics.update(best_lower=best_lower,
                potential_winner_ids=[h['summary']['hypothesis_id'] for h in winners],
                potential_winner_count=len(winners))
            if not any(h['summary']['known_region_count'] for h in winners):
                result.update(status='insufficient_appearance', reasons=['no_potential_winner_region_raw_score'])
            elif any(h['summary']['center_projection_ambiguous'] for h in winners):
                result.update(status='rejected_ambiguous_region', reasons=['center_projection_ambiguous'])
            elif len(winners) == 1:
                winner_mode, reason = geometry.region_for_transform(
                    surface, source_pack, winners[0]['transform'], **sampling)
                if reason is not None:
                    raise RuntimeError('deterministic winning region reconstruction changed its result')
                unique_transform = winners[0]
            else:
                def winning_modes():
                    for h in winners:
                        mode, reason = geometry.region_for_transform(surface, source_pack, h['transform'], **sampling)
                        if reason is not None:
                            raise RuntimeError('deterministic winning region reconstruction changed its result')
                        yield mode
                winner_mode, reason = geometry.consensus_region(surface, winning_modes(),
                    np.asarray([h['center'] for h in winners]),
                    np.asarray([h['normal'] for h in winners]), **sampling)
                if reason is not None:
                    result.update(status='rejected_ambiguous_region', reasons=[reason])
                else:
                    diagnostics['quality_warnings'].append('multiple_transforms_common_region_only')
            if winner_mode is not None:
                winner_mode.update(type='primary',
                    hypothesis_indices=[h['summary']['hypothesis_id'] for h in winners])
                result.update(accepted=True, status='accepted_finite_domain',
                              region_identified=True, modes=[winner_mode])
                diagnostics['region_identified'] = True
                if any(h['summary']['unknown_count'] for h in winners):
                    diagnostics['quality_warnings'].append('potential_winner_has_unknown_query_appearance')
                if unique_transform is not None:
                    transform = unique_transform['transform']
                    result.update(transform_identified=True, sim3_scale=transform[0],
                        sim3_rotation=transform[1], sim3_translation=transform[2],
                        src_anchor_idx=query_indices[unique_transform['rows']],
                        tgt_anchor_idx=surface['observation_to_element'][unique_transform['observation_ids']])
                    diagnostics['transform_identified'] = True
        if not result['accepted']:
            result['modes'] = diagnostic_modes()
        diagnostics.update(search_status='finite_domain_complete', search_stage='complete')
    except (MemoryError, InterruptedError) as error:
        # The runner also records external termination. A caught resource stop
        # cannot inherit acceptance from partially completed candidate work.
        result.update(accepted=False, status='search_incomplete',
            reasons=['declared_finite_search_interrupted'], modes=[],
            region_identified=False, transform_identified=False,
            sim3_scale=None, sim3_rotation=None, sim3_translation=None)
        diagnostics.update(search_status='search_incomplete',
            resource_failure=type(error).__name__, resource_message=str(error),
            region_identified=False, transform_identified=False)
    diagnostics['decision_status'] = result['status']
    return result
# END independent dual-view full-query scoring and region decision.
