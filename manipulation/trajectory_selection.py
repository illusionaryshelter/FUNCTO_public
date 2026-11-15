"""Apply fixed affine preferences to the original accepted planning set.

The four physical improvement intervals and their weights are caller data.
The same candidate retains the same score across ordering and set changes.
Model confidence, current simulation truth and physical outcomes have no role
in this comparison of completed planned trajectories.
"""
from __future__ import annotations
from math import fsum, radians

def metric_values(case):
    """Expose the declared four metrics in cost units, metres and radians."""
    return dict(
        original_objective=case['original_objective'],
        minimum_joint_position_margin_rad=case['joint']['position']['minimum_margin']['margin'],
        minimum_required_robot_pair_clearance_m=case['robot_self']['minimum']['evidence']['signed_distance_m'],
        tool_rotation_length_rad=radians(case['tool_movement']['rotation_travel_deg']))


def score_metrics(values, criteria):
    """Return affine cost components; reference and swing share native units.

    Positive cost_sign denotes a smaller-is-better metric. Negative denotes
    a larger-is-better metric. Values outside the fixed calibration interval
    retain the same linear exchange rate. Scores are preference units.
    """
    components=[]
    for criterion in criteria:
        value=values[criterion['id']]
        normalized=criterion['cost_sign']*(value-criterion['reference_value'])/criterion['improvement_interval']
        components.append(dict(id=criterion['id'],unit=criterion['unit'],value=value,
            normalized_cost=normalized,weight=criterion['weight'],
            weighted_cost=criterion['weight']*normalized))
    return dict(score=fsum(row['weighted_cost'] for row in components),components=components)


def select(comparison, calibration):
    """Rank only the original accepted set and preserve every exact minimizer.

    An empty accepted set is a planning refusal. Multiple score minimizers
    remain an explicit user choice, following the requested pause boundary;
    display rounding does not create a numerical tie rule.
    """
    scored=[]; rejected=[]; missing=[]
    for case in comparison['cases']:
        if case['accepted']:
            # BEGIN measurement availability cannot relabel planner acceptance.
            if case.get('metrics_status', 'completed') != 'completed':
                missing.append(dict(id=case['id'], accepted=True,
                    reason=case['metrics_status'], source_report=case['source_report'],
                    source_curve=case['source_curve']))
                continue
            # END an incomplete accepted score leaves the comparison incomplete.
            scored.append(dict(id=case['id'],accepted=True,
                **score_metrics(metric_values(case),calibration['criteria']),
                source_report=case['source_report'],source_curve=case['source_curve']))
        else:
            rejected.append(dict(id=case['id'],accepted=False,
                reason=case.get('failure', 'original_planning_rejected'),
                status=case.get('status', 'planning_rejected'),
                source_report=case['source_report'],source_curve=case['source_curve']))
    if scored:
        minimum=min(row['score'] for row in scored)
        best=[row['id'] for row in scored if row['score']==minimum]
    else:
        best=[]
    if missing:
        best=[]
    return dict(status='incomplete_metrics' if missing else 'selected' if len(best)==1 else 'tie_requires_user_choice' if best else 'no_accepted_plan',
        eligible_candidate_ids=[row['id'] for row in comparison['cases'] if row['accepted']],
        scored_candidate_ids=[row['id'] for row in scored],scores=scored,rejected=rejected,
        accepted_with_missing_metrics=missing,
        best_candidate_ids=best,preferred_id=best[0] if len(best)==1 else None,
        physical_status=comparison['physical_status'],calibration=calibration,
        score_direction='min',score_semantics='fixed affine preference units; not a probability')
