"""Held-out transform errors for frozen dual-view predictions only.

Nothing in inference imports this module. A correct observed region is a
separate outcome from a correct scale or full rotation. An anisotropic affine
has a polar rotation but no unique true isotropic Sim(3) scale.
"""

# BEGIN evaluation-only affine reference and frozen-candidate diagnostics.
import numpy as np
from scipy.spatial.transform import Rotation


def affine_reference(affine):
    """Describe a proper affine without inventing an anisotropic true scale."""
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError('Expected a finite held-out 4x4 affine transform')
    if not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-10):
        raise ValueError('Held-out transform must have an affine homogeneous row')
    linear = affine[:3, :3]
    if np.linalg.det(linear) <= 0:
        raise ValueError('Transform evaluation requires a proper nonsingular affine')
    left, singular, right = np.linalg.svd(linear)
    is_sim3 = bool(np.allclose(singular, singular[0], rtol=1e-10, atol=1e-10))
    return {'truth_is_sim3': is_sim3,
            'true_scale': float(singular[0]) if is_sim3 else None,
            'affine_singular_values': singular.tolist(),
            'polar_rotation': (left @ right).tolist(),
            'volume_equivalent_scale_descriptive_only': float(np.cbrt(np.linalg.det(linear))),
            'scale_error_definition': '100*(estimated_scale/true_scale-1)' if is_sim3
                else 'undefined_for_anisotropic_affine_no_unique_true_sim3_scale'}


def _normal_angle(first, second):
    """Compare oriented unit normals, including a reversed-face error of 180°."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    first, second = first/np.linalg.norm(first), second/np.linalg.norm(second)
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(first, second)), first @ second)))


def transform_error(transform, affine, source):
    """Evaluate one estimate on the unchanged local and context source rows.

    The local mask covers both fitting queries and independent checks. The
    transformed source center, rather than a snapped target seed, supplies the
    transform's center error; the report retains region-center error separately.
    """
    scale, rotation, translation = transform
    scale, rotation, translation = float(scale), np.asarray(rotation, float), np.asarray(translation, float)
    if (not np.isfinite(scale) or scale <= 0 or rotation.shape != (3, 3)
            or translation.shape != (3,) or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            or np.linalg.det(rotation) <= 0):
        raise ValueError('Frozen estimate must be a positive-scale proper Sim(3)')
    reference = affine_reference(affine)
    affine = np.asarray(affine, float)
    points = np.vstack((source['query_xyz'], source['check_xyz']))
    region = np.r_[source['query_region'], source['check_query_region']].astype(bool)
    if points.shape != (len(region), 3) or not np.isfinite(points).all():
        raise ValueError('Fixed query/check points and local membership must align')
    mapped = scale*(points @ rotation.T)+translation
    target = points @ affine[:3, :3].T+affine[:3, 3]
    error2 = np.sum((mapped-target)**2, axis=1)
    center, normal = np.asarray(source['center'], float), np.asarray(source['normal'], float)
    if center.shape != (3,) or normal.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
        raise ValueError('Source center and oriented normal must be finite 3D vectors')
    scale_error = None if reference['true_scale'] is None else 100*(scale/reference['true_scale']-1)
    return {'estimated_scale': scale, 'scale_signed_error_percent': scale_error,
            'scale_absolute_error_percent': None if scale_error is None else abs(scale_error),
            'rotation_vs_polar_deg': float(np.degrees(Rotation.from_matrix(
                np.asarray(reference['polar_rotation']).T @ rotation).magnitude())),
            'normal_error_deg': _normal_angle(rotation @ normal, np.linalg.solve(affine[:3, :3].T, normal)),
            'local_mapping_rmse_mm': float(1000*np.sqrt(error2[region].mean())) if region.any() else None,
            'context_mapping_rmse_mm': float(1000*np.sqrt(error2[~region].mean())) if (~region).any() else None,
            'mapped_center_error_mm': float(1000*np.linalg.norm(scale*(rotation @ center)+translation-
                (affine[:3, :3] @ center+affine[:3, 3]))),
            'local_point_count': int(region.sum()), 'context_point_count': int((~region).sum())}


def _ranges(records):
    """Ranges describe a finite saved set; they are never confidence bounds."""
    keys = ('estimated_scale', 'scale_signed_error_percent', 'rotation_vs_polar_deg',
            'normal_error_deg', 'local_mapping_rmse_mm', 'context_mapping_rmse_mm')
    result = {}
    for key in keys:
        values = [record[key] for record in records if record[key] is not None]
        result[key] = {'min': min(values), 'max': max(values)} if values else None
    return result


def evaluate_transforms(result, source, affine, *, primary_transform=None):
    """Keep the published estimate and displayed candidates distinct from GT.

    Candidate IDs come exclusively from the frozen displayed mode. No held-out
    error selects, removes, ranks, or certifies a prediction. A rejected result
    may retain diagnostics but never acquires a primary estimated transform.
    """
    diagnostic = result['diagnostics']
    candidates = {h['hypothesis_id']: h for h in diagnostic.get('hypotheses', [])}
    first = result['modes'][0] if result['modes'] else None
    ids = ([] if first is None else first['hypothesis_indices']
           if 'hypothesis_indices' in first else [first['hypothesis_index']])
    records = []
    for identifier in ids:
        candidate = candidates[identifier]
        records.append({'hypothesis_id': identifier,
            **transform_error(candidate['transform'], affine, source)})
    # Additional accepted-result modes are competitor displays, not extra
    # accepted estimates. Their GT errors remain separate from primary metrics.
    additional = []
    for mode in result['modes'][1:]:
        mode_ids = mode['hypothesis_indices'] if 'hypothesis_indices' in mode else [mode['hypothesis_index']]
        additional.append({'mode_type': mode.get('type'),
            'role': 'competitor_or_rejected_diagnostic_not_primary',
            'candidates': [{'hypothesis_id': identifier,
                **transform_error(candidates[identifier]['transform'], affine, source)}
                for identifier in mode_ids]})
    primary = (transform_error(primary_transform, affine, source)
               if result['accepted'] and primary_transform is not None else None)
    return {'evaluation_only': True, 'target_truth_used_for_prediction': False,
            'reference': affine_reference(affine), 'primary_estimate': primary,
            'displayed_candidate_role': 'accepted_region_candidates' if result['accepted'] else 'rejected_diagnostics_only',
            'displayed_candidates': records, 'displayed_candidate_ranges': _ranges(records),
            'additional_diagnostic_modes': additional,
            'finite_candidate_unique': diagnostic.get('finite_candidate_unique'),
            'transform_certified': False,
            'finite_range_is_confidence_interval': False,
            'sampling_radius_m': diagnostic.get('sampling_radius_m'),
            'sampling_scale_applied': diagnostic.get('sampling_scale_applied')}
# END evaluation-only transform metrics.
