"""Rigid correspondence with observed context at the current region centre.

All positions use metres. Source identities have equal empirical weight and
may reuse target observation elements. The source region has positive RMS;
the fixed observed owner has positive variance. These are the data meanings
of the numerical problem. Owner moments are observation statistics, rather
than a physical centre, surface normal, or calibrated confidence estimate.

The public cross-instance pipeline and independent numerical checks share
this array interface. Its optimizer is SciPy's six-variable nonlinear least-squares
solver; the caller owns subsequent neural refinement and rigid grasp generation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from .context import _center, context_vector, fit_rigid_with_context


# BEGIN candidate-centre context and one shared correspondence objective.
def owner_moments(owner):
    """Return the fixed observed mean and empirical squared radius in m²."""
    mean, centered = _center(owner)
    return mean, float(np.mean(np.sum(centered * centered, axis=1)))


def context_at(center, owner_mean, owner_variance):
    """Return d(c) and its 3x3 derivative with respect to metric centre c.

    With u=c-mu and rho²=variance+uᵀu, d=u/rho. The derivative has units
    1/metre; the normalized translation derivative also multiplies source RMS.
    At c=mu the vector is zero while the derivative remains I/sqrt(variance).
    """
    u = np.asarray(center, dtype=float) - owner_mean
    rho = float(np.sqrt(owner_variance + u @ u))
    return u / rho, np.eye(3) / rho - np.outer(u, u) / rho ** 3


def joint_residual(parameters, x, paired_y, source_context, owner_mean,
                   owner_variance, origin, length, initial_rotation, initial_translation):
    """Six-variable residual for fixed source-to-target identities.

    x is centred and normalized by the source-region RMS. R=Exp(w)R_initial
    and t=t_initial+parameters[3:] give mapped world points origin+length*(R*x+t).
    Their centre is origin+length*t. Geometry is divided by sqrt(N), so the
    squared residual norm is mean geometry plus one unit-weight context term.
    SciPy minimizes half this norm; appearance is constant in this pose block.
    """
    rotation = Rotation.from_rotvec(parameters[:3]).as_matrix() @ initial_rotation
    translation = initial_translation + parameters[3:]
    target_context, _ = context_at(origin + length * translation, owner_mean, owner_variance)
    geometry = (x @ rotation.T + translation - paired_y) / np.sqrt(len(x))
    return np.concatenate((geometry.ravel(), rotation @ source_context - target_context))


def fit_candidate_context(x, paired_y, source_context, owner_mean, owner_variance,
                          origin, length, rotation, translation):
    """Locally fit R/t with the existing SciPy solver and numerical Jacobian.

    The returned OptimizeResult retains termination status, evaluations and
    gradient. A successful local termination does not certify global optimum,
    correct correspondences, or physical pose accuracy. Numerical tolerances
    set optimizer precision, rather than task acceptance rules.
    """
    # Both pose components use increments. An absolute translation very near
    # zero would give this installed TRF implementation a near-zero initial
    # trust radius. Exact zero increments use its standard unit initial radius
    # without changing this objective, its units, tolerances, or feasible poses.
    parameters = np.zeros(6)
    result = least_squares(
        joint_residual, parameters,
        args=(x, paired_y, source_context, owner_mean, owner_variance, origin, length, rotation, translation),
        jac='2-point', method='trf', loss='linear',
        ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=100)
    fitted_rotation = Rotation.from_rotvec(result.x[:3]).as_matrix() @ rotation
    return fitted_rotation, translation + result.x[3:], result


def objective_terms(x, y, appearance, columns, rotation, translation,
                    source_context, owner_mean, owner_variance, origin, length):
    """Appearance, normalized squared distance, current context, and total."""
    mapped = x @ rotation.T + translation
    target_context, _ = context_at(origin + length * translation, owner_mean, owner_variance)
    terms = np.array([
        appearance[np.arange(len(x)), columns].mean(),
        np.mean(np.sum((mapped - y[columns]) ** 2, axis=1)),
        np.sum((rotation @ source_context - target_context) ** 2),
    ])
    return np.concatenate((terms, [terms.sum()]))


def register_candidate_context(source, target, raw, source_frame, source_owner,
                               target_owner, *, steps=40, channels=1024):
    """Fit every original region identity against the full observed domain.

    The 25 initial correspondence sets retain the existing raw/rotation-grid
    construction. Each begins from a fixed-context SVD evaluated at its paired
    centre. Thereafter every R/t and assignment block evaluates one shared
    moving-centre objective. At fixed pose context is independent of assignment,
    so the complete target-domain row argmin remains exact.

    Outputs retain all states and solver diagnostics. No solver-status-based
    candidate rejection or alternate-method branch is introduced. The caller
    decides how these numerical predictions are inspected and consumed.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    raw = np.asarray(raw, dtype=float)
    center = source.mean(axis=0)
    length = float(np.sqrt(np.mean(np.sum((source - center) ** 2, axis=1))))
    x = (source - center) / length
    raw_columns = raw.argmax(axis=1)
    origin = target[raw_columns].mean(axis=0)
    y = (target - origin) / length
    source_context = context_vector(source_owner, center)['vector']
    mean, variance = owner_moments(target_owner)
    appearance = (raw.max(axis=1, keepdims=True) - raw) / channels
    raw_rotation = fit_rigid_with_context(x, y[raw_columns], source_context,
        context_at(origin, mean, variance)[0], kappa=0.)['rotation']
    basis = np.asarray(source_frame)[:3, :3]
    seeds = [raw_columns]
    names = ['native_raw_argmax']
    for index, element in enumerate(Rotation.create_group('O').as_matrix()):
        rotation = raw_rotation @ basis @ element @ basis.T
        seeds.append((appearance + cdist(x @ rotation.T, y, 'sqeuclidean')).argmin(axis=1))
        names.append('source_frame_octahedral_' + str(index))

    histories = []
    for seed in seeds:
        columns = seed.copy()
        seed_context = context_at(origin + length * y[columns].mean(axis=0), mean, variance)[0]
        initial = fit_rigid_with_context(x, y[columns], source_context, seed_context)
        rotation, translation = initial['rotation'], initial['translation']
        history = []
        for _ in range(steps):
            before = objective_terms(x, y, appearance, columns, rotation, translation,
                                     source_context, mean, variance, origin, length)
            start_rotation, start_translation = rotation.copy(), translation.copy()
            rotation, translation, optimizer = fit_candidate_context(
                x, y[columns], source_context, mean, variance, origin, length, rotation, translation)
            fitted = objective_terms(x, y, appearance, columns, rotation, translation,
                                     source_context, mean, variance, origin, length)
            normalized_map = x @ rotation.T + translation
            fresh = (appearance + cdist(normalized_map, y, 'sqeuclidean')).argmin(axis=1)
            assigned = objective_terms(x, y, appearance, fresh, rotation, translation,
                                       source_context, mean, variance, origin, length)
            current_center = origin + length * translation
            history.append(dict(
                mapped_points_world_m=origin + length * normalized_map,
                rotations=rotation, translations_normalized=translation,
                target_indices=fresh, fit_target_indices=columns,
                fit_start_rotations=start_rotation, fit_start_translations_normalized=start_translation,
                before_fit_energy_components=before, after_fit_energy_components=fitted,
                energy_components=assigned,
                context_target_vectors=context_at(current_center, mean, variance)[0],
                centers_world_m=current_center,
                solver_parameters=optimizer.x, solver_status=optimizer.status,
                solver_success=optimizer.success, solver_message=optimizer.message,
                solver_optimality=optimizer.optimality,
                solver_gradient_half_sse=optimizer.grad,
                solver_cost_half_sse=optimizer.cost,
                solver_nfev=optimizer.nfev, solver_njev=optimizer.njev))
            columns = fresh
        histories.append(history)

    arrays = {'coarse_' + key: np.asarray([[state[key] for state in history] for history in histories])
              for key in histories[0][0]}
    arrays.update(coarse_initial_target_indices=np.asarray(seeds),
                  coarse_seed_names=np.asarray(names), appearance_cost=appearance,
                  coarse_context_source_vector=source_context)
    selected = int(arrays['coarse_energy_components'][:, -1, -1].argmin())
    report = dict(
        status='computed_coarse_feasibility', method='candidate_center_context_rigid_correspondence',
        source_rows=len(source), target_elements=len(target), candidate_count=len(seeds),
        fits_per_candidate=steps, selected_candidate=selected, selected_seed=names[selected],
        source_center_world_m=center, source_rms_m=length, target_origin_world_m=origin,
        target_owner_mean_world_m=mean, target_owner_variance_m2=variance,
        descriptor_channels=channels, context_weight=1.,
        energy_terms=['appearance', 'normalized_squared_distance', 'context_squared_error', 'total'],
        selected_energy=arrays['coarse_energy_components'][selected, -1],
        solver=dict(api='scipy.optimize.least_squares', method='trf', loss='linear',
                    jac='2-point', ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=100,
                    gradient='half squared residual; rotation and normalized translation increments about saved block start',
                    parameterization='six zero-initialized increments; R=Exp(w)R_start, t=t_start+delta_t'),
        consumption='full-source-identity coarse initialization; shared by the public cross-instance pipeline and independent checks')
    return arrays, report
# END one objective is shared by pose fitting, full-domain assignment and ranking.
