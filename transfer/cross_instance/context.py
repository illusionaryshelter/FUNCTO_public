"""Observed region/context relations and independent rigid grasp ranking.

All public functions consume arrays, without task names or simulator state.
Regions retain equal weight for every original source identity; owners contain
their unique visible observations. Positions and tool anchors are metres.
Source and target worlds may have independent rigid coordinate changes.

Region RMS lengths and the owner RMS about each requested region centre must
be positive. These are mathematical input conditions, not grasp acceptance
thresholds. A zero context vector and a nonunique rotation remain valid
diagnostics. Relation costs rank candidates; robot geometry, observed unknown
space, contact and execution remain responsibilities of their own consumers.
"""

from __future__ import annotations

import numpy as np


# BEGIN complete-region and observed-context numerical interfaces.
def _center(points):
    """Centre after removing one point so constant clouds stay exactly zero."""
    points = np.asarray(points, dtype=float)
    relative = points - points[0]
    relative_mean = relative.mean(axis=0)
    return points[0] + relative_mean, relative - relative_mean


def _proper_rotation(covariance, sample_count):
    """Proper Procrustes rotation and its numerical uniqueness diagnostic.

    A rank-two paired covariance can determine the proper rotation. The
    smallest rotation curvature is sigma_2 + parity*sigma_3; checking each
    cloud's rank independently would not establish this pairing property.
    """
    left, singular, right = np.linalg.svd(covariance)
    parity = 1.0 if np.linalg.det(left @ right) >= 0.0 else -1.0
    rotation = (left * np.array([1.0, 1.0, parity])) @ right
    curvature = singular[1] + parity * singular[2]
    numerical_zero = np.finfo(float).eps * max(sample_count, 3) * singular[0]
    return rotation, singular, bool(curvature > numerical_zero)


def context_vector(owner, region_center):
    """Bound the observed owner-to-region relation without normalizing its tip.

    The denominator squared equals owner variance plus squared centre offset,
    so ||vector|| <= 1. Coincident means therefore yield zero evidence exactly.
    This empirical observation relation is neither a physical surface normal
    nor a declaration of the object's unobserved centre or occupied volume.
    """
    owner = np.asarray(owner, dtype=float)
    region_center = np.asarray(region_center, dtype=float)
    owner_center, _ = _center(owner)
    rms = float(np.sqrt(np.mean(np.sum((owner - region_center) ** 2, axis=1))))
    return dict(vector=(region_center - owner_center) / rms,
                owner_center=owner_center, context_rms_m=rms, count=len(owner))


def fit_rigid_with_context(source_normalized, paired_target_normalized,
                           d_source, d_target, kappa=1.0):
    """Exactly fit one fixed correspondence with one fixed context relation.

    Minimize mean(||R*x_i+t-y_i||²)+kappa*||R*d_source-d_target||².
    The caller freezes both context vectors throughout correspondence search;
    changing d_target when J changes would define another assignment block.
    The rotation is proper, the translation uses the complete paired means,
    and the returned uniqueness flag is diagnostic rather than a gate.
    """
    source_mean, x = _center(source_normalized)
    target_mean, y = _center(paired_target_normalized)
    ds, dt = np.asarray(d_source, dtype=float), np.asarray(d_target, dtype=float)
    covariance = y.T @ x / len(x) + kappa * np.outer(dt, ds)
    rotation, singular, unique = _proper_rotation(covariance, len(x))
    translation = target_mean - rotation @ source_mean
    return dict(rotation=rotation, translation=translation,
                covariance_singular_values=singular, rotation_unique=unique,
                context_squared_error=float(np.sum((rotation @ ds - dt) ** 2)))


def read_region_frame(source, mapped, source_frame):
    """Summarize complete identity pairs by their least-squares similarity.

    This geometric summary transports the original functional origin with
    the fitted scale and translation, and rotates its unit axes. It does not
    scale the robot, its tool anchor, or the source skill's metric H_g offsets.
    Source variance must be positive. Nonunique rotations still admit an SVD
    minimizer: the returned frame is one such numerical summary, accompanied
    by rotation_unique=False, rather than a uniquely observed direction.

    Independent grasp evidence consumes the complete mapped rows directly.
    This summary's uniqueness therefore supplies a diagnostic, not a gate on
    grasp generation or a substitute for the observed owner context.
    """
    source = np.asarray(source, dtype=float)
    mapped = np.asarray(mapped, dtype=float)
    original = np.asarray(source_frame, dtype=float)
    source_center, p = _center(source)
    target_center, q = _center(mapped)
    fit = fit_rigid_with_context(source, mapped, np.zeros(3), np.zeros(3), kappa=0.0)
    rotation = fit['rotation']
    rotated = p @ rotation.T
    scale = float(np.sum(rotated * q) / np.sum(p * p))
    translation = target_center - scale * rotation @ source_center
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    frame = np.eye(4)
    frame[:3, :3] = rotation @ original[:3, :3]
    frame[:3, 3] = target_center + scale * rotation @ (original[:3, 3] - source_center)
    residuals = np.linalg.norm(scale * rotated - q, axis=1)
    return dict(status='computed', matrix=matrix, rotation=rotation,
                translation=translation, scale=scale, T_world_region=frame,
                source_center=source_center, target_center=target_center,
                rms_residual_m=float(np.sqrt(np.mean(residuals ** 2))),
                residuals_m=residuals, paired_points=len(source),
                covariance_singular_values=fit['covariance_singular_values'],
                rotation_unique=fit['rotation_unique'],
                construction='least_squares_similarity_of_complete_source_identities',
                origin_centroid_offset_m=float(np.linalg.norm(original[:3, 3] - source_center)))


def orientation_evidence(source_region, mapped_region, source_owner, target_owner,
                         kappa=1.0):
    """Combine full identity-preserving region pairs and visible owner context.

    Both region arrays use the same original source rows. Each is centred and
    divided by its own RMS, giving mean squared norm one. Consequently the
    local relation cost has constant 2. Context contributes its actual bounded
    vector norms. The returned bound is a uniform upper bound on this complete
    direction cost, making its normalized value lie in [0,1].

    OAReg's target measurements and its mapped source rows have different
    roles: this function consistently uses mapped rows for the target centre,
    RMS, covariance and context relation. No uniqueness of a physical frame
    is inferred from numerical rotation uniqueness.
    """
    source_center, p = _center(source_region)
    target_center, q = _center(mapped_region)
    source_rms = float(np.sqrt(np.mean(np.sum(p * p, axis=1))))
    target_rms = float(np.sqrt(np.mean(np.sum(q * q, axis=1))))
    source_context = context_vector(source_owner, source_center)
    target_context = context_vector(target_owner, target_center)
    ds, dt = source_context["vector"], target_context["vector"]
    h = (q / target_rms).T @ (p / source_rms) / len(p)
    matrix = h + kappa * np.outer(dt, ds)
    rotation, singular, unique = _proper_rotation(matrix, len(p))
    direction_constant = 2.0 + kappa * float(ds @ ds + dt @ dt)
    direction_bound = 4.0 + kappa * float((np.linalg.norm(ds) + np.linalg.norm(dt)) ** 2)
    return dict(matrix=matrix, source_center=source_center, target_center=target_center,
                source_rms_m=source_rms, target_rms_m=target_rms,
                d_source=ds, d_target=dt, direction_constant=direction_constant,
                direction_bound=direction_bound, rotation_wahba=rotation,
                rotation_unique=unique, source_context=source_context,
                target_context=target_context, covariance_singular_values=singular)


def _source_grasp_anchor(evidence, source_grasp):
    """Original region centre in the measured source grasp G, in metres."""
    return source_grasp[:3, :3].T @ (evidence["source_center"] - source_grasp[:3, 3])


def context_grasp_proposal(evidence, source_G):
    """One ordinary rigid proposal with optimal relation rotation and anchor.

    Its position places the unchanged metric source anchor at the mapped
    region centre. The region scale does not scale the robot or anchor. Model
    quality is evaluated later by the same scorer used for all other poses.
    """
    source_grasp = np.asarray(source_G, dtype=float)
    anchor = _source_grasp_anchor(evidence, source_grasp)
    rotation = evidence["rotation_wahba"] @ source_grasp[:3, :3]
    proposal = np.eye(4)
    proposal[:3, :3] = rotation
    proposal[:3, 3] = evidence["target_center"] - rotation @ anchor
    return proposal


# BEGIN complete controlled-tool pose prior replaces label-minimized ranking.
def rank_candidates(tool_poses_world, quality, tool_prior_world, position_scale_m):
    """Rank actual controlled E poses against one complete E pose prior.

    Inputs are a nonempty Nx4x4 array of proper rigid tool poses, one native
    quality value per pose, and a 4x4 prior for the same controlled tool origin
    and axes. All positions share the target world and metre unit. The positive
    position_scale_m is the saved region RMS length, a dimensional normalization
    rather than an orientation confidence or an acceptance threshold.

    The three unit-weight terms are ||p-p_prior||²/position_scale_m²,
    ||R-R_prior||_F²/8, and 1-quality. For relative rotation angle theta, the
    orientation term equals sin²(theta/2), spanning [0,1]. It is a soft cost on
    the complete physical pose, including rotation about the approach axis.
    Alternative grasp labels enter as explicitly bound physical E poses before
    this call; every row is scored exactly as supplied, without label changes.

    Returned angle and position errors describe this frozen candidate/prior
    comparison. They do not control execution. Stable sorting preserves the
    supplied order of exact ties. The SO(3) angle uses atan2 of the relative
    matrix's skew magnitude and trace, including both zero and pi endpoints.
    """
    poses = np.asarray(tool_poses_world, dtype=float)
    quality = np.asarray(quality, dtype=float)
    prior = np.asarray(tool_prior_world, dtype=float)
    position_squared = np.sum((poses[:, :3, 3] - prior[:3, 3]) ** 2, axis=1)
    position_cost = position_squared / float(position_scale_m) ** 2
    orientation_cost = np.sum((poses[:, :3, :3] - prior[:3, :3]) ** 2, axis=(1, 2)) / 8.0

    relative = prior[:3, :3].T @ poses[:, :3, :3]
    skew = np.stack((relative[:, 2, 1] - relative[:, 1, 2],
                     relative[:, 0, 2] - relative[:, 2, 0],
                     relative[:, 1, 0] - relative[:, 0, 1]), axis=1)
    sine = 0.5 * np.linalg.norm(skew, axis=1)
    cosine = 0.5 * (np.trace(relative, axis1=1, axis2=2) - 1.0)
    angles = np.arctan2(sine, cosine)

    quality_cost = 1.0 - quality
    total = position_cost + orientation_cost + quality_cost
    order = np.argsort(total, kind="stable")
    return dict(costs_total=total, costs_quality=quality_cost,
                costs_orientation=orientation_cost, costs_position=position_cost,
                orientation_error_rad=angles, position_error_m=np.sqrt(position_squared),
                order=order, selected_index=int(order[0]))
# END complete controlled-tool pose prior replaces label-minimized ranking.
# END complete-region and observed-context numerical interfaces.
