"""Exhaustive rigid hypotheses from a small set of explicit point pairs.

All point coordinates and residuals are metres. Three noncollinear pairs are
minimal hypotheses, not fixed semantic roles. Every final hypothesis is fitted
to its whole stable consensus using the existing Open3D point-to-point kernel.
The consistency threshold is not a sensor-noise or pose-accuracy certificate.
"""

# BEGIN unordered explicit-correspondence rigid consensus.
import itertools
import math
from numbers import Integral, Real

import numpy as np

from .observations import _noncollinear, _RigidCorrespondences


def _identity(value):
    """Keep JSON string/integer identities distinct; never infer spatial order."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    raise ValueError("Pair IDs must be nonempty strings or integers, not booleans")


def _identity_key(value):
    return (0, value) if isinstance(value, int) else (1, value)


def _validate(source, target, pair_ids, max_error):
    """Reject repeated samples and repeated identities rather than giving votes."""
    if np.iscomplexobj(source) or np.iscomplexobj(target):
        raise ValueError("Metric point coordinates must be real")
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Source and target must be finite corresponding N-by-3 arrays")
    if (not isinstance(max_error, Real) or isinstance(max_error, (bool, np.bool_))
            or not np.isfinite(max_error) or max_error <= 0):
        raise ValueError("max_error must be a finite positive metric distance")
    try:
        raw_pairs = list(pair_ids)
    except TypeError as error:
        raise ValueError("pair_ids must contain one source/target ID pair per point") from error
    if len(raw_pairs) != len(source):
        raise ValueError("pair_ids must contain one source/target ID pair per point")
    pairs = []
    for pair in raw_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each pair ID must be a (source_id, target_id) pair")
        pairs.append((_identity(pair[0]), _identity(pair[1])))
    if (len({p[0] for p in pairs}) != len(pairs)
            or len({p[1] for p in pairs}) != len(pairs)):
        raise ValueError("Source and target IDs must each be unique one-to-one identities")
    for name, points in (("source", source), ("target", target)):
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError(f"Repeated {name} 3D samples cannot receive separate votes")
    order = sorted(range(len(pairs)),
                   key=lambda i: (_identity_key(pairs[i][0]), _identity_key(pairs[i][1])))
    # Canonicalizing explicit IDs makes all accumulation, fitting, and tie
    # handling invariant to a joint permutation of the supplied pair rows.
    return (source[order], target[order], [pairs[i] for i in order],
            np.asarray(order, dtype=int), float(max_error))


def _fit(source, target):
    """Reuse the same proper rigid estimator as measured-part registration."""
    model = _RigidCorrespondences()
    if not model.estimate(source, target):
        return None
    rotation = model.params[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7)):
        return None
    return model


def _stable_consensus(source, target, seed, max_error):
    """Refit and rescore until fitted rows and accepted rows are identical.

    A finite-set history rejects numerical cycles explicitly. No failed seed
    falls back to its original triplet or to an earlier, obsolete consensus.
    """
    seed_rows = list(seed)
    if not _noncollinear(source[seed_rows]) or not _noncollinear(target[seed_rows]):
        return None, {"reason": "collinear_seed", "refit_steps": 0}
    model = _fit(source[seed_rows], target[seed_rows])
    if model is None:
        return None, {"reason": "invalid_rigid_fit", "refit_steps": 0}
    errors = model.residuals(source, target)
    if not np.isfinite(errors).all() or np.any(errors[seed_rows] > max_error):
        return None, {"reason": "seed_inconsistent", "refit_steps": 0}
    consensus = tuple(np.flatnonzero(errors <= max_error))
    seen = set()
    steps = 0
    while True:
        if consensus in seen:
            return None, {"reason": "consensus_cycle", "refit_steps": steps}
        seen.add(consensus)
        indices = list(consensus)
        if not _noncollinear(source[indices]) or not _noncollinear(target[indices]):
            return None, {"reason": "degenerate_consensus", "refit_steps": steps}
        model = _fit(source[indices], target[indices])
        steps += 1
        if model is None:
            return None, {"reason": "invalid_rigid_fit", "refit_steps": steps}
        errors = model.residuals(source, target)
        if not np.isfinite(errors).all():
            return None, {"reason": "nonfinite_residual", "refit_steps": steps}
        updated = tuple(np.flatnonzero(errors <= max_error))
        if updated == consensus:
            return {"transform": model.params.copy(), "consensus": consensus,
                    "residuals": errors}, {"refit_steps": steps}
        consensus = updated


def rigid_correspondence_hypotheses(source, target, *, pair_ids, max_error=0.01):
    """Return all maximum-support stable rigid hypotheses as JSON-native data.

    pair_ids is a sequence of unique (source_id, target_id) pairs. Both sides
    must be one-to-one, with string or integer identities. Rows already encode
    the proposed association; this function never matches unordered clouds,
    infers correspondence by image order, or changes the association.
    Repeated exact 3D samples on either side are rejected, even under new IDs.

    The function exhaustively considers N-choose-3 seeds and is intended for
    small correspondence sets. It has no fixed point count or semantic roles.
    The default 10 mm threshold follows the existing registration interface
    and is an explicit geometric consistency tolerance, not a noise estimate.

    accepted means one highest-support consensus; ambiguous retains multiple
    highest-support sets without choosing by their residuals. Distinct sets
    need not imply physically distinct poses. Neither state certifies material
    identity, pose accuracy, functional equivalence, or execution. A three-point
    consensus has weak_evidence=True, regardless of input N. insufficient means
    no stable noncollinear consensus. Invalid inputs raise ValueError, and
    failed estimation is never silently repaired.

    Residual arrays use original input order. Index lists refer to original
    rows but are ordered by explicit pair identity. fit_* and inlier_* describe
    the same final set, which actually participated in the final fit.
    Lower-support stable hypotheses and every rejected seed remain diagnostic.
    """
    p, q, pairs, original_indices, threshold = _validate(
        source, target, pair_ids, max_error)
    count = len(p)
    input_pairs = [None] * count
    for pair, index in zip(pairs, original_indices):
        input_pairs[int(index)] = list(pair)
    diagnostics = {
        "input_pair_ids": input_pairs, "num_correspondences": count,
        "max_error_m": threshold,
        "threshold_scope": "geometric consistency only; not a noise or accuracy model",
        "total_seed_count": math.comb(count, 3) if count >= 3 else 0,
        "stable_seed_count": 0, "stable_consensus_count": 0,
        "rejected_seeds": [], "lower_support_candidates": [],
    }
    result = {"status": "insufficient", "candidates": [], "diagnostics": diagnostics}
    if count < 3:
        diagnostics["reason"] = "fewer_than_three_correspondences"
        return result

    grouped = {}
    for seed in itertools.combinations(range(count), 3):
        trace = {
            "input_indices": [int(original_indices[i]) for i in seed],
            "pair_ids": [list(pairs[i]) for i in seed],
        }
        fitted, seed_diagnostics = _stable_consensus(p, q, seed, threshold)
        trace.update(seed_diagnostics)
        if fitted is None:
            diagnostics["rejected_seeds"].append(trace)
            continue
        diagnostics["stable_seed_count"] += 1
        key = fitted["consensus"]
        if key not in grouped:
            fitted["seeds"] = []
            grouped[key] = fitted
        grouped[key]["seeds"].append(trace)

    if not grouped:
        diagnostics["reason"] = "no_stable_noncollinear_consensus"
        return result
    maximum = max(len(key) for key in grouped)
    diagnostics["max_support_count"] = maximum
    diagnostics["stable_consensus_count"] = len(grouped)
    # Full pair sets, not floating residuals or world/image positions, establish
    # deterministic output order. No near-pose merging conceals alternative sets.
    for number, key in enumerate(sorted(grouped, key=lambda k: (-len(k), k)), 1):
        fitted = grouped[key]
        indices = list(key)
        errors = fitted["residuals"]
        original_errors = np.empty(count, dtype=float)
        original_errors[original_indices] = errors
        rows = [int(original_indices[i]) for i in indices]
        identities = [list(pairs[i]) for i in indices]
        candidate = {
            "candidate_id": f"consensus_{number:03d}",
            "transform": fitted["transform"].tolist(),
            "inlier_indices": rows, "inlier_pair_ids": identities,
            "fit_indices": rows.copy(), "fit_pair_ids": [pair.copy() for pair in identities],
            "residuals_m": original_errors.tolist(),
            "support_count": len(indices), "extra_support_count": len(indices) - 3,
            "weak_evidence": len(indices) <= 3,
            "inlier_rmse_m": float(np.sqrt(np.mean(errors[indices] ** 2))),
            "max_inlier_error_m": float(np.max(errors[indices])),
            "seeds": fitted["seeds"],
        }
        destination = (result["candidates"] if len(indices) == maximum
                       else diagnostics["lower_support_candidates"])
        destination.append(candidate)
    result["status"] = "accepted" if len(result["candidates"]) == 1 else "ambiguous"
    return result
# END unordered explicit-correspondence rigid consensus.


# BEGIN exact one-dimensional translation consensus for a caller-known axis.
def axis_translation_hypotheses(source, target, *, direction, pair_ids,
                                max_error=0.01):
    """Estimate only translation along an already evidenced common-frame axis.

    ``source`` and ``target`` are corresponding metric points in the SAME
    reference frame. The caller has already applied any known rotation and
    transverse displacement; this function does not estimate either, identify
    a mechanism, or infer its coordinate origin from an action name.
    ``direction`` is a finite nonzero XYZ vector, normalized here so the signed
    returned ``offset_m`` has metre units. Its sign is a coordinate convention.
    Identity, duplicate-sample and point validation follow the rigid estimator.

    For unit direction a and paired displacement d_i=target_i-source_i, write
    s_i=a.T@d_i and l_i=||d_i-a*s_i||. Then

        ||d_i-a*s||^2 = l_i^2 + (s-s_i)^2.

    Pair i is impossible when l_i exceeds the geometric distance gate tau;
    otherwise it allows the closed interval
    [s_i-sqrt(tau^2-l_i^2), s_i+sqrt(tau^2-l_i^2)]. Maximum correspondence
    support is therefore exact interval overlap, up to reported floating-point
    roundoff. Every maximum overlap occurs at an interval endpoint. Merely
    seeding at s_i would miss overlaps that contain none of their own centres.
    Enumerating at most 2*N endpoints avoids a grid, random seeds, and a fixed
    three-point minimum. Every tied maximum-support SET is retained.

    Within each set's common interval, the equal-weight least-squares solution
    is its mean(s_i), clamped to the interval. Clamping is necessary: a point
    with large transverse residual may allow a much narrower axial interval
    than the others, even though all have the same Euclidean tolerance.
    ``allowed_offset_interval_m`` describes this set's geometric constraints,
    NOT a noise confidence interval, complete physical feasibility interval,
    or a proof that semantic correspondence is correct.

    ``accepted`` means one maximal support set; ``ambiguous`` retains ties;
    ``insufficient`` means no point pair fits the supplied axis. One pair is
    mathematically sufficient for this known-axis problem but is explicitly
    weak evidence. Residuals follow original input order; index lists refer to
    original rows and are sorted by explicit pair identity. The returned 4x4
    transform is translation only, mapping the supplied source coordinates
    to target coordinates; it is not the caller's earlier registration.
    """
    # Reject booleans/string coercion in this new metric API without changing
    # the established rigid consumer's input behaviour.
    for name, values in (("source", source), ("target", target)):
        raw = np.asarray(values)
        if (raw.dtype.kind not in "iuf"
                or any(isinstance(value, (bool, np.bool_))
                       for value in np.asarray(values, dtype=object).flat)):
            raise ValueError(f"{name} coordinates must be real numeric values, not booleans")
    p, t, pairs, original_indices, threshold = _validate(
        source, target, pair_ids, max_error)
    raw_direction = np.asarray(direction)
    if (raw_direction.shape != (3,) or raw_direction.dtype.kind not in "iuf"
            or any(isinstance(value, (bool, np.bool_))
                   for value in np.asarray(direction, dtype=object).flat)):
        raise ValueError("direction must be a finite nonzero real numeric XYZ vector")
    axis = raw_direction.astype(float, copy=True)
    if not np.isfinite(axis).all() or not np.any(axis):
        raise ValueError("direction must be a finite nonzero real numeric XYZ vector")
    # Scaling first prevents overflow/underflow in normalization and preserves
    # the physical sign even for unusually scaled, but finite, caller vectors.
    axis /= np.max(np.abs(axis))
    axis /= np.linalg.norm(axis)
    count = len(p)
    input_pairs = [None] * count
    for pair, index in zip(pairs, original_indices):
        input_pairs[int(index)] = list(pair)
    diagnostics = {
        "input_pair_ids": input_pairs, "num_correspondences": count,
        "direction_unit": axis.tolist(), "max_error_m": threshold,
        "threshold_scope": "geometric consistency only; not a noise or accuracy model",
        "method": "all closed-interval endpoints; exact maximum overlap, then constrained least squares",
        "total_seed_count": 0, "max_support_count": 0,
        "interval_scope": "same inlier set's geometric allowance; not statistical confidence",
        "numerical_tolerance_m": 0.0,
        "pair_evidence": [],
    }
    result = {"status": "insufficient", "candidates": [], "diagnostics": diagnostics}
    if not count:
        diagnostics["reason"] = "no_correspondences"
        return result
    try:
        with np.errstate(over="raise", invalid="raise"):
            delta = t-p
            offsets = delta @ axis
            lateral = np.hypot.reduce(delta-offsets[:, None]*axis, axis=1)
    except FloatingPointError as error:
        raise ValueError("Point displacements are outside finite metric arithmetic") from error
    if not np.isfinite(offsets).all() or not np.isfinite(lateral).all():
        raise ValueError("Point displacements are outside finite metric arithmetic")
    # This allowance covers arithmetic at closed interval endpoints only. It
    # scales with the measured displacements, never a sensor covariance, camera
    # model, scene identifier, or desired offset. The geometric gate is intact.
    scale = max(threshold, float(np.max(np.abs(delta))), float(np.max(np.abs(offsets))))
    roundoff = 64*np.finfo(float).eps*scale
    diagnostics["numerical_tolerance_m"] = float(roundoff)
    # A mathematically tangent pair may round just above tau after a common
    # frame rotation. Admit only the declared arithmetic allowance there and
    # give it a singleton interval, rather than inventing axial clearance.
    feasible = lateral <= threshold+roundoff
    widths = np.zeros(count)
    # The ratio form avoids squaring metre values that may be extremely large
    # or small. A pair exactly on the transverse gate has a singleton interval.
    ratio = np.minimum(lateral[feasible]/threshold, 1.)
    widths[feasible] = threshold*np.sqrt(np.maximum(0., (1.-ratio)*(1.+ratio)))
    with np.errstate(over="ignore", invalid="ignore"):
        lower, upper = offsets-widths, offsets+widths
    if not np.isfinite(lower[feasible]).all() or not np.isfinite(upper[feasible]).all():
        raise ValueError("Allowed offsets are outside finite metric arithmetic")
    for row, pair in enumerate(pairs):
        diagnostics["pair_evidence"].append({
            "input_index": int(original_indices[row]), "pair_ids": list(pair),
            "axial_offset_m": float(offsets[row]), "transverse_error_m": float(lateral[row]),
            "status": "compatible" if feasible[row] else "transverse_inconsistent",
            "allowed_offset_interval_m": [float(lower[row]), float(upper[row])] if feasible[row] else None,
        })
    if not feasible.any():
        diagnostics["reason"] = "no_correspondence_compatible_with_known_axis"
        return result

    endpoints = np.unique(np.concatenate((lower[feasible], upper[feasible])))
    diagnostics["total_seed_count"] = len(endpoints)
    # Closed interval comparisons remain exact. The documented roundoff only
    # handles transverse tangency and checks reconstructed residuals; it never
    # enlarges axial intervals to create overlap across a real geometric gap.
    sets = {tuple(np.flatnonzero(feasible & (lower <= value) & (value <= upper)))
            for value in endpoints}
    maximum = max(map(len, sets))
    maximum_sets = sorted(key for key in sets if len(key) == maximum)
    diagnostics["max_support_count"] = maximum
    diagnostics["distinct_endpoint_support_sets"] = len(sets)
    diagnostics["max_support_set_count"] = len(maximum_sets)
    for number, key in enumerate(maximum_sets, 1):
        indices = list(key)
        left, right = float(np.max(lower[indices])), float(np.min(upper[indices]))
        # Divide before fsum to avoid overflowing an otherwise finite mean.
        # Canonical explicit-ID order also fixes all output and fit ordering.
        mean = math.fsum(float(offsets[i])/len(indices) for i in indices)
        offset = float(np.clip(mean, left, right))
        with np.errstate(over="ignore", invalid="ignore"):
            residuals = np.hypot.reduce(delta-offset*axis, axis=1)
        if not np.isfinite(residuals).all():
            raise ValueError("Fitted residuals are outside finite metric arithmetic")
        if np.any(residuals[indices] > threshold+roundoff):
            raise ArithmeticError("Interval solution disagrees with its Euclidean inlier bound")
        # Membership is defined by the exact computed intervals. Validate that
        # the chosen constrained optimum still represents that maximal set.
        membership = tuple(np.flatnonzero(feasible & (lower <= offset) & (offset <= upper)))
        if membership != key:
            raise ArithmeticError("Constrained fit changed the maximum-overlap support set")
        original_residuals = np.empty(count, float)
        original_residuals[original_indices] = residuals
        rows = [int(original_indices[i]) for i in indices]
        identities = [list(pairs[i]) for i in indices]
        transform = np.eye(4)
        transform[:3, 3] = offset*axis
        # hypot/fnorm avoids overflow from squaring large finite residuals.
        rmse = float(np.hypot.reduce(residuals[indices]/np.sqrt(len(indices))))
        result["candidates"].append({
            "candidate_id": f"axis_consensus_{number:03d}",
            "offset_m": offset, "direction_unit": axis.tolist(),
            "transform": transform.tolist(),
            "inlier_indices": rows, "inlier_pair_ids": identities,
            "fit_indices": rows.copy(), "fit_pair_ids": [pair.copy() for pair in identities],
            "residuals_m": original_residuals.tolist(),
            "support_count": len(indices), "extra_support_count": len(indices)-1,
            "weak_evidence": len(indices) == 1,
            "unconstrained_offset_mean_m": mean,
            "allowed_offset_interval_m": [left, right],
            "interval_scope": diagnostics["interval_scope"],
            "inlier_rmse_m": rmse,
            "max_inlier_error_m": float(np.max(residuals[indices])),
        })
    result["status"] = "accepted" if len(result["candidates"]) == 1 else "ambiguous"
    return result
# END exact one-dimensional translation consensus for a caller-known axis.
