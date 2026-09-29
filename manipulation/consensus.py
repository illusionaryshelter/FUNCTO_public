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
