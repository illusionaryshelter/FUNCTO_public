"""Interchangeable robust Sim(3) estimators for paired 3-D observations.

RANSAC preserves the experiment's original samples, scoring, and final fits.
PROSAC changes only sampling priority and adds simple local optimization when
a new best hypothesis is found.  TEASER++ is an optional native dependency;
selecting it never silently substitutes another estimator.  All methods use
the same final Euclidean residual test and consensus-refitting procedure.
"""

from dataclasses import dataclass
import math

import numpy as np

from .sim3 import _fit_sim3_validated, _points, apply_sim3


# BEGIN selectable correspondence-based Sim(3) registration.
@dataclass(frozen=True)
class RegistrationResult:
    """A transform plus a mask indexed in the caller's original pair order.

    Failed geometric estimation has a non-None ``reason``, no transform or
    residual array, and an all-false inlier mask.  Missing optional software is
    an exception rather than a geometric failure.  Diagnostic native solutions
    are recorded before the common final refit, including for TEASER++.
    """

    scale: object
    rotation: object
    translation: object
    residuals: object
    inliers: np.ndarray
    reason: object
    diagnostics: dict


class RegistrationBackendUnavailable(ImportError):
    """The explicitly requested native registration backend cannot be imported."""


def _scores(value, count):
    """Validate caller-provided correspondence priorities without calibration."""
    raw = np.asarray(value)
    if raw.shape != (count,) or raw.dtype.kind not in "fiu":
        raise ValueError("scores must be a numeric vector with one value per pair")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("scores must be finite")
    return result


def _prosac_samples(scores, iterations, rng):
    """Yield fixed-budget PROSAC triangles in original correspondence indices.

    This implements the growth distribution in Chum and Matas (CVPR 2005),
    equations 3--6: T_n = T_N * C(n, 3) / C(N, 3), T'_3 = 1, and
    T'_(n+1) = T'_n + ceil(T_(n+1) - T_n).  At trial t the active prefix is
    the smallest n with T'_n >= t.  Its newest element is forced into the
    triangle and the other two are uniform distinct points from its prefix.

    Equal scores retain input order.  The budget T_N is the same caller-set
    hypothesis budget used by RANSAC.  There is no claim of the original
    paper's adaptive non-randomness stopping test; that test needs additional
    assumptions about accidental agreement with incorrect models.

    Reference: https://cmp.felk.cvut.cz/~chum/papers/chum-cvpr05b.pdf
    """
    count = len(scores)
    order = np.argsort(-scores, kind="stable")
    prefix = 3
    denominator = math.comb(count, 3)
    growth_end = 1
    for trial in range(1, iterations + 1):
        while trial > growth_end and prefix < count:
            # C(n+1,3)-C(n,3)=C(n,2).  Integer ceiling prevents a rounded
            # exact-integer increment from accidentally growing by one more.
            numerator = int(iterations) * prefix * (prefix - 1) // 2
            growth_end += (numerator + denominator - 1) // denominator
            prefix += 1
        if trial > growth_end:
            # The entire population is available after the growth schedule.
            yield order[rng.choice(count, 3, replace=False)]
        else:
            sample = np.empty(3, dtype=np.int64)
            sample[:2] = rng.choice(prefix - 1, 2, replace=False)
            sample[2] = prefix - 1
            yield order[sample]


def _residuals(src, tgt, transform):
    """Keep the original transform and Euclidean norm arithmetic unchanged."""
    return np.linalg.norm(apply_sim3(src, *transform) - tgt, axis=1)


def _quality(residual, inliers, count):
    """Original lexicographic score: count first, negative median second."""
    return count, -float(np.median(residual[inliers])) if count else -np.inf


def _local_optimize(src, tgt, transform, residual, keep, quality, *,
                    tolerance_m, size_ratio, rank_relative_tolerance):
    """Apply simple all-inlier fitting only to a newly best PROSAC hypothesis.

    This is the Simple LO step of Chum et al. (DAGM 2003), repeated at most
    three times using the same limit as the existing consensus refinement.
    A fit is retained only when its count/median score improves, so LO cannot
    replace this hypothesis by a worse-scoring candidate.  This monotonicity
    is a statement about the stated score, not about unknown physical truth.

    Reference: https://cmp.felk.cvut.cz/~matas/papers/chum-dagm03.pdf
    """
    fits = 0
    for _ in range(3):
        if int(keep.sum()) < 3:
            break
        try:
            candidate = _fit_sim3_validated(src[keep], tgt[keep],
                                            rank_relative_tolerance)
        except (ValueError, np.linalg.LinAlgError):
            break
        fits += 1
        if not 0.2 <= candidate[0] / size_ratio <= 5.0:
            break
        candidate_residual = _residuals(src, tgt, candidate)
        proposed = candidate_residual <= tolerance_m
        candidate_quality = _quality(candidate_residual, proposed, int(proposed.sum()))
        if candidate_quality <= quality:
            break
        same = np.array_equal(proposed, keep)
        transform, residual, keep, quality = candidate, candidate_residual, proposed, candidate_quality
        if same:
            break
    return transform, residual, keep, quality, fits


def _native_record(transform, residual, inliers):
    """Store a JSON-compatible native result before shared final refinement."""
    scale, rotation, translation = transform
    return {"scale": float(scale), "rotation": rotation.tolist(),
            "translation": translation.tolist(), "inliers": int(inliers.sum()),
            "median_inlier_error_m": (float(np.median(residual[inliers]))
                                      if inliers.any() else None)}


def _teaser_solution(src, tgt, tolerance_m):
    """Call the native TEASER++ Sim(3) API, preserving its separate scale.

    ``noise_bound`` uses target-frame metres.  The binding's internal TLS
    inlier indices are not our Euclidean inliers, and translation indices
    refer to its selected maximum clique rather than all original pairs.
    The caller therefore recomputes every residual from the returned Sim(3).
    No optimality certificate is implied by this solver call.

    API: https://github.com/MIT-SPARK/TEASER-plusplus/blob/master/
         python/teaserpp_python/teaserpp_python.cc
    """
    try:
        import teaserpp_python
    except ImportError as exc:
        raise RegistrationBackendUnavailable(
            "teaserpp requires the native teaserpp_python package in the active "
            "environment; no replacement estimator was run") from exc
    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.noise_bound = float(tolerance_m)
    params.cbar2 = 1.0
    params.estimate_scaling = True
    params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS)
    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(np.ascontiguousarray(src.T), np.ascontiguousarray(tgt.T))
    # The supported official Python binding exposes the maximum clique but
    # does not expose RegistrationSolution.valid.  C++ can return early with
    # that flag false when its clique is too small; never interpret its other
    # fields as an estimated transform in that case.  Two paired points also
    # cannot identify a 3-D similarity rotation, so require at least three.
    if len(solver.getInlierMaxClique()) < 3:
        raise ValueError("TEASER++ did not retain enough pairs for a Sim3 solution")
    solution = solver.getSolution()
    scale = float(solution.scale)
    rotation = np.asarray(solution.rotation, dtype=np.float64)
    translation = np.asarray(solution.translation, dtype=np.float64).reshape(-1)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(scale) or scale <= np.finfo(float).tiny
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            or np.linalg.det(rotation) <= 0):
        raise ValueError("TEASER++ returned an invalid proper Sim3 transform")
    return scale, rotation, translation


def register_sim3(src, tgt, *, algorithm="ransac", scores=None, tolerance_m,
                  size_ratio, iterations=1500, seed=0, rank_relative_tolerance=1e-6,
                  min_correspondences=6):
    """Estimate a proper Sim(3) from paired points with a selectable solver.

    ``tolerance_m`` is the common target-frame Euclidean inlier threshold.
    ``size_ratio`` is the ratio of observed target/source object RMS radii;
    sampling rejects scales outside [0.2, 5] times this same observed ratio.
    It is supplied explicitly because the correspondence subset is not the
    entire observed object.  ``scores`` are matching priorities, not measured
    probabilities; only ``prosac_lo`` requires them.

    The default RANSAC keeps exactly the original ``default_rng(seed).choice``
    call sequence, three scalar SVDs per valid triangle, inclusive thresholds,
    earliest exact tie, and final three consensus updates plus one final fit.
    Skipping a median when count is already below the best count cannot change
    the lexicographic decision.  Input validation is performed once here;
    fitting subsets still perform their own geometric rank/scale checks.
    """
    src, tgt = _points(src, "src"), _points(tgt, "tgt")
    if src.shape != tgt.shape:
        raise ValueError("Source and target correspondence arrays must have the same shape")
    if algorithm not in {"ransac", "prosac_lo", "teaserpp"}:
        raise ValueError("Unknown Sim3 registration algorithm")
    for name, value, minimum in (("iterations", iterations, 1), ("seed", seed, 0),
                                  ("min_correspondences", min_correspondences, 3)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("tolerance_m", tolerance_m), ("size_ratio", size_ratio),
                        ("rank_relative_tolerance", rank_relative_tolerance)):
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if scores is not None:
        scores = _scores(scores, len(src))
    elif algorithm == "prosac_lo":
        raise ValueError("prosac_lo requires correspondence scores")
    diagnostics = {"algorithm": algorithm, "tolerance_m": float(tolerance_m),
                   "size_ratio": float(size_ratio), "hypotheses_tested": 0,
                   "local_optimization_fits": 0, "final_refit_fits": 0}

    def failed(reason):
        return RegistrationResult(None, None, None, None, np.zeros(len(src), dtype=bool),
                                  reason, diagnostics)

    if len(src) < min_correspondences:
        return failed("insufficient_correspondences")
    keep = None
    if algorithm == "teaserpp":
        try:
            transform = _teaser_solution(src, tgt, tolerance_m)
        except (ValueError, np.linalg.LinAlgError):
            return failed("degenerate_anchors")
        residual = _residuals(src, tgt, transform)
        keep = residual <= tolerance_m
        diagnostics["native_solution"] = _native_record(transform, residual, keep)
        if not 0.2 <= transform[0] / size_ratio <= 5.0:
            return failed("weak_correspondence_consensus")
    else:
        rng = np.random.default_rng(seed)
        samples = (_prosac_samples(scores, iterations, rng) if algorithm == "prosac_lo"
                   else (rng.choice(len(src), 3, replace=False) for _ in range(iterations)))
        quality = (-1, -np.inf)
        for sample in samples:
            diagnostics["hypotheses_tested"] += 1
            try:
                transform = _fit_sim3_validated(src[sample], tgt[sample],
                                                rank_relative_tolerance)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not 0.2 <= transform[0] / size_ratio <= 5.0:
                continue
            residual = _residuals(src, tgt, transform)
            inlier = residual <= tolerance_m
            count = int(inlier.sum())
            # Count is the primary component: this candidate cannot win,
            # independently of its median.  Equal-count ties still compute it.
            if count < quality[0]:
                continue
            candidate_quality = _quality(residual, inlier, count)
            if candidate_quality > quality:
                if algorithm == "prosac_lo":
                    transform, residual, inlier, candidate_quality, fits = _local_optimize(
                        src, tgt, transform, residual, inlier, candidate_quality,
                        tolerance_m=tolerance_m, size_ratio=size_ratio,
                        rank_relative_tolerance=rank_relative_tolerance)
                    diagnostics["local_optimization_fits"] += fits
                quality, keep = candidate_quality, inlier
                diagnostics["native_solution"] = _native_record(transform, residual, keep)
    if keep is None or keep.sum() < min_correspondences:
        return failed("weak_correspondence_consensus")
    # Preserve the original final refinement exactly, including the extra fit
    # after the bounded loop and recomputation against the returned transform.
    try:
        for _ in range(3):
            transform = _fit_sim3_validated(src[keep], tgt[keep], rank_relative_tolerance)
            diagnostics["final_refit_fits"] += 1
            residual = _residuals(src, tgt, transform)
            proposed = residual <= tolerance_m
            if np.array_equal(proposed, keep) or proposed.sum() < min_correspondences:
                break
            keep = proposed
        transform = _fit_sim3_validated(src[keep], tgt[keep], rank_relative_tolerance)
        diagnostics["final_refit_fits"] += 1
        residual = _residuals(src, tgt, transform)
    except (ValueError, np.linalg.LinAlgError):
        return failed("degenerate_anchors")
    final_keep = residual <= tolerance_m
    if final_keep.sum() < min_correspondences:
        return failed("weak_correspondence_consensus")
    return RegistrationResult(*transform, residual, final_keep, None, diagnostics)
# END selectable correspondence-based Sim(3) registration.
