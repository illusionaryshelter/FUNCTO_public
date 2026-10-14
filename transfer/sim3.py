"""Shared, double-precision geometry for proper 3-D similarity transforms.

The convention throughout the transfer experiment is
``target = scale * source @ rotation.T + translation``.  These helpers know
nothing about object categories, feature networks, interaction labels, or
ground truth.  Public fitting validates its arrays; registration can reuse the
private fitting kernel after validating the complete correspondence arrays.
"""

import numpy as np


# BEGIN shared Sim(3) primitives and validation-preserving fitting optimization.
def _points(value, name):
    """Validate a finite numeric ``[N, 3]`` array and expose float64 values."""
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric [N,3] array")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite coordinates")
    return result


def rms_size(points):
    """Rotation-equivariant RMS radius, independent of coordinate origin."""
    return float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))


def _rank_centered(centered, tolerance):
    """Apply the existing singular-value rank rule to already centered data.

    Sharing the centered array with Umeyama fitting avoids computing the same
    means and differences twice.  The SVD, threshold comparison, and diagnostic
    condition ratio retain their original arithmetic and ordering.
    """
    if len(centered) < 2:
        return 0, 0.0
    singular = np.linalg.svd(centered, compute_uv=False)
    if not len(singular) or singular[0] <= np.finfo(float).tiny:
        return 0, 0.0
    dimension = int(np.count_nonzero(singular > tolerance * singular[0]))
    condition = float(singular[-1] / singular[0]) if len(singular) == 3 else 0.0
    return dimension, condition


def rank(points, tolerance=1e-6):
    """Return centered rank and the original smallest/largest singular ratio.

    Like the geometry routine formerly local to ``core.py``, this helper takes
    an already numeric array.  Callers at external boundaries use ``_points``.
    """
    if len(points) < 2:
        return 0, 0.0
    return _rank_centered(points - points.mean(0), tolerance)


def _fit_sim3_validated(src, tgt, rank_relative_tolerance=1e-6):
    """Fit validated paired float64 arrays with at least three rows.

    Registration subsets inherit shape, dtype, and finiteness from its public
    entry point.  Geometry still needs a rank check for *every* subset: a valid
    full cloud can contain coincident or collinear sampled triangles.  Reusing
    the centered coordinates changes neither the covariance expression nor
    the individual scalar SVD calls used by the original implementation.
    """
    source_mean, target_mean = src.mean(0), tgt.mean(0)
    x, y = src - source_mean, tgt - target_mean
    if min(_rank_centered(x, rank_relative_tolerance)[0],
           _rank_centered(y, rank_relative_tolerance)[0]) < 2:
        raise ValueError("Sim3 anchors are coincident or collinear")
    covariance = y.T @ x / len(x)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    correction[-1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = (u * correction) @ vt
    variance = float(np.mean(np.sum(x * x, axis=1)))
    scale = float(singular @ correction / variance)
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        raise ValueError("Sim3 must have a finite positive scale")
    translation = target_mean - scale * (rotation @ source_mean)
    if not np.isfinite(translation).all():
        raise ValueError("Sim3 translation is nonfinite")
    return scale, rotation, translation


def fit_sim3(src, tgt, rank_relative_tolerance=1e-6):
    """Validated Umeyama fit with positive scale and a proper rotation.

    A non-collinear triangle has centered rank two and identifies a proper
    Sim(3).  Requiring rank three would wrongly reject usable planar matches.
    """
    src, tgt = _points(src, "src"), _points(tgt, "tgt")
    if src.shape != tgt.shape or len(src) < 3:
        raise ValueError("Sim3 requires at least three matching 3-D points")
    return _fit_sim3_validated(src, tgt, rank_relative_tolerance)


def apply_sim3(points, scale, rotation, translation):
    """Apply the experiment's row-vector similarity convention unchanged."""
    return scale * (np.asarray(points) @ rotation.T) + translation
# END shared Sim(3) primitives and validation-preserving fitting optimization.
