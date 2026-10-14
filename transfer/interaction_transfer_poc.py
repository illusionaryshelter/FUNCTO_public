#!/usr/bin/env python3
"""Training-free RGB-D interaction region transfer prototype.

Inputs are SAM-filtered 3D points and MARCO features *aligned point-by-point*.
No CAD, mesh, part labels, learning, or explicit shelf/drawer models.

Run:
    python interaction_transfer_poc.py pair.npz --out pred.npz
    python interaction_transfer_poc.py pair.npz --oracle --out oracle_pred.npz

Required NPZ arrays:
    src_xyz          (N, 3): source object's visible point cloud, meters
    tgt_xyz          (M, 3): target object's visible point cloud, meters
    src_region_mask  (N,):   bool, demonstration's interaction surface
    src_feat         (N, C): MARCO image descriptors lifted onto src_xyz
    tgt_feat         (M, C): MARCO image descriptors lifted onto tgt_xyz

Optional for evaluation:
    tgt_gt_mask      (M,): bool, manual target region GT (never used in inference)

Optional for --oracle diagnostic only:
    src_anchor_idx, tgt_anchor_idx: (K,), corresponding surface indices

This intentionally tests the *observable-state* transfer problem, not opening
closed doors/drawers. Source region should be predominantly planar and visible.
"""

import argparse
import json
import sys

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree
from scipy.spatial.transform import Rotation


def rms_size(p):
    return np.sqrt(np.mean(np.sum((p - np.mean(p, axis=0)) ** 2, axis=1)))


def row_norm(f):
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)


def fps_indices(x, count):
    """Farthest-point sampling (no semantic part labels)."""
    count = min(int(count), len(x))
    if count < 1:
        return np.empty((0,), dtype=np.int64)
    start = int(np.argmax(np.sum((x - x.mean(0)) ** 2, axis=1)))
    chosen = np.empty(count, dtype=np.int64)
    d2 = np.full(len(x), np.inf)
    cur = start
    for i in range(count):
        chosen[i] = cur
        d2 = np.minimum(d2, np.sum((x - x[cur]) ** 2, axis=1))
        cur = int(np.argmax(d2))
    return chosen


def fit_sim3(src, tgt):
    """Umeyama least-squares 3D similarity: target ~= s * R @ source + t."""
    if len(src) < 3:
        raise ValueError("Require at least three corresponding 3D anchors")
    xbar, ybar = src.mean(axis=0), tgt.mean(axis=0)
    x, y = src - xbar, tgt - ybar
    if np.linalg.matrix_rank(x, tol=1e-9) < 2:
        raise ValueError("Matching anchors are collinear")
    covariance = (y.T @ x) / len(x)
    U, D, Vt = np.linalg.svd(covariance)
    diag = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        diag[-1] = -1
    R = U @ np.diag(diag) @ Vt
    var_x = np.mean(np.sum(x ** 2, axis=1))
    s = float(np.dot(D, diag) / max(var_x, 1e-12))
    t = ybar - s * (R @ xbar)
    return s, R, t


def apply_sim3(p, s, R, t):
    return s * (np.asarray(p) @ R.T) + t


def marco_mutual_pairs(ps, fs, pt, ft, src_max=450, tgt_max=1400):
    """Find mutual-nearest MARCO features on spatially-distributed points."""
    si = fps_indices(ps, src_max)
    ti = fps_indices(pt, tgt_max)
    sim = row_norm(fs[si]) @ row_norm(ft[ti]).T
    best_t = sim.argmax(axis=1)
    best_s = sim.argmax(axis=0)
    good = np.flatnonzero(best_s[best_t] == np.arange(len(si)))
    return si[good], ti[best_t[good]], sim[good, best_t[good]]


def ransac_sim3(ps, pt, ids, idt, scores, iterations=3500, tol_ratio=0.075, seed=0):
    """Global robust registration is a correspondence-consistency filter.

    Allows modest inter-instance shape differences; does not assume a CAD mesh.
    """
    rng = np.random.default_rng(seed)
    a, b = ps[ids], pt[idt]
    scale_t = rms_size(pt)
    eps = tol_ratio * scale_t
    if len(a) < 6:
        raise RuntimeError(f"Only {len(a)} reciprocal MARCO matches, need >= 6")
    best = None
    best_quality = (-1, -np.inf)
    for _ in range(iterations):
        j = rng.choice(len(a), 3, replace=False)
        if np.linalg.norm(np.cross(a[j[1]] - a[j[0]], a[j[2]] - a[j[0]])) < 0.001 * rms_size(ps) ** 2:
            continue
        try:
            s, R, t = fit_sim3(a[j], b[j])
        except ValueError:
            continue
        # Excludes nonsensical scales; relax for unusually differently sized pairs.
        if not (0.2 * scale_t / rms_size(ps) <= s <= 5 * scale_t / rms_size(ps)):
            continue
        err = np.linalg.norm(apply_sim3(a, s, R, t) - b, axis=1)
        inlier = err < eps
        cnt = int(np.sum(inlier))
        quality = (cnt, -float(np.median(err[inlier])) if cnt else -np.inf)
        if quality > best_quality:
            best_quality, best = quality, inlier
    if best is None or best.sum() < 6:
        raise RuntimeError("No reliable Sim(3) consensus: likely weak correspondences or shape mismatch")
    s, R, t = fit_sim3(a[best], b[best])
    err = np.linalg.norm(apply_sim3(a, s, R, t) - b, axis=1)
    refined = err < eps
    if refined.sum() >= 6:
        best = refined
        s, R, t = fit_sim3(a[best], b[best])
    stats = {
        "reciprocal_matches": int(len(a)),
        "inliers": int(best.sum()),
        "inlier_ratio": float(best.mean()),
        "median_inlier_3d_error_m": float(np.median(err[best])),
    }
    return s, R, t, ids[best], idt[best], scores[best], stats


def plane_basis(p):
    """Orthonormal plane basis; last vector is the least-variance direction."""
    v = np.linalg.svd(p - p.mean(0), full_matrices=False)[2]
    return v[0], v[1], v[2]


def choose_anchors(ps, ids, idt, n=32):
    take = fps_indices(ps[ids], min(n, len(ids)))
    return ids[take], idt[take]


def select_center(ps, pt, region, sa, ta, s, R, t, top_n=250):
    """Match the object's relational distance signature at the region center.

    All distances are dimensionless, yielding Sim(3) invariance for correct
    matched anchors (NOT automatic view-invariance of the 2D MARCO features).
    """
    center_s = np.median(region, axis=0)
    A, B = ps[sa], pt[ta]
    da = np.linalg.norm(A[:, None] - A[None, :], axis=-1)
    db = np.linalg.norm(B[:, None] - B[None, :], axis=-1)
    iu = np.triu_indices(len(A), k=1)
    Ls = np.median(da[iu])
    Lt = np.median(db[iu])
    if Ls < 1e-7 or Lt < 1e-7:
        raise RuntimeError("Anchors are too tightly clustered for stable relation matching")
    ds = np.linalg.norm(center_s - A, axis=1) / Ls
    coarse_center = apply_sim3(center_s, s, R, t)
    # Pairwise anchor distances vs. target candidate distances. Chunk to avoid O(M*K) spikes.
    cost = np.empty(len(pt), dtype=np.float64)
    for beg in range(0, len(pt), 5000):
        q = pt[beg:beg + 5000]
        dt = np.linalg.norm(q[:, None, :] - B[None, :, :], axis=-1) / Lt
        diff = np.abs(dt - ds[None, :])
        # Trim outliers due to occlusion, nonuniform geometry and wrong matches.
        num = max(3, int(np.ceil(0.8 * len(A))))
        score_rel = np.mean(np.partition(diff, num - 1, axis=1)[:, :num], axis=1)
        score_global = np.linalg.norm(q - coarse_center, axis=1) / Lt
        cost[beg:beg + len(q)] = score_rel + 0.12 * score_global
    best = np.argpartition(cost, min(top_n, len(pt) - 1))[:min(top_n, len(pt))]
    tree = cKDTree(pt)
    ns = plane_basis(region)[2]
    wanted = R @ ns
    final = []
    for j in best:
        k = min(35, len(pt))
        near = pt[tree.query(pt[j], k=k)[1]]
        u, v, nt = plane_basis(near)
        nt = nt if np.dot(nt, wanted) >= 0 else -nt
        spread = np.linalg.svd(near - near.mean(0), compute_uv=False)
        # Use only very local planarity as a soft preference.
        flat_penalty = 0.025 * (spread[-1] / max(spread[-2], 1e-8))
        normal_penalty = 0.035 * (1 - np.clip(np.dot(nt, wanted), 0, 1))
        final.append((float(cost[j] + flat_penalty + normal_penalty), int(j), nt))
    final.sort(key=lambda z: z[0])
    best_score, center_idx, normal_t = final[0]
    # Distinct alternative modes, not trivially adjacent candidate points.
    alternate = None
    for score, j, _ in final[1:]:
        if np.linalg.norm(pt[j] - pt[center_idx]) > 0.18 * Lt:
            alternate = (score, j)
            break
    diagnostics = {
        "source_anchor_extent_m": float(Ls),
        "target_anchor_extent_m": float(Lt),
        "center_signature_score": float(best_score),
        "second_spatial_mode_score": None if alternate is None else float(alternate[0]),
        "second_mode_center_m": None if alternate is None else pt[alternate[1]].tolist(),
        "center_to_coarse_m": float(np.linalg.norm(pt[center_idx] - coarse_center)),
        "region_normal_agreement": float(np.dot(normal_t, wanted)),
    }
    return center_s, pt[center_idx], normal_t, diagnostics


def transfer_planar_patch(src_region, src_center, tgt_center, tgt_normal, pt, s, R, Lt):
    """Move a demo surface patch, rotate to the observed target local plane, then
    return target points inside its convex footprint. Does not create a mesh.
    """
    e1, e2, ns = plane_basis(src_region)
    predicted_normal = R @ ns
    if np.dot(predicted_normal, tgt_normal) < 0:
        tgt_normal = -tgt_normal
    align, _ = Rotation.align_vectors([tgt_normal], [predicted_normal])
    Rlocal = align.as_matrix() @ R
    pred = tgt_center + s * ((src_region - src_center) @ Rlocal.T)
    # Project demonstration's footprint onto the target's local tangent plane.
    pred = pred - np.outer((pred - tgt_center) @ tgt_normal, tgt_normal)
    axis_u = Rlocal @ e1
    axis_u = axis_u - tgt_normal * np.dot(tgt_normal, axis_u)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(tgt_normal, axis_u)
    xy_region = np.stack([(pred - tgt_center) @ axis_u, (pred - tgt_center) @ axis_v], axis=1)
    xy_target = np.stack([(pt - tgt_center) @ axis_u, (pt - tgt_center) @ axis_v], axis=1)
    try:
        hull = ConvexHull(xy_region)
        poly = xy_region[hull.vertices]
        inside = Delaunay(poly).find_simplex(xy_target) >= 0
    except QhullError as e:
        raise RuntimeError("Region footprint is degenerate") from e
    plane_residual = np.abs((pt - tgt_center) @ tgt_normal)
    on_plane = plane_residual < 0.03 * Lt
    mask = inside & on_plane
    if mask.any():
        d, _ = cKDTree(pt[mask]).query(pred, k=1)
        observed_coverage = float(np.mean(d < 0.06 * Lt))
    else:
        observed_coverage = 0.0
    result = {
        "region_footprint_observed_fraction": observed_coverage,
        "target_patch_points": int(mask.sum()),
        "target_plane_tolerance_m": float(0.03 * Lt),
        "predicted_region_extent_m": float(np.max(np.linalg.norm(pred - tgt_center, axis=1))),
        "source_region_planarity_ratio": float(
            np.linalg.svd(src_region - src_region.mean(0), compute_uv=False)[-1] /
            max(np.linalg.svd(src_region - src_region.mean(0), compute_uv=False)[-2], 1e-8)
        ),
    }
    return mask, pred, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="NPZ containing source/target xyz, features, source region mask")
    parser.add_argument("--out", default="region_transfer_pred.npz")
    parser.add_argument("--oracle", action="store_true", help="Use manually matched anchors ONLY as an upper-bound diagnostic")
    parser.add_argument("--iters", type=int, default=3500)
    parser.add_argument("--tol-ratio", type=float, default=0.075,
                        help="RANSAC 3D threshold / target RMS radius")
    args = parser.parse_args()
    x = np.load(args.input)
    ps = np.asarray(x["src_xyz"], dtype=np.float64)
    pt = np.asarray(x["tgt_xyz"], dtype=np.float64)
    if min(len(ps), len(pt)) < 25:
        raise RuntimeError("Too few SAM points")
    mask_s = np.asarray(x["src_region_mask"], bool)
    if mask_s.shape != (len(ps),) or np.sum(mask_s) < 12:
        raise RuntimeError("src_region_mask must contain >= 12 source surface points")
    fs = np.asarray(x["src_feat"], np.float32)
    ft = np.asarray(x["tgt_feat"], np.float32)
    if fs.shape[0] != len(ps) or ft.shape[0] != len(pt) or fs.shape[1] != ft.shape[1]:
        raise RuntimeError("MARCO descriptors must align pointwise with xyz")
    if not np.isfinite(ps).all() or not np.isfinite(pt).all() or not np.isfinite(fs).all() or not np.isfinite(ft).all():
        raise RuntimeError("Remove invalid xyz, depth, and feature values first")
    region = ps[mask_s]
    if args.oracle:
        ids = np.asarray(x["src_anchor_idx"], np.int64)
        idt = np.asarray(x["tgt_anchor_idx"], np.int64)
        if len(ids) != len(idt) or len(ids) < 3:
            raise RuntimeError("Provide >=3 matching oracle anchor indices")
        s, R, t = fit_sim3(ps[ids], pt[idt])
        stats = {"mode": "oracle", "matched_anchors": len(ids)}
    else:
        ids_all, idt_all, scores = marco_mutual_pairs(ps, fs, pt, ft)
        s, R, t, ids, idt, scores, stats = ransac_sim3(
            ps, pt, ids_all, idt_all, scores, iterations=args.iters, tol_ratio=args.tol_ratio
        )
        stats["mode"] = "marco+geometry"
    sa, ta = choose_anchors(ps, ids, idt, n=32)
    if len(sa) < 3:
        raise RuntimeError("Not enough geometrically consistent anchors")
    center_s, center_t, normal_t, geom_stats = select_center(ps, pt, region, sa, ta, s, R, t)
    mask_t, pred_src_region, patch_stats = transfer_planar_patch(
        region, center_s, center_t, normal_t, pt, s, R, geom_stats["target_anchor_extent_m"]
    )
    stats.update(geom_stats)
    stats.update(patch_stats)
    stats["sim3_scale"] = float(s)
    stats["source_region_points"] = int(mask_s.sum())
    # These are early reject checks, NOT calibrated probabilities.
    reasons = []
    if stats["mode"] != "oracle" and (stats["inliers"] < 10 or stats["inlier_ratio"] < 0.2):
        reasons.append("weak_correspondence_consensus")
    if stats["target_patch_points"] < 12 or stats["region_footprint_observed_fraction"] < 0.4:
        reasons.append("little_visible_target_support")
    second = stats["second_spatial_mode_score"]
    if second is not None and second - stats["center_signature_score"] < 0.01:
        reasons.append("ambiguous_spatial_modes")
    if stats["region_normal_agreement"] < 0.65:
        reasons.append("contact_plane_orientation_mismatch")
    if stats["source_region_planarity_ratio"] > 0.15:
        reasons.append("source_region_not_approximately_planar")
    stats["poc_status"] = "candidate" if not reasons else "uncertain"
    stats["warnings"] = reasons
    if "tgt_gt_mask" in x:
        gt = np.asarray(x["tgt_gt_mask"], bool)
        if gt.shape == mask_t.shape:
            inter = int((mask_t & gt).sum())
            union = int((mask_t | gt).sum())
            stats["target_mask_iou"] = float(inter / union) if union else 0.0
            if mask_t.any() and gt.any():
                stats["target_center_error_m"] = float(
                    np.linalg.norm(pt[mask_t].mean(0) - pt[gt].mean(0))
                )
                stats["target_center_error_normalized"] = (
                    stats["target_center_error_m"] / rms_size(pt)
                )
    np.savez_compressed(args.out, tgt_region_mask=mask_t, tgt_region_points=pt[mask_t],
                        tgt_center=center_t, estimated_region_surface=pred_src_region,
                        src_anchor_idx=sa, tgt_anchor_idx=ta, sim3_R=R, sim3_t=t,
                        sim3_scale=s, stats_json=json.dumps(stats, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Saved:", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"FAILED / NO PREDICTION: {ex}", file=sys.stderr)
        sys.exit(2)
