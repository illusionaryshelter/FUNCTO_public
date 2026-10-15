#!/usr/bin/env python3
"""Reproducible *synthetic correspondence* stress test for multi-view interaction grounding.

NOT a MARCO validation: this generates MARCO-like true/false correspondences with
specified viewpoint-dependent accuracy, and tests the inference geometry. It uses
synthetic point-sampled shelf and open drawer scenes, pinhole z-buffer observations,
separate left/right match sets, and no source left camera.

Run: OPENBLAS_NUM_THREADS=1 python multiview_grounding_experiment.py --trials 30
Outputs CSV and a JSON summary, plus a plot if matplotlib is available.
"""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Camera:
    eye: np.ndarray
    at: np.ndarray
    width: int = 176
    height: int = 132
    focal: float = 135.

    def frame(self):
        fw = self.at - self.eye
        fw /= np.linalg.norm(fw)
        r = np.cross(fw, np.array([0., 0., 1.]))
        if np.linalg.norm(r) < 1e-7:
            r = np.cross(fw, np.array([0., 1., 0.]))
        r /= np.linalg.norm(r)
        up = np.cross(r, fw)
        return r, up, fw

    def project(self, xyz):
        r, up, fw = self.frame()
        v = xyz - self.eye
        dep = v @ fw
        valid = dep > .08
        uu = self.focal * (v @ r) / np.maximum(dep, .08) + self.width / 2
        vv = -self.focal * (v @ up) / np.maximum(dep, .08) + self.height / 2
        u = np.rint(uu).astype(int)
        v = np.rint(vv).astype(int)
        valid &= (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return u, v, dep, valid

    def zbuffer(self, xyz):
        u, v, z, valid = self.project(xyz)
        ind = np.flatnonzero(valid)
        # Point cloud renderer, nearest sample for each projected camera pixel.
        pix = v[ind] * self.width + u[ind]
        order = np.argsort(z[ind], kind='stable')
        _, first = np.unique(pix[order], return_index=True)
        kept = ind[order[first]]
        depth = np.full((self.height, self.width), np.nan)
        depth[v[kept], u[kept]] = z[kept]
        return kept, depth

    def classify(self, xyz, depth, margin=.045):
        """Positive: near measured surface; negative: in measured free space;
        neutral: behind occluder, no depth, or outside FOV."""
        u, v, z, inframe = self.project(xyz)
        state = np.zeros(len(xyz), dtype=np.int8)
        ids = np.flatnonzero(inframe)
        observed = depth[v[ids], u[ids]]
        ok = np.isfinite(observed)
        ids = ids[ok]
        observed = observed[ok]
        difference = z[ids] - observed
        state[ids[np.abs(difference) <= margin]] = 1
        state[ids[difference < -margin]] = -1
        return state


def make_scene(name='shelf'):
    """Each sampled surface point has an internal ID ONLY for synthetic matching generation."""
    parts = []
    if name == 'shelf':
        xs = np.linspace(-.58, .58, 33)
        ys = np.linspace(-.4, .4, 25)
        zz = np.linspace(.12, 1.05, 33)
        for z in [.20, .57, .94]:
            x, y = np.meshgrid(xs, ys)
            parts.append(np.c_[x.ravel(), y.ravel(), np.full(x.size, z)])
        for x in [-.58, .58]:
            y, z = np.meshgrid(ys, zz)
            parts.append(np.c_[np.full(y.size, x), y.ravel(), z.ravel()])
        x, z = np.meshgrid(xs, zz)
        parts.append(np.c_[x.ravel(), np.full(x.size, .4), z.ravel()])
        ref = np.array([-.15, -.13, .57])
    else:
        xs = np.linspace(-.58, .58, 37)
        ys = np.linspace(-.40, .40, 27)
        zz = np.linspace(.12, .35, 14)
        x, y = np.meshgrid(xs, ys)
        parts.append(np.c_[x.ravel(), y.ravel(), np.full(x.size, .12)])
        for x in [-.58, .58]:
            y, z = np.meshgrid(ys, zz)
            parts.append(np.c_[np.full(y.size, x), y.ravel(), z.ravel()])
        for y in [-.4, .4]:
            x, z = np.meshgrid(xs, zz)
            parts.append(np.c_[x.ravel(), np.full(x.size, y), z.ravel()])
        ref = np.array([-.15, -.12, .12])
    points = np.concatenate(parts)
    return points, ref


def fps(x, n):
    n = min(n, len(x))
    out = np.empty(n, dtype=np.int64)
    d = np.full(len(x), np.inf)
    cur = np.argmax(np.linalg.norm(x - x.mean(0), axis=1))
    for i in range(n):
        out[i] = cur
        d = np.minimum(d, np.sum((x - x[cur])**2, axis=1))
        cur = np.argmax(d)
    return out


def voxel_downsample(points, grid=.025):
    if not len(points):
        return points
    cell = np.floor(points / grid).astype(np.int32)
    uniq, inverse = np.unique(cell, return_inverse=True, axis=0)
    sums = np.zeros((len(uniq), 3), dtype=float)
    count = np.bincount(inverse)
    np.add.at(sums, inverse, points)
    return sums / count[:, None]


def sim_transform(p, anisotropic=False, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    widths = np.array([1.38, 1.05, 1.0]) if anisotropic else np.ones(3) * 1.25
    yaw = np.deg2rad(rng.uniform(-20, 20))
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.]])
    trans = np.array([.07, -.07, .0]) + rng.normal(0, [.025, .025, 0])
    return (p * widths) @ rot.T + trans, widths, rot, trans


def cam_setup(shelf=True):
    z = .60 if shelf else .19
    src = Camera(np.array([1.65, -1.9, 1.60 if shelf else 1.18]), np.array([0, 0, z]))
    right = Camera(np.array([1.73, -1.75, 1.58 if shelf else 1.19]), np.array([0, 0, z]))
    left = Camera(np.array([-1.75, -1.75, 1.58 if shelf else 1.19]), np.array([0, 0, z]))
    return src, left, right


def make_matches(ids_source, seen_tgt_ids, src_pts, tgt_pts, ref, rng, prob_true,
                 bad_layer_bias=False, calibration_noise=0.004):
    """Generate pseudo-MARCO sparse correspondences. Ground truth is only
    used by the DATA GENERATOR to control error rates, NOT by the solver."""
    seen = set(map(int, seen_tgt_ids))
    mt_ids = []
    mt_pts = []
    confidence = []
    seen_tgt_ids = np.asarray(seen_tgt_ids)
    # Restrict source anchors to those with a meaningful baseline to interaction center.
    for src_id in ids_source:
        if rng.random() < .11:  # MARCO may abstain or have low-quality depth
            continue
        is_visible = int(src_id) in seen
        correct = is_visible and rng.random() < prob_true
        if correct:
            dest = int(src_id)
            conf = rng.uniform(.72, .99)
        else:
            if len(seen_tgt_ids) < 5:
                continue
            if bad_layer_bias and rng.random() < .72:
                # Systematic semantic confusion: favor similar x/y on another shelf.
                sp = src_pts[src_id]
                xy = np.linalg.norm(src_pts[seen_tgt_ids, :2] - sp[:2], axis=1)
                dz = np.abs(src_pts[seen_tgt_ids, 2] - sp[2])
                usable = (dz > .20) & (xy < .22)
                pool = seen_tgt_ids[usable]
                dest = int(rng.choice(pool)) if len(pool) else int(rng.choice(seen_tgt_ids))
            else:
                dest = int(rng.choice(seen_tgt_ids))
            conf = rng.uniform(.58, .94)  # Wrong matches can be visually confident.
        mt_ids.append(int(src_id))
        mt_pts.append(tgt_pts[dest] + rng.normal(0, calibration_noise, size=3))
        confidence.append(conf)
    return (np.asarray(mt_ids, dtype=np.int32),
            np.asarray(mt_pts, dtype=float).reshape(-1, 3),
            np.asarray(confidence, dtype=float))


def source_anchor_coverage(src_pts, ids, anchor_weight):
    if len(ids) < 4:
        return 0.
    # Source anchor geometry must constrain >=2 in-plane axes for a valid placement center.
    p = src_pts[ids]
    w = np.asarray(anchor_weight) / max(sum(anchor_weight), 1e-10)
    c = np.sum(w[:, None] * p, axis=0)
    cov = (p - c).T @ ((p - c) * w[:, None])
    s = np.linalg.eigvalsh(cov)
    return float(np.sqrt(max(s[1], 0.) / max(s[2], 1e-10)))


def geometric_ground(src_pts, src_center, samples, matches, mode='joint', cameras=None,
                     depth_maps=None, max_candidates=1800):
    """Joint 3D grounding; no camera descriptor concatenation/averaging.

    samples: candidate target 3D support points from selected camera(s).
    matches: mapping 'L'/'R' -> (src_ids, target_xyz, confidence), each source
             anchor can occur once in each view.
    Estimate the region reference q and uniform scale *jointly* via a robust
    per-source-keypoint multi-view scale-consistency vote.
    """
    if len(samples) == 0:
        return None
    if len(samples) > max_candidates:
        samples = samples[fps(samples, max_candidates)]
    src_extent = np.linalg.norm(np.ptp(src_pts, axis=0))
    pool_ids = sorted({int(a) for ids, xyz, w in matches.values() for a in ids
                       if np.linalg.norm(src_center-src_pts[a]) > max(.07, .06*src_extent)})
    if len(pool_ids) < 6:
        return None
    id2col = {srcid:i for i, srcid in enumerate(pool_ids)}
    n = len(pool_ids)
    views = list(matches.keys())
    weights = np.zeros((n, len(views)))
    dests = np.zeros((n, len(views), 3))
    src_dist = np.linalg.norm(src_pts[np.array(pool_ids)] - src_center, axis=1)
    for vidx, name in enumerate(views):
        ids, pts, confs = matches[name]
        for i, b, conf in zip(ids, pts, confs):
            idx = id2col.get(int(i))
            if idx is not None:
                weights[idx, vidx] = conf
                dests[idx, vidx] = b
    # Fusing views must never count repeated observations as independent anchors.
    present = weights > 0
    scale_grid = np.linspace(np.log(.65), np.log(1.75), 45)
    sigma_log = .095  # supports moderate nonuniform change, e.g. 1.38x width
    best_scores = np.full(len(samples), -np.inf)
    best_scales = np.zeros(len(samples))
    corrob = np.zeros(n)
    if len(views) > 1:
        both = present[:, 0] & present[:, 1]
        delta = np.linalg.norm(dests[:, 0] - dests[:, 1], axis=1)
        # Only independent views locating the SAME 3D landmark are corroboration.
        corrob = both * (delta < .035)
    batch = 160
    for lo in range(0, len(samples), batch):
        q = samples[lo:lo+batch]
        err = np.linalg.norm(q[:, None, None, :] - dests[None, :, :, :], axis=-1)
        logratios = np.log(np.maximum(err, 1e-6) / src_dist[None, :, None])
        # Broadcast across all logscale proposals at once; avoid Python loops over scales.
        z = (logratios[..., None] - scale_grid[None, None, None, :]) / sigma_log
        votes = np.exp(-.5*z*z) * weights[None, :, :, None]
        if mode == 'pool':
            support = votes.sum(axis=(1,2))
        else:
            support = votes.max(axis=2).sum(axis=1)
            if len(views)>1:
                support += (.16 * corrob[None,:,None] * votes.min(axis=2)).sum(axis=1)
        at = support.argmax(axis=1)
        best_scores[lo:lo+len(q)] = support[np.arange(len(q)),at]
        best_scales[lo:lo+len(q)] = np.exp(scale_grid[at])
    # Reject candidate surface points contradicted by *measured* free space.
    if cameras and depth_maps:
        for v in cameras:
            evidence = cameras[v].classify(samples, depth_maps[v])
            best_scores += np.where(evidence < 0, -3.0, 0.0)
    peak = int(np.argmax(best_scores))
    pick = samples[peak]
    second = -np.inf
    for i in np.argsort(-best_scores):
        if np.linalg.norm(samples[i]-pick) > .13:
            second = best_scores[i]
            break
    # Confidence diagnostics, NOT calibrated probabilities.
    source_weights = np.max(weights, axis=1)
    geometry_coverage = source_anchor_coverage(src_pts, np.asarray(pool_ids), source_weights)
    diagnostics = dict(score=float(best_scores[peak]), second_score=float(second),
                       margin=float(best_scores[peak]-second),
                       scale=float(best_scales[peak]),
                       unique_source_anchors=int(n),
                       matched_both=int((present.sum(axis=1)==2).sum()),
                       cross_view_agree=int(corrob.sum()),
                       source_anchor_spread=float(geometry_coverage))
    return pick, diagnostics



def robust_affine_ground(src_pts, samples, matches, rngseed=167, trials=260, tol=.030):
    """Joint outlier-robust 3D affine alignment, no category/CAD/learning.

    Affine is a *localization hypothesis*, not a contact footprint transform.
    Every source feature anchor counts at most once in consensus. Only use if
    noncoplanar geometrically spread anchors permit stable interpolation.
    """
    if len(samples)<8:
        return None
    rec=[]
    for _,(ids,points,confs) in matches.items():
        rec.extend((int(i),np.asarray(q),float(w)) for i,q,w in zip(ids,points,confs))
    if len(rec)<9:
        return None
    rid=np.array([u[0] for u in rec],dtype=int)
    Y=np.array([u[1] for u in rec])
    W=np.array([u[2] for u in rec])
    X=src_pts[rid]
    H=np.c_[X,np.ones(len(X))]
    names, inv=np.unique(rid,return_inverse=True)
    if len(names)<8:
        return None
    rng=np.random.default_rng(rngseed)
    extent=np.linalg.norm(np.ptp(src_pts,axis=0))
    best_score=(-1,-np.inf)
    best_mask=None
    def valid_affine(B):
        A=B[:3].T
        sv=np.linalg.svd(A,compute_uv=False)
        return (np.linalg.det(A)>0 and np.min(sv)>.6 and np.max(sv)<1.85
                and np.max(sv)/np.min(sv)<2.8)
    def consensus(B):
        err=np.linalg.norm(H@B-Y,axis=1)
        good=(err<tol)
        # Unique source IDs only; if both views agree, no duplicate votes.
        count=np.bincount(inv,weights=(good*W),minlength=len(names))
        supported=count>0
        n=int(np.sum(supported))
        med=float(np.median(err[good])) if np.any(good) else np.inf
        return (n,-med),good
    for _ in range(trials):
        ix=rng.choice(len(H),4,replace=False)
        if len(np.unique(rid[ix]))!=4 or abs(np.linalg.det(H[ix]))<2e-4*extent**3:
            continue
        try:
            B=np.linalg.solve(H[ix],Y[ix])
        except np.linalg.LinAlgError:
            continue
        if not valid_affine(B):
            continue
        sc,good=consensus(B)
        if sc>best_score:
            best_score=sc
            best_mask=good
    if best_mask is None or best_score[0]<8:
        return None
    # Refine on consensus, weighting both views jointly without letting duplicates
    # contribute more than one unit per source landmark.
    corr=np.bincount(inv,weights=best_mask.astype(float),minlength=len(names))
    ww=W*best_mask/np.maximum(corr[inv],1)
    try:
        B=np.linalg.lstsq(H*np.sqrt(ww)[:,None],Y*np.sqrt(ww)[:,None],rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    if not valid_affine(B):
        return None
    sc,good=consensus(B)
    inliers=np.unique(rid[good])
    if len(inliers)<8:
        return None
    cov=np.cov(src_pts[inliers].T)
    evals=np.linalg.eigvalsh(cov)
    source_spread=float(evals[0]/max(evals[-1],1e-10))
    # Fragile full 3D affine with nearly planar anchors is not trusted.
    if source_spread<.008:
        return None
    return B,dict(inliers=int(len(inliers)),median_m=-float(sc[1]),
                  source_3d_spread=source_spread)


def affine_region_ground(src_pts, src_center, samples, matches):
    fit=robust_affine_ground(src_pts,samples,matches)
    if fit is None:
        return None
    B,info=fit
    qhat=np.r_[src_center,1]@B
    idx=np.argmin(np.sum((samples-qhat)**2,axis=1))
    q=samples[idx]
    offset=np.linalg.norm(q-qhat)
    if offset>.085:
        return None
    info.update(source_predicted_to_visible_m=float(offset))
    return q,info


def simulate_one(scene, setting, seed):
    rng = np.random.default_rng(seed)
    src, ref = make_scene(scene)
    tgt, axes, R, t = sim_transform(src, anisotropic=setting in ('anisotropic', 'anisotropic_occluded'),rng=rng)
    camera_s, cam_l, cam_r = cam_setup(scene=='shelf')
    # Source viewpoint ALWAYS RIGHT: source geometry supplied to any inference
    # stage is exclusively the visible right-camera point cloud.
    obs_s, dep_s = camera_s.zbuffer(src)
    src_visible = src[obs_s]
    selected = (np.linalg.norm(src_visible[:,:2]-ref[:2],axis=1)<.16) & (np.abs(src_visible[:,2]-ref[2])<.018)
    if np.sum(selected)<8:
        raise RuntimeError('Source interaction region is not observable from demonstration camera')
    src_center = src_visible[selected].mean(axis=0)
    gt = (src_center * axes) @ R.T + t
    obs_l, dep_l = cam_l.zbuffer(tgt)
    obs_r, dep_r = cam_r.zbuffer(tgt)
    # Right-side foreground occluder: eliminate those SAM object pixels,
    # and write nearer (non-object) depths into the actual RGB-D depth map.
    # This matters: behind a foreground occluder is UNKNOWN, not free space.
    if setting in ('right_occluded', 'anisotropic_occluded'):
        x0,x1,y0,y1=54,127,38,108
        uu,vv,zz,inside=cam_r.project(tgt[obs_r])
        blocked=(uu>=x0)&(uu<x1)&(vv>=y0)&(vv<y1)
        obs_r=obs_r[~blocked]
        patch=dep_r[y0:y1,x0:x1]
        nominal=float(np.nanmedian(dep_r))-.32
        dep_r[y0:y1,x0:x1]=np.where(np.isfinite(patch),np.minimum(patch-.30,nominal),nominal)
    if setting == 'left_unreliable':
        obs_l = obs_l[rng.random(len(obs_l)) < .72]
    if setting == 'complementary':
        obs_r = obs_r[rng.random(len(obs_r)) < .65]
        obs_l = obs_l[rng.random(len(obs_l)) < .70]
    cam_noise = .0035
    pts_r = tgt[obs_r]+rng.normal(0,cam_noise,size=(len(obs_r),3))
    pts_l = tgt[obs_l]+rng.normal(0,cam_noise,size=(len(obs_l),3))
    # global IDs below are accessible ONLY to the simulator generating
    # correspondence mistakes. They never enter any grounding routine.
    sel = fps(src_visible, min(76, len(obs_s)))
    src_anchor_ids = obs_s[sel]
    # Compare source right features against target view-specific MARCO-like output.
    p_right = {'normal':.78,'right_occluded':.50,'anisotropic':.77,
               'anisotropic_occluded':.50,'left_unreliable':.80,
               'complementary':.58,'structured_wrong':.43}.get(setting,.72)
    p_left = {'normal':.43,'right_occluded':.58,'anisotropic':.39,
              'anisotropic_occluded':.55,'left_unreliable':.06,
              'complementary':.48,'structured_wrong':.46}.get(setting,.42)
    wrong = setting in ('structured_wrong','complementary')
    m_r = make_matches(src_anchor_ids, obs_r, src, tgt, ref, rng,p_right,
                       bad_layer_bias=wrong,calibration_noise=.004)
    m_l = make_matches(src_anchor_ids, obs_l, src, tgt, ref, rng,p_left,
                       bad_layer_bias=wrong,calibration_noise=.004)
    # Remap synthetic correspondence IDs to actual observed source point-cloud indices.
    global_to_local = np.full(len(src),-1,dtype=int)
    global_to_local[obs_s] = np.arange(len(obs_s))
    m_r = (global_to_local[m_r[0]],m_r[1],m_r[2])
    m_l = (global_to_local[m_l[0]],m_l[1],m_l[2])
    candidates_r = voxel_downsample(pts_r)
    candidates_l = voxel_downsample(pts_l)
    candidates_f = voxel_downsample(np.vstack([pts_l,pts_r]))
    output = {}
    specs = [
        ('right', candidates_r, {'R':m_r},'joint'),
        ('left', candidates_l, {'L':m_l},'joint'),
        ('naive_pool', candidates_f, {'L':m_l,'R':m_r}, 'pool'),
        ('multiview', candidates_f, {'L':m_l,'R':m_r},'joint')
    ]
    for method, xyz, mt, mode in specs:
        start = perf_counter()
        result = geometric_ground(src_visible, src_center, xyz, mt, mode=mode,
                                  cameras={'L':cam_l,'R':cam_r} if method=='multiview' else None,
                                  depth_maps={'L':dep_l,'R':dep_r} if method=='multiview' else None)
        duration = (perf_counter() - start)*1000
        if result is None:
            output[method] = dict(error_m=np.nan, ok=False, time_ms=duration, score=-1e9, margin=-1e9)
            continue
        q, diag = result
        error = np.linalg.norm(q-gt)
        output[method] = dict(error_m=float(error), ok=bool(error<.075),time_ms=duration,
                              score=diag['score'],margin=diag['margin'],
                              scale=diag['scale'],source_anchor_spread=diag['source_anchor_spread'],
                              cross_view_agree=diag['cross_view_agree'])
    # Oracle upper bound based on visibility: closest actually-observed point to GT.
    oracle_err = np.linalg.norm(candidates_f-gt,axis=1).min() if len(candidates_f) else np.nan
    # Best-of-views by a non-oracle score: per-view peak evidence.
    # Caution: raw scores comparable here only because source anchor IDs, feature scoring,
    # and scale search grid are identical. In deployment calibrate probabilities.
    best = max(['left','right'],key=lambda v:output[v]['score'])
    output['choose_view'] = dict(output[best],chosen_view=best)
    for method,cloud,mt in [('affine_right',candidates_r,{'R':m_r}),
                            ('affine_multiview',candidates_f,{'L':m_l,'R':m_r})]:
        start=perf_counter()
        estimate=affine_region_ground(src_visible,src_center,cloud,mt)
        duration=(perf_counter()-start)*1000
        if estimate is None:
            output[method]=dict(error_m=np.nan,ok=False,time_ms=duration,score=-1e9,margin=-1e9)
        else:
            q,meta=estimate
            err=np.linalg.norm(q-gt)
            output[method]=dict(error_m=float(err),ok=bool(err<.075),time_ms=duration,
                                 score=float(meta['inliers']),margin=0,
                                 source_anchor_spread=meta['source_3d_spread'])
    return output, float(oracle_err), (len(obs_s),len(obs_l),len(obs_r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--scenario-indices',default='all',help='all or comma-delimited 0..7')
    ap.add_argument('--output',default='/mnt/data/multiview_grounding_results')
    opt=ap.parse_args()
    base = Path(opt.output)
    base.parent.mkdir(parents=True,exist_ok=True)
    # Each is a deterministic family of randomly perturbed poses and matches.
    scenarios=[('shelf','normal'), ('shelf','right_occluded'),('shelf','structured_wrong'),
               ('shelf','anisotropic'),('drawer','normal'),('drawer','right_occluded'),
               ('drawer','anisotropic_occluded'),('drawer','left_unreliable')]
    records=[]
    selected=(set(range(len(scenarios))) if opt.scenario_indices=='all'
              else set(int(q) for q in opt.scenario_indices.split(',')))
    for si,(scene,setting) in enumerate(scenarios):
        if si not in selected:
            continue
        for k in range(opt.trials):
            trial,oracle_err,observations=simulate_one(scene,setting,seed=3019+si*1007+k)
            for method,vals in trial.items():
                records.append(dict(scene=scene,setting=setting,case=f'{scene}/{setting}',
                                    trial=k,method=method,oracle_visible_error_m=oracle_err,
                                    obs_source=observations[0],obs_left=observations[1],
                                    obs_right=observations[2],**vals))
        print('Completed',scene,setting, flush=True)
    fields=sorted({k for r in records for k in r})
    with open(base.with_suffix('.csv'),'w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary={}
    methods=['right','left','choose_view','naive_pool','multiview','affine_right','affine_multiview']
    for s in sorted(set(r['case'] for r in records)):
        summary[s]={}
        for m in methods:
            rows=[r for r in records if r['case']==s and r['method']==m]
            errors=np.array([r['error_m'] for r in rows])
            durations=np.array([r['time_ms'] for r in rows])
            summary[s][m]=dict(success_rate=float(np.mean(errors<.075)),
                               median_error_mm=float(np.nanmedian(errors)*1000) if np.isfinite(errors).any() else None,
                               p90_error_mm=float(np.nanquantile(errors,.9)*1000) if np.isfinite(errors).any() else None,
                               solver_median_ms=float(np.median(durations)))
    with open(base.with_suffix('.json'),'w') as fp:
        json.dump(dict(note='Synthetic 3D geometry and simulated MARCO-like correspondences; no actual MARCO model was run.',
                       seed=3019,trials=opt.trials,summary=summary),fp,indent=2)
    print('\nSUCCESS_RATE vs SINGLE RIGHT')
    print('scenario                         '+' '.join(f'{m[:12]:>12s}' for m in methods))
    for s in sorted(summary):
        vals=[summary[s][m]['success_rate'] for m in methods]
        print(f"{s:31s}"+' '.join(f'{x*100:7.1f}%' for x in vals))
    print('\nAggregated:')
    for m in methods:
        rows=[r for r in records if r['method']==m]
        print(f"{m:12s}: success {np.mean([r['ok'] for r in rows])*100:5.1f}%"
              f", med_error {np.median([r['error_m'] for r in rows])*1000:6.1f} mm"
              f", grounding_core_med {np.median([r['time_ms'] for r in rows]):6.1f} ms")
    print('Saved',str(base.with_suffix('.csv')),str(base.with_suffix('.json')))
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,5.2))
        x=np.arange(len(summary))
        w=.155
        for j,m in enumerate(methods):
            ax.bar(x+(j-2)*w,[summary[s][m]['success_rate']*100 for s in sorted(summary)],w,label=m)
        ax.set_ylim(0,105)
        ax.set_ylabel('Success rate, center error < 75 mm (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('/','\n') for s in sorted(summary)],rotation=0,fontsize=8)
        ax.legend(ncol=5,loc='upper center')
        ax.grid(axis='y',alpha=.15)
        fig.tight_layout()
        path=base.with_suffix('.png')
        fig.savefig(path,dpi=160)
        plt.close(fig)
        print('Saved',path)
    except Exception as ex:
        print('Skipping plot',ex)

if __name__=='__main__':
    main()
