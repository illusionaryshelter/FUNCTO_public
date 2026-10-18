"""Demonstration-conditioned grasp planning on top of GraspGen-X.

Only the backend adapter requires GraspGen-X/PyTorch; the policy can be tested
with NumPy/SciPy and a mocked backend. This module does NOT perform robot IK,
contact simulation or trajectory collision checking on its own. It fails closed:
'EXECUTION_READY' requires every supplied execution validator to succeed.

Coordinate convention: all points and T_world_grasp are in one gravity-aligned
world frame, meters, GraspGen-X gripper +Z approach, +X finger closing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class CandidateBackend(Protocol):
    @property
    def sweep_volume(self) -> np.ndarray: ...  # (6,) [extents_xyz, offset_xyz], meters
    @property
    def symmetric(self) -> bool: ...
    def score(self, poses_WG: np.ndarray, model_pc_W: np.ndarray) -> np.ndarray: ...
    def generate(self, model_pc_W: np.ndarray, *, dense: bool = False): ...
    # returns poses (K,4,4), quality (K,), source labels (K,)


@dataclass
class GraspRequest:
    model_pc_W: np.ndarray              # model input: whole object or grasp-part+context
    object_pc_W: np.ndarray             # entire target; no other objects
    allowed_pc_W: np.ndarray            # graspable patch only (not functional action tip)
    scene_without_target_pc_W: np.ndarray  # table, neighbors, cabinet, etc.
    T_WG_prior: Optional[np.ndarray] = None  # gripper BASE, not functional frame / TCP
    approach_W: Optional[np.ndarray] = None   # desired +Z direction in world
    preferred_approach_deg: float = 30.0  # soft preference
    hard_max_approach_deg: float = 110.0 # hard ceiling. Set = preferred for strict.
    max_prior_position_sigma_m: float = 0.035
    max_prior_rotation_sigma_deg: float = 35.0
    prior_confidence: float = 0.8      # in [0,1], downweight if human-to-robot uncertain
    skill: str = "pick"                # pick / pull / door / tool-use
    task_data: dict = field(default_factory=dict)


@dataclass
class PlannerOptions:
    prior_scales: tuple = ((32, 0.008, 8.), (64, 0.025, 30.), (96, 0.065, 75.))
    min_object_points_in_pocket: int = 3
    min_allowed_points_in_pocket: int = 3
    min_allowed_fraction: float = 0.45
    region_point_tol_m: float = 0.006
    min_quality: float = 0.25   # needs validation on actual data; not a calibrated probability
    pregrasp_retreat_m: float = 0.10
    max_expensive_checks: int = 200
    fast_path_enabled: bool = True
    fast_path_min_quality: float = 0.80
    # q + demo likelihood + preferred approach penalty; not a physical success probability
    weight_quality: float = 0.65
    weight_demo: float = 0.30
    weight_approach: float = 0.05
    seed: int = 2026


@dataclass
class GraspPlan:
    status: str      # EXECUTION_READY / CANDIDATE_ONLY / NO_FEASIBLE_GRASP
    T_WG: Optional[np.ndarray] = None
    T_WG_pre: Optional[np.ndarray] = None
    quality: float = 0.
    utility: float = float("-inf")
    source: str = ""
    diagnostics: dict = field(default_factory=dict)


def _xyz(pc, name, min_len=0):
    a = np.asarray(pc, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < min_len or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite (N,3) with N>={min_len}")
    return np.ascontiguousarray(a)


def _pose(T, name="pose"):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4,4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1], atol=1e-4):
        raise ValueError(f"{name}: not a finite 4x4 homogeneous transform")
    R = T[:3,:3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R),1.,atol=1e-3):
        raise ValueError(f"{name}: rotation must belong to SO(3), no scale or reflection")
    return T


def _normalize(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-8:
        raise ValueError("direction must be a finite nonzero vector")
    return v / n


def angle_deg(vectors, reference):
    # vectors Kx3, reference 3
    dots = np.clip(np.asarray(vectors) @ _normalize(reference), -1., 1.)
    return np.rad2deg(np.arccos(dots))


def geodesic_rotation_deg(R_batch, R_prior, symmetric=False):
    """Rotation differences incl. optional two-finger 180 deg closing-axis symmetry."""
    Rp = np.asarray(R_prior)
    if not symmetric:
        rel = np.einsum('ji,kjl->kil', Rp, R_batch)
        cs = (np.trace(rel,axis1=1,axis2=2)-1)/2
        return np.degrees(np.arccos(np.clip(cs,-1,1)))
    sym = np.diag([-1.,-1.,1.])  # local 180 degrees about approach +Z
    return np.minimum(
        geodesic_rotation_deg(R_batch, Rp, symmetric=False),
        geodesic_rotation_deg(R_batch, Rp @ sym, symmetric=False),
    )


def sample_prior(T_WG_prior, scales, seed=2026):
    """Right-local perturbations in gripper coordinates; always include prior."""
    T = _pose(T_WG_prior, "T_WG_prior")
    rng = np.random.default_rng(seed)
    all_T = [T.astype(np.float32)]
    R0 = T[:3,:3]
    for count, sigma_m, sigma_deg in scales:
        delta_p = rng.normal(0, sigma_m, (int(count),3))
        delta_r = rng.normal(0, np.deg2rad(sigma_deg), (int(count),3))
        for dp,dr in zip(delta_p,delta_r):
            Ti = np.eye(4)
            Ti[:3,:3] = R0 @ Rotation.from_rotvec(dr).as_matrix()
            Ti[:3,3] = T[:3,3] + R0 @ dp
            all_T.append(Ti.astype(np.float32))
    return np.stack(all_T)


def _pregrasp(Ts, retreat):
    P = np.array(Ts, copy=True)
    P[:,:3,3] -= float(retreat) * Ts[:,:3,2]  # retreat opposite GraspGen-X +Z
    return P


class DemoGraspPlanner:
    def __init__(
        self,
        backend: CandidateBackend,
        *,
        opts: PlannerOptions = None,
        # Environment gripper collision: (K,4,4), scene_pc -> (K,) bool.
        scene_collision_free: Optional[Callable] = None,
        # Real finger-pad contact & non-pad/target collision: T, request -> bool.
        target_contact_valid: Optional[Callable] = None,
        # Robot reachability, pregrasp, approach, postgrasp skill: T,T_pre,request -> bool.
        robot_skill_feasible: Optional[Callable] = None,
    ):
        self.b = backend
        self.opts = opts or PlannerOptions()
        self.scene_collision_free = scene_collision_free
        self.target_contact_valid = target_contact_valid
        self.robot_skill_feasible = robot_skill_feasible
        self.sweep = np.asarray(backend.sweep_volume, dtype=np.float64)
        if self.sweep.shape != (6,) or (self.sweep[:3] <= 0).any():
            raise ValueError("gripper sweep_volume must be [positive extents_xyz, offset_xyz]")

    def _geometry_mask(self, Ts, req):
        """Conservative ROI proxy. True contact needs target_contact_valid()."""
        valid = np.zeros(len(Ts),dtype=bool)
        evidence = np.zeros(len(Ts),dtype=float)
        allowed_tree = cKDTree(req.allowed_pc_W)
        in_roi = allowed_tree.query(req.object_pc_W, k=1)[0] <= self.opts.region_point_tol_m
        half = self.sweep[:3] * 0.5
        center = self.sweep[3:]
        for k,T in enumerate(Ts):
            local = (req.object_pc_W - T[:3,3]) @ T[:3,:3]
            pocket = np.all(np.abs(local-center) <= half + 1e-5,axis=1)
            n = int(np.sum(pocket))
            n_allowed = int(np.sum(pocket & in_roi))
            ratio = n_allowed / max(n,1)
            valid[k] = (
                n >= self.opts.min_object_points_in_pocket
                and n_allowed >= self.opts.min_allowed_points_in_pocket
                and ratio >= self.opts.min_allowed_fraction
            )
            evidence[k] = ratio
        return valid, evidence

    def _candidate_info(self, Ts, qs, labels, req):
        valid_pose = []
        for T in Ts:
            try: _pose(T)
            except ValueError: valid_pose.append(False)
            else: valid_pose.append(True)
        valid = np.array(valid_pose,dtype=bool) & np.isfinite(qs)
        geometry, roi_conf = self._geometry_mask(Ts,req)
        valid &= geometry
        if req.approach_W is not None:
            angle = angle_deg(Ts[:,:3,2], req.approach_W)
            valid &= angle <= req.hard_max_approach_deg + 1e-6
        else:
            angle = np.zeros(len(Ts),dtype=float)
        valid &= qs >= self.opts.min_quality
        if req.T_WG_prior is not None:
            dp = np.linalg.norm(Ts[:,:3,3] - req.T_WG_prior[:3,3], axis=1)
            da = geodesic_rotation_deg(Ts[:,:3,:3], req.T_WG_prior[:3,:3], self.b.symmetric)
            demo = np.exp(-0.5*((dp/max(req.max_prior_position_sigma_m,1e-5))**2
                           +(da/max(req.max_prior_rotation_sigma_deg,1e-5))**2))
        else:
            demo = np.zeros(len(Ts),dtype=float)
        # A preference, not a hard ban. Can move far from demonstration when necessary.
        app = np.exp(-0.5 * (angle / max(req.preferred_approach_deg,1e-5))**2)
        utility = (self.opts.weight_quality * qs
                   + self.opts.weight_demo * req.prior_confidence * demo
                   + self.opts.weight_approach * app)
        # roi_conf used as a hard support criterion, not double-counted as 'probability'.
        return valid, utility, angle, roi_conf

    def _check(self, Ts, qs, labels, req, trace):
        if len(Ts)==0:
            return None
        valid, utility, ang, roi = self._candidate_info(Ts,qs,labels,req)
        trace['generated'] += len(Ts)
        trace['passed_roi_approach_quality'] += int(valid.sum())
        order = np.argsort(-utility)
        order = [int(i) for i in order if valid[i]][:self.opts.max_expensive_checks]
        if not order:
            return None
        arr = Ts[order]
        if self.scene_collision_free is not None:
            cf = np.asarray(self.scene_collision_free(arr,req.scene_without_target_pc_W),dtype=bool)
            if cf.shape != (len(arr),):
                raise ValueError('scene_collision_free must return (K,) bool array')
            trace['scene_collision_rejects'] += int((~cf).sum())
            order = [i for i,passed in zip(order,cf) if passed]
        if not order:
            return None
        pre = _pregrasp(Ts[order],self.opts.pregrasp_retreat_m)
        ready = (self.scene_collision_free is not None
                 and self.target_contact_valid is not None
                 and self.robot_skill_feasible is not None)
        for j,i in enumerate(order):
            T = Ts[i]
            if self.target_contact_valid is not None and not bool(self.target_contact_valid(T,req)):
                trace['target_contact_rejects'] += 1
                continue
            if self.robot_skill_feasible is not None and not bool(self.robot_skill_feasible(T,pre[j],req)):
                trace['robot_or_skill_rejects'] += 1
                continue
            trace['accepted_candidates'] += 1
            return GraspPlan(
                status='EXECUTION_READY' if ready else 'CANDIDATE_ONLY',
                T_WG=T.copy(), T_WG_pre=pre[j].copy(), quality=float(qs[i]),
                utility=float(utility[i]), source=str(labels[i]),
                diagnostics={**trace, 'approach_error_deg':float(ang[i]),
                             'roi_proxy_fraction':float(roi[i]),
                             'validators_complete':ready},
            )
        return None

    def plan(self, request: GraspRequest):
        req = request
        req.model_pc_W = _xyz(req.model_pc_W, 'model_pc_W', min_len=100)
        req.object_pc_W = _xyz(req.object_pc_W, 'object_pc_W', min_len=100)
        req.allowed_pc_W = _xyz(req.allowed_pc_W, 'allowed_pc_W', min_len=3)
        req.scene_without_target_pc_W = _xyz(req.scene_without_target_pc_W,'scene_without_target_pc_W')
        if req.T_WG_prior is not None:
            req.T_WG_prior = _pose(req.T_WG_prior, 'T_WG_prior')
        if req.approach_W is None and req.T_WG_prior is not None:
            req.approach_W = req.T_WG_prior[:3,2].copy()
        if req.approach_W is not None:
            req.approach_W = _normalize(req.approach_W)
        if not (0 <= req.preferred_approach_deg <= req.hard_max_approach_deg <= 180):
            raise ValueError('require 0<=preferred_approach_deg<=hard_max_approach_deg<=180')
        if not (0 <= req.prior_confidence <= 1):
            raise ValueError('prior_confidence must be in [0,1]')
        # ROI MUST correspond to object points in same world frame.
        support = cKDTree(req.object_pc_W).query(req.allowed_pc_W)[0]
        if np.median(support) > self.opts.region_point_tol_m*2:
            raise ValueError('allowed_pc_W is not aligned with object_pc_W; check units/transforms')
        trace = dict(generated=0,passed_roi_approach_quality=0,
                     scene_collision_rejects=0,target_contact_rejects=0,
                     robot_or_skill_rejects=0,accepted_candidates=0,phase='prior')
        if req.T_WG_prior is not None:
            Ts = sample_prior(req.T_WG_prior,self.opts.prior_scales,self.opts.seed)
            # Avoid scoring candidates that cannot physically intersect allowed region.
            roi, _ = self._geometry_mask(Ts,req)
            if req.approach_W is not None:
                roi &= angle_deg(Ts[:,:3,2],req.approach_W) <= req.hard_max_approach_deg
            local = Ts[roi]
            if len(local):
                q = np.asarray(self.b.score(local,req.model_pc_W),dtype=np.float32)
                if q.shape != (len(local),):
                    raise ValueError('backend score must return (K,)')
                labels=['demo']*len(local)
                if self.opts.fast_path_enabled and self.target_contact_valid and self.robot_skill_feasible and self.scene_collision_free:
                    eligible = q >= self.opts.fast_path_min_quality
                    if np.any(eligible):
                        fast = self._check(local[eligible],q[eligible],
                                           np.array(labels)[eligible],req,trace)
                        if fast is not None:
                            fast.diagnostics['phase']='fast-prior'
                            return fast
            else:
                q = np.empty((0,),dtype=np.float32)
        else:
            local=np.empty((0,4,4),dtype=np.float32);q=np.empty((0,),dtype=np.float32)
        trace['phase']='union'
        gg, gq, gl = self.b.generate(req.model_pc_W,dense=False)
        gg=np.asarray(gg,dtype=np.float32).reshape((-1,4,4))
        gq=np.asarray(gq,dtype=np.float32).reshape(-1)
        if len(gg)!=len(gq) or len(gl)!=len(gg):
            raise ValueError('backend.generate returned inconsistent lengths')
        Ts = np.concatenate([local,gg])
        qs = np.concatenate([q,gq])
        labels = ['demo']*len(local)+list(gl)
        result = self._check(Ts,qs,labels,req,trace)
        if result is not None:
            return result
        # Optional higher coverage fallback: denser OBB and extra global diffusion.
        trace['phase']='dense-fallback'
        gg2,gq2,gl2=self.b.generate(req.model_pc_W,dense=True)
        gg2=np.asarray(gg2,dtype=np.float32).reshape((-1,4,4))
        gq2=np.asarray(gq2,dtype=np.float32).reshape(-1)
        if len(gg2)!=len(gq2) or len(gl2)!=len(gg2):
            raise ValueError('backend.generate(dense=True) inconsistent lengths')
        result = self._check(gg2,gq2,gl2,req,trace)
        if result is not None:
            return result
        return GraspPlan(status='NO_FEASIBLE_GRASP',diagnostics=trace)


class GraspGenXBackend:
    """Official GraspGen-X adapter, requires the installed repo & pretrained weights.

    Uses the official private GraspMoE discriminator scoring helper. Pin your
    GraspGen-X git commit, test on each upstream upgrade. This adapter is not a
    'ready' controller; arm/contact callbacks must be integrated separately.
    """
    def __init__(self, checkpoints_dir, gripper_name, assets_dir=None, score_chunk=64, num_grasps=200):
        from pathlib import Path
        import torch
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.utils.checkpoint_io import load_model_cfg
        self._torch=torch
        root=Path(checkpoints_dir)
        cfg=load_model_cfg(str(root/'gen'),str(root/'dis'))
        self.sampler=GraspGenXSampler(cfg,gripper_name,assets_dir=assets_dir,use_tensorrt=False)
        self.score_chunk=int(score_chunk)
        self.num_grasps = int(num_grasps)
        if self.num_grasps < 1:
            raise ValueError("num_grasps must be positive")
        if self.score_chunk < 1:
            raise ValueError('score_chunk must be positive')
        self.sweep_volume=self.sampler.gripper.sweep_volume
        self.symmetric=bool(self.sampler.gripper.symmetric)

    def generate(self, model_pc_W, *, dense=False):
        from graspgenx.samplers import run_planner_on_object
        # Bounded top+side OBB fallback: avoid huge one-shot discriminator batches
        # on very elongated objects (e.g. a drawer panel or cabinet door).
        extent_max_m = float(np.ptp(model_pc_W, axis=0).max())
        spacing_cm = max(3.0, 100.0 * extent_max_m / 8.0)
        g,q,tags,_=run_planner_on_object(
            model_pc_W,self.sampler,planner='graspmoe',
            grasp_threshold=-1.0, num_grasps=self.num_grasps if not dense else int(self.num_grasps * 1.5),
            topk_num_grasps=-1, moe_num_yaws=12 if not dense else 8,
            moe_z_offsets_cm=(-2,0),
            moe_obb_density='sparse' if not dense else 'dense-topandside',
            moe_obb_position_spacing_cm=spacing_cm,
        )
        return np.asarray(g,dtype=np.float32),np.asarray(q,dtype=np.float32),tags

    def score(self, poses_WG, model_pc_W):
        from graspgenx.samplers.graspmoe import _score_grasps_with_discriminator
        from graspgenx.utils.point_cloud import point_cloud_outlier_removal
        if len(poses_WG)==0:
            return np.empty((0,),dtype=np.float32)
        torch=self._torch
        cloud_t=torch.from_numpy(np.asarray(model_pc_W,dtype=np.float32))
        filtered,_=point_cloud_outlier_removal(cloud_t,threshold=0.014)
        if len(filtered)<10:
            filtered=cloud_t
        filtered_np=filtered.cpu().numpy()
        center=filtered_np.mean(axis=0)
        device=next(self.sampler.model.parameters()).device
        centered=torch.from_numpy(filtered_np-center).to(device)
        out=[]
        with torch.inference_mode():
            for start in range(0,len(poses_WG),self.score_chunk):
                out.append(_score_grasps_with_discriminator(
                    np.asarray(poses_WG[start:start+self.score_chunk],dtype=np.float32),
                    centered,center,self.sampler))
        return np.concatenate(out)

    def make_scene_collision_callback(self, collision_threshold=0.008,
                                     num_collision_samples=768,batch_size=2):
        """Approx. *environment* collision at final grasp pose only.
        Use target_contact_valid and robot_skill_feasible for execution readiness.
        """
        from graspgenx.utils.collision_filter import filter_colliding_grasps
        import trimesh
        mesh=self.sampler.gripper.collision_mesh
        if (getattr(mesh,'vertices',None) is None or len(mesh.vertices)<8
                or np.max(np.asarray(mesh.extents)) < 0.03):
            raise RuntimeError('Gripper collision mesh missing or dummy; verify actual gripper assets')
        samples,_=trimesh.sample.sample_surface(mesh,int(num_collision_samples))
        pts=np.asarray(samples,dtype=np.float32)
        def check(poses,scene_pc):
            return filter_colliding_grasps(
                scene_pc=scene_pc,grasp_poses=poses,
                collision_threshold=collision_threshold,
                gripper_surface_points=pts, batch_size=batch_size)
        return check
