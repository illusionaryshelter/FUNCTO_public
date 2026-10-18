import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from demo_guided_grasp import (GraspRequest, PlannerOptions, DemoGraspPlanner,
                               sample_prior, geodesic_rotation_deg)

rng=np.random.default_rng(7)
PC=(rng.uniform(-0.014,0.014,(1300,3))+np.array([0,0,0.04])).astype(np.float32)

def tf(angle=0, axis='z', p=(0,0,0)):
    T=np.eye(4,dtype=np.float32)
    T[:3,:3]=Rotation.from_euler(axis,angle,degrees=True).as_matrix()
    T[:3,3]=p
    return T

class Backend:
    sweep_volume=np.array([0.08,0.05,0.08,0,0,0.04])
    symmetric=True
    def __init__(self,quality=.1): self.quality=quality
    def score(self,Ts,pc): return np.full(len(Ts),self.quality,dtype=np.float32)
    def generate(self,pc,dense=False):
        g=np.stack([tf(0),tf(90),tf(180)])
        return g,np.array([.85,.93,.88],dtype=np.float32),['diff','obb','obb']

def req(prior=True):
    return GraspRequest(model_pc_W=PC.copy(),object_pc_W=PC.copy(),allowed_pc_W=PC.copy(),
                        scene_without_target_pc_W=np.zeros((0,3),dtype=np.float32),
                        T_WG_prior=tf(0) if prior else None,
                        approach_W=np.array([0.,0.,1.]))

def test_prior_fast_path():
    planner=DemoGraspPlanner(Backend(quality=.92),scene_collision_free=lambda Ts,pc:np.ones(len(Ts),bool),
                             target_contact_valid=lambda T,r:True,
                             robot_skill_feasible=lambda T,P,r:True)
    plan=planner.plan(req())
    assert plan.status=='EXECUTION_READY'
    assert plan.source=='demo'
    assert plan.diagnostics['phase']=='fast-prior'
    assert np.allclose(plan.T_WG_pre[:3,3], plan.T_WG[:3,3]-0.10*plan.T_WG[:3,2])

def test_change_way_when_prior_blocked():
    # No local cand accepted by quality gate; original pose collides, turn 90deg yaw instead.
    def scene_ok(Ts,pc): return np.abs(Ts[:,0,0]) < 0.4
    planner=DemoGraspPlanner(Backend(quality=.1),scene_collision_free=scene_ok,
                             target_contact_valid=lambda T,r:True,
                             robot_skill_feasible=lambda T,P,r:True,
                             opts=PlannerOptions(fast_path_enabled=False))
    plan=planner.plan(req())
    assert plan.status=='EXECUTION_READY'
    assert plan.source=='obb'
    assert abs(plan.T_WG[0,0]) < 0.4

def test_no_false_execution_readiness_without_robot_checks():
    p=DemoGraspPlanner(Backend(quality=.1))
    plan=p.plan(req(prior=False))
    assert plan.status=='CANDIDATE_ONLY'
    assert plan.T_WG is not None

def test_reject_region_outside_object():
    r=req()
    r.allowed_pc_W=r.allowed_pc_W+10
    with pytest.raises(ValueError,match='not aligned'):
        DemoGraspPlanner(Backend()).plan(r)

def test_hard_approach_ceiling():
    class TiltBackend(Backend):
        def generate(self,pc,dense=False):
            return np.stack([tf(90,'x')]),np.array([.9]),['diff']
    r=req(prior=False)
    r.hard_max_approach_deg=30
    p=DemoGraspPlanner(TiltBackend())
    result=p.plan(r)
    assert result.status=='NO_FEASIBLE_GRASP'

def test_rotation_symmetry():
    a=np.stack([np.eye(3),Rotation.from_euler('z',180,degrees=True).as_matrix()])
    d=geodesic_rotation_deg(a,np.eye(3),True)
    assert np.max(d) < 1e-4
    d2=geodesic_rotation_deg(a,np.eye(3),False)
    assert d2[1]>179

def test_prior_sampler_produces_SO3():
    s=sample_prior(tf(10),[(10,0.01,20)],seed=7)
    assert s.shape==(11,4,4)
    for T in s:
        assert abs(np.linalg.det(T[:3,:3])-1)<1e-5
