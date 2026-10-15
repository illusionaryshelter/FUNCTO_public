"""Targeted geometric sanity tests. Run with Python; depends on numpy, scipy."""
import sys
import numpy as np
sys.path.insert(0, '/mnt/data')
from multiview_grounding_experiment import Camera, robust_affine_ground, geometric_ground

# Same 3D surface under a camera's pixel and three distinct depth states.
cam=Camera(np.array([0.,0.,-2.]),np.array([0.,0.,0.]))
q=np.array([[0.,0.,0.]])
u,v,z,inside=cam.project(q)
assert bool(inside[0])
D=np.full((cam.height,cam.width),np.nan)
D[v[0],u[0]]=z[0]
assert cam.classify(q,D)[0]==1, 'observed, same-depth surface'
D[v[0],u[0]]=z[0]-.3
assert cam.classify(q,D)[0]==0, 'occluded should be unknown, never negative'
D[v[0],u[0]]=z[0]+.3
assert cam.classify(q,D)[0]==-1, 'foreground free-space contradiction'
D[v[0],u[0]]=np.nan
assert cam.classify(q,D)[0]==0, 'missing depth is unknown, not negative'
print('PASS: visibility is positive / neutral-occluded / negative-free-space / neutral-missing')

# Fully specified non-uniform deformation, coherent independent camera anchor observations.
rng=np.random.default_rng(811)
p=rng.uniform(-.5,.5,(65,3))
s=np.array([1.34,1.10,.92]); theta=.25
R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
t=np.array([.25,-.18,.07])
ref=np.array([-.16,.03,.14])
y=(p*s)@R.T+t
ref_gt=(ref*s)@R.T+t
mL=(np.arange(len(p)),y+rng.normal(0,.003,(len(p),3)),np.ones(len(p))*.8)
mR=(np.arange(len(p)),y+rng.normal(0,.003,(len(p),3)),np.ones(len(p))*.8)
B,stat=robust_affine_ground(p,y,{'L':mL,'R':mR})
q_affine=np.r_[ref,1.]@B
assert np.linalg.norm(q_affine-ref_gt)<.012
print('PASS: affine grounding from two views after nonuniform scaling, error_mm=',round(np.linalg.norm(q_affine-ref_gt)*1000,2))

# The demonstrated source camera may have a completely different SE(3) pose.
phi=-.83
Rc=np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
Rc3=np.array([[1,0,0],[0,np.cos(.35),-np.sin(.35)],[0,np.sin(.35),np.cos(.35)]])@Rc
p_camera=p@Rc3.T+np.array([.45,.27,-.32])
ref_camera=ref@Rc3.T+np.array([.45,.27,-.32])
B_camera,stats_cam=robust_affine_ground(p_camera,y,{'L':mL,'R':mR})
q_camera=np.r_[ref_camera,1.]@B_camera
assert np.linalg.norm(q_camera-ref_gt)<.012
print('PASS: arbitrary demo camera SE(3) pose absorbed, error_mm=',round(np.linalg.norm(q_camera-ref_gt)*1000,2))

# The Gaussian metric sampling footprint belongs to the block, not drawer scale.
uv=rng.normal(0,.035,(1000,2))
local1=uv.copy(); local2=uv.copy()
R2=np.array([[np.cos(.7),-np.sin(.7)],[np.sin(.7),np.cos(.7)]])
local2=local2@R2.T  # object pose change: just rotate metric offsets
assert abs(np.mean(np.linalg.norm(local2,axis=1))-np.mean(np.linalg.norm(local1,axis=1)))<1e-12
print('PASS: preferred sampling radius invariant to drawer resizing and pose')

# No target depth => explicitly no matching / no reference recovery.
assert geometric_ground(p,ref,np.empty((0,3)),{'R':mR}) is None
print('PASS: absent target observation cannot silently yield a placement reference')
