"""CPU-only invariants of the shared fixed online geometry model."""
# BEGIN semantic tests: fixed field and unknown-space semantics remain shared with execution.
from types import SimpleNamespace

import numpy as np
import torch

from manipulation.online_geometry import FrozenESDF, FrozenPrimitives, OnlinePlanningScene
from manipulation.planning_scene import ObservedSpace


def field(distance=None, valid=True, owner='background'):
    values=np.full((21,21,21),1.,np.float32) if distance is None else distance
    mask=np.full(values.shape,valid,dtype=bool)
    return FrozenESDF(values,origin=[-.1,-.1,-.1],voxel_size=.01,
        site_valid=mask,observed_sign=mask,owner_ids=(owner,),source='test measured snapshot')


def scene(*,conditional=False,background=None):
    known=np.zeros((21,21,21),dtype=bool)
    cover=ObservedSpace(known,origin=[-.1,-.1,-.1],voxel_size=.01,
        observation_ids=('A',),reference_frame='world')
    policy={'outside_regions':'forbid','conditional_regions':(
        [{'lower_m':[-.08]*3,'upper_m':[.08]*3,'source':'declared finite operation volume'}] if conditional else [])}
    result=OnlinePlanningScene(background=field(valid=False) if background is None else background,
        owners=(),coverage=cover,reference_frame='world',robot_sphere_materials=('body',),
        robot_geometry_source='calibrated robot',unknown_policy=policy,has_gripper=False)
    result.bind_online_robot(robot_sphere_geometry=[{'link_name':'link','sphere_indices':[0],
        'centers_local':[[0.,0.,0.]],'radii':[.01]}],robot_self_collision_pairs=[])
    return result


def test_bounded_unknown_is_conditional_not_free_and_outside_map_is_update():
    sphere=np.array([[0.,0.,0.,.01]])
    denied=scene().query_state(sphere,np.eye(4),0.)
    assert not denied['passed']
    allowed=scene(conditional=True).query_state(sphere,np.eye(4),0.)
    assert allowed['passed'] and allowed['conditional_unknown']
    assert not scene(conditional=True).coverage.known_free.any()
    outside=scene(conditional=True).query_state(np.array([[.11,0.,0.,.01]]),np.eye(4),0.)
    assert not outside['passed'] and outside['status']=='SCENE_UPDATE_REQUIRED'


def test_absent_surface_cannot_create_collision_and_owner_outside_grid_is_disjoint():
    spheres=torch.tensor([[[[0.,0.,0.,.01],[1.,0.,0.,.01]]]],dtype=torch.float64)
    assert torch.equal(field(np.zeros((21,21,21),np.float32),valid=False)(spheres),torch.zeros((1,1,2),dtype=torch.float64))
    query=field(np.zeros((21,21,21),np.float32),valid=True,owner='part')
    values=query(spheres)
    assert values[0,0,0]>0 and values[0,0,1]==0


def test_execution_binding_retains_same_field_decision():
    original=scene(conditional=True)
    bound=OnlinePlanningScene.from_binding(original.to_execution_binding())
    spheres=np.array([[0.,0.,0.,.01]])
    assert original.query_state(spheres,np.eye(4),.5)==bound.query_state(spheres,np.eye(4),.5)


def test_known_collision_survives_conditional_unknown_permission():
    axis=np.linspace(-.1,.1,21)
    distances=np.broadcast_to(axis[:,None,None],(21,21,21)).astype(np.float32)
    result=scene(conditional=True,background=field(distances)).query_state(
        np.array([[-.01,0.,0.,.01]]),np.eye(4),0.)
    assert not result['passed']
    assert any(row['name']=='collision/robot/background' for row in result['violations'])



def test_primitive_pose_and_dimensions_keep_exact_native_meaning():
    query=FrozenPrimitives(primitives=[{'kind':'cuboid','dimensions':[.02,.04,.06],
        'pose':[.05,0.,0.,1.,0.,0.,0.]}],owner_ids=('mount',),source='calibrated primitive')
    points=torch.tensor([[.05,0.,0.],[.07,0.,0.]],dtype=torch.float64)
    assert torch.allclose(query.signed_distance(points),torch.tensor([-.01,.01],dtype=torch.float64))
    archived=FrozenESDF.from_record(query.record())
    assert torch.equal(query.signed_distance(points),archived.signed_distance(points))
# END tests use no Mapper, CUDA, simulator, external models or installations.


# BEGIN exact zero-set and request-isolation regressions.
def test_small_positive_collision_is_not_erased_by_an_execution_tolerance():
    axis=np.linspace(-.1,.1,21)
    distances=np.broadcast_to(axis[:,None,None],(21,21,21)).astype(np.float32)
    model=scene(conditional=True,background=field(distances))
    sphere=np.array([[.01-1e-8,0.,0.,.01]])
    report=model.query_state(sphere,np.eye(4),0.)
    residual=report['components']['collision/robot/background']
    assert 0<residual<model.numerical_position_tolerance_m
    assert not report['passed']


def test_cloned_queries_do_not_replace_another_requests_prepared_model():
    query=field()
    original=object()
    query._contact_model=original
    other=query.clone_for_rollout()
    other._contact_model=object()
    assert query._contact_model is original
    assert other._contact_model is not original
# END no positive geometric residual becomes a different runtime success set.
