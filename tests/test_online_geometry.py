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


def test_bounded_unknown_is_conditional_and_empty_local_map_does_not_limit_arm_domain():
    sphere=np.array([[0.,0.,0.,.01]])
    denied=scene().query_state(sphere,np.eye(4),0.)
    assert not denied['passed']
    allowed=scene(conditional=True).query_state(sphere,np.eye(4),0.)
    assert allowed['passed'] and allowed['conditional_unknown']
    assert not scene(conditional=True).coverage.known_free.any()
    outside=scene(conditional=True).query_state(np.array([[.11,0.,0.,.01]]),np.eye(4),0.)
    assert not outside['passed'] and outside['status']=='COLLISION_OR_CONSTRAINT'
    assert outside['components']['scene_update/background/robot'] == 0
    assert outside['components']['unknown/forbidden/robot'] > 0


def test_outside_centre_with_radius_intersecting_known_local_geometry_still_requires_update():
    current = scene(conditional=True, background=field(np.zeros((21,21,21),np.float32)))
    overlap = current.query_state(np.array([[.11,0.,0.,.02]]), np.eye(4), .5)
    assert overlap['status'] == 'SCENE_UPDATE_REQUIRED'
    disjoint = current.query_state(np.array([[.14,0.,0.,.02]]), np.eye(4), .5)
    assert disjoint['components']['scene_update/background/robot'] == 0
    assert disjoint['components']['unknown/forbidden/robot'] > 0


# BEGIN finite-field boundary regression: missing samples are not zero distances.
def test_partial_zero_padding_cannot_manufacture_a_background_collision():
    # The two remote observed sites make the coarse geometry bounds cover the
    # query, while every interpolation vertex near it has +0.2 m clearance.
    # One voxel beyond the last centre is missing data, not an obstacle plane.
    distances = np.full((21, 21, 21), .2, np.float32)
    distances[0, 0, 0] = distances[-1, -1, -1] = 0.
    query = field(distances)
    spheres = torch.tensor([[.1, 0., 0., .03], [.109, 0., 0., .03],
                            [.12, 0., 0., .03]], dtype=torch.float64, requires_grad=True)
    assert query.intersects_geometry(spheres).all()
    assert query.supports_centres(spheres).tolist() == [True, False, False]
    residual = query(spheres)
    assert torch.equal(residual, torch.zeros_like(residual))
    residual.sum().backward()
    assert torch.equal(spheres.grad, torch.zeros_like(spheres))

    # Removing an invented collision must leave the incomplete query and the
    # independent lack of observed free space visible to execution validation.
    result = scene(conditional=True, background=query).query_state(
        spheres.detach().numpy()[1:2], np.eye(4), .5)
    assert result['components']['collision/robot/background'] == 0
    assert result['components']['scene_update/background/robot'] == 1
    assert result['components']['unknown/forbidden/robot'] > 0
    assert not result['passed'] and result['status'] == 'SCENE_UPDATE_REQUIRED'
# END boundary handling never turns an unsupported query into free evidence.


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


# BEGIN one-state and large-state admission must cover expanded self-pair scratch.
def test_pair_workspace_cannot_hide_behind_a_small_sphere_count():
    current = scene()
    current.robot_sphere_materials = ('body', 'body')
    # Repeated declared pairs remain distinct positive terms at this generic
    # interface. They make pair storage much larger than two sphere records.
    pairs = [[0, 1]]*2048
    current.bind_online_robot(robot_sphere_geometry=[{
        'link_name': 'link', 'sphere_indices': [0, 1],
        'centers_local': [[0., 0., 0.], [1., 0., 0.]], 'radii': [.01, .01]}],
        robot_self_collision_pairs=pairs)
    for states, expanded_pairs in ((1, 2048), (4096, 1048576)):
        allocation = current.estimated_workspace_allocation(1, states)
        assert allocation['gpu_live'] >= expanded_pairs*24*4
        assert allocation['cpu_live'] >= expanded_pairs*24*8
# END float64 host scratch and float32 native scratch have separate admissions.
