"""Holding carries an assembly into the next skill without a grasp-pose goal."""
# BEGIN CPU counterexamples keep internal materials, other obstacles and release distinct.
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation.holding_contact import HoldingContact, bind_holding_contacts, validate_holding_conditions
from manipulation.online_geometry import FrozenPrimitives, OnlinePlanningScene
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner
from test_online_geometry import scene as empty_scene


def setup(*, release=True, other_owner=False, self_collision=False, angle=0.):
    original = empty_scene()
    pose = np.eye(4)
    pose[:3,:3] = Rotation.from_euler('z',angle,degrees=True).as_matrix()
    attachment = Attachment((0.,.75),[0.,0.,0.],
        Rotation.from_matrix(pose[:3,:3].T).as_quat()[[3,0,1,2]],'held','release' if release else None)
    query = FrozenPrimitives(primitives=[{'kind':'cuboid','dimensions':[.02]*3,
        'pose':[0.,0.,0.,1.,0.,0.,0.]}],owner_ids=('part',),source='complete test object')
    owner = CollisionOwner('part',query,PosePath(np.array([0.,1.]),np.stack([np.eye(4)]*2),'current'),
        'current held model',geometry_complete=True,moving=True,local_spheres=[[0.,0.,0.,.018]],
        attachments=(attachment,))
    owners = [owner]
    if other_owner:
        obstacle = FrozenPrimitives(primitives=deepcopy(query.primitives),owner_ids=('obstacle',),
                                    source='separately observed obstacle')
        owners.append(CollisionOwner('obstacle',obstacle,owner.pose_path,'distinct obstacle',geometry_complete=True))
    contact = HoldingContact('held-pad','held','part','pad','native-pad',(0,),(0.,.75),'current Holding fixture')
    # BEGIN retain the actual robot-body schema produced by run_recorded_skill.
    bodies = [{'body_id':'pad','frame':'pad','geometry_names':['native-pad'],
               'sphere_indices':[0],'local_spheres':[[0.,0.,0.,.005]],'source':'calibrated test robot'},
              {'body_id':'arm','frame':'arm','geometry_names':['native-arm'],
               'sphere_indices':[1],'local_spheres':[[0.,0.,0.,.005]],'source':'calibrated test robot'}]
    # END owner aliases are a separate schema and cannot stand in for bodies.
    current = OnlinePlanningScene(background=original.background,owners=owners,coverage=original.coverage,
        reference_frame='world',robot_sphere_materials=('pad',None),robot_geometry_source='calibrated test robot',
        holding_contacts=(contact,),robot_collision_bodies=bodies,
        native_geometry_materials={'native-pad':'pad'},has_gripper=False)
    current.bind_online_robot(robot_sphere_geometry=[{'link_name':'pad','sphere_indices':[0],
        'centers_local':[[0.,0.,0.]],'radii':[.005]}, {'link_name':'arm','sphere_indices':[1],
        'centers_local':[[0.,0.,0.]],'radii':[.005]}],
        robot_self_collision_pairs=[[0,1]] if self_collision else [])
    condition = {'kind':'observed_attachment','condition_id':'held','carrier_id':'part','progress':0.,
                 'T_carrier_eef':pose.tolist(),'contact_rule_ids':['held-pad']}
    validate_holding_conditions(current.holding_contacts,current.owners,[condition])
    return current,pose,condition


def query(current,pose,*,progress=.5,evidence=True,arm_center=.06):
    return current.query_state(np.array([[0.,0.,0.,.005],[arm_center,0.,0.,.005]]),pose,progress,
        condition_evidence={'held':evidence},owner_poses={'part':np.eye(4)})


@pytest.mark.parametrize('angle',[0.,27.81,85.])
def test_current_holding_accepts_arbitrary_observed_attachment_without_previous_grasp_target(angle):
    current,pose,_ = setup(angle=angle)
    result = query(current,pose)
    assert result['passed'],result
    assert result['components']['collision/robot/part']==0
    assert not any(name.startswith('contact/') for name in result['components'])
    assert not {'patch_origin','patch_axes','gap_bounds','normal_tolerance_rad'} & set(asdict(current.holding_contacts[0]))


def test_holding_does_not_clear_other_robot_material_or_other_owner_or_self():
    current,pose,_ = setup()
    assert query(current,pose,arm_center=0.)['components']['collision/robot/part']>0
    current,pose,_ = setup(other_owner=True)
    report = query(current,pose)
    assert report['components']['collision/robot/obstacle']>0
    assert report['components']['collision/part/obstacle']>0
    current,pose,_ = setup(self_collision=True)
    assert query(current,pose,arm_center=0.)['components']['collision/robot/self']>0


@pytest.mark.parametrize('evidence',[False,None])
def test_false_or_unknown_holding_cannot_authorize_current_internal_contact(evidence):
    current,pose,_ = setup()
    result = query(current,pose,evidence=evidence)
    assert not result['passed'] and result['components']['collision/robot/part']>0


@pytest.mark.parametrize('release',[False,True])
def test_release_boundary_restores_internal_material_collision(release):
    current,pose,_ = setup(release=release)
    assert query(current,pose,progress=.75-1e-6)['passed']
    result = query(current,pose,progress=.75)
    assert result['passed'] is (not release)
    assert (result['components']['collision/robot/part']>0) is release


def test_saved_binding_preserves_current_material_and_release_semantics():
    current,pose,_ = setup(angle=27.81)
    restored = OnlinePlanningScene.from_binding(current.to_execution_binding())
    for progress in (.5,.75):
        assert query(current,pose,progress=progress)==query(restored,pose,progress=progress)


def test_partial_or_relabelled_material_and_wrong_current_attachment_are_rejected():
    from manipulation.holding_contact import validate_holding_materials
    current,_,condition = setup()
    contact, = current.holding_contacts
    for changed in (replace(contact,sphere_indices=(1,)),replace(contact,native_geometry='native-arm'),
                    replace(contact,target_owner='missing'),replace(contact,material_id='another')):
        with pytest.raises(ValueError):
            validate_holding_materials((changed,),current.owners,current.robot_sphere_materials,
                                      current.native_geometry_materials,current.robot_collision_bodies)
    for field,value in (('carrier_id','missing'),('condition_id','another'),('contact_rule_ids',[])):
        with pytest.raises(ValueError):
            validate_holding_conditions(current.holding_contacts,current.owners,[dict(condition,**{field:value})])


def test_binder_selects_current_materials_without_reading_a_previous_skill():
    condition = {'kind':'observed_attachment','condition_id':'held','carrier_id':'part'}
    problem = SimpleNamespace(conditions=(condition,),phases=[SimpleNamespace(metadata={
        'carrier_id':'part','attachment_condition_id':'held','attachment_interval':[0.,1.]})])
    material = {'materials':{'pad':{'native_geometry':'native-pad','sphere_indices':[0]}}}
    original = deepcopy(condition)
    with pytest.raises(ValueError):
        bind_holding_contacts(problem,material,{'held':['missing']},source='current tool materials')
    assert problem.conditions==(condition,)
    result = bind_holding_contacts(problem,material,{'held':['pad']},source='current tool materials')
    assert result[0].condition_id=='held'
    assert problem.conditions[0]['contact_rule_ids']==[result[0].rule_id]
    assert condition==original
# END tests provide explicit semantic facts; synthetic geometry proves no physical grasp.
