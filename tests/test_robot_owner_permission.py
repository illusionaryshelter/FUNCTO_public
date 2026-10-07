"""CPU counterexamples for arrival-gated robot/owner software contact policy."""

# BEGIN the same permission drives nominal spheres and actual native solids.
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning import _ProgressMap, _scene_boundary_evaluate
from manipulation.planning_scene import CollisionOwner, RobotOwnerContactPermission
from manipulation.robot_geometry_check import NativePrimitive, PairQuery, RobotGeometryChecker
from test_independent_scene_validation import binding, checker, rigid
from test_manipulation_planning_scene import SphereQuery, inputs, pose_path, scene


def permission(**changes):
    values = dict(rule_id="arrived_close", target_owner="target", native_geometry_names=("pad",),
        sphere_indices=(0,), interval=(.5, .9), arrival_goal_id=("pose_events", 1),
        activation_event={"skill_id": "skill", "event_id": "close", "occurrence": 0, "kind": "close"},
        source="explicit named gripper and frozen-perception arrival")
    values.update(changes)
    return RobotOwnerContactPermission(**values)


def model_binding(**changes):
    model = binding()
    model["robot_owner_permissions"] = [asdict(permission(**changes))]
    return model


def native_query(checker, model, *, progress=.5, kind="actual", active=frozenset()):
    checker.bind_scene(model)
    state = checker.query(joint_names=checker.joint_names, joint_positions=[0., 0.],
        T_reference_base=np.eye(4), pairs=())
    return checker.query_scene(state, tool_pose=np.eye(4), progress=progress,
        finger_positions=[0., 0.], state_kind=kind, active_robot_owner_permissions=active)


def test_actual_arrival_latch_and_interval_are_both_required(checker):
    model = model_binding()
    assert not native_query(checker, model)["passed"]
    for progress in (.499, .901):
        assert not native_query(checker, model, progress=progress, active={"arrived_close"})["passed"]
    result = native_query(checker, model, active={"arrived_close"})
    row = result["pairs"][0]
    assert result["passed"] and row["permission"] == "arrived_close"
    assert row["signed_distance_m"] < 0 and row["contacts"]
    assert row["permission_scope"] == "explicit_robot_owner_pair"
    assert row["contact_region_checks"] == []
    assert result["active_robot_owner_permissions"] == ["arrived_close"]
    assert native_query(checker, model, kind="nominal", active=None)["passed"]
    # A new execution's empty latch cannot inherit the previous successful query.
    assert not native_query(checker, model)["passed"]
    with pytest.raises(ValueError, match="explicit active set"):
        native_query(checker, model, active=None)
    with pytest.raises(ValueError, match="bound scene"):
        native_query(checker, model, active={"another_execution"})


def test_whole_pair_permission_has_no_patch_or_penetration_allowance(checker):
    model = model_binding()
    model["owners"][0]["geometry"]["solids"][0]["pose"][2] = 0.
    result = native_query(checker, model, active={"arrived_close"})
    assert result["passed"] and result["pairs"][0]["signed_distance_m"] < -.01
    assert result["pairs"][0]["contacts"]


def test_arm_other_owner_background_and_native_self_pairs_remain_checked(checker):
    solids = [NativePrimitive(name, "base", "box", (.1, .1, .1), rigid(.049), "native named solid")
              for name in ("pad", "arm")]
    robot = RobotGeometryChecker(checker.urdf_path, base_frame="base", geometries=solids)
    model = model_binding()
    other = deepcopy(model["owners"][0])
    other["owner_id"] = "other_owner"
    model["owners"].append(other)
    model["background"]["geometry"] = deepcopy(other["geometry"])
    result = native_query(robot, model, active={"arrived_close"})
    assert not result["passed"]
    rows = {(row["actor_geometry"], row["target_owner"]): row for row in result["pairs"]}
    assert rows["pad", "target"]["permission"] == "arrived_close"
    for key in (("arm", "target"), ("pad", "other_owner"), ("pad", "background")):
        assert rows[key]["permission"] is None and rows[key]["signed_distance_m"] < 0
    state = robot.query(joint_names=robot.joint_names, joint_positions=[0., 0.], T_reference_base=np.eye(4),
        pairs=[PairQuery("pad", "arm", "required", "native robot self policy")])
    assert state["pairs"][0]["policy"] == "required"
    assert state["pairs"][0]["signed_distance_m"] < 0


def test_nominal_spheres_exempt_only_selected_pair_and_keep_raw_query():
    target = CollisionOwner("target", SphereQuery(("target",), center=(0., 0., 0.)), pose_path(), "observed target", True)
    other = CollisionOwner("other", SphereQuery(("other",), center=(0., 0., 0.)), pose_path(), "observed other", True)
    current = scene((target, other), materials=(None, None), robot_owner_permissions=(permission(),),
        background=SphereQuery(("background",), center=(0., 0., 0.)))
    args = inputs(progress=(.499, .5, .9, .901), sphere_rows=[[[0., 0., 0., .1]]*2]*4)
    terms = current.components(*args)
    torch.testing.assert_close(terms["collision/robot/target"], torch.tensor([[.4, .2, .2, .4]], dtype=torch.float64))
    for name in ("other", "background"):
        torch.testing.assert_close(terms[f"collision/robot/{name}"], torch.full((1, 4), .4, dtype=torch.float64))
    assert target.query.calls["robot", "target"].shape == args[0].shape
    assert current.required_progress_anchors == (.5, .9)


def test_binding_and_rollout_own_complete_permission_values():
    target = CollisionOwner("target", SphereQuery(("target",)), pose_path(), "observed target", True)
    current = scene((target,), robot_owner_permissions=(permission(),))
    current.background.source = "explicit empty background"
    snapshot = current.to_execution_binding()
    json.dumps(snapshot, allow_nan=False)
    clone = current.clone_for_rollout()
    current.robot_owner_permissions[0].activation_event["occurrence"] = 7
    assert snapshot["robot_owner_permissions"][0]["activation_event"]["occurrence"] == 0
    assert clone.robot_owner_permissions[0].activation_event["occurrence"] == 0
    restored = RobotOwnerContactPermission(**snapshot["robot_owner_permissions"][0])
    assert restored.arrival_goal_id == ("pose_events", 1)
    assert restored.native_geometry_names == ("pad",) and restored.sphere_indices == (0,)


@pytest.mark.parametrize("changes", [
    {"sphere_indices": (0, 0)}, {"native_geometry_names": ("pad", "pad")},
    {"native_geometry_names": "pad"}, {"arrival_goal_id": ("supports", 0)},
    {"interval": (.5, .5)}, {"activation_event": {"kind": "close"}},
])
def test_invalid_permission_identity_is_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        permission(**changes)


def test_unbound_native_owner_or_sphere_identity_is_rejected(checker):
    for changes in ({"target_owner": "missing"}, {"native_geometry_names": ("missing",)}):
        with pytest.raises(ValueError, match="bound owner"):
            checker.bind_scene(model_binding(**changes))
    target = CollisionOwner("target", SphereQuery(("target",)), pose_path(), "observed target", True)
    with pytest.raises(ValueError, match="robot sphere layout"):
        scene((target,), robot_owner_permissions=(permission(sphere_indices=(3,)),))



# BEGIN online boundaries retain finite contact only at the exact arrival.
def test_native_boundary_keeps_finite_arrival_permission_and_other_laws():
    from manipulation.online_geometry import FrozenESDF,OnlinePlanningScene
    from manipulation.planning_scene import ContactRule,ObservedSpace
    axis=np.arange(-20,21)*.001
    values=np.broadcast_to(axis[:,None,None],(41,41,41)).astype(np.float32)
    valid=np.ones_like(values,dtype=bool)
    target_query=FrozenESDF(values,origin=[-.02]*3,voxel_size=.001,
        site_valid=valid,observed_sign=valid,owner_ids=('target',),source='observed finite plane')
    background=FrozenESDF(np.ones_like(values),origin=[-.02]*3,voxel_size=.001,
        site_valid=~valid,observed_sign=~valid,owner_ids=('background',),source='no visible background')
    target=CollisionOwner('target',target_query,pose_path(),'current target',False)
    rule=ContactRule(rule_id='finite_pad',actor='robot',target_owner='target',interval=(.5,.9),
        sphere_indices=(0,),material_id='pad',feature_points=[[0.,-.005,-.005],[0.,.005,-.005],
            [0.,.005,.005],[0.,-.005,.005]],feature_normal=[-1.,0.,0.],patch_origin=[0.,0.,0.],
        patch_axes=[[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]],patch_half_size=[.005,.005],
        gap_bounds=(-.003,.003),normal_tolerance_rad=.1,modeling_allowance_m=.003,
        geometry_error_m=.003,minimum_thickness_m=.01,source='explicit complete pad polygon')
    coverage=ObservedSpace(~valid,origin=[-.02]*3,voxel_size=.001,
        observation_ids=('A',),reference_frame='world')
    current=OnlinePlanningScene(background=background,owners=(target,),coverage=coverage,
        reference_frame='world',robot_sphere_materials=('pad',),robot_geometry_source='native sphere',
        contacts=(rule,),has_gripper=False,unknown_policy={'outside_regions':'forbid',
            'conditional_regions':[{'lower_m':[-.02]*3,'upper_m':[.02]*3,'source':'explicit finite operation'}]})
    current.bind_online_robot(robot_sphere_geometry=[{'link_name':'pad','sphere_indices':[0],
        'centers_local':[[.003,0.,0.]],'radii':[.004]}],robot_self_collision_pairs=[])
    mapping=_ProgressMap(np.array([0.,1.]),np.array([0.,1.]))
    arguments=inputs(progress=(.499,.5,.9),sphere_rows=[[[.003,0.,0.,.004]]]*3,dtype=torch.float32)
    spheres,position,quaternion,progress=arguments
    result=_scene_boundary_evaluate(current,SimpleNamespace(start_progress=0.,end_progress=1.),mapping,
        spheres,position,quaternion,progress,torch.zeros((1,3,1)),('joint',),
        trajectory_time=progress,normalized_time=progress)
    assert result[0,0]>0 and result[0,1]==0 and result[0,2]>0
    # Reaching the arrival cannot forgive a reversed material normal. The
    # finite rule retains its normal test and the ordinary occupied cells.
    quaternion=quaternion.clone()
    quaternion[...,0]=0.;quaternion[...,3]=1.
    reversed_normal=_scene_boundary_evaluate(current,SimpleNamespace(start_progress=0.,end_progress=1.),mapping,
        spheres,position,quaternion,progress,torch.zeros((1,3,1)),('joint',),
        trajectory_time=progress,normalized_time=progress)
    assert reversed_normal[0,1]>0
# END legacy pair-level source geometry tests above are explicit audit only.
