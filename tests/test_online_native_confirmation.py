"""Native confirmation must remove proxy false positives without permissions leaks."""
# BEGIN CPU regressions for positive-sphere/native-solid confirmation semantics.
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.online_geometry import FrozenESDF, FrozenPrimitives, OnlinePlanningScene
from manipulation.online_native_confirmation import NativeCollisionConfirmation
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import CollisionOwner, ContactRule, ObservedSpace
from manipulation.robot_geometry_check import NativeSolid, RobotGeometryChecker
from observed_geometry_fixtures import point_source, surface_source


def _field(points=((.005, 0., 0.),), geometry=None):
    # BEGIN source points have zero extent; proposal raster cells are not solids.
    return FrozenESDF.from_source(point_source(points) if geometry is None else geometry,
        origin=[-.02]*3, shape=(41, 41, 41), voxel_size=.001,
        owner_ids=('part',), source='synthetic observed source')
    # END native confirmation consumes this exact source even when its field is positive.


def _box(center=(0., 0., 0.), half=(.0004, .0004, .0004)):
    return np.asarray(center) + np.asarray(list(product((-1., 1.), repeat=3))) * half


def _rule(*, indices=(0,), half=(.002, .002)):
    return ContactRule(rule_id='patch', actor='robot', target_owner='part', interval=(.2, .8),
        sphere_indices=indices, material_id='pad', feature_points=np.array([
            [-.01, -.01, 0.], [.01, -.01, 0.], [.01, .01, 0.], [-.01, .01, 0.]]),
        feature_normal=np.array([0., 0., -1.]), patch_origin=np.zeros(3),
        patch_axes=np.eye(3), patch_half_size=np.array(half), gap_bounds=(-.0015, .0015),
        normal_tolerance_rad=.2, modeling_allowance_m=.0015, geometry_error_m=.0015,
        minimum_thickness_m=.008, source='declared finite source patch', required=False,
        online_material={'kind': 'native_planar_extrusion_v1', 'thickness_m': .008,
            'native_geometry': 'pad_box', 'sphere_indices': list(indices),
            'source': 'registered native planar box', 'sphere_binding_source': 'complete synthetic native cover'})


def _setup(tmp_path, *, query=None, native=None, contacts=(), second_body=False):
    urdf = tmp_path / 'robot.urdf'
    urdf.write_text('''<robot name="slide"><link name="base"/><link name="tool"/>
        <joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
        <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
        </joint></robot>''')
    names = ['pad_box'] if contacts else ['body_box']
    vertices = [_box() if native is None else native]
    materials = ['pad' if contacts else None]
    if second_body:
        names.append('body_box'); vertices.append(_box()); materials.append(None)
    checker = RobotGeometryChecker(urdf, base_frame='base', geometries=[
        NativeSolid(name, 'tool', shape, 'explicit synthetic native solid')
        for name, shape in zip(names, vertices)])
    query = _field() if query is None else query
    background = FrozenESDF.from_source(point_source(), origin=[-1.]*3, shape=(3, 3, 3), voxel_size=1.,
        owner_ids=('background',), source='empty synthetic background')
    cover = ObservedSpace(np.ones((3, 3, 3), bool), origin=[-1.]*3, voxel_size=1.,
        observation_ids=('test',), reference_frame='world')
    owner = CollisionOwner('part', query, PosePath(np.array([0., 1.]),
        np.stack((np.eye(4), np.eye(4))), 'explicit owner pose'), 'observed owner', True)
    scene = OnlinePlanningScene(background=background, owners=(owner,), coverage=cover,
        reference_frame='world', robot_sphere_materials=materials, robot_geometry_source='synthetic robot',
        contacts=contacts, has_gripper=False, unknown_policy={'conditional_regions': [], 'outside_regions': 'forbid'},
        robot_collision_bodies=[{'geometry_names': [name], 'sphere_indices': [index]}
                                for index, name in enumerate(names)],
        native_geometry_materials={'pad_box': 'pad'} if contacts else {})
    scene.bind_online_robot(robot_sphere_geometry=[{'link_name': 'tool', 'sphere_indices': list(range(len(names))),
        'centers_local': [[0., 0., 0.]]*len(names), 'radii': [.018]*len(names)}], robot_self_collision_pairs=[])
    return checker, scene


def _state(checker, x=0.):
    return checker.query(joint_names=['slide'], joint_positions=[x],
                         T_reference_base=np.eye(4), pairs=())


def _confirm(confirm, checker, *, ids=(0,), active=(), x=0., owner_pose=None):
    return confirm.confirm(robot_state=_state(checker, x), candidates={'part': ids},
        owner_poses={'part': np.eye(4) if owner_pose is None else owner_pose}, effective_contact_rule_ids=active)


def test_proxy_overlap_clears_but_native_intersection_and_current_owner_pose_do_not(tmp_path):
    checker, scene = _setup(tmp_path)
    confirm = NativeCollisionConfirmation(checker, scene)
    assert _confirm(confirm, checker)['cleared_targets'] == ['part']
    assert _confirm(confirm, checker, x=.005)['collision_targets'] == ['part']
    moved = np.eye(4); moved[0, 3] = -.005
    assert _confirm(confirm, checker, owner_pose=moved)['collision_targets'] == ['part']
    none = confirm.confirm(robot_state=_state(checker), candidates={}, owner_poses={}, effective_contact_rule_ids=())
    assert not none['pairs'] and none['statistics']['coal_queries'] == 0


@pytest.mark.parametrize('kind,dimensions', [('cuboid', [.001]*3), ('box', [.0005]*3),
    ('sphere', [.0005]), ('cylinder', [.0005, .0005]), ('capsule', [.0005, .0005])])
def test_analytic_primitives_keep_exact_geometry_and_half_height_conventions(tmp_path, kind, dimensions):
    query = FrozenPrimitives(primitives=[{'kind': kind, 'dimensions': dimensions,
        'pose': [.005, 0., 0., 1., 0., 0., 0.]}], owner_ids=('part',), source='calibrated primitive')
    checker, scene = _setup(tmp_path, query=query)
    confirm = NativeCollisionConfirmation(checker, scene)
    assert _confirm(confirm, checker)['cleared_targets'] == ['part']
    result = _confirm(confirm, checker, x=.005)
    assert result['collision_targets'] == ['part']
    assert result['pairs'][0]['target_representation'] == 'analytic_frozen_primitive'


def test_effective_finite_permission_never_exempts_nonpad_or_other_owner_material(tmp_path):
    contact = _rule()
    native = _box(center=(0., 0., .003), half=(.01, .01, .004))
    checker, scene = _setup(tmp_path, query=_field(points=((0., 0., 0.),)), native=native,
                            contacts=(contact,), second_body=True)
    confirm = NativeCollisionConfirmation(checker, scene)
    assert _confirm(confirm, checker, active=())['collision_targets'] == ['part']
    assert _confirm(confirm, checker, active={'patch'})['cleared_targets'] == ['part']
    both = _confirm(confirm, checker, ids=(0, 1), active={'patch'})
    assert both['collision_targets'] == ['part'] and not both['cleared_targets']
    assert next(row for row in both['pairs'] if row['actor_geometry'] == 'body_box')['effective_contact_rule_ids'] == []


@pytest.mark.parametrize('points', [((.0005, 0., 0.), (.0015, 0., 0.)), ((.005, 0., 0.),)])
def test_permission_removes_whole_source_components_only_and_retains_other_patch_obstacle(tmp_path, points):
    # BEGIN a source segment crossing the patch boundary cannot inherit permission.
    # Its centre is inside the patch, but its endpoint at 1.5 mm is outside.
    contact = _rule(half=(.0011, .002))
    checker, scene = _setup(tmp_path, query=_field(geometry=surface_source([points])),
        native=_box(center=(0., 0., .003), half=(.01, .01, .004)), contacts=(contact,))
    assert _confirm(NativeCollisionConfirmation(checker, scene), checker,
                    active={'patch'})['collision_targets'] == ['part']
    # END no voxel half-width is invented for the original source.


def test_primitive_finite_mode_uses_same_prepared_cell_removal(tmp_path):
    # The primitive's prepared grid conservatively covers cells around its
    # analytic surface. This declared synthetic patch contains those WHOLE
    # cells; a narrower patch correctly leaves forbidden boundary cells.
    contact = replace(_rule(half=(.003, .003)), modeling_allowance_m=.003,
                      geometry_error_m=.003, gap_bounds=(-.003, .003))
    query = FrozenPrimitives(primitives=[{'kind': 'cuboid', 'dimensions': [.001]*3}],
        owner_ids=('part',), source='calibrated contact primitive', contact_voxel_size_m=.001)
    checker, scene = _setup(tmp_path, query=query,
        native=_box(center=(0., 0., .004), half=(.01, .01, .004)), contacts=(contact,))
    confirm = NativeCollisionConfirmation(checker, scene)
    assert _confirm(confirm, checker, active=())['collision_targets'] == ['part']
    allowed = _confirm(confirm, checker, active={'patch'})
    assert allowed['cleared_targets'] == ['part']
    assert allowed['pairs'][0]['target_representation'] == 'frozen_primitive_contact_cells'


def test_partial_native_sphere_permission_cannot_expand_to_whole_native_body(tmp_path):
    contact = _rule()
    checker, scene = _setup(tmp_path, contacts=(contact,), second_body=True)
    scene.robot_collision_bodies = ({'geometry_names': ['pad_box'], 'sphere_indices': [0, 1]},)
    with pytest.raises(ValueError, match='complete material/sphere ownership'):
        NativeCollisionConfirmation(checker, scene)


def test_numerical_boundary_and_request_cancellation_cannot_publish_clear(tmp_path):
    checker, scene = _setup(tmp_path)
    confirmation = NativeCollisionConfirmation(checker, scene)
    touching = _confirm(confirmation, checker, x=.0046)
    assert touching['unresolved_targets'] == ['part'] and not touching['cleared_targets']

    def cancelled():
        raise TimeoutError('existing caller request expired')

    with pytest.raises(TimeoutError, match='request expired'):
        confirmation.confirm(robot_state=_state(checker), candidates={'part': [0]},
            owner_poses={'part': np.eye(4)}, effective_contact_rule_ids=(), check_request=cancelled)


def test_effective_rule_for_one_owner_does_not_remove_another_owner(tmp_path):
    contact = _rule()
    checker, scene = _setup(tmp_path, query=_field(points=((0., 0., 0.),)),
        native=_box(center=(0., 0., .003), half=(.01, .01, .004)), contacts=(contact,))
    other = _field(points=((0., 0., 0.),))
    other.owner_ids = frozenset(('other',))
    scene.owners += (CollisionOwner('other', other, scene.owners[0].pose_path,
                                    'independent observed owner', True),)
    result = NativeCollisionConfirmation(checker, scene).confirm(robot_state=_state(checker),
        candidates={'part': [0], 'other': [0]}, owner_poses={'part': np.eye(4), 'other': np.eye(4)},
        effective_contact_rule_ids={'patch'})
    assert result['cleared_targets'] == ['part'] and result['collision_targets'] == ['other']
# END unsupported or unpermitted geometry never inherits a cleared pad candidate.


# BEGIN adapter regressions preserve nonfinite and non-collision obligations.
def _adapter(checker, scene):
    from manipulation.online_native_confirmation import OnlineCollisionConfirmation
    validator = SimpleNamespace(checker=checker, joint_names=['slide'], T_reference_base=np.eye(4))
    return OnlineCollisionConfirmation(validator, scene)


def _candidate_evidence(values):
    values = torch.tensor(values, dtype=torch.float64)
    batch, horizon = values.shape[:2]
    quaternion = torch.zeros((batch, horizon, 4), dtype=torch.float64)
    quaternion[..., 0] = 1.
    return {'targets': {'collision/robot/part': values},
        'joint_position': torch.zeros((batch, horizon, 1), dtype=torch.float64), 'joint_names': ['slide'],
        'finger_positions': None,
        'owner_poses': {'part': (torch.zeros((batch, horizon, 3), dtype=torch.float64), quaternion)},
        'effective_rules': {}}


def test_adapter_only_clears_confirmed_collision_and_keeps_original_other_terms(tmp_path):
    checker, scene = _setup(tmp_path)
    adapter = _adapter(checker, scene)
    evidence = _candidate_evidence([[[.01]]])
    collision = torch.tensor([[.01]], dtype=torch.float64)
    costs = {'collision/robot/part': collision,
        'unknown/forbidden/robot': torch.tensor([[.2]]),
        'scene_update/robot/part': torch.tensor([[1.]]),
        'contact/normal/declared': torch.tensor([[.3]]),
        'contact/material_band/declared': torch.tensor([[.4]]),
        'gripper/controller': torch.tensor([[.5]]),
        'collision/robot/self': torch.tensor([[.6]])}
    result = adapter(costs, evidence)
    assert result['collision/robot/part'].item() == 0 and collision.item() == .01
    for name, value in costs.items():
        if name != 'collision/robot/part':
            assert result[name] is value
    assert adapter.last_report['states'][0]['sphere_residuals']['collision/robot/part'] == .01


@pytest.mark.parametrize('sphere_values,cost', [([float('nan')], .01),
    ([float('inf')], .01), ([.01, float('nan')], .01), ([.01], float('nan')),
    ([.01], float('inf')), ([float('-inf'), .01], .01)])
def test_adapter_cannot_clear_nonfinite_sphere_or_aggregate_evidence(tmp_path, sphere_values, cost):
    checker, scene = _setup(tmp_path)
    adapter = _adapter(checker, scene)
    evidence = _candidate_evidence([[sphere_values]])
    original = torch.tensor([[cost]], dtype=torch.float64)
    result = adapter({'collision/robot/part': original}, evidence)
    assert result['collision/robot/part'] is original
    assert adapter.engine is None and adapter.last_report['candidate_states'] == 0


def test_adapter_finite_state_clear_does_not_erase_nonfinite_neighbor(tmp_path):
    checker, scene = _setup(tmp_path)
    adapter = _adapter(checker, scene)
    evidence = _candidate_evidence([[[float('nan')], [.01]]])
    costs = {'collision/robot/part': torch.tensor([[float('nan'), .01]], dtype=torch.float64)}
    result = adapter(costs, evidence)
    assert torch.isnan(result['collision/robot/part'][0, 0])
    assert result['collision/robot/part'][0, 1] == 0
    assert adapter.last_report['candidate_states'] == 1
# END the shared adapter never converts absent/nonfinite evidence into clearance.


# BEGIN candidate retention and native confirmation share one exact authority set.
def test_world_component_inventory_excludes_self_even_when_an_owner_uses_that_name(tmp_path):
    checker, scene = _setup(tmp_path)
    alias = scene.owners[0].query.clone_for_rollout()
    alias.owner_ids = frozenset(('self',))
    scene.owners += (replace(scene.owners[0], owner_id='self', query=alias),)
    assert scene.native_world_collision_components == frozenset((
        'collision/robot/background', 'collision/robot/part'))
    # The declared name set does not depend on a lazy checker being created,
    # and the native engine cannot reinterpret a self term as a world target.
    engine = NativeCollisionConfirmation(checker, scene)
    with pytest.raises(ValueError, match='Candidate target is absent'):
        engine.confirm(robot_state=_state(checker), candidates={'self': [0]},
            owner_poses={'self': np.eye(4)}, effective_contact_rule_ids=())


@pytest.mark.parametrize('name,value', [
    ('collision/robot/self', .01),
    ('collision/payload/part', .01),
    ('collision/robot/missing', 0.),
    ('collision/robot/part/extra', .01),
    ('conditional/collision/robot/part', .01),
    ('contact/required/patch', .01),
])
def test_adapter_rejects_undeclared_components_before_creating_a_checker(tmp_path, name, value):
    checker, scene = _setup(tmp_path)
    adapter = _adapter(checker, scene)
    evidence = _candidate_evidence([[[value]]])
    evidence['targets'] = {name: evidence['targets']['collision/robot/part']}
    with pytest.raises(ValueError, match='declared actor/world collision components'):
        adapter({name: torch.tensor([[value]], dtype=torch.float64)}, evidence)
    assert adapter.engine is None


@pytest.mark.parametrize('target', ['self', 'background'])
def test_adapter_cannot_clear_an_undeclared_or_unqueried_component(tmp_path, target):
    checker, scene = _setup(tmp_path)
    adapter = _adapter(checker, scene)
    # The engine response is deliberately invalid: background is declared but
    # was not queried, while self is never in the confirmation inventory.
    adapter.engine = SimpleNamespace(confirm=lambda **kwargs: {'cleared_targets': [target]})
    evidence = _candidate_evidence([[[.01]]])
    original = torch.tensor([[.01]], dtype=torch.float64)
    other = torch.tensor([[.02]], dtype=torch.float64)
    with pytest.raises(ValueError, match='unqueried collision target'):
        adapter({'collision/robot/part': original, 'collision/robot/'+target: other}, evidence)
    assert original.item() == .01 and other.item() == .02
# END neither an input name nor a checker response can enlarge confirmation authority.


# BEGIN registered attached boxes use the same native candidate confirmation.
def _payload(scene, *, two_boxes=False):
    rows, balls = [], []
    for index, x in enumerate((0., .005) if two_boxes else (0.,)):
        rows.append({'kind': 'cuboid', 'dimensions': [.0008]*3,
            'pose': [x, 0., 0., 1., 0., 0., 0.],
            'native_binding': {'geometry_name': f'payload_box_{index}', 'sphere_indices': [index],
                'subdivisions': [1, 1, 1], 'source': 'complete synthetic Cartesian box cells'}})
        balls.append([x, 0., 0., .018])
    query = FrozenPrimitives(primitives=rows, owner_ids=('payload',), source='known synthetic payload')
    owner = CollisionOwner('payload', query, scene.owners[0].pose_path, 'fixed observed attachment',
        True, local_spheres=balls, sphere_materials=tuple(None for _ in balls), moving=True)
    scene.owners += (owner,)
    return owner


def test_owner_proxy_clear_preserves_other_body_collision_and_current_actor_pose(tmp_path):
    checker, scene = _setup(tmp_path)
    _payload(scene, two_boxes=True)
    engine = NativeCollisionConfirmation(checker, scene)
    poses = {'part': np.eye(4), 'payload': np.eye(4)}
    def query(ids):
        return engine.confirm(robot_state=None, candidates={'part': ids}, owner_poses=poses,
            effective_contact_rule_ids=(), actor='payload')
    assert query([0])['cleared_targets'] == ['part']
    assert query([0, 1])['collision_targets'] == ['part']
    poses['payload'][0, 3] = .005
    assert query([0])['collision_targets'] == ['part']
    assert 'collision/payload/part' in scene.native_world_collision_components
    assert 'collision/payload/payload' not in scene.native_world_collision_components


@pytest.mark.parametrize('fault', ['missing_cell', 'undersized_sphere', 'overlap_indices', 'unbound_primitive'])
def test_owner_confirmation_rejects_incomplete_or_false_sphere_registration(tmp_path, fault):
    checker, scene = _setup(tmp_path)
    owner = _payload(scene, two_boxes=True)
    if fault == 'missing_cell':
        owner.query.primitives[0]['native_binding']['subdivisions'] = [2, 1, 1]
    elif fault == 'undersized_sphere':
        owner.local_spheres[0, 3] = .0001
    elif fault == 'overlap_indices':
        owner.query.primitives[1]['native_binding']['sphere_indices'] = [0]
    else:
        del owner.query.primitives[1]['native_binding']
        assert 'collision/payload/part' not in scene.native_world_collision_components
        engine = NativeCollisionConfirmation(checker, scene)
        with pytest.raises(ValueError, match='actor lacks complete registered geometry'):
            engine.confirm(robot_state=None, candidates={'part': [0]}, owner_poses={},
                effective_contact_rule_ids=(), actor='payload')
        return
    with pytest.raises(ValueError, match='Native owner'):
        NativeCollisionConfirmation(checker, scene)


def test_owner_adapter_uses_query_pose_without_robot_fk_and_keeps_contact_unknown(tmp_path, monkeypatch):
    checker, scene = _setup(tmp_path)
    _payload(scene)
    evidence = _candidate_evidence([[[.01]]])
    evidence['targets'] = {'collision/payload/part': evidence['targets']['collision/robot/part']}
    evidence['owner_poses']['payload'] = evidence['owner_poses']['part']
    evidence['joint_position'], evidence['joint_names'] = None, None
    monkeypatch.setattr(checker, 'query', lambda **kw: pytest.fail('owner poses already came from this query'))
    adapter = _adapter(checker, scene)
    costs = {'collision/payload/part': torch.tensor([[.01]], dtype=torch.float64),
        'unknown/forbidden/payload': torch.ones(1, 1), 'contact/material_band/edge': torch.ones(1, 1)}
    result = adapter(costs, evidence)
    assert result['collision/payload/part'].item() == 0.
    assert result['unknown/forbidden/payload'] is costs['unknown/forbidden/payload']
    assert result['contact/material_band/edge'] is costs['contact/material_band/edge']
    assert adapter.last_report['candidate_states'] == 1
    assert adapter.last_report['states'][0]['actor'] == 'payload'


def test_registered_owner_box_geometry_and_sphere_partition_survive_scene_binding(tmp_path):
    _, scene = _setup(tmp_path)
    original = _payload(scene)
    restored = OnlinePlanningScene.from_binding(scene.to_execution_binding())
    current = next(owner for owner in restored.owners if owner.owner_id == 'payload')
    assert current.query.primitives == original.query.primitives
    assert restored.native_world_collision_components == scene.native_world_collision_components


def _edge_payload(scene):
    from test_online_contact import edge_rule
    from scipy.spatial.transform import Rotation
    rule = edge_rule()
    material = rule.online_material
    pose = np.asarray(material['T_actor_box'])
    query = FrozenPrimitives(primitives=[{'kind': 'cuboid',
        'dimensions': (2*np.asarray(material['half_size_m'])).tolist(),
        'pose': [*pose[:3, 3], *Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]],
        'native_binding': {'geometry_name': 'edge_box', 'sphere_indices': [0],
            'subdivisions': [1, 1, 1], 'source': 'one complete native box cell'}}],
        owner_ids=('payload',), source='registered actual edge body')
    owner = CollisionOwner('payload', query, scene.owners[0].pose_path, 'same-query owner pose', True,
        local_spheres=[[*pose[:3, 3], .02]], sphere_materials=(rule.material_id,), moving=True)
    scene.owners += (owner,)
    scene.contacts = (rule,)
    return rule, owner


def test_edge_native_clearance_does_not_clear_forbidden_side_material_condition(tmp_path):
    # The source point lies inside a shallow box wedge, not on the finite edge.
    # A finite target permission can remove it from the forbidden field; the
    # material-band term must independently retain the actual side violation.
    checker, scene = _setup(tmp_path, query=_field(points=((0., .0002, .0003),)))
    rule, owner = _edge_payload(scene)
    scene = OnlinePlanningScene.from_binding(scene.to_execution_binding())
    engine = NativeCollisionConfirmation(checker, scene)
    poses = {'part': np.eye(4), 'payload': np.eye(4)}
    blocked = engine.confirm(robot_state=None, candidates={'part': [0]}, owner_poses=poses,
        effective_contact_rule_ids=(), actor='payload')
    assert blocked['collision_targets'] == ['part']
    from test_online_contact import policy
    from manipulation.online_contact import native_material_conditions
    current, normal = policy(rule)
    terms, _ = native_material_conditions(rule, current, normal, scene.owners[0].query._contact_model)
    assert terms['material_band'].item() > .00029
    permitted = engine.confirm(robot_state=None, candidates={'part': [0]}, owner_poses=poses,
        effective_contact_rule_ids={'edge'}, actor='payload')
    assert permitted['cleared_targets'] == ['part']


@pytest.mark.parametrize('fault', ['body_changed', 'cover_changed'])
def test_scene_admission_validates_native_owner_even_before_any_collision_query(tmp_path, fault):
    _, scene = _setup(tmp_path)
    _, owner = _edge_payload(scene)
    if fault == 'body_changed':
        # Keep the declared edge and its proof internally valid while changing
        # the actual solid; labels alone must not authorize this mismatch.
        owner.query.primitives[0]['dimensions'][0] *= 1.1
        message = 'material differs from its registered complete solid'
    else:
        owner.local_spheres[0, 3] = .0001
        message = 'sphere does not enclose'
    with pytest.raises(ValueError, match=message):
        OnlinePlanningScene.from_binding(scene.to_execution_binding())


@pytest.mark.parametrize('required', [False, True])
@pytest.mark.parametrize('separated', [False, True])
def test_optional_normal_only_constrains_possible_full_body_contact(tmp_path, required, separated):
    # Rotate a pad 90 degrees: its side is inside the patch at the near pose,
    # and the entire solid is separated at the far pose. A required contact
    # keeps its normal requirement even when separated; an optional one does not.
    contact = replace(_rule(), required=required)
    _, scene = _setup(tmp_path, query=_field(points=((0., -.001, .0003),)), contacts=(contact,),
        native=_box(center=(0., 0., .004), half=(.01, .01, .004)))
    position = torch.tensor([[[.1 if separated else 0., 0., 0.]]], dtype=torch.float64)
    angle = np.sqrt(.5)
    quaternion = torch.tensor([[[angle, angle, 0., 0.]]], dtype=torch.float64)
    body = torch.cat((position.unsqueeze(-2), torch.full((1, 1, 1, 1), .018, dtype=torch.float64)), -1)
    terms = scene.components(body, position, quaternion, torch.full((1, 1), .5, dtype=torch.float64))
    residual = terms['contact/normal/patch'].item()
    assert (residual > 0) == (required or not separated)
# END native clearance never changes finite contact or observation requirements.
