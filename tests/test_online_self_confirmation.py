"""Positive self proxies use robot CAD only, preserving every other obligation."""

# BEGIN bounded CPU counterexamples for the shared self-confirmation boundary.
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
from manipulation.online_native_confirmation import OnlineCollisionConfirmation
from manipulation.planning_scene import ObservedSpace
from manipulation.robot_geometry_check import NativePrimitive, PairQuery, RobotGeometryChecker
from manipulation.robot_validation import RobotStateValidator
from observed_geometry_fixtures import point_source


@pytest.fixture
def assembly(tmp_path):
    """Two boxes have tighter geometry than their enclosing sphere pair."""
    path = tmp_path / 'robot.urdf'
    path.write_text('''<robot name="slide"><link name="base"/><link name="tool"/>
        <joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
        <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
        </joint></robot>''')
    checker = RobotGeometryChecker(path, base_frame='base', geometries=[
        NativePrimitive(name, frame, 'box', (.01, .01, .01), np.eye(4), 'known robot CAD')
        for name, frame in (('base_box', 'base'), ('slider_box', 'tool'))])
    validator = RobotStateValidator(checker, joint_names=('slide',), arm_joint_names=('slide',),
        finger_joint_names=(), pairs=[PairQuery('base_box', 'slider_box', 'required', 'robot model policy')],
        named_velocity_limits={'slide': 1.}, reference_frame='world', T_reference_base=np.eye(4),
        tool_frame='tool', T_frame_tcp=np.eye(4), fk_translation_tolerance_m=1e-8,
        fk_rotation_tolerance_rad=1e-8, source='explicit robot model')
    background = FrozenESDF.from_source(point_source(), origin=[-1.]*3, shape=(3, 3, 3),
        voxel_size=1., owner_ids=('background',), source='empty observed background')
    coverage = ObservedSpace(np.ones((3, 3, 3), bool), origin=[-1.]*3, voxel_size=1.,
        observation_ids=('test',), reference_frame='world')
    scene = OnlinePlanningScene(background=background, owners=(), coverage=coverage,
        reference_frame='world', robot_sphere_materials=(None, None),
        robot_geometry_source='known two-body sphere cover', has_gripper=False,
        unknown_policy={'outside_regions': 'forbid', 'conditional_regions': []},
        robot_collision_bodies=[{'geometry_names': [name], 'sphere_indices': [index]}
            for index, name in enumerate(('base_box', 'slider_box'))])
    scene.bind_online_robot(robot_sphere_geometry=[
        {'link_name': link, 'sphere_indices': [index], 'centers_local': [[0., 0., 0.]], 'radii': [.04]}
        for index, link in enumerate(('base', 'tool'))], robot_self_collision_pairs=[[0, 1]])
    validator.check_planning_self = Mock(wraps=validator.check_planning_self)
    return validator, scene


def _query(adapter, q=(.05,), residual=(.03,), time=None, source='self_collision', dtype=torch.float64):
    proxy = torch.tensor([residual], dtype=dtype)
    positions = torch.tensor([q], dtype=dtype)[..., None]
    times = torch.tensor(np.asarray([np.arange(len(q)) if time is None else time]), dtype=dtype)
    return proxy, adapter.confirm_self(proxy, positions, ('slide',), times, source=source)


def test_real_cad_clear_intersection_and_numerical_boundary(assembly):
    validator, scene = assembly
    adapter = OnlineCollisionConfirmation(validator, scene)
    proxy, revised = _query(adapter, q=(.05, .005, .01), residual=(.03, .075, .07))
    assert revised.tolist() == [[0., .075, .07]]
    assert proxy.tolist() == [[.03, .075, .07]], 'Confirmation never changes the caller proxy tensor'
    records = adapter.self_report['states']
    assert [row['passed'] for row in records] == [True, False, False]
    assert all(row['queried_pairs'] == 1 for row in records)
    assert records[0]['minimum_required_clearance_m'] == pytest.approx(.04)
    assert records[1]['violations'][0]['relation'] == 'penetrating'
    assert records[2]['violations'][0]['relation'] == 'numerical_boundary'
    assert validator.check_planning_self.call_count == 3


def test_exact_state_reuses_one_check_across_scene_native_and_anchor_channels(assembly):
    validator, scene = assembly
    adapter = OnlineCollisionConfirmation(validator, scene)
    for source in ('self_collision', 'anchor_self_collision', 'collision/robot/self'):
        _, revised = _query(adapter, source=source)
        assert revised.item() == 0.
    assert validator.check_planning_self.call_count == 1
    assert set(adapter.self_report['states'][0]['sphere_residuals']) == {
        'self_collision', 'anchor_self_collision', 'collision/robot/self'}
    _query(adapter, time=(.5,))
    _query(adapter, q=(.06,))
    assert validator.check_planning_self.call_count == 3
    _query(OnlineCollisionConfirmation(validator, scene))
    assert validator.check_planning_self.call_count == 4, 'A new candidate cannot reuse old state evidence'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_channel_survives_another_channel_clear_at_the_same_state(assembly, bad):
    validator, scene = assembly
    adapter = OnlineCollisionConfirmation(validator, scene)
    _query(adapter)
    original, result = _query(adapter, residual=(bad,), source='collision/robot/self')
    assert result is original
    assert not torch.isfinite(result).any()
    assert validator.check_planning_self.call_count == 1
    zero, result = _query(adapter, residual=(0.,))
    assert result is zero and validator.check_planning_self.call_count == 1


def test_zero_proxy_does_not_query_cad_and_positive_requires_explicit_state(assembly):
    validator, scene = assembly
    adapter = OnlineCollisionConfirmation(validator, scene)
    zero = torch.zeros((1, 1))
    assert adapter.confirm_self(zero, None, None, None, source='self_collision') is zero
    validator.check_planning_self.assert_not_called()
    with pytest.raises(ValueError, match='same named q'):
        adapter.confirm_self(torch.ones((1, 1)), None, None, None, source='self_collision')
    with pytest.raises(ValueError, match='sphere-proposal identity'):
        adapter.confirm_self(zero, None, None, None, source='collision/robot/self_prismatic')


def test_cancellation_after_cad_cannot_publish_or_cache_a_clear(assembly):
    validator, scene = assembly
    check = Mock(side_effect=[None, TimeoutError('request expired')])
    adapter = OnlineCollisionConfirmation(validator, scene, check_request=check)
    with pytest.raises(TimeoutError, match='request expired'):
        _query(adapter)
    assert adapter.self_report['candidate_states'] == 0
    assert not adapter.self_report['states']
    assert validator.check_planning_self.call_count == 1


def test_scene_self_query_clears_only_spheres_and_retains_original_diagnostics(assembly):
    validator, scene = assembly
    adapter = OnlineCollisionConfirmation(validator, scene)
    spheres = torch.tensor([[[[0., 0., 0., .04], [.05, 0., 0., .04]]]], dtype=torch.float64)
    q = torch.tensor([[[.05]]], dtype=torch.float64)
    arguments = (spheres, torch.tensor([[[.05, 0., 0.]]], dtype=torch.float64),
        torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64), torch.zeros((1, 1), dtype=torch.float64),
        q, ('slide',))
    raw = scene.components(*arguments, trajectory_time=torch.zeros((1, 1), dtype=torch.float64))
    corrected = scene.components(*arguments, trajectory_time=torch.zeros((1, 1), dtype=torch.float64),
        native_confirmation=adapter)
    assert raw['collision/robot/self'].item() == pytest.approx(.03)
    assert corrected['collision/robot/self'].item() == 0.
    assert all(torch.equal(value, corrected[name]) for name, value in raw.items()
               if name != 'collision/robot/self')
    assert adapter.last_report['self_collision']['states'][0]['sphere_residuals'] == {
        'collision/robot/self': pytest.approx(.03)}
    assert 'collision/robot/self' in scene.native_collision_components
    assert 'collision/robot/self_prismatic' not in scene.native_collision_components
    assert 'collision/robot/self' not in scene.native_world_collision_components


def test_analytic_prismatic_collision_never_enters_sphere_confirmation():
    # This mock isolates dispatch, not solid geometry: the existing SAT tests
    # independently prove the hand residual. A deliberately permissive sphere
    # checker must still be unable to clear that analytic hard term.
    from test_prismatic_self_collision import hand, scene as hand_scene
    current = hand(nominal=True)
    world = hand_scene(current)
    current.self_collision_model.components = Mock(return_value=(torch.tensor([[.015]], dtype=torch.float64),
                                                                 torch.tensor([[.02]], dtype=torch.float64)))
    adapter = SimpleNamespace(confirm_self=Mock(side_effect=lambda residual, *args, **kwargs:
        torch.zeros_like(residual)))
    class Confirm:
        confirm_self = adapter.confirm_self
        def __call__(self, costs, evidence):
            return costs
    spheres = torch.tensor([[[[-.05, 0., 0., .102], [.05, 0., 0., .102], [-.05, 0., 0., .01]]]],
                           dtype=torch.float64)
    terms = world.components(spheres, torch.zeros((1, 1, 3), dtype=torch.float64),
        torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64), torch.zeros((1, 1), dtype=torch.float64),
        torch.tensor([[[-.05, .05]]], dtype=torch.float64), ('left', 'right'),
        trajectory_time=torch.tensor([[.5]], dtype=torch.float64), native_confirmation=Confirm())
    assert terms['collision/robot/self'].item() == 0.
    assert terms['collision/robot/self_prismatic'].item() == .015
    assert terms['conditional/collision/robot/self_prismatic'].item() == .02
    assert adapter.confirm_self.call_count == 1
    assert adapter.confirm_self.call_args.kwargs['source'] == 'collision/robot/self'
# END no simulator object pose, contact query, or skill-specific exception is used.
