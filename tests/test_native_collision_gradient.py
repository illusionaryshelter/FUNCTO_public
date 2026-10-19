"""Native penetration derivatives and permissions, with actual Coal and pin."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from manipulation.online_geometry import FrozenPrimitives
from manipulation.online_native_confirmation import OnlineCollisionConfirmation
from test_online_native_confirmation import _setup, _box, _candidate_evidence, _payload


def fixture(tmp_path):
    query = FrozenPrimitives(primitives=[{'kind': 'cuboid', 'dimensions': [.02, .024, .028],
        'pose': [0., 0., 0., 1., 0., 0., 0.]}], owner_ids=('part',), source='calibrated test box')
    checker, scene = _setup(tmp_path, query=query, native=_box(half=(.02, .01, .015)))
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler('z', .37).as_matrix()
    validator = SimpleNamespace(checker=checker, joint_names=['slide'], T_reference_base=base)
    return validator, scene


def evaluate(validator, scene, variables):
    evidence = _candidate_evidence([[[.1]]])
    evidence['joint_position'] = variables[0].reshape(1, 1, 1)
    p = torch.stack((variables[1], variables[2], variables.new_tensor(0.))).reshape(1, 1, 3)
    yaw = variables[3]
    quat = torch.stack((torch.cos(yaw/2), yaw*0, yaw*0, torch.sin(yaw/2))).reshape(1, 1, 4)
    evidence['owner_poses']['part'] = p, quat
    original = variables.new_full((1, 1), .1)
    unknown = variables.new_full((1, 1), .014)
    adapter = OnlineCollisionConfirmation(validator, scene)
    costs = adapter({'collision/robot/part': original, 'unknown/forbidden/robot': unknown}, evidence)
    assert costs['unknown/forbidden/robot'] is unknown
    return costs['collision/robot/part'].sum(), adapter.last_report


def test_penetration_joint_base_rotation_and_moving_target_match_finite_difference(tmp_path):
    validator, scene = fixture(tmp_path)
    variables = torch.tensor([.025, .001, .002, -.17], dtype=torch.float64, requires_grad=True)
    value, report = evaluate(validator, scene, variables)
    assert 0 < value < .1
    assert report['states'][0]['pairs'][0]['status'] == 'collision'
    gradient, = torch.autograd.grad(value, variables)
    step = 1e-6
    differences = []
    for i in range(len(variables)):
        direction = torch.zeros_like(variables); direction[i] = step
        plus, _ = evaluate(validator, scene, variables.detach()+direction)
        minus, _ = evaluate(validator, scene, variables.detach()-direction)
        differences.append((plus-minus)/(2*step))
    torch.testing.assert_close(gradient, torch.stack(differences), rtol=2e-4, atol=2e-6)


def test_proxy_clear_has_zero_cost_without_clearing_unknown(tmp_path):
    validator, scene = fixture(tmp_path)
    value, report = evaluate(validator, scene, torch.tensor([.2, 0., 0., 0.], dtype=torch.float64))
    assert value == 0
    assert report['states'][0]['cleared_targets'] == ['part']


def test_component_order_does_not_choose_first_shallow_penetration(tmp_path):
    from manipulation.online_native_confirmation import NativeCollisionConfirmation
    from test_online_native_confirmation import _confirm
    primitives = [dict(kind='cuboid', dimensions=[.02]*3, pose=[x, 0., 0., 1., 0., 0., 0.])
                  for x in (.025, .015)]
    minimum = []
    for rows in (primitives, primitives[::-1]):
        query = FrozenPrimitives(primitives=rows, owner_ids=('part',), source='two calibrated solids')
        checker, scene = _setup(tmp_path, query=query, native=_box(half=(.02, .01, .015)))
        report = _confirm(NativeCollisionConfirmation(checker, scene), checker)
        pair, = report['pairs']
        assert pair['coal_queries'] == 2
        minimum.append(pair['minimum_queried_distance_m'])
    assert minimum[0] == pytest.approx(-.015, abs=1e-6)
    assert minimum[0] == pytest.approx(minimum[1], abs=1e-8)


def test_robot_witness_jacobian_keeps_revolute_offset_and_named_order(tmp_path):
    from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker
    path = tmp_path/'chain.urdf'
    path.write_text('''<robot name="chain"><link name="base"/><link name="turn"/><link name="tool"/>
      <joint name="yaw" type="revolute"><parent link="base"/><child link="turn"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="1"/></joint>
      <joint name="slide" type="prismatic"><parent link="turn"/><child link="tool"/>
        <origin xyz="0.2 0 0"/><axis xyz="1 0 0"/>
        <limit lower="-0.1" upper="0.1" effort="10" velocity="1"/></joint></robot>''')
    local = np.eye(4); local[:3, 3] = [.03, .02, .01]
    checker = RobotGeometryChecker(path, base_frame='base', geometries=[
        NativePrimitive('body', 'tool', 'box', (.04, .03, .02), local, 'declared test CAD')])
    base = np.eye(4); base[:3, :3] = Rotation.from_euler('xyz', [.2, -.4, .7]).as_matrix()
    base[:3, 3] = [.3, -.5, .2]
    names, q = ['slide', 'yaw'], np.array([.025, .63])
    point_local = np.array([.01, -.015, .01])
    def query(values):
        state = checker.query(joint_names=names, joint_positions=values,
            T_reference_base=base, pairs=())
        pose = np.asarray(state['geometry_poses']['body'])
        return state, pose[:3, :3] @ point_local+pose[:3, 3]
    state, point = query(q)
    jacobian, = checker.geometry_point_jacobians(state, [('body', point)], base)
    expected = np.empty((3, 2))
    for i in range(2):
        delta = np.zeros(2); delta[i] = 1e-6
        expected[:, i] = (query(q+delta)[1]-query(q-delta)[1])/(2e-6)
    np.testing.assert_allclose(jacobian, expected, rtol=1e-8, atol=1e-9)


def test_moving_payload_and_target_translation_rotation_match_finite_difference(tmp_path):
    """Both owner poses move; independent offset solids contribute once each."""
    validator, scene = fixture(tmp_path)
    _payload(scene, two_boxes=True)

    def objective(values):
        evidence = _candidate_evidence([[[.1, .1]]])
        evidence['targets'] = {'collision/payload/part': evidence['targets']['collision/robot/part']}
        evidence['joint_position'], evidence['joint_names'] = None, None
        for name, start in (('payload', 0), ('part', 3)):
            x, y, yaw = values[start:start+3]
            position = torch.stack((x, y, yaw*0)).reshape(1, 1, 3)
            quaternion = torch.stack((torch.cos(yaw/2), yaw*0, yaw*0,
                                      torch.sin(yaw/2))).reshape(1, 1, 4)
            evidence['owner_poses'][name] = position, quaternion
        adapter = OnlineCollisionConfirmation(validator, scene)
        return adapter({'collision/payload/part': values.new_full((1, 1), .2)},
                       evidence)['collision/payload/part'].sum()

    values = torch.tensor([.006, .001, .23, .0001, .0003, -.17],
                          dtype=torch.float64, requires_grad=True)
    cost = objective(values)
    assert 0 < cost < .2
    gradient, = torch.autograd.grad(cost, values)
    differences = []
    for index in range(len(values)):
        delta = torch.zeros_like(values); delta[index] = 1e-7
        differences.append((objective(values.detach()+delta)-objective(values.detach()-delta))/2e-7)
    torch.testing.assert_close(gradient, torch.stack(differences), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(gradient[:2], -gradient[3:5])


@pytest.mark.parametrize('fault', ['nonfinite_normal', 'nonfinite_witness', 'inconsistent_witness'])
def test_invalid_penetrating_native_feature_remains_unresolved(tmp_path, monkeypatch, fault):
    import manipulation.online_native_confirmation as confirmation
    from test_online_native_confirmation import _confirm
    validator, scene = fixture(tmp_path)
    original = confirmation._direct_distances

    def corrupt(*args):
        for result, identity, count in original(*args):
            first, second = result.getNearestPoint1(), result.getNearestPoint2()
            normal = np.asarray(result.normal).copy()
            if fault == 'nonfinite_normal': normal[0] = np.nan
            if fault == 'nonfinite_witness': first[0] = np.nan
            if fault == 'inconsistent_witness': second[0] += .01
            yield SimpleNamespace(min_distance=result.min_distance, normal=normal,
                getNearestPoint1=lambda: first, getNearestPoint2=lambda: second), identity, count
    monkeypatch.setattr(confirmation, '_direct_distances', corrupt)
    report = _confirm(confirmation.NativeCollisionConfirmation(validator.checker, scene), validator.checker)
    assert report['unresolved_targets'] == ['part']
    assert report['pairs'][0]['reason'] == 'invalid_native_distance_feature'


def test_observed_batches_preserve_full_minimum_witnesses_and_boundaries():
    import coal
    from manipulation.online_native_confirmation import _observed_distance_minima
    from observed_geometry_fixtures import point_source
    # A later block contains the deepest point. A positive prefix must not
    # hide either an earlier penetration or a later deeper feature.
    points = np.column_stack((np.linspace(.035, .015, 4101), np.zeros((4101, 2))))
    geometry = point_source(points)
    shape, transform = coal.Box(.04, .04, .04), coal.Transform3s()
    ids = np.arange(geometry.component_count, dtype=np.intp)
    batched = list(_observed_distance_minima(geometry, ids, shape, transform, 1e-6, None))
    assert len(batched) == 2 and sum(row[2] for row in batched) == len(ids)
    best, identity, _ = min(batched, key=lambda row: row[0].min_distance)
    request = coal.DistanceRequest(); request.enable_signed_distance = True
    reference = []
    for index in ids:
        _, _, target, pose, _ = geometry.collision_component(int(index))
        result = coal.DistanceResult()
        coal.distance(shape, transform, target, coal.Transform3s(pose[:3, :3], pose[:3, 3]), request, result)
        reference.append(result)
    expected_index = min(range(len(reference)), key=lambda i: reference[i].min_distance)
    assert identity['source_component_id'] == expected_index
    def values(result):
        return np.r_[result.min_distance, result.getNearestPoint1(), result.getNearestPoint2(), result.normal]
    assert values(best).tobytes() == values(reference[expected_index]).tobytes()
    # A new actor pose must query again; only source shapes/transforms persist.
    moved = coal.Transform3s(np.eye(3), np.array([.1, 0., 0.]))
    later = list(_observed_distance_minima(geometry, ids, shape, moved, 1e-6, None))
    assert min(row[0].min_distance for row in later) > 0
