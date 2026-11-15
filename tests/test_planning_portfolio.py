"""CPU contracts for independent planning inputs and fixed curve preferences."""
from copy import deepcopy
import json
from math import radians

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from manipulation.planning_portfolio import save_inputs, load_inputs, _worker, plan_grounding_hypotheses
from manipulation.planning_problem import PosePath
from manipulation.planning_request import bind_task_inputs, reexpress_contact_bindings
from manipulation.trajectory_metrics import joint_metrics
from manipulation.trajectory_selection import select


def pose(x=0., angle=0.):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler('z', angle).as_matrix()
    value[0, 3] = x
    return value


def case(identity, accepted, *, objective=2.):
    return dict(id=identity, accepted=accepted, original_objective=objective,
        joint={'position': {'minimum_margin': {'margin': .3}}},
        robot_self={'minimum': {'evidence': {'signed_distance_m': .002}}},
        tool_movement={'rotation_travel_deg': 40.},
        pose_accuracy={'status': 'warning'}, source_report=identity+'/report.json',
        source_curve=identity+'/trajectory.npz', metrics_status='completed')


def calibration():
    return dict(criteria=[dict(id=identity, unit=unit, cost_sign=sign,
        reference_value=0., improvement_interval=interval, weight=.25)
        for identity, unit, sign, interval in (
            ('original_objective', 'dimensionless', 1, 2.),
            ('minimum_joint_position_margin_rad', 'rad', -1, .2),
            ('minimum_required_robot_pair_clearance_m', 'm', -1, .001),
            ('tool_rotation_length_rad', 'rad', 1, radians(30.)))])


def choose(cases):
    return select(dict(cases=cases, physical_status='missing_not_executed'), calibration())


def test_fixed_scores_do_not_depend_on_order_subset_or_pose_warning():
    a, b = case('first', True), case('second', True, objective=3.)
    both = choose([b, a])
    assert both['preferred_id'] == 'first'
    single = choose([a])
    assert single['scores'][0]['score'] == both['scores'][1]['score']
    assert choose([a, b])['preferred_id'] == both['preferred_id']
    assert choose([case('refused', False), a])['preferred_id'] == 'first'
    assert choose([case('refused', False)])['status'] == 'no_accepted_plan'
    tie = choose([a, {**deepcopy(a), 'id': 'same_cost'}])
    assert tie['status'] == 'tie_requires_user_choice'
    assert tie['best_candidate_ids'] == ['first', 'same_cost']


def test_missing_metrics_preserve_formal_acceptance_and_incomplete_comparison():
    a = case('first', True)
    b = {**case('unmeasured', True), 'metrics_status': 'measurement_failed'}
    result = choose([a, b])
    assert result['status'] == 'incomplete_metrics'
    assert result['preferred_id'] is None
    assert result['accepted_with_missing_metrics'][0]['accepted']


def test_file_roundtrip_preserves_parent_arc_and_original_cpu_array_bytes(tmp_path):
    parent = PosePath([0., 1.], np.array([pose(), pose(angle=np.pi/2)]), 'parent')
    path = PosePath([0., 1.], np.array([pose(.7), pose(.7)]), 'source', parent=parent)
    image = np.arange(96, dtype=np.uint8).reshape(4, 8, 3)
    data = dict(hypothesis={'reference': {'eef_path': path}, 'id': 'stable'},
                request={'observations': {'camera': {'image': image}},
                         'events': [{'event_id': 'non_node', 'timestamp_s': 1.37}]})
    manifest = save_inputs(tmp_path/'inputs', data)
    restored = load_inputs(manifest)
    samples = np.array([0., .19, .63, 1.])
    assert restored['request']['observations']['camera']['image'].tobytes() == image.tobytes()
    assert restored['request']['events'] == data['request']['events']
    assert restored['hypothesis']['reference']['eef_path'].sample_numpy(samples).tobytes() == path.sample_numpy(samples).tobytes()
    middle = restored['hypothesis']['reference']['eef_path'].sample_numpy(.5)
    np.testing.assert_allclose(middle[:2, 3], [.7/np.sqrt(2), .7/np.sqrt(2)], atol=1e-15)


def test_cubic_margin_uses_interior_extremum_and_exact_physical_integral():
    # q(t)=4*t*(1-t), so its maximum is one at t=.5 and integral(q²)=8/15.
    curve = CubicHermiteSpline([0., 1.], [[0.], [0.]], [[4.], [-4.]])
    bounds = dict(joint_names=['arm'], position=[[-2.], [2.]],
                  velocity=[[-5.], [5.]], acceleration=[[-10.], [10.]],
                  jerk=[[-100.], [100.]])
    result = joint_metrics(curve, ['arm'], bounds)
    assert result['position']['minimum_margin']['margin'] == 1.
    assert result['position']['minimum_margin']['margin_time_s'] == .5
    assert result['position']['mean_squared_joint_range_normalized'] == pytest.approx(1/30)


def test_task_binding_keeps_f_distinct_from_tool_anchor_and_source_allowances():
    contact = dict(actor='robot', material_id='pad', target_owner='source', patch_origin=[.1, .2, .3],
        patch_axes=np.eye(3), patch_half_size=[.01, .02], gap_bounds=[-.001, .002],
        minimum_thickness_m=.003, normal_tolerance_rad=.05, rule_id='finite_patch',
        interval=[0., 1.], modeling_allowance_m=0., geometry_error_m=.001)
    raw = dict(contacts=[contact], material_faces={}, source='demonstrated patch')
    material = dict(actor='robot', material_id='pad', feature_kind='edge',
        feature_points=[[0., 0., 0.], [.02, 0., 0.]], source='official robot feature',
        feature_normal=[0., 0., 1.], feature_joint_translation=None, robot_body='finger')
    F, B, E, chart, L = pose(.4, .3), pose(2., -.2), pose(.6, .8), pose(.1, -.1), pose(.05, .2)
    original_mapping = pose(.02, .1)
    goal = dict(T_carrier_eef=pose(), position_tolerance_m=.002, rotation_tolerance_rad=.03,
                source='explicit task')
    request = dict(task=dict(program={'skill_kind': 'grasp'},
        scene={'carrier_id': 'source', 'region_transform': original_mapping, 'grasp_goal': goal},
        contact_bindings=raw), task_binding=dict(owner_id='current', owner_role='carrier',
        T_world_owner=chart, L_source_owner_region=L, T_task_owner_source_owner=original_mapping,
        current_material_contacts={'contacts': [material]},
        scene={}, source='current calibrated observation'),
        feedback=dict(reference_frame='world', eef_pose=pose(), tool_id='tool', tool_frame='tcp'))
    hypothesis = dict(id='stable', T_world_function=F, T_world_tool=E,
                      tool_reference_anchor_world=B)
    task = bind_task_inputs(hypothesis, request)
    mapping = np.linalg.inv(chart) @ F @ np.linalg.inv(L)
    relocation = mapping @ np.linalg.inv(original_mapping)
    assert task['scene']['region_transform'].tobytes() == mapping.tobytes()
    assert task['scene']['grasp_goal']['T_carrier_eef'].tobytes() == (np.linalg.inv(chart) @ E).tobytes()
    assert task['contact_bindings']['contacts'][0]['patch_origin'].tobytes() == (
        relocation[:3, :3] @ contact['patch_origin']+relocation[:3, 3]).tobytes()
    assert task['scene']['grasp_goal']['position_tolerance_m'] == .002
    np.testing.assert_array_equal(task['contact_bindings']['contacts'][0]['gap_bounds'], [-.001, .002])
    assert 'grasp_reference_prior' not in task['scene']
    assert raw['contacts'][0]['minimum_thickness_m'] == .003


def test_worker_import_or_input_failure_keeps_actual_stage_without_curve(tmp_path):
    directory = tmp_path/'worker'/'inputs'
    directory.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        _worker(directory/'missing-inputs.json')
    state = json.loads((directory.parent/'worker-result.json').read_text())
    assert state['status'] == 'failed' and state['stage'] == 'imports'
    assert state['source_curve'] is None
    assert 'FileNotFoundError' in state['error']
    assert not (directory.parent/'planning').exists()


def test_manipulation_keeps_declared_attachment_model_and_scalar_event_inputs():
    # A scalar event needs its model coordinate, not a manufactured full frame.
    current = dict(carrier_id='current_carrier', reference_id='current_reference',
        initial_pose=pose(.2), holding={'T_carrier_eef': pose(.05), 'source': 'declared model'},
        motion_model={'kind': 'revolute', 'source': 'observed signed coordinate model'},
        functional_bindings={'event': {'current_coordinates': .6, 'source': 'declared coordinate'}})
    request = dict(task=dict(program={'skill_kind': 'manipulation',
        'demo': {'functional_events': [{'event_id': 'event', 'constraint_kind': 'joint_coordinate'}]}},
        scene={'reference_id': 'source_reference'}, contact_bindings={'contacts': [], 'source': 'source'}),
        task_binding=dict(owner_id='current_reference', owner_role='reference',
            T_world_owner=pose(.7), L_source_owner_region=np.eye(4),
            T_task_owner_source_owner=np.eye(4), current_material_contacts={'contacts': []},
            scene=current, source='observed task frame'),
        feedback=dict(reference_frame='world', eef_pose=pose(.25), tool_id='tool', tool_frame='tcp'))
    task = bind_task_inputs(dict(id='stable', T_world_function=pose(.9)), request)
    np.testing.assert_array_equal(task['scene']['holding']['T_carrier_eef'],
                                  current['holding']['T_carrier_eef'])
    assert task['scene']['holding']['source'] == current['holding']['source']
    assert task['scene']['motion_model'] == current['motion_model']
    bound = task['scene']['functional_bindings']['event']
    assert bound['current_coordinates'] == current['functional_bindings']['event']['current_coordinates']
    assert bound['source'] == current['functional_bindings']['event']['source']
    assert (bound['carrier_id'], bound['reference_id']) == ('current_carrier', 'current_reference')
    np.testing.assert_array_equal(task['scene']['initial_pose'], current['initial_pose'])


def test_contact_material_identity_includes_actor_and_keeps_nonrobot_feature_role():
    declaration = dict(target_owner='destination', patch_origin=[0., 0., 0.],
        patch_axes=np.eye(3), interval=[0., 1.], patch_half_size=[.1, .1],
        gap_bounds=[-.001, .001], normal_tolerance_rad=.1,
        modeling_allowance_m=0., geometry_error_m=.001)
    sources = dict(source='source task', contacts=[dict(declaration,
        actor=actor, material_id='shared', rule_id=actor+'_contact') for actor in ('robot', 'payload')])
    features = dict(contacts=[dict(actor=actor, material_id='shared', feature_kind='edge',
        feature_points=[[value, 0., 0.], [value+.1, 0., 0.]], source='original material feature',
        feature_normal=[0., 0., 1.],
        feature_joint_translation=None, **({'robot_body': 'official_finger'} if actor == 'robot' else {}))
        for actor, value in [('robot', .1), ('payload', .2)]])
    result = reexpress_contact_bindings(sources, features,
        {'destination': {'owner_id': 'current', 'T_current_source': pose(.3)},
         'payload': {'owner_id': 'carried', 'T_current_source': pose(1., np.pi/2)}}, source='bound source')
    np.testing.assert_array_equal(result['contacts'][0]['feature_points'], [[.1, 0., 0.], [.2, 0., 0.]])
    np.testing.assert_allclose(result['contacts'][1]['feature_points'], [[1., .2, 0.], [1., .3, 0.]], atol=1e-15)
    assert result['contacts'][1]['actor'] == 'carried'
    assert result['contacts'][1]['robot_body'] is None
    np.testing.assert_array_equal(result['contacts'][1]['patch_origin'], [.3, 0., 0.])


def test_launch_failure_retains_every_hypothesis_and_never_fabricates_a_curve(tmp_path, monkeypatch):
    starts = []
    def fail_start(command, **kwargs):
        starts.append(command)
        raise FileNotFoundError('The declared worker interpreter is unavailable')
    monkeypatch.setattr('manipulation.planning_portfolio.subprocess.Popen', fail_start)
    path = PosePath([0., 1.], np.array([pose(), pose(.1)]), 'declared nominal motion')
    candidates = [dict(id=identity, reference={'eef_path': path}, T_world_function=pose(),
        T_world_tool=pose(.1), tool_reference_anchor_world=pose(), source='declared source')
        for identity in ('first', 'second')]
    request = dict(output_directory=str(tmp_path/'portfolio'), calibration=calibration(),
        resources=dict(max_host_rss_bytes=12*1024**3, max_device_used_bytes=10*1024**3))
    result = plan_grounding_hypotheses({'reference_hypotheses': candidates}, request)
    assert len(starts) == 2
    assert result['selection']['status'] == 'no_accepted_plan'
    assert result['comparison']['incomplete_worker_ids'] == ['first', 'second']
    for row in result['workers']:
        assert row['result']['stage'] == 'launch'
        assert 'FileNotFoundError' in row['result']['error']
        assert row['result']['source_curve'] is None
        assert row['process_returncode'] is None
        assert not (tmp_path/'portfolio'/row['id']/'planning').exists()
