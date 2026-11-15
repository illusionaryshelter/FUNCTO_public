"""Physical coordinate counterexamples through the actual task compiler.

Analytic source knowledge and explicit current rigid relationships establish
the frames. No simulator state, collision mesh or measured Holding label is
used by these CPU planning-input contracts.
"""
from copy import deepcopy

import numpy as np

from manipulation.planning_request import bind, bind_task_inputs
from manipulation.representation import build_representation, generate_reference
from manipulation.program import _current_model
from test_manipulation_planning_problem import manipulation_input, on_input, prismatic_input, pose


def source_representation(program):
    demo = program['demo']
    frames, times = np.asarray(demo['frame_indices']), np.asarray(demo['timestamps'])
    event, = demo['functional_events']
    parameters = dict(kind='manipulation', frame_indices=frames, timestamps=times,
        interaction_time=times[np.flatnonzero(frames == event['frame_index'])[0]],
        release_time=times[np.flatnonzero(frames == demo['release_frame'])[0]],
        events=[{**deepcopy(event), 'timestamp_s': times[np.flatnonzero(frames == event['frame_index'])[0]]}])
    if event.get('constraint_kind') != 'joint_coordinate':
        parameters.update(T_world_carrier=demo['T_reference_carrier'],
            T_world_reference=np.repeat(np.eye(4)[None], len(frames), axis=0), L_R=np.eye(4))
    return build_representation(**parameters)


def request(spec, *, functional_frame, chart, actor_chart=np.eye(4), contacts=None, materials=None):
    original = deepcopy(spec['scene'])
    current = deepcopy(original)
    current.update(carrier_id='current_carrier')
    current['initial_pose'] = np.asarray(original['initial_pose']) @ np.linalg.inv(actor_chart)
    current['holding'] = dict(T_carrier_eef=actor_chart @ original['holding']['T_carrier_eef'],
                              source='Declared rigid robot-owner relationship')
    current['eef_pose'] = current['initial_pose'] @ current['holding']['T_carrier_eef']
    for event in spec['program']['demo']['functional_events']:
        if 'carrier_frame' in event and 'carrier_frame' in current['functional_bindings'][event['event_id']]:
            current['functional_bindings'][event['event_id']]['carrier_frame'] = actor_chart @ event['carrier_frame']
    # BEGIN the generic encoder record carries the public observation clock.
    feedback = dict(joint_names=['a', 'b', 'c'], arm_joint_names=['a', 'b', 'c'],
        start_q=[.1, .2, .3], start_dq=[.01, -.02, .03],
        tool_link='declared_tool', reference_frame='world', base_pose=np.eye(4),
        tool_transform=np.eye(4), tool_id=current['tool_id'], tool_frame=current['tool_frame'],
        eef_pose=current['eef_pose'], feedback_id='analytic-encoder-observation', timestamp_s=4.)
    # END task coordinates and time metadata use the same feedback contract.
    binding = dict(owner_id='current_reference', owner_role='reference',
        T_world_owner=chart, L_source_owner_region=np.eye(4),
        T_task_owner_source_owner=np.eye(4), scene=current,
        current_material_contacts={'contacts': [] if materials is None else materials},
        other_contact_owner_bindings={original['carrier_id']: dict(owner_id='current_carrier',
                                                                 T_current_source=actor_chart)},
        source='Calibrated observation chart and grounded functional frame')
    return dict(skill_id='current_operation', feedback=feedback, pose_objectives=[],
        task=dict(program=deepcopy(spec['program']), scene=original,
            contact_bindings={'contacts': [] if contacts is None else contacts, 'source': 'Source task contact'}),
        task_binding=binding)


def hypothesis(spec, req, functional_frame, *, scalar_coordinate=None):
    current = req['task_binding']['scene']
    rep = source_representation(spec['program'])
    A = np.linalg.inv(current['holding']['T_carrier_eef'])
    if scalar_coordinate is None:
        reference = generate_reference(rep, T_world_function=functional_frame,
            J=np.linalg.inv(req['task_binding']['other_contact_owner_bindings'][spec['scene']['carrier_id']]['T_current_source']),
            A=A, T_world_eef=req['feedback']['eef_pose'])
    else:
        model, _ = _current_model(current['motion_model'])
        reference = generate_reference(rep, motion_model=model,
            source_coordinates=spec['program']['demo']['motion_coordinates'],
            current_coordinates=scalar_coordinate, A=A)
    return dict(id='reference_prediction', reference=reference, T_world_function=functional_frame,
        T_world_tool=None, tool_reference_anchor_world=None, source='Analytic source and current model')


def test_full_functional_frame_updates_owner_and_separates_actor_from_patch_coordinates():
    spec = manipulation_input()
    chart = pose((.4, -.2, .1), (.1, .2, -.3))
    frame = pose((.3, .2, .4), (-.2, .1, .4))
    actor_chart = pose((.02, -.03, .01), (.3, -.2, .5))
    contacts = [dict(rule_id='retained_edge_contact', actor=spec['scene']['carrier_id'],
        target_owner=spec['scene']['reference_id'], interval=[0., 1.], material_id='source_edge',
        patch_origin=[.01, -.02, 0.], patch_axes=np.eye(3), patch_half_size=[.1, .1],
        gap_bounds=[-.001, .002], normal_tolerance_rad=.1,
        modeling_allowance_m=0., geometry_error_m=.001, source='Source functional contact')]
    materials = [dict(actor=spec['scene']['carrier_id'], material_id='source_edge', feature_kind='edge',
        feature_points=[[0., 0., 0.], [.02, 0., 0.]], feature_normal=[0., 0., -1.],
        feature_joint_translation=None, source='Source material edge')]
    req = request(spec, functional_frame=frame, chart=chart, actor_chart=actor_chart,
                  contacts=contacts, materials=materials)
    req['pose_objectives'] = [dict(goal_id=['pose_events', 0], role='required',
        scales={'position': 1e-7, 'rotation': 1e-7}, scale_source='Original analytic event precision')]
    hyp = hypothesis(spec, req, frame)
    task = bind_task_inputs(hyp, req)
    problem, _ = bind(hyp, req)
    goal, = problem.pose_events
    event, = spec['program']['demo']['functional_events']
    source_index = list(spec['program']['demo']['frame_indices']).index(event['frame_index'])
    expected_target = frame @ spec['program']['demo']['T_reference_carrier'][source_index] @ event['carrier_frame']
    np.testing.assert_allclose(goal.target, expected_target, atol=1e-12)
    assert goal.actual_relation['function_owner_id'] == 'current_carrier'
    assert goal.actual_relation['target_owner_id'] == 'current_reference'
    assert (goal.position_tolerance_m, goal.rotation_tolerance_rad) == (
        event['position_tolerance_m'], event['rotation_tolerance_rad'])
    bound = task['scene']['functional_bindings'][event['event_id']]
    assert bound['source'] == spec['scene']['functional_bindings'][event['event_id']]['source']
    assert bound['carrier_id'] == 'current_carrier' and bound['reference_id'] == 'current_reference'
    contact, = task['contact_bindings']['contacts']
    assert contact['actor'] == 'current_carrier' and contact['target_owner'] == 'current_reference'
    # Recharting the actor leaves each source material point at the same world
    # position, while the independent target patch moves with grounded F.
    initial = task['scene']['initial_pose']
    world_feature = contact['feature_points'] @ initial[:3, :3].T + initial[:3, 3]
    np.testing.assert_allclose(world_feature, materials[0]['feature_points'], atol=1e-15)
    world_patch = chart[:3, :3] @ contact['patch_origin']+chart[:3, 3]
    np.testing.assert_allclose(world_patch, frame[:3, :3] @ contacts[0]['patch_origin']+frame[:3, 3], atol=1e-15)
    np.testing.assert_array_equal(contact['gap_bounds'], contacts[0]['gap_bounds'])


def test_scalar_coordinate_and_original_model_reach_compiler_without_frame_goals():
    spec = prismatic_input()
    req = request(spec, functional_frame=np.eye(4), chart=np.eye(4))
    hyp = hypothesis(spec, req, np.eye(4), scalar_coordinate=[.1])
    problem, _ = bind(hyp, req)
    goal, = problem.coordinate_events
    assert goal.target_coordinate == .14
    assert goal.tolerance == 1e-7
    assert not problem.pose_events
    assert len(problem.mechanisms) == 1
    assert problem.events[0].event_id == 'function'
    # The current named model and signed coordinate generate the reference
    # through scalar FK before the fixed tool attachment is composed.
    np.testing.assert_allclose(hyp['reference']['eef_path'].sample_numpy(0.), req['feedback']['eef_pose'], atol=1e-12)


def test_finite_support_geometry_and_allowed_band_reach_original_support_requirement():
    spec = on_input()
    req = request(spec, functional_frame=np.eye(4), chart=np.eye(4))
    hyp = hypothesis(spec, req, np.eye(4))
    problem, _ = bind(hyp, req)
    support, = problem.supports
    original = spec['scene']['functional_bindings']['function']['on_geometry']
    np.testing.assert_array_equal(support.contact_points, original['carrier_region_points'])
    np.testing.assert_array_equal(support.outward_normal, original['carrier_outward_normal'])
    assert support.gap_tolerance == original['gap_tolerance']
    assert support.normal_tolerance == original['normal_tolerance']
    np.testing.assert_array_equal(support.surface.boundary, spec['scene']['support']['boundary'])
    assert not problem.pose_events
