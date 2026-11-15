"""Grounding handoff counterexamples through the actual public consumer.

Region registration and grasp inference are external computations supplied at
their public boundaries. Complete reference generation, physical-time entry
binding, typed failure capture and result/HTML archival execute their real
production implementations on analytic calibrated observations.
"""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from manipulation.cross_instance_grasp import generate_and_rank
from manipulation.grounding_records import save
from manipulation.planning_problem import PosePath
from manipulation.representation import build_representation
from manipulation.skill_grounding import ground_prepared_skill
from test_manipulation_skill import source_inputs


def pose(position=(0., 0., 0.), angles=(0., 0., 0.)):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('xyz', angles).as_matrix()
    matrix[:3, 3] = position
    return matrix


def source_grasp(time):
    progress = (time-4.)/1.6
    return pose((.05*progress, -.03*progress, .2-.1*progress), (0., 0., .8*progress))


def inputs(kind='grasp'):
    metadata, source, _ = source_inputs()
    metadata.update(skill_id='explicit_source_segment', owner_role='carrier' if kind == 'grasp' else 'reference',
                    tool_id='fixed_tool', tool_frame='fixed_tcp')
    times, frames = np.array([4., 4.3, 5.6]), np.array([17, 20, 25])
    events = [dict(event_id='interaction', timestamp_s=5.1, source='Source functional event'),
              dict(event_id='release', timestamp_s=5.45, source='Source release event'),
              dict(event_id='later_source_evidence', timestamp_s=5.52, source='Complete source recording')]
    if kind == 'grasp':
        body = np.repeat(pose((0., 0., 1.))[None], len(times), axis=0)
        tools = np.array([body[index] @ source_grasp(time) for index, time in enumerate(times)])
        rep = build_representation(kind=kind, frame_indices=frames, timestamps=times,
            T_world_carrier=body, T_world_eef=tools, L_G=np.eye(4),
            approach_time=4., interaction_time=5.1, release_time=5.45, events=events)
        source.update(T_world_body_at_image=body[0], L_body_region=np.eye(4))
    else:
        progress = (times-times[0])/(times[-1]-times[0])
        body = np.array([pose((.05*u, .03*u, 1.+.02*u), (.15*u, -.1*u, .3+.6*u)) for u in progress])
        references = np.repeat(pose((0., 0., .9))[None], len(times), axis=0)
        L = pose((0., 0., .1), (.1, 0., -.2))
        tools = np.array([body[index] @ pose((.01*u, 0., .15), (0., -.2*u, 0.))
                          for index, u in enumerate(progress)])
        rep = build_representation(kind=kind, frame_indices=frames, timestamps=times,
            T_world_carrier=body, T_world_reference=references, T_world_eef=tools, L_R=L,
            approach_time=4., interaction_time=5.1, release_time=5.45, events=events)
        source.update(T_world_body_at_image=references[0], L_body_region=L)
    skill = dict(metadata=metadata, source=source, representation=rep)
    views = []
    for index, identity in enumerate(('near_camera', 'far_camera')):
        camera = pose((.04*index, 0., 0.))
        owner = np.asarray(source['observation']['object_mask']).copy()
        bank = deepcopy(source['observation'])
        bank['T_world_camera_cv'] = camera.copy()
        background = np.zeros(owner.shape, bool)
        background[0, 0] = True
        views.append(dict(id=identity, image=source['image'].copy(),
            depth_m=np.asarray(source['depth_m']).copy(), K=np.asarray(bank['K']).copy(),
            T_world_camera_cv=camera, timestamp_s=11., observation=bank,
            owner_observation=bank, owner_mask=owner,
            owner_evidence=dict(scope='selected_entity_observation', owner_id='current_owner',
                                source='Explicit whole-entity observation'),
            background_mask=background, crop_xyxy=np.asarray(source['crop_xyxy']).copy(),
            feature=np.ones((1, 3, 4, 4), np.float32), missing_reasons=[]))
    selected = dict(selection_scope='entity', same_entity=True,
        same_entity_reason='Two explicitly associated calibrated observations',
        source_target_relation='same_category_different_instance', relation_reason='Declared fixture identity relation',
        views={view['id']: dict(status='selected', candidate_ids=[index+7], reason='Explicit semantic target',
            regions=[dict(candidate_id=index+7, mask=view['owner_mask'].copy())])
            for index, view in enumerate(views)})
    binding_id = 'selected_binding'
    prepared = dict(bindings={binding_id: dict(strategy='cross_instance', views=views, selection=selected)})
    feedback = dict(tool_id=metadata['tool_id'], tool_frame=metadata['tool_frame'], reference_frame='world',
        eef_pose=pose((.05, -.1, 1.4), (0., 0., -.2)),
        eef_twist=np.array([.01, -.02, .03, -.04, .02, .01]),
        feedback_id='current_encoder_snapshot', timestamp_s=11.2)
    request = dict(skill=skill, reference_binding_id=binding_id,
        bindings={binding_id: dict(skill=skill, views=views, owner_id='current_owner',
                                  owner_source='Explicit visible geometry owner')}, feedback=feedback,
        capture_feedback={**feedback, 'feedback_id': 'image_capture_snapshot', 'timestamp_s': 11.},
        context=dict(skill_id=metadata['skill_id'], prerequisite=dict(status='not_required'),
                     start_time_s=11.2, capture_time_s=11.),
        same_state_source='Explicit calibrated capture', region_parameters={}, grasp_parameters={})
    return request, prepared


def region_result(request, prepared, functional_frame):
    bank = request['skill']['source']['observation']
    points = bank['xyz'][bank['region_mask']]
    source_frame = (request['skill']['source']['T_world_body_at_image']
                    @ request['skill']['source']['L_body_region'])
    local = (points-source_frame[:3, 3]) @ source_frame[:3, :3]
    mapped = local @ functional_frame[:3, :3].T+functional_frame[:3, 3]
    views = prepared['bindings'][request['reference_binding_id']]['views']
    return dict(status='completed', region_frame=dict(T_world_region=functional_frame),
        arrays=dict(source_points_world_m=points, target_points_world_m=mapped,
                    mapped_points_world_m=mapped.copy(), matching_target_indices=np.arange(len(mapped))),
        views=views, input_context=request['context'])


def test_two_frozen_tool_hypotheses_keep_complete_paths_and_individual_diagnostics(monkeypatch, tmp_path):
    request, prepared = inputs()
    F = pose((.05, .02, 1.), (.1, -.15, .25))
    region = region_result(request, prepared, F)
    monkeypatch.setattr('transfer.cross_instance.pipeline.run_region_transfer', lambda *args, **kwargs: region)
    targets = [pose((.02, 0., 1.15), (0., 0., angle)) for angle in (.1, 1.2)]
    anchors = [target @ np.linalg.inv(source_grasp(5.1)) for target in targets]
    observed_call = []
    def computed_grasps(skill, regional, views, **kwargs):
        observed_call.append(kwargs['feedback'])
        return dict(report={'status': 'computed'}, arrays=regional['arrays'],
            hypotheses=[dict(id=identity, T_world_function=F.copy(), T_world_tool=target,
                tool_reference_anchor_world=anchor, source={'id': identity, 'computation': 'Explicit external grasp result'})
                for identity, target, anchor in zip(('orientation_a', 'orientation_b'), targets, anchors)])
    monkeypatch.setattr('manipulation.cross_instance_grasp.run_cross_instance_grasp', computed_grasps)
    result = ground_prepared_skill(request, prepared)
    assert result['status'] == 'completed'
    assert result['reference_binding_status'] == 'per_hypothesis'
    assert result['nominal_reference_status'] == 'per_hypothesis'
    assert len(observed_call) == 1 and observed_call[0] is request['capture_feedback']
    first, second = result['reference_hypotheses']
    assert first['reference_diagnostics'] is not second['reference_diagnostics']
    assert first['bound_reference'] is not second['bound_reference']
    query_times = np.array([4., 4.17, 4.83, 5.1, 5.45])
    progress = (query_times-4.)/(5.45-4.)
    for row, target, anchor in zip(result['reference_hypotheses'], targets, anchors):
        ref, diagnostic = row['reference'], row['reference_diagnostics']
        np.testing.assert_array_equal(ref['T_world_function'], F)
        np.testing.assert_array_equal(ref['tool_reference_anchor'], anchor)
        expected = np.array([anchor @ source_grasp(time) for time in query_times])
        np.testing.assert_allclose(ref['eef_path'].sample_numpy(progress), expected, atol=1e-12)
        np.testing.assert_allclose(ref['eef_boundaries']['goals'][0]['pose'], target, atol=1e-12)
        np.testing.assert_allclose(row['bound_reference'].sample_numpy(1.1), target, atol=1e-12)
        np.testing.assert_allclose(row['bound_reference'].sample_numpy(0.), request['feedback']['eef_pose'], atol=1e-12)
        np.testing.assert_array_equal(row['bound_reference'].initial_twist, request['feedback']['eef_twist'])
        np.testing.assert_array_equal(ref['frame_indices'], request['skill']['representation']['frame_indices'])
        np.testing.assert_array_equal(ref['timestamps'], request['skill']['representation']['timestamps'])
        assert ref['events'] == request['skill']['representation']['events']
        assert ref['eef_timestamps'][-1] == 5.45
        assert diagnostic['reference_binding'] is row['bound_reference'].diagnostics
    assert not np.allclose(first['reference']['eef_poses'], second['reference']['eef_poses'])
    manifest = save(result, request, tmp_path/'saved')
    assert [row['id'] for row in manifest['reference_hypotheses']] == ['orientation_a', 'orientation_b']
    for row in result['reference_hypotheses']:
        directory = tmp_path/'saved'/'hypotheses'/row['id']
        archived = json.loads((directory/'nominal-reference-paths.json').read_text())
        restored = PosePath.from_record(archived['paths']['eef']['path'])
        assert restored.sample_numpy(progress).tobytes() == row['reference']['eef_path'].sample_numpy(progress).tobytes()
        with np.load(directory/'nominal-reference.npz', allow_pickle=False) as arrays:
            assert arrays['H_g'].tobytes() == request['skill']['representation']['H_g'].tobytes()
        assert (directory/'index.html').is_file()
    archived_report = json.loads((tmp_path/'saved'/'report.json').read_text())
    assert all('regions' not in view for view in archived_report['selection']['views'].values())
    with np.load(tmp_path/'saved'/'selected-regions.npz', allow_pickle=False) as masks:
        for index, view in enumerate(region['views']):
            assert masks[f"{view['id']}__{index+7}"].tobytes() == view['owner_mask'].tobytes()


def test_empty_native_forward_retains_typed_failure_inputs_and_actual_archive(monkeypatch, tmp_path):
    request, prepared = inputs()
    F = pose((.05, .02, 1.))
    region = region_result(request, prepared, F)
    monkeypatch.setattr('transfer.cross_instance.pipeline.run_region_transfer', lambda *args, **kwargs: region)
    points = np.array([[.01, .02, .9], [.02, .01, 1.], [.03, -.01, 1.1],
                       [.04, 0., 1.2], [.05, .03, 1.3]], np.float32)
    steering = np.array([True, False, True, False, True])
    consumed = np.array([4, 0, 2])
    class EmptyForward:
        model_sources = {'forward': 'Explicit CPU model-boundary fixture'}
        geometry_source = 'Declared fixed official tool metadata'
        def predict(self, actual_points, controls, seed):
            np.testing.assert_array_equal(actual_points, points)
            np.testing.assert_array_equal(controls['region_steering'], steering)
            assert seed == 29
            self.prepared_points_camera = actual_points[consumed].copy()
            self.prepared_region_mask = controls['region_steering'][consumed].copy()
            self.sampled_seed_indices = np.array([0, 2], np.int64)
            return np.empty((0, 17), np.float64)
    camera = pose((.2, -.1, .05), (0., 0., .3))
    backend = EmptyForward()
    def empty_grasp(*args, **kwargs):
        return generate_and_rank(points, np.eye(4), dict(rotation_wahba=np.eye(3),
            source_center=np.zeros(3), target_center=np.array([.01, .02, 1.])), backend,
            seed=29, T_world_camera=camera, region_mask=steering, approach_thresh=.2)
    monkeypatch.setattr('manipulation.cross_instance_grasp.run_cross_instance_grasp', empty_grasp)
    result = ground_prepared_skill(request, prepared)
    assert result['status'] == 'failed' and result['failed_stage'] == 'cross_instance_grasp'
    assert result['error_type'] == 'NoGraspCandidates'
    assert result['reference_hypotheses'] == []
    failure = result['grasp_failure']
    expected_cloud = points[consumed] @ camera[:3, :3].T+camera[:3, 3]
    np.testing.assert_array_equal(failure['prepared_points_world'], expected_cloud)
    np.testing.assert_array_equal(failure['model_region_mask'], steering[consumed])
    np.testing.assert_array_equal(failure['model_seed_indices'], [0, 2])
    assert failure['preprocessing']['model_forward_observed'] is True and failure['seed'] == 29
    saved = save(result, request, tmp_path/'failed')
    archived = json.loads((tmp_path/'failed'/'report.json').read_text())
    assert saved['failed_stage'] == archived['failed_stage'] == 'cross_instance_grasp'
    np.testing.assert_array_equal(archived['grasp_failure']['prepared_points_world'], expected_cloud)
    assert archived['grasp_failure']['model_region_mask'] == steering[consumed].tolist()
    assert archived['grasp_failure']['model_seed_indices'] == [0, 2]
    assert archived['reference_hypotheses'] == []
    assert (tmp_path/'failed'/'selected-regions.npz').is_file()
    assert 'cross_instance_grasp' in (tmp_path/'failed'/'index.html').read_text()


def test_one_reference_failure_does_not_relabel_another_hypothesis_archive(monkeypatch, tmp_path):
    request, prepared = inputs()
    F = pose((.05, .02, 1.))
    region = region_result(request, prepared, F)
    target = pose((.02, 0., 1.15), (0., 0., .4))
    valid = target @ np.linalg.inv(source_grasp(5.1))
    invalid = valid.copy()
    invalid[3, 3] = .5
    monkeypatch.setattr('transfer.cross_instance.pipeline.run_region_transfer', lambda *args, **kwargs: region)
    monkeypatch.setattr('manipulation.cross_instance_grasp.run_cross_instance_grasp',
        lambda *args, **kwargs: dict(arrays=region['arrays'], report={'status': 'computed'},
            hypotheses=[dict(id=identity, T_world_function=F.copy(), T_world_tool=target.copy(),
                tool_reference_anchor_world=anchor, source={'id': identity})
                for identity, anchor in (('invalid_anchor', invalid), ('complete_reference', valid))]))
    result = ground_prepared_skill(request, prepared)
    assert result['status'] == 'failed'
    failed, completed = result['reference_hypotheses']
    assert failed['status'] == 'failed' and failed['reference_diagnostics']['failed_stage'] == 'generate_reference'
    assert completed['status'] == 'completed' and completed['reference_status'] == 'reference'
    assert 'failed_stage' not in completed['reference_diagnostics']
    save(result, request, tmp_path/'mixed')
    failed_record = json.loads((tmp_path/'mixed'/'hypotheses'/'invalid_anchor'/'report.json').read_text())
    complete_record = json.loads((tmp_path/'mixed'/'hypotheses'/'complete_reference'/'report.json').read_text())
    assert failed_record['status'] == 'failed' and failed_record['failed_stage'] == 'generate_reference'
    assert complete_record['status'] == 'completed' and complete_record['reference_status'] == 'reference'
    assert 'failed_stage' not in complete_record and 'error' not in complete_record
    assert (tmp_path/'mixed'/'hypotheses'/'complete_reference'/'reference-paths.json').is_file()


def test_manipulation_keeps_complete_relations_and_independent_interaction_goal(monkeypatch, tmp_path):
    request, prepared = inputs('manipulation')
    request['context']['prerequisite'] = dict(skill_id='declared_relation_dependency', status='completed')
    F = pose((.08, .03, 1.), (.2, -.1, -.35))
    J, A = pose((.02, -.01, .03), (0., .1, .45)), pose((-.05, .01, -.12), (.1, -.2, .15))
    request['relations'] = dict(J=J, A=A, J_source='Source/current chart correspondence',
                               A_source='Declared rigid tool-owner model')
    region = region_result(request, prepared, F)
    monkeypatch.setattr('transfer.cross_instance.pipeline.run_region_transfer', lambda *args, **kwargs: region)
    def wrong_motion_relation(*args, **kwargs):
        raise AssertionError('A manipulation relation entered the grasp inference boundary')
    monkeypatch.setattr('manipulation.cross_instance_grasp.run_cross_instance_grasp', wrong_motion_relation)
    result = ground_prepared_skill(request, prepared)
    assert result['status'] == 'completed'
    row, = result['reference_hypotheses']
    reference = row['reference']
    rep = request['skill']['representation']
    for field in ('H_m', 'C_D'):
        assert reference[field].tobytes() == rep[field].tobytes()
    np.testing.assert_array_equal(reference['J'], J)
    np.testing.assert_array_equal(reference['A'], A)
    assert reference['events'] == rep['events'] and reference['eef_timestamps'][-1] == rep['release_time']
    # Independently interpolate the source's chart-invariant displacement at
    # the actual interaction instant, then compose the declared body and tool
    # charts. The release endpoint is a different time and a different pose.
    times, interaction = np.asarray(rep['timestamps']), rep['interaction_time']
    H = np.asarray(rep['H_m'])
    displacement = H @ np.linalg.inv(H[0])
    after = int(np.searchsorted(times, interaction))
    before = after-1
    fraction = (interaction-times[before])/(times[after]-times[before])
    event_displacement = np.eye(4)
    event_displacement[:3, :3] = Slerp(times[[before, after]],
        Rotation.from_matrix(displacement[[before, after], :3, :3]))([interaction]).as_matrix()[0]
    event_displacement[:3, 3] = ((1-fraction)*displacement[before, :3, 3]
                                 + fraction*displacement[after, :3, 3])
    expected_body = F @ event_displacement @ H[0] @ J
    expected_tool = expected_body @ np.linalg.inv(A)
    np.testing.assert_allclose(reference['object_boundaries']['goals'][0]['pose'], expected_body, atol=1e-12)
    np.testing.assert_allclose(reference['eef_boundaries']['goals'][0]['pose'], expected_tool, atol=1e-12)
    np.testing.assert_allclose(row['T_world_tool'], expected_tool, atol=1e-12)
    np.testing.assert_allclose(row['bound_reference'].sample_numpy(interaction-times[0]), expected_tool, atol=1e-12)
    assert not np.allclose(reference['eef_path'].sample_numpy(1.), expected_tool)
    archive = save(result, request, tmp_path/'operation')
    assert archive['reference_hypotheses'][0]['id'] == request['reference_binding_id']
    with np.load(tmp_path/'operation'/'hypotheses'/row['id']/'nominal-reference.npz', allow_pickle=False) as arrays:
        assert arrays['H_m'].tobytes() == rep['H_m'].tobytes()
        assert arrays['C_D'].tobytes() == rep['C_D'].tobytes()


@pytest.mark.parametrize('repeat_same_observation', [False, True])
def test_cross_instance_requires_two_distinct_observations_before_region_fit(monkeypatch, repeat_same_observation):
    request, prepared = inputs()
    binding = prepared['bindings'][request['reference_binding_id']]
    only = binding['views'][0]
    binding['views'] = [only, only] if repeat_same_observation else [only]
    def unavailable_computation(*args, **kwargs):
        raise AssertionError('Region fitting ran without two independent observation identities')
    monkeypatch.setattr('transfer.cross_instance.pipeline.run_region_transfer', unavailable_computation)
    monkeypatch.setattr('manipulation.cross_instance_grasp.run_cross_instance_grasp', unavailable_computation)
    result = ground_prepared_skill(request, prepared)
    assert result['status'] == 'completed' and result['region_status'] == 'missing'
    assert result['region_reasons'] == ['cross_instance_requires_two_distinct_observations']
    assert result['observation_contract'] == dict(required_count=2,
        provided_view_ids=[view['id'] for view in binding['views']], distinct_count=1)
    assert result['reference_hypotheses'] == []
    assert [view['id'] for view in result['views']] == [view['id'] for view in binding['views']]


def test_explicit_owner_mask_with_missing_depth_support_preserves_input_missing_state(monkeypatch):
    from transfer.segmentation import prepare_mask_observation
    request, prepared = inputs()
    binding = prepared['bindings'][request['reference_binding_id']]
    missing = binding['views'][1]
    # This is a separate semantic owner observation. Its selected pixels have
    # no valid depth, while the task part's reference bank remains observed.
    original_mask = np.zeros_like(missing['owner_mask'])
    original_mask[0, 0] = True
    missing['owner_mask'] = original_mask.copy()
    missing['depth_m'][0, 0] = 0.
    missing['background_mask'][0, 0] = False
    missing['owner_evidence'] = dict(owner_id='current_owner', scope='selected_owner_observation',
                                     source='Independent semantic owner mask with absent optical depth')
    owner = prepare_mask_observation(np.asarray(missing['image']), missing['depth_m'],
        missing['K'], missing['T_world_camera_cv'], original_mask,
        source=missing['owner_evidence']['source'])
    assert not owner['prepared'] and owner['observation'] is None
    missing['owner_observation'] = owner['observation']
    missing['missing_reasons'] = list(owner['reasons'])
    def unavailable_computation(*args, **kwargs):
        raise AssertionError('Region fitting ran with missing owner depth support')
    monkeypatch.setattr('transfer.cross_instance.pipeline.run_region_transfer', unavailable_computation)
    result = ground_prepared_skill(request, prepared)
    assert result['status'] == 'completed' and result['region_status'] == 'missing'
    assert result['region_reasons'] == ['missing_cross_instance_observations']
    assert result['missing_views'] == {missing['id']: owner['reasons']}
    np.testing.assert_array_equal(result['views'][1]['owner_mask'], original_mask)
    assert result['views'][1]['owner_observation'] is None
    assert result['reference_hypotheses'] == []
