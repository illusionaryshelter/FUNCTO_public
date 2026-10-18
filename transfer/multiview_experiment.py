"""Freeze joint-surface predictions from existing, observed RGB-D evidence.

Prediction reads the explicit pair index, original camera observations, and
cached single-view MARCO costs/peaks. It never runs a model or a baseline.
Evaluation is a separate command which opens truth only after every declared
task has a terminal frozen record. Task/category names are display metadata.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

from .experiment import read_archive, restore_prediction, store_prediction
from .features import write_json


# BEGIN joint-surface experiment contract; the retired fusion schema is removed.
BASELINES = ('old_ransac', 'old_prosac_lo', 'old_teaserpp', 'joint')
SAMPLING = {'sampling_radius_m': .03, 'sampling_spacing_m': .005}


def _identifier(value, label):
    """Identifiers name local output folders, never filesystem traversal."""
    if not isinstance(value, str) or not value or Path(value).name != value or value in ('.', '..'):
        raise ValueError(f'{label} must be one nonempty directory name')
    return value


def _path(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def read_manifest(index):
    """Read an explicit paired-observation index without opening truth paths."""
    path = Path(index).resolve()
    description = json.loads(path.read_text())
    cases = description['cases']
    if not isinstance(cases, list) or not cases:
        raise ValueError('The index requires a nonempty cases list')
    ids, groups = set(), {}
    for case in cases:
        identifier = _identifier(case['id'], 'Case ID')
        task = _identifier(case['task'], 'Task ID')
        _identifier(case['view'], 'View ID')
        if identifier in ids:
            raise ValueError('Case IDs must be unique')
        ids.add(identifier)
        for key in ('source', 'target', 'input_dir', 'evidence'):
            if not isinstance(case[key], str) or not case[key]:
                raise ValueError(f'Case requires {key}')
        groups.setdefault(task, []).append(case)
    for task, members in groups.items():
        if len(members) != 2 or len({case['view'] for case in members}) != 2:
            raise ValueError(f'Task {task} requires two unique view identities')
        if len({case['source'] for case in members}) != 1 or len({case['evidence'] for case in members}) != 1:
            raise ValueError('Paired views must declare the same source and demonstration')
        if len({case['target'] for case in members}) != 2:
            raise ValueError('Paired views must reference two distinct frames')
    description['data'] = str(_path(path.parent, description['data']))
    description['tasks'] = [{'id': task, 'cases': members} for task, members in groups.items()]
    return path, description


def _check_observation(observation, depth, calibration):
    """Reject stale pixel/XYZ or camera bindings instead of snapping points."""
    k, pose = np.asarray(calibration['K']), np.asarray(calibration['T_world_camera_cv'])
    np.testing.assert_array_equal(observation['K'], k)
    np.testing.assert_array_equal(observation['T_world_camera_cv'], pose)
    pixels = np.asarray(observation['pixels'])
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.dtype.kind not in 'iu'
            or np.any(pixels < 0) or np.any(pixels >= np.asarray(depth.shape[::-1]))):
        raise ValueError('Observation pixels must index their original depth image')
    values = depth[pixels[:, 1], pixels[:, 0]]
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Entity observations require positive finite original depth')
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(k).T
    xyz = (rays * values[:, None]) @ pose[:3, :3].T + pose[:3, 3]
    np.testing.assert_allclose(observation['xyz'], xyz, rtol=0, atol=1e-8)


def _read_frame(data, record):
    folder = _path(data, record['path'])
    depth = np.load(folder / 'depth_m.npy', allow_pickle=False)
    calibration = json.loads((folder / 'calibration.json').read_text())
    if depth.ndim != 2:
        raise ValueError('Full depth must be a two-dimensional optical-axis image')
    if not (folder / 'rgb.png').is_file():
        raise FileNotFoundError(folder / 'rgb.png')
    return folder, depth, calibration


def _check_evidence(evidence, source, target):
    """Cached row identities and target columns remain the original point IDs."""
    query = np.asarray(evidence['query_indices'])
    region = np.asarray(evidence['query_region'])
    cost = np.asarray(evidence['appearance_cost'])
    peaks = np.asarray(evidence['peak_indices'])
    supported = np.asarray(evidence['supported_target'])
    if (query.ndim != 1 or query.dtype.kind not in 'iu' or len(np.unique(query)) != len(query)
            or np.any(query < 0) or np.any(query >= len(source['xyz']))):
        raise ValueError('Cached queries must be distinct original source indices')
    if region.dtype != np.bool_ or region.shape != query.shape:
        raise ValueError('Cached query_region must be Boolean and query aligned')
    np.testing.assert_array_equal(region, source['region_mask'][query])
    if cost.shape != (len(query), len(target['xyz'])) or not np.isfinite(cost).all() or np.any((cost < 0) | (cost > 1)):
        raise ValueError('Appearance cost must be finite and source-query/target-point aligned')
    if supported.dtype != np.bool_ or supported.shape != (len(target['xyz']),):
        raise ValueError('supported_target must retain the cached interpolation domain')
    if (peaks.ndim != 2 or peaks.shape[0] != len(query) or peaks.dtype.kind not in 'iu'
            or np.any(peaks < -1) or np.any(peaks >= len(target['xyz']))):
        raise ValueError('Cached peaks must index the original target entity observation')


def load_inputs(index, task, cache_root):
    """Load one declared pair, retaining full depth when entity evidence is absent.

    Missing entity preparation is an explicit input state. A prepared entity
    with a missing/corrupt cache is an execution error, not an invented empty
    view. A source and fixed query identities are shared across both views.
    """
    path, description = read_manifest(index)
    data, cache_root = Path(description['data']), Path(cache_root).resolve()
    frame_manifest = json.loads((data / 'manifest.json').read_text())
    source, source_signature, queries, query_region, views = None, None, None, None, []
    source_folder = None
    for case in sorted(task['cases'], key=lambda member: member['view']):
        folder = _path(path.parent, case['input_dir'])
        prepared = json.loads((folder / 'preparation.json').read_text())
        chart = json.loads((folder / 'inputs.json').read_text())
        current = read_archive(folder / 'source_observation.npz')
        current_source_folder, source_depth, source_camera = _read_frame(data, frame_manifest['views'][case['source']])
        _check_observation(current, source_depth, source_camera)
        supplied_region = np.load(_path(data, case['evidence']), allow_pickle=False)
        xy = current['pixels']
        np.testing.assert_array_equal(current['region_mask'], supplied_region[xy[:, 1], xy[:, 0]])
        signature = (chart['source_crop_xyxy'], (folder / 'source.png').read_bytes())
        if source is None:
            source, source_signature, source_folder = current, signature, current_source_folder
        else:
            if source.keys() != current.keys() or signature != source_signature:
                raise ValueError('Paired target views require one identical source observation and crop')
            for key in source:
                np.testing.assert_array_equal(source[key], current[key])
        target_folder, depth, camera = _read_frame(data, frame_manifest['views'][case['target']])
        available = bool(prepared['target']['prepared'])
        reasons = list(prepared['target']['reasons'])
        observation, evidence = None, None
        cache_folder = cache_root / case['id']
        if available:
            frozen = json.loads((cache_folder / 'frozen.json').read_text())
            if frozen.get('complete') is not True or frozen.get('target_truth_used') is not False:
                raise ValueError('Cached appearance requires a complete, truth-free prediction record')
            observation = read_archive(folder / 'target_observation.npz')
            _check_observation(observation, depth, camera)
            evidence = read_archive(cache_folder / 'joint_evidence.npz')
            _check_evidence(evidence, source, observation)
            if queries is None:
                queries, query_region = evidence['query_indices'], evidence['query_region']
            else:
                np.testing.assert_array_equal(queries, evidence['query_indices'])
                np.testing.assert_array_equal(query_region, evidence['query_region'])
        elif not reasons:
            raise ValueError('Unavailable target entity preparation must state its reason')
        views.append({'id': case['view'], 'K': np.asarray(camera['K'], dtype=float),
            'T_world_camera_cv': np.asarray(camera['T_world_camera_cv'], dtype=float),
            'depth_m': depth, 'observation': observation, 'evidence': evidence,
            'missing_reasons': reasons, 'rgb_path': str(target_folder / 'rgb.png'),
            'frame_dir': str(target_folder), 'case': case, 'cache_dir': str(cache_folder)})
    return {'source': source, 'source_rgb': str(source_folder / 'rgb.png'), 'views': views,
            'query_indices': queries, 'query_region': query_region, 'task': task}


def initial_proposals(views, backend):
    """Reuse frozen native transforms even when their old region was rejected."""
    proposals, records = [], []
    for view_index, view in enumerate(views):
        path = Path(view['cache_dir']) / f'old_{backend}.npz'
        state = {'view': view['id'], 'backend': backend, 'available': False}
        if path.is_file():
            values = read_archive(path)
            names = ('sim3_scale', 'sim3_rotation', 'sim3_translation')
            if all(name in values for name in names):
                transform = (float(values[names[0]]), values[names[1]], values[names[2]])
                proposals.append({'transform': transform, 'origin': f'native_{backend}', 'view_index': view_index})
                state['available'] = True
            else:
                state['reason'] = 'frozen_native_transform_unavailable'
        else:
            state['reason'] = 'frozen_native_prediction_unavailable'
        records.append(state)
    return proposals, records


def save_surface(directory, surface):
    """Freeze element/provenance arrays; full images remain in their input files."""
    directory = Path(directory)
    omitted = {'depth_m', 'K', 'T_world_camera_cv', 'pixel_to_observation', 'adjacency'}
    # BEGIN: curved caches retain both raw-depth and entity-qualified graphs.
    curved = surface.get('region_geometry') == 'surface'
    graphs = ('adjacency', 'observation_adjacency', 'depth_adjacency') if curved else ('adjacency',)
    omitted.update(graphs)
    if curved:
        omitted.add('observation_model')
    # END: explicit graph schema; private spatial indices are rebuilt on load.
    arrays = {key: value for key, value in surface.items()
              if isinstance(value, np.ndarray) and key not in omitted and not key.startswith('_')}
    for name in graphs:
        adjacency = surface[name].tocsr()
        arrays.update({name+'_data': adjacency.data, name+'_indices': adjacency.indices,
                       name+'_indptr': adjacency.indptr, name+'_shape': np.asarray(adjacency.shape)})
    np.savez_compressed(directory / 'surface.npz', **arrays)
    metadata = {key: value for key, value in surface.items()
                if key not in omitted and not isinstance(value, np.ndarray) and not key.startswith('_')}
    if curved:
        from dataclasses import asdict
        metadata['observation_model'] = asdict(surface['observation_model'])
    write_json(directory / 'surface.json', metadata)


def load_surface(directory, views):
    """Restore a frozen surface and bind original depth to its saved view IDs."""
    directory = Path(directory)
    surface = read_archive(directory / 'surface.npz')
    surface.update(json.loads((directory / 'surface.json').read_text()))
    if list(surface['view_ids']) != [view['id'] for view in views]:
        raise ValueError('Surface provenance does not match the declared view order')
    # BEGIN: restore declared curved graph domains without manufacturing normals.
    curved = surface.get('region_geometry') == 'surface'
    graphs = ('adjacency', 'observation_adjacency', 'depth_adjacency') if curved else ('adjacency',)
    for name in graphs:
        shape = tuple(surface.pop(name+'_shape'))
        surface[name] = csr_matrix((surface.pop(name+'_data'), surface.pop(name+'_indices'),
                                    surface.pop(name+'_indptr')), shape=shape)
    if curved:
        from scipy.spatial import cKDTree
        from .dual_view.curved_geometry import ObservationModel
        surface['observation_model'] = ObservationModel(**surface['observation_model'])
        for prefix, points, membership in (('_entity_', 'xyz', 'entity_member'),
                ('_observation_entity_', 'observation_xyz', 'observation_entity_member')):
            ids = np.flatnonzero(surface[membership])
            surface[prefix+'ids'] = ids
            surface[prefix+'tree'] = cKDTree(surface[points][ids]) if len(ids) else None
    # END: cached measurement identity and explicit observation model.
    surface['K'] = np.asarray([view['K'] for view in views])
    surface['T_world_camera_cv'] = np.asarray([view['T_world_camera_cv'] for view in views])
    surface['depth_m'] = [view['depth_m'] for view in views]
    maps = []
    for view_index, view in enumerate(views):
        mapping = np.full(view['depth_m'].shape, -1, dtype=np.int64)
        ids = np.flatnonzero(surface['observation_view_index'] == view_index)
        pixels = surface['observation_pixels'][ids]
        mapping[pixels[:, 1], pixels[:, 0]] = ids
        maps.append(mapping)
    surface['pixel_to_observation'] = maps
    return surface


def _save_candidate_evidence(directory, evidence):
    """Preserve typed candidate records separately from original model caches."""
    arrays = {key: value for key, value in evidence.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(Path(directory) / 'candidates.npz', **arrays)
    write_json(Path(directory) / 'candidates.json', {
        key: value for key, value in evidence.items() if not isinstance(value, np.ndarray)})


def block_evaluation_reads():
    """Install a process-local truth barrier for the prediction CLI only."""
    forbidden_names = {'truth.npz', 'evaluation.npz', 'evaluation.json', 'evaluation-audit.json',
                       'state-checks.json', 'intervention-checks.json', 'summary.json', 'metrics.csv'}
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        if (path.name in forbidden_names or path.name.endswith('_evaluation.json')
                or 'pair-truth' in path.parts or 'evaluation-scenes' in path.parts):
            raise PermissionError(f'Evaluation-only artifact requested during prediction: {path}')
    sys.addaudithook(audit)


def predict_suite(index, cache_root, output, selected=None, backend='teaserpp'):
    """Run each task serially and freeze failures as well as accepted results."""
    from .multiview_geometry import build_target_surface
    from .multiview_features import build_correspondence_candidates
    from .multiview_grounding import infer_multiview_region
    if backend not in ('teaserpp', 'ransac'):
        raise ValueError('Native proposal backend must be teaserpp or ransac')
    path, description = read_manifest(index)
    tasks = description['tasks']
    if selected:
        if not set(selected) <= {task['id'] for task in tasks}:
            raise ValueError('Unknown selected task')
        tasks = [task for task in tasks if task['id'] in selected]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {'frozen': False, 'target_truth_used': False, 'index': str(path),
              'cache_root': str(Path(cache_root).resolve()), 'native_backend': backend,
              'sampling': dict(SAMPLING), 'tasks': []}
    for task in tasks:
        directory = output / task['id']
        directory.mkdir()
        state = {'id': task['id'], 'cases': task['cases'], 'status': 'not_run',
                 'accepted': False, 'reasons': [], 'warnings': []}
        record['tasks'].append(state)
        write_json(output / 'predictions.json', record)
        stage, started = 'load_observations', perf_counter()
        try:
            loaded = load_inputs(path, task, cache_root)
            state['input_availability'] = [{'view': view['id'], 'geometry': True,
                'entity': view['observation'] is not None, 'appearance': view['evidence'] is not None,
                'reasons': view['missing_reasons']} for view in loaded['views']]
            if loaded['query_indices'] is None:
                state.update(status='missing_input', reasons=['no_cached_appearance_queries'])
            else:
                stage = 'build_joint_surface'
                surface = build_target_surface(loaded['views'], sampling_radius_m=SAMPLING['sampling_radius_m'])
                stage = 'build_correspondence_candidates'
                evidence = build_correspondence_candidates(loaded['source'], loaded['views'], surface,
                    query_indices=loaded['query_indices'], query_region=loaded['query_region'])
                initial, native_records = initial_proposals(loaded['views'], backend)
                stage = 'infer_joint_region'
                result = infer_multiview_region(loaded['source'], surface, evidence,
                                               initial_transforms=initial, **SAMPLING)
                result['diagnostics'].update(target_truth_used=False, native_proposals=native_records,
                    prediction_ms=1000 * (perf_counter() - started))
                stage = 'freeze_artifacts'
                save_surface(directory, surface)
                _save_candidate_evidence(directory, evidence)
                store_prediction(directory / 'prediction', result, {})
                state.update(status='completed', accepted=bool(result['accepted']), reasons=result['reasons'],
                             warnings=result['diagnostics'].get('quality_warnings', []))
        except Exception as error:
            state.update(status='failed', reasons=[str(error)], failure_stage=stage,
                         exception_type=type(error).__name__)
        state['elapsed_ms'] = 1000 * (perf_counter() - started)
        state['target_truth_used'] = False
        write_json(directory / 'frozen.json', state)
        write_json(output / 'predictions.json', record)
        print(json.dumps({key: state[key] for key in ('id', 'status', 'accepted', 'reasons')}, ensure_ascii=False), flush=True)
    record['frozen'] = True
    write_json(output / 'predictions.json', record)
    return record


def frozen_index(output):
    """Validate the entire terminal matrix before any evaluator opens truth."""
    output = Path(output)
    path = output / 'predictions.json'
    if not path.is_file():
        raise FileNotFoundError('Finish every declared prediction before evaluation')
    record = json.loads(path.read_text())
    if record.get('frozen') is not True or record.get('target_truth_used') is not False or not record.get('tasks'):
        raise ValueError('Evaluation requires a complete truth-free frozen prediction index')
    for state in record['tasks']:
        frozen = json.loads((output / state['id'] / 'frozen.json').read_text())
        if (state['status'] not in ('completed', 'failed', 'missing_input') or frozen != state
                or frozen.get('target_truth_used') is not False):
            raise ValueError('Every declared task must have a matching terminal frozen record')
        if state['status'] == 'completed':
            for name in ('prediction.json', 'prediction.npz', 'surface.json', 'surface.npz'):
                if not (output / state['id'] / name).is_file():
                    raise FileNotFoundError(output / state['id'] / name)
    return record


def evaluate_suite(output):
    """Evaluate union representatives using held-out original-pixel identities."""
    from .evaluation import evaluate_prediction
    output = Path(output).resolve()
    record = frozen_index(output)
    rows = []
    for state in record['tasks']:
        row = {key: state[key] for key in ('id', 'status', 'accepted', 'reasons', 'warnings')}
        row.update(primary_metrics=None, diagnostic_first_mode_metrics=None, mode_metrics=[])
        if state['status'] == 'completed':
            task = {'id': state['id'], 'cases': state['cases']}
            loaded = load_inputs(record['index'], task, record['cache_root'])
            surface = load_surface(output / state['id'], loaded['views'])
            result = restore_prediction(output / state['id'] / 'prediction')
            _, description = read_manifest(record['index'])
            references, preferred_ids, support_images = [], [], []
            for view in loaded['views']:
                truth = read_archive(Path(view['frame_dir']) / 'truth.npz')
                pair = json.loads(_path(description['data'], view['case']['pair_truth']).read_text())
                support_images.append(truth['support_id_image'])
                references.append(np.asarray(pair['target_reference_evidence_center_world']))
                preferred_ids.append(int(pair['preferred_support_id']))
            if len(set(preferred_ids)) != 1:
                raise ValueError('Paired evaluation annotations disagree on the preferred surface')
            for reference in references[1:]:
                np.testing.assert_allclose(reference, references[0], rtol=0, atol=1e-10)
            observation_support = np.empty(len(surface['observation_xyz']), dtype=np.int64)
            for view_index, support_image in enumerate(support_images):
                ids = np.flatnonzero(surface['observation_view_index'] == view_index)
                pixels = surface['observation_pixels'][ids]
                observation_support[ids] = support_image[pixels[:, 1], pixels[:, 0]]
            support = observation_support[surface['representative_observation']]
            measurements = evaluate_prediction(result, surface['xyz'],
                {'support_id': support, 'preferred_support_id': np.asarray(preferred_ids[0])}, references[0])
            row['mode_metrics'] = measurements['mode_metrics']
            if result['accepted']:
                row['primary_metrics'] = {key: value for key, value in measurements.items() if key != 'mode_metrics'}
            elif measurements['mode_metrics']:
                # An ambiguous/rejected result has no executable primary. Its
                # first saved diagnostic is not promoted by the evaluator.
                row['diagnostic_first_mode_metrics'] = measurements['mode_metrics'][0]
            mapped = surface['observation_to_element']
            valid = mapped >= 0
            row['association_truth_disagreements'] = int(np.count_nonzero(
                observation_support[valid] != support[mapped[valid]]))
            row['reference_center'] = references[0].tolist()
        rows.append(row)
    evaluation = {'prediction_index': str(output / 'predictions.json'),
                  'read_after_all_predictions_frozen': True, 'tasks': rows}
    write_json(output / 'evaluation.json', evaluation)
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    prediction = commands.add_parser('predict')
    prediction.add_argument('--index', type=Path, required=True)
    prediction.add_argument('--cache-root', type=Path, required=True)
    prediction.add_argument('--output', type=Path, required=True)
    prediction.add_argument('--task', action='append')
    prediction.add_argument('--native-backend', choices=('teaserpp', 'ransac'), default='teaserpp')
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'predict':
        block_evaluation_reads()
        predict_suite(args.index, args.cache_root, args.output, args.task, args.native_backend)
    else:
        evaluate_suite(args.output)


if __name__ == '__main__':
    main()
# END joint-surface experiment contract.
