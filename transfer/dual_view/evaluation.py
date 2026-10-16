"""Evaluate frozen revised dual-view predictions against immutable comparisons.

This entry point is intentionally separate from prediction: only this process
opens target truth. Historical outputs are read, never reoptimized or edited.
"""

import argparse
from collections import Counter
import json
# BEGIN relative historical asset paths and held-out transform metrics.
import os
# END evaluation-only import.
from pathlib import Path

import numpy as np
from PIL import Image

from ..evaluation import evaluate_prediction
from ..experiment import read_archive, restore_prediction
from ..features import write_json
from ..multiview_experiment import _path, load_surface, read_manifest
from ..multiview_visualization import read_rgb, single_view_picture, source_picture
from .io import load_observations
# BEGIN transform evaluator is never imported by prediction.
from .transform_metrics import evaluate_transforms
# END held-out metric import.
# BEGIN shared GT overlay for all three frozen schemes.
from .visualization import ground_truth_layer, region_picture, with_ground_truth, write_html
# END shared GT overlay import.


# BEGIN current revised freeze validation excludes the historical schema.
def frozen_index(output):
    """Require every declared prediction to have a matching terminal record."""
    output = Path(output)
    record = json.loads((output / 'predictions.json').read_text())
    if (record.get('scheme') != 'dual_view_observed_partial' or record.get('frozen') is not True
            or record.get('target_truth_used') is not False):
        raise ValueError('Evaluation requires a complete truth-free dual-view run')
    ids = [state['id'] for state in record['tasks']]
    if ids != record['declared_task_ids'] or len(set(ids)) != len(ids):
        raise ValueError('Frozen task domain differs from the declared prediction domain')
    for state in record['tasks']:
        saved = json.loads((output / state['id'] / 'frozen.json').read_text())
        if (saved != state or state.get('target_truth_used') is not False
                or state['status'] not in ('completed', 'failed', 'missing_input')):
            raise ValueError('A task lacks its matching terminal record')
        if state['status'] == 'completed':
            if set(state['backends']) != set(record['backends']):
                raise ValueError('A declared native backend was not run')
            for backend, item in state['backends'].items():
                if item['status'] not in ('completed', 'failed'):
                    raise ValueError('A backend has not reached a terminal state')
                if item['status'] == 'completed':
                    for suffix in ('.json', '.npz'):
                        if not (output / state['id'] / (backend+suffix)).is_file():
                            raise FileNotFoundError(f'{state["id"]}/{backend}{suffix}')
                    result = json.loads((output / state['id'] / (backend+'.json')).read_text())
                    diagnostics = result['diagnostics']
                    if (diagnostics.get('scheme') != 'dual_view_observed_partial'
                            or diagnostics.get('native_backend') != backend
                            or diagnostics.get('target_truth_used') is not False):
                        raise ValueError('Frozen prediction scheme/backend/provenance mismatch')
    return record
# END revised-only freeze validation.


def _empty(reasons):
    return {'accepted': False, 'reasons': list(reasons), 'diagnostics': {}, 'modes': []}


def _row(task, scheme, backend, view, status, result, metrics):
    """Keep rejected diagnostic error separate from accepted primary error."""
    warnings = result['diagnostics'].get('quality_warnings', [])
    decision = result['diagnostics'].get('decision_status')
    display = {'missing_input': '输入缺失 / 模型不适用', 'failed': '运行失败'}.get(status)
    if display is None:
        display = ('接受（有警告）' if warnings else '接受') if result['accepted'] else '拒绝'
        if decision == 'search_incomplete':
            display = '搜索未完成 / 不接受'
    return {'task': task, 'scheme': scheme, 'backend': backend, 'view': view,
        'status': status, 'decision_status': decision, 'display_status': display,
        'accepted': bool(result['accepted']), 'warnings': warnings, 'reasons': result['reasons'],
        'primary_metrics': {k: v for k, v in metrics.items() if k != 'mode_metrics'}
            if metrics is not None and result['accepted'] else None,
        'diagnostic_first_mode_metrics': metrics['mode_metrics'][0]
            if metrics is not None and not result['accepted'] and metrics['mode_metrics'] else None,
        'mode_metrics': [] if metrics is None else metrics['mode_metrics']}


def _summary(rows):
    # BEGIN original dual stays a separate denominator from its revision.
    labels = {'old': '旧方案', 'new': '新方案', 'dual_original': '原双视角', 'dual': '修订双视角',
              'baseline': '当前双视角基线', 'trial': '精度试验'}
    # END explicit comparison labels.
    answer = []
    for key in dict.fromkeys((row['scheme'], row['backend'], row['view']) for row in rows):
        values = [row for row in rows if (row['scheme'], row['backend'], row['view']) == key]
        accepted = [row for row in values if row['accepted']]
        errors = [row['primary_metrics']['center_error_m']*1000 for row in accepted]
        good = sum(row['primary_metrics']['top1_preferred_support_hit'] for row in accepted)
        answer.append({'scheme': key[0], 'backend': key[1], 'view': key[2],
            'label': f'{labels[key[0]]} / {key[1]}', 'total': len(values),
            'accepted': len(accepted), 'correct_surface': good,
            'wrong_surface': len(accepted)-good,
            'rejected': sum(row['status'] == 'completed' and not row['accepted']
                            and row['decision_status'] != 'search_incomplete' for row in values),
            'incomplete': sum(row['decision_status'] == 'search_incomplete' for row in values),
            'missing': sum(row['status'] == 'missing_input' for row in values),
            'failed': sum(row['status'] == 'failed' for row in values),
            'median_center_error_mm': float(np.median(errors)) if errors else None})
    return answer


# BEGIN held-out transform/performance comparison against immutable original dual.
def _center_radius_metric(metric, sampling_radius_m):
    """Normalize the final observed-region center error by the recorded radius.

    This is a held-out evaluation ratio, never an acceptance threshold. The
    radius is a physical task input and must not be multiplied by fitted scale.
    Millimeters and percentages are display conversions of these two fields.
    """
    radius = float(sampling_radius_m)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Center normalization requires a finite positive recorded sampling radius')
    if metric is None:
        return None
    error = metric.get('center_error_m')
    if error is not None and (not np.isfinite(error) or error < 0):
        raise ValueError('Final region center error must be finite and nonnegative when present')
    return {**metric, 'sampling_radius_m': radius,
            'center_error_radius_ratio': None if error is None else float(error/radius)}


def _dual_region_metrics(result, points, truth, reference, *, sampling_radius_m):
    """Keep accepted top-k/sample aggregates limited to the one primary region.

    The shared evaluator treats every mode as an output. Revised dual modes
    after index zero are diagnostic witnesses, so their good GT outcome must
    never improve accepted top-k hits, sample quality, or geometric bounds.
    Their individual measurements remain explicitly labelled diagnostics.
    """
    all_metrics = evaluate_prediction(result, points, truth, reference)
    primary = (evaluate_prediction({**result, 'modes': result['modes'][:1]}, points, truth, reference)
               if result['accepted'] and len(result['modes']) > 1 else all_metrics)
    primary = _center_radius_metric(primary, sampling_radius_m)
    primary['aggregate_scope'] = 'primary_mode_only' if result['accepted'] else 'rejected_diagnostics_only'
    primary['center_error_definition'] = 'final_observed_region_center_to_held_out_demonstration_center_3d'
    primary['mode_metrics'] = [{**_center_radius_metric(record, sampling_radius_m),
                               'role': 'primary' if result['accepted'] and index == 0
                               else 'competitor_diagnostic' if result['accepted'] else 'rejected_diagnostic'}
                              for index, record in enumerate(all_metrics['mode_metrics'])]
    return primary


def _source_transform_points(folder):
    """Read only the frozen small source rows, never feature maps or target GT."""
    with np.load(Path(folder) / 'source_queries.npz', allow_pickle=False) as archive:
        source = {key: archive[key] for key in ('query_xyz', 'check_xyz', 'query_region',
                                               'center', 'normal')}
    description = json.loads((Path(folder) / 'source_geometry.json').read_text())
    source['check_query_region'] = description['diagnostics']['check_query_region']
    return source


def _transform_evaluation(stem, result, pair):
    """Only this post-freeze evaluator opens affine GT and the saved estimate."""
    source = _source_transform_points(Path(stem).parent)
    with np.load(Path(stem).with_suffix('.npz'), allow_pickle=False) as archive:
        keys = ('sim3_scale', 'sim3_rotation', 'sim3_translation')
        present = [key in archive for key in keys]
        if any(present) and not all(present):
            raise ValueError('Frozen estimate has only part of its Sim(3) parameters')
        primary = tuple(archive[key] for key in keys) if all(present) else None
    return evaluate_transforms(result, source, pair['true_affine_transform'],
                               primary_transform=primary)


def _original_dual(results, web, *, old_results=None, new_results=None):
    """Read the explicitly requested historical schema and its original assets.

    The current freeze validator accepts only the revised scheme. This reader
    has the separate purpose of evaluating the named immutable prior run; it
    never admits a historical run as a current prediction.
    """
    record = json.loads((results / 'predictions.json').read_text())
    evaluation = json.loads((web / 'evaluation.json').read_text())
    if (record.get('scheme') != 'dual_view_continuous' or record.get('frozen') is not True
            or record.get('target_truth_used') is not False
            or Path(evaluation['prediction_index']).resolve() != (results / 'predictions.json').resolve()
            or evaluation.get('read_after_all_predictions_frozen') is not True):
        raise ValueError('Original dual comparison must name its frozen run and matching evaluation')
    for key, expected in (('old_results', old_results), ('new_results', new_results)):
        if expected is not None and Path(evaluation[key]).resolve() != Path(expected).resolve():
            raise ValueError('Reused single-view images must match the named frozen single-view result directories')
    rows = {(row['task'], row['backend']): row for row in evaluation['rows'] if row['scheme'] == 'dual'}
    return record, rows


def _supervisor_metrics(path, output):
    """Bind an optional supervisor record to this exact prediction directory."""
    if path is None:
        return None
    report = json.loads(Path(path).read_text())
    command = report['command']
    if '--output' not in command or Path(command[command.index('--output')+1]).resolve() != Path(output).resolve():
        raise ValueError('Supervisor timings must correspond to the named prediction output')
    return {'record_file': str(Path(path).resolve()), 'wall_seconds': float(report['elapsed_s']),
            'status': report['status'], 'returncode': report.get('returncode'),
            'peak_process_tree_rss_mib': report.get('peak_process_tree_rss_mib'),
            'scope': 'Supervisor wall time includes process startup and shutdown; distinct from sum of task timers.'}


def _runtime_summary(record, supervisor=None):
    """Actual recorded serial times; never extrapolate an unmeasured stage."""
    tasks = record['tasks']
    times = [float(state['seconds']) for state in tasks if 'seconds' in state]
    backend = {}
    for name in record['backends']:
        values = [float(state['backends'][name]['seconds']) for state in tasks
                  if name in state.get('backends', {}) and 'seconds' in state['backends'][name]]
        backend[name] = {'recorded_tasks': len(values), 'sum_seconds': sum(values),
            'median_seconds': float(np.median(values)) if values else None,
            'minimum_seconds': min(values) if values else None,
            'maximum_seconds': max(values) if values else None}
    preparation_stages = {}
    for state in tasks:
        for name, value in state.get('stage_seconds', {}).items():
            preparation_stages[name] = preparation_stages.get(name, 0.)+float(value)
    return {'recorded_tasks': len(times), 'sum_task_seconds': sum(times),
            'median_task_seconds': float(np.median(times)) if times else None,
            'minimum_task_seconds': min(times) if times else None,
            'maximum_task_seconds': max(times) if times else None,
            'backends': backend,
            'recorded_preparation_stage_sums_seconds': preparation_stages,
            'supervisor': supervisor,
            'scope': 'Recorded cached prediction only; excludes SAM/MARCO extraction and creation of reused target surface caches.'}


def evaluate(output, destination, *, old_results, new_results,
             baseline_dual_results=None, baseline_dual_web=None,
             prediction_resources=None, baseline_prediction_resources=None):
    """Render frozen modes on original RGB, with no selection from target truth."""
    output, destination = Path(output).resolve(), Path(destination).resolve()
    record = frozen_index(output)
    if (baseline_dual_results is None) != (baseline_dual_web is None):
        raise ValueError('Original dual results and its rendered web directory must be supplied together')
    baseline_record, baseline_rows = None, {}
    if baseline_dual_results is not None:
        baseline_dual_results, baseline_dual_web = Path(baseline_dual_results).resolve(), Path(baseline_dual_web).resolve()
        if destination == baseline_dual_web or output == baseline_dual_results:
            raise ValueError('Revised evaluation must not overwrite the original frozen comparison')
        baseline_record, baseline_rows = _original_dual(baseline_dual_results, baseline_dual_web,
            old_results=old_results, new_results=new_results)
        original_tasks = {item['id']: item for item in baseline_record['tasks']}
        for state in record['tasks']:
            if state['id'] not in original_tasks or state['cases'] != original_tasks[state['id']]['cases']:
                raise ValueError('Reused original images require identical declared source/target cases')
    if baseline_prediction_resources is not None and baseline_record is None:
        raise ValueError('Historical supervisor timings require the historical prediction input')
    runtime = {'revision': _runtime_summary(record, _supervisor_metrics(prediction_resources, output)),
               'original_dual': None if baseline_record is None else _runtime_summary(baseline_record,
                   _supervisor_metrics(baseline_prediction_resources, baseline_dual_results))}
    _, description = read_manifest(record['index'])
    # BEGIN in-place regeneration of derived evaluation/web artifacts only.
    # Frozen prediction directories are opened read-only. Rendering always
    # starts from raw RGB, so repeated refreshes cannot accumulate GT strokes.
    destination.mkdir(parents=True, exist_ok=True)
    assets = destination / 'assets'
    assets.mkdir(exist_ok=True)
    ground_truth = []
    # END in-place report regeneration.
    rows, cards, visibility, rejection_counts = [], [], [], Counter()
    for state in record['tasks']:
        task = {'id': state['id'], 'cases': state['cases']}
        try:
            loaded = load_observations(record['index'], task, record['cache_root'])
        except Exception as error:
            # A broken input/cache binding must not remove this task from the
            # report denominator or prevent unrelated tasks from being shown.
            card = {'id': task['id'], 'source': None, 'methods': []}
            cards.append(card)
            reason = f'evaluation_observation_binding_failed:{type(error).__name__}:{error}'
            for scheme in (('old', 'new', 'dual_original', 'dual') if baseline_record else ('old', 'new', 'dual')):
                for backend in record['backends']:
                    method = {'scheme': scheme, 'label': f'{scheme} / {backend}', 'panels': []}
                    for view_id in (['A+B'] if scheme in ('dual', 'dual_original') else
                                    [case['view'] for case in sorted(task['cases'], key=lambda c: c['view'])]):
                        row = _row(task['id'], scheme, backend, view_id, 'failed', _empty([reason]), None)
                        row['prediction_task_status'] = state['status']
                        rows.append(row)
                        method['panels'].append({**row, 'images': [], 'search': {}})
                    card['methods'].append(method)
            continue
        source = loaded['source']
        demo = np.load(_path(description['data'], task['cases'][0]['evidence']), allow_pickle=False)
        source_name = f'{task["id"]}_source.png'
        if baseline_record is None:
            source_picture(read_rgb(loaded['source_rgb']), demo, source,
                           loaded['query_indices'], loaded['query_region']).save(assets / source_name)
            source_asset = 'assets/'+source_name
        else:
            original_source = baseline_dual_web / 'assets' / source_name
            if not original_source.is_file():
                raise FileNotFoundError(f'Original source comparison image is missing: {original_source}')
            source_asset = os.path.relpath(original_source, destination)
        card = {'id': task['id'], 'source': source_asset, 'methods': []}
        cards.append(card)
        # BEGIN retain already-read pair annotations for the shared GT layers.
        truths, references, preferred, rgbs, pairs = [], [], [], [], []
        # END pair reuse.
        for view in loaded['views']:
            truth = read_archive(Path(view['frame_dir']) / 'truth.npz')
            pair = json.loads(_path(description['data'], view['case']['pair_truth']).read_text())
            # BEGIN reuse the same annotation for metrics and visual truth.
            pairs.append(pair)
            # END annotation reuse.
            truths.append(truth['support_id_image'])
            references.append(np.asarray(pair['target_reference_evidence_center_world']))
            preferred.append(int(pair['preferred_support_id']))
            rgbs.append(read_rgb(view['rgb_path']))
        if len(set(preferred)) != 1:
            raise ValueError('Target views disagree on preferred support identity')
        np.testing.assert_allclose(references[0], references[1], rtol=0, atol=1e-10)
        np.testing.assert_allclose(pairs[0]['true_affine_transform'], pairs[1]['true_affine_transform'], rtol=0, atol=1e-10)

        # BEGIN one held-out GT layer per camera, reused across every method/mode.
        gt_layers, gt_captions, gt_records = [], [], []
        for v, view in enumerate(loaded['views']):
            pair_path = _path(description['data'], view['case']['pair_truth'])
            pair = pairs[v]
            layer, details = ground_truth_layer(truths[v] == preferred[v], references[v],
                view['K'], view['T_world_camera_cv'],
                center_visible=pair['target_reference_center_visible'])
            caption = {'visible': 'GT 中心可见（实线）', 'occluded': 'GT 中心被遮挡（虚线投影）',
                       'outside_image': 'GT 中心在画面外', 'behind_camera': 'GT 中心在相机后方'}[
                           details['reference_display']]
            caption += f'；GT 支持面可见 {details["support_visible_pixels"]} 像素'
            gt_layers.append(layer)
            gt_captions.append(caption)
            gt_records.append({'task': task['id'], 'view': view['id'], **details,
                'preferred_support_id': preferred[v],
                'truth_file': str(Path(view['frame_dir']) / 'truth.npz'),
                'pair_truth_file': str(pair_path), 'images': []})
        ground_truth.extend(gt_records)
        # END GT layers, which never participate in prediction or ranking.

        # Single-view methods retain their own target point domains. Their A/B
        # outputs are displayed independently, never GT-selected as a joint run.
        for scheme, root in (('old', Path(old_results)), ('new', Path(new_results))):
            for backend in record['backends']:
                label = f'{"旧方案：原生对应" if scheme == "old" else "新方案：原生 + 6/12 点区域评分"} / {backend}'
                method = {'scheme': scheme, 'label': label, 'panels': []}
                card['methods'].append(method)
                for v, view in enumerate(loaded['views']):
                    stem = root / view['case']['id'] / (f'old_{backend}' if scheme == 'old' else backend)
                    status = 'completed'
                    if view['observation'] is None:
                        result, status = _empty(view['missing_reasons']), 'missing_input'
                    elif not stem.with_suffix('.npz').is_file() or not stem.with_suffix('.json').is_file():
                        result, status = _empty(['frozen_comparison_artifact_missing']), 'failed'
                    else:
                        result = restore_prediction(stem)
                    metrics = None
                    if view['observation'] is not None:
                        xy = view['observation']['pixels']
                        metrics = evaluate_prediction(result, view['observation']['xyz'],
                            {'support_id': truths[v][xy[:, 1], xy[:, 0]],
                             'preferred_support_id': np.asarray(preferred[v])}, references[v])
                    row = _row(task['id'], scheme, backend, view['id'], status, result, metrics)
                    rows.append(row)
                    name = f'{task["id"]}_{scheme}_{backend}_{view["id"]}.png'
                    # BEGIN GT is the final layer, including failed/missing predictions.
                    if baseline_record is None:
                        picture = single_view_picture(rgbs[v], view['observation'], result)
                        with_ground_truth(picture, gt_layers[v]).save(assets / name)
                        comparison_asset = 'assets/'+name
                    else:
                        original_asset = baseline_dual_web / 'assets' / name
                        if not original_asset.is_file():
                            raise FileNotFoundError(f'Original single-view comparison image is missing: {original_asset}')
                        comparison_asset = os.path.relpath(original_asset, destination)
                    gt_records[v]['images'].append(comparison_asset)
                    # END old/new GT display; frozen result and metrics remain unchanged.
                    method['panels'].append({**row, 'images': [{'path': comparison_asset,
                        # BEGIN per-image GT visibility remains available when zoomed.
                        'caption': f'{view["id"]} · {row["display_status"]} · {gt_captions[v]}'}],
                        # END per-image GT visibility.
                        'search': {'frozen_prediction': str(stem),
                                   'selected_origin': result['diagnostics'].get('selected_origin')}})

        surface = None
        if state.get('surface_directory'):
            surface = load_surface(state['surface_directory'], loaded['views'])
            observation_support = np.empty(len(surface['observation_xyz']), dtype=np.int64)
            for v in range(2):
                ids = np.flatnonzero(surface['observation_view_index'] == v)
                xy = surface['observation_pixels'][ids]
                observation_support[ids] = truths[v][xy[:, 1], xy[:, 0]]
            support = observation_support[surface['representative_observation']]
        for backend in record['backends']:
            if baseline_record is not None:
                original = dict(baseline_rows[(task['id'], backend)])
                original.update(scheme='dual_original', region_metric_scope='historical_frozen_mode_semantics')
                original_stem = baseline_dual_results / task['id'] / backend
                original_prediction = json.loads(original_stem.with_suffix('.json').read_text()) if original['status'] == 'completed' else None
                if original_prediction is not None:
                    radius = original_prediction['diagnostics']['sampling_radius_m']
                    for key in ('primary_metrics', 'diagnostic_first_mode_metrics'):
                        original[key] = _center_radius_metric(original[key], radius)
                    original['mode_metrics'] = [_center_radius_metric(metric, radius) for metric in original['mode_metrics']]
                original['transform_metrics'] = (_transform_evaluation(original_stem, original_prediction, pairs[0])
                    if original_prediction is not None else None)
                original_state = original_tasks[task['id']]
                original['runtime'] = {'task_seconds': original_state.get('seconds'),
                    'preparation_seconds': original_state.get('preparation_seconds'),
                    'preparation_stage_seconds': original_state.get('stage_seconds', {}),
                    'backend_seconds': original_state.get('backends', {}).get(backend, {}).get('seconds'),
                    'native_registration_seconds': None if original_prediction is None else original_prediction['diagnostics'].get('native_seconds'),
                    'timing_seconds': {} if original_prediction is None else original_prediction['diagnostics'].get('timing_seconds', {})}
                rows.append(original)
                baseline_images = []
                mode_count = max(1, len(original_prediction['modes'])) if original_prediction else 1
                for rank in range(mode_count):
                    for v, view in enumerate(loaded['views']):
                        asset = baseline_dual_web / 'assets' / f'{task["id"]}_dual_{backend}_{view["id"]}_mode{rank}.png'
                        if not asset.is_file():
                            raise FileNotFoundError(f'Original dual comparison image is missing: {asset}')
                        baseline_images.append({'path': os.path.relpath(asset, destination),
                            'caption': f'{view["id"]} · 原双视角冻结输出 · {original["display_status"]} · 区域 {rank+1}'})
                        gt_records[v]['images'].append(os.path.relpath(asset, destination))
                card['methods'].append({'scheme': 'dual_original', 'label': f'原双视角 / {backend}',
                    'panels': [{**original, 'images': baseline_images, 'search': {
                        'full_prediction': str(original_stem.with_suffix('.json')),
                        'historical_assets_reused_without_copy': True}}]})
            backend_state = state['backends'].get(backend, state)
            status = backend_state['status']
            stem = output / task['id'] / backend
            result = (restore_prediction(stem) if status == 'completed' else
                      _empty(backend_state.get('reasons', [backend_state.get('error', status)])))
            metrics = None
            if surface is not None:
                metrics = _dual_region_metrics(result, surface['xyz'],
                    {'support_id': support, 'preferred_support_id': np.asarray(preferred[0])}, references[0],
                    sampling_radius_m=record['sampling']['sampling_radius_m'])
            row = _row(task['id'], 'dual', backend, 'A+B', status, result, metrics)
            row['transform_metrics'] = (_transform_evaluation(stem, result, pairs[0])
                if status == 'completed' else None)
            row['runtime'] = {'task_seconds': state.get('seconds'),
                'preparation_seconds': state.get('preparation_seconds'),
                'preparation_stage_seconds': state.get('stage_seconds', {}),
                'backend_seconds': backend_state.get('seconds'),
                'native_registration_seconds': result['diagnostics'].get('native_seconds'),
                'timing_seconds': result['diagnostics'].get('timing_seconds', {})}
            rows.append(row)
            rejection_counts.update(row['reasons'])
            diag = result['diagnostics']
            candidates = diag.get('hypotheses', [])
            search = {key: diag.get(key) for key in ('search_status', 'decision_status',
                'fixed_query_denominator', 'evaluated_transform_count', 'hypotheses_admitted',
                'potential_winner_count', 'potential_winner_ids', 'region_identified',
                'finite_candidate_unique', 'transform_certified',
                'best_score', 'competing_regions', 'appearance_reference', 'timing_seconds')}
            search.update(candidate_rejection_counts=dict(Counter(reason for h in candidates
                for reason in h.get('reasons', h.get('rejections', [])))),
                full_prediction=str(stem.with_suffix('.json')))
            images = []
            modes = result['modes'] or [None]
            for rank, mode in enumerate(modes):
                for v, view in enumerate(loaded['views']):
                    name = f'{task["id"]}_dual_{backend}_{view["id"]}_mode{rank}.png'
                    if mode is None:
                        picture = Image.fromarray(rgbs[v])
                    else:
                        picture, counts = region_picture(rgbs[v], surface, mode, v)
                        visibility.append({'task': task['id'], 'backend': backend,
                                           'view': view['id'], 'mode': rank, **counts})
                    # BEGIN identical final GT layer for accepted and diagnostic modes.
                    with_ground_truth(picture, gt_layers[v]).save(assets / name)
                    gt_records[v]['images'].append('assets/'+name)
                    # END dual-view GT display.
                    if result['accepted']:
                        role = '接受的主区域' if rank == 0 else '竞争诊断区域（非主输出）'
                    else:
                        role = '诊断区域' if mode is not None else '无可展示区域'
                    # BEGIN visibility caption for every retained mode in both cameras.
                    images.append({'path': 'assets/'+name,
                                   'caption': f'{view["id"]} · {role} {rank+1} · {gt_captions[v]}'})
                    # END dual-view GT caption.
            card['methods'].append({'scheme': 'dual', 'label': f'修订双视角：局部区域与观测支持 / {backend}',
                                    'panels': [{**row, 'images': images, 'search': search}]})
    summary = _summary(rows)
    write_json(destination / 'evaluation.json', {'read_after_all_predictions_frozen': True,
        'prediction_index': str(output / 'predictions.json'), 'rows': rows,
        'old_results': str(Path(old_results).resolve()), 'new_results': str(Path(new_results).resolve()),
        'baseline_dual_results': None if baseline_dual_results is None else str(baseline_dual_results),
        'baseline_dual_web': None if baseline_dual_web is None else str(baseline_dual_web)})
    write_json(destination / 'summary.json', {'methods': summary,
        'dual_rejection_counts': dict(rejection_counts), 'sample_visibility': visibility,
        'prediction_runtime': runtime['revision'],
        'original_dual_runtime': runtime['original_dual']})
    # BEGIN evaluation-only provenance and exhaustive target-image coverage.
    write_json(destination / 'ground-truth.json', {'read_after_all_predictions_frozen': True,
        'prediction_index': str(output / 'predictions.json'), 'views': ground_truth})
    # END GT provenance; existing prediction/evaluation metrics keep their meanings.
    write_html(destination / 'index.html', cards, summary,
        runtime=runtime,
        baseline_link=None if baseline_dual_web is None else os.path.relpath(baseline_dual_web / 'index.html', destination))
    return summary
# END held-out transform/performance comparison.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--old-results', type=Path, required=True)
    parser.add_argument('--new-results', type=Path, required=True)
    # BEGIN original assets are an explicit read-only comparison input.
    parser.add_argument('--baseline-dual-results', type=Path)
    parser.add_argument('--baseline-dual-web', type=Path)
    parser.add_argument('--prediction-resources', type=Path)
    parser.add_argument('--baseline-prediction-resources', type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output, args.destination, old_results=args.old_results,
                             new_results=args.new_results,
                             baseline_dual_results=args.baseline_dual_results,
                             baseline_dual_web=args.baseline_dual_web,
                             prediction_resources=args.prediction_resources,
                             baseline_prediction_resources=args.baseline_prediction_resources), ensure_ascii=False, indent=2))
    # END comparison arguments.


if __name__ == '__main__':
    main()
