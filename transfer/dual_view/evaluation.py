"""Evaluate frozen dual-view predictions and display three separate schemes.

This entry point is intentionally separate from prediction: only this process
opens target truth. Historical outputs are read, never reoptimized or edited.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..evaluation import evaluate_prediction
from ..experiment import read_archive, restore_prediction
from ..features import write_json
from ..multiview_experiment import _path, load_surface, read_manifest
from ..multiview_visualization import read_rgb, single_view_picture, source_picture
from .io import load_observations
# BEGIN shared GT overlay for all three frozen schemes.
from .visualization import ground_truth_layer, region_picture, with_ground_truth, write_html
# END shared GT overlay import.


def frozen_index(output):
    """Require every declared prediction to have a matching terminal record."""
    output = Path(output)
    record = json.loads((output / 'predictions.json').read_text())
    if (record.get('scheme') != 'dual_view_continuous' or record.get('frozen') is not True
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
                    if (diagnostics.get('scheme') != 'dual_view_continuous'
                            or diagnostics.get('native_backend') != backend
                            or diagnostics.get('target_truth_used') is not False):
                        raise ValueError('Frozen prediction scheme/backend/provenance mismatch')
    return record


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
    labels = {'old': '旧方案', 'new': '新方案', 'dual': '双视角方案'}
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


def evaluate(output, destination, *, old_results, new_results):
    """Render frozen modes on original RGB, with no selection from target truth."""
    output, destination = Path(output).resolve(), Path(destination).resolve()
    record = frozen_index(output)
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
            for scheme in ('old', 'new', 'dual'):
                for backend in record['backends']:
                    method = {'scheme': scheme, 'label': f'{scheme} / {backend}', 'panels': []}
                    for view_id in (['A+B'] if scheme == 'dual' else
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
        source_picture(read_rgb(loaded['source_rgb']), demo, source,
                       loaded['query_indices'], loaded['query_region']).save(assets / source_name)
        card = {'id': task['id'], 'source': 'assets/'+source_name, 'methods': []}
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
                    picture = single_view_picture(rgbs[v], view['observation'], result)
                    with_ground_truth(picture, gt_layers[v]).save(assets / name)
                    gt_records[v]['images'].append('assets/'+name)
                    # END old/new GT display; frozen result and metrics remain unchanged.
                    method['panels'].append({**row, 'images': [{'path': 'assets/'+name,
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
            backend_state = state['backends'].get(backend, state)
            status = backend_state['status']
            stem = output / task['id'] / backend
            result = (restore_prediction(stem) if status == 'completed' else
                      _empty(backend_state.get('reasons', [backend_state.get('error', status)])))
            metrics = None
            if surface is not None:
                metrics = evaluate_prediction(result, surface['xyz'],
                    {'support_id': support, 'preferred_support_id': np.asarray(preferred[0])}, references[0])
            row = _row(task['id'], 'dual', backend, 'A+B', status, result, metrics)
            rows.append(row)
            rejection_counts.update(row['reasons'])
            diag = result['diagnostics']
            candidates = diag.get('hypotheses', [])
            search = {key: diag.get(key) for key in ('search_status', 'decision_status',
                'fixed_query_denominator', 'evaluated_transform_count', 'hypotheses_admitted',
                'potential_winner_count', 'potential_winner_ids', 'region_identified',
                'transform_identified', 'best_lower')}
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
                    role = '接受区域' if result['accepted'] else ('诊断区域' if mode is not None else '无可展示区域')
                    # BEGIN visibility caption for every retained mode in both cameras.
                    images.append({'path': 'assets/'+name,
                                   'caption': f'{view["id"]} · {role} {rank+1} · {gt_captions[v]}'})
                    # END dual-view GT caption.
            card['methods'].append({'scheme': 'dual', 'label': f'双视角方案：共同变换与区间判定 / {backend}',
                                    'panels': [{**row, 'images': images, 'search': search}]})
    summary = _summary(rows)
    write_json(destination / 'evaluation.json', {'read_after_all_predictions_frozen': True,
        'prediction_index': str(output / 'predictions.json'), 'rows': rows,
        'old_results': str(Path(old_results).resolve()), 'new_results': str(Path(new_results).resolve())})
    write_json(destination / 'summary.json', {'methods': summary,
        'dual_rejection_counts': dict(rejection_counts), 'sample_visibility': visibility})
    # BEGIN evaluation-only provenance and exhaustive target-image coverage.
    write_json(destination / 'ground-truth.json', {'read_after_all_predictions_frozen': True,
        'prediction_index': str(output / 'predictions.json'), 'views': ground_truth})
    # END GT provenance; existing prediction/evaluation metrics keep their meanings.
    write_html(destination / 'index.html', cards, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--old-results', type=Path, required=True)
    parser.add_argument('--new-results', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output, args.destination, old_results=args.old_results,
                             new_results=args.new_results), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
