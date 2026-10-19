"""Compare two frozen observed-partial dual-view runs on held-out truth.

Both inputs must use the current prediction schema and exactly the same task,
camera, cache and physical sampling domain. The preceding four-scheme report
is an explicit immutable image input; it is never traversed for other assets.
Inference does not import this module. GT is opened only after both freezes
have been validated. A later adoption decision only rerenders saved cards.
"""

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..experiment import read_archive, restore_prediction
from ..features import write_json
from ..multiview_experiment import _path, load_surface, read_manifest
from ..multiview_visualization import read_rgb, source_picture
from .evaluation import (_center_radius_metric, _dual_region_metrics, _empty, _row,
                         _runtime_summary, _summary, _supervisor_metrics,
                         _transform_evaluation, frozen_index)
from .io import load_observations
from .transform_metrics import affine_reference
from .visualization import ground_truth_layer, region_picture, with_ground_truth, write_html


def _distribution(values):
    """Retain empty domains, rather than inventing a zero error for failures."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Precision summaries require finite measured errors')
    return {'count': len(values),
            # BEGIN all reported distributions retain both average and tails.
            'mean': float(np.mean(values)) if len(values) else None,
            # END complete distribution summary.
            'median': float(np.median(values)) if len(values) else None,
            'p90': float(np.percentile(values, 90)) if len(values) else None,
            'maximum': float(np.max(values)) if len(values) else None}


def precision_summary(rows):
    """Report every primary metric over the complete declared task domain.

    Each native backend keeps its own task denominator. Accepted primary
    regions alone contribute errors; rejected candidates stay in row-level
    diagnostic fields and cannot improve the success statistics.
    """
    answer = []
    for scheme, backend in dict.fromkeys((row['scheme'], row['backend']) for row in rows):
        values = [row for row in rows if (row['scheme'], row['backend']) == (scheme, backend)]
        accepted = [row for row in values if row['accepted']]
        centers = [row['primary_metrics'] for row in accepted if row['primary_metrics'] is not None]
        transforms = [row['transform_metrics'] for row in accepted
                      if row.get('transform_metrics') is not None
                      and row['transform_metrics']['primary_estimate'] is not None]
        rotations = lambda flag: [row['primary_estimate']['rotation_vs_polar_deg'] for row in transforms
                                  if flag is None or row['reference']['truth_is_sim3'] == flag]
        # BEGIN scalar scale error is undefined for an affine truth that is not
        # a Sim(3). Other primary mapping/rotation measurements still retain
        # those tasks. Null values never become fabricated zero-error samples.
        def transform_values(key):
            return [row['primary_estimate'][key] for row in transforms
                    if row['primary_estimate'][key] is not None]

        answer.append({'scheme': scheme, 'backend': backend, 'total': len(values),
            'accepted': len(accepted),
            'correct_surface': sum(row['top1_preferred_support_hit'] for row in centers),
            'status_counts': dict(Counter(row['status'] for row in values)),
            'decision_counts': dict(Counter(row['decision_status'] for row in values)),
            'center_error_mm': _distribution([row['center_error_m']*1000 for row in centers]),
            'center_error_radius_ratio': _distribution([row['center_error_radius_ratio'] for row in centers]),
            'best_sample_reference_distance_mm': _distribution([
                row['top1_best_sample_reference_distance_m']*1000 for row in centers
                if row['top1_best_sample_reference_distance_m'] is not None]),
            'rotation_true_sim3_deg': _distribution(rotations(True)),
            'rotation_anisotropic_polar_deg': _distribution(rotations(False)),
            'rotation_all_polar_deg': _distribution(rotations(None)),
            'scale_absolute_error_percent': _distribution(transform_values('scale_absolute_error_percent')),
            'normal_error_deg': _distribution(transform_values('normal_error_deg')),
            'local_mapping_rmse_mm': _distribution(transform_values('local_mapping_rmse_mm')),
            'context_mapping_rmse_mm': _distribution(transform_values('context_mapping_rmse_mm')),
            'mapped_center_error_mm': _distribution(transform_values('mapped_center_error_mm')),
            'ratio_above_one': sum(row['center_error_radius_ratio'] > 1 for row in centers)})
        # END full-domain precision measurements with explicit valid counts.
    return answer


# BEGIN refreshing derived statistics never reopens predictions, images or GT.
def _write_precision_summary(destination, metrics):
    """Use one explicit current summary schema for evaluation and refresh."""
    write_json(destination/'precision-summary.json', {'methods': metrics,
        'percentile_method': 'numpy percentile linear',
        'center_definition': 'final mode0.center versus held-out demonstration center',
        'radius_source': 'frozen prediction index sampling.sampling_radius_m',
        'radius_ratio_is_acceptance_gate': False, 'backends_are_independent_samples': False,
        'valid_metric_count_denominator': 'Each metric count is shown against all declared tasks; undefined or missing values are not zero.',
        'sampling_support_purity_is_local_gt_coverage': False})


def refresh_summary(destination):
    """Recompute statistics from the saved complete evaluation, without GT I/O."""
    destination = Path(destination).resolve()
    evaluation = json.loads((destination/'evaluation.json').read_text())
    report = json.loads((destination/'report-view.json').read_text())
    if (evaluation['comparison'] != 'dual_view_precision'
            or report['comparison'] != 'dual_view_precision'
            or evaluation['read_after_all_predictions_frozen'] is not True):
        raise ValueError('Summary refresh requires an existing frozen precision evaluation')
    metrics = precision_summary(evaluation['rows'])
    _write_precision_summary(destination, metrics)
    report['precision_metrics'] = metrics
    write_json(destination/'report-view.json', report)
    return metrics
# END saved-evaluation statistics refresh.


def _comparison_inputs(output, baseline, baseline_web):
    """Validate both current freezes before any GT file can be opened."""
    trial, previous = frozen_index(output), frozen_index(baseline)
    for key in ('declared_task_ids', 'backends', 'sampling'):
        if trial[key] != previous[key]:
            raise ValueError(f'Precision comparison requires identical {key}')
    # BEGIN current inference has observed inputs, not a prediction cache root.
    for key in ('index',):
        if Path(trial[key]).resolve() != Path(previous[key]).resolve():
            raise ValueError(f'Precision comparison requires identical {key}')
    previous_tasks = {state['id']: state for state in previous['tasks']}
    for state in trial['tasks']:
        other = previous_tasks[state['id']]
        if state['cases'] != other['cases']:
            raise ValueError('Precision comparison requires identical camera cases')
        left, right = state.get('surface_directory'), other.get('surface_directory')
        if left != right:
            if left is None or right is None:
                raise ValueError('Precision comparison requires the same available target surface')
            left, right = Path(left), Path(right)
            if json.loads((left/'surface.json').read_text()) != json.loads((right/'surface.json').read_text()):
                raise ValueError('Precision comparison requires identical surface metadata')
            with np.load(left/'surface.npz', allow_pickle=False) as a, \
                 np.load(right/'surface.npz', allow_pickle=False) as b:
                if set(a.files) != set(b.files):
                    raise ValueError('Precision comparison requires identical surface arrays')
                for name in a.files:
                    x, y = a[name], b[name]
                    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                        raise ValueError('Precision comparison changed target surface: '+name)
    # END separately saved surfaces must contain exactly the same measurements.
    evaluation = json.loads((baseline_web/'evaluation.json').read_text())
    if (Path(evaluation['prediction_index']).resolve() != baseline/'predictions.json'
            or evaluation.get('read_after_all_predictions_frozen') is not True):
        raise ValueError('Baseline web must name this exact frozen baseline prediction')
    prior_rows = [row for row in evaluation['rows'] if row['scheme'] == 'dual']
    rows = {(row['task'], row['backend']): row for row in prior_rows}
    expected = {(task, backend) for task in trial['declared_task_ids'] for backend in trial['backends']}
    if set(rows) != expected or len(rows) != len(prior_rows):
        raise ValueError('Baseline web does not contain exactly one current dual row per task/backend')
    return trial, previous, rows


def _decision(path):
    """Decisions label the frozen trial; they never select prediction modes."""
    result = ({'status': 'pending', 'reason': '按预先固定的旋转、区域中心及成本门槛审查。'}
              if path is None else json.loads(Path(path).read_text()))
    if result.get('status') not in ('pending', 'adopted', 'not_adopted'):
        raise ValueError('Decision status must be pending, adopted or not_adopted')
    if not isinstance(result.get('reason'), str) or not result['reason'].strip():
        raise ValueError('Decision requires a nonempty explanation')
    return result


def render(destination, *, decision_json=None, historical_pages=()):
    """Refresh decision labels using saved evaluation/cards, without rereading GT."""
    destination = Path(destination).resolve()
    report = json.loads((destination/'report-view.json').read_text())
    if report['comparison'] != 'dual_view_precision':
        raise ValueError('Render-only requires this evaluator\'s saved precision report')
    saved_decision = destination/'decision.json'
    decision = _decision(decision_json if decision_json is not None else
                         saved_decision if saved_decision.is_file() else None)
    trial_label = {'pending': '待评估试验', 'adopted': '已采用试验',
                   'not_adopted': '未采用试验'}[decision['status']]
    cards = report['cards']
    for card in cards:
        for method in card['methods']:
            if method['scheme'] == 'trial':
                method['label'] = f'{trial_label} / {method["panels"][0]["backend"]}'
    summary = report['summary']
    for item in summary:
        if item['scheme'] == 'trial':
            item['label'] = f'{trial_label} / {item["backend"]}'
    # BEGIN explicit review records supplement the preserved comparison links.
    # Relative links are saved in the report so subsequent render-only calls
    # retain the same named documents without a recursive history search.
    for page in historical_pages:
        page = Path(page).resolve()
        if not page.is_file():
            raise FileNotFoundError(page)
        link = {'href': os.path.relpath(page, destination), 'label': f'审查记录：{page.name}'}
        if not any(item['href'] == link['href'] for item in report['historical_links']):
            report['historical_links'].append(link)
    if historical_pages:
        write_json(destination/'report-view.json', report)
    # END explicit immutable review links.
    write_json(destination/'decision.json', decision)
    write_html(destination/'index.html', cards, summary, runtime=report['runtime'], precision={
        'decision': decision, 'historical_links': report['historical_links'],
        'metrics': report['precision_metrics']})
    return summary


def evaluate(output, destination, *, baseline_results, baseline_web,
             prediction_resources=None, baseline_prediction_resources=None,
             decision_json=None, historical_pages=()):
    """Reevaluate only the trial; reuse immutable baseline measurements/images."""
    output, destination = Path(output).resolve(), Path(destination).resolve()
    baseline, baseline_web = Path(baseline_results).resolve(), Path(baseline_web).resolve()
    if output == baseline or destination in (output, baseline, baseline_web):
        raise ValueError('Precision evaluation must not overwrite frozen predictions or the baseline web')
    record, prior_record, prior_rows = _comparison_inputs(output, baseline, baseline_web)
    radius = record['sampling']['sampling_radius_m']
    _center_radius_metric(None, radius)
    _decision(decision_json)
    prior_states = {state['id']: state for state in prior_record['tasks']}
    runtime = {'baseline': _runtime_summary(prior_record, _supervisor_metrics(baseline_prediction_resources, baseline)),
               'trial': _runtime_summary(record, _supervisor_metrics(prediction_resources, output))}
    links = [{'href': os.path.relpath(baseline_web/'index.html', destination), 'label': '当前双视角完整基线网页'}]
    for page in historical_pages:
        page = Path(page).resolve()
        if not page.is_file():
            raise FileNotFoundError(page)
        links.append({'href': os.path.relpath(page, destination), 'label': f'历史对照：{page.parent.name}'})
    _, description = read_manifest(record['index'])
    destination.mkdir(parents=True, exist_ok=True)
    assets = destination/'assets'
    assets.mkdir(exist_ok=True)
    rows, cards, gt_records, visibility = [], [], [], []
    for state in record['tasks']:
        task = {'id': state['id'], 'cases': state['cases']}
        card = {'id': state['id'], 'source': None, 'methods': []}
        cards.append(card)
        try:
            # BEGIN current observations do not require historical prediction caches.
            # BEGIN this evaluator consumes the existing planar benchmark.
            loaded = load_observations(record['index'], task, region_geometry='planar')
            # END explicit geometry declaration.
            # END observation-only loading.
        except Exception as error:
            # Broken observation bindings cannot silently shrink the comparison.
            reason = f'evaluation_observation_binding_failed:{type(error).__name__}:{error}'
            for backend in record['backends']:
                for scheme in ('baseline', 'trial'):
                    row = _row(state['id'], scheme, backend, 'A+B', 'failed', _empty([reason]), None)
                    row['prediction_task_status'] = (state if scheme == 'trial' else prior_states[state['id']])['status']
                    rows.append(row)
                    card['methods'].append({'scheme': scheme, 'label': f'{scheme} / {backend}',
                                           'panels': [{**row, 'images': [], 'search': {}}]})
            continue
        # Source input is drawn once here. Baseline target images are explicit
        # assets in baseline_web; no recursive historical lookup is permitted.
        demo = np.load(_path(description['data'], task['cases'][0]['evidence']), allow_pickle=False)
        source_name = f'{state["id"]}_source.png'
        source_picture(read_rgb(loaded['source_rgb']), demo, loaded['source'],
                       loaded['query_indices'], loaded['query_region']).save(assets/source_name)
        card['source'] = 'assets/'+source_name
        truths, pairs, rgbs, layers, captions, task_gt = [], [], [], [], [], []
        for view in loaded['views']:
            truth_path = Path(view['frame_dir'])/'truth.npz'
            pair_path = _path(description['data'], view['case']['pair_truth'])
            truth = read_archive(truth_path)['support_id_image']
            pair = json.loads(pair_path.read_text())
            reference = np.asarray(pair['target_reference_evidence_center_world'])
            layer, detail = ground_truth_layer(truth == pair['preferred_support_id'], reference,
                view['K'], view['T_world_camera_cv'], center_visible=pair['target_reference_center_visible'])
            caption = {'visible': 'GT 中心可见（实线）', 'occluded': 'GT 中心被遮挡（虚线投影）',
                       'outside_image': 'GT 中心在画面外', 'behind_camera': 'GT 中心在相机后方'}[detail['reference_display']]
            captions.append(caption+f'；GT 支持面可见 {detail["support_visible_pixels"]} 像素')
            truths.append(truth)
            pairs.append(pair)
            layers.append(layer)
            rgbs.append(read_rgb(view['rgb_path']))
            task_gt.append({'task': state['id'], 'view': view['id'], **detail,
                'preferred_support_id': pair['preferred_support_id'], 'truth_file': str(truth_path),
                'pair_truth_file': str(pair_path), 'images': []})
        np.testing.assert_allclose(pairs[0]['target_reference_evidence_center_world'],
                                   pairs[1]['target_reference_evidence_center_world'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(pairs[0]['true_affine_transform'], pairs[1]['true_affine_transform'], rtol=0, atol=1e-10)
        if pairs[0]['preferred_support_id'] != pairs[1]['preferred_support_id']:
            raise ValueError('Target views disagree on the preferred support')
        gt_records.extend(task_gt)
        reference = np.asarray(pairs[0]['target_reference_evidence_center_world'])
        truth_model = affine_reference(pairs[0]['true_affine_transform'])['truth_is_sim3']
        surface = None
        if state.get('surface_directory'):
            surface = load_surface(state['surface_directory'], loaded['views'])
            observation_support = np.empty(len(surface['observation_xyz']), dtype=np.int64)
            for v in range(2):
                indices = np.flatnonzero(surface['observation_view_index'] == v)
                xy = surface['observation_pixels'][indices]
                observation_support[indices] = truths[v][xy[:, 1], xy[:, 0]]
            support = observation_support[surface['representative_observation']]
        for backend in record['backends']:
            prior = copy.deepcopy(prior_rows[(state['id'], backend)])
            prior.update(scheme='baseline', truth_is_sim3=truth_model)
            prior_stem = baseline/state['id']/backend
            prior_result = (json.loads(prior_stem.with_suffix('.json').read_text())
                            if prior['status'] == 'completed' else None)
            if prior_result is not None and prior_result['accepted'] != prior['accepted']:
                raise ValueError('Baseline evaluation acceptance differs from the frozen prediction')
            for key in ('primary_metrics', 'diagnostic_first_mode_metrics'):
                prior[key] = _center_radius_metric(prior[key], radius)
            prior['mode_metrics'] = [_center_radius_metric(metric, radius) for metric in prior['mode_metrics']]
            rows.append(prior)
            images = []
            for rank in range(max(1, len(prior_result['modes']) if prior_result else 0)):
                for v, view in enumerate(loaded['views']):
                    asset = baseline_web/'assets'/f'{state["id"]}_dual_{backend}_{view["id"]}_mode{rank}.png'
                    if not asset.is_file():
                        raise FileNotFoundError(f'Explicit baseline image is missing: {asset}')
                    relative = os.path.relpath(asset, destination)
                    images.append({'path': relative, 'caption': f'{view["id"]} · 当前基线 · {prior["display_status"]} · 区域 {rank+1} · {captions[v]}'})
                    task_gt[v]['images'].append(relative)
            card['methods'].append({'scheme': 'baseline', 'label': f'当前双视角基线 / {backend}',
                'panels': [{**prior, 'images': images, 'search': {'full_prediction': str(prior_stem.with_suffix('.json')),
                    'explicit_baseline_assets_reused_without_copy': True}}]})
            backend_state = state.get('backends', {}).get(backend, state)
            status = backend_state['status']
            stem = output/state['id']/backend
            result = (restore_prediction(stem) if status == 'completed' else
                      _empty(backend_state.get('reasons', [backend_state.get('error', status)])))
            metrics = None if surface is None else _dual_region_metrics(result, surface['xyz'],
                {'support_id': support, 'preferred_support_id': np.asarray(pairs[0]['preferred_support_id'])},
                reference, sampling_radius_m=radius)
            row = _row(state['id'], 'trial', backend, 'A+B', status, result, metrics)
            row['truth_is_sim3'] = truth_model
            row['transform_metrics'] = _transform_evaluation(stem, result, pairs[0]) if status == 'completed' else None
            row['runtime'] = {'task_seconds': state.get('seconds'), 'preparation_seconds': state.get('preparation_seconds'),
                'preparation_stage_seconds': state.get('stage_seconds', {}), 'backend_seconds': backend_state.get('seconds'),
                'native_registration_seconds': result['diagnostics'].get('native_seconds'),
                'timing_seconds': result['diagnostics'].get('timing_seconds', {})}
            rows.append(row)
            images = []
            for rank, mode in enumerate(result['modes'] or [None]):
                for v, view in enumerate(loaded['views']):
                    name = f'{state["id"]}_trial_{backend}_{view["id"]}_mode{rank}.png'
                    if mode is None:
                        picture = Image.fromarray(rgbs[v])
                    else:
                        picture, counts = region_picture(rgbs[v], surface, mode, v)
                        visibility.append({'task': state['id'], 'backend': backend, 'view': view['id'], 'mode': rank, **counts})
                    with_ground_truth(picture, layers[v]).save(assets/name)
                    relative = 'assets/'+name
                    role = ('接受的主区域' if rank == 0 else '竞争诊断区域（非主输出）') if result['accepted'] else '拒绝/缺失诊断'
                    images.append({'path': relative, 'caption': f'{view["id"]} · 试验 · {role} {rank+1} · {captions[v]}'})
                    task_gt[v]['images'].append(relative)
            search = {key: result['diagnostics'].get(key) for key in ('search_status', 'decision_status',
                'fixed_query_denominator', 'evaluated_transform_count', 'hypotheses_admitted',
                'potential_winner_count', 'potential_winner_ids', 'region_identified', 'finite_candidate_unique',
                # BEGIN expose the native surface score decision in evaluation.
                'ranked_hypothesis_ids', 'top_score_hypothesis_ids', 'selected_hypothesis_id',
                'selected_score', 'selection_rule',
                # END no interval-survivor compatibility fields are fabricated.
                'transform_certified', 'best_score', 'competing_regions', 'appearance_reference', 'polishing')}
            search['full_prediction'] = str(stem.with_suffix('.json'))
            card['methods'].append({'scheme': 'trial', 'label': f'精度试验 / {backend}',
                                    'panels': [{**row, 'images': images, 'search': search}]})
    summary, metrics = _summary(rows), precision_summary(rows)
    write_json(destination/'evaluation.json', {'comparison': 'dual_view_precision',
        'read_after_all_predictions_frozen': True, 'prediction_index': str(output/'predictions.json'),
        'baseline_prediction_index': str(baseline/'predictions.json'), 'baseline_web': str(baseline_web), 'rows': rows})
    write_json(destination/'summary.json', {'methods': summary, 'prediction_runtime': runtime,
        'sample_visibility': visibility, 'trial_rejection_counts': dict(Counter(reason for row in rows
            if row['scheme'] == 'trial' for reason in row['reasons']))})
    _write_precision_summary(destination, metrics)
    write_json(destination/'ground-truth.json', {'read_after_all_predictions_frozen': True,
        'prediction_index': str(output/'predictions.json'), 'views': gt_records})
    write_json(destination/'report-view.json', {'comparison': 'dual_view_precision', 'cards': cards,
        'summary': summary, 'precision_metrics': metrics, 'runtime': runtime, 'historical_links': links})
    return render(destination, decision_json=decision_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline-results', type=Path)
    parser.add_argument('--baseline-web', type=Path)
    parser.add_argument('--prediction-resources', type=Path)
    parser.add_argument('--baseline-prediction-resources', type=Path)
    parser.add_argument('--decision-json', type=Path)
    parser.add_argument('--historical-page', type=Path, action='append', default=[])
    # BEGIN mutually exclusive derived-artifact actions require no inference.
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--render-only', action='store_true')
    actions.add_argument('--refresh-summary-only', action='store_true')
    # END explicit derived-only actions.
    args = parser.parse_args()
    if args.refresh_summary_only:
        answer = refresh_summary(args.destination)
    elif args.render_only:
        answer = render(args.destination, decision_json=args.decision_json,
                        historical_pages=args.historical_page)
    else:
        if any(value is None for value in (args.output, args.baseline_results, args.baseline_web)):
            parser.error('--output, --baseline-results and --baseline-web are required unless a derived-only action is selected')
        answer = evaluate(args.output, args.destination, baseline_results=args.baseline_results,
            baseline_web=args.baseline_web, prediction_resources=args.prediction_resources,
            baseline_prediction_resources=args.baseline_prediction_resources,
            decision_json=args.decision_json, historical_pages=args.historical_page)
    print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
