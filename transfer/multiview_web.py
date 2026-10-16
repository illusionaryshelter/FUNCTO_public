"""Export an offline, image-first viewer of frozen joint-surface predictions.

This module reads saved inference and evaluation artifacts only. Historical
single-view baselines are rendered from their unchanged observations/results.
No model, matching, registration or region inference is called by the viewer.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .experiment import read_archive, restore_prediction
from .features import write_json
from .multiview_experiment import (BASELINES, _path, frozen_index, load_inputs,
                                  load_surface, read_manifest)
from .multiview_visualization import (joint_region_picture, read_rgb, save_picture,
                                     single_view_picture, source_picture)


# BEGIN frozen joint-surface viewer; old mean-fusion payload is retired.
BASELINE_LABELS = {'old_ransac': '旧方案 · RANSAC',
                   'old_prosac_lo': '旧方案 · PROSAC + LO',
                   'old_teaserpp': '旧方案 · TEASER++',
                   'joint': '新单视角方案 · 6/12 点提案'}


def _relative(path, destination):
    return Path(os.path.relpath(Path(path).resolve(), Path(destination).parent.resolve())).as_posix()


def _picture(image, path, destination, caption):
    save_picture(image, path)
    return {'src': _relative(path, destination), 'caption': caption}


def _historical(view, rgb, assets, destination):
    """Read exact old/new single-view decisions, including rejected modes."""
    records = []
    folder = Path(view['cache_dir'])
    for method in BASELINES:
        record = {'method': method, 'label': BASELINE_LABELS[method], 'view': view['id'],
                  'accepted': False, 'warnings': [], 'reasons': [], 'metrics': None}
        missing = folder / 'missing.json'
        if missing.is_file():
            stored = json.loads(missing.read_text())
            result = {'accepted': False, 'reasons': stored['reasons'],
                      'diagnostics': {'quality_warnings': []}, 'modes': []}
            record.update(status='missing_input', reasons=stored['reasons'])
        elif (folder / f'{method}.json').is_file() and (folder / f'{method}.npz').is_file():
            result = restore_prediction(folder / method)
            record.update(status='completed', accepted=result['accepted'], reasons=result['reasons'],
                          warnings=result['diagnostics'].get('quality_warnings', []))
            evaluation = folder / f'{method}_evaluation.json'
            if evaluation.is_file():
                record['metrics'] = json.loads(evaluation.read_text())['metrics']
        else:
            result = {'accepted': False, 'reasons': ['historical_artifact_unavailable'],
                      'diagnostics': {'quality_warnings': []}, 'modes': []}
            record.update(status='not_run', reasons=result['reasons'])
        caption = f'{record["label"]} · {view["id"]} · 历史冻结结果'
        image = single_view_picture(rgb, view['observation'], result)
        record['image'] = _picture(image, assets / f'{view["case"]["id"]}_{method}.png', destination, caption)
        records.append(record)
    return records


def export_web(results, destination):
    """Build one local HTML and PNG folder without modifying frozen evidence."""
    results, destination = Path(results).resolve(), Path(destination).resolve()
    record = frozen_index(results)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assets = destination.parent / f'{destination.stem}-assets'
    assets.mkdir(exist_ok=True)
    _, description = read_manifest(record['index'])
    data = Path(description['data'])
    frame_manifest = json.loads((data / 'manifest.json').read_text())
    evaluation_path = results / 'evaluation.json'
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.is_file() else None
    if evaluation is not None and evaluation.get('read_after_all_predictions_frozen') is not True:
        raise ValueError('Viewer evaluation must come after the frozen prediction barrier')
    metrics = {} if evaluation is None else {row['id']: row for row in evaluation['tasks']}
    cards = []
    for state in record['tasks']:
        task = {'id': state['id'], 'cases': state['cases']}
        first = task['cases'][0]
        source_folder = _path(data, frame_manifest['views'][first['source']]['path'])
        source_rgb = read_rgb(source_folder / 'rgb.png')
        supplied_region = np.load(_path(data, first['evidence']), allow_pickle=False)
        card = {key: state[key] for key in ('id', 'status', 'accepted', 'reasons', 'warnings')}
        card.update(group=first.get('group', ''), layer=first.get('layer', ''),
                    variation=first.get('variation', ''), input_availability=state.get('input_availability', []),
                    elapsed_ms=state['elapsed_ms'], evaluation=metrics.get(state['id']),
                    source=None, source_region=None, raw_views=[], modes=[], baselines=[], diagnostics={})
        loaded = None
        try:
            loaded = load_inputs(record['index'], task, record['cache_root'])
        except Exception as error:
            # A failed input remains a visible task. Original RGB/demonstration
            # is still useful even when a stale entity/cache cannot be restored.
            card['visualization_error'] = f'{type(error).__name__}: {error}'
        card['source'] = _picture(source_picture(source_rgb, supplied_region,
            None if loaded is None else loaded['source'],
            None if loaded is None else loaded['query_indices'],
            None if loaded is None else loaded['query_region']),
            assets / f'{task["id"]}_source.png', destination,
            '源示教区域（红）；区域查询（黄）／同实体上下文查询（青）')
        card['source_region'] = _picture(source_picture(source_rgb, supplied_region),
            assets / f'{task["id"]}_source_region.png', destination, '源示教区域（红）；查询点已隐藏')
        if loaded is not None:
            views = loaded['views']
            rgbs = [read_rgb(view['rgb_path']) for view in views]
        else:
            views, rgbs = [], []
            for case in task['cases']:
                folder = _path(data, frame_manifest['views'][case['target']]['path'])
                rgbs.append(read_rgb(folder / 'rgb.png'))
                views.append({'id': case['view'], 'case': case, 'observation': None,
                              'cache_dir': str(Path(record['cache_root']) / case['id'])})
        for view, rgb in zip(views, rgbs):
            card['raw_views'].append(_picture(source_picture(rgb, np.zeros(rgb.shape[:2], bool)),
                assets / f'{view["case"]["id"]}_raw.png', destination, f'目标 {view["id"]} 原始 RGB'))
            card['baselines'].extend(_historical(view, rgb, assets, destination))
        if state['status'] == 'completed':
            if loaded is None:
                raise ValueError(f'Completed prediction cannot restore its input identity: {state["id"]}')
            surface = load_surface(results / task['id'], views)
            result = restore_prediction(results / task['id'] / 'prediction')
            if (result['accepted'] != state['accepted'] or result['reasons'] != state['reasons']
                    or result['diagnostics'].get('quality_warnings', []) != state['warnings']):
                raise ValueError('Viewer state disagrees with its frozen prediction')
            card['diagnostics'] = result['diagnostics']
            card['surface_diagnostics'] = surface.get('diagnostics', {})
            for mode_index, mode in enumerate(result['modes']):
                item = {'index': mode_index, 'type': mode['type'],
                        'executable': bool(result['accepted'] and mode['type'] == 'primary'),
                        'region_elements': int(np.sum(mode['mask'])), 'sample_count': int(len(mode['sample_indices'])),
                        'images': [], 'region_images': [], 'visibility': []}
                for view_index, (view, rgb) in enumerate(zip(views, rgbs)):
                    image, counts = joint_region_picture(rgb, surface, mode, view_index)
                    caption = (f'联合区域 · {view["id"]} · '
                               f'可见实际采样 {counts["visible_samples"]}/{counts["sample_count"]}')
                    item['images'].append(_picture(image,
                        assets / f'{task["id"]}_region_{mode_index}_{view_index}.png', destination, caption))
                    region_image, _ = joint_region_picture(rgb, surface, mode, view_index, show_samples=False)
                    item['region_images'].append(_picture(region_image,
                        assets / f'{task["id"]}_region_only_{mode_index}_{view_index}.png', destination,
                        f'联合区域 · {view["id"]} · 采样点已隐藏'))
                    item['visibility'].append({'view': view['id'], **counts})
                card['modes'].append(item)
            card['prediction_link'] = _relative(results / task['id'] / 'prediction.json', destination)
        card['state_link'] = _relative(results / task['id'] / 'frozen.json', destination)
        cards.append(card)
    payload = {'title': '联合目标表面 · 区域迁移验证', 'frozen': record['frozen'],
               'native_backend': record['native_backend'], 'sampling': record['sampling'],
               'cards': cards, 'prediction_link': _relative(results / 'predictions.json', destination),
               'evaluation_link': _relative(evaluation_path, destination) if evaluation is not None else None}
    write_json(destination.with_suffix('.json'), payload)
    template = Path(__file__).with_suffix('.html').read_text()
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    destination.write_text(template.replace('__PAYLOAD__', encoded))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_web(args.results, args.output)


if __name__ == '__main__':
    main()
# END frozen joint-surface viewer.
