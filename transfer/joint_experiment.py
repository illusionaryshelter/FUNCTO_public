"""Freeze old and independent joint predictions from prepared observed inputs.

The JSON index declares pair directories and grouping for cross-view selection.
This stage never opens evaluation files. Evaluation and HTML belong in a
separate process after the complete prediction index has been frozen.
"""

import argparse
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .core import infer_region, prepare_anchors
from .experiment import feature_observation, store_pairs, store_prediction
from .features import native_region_pairs, write_json
from .joint_features import appearance_evidence, source_queries
from .joint_region import infer_joint_region, select_observation
from .segmented_experiment import archive, SAMPLING


def block_evaluation_reads():
    """Fail loudly if a prediction process tries to open known truth artifacts.

    This supplements the data-only numerical API. The manifest may contain
    evaluation path strings for later reporting; opening those paths here is
    forbidden. The guard is process local and never modifies model software.
    """
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        if (path.name in ('truth.npz', 'evaluation.npz', 'evaluation-audit.json', 'state-checks.json',
                          'intervention-checks.json')
                or 'pair-truth' in path.parts or 'evaluation-scenes' in path.parts):
            raise PermissionError(f'Evaluation-only artifact requested during prediction: {path}')
    sys.addaudithook(audit)


def predict_pair(input_dir, output_dir):
    """Run fixed old backends and one new matcher on exactly the same RGB-D."""
    folder, output = Path(input_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    prepared = json.loads((folder/'preparation.json').read_text())
    if not prepared['prepared']:
        record = {'status': 'missing_input', 'reasons': prepared['reasons'], 'accepted': False,
                  'target_truth_used': False}
        write_json(output/'missing.json', record)
        return {'accepted': False, 'reasons': prepared['reasons'], 'diagnostics': {}, 'modes': []}
    source, target = archive(folder/'source_observation.npz'), archive(folder/'target_observation.npz')
    inputs = json.loads((folder/'inputs.json').read_text())
    features = archive(folder/'features.npz')
    begin = perf_counter()
    native = native_region_pairs(feature_observation(source, inputs['source_crop_xyxy']),
        feature_observation(target, inputs['target_crop_xyxy']), source['region_mask'],
        (features['feature1'], features['feature2']))
    native_ms = 1000*(perf_counter()-begin)
    store_pairs(output/'native_pairs.npz', native)
    controls = {**SAMPLING, 'target_pixels': target['pixels'],
                'source_camera_origin': source['T_world_camera_cv'][:3, 3],
                'target_camera_origin': target['T_world_camera_cv'][:3, 3]}
    initial = []
    for algorithm in ('ransac', 'prosac_lo', 'teaserpp'):
        start = perf_counter()
        anchors = prepare_anchors(source['xyz'], target['xyz'], native['src_anchor_idx'],
            native['tgt_anchor_idx'], mode='direct', registration_algorithm=algorithm,
            correspondence_scores=native['scores'])
        result = infer_region(source['region_mask'], anchors, method='direct_local', **controls)
        result['diagnostics'].update(prediction_ms=1000*(perf_counter()-start),
                                     native_matching_ms=native_ms, target_truth_used=False)
        store_prediction(output/f'old_{algorithm}', result, {})
        if algorithm == 'ransac' and anchors.scale is not None and not anchors.reasons:
            initial.append((anchors.scale, anchors.rotation, anchors.translation))
    begin = perf_counter()
    ids, query_region = source_queries(source)
    evidence = appearance_evidence(source, target, features['feature1'], features['feature2'],
        inputs['source_crop_xyxy'], inputs['target_crop_xyxy'], query_indices=ids)
    matching_ms = 1000*(perf_counter()-begin)
    del features
    gc.collect()
    np.savez_compressed(output/'joint_evidence.npz', **evidence, query_region=query_region)
    begin = perf_counter()
    result = infer_joint_region(source, target, evidence, query_region=query_region,
        initial_transforms=initial, sampling_radius_m=SAMPLING['sampling_radius_m'],
        sampling_spacing_m=SAMPLING['sampling_spacing_m'])
    result['diagnostics'].update(prediction_ms=1000*(perf_counter()-begin), matching_ms=matching_ms,
        native_matching_ms=native_ms, initial_native_consensus_count=len(initial))
    store_prediction(output/'joint', result, {})
    write_json(output/'frozen.json', {'complete': True, 'target_truth_used': False,
        'algorithms': ['old_ransac', 'old_prosac_lo', 'old_teaserpp', 'joint']})
    print(json.dumps({'pair': output.name, 'accepted': result['accepted'],
        'reasons': result['reasons'], 'matching_ms': matching_ms,
        'prediction_ms': result['diagnostics']['prediction_ms']}), flush=True)
    return result


def predict_suite(index, output, selected=None):
    """Freeze every listed pair, then select between each task's observed views."""
    description = json.loads(Path(index).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = description['cases']
    if selected:
        known = {case['id'] for case in cases}
        if not set(selected) <= known:
            raise ValueError('Unknown selected case')
        cases = [case for case in cases if case['id'] in selected]
    records, results, source_records = [], {}, {}
    for case in cases:
        # BEGIN bind cross-view score comparisons to identical source evidence.
        # Task IDs only group comparisons; equality is checked on actual arrays
        # and model-input pixels, so an ID cannot substitute another source.
        input_dir = Path(case['input_dir'])
        source_path = input_dir/'source_observation.npz'
        if source_path.exists():
            current = archive(source_path)
            crop = (input_dir/'source.png').read_bytes()
            if case['task'] in source_records:
                previous, previous_crop = source_records[case['task']]
                if current.keys() != previous.keys() or crop != previous_crop:
                    raise ValueError('Cross-view candidates must share one identical source observation')
                for key in current:
                    np.testing.assert_array_equal(current[key], previous[key])
            else:
                source_records[case['task']] = current, crop
        # END exact source identity validation before comparing view scores.
        results[case['id']] = predict_pair(case['input_dir'], output/case['id'])
        records.append({'id': case['id'], 'task': case['task'], 'view': case['view'],
                        'output_dir': str(output/case['id'])})
    tasks = []
    for task in dict.fromkeys(case['task'] for case in cases):
        group = [case for case in cases if case['task'] == task]
        winner = select_observation([results[case['id']] for case in group])
        tasks.append({'task': task, 'candidates': [case['id'] for case in group],
                      'selected': None if winner is None else group[winner]['id']})
    write_json(output/'predictions.json', {'frozen': True, 'target_truth_used': False,
        'cases': records, 'view_selection': tasks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    block_evaluation_reads()
    predict_suite(args.index, args.output, args.case)


if __name__ == '__main__':
    main()
