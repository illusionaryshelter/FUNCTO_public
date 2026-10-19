"""Freeze the selected TEASER++ dual-view estimator from observed inputs.

Features are prepared MARCO outputs; pairs, surfaces and candidate results are
computed freshly. Evaluation is separate after every task has a terminal record.
"""
import argparse
import gc
from pathlib import Path
import resource
from time import perf_counter
import traceback
import numpy as np
from ..experiment import store_pairs, store_prediction
from ..features import write_json
from ..multiview_experiment import block_evaluation_reads, read_manifest, save_surface
from .io import load_feature_pairs, load_observations
from .pipeline import InsufficientSource, prepare, run


def predict(index, output, *, tasks=None):
    """Run tasks serially, retaining missing, failed and rejected outcomes."""
    path, description = read_manifest(index)
    declared = description['tasks']
    if tasks:
        if not set(tasks) <= {task['id'] for task in declared}:
            raise ValueError('Unknown selected task')
        declared = [task for task in declared if task['id'] in tasks]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = dict(scheme='dual_view_observed_partial', frozen=False, target_truth_used=False,
        index=str(path), backends=['teaserpp'],
        declared_task_ids=[task['id'] for task in declared],
        sampling=dict(sampling_radius_m=.03, sampling_spacing_m=.005), tasks=[],
        online_scope='resident observations and CPU feature pairs through complete inference',
        online_excludes=['model', 'observation and crop preparation', 'source query selection',
                         'native code preparation', 'file reads and writes', 'evaluation'])
    write_json(output/'predictions.json', record)
    # BEGIN this recorded benchmark explicitly uses the planar estimator.
    record['runtime'] = prepare('planar')
    # END explicit backend preparation matches load_observations below.
    for task in declared:
        folder = output/task['id']
        folder.mkdir()
        state = dict(id=task['id'], cases=task['cases'], status='running',
                     target_truth_used=False, backends={})
        record['tasks'].append(state)
        write_json(output/'predictions.json', record)
        started, stage = perf_counter(), 'load_observations'
        loaded = features = returned = result = source = None
        try:
            # BEGIN this existing planar experiment declares its original model.
            loaded = load_observations(path, task, region_geometry='planar')
            # END explicit experimental model selection.
            state['input_availability'] = [dict(view=v['id'], geometry=True,
                entity=v['observation'] is not None, reasons=v['missing_reasons'])
                for v in loaded['views']]
            if not any(v['observation'] is not None for v in loaded['views']):
                state.update(status='missing_input', reasons=['no_prepared_entity_feature_pair'])
            else:
                stage = 'load_features'
                features = load_feature_pairs(loaded)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 >= 7*1024**3:
                    raise MemoryError('Resident inputs reach the 7 GiB process stop threshold')
                stage = 'online_inference'
                returned = run(loaded, features, **record['sampling'])
                result = returned['result']
                state.update(surface_directory=str(folder),
                    stage_seconds=returned['stages_seconds'],
                    continuous_online_seconds=returned['continuous_seconds'])
                stage = 'save_prediction'
                save_surface(folder, returned['surface'])
                source = returned['source']
                np.savez_compressed(folder/'source_queries.npz', **{
                    key: value for key, value in source.items() if isinstance(value, np.ndarray)})
                write_json(folder/'source_geometry.json', {
                    key: value for key, value in source.items() if not isinstance(value, np.ndarray)})
                for view, pair in returned['native_pairs'].items():
                    store_pairs(folder/(view+'_native_pairs.npz'), pair)
                store_prediction(folder/'teaserpp', result, {})
                state['backends']['teaserpp'] = dict(status='completed',
                    accepted=bool(result['accepted']), reasons=result['reasons'],
                    decision_status=result['status'], search_status=result['diagnostics']['search_status'],
                    timing_seconds=result['diagnostics']['timing_seconds'],
                    warnings=result['diagnostics']['quality_warnings'],
                    seconds=returned['stages_seconds']['native_registration']
                            +returned['stages_seconds']['candidate_inference'])
                state['status'] = 'completed'
        except InsufficientSource as error:
            state.update(status='missing_input', reasons=error.reasons, model_applicable=False)
        except Exception as error:
            state.update(status='failed', stage=stage, error_type=type(error).__name__, error=str(error))
            (folder/'error.txt').write_text(traceback.format_exc())
        state['seconds'] = perf_counter()-started
        write_json(folder/'frozen.json', state)
        write_json(output/'predictions.json', record)
        print(f"{task['id']}: {state['status']} {state['backends']}", flush=True)
        loaded = features = returned = result = source = None
        gc.collect()
    record['frozen'] = True
    write_json(output/'predictions.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', action='append')
    args = parser.parse_args()
    block_evaluation_reads()
    predict(args.index, args.output, tasks=args.task)


if __name__ == '__main__':
    main()
