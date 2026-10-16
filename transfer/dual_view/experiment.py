"""Serial, truth-isolated experiment entry point for the dual-view scheme.

Use ``python -m transfer.dual_view.experiment --help``. Raw model outputs,
source demonstration and target depth are observed inputs. Evaluation is a
separate process after all declared tasks/backends have terminal records.
"""

import argparse
import gc
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from ..experiment import store_prediction
from ..features import write_json
from ..multiview_experiment import block_evaluation_reads, read_manifest, save_surface
from .geometry import prepare_source
from .io import (appearance_field, layernorm_parameters, load_observations,
                 native_transforms, target_surface)


class InsufficientSource(ValueError):
    """Observed source inputs cannot instantiate the declared planar model."""


def predict(index, cache_root, checkpoint, output, *, surface_cache=None,
            tasks=None, backends=('ransac', 'teaserpp')):
    """Freeze every result, including failures; never tune from target truth."""
    from .inference import infer
    path, description = read_manifest(index)
    declared = description['tasks']
    if tasks:
        if not set(tasks) <= {task['id'] for task in declared}:
            raise ValueError('Unknown selected task')
        declared = [task for task in declared if task['id'] in tasks]
    if not backends or len(set(backends)) != len(backends) or not set(backends) <= {'ransac', 'teaserpp'}:
        raise ValueError('Supply unique declared native backends')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {'scheme': 'dual_view_continuous', 'frozen': False, 'target_truth_used': False,
        'index': str(path), 'cache_root': str(Path(cache_root).resolve()),
        'checkpoint': str(Path(checkpoint).resolve()), 'backends': list(backends),
        'declared_task_ids': [task['id'] for task in declared],
        'surface_cache': None if surface_cache is None else str(Path(surface_cache).resolve()),
        'sampling': {'sampling_radius_m': .03, 'sampling_spacing_m': .005}, 'tasks': []}
    write_json(output / 'predictions.json', record)
    gamma, beta = layernorm_parameters(checkpoint)
    for task in declared:
        folder = output / task['id']
        folder.mkdir()
        state = {'id': task['id'], 'cases': task['cases'], 'status': 'running',
                 'target_truth_used': False, 'backends': {}}
        record['tasks'].append(state)
        write_json(output / 'predictions.json', record)
        started, stage = perf_counter(), 'load_observations'
        loaded = surface = source_pack = appearance = None
        try:
            loaded = load_observations(path, task, cache_root)
            state['input_availability'] = [{'view': view['id'], 'geometry': True,
                'entity': view['observation'] is not None, 'reasons': view['missing_reasons']}
                for view in loaded['views']]
            if not any(view['observation'] is not None for view in loaded['views']):
                state.update(status='missing_input', reasons=['no_prepared_entity_feature_pair'])
            else:
                stage = 'target_surface'
                cached = None if surface_cache is None else Path(surface_cache) / task['id']
                surface = target_surface(loaded, cached)
                if cached is None:
                    save_surface(folder, surface)
                    state['surface_directory'] = str(folder)
                else:
                    state['surface_directory'] = str(cached.resolve())
                stage = 'source_geometry'
                source_pack = prepare_source(loaded['source'], loaded['source_depth'],
                    query_indices=loaded['query_indices'], query_region=loaded['query_region'])
                np.savez_compressed(folder / 'source_queries.npz', **{
                    key: value for key, value in source_pack.items() if isinstance(value, np.ndarray)})
                write_json(folder / 'source_geometry.json', {
                    key: value for key, value in source_pack.items() if not isinstance(value, np.ndarray)})
                if source_pack['reasons']:
                    raise InsufficientSource(';'.join(source_pack['reasons']))
                stage = 'raw_response_maps'
                appearance = appearance_field(loaded, surface, gamma, beta)
                state['preparation_seconds'] = perf_counter() - started
                for backend in backends:
                    backend_started = perf_counter()
                    stage = f'native_and_inference_{backend}'
                    try:
                        native, native_records = native_transforms(loaded, backend)
                        result = infer(source_pack, surface, appearance,
                            initial_transforms=native, **record['sampling'])
                        result['diagnostics'].update(scheme='dual_view_continuous',
                            native_backend=backend, native_proposals=native_records,
                            target_truth_used=False, numerical_bounds_are_formal_fp_certificate=False,
                            input_missing_views=[view['id'] for view in loaded['views']
                                                 if view['observation'] is None])
                        store_prediction(folder / backend, result, {})
                        state['backends'][backend] = {'status': 'completed',
                            'accepted': bool(result['accepted']), 'reasons': result['reasons'],
                            'decision_status': result.get('status'),
                            'search_status': result['diagnostics'].get('search_status'),
                            'warnings': result['diagnostics'].get('quality_warnings', []),
                            'seconds': perf_counter() - backend_started}
                        del result, native
                    except Exception as error:
                        state['backends'][backend] = {'status': 'failed', 'accepted': False,
                            'stage': stage, 'error_type': type(error).__name__, 'error': str(error),
                            'seconds': perf_counter() - backend_started}
                        (folder / f'{backend}_error.txt').write_text(traceback.format_exc())
                    write_json(output / 'predictions.json', record)
                state['status'] = 'completed'
        except InsufficientSource:
            state.update(status='missing_input', reasons=list(source_pack['reasons']),
                         model_applicable=False)
        except Exception as error:
            state.update(status='failed', stage=stage, error_type=type(error).__name__, error=str(error))
            (folder / 'error.txt').write_text(traceback.format_exc())
        state['seconds'] = perf_counter() - started
        write_json(folder / 'frozen.json', state)
        write_json(output / 'predictions.json', record)
        print(f"{task['id']}: {state['status']} {state['backends']} ({state['seconds']:.1f}s)", flush=True)
        del loaded, surface, source_pack, appearance
        gc.collect()
    record['frozen'] = True
    write_json(output / 'predictions.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--surface-cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', action='append')
    parser.add_argument('--backend', action='append', choices=('ransac', 'teaserpp'))
    args = parser.parse_args()
    block_evaluation_reads()
    predict(args.index, args.cache_root, args.checkpoint, args.output,
            surface_cache=args.surface_cache, tasks=args.task,
            backends=args.backend or ('ransac', 'teaserpp'))


if __name__ == '__main__':
    main()
