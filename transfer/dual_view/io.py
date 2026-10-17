"""Observed inputs for the selected TEASER++ dual-view transfer scheme.

An index supplies frame/crop/cache paths; no task identity changes computation.
Correspondences are calculated from current features by the numerical pipeline;
historical predictions and target annotations are not required inputs.
"""

import json
from pathlib import Path
import zipfile

import numpy as np

from ..core import prepare_anchors
from ..experiment import read_archive
from ..joint_features import source_queries
from ..multiview_experiment import (_check_observation, _path, _read_frame,
                                    load_surface, read_manifest)
from ..multiview_geometry import _view_arrays
from .surface import build_target_surface


# BEGIN current observation loading, independent of historical prediction caches.
def load_observations(index, task):
    """Bind two target observations to one unchanged source demonstration."""
    path, description = read_manifest(index)
    data = Path(description['data'])
    frames = json.loads((data / 'manifest.json').read_text())['views']
    source, signature, views = None, None, []
    source_depth = source_rgb = source_crop = None
    for case in sorted(task['cases'], key=lambda item: item['view']):
        folder = _path(path.parent, case['input_dir'])
        prepared = json.loads((folder / 'preparation.json').read_text())
        chart = json.loads((folder / 'inputs.json').read_text())
        current = read_archive(folder / 'source_observation.npz')
        source_dir, depth, calibration = _read_frame(data, frames[case['source']])
        _check_observation(current, depth, calibration)
        evidence = np.load(_path(data, case['evidence']), allow_pickle=False)
        xy = current['pixels']
        np.testing.assert_array_equal(current['region_mask'], evidence[xy[:, 1], xy[:, 0]])
        current_signature = (chart['source_crop_xyxy'], (folder / 'source.png').read_bytes())
        if source is None:
            source, signature, source_depth = current, current_signature, depth
            source_rgb, source_crop = str(source_dir / 'rgb.png'), chart['source_crop_xyxy']
        else:
            if source.keys() != current.keys() or signature != current_signature:
                raise ValueError('Paired views must retain identical source arrays, crop and RGB')
            for key in source:
                np.testing.assert_array_equal(source[key], current[key])
            np.testing.assert_array_equal(source_depth, depth)
        target_dir, depth, calibration = _read_frame(data, frames[case['target']])
        available = bool(prepared['target']['prepared'])
        reasons = list(prepared['target']['reasons'])
        target = read_archive(folder / 'target_observation.npz') if available else None
        if target is not None:
            _check_observation(target, depth, calibration)
            if not np.asarray(target['entity_member'], dtype=bool).all():
                raise ValueError('Target entity observations contain nonentity points')
        elif not reasons:
            raise ValueError('Missing target preparation must state its reason')
        views.append({'id': case['view'], 'case': case, 'input_dir': str(folder),
            'K': np.asarray(calibration['K']),
            'T_world_camera_cv': np.asarray(calibration['T_world_camera_cv']),
            'depth_m': depth, 'observation': target, 'missing_reasons': reasons,
            'crop_xyxy': chart.get('target_crop_xyxy'), 'frame_dir': str(target_dir),
            'rgb_path': str(target_dir / 'rgb.png')})
    query, region = source_queries(source)
    # Both target views share these identities derived from the actual source.
    return {'source': source, 'source_depth': source_depth, 'source_rgb': source_rgb,
            'source_crop': source_crop, 'query_indices': query, 'query_region': region,
            'views': views, 'task': task}
# END source equality is checked on arrays, crop and RGB before query selection.


def target_surface(loaded, cached_directory=None):
    """Build geometry or explicitly validate a supplied frozen geometry cache.

    Cache reuse is a declared execution choice. A stale cache raises; it never
    silently switches inputs. Complete calibrated depth pixels and own-view
    entity labels are compared, including observations outside either entity.
    """
    views = loaded['views']
    if cached_directory is None:
        return build_target_surface(views, sampling_radius_m=.03)
    surface = load_surface(cached_directory, views)
    for name, value in (('sampling_radius_m', .03),
                        ('surface_height_spacing_factor', 2.),
                        ('normal_min_agreement', .65)):
        if surface[name] != value:
            raise ValueError(f'Frozen geometry uses a different {name}')
    for view_index, view in enumerate(views):
        xyz, pixels, entity, *_ = _view_arrays(view)
        ids = np.flatnonzero(surface['observation_view_index'] == view_index)
        np.testing.assert_array_equal(surface['observation_pixels'][ids], pixels)
        np.testing.assert_allclose(surface['observation_xyz'][ids], xyz, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(surface['observation_entity_member'][ids], entity)
    # BEGIN independent dual-view cache structural/provenance validation.
    # Matching the raw RGB-D observations is necessary but insufficient: a
    # stale representative index could otherwise point the new world sampler
    # at a different measurement while the raw observations still match.
    observation_count, element_count = len(surface['observation_xyz']), len(surface['xyz'])
    representative = np.asarray(surface['representative_observation'])
    mapping = np.asarray(surface['observation_to_element'])
    observation_view = np.asarray(surface['observation_view_index'])
    if (representative.shape != (element_count,) or representative.dtype.kind not in 'iu'
            or mapping.shape != (observation_count,) or mapping.dtype.kind not in 'iu'
            or observation_view.shape != (observation_count,) or observation_view.dtype.kind not in 'iu'
            or np.any(representative < 0) or np.any(representative >= observation_count)
            or np.any(mapping < 0) or np.any(mapping >= element_count)
            or np.any(observation_view < 0) or np.any(observation_view > 1)):
        raise ValueError('Frozen surface has invalid observation/representative index domains')
    np.testing.assert_array_equal(mapping[representative], np.arange(element_count))
    for name in ('xyz', 'normals', 'spacing', 'valid'):
        observation_values = np.asarray(surface[f'observation_{name}'])
        if len(observation_values) != observation_count:
            raise ValueError(f'Frozen observation_{name} is not observation aligned')
        np.testing.assert_array_equal(surface[name], observation_values[representative])
    for name, size in (('valid', element_count), ('entity_member', element_count),
                       ('observation_valid', observation_count),
                       ('observation_entity_member', observation_count)):
        values = np.asarray(surface[name])
        if values.dtype != np.bool_ or values.shape != (size,):
            raise ValueError(f'Frozen {name} must be a Boolean aligned mask')
    normal = np.asarray(surface['observation_normals'])
    spacing = np.asarray(surface['observation_spacing'])
    valid = surface['observation_valid']
    if (normal.shape != (observation_count, 3) or spacing.shape != (observation_count,)
            or not np.isfinite(normal[valid]).all() or not np.isfinite(spacing[valid]).all()
            or np.any(spacing[valid] <= 0)
            or not np.allclose(np.linalg.norm(normal[valid], axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Frozen valid observations need unit normals and positive spacing')
    member_counts = np.bincount(mapping.astype(np.int64) * 2 + observation_view.astype(np.int64),
                               minlength=2 * element_count).reshape(element_count, 2)
    if np.any(member_counts > 1):
        raise ValueError('A frozen element contains multiple measurements from one view')
    np.testing.assert_array_equal(surface['member_views'], member_counts > 0)
    entity = np.zeros(element_count, bool)
    np.logical_or.at(entity, mapping, surface['observation_entity_member'])
    np.testing.assert_array_equal(surface['entity_member'], entity)
    if surface['adjacency'].shape != (element_count, element_count):
        raise ValueError('Frozen adjacency must use the complete surface-element domain')
    # END independent dual-view cache structural/provenance validation.
    return surface


# BEGIN observed-support revision: model bounds are no longer the objective.
def load_feature_pairs(loaded):
    """Read current feature pairs with an explicit decompressed allocation cap."""
    pairs = {}
    for view in loaded['views']:
        if view['observation'] is None:
            continue
        path = Path(view['input_dir']) / 'features.npz'
        with zipfile.ZipFile(path) as archive:
            size = sum(archive.getinfo(name+'.npy').file_size
                       for name in ('feature1', 'feature2'))
            if size > 512*1024**2:
                raise MemoryError('One feature pair exceeds the 512 MiB input budget')
        with np.load(path, allow_pickle=False) as archive:
            pairs[view['id']] = (archive['feature1'], archive['feature2'])
    return pairs


def appearance_field(loaded, surface, feature_pairs):
    """Convert this call's resident feature pairs into compact response maps."""
    from .features import ResponseField, prepare_view, source_descriptors
    available = [view for view in loaded['views'] if view['observation'] is not None]
    if not available:
        raise ValueError('No prepared feature pair supplies the fixed source descriptors')
    queries = source_descriptors(loaded['source'], loaded['query_indices'],
                                 feature_pairs[available[0]['id']][0], loaded['source_crop'])
    prepared = []
    for view in loaded['views']:
        if view['observation'] is None:
            prepared.append(prepare_view(queries, None, crop_xyxy=None,
                                         observation=None, view_id=view['id']))
            continue
        prepared.append(prepare_view(queries, feature_pairs[view['id']][1],
            crop_xyxy=view['crop_xyxy'], observation=view['observation'], view_id=view['id']))
    return ResponseField(surface, queries, prepared)
# END observed-support response construction.


# BEGIN fixed TEASER++ registration from newly computed native pairs.
def native_transforms(loaded, native_pairs):
    """Compute TEASER++ proposals from this call's actual matching outputs."""
    backend = 'teaserpp'
    proposals, records = [], []
    for view_index, view in enumerate(loaded['views']):
        record = {'view': view['id'], 'backend': backend, 'available': False}
        records.append(record)
        if view['observation'] is None:
            record['reasons'] = list(view['missing_reasons'])
            continue
        pairs = native_pairs[view['id']]
        anchors = prepare_anchors(loaded['source']['xyz'], view['observation']['xyz'],
            pairs['src_anchor_idx'], pairs['tgt_anchor_idx'], mode='direct',
            registration_algorithm=backend, correspondence_scores=pairs['scores'])
        record.update(reasons=list(anchors.reasons), diagnostics=anchors.diagnostics)
        # The dual-view estimator validates this transform again against all Q.
        # An anchor quality decision must not delete a legal native transform.
        if anchors.scale is not None:
            proposals.append({'transform': (anchors.scale, anchors.rotation, anchors.translation),
                              'origin': f'native_{backend}', 'view_index': view_index})
            record['available'] = True
    return proposals, records
# END unavailable views retain their reasons and never change the backend.
