"""Entity-only MARCO evidence for independent, single-source joint transfer.

The model and its crop conventions stay unchanged. This module retains several
appearance peaks instead of turning each response into one mandatory match.
It accepts no task names, target labels, simulator geometry, or true transform.
"""

import numpy as np

from .core import fps_indices
from .experiment import feature_observation
from .features import sample_descriptors
from .multiview_features import centered_rank_maps, rank_sampling_plan, sample_rank_maps


def source_queries(source, region_count=32, context_count=32):
    """Keep demonstration and strictly entity-internal context identities apart.

    Independent demonstration pixels outside an imperfect entity segmentation
    remain legitimate task inputs. They never become new context observations.
    Query selection depends only on the one source and is identical for every
    target view, which keeps cross-view score weights directly comparable.
    """
    points = np.asarray(source['xyz'])
    region, entity = np.asarray(source['region_mask']), np.asarray(source['entity_member'])
    if any(mask.dtype != np.bool_ or mask.shape != (len(points),) for mask in (region, entity)):
        raise ValueError('Source masks must be Boolean and point aligned')
    ids = []
    for mask, count in ((region, region_count), (entity & ~region, context_count)):
        bank = np.flatnonzero(mask)
        ids.append(bank[fps_indices(points[bank], count)])
    return np.concatenate(ids), np.r_[np.ones(len(ids[0]), bool), np.zeros(len(ids[1]), bool)]


def appearance_evidence(source, target, source_feature, target_feature,
                        source_crop, target_crop, *, query_indices, peak_count=4):
    """Rank full entity response maps, then sample only observed depth pixels.

    Costs are one minus the tie-aware empirical response percentile. Constant
    responses cost 0.5, and unsupported interpolation costs 1.0. Peaks must be
    separated by at least three feature cells; native Gaussian decoding is an
    additional observed candidate, never snapped from background onto a mask.
    The full [query, observed-target] cost field is returned for frozen scoring.
    """
    import torch
    from local_models.marco import RESOLUTION, allowed_grid, masked_decode, resize_scales

    if not np.all(np.asarray(target['entity_member'], dtype=bool)):
        raise ValueError('Every target observation must belong to the entity mask')
    src = feature_observation(source, source_crop)
    tgt = feature_observation(target, target_crop)
    descriptors = sample_descriptors(source_feature, src['pixels'][query_indices],
                                     src['object_mask'].shape[::-1])
    mask = np.zeros_like(tgt['object_mask'], dtype=bool)
    xy = tgt['pixels']
    mask[xy[:, 1], xy[:, 0]] = True
    allowed = allowed_grid(mask, target_feature.shape[-2:])
    if not allowed.any():
        raise ValueError('Target entity has no supported feature cells')
    height, width = target_feature.shape[-2:]
    feature_xy = xy * resize_scales(mask.shape[::-1]) * [width-1, height-1] / (RESOLUTION-1)
    plan = rank_sampling_plan(feature_xy, allowed)
    point_at_pixel = np.full(mask.shape, -1, dtype=np.int64)
    point_at_pixel[xy[:, 1], xy[:, 0]] = np.arange(len(xy))
    costs = np.empty((len(query_indices), len(xy)), dtype=np.float32)
    peaks = np.full((len(query_indices), peak_count+1), -1, dtype=np.int64)
    target_tensor = torch.from_numpy(target_feature)
    with torch.inference_mode():
        for start in range(0, len(query_indices), 8):
            values = torch.from_numpy(descriptors[start:start+8])[None]
            logits = torch.bmm(values, target_tensor.flatten(2)).reshape(1, -1, height, width)
            ranks = centered_rank_maps(logits[0].numpy(), allowed)
            batch = (1-sample_rank_maps(ranks, plan)) / 2
            batch[:, ~plan['supported']] = 1
            costs[start:start+len(batch)] = batch
            decoded = masked_decode(logits, allowed)[0].numpy() / resize_scales(mask.shape[::-1])
            rounded = np.rint(decoded).astype(int)
            for offset, row in enumerate(batch):
                active = plan['supported'].copy()
                for column in range(peak_count):
                    if not active.any():
                        break
                    index = int(np.argmin(np.where(active, row, np.inf)))
                    peaks[start+offset, column] = index
                    active &= np.linalg.norm(feature_xy-feature_xy[index], axis=1) >= 3
                x, y = rounded[offset]
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    peaks[start+offset, -1] = point_at_pixel[y, x]
    return {'appearance_cost': costs, 'peak_indices': peaks,
            'query_indices': np.asarray(query_indices, dtype=np.int64),
            'supported_target': plan['supported'],
            'allowed_feature_cells': int(allowed.sum())}
