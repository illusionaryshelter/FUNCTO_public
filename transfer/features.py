"""MARCO evidence for the independent region-transfer experiment.

The pretrained model and native coordinate operations are reused unchanged.
Whole-entity masks are supplied inputs; this module never segments a scene or
reads target region labels. Only selected point descriptors are materialized.
The two matching rules are deliberately named separately: reciprocal raw-dot
descriptor anchors and native Gaussian-decoded region correspondences.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
# BEGIN shared response ranking uses the existing SciPy implementation.
from scipy.stats import rankdata
# END ranking dependency shared by single-view and dual-view evidence.

from .core import fps_indices


def write_json(path, value):
    """Write strict JSON, retaining NumPy scalars without nonstandard NaNs."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=convert) + "\n")


def extract_pair(source_image, target_image, output):
    """Run one isolated, offline MARCO worker and retain its two CPU maps.

    The experiment supervisor separately bounds total device use and process
    tree RSS. This worker also sets a 6 GiB Torch allocator cap, instead of
    changing the shared model runtime's broader limits.
    """
    # BEGIN shared isolated extraction for real single images and image pairs.
    _extract_images([source_image, target_image], output)


def extract_image(image, output):
    """Save one native FP32 source/target map without constructing a fake pair."""
    _extract_images([image], output)


def _extract_images(paths, output):
    """Run the selected image count with the existing bounded offline worker."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Feature evidence already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    command = [sys.executable, "-m", "transfer.features", "worker",
               *[str(Path(path).resolve()) for path in paths], str(output.resolve())]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                   check=True, timeout=1800)


def _worker(paths, output):
    """Use the cached official model; never download missing model assets."""
    from local_models.runtime import _block_network
    _block_network()
    import torch
    from local_models.marco import worker_main

    if not torch.cuda.is_available():
        raise RuntimeError("MARCO requires the existing recovery CUDA runtime")
    torch.set_num_threads(2)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.5, 6 * 2**30 / total))
    worker_main([Path(path) for path in paths], Path(output))
# END no third-party model or preprocessing implementation is changed.


def sample_descriptors(feature, pixels, image_size):
    """Sample raw descriptors with MARCO's source-coordinate convention.

    Image size is (width,height). The native model pads the rounded resize to
    840 square, so source sampling must use that padded size rather than the
    unpadded observation dimensions. No descriptor normalization is applied.
    """
    import torch
    from torch.nn import functional as F
    from local_models.marco import _geometry, resize_scales, RESOLUTION

    values = np.asarray(feature)
    if values.dtype != np.float32 or values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("Expected one FP32 [1,C,Hf,Wf] MARCO map")
    pixels = np.asarray(pixels, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError("Query pixels must be finite [N,2]")
    geometry = _geometry()
    points = torch.as_tensor(pixels * resize_scales(image_size), dtype=torch.float32)[None]
    scaled = geometry.scale_keypoints_to_featmap(
        points, (RESOLUTION, RESOLUTION), values.shape[-2:])
    grid = geometry.normalize_coordinates(scaled, values.shape[-2:]).unsqueeze(2)
    with torch.inference_mode():
        sampled = F.grid_sample(torch.from_numpy(values), grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
    descriptors = sampled.squeeze(3).transpose(1, 2)[0].numpy()
    if not np.isfinite(descriptors).all():
        raise ValueError("Non-finite MARCO descriptors")
    return descriptors


# BEGIN shared rank/interpolation primitives moved without numerical changes.
def centered_rank_maps(logits, allowed):
    """Return tie-aware centered empirical ranks on a fixed feature-grid domain.

    Forbidden cells are zero placeholders, always excluded by the interpolation
    mask. Constant allowed responses have exactly zero evidence, even when the
    constant raw dot product is positive or negative. Ranks are unchanged by a
    strictly increasing transformation of the allowed raw responses.
    """
    values, mask = np.asarray(logits), np.asarray(allowed)
    if (values.ndim != 3 or values.dtype.kind not in 'fiu'
            or mask.dtype != np.bool_ or mask.shape != values.shape[1:]):
        raise ValueError('Require numeric [Q,Hf,Wf] responses and a Boolean [Hf,Wf] domain')
    if not np.isfinite(values).all():
        raise ValueError('MARCO responses must be finite')
    output = np.zeros(values.shape, dtype=np.float64)
    count = int(mask.sum())
    if count:
        ranks = rankdata(values[:, mask], method='average', axis=1)
        output[:, mask] = 2.0 * ((ranks - 0.5) / count) - 1.0
    return output


def rank_sampling_plan(feature_xy, allowed):
    """Precompute allowed-neighbor bilinear weights without clamping predictions.

    A prediction outside the closed feature-grid rectangle has no support.
    Safe clipped indices below are used only to gather arrays; their weights
    remain zero outside that rectangle or outside the fixed allowed domain.
    No nearest allowed cell replaces a missing interpolation neighborhood.
    """
    points, mask = np.asarray(feature_xy, dtype=float), np.asarray(allowed)
    if points.ndim != 2 or points.shape[1] != 2 or mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Require feature [N,2] coordinates and a Boolean grid')
    height, width = mask.shape
    if min(height, width) < 2:
        raise ValueError('Feature grid must have at least two cells on each axis')
    finite = np.isfinite(points).all(axis=1)
    inside = finite & np.all(points >= 0, axis=1) & np.all(points <= [width-1, height-1], axis=1)
    safe = np.where(inside[:, None], points, 0.0)
    lower = np.floor(safe).astype(np.int64)
    fraction = safe - lower
    indices, weights = [], []
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x, y = lower[:, 0] + dx, lower[:, 1] + dy
        valid = inside & (x < width) & (y < height)
        x, y = np.minimum(x, width-1), np.minimum(y, height-1)
        weight = (fraction[:, 0] if dx else 1-fraction[:, 0])
        weight = weight * (fraction[:, 1] if dy else 1-fraction[:, 1])
        weight = weight * (valid & mask[y, x])
        indices.append(y*width+x)
        weights.append(weight)
    indices, weights = np.stack(indices, axis=1), np.stack(weights, axis=1)
    denominator = weights.sum(axis=1)
    supported = denominator > 0
    weights = np.divide(weights, denominator[:, None], out=np.zeros_like(weights),
                        where=supported[:, None])
    return {'indices': indices, 'weights': weights, 'supported': supported,
            'grid_shape': mask.shape}


def sample_rank_maps(ranks, plan):
    """Interpolate ranked evidence, retaining zero evidence for unknown points."""
    values = np.asarray(ranks, dtype=float)
    if values.ndim != 3 or values.shape[1:] != tuple(plan['grid_shape']):
        raise ValueError('Rank map shape differs from its interpolation plan')
    if not np.isfinite(values).all():
        raise ValueError('Rank evidence must be finite')
    flat = values.reshape(len(values), -1)
    # Accumulate one neighbor at a time instead of allocating [Q,N,4].
    output = np.zeros((len(values), len(plan['indices'])), dtype=np.float64)
    for corner in range(4):
        output += flat[:, plan['indices'][:, corner]] * plan['weights'][None, :, corner]
    return output
# END shared primitives; view fusion and candidate selection remain separate.


def reciprocal_anchors(source, target, feature_pair, source_count=450, target_count=1400):
    """Compute reciprocal descriptor anchors on the two observed entities.

    This is raw-dot nearest-neighbour matching over spatial samples, an
    experimental anchor rule. It is not the native MARCO Gaussian decoder.
    The exact sample indices and scores are returned for audit and reuse.
    """
    source_indices = fps_indices(source["xyz"], source_count)
    target_indices = fps_indices(target["xyz"], target_count)
    fs = sample_descriptors(feature_pair[0], source["pixels"][source_indices],
                            source["object_mask"].shape[::-1])
    ft = sample_descriptors(feature_pair[1], target["pixels"][target_indices],
                            target["object_mask"].shape[::-1])
    source_valid = np.linalg.norm(fs, axis=1) > 1e-12
    target_valid = np.linalg.norm(ft, axis=1) > 1e-12
    source_indices, fs = source_indices[source_valid], fs[source_valid]
    target_indices, ft = target_indices[target_valid], ft[target_valid]
    if not len(fs) or not len(ft):
        raise ValueError("No nonzero descriptors in the observed entities")
    logits = fs @ ft.T
    best_target = logits.argmax(axis=1)
    best_source = logits.argmax(axis=0)
    keep = np.flatnonzero(best_source[best_target] == np.arange(len(fs)))
    return {
        "src_anchor_idx": source_indices[keep],
        "tgt_anchor_idx": target_indices[best_target[keep]],
        "scores": logits[keep, best_target[keep]],
        "source_sample_idx": source_indices,
        "target_sample_idx": target_indices,
        "matching_rule": "reciprocal raw MARCO descriptor dot product on spatial samples",
    }


def native_region_pairs(source, target, source_region_mask, feature_pair, count=160):
    """Decode source-region queries natively and expose invalid observations.

    Each query keeps MARCO's unnormalized dot products, temperature, Gaussian
    decoder and output scaling. The supplied target entity/depth domain is
    the only mask. Rounding selects a depth pixel; there is no nearest-surface
    snapping, no replacement correspondence and no target patch information.
    """
    import torch
    from local_models.marco import allowed_grid, masked_decode, resize_scales

    mask = np.asarray(source_region_mask)
    if mask.dtype != np.bool_ or mask.shape != (len(source["xyz"]),):
        raise ValueError("Source region must be a point-aligned Boolean mask")
    region_indices = np.flatnonzero(mask)
    selected = fps_indices(source["xyz"][region_indices], count)
    source_indices = region_indices[selected]
    if not len(source_indices):
        raise ValueError("Source region has no visible points")
    descriptors = sample_descriptors(feature_pair[0], source["pixels"][source_indices],
                                     source["object_mask"].shape[::-1])
    target_mask = np.zeros_like(target["object_mask"], dtype=bool)
    pixels = np.asarray(target["pixels"], dtype=int)
    target_mask[pixels[:, 1], pixels[:, 0]] = True
    allowed = allowed_grid(target_mask, feature_pair[1].shape[-2:])
    if not allowed.any():
        raise ValueError("No native feature cell observes the target entity")
    target_map = torch.from_numpy(feature_pair[1])
    decoded, scores = [], []
    with torch.inference_mode():
        for start in range(0, len(descriptors), 32):
            queries = torch.from_numpy(descriptors[start:start + 32])[None]
            logits = torch.bmm(queries, target_map.flatten(2)).reshape(
                1, -1, *target_map.shape[-2:])
            decoded.append(masked_decode(logits, allowed)[0].numpy())
            scores.append(logits[0, :, allowed].max(dim=1).values.numpy())
    output_pixels = np.concatenate(decoded) / resize_scales(target_mask.shape[::-1])
    nearest = np.rint(output_pixels).astype(int)
    point_index = np.full(target_mask.shape, -1, dtype=np.int64)
    point_index[pixels[:, 1], pixels[:, 0]] = np.arange(len(pixels))
    in_image = ((nearest[:, 0] >= 0) & (nearest[:, 0] < target_mask.shape[1])
                & (nearest[:, 1] >= 0) & (nearest[:, 1] < target_mask.shape[0]))
    target_indices = np.full(len(nearest), -1, dtype=np.int64)
    target_indices[in_image] = point_index[nearest[in_image, 1], nearest[in_image, 0]]
    valid = target_indices >= 0
    # A depth pixel supplies one observed 3D point. Repeated decoded pixels
    # remain in the raw evidence; the fitting input uses its strongest pair.
    raw_scores = np.concatenate(scores)
    fitting = []
    for target_index in np.unique(target_indices[valid]):
        candidates = np.flatnonzero(valid & (target_indices == target_index))
        fitting.append(int(candidates[np.argmax(raw_scores[candidates])]))
    fitting = np.asarray(fitting, dtype=np.int64)
    return {
        "src_anchor_idx": source_indices[fitting],
        "tgt_anchor_idx": target_indices[fitting],
        "scores": raw_scores[fitting],
        "raw_source_idx": source_indices,
        "raw_target_idx": target_indices,
        "raw_target_pixels": output_pixels,
        "raw_valid": valid,
        "raw_scores": raw_scores,
        "matching_rule": "native MARCO masked Gaussian soft-argmax; rounded pixel for RGB-D",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["worker"])
    # BEGIN exactly one real image or one real pair enters the same worker.
    parser.add_argument("images", nargs="+")
    parser.add_argument("output")
    args = parser.parse_args()
    if len(args.images) not in (1, 2):
        parser.error("Provide one image or a pair of images")
    _worker(args.images, args.output)
    # END existing pair commands keep their original input/output ordering.


if __name__ == "__main__":
    main()
