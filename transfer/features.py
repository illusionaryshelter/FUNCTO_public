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
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Feature evidence already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    command = [sys.executable, "-m", "transfer.features", "worker",
               str(Path(source_image).resolve()), str(Path(target_image).resolve()),
               str(output.resolve())]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                   check=True, timeout=1800)


def _worker(source, target, output):
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
    worker_main([Path(source), Path(target)], Path(output))


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
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("output")
    args = parser.parse_args()
    _worker(args.source, args.target, args.output)


if __name__ == "__main__":
    main()
