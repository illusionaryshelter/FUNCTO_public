"""Offline MARCO image features and its native masked correspondence rule.

The backbone runs only in the existing isolated model worker. CPU feature maps
can be reused for several query windows after that worker releases its GPU.
Matching preserves MARCO's resize rounding, source sampling and output scaling;
it does not reinterpret these descriptors as SD-DINO cosine features.
"""

from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2] / "MARCO"
RESOLUTION = 840
TEMPERATURE = 10.0
SIGMA = 7
FEATURE_SHAPE = (1, 1024, 240, 240)
QUERY_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def _geometry():
    """Load native pure geometry without importing a second global `models`.

MARCO's repository uses top-level `models`/`util` package names. The backbone
owns those names only in its worker; this file-based import lets the caller use
the maintained native geometry functions without polluting other model imports.
    """
    spec = importlib.util.spec_from_file_location(
        "_functo_marco_geometry", REPOSITORY / "util/geometry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resize_scales(size):
    """Original xy pixels to MARCO's rounded longest-side resize, no offset."""
    dimensions = np.asarray(size, dtype=float)
    resized = np.rint(dimensions * RESOLUTION / dimensions.max())
    if np.any(resized < 1):
        raise ValueError("MARCO resize would produce an empty image dimension")
    return resized / dimensions


def allowed_grid(mask, shape):
    """Admit feature cells whose decoded original-image pixel is permitted.

The output grid uses native align-corner scaling; the input point sampler uses
MARCO's distinct feature-size/image-size scaling. Keeping both conventions is
necessary to reproduce the model. Mask lookup uses nearest pixel, ties to even,
matching the RGB-D backprojection caller.
    """
    height, width = shape
    scale = resize_scales((mask.shape[1], mask.shape[0]))
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.rint(np.stack((x * (RESOLUTION-1)/(width-1)/scale[0],
                              y * (RESOLUTION-1)/(height-1)/scale[1]), axis=-1)).astype(int)
    inside = ((pixels[..., 0] >= 0) & (pixels[..., 0] < mask.shape[1])
              & (pixels[..., 1] >= 0) & (pixels[..., 1] < mask.shape[0]))
    allowed = np.zeros(shape, dtype=bool)
    allowed[inside] = mask[pixels[..., 1][inside], pixels[..., 0][inside]]
    return allowed


def masked_decode(logits, allowed):
    """Apply the native Gaussian decoder, with zero mass outside the mask.

The Gaussian is centred at the permitted maximum. Masking again afterwards
prevents multiplication by the Gaussian from reviving forbidden cells. With
every cell allowed, this is MARCO's original decoder computation.
    """
    import torch

    geometry = _geometry()
    valid = torch.as_tensor(allowed, dtype=torch.bool, device=logits.device)[None, None]
    if not bool(valid.any()):
        raise ValueError("No MARCO feature cell is supported by the target mask")
    bounded = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weighted = geometry.apply_gaussian_kernel(bounded, sigma=SIGMA).masked_fill(~valid, -torch.inf)
    probabilities = geometry.softmax_with_temperature(weighted.flatten(2), TEMPERATURE, -1)
    height, width = logits.shape[-2:]
    grid = geometry.create_grid(height, width, device=logits.device).reshape(1, 1, -1, 2)
    coordinates = (probabilities[..., None] * grid).sum(dim=2)
    return geometry.scaling_coordinates(coordinates, (height, width), (RESOLUTION, RESOLUTION))


def extract_pair(image1, image2):
    """Return two FP32 CPU [1,1024,240,240] maps after the model worker exits."""
    from .runtime import run_worker

    arrays = run_worker(kind="marco", images=[image1, image2], options={})
    pair = (arrays["feature1"], arrays["feature2"])
    for feature in pair:
        if feature.shape != FEATURE_SHAPE or feature.dtype != np.float32 or not np.isfinite(feature).all():
            raise ValueError(f"MARCO requires finite FP32 {FEATURE_SHAPE} pair features")
    return pair


# BEGIN one-image features retain the native pair preprocessing and backbone.
def extract_image(image):
    """Return one real image's map for reuse with both current target views."""
    from .runtime import run_worker

    feature = run_worker(kind="marco", images=[image], options={})["feature"]
    if feature.shape != FEATURE_SHAPE or feature.dtype != np.float32 or not np.isfinite(feature).all():
        raise ValueError(f"MARCO requires finite FP32 {FEATURE_SHAPE} image features")
    return feature


@lru_cache(maxsize=1)
def _preprocessing():
    """Read the installed native image operations without a global util import.

    MARCO's public pair helper uses these same three per-image operations.
    Reusing them avoids duplicating an image or running a second backbone just
    to extract a source feature map. The third-party source is never modified.
    """
    spec = importlib.util.spec_from_file_location(
        "_functo_marco_preprocessing", REPOSITORY / "util/visualization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess_image(path, *, device="cuda"):
    """Return native normalized [1,1,3,S,S] input for one actual RGB image."""
    from PIL import Image

    native = _preprocessing()
    with Image.open(path) as image:
        resized = native._resize_longest_side(image.convert("RGB"), RESOLUTION)
    tensor, _ = native._pad_tensor_no_resize(native._NORM(resized))
    return tensor[None, None].to(device)
# END independent source extraction changes neither image pixels nor sampling.


def match_features(pair, pixels, source_size, target_mask):
    """Decode original-image queries using native MARCO dot-product logits.

Only query descriptors and at most 32 score maps are materialized per batch.
The supplied CPU feature arrays are shared with Torch, without stacking or
copying the two large maps. Subpixel predictions outside the original candidate
mask fail explicitly; neither snapping nor another matcher changes the result.
    """
    import torch
    from torch.nn import functional as F

    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError("Extractor must return a pair of CPU feature arrays")
    features = []
    for feature in pair:
        value = np.asarray(feature)
        if (value.ndim != 4 or value.shape[0] != 1 or value.shape[1] < 1
                or min(value.shape[-2:]) < 2 or value.dtype != np.float32
                or not np.isfinite(value).all()):
            raise ValueError("MARCO features must be finite FP32 [1,C,Hf,Wf] arrays with Hf,Wf >= 2")
        features.append(torch.from_numpy(value))
    first, second = features
    if first.shape[1] != second.shape[1]:
        raise ValueError("Source and target descriptors must have equal channel counts")
    allowed = allowed_grid(target_mask, second.shape[-2:])
    candidate_count = int(allowed.sum())
    if not candidate_count:
        raise ValueError("No MARCO feature cell is supported by the target mask")
    geometry = _geometry()
    source_scale = resize_scales(source_size)
    target_scale = resize_scales((target_mask.shape[1], target_mask.shape[0]))
    outputs, native_outputs, scores, runners = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(pixels), QUERY_BATCH_SIZE):
            queries = torch.as_tensor(pixels[start:start+QUERY_BATCH_SIZE] * source_scale,
                                      dtype=torch.float32)[None]
            # Same operations as MARCO.sample_descriptors: retaining separate
            # maps avoids a 450 MiB stack solely to invoke the pair convenience API.
            feature_points = geometry.scale_keypoints_to_featmap(
                queries, (RESOLUTION, RESOLUTION), first.shape[-2:])
            grid = geometry.normalize_coordinates(feature_points, first.shape[-2:]).unsqueeze(2)
            descriptors = F.grid_sample(first, grid, mode="bilinear", padding_mode="border",
                                        align_corners=True).squeeze(3).transpose(1, 2)
            logits = torch.bmm(descriptors, second.flatten(2)).reshape(1, -1, *second.shape[-2:])
            if not torch.isfinite(logits).all():
                raise ValueError("MARCO descriptor dot products are non-finite")
            decoded = masked_decode(logits, allowed)
            native = geometry.kernel_softargmax_get_matches_logits(logits, TEMPERATURE, SIGMA)
            native = geometry.scaling_coordinates(native, second.shape[-2:], (RESOLUTION, RESOLUTION))
            outputs.append(decoded[0].numpy().astype(float) / target_scale)
            native_outputs.append(native[0].numpy().astype(float) / target_scale)
            supported = logits[0, :, allowed]
            best = torch.topk(supported, min(2, candidate_count), dim=1).values.numpy()
            scores.extend(best[:, 0].astype(float).tolist())
            runners.extend(best[:, 1].astype(float).tolist() if candidate_count > 1 else [None]*len(best))
    output = np.concatenate(outputs)
    nearest = np.rint(output).astype(int)
    inside = np.all((nearest >= 0) & (nearest < [target_mask.shape[1], target_mask.shape[0]]), axis=1)
    valid = np.zeros(len(output), bool)
    valid[inside] = target_mask[nearest[inside, 1], nearest[inside, 0]]
    if not valid.all():
        raise ValueError("MARCO soft-argmax lies outside the target mask; prediction is not snapped")
    margins = [a-b if b is not None else None for a, b in zip(scores, runners)]
    duplicates = [np.flatnonzero(np.all(nearest == pixel, axis=1)).tolist()
                  for pixel in np.unique(nearest, axis=0)
                  if np.count_nonzero(np.all(nearest == pixel, axis=1)) > 1]
    return output, {
        "point_count": len(pixels), "extractor_called": True, "matcher": "MARCO",
        "candidate_count": candidate_count, "masked_pixel_count": int(target_mask.sum()),
        "feature_shapes": [list(feature.shape) for feature in features],
        "source_batch_size": QUERY_BATCH_SIZE, "resolution": RESOLUTION,
        "source_scale_xy": source_scale.tolist(), "target_scale_xy": target_scale.tolist(),
        "mapping": "MARCO rounded resize; bottom/right padding; native source sampling and align-corner decode",
        "score_kind": "native MARCO descriptor dot product; not cosine or probability",
        "scores": scores, "second_scores": runners, "margins": margins,
        "numerical_ties": [m is not None and m <= 1e-6 for m in margins],
        "nonpositive_scores": [score <= 0 for score in scores],
        "duplicate_target_groups": duplicates, "native_unmasked_pixels": np.concatenate(native_outputs).tolist(),
    }


def worker_main(input_paths, output_path):
    """Extract the reviewed native MARCO profile with the existing offline hub."""
    import torch

    # BEGIN the same backbone accepts one offline source or the original pair.
    if len(input_paths) not in (1, 2):
        raise ValueError("MARCO requires one image or a pair of images")
    # END no source second view is manufactured.
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = torch.hub.load(str(REPOSITORY), "marco", source="local", device="cuda").eval()
    if float(model.softmax_temp) != TEMPERATURE:
        raise ValueError("MARCO model temperature differs from the declared decoder profile")
    # BEGIN preserve the original pair path, sharing its native per-image ops.
    if len(input_paths) == 1:
        samples = preprocess_image(input_paths[0])
    else:
        samples = model.preprocess_data(str(input_paths[0]), str(input_paths[1]), [[0., 0.]],
                                       inference_res=RESOLUTION, device="cuda")["samples"]
    # END source features are independent of which current view is later paired.
    with torch.inference_mode():
        features = model.forward_backbone(samples)
    if tuple(features.shape) != (1, len(input_paths), *FEATURE_SHAPE[1:]) or features.dtype != torch.float32:
        raise ValueError("MARCO backbone returned an unexpected feature profile")
    arrays = features.detach().cpu().numpy()
    # BEGIN serialized single-image evidence has one unambiguous feature key.
    if len(input_paths) == 1:
        np.savez(output_path, feature=arrays[:, 0])
    else:
        np.savez(output_path, feature1=arrays[:, 0], feature2=arrays[:, 1])
    # END pair evidence remains an independently useful experiment output.
