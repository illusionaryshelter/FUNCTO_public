"""Bounded local CoTracker3 inference for ordered RGB files and pixel queries.

Only CPU arrays cross the worker boundary. RGB decoding is bounded by one
window; cumulative point trajectories still grow linearly with frame count.
Point columns retain their query identity; visibility is a model estimate, not
a guarantee of rigid 3-D motion. The caller must validate depth and geometry.
"""

# BEGIN native local CoTracker3 adapter. Heavy imports remain inside the worker
# or the explicitly injected inference helper used by CPU contract tests.
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


WINDOW_LENGTH = 16
STEP = WINDOW_LENGTH // 2
MAX_QUERIES = 512


def _validate_inputs(frame_paths, queries_uv):
    """Preserve supplied frame order and require first-image pixel coordinates."""
    if isinstance(frame_paths, (str, bytes, Path)):
        raise TypeError("frame_paths must be an ordered sequence of image paths")
    paths = [Path(path) for path in frame_paths]
    if len(paths) < 2:
        raise ValueError("CoTracker requires at least two RGB frames")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local tracking frames: " + ", ".join(missing))
    with Image.open(paths[0]) as opened:
        width, height = opened.size
    if min(width, height) < 2:
        raise ValueError("CoTracker image width and height must each be at least two")
    queries = np.asarray(queries_uv, dtype=np.float64)
    if (queries.ndim != 2 or queries.shape[1] != 2
            or not 1 <= len(queries) <= MAX_QUERIES):
        raise ValueError(f"queries_uv must have shape [N,2], with 1 <= N <= {MAX_QUERIES}")
    if not np.isfinite(queries).all():
        raise ValueError("CoTracker queries must contain finite pixel coordinates")
    # Pixel zero and the last pixel centre are valid. No normalization, clipping
    # or integer rounding is applied before the native predictor rescales them.
    if ((queries < 0).any() or (queries[:, 0] > width - 1).any()
            or (queries[:, 1] > height - 1).any()):
        raise ValueError("CoTracker queries must lie inside the first RGB image")
    return paths, queries.astype(np.float32), (width, height)


def _validate_outputs(result, frame_count, query_count):
    """Reject truncated, non-finite or incorrectly typed worker predictions."""
    if not isinstance(result, dict) or not {"pixels", "visible"}.issubset(result):
        raise ValueError("CoTracker must return pixels and visible arrays")
    pixels = np.asarray(result["pixels"])
    visible = np.asarray(result["visible"])
    if pixels.shape != (frame_count, query_count, 2) or pixels.dtype != np.float32:
        raise ValueError("CoTracker pixels must be float32 with shape [T,N,2]")
    if visible.shape != (frame_count, query_count) or visible.dtype != np.bool_:
        raise ValueError("CoTracker visibility must be boolean with shape [T,N]")
    if not np.isfinite(pixels).all() or not np.isfinite(visible).all():
        raise ValueError("CoTracker returned non-finite pixels or visibility")
    # Out-of-image predictions are retained rather than projected onto an edge;
    # the depth sampler subsequently rejects those observations explicitly.
    return {"pixels": pixels, "visible": visible}


def track(frame_paths, queries_uv):
    """Return CPU pixels[T,N,2] and visible[T,N] for at most 512 initial queries.

    The RGB file sequence defines time order. Queries are original-resolution
    (x,y) pixel centres in frame zero, including fractional coordinates. Both
    the coarse and fixed-crop passes run every time. ``crop_xyxy`` reports the
    refinement crop with exclusive right/bottom edges; returned pixels always
    use the original image. The isolated worker owns CUDA until it exits; no
    model is loaded by importing this module or by validating its arguments.
    """
    paths, queries, image_size = _validate_inputs(frame_paths, queries_uv)
    from .runtime import run_file_worker

    output = run_file_worker(kind="cotracker", paths=paths,
                            options={"queries_uv": queries.tolist()})
    result = _validate_outputs(output, len(paths), len(queries))
    if "crop_xyxy" not in output:
        raise ValueError("CoTracker worker must report its fixed refinement crop")
    result["crop_xyxy"] = _validate_crop(output["crop_xyxy"], image_size)
    crop = result["crop_xyxy"]
    if ((queries < crop[:2]).any() or (queries > crop[2:] - 1).any()):
        raise ValueError("CoTracker worker crop does not contain every initial query")
    if (not np.array_equal(result["pixels"][0], queries)
            or not result["visible"][0].all()):
        raise ValueError("CoTracker worker did not preserve the initial query observations")
    return result


def _local_checkpoint():
    """Resolve the existing installed checkout; never invoke a hub downloader."""
    spec = importlib.util.find_spec("cotracker")
    if spec is None or spec.origin is None:
        raise ImportError("The local cotracker package is unavailable; no download is attempted")
    checkpoint = Path(spec.origin).resolve().parents[1] / "checkpoints" / "scaled_online.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local CoTracker3 checkpoint is missing: {checkpoint}")
    return checkpoint


def _validate_crop(crop_xyxy, image_size):
    """Require a bounded integer PIL crop, with exclusive right/bottom edges."""
    crop = np.asarray(crop_xyxy)
    width, height = image_size
    if (crop.shape != (4,) or crop.dtype.kind not in "iu" or (crop < 0).any()
            or crop[0] > width or crop[2] > width or crop[1] > height or crop[3] > height
            or crop[0] + 2 > crop[2] or crop[1] + 2 > crop[3]):
        raise ValueError("CoTracker crop_xyxy must be bounded integer edges enclosing at least 2x2 pixels")
    return crop.astype(np.int64)


def _refinement_crop(pixels, queries, image_size):
    """Envelope every coarse prediction and query, including occluded points.

    A fixed 25% margin of the largest pixel extent supplies context for the
    entire second pass. This is measured from this invocation, never from a
    released trajectory, previous result, per-frame mask, or task-specific ROI.
    """
    samples = np.concatenate((np.asarray(pixels, dtype=np.float64).reshape(-1, 2),
                              np.asarray(queries, dtype=np.float64)), axis=0)
    if not np.isfinite(samples).all():
        raise ValueError("Non-finite coarse predictions cannot define a refinement crop")
    low, high = samples.min(axis=0), samples.max(axis=0)
    margin = float(np.max(high - low)) * 0.25
    size = np.asarray(image_size, dtype=np.int64)
    # Clip before integer conversion: finite off-screen predictions can exceed
    # the integer range. Inclusive pixel centres become exclusive PIL edges.
    low = np.clip(np.floor(low - margin), 0, size).astype(np.int64)
    high = np.clip(np.ceil(high + margin + 1), 0, size).astype(np.int64)
    # A stationary single query can span one pixel. The native predictor uses
    # (width-1,height-1), so enlarge only that degenerate dimension to two pixels.
    for axis in range(2):
        if high[axis] - low[axis] < 2:
            low[axis] = min(low[axis], size[axis] - 2)
            high[axis] = low[axis] + 2
    return _validate_crop(np.concatenate((low, high)), image_size)


def _run_native(frame_paths, queries_uv, predictor, *, device, crop_xyxy=None):
    """Run native overlapping windows; an injected predictor permits CPU tests.

    The installed Point-Policy fork defaults to its one-frame extension. Passing
    one_frame=False is essential: a 16-frame window advances by eight frames,
    retains overlap, and revises previously published overlap predictions.
    """
    import torch

    paths, queries, image_size = _validate_inputs(frame_paths, queries_uv)
    crop = _validate_crop((0, 0, *image_size) if crop_xyxy is None else crop_xyxy,
                          image_size)
    offset = crop[:2].astype(np.float32)
    if ((queries < crop[:2]).any() or (queries > crop[2:] - 1).any()):
        raise ValueError("The fixed CoTracker crop must contain every initial query")
    if predictor.step != STEP:
        raise ValueError("CoTracker predictor must use the native 16-frame, step-8 schedule")

    def read_chunk(start, stop):
        # Only this window is decoded. Original 0..255 RGB values reach the
        # native predictor, which handles resizing the fixed crop internally.
        frames = []
        for path in paths[start:stop]:
            with Image.open(path) as opened:
                if opened.size != image_size:
                    raise ValueError("All CoTracker RGB frames must have the same dimensions")
                frames.append(np.array(opened.convert("RGB").crop(tuple(crop)), dtype=np.uint8))
        return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to(
            device=device, dtype=torch.float32)

    query_tensor = torch.from_numpy(np.column_stack((
        np.zeros(len(queries), dtype=np.float32), queries - offset)))[None].to(device)
    options = {"grid_size": 0, "add_support_grid": True, "one_frame": False}
    with torch.inference_mode():
        # Native initialization only stores queries and resets caches. One frame
        # suffices here; the first actual inference still receives frames 0:16.
        predictor(read_chunk(0, 1), is_first_step=True, queries=query_tensor, **options)
        start = 0
        while True:
            stop = min(start + WINDOW_LENGTH, len(paths))
            video = read_chunk(start, stop)
            pixels, visible = predictor(video, is_first_step=False, **options)
            del video
            # The native result includes all frames published so far, not just
            # the latest chunk. A mismatch cannot be repaired by slicing it.
            if (pixels is None or visible is None
                    or tuple(pixels.shape) != (1, stop, len(queries), 2)
                    or tuple(visible.shape) != (1, stop, len(queries))):
                raise ValueError(f"CoTracker returned an incomplete window ending at frame {stop}")
            if stop == len(paths):
                break
            start += STEP
        result = _validate_outputs({
            "pixels": pixels[0].detach().cpu().numpy().copy(),
            "visible": visible[0].detach().cpu().numpy().copy(),
        }, len(paths), len(queries))
    # Queries are supplied observations rather than network predictions. Restore
    # them explicitly, as the native sparse predictor does for queried frames.
    result["pixels"] += offset
    result["pixels"][0] = queries
    result["visible"][0] = True
    result["crop_xyxy"] = crop
    return result


def _run_refined(frame_paths, queries_uv, predictor, *, device):
    """Always run a coarse pass followed by one fixed-crop native pass.

    This is the inference definition, not a retry after geometry failure. Both
    passes share one loaded model and independent video caches. The second
    pass returns original-image coordinates and its crop for auditing.
    """
    paths, queries, image_size = _validate_inputs(frame_paths, queries_uv)
    coarse = _run_native(paths, queries, predictor, device=device)
    crop = _refinement_crop(coarse["pixels"], queries, image_size)
    del coarse
    # The model owns cumulative GPU outputs even after the local tensors go out
    # of scope. Explicitly drop them before reading the refinement window; the
    # native initialization below will also reset its query-specific features.
    predictor.model.init_video_online_processing()
    return _run_native(paths, queries, predictor, device=device, crop_xyxy=crop)


def worker_main(paths, output, queries_uv):
    """Save native online predictions using only the installed local checkpoint.

    The shared runtime blocks network access and bounds host/CUDA memory. Model
    resolution, support grid, six native refinement iterations and checkpoint
    precision are unchanged; this adapter only schedules bounded RGB windows.
    """
    paths, queries, _ = _validate_inputs(paths, queries_uv)
    checkpoint = _local_checkpoint()
    import torch
    from cotracker.predictor import CoTrackerOnlinePredictor

    if not torch.cuda.is_available():
        raise RuntimeError("Local CoTracker3 requires the configured recovery CUDA environment")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    predictor = CoTrackerOnlinePredictor(
        checkpoint=str(checkpoint), window_len=WINDOW_LENGTH).eval().to("cuda")
    result = _run_refined(paths, queries, predictor, device="cuda")
    np.savez_compressed(Path(output), **result)
# END native local CoTracker3 adapter.
