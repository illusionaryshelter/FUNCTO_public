"""Local SAM 3.1 text detection and segmentation for arbitrary image labels.

The public function exchanges only CPU arrays with an isolated worker. SAM's
CUDA context, BF16 autocast, and model weights therefore cannot leak into the
FP32 SD-DINO feature stage in the parent process.
"""

# BEGIN local SAM 3.1 integration: one native multiplex session per input image.
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def _validate_labels(labels: list[str]) -> list[str]:
    """Require explicit concepts and avoid repeating an identical text prompt."""
    if not isinstance(labels, list):
        raise TypeError("SAM 3.1 text labels must be a list of strings")
    if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("SAM 3.1 requires at least one nonempty text label")
    return list(dict.fromkeys(labels))


def detect(image: Image.Image, labels: list[str]) -> list[dict]:
    """Return all nonempty instances with pixel xyxy boxes and original-size masks.

    No semantic role or task-specific instance selection is performed here.
    Callers decide which instance is their tool, target, or other desired object.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("SAM 3.1 detection requires a PIL image")
    labels = _validate_labels(labels)
    from local_models.runtime import run_worker

    output = run_worker(kind="sam31", images=[image], options={"labels": labels})
    names = np.asarray(output["labels"])
    scores = np.asarray(output["scores"])
    boxes = np.asarray(output["boxes"])
    masks = np.asarray(output["masks"])
    count = len(names)
    if (
        names.shape != (count,)
        or scores.shape != (count,)
        or boxes.shape != (count, 4)
        or masks.shape != (count, image.height, image.width)
        or masks.dtype != np.bool_
    ):
        raise ValueError("SAM 3.1 worker returned inconsistent detection shapes or masks")
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError("SAM 3.1 worker returned non-finite scores or boxes")
    if any(str(name) not in labels for name in names):
        raise ValueError("SAM 3.1 worker returned an unrequested text label")
    # BEGIN empty-mask boundary: SAM's non-overlap constraint runs after its
    # initial empty-mask filtering and can erase an instance. Keep only occupied
    # masks, selecting every associated field with the same original index.
    valid_indices = np.flatnonzero(masks.any(axis=(1, 2)))
    return [
        {
            "label": str(names[index]),
            "score": float(scores[index]),
            "bbox": boxes[index].astype(float).tolist(),
            "mask": masks[index],
        }
        for index in valid_indices
    ]
    # END empty-mask boundary.


def worker_main(input_path: Path, output_path: Path, labels: list[str]) -> None:
    """Load local weights, run each concept on one image, and save safe NPZ arrays.

    NPZ fields are labels[str, N], scores[float32, N], boxes[float32, N, 4]
    in pixel xyxy coordinates, and masks[bool, N, H, W]. Even an empty result
    keeps those shapes, and loading never requires pickle.
    """
    labels = _validate_labels(labels)
    recovery_root = Path(__file__).resolve().parents[2]
    checkpoint = recovery_root / "models" / "sam3.1" / "sam3.1_multiplex.pt"
    source = recovery_root / "sam3"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local SAM 3.1 checkpoint is missing: {checkpoint}")
    if not (source / "sam3" / "model_builder.py").is_file():
        raise FileNotFoundError(f"Local SAM 3.1 source is missing: {source}")
    sys.path.insert(0, str(source))
    import torch
    from sam3.model_builder import build_sam3_predictor

    with Image.open(input_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    names, scores, boxes, masks = [], [], [], []
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    predictor = None
    session_id = None
    try:
        predictor = build_sam3_predictor(
            checkpoint_path=str(checkpoint),
            version="sam3.1",
            use_fa3=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
        # The upstream constructor enters a persistent autocast context. Pair
        # that entry immediately, then limit BF16 to the SAM call scope below.
        predictor.bf16_context.__exit__(None, None, None)
        # Keep the previously validated one-frame memory controls. These affect
        # batching only; the native 1008 input resolution and weights are intact.
        predictor.model.batched_grounding_batch_size = 1
        predictor.model.postprocess_batch_size = 1
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            session_id = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": [image],
                    "offload_video_to_cpu": True,
                }
            )["session_id"]
            for label in labels:
                # Native add_prompt resets semantic prompts and prior objects;
                # each label gets independent detections within this session.
                result = predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": label,
                    }
                )["outputs"]
                label_scores = np.asarray(result["out_probs"], dtype=np.float32)
                label_boxes = np.asarray(result["out_boxes_xywh"], dtype=np.float32).copy()
                label_masks = np.asarray(result["out_binary_masks"], dtype=np.bool_).copy()
                count = len(label_scores)
                if label_boxes.shape != (count, 4) or label_masks.shape != (count, height, width):
                    raise ValueError("SAM 3.1 returned inconsistent native output shapes")
                # Native boxes are normalized xywh. Convert once, keeping
                # fractional pixel coordinates for the downstream crop code.
                label_boxes[:, 2:] += label_boxes[:, :2]
                label_boxes *= np.asarray([width, height, width, height], dtype=np.float32)
                names.extend([label] * count)
                scores.append(label_scores)
                boxes.append(label_boxes)
                masks.append(label_masks)
        with Path(output_path).open("wb") as output_file:
            np.savez_compressed(
                output_file,
                labels=np.asarray(names, dtype=str),
                scores=np.concatenate(scores, axis=0),
                boxes=np.concatenate(boxes, axis=0),
                masks=np.concatenate(masks, axis=0),
            )
    finally:
        try:
            if predictor is not None and session_id is not None:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
        finally:
            # A worker process also isolates these flags, but restoring them
            # keeps this callable well behaved in direct integration checks.
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32

# END local SAM 3.1 integration.
