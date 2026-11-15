"""One isolated UNOPose forward for an explicitly ordered set of RGB-D pairs.

Each observation supplies RGB, metric depth, intrinsics and an observed owner
mask. The raw prediction maps reference-camera coordinates into query-camera
coordinates. Camera calibration and demonstrated tool relations are composed
by the grounding caller, independently of the learned model.
"""
from pathlib import Path
from time import perf_counter

import numpy as np


# BEGIN native batched relative-pose model with complete output-row identity.
def predict_batch(reference, queries, *, seed=1, checkpoint_path=None):
    """Return every native rotation, translation, score and prepared input.

    ``reference`` and each query use the native prepare_pair fields ``rgb``,
    ``depth``, ``intrinsics`` and ``mask``. Each pair retains its own original
    image size. Query order determines batch-row identity; no row is selected.
    """
    from .runtime import run_array_worker

    arrays = {"reference_" + key: np.asarray(value) for key, value in reference.items()}
    for index, query in enumerate(queries):
        arrays.update({f"query_{index}_" + key: np.asarray(value) for key, value in query.items()})
    options = dict(query_count=len(queries), seed=int(seed))
    if checkpoint_path is not None:
        options["checkpoint_path"] = str(Path(checkpoint_path).resolve())
    return run_array_worker("unopose", arrays, options)


def worker_main(arrays, output, *, query_count, seed, checkpoint_path=None):
    """Use the maintained native preparation and one full batched forward."""
    import sys
    import torch

    repository = Path(__file__).resolve().parents[2] / "UNOPose"
    sys.path.insert(0, str(repository))
    from core.unopose.inference import load_model, prepare_pair

    started = perf_counter()
    fields = ("rgb", "depth", "intrinsics", "mask")
    reference = {name: arrays["reference_" + name] for name in fields}
    prepared = [prepare_pair(reference,
        {name: arrays[f"query_{index}_" + name] for name in fields}, seed=seed)
        for index in range(query_count)]
    batch_cpu = {name: torch.cat([row[name] for row in prepared], dim=0)
                 for name in prepared[0]}
    prepare_seconds = perf_counter() - started
    torch.manual_seed(seed)
    phase = perf_counter()
    model = load_model(device="cuda") if checkpoint_path is None else load_model(
        checkpoint=Path(checkpoint_path), device="cuda")
    torch.cuda.synchronize()
    load_seconds = perf_counter() - phase
    phase = perf_counter()
    batch = {name: value.to("cuda") for name, value in batch_cpu.items()}
    torch.cuda.synchronize()
    upload_seconds = perf_counter() - phase
    phase = perf_counter()
    with torch.inference_mode():
        prediction = model(batch)
    torch.cuda.synchronize()
    forward_seconds = perf_counter() - phase
    np.savez(output,
        pred_R=prediction["pred_R"].detach().cpu().numpy(),
        pred_t=prediction["pred_t"].detach().cpu().numpy(),
        pred_pose_score=prediction["pred_pose_score"].detach().cpu().numpy(),
        batch_size=np.asarray(query_count), model_forward_calls=np.asarray(1),
        prepare_seconds=np.asarray(prepare_seconds), load_seconds=np.asarray(load_seconds),
        upload_seconds=np.asarray(upload_seconds), forward_seconds=np.asarray(forward_seconds),
        **{"model_input_" + name: value.numpy() for name, value in batch_cpu.items()})
# END native batched relative-pose model.
