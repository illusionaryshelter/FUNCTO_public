"""Run one local perception stage at a time, without persistent GPU models.

Separate processes reclaim model weights, CUDA allocations and model-specific
autocast state before FUNCTO's dense correspondence calculation starts. Files
are a local transport only: there is no model server or feature cache.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_worker(kind: str, images: list[Image.Image], options: dict) -> dict[str, np.ndarray]:
    """Return CPU arrays after the isolated model process has fully exited.

    The caller must finish previous GPU operations before calling this method.
    Reclaim unused allocator blocks left by FUNCTO's previous matching stage;
    live tensors are never offloaded or modified by this function.
    """
    if kind not in {"sam31", "sd_dino"}:
        raise ValueError(f"Unknown local model: {kind}")
    if not images or any(not isinstance(im, Image.Image) for im in images):
        raise TypeError("images must be a nonempty list of PIL images")
    torch = sys.modules.get("torch")
    gc.collect()
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix=f"functo-{kind}-") as directory:
        directory = Path(directory)
        paths = []
        for index, image in enumerate(images):
            path = directory / f"input-{index}.png"
            image.convert("RGB").save(path)
            paths.append(str(path))
        output = directory / "output.npz"
        request = directory / "request.json"
        request.write_text(json.dumps({"kind": kind, "images": paths,
                                       "output": str(output), "options": options}))
        env = os.environ.copy()
        env.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   MPLBACKEND="Agg", MPLCONFIGDIR=str(directory / "matplotlib"),
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        # A checked exit prevents missing/partial features becoming a successful
        # inference result. Model logging stays visible, without request secrets.
        subprocess.run([sys.executable, "-m", "local_models.runtime", str(request)],
                       cwd=PROJECT_ROOT, env=env, check=True, timeout=1800)
        with np.load(output, allow_pickle=False) as arrays:
            return {key: arrays[key] for key in arrays.files}


def _guard_memory(parent_pid: int) -> None:
    """Stop this worker before the 16 GiB host is exhausted.

    RSS includes the caller because both processes must fit simultaneously.
    The CUDA allocator has a separate 8 GiB bound below. This guard never kills
    another application and a stopped worker makes run_worker fail explicitly.
    """
    import psutil

    worker = psutil.Process()
    parent = psutil.Process(parent_pid)
    while True:
        try:
            rss = worker.memory_info().rss + parent.memory_info().rss
            available = psutil.virtual_memory().available
            if rss > 12 * 2**30 or available < 1.5 * 2**30:
                print(f"Local model stopped: combined RSS={rss / 2**30:.2f} GiB, "
                      f"available={available / 2**30:.2f} GiB", file=sys.stderr, flush=True)
                os._exit(75)
        except psutil.NoSuchProcess:
            os._exit(75)
        time.sleep(0.25)


def _block_network() -> None:
    """Fail on missing weights rather than allowing a library to download them."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Local perception is offline; prepare all model weights first")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Local perception is offline; prepare all model weights first")
        return original_connect_ex(sock, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex


def main() -> None:
    """Private command-line entry used only by run_worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    _block_network()
    threading.Thread(target=_guard_memory, args=(os.getppid(),), daemon=True).start()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Local SAM3.1 and SD-DINO require the recovery CUDA environment")
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8 * 2**30 / total_memory))
    started = time.monotonic()
    paths = [Path(path) for path in request["images"]]
    output = Path(request["output"])
    if request["kind"] == "sam31":
        if len(paths) != 1:
            raise ValueError("SAM3.1 expects one image")
        from .sam31 import worker_main
        worker_main(paths[0], output, request["options"]["labels"])
    elif request["kind"] == "sd_dino":
        if len(paths) != 2:
            raise ValueError("SD-DINO expects a pair of images")
        from .sd_dino import worker_main
        worker_main(paths, output)
    else:
        raise ValueError(f"Unknown worker: {request['kind']}")
    print(json.dumps({"local_model": request["kind"],
                      "seconds": round(time.monotonic() - started, 2),
                      "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}), flush=True)


if __name__ == "__main__":
    main()
