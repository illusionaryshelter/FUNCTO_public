"""Checked FP64 CUDA Jacobi binding for dual-view image geometry.

Importing this module does not initialize CUDA or compile. prepare/load occurs
before inference; generated artifacts live beside these packaged sources.
Only the installed runtime libraries are linked, with serial compilation.
"""

from functools import lru_cache
import os
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "_build_jacobi"
LOAD_RECORD = {}


@lru_cache(maxsize=1)
def load_extension():
    """Build serially with minimal ATen/pybind headers and explicit libraries."""
    import torch
    from torch.utils.cpp_extension import load

    started = perf_counter()
    wheel = Path(torch.__file__).resolve().parent
    nvidia = wheel.parent / "nvidia"
    includes = [nvidia / name / "include" for name in
                ("cuda_runtime", "cuda_nvcc", "cublas", "cusparse", "cusolver")]
    # The installed runtime wheel's FP16 headers include nv/target from the
    # matching existing toolkit. Put it last; solver/runtime headers come from
    # the same wheel libraries linked below, never the system CUDA 13 default.
    includes.append(Path("/usr/local/cuda-12.8/include"))
    # BEGIN link the installed library that actually exports the solver handle.
    # readelf --dyn-syms confirms getCurrentCUDASolverDnHandle is GLOBAL DEFAULT
    # in libtorch_cuda_linalg.so, not libtorch_cuda.so in this Torch wheel.
    libraries = [wheel/"lib/libtorch_cuda.so", wheel/"lib/libtorch_cuda_linalg.so",
                 wheel/"lib/libc10_cuda.so",
                 nvidia/"cusolver/lib/libcusolver.so.11",
                 nvidia/"cuda_runtime/lib/libcudart.so.12"]
    # END preserve the existing borrowed Torch device/stream/handle contract.
    for path in (*includes, *libraries):
        if not path.exists():
            raise FileNotFoundError(f"Required installed build dependency is absent: {path}")
    BUILD.mkdir(parents=True, exist_ok=True)
    existed = sorted(path.name for path in BUILD.glob("*.so"))
    os.environ["MAX_JOBS"] = "1"
    # with_cuda=False deliberately selects the host compiler. Explicit ATen
    # CUDA libraries and CUDA headers supply the binding's only CUDA needs.
    module = load(name="functo_dual_view_jacobi", sources=[str(ROOT/"cusolver_jacobi.cpp")],
        extra_cflags=["-O2", "-g0", "-std=c++17"],
        extra_include_paths=[str(path) for path in includes],
        extra_ldflags=[*[str(path) for path in libraries],
                       "-Wl,-rpath," + ":".join(dict.fromkeys(str(path.parent) for path in libraries))],
        build_directory=str(BUILD), with_cuda=False, verbose=True)
    LOAD_RECORD.update(seconds=perf_counter()-started, build_directory=str(BUILD),
                       existing_shared_objects=existed, max_jobs=1, host_cpp_only=True,
                       linked_libraries=[str(path) for path in libraries],
                       include_directories=[str(path) for path in includes])
    return module


def eigh_with_info(matrix):
    """Return W,V,CPU info,lwork; every info entry was already checked in C++."""
    return load_extension().eigh3x3(matrix)


def workspace_lwork(matrix):
    """Query required double workspace elements without solving the matrices."""
    return load_extension().workspace_lwork(matrix)


def eigh(matrix, UPLO="L"):
    """Lower-triangle, forward-only eigendecomposition for dual-view geometry."""
    if UPLO != "L":
        raise ValueError("The dual-view Jacobi binding supports explicit lower triangle only")
    values, vectors, _, _ = eigh_with_info(matrix)
    return values, vectors
