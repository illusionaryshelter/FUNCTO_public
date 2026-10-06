"""Build the project's small Coal bridge using only installed recovery files.

Run explicitly with PYTHONNOUSERSITE=1 and the recovery interpreter. This writes
one extension beside its project-owned C++ source; it never installs a package,
downloads dependencies, edits an environment, or changes third-party sources.
"""

# BEGIN an explicit single-compiler build of the ordered Coal distance bridge.
import importlib.util
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import time

import numpy as np


def build():
    if not sys.flags.no_user_site:
        raise RuntimeError("Run with PYTHONNOUSERSITE=1 using the recovery interpreter")
    project = Path(__file__).resolve().parents[2]
    source = project / "manipulation" / "_coal_scene_batch.cpp"
    specification = importlib.util.find_spec("coal")
    if specification is None or specification.origin is None:
        raise RuntimeError("The existing recovery Coal installation is required; do not install a replacement")
    # Locate headers belonging to the same existing installation that Python
    # imports. No guessed alternate ABI or fallback library is permitted.
    prefixes = [parent for parent in Path(specification.origin).resolve().parents
                if (parent / "include" / "coal" / "distance.h").is_file()]
    if len(prefixes) != 1:
        raise RuntimeError("Could not uniquely locate the imported Coal installation's headers")
    prefix = prefixes[0]
    libraries = [prefix / "lib" / "libcoal.so",
                 prefix / "lib" / f"libboost_python{sys.version_info.major}{sys.version_info.minor}.so"]
    includes = [Path(sysconfig.get_path("include")), Path(np.get_include()), prefix / "include"]
    missing = [str(path) for path in [source, *libraries, *includes] if not path.exists()]
    if missing:
        raise RuntimeError(f"Existing build inputs are missing: {missing}; no dependencies were installed")
    # Coal declares Eigen3 as an external dependency, rather than promising to
    # bundle its headers in cmeel.prefix. Read the installed Eigen pkg-config
    # declaration; this command is read-only and cannot install a missing library.
    eigen_flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "eigen3"], text=True))
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("The current interpreter does not provide an extension-module suffix")
    output = source.with_name("_coal_scene_batch" + suffix)
    compiler = shlex.split(os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++")
    started = time.perf_counter()
    # Compile atomically in the project directory. Importers never see a partly
    # linked module, and an unsuccessful rebuild leaves a previous artifact intact.
    with tempfile.NamedTemporaryFile(prefix="_coal_scene_batch-build-", suffix=".tmp",
                                     dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    # BEGIN interval support certificates require real directed IEEE rounding.
    # Disable reassociation/FMA and tell the compiler that arithmetic observes
    # the saved rounding mode. Existing Coal itself remains the installed lib.
    # BEGIN standard threads parallelize independent raw-distance batch slots.
    command = [*compiler, "-std=c++17", "-O3", "-DNDEBUG", "-shared", "-fPIC", "-pthread",
               "-frounding-math", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra",
               *[f"-I{path}" for path in includes], *eigen_flags, str(source),
               *map(str, libraries), f"-Wl,-rpath,{prefix / 'lib'}", "-o", str(temporary_path)]
    # END no additional threading package or runtime is installed.
    # END building never substitutes an approximate epsilon for this contract.
    try:
        # One compiler process: no build pool, package manager, or parallel jobs.
        subprocess.run(command, check=True)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"status": "built", "output": str(output),
            "compile_seconds": time.perf_counter() - started,
            "compiler_peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.,
            "command": command}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
# END importing production code does not secretly compile or select another algorithm.
