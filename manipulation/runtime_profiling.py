"""Optional host runtime measurements using Python's maintained profiler.

The disabled path calls the supplied operation directly. The enabled path
records Python and native calls on this thread, without synchronizing CUDA,
changing simulation time, wrapping individual control commands, or retaining
per-step samples. Call statistics use memory per distinct function and call
edge, not per physics tick; the caller's resource guard remains authoritative.
Asynchronous GPU kernel durations are not measured here. Instrumentation adds
wall time and does not refund any existing planning or search time budget.
"""

from pathlib import Path
import json
import sys
import time


def _write_profile(profiler, output, elapsed, succeeded):
    """Keep the full pstats archive and bounded, sortable JSON summaries.

    Self time excludes nested calls and can be summed across functions or
    files. Cumulative time includes callees and must not be added across rows.
    Both include measurement overhead; neither is a separate unprofiled run.
    """
    import pstats

    output = Path(output)
    # BEGIN a failed rerun cannot overwrite a previous operation's profile.
    output.mkdir(parents=True, exist_ok=False)
    # END the caller must choose a fresh profile directory just like a fresh run.
    profiler.dump_stats(str(output/"runtime.pstats"))
    statistics = pstats.Stats(profiler)
    functions, modules = [], {}
    for (filename, line, name), (primitive, calls, own, cumulative, _) in statistics.stats.items():
        row = {"file": filename, "line": line, "function": name,
               "primitive_calls": primitive, "calls": calls,
               "self_seconds": own, "cumulative_seconds": cumulative}
        functions.append(row)
        module = modules.setdefault(filename, {"file": filename, "calls": 0, "self_seconds": 0.})
        module["calls"] += calls
        module["self_seconds"] += own
    report = {
        "enabled": True, "call_succeeded": succeeded,
        "elapsed_wall_seconds": elapsed,
        "profiled_self_seconds": sum(row["self_seconds"] for row in functions),
        "total_calls": statistics.total_calls,
        "scope": "Current-thread host call durations including profiler overhead; asynchronous GPU kernels, child processes and other Python threads are not separately timed. Existing wall-time budgets include measurement overhead.",
        "self_time_definition": "Time inside the named function excluding callees; additive across functions.",
        "cumulative_time_definition": "Time in the named function and its callees; overlapping rows are not additive.",
        "full_statistics_path": "runtime.pstats",
        "summary_limit_per_list": 100,
        "functions_by_self_seconds": sorted(functions, key=lambda row: row["self_seconds"], reverse=True)[:100],
        "functions_by_cumulative_seconds": sorted(functions, key=lambda row: row["cumulative_seconds"], reverse=True)[:100],
        "files_by_self_seconds": sorted(modules.values(), key=lambda row: row["self_seconds"], reverse=True)[:100],
    }
    (output/"runtime.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


def run_with_optional_profile(operation, *args, enabled=False, output, **kwargs):
    """Call an operation unchanged, optionally recording its host call tree.

    Profiling is explicitly opt-in. It is disabled before exporting statistics
    so report generation is not misreported as application work. An operation's
    return value or exception is preserved even if profile export fails.
    The operation owns its output directory lifecycle; this helper creates its
    profile subdirectory only after the operation has returned or raised.
    """
    if not enabled:
        return operation(*args, **kwargs)
    import cProfile

    profiler = cProfile.Profile()
    succeeded = False
    started = time.perf_counter()
    profiler.enable()
    try:
        result = operation(*args, **kwargs)
        succeeded = True
        return result
    finally:
        profiler.disable()
        elapsed = time.perf_counter()-started
        try:
            _write_profile(profiler, output, elapsed, succeeded)
        except Exception as error:
            # Optional diagnostic storage must not mask a control failure or
            # turn a completed operation into a new physical execution error.
            # BEGIN even a closed diagnostic stream must preserve the result.
            try:
                print(f"Runtime profile export failed: {type(error).__name__}: {error}", file=sys.stderr)
            except Exception:
                pass
            # END profiler persistence and its warning are both non-authoritative.
