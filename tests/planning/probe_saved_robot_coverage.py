"""Bounded CPU post-hoc volume queries on original rejected arm/finger curves.

This diagnostic never restores a world, steps physics, invokes a planner, or
creates a GPU mapper. Original calibrated depth must reproduce the archived
free mask exactly. The existing saved-trajectory adapter reloads the original
arm Hermite polynomial and FingerTrajectory without smoothing or retiming.
At most ten state queries, including the measured baseline, are permitted.
"""

# BEGIN independent saved-curve coverage diagnostic; no planner acceptance write.
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.execution import ExecutionLimits
from manipulation.planning_observation import _known_free_voxels, _views
from recorded_simulation import load_curve_archive
from robot_coverage_fixture import build_robot_coverage


def _read_json(path):
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Diagnostic metadata exceeds its 16 MiB input bound")
    return json.loads(path.read_text())


def run(args):
    started = time.monotonic()
    times = np.asarray(args.times, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] != 0. or np.any(np.diff(times) <= 0)
            or not 1 <= len(args.candidate) <= 2
            or 1 + len(args.candidate) * len(times) > 10):
        raise ValueError("Use ordered times beginning at zero and at most ten total native state queries")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "preparing", "physics_run": False, "gpu_used": False,
        "fixture": str(args.fixture.resolve()), "feedback": str(args.feedback.resolve()),
        "observed_free": str(args.observed_free.resolve()), "views": args.view,
        "depth_range_m": args.depth_range, "depth_error_m": args.depth_error,
        "initial_state_assumption_source": args.initial_state_assumption_source,
        "requested_times_s": times.tolist(), "maximum_state_queries": 10,
        "maximum_native_working_bytes": 2 * 1024**3,
        "unknown_volume_scope": ("First failing native component only, not total robot unknown volume; "
                                 "native joint-limit rejection precedes any solid subtraction"),
        "scope": ("Selected original nominal arm/finger states only; native-model floating-point "
                  "whole-volume queries, not full-curve validation, continuous sweep proof, "
                  "actual tracking certification, hidden-environment evidence or task execution"),
        "candidates": [],
    }
    oracle = None
    try:
        fixture, feedback = _read_json(args.fixture), _read_json(args.feedback)
        raw = {}
        for identity, depth_path, calibration_path in args.view:
            if identity in raw:
                raise ValueError("Each raw calibrated view requires a unique observation identity")
            calibration = _read_json(Path(calibration_path))
            raw[identity] = {
                "depth": np.load(depth_path, allow_pickle=False), "K": calibration["K"],
                "T_output_camera": calibration["T_world_camera_cv"],
                "reference_frame": feedback["reference_frame"],
            }
        views = _views(raw, feedback["reference_frame"], args.depth_range)
        with np.load(args.observed_free, allow_pickle=False) as archived:
            expected_free = archived["known_free"].copy()
            origin = archived["origin"].copy()
            voxel_size = float(archived["voxel_size"])
            observation_ids = tuple(archived["observation_ids"].tolist())
        if (expected_free.dtype != np.bool_ or expected_free.ndim != 3
                or expected_free.size > 8_000_000 or set(observation_ids) != set(raw)):
            raise ValueError("Archived raw free cells require the exact bounded original view set")
        known = _known_free_voxels(views, origin=origin, shape=expected_free.shape,
                                  voxel_size=voxel_size, depth_error_m=args.depth_error)
        if not np.array_equal(known, expected_free):
            raise ValueError("Original calibrated depth does not exactly reproduce the archived free mask")
        coverage = SimpleNamespace(known_free=known, origin=origin, voxel_size=voxel_size,
                                   reference_frame=feedback["reference_frame"])
        oracle = build_robot_coverage(fixture, feedback, coverage,
            initialization_source=args.initial_state_assumption_source,
            max_queries=10, max_working_bytes=2 * 1024**3)
        report.update(raw_free_reproduced_exactly=True, known_free_cells=int(known.sum()),
                      grid_cells=int(known.size), prepare_seconds=time.monotonic() - started)
        # This separate measured query retains the authorized valid-initial-
        # model assumption. Archived curve samples below never use entrance
        # canonicalization, including their raw t=0 float32 arm coordinates.
        report["measured_baseline"] = oracle.check_state(
            joint_names=feedback["joint_names"], joint_positions=feedback["start_q"], entrance=True)
        baseline_names = tuple(feedback["joint_names"])
        baseline_q = np.asarray(feedback["start_q"], float)
        for path in args.candidate:
            metadata = _read_json(path.with_suffix(".json"))
            if metadata["classification"] != "rejected_diagnostic_not_executable":
                raise ValueError("This post-hoc probe requires an explicitly rejected diagnostic candidate")
            limits = ExecutionLimits.from_mapping(metadata["execution_limits"])
            trajectory, _ = load_curve_archive(path)
            fingers = trajectory.finger_trajectory
            if fingers is None or np.any(times > trajectory.executable_until):
                raise ValueError("Queries require saved fingers and must remain before the original evidence barrier")
            names = tuple(trajectory.joint_names) + tuple(fingers.joint_names)
            if len(set(names)) != len(names) or set(names) != set(baseline_names):
                raise ValueError("Original arm and finger polynomials must name every actual baseline coordinate once")
            # BEGIN keep coverage on the same four-part trajectory sample.
            arm_q, arm_dq, _, progress = trajectory.sample(times)
            # END acceleration is available but is not a coverage predicate.
            finger_q, finger_dq = fingers.sample(times)
            positions = np.concatenate((arm_q, finger_q), axis=1)
            velocities = np.concatenate((arm_dq, finger_dq), axis=1)
            current_baseline = baseline_q[[baseline_names.index(name) for name in names]]
            row = {
                "path": str(path.resolve()), "classification": metadata["classification"],
                "duration_s": trajectory.duration, "executable_until_s": trajectory.executable_until,
                "arm_polynomial": "Original archived Hermite times/position/velocity; unchanged",
                "finger_polynomial": "Original validated saved FingerTrajectory; unchanged nominal command reference",
                "joint_names": list(names), "finger_source": fingers.source,
                "finger_feedback_id": fingers.feedback_id, "queries": [],
            }
            report["candidates"].append(row)
            for index, moment in enumerate(times):
                before = time.monotonic()
                result = oracle.check_state(joint_names=names, joint_positions=positions[index])
                row["queries"].append({
                    "time_s": float(moment), "progress": float(progress[index]),
                    "joint_positions": positions[index].tolist(),
                    "joint_velocities": velocities[index].tolist(),
                    "delta_from_measured_joint_positions": (positions[index] - current_baseline).tolist(),
                    "entrance_canonicalization": False, "result": result,
                    "elapsed_seconds": time.monotonic() - before,
                })
                print(json.dumps({"candidate": path.stem, "time_s": float(moment),
                    "passed": result["passed"], "failure": result.get("failure"),
                    # BEGIN preserve the oracle's actual F-deficit semantics.
                    "free_evidence_deficit_volume_m3": result.get("free_evidence_deficit_volume_m3"),
                    "residual_classification": result["residual_classification"],
                    # END no occupied-space classification is inferred here.
                    "witness": result.get("witness")}, allow_nan=False), flush=True)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        (args.output / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        import torch
        if oracle is not None:
            report["coverage"] = oracle.describe()
        report.update(elapsed_seconds=time.monotonic() - started,
                      maximum_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      torch_cuda_initialized=torch.cuda.is_initialized())
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--observed-free", type=Path, required=True)
    parser.add_argument("--view", nargs=3, action="append", required=True,
                        metavar=("ID", "DEPTH", "CALIBRATION"))
    parser.add_argument("--depth-range", nargs=2, type=float, required=True)
    parser.add_argument("--depth-error", type=float, required=True)
    parser.add_argument("--initial-state-assumption-source", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--times", nargs="+", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
# END diagnostic reports do not alter candidate classification or coverage rules.
