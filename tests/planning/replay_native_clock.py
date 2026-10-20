"""Replay saved native cubic controls through the production interpolation adapter.

This bounded diagnostic runs no IK, optimizer, FK, collision model or simulation.
It keeps the archived nonzero initial velocity and raw controls, calls the installed
CUDA interpolation kernel, and compares common physical times. Run serially with
other GPU probes. It does not turn a rejected candidate into an accepted plan.
"""

# BEGIN bounded original-control replay of the production fixed-clock adapter.
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def interpolation_method(torch, JointState, formal_clock):
    """Load the actual nested adapter without constructing the full GPU solver.

    The only omitted work is solver construction. The function body, low-level
    CUDA import, buffer handling and both dense-count notifications are unchanged.
    This diagnostic extraction is not a production solver replacement.
    """
    source = PROJECT_ROOT / "manipulation/planning.py"
    tree = ast.parse(source.read_text())
    owner = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.ClassDef) and node.name == "ProcessTrajectoryOptimizer")
    function = next(node for node in owner.body
                    if isinstance(node, ast.FunctionDef) and node.name == "get_interpolated_trajectory")
    scope = {"torch": torch, "JointState": JointState, "_formal_native_clock": formal_clock}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(source), "exec"), scope)
    return scope[function.name]


def bounded_arrays(path):
    """Reject large files before loading; all expected replay arrays are tiny."""
    path = Path(path)
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Interpolation replay requires an archive smaller than 64 MiB")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: array.copy() for name, array in archive.items()}
    if sum(value.nbytes for value in arrays.values()) > 64 * 1024**2:
        raise ValueError("Expanded interpolation archive exceeds its 64-MiB budget")
    return arrays


def run(arguments):
    destination = Path(arguments.output)
    array_path = destination.with_suffix(".npz")
    if destination.exists() or array_path.exists():
        raise ValueError("Replay output must use a new diagnostic path")
    saved = bounded_arrays(arguments.native_result)
    seed = bounded_arrays(arguments.optimizer_input)
    metadata = json.loads(Path(arguments.native_result).with_suffix(".json").read_text())
    effective = json.loads(Path(arguments.effective_config).read_text())
    names = saved["active_joint_names"].tolist()
    state_names = metadata["states"]["js_solution"]["joint_names"]
    columns = [state_names.index(name) for name in names]
    if (metadata["batch_size"] != 1 or metadata["num_seeds"] != 1
            or names != seed["joint_names"].tolist() or len(set(names)) != len(names)
            or not 1 <= len(names) <= 16):
        raise ValueError("Replay requires one saved seed with one complete active-joint order")
    controls = saved["solution"][0, 0]
    if controls.ndim != 2 or controls.shape[1] != len(names) or not 6 <= len(controls) <= 24:
        raise ValueError("Saved controls exceed the existing bounded planning dimensions")
    raw = {name: np.ascontiguousarray(saved["js_solution." + name][0, 0][:, columns])
           for name in ("position", "velocity", "acceleration", "jerk")}
    q0, dq0 = seed["start_q"], seed["start_dq"]
    if q0.shape != (1, len(names)) or dq0.shape != q0.shape or not np.any(dq0 != 0):
        raise ValueError("This boundary regression requires the original nonzero initial velocity")
    if not all(np.isfinite(value).all() for value in [controls, q0, dq0, *raw.values()]):
        raise ValueError("Replay requires finite original states and controls")
    config = effective["trajopt"]
    steps = config["core_cfg"]["metrics_rollout_config"]["transition_model_cfg"]["interpolation_steps"]
    horizon = (len(controls) + 4) * steps + 1
    if raw["position"].shape != (horizon, len(names)):
        raise ValueError("Archived raw state does not match its actual cubic horizon")
    capacity = config["interpolation_buffer_size"]
    if not horizon < capacity <= 2048:
        raise ValueError("Saved interpolation capacity exceeds the declared local bound")

    # Create only the CUDA context and small interpolation tensors. The one-GiB
    # allocator limit includes all Torch allocations made by this process.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This native interpolation diagnostic requires explicit GPU access")
    torch.set_num_threads(1)
    free, total = torch.cuda.mem_get_info()
    if free < 2 * 1024**3:
        raise RuntimeError("Native interpolation replay requires two GiB of free GPU memory")
    torch.cuda.set_per_process_memory_fraction(1024**3 / total)
    torch.cuda.reset_peak_memory_stats()
    from curobo.types import DeviceCfg, JointState
    from curobo._src.types.control_space import ControlSpace
    from curobo._src.util.trajectory import TrajInterpolationType
    from manipulation.planning import _formal_native_clock

    device = DeviceCfg("cuda:0")
    state = JointState.from_numpy(names, raw["position"][None], raw["velocity"][None],
                                  raw["acceleration"][None], raw["jerk"][None], device_cfg=device)
    state.knot = device.to_device(controls[None])
    state.dt = device.to_device(saved["js_solution.dt"].reshape(1))
    state.knot_dt = device.to_device(saved["js_solution.knot_dt"].reshape(1))
    current = JointState.from_numpy(names, q0, dq0, device_cfg=device)
    goal = JointState.from_position(device.to_device(controls[-1:]), joint_names=names)
    count_notifications = [[], []]
    managers = [SimpleNamespace(set_dense_counts=lambda counts, log=log: log.append(counts.clone()))
                for log in count_notifications]
    optimizer = SimpleNamespace(
        config=SimpleNamespace(minimum_trajectory_dt=config["minimum_trajectory_dt"],
            maximum_trajectory_dt=config["maximum_trajectory_dt"],
            interpolation_dt=config["interpolation_dt"], interpolation_buffer_size=capacity,
            interpolation_type=TrajInterpolationType.BSPLINE_KNOTS_CUDA, num_seeds=1),
        action_horizon=len(controls), action_dim=len(names), horizon=horizon,
        interpolation_steps=steps, transition_model=SimpleNamespace(control_space=ControlSpace.BSPLINE_3),
        device_cfg=device, joint_names=names, _interpolated_traj_buffer=None,
        _interpolation_dt_buffer=None,
        core=SimpleNamespace(_goal_buffer=SimpleNamespace(current_js=current, seed_goal_js=goal,
            idxs_current_js=torch.zeros(1, dtype=torch.int32, device=device.device),
            idxs_seed_goal_js=torch.zeros(1, dtype=torch.int32, device=device.device),
            # The production planner explicitly solves with use_implicit_goal=False.
            seed_enable_implicit_goal_js=torch.zeros(1, dtype=torch.uint8, device=device.device))),
        additional_metrics_rollouts={"interpolated_rollout": SimpleNamespace(
            constraint_manager=managers[0], metrics_constraint_manager=managers[1])})
    clock = _formal_native_clock(optimizer)
    interpolate = interpolation_method(torch, JointState, _formal_native_clock)
    with torch.no_grad():
        dense, counts, resized = interpolate(optimizer, state)
        torch.cuda.synchronize()
        count = int(counts.item())
        dense_values = {name: getattr(dense, name)[0, :count].cpu().numpy().copy() for name in raw}
        dense_dt = float(dense.dt.item())
        padding = dense.position[0, count:].cpu().numpy().copy()
        unchanged = np.array_equal(state.knot.cpu().numpy()[0], controls)
        # Calling the real adapter again must reuse its declared output buffers.
        original_buffer = dense.position.data_ptr()
        repeated, repeated_counts, repeated_resize = interpolate(optimizer, state)
        reused = repeated.position.data_ptr() == original_buffer
        # A changed raw knot interval must be rejected before launching the kernel.
        original_knot_dt = state.knot_dt.clone()
        state.knot_dt = torch.nextafter(state.knot_dt, torch.full_like(state.knot_dt, float("inf")))
        rejected_clock_drift = False
        try:
            interpolate(optimizer, state)
        except ValueError as error:
            rejected_clock_drift = "fixed request-wide spline clock" in str(error)
        finally:
            state.knot_dt = original_knot_dt
    if not (count == clock["valid_count"] and dense_dt == clock["interpolation_dt"]
            and not resized and not repeated_resize and reused and unchanged and rejected_clock_drift
            and int(repeated_counts.item()) == count
            and all(len(log) == 2 and all(int(item.item()) == count for item in log)
                    for log in count_notifications)):
        raise AssertionError("Production adapter changed the clock, buffers, controls or dense-count contract")
    stride = clock["subdivisions"] / steps
    if not stride.is_integer():
        raise ValueError("This common-sample diagnostic requires dense output to include raw sample times")
    stride = int(stride)
    common_errors = {name: float(np.max(np.abs(values[::stride] - raw[name])))
                     for name, values in dense_values.items()}
    for name, error in common_errors.items():
        tolerance = 128 * np.finfo(np.float32).eps * max(1., float(np.max(np.abs(raw[name]))))
        if error > tolerance:
            raise AssertionError(f"Native {name} changed at the same physical times: {error} > {tolerance}")
    raw_duration = (horizon - 1) * float(state.dt.item())
    if clock["duration"] != raw_duration:
        raise AssertionError("Dense output changed the original raw physical duration")

    times = np.arange(count, dtype=float) * dense_dt
    curve = CubicHermiteSpline(times, dense_values["position"], dense_values["velocity"], axis=0)
    coefficients = np.vstack((q0[0] + np.array([-1., 0., 1., 2.])[:, None]*dq0[0]*clock["knot_dt"],
                              controls, np.repeat(controls[-1:], 3, axis=0)))
    oracle = BSpline(np.arange(-3., len(coefficients) + 1.) * clock["knot_dt"], coefficients, 3)
    query = np.unique(np.r_[np.arange(horizon)*clock["native_dt"],
                            (times[:-1]+times[1:])/2., times[-1]])
    oracle_errors = {name: float(np.max(np.abs(curve(query, order)-oracle(query, order))))
                     for order, name in enumerate(("position", "velocity", "acceleration"))}

    # Persist only diagnostic arrays. No old fingers/events are compressed onto
    # a new clock, and no executable or accepted-plan classification is created.
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(array_path, joint_names=np.asarray(names), times=times,
             position=dense_values["position"], velocity=dense_values["velocity"],
             acceleration=dense_values["acceleration"], raw_position=raw["position"],
             raw_velocity=raw["velocity"], raw_acceleration=raw["acceleration"])
    with np.load(array_path, allow_pickle=False) as reloaded:
        loaded_curve = CubicHermiteSpline(reloaded["times"], reloaded["position"], reloaded["velocity"], axis=0)
        roundtrip_errors = {name: float(np.max(np.abs(curve(query, order)-loaded_curve(query, order))))
                            for order, name in enumerate(("position", "velocity", "acceleration"))}
    if any(roundtrip_errors.values()):
        raise AssertionError("Array roundtrip changed the diagnostic Hermite polynomial")
    report = {
        "classification": "native_clock_diagnostic_not_executable",
        "scope": "saved controls, actual production interpolation method and native CUDA kernel; no solve or scene feasibility check",
        "status": "passed", "source_success_unchanged": saved["success"].tolist(),
        "source_native_result": str(Path(arguments.native_result).resolve()),
        "source_optimizer_input": str(Path(arguments.optimizer_input).resolve()),
        "source_formal_clock": effective["formal_clock"], "corrected_clock": clock,
        "raw_duration": raw_duration, "raw_count": horizon, "dense_valid_count": count,
        "declared_capacity": capacity, "same_time_max_absolute_errors": common_errors,
        "same_time_bitwise_equal": {name: bool(np.array_equal(values[::stride], raw[name]))
                                    for name, values in dense_values.items()},
        "initial_q_max_absolute_error": float(np.max(np.abs(dense_values["position"][0]-q0[0]))),
        "initial_dq_max_absolute_error": float(np.max(np.abs(dense_values["velocity"][0]-dq0[0]))),
        "initial_velocity_nonzero": True, "controls_unchanged": unchanged,
        "buffer_reused": reused, "runtime_knot_dt_drift_rejected": rejected_clock_drift,
        "both_dense_managers_received_correct_counts": True,
        "padding_count": len(padding),
        "padding_q_max_absolute_error_from_final": (float(np.max(np.abs(padding-dense_values["position"][-1])))
                                                    if len(padding) else 0.),
        "float64_cubic_oracle_errors_including_midpoints": oracle_errors,
        "array_roundtrip_polynomial_errors": roundtrip_errors,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-input", required=True)
    parser.add_argument("--native-result", required=True)
    parser.add_argument("--effective-config", required=True)
    parser.add_argument("--output", required=True, help="New diagnostic JSON path; arrays use the same stem")
    run(parser.parse_args())
# END original saved success and all source artifacts remain unchanged.
