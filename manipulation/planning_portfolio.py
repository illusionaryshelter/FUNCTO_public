"""Plan all grounded references independently, then score completed curves.

The parent persists CPU input arrays and continuous path expressions, starts
independent Python workers, and waits for every worker to exit. Each worker
owns one task problem, observed scene, official robot data and direct backend.
The parent compares original accepted results under one fixed calibration;
the selected archive is delivered directly with its original authorization.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np


def _write(path, value):
    """Persist a complete finite lifecycle document before the next stage."""
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+"\n")
    temporary.replace(path)


def save_inputs(directory, value):
    """Save CPU arrays and declared path operands without native objects.

    NumPy files preserve dtype and binary values. Continuous PosePath objects
    declare their own interpolation/composition operands. The numeric array
    references name immutable inputs; each worker loads its own mutable model
    and cache state after process startup.
    """
    from .planning_problem import PosePath
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    count = 0

    def encode(item):
        nonlocal count
        if isinstance(item, np.ndarray):
            path = directory/f"array-{count:06d}.npy"
            count += 1
            np.save(path, item, allow_pickle=False)
            return {"array_file": str(path.resolve())}
        if isinstance(item, PosePath):
            return encode(item.to_record())
        if is_dataclass(item):
            return {field.name: encode(getattr(item, field.name))
                    for field in fields(item) if not field.name.startswith("_")}
        if isinstance(item, dict):
            return {str(key): encode(row) for key, row in item.items()}
        if isinstance(item, (tuple, list)):
            return [encode(row) for row in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        return item

    manifest = directory/"inputs.json"
    _write(manifest, encode(value))
    return manifest


def load_inputs(path):
    """Load this worker's exact arrays and restore complete path operations."""
    from .planning_problem import PosePath

    def decode(item):
        if isinstance(item, dict):
            if set(item) == {"array_file"}:
                return np.load(item["array_file"], allow_pickle=False)
            return {key: decode(value) for key, value in item.items()}
        if isinstance(item, list):
            return [decode(value) for value in item]
        return item

    value = decode(json.loads(Path(path).read_text()))
    reference = value["hypothesis"]["reference"]
    for key in tuple(reference):
        if key.endswith("_path") and isinstance(reference[key], dict):
            reference[key] = PosePath.from_record(reference[key])
    return value


def _worker(input_path):
    """Record import, preparation, planning and metric failures separately."""
    folder = Path(input_path).parent.parent
    lifecycle = dict(status="running", stage="imports", accepted=False,
                     worker_started_unix_time_ns=time.time_ns(),
                     source_curve=None, source_report=str(folder/"planning/planning-report.json"))
    started = time.monotonic()
    _write(folder/"worker-result.json", lifecycle)
    try:
        from .planning_request import plan
        from .planning_records import load_curve_archive, _write_diagnostic_json
        from .trajectory_metrics import measure
        payload = load_inputs(input_path)
        hypothesis, request = payload["hypothesis"], payload["request"]
        lifecycle.update(id=hypothesis["id"], stage="preparation")
        _write(folder/"worker-result.json", lifecycle)
        result = plan(hypothesis, request)
        lifecycle.update(stage="metrics", accepted=bool(result.success),
                         planning_status="accepted" if result.success else "rejected")
        _write(folder/"worker-result.json", lifecycle)
        archive = folder/"planning"/("trajectory.npz" if result.success
                                     else "diagnostic-trajectory.npz")
        diagnostic = None
        if archive.exists():
            lifecycle["source_curve"] = str(archive)
            if not result.success:
                diagnostic, _ = load_curve_archive(archive)
        metrics, arrays = measure(result, hypothesis_id=hypothesis["id"],
            fixture=request["robot_fixture"], feedback=request["feedback"],
            source_report=lifecycle["source_report"],
            source_curve=lifecycle["source_curve"], curve=diagnostic)
        _write_diagnostic_json(folder/"metrics.json", metrics)
        if arrays:
            np.savez_compressed(folder/"metrics-arrays.npz", **arrays)
        lifecycle.update(status="completed", stage="complete",
                         metrics_status=metrics["metrics_status"], metrics_path=str(folder/"metrics.json"))
    except BaseException as error:
        lifecycle.update(status="failed", error=f"{type(error).__name__}: {error}",
                         traceback=traceback.format_exc())
        preparation_report = folder/"run-report.json"
        if lifecycle["stage"] == "preparation" and preparation_report.exists():
            current = json.loads(preparation_report.read_text())
            if current["planning_called"]:
                lifecycle["stage"] = "planning"
        # The plan function writes its current stage even when it raises.
        # Its completed original report, if present, remains authoritative.
        report_path = Path(lifecycle["source_report"])
        if report_path.exists():
            report = json.loads(report_path.read_text())
            lifecycle["accepted"] = bool(report["success"])
            lifecycle["planning_status"] = "accepted" if report["success"] else "rejected"
        raise
    finally:
        lifecycle.update(worker_seconds=time.monotonic()-started,
                         worker_finished_unix_time_ns=time.time_ns())
        _write(folder/"worker-result.json", lifecycle)


def _resource_sample(root_pid):
    """Measure the complete explicit process tree and whole CUDA device zero."""
    import psutil
    root = psutil.Process(root_pid)
    rss = root.memory_info().rss
    for child in root.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    device = int(query.stdout.splitlines()[0].strip())*1024**2
    return dict(unix_time_ns=time.time_ns(), host_rss_bytes=rss,
                device_used_bytes=device)


def _stop_workers(processes):
    """Stop live worker groups and reap them before returning to the caller."""
    live = [process for process in processes if process.poll() is None]
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # Exit between poll and signal is an ordinary process completion.
            pass
    deadline = time.monotonic()+5.
    for process in live:
        try:
            process.wait(timeout=max(0., deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def plan_grounding_hypotheses(grounding_result, planning_request):
    """Plan the complete hypothesis set and deliver its original selected curve.

    ``reference_hypotheses`` carry stable ids, F/E/B, complete references and
    evidence. ``planning_request`` supplies source task declarations and
    current chart/material bindings plus observation, feedback, robot, budget
    and fixed preference data. Processes share files read-only, never native
    objects or a backend. A numerical rejection is a completed planning result;
    startup, preparation and resource failures retain their actual stage.
    """
    from .resource_admission import MAX_HOST_RSS_BYTES, MAX_DEVICE_USED_BYTES
    from .trajectory_selection import select
    from .planning_records import _write_diagnostic_json, load_planning_result

    hypotheses = grounding_result["reference_hypotheses"]
    ids = [row["id"] for row in hypotheses]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("A planning portfolio requires distinct stable hypothesis ids")
    output = Path(planning_request["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    resources = planning_request["resources"]
    host_limit, device_limit = resources["max_host_rss_bytes"], resources["max_device_used_bytes"]
    if not 0 < host_limit <= MAX_HOST_RSS_BYTES or not 0 < device_limit <= MAX_DEVICE_USED_BYTES:
        raise ValueError("Portfolio limits must fit the declared whole host/device envelope")
    root_pid = planning_request.get("host_process_root_pid", os.getpid())
    from .resource_admission import configure_host_process_root
    configure_host_process_root(root_pid)
    start = time.monotonic()
    manifest = dict(status="preparing", workers=[], resource_samples=[],
                    parallel_started_unix_time_ns=time.time_ns(),
                    physical_status="missing_not_executed")
    processes, launched, logs = [], [], []
    try:
        for hypothesis in hypotheses:
            folder = output/hypothesis["id"]
            folder.mkdir()
            worker = dict(id=hypothesis["id"], output_directory=str(folder),
                          process_returncode=None)
            manifest["workers"].append(worker)
            request = deepcopy(planning_request)
            request.update(output_directory=str(folder), host_process_root_pid=root_pid)
            # Grounding's separately bound source-clock preview is a consumer
            # object. Workers receive the complete nominal path operands and
            # bind their own actual planning clock through the original backend.
            frozen = {key: hypothesis[key] for key in ("id", "reference", "T_world_function",
                "T_world_tool", "tool_reference_anchor_world", "source")}
            try:
                inputs = save_inputs(folder/"inputs", dict(hypothesis=frozen, request=request))
                worker["input_path"] = str(inputs)
            except Exception as error:
                _write(folder/"worker-result.json", dict(id=hypothesis["id"],
                    status="failed", stage="input_preparation", accepted=False,
                    error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                    source_curve=None, source_report=str(folder/"planning/planning-report.json")))
        # Launch every prepared request before waiting for any one request.
        env = dict(os.environ, PYTHONNOUSERSITE="1")
        project = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = project + os.pathsep + env.get("PYTHONPATH", "")
        for worker in manifest["workers"]:
            if "input_path" not in worker:
                continue
            log = (Path(worker["output_directory"])/"run.log").open("x")
            logs.append(log)
            command = [sys.executable, "-m", "manipulation.planning_portfolio", "--worker", worker["input_path"]]
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=project, env=env, start_new_session=True)
            except Exception as error:
                _write(Path(worker["output_directory"])/"worker-result.json", dict(
                    id=worker["id"], status="failed", stage="launch", accepted=False,
                    error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                    source_curve=None, source_report=str(Path(worker["output_directory"])/"planning/planning-report.json")))
                continue
            processes.append(process)
            launched.append((worker, process))
            worker.update(pid=process.pid, command=command,
                          launched_since_start_seconds=time.monotonic()-start)
        manifest["status"] = "running"
        _write(output/"parallel-run.json", manifest)
        next_sample = 0.
        while any(process.poll() is None for process in processes):
            if time.monotonic() >= next_sample:
                sample = _resource_sample(root_pid)
                manifest["resource_samples"].append(sample)
                if sample["host_rss_bytes"] > host_limit or sample["device_used_bytes"] > device_limit:
                    manifest.update(status="resource_limit", resource_limit_sample=sample)
                    _stop_workers(processes)
                    break
                next_sample = time.monotonic()+2.
            time.sleep(.1)
        for worker, process in launched:
            worker["process_returncode"] = process.wait()
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc())
        _stop_workers(processes)
        raise
    finally:
        for log in logs:
            log.close()
        manifest.update(parallel_seconds=time.monotonic()-start,
                        parallel_finished_unix_time_ns=time.time_ns())
        _write(output/"parallel-run.json", manifest)

    cases = []
    for worker in manifest["workers"]:
        folder = Path(worker["output_directory"])
        state_path = folder/"worker-result.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
        else:
            state = dict(status="failed", stage="preimport", accepted=False,
                failure="worker_exited_without_lifecycle_record", source_curve=None,
                source_report=str(folder/"planning/planning-report.json"))
        if state["status"] == "running":
            state.update(status="terminated", failure=manifest["status"]
                         if manifest["status"] == "resource_limit" else "worker_process_exit")
        worker["result"] = state
        metrics_path = folder/"metrics.json"
        if metrics_path.exists():
            from .evidence_archive import read_evidence_json
            case = read_evidence_json(metrics_path)
        else:
            case = dict(id=worker["id"], accepted=state["accepted"],
                metrics_status="missing_curve" if not state["accepted"] else "measurement_failed",
                source_curve=state.get("source_curve"), source_report=state["source_report"],
                status=state["status"], failure=state.get("error", state.get("failure")))
        cases.append(case)
    comparison = dict(status="completed", cases=cases,
        parallel_seconds=manifest["parallel_seconds"], physical_status="missing_not_executed",
        incomplete_worker_ids=[row["id"] for row in manifest["workers"]
            if row["result"]["status"] != "completed"])
    selection = select(comparison, planning_request["calibration"])
    selected_id = selection["preferred_id"]
    selected_result = None
    selected_curve = None
    if selected_id is not None:
        selected_result = load_planning_result(output/selected_id/"planning")
        selected_curve = str(output/selected_id/"planning/trajectory.npz")
    manifest["status"] = ("resource_limit" if manifest["status"] == "resource_limit" else
        "worker_failed" if comparison["incomplete_worker_ids"] else "completed")
    _write_diagnostic_json(output/"comparison.json", comparison)
    _write_diagnostic_json(output/"selection.json", selection)
    _write(output/"parallel-run.json", manifest)
    return dict(selection=selection, comparison=comparison, workers=manifest["workers"],
                selected_id=selected_id, selected_curve=selected_curve,
                selected_result=selected_result, output_directory=str(output))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    _worker(parser.parse_args().worker)
