"""Replay archived native q/dq through the current online acceptance managers.

Use the ordinary recorded runner arguments plus --native-result. The runner
builds its unchanged observed scene, then this scoped boundary performs only
native constraint evaluation. No IK, trajectory optimization, independent
export or physical execution is called. Outputs remain diagnostic evidence.
"""

# BEGIN bounded replay reuses production preparation and original native arrays.
import argparse
import json
from pathlib import Path
import resource
import runpy
import signal
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--native-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-seconds", type=int, default=900)
    args, runner_arguments = parser.parse_known_args()
    if args.output.exists():
        raise ValueError("Replay requires a new output directory")
    from replay_native_clock import bounded_arrays
    from recorded_task_planning import _write_diagnostic_json
    saved = bounded_arrays(args.native_result)
    metadata = json.loads(args.native_result.with_suffix(".json").read_text())
    if metadata["batch_size"] != 1:
        raise ValueError("Recorded replay requires the planner's single-batch contract")
    started = time.monotonic()
    report = {"schema": "functo.native_constraint_replay.1", "status": "running",
        "classification": "diagnostic_not_executable", "native_result": str(args.native_result.resolve()),
        "optimizer_called": False, "physics_steps": 0, "labels": {}}
    def timeout(*_):
        raise TimeoutError("Native constraint replay exceeded its separate diagnostic budget")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.replay_seconds)

    def replay(problem, scene, robot, output, *, robot_validator, budget, provenance, execution_limits,
               continuation=None, rebind=None, request_ledger=None, backend=None, joint_reference=None):
        if any(value is not None for value in (continuation, rebind, request_ledger, joint_reference)):
            raise ValueError("Diagnostic replay cannot continue or reset a planning request")
        import torch
        from curobo.types import JointState
        from manipulation.planning import _CuroboPlanner
        planner = _CuroboPlanner(problem, scene, robot, budget, "cuda:0", robot_validator=robot_validator)
        try:
            report["initial_state"] = planner.initial_state_diagnostics()
            for label, state_name, rollout in (
                    ("rollout", "js_solution", planner.trajopt.metrics_rollout),
                    ("interpolated", "interpolated_trajectory",
                     planner.trajopt.additional_metrics_rollouts["interpolated_rollout"])):
                print(json.dumps({"stage": label, "state": "started"}), flush=True)
                beginning = time.monotonic()
                names = metadata["states"][state_name]["joint_names"]
                columns = [names.index(name) for name in planner.joint_names]
                # Basic slicing before the named column gather preserves [seed,H,J].
                arrays = [np.ascontiguousarray(saved[state_name+"."+field][0][..., columns])
                          for field in ("position", "velocity", "acceleration", "jerk")]
                state = JointState.from_numpy(list(planner.joint_names), *arrays, device_cfg=planner.device_cfg)
                state.dt = planner.device_cfg.to_device(saved[state_name+".dt"].reshape(-1))
                manager = rollout.metrics_constraint_manager
                if label == "interpolated":
                    manager.set_dense_counts(planner.device_cfg.to_device(saved["interpolated_last_tstep"]).long())
                with torch.no_grad():
                    augmented = rollout.transition_model.compute_augmented_state(state)
                    constraints = manager.compute_costs(augmented)
                native_state = SimpleNamespace(joint_names=list(planner.joint_names),
                    position=state.position[None], velocity=state.velocity[None])
                values = {name: value.detach().cpu().numpy().copy()
                          for name, value in zip(constraints.names, constraints.values)}
                np.savez_compressed(args.output/(label+"-constraints.npz"), **values)
                # Exercise the same production diagnostic extraction used before
                # export. Convergence is deliberately not recomputed: this replay
                # isolates constraints and retains the original success flag only.
                metrics = SimpleNamespace(costs_and_constraints=SimpleNamespace(constraints=constraints),
                                          convergence=SimpleNamespace(names=[], values=[]))
                replay_result = SimpleNamespace(batch_size=1, num_seeds=metadata["num_seeds"],
                    success=planner.device_cfg.to_device(saved["success"]).bool(),
                    metrics=metrics if label == "rollout" else None,
                    interpolated_metrics=metrics if label == "interpolated" else None,
                    js_solution=native_state, interpolated_trajectory=native_state)
                seeds = [{"seed_index": i, **planner.native_diagnostics(replay_result, seed_index=i)[label]}
                         for i in range(metadata["num_seeds"])]
                report["labels"][label] = {"seconds": time.monotonic()-beginning, "seeds": seeds,
                    "sample_count": arrays[0].shape[1], "convergence_replayed": False}
                _write_diagnostic_json(args.output/"replay-report.json", report)
                print(json.dumps({"stage": label, "state": "completed", "seconds": time.monotonic()-beginning,
                                  "constraints": seeds[0]["constraints"]}), flush=True)
            report["status"] = "completed"
            return SimpleNamespace(success=False, diagnostics={"initial_state": report["initial_state"]})
        finally:
            planner.close()

    runner = Path(__file__).with_name("run_recorded_skill.py")
    sys.argv = [str(runner), *runner_arguments, "--output", str(args.output)]
    try:
        with patch("recorded_task_planning.plan_and_record", new=replay):
            runpy.run_path(str(runner), run_name="__main__")
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if args.output.exists():
            _write_diagnostic_json(args.output/"replay-report.json", report)
            source = args.output/"run-report.json"
            if source.exists():
                preparation = json.loads(source.read_text())
                preparation.update(planning_called=False, native_constraint_replay_called=True,
                                   execution_tested=False)
                preparation.pop("planning_success", None)
                _write_diagnostic_json(args.output/"scene-preparation-report.json", preparation)
                source.unlink()


if __name__ == "__main__":
    main()
# END no rejected archived candidate acquires execution authority.
