"""Native one-axis generation from a true net direction and constant reference.

The unchanged explicit contact fixture supplies geometry and actual entrance;
this probe only plans and never advances contact physics. A held reference has
no pose goal. Adding one declared net change must produce a 15 mm excursion.
"""

# BEGIN a real direction alone drives the existing native planner.
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from probe_contact_execution_gpu import build_native, build_problem, build_scene


def run(output):
    import torch
    from curobo.types import DeviceCfg
    from manipulation.execution_binding import plain
    from manipulation.motion import MotionModel
    from manipulation.planning import PlanningBudget, plan_trajectory
    from manipulation.planning_problem import CoordinateDirectionRequirement, MechanismRequirement, MotionPhase, PosePath

    if not torch.cuda.is_available():
        raise RuntimeError("Native direction generation probe requires accessible CUDA")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(min(2*1024**3/torch.cuda.get_device_properties(0).total_memory, 1.))
    torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True, exist_ok=False)
    # BEGIN preserve the exact source model returned by the shared native fixture.
    backend, validator, targets, _, robot = build_native(output)
    # END direction variants use the same robot solids and complete pair policy.
    initial = backend.proprioception()
    reference = PosePath(np.array([0., 1.]), np.stack((initial["tool_pose"], initial["tool_pose"])),
                         "explicit unchanged actual pose; no excursion reference")
    original = build_problem(backend)
    original = replace(original, phases=(MotionPhase("held reference", reference),),
                       pose_events=(), events=original.events[:1], conditions=())
    model = MotionModel("prismatic", np.zeros((1, 1)), initial["tool_pose"], np.eye(3), np.zeros(3),
                        1e-4, 1e-4, {"source": "native single arm slider geometry"})
    budget = PlanningBudget(ik_seeds=2, trajectory_seeds=1, knots=6, ik_iterations=80,
        trajectory_iterations=200, maximum_trajectory_dt=.1, maximum_trajectory_solves=2,
        search_seconds=0., max_robot_spheres=64, max_validation_points=256)
    rows = []
    for name, directions in (("hold", ()), ("required_negative_net", (
            CoordinateDirectionRequirement(0., 1., 0, "net", -1, .015, 1e-4,
                                           "explicit required downward 15 mm net change"),))):
        mechanism = MechanismRequirement(0., 1., model, np.eye(4), [[-.04, .04]], [0.],
                                         "independently declared physical one-axis fixture", directions)
        problem = replace(original, mechanisms=(mechanism,))
        scene = build_scene(DeviceCfg("cuda:0"), robot, backend, targets, problem)
        start = time.monotonic()
        # BEGIN every independent planning request supplies its source validator.
        result = plan_trajectory(problem, scene, robot, robot_validator=validator, budget=budget)
        # END an empty direction requirement does not bypass geometry validation.
        row = {"case": name, "success": result.success, "elapsed_s": time.monotonic()-start,
               "diagnostics": result.diagnostics, "request_ledger": result.request_ledger,
               "execution_binding": result.execution_binding}
        if result.trajectory is not None:
            curve = result.trajectory
            values = curve.sample(np.array([0., curve.duration]))[0]
            row.update(net_arm_change_m=float(values[-1, 0]-values[0, 0]),
                       residuals=curve.residuals)
        row["expected_behavior"] = ("hold unchanged actual entrance" if not directions
                                    else "negative net arm change at least .0149 m, with all original residuals zero")
        row["passed"] = bool(result.success and "net_arm_change_m" in row and
            (abs(row["net_arm_change_m"]) <= 1e-6 if not directions else row["net_arm_change_m"] <= -.0149))
        (output/f"{name}.json").write_text(json.dumps(plain(row), indent=2, allow_nan=False)+"\n")
        rows.append({key: row[key] for key in ("case", "success", "passed", "elapsed_s", "expected_behavior")})
        if "net_arm_change_m" in row:
            rows[-1]["net_arm_change_m"] = row["net_arm_change_m"]
        del result, scene, problem
        gc.collect()
        torch.cuda.empty_cache()
    report = {"scope": __doc__, "cases": rows, "budget": asdict(budget),
        "passed": all(row["passed"] for row in rows), "physical_execution_tested": False,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
# END reference, actual entrance, timing, budget and model stay the same in both cases.
